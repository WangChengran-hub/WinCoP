from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .csct import (
    apply_csct_transport,
    build_semantic_laplacian,
)


PATCH_GRID_SIZE = 14
PATCH_NODE_COUNT = 196

DINO_RELATION_K = 8
DINO_RELATION_TAU = 0.07

LOW_MODE_COUNT = 66
TRANSPORT_ALPHA = 0.75
SEMANTIC_LOGIT_SCALE = 50.0

EPS = 1e-12


# ============================================================
# Dataset semantic-query mappings.
#
# These are NOT CSCT parameters.
# They only reproduce each frozen SegEarth dataset's
# synonym-query grouping.
# ============================================================

LOVEDA_QUERY_GROUPS: Tuple[Tuple[int, ...], ...] = (
    (0,),
    (1, 2, 3),
    (4,),
    (5,),
    (6,),
    (7,),
    (8,),
)

POTSDAM_QUERY_GROUPS: Tuple[Tuple[int, ...], ...] = (
    (0, 1),   # road, parking lot
    (2,),     # building
    (3,),     # low vegetation
    (4,),     # tree
    (5,),     # car
    (6, 7),   # clutter, background
)


@dataclass(frozen=True)
class RelationalTransportOutput:
    coarse_query_logits: Tensor

    relation_operator: Tensor

    dino_eigenvectors: Tensor
    low_eigenvectors: Tensor

    semantic_probabilities: Tensor
    semantic_laplacian: Tensor

    compatibility: Tensor
    spectral_weights: Tensor

    h1_residual: Tensor
    delta_41: Tensor
    csct_delta: Tensor

    coarse_residual: Tensor
    window_residual: Tensor

    refined_window_logits: Tensor


def _validate_matrix(
    tensor: Tensor,
    expected_rows: int,
    expected_columns: int,
    name: str,
) -> None:
    if tensor.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D, "
            f"got shape={tuple(tensor.shape)}"
        )

    if tensor.shape != (
        expected_rows,
        expected_columns,
    ):
        raise ValueError(
            f"{name} must have shape "
            f"{(expected_rows, expected_columns)}, "
            f"got {tuple(tensor.shape)}"
        )

    if not torch.isfinite(
        tensor
    ).all():
        raise ValueError(
            f"{name} contains NaN or Inf"
        )


def normalize_query_groups(
    query_groups: Sequence[Sequence[int]],
) -> Tuple[Tuple[int, ...], ...]:
    groups = tuple(
        tuple(
            int(q)
            for q in group
        )
        for group in query_groups
    )

    if not groups:
        raise ValueError(
            "query_groups cannot be empty"
        )

    if any(
        len(group) == 0
        for group in groups
    ):
        raise ValueError(
            "query_groups contains an empty class"
        )

    flat = [
        q
        for group in groups
        for q in group
    ]

    if min(flat) < 0:
        raise ValueError(
            "query indices must be nonnegative"
        )

    expected = list(
        range(
            max(flat) + 1
        )
    )

    if sorted(flat) != expected:
        raise ValueError(
            "query_groups must contain every query "
            "index exactly once from 0..Q-1; "
            f"got {flat}"
        )

    return groups


def query_count_from_groups(
    query_groups: Sequence[Sequence[int]],
) -> int:
    groups = normalize_query_groups(
        query_groups
    )

    return (
        max(
            q
            for group in groups
            for q in group
        )
        + 1
    )


def build_global_relation_mask(
    node_count: int = PATCH_NODE_COUNT,
    *,
    device: torch.device | None = None,
) -> Tensor:
    if node_count <= 1:
        raise ValueError(
            "node_count must be > 1"
        )

    mask = torch.ones(
        (
            node_count,
            node_count,
        ),
        dtype=torch.bool,
        device=device,
    )

    mask.fill_diagonal_(
        False
    )

    return mask


def build_dino_relation_operator(
    patch_tokens: Tensor,
    allowed_mask: Tensor | None = None,
) -> Tensor:
    if patch_tokens.ndim != 2:
        raise ValueError(
            "patch_tokens must be [N,C], "
            f"got {tuple(patch_tokens.shape)}"
        )

    if patch_tokens.shape[0] != PATCH_NODE_COUNT:
        raise ValueError(
            "Frozen GeoAlign-OV expects "
            f"{PATCH_NODE_COUNT} DINO nodes, "
            f"got {patch_tokens.shape[0]}"
        )

    if not torch.isfinite(
        patch_tokens
    ).all():
        raise ValueError(
            "patch_tokens contains NaN or Inf"
        )

    if allowed_mask is None:
        allowed_mask = build_global_relation_mask(
            PATCH_NODE_COUNT,
            device=patch_tokens.device,
        )

    if allowed_mask.shape != (
        PATCH_NODE_COUNT,
        PATCH_NODE_COUNT,
    ):
        raise ValueError(
            "allowed_mask shape mismatch: "
            f"{tuple(allowed_mask.shape)}"
        )

    if allowed_mask.dtype != torch.bool:
        raise TypeError(
            "allowed_mask must be bool"
        )

    if allowed_mask.device != patch_tokens.device:
        allowed_mask = allowed_mask.to(
            patch_tokens.device
        )

    tokens = F.normalize(
        patch_tokens,
        dim=-1,
    )

    similarity = (
        tokens
        @ tokens.T
    )

    similarity = similarity.masked_fill(
        ~allowed_mask,
        -float("inf"),
    )

    values, indices = torch.topk(
        similarity,
        k=DINO_RELATION_K,
        dim=-1,
    )

    if not torch.isfinite(
        values
    ).all():
        raise RuntimeError(
            "Non-finite top-k DINO relation similarity"
        )

    weights = torch.softmax(
        values
        / DINO_RELATION_TAU,
        dim=-1,
    )

    operator = torch.zeros(
        (
            PATCH_NODE_COUNT,
            PATCH_NODE_COUNT,
        ),
        dtype=torch.float32,
        device=patch_tokens.device,
    )

    operator.scatter_(
        1,
        indices,
        weights,
    )

    return operator


def dino_normalized_laplacian_eigenbasis(
    relation_operator: Tensor,
) -> Tensor:
    _validate_matrix(
        relation_operator,
        PATCH_NODE_COUNT,
        PATCH_NODE_COUNT,
        "relation_operator",
    )

    W = 0.5 * (
        relation_operator
        + relation_operator.T
    )

    W = W.clone()

    W.fill_diagonal_(
        0.0
    )

    degree = W.sum(
        dim=1
    ).clamp_min(
        1e-8
    )

    inv_sqrt_degree = torch.rsqrt(
        degree
    )

    normalized_W = (
        inv_sqrt_degree[:, None]
        * W
        * inv_sqrt_degree[None, :]
    )

    laplacian = (
        torch.eye(
            PATCH_NODE_COUNT,
            dtype=torch.float32,
            device=relation_operator.device,
        )
        - normalized_W
    )

    _, eigenvectors = torch.linalg.eigh(
        laplacian
    )

    return eigenvectors


def query_logits_to_class_probabilities(
    coarse_query_logits: Tensor,
    query_groups: Sequence[Sequence[int]] = (
        LOVEDA_QUERY_GROUPS
    ),
    logit_scale: float = SEMANTIC_LOGIT_SCALE,
) -> Tensor:
    """
    Convert SegEarth raw query logits into the semantic
    probability field used only for CLIP relation geometry.

    Frozen ordering:
        raw query logits
        -> * logit_scale
        -> query softmax
        -> same-class synonym max
        -> class re-normalization

    Supports arbitrary datasets / query groups.
    """

    groups = normalize_query_groups(
        query_groups
    )

    query_count = query_count_from_groups(
        groups
    )

    _validate_matrix(
        coarse_query_logits,
        PATCH_NODE_COUNT,
        query_count,
        "coarse_query_logits",
    )

    query_prob = (
        coarse_query_logits
        * float(
            logit_scale
        )
    ).softmax(
        dim=-1
    )

    class_prob = torch.empty(
        (
            PATCH_NODE_COUNT,
            len(groups),
        ),
        dtype=torch.float32,
        device=coarse_query_logits.device,
    )

    for class_index, group in enumerate(
        groups
    ):
        indices = torch.tensor(
            group,
            dtype=torch.long,
            device=coarse_query_logits.device,
        )

        class_prob[
            :,
            class_index,
        ] = (
            query_prob
            .index_select(
                1,
                indices,
            )
            .max(
                dim=1
            )
            .values
        )

    class_prob = (
        class_prob
        / class_prob.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(
            EPS
        )
    )

    return class_prob


def coarse_query_logits_from_window(
    window_logits: Tensor,
) -> Tensor:
    """
    [Q,224,224] -> [196,Q]
    """

    if window_logits.ndim != 3:
        raise ValueError(
            "window_logits must be [Q,H,W], "
            f"got {tuple(window_logits.shape)}"
        )

    if tuple(
        window_logits.shape[
            -2:
        ]
    ) != (
        224,
        224,
    ):
        raise ValueError(
            "Frozen transport expects 224x224 windows, "
            f"got {tuple(window_logits.shape[-2:])}"
        )

    if window_logits.shape[0] <= 0:
        raise ValueError(
            "window_logits contains no queries"
        )

    if not torch.isfinite(
        window_logits
    ).all():
        raise ValueError(
            "window_logits contains NaN or Inf"
        )

    query_count = int(
        window_logits.shape[0]
    )

    coarse = F.adaptive_avg_pool2d(
        window_logits.unsqueeze(0),
        output_size=(
            PATCH_GRID_SIZE,
            PATCH_GRID_SIZE,
        ),
    )[0]

    coarse = (
        coarse
        .permute(
            1,
            2,
            0,
        )
        .reshape(
            PATCH_NODE_COUNT,
            query_count,
        )
    )

    return coarse


def residual_to_window(
    coarse_residual: Tensor,
) -> Tensor:
    """
    [196,Q] -> [Q,224,224]
    """

    if coarse_residual.ndim != 2:
        raise ValueError(
            "coarse_residual must be [196,Q]"
        )

    if (
        coarse_residual.shape[0]
        != PATCH_NODE_COUNT
    ):
        raise ValueError(
            "coarse_residual node count mismatch"
        )

    query_count = int(
        coarse_residual.shape[1]
    )

    residual = (
        coarse_residual
        .reshape(
            PATCH_GRID_SIZE,
            PATCH_GRID_SIZE,
            query_count,
        )
        .permute(
            2,
            0,
            1,
        )
        .unsqueeze(0)
    )

    return F.interpolate(
        residual,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0]


def compute_relational_transport(
    window_logits: Tensor,
    patch_tokens: Tensor,
    allowed_mask: Tensor | None = None,
    query_groups: Sequence[Sequence[int]] = (
        LOVEDA_QUERY_GROUPS
    ),
    semantic_logit_scale: float = SEMANTIC_LOGIT_SCALE,
) -> RelationalTransportOutput:
    groups = normalize_query_groups(
        query_groups
    )

    query_count = query_count_from_groups(
        groups
    )

    if window_logits.shape != (
        query_count,
        224,
        224,
    ):
        raise ValueError(
            "window_logits/query_groups mismatch: "
            f"logits={tuple(window_logits.shape)}, "
            f"query_count={query_count}"
        )

    coarse_query_logits = (
        coarse_query_logits_from_window(
            window_logits
        )
    )

    relation_operator = (
        build_dino_relation_operator(
            patch_tokens,
            allowed_mask=allowed_mask,
        )
    )

    dino_eigenvectors = (
        dino_normalized_laplacian_eigenbasis(
            relation_operator
        )
    )

    low_eigenvectors = (
        dino_eigenvectors[
            :,
            :LOW_MODE_COUNT,
        ]
    )

    # Frozen DINO relational dynamics.
    state1 = (
        relation_operator
        @ coarse_query_logits
    )

    state4 = state1

    for _ in range(3):
        state4 = (
            relation_operator
            @ state4
        )

    h1_residual = (
        state1
        - coarse_query_logits
    )

    delta_41 = (
        state4
        - state1
    )

    # Dataset-specific CLIP semantic relation geometry.
    semantic_probabilities = (
        query_logits_to_class_probabilities(
            coarse_query_logits,
            query_groups=groups,
            logit_scale=semantic_logit_scale,
        )
    )

    semantic_laplacian = (
        build_semantic_laplacian(
            semantic_probabilities
        )
    )

    (
        csct_delta,
        compatibility,
        spectral_weights,
    ) = apply_csct_transport(
        delta_41,
        low_eigenvectors,
        semantic_laplacian,
    )

    coarse_residual = (
        h1_residual
        + csct_delta
    )

    window_residual = (
        residual_to_window(
            coarse_residual
        )
    )

    refined_window_logits = (
        window_logits
        + TRANSPORT_ALPHA
        * window_residual
    )

    return RelationalTransportOutput(
        coarse_query_logits=coarse_query_logits,
        relation_operator=relation_operator,
        dino_eigenvectors=dino_eigenvectors,
        low_eigenvectors=low_eigenvectors,
        semantic_probabilities=semantic_probabilities,
        semantic_laplacian=semantic_laplacian,
        compatibility=compatibility,
        spectral_weights=spectral_weights,
        h1_residual=h1_residual,
        delta_41=delta_41,
        csct_delta=csct_delta,
        coarse_residual=coarse_residual,
        window_residual=window_residual,
        refined_window_logits=refined_window_logits,
    )
