"""Relation operators and semantic residual utilities for WinCoP."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

PATCH_GRID_SIZE = 14
PATCH_NODE_COUNT = 196

DINO_RELATION_K = 8
DINO_RELATION_TAU = 0.07
TRANSPORT_ALPHA = 0.75
SEMANTIC_LOGIT_SCALE = 50.0

LOVEDA_QUERY_GROUPS = (
    (0,), (1, 2, 3), (4,), (5,), (6,), (7,), (8,)
)

POTSDAM_QUERY_GROUPS = (
    (0, 1), (2,), (3,), (4,), (5,), (6, 7)
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


