from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
    (0, 1),
    (2,),
    (3,),
    (4,),
    (5,),
    (6, 7),
)


@dataclass(frozen=True)
class SegEarthPostprocessOutput:
    prediction: Tensor
    confidence: Tensor
    class_probability: Tensor
    query_probability: Tensor


class FrozenSegEarthPostProcessor(nn.Module):
    def __init__(
        self,
        query_groups: Sequence[Sequence[int]],
        logit_scale: float = 50.0,
        probability_threshold: float = 0.3,
        background_index: int = 0,
    ) -> None:
        super().__init__()

        self.query_groups = tuple(
            tuple(
                int(q)
                for q in group
            )
            for group in query_groups
        )

        flat = [
            q
            for group in self.query_groups
            for q in group
        ]

        if not flat:
            raise ValueError(
                "query_groups contains no queries"
            )

        if min(flat) < 0:
            raise ValueError(
                "query indices must be nonnegative"
            )

        if sorted(flat) != list(
            range(
                max(flat) + 1
            )
        ):
            raise ValueError(
                "query_groups must contain every "
                "query exactly once"
            )

        self.query_count = (
            max(flat) + 1
        )

        self.class_count = len(
            self.query_groups
        )

        self.logit_scale = float(
            logit_scale
        )

        self.probability_threshold = float(
            probability_threshold
        )

        self.background_index = int(
            background_index
        )

        if not (
            0
            <= self.background_index
            < self.class_count
        ):
            raise ValueError(
                "Invalid background_index"
            )

    @torch.inference_mode()
    def forward(
        self,
        query_logits_canvas: Tensor,
        output_shape: tuple[int, int],
    ) -> SegEarthPostprocessOutput:
        if query_logits_canvas.ndim != 3:
            raise ValueError(
                "query_logits_canvas must be [Q,H,W]"
            )

        if (
            query_logits_canvas.shape[0]
            != self.query_count
        ):
            raise ValueError(
                f"Expected {self.query_count} queries, "
                f"got {query_logits_canvas.shape[0]}"
            )

        logits = F.interpolate(
            query_logits_canvas.unsqueeze(0),
            size=(
                int(output_shape[0]),
                int(output_shape[1]),
            ),
            mode="bilinear",
        )[0]

        query_probability = (
            logits
            * self.logit_scale
        ).softmax(
            dim=0
        )

        class_maps = []

        for group in self.query_groups:
            indices = torch.tensor(
                group,
                dtype=torch.long,
                device=query_probability.device,
            )

            class_maps.append(
                query_probability
                .index_select(
                    0,
                    indices,
                )
                .max(
                    dim=0
                )
                .values
            )

        class_probability = torch.stack(
            class_maps,
            dim=0,
        )

        confidence, prediction = (
            class_probability.max(
                dim=0
            )
        )

        prediction = prediction.clone()

        prediction[
            confidence
            < self.probability_threshold
        ] = self.background_index

        return SegEarthPostprocessOutput(
            prediction=prediction,
            confidence=confidence,
            class_probability=class_probability,
            query_probability=query_probability,
        )


def build_loveda_postprocessor() -> FrozenSegEarthPostProcessor:
    return FrozenSegEarthPostProcessor(
        query_groups=LOVEDA_QUERY_GROUPS,
        logit_scale=50.0,
        probability_threshold=0.3,
        background_index=0,
    )


def build_potsdam_postprocessor() -> FrozenSegEarthPostProcessor:
    return FrozenSegEarthPostProcessor(
        query_groups=POTSDAM_QUERY_GROUPS,
        logit_scale=50.0,
        probability_threshold=0.1,
        background_index=5,
    )
