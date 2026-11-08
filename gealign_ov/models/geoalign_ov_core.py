from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn

from gealign_ov.adapters.segearth_input import (
    SegEarthInputAdapter,
)
from gealign_ov.backbones.dinov3_timm import (
    DINOv3FeatureOutput,
    DINOv3TimmBackbone,
)
from gealign_ov.data.sliding_windows import (
    SlidingWindow,
    compute_sliding_windows,
)
from gealign_ov.methods.relational_transport import (
    LOVEDA_QUERY_GROUPS,
    SEMANTIC_LOGIT_SCALE,
    normalize_query_groups,
    query_count_from_groups,
)


SLIDE_CROP = 224
SLIDE_STRIDE = 112


@dataclass(frozen=True)
class GeoAlignOVCoreOutput:
    windows: Tuple[SlidingWindow, ...]

    base_window_logits: Tensor
    patch_tokens: Tensor
    refined_window_logits: Tensor

    base_logits_canvas: Tensor
    refined_logits_canvas: Tensor


def overlap_average_window_logits(
    window_logits: Tensor,
    windows: Tuple[SlidingWindow, ...],
    image_height: int,
    image_width: int,
) -> Tensor:
    if window_logits.ndim != 4:
        raise ValueError(
            "window_logits must be [W,Q,H,W], "
            f"got {tuple(window_logits.shape)}"
        )

    if (
        window_logits.shape[0]
        != len(windows)
    ):
        raise ValueError(
            "window count mismatch"
        )

    query_count = int(
        window_logits.shape[1]
    )

    if query_count <= 0:
        raise ValueError(
            "No query channels"
        )

    device = window_logits.device

    canvas = torch.zeros(
        (
            1,
            query_count,
            image_height,
            image_width,
        ),
        dtype=torch.float16,
        device=device,
    )

    count = torch.zeros(
        (
            1,
            1,
            image_height,
            image_width,
        ),
        dtype=torch.float16,
        device=device,
    )

    for i, window in enumerate(
        windows
    ):
        if (
            window.height
            != window_logits.shape[-2]
            or window.width
            != window_logits.shape[-1]
        ):
            raise ValueError(
                "Window/logit spatial mismatch"
            )

        y_slice, x_slice = (
            window.slices
        )

        canvas[
            :,
            :,
            y_slice,
            x_slice,
        ] += (
            window_logits[
                i
            ]
            .half()
            .unsqueeze(0)
        )

        count[
            :,
            :,
            y_slice,
            x_slice,
        ] += 1

    if (
        count <= 0
    ).any():
        raise RuntimeError(
            "Sliding-window reconstruction "
            "contains uncovered pixels"
        )

    return (
        canvas
        / count
    )[0]


class GeoAlignOVCore(nn.Module):
    def __init__(
        self,
        segearth_model: nn.Module,
        dino_backbone: DINOv3TimmBackbone,
        query_groups: Sequence[Sequence[int]] = (
            LOVEDA_QUERY_GROUPS
        ),
        semantic_logit_scale: float = (
            SEMANTIC_LOGIT_SCALE
        ),
    ) -> None:
        super().__init__()

        self.segearth_model = (
            segearth_model
        )

        self.dino_backbone = (
            dino_backbone
        )

        self.input_adapter = (
            SegEarthInputAdapter()
        )

        self.query_groups = (
            normalize_query_groups(
                query_groups
            )
        )

        self.query_count = (
            query_count_from_groups(
                self.query_groups
            )
        )

        self.semantic_logit_scale = float(
            semantic_logit_scale
        )

        slide_crop = int(
            getattr(
                self.segearth_model,
                "slide_crop",
                -1,
            )
        )

        slide_stride = int(
            getattr(
                self.segearth_model,
                "slide_stride",
                -1,
            )
        )

        if slide_crop != SLIDE_CROP:
            raise RuntimeError(
                "Expected slide_crop=224, "
                f"got {slide_crop}"
            )

        if slide_stride != SLIDE_STRIDE:
            raise RuntimeError(
                "Expected slide_stride=112, "
                f"got {slide_stride}"
            )

        model_query_count = int(
            self.segearth_model
            .query_idx
            .numel()
        )

        if (
            model_query_count
            != self.query_count
        ):
            raise RuntimeError(
                "SegEarth/query-group mismatch: "
                f"model={model_query_count}, "
                f"groups={self.query_count}"
            )

        self.train(False)

    def train(
        self,
        mode: bool = True,
    ) -> "GeoAlignOVCore":
        super().train(False)

        self.segearth_model.eval()
        self.dino_backbone.eval()
        self.input_adapter.eval()

        return self

    @staticmethod
    def _validate_input(
        normalized_rgb: Tensor,
    ) -> None:
        if normalized_rgb.ndim != 4:
            raise ValueError(
                "Expected BCHW input"
            )

        if normalized_rgb.shape[0] != 1:
            raise ValueError(
                "Expected batch size 1"
            )

        if normalized_rgb.shape[1] != 3:
            raise ValueError(
                "Expected RGB input"
            )

        if (
            normalized_rgb.dtype
            != torch.float32
        ):
            raise TypeError(
                "Input must remain FP32 until "
                "the DINO branch is extracted"
            )

        if not torch.isfinite(
            normalized_rgb
        ).all():
            raise ValueError(
                "Input contains NaN or Inf"
            )

    @torch.inference_mode()
    def forward(self, normalized_rgb: Tensor) -> GeoAlignOVCoreOutput:
        raise NotImplementedError(
            "Use GeoAlignAP4Core for WinCoP inference."
        )
