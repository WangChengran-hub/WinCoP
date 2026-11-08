"""Input conversion between SegEarth-OV and DINOv3."""

from __future__ import annotations

import torch
from torch import Tensor, nn


SEGEARTH_RGB_MEAN_255 = (
    122.771,
    116.746,
    104.094,
)

SEGEARTH_RGB_STD_255 = (
    68.501,
    66.632,
    70.323,
)


class SegEarthInputAdapter(nn.Module):
    """Recover RGB [0, 1] images from SegEarth-normalized tensors.

    SegEarth-OV's ``SegDataPreProcessor`` performs:

        BGR uint8
        -> RGB float32
        -> (RGB - mean) / std

    This adapter reverses only the normalization:

        normalized RGB float32
        -> RGB float32 in [0, 1]

    It must be called before SegEarth-OV converts the model input to FP16.
    """

    def __init__(self) -> None:
        super().__init__()

        mean = torch.tensor(
            SEGEARTH_RGB_MEAN_255,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            SEGEARTH_RGB_STD_255,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        self.register_buffer(
            "mean_255",
            mean,
            persistent=False,
        )
        self.register_buffer(
            "std_255",
            std,
            persistent=False,
        )

    @staticmethod
    def _validate(normalized_rgb: Tensor) -> None:
        if normalized_rgb.ndim != 4:
            raise ValueError(
                "Expected a BCHW tensor, "
                f"got shape {tuple(normalized_rgb.shape)}"
            )

        if normalized_rgb.shape[1] != 3:
            raise ValueError(
                "Expected 3 RGB channels, "
                f"got {normalized_rgb.shape[1]}"
            )

        if normalized_rgb.dtype != torch.float32:
            raise TypeError(
                "SegEarth input must be preserved as FP32 before "
                f"DINOv3 conversion, got {normalized_rgb.dtype}"
            )

        if not torch.isfinite(normalized_rgb).all():
            raise ValueError(
                "SegEarth input contains NaN or Inf values"
            )

    def forward(self, normalized_rgb: Tensor) -> Tensor:
        self._validate(normalized_rgb)

        rgb_01 = (
            normalized_rgb * self.std_255
            + self.mean_255
        ) / 255.0

        min_value, max_value = torch.aminmax(rgb_01)

        tolerance = 1e-5
        if (
            min_value.item() < -tolerance
            or max_value.item() > 1.0 + tolerance
        ):
            raise ValueError(
                "Recovered RGB values fall outside [0, 1]: "
                f"[{min_value.item():.6f}, "
                f"{max_value.item():.6f}]"
            )

        return rgb_01.clamp(0.0, 1.0).contiguous()
