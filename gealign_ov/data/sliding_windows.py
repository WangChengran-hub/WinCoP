"""Deterministic sliding-window geometry shared by both model branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SlidingWindow:
    """One spatial crop in exclusive-boundary coordinates."""

    index: int
    row: int
    column: int
    y1: int
    x1: int
    y2: int
    x2: int

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def slices(self) -> Tuple[slice, slice]:
        return (
            slice(self.y1, self.y2),
            slice(self.x1, self.x2),
        )


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

    return value


def compute_sliding_windows(
    image_height: int,
    image_width: int,
    crop_height: int,
    crop_width: int,
    stride_height: int,
    stride_width: int,
) -> Tuple[SlidingWindow, ...]:
    """Reproduce the crop geometry used by SegEarth-OV ``forward_slide``.

    The final window in each direction is shifted backward when necessary so
    that its lower/right boundary coincides with the image boundary. Images
    smaller than the requested crop produce one full-image window; model-level
    padding remains the responsibility of the feature extractor.
    """

    image_height = _validate_positive_int(image_height, "image_height")
    image_width = _validate_positive_int(image_width, "image_width")
    crop_height = _validate_positive_int(crop_height, "crop_height")
    crop_width = _validate_positive_int(crop_width, "crop_width")
    stride_height = _validate_positive_int(stride_height, "stride_height")
    stride_width = _validate_positive_int(stride_width, "stride_width")

    height_grids = (
        max(image_height - crop_height + stride_height - 1, 0)
        // stride_height
        + 1
    )
    width_grids = (
        max(image_width - crop_width + stride_width - 1, 0)
        // stride_width
        + 1
    )

    windows = []
    index = 0

    for row in range(height_grids):
        for column in range(width_grids):
            y1 = row * stride_height
            x1 = column * stride_width

            y2 = min(y1 + crop_height, image_height)
            x2 = min(x1 + crop_width, image_width)

            y1 = max(y2 - crop_height, 0)
            x1 = max(x2 - crop_width, 0)

            windows.append(
                SlidingWindow(
                    index=index,
                    row=row,
                    column=column,
                    y1=y1,
                    x1=x1,
                    y2=y2,
                    x2=x2,
                )
            )
            index += 1

    return tuple(windows)
