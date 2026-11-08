from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .geoalign_ov_core import (
    GeoAlignOVCore,
    GeoAlignOVCoreOutput,
)

from .segearth_postprocess import (
    FrozenSegEarthPostProcessor,
    SegEarthPostprocessOutput,
)


@dataclass(frozen=True)
class GeoAlignOVPrediction:
    """
    Final prediction plus baseline control generated from
    exactly the same live SegEarth windows.
    """

    prediction: Tensor
    baseline_prediction: Tensor

    confidence: Tensor
    baseline_confidence: Tensor

    core: GeoAlignOVCoreOutput


class GeoAlignOVSegmentor(nn.Module):
    """
    Formal frozen GeoAlign-OV inference model.

    Pipeline
    --------
    FP32 SegEarth-preprocessed input
        -> shared 224 / stride 112 windows
        -> SegEarth raw 9-query logits
        -> DINOv3 patch relations
        -> frozen relational transport
        -> CSCT refined 9-query logits
        -> overlap average
        -> frozen SegEarth postprocess
        -> semantic prediction
    """

    def __init__(
        self,
        core: GeoAlignOVCore,
        postprocessor: FrozenSegEarthPostProcessor,
    ) -> None:
        super().__init__()

        self.core = core
        self.postprocessor = (
            postprocessor
        )

        self.train(False)

    def train(
        self,
        mode: bool = True,
    ) -> "GeoAlignOVSegmentor":
        super().train(False)

        self.core.eval()
        self.postprocessor.eval()

        return self

    @torch.inference_mode()
    def forward(
        self,
        normalized_rgb: Tensor,
        output_shape: tuple[int, int],
    ) -> GeoAlignOVPrediction:
        core_output = self.core(
            normalized_rgb
        )

        baseline: (
            SegEarthPostprocessOutput
        ) = self.postprocessor(
            core_output.base_logits_canvas,
            output_shape=output_shape,
        )

        refined: (
            SegEarthPostprocessOutput
        ) = self.postprocessor(
            core_output.refined_logits_canvas,
            output_shape=output_shape,
        )

        return GeoAlignOVPrediction(
            prediction=refined.prediction,
            baseline_prediction=baseline.prediction,
            confidence=refined.confidence,
            baseline_confidence=baseline.confidence,
            core=core_output,
        )
