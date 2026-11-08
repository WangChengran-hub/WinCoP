# WinCoP

## Training-Free Cross-Window Consensus for Open-Vocabulary Remote Sensing Segmentation

WinCoP is a training-free approach for refining open-vocabulary
remote sensing semantic segmentation using cross-window coordinate
consensus and DINOv3-guided structural propagation.

## Overview

WinCoP combines frozen SegEarth-OV semantic responses with
frozen DINOv3 visual features.

At each iteration, semantic states are aggregated across windows
at corresponding global patch coordinates and then propagated
through DINOv3-derived within-window relationships.

    X_0 = S
    X_(t+1) = A_D P(X_t)

The reference implementation uses four alternating iterations.
The resulting semantic residuals are restored to dense window
responses, followed by overlapping-window reconstruction.

No additional model training is required.

## Supported Datasets

- LoveDA
- Potsdam
- UAVid
- DLRSD
- UDD5
- Vaihingen
- VDD

## Installation

See docs/INSTALL.md for environment setup, external model
dependencies, and pretrained weight preparation.

Pretrained model weights and datasets are not included.

## Single-Image Inference

Run from the WinCoP repository root:

    python tools/inference.py \
      --dataset loveda \
      --image /path/to/image.png \
      --output outputs/prediction.png \
      --segearth-root /path/to/SegEarth-OV \
      --dino-weights /path/to/dinov3/model.safetensors

The output PNG contains semantic class IDs.

## Dataset Evaluation

    python tools/evaluate.py \
      --dataset loveda \
      --segearth-root /path/to/SegEarth-OV \
      --dino-weights /path/to/dinov3/model.safetensors \
      --output outputs/loveda_metrics.json

The evaluator reports pooled mIoU, mean class accuracy,
overall pixel accuracy, per-class IoU, and the confusion matrix.

See docs/DATASETS.md for the supported datasets.

## Implementation

- gealign_ov/models/geoalign_ap4_core.py: WinCoP refinement
- gealign_ov/models/geoalign_ov_core.py: shared model utilities
- gealign_ov/methods/relational_transport.py: structural relations
- gealign_ov/data/sliding_windows.py: window geometry
- gealign_ov/backbones/dinov3_timm.py: DINOv3 features
- gealign_ov/models/segearth_postprocess.py: prediction recovery
- tools/inference.py: standalone image inference
- tools/evaluate.py: dataset evaluation

## Reproducibility

The implementation uses a frozen SegEarth-OV model,
a frozen DINOv3 ViT-B/16 backbone, 224-pixel windows,
a window stride of 112 pixels, and four refinement iterations.

The reference relation operator uses K=8 and tau=0.07,
with residual coefficient alpha=0.75.

Dataset-specific semantic prompts, preprocessing, and annotation
handling follow the corresponding SegEarth-OV configurations.

## Acknowledgments

This project uses SegEarth-OV and DINOv3 as external
pretrained model dependencies. Please follow their respective
licensing and citation requirements.
