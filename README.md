# WinCoP

## Training-Free Cross-Window Consensus for Open-Vocabulary Remote Sensing Segmentation

WinCoP refines open-vocabulary semantic segmentation predictions
through alternating cross-window coordinate consensus and
DINOv3-guided within-window structural propagation.

## Method

WinCoP uses frozen SegEarth-OV semantic responses and frozen
DINOv3 visual features. No additional model training is required.

Starting from patch-level semantic responses S:

    X_0 = S
    X_(t+1) = A_D P(X_t)

P aggregates semantic states sharing global patch coordinates.
A_D propagates information using DINOv3 patch relationships.

The reference AP4 implementation performs four iterations.
The resulting residual is projected back to dense window logits,
followed by overlapping-window reconstruction.

## Core Implementation

- gealign_ov/models/geoalign_ap4_core.py
- gealign_ov/models/geoalign_ov_core.py
- gealign_ov/methods/relational_transport.py
- gealign_ov/data/sliding_windows.py
- gealign_ov/backbones/dinov3_timm.py
- gealign_ov/adapters/segearth_input.py
- gealign_ov/models/geoalign_ov_segmentor.py
- gealign_ov/models/segearth_postprocess.py

## Dependencies

Core Python dependencies are listed in requirements.txt.
A compatible SegEarth-OV implementation is also required.

## Pretrained Models

Model weights and datasets are NOT included.
Users must obtain pretrained DINOv3 and SegEarth-OV weights separately.

The DINOv3 loader accepts a custom weight_path.
Its default relative path is weights/dinov3/model.safetensors.

## Reproducibility

This initial release provides the WinCoP core implementation.
Complete dataset-specific evaluation scripts and verified
end-to-end reproduction instructions are not yet included.

Dataset-specific Oracle configurations are separate from
the reference AP4 implementation.

## License

No open-source license has been selected yet.
