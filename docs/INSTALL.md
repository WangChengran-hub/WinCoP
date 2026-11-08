# Installation

## Reference Environment

The project was tested with:

- Python 3.9-compatible environment
- PyTorch 2.1.2+cu118
- torchvision 0.16.2+cu118
- NumPy 1.26.4
- timm 1.0.20
- safetensors 0.4.5
- MMEngine 0.10.4
- MMSegmentation 1.2.2
- MMCV 2.1.0

See requirements-reference.txt for recorded package versions.

## External Dependencies

Obtain and set up a compatible SegEarth-OV implementation.
Follow the original SegEarth-OV installation instructions,
including its CLIP, feature upsampling, and framework dependencies.

The WinCoP evaluation and inference scripts load dataset
configurations and pretrained semantic models from that project.

## Pretrained Weights

The repository does not distribute pretrained weights.

Provide the local DINOv3 ViT-B/16 safetensors checkpoint
using the --dino-weights argument.

SegEarth-OV model weights and any feature upsampling
checkpoints must be configured in the external SegEarth-OV
installation.

## Basic Verification

From the WinCoP repository root, check the commands:

    python tools/inference.py --help
    python tools/evaluate.py --help

Both inference and evaluation require a CUDA-capable environment
with the pretrained models and dataset configurations available.

## Execution

Use the environment prepared for SegEarth-OV.
Run commands from the WinCoP repository root.

The --segearth-root argument should point to the directory
containing segearth_segmentor.py, custom_datasets.py, and configs.
