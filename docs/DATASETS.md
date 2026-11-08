# Dataset Preparation

WinCoP supports the following seven evaluation configurations.

| Dataset | Classes | External configuration |
| --- | ---: | --- |
| LoveDA | 7 | configs/cfg_loveda.py |
| Potsdam | 6 | configs/cfg_potsdam.py |
| UAVid | 7 | configs/cfg_uavid.py |
| DLRSD | 17 | configs/cfg_dlrsd.py |
| UDD5 | 5 | configs/cfg_udd5.py |
| Vaihingen | 6 | configs/cfg_vaihingen.py |
| VDD | 7 | configs/cfg_vdd.py |

These configuration paths are relative to the external
SegEarth-OV repository.

## Dataset Layout

Prepare each dataset according to the paths declared in
its SegEarth-OV configuration.

For evaluation, the corresponding image directories and
annotation directories must exist.

Do not modify the annotation indexing conventions to match
text prompts manually. The dataset classes, ignore indices,
and label remapping are controlled by the configured
SegEarth-OV dataset implementations.

## Evaluation

Example:

    python tools/evaluate.py \
      --dataset potsdam \
      --segearth-root /path/to/SegEarth-OV \
      --dino-weights /path/to/dinov3/model.safetensors \
      --output outputs/potsdam_metrics.json

The evaluation script computes metrics from the pooled
confusion matrix across processed images.

For a short functional test, add --limit 1.
Results obtained from a limited subset are not full-dataset
evaluation results.

## Inference Without Annotations

The inference script accepts an individual image and saves
a single-channel PNG of predicted semantic class IDs.
Annotation files are not required for this operation.
