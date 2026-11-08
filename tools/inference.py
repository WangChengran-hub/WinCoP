"""Single-image inference with frozen WinCoP models."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DATASETS = (
    "loveda", "potsdam", "uavid", "dlrsd",
    "udd5", "vaihingen", "vdd",
)


def main():
    parser = argparse.ArgumentParser(
        description="WinCoP single-image inference"
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segearth-root", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    segearth_root = args.segearth_root.resolve()
    image_path = args.image.resolve()
    output_path = args.output.resolve()

    config_path = (
        segearth_root / "configs" / f"cfg_{args.dataset}.py"
    )

    for path in (config_path, image_path, args.dino_weights):
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_path.exists():
        raise FileExistsError(output_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    sys.path.insert(0, str(root))
    sys.path.insert(0, str(segearth_root))

    from mmcv.transforms import Compose
    from mmengine.config import Config
    from mmengine.runner import Runner

    import segearth_segmentor
    import custom_datasets

    from gealign_ov.backbones.dinov3_timm import (
        DINOv3TimmBackbone,
    )
    from gealign_ov.models.geoalign_ap4_core import (
        GeoAlignAP4Core,
    )
    from gealign_ov.models.geoalign_ov_segmentor import (
        GeoAlignOVSegmentor,
    )
    from gealign_ov.models.segearth_postprocess import (
        FrozenSegEarthPostProcessor,
    )
    from evaluate import build_query_groups

    os.chdir(segearth_root)

    cfg = Config.fromfile(str(config_path))
    cfg.work_dir = str(root / "outputs" / args.dataset)

    runner = Runner.from_cfg(cfg)
    model = runner.model.eval()

    class_names = tuple(
        runner.test_dataloader.dataset.metainfo["classes"]
    )

    prompt_path = segearth_root / cfg.model.name_path
    query_groups = build_query_groups(prompt_path)

    if len(query_groups) != len(class_names):
        raise RuntimeError("Class/query-group mismatch")

    if sum(map(len, query_groups)) != int(model.query_idx.numel()):
        raise RuntimeError("Query count mismatch")

    device = torch.device("cuda")

    dino = DINOv3TimmBackbone(
        weight_path=args.dino_weights.resolve()
    ).to(device).eval()

    core = GeoAlignAP4Core(
        segearth_model=model,
        dino_backbone=dino,
        query_groups=query_groups,
        semantic_logit_scale=float(model.logit_scale),
    ).to(device).eval()

    postprocessor = FrozenSegEarthPostProcessor(
        query_groups=query_groups,
        logit_scale=float(model.logit_scale),
        probability_threshold=float(model.prob_thd),
        background_index=int(model.bg_idx),
    )

    segmentor = GeoAlignOVSegmentor(
        core=core,
        postprocessor=postprocessor,
    ).to(device).eval()

    # Reuse the dataset's official image-loading and resizing
    # transforms, without requiring an annotation.
    pipeline_cfg = [
        step for step in cfg.test_pipeline
        if step["type"] != "LoadAnnotations"
    ]
    pipeline = Compose(pipeline_cfg)
    packed = pipeline({"img_path": str(image_path)})

    if packed is None:
        raise RuntimeError("Image preprocessing failed")

    batch = {
        "inputs": [packed["inputs"]],
        "data_samples": [packed["data_samples"]],
    }

    with torch.inference_mode():
        processed = model.data_preprocessor(
            batch, training=False
        )
        inputs = processed["inputs"].to(device)
        sample = processed["data_samples"][0]

        if inputs.dtype != torch.float32:
            raise RuntimeError("Expected FP32 input")

        output_shape = tuple(
            int(v) for v in sample.ori_shape[:2]
        )

        result = segmentor(
            inputs,
            output_shape=output_shape,
        )

        prediction = (
            result.prediction.detach().cpu().numpy()
        )

    prediction = np.asarray(prediction).squeeze()

    if prediction.shape != output_shape:
        raise RuntimeError("Prediction shape mismatch")

    if not np.isfinite(prediction).all():
        raise RuntimeError("Invalid prediction")

    if prediction.min() < 0 or prediction.max() >= len(class_names):
        raise RuntimeError("Invalid class index")

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    Image.fromarray(
        prediction.astype(np.uint8)
    ).save(output_path)

    print("WINCOP_INFERENCE_PASS")
    print("Dataset:", args.dataset)
    print("Image:", image_path)
    print("Prediction:", output_path)
    print("Output shape:", prediction.shape)
    print("Classes:", class_names)


if __name__ == "__main__":
    main()
