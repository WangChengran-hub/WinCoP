"""Standard dataset evaluation for WinCoP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


DATASETS = (
    "loveda",
    "potsdam",
    "uavid",
    "dlrsd",
    "udd5",
    "vaihingen",
    "vdd",
)


def build_query_groups(prompt_path):
    lines = [
        line.strip()
        for line in prompt_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    groups = []
    offset = 0

    for line in lines:
        queries = [
            item.strip()
            for item in line.split(",")
            if item.strip()
        ]
        if not queries:
            raise ValueError("Empty semantic query group")

        groups.append(
            tuple(range(offset, offset + len(queries)))
        )
        offset += len(queries)

    return tuple(groups)


def update_confusion(matrix, prediction, gt, num_classes):
    prediction = np.asarray(prediction, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)

    if prediction.shape != gt.shape:
        raise ValueError("Prediction/GT shape mismatch")

    valid = (gt >= 0) & (gt < num_classes)

    if np.any(
        (prediction[valid] < 0)
        | (prediction[valid] >= num_classes)
    ):
        raise ValueError("Invalid predicted class ID")

    encoded = (
        gt[valid] * num_classes + prediction[valid]
    )

    matrix += np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def calculate_metrics(matrix):
    matrix = matrix.astype(np.float64)
    tp = np.diag(matrix)
    gt_count = matrix.sum(axis=1)
    pred_count = matrix.sum(axis=0)
    union = gt_count + pred_count - tp

    iou = np.divide(
        tp, union,
        out=np.full_like(tp, np.nan),
        where=union > 0,
    )

    accuracy = np.divide(
        tp, gt_count,
        out=np.full_like(tp, np.nan),
        where=gt_count > 0,
    )

    return {
        "mIoU": float(np.nanmean(iou) * 100),
        "mAcc": float(np.nanmean(accuracy) * 100),
        "aAcc": float(
            tp.sum() / max(gt_count.sum(), 1) * 100
        ),
        "class_iou": iou * 100,
        "class_acc": accuracy * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        required=True,
    )
    parser.add_argument(
        "--segearth-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dino-weights",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of samples; 0 evaluates all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    project_root = Path(__file__).resolve().parents[1]
    segearth_root = args.segearth_root.resolve()
    dino_weights = args.dino_weights.resolve()

    config_path = (
        segearth_root
        / "configs"
        / f"cfg_{args.dataset}.py"
    )

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    if not dino_weights.is_file():
        raise FileNotFoundError(dino_weights)

    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(segearth_root))
    os.chdir(segearth_root)

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

    cfg = Config.fromfile(str(config_path))
    cfg.work_dir = str(
        project_root / "outputs" / args.dataset
    )

    runner = Runner.from_cfg(cfg)
    model = runner.model.eval()
    dataloader = runner.test_dataloader
    dataset = dataloader.dataset

    class_names = tuple(dataset.metainfo["classes"])
    num_classes = len(class_names)

    prompt_path = segearth_root / cfg.model.name_path

    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)

    query_groups = build_query_groups(prompt_path)

    if len(query_groups) != num_classes:
        raise RuntimeError(
            "Prompt groups and dataset classes differ: "
            f"{len(query_groups)} vs {num_classes}"
        )

    query_count = sum(len(g) for g in query_groups)
    model_query_count = int(model.query_idx.numel())

    if query_count != model_query_count:
        raise RuntimeError(
            "Semantic query count mismatch: "
            f"{query_count} vs {model_query_count}"
        )

    device = torch.device("cuda")

    dino = DINOv3TimmBackbone(
        weight_path=dino_weights
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

    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    processed_count = 0

    print("Dataset:", args.dataset, flush=True)
    print("Classes:", class_names, flush=True)
    print("Samples:", len(dataset), flush=True)
    print("Queries:", query_count, flush=True)

    with torch.inference_mode():
        for batch in dataloader:
            processed = model.data_preprocessor(
                batch,
                training=False,
            )

            inputs = processed["inputs"]
            samples = processed["data_samples"]

            if len(samples) != 1:
                raise RuntimeError("Expected batch size 1")

            if inputs.dtype != torch.float32:
                raise RuntimeError("Expected FP32 inputs")

            inputs = inputs.to(device)
            sample = samples[0]

            gt = sample.gt_sem_seg.data

            if gt.ndim == 3:
                if gt.shape[0] != 1:
                    raise RuntimeError("Invalid GT shape")
                gt = gt[0]

            gt_np = gt.detach().cpu().numpy().astype(
                np.int64
            )

            output_shape = tuple(
                int(v) for v in sample.ori_shape[:2]
            )

            if gt_np.shape != output_shape:
                raise RuntimeError(
                    "GT/original image geometry mismatch"
                )

            result = segmentor(
                inputs,
                output_shape=output_shape,
            )

            prediction = (
                result.prediction
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )

            if prediction.ndim == 3 and prediction.shape[0] == 1:
                prediction = prediction[0]

            update_confusion(
                matrix,
                prediction,
                gt_np,
                num_classes,
            )

            processed_count += 1

            if processed_count % 50 == 0:
                print(
                    f"Processed {processed_count}/{len(dataset)}",
                    flush=True,
                )

            if args.limit > 0 and processed_count >= args.limit:
                break

    metrics = calculate_metrics(matrix)

    report = {
        "dataset": args.dataset,
        "processed_images": processed_count,
        "num_classes": num_classes,
        "class_names": list(class_names),
        "mIoU": metrics["mIoU"],
        "mAcc": metrics["mAcc"],
        "aAcc": metrics["aAcc"],
        "class_IoU": {
            name: (
                float(value)
                if np.isfinite(value)
                else None
            )
            for name, value in zip(
                class_names, metrics["class_iou"]
            )
        },
        "confusion_matrix": matrix.tolist(),
    }

    print(json.dumps(report, indent=2))

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        if output_path.exists():
            raise FileExistsError(output_path)
        output_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
