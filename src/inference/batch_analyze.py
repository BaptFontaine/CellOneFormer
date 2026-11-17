"""
Batch inference, morphometry export, and benchmark evaluation across a folder of microscopy images.
Generates 16-bit instance masks, overlays, per-image cell CSVs, and a global summary report.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.tiler import load_instance_mask, load_microscopy_image
from src.inference.predict import create_overlay, load_trained_model, predict_full_image
from src.training.train import find_image_mask_pairs
from src.utils.metrics import compute_challenge_benchmark_metrics_from_labels
from src.utils.morphometry import (
    extract_cell_morphometry,
    save_morphometry_csv,
    summarize_morphometry,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch cell segmentation and morphometric analysis")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder of images (or parent with images/ and labels/)")
    parser.add_argument("--weights", type=str, default="checkpoints/oneformer_real_cells_best.pt", help="Path to trained weights (.pt)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default="predictions/batch_analysis", help="Output folder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--stitch-iou", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--max-area", type=int, default=None)
    parser.add_argument("--pixel-size-um", type=float, default=None, help="Physical pixel size (um/px)")
    parser.add_argument("--tta", action="store_true", help="Enable flip TTA")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images to process")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    masks_out = output_dir / "masks"
    overlays_out = output_dir / "overlays"
    csv_out = output_dir / "morphometry"
    masks_out.mkdir(parents=True, exist_ok=True)
    overlays_out.mkdir(parents=True, exist_ok=True)
    csv_out.mkdir(parents=True, exist_ok=True)

    pairs = find_image_mask_pairs(Path(args.input_dir))
    if not pairs:
        # Fallback: scan direct files inside input-dir
        valid_exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
        pairs = [
            (p, None)
            for p in sorted(Path(args.input_dir).iterdir())
            if p.is_file() and p.suffix.lower() in valid_exts
        ]

    if args.max_images is not None and args.max_images > 0:
        pairs = pairs[: args.max_images]

    if not pairs:
        raise FileNotFoundError(f"No microscopy images found in {args.input_dir}")

    print(f"[Batch] Loading weights from {args.weights} on {device}...")
    model_module = load_trained_model(cfg, args.weights, device)

    percentile_norm = cfg["data"].get("percentile_norm", True)
    apply_clahe = cfg["data"].get("apply_clahe", False)

    summary_rows = []
    for img_path, mask_path in tqdm(pairs, desc="Analyzing images"):
        image = load_microscopy_image(str(img_path), percentile_norm=percentile_norm, apply_clahe=apply_clahe)
        label_matrix, _ = predict_full_image(
            image=image,
            model_module=model_module,
            cfg=cfg,
            device=device,
            score_thresh=args.score_threshold,
            stitch_iou=args.stitch_iou,
            min_area=args.min_area,
            max_area=args.max_area,
            use_tta=args.tta if args.tta else None,
            return_masks=False,
        )

        stem = img_path.stem
        cv2.imwrite(str(masks_out / f"{stem}_pred_mask.png"), label_matrix.astype(np.uint16))
        overlay = create_overlay(image, label_matrix)
        cv2.imwrite(str(overlays_out / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        cell_rows = extract_cell_morphometry(label_matrix, image=image, pixel_size_um=args.pixel_size_um)
        save_morphometry_csv(cell_rows, str(csv_out / f"{stem}_cells.csv"))

        summary = {"image": img_path.name, **summarize_morphometry(cell_rows, image.shape)}

        if mask_path is not None and mask_path.exists():
            gt_mask = load_instance_mask(str(mask_path))
            gt_ids = [gid for gid in np.unique(gt_mask) if gid > 0]
            if gt_ids:
                metrics = compute_challenge_benchmark_metrics_from_labels(label_matrix, gt_mask)
                summary["gt_count"] = len(gt_ids)
                summary["F1_50"] = round(float(metrics["F1_50"]), 4)
                summary["AP_50"] = round(float(metrics["AP_50"]), 4)
                summary["AP_75"] = round(float(metrics["AP_75"]), 4)
                summary["mean_f1"] = round(float(metrics["mean_f1"]), 4)

        summary_rows.append(summary)

    summary_path = output_dir / "summary_report.csv"
    if summary_rows:
        # Collect all keys across rows (some may include gt metrics)
        fieldnames = list(dict.fromkeys(k for r in summary_rows for k in r.keys()))
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\n[Batch] Processed {len(summary_rows)} images.")
    print(f"[Batch] Summary report saved to: {summary_path}")

    f1_vals = [r["F1_50"] for r in summary_rows if "F1_50" in r]
    if f1_vals:
        print(f"[Batch] Mean F1@0.5 across evaluated images: {np.mean(f1_vals):.4f}")


if __name__ == "__main__":
    main()
