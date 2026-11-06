"""
End-to-end inference pipeline for full-field microscopy images.
Slices large images into tiles, predicts cell masks with OneFormer,
and stitches them back into global instance maps with Soft-IoU suppression.
"""

import os
import argparse
import yaml
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image

from src.dataset.tiler import SlidingWindowTiler, load_microscopy_image, load_instance_mask
from src.inference.stitcher import TileStitcher
from src.models.oneformer_wrapper import CellOneFormerModule
from src.utils.metrics import compute_challenge_benchmark_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Predict cell instances on full-field microscopy images")
    parser.add_argument("--image", type=str, required=True, help="Path to input microscopy image")
    parser.add_argument("--mask", type=str, default=None, help="Optional path to ground-truth mask for evaluation")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default="predictions", help="Directory to save output masks and visualisations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    return parser.parse_args()


def colorize_instance_mask(label_matrix: np.ndarray) -> np.ndarray:
    """Assigns random distinct colors to each cell instance for visual inspection."""
    h, w = label_matrix.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    num_cells = int(label_matrix.max())

    if num_cells == 0:
        return colored

    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_cells + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    colored = colors[label_matrix]
    return colored


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Inference] Loading image: {args.image}")
    image = load_microscopy_image(args.image)
    orig_h, orig_w = image.shape[:2]

    # Initialize model
    model_module = CellOneFormerModule(
        pretrained_model_name=cfg["model"]["pretrained_model_name"],
        num_classes=cfg["model"]["num_classes"],
        num_queries=cfg["model"]["num_queries"],
    )

    if args.weights and os.path.exists(args.weights):
        print(f"[Inference] Loading trained weights from {args.weights}...")
        model_module.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print("[Inference] No weights provided; running zero-shot with base weights.")

    model_module.to(device)
    model_module.eval()

    # Slice image into tiles
    tiler = SlidingWindowTiler(
        tile_size=cfg["inference"]["tile_size"],
        tile_overlap=cfg["inference"]["tile_overlap"],
    )
    tiles = tiler.tile_image_and_mask(image)
    print(f"[Inference] Sliced {orig_h}x{orig_w} image into {len(tiles)} overlapping tiles.")

    tile_results = []
    task_prompt = f"the task is {cfg['model']['task']}"

    # Predict on each tile
    with torch.no_grad():
        for tile in tiles:
            img_crop = tile["image"]
            pil_img = Image.fromarray(img_crop)

            inputs = model_module.processor(
                images=pil_img,
                task_inputs=[task_prompt],
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(device)
            text_inputs = inputs["text_inputs"].to(device)

            outputs = model_module(
                pixel_values=pixel_values,
                text_inputs=text_inputs,
            )

            # Post-process into binary masks
            post_processed = model_module.post_process_instances(
                outputs,
                target_sizes=[(cfg["inference"]["tile_size"], cfg["inference"]["tile_size"])],
                threshold=cfg["inference"]["score_threshold"],
            )

            tile_masks = []
            tile_scores = []
            if len(post_processed) > 0 and "segmentation" in post_processed[0]:
                inst_map = post_processed[0]["segmentation"].cpu().numpy()
                u_ids = np.unique(inst_map)
                for uid in u_ids:
                    if uid <= 0:
                        continue
                    tile_masks.append(inst_map == uid)
                    tile_scores.append(1.0)

            tile_results.append({
                "masks": tile_masks,
                "scores": tile_scores,
                "coords": tile["coords"],
            })

    # Stitch predictions
    stitcher = TileStitcher(
        iou_threshold=cfg["inference"]["iou_threshold_stitch"],
        min_area=cfg["data"]["min_cell_area"],
    )
    label_matrix, kept_masks = stitcher.stitch_tile_predictions(orig_h, orig_w, tile_results)

    num_detected_cells = int(label_matrix.max())
    print(f"[Inference] Stitched predictions: detected {num_detected_cells} individual cells.")

    # Save output label mask
    stem = Path(args.image).stem
    out_mask_path = output_dir / f"{stem}_pred_mask.png"
    cv2.imwrite(str(out_mask_path), label_matrix)

    # Save visual overlay
    colored_vis = colorize_instance_mask(label_matrix)
    alpha = 0.5
    overlay = cv2.addWeighted(image, 1 - alpha, colored_vis, alpha, 0)
    out_vis_path = output_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(out_vis_path), overlay)

    print(f"[Inference] Saved label mask to: {out_mask_path}")
    print(f"[Inference] Saved overlay to:    {out_vis_path}")

    # Evaluate against ground truth if supplied
    if args.mask and os.path.exists(args.mask):
        gt_mask = load_instance_mask(args.mask)
        gt_ids = np.unique(gt_mask)
        gt_instances = [gt_mask == gid for gid in gt_ids if gid > 0]
        if len(gt_instances) > 0:
            gt_instances = np.array(gt_instances)
            metrics = compute_challenge_benchmark_metrics(kept_masks, gt_instances)
            print("\n" + "=" * 40)
            print(" Benchmark Evaluation Metrics")
            print("=" * 40)
            for k, v in metrics.items():
                print(f"  {k:12s} : {v:.4f}" if isinstance(v, float) else f"  {k:12s} : {v}")
            print("=" * 40)


if __name__ == "__main__":
    main()
