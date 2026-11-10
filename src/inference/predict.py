"""
End-to-end inference pipeline for full-field microscopy images.
Slices large images into tiles, predicts cell masks with OneFormer,
and stitches them back into global instance maps with Soft-IoU suppression.
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path

# Add project root to sys.path so 'from src...' always resolves cleanly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
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
    parser.add_argument("--batch-size", type=int, default=None, help="Tile inference batch size")
    parser.add_argument("--score-threshold", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--stitch-iou", type=float, default=None, help="Soft-IoU seam deduplication threshold")
    parser.add_argument("--min-area", type=int, default=None, help="Minimum cell area in pixels")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (flips)")
    return parser.parse_args()


def colorize_instance_mask(label_matrix: np.ndarray) -> np.ndarray:
    """Assigns random distinct colors to each cell instance for visual inspection."""
    h, w = label_matrix.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    num_cells = int(label_matrix.max())

    if num_cells == 0:
        return colored

    np.random.seed(42)
    colors = np.random.randint(60, 255, size=(num_cells + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    colored = colors[label_matrix]
    return colored


def infer_tiles_batched(
    tiles: list,
    model_module: CellOneFormerModule,
    device: torch.device,
    task_key: str,
    batch_size: int,
    score_thresh: float,
    min_area: int,
) -> list:
    """Runs batched GPU inference across overlapping tiles."""
    tile_results = []
    with torch.no_grad():
        for start_idx in range(0, len(tiles), batch_size):
            batch_tiles = tiles[start_idx : start_idx + batch_size]
            pil_imgs = [Image.fromarray(t["image"]) for t in batch_tiles]
            target_sizes = [(t["image"].shape[0], t["image"].shape[1]) for t in batch_tiles]

            inputs = model_module.processor(
                images=pil_imgs,
                task_inputs=[task_key] * len(pil_imgs),
                return_tensors="pt",
            )
            device_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            outputs = model_module(**device_inputs)
            post_processed = model_module.post_process_instances(
                outputs,
                target_sizes=target_sizes,
                threshold=score_thresh,
                min_area=min_area,
            )

            for b_idx, tile in enumerate(batch_tiles):
                res = post_processed[b_idx] if b_idx < len(post_processed) else {}
                tile_masks = []
                tile_scores = []
                if "masks" in res and len(res["masks"]) > 0:
                    tile_masks = res["masks"]
                    tile_scores = res.get("scores", [1.0] * len(tile_masks))
                elif "segmentation" in res:
                    inst_map = res["segmentation"]
                    if isinstance(inst_map, torch.Tensor):
                        inst_map = inst_map.cpu().numpy()
                    for uid in np.unique(inst_map):
                        if uid > 0:
                            tile_masks.append(inst_map == uid)
                            tile_scores.append(1.0)

                tile_results.append({
                    "masks": tile_masks,
                    "scores": tile_scores,
                    "coords": tile["coords"],
                })
    return tile_results


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    percentile_norm = cfg["data"].get("percentile_norm", True)
    apply_clahe = cfg["data"].get("apply_clahe", False)

    print(f"[Inference] Loading image: {args.image}")
    image = load_microscopy_image(args.image, percentile_norm=percentile_norm, apply_clahe=apply_clahe)
    orig_h, orig_w = image.shape[:2]

    # Auto-detect num_queries from checkpoint if weights are provided
    num_queries = cfg["model"]["num_queries"]
    state_dict = None
    if args.weights and os.path.exists(args.weights):
        print(f"[Inference] Loading trained weights from {args.weights}...")
        state_dict = torch.load(args.weights, map_location=device)
        for k, v in state_dict.items():
            if "query_embed.weight" in k and hasattr(v, "shape"):
                num_queries = int(v.shape[0])
                break

    # Initialize model
    model_module = CellOneFormerModule(
        pretrained_model_name=cfg["model"]["pretrained_model_name"],
        num_classes=cfg["model"]["num_classes"],
        num_queries=num_queries,
    )

    if state_dict is not None:
        model_module.load_state_dict(state_dict, strict=False)
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
    batch_size = args.batch_size or cfg["inference"].get("batch_size", 4)
    score_thresh = args.score_threshold if args.score_threshold is not None else cfg["inference"]["score_threshold"]
    min_area = args.min_area if args.min_area is not None else cfg["data"]["min_cell_area"]
    use_tta = args.tta or cfg["inference"].get("tta", False)

    print(f"[Inference] Sliced {orig_h}x{orig_w} image into {len(tiles)} overlapping tiles (batch_size={batch_size}, TTA={use_tta}).")

    task_name = cfg["model"].get("task", "instance").lower()
    task_key = "instance" if "instance" in task_name else ("semantic" if "semantic" in task_name else "panoptic")

    tile_results = infer_tiles_batched(tiles, model_module, device, task_key, batch_size, score_thresh, min_area)

    # Optional Test-Time Augmentation (Horizontal & Vertical Flips)
    if use_tta:
        for flip_mode in ("hflip", "vflip"):
            flipped_img = np.fliplr(image).copy() if flip_mode == "hflip" else np.flipud(image).copy()
            f_tiles = tiler.tile_image_and_mask(flipped_img)
            f_results = infer_tiles_batched(f_tiles, model_module, device, task_key, batch_size, score_thresh, min_area)
            for res in f_results:
                y1, y2, x1, x2 = res["coords"]
                vh, vw = min(orig_h, y2) - y1, min(orig_w, x2) - x1
                if vh <= 0 or vw <= 0:
                    continue
                if flip_mode == "hflip":
                    inv_coords = (y1, y1 + vh, orig_w - (x1 + vw), orig_w - x1)
                    inv_masks = [np.fliplr(m[:vh, :vw]) for m in res["masks"]]
                else:
                    inv_coords = (orig_h - (y1 + vh), orig_h - y1, x1, x1 + vw)
                    inv_masks = [np.flipud(m[:vh, :vw]) for m in res["masks"]]
                tile_results.append({
                    "masks": inv_masks,
                    "scores": [s * 0.95 for s in res["scores"]],
                    "coords": inv_coords,
                })

    # Stitch predictions
    stitch_iou = args.stitch_iou if args.stitch_iou is not None else cfg["inference"]["iou_threshold_stitch"]
    max_area = cfg["data"].get("max_cell_area", 8000)
    gaussian_seam = cfg["inference"].get("gaussian_seam_weighting", True)
    stitcher = TileStitcher(
        iou_threshold=stitch_iou,
        min_area=min_area,
        max_area=max_area,
        gaussian_seam_weighting=gaussian_seam,
    )
    label_matrix, kept_masks = stitcher.stitch_tile_predictions(orig_h, orig_w, tile_results)

    num_detected_cells = int(label_matrix.max())
    print(f"[Inference] Stitched predictions: detected {num_detected_cells} individual cells.")

    # Save output label mask as uint16 to preserve >255 cell IDs
    stem = Path(args.image).stem
    out_mask_path = output_dir / f"{stem}_pred_mask.png"
    cv2.imwrite(str(out_mask_path), label_matrix.astype(np.uint16))

    # Save visual overlay with crisp contours and untouched background
    colored_vis = colorize_instance_mask(label_matrix)
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = np.stack([overlay] * 3, axis=-1)
    if num_detected_cells > 0:
        cell_mask = (label_matrix > 0)
        overlay[cell_mask] = (overlay[cell_mask].astype(float) * 0.45 + colored_vis[cell_mask].astype(float) * 0.55).astype(np.uint8)
        for uid in range(1, num_detected_cells + 1):
            single_cell = (label_matrix == uid).astype(np.uint8)
            contours, _ = cv2.findContours(single_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    out_vis_path = output_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(out_vis_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

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
