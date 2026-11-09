"""
Training script for OneFormer Multi-Modality Cell Segmentation.
Designed for remote execution (GPU servers, Docker, Colab, Slurm).
Includes --dry-run mode for local pipeline verification without heavy training.
Supports official challenge structure: DatasetCellSeg/{training, tuning, testing}.
"""

import os
import sys

# Configure PyTorch CUDA memory allocator to reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import yaml
import glob
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to sys.path so 'from src...' always resolves cleanly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.dataset.dataset_loader import CellOneFormerDataset, collate_fn_oneformer
from src.dataset.tiler import (
    generate_synthetic_cell_data,
    SlidingWindowTiler,
    load_microscopy_image,
    load_instance_mask,
)
from src.models.oneformer_wrapper import CellOneFormerModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train OneFormer on Microscopy Cell Segmentation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Run a lightweight pipeline check with 2 samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()


def find_image_mask_pairs(folder_path: Path) -> List[Tuple[Path, Optional[Path]]]:
    """
    Universally locates image and mask pairs inside a folder, supporting both:
    1. Subdirectories: folder/images and folder/labels (or masks)
    2. Flat folder: image.png and image_label.png / image_mask.png
    """
    pairs = []
    if not folder_path.exists():
        return pairs

    # Case 1: Separate images/ and labels/ subdirectories
    img_sub = folder_path / "images"
    lbl_sub = folder_path / "labels"
    if not lbl_sub.exists():
        lbl_sub = folder_path / "masks"

    if img_sub.exists():
        exts = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
        img_files = []
        for ext in exts:
            img_files.extend(list(img_sub.glob(ext)))
        img_files.sort()

        for img_p in img_files:
            stem = img_p.stem
            # Check corresponding mask in labels dir
            mask_p = None
            if lbl_sub.exists():
                for candidate in [
                    lbl_sub / f"{stem}.png",
                    lbl_sub / f"{stem}.tif",
                    lbl_sub / f"{stem}_label.png",
                    lbl_sub / f"{stem}_mask.png",
                    lbl_sub / f"{stem}_label.tif",
                    lbl_sub / f"{stem}_mask.tif",
                ]:
                    if candidate.exists():
                        mask_p = candidate
                        break
            pairs.append((img_p, mask_p))
        return pairs

    # Case 2: Flat directory containing images and masks
    exts = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
    all_files = []
    for ext in exts:
        all_files.extend(list(folder_path.glob(ext)))
    all_files.sort()

    # Distinguish images from label masks
    label_files = {f.stem: f for f in all_files if any(kw in f.stem.lower() for kw in ["label", "mask", "gt"])}
    image_files = [f for f in all_files if f.stem not in label_files]

    for img_p in image_files:
        stem = img_p.stem
        mask_p = None
        for cand_key in [f"{stem}_label", f"{stem}_mask", f"{stem}_gt", stem]:
            if cand_key in label_files:
                mask_p = label_files[cand_key]
                break
        pairs.append((img_p, mask_p))

    return pairs


def tile_and_save_split(
    pairs: List[Tuple[Path, Optional[Path]]],
    output_split_dir: Path,
    tiler: SlidingWindowTiler,
    is_dry_run: bool = False,
    max_samples: Optional[int] = None,
) -> int:
    """Slices a list of (image, mask) pairs into tiles and writes them to disk."""
    out_img_dir = output_split_dir / "images"
    out_msk_dir = output_split_dir / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    if is_dry_run:
        pairs = pairs[:2]
    elif max_samples is not None and max_samples > 0:
        pairs = pairs[:max_samples]

    tile_count = 0
    for img_path, mask_path in tqdm(pairs, desc=f"Tiling {output_split_dir.name}"):
        img = load_microscopy_image(str(img_path))
        mask = load_instance_mask(str(mask_path)) if (mask_path and mask_path.exists()) else None

        tiles = tiler.tile_image_and_mask(img, mask)
        for t_idx, tile in enumerate(tiles):
            save_img_path = out_img_dir / f"{img_path.stem}_tile{t_idx:03d}.png"
            cv2.imwrite(str(save_img_path), tile["image"])
            if tile["mask"] is not None:
                save_msk_path = out_msk_dir / f"{img_path.stem}_tile{t_idx:03d}_mask.png"
                cv2.imwrite(str(save_msk_path), tile["mask"])
            tile_count += 1

    return tile_count


def prepare_tiled_data(cfg, is_dry_run=False):
    """
    Prepares and tiles datasets.
    Supports either DatasetCellSeg/{training, tuning, testing} or standard data/raw/images.
    """
    raw_root = Path(cfg["data"]["raw_dir"])
    cellseg_root = raw_root / "DatasetCellSeg"
    if not cellseg_root.exists() and (raw_root / "training").exists():
        cellseg_root = raw_root

    def find_case_dir(root: Path, name: str) -> Path:
        for c in [name, name.capitalize(), name.lower(), name.upper()]:
            p = root / c
            if p.exists():
                return p
        return root / name

    # Identify training and tuning (val) folders (case-insensitive)
    train_source = find_case_dir(cellseg_root, "training")
    tuning_source = find_case_dir(cellseg_root, "tuning")
    test_source = find_case_dir(cellseg_root, "testing")

    processed_train_img = Path(cfg["data"]["processed_dir"]) / "train" / "images"
    processed_val_img = Path(cfg["data"]["processed_dir"]) / "val" / "images"

    # If processed data already exists, skip
    if processed_train_img.exists() and len(list(processed_train_img.glob("*.png"))) > 0:
        print(f"[Setup] Processed tiles already present in {cfg['data']['processed_dir']}. Skipping tiling.")
        return

    # Check if raw files exist
    train_pairs = find_image_mask_pairs(train_source)
    tuning_pairs = find_image_mask_pairs(tuning_source)

    # Fallback to data/raw/images if DatasetCellSeg/training is not found
    if len(train_pairs) == 0:
        train_pairs = find_image_mask_pairs(raw_root)

    # If still empty, generate synthetic data for testing
    if len(train_pairs) == 0:
        print("[Setup] Raw data directory empty. Creating synthetic dataset for pipeline validation...")
        synth_dir = str(raw_root / "DatasetCellSeg" / "training")
        generate_synthetic_cell_data(
            output_dir=synth_dir,
            num_samples=2 if is_dry_run else 6,
            height=1024,
            width=1024,
        )
        train_pairs = find_image_mask_pairs(Path(synth_dir))

    tiler = SlidingWindowTiler(
        tile_size=cfg["data"]["tile_size"],
        tile_overlap=cfg["data"]["tile_overlap"],
        min_cell_area=cfg["data"]["min_cell_area"],
    )

    # Tile training set
    max_train = cfg["data"].get("max_train_samples", None)
    n_train_tiles = tile_and_save_split(
        train_pairs,
        Path(cfg["data"]["processed_dir"]) / "train",
        tiler,
        is_dry_run=is_dry_run,
        max_samples=max_train,
    )
    print(f"[Setup] Generated {n_train_tiles} training tiles.")

    # Tile tuning set (if available)
    if len(tuning_pairs) > 0:
        max_val = cfg["data"].get("max_val_samples", None)
        n_val_tiles = tile_and_save_split(
            tuning_pairs,
            Path(cfg["data"]["processed_dir"]) / "val",
            tiler,
            is_dry_run=is_dry_run,
            max_samples=max_val,
        )
        print(f"[Setup] Generated {n_val_tiles} validation/tuning tiles.")


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    device = torch.device(args.device)

    print("=" * 60)
    print(" Multi-Modality Cell Segmentation with OneFormer")
    print(f" Target Device       : {device}")
    print(f" Dry-Run Mode        : {args.dry_run}")
    print(f" Config File         : {args.config}")
    print("=" * 60)

    # 1. Prepare data & tiling
    prepare_tiled_data(cfg, is_dry_run=args.dry_run)

    # In dry-run mode on local PC, confirm pipeline compatibility and exit cleanly without burning GPU/CPU
    if args.dry_run:
        print("[Dry-Run] Initializing dataset and verifying batch shapes...")
        train_img_dir = str(Path(cfg["data"]["processed_dir"]) / "train" / "images")
        train_msk_dir = str(Path(cfg["data"]["processed_dir"]) / "train" / "masks")

        dataset = CellOneFormerDataset(
            images_dir=train_img_dir,
            masks_dir=train_msk_dir,
            is_train=False,
            max_samples=2,
        )
        sample = dataset[0]
        print(f"[Dry-Run] Sample loaded: Image shape = {sample['image'].shape}, Task = '{sample['task']}'")
        print("=" * 60)
        print(" [SUCCESS] Pipeline verification complete!")
        print(" The data preprocessing, tiling and dataset loading are valid.")
        print(" Ready for remote training on Google Colab or Docker.")
        print("=" * 60)
        return

    # 2. Remote Full Training Execution
    train_img_dir = str(Path(cfg["data"]["processed_dir"]) / "train" / "images")
    train_msk_dir = str(Path(cfg["data"]["processed_dir"]) / "train" / "masks")
    val_img_dir = str(Path(cfg["data"]["processed_dir"]) / "val" / "images")
    val_msk_dir = str(Path(cfg["data"]["processed_dir"]) / "val" / "masks")

    # Load Model & Processor
    model_module = CellOneFormerModule(
        pretrained_model_name=cfg["model"]["pretrained_model_name"],
        num_classes=cfg["model"]["num_classes"],
        num_queries=cfg["model"]["num_queries"],
        is_training=True,
    )
    model_module.to(device)

    # Training Dataset & DataLoader
    train_dataset = CellOneFormerDataset(
        images_dir=train_img_dir,
        masks_dir=train_msk_dir,
        processor=model_module.processor,
        task=cfg["model"]["task"],
        is_train=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_oneformer,
        num_workers=2,
    )

    optimizer = AdamW(model_module.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = (cfg["training"]["mixed_precision"] in ("fp16", "bf16")) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    grad_accum_steps = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    print(f"\n[Training] Starting remote training for {epochs} epochs on {device} (batch_size={batch_size}, grad_accum={grad_accum_steps})...")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model_module.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            if "mask_labels" not in batch or batch["mask_labels"] is None or len(batch["mask_labels"]) == 0:
                continue
            pixel_values = batch["pixel_values"].to(device)
            text_inputs = batch.get("text_inputs")
            if text_inputs is not None:
                text_inputs = text_inputs.to(device)
            task_inputs = batch.get("task_inputs")
            if task_inputs is not None:
                task_inputs = task_inputs.to(device)
            pixel_mask = batch.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device)
            mask_labels = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]

            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp) if hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=use_amp)
            with autocast_ctx:
                outputs = model_module(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    text_inputs=text_inputs,
                    task_inputs=task_inputs,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                loss = outputs.loss
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(1, len(train_loader))
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[Epoch {epoch}/{epochs}] Average Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = output_dir / "oneformer_cells_best.pt"
            torch.save(model_module.state_dict(), save_path)
            print(f"[*] Best checkpoint saved to {save_path} (loss: {best_loss:.4f})")

    print("[Training] Remote training finished successfully.")


if __name__ == "__main__":
    main()
