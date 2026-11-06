"""
Sliding-window tiling module for high-resolution microscopy images.
Handles multi-channel/grayscale images and integer instance masks (where each cell has a unique ID).
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image

try:
    import tifffile
except ImportError:
    tifffile = None


class SlidingWindowTiler:
    """
    Slices large microscopy images and corresponding instance masks into overlapping tiles.
    """

    def __init__(
        self,
        tile_size: int = 512,
        tile_overlap: int = 64,
        min_cell_area: int = 15,
    ):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.stride = tile_size - tile_overlap
        self.min_cell_area = min_cell_area

        if self.stride <= 0:
            raise ValueError(f"Tile overlap ({tile_overlap}) must be smaller than tile size ({tile_size}).")

    def compute_tile_coords(self, height: int, width: int) -> List[Tuple[int, int, int, int]]:
        """
        Calculates (y1, y2, x1, x2) bounding coordinates for all tiles covering (height, width).
        Ensures the entire image is covered without clipping edge cells.
        """
        y_starts = list(range(0, max(1, height - self.tile_size + 1), self.stride))
        if y_starts[-1] + self.tile_size < height:
            y_starts.append(height - self.tile_size)

        x_starts = list(range(0, max(1, width - self.tile_size + 1), self.stride))
        if x_starts[-1] + self.tile_size < width:
            x_starts.append(width - self.tile_size)

        coords = []
        for y1 in y_starts:
            y2 = min(height, y1 + self.tile_size)
            for x1 in x_starts:
                x2 = min(width, x1 + self.tile_size)
                coords.append((y1, y2, x1, x2))
        return coords

    def tile_image_and_mask(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Crops an image (and optional mask) into tiles.
        
        Args:
            image: numpy array of shape (H, W) or (H, W, C).
            mask: optional integer instance mask of shape (H, W), where 0 is background,
                  and each cell has a unique positive integer ID.

        Returns:
            List of dicts containing:
                - 'image': tiled image crop (H_tile, W_tile, C) normalized/uint8
                - 'mask': tiled instance mask (H_tile, W_tile) or None
                - 'coords': (y1, y2, x1, x2)
                - 'num_cells': count of valid cells in this tile
        """
        h, w = image.shape[:2]

        # Ensure image is 3-channel for transformer backbones
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        # Pad image and mask if smaller than tile_size
        pad_y = max(0, self.tile_size - h)
        pad_x = max(0, self.tile_size - w)
        if pad_y > 0 or pad_x > 0:
            image = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect")
            if mask is not None:
                mask = np.pad(mask, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
            h, w = image.shape[:2]

        coords = self.compute_tile_coords(h, w)
        tiles = []

        for y1, y2, x1, x2 in coords:
            img_crop = image[y1:y2, x1:x2].copy()
            mask_crop = None
            cell_count = 0

            if mask is not None:
                raw_mask_crop = mask[y1:y2, x1:x2].copy()
                remapped_mask = np.zeros_like(raw_mask_crop, dtype=np.int32)

                # Re-index cells in this crop to 1..K and filter tiny remnants
                unique_ids = np.unique(raw_mask_crop)
                new_id = 1
                for uid in unique_ids:
                    if uid == 0:
                        continue
                    cell_pixels = (raw_mask_crop == uid)
                    if np.sum(cell_pixels) >= self.min_cell_area:
                        remapped_mask[cell_pixels] = new_id
                        new_id += 1

                mask_crop = remapped_mask
                cell_count = new_id - 1

            tiles.append({
                "image": img_crop,
                "mask": mask_crop,
                "coords": (int(y1), int(y2), int(x1), int(x2)),
                "num_cells": cell_count,
            })

        return tiles


def load_microscopy_image(path: str) -> np.ndarray:
    """Reads a microscopy image from PNG, JPG, or TIFF format, normalized to uint8 0-255."""
    path = str(path)
    if path.lower().endswith((".tif", ".tiff")) and tifffile is not None:
        img = tifffile.imread(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Failed to read image at {path}")

    # Handle float or 16-bit TIFF normalization
    if img.dtype == np.uint16:
        img = (img / 256.0).astype(np.uint8)
    elif np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)

    return img


def load_instance_mask(path: str) -> np.ndarray:
    """Loads an integer instance mask from TIFF or PNG."""
    path = str(path)
    if path.lower().endswith((".tif", ".tiff")) and tifffile is not None:
        mask = tifffile.imread(path)
    else:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise FileNotFoundError(f"Failed to read mask at {path}")

    if mask.ndim == 3:
        # If saved as RGB, reduce to single 2D channel
        mask = mask[..., 0]

    return mask.astype(np.int32)


def generate_synthetic_cell_data(
    output_dir: str,
    num_samples: int = 5,
    height: int = 1024,
    width: int = 1024,
) -> None:
    """
    Generates synthetic multi-modal microscopy images and ground truth masks
    for smoke testing and pipeline verification when raw data is not yet downloaded.
    """
    raw_images_dir = Path(output_dir) / "images"
    raw_masks_dir = Path(output_dir) / "masks"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    raw_masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Tiler] Generating {num_samples} synthetic microscopy images in {output_dir}...")

    for i in range(num_samples):
        # Create a synthetic noisy background (fluorescence or phase-contrast style)
        bg = np.random.normal(loc=30, scale=10, size=(height, width)).clip(0, 255).astype(np.uint8)
        mask = np.zeros((height, width), dtype=np.int32)

        # Draw 40-100 random elliptical cells
        num_cells = np.random.randint(40, 100)
        for cell_id in range(1, num_cells + 1):
            cx = np.random.randint(50, width - 50)
            cy = np.random.randint(50, height - 50)
            axes = (np.random.randint(15, 35), np.random.randint(10, 25))
            angle = np.random.randint(0, 180)

            # Draw cell instance in mask
            cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, int(cell_id), -1)

            # Draw bright cell body and halo on synthetic image
            cv2.ellipse(bg, (cx, cy), axes, angle, 0, 360, int(np.random.randint(140, 230)), -1)
            cv2.ellipse(bg, (cx, cy), (axes[0] + 3, axes[1] + 3), angle, 0, 360, int(np.random.randint(70, 120)), 2)

        img_rgb = np.stack([bg] * 3, axis=-1)

        cv2.imwrite(str(raw_images_dir / f"synthetic_sample_{i:03d}.png"), img_rgb)
        cv2.imwrite(str(raw_masks_dir / f"synthetic_sample_{i:03d}_mask.png"), mask)

    print(f"[Tiler] Synthetic dataset created successfully in {output_dir}")
