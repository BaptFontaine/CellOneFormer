"""
Tile stitching and mask deduplication module.
Reconstructs full-field microscopy images from overlapping tile predictions
using soft-IoU suppression across seams.
"""

import numpy as np
from typing import Dict, List, Tuple


class TileStitcher:
    """
    Assembles predicted instance masks from sliding-window tiles back into a single full-resolution image.
    """

    def __init__(self, iou_threshold: float = 0.5, min_area: int = 15):
        self.iou_threshold = iou_threshold
        self.min_area = min_area

    def stitch_tile_predictions(
        self,
        full_height: int,
        full_width: int,
        tile_results: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            full_height: height of original microscopy image
            full_width: width of original microscopy image
            tile_results: list of dicts, each with:
                - 'masks': binary masks list or array of shape (K, H_tile, W_tile)
                - 'scores': list/array of confidence scores for each mask
                - 'coords': (y1, y2, x1, x2) in original image space

        Returns:
            label_matrix: 2D integer array (H, W) where 0 is background, 1..N are unique cells
            global_masks: 3D boolean array (N, H, W)
        """
        all_global_masks = []
        all_scores = []

        # 1. Project local tile masks into global canvas
        for res in tile_results:
            y1, y2, x1, x2 = res["coords"]
            masks = res.get("masks", [])
            scores = res.get("scores", [])

            for idx, mask in enumerate(masks):
                score = float(scores[idx]) if len(scores) > idx else 1.0
                
                # Check area
                if np.sum(mask) < self.min_area:
                    continue

                target_h = y2 - y1
                target_w = x2 - x1
                crop_mask = mask[:target_h, :target_w]

                canvas = np.zeros((full_height, full_width), dtype=bool)
                canvas[y1:y2, x1:x2] = crop_mask.astype(bool)
                all_global_masks.append(canvas)
                all_scores.append(score)

        if len(all_global_masks) == 0:
            return np.zeros((full_height, full_width), dtype=np.int32), np.empty((0, full_height, full_width), dtype=bool)

        all_global_masks = np.array(all_global_masks)
        all_scores = np.array(all_scores)

        # 2. Sort by confidence descending
        order = np.argsort(-all_scores)
        keep_indices = []

        # 3. Soft-IoU suppression for overlapping duplicate cells
        suppressed = np.zeros(len(order), dtype=bool)

        for i in range(len(order)):
            idx_i = order[i]
            if suppressed[idx_i]:
                continue

            keep_indices.append(idx_i)
            mask_i = all_global_masks[idx_i]
            area_i = np.sum(mask_i)

            for j in range(i + 1, len(order)):
                idx_j = order[j]
                if suppressed[idx_j]:
                    continue

                mask_j = all_global_masks[idx_j]
                intersection = np.sum(mask_i & mask_j)

                if intersection > 0:
                    union = area_i + np.sum(mask_j) - intersection
                    iou = intersection / max(union, 1e-6)

                    # If IoU exceeds threshold, suppress redundant detection
                    if iou >= self.iou_threshold:
                        suppressed[idx_j] = True

        kept_masks = all_global_masks[keep_indices]

        # 4. Create final integer instance map (1..N)
        label_matrix = np.zeros((full_height, full_width), dtype=np.int32)
        for cell_id, mask in enumerate(kept_masks, start=1):
            label_matrix[mask] = cell_id

        return label_matrix, kept_masks
