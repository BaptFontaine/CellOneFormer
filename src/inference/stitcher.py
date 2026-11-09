"""
Tile stitching and mask deduplication module.
Reconstructs full-field microscopy images from overlapping tile predictions
using soft-IoU suppression across seams.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


class TileStitcher:
    """
    Assembles predicted instance masks from sliding-window tiles back into a single full-resolution image.
    """

    def __init__(self, iou_threshold: float = 0.5, min_area: int = 15, max_area: Optional[int] = 8000):
        self.iou_threshold = iou_threshold
        self.min_area = min_area
        self.max_area = max_area

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
                - 'masks': List of boolean 2D masks
                - 'scores': List of float confidence scores
                - 'coords': (y1, y2, x1, x2) in original image coordinates
        Returns:
            label_matrix: (full_height, full_width) int32 map where each cell has a unique integer ID >= 1
            kept_masks: (N_cells, full_height, full_width) boolean array of deduplicated instances
        """
        all_global_masks = []
        all_scores = []

        # 1. Project local tile masks into global canvas
        for res in tile_results:
            y1, y2, x1, x2 = res["coords"]
            masks = res.get("masks", [])
            scores = res.get("scores", [])

            target_h = y2 - y1
            target_w = x2 - x1

            for idx, mask in enumerate(masks):
                score = float(scores[idx]) if len(scores) > idx else 1.0
                
                # Check area (filter spurious noise < min_area and giant background masks > max_area)
                mask_area = int(np.sum(mask))
                if mask_area < self.min_area or (self.max_area is not None and mask_area > self.max_area):
                    continue

                # Ensure mask is exactly target_h x target_w (handles any resolution difference)
                if mask.shape[0] != target_h or mask.shape[1] != target_w:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (target_w, target_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                canvas = np.zeros((full_height, full_width), dtype=bool)
                canvas[y1:y2, x1:x2] = mask[:target_h, :target_w]
                all_global_masks.append(canvas)
                all_scores.append(score)

        if len(all_global_masks) == 0:
            return np.zeros((full_height, full_width), dtype=np.int32), np.empty((0, full_height, full_width), dtype=bool)

        all_global_masks = np.array(all_global_masks)
        all_scores = np.array(all_scores)

        # 2. Sort by confidence descending
        order = np.argsort(-all_scores)
        keep_indices = []

        # 3. Soft-IoU suppression for overlapping duplicate cells across tile seams
        suppressed = np.zeros(len(order), dtype=bool)

        for i in range(len(order)):
            idx_i = order[i]
            if suppressed[idx_i]:
                continue

            keep_indices.append(idx_i)
            mask_i = all_global_masks[idx_i]

            for j in range(i + 1, len(order)):
                idx_j = order[j]
                if suppressed[idx_j]:
                    continue

                mask_j = all_global_masks[idx_j]

                # Bounding-box intersection fast-check
                intersection = np.logical_and(mask_i, mask_j).sum()
                if intersection == 0:
                    continue

                union = np.logical_or(mask_i, mask_j).sum()
                iou = intersection / union if union > 0 else 0.0

                if iou >= self.iou_threshold:
                    suppressed[idx_j] = True

        kept_masks = all_global_masks[keep_indices]

        # 4. Construct integer label matrix for challenge submission / export
        label_matrix = np.zeros((full_height, full_width), dtype=np.int32)
        for cell_id, mask in enumerate(kept_masks, 1):
            label_matrix[mask] = cell_id

        return label_matrix, kept_masks
