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
    Supports Gaussian center-weighting at tile seams and fast bounding-box IoU suppression.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        min_area: int = 15,
        max_area: Optional[int] = 8000,
        gaussian_seam_weighting: bool = True,
    ):
        self.iou_threshold = iou_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.gaussian_seam_weighting = gaussian_seam_weighting

    @staticmethod
    def _compute_center_weight(mask: np.ndarray, tile_h: int, tile_w: int) -> float:
        """
        Computes a spatial confidence multiplier in [0.65, 1.0] that favors masks near the
        center of a tile over masks truncated at tile borders.
        """
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return 0.65
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        dy = (cy - tile_h / 2.0) / max(1.0, tile_h / 2.0)
        dx = (cx - tile_w / 2.0) / max(1.0, tile_w / 2.0)
        dist_sq = dy * dy + dx * dx
        return float(0.65 + 0.35 * np.exp(-1.5 * dist_sq))

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
        all_bboxes = []
        all_areas = []
        all_scores = []

        # 1. Project local tile masks into global canvas
        for res in tile_results:
            y1, y2, x1, x2 = res["coords"]
            masks = res.get("masks", [])
            scores = res.get("scores", [])

            target_h = y2 - y1
            target_w = x2 - x1
            y1_v, y2_v = max(0, y1), min(full_height, y2)
            x1_v, x2_v = max(0, x1), min(full_width, x2)
            valid_h, valid_w = y2_v - y1_v, x2_v - x1_v
            if valid_h <= 0 or valid_w <= 0:
                continue

            for idx, mask in enumerate(masks):
                score = float(scores[idx]) if len(scores) > idx else 1.0

                # Ensure mask is exactly target_h x target_w (handles any resolution difference)
                if mask.shape[0] != target_h or mask.shape[1] != target_w:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (target_w, target_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    mask = mask.astype(bool)

                cropped_mask = mask[:valid_h, :valid_w]
                mask_area = int(np.sum(cropped_mask))
                if mask_area < self.min_area or (self.max_area is not None and mask_area > self.max_area):
                    continue

                if self.gaussian_seam_weighting:
                    score *= self._compute_center_weight(cropped_mask, valid_h, valid_w)

                ys, xs = np.nonzero(cropped_mask)
                by1, by2 = int(ys.min()) + y1_v, int(ys.max()) + y1_v + 1
                bx1, bx2 = int(xs.min()) + x1_v, int(xs.max()) + x1_v + 1

                canvas = np.zeros((full_height, full_width), dtype=bool)
                canvas[y1_v:y2_v, x1_v:x2_v] = cropped_mask

                all_global_masks.append(canvas)
                all_bboxes.append((by1, by2, bx1, bx2))
                all_areas.append(mask_area)
                all_scores.append(score)

        if len(all_global_masks) == 0:
            return np.zeros((full_height, full_width), dtype=np.int32), np.empty((0, full_height, full_width), dtype=bool)

        all_global_masks = np.array(all_global_masks)
        all_scores = np.array(all_scores)

        # 2. Sort by confidence descending
        order = np.argsort(-all_scores)
        keep_indices = []

        # 3. Fast Bounding-Box + Soft-IoU suppression across tile seams
        suppressed = np.zeros(len(order), dtype=bool)

        for i in range(len(order)):
            idx_i = order[i]
            if suppressed[idx_i]:
                continue

            keep_indices.append(idx_i)
            mask_i = all_global_masks[idx_i]
            by1_i, by2_i, bx1_i, bx2_i = all_bboxes[idx_i]
            area_i = all_areas[idx_i]

            for j in range(i + 1, len(order)):
                idx_j = order[j]
                if suppressed[idx_j]:
                    continue

                by1_j, by2_j, bx1_j, bx2_j = all_bboxes[idx_j]

                # Fast bounding-box disjoint check
                if by2_i <= by1_j or by2_j <= by1_i or bx2_i <= bx1_j or bx2_j <= bx1_i:
                    continue

                # Compute intersection only within overlapping bounding box
                iy1, iy2 = max(by1_i, by1_j), min(by2_i, by2_j)
                ix1, ix2 = max(bx1_i, bx1_j), min(bx2_i, bx2_j)

                mask_j = all_global_masks[idx_j]
                intersection = int(np.logical_and(mask_i[iy1:iy2, ix1:ix2], mask_j[iy1:iy2, ix1:ix2]).sum())
                if intersection == 0:
                    continue

                area_j = all_areas[idx_j]
                union = area_i + area_j - intersection
                iou = intersection / union if union > 0 else 0.0

                if iou >= self.iou_threshold:
                    suppressed[idx_j] = True

        kept_masks = all_global_masks[keep_indices]

        # 4. Construct integer label matrix (painters order from lowest to highest score so top cells stay sharp)
        label_matrix = np.zeros((full_height, full_width), dtype=np.int32)
        for cell_id, mask in enumerate(kept_masks, 1):
            label_matrix[mask] = cell_id

        return label_matrix, kept_masks
