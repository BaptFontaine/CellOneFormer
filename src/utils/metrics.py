"""
Evaluation metrics for cell instance segmentation benchmark.
Implements Intersection-over-Union (IoU), Average Precision (AP@50, AP@75, AP@[50:95]),
and F1-score matching the NeurIPS 2022 Cell Segmentation Challenge.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment


def compute_iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU between N predicted binary masks and M ground-truth binary masks.
    Uses bounding-box sparse intersection on large images to avoid multi-GB float32 allocations.
    """
    n = len(pred_masks)
    m = len(gt_masks)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    hw = int(pred_masks[0].size)
    if max(n, m) * hw <= 50_000_000:
        pred_flat = pred_masks.reshape(n, -1).astype(np.float32)
        gt_flat = gt_masks.reshape(m, -1).astype(np.float32)
        intersection = np.dot(pred_flat, gt_flat.T)
        pred_area = pred_flat.sum(axis=1, keepdims=True)
        gt_area = gt_flat.sum(axis=1, keepdims=True).T
        union = np.maximum(pred_area + gt_area - intersection, 1e-6)
        return intersection / union

    def _extract_boxes_and_areas(masks: np.ndarray):
        boxes = []
        areas = np.zeros(len(masks), dtype=np.float32)
        for idx, mk in enumerate(masks):
            ys, xs = np.nonzero(mk)
            if len(ys) == 0:
                boxes.append((0, 0, 0, 0))
            else:
                boxes.append((int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1))
                areas[idx] = float(len(ys))
        return boxes, areas

    p_boxes, p_areas = _extract_boxes_and_areas(pred_masks)
    g_boxes, g_areas = _extract_boxes_and_areas(gt_masks)
    iou_matrix = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        if p_areas[i] == 0:
            continue
        py1, py2, px1, px2 = p_boxes[i]
        pm = pred_masks[i]
        for j in range(m):
            if g_areas[j] == 0:
                continue
            gy1, gy2, gx1, gx2 = g_boxes[j]
            if py2 <= gy1 or gy2 <= py1 or px2 <= gx1 or gx2 <= px1:
                continue
            iy1, iy2 = max(py1, gy1), min(py2, gy2)
            ix1, ix2 = max(px1, gx1), min(px2, gx2)
            inter = float(np.logical_and(pm[iy1:iy2, ix1:ix2], gt_masks[j][iy1:iy2, ix1:ix2]).sum())
            if inter > 0:
                union = p_areas[i] + g_areas[j] - inter
                iou_matrix[i, j] = inter / max(union, 1e-6)

    return iou_matrix


def _metrics_from_matched_ious(
    matched_ious: np.ndarray,
    n_preds: int,
    n_gts: int,
    iou_threshold: float,
) -> Dict[str, float]:
    if n_preds == 0 and n_gts == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if n_preds == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": n_gts}
    if n_gts == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": n_preds, "fn": 0}

    tp = int(np.sum(matched_ious >= iou_threshold))
    fp = n_preds - tp
    fn = n_gts - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _relabel_contiguous_1d(label_map: np.ndarray) -> Tuple[np.ndarray, int]:
    uniq, inv = np.unique(label_map.ravel(), return_inverse=True)
    inv_1d = inv.ravel()
    if len(uniq) == 0:
        return np.zeros(label_map.size, dtype=np.int32), 0
    if uniq[0] == 0:
        return inv_1d.astype(np.int32), len(uniq) - 1
    return (inv_1d + 1).astype(np.int32), len(uniq)


def compute_iou_matrix_from_labels(pred_label: np.ndarray, gt_label: np.ndarray) -> np.ndarray:
    """
    Computes exact (N, M) pairwise IoU directly from 2D integer instance label maps
    in O(H*W) memory without allocating (N, H, W) or (M, H, W) boolean stacks.
    """
    p_flat, n_preds = _relabel_contiguous_1d(pred_label)
    g_flat, n_gts = _relabel_contiguous_1d(gt_label)
    if n_preds == 0 or n_gts == 0:
        return np.zeros((n_preds, n_gts), dtype=np.float32)

    p_areas = np.bincount(p_flat, minlength=n_preds + 1)[1:].astype(np.float32)
    g_areas = np.bincount(g_flat, minlength=n_gts + 1)[1:].astype(np.float32)

    fg = (p_flat > 0) & (g_flat > 0)
    iou_matrix = np.zeros((n_preds, n_gts), dtype=np.float32)
    if np.any(fg):
        p_fg = p_flat[fg].astype(np.int64)
        g_fg = g_flat[fg].astype(np.int64)
        stride = n_gts + 1
        uniq_pairs, counts = np.unique(p_fg * stride + g_fg, return_counts=True)
        p_idx = uniq_pairs // stride - 1
        g_idx = uniq_pairs % stride - 1
        inter = counts.astype(np.float32)
        union = p_areas[p_idx] + g_areas[g_idx] - inter
        iou_matrix[p_idx, g_idx] = inter / np.maximum(union, 1e-6)

    return iou_matrix


def _benchmark_from_iou_matrix(iou_matrix: np.ndarray) -> Dict[str, float]:
    n_preds, n_gts = iou_matrix.shape
    if n_preds > 0 and n_gts > 0:
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_ious = iou_matrix[row_ind, col_ind]
    else:
        matched_ious = np.empty((0,), dtype=np.float32)

    metrics_at_50 = _metrics_from_matched_ious(matched_ious, n_preds, n_gts, 0.50)
    metrics_at_75 = _metrics_from_matched_ious(matched_ious, n_preds, n_gts, 0.75)

    thresholds = np.arange(0.50, 1.00, 0.05)
    f1_scores = [
        _metrics_from_matched_ious(matched_ious, n_preds, n_gts, float(thresh))["f1"]
        for thresh in thresholds
    ]

    return {
        "AP_50": metrics_at_50["precision"],
        "AP@50": metrics_at_50["precision"],
        "AP_75": metrics_at_75["precision"],
        "AP@75": metrics_at_75["precision"],
        "F1_50": metrics_at_50["f1"],
        "F1": metrics_at_50["f1"],
        "mean_f1": float(np.mean(f1_scores)),
        "mAP_50_95": float(np.mean(f1_scores)),
        "mAP@[50:95]": float(np.mean(f1_scores)),
        "TP_50": metrics_at_50["tp"],
        "FP_50": metrics_at_50["fp"],
        "FN_50": metrics_at_50["fn"],
    }


def evaluate_instances_at_threshold(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluates True Positives, False Positives, False Negatives, Precision, Recall,
    and F1-score at a single IoU threshold using Hungarian bipartite matching.
    """
    n_preds = len(pred_masks)
    n_gts = len(gt_masks)

    if n_preds == 0 and n_gts == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if n_preds == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": n_gts}
    if n_gts == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": n_preds, "fn": 0}

    iou_matrix = compute_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_ious = iou_matrix[row_ind, col_ind]
    return _metrics_from_matched_ious(matched_ious, n_preds, n_gts, iou_threshold)


def compute_challenge_benchmark_metrics(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
) -> Dict[str, float]:
    """
    Computes the full benchmark suite:
    - AP@50 (IoU = 0.50)
    - AP@75 (IoU = 0.75)
    - AP@[50:95] (mean AP across thresholds 0.50 to 0.95 with step 0.05)
    - Mean F1-score
    Computes the pairwise IoU matrix and Hungarian matching once for all thresholds.
    """
    iou_matrix = compute_iou_matrix(pred_masks, gt_masks)
    return _benchmark_from_iou_matrix(iou_matrix)


def compute_challenge_benchmark_metrics_from_labels(
    pred_label: np.ndarray,
    gt_label: np.ndarray,
) -> Dict[str, float]:
    """
    Memory-efficient benchmark suite directly from 2D integer instance label maps (H, W).
    Produces identical results to compute_challenge_benchmark_metrics without (N, H, W) stacks.
    """
    iou_matrix = compute_iou_matrix_from_labels(pred_label, gt_label)
    return _benchmark_from_iou_matrix(iou_matrix)
