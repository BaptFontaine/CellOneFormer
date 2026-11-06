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
    
    Args:
        pred_masks: shape (N, H, W) of boolean / binary arrays
        gt_masks: shape (M, H, W) of boolean / binary arrays
        
    Returns:
        iou_matrix: shape (N, M)
    """
    n = len(pred_masks)
    m = len(gt_masks)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    pred_flat = pred_masks.reshape(n, -1).astype(np.float32)
    gt_flat = gt_masks.reshape(m, -1).astype(np.float32)

    # Intersection: N x M
    intersection = np.dot(pred_flat, gt_flat.T)

    # Areas
    pred_area = pred_flat.sum(axis=1, keepdims=True)  # (N, 1)
    gt_area = gt_flat.sum(axis=1, keepdims=True).T    # (1, M)

    union = pred_area + gt_area - intersection
    union = np.maximum(union, 1e-6)

    return intersection / union


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

    # Hungarian matching for maximum total IoU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_ious = iou_matrix[row_ind, col_ind]
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
    """
    thresholds = np.arange(0.50, 1.00, 0.05)
    f1_scores = []

    metrics_at_50 = evaluate_instances_at_threshold(pred_masks, gt_masks, 0.50)
    metrics_at_75 = evaluate_instances_at_threshold(pred_masks, gt_masks, 0.75)

    for thresh in thresholds:
        res = evaluate_instances_at_threshold(pred_masks, gt_masks, float(thresh))
        f1_scores.append(res["f1"])

    return {
        "AP_50": metrics_at_50["precision"],
        "AP_75": metrics_at_75["precision"],
        "F1_50": metrics_at_50["f1"],
        "mAP_50_95": float(np.mean(f1_scores)),
        "TP_50": metrics_at_50["tp"],
        "FP_50": metrics_at_50["fp"],
        "FN_50": metrics_at_50["fn"],
    }
