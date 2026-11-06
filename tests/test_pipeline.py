"""
Unit tests for tiler, metrics, and stitcher modules.
"""

import numpy as np
import pytest

from src.dataset.tiler import SlidingWindowTiler
from src.utils.metrics import compute_iou_matrix, evaluate_instances_at_threshold
from src.inference.stitcher import TileStitcher


def test_tiler_coords_coverage():
    tiler = SlidingWindowTiler(tile_size=512, tile_overlap=64)
    h, w = 1200, 1600
    coords = tiler.compute_tile_coords(h, w)

    assert len(coords) > 0
    # Check that all coordinates are within image boundaries
    for y1, y2, x1, x2 in coords:
        assert 0 <= y1 < y2 <= h
        assert 0 <= x1 < x2 <= w
        assert (y2 - y1) == 512
        assert (x2 - x1) == 512


def test_iou_matrix():
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[10:30, 10:30] = True

    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[10:30, 10:30] = True  # Exact duplicate

    mask3 = np.zeros((100, 100), dtype=bool)
    mask3[50:70, 50:70] = True  # Disjoint

    ious = compute_iou_matrix(np.array([mask1]), np.array([mask2, mask3]))
    assert np.isclose(ious[0, 0], 1.0)
    assert np.isclose(ious[0, 1], 0.0)


def test_stitcher_deduplication():
    stitcher = TileStitcher(iou_threshold=0.5, min_area=5)

    # Simulate two overlapping tiles detecting the exact same cell
    tile_results = [
        {
            "masks": [np.ones((50, 50), dtype=bool)],
            "scores": [0.95],
            "coords": (0, 100, 0, 100),
        },
        {
            "masks": [np.ones((50, 50), dtype=bool)],
            "scores": [0.85],
            "coords": (0, 100, 0, 100),
        },
    ]

    label_matrix, kept_masks = stitcher.stitch_tile_predictions(200, 200, tile_results)
    # The duplicate should be deduplicated, leaving exactly 1 cell
    assert len(kept_masks) == 1
    assert int(label_matrix.max()) == 1
