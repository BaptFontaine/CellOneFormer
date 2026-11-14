"""
Cell morphometry and quantitative biological feature extraction from instance label masks.
Uses scikit-image regionprops without requiring heavy external table dependencies.
"""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.measure import regionprops


def extract_cell_morphometry(
    label_matrix: np.ndarray,
    image: Optional[np.ndarray] = None,
    pixel_size_um: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts per-cell morphological and intensity measurements from an integer instance mask.

    Args:
        label_matrix: 2D integer array (H, W) where 0 is background and >=1 are cell IDs.
        image: Optional 2D or 3D RGB image matching (H, W) for intensity statistics.
        pixel_size_um: Optional physical pixel size in micrometers to compute calibrated areas.

    Returns:
        List of dictionaries, one per detected cell instance.
    """
    if label_matrix.size == 0 or int(label_matrix.max()) == 0:
        return []

    intensity_img = None
    if image is not None:
        if image.ndim == 3:
            intensity_img = image.mean(axis=-1).astype(np.float32)
        else:
            intensity_img = image.astype(np.float32)

    props = regionprops(label_matrix.astype(np.int32), intensity_image=intensity_img)
    rows: List[Dict[str, Any]] = []

    um_sq = (pixel_size_um ** 2) if (pixel_size_um is not None and pixel_size_um > 0) else None

    for prop in props:
        area = int(prop.area)
        perimeter = float(prop.perimeter)
        circularity = min(1.0, (4.0 * math.pi * area) / (perimeter * perimeter)) if perimeter > 0 else 0.0
        major = float(prop.axis_major_length)
        minor = float(prop.axis_minor_length)
        aspect_ratio = major / max(minor, 1e-6) if major > 0 else 1.0
        cy, cx = prop.centroid

        row: Dict[str, Any] = {
            "cell_id": int(prop.label),
            "area_px": area,
            "perimeter_px": round(perimeter, 2),
            "circularity": round(circularity, 4),
            "eccentricity": round(float(prop.eccentricity), 4),
            "solidity": round(float(prop.solidity), 4),
            "major_axis_px": round(major, 2),
            "minor_axis_px": round(minor, 2),
            "aspect_ratio": round(aspect_ratio, 3),
            "centroid_y": round(float(cy), 1),
            "centroid_x": round(float(cx), 1),
        }

        if intensity_img is not None:
            row["mean_intensity"] = round(float(prop.intensity_mean), 2)

        if um_sq is not None:
            area_um2 = area * um_sq
            row["area_um2"] = round(area_um2, 3)
            row["equiv_diameter_um"] = round(2.0 * math.sqrt(area_um2 / math.pi), 3)

        rows.append(row)

    return rows


def summarize_morphometry(
    rows: List[Dict[str, Any]],
    image_shape: Tuple[int, ...],
) -> Dict[str, Any]:
    """Computes image-level summary statistics (cell count, confluence, mean/median area)."""
    h, w = image_shape[:2]
    total_pixels = max(1, h * w)

    if not rows:
        return {
            "cell_count": 0,
            "confluence_pct": 0.0,
            "mean_area_px": 0.0,
            "median_area_px": 0.0,
            "std_area_px": 0.0,
            "mean_circularity": 0.0,
            "mean_eccentricity": 0.0,
            "mean_solidity": 0.0,
        }

    areas = np.array([r["area_px"] for r in rows], dtype=np.float64)
    circs = np.array([r["circularity"] for r in rows], dtype=np.float64)
    eccs = np.array([r["eccentricity"] for r in rows], dtype=np.float64)
    sols = np.array([r["solidity"] for r in rows], dtype=np.float64)

    return {
        "cell_count": len(rows),
        "confluence_pct": round(float(areas.sum() / total_pixels * 100.0), 2),
        "mean_area_px": round(float(areas.mean()), 2),
        "median_area_px": round(float(np.median(areas)), 2),
        "std_area_px": round(float(areas.std()), 2),
        "mean_circularity": round(float(circs.mean()), 4),
        "mean_eccentricity": round(float(eccs.mean()), 4),
        "mean_solidity": round(float(sols.mean()), 4),
    }


def save_morphometry_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Writes morphometric feature rows to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "cell_id", "area_px", "perimeter_px", "circularity",
        "eccentricity", "solidity", "major_axis_px", "minor_axis_px",
        "aspect_ratio", "centroid_y", "centroid_x",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
