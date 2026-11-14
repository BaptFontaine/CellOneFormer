"""
Interactive Gradio web application for CellOneFormer.
Allows uploading microscopy images (PNG/JPG/TIFF), tuning inference parameters,
visualizing segmented cell instances, and exporting 16-bit masks + morphometry CSVs.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError(
        "Gradio is required to run app.py. Install it with: pip install gradio"
    ) from exc

from src.dataset.tiler import load_microscopy_image
from src.inference.predict import create_overlay, load_trained_model, predict_full_image
from src.utils.morphometry import (
    extract_cell_morphometry,
    save_morphometry_csv,
    summarize_morphometry,
)

DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "configs" / "config.yaml")
DEFAULT_WEIGHTS_CANDIDATES = [
    PROJECT_ROOT / "checkpoints" / "oneformer_real_cells_best.pt",
    PROJECT_ROOT / "checkpoints" / "oneformer_cells_best.pt",
]

_MODEL_CACHE = {}


def _get_default_weights() -> str:
    for p in DEFAULT_WEIGHTS_CANDIDATES:
        if p.exists():
            return str(p)
    return str(DEFAULT_WEIGHTS_CANDIDATES[0])


def _get_model_and_cfg(weights_path: str):
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (weights_path, str(device))
    if cache_key not in _MODEL_CACHE:
        model = load_trained_model(cfg, weights_path if os.path.exists(weights_path) else None, device)
        _MODEL_CACHE[cache_key] = model
    return _MODEL_CACHE[cache_key], cfg, device


def run_segmentation(
    image_path: Optional[str],
    weights_path: str,
    score_threshold: float,
    stitch_iou: float,
    min_area: int,
    max_area: int,
    tile_overlap: int,
    apply_clahe: bool,
    use_tta: bool,
    pixel_size_um: float,
) -> Tuple[Optional[np.ndarray], str, list, Optional[str], Optional[str]]:
    if not image_path:
        return None, "Veuillez importer une image de microscopie.", [], None, None

    model, cfg, device = _get_model_and_cfg(weights_path)
    percentile_norm = cfg["data"].get("percentile_norm", True)
    image = load_microscopy_image(image_path, percentile_norm=percentile_norm, apply_clahe=apply_clahe)

    label_matrix, _ = predict_full_image(
        image=image,
        model_module=model,
        cfg=cfg,
        device=device,
        score_thresh=score_threshold,
        stitch_iou=stitch_iou,
        min_area=int(min_area),
        max_area=int(max_area),
        tile_overlap=int(tile_overlap),
        use_tta=use_tta,
    )

    overlay = create_overlay(image, label_matrix)
    um = pixel_size_um if pixel_size_um > 0 else None
    rows = extract_cell_morphometry(label_matrix, image=image, pixel_size_um=um)
    summary = summarize_morphometry(rows, image.shape)

    tmp_dir = Path(tempfile.mkdtemp(prefix="celloneformer_"))
    stem = Path(image_path).stem
    mask_out = tmp_dir / f"{stem}_pred_mask.png"
    csv_out = tmp_dir / f"{stem}_cells.csv"
    cv2.imwrite(str(mask_out), label_matrix.astype(np.uint16))
    save_morphometry_csv(rows, str(csv_out))

    summary_md = (
        f"### Résultats ({device.type.upper()})\n"
        f"- **Cellules détectées** : `{summary['cell_count']}`\n"
        f"- **Confluence (couverture)** : `{summary['confluence_pct']} %`\n"
        f"- **Aire moyenne / médiane** : `{summary['mean_area_px']} px` / `{summary['median_area_px']} px`\n"
        f"- **Circularité moyenne** : `{summary['mean_circularity']}`\n"
        f"- **Solidité moyenne** : `{summary['mean_solidity']}`"
    )

    table_headers = [
        "cell_id", "area_px", "perimeter_px", "circularity",
        "eccentricity", "solidity", "aspect_ratio",
    ]
    table_rows = [[r[k] for k in table_headers] for r in rows[:100]]
    return overlay, summary_md, table_rows, str(mask_out), str(csv_out)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="CellOneFormer — Segmentation Cellulaire Multi-Modalité") as demo:
        gr.Markdown(
            "# CellOneFormer — Segmentation & Morphométrie Cellulaire\n"
            "Importez une image de microscopie (`.png`, `.jpg`, `.tif`, `.tiff`) pour détecter les cellules, "
            "exporter le masque d'instances 16-bit et obtenir les statistiques morphométriques."
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="filepath", label="Image microscopique (PNG / TIFF)")
                weights_input = gr.Textbox(value=_get_default_weights(), label="Chemin des poids (.pt)")
                score_sl = gr.Slider(0.05, 0.90, value=0.25, step=0.05, label="Seuil de confiance (score_threshold)")
                iou_sl = gr.Slider(0.10, 0.90, value=0.50, step=0.05, label="Seuil Soft-IoU Stitching")
                with gr.Row():
                    min_area_num = gr.Number(value=15, label="Aire min (px)", precision=0)
                    max_area_num = gr.Number(value=8000, label="Aire max (px)", precision=0)
                overlap_sl = gr.Slider(32, 192, value=96, step=16, label="Recouvrement des tuiles (px)")
                pixel_um_num = gr.Number(value=0.0, label="Taille pixel (µm, 0 = désactivé)")
                with gr.Row():
                    clahe_chk = gr.Checkbox(value=False, label="Appliquer CLAHE")
                    tta_chk = gr.Checkbox(value=False, label="Test-Time Augmentation (TTA)")
                run_btn = gr.Button("Lancer la segmentation", variant="primary")

            with gr.Column(scale=2):
                overlay_out = gr.Image(type="numpy", label="Instances segmentées (Overlay)")
                summary_out = gr.Markdown("En attente d'une image...")
                with gr.Row():
                    mask_dl = gr.File(label="Télécharger masque 16-bit (.png)")
                    csv_dl = gr.File(label="Télécharger morphométrie (.csv)")
                table_out = gr.Dataframe(
                    headers=[
                        "cell_id", "area_px", "perimeter_px", "circularity",
                        "eccentricity", "solidity", "aspect_ratio",
                    ],
                    label="Aperçu des cellules détectées (100 premières)",
                    interactive=False,
                )

        run_btn.click(
            fn=run_segmentation,
            inputs=[
                img_input,
                weights_input,
                score_sl,
                iou_sl,
                min_area_num,
                max_area_num,
                overlap_sl,
                clahe_chk,
                tta_chk,
                pixel_um_num,
            ],
            outputs=[overlay_out, summary_out, table_out, mask_dl, csv_dl],
        )
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch CellOneFormer Gradio App")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (auto-enabled on Colab)")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (use 0.0.0.0 for remote/Docker)")
    cli_args = parser.parse_args()

    in_colab = ("google.colab" in sys.modules) or os.path.exists("/content")
    app = build_app()
    app.launch(
        share=cli_args.share or in_colab,
        server_name=cli_args.host,
        server_port=cli_args.port,
    )
