# Multi-Modality Cell Segmentation with OneFormer

[![CI Pipeline](https://github.com/<your-username>/cell-segmentation-oneformer/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/cell-segmentation-oneformer/actions)
[![Docker Image](https://img.shields.io/badge/docker-ready-blue.svg)](docker/Dockerfile)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-OneFormer-yellow)](https://huggingface.co/docs/transformers/model_doc/oneformer)

Implementation of a universal Transformer-based instance segmentation pipeline using **OneFormer** for the benchmark:
> **"The Multi-modality Cell Segmentation Challenge: Towards Universal Solutions"**  
> *Jun Ma et al., Nature Methods 21, 1103–1113 (2024) / NeurIPS 2022 Benchmark.*  
> arXiv: [2308.05864](https://arxiv.org/abs/2308.05864)

---

## 🔬 Project Overview

Segmenting individual cell boundaries across diverse microscopy modalities (**Brightfield, Fluorescence, Phase-Contrast, and DIC**) presents severe challenges:
1. **Multi-modality distribution shift** (varying illumination, halos, stains, and contrasts).
2. **Extreme cell density** (hundreds to thousands of touching cells per field).
3. **Query limitation of mask transformers** (standard OneFormer uses 100–150 queries, while an uncropped microscope field contains far more instances).

### Our Architectural Solution:
* **Tiling with Overlap (`SlidingWindowTiler`)**: High-resolution fields ($1024 \times 1024$ to $4096 \times 4096$) are sliced into $512 \times 512$ patches with a 64 px overlap. This keeps cell counts within OneFormer's query capacity.
* **Universal OneFormer Backbone**: Multi-scale attention (Swin / DiNAT) pre-trained on diverse domains, fine-tuned specifically for cell instance segmentation (`the task is instance`).
* **Tile Stitching (`TileStitcher`)**: Assembles local predictions into the full field, resolving boundary conflicts with **Soft-IoU non-maximum suppression**.

---

## 📁 Repository Structure

```
.
├── .github/workflows/ci.yml       # Continuous integration & test checks
├── configs/
│   └── config.yaml                # Centralized hyperparameters & paths
├── data/
│   ├── raw/                       # Place raw challenge dataset here (TIF/PNG)
│   └── processed/                 # Generated tiled crops (train/val)
├── docker/
│   ├── Dockerfile                 # GPU-ready CUDA 12.1 + PyTorch environment
│   └── docker-compose.yml         # Simple single-command container launch
├── notebooks/
│   └── remote_training_colab.ipynb# Free GPU training notebook for Colab / Kaggle
├── src/
│   ├── dataset/
│   │   ├── tiler.py               # Sliding window tiling & synthetic generator
│   │   └── dataset_loader.py      # PyTorch Dataset & OneFormer collator
│   ├── models/
│   │   └── oneformer_wrapper.py   # OneFormer module & query setup
│   ├── training/
│   │   └── train.py               # Training script (supports --dry-run)
│   ├── inference/
│   │   ├── predict.py             # Full-field inference with stitching
│   │   └── stitcher.py            # Overlap resolution & Soft-IoU suppression
│   └── utils/
│       └── metrics.py             # AP@50, AP@75, AP@[50:95], F1 / SEG benchmark
├── tests/
│   └── test_pipeline.py           # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start & Pipeline Verification (Local Machine)

No GPU is needed on your local machine. You can verify that the data pipeline, tiling, and model definitions are valid using the `--dry-run` flag:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests
pytest tests/

# 3. Run dry-run verification (creates synthetic sample and verifies tensors)
python src/training/train.py --config configs/config.yaml --dry-run
```

---

## 🚀 Remote Training (GPU Server, Docker or Colab)

As planned, the heavy model training is decoupled and executed remotely.

### Option 1: Docker / Docker Compose (RunPod, Lambda Labs, Cloud GPU)
Clone your repository on the remote GPU instance and launch:

```bash
# Build and run GPU training container
cd docker
docker compose up --build
```

### Option 2: Google Colab / Kaggle (Free T4 / A100 GPU)
1. Open [`notebooks/remote_training_colab.ipynb`](notebooks/remote_training_colab.ipynb) in Google Colab.
2. Select **Runtime > Change runtime type > GPU**.
3. Follow the 5 simple cells to clone your repo, tile data, and train.

### Option 3: Remote CLI (Slurm / SSH Server)
```bash
python src/training/train.py \
    --config configs/config.yaml \
    --device cuda \
    --epochs 40 \
    --batch-size 4
```

---

## 🔍 Full-Field Inference & Evaluation

To segment a complete high-resolution image with automatic tiling and stitching:

```bash
python src/inference/predict.py \
    --image path/to/microscopy_image.tif \
    --weights checkpoints/oneformer_cells_best.pt \
    --output-dir predictions \
    --mask path/to/ground_truth_mask.tif   # Optional: prints AP & F1 metrics
```

Outputs:
* `<name>_pred_mask.png`: 2D integer instance mask (where each cell has an ID $1 \dots N$).
* `<name>_overlay.png`: Visual multi-colored overlay on the original microscope image.

---

## 📊 Benchmark Metrics

The pipeline evaluates predictions according to the official NeurIPS 2022 Cell Segmentation Challenge metrics:
* **AP@50**: Average Precision at IoU threshold 0.50.
* **AP@75**: Average Precision at IoU threshold 0.75.
* **mAP@[50:95]**: Mean AP over IoU thresholds from 0.50 to 0.95 (step 0.05).
* **F1-score / SEG**: Instance-level Dice score matching ground-truth cells.
