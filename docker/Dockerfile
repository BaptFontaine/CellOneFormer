# Base image with PyTorch and CUDA 12.1 runtime
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (OpenCV, Git, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code and configurations
COPY configs/ /workspace/configs/
COPY src/ /workspace/src/
COPY README.md /workspace/README.md

# Create persistent directories
RUN mkdir -p /workspace/data/raw /workspace/data/processed /workspace/checkpoints /workspace/logs

# Set default command (can be overridden to run training, preprocessing, etc.)
CMD ["python", "src/training/train.py", "--config", "configs/config.yaml"]
