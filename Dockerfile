# =============================================================================
# RunPod Serverless Worker — Z-Image-Turbo (text-to-image)
#
# Uses RunPod model caching — weights are NOT baked into the image.
# Set the "Model" field on your endpoint to: Tongyi-MAI/Z-Image-Turbo
#
# Build:
#   docker build --platform linux/amd64 -t <your-dockerhub>/z-image-turbo:v1 .
#
# Push:
#   docker push <your-dockerhub>/z-image-turbo:v1
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# --- System packages ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        git \
        wget && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Upgrade pip ---
RUN python -m pip install --upgrade pip setuptools wheel

# --- Install PyTorch with CUDA 12.1 ---
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# --- Install Python dependencies ---
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# --- Copy handler ---
COPY src/handler.py /handler.py

# --- Start the serverless worker ---
CMD ["python", "-u", "/handler.py"]
