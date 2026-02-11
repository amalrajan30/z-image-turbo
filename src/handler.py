"""
RunPod Serverless Handler for Z-Image-Turbo text-to-image generation.

Z-Image-Turbo is a 6B-parameter distilled diffusion model from Tongyi-MAI
that generates high-quality images in just 8 DiT forward passes.
"""

import base64
import io
import os
import time

import runpod
import torch
from diffusers import ZImagePipeline

# ---------------------------------------------------------------------------
# Model loading — runs ONCE at worker startup (not per-request)
#
# Supports two model sources (checked in order):
#   1. RunPod model cache  — /runpod-volume/huggingface-cache/hub/
#   2. HuggingFace Hub     — downloads from huggingface.co (fallback)
# ---------------------------------------------------------------------------

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
RUNPOD_CACHE_DIR = "/runpod-volume/huggingface-cache/hub"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_model_path():
    """Return the local cached path if available, else the HF Hub model ID."""
    cache_name = MODEL_ID.replace("/", "--")
    snapshots_dir = os.path.join(RUNPOD_CACHE_DIR, f"models--{cache_name}", "snapshots")
    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            path = os.path.join(snapshots_dir, snapshots[0])
            print(f"Using RunPod cached model at: {path}")
            return path
    print(f"RunPod cache not found, loading from HuggingFace Hub: {MODEL_ID}")
    return MODEL_ID


print("Loading Z-Image-Turbo pipeline …")
_start = time.time()

pipe = ZImagePipeline.from_pretrained(
    _resolve_model_path(),
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to(DEVICE)

# Enable Flash Attention if available for faster inference
try:
    pipe.transformer.set_attention_backend("flash")
    print("Flash Attention enabled")
except Exception:
    print("Flash Attention not available, using default attention")

print(f"Model loaded in {time.time() - _start:.1f}s")

# ---------------------------------------------------------------------------
# Supported resolutions (width x height)
# ---------------------------------------------------------------------------

SUPPORTED_RESOLUTIONS = {
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1024, 768),
    (768, 1024),
    (1280, 720),
    (720, 1280),
    (1536, 1024),
    (1024, 1536),
    (2048, 2048),
}

MAX_DIMENSION = 2048
MIN_DIMENSION = 512

# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def handler(job):
    """
    Process a single image generation request.

    Expected input:
    {
        "prompt": str,              # required — text description of the image
        "width": int,               # optional — default 1024
        "height": int,              # optional — default 1024
        "num_inference_steps": int,  # optional — default 9 (yields 8 DiT forwards)
        "seed": int,                # optional — for reproducibility (-1 = random)
        "max_sequence_length": int,  # optional — max prompt tokens, default 512, max 1024
        "num_images": int,          # optional — images per prompt, default 1, max 4
    }

    Returns:
    {
        "images": [str],            # list of base64-encoded PNG images
        "seed": int,                # seed used
        "timings": {
            "inference_ms": float
        }
    }
    """
    job_input = job["input"]

    # --- Validate required fields ---
    prompt = job_input.get("prompt")
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return {"error": "A non-empty 'prompt' string is required."}

    # --- Parse optional parameters ---
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    num_inference_steps = int(job_input.get("num_inference_steps", 9))
    seed = int(job_input.get("seed", -1))
    max_sequence_length = int(job_input.get("max_sequence_length", 512))
    num_images = int(job_input.get("num_images", 1))

    # --- Validate dimensions ---
    width = max(MIN_DIMENSION, min(MAX_DIMENSION, width))
    height = max(MIN_DIMENSION, min(MAX_DIMENSION, height))
    # Round to nearest multiple of 8 (required by VAE)
    width = (width // 8) * 8
    height = (height // 8) * 8

    # --- Validate other params ---
    num_inference_steps = max(1, min(20, num_inference_steps))
    max_sequence_length = max(128, min(1024, max_sequence_length))
    num_images = max(1, min(4, num_images))

    # --- Seed ---
    if seed < 0:
        seed = torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # --- Generate ---
    t0 = time.time()

    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,  # Turbo model: NO classifier-free guidance
        max_sequence_length=max_sequence_length,
        num_images_per_prompt=num_images,
        generator=generator,
    )

    inference_ms = (time.time() - t0) * 1000

    # --- Encode images to base64 ---
    images_b64 = []
    for img in result.images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    return {
        "images": images_b64,
        "seed": seed,
        "timings": {
            "inference_ms": round(inference_ms, 1),
        },
    }


# ---------------------------------------------------------------------------
# Start the serverless worker
# ---------------------------------------------------------------------------
runpod.serverless.start({"handler": handler})
