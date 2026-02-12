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
# Load order:
#   1. RunPod model cache at /runpod-volume/huggingface-cache/hub/
#      (pre-downloaded by RunPod when "Model" is set on the endpoint)
#   2. HuggingFace Hub fallback (downloads ~33GB on first cold start)
#
# The Dockerfile sets HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE to point
# at the RunPod cache volume, so `from_pretrained(MODEL_ID)` will find the
# cached files automatically without needing an explicit local path.
# ---------------------------------------------------------------------------

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[init] Device: {DEVICE}")
print(f"[init] HF_HOME={os.environ.get('HF_HOME', 'not set')}")
print(f"[init] HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE', 'not set')}")

# Log whether RunPod cache volume is mounted
cache_dir = os.environ.get("HF_HUB_CACHE", "/runpod-volume/huggingface-cache/hub")
if os.path.isdir(cache_dir):
    print(f"[init] Cache directory exists: {cache_dir}")
    try:
        contents = os.listdir(cache_dir)
        model_dirs = [d for d in contents if "Z-Image-Turbo" in d]
        print(f"[init] Matching model dirs in cache: {model_dirs or 'none'}")
    except Exception as e:
        print(f"[init] Could not list cache dir: {e}")
else:
    print(f"[init] Cache directory NOT found: {cache_dir}")
    print("[init] Will download from HuggingFace Hub (this may take a while)")

print(f"[init] Loading {MODEL_ID} …")
_start = time.time()

pipe = ZImagePipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

print(f"[init] Pipeline loaded to CPU in {time.time() - _start:.1f}s, moving to {DEVICE} …")
_t_gpu = time.time()
pipe.to(DEVICE)
print(f"[init] Moved to {DEVICE} in {time.time() - _t_gpu:.1f}s")

# Enable Flash Attention if available for faster inference
try:
    pipe.transformer.set_attention_backend("flash")
    print("[init] Flash Attention enabled")
except Exception:
    print("[init] Flash Attention not available, using default attention")

print(f"[init] Ready. Total model load time: {time.time() - _start:.1f}s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
