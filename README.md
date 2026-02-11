# Z-Image-Turbo on RunPod Serverless

Deploy [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) as a serverless API on [RunPod](https://runpod.io). Generate high-quality images in ~8 inference steps with a 6B-parameter distilled diffusion model from Alibaba's Tongyi Lab.

## Features

- **Fast inference** — 8 DiT forward passes (vs 50 for the base model), no classifier-free guidance needed
- **RunPod model caching** — ~33GB model loaded from host NVMe, not baked into the Docker image (~2-3GB image size)
- **Flash Attention** — automatically enabled when available
- **Bilingual** — supports both English and Chinese prompts
- **Flexible resolution** — 512x512 up to 2048x2048

## Project Structure

```
├── Dockerfile           # CUDA 12.1 + Python 3.11 runtime
├── requirements.txt     # Python dependencies
├── src/
│   └── handler.py       # RunPod serverless handler
├── deploy.py            # Programmatic deployment via RunPod SDK
├── client.py            # CLI client to call the endpoint
└── test_input.json      # Test payload for local testing
```

## Deploy

### 1. Build and push the Docker image

```bash
docker build --platform linux/amd64 -t yourdockerhub/z-image-turbo:v1 .
docker push yourdockerhub/z-image-turbo:v1
```

### 2. Create the endpoint on RunPod

**Option A — Via the Console:**

1. Go to **Serverless > New Endpoint**
2. Set Docker image to `yourdockerhub/z-image-turbo:v1`
3. Select GPU: **24GB AMPERE** (L4/A5000/3090) or **24GB ADA** (4090)
4. Under **Settings > Model**, enter: `Tongyi-MAI/Z-Image-Turbo`
5. Deploy

**Option B — Via script:**

```bash
export RUNPOD_API_KEY="your-key"
python deploy.py --image docker.io/yourdockerhub/z-image-turbo:v1
```

Then enable model caching in the console as prompted.

### 3. Enable Model Caching

This is the critical step. In the RunPod Console:

**Serverless > your endpoint > Settings > Model** → `Tongyi-MAI/Z-Image-Turbo`

RunPod will pre-download the ~33GB model onto host NVMe storage. Workers load from this cache at startup — no download during cold start, no billing for download time.

## Usage

### Python client

```bash
pip install runpod
export RUNPOD_API_KEY="your-key"

python client.py \
  --endpoint-id YOUR_ENDPOINT_ID \
  --prompt "A futuristic city at sunset, cyberpunk aesthetic" \
  --output city.png
```

### curl

```bash
curl -s -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A red panda reading a book in a cozy library",
      "width": 1024,
      "height": 1024
    }
  }'
```

### Python SDK

```python
import runpod, base64

runpod.api_key = "your-key"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

result = endpoint.run_sync({
    "prompt": "An astronaut riding a horse on Mars",
    "width": 1024,
    "height": 1024,
    "seed": 42,
})

img_bytes = base64.b64decode(result["images"][0])
with open("output.png", "wb") as f:
    f.write(img_bytes)
```

## API Reference

### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Text description of the image (English or Chinese) |
| `width` | int | 1024 | Image width (512–2048, rounded to multiple of 8) |
| `height` | int | 1024 | Image height (512–2048, rounded to multiple of 8) |
| `num_inference_steps` | int | 9 | Denoising steps (9 = 8 DiT forwards) |
| `seed` | int | -1 | Random seed (-1 for random) |
| `max_sequence_length` | int | 512 | Max prompt tokens (up to 1024) |
| `num_images` | int | 1 | Images per prompt (1–4) |

### Output

```json
{
  "images": ["<base64 PNG>"],
  "seed": 42,
  "timings": {
    "inference_ms": 1234.5
  }
}
```

## GPU Recommendations

| Pool | GPUs | VRAM | $/sec | Notes |
|------|------|------|-------|-------|
| AMPERE_24 | L4, A5000, 3090 | 24GB | $0.00019 | Best cost/performance |
| ADA_24 | 4090 | 24GB | $0.00031 | Fastest consumer GPU |
| AMPERE_48 | A6000, A40 | 48GB | $0.00034 | Needed for 2048x2048 or batch |

## Local Testing

```bash
# Install test dependencies
pip install -r requirements.txt

# Run handler locally with RunPod test mode
cd src && python handler.py --test_input ../test_input.json
```

## License

Apache 2.0 — same as the [Z-Image-Turbo model](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).
