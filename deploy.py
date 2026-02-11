"""
Deploy Z-Image-Turbo to RunPod Serverless programmatically.

Uses RunPod's model caching feature — the ~33GB model weights are
pre-loaded onto host NVMe by RunPod, NOT baked into the Docker image.

Usage:
    export RUNPOD_API_KEY="your-api-key"
    python deploy.py --image docker.io/<user>/z-image-turbo:v1
"""

import argparse
import os

import runpod

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


def main():
    parser = argparse.ArgumentParser(description="Deploy Z-Image-Turbo to RunPod")
    parser.add_argument("--image", required=True, help="Docker image URL (e.g. docker.io/user/z-image-turbo:v1)")
    parser.add_argument("--name", default="z-image-turbo", help="Endpoint name")
    parser.add_argument("--gpu", default="AMPERE_24", help="GPU pool ID (default: AMPERE_24 — 24GB VRAM)")
    parser.add_argument("--min-workers", type=int, default=0, help="Minimum active workers (0 = scale to zero)")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum workers")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable is required.")
        raise SystemExit(1)

    runpod.api_key = api_key

    print(f"Creating template for image: {args.image}")
    print(f"Model cache configured for: {MODEL_ID}")
    template = runpod.create_template(
        name=f"{args.name}-template",
        image_name=args.image,
        is_serverless=True,
    )
    print(f"Template created: {template['id']}")

    print(f"Creating endpoint: {args.name}")
    endpoint = runpod.create_endpoint(
        name=args.name,
        template_id=template["id"],
        gpu_ids=args.gpu,
        workers_min=args.min_workers,
        workers_max=args.max_workers,
    )
    print(f"Endpoint created: {endpoint['id']}")
    print(f"\nEndpoint URL: https://api.runpod.ai/v2/{endpoint['id']}")
    print()
    print("IMPORTANT: Enable model caching in the RunPod Console:")
    print(f"  Serverless > {args.name} > Settings > Model > {MODEL_ID}")
    print("  This pre-loads the ~33GB model onto host NVMe (no billing during download).")


if __name__ == "__main__":
    main()
