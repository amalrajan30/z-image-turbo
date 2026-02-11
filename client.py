"""
Client for the Z-Image-Turbo RunPod Serverless endpoint.

Usage:
    export RUNPOD_API_KEY="your-api-key"
    python client.py --endpoint-id YOUR_ENDPOINT_ID --prompt "A cat astronaut"
"""

import argparse
import base64
import os
import time

import runpod


def main():
    parser = argparse.ArgumentParser(description="Generate images via Z-Image-Turbo on RunPod")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--output", default="output.png", help="Output filename (or prefix for multiple images)")
    parser.add_argument("--sync", action="store_true", help="Use synchronous /runsync (blocks until done)")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable is required.")
        raise SystemExit(1)

    runpod.api_key = api_key
    endpoint = runpod.Endpoint(args.endpoint_id)

    payload = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "num_images": args.num_images,
    }

    print(f"Sending request to endpoint {args.endpoint_id} …")
    t0 = time.time()

    if args.sync:
        result = endpoint.run_sync(payload, timeout=120)
    else:
        job = endpoint.run(payload)
        print(f"Job queued: {job.job_id}")
        result = job.output(timeout=120)

    elapsed = time.time() - t0
    print(f"Response received in {elapsed:.1f}s")

    if "error" in result:
        print(f"Error: {result['error']}")
        raise SystemExit(1)

    print(f"Seed: {result['seed']}")
    print(f"Inference time: {result['timings']['inference_ms']:.0f}ms")

    images = result["images"]
    for i, img_b64 in enumerate(images):
        if len(images) == 1:
            filename = args.output
        else:
            base, ext = os.path.splitext(args.output)
            filename = f"{base}_{i}{ext}"

        img_bytes = base64.b64decode(img_b64)
        with open(filename, "wb") as f:
            f.write(img_bytes)
        print(f"Saved: {filename} ({len(img_bytes) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
