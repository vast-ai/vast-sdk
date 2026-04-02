"""Client script: chain image-gen nodes like a ComfyUI workflow.

Usage:
    python3 examples/deployments/image_gen/run.py
"""

import asyncio
from pathlib import Path
from deploy import txt2img, img2img, upscale


async def main():
    prompt = "a corgi astronaut floating in space, earth in background, cinematic lighting"
    negative = "blurry, low quality, watermark, text"

    # --- Node 1: txt2img ---
    print(f"Generating base image ...")
    base_png = await txt2img(prompt, negative_prompt=negative, steps=30, seed=42)
    Path("01_base.png").write_bytes(base_png)
    print(f"  -> 01_base.png ({len(base_png)} bytes)")

    # --- Node 2: img2img refinement ---
    print("Refining with img2img ...")
    refined_png = await img2img(
        base_png,
        prompt=f"{prompt}, highly detailed, 8k",
        negative_prompt=negative,
        strength=0.35,
        steps=25,
    )
    Path("02_refined.png").write_bytes(refined_png)
    print(f"  -> 02_refined.png ({len(refined_png)} bytes)")

    # --- Node 3: upscale ---
    print("Upscaling 2x ...")
    upscaled_png = await upscale(refined_png, scale_factor=2)
    Path("03_upscaled.png").write_bytes(upscaled_png)
    print(f"  -> 03_upscaled.png ({len(upscaled_png)} bytes)")

    print("Done! Pipeline: txt2img -> img2img -> upscale")


if __name__ == "__main__":
    asyncio.run(main())
