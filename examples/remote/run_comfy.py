import comfy_endpoint
import asyncio
import base64
import sys
from datetime import datetime

def download_base64(data_b64: str, output_path: str):
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(data_b64))
    print(f"Saved image to {output_path}")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py \"your prompt here\"")
        sys.exit(1)

    prompt = sys.argv[1]

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"comfy_image_{timestamp}.png"

    image_b64 = await comfy_endpoint.generate_image(prompt)
    download_base64(image_b64, filename)

if __name__ == "__main__":
    asyncio.run(main())
