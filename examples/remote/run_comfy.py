import comfy_endpoint
import asyncio
import base64

def download_base64(data_b64: str, output_path: str = "generated_image.png"):
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(data_b64))

async def main():
    prompt = "View from a balcony in Tuscany"
    image_b64 = await comfy_endpoint.generate_image(prompt)
    download_base64(image_b64)

if __name__ == "__main__":
    asyncio.run(main())