import mandelbrot
import asyncio
import base64
from io import BytesIO
from PIL import Image

async def main():
    # Call the remote endpoint
    result = await mandelbrot.render_mandelbrot(
        width=1024,
        height=768,
        max_iter=300
    )
    result = result["result"]
    print("Fractal render complete:")
    print(f"  width={result['width']}")
    print(f"  height={result['height']}")
    print(f"  max_iter={result['max_iter']}")

    # Decode the base64 PNG and save it locally
    png_data = base64.b64decode(result["png_base64"])
    img = Image.open(BytesIO(png_data))
    img.save("fractal.png")

    print("Saved fractal.png")

if __name__ == "__main__":
    asyncio.run(main())
