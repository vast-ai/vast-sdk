import slime_mold
import asyncio
import base64
from io import BytesIO
from PIL import Image

async def main():
    # Call the remote slime mold simulation endpoint
    result = await slime_mold.render_slime_mold(
        width=512,
        height=512,
        steps=150000,
        n_agents=20000
    )
    result = result["result"]

    print("Slime mold simulation complete:")
    print(f"  width={result['width']}")
    print(f"  height={result['height']}")
    print(f"  steps={result['steps']}")
    print(f"  n_agents={result['n_agents']}")
    print(f"  device_used={result['device_used']}")

    # Decode the base64 PNG and save it locally
    png_data = base64.b64decode(result["png_base64"])
    img = Image.open(BytesIO(png_data))
    img.save("slime_mold.png")

    print("Saved slime_mold.png")

if __name__ == "__main__":
    asyncio.run(main())
