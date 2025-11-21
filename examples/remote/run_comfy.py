import comfy_endpoint
import asyncio

async def main():
    image_b64 = await comfy_endpoint.generate_image("A scenic photo from a balcony in Tuscany.")
    print(image_b64)

if __name__ == "__main__":
    asyncio.run(main())