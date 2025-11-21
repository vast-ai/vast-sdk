import comfy_endpoint
import asyncio
import base64


def download_base64(data_b64: str, output_path: str = "generated_image.png"):
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(data_b64))


async def worker(name: str, sem: asyncio.Semaphore):
    while True:
        async with sem:
            image_b64 = await comfy_endpoint.generate_image(
                "Trippy, psychedlic visuals, with swirls, eyeballs, spirits, figures. Reminsicent of Google DeepDream"
            )
            download_base64(image_b64, f"generated_image.png")
        await asyncio.sleep(0.5)


async def main():
    sem = asyncio.Semaphore(5)  # at most 5 requests in flight
    workers = [asyncio.create_task(worker(f"worker-{i}", sem)) for i in range(5)]
    await asyncio.gather(*workers)


if __name__ == "__main__":
    asyncio.run(main())
