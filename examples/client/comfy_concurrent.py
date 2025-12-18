import asyncio
import random
from vastai import Serverless


async def generate(endpoint, idx):
    payload = {
        "input": {
            "modifier": "Text2Image",
            "modifications": {
                "prompt": "Generate a page from a peanuts comic strip.",
                "width": 512,
                "height": 512,
                "steps": 10,
                "seed": random.randint(1, 1000),
            }
        }
    }

    response = await endpoint.request("/generate/sync", payload)

    local_path = response["response"]["output"][0]["local_path"]
    print(f"[{idx}] Generated image at:", local_path)

    return local_path


async def main(concurrency=256):
    async with Serverless(debug=True) as client:
        endpoint = await client.get_endpoint(name="my-comfy-endpoint")

        tasks = [
            generate(endpoint, idx)
            for idx in range(concurrency)
        ]

        results = await asyncio.gather(*tasks)

        print("\nAll generations complete:")
        for path in results:
            print(" -", path)


if __name__ == "__main__":
    asyncio.run(main())
