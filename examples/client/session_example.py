import asyncio
import random
from vastai import Serverless

async def main():
    async with Serverless() as client:
        endpoint = await client.get_endpoint(name="my-comfy-endpoint")

        payload = {
            "input": {
                "modifier": "Text2Image",
                "modifications": {
                    "prompt": "Generate a page from a peanuts comic strip.",
                    "width": 512,
                    "height": 512,
                    "steps": 10,
                    "seed": random.randint(1, 1000),
                },
            }
        }

        # Creating a session ensures all requests are routed to a single worker
        # for the lifetime of the session.
        # Use this for asynchronous jobs
        session = await endpoint.session()
        try:
            response_a = await session.request("/generate/sync", payload)
            response_b = await session.request("/generate/sync", payload)
            print(response_a["response"]["output"][0]["local_path"])
            print(response_b["response"]["output"][0]["local_path"])
        finally:
            await session.close()

if __name__ == "__main__":
    asyncio.run(main())
