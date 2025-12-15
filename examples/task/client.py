import asyncio
from vastai import Serverless

my_tasks = []

async def main():
    async with Serverless() as client:
        endpoint = await client.get_endpoint(name="my-pytorch-endpoint")


        payload = {
            # "epochs" : 100,
        }
        response = await endpoint.request(route="/status", payload=payload)

        print(response["response"])

if __name__ == "__main__":
    asyncio.run(main())