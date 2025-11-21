import asyncio
from vastai import Serverless

MAX_TOKENS = 128

async def main():
    async with Serverless() as client:
        endpoint = await client.get_endpoint(name="test-endpoint")

        payload = {
            "input" : {
                "x" : 2,
                "y" : 3
            }
        }
        
        response = await endpoint.request("/remote/remote_func_a", payload)
        print(f"remote_func_a(2, 3) = {response["response"]["result"]}")
        
        response = await endpoint.request("/remote/remote_func_b", payload)
        print(f"remote_func_b(2, 3) = {response["response"]["result"]}")

if __name__ == "__main__":
    asyncio.run(main())