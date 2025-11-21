import asyncio
from vastai import Serverless

MAX_TOKENS = 128

async def main():
    async with Serverless() as client:
        endpoint = await client.get_endpoint(name="dispatch_test")

        payload = {
            "input" : {
                "a" : 1
            }
        }
        
        response = await endpoint.request("/remote/remote_func_a", payload)
        print(response["response"])

        payload = {
            "input" : {
                "b" : "hello"
            }
        }
        
        response = await endpoint.request("/remote/remote_func_b", payload)
        print(response["response"])

if __name__ == "__main__":
    asyncio.run(main())