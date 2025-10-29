import asyncio
from vastai import Serverless

async def main():
    async with Serverless() as client:
        endpoint = await client.get_endpoint(name="my-endpoint")

        payload = {
            "input" : {
                "model": "Qwen/Qwen3-8B",
                "prompt" : "Who are you?",
                "max_tokens" : 100,
                "temperature" : 0.7
            }
        }
        
        response = await endpoint.request("/v1/completions", payload)
        print(response["response"]["choices"][0]["text"])

if __name__ == "__main__":
    asyncio.run(main())