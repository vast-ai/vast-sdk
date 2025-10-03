import asyncio
from vastai_sdk import Serverless
import os

API_KEY = os.environ.get("VAST_API_KEY")

async def main():
    client = Serverless(API_KEY, debug=True)
    endpoint = await client.get_endpoint(name="my_endpoint")

    payload = {
        "input" : {
            "model": "Qwen/Qwen3-8B",
            "prompt" : "Who are you?",
            "max_tokens" : 100,
            "temperature" : 0.7
        }
    }
    
    response = await endpoint.request("/v1/completions", payload)
    print(response["choices"][0]["text"])
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())