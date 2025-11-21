import asyncio
from vastai import Serverless

MAX_TOKENS = 128

async def main():
    async with Serverless() as client:
        endpoint = await client.get_endpoint(name="qwendpoint")

        prompt = {
            "input": {
                "body": {
                    "model": "Qwen/Qwen3-8B",
                    "prompt": "Who was the first person to walk on the moon?",
                    "temperature": 0.7,
                    "max_tokens": 500,
                }
            }
        }
        
        response = await endpoint.request("/remote/llm_infer", prompt)
        print(f"{response["response"]}")

if __name__ == "__main__":
    asyncio.run(main())