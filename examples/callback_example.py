import asyncio
from vastai import Serverless, ServerlessRequest

async def main():
    async with Serverless() as client:
        endpoint = await client.get_endpoint(name="endpoint")

        payload = {
            "input" : {
                "model": "Qwen/Qwen3-8B",
                "prompt" : "Who are you?",
                "max_tokens" : 100,
                "temperature" : 0.7
            }
        }
        
        # Create a ServerlessRequest object to attach callbacks before submitting the request
        req = ServerlessRequest()

        # Attach a callback to run when the machine finished work on the request
        def work_finished_callback(response):
            print(f"Request finished. Got response of length {len(response["response"]["choices"][0]["text"])}")

        req.then(work_finished_callback)

        response = await endpoint.request(route="/v1/completions", payload=payload, serverless_request=req)
        print(response)

if __name__ == "__main__":
    asyncio.run(main())