import asyncio
from vastai_sdk import ServerlessClient, ServerlessRequest
import os

API_KEY = os.environ.get("VAST_API_KEY")

async def main():
    client = ServerlessClient(API_KEY, debug=True)
    endpoint = await client.get_endpoint(name="my_endpoint")

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

    # Attach a callback to run when the machine starts work on the request
    def work_start_callback():
        print("Request is being processed")

    req.add_on_work_start_callback(work_start_callback)

    # Attach a callback to run when the machine finished work on the request
    def work_finished_callback(response):
        print(f"Request finished. Got response of length {len(response["choices"][0]["text"])}")

    req.then(work_finished_callback)

    response = await endpoint.request(route="/v1/completions", payload=payload, serverless_request=req)
    print(response)
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())