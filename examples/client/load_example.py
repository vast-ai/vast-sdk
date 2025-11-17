import asyncio
from vastai import Serverless, ServerlessRequest

MAX_TOKENS = 128

async def main():
    async with Serverless(debug=True) as client:
        endpoint = await client.get_endpoint(name="my-vllm-endpoint")

        payload = {
            "input" : {
                "model": "Qwen/Qwen3-8B",
                "prompt" : "Who are you?",
                "max_tokens" : MAX_TOKENS,
                "temperature" : 0.7
            }
        }
        
        responses = []

        CUR_LOAD = 500
        while True:
            # Create a ServerlessRequest object to attach callbacks before submitting the request
            req = ServerlessRequest()
            # Attach a callback to run when the machine finished work on the request
            def work_finished_callback(response):
                print(response["response"]["choices"][0]["text"])
            req.then(work_finished_callback)
            responses.append(endpoint.request(route="/v1/completions", payload=payload, serverless_request=req, cost=MAX_TOKENS))
            await asyncio.sleep(100 / CUR_LOAD)

if __name__ == "__main__":
    asyncio.run(main())