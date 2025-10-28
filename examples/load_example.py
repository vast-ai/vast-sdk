import asyncio
from vastai_sdk import Serverless, ServerlessRequest

async def main():
    async with Serverless(instance="alpha") as client:
        endpoint = await client.get_endpoint(name="test")

        payload = {
            "input" : {
                "model": "Qwen/Qwen3-8B",
                "prompt" : "Who are you?",
                "max_tokens" : 100,
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
            responses.append(endpoint.request(route="/v1/completions", payload=payload, serverless_request=req, cost=50.0))
            await asyncio.sleep(50 / CUR_LOAD)

if __name__ == "__main__":
    asyncio.run(main())