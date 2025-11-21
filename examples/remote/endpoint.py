from vastai.serverless.remote.endpoint import Endpoint
import asyncio

my_benchmark_dataset = [
    {"a": 1},
    {"a": 2},
    {"a": 3},
]

@remote(endpoint_name="my-remote-endpoint")
@benchmark(dataset=my_benchmark_dataset)
async def remote_func_a(x: int):
    return x + 1

if __name__ == "__main__":
    endpoint = Endpoint()
    endpoint.ready()

    