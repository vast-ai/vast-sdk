#!/usr/bin/env python3
from vastai.serverless.remote.endpoint import remote, Endpoint

my_benchmark_dataset = [
    {"a": 1},
    {"a": 2},
    {"a": 3},
]

@remote(endpoint_name="my-remote-endpoint")
async def remote_func_a(x: int):
    return x + 1

endpoint = Endpoint(
    "test-endpoint"
)

endpoint.on_start("bash -c \"export PYWORKER_REPO='https://github.com/LucasArmandVast/example-worker'; export PYWORKER_REF='vast-server'; WORKER_SDK='true'; curl -L https://raw.githubusercontent.com/vast-ai/vast-sdk/refs/heads/remote/start_server.sh | bash; apt-get install -y wget; wget -O endpoint.py http://sshdev.vast.ai:3002/download && VAST_REMOTE_DISPATCH_MODE=serve python3 endpoint.py\"")

endpoint.ready()

