from vastai import Worker, WorkerConfig, HandlerConfig
import asyncio 
import time

handler_config = HandlerConfig(
    endpoint="/v1/completions",
    route="/v1/completions"
)

worker_config = WorkerConfig(
    model_server_port=18000,
    model_log_file="/var/log/portal/vllm.log",
    handlers=[handler_config]
)

pyworker = asyncio.run(Worker(worker_config).run())

while True:
    print("worker.py still running")
    time.sleep(1)