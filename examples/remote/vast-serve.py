from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig
from typing import Callable, Awaitable
from aiohttp import web, ClientResponse
import asyncio
import random
# This code runs inside of endpoint.submit()

MODE = "serve"
MODEL_SERVER_URL = "127.0.0.1"
MODEL_SERVER_PORT = 5001
MODEL_LOG_FILE = "/var/log/remote/debug.log"
MODEL_HEALTHCHECK_ENDPOINT = "health"

MODEL_LOAD_LOG_MSG = [
    "Remote Dispatch ready",
]

MODEL_ERROR_LOG_MSGS = [
    "Remote Dispatch error"
]

def remote_func_a(a: int):
    return a + 1

def remote_func_b(b: str):
    return b + "-str"

ENDPOINT_BENCHMARK_FUNCTION = "remote_func_a"
ENDPOINT_BENCHMARK_DATASET = [
    {
        "a": 1
    },
    {
        "a": 2
    },
    {
        "a": 3
    },
]

def benchmark_generator() -> dict:
    return { "a" : random.randint(0, 100)}

ENDPOINT_BENCHMARK_GENERATOR: Callable[[], dict] = benchmark_generator

ENDPOINT_REMOTE_DISPATCH_FUNCTIONS : dict[Callable] = {
    "remote_func_a" : remote_func_a,
    "remote_func_b" : remote_func_b
}

def on_init():
    print("I'm the initialization function!")

ENDPOINT_INIT_FUNCTION: Callable = on_init

async def background_task():
    while True:
        print("Running the backgroun task!")
        await asyncio.sleep(10)

ENDPOINT_BACKGROUND_TASK: Callable[[], Awaitable[None]] = background_task

def extract_remote_dispatch_params(json: dict) -> dict:
    """
    Extract the params from the remote dispatch request
    """
    if json.get("params"):
        return json["params"]
    else:
        raise ValueError("Request JSON missing params")


async def endpoint_submit():
    if MODE == "serve":
        remote_function_handlers : list[HandlerConfig]

        for remote_func_name, remote_func in ENDPOINT_REMOTE_DISPATCH_FUNCTIONS.items():
            benchmark_config: BenchmarkConfig = None
            if remote_func_name == ENDPOINT_BENCHMARK_FUNCTION:
                if len(ENDPOINT_BENCHMARK_DATASET) > 0:
                    benchmark_config = BenchmarkConfig(
                        dataset=ENDPOINT_BENCHMARK_DATASET,
                        runs=10
                    )
                elif ENDPOINT_BENCHMARK_GENERATOR is not None:
                    benchmark_config = BenchmarkConfig(
                        generator=ENDPOINT_BENCHMARK_DATASET,
                        runs=10
                    )
                else:
                    raise ValueError("Must specify either a benchmark dataset or benchmark generator for benchmark function")

            remote_func_hander = HandlerConfig(
                route="/remote/{remote_func_name}",
                is_remote_dispatch=True,
                remote_dispatch_function=remote_func,
                allow_parallel_requests=False,
                request_parser=extract_remote_dispatch_params,
                benchmark_config=benchmark_config,
                max_queue_time=30.0
            )
            
            remote_function_handlers.append(remote_func_hander)


        remote_worker_config = WorkerConfig(
            model_log_file=MODEL_LOG_FILE,
            handlers = remote_function_handlers,
            log_action_config=LogActionConfig(
                on_load=MODEL_LOAD_LOG_MSG,
                on_error=MODEL_ERROR_LOG_MSGS,
            )
        )

        # Build the worker handling our remote dispatch functions
        remote_worker = Worker(remote_worker_config)

        # Call on_init if present
        if ENDPOINT_INIT_FUNCTION is not None:
            ENDPOINT_INIT_FUNCTION()

        # Run the worker asyncronously 
        worker_task = asyncio.run(remote_worker.run())

        # Enter the background task if present
        if ENDPOINT_BACKGROUND_TASK:
            await ENDPOINT_BACKGROUND_TASK
        else:
            await worker_task