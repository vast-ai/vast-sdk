#!/usr/bin/env python3
import os
import inspect
import functools
import asyncio

from anyio import Path
from vastai.serverless.remote.endpoint_group import EndpointGroup
from vastai.serverless.remote.worker_group import WorkerGroup
from vastai.serverless.remote.template import Template


mode = os.getenv("VAST_REMOTE_DISPATCH_MODE", "client")

REMOTE_DISPATCH_FUNCTIONS_BY_ENDPOINT_NAME = {}

def get_mode():
    return mode


def remote(endpoint_name: str):
    """
    Decorator that converts a function into a remote call to a Vast.ai endpoint.

    Args:
        endpoint_name: The name of the endpoint to call

    Example:
        @remote(endpoint_name="my-endpoint")
        def my_function(a: int, b: str) -> dict:
            return {"result": f"{a} {b}"}

        # When called, it will make a request to the endpoint
        result = await my_function(a=1, b="hello")
    """
    def decorator(func):
        func_name = func.__name__
        sig = inspect.signature(func)
        if get_mode() == "client":
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                from vastai import Serverless

                # Bind arguments to get a mapping of parameter names to values
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Construct the payload in the format expected by the endpoint
                payload = {
                    "input": dict(bound_args.arguments)
                }

                print("payload:", payload)

                # Make the remote request
                async with Serverless() as client:
                    endpoint = await client.get_endpoint(name=endpoint_name)
                    print("endpoint:", endpoint)
                    response = await endpoint.request(f"/remote/{func_name}", payload)
                    print("response:", response)
                    return response["response"]

            return async_wrapper
        elif get_mode() == "serve":
            # Register this function under the configured endpoint so that
            # Endpoint.ready() in serve mode can expose it via Worker.
            funcs_for_endpoint = REMOTE_DISPATCH_FUNCTIONS_BY_ENDPOINT_NAME.setdefault(
                endpoint_name, {}
            )
            funcs_for_endpoint[func_name] = func

            # In serve mode, the function should just run locally when called
            # (e.g. useful for tests or local invocation), so we return it unchanged.
            return func

        # Optional: default behavior (e.g. deploy mode) – just return the original
        return func

    return decorator


#TODO: THe remote decorator in serve mode needs to get the endpoint URL, construct the params,
#then call the endpoint with the structured params the pyworker endpoint expects
@remote(endpoint_name="test-endpoint")
async def remote_func(x: int, y: int):
    return x + y

class Endpoint:
    def __init__(
        self,
        name: str,
        cold_mult: int = 3,
        min_workers: int = 5,
        max_workers: int = 16,
        min_load: int = 1,
        min_cold_load: int = 0,
        target_util: float = 0.9,
        image_name: str = "vastai/base-image:@vastai-automatic-tag",
        env_vars: dict = {},
        search_params: str = "",
        disk_space: int = 128,
    ):
        # --- Endpoint Configuration ---
        self.name = name
        self.cold_mult = cold_mult
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.min_load = min_load
        self.min_cold_load = min_cold_load
        self.target_util = target_util

        # --- Template Configuration ---
        self.image_name = image_name
        self.env_vars = env_vars
        self.search_params = search_params
        self.disk_space = disk_space
        self.__onstart_cmd = ""
        self.__pip_packages_requested = False

        # --- Worker Configuration ---
        self.remote_dispatch_functions = REMOTE_DISPATCH_FUNCTIONS_BY_ENDPOINT_NAME.get(name, {})
        self.benchmark_function_name = None
        self.benchmark_dataset = []
        self.benchmark_generator = None
        self.model_log_file = "/var/log/remote/debug.log"
        self.model_healthcheck_endpoint = "health"
        self.on_init_function = None
        self.background_task = None

    def apt_get(self, package: str):
        self.__onstart_cmd += f"apt-get install -y {package}\n"

    def uv_pip_install(self, package: str):
        if not self.__pip_packages_requested:
            self.__onstart_cmd += "curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv && source .venv/bin/activate\n"
            self.__pip_packages_requested = True

        self.__onstart_cmd += f"uv pip install {package}\n"

    def on_start(self, cmd: str):
        self.__onstart_cmd += f"{cmd}\n"

    def __install_remote_worker_script(self):
        worker_script_download_url = os.environ["VAST_WORKER_DOWNLOAD_URL"]
        self.apt_get("wget")
        self.__onstart_cmd += f"""
wget -O endpoint.py {worker_script_download_url} && VAST_REMOTE_DISPATCH_MODE=serve python3 endpoint.py

"""

    def ready(self):
        if (mode := get_mode()) == "deploy":

            self.__install_remote_worker_script()

            vast_api_key = os.environ.get("VAST_API_KEY")
            if not vast_api_key:
                raise ValueError("VAST_API_KEY environment variable is not set")

            template = Template(
                vast_api_key,
                self.image_name,
                self.env_vars,
                self.disk_space,
                template_name="template-test",
                onstart_cmd=self.__onstart_cmd,
            )
            template_id = template.create_template()

            endpoint_group = EndpointGroup(
                vast_api_key,
                self.name,
                self.cold_mult,
                self.min_workers,
                self.max_workers,
                self.min_load,
                self.min_cold_load,
                self.target_util,
            )
            endpoint_group_id = endpoint_group.create_endpoint_group()

            worker_group = WorkerGroup(
                vast_api_key,
                self.cold_mult,
                endpoint_group_id,
                self.name,
                self.min_load,
                self.search_params,
                self.target_util,
                template_id,
            )


        elif mode == "serve":
            from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig

            remote_function_handlers : list[HandlerConfig] = []

            for remote_func_name, remote_func in self.remote_dispatch_functions.items():
                benchmark_config: BenchmarkConfig = None
                if remote_func_name == self.benchmark_function_name:
                    if len(self.benchmark_dataset) > 0:
                        benchmark_config = BenchmarkConfig(
                            dataset=self.benchmark_dataset,
                            runs=10
                        )
                    elif self.benchmark_generator is not None:
                        benchmark_config = BenchmarkConfig(
                            generator=self.benchmark_generator,
                            runs=10
                        )
                    else:
                        raise ValueError("Must specify either a benchmark dataset or benchmark generator for benchmark function")

                remote_func_handler = HandlerConfig(
                    route=f"/remote/{remote_func_name}",
                    is_remote_dispatch=True,
                    remote_dispatch_function=remote_func,
                    allow_parallel_requests=False,
                    benchmark_config=benchmark_config,
                    max_queue_time=30.0
                )
                
                remote_function_handlers.append(remote_func_handler)


            remote_worker_config = WorkerConfig(
                model_log_file=self.model_log_file,
                handlers = remote_function_handlers,
                log_action_config=LogActionConfig(
                    on_load=["Remote Dispatch ready"],
                    on_error=["Remote Dispatch error"]
                )
            )

            # Build the worker handling our remote dispatch functions
            remote_worker = Worker(remote_worker_config)

            # Call on_init if present
            if self.on_init_function is not None:
                try:
                    self.on_init_function()
                except Exception as ex:
                    raise Exception(f"Error in on_init function: {ex}")

            worker_task = asyncio.create_task(remote_worker.run_async())

            model_log = Path(self.model_log_file)
            async def model_load_task():
                # Create parent directories if they don't exist
                await model_log.parent.mkdir(parents=True, exist_ok=True)
                await model_log.write_text("Remote Dispatch ready")
            asyncio.run(model_load_task)

            # Enter the background task if present
            if self.background_task:
                asyncio.gather(worker_task, self.background_task())
            else:
                asyncio.run(worker_task)
        elif mode == "client":
            # Nothing to do here
            pass

