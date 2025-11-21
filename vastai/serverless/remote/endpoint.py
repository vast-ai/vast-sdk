#!/usr/bin/env python3
import os
import inspect
import functools
import asyncio
import requests
import sys

from anyio import Path
from vastai.serverless.remote.endpoint_group import EndpointGroup
from vastai.serverless.remote.worker_group import WorkerGroup
from vastai.serverless.remote.template import Template


mode = os.getenv("VAST_REMOTE_DISPATCH_MODE", "client")

REMOTE_DISPATCH_FUNCTIONS_BY_ENDPOINT_NAME = {}

BENCHMARK_CONFIG_BY_ENDPOINT_NAME = {}

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


def benchmark(
    endpoint_name: str,
    *,
    dataset: list[dict] | None = None,
    generator=None,
):
    """
    Decorator to mark a remote function as the benchmark function for an endpoint.

    Exactly one of `dataset` or `generator` should be provided.

    Example:

        @benchmark(
            endpoint_name="my-endpoint",
            dataset=[{"a": 1}, {"a": 2}, {"a": 3}],
        )
        @remote(endpoint_name="my-endpoint")
        async def remote_func_a(a: int):
            ...

    Or:

        import random

        def benchmark_generator() -> dict:
            return {"a": random.randint(0, 100)}

        @benchmark(
            endpoint_name="my-endpoint",
            generator=benchmark_generator,
        )
        @remote(endpoint_name="my-endpoint")
        async def remote_func_a(a: int):
            ...
    """
    if dataset is not None and generator is not None:
        raise ValueError("Specify either dataset or generator, not both")
    if dataset is None and generator is None:
        raise ValueError("Must specify either dataset or generator")

    # Normalize dataset to a list
    if dataset is not None and not isinstance(dataset, list):
        raise TypeError("dataset must be a list of dicts")

    def decorator(func):
        func_name = func.__name__

        BENCHMARK_CONFIG_BY_ENDPOINT_NAME[endpoint_name] = {
            "function_name": func_name,
            "dataset": dataset or [],
            "generator": generator,
        }

        return func

    return decorator

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
        model_log_file: str = "/var/log/remote/debug.log",
        model_backend_load_logs: str = ["Remote Dispatch ready"],
        model_healthcheck_endpoint: str = "health"
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
        benchmark_cfg = BENCHMARK_CONFIG_BY_ENDPOINT_NAME.get(name)
        if benchmark_cfg:
            self.benchmark_function_name = benchmark_cfg["function_name"]
            self.benchmark_dataset = benchmark_cfg["dataset"]
            self.benchmark_generator = benchmark_cfg["generator"]
        self.model_log_file = model_log_file
        self.model_healthcheck_endpoint = model_healthcheck_endpoint
        self.on_init_function = None
        self.background_task = None
        self.model_backend_load_logs = model_backend_load_logs

    def apt_get(self, package: str):
        self.__onstart_cmd += f"apt-get install -y {package}\n"

    def uv_pip_install(self, package: str):
        if not self.__pip_packages_requested:
            self.__onstart_cmd += "curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv && source .venv/bin/activate\n"
            self.__pip_packages_requested = True

        self.__onstart_cmd += f"uv pip install {package}\n"

    def on_start(self, cmd: str):
        self.__onstart_cmd += f"{cmd}\n"

    def env_vars(self, vars: dict[str, str]):
        self.env_vars.update(vars)

    def __upload_deploy_script(self):
        vast_upload_url = os.environ["VAST_UPLOAD_URL"]
        vast_upload_auth_token = os.environ["VAST_UPLOAD_AUTH_TOKEN"]
        vast_download_url_base = os.environ["VAST_DOWNLOAD_URL"]
        deploy_script = sys.argv[0]
        with open(deploy_script, 'r') as deploy_script_file:
            blob_id = requests.post(vast_upload_url, headers = {'Authorization': f'Bearer {vast_upload_auth_token}'}, data = deploy_script_file.read()).text
        return vast_download_url_base.rstrip('/') + '/' + blob_id
    def __install_remote_worker_script(self):
        worker_script_download_url = self.__upload_deploy_script()
        self.apt_get("wget")
        self.__onstart_cmd += f"""
mkdir -p /workspace
wget -O /workspace/worker.py {worker_script_download_url} && curl -L https://raw.githubusercontent.com/vast-ai/vast-sdk/refs/heads/remote/start_server_sdk.sh | VAST_REMOTE_DISPATCH_MODE=serve bash
        """

    def ready(self):
        if (mode := get_mode()) == "deploy":

            print("Deploying...")
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

            worker_group.create_worker_group()
            #INFO: blocks until worker group is ready
            worker_group.check_worker_group_status()

            print("Deployment Ready!")


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
                    on_load=self.model_backend_load_logs,
                    on_error=["Remote Dispatch error"]
                )
            )

            # Build the worker handling our remote dispatch functions
            remote_worker = Worker(remote_worker_config)

            async def main():
                # Call on_init if present
                if self.on_init_function is not None:
                    try:
                        self.on_init_function()
                    except Exception as ex:
                        raise Exception(f"Error in on_init function: {ex}")

                # Write initial log
                model_log = Path(self.model_log_file)

                await model_log.parent.mkdir(parents=True, exist_ok=True)
                await model_log.write_text("Remote Dispatch ready")
                
                if self.background_task:
                    await asyncio.gather(
                        remote_worker.run_async(),
                        self.background_task(),
                    )
                else:
                    await remote_worker.run_async()

            asyncio.run(main())
        elif mode == "client":
            # Nothing to do here
            pass

