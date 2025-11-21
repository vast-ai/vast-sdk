#!/usr/bin/env python3
import os
import asyncio
from anyio import Path
from vastai.serverless.remote.endpoint_group import EndpointGroup
from vastai.serverless.remote.worker_group import WorkerGroup


mode = os.getenv("VAST_REMOTE_DISPATCH_MODE", "client")


def get_mode():
    return mode


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
        self.remote_dispatch_functions = {}
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

    async def ready(self):
        mode = get_mode()
        if mode == "deploy":
            from vastai.serverless.remote.template import Template

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
            print("endpoint_group_id:", endpoint_group_id)

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

            print("worker_group_id:", worker_group.create_worker_group())

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
            # Create parent directories if they don't exist
            await model_log.parent.mkdir(parents=True, exist_ok=True)
            await model_log.write_text("Remote Dispatch ready")

            # Enter the background task if present
            if self.background_task:
                await asyncio.gather(worker_task, self.background_task())
            else:
                await worker_task
        elif mode == "client":
            pass

