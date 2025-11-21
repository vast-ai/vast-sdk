#!/usr/bin/env python3
import os
import asyncio


mode = os.getenv("VAST_REMOTE_DISPATCH_MODE", "client")


def get_mode():
    return mode


class Endpoint:
    # TODO: needs to create a template, then workergroup with the new template's params
    # To create the template, we should use the _make_request
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
        self.name = name
        self.cold_mult = cold_mult
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.min_load = min_load
        self.min_cold_load = min_cold_load
        self.target_util = target_util
        self.image_name = image_name
        self.env_vars = env_vars
        self.search_params = search_params
        self.disk_space = disk_space
        self.__onstart_cmd = ""
        self.__pip_packages_requested = False

    def apt_get(self, package: str):
        self.__onstart_cmd += f"apt-get install -y {package}\n"

    def uv_pip_install(self, package: str):
        if not self.__pip_packages_requested:
            self.__onstart_cmd += "curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv && source .venv/bin/activate\n"
            self.__pip_packages_requested = True

        self.__onstart_cmd += f"uv pip install {package}\n"

    def on_start(self, cmd: str):
        self.__onstart_cmd += f"{cmd}\n"

    async def ready(self):
        if (mode := get_mode()) == "deploy":
            from vastai.serverless.remote.template import Template

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
            print(f"Template ID: {template_id}")

        elif mode == "serve":
            pass
        elif mode == "client":
            pass


async def main():
    ep = Endpoint("test-endpoint")
    await ep.ready()


asyncio.run(main())
