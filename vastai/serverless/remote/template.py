from vastai import Serverless
from typing import Optional
from ..client.connection import _make_request
import os


class Template:
    WEBSERVER_URL = "https://console.vast.ai"

    def __init__(
        self,
        client: Serverless,
        api_key: str,
        image_name: str,
        env_vars: dict,
        disk_space: int,
        template_name: Optional[str] = None,
        onstart_cmd: str = "",
    ):
        self.client = client
        self.api_key = api_key
        self.name = template_name if template_name else f"template-{os.urandom(15)}"
        self.image_name = image_name
        self.env_vars = env_vars
        self.disk_space = disk_space
        self.onstart_cmd = onstart_cmd

    async def create_template(self):
        try:
            response = await _make_request(
                client=self.client,
                url=self.WEBSERVER_URL,
                route="/api/v0/template/",
                api_key=self.api_key,
                method="POST",
                body={
                    "name": self.name,
                    "image": self.image_name,
                    "env": self.env_vars,
                    "onstart": self.onstart_cmd,
                    "recommended_disk_space": self.disk_space,
                },
                retries=1,
            )

            return response["template"]["id"]
        except Exception as ex:
            raise RuntimeError(f"Failed to create template: {ex}")
