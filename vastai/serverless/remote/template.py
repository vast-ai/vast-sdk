from typing import Optional
import requests
from ..client.connection import _make_request
import os


class Template:
    WEBSERVER_URL = "https://console.vast.ai"

    def __init__(
        self,
        api_key: str,
        image_name: str,
        env_vars: dict,
        disk_space: int,
        template_name: Optional[str] = None,
        onstart_cmd: str = "",
    ):
        self.api_key = api_key
        self.name = template_name if template_name else f"template-{os.urandom(15)}"
        self.image_name = image_name
        self.env_vars = env_vars
        self.disk_space = disk_space
        self.onstart_cmd = onstart_cmd

    def create_template(self):
        try:
            request_body = {
                "name": self.name,
                "image": self.image_name,
                "onstart": self.onstart_cmd,
                "recommended_disk_space": self.disk_space,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}
            if self.env_vars:
                request_body["env"] = self.env_vars

            response = requests.post(
                url=f"{self.WEBSERVER_URL}/api/v0/template/",
                json=request_body,
                headers=headers,
            )

            return response.json()["template"]["id"]
        except Exception as ex:
            raise RuntimeError(f"Failed to create template: {ex}")
