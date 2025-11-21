from typing import Optional
import requests
import os


class Template:
    WEBSERVER_URL = "https://console.vast.ai"

    def __init__(
        self,
        api_key: str,
        image_name: str,
        env_vars: str,
        disk_space: int,
        template_name: Optional[str] = None,
        onstart_cmd: str = "",
        port : int = 3000,
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
                "runtype": "ssh_direc",
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}
            env_vars = self.env_vars if self.env_vars else ""
            env_vars = "-p 3000:3000 " + env_vars

            request_body["env"] = env_vars

            response = requests.post(
                url=f"{self.WEBSERVER_URL}/api/v0/template/",
                json=request_body,
                headers=headers,
            )

            return response.json()["template"]["id"]
        except Exception as ex:
            raise RuntimeError(f"Failed to create template: {ex}")
