import requests
import time


class WorkerGroup:
    WEBSERVER_URL = "https://console.vast.ai"
    AUTOSCALER_URL = "https://run.vast.ai/"

    def __init__(
        self,
        api_key: str,
        cold_mult: int,
        endpoint_id: int,
        endpoint_name: str,
        min_load: int,
        search_params: str,
        target_util: float,
        template_id: int,
    ):
        self.api_key = api_key
        self.cold_mult = cold_mult
        self.endpoint_id = endpoint_id
        self.endpoint_name = endpoint_name
        self.min_load = min_load
        self.search_params = search_params
        self.target_util = target_util
        self.template_id = template_id

    def create_worker_group(self):
        try:
            request_body = {
                "cold_mult": self.cold_mult,
                "endpoint_id": self.endpoint_id,
                "endpoint_name": self.endpoint_name,
                "min_load": self.min_load,
                "search_params": self.search_params,
                "target_util": self.target_util,
                "template_id": self.template_id,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.post(
                url=f"{self.WEBSERVER_URL}/api/v0/autojobs/",
                json=request_body,
                headers=headers,
            )

            return response.json()["id"]
        except Exception as ex:
            raise RuntimeError(f"Failed to create worker group: {ex}")

    def check_worker_group_status(self, max_attempts: int = 60) -> bool:
        try:
            request_body = {"id": self.endpoint_id, "api_key": self.api_key}
            max_attempts = 60
            time.sleep(5) #INFO: ensures autoscaler has the info to be parsed
            for _ in range(max_attempts):
                try:
                    response = requests.post(url=f"{self.AUTOSCALER_URL}/get_endpoint_workers/", json=request_body)
                    response.raise_for_status()

                    workers = response.json()

                    for worker in workers:
                        if worker["status"] == "idle":
                            return True
                except Exception:
                    pass

                time.sleep(5)

            return False
        except Exception as ex:
            raise RuntimeError(f"Failed to check workergroup group status: {ex}")
