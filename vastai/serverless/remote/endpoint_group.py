from vastai import Serverless


class EndpointGroup:
    WEBSERVER_URL = "https://console.vast.ai"

    def __init__(
        self,
        endpoint_name: str,
        cold_mult: int,
        min_workers: int,
        max_workers: int,
        min_load: int,
        min_cold_load: int,
        target_util: float,
    ):
        self.endpoint_name = endpoint_name
        self.cold_mult = cold_mult
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.min_load = min_load
        self.min_cold_load = min_cold_load
        self.target_util = target_util

    async def create_endpoint_group(self):
        try:
            request_body = {
                "endpoint_name": self.endpoint_name,
                "cold_mult": self.cold_mult,
                "min_load": self.min_load,
        catch Exception as ex:
            raise RuntimeError(f"Failed to create endpoint group: {ex}")
