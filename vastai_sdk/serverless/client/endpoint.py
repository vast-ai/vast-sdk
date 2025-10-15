import aiohttp
from .connection import _make_request

class Endpoint:
    name: str
    id: int

    def __repr__(self):
        return f"<Endpoint {self.name} (id={self.id})>"

    def __init__(self, client, name, id, api_key):
        if client is None:
            raise ValueError("Endpoint cannot be created without client reference")
        if not name:
            raise ValueError("Endpoint name cannot be empty")
        if id is None:
            raise ValueError("Endpoint id cannot be empty")
        self.client = client
        self.name = name
        self.id = id
        self.api_key = api_key

    def request(self, route, payload, serverless_request=None):
        """Forward requests to the parent client."""
        return self.client.queue_endpoint_request(endpoint=self, worker_route=route, worker_payload=payload, serverless_request=serverless_request)
    
    async def _route(self, cost=0.0):
        if self.client is None or not self.client.is_open():
            raise ValueError("Client is invalid")
        try:
            response = await _make_request(
                    client=self.client,
                    url= "http://localhost:8080", #"https://run.vast.ai",, #"https://run-alpha.vast.ai",
                    route="/route/",
                    api_key=self.api_key,
                    body={
                            "endpoint" : self.name,
                            "api_key" : self.api_key,
                            "cost" : cost
                        },
                    method="POST"
                )
        except Exception as ex:
            raise Exception(
                f"Failed to route endpoint:\nReason={ex}"
            )
        return RouteResponse(response)
    
class RouteResponse:
    status: str
    body: dict

    def __repr__(self):
        return f"<RouteResponse status={self.status}>"

    def __init__(self, body: dict):
        if "url" in body.keys():
            self.status = "READY"
            self.body = body
        else:
            self.status = "WAITING"
            self.body = body
            
    def get_url(self):
        return self.body.get("url")
        