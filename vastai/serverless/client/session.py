from .endpoint import Endpoint

class Session:
    endpoint: Endpoint
    session_id: int
    open: bool
    url: str
    lifetime: float
    auth_data: dict

    def __init__(self, endpoint: Endpoint, session_id: int, lifetime: float, url: str, auth_data: dict):
        if endpoint is None:
            raise ValueError("Session cannot be created with empty endpoint")
        if session_id is None:
            raise ValueError("Session cannot be created with empty session_id")
        if url is None:
            raise ValueError("Session cannot be created with empty url")
        self.endpoint = endpoint
        self.session_id = session_id
        self.lifetime = lifetime
        self.url = url
        self.auth_data = auth_data
        self.open = True

    def close(self):
        self.open = False
        return self.endpoint.close_session(self)

    def request(self, route, payload, serverless_request=None, cost: int = 100, retry: bool = True, stream: bool = False):
        """Forward requests to the endpoint"""
        if self.open:
            return self.endpoint.request(
                route=route,
                payload=payload,
                serverless_request=serverless_request,
                cost=cost,
                retry=retry,
                stream=stream,
                session=self
            )
        else:
            raise ValueError("Cannot make request on closed session.")
