from .endpoint import Endpoint
from typing import Callable, Optional
import inspect


class Session:
    endpoint: Endpoint
    session_id: str
    open: bool
    url: str
    lifetime: float
    auth_data: dict
    on_close: Optional[Callable]

    def __init__(self, endpoint: Endpoint, session_id: str, lifetime: float, url: str, auth_data: dict):
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
        self.on_close: Optional[Callable[[ "Session" ], object]] = None
        self._closing = False  # guard to ensure one-time execution

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._run_close_hooks()
        return False

    def set_on_close(self, on_close_fn: Callable):
        self.on_close = on_close_fn

    async def is_open(self):
        result = await self.endpoint.session_healthcheck(self)
        self.open = result
        return result

    async def _run_close_hooks(self):
        """
        Execute on_close hook (sync or async) exactly once,
        then close the underlying endpoint session.
        """
        if not self.open or self._closing:
            return

        self._closing = True

        if self.on_close is not None:
            try:
                result = self.on_close(self)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass

        # Close endpoint session
        result = self.endpoint.close_session(self)
        if inspect.isawaitable(result):
            await result

        self.open = False

    def close(self):
        """
        Explicit close for non-async contexts.
        Returns an awaitable if async work is required.
        """
        if not self.open or self._closing:
            return None

        async def _close_async():
            await self._run_close_hooks()

        return _close_async()

    async def request(
        self,
        route,
        payload,
        serverless_request=None,
        cost: int = 100,
        retry: bool = True,
        stream: bool = False,
    ):
        """Forward requests to the endpoint"""
        if not self.open:
            raise ValueError("Cannot make request on closed session.")

        result = await self.endpoint.request(
            route=route,
            payload=payload,
            serverless_request=serverless_request,
            cost=cost,
            retry=retry,
            stream=stream,
            session=self,
        )
        if result.get("status") == 410:
            self.open = False
            raise ValueError("Cannot make request on closed session.")
        return result
