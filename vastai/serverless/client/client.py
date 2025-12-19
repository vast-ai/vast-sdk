# client.py
from .connection import _make_request
from .endpoint import Endpoint
from .worker import Worker
from .session import Session
import asyncio
import aiohttp
import ssl
import os
import tempfile
import logging
import random
import time
import collections
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Union, List

class ServerlessRequest(asyncio.Future):
    """A request to a Serverless endpoint managed by the client"""
    def __init__(self):
        super().__init__()
        self.status = "New"
        self.create_time = time.time()
        self.start_time = None
        self.complete_time = None
        self.req_idx = 0

    def then(self, callback) -> "ServerlessRequest":
        def _done(fut):
            if fut.exception() is not None:
                print(fut.exception())
                return
            callback(fut.result())
        self.add_done_callback(_done)
        return self

class Serverless:
    SSL_CERT_URL        = "https://console.vast.ai/static/jvastai_root.cer"
    VAST_WEB_URL        = "https://console.vast.ai"
    VAST_SERVERLESS_URL = "https://run.vast.ai"

    def __init__(
        self,
        api_key: Optional[str] = os.environ.get("VAST_API_KEY", None),
        *,
        debug: bool = False,
        instance: str = "prod",
        connection_limit: int = 500,
        default_request_timeout: float = 600.0,
        max_poll_interval: float = 15.0
    ):
        if api_key is None or api_key == "":
            raise AttributeError("API key missing. Please set VAST_API_KEY in your environment variables.")
        self.api_key = api_key

        match instance:
            case "prod":
                self.autoscaler_url = "https://run.vast.ai"
            case "alpha":
                self.autoscaler_url = "https://run-alpha.vast.ai"
            case "candidate":
                self.autoscaler_url = "https://run-candidate.vast.ai"
            case "local":
                self.autoscaler_url = "http://localhost:8080"
            case _:
                self.autoscaler_url = "https://run.vast.ai"

        self.latencies = collections.deque(maxlen=50)
        self.debug = debug
        self.default_request_timeout = float(default_request_timeout)
        self.max_poll_interval = float(max_poll_interval)
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

        if debug:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)

            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

            # If we attach our own handler, avoid double-logging via root handlers.
            self.logger.propagate = False
        else:
            # Let the application decide if/where logs go.
            self.logger.propagate = True

        self.connection_limit = connection_limit
        self._session: aiohttp.ClientSession | None = None
        self._ssl_context: ssl.SSLContext | None = None

    async def __aenter__(self):
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self.logger.info("Started aiohttp ClientSession")
            connector = aiohttp.TCPConnector(limit=self.connection_limit, ssl=await self.get_ssl_context())
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    def is_open(self):
        return self._session is not None and not self._session.closed

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info("Closed aiohttp ClientSession")

    async def get_ssl_context(self) -> ssl.SSLContext:
        """Download Vast.ai root cert and build SSL context (cached)."""
        if self._ssl_context is None:
            async with aiohttp.ClientSession() as s:
                async with s.get(self.SSL_CERT_URL) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to fetch SSL cert: {resp.status}")
                    cert_bytes = await resp.read()

            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".cer")
            tmpfile.write(cert_bytes)
            tmpfile.close()

            ctx = ssl.create_default_context()
            ctx.load_verify_locations(cafile=tmpfile.name)
            self.logger.info("Loaded Vast.ai SSL certificate")

            self._ssl_context = ctx
            os.unlink(tmpfile.name)

        return self._ssl_context

    async def get_endpoint(self, name="") -> Endpoint:
        endpoints = await self.get_endpoints()
        for e in endpoints:
            if e.name == name:
                return e
        raise Exception(f"Endpoint {name} could not be found")

    async def get_endpoints(self) -> list[Endpoint]:
        try:
            result = await _make_request(
                client=self,
                url=self.VAST_WEB_URL,
                route="/api/v0/endptjobs/",
                api_key=self.api_key,
                params={"client_id": "me"}
            )
        except Exception as ex:
            raise Exception(f"Failed to get endpoints:\nReason={ex}")

        if not result.get("ok"):
            raise Exception(f"Failed to get endpoints: HTTP {result.get('status')} - {result.get('text','')[:512]}")

        response = result.get("json") or {}
        endpoints = []
        for e in response.get("results", []):
            endpoints.append(Endpoint(client=self, name=e["endpoint_name"], id=e["id"], api_key=e["api_key"]))
        self.logger.info(f"Found {len(endpoints)} endpoints")
        return endpoints

    async def get_endpoint_workers(self, endpoint: Endpoint) -> List[Worker]:
        if not isinstance(endpoint, Endpoint):
            raise ValueError("endpoint must be an Endpoint")

        url = f"{self.autoscaler_url}/get_endpoint_workers/"
        payload = {"id": endpoint.id, "api_key": self.api_key}

        async with self._session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"get_endpoint_workers failed: HTTP {resp.status} - {text}")

            data = await resp.json(content_type=None)
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected response type (wanted list): {type(data)}")

            return [Worker.from_dict(item) for item in data]

    async def get_endpoint_session(
        self,
        endpoint,
        session_id: int,
        session_auth: str
    ):
        try:
            result = await _make_request(
                client=self,
                url=session_auth.get("url"),
                api_key="",
                route="/session/get",
                body={"session_id": session_id, "session_auth": session_auth},
                method="POST",
                retries=1,
                timeout=600,
                stream=False,
            )

            if not result.get("ok"):
                self.logger.info(
                    f"get_session returned HTTP {result.get('status')}: {result.get('text','')[:256]}"
                )
                return None

            worker_response = result.get("json") or {}

            session = Session(
                endpoint=endpoint,
                session_id=session_id,
                lifetime=worker_response.get("lifetime"),
                expiration=worker_response.get("expiration"),
                auth_data=worker_response.get("auth_data"),
                url=worker_response.get("auth_data").get("url")
            )
            return session

        except Exception as ex:
            self.logger.info(f"Got error message from get_session: {ex}")
            return None

    async def end_endpoint_session(
        self,
        endpoint: Endpoint,
        session: Session
    ):
        session_end_response = await self.queue_endpoint_request(
            endpoint=endpoint,
            worker_route="/session/end",
            worker_payload={"session_id": session.session_id},
            session=session
        )
        if session_end_response.get("response").get("ended"):
            return
        else:
            raise Exception(f"Failed to end session: {session_end_response.get('response')}")

    async def start_endpoint_session(
        self,
        endpoint: Endpoint,
        cost: int = 100,
        lifetime: float = 60
    ) -> Session:
        session_start_response = await self.queue_endpoint_request(
            endpoint=endpoint,
            worker_route="/session/create",
            worker_payload={"lifetime": lifetime},
            cost=cost,
        )
        session_id = session_start_response.get("response").get("session_id")
        lifetime = session_start_response.get("response").get("lifetime")
        expiration = session_start_response.get("response").get("lifetime")
        url = session_start_response.get("url")
        auth_data = session_start_response.get("auth_data")
        if session_id:
            return Session(endpoint=endpoint, session_id=session_id, lifetime=lifetime, expiration=expiration, url=url, auth_data=auth_data)
        else:
            raise Exception(f"Failed to create session: {session_start_response['response']}")

    def queue_endpoint_request(
        self,
        endpoint: Endpoint,
        worker_route: str,
        worker_payload: dict,
        session: Session = None,
        serverless_request: Optional[ServerlessRequest] = None,
        cost: int = 100,
        max_wait_time: Optional[float] = None,
        retry: bool = True,
        max_retries: int = None,
        stream: bool = False
    ) -> ServerlessRequest:
        """Return a Future that will resolve once the request completes."""
        if serverless_request is None:
            serverless_request = ServerlessRequest()

        async def task(request: ServerlessRequest):
            request_idx: int = 0
            total_attempts = 0
            try:
                while True:
                    total_attempts += 1
                    request.status = "Queued"
                    worker_url = ""
                    auth_data = {}
                    session_id = None

                    if session is None:
                        self.logger.debug("Sending initial route call")

                        route = await endpoint._route(cost=cost, req_idx=request_idx, timeout=self.get_avg_request_time())

                        poll_interval = 1
                        elapsed_time = 0
                        attempt = 0
                        while route.status != "READY":
                            request.status = "Polling"
                            if max_wait_time is not None and elapsed_time >= max_wait_time:
                                raise asyncio.TimeoutError("Timed out waiting for worker to become ready")

                            await asyncio.sleep(poll_interval)
                            elapsed_time += poll_interval

                            route = await endpoint._route(cost=cost, req_idx=request_idx, timeout=self.get_avg_request_time())

                            attempt += 1
                            poll_interval = min((2 ** attempt) + random.uniform(0, 1), self.max_poll_interval)
                            self.logger.debug("Polling route...")

                        worker_url = route.get_url()
                        auth_data = route.body
                    else:
                        if session.url is not None:
                            worker_url = session.url
                            auth_data = session.auth_data
                            session_id = session.session_id

                    payload = worker_payload
                    worker_request_body = {
                        "auth_data": auth_data,
                        "session_id": session_id,
                        "payload": payload
                    }

                    self.logger.debug("Found worker machine, starting work")
                    if request.status != "Retrying":
                        request.status = "In Progress"
                        request.start_time = time.time()

                    # Transport/JSON failures may raise; HTTP errors return ok=False.
                    result = await _make_request(
                        client=self,
                        url=worker_url,
                        route=worker_route,
                        api_key=endpoint.api_key,
                        body=worker_request_body,
                        method="POST",
                        retries=1,  # avoid stacking retries with the outer loop
                        timeout=600,
                        stream=stream
                    )

                    if not result.get("ok"):
                        # Retry decision is now *outside* _make_request()
                        if retry and result.get("retryable") and (max_retries is None or total_attempts < max_retries):
                            request.status = "Retrying"
                            await asyncio.sleep(min((2 ** total_attempts) + random.uniform(0, 1), self.max_poll_interval))
                            continue

                        # Return the raw HTTP result to the caller
                        request.status = "Complete"
                        request.complete_time = time.time()
                        if request.start_time is not None:
                            self.latencies.append(request.complete_time - request.start_time)

                        response = {
                            "response": result.get("json") if result.get("json") is not None else {"error": result.get("text", "")},
                            "result": result,
                            "latency": (request.complete_time - request.start_time) if request.start_time else None,
                            "url": worker_url,
                            "request_idx": request_idx,
                            "auth_data": auth_data
                        }
                        request.set_result(response)
                        return

                    # Success
                    worker_response = result.get("stream") if stream else result.get("json")

                    request.status = "Complete"
                    request.complete_time = time.time()
                    self.latencies.append(request.complete_time - request.start_time)
                    self.logger.info("Endpoint request task completed")

                    response = {
                        "response": worker_response,
                        "result": result,
                        "latency": request.complete_time - request.start_time,
                        "url": worker_url,
                        "request_idx": request_idx,
                        "auth_data": auth_data
                    }
                    request.set_result(response)
                    return

            except asyncio.CancelledError:
                request.status = "Cancelled"
                request.set_exception(asyncio.CancelledError())
                return
            except Exception as ex:
                request.status = "Errored"
                request.set_exception(ex)
                self.logger.error(f"Request errored: {ex}")
                return

        bg_task = asyncio.create_task(task(serverless_request))

        def _propagate_cancel(fut: ServerlessRequest):
            if fut.cancelled():
                bg_task.cancel()

        serverless_request.add_done_callback(_propagate_cancel)
        self.logger.info("Queued endpoint request")
        return serverless_request
