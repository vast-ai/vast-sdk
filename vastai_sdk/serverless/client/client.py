from .connection import _make_request
from .endpoint import Endpoint
import asyncio
import aiohttp
import ssl
import os
import tempfile
import logging
import random

class ServerlessRequest(asyncio.Future):
    """A promise-like Future that also supports .then()."""
    def __init__(self):
        super().__init__()
        self.on_work_start_callbacks = []
        self.status = "New"

    def then(self, callback):
        def _done(fut):
            if fut.exception() is not None:
                print(fut.exception())
                return
            callback(fut.result())
        self.add_done_callback(_done)
        return self
    
    def add_on_work_start_callback(self, callback):
        """Register a callback (sync or async) that will be invoked when work starts."""
        self.on_work_start_callbacks.append(callback)
    
    async def trigger_on_work_start(self):
        """Run the registered callback when worker starts."""
        if len(self.on_work_start_callbacks) > 0:
            for cb in self.on_work_start_callbacks:
                if asyncio.iscoroutinefunction(cb):
                    await cb()
                else:
                    cb()

class Serverless:
    SSL_CERT_URL = "https://console.vast.ai/static/jvastai_root.cer"

    def __init__(self, api_key: str, debug=False, instance="prod"):
        if api_key is None or api_key == "":
            raise ValueError("api_key cannot be empty")
        self.api_key = api_key
        match instance:
            case "prod":
                self.autoscaler_url = "https://run.vast.ai"
            case "alpha":
                self.autoscaler_url = "https://run-alpha.vast.ai"
            case "local":
                self.autoscaler_url = "http://localhost:8080"
            case _:
                self.autoscaler_url = "https://run.vast.ai"


        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.debug:
            # Only set up logging if debug is True
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            # If debug is False, disable logging
            self.logger.addHandler(logging.NullHandler())


        self._session: aiohttp.ClientSession | None = None
        self._ssl_context: ssl.SSLContext | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self.logger.info("Started aiohttp ClientSession")
            self._session = aiohttp.ClientSession()
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
            # Download the Vast root cert
            async with aiohttp.ClientSession() as s:
                async with s.get(self.SSL_CERT_URL) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to fetch SSL cert: {resp.status}")
                    cert_bytes = await resp.read()

            # Write to a temporary PEM file
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".cer")
            tmpfile.write(cert_bytes)
            tmpfile.close()

            # Start with system defaults
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
            response = await _make_request(
                client=self,
                route="/api/v0/endptjobs/",
                api_key=self.api_key,
                body={"client_id": "me"}
            )
        except Exception as ex:
            raise Exception(
                f"Failed to get endpoints:\nReason={ex}"
            )
        endpoints = []
        for e in response["results"]:
            endpoints.append(Endpoint(client=self, name=e["endpoint_name"], id=e["id"], api_key=e["api_key"]))
        self.logger.info(f"Found {len(endpoints)} endpoints")
        return endpoints
    
    def queue_endpoint_request(self, endpoint: Endpoint, worker_route: str, worker_payload: dict, serverless_request: ServerlessRequest = None):
        """Return a Future that will resolve once the request completes."""
        if serverless_request is None:
            serverless_request = ServerlessRequest()

        async def task(request: ServerlessRequest):
            try:
                if request.status != "Retrying":
                    request.status = "Queued"

                # Initial high-cost route to wake up stopped workers
                route = await endpoint._route(cost=100)
                self.logger.debug("Sending initial route call")

                poll_interval = 1
                elapsed_time = 0
                attempt = 0
                while route.status != "READY":
                    if request.status != "Retrying":
                        request.status = "Polling"
                    #  if elapsed_time >= max_wait_time:
                    #     raise TimeoutError("Timed out waiting for worker to become ready")
                    await asyncio.sleep(poll_interval)
                    # Call route with no cost to poll endpoint status
                    route = await endpoint._route()
                    # exponential backoff + jitter
                    poll_interval = min(2 ** attempt + random.uniform(0, 1), 30)
                    attempt += 1
                    self.logger.debug("Sending polling route call...")

                self.logger.debug("Found worker machine, starting work")
                if request.status != "Retrying":
                    request.status = "In Progress"

                # Trigger the on_work_start callback
                await request.trigger_on_work_start()

                # Now, route is ready for sending request to worker
                worker_url = route.get_url()
                auth_data = route.body
                payload = worker_payload
                worker_request_body = {
                                        "auth_data" : auth_data,
                                        "payload" : payload
                                    }
                
                try:
                    worker_response = await _make_request(
                        client=self,
                        url=worker_url,
                        route=worker_route,
                        api_key=endpoint.api_key,
                        body=worker_request_body,
                        method="POST",
                        retries=1
                    )
                except Exception as ex:
                    raise Exception(
                        f"ERROR: Worker request failed:\nReason={ex}"
                    )
                
                response = {
                    "response" : worker_response,
                    "url" : worker_url
                }

                # Resolve future, task complete 
                request.status = "Complete"
                self.logger.info("Endpoint request task completed")
                request.set_result(response)
                return


            except Exception as ex:
                request.status = "Errored"
                self.logger.error(f"Error on worker response, retrying: {ex}")
                return
                    #request.set_exception(ex)

        # Create asyncio task for request lifetime management
        asyncio.create_task(task(serverless_request))
        self.logger.info("Queued endpoint request")
        return serverless_request