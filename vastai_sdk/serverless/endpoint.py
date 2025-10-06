from .connection import _make_request
import asyncio

class ServerlessRequest(asyncio.Future):
    """A promise-like Future that also supports .then()."""
    def __init__(self):
        super().__init__()
        self.on_work_start_callbacks = []
        self.status = "New"

    def then(self, callback: callable) -> "ServerlessRequest":
        def _done(fut):
            try:
                callback(fut.result())
            except Exception as e:
                print(f"Callback error: {e}")
        self.add_done_callback(_done)
        return self

    def add_on_work_start_callback(self, callback: callable) -> None:
        """Register a callback (sync or async) that will be invoked when work starts."""
        self.on_work_start_callbacks.append(callback)

    async def trigger_on_work_start(self) -> None:
        """Run the registered callback when worker starts."""
        for cb in self.on_work_start_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb()
                else:
                    cb()
            except Exception as e:
                print(f"Callback error: {e}")

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

    def request(self, route: str, payload: dict, serverless_request: ServerlessRequest = None) -> ServerlessRequest:
        """Return a Future that will resolve once the request completes."""
        if serverless_request is None:
            serverless_request = ServerlessRequest()

        async def task(request: ServerlessRequest):
            try:
                request.status = "Queued"

                # Wait for worker to be ready
                route_response = await self._wait_for_worker()
                worker_url = route_response.get_url()

                self.client.logger.info("Found worker machine, starting work")
                request.status = "In Progress"

                # Trigger the on_work_start callback
                await request.trigger_on_work_start()

                # Now, route is ready for sending request to worker
                worker_request_body = {
                    "auth_data": route_response.body,
                    "payload": payload
                }

                try:
                    worker_response = await _make_request(
                        client=self.client,
                        url=worker_url,
                        route=route,
                        api_key=self.api_key,
                        body=worker_request_body,
                        method="POST"
                    )
                except Exception as ex:
                    raise Exception(
                        f"ERROR: Worker request failed:\nReason={ex}"
                    )
                
                # Resolve future, task complete
                request.set_result(worker_response)
                request.status = "Complete"
                self.client.logger.info("Endpoint request task completed")

            except Exception as e:
                self.client.logger.error(f"Request failed: {e}")
                request.set_exception(e)

        # Create asyncio task for request lifetime management
        asyncio.create_task(task(serverless_request))
        self.client.logger.info("Queued endpoint request")
        return serverless_request
    
    async def _wait_for_worker(self, max_wait_time: int = 300):
        """Poll endpoint until a worker is ready, return RouteResponse."""
        # Initial high-cost route to wake up stopped workers
        route = await self._route(cost=0)
        self.client.logger.info("Sending initial route call")

        initial_poll_interval = 1
        max_poll_interval = 10
        poll_interval = initial_poll_interval
        elapsed_time = 0

        if route.status != "READY":
            # Call route with high cost to wake up a worker
            self.client.logger.info("No workers present, attempting to wake up stopped worker")
            route = await self._route(cost=100)

            while route.status != "READY":
                if elapsed_time >= max_wait_time:
                    raise TimeoutError("Timed out waiting for worker to become ready")
                
                self.client.logger.info(f"Waiting {poll_interval}s before next poll... ({elapsed_time}s elapsed)")
                await asyncio.sleep(poll_interval)
                elapsed_time += poll_interval
                
                # Exponential backoff
                poll_interval = min(poll_interval * 2, max_poll_interval)
                
                # Call route with no cost to poll endpoint status
                route = await self._route()
                self.client.logger.info("Sending polling route call...")

        return route
    
    async def _route(self, cost=0.0):
        if self.client is None or not self.client.is_open():
            raise ValueError("Client is invalid")
        try:
            response = await _make_request(
                    client=self.client,
                    url="https://run-alpha.vast.ai",
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