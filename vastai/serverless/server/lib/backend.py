import os
import json
import time
import base64
import subprocess
import dataclasses
import logging
from asyncio import wait, sleep, gather, Semaphore, FIRST_COMPLETED, create_task
from typing import Tuple, Awaitable, NoReturn, List, Union, Callable, Optional
from functools import cached_property
from distutils.util import strtobool
from collections import deque


from anyio import open_file
from aiohttp import web, ClientResponse, ClientSession, ClientConnectorError, ClientTimeout, TCPConnector
import asyncio

import requests
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA

from .metrics import Metrics
from .data_types import (
    AuthData,
    EndpointHandler,
    LogAction,
    ApiPayload_T,
    JsonDataException,
    RequestMetrics,
    BenchmarkResult
)

VERSION = "0.3.0"

MSG_HISTORY_LEN = 100
log = logging.getLogger(__file__)

# defines the minimum wait time between sending updates to autoscaler
LOG_POLL_INTERVAL = 0.1
BENCHMARK_INDICATOR_FILE = ".has_benchmark"
MAX_PUBKEY_FETCH_ATTEMPTS = 3


@dataclasses.dataclass
class Backend:
    """
    This class is responsible for:
    1. Tailing logs and updating load time metrics
    2. Taking an EndpointHandler alongside incoming payload, preparing a json to be sent to the model, and
    sending the request. It also updates metrics as it makes those requests.
    3. Running a benchmark from an EndpointHandler
    """

    model_server_url: str
    model_log_file: str
    benchmark_handler: (
        EndpointHandler  # this endpoint handler will be used for benchmarking
    )
    log_actions: List[Tuple[LogAction, str]]
    reqnum = -1
    version = VERSION
    msg_history = []
    sem: Semaphore = dataclasses.field(default_factory=Semaphore)
    queue: deque = dataclasses.field(default_factory=deque, repr=False)
    unsecured: bool = dataclasses.field(
        default_factory=lambda: bool(strtobool(os.environ.get("UNSECURED", "false"))),
    )
    report_addr: str = dataclasses.field(
        default_factory=lambda: os.environ.get("REPORT_ADDR", "https://run.vast.ai")
    )
    mtoken: str = dataclasses.field(
        default_factory=lambda: os.environ.get("MASTER_TOKEN", "")
    )
    healthcheck_url: str = dataclasses.field(
        default_factory=lambda: os.environ.get("MODEL_HEALTH_ENDPOINT", "")
    )

    def __post_init__(self):
        self.metrics = Metrics()
        self.metrics._set_version(self.version)
        self.metrics._set_mtoken(self.mtoken)
        self._total_pubkey_fetch_errors = 0
        self._pubkey = self._fetch_pubkey()
        self.__start_healthcheck: bool = False

    @property
    def pubkey(self) -> Optional[RSA.RsaKey]:
        if self._pubkey is None:
            self._pubkey = self._fetch_pubkey()
        return self._pubkey

    @cached_property
    def session(self):
        if not self.model_server_url:
            raise ValueError("Attempting to create Session with non-existent model server url. Please specify a model server URL and Port")
        log.debug(f"starting session with {self.model_server_url}")
        connector = TCPConnector(
            force_close=True, # Required for long running jobs
            enable_cleanup_closed=True,
        )
        
        timeout = ClientTimeout(total=None)
        return ClientSession(self.model_server_url, timeout=timeout, connector=connector)

    def create_handler(
        self,
        handler: EndpointHandler[ApiPayload_T],
    ) -> Callable[[web.Request], Awaitable[Union[web.Response, web.StreamResponse]]]:
        async def handler_fn(
            request: web.Request,
        ) -> Union[web.Response, web.StreamResponse]:
            return await self.__handle_request(handler=handler, request=request)

        return handler_fn

    #######################################Private#######################################
    def _fetch_pubkey(self):
        report_addr = self.report_addr.rstrip("/")
        command = ["curl", "-X", "GET", f"{report_addr}/pubkey/"]
        try:
            result = subprocess.check_output(command, universal_newlines=True)
            log.debug("public key:")
            log.debug(result)
            key = RSA.import_key(result)
            if key is not None:
                return key
        except (ValueError , subprocess.CalledProcessError) as e:
            log.debug(f"Error downloading key: {e}")
        self.backend_errored("Failed to get autoscaler pubkey")
       

    async def __handle_request(
        self,
        handler: EndpointHandler[ApiPayload_T],
        request: web.Request,
    ) -> Union[web.Response, web.StreamResponse]:
        """use this function to forward requests to the model endpoint"""
        try:
            data = await request.json()
            auth_data, payload = handler.get_data_from_request(data)
        except JsonDataException as e:
            return web.json_response(data=e.message, status=422)
        except json.JSONDecodeError:
            return web.json_response(dict(error="invalid JSON"), status=422)
        workload = payload.count_workload()
        request_metrics: RequestMetrics = RequestMetrics(request_idx=auth_data.request_idx, reqnum=auth_data.reqnum, workload=workload, status="Created")


        def advance_queue_after_completion(event: asyncio.Event):
            """Pop current head and wake next waiter, if any."""
            # If this event is current head, wake next waiter
            if self.queue and self.queue[0] is event:
                self.queue.popleft()
                if self.queue:
                    self.queue[0].set()
            else:
                # Else, remove it from the queue
                try:
                    self.queue.remove(event)
                except ValueError:
                    pass

        async def cancel_api_call_if_disconnected() -> None:
            await request.wait_for_disconnection()
            log.debug(f"Request with reqnum: {request_metrics.reqnum} was canceled")
            self.metrics._request_canceled(request_metrics)
            return

        async def make_request() -> Union[web.Response, web.StreamResponse]:
            try:
                response = await self.__call_backend(handler=handler, payload=payload)
             
                status_code = response.status
                log.debug(
                    " ".join(
                        [
                            f"request with reqnum:{request_metrics.reqnum}",
                            f"returned status code: {status_code},",
                        ]
                    )
                )
                res = await handler.generate_client_response(request, response)
                self.metrics._request_success(request_metrics)
                return res
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.debug(f"[backend] Request error: {e}")
                self.metrics._request_errored(request_metrics)
                return web.Response(status=500)

        ###########

        if self.__check_signature(auth_data) is False:
            self.metrics._request_reject(request_metrics)
            return web.Response(status=401)
        
        if self.metrics.model_metrics.wait_time > handler.max_queue_time:
            self.metrics._request_reject(request_metrics)
            return web.Response(status=429)

        disconnect_task = create_task(cancel_api_call_if_disconnected())
        next_request_task = None
        work_task = None
        event = asyncio.Event() # Used in finally block, so initialize here

        self.metrics._request_start(request_metrics)

        try:
            if handler.allow_parallel_requests:
                log.debug(f"Starting request for reqnum:{request_metrics.reqnum}")
                work_task = create_task(make_request())
                done, pending = await wait([work_task, disconnect_task], return_when=FIRST_COMPLETED)

                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                if disconnect_task in done:
                    return web.Response(status=499)

                # otherwise work_task completed
                return await work_task
            
            # FIFO-queue branch
            else:
                # Insert a Event into the queue for this request
                # Event.set() == our request is up next
                self.queue.append(event)
                if self.queue and self.queue[0] is event:
                    event.set()

                # Race between our request being next and request being cancelled
                next_request_task = create_task(event.wait())
                first_done, first_pending = await wait(
                    [next_request_task, disconnect_task], return_when=FIRST_COMPLETED
                )

                # If the disconnect task wins the race
                if disconnect_task in first_done:
                    # Clean up the next_request_task, then exit
                    for t in first_pending:
                        t.cancel()
                    await asyncio.gather(*first_pending, return_exceptions=True)
                    return web.Response(status=499)

                # We are the next-up request in the queue
                log.debug(f"Starting work on request {request_metrics.reqnum}...")

                # Race the backend API call with the disconnect task
                work_task = create_task(make_request())

                done, pending = await wait([work_task, disconnect_task], return_when=FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                if disconnect_task in done:
                    return web.Response(status=499)

                # otherwise work_task completed
                return await work_task
            
        except asyncio.CancelledError:
            return web.Response(status=499)
        
        except Exception as e:
            log.debug(f"Exception in main handler loop {e}")
            return web.Response(status=500)
        
        finally:
            if not handler.allow_parallel_requests:
                advance_queue_after_completion(event)

            self.metrics._request_end(request_metrics)
            cleanup_tasks = [t for t in (next_request_task, work_task, disconnect_task) if t]
            for t in cleanup_tasks:
                if not t.done():
                    t.cancel()
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)


    @cached_property  
    def healthcheck_session(self):
        """Dedicated session for healthchecks to avoid conflicts with API session"""
        log.debug("creating dedicated healthcheck session")
        connector = TCPConnector(
            force_close=True,  # Keep this for isolation
            enable_cleanup_closed=True,
        )
        timeout = ClientTimeout(total=10)  # Reasonable timeout for healthchecks
        return ClientSession(timeout=timeout, connector=connector)

    async def __healthcheck(self):
        health_check_url = self.benchmark_handler.healthcheck_endpoint
        if health_check_url is None:
            log.debug("No healthcheck endpoint defined, skipping healthcheck")
            return

        while True:
            await sleep(10)
            if self.__start_healthcheck is False:
                continue
            try:
                log.debug(f"Performing healthcheck on {health_check_url}")
                async with self.healthcheck_session.get(health_check_url) as response:
                    if response.status == 200:
                        log.debug("Healthcheck successful")
                    elif response.status == 503:
                        log.debug(f"Healthcheck failed with status: {response.status}")
                        self.backend_errored(
                            f"Healthcheck failed with status: {response.status}"
                        )
                    else:
                        log.debug(f"Healthcheck Endpoint not ready: {response.status}")
            except Exception as e:
                log.debug(f"Healthcheck failed with exception: {e}")
                self.backend_errored(str(e))

    async def _start_tracking(self) -> None:
        await gather(
            self.__read_logs(), self.metrics._send_metrics_loop(), self.__healthcheck(), self.metrics._send_delete_requests_loop()
        )

    def backend_errored(self, msg: str) -> None:
        self.metrics._model_errored(msg)

    async def __call_backend(self, handler: EndpointHandler[ApiPayload_T], payload: ApiPayload_T
    ) -> ClientResponse:
        if handler.is_remote_dispatch:
            return await self.__call_remote_dispatch(handler=handler, payload=payload)
        else:
            return await self.__call_api(handler=handler, payload=payload)

    async def __call_api(
        self, handler: EndpointHandler[ApiPayload_T], payload: ApiPayload_T
    ) -> ClientResponse:
        api_payload = payload.generate_payload_json()
        log.debug(f"posting to endpoint: '{handler.endpoint}', payload: {api_payload}")
        return await self.session.post(url=handler.endpoint, json=api_payload)

    async def __call_remote_dispatch(
        self, handler: EndpointHandler[ApiPayload_T], payload: ApiPayload_T
    ) -> ClientResponse:
        remote_func_params = payload.generate_payload_json()
        log.debug(f"Calling remote dispatch function on {handler.route} with params {remote_func_params}")
        return await handler.call_remote_dispatch_function(params=remote_func_params)

    def __check_signature(self, auth_data: AuthData) -> bool:
        if self.unsecured is True:
            return True

        def verify_signature(message, signature):
            if self.pubkey is None:
                log.debug(f"No Public Key!")
                return False

            h = SHA256.new(message.encode())
            try:
                pkcs1_15.new(self.pubkey).verify(h, base64.b64decode(signature))
                return True
            except (ValueError, TypeError):
                return False

        message = {
            key: value
            for (key, value) in (dataclasses.asdict(auth_data).items())
            if key != "signature" and key != "__request_id"
        }
        if auth_data.reqnum < (self.reqnum - MSG_HISTORY_LEN):
            log.debug(
                f"reqnum failure, got {auth_data.reqnum}, current_reqnum: {self.reqnum}"
            )
            return False
        elif message in self.msg_history:
            log.debug(f"message: {message} already in message history")
            return False
        elif verify_signature(json.dumps(message, indent=4, sort_keys=True), auth_data.signature):
            self.reqnum = max(auth_data.reqnum, self.reqnum)
            self.msg_history.append(message)
            self.msg_history = self.msg_history[-MSG_HISTORY_LEN:]
            return True
        else:
            log.debug(
                f"signature verification failed, sig:{auth_data.signature}, message: {message}"
            )
            return False

    async def __read_logs(self) -> Awaitable[NoReturn]:

        async def run_benchmark() -> float:
            log.debug("starting benchmark")
            try:
                with open(BENCHMARK_INDICATOR_FILE, "r") as f:
                    log.debug("already ran benchmark")
                    # trigger model load
                    # payload = self.benchmark_handler.make_benchmark_payload()
                    # _ = await self.__call_api(
                    #     handler=self.benchmark_handler, payload=payload
                    # )
                    return float(f.readline())
            except FileNotFoundError:
                pass

            log.debug("Initial run to trigger model loading...")
            payload = self.benchmark_handler.make_benchmark_payload()
            await self.__call_backend(handler=self.benchmark_handler, payload=payload)

            max_throughput = 0
            sum_throughput = 0
            concurrent_requests = self.benchmark_handler.concurrency if self.benchmark_handler.allow_parallel_requests else 1
            for run in range(1, self.benchmark_handler.benchmark_runs + 1):
                start = time.time()
                benchmark_requests = []

                for i in range(concurrent_requests):
                    payload = self.benchmark_handler.make_benchmark_payload()
                    workload = payload.count_workload()
                    task = self.__call_backend(handler=self.benchmark_handler, payload=payload)
                    benchmark_requests.append(
                        BenchmarkResult(request_idx=i, workload=workload, task=task)
                    )

                responses = await gather(*[br.task for br in benchmark_requests])
                for br, response in zip(benchmark_requests, responses):
                    br.response = response

                total_workload = sum(br.workload for br in benchmark_requests if br.is_successful)
                time_elapsed = time.time() - start
                successful_responses = sum([1 for br in benchmark_requests if br.is_successful])
                if successful_responses == 0:
                    self.backend_errored("No successful responses from benchmark")
                    log.debug(f"benchmark failed: {successful_responses}/{concurrent_requests} successful responses")

                throughput = total_workload / time_elapsed
                sum_throughput += throughput
                max_throughput = max(max_throughput, throughput)

                # Log results for debugging
                log.debug(
                    "\n".join(
                        [
                            "#" * 60,
                            f"Run: {run}, concurrent_requests: {concurrent_requests}",
                            f"Total workload: {total_workload}, time_elapsed: {time_elapsed}s",
                            f"Throughput: {throughput} workload/s",
                            f"Successful responses: {successful_responses}/{concurrent_requests}",
                            "#" * 60,
                        ]
                    )
                )

            average_throughput = sum_throughput / self.benchmark_handler.benchmark_runs
            log.debug(
                f"benchmark result: avg {average_throughput} workload per second, max {max_throughput}"
            )
            with open(BENCHMARK_INDICATOR_FILE, "w") as f:
                f.write(str(max_throughput))
            return max_throughput

        async def handle_log_line(log_line: str) -> None:
            """
            Implement this function to handle each log line for your model.
            This function should mutate self.system_metrics and self.model_metrics
            """
            for action, msg in self.log_actions:
                match action:
                    case LogAction.ModelLoaded if msg in log_line:
                        log.debug(
                            f"Got log line indicating model is loaded: {log_line}"
                        )
                        # some backends need a few seconds after logging successful startup before
                        # they can begin accepting requests
                        # await sleep(5)
                        try:
                            max_throughput = await run_benchmark()
                            self.__start_healthcheck = True
                            self.metrics._model_loaded(
                                max_throughput=max_throughput,
                            )
                        except ClientConnectorError as e:
                            log.debug(
                                f"failed to connect to comfyui api during benchmark"
                            )
                            self.backend_errored(str(e))
                    case LogAction.ModelError if msg in log_line:
                        log.debug(f"Got log line indicating error: {log_line}")
                        self.backend_errored(msg)
                        break
                    case LogAction.Info if msg in log_line:
                        log.debug(f"Info from model logs: {log_line}")

        async def tail_log():
            log.debug(f"tailing file: {self.model_log_file}")
            async with await open_file(self.model_log_file, encoding='utf-8', errors='ignore') as f:
                while True:
                    line = await f.readline()
                    if line:
                        await handle_log_line(line.rstrip())
                    else:
                        await asyncio.sleep(LOG_POLL_INTERVAL)

        ###########

        while True:
            if os.path.isfile(self.model_log_file) is True:
                return await tail_log()
            else:
                await sleep(1)
