import os
import json
import time
import base64
import subprocess
import dataclasses
import logging
from asyncio import wait, sleep, gather, Semaphore, FIRST_COMPLETED, create_task
from typing import Tuple, Awaitable, NoReturn, List, Union, Callable, Optional, Any, Dict
from functools import cached_property
from distutils.util import strtobool
from collections import deque
from asyncio import sleep, CancelledError

from anyio import open_file
from aiohttp import web, ClientResponse, ClientSession, ClientConnectorError, ClientTimeout, TCPConnector
import asyncio
import string
import random

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
    BenchmarkResult,
    Session
)

VERSION = "1.0.1"

MSG_HISTORY_LEN = 100
log = logging.getLogger(__file__)

# defines the minimum wait time between sending updates to autoscaler
LOG_POLL_INTERVAL = 0.1
# Defines waiting interval for session garbage collection
SESSION_GC_INTERVAL = 5.0
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
    _sessions_lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock, repr=False)
    sessions: Dict[str, Session] = dataclasses.field(default_factory=dict)
    session_metrics: Dict[str, RequestMetrics] = dataclasses.field(default_factory=dict)
    max_sessions: int = dataclasses.field(default=-1)
        
    async def session_health_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            session_id = data.get("session_id")
            session_auth = data.get("session_auth")
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=422)

        if not session_id:
            return web.json_response({"error": "missing session_id"}, status=422)

        async with self._sessions_lock:
            session = self.sessions.get(session_id)
            if session is None:
                return web.json_response({"ok": False}, status=200)

            if session_auth is None or session.auth_data != session_auth:
                return web.json_response({"error": "session_auth is not valid"}, status=401)

            return web.json_response({"ok": True}, status=200)
        
    async def session_get_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            session_id = data.get("session_id")
            session_auth = data.get("session_auth")
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=422)

        if not session_id:
            return web.json_response({"error": "missing session_id"}, status=422)
        session = None
        async with self._sessions_lock:
            session = self.sessions.get(session_id)
        if session is None:
            return web.json_response({"error": "session does not exist"}, status=400)
        
        if session_auth is None or session.auth_data != session_auth:
            return web.json_response({"error": "session_auth is not valid"}, status=401)
        else:
            return web.json_response(
                {
                    "session_id" : session_id,
                    "auth_data" : session.auth_data,
                    "lifetime" : session.lifetime,
                    "expiration" : session.expiration,
                    "created_at": session.created_at
                }
            )

    async def __close_session(self, session_id: str) -> bool:
        """
        Close a session and cancel all in-flight tasks associated with it.
        Returns True if a session was removed, False if it didn't exist.
        """
        async with self._sessions_lock:
            session = self.sessions.pop(session_id, None)
            if session is None:
                return False

            session.cancel_event.set()

            for req in list(session.requests):
                try:
                    tr = getattr(req, "transport", None)
                    if tr is not None and not tr.is_closing():
                        tr.close()
                except Exception:
                    pass
            session.requests.clear()

            request_metrics = self.session_metrics.pop(session_id, None)

        # metrics outside lock
        if request_metrics is not None:
            self.metrics._request_success(request_metrics)
            self.metrics._request_end(request_metrics)

        return True


    async def session_end_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            session_id = data.get("session_id")
            session_auth = data.get("session_auth")
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=422)

        if not session_id:
            return web.json_response({"error": "missing session_id"}, status=422)
        
        session = None
        async with self._sessions_lock:
            session = self.sessions.get(session_id)

            if session is None:
                return web.json_response(
                    {"error": f"session with id {session_id} not found"},
                    status=400,
                )
            if session_auth is None or session.auth_data != session_auth:
                return web.json_response({"error": "session_auth is not valid"}, status=401)
                    
        closed = await self.__close_session(session_id)
        if not closed:
            return web.json_response({"error": "session already closed"}, status=410)

        return web.json_response(
            {"ended": True, "removed_session": session_id},
            status=200,
        )


    def generate_session_id(self):
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choices(characters, k=13))
        return random_string


    async def session_create_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            auth_data = data.get("auth_data")
            payload = data.get("payload")
        except json.JSONDecodeError:
            return web.json_response(dict(error="invalid JSON"), status=422)

        session_request_metrics = RequestMetrics(
            request_idx= auth_data.get("request_idx"),
            reqnum= auth_data.get("reqnum"),
            workload= auth_data.get("cost"),
            status="SessionActive"
        )

        async with self._sessions_lock:
            if self.max_sessions > 0 and len(self.sessions) >= self.max_sessions:
                self.metrics._request_reject(session_request_metrics)
                return web.Response(status=429)
            
            # Set the session expiration time, and the TTL extension per request/get
            lifetime = payload.get("lifetime", 60.0)
            now = time.time()
            expiration = now + lifetime

            session_id = self.generate_session_id()
            session = Session(
                session_id=session_id,
                lifetime=lifetime,
                expiration=expiration,
                auth_data=auth_data
            )
            self.sessions[session_id] = session
            self.session_metrics[session_id] = session_request_metrics
            self.metrics._request_start(session_request_metrics)

        return web.json_response(
            {
                "session_id": session.session_id,
                "expiration": session.expiration
            },
            status=201,
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
        log.debug(f"Starting TCP session with model server at {self.model_server_url}")
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
            log.debug("Serverless Public Key:")
            log.debug(result)
            key = RSA.import_key(result)
            if key is not None:
                return key
        except (ValueError , subprocess.CalledProcessError) as e:
            log.debug(f"Error downloading key: {e}")
        self.backend_errored("Failed to get Serverless public key")
       

    async def __handle_request(
        self,
        handler: EndpointHandler[ApiPayload_T],
        request: web.Request,
    ) -> Union[web.Response, web.StreamResponse]:
        """use this function to forward requests to the model endpoint"""
        try:
            data = await request.json()
            auth_data, payload, session_id = handler.get_data_from_request(data)
        except JsonDataException as e:
            return web.json_response(data=e.message, status=422)
        except json.JSONDecodeError:
            return web.json_response(dict(error="invalid JSON"), status=422)
        workload = payload.count_workload()
        request_metrics: RequestMetrics = RequestMetrics(request_idx=auth_data.request_idx, reqnum=auth_data.reqnum, workload=workload, status="Created")

        event = asyncio.Event()
        finished = asyncio.Event()

        session = None
        if session_id is not None:
            async with self._sessions_lock:
                session = self.sessions.get(session_id)
                if session is None:
                    return web.json_response(dict(error="invalid session"), status=410)
                session.expiration += session.lifetime
                session.requests.append(request)
        
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
            if not finished.is_set():
                self.metrics._request_canceled(request_metrics)
        
        async def cancel_if_session_closed() -> None:
            await session.cancel_event.wait()

        async def make_request() -> Union[web.Response, web.StreamResponse]:
            try:
                response = await self.__call_backend(handler=handler, payload=payload)
                res = await handler.generate_client_response(request, response)
                self.metrics._request_success(request_metrics)
                return res
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.metrics._request_errored(request_metrics, str(e))
                return web.Response(status=500)

        ###########

        if self.__check_signature(auth_data) is False:
            self.metrics._request_reject(request_metrics)
            return web.Response(status=401)
        
        if handler.max_queue_time is not None and self.metrics.model_metrics.wait_time > handler.max_queue_time:
            self.metrics._request_reject(request_metrics)
            return web.Response(status=429)
        
        disconnect_task = create_task(cancel_api_call_if_disconnected())
        session_cancel_task = None
        if session is not None:
            session_cancel_task = create_task(cancel_if_session_closed())

        next_request_task = None
        work_task = None

        self.metrics._request_start(request_metrics, session)

        try:
            if handler.allow_parallel_requests:
                work_task = create_task(make_request())

                wait_set = [work_task, disconnect_task]
                if session_cancel_task is not None:
                    wait_set.append(session_cancel_task)

                done, pending = await wait(
                    wait_set,
                    return_when=FIRST_COMPLETED,
                )

                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                if disconnect_task in done or (session_cancel_task is not None and session_cancel_task in done):
                    return web.Response(status=499)  # request cancelled

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

                wait_set = [next_request_task, disconnect_task]
                if session_cancel_task is not None:
                    wait_set.append(session_cancel_task)

                first_done, first_pending = await wait(
                    wait_set,
                    return_when=FIRST_COMPLETED,
                )
                # If the disconnect task wins the race
                if disconnect_task in first_done or (session_cancel_task is not None and session_cancel_task in first_done):
                    # Clean up the next_request_task, then exit
                    for t in first_pending:
                        t.cancel()
                    await asyncio.gather(*first_pending, return_exceptions=True)
                    return web.Response(status=499)

                # We are the next-up request in the queue
                if session is not None:
                    log.debug(f"Starting work on request {request_metrics.reqnum}")

                # Race the backend API call with the disconnect task
                work_task = create_task(make_request())

                wait_set = [work_task, disconnect_task]
                if session_cancel_task is not None:
                    wait_set.append(session_cancel_task)

                done, pending = await wait(
                    wait_set,
                    return_when=FIRST_COMPLETED,
                )

                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                if disconnect_task in done or (session_cancel_task is not None and session_cancel_task in done):
                    return web.Response(status=499)

                # otherwise work_task completed
                return await work_task

        except asyncio.CancelledError:
            return web.Response(status=499)
        
        except Exception as e:
            log.debug(f"Exception in main handler loop {e}")
            return web.Response(status=500)
        
        finally:
            # Set finished flag so we don't cancel after completion
            finished.set()

            if session is not None and session_id is not None:
                async with self._sessions_lock:
                    s = self.sessions.get(session_id)
                    if s is not None:
                        try:
                            s.requests.remove(request)
                        except ValueError:
                            pass

            if not handler.allow_parallel_requests:
                advance_queue_after_completion(event)
            self.metrics._request_end(request_metrics, session)
            cleanup_tasks = [t for t in (next_request_task, work_task, disconnect_task, session_cancel_task) if t]
            for t in cleanup_tasks:
                if not t.done():
                    t.cancel()
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    async def __healthcheck(self) -> None:
        """
        Periodically hit the healthcheck endpoint using the same session
        configuration as normal API calls.
        """
        health_check_url = self.healthcheck_url
        if not health_check_url:
            log.debug("No healthcheck endpoint defined, skipping healthcheck")
            return

        # Per-request timeout for healthchecks
        timeout = ClientTimeout(total=10)

        while True:
            try:
                await sleep(10)

                if not self.__start_healthcheck:
                    continue

                log.debug(f"Performing healthcheck on {health_check_url}")

                async with self.session.get(
                    health_check_url,
                    timeout=timeout,
                ) as response:
                    status = response.status

                    if status == 200:
                        log.debug("Healthcheck successful")
                    elif status == 503:
                        msg = f"Healthcheck failed with status: {status}"
                        log.debug(msg)
                        self.backend_errored(msg)
                    else:
                        log.debug(f"Healthcheck endpoint not ready: {status}")

            except CancelledError:
                log.debug("Healthcheck task cancelled; exiting loop")
                break

            except Exception as e:
                log.debug(f"Healthcheck failed with exception: {e}")
                self.backend_errored(str(e))

    async def _start_tracking(self) -> None:
        await gather(
            self.__read_logs(), self.metrics._send_metrics_loop(), self.__healthcheck(), self.metrics._send_delete_requests_loop(), self.__session_gc_loop(), 
        )

    def backend_errored(self, msg: str) -> None:
        self.metrics._model_errored(msg)

    async def __call_backend(
        self, handler: EndpointHandler[ApiPayload_T], payload: ApiPayload_T
    ) -> ClientResponse:
        if handler.remote_dispatch_function:
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
        log.debug(
            f"Calling remote dispatch function on {handler.endpoint} "
            f"with params {remote_func_params}"
        )

        result = await handler.call_remote_dispatch_function(params=remote_func_params)

        # Wrap the result in a fake ClientResponse-like object
        class RemoteDispatchClientResponse:
            def __init__(self, data: Any, status: int = 200):
                self._body = json.dumps({"result": data}).encode("utf-8")
                self.status = status
                self.content_type = "application/json"
                self.headers = {"Content-Type": self.content_type}

            async def read(self) -> bytes:
                return self._body

        return RemoteDispatchClientResponse(result) 

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
            log.error(f"Signature error: reqnum failure, got {auth_data.reqnum}, current_reqnum: {self.reqnum}")
            return False
        elif verify_signature(json.dumps(message, indent=4, sort_keys=True), auth_data.signature):
            self.reqnum = max(auth_data.reqnum, self.reqnum)
            self.msg_history.append(message)
            self.msg_history = self.msg_history[-MSG_HISTORY_LEN:]
            return True
        else:
            log.error(f"Signature error: signature verification failed, sig:{auth_data.signature}, message: {message}")
            return False

    async def __read_logs(self) -> Awaitable[NoReturn]:

        async def run_benchmark() -> float:
            log.debug("Model load detected")
            try:
                with open(BENCHMARK_INDICATOR_FILE, "r") as f:
                    perf = float(f.readline())
                    log.debug(f"Already ran benchmark for perf score of {perf}")
                    return perf
            except FileNotFoundError:
                pass
            if self.benchmark_handler.do_warmup_benchmark:
                log.debug(f"Performing benchmark on endpoint {self.benchmark_handler.endpoint}")
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
                    log.error(f"Benchmark Failed: No successful responses")
                    return 0.0
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
            log.debug( f"Benchmark complete: average perf is {average_throughput}, measured perf is {max_throughput}")
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
                        except Exception as e:
                            log.debug(f"Benchmark failed with errror: {e}")
                            self.backend_errored(f"Benchmark failed with errror: {e}")
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
        while True:
            if os.path.isfile(self.model_log_file) is True:
                return await tail_log()
            else:
                await sleep(1)
      


    async def __session_gc_loop(self) -> NoReturn:
        while True:
            try:
                await sleep(SESSION_GC_INTERVAL)
                now = time.time()

                async with self._sessions_lock:
                    expired = [
                        sid for sid, s in self.sessions.items()
                        if s.expiration is not None and s.expiration <= now
                    ]

                if expired:
                    log.debug(f"Removing {len(expired)} expired sessions")
                for sid in expired:
                    await self.__close_session(sid)

            except CancelledError:
                log.debug("Session GC task cancelled; exiting loop")
                raise
            except Exception as e:
                log.debug(f"Session GC loop error: {e}")
                continue
                
