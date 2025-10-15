import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple, Optional, Set, TypeVar, Generic, Type, Callable, Awaitable
from aiohttp import web, ClientResponse
import inspect
import random
import json
import psutil


"""
type variable representing an incoming payload to pyworker that will used to calculate load and will then
be forwarded to the model
"""

log = logging.getLogger(__file__)


class JsonDataException(Exception):
    def __init__(self, json_msg: Dict[str, Any]):
        self.message = json_msg


ApiPayload_T = TypeVar("ApiPayload_T", bound="ApiPayload")


@dataclass
class ApiPayload(ABC):

    @classmethod
    @abstractmethod
    def for_test(cls: Type[ApiPayload_T]) -> ApiPayload_T:
        """defines how create a payload for load testing"""
        pass

    @abstractmethod
    def generate_payload_json(self) -> Dict[str, Any]:
        """defines how to convert an ApiPayload to JSON that will be sent to model API"""
        pass

    @abstractmethod
    def count_workload(self) -> float:
        """defines how to calculate workload for a payload"""
        pass

    @classmethod
    @abstractmethod
    def from_json_msg(
        cls: Type[ApiPayload_T], json_msg: Dict[str, Any]
    ) -> ApiPayload_T:
        """
        defines how to create an API payload from a JSON message,
        it should throw an JsonDataException if there are issues with some fields
        or they are missing in the format of
        {
            "field": "error msg"
        }
        """
        pass


@dataclass
class AuthData:
    """data used to authenticate requester"""

    signature: str
    cost: str
    endpoint: str
    reqnum: int
    url: str

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]):
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg:
                errors[param] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        return cls(
            **{
                k: v
                for k, v in json_msg.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class EndpointHandler(ABC, Generic[ApiPayload_T]):
    """
    Each model endpoint will have a handler responsible for counting workload from the incoming ApiPayload
    and converting it to json to be forwarded to model API
    """

    benchmark_runs: int = 8
    benchmark_words: int = 100

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """the endpoint on the model API"""
        pass

    @property
    @abstractmethod
    def healthcheck_endpoint(self) -> Optional[str]:
        """the endpoint on the model API that is used for healthchecks"""
        pass

    @classmethod
    @abstractmethod
    def payload_cls(cls) -> Type[ApiPayload_T]:
        """ApiPayload class"""
        pass

    @abstractmethod
    def make_benchmark_payload(self) -> ApiPayload_T:
        """defines how to create an ApiPayload for benchmarking."""
        pass

    @abstractmethod
    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        """
        defines how to convert a model API response to a response to PyWorker client
        """
        pass

    @classmethod
    def get_data_from_request(
        cls, req_data: Dict[str, Any]
    ) -> Tuple[AuthData, ApiPayload_T]:
        errors = {}
        auth_data: Optional[AuthData] = None
        payload: Optional[ApiPayload_T] = None
        try:
            if "auth_data" in req_data:
                auth_data = AuthData.from_json_msg(req_data["auth_data"])
            else:
                errors["auth_data"] = "field missing"
        except JsonDataException as e:
            errors["auth_data"] = e.message
        try:
            if "payload" in req_data:
                payload_cls = cls.payload_cls()
                payload = payload_cls.from_json_msg(req_data["payload"])
            else:
                errors["payload"] = "field missing"
        except JsonDataException as e:
            errors["payload"] = e.message
        if errors:
            raise JsonDataException(errors)
        if auth_data and payload:
            return (auth_data, payload)
        else:
            raise Exception("error deserializing request data")


@dataclass
class SystemMetrics:
    """General system metrics"""

    model_loading_start: float
    model_loading_time: Union[float, None]
    last_disk_usage: float
    additional_disk_usage: float
    model_is_loaded: bool

    @staticmethod
    def get_disk_usage_GB():
        return psutil.disk_usage("/").used / (2**30)  # want units of GB

    @classmethod
    def empty(cls):
        return cls(
            model_loading_start=time.time(),
            model_loading_time=None,
            last_disk_usage=SystemMetrics.get_disk_usage_GB(),
            additional_disk_usage=0.0,
            model_is_loaded=False,
        )

    def update_disk_usage(self):
        disk_usage = SystemMetrics.get_disk_usage_GB()
        self.additional_disk_usage = disk_usage - self.last_disk_usage
        self.last_disk_usage = disk_usage

    def reset(self):
        # autoscaler excepts model_loading_time to be populated only once, when the instance has
        # finished benchmarking and is ready to receive requests. This applies to restarted instances
        # as well: they should send model_loading_time once when they are done loading
        self.model_loading_time = None


@dataclass
class ModelMetrics:
    """Model specific metrics"""

    # these are reset after being sent to autoscaler
    workload_served: float
    workload_received: float
    workload_cancelled: float
    workload_errored: float
    # these are not
    workload_pending: float
    error_msg: Optional[str]
    max_throughput: float
    requests_recieved: Set[int] = field(default_factory=set)
    requests_working: Set[int] = field(default_factory=set)
    last_update: float = field(default_factory=time.time)

    @classmethod
    def empty(cls):
        return cls(
            workload_pending=0.0,
            workload_served=0.0,
            workload_cancelled=0.0,
            workload_errored=0.0,
            workload_received=0.0,
            error_msg=None,
            max_throughput=0.0,
        )

    @property
    def cur_perf(self) -> float:
        return max(self.workload_served / (time.time() - self.last_update), 0.0)

    @property
    def workload_processing(self) -> float:
        return max(self.workload_received - self.workload_cancelled, 0.0)

    def set_errored(self, error_msg):
        self.reset()
        self.error_msg = error_msg

    def reset(self):
        self.workload_served = 0
        self.workload_received = 0
        self.workload_cancelled = 0
        self.workload_errored = 0
        self.last_update = time.time()


@dataclass
class AutoScalaerData:
    """Data that is reported to autoscaler"""

    id: int
    loadtime: float
    cur_load: float
    error_msg: str
    max_perf: float
    cur_perf: float
    cur_capacity: float
    max_capacity: float
    num_requests_working: int
    num_requests_recieved: int
    additional_disk_usage: float
    url: str


class LogAction(Enum):
    """
    These actions tell the backend what a log value means, for example:
    actions [
        # this marks the model server as loaded
        (LogAction.ModelLoaded, "Starting server"),
        # these mark the model server as errored
        (LogAction.ModelError, "Exception loading model"),
        (LogAction.ModelError, "Server failed to bind to port"),
        # this tells the backend to print any logs containing the string into its own logs
        # which are visible in the vast console instance logs
        (LogAction.Info, "Starting model download"),
    ]
    """

    ModelLoaded = 1
    ModelError = 2
    Info = 3



RequestPayloadParser = Callable[[Dict[str, Any]], Dict[str, Any]]
# on_response: handles the generate_client_response logic (takes web.Request and ClientResponse)
ClientResponseHandler = Callable[[web.Request, ClientResponse], Awaitable[Union[web.Response, web.StreamResponse]]]
# calculate_workload: custom workload calculation
WorkloadCalculator = Callable[[Dict[str, Any]], float]


@dataclass
class HandlerConfig:
    """Friendly configuration for defining handlers"""
    route: str
    healthcheck: Optional[str] = None
    benchmark_data: list[dict[str, Any]] = field(default_factory=list)
    # Optional: custom ApiPayload class (if None, uses generic)
    payload_class: Optional[Type[ApiPayload]] = None
    # Optional: custom logic to parse/modify request JSON before creating ApiPayload
    on_request: Optional[RequestPayloadParser] = None
    # Optional: custom logic to handle model response and create client response
    on_response: Optional[ClientResponseHandler] = None
    # Optional: custom workload calculation
    calculate_workload: Optional[WorkloadCalculator] = None


@dataclass
class WorkerConfig:
    model_server_port: int
    model_log_file: str
    benchmark_data: list[dict[str, Any]] = field(default_factory=list)
    handlers: list[HandlerConfig] = field(default_factory=list)
    model_server_url: str = "http://127.0.0.1"
    model_healthcheck_url: str = "/health"
    benchmark_route: Optional[str] = None
    allow_parallel_requests: bool = False
    max_model_latency: Optional[float] = None,
    log_actions: list[tuple[LogAction, str]] = field(default_factory=list)


class GenericEndpointFactory:
    """Factory for creating generic endpoint handlers from WorkerConfig"""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self._handlers: Dict[str, EndpointHandler] = {}
        self._build_handlers()
    
    def _build_handlers(self) -> None:
        """Build endpoint handlers from config"""
        # If no handlers, create a default handler at /
        if not self.config.handlers:
            default_handler_config = HandlerConfig(
                route="/",
                healthcheck=self.config.model_healthcheck_url
            )
            handler = self._create_handler(default_handler_config)
            self._handlers["/"] = handler
        else:
            # Build handlers from config
            for handler_config in self.config.handlers:
                handler = self._create_handler(handler_config)
                self._handlers[handler_config.route] = handler
    
    def _create_handler(self, handler_config: HandlerConfig) -> EndpointHandler:
        """Create a generic endpoint handler from HandlerConfig"""
        
        # Extract config values with defaults
        route_path = handler_config.route
        healthcheck_path = handler_config.healthcheck or self.config.model_healthcheck_url
        benchmark_data = handler_config.benchmark_data.copy() if handler_config.benchmark_data else []
        user_payload_class = handler_config.payload_class
        user_request_parser = handler_config.on_request
        user_response_handler = handler_config.on_response
        user_workload_calculator = handler_config.calculate_workload
        
        # If user provided a custom payload class, use it
        if user_payload_class:
            PayloadClass = user_payload_class
        else:
            # Create a generic ApiPayload class
            @dataclass
            class GenericApiPayload(ApiPayload):
                data: Dict[str, Any] = field(default_factory=dict)
                
                @classmethod
                def for_test(cls) -> "GenericApiPayload":
                    # Use random.choice from benchmark_data if available
                    if benchmark_data:
                        test_data = random.choice(benchmark_data)
                    else:
                        test_data = {}
                    return cls(data=test_data.copy())
                
                def generate_payload_json(self) -> Dict[str, Any]:
                    return self.data
                
                def count_workload(self) -> float:
                    # Use custom workload calculator if provided
                    if user_workload_calculator:
                        return user_workload_calculator(self.data)
                    # Default to 100 unless overridden
                    return 100.0
                
                @classmethod
                def from_json_msg(cls, json_msg: Dict[str, Any]) -> "GenericApiPayload":
                    if not isinstance(json_msg, dict):
                        raise JsonDataException({"data": "payload must be a dictionary"})
                    
                    # Apply user's request parser if provided
                    if user_request_parser:
                        try:
                            json_msg = user_request_parser(json_msg)
                        except Exception as e:
                            log.error(f"Error in user request parser: {e}")
                            # Continue with original json_msg on error
                    
                    return cls(data=json_msg)
            
            PayloadClass = GenericApiPayload
        
        # Create a generic EndpointHandler class
        @dataclass
        class GenericEndpointHandler(EndpointHandler[PayloadClass]):
            _route: str = field(default=route_path)
            _healthcheck_endpoint: Optional[str] = field(default=healthcheck_path)
            
            @property
            def endpoint(self) -> str:
                """The endpoint is the same as the route"""
                return self._route

            @property
            def healthcheck_endpoint(self) -> Optional[str]:
                return self._healthcheck_endpoint
            
            @classmethod
            def payload_cls(cls) -> Type[PayloadClass]:
                return PayloadClass
            
            def make_benchmark_payload(self) -> PayloadClass:
                """Just call the payload class's for_test() method"""
                return PayloadClass.for_test()
            
            async def generate_client_response(
                self, 
                client_request: web.Request, 
                model_response: ClientResponse
            ) -> Union[web.Response, web.StreamResponse]:
                """
                Generate client response from model response.
                Uses the raw aiohttp objects as expected by backend.
                """
                # Use custom response handler if provided
                if user_response_handler:
                    try:
                        # Call user's handler with web.Request and ClientResponse
                        return await user_response_handler(client_request, model_response)
                    except Exception as e:
                        log.error(f"Error in user response handler: {e}")
                        # Fall back to default behavior on error
                
                # Default implementation: pass through the response
                body = await model_response.read()
                return web.Response(
                    body=body,
                    status=model_response.status,
                    headers=model_response.headers
                )
        
        return GenericEndpointHandler()
    
    def get_handler(self, route: str) -> Optional[EndpointHandler]:
        """Get handler for a specific route"""
        return self._handlers.get(route)
    
    def get_all_handlers(self) -> Dict[str, EndpointHandler]:
        """Get all registered handlers"""
        return self._handlers.copy()
    
    def get_benchmark_handler(self) -> Optional[EndpointHandler]:
        """
        Get the benchmark handler. 
        If benchmark_route is specified in config, use that handler.
        Otherwise, return the first available handler.
        Returns None if no handlers exist.
        """
        if not self._handlers:
            return None
        
        # If benchmark_route is specified and exists, use it
        if self.config.benchmark_route:
            handler = self._handlers.get(self.config.benchmark_route)
            if handler:
                return handler
        
        # Otherwise, return the first handler
        return next(iter(self._handlers.values()))
    
    def has_handlers(self) -> bool:
        """Check if any handlers are registered"""
        return len(self._handlers) > 0
    
    @property
    def model_server_base_url(self) -> str:
        """Get the full model server base URL"""
        return f"{self.config.model_server_url}:{self.config.model_server_port}"