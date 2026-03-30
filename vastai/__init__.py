import logging
import os

_VAST_LOG_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
_env_level = os.environ.get("VAST_LOG_LEVEL")
_level = _VAST_LOG_LEVELS.get(_env_level, logging.INFO) if _env_level else logging.INFO

logger = logging.getLogger("vastai")
logger.setLevel(_level)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

from .vastai_sdk import VastAI
from .sync.client import SyncClient
from .async_.client import AsyncClient
from .serverless.client.client import (
    Serverless,
    CoroutineServerless,
    ServerlessRequest,
    _ServerlessBase,
    SessionCreateError,
)
from .serverless.client.request_status import RequestStatus
from .serverless.client.endpoint import Endpoint
from .serverless.server.worker import Worker
from .serverless.server.worker import (
    WorkerConfig,
    HandlerConfig,
    LogActionConfig,
    BenchmarkConfig,
)

__all__ = [
    # Clients
    "SyncClient",
    "AsyncClient",
    # Pre-refactor top-level re-exports
    "VastAI",
    "Serverless",
    "CoroutineServerless",
    "ServerlessRequest",
    "RequestStatus",
    "SessionCreateError",
    "Endpoint",
    "Worker",
    "WorkerConfig",
    "HandlerConfig",
    "LogActionConfig",
    "BenchmarkConfig",
]
