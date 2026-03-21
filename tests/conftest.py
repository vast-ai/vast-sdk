"""Shared pytest fixtures for vast-sdk tests.

Fixtures follow unit-test-requirements: one fixture per concept, defined in
conftest.py for reuse across test files. Pyworker ``Metrics`` helpers live in the
``# Pyworker server`` section below.

An autouse fixture restores the ``Serverless`` logger after each test so global
logging state follows RAII and cannot leak between cases.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientResponseError

from vastai.serverless.client.client import Serverless
from vastai.serverless.client.endpoint import Endpoint
from vastai.serverless.server.lib.data_types import RequestMetrics, Session as PyworkerSession
from vastai.serverless.server.worker import (
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
)


# ---------------------------------------------------------------------------
# Global RAII (resource cleanup after every test)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_serverless_logger_state():
    """Snapshot and restore the ``Serverless`` client logger after each test.

    ``Serverless.__init__`` configures ``logging.getLogger("Serverless")``. Leaked
    handlers or ``propagate=False`` can make later tests call ``time.time()`` via
    logging and break strict clock mocks.
    """
    log = logging.getLogger("Serverless")
    old_handlers = list(log.handlers)
    old_level = log.level
    old_propagate = log.propagate
    old_disabled = log.disabled
    yield
    log.handlers.clear()
    for h in old_handlers:
        log.addHandler(h)
    log.setLevel(old_level)
    log.propagate = old_propagate
    log.disabled = old_disabled


# ---------------------------------------------------------------------------
# Server Worker test helpers (mocks for generate_client_response etc.)
# ---------------------------------------------------------------------------


@pytest.fixture
def make_mock_web_request():
    """Factory: create a mock object for aiohttp web.Request in handler tests."""
    def _make(spec_request: bool = False):
        if spec_request:
            from aiohttp import web
            return MagicMock(spec=web.Request)
        return MagicMock()
    return _make


@pytest.fixture
def make_mock_model_response():
    """Factory: create a mock model response for generate_client_response tests.

    Returns a callable that accepts content_type, body, status, and optional
    stream_chunks. If stream_chunks is provided, content.iter_any is an async
    generator yielding those chunks; otherwise read() returns body.
    """
    def _make(
        content_type: str = "application/json",
        body: bytes | None = None,
        status: int = 200,
        stream_chunks: list[bytes] | None = None,
    ):
        mock = MagicMock()
        mock.content_type = content_type
        mock.status = status
        mock.headers = MagicMock()
        mock.headers.get = MagicMock(return_value=None)
        if stream_chunks is not None:
            async def _iter():
                for c in stream_chunks:
                    yield c
            mock.content.iter_any = _iter
        else:
            mock.read = AsyncMock(return_value=body or b"")
        return mock
    return _make


# ---------------------------------------------------------------------------
# Client Worker (vastai.serverless.client.worker) fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_client_worker_dict() -> dict:
    """Minimal dict with only required id field for client Worker.from_dict tests."""
    return {"id": 1}


@pytest.fixture
def full_client_worker_dict() -> dict:
    """Complete dict with all Worker fields for client Worker.from_dict tests.

    Returns a dict that exercises every field. Tests may override specific
    keys to test edge cases.
    """
    return {
        "id": 42,
        "status": "RUNNING",
        "cur_load": 0.5,
        "new_load": 0.6,
        "cur_load_rolling_avg": 0.55,
        "cur_perf": 1.2,
        "perf": 1.1,
        "measured_perf": 1.0,
        "dlperf": 0.9,
        "reliability": 0.95,
        "reqs_working": 3,
        "disk_usage": 0.4,
        "loaded_at": 1700000000.0,
        "started_at": 1699999000.0,
    }


# ---------------------------------------------------------------------------
# Server Worker (vastai.serverless.server.worker) fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_worker_config() -> WorkerConfig:
    """WorkerConfig with minimal required fields for EndpointHandlerFactory.

    Use when tests need a valid config without custom handlers.
    """
    return WorkerConfig(
        model_server_url="http://localhost",
        model_server_port=8000,
    )


@pytest.fixture
def worker_config_with_handler():
    """Factory fixture: build WorkerConfig with HandlerConfig(s) and BenchmarkConfig.

    Returns a callable that accepts route, dataset, and optional extra handlers.
    Use to avoid repeating WorkerConfig + HandlerConfig + BenchmarkConfig setup.
    """

    def _make(
        route: str = "/predict",
        dataset: list | None = None,
        extra_handlers: list[HandlerConfig] | None = None,
    ) -> WorkerConfig:
        if dataset is None:
            dataset = [{"input": "test"}]
        handlers = [
            HandlerConfig(
                route=route,
                benchmark_config=BenchmarkConfig(dataset=dataset),
            ),
        ]
        if extra_handlers:
            handlers.extend(extra_handlers)
        return WorkerConfig(
            model_server_url="http://localhost",
            model_server_port=8000,
            handlers=handlers,
        )

    return _make


@pytest.fixture
def worker_config_from_handlers():
    """Factory fixture: build WorkerConfig from an explicit list of HandlerConfigs.

    Returns a callable that accepts a list of HandlerConfig and returns WorkerConfig
    with standard model_server_url and model_server_port. Use when tests need custom
    handler options (request_parser, response_generator, payload_class, etc.).
    """

    def _make(handlers: list[HandlerConfig]) -> WorkerConfig:
        return WorkerConfig(
            model_server_url="http://localhost",
            model_server_port=8000,
            handlers=handlers,
        )

    return _make


# ---------------------------------------------------------------------------
# Connection (vastai.serverless.client.connection) fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_sse_response():
    """Factory: create mock aiohttp response with SSE/JSONL stream content.

    Returns a callable that accepts an iterable of bytes chunks and returns
    a mock response whose content.iter_any yields those chunks.

    Use for _iter_sse_json tests.
    """

    def _make(chunks):
        async def mock_iter():
            for c in chunks:
                yield c

        mock_resp = MagicMock()
        mock_resp.content.iter_any = mock_iter
        return mock_resp

    return _make


@pytest.fixture
def make_mock_http_response():
    """Factory: create mock aiohttp response for async with session.get/post.

    Returns a callable that accepts status, text, json, json_side_effect
    and returns a mock response configured for use in 'async with' context.
    Use for _make_request tests.
    """

    def _make(
        status: int = 200,
        text: str = "",
        json_data=None,
        json_side_effect=None,
    ):
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.headers = {}
        mock_resp.text = AsyncMock(return_value=text)
        mock_resp.json = AsyncMock(
            return_value=json_data,
            side_effect=json_side_effect,
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)
        return mock_resp

    return _make


@pytest.fixture
def make_mock_make_request_client():
    """Factory: create mock session and client for _make_request.

    Returns a callable ``(mock_session, mock_client)`` configured for _make_request.

    - ``make_mock_make_request_client(mock_resp)`` — ``session.get`` returns ``mock_resp``.
    - ``make_mock_make_request_client(get_side_effect=...)`` — ``session.get`` uses
      ``AsyncMock(side_effect=...)`` (exception, list of responses, etc.).
    - ``make_mock_make_request_client(post_return=resp)`` — ``session.post`` returns
      ``resp``; ``session.get`` is a bare ``AsyncMock()`` (unused).
    """

    def _make(mock_resp=None, *, get_side_effect=None, post_return=None):
        mock_session = MagicMock()
        if get_side_effect is not None:
            mock_session.get = AsyncMock(side_effect=get_side_effect)
        elif post_return is not None:
            mock_session.get = AsyncMock()
        else:
            mock_session.get = AsyncMock(return_value=mock_resp)

        if post_return is not None:
            mock_session.post = AsyncMock(return_value=post_return)
        else:
            mock_session.post = AsyncMock()

        mock_client = MagicMock()
        mock_client._get_session = AsyncMock(return_value=mock_session)
        mock_client.get_ssl_context = AsyncMock(return_value=None)

        return mock_session, mock_client

    return _make


@pytest.fixture
def patch_build_kwargs():
    """Patch _build_kwargs for _make_request tests.

    Yields the mock; tests run with _build_kwargs patched to return
    standard kwargs (headers, params, timeout).
    """
    with patch(
        "vastai.serverless.client.connection._build_kwargs"
    ) as mock_build:
        mock_build.return_value = {
            "headers": {},
            "params": {},
            "timeout": MagicMock(),
        }
        yield mock_build


@pytest.fixture
def make_mock_session():
    """Factory: create mock aiohttp session for _open_once tests.

    Returns a callable that accepts get_returns and post_returns (optional)
    and returns a mock session with get/post configured.
    """

    def _make(get_returns=None, post_returns=None):
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=get_returns or MagicMock())
        mock_session.post = AsyncMock(return_value=post_returns or MagicMock())
        return mock_session

    return _make


@pytest.fixture
def build_kwargs_defaults():
    """Default kwargs for _build_kwargs tests.

    Returns a dict of common defaults; tests can override as needed.
    """
    return {
        "headers": {},
        "params": {},
        "ssl_context": None,
        "timeout": 30.0,
        "body": None,
        "method": "GET",
        "stream": False,
    }


# ---------------------------------------------------------------------------
# Pyworker server (vastai.serverless.server.lib.metrics) fixtures
# ---------------------------------------------------------------------------


def _metrics_post_ok_context_and_response() -> tuple[MagicMock, MagicMock]:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    return mock_ctx, mock_resp


@pytest.fixture
def clear_get_url_cache():
    """Clear functools.cache on get_url() before and after each test.

    Use via ``pytestmark = pytest.mark.usefixtures("clear_get_url_cache")`` on
    modules that patch os.environ for URL tests.
    """
    from vastai.serverless.server.lib.metrics import get_url

    get_url.cache_clear()
    yield
    get_url.cache_clear()


@pytest.fixture
def make_pyworker_metrics():
    """Factory: build Metrics with explicit id/report_addr/url (no CONTAINER_ID env)."""

    def _make(**kwargs):
        from vastai.serverless.server.lib.metrics import Metrics

        defaults = dict(
            id=1,
            report_addr=["http://report.test"],
            url="http://worker.test:9000",
        )
        defaults.update(kwargs)
        return Metrics(**defaults)

    return _make


@pytest.fixture
def make_pyworker_session():
    """Factory: ``PyworkerSession`` for metrics tests (server ``Session``, not aiohttp)."""

    def _make(**kwargs):
        defaults = dict(
            session_id="s1",
            lifetime=0.0,
            auth_data={},
            expiration=0.0,
            on_close_route="",
            on_close_payload={},
            request_idx=1,
        )
        defaults.update(kwargs)
        return PyworkerSession(**defaults)

    return _make


@pytest.fixture
def make_pyworker_request_metrics():
    """Factory: ``RequestMetrics`` with defaults; override fields per test."""

    def _make(**kwargs):
        defaults = dict(
            request_idx=1,
            reqnum=1,
            workload=1.0,
            status="",
            success=False,
            is_session=False,
            session=None,
            session_reqnum=None,
        )
        defaults.update(kwargs)
        return RequestMetrics(**defaults)

    return _make


@pytest.fixture
def make_metrics_aiohttp_post():
    """Single factory for aiohttp-style ``session.post`` / async context mocks in metrics tests.

    - ``session_ok()`` → ``(mock_session, mock_response)``
    - ``context_ok()`` → ``(context_manager, response)``
    - ``context_timeout()`` → context whose ``__aenter__`` raises ``asyncio.TimeoutError``
    - ``context_client_error(status=500)`` → context; ``raise_for_status`` raises ``ClientResponseError``
    - ``context_enter_raises(exc)`` → context whose ``__aenter__`` raises ``exc``
    """

    def session_ok():
        mock_ctx, mock_resp = _metrics_post_ok_context_and_response()
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_ctx)
        return mock_session, mock_resp

    def context_ok():
        return _metrics_post_ok_context_and_response()

    def context_timeout():
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        return mock_ctx

    def context_client_error(*, status: int = 500):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock(
            side_effect=ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=status,
            )
        )
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        return mock_ctx

    def context_enter_raises(exc: BaseException):
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=exc)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        return mock_ctx

    return SimpleNamespace(
        session_ok=session_ok,
        context_ok=context_ok,
        context_timeout=context_timeout,
        context_client_error=context_client_error,
        context_enter_raises=context_enter_raises,
    )


@pytest.fixture
def metrics_worker_status_context():
    """Patch ``Metrics.http``, disk usage, and optionally ``asyncio.sleep`` for ``__send_metrics_and_reset``."""

    @contextmanager
    def _cm(m, mock_session, *, disk_gb: float = 1.0, mock_asyncio_sleep: bool = False):
        with patch.object(m, "http", new_callable=AsyncMock, return_value=mock_session):
            with patch(
                "vastai.serverless.server.lib.data_types.SystemMetrics.get_disk_usage_GB",
                return_value=disk_gb,
            ):
                if mock_asyncio_sleep:
                    with patch(
                        "vastai.serverless.server.lib.metrics.asyncio.sleep",
                        new_callable=AsyncMock,
                    ):
                        yield
                else:
                    yield

    return _cm


@pytest.fixture
def metrics_delete_send_context():
    """Patch ``Metrics.http`` and ``asyncio.sleep`` for ``__send_delete_requests_and_reset``."""

    @contextmanager
    def _cm(m, mock_session):
        with patch.object(m, "http", new_callable=AsyncMock, return_value=mock_session):
            with patch(
                "vastai.serverless.server.lib.metrics.asyncio.sleep",
                new_callable=AsyncMock,
            ):
                yield

    return _cm


@pytest.fixture
def patch_pyworker_metrics_loop():
    """Patch ``time``, ``_Metrics__send_metrics_and_reset``, and ``metrics.sleep`` for ``_send_metrics_loop`` tests."""

    @contextmanager
    def _cm(m, mock_send, *, time_return: float):
        with patch("vastai.serverless.server.lib.metrics.time") as mock_time:
            mock_time.time.return_value = time_return
            with patch.object(m, "_Metrics__send_metrics_and_reset", mock_send):
                with patch(
                    "vastai.serverless.server.lib.metrics.sleep",
                    new_callable=AsyncMock,
                ):
                    yield mock_time

    return _cm


@pytest.fixture
def make_metrics_client_session_instance():
    """Factory: MagicMock standing in for ``aiohttp.ClientSession`` in metrics ``http()`` tests."""

    def _make(*, close_async: bool = False):
        inst = MagicMock()
        if close_async:
            inst.close = AsyncMock()
        return inst

    return _make


# ---------------------------------------------------------------------------
# Serverless client (vastai.serverless.client.client) fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> Serverless:
    """Serverless client with a fixed test API key (no aiohttp session)."""
    return Serverless(api_key="k")


@pytest.fixture
def client_with_session(client: Serverless) -> Serverless:
    """Serverless client with a mocked open aiohttp session on ``_session``."""
    mock_sess = MagicMock()
    mock_sess.closed = False
    mock_sess.close = AsyncMock()
    client._session = mock_sess
    return client


@pytest.fixture
def serverless_master_client() -> Serverless:
    """Serverless client using api_key ``master`` (autoscaler POST payload tests)."""
    return Serverless(api_key="master")


@pytest.fixture
def make_serverless_endpoint():
    """Build an :class:`Endpoint` bound to a :class:`Serverless` client."""

    def _make(
        sl_client: Serverless,
        *,
        name: str = "myep",
        endpoint_id: int = 5,
        api_key: str = "ekey",
    ) -> Endpoint:
        return Endpoint(sl_client, name, endpoint_id, api_key)

    return _make


@pytest.fixture
def make_route_waiting_mock():
    """Factory: MagicMock route with status WAITING (polling loop)."""

    def _make(*, request_idx: int = 1, body: dict | None = None) -> MagicMock:
        waiting = MagicMock()
        waiting.status = "WAITING"
        waiting.request_idx = request_idx
        waiting.body = body if body is not None else {}
        return waiting

    return _make


@pytest.fixture
def make_route_ready_mock():
    """Factory: MagicMock route with status READY, ``get_url``, and body."""

    def _make(
        *,
        url: str = "https://w/",
        request_idx: int = 1,
        body: dict | None = None,
    ) -> MagicMock:
        b = {"url": url, **(body or {})}
        ready = MagicMock()
        ready.status = "READY"
        ready.request_idx = request_idx
        ready.body = b
        ready.get_url = MagicMock(return_value=url)
        return ready

    return _make


@pytest.fixture
def patch_serverless_queue_async_stubs():
    """Patch ``asyncio.sleep`` and ``random.uniform`` on the serverless client module.

    Queue/routing tests use instant sleep and fixed jitter so they stay fast and
    deterministic. Does not apply when a test replaces ``sleep`` with a custom
    ``side_effect`` (e.g. cancellation).
    """
    with (
        patch(
            "vastai.serverless.client.client.asyncio.sleep",
            new_callable=AsyncMock,
        ),
        patch(
            "vastai.serverless.client.client.random.uniform",
            return_value=0.1,
        ),
    ):
        yield
