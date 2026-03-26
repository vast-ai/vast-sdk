"""Shared pytest fixtures for vast-sdk tests.

Fixtures follow unit-test-requirements: one fixture per concept, defined in
conftest.py for reuse across test files. Pyworker ``Metrics`` helpers live in the
``# Pyworker server`` section below.

One fixture name per *kind* of resource; use factory callables for variants
(``make_request_http_mocks``, ``make_route_response_mock``, ``server_worker_config``,
``client_worker_dict``, ``make_session_mock``, ``make_test_endpoint``,
``make_backend_http_request``, ``make_mock_root_logger``, …) instead of parallel fixtures.

Pyworker: ``pyworker_backend`` (Backend with Metrics mocked), ``patch_pyworker_backend_class``,
``make_pyworker_session`` (server Session), ``valid_auth_data_dict`` (AuthData-shaped JSON).

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

from vastai.serverless.client.client import Serverless, ServerlessRequest
from vastai.serverless.client.endpoint import Endpoint
from vastai.serverless.client.session import Session
from vastai.serverless.server.lib.data_types import (
    RequestMetrics,
    Session as PyworkerSession,
)
from vastai.serverless.server.worker import (
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
)
import os
import asyncio
import pytest
import pytest_asyncio

from vastai._base import _resolve_api_key, _APIKEY_SENTINEL
from vastai.sync.client import SyncClient
from vastai.async_.client import AsyncClient
from vastai.serverless.client.client import CoroutineServerless
from vastai.data.endpoint import EndpointConfig


# ── Configuration fixtures ──────────────────────────────────────────────────
def _attach_mock_aiohttp_session(sl: Serverless) -> None:
    """Put an open mocked aiohttp-backed session on ``sl._session`` (in-place)."""
    mock_sess = MagicMock()
    mock_sess.closed = False
    mock_sess.close = AsyncMock()
    sl._session = mock_sess


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


@pytest.fixture(scope="session")
def api_key():
    return _resolve_api_key(os.environ.get("VAST_API_KEY", _APIKEY_SENTINEL))


@pytest.fixture(scope="session")
def vast_server():
    return os.environ.get("VAST_SERVER", "https://console.vast.ai")


@pytest.fixture(scope="session")
def serverless_instance():
    return os.environ.get("VAST_SERVERLESS_INSTANCE", "prod")


@pytest.fixture(scope="session")
def template_hash():
    return os.environ.get("VAST_TEST_TEMPLATE_HASH", "490c0ed717a7da3bc5e2677a80f9c4c2")


# ── Sync client fixture ─────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def sync_client(api_key, vast_server):
    return SyncClient(api_key=api_key, vast_server=vast_server)


# ── Async client fixture (function-scoped — each test gets a fresh client) ──


@pytest_asyncio.fixture
async def async_client(api_key, vast_server):
    client = AsyncClient(api_key=api_key, vast_server=vast_server)
    yield client
    await client.close()


# ── Serverless client fixture ───────────────────────────────────────────────
# Session-scoped, managed via its own event loop for setup/teardown.


@pytest.fixture(scope="session")
def _serverless_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def serverless_client(api_key, serverless_instance, _serverless_loop):
    client = CoroutineServerless(api_key=api_key, instance=serverless_instance)
    _serverless_loop.run_until_complete(client._get_session())
    yield client
    _serverless_loop.run_until_complete(client.close())


# ── Serverless endpoint fixture ─────────────────────────────────────────────


@pytest.fixture(scope="session")
def managed_endpoint(serverless_client, template_hash, _serverless_loop):
    """Session-scoped managed endpoint with a workergroup attached.

    Created once, shared across all serverless tests, torn down at end of session.
    """
    ep = None
    wg_id = None

    async def setup():
        nonlocal ep, wg_id
        ep = await serverless_client.create_endpoint(
            EndpointConfig(endpoint_name="sdk-test-session")
        )
        wg_id = await ep.add_workergroup(template_hash)
        return ep, wg_id

    async def teardown():
        errors = []
        if wg_id is not None:
            try:
                await serverless_client.delete_workergroup(wg_id)
            except Exception as e:
                errors.append(f"Failed to delete workergroup {wg_id}: {e}")
        if ep is not None:
            try:
                await ep.delete()
            except Exception as e:
                errors.append(f"Failed to delete endpoint {ep.id}: {e}")
        if errors:
            print("\n".join(["Teardown errors:"] + errors))

    ep, wg_id = _serverless_loop.run_until_complete(setup())
    yield ep
    _serverless_loop.run_until_complete(teardown())


# ---------------------------------------------------------------------------
# Client Worker (vastai.serverless.client.worker) — single dict factory
# ---------------------------------------------------------------------------

_FULL_CLIENT_WORKER_DICT = {
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


@pytest.fixture
def client_worker_dict():
    """Single factory for API worker payload dicts (``minimal`` / ``full`` + overrides)."""

    def _make(kind: str = "minimal", **overrides: object) -> dict:
        if kind == "minimal":
            d = {"id": 1}
        elif kind == "full":
            d = dict(_FULL_CLIENT_WORKER_DICT)
        else:
            raise ValueError(f"unknown kind {kind!r}, use 'minimal' or 'full'")
        d.update(overrides)
        return d

    return _make


# ---------------------------------------------------------------------------
# Server Worker (vastai.serverless.server.worker) — single WorkerConfig factory
# ---------------------------------------------------------------------------


@pytest.fixture
def server_worker_config():
    """Single factory for :class:`WorkerConfig` (minimal / handler / from_handlers)."""

    def _make(
        kind: str,
        *,
        route: str = "/predict",
        dataset: list | None = None,
        extra_handlers: list[HandlerConfig] | None = None,
        handlers: list[HandlerConfig] | None = None,
    ) -> WorkerConfig:
        if kind == "minimal":
            return WorkerConfig(
                model_server_url="http://localhost",
                model_server_port=8000,
            )
        if kind == "handler":
            if dataset is None:
                dataset = [{"input": "test"}]
            hs = [
                HandlerConfig(
                    route=route,
                    benchmark_config=BenchmarkConfig(dataset=dataset),
                ),
            ]
            if extra_handlers:
                hs.extend(extra_handlers)
            return WorkerConfig(
                model_server_url="http://localhost",
                model_server_port=8000,
                handlers=hs,
            )
        if kind == "from_handlers":
            if handlers is None:
                raise ValueError("from_handlers requires handlers=")
            return WorkerConfig(
                model_server_url="http://localhost",
                model_server_port=8000,
                handlers=handlers,
            )
        raise ValueError(f"unknown kind {kind!r}")

    return _make


# ---------------------------------------------------------------------------
# Pyworker Backend / Worker (server.lib.backend, server.worker) — single mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_pyworker_backend_class():
    """Patch :class:`Backend` in ``server.lib.backend`` with ``MagicMock`` (constructor)."""
    from vastai.serverless.server.lib import backend as backend_mod

    with patch.object(backend_mod, "Backend", MagicMock()) as mock_cls:
        yield mock_cls


@pytest.fixture
def make_mock_root_logger():
    """Factory: mock root logger for :class:`Worker` ``__init__`` logging branches.

    Returns ``(root_mock, handler_mock_or_none)`` — ``handler_mock_or_none`` is the
    first handler when ``with_handlers=True``, else ``None``.
    """

    def _make(*, with_handlers: bool):
        mock_root = MagicMock()
        mock_root.setLevel = MagicMock()
        if with_handlers:
            h = MagicMock()
            mock_root.handlers = [h]
            mock_root.addHandler = MagicMock()
            return mock_root, h
        mock_root.handlers = []
        mock_root.addHandler = MagicMock()
        return mock_root, None

    return _make


@pytest.fixture
def pyworker_backend():
    """Single :class:`Backend` test instance; ``Metrics`` is mocked (no ``CONTAINER_ID`` env)."""
    from vastai.serverless.server.lib.backend import Backend
    from vastai.serverless.server.lib.data_types import LogAction

    with patch(
        "vastai.serverless.server.lib.backend.Metrics",
        return_value=MagicMock(),
    ):
        return Backend(
            model_server_url="http://localhost:8000",
            model_log_file="/tmp/model.log",
            benchmark_handler=MagicMock(),
            log_actions=[(LogAction.Info, "ready")],
        )


@pytest.fixture
def make_backend_http_request():
    """Factory: mock ``aiohttp.web.Request`` with ``.json`` async for Backend handler tests."""

    def _make(
        *,
        json_data=None,
        json_side_effect=None,
    ):
        req = MagicMock()
        if json_side_effect is not None:
            req.json = AsyncMock(side_effect=json_side_effect)
        else:
            req.json = AsyncMock(
                return_value=json_data if json_data is not None else {}
            )
        return req

    return _make


@pytest.fixture
def web_json_body():
    """Callable: parse JSON dict from ``web.json_response`` result ``.body`` bytes."""

    def _parse(resp):
        import json

        return json.loads(resp.body.decode())

    return _parse


@pytest.fixture
def valid_auth_data_dict():
    """Valid ``auth_data`` payload for :class:`AuthData` / ``get_data_from_request`` tests."""
    return {
        "cost": "1",
        "endpoint": "/predict",
        "reqnum": 1,
        "request_idx": 0,
        "signature": "sig",
        "url": "http://example.com",
    }


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
def make_request_http_mocks():
    """Single factory for ``(mock_session, mock_client)`` used by ``_make_request`` tests.

    Returns a callable ``(mock_session, mock_client)`` configured for _make_request.

    - ``make_request_http_mocks(mock_resp)`` — ``session.get`` returns ``mock_resp``.
    - ``make_request_http_mocks(get_side_effect=...)`` — ``session.get`` uses
      ``AsyncMock(side_effect=...)`` (exception, list of responses, etc.).
    - ``make_request_http_mocks(post_return=resp)`` — ``session.post`` returns
      ``resp``; ``session.get`` is a bare ``AsyncMock()`` (unused).
    Pass ``mock_resp=None`` and omit the kwargs to configure ``get``/``post`` manually
    on the returned ``mock_session``.
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
    with patch("vastai.serverless.client.connection._build_kwargs") as mock_build:
        mock_build.return_value = {
            "headers": {},
            "params": {},
            "timeout": MagicMock(),
        }
        yield mock_build


@pytest.fixture
def make_aiohttp_client_session_mock():
    """Factory: mock aiohttp ``ClientSession`` for ``_open_once`` tests."""

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
    _attach_mock_aiohttp_session(client)
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
def make_route_response_mock():
    """Single factory for autoscaler route polling mocks (WAITING / READY)."""

    def _make(
        *,
        status: str = "WAITING",
        url: str = "https://w/",
        request_idx: int = 1,
        body: dict | None = None,
    ) -> MagicMock:
        if status == "READY":
            b = {"url": url, **(body or {})}
            m = MagicMock()
            m.status = "READY"
            m.request_idx = request_idx
            m.body = b
            m.get_url = MagicMock(return_value=url)
            return m
        if status == "WAITING":
            m = MagicMock()
            m.status = "WAITING"
            m.request_idx = request_idx
            m.body = body if body is not None else {}
            return m
        raise ValueError(f"unknown status {status!r}, use WAITING or READY")

    return _make


@pytest.fixture
def make_completed_serverless_request():
    """Factory: build a resolved :class:`ServerlessRequest` (call from async tests only).

    Pass either ``result=`` for ``set_result`` or ``exception=`` for ``set_exception``.
    """

    def _make(
        *,
        result: dict | None = None,
        exception: BaseException | None = None,
    ) -> ServerlessRequest:
        if (result is None) == (exception is None):
            raise ValueError("Exactly one of result= or exception= must be given")
        req = ServerlessRequest()
        if exception is not None:
            req.set_exception(exception)
        else:
            req.set_result(result)
        return req

    return _make


@pytest.fixture
def make_session_mock():
    """Single factory: ``MagicMock(spec=Session)`` for client session HTTP tests."""

    def _make(
        *,
        session_id: int = 1,
        url: str | None = "https://worker/u",
        auth_data: dict | None = None,
        open_: bool = True,
    ) -> MagicMock:
        m = MagicMock(spec=Session)
        m.session_id = session_id
        m.url = url
        m.auth_data = {} if auth_data is None else auth_data
        m.open = open_
        return m

    return _make


@pytest.fixture
def default_start_endpoint_session_ep(client, make_serverless_endpoint):
    """Shared :class:`Endpoint` for ``start_endpoint_session`` tests."""
    return make_serverless_endpoint(client, name="ep", endpoint_id=3, api_key="ek")


@pytest.fixture
def make_test_endpoint(client, make_serverless_endpoint):
    """Factory for test :class:`Endpoint` instances.

    Do not depend on ``client_with_session``: that fixture mutates the shared
    ``client`` in place. ``open_session=True`` uses a **new** ``Serverless`` with
    its own mock aiohttp session so ``open_session=False`` keeps ``client`` without
    ``_session`` (unless another fixture or test attaches one).
    """

    def _make(
        *,
        open_session: bool = False,
        name: str = "ep",
        endpoint_id: int = 1,
        api_key: str = "ek",
    ) -> Endpoint:
        if open_session:
            sl = Serverless(api_key="k")
            _attach_mock_aiohttp_session(sl)
        else:
            sl = client
        return make_serverless_endpoint(
            sl, name=name, endpoint_id=endpoint_id, api_key=api_key
        )

    return _make


@pytest.fixture
def bound_session(make_test_endpoint) -> tuple[Endpoint, Session]:
    """Real :class:`Session` tied to a default test :class:`Endpoint`."""
    ep = make_test_endpoint()
    sess = Session(
        ep,
        session_id=99,
        lifetime=60.0,
        expiration="later",
        url="https://worker/w",
        auth_data={"t": 1},
    )
    return ep, sess


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
