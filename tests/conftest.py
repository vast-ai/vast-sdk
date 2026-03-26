"""Shared pytest fixtures for vast-sdk tests.

Fixtures follow unit-test-requirements: one fixture per concept, defined in
conftest.py for reuse across test files.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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

    Returns a callable that accepts a mock response and returns
    (mock_session, mock_client) configured for _make_request.
    """

    def _make(mock_resp):
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_resp)
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
