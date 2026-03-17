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
