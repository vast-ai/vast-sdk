"""Unit tests for vastai.serverless.server.lib.server start/stop wiring.

Exercises route registration, SSL and plain TCP branches, ``AppRunner``/``TCPSite``
kwargs (including ``handler_cancellation`` and forwarded ``host``), default HTTP
port, and failure beacon behavior with real binds and long loops mocked per
unit-test-requirements.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vastai.serverless.server.lib import server as server_mod
from vastai.serverless.server.lib.server import start_server, start_server_async


@pytest.mark.asyncio
async def test_start_server_async_registers_session_routes_and_starts_sites(
    serverless_backend_and_handler_default,
    serverless_tracked_runner_and_tcp_site,
    serverless_gather_await_all,
    serverless_aiohttp_route_path_tuples,
    run_serverless_start_server_async_patched,
    make_serverless_tcp_worker_env,
) -> None:
    """
    Verifies start_server_async builds apps with session endpoints and starts TCPSites.

    This test verifies by:
    1. Patching AppRunner, TCPSite, and gather so nothing listens on real ports
    2. Capturing Application instances passed to AppRunner
    3. Asserting main app includes POST /session/create, /session/end, /session/get, /session/health
    4. Asserting HTTP app includes POST /session/end

    Assumptions:
    - backend._start_tracking is mocked so gather completes; WORKER_PORT and related env are set
    """
    backend, _ = serverless_backend_and_handler_default
    routes: list = []
    st = serverless_tracked_runner_and_tcp_site
    apps_seen = st.apps_seen
    route_paths = serverless_aiohttp_route_path_tuples

    env = make_serverless_tcp_worker_env(9100, http_port=9101)
    mock_track = await run_serverless_start_server_async_patched(
        backend,
        routes,
        env,
        gather_side_effect=serverless_gather_await_all,
    )

    mock_track.assert_awaited_once()
    assert len(apps_seen) >= 2
    main_app = apps_seen[0]
    http_app = apps_seen[1]
    paths = route_paths(main_app)
    for suffix in ("/session/create", "/session/end", "/session/get", "/session/health"):
        assert any(suffix in p for _, p in paths)
    http_paths = route_paths(http_app)
    assert any("/session/end" in p for _, p in http_paths)


@pytest.mark.asyncio
async def test_start_server_async_defaults_http_port_to_worker_plus_one(
    serverless_backend_and_handler_default,
    serverless_tracked_runner_and_tcp_site,
    serverless_gather_await_all,
    run_serverless_start_server_async_patched,
    make_serverless_tcp_worker_env,
) -> None:
    """
    Verifies WORKER_HTTP_PORT defaults to int(WORKER_PORT) + 1 when unset.

    This test verifies by:
    1. Removing WORKER_HTTP_PORT from the environment for the duration of the call
    2. Capturing TCPSite kwargs for the HTTP-only app
    3. Asserting its port is WORKER_PORT + 1 while the TLS/plain worker uses WORKER_PORT

    Assumptions:
    - Same patched gather/runner path as other start_server_async tests
    """
    backend, _ = serverless_backend_and_handler_default
    routes: list = []
    st = serverless_tracked_runner_and_tcp_site
    tcp_calls = st.tcp_calls

    env = make_serverless_tcp_worker_env(7122)
    old_http = os.environ.pop("WORKER_HTTP_PORT", None)
    try:
        await run_serverless_start_server_async_patched(
            backend,
            routes,
            env,
            gather_side_effect=serverless_gather_await_all,
        )
    finally:
        if old_http is not None:
            os.environ["WORKER_HTTP_PORT"] = old_http
    assert tcp_calls[0]["port"] == 7122
    assert tcp_calls[1]["port"] == 7123


@pytest.mark.asyncio
async def test_start_server_async_ssl_branch_loads_cert_chain(
    serverless_backend_and_handler_default,
    serverless_tracked_runner_and_tcp_site,
    serverless_gather_await_all,
    run_serverless_start_server_async_patched,
    make_serverless_tcp_worker_env,
) -> None:
    """
    Verifies USE_SSL=true passes an ssl.SSLContext into TCPSite for the main listener.

    This test verifies by:
    1. Enabling USE_SSL and patching ssl.create_default_context to return a mock context
    2. Recording kwargs passed to TCPSite for the first site (HTTPS worker)

    Assumptions:
    - Certificate load succeeds under patch; second TCPSite remains plain HTTP
    """
    backend, _ = serverless_backend_and_handler_default
    routes: list = []
    st = serverless_tracked_runner_and_tcp_site
    tcp_calls = st.tcp_calls

    mock_ctx = MagicMock()
    mock_ctx.load_cert_chain = MagicMock()

    env = make_serverless_tcp_worker_env(9200, USE_SSL="true")
    await run_serverless_start_server_async_patched(
        backend,
        routes,
        env,
        gather_side_effect=serverless_gather_await_all,
        ssl_create_default_context_patch=patch.object(
            server_mod.ssl, "create_default_context", return_value=mock_ctx
        ),
    )

    mock_ctx.load_cert_chain.assert_called_once_with(
        certfile="/etc/instance.crt",
        keyfile="/etc/instance.key",
    )
    assert tcp_calls[0].get("ssl_context") is mock_ctx
    assert tcp_calls[1].get("ssl_context") is None


@pytest.mark.asyncio
async def test_start_server_async_ssl_cert_load_failure_enters_error_beacon(
    serverless_backend_and_handler_default,
    serverless_error_beacon_mocks,
    make_serverless_tcp_worker_env,
) -> None:
    """
    Verifies SSL certificate load errors are caught like other launch failures and enter the beacon.

    This test verifies by:
    1. Enabling USE_SSL with load_cert_chain raising OSError
    2. Patching Metrics send/sleep like other beacon tests
    3. Asserting _model_errored mentions SSL certificate failure

    Assumptions:
    - Outer try/except wraps all startup failures; beacon runs for any Exception
    """
    mock_err = serverless_error_beacon_mocks
    backend, _ = serverless_backend_and_handler_default
    routes: list = []
    mock_ctx = MagicMock()
    mock_ctx.load_cert_chain = MagicMock(side_effect=OSError("no cert file"))

    env = make_serverless_tcp_worker_env(9201, USE_SSL="true")
    with patch.dict(os.environ, env, clear=False):
        with patch.object(server_mod.ssl, "create_default_context", return_value=mock_ctx):
            with pytest.raises(RuntimeError, match="stop-beacon"):
                await start_server_async(backend, routes)

    mock_err.assert_called()
    assert "SSL Certificate" in mock_err.call_args[0][0]


@pytest.mark.asyncio
async def test_start_server_async_gather_failure_runs_beacon_until_sleep_stops(
    serverless_backend_and_handler_default,
    serverless_tracked_runner_and_tcp_site,
    serverless_error_beacon_mocks,
    serverless_gather_raise_bind_failed,
    make_serverless_tcp_worker_env,
) -> None:
    """
    Verifies launch failure enters the metrics beacon loop (error reporting path).

    This test verifies by:
    1. Patching gather to raise after AppRunner/TCPSite construction (using a replacement
       that closes the already-created site.start / _start_tracking coroutines so GC does
       not emit RuntimeWarning)
    2. Patching asyncio.sleep in the server module so the second iteration raises
    3. Asserting _model_errored was invoked with the launch error message

    Assumptions:
    - Metrics in beacon needs CONTAINER_ID etc.; send_metrics_reset is mocked to avoid I/O
    """
    mock_err = serverless_error_beacon_mocks
    backend, _ = serverless_backend_and_handler_default
    routes: list = []
    st = serverless_tracked_runner_and_tcp_site
    app_runner, tcp_site = st.app_runner, st.tcp_site

    env = make_serverless_tcp_worker_env(9300)
    with patch.dict(os.environ, env, clear=False):
        with patch.object(server_mod.web, "AppRunner", side_effect=app_runner):
            with patch.object(server_mod.web, "TCPSite", side_effect=tcp_site):
                with patch.object(
                    server_mod,
                    "gather",
                    side_effect=serverless_gather_raise_bind_failed,
                ):
                    with pytest.raises(RuntimeError, match="stop-beacon"):
                        await start_server_async(backend, routes)

    mock_err.assert_called()
    assert "bind failed" in mock_err.call_args[0][0]


def test_start_server_invokes_asyncio_run(
    serverless_backend_and_handler_default,
) -> None:
    """
    Verifies start_server delegates to asyncio.run(start_server_async(...)).

    This test verifies by:
    1. Patching asyncio.run in the server module
    2. Replacing start_server_async with a trivial async function
    3. Calling start_server(backend, routes, host="127.0.0.1") and asserting run received a coroutine

    Assumptions:
    - kwargs are forwarded into the coroutine factory call before run() sees it
    """
    backend, _ = serverless_backend_and_handler_default
    routes: list = []

    async def fake_start_server_async(b, r, **kwargs):
        assert kwargs.get("host") == "127.0.0.1"
        return None

    ran = []

    def _run_impl(coro):
        ran.append(coro)
        policy = asyncio.get_event_loop_policy()
        loop = policy.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    with patch.object(server_mod, "run", side_effect=_run_impl):
        with patch.object(server_mod, "start_server_async", fake_start_server_async):
            start_server(backend, routes, host="127.0.0.1")

    assert len(ran) == 1
    assert asyncio.iscoroutine(ran[0])


@pytest.mark.asyncio
async def test_start_server_async_forwards_kwargs_to_both_tcp_sites(
    serverless_backend_and_handler_default,
    serverless_tracked_runner_and_tcp_site,
    serverless_gather_await_all,
    run_serverless_start_server_async_patched,
    make_serverless_tcp_worker_env,
) -> None:
    """
    Verifies ``**kwargs`` from ``start_server_async`` are passed to each ``web.TCPSite``.

    This test verifies by:
    1. Calling the patched runner with ``start_server_async_kwargs={"host": "10.0.0.2"}``
    2. Inspecting captured TCPSite kwargs for both the main and HTTP listeners

    Assumptions:
    - Same patched gather/runner path as other happy-path server tests
    """
    backend, _ = serverless_backend_and_handler_default
    st = serverless_tracked_runner_and_tcp_site
    tcp_calls = st.tcp_calls

    env = make_serverless_tcp_worker_env(9400, http_port=9401)
    await run_serverless_start_server_async_patched(
        backend,
        [],
        env,
        gather_side_effect=serverless_gather_await_all,
        start_server_async_kwargs={"host": "10.0.0.2"},
    )
    assert tcp_calls[0]["host"] == "10.0.0.2"
    assert tcp_calls[1]["host"] == "10.0.0.2"


@pytest.mark.asyncio
async def test_start_server_async_main_app_runner_uses_handler_cancellation(
    serverless_backend_and_handler_default,
    serverless_tracked_runner_and_tcp_site,
    serverless_gather_await_all,
    make_serverless_tcp_worker_env,
    make_serverless_app_runner_capture_kwargs,
) -> None:
    """
    Verifies the primary ``web.AppRunner`` enables client disconnect cancellation; the HTTP app does not.

    This test verifies by:
    1. Replacing ``AppRunner`` with a side effect that records constructor kwargs
    2. Running ``start_server_async`` under the usual TCPSite/gather patches

    Assumptions:
    - First ``AppRunner`` is the main API app; second is the internal HTTP-only app
    """
    backend, _ = serverless_backend_and_handler_default
    st = serverless_tracked_runner_and_tcp_site
    tcp_site = st.tcp_site
    runner_kw_captured, capture_app_runner = make_serverless_app_runner_capture_kwargs()

    env = make_serverless_tcp_worker_env(9402, http_port=9403)
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, env, clear=False))
        stack.enter_context(patch.object(server_mod.web, "AppRunner", side_effect=capture_app_runner))
        stack.enter_context(patch.object(server_mod.web, "TCPSite", side_effect=tcp_site))
        stack.enter_context(patch.object(backend, "_start_tracking", new_callable=AsyncMock))
        stack.enter_context(
            patch.object(server_mod, "gather", side_effect=serverless_gather_await_all),
        )
        await start_server_async(backend, [])

    assert len(runner_kw_captured) == 2
    assert runner_kw_captured[0].get("handler_cancellation") is True
    assert runner_kw_captured[1].get("handler_cancellation", False) is not True


@pytest.mark.asyncio
async def test_start_server_async_without_ssl_main_site_uses_plain_tcp(
    serverless_backend_and_handler_default,
    serverless_tracked_runner_and_tcp_site,
    serverless_gather_await_all,
    run_serverless_start_server_async_patched,
    make_serverless_tcp_worker_env,
) -> None:
    """
    Verifies when ``USE_SSL`` is not true, the main listener's ``TCPSite`` gets ``ssl_context=None``.

    This test verifies by:
    1. Omitting ``USE_SSL`` or setting it to a value other than the string ``true``
    2. Asserting the first TCPSite call has no SSL context while the internal HTTP site stays plain

    Assumptions:
    - Environment does not enable the SSL branch in ``start_server_async``
    """
    backend, _ = serverless_backend_and_handler_default
    st = serverless_tracked_runner_and_tcp_site
    tcp_calls = st.tcp_calls

    env = make_serverless_tcp_worker_env(9404, http_port=9405, USE_SSL="false")
    await run_serverless_start_server_async_patched(
        backend,
        [],
        env,
        gather_side_effect=serverless_gather_await_all,
    )
    assert tcp_calls[0].get("ssl_context") is None
    assert tcp_calls[1].get("ssl_context") is None


@pytest.mark.asyncio
async def test_start_server_async_app_runner_setup_failure_triggers_error_beacon(
    serverless_backend_and_handler_default,
    serverless_error_beacon_mocks,
    make_serverless_tcp_worker_env,
    make_serverless_app_runner_first_setup_raises,
) -> None:
    """
    Verifies failures during ``AppRunner.setup`` are caught and reported via the metrics beacon.

    This test verifies by:
    1. Using a full metrics env so the post-failure ``Metrics()`` in the beacon can initialize
    2. Patching ``web.AppRunner`` so the first runner's ``setup`` raises ``RuntimeError``
    3. Asserting ``_model_errored`` received a launch message containing that error

    Assumptions:
    - A missing ``WORKER_PORT`` during startup prevents the beacon's ``Metrics()`` from running;
      this test uses a later failure so the beacon path matches production SSL/gather failures
    """
    mock_err = serverless_error_beacon_mocks
    backend, _ = serverless_backend_and_handler_default
    routes: list = []
    env = make_serverless_tcp_worker_env(9500)
    app_runner = make_serverless_app_runner_first_setup_raises()

    with patch.dict(os.environ, env, clear=False):
        with patch.object(server_mod.web, "AppRunner", side_effect=app_runner):
            with pytest.raises(RuntimeError, match="stop-beacon"):
                await start_server_async(backend, routes)

    mock_err.assert_called()
    assert "runner setup failed" in mock_err.call_args[0][0]
