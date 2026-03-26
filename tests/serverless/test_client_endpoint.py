"""Unit tests for vastai.serverless.client.endpoint (Endpoint, RouteResponse).

Uses the real ``client`` fixture and route/HTTP patches. For delegation-only tests against a
minimal mock Serverless client, see ``test_endpoint_client.py``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vastai.serverless.client.endpoint import Endpoint, RouteResponse


class TestEndpointInitAndRepr:
    def test_repr_contains_name_and_id(
        self, client, make_serverless_endpoint
    ) -> None:
        ep = make_serverless_endpoint(
            client, name="my-endpoint", endpoint_id=42, api_key="ek"
        )
        assert "my-endpoint" in repr(ep) and "42" in repr(ep)

    def test_init_raises_without_client(self) -> None:
        with pytest.raises(ValueError, match="without client"):
            Endpoint(None, "n", 1, "k")

    def test_init_raises_on_empty_name(self, client) -> None:
        with pytest.raises(ValueError, match="name cannot be empty"):
            Endpoint(client, "", 1, "k")

    def test_init_raises_on_empty_id(self, client) -> None:
        with pytest.raises(ValueError, match="id cannot be empty"):
            Endpoint(client, "n", None, "k")


class TestEndpointDelegatesToClient:
    @pytest.mark.asyncio
    async def test_request_forwards_to_queue_endpoint_request(
        self, client, make_serverless_endpoint
    ) -> None:
        ep = make_serverless_endpoint(client, name="ep", endpoint_id=3, api_key="ekey")
        fut = MagicMock()
        with patch.object(client, "queue_endpoint_request", return_value=fut) as q:
            out = ep.request(
                "/worker",
                {"a": 1},
                cost=50,
                retry=False,
                stream=True,
                timeout=12.0,
            )
        assert out is fut
        q.assert_called_once()
        kw = q.call_args.kwargs
        assert kw["endpoint"] is ep
        assert kw["worker_route"] == "/worker"
        assert kw["worker_payload"] == {"a": 1}
        assert kw["cost"] == 50
        assert kw["retry"] is False
        assert kw["stream"] is True
        assert kw["timeout"] == 12.0
        assert kw["session"] is None

    @pytest.mark.asyncio
    async def test_close_session_delegates(self, client, make_test_endpoint) -> None:
        ep = make_test_endpoint()
        sess = MagicMock()
        with patch.object(
            client,
            "end_endpoint_session",
            new_callable=AsyncMock,
            return_value="x",
        ) as end:
            assert await ep.close_session(sess) == "x"
        end.assert_awaited_once_with(session=sess)

    @pytest.mark.asyncio
    async def test_session_healthcheck_true_when_get_returns_session(
        self, client, make_test_endpoint
    ) -> None:
        ep = make_test_endpoint()
        got = MagicMock()
        chk = MagicMock()
        chk.session_id = 7
        chk.auth_data = {"tok": "a"}
        with patch.object(
            client,
            "get_endpoint_session",
            new_callable=AsyncMock,
            return_value=got,
        ) as g:
            assert await ep.session_healthcheck(chk) is True
        g.assert_awaited_once()
        assert g.call_args.kwargs["endpoint"] is ep
        assert g.call_args.kwargs["session_id"] == 7
        assert g.call_args.kwargs["session_auth"] == {"tok": "a"}

    @pytest.mark.asyncio
    async def test_session_healthcheck_false_when_get_returns_none(
        self, client, make_test_endpoint
    ) -> None:
        ep = make_test_endpoint()
        with patch.object(
            client,
            "get_endpoint_session",
            new_callable=AsyncMock,
            return_value=None,
        ):
            assert await ep.session_healthcheck(MagicMock()) is False

    @pytest.mark.asyncio
    async def test_get_session_delegates(self, client, make_test_endpoint) -> None:
        ep = make_test_endpoint()
        with patch.object(
            client,
            "get_endpoint_session",
            new_callable=AsyncMock,
            return_value="sess",
        ) as g:
            out = await ep.get_session(9, {"u": 1}, timeout=7.0)
        assert out == "sess"
        g.assert_awaited_once_with(
            endpoint=ep,
            session_id=9,
            session_auth={"u": 1},
            timeout=7.0,
        )

    @pytest.mark.asyncio
    async def test_session_delegates_to_start_endpoint_session(
        self, client, make_test_endpoint
    ) -> None:
        ep = make_test_endpoint()
        with patch.object(
            client,
            "start_endpoint_session",
            new_callable=AsyncMock,
            return_value="started",
        ) as s:
            out = await ep.session(
                cost=200,
                lifetime=30.0,
                on_close_route="/cb",
                on_close_payload={"x": 1},
                timeout=5.0,
            )
        assert out == "started"
        s.assert_awaited_once_with(
            endpoint=ep,
            cost=200,
            lifetime=30.0,
            on_close_route="/cb",
            on_close_payload={"x": 1},
            timeout=5.0,
        )

    @pytest.mark.asyncio
    async def test_get_workers_delegates(self, client, make_test_endpoint) -> None:
        ep = make_test_endpoint()
        with patch.object(
            client,
            "get_endpoint_workers",
            new_callable=AsyncMock,
            return_value=[],
        ) as g:
            assert await ep.get_workers() == []
        g.assert_awaited_once_with(ep)


class TestEndpointRoute:
    @pytest.mark.asyncio
    async def test_route_raises_when_client_not_open(
        self, client, make_test_endpoint
    ) -> None:
        ep = make_test_endpoint()
        client.is_open = lambda: False  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="invalid"):
            await ep._route()

    @pytest.mark.asyncio
    async def test_route_wraps_make_request_exception(
        self, make_test_endpoint
    ) -> None:
        ep = make_test_endpoint(open_session=True)
        with patch(
            "vastai.serverless.client.endpoint._make_request",
            new_callable=AsyncMock,
            side_effect=OSError("boom"),
        ):
            with pytest.raises(RuntimeError, match="Failed to route endpoint"):
                await ep._route()

    @pytest.mark.asyncio
    async def test_route_raises_on_http_not_ok(
        self, make_test_endpoint
    ) -> None:
        ep = make_test_endpoint(open_session=True)
        with patch(
            "vastai.serverless.client.endpoint._make_request",
            new_callable=AsyncMock,
            return_value={"ok": False, "status": 502, "text": "bad"},
        ):
            with pytest.raises(RuntimeError, match="HTTP 502"):
                await ep._route()

    @pytest.mark.asyncio
    async def test_route_returns_route_response_on_success(
        self, client_with_session, make_serverless_endpoint
    ) -> None:
        client_with_session.autoscaler_url = "https://run.test"
        ep = make_serverless_endpoint(
            client_with_session, name="epname", endpoint_id=1, api_key="ek"
        )
        body = {"request_idx": 2, "url": "https://worker/"}
        with patch(
            "vastai.serverless.client.endpoint._make_request",
            new_callable=AsyncMock,
            return_value={"ok": True, "json": body},
        ) as m:
            rr = await ep._route(cost=10.0, req_idx=1, timeout=30.0)
        m.assert_awaited_once()
        assert isinstance(rr, RouteResponse)
        assert rr.status == "READY"
        assert rr.request_idx == 2
        assert rr.get_url() == "https://worker/"
        call_kw = m.call_args.kwargs
        assert call_kw["url"] == "https://run.test"
        assert call_kw["route"] == "/route/"
        assert call_kw["api_key"] == "ek"
        assert call_kw["body"]["endpoint"] == "epname"
        assert call_kw["body"]["cost"] == 10.0
        assert call_kw["body"]["request_idx"] == 1
        assert call_kw["body"]["replay_timeout"] == 30.0


class TestMakeTestEndpointClientIsolation:
    """Regression: ``make_test_endpoint`` must not pull in ``client_with_session`` (mutates ``client``)."""

    def test_default_leaves_shared_client_without_aiohttp_session(
        self, client, make_test_endpoint
    ) -> None:
        assert client._session is None
        ep = make_test_endpoint()
        assert ep.client is client
        assert client._session is None

    def test_open_session_binds_a_fresh_serverless_instance(
        self, client, make_test_endpoint
    ) -> None:
        ep = make_test_endpoint(open_session=True)
        assert ep.client is not client
        assert client._session is None
        assert ep.client._session is not None
        assert ep.client.is_open() is True


class TestRouteResponse:
    def test_repr_shows_status(self) -> None:
        rr = RouteResponse({"url": "https://x"})
        assert "READY" in repr(rr)

    def test_waiting_when_no_url_in_body(self) -> None:
        rr = RouteResponse({"queue": "pos"})
        assert rr.status == "WAITING"
        assert rr.request_idx == 0

    def test_get_url_none_when_missing(self) -> None:
        rr = RouteResponse({})
        assert rr.get_url() is None
