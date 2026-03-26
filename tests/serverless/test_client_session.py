"""Unit tests for vastai.serverless.client.session.Session."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vastai.serverless.client.client import ServerlessRequest
from vastai.serverless.client.session import Session


class TestSessionInitValidation:
    def test_raises_without_endpoint(self) -> None:
        with pytest.raises(ValueError, match="empty endpoint"):
            Session(
                None,  # type: ignore[arg-type]
                1,
                60.0,
                "e",
                "https://u",
                {},
            )

    def test_raises_without_session_id(self, make_test_endpoint) -> None:
        ep = make_test_endpoint()
        with pytest.raises(ValueError, match="empty session_id"):
            Session(ep, None, 60.0, "e", "https://u", {})  # type: ignore[arg-type]

    def test_raises_without_url(self, make_test_endpoint) -> None:
        ep = make_test_endpoint()
        with pytest.raises(ValueError, match="empty url"):
            Session(ep, 1, 60.0, "e", None, {})  # type: ignore[arg-type]


class TestSessionAsyncContext:
    @pytest.mark.asyncio
    async def test_aenter_returns_self(self, bound_session) -> None:
        _ep, sess = bound_session
        assert await sess.__aenter__() is sess

    @pytest.mark.asyncio
    async def test_aexit_awaits_close(self, bound_session) -> None:
        _ep, sess = bound_session
        with patch.object(sess, "close", new_callable=AsyncMock) as c:
            assert await sess.__aexit__(None, None, None) is False
        c.assert_awaited_once()


class TestSessionIsOpen:
    @pytest.mark.asyncio
    async def test_is_open_updates_open_from_healthcheck(self, bound_session) -> None:
        ep, sess = bound_session
        with patch.object(
            ep,
            "session_healthcheck",
            new_callable=AsyncMock,
            return_value=False,
        ):
            assert await sess.is_open() is False
        assert sess.open is False


class TestSessionClose:
    @pytest.mark.asyncio
    async def test_close_noop_when_already_closed(self, bound_session) -> None:
        _ep, sess = bound_session
        sess.open = False
        with patch.object(sess.endpoint, "close_session", new_callable=AsyncMock) as c:
            await sess.close()
        c.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_noop_when_closing_in_progress(self, bound_session) -> None:
        _ep, sess = bound_session
        sess._closing = True
        with patch.object(sess.endpoint, "close_session", new_callable=AsyncMock) as c:
            await sess.close()
        c.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_calls_endpoint_and_marks_closed(self, bound_session) -> None:
        ep, sess = bound_session
        with patch.object(ep, "close_session", new_callable=AsyncMock) as c:
            await sess.close()
        c.assert_awaited_once_with(sess)
        assert sess.open is False

    @pytest.mark.asyncio
    async def test_close_swallows_endpoint_errors_but_marks_closed(
        self, bound_session
    ) -> None:
        ep, sess = bound_session
        with patch.object(
            ep,
            "close_session",
            new_callable=AsyncMock,
            side_effect=RuntimeError("remote"),
        ):
            await sess.close()
        assert sess.open is False


class TestSessionRequest:
    def test_request_raises_when_session_closed(self, bound_session) -> None:
        """Closed check runs synchronously before a coroutine is returned."""
        _ep, sess = bound_session
        sess.open = False
        with pytest.raises(ValueError, match="closed session"):
            sess.request("/r", {})

    @pytest.mark.asyncio
    async def test_request_awaits_endpoint_and_returns_result(self, bound_session) -> None:
        ep, sess = bound_session
        fut = ServerlessRequest()
        fut.set_result({"ok": True, "status": 200})

        with patch.object(ep, "request", return_value=fut) as req:
            out = await sess.request("/do", {"p": 1}, cost=11, retry=False, stream=True)
        assert out["ok"] is True
        req.assert_called_once()
        assert req.call_args.kwargs["session"] is sess
        assert req.call_args.kwargs["route"] == "/do"
        assert req.call_args.kwargs["payload"] == {"p": 1}

    @pytest.mark.asyncio
    async def test_request_status_410_closes_session(self, bound_session) -> None:
        ep, sess = bound_session
        fut = ServerlessRequest()
        fut.set_result({"status": 410})

        with patch.object(ep, "request", return_value=fut):
            with pytest.raises(ValueError, match="closed session"):
                await sess.request("/r", {})
        assert sess.open is False
