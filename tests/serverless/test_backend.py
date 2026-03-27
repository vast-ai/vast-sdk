"""Unit tests for vastai.serverless.server.lib.backend.Backend.

Covers session HTTP handlers, request forwarding entry points, and small helpers.
All I/O and crypto verification are mocked per unit-test-requirements (no real network).
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientTimeout, web

from vastai.serverless.server.lib.data_types import (
    JsonDataException,
    RequestMetrics,
)

pytestmark = pytest.mark.usefixtures("clear_get_url_cache")

# ---------------------------------------------------------------------------
# Session: health
# ---------------------------------------------------------------------------


class TestBackendSessionHealth:
    """Tests for Backend.session_health_handler."""

    @pytest.mark.asyncio
    async def test_session_health_invalid_json_returns_422(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies session_health_handler returns 422 when request body is not valid JSON.

        This test verifies by:
        1. Using ``serverless_backend_and_handler_default`` for the Backend instance
        2. Calling session_health_handler with a mock request whose json() raises JSONDecodeError
        3. Asserting status 422 and error payload

        Assumptions:
        - No session state is required for this path
        """
        backend, _ = serverless_backend_and_handler_default
        req = serverless_backend_testkit.json_request("{")
        resp = await backend.session_health_handler(req)
        assert resp.status == 422
        body = serverless_backend_testkit.response_json(resp)
        assert body.get("error") == "invalid JSON"

    @pytest.mark.asyncio
    async def test_session_health_missing_session_id_returns_422(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies session_health_handler returns 422 when session_id is absent or empty.

        This test verifies by:
        1. POSTing JSON without session_id
        2. Asserting 422 and missing session_id error

        Assumptions:
        - Empty string session_id is treated as missing (falsy)
        """
        backend, _ = serverless_backend_and_handler_default
        req = serverless_backend_testkit.json_request({"session_auth": "x"})
        resp = await backend.session_health_handler(req)
        assert resp.status == 422
        assert serverless_backend_testkit.response_json(resp).get("error") == "missing session_id"

    @pytest.mark.asyncio
    async def test_session_health_unknown_session_returns_ok_false(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies unknown session_id returns 200 with ok False (not an error status).

        This test verifies by:
        1. Sending a session_id that is not in backend.sessions
        2. Asserting 200 and {"ok": false}

        Assumptions:
        - Handler distinguishes missing session from invalid auth for known sessions
        """
        backend, _ = serverless_backend_and_handler_default
        req = serverless_backend_testkit.json_request({"session_id": "nope", "session_auth": None})
        resp = await backend.session_health_handler(req)
        assert resp.status == 200
        assert serverless_backend_testkit.response_json(resp) == {"ok": False}

    @pytest.mark.asyncio
    async def test_session_health_invalid_auth_returns_401(
        self, serverless_backend_and_handler_default, serverless_backend_testkit, make_pyworker_session
    ) -> None:
        """
        Verifies session_health_handler returns 401 when session_auth does not match.

        This test verifies by:
        1. Inserting a session with auth_data {"secret": 1}
        2. Sending wrong session_auth
        3. Asserting 401

        Assumptions:
        - auth_data equality is compared to session_auth as stored
        """
        backend, _ = serverless_backend_and_handler_default
        sid = "sess1"
        backend.sessions[sid] = make_pyworker_session(
            session_id=sid,
            lifetime=60.0,
            auth_data={"k": "good"},
            expiration=time.time() + 120,
            on_close_route=None,
            on_close_payload=None,
        )
        req = serverless_backend_testkit.json_request({"session_id": sid, "session_auth": {"k": "bad"}})
        resp = await backend.session_health_handler(req)
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_session_health_valid_returns_ok_true(
        self, serverless_backend_and_handler_default, serverless_backend_testkit, make_pyworker_session
    ) -> None:
        """
        Verifies session_health_handler returns 200 ok True when id and auth match.

        This test verifies by:
        1. Storing a session whose auth_data matches session_auth in the request
        2. Asserting 200 and ok True

        Assumptions:
        - session_auth may be a dict matching session.auth_data
        """
        backend, _ = serverless_backend_and_handler_default
        auth = {"token": "abc"}
        sid = "sess2"
        backend.sessions[sid] = make_pyworker_session(
            session_id=sid,
            lifetime=60.0,
            auth_data=auth,
            expiration=time.time() + 120,
            on_close_route=None,
            on_close_payload=None,
        )
        req = serverless_backend_testkit.json_request({"session_id": sid, "session_auth": auth})
        resp = await backend.session_health_handler(req)
        assert resp.status == 200
        assert serverless_backend_testkit.response_json(resp) == {"ok": True}


# ---------------------------------------------------------------------------
# Session: get
# ---------------------------------------------------------------------------


class TestBackendSessionGet:
    """Tests for Backend.session_get_handler."""

    @pytest.mark.asyncio
    async def test_session_get_unknown_session_returns_400(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies session_get_handler returns 400 when session does not exist.

        This test verifies by:
        1. Requesting a non-existent session_id
        2. Asserting 400 and error message

        Assumptions:
        - Unlike health, get uses an error status for missing session
        """
        backend, _ = serverless_backend_and_handler_default
        req = serverless_backend_testkit.json_request(
            {"session_id": "missing", "session_auth": None},
        )
        resp = await backend.session_get_handler(req)
        assert resp.status == 400
        assert "does not exist" in serverless_backend_testkit.response_json(resp).get("error", "")

    @pytest.mark.asyncio
    async def test_session_get_success_returns_session_fields(
        self, serverless_backend_and_handler_default, serverless_backend_testkit, make_pyworker_session
    ) -> None:
        """
        Verifies session_get_handler returns session metadata when auth matches.

        This test verifies by:
        1. Storing a Session with known fields
        2. Calling get with matching session_auth
        3. Asserting JSON includes session_id, lifetime, expiration, on_close fields

        Assumptions:
        - auth_data in response is labeled auth_data in JSON (handler uses auth_data key)
        """
        backend, _ = serverless_backend_and_handler_default
        auth = {"role": "user"}
        sid = "sess3"
        backend.sessions[sid] = make_pyworker_session(
            session_id=sid,
            lifetime=30.0,
            auth_data=auth,
            expiration=12345.0,
            on_close_route="http://cb/end",
            on_close_payload={"a": 1},
            created_at=100.0,
            request_idx=7,
        )
        req = serverless_backend_testkit.json_request({"session_id": sid, "session_auth": auth})
        resp = await backend.session_get_handler(req)
        assert resp.status == 200
        data = serverless_backend_testkit.response_json(resp)
        assert data["session_id"] == sid
        assert data["auth_data"] == auth
        assert data["lifetime"] == 30.0
        assert data["expiration"] == 12345.0
        assert data["on_close_route"] == "http://cb/end"
        assert data["on_close_payload"] == {"a": 1}
        assert data["created_at"] == 100.0

    @pytest.mark.asyncio
    async def test_session_get_missing_session_id_returns_422(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies session_get_handler returns 422 when session_id is missing.

        This test verifies by:
        1. Sending JSON with only session_auth
        2. Asserting 422 and missing session_id error

        Assumptions:
        - Same validation as other session handlers for session_id
        """
        backend, _ = serverless_backend_and_handler_default
        req = serverless_backend_testkit.json_request({"session_auth": {}})
        resp = await backend.session_get_handler(req)
        assert resp.status == 422
        assert serverless_backend_testkit.response_json(resp).get("error") == "missing session_id"

    @pytest.mark.asyncio
    async def test_session_get_invalid_json_returns_422(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies session_get_handler returns 422 when the body is not valid JSON.

        This test verifies by:
        1. Using a mock request whose json() raises JSONDecodeError
        2. Asserting 422 and invalid JSON error

        Assumptions:
        - Same error shape as session_health_handler for decode failures
        """
        backend, _ = serverless_backend_and_handler_default
        req = serverless_backend_testkit.json_request("{")
        resp = await backend.session_get_handler(req)
        assert resp.status == 422
        assert serverless_backend_testkit.response_json(resp).get("error") == "invalid JSON"

    @pytest.mark.asyncio
    async def test_session_get_wrong_auth_returns_401(
        self, serverless_backend_and_handler_default, serverless_backend_testkit, make_pyworker_session
    ) -> None:
        """
        Verifies session_get_handler returns 401 when session_auth does not match.

        This test verifies by:
        1. Storing a session with known auth_data
        2. Calling get with a different session_auth dict
        3. Asserting 401

        Assumptions:
        - Validation matches session_health_handler semantics
        """
        backend, _ = serverless_backend_and_handler_default
        sid = "sg-auth"
        backend.sessions[sid] = make_pyworker_session(
            session_id=sid,
            lifetime=10.0,
            auth_data={"role": "a"},
            expiration=time.time() + 100,
            on_close_route=None,
            on_close_payload=None,
        )
        req = serverless_backend_testkit.json_request(
            {"session_id": sid, "session_auth": {"role": "b"}},
        )
        resp = await backend.session_get_handler(req)
        assert resp.status == 401


# ---------------------------------------------------------------------------
# Session: create / end
# ---------------------------------------------------------------------------


class TestBackendSessionCreate:
    """Tests for Backend.session_create_handler."""

    @pytest.mark.asyncio
    async def test_session_create_invalid_json_returns_422(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies session_create_handler returns 422 on JSON decode errors.

        This test verifies by:
        1. Using a mock request with json() raising JSONDecodeError
        2. Asserting 422 and invalid JSON error

        Assumptions:
        - Handler catches json.JSONDecodeError specifically
        """
        backend, _ = serverless_backend_and_handler_default
        req = serverless_backend_testkit.json_request("not-json")
        resp = await backend.session_create_handler(req)
        assert resp.status == 422
        assert serverless_backend_testkit.response_json(resp).get("error") == "invalid JSON"

    @pytest.mark.asyncio
    async def test_session_create_at_max_sessions_returns_429(
        self, serverless_backend_testkit, make_pyworker_session
    ) -> None:
        """
        Verifies session_create_handler returns 429 when session cap is reached.

        This test verifies by:
        1. Setting max_sessions=1 and pre-populating one session
        2. POSTing a valid create body
        3. Asserting 429

        Assumptions:
        - max_sessions=None and 0 mean unlimited (no reject); use 1 for cap
        """
        backend, _ = serverless_backend_testkit.make_backend(max_sessions=1)
        backend.sessions["existing"] = make_pyworker_session(
            session_id="existing",
            lifetime=1.0,
            auth_data={},
            expiration=time.time() + 10,
            on_close_route=None,
            on_close_payload=None,
        )
        req = serverless_backend_testkit.json_request(
            {
                "auth_data": {"request_idx": 0, "reqnum": 0, "cost": 1.0},
                "payload": {"lifetime": 10.0},
            },
        )
        resp = await backend.session_create_handler(req)
        assert resp.status == 429

    @pytest.mark.asyncio
    async def test_session_create_max_sessions_negative_one_empty_sessions_returns_429(
        self, serverless_backend_testkit
    ) -> None:
        """
        Documents dataclass default ``max_sessions=-1``: cap logic treats only ``None``/``0``
        as unlimited, so ``len(sessions) >= -1`` is true immediately and the first create
        gets 429 (pre-existing product semantics; see peer-review consensus).
        """
        backend, _ = serverless_backend_testkit.make_backend(max_sessions=-1)
        assert len(backend.sessions) == 0
        req = serverless_backend_testkit.json_request(
            {
                "auth_data": {"request_idx": 0, "reqnum": 1, "cost": 1.0},
                "payload": {"lifetime": 10.0},
            },
        )
        resp = await backend.session_create_handler(req)
        assert resp.status == 429

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"auth_data": None, "payload": {"lifetime": 10.0}}, id="auth_null"),
            pytest.param({"payload": {"lifetime": 10.0}}, id="auth_omitted"),
        ],
    )
    @pytest.mark.asyncio
    async def test_session_create_null_or_missing_auth_data_raises_attribute_error(
        self, serverless_backend_and_handler_default, serverless_backend_testkit, body
    ) -> None:
        """
        Documents current behavior: ``auth_data`` must be a mapping; ``null`` or a missing
        key yields ``None`` and ``auth_data.get`` raises (not a structured 4xx).
        """
        backend, _ = serverless_backend_and_handler_default
        req = serverless_backend_testkit.json_request(body)
        with pytest.raises(AttributeError):
            await backend.session_create_handler(req)

    @pytest.mark.asyncio
    async def test_session_create_null_payload_raises_attribute_error(
        self, serverless_backend_and_handler_default, serverless_backend_testkit
    ) -> None:
        """Documents current behavior when ``payload`` JSON is ``null``."""
        backend, _ = serverless_backend_and_handler_default
        req = serverless_backend_testkit.json_request(
            {
                "auth_data": {"request_idx": 0, "reqnum": 1, "cost": 1.0},
                "payload": None,
            },
        )
        with pytest.raises(AttributeError):
            await backend.session_create_handler(req)

    @pytest.mark.asyncio
    async def test_session_create_on_close_route_without_on_close_payload(
        self, serverless_backend_and_handler_default, serverless_backend_testkit
    ) -> None:
        """``on_close_route`` set without ``on_close_payload`` leaves session callback payload None."""
        backend, _ = serverless_backend_and_handler_default
        fixed_id = "sess-route-only"
        body = {
            "auth_data": {"request_idx": 3, "reqnum": 4, "cost": 1.0},
            "payload": {
                "lifetime": 12.0,
                "on_close_route": "http://notify/only-route",
            },
        }
        req = serverless_backend_testkit.json_request(body)
        with patch.object(backend, "generate_session_id", return_value=fixed_id):
            resp = await backend.session_create_handler(req)
        assert resp.status == 201
        stored = backend.sessions[fixed_id]
        assert stored.on_close_route == "http://notify/only-route"
        assert stored.on_close_payload is None

    @pytest.mark.asyncio
    async def test_session_create_omitted_lifetime_defaults_to_sixty(
        self, serverless_backend_and_handler_default, serverless_backend_testkit
    ) -> None:
        """Omitting ``lifetime`` uses default ``60.0`` for ``Session.lifetime`` and expiration."""
        backend, _ = serverless_backend_and_handler_default
        fixed_id = "sess-default-life"
        body = {
            "auth_data": {"request_idx": 4, "reqnum": 5, "cost": 1.0},
            "payload": {},
        }
        req = serverless_backend_testkit.json_request(body)
        before = time.time()
        with patch.object(backend, "generate_session_id", return_value=fixed_id):
            resp = await backend.session_create_handler(req)
        assert resp.status == 201
        stored = backend.sessions[fixed_id]
        assert stored.lifetime == 60.0
        assert stored.expiration == pytest.approx(before + 60.0, abs=2.0)

    @pytest.mark.asyncio
    async def test_session_create_returns_201_with_session_id(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies session_create_handler creates a session and returns 201.

        This test verifies by:
        1. Patching generate_session_id to a fixed id
        2. Sending auth_data and payload with lifetime
        3. Asserting 201, body session_id and expiration, and internal sessions map

        Assumptions:
        - Metrics hooks run without error on real Metrics instance
        """
        backend, _ = serverless_backend_and_handler_default
        fixed_id = "fixedsessionid"
        body = {
            "auth_data": {"request_idx": 1, "reqnum": 2, "cost": 0.5},
            "payload": {"lifetime": 45.0, "on_close_route": None},
        }
        req = serverless_backend_testkit.json_request(body)
        with patch.object(backend, "generate_session_id", return_value=fixed_id):
            resp = await backend.session_create_handler(req)
        assert resp.status == 201
        out = serverless_backend_testkit.response_json(resp)
        assert out["session_id"] == fixed_id
        assert "expiration" in out
        assert fixed_id in backend.sessions
        assert backend.sessions[fixed_id].lifetime == 45.0

    @pytest.mark.asyncio
    async def test_session_create_persists_on_close_route_and_payload(
        self, serverless_backend_and_handler_default, serverless_backend_testkit
    ) -> None:
        """
        Verifies session_create_handler copies on_close_route and on_close_payload into Session.

        This test verifies by:
        1. POSTing payload with both callback fields set
        2. Asserting the stored Session matches

        Assumptions:
        - Branch runs when on_close_route is not None (payload may still omit on_close_payload)
        """
        backend, _ = serverless_backend_and_handler_default
        fixed_id = "sess-close-fields"
        body = {
            "auth_data": {"request_idx": 2, "reqnum": 3, "cost": 1.0},
            "payload": {
                "lifetime": 20.0,
                "on_close_route": "http://internal/session-ended",
                "on_close_payload": {"tag": "x"},
            },
        }
        req = serverless_backend_testkit.json_request(body)
        with patch.object(backend, "generate_session_id", return_value=fixed_id):
            resp = await backend.session_create_handler(req)
        assert resp.status == 201
        stored = backend.sessions[fixed_id]
        assert stored.on_close_route == "http://internal/session-ended"
        assert stored.on_close_payload == {"tag": "x"}


class TestBackendSessionEnd:
    """Tests for Backend.session_end_handler."""

    @pytest.mark.asyncio
    async def test_session_end_not_found_returns_400(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies session_end_handler returns 400 when session_id is unknown.

        This test verifies by:
        1. POSTing end for a missing session with arbitrary auth
        2. Asserting 400

        Assumptions:
        - Error is returned while holding the sessions lock (before close)
        """
        backend, _ = serverless_backend_and_handler_default
        req = serverless_backend_testkit.json_request(
            {"session_id": "ghost", "session_auth": {"x": 1}},
        )
        resp = await backend.session_end_handler(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_session_end_invalid_json_returns_422(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies session_end_handler returns 422 on invalid JSON.

        This test verifies by:
        1. Using a mock request whose json() raises JSONDecodeError
        2. Asserting status 422

        Assumptions:
        - Handler uses the same JSON error pattern as session health
        """
        backend, _ = serverless_backend_and_handler_default
        req = serverless_backend_testkit.json_request("not-json")
        resp = await backend.session_end_handler(req)
        assert resp.status == 422

    @pytest.mark.asyncio
    async def test_session_end_missing_session_id_returns_422(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies session_end_handler returns 422 when session_id is missing.

        This test verifies by:
        1. Sending a JSON object without session_id
        2. Asserting 422

        Assumptions:
        - Empty string session_id counts as missing (falsy)
        """
        backend, _ = serverless_backend_and_handler_default
        req = serverless_backend_testkit.json_request({"session_auth": {}})
        resp = await backend.session_end_handler(req)
        assert resp.status == 422

    @pytest.mark.asyncio
    async def test_session_end_wrong_auth_returns_401(
        self,
        serverless_backend_and_handler_default,
        serverless_backend_testkit,
        make_pyworker_session,
        make_patch_mock_backend_close_session,
    ) -> None:
        """
        Verifies session_end_handler returns 401 when session_auth does not match.

        This test verifies by:
        1. Storing a session with known auth_data
        2. POSTing end with a different session_auth
        3. Asserting 401 before any close runs

        Assumptions:
        - Validation occurs under the sessions lock before __close_session
        """
        backend, _ = serverless_backend_and_handler_default
        sid = "s1"
        backend.sessions[sid] = make_pyworker_session(
            session_id=sid,
            lifetime=60.0,
            auth_data={"ok": True},
            expiration=time.time() + 100,
            on_close_route=None,
            on_close_payload=None,
        )
        req = serverless_backend_testkit.json_request({"session_id": sid, "session_auth": {"ok": False}})
        with make_patch_mock_backend_close_session(backend) as mock_close:
            resp = await backend.session_end_handler(req)
        mock_close.assert_not_awaited()
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_session_end_success_removes_session(
        self,
        serverless_backend_and_handler_default,
        serverless_backend_testkit,
        make_pyworker_session,
        register_pyworker_backend_session_with_metrics,
        make_patch_skip_backend_run_session_on_close,
    ) -> None:
        """
        Verifies session_end_handler closes session and returns 200 ended true.

        This test verifies by:
        1. Inserting a session with matching auth
        2. Patching __run_session_on_close to avoid HTTP
        3. Calling session_end_handler and asserting session removed from backend.sessions

        Assumptions:
        - __close_session runs metrics updates; real Metrics is acceptable
        """
        backend, _ = serverless_backend_and_handler_default
        auth = {"t": 1}
        sid = "to-close"
        sess = make_pyworker_session(
            session_id=sid,
            lifetime=60.0,
            auth_data=auth,
            expiration=time.time() + 100,
            on_close_route=None,
            on_close_payload=None,
        )
        register_pyworker_backend_session_with_metrics(backend, sess)
        req = serverless_backend_testkit.json_request({"session_id": sid, "session_auth": auth})
        with make_patch_skip_backend_run_session_on_close(backend):
            resp = await backend.session_end_handler(req)
        assert resp.status == 200
        data = serverless_backend_testkit.response_json(resp)
        assert data.get("ended") is True
        assert data.get("removed_session") == sid
        assert sid not in backend.sessions

    @pytest.mark.asyncio
    async def test_session_end_returns_410_when_close_returns_false(
        self,
        serverless_backend_and_handler_default,
        serverless_backend_testkit,
        make_pyworker_session,
        make_patch_mock_backend_close_session,
    ) -> None:
        """
        Verifies session_end_handler returns 410 if the session vanishes before __close_session.

        This test verifies by:
        1. Passing auth checks with a present session
        2. Patching __close_session to return False (simulating a concurrent close)
        3. Asserting 410 and 'already closed' error

        Assumptions:
        - __close_session returns False when the session id is no longer in self.sessions
        """
        backend, _ = serverless_backend_and_handler_default
        auth = {"same": True}
        sid = "double-end"
        backend.sessions[sid] = make_pyworker_session(
            session_id=sid,
            lifetime=60.0,
            auth_data=auth,
            expiration=time.time() + 100,
            on_close_route=None,
            on_close_payload=None,
        )
        req = serverless_backend_testkit.json_request({"session_id": sid, "session_auth": auth})
        with make_patch_mock_backend_close_session(backend) as mock_close:
            mock_close.return_value = False
            resp = await backend.session_end_handler(req)
        assert resp.status == 410
        err = serverless_backend_testkit.response_json(resp).get("error", "")
        assert "already closed" in err


class TestBackendCloseSession:
    """Direct tests for Backend.__close_session (transport teardown, removal)."""

    @pytest.mark.asyncio
    async def test_close_session_closes_open_request_transports(
        self,
        serverless_backend_and_handler_default,
        make_pyworker_session,
        register_pyworker_backend_session_with_metrics,
        make_patch_skip_backend_run_session_on_close,
    ) -> None:
        """
        Verifies __close_session closes each in-flight request transport when open.

        This test verifies by:
        1. Building a session whose requests list holds a mock with transport.is_closing False
        2. Awaiting __close_session and asserting transport.close() was called
        3. Asserting the session is removed from backend.sessions

        Assumptions:
        - Transport close failures are swallowed inside __close_session
        """
        backend, _ = serverless_backend_and_handler_default
        mock_tr = MagicMock()
        mock_tr.is_closing.return_value = False
        mock_req = MagicMock()
        mock_req.transport = mock_tr
        sid = "sess-transport"
        sess = make_pyworker_session(
            session_id=sid,
            lifetime=1.0,
            auth_data={},
            expiration=time.time() + 60,
            on_close_route=None,
            on_close_payload=None,
            requests=[mock_req],
        )
        register_pyworker_backend_session_with_metrics(backend, sess)
        with make_patch_skip_backend_run_session_on_close(backend):
            removed = await backend._Backend__close_session(sid)
        assert removed is True
        assert sid not in backend.sessions
        mock_tr.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_session_swallows_exception_when_transport_close_fails(
        self,
        serverless_backend_and_handler_default,
        make_pyworker_session,
        register_pyworker_backend_session_with_metrics,
        make_patch_skip_backend_run_session_on_close,
    ) -> None:
        """
        Verifies errors from ``transport.close()`` are ignored so shutdown continues.

        This test verifies by:
        1. Making ``close`` raise ``OSError`` while ``is_closing`` is False
        2. Asserting the session is still removed and __close_session returns True

        Assumptions:
        - Per-request teardown must not abort the whole session close on transport errors
        """
        backend, _ = serverless_backend_and_handler_default
        mock_tr = MagicMock()
        mock_tr.is_closing.return_value = False
        mock_tr.close.side_effect = OSError("close failed")
        mock_req = MagicMock()
        mock_req.transport = mock_tr
        sid = "sess-tr-close-err"
        sess = make_pyworker_session(
            session_id=sid,
            lifetime=1.0,
            auth_data={},
            expiration=time.time() + 60,
            on_close_route=None,
            on_close_payload=None,
            requests=[mock_req],
        )
        register_pyworker_backend_session_with_metrics(backend, sess)
        with make_patch_skip_backend_run_session_on_close(backend):
            removed = await backend._Backend__close_session(sid)
        assert removed is True
        assert sid not in backend.sessions

    @pytest.mark.asyncio
    async def test_close_session_unknown_session_returns_false(
        self, serverless_backend_and_handler_default
    ) -> None:
        """
        Verifies __close_session returns False when the session id is not present.

        This test verifies by:
        1. Calling ``_Backend__close_session`` with an id that was never inserted
        2. Asserting the return value is False and no exception is raised

        Assumptions:
        - ``pop(session_id, None)`` is the source of truth for absence
        """
        backend, _ = serverless_backend_and_handler_default
        removed = await backend._Backend__close_session("no-such-session")
        assert removed is False

    @pytest.mark.asyncio
    async def test_close_session_skips_transport_close_when_already_closing(
        self,
        serverless_backend_and_handler_default,
        make_pyworker_session,
        register_pyworker_backend_session_with_metrics,
        make_patch_skip_backend_run_session_on_close,
    ) -> None:
        """
        Verifies transports that report ``is_closing()`` are not closed again.

        This test verifies by:
        1. Using a mock transport with ``is_closing`` returning True
        2. Awaiting __close_session and asserting ``close`` was never called

        Assumptions:
        - Branch matches ``tr is not None and not tr.is_closing()`` in __close_session
        """
        backend, _ = serverless_backend_and_handler_default
        mock_tr = MagicMock()
        mock_tr.is_closing.return_value = True
        mock_req = MagicMock()
        mock_req.transport = mock_tr
        sid = "sess-tr-closing"
        sess = make_pyworker_session(
            session_id=sid,
            lifetime=1.0,
            auth_data={},
            expiration=time.time() + 60,
            on_close_route=None,
            on_close_payload=None,
            requests=[mock_req],
        )
        register_pyworker_backend_session_with_metrics(backend, sess)
        with make_patch_skip_backend_run_session_on_close(backend):
            await backend._Backend__close_session(sid)
        mock_tr.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_session_skips_transport_when_request_has_no_transport(
        self,
        serverless_backend_and_handler_default,
        make_pyworker_session,
        register_pyworker_backend_session_with_metrics,
        make_patch_skip_backend_run_session_on_close,
    ) -> None:
        """
        Verifies in-flight aiohttp requests without a transport do not trigger close.

        This test verifies by:
        1. Storing a session whose ``requests`` entry has ``transport`` set to None
        2. Completing __close_session without error

        Assumptions:
        - ``getattr(req, "transport", None)`` yields None and the inner branch is skipped
        """
        backend, _ = serverless_backend_and_handler_default
        mock_req = MagicMock(spec=[])  # no transport attribute
        sid = "sess-no-tr"
        sess = make_pyworker_session(
            session_id=sid,
            lifetime=1.0,
            auth_data={},
            expiration=time.time() + 60,
            on_close_route=None,
            on_close_payload=None,
            requests=[mock_req],
        )
        register_pyworker_backend_session_with_metrics(backend, sess)
        with make_patch_skip_backend_run_session_on_close(backend):
            removed = await backend._Backend__close_session(sid)
        assert removed is True

    @pytest.mark.asyncio
    async def test_close_session_swallows_exception_from_run_session_on_close(
        self,
        serverless_backend_and_handler_default,
        make_pyworker_session,
        register_pyworker_backend_session_with_metrics,
    ) -> None:
        """
        Verifies exceptions from __run_session_on_close do not escape __close_session.

        This test verifies by:
        1. Patching ``_Backend__run_session_on_close`` to raise ``RuntimeError``
        2. Asserting __close_session still returns True and removes the session

        Assumptions:
        - The broad ``except Exception: pass`` around the on_close await is intentional
        """
        backend, _ = serverless_backend_and_handler_default
        sid = "sess-on-close-err"
        sess = make_pyworker_session(
            session_id=sid,
            lifetime=1.0,
            auth_data={},
            expiration=time.time() + 60,
            on_close_route=None,
            on_close_payload=None,
        )
        register_pyworker_backend_session_with_metrics(backend, sess)
        with patch.object(backend, "_Backend__run_session_on_close", new_callable=AsyncMock) as mock_on_close:
            mock_on_close.side_effect = RuntimeError("callback exploded")
            removed = await backend._Backend__close_session(sid)
        assert removed is True
        assert sid not in backend.sessions


# ---------------------------------------------------------------------------
# __handle_request (via create_handler)
# ---------------------------------------------------------------------------


class TestBackendHandleRequest:
    """Tests for Backend.create_handler / __handle_request."""

    @pytest.mark.asyncio
    async def test_handle_request_json_decode_error_returns_422(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies the endpoint handler returns 422 when body is not valid JSON.

        This test verifies by:
        1. Creating handler_fn via create_handler
        2. Passing a request whose json() raises JSONDecodeError
        3. Asserting 422

        Assumptions:
        - unsecured=True so signature check does not block earlier paths
        """
        backend, handler = serverless_backend_and_handler_default
        fn = backend.create_handler(handler)
        req = serverless_backend_testkit.json_request("[[[")
        resp = await fn(req)
        assert resp.status == 422

    @pytest.mark.asyncio
    async def test_handle_request_json_data_exception_returns_422(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies JsonDataException from handler.get_data_from_request yields 422.

        This test verifies by:
        1. Patching handler.get_data_from_request to raise JsonDataException
        2. Calling the wrapped handler with valid JSON object
        3. Asserting 422 and message payload

        Assumptions:
        - Exception message is passed as json_response data= for this exception type
        """
        backend, handler = serverless_backend_and_handler_default
        fn = backend.create_handler(handler)
        req = serverless_backend_testkit.json_request({"any": "body"})

        def _raise_json_data_exc(cls, req_data):
            raise JsonDataException({"field": "bad"})

        with patch.object(
            type(handler),
            "get_data_from_request",
            classmethod(_raise_json_data_exc),
        ):
            resp = await fn(req)
        assert resp.status == 422
        assert serverless_backend_testkit.response_json(resp) == {"field": "bad"}

    @pytest.mark.asyncio
    async def test_handle_request_invalid_session_returns_410(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies requests with session_id for unknown session return 410.

        This test verifies by:
        1. Sending valid auth_data and payload plus session_id not in backend.sessions
        2. Patching __call_backend so it would not be reached incorrectly
        3. Asserting 410

        Assumptions:
        - unsecured=True; signature check passes without pubkey
        """
        backend, handler = serverless_backend_and_handler_default
        fn = backend.create_handler(handler)
        data = serverless_backend_testkit.auth_payload()
        data["session_id"] = "no-such-session"
        req = serverless_backend_testkit.json_request(data)
        with patch.object(backend, "_Backend__call_backend", new_callable=AsyncMock):
            resp = await fn(req)
        assert resp.status == 410

    @pytest.mark.asyncio
    async def test_handle_request_secured_without_pubkey_returns_401(self, serverless_backend_testkit) -> None:
        """
        Verifies secured mode rejects when public key was never loaded.

        This test verifies by:
        1. Building backend with unsecured=False and _pubkey None
        2. Sending a syntactically valid request
        3. Asserting 401

        Assumptions:
        - __check_signature returns False when _pubkey is None and not unsecured
        """
        backend, handler = serverless_backend_testkit.make_backend(unsecured=False)
        backend._pubkey = None
        fn = backend.create_handler(handler)
        req = serverless_backend_testkit.json_request(serverless_backend_testkit.auth_payload())
        resp = await fn(req)
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_handle_request_success_calls_backend_and_returns_response(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies happy path calls __call_backend and generate_client_response.

        This test verifies by:
        1. Patching __call_backend to return a mock ClientResponse
        2. Patching handler.generate_client_response to return a fixed web.Response
        3. Asserting returned status and body match the patched response

        Assumptions:
        - allow_parallel_requests=True so queue wait is skipped
        """
        backend, handler = serverless_backend_and_handler_default
        fn = backend.create_handler(handler)
        req = serverless_backend_testkit.json_request(serverless_backend_testkit.auth_payload())
        mock_model_resp = MagicMock()
        mock_model_resp.status = 200
        expected = web.json_response({"result": "ok"}, status=200)
        with patch.object(backend, "_Backend__call_backend", new_callable=AsyncMock) as mock_back:
            mock_back.return_value = mock_model_resp
            with patch.object(
                handler,
                "generate_client_response",
                new_callable=AsyncMock,
                return_value=expected,
            ):
                resp = await fn(req)
        assert resp.status == 200
        assert serverless_backend_testkit.response_json(resp) == {"result": "ok"}
        mock_back.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_request_max_queue_time_exceeded_returns_429(self, serverless_backend_testkit) -> None:
        """
        Verifies __handle_request returns 429 when model wait_time exceeds handler cap.

        This test verifies by:
        1. Using a handler with max_queue_time set and allow_parallel True
        2. Seeding metrics.model_metrics.requests_working and low max_throughput so wait_time is huge
        3. Asserting 429 without calling the model

        Assumptions:
        - wait_time is derived from pending workloads / max_throughput (see ModelMetrics)
        """
        backend, handler = serverless_backend_testkit.make_backend(max_queue_time=10.0)
        fn = backend.create_handler(handler)
        rm = RequestMetrics(
            request_idx=0, reqnum=99, workload=100.0, status="Started"
        )
        backend.metrics.model_metrics.requests_working[99] = rm
        backend.metrics.model_metrics.max_throughput = 0.00001
        req = serverless_backend_testkit.json_request(serverless_backend_testkit.auth_payload())
        with patch.object(backend.metrics, "_request_reject") as mock_reject:
            with patch.object(backend, "_Backend__call_backend", new_callable=AsyncMock) as mock_cb:
                resp = await fn(req)
        assert resp.status == 429
        mock_cb.assert_not_awaited()
        mock_reject.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_request_model_error_returns_500(self, serverless_backend_and_handler_default, serverless_backend_testkit) -> None:
        """
        Verifies exceptions from generate_client_response become HTTP 500.

        This test verifies by:
        1. Patching __call_backend to return a dummy model response
        2. Patching generate_client_response to raise RuntimeError
        3. Asserting response status 500

        Assumptions:
        - make_request catches non-cancel exceptions and returns web.Response(500)
        """
        backend, handler = serverless_backend_and_handler_default
        fn = backend.create_handler(handler)
        req = serverless_backend_testkit.json_request(serverless_backend_testkit.auth_payload())
        mock_model = MagicMock()
        with patch.object(backend.metrics, "_request_errored") as mock_errored:
            with patch.object(backend, "_Backend__call_backend", new_callable=AsyncMock) as mock_cb:
                mock_cb.return_value = mock_model
                with patch.object(
                    handler,
                    "generate_client_response",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("model exploded"),
                ):
                    resp = await fn(req)
        assert resp.status == 500
        mock_errored.assert_called_once()
        assert "model exploded" in mock_errored.call_args[0][1]

    @pytest.mark.asyncio
    async def test_handle_request_call_api_uses_session_post(
        self, serverless_backend_and_handler_default, serverless_backend_testkit, make_mock_model_response
    ) -> None:
        """
        Verifies the non-remote path posts JSON to handler.endpoint via ClientSession.

        This test verifies by:
        1. Replacing backend.session with a mock whose post() returns an async context manager
        2. Letting __call_backend run (no patch) and using the default generate_client_response
        3. Asserting session.post awaited with url and json from the payload

        Assumptions:
        - Generic handler reads non-streaming bodies via model_response.read()
        """
        backend, handler = serverless_backend_and_handler_default
        fn = backend.create_handler(handler)
        req = serverless_backend_testkit.json_request(serverless_backend_testkit.auth_payload())
        mock_resp = make_mock_model_response(body=b'{"model": true}')
        mock_sess = MagicMock()
        # Backend.__call_api does `return await self.session.post(...)` (awaitable, not async-with).
        mock_sess.post = AsyncMock(return_value=mock_resp)
        object.__setattr__(backend, "session", mock_sess)
        resp = await fn(req)
        assert resp.status == 200
        mock_sess.post.assert_awaited_once()
        assert mock_sess.post.await_args.kwargs["url"] == handler.endpoint
        assert mock_sess.post.await_args.kwargs["json"] == {"input": {}}

    @pytest.mark.asyncio
    async def test_handle_request_remote_dispatch_wraps_result(self, serverless_backend_testkit) -> None:
        """
        Verifies remote_dispatch_function path wraps return value as JSON ClientResponse-like.

        This test verifies by:
        1. Building a handler with remote_function returning a plain dict
        2. Running create_handler without patching __call_backend
        3. Asserting client response body contains the remote result (via default handler read())

        Assumptions:
        - __call_remote_dispatch builds RemoteDispatchClientResponse with read()
        """

        async def remote_fn(**params):
            return {"dispatched": True, "params": params}

        backend, handler = serverless_backend_testkit.make_backend(remote_function=remote_fn)
        fn = backend.create_handler(handler)
        req = serverless_backend_testkit.json_request(serverless_backend_testkit.auth_payload())
        resp = await fn(req)
        assert resp.status == 200
        body = serverless_backend_testkit.response_json(resp)
        assert body["result"]["dispatched"] is True
        assert body["result"]["params"] == {"input": {}}

    @pytest.mark.asyncio
    async def test_handle_request_verified_signature_allows_request(
        self, serverless_backend_testkit, make_serverless_test_rsa_key
    ) -> None:
        """
        Verifies secured mode accepts a correctly signed auth_data payload.

        This test verifies by:
        1. Generating an RSA key pair and signing the canonical url message
        2. Setting backend._pubkey to the public key and unsecured False
        3. Patching __call_backend and asserting it is awaited (signature path passed)

        Assumptions:
        - Message format matches __check_signature (json.dumps with indent=4, sort_keys=True)
        """
        key = make_serverless_test_rsa_key()
        url = "https://tenant.example/v1/predict"
        backend, handler = serverless_backend_testkit.make_backend(unsecured=False)
        backend._pubkey = key.publickey()
        fn = backend.create_handler(handler)
        req = serverless_backend_testkit.json_request(serverless_backend_testkit.signed_auth(url, key))
        mock_model = MagicMock()
        with patch.object(backend, "_Backend__call_backend", new_callable=AsyncMock) as mock_cb:
            mock_cb.return_value = mock_model
            with patch.object(
                handler,
                "generate_client_response",
                new_callable=AsyncMock,
                return_value=web.json_response({"ok": 1}),
            ):
                resp = await fn(req)
        assert resp.status == 200
        mock_cb.assert_awaited_once()
        assert backend.reqnum >= 7

    @pytest.mark.asyncio
    async def test_handle_request_fifo_mode_single_request_succeeds(self, serverless_backend_testkit) -> None:
        """
        Verifies queued (non-parallel) handler still completes when the request is alone.

        This test verifies by:
        1. Using allow_parallel_requests=False so the FIFO branch runs
        2. Patching __call_backend and generate_client_response as in the parallel happy path
        3. Asserting 200 response

        Assumptions:
        - A sole queued request is head-of-line and its Event is set before wait
        """
        backend, handler = serverless_backend_testkit.make_backend(allow_parallel=False)
        fn = backend.create_handler(handler)
        req = serverless_backend_testkit.json_request(serverless_backend_testkit.auth_payload())
        mock_model = MagicMock()
        with patch.object(backend, "_Backend__call_backend", new_callable=AsyncMock) as mock_cb:
            mock_cb.return_value = mock_model
            with patch.object(
                handler,
                "generate_client_response",
                new_callable=AsyncMock,
                return_value=web.json_response({"queued": False}),
            ):
                resp = await fn(req)
        assert resp.status == 200
        assert serverless_backend_testkit.response_json(resp) == {"queued": False}

    @pytest.mark.asyncio
    async def test_handle_request_fifo_two_concurrent_requests_both_succeed(
        self, serverless_backend_testkit
    ) -> None:
        """
        Verifies FIFO mode processes a second request after the first completes.

        This test verifies by:
        1. Starting the first handler call so it begins model work
        2. Starting the second (waits on queue) before the first finishes
        3. Asserting both return 200 in some order

        Assumptions:
        - advance_queue_after_completion wakes the next waiter when the head finishes
        """
        backend, handler = serverless_backend_testkit.make_backend(allow_parallel=False)
        fn = backend.create_handler(handler)
        mock_model = MagicMock()
        first_in_backend = asyncio.Event()
        release_first = asyncio.Event()
        call_count = {"n": 0}

        async def _gated_backend(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                first_in_backend.set()
                await release_first.wait()
            return mock_model

        req1 = serverless_backend_testkit.json_request(serverless_backend_testkit.auth_payload(reqnum=1))
        req2 = serverless_backend_testkit.json_request(serverless_backend_testkit.auth_payload(reqnum=2))
        with patch.object(backend, "_Backend__call_backend", side_effect=_gated_backend):
            with patch.object(
                handler,
                "generate_client_response",
                new_callable=AsyncMock,
                return_value=web.json_response({"ok": True}),
            ):
                t1 = asyncio.create_task(fn(req1))
                await first_in_backend.wait()
                t2 = asyncio.create_task(fn(req2))
                await asyncio.sleep(0)
                release_first.set()
                r1, r2 = await asyncio.gather(t1, t2)
        assert r1.status == 200
        assert r2.status == 200

    @pytest.mark.asyncio
    async def test_handle_request_fifo_cancelled_during_queue_wait_returns_499(
        self, serverless_backend_testkit, vast_serverless_backend_module
    ) -> None:
        """
        Verifies FIFO mode maps client disconnect (CancelledError) to HTTP 499.

        This test verifies by:
        1. Patching ``asyncio.Event`` in the backend module so ``wait`` raises ``CancelledError``
        2. Invoking the wrapped handler in non-parallel mode
        3. Asserting status 499 and ``_request_canceled`` on metrics

        Assumptions:
        - aiohttp handler cancellation surfaces as ``CancelledError`` from ``event.wait()``
        """
        backend, handler = serverless_backend_testkit.make_backend(allow_parallel=False)
        fn = backend.create_handler(handler)
        req = serverless_backend_testkit.json_request(serverless_backend_testkit.auth_payload())

        mock_ev = MagicMock()
        mock_ev.wait = AsyncMock(side_effect=asyncio.CancelledError())
        mock_ev.set = MagicMock()

        with patch.object(vast_serverless_backend_module.asyncio, "Event", return_value=mock_ev):
            with patch.object(backend.metrics, "_request_canceled") as mock_cancel:
                resp = await fn(req)

        assert resp.status == 499
        mock_cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_request_fifo_create_task_failure_returns_500(
        self,
        serverless_backend_testkit,
        vast_serverless_backend_module,
        serverless_create_task_side_effect_close_coro,
    ) -> None:
        """
        Verifies unexpected errors after the queue wait yield HTTP 500 from the outer handler.

        This test verifies by:
        1. Using a single FIFO request so ``event.wait`` resolves immediately
        2. Patching ``create_task`` in the backend module to raise ``RuntimeError``
        3. Asserting response status 500

        Assumptions:
        - The ``except Exception`` block in ``__handle_request`` covers failures between wait and work completion
        """
        backend, handler = serverless_backend_testkit.make_backend(allow_parallel=False)
        fn = backend.create_handler(handler)
        req = serverless_backend_testkit.json_request(serverless_backend_testkit.auth_payload())

        with patch.object(
            vast_serverless_backend_module,
            "create_task",
            side_effect=serverless_create_task_side_effect_close_coro(),
        ):
            resp = await fn(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_handle_request_invalid_signature_returns_401(
        self, serverless_backend_testkit, make_serverless_test_rsa_key
    ) -> None:
        """
        Verifies secured mode rejects when the signature does not match the claimed URL.

        This test verifies by:
        1. Setting a real RSA public key on the backend
        2. Sending a PKCS1 signature over a different URL than ``auth_data.url``
        3. Asserting 401 without calling the model

        Assumptions:
        - ``__check_signature`` returns False when verify fails (cryptographically wrong sig)
        """
        key = make_serverless_test_rsa_key()
        backend, handler = serverless_backend_testkit.make_backend(unsecured=False)
        backend._pubkey = key.publickey()
        fn = backend.create_handler(handler)
        signed_url = "https://tenant.example/v1/predict"
        body = serverless_backend_testkit.signed_auth(signed_url, key)
        body["auth_data"]["url"] = "https://other.example/different"
        req = serverless_backend_testkit.json_request(body)
        with patch.object(backend, "_Backend__call_backend", new_callable=AsyncMock) as mock_cb:
            resp = await fn(req)
        assert resp.status == 401
        mock_cb.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handle_request_session_request_extends_expiration(
        self,
        serverless_backend_testkit,
        make_pyworker_session,
    ) -> None:
        """
        Verifies an authenticated session request extends expiration and counts reqnums.

        This test verifies by:
        1. Seeding backend.sessions with a known session_id and lifetime
        2. POSTing handler payload that includes that session_id
        3. Asserting expiration increased by lifetime and session_reqnum incremented

        Assumptions:
        - Generic handler get_data_from_request passes session_id through from top-level JSON
        """
        backend, handler = serverless_backend_testkit.make_backend()
        fn = backend.create_handler(handler)
        sid = "live-sess"
        exp_before = 1_700_000_000.0
        sess = make_pyworker_session(
            session_id=sid,
            lifetime=30.0,
            auth_data={},
            expiration=exp_before,
            on_close_route=None,
            on_close_payload=None,
            session_reqnum=0,
        )
        backend.sessions[sid] = sess
        data = serverless_backend_testkit.auth_payload()
        data["session_id"] = sid
        req = serverless_backend_testkit.json_request(data)
        mock_model = MagicMock()
        with patch.object(backend, "_Backend__call_backend", new_callable=AsyncMock) as mock_cb:
            mock_cb.return_value = mock_model
            with patch.object(
                handler,
                "generate_client_response",
                new_callable=AsyncMock,
                return_value=web.json_response({"s": "ok"}),
            ):
                resp = await fn(req)
        assert resp.status == 200
        assert sess.expiration == exp_before + sess.lifetime
        assert sess.session_reqnum == 1
        mock_cb.assert_awaited_once()


# ---------------------------------------------------------------------------
# Session garbage collection
# ---------------------------------------------------------------------------


class TestBackendSessionGc:
    """Tests for Backend.__session_gc_loop periodic cleanup."""

    @pytest.mark.asyncio
    async def test_session_gc_loop_closes_expired_sessions(
        self,
        serverless_backend_and_handler_default,
        make_pyworker_session,
        register_pyworker_backend_session_with_metrics,
        vast_serverless_backend_module,
        make_patch_skip_backend_run_session_on_close,
    ) -> None:
        """
        Verifies the GC loop removes sessions whose expiration is in the past.

        This test verifies by:
        1. Inserting a session with expiration already elapsed
        2. Patching backend.sleep to return immediately so the loop ticks
        3. Running the loop briefly and asserting the session was closed/removed

        Assumptions:
        - __session_gc_loop uses module-level asyncio.sleep imported as sleep
        """
        backend, _ = serverless_backend_and_handler_default
        sid = "expired-gc"
        sess = make_pyworker_session(
            session_id=sid,
            lifetime=10.0,
            auth_data={},
            expiration=time.time() - 1.0,
            on_close_route=None,
            on_close_payload=None,
        )
        register_pyworker_backend_session_with_metrics(backend, sess)
        session_removed = asyncio.Event()
        orig_close = backend._Backend__close_session

        async def _close_then_signal(session_id: str):
            try:
                return await orig_close(session_id)
            finally:
                if session_id == sid:
                    session_removed.set()

        async def _yield_only(_delay: float = 0) -> None:
            await asyncio.sleep(0)

        with patch.object(vast_serverless_backend_module, "sleep", side_effect=_yield_only):
            with make_patch_skip_backend_run_session_on_close(backend):
                with patch.object(backend, "_Backend__close_session", _close_then_signal):
                    task = asyncio.create_task(backend._Backend__session_gc_loop())
                    await asyncio.wait_for(session_removed.wait(), timeout=2.0)
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        assert sid not in backend.sessions

    @pytest.mark.asyncio
    async def test_session_gc_loop_propagates_cancellation(
        self, serverless_backend_and_handler_default, vast_serverless_backend_module
    ) -> None:
        """
        Verifies ``CancelledError`` from the GC sleep exits the loop (task teardown).

        This test verifies by:
        1. Patching ``sleep`` to raise ``CancelledError`` on the first iteration
        2. Awaiting ``__session_gc_loop`` and expecting ``CancelledError`` to propagate

        Assumptions:
        - The loop re-raises cancellation so asyncio can retire the background task
        """
        backend, _ = serverless_backend_and_handler_default

        async def sleep_raises_cancelled(_delay: float) -> None:
            raise asyncio.CancelledError()

        with patch.object(vast_serverless_backend_module, "sleep", side_effect=sleep_raises_cancelled):
            with pytest.raises(asyncio.CancelledError):
                await backend._Backend__session_gc_loop()


# ---------------------------------------------------------------------------
# Helpers and metrics
# ---------------------------------------------------------------------------


class TestBackendHelpers:
    """Tests for generate_session_id, backend_errored, and __run_session_on_close shape."""

    def test_generate_session_id_length_and_charset(self, serverless_backend_and_handler_default) -> None:
        """
        Verifies generate_session_id returns 13 alphanumeric characters.

        This test verifies by:
        1. Calling generate_session_id many times with patched random.choices
        2. Asserting length 13 and allowed character set

        Assumptions:
        - Implementation uses string.ascii_letters + digits and k=13
        """
        backend, _ = serverless_backend_and_handler_default
        with patch("vastai.serverless.server.lib.backend.random.choices") as mock_choices:
            mock_choices.return_value = list("abcdefghijklm")
            sid = backend.generate_session_id()
        assert len(sid) == 13
        assert sid == "abcdefghijklm"
        mock_choices.assert_called_once()
        args, kwargs = mock_choices.call_args
        assert kwargs.get("k") == 13

    def test_backend_errored_forwards_to_metrics(self, serverless_backend_and_handler_default) -> None:
        """
        Verifies backend_errored delegates to metrics._model_errored.

        This test verifies by:
        1. Patching metrics._model_errored on the backend instance
        2. Calling backend_errored("msg")
        3. Asserting the mock was called with the message

        Assumptions:
        - Metrics is constructed in __post_init__
        """
        backend, _ = serverless_backend_and_handler_default
        with patch.object(backend.metrics, "_model_errored") as mock_err:
            backend.backend_errored("failure-reason")
        mock_err.assert_called_once_with("failure-reason")

    @pytest.mark.asyncio
    async def test_run_session_on_close_posts_json_with_session_id(
        self,
        serverless_backend_and_handler_default,
        make_pyworker_session,
        attach_serverless_backend_mock_session_post,
    ) -> None:
        """
        Verifies __run_session_on_close POSTs merged body including session_id.

        This test verifies by:
        1. Building a session with on_close_route and dict on_close_payload
        2. Replacing backend.session with a mock ClientSession whose post returns async context
        3. Awaiting __run_session_on_close and asserting post called with json containing session_id

        Assumptions:
        - ClientSession.post is used with json= and timeout; response text is read
        """
        backend, _ = serverless_backend_and_handler_default
        session = make_pyworker_session(
            session_id="cb1",
            lifetime=1.0,
            auth_data={},
            expiration=time.time() + 10,
            on_close_route="http://internal/hook",
            on_close_payload={"foo": "bar"},
        )
        mock_sess = attach_serverless_backend_mock_session_post(backend, response_text="ok")
        await backend._Backend__run_session_on_close(session)
        mock_sess.post.assert_called_once()
        call_kw = mock_sess.post.call_args.kwargs
        assert call_kw["url"] == "http://internal/hook"
        assert call_kw["timeout"] == ClientTimeout(total=10)
        body = call_kw["json"]
        assert body["foo"] == "bar"
        assert body["session_id"] == "cb1"

    @pytest.mark.asyncio
    async def test_run_session_on_close_wraps_scalar_payload(
        self,
        serverless_backend_and_handler_default,
        make_pyworker_session,
        attach_serverless_backend_mock_session_post,
    ) -> None:
        """
        Verifies __run_session_on_close wraps non-dict on_close_payload under 'payload'.

        This test verifies by:
        1. Using on_close_payload that is a string scalar
        2. Inspecting the JSON body sent to POST

        Assumptions:
        - Non-dict branches use body = {"payload": on_close_payload} plus session_id
        """
        backend, _ = serverless_backend_and_handler_default
        session = make_pyworker_session(
            session_id="cb2",
            lifetime=1.0,
            auth_data={},
            expiration=time.time() + 10,
            on_close_route="http://hook/cb",
            on_close_payload="done",
        )
        mock_sess = attach_serverless_backend_mock_session_post(backend, response_text="")
        await backend._Backend__run_session_on_close(session)
        body = mock_sess.post.call_args.kwargs["json"]
        assert body == {"payload": "done", "session_id": "cb2"}

    @pytest.mark.asyncio
    async def test_run_session_on_close_no_op_without_route(
        self,
        serverless_backend_and_handler_default,
        make_pyworker_session,
        attach_serverless_backend_mock_session_post,
    ) -> None:
        """
        Verifies __run_session_on_close returns immediately when on_close_route is falsy.

        This test verifies by:
        1. Attaching a mock ClientSession with post
        2. Awaiting __run_session_on_close for a session with no callback URL
        3. Asserting post was never called

        Assumptions:
        - Early return happens before any HTTP
        """
        backend, _ = serverless_backend_and_handler_default
        mock_sess = attach_serverless_backend_mock_session_post(backend, spy_only=True)
        session = make_pyworker_session(
            session_id="no-cb",
            lifetime=1.0,
            auth_data={},
            expiration=time.time() + 10,
            on_close_route=None,
            on_close_payload={"ignored": True},
        )
        await backend._Backend__run_session_on_close(session)
        mock_sess.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_session_on_close_none_payload_sends_session_id_only(
        self,
        serverless_backend_and_handler_default,
        make_pyworker_session,
        attach_serverless_backend_mock_session_post,
    ) -> None:
        """
        Verifies __run_session_on_close uses an empty dict when on_close_payload is None.

        This test verifies by:
        1. Session with route set and on_close_payload None
        2. Asserting POST json is only session_id (setdefault)

        Assumptions:
        - None branch uses body = {} then setdefault session_id
        """
        backend, _ = serverless_backend_and_handler_default
        session = make_pyworker_session(
            session_id="cb-null-payload",
            lifetime=1.0,
            auth_data={},
            expiration=time.time() + 10,
            on_close_route="http://notify/",
            on_close_payload=None,
        )
        mock_sess = attach_serverless_backend_mock_session_post(backend)
        await backend._Backend__run_session_on_close(session)
        assert mock_sess.post.call_args.kwargs["json"] == {"session_id": "cb-null-payload"}

    @pytest.mark.asyncio
    async def test_run_session_on_close_completes_on_http_error_status(
        self,
        serverless_backend_and_handler_default,
        make_pyworker_session,
        attach_serverless_backend_mock_session_post,
    ) -> None:
        """
        Verifies __run_session_on_close does not raise when the callback returns HTTP >= 400.

        This test verifies by:
        1. Mocking POST context with response status 503
        2. Awaiting __run_session_on_close successfully

        Assumptions:
        - Failures are recorded at DEBUG via ``log.debug`` (invisible at INFO); caller
          sees no exception
        """
        backend, _ = serverless_backend_and_handler_default
        mock_sess = attach_serverless_backend_mock_session_post(
            backend, response_status=503, response_text="unavailable"
        )
        session = make_pyworker_session(
            session_id="cb-http-err",
            lifetime=1.0,
            auth_data={},
            expiration=time.time() + 10,
            on_close_route="http://hook/err",
            on_close_payload={},
        )
        await backend._Backend__run_session_on_close(session)
        mock_sess.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_session_on_close_swallows_post_exception(
        self,
        serverless_backend_and_handler_default,
        make_pyworker_session,
        attach_serverless_backend_mock_session_post,
    ) -> None:
        """
        Verifies __run_session_on_close catches exceptions from session.post.

        This test verifies by:
        1. Making post() raise ConnectionError
        2. Asserting the await completes without propagating

        Assumptions:
        - Broad ``except`` logs at DEBUG and returns (not visible at INFO by default)
        """
        backend, _ = serverless_backend_and_handler_default
        mock_sess = attach_serverless_backend_mock_session_post(
            backend, post_side_effect=ConnectionError("refused")
        )
        session = make_pyworker_session(
            session_id="cb-exc",
            lifetime=1.0,
            auth_data={},
            expiration=time.time() + 10,
            on_close_route="http://hook/x",
            on_close_payload={},
        )
        await backend._Backend__run_session_on_close(session)
        mock_sess.post.assert_called_once()


# ---------------------------------------------------------------------------
# Public key fetch, start_tracking, healthcheck loop
# ---------------------------------------------------------------------------


class TestBackendFetchPubkey:
    """Tests for Backend._fetch_pubkey (retry + RSA import)."""

    @pytest.mark.asyncio
    async def test_fetch_pubkey_success_sets_key_and_event(
        self,
        serverless_backend_and_handler_default,
        make_serverless_test_rsa_key,
        vast_serverless_backend_module,
        make_serverless_pubkey_fetch_client_session_cm,
    ) -> None:
        """
        Verifies a successful HTTP fetch imports the PEM and sets __pubkey_fetch_complete.

        This test verifies by:
        1. Patching ``ClientSession`` in ``backend`` module to return a mock GET with PEM body
        2. Awaiting ``_fetch_pubkey`` and asserting ``_pubkey`` is non-None and the event is set

        Assumptions:
        - First GET attempt succeeds; no backoff sleep path is exercised
        """
        backend, _ = serverless_backend_and_handler_default
        backend._Backend__pubkey_fetch_complete.clear()
        pem = make_serverless_test_rsa_key().export_key(format="PEM").decode()
        sess_cm = make_serverless_pubkey_fetch_client_session_cm(response_text=pem)

        with patch.object(vast_serverless_backend_module, "ClientSession", return_value=sess_cm):
            await backend._fetch_pubkey()

        assert backend._pubkey is not None
        assert backend._Backend__pubkey_fetch_complete.is_set()
        assert backend._Backend__pubkey_failed is False

    @pytest.mark.asyncio
    async def test_fetch_pubkey_exhausts_retries_sets_failed_and_signals_complete(
        self,
        serverless_backend_and_handler_default,
        vast_serverless_backend_module,
        make_serverless_pubkey_fetch_client_session_cm,
    ) -> None:
        """
        Verifies repeated failures mark pubkey fetch as failed and call backend_errored.

        This test verifies by:
        1. Making every ``client.get`` raise ``ClientConnectorError``
        2. Patching ``asyncio.sleep`` in the backend module to avoid real delays
        3. Patching ``backend_errored`` and asserting failure state after all attempts

        Assumptions:
        - ``MAX_PUBKEY_FETCH_ATTEMPTS`` attempts run; final state matches the error path
        """
        backend, _ = serverless_backend_and_handler_default
        backend._Backend__pubkey_fetch_complete.clear()
        backend._Backend__pubkey_failed = False

        sess_cm = make_serverless_pubkey_fetch_client_session_cm(
            get_aenter_error=vast_serverless_backend_module.ClientConnectorError(
                MagicMock(), MagicMock()
            ),
        )

        with patch.object(vast_serverless_backend_module, "ClientSession", return_value=sess_cm):
            with patch.object(vast_serverless_backend_module.asyncio, "sleep", new_callable=AsyncMock):
                with patch.object(backend, "backend_errored") as mock_errored:
                    await backend._fetch_pubkey()

        assert backend._Backend__pubkey_failed is True
        assert backend._Backend__pubkey_fetch_complete.is_set()
        mock_errored.assert_called_once()
        assert "public key" in mock_errored.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_fetch_pubkey_succeeds_on_second_attempt_after_invalid_pem(
        self,
        serverless_backend_and_handler_default,
        make_serverless_test_rsa_key,
        vast_serverless_backend_module,
        make_serverless_pubkey_fetch_client_session_cm,
    ) -> None:
        """
        Verifies a failed RSA import on the first attempt retries after backoff sleep.

        This test verifies by:
        1. Returning invalid PEM text on the first HTTP response and valid PEM on the second
        2. Patching ``asyncio.sleep`` in the backend module so backoff is instant
        3. Asserting ``_pubkey`` is set and sleep was awaited between attempts

        Assumptions:
        - ``RSA.import_key`` raises ``ValueError`` for the first body, which hits the retry path
        """
        backend, _ = serverless_backend_and_handler_default
        backend._Backend__pubkey_fetch_complete.clear()
        good_pem = make_serverless_test_rsa_key().export_key(format="PEM").decode()
        bodies = iter(["-----BEGIN NOT REAL-----\nxx\n-----END NOT REAL-----\n", good_pem])
        build = make_serverless_pubkey_fetch_client_session_cm

        with patch.object(
            vast_serverless_backend_module,
            "ClientSession",
            side_effect=lambda *a, **k: build(response_text=next(bodies)),
        ):
            with patch.object(vast_serverless_backend_module.asyncio, "sleep", new_callable=AsyncMock) as mock_sleep:
                await backend._fetch_pubkey()

        mock_sleep.assert_awaited()
        assert backend._pubkey is not None
        assert backend._Backend__pubkey_fetch_complete.is_set()
        assert backend._Backend__pubkey_failed is False


class TestBackendStartTracking:
    """Tests for Backend._start_tracking orchestration."""

    @pytest.mark.asyncio
    async def test_start_tracking_gathers_six_background_coroutines(
        self, serverless_backend_and_handler_default, vast_serverless_backend_module
    ) -> None:
        """
        Verifies _start_tracking awaits gather with six concurrent background tasks.

        This test verifies by:
        1. Replacing ``gather`` in ``vastai.serverless.server.lib.backend`` with a spy
           that records arity then delegates to ``asyncio.gather``
        2. Patching each spawned coroutine source so none block on I/O

        Assumptions:
        - Implementation continues to pass exactly six awaitables to ``gather``
        """
        backend, _ = serverless_backend_and_handler_default
        captured: list[int] = []

        async def spy_gather(*aws, **kwargs):
            captured.append(len(aws))
            return await asyncio.gather(*aws, **kwargs)

        with patch.object(vast_serverless_backend_module, "gather", side_effect=spy_gather):
            with patch.object(backend, "_fetch_pubkey", new_callable=AsyncMock):
                with patch.object(backend, "_Backend__read_logs", new_callable=AsyncMock):
                    with patch.object(backend.metrics, "_send_metrics_loop", new_callable=AsyncMock):
                        with patch.object(backend, "_Backend__healthcheck", new_callable=AsyncMock):
                            with patch.object(
                                backend.metrics, "_send_delete_requests_loop", new_callable=AsyncMock
                            ):
                                with patch.object(
                                    backend, "_Backend__session_gc_loop", new_callable=AsyncMock
                                ):
                                    await backend._start_tracking()

        assert captured == [6]


class TestBackendHealthcheck:
    """Tests for Backend.__healthcheck."""

    @pytest.mark.asyncio
    async def test_healthcheck_no_endpoint_returns_immediately(
        self,
        serverless_backend_and_handler_default,
        attach_serverless_backend_aiohttp_session_get_spy,
    ) -> None:
        """
        Verifies __healthcheck exits without looping when MODEL_HEALTH_ENDPOINT is unset.

        This test verifies by:
        1. Ensuring ``healthcheck_url`` is empty on the backend instance
        2. Awaiting ``__healthcheck`` and asserting it returns without touching ``session.get``

        Assumptions:
        - Early return happens before the infinite loop and before any sleep
        """
        backend, _ = serverless_backend_and_handler_default
        object.__setattr__(backend, "healthcheck_url", "")
        mock_sess = attach_serverless_backend_aiohttp_session_get_spy(backend)

        await backend._Backend__healthcheck()

        mock_sess.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_healthcheck_skips_http_until_start_healthcheck_enabled(
        self,
        serverless_backend_and_handler_default,
        vast_serverless_backend_module,
        attach_serverless_backend_aiohttp_session_get_spy,
    ) -> None:
        """
        Verifies the loop continues without GET while ``__start_healthcheck`` is false.

        This test verifies by:
        1. Setting a non-empty ``healthcheck_url`` but leaving ``__start_healthcheck`` false
        2. Patching ``sleep`` to run several iterations then raise ``CancelledError``
        3. Asserting ``session.get`` was never called

        Assumptions:
        - Production sets ``__start_healthcheck`` after model load; until then health probes are skipped
        """
        backend, _ = serverless_backend_and_handler_default
        backend._Backend__start_healthcheck = False
        object.__setattr__(backend, "healthcheck_url", "http://model/pending")

        mock_sess = attach_serverless_backend_aiohttp_session_get_spy(backend)

        phase = {"n": 0}

        async def sleep_then_cancel(_delay: float) -> None:
            phase["n"] += 1
            if phase["n"] >= 3:
                raise asyncio.CancelledError()

        with patch.object(vast_serverless_backend_module, "sleep", side_effect=sleep_then_cancel):
            await backend._Backend__healthcheck()

        mock_sess.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_healthcheck_first_success_sets_ready_event(
        self,
        serverless_backend_and_handler_default,
        vast_serverless_backend_module,
        attach_serverless_backend_aiohttp_session_get_spy,
        make_serverless_aiohttp_get_context_manager,
    ) -> None:
        """
        Verifies the first HTTP 200 healthcheck sets __healthcheck_ready when startup flag is on.

        This test verifies by:
        1. Setting ``healthcheck_url``, ``__start_healthcheck``, and a mock ``session.get`` returning 200
        2. Patching ``sleep`` so the first iteration runs then CancelledError ends the loop
        3. Asserting ``__healthcheck_ready`` is set and ``get`` was used

        Assumptions:
        - CancelledError from ``sleep`` on the second iteration exits the loop cleanly
        """
        backend, _ = serverless_backend_and_handler_default
        backend._Backend__healthcheck_ready = asyncio.Event()
        backend._Backend__healthcheck_succeeded = False
        backend._Backend__start_healthcheck = True
        object.__setattr__(backend, "healthcheck_url", "http://model/ready")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_sess = attach_serverless_backend_aiohttp_session_get_spy(backend)
        mock_sess.get = MagicMock(return_value=make_serverless_aiohttp_get_context_manager(mock_resp))

        sleep_phase = {"n": 0}

        async def sleep_then_stop(_delay: float) -> None:
            sleep_phase["n"] += 1
            if sleep_phase["n"] > 1:
                raise asyncio.CancelledError()

        with patch.object(vast_serverless_backend_module, "sleep", side_effect=sleep_then_stop):
            await backend._Backend__healthcheck()

        assert backend._Backend__healthcheck_ready.is_set()
        assert backend._Backend__healthcheck_succeeded is True
        mock_sess.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_healthcheck_non_200_after_success_calls_backend_errored(
        self,
        serverless_backend_and_handler_default,
        vast_serverless_backend_module,
        attach_serverless_backend_aiohttp_session_get_spy,
        make_serverless_aiohttp_get_context_manager,
    ) -> None:
        """
        Verifies a failing status after a prior success reports via backend_errored.

        This test verifies by:
        1. Simulating one 200 response then a 503 on the next iteration
        2. Patching sleep to advance iterations then cancel
        3. Asserting ``backend_errored`` was called with a status message

        Assumptions:
        - ``__healthcheck_succeeded`` is only True after the first 200; second GET is 503
        """
        backend, _ = serverless_backend_and_handler_default
        backend._Backend__healthcheck_ready = asyncio.Event()
        backend._Backend__healthcheck_succeeded = False
        backend._Backend__start_healthcheck = True
        object.__setattr__(backend, "healthcheck_url", "http://model/hc")

        wrap = make_serverless_aiohttp_get_context_manager
        responses = [
            MagicMock(status=200),
            MagicMock(status=503),
        ]

        def next_get_cm():
            return wrap(responses.pop(0))

        mock_sess = attach_serverless_backend_aiohttp_session_get_spy(backend)
        mock_sess.get = MagicMock(side_effect=lambda *a, **k: next_get_cm())

        sleep_phase = {"n": 0}

        async def sleep_then_stop(_delay: float) -> None:
            sleep_phase["n"] += 1
            if sleep_phase["n"] > 2:
                raise asyncio.CancelledError()

        with patch.object(vast_serverless_backend_module, "sleep", side_effect=sleep_then_stop):
            with patch.object(backend, "backend_errored") as mock_err:
                await backend._Backend__healthcheck()

        mock_err.assert_called_once()
        assert "503" in mock_err.call_args[0][0]

    @pytest.mark.asyncio
    async def test_healthcheck_exception_after_success_calls_backend_errored(
        self,
        serverless_backend_and_handler_default,
        vast_serverless_backend_module,
        attach_serverless_backend_aiohttp_session_get_spy,
        make_serverless_aiohttp_get_context_manager,
    ) -> None:
        """
        Verifies connection errors after a successful check invoke backend_errored.

        This test verifies by:
        1. First GET succeeds (200), second ``session.get`` raises RuntimeError
        2. Patching sleep between iterations
        3. Asserting backend_errored receives the exception string

        Assumptions:
        - Exception path uses the same ``if self.__healthcheck_succeeded`` guard as HTTP errors
        """
        backend, _ = serverless_backend_and_handler_default
        backend._Backend__healthcheck_ready = asyncio.Event()
        backend._Backend__healthcheck_succeeded = False
        backend._Backend__start_healthcheck = True
        object.__setattr__(backend, "healthcheck_url", "http://model/hc")

        ok_resp = MagicMock(status=200)
        ok_cm = make_serverless_aiohttp_get_context_manager(ok_resp)

        call_count = {"n": 0}

        def get_side_effect(*a, **k):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ok_cm
            raise RuntimeError("connection reset")

        mock_sess = attach_serverless_backend_aiohttp_session_get_spy(backend)
        mock_sess.get = MagicMock(side_effect=get_side_effect)

        sleep_phase = {"n": 0}

        async def sleep_then_stop(_delay: float) -> None:
            sleep_phase["n"] += 1
            if sleep_phase["n"] > 2:
                raise asyncio.CancelledError()

        with patch.object(vast_serverless_backend_module, "sleep", side_effect=sleep_then_stop):
            with patch.object(backend, "backend_errored") as mock_err:
                await backend._Backend__healthcheck()

        mock_err.assert_called_once()
        assert "connection reset" in mock_err.call_args[0][0]
