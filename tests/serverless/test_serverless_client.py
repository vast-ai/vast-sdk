"""Unit tests for vastai.serverless.client.client (Serverless, ServerlessRequest).

HTTP and routing are mocked via _make_request, queue_endpoint_request, or aiohttp session mocks.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vastai.serverless.client.client import Serverless, ServerlessRequest
from vastai.serverless.client.endpoint import Endpoint
from vastai.serverless.client.session import Session as ClientSession


class TestServerlessRequest:
    """ServerlessRequest future wrapper behavior."""

    @pytest.mark.asyncio
    async def test_then_invokes_callback_on_success(self) -> None:
        """
        Verifies ServerlessRequest.then registers a done callback that receives the result.

        This test verifies by:
        1. Creating a ServerlessRequest and chaining .then with a MagicMock callback
        2. Calling set_result with a payload
        3. Yielding to the loop so the callback runs, then asserting call args

        Assumptions:
        - asyncio schedules done callbacks on the next loop iteration
        """
        cb = MagicMock()
        req = ServerlessRequest()
        req.then(cb)
        req.set_result({"ok": True})
        await asyncio.sleep(0)
        cb.assert_called_once_with({"ok": True})


class TestServerlessInitAndConfig:
    """Constructor, API key, and instance URL selection."""

    def test_init_raises_when_api_key_missing(self) -> None:
        """
        Verifies Serverless rejects a missing or empty api_key.

        This test verifies by:
        1. Instantiating Serverless(api_key=None) and Serverless(api_key="")
        2. Asserting AttributeError mentioning API key each time

        Assumptions:
        - __init__ treats None and empty string as missing
        """
        with pytest.raises(AttributeError, match="API key missing"):
            Serverless(api_key=None)
        with pytest.raises(AttributeError, match="API key missing"):
            Serverless(api_key="")

    def test_init_accepts_explicit_api_key(self, serverless_client, api_key: str) -> None:
        """
        Verifies explicit api_key is stored on the client.

        This test verifies by:
        1. Constructing Serverless(api_key=...) via serverless_client fixture
        2. Asserting client.api_key matches

        Assumptions:
        - No environment key is required when api_key is passed
        """
        assert serverless_client.api_key == api_key

    @pytest.mark.parametrize(
        ("instance", "autoscaler_substr"),
        [
            ("prod", "run.vast.ai"),
            ("alpha", "run-alpha.vast.ai"),
            ("candidate", "run-candidate.vast.ai"),
            ("local", "localhost:8080"),
        ],
    )
    def test_instance_selects_autoscaler_url(self, api_key: str, instance: str, autoscaler_substr: str) -> None:
        """
        Verifies instance keyword maps to expected autoscaler base URL.

        This test verifies by:
        1. Creating Serverless with instance=...
        2. Asserting autoscaler_url contains the expected host fragment

        Assumptions:
        - Mapping matches current client.py match/case branches
        """
        client = Serverless(api_key=api_key, instance=instance)
        assert autoscaler_substr in client.autoscaler_url


@pytest.mark.asyncio
class TestServerlessEndpoints:
    """get_endpoints and get_endpoint."""

    async def test_get_endpoints_parses_results_into_endpoints(self, serverless_client) -> None:
        """
        Verifies get_endpoints builds Endpoint objects from JSON results.

        This test verifies by:
        1. Patching _make_request to return ok JSON with one result row
        2. Calling await client.get_endpoints()
        3. Asserting one Endpoint with matching name, id, api_key

        Assumptions:
        - _make_request is patched at the module where client.py uses it
        """
        client = serverless_client
        fake = {
            "ok": True,
            "json": {
                "results": [
                    {"endpoint_name": "ep-a", "id": 7, "api_key": "wk"},
                ]
            },
        }
        with patch("vastai.serverless.client.client._make_request", new_callable=AsyncMock, return_value=fake):
            endpoints = await client.get_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0].name == "ep-a"
        assert endpoints[0].id == 7
        assert endpoints[0].api_key == "wk"
        assert endpoints[0].client is client

    async def test_get_endpoints_raises_when_http_not_ok(self, serverless_client) -> None:
        """
        Verifies get_endpoints wraps failed HTTP into Exception.

        This test verifies by:
        1. Returning ok=False from _make_request
        2. Asserting Exception is raised with status context

        Assumptions:
        - Client surfaces HTTP failures as generic Exception
        """
        client = serverless_client
        with patch(
            "vastai.serverless.client.client._make_request",
            new_callable=AsyncMock,
            return_value={"ok": False, "status": 500, "text": "err"},
        ):
            with pytest.raises(Exception, match="Failed to get endpoints"):
                await client.get_endpoints()

    async def test_get_endpoint_returns_matching_endpoint(self, serverless_client) -> None:
        """
        Verifies get_endpoint selects by name from get_endpoints.

        This test verifies by:
        1. Patching get_endpoints to return two Endpoint mocks
        2. Calling get_endpoint('two')
        3. Asserting the correct endpoint is returned

        Assumptions:
        - get_endpoint only compares e.name
        """
        client = serverless_client
        e1 = Endpoint(client, "one", 1, "k")
        e2 = Endpoint(client, "two", 2, "k")
        with patch.object(client, "get_endpoints", new_callable=AsyncMock, return_value=[e1, e2]):
            got = await client.get_endpoint("two")
        assert got is e2

    async def test_get_endpoint_raises_when_name_not_found(self, serverless_client) -> None:
        """
        Verifies get_endpoint raises when no endpoint matches the name.

        This test verifies by:
        1. Patching get_endpoints to return an empty list
        2. Asserting Exception mentioning the endpoint name

        Assumptions:
        - Empty list yields no match
        """
        client = serverless_client
        with patch.object(client, "get_endpoints", new_callable=AsyncMock, return_value=[]):
            with pytest.raises(Exception, match="could not be found"):
                await client.get_endpoint("nope")


@pytest.mark.asyncio
class TestServerlessWorkersAndSessions:
    """get_endpoint_workers, get_endpoint_session, end_endpoint_session, start_endpoint_session."""

    async def test_get_endpoint_workers_requires_endpoint_type(self, serverless_client) -> None:
        """
        Verifies get_endpoint_workers rejects non-Endpoint values.

        This test verifies by:
        1. Passing a MagicMock instead of Endpoint
        2. Asserting ValueError

        Assumptions:
        - isinstance check runs before HTTP
        """
        client = serverless_client
        with pytest.raises(ValueError, match="endpoint must be an Endpoint"):
            await client.get_endpoint_workers(MagicMock())

    async def test_get_endpoint_workers_returns_worker_list(
        self, serverless_client, make_mock_http_response, make_serverless_endpoint
    ) -> None:
        """
        Verifies get_endpoint_workers parses JSON list into Worker models.

        This test verifies by:
        1. Attaching a mock aiohttp session with post returning worker dicts
        2. Calling await client.get_endpoint_workers(endpoint)
        3. Asserting Worker.id and count

        Assumptions:
        - _session.post is used with JSON body containing endpoint id and api_key
        """
        client = serverless_client
        mock_sess = MagicMock()
        mock_sess.post = MagicMock(
            return_value=make_mock_http_response(
                status=200,
                json_data=[{"id": 99, "status": "RUNNING"}],
            )
        )
        client._session = mock_sess
        ep = make_serverless_endpoint(endpoint_id=3)
        workers = await client.get_endpoint_workers(ep)
        assert len(workers) == 1
        assert workers[0].id == 99

    async def test_get_endpoint_workers_error_msg_returns_empty_list(
        self, serverless_client, make_mock_http_response, make_serverless_endpoint
    ) -> None:
        """
        Verifies get_endpoint_workers returns [] when API returns error_msg dict.

        This test verifies by:
        1. Returning JSON dict with error_msg key from post()
        2. Asserting empty list result

        Assumptions:
        - Client treats error_msg as soft failure for not-ready endpoints
        """
        client = serverless_client
        mock_sess = MagicMock()
        mock_sess.post = MagicMock(
            return_value=make_mock_http_response(status=200, json_data={"error_msg": "not ready"})
        )
        client._session = mock_sess
        ep = make_serverless_endpoint(endpoint_id=3)
        workers = await client.get_endpoint_workers(ep)
        assert workers == []

    async def test_get_endpoint_session_builds_session(
        self, serverless_client, make_serverless_endpoint
    ) -> None:
        """
        Verifies get_endpoint_session calls _make_request and constructs Session.

        This test verifies by:
        1. Patching _make_request with ok JSON containing auth_data with url
        2. Awaiting get_endpoint_session
        3. Asserting Session fields

        Assumptions:
        - session_auth dict includes url used for worker request and Session.url
        """
        client = serverless_client
        ep = make_serverless_endpoint()
        auth = {"url": "https://worker/s", "token": "t"}
        fake = {
            "ok": True,
            "json": {
                "auth_data": auth,
                "lifetime": 120.0,
                "expiration": "2099-01-01",
            },
        }
        with patch("vastai.serverless.client.client._make_request", new_callable=AsyncMock, return_value=fake):
            sess = await client.get_endpoint_session(ep, 42, auth, timeout=5.0)
        assert sess.endpoint is ep
        assert sess.session_id == 42
        assert sess.auth_data == auth
        assert sess.url == "https://worker/s"

    async def test_end_endpoint_session_calls_make_request(
        self, serverless_client, make_serverless_endpoint
    ) -> None:
        """
        Verifies end_endpoint_session POSTs to session.url via _make_request.

        This test verifies by:
        1. Patching _make_request to return ok
        2. Building a minimal Session with url and auth_data
        3. Awaiting end_endpoint_session and asserting _make_request kwargs

        Assumptions:
        - Route is /session/end and body includes session_id and session_auth
        """
        client = serverless_client
        ep = make_serverless_endpoint()
        sess = ClientSession(
            endpoint=ep,
            session_id="sid",
            lifetime=60.0,
            expiration="e",
            url="https://worker/end",
            auth_data={"a": 1},
        )
        with patch("vastai.serverless.client.client._make_request", new_callable=AsyncMock) as mock_mr:
            mock_mr.return_value = {"ok": True, "json": {}}
            await client.end_endpoint_session(sess, timeout=8.0)
        mock_mr.assert_awaited_once()
        call_kw = mock_mr.await_args.kwargs
        assert call_kw["url"] == "https://worker/end"
        assert call_kw["route"] == "/session/end"
        assert call_kw["body"]["session_id"] == "sid"

    async def test_start_endpoint_session_uses_queue_result(
        self, serverless_client, make_serverless_endpoint
    ) -> None:
        """
        Verifies start_endpoint_session awaits queue_endpoint_request and returns Session.

        This test verifies by:
        1. Patching queue_endpoint_request to return a pre-resolved ServerlessRequest
        2. Awaiting start_endpoint_session
        3. Asserting Session session_id, url, and auth_data

        Assumptions:
        - queue_endpoint_request result dict matches successful worker create shape
        """
        client = serverless_client
        ep = make_serverless_endpoint()
        fut = ServerlessRequest()
        fut.set_result(
            {
                "ok": True,
                "response": {"session_id": "new-sid", "expiration": "ex"},
                "url": "https://w/u",
                "auth_data": {"k": "v"},
            }
        )
        with patch.object(client, "queue_endpoint_request", return_value=fut):
            sess = await client.start_endpoint_session(ep, cost=50, lifetime=30.0)
        assert sess.session_id == "new-sid"
        assert sess.expiration == "ex"
        assert sess.url == "https://w/u"
        assert sess.auth_data == {"k": "v"}
        assert sess.lifetime == 30.0


@pytest.mark.asyncio
class TestQueueEndpointRequest:
    """Background task for worker requests (session-bound path)."""

    async def test_queue_endpoint_request_with_session_sets_result_on_success(
        self, serverless_client, make_serverless_endpoint
    ) -> None:
        """
        Verifies queue_endpoint_request completes when session is set and worker returns ok JSON.

        This test verifies by:
        1. Patching _make_request to return ok with JSON body
        2. Awaiting the returned ServerlessRequest future
        3. Asserting response payload, url, and auth_data echo session fields

        Assumptions:
        - Session-bound path skips _route polling and posts directly to session.url
        """
        client = serverless_client
        ep = make_serverless_endpoint()
        sess = ClientSession(
            ep,
            "sid-99",
            60.0,
            "2099-01-01",
            "https://worker/direct",
            {"token": "t"},
        )
        with patch("vastai.serverless.client.client._make_request", new_callable=AsyncMock) as mock_mr:
            mock_mr.return_value = {"ok": True, "json": {"answer": 42}}
            fut = client.queue_endpoint_request(
                ep, "/do", {"p": 1}, session=sess, worker_timeout=30.0
            )
            result = await fut
        assert result["ok"] is True
        assert result["response"] == {"answer": 42}
        assert result["url"] == "https://worker/direct"
        assert result["auth_data"] == {"token": "t"}
        mock_mr.assert_awaited()
        call_kw = mock_mr.await_args.kwargs
        assert call_kw["url"] == "https://worker/direct"
        assert call_kw["route"] == "/do"


@pytest.mark.asyncio
class TestServerlessContextAndSessionState:
    """Async context manager, is_open, close."""

    async def test_is_open_false_without_session(self, serverless_client) -> None:
        """
        Verifies is_open is False before _get_session creates a session.

        This test verifies by:
        1. Constructing Serverless
        2. Calling is_open() synchronously

        Assumptions:
        - _session starts as None
        """
        assert serverless_client.is_open() is False

    async def test_close_noop_when_no_session(self, serverless_client) -> None:
        """
        Verifies close does not fail when no session exists.

        This test verifies by:
        1. Calling await close() on a new client
        2. Completing without error

        Assumptions:
        - close checks _session truthiness and closed flag
        """
        await serverless_client.close()

    async def test_aenter_calls_get_session(self, serverless_client) -> None:
        """
        Verifies __aenter__ awaits _get_session and returns self.

        This test verifies by:
        1. Patching _get_session with AsyncMock
        2. Using async with Serverless(...)
        3. Asserting _get_session awaited and entered object is the client

        Assumptions:
        - __aenter__ only opens session, does not require real aiohttp
        """
        client = serverless_client
        with patch.object(client, "_get_session", new_callable=AsyncMock) as mock_gs:
            async with client as entered:
                assert entered is client
            mock_gs.assert_awaited()
