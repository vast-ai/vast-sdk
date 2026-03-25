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
