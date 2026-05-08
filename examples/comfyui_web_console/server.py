"""ComfyUI Web Console — local UI for Vast.ai serverless ComfyUI endpoints.

A small FastAPI app that wraps the Vast SDK and serves a single-page UI.
Lets you paste an API key + endpoint id, submit a ComfyUI workflow,
watch the worker pool's state in real time, and replay past requests
with modifications. Intended for local development — no auth, no TLS
— bind to 127.0.0.1 only.

Run:
    pip install -r requirements.txt
    python server.py
    # open http://127.0.0.1:8000

The browser sends the API key with every request rather than the server
holding it in memory, so multiple users on the same host don't share
credentials.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# When running from a vast-sdk checkout, prefer the local SDK source
# over whatever's on the PYPI index. Lets the example pick up
# in-progress SDK changes (e.g. tracker.worker_url) without the
# operator having to remember to `pip install -e` first. Skipped
# when the example is shipped standalone (no sibling vastai/ dir).
_SDK_ROOT = Path(__file__).parent.parent.parent
if (_SDK_ROOT / "vastai" / "__init__.py").is_file():
    sys.path.insert(0, str(_SDK_ROOT))

from vastai import Serverless
from vastai.serverless.client.endpoint import Endpoint

# Bound on a single /api/submit, in seconds. Without this the SDK
# retries 5xx (e.g. a worker that errors on every request because the
# payload was sent to the wrong queue) until eternity, and the
# browser fetch never resolves — entries stay "in-flight" forever.
# Configurable so long-running workflows aren't capped.
SUBMIT_TIMEOUT_S = float(os.getenv("SUBMIT_TIMEOUT_S", "300"))

# Per-process registry of in-flight ServerlessRequest objects keyed
# by a browser-supplied tracking_id. The submit endpoint stashes the
# request here for the duration of the call; /api/status reads
# tracker.worker_url out of it so the UI can show "via <host>" the
# moment the autoscaler routes the job, rather than waiting for the
# response. Cleared on completion to keep memory bounded.
_inflight: Dict[str, "object"] = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("comfyui_web_console")

# Quieter SDK chatter on the steady-state poll path. Connection
# lifecycle and endpoint listings still log on first use.
logging.getLogger("vastai").setLevel(logging.WARNING)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="ComfyUI Web Console")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ---------- shared SDK clients ---------------------------------------------
#
# A naive `async with Serverless(api_key=...)` per request meant that
# every 3s worker poll opened a fresh aiohttp session, fetched the Vast
# SSL cert, listed *all* endpoints just to find one by id, and closed
# the session — visible as a flood of repeated lifecycle log lines and
# a real per-request latency tax. Cache one live client per API key
# and one Endpoint per (key, id) for the lifetime of the process. Two
# operators on the same host with different keys still get isolated
# sessions.

_clients:   Dict[str, Serverless]    = {}
_endpoints: Dict[Tuple[str, int], Endpoint] = {}


async def _client(api_key: str) -> Serverless:
    cli = _clients.get(api_key)
    if cli is not None and cli.is_open():
        return cli
    cli = Serverless(api_key=api_key)
    await cli.__aenter__()
    _clients[api_key] = cli
    # Endpoints captured a reference to the prior, now-dead client —
    # drop them so the next lookup rebuilds against the fresh session.
    for key in [k for k in _endpoints if k[0] == api_key]:
        _endpoints.pop(key, None)
    return cli


async def _endpoint(api_key: str, endpoint_id: int) -> Endpoint:
    key = (api_key, int(endpoint_id))
    cached = _endpoints.get(key)
    if cached is not None and cached.client is _clients.get(api_key):
        return cached
    cli = await _client(api_key)
    for ep in await cli.get_endpoints():
        _endpoints[(api_key, int(ep.id))] = ep
    if key not in _endpoints:
        raise HTTPException(
            status_code=404,
            detail=f"endpoint id={endpoint_id} not found in this account",
        )
    return _endpoints[key]


@app.on_event("shutdown")
async def _close_clients() -> None:
    for cli in _clients.values():
        try:
            await cli.__aexit__(None, None, None)
        except Exception:
            pass
    _clients.clear()
    _endpoints.clear()


def _api_key(body: dict) -> str:
    api_key = (body or {}).get("api_key")
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key required")
    return api_key


# ---------- routes ----------------------------------------------------------


@app.post("/api/endpoints")
async def list_endpoints(req: Request) -> JSONResponse:
    """Return the caller's endpoints so the UI can offer a dropdown.

    Always re-fetches and refreshes the cache, so clicking
    "Load endpoints" picks up new ones added to the account.
    """
    body = await req.json()
    api_key = _api_key(body)
    cli = await _client(api_key)
    endpoints = await cli.get_endpoints()
    for ep in endpoints:
        _endpoints[(api_key, int(ep.id))] = ep
    return JSONResponse([{"id": ep.id, "name": ep.name} for ep in endpoints])


def _parse_429_retry_after(msg: str, default: int = 5) -> int:
    """Pull retry_after seconds out of the SDK's 429 RuntimeError text.

    The SDK serialises upstream 429s as
        ``RuntimeError: ... HTTP 429 - {"retry_after": 1, ...}``
    so we can recover the suggested wait without re-issuing the call.
    """
    try:
        payload = json.loads(msg.split(" - ", 1)[1])
        return max(1, int(payload.get("retry_after", default)))
    except Exception:
        return default


@app.post("/api/workers")
async def get_workers(req: Request) -> JSONResponse:
    """Snapshot of the workers behind an endpoint. UI polls this."""
    body = await req.json()
    endpoint = await _endpoint(_api_key(body), body["endpoint_id"])
    try:
        workers = await endpoint.get_workers()
    except RuntimeError as ex:
        # The autoscaler rate-limits get_endpoint_workers at ~1 req/s.
        # Surface 429s to the caller as 429 with a Retry-After header
        # so the browser's poll loop can back off cleanly instead of
        # treating it as a generic error.
        msg = str(ex)
        if "HTTP 429" in msg:
            retry = _parse_429_retry_after(msg)
            return JSONResponse(
                status_code=429,
                content={"retry_after": retry, "message": "rate limited by autoscaler"},
                headers={"Retry-After": str(retry)},
            )
        raise
    return JSONResponse(
        [
            {
                "id": w.id,
                "status": w.status,
                "cur_load": w.cur_load,
                "new_load": w.new_load,
                "cur_perf": w.cur_perf,
                "perf": w.perf,
                "measured_perf": w.measured_perf,
                "reqs_working": w.reqs_working,
                "reliability": w.reliability,
                "loaded_at": w.loaded_at,
                "started_at": w.started_at,
            }
            for w in workers
        ]
    )


@app.post("/api/submit")
async def submit_request(req: Request) -> JSONResponse:
    """Forward a worker payload to the chosen endpoint, return the result.

    Body shape:
        {
            "api_key":     "...",
            "endpoint_id": 12345,
            "route":       "/generate/sync",     # optional, defaults shown
            "payload":     { ... worker JSON ... },
            "cost":        100,                  # optional
            "tracking_id": "abc123"              # optional, browser-supplied
                                                  # — exposes the in-flight tracker
                                                  # via /api/status?id=...
        }
    """
    body         = await req.json()
    api_key      = _api_key(body)
    endpoint_id  = body["endpoint_id"]
    route        = body.get("route", "/generate/sync")
    payload      = body.get("payload") or {}
    cost         = int(body.get("cost", 100))
    tracking_id  = body.get("tracking_id")

    endpoint = await _endpoint(api_key, endpoint_id)
    log.info("submit -> endpoint=%s route=%s", endpoint.name, route)

    # Hand the SDK a ServerlessRequest we control so we can read
    # tracker.worker_url out of it while we're awaiting the result.
    # `endpoint.request` accepts a `serverless_request=` parameter
    # for exactly this kind of observation.
    from vastai.serverless.client.client import ServerlessRequest
    sreq = ServerlessRequest()
    if tracking_id:
        _inflight[tracking_id] = sreq

    try:
        # Pass timeout through to the SDK so a worker that
        # consistently errors (5xx => retryable) can't park us in an
        # unbounded retry loop. asyncio.TimeoutError surfaces as 504
        # so the browser entry transitions to failed cleanly.
        response = await endpoint.request(
            route, payload, cost=cost, timeout=SUBMIT_TIMEOUT_S,
            serverless_request=sreq,
        )
    except asyncio.TimeoutError:
        log.warning("submit timed out after %.0fs", SUBMIT_TIMEOUT_S)
        raise HTTPException(
            status_code=504,
            detail=f"endpoint request timed out after {SUBMIT_TIMEOUT_S:.0f}s",
        )
    except Exception as exc:
        log.exception("submit failed")
        raise HTTPException(status_code=502, detail=f"endpoint request failed: {exc}")
    finally:
        if tracking_id:
            _inflight.pop(tracking_id, None)
    return JSONResponse({"response": response})


@app.post("/api/status")
async def request_status_batch(req: Request) -> JSONResponse:
    """Batched snapshot of in-flight submissions' trackers.

    Body shape:  {"ids": ["c1...", "c2..."]}

    The browser polls this while *any* request is in-flight to
    surface the worker URL the moment the autoscaler routes the job
    — `tracker.worker_url` is written by `_do_request` before the
    SDK posts to the worker, so the UI can replace its
    "waiting for worker" placeholder with "via <host>" without
    waiting for the response itself.

    Single batched call rather than one-per-entry so a 50-burst
    doesn't generate 50 polls/sec; entries that have completed
    between the browser preparing the list and the server seeing
    the request are simply absent from the response (terminal
    worker_url comes back inside the submit response itself).
    """
    body = await req.json()
    ids: List[str] = list(body.get("ids", []))
    out: Dict[str, dict] = {}
    for tid in ids:
        sreq = _inflight.get(tid)
        if sreq is None:
            continue
        out[tid] = {
            "status":     getattr(sreq, "status", None),
            # `worker_url` is None on SDK builds older than the
            # tracker.worker_url change; the UI tolerates that.
            "worker_url": getattr(sreq, "worker_url", None),
        }
    return JSONResponse(out)


# ---------- main ------------------------------------------------------------


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    log.info("ComfyUI Web Console listening at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
