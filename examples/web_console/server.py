"""Local web console for Vast.ai serverless ComfyUI endpoints.

A small FastAPI app that wraps the Vast SDK and serves a single-page UI.
Lets you paste an API key + endpoint id, submit a ComfyUI workflow, and
watch the worker pool's state in real time. Intended for local
development — no auth, no TLS — bind to 127.0.0.1 only.

Run:
    pip install -r requirements.txt
    python server.py
    # open http://127.0.0.1:8000

The browser sends the API key with every request rather than the server
holding it in memory, so multiple users on the same host don't share
credentials.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vastai import Serverless
from vastai.serverless.client.endpoint import Endpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("web_console")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Vast Serverless Web Console")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ---------- helpers ---------------------------------------------------------


async def _resolve_endpoint(client: Serverless, endpoint_id: int) -> Endpoint:
    """Look up an Endpoint object by id.

    The SDK exposes `get_endpoint(name=...)` but not by id, so we list
    and filter. Cheap — endpoint counts are typically small per account.
    """
    endpoints = await client.get_endpoints()
    for ep in endpoints:
        if int(ep.id) == int(endpoint_id):
            return ep
    raise HTTPException(
        status_code=404,
        detail=f"endpoint id={endpoint_id} not found in this account",
    )


def _api_key(body: dict) -> str:
    api_key = (body or {}).get("api_key")
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key required")
    return api_key


# ---------- routes ----------------------------------------------------------


@app.post("/api/endpoints")
async def list_endpoints(req: Request) -> JSONResponse:
    """Return the caller's endpoints so the UI can offer a dropdown."""
    body = await req.json()
    async with Serverless(api_key=_api_key(body)) as client:
        endpoints = await client.get_endpoints()
        return JSONResponse(
            [{"id": ep.id, "name": ep.name} for ep in endpoints]
        )


@app.post("/api/workers")
async def get_workers(req: Request) -> JSONResponse:
    """Snapshot of the workers behind an endpoint. UI polls this."""
    body = await req.json()
    async with Serverless(api_key=_api_key(body)) as client:
        endpoint = await _resolve_endpoint(client, body["endpoint_id"])
        workers = await endpoint.get_workers()
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
            "cost":        100                   # optional
        }
    """
    body = await req.json()
    api_key      = _api_key(body)
    endpoint_id  = body["endpoint_id"]
    route        = body.get("route", "/generate/sync")
    payload      = body.get("payload") or {}
    cost         = int(body.get("cost", 100))

    async with Serverless(api_key=api_key) as client:
        endpoint = await _resolve_endpoint(client, endpoint_id)
        log.info("submit -> endpoint=%s route=%s", endpoint.name, route)
        try:
            response = await endpoint.request(route, payload, cost=cost)
        except Exception as exc:
            log.exception("submit failed")
            raise HTTPException(status_code=502, detail=f"endpoint request failed: {exc}")
        return JSONResponse({"response": response})


# ---------- main ------------------------------------------------------------


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    log.info("starting web console at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
