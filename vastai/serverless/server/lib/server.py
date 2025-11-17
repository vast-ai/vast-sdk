import os
import logging
from typing import List
import ssl
from asyncio import run, gather
from .backend import Backend
from aiohttp import web

log = logging.getLogger(__file__)

async def start_server_async(backend: Backend, routes: List[web.RouteDef], **kwargs):
    log.debug("getting certificate...")
    use_ssl = os.environ.get("USE_SSL", "false") == "true"
    if use_ssl:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(
            certfile="/etc/instance.crt",
            keyfile="/etc/instance.key",
        )
    else:
        ssl_context = None

    log.debug("starting server...")
    app = web.Application()
    app.add_routes(routes)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(
        runner,
        ssl_context=ssl_context,
        port=int(os.environ["WORKER_PORT"]),
        **kwargs,
    )
    # This never returns: HTTP server + metrics/log/health tracking
    await gather(site.start(), backend._start_tracking())


def start_server(backend: Backend, routes: List[web.RouteDef], **kwargs):
    run(start_server_async(backend, routes, **kwargs))

