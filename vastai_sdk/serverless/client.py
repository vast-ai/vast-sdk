from .connection import _make_request
from .endpoint import Endpoint
import asyncio
import aiohttp
import ssl
import os
import tempfile
import logging

class ServerlessClient:
    SSL_CERT_URL = "https://console.vast.ai/static/jvastai_root.cer"

    def __init__(self, api_key: str, debug=False):
        if api_key is None or api_key == "":
            raise ValueError("api_key cannot be empty")
        self.api_key = api_key

        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.debug:
            # Only set up logging if debug is True
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        else:
            # If debug is False, disable logging
            self.logger.addHandler(logging.NullHandler())

        self._session: aiohttp.ClientSession | None = None
        self._ssl_context: ssl.SSLContext | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self.logger.info("Started aiohttp ClientSession")
            self._session = aiohttp.ClientSession()
        return self._session
    
    def is_open(self):
        return self._session is not None and not self._session.closed

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info("Closed aiohttp ClientSession")

    async def get_ssl_context(self) -> ssl.SSLContext:
        """Download Vast.ai root cert and build SSL context (cached)."""
        if self._ssl_context is None:
            # Download the Vast root cert
            async with aiohttp.ClientSession() as s:
                async with s.get(self.SSL_CERT_URL) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to fetch SSL cert: {resp.status}")
                    cert_bytes = await resp.read()

            # Write to a temporary PEM file
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".cer")
            tmpfile.write(cert_bytes)
            tmpfile.close()

            # Start with system defaults
            ctx = ssl.create_default_context()
            ctx.load_verify_locations(cafile=tmpfile.name)
            self.logger.info("Loaded Vast.ai SSL certificate")

            self._ssl_context = ctx

            os.unlink(tmpfile.name)

        return self._ssl_context

    async def get_endpoint(self, name="") -> Endpoint:
        endpoints = await self.get_endpoints()
        for e in endpoints:
            if e.name == name:
                return e
        raise Exception(f"Endpoint {name} could not be found")

    async def get_endpoints(self) -> list[Endpoint]:
        try:
            response = await _make_request(
                client=self,
                route="/api/v0/endptjobs/",
                api_key=self.api_key,
                body={"client_id": "me"}
            )
        except Exception as ex:
            raise Exception(
                f"Failed to get endpoints:\nReason={ex}"
            )

        endpoints = []
        for e in response["results"]:
            endpoints.append(Endpoint(client=self, name=e["endpoint_name"], id=e["id"], api_key=e["api_key"]))
        self.logger.info(f"Found {len(endpoints)} endpoints")
        return endpoints
    