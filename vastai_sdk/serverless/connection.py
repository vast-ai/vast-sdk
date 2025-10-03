import aiohttp
import random
import asyncio

VAST_WEB_URL = "https://console.vast.ai"
VAST_SERVERLESS_URL = "https://run.vast.ai"

async def _make_request(client, route : str, api_key : str, url=VAST_WEB_URL, body={}, params={}, method="GET", retries=5):

    auth_header = f"Bearer {api_key}"
    headers = {"Authorization" : auth_header}

    params["api_key"] = api_key

    session = await client._get_session()

    request_fn = session.get if method == "GET" else session.post

    ssl_context = await client.get_ssl_context() if client else None

    for attempt in range(retries):
            async with request_fn(url + route, headers=headers, json=body, params=params, ssl=ssl_context) as resp:
                text = await resp.text()

                if resp.status != 200:
                    # Check for Retry-After header if provided
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        wait_time = float(retry_after)
                    else:
                        # exponential backoff + jitter
                        wait_time = min(2 ** attempt + random.uniform(0, 1), 30)

                    await asyncio.sleep(wait_time)
                    continue  # retry

                try:
                    return await resp.json(content_type=None)
                except Exception:
                    raise Exception(f"Invalid JSON from {url + route}:\n{text}")

    raise Exception(f"Too many retries for {url + route}")
