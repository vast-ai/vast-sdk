import aiohttp
import random
import asyncio

async def _make_request(client, route : str, api_key: str, url:str = "", body={}, params={}, method="GET", retries=5, timeout=30):

    auth_header = f"Bearer {api_key}"
    headers = {"Authorization" : auth_header}

    params["api_key"] = api_key

    session = await client._get_session()

    request_fn = session.get if method == "GET" else session.post

    ssl_context = await client.get_ssl_context() if client else None

    for attempt in range(1, retries + 1):
        try:
            async with request_fn(
                url + route,
                headers=headers,
                json=body,
                params=params,
                ssl=ssl_context,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                text = await resp.text()

                if resp.status == 200:
                    try:
                        return await resp.json(content_type=None)
                    except Exception:
                        raise Exception(f"Invalid JSON from {url + route}:\n{text}")

                retry_after = resp.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after else min(2 ** attempt + random.uniform(0, 1), 30)
                await asyncio.sleep(wait_time)
        except Exception as ex:
            client.logger.error(f"Attempt {attempt} failed: {ex}")
            await asyncio.sleep(min(2 ** attempt + random.uniform(0, 1), 30))

    raise Exception(f"Too many retries for {url + route}")
