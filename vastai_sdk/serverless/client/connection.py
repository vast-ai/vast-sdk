import aiohttp
import random
import asyncio
import uuid

async def _make_request(
    client,
    route: str,
    api_key: str,
    url: str = "",
    body: dict = {},
    params: dict = {},
    method: str = "GET",
    retries: int = 5,
    timeout: float = 30,
):
    """
    Make an HTTP request with exponential backoff.
    - For GET, we do NOT send a JSON body.
    - Retries: only for 408/429/5xx; 4xx (except 408/429) fail fast.
    """
    auth_header = f"Bearer {api_key}"
    headers = {
        "Authorization": auth_header,
        "X-Request-ID": str(uuid.uuid4()),
    }

    # Optional: avoid putting API key in query at all (prefer header)
    params = dict(params) if params else {}
    if getattr(client, "send_api_key_in_query", True):
        params["api_key"] = api_key

    session = await client._get_session()
    ssl_context = await client.get_ssl_context() if client else None

    request_fn = session.get if method.upper() == "GET" else session.post

    last_text = ""
    last_status = None

    for attempt in range(1, retries + 1):
        try:
            kwargs = {
                "headers": headers,
                "params": params,
                "ssl": ssl_context,
                "timeout": aiohttp.ClientTimeout(total=timeout),
            }
            if method.upper() != "GET" and body:
                kwargs["json"] = body

            async with request_fn(url + route, **kwargs) as resp:
                last_status = resp.status
                text = await resp.text()
                last_text = text

                if resp.status == 200:
                    try:
                        return await resp.json(content_type=None)
                    except Exception:
                        raise Exception(f"Invalid JSON from {url + route}:\n{text}")

                # Retryable?
                retry_after = resp.headers.get("Retry-After")
                retryable = (resp.status in (408, 429)) or (500 <= resp.status < 600)
                if not retryable:
                    raise Exception(f"HTTP {resp.status} from {url + route}: {text}")

                wait_time = float(1.0)
                await asyncio.sleep(wait_time)

        except Exception as ex:
            # On network or JSON errors we still backoff and retry
            client.logger.error(f"Attempt {attempt} failed: {ex}")
            if attempt == retries:
                break
            await asyncio.sleep(min(2 ** attempt + random.uniform(0, 1), 5))

    raise Exception(f"Too many retries for {url + route} (last_status={last_status}, last_text={last_text[:256]!r})")
