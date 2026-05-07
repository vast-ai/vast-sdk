# ComfyUI Web Console

A small local web app that wraps the Vast SDK so you can interactively
submit ComfyUI workflows to a serverless endpoint, view the generated
assets, and watch the worker pool's state in real time. Same flow as
`examples/comfy_example.py`, but graphical and stateful — past requests
stack up so you can replay any of them with modifications.

## Quickstart

```bash
cd examples/comfyui_web_console
pip install -r requirements.txt
python server.py
# open http://127.0.0.1:8000
```

In the UI:

1. Paste your Vast API key.
2. Click **Load endpoints** and pick yours from the dropdown.
3. Edit the workflow JSON in the textarea. The default template is the
   standard ComfyUI api-wrapper `Text2Image` modifier.
4. Hit **Submit /generate/sync**. The right-hand panel polls
   `/get_endpoint_workers/` every 3s and shows live status / load /
   throughput per worker. The completed request lands at the top of
   the request feed below.

By default the console expects the worker to upload outputs to S3 and
pulls the assets directly from the returned URL — much cheaper than
shipping a base64-encoded video back through the local server.

If your endpoint has no S3 configured, tick **inline base64 outputs**.
That sets `input.return_outputs_as_base64=true` so the worker stuffs
each asset into the response as a base64 string, which the console
turns into a `Blob` + object URL for display. Useful for one-off
local demos; avoid for video-heavy workflows where the encoding
overhead is noticeable.

When an output entry comes back without either a `url` or inline
`data` (i.e. neither S3 nor base64), the card surfaces a hint instead
of rendering blank.

## Outputs

Each request card renders all returned assets — single image, batches
of N images, video, audio, or a mix — using the right `<img>` /
`<video>` / `<audio>` tag based on the file extension. Supported:

| Kind  | Extensions |
|-------|------------|
| image | png, jpg, jpeg, gif, webp, avif |
| video | mp4, webm, mov, mkv, m4v |
| audio | mp3, wav, ogg, flac, m4a, aac |

Anything else falls back to a download link.

## Request history & memory

Past requests pile up at the top of the **Requests** feed (newest
first). Each card carries:

- a status pill, timestamp, duration
- the original payload (collapsible)
- a `Reload workflow` button that drops the request's payload back into
  the editor for replay-with-modifications
- the rendered output assets

Inline base64 assets are converted to `Blob` + `URL.createObjectURL`
so the browser can manage them as binary blobs rather than holding
megabyte-scale data URLs in the DOM. The feed caps at 20 entries by
default; older entries are pruned and their object URLs revoked
explicitly. Bump `MAX_HISTORY` in `static/index.html` if you want a
deeper history (memory cost scales with output size).

## How it works

```
browser  ──POST──▶  FastAPI server  ──Serverless SDK──▶  vast.ai
              ◀──JSON─                            ◀──JSON─
```

The server doesn't hold the API key — the browser sends it with each
request. Means you can leave the server running and rotate keys without
restarting it, and two operators on the same machine don't share creds.
Sessions and endpoint lookups are cached per-API-key for the lifetime
of the process so the 3-second worker poll doesn't churn aiohttp
connections or relist endpoints every tick.

Bound to `127.0.0.1` by default. Override with `HOST=0.0.0.0 PORT=9000
python server.py` if you need to share it on a LAN — but note there's
no auth, so don't put it on the open internet.

## Endpoints

The server proxies three operations, mirroring the SDK methods:

| Route | SDK call |
|-------|----------|
| `POST /api/endpoints` | `Serverless.get_endpoints()` |
| `POST /api/workers`   | `Endpoint.get_workers()` |
| `POST /api/submit`    | `Endpoint.request(route, payload)` |

All three accept `api_key` in the body; submit/workers also need
`endpoint_id`. The OpenAPI docs at `/docs` are the canonical reference.
