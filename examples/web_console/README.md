# Vast Serverless Web Console

A small local web app that wraps the Vast SDK so you can interactively
submit ComfyUI workflows to a serverless endpoint and watch the worker
pool's state in real time. Same flow as `examples/comfy_example.py`,
but graphical.

## Quickstart

```bash
cd examples/web_console
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
   throughput per worker.

The "request base64 outputs" checkbox adds
`input.return_outputs_as_base64=true` so generated images / videos
come back inline without needing S3 set up. Toggle it off if your
endpoint already uploads to S3 and you'd rather load from there.

## How it works

```
browser  ──POST──▶  FastAPI server  ──Serverless SDK──▶  vast.ai
              ◀──JSON─                            ◀──JSON─
```

The server doesn't hold the API key — the browser sends it with each
request. Means you can leave the server running and rotate keys without
restarting it, and two operators on the same machine don't share creds.

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
