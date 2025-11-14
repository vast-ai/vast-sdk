import os

if os.environ.get("VAST_SKIP_PUBKEY") == "1":
    try:
        from vastai_sdk.serverless.server.lib import backend as _backend_mod

        def _no_pubkey(self):
            return None

        _backend_mod.Backend._fetch_pubkey = _no_pubkey
    except Exception as e:
        print(f"[sitecustomize] Could not patch pubkey fetch: {e}")
