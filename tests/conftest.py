"""
Pytest configuration for the vast-sdk test suite.

The vastai package imports serverless server code (vastai/serverless/server/lib/backend.py)
which requires pycryptodome (Crypto).  That package may not be present in all dev
environments, so we stub it in sys.modules here — before any test module is imported —
so the package initialisation doesn't blow up.
"""
import sys
from unittest.mock import MagicMock

_crypto = MagicMock()
for _mod in (
    "Crypto",
    "Crypto.Signature",
    "Crypto.Signature.pkcs1_15",
    "Crypto.Hash",
    "Crypto.Hash.SHA256",
    "Crypto.PublicKey",
    "Crypto.PublicKey.RSA",
):
    sys.modules.setdefault(_mod, _crypto)
