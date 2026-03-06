"""Backward compatibility shim for old import paths.

This module provides backward-compatible imports for code that used the old
vastai_sdk.py or vastai_base.py import paths.
"""

# Old: from vastai.vastai_sdk import VastAI
# New: from vastai.sdk import VastAI
# (handled by __init__.py already)

# Old: from vastai.vastai_base import VastAIBase
# Provide a stub for code that might reference this
from vastai.sdk import VastAI as VastAIBase

__all__ = ["VastAIBase"]
