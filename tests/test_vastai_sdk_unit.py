"""Unit tests for vastai.vastai_sdk helpers and VastAI configuration surface.

CLI wrapper execution is not exercised; tests focus on creds, query parsing, and introspection.
"""
from __future__ import annotations

import argparse
from unittest.mock import MagicMock, mock_open, patch

import pytest

from vastai.vast import APIKEY_FILE


class TestQueryParser:
    """queryParser pre/post processing for search hooks."""

    def test_query_parser_leaves_kwargs_when_query_absent(self) -> None:
        """
        Verifies queryParser returns unchanged kwargs when no query key is set.

        This test verifies by:
        1. Calling queryParser with kwargs without 'query'
        2. Asserting returned kwargs equal input and default state flags

        Assumptions:
        - queryParser only mutates kwargs when query is not None
        """
        from vastai.vastai_sdk import queryParser

        inst = MagicMock()
        kwargs = {"other": 1}
        state, out = queryParser(dict(kwargs), inst)
        assert out == kwargs
        assert state["georegion"] is False
        assert state["chunked"] is False

    def test_query_parser_transforms_simple_query_string(self) -> None:
        """
        Verifies queryParser parses a simple filter expression into normalized query text.

        This test verifies by:
        1. Passing a query string with a key/operator/value triple
        2. Asserting output query string is non-empty and contains the key

        Assumptions:
        - pyparsing accepts the chosen expression format
        """
        from vastai.vastai_sdk import queryParser

        inst = MagicMock()
        _, out = queryParser({"query": "gpu_name = foo"}, inst)
        assert "gpu_name" in out["query"]


class TestQueryFormatter:
    """queryFormatter catalog post-processing."""

    def test_query_formatter_adds_datacenter_flag(self) -> None:
        """
        Verifies queryFormatter sets datacenter boolean from hosting_type.

        This test verifies by:
        1. Passing a single result dict with hosting_type=1
        2. Asserting datacenter True in filtered output

        Assumptions:
        - state georegion/chunked false skips optional branches
        """
        from vastai.vastai_sdk import queryFormatter

        inst = MagicMock()
        state = {"georegion": False, "chunked": False}
        rows = [{"hosting_type": 1, "geolocation": None}]
        out = queryFormatter(state, rows, inst)
        assert len(out) == 1
        assert out[0]["datacenter"] is True


class TestVastAICredentials:
    """VastAI api_key and creds_source behavior."""

    def test_creds_source_code_when_api_key_passed(self) -> None:
        """
        Verifies creds_source is CODE when constructor receives api_key.

        This test verifies by:
        1. Instantiating VastAI(api_key=...)
        2. Asserting creds_source and api_key fields

        Assumptions:
        - import_cli_functions completes without error
        """
        from vastai import VastAI

        v = VastAI(api_key="sk-direct")
        assert v.api_key == "sk-direct"
        assert v.creds_source == "CODE"

    def test_creds_source_file_when_keyfile_exists(self) -> None:
        """
        Verifies VastAI reads api_key from APIKEY_FILE when api_key is omitted and file exists.

        This test verifies by:
        1. Patching os.path.exists to True only for APIKEY_FILE
        2. Patching open to return a stripped key line
        3. Asserting creds_source FILE and api_key value

        Assumptions:
        - VastAI.__init__ uses vastai.vastai_sdk.os.path.exists for the key file check
        """
        from vastai.vastai_sdk import VastAI

        def _exists(path: str) -> bool:
            return path == APIKEY_FILE

        with patch("vastai.vastai_sdk.os.path.exists", side_effect=_exists):
            with patch("builtins.open", mock_open(read_data="sk-from-file\n")):
                v = VastAI(api_key=None)
        assert v.creds_source == "FILE"
        assert v.api_key == "sk-from-file"

    def test_creds_source_none_when_no_key_and_no_file(self) -> None:
        """
        Verifies VastAI records NONE when no api_key and no key file.

        This test verifies by:
        1. Patching exists to False for the key file path
        2. Constructing VastAI(api_key=None)
        3. Asserting creds_source NONE and api_key is None

        Assumptions:
        - No other code path sets api_key when file is missing
        """
        from vastai.vastai_sdk import VastAI

        with patch("vastai.vastai_sdk.os.path.exists", return_value=False):
            v = VastAI(api_key=None)
        assert v.creds_source == "NONE"
        assert v.api_key is None

    def test_credentials_on_disk_is_noop(self) -> None:
        """
        Verifies credentials_on_disk callable does not raise.

        This test verifies by:
        1. Instantiating VastAI with api_key
        2. Calling credentials_on_disk()

        Assumptions:
        - Method exists for compatibility and is intentionally empty
        """
        from vastai import VastAI

        v = VastAI(api_key="k")
        v.credentials_on_disk()


class TestVastAIIntrospection:
    """Dynamic CLI method binding and getattr behavior."""

    def test_getattr_unknown_raises(self) -> None:
        """
        Verifies __getattr__ raises for names not in imported_methods.

        This test verifies by:
        1. Instantiating VastAI
        2. Accessing a clearly invalid attribute name
        3. Asserting AttributeError message

        Assumptions:
        - 'totally_missing_attr_xyz_123' is not a CLI command
        """
        from vastai import VastAI

        v = VastAI(api_key="k")
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = v.totally_missing_attr_xyz_123

    def test_imported_methods_populated_for_known_commands(self) -> None:
        """
        Verifies import_cli_functions records metadata for at least one CLI command.

        This test verifies by:
        1. Instantiating VastAI
        2. Asserting imported_methods is a non-empty dict

        Assumptions:
        - vast.py parser exposes subparsers in this environment
        """
        from vastai import VastAI

        v = VastAI(api_key="k")
        assert isinstance(v.imported_methods, dict)
        assert len(v.imported_methods) > 0

    def test_generate_signature_from_argparse_builds_signature(self) -> None:
        """
        Verifies generate_signature_from_argparse returns Signature and docstring text.

        This test verifies by:
        1. Building a minimal argparse.ArgumentParser with one option
        2. Calling the method on VastAI
        3. Asserting 'self' and the dest appear in the signature parameters

        Assumptions:
        - Parser has standard _actions including help (filtered out in implementation)
        """
        from vastai import VastAI

        p = argparse.ArgumentParser(prog="t")
        p.add_argument("--count", type=int, default=3, help="count help")

        v = VastAI(api_key="k")
        sig, doc = v.generate_signature_from_argparse(p)
        assert "self" in sig.parameters
        assert "count" in sig.parameters
        assert "Args:" in doc
