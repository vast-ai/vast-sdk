"""
Tests for search__templates and create__template return-value behavior.

These tests verify that both functions correctly return data when args.raw=True
(the SDK path) and fall through to display/print when args.raw=False (the CLI path).
"""
import argparse
from unittest.mock import MagicMock, patch

import pytest


def make_args(**overrides):
    """Build a minimal argparse.Namespace sufficient for both tested functions."""
    defaults = dict(
        raw=True,
        query=None,
        api_key="test-key",
        url="https://console.vast.ai",
        explain=False,
        curl=False,
        retry=3,
        # create_template fields
        name="test-template",
        image="ubuntu:22.04",
        image_tag=None,
        href=None,
        repo=None,
        login=None,
        env=None,
        onstart_cmd=None,
        no_default=True,   # skip parse_query side-effects
        search_params=None,
        disk_space=None,
        jupyter=False,
        direct=False,
        jupyter_lab=False,
        jupyter_dir=None,
        ssh=False,
        readme=None,
        hide_readme=False,
        desc=None,
        public=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_get_response(templates):
    """Return a mock HTTP response for search__templates."""
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"Content-Type": "application/json"}
    resp.json.return_value = {"templates": templates}
    resp.raise_for_status.return_value = None
    return resp


def make_post_response(success=True, template=None, msg="error"):
    """Return a mock HTTP response for create__template."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    if success:
        resp.json.return_value = {"success": True, "template": template or {"id": 42, "name": "test-template"}}
    else:
        resp.json.return_value = {"success": False, "msg": msg}
    return resp


# ---------------------------------------------------------------------------
# search__templates
# ---------------------------------------------------------------------------

@patch("vastai.vast.apiurl", return_value="https://console.vast.ai/api/v0/template/")
@patch("vastai.vast.http_get")
def test_search_templates_raw_returns_list(mock_http_get, mock_apiurl):
    from vastai.vast import search__templates

    mock_http_get.return_value = make_get_response([{"id": 1, "name": "my-template"}])

    result = search__templates(make_args())

    assert result == [{"id": 1, "name": "my-template"}]


@patch("vastai.vast.apiurl", return_value="https://console.vast.ai/api/v0/template/")
@patch("vastai.vast.http_get")
def test_search_templates_empty_list_returns_empty_list(mock_http_get, mock_apiurl):
    """An empty templates array should return [] not None."""
    from vastai.vast import search__templates

    mock_http_get.return_value = make_get_response([])

    result = search__templates(make_args())

    assert result == []
    assert result is not None


@patch("vastai.vast.display_table")
@patch("vastai.vast.apiurl", return_value="https://console.vast.ai/api/v0/template/")
@patch("vastai.vast.http_get")
def test_search_templates_not_raw_calls_display_table(mock_http_get, mock_apiurl, mock_display):
    from vastai.vast import search__templates

    mock_http_get.return_value = make_get_response([{"id": 1}])

    result = search__templates(make_args(raw=False))

    assert result is None
    mock_display.assert_called_once()


# ---------------------------------------------------------------------------
# create__template
# ---------------------------------------------------------------------------

@patch("vastai.vast.apiurl", return_value="https://console.vast.ai/api/v0/template/")
@patch("vastai.vast.http_post")
def test_create_template_raw_returns_template_dict(mock_http_post, mock_apiurl):
    from vastai.vast import create__template

    expected = {"id": 42, "name": "test-template", "hash": "abc123"}
    mock_http_post.return_value = make_post_response(template=expected)

    result = create__template(make_args())

    assert result == expected


@patch("vastai.vast.apiurl", return_value="https://console.vast.ai/api/v0/template/")
@patch("vastai.vast.http_post")
def test_create_template_not_raw_returns_none(mock_http_post, mock_apiurl):
    from vastai.vast import create__template

    mock_http_post.return_value = make_post_response()

    result = create__template(make_args(raw=False))

    assert result is None


@patch("vastai.vast.apiurl", return_value="https://console.vast.ai/api/v0/template/")
@patch("vastai.vast.http_post")
def test_create_template_api_failure_returns_none(mock_http_post, mock_apiurl):
    """When the API returns success=False the function should print the message and return None."""
    from vastai.vast import create__template

    mock_http_post.return_value = make_post_response(success=False, msg="duplicate name")

    result = create__template(make_args())

    assert result is None
