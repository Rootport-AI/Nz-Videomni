"""Error-translation contract for mcp_server/client.py.

Three shapes:
  1. the backend's ``{"error": {code, message, detail}}`` envelope -> a
     ``ToolError`` whose message contains code + message + detail,
  2. a non-JSON error body (still a 4xx/5xx) -> ``ToolError`` containing
     ``"HTTP_<status>"``,
  3. a transport-level connect failure -> ``ToolError`` containing
     ``"BACKEND_UNREACHABLE"`` for a normal tool, but ``backend_status``
     swallows it and reports ``reachable: false`` instead (it must never
     raise -- see test_mcp_registration.py for the full round-trip version of
     this one).
"""

from __future__ import annotations

from pathlib import Path

import anyio
import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from mcp_server.client import BackendClient, set_client
from mcp_server.settings import Settings
from mcp_server.tools import system


def _client_for_handler(handler) -> BackendClient:
    settings = Settings(base_url="http://testserver", api_key=None, output_dir=Path("."))
    return BackendClient(settings, transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _reset_client():
    set_client(None)
    yield
    set_client(None)


def test_error_envelope_code_message_detail_all_appear():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error": {
                    "code": "IMAGE_NOT_FOUND",
                    "message": "image_id not found: abc123",
                    "detail": "some extra detail",
                }
            },
        )

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(system.get_config)  # any GET-based tool exercises the same _raise_for_error path

    text = str(exc_info.value)
    assert "IMAGE_NOT_FOUND" in text
    assert "image_id not found: abc123" in text
    assert "some extra detail" in text


def test_error_envelope_without_detail_omits_the_dash():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": {"code": "JOB_NOT_FOUND", "message": "job not found: xyz"}}
        )

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(system.get_config)

    assert "JOB_NOT_FOUND: job not found: xyz" in str(exc_info.value)


def test_non_json_500_becomes_http_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"<html>Internal Server Error</html>")

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(system.get_config)

    assert "HTTP_500" in str(exc_info.value)


def test_connect_error_becomes_backend_unreachable_for_a_normal_tool():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(system.get_config)

    assert "BACKEND_UNREACHABLE" in str(exc_info.value)


def test_connect_error_leaves_backend_status_reachable_false():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    set_client(_client_for_handler(handler))

    result = anyio.run(system.backend_status)

    assert result["reachable"] is False
    assert "BACKEND_UNREACHABLE" in result["error"]
