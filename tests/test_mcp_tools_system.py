"""Contract tests for mcp_server/tools/system.py.

Two flavors of backend double, matching the plan's test-plan section:

* ``_client_for_app`` -- a real (mock-backend) FastAPI app reached through
  ``httpx.ASGITransport`` (via the ``mcp_app`` fixture). Used for the tools
  that just proxy a GET (get_config / list_models / list_loras): these check
  the REAL response shape survives the round trip.
* ``httpx.MockTransport`` with a hand-written handler. Used for
  ``load_pipeline``'s branches and ``unload_pipeline``'s error path, where the
  test needs to assert exactly which HTTP calls happened (or didn't) --
  something a real app can't tell us directly.

All tool functions are plain ``async def``, so they're driven with
``anyio.run`` directly (no new pytest plugin, per the plan).
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


def _client_for_app(app, output_dir: Path) -> BackendClient:
    transport = httpx.ASGITransport(app=app)
    settings = Settings(base_url="http://testserver", api_key=None, output_dir=output_dir)
    return BackendClient(settings, transport=transport)


def _client_for_handler(handler) -> BackendClient:
    settings = Settings(base_url="http://testserver", api_key=None, output_dir=Path("."))
    return BackendClient(settings, transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _reset_client():
    """Every test installs its own client; never leak one into the next test."""
    set_client(None)
    yield
    set_client(None)


# ----- real-backend-backed tools (get_config / list_models / list_loras) ---


def test_get_config_returns_limits(mcp_app):
    app, output_dir = mcp_app
    set_client(_client_for_app(app, output_dir))
    result = anyio.run(system.get_config)
    assert "limits" in result
    assert "max_num_frames" in result["limits"]


def test_list_models_has_four_categories(mcp_app):
    app, output_dir = mcp_app
    set_client(_client_for_app(app, output_dir))
    result = anyio.run(system.list_models)
    categories = result["categories"]
    assert set(categories) == {"transformer", "text_encoder", "video_vae", "audio"}
    for entry in categories.values():
        assert "active" in entry and "entries" in entry


def test_list_loras_ok(mcp_app):
    app, output_dir = mcp_app
    set_client(_client_for_app(app, output_dir))
    result = anyio.run(system.list_loras)
    assert "loras" in result
    assert isinstance(result["loras"], list)


# ----- load_pipeline branches (MockTransport: assert which calls happen) ---


def test_load_pipeline_no_op_when_already_loaded():
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        assert request.method == "GET", "no-op branch must not POST"
        return httpx.Response(
            200, json={"pipeline_loaded": True, "pipeline_type": "distilled"}
        )

    set_client(_client_for_handler(handler))
    result = anyio.run(system.load_pipeline, None, 45)

    assert result == {"no_op": True, "pipeline_loaded": True, "pipeline_type": "distilled"}
    assert calls == [("GET", "/api/v1/status")]


def test_load_pipeline_in_flight_skips_post_after_precheck():
    """models=None -> the pre-check GET still runs, but the in-flight guard
    stops it short of any POST (order per the plan: pre-check first, then
    in-flight)."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        assert request.method == "GET"
        return httpx.Response(200, json={"pipeline_loaded": False})

    client = _client_for_handler(handler)
    client.pipeline_load_in_flight = True
    set_client(client)

    result = anyio.run(system.load_pipeline, None, 45)

    assert result["in_flight"] is True
    assert result["started"] is False
    assert result["finished"] is False
    assert calls == ["GET"]


def test_load_pipeline_in_flight_with_explicit_models_makes_no_http_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.method} {request.url.path}")

    client = _client_for_handler(handler)
    client.pipeline_load_in_flight = True
    set_client(client)

    result = anyio.run(system.load_pipeline, {"transformer": "some-name"}, 45)

    assert result["in_flight"] is True


def test_load_pipeline_read_timeout_returns_finished_false_not_an_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"pipeline_loaded": False})
        raise httpx.ReadTimeout("simulated timeout", request=request)

    client = _client_for_handler(handler)
    set_client(client)

    result = anyio.run(system.load_pipeline, None, 45)

    assert result["started"] is True
    assert result["finished"] is False
    assert "hint" in result
    # in-flight flag must be cleared even on the timeout path (try/finally)
    assert client.pipeline_load_in_flight is False


def test_load_pipeline_clamps_wait_sec():
    seen_read_timeouts: list[float | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"pipeline_loaded": False})
        timeout_ext = request.extensions.get("timeout") or {}
        seen_read_timeouts.append(timeout_ext.get("read"))
        return httpx.Response(
            200, json={"pipeline_loaded": True, "pipeline_type": "distilled", "state": "loaded"}
        )

    client = _client_for_handler(handler)
    set_client(client)

    anyio.run(system.load_pipeline, None, 1)  # below min -> clamped to 5
    anyio.run(system.load_pipeline, None, 999)  # above max -> clamped to 45

    assert seen_read_timeouts == [5, 45]


# ----- unload_pipeline error translation -----------------------------------


def test_unload_pipeline_job_busy_raises_tool_error_with_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "error": {
                    "code": "JOB_BUSY",
                    "message": "A job is already running (Phase 1 allows one concurrent job)",
                    "detail": "cannot unload while a job is running",
                }
            },
        )

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(system.unload_pipeline)

    assert "JOB_BUSY" in str(exc_info.value)
