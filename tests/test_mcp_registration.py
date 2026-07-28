"""MCP tool registration contract.

Checks that stay true across waves: the tool NAME set (grown by later waves --
this module's EXPECTED_TOOLS is the running total, extended in place as
uploads/generate/jobs/outputs/batch land), every tool has a non-empty
description (an agent reads only the description to decide whether to call a
tool), and one end-to-end call through the real MCP protocol machinery
(``mcp.shared.memory.create_connected_server_and_client_session``) proving the
structured-output contract: a ``dict[str, Any]``-returning tool's result comes
back as-is in ``structuredContent`` -- NOT wrapped in ``{"result": ...}``
(plan's confirmed SDK behavior for FastMCP 1.28).
"""

from __future__ import annotations

import anyio

import mcp.shared.memory as mcp_memory
from mcp_server.client import set_client
from mcp_server.server import build_server

# Final total (system 6 + uploads 3 + generate 2 + jobs 7 + outputs 3 +
# batch 1 = 22). This is the one place the running total lives.
EXPECTED_TOOLS = {
    "backend_status",
    "get_config",
    "list_models",
    "load_pipeline",
    "unload_pipeline",
    "list_loras",
    "upload_image",
    "upload_video",
    "upload_audio",
    "submit_generate",
    "submit_chain",
    "job_status",
    "list_jobs",
    "wait_for_job",
    "cancel_job",
    "delete_job",
    "purge_terminal_jobs",
    "join_job",
    "get_job_video_path",
    "get_joined_video_path",
    "save_job_video",
    "plan_a2v_batch",
}


def test_tool_name_set_matches_expected():
    async def _run() -> set[str]:
        mcp = build_server()
        tools = await mcp.list_tools()
        return {t.name for t in tools}

    names = anyio.run(_run)
    assert names == EXPECTED_TOOLS


def test_every_tool_has_a_nonempty_description():
    async def _run():
        mcp = build_server()
        return await mcp.list_tools()

    tools = anyio.run(_run)
    assert tools, "expected at least one registered tool"
    for tool in tools:
        assert tool.description and tool.description.strip(), (
            f"tool {tool.name!r} has no description"
        )


def test_backend_status_structured_content_not_wrapped_and_reachable_false():
    """No backend running -> reachable:false, and the dict comes back as the
    RAW structuredContent (not {"result": {...}})."""
    set_client(None)  # ensure a fresh singleton pointed at an unreachable default

    async def _run():
        server = build_server()
        async with mcp_memory.create_connected_server_and_client_session(server) as session:
            return await session.call_tool("backend_status", {})

    try:
        result = anyio.run(_run)
    finally:
        set_client(None)

    assert result.isError is False
    content = result.structuredContent
    assert content is not None
    assert "result" not in content  # not wrapped
    assert content["reachable"] is False
    assert "base_url" in content
    assert "error" in content
