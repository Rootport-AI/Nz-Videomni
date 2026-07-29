"""Tool registration entrypoint.

Each ``tools/*.py`` module exposes a ``register(mcp: FastMCP) -> None`` that
registers its own tools. :func:`register_all` just calls each in turn.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server.tools import batch, generate, jobs, outputs, system, uploads


def register_all(mcp: FastMCP) -> None:
    system.register(mcp)
    uploads.register(mcp)
    generate.register(mcp)
    jobs.register(mcp)
    outputs.register(mcp)
    batch.register(mcp)
