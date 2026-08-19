"""Nz-Videomni MCP server (AI-agent facing bridge over the existing HTTP API).

Import-only: this module must have zero side effects (no logging config, no
network, no filesystem access) so ``import mcp_server`` is always safe from
any process, including the test suite.
"""

from __future__ import annotations

__version__ = "0.1.0"
