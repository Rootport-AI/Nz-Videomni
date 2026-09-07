"""Reusable stdio MCP client: launch `python -m mcp_server` and call ONE tool.

Usage:
    python mcp_call.py <tool_name> '<json args>'
    python mcp_call.py <tool_name> --args-file <path>
    python mcp_call.py --list-tools
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni"
PY = r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\.venv\Scripts\python.exe"


def server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=PY,
        args=["-m", "mcp_server"],
        cwd=REPO,
        env={**os.environ, "PYTHONUTF8": "1"},
    )


def result_payload(res):
    sc = getattr(res, "structuredContent", None)
    if sc:
        return sc
    out = []
    for c in getattr(res, "content", []) or []:
        out.append(getattr(c, "text", None) or str(c))
    return {"_text": out, "_isError": getattr(res, "isError", None)}


async def run(tool: str | None, args: dict, list_tools: bool):
    async with stdio_client(server_params()) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            if list_tools:
                tools = await s.list_tools()
                print(json.dumps([t.name for t in tools.tools], ensure_ascii=False, indent=2))
                return
            res = await s.call_tool(tool, args)
            print(json.dumps(result_payload(res), ensure_ascii=False, indent=2))


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        print("usage: mcp_call.py <tool> '<json>' | --args-file <path> | --list-tools", file=sys.stderr)
        sys.exit(2)
    if argv[0] == "--list-tools":
        asyncio.run(run(None, {}, True))
        return
    tool = argv[0]
    args: dict = {}
    rest = argv[1:]
    if rest:
        if rest[0] == "--args-file":
            with open(rest[1], "r", encoding="utf-8") as fh:
                args = json.load(fh)
        else:
            args = json.loads(rest[0])
    asyncio.run(run(tool, args, False))


if __name__ == "__main__":
    main()
