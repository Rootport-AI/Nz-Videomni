"""``python -m mcp_server`` entrypoint (stdio transport).

Resolves the repo root from ``__file__`` (the same trick as ``config.py``'s
``PROJECT_ROOT``) and inserts it into ``sys.path`` before importing the rest of
the package. This is deliberate, not defensive dead code: the generated
``.mcp.json`` invokes this module with an absolute ``python.exe`` path, but the
MCP client (Claude Code, etc.) may launch the process from an arbitrary
working directory -- e.g. the owner's dev machine opens the PARENT folder as
the workspace, not this repo root. Without this fixup, ``import config`` and
``import mcp_server.*`` would fail depending on cwd/PYTHONPATH, which a
production review flagged as the top blocking risk (see the approved plan,
review item 1).

NEVER print() here or anywhere else under ``mcp_server/``: stdio carries
JSON-RPC on stdout, and any stray byte on that stream corrupts every message
after it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcp_server.server import main  # noqa: E402  (path fixup must run first)

main()
