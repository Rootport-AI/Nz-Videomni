"""MCP server settings resolution.

Precedence: ``LTX_MCP_*`` environment variables override ``config.yaml``
(``config.load_config`` stays the single source of truth otherwise -- this
mirrors ``main.py``'s "CLI overrides on top of config.yaml" pattern, but for
env vars instead of argv, since a stdio MCP server has no argv of its own).

Read once at process start (``load_settings()`` is called lazily by
``client.get_client()`` the first time a tool needs the backend); there is no
hot-reload -- a ``config.yaml`` edit that changes ``server.host``/``port``
needs the MCP server process restarted, same as the backend itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from config import UploadConfig, load_config

# host values that mean "listen on every interface" -- never something an HTTP
# client should connect *to*. Read back as loopback instead (mirrors how a
# browser would reach a --listen server: via 127.0.0.1 from the same machine).
_UNROUTABLE_HOSTS = {"0.0.0.0", "::"}


@dataclass(frozen=True)
class Settings:
    """Resolved MCP server settings for one process lifetime."""

    base_url: str  # e.g. "http://127.0.0.1:18620" -- WITHOUT the /api/v1 prefix
    api_key: str | None
    output_dir: Path
    # W2: local pre-check source for tools/uploads.py (extension + size gate) --
    # read straight from config.yaml's ``upload:`` section so the MCP-side
    # precheck can never drift from the server's own allow-list (services/
    # {upload,video_upload,audio_upload}_store.py), which stays the actual
    # authority (a 400 UPLOAD_INVALID_TYPE / UPLOAD_TOO_LARGE from the server
    # is still possible if config.yaml changes between precheck and POST).
    upload: UploadConfig = field(default_factory=UploadConfig)


def load_settings() -> Settings:
    """Resolve :class:`Settings` from ``LTX_MCP_*`` env vars + ``config.yaml``."""
    config_path = os.environ.get("LTX_MCP_CONFIG") or None
    config = load_config(config_path)

    base_url_env = os.environ.get("LTX_MCP_BASE_URL")
    if base_url_env:
        base_url = base_url_env.rstrip("/")
    else:
        host = config.server.host
        if host in _UNROUTABLE_HOSTS:
            host = "127.0.0.1"
        base_url = f"http://{host}:{config.server.port}"

    # An env var that is SET (even to "") always wins over config.yaml: that is
    # how an operator explicitly clears the key from the MCP side without
    # editing config.yaml. Unset env -> fall back to config; an empty string in
    # config.yaml also means "no key" (config.yaml's api_key: "" is a common
    # accidental-empty state, not a request for Bearer "").
    api_key_env = os.environ.get("LTX_MCP_API_KEY")
    if api_key_env is not None:
        api_key = api_key_env or None
    else:
        api_key = config.server.api_key or None

    output_dir_env = os.environ.get("LTX_MCP_OUTPUT_DIR")
    output_dir = Path(output_dir_env) if output_dir_env else config.output_dir

    return Settings(
        base_url=base_url, api_key=api_key, output_dir=output_dir, upload=config.upload
    )
