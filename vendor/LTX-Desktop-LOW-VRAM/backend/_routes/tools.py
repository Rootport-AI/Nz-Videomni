"""Route handlers for /api/tools/status — checks optional tool availability."""
from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from handlers.prismaudio_handler import (  # type: ignore[import]
    _USE_WSL as _PRISMAUDIO_USE_WSL,
    _PRISMAUDIO_DIR_WIN,
    _PRISMAUDIO_DIR_WSL,
    _PRISMAUDIO_CONDA_ENV,
    _PRISMAUDIO_CONDA_ENV_WSL,
)

router = APIRouter(prefix="/api/tools", tags=["tools"])

_WSL_DISTRO = "Ubuntu-24.04"
_WSL_USER = "mike_hunt"
_MMAUDIO_DIR = "/home/mike_hunt/MMAudio"
_MAGI_SH = "/home/mike_hunt/magi.sh"


class ToolStatus(BaseModel):
    name: str
    available: bool
    detail: str  # short human-readable status message


class ToolsStatusResponse(BaseModel):
    tools: list[ToolStatus]
    wsl_available: bool


def _wsl_available() -> bool:
    """Return True if WSL is present and Ubuntu-24.04 is registered."""
    try:
        result = subprocess.run(
            ["wsl", "--list", "--quiet"],
            capture_output=True,
            timeout=8,
        )
        # wsl --list --quiet outputs UTF-16-LE on Windows; decode both variants
        try:
            text = result.stdout.decode("utf-16-le")
        except (UnicodeDecodeError, ValueError):
            text = result.stdout.decode("utf-8", errors="replace")
        return _WSL_DISTRO in text
    except Exception:
        return False


def _wsl_check(cmd: str) -> tuple[bool, str]:
    """
    Run *cmd* inside the configured WSL distro.
    Returns (success, stdout_stripped).
    """
    result = subprocess.run(
        ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c", cmd],
        capture_output=True,
        timeout=5,
    )
    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    return result.returncode == 0, stdout


def _check_mmaudio(wsl_ok: bool) -> ToolStatus:
    name = "MMAudio"
    if not wsl_ok:
        return ToolStatus(
            name=name,
            available=False,
            detail="WSL not available — required for MMAudio",
        )
    try:
        ok, out = _wsl_check(
            f"test -d '{_MMAUDIO_DIR}' && echo OK || echo MISSING"
        )
        if "OK" in out:
            return ToolStatus(
                name=name,
                available=True,
                detail=f"Installed at {_MMAUDIO_DIR}",
            )
        return ToolStatus(
            name=name,
            available=False,
            detail="Not found — run installers/install-mmaudio.ps1",
        )
    except Exception:
        return ToolStatus(name=name, available=False, detail="Check failed")


def _check_prismaudio(wsl_ok: bool) -> ToolStatus:
    name = "PrismAudio"
    try:
        if _PRISMAUDIO_USE_WSL:
            # WSL mode — check path inside WSL
            if not wsl_ok:
                return ToolStatus(
                    name=name,
                    available=False,
                    detail="WSL not available — required (handler configured with _USE_WSL=True)",
                )
            ok, out = _wsl_check(
                f"test -f '{_PRISMAUDIO_DIR_WSL}/predict.py' && "
                f"conda env list 2>/dev/null | grep -q '^{_PRISMAUDIO_CONDA_ENV_WSL}' && echo OK || echo MISSING"
            )
            if "OK" in out:
                return ToolStatus(
                    name=name,
                    available=True,
                    detail=f"WSL mode — installed at {_PRISMAUDIO_DIR_WSL} (env: {_PRISMAUDIO_CONDA_ENV_WSL})",
                )
            return ToolStatus(
                name=name,
                available=False,
                detail=f"WSL mode — not found at {_PRISMAUDIO_DIR_WSL} — run installers/install-prismaudio.ps1 -UseWSL",
            )
        else:
            # Windows-native mode
            root = Path(_PRISMAUDIO_DIR_WIN)
            if not (root.exists() and (root / "predict.py").exists()):
                return ToolStatus(
                    name=name,
                    available=False,
                    detail=f"Not found at {_PRISMAUDIO_DIR_WIN} — run installers/install-prismaudio.ps1",
                )
            conda_result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                timeout=10,
            )
            conda_out = conda_result.stdout.decode("utf-8", errors="replace")
            if _PRISMAUDIO_CONDA_ENV in conda_out:
                return ToolStatus(
                    name=name,
                    available=True,
                    detail=f"Installed at {_PRISMAUDIO_DIR_WIN} (conda env: {_PRISMAUDIO_CONDA_ENV})",
                )
            return ToolStatus(
                name=name,
                available=False,
                detail=f"Files found but '{_PRISMAUDIO_CONDA_ENV}' conda env missing — re-run installer",
            )
    except Exception:
        return ToolStatus(name=name, available=False, detail="Check failed")


def _check_magihuman(wsl_ok: bool) -> ToolStatus:
    name = "MagiHuman"
    if not wsl_ok:
        return ToolStatus(
            name=name,
            available=False,
            detail="WSL not available — required for MagiHuman",
        )
    try:
        ok, out = _wsl_check(
            f"test -f '{_MAGI_SH}' && echo OK || echo MISSING"
        )
        if "OK" in out:
            return ToolStatus(
                name=name,
                available=True,
                detail=f"Installed — launch script at {_MAGI_SH}",
            )
        return ToolStatus(
            name=name,
            available=False,
            detail="Not found — run installers/install-magihuman.ps1",
        )
    except Exception:
        return ToolStatus(name=name, available=False, detail="Check failed")


@router.get("/status", response_model=ToolsStatusResponse)
def route_tools_status() -> ToolsStatusResponse:
    """Check which optional AI tools are installed and available."""
    wsl_ok = _wsl_available()

    tools: list[ToolStatus] = [
        _check_mmaudio(wsl_ok),
        _check_prismaudio(wsl_ok),
        _check_magihuman(wsl_ok),
    ]

    return ToolsStatusResponse(tools=tools, wsl_available=wsl_ok)
