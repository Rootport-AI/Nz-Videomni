"""MagiHuman handler — runs daVinci-MagiHuman inference via WSL subprocess."""

from __future__ import annotations

import logging
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from _routes._errors import HTTPError
from api_types import MagiGenerateRequest, MagiProgressResponse

logger = logging.getLogger(__name__)

MagiStatus = Literal["idle", "running", "complete", "error", "cancelled"]

_MAGI_SH = "/home/mike_hunt/magi.sh"
_WSL_DISTRO = "Ubuntu-24.04"
_WSL_USER = "mike_hunt"
_WSL_OUTPUT_DIR = "/home/mike_hunt/daVinci-MagiHuman"
_WSL_SR_MODEL_DIR = "/home/mike_hunt/models/daVinci-MagiHuman/540p_sr"
_MAX_LOG_LINES = 200


@dataclass
class _MagiJob:
    status: MagiStatus = "idle"
    output_path: str | None = None
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = None  # type: ignore[type-arg]


class MagiHandler:
    """Runs daVinci-MagiHuman inference in a background thread via WSL."""

    def __init__(self, outputs_dir: Path) -> None:
        self._outputs_dir = outputs_dir
        self._lock = threading.Lock()
        self._job = _MagiJob()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_progress(self) -> MagiProgressResponse:
        with self._lock:
            j = self._job
            return MagiProgressResponse(
                status=j.status,
                output_path=j.output_path,
                error=j.error,
                log_tail="\n".join(j.log_lines[-50:]),
                sr_model_ready=_sr_model_ready(),
            )

    def generate(self, req: MagiGenerateRequest) -> MagiProgressResponse:
        with self._lock:
            if self._job.status == "running":
                raise HTTPError(409, "MagiHuman generation already in progress")
            self._job = _MagiJob(status="running")

        thread = threading.Thread(target=self._run, args=(req,), daemon=True)
        thread.start()
        return self.get_progress()

    def cancel(self) -> None:
        with self._lock:
            if self._job.process and self._job.status == "running":
                try:
                    self._job.process.terminate()
                except Exception:
                    pass
                self._job.status = "cancelled"

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _run(self, req: MagiGenerateRequest) -> None:
        # Pre-flight: verify SR model exists before attempting a run that will
        # fail 3 minutes in with a cryptic FileNotFoundError.
        if req.sr:
            check = subprocess.run(
                ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c",
                 f"test -f '{_WSL_SR_MODEL_DIR}/model.safetensors.index.json' && echo OK || echo MISSING"],
                capture_output=True, text=True,
            )
            if "OK" not in check.stdout:
                with self._lock:
                    self._job.status = "error"
                    self._job.error = (
                        "SR model not downloaded. Run in WSL: bash ~/download_sr.sh\n"
                        "(~61 GB download, resumes if interrupted)"
                    )
                return

        stem = f"magi_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        wsl_output_stem = f"{_WSL_OUTPUT_DIR}/{stem}"
        # magi.sh (and entry.py) appends _{secs}s_{w}x{h}.mp4 to the output stem.
        # When SR is active, entry.py names the file with the SR (2×) dimensions.
        out_w = req.width * 2 if req.sr else req.width
        out_h = req.height * 2 if req.sr else req.height
        expected_filename = f"{stem}_{req.seconds}s_{out_w}x{out_h}.mp4"
        expected_wsl_path = f"{_WSL_OUTPUT_DIR}/{expected_filename}"

        image_wsl = _win_to_wsl_path(req.image_path)

        cmd = [
            "wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER,
            "bash", _MAGI_SH,
            "--prompt", req.prompt,
            "--image", image_wsl,
            "--seconds", str(req.seconds),
            "--width", str(req.width),
            "--height", str(req.height),
            "--output", wsl_output_stem,
            "--gpus", str(req.gpus),
        ]
        if req.seed is not None:
            cmd += ["--seed", str(req.seed)]
        if req.sr:
            cmd += ["--sr_width", str(req.width * 2), "--sr_height", str(req.height * 2)]

        logger.info("[magi] Starting: %s", " ".join(cmd[:8]))

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            with self._lock:
                self._job.process = process

            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                logger.info("[magi] %s", line)
                with self._lock:
                    self._job.log_lines.append(line)
                    if len(self._job.log_lines) > _MAX_LOG_LINES:
                        self._job.log_lines = self._job.log_lines[-_MAX_LOG_LINES:]

            process.wait()

            # Wait for both GPUs to fully release before any subsequent torchrun.
            # A fixed sleep isn't reliable — poll nvidia-smi until VRAM used drops
            # below 1 GB on both devices (idle state), with a 30-second hard cap.
            _wait_for_gpu_idle(distro=_WSL_DISTRO, user=_WSL_USER, timeout=30)

            with self._lock:
                if self._job.status == "cancelled":
                    return
                if process.returncode != 0:
                    self._job.status = "error"
                    self._job.error = f"magi.sh exited with code {process.returncode}"
                    return

            # Copy output from WSL to Windows outputs_dir
            win_out = self._outputs_dir / expected_filename
            win_wsl = _win_to_wsl_path(str(win_out))

            # Try exact filename first, then glob fallback
            copy_bash = (
                f"if [ -f '{expected_wsl_path}' ]; then "
                f"  cp '{expected_wsl_path}' '{win_wsl}' && echo OK; "
                f"else "
                f"  f=$(ls -t '{_WSL_OUTPUT_DIR}/{stem}'*.mp4 2>/dev/null | head -1); "
                f"  [ -n \"$f\" ] && cp \"$f\" '{win_wsl}' && echo OK || echo NOTFOUND; "
                f"fi"
            )
            cp = subprocess.run(
                ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c", copy_bash],
                capture_output=True, text=True,
            )
            if "OK" not in cp.stdout:
                with self._lock:
                    self._job.status = "error"
                    self._job.error = f"Output file not found after generation. stderr: {cp.stderr.strip()}"
                return

            with self._lock:
                self._job.status = "complete"
                self._job.output_path = str(win_out)
                self._job.log_lines.append(f"✓ Output: {win_out}")

        except Exception as exc:
            logger.exception("[magi] Unexpected error")
            with self._lock:
                self._job.status = "error"
                self._job.error = str(exc)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _wait_for_gpu_idle(distro: str, user: str, timeout: int = 30) -> None:
    """Poll nvidia-smi until all GPUs show < 1 GB used, or until timeout seconds."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["wsl", "-d", distro, "-u", user, "bash", "-c",
                 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            used_mib = [int(l) for l in lines if l.isdigit()]
            if used_mib and all(v < 1024 for v in used_mib):
                return
        except Exception:
            pass
        time.sleep(2)
    # Hard cap reached — sleep a bit more as a last resort
    time.sleep(3)


def _sr_model_ready() -> bool:
    """Return True if the 540p_sr model index file exists in WSL."""
    try:
        result = subprocess.run(
            ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c",
             f"test -f '{_WSL_SR_MODEL_DIR}/model.safetensors.index.json' && echo OK || echo MISSING"],
            capture_output=True, text=True, timeout=5,
        )
        return "OK" in result.stdout
    except Exception:
        return False


def _win_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path like C:\\Users\\... to /mnt/c/Users/..."""
    p = win_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        p = f"/mnt/{drive}{p[2:]}"
    return p
