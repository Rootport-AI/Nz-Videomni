"""MMAudio handler — runs MMAudio video-to-audio inference via WSL subprocess."""

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
from api_types import MMAudioGenerateRequest, MMAudioProgressResponse

logger = logging.getLogger(__name__)

MMAudioStatus = Literal["idle", "running", "complete", "error", "cancelled"]

# Adjust these paths to match your WSL MMAudio install.
_WSL_DISTRO = "Ubuntu-24.04"
_WSL_USER = "mike_hunt"
_MMAUDIO_DIR = "/home/mike_hunt/MMAudio"
_MMAUDIO_CONDA_ENV = "mmaudio"   # conda env name, or set _MMAUDIO_PYTHON to a venv path
_MMAUDIO_PYTHON: str | None = "/home/mike_hunt/miniconda3/envs/mmaudio/bin/python"
_MAX_LOG_LINES = 200


@dataclass
class _MMAudioJob:
    status: MMAudioStatus = "idle"
    output_path: str | None = None
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = None  # type: ignore[type-arg]


class MMAudioHandler:
    """Runs MMAudio video-to-audio inference in a background thread via WSL."""

    def __init__(self, outputs_dir: Path) -> None:
        self._outputs_dir = outputs_dir
        self._lock = threading.Lock()
        self._job = _MMAudioJob()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_progress(self) -> MMAudioProgressResponse:
        with self._lock:
            j = self._job
            return MMAudioProgressResponse(
                status=j.status,
                output_path=j.output_path,
                error=j.error,
                log_tail="\n".join(j.log_lines[-50:]),
            )

    def generate(self, req: MMAudioGenerateRequest) -> MMAudioProgressResponse:
        with self._lock:
            if self._job.status == "running":
                raise HTTPError(409, "MMAudio generation already in progress")
            self._job = _MMAudioJob(status="running")

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

    def _run(self, req: MMAudioGenerateRequest) -> None:
        stem = f"mmaudio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        wsl_out_dir = f"/tmp/mmaudio_out_{stem}"
        wsl_video_path = _win_to_wsl_path(req.video_path)

        # Build the python interpreter / run command
        if _MMAUDIO_PYTHON:
            py_prefix = _MMAUDIO_PYTHON
        else:
            py_prefix = f"conda run -n {_MMAUDIO_CONDA_ENV} --no-capture-output python"

        demo_script = f"{_MMAUDIO_DIR}/demo.py"
        prompt_arg = req.prompt if req.prompt.strip() else "natural ambient sound"
        duration_arg = str(req.duration)

        safe_prompt = prompt_arg.replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))
        inner_cmd = (
            f"mkdir -p '{wsl_out_dir}' && "
            f"cd '{_MMAUDIO_DIR}' && "
            f"{py_prefix} '{demo_script}'"
            f" --video '{wsl_video_path}'"
            f" --prompt '{safe_prompt}'"
            f" --duration {duration_arg}"
            f" --output '{wsl_out_dir}'"
            f" --cfg_strength {req.cfg_strength}"
            f" --num_steps {req.num_steps}"
        )
        if req.seed is not None:
            inner_cmd += f" --seed {req.seed}"
        if req.negative_prompt:
            safe_neg = req.negative_prompt.replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))
            inner_cmd += f" --negative_prompt '{safe_neg}'"

        cmd = ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c", inner_cmd]
        logger.info("[mmaudio] Starting: video=%s prompt=%r duration=%s", req.video_path, req.prompt, req.duration)

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
                logger.info("[mmaudio] %s", line)
                with self._lock:
                    self._job.log_lines.append(line)
                    if len(self._job.log_lines) > _MAX_LOG_LINES:
                        self._job.log_lines = self._job.log_lines[-_MAX_LOG_LINES:]

            process.wait()

            with self._lock:
                if self._job.status == "cancelled":
                    return
                if process.returncode != 0:
                    self._job.status = "error"
                    self._job.error = f"MMAudio exited with code {process.returncode}"
                    return

            # Stream the output MP4 from WSL via cat → write on Windows side.
            # Avoids WSL→Windows cp failures on /mnt/c/ paths.
            # Pipe find → xargs cat — avoids $() subshell which returns empty
            # when called via wsl.exe bash -c from Windows Python.
            cat_result = subprocess.run(
                ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c",
                 f"find '{wsl_out_dir}' -maxdepth 1 -name '*.mp4' -type f"
                 f" | head -1 | xargs -r cat"],
                capture_output=True,
            )
            if not cat_result.stdout:
                ls_out = subprocess.run(
                    ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c",
                     f"ls '{wsl_out_dir}/' 2>&1"],
                    capture_output=True, text=True,
                ).stdout.strip()
                with self._lock:
                    self._job.status = "error"
                    self._job.error = (
                        f"MMAudio output MP4 not found or empty. "
                        f"Dir: {ls_out or '(empty)'}. "
                        f"stderr: {cat_result.stderr.decode('utf-8', errors='replace').strip()}"
                    )
                return

            win_out = self._outputs_dir / f"{stem}.mp4"
            win_out.write_bytes(cat_result.stdout)

            # Clean up WSL temp dir
            subprocess.run(
                ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c", f"rm -rf '{wsl_out_dir}'"],
                capture_output=True,
            )

            with self._lock:
                self._job.status = "complete"
                self._job.output_path = str(win_out)
                self._job.log_lines.append(f"✓ Output: {win_out}")

        except Exception as exc:
            logger.exception("[mmaudio] Unexpected error")
            with self._lock:
                self._job.status = "error"
                self._job.error = str(exc)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _win_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path like C:\\Users\\... to /mnt/c/Users/..."""
    p = win_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        p = f"/mnt/{drive}{p[2:]}"
    return p
