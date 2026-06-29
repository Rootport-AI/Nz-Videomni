"""PrismAudio handler — runs ThinkSound/PrismAudio video-to-audio inference.

Supports two modes controlled by _USE_WSL:
  False (default) — native Windows, uses conda env or direct Python executable
  True            — WSL2 subprocess (same pattern as MMAudio/MagiHuman)

Inference flow (replicates scripts/PrismAudio/demo.sh):
  1. Copy video → videos/demo.mp4 in repo dir
  2. Write cot_coarse/cot.csv with the text prompt
  3. Feature extraction: torchrun data_utils/prismaudio_data_process.py --inference_mode True
  4. Inference: python predict.py --model-config ... --duration-sec ... --results-dir <tmp>
  5. Mix generated WAV with source video via ffmpeg → output .mp4
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from subprocess import PIPE, STDOUT
from typing import Literal

from _routes._errors import HTTPError
from api_types import PrismAudioGenerateRequest, PrismAudioProgressResponse

logger = logging.getLogger(__name__)

PrismAudioStatus = Literal["idle", "running", "complete", "error", "cancelled"]

# ---------------------------------------------------------------------------
# Configurable constants — adjust to match your local PrismAudio install.
# ---------------------------------------------------------------------------

# Set True to run PrismAudio inside WSL2 (Linux) instead of native Windows.
_USE_WSL = True

# --- Windows-native settings (used when _USE_WSL = False) ---
_PRISMAUDIO_DIR_WIN = "C:/AI/ThinkSound"
_PRISMAUDIO_CONDA_ENV = "prismaudio"
# Set to a direct Python path to skip conda run entirely, e.g.:
#   r"C:\miniconda3\envs\prismaudio\python.exe"
_PRISMAUDIO_PYTHON_WIN: str | None = None

# --- WSL settings (used when _USE_WSL = True) ---
_WSL_DISTRO = "Ubuntu-24.04"
_WSL_USER = "mike_hunt"
_PRISMAUDIO_DIR_WSL = "/home/mike_hunt/ThinkSound"
_PRISMAUDIO_CONDA_ENV_WSL = "prismaudio"
# Set to a direct Python path inside WSL to skip conda run, e.g.:
#   "/home/mike_hunt/miniconda3/envs/prismaudio/bin/python"
_PRISMAUDIO_PYTHON_WSL: str | None = "/home/mike_hunt/miniconda3/envs/prismaudio/bin/python"

_MAX_LOG_LINES = 200


@dataclass
class _PrismAudioJob:
    status: PrismAudioStatus = "idle"
    output_path: str | None = None
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = None  # type: ignore[type-arg]


class PrismAudioHandler:
    """Runs PrismAudio (ThinkSound) video-to-audio inference in a background thread."""

    def __init__(self, outputs_dir: Path) -> None:
        self._outputs_dir = outputs_dir
        self._lock = threading.Lock()
        self._job = _PrismAudioJob()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_progress(self) -> PrismAudioProgressResponse:
        with self._lock:
            j = self._job
            return PrismAudioProgressResponse(
                status=j.status,
                output_path=j.output_path,
                error=j.error,
                log_tail="\n".join(j.log_lines[-50:]),
            )

    def generate(self, req: PrismAudioGenerateRequest) -> PrismAudioProgressResponse:
        with self._lock:
            if self._job.status == "running":
                raise HTTPError(409, "PrismAudio generation already in progress")
            self._job = _PrismAudioJob(status="running")

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

    def _run(self, req: PrismAudioGenerateRequest) -> None:
        if _USE_WSL:
            self._run_wsl(req)
        else:
            self._run_windows(req)

    # ------------------------------------------------------------------
    # Windows-native path
    # ------------------------------------------------------------------

    def _run_windows(self, req: PrismAudioGenerateRequest) -> None:
        stem = f"prismaudio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        repo_dir = Path(_PRISMAUDIO_DIR_WIN)
        results_dir = Path(tempfile.mkdtemp(prefix="prismaudio_"))

        logger.info("[prismaudio/win] Starting: video=%s prompt=%r", req.video_path, req.prompt)

        try:
            # Step 1: Copy video to videos/demo.mp4 (expected by data_process.py)
            videos_dir = repo_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(req.video_path, str(videos_dir / "demo.mp4"))

            # Step 2: Get video duration for predict.py --duration-sec
            duration = _get_video_duration(req.video_path)

            # Step 3: Write cot_coarse/cot.csv with the text prompt
            cot_dir = repo_dir / "cot_coarse"
            cot_dir.mkdir(parents=True, exist_ok=True)
            safe_csv_prompt = req.prompt.replace('"', '""')
            (cot_dir / "cot.csv").write_text(
                f'id,caption_cot\ndemo,"{safe_csv_prompt}"\n', encoding="utf-8"
            )

            # Step 4: Feature extraction via torchrun
            if _PRISMAUDIO_PYTHON_WIN:
                tr_cmd: list[str] = [
                    _PRISMAUDIO_PYTHON_WIN, "-m", "torch.distributed.run", "--nproc_per_node=1",
                ]
            else:
                tr_cmd = [
                    "conda", "run", "-n", _PRISMAUDIO_CONDA_ENV, "--no-capture-output",
                    "torchrun", "--nproc_per_node=1",
                ]
            tr_cmd += ["data_utils/prismaudio_data_process.py", "--inference_mode", "True"]
            if not self._run_step(tr_cmd, str(repo_dir), "Feature extraction"):
                return

            # Step 5: Inference via predict.py
            if _PRISMAUDIO_PYTHON_WIN:
                pred_cmd: list[str] = [_PRISMAUDIO_PYTHON_WIN]
            else:
                pred_cmd = ["conda", "run", "-n", _PRISMAUDIO_CONDA_ENV, "--no-capture-output", "python"]
            pred_cmd += [
                "predict.py",
                "--model-config", "PrismAudio/configs/model_configs/prismaudio.json",
                "--duration-sec", str(duration),
                "--ckpt-dir", "ckpts/prismaudio.ckpt",
                "--results-dir", str(results_dir).replace("\\", "/"),
            ]
            if req.seed is not None:
                pred_cmd += ["--seed", str(req.seed)]
            if not self._run_step(pred_cmd, str(repo_dir), "Inference"):
                return

            # Step 6: Find WAV output
            wav_matches = list(results_dir.rglob("*.wav"))
            if not wav_matches:
                with self._lock:
                    self._job.status = "error"
                    self._job.error = "PrismAudio output WAV not found after generation"
                return

            # Step 7: Mix WAV + source video → MP4
            win_out = self._outputs_dir / f"{stem}.mp4"
            ffmpeg_res = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", req.video_path,
                    "-i", str(wav_matches[0]),
                    "-c:v", "copy", "-c:a", "aac",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-shortest", str(win_out),
                ],
                capture_output=True,
            )
            if ffmpeg_res.returncode != 0:
                with self._lock:
                    self._job.status = "error"
                    self._job.error = (
                        f"ffmpeg audio mix failed: "
                        f"{ffmpeg_res.stderr.decode('utf-8', errors='replace').strip()}"
                    )
                return

            with self._lock:
                self._job.status = "complete"
                self._job.output_path = str(win_out)
                self._job.log_lines.append(f"✓ Output: {win_out}")

        except Exception as exc:
            logger.exception("[prismaudio/win] Unexpected error")
            with self._lock:
                self._job.status = "error"
                self._job.error = str(exc)
        finally:
            shutil.rmtree(str(results_dir), ignore_errors=True)

    # ------------------------------------------------------------------
    # WSL path
    # ------------------------------------------------------------------

    def _run_wsl(self, req: PrismAudioGenerateRequest) -> None:
        stem = f"prismaudio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        # Feature extraction writes results/demo.npz to the repo dir by default.
        # predict.py reads from the same results/ dir — don't override with a custom path.
        wsl_repo_results = f"{_PRISMAUDIO_DIR_WSL}/results"
        wsl_video_path = _win_to_wsl_path(req.video_path)

        if _PRISMAUDIO_PYTHON_WSL:
            python_bin = _PRISMAUDIO_PYTHON_WSL
            # Use string split — Path() on Windows produces backslashes for Linux paths
            _wsl_bin_dir = _PRISMAUDIO_PYTHON_WSL.rsplit("/", 1)[0]
            torchrun_bin = f"{_wsl_bin_dir}/torchrun"
        else:
            python_bin = f"conda run -n {_PRISMAUDIO_CONDA_ENV_WSL} --no-capture-output python"
            torchrun_bin = f"conda run -n {_PRISMAUDIO_CONDA_ENV_WSL} --no-capture-output torchrun"

        # Escape single quotes in prompt for embedding in bash single-quoted strings
        safe_prompt = req.prompt.replace("'", "'\\''")
        seed_arg = f" --seed {req.seed}" if req.seed is not None else ""

        # Replicate demo.sh steps. Feature extraction writes results/demo.npz to the repo dir;
        # predict.py reads from the same results/ dir — both must share the repo's default path.
        inner = (
            f"set -e\n"
            f"cd '{_PRISMAUDIO_DIR_WSL}'\n"
            f"mkdir -p videos cot_coarse results\n"
            f"cp '{wsl_video_path}' videos/demo.mp4\n"
            f"DURATION=$(ffprobe -v error -show_entries format=duration"
            f" -of default=noprint_wrappers=1:nokey=1 videos/demo.mp4 2>/dev/null || echo 10)\n"
            f"printf 'id,caption_cot\\ndemo,\"{safe_prompt}\"\\n' > cot_coarse/cot.csv\n"
            f"echo '==> Feature extraction'\n"
            f"{torchrun_bin} --nproc_per_node=1"
            f" data_utils/prismaudio_data_process.py --inference_mode True\n"
            f"echo '==> Running inference'\n"
            f"{python_bin} predict.py"
            f" --model-config PrismAudio/configs/model_configs/prismaudio.json"
            f" --duration-sec \"$DURATION\""
            f" --ckpt-dir ckpts/prismaudio.ckpt"
            f"{seed_arg}\n"
        )

        cmd = ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c", inner]
        logger.info("[prismaudio/wsl] Starting: video=%s prompt=%r", req.video_path, req.prompt)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=PIPE, stderr=STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
            )
            with self._lock:
                self._job.process = process

            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(line.rstrip())

            process.wait()

            with self._lock:
                if self._job.status == "cancelled":
                    return
                if process.returncode != 0:
                    self._job.status = "error"
                    self._job.error = f"PrismAudio exited with code {process.returncode}"
                    return

            # Mix WAV + source video → MP4 inside WSL, then stream to Windows via cat.
            # Avoids WSL→Windows cp failures on /mnt/c/ paths.
            wsl_mixed = f"/tmp/prismaudio_mixed_{stem}.mp4"
            # Use find|xargs to avoid $() subshell issues when called from Windows Python.
            # WAV output lands in the repo's results/ dir (predict.py default).
            mix_bash = (
                f"find '{wsl_repo_results}' -name '*.wav' -type f | head -1 | "
                f"xargs -r -I WAV ffmpeg -y -i '{wsl_video_path}' -i WAV "
                f"-c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest '{wsl_mixed}' "
                f"&& echo OK || echo FAILED"
            )
            mix = subprocess.run(
                ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c", mix_bash],
                capture_output=True, text=True,
            )

            # Cleanup WSL repo working files regardless of mix result
            subprocess.run(
                ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c",
                 f"rm -rf '{wsl_repo_results}'"
                 f" '{_PRISMAUDIO_DIR_WSL}/videos/demo.mp4'"
                 f" '{_PRISMAUDIO_DIR_WSL}/cot_coarse/cot.csv'"],
                capture_output=True,
            )

            if "OK" not in mix.stdout:
                with self._lock:
                    self._job.status = "error"
                    self._job.error = (
                        f"PrismAudio WAV not found or ffmpeg mix failed. "
                        f"stderr: {mix.stderr.strip()}"
                    )
                return

            # Stream mixed MP4 from WSL → write on Windows side
            cat_result = subprocess.run(
                ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c",
                 f"cat '{wsl_mixed}'; rm -f '{wsl_mixed}'"],
                capture_output=True,
            )
            if not cat_result.stdout:
                with self._lock:
                    self._job.status = "error"
                    self._job.error = "PrismAudio mixed MP4 was empty after ffmpeg"
                return

            win_out = self._outputs_dir / f"{stem}.mp4"
            win_out.write_bytes(cat_result.stdout)

            with self._lock:
                self._job.status = "complete"
                self._job.output_path = str(win_out)
                self._job.log_lines.append(f"✓ Output: {win_out}")

        except Exception as exc:
            logger.exception("[prismaudio/wsl] Unexpected error")
            with self._lock:
                self._job.status = "error"
                self._job.error = str(exc)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _run_step(self, cmd: list[str], cwd: str, label: str) -> bool:
        """Run a subprocess step, stream logs. Returns True on success."""
        self._append_log(f"==> {label}")
        process = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=PIPE, stderr=STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        with self._lock:
            self._job.process = process
        assert process.stdout is not None
        for line in process.stdout:
            self._append_log(line.rstrip())
        process.wait()
        with self._lock:
            if self._job.status == "cancelled":
                return False
            if process.returncode != 0:
                self._job.status = "error"
                self._job.error = f"{label} failed (exit {process.returncode})"
                return False
        return True

    def _append_log(self, line: str) -> None:
        logger.info("[prismaudio] %s", line)
        with self._lock:
            self._job.log_lines.append(line)
            if len(self._job.log_lines) > _MAX_LOG_LINES:
                self._job.log_lines = self._job.log_lines[-_MAX_LOG_LINES:]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _win_to_wsl_path(win_path: str) -> str:
    """Convert C:\\Users\\... → /mnt/c/Users/..."""
    p = win_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        p = f"/mnt/{drive}{p[2:]}"
    return p


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe. Returns 10.0 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 10.0
