r"""MiniMax H3 mixed-int8 gate (R0 / R1 / X / Y / Z / YL), driven through the API
(MCP server -> app backend -> engine worker). Compares the product before and after two engine fixes:
  fix 1: the int8 ``adaln_proj`` gets its weight scale applied;
  fix 2: the VRAM 16GB mode qkv path also runs when qkv is a plain BF16 Linear.
R0 is run BEFORE the fixes, R1 / X / Y / Z / YL AFTER them. R0 and R1 are the same condition; R1's
video / audio stream MD5 must equal R0's (regression check).

Product code is never edited (read only). Copied from
``outputs/h3-fp8-w4a8-gate-2026-11-12/h3_quant_gate_api.py`` (temp yaml + temp state + backend on its own
port, MCP stdio client, argument-name check, load wait, job wait, save, stop in ``finally``, the
preflight). Differences from that script: the six conditions below; ``--only`` takes a comma list;
``--stop-on-fail`` (always on for X); a live run needs ``--live``; the dry-run makes no socket probe;
stream MD5 (``stream_md5`` copied from ``Docs/Outputs-archive/h3-keyref-2026-11-19/kr_single.py``) and
noise indicators recorded per condition; the block-swap ring line and the denoise VRAM peak read; the
git HEAD and ``git status --porcelain`` recorded; no upload folder (``hf-round*``) and no spike lookup;
home-folder paths are written as ``<user>``.

From the repo root (Git Bash, the APP venv; this script never imports torch):

    export PYTHONUTF8=1
    G=outputs/h3-mixed-int8-gate-2026-11-22/h3_mixed_int8_gate_api.py
    .venv/Scripts/python.exe -X utf8 $G --dry-run                 # plan only: nothing is started
    .venv/Scripts/python.exe -X utf8 $G --live --only R0          # BEFORE the fixes
    .venv/Scripts/python.exe -X utf8 $G --live                    # AFTER the fixes: R1, X, Y, Z, YL
    .venv/Scripts/python.exe -X utf8 $G --live --only X,Y         # a subset (canonical order kept)
    .venv/Scripts/python.exe -X utf8 $G --post-only               # checks, metrics, summary from runs/

Live flow:
  1. Preflight: refuse if 18620 accepts a connection / answers, if an engine worker is left over,
     or if nvidia-smi shows another python/uvicorn process or more than 3,072 MiB used.
  2. ``config.yaml`` is only read; a temp yaml (port, log_dir, output.dir, upload.dir, state_file ->
     ``_rt/``) and a temp state (MiniMax H3 active) are written; ``main.py --config <temp yaml>``
     on port 18699.
  3. MCP server over stdio (``.venv/Scripts/python.exe -m mcp_server``), ``tools/list`` argument check.
  4. Per condition, strictly sequential: ``unload_pipeline`` -> ``load_pipeline(base_model=MiniMaxH3,
     models={"transformer": <name from list_models>})`` -> ``submit_generate`` -> ``wait_for_job`` ->
     ``save_job_video`` -> copy ``metadata.json`` -> slice the server log and the worker log
     (``_rt/logs/h3_worker.log``) -> ``runs/<cond>/result.json`` -> post (self-checks, MD5, metrics).
     A failed self-check stops the remaining conditions when ``--stop-on-fail`` is given or the
     condition is X.
  5. Stop the backend (process tree) in ``finally``.
  6. Post again for every condition that ran (same as ``--post-only``), ``summary.md`` / ``summary.json``.

Where the load report comes from: the worker's ``build_report`` is logged as one line
``[h3_worker] LOAD_OK {json}`` (``engine_h3/worker.py`` ``_do_load``) into ``<log_dir>/h3_worker.log``.
The block-swap ring comes from ``engine/transformer/block_swap_prefetch.py``'s
``BlockSwap prefetch ready: ... arenas ring of N x M MB (held)`` line; the denoise VRAM peak from
``metadata.json`` ``h3.phases.denoise.peak_vram_mb`` / ``peak_vram_reserved_mb``.

Noise indicators are only a guide: the final pass / fail is decided by a person watching the clips.

Exit codes: 0 = every run completed and post-processing ran (pass/fail is recorded, not in the exit
code), 1 = stopped in preflight / setup, 2 = bad arguments, 3 = a job failed / was refused / timed out /
a run was stopped by --stop-on-fail.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # outputs/<this folder> -> repo root
PY = REPO / ".venv" / "Scripts" / "python.exe"
OUT_ROOT = HERE
RUNS = OUT_ROOT / "runs"
RT = OUT_ROOT / "_rt"
LOG_DIR = OUT_ROOT / "logs"
PORT_DEFAULT = 18699
OWNER_PORT = 18620
TEMP_CONFIG_NAME = "config.h3mixedint8gate.yaml"
TEMP_STATE_NAME = "state.h3mixedint8gate.json"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
SHEET_ALL = OUT_ROOT / "sheet_all.png"
GPU_BUSY_MIB = 3072  # same bound as h3_quant_gate_api.py

H3_BASE_MODEL = "MiniMaxH3"
H3_CATEGORIES = ("transformer", "text_encoder", "video_vae", "audio")
H3_FPS = 24
WIDTH, HEIGHT = 1280, 768
NUM_FRAMES = 73
SEED = 1234
ATTENTION = "sage"
VRAM16 = True
PREFETCH = True
KEEP_RESIDENT = False
#: Same string as h3_quant_gate_api.py (G1).
DEFAULT_PROMPT = (
    "A busy city street at dusk seen from the sidewalk. Pedestrians walk past the camera, their footsteps "
    "clicking on the pavement; cars and a bus drive by with engine hum and tire noise, and a car horn honks "
    "in the distance. Steady camera."
)

WEIGHTS = REPO / "models" / "MiniMaxH3" / "Weights"
LORA_DIR = REPO / "models" / "MiniMaxH3" / "LoRA"
TURBO_LORA_FILE = "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"  # G6's

FP8 = "float8_e4m3fn"       # h3_format.FP8
W4A8 = "asym_w4a8_int8"     # h3_format.W4A8
INT8 = "int8_tensorwise"    # h3_format.INT8

R_FILE = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"           # G1's file
X_FILE = "s3_raw_int8.safetensors"
Y_FILE = "Sulphur-3_Ref2VA-hybrid_mid_int8_convrot_keepB0-1-49.safetensors"
Z_FILE = "10Eros_Max_h3_8stepTURBO_beta7_int8HQ.safetensors"

#: shift: product default 12/3 (services/engines/h3/adapter.py SamplingRange defaults) = G1's;
#: Z uses the Eros shift of h3_quant_gate_api.py (G4/G5: 10/3); YL = G6 (4 steps, 12/3, Turbo LoRA 1.0).
#: expect_quant_layers: R from the header survey (200), X 250, Y 235, Z 180 (header, inspect_h3_dit).
CONDITIONS: dict[str, dict[str, Any]] = {
    "R0": {"file": R_FILE, "steps": 20, "shift": (12.0, 3.0), "lora": None, "expect_quant_layers": 200,
           "stop_on_fail": False, "purpose": "公式 int8 ConvRot（G1 と同じ）。修正の前に回す"},
    "R1": {"file": R_FILE, "steps": 20, "shift": (12.0, 3.0), "lora": None, "expect_quant_layers": 200,
           "stop_on_fail": False, "purpose": "R0 と同じ条件。修正の後に回し、R0 とストリームの MD5 を比べる"},
    "X": {"file": X_FILE, "steps": 20, "shift": (12.0, 3.0), "lora": None, "expect_quant_layers": 250,
          "stop_on_fail": True, "purpose": "s3_raw_int8（adaln_proj も int8。最初の関門）"},
    "Y": {"file": Y_FILE, "steps": 20, "shift": (12.0, 3.0), "lora": None, "expect_quant_layers": 235,
          "stop_on_fail": False, "purpose": "Sulphur-3 keepB0-1-49（ブロック 0・1・49 が BF16）"},
    "Z": {"file": Z_FILE, "steps": 8, "shift": (10.0, 3.0), "lora": None, "expect_quant_layers": 180,
          "stop_on_fail": False, "purpose": "10Eros 8stepTURBO int8HQ（ブロック 0・1・47〜49 が BF16）"},
    "YL": {"file": Y_FILE, "steps": 4, "shift": (12.0, 3.0), "lora": {"file": TURBO_LORA_FILE, "strength": 1.0},
           "expect_quant_layers": 235, "stop_on_fail": False,
           "purpose": "Y に Turbo LoRA（強さ 1.0・4 ステップ。G6 と同じ）"},
}
DEFAULT_ORDER = ("R1", "X", "Y", "Z", "YL")      # R0 only with an explicit --only
BASELINES = ("R1", "R0")                          # noise baseline: R1, else R0
NOISE_CHECKED = ("X", "Y", "Z")                   # YL: recorded only
NOISE_LIMITS = {"bytes_per_frame_ratio_max": 2.0, "laplacian_ratio_min": 0.5, "laplacian_ratio_max": 2.0,
                "frame_diff_ratio_max": 2.0, "audio_rms_dbfs_min": -50.0}
NOISE_NOTE = "目安は機械の指標で、合否ではない。最終の合否は人が動画を見て決める。"

TOOL_READ_TIMEOUT = timedelta(seconds=360)
WAIT_TIMEOUT_SEC = 45
WAIT_POLL_SEC = 5.0
LOAD_WAIT_SEC = 45
MAX_CONSECUTIVE_TRANSPORT_ERRORS = 12
SHEET_FRAMES = (0, 36, 72)
SHEET_TILE_LONG = 640

#: Argument names the script passes, checked against tools/list (live) and the tool functions (dry-run).
REQUIRED_TOOL_ARGS: dict[str, set[str]] = {
    "backend_status": set(),
    "list_models": set(),
    "list_loras": set(),
    "load_pipeline": {"base_model", "models", "wait_sec"},
    "unload_pipeline": set(),
    "submit_generate": {"prompt", "width", "height", "num_frames", "frame_rate", "seed", "attention_backend",
                        "num_inference_steps", "shift_video", "shift_audio", "vram16_mode",
                        "block_swap_prefetch", "keep_resident", "loras"},
    "wait_for_job": {"job_id", "timeout_sec", "poll_interval_sec"},
    "save_job_video": {"job_id", "dest_dir", "filename", "no_clobber", "which"},
    "get_job_video_path": {"job_id"},
}

STEP_RE = re.compile(r"denoise (\d+)/(\d+) steps \([^)]*?([\d.]+)s elapsed\)")
LOAD_OK_RE = re.compile(r"^\[h3_worker\] LOAD_OK (\{.*\})\s*$", re.M)
FP8_LINE_RE = re.compile(r"^\[h3_worker\] fp8: (\d+) layer\(s\), path '(\w+)'.*$", re.M)
#: engine/transformer/block_swap_prefetch.py (install(): "BlockSwap prefetch ready: ...").
RING_READY_RE = re.compile(r"^.*BlockSwap prefetch ready: .*$", re.M)
RING_RE = re.compile(r"ring of (\d+) x ([\d.]+) MB")
STAGING_RE = re.compile(r"pinned staging pool = (\d+) x ([\d.]+) MB")
INT8_LINE_RE = re.compile(r"^\[h3_worker\] int8: (.*)$", re.M)
WORKER_RE = re.compile(r"-m\s+engine\w*\.worker|engine\w*[\\/]worker\.py")


# ---------------------------------------------------------------------------------------------
# Home-folder scrubbing (absolute paths under the home folder are written as <user>)
# ---------------------------------------------------------------------------------------------

def _home_variants() -> list[str]:
    v: set[str] = set()
    for h in (os.path.expanduser("~"), os.environ.get("USERPROFILE"), os.environ.get("HOME")):
        if h and len(h) >= 6:
            v.add(h.rstrip("\\/"))
    if os.name == "nt":
        try:
            import ctypes  # noqa: PLC0415

            for h in list(v):
                buf = ctypes.create_unicode_buffer(1024)
                if ctypes.windll.kernel32.GetShortPathNameW(h, buf, 1024):
                    v.add(buf.value.rstrip("\\/"))
        except Exception:  # noqa: BLE001
            pass
    out: set[str] = set()
    for h in v:
        out |= {h, h.replace("\\", "/"), h.replace("\\", "\\\\")}
    return sorted(out, key=len, reverse=True)


_HOME_RE = re.compile("|".join(re.escape(h) for h in _home_variants()) or r"(?!x)x", re.I)


def scrub(obj: Any) -> Any:
    if isinstance(obj, str):
        return _HOME_RE.sub("<user>", obj)
    if isinstance(obj, dict):
        return {scrub(k) if isinstance(k, str) else k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(x) for x in obj]
    return obj


def scrub_files() -> list[str]:
    """Rewrite the raw logs written by other processes (backend console, MCP stderr, the temp config,
    backend metadata) inside this folder with home-folder paths replaced."""
    changed: list[str] = []
    for base in (LOG_DIR, RT):
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in (".log", ".txt", ".json", ".yaml", ".yml"):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            new = _HOME_RE.sub("<user>", text)
            if new != text:
                p.write_text(new, encoding="utf-8", newline="\n")
                changed.append(rel(p))
    return changed


def rel(p: Path | str) -> str:
    """Path relative to the repo when it is inside it, otherwise scrubbed."""
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except (ValueError, OSError):
        return scrub(str(p))


# ---------------------------------------------------------------------------------------------
# Small helpers (copied from h3_quant_gate_api.py)
# ---------------------------------------------------------------------------------------------

class Log:
    """Same line to stdout and to this invocation's log file."""

    def __init__(self) -> None:
        self.file = None
        self.path: Path | None = None

    def open(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = open(path, "a", encoding="utf-8", newline="\n")

    def __call__(self, msg: str) -> None:
        line = scrub(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        print(line, flush=True)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


log = Log()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(scrub(json.loads(json.dumps(obj, default=str))), ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scrub(text.replace("\r\n", "\n").replace("\r", "\n")), encoding="utf-8", newline="\n")


def tool(name: str) -> str:
    local = REPO / "tools" / "ffmpeg" / "bin" / f"{name}.exe"
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found (<repo>/tools/ffmpeg/bin or PATH)")
    return found


def run_cmd(cmd: list[str], timeout: float | None = None, cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                          cwd=cwd)


def ffprobe(path: Path) -> dict:
    """Video (frame count measured with -count_frames) and audio essentials."""
    exe = tool("ffprobe")
    v = run_cmd([exe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                 "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration",
                 "-of", "json", str(path)])
    a = run_cmd([exe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                 "stream=codec_name,sample_rate,channels,duration", "-of", "json", str(path)])
    vs = (json.loads(v.stdout or "{}").get("streams") or [{}])[0]
    as_ = (json.loads(a.stdout or "{}").get("streams") or [None])[0]
    nb = vs.get("nb_read_frames")
    fps = None
    if vs.get("r_frame_rate"):
        num, _, den = str(vs["r_frame_rate"]).partition("/")
        try:
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            fps = None
    return {"path": rel(path), "video": {**vs, "fps": fps, "nb_read_frames": int(nb) if nb not in (None, "N/A") else None},
            "audio": as_}


def file_offset(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def read_since(p: Path, offset: int) -> str:
    if not p.exists():
        return ""
    with p.open("rb") as fh:
        fh.seek(offset)
        return fh.read().decode("utf-8", errors="replace")


def worker_log_path() -> Path:
    return RT / "logs" / "h3_worker.log"


def server_log_path() -> Path:
    return RT / "logs" / "server.log"


def _p(x, nd: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "合格" if x else "不合格"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def git_info() -> dict:
    """HEAD and ``git status --porcelain`` of the repo (read-only git commands)."""
    out: dict[str, Any] = {}
    try:
        out["head"] = run_cmd(["git", "rev-parse", "HEAD"], timeout=60, cwd=str(REPO)).stdout.strip() or None
        out["branch"] = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=60, cwd=str(REPO)).stdout.strip() or None
        st = run_cmd(["git", "status", "--porcelain"], timeout=120, cwd=str(REPO))
        out["status_porcelain"] = [ln for ln in (st.stdout or "").splitlines() if ln.strip()]
        out["status_returncode"] = st.returncode
        out["dirty"] = bool(out["status_porcelain"])
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------------------------
# Stream MD5 and noise indicators (no GPU; ffmpeg + numpy)
# ---------------------------------------------------------------------------------------------

def stream_md5(mp4: Path, which: str) -> str | None:
    """Copied from kr_single.py. MD5 of one stream's packets, the container left out:
    ``ffmpeg -v error -i <mp4> -map 0:<v|a>:0 -c copy -f md5 -`` -> ``MD5=<hex>``. None = no such stream."""
    exe = tool("ffmpeg")
    proc = subprocess.run([exe, "-v", "error", "-i", str(mp4), "-map", f"0:{which}:0", "-c", "copy", "-f", "md5", "-"],
                          capture_output=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        if "matches no streams" in (proc.stderr or ""):
            return None
        raise RuntimeError(f"ffmpeg md5 ({which}) failed: {proc.stderr.strip()}")
    line = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    return line[0].split("=", 1)[1] if line[0].startswith("MD5=") else None


def stream_md5s(mp4: Path) -> dict:
    return {"video": stream_md5(mp4, "v"), "audio": stream_md5(mp4, "a")}


def _decode_raw(args: list[str]) -> bytes | None:
    proc = subprocess.run([tool("ffmpeg"), "-v", "error", *args], capture_output=True)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")
        if "matches no streams" in err:
            return None
        raise RuntimeError(f"ffmpeg decode failed: {err.strip()[-500:]}")
    return proc.stdout


def noise_metrics(mp4: Path) -> dict:
    """Guides for noise: bytes per frame of the video stream, mean |Laplacian| of the luma on 5 frames,
    mean |difference| of adjacent frames (luma), audio RMS in dBFS."""
    import numpy as np  # noqa: PLC0415 -- app venv

    exe = tool("ffprobe")
    pk = run_cmd([exe, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=size",
                  "-of", "csv=p=0", str(mp4)])
    sizes = [int(t) for t in re.split(r"[\s,]+", pk.stdout or "") if t.isdigit()]
    st = run_cmd([exe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                  "-of", "json", str(mp4)])
    vs = (json.loads(st.stdout or "{}").get("streams") or [{}])[0]
    w, h = int(vs.get("width") or 0), int(vs.get("height") or 0)
    out: dict[str, Any] = {"video_packets": len(sizes), "video_stream_bytes": sum(sizes),
                           "bytes_per_frame": round(sum(sizes) / len(sizes), 1) if sizes else None}
    raw = _decode_raw(["-i", str(mp4), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    if raw and w and h:
        n = len(raw) // (w * h)
        frames = np.frombuffer(raw[: n * w * h], dtype=np.uint8).reshape(n, h, w)
        idx = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
        laps = []
        for i in idx:
            f = frames[i].astype(np.float32)
            lap = (f[1:-1, :-2] + f[1:-1, 2:] + f[:-2, 1:-1] + f[2:, 1:-1] - 4.0 * f[1:-1, 1:-1])
            laps.append(float(np.abs(lap).mean()))
        diffs = [float(np.abs(frames[i + 1].astype(np.int16) - frames[i].astype(np.int16)).mean())
                 for i in range(n - 1)]
        out.update(decoded_frames=n, laplacian_frames=idx, laplacian_per_frame=[round(x, 3) for x in laps],
                   laplacian_mean=round(float(np.mean(laps)), 3),
                   frame_diff_mean=round(float(np.mean(diffs)), 3) if diffs else None,
                   frame_diff_max=round(float(np.max(diffs)), 3) if diffs else None)
    else:
        out.update(decoded_frames=0, laplacian_mean=None, frame_diff_mean=None)
    araw = _decode_raw(["-i", str(mp4), "-map", "0:a:0", "-f", "f32le", "-"])
    if araw:
        x = np.frombuffer(araw[: len(araw) // 4 * 4], dtype=np.float32).astype(np.float64)
        rms = float(math.sqrt(float(np.mean(x * x)))) if x.size else 0.0
        out.update(audio_samples=int(x.size), audio_rms=rms,
                   audio_rms_dbfs=round(20.0 * math.log10(rms), 2) if rms > 0 else None,
                   audio_silent=rms == 0.0)
    else:
        out.update(audio_samples=0, audio_rms=None, audio_rms_dbfs=None, audio_silent=None)
    return out


def noise_flags(cond: str, noise: dict, baseline: tuple[str, dict] | None) -> dict:
    """X / Y / Z against the baseline (R1, else R0): out of range -> 要確認. A guide only."""
    rec: dict[str, Any] = {"checked": cond in NOISE_CHECKED, "baseline": baseline[0] if baseline else None,
                           "limits": NOISE_LIMITS, "ratios": {}, "flags": [], "note": NOISE_NOTE}
    if cond not in NOISE_CHECKED:
        return rec
    lim = NOISE_LIMITS
    if baseline:
        b = baseline[1]

        def ratio(k: str) -> float | None:
            a, z = noise.get(k), b.get(k)
            return round(a / z, 3) if isinstance(a, (int, float)) and isinstance(z, (int, float)) and z else None

        r = {"bytes_per_frame": ratio("bytes_per_frame"), "laplacian_mean": ratio("laplacian_mean"),
             "frame_diff_mean": ratio("frame_diff_mean")}
        rec["ratios"] = r
        if r["bytes_per_frame"] is None or r["bytes_per_frame"] > lim["bytes_per_frame_ratio_max"]:
            rec["flags"].append(f"1 フレームあたりのバイト数が基準の {_p(r['bytes_per_frame'], 2)} 倍（目安 2 倍以下）")
        if r["laplacian_mean"] is None or not (lim["laplacian_ratio_min"] <= r["laplacian_mean"] <= lim["laplacian_ratio_max"]):
            rec["flags"].append(f"ラプラシアンの平均が基準の {_p(r['laplacian_mean'], 2)} 倍（目安 0.5〜2 倍）")
        if r["frame_diff_mean"] is None or r["frame_diff_mean"] > lim["frame_diff_ratio_max"]:
            rec["flags"].append(f"フレーム差が基準の {_p(r['frame_diff_mean'], 2)} 倍（目安 2 倍以下）")
    else:
        rec["flags"].append("基準（R1・R0）の記録が無いため比率を出せない")
    db = noise.get("audio_rms_dbfs")
    if db is None or db <= lim["audio_rms_dbfs_min"]:
        rec["flags"].append(f"音声の RMS が {_p(db, 1)} dBFS（目安 −50dBFS より大きい）")
    rec["needs_review"] = bool(rec["flags"])
    return rec


# ---------------------------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------------------------

def cond_list(only: str | None) -> list[str]:
    if not only:
        return list(DEFAULT_ORDER)
    want = {x.strip() for x in only.split(",") if x.strip()}
    return [c for c in CONDITIONS if c in want]   # canonical order (R0, R1, X, Y, Z, YL)


def run_dir(cond: str) -> Path:
    return RUNS / cond


def expected_formats(path: Path) -> dict:
    """Per-format layer counts from the file header (``h3_format.inspect_h3_dit``: header only, torch-free)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        import h3_format  # noqa: PLC0415

        _header, cfg = h3_format.inspect_h3_dit(str(path))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    by_fmt: dict[str, int] = {}
    for q in cfg.quant.values():
        by_fmt[q.format] = by_fmt.get(q.format, 0) + 1
    specs = sorted({f"{q.format}/convrot={q.convrot}/g{q.convrot_groupsize}"
                    + ("" if q.group_size is None else f"/G{q.group_size}") for q in cfg.quant.values()})
    return {"ok": True, "quant_layers": len(cfg.quant), "quant_layers_by_format": dict(sorted(by_fmt.items())),
            "quant_specs": specs, "unknown": len(cfg.unknown)}


def build_job(cond: str, prompt: str) -> dict:
    c = CONDITIONS[cond]
    sv, sa = c["shift"]
    tool_args: dict[str, Any] = {
        "prompt": prompt, "width": WIDTH, "height": HEIGHT, "num_frames": NUM_FRAMES, "frame_rate": float(H3_FPS),
        "seed": SEED, "attention_backend": ATTENTION, "num_inference_steps": c["steps"], "shift_video": sv,
        "shift_audio": sa, "vram16_mode": VRAM16, "block_swap_prefetch": PREFETCH, "keep_resident": KEEP_RESIDENT,
    }
    if c["lora"]:
        # the name is resolved from list_loras at run time (stem of the file, lora_registry.py)
        tool_args["loras"] = [{"name": Path(c["lora"]["file"]).stem, "strength": c["lora"]["strength"]}]
    return {"cond": cond, "purpose": c["purpose"], "file": c["file"], "path": rel(WEIGHTS / c["file"]),
            "steps": c["steps"], "shift": [sv, sa], "lora": c["lora"], "expect_quant_layers": c["expect_quant_layers"],
            "stop_on_fail": c["stop_on_fail"], "tool_args": tool_args}


# ---------------------------------------------------------------------------------------------
# Preflight (owner server, GPU, leftover workers) -- copied
# ---------------------------------------------------------------------------------------------

def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def http(port: int, method: str, path: str, body=None, timeout: float = 60.0):
    url = base_url(port) + "/api/v1" + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except Exception:  # noqa: BLE001
            return exc.code, {"raw": raw}


def port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def all_processes() -> list[dict]:
    ps = ("Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
          "ConvertTo-Json -Depth 3")
    r = run_cmd(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=120)
    try:
        data = json.loads(r.stdout or "null")
    except json.JSONDecodeError:
        return []
    if data is None:
        return []
    if isinstance(data, dict):
        data = [data]
    return [{"pid": d.get("ProcessId"), "ppid": d.get("ParentProcessId"), "name": d.get("Name"),
             "cmdline": d.get("CommandLine") or ""} for d in data]


def descendants(root_pid: int, procs: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen = {root_pid}
    frontier = {root_pid}
    while frontier:
        children = [p for p in procs if p["ppid"] in frontier and p["pid"] not in seen]
        for p in children:
            seen.add(p["pid"])
            out.append(p)
        frontier = {p["pid"] for p in children}
    return out


def gpu_busy_check() -> dict:
    """No other python / uvicorn process on the GPU, and the GPU's used memory below a desktop-only bound."""
    rec: dict[str, Any] = {"blocking": []}
    exe = shutil.which("nvidia-smi")
    if not exe:
        rec["blocking"].append("nvidia-smi が見つからない（GPU が空いていることを確かめられない）")
        return rec
    proc = run_cmd([exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], timeout=60)
    apps = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    mine = {str(os.getpid())}
    others = [a for a in apps if a.split(",", 1)[0].strip() not in mine
              and any(w in a.lower() for w in ("python", "uvicorn"))]
    mem = run_cmd([exe, "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"], timeout=60)
    parts = [x.strip() for x in (mem.stdout or ",0,0").splitlines()[0].split(",")] if mem.stdout else ["", "0", "0"]
    try:
        used = int(parts[1])
    except (IndexError, ValueError):
        used = -1
    rec.update(nvidia_smi_compute_apps=apps, returncode=proc.returncode, gpu=parts, memory_used_mib=used,
               python_like_others=others)
    if proc.returncode != 0 or mem.returncode != 0 or used < 0:
        rec["blocking"].append(f"nvidia-smi が失敗（{proc.returncode}/{mem.returncode}）: {(proc.stderr or '')[-300:]}")
    if others:
        rec["blocking"].append(f"GPU を別の python／uvicorn が使っている: {others}")
    if used > GPU_BUSY_MIB:
        rec["blocking"].append(f"GPU の使用量 {used} MiB が {GPU_BUSY_MIB} MiB を超えている")
    return rec


def preflight_checks() -> dict:
    """``blocking`` non-empty -> the live run stops."""
    rec: dict[str, Any] = {"at": now_iso(), "blocking": []}
    listening = port_listening(OWNER_PORT)
    rec["owner_server"] = {"port": OWNER_PORT, "listening": listening}
    if listening:
        rec["blocking"].append(f"ポート {OWNER_PORT} で何かが待ち受けている（オーナーのサーバー）")
        try:
            status, body = http(OWNER_PORT, "GET", "/status", timeout=3)
            rec["owner_server"].update(http=status, body=body if isinstance(body, dict) else str(body)[:300])
        except Exception as exc:  # noqa: BLE001
            rec["owner_server"]["error"] = f"{type(exc).__name__}: {exc}"
    gpu = gpu_busy_check()
    rec["gpu"] = gpu
    rec["blocking"] += gpu["blocking"]
    procs = all_processes()
    workers = [{"pid": p["pid"], "ppid": p["ppid"], "cmdline": p["cmdline"][:300]}
               for p in procs if WORKER_RE.search(p["cmdline"] or "")]
    rec["engine_workers"] = workers
    if workers:
        rec["blocking"].append(f"ワーカーのプロセスが残っている（{[w['pid'] for w in workers]}）")
    return rec


def log_preflight(rec: dict) -> None:
    log(f"preflight: port {OWNER_PORT} {'LISTENING' if rec['owner_server']['listening'] else 'free'}")
    g = rec["gpu"]
    log(f"preflight: GPU {g.get('gpu')} used {g.get('memory_used_mib')} MiB (bound {GPU_BUSY_MIB}); "
        f"python/uvicorn on GPU: {g.get('python_like_others') or 'none'}")
    log(f"preflight: leftover workers {rec['engine_workers'] or 'none'}")


# ---------------------------------------------------------------------------------------------
# Temp yaml and the backend (copied; paths under this folder's _rt/)
# ---------------------------------------------------------------------------------------------

def write_temp_config(port: int) -> tuple[Path, Path]:
    """``config.yaml`` is read only. Five keys point at ``_rt``; a temp state selects H3."""
    import yaml  # noqa: PLC0415 -- app venv dependency (config.py uses it)

    RT.mkdir(parents=True, exist_ok=True)
    for sub in ("logs", "outputs", "uploads"):
        (RT / sub).mkdir(exist_ok=True)
    raw = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = port
    raw["server"]["log_dir"] = str(RT / "logs")
    raw.setdefault("output", {})["dir"] = str(RT / "outputs")
    raw.setdefault("upload", {})["dir"] = str(RT / "uploads")
    raw["state_file"] = str(RT / TEMP_STATE_NAME)
    cfg = RT / TEMP_CONFIG_NAME
    cfg.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    selections: dict[str, dict[str, str]] = {}
    try:
        current = json.loads((REPO / "state.json").read_text(encoding="utf-8"))
        if isinstance(current.get("selections"), dict):
            selections = {k: dict(v) for k, v in current["selections"].items() if isinstance(v, dict)}
    except Exception:  # noqa: BLE001 -- state.json is a cache; defaults are fine
        pass
    for base in ("LTX23", "LTX25"):
        selections.setdefault(base, {c: "default" for c in H3_CATEGORIES})
    selections[H3_BASE_MODEL] = {c: "default" for c in H3_CATEGORIES}
    state = RT / TEMP_STATE_NAME
    state.write_text(json.dumps({"schema": 1, "active_base_model": H3_BASE_MODEL, "selections": selections},
                                indent=2), encoding="utf-8", newline="\n")
    return cfg, state


def server_cmd() -> list[str]:
    return [str(PY), "-u", str(REPO / "main.py"), "--config", str(RT / TEMP_CONFIG_NAME)]


class Server:
    def __init__(self, port: int) -> None:
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.log_path = RT / "server.console.log"
        self._fh = None
        self.record: dict[str, Any] = {"port": port}

    def alive(self) -> bool:
        try:
            status, _ = http(self.port, "GET", "/status", timeout=5)
            return status == 200
        except Exception:  # noqa: BLE001
            return False

    def start(self) -> None:
        if self.alive() or port_listening(self.port):
            raise RuntimeError(f"port {self.port} is already in use; pick another with --port")
        cfg, state = write_temp_config(self.port)
        env = dict(os.environ)
        prefix = ""
        for d in (REPO / "tools" / "uv", REPO / "tools" / "ffmpeg" / "bin"):
            if d.exists():
                prefix += f"{d};"
        env["PATH"] = prefix + env.get("PATH", "")
        env["UV_PYTHON_INSTALL_DIR"] = str(REPO / ".python")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        self._fh = self.log_path.open("a", encoding="utf-8", errors="replace", newline="\n")
        self._fh.write(f"\n===== {now_iso()} h3_mixed_int8_gate_api.py: main.py --config {rel(cfg)} =====\n")
        self._fh.flush()
        cmd = server_cmd()
        self.proc = subprocess.Popen(cmd, cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=self._fh,
                                     stderr=subprocess.STDOUT, env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.record.update(pid=self.proc.pid, cmd=[rel(x) for x in cmd], config=rel(cfg), state=rel(state),
                           log=rel(self.log_path), started_at=now_iso())
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.alive():
                log(f"backend up (pid {self.proc.pid}, port {self.port})")
                return
            if self.proc.poll() is not None:
                raise RuntimeError("backend exited while starting:\n"
                                   + self.log_path.read_text(encoding="utf-8", errors="replace")[-3000:])
            time.sleep(1.0)
        raise RuntimeError("/api/v1/status did not answer within 180 s")

    def stop(self) -> None:
        """unload -> terminate -> taskkill /T /F if needed; then kill any descendant (worker) left."""
        if self.proc is None:
            return
        rec: dict[str, Any] = {"at": now_iso()}
        try:
            rec["unload_http"] = http(self.port, "POST", "/pipeline/unload", timeout=600)[0]
        except Exception as exc:  # noqa: BLE001
            rec["unload_http"] = f"{type(exc).__name__}: {exc}"
        pid = self.proc.pid
        kids = descendants(pid, all_processes())
        rec["descendants_before"] = [{"pid": p["pid"], "cmdline": p["cmdline"][:300]} for p in kids]
        try:
            self.proc.terminate()
            self.proc.wait(timeout=60)
            rec["terminate"] = "ok"
        except Exception as exc:  # noqa: BLE001
            rec["terminate"] = f"{type(exc).__name__}: {exc}"
            rec["taskkill"] = run_cmd(["taskkill", "/PID", str(pid), "/T", "/F"]).stdout.strip()
        self.proc = None
        alive_pids = {p["pid"] for p in all_processes()}
        left = [p for p in kids if p["pid"] in alive_pids]
        for p in left:
            run_cmd(["taskkill", "/PID", str(p["pid"]), "/T", "/F"])
        rec["descendants_killed_after"] = [p["pid"] for p in left]
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._fh = None
        self.record["stop"] = rec
        log(f"backend stopped (pid {pid}; descendants killed afterwards {rec['descendants_killed_after']})")


# ---------------------------------------------------------------------------------------------
# MCP (copied)
# ---------------------------------------------------------------------------------------------

class ToolCallError(Exception):
    def __init__(self, tool_name: str, text: str) -> None:
        super().__init__(f"{tool_name}: {text}")
        self.tool = tool_name
        self.text = text


class Mcp:
    def __init__(self, session, calls_log: list) -> None:
        self.session = session
        self.calls_log = calls_log

    async def call(self, tool_name: str, args: dict[str, Any] | None = None) -> dict:
        t0 = time.monotonic()
        entry: dict[str, Any] = {"at": now_iso(), "tool": tool_name, "args": args or {}}
        try:
            result = await self.session.call_tool(tool_name, args or {}, read_timeout_seconds=TOOL_READ_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            entry.update(ok=False, seconds=round(time.monotonic() - t0, 2),
                         error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            self.calls_log.append(entry)
            raise ToolCallError(tool_name, entry["error"]) from exc
        entry["seconds"] = round(time.monotonic() - t0, 2)
        text = "\n".join(getattr(c, "text", "") for c in (result.content or []) if getattr(c, "text", None))
        if result.isError:
            entry.update(ok=False, error=text)
            self.calls_log.append(entry)
            raise ToolCallError(tool_name, text)
        if isinstance(result.structuredContent, dict):
            data = result.structuredContent
        else:
            try:
                data = json.loads(text) if text else {}
            except json.JSONDecodeError:
                data = {"_text": text}
        entry["ok"] = True
        if tool_name not in ("wait_for_job", "backend_status", "list_models", "list_loras"):
            entry["result"] = data
        self.calls_log.append(entry)
        return data


def check_tool_schemas(tools: list) -> dict:
    by_name = {t.name: t for t in tools}
    report: dict[str, Any] = {"tool_count": len(tools), "problems": []}
    for name, needed in REQUIRED_TOOL_ARGS.items():
        t = by_name.get(name)
        if t is None:
            report["problems"].append(f"tool {name} missing from tools/list")
            continue
        props = set((t.inputSchema or {}).get("properties", {}))
        missing = needed - props
        if missing:
            report["problems"].append(f"{name}: inputSchema lacks {sorted(missing)}")
    report["ok"] = not report["problems"]
    return report


def check_tool_signatures_static() -> dict:
    """Dry-run counterpart of :func:`check_tool_schemas` without starting the MCP server: parse
    ``mcp_server/tools/*.py`` and compare the registered functions' parameters with REQUIRED_TOOL_ARGS."""
    funcs: dict[str, set[str]] = {}
    registered: set[str] = set()
    for f in sorted((REPO / "mcp_server" / "tools").glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.col_offset == 0:
                a = node.args
                funcs[node.name] = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Call)
                    and isinstance(node.func.func, ast.Attribute) and node.func.func.attr == "tool"
                    and node.args and isinstance(node.args[0], ast.Name)):
                registered.add(node.args[0].id)
    problems = []
    for name, needed in REQUIRED_TOOL_ARGS.items():
        if name not in registered:
            problems.append(f"{name} is not registered with mcp.tool() in mcp_server/tools/")
            continue
        missing = needed - funcs.get(name, set())
        if missing:
            problems.append(f"{name}() has no parameter(s) {sorted(missing)}")
    return {"ok": not problems, "problems": problems, "registered_count": len(registered)}


def mcp_env(port: int) -> dict:
    return {"PYTHONUTF8": "1", "LTX_MCP_BASE_URL": base_url(port), "LTX_MCP_OUTPUT_DIR": str(RT / "outputs"),
            "LTX_MCP_CONFIG": str(RT / TEMP_CONFIG_NAME)}


def mcp_params(port: int):
    from mcp import StdioServerParameters  # noqa: PLC0415 -- app venv dependency

    env = {**os.environ, **mcp_env(port)}
    return StdioServerParameters(command=str(PY), args=["-m", "mcp_server"], cwd=str(REPO), env=env, encoding="utf-8")


async def wait_loaded(mcp: Mcp, load: dict, timeout_sec: float) -> bool:
    """Poll backend_status every 10 s until H3 is loaded."""
    t0 = time.monotonic()
    while True:
        st = await mcp.call("backend_status")
        load["status_polls"] = load.get("status_polls", 0) + 1
        s = st.get("status") or {}
        load["last_status"] = s
        if not st.get("reachable"):
            load["error"] = f"backend unreachable: {st.get('error')}"
            return False
        if s.get("pipeline_loaded") and s.get("base_model") == H3_BASE_MODEL and s.get("state") in ("ready", "running"):
            return True
        if s.get("state") == "error":
            load["error"] = f"pipeline state is error: {s}"
            return False
        if time.monotonic() - t0 > timeout_sec:
            load["error"] = f"load not finished after {timeout_sec} s (last status {s})"
            return False
        await asyncio.sleep(10)


async def load_condition(mcp: Mcp, args, job: dict, model_name: str) -> dict:
    """unload -> load_pipeline(base_model, models={"transformer": name}) -> wait until ready."""
    load: dict[str, Any] = {"requested": {"base_model": H3_BASE_MODEL, "models": {"transformer": model_name},
                                          "wait_sec": LOAD_WAIT_SEC}, "responses": []}
    try:
        load["unload_before"] = await mcp.call("unload_pipeline")
    except ToolCallError as exc:
        load["unload_before"] = {"error": exc.text}
    t0 = time.monotonic()
    try:
        resp = await mcp.call("load_pipeline", load["requested"])
        load["responses"].append(resp)
        log(f"  load_pipeline: finished={resp.get('finished')} pipeline_loaded={resp.get('pipeline_loaded')}")
    except ToolCallError as exc:
        load["responses"].append({"error": exc.text})
        if "PIPELINE_LOADING" not in exc.text:
            load["error"] = exc.text
            load["seconds"] = round(time.monotonic() - t0, 2)
            return load
    ok = await wait_loaded(mcp, load, args.load_timeout_sec)
    load["seconds"] = round(time.monotonic() - t0, 2)
    load["ok"] = ok
    return load


async def wait_job(mcp: Mcp, job_id: str, timeout_sec: float) -> dict:
    t0 = time.monotonic()
    calls = 0
    errors_in_row = 0
    samples: list[dict] = []
    last_key = None
    last: dict = {}
    while True:
        try:
            st = await mcp.call("wait_for_job", {"job_id": job_id, "timeout_sec": WAIT_TIMEOUT_SEC,
                                                 "poll_interval_sec": WAIT_POLL_SEC})
            errors_in_row = 0
        except ToolCallError as exc:
            errors_in_row += 1
            samples.append({"t": round(time.monotonic() - t0, 1), "error": exc.text})
            if errors_in_row >= MAX_CONSECUTIVE_TRANSPORT_ERRORS:
                return {"final": last, "calls": calls, "seconds": round(time.monotonic() - t0, 1),
                        "samples": samples, "gave_up": f"wait_for_job failed {errors_in_row} times in a row: {exc.text}"}
            await asyncio.sleep(5)
            continue
        calls += 1
        last = st
        key = (st.get("status"), st.get("stage"), st.get("current_step"))
        if key != last_key:
            samples.append({"t": round(time.monotonic() - t0, 1), "status": st.get("status"),
                            "progress": st.get("progress"), "stage": st.get("stage"),
                            "current_step": st.get("current_step")})
            log(f"  job {job_id[:8]} status={st.get('status')} progress={st.get('progress')} stage={st.get('stage')} "
                f"step={st.get('current_step')}")
            last_key = key
        if not st.get("timed_out"):
            return {"final": st, "calls": calls, "seconds": round(time.monotonic() - t0, 1), "samples": samples}
        if time.monotonic() - t0 > timeout_sec:
            return {"final": st, "calls": calls, "seconds": round(time.monotonic() - t0, 1), "samples": samples,
                    "gave_up": f"not finished after {timeout_sec} s (the job keeps running in the backend)"}


# ---------------------------------------------------------------------------------------------
# Readers of the logs and the metadata
# ---------------------------------------------------------------------------------------------

def parse_load_report(worker_text: str) -> dict:
    """The last ``LOAD_OK {json}`` line, the last ``fp8:`` line, the block-swap ring / staging lines and
    the ``int8:`` lines (qkv sharing) of this condition's slice of the worker log."""
    text = worker_text or ""
    out: dict[str, Any] = {"found": False}
    hits = LOAD_OK_RE.findall(text)
    if hits:
        try:
            report = json.loads(hits[-1])
            out.update(found=True, load_ok_lines=len(hits), build_report=report)
        except json.JSONDecodeError as exc:
            out["error"] = f"LOAD_OK is not JSON: {exc}"
    fp8 = FP8_LINE_RE.findall(text)
    if fp8:
        out["fp8_log_line"] = {"layers": int(fp8[-1][0]), "path": fp8[-1][1]}
    ready = RING_READY_RE.findall(text)
    if ready:
        line = ready[-1].strip()
        m = RING_RE.search(line)
        out["block_swap_ring"] = {"line": line, "found": True,
                                  "slots": int(m.group(1)) if m else None, "slot_mb": float(m.group(2)) if m else None,
                                  "total_mb": round(int(m.group(1)) * float(m.group(2)), 1) if m else None,
                                  "ring_off": "ring off" in line}
    else:
        out["block_swap_ring"] = {"found": False}
    stg = STAGING_RE.findall(text)
    if stg:
        out["staging_pool"] = {"slots": int(stg[-1][0]), "slot_mb": float(stg[-1][1])}
    out["int8_lines"] = INT8_LINE_RE.findall(text)
    return out


def parse_step_times(server_text: str, steps: int) -> dict:
    """s/step from the server log's ``denoise k/N steps (..., Ts elapsed)`` lines."""
    el: dict[int, float] = {}
    for k, n, t in STEP_RE.findall(server_text or ""):
        if int(n) == steps:
            el[int(k)] = float(t)
    if steps not in el:
        return {"mean": None, "median_after_first": None, "points": len(el)}
    el[0] = 0.0
    diffs = [el[i + 1] - el[i] for i in range(steps) if i in el and i + 1 in el]
    return {"mean": round(el[steps] / steps, 3),
            "median_after_first": round(statistics.median(diffs[1:]), 3) if len(diffs) > 1 else None,
            "points": len(el) - 1}


def metadata_picks(meta: dict) -> dict:
    h3 = meta.get("h3") or {}
    den = ((h3.get("phases") or {}).get("denoise")) or {}
    v16 = den.get("vram16") or {}
    sel = (((meta.get("models") or {}).get("selection") or {}).get("transformer")) or {}
    return {
        "transformer_selection": sel,
        "generation_time_seconds": meta.get("generation_time_seconds"),
        "num_inference_steps": h3.get("num_inference_steps"),
        "shift": [h3.get("shift_video"), h3.get("shift_audio")],
        "attention_used": meta.get("attention_used"), "vram16_mode_used": meta.get("vram16_mode_used"),
        "block_swap_prefetch_used": meta.get("block_swap_prefetch_used"),
        "peak_vram_mb": den.get("peak_vram_mb"), "peak_vram_reserved_mb": den.get("peak_vram_reserved_mb"),
        "top_peak_vram_mb": (meta.get("vram_optimization") or {}).get("peak_vram_mb"), "top_peak_vram_reserved_mb": meta.get("peak_vram_reserved_mb"),
        "denoise_seconds": den.get("seconds"), "dit_read": den.get("dit"),
        "vram16": {k: v16.get(k) for k in ("cap_mib", "num_alloc_retries", "num_ooms", "blocks_on_gpu",
                                           "qkv_quant_shared")},
        "quant_backends": h3.get("quant_backends"), "loras": h3.get("loras"),
        "request_loras": (meta.get("request") or {}).get("loras"),
    }


# ---------------------------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------------------------

def resolve_model_name(models: dict, filename: str) -> str | None:
    """The list_models entry whose path's file name is ``filename`` (``default`` for the descriptor's file)."""
    entries = ((models.get("categories") or {}).get("transformer") or {}).get("entries") or []
    for e in entries:
        if Path(str(e.get("path") or "")).name == filename and e.get("exists", True):
            return e.get("name")
    return None


def resolve_lora_name(loras: dict, filename: str) -> dict | None:
    stem = Path(filename).stem
    for e in loras.get("loras") or []:
        if e.get("name") == stem:
            return e
    return None


async def run_condition(mcp: Mcp, args, job: dict, models: dict, loras: dict) -> dict:
    cond = job["cond"]
    d = run_dir(cond)
    d.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {k: job[k] for k in ("cond", "purpose", "file", "steps", "shift", "lora",
                                               "expect_quant_layers")}
    rec.update(started_at=now_iso(), status=None, error=None, dir=rel(d), git=git_info())
    log(f"=== {cond}: {job['purpose']} (git {str(rec['git'].get('head'))[:10]}, "
        f"{len(rec['git'].get('status_porcelain') or [])} changed path(s)) ===")
    name = resolve_model_name(models, job["file"])
    rec["model_name"] = name
    if name is None:
        rec.update(status="model_not_listed", error=f"{job['file']} is not in list_models transformer entries",
                   finished_at=now_iso())
        write_json(d / "result.json", {"cond": cond, "run": rec})
        return rec
    tool_args = dict(job["tool_args"])
    if job["lora"]:
        entry = resolve_lora_name(loras, job["lora"]["file"])
        rec["lora_entry"] = entry
        if entry is None:
            rec.update(status="lora_not_listed", error=f"{job['lora']['file']} is not in list_loras",
                       finished_at=now_iso())
            write_json(d / "result.json", {"cond": cond, "run": rec})
            return rec
    wk_off = file_offset(worker_log_path())
    srv_off = file_offset(server_log_path())
    rec["load"] = await load_condition(mcp, args, job, name)
    if not rec["load"].get("ok"):
        rec.update(status="load_error", error=rec["load"].get("error"), finished_at=now_iso())
        rec["load_report"] = parse_load_report(read_since(worker_log_path(), wk_off))
        write_text(d / "worker_log_excerpt.txt", read_since(worker_log_path(), wk_off))
        log(f"  load failed: {rec['error']}")
        write_json(d / "result.json", {"cond": cond, "run": rec})
        return rec
    log(f"  loaded {name} in {rec['load']['seconds']} s")
    write_json(d / "request.json", {"tool": "submit_generate", "arguments": tool_args, "cond": cond,
                                    "load_pipeline": rec["load"]["requested"]})
    rec["tool_args"] = tool_args
    t0 = time.monotonic()
    try:
        sub = await mcp.call("submit_generate", tool_args)
    except ToolCallError as exc:
        rec.update(status="submit_error", error=exc.text, finished_at=now_iso())
        log(f"  submit_generate refused: {exc.text}")
        write_json(d / "result.json", {"cond": cond, "run": rec})
        return rec
    job_id = sub.get("job_id")
    rec["job_id"] = job_id
    log(f"  job_id={job_id}")
    waited = await wait_job(mcp, job_id, args.job_timeout_sec)
    final = waited.pop("final") or {}
    rec["wait"] = waited
    rec["job_final"] = final
    rec["status"] = final.get("status") or "unknown"
    rec["seconds_wall"] = round(time.monotonic() - t0, 1)
    wk = read_since(worker_log_path(), wk_off)
    srv = read_since(server_log_path(), srv_off)
    write_text(d / "worker_log_excerpt.txt", wk)
    write_text(d / "server_log_excerpt.txt", srv)
    rec["load_report"] = parse_load_report(wk)
    rec["sec_per_step"] = parse_step_times(srv, job["steps"])
    if waited.get("gave_up"):
        rec.update(status="wait_gave_up", error=waited["gave_up"], finished_at=now_iso())
        write_json(d / "result.json", {"cond": cond, "run": rec})
        return rec
    if rec["status"] != "completed":
        rec["error"] = final.get("error")
        log(f"  job {rec['status']}: {(str(final.get('error') or ''))[:300]}")
        rec["finished_at"] = now_iso()
        write_json(d / "result.json", {"cond": cond, "run": rec})
        return rec
    try:
        saved = await mcp.call("save_job_video", {"job_id": job_id, "dest_dir": str(d), "filename": "output.mp4",
                                                  "no_clobber": True, "which": "output"})
        rec["saved_path"] = rel(saved.get("saved_path") or "")
    except ToolCallError as exc:
        rec["error"] = f"save_job_video: {exc.text}"
    meta: dict = {}
    try:
        vp = await mcp.call("get_job_video_path", {"job_id": job_id})
        src = Path(vp["path"]).parent / "metadata.json"
        rec["job_dir"] = rel(src.parent)
        if src.exists():
            write_text(d / "metadata.json", src.read_text(encoding="utf-8"))
            meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        else:
            rec["error"] = rec["error"] or f"metadata.json missing in {rel(src.parent)}"
    except ToolCallError as exc:
        rec["error"] = rec["error"] or f"get_job_video_path: {exc.text}"
    rec["metadata"] = metadata_picks(meta)
    rec["finished_at"] = now_iso()
    spp = rec["sec_per_step"]
    ring = rec["load_report"].get("block_swap_ring") or {}
    log(f"  done: job {rec['seconds_wall']} s, s/step {spp.get('mean')}, denoise peak VRAM "
        f"{rec['metadata']['peak_vram_mb']}/{rec['metadata']['peak_vram_reserved_mb']} MiB, "
        f"ring {ring.get('slots')} x {ring.get('slot_mb')} MB, quant layers "
        f"{((rec['load_report'].get('build_report') or {}).get('dit') or {}).get('quant_layers')}")
    write_json(d / "result.json", {"cond": cond, "run": rec})
    return rec


# ---------------------------------------------------------------------------------------------
# Post: checks, MD5, metrics, sheets, summary
# ---------------------------------------------------------------------------------------------

def _post_of(cond: str) -> dict:
    p = run_dir(cond) / "result.json"
    if not p.exists():
        return {}
    try:
        return (json.loads(p.read_text(encoding="utf-8")).get("post")) or {}
    except Exception:  # noqa: BLE001
        return {}


def baseline_noise() -> tuple[str, dict] | None:
    for b in BASELINES:
        n = _post_of(b).get("noise")
        if isinstance(n, dict) and n.get("bytes_per_frame") is not None:
            return b, n
    return None


def r0_md5() -> dict | None:
    """R0's recorded stream MD5 (runs/R0/result.json post.stream_md5); else computed from runs/R0/output.mp4."""
    rec = _post_of("R0").get("stream_md5")
    if isinstance(rec, dict) and rec.get("video"):
        return rec
    mp4 = run_dir("R0") / "output.mp4"
    return stream_md5s(mp4) if mp4.exists() else None


def self_checks(cond: str, run: dict, probe: dict | None, md5: dict | None) -> dict:
    c = CONDITIONS[cond]
    m = run.get("metadata") or {}
    lr = run.get("load_report") or {}
    dit = ((lr.get("build_report") or {}).get("dit")) or {}
    exp = expected_formats(WEIGHTS / c["file"])
    qb = m.get("quant_backends") or {}
    cuda = qb.get("cuda") if isinstance(qb.get("cuda"), dict) else {}
    items: dict[str, dict] = {}
    warnings: list[str] = []
    items["status_completed"] = {"value": run.get("status"), "ok": run.get("status") == "completed"}
    v = (probe or {}).get("video") or {}
    items["mp4_frames_73_audio"] = {"value": [v.get("width"), v.get("height"), v.get("nb_read_frames"),
                                              ((probe or {}).get("audio") or {}).get("codec_name")],
                                    "ok": bool(probe) and v.get("nb_read_frames") == NUM_FRAMES
                                    and (v.get("width"), v.get("height")) == (WIDTH, HEIGHT)
                                    and bool((probe or {}).get("audio"))}
    sel = (m.get("transformer_selection") or {}).get("file")
    items["transformer_file"] = {"value": sel, "expect": c["file"], "ok": sel == c["file"]}
    items["steps_shift"] = {"value": [m.get("num_inference_steps"), m.get("shift")],
                            "expect": [c["steps"], list(c["shift"])],
                            "ok": m.get("num_inference_steps") == c["steps"] and m.get("shift") == list(c["shift"])}
    # metadata.json writes it as "on" / "off" (e.g. the 2026-11-12 G1 run).
    items["vram16_used"] = {"value": m.get("vram16_mode_used"), "expect": "on",
                            "ok": m.get("vram16_mode_used") in ("on", True)}
    items["quant_backends_flags"] = {
        "value": cuda, "ok": bool(cuda.get("int8_linear")) and "w4a8_int8_linear" in cuda
        and "quantize_per_tensor_fp8" in cuda
        and (FP8 not in (exp.get("quant_layers_by_format") or {}) or bool(cuda.get("quantize_per_tensor_fp8")))
        and (W4A8 not in (exp.get("quant_layers_by_format") or {}) or bool(cuda.get("w4a8_int8_linear")))}
    items["load_report_found"] = {"value": lr.get("found"), "ok": bool(lr.get("found"))}
    items["fp8_mode_none"] = {"value": dit.get("fp8_mode"), "expect": None,
                              "ok": lr.get("found", False) and dit.get("fp8_mode") is None}
    items["quant_layers_expected"] = {"value": dit.get("quant_layers"), "expect": c["expect_quant_layers"],
                                      "ok": dit.get("quant_layers") == c["expect_quant_layers"]}
    items["layers_by_format_vs_header"] = {
        "value": dit.get("quant_layers_by_format"),
        "expect": exp.get("quant_layers_by_format") if exp.get("ok") else exp.get("error"),
        "ok": exp.get("ok", False) and dit.get("quant_layers_by_format") == exp["quant_layers_by_format"]
        and dit.get("quant_layers") == exp["quant_layers"]}
    items["block_swap_ring_logged"] = {"value": (lr.get("block_swap_ring") or {}).get("line"),
                                       "ok": bool((lr.get("block_swap_ring") or {}).get("found"))}
    if c["lora"]:
        lo = m.get("loras")
        items["lora_applied"] = {"value": lo, "ok": bool(lo)}
    blob = json.dumps({k: run.get(k) for k in ("error", "load", "job_final")}, ensure_ascii=False, default=str)
    items["no_model_incompatible"] = {"value": "MODEL_INCOMPATIBLE" in blob, "ok": "MODEL_INCOMPATIBLE" not in blob}
    if cond == "R1":
        ref = r0_md5()
        if ref is None:
            warnings.append("R0 の記録（runs/R0/）が無いため、R1 とのストリームの MD5 を比べていない")
        else:
            items["stream_md5_equals_R0"] = {"value": md5, "expect": ref,
                                             "ok": bool(md5) and md5.get("video") == ref.get("video")
                                             and md5.get("audio") == ref.get("audio")}
    return {"items": items, "ok": all(i["ok"] for i in items.values()), "warnings": warnings,
            "expected_from_header": exp}


def extract_frames(mp4: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = "+".join(f"eq(n\\,{f})" for f in SHEET_FRAMES)
    pat = out_dir / "f_%d.png"
    r = run_cmd([tool("ffmpeg"), "-v", "error", "-y", "-i", str(mp4), "-vf", f"select='{sel}'",
                 "-fps_mode", "passthrough", str(pat)], timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {r.stderr[-500:]}")
    frames = [out_dir / f"f_{i + 1}.png" for i in range(len(SHEET_FRAMES))]
    missing = [str(p) for p in frames if not p.exists()]
    if missing:
        raise RuntimeError(f"ffmpeg wrote fewer frames than {SHEET_FRAMES}: {missing}")
    return frames


def make_sheet(rows: dict[str, list[Path]], path: Path) -> None:
    """Rows = conditions, columns = frames 0/36/72."""
    from PIL import Image, ImageDraw  # noqa: PLC0415 -- app venv (Pillow)

    first = Image.open(next(iter(rows.values()))[0])
    w, h = first.size
    scale = SHEET_TILE_LONG / max(w, h)
    tw, th = int(round(w * scale)), int(round(h * scale))
    head = 22
    sheet = Image.new("RGB", (tw * len(SHEET_FRAMES), (th + head) * len(rows)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (label, frames) in enumerate(rows.items()):
        y = i * (th + head)
        for j, fp in enumerate(frames):
            with Image.open(fp) as im:
                sheet.paste(im.convert("RGB").resize((tw, th), Image.LANCZOS), (j * tw, y + head))
            draw.text((j * tw + 6, y + 4), f"{label}  frame {SHEET_FRAMES[j]}", fill=(235, 235, 235))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def post_condition(cond: str) -> dict:
    d = run_dir(cond)
    rpath = d / "result.json"
    if not rpath.exists():
        return {"cond": cond, "status": "no_run"}
    rj = json.loads(rpath.read_text(encoding="utf-8"))
    run = rj.get("run") or {}
    mp4 = d / "output.mp4"
    probe = ffprobe(mp4) if mp4.exists() else None
    md5 = stream_md5s(mp4) if mp4.exists() else None
    post: dict[str, Any] = {"cond": cond, "at": now_iso(), "output_probe": probe, "stream_md5": md5}
    if probe:
        post["noise"] = noise_metrics(mp4)
        post["noise_flags"] = noise_flags(cond, post["noise"], None if cond in BASELINES else baseline_noise())
        with tempfile.TemporaryDirectory(dir=str(d)) as tmp:
            frames = extract_frames(mp4, Path(tmp))
            make_sheet({cond: frames}, d / "sheet.png")
    post["checks"] = self_checks(cond, run, probe, md5)
    post["status"] = "checked"
    rj["post"] = post
    write_json(rpath, rj)
    checks = post["checks"]
    bad = [k for k, v in checks["items"].items() if not v["ok"]]
    log(f"{cond}: self-checks {'all passed' if not bad else 'FAILED ' + str(bad)}"
        + (f"; warnings {checks['warnings']}" if checks["warnings"] else ""))
    nf = post.get("noise_flags") or {}
    if nf.get("flags"):
        log(f"{cond}: 要確認 {nf['flags']} ({NOISE_NOTE})")
    return post


def make_sheet_all(conds: list[str]) -> None:
    rows: dict[str, list[Path]] = {}
    with tempfile.TemporaryDirectory(dir=str(OUT_ROOT)) as tmp:
        for cond in conds:
            mp4 = run_dir(cond) / "output.mp4"
            if mp4.exists():
                rows[cond] = extract_frames(mp4, Path(tmp) / cond)
        if rows:
            make_sheet(rows, SHEET_ALL)


def load_summary() -> dict:
    if SUMMARY_JSON.exists():
        try:
            data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("conds", {})
                data.setdefault("invocations", [])
                return data
        except Exception:  # noqa: BLE001
            pass
    return {"script": "h3_mixed_int8_gate_api.py", "conds": {}, "invocations": []}


def _short(v: Any, n: int = 100) -> str:
    s = json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"


def write_summary_md(summary: dict, conds: list[str]) -> None:
    L = ["# MiniMax H3 の混在 int8 の門（記録）", "",
         "台本 `outputs/h3-mixed-int8-gate-2026-11-22/h3_mixed_int8_gate_api.py`。",
         "R0 は修正の前、R1・X・Y・Z・YL は修正の後に回す。", "",
         f"共通: T2V・{WIDTH}×{HEIGHT}・{NUM_FRAMES} 枚・{H3_FPS}fps・seed {SEED}・"
         f"注意機構 {ATTENTION}・VRAM 16GB mode on・先読み on・常駐 off。", "",
         f"**{NOISE_NOTE}**", "",
         "| 条件 | 状態 | 自己検査 | 目安 |", "|---|---|---|---|"]
    for cond in conds:
        g = summary["conds"].get(cond) or {}
        run, post = g.get("run") or {}, g.get("post") or {}
        chk = (post.get("checks") or {}).get("ok")
        nf = post.get("noise_flags") or {}
        mark = "—" if not nf.get("checked") else ("要確認" if nf.get("flags") else "範囲内")
        L.append(f"| {cond} | {run.get('status') or '—'} | {_p(chk) if chk is not None else '—'} | {mark} |")
    L.append("")
    for cond in conds:
        c = CONDITIONS[cond]
        g = summary["conds"].get(cond) or {}
        run, post = g.get("run") or {}, g.get("post") or {}
        m = run.get("metadata") or {}
        lr = run.get("load_report") or {}
        dit = ((lr.get("build_report") or {}).get("dit")) or {}
        ring = lr.get("block_swap_ring") or {}
        git = run.get("git") or {}
        noise = post.get("noise") or {}
        nf = post.get("noise_flags") or {}
        md5 = post.get("stream_md5") or {}
        lora = f"`{Path(c['lora']['file']).stem}`（強さ {c['lora']['strength']}）" if c["lora"] else "なし"
        L += [f"## {cond}", "", f"{c['purpose']}", "",
              f"- 重み: `{c['file']}`",
              f"- ステップ／shift: {c['steps']}／{c['shift'][0]:g}・{c['shift'][1]:g}",
              f"- LoRA: {lora}",
              f"- 状態: {run.get('status') or '—'}",
              f"- git: `{str(git.get('head'))[:12]}`"
              f"（作業ツリーの変更 {len(git.get('status_porcelain') or [])} 件）",
              f"- 読み込み秒: {_p((run.get('load') or {}).get('seconds'), 1)}",
              f"- ジョブ秒: {_p(run.get('seconds_wall'), 1)}",
              f"- 秒／ステップ: {_p((run.get('sec_per_step') or {}).get('mean'), 3)}",
              f"- 量子化層: {_p(dit.get('quant_layers'))}（期待 {c['expect_quant_layers']}）",
              f"- 置き場: {_p(ring.get('slots'))} × {_p(ring.get('slot_mb'), 1)} MB",
              f"- ノイズ除去の VRAM の山: {_p(m.get('peak_vram_mb'))}／"
              f"{_p(m.get('peak_vram_reserved_mb'))} MiB（割り当て／予約）",
              f"- 映像の MD5: `{md5.get('video') or '—'}`",
              f"- 音声の MD5: `{md5.get('audio') or '—'}`",
              f"- バイト／フレーム: {_p(noise.get('bytes_per_frame'), 0)}",
              f"- ラプラシアンの平均: {_p(noise.get('laplacian_mean'), 2)}",
              f"- フレーム差の平均: {_p(noise.get('frame_diff_mean'), 2)}",
              f"- 音声の RMS: {_p(noise.get('audio_rms_dbfs'), 1)} dBFS"]
        if nf.get("checked"):
            r = nf.get("ratios") or {}
            L.append(f"- 基準 {nf.get('baseline') or '—'} との比: バイト {_p(r.get('bytes_per_frame'), 2)}・"
                     f"ラプラシアン {_p(r.get('laplacian_mean'), 2)}・フレーム差 {_p(r.get('frame_diff_mean'), 2)}")
            L.append(f"- 目安: {'要確認' if nf.get('flags') else '範囲内'}")
            for f in nf.get("flags") or []:
                L.append(f"  - 要確認: {f}")
        items = ((post.get("checks") or {}).get("items")) or {}
        if items:
            L += ["", "自己検査:", ""]
            for k, v in items.items():
                L.append(f"- {k}: {_p(v['ok'])}")
                if not v["ok"]:
                    L.append(f"  - 値 `{_short(v.get('value'))}`")
                    if "expect" in v:
                        L.append(f"  - 期待 `{_short(v.get('expect'))}`")
        for w in (post.get("checks") or {}).get("warnings") or []:
            L.append(f"- 警告: {w}")
        if run.get("error"):
            L.append(f"\nエラー: `{str(run['error'])[:500]}`")
        L.append("")
    write_text(SUMMARY_MD, "\n".join(L))


def flush(summary: dict, conds: list[str]) -> None:
    write_json(SUMMARY_JSON, summary)
    write_summary_md(summary, conds)


def done_conds(summary: dict) -> list[str]:
    return [c for c in CONDITIONS if c in summary["conds"]]


def do_post(conds: list[str], summary: dict) -> int:
    code = 0
    for cond in [c for c in CONDITIONS if c in conds]:   # baselines (R0, R1) first
        try:
            post = post_condition(cond)
            rj = json.loads((run_dir(cond) / "result.json").read_text(encoding="utf-8")) \
                if (run_dir(cond) / "result.json").exists() else {}
            summary["conds"].setdefault(cond, {})["run"] = rj.get("run") or summary["conds"].get(cond, {}).get("run")
        except Exception as exc:  # noqa: BLE001
            post = {"cond": cond, "status": "post_error", "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
            log(f"{cond}: post failed: {post['error']}")
        summary["conds"].setdefault(cond, {})["post"] = post
        if post.get("status") != "checked":
            code = 3
    try:
        make_sheet_all(list(CONDITIONS))
    except Exception as exc:  # noqa: BLE001
        log(f"sheet_all failed: {type(exc).__name__}: {exc}")
        code = 3
    flush(summary, done_conds(summary))
    return code


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

async def run_live(args, jobs: list[dict], summary: dict, inv: dict) -> int:
    from mcp import ClientSession  # noqa: PLC0415
    from mcp.client.stdio import stdio_client  # noqa: PLC0415

    calls: list = []
    calls_path = LOG_DIR / f"mcp_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    inv["mcp_calls"] = rel(calls_path)
    inv["mcp_env"] = mcp_env(args.port)
    code = 0
    conds = [j["cond"] for j in jobs]
    try:
        with open(LOG_DIR / "mcp_server.stderr.log", "a", encoding="utf-8", newline="\n") as errlog:
            errlog.write(f"\n===== {now_iso()} h3_mixed_int8_gate_api.py {conds} =====\n")
            errlog.flush()
            async with stdio_client(mcp_params(args.port), errlog=errlog) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=TOOL_READ_TIMEOUT) as session:
                    init = await session.initialize()
                    inv["server_info"] = {"name": init.serverInfo.name, "version": init.serverInfo.version,
                                          "protocol": init.protocolVersion}
                    schema = check_tool_schemas((await session.list_tools()).tools)
                    inv["tool_schema_check"] = schema
                    if not schema["ok"]:
                        for p in schema["problems"]:
                            log(f"argument check failed: {p}")
                        return 1
                    log("argument check against tools/list: passed")
                    mcp = Mcp(session, calls)
                    st = await mcp.call("backend_status")
                    inv["status_before"] = st
                    if not st.get("reachable"):
                        log(f"backend unreachable: {st.get('error')}")
                        return 1
                    models = await mcp.call("list_models")
                    loras = await mcp.call("list_loras")
                    inv["active_base_model"] = models.get("active_base_model")
                    inv["transformer_entries"] = [
                        {"name": e.get("name"), "path": rel(str(e.get("path") or "")), "exists": e.get("exists")}
                        for e in ((models.get("categories") or {}).get("transformer") or {}).get("entries") or []]
                    inv["lora_names"] = [e.get("name") for e in loras.get("loras") or []]
                    if models.get("active_base_model") != H3_BASE_MODEL:
                        log(f"active base model is {models.get('active_base_model')}, not {H3_BASE_MODEL}")
                        return 1
                    flush(summary, conds)
                    for job in jobs:
                        cond = job["cond"]
                        try:
                            rec = await run_condition(mcp, args, job, models, loras)
                        except Exception as exc:  # noqa: BLE001
                            rec = {"cond": cond, "status": "script_error",
                                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                            write_json(run_dir(cond) / "result.json", {"cond": cond, "run": rec})
                        summary["conds"][cond] = {"conditions": job, "run": rec}
                        if rec.get("status") != "completed" or rec.get("error"):
                            code = 3
                        try:
                            post = post_condition(cond)
                        except Exception as exc:  # noqa: BLE001
                            post = {"cond": cond, "status": "post_error", "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()}
                            log(f"{cond}: post failed: {post['error']}")
                        summary["conds"][cond]["post"] = post
                        flush(summary, conds)
                        passed = (post.get("checks") or {}).get("ok") is True
                        if not passed and (args.stop_on_fail or job["stop_on_fail"]):
                            rest = conds[conds.index(cond) + 1:]
                            inv["stopped_after"] = {"cond": cond, "skipped": rest}
                            log(f"{cond}: self-checks failed -> stop (skipped {rest})")
                            code = 3
                            break
                    try:
                        inv["unload"] = await mcp.call("unload_pipeline")
                    except ToolCallError as exc:
                        inv["unload"] = {"error": exc.text}
    finally:
        write_json(calls_path, calls)
    return code


def dry_run(args, jobs: list[dict]) -> int:
    """Print the plan. Starts nothing and opens no socket (no backend, no MCP server, no port probe,
    no nvidia-smi, no torch). Reads the weight headers (h3_format.inspect_h3_dit) and git (read only)."""
    ok = True
    log(f"repo {rel(REPO) or '.'} (the folder two levels above this script)")
    log(f"backend (not started): {' '.join(rel(x) for x in server_cmd())}")
    log(f"  temp config {rel(RT / TEMP_CONFIG_NAME)} / temp state {rel(RT / TEMP_STATE_NAME)} "
        "(config.yaml read only; port, log_dir, output.dir, upload.dir, state_file -> _rt/)")
    env_show = {k: (rel(v) if k in ("LTX_MCP_OUTPUT_DIR", "LTX_MCP_CONFIG") else v)
                for k, v in mcp_env(args.port).items()}
    log(f"MCP server (not started): .venv/Scripts/python.exe -m mcp_server  env {env_show}")
    log(f"preflight in a live run: port {OWNER_PORT} probe, leftover engine workers, nvidia-smi (python/uvicorn on "
        f"the GPU, used memory <= {GPU_BUSY_MIB} MiB) -- none of these is run in dry-run (no socket is opened)")
    gi = git_info()
    log(f"git: {gi.get('branch')} {gi.get('head')}; status --porcelain: {len(gi.get('status_porcelain') or [])} line(s)")
    sig = check_tool_signatures_static()
    log(f"tool arguments vs mcp_server/tools/*.py: {'passed' if sig['ok'] else 'FAILED ' + str(sig['problems'])} "
        f"({sig['registered_count']} registered tools)")
    ok = ok and sig["ok"]
    out: dict[str, Any] = {"made_at": now_iso(), "argv": sys.argv[1:], "git": gi, "tool_signatures": sig,
                           "order": [j["cond"] for j in jobs], "conds": {}}
    for job in jobs:
        cond = job["cond"]
        p = WEIGHTS / job["file"]
        exists = p.is_file()
        exp = expected_formats(p) if exists else {"ok": False, "error": "file missing"}
        lora_ok = (LORA_DIR / job["lora"]["file"]).is_file() if job["lora"] else None
        existing = [rel(x) for x in (run_dir(cond) / "output.mp4", run_dir(cond) / "result.json") if x.exists()]
        layers_ok = exp.get("ok") and exp.get("quant_layers") == job["expect_quant_layers"]
        entry = {"purpose": job["purpose"], "file": job["file"], "exists": exists,
                 "size": p.stat().st_size if exists else None,
                 "model_name_expected": "default" if job["file"] == R_FILE else p.stem,
                 "header": exp, "expect_quant_layers": job["expect_quant_layers"], "header_layers_match": layers_ok,
                 "lora_file_exists": lora_ok, "stop_on_fail": job["stop_on_fail"] or args.stop_on_fail,
                 "load_pipeline": {"base_model": H3_BASE_MODEL, "models": {"transformer": "<list_models name>"},
                                   "wait_sec": LOAD_WAIT_SEC},
                 "submit_generate": job["tool_args"], "existing_outputs": existing,
                 "noise_checked": cond in NOISE_CHECKED}
        out["conds"][cond] = entry
        size = "%.2f GB" % (entry["size"] / 1e9) if exists else "MISSING"
        if job["lora"]:
            lora_txt = (f"{Path(job['lora']['file']).stem} x{job['lora']['strength']} "
                        f"(file {'present' if lora_ok else 'MISSING'})")
        else:
            lora_txt = "none"
        log(f"{cond}: {job['file']} ({size}) steps {job['steps']} shift {job['shift'][0]:g}/{job['shift'][1]:g} "
            f"lora {lora_txt}")
        log(f"    header: {exp.get('quant_layers') if exp.get('ok') else 'REFUSED/ERROR ' + str(exp.get('error'))} layers "
            f"{exp.get('quant_layers_by_format')} specs {exp.get('quant_specs')} unknown {exp.get('unknown')}; "
            f"expect {job['expect_quant_layers']} -> {'match' if layers_ok else 'MISMATCH'}; "
            f"stop-on-fail {entry['stop_on_fail']}; model name '{entry['model_name_expected']}'")
        log(f"    submit_generate: {json.dumps({k: v for k, v in job['tool_args'].items() if k != 'prompt'})}")
        if not exists or not exp.get("ok") or lora_ok is False or not layers_ok:
            ok = False
            log("    PROBLEM: file missing, header refused, LoRA missing or layer count differs")
        if existing:
            log(f"    note: outputs already present (a live run of this condition refuses): {existing}")
    if "R1" in out["conds"]:
        r0 = _post_of("R0").get("stream_md5")
        log(f"R1 vs R0 stream MD5: {'R0 record present' if r0 else 'no R0 record yet (a warning only)'}")
    log(f"prompt: {jobs[0]['tool_args']['prompt']}")
    log(f"outputs: {rel(RUNS)}/<cond>/ (output.mp4, metadata.json, result.json, sheet.png), "
        f"{rel(SUMMARY_MD)}, {rel(SUMMARY_JSON)}, {rel(SHEET_ALL)}")
    out["ok"] = bool(ok)
    write_json(OUT_ROOT / "dry_run.json", out)
    log(f"dry-run record: {rel(OUT_ROOT / 'dry_run.json')} ({'OK' if ok else 'PROBLEMS'})")
    return 0 if ok else 1


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="MiniMax H3 mixed-int8 gate (R0/R1/X/Y/Z/YL; MCP, own-port backend)")
    ap.add_argument("--only", default=None,
                    help=f"comma list of {','.join(CONDITIONS)} (default {','.join(DEFAULT_ORDER)}; R0 only when named)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the plan; start nothing, open no socket")
    mode.add_argument("--live", action="store_true", help="start the backend and generate (GPU)")
    mode.add_argument("--post-only", action="store_true", help="checks, MD5, metrics, sheets, summary from runs/")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="stop at the first condition whose self-checks fail (always on for X)")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT, help=f"temp backend port (default {PORT_DEFAULT})")
    ap.add_argument("--prompt", default=None, help="override the prompt (default: G1's)")
    ap.add_argument("--load-timeout-sec", type=float, default=1200.0)
    ap.add_argument("--job-timeout-sec", type=float, default=3600.0, help="per-job wait limit (default 3600 s)")
    args = ap.parse_args(argv)
    if not (args.dry_run or args.live or args.post_only):
        ap.error("choose one of --dry-run, --live (generates on the GPU) or --post-only")
    if args.only is not None:
        names = [x.strip() for x in args.only.split(",") if x.strip()]
        bad = [x for x in names if x not in CONDITIONS]
        if bad or not names:
            ap.error(f"--only: unknown condition(s) {bad or '(empty)'}; choose from {','.join(CONDITIONS)}")
    if args.port == OWNER_PORT:
        ap.error(f"port {OWNER_PORT} is the owner's server")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args(argv)
    mode = "dry_run" if args.dry_run else ("post_only" if args.post_only else "live")
    log.open(LOG_DIR / f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    prompt = args.prompt or DEFAULT_PROMPT
    conds = cond_list(args.only)
    try:
        jobs = [build_job(c, prompt) for c in conds]
        if args.dry_run:
            return dry_run(args, jobs)
        summary = load_summary()
        inv: dict[str, Any] = {"started_at": now_iso(), "argv": sys.argv[1:], "mode": mode, "conds": conds,
                               "log": rel(log.path), "git": git_info()}
        summary["invocations"].append(inv)
        if args.post_only:
            code = do_post(conds, summary)
            inv.update(finished_at=now_iso(), exit_code=code)
            flush(summary, done_conds(summary))
            log(f"records: {rel(SUMMARY_JSON)}, {rel(SUMMARY_MD)} (exit {code})")
            return code
        for job in jobs:
            existing = [rel(x) for x in (run_dir(job["cond"]) / "output.mp4", run_dir(job["cond"]) / "result.json")
                        if x.exists()]
            if existing:
                log(f"{job['cond']}: outputs already present (not overwritten); move them away first: {existing}")
                return 1
            if not (WEIGHTS / job["file"]).is_file():
                log(f"{job['cond']}: weight file missing: {job['path']}")
                return 1
        if "R1" in conds and "R0" not in conds and not (run_dir("R0") / "result.json").exists():
            log("warning: runs/R0/ has no record; R1's stream MD5 will not be compared (warning only)")
        pre = preflight_checks()
        inv["preflight"] = pre
        log_preflight(pre)
        if pre["blocking"]:
            for b in pre["blocking"]:
                log(f"preflight refuses: {b}")
            flush(summary, conds)
            return 1
        srv = Server(args.port)
        code = 1
        try:
            srv.start()
            inv["server"] = srv.record
            code = asyncio.run(run_live(args, jobs, summary, inv))
        except BaseException as exc:  # noqa: BLE001 -- record exceptions and Ctrl+C, then stop the backend
            inv["error"] = f"{type(exc).__name__}: {exc}"
            inv["traceback"] = traceback.format_exc()
            log(f"stopped: {inv['error']}")
            code = 1
        finally:
            try:
                srv.stop()
            except Exception as exc:  # noqa: BLE001
                srv.record["stop_error"] = f"{type(exc).__name__}: {exc}"
            inv["server"] = srv.record
            inv["workers_after_stop"] = [{"pid": p["pid"], "cmdline": p["cmdline"][:300]}
                                         for p in all_processes() if WORKER_RE.search(p["cmdline"] or "")]
            if inv["workers_after_stop"]:
                log(f"WARNING workers still alive after stop: {inv['workers_after_stop']}")
            flush(summary, done_conds(summary))
        ran = [c for c in conds if (run_dir(c) / "result.json").exists()]
        if ran:
            pcode = do_post(ran, summary)
            code = code or pcode
        inv["scrubbed_files"] = scrub_files()
        inv.update(finished_at=now_iso(), exit_code=code)
        flush(summary, done_conds(summary))
        log(f"records: {rel(SUMMARY_JSON)}, {rel(SUMMARY_MD)} (exit {code})")
        return code
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
