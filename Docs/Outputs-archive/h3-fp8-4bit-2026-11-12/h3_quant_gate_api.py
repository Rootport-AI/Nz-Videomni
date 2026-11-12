r"""MiniMax H3 fp8 / w4a8 product gate G1-G6 (ledger §1-283), driven through the API
(MCP server -> app backend -> engine worker). G7 (the Settings refusal text) is a manual check.

Plan: ``imperative-crafting-lightning.md`` §5.2. Product code is never edited (read only).
Structure copied from ``Docs/Outputs-archive/h3-inpaint-2026-11-11/h3_inpaint_gate_api.py``
(temp yaml + temp state + backend on its own port, MCP stdio client, argument-name check,
load wait, job wait, save, stop in ``finally``). Those helpers live inside that script (no shared
module; it binds its own output folder at import time), so they are copied here, as that script
copied them from the outpaint gate. ``gpu_busy_check`` is copied from
``outputs/h3-fp8-w4a8-spike-2026-11-12/h3_quant_spike.py`` (which imports torch, so it cannot be imported).

From the repo root (Git Bash, the APP venv; this script never imports torch):

    export PYTHONUTF8=1
    G=outputs/h3-fp8-w4a8-gate-2026-11-12/h3_quant_gate_api.py
    .venv/Scripts/python.exe $G --dry-run              # plan only: nothing is started
    .venv/Scripts/python.exe $G                        # G1..G6 in order, one backend start
    .venv/Scripts/python.exe $G --only fp8             # one condition
    .venv/Scripts/python.exe $G --post-only            # checks, sheets, summary.md, hf-round17/ from runs/

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
     (``_rt/logs/h3_worker.log``) for this condition -> ``runs/<cond>/result.json``.
  5. Stop the backend (process tree) in ``finally``.
  6. Post (same as ``--post-only``): ffprobe, self-checks, contact sheets, ``summary.md``,
     ``hf-round17/`` (clips, sheets, ``sheet_all.png``, Japanese README skeleton).

Where the load report comes from: the worker's ``build_report`` (``engine_h3/pipeline_h3.py``
``H3Pipeline.load``: ``dit.quant_layers``, ``dit.quant_layers_by_format``, ``dit.fp8_mode``) is NOT in
any API response or in ``metadata.json``; the worker logs it as one line
``[h3_worker] LOAD_OK {json}`` (``engine_h3/worker.py`` ``_do_load``) on its stderr, which the adapter
appends to ``<log_dir>/h3_worker.log`` (``services/engines/h3/adapter.py`` ``_LOG_NAME``). The gate
reads the last ``LOAD_OK`` line written after this condition's load started, plus the
``fp8: N layer(s), path '<mode>'`` line (``H3Pipeline._log_fp8``).

Exit codes: 0 = every run completed and post-processing ran (pass/fail is recorded, not in the exit
code), 1 = stopped in preflight / setup, 2 = bad arguments, 3 = a job failed / was refused / timed out.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
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
HF_DIR = OUT_ROOT / "hf-round17"
HF_REPO = "Rootport/Nz-LTX23-eval-clips"
HF_NAME = "round17-h3-quant-product"
SPIKE_DIR = REPO / "outputs" / "h3-fp8-w4a8-spike-2026-11-12"
PORT_DEFAULT = 18699
OWNER_PORT = 18620
TEMP_CONFIG_NAME = "config.h3quantgate.yaml"
TEMP_STATE_NAME = "state.h3quantgate.json"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
GPU_BUSY_MIB = 3072  # same bound as h3_quant_spike.py

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
#: Same string as the spike (h3_quant_spike.py DEFAULT_PROMPT).
DEFAULT_PROMPT = (
    "A busy city street at dusk seen from the sidewalk. Pedestrians walk past the camera, their footsteps "
    "clicking on the pavement; cars and a bus drive by with engine hum and tire noise, and a car horn honks "
    "in the distance. Steady camera."
)

WEIGHTS = REPO / "models" / "MiniMaxH3" / "Weights"
LORA_DIR = REPO / "models" / "MiniMaxH3" / "LoRA"
TURBO_LORA_FILE = "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"

FP8 = "float8_e4m3fn"       # h3_format.FP8
W4A8 = "asym_w4a8_int8"     # h3_format.W4A8
INT8 = "int8_tensorwise"    # h3_format.INT8

#: G6 shift: Docs/MINIMAX_H3_RESEARCH_NOTES.md has no shift recommendation for the FL2V 768p Turbo LoRA
#: (its line 835 is the Ref2VA Turbo: 4 steps, shift 12/3); the project's earlier Turbo LoRA runs used
#: 4 steps, shift 12/3 (Docs/VERIFICATION_LOG.md lines 22308 and 22565). So 12/3.
CONDITIONS: dict[str, dict[str, Any]] = {
    "int8": {"gate": "G1", "file": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "steps": 20,
             "shift": (12.0, 3.0), "lora": None, "spike": "int8", "expect_fp8_mode": None,
             "purpose": "公式 int8（回帰の基準）"},
    "fp8": {"gate": "G2", "file": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "steps": 20,
            "shift": (12.0, 3.0), "lora": None, "spike": "fp8", "expect_fp8_mode": "scaled",
            "purpose": "公式 fp8（fp8_mode=scaled）"},
    "w4a8_kijai": {"gate": "G3", "file": "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors", "steps": 20,
                   "shift": (12.0, 3.0), "lora": None, "spike": "w4a8_kijai", "expect_fp8_mode": None,
                   "purpose": "Kijai の w4a8（メタデータ式の印だけ）"},
    "w4a8_eros": {"gate": "G4", "file": "10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors",
                  "steps": 8, "shift": (10.0, 3.0), "lora": None, "spike": "w4a8_eros", "expect_fp8_mode": None,
                  "purpose": "h3ErosMax の w4a8（混在・G 16/8/4・BF16 の表）"},
    "int8_oldmarker": {"gate": "G5", "file": "h3ErosMax_beta5_3185154.safetensors", "steps": 8,
                       "shift": (10.0, 3.0), "lora": None, "spike": None, "expect_fp8_mode": None,
                       "purpose": "旧式の印の int8（h3ErosMax beta5。読み込めて完走すること）"},
    "int8_oldmarker_s20": {"gate": "G5b", "file": "h3ErosMax_beta5_3185154.safetensors", "steps": 20,
                           "shift": (12.0, 3.0), "lora": None, "spike": None, "expect_fp8_mode": None,
                           "purpose": "G5 の回し直し（公式と同じ 20 ステップ・shift 12/3。暗さがステップ数の不足か傾向かの切り分け）"},
    "fp8_turbo": {"gate": "G6", "file": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "steps": 4,
                  "shift": (12.0, 3.0), "lora": {"file": TURBO_LORA_FILE, "strength": 1.0}, "spike": None,
                  "expect_fp8_mode": "scaled", "purpose": "fp8 に Turbo LoRA（強さ 1.0・4 ステップ）"},
}

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
WORKER_RE = re.compile(r"-m\s+engine\w*\.worker|engine\w*[\\/]worker\.py")


# ---------------------------------------------------------------------------------------------
# Small helpers (copied from the inpaint gate)
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
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
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
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


def tool(name: str) -> str:
    local = REPO / "tools" / "ffmpeg" / "bin" / f"{name}.exe"
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found (<repo>/tools/ffmpeg/bin or PATH)")
    return found


def run_cmd(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


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
    return {"path": str(path), "video": {**vs, "fps": fps, "nb_read_frames": int(nb) if nb not in (None, "N/A") else None},
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


# ---------------------------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------------------------

def cond_list(only: str | None) -> list[str]:
    return [only] if only else list(CONDITIONS)


def hf_clip_name(cond: str) -> str:
    return f"h3_{cond}_{WIDTH}x{HEIGHT}_{CONDITIONS[cond]['steps']}steps_seed{SEED}.mp4"


def run_dir(cond: str) -> Path:
    return RUNS / cond


def expected_formats(path: Path) -> dict:
    """Per-format layer counts from the file header (``h3_format.inspect_h3_dit``: header only, torch-free).
    This is what the worker's ``build_report.dit.quant_layers_by_format`` must equal."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        import h3_format  # noqa: PLC0415

        _header, cfg = h3_format.inspect_h3_dit(str(path))
    except Exception as exc:  # noqa: BLE001 -- a refusal here is exactly what G5 must not hit
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
    return {"cond": cond, "gate": c["gate"], "purpose": c["purpose"], "file": c["file"],
            "path": str(WEIGHTS / c["file"]), "steps": c["steps"], "shift": [sv, sa], "lora": c["lora"],
            "expect_fp8_mode": c["expect_fp8_mode"], "tool_args": tool_args}


# ---------------------------------------------------------------------------------------------
# Preflight (owner server, GPU, leftover workers)
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
    """Copied from h3_quant_spike.py: no other python / uvicorn process on the GPU, and the GPU's used
    memory below a desktop-only bound. Returns a record with ``blocking`` instead of raising."""
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
# Temp yaml and the backend (copied from the inpaint gate; paths under this folder's _rt/)
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
    write_text(cfg, yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
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
    write_text(state, json.dumps({"schema": 1, "active_base_model": H3_BASE_MODEL, "selections": selections},
                                 indent=2))
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
        self._fh.write(f"\n===== {now_iso()} h3_quant_gate_api.py: main.py --config {cfg} =====\n")
        self._fh.flush()
        cmd = server_cmd()
        self.proc = subprocess.Popen(cmd, cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=self._fh,
                                     stderr=subprocess.STDOUT, env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.record.update(pid=self.proc.pid, cmd=cmd, config=str(cfg), state=str(state),
                           log=str(self.log_path), started_at=now_iso())
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
# MCP (copied from the inpaint gate)
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
    ``mcp_server/tools/*.py``, collect the functions registered with ``mcp.tool()(<name>)`` and their
    parameter names, and compare with :data:`REQUIRED_TOOL_ARGS`."""
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
    """Poll backend_status every 10 s until H3 is loaded (copied from the inpaint gate's ensure_pipeline)."""
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
    """The last ``LOAD_OK {json}`` line (worker build_report) and the last ``fp8: ... path '<mode>'`` line."""
    out: dict[str, Any] = {"found": False}
    hits = LOAD_OK_RE.findall(worker_text or "")
    if hits:
        try:
            report = json.loads(hits[-1])
            out.update(found=True, load_ok_lines=len(hits), build_report=report)
        except json.JSONDecodeError as exc:
            out["error"] = f"LOAD_OK is not JSON: {exc}"
    fp8 = FP8_LINE_RE.findall(worker_text or "")
    if fp8:
        out["fp8_log_line"] = {"layers": int(fp8[-1][0]), "path": fp8[-1][1]}
    return out


def parse_step_times(server_text: str, steps: int) -> dict:
    """s/step from the server log's ``denoise k/N steps (..., Ts elapsed)`` lines (ltx.runner): mean =
    elapsed at step N / N (same definition as the spike's StepClock: step 0 -> step N), and the median of
    the per-step differences after the first."""
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
        "top_peak_vram_reserved_mb": meta.get("peak_vram_reserved_mb"),
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
    rec: dict[str, Any] = {k: job[k] for k in ("cond", "gate", "purpose", "file", "steps", "shift", "lora",
                                               "expect_fp8_mode")}
    rec.update(started_at=now_iso(), status=None, error=None, dir=str(d))
    log(f"=== {job['gate']} {cond}: {job['purpose']} ===")
    name = resolve_model_name(models, job["file"])
    rec["model_name"] = name
    if name is None:
        rec.update(status="model_not_listed", error=f"{job['file']} is not in list_models transformer entries",
                   finished_at=now_iso())
        return rec
    tool_args = dict(job["tool_args"])
    if job["lora"]:
        entry = resolve_lora_name(loras, job["lora"]["file"])
        rec["lora_entry"] = entry
        if entry is None:
            rec.update(status="lora_not_listed", error=f"{job['lora']['file']} is not in list_loras",
                       finished_at=now_iso())
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
        rec["saved_path"] = saved.get("saved_path")
    except ToolCallError as exc:
        rec["error"] = f"save_job_video: {exc.text}"
    meta: dict = {}
    try:
        vp = await mcp.call("get_job_video_path", {"job_id": job_id})
        src = Path(vp["path"]).parent / "metadata.json"
        rec["job_dir"] = str(src.parent)
        if src.exists():
            shutil.copy2(src, d / "metadata.json")
            meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        else:
            rec["error"] = rec["error"] or f"metadata.json missing in {src.parent}"
    except ToolCallError as exc:
        rec["error"] = rec["error"] or f"get_job_video_path: {exc.text}"
    rec["metadata"] = metadata_picks(meta)
    rec["finished_at"] = now_iso()
    spp = rec["sec_per_step"]
    log(f"  done: job {rec['seconds_wall']} s, s/step {spp.get('mean')}, peak VRAM "
        f"{rec['metadata']['peak_vram_mb']}/{rec['metadata']['peak_vram_reserved_mb']} MiB, "
        f"fp8_mode {((rec['load_report'].get('build_report') or {}).get('dit') or {}).get('fp8_mode')}")
    write_json(d / "result.json", {"cond": cond, "run": rec})
    return rec


# ---------------------------------------------------------------------------------------------
# Post: checks, sheets, summary, hf-round17/
# ---------------------------------------------------------------------------------------------

def spike_reference(cond: str) -> dict | None:
    """The spike's S3 numbers for the same file (``outputs/h3-fp8-w4a8-spike-2026-11-12/s3/<name>/result.json``)."""
    name = CONDITIONS[cond]["spike"]
    if not name:
        return None
    p = SPIKE_DIR / "s3" / name / "result.json"
    if not p.exists():
        return None
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return {"name": name, "sec_per_step": r.get("sec_per_step"), "status": r.get("status")}


def self_checks(cond: str, run: dict, probe: dict | None) -> dict:
    c = CONDITIONS[cond]
    m = run.get("metadata") or {}
    lr = run.get("load_report") or {}
    dit = ((lr.get("build_report") or {}).get("dit")) or {}
    exp = expected_formats(WEIGHTS / c["file"])
    qb = m.get("quant_backends") or {}
    cuda = qb.get("cuda") if isinstance(qb.get("cuda"), dict) else {}
    items: dict[str, dict] = {}
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
    items["quant_backends_flags"] = {
        "value": cuda, "ok": bool(cuda.get("int8_linear")) and "w4a8_int8_linear" in cuda
        and "quantize_per_tensor_fp8" in cuda
        and (FP8 not in (exp.get("quant_layers_by_format") or {}) or bool(cuda.get("quantize_per_tensor_fp8")))
        and (W4A8 not in (exp.get("quant_layers_by_format") or {}) or bool(cuda.get("w4a8_int8_linear")))}
    items["load_report_found"] = {"value": lr.get("found"), "ok": bool(lr.get("found"))}
    items["fp8_mode"] = {"value": dit.get("fp8_mode"), "expect": c["expect_fp8_mode"],
                         "ok": lr.get("found", False) and dit.get("fp8_mode") == c["expect_fp8_mode"]}
    items["layers_by_format"] = {"value": dit.get("quant_layers_by_format"),
                                 "expect": exp.get("quant_layers_by_format") if exp.get("ok") else exp.get("error"),
                                 "ok": exp.get("ok", False) and dit.get("quant_layers_by_format") == exp["quant_layers_by_format"]
                                 and dit.get("quant_layers") == exp["quant_layers"]}
    if c["lora"]:
        lo = m.get("loras")
        items["lora_applied"] = {"value": lo, "ok": bool(lo)}
    if cond != "int8":
        blob = json.dumps({k: run.get(k) for k in ("error", "load", "job_final")}, ensure_ascii=False, default=str)
        items["no_model_incompatible"] = {"value": "MODEL_INCOMPATIBLE" in blob, "ok": "MODEL_INCOMPATIBLE" not in blob}
    # G1 (int8) is the regression baseline: there is no fp8 layer, so the log has no fp8 line either.
    return {"items": items, "ok": all(i["ok"] for i in items.values()), "expected_from_header": exp}


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
    """Rows = conditions, columns = frames 0/36/72 (as the spike's make_sheet)."""
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
    checks = self_checks(cond, run, probe)
    post: dict[str, Any] = {"cond": cond, "at": now_iso(), "output_probe": probe, "checks": checks,
                            "spike_reference": spike_reference(cond)}
    if probe:
        with tempfile.TemporaryDirectory(dir=str(d)) as tmp:
            frames = extract_frames(mp4, Path(tmp))
            make_sheet({cond: frames}, d / "sheet.png")
        HF_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mp4, HF_DIR / hf_clip_name(cond))
        shutil.copy2(d / "sheet.png", HF_DIR / f"sheet_{cond}.png")
        post["hf_clip"] = hf_clip_name(cond)
    post["status"] = "checked"
    rj["post"] = post
    write_json(rpath, rj)
    bad = [k for k, v in checks["items"].items() if not v["ok"]]
    log(f"{cond}: self-checks {'all passed' if not bad else 'FAILED ' + str(bad)}")
    return post


def make_sheet_all(conds: list[str]) -> None:
    rows: dict[str, list[Path]] = {}
    with tempfile.TemporaryDirectory(dir=str(OUT_ROOT)) as tmp:
        for cond in conds:
            mp4 = run_dir(cond) / "output.mp4"
            if mp4.exists():
                rows[f"{CONDITIONS[cond]['gate']} {cond}"] = extract_frames(mp4, Path(tmp) / cond)
        if rows:
            make_sheet(rows, HF_DIR / "sheet_all.png")


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
    return {"script": "h3_quant_gate_api.py", "plan": "imperative-crafting-lightning.md §5.2（G1〜G6）",
            "conds": {}, "invocations": []}


def row_values(cond: str, summary: dict) -> dict:
    g = summary["conds"].get(cond) or {}
    run, post = g.get("run") or {}, g.get("post") or {}
    m = run.get("metadata") or {}
    dit = (((run.get("load_report") or {}).get("build_report") or {}).get("dit")) or {}
    spike = (post.get("spike_reference") or {}).get("sec_per_step") or {}
    checks = (post.get("checks") or {})
    return {"run": run, "post": post, "status": run.get("status"), "job_id": run.get("job_id"),
            "load_s": (run.get("load") or {}).get("seconds"), "job_s": run.get("seconds_wall"),
            "gen_s": m.get("generation_time_seconds"), "spp": (run.get("sec_per_step") or {}).get("mean"),
            "spp_med": (run.get("sec_per_step") or {}).get("median_after_first"), "spike_spp": spike.get("mean"),
            "peak": m.get("peak_vram_mb"), "peak_res": m.get("peak_vram_reserved_mb"),
            "dit_read_s": (m.get("dit_read") or {}).get("read_seconds"), "fp8_mode": dit.get("fp8_mode"),
            "by_fmt": dit.get("quant_layers_by_format"), "checks_ok": checks.get("ok"),
            "failed": [k for k, v in (checks.get("items") or {}).items() if not v.get("ok")]}


def write_summary_md(summary: dict, conds: list[str]) -> None:
    L = ["# MiniMax H3 の fp8・w4a8 の GPU の門（G1〜G6。API 経由。記録）", "",
         "台本 `outputs/h3-fp8-w4a8-gate-2026-11-12/h3_quant_gate_api.py`。計画 `imperative-crafting-lightning.md` §5.2。"
         "G7（Settings の拒否の理由文）は手で確かめる。", "",
         "| 門 | 条件 | 状態 | 読み込み秒 | ジョブ秒（壁） | 生成秒 | 秒／ステップ（中央値） | スパイク S3 の秒／ステップ | "
         "VRAM の山 割り当て／予約（MiB） | fp8_mode | 形式ごとの層の数 | 自己検査 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cond in conds:
        r = row_values(cond, summary)
        chk = "—" if r["checks_ok"] is None else ("すべて合格" if r["checks_ok"] else "不合格 " + "・".join(r["failed"]))
        L.append(f"| {CONDITIONS[cond]['gate']} | `{cond}` | {r['status'] or '—'} | {_p(r['load_s'], 1)} | {_p(r['job_s'], 1)} | "
                 f"{_p(r['gen_s'], 1)} | {_p(r['spp'], 3)}（{_p(r['spp_med'], 3)}） | {_p(r['spike_spp'], 3)} | "
                 f"{_p(r['peak'], 0)}／{_p(r['peak_res'], 0)} | {r['fp8_mode'] or '—'} | "
                 f"{json.dumps(r['by_fmt']) if r['by_fmt'] else '—'} | {chk} |")
    L += ["", "秒／ステップはサーバーのログの `denoise k/N steps (..., Ts elapsed)` から（N ステップ目の経過秒 ÷ N。"
          "スパイクの StepClock と同じ定義）。fp8_mode と層の数はワーカーのログの `LOAD_OK` の行（`build_report`）から。", ""]
    for cond in conds:
        r = row_values(cond, summary)
        items = ((r["post"].get("checks") or {}).get("items")) or {}
        if not items:
            continue
        L += [f"## {CONDITIONS[cond]['gate']} `{cond}`", "", "| 検査 | 値 | 期待 | 合否 |", "|---|---|---|---|"]
        for k, v in items.items():
            L.append(f"| {k} | `{json.dumps(v.get('value'), ensure_ascii=False, default=str)}` | "
                     f"{'`' + json.dumps(v.get('expect'), ensure_ascii=False) + '`' if 'expect' in v else '—'} | {_p(v['ok'])} |")
        if r["run"].get("error"):
            L.append(f"\nエラー: `{str(r['run']['error'])[:500]}`")
        L.append("")
    write_text(SUMMARY_MD, "\n".join(L))


def write_hf_readme(summary: dict, conds: list[str], prompt: str) -> None:
    done = [c for c in conds if (HF_DIR / hf_clip_name(c)).exists()]
    if not done:
        return
    L = [f"# {HF_NAME} — MiniMax H3 の fp8・w4a8 の重みを製品で回した記録（2026-11-12）", "",
         f"置き場: `{HF_REPO}` の `{HF_NAME}/`。", "",
         "## 目的", "",
         "製品に入れた MiniMax H3 の fp8（8 ビットの浮動小数点の重み）と w4a8（重みを 4 ビット・入力を 8 ビットにした形式）の"
         "読み込みを、利用者と同じ経路（MCP〔Model Context Protocol。外部の道具からアプリを操作する仕組み〕→ アプリのサーバー → "
         "エンジン）で回し、完走することと絵が破綻しないことを確かめました（台帳 §1-283）。", "",
         "## 条件", "",
         f"共通: T2V（文章から動画）・{WIDTH}×{HEIGHT}・{NUM_FRAMES} 枚・24fps・種（seed）{SEED}・注意の実装（attention）{ATTENTION}・"
         "VRAM 16GB mode on・先読み（prefetch）on。", "",
         f"プロンプト: {prompt}", "",
         "| 門 | 条件 | 重みのファイル | ステップ | shift（映像／音声） | LoRA | 読み込み秒 | 秒／ステップ | VRAM の山 割り当て／予約（MiB） | fp8_mode | 形式ごとの層の数 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cond in conds:
        c = CONDITIONS[cond]
        r = row_values(cond, summary)
        lora = f"`{Path(c['lora']['file']).stem}`（強さ {c['lora']['strength']}）" if c["lora"] else "なし"
        L.append(f"| {c['gate']} | `{cond}` | `{c['file']}` | {c['steps']} | {c['shift'][0]:g}／{c['shift'][1]:g} | {lora} | "
                 f"{_p(r['load_s'], 1)} | {_p(r['spp'], 3)} | {_p(r['peak'], 0)}／{_p(r['peak_res'], 0)} | "
                 f"{r['fp8_mode'] or '—'} | {json.dumps(r['by_fmt']) if r['by_fmt'] else '—'} |")
    L += ["", "## ファイル", ""]
    for cond in done:
        L.append(f"- `{hf_clip_name(cond)}`・`sheet_{cond}.png`（{CONDITIONS[cond]['gate']}。並び画像はフレーム 0・36・72）")
    L += ["- `sheet_all.png`: 全条件の並び画像（行が条件、列がフレーム 0・36・72）", "",
          "## 所見", "", "", ""]
    write_text(HF_DIR / "README.md", "\n".join(L))


def flush(summary: dict, conds: list[str]) -> None:
    write_json(SUMMARY_JSON, summary)
    write_summary_md(summary, conds)


def do_post(conds: list[str], summary: dict, prompt: str) -> int:
    code = 0
    for cond in conds:
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
    all_conds = [c for c in CONDITIONS if c in summary["conds"]]
    flush(summary, all_conds)
    write_hf_readme(summary, all_conds, prompt)
    if (HF_DIR / "README.md").exists():
        log(f"HF folder: {HF_DIR}")
    return code


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

async def run_live(args, jobs: list[dict], summary: dict, inv: dict) -> int:
    from mcp import ClientSession  # noqa: PLC0415
    from mcp.client.stdio import stdio_client  # noqa: PLC0415

    calls: list = []
    calls_path = LOG_DIR / f"mcp_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    inv["mcp_calls"] = str(calls_path)
    inv["mcp_env"] = mcp_env(args.port)
    code = 0
    conds = [j["cond"] for j in jobs]
    try:
        with open(LOG_DIR / "mcp_server.stderr.log", "a", encoding="utf-8", newline="\n") as errlog:
            errlog.write(f"\n===== {now_iso()} h3_quant_gate_api.py {conds} =====\n")
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
                    # H3 is the active base model of the temp state, so list_models / list_loras list H3's files.
                    models = await mcp.call("list_models")
                    loras = await mcp.call("list_loras")
                    inv["active_base_model"] = models.get("active_base_model")
                    inv["transformer_entries"] = [
                        {k: e.get(k) for k in ("name", "path", "exists")}
                        for e in ((models.get("categories") or {}).get("transformer") or {}).get("entries") or []]
                    inv["lora_names"] = [e.get("name") for e in loras.get("loras") or []]
                    if models.get("active_base_model") != H3_BASE_MODEL:
                        log(f"active base model is {models.get('active_base_model')}, not {H3_BASE_MODEL}")
                        return 1
                    flush(summary, conds)
                    for job in jobs:
                        try:
                            rec = await run_condition(mcp, args, job, models, loras)
                        except Exception as exc:  # noqa: BLE001
                            rec = {"cond": job["cond"], "status": "script_error",
                                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                            write_json(run_dir(job["cond"]) / "result.json", {"cond": job["cond"], "run": rec})
                        summary["conds"][job["cond"]] = {"conditions": job, "run": rec}
                        flush(summary, conds)
                        if rec.get("status") != "completed" or rec.get("error"):
                            code = 3
                    try:
                        inv["unload"] = await mcp.call("unload_pipeline")
                    except ToolCallError as exc:
                        inv["unload"] = {"error": exc.text}
    finally:
        write_json(calls_path, calls)
    return code


def dry_run(args, jobs: list[dict]) -> int:
    """Print the plan. Starts nothing (no backend, no MCP server, no nvidia-smi, no torch)."""
    ok = True
    log(f"repo {REPO}")
    log(f"backend (not started): {' '.join(server_cmd())}")
    log(f"  temp config {RT / TEMP_CONFIG_NAME} / temp state {RT / TEMP_STATE_NAME} "
        "(config.yaml read only; port, log_dir, output.dir, upload.dir, state_file -> _rt/)")
    log(f"MCP server (not started): {PY} -m mcp_server  env {mcp_env(args.port)}")
    log(f"port {OWNER_PORT} now: {'LISTENING (a live run would refuse)' if port_listening(OWNER_PORT) else 'free'}; "
        f"port {args.port} now: {'LISTENING (a live run would refuse)' if port_listening(args.port) else 'free'}")
    log("preflight in a live run: port 18620, leftover engine workers, nvidia-smi (python/uvicorn on the GPU, "
        f"used memory <= {GPU_BUSY_MIB} MiB) -- not run in dry-run")
    sig = check_tool_signatures_static()
    log(f"tool arguments vs mcp_server/tools/*.py: {'passed' if sig['ok'] else 'FAILED ' + str(sig['problems'])} "
        f"({sig['registered_count']} registered tools)")
    ok = ok and sig["ok"]
    out: dict[str, Any] = {"made_at": now_iso(), "argv": sys.argv[1:], "tool_signatures": sig, "conds": {}}
    for job in jobs:
        cond = job["cond"]
        p = Path(job["path"])
        exists = p.is_file()
        exp = expected_formats(p) if exists else {"ok": False, "error": "file missing"}
        lora_ok = True
        if job["lora"]:
            lora_ok = (LORA_DIR / job["lora"]["file"]).is_file()
        existing = [str(x) for x in (run_dir(cond) / "output.mp4", run_dir(cond) / "result.json") if x.exists()]
        spike = spike_reference(cond)
        entry = {"gate": job["gate"], "purpose": job["purpose"], "file": job["file"], "exists": exists,
                 "size": p.stat().st_size if exists else None, "model_name_expected": "default" if cond == "int8" else p.stem,
                 "header": exp, "lora_file_exists": lora_ok if job["lora"] else None,
                 "load_pipeline": {"base_model": H3_BASE_MODEL, "models": {"transformer": "<list_models name>"},
                                   "wait_sec": LOAD_WAIT_SEC},
                 "submit_generate": job["tool_args"], "expect_fp8_mode": job["expect_fp8_mode"],
                 "existing_outputs": existing, "spike_reference": spike, "hf_clip": hf_clip_name(cond)}
        out["conds"][cond] = entry
        log(f"{job['gate']} {cond}: {job['file']} ({'%.2f GB' % (entry['size'] / 1e9) if exists else 'MISSING'}) "
            f"steps {job['steps']} shift {job['shift'][0]:g}/{job['shift'][1]:g} "
            f"lora {job['lora'] and Path(job['lora']['file']).stem} x{job['lora'] and job['lora']['strength']} "
            f"-> model name '{entry['model_name_expected']}'")
        log(f"    header: {exp.get('quant_layers_by_format') if exp.get('ok') else 'REFUSED/ERROR ' + str(exp.get('error'))}"
            f"; expect fp8_mode {job['expect_fp8_mode']}; spike S3 s/step "
            f"{((spike or {}).get('sec_per_step') or {}).get('mean')}; hf clip {entry['hf_clip']}")
        if not exists or not exp.get("ok") or not lora_ok:
            ok = False
            log(f"    PROBLEM: file missing, header refused, or LoRA missing")
        if existing:
            log(f"    note: outputs already present (a live run of this condition refuses): {existing}")
    log(f"submit_generate common args: {json.dumps({k: v for k, v in jobs[0]['tool_args'].items() if k not in ('prompt', 'num_inference_steps', 'shift_video', 'shift_audio')})}")
    log(f"prompt: {jobs[0]['tool_args']['prompt']}")
    log(f"outputs: {RUNS}/<cond>/ (output.mp4, metadata.json, result.json), {SUMMARY_MD}, {HF_DIR}/ "
        f"(HF dataset {HF_REPO} / {HF_NAME}; the upload is done by the supervisor)")
    out["ok"] = ok
    write_json(OUT_ROOT / "dry_run.json", out)
    log(f"dry-run record: {OUT_ROOT / 'dry_run.json'} ({'OK' if ok else 'PROBLEMS'})")
    return 0 if ok else 1


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="MiniMax H3 fp8 / w4a8 product gate G1-G6 (MCP, own-port backend)")
    ap.add_argument("--only", choices=tuple(CONDITIONS), help="one condition (default: all, in G1..G6 order)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; start nothing")
    ap.add_argument("--post-only", action="store_true", help="checks, sheets, summary.md and hf-round17/ from runs/")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT, help=f"temp backend port (default {PORT_DEFAULT})")
    ap.add_argument("--prompt", default=None, help="override the prompt (default: the spike's)")
    ap.add_argument("--load-timeout-sec", type=float, default=1200.0)
    ap.add_argument("--job-timeout-sec", type=float, default=3600.0, help="per-job wait limit (default 3600 s)")
    args = ap.parse_args(argv)
    if args.dry_run and args.post_only:
        ap.error("--dry-run and --post-only are exclusive")
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
                               "log": str(log.path)}
        summary["invocations"].append(inv)
        if args.post_only:
            code = do_post(conds, summary, prompt)
            inv.update(finished_at=now_iso(), exit_code=code)
            flush(summary, [c for c in CONDITIONS if c in summary["conds"]])
            log(f"records: {SUMMARY_JSON}, {SUMMARY_MD} (exit {code})")
            return code
        for job in jobs:
            existing = [str(x) for x in (run_dir(job["cond"]) / "output.mp4", run_dir(job["cond"]) / "result.json")
                        if x.exists()]
            if existing:
                log(f"{job['cond']}: outputs already present (not overwritten); move them away first: {existing}")
                return 1
            if not Path(job["path"]).is_file():
                log(f"{job['cond']}: weight file missing: {job['path']}")
                return 1
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
            flush(summary, conds)
        ran = [c for c in conds if (run_dir(c) / "result.json").exists()]
        if ran:
            pcode = do_post(ran, summary, prompt)
            code = code or pcode
        inv.update(finished_at=now_iso(), exit_code=code)
        flush(summary, [c for c in CONDITIONS if c in summary["conds"]])
        log(f"records: {SUMMARY_JSON}, {SUMMARY_MD} (exit {code})")
        return code
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
