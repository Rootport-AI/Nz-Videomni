r"""MiniMax H3 Sol-H3-VAE spike (2026-11-23): the 24-layer Sol decoder (``sol_decoder.py``) against
NVIDIA's reference and the product's standard video decoder, without editing any product file.

Plan (authoritative on scope): ``parsed-greeting-pancake-of-328841ff-d123-4805-a662-cd263e6a068b.md``.
Patterns from ``outputs/h3-pih3-spike-2026-11-22/h3_pih3_spike.py`` (``Log``, the GPU gate, the
replacement in / out, the result records) and ``outputs/comfort-calib-h3-2026-11-04/calib.py`` (the
WDDM shared-memory sampler: ``wddm_sampler.ps1`` is its ``vram_sampler2.ps1``).

From the repository root (Git Bash):

    PY=./.venv-engine-h3/Scripts/python.exe; S=outputs/h3-sol-spike-2026-11-23/h3_sol_spike.py
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --dry-run            # CPU only (GPU hidden)
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --gate               # no GPU work: the checks before a GPU run
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --selfcheck          # GPU: the kitchen functions vs torch (small)
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --verify             # GPU: plan 5.2 (random latents by default)
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --gen a              # GPU: plan 5.3, one of a b c d e
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --compare a          # GPU: plan 5.4 / 5.5 on the saved latent
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --bench              # GPU: plan 5.6 (resumes; --force redoes)

Exit codes: 0 = passed / done, 1 = failed (recorded in the JSON), 2 = refused by the gate / missing input.
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
CPU_ONLY = any(a in sys.argv for a in ("--dry-run", "--gate"))
if CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import h3_format  # noqa: E402
from engine_h3 import pipeline_h3 as P  # noqa: E402
from engine_h3 import video_vae as VV  # noqa: E402
from engine_h3.constants import (  # noqa: E402
    DEFAULT_SHIFT_AUDIO, DEFAULT_SHIFT_VIDEO, DEFAULT_STEPS, FRAME_RATE, latent_frames)

import sol_decoder as SD  # noqa: E402
import sol_metrics as SM  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixed conditions
# --------------------------------------------------------------------------- #

RUNS = HERE / "runs"
REF_FILE = HERE / "ref_nvidia" / "decoder.py"
REF_SHA256 = "87f25f35bb1ceebbcae4abcfb0439d522dfbc0c233d3fce84a3e19f12ce242de"
SAMPLER = HERE / "wddm_sampler.ps1"

PATHS = {   # the product's default files (scripts/manifests/40-minimaxh3.json default_file)
    "transformer": "models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoder": "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "video_vae": "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
    "tokenizer_dir": "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
}
SOL_FILES = {"fp16": "models/MiniMaxH3/VAE/Sol/minimax_h3_sol_vae_fp16.safetensors",
             "int8": "models/MiniMaxH3/VAE/Sol/minimax_h3_sol_vae_int8_convrot.safetensors"}
TURBO_LORA_FILE = "models/MiniMaxH3/LoRA/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
TURBO_LORA_STRENGTH = 0.8        # the Turbo LoRA control of VERIFICATION_LOG §223.12
TURBO_STEPS = 4
TURBO_LORA_PAIRS = 208           # §205.7: pairs_applied 208
ILLUSTRATION = "uploads/h3-a2v-development/00043-2168133585.jpg"   # §224.12 (1) (896 x 1152)

# The product's defaults (the panel's form and acceleration defaults, the API / worker defaults).
DEFAULTS = {
    "seed": -1, "steps": DEFAULT_STEPS, "shift_video": float(DEFAULT_SHIFT_VIDEO),
    "shift_audio": float(DEFAULT_SHIFT_AUDIO), "width": 1280, "height": 768, "num_frames": 345,
    "frame_rate": FRAME_RATE, "loras": "なし", "negative_prompt": "", "attention_backend": "sdpa",
    "vram16_mode": True, "block_swap_prefetch": True, "keep_resident": False, "blocks_on_gpu": 8,
    "deterministic": True,
}
DEFAULT_SOURCES = {
    "seed": "config.yaml generation_defaults.seed（-1 = 毎回ランダム）",
    "steps": "engine_h3/constants.py DEFAULT_STEPS・services/engines/h3/adapter.py H3_DEFAULT_STEPS",
    "shift_video": "engine_h3/constants.py DEFAULT_SHIFT_VIDEO", "shift_audio": "engine_h3/constants.py DEFAULT_SHIFT_AUDIO",
    "width": "config.yaml generation_defaults.width", "height": "config.yaml generation_defaults.height",
    "num_frames": "config.yaml generation_defaults.num_frames", "frame_rate": "engine_h3/constants.py FRAME_RATE",
    "loras": "操作パネルの既定（LoRA なし）", "negative_prompt": "engine_h3/pipeline_h3.py H3GenerateRequest",
    "attention_backend": "webui shell/accelerationSettings.ts ATTENTION_BACKEND_DEFAULT・api/models.py",
    "vram16_mode": "webui VRAM16_MODE_SERVER_DEFAULT・api/models.py VRAM16_MODE_DEFAULT_H3",
    "block_swap_prefetch": "webui BLOCK_SWAP_PREFETCH_SERVER_DEFAULT・api/models.py",
    "keep_resident": "webui KEEP_RESIDENT_SERVER_DEFAULT・api/models.py",
    "blocks_on_gpu": "services/engines/h3/adapter.py DEFAULT_BLOCKS_ON_GPU",
    "deterministic": "services/engines/h3/adapter.py（読み込みの指示の deterministic）",
}
SEED = 1234                      # the pi-H3 spike's seed (fixed so that every decoder sees the same latents)

PROMPTS = {
    "a": ("A close-up portrait of a young woman's face in soft natural window light. She slowly turns her head "
          "toward the camera and gives a gentle smile; the fine natural texture of her skin is clearly visible. "
          "Quiet room tone. Shallow depth of field, steady camera."),
    "b": ("A calm, minimal scene: a clear, cloudless blue sky above the plain white wall of a modern building. "
          "A single small bird glides slowly across the empty sky. Large flat areas of even color, soft wind "
          "sound. Steady camera."),
    "c": ("A woman with long, wavy brown hair stands under a leafy tree on a breezy day. The wind moves the "
          "countless small leaves, her loose strands of hair and the fine woven texture of her knitted wool "
          "sweater. Rustling leaves. Rich fine detail, natural daylight, steady camera."),
    "d": ("An anime-style illustration of a girl drawn with flat cel-shaded colors and clean line art. She "
          "blinks and gently tilts her head while smiling at the viewer; the plain background stays still. "
          "Soft indoor ambience. Steady camera."),
    "e": ("A young girl in a white summer dress and a straw hat walks along a sandy beach at the water's edge on "
          "a sunny day. Gentle waves wash ashore with the sound of surf, seagulls call in the distance, and her "
          "footsteps are soft on the wet sand. Steady tracking camera."),   # the pi-H3 spike's CHAIN_SAME_PROMPT
}
GEN = {
    "a": {"kind": "t2v", "width": 1280, "height": 768, "frames": 124, "turbo": True},
    "b": {"kind": "t2v", "width": 1280, "height": 768, "frames": 124, "turbo": True},
    "c": {"kind": "t2v", "width": 1280, "height": 768, "frames": 345, "turbo": False},
    "d": {"kind": "i2v", "width": 896, "height": 1152, "frames": 124, "turbo": True},
    "e": {"kind": "chain", "width": 640, "height": 384, "frames": 124, "clips": 2, "turbo": True},
}
GEN_TITLES = {"a": "人物の顔の大写し（肌）", "b": "平坦な領域の多い場面（空・無地の壁）",
              "c": "細部の多い場面（木の葉・布地・髪）", "d": "イラスト調（平坦な塗り）",
              "e": "連結 2 本（共通のプロンプトだけ）"}
LATENT_FILE = "latent.safetensors"

#: Plan 5.2: (T, h, w) latent grids. Largest first; an out-of-memory goes on to the smaller ones.
VERIFY = {
    "V1_640x384_73f": (latent_frames(73), 384 // 16, 640 // 16),
    "V3_odd_592x368_T14": (14, 23, 37),     # windows cut at the edges (23 = 8 + 15, 37 = 16 + 16 + 5 / 8 + 16 + 13)
    "V2_512x320_39f": (latent_frames(39), 320 // 16, 512 // 16),
}
VERIFY_PASS_DB = 80.0
VERIFY_A_FLOOR_DB = 40.0         # A (fp16) against the float32 reference: a rough floor, below it the run stops

#: Plan 5.6: key -> (width, height, frames, the standard decoder measured too).
BENCH = {
    "1280x768_124": (1280, 768, 124, True),
    "1280x768_345": (1280, 768, 345, True),
    "1280x768_481": (1280, 768, 481, False),
    "1920x1088_124": (1920, 1088, 124, True),
    "1920x1088_362": (1920, 1088, 362, False),
    "2560x1472_73": (2560, 1472, 73, False),
    "2560x1472_192": (2560, 1472, 192, False),
    "chain_1280x768_940": (1280, 768, 940, False),
    "chain_1920x1088_685": (1920, 1088, 685, False),
    "chain_2560x1472_362": (2560, 1472, 362, False),
}
BENCH_RUNS = 4                   # the first is discarded; the median of the other 3
COMFORT_ALLOC_MIB = 14000        # COMFORT_LIMIT_TABLE §17.1: what the calibration could allocate

OWNER_PORT = 18620
GPU_BUSY_MIB = 3072
WORKER_RE = re.compile(r"-m\s+engine\w*\.worker|engine\w*[\\/]worker\.py")


# --------------------------------------------------------------------------- #
# Small helpers (from h3_pih3_spike.py)
# --------------------------------------------------------------------------- #

def _short_path(path: str) -> str:
    """The 8.3 short form of ``path`` (Windows), or ``path`` itself."""
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024)
        return buf.value if 0 < n < 1024 else str(path)
    except Exception:  # noqa: BLE001
        return str(path)


def user_names() -> list[str]:
    """This machine's user name as it can appear in a path: the environment's user name and the last
    part of the home folder, long and 8.3 short (spaces and all)."""
    home = os.environ.get("USERPROFILE") or str(Path.home())
    names = {os.environ.get("USERNAME") or "", Path(home).name, Path(_short_path(home)).name}
    return sorted((n for n in names if n.strip()), key=len, reverse=True)


def make_scrub(names: list[str]):
    """``text -> text`` replacing each name literally (case-insensitive) after ``Users`` and any run of
    slashes / backslashes (a JSON string doubles them), whatever characters the name holds."""
    pats = [re.compile(r"(Users(?:\\\\|\\|/)+)" + re.escape(n), re.IGNORECASE) for n in names]

    def scrub_(text: str) -> str:
        for p in pats:
            text = p.sub(lambda m: m.group(1) + "<user>", text)
        return text
    return scrub_


scrub = make_scrub(user_names())


class Log:
    def __init__(self) -> None:
        self.file = None

    def __call__(self, msg: str) -> None:
        line = scrub(f"[h3_sol_spike] {msg}")
        print(line, flush=True)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()

    def progress(self, stage: str, index: int, total: int, *, outer_index=None, outer_total=None) -> None:
        """The pipeline's / chain's progress callback (``outer_*``: the chain's clip)."""
        extra = "" if outer_index is None else f" (clip {outer_index + 1}/{outer_total})"
        self(f"[progress] {stage} {index}/{total}{extra}")

    def open(self, path: Path, mode: str = "a") -> None:
        self.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, mode, encoding="utf-8")

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


LOG = Log()


class _LogBridge(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            LOG(f"[{record.levelname} {record.name}] {record.getMessage()}")
        except Exception:  # noqa: BLE001
            pass


def install_log_bridge() -> None:
    root = logging.getLogger()
    if not any(isinstance(h, _LogBridge) for h in root.handlers):
        root.addHandler(_LogBridge(level=logging.INFO))
    root.setLevel(logging.INFO)


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def rel(p) -> str:
    q = Path(p).resolve()
    try:
        return str(q.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return q.name


def jdump(obj) -> str:
    return scrub(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(jdump(obj) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def run_cmd(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def tool(name: str) -> str:
    local = ROOT / "tools" / "ffmpeg" / "bin" / f"{name}.exe"
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found (<repo>\\tools\\ffmpeg\\bin or PATH)")
    return found


def ffprobe(path: Path) -> dict:
    v = run_cmd([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                 "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames", "-of", "json", str(path)])
    return {"path": rel(path), "video": (json.loads(v.stdout or "{}").get("streams") or [{}])[0]}


def failure(rec: dict, exc: BaseException) -> None:
    rec["status"] = "failed"
    rec["error"] = repr(exc)
    rec["traceback"] = traceback.format_exc()
    LOG(f"FAILED: {exc!r}\n{rec['traceback']}")


def git_state() -> dict:
    head = run_cmd(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    st = run_cmd(["git", "-C", str(ROOT), "status", "--short"])
    return {"head": head.stdout.strip(), "status_short": st.stdout.splitlines(), "clean": st.stdout.strip() == ""}


def versions() -> dict:
    out = {"torch": torch.__version__, "python": sys.version.split()[0]}
    for dist in ("comfy-kitchen", "safetensors"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def file_info(path) -> dict:
    p = ROOT / path
    return {"path": str(path), "bytes": p.stat().st_size if p.is_file() else None}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mib(v) -> int:
    return int(round(v / 2**20))


def free_cuda() -> None:
    P._free_cuda()


def peaks() -> dict:
    return {"alloc_mib": mib(torch.cuda.max_memory_allocated()), "reserved_mib": mib(torch.cuda.max_memory_reserved())}


def tensor_stats(t: torch.Tensor) -> dict:
    x = t.detach().double()
    return {"shape": list(t.shape), "mean": float(x.mean()), "std": float(x.std()), "max_abs": float(x.abs().max()),
            "nan": int(torch.isnan(x).sum()), "inf": int(torch.isinf(x).sum())}


# --------------------------------------------------------------------------- #
# GPU gate (h3_pih3_spike.py, with the other processes' VRAM recorded)
# --------------------------------------------------------------------------- #

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


def gpu_busy_check() -> dict:
    rec: dict = {"blocking": []}
    exe = shutil.which("nvidia-smi")
    if not exe:
        rec["blocking"].append("nvidia-smi が見つからない（GPU が空いていることを確かめられない）")
        return rec
    proc = run_cmd([exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], timeout=60)
    apps = []
    for line in (proc.stdout or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:   # the executable's name only (a full path could carry a user name)
            apps.append({"pid": parts[0], "name": re.split(r"[\\/]", ",".join(parts[1:-1]))[-1], "used_memory": parts[-1]})
    mine = {str(os.getpid())}
    others = [a for a in apps if a["pid"] not in mine and any(w in a["name"].lower() for w in ("python", "uvicorn"))]
    mem = run_cmd([exe, "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"], timeout=60)
    parts = [x.strip() for x in (mem.stdout or ",0,0").splitlines()[0].split(",")] if mem.stdout else ["", "0", "0"]
    try:
        used = int(parts[1])
    except (IndexError, ValueError):
        used = -1
    # recorded: how many processes use the GPU and whether a python / uvicorn is among them (no names);
    # the VRAM of the others is the total memory_used_mib (nvidia-smi on WDDM reports "[N/A]" per process)
    rec.update(compute_apps={"count": len(apps), "python_or_uvicorn_present": bool(others),
                             "python_or_uvicorn_count": len(others)},
               returncode=proc.returncode, gpu=parts, memory_used_mib=used)
    if proc.returncode != 0 or mem.returncode != 0 or used < 0:
        rec["blocking"].append(f"nvidia-smi が失敗（{proc.returncode}/{mem.returncode}）: {(proc.stderr or '')[-300:]}")
    if others:
        rec["blocking"].append(f"GPU を別の python／uvicorn が使っている（{len(others)} 件）")
    if used > GPU_BUSY_MIB:
        rec["blocking"].append(f"GPU の使用量 {used} MiB が {GPU_BUSY_MIB} MiB を超えている")
    return rec


def gate() -> dict:
    """``blocking`` non-empty -> the GPU run stops before anything is loaded."""
    rec: dict = {"at": now(), "blocking": []}
    listening = port_listening(OWNER_PORT)
    rec["owner_server"] = {"port": OWNER_PORT, "listening": listening}
    if listening:
        rec["blocking"].append(f"ポート {OWNER_PORT} で何かが待ち受けている（オーナーのサーバー）")
    gpu = gpu_busy_check()
    rec["gpu"] = gpu
    rec["blocking"] += gpu["blocking"]
    workers = [{"pid": p["pid"], "ppid": p["ppid"]} for p in all_processes() if WORKER_RE.search(p["cmdline"] or "")]
    rec["engine_workers"] = workers
    if workers:
        rec["blocking"].append(f"ワーカーのプロセスが残っている（{[w['pid'] for w in workers]}）")
    LOG(f"gate: port {OWNER_PORT} {'LISTENING' if listening else 'free'}; GPU used {gpu.get('memory_used_mib')} MiB "
        f"(bound {GPU_BUSY_MIB}); GPU processes {gpu.get('compute_apps')}; workers {workers or 'none'}")
    return rec


def gate_or_refuse(rec: dict, result_path: Path) -> bool:
    rec["gate"] = gate()
    if not rec["gate"]["blocking"] and not torch.cuda.is_available():
        rec["gate"]["blocking"].append("CUDA が使えない")
    if rec["gate"]["blocking"]:
        rec["status"] = "refused"
        rec["finished_at"] = now()
        write_json(result_path, rec)
        LOG(f"refused by the gate: {rec['gate']['blocking']}")
        LOG.close()
        return False
    rec["gpu_name"] = torch.cuda.get_device_name(0)
    return True


def begin(name: str, *, mode: str = "w") -> tuple[Path, dict]:
    out_dir = RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.open(out_dir / "run.log", mode=mode)
    install_log_bridge()
    rec = {"item": name, "script": rel(Path(__file__)), "started_at": now(), "status": "running",
           "git": git_state(), "versions": versions()}
    return out_dir, rec


# --------------------------------------------------------------------------- #
# WDDM sampler (calib.py's LUID discovery and vram_sampler2.ps1)
# --------------------------------------------------------------------------- #

LUID_PS = r"""
$s = Get-Counter -Counter "\GPU Adapter Memory(*)\Dedicated Usage" -MaxSamples 1
$rows = @()
foreach ($smp in $s.CounterSamples) {
  if ($smp.Path -match '\(([^)]+)\)') {
    $rows += [pscustomobject]@{ instance = $Matches[1]; mb = [math]::Round($smp.CookedValue / 1MB, 1) }
  }
}
ConvertTo-Json -Compress -InputObject @($rows | Sort-Object mb -Descending)
"""


def discover_luid() -> dict:
    """The adapter with the most dedicated memory in use (calib.py ``discover_luid``; not cached: the
    LUID changes across reboots)."""
    out = run_cmd(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", LUID_PS], timeout=180)
    rows = json.loads(out.stdout.strip()) if out.stdout.strip() else []
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        raise RuntimeError("no GPU Adapter Memory instances found")
    rows.sort(key=lambda r: float(r.get("mb") or 0.0), reverse=True)
    return {"chosen_luid": rows[0]["instance"], "chosen_dedicated_mb": rows[0]["mb"], "candidates": rows}


class WddmSampler:
    """Dedicated / shared GPU memory of one adapter, ~1 Hz or faster, into ``csv`` (calib.py ``Monitors``)."""

    def __init__(self, csv: Path, luid: str) -> None:
        self.csv, self.luid = csv, luid
        self.stop_file = csv.with_suffix(".stop")
        self.proc = None

    def __enter__(self):
        self.stop_file.unlink(missing_ok=True)
        self.proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SAMPLER),
             "-Luid", self.luid, "-Csv", str(self.csv), "-Stop", str(self.stop_file)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        time.sleep(3.0)          # a baseline before the work
        return self

    def __exit__(self, *exc) -> None:
        time.sleep(2.0)
        self.stop_file.write_text("stop", encoding="ascii")
        try:
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.proc.kill()

    def rows(self) -> list[dict]:
        out = []
        if not self.csv.is_file():
            return out
        for line in self.csv.read_text(encoding="ascii", errors="replace").splitlines():
            parts = line.strip().split(",")
            if len(parts) < 4 or parts[0].startswith("timestamp"):
                continue
            try:
                t = _dt.datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S.%f").timestamp()
                out.append({"t": t, "dedicated_mb": float(parts[1]), "shared_mb": float(parts[2])})
            except ValueError:
                continue
        return out


class NoSampler:
    """When the adapter could not be found: nothing is sampled (recorded as such)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def rows(self) -> list[dict]:
        return []


def sampler_for(csv: Path, luid_info: dict):
    luid = (luid_info or {}).get("chosen_luid")
    return WddmSampler(csv, luid) if luid else NoSampler()


def wddm_window(rows: list[dict], t0: float, t1: float, baseline: list[dict]) -> dict:
    win = [r for r in rows if t0 <= r["t"] <= t1]
    base_sh = statistics.median([r["shared_mb"] for r in baseline]) if baseline else None
    if not win:
        return {"samples": 0, "shared_baseline_mb": base_sh}
    sh = [r["shared_mb"] for r in win]
    de = [r["dedicated_mb"] for r in win]
    return {"samples": len(win), "dedicated_peak_mb": max(de), "dedicated_median_mb": statistics.median(de),
            "shared_peak_mb": max(sh),
            "shared_median_mb": statistics.median(sh), "shared_baseline_mb": base_sh,
            "shared_rise_mb": None if base_sh is None else round(max(sh) - base_sh, 1)}


# --------------------------------------------------------------------------- #
# The decoders
# --------------------------------------------------------------------------- #

def load_ref_module():
    """``NonCausalViTDecoder`` of the copied reference, by path (nothing else of that repository)."""
    spec = importlib.util.spec_from_file_location("ref_nvidia_decoder", str(REF_FILE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_state_dict(path) -> dict:
    """The FP16 file's 297 decoder tensors as float32, ``decoder.`` removed (the reference's own names)."""
    from engine.sft_quant.sft_reader import read_tensors

    info = SD.inspect_sol_file(path)
    out = {}
    for key, t in read_tensors(str(path), info.decoder_keys, info.header):
        out[key[len("decoder."):] if key.startswith("decoder.") else key] = t.float()
    return out, info


def load_ref(device):
    mod = load_ref_module()
    path = ROOT / SOL_FILES["fp16"]
    t0 = time.perf_counter()
    sd, info = ref_state_dict(path)
    model = mod.NonCausalViTDecoder(info.mean, info.std, mod.KEEP[24], time_norm=5)
    model.load_state_dict(sd, strict=True)
    del sd
    model = model.to(device).eval().requires_grad_(False)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return model, {"seconds": round(time.perf_counter() - t0, 2), "keys": 297}


@torch.inference_mode()
def ref_decode_float(model, latents: torch.Tensor, device) -> torch.Tensor:
    """The reference without autocast (float32 throughout): [1, 24, T, h, w] -> CPU float32 [F, H, W, 3] in [0, 1]."""
    z = latents.to(device).float().transpose(1, 2).contiguous()        # (1, T, 24, h, w)
    px = model(z)
    m = torch.tensor(VV.IMAGENET_MEAN, device=px.device).view(1, 3, 1, 1, 1)
    s = torch.tensor(VV.IMAGENET_STD, device=px.device).view(1, 3, 1, 1, 1)
    return (px.float() * s + m).clamp_(0.0, 1.0)[0].permute(1, 2, 3, 0).cpu()


def sol_path(config: SD.SolConfig) -> Path:
    return ROOT / SOL_FILES[config.weights]


def pin_cuda_quant() -> dict:
    """The product's pinning of comfy-kitchen to its CUDA backend (``require_cuda_quant``)."""
    from engine_h3.quant_linear import require_cuda_quant

    back = require_cuda_quant((h3_format.INT8,))
    return {k: ({kk: vv for kk, vv in v.items() if kk in ("available", "disabled")} if isinstance(v, dict) else str(v))
            for k, v in back.items()}


def load_sol(name: str, device) -> SD.SolDecoder:
    cfg = SD.CONFIGS[name]
    return SD.SolDecoder.from_file(sol_path(cfg), config=cfg, device=device)


def load_standard(device):
    t0 = time.perf_counter()
    vae = VV.H3VideoDecoder.from_file(str(ROOT / PATHS["video_vae"]), device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    return vae, {"seconds": round(time.perf_counter() - t0, 2)}


def decode_with(name: str, dec, latents: torch.Tensor, device) -> torch.Tensor:
    """CPU uint8 frames from either decoder; the standard one as the product calls it (pipeline_h3._decode)."""
    if name == "standard":
        return dec.decode(latents.to(device), tiled=True)
    return dec.decode(latents)


def cudnn_like_product() -> dict:
    """cuDNN as the product loads H3 (deterministic=True at load: services/engines/h3/adapter.py)."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {"cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark}


def tf32_off() -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


# --------------------------------------------------------------------------- #
# Latents
# --------------------------------------------------------------------------- #

def latent_path(item: str) -> Path:
    return RUNS / f"gen_{item}" / LATENT_FILE


def load_latent(item: str) -> torch.Tensor:
    from safetensors.torch import load_file

    return load_file(str(latent_path(item)))["video_latent"].float()


def random_latent(T: int, h: int, w: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn((1, 24, T, h, w), generator=g, dtype=torch.float32)


def fit_latent(src: torch.Tensor, T: int, h: int, w: int) -> torch.Tensor:
    """``src`` [1, 24, t, hh, ww] repeated along time / height / width and cut to (T, h, w)."""
    reps = (1, 1, -(-T // src.shape[2]), -(-h // src.shape[3]), -(-w // src.shape[4]))
    return src.repeat(*reps)[:, :, :T, :h, :w].contiguous()


# --------------------------------------------------------------------------- #
# Plan 5.3: the latents from the product's engine (one replacement, restored)
# --------------------------------------------------------------------------- #

_ORIG_DECODER = VV.H3VideoDecoder
GEN_STATE: dict = {}


class SaveLatentDecoder:
    """Stands in for ``H3VideoDecoder`` in ``H3Pipeline._decode``: builds the real decoder, saves the
    latents it is handed (float32, CPU) and decodes them with the real decoder (the product's mp4 is
    unchanged). One decode per job is expected (a chain decodes its stitched latents once)."""

    @classmethod
    def from_file(cls, path, *, device, dtype=torch.bfloat16):
        GEN_STATE["from_file"] = GEN_STATE.get("from_file", 0) + 1
        return cls(_ORIG_DECODER.from_file(path, device=device, dtype=dtype))

    def __init__(self, inner) -> None:
        self.inner = inner

    def to(self, *args, **kwargs):
        self.inner.to(*args, **kwargs)
        return self

    def decode(self, latents: torch.Tensor, *, tiled: bool = True) -> torch.Tensor:
        from safetensors.torch import save_file

        GEN_STATE["decode"] = GEN_STATE.get("decode", 0) + 1
        lat = latents.detach().float().cpu().contiguous()
        out = GEN_STATE["out_dir"] / (LATENT_FILE if GEN_STATE["decode"] == 1 else f"latent_{GEN_STATE['decode']}.safetensors")
        save_file({"video_latent": lat}, str(out), metadata={"item": GEN_STATE["item"], "seed": str(SEED)})
        GEN_STATE["latent"] = {"file": rel(out), "stats": tensor_stats(lat), "dtype_handed": str(latents.dtype),
                               "channel_std": [round(float(v), 5) for v in lat[0].flatten(1).std(dim=1)]}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        frames = self.inner.decode(latents, tiled=tiled)
        torch.cuda.synchronize()
        GEN_STATE["standard_decode_seconds"] = round(time.perf_counter() - t0, 2)
        GEN_STATE["tiled_arg"] = tiled
        return frames


@contextlib.contextmanager
def save_latent_patch():
    if VV.__dict__["H3VideoDecoder"] is not _ORIG_DECODER:
        raise RuntimeError("video_vae.H3VideoDecoder is not the product's own before patching")
    VV.H3VideoDecoder = SaveLatentDecoder
    try:
        yield
    finally:
        VV.H3VideoDecoder = _ORIG_DECODER


def patch_in() -> bool:
    from engine_h3.video_vae import H3VideoDecoder as late   # noqa: PLC0415 -- as _decode imports it

    return late is SaveLatentDecoder


def patch_restored() -> bool:
    return VV.__dict__["H3VideoDecoder"] is _ORIG_DECODER


def gen_settings(item: str, attention: str) -> dict:
    g = GEN[item]
    return {"seed": SEED, "steps": TURBO_STEPS if g["turbo"] else DEFAULTS["steps"],
            "shift_video": DEFAULTS["shift_video"], "shift_audio": DEFAULTS["shift_audio"],
            "width": g["width"], "height": g["height"],
            "num_frames": g["frames"] if g["kind"] != "chain" else f"{g['frames']} x {g['clips']}",
            "frame_rate": FRAME_RATE,
            "loras": f"Turbo LoRA {Path(TURBO_LORA_FILE).name} {TURBO_LORA_STRENGTH}" if g["turbo"] else "なし",
            "negative_prompt": "", "attention_backend": attention, "vram16_mode": DEFAULTS["vram16_mode"],
            "block_swap_prefetch": DEFAULTS["block_swap_prefetch"], "keep_resident": DEFAULTS["keep_resident"],
            "blocks_on_gpu": DEFAULTS["blocks_on_gpu"], "deterministic": DEFAULTS["deterministic"]}


REASONS = {
    "seed": "同じ潜在をすべての復号器で比べるため、種を固定する（pi-H3 の実験と同じ 1234）",
    "steps": "計画 5.3: Turbo LoRA の対照の条件（記録簿 §223.12）は 4 ステップ。c だけ操作パネルの既定の 20",
    "loras": "計画 5.3: a・b・d・e は Turbo LoRA 0.8（記録簿 §223.12 の対照の条件）。c は既定のなし",
    "width": "計画 5.3: d は素材のイラスト（896×1152）の大きさ、e は連結の実験の 640×384",
    "height": "計画 5.3: d は素材のイラスト（896×1152）の大きさ、e は連結の実験の 640×384",
    "num_frames": "計画 5.3: ComfyUI の実測の条件（124）と比べやすくし GPU の時間を抑える。c は既定の 345",
    "attention_backend": "製品の既定（sdpa）。--attention sage で変えられる（変えたら記録に残る）",
}


def conditions_table(attention: str) -> list[dict]:
    rows = []
    for key, dv in DEFAULTS.items():
        vals = {it: gen_settings(it, attention)[key] for it in GEN}
        differs = any(str(v) != str(dv) for v in vals.values())
        rows.append({"項目": key, "製品の既定": dv, "出どころ": DEFAULT_SOURCES[key], "この実験の値": vals,
                     "違う理由": (REASONS.get(key) or "（理由の記載なし）") if differs else "同じ"})
    rows.append({"項目": "重み・テキストエンコーダ・動画VAE・音声VAE", "製品の既定": "マニフェストの default_file",
                 "出どころ": "scripts/manifests/40-minimaxh3.json", "この実験の値": {k: Path(v).name for k, v in PATHS.items()},
                 "違う理由": "同じ（計画 5.3: FL2VA 版の INT8 ConvRot）"})
    rows.append({"項目": "生成の種類・入力", "製品の既定": "T2V", "出どころ": "操作パネル",
                 "この実験の値": {it: GEN[it]["kind"] + (f"（先頭の画像 {Path(ILLUSTRATION).name}）" if GEN[it]["kind"] == "i2v" else "")
                             for it in GEN},
                 "違う理由": "計画 5.3: d は I2V（記録簿 §224.12 の縦長のイラスト）、e は連結 2 本（共通のプロンプト）"})
    return rows


def table_markdown(rows: list[dict]) -> str:
    lines = ["| 項目 | 製品の既定（出どころ） | この実験の値（a / b / c / d / e） | 違う場合の理由 |", "|---|---|---|---|"]
    for r in rows:
        vals = r["この実験の値"]
        shown = " / ".join(f"{k}: {v}" for k, v in vals.items())
        lines.append(f"| {r['項目']} | {r['製品の既定']}（{r['出どころ']}） | {shown} | {r['違う理由']} |")
    return "\n".join(lines)


def gen_request(item: str, out_dir: Path, attention: str):
    g = GEN[item]
    loras = (P.H3LoraInput(path=str(ROOT / TURBO_LORA_FILE), strength=TURBO_LORA_STRENGTH),) if g["turbo"] else ()
    images = (P.H3ImageInput(path=str(ROOT / ILLUSTRATION), frame_idx=0),) if g["kind"] == "i2v" else ()
    base = P.H3GenerateRequest(
        prompt=PROMPTS[item], seed=SEED, width=g["width"], height=g["height"], num_frames=g["frames"],
        output_path=str(out_dir / "output.mp4"), num_inference_steps=TURBO_STEPS if g["turbo"] else DEFAULTS["steps"],
        shift_video=DEFAULTS["shift_video"], shift_audio=DEFAULTS["shift_audio"], images=images,
        keep_resident=DEFAULTS["keep_resident"], attention_backend=attention,
        block_swap_prefetch=DEFAULTS["block_swap_prefetch"], vram16_mode=DEFAULTS["vram16_mode"], loras=loras)
    if g["kind"] != "chain":
        return base
    from engine_h3.chain_h3 import H3ChainClipSpec, H3ChainSpec

    clips = tuple(H3ChainClipSpec(prompt=PROMPTS[item], num_frames=g["frames"]) for _ in range(g["clips"]))
    return H3ChainSpec(base=base, clips=clips, seed=SEED)


def check_request(item: str, req) -> dict:
    """The product's own refusals, CPU only (validate_request; the chain's _check_spec and clip requests)."""
    if GEN[item]["kind"] != "chain":
        P.validate_request(req)
        return {"validate": "passed", "expected_latents": latent_frames(req.num_frames), "frames": req.num_frames}
    from engine_h3.chain_h3 import _check_spec, clip_requests
    from h3_chain_math import compute_h3_chain_layout

    _check_spec(req)
    layout = compute_h3_chain_layout([c.num_frames for c in req.clips], retake=False)
    for r in clip_requests(req):
        P.validate_request(r)
    return {"validate": "passed", "expected_latents": int(layout.total_latent_frames),
            "frames": int(layout.total_frames), "layout": layout.to_dict()}


def run_gen(item: str, attention: str) -> int:
    out_dir, rec = begin(f"gen_{item}")
    result_path = out_dir / "result.json"
    rec["title"] = GEN_TITLES[item]
    rec["conditions"] = {**gen_settings(item, attention), "kind": GEN[item]["kind"], "prompt": PROMPTS[item],
                         "paths": {k: file_info(v) for k, v in PATHS.items()},
                         "image": file_info(ILLUSTRATION) if GEN[item]["kind"] == "i2v" else None}
    if GEN[item]["kind"] == "i2v" and not (ROOT / ILLUSTRATION).is_file():
        rec.update(status="refused", material_missing=ILLUSTRATION, finished_at=now())
        write_json(result_path, rec)
        LOG(f"素材なし: {ILLUSTRATION}")
        return 2
    if not gate_or_refuse(rec, result_path):
        return 2
    ok = False
    try:
        req = gen_request(item, out_dir, attention)
        rec["request_check"] = check_request(item, req)
        rec["request"] = dataclasses.asdict(req)
        GEN_STATE.clear()
        GEN_STATE.update(item=item, out_dir=out_dir)
        write_json(result_path, rec)
        paths = P.H3Paths(**{k: str(ROOT / v) for k, v in PATHS.items()})
        options = P.H3LoadOptions(blocks_on_gpu=DEFAULTS["blocks_on_gpu"], attention_backend=attention,
                                  block_swap_prefetch=DEFAULTS["block_swap_prefetch"],
                                  deterministic=DEFAULTS["deterministic"])
        with save_latent_patch():
            rec["patch_in"] = patch_in()
            pipe = P.H3Pipeline(paths, options, device=torch.device("cuda"), progress=LOG.progress, log=LOG)
            try:
                t0 = time.perf_counter()
                rec["build_report"] = pipe.load()
                rec["load_seconds"] = round(time.perf_counter() - t0, 2)
                write_json(result_path, rec)
                if GEN[item]["kind"] == "chain":
                    from engine_h3.chain_h3 import run_chain_h3

                    result = run_chain_h3(pipe, req, LOG.progress, LOG)
                else:
                    result = pipe.generate(req)
                rec["engine"] = {"seconds": result.get("seconds"), "peak_vram_mb": result.get("peak_vram_mb"),
                                 "peak_vram_reserved_mb": result.get("peak_vram_reserved_mb"), "full": result}
            finally:
                try:
                    pipe.close()
                except Exception as exc:  # noqa: BLE001
                    LOG(f"pipe.close failed (ignored): {exc!r}")
        rec["restored"] = patch_restored()
        rec["standard_decoder"] = {k: GEN_STATE.get(k) for k in ("from_file", "decode", "standard_decode_seconds", "tiled_arg")}
        rec["latent"] = GEN_STATE.get("latent")
        out_mp4 = out_dir / "output.mp4"
        rec["outputs"] = {"mp4": rel(out_mp4), "probe": ffprobe(out_mp4) if out_mp4.is_file() else None}
        bad = []
        if GEN_STATE.get("decode") != 1:
            bad.append(f"decode called {GEN_STATE.get('decode')} times (1 expected)")
        shape = (rec.get("latent") or {}).get("stats", {}).get("shape")
        g = GEN[item]
        want = [1, 24, rec["request_check"]["expected_latents"], g["height"] // 16, g["width"] // 16]
        if shape != want:
            bad.append(f"latent shape {shape} != {want}")
        loras = (((rec.get("engine") or {}).get("full") or {}).get("h3") or {}).get("loras") or []
        if g["turbo"] and (not loras or loras[0].get("pairs_applied") != TURBO_LORA_PAIRS):
            bad.append(f"Turbo LoRA pairs applied {[r.get('pairs_applied') for r in loras]} != {TURBO_LORA_PAIRS}")
        if not rec["restored"] or not rec.get("patch_in"):
            bad.append("the replacement was not in / not restored")
        rec["self_check"] = {"passed": not bad, "failures": bad}
        rec["status"] = "done"
        ok = not bad
        LOG(f"gen {item}: done; self_check={rec['self_check']}")
    except BaseException as exc:  # noqa: BLE001 -- an interruption is recorded too
        failure(rec, exc)
        rec["restored"] = patch_restored()
    finally:
        rec["finished_at"] = now()
        write_json(result_path, rec)
        LOG.close()
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# GPU self-check: the comfy-kitchen functions against the torch formulas
# --------------------------------------------------------------------------- #

#: The kitchen rope vs the torch formula (fp16): normalized values are at most |x| = sqrt(64) = 8, where one
#: fp16 step is 2^-7; a rotation that cancels to near zero can differ by a step of its inputs, which is far more
#: than one step of the (small) result. So: max |diff| <= 2^-8 (one step at [4, 8)) and at most 0.1 % of the
#: elements different. A layout error (a wrong half, angle or table) gives differences of order 1.
ROPE_MAX_ABS = 2.0 ** -8
ROPE_MAX_FRAC = 1e-3


def run_selfcheck() -> int:
    out_dir, rec = begin("selfcheck")
    result_path = out_dir / "result.json"
    if not gate_or_refuse(rec, result_path):
        return 2
    ok = False
    try:
        import comfy_kitchen as ck
        from engine.sft_quant.sft_reader import read_tensors
        import sft_quant_format

        dev = torch.device("cuda")
        rec["backends"] = pin_cuda_quant()
        rec["int8_attention_is_available"] = bool(ck.int8_attention_is_available())
        g = torch.Generator(device=dev).manual_seed(11)
        checks = {}
        # 1. rms_rope_split_half_ (fp16, the fp16 table and q_scale the decoder passes) vs rms_rope_torch
        #    (the same fp16-rounded cos / sin), a 1280 x 768 block, both shifts, and the identity table
        h, w = 768 // 16, 1280 // 16
        ones16 = torch.ones(64, device=dev, dtype=torch.float16)
        for shifted in (False, True):
            cos, sin = (t.half().float().to(dev) for t in SD.rope_cos_sin(h, w, shifted, 48))
            n = cos.shape[0]
            q = (torch.randn(n, 32, 64, generator=g, device=dev) * 3).half()
            k = (torch.randn(n, 32, 64, generator=g, device=dev) * 3).half()
            qt, kt = SD.rms_rope_torch(q, cos, sin), SD.rms_rope_torch(k, cos, sin)
            qk, kk = q.clone(), k.clone()
            ck.rms_rope_split_half_(qk.view(1, n, 32, 64), kk.view(1, n, 32, 64), SD.kitchen_freqs(cos, sin).half(),
                                    ones16, None, SD.EPS, 48)
            dq, dk = (qk.float() - qt.float()).abs(), (kk.float() - kt.float()).abs()
            frac = float(((qk != qt).float().mean() + (kk != kt).float().mean()) / 2)
            checks[f"rms_rope_{'shifted' if shifted else 'plain'}"] = {
                "rows": n, "q_max_abs": float(dq.max()), "k_max_abs": float(dk.max()),
                "differing_fraction": frac, "elements": int(qk.numel()),
                "within_rounding": bool(max(float(dq.max()), float(dk.max())) <= ROPE_MAX_ABS and frac <= ROPE_MAX_FRAC)}
        one = torch.ones(5, 24, device=dev)
        q = (torch.randn(5, 32, 64, generator=g, device=dev) * 3).half()
        k = (torch.randn(5, 32, 64, generator=g, device=dev) * 3).half()
        qt, kt = SD.rms_rope_torch(q, None, None), SD.rms_rope_torch(k, None, None)
        ck.rms_rope_split_half_(q.view(1, 5, 32, 64), k.view(1, 5, 32, 64),
                                SD.kitchen_freqs(one, torch.zeros_like(one)).half(), ones16, None, SD.EPS, 48)
        d = max(float((q.float() - qt.float()).abs().max()), float((k.float() - kt.float()).abs().max()))
        frac = float(((q != qt).float().mean() + (k != kt).float().mean()) / 2)
        checks["rms_only_identity_table"] = {"max_abs": d, "differing_fraction": frac,
                                             "within_rounding": d <= ROPE_MAX_ABS and frac <= ROPE_MAX_FRAC}
        # 2. int8_linear with the fused arguments vs the same steps applied separately, block 0 of the INT8 file
        p8 = ROOT / SOL_FILES["int8"]
        p16 = ROOT / SOL_FILES["fp16"]
        info8 = SD.inspect_sol_file(p8)
        h16 = sft_quant_format.read_header(str(p16))
        names = [f"decoder.blocks.0.{leaf}.{s}" for leaf in SD._INT8_LEAVES for s in ("weight", "weight_scale", "bias")]
        names += [f"decoder.blocks.0.{n}" for n in ("norm1.weight", "norm2.weight", "scale1", "scale2")]
        t8 = {k: v.to(dev) for k, v in read_tensors(str(p8), names, info8.header)}
        t16 = {k: v.to(dev) for k, v in read_tensors(str(p16), [n for n in names if not n.endswith("weight_scale")], h16)}
        spec = info8.quant["blocks.0.attn.to_qkv"]
        L = "decoder.blocks.0."
        x = torch.randn(4096, 2048, generator=g, device=dev).half()
        res = torch.randn(4096, 2048, generator=g, device=dev).half()

        def lin8(leaf, inp, **kw):
            return ck.int8_linear(inp.contiguous(), t8[L + leaf + ".weight"], t8[L + leaf + ".weight_scale"],
                                  t8[L + leaf + ".bias"], out_dtype=torch.float16, convrot=bool(spec.convrot),
                                  convrot_groupsize=int(spec.convrot_groupsize), **kw)

        def lin16(leaf, inp):
            return torch.nn.functional.linear(inp, t16[L + leaf + ".weight"], t16[L + leaf + ".bias"])

        def cmp(a, b):
            d = (a.float() - b.float())
            return {"max_abs": float(d.abs().max()), "rel_l2": float(d.norm() / b.float().norm())}

        rms1 = torch.nn.functional.rms_norm(x, (2048,), weight=t8[L + "norm1.weight"], eps=SD.EPS)
        fused = lin8("attn.to_qkv", x, input_act="rms_norm", input_act_weight=t8[L + "norm1.weight"], input_act_eps=SD.EPS)
        checks["int8_qkv_fused_vs_separate"] = cmp(fused, lin8("attn.to_qkv", rms1))
        checks["int8_qkv_vs_fp16_file"] = cmp(fused, lin16("attn.to_qkv", rms1))
        fused = lin8("attn.to_out", x, residual=res, residual_scale=t8[L + "scale1"])
        sep = torch.addcmul(res, lin8("attn.to_out", x), t8[L + "scale1"])
        checks["int8_out_fused_residual_vs_addcmul"] = cmp(fused, sep)
        a = lin8("ff.w1", x, input_act="rms_norm", input_act_weight=t8[L + "norm2.weight"], input_act_eps=SD.EPS)
        gate_, up = a.chunk(2, dim=-1)
        fused = lin8("ff.w2", a, input_act="swiglu", residual=res, residual_scale=t8[L + "scale2"])
        sep = torch.addcmul(res, lin8("ff.w2", torch.nn.functional.silu(gate_) * up), t8[L + "scale2"])
        checks["int8_w2_fused_swiglu_residual_vs_separate"] = cmp(fused, sep)
        sep16 = torch.addcmul(res, lin16("ff.w2", torch.nn.functional.silu(gate_) * up), t16[L + "scale2"])
        checks["int8_w2_vs_fp16_file"] = cmp(fused, sep16)
        del t8, t16
        # 3. int8_attention vs SDPA on a window batch of the real shape (64 windows, 1280 queries, 1290 keys)
        if rec["int8_attention_is_available"]:
            q = torch.randn(64, 32, 1280, 64, generator=g, device=dev).half()
            k = torch.randn(64, 32, 1290, 64, generator=g, device=dev).half()
            v = torch.randn(64, 32, 1290, 64, generator=g, device=dev).half()
            checks["int8_attention_vs_sdpa"] = cmp(ck.int8_attention(q, k, v),
                                                   torch.nn.functional.scaled_dot_product_attention(q, k, v))
            del q, k, v
        rec["checks"] = checks
        rec["peak"] = peaks()
        # judged: the kitchen rope within 1 ulp of the torch formula, the int8 layer close to the FP16 one,
        # int8_attention available (else config C cannot run). Recorded only: fused vs separate, int8 vs float attention.
        rec["rope_bounds"] = {"max_abs": ROPE_MAX_ABS, "max_differing_fraction": ROPE_MAX_FRAC}
        rec["judged"] = {"rope_within_rounding": all(v.get("within_rounding", True) for v in checks.values()),
                         "int8_qkv_vs_fp16_rel_l2_below_0.05": checks["int8_qkv_vs_fp16_file"]["rel_l2"] < 0.05,
                         "int8_attention_is_available": rec["int8_attention_is_available"]}
        ok = all(rec["judged"].values())
        rec["passed"] = ok
        rec["status"] = "done"
        LOG(f"selfcheck: passed={ok}\n{jdump(checks)}")
    except BaseException as exc:  # noqa: BLE001
        failure(rec, exc)
    finally:
        rec["finished_at"] = now()
        write_json(result_path, rec)
        LOG.close()
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Plan 5.2: the cross-check against NVIDIA's reference
# --------------------------------------------------------------------------- #

def verify_latents(source: str) -> dict:
    out = {}
    for i, (name, (T, h, w)) in enumerate(VERIFY.items()):
        if source == "random":
            out[name] = random_latent(T, h, w, seed=100 + i)
        else:
            src = load_latent(source)
            out[name] = src[:, :, :T, :h, :w].contiguous() if all(
                a >= b for a, b in zip(src.shape[2:], (T, h, w))) else fit_latent(src, T, h, w)
    return out


def run_verify(conds: list[str], source: str) -> int:
    out_dir, rec = begin("verify")
    result_path = out_dir / "result.json"
    rec["conditions"] = {c: {"latent_grid": list(VERIFY[c]), "frames": SD.frames_for_latents(VERIFY[c][0]),
                             "pixels": [VERIFY[c][2] * 16, VERIFY[c][1] * 16]} for c in conds}
    rec["latent_source"] = source if source == "random" else f"gen_{source} の潜在（切り出し／並べ）"
    rec["pass_db"] = VERIFY_PASS_DB
    rec["a_floor_db"] = VERIFY_A_FLOOR_DB
    if source != "random" and not latent_path(source).is_file():
        rec.update(status="refused", missing=rel(latent_path(source)), finished_at=now())
        write_json(result_path, rec)
        return 2
    sc = read_json(RUNS / "selfcheck" / "result.json", {})
    rec["selfcheck"] = {"file": rel(RUNS / "selfcheck" / "result.json"), "passed": sc.get("passed"),
                        "finished_at": sc.get("finished_at"), "git_head": (sc.get("git") or {}).get("head")}
    if sc.get("passed") is not True:
        rec.update(status="refused", finished_at=now(),
                   refused_because="--selfcheck が合格していない（CUDA 版の comfy_kitchen の関数が確かめられていない）")
        write_json(result_path, rec)
        LOG(f"refused: {rec['refused_because']}")
        LOG.close()
        return 2
    if not gate_or_refuse(rec, result_path):
        return 2
    ok = False
    dev = torch.device("cuda")
    try:
        rec["precision"] = tf32_off()
        lats = verify_latents(source)
        rec["latents"] = {c: tensor_stats(lats[c]) for c in conds}
        outs: dict = {"ref": {}, "F32": {}, "A": {}}
        runs: dict = {c: {} for c in conds}

        def run_all(label, loader, fn):
            free_cuda()
            torch.cuda.reset_peak_memory_stats()
            try:
                model, info = loader()
            except torch.cuda.OutOfMemoryError as exc:
                for c in conds:
                    runs[c][label] = {"result": "メモリ不足（読み込み）", "error": repr(exc)[:400]}
                return
            except Exception as exc:  # noqa: BLE001 -- recorded; the other decoders still run
                for c in conds:
                    runs[c][label] = {"result": "failed（読み込み）", "error": repr(exc)[:600],
                                      "traceback": traceback.format_exc()}
                LOG(f"verify {label} load FAILED: {exc!r}\n{traceback.format_exc()}")
                free_cuda()
                return
            rec.setdefault("loads", {})[label] = {**info, "after_load_alloc_mib": mib(torch.cuda.memory_allocated())}
            for c in conds:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                try:
                    outs[label][c] = fn(model, lats[c])
                    torch.cuda.synchronize()
                    runs[c][label] = {"result": "done", "seconds": round(time.perf_counter() - t0, 2), "peak": peaks(),
                                      "frames": int(outs[label][c].shape[0])}
                except torch.cuda.OutOfMemoryError as exc:
                    runs[c][label] = {"result": "メモリ不足", "error": repr(exc)[:400], "peak": peaks()}
                except Exception as exc:  # noqa: BLE001 -- recorded; the next condition still runs
                    runs[c][label] = {"result": "failed", "error": repr(exc)[:600], "traceback": traceback.format_exc()}
                free_cuda()
                LOG(f"verify {label} {c}: {runs[c][label]}")
                write_json(result_path, {**rec, "runs": runs})
            del model
            free_cuda()

        run_all("ref", lambda: load_ref(dev), lambda m, z: ref_decode_float(m, z, dev))
        run_all("F32", lambda: (lambda d: (d, d.load_report))(load_sol("F32", dev)),
                lambda d, z: d.decode(z, output="float"))
        rec["backends"] = pin_cuda_quant()
        run_all("A", lambda: (lambda d: (d, d.load_report))(load_sol("A", dev)),
                lambda d, z: d.decode(z, output="float"))
        def clipped(t):
            """Share of the [0, 1] pixels sitting on 0 or 1 after the clamp (the comparison cannot see past them)."""
            return {"at_0": round(float((t == 0).float().mean()), 6), "at_1": round(float((t == 1).float().mean()), 6)}

        cmp, failures = {}, []
        for c in conds:
            r = outs["ref"].get(c)
            row = {"clipped": {k: clipped(v[c]) for k, v in outs.items() if v.get(c) is not None}}
            for label in ("F32", "A"):
                m = outs[label].get(c)
                if r is None or m is None:
                    row[label] = None
                    continue
                if m.shape != r.shape:
                    row[label] = {"error": f"shape {list(m.shape)} vs ref {list(r.shape)}"}
                    failures.append(f"{c} {label}: 形が合わない {list(m.shape)} / {list(r.shape)}")
                    continue
                u8 = SM.psnr_u8((m * 255).round().to(torch.uint8), (r * 255).round().to(torch.uint8))
                row[label] = {"float": SM.psnr_float(m, r), "uint8": u8}
            cmp[c] = row
        rec["runs"], rec["compare_vs_ref"] = runs, cmp
        judged = [c for c in conds if (cmp[c].get("F32") or {}).get("float")]
        rec["judged_conditions"] = judged
        for c in judged:
            if cmp[c]["F32"]["float"]["psnr_db"] < VERIFY_PASS_DB:
                failures.append(f"{c} F32: {cmp[c]['F32']['float']['psnr_db']} dB < {VERIFY_PASS_DB}")
            a = (cmp[c].get("A") or {}).get("float")
            if a is not None and a["psnr_db"] < VERIFY_A_FLOOR_DB:
                failures.append(f"{c} A: {a['psnr_db']} dB < 目安の下限 {VERIFY_A_FLOOR_DB}")
        for c in conds:
            for label, r in runs[c].items():
                if str(r.get("result", "")).startswith("failed"):
                    failures.append(f"{c} {label}: {r['result']} {r.get('error')}")
        if not judged:
            failures.append("比べられた条件が 1 つも無い")
        rec["failures"] = failures
        ok = not failures
        rec["passed"] = ok
        rec["status"] = "done"
        LOG(f"verify: passed={ok}; {jdump(cmp)}")
    except BaseException as exc:  # noqa: BLE001
        failure(rec, exc)
    finally:
        rec["finished_at"] = now()
        write_json(result_path, rec)
        LOG.close()
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Plan 5.4 / 5.5: the standard decoder and A, B, C on a saved latent
# --------------------------------------------------------------------------- #

def run_compare(item: str, configs: list[str]) -> int:
    from engine_h3.media_out import write_mp4

    out_dir, rec = begin(f"compare_{item}")
    result_path = out_dir / "result.json"
    rec["configs"] = {"standard": "いまの 36 層の復号器（標準の FP16 を bf16 で計算・タイル on）",
                      **{c: SD.CONFIGS[c].label() for c in configs}}
    if not latent_path(item).is_file():
        rec.update(status="refused", missing=rel(latent_path(item)), finished_at=now())
        write_json(result_path, rec)
        LOG(f"潜在がありません: {rel(latent_path(item))}（先に --gen {item}）")
        return 2
    if not gate_or_refuse(rec, result_path):
        return 2
    ok = False
    dev = torch.device("cuda")
    try:
        lat = load_latent(item)
        T = int(lat.shape[2])
        bf = SD.D.block_frames(T)
        rec["latent"] = tensor_stats(lat)
        rec["frames_expected"] = sum(bf)
        rec["backends"] = pin_cuda_quant()
        rec["cudnn"] = cudnn_like_product()
        frames: dict = {}
        rec["decodes"] = {}
        for name in ["standard", *configs]:
            # load + decode of one decoder in one try: a failure is recorded and the next decoder goes on
            dec, info, d = None, None, {}
            try:
                free_cuda()
                torch.cuda.reset_peak_memory_stats()
                if name == "standard":
                    dec, info = load_standard(dev)
                else:
                    dec = load_sol(name, dev)
                    info = dec.load_report
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                fr = decode_with(name, dec, lat, dev)
                torch.cuda.synchronize()
                d = {"result": "done", "seconds": round(time.perf_counter() - t0, 2), "peak": peaks(), "load": info,
                     "frames": int(fr.shape[0])}
                if name != "standard":
                    d["stats"] = dec.last_stats
                if int(fr.shape[0]) != rec["frames_expected"]:
                    d["result"] = "failed"
                    d["error"] = f"{fr.shape[0]} frames, {rec['frames_expected']} expected"
                else:
                    frames[name] = fr
                    mp4 = out_dir / f"{name}.mp4"
                    write_mp4(str(mp4), fr, fps=FRAME_RATE, audio=None)
                    d["mp4"] = rel(mp4)
            except torch.cuda.OutOfMemoryError as exc:
                d = {"result": "メモリ不足" if info is not None else "メモリ不足（読み込み）", "error": repr(exc)[:400],
                     "peak": peaks(), "load": info}
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                d = {"result": "failed", "error": repr(exc)[:600], "traceback": traceback.format_exc(), "load": info}
                LOG(f"compare {item} {name} FAILED: {exc!r}\n{d['traceback']}")
            finally:
                del dec
                free_cuda()
            rec["decodes"][name] = d
            LOG(f"compare {item} {name}: {d.get('result')} {d.get('seconds')} s peak {d.get('peak')}")
            write_json(result_path, rec)
        names = list(frames)
        Fn = frames[names[0]].shape[0] if names else 0
        stride = max(1, Fn // 96)
        m: dict = {"ssim_stride": stride, "vs_standard": {}, "pairs": {}, "grid_edges": {}, "flicker": {}}
        if "standard" in frames:
            for n in names:
                if n != "standard":
                    m["vs_standard"][n] = {**SM.psnr_u8(frames[n], frames["standard"]),
                                           **SM.ssim_luma(frames[n], frames["standard"], stride=stride)}
        sol = [n for n in names if n != "standard"]
        for i, a in enumerate(sol):
            for b in sol[i + 1:]:
                m["pairs"][f"{a}_vs_{b}"] = SM.psnr_u8(frames[a], frames[b])
        for n in names:
            m["grid_edges"][n] = SM.grid_edges(frames[n], stride=stride)
            m["flicker"][n] = SM.flicker(frames[n], bf)
        rec["metrics"] = m
        if names:
            H, W = frames[names[0]].shape[1:3]
            centres = SM.grid_centres(W, H)
            fidx = sorted({0, Fn // 2, max(0, Fn - 1)})
            labels = [f"f{f} x{cx} y{cy}" for f in fidx for cx, cy in centres]
            rows = [(n, SM.grid_crops(frames[n], centres, fidx)) for n in names]
            rec["sheet"] = rel(Path(SM.save_sheet(out_dir / f"{item}_grid128_crops.png", rows, labels)))
            rec["sheet_note"] = "各切り出しの中心（48 画素目）が 128 画素の倍数の縦横の線の交点。最近傍で 4 倍に拡大"
        rec["status"] = "done"
        ok = len(frames) == 1 + len(configs)
    except BaseException as exc:  # noqa: BLE001
        failure(rec, exc)
    finally:
        rec["finished_at"] = now()
        write_json(result_path, rec)
        LOG.close()
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Plan 5.6: speed and VRAM
# --------------------------------------------------------------------------- #

def bench_latent(point: str, source: str, cache: dict) -> tuple[torch.Tensor, dict]:
    w, h, frames, _std = BENCH[point]
    T, lh, lw = latent_frames(frames), h // 16, w // 16
    if source == "random":
        return random_latent(T, lh, lw, seed=7), {"source": "乱数（正規分布・種 7）", "grid": [T, lh, lw]}
    if "a" not in cache:
        cache["a"] = load_latent("a")
    src = cache["a"]
    return fit_latent(src, T, lh, lw), {"source": "潜在 a を時間・空間の方向に並べて切りそろえたもの",
                                        "a_grid": list(src.shape[2:]), "grid": [T, lh, lw]}


def run_bench(points: list[str], configs: list[str], source: str, force: bool) -> int:
    out_dir, rec = begin("bench", mode="a")
    result_path = out_dir / "result.json"
    data = read_json(result_path, {"points": {}, "loads": {}, "sessions": []})
    if source != "random" and not latent_path("a").is_file():
        LOG(f"潜在 a がありません: {rel(latent_path('a'))}（先に --gen a、または --latent-source random）")
        return 2
    sess = {"started_at": now(), "git": rec["git"], "versions": rec["versions"], "points": points,
            "configs": configs, "latent_source": source}
    if not gate_or_refuse(sess, out_dir / "refused.json"):
        return 2
    data["sessions"].append(sess)
    try:
        sess["luid"] = discover_luid()
    except Exception as exc:  # noqa: BLE001
        sess["luid"] = {"error": repr(exc)}
    write_json(result_path, data)
    dev = torch.device("cuda")
    sess["backends"] = pin_cuda_quant()
    sess["cudnn"] = cudnn_like_product()
    cache: dict = {}
    files_read: set = set()
    ok = True
    for name in configs:
        todo = [p for p in points if (name != "standard" or BENCH[p][3])
                and (force or (data["points"].get(f"{name}/{p}") or {}).get("result") not in ("done", "メモリ不足"))]
        if not todo:
            continue
        file = PATHS["video_vae"] if name == "standard" else SOL_FILES[SD.CONFIGS[name].weights]
        loads = []
        dec = None
        try:
            for k in range(2):              # first read, then a second read of the same file (warm)
                free_cuda()
                warm = file in files_read
                t0 = time.perf_counter()
                dec = load_standard(dev)[0] if name == "standard" else load_sol(name, dev)
                torch.cuda.synchronize()
                loads.append({"seconds": round(time.perf_counter() - t0, 2),
                              "file_read_earlier_in_this_process": warm,
                              "cache": "温まっている（この台本の中で同じファイルを読んだ直後）" if warm else
                              "未確認（この台本の中では初めての読み込み。直前にほかで読まれていれば温まっている）",
                              "alloc_after_mib": mib(torch.cuda.memory_allocated())})
                files_read.add(file)
                if k == 0:
                    del dec
                    dec = None
            data["loads"][f"{name}@{sess['started_at']}"] = {"file": Path(file).name, "loads": loads}
            weights_mib = loads[-1]["alloc_after_mib"]
            for p in todo:
                w, h, frames, _ = BENCH[p]
                lat, lat_info = bench_latent(p, source, cache)
                r = {"config": name, "point": p, "width": w, "height": h, "frames": frames, "latent": lat_info,
                     "at": now(), "weights_alloc_mib": weights_mib,
                     "comfy_estimate_mib_fp16_without_weights": round(SD.comfy_memory_estimate_mib(*lat_info["grid"]), 1)}
                csv = out_dir / f"wddm_{name}_{p}.csv"
                runs = []
                try:
                    with sampler_for(csv, sess["luid"]) as smp:
                        t_base = time.time()
                        for i in range(BENCH_RUNS):
                            free_cuda()
                            torch.cuda.reset_peak_memory_stats()
                            ta = time.time()
                            t0 = time.perf_counter()
                            fr = decode_with(name, dec, lat, dev)
                            torch.cuda.synchronize()
                            sec = time.perf_counter() - t0
                            runs.append({"seconds": round(sec, 3), "peak": peaks(), "wall": [ta, time.time()],
                                         "frames": int(fr.shape[0])})
                            if name != "standard":
                                runs[-1]["stats"] = dec.last_stats
                            del fr
                            LOG(f"bench {name} {p} run {i}: {sec:.2f} s peak {runs[-1]['peak']}")
                    srows = smp.rows()
                    base = [x for x in srows if x["t"] < t_base]
                    for x in runs:
                        x["wddm"] = wddm_window(srows, x["wall"][0], x["wall"][1], base)
                    # the whole point (every run, first included): its max and median (the CSV is kept)
                    r["wddm_whole_point"] = wddm_window(srows, runs[0]["wall"][0], runs[-1]["wall"][1], base)
                    kept = runs[1:]
                    r.update(result="done", runs=runs, discarded_first=True,
                             median_seconds=round(statistics.median(x["seconds"] for x in kept), 3),
                             peak_alloc_mib=max(x["peak"]["alloc_mib"] for x in kept),
                             peak_reserved_mib=max(x["peak"]["reserved_mib"] for x in kept),
                             wddm_dedicated_peak_mb=max((x["wddm"].get("dedicated_peak_mb") or 0) for x in runs),
                             wddm_shared_rise_mb=max((x["wddm"].get("shared_rise_mb") or 0) for x in runs),
                             frames_ok=all(x["frames"] == frames for x in runs))
                    r["fits_comfort_alloc"] = r["peak_reserved_mib"] <= COMFORT_ALLOC_MIB
                except torch.cuda.OutOfMemoryError as exc:
                    r.update(result="メモリ不足", error=repr(exc)[:400], runs=runs, peak_at_oom=peaks())
                except BaseException as exc:  # noqa: BLE001
                    failure(r, exc)
                    r["result"] = "failed"
                    ok = False
                    if isinstance(exc, KeyboardInterrupt):
                        data["points"][f"{name}/{p}"] = r
                        write_json(result_path, data)
                        raise
                finally:
                    free_cuda()
                r["wddm_csv"] = rel(csv)
                data["points"][f"{name}/{p}"] = r
                write_json(result_path, data)
                LOG(f"bench {name} {p}: {r.get('result')} median {r.get('median_seconds')} s "
                    f"peak {r.get('peak_alloc_mib')}/{r.get('peak_reserved_mib')} MiB shared+{r.get('wddm_shared_rise_mb')}")
        except torch.cuda.OutOfMemoryError as exc:
            data["loads"][f"{name}@{sess['started_at']}"] = {"file": Path(file).name, "loads": loads,
                                                             "result": "メモリ不足（読み込み）", "error": repr(exc)[:400]}
        finally:
            del dec
            free_cuda()
            write_json(result_path, data)
    sess["finished_at"] = now()
    write_json(result_path, data)
    LOG.close()
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Dry run (CPU only)
# --------------------------------------------------------------------------- #

def _tiny_ref_pair(mod, T_shift_check: bool = True) -> dict:
    """A small random decoder (dim 128, 2 heads of 64, 4 layers) built as NVIDIA's reference and loaded
    into :class:`sol_decoder.SolModel` under the file's key names; both run float32 on the CPU."""
    g = torch.Generator().manual_seed(1)
    dims = SD.SolDims(dim=128, heads=2, head_dim=64, num_layers=4)
    mean = torch.randn(24, generator=g).tolist()
    std = (torch.rand(24, generator=g) + 0.5).tolist()
    ref = mod.NonCausalViTDecoder(mean, std, keep_layers=list(range(4)), time_norm=5, heads=2, dim_head=64)
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if n.endswith(("scale1", "scale2")):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif "norm" in n and n.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif p.ndim > 1:
                p.copy_(torch.randn(p.shape, generator=g) / max(1, p.shape[-1]) ** 0.5)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    model = SD.SolModel(dims, None, torch.float32)
    model.load_state_dict({(k if k.startswith("post_quant_conv") else "decoder." + k): v.clone()
                           for k, v in ref.state_dict().items()}, strict=True)
    dec = SD.SolDecoder.from_model(model, config=SD.CONFIGS["F32"], mean=mean, std=std, device="cpu")
    out = {"dims": dataclasses.asdict(dims), "shapes": {}}
    pm = torch.tensor(VV.IMAGENET_MEAN).view(1, 3, 1, 1, 1)
    ps = torch.tensor(VV.IMAGENET_STD).view(1, 3, 1, 1, 1)
    worst = 1e9
    for T, h, w in ((14, 23, 37), (12, 20, 32), (7, 17, 33), (5, 16, 16)):
        z = torch.randn(1, 24, T, h, w, generator=g)
        with torch.no_grad():
            rf = (ref(z.transpose(1, 2).contiguous()) * ps + pm).clamp(0, 1)[0].permute(1, 2, 3, 0)
        mf = dec.decode(z, output="float")
        c = SM.psnr_float(mf, rf) if mf.shape == rf.shape else {"error": f"{list(mf.shape)} vs {list(rf.shape)}"}
        out["shapes"][f"T{T}_h{h}_w{w}"] = {**c, "frames": int(mf.shape[0]), "expected_frames": SD.frames_for_latents(T),
                                            "max_sections_alive": dec.last_stats["max_sections_alive"],
                                            "sections": dec.last_stats["sections"]}
        worst = min(worst, c.get("psnr_db", 0.0))
    # the test can fail: the same decoder with the temporal shift set to 1 must differ
    z = torch.randn(1, 24, 14, 23, 37, generator=g)
    with torch.no_grad():
        rf = (ref(z.transpose(1, 2).contiguous()) * ps + pm).clamp(0, 1)[0].permute(1, 2, 3, 0)
    saved = SD.T_SHIFT
    try:
        SD.T_SHIFT = 1
        broken = SM.psnr_float(dec.decode(z, output="float"), rf)
    finally:
        SD.T_SHIFT = saved
    out["sensitivity_t_shift_1"] = broken
    out["worst_psnr_db"] = worst
    # the fp16 code path (config A's steps) on the CPU: the same tiny weights rounded to F16, the torch
    # rope standing in for the CUDA-only kitchen kernel; it must run and stay near the float32 reference
    m16 = SD.SolModel(dims, None, torch.float16)
    m16.load_state_dict({(k if k.startswith("post_quant_conv") else "decoder." + k): v.half()
                         for k, v in ref.state_dict().items()}, strict=True)
    d16 = SD.SolDecoder.from_model(m16, config=SD.CONFIGS["A"], mean=mean, std=std, device="cpu", rope_impl="torch")
    f16 = d16.decode(z, output="float")
    out["fp16_path_cpu_vs_reference_fp32"] = {**SM.psnr_float(f16, rf), "rope_impl": d16.rope_impl,
                                              "uint8_out_dtype": str(d16.decode(z).dtype)}
    out["passed"] = (worst >= 120.0 and broken["psnr_db"] < 100.0
                     and out["fp16_path_cpu_vs_reference_fp32"]["psnr_db"] >= 40.0
                     and all(v["max_sections_alive"] <= 2 and v["frames"] == v["expected_frames"]
                             for v in out["shapes"].values()))
    return out


def dry_run(attention: str) -> int:
    LOG.open(HERE / "dry_run.log", mode="w")
    install_log_bridge()
    rec: dict = {"script": rel(Path(__file__)), "started_at": now(), "versions": versions(),
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "cuda_available": torch.cuda.is_available(), "git": git_state(), "items": {}}
    items = rec["items"]

    def item(name: str, fn):
        LOG(f"dry-run: {name}")
        try:
            items[name] = fn()
        except BaseException as exc:  # noqa: BLE001
            items[name] = {"passed": False}
            failure(items[name], exc)
        LOG(f"dry-run: {name} passed={items[name].get('passed')}")
        write_json(HERE / "dry_run.json", rec)

    def it_paths():
        files = {**{k: v for k, v in PATHS.items() if k != "tokenizer_dir"},
                 **{f"sol_{k}": v for k, v in SOL_FILES.items()}, "turbo_lora": TURBO_LORA_FILE,
                 "illustration": ILLUSTRATION}
        out = {"files": {k: file_info(v) for k, v in files.items()},
               "tokenizer_dir": (ROOT / PATHS["tokenizer_dir"]).is_dir(),
               "ref_nvidia": {"file": rel(REF_FILE), "sha256": sha256(REF_FILE), "expected": REF_SHA256},
               "sampler": SAMPLER.is_file(), "ffmpeg": tool("ffmpeg") is not None}
        from PIL import Image

        with Image.open(ROOT / ILLUSTRATION) as im:
            out["illustration_size"] = list(im.size)
        out["comfy_kitchen"] = metadata.version("comfy-kitchen")
        out["passed"] = (all(v["bytes"] for v in out["files"].values()) and out["tokenizer_dir"]
                         and out["ref_nvidia"]["sha256"] == REF_SHA256 and out["sampler"]
                         and out["illustration_size"] == [896, 1152] and out["comfy_kitchen"] == "0.2.37")
        return out

    def it_conditions():
        rows = conditions_table(attention)
        md = table_markdown(rows)
        (HERE / "conditions_table.md").write_text(md + "\n", encoding="utf-8")
        LOG("生成の条件の表（製品の既定／この実験で使う値／違う場合の理由）:\n" + md)
        checks = {}
        for it in GEN:
            req = gen_request(it, RUNS / f"gen_{it}", attention)
            checks[it] = check_request(it, req)
        return {"table": rows, "table_file": rel(HERE / "conditions_table.md"), "requests": checks,
                "prompts": PROMPTS, "passed": all(v["validate"] == "passed" for v in checks.values())}

    def it_sol_files():
        out = {}
        for kind, path in SOL_FILES.items():
            info = SD.inspect_sol_file(ROOT / path)
            model = SD.build_on_meta(info)
            sd = model.state_dict()
            t = info.header.tensors
            file_keys = set(info.decoder_keys)
            shape_bad = [k for k in file_keys & set(sd) if tuple(t[k].shape) != tuple(sd[k].shape)]
            specs = sorted({f"{q.format}/convrot={q.convrot}/g{q.convrot_groupsize}" for q in info.quant.values()})
            dtypes = sorted({t[k].dtype for k in file_keys})
            r = {"num_layers": info.num_layers, "int8_layers": len(info.quant), "specs": specs,
                 "decoder_keys": len(file_keys), "model_keys": len(sd),
                 "file_keys_not_in_model": sorted(file_keys - set(sd))[:10],
                 "model_keys_not_in_file": sorted(set(sd) - file_keys)[:10], "shape_mismatch": shape_bad[:10],
                 "dtypes": dtypes, "problems": info.problems}
            want_int8 = 96 if kind == "int8" else 0
            r["passed"] = (info.num_layers == 24 and len(info.quant) == want_int8 and not info.problems
                           and set(sd) == file_keys and not shape_bad
                           and (specs == ["int8_tensorwise/convrot=True/g256"] if kind == "int8" else specs == []))
            out[kind] = r
        out["passed"] = all(v["passed"] for v in out.values())
        return out

    def it_ref_keys():
        mod = load_ref_module()
        info = SD.inspect_sol_file(ROOT / SOL_FILES["fp16"])
        with torch.device("meta"):
            ref = mod.NonCausalViTDecoder(info.mean, info.std, mod.KEEP[24], time_norm=5)
        sd = ref.state_dict()
        t = info.header.tensors
        file = {(k[len("decoder."):] if k.startswith("decoder.") else k): k for k in info.decoder_keys}
        bad = [k for k in set(file) & set(sd) if tuple(t[file[k]].shape) != tuple(sd[k].shape)]
        r = {"ref_keys": len(sd), "file_keys": len(file), "only_ref": sorted(set(sd) - set(file))[:10],
             "only_file": sorted(set(file) - set(sd))[:10], "shape_mismatch": bad[:10],
             "keep_layers": mod.KEEP[24], "imports_of_ref": sorted(set(re.findall(r"^(?:from|import) (\S+)",
                                                                                   REF_FILE.read_text(encoding="utf-8"), re.M)))}
        r["passed"] = len(sd) == 297 and set(sd) == set(file) and not bad
        return r

    def it_stats():
        import sft_quant_format

        out = {}
        std_h = sft_quant_format.read_header(str(ROOT / PATHS["video_vae"]))
        sm, ss = VV.latent_stats_from_metadata(std_h.metadata, VV.METADATA_KEY, 24)
        for kind, path in SOL_FILES.items():
            info = SD.inspect_sol_file(ROOT / path)
            out[kind] = {"mean_max_diff_vs_standard_vae": max(abs(a - b) for a, b in zip(info.mean, sm)),
                         "std_max_diff_vs_standard_vae": max(abs(a - b) for a, b in zip(info.std, ss))}
        out["passed"] = True       # information (the experiment reads each file's own values)
        return out

    def it_tiny():
        return _tiny_ref_pair(load_ref_module())

    def it_rope():
        from comfy_kitchen.backends.eager import rope as ER

        g = torch.Generator().manual_seed(3)
        out = {}
        for shifted in (False, True):
            cos, sin = SD.rope_cos_sin(20, 32, shifted, 48)
            n = cos.shape[0]
            q = torch.randn(n, 32, 64, generator=g)
            k = torch.randn(n, 32, 64, generator=g)
            qe, ke = ER.rms_rope_split_half(q[None], k[None], SD.kitchen_freqs(cos, sin), torch.ones(64), None,
                                            SD.EPS, rot_dim=48)
            qt, kt = SD.rms_rope_torch(q, cos, sin), SD.rms_rope_torch(k, cos, sin)
            c48, s48 = torch.cat([cos, cos], -1)[:, None], torch.cat([sin, sin], -1)[:, None]
            qv = VV._apply_rope(torch.nn.functional.rms_norm(q, (64,), eps=SD.EPS), c48, s48)
            out["shifted" if shifted else "plain"] = {
                "torch_vs_kitchen_eager_q": float((qe[0] - qt).abs().max()),
                "torch_vs_kitchen_eager_k": float((ke[0] - kt).abs().max()),
                "torch_vs_video_vae_apply_rope": float((qv - qt).abs().max())}
        out["note"] = ("fp32 on the CPU: the torch formula against comfy-kitchen's eager rms_rope_split_half (the "
                       "table layout) and video_vae._apply_rope. The fp16 CUDA kernel is compared in --selfcheck.")
        out["passed"] = all(v <= 1e-6 for r in out.values() if isinstance(r, dict) for v in r.values())
        return out

    def it_int8_cpu():
        from comfy_kitchen.backends.eager import quantization as EQ
        from engine.sft_quant.sft_reader import read_tensors
        import sft_quant_format

        p8, p16 = ROOT / SOL_FILES["int8"], ROOT / SOL_FILES["fp16"]
        info8 = SD.inspect_sol_file(p8)
        h16 = sft_quant_format.read_header(str(p16))
        L = "decoder.blocks.0."
        names = [f"{L}{leaf}.{s}" for leaf in SD._INT8_LEAVES for s in ("weight", "weight_scale", "bias")]
        names += [f"{L}{n}" for n in ("norm1.weight", "norm2.weight", "scale1", "scale2")]
        t8 = {k: (v.float() if v.dtype == torch.float16 else v) for k, v in read_tensors(str(p8), names, info8.header)}
        t16 = {k: v.float() for k, v in read_tensors(str(p16), [n for n in names if not n.endswith("weight_scale")], h16)}
        spec = info8.quant["blocks.0.attn.to_qkv"]
        g = torch.Generator().manual_seed(5)
        x = torch.randn(64, 2048, generator=g)
        res = torch.randn(64, 2048, generator=g)

        def lin8(leaf, inp, **kw):
            return EQ.int8_linear(inp, t8[L + leaf + ".weight"], t8[L + leaf + ".weight_scale"], t8[L + leaf + ".bias"],
                                  out_dtype=torch.float32, convrot=bool(spec.convrot),
                                  convrot_groupsize=int(spec.convrot_groupsize), **kw)

        def lin16(leaf, inp):
            return torch.nn.functional.linear(inp, t16[L + leaf + ".weight"], t16[L + leaf + ".bias"])

        def rel_err(a, b):
            return {"rel_l2": float((a - b).norm() / b.norm()),
                    "cosine": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))}

        rms1 = torch.nn.functional.rms_norm(x, (2048,), weight=t16[L + "norm1.weight"], eps=SD.EPS)
        rms2 = torch.nn.functional.rms_norm(x, (2048,), weight=t16[L + "norm2.weight"], eps=SD.EPS)
        out = {"to_qkv(rms_norm)": rel_err(lin8("attn.to_qkv", x, input_act="rms_norm",
                                                input_act_weight=t8[L + "norm1.weight"], input_act_eps=SD.EPS),
                                           lin16("attn.to_qkv", rms1)),
               "to_out(+residual)": rel_err(lin8("attn.to_out", x, residual=res, residual_scale=t8[L + "scale1"]),
                                            torch.addcmul(res, lin16("attn.to_out", x), t16[L + "scale1"]))}
        a16 = lin16("ff.w1", rms2)
        out["w1(rms_norm)"] = rel_err(lin8("ff.w1", x, input_act="rms_norm", input_act_weight=t8[L + "norm2.weight"],
                                           input_act_eps=SD.EPS), a16)
        gate_, up = a16.chunk(2, dim=-1)
        out["w2(swiglu,+residual)"] = rel_err(
            lin8("ff.w2", a16, input_act="swiglu", residual=res, residual_scale=t8[L + "scale2"]),
            torch.addcmul(res, lin16("ff.w2", torch.nn.functional.silu(gate_) * up), t16[L + "scale2"]))
        out["note"] = ("block 0 of the INT8 file through comfy-kitchen's eager int8_linear with the fused arguments "
                       "(CPU, float32) against the FP16 file's layer with the same steps written out: a wrong scale, "
                       "rotation or half of the swiglu would give an error near 100 %.")
        out["passed"] = all(v["rel_l2"] < 0.05 and v["cosine"] > 0.999 for k, v in out.items() if isinstance(v, dict))
        return out

    def it_patch():
        with save_latent_patch():
            inside = patch_in()
        return {"in": inside, "restored": patch_restored(), "passed": inside and patch_restored()}

    def it_frames():
        out = {"verify": {c: {"grid": list(v), "frames": SD.frames_for_latents(v[0]),
                              "windows_odd_layer_time": len(SD.D._spans(v[0], SD.T_SHIFT, SD.BLOCK)),
                              "blocks": -(-v[0] // SD.BLOCK)} for c, v in VERIFY.items()},
               "bench": {}}
        for p, (w, h, f, std) in BENCH.items():
            T = latent_frames(f)
            out["bench"][p] = {"latents": T, "frames": SD.frames_for_latents(T), "grid": [T, h // 16, w // 16],
                               "standard_too": std, "tokens": T * (h // 16) * (w // 16),
                               "comfy_estimate_mib_fp16_without_weights": round(SD.comfy_memory_estimate_mib(T, h // 16, w // 16), 1),
                               "latent_a_tiling": [-(-T // 37), -(-(h // 16) // 48), -(-(w // 16) // 80)]}
        out["passed"] = all(v["frames"] == BENCH[p][2] for p, v in out["bench"].items())
        return out

    def it_scrub():
        fake = "first last"
        f = make_scrub([fake, "FIRSTL~1"])
        samples = [r"C:\Users\first last\AppData\x.py", "C:/Users/first last/x", r"C:\\Users\\first last\\x",
                   r"c:\users\FIRST LAST\x", r"C:\Users\FIRSTL~1\AppData\Local\Temp\x", "Users/first last's file"]
        fake_ok = all(fake.lower() not in f(s).lower() and "firstl~1" not in f(s).lower() for s in samples)
        home = os.environ.get("USERPROFILE") or str(Path.home())
        real = [home, _short_path(home), home.replace("\\", "/"), json.dumps(home), json.dumps(_short_path(home)),
                home + r"\AppData\Local\Temp\a b.txt"]
        names = user_names()
        real_ok = all(not any(n.lower() in scrub(s).lower() for n in names) for s in real)
        return {"names_counted": len(names), "names_have_space": any(" " in n for n in names),
                "fake_name_with_space_removed": fake_ok, "real_home_long_short_json_removed": real_ok,
                "example": f(samples[0]), "passed": fake_ok and real_ok and bool(names)}

    def it_metrics():
        img = torch.full((3, 256, 384, 3), 100, dtype=torch.uint8)
        img[:, :, 128] = 160                      # a vertical line at x = 128
        img[:, :, 256] = 160
        ge = SM.grid_edges(img)
        flat = SM.grid_edges(torch.full((2, 256, 384, 3), 100, dtype=torch.uint8))
        fr = torch.zeros((22, 32, 32, 3), dtype=torch.uint8)
        fr[17:] = 50                              # a jump between block 0 (17 frames) and block 1
        fl = SM.flicker(fr, [17, 5])
        same = SM.psnr_u8(img, img)
        ok = (ge["p128"]["x"]["ratio"] > 5 and ge["p128"]["x"]["ratio_excluding_16"] > 5 and flat["p128"]["x"]["off"] == 0
              and fl["boundary_pair_index"] == [16] and fl["boundary_mean"] == 50.0 and fl["inside_mean"] == 0.0
              and same["psnr_db"] == 100.0)
        return {"grid_line_x128": ge["p128"]["x"], "flicker": {k: fl[k] for k in ("boundary_mean", "inside_mean")},
                "passed": bool(ok)}

    item("1_paths_and_versions", it_paths)
    item("2_generation_conditions", it_conditions)
    item("3_sol_files_and_model_keys", it_sol_files)
    item("4_reference_keys", it_ref_keys)
    item("5_latent_statistics", it_stats)
    item("6_tiny_decoder_vs_reference_cpu", it_tiny)
    item("7_rope_formula_cpu", it_rope)
    item("8_int8_fused_arguments_cpu", it_int8_cpu)
    item("9_replacement_in_and_out", it_patch)
    item("10_frame_counts_and_bench_points", it_frames)
    item("11_user_name_scrub", it_scrub)
    item("12_metrics_synthetic", it_metrics)
    rec["all_passed"] = all(v.get("passed") for v in items.values())
    rec["git_after"] = git_state()
    rec["status"] = "done"
    rec["finished_at"] = now()
    write_json(HERE / "dry_run.json", rec)
    LOG(f"dry-run: all_passed={rec['all_passed']} -> {rel(HERE / 'dry_run.json')}")
    LOG.close()
    return 0 if rec["all_passed"] else 1


def run_gate_only() -> int:
    LOG.open(RUNS / "gate" / "run.log", mode="a")
    rec = gate()
    write_json(RUNS / "gate" / "result.json", rec)
    LOG(f"gate: blocking={rec['blocking'] or 'none'}")
    LOG.close()
    return 0 if not rec["blocking"] else 2


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="CPU only: paths, the conditions table, the checks")
    g.add_argument("--gate", action="store_true", help="no GPU work: the checks made before every GPU run")
    g.add_argument("--selfcheck", action="store_true", help="GPU: comfy-kitchen's functions vs the torch formulas")
    g.add_argument("--verify", nargs="*", choices=list(VERIFY), metavar="COND",
                   help=f"GPU: plan 5.2 (default: all of {list(VERIFY)})")
    g.add_argument("--gen", choices=list(GEN), help="GPU: plan 5.3, generate one latent (a..e)")
    g.add_argument("--compare", nargs="+", choices=list(GEN), help="GPU: plan 5.4 / 5.5 on saved latents")
    g.add_argument("--bench", nargs="*", choices=list(BENCH), metavar="POINT",
                   help=f"GPU: plan 5.6 (default: every point of {list(BENCH)})")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="--compare: Sol configurations (default A B C); --bench: standard and / or A B C (default all)")
    ap.add_argument("--latent", default="random", choices=["random", *GEN], help="--verify: random latents or a saved one")
    ap.add_argument("--latent-source", default="tile", choices=["tile", "random"],
                    help="--bench: latent a repeated to each size (tile) or random latents")
    ap.add_argument("--attention", default=DEFAULTS["attention_backend"], choices=["sdpa", "sage"],
                    help="--gen / --dry-run: the DiT's attention (default: the product's sdpa)")
    ap.add_argument("--force", action="store_true", help="--bench: redo the points already recorded")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        return dry_run(a.attention)
    if a.gate:
        return run_gate_only()
    if a.selfcheck:
        return run_selfcheck()
    if a.verify is not None:
        return run_verify(a.verify or list(VERIFY), a.latent)
    if a.gen:
        return run_gen(a.gen, a.attention)
    if a.compare:
        configs = a.configs or ["A", "B", "C"]
        bad = [c for c in configs if c not in ("A", "B", "C")]
        if bad:
            raise SystemExit(f"--configs: {bad} (A B C; the standard decoder is always decoded)")
        rc = 0
        for it in a.compare:
            rc = max(rc, run_compare(it, configs))
        return rc
    configs = a.configs or ["standard", "A", "B", "C"]
    bad = [c for c in configs if c != "standard" and c not in ("A", "B", "C")]
    if bad:
        raise SystemExit(f"--configs: {bad} (standard A B C)")
    return run_bench(a.bench or list(BENCH), configs, "random" if a.latent_source == "random" else "tile", a.force)


if __name__ == "__main__":
    raise SystemExit(main())
