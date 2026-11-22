r"""MiniMax H3 pi-H3 spike (2026-11-22): the pi-H3 adapter, its sigma schedule and its DWD head
run through the product's ``H3Pipeline`` without editing any product file.

Plan (authoritative on scope): ``inherited-beaming-brooks-of-f65207cb-0887-4efd-905c-909964b3a2ea.md``.
Patterns copied from ``outputs/h3-fp8-w4a8-spike-2026-11-12/h3_quant_spike.py`` (header, ``Log``,
``StepClock``, ``ffprobe``, ``failure``, patches in / out) and the GPU gate from
``Docs/Outputs-archive/h3-mixed-int8-2026-11-22/h3_mixed_int8_gate_api.py``.

Five module / class attributes are replaced inside :func:`spike_patches` and restored in ``finally``:

* ``engine_h3.pipeline_h3.sigma_schedule``  -> the pi-Flow schedule (2 or 4 steps only)
* ``engine_h3.denoise_mask.dit_input``      -> first call per denoise and stream: the state x sigma_0 (in place)
* ``engine_h3.dit.H3DiT.final_rows``        -> copies the generated video rows' hidden state + t_emb to the CPU
* ``H3Pipeline._build_dit``                 -> the pi-H3 non-LoRA tensors applied to the CPU DiT
* ``engine_h3.video_vae.H3VideoDecoder``    -> the DWD head decodes the copied hidden states

From the repository root (Git Bash):

    PY=./.venv-engine-h3/Scripts/python.exe; S=outputs/h3-pih3-spike-2026-11-22/h3_pih3_spike.py
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --dry-run              # CPU only
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --single P1            # GPU: P1..P4 (1280x768, 73 frames)
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --chain C3_4step       # GPU: 2 clips of 124 frames (640x384)
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --split C3_4step       # GPU: form 4 on C3's saved hidden states

The edit items (the supervisor's plan file, the follow-up for Retake / Inpainting / Outpainting):

    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --prepare              # CPU: cut the material into _material/
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --edit R1              # GPU: R1 R1c I1 I1c O1 O1c (c = control)
    PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --post R1 I1 O1        # CPU: comparison sheets into runs/_compare/

Exit codes: 0 = passed / done, 1 = failed (recorded in the JSON), 2 = refused by the gate / bad environment.
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
CPU_ONLY = any(a in sys.argv for a in ("--dry-run", "--prepare", "--post"))
if CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import collections  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
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
import sft_quant_format  # noqa: E402
from engine_h3 import denoise_mask as DM  # noqa: E402
from engine_h3 import dit as D  # noqa: E402
from engine_h3 import pipeline_h3 as P  # noqa: E402
from engine_h3 import video_vae as VV  # noqa: E402
from engine_h3.constants import DEFAULT_SHIFT_AUDIO, DEFAULT_SHIFT_VIDEO, audio_latents, latent_frames  # noqa: E402
from engine_h3.scheduler import euler_step  # noqa: E402

import pih3_adapter as PA  # noqa: E402
import pih3_dwd as PD  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixed conditions (h3_quant_spike.py 87-117; plan "試験の項目")
# --------------------------------------------------------------------------- #

OUT_ROOT = HERE
RUNS = OUT_ROOT / "runs"
WEIGHTS = "models/MiniMaxH3/Weights/"
BASES = {
    "fl2va": WEIGHTS + "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "sulphur": WEIGHTS + "Sulphur-3_Ref2VA-hybrid_mid_int8_convrot_keepB0-1-49.safetensors",
}
PATHS = {
    "transformer": BASES["fl2va"],
    "text_encoder": "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "video_vae": "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
    "tokenizer_dir": "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
}
WIDTH, HEIGHT, NUM_FRAMES = 1280, 768, 73
CHAIN_WIDTH, CHAIN_HEIGHT, CHAIN_FRAMES = 640, 384, 124
DEFAULT_SEED = 1234
ATTENTION = "sage"
VRAM16 = True
BLOCKS_ON_GPU = 8
PREFETCH = True
KEEP_RESIDENT = False
DETERMINISTIC = False
SHIFT_VIDEO, SHIFT_AUDIO = float(DEFAULT_SHIFT_VIDEO), float(DEFAULT_SHIFT_AUDIO)   # 12 / 3 (ComfyUI workflow too)
DEFAULT_PROMPT = (
    "A busy city street at dusk seen from the sidewalk. Pedestrians walk past the camera, their footsteps "
    "clicking on the pavement; cars and a bus drive by with engine hum and tire noise, and a car horn honks "
    "in the distance. Steady camera."
)
CHAIN_PROMPT_2 = (
    "The same busy city street at dusk, a moment later, seen from the same spot on the sidewalk. The street "
    "lights flicker on as more pedestrians pass the camera, their footsteps clicking on the pavement; a taxi "
    "and a bus drive by with engine hum and tire noise, and a car horn honks in the distance. Steady camera."
)
SINGLE = {"P1": ("fl2va", 4), "P2": ("fl2va", 2), "P3": ("sulphur", 4), "P4": ("sulphur", 2)}
CHAIN = {"C3_4step": 4, "C3_2step": 2, "C3b_4step": 4, "C3b_2step": 2}
# C3b: same conditions as C3 but both clips carry the same prompt (a product run with only the main prompt filled in)
CHAIN_SAME_PROMPT = (
    "A young girl in a white summer dress and a straw hat walks along a sandy beach at the water's edge on a sunny "
    "day. Gentle waves wash ashore with the sound of surf, seagulls call in the distance, and her footsteps are soft "
    "on the wet sand. Steady tracking camera."
)
CHAIN_PROMPTS = {"C3b_4step": (CHAIN_SAME_PROMPT, CHAIN_SAME_PROMPT), "C3b_2step": (CHAIN_SAME_PROMPT, CHAIN_SAME_PROMPT)}


def chain_prompts(item: str) -> tuple:
    return CHAIN_PROMPTS.get(item, (DEFAULT_PROMPT, CHAIN_PROMPT_2))

# pi-Flow scheduler settings of the pi-H3 workflow (ComfyUI-piFlow docs/piH3.md, LakonLab demo)
MAX_RAW_T = {2: 0.93, 4: 0.96}
MIN_RAW_T = 0.0
FINAL_STEP_SIZE_SCALE = 0.6
EXPECTED_SIGMAS = {
    2: {"video": [0.993767, 0.865340, 0.0], "audio": [0.975524, 0.616348, 0.0]},
    4: {"video": [0.99654, 0.964451, 0.899297, 0.695652, 0.0], "audio": [0.9863, 0.87151, 0.69065, 0.36364, 0.0]},
}
SIGMA_TOL = 1e-6

OWNER_PORT = 18620
GPU_BUSY_MIB = 3072
WORKER_RE = re.compile(r"-m\s+engine\w*\.worker|engine\w*[\\/]worker\.py")
SPLIT_CONTENT = (5, 1)
SPLIT_MARGINS = (4, 2, 1)
HIDDEN_FILE = "hidden_concat.safetensors"


# --------------------------------------------------------------------------- #
# Small helpers (copied from h3_quant_spike.py)
# --------------------------------------------------------------------------- #

class Log:
    def __init__(self) -> None:
        self.file = None

    def __call__(self, msg: str) -> None:
        line = f"[h3_pih3_spike] {msg}"
        print(line, flush=True)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()

    def progress(self, stage: str, index: int, total: int, extra: str = "") -> None:
        line = f"[progress] {stage} {index}/{total}{extra}"
        print(line, flush=True)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()

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
        return str(q.relative_to(ROOT))
    except ValueError:
        return str(q)


def jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jdump(obj) + "\n", encoding="utf-8")


def tool(name: str) -> str:
    local = ROOT / "tools" / "ffmpeg" / "bin" / f"{name}.exe"
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found (<repo>\\tools\\ffmpeg\\bin or PATH)")
    return found


def run_cmd(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def ffprobe(path: Path) -> dict:
    exe = tool("ffprobe")
    v = run_cmd([exe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                 "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration",
                 "-of", "json", str(path)])
    a = run_cmd([exe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                 "stream=codec_name,sample_rate,channels,duration", "-of", "json", str(path)])
    vs = (json.loads(v.stdout or "{}").get("streams") or [{}])[0]
    as_ = (json.loads(a.stdout or "{}").get("streams") or [None])[0]
    return {"path": rel(path), "video": vs, "audio": as_}


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
    out = {"torch": torch.__version__}
    for dist in ("comfy-kitchen", "safetensors"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def file_info(path) -> dict:
    p = ROOT / path
    return {"path": str(path), "name": Path(path).name, "bytes": p.stat().st_size if p.is_file() else None}


def tensor_stats(t: torch.Tensor, chunk: int = 1 << 22) -> dict:
    """mean / std / max |x| / NaN / Inf of ``t`` in float64, chunk by chunk (CPU or GPU)."""
    flat = t.detach().reshape(-1)
    n, s, ss, mx, nan, inf = 0, 0.0, 0.0, 0.0, 0, 0
    for i in range(0, flat.numel(), chunk):
        x = flat[i:i + chunk].double()
        bad_nan, bad_inf = torch.isnan(x), torch.isinf(x)
        nan += int(bad_nan.sum())
        inf += int(bad_inf.sum())
        x = x[~(bad_nan | bad_inf)]
        n += x.numel()
        s += float(x.sum())
        ss += float((x * x).sum())
        if x.numel():
            mx = max(mx, float(x.abs().max()))
    mean = s / n if n else None
    std = math.sqrt(max(ss / n - mean * mean, 0.0)) if n else None
    return {"numel": int(flat.numel()), "mean": mean, "std": std, "max_abs": mx, "nan": nan, "inf": inf}


class StepClock:
    """The progress callback (single job and chain): forwards to LOG and stamps each denoise step."""

    def __init__(self) -> None:
        self.stamps: dict[tuple[int, int], float] = {}

    def __call__(self, stage: str, index: int, total: int, *, outer_index=None, outer_total=None) -> None:
        LOG.progress(stage, index, total, "" if outer_index is None else f" (clip {outer_index + 1}/{outer_total})")
        if stage == "stage1_denoise":
            self.stamps[(int(outer_index or 0), int(index))] = time.perf_counter()

    def per_step(self) -> dict:
        out = {}
        for clip in sorted({c for c, _ in self.stamps}):
            idx = sorted(i for c, i in self.stamps if c == clip)
            diffs = [round(self.stamps[(clip, i + 1)] - self.stamps[(clip, i)], 3) for i in idx if (clip, i + 1) in self.stamps]
            out[f"clip{clip}"] = {"seconds_per_step": diffs,
                                  "median_after_first": statistics.median(diffs[1:]) if len(diffs) > 1 else None}
        return out


# --------------------------------------------------------------------------- #
# GPU gate (h3_mixed_int8_gate_api.py 576-661, reduced to the three checks)
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
    workers = [{"pid": p["pid"], "ppid": p["ppid"], "cmdline": p["cmdline"][:300]}
               for p in all_processes() if WORKER_RE.search(p["cmdline"] or "")]
    rec["engine_workers"] = workers
    if workers:
        rec["blocking"].append(f"ワーカーのプロセスが残っている（{[w['pid'] for w in workers]}）")
    LOG(f"gate: port {OWNER_PORT} {'LISTENING' if listening else 'free'}; GPU used {gpu.get('memory_used_mib')} MiB "
        f"(bound {GPU_BUSY_MIB}); python/uvicorn on GPU: {gpu.get('python_like_others') or 'none'}; "
        f"workers: {workers or 'none'}")
    return rec


# --------------------------------------------------------------------------- #
# Spike state and the five replacements
# --------------------------------------------------------------------------- #

class _State:
    def __init__(self) -> None:
        self.reset()

    def reset(self, *, steps: int = 0, pi_path: Path | None = None, out_dir: Path | None = None,
              fixed_head_latents: int = 0, save_hidden: bool = False) -> None:
        self.steps = steps
        self.pi_path = pi_path
        self.out_dir = out_dir
        self.fixed_head_latents = fixed_head_latents
        self.save_hidden = save_hidden
        self.sigma_calls: list[dict] = []
        self.sessions: list[dict] = []       # one per _denoise (two sigma_schedule calls each)
        self.adapter_reports: list[dict] = []
        self.decoder: dict = {}
        self.hits: collections.Counter = collections.Counter()


STATE = _State()


def _time_shift(shift: float, t: float) -> float:
    """ComfyUI ``time_snr_shift`` (and the engine's ``sigma_schedule`` formula)."""
    if shift == 1.0:
        return t
    return shift * t / (1 + (shift - 1) * t)


def pih3_sigmas(n_steps: int, shift: float) -> torch.Tensor:
    """float32 ``[n + 1]``: ComfyUI-piFlow ``get_piflow_schedule`` with ``percent_to_sigma`` of a flow
    model of ``shift``, then a terminal 0 (the engine's ``sigma_schedule`` contract)."""
    n = int(n_steps)
    if n not in MAX_RAW_T:
        raise ValueError(f"pi-H3 のシグマの列は 2 ステップと 4 ステップだけです（{n_steps}）")
    s = float(shift)
    if not s > 0.0:
        raise ValueError(f"shift は正の値にしてください（{shift}）")
    max_t, min_t, f = MAX_RAW_T[n], MIN_RAW_T, FINAL_STEP_SIZE_SCALE
    raw = torch.linspace(max_t, (max_t - min_t) * f / (n - 1 + f) + min_t, n)
    sig = []
    for t in raw.tolist():
        percent = 1.0 - t
        sig.append(1.0 if percent <= 0.0 else 0.0 if percent >= 1.0 else _time_shift(s, 1.0 - percent))
    return torch.cat([torch.tensor(sig, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)])


_ORIG_SIGMA = P.sigma_schedule
_ORIG_DIT_INPUT = DM.dit_input
_ORIG_FINAL_ROWS = D.H3DiT.__dict__["final_rows"]
_ORIG_BUILD_DIT = P.H3Pipeline.__dict__["_build_dit"]
_ORIG_DECODER = VV.H3VideoDecoder


def _sigma_patch(n_steps: int, shift: float) -> torch.Tensor:
    STATE.hits["sigma_schedule"] += 1
    out = pih3_sigmas(n_steps, shift)
    k = len(STATE.sigma_calls)
    entry = {"steps": int(n_steps), "shift": float(shift), "sigmas": [float(v) for v in out]}
    STATE.sigma_calls.append(entry)
    if k % 2 == 0:       # _denoise: video first (pipeline_h3.py 1928), then audio (1929)
        STATE.sessions.append({"index": len(STATE.sessions), "video": entry, "audio": None,
                               "noise_scale": {}, "dit_input_calls": collections.Counter(),
                               "final_rows_calls": 0, "capture": None})
    else:
        STATE.sessions[-1]["audio"] = entry
    return out


def _dit_input_patch(x: torch.Tensor, mask):
    STATE.hits["dit_input"] += 1
    sess = STATE.sessions[-1] if STATE.sessions else None
    kind = "video" if x.dim() == 5 else "audio" if x.dim() == 3 else None
    if sess is None or sess["audio"] is None or kind is None:
        STATE.hits["dit_input_outside_session"] += 1
        return _ORIG_DIT_INPUT(x, mask)
    sess["dit_input_calls"][kind] += 1
    if kind not in sess["noise_scale"]:
        mask_values = mask_rows = None
        if mask is not None:
            rows = mask.rows.float().cpu()
            mask_values = sorted({float(v) for v in torch.unique(rows)})
            if not set(mask_values) <= {0.0, 1.0}:
                raise RuntimeError(f"pi-H3 noise scale: {kind} mask has values {mask_values[:8]} other than 0 / 1; "
                                   "scaling the whole state would change the partly kept rows")
            mask_rows = {"zero": int((rows == 0).sum()), "one": int((rows == 1).sum())}
        sigma0 = float(sess[kind]["sigmas"][0])
        with torch.no_grad():
            x.mul_(sigma0)           # the denoise's own state tensor: ComfyUI's noise * sigma_0
        sess["noise_scale"][kind] = {"value": sigma0, "times": 1, "shape": list(x.shape),
                                     "dtype": str(x.dtype).replace("torch.", ""),
                                     "masked": mask is not None, "mask_values": mask_values, "mask_rows": mask_rows}
    return _ORIG_DIT_INPUT(x, mask)


def _final_rows_patch(self, x, t_emb, time_index, layout, vram16=None):
    STATE.hits["final_rows"] += 1
    sess = STATE.sessions[-1] if STATE.sessions else None
    if sess is not None:
        sess["final_rows_calls"] += 1
        v = layout.video
        sess["capture"] = {"hidden": x[v].detach().to("cpu"), "t_emb": t_emb.detach().float().cpu(),
                           "time_index": time_index[v].detach().to("cpu"), "call": sess["final_rows_calls"]}
    return _ORIG_FINAL_ROWS(self, x, t_emb, time_index, layout, vram16=vram16)


def _build_dit_patch(self, entry: dict):
    STATE.hits["build_dit"] += 1
    dit, cfg = _ORIG_BUILD_DIT(self, entry)
    t0 = time.perf_counter()
    rep = PA.apply_adapter(dit, ROOT / STATE.pi_path, log=LOG)
    rep["seconds"] = round(time.perf_counter() - t0, 2)
    STATE.adapter_reports.append(rep)
    return dit, cfg


class DWDDecoderStandIn:
    """Stands in for ``H3VideoDecoder`` in ``H3Pipeline._decode``: ``from_file`` builds the DWD head,
    ``decode`` ignores the latents' values (it saves them) and returns the head's CPU uint8 frames."""

    @classmethod
    def from_file(cls, path, *, device, dtype=torch.bfloat16):
        STATE.hits["decoder_from_file"] += 1
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        head, info = PD.load_head(ROOT / STATE.pi_path, device)
        STATE.decoder["head_load"] = {**info, "ignored_vae_path": rel(path), "dtype_arg": str(dtype)}
        return cls(head, torch.device(device))

    def __init__(self, head, device) -> None:
        self.head, self.device = head, device

    def to(self, *args, **kwargs):
        return self

    def decode(self, latents: torch.Tensor, *, tiled: bool = True) -> torch.Tensor:
        STATE.hits["decoder_decode"] += 1
        rec = STATE.decoder
        rec["tiled_arg"] = tiled
        from safetensors.torch import save_file  # noqa: PLC0415

        lat = latents.detach().float().cpu().contiguous()
        save_file({"video_latent": lat}, str(STATE.out_dir / "last_latent.safetensors"))
        T, lh, lw = (int(v) for v in lat.shape[2:])
        gh, gw = lh // 2, lw // 2
        hw = gh * gw
        rec["latent"] = {"shape": list(lat.shape), "file": rel(STATE.out_dir / "last_latent.safetensors"),
                         "channel_std": [round(float(v), 6) for v in lat[0].flatten(1).std(dim=1)]}
        k = int(STATE.fixed_head_latents)
        hs, idxs, embs, clips, off = [], [], [], [], 0
        for i, sess in enumerate(STATE.sessions):
            cap = sess["capture"]
            if cap is None:
                raise RuntimeError(f"DWD: denoise {i} left no hidden state")
            h, ti, te = cap["hidden"], cap["time_index"], cap["t_emb"]
            if h.shape[0] % hw:
                raise RuntimeError(f"DWD: denoise {i} hidden rows {h.shape[0]} are not whole latents of {hw}")
            drop = k * hw if i > 0 else 0
            hs.append(h[drop:])
            idxs.append(ti[drop:].long() + off)
            embs.append(te)
            nums, cnts = torch.unique(ti, return_counts=True)
            clips.append({"denoise": i, "latents": h.shape[0] // hw, "dropped_head_latents": drop // hw,
                          "final_rows_call": cap["call"], "t_emb_rows": int(te.shape[0]),
                          "video_time_numbers": sorted({int(v) for v in torch.unique(ti)}),
                          "video_time_numbers_count": int(torch.unique(ti).numel()),
                          "video_time_number_rows": {int(n): int(c) for n, c in zip(nums, cnts)}})
            off += int(te.shape[0])
        hidden, tidx, temb = torch.cat(hs), torch.cat(idxs), torch.cat(embs)
        del hs, idxs
        rec["clips"] = clips
        if hidden.shape[0] != T * hw:
            raise RuntimeError(f"DWD: {hidden.shape[0] // hw} latents of hidden state, the latents have {T}")
        rec["hidden"] = {"shape": list(hidden.shape), "dtype": str(hidden.dtype).replace("torch.", ""),
                         "mib": round(hidden.numel() * hidden.element_size() / 2**20, 1), "grid": [T, gh, gw],
                         "stats": tensor_stats(hidden)}
        if STATE.save_hidden:
            path = STATE.out_dir / HIDDEN_FILE
            save_file({"hidden": hidden.contiguous(), "t_emb": temb.contiguous(), "time_index": tidx.contiguous()},
                      str(path), metadata={"grid": json.dumps([T, gh, gw]), "pi_file": str(STATE.pi_path),
                                           "steps": str(STATE.steps), "fixed_head_latents": str(k)})
            rec["hidden"]["saved"] = rel(path)
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        feats = PD.normalize_features(self.head, hidden.to(self.device), temb, tidx)
        if cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        rec["features_stats"] = tensor_stats(feats)
        pixels, chunks = PD.decode(self.head, feats, (T, gh, gw))
        del feats
        if cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        frames, pstats = PD.to_uint8(pixels)
        t3 = time.perf_counter()
        rec["pixels"] = {"frames": int(pixels.shape[0]), "expected_frames": PD.frames_for_latents(T), **pstats}
        rec["seconds"] = {"normalize": round(t1 - t0, 3), "refiner": round(t2 - t1, 3), "to_uint8": round(t3 - t2, 3)}
        rec["chunks"] = chunks
        if cuda:
            # since the decode phase began (pipeline_h3 _Phases resets the peak there): head load + head run
            rec["peak_vram_since_decode_start_mib"] = {
                "alloc": int(round(torch.cuda.max_memory_allocated() / 2**20)),
                "reserved": int(round(torch.cuda.max_memory_reserved() / 2**20))}
        LOG(f"DWD: {T} latents -> {frames.shape[0]} frames in {rec['seconds']} "
            f"(peak {rec.get('peak_vram_since_decode_start_mib')})")
        return frames


_ACTIVE = False
PATCH_TABLE = (
    (P, "sigma_schedule", _ORIG_SIGMA, _sigma_patch),
    (DM, "dit_input", _ORIG_DIT_INPUT, _dit_input_patch),
    (D.H3DiT, "final_rows", _ORIG_FINAL_ROWS, _final_rows_patch),
    (P.H3Pipeline, "_build_dit", _ORIG_BUILD_DIT, _build_dit_patch),
    (VV, "H3VideoDecoder", _ORIG_DECODER, DWDDecoderStandIn),
)


def _current(owner, attr):
    return owner.__dict__[attr] if isinstance(owner, type) else getattr(owner, attr)


@contextlib.contextmanager
def spike_patches():
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("spike_patches is already active")
    applied = []
    try:
        for owner, attr, orig, new in PATCH_TABLE:
            if _current(owner, attr) is not orig:
                raise RuntimeError(f"{attr} is not the product's own before patching")
            setattr(owner, attr, new)
            applied.append((owner, attr, orig))
        _ACTIVE = True
        yield
    finally:
        for owner, attr, orig in reversed(applied):
            setattr(owner, attr, orig)
        _ACTIVE = False


def patches_in() -> dict:
    from engine_h3.denoise_mask import dit_input as late_dit_input  # noqa: PLC0415 -- as _denoise
    from engine_h3.video_vae import H3VideoDecoder as late_decoder  # noqa: PLC0415 -- as _decode

    return {"pipeline_h3.sigma_schedule": P.sigma_schedule is _sigma_patch,
            "denoise_mask.dit_input (late import)": late_dit_input is _dit_input_patch,
            "denoise_mask.state_velocity sees dit_input": DM.state_velocity.__globals__["dit_input"] is _dit_input_patch,
            "H3DiT.final_rows": D.H3DiT.__dict__["final_rows"] is _final_rows_patch,
            "H3Pipeline._build_dit": P.H3Pipeline.__dict__["_build_dit"] is _build_dit_patch,
            "video_vae.H3VideoDecoder (late import)": late_decoder is DWDDecoderStandIn}


def patches_restored() -> bool:
    return all(_current(owner, attr) is orig for owner, attr, orig, _new in PATCH_TABLE)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #

def lora_for(steps: int):
    return (P.H3LoraInput(path=str(ROOT / PA.PI_FILES[steps]), strength=1.0),)


def single_request(item: str, out_dir: Path) -> P.H3GenerateRequest:
    _base, steps = SINGLE[item]
    return P.H3GenerateRequest(
        prompt=DEFAULT_PROMPT, seed=DEFAULT_SEED, width=WIDTH, height=HEIGHT, num_frames=NUM_FRAMES,
        output_path=str(out_dir / "output.mp4"), num_inference_steps=steps,
        shift_video=SHIFT_VIDEO, shift_audio=SHIFT_AUDIO, keep_resident=KEEP_RESIDENT,
        attention_backend=ATTENTION, block_swap_prefetch=PREFETCH, vram16_mode=VRAM16, loras=lora_for(steps),
    )


def chain_spec(item: str, out_dir: Path):
    """As ``worker._do_generate_chain``: ``base`` carries clip 0's prompt / frames, the LoRA and the seed."""
    from engine_h3.chain_h3 import H3ChainClipSpec, H3ChainSpec  # noqa: PLC0415

    steps = CHAIN[item]
    base = P.H3GenerateRequest(
        prompt=chain_prompts(item)[0], seed=DEFAULT_SEED, width=CHAIN_WIDTH, height=CHAIN_HEIGHT, num_frames=CHAIN_FRAMES,
        output_path=str(out_dir / "output.mp4"), num_inference_steps=steps,
        shift_video=SHIFT_VIDEO, shift_audio=SHIFT_AUDIO, keep_resident=KEEP_RESIDENT,
        attention_backend=ATTENTION, block_swap_prefetch=PREFETCH, vram16_mode=VRAM16, loras=lora_for(steps),
    )
    p1, p2 = chain_prompts(item)
    clips = (H3ChainClipSpec(prompt=p1, num_frames=CHAIN_FRAMES),
             H3ChainClipSpec(prompt=p2, num_frames=CHAIN_FRAMES))
    return H3ChainSpec(base=base, clips=clips, seed=DEFAULT_SEED)


def check_chain_spec(spec) -> dict:
    """The chain's own refusals and every clip request's validate_request, CPU only (run_chain_h3 435-441)."""
    from engine_h3.chain_h3 import _check_spec, clip_requests  # noqa: PLC0415
    from h3_chain_math import compute_h3_chain_layout  # noqa: PLC0415

    _check_spec(spec)
    # as run_chain_h3 436-438 (retake=False for every chain item: unchanged)
    layout = compute_h3_chain_layout([c.num_frames for c in spec.clips], retake=spec.retake is not None)
    reqs = clip_requests(spec)
    for r in reqs:
        P.validate_request(r)
    return {"layout": layout, "clip_requests": len(reqs), "seeds": [r.seed for r in reqs]}


def conditions_record(steps: int, base: str, width: int, height: int, frames, prompts) -> dict:
    return {"prompts": prompts, "negative_prompt": "", "seed": DEFAULT_SEED, "steps": steps,
            "shift_video": SHIFT_VIDEO, "shift_audio": SHIFT_AUDIO, "attention": ATTENTION, "vram16_mode": VRAM16,
            "blocks_on_gpu": BLOCKS_ON_GPU, "block_swap_prefetch": PREFETCH, "keep_resident": KEEP_RESIDENT,
            "deterministic": DETERMINISTIC, "width": width, "height": height, "num_frames": frames,
            "frame_rate": 24, "base_weights": file_info(BASES[base]), "pi_h3": file_info(PA.PI_FILES[steps]),
            "lora": {"file": Path(PA.PI_FILES[steps]).name, "strength": 1.0},
            "pi_flow": {"max_raw_t": MAX_RAW_T[steps], "min_raw_t": MIN_RAW_T,
                        "final_step_size_scale": FINAL_STEP_SIZE_SCALE}}


def sigma_check(steps: int, sessions: list[dict]) -> dict:
    """The sigmas each denoise used vs this script's own schedule (checked in the dry run against an
    independent float64 closed form and the plan's values)."""
    exp = {"video": [float(v) for v in pih3_sigmas(steps, SHIFT_VIDEO)],
           "audio": [float(v) for v in pih3_sigmas(steps, SHIFT_AUDIO)]}
    out, ok = [], bool(sessions)
    for s in sessions:
        d = {kind: max(abs(a - b) for a, b in zip(s[kind]["sigmas"], exp[kind])) if s[kind] else None
             for kind in ("video", "audio")}
        good = all(v is not None and v <= SIGMA_TOL for v in d.values()) and all(
            s[kind] is not None and len(s[kind]["sigmas"]) == len(exp[kind]) for kind in ("video", "audio"))
        ok = ok and good
        out.append({"max_abs_diff": d, "passed": good})
    return {"sessions": out, "passed": ok}


def sessions_record(sessions: list[dict]) -> list[dict]:
    return [{"index": s["index"], "video_sigmas": s["video"]["sigmas"], "audio_sigmas": (s["audio"] or {}).get("sigmas"),
             "shifts": [s["video"]["shift"], (s["audio"] or {}).get("shift")], "noise_scale": s["noise_scale"],
             "dit_input_calls": dict(s["dit_input_calls"]), "final_rows_calls": s["final_rows_calls"],
             "captured": s["capture"] is not None} for s in sessions]


def time_rows_check(expected: dict) -> dict:
    """The one denoise's last captured video rows, counted per time number, against the expected fixed /
    generated rows. The numbers are torch.unique's ascending order of the row times, so with two of them
    the larger is the fixed rows' 0.999 (denoise_mask.masked_row_times, cap) and the smaller the
    generated rows' 1 - sigma of the last step (0.304 at 4 steps). Cross-checked with the product's own
    video mask rows (0 / 1) recorded by the noise-scale replacement."""
    clips = STATE.decoder.get("clips") or []
    rows = (clips[0].get("video_time_number_rows") if clips else None) or {}
    nums = sorted(rows)
    want = {"fixed": int(expected["fixed"]), "generated": int(expected["generated"])}
    got = {"fixed": rows[nums[-1]], "generated": rows[nums[0]]} if len(nums) == 2 else None
    sess = STATE.sessions[0] if STATE.sessions else {}
    mask = ((sess.get("noise_scale") or {}).get("video") or {}).get("mask_rows")
    mask_ok = mask == {"zero": want["fixed"], "one": want["generated"]}
    return {"expected": want, "time_number_rows": rows, "got": got, "product_video_mask_rows": mask,
            "dropped_head_latents": [c.get("dropped_head_latents") for c in clips],
            "passed": got == want and mask_ok and [c.get("dropped_head_latents") for c in clips] == [0]}


def self_check(rec: dict, *, steps: int, n_sessions: int, frames: int, pi: bool = True,
               lora_pairs: int | None = None, rows: dict | None = None, out_frames: int | None = None) -> dict:
    """``pi`` False (a control): the pi-H3 checks (adapter, sigmas, sessions, DWD) are left out and the
    LoRA is the Turbo one (``lora_pairs``). ``rows``: the expected fixed / generated video rows
    (:func:`time_rows_check`). ``out_frames``: the output mp4's frame count (ffprobe)."""
    bad = []
    loras = (((rec.get("engine") or {}).get("full") or {}).get("h3") or {}).get("loras") or []
    want_pairs = PA.EXPECTED["lora_pairs"] if lora_pairs is None else int(lora_pairs)
    if not loras or loras[0].get("pairs_applied") != want_pairs:
        bad.append(f"LoRA pairs applied {[r.get('pairs_applied') for r in loras]} != {want_pairs}")
    if pi:
        reps = STATE.adapter_reports
        if len(reps) != 1:
            bad.append(f"adapter applied {len(reps)} times (1 expected)")
        for rep in reps:
            for key, want in (("quant_layers", 12), ("adaln_weights", 50), ("adaln_biases", 4), ("final_tensors", 7)):
                if rep["counts"][key] != want:
                    bad.append(f"adapter {key} {rep['counts'][key]} != {want}")
        if len(STATE.sessions) != n_sessions:
            bad.append(f"denoise sessions {len(STATE.sessions)} != {n_sessions}")
        if not rec["sigmas"]["passed"]:
            bad.append("sigmas differ from the expected values")
        for s in STATE.sessions:
            if set(s["noise_scale"]) != {"video", "audio"}:
                bad.append(f"session {s['index']}: noise scaled for {sorted(s['noise_scale'])}")
            if s["final_rows_calls"] != steps or s["capture"] is None:
                bad.append(f"session {s['index']}: final_rows {s['final_rows_calls']} calls, {steps} expected")
        if STATE.hits["decoder_decode"] != 1:
            bad.append(f"DWD decode called {STATE.hits['decoder_decode']} times")
        px = STATE.decoder.get("pixels") or {}
        if px.get("frames") != frames:
            bad.append(f"DWD frames {px.get('frames')} != {frames}")
        if rows is not None:
            rec["time_rows"] = time_rows_check(rows)
            if not rec["time_rows"]["passed"]:
                bad.append(f"time-number rows {rec['time_rows']['got']} / mask {rec['time_rows']['product_video_mask_rows']}"
                           f" != expected {rec['time_rows']['expected']}")
    if out_frames is not None:
        got = ((((rec.get("outputs") or {}).get("probe") or {}).get("video")) or {}).get("nb_read_frames")
        if str(got) != str(out_frames):          # ffprobe's JSON gives the count as a string
            bad.append(f"output mp4 frames {got} != {out_frames}")
    if not rec.get("restored"):
        bad.append("patches not restored")
    return {"passed": not bad, "failures": bad}


# --------------------------------------------------------------------------- #
# GPU runs
# --------------------------------------------------------------------------- #

def _begin(name: str) -> tuple[Path, dict]:
    out_dir = RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.open(out_dir / "run.log", mode="w")
    install_log_bridge()
    rec = {"item": name, "script": rel(Path(__file__)), "started_at": now(), "status": "running",
           "git": git_state(), "versions": versions()}
    return out_dir, rec


def _gate_or_refuse(rec: dict, result_path: Path) -> bool:
    rec["gate"] = gate()
    if rec["gate"]["blocking"]:
        rec["status"] = "refused"
        rec["finished_at"] = now()
        write_json(result_path, rec)
        LOG(f"refused by the gate: {rec['gate']['blocking']}")
        LOG.close()
        return False
    if not torch.cuda.is_available():
        rec["status"] = "refused"
        rec["gate"]["blocking"].append("CUDA が使えない")
        write_json(result_path, rec)
        LOG.close()
        return False
    return True


def _pipeline(base: str, clock: StepClock):
    paths = P.H3Paths(**{k: str(ROOT / v) for k, v in {**PATHS, "transformer": BASES[base]}.items()})
    options = P.H3LoadOptions(blocks_on_gpu=BLOCKS_ON_GPU, attention_backend=ATTENTION,
                              block_swap_prefetch=PREFETCH, deterministic=DETERMINISTIC)
    return P.H3Pipeline(paths, options, device=torch.device("cuda"), progress=clock, log=LOG)


def _finish_run(rec: dict, out_dir: Path, clock: StepClock, *, steps: int, n_sessions: int, frames: int,
                **check_kw) -> bool:
    if check_kw.get("pi", True):
        rec["sigmas"] = sigma_check(steps, STATE.sessions)
        rec["sessions"] = sessions_record(STATE.sessions)
        rec["adapter"] = STATE.adapter_reports
        rec["dwd"] = STATE.decoder
    rec["hits"] = dict(STATE.hits)
    rec["per_step"] = clock.per_step()
    out_mp4 = out_dir / "output.mp4"
    rec["outputs"] = {"mp4": rel(out_mp4), "exists": out_mp4.is_file(), "probe": ffprobe(out_mp4) if out_mp4.is_file() else None}
    rec["self_check"] = self_check(rec, steps=steps, n_sessions=n_sessions, frames=frames, **check_kw)
    rec["status"] = "done"
    return bool(rec["self_check"]["passed"] and out_mp4.is_file())


def run_single(item: str) -> int:
    base, steps = SINGLE[item]
    out_dir, rec = _begin(item)
    result_path = out_dir / "result.json"
    rec["conditions"] = conditions_record(steps, base, WIDTH, HEIGHT, NUM_FRAMES, [DEFAULT_PROMPT])
    if not _gate_or_refuse(rec, result_path):
        return 2
    ok = False
    try:
        req = single_request(item, out_dir)
        P.validate_request(req)
        rec["request"] = dataclasses.asdict(req)
        STATE.reset(steps=steps, pi_path=PA.PI_FILES[steps], out_dir=out_dir)
        write_json(result_path, rec)
        clock = StepClock()
        with spike_patches():
            rec["patches_in"] = patches_in()
            pipe = _pipeline(base, clock)
            try:
                t0 = time.perf_counter()
                rec["build_report"] = pipe.load()
                rec["load_seconds"] = round(time.perf_counter() - t0, 2)
                write_json(result_path, rec)
                result = pipe.generate(req)
                rec["engine"] = {"seconds": result.get("seconds"), "peak_vram_mb": result.get("peak_vram_mb"),
                                 "peak_vram_reserved_mb": result.get("peak_vram_reserved_mb"), "full": result}
            finally:
                try:
                    pipe.close()
                except Exception as exc:  # noqa: BLE001
                    LOG(f"pipe.close failed (ignored): {exc!r}")
        rec["restored"] = patches_restored()
        ok = _finish_run(rec, out_dir, clock, steps=steps, n_sessions=1, frames=NUM_FRAMES)
        LOG(f"{item}: done; self_check={rec['self_check']}")
    except BaseException as exc:  # noqa: BLE001 -- interruption is recorded too
        failure(rec, exc)
        rec["restored"] = patches_restored()
        rec["sessions"] = sessions_record(STATE.sessions)
        rec["adapter"] = STATE.adapter_reports
        rec["dwd"] = STATE.decoder
        rec["hits"] = dict(STATE.hits)
    finally:
        rec["finished_at"] = now()
        write_json(result_path, rec)
        LOG.close()
    return 0 if ok else 1


def run_chain(item: str) -> int:
    from engine_h3.chain_h3 import run_chain_h3  # noqa: PLC0415

    steps = CHAIN[item]
    out_dir, rec = _begin(item)
    result_path = out_dir / "result.json"
    rec["conditions"] = conditions_record(steps, "fl2va", CHAIN_WIDTH, CHAIN_HEIGHT, [CHAIN_FRAMES, CHAIN_FRAMES],
                                          list(chain_prompts(item)))
    if not _gate_or_refuse(rec, result_path):
        return 2
    ok = False
    try:
        spec = chain_spec(item, out_dir)
        chk = check_chain_spec(spec)
        layout = chk["layout"]
        rec["chain_layout"] = layout.to_dict()
        rec["request_base"] = dataclasses.asdict(spec.base)
        STATE.reset(steps=steps, pi_path=PA.PI_FILES[steps], out_dir=out_dir,
                    fixed_head_latents=int(layout.overlap_latents), save_hidden=True)
        write_json(result_path, rec)
        clock = StepClock()
        with spike_patches():
            rec["patches_in"] = patches_in()
            pipe = _pipeline("fl2va", clock)
            try:
                t0 = time.perf_counter()
                rec["build_report"] = pipe.load()
                rec["load_seconds"] = round(time.perf_counter() - t0, 2)
                write_json(result_path, rec)
                result = run_chain_h3(pipe, spec, clock, LOG)
                rec["engine"] = {"seconds": result.get("seconds"), "peak_vram_mb": result.get("peak_vram_mb"),
                                 "peak_vram_reserved_mb": result.get("peak_vram_reserved_mb"), "full": result}
            finally:
                try:
                    pipe.close()
                except Exception as exc:  # noqa: BLE001
                    LOG(f"pipe.close failed (ignored): {exc!r}")
        rec["restored"] = patches_restored()
        ok = _finish_run(rec, out_dir, clock, steps=steps, n_sessions=len(spec.clips), frames=int(layout.total_frames))
        LOG(f"{item}: done; self_check={rec['self_check']}")
    except BaseException as exc:  # noqa: BLE001
        failure(rec, exc)
        rec["restored"] = patches_restored()
        rec["sessions"] = sessions_record(STATE.sessions)
        rec["adapter"] = STATE.adapter_reports
        rec["dwd"] = STATE.decoder
        rec["hits"] = dict(STATE.hits)
    finally:
        rec["finished_at"] = now()
        write_json(result_path, rec)
        LOG.close()
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Edit items: Retake / Inpainting / Outpainting (the supervisor's plan file)
# --------------------------------------------------------------------------- #

MATERIAL_DIR = OUT_ROOT / "_material"
COMPARE = RUNS / "_compare"
FPS = 24
EDIT_STEPS = 4
EDIT_AUDIO_STRENGTH = 1.0        # plan: the source audio's Strength 1.0 (masks of 0 / 1 only)
# The control: the existing Turbo LoRA at 0.8, 4 steps, the product's sigmas and video VAE. Its shift is
# the 2026-11-10 Outpainting gate's with Turbo (VERIFICATION_LOG §205.7; h3_outpaint_gate.py 96-99): 12 / 3.
TURBO_LORA_FILE = "models/MiniMaxH3/LoRA/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
TURBO_LORA_STRENGTH = 0.8
TURBO_LORA_PAIRS = 208           # §205.7: pairs_applied 208 on all 6 runs
CONTROL_SHIFT_VIDEO, CONTROL_SHIFT_AUDIO = 12.0, 3.0


def feather_px_of(r: int, width: int, height: int) -> int:
    """services/engines/h3/adapter.py ``_feather_px`` (758-765): floor(r * max(W, H) / 64 + 0.5)."""
    return int(math.floor(int(r) * max(int(width), int(height)) / 64 + 0.5))


FEATHER_STEPS = 1                # the product's default blend_dilation (adapter.py 217 / 223)
FEATHER_PX = feather_px_of(FEATHER_STEPS, WIDTH, HEIGHT)   # 20 at the 1280 canvas
OUTPAINT_SOURCE = (1024, 576)
OUTPAINT_PADS = {"pad_left": 128, "pad_right": 128, "pad_top": 96, "pad_bottom": 96}
REF_VIDEO = "uploads/h3-ref-calibration/H3-Ref-1080p-15s.mp4"
INPAINT_DIR = "uploads/h3-inpaint-development/"
#: Cut as services/video_io.py cut_window_mp4 (select window, atrim, crf 12, yuv420p, CFR 24, PCM audio
#: else AAC 192k); the Outpainting source also scaled (h3_outpaint_gate.py _prepare_main, 401-455).
MATERIALS = {
    "R1_window": {"src": REF_VIDEO, "start_sec": 4.0, "frames": 124, "scale": None,
                  "file": "R1_window_1920x1080_124f.mp4"},
    "O1_source": {"src": REF_VIDEO, "start_sec": 2.0, "frames": 124, "scale": OUTPAINT_SOURCE,
                  "file": "O1_source_1024x576_124f.mp4"},
    "I1_source": {"src": INPAINT_DIR + "inpaint-base.mp4", "start_sec": 0.0, "frames": 73, "scale": None,
                  "file": "I1_source_1280x768_73f.mp4"},
    "I1_mask": {"src": INPAINT_DIR + "mask-768p.mp4", "start_sec": 0.0, "frames": 73, "scale": None,
                "file": "I1_mask_1280x768_73f.mp4"},
}
KIND_MATERIALS = {"retake": ("R1_window",), "inpaint": ("I1_source", "I1_mask"), "outpaint": ("O1_source",)}
EDIT_FRAMES = {"retake": 124, "inpaint": 73, "outpaint": 124}
EDIT_PROMPTS = {
    # VERIFICATION_LOG §200 main condition: Docs/Outputs-archive/h3-retake-2026-11-08/s1.0/request.json
    "retake": ("A young woman in a black-and-white maid costume talks cheerfully to the camera in a pastel room "
               "decorated with sweets and flowers; natural head and hand movement, steady camera."),
    # §208 I1 (= §207.5): Docs/Outputs-archive/h3-inpaint-2026-11-11/h3_inpaint_gate_api.py DEFAULT_PROMPT
    "inpaint": ("Four young girls walking toward the camera on a sunny beach. The girl on the far right has blonde hair "
                "and wears a red dress, red outfit. Steady camera."),
    # §205: Docs/Outputs-archive/h3-outpaint-2026-11-10/h3_outpaint_gate.py DEFAULT_PROMPT
    "outpaint": ("A wide shot of a young woman in a black-and-white maid costume talking cheerfully to the camera in a "
                 "spacious pastel room decorated with sweets and flowers; natural head and hand movement, steady camera."),
}
EDIT = {"R1": ("retake", True), "R1c": ("retake", False), "I1": ("inpaint", True), "I1c": ("inpaint", False),
        "O1": ("outpaint", True), "O1c": ("outpaint", False)}
RAW_FRAMES = {"inpaint": (0, 36, 72), "outpaint": (0, 61, 123)}


def material_path(key: str) -> Path:
    return MATERIAL_DIR / MATERIALS[key]["file"]


def missing_material(kind: str) -> list[str]:
    return [rel(material_path(k)) for k in KIND_MATERIALS[kind] if not material_path(k).is_file()]


def edit_request(item: str, out_dir: Path):
    """R*: the chain's retake spec (worker._do_generate_chain's shape, one clip); I* / O*: a single request."""
    kind, pi = EDIT[item]
    mats = {k: str(material_path(k)) for k in KIND_MATERIALS[kind]}
    loras = lora_for(EDIT_STEPS) if pi else (P.H3LoraInput(path=str(ROOT / TURBO_LORA_FILE), strength=TURBO_LORA_STRENGTH),)
    base = P.H3GenerateRequest(
        prompt=EDIT_PROMPTS[kind], seed=DEFAULT_SEED, width=WIDTH, height=HEIGHT, num_frames=EDIT_FRAMES[kind],
        output_path=str(out_dir / "output.mp4"), num_inference_steps=EDIT_STEPS,
        shift_video=SHIFT_VIDEO if pi else CONTROL_SHIFT_VIDEO, shift_audio=SHIFT_AUDIO if pi else CONTROL_SHIFT_AUDIO,
        keep_resident=KEEP_RESIDENT, attention_backend=ATTENTION, block_swap_prefetch=PREFETCH, vram16_mode=VRAM16,
        loras=loras,
    )
    if kind == "retake":
        from engine_h3.chain_h3 import H3ChainClipSpec, H3ChainRetake, H3ChainSpec  # noqa: PLC0415

        return H3ChainSpec(base=base, clips=(H3ChainClipSpec(prompt=base.prompt, num_frames=base.num_frames),),
                           seed=DEFAULT_SEED,
                           retake=H3ChainRetake(path=mats["R1_window"], audio_strength=EDIT_AUDIO_STRENGTH))
    if kind == "outpaint":
        return dataclasses.replace(base, outpaint=P.H3OutpaintInput(
            path=mats["O1_source"], audio_strength=EDIT_AUDIO_STRENGTH, feather_px=FEATHER_PX, **OUTPAINT_PADS))
    return dataclasses.replace(base, inpaint=P.H3InpaintInput(
        path=mats["I1_source"], mask_path=mats["I1_mask"], audio_strength=EDIT_AUDIO_STRENGTH, feather_px=FEATHER_PX))


def check_edit_request(item: str, req) -> dict:
    if EDIT[item][0] == "retake":
        chk = check_chain_spec(req)
        lay = chk["layout"]
        return {"validate": "passed (chain _check_spec + validate_request)", "clip_requests": chk["clip_requests"],
                "seeds": chk["seeds"], "total_frames": lay.total_frames, "total_latent_frames": lay.total_latent_frames,
                "retake": lay.to_dict().get("retake")}
    P.validate_request(req)
    return {"validate": "passed (validate_request)"}


def _source_size(path: Path) -> tuple[int, int]:
    v = ffprobe(path)["video"]
    return int(v["width"]), int(v["height"])


def expected_rows(kind: str) -> dict:
    """CPU: the fixed / generated video rows and the 0 / 1 mask values the product builds, through the
    product's own functions on zero latents (retake_fix / outpaint_fix / apply_inpaint_tokens; the
    Inpainting mask read with read_inpaint_mask from the cut mask). Audio: every read latent assumed
    (te_a = A), the widest fixed range."""
    frames = EDIT_FRAMES[kind]
    t, a = latent_frames(frames), audio_latents(frames)
    gh, gw = HEIGHT // 32, WIDTH // 32
    out: dict = {"latent_frames": t, "audio_latents": a, "grid": [gh, gw]}
    if kind == "retake":
        from engine_h3.chain_h3 import retake_fix  # noqa: PLC0415
        from h3_chain_math import compute_h3_chain_layout  # noqa: PLC0415

        layout = compute_h3_chain_layout([frames], retake=True)
        vfix, afix = retake_fix(torch.zeros((1, 1, t, 2 * gh, 2 * gw)), torch.zeros((2, 1, a)), layout,
                                EDIT_AUDIO_STRENGTH)
        values_v = torch.tensor(vfix[1], dtype=torch.float32).repeat_interleave(gh * gw)
        out["fixed_latents"] = [int(layout.retake_n_head_v), int(layout.retake_n_tail_v)]
    else:
        from engine_h3 import outpaint_h3 as OH  # noqa: PLC0415
        from engine_h3.outpaint_geometry import LATENT_PX, compute_outpaint_geometry  # noqa: PLC0415

        if kind == "outpaint":
            sw, sh = _source_size(material_path("O1_source"))
            p = OUTPAINT_PADS
            geom = compute_outpaint_geometry(sw, sh, WIDTH, HEIGHT, p["pad_left"], p["pad_right"], p["pad_top"],
                                             p["pad_bottom"])
        else:
            from engine_h3.inpaint_geometry import compute_inpaint_geometry  # noqa: PLC0415

            sw, sh = _source_size(material_path("I1_source"))
            geom = compute_inpaint_geometry(sw, sh, WIDTH, HEIGHT)
        vfix, afix = OH.outpaint_fix(torch.zeros((1, 1, t, geom.aligned_h // LATENT_PX, geom.aligned_w // LATENT_PX)),
                                     torch.zeros((2, 1, a)), geom, EDIT_AUDIO_STRENGTH, t, a)
        if kind == "inpaint":
            from engine_h3 import inpaint_h3 as IH  # noqa: PLC0415
            from engine_h3.inpaint_geometry import chunk_block_masks, token_values_from_chunks  # noqa: PLC0415

            blocks, proof = IH.read_inpaint_mask(str(material_path("I1_mask")), frames, (sw, sh), geom)
            vfix = IH.apply_inpaint_tokens(vfix, token_values_from_chunks(chunk_block_masks(blocks, frames), t))
            out["mask_proof"] = proof
        values_v = vfix[1].reshape(-1)
        out.update(source=[sw, sh], canvas=[geom.canvas_w, geom.canvas_h], geometry_fixed_tokens=int(geom.fixed_count),
                   geometry_generated_tokens=int(geom.generated_count))
    out["video_values"] = sorted({float(v) for v in torch.unique(values_v)})
    out["audio_values"] = sorted({float(v) for v in afix[1]}) if afix is not None else None
    out["fixed"], out["generated"] = int((values_v == 0).sum()), int((values_v == 1).sum())
    out["only_0_and_1"] = set(out["video_values"]) <= {0.0, 1.0} and set(out["audio_values"] or []) <= {0.0, 1.0}
    return out


PASTE: dict = {}


def _paste_owner(kind: str):
    from engine_h3 import inpaint_geometry as IG  # noqa: PLC0415
    from engine_h3 import outpaint_geometry as OG  # noqa: PLC0415

    # generate() imports both at call time (pipeline_h3.py 1131 / 1141), so the module attribute is what it gets
    return (IG, "compose_inpaint") if kind == "inpaint" else (OG, "composite")


@contextlib.contextmanager
def paste_wrap(kind: str, raw_dir: Path | None):
    """The product's paste-back function wrapped: it saves frames ``RAW_FRAMES[kind]`` of the decoded canvas
    (before the paste-back) as PNG into ``raw_dir`` (None = save nothing), then calls the original."""
    from PIL import Image  # noqa: PLC0415

    owner, attr = _paste_owner(kind)
    orig = getattr(owner, attr)
    if getattr(orig, "_spike_wrap", False):
        raise RuntimeError(f"{attr} is already wrapped")
    PASTE.clear()
    PASTE.update(kind=kind, function=f"{owner.__name__}.{attr}", calls=0, saved=[])

    def wrapped(decoded, *args, **kwargs):
        PASTE["calls"] += 1
        PASTE["decoded"] = {"shape": list(decoded.shape), "dtype": str(decoded.dtype)}
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            for f in RAW_FRAMES[kind]:
                if f < decoded.shape[0]:
                    path = raw_dir / f"f{f:03d}.png"
                    Image.fromarray(decoded[f].copy()).save(path)
                    PASTE["saved"].append(rel(path))
        return orig(decoded, *args, **kwargs)

    wrapped._spike_wrap = True
    setattr(owner, attr, wrapped)
    try:
        yield wrapped
    finally:
        setattr(owner, attr, orig)
        PASTE["restored"] = getattr(owner, attr) is orig


def edit_conditions(item: str) -> dict:
    kind, pi = EDIT[item]
    rec = conditions_record(EDIT_STEPS, "fl2va", WIDTH, HEIGHT, EDIT_FRAMES[kind], [EDIT_PROMPTS[kind]])
    rec.update(kind=kind, pi_h3_on=pi, audio_strength=EDIT_AUDIO_STRENGTH,
               materials={k: rel(material_path(k)) for k in KIND_MATERIALS[kind]})
    if kind != "retake":
        rec.update(feather_px=FEATHER_PX, feather_steps=FEATHER_STEPS, raw_frames=list(RAW_FRAMES[kind]))
    if kind == "outpaint":
        rec["outpaint"] = {"source": list(OUTPAINT_SOURCE), **OUTPAINT_PADS}
    if not pi:
        rec.update(shift_video=CONTROL_SHIFT_VIDEO, shift_audio=CONTROL_SHIFT_AUDIO, pi_h3=None, pi_flow=None,
                   lora={"file": Path(TURBO_LORA_FILE).name, "strength": TURBO_LORA_STRENGTH},
                   sigmas="the product's sigma_schedule", decoder="the product's H3VideoDecoder")
    return rec


def run_edit(item: str) -> int:
    from engine_h3.chain_h3 import run_chain_h3  # noqa: PLC0415

    kind, pi = EDIT[item]
    frames = EDIT_FRAMES[kind]
    out_dir, rec = _begin(item)
    result_path = out_dir / "result.json"
    rec["conditions"] = edit_conditions(item)
    missing = missing_material(kind)
    if missing:
        rec.update(status="refused", material_missing=missing, finished_at=now())
        write_json(result_path, rec)
        LOG(f"素材なし: {missing}（先に --prepare）")
        LOG.close()
        return 2
    if not _gate_or_refuse(rec, result_path):
        return 2
    ok = False
    try:
        req = edit_request(item, out_dir)
        rec["request_check"] = check_edit_request(item, req)
        rec["request"] = dataclasses.asdict(req)
        rec["expected_rows"] = expected_rows(kind)
        STATE.reset(steps=EDIT_STEPS, pi_path=PA.PI_FILES[EDIT_STEPS] if pi else None, out_dir=out_dir)
        PASTE.clear()
        write_json(result_path, rec)
        clock = StepClock()
        with contextlib.ExitStack() as stack:
            if pi:
                stack.enter_context(spike_patches())
                rec["patches_in"] = patches_in()
            if kind != "retake":
                w = stack.enter_context(paste_wrap(kind, out_dir / "raw"))
                owner, attr = _paste_owner(kind)
                rec["paste_in"] = getattr(owner, attr) is w
            pipe = _pipeline("fl2va", clock)
            try:
                t0 = time.perf_counter()
                rec["build_report"] = pipe.load()
                rec["load_seconds"] = round(time.perf_counter() - t0, 2)
                write_json(result_path, rec)
                result = run_chain_h3(pipe, req, clock, LOG) if kind == "retake" else pipe.generate(req)
                rec["engine"] = {"seconds": result.get("seconds"), "peak_vram_mb": result.get("peak_vram_mb"),
                                 "peak_vram_reserved_mb": result.get("peak_vram_reserved_mb"), "full": result}
            finally:
                try:
                    pipe.close()
                except Exception as exc:  # noqa: BLE001
                    LOG(f"pipe.close failed (ignored): {exc!r}")
        rec["paste"] = dict(PASTE) if kind != "retake" else None
        rec["restored"] = patches_restored() and (kind == "retake" or bool(PASTE.get("restored")))
        ok = _finish_run(rec, out_dir, clock, steps=EDIT_STEPS, n_sessions=1, frames=frames, pi=pi,
                         lora_pairs=None if pi else TURBO_LORA_PAIRS, rows=rec["expected_rows"], out_frames=frames)
        if kind != "retake" and (PASTE.get("calls") != 1 or len(PASTE.get("saved") or []) != len(RAW_FRAMES[kind])):
            rec["self_check"]["failures"].append(f"paste-back wrapper: {PASTE.get('calls')} calls, "
                                                 f"{len(PASTE.get('saved') or [])} raw frames saved")
            rec["self_check"]["passed"] = ok = False
        LOG(f"{item}: done; self_check={rec['self_check']}")
    except BaseException as exc:  # noqa: BLE001
        failure(rec, exc)
        rec["restored"] = patches_restored() and (kind == "retake" or PASTE.get("restored", True))
        rec["paste"] = dict(PASTE) if kind != "retake" else None
        rec["sessions"] = sessions_record(STATE.sessions)
        rec["adapter"] = STATE.adapter_reports
        rec["dwd"] = STATE.decoder
        rec["hits"] = dict(STATE.hits)
    finally:
        rec["finished_at"] = now()
        write_json(result_path, rec)
        LOG.close()
    return 0 if ok else 1


def _diff(a: torch.Tensor, b: torch.Tensor, ua: torch.Tensor, ub: torch.Tensor) -> dict:
    d = (a - b).abs()
    du = (ua.to(torch.int16) - ub.to(torch.int16)).abs()
    return {"float_max_abs_diff": float(d.max()), "float_mean_abs_diff": float(d.mean()),
            "uint8_differing_ratio": float((du > 0).float().mean()), "uint8_max_diff": int(du.max())}


def run_split(source_item: str, contents, margins) -> int:
    from safetensors import safe_open  # noqa: PLC0415

    from engine_h3.quant_linear import require_cuda_quant  # noqa: PLC0415

    name = "C4_" + source_item.split("_", 1)[1]
    out_dir, rec = _begin(name)
    result_path = out_dir / "result.json"
    src = RUNS / source_item / HIDDEN_FILE
    rec["source"] = {"item": source_item, "file": rel(src), "exists": src.is_file(),
                     "contents": list(contents), "margins": list(margins)}
    if not _gate_or_refuse(rec, result_path):
        return 2
    ok = False
    try:
        if not src.is_file():
            raise FileNotFoundError(f"{rel(src)} がありません（先に --chain {source_item} を回す）")
        with safe_open(str(src), "pt") as f:
            meta = f.metadata()
            hidden, temb, tidx = f.get_tensor("hidden"), f.get_tensor("t_emb"), f.get_tensor("time_index")
        grid = tuple(json.loads(meta["grid"]))
        rec["source"].update(meta=meta, hidden_shape=list(hidden.shape))
        rec["backends"] = {k: bool((v or {}).get("available")) if isinstance(v, dict) else str(v)
                           for k, v in require_cuda_quant((h3_format.INT8,)).items()}
        dev = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()
        head, info = PD.load_head(ROOT / meta["pi_file"], dev)
        rec["head_load"] = {**info, "peak_alloc_mib": int(round(torch.cuda.max_memory_allocated() / 2**20))}
        feats = PD.normalize_features(head, hidden.to(dev), temb, tidx)
        rec["features_stats"] = tensor_stats(feats)
        whole, wchunks = PD.decode(head, feats, grid, measure_peak=True)
        uwhole, wstats = PD.to_uint8(whole)
        rec["whole"] = {"chunks": wchunks, "frames": int(whole.shape[0]), **wstats}
        write_json(result_path, rec)
        rec["variants"] = {}
        for content in contents:
            for margin in margins:
                key = f"content{content}_margin{margin}"
                px, chunks = PD.decode(head, feats, grid, content_blocks=content, margin_blocks=margin, measure_peak=True)
                upx, _ = PD.to_uint8(px)
                rec["variants"][key] = {"frames": int(px.shape[0]), "chunks": chunks,
                                        "seconds_total": round(sum(c["seconds"] for c in chunks), 3),
                                        "peak_alloc_mib_max": max(c.get("peak_alloc_mib", 0) for c in chunks),
                                        "vs_whole": _diff(px, whole, upx, uwhole) if px.shape == whole.shape else None}
                LOG(f"split {key}: {rec['variants'][key]['vs_whole']} peak {rec['variants'][key]['peak_alloc_mib_max']} MiB")
                del px, upx
                write_json(result_path, rec)
        rec["status"] = "done"
        ok = all(v["vs_whole"] is not None for v in rec["variants"].values())
    except BaseException as exc:  # noqa: BLE001
        failure(rec, exc)
    finally:
        rec["finished_at"] = now()
        write_json(result_path, rec)
        LOG.close()
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Dry run (CPU only)
# --------------------------------------------------------------------------- #

def _bias_dtype_of(header, cfg):
    from engine.sft_quant.sft_reader import TORCH_DTYPES  # noqa: PLC0415

    def bias_dtype_of(name: str) -> torch.dtype:          # pipeline_h3.py 1583-1585
        info = header.tensors.get(f"{cfg.prefix}{name}.bias")
        return TORCH_DTYPES.get(info.dtype, torch.bfloat16) if info is not None else torch.bfloat16
    return bias_dtype_of


def meta_dit(path):
    """The base file's DiT built on meta exactly as _build_dit does, without reading any weight."""
    from engine_h3.quant_linear import swap_in_int8_linears  # noqa: PLC0415

    header, cfg = h3_format.inspect_h3_dit(str(path))
    with torch.device("meta"):
        dit = D.H3DiT(cfg)
        swap_in_int8_linears(dit, cfg.quant, bias_dtype_of=_bias_dtype_of(header, cfg))
    return header, cfg, dit


def _mask_simulation() -> dict:
    """The chain's 0 / 1 masks: scaling the whole state by sigma_0 vs only the generated rows gives the
    same DiT inputs at every step and the same final latents (fake mixing velocity, real mask helpers)."""
    g = torch.Generator().manual_seed(7)
    clean = torch.randn((1, 24, 6, 4, 4), generator=g)
    noise = torch.randn((1, 24, 6, 4, 4), generator=g)
    vm = DM.video_mask(clean, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0], grid=4, noise=noise, inject_clean=1.0)
    sig = pih3_sigmas(2, SHIFT_VIDEO)

    def run(scale_all: bool):
        x = noise.clone()
        x = x * float(sig[0]) if scale_all else torch.where(vm.m == 1, x * float(sig[0]), x)
        inputs = []
        for i in range(2):
            x_in = _ORIG_DIT_INPUT(x, vm)
            inputs.append(x_in.clone())
            v = -x_in + x_in.mean(dim=(2, 3, 4), keepdim=True)     # mixes every row (as attention does)
            v = DM.state_velocity(v, x, vm, float(sig[i]))
            x = euler_step(x, v, float(sig[i]), float(sig[i + 1]))
        return DM.finish(x, vm), inputs

    a, ia = run(True)
    b, ib = run(False)
    return {"same_dit_inputs": all(torch.equal(p, q) for p, q in zip(ia, ib)), "same_result": bool(torch.equal(a, b)),
            "fixed_rows_end_clean": bool(torch.equal(a[:, :, :2], clean[:, :, :2]))}


def dry_patches() -> dict:
    rec: dict = {}
    STATE.reset(steps=2)
    with spike_patches():
        rec["patches_in"] = patches_in()
        sv = P.sigma_schedule(2, SHIFT_VIDEO)
        sa = P.sigma_schedule(2, SHIFT_AUDIO)
        video = torch.ones((1, 24, 2, 2, 2))
        audio = torch.ones((2, 32, 3))
        out_a = DM.dit_input(audio, None)
        out_v = DM.dit_input(video, None)
        again = DM.dit_input(video, None)
        rec["dit_input"] = {
            "audio_scaled_in_place": out_a is audio and torch.allclose(audio, torch.full_like(audio, float(sa[0]))),
            "video_scaled_in_place": out_v is video and torch.allclose(video, torch.full_like(video, float(sv[0]))),
            "second_call_unchanged": again is video and torch.allclose(video, torch.full_like(video, float(sv[0]))),
            "record": STATE.sessions[-1]["noise_scale"]}
    rec["restored"] = patches_restored()
    rec["mask_simulation"] = _mask_simulation()
    d = rec["dit_input"]
    rec["passed"] = (all(rec["patches_in"].values()) and rec["restored"] and d["audio_scaled_in_place"]
                     and d["video_scaled_in_place"] and d["second_call_unchanged"]
                     and all(rec["mask_simulation"].values()))
    return rec


def dry_run(args) -> int:
    LOG.open(OUT_ROOT / "dry_run.log", mode="w")
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
        write_json(OUT_ROOT / "dry_run.json", rec)

    pi_headers = {s: sft_quant_format.read_header(str(ROOT / p)) for s, p in PA.PI_FILES.items()}
    pi_keys = {s: PA.classify(h) for s, h in pi_headers.items()}
    metas = {}

    def it1():
        return dry_patches()

    def it2():
        out = {}
        for base, path in BASES.items():
            header, cfg, dit = meta_dit(ROOT / path)
            metas[base] = (header, cfg, dit)
            tab = PA.dtype_table(header, cfg)
            r = {"base": file_info(path), "dtype_table": tab,
                 "dtype_problems": PA.dtype_expectation_problems(base, tab), "pi": {}}
            for s, ph in pi_headers.items():
                keys = pi_keys[s]
                r["pi"][s] = {"counts": PA.counts(keys), "count_problems": PA.count_problems(keys),
                              **PA.model_match(keys, ph, dit, header, cfg)}
            r["passed"] = not r["dtype_problems"] and all(
                v["passed"] and not v["count_problems"] for v in r["pi"].values())
            out[base] = r
        out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
        return out

    def it3():
        from engine_h3 import lora  # noqa: PLC0415

        out = {}
        want = {f"blocks.{i}.{leaf}" for i in PA.LORA_BLOCKS for leaf in PA.LEAVES}
        for s, path in PA.PI_FILES.items():
            kmap = lora.inspect_lora_file(str(ROOT / path))
            r = {"pairs": len(kmap.pairs), "pairs_are_blocks_0_45": set(kmap.pairs) == want,
                 "alpha_keys": sum(1 for k in kmap.pairs.values() if k.alpha is not None),
                 "unmapped": len(kmap.unmapped), "supported_flag": kmap.supported, "layout": kmap.layout,
                 "attach_on_meta": {}}
            h = pi_headers[s].tensors
            for base, (_hdr, _cfg, dit) in metas.items():
                mods = {m: (torch.empty(tuple(h[k.a].shape), dtype=torch.float16, device="meta"),
                            torch.empty(tuple(h[k.b].shape), dtype=torch.float16, device="meta"), 1.0)
                        for m, k in kmap.pairs.items()}
                f = lora.H3LoraFile(name=Path(path).name, layout=kmap.layout, dtype="F16", modules=mods,
                                    skipped=[{"module": "(unmapped keys)", "reason": f"{len(kmap.unmapped)} key(s)"}])
                rep = lora.attach(dit, [(f, 1.0)])[0]
                cleaned = lora.detach(dit)
                r["attach_on_meta"][base] = {"pairs_applied": rep["pairs_applied"], "skipped": len(rep["skipped"]),
                                             "detached_modules": cleaned}
            r["passed"] = (r["pairs"] == 138 and r["pairs_are_blocks_0_45"] and r["alpha_keys"] == 0
                           and r["unmapped"] == len(h) - 276
                           and all(v["pairs_applied"] == 138 and v["skipped"] == 1 for v in r["attach_on_meta"].values()))
            out[s] = r
        out["code_reading"] = ("validate_request calls inspect_lora_file for the header only and never reads "
                               "'supported' (pipeline_h3.py 336-351); load_lora_file records the unmapped keys as one "
                               "skipped entry and reads only the mapped pairs (lora.py 123-147); attach skips "
                               "per module (lora.py 187-207). Nothing refuses the job.")
        out["passed"] = all(out[s]["passed"] for s in PA.PI_FILES)
        return out

    def it4():
        out = {}
        for s, path in PA.PI_FILES.items():
            m12 = PA.read_markers(ROOT / path, pi_headers[s], pi_keys[s])
            mdwd = PD.read_markers(ROOT / path, pi_headers[s])
            specs = {k: h3_format.parse_marker(v, accepted=h3_format.DIT_FORMATS) for k, v in {**m12, **mdwd}.items()}
            r = {"dit_layers": len(m12), "dwd_layers": len(mdwd), "problems": PA.marker_problems(m12) + PA.marker_problems(mdwd),
                 "specs": sorted({f"{q.format}/convrot={q.convrot}/g{q.convrot_groupsize}" for q in specs.values()})}
            r["passed"] = len(m12) == 12 and len(mdwd) == 32 and not r["problems"] and r["specs"] == [
                "int8_tensorwise/convrot=True/g256"]
            out[s] = r
        out["passed"] = all(out[s]["passed"] for s in PA.PI_FILES)
        return out

    def it5():
        out = {}
        for steps, exp in EXPECTED_SIGMAS.items():
            shifts = {"video": SHIFT_VIDEO, "audio": SHIFT_AUDIO}
            got = {k: [float(v) for v in pih3_sigmas(steps, s)] for k, s in shifts.items()}
            # independent float64 closed form: raw_t evenly spaced from max_raw_t to max_raw_t * f / (n - 1 + f)
            m = MAX_RAW_T[steps]
            end = m * FINAL_STEP_SIZE_SCALE / (steps - 1 + FINAL_STEP_SIZE_SCALE)
            raw = [m + i * (end - m) / (steps - 1) for i in range(steps)]
            exact = {k: [s * r / (1 + (s - 1) * r) for r in raw] + [0.0] for k, s in shifts.items()}
            d_exact = {k: max(abs(a - b) for a, b in zip(got[k], exact[k])) for k in got}
            # the brief's values are rounded (4-6 significant digits): compared at half a unit of their last digit
            tol_given = {k: [0.5 * 10 ** -len(f"{v:.10g}".split(".")[1]) if "." in f"{v:.10g}" else 0.5 for v in exp[k]]
                         for k in exp}
            given_ok = {k: all(abs(a - b) <= t + 1e-12 for a, b, t in zip(got[k], exp[k], tol_given[k])) for k in got}
            d_given = {k: max(abs(a - b) for a, b in zip(got[k], exp[k])) for k in got}
            shape_ok = all(len(got[k]) == steps + 1 and got[k][-1] == 0.0 for k in got)
            out[steps] = {"got": got, "exact_float64": exact, "max_abs_diff_vs_exact": d_exact,
                          "given": exp, "max_abs_diff_vs_given": d_given, "given_within_its_digits": given_ok,
                          "passed": shape_ok and all(v <= SIGMA_TOL for v in d_exact.values()) and all(given_ok.values())}
        ref = P.sigma_schedule(4, SHIFT_VIDEO)
        out["engine_contract"] = {"dtype": str(ref.dtype), "len_for_4_steps": int(ref.numel()), "last": float(ref[-1])}
        try:
            pih3_sigmas(3, SHIFT_VIDEO)
            out["refuses_3_steps"] = False
        except ValueError:
            out["refuses_3_steps"] = True
        out["passed"] = all(out[s]["passed"] for s in EXPECTED_SIGMAS) and out["refuses_3_steps"] and \
            pih3_sigmas(2, 12).dtype == ref.dtype
        return out

    def it6():
        out = {s: PD.meta_check(ROOT / path, pi_headers[s]) for s, path in PA.PI_FILES.items()}
        out["passed"] = all(out[s]["passed"] for s in PA.PI_FILES)
        return out

    def it7():
        from h3_chain_math import compute_h3_chain_layout  # noqa: PLC0415

        cfg = PD.HeadConfig()
        t1 = latent_frames(NUM_FRAMES)
        layout = compute_h3_chain_layout([CHAIN_FRAMES, CHAIN_FRAMES])
        k = int(layout.overlap_latents)
        per_clip = [latent_frames(CHAIN_FRAMES)] * 2
        stitched = per_clip[0] + (per_clip[1] - k)
        g1 = PD.build_geometry(t1, HEIGHT // 32 * cfg.expansion[0], WIDTH // 32 * cfg.expansion[1], cfg.rope_dim)
        g2 = PD.build_geometry(stitched, CHAIN_HEIGHT // 32 * 2, CHAIN_WIDTH // 32 * 2, cfg.rope_dim)
        plans = {}
        for content in SPLIT_CONTENT:
            for margin in SPLIT_MARGINS:
                plan = PD.plan_chunks(stitched, content, margin)
                bf = PD.block_frames(stitched)
                kept = [b for c in plan for b in range(*c["keep"])]
                plans[f"content{content}_margin{margin}"] = {
                    "chunks": len(plan), "kept_frames": sum(bf[b] for b in kept),
                    "every_block_kept_once": kept == list(range(len(bf))),
                    "latent_starts": [c["context"][0] * PD.BLOCK for c in plan]}
        out = {
            "single": {"latents": t1, "frames": PD.frames_for_latents(t1), "expected": NUM_FRAMES,
                       "sol_grid": [g1.T, g1.H, g1.W], "native_blocks": g1.nb, "windows": g1.window_count},
            "chain": {"clip_latents": per_clip, "fixed_head_latents": k, "stitched_latents": stitched,
                      "layout_total_latents": layout.total_latent_frames, "frames": PD.frames_for_latents(stitched),
                      "layout_total_frames": layout.total_frames, "is_5n_plus_2": (stitched - 2) % 5 == 0,
                      "sol_grid": [g2.T, g2.H, g2.W], "native_blocks": g2.nb, "windows": g2.window_count},
            "split_plans": plans}
        out["passed"] = (out["single"]["frames"] == NUM_FRAMES and stitched == layout.total_latent_frames
                         and out["chain"]["frames"] == layout.total_frames and k == 7 and out["chain"]["is_5n_plus_2"]
                         and all(p["kept_frames"] == layout.total_frames and p["every_block_kept_once"]
                                 for p in plans.values()))
        return out

    def it8():
        return PD.chunking_self_test()

    def it9():
        import comfy_kitchen as ck  # noqa: PLC0415

        out = {"comfy_kitchen_version": metadata.version("comfy-kitchen"),
               "functions": {n: hasattr(ck, n) for n in ("int8_linear", "list_backends", "disable_backend")}}
        reqs = {}
        for item_name in SINGLE:
            req = single_request(item_name, RUNS / item_name)
            P.validate_request(req)
            reqs[item_name] = "validate_request passed"
        for item_name in CHAIN:
            chk = check_chain_spec(chain_spec(item_name, RUNS / item_name))
            reqs[item_name] = {"validate": "passed", "clip_requests": chk["clip_requests"], "seeds": chk["seeds"],
                               "total_frames": chk["layout"].total_frames}
        out["requests"] = reqs
        out["model_paths"] = h3_format.check_model_paths(
            files={k: str(ROOT / v) for k, v in PATHS.items() if k != "tokenizer_dir"} | {
                "sulphur": str(ROOT / BASES["sulphur"])},
            dirs={"tokenizer_dir": str(ROOT / PATHS["tokenizer_dir"])})
        out["passed"] = out["comfy_kitchen_version"] == "0.2.37" and all(out["functions"].values())
        return out

    reads: dict = {}

    def product_reads(kind: str) -> dict:
        """The product's own CPU reads of the cut material (once per kind, shared by the control)."""
        if kind in reads:
            return reads[kind]
        frames = EDIT_FRAMES[kind]
        if kind == "retake":
            from engine_h3.chain_h3 import H3ChainRetake, _read_retake_window  # noqa: PLC0415
            from h3_chain_math import compute_h3_chain_layout  # noqa: PLC0415

            spec = edit_request("R1", RUNS / "R1")
            layout = compute_h3_chain_layout([frames], retake=True)
            px, wave, info = _read_retake_window(
                H3ChainRetake(path=str(material_path("R1_window")), audio_strength=EDIT_AUDIO_STRENGTH), spec.base,
                layout, LOG)
            r = {"function": "chain_h3._read_retake_window", "frames": list(px.shape), "info": info,
                 "wave": None if wave is None else list(wave.shape),
                 "required_samples": int(round(frames * 4000 / 3))}
        else:
            from engine_h3 import outpaint_h3 as OH  # noqa: PLC0415

            path = material_path("O1_source" if kind == "outpaint" else "I1_source")
            px, wave, info = OH.read_outpaint_source(str(path), frames, FPS, EDIT_AUDIO_STRENGTH, log=LOG)
            r = {"function": "outpaint_h3.read_outpaint_source", "frames": list(px.shape), "info": info,
                 "wave": None if wave is None else list(wave.shape),
                 "required_samples": int(info["required_samples"])}
        del px, wave
        r["expected_rows"] = expected_rows(kind)
        r["passed"] = (r["frames"][0] == frames and r["wave"] is not None and r["wave"][-1] == r["required_samples"]
                       and r["expected_rows"]["only_0_and_1"])
        reads[kind] = r
        return r

    def it10():
        out = {}
        for name, (kind, pi) in EDIT.items():
            missing = missing_material(kind)
            if missing:
                out[name] = {"passed": False, "material": "素材なし", "missing": missing}
                continue
            try:
                req = edit_request(name, RUNS / name)
                base = req.base if kind == "retake" else req
                r = {"request_check": check_edit_request(name, req), "reads": product_reads(kind),
                     "lora": [Path(lo.path).name for lo in base.loras], "lora_strength": [lo.strength for lo in base.loras],
                     "shift": [base.shift_video, base.shift_audio], "steps": base.num_inference_steps,
                     "frames": base.num_frames, "canvas": [base.width, base.height]}
                if kind != "retake":
                    blk = base.outpaint or base.inpaint
                    r["feather_px"] = blk.feather_px
                    r["audio_strength"] = blk.audio_strength
                want_lora = Path(PA.PI_FILES[EDIT_STEPS] if pi else TURBO_LORA_FILE).name
                r["passed"] = (r["reads"]["passed"] and r["lora"] == [want_lora] and r["steps"] == EDIT_STEPS
                               and r.get("feather_px", FEATHER_PX) == 20 and FEATHER_PX == 20)
                out[name] = r
            except BaseException as exc:  # noqa: BLE001
                out[name] = {"passed": False}
                failure(out[name], exc)
        out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
        return out

    def it11():
        cfg = PD.HeadConfig()
        out = {}
        for kind in ("retake", "inpaint", "outpaint"):
            t = latent_frames(EDIT_FRAMES[kind])
            g = PD.build_geometry(t, HEIGHT // 32 * cfg.expansion[0], WIDTH // 32 * cfg.expansion[1], cfg.rope_dim)
            out[kind] = {"latents": t, "frames": PD.frames_for_latents(t), "expected_frames": EDIT_FRAMES[kind],
                         "sol_grid": [g.T, g.H, g.W], "native_blocks": g.nb, "windows": g.window_count,
                         "is_5n_plus_2": (t - 2) % 5 == 0}
        out["one_denoise_drops_nothing"] = ("DWDDecoderStandIn.decode drops k * hw head rows only for denoise "
                                            "i > 0; an edit item has one denoise (retake: run_chain_h3 with one "
                                            "clip; inpaint / outpaint: generate), so nothing is dropped")
        out["passed"] = (out["retake"]["latents"] == 37 and out["retake"]["frames"] == 124
                         and out["inpaint"]["latents"] == 22 and out["inpaint"]["frames"] == 73
                         and out["outpaint"]["frames"] == 124)
        return out

    def it12():
        import numpy as np  # noqa: PLC0415

        from engine_h3.inpaint_geometry import compute_inpaint_geometry  # noqa: PLC0415
        from engine_h3.outpaint_geometry import compute_outpaint_geometry, feather_weights  # noqa: PLC0415

        out = {}
        rng = np.random.default_rng(5)
        for kind in ("inpaint", "outpaint"):
            owner, attr = _paste_owner(kind)
            orig = getattr(owner, attr)
            if kind == "outpaint":
                geom = compute_outpaint_geometry(64, 64, 128, 128, 32, 32, 32, 32)
                extra = (feather_weights(geom, 8), geom)
            else:
                geom = compute_inpaint_geometry(64, 64, 64, 64)
                blocks = np.zeros((1, geom.grid_h, geom.grid_w), dtype=bool)
                blocks[0, 0, 0] = True
                extra = (blocks, geom, 8)
            decoded = rng.integers(0, 256, (3, geom.canvas_h, geom.canvas_w, 3), dtype=np.uint8)
            source = rng.integers(0, 256, (3, geom.source_h, geom.source_w, 3), dtype=np.uint8)
            a = orig(decoded.copy(), source, *extra)
            with paste_wrap(kind, None) as w:
                mod = __import__(owner.__name__, fromlist=[attr])
                late = getattr(mod, attr) is w            # as generate()'s import at call time
                b = getattr(owner, attr)(decoded.copy(), source, *extra)
                calls = PASTE["calls"]
            out[kind] = {"function": f"{owner.__name__}.{attr}", "wrapped_while_active": late, "calls": calls,
                         "same_result_as_original": bool(np.array_equal(a, b)),
                         "restored": getattr(owner, attr) is orig and PASTE.get("restored") is True}
            out[kind]["passed"] = all(out[kind][k] for k in ("wrapped_while_active", "same_result_as_original",
                                                              "restored")) and calls == 1
        out["product_patches_untouched"] = patches_restored()
        out["passed"] = all(out[k]["passed"] for k in ("inpaint", "outpaint")) and out["product_patches_untouched"]
        return out

    item("1_imports_and_patches", it1)
    item("2_key_match_and_dtypes", it2)
    item("3_lora_pairs", it3)
    item("4_markers", it4)
    item("5_sigmas", it5)
    item("6_dwd_keys", it6)
    item("7_frame_counts", it7)
    item("8_chunked_decode_logic", it8)
    item("9_kitchen_and_requests", it9)
    item("10_edit_requests", it10)
    item("11_edit_dwd_grid", it11)
    item("12_paste_wrap", it12)
    rec["all_passed"] = all(v.get("passed") for v in items.values())
    rec["git_after"] = git_state()
    rec["status"] = "done"
    rec["finished_at"] = now()
    write_json(OUT_ROOT / "dry_run.json", rec)
    LOG(f"dry-run: all_passed={rec['all_passed']} -> {rel(OUT_ROOT / 'dry_run.json')}")
    LOG.close()
    return 0 if rec["all_passed"] else 1


# --------------------------------------------------------------------------- #
# --prepare / --post (CPU only, edit items)
# --------------------------------------------------------------------------- #

def cut_material(key: str) -> dict:
    """One material: the services/video_io.py cut_window_mp4 command (no fps conversion: the sources are
    24 fps, refused otherwise) with the Outpainting source's scale added (h3_outpaint_gate.py 413-415)."""
    m = MATERIALS[key]
    src, out = ROOT / m["src"], material_path(key)
    if not src.is_file():
        raise FileNotFoundError(f"元の素材がありません: {rel(src)}")
    sp = ffprobe(src)
    if sp["video"].get("r_frame_rate") != f"{FPS}/1":
        raise RuntimeError(f"元の素材が {FPS}fps ではありません（{sp['video'].get('r_frame_rate')}）: {rel(src)}")
    start = round(float(m["start_sec"]) * FPS)
    end = start + int(m["frames"]) - 1
    if end > int(sp["video"]["nb_read_frames"]) - 1:
        raise RuntimeError(f"窓 [{start}, {end}] が元の素材（{sp['video']['nb_read_frames']} 枚）を越えます")
    scale = f",scale={m['scale'][0]}:{m['scale'][1]}:flags=lanczos" if m["scale"] else ""
    parts = [f"[0:v]select='between(n\\,{start}\\,{end})',setpts=PTS-STARTPTS{scale}[v]"]
    has_audio = sp["audio"] is not None
    if has_audio:
        parts.append(f"[0:a]atrim=start={start / FPS}:end={(end + 1) / FPS},asetpts=PTS-STARTPTS[a]")

    def cut(audio_args):
        cmd = [tool("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-filter_complex", ";".join(parts), "-map", "[v]"]
        if has_audio:
            cmd += ["-map", "[a]", *audio_args]
        cmd += ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", str(FPS), str(out)]
        return cmd, run_cmd(cmd)

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd, proc = cut(["-c:a", "pcm_s16le"])
    codec = "pcm_s16le" if has_audio else None
    if proc.returncode != 0 and has_audio:
        cmd, proc = cut(["-c:a", "aac", "-b:a", "192k"])
        codec = "aac 192k"
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg の切り出しに失敗しました（{proc.returncode}）: {proc.stderr[-2000:]}")
    probe = ffprobe(out)
    vid = probe["video"]
    want = tuple(m["scale"]) if m["scale"] else (int(sp["video"]["width"]), int(sp["video"]["height"]))
    if int(vid.get("nb_read_frames") or -1) != m["frames"] or (vid.get("width"), vid.get("height")) != want:
        raise RuntimeError(f"切り出しの枚数か寸法が合いません（{vid.get('nb_read_frames')} 枚・"
                           f"{vid.get('width')}x{vid.get('height')}。{m['frames']} 枚・{want[0]}x{want[1]} が必要）")
    LOG(f"prepare: {key} -> {rel(out)} {vid.get('width')}x{vid.get('height')} {vid.get('nb_read_frames')} frames "
        f"audio={probe['audio']}")
    return {"source": rel(src), "source_probe": sp, "cut": rel(out), "ffmpeg_cmd": cmd, "audio_codec": codec,
            "start_sec": float(m["start_sec"]), "start_frame": start, "end_frame": end,
            "scale": list(m["scale"]) if m["scale"] else None, "probe": probe}


def prepare() -> int:
    """Cut every material that is not there yet (an existing one is kept, as h3_outpaint_gate.py's prepare)."""
    MATERIAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG.open(MATERIAL_DIR / "prepare.log", mode="a")
    pj = MATERIAL_DIR / "prepare.json"
    data = json.loads(pj.read_text(encoding="utf-8")) if pj.is_file() else {}
    ok = True
    for key in MATERIALS:
        if material_path(key).is_file() and (data.get(key) or {}).get("probe"):
            LOG(f"prepare: 既存の {rel(material_path(key))} を使う（作り直さない）")
            continue
        try:
            data[key] = {**cut_material(key), "prepared_at": now()}
        except Exception as exc:  # noqa: BLE001
            data[key] = {"status": "failed"}
            failure(data[key], exc)
            ok = False
        write_json(pj, data)
    LOG(f"prepare: {'done' if ok else 'FAILED'} -> {rel(pj)}")
    LOG.close()
    return 0 if ok else 1


def _read_mp4(path: Path, n: int):
    from engine_h3.source_tail import read_source_frames  # noqa: PLC0415

    return read_source_frames(str(path), n) if path.is_file() else None


def _psnr(a, b) -> float:
    import numpy as np  # noqa: PLC0415

    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def save_sheet(path: Path, rows: list, col_labels: list, max_w: int = 4000, upscale: int = 1) -> str:
    """``rows``: [(label, [uint8 [H, W, 3], ...])], every tile the same size; scaled down to ``max_w``."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    h, w = rows[0][1][0].shape[:2]
    n = len(col_labels)
    s = min(float(upscale), max_w / float(n * w))
    tw, th = int(w * s), int(h * s)
    sheet = Image.new("RGB", (n * tw, len(rows) * th), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for r, (label, tiles) in enumerate(rows):
        for c, tile in enumerate(tiles):
            im = Image.fromarray(tile)
            if (tw, th) != (w, h):
                im = im.resize((tw, th), Image.LANCZOS)
            sheet.paste(im, (c * tw, r * th))
            text = f"{label} | {col_labels[c]}"
            draw.rectangle([c * tw, r * th, c * tw + 7 * len(text) + 8, r * th + 14], fill=(0, 0, 0))
            draw.text((c * tw + 4, r * th + 2), text, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    LOG(f"post: {rel(path)} {sheet.size[0]}x{sheet.size[1]}")
    return rel(path)


def signalstats(path: Path) -> list[tuple[float, float]]:
    """Per frame (YAVG, SATAVG) of ffmpeg's signalstats (parsed from metadata=mode=print on the log)."""
    proc = run_cmd([tool("ffmpeg"), "-hide_banner", "-nostats", "-v", "info", "-i", str(path), "-an",
                    "-vf", "signalstats,metadata=mode=print", "-f", "null", "-"])
    y = [float(v) for v in re.findall(r"lavfi\.signalstats\.YAVG=([0-9.]+)", proc.stderr)]
    s = [float(v) for v in re.findall(r"lavfi\.signalstats\.SATAVG=([0-9.]+)", proc.stderr)]
    if proc.returncode != 0 or len(y) != len(s):
        raise RuntimeError(f"signalstats failed for {rel(path)}: {proc.stderr[-500:]}")
    return list(zip(y, s))


def _rows(pairs) -> list:
    """[(label, frames or None)] -> the labels and frame arrays that exist."""
    return [(label, f) for label, f in pairs if f is not None]


def post_retake() -> dict:
    from engine_h3.source_tail import read_source_tail  # noqa: PLC0415

    n = EDIT_FRAMES["retake"]
    window = material_path("R1_window")
    orig, _info = read_source_tail(str(window), WIDTH, HEIGHT, n)     # the canvas the engine encodes (cover crop)
    outs = {k: _read_mp4(RUNS / k / "output.mp4", n) for k in ("R1", "R1c")}
    rows = _rows([("original window", orig), ("pi-H3 R1", outs["R1"]), ("control R1c", outs["R1c"])])
    rec: dict = {"rows": [r[0] for r in rows], "sheets": {}}

    def sheet(name: str, idx):
        rec["sheets"][name] = save_sheet(COMPARE / f"{name}.png", [(lab, [f[i] for i in idx]) for lab, f in rows],
                                         [f"f{i}" for i in idx])

    sheet("R1_boundary_head_f019-025", range(19, 26))
    sheet("R1_boundary_tail_f099-105", range(99, 106))
    sheet("R1_fixed_head_f000_f011_f021", (0, 11, 21))
    head, tail = range(0, 22), range(n - 22, n)
    rec["psnr_vs_original_window_db"] = {}
    for k, f in outs.items():
        if f is None:
            continue
        ph = [_psnr(f[i], orig[i]) for i in head]
        pt = [_psnr(f[i], orig[i]) for i in tail]
        rec["psnr_vs_original_window_db"][k] = {
            "head_f000-021": {"mean": round(statistics.mean(ph), 3), "min": round(min(ph), 3)},
            "tail_f102-123": {"mean": round(statistics.mean(pt), 3), "min": round(min(pt), 3)}}
    stats = {"window": signalstats(window)}
    for k in outs:
        if outs[k] is not None:
            stats[k] = signalstats(RUNS / k / "output.mp4")
    csv = COMPARE / "R1_signalstats.csv"
    cols = list(stats)
    lines = ["frame," + ",".join(f"{c}_yavg,{c}_satavg" for c in cols)]
    for i in range(n):
        lines.append(f"{i}," + ",".join(f"{stats[c][i][0]},{stats[c][i][1]}" if i < len(stats[c]) else ","
                                         for c in cols))
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rec["signalstats_csv"] = rel(csv)
    rec["signalstats_note"] = ("window = the cut mp4 itself (1920x1080, before the engine's cover crop); "
                               "R1 / R1c = the outputs (1280x768)")
    rec["signalstats_mean"] = {c: {"yavg": round(statistics.mean(v[0] for v in stats[c]), 3),
                                   "satavg": round(statistics.mean(v[1] for v in stats[c]), 3)} for c in cols}
    return rec


def _raw_sheet(kind: str, items: tuple, frame: int) -> str | None:
    from PIL import Image  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    tiles, labels = [], []
    for k, lab in zip(items, ("pi-H3 (DWD) raw", "control (VAE) raw")):
        p = RUNS / k / "raw" / f"f{frame:03d}.png"
        if p.is_file():
            tiles.append(np.asarray(Image.open(p).convert("RGB")))
            labels.append(f"{k} {lab}")
    if not tiles:
        return None
    pre = "I1" if kind == "inpaint" else "O1"
    return save_sheet(COMPARE / f"{pre}_raw_before_paste_f{frame:03d}.png", [(f"f{frame}", tiles)], labels)


def post_inpaint() -> dict:
    import numpy as np  # noqa: PLC0415

    n = EDIT_FRAMES["inpaint"]
    orig = _read_mp4(material_path("I1_source"), n)
    mask = _read_mp4(material_path("I1_mask"), n)
    outs = {k: _read_mp4(RUNS / k / "output.mp4", n) for k in ("I1", "I1c")}
    rows = _rows([("original", orig), ("pi-H3 I1", outs["I1"]), ("control I1c", outs["I1c"])])
    idx = RAW_FRAMES["inpaint"]
    rec: dict = {"rows": [r[0] for r in rows], "sheets": {}}
    rec["sheets"]["frames"] = save_sheet(COMPARE / "I1_frames_f000_f036_f072.png",
                                         [(lab, [f[i] for i in idx]) for lab, f in rows], [f"f{i}" for i in idx])
    white = np.zeros(mask.shape[1:3], dtype=bool)
    for i in idx:
        white |= mask[i][..., 0] >= 128        # inpaint_h3.read_inpaint_mask's rule (red >= 128)
    ys, xs = np.nonzero(white)
    m = 48
    y0, y1 = max(0, int(ys.min()) - m), min(white.shape[0], int(ys.max()) + 1 + m)
    x0, x1 = max(0, int(xs.min()) - m), min(white.shape[1], int(xs.max()) + 1 + m)
    rec["mask_zoom_box_xyxy"] = [x0, y0, x1, y1]
    rec["sheets"]["mask_zoom"] = save_sheet(
        COMPARE / "I1_mask_zoom_f000_f036_f072.png",
        [(lab, [np.ascontiguousarray(f[i, y0:y1, x0:x1]) for i in idx]) for lab, f in rows],
        [f"f{i} x{x0}-{x1} y{y0}-{y1}" for i in idx], upscale=4)
    rec["sheets"]["raw"] = _raw_sheet("inpaint", ("I1", "I1c"), 36)
    return rec


def post_outpaint() -> dict:
    import numpy as np  # noqa: PLC0415

    n = EDIT_FRAMES["outpaint"]
    src = _read_mp4(material_path("O1_source"), n)
    p = OUTPAINT_PADS
    canvas = np.full((n, HEIGHT, WIDTH, 3), 128, dtype=np.uint8)     # the pads shown grey
    canvas[:, p["pad_top"]:p["pad_top"] + src.shape[1], p["pad_left"]:p["pad_left"] + src.shape[2]] = src
    outs = {k: _read_mp4(RUNS / k / "output.mp4", n) for k in ("O1", "O1c")}
    rows = _rows([("source on canvas", canvas), ("pi-H3 O1", outs["O1"]), ("control O1c", outs["O1c"])])
    idx = RAW_FRAMES["outpaint"]
    rec: dict = {"rows": [r[0] for r in rows], "sheets": {}}
    rec["sheets"]["frames"] = save_sheet(COMPARE / "O1_frames_f000_f061_f123.png",
                                         [(lab, [f[i] for i in idx]) for lab, f in rows], [f"f{i}" for i in idx])
    half, f = 96, 61
    cx, cy = WIDTH // 2, HEIGHT // 2
    seams = {"left": (p["pad_left"], cy), "right": (WIDTH - p["pad_right"], cy),
             "top": (cx, p["pad_top"]), "bottom": (cx, HEIGHT - p["pad_bottom"])}
    rec["seam_centres_xy"] = {k: list(v) for k, v in seams.items()}
    rec["sheets"]["seams"] = save_sheet(
        COMPARE / "O1_seams_f061.png",
        [(lab, [np.ascontiguousarray(fr[f, y - half:y + half, x - half:x + half]) for x, y in seams.values()])
         for lab, fr in rows], [f"f{f} {k} seam" for k in seams], upscale=2)
    rec["sheets"]["raw"] = _raw_sheet("outpaint", ("O1", "O1c"), 61)
    return rec


POST = {"R1": post_retake, "I1": post_inpaint, "O1": post_outpaint}


def post(items: list[str]) -> int:
    COMPARE.mkdir(parents=True, exist_ok=True)
    LOG.open(COMPARE / "post.log", mode="a")
    path = COMPARE / "post.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    ok = True
    for it in items:
        try:
            data[it] = {**POST[it](), "made_at": now()}
        except Exception as exc:  # noqa: BLE001
            data[it] = {"status": "failed"}
            failure(data[it], exc)
            ok = False
        write_json(path, data)
    LOG.close()
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def run_vae_decode(item: str) -> int:
    """Control: decode the item's saved last_latent with the product's standard video VAE (pipeline_h3._decode
    2084-2103: from_file(video_vae, device, bfloat16); decode(video_lat, tiled=True)) -> output_vae.mp4."""
    from engine_h3.media_out import write_mp4  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    out_dir = RUNS / item
    lat_path = out_dir / "last_latent.safetensors"
    if not lat_path.is_file():
        print(f"no {lat_path}")
        return 2
    LOG.open(out_dir / "vae_decode.log", mode="w")
    install_log_bridge()
    result_path = out_dir / "vae_decode.json"
    rec = {"item": item, "mode": "vae_decode", "script": rel(Path(__file__)), "started_at": now(), "status": "running"}
    if not _gate_or_refuse(rec, result_path):
        return 2
    ok = False
    try:
        tensors = load_file(str(lat_path))
        rec["latent_file"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()}
        video_lat = tensors["video_latent"]
        expected = (int(video_lat.shape[2]) - 1) * 8 + 1 if video_lat.ndim == 5 else None
        device = torch.device("cuda")
        vae_path = str(ROOT / PATHS["video_vae"])
        rec["video_vae_file"] = Path(vae_path).name
        # same transfer as the product: the latent as the pipeline holds it (video_lat passed straight in)
        video_lat = video_lat.to(device)
        vae = VV.H3VideoDecoder.from_file(vae_path, device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            frames = vae.decode(video_lat, tiled=True)
            torch.cuda.synchronize()
        finally:
            rec["decode_seconds"] = round(time.perf_counter() - t0, 2)
            rec["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            rec["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
            del vae
            P._free_cuda()
        rec["latent_used"] = {"shape": list(video_lat.shape), "dtype": str(video_lat.dtype)}
        rec["frames"] = int(frames.shape[0])
        rec["frames_shape"] = list(frames.shape)
        write_json(result_path, rec)
        if int(frames.shape[0]) != 226:
            raise RuntimeError(f"decode: {frames.shape[0]} frames decoded, 226 expected")
        silent = out_dir / "output_vae_silent.mp4"
        final = out_dir / "output_vae.mp4"
        write_mp4(str(silent), frames, fps=24, audio=None)
        proc = run_cmd([tool("ffmpeg"), "-y", "-loglevel", "error", "-i", str(silent), "-i", str(out_dir / "output.mp4"),
                        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(final)])
        rec["mux"] = {"returncode": proc.returncode, "stderr": proc.stderr[-500:], "audio_from": "output.mp4 (-c copy)"}
        if proc.returncode != 0:
            raise RuntimeError(f"mux failed: {proc.stderr[-300:]}")
        silent.unlink()
        rec["output_probe"] = ffprobe(final)
        rec["status"] = "done"
        ok = True
    except BaseException as exc:  # noqa: BLE001
        failure(rec, exc)
    finally:
        rec["finished_at"] = now()
        write_json(result_path, rec)
        LOG.close()
    return 0 if ok else 1


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="CPU only: the checks of the plan's step 3")
    g.add_argument("--single", choices=sorted(SINGLE), help="GPU: one T2V item (P1..P4)")
    g.add_argument("--chain", choices=sorted(CHAIN), help="GPU: form 3 (2 clips, one DWD pass)")
    g.add_argument("--split", choices=sorted(CHAIN), help="GPU: form 4 on that C3 item's saved hidden states")
    g.add_argument("--vae-decode", choices=sorted(CHAIN), help="GPU: decode that item's last_latent with the product's video VAE")
    g.add_argument("--prepare", action="store_true", help="CPU only: cut the edit items' material into _material/")
    g.add_argument("--edit", choices=list(EDIT), help="GPU: one edit item (R1 / I1 / O1 pi-H3, R1c / I1c / O1c control)")
    g.add_argument("--post", choices=list(POST), nargs="+", help="CPU only: comparison sheets into runs/_compare/")
    ap.add_argument("--split-content", type=int, nargs="+", default=list(SPLIT_CONTENT),
                    help="native blocks kept per chunk (default 5 1)")
    ap.add_argument("--split-margins", type=int, nargs="+", default=list(SPLIT_MARGINS),
                    help="context blocks on each side (default 4 2 1)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    if args.prepare:
        return prepare()
    if args.edit:
        return run_edit(args.edit)
    if args.post:
        return post(args.post)
    if args.single:
        return run_single(args.single)
    if args.vae_decode:
        return run_vae_decode(args.vae_decode)
    if args.chain:
        return run_chain(args.chain)
    return run_split(args.split, args.split_content, args.split_margins)


if __name__ == "__main__":
    raise SystemExit(main())
