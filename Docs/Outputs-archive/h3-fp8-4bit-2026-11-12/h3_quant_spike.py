r"""MiniMax H3 fp8 / w4a8 spike (ledger §1-283, stage 0, 2026-11-12): prototype layers + measurements.

Plan (authoritative on scope and pass criteria): ``imperative-crafting-lightning.md``.
Implementer design (authoritative on code structure): ``DESIGN.md`` next to this file.
Patterns copied from ``Docs/Outputs-archive/h3-inpaint-2026-11-11/h3_inpaint_spike.py``
(header, ``Log``, ``PATHS``, ``build_request``, ``run_generation``, ``post_process``).

Product code is never edited: every change is a module-attribute assignment inside
:func:`spike_patches` (restored in ``finally``). Fidelity principle: the prototype layers
call the SAME comfy_kitchen public functions with the SAME arguments ComfyUI uses.

From the repo root (Git Bash):

    export PYTHONPATH="$PWD" PYTHONUTF8=1
    PY=./.venv-engine-h3/Scripts/python.exe
    S=outputs/h3-fp8-w4a8-spike-2026-11-12/h3_quant_spike.py
    $PY $S --dry-run          # CPU: inspectors, meta build, key map, dtype audit, S4
    $PY $S --s4               # CPU: sft_quant_format.layer_schemes only
    $PY $S --s1               # GPU: bit identity + backends + eager counting
    $PY $S --s2               # GPU: per-layer timing
    $PY $S --s3 fp8           # GPU: one end-to-end condition (int8|fp8|fp8_dequant|w4a8_kijai|w4a8_eros)
    $PY $S --s3-all           # GPU: every condition, each in its own subprocess
    $PY $S --post             # CPU: contact sheets, summary.md, hf-round16/

``--dry-run`` / ``--s4`` / ``--post`` hide the GPU (CUDA_VISIBLE_DEVICES=-1) before torch is imported.
Exit codes: 0 = passed / done, 1 = failed (recorded in the JSON), 2 = bad arguments / environment.
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

# Before torch is imported (as the product worker does). CPU-only modes hide the GPU.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
CPU_ONLY = any(flag in sys.argv for flag in ("--dry-run", "--s4", "--post"))
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
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import h3_format  # noqa: E402
import sft_quant_format  # noqa: E402
from h3_format import H3DiTConfig, H3FormatError, QuantSpec  # noqa: E402
from engine.sft_quant.sft_reader import TORCH_DTYPES, read_tensors  # noqa: E402
from engine_h3 import dit as D  # noqa: E402
from engine_h3 import lowvram as LV  # noqa: E402
from engine_h3 import pipeline_h3 as P  # noqa: E402
from engine_h3 import quant_linear as QL  # noqa: E402
from engine_h3.constants import VRAM16_HEAD_GROUP, VRAM16_ROW_CHUNK  # noqa: E402
from engine_h3.quant_linear import H3Int8Linear  # noqa: E402

import comfy_kitchen as ck  # noqa: E402
import comfy_kitchen.backends.cuda as CB  # noqa: E402
from comfy_kitchen.registry import registry as CK_REGISTRY  # noqa: E402
from comfy_kitchen.scaled_mm_v2 import has_scaled_mm_v2, scaled_mm_v2  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixed conditions (plan §4.3, DESIGN §1 / §10)
# --------------------------------------------------------------------------- #

OUT_ROOT = HERE
WEIGHTS = "models/MiniMaxH3/Weights/"
FILES = {
    "int8": WEIGHTS + "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "fp8": WEIGHTS + "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    "w4a8_kijai": WEIGHTS + "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    "w4a8_eros": WEIGHTS + "10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors",
}
#: Same as the inpaint spike (FL2VA); "transformer" is replaced per condition.
PATHS = {
    "transformer": FILES["int8"],
    "text_encoder": "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "video_vae": "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
    "tokenizer_dir": "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
}
WIDTH, HEIGHT = 1280, 768
NUM_FRAMES = 73
DEFAULT_SEED = 1234
ATTENTION = "sage"
VRAM16 = True
BLOCKS_ON_GPU = 8
PREFETCH = True
KEEP_RESIDENT = False
DETERMINISTIC = False
DEFAULT_PROMPT = (
    "A busy city street at dusk seen from the sidewalk. Pedestrians walk past the camera, their footsteps "
    "clicking on the pavement; cars and a bus drive by with engine hum and tire noise, and a car horn honks "
    "in the distance. Steady camera."
)


@dataclass(frozen=True)
class Condition:
    name: str
    file: str
    steps: int
    shift_video: float
    shift_audio: float
    fp8_force_dequant: bool = False
    dequant_cache: bool = True


CONDITIONS = {
    "int8": Condition("int8", "int8", 20, 12.0, 3.0),
    "fp8": Condition("fp8", "fp8", 20, 12.0, 3.0),
    "w4a8_kijai": Condition("w4a8_kijai", "w4a8_kijai", 20, 12.0, 3.0),
    "w4a8_eros": Condition("w4a8_eros", "w4a8_eros", 8, 10.0, 3.0),
    "fp8_dequant": Condition("fp8_dequant", "fp8", 20, 12.0, 3.0, fp8_force_dequant=True),
}
S3_FIRST = ("int8", "fp8", "w4a8_kijai", "w4a8_eros")
S3_LAST = "fp8_dequant"
S3_DIR = "s3"
HF_DIR = "hf-round16"
SHEET_FRAMES = (0, 36, 72)
SHEET_TILE_LONG = 640

FP8 = "float8_e4m3fn"
W4A8 = "asym_w4a8_int8"
INT8 = "int8_tensorwise"
W4A8_CONVROT = 256
LEAVES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
S1_BLOCKS = (0, 25, 49)
S1_M = (4096, 3977, 39764)
S1_SEED = 20261112
S1_SANITY_ROWS = 4096
S1_SANITY_MAX_REL_L2 = 0.2
S2_BLOCK = 10
S2_M_NORMAL = 39764
S2_M_CHUNK = 3977
S2_SLICE = 1024
S2_WARMUP = 3
S2_REPS = 10
S2_FLAG_RATIO = 1.5
GPU_BUSY_MIB = 3072


# --------------------------------------------------------------------------- #
# Small helpers (copied from the inpaint spike)
# --------------------------------------------------------------------------- #

class Log:
    """Write the same line to stdout and to a log file."""

    def __init__(self) -> None:
        self.file = None

    def __call__(self, msg: str) -> None:
        line = f"[h3_quant_spike] {msg}"
        print(line, flush=True)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()

    def progress(self, stage: str, index: int, total: int) -> None:
        line = f"[progress] {stage} {index}/{total}"
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
    """Python logging (sft_load, pipeline_h3 logger, comfy_kitchen) -> LOG."""

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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def tool(name: str) -> str:
    local = ROOT / "tools" / "ffmpeg" / "bin" / f"{name}.exe"
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found (<repo>\\tools\\ffmpeg\\bin or PATH)")
    return found


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


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


def read_frames(path: Path, n: int) -> np.ndarray:
    from engine_h3.source_tail import read_source_frames  # noqa: PLC0415

    return read_source_frames(str(path), n)


def failure(rec: dict, exc: BaseException) -> None:
    rec["status"] = "failed"
    rec["error"] = repr(exc)
    rec["traceback"] = traceback.format_exc()
    LOG(f"FAILED: {exc!r}\n{rec['traceback']}")


def gpu_busy_check() -> dict:
    """Refuse a GPU stage while another process (the owner's run.ps1 / uvicorn server) uses the GPU."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        raise SystemExit("nvidia-smi not found: cannot confirm that the GPU is free")
    proc = run_cmd([exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    apps = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    # WDDM lists every desktop app as a compute app, so the test is: no other python / uvicorn
    # process on the GPU, and the GPU's used memory below a desktop-only bound.
    mine = {str(os.getpid()), os.environ.get("H3_QUANT_SPIKE_PARENT_PID", "")} - {""}
    others = [a for a in apps if a.split(",", 1)[0].strip() not in mine
              and any(w in a.lower() for w in ("python", "uvicorn"))]
    mem = run_cmd([exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    used = int((mem.stdout or "0,0").split(",")[0].strip() or 0)
    rec = {"nvidia_smi_compute_apps": apps, "returncode": proc.returncode, "memory_used_mib": used,
           "python_like_others": others}
    if proc.returncode != 0 or mem.returncode != 0:
        raise SystemExit(f"nvidia-smi failed ({proc.returncode}/{mem.returncode}): {proc.stderr[-500:]}")
    if others:
        raise SystemExit(f"GPU is in use by other python / uvicorn processes, refusing to run: {others}")
    if used > GPU_BUSY_MIB:
        raise SystemExit(f"GPU memory in use {used} MiB > {GPU_BUSY_MIB} MiB (desktop only), refusing to run")
    return rec


# --------------------------------------------------------------------------- #
# Spike spec, switches and counters (DESIGN §2)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SpikeSpec:
    format: str                 # "float8_e4m3fn" | "asym_w4a8_int8"
    convrot: bool               # fp8: False; w4a8: True   (read by pipeline_h3.py:812-813 for EVERY quant entry)
    convrot_groupsize: int      # fp8: 0; w4a8: 256
    group_size: int | None = None
    full_precision: bool = False
    has_input_scale: bool = False
    has_codebook: bool = False
    marker_source: str = "tensor"   # "tensor" | "metadata" | "both"


class _Switches:
    """Read at call time by the prototype layers and the block_forward wrapper."""

    def __init__(self) -> None:
        self.fp8_force_dequant = False
        self.dequant_cache = True
        #: Leaves whose dequantized weight is kept for one block forward (DESIGN §4; risk §11
        #: "fc1 cache off" = remove "fc1"). out_proj is called once per block: never cached.
        self.dequant_cache_leaves: tuple[str, ...] = ("fc1", "fc2")

    def record(self) -> dict:
        return {"fp8_force_dequant": self.fp8_force_dequant, "dequant_cache": self.dequant_cache,
                "dequant_cache_leaves": list(self.dequant_cache_leaves)}


SPIKE = _Switches()
COUNTS: collections.Counter = collections.Counter()
PATCH_HITS: collections.Counter = collections.Counter()
BACKEND_SEEN: dict[str, str] = {}
SWAP_RECORD: dict = {}
AUDIT_LAST: dict = {}
INSPECT_LAST: dict = {}
_DEQ_CACHE: dict | None = None


def reset_counts() -> None:
    COUNTS.clear()
    PATCH_HITS.clear()
    BACKEND_SEEN.clear()


# --------------------------------------------------------------------------- #
# Inspector (DESIGN §3): a copy of h3_format.inspect_h3_dit's body that accepts fp8 / w4a8
# --------------------------------------------------------------------------- #

_AUX = {
    INT8: ("weight_scale", "comfy_quant"),
    FP8: ("weight_scale", "input_scale", "comfy_quant"),
    W4A8: ("weight_s_rel", "weight_s_channel", "weight_codebook", "comfy_quant"),
}
_CARRIER_KEYS = ("format", "group_size", "convrot", "convrot_groupsize")


def _ng(msg: str) -> H3FormatError:
    return H3FormatError(h3_format._NG + "[spike] " + msg)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise _ng(msg)


def _flatten_marker(doc, where: str) -> dict:
    """``params`` merged under the top level (the top level wins), as parse_comfy_quant does."""
    _require(isinstance(doc, dict), f"quantization marker is not a JSON object ({where})")
    params = doc.get("params", {})
    if not isinstance(params, dict):
        params = {}
    return {**params, **doc}


def _read_markers(path, header, t: dict, prefix: str) -> tuple[dict, dict]:
    """(metadata markers, tensor markers), both keyed by the prefix-stripped layer name."""
    raw = header.metadata.get("_quantization_metadata", "{}")
    try:
        doc = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise _ng(f"__metadata__._quantization_metadata is not JSON ({exc})") from exc
    layers = doc.get("layers", {}) if isinstance(doc, dict) else {}
    _require(isinstance(layers, dict), "__metadata__._quantization_metadata.layers is not an object")
    meta = {}
    for key, conf in layers.items():
        name = key[len(prefix):] if prefix and key.startswith(prefix) else key
        meta[name] = _flatten_marker(conf, f"__metadata__ layers['{key}']")
    tensor = {}
    names = sorted(n[: -len(".comfy_quant")] for n in t if n.endswith(".comfy_quant"))
    with Path(path).open("rb") as fh:          # one handle, seek + read (h3_format.py:263-285)
        for layer in names:
            key = f"{prefix}{layer}.comfy_quant"
            begin, end = header.tensors[key].data_offsets
            fh.seek(header.data_base + begin)
            data = fh.read(end - begin)
            _require(len(data) == end - begin, f"short read of '{key}'")
            try:
                parsed = json.loads(bytes(data).decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise _ng(f"'{key}' is not JSON ({exc})") from exc
            tensor[layer] = _flatten_marker(parsed, f"'{key}'")
    return meta, tensor


def _resolve_markers(meta: dict, tensor: dict, t: dict) -> dict[str, tuple[str, dict, str, bool]]:
    """layer -> (format, marker, source, full_precision). Both carriers must agree."""
    out = {}
    for layer in sorted(set(meta) | set(tensor)):
        m, tq = meta.get(layer), tensor.get(layer)
        if m is not None and tq is not None:
            diff = {k: [m.get(k), tq.get(k)] for k in _CARRIER_KEYS if m.get(k) != tq.get(k)}
            _require(not diff, f"layer '{layer}': metadata and comfy_quant markers disagree {diff}")
            source = "both"
        else:
            source = "tensor" if tq is not None else "metadata"
        conf = tq if tq is not None else m
        fmt = conf.get("format")
        _require(fmt in (INT8, FP8, W4A8), f"layer '{layer}': format={fmt!r} is not supported by the spike")
        _require(f"{layer}.weight" in t, f"layer '{layer}' has a marker but no .weight")
        full = (tq or {}).get("full_precision_matrix_mult", False)     # tensor marker only (DESIGN §3.1)
        _require(isinstance(full, bool), f"layer '{layer}': full_precision_matrix_mult={full!r} is not a bool")
        out[layer] = (fmt, conf, source, full)
    return out


def _leaves_by_layer(t: dict) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = collections.defaultdict(set)
    for n in t:
        layer, _, leaf = n.rpartition(".")
        groups[layer].add(leaf)
    return groups


def _check_fp8(layer: str, conf: dict, source: str, full: bool, t: dict, leaves: set[str]) -> SpikeSpec:
    w = t[f"{layer}.weight"]
    _require(w.dtype == "F8_E4M3" and len(w.shape) == 2, f"fp8 '{layer}.weight' must be F8_E4M3 [N,K] ({w.dtype} {list(w.shape)})")
    s = t.get(f"{layer}.weight_scale")
    _require(s is not None, f"fp8 layer '{layer}' has no weight_scale")
    _require(s.dtype == "F32" and tuple(s.shape) == (), f"fp8 '{layer}.weight_scale' must be F32 [] ({s.dtype} {list(s.shape)})")
    i = t.get(f"{layer}.input_scale")
    if i is not None:
        _require(i.dtype == "F32" and tuple(i.shape) == (), f"fp8 '{layer}.input_scale' must be F32 [] ({i.dtype} {list(i.shape)})")
    _require(conf.get("convrot", False) is False, f"fp8 layer '{layer}' carries convrot={conf.get('convrot')!r} (not supported)")
    _require(full or i is not None, f"fp8 layer '{layer}' would run scaled (no full_precision_matrix_mult) but has no input_scale")
    _require("bias" not in leaves, f"fp8 layer '{layer}' has a bias (the spike layers have none)")
    return SpikeSpec(format=FP8, convrot=False, convrot_groupsize=0, group_size=None, full_precision=full,
                     has_input_scale=i is not None, has_codebook=False, marker_source=source)


def _check_w4a8(layer: str, conf: dict, source: str, full: bool, t: dict, leaves: set[str]) -> SpikeSpec:
    w = t[f"{layer}.weight"]
    _require(w.dtype == "I8" and len(w.shape) == 2, f"w4a8 '{layer}.weight' must be I8 [N,K/2] ({w.dtype} {list(w.shape)})")
    n, kh = w.shape
    k = 2 * kh
    g = conf.get("group_size", 16)                     # ComfyUI ops default (sft_quant_format.W4A8_GROUP_SIZE)
    _require(isinstance(g, int) and not isinstance(g, bool), f"w4a8 '{layer}': group_size={g!r} is not an int")
    _require(conf.get("convrot", True) is True, f"w4a8 '{layer}': convrot={conf.get('convrot')!r} (must be true)")
    cg = conf.get("convrot_groupsize", W4A8_CONVROT)
    _require(cg == W4A8_CONVROT, f"w4a8 '{layer}': convrot_groupsize={cg!r} (must be {W4A8_CONVROT})")
    # mirrors AsymW4A8Int8Layout.Params (tensor/w4a8_int8.py:160-166)
    _require(g >= 4 and k % 16 == 0 and k % g == 0 and k % W4A8_CONVROT == 0 and (16 % g == 0 or g % 16 == 0),
             f"w4a8 '{layer}': K={k}, group_size={g} violate AsymW4A8Int8Layout.Params")
    sr = t.get(f"{layer}.weight_s_rel")
    _require(sr is not None and sr.dtype == "F8_E4M3" and tuple(sr.shape) == (n, k // g),
             f"w4a8 '{layer}.weight_s_rel' must be F8_E4M3 [{n},{k // g}] ({None if sr is None else (sr.dtype, list(sr.shape))})")
    sc = t.get(f"{layer}.weight_s_channel")
    _require(sc is not None and sc.dtype == "F32" and tuple(sc.shape) == (n,),
             f"w4a8 '{layer}.weight_s_channel' must be F32 [{n}] ({None if sc is None else (sc.dtype, list(sc.shape))})")
    cb = t.get(f"{layer}.weight_codebook")
    if cb is not None:
        _require(cb.dtype == "F32" and tuple(cb.shape) == (16,), f"w4a8 '{layer}.weight_codebook' must be F32 [16] ({cb.dtype} {list(cb.shape)})")
    corr = sorted(leaf for leaf in leaves if leaf.endswith("correction"))
    _require(not corr, f"w4a8 '{layer}' has {corr} (asymmetric correction is refused by the spike)")
    _require(not full, f"w4a8 '{layer}' carries full_precision_matrix_mult (not supported by the spike)")
    _require("bias" not in leaves, f"w4a8 layer '{layer}' has a bias (the spike layers have none)")
    return SpikeSpec(format=W4A8, convrot=True, convrot_groupsize=W4A8_CONVROT, group_size=g, full_precision=False,
                     has_input_scale=False, has_codebook=cb is not None, marker_source=source)


def _derive_config(t: dict, num_layers: int, refiner_layers: int) -> dict:
    """h3_format.inspect_h3_dit :323-379 on the logical tensor table (w4a8 weights shown as [N, K])."""
    _shape, _expect = h3_format._shape, h3_format._expect
    vp = _shape(t, "video_patch_proj.weight")
    ap = _shape(t, "audio_patch_proj.weight")
    _expect(len(vp) == 2 and len(ap) == 2, "patch_proj の重みは 2 次元であるべきです")
    hidden = vp[0]
    head_dim = _shape(t, "blocks.0.attn.q_norm.weight")[0]
    qkv = _shape(t, "blocks.0.attn.qkv_proj.weight")
    _expect(head_dim > 0 and qkv[0] % (3 * head_dim) == 0, f"qkv_proj の出力 {qkv[0]} が 3×head_dim（{head_dim}）で割り切れません")
    num_heads = qkv[0] // (3 * head_dim)
    fc1 = _shape(t, "blocks.0.mlp.fc1.weight")
    _expect(fc1[0] % 2 == 0, f"fc1 の出力 {fc1[0]} が偶数ではありません")
    ffn = fc1[0] // 2
    cond = _shape(t, "condition_proj.weight")
    _expect(len(cond) == 2, "condition_proj.weight は 2 次元であるべきです")
    vout = _shape(t, "final_layer.video_out.weight")
    aout = _shape(t, "final_layer.audio_out.weight")
    _expect(len(vout) == 2 and len(aout) == 2, "final_layer の出力 2 本の重みは 2 次元であるべきです")
    _expect(vout[0] % 4 == 0, f"video_out の出力 {vout[0]} が 4（patch 1×2×2）で割り切れません")
    latents_dim, audio_dim = vout[0] // 4, aout[0]
    table = _shape(t, "adaln_t_table")
    _expect(len(table) == 2 and table[0] >= 2 and table[1] >= 1, f"adaln_t_table の形 {list(table)} が不正です")
    grid, rank = table
    inv = _shape(t, "rope.inv_freq")
    _expect(len(inv) == 1 and inv[0] >= 1, f"rope.inv_freq の形 {list(inv)} が不正です")
    inner = num_heads * head_dim
    _expect(vp == (hidden, latents_dim * 4), f"video_patch_proj.weight {list(vp)} ≠ [{hidden}, {latents_dim * 4}]")
    _expect(ap == (hidden, audio_dim), f"audio_patch_proj.weight {list(ap)} ≠ [{hidden}, {audio_dim}]")
    _expect(cond[0] == hidden, f"condition_proj.weight {list(cond)} の出力が {hidden} ではありません")
    _expect(vout[1] == hidden and aout[1] == hidden, "final_layer の出力 2 本の入力幅が hidden_size と一致しません")
    fl = _shape(t, "final_layer.adaln_proj.linear.weight")
    _expect(fl == (2 * hidden, rank), f"final_layer.adaln_proj.linear.weight {list(fl)} ≠ [{2 * hidden}, {rank}]")
    expect_block = {
        "attn.qkv_proj.weight": (3 * inner, hidden),
        "attn.out_proj.weight": (hidden, inner),
        "attn.q_norm.weight": (head_dim,),
        "attn.k_norm.weight": (head_dim,),
        "mlp.fc1.weight": (2 * ffn, hidden),
        "mlp.fc2.weight": (hidden, ffn),
        "norm1.weight": (hidden,),
        "norm2.weight": (hidden,),
        "adaln_proj.linear.weight": (18 * hidden, rank),
    }
    for i in range(num_layers):
        for leaf, shp in expect_block.items():
            got = _shape(t, f"blocks.{i}.{leaf}")
            _expect(got == shp, f"blocks.{i}.{leaf} {list(got)} ≠ {list(shp)}")
    for i in range(refiner_layers):
        for leaf in h3_format._REFINER_REQUIRED:
            got = _shape(t, f"token_refiner.blocks.{i}.{leaf}")
            _expect(got == expect_block[leaf], f"token_refiner.blocks.{i}.{leaf} {list(got)} ≠ {list(expect_block[leaf])}")
    return {"hidden": hidden, "head_dim": head_dim, "num_heads": num_heads, "ffn": ffn, "text_dim": cond[1],
            "latents_dim": latents_dim, "audio_dim": audio_dim, "grid": grid, "rank": rank, "inv": inv[0]}


def _summarize(prefix: str, quant: dict, unknown: list, outside: list) -> dict:
    by_format = collections.Counter("int8_tensorwise" if isinstance(s, QuantSpec) else s.format for s in quant.values())
    w4 = [s for s in quant.values() if isinstance(s, SpikeSpec) and s.format == W4A8]
    fp = [s for s in quant.values() if isinstance(s, SpikeSpec) and s.format == FP8]
    leaf_by_format = collections.Counter(
        f"{'int8_tensorwise' if isinstance(s, QuantSpec) else s.format}:{n.rsplit('.', 1)[1]}" for n, s in quant.items())
    int8_blocks = sorted({int(n.split(".")[1]) for n, s in quant.items() if isinstance(s, QuantSpec) and n.startswith("blocks.")})
    return {
        "prefix": prefix,
        "quantized_layers": len(quant),
        "per_format": dict(sorted(by_format.items())),
        "per_format_leaf": dict(sorted(leaf_by_format.items())),
        "w4a8_group_size_histogram": dict(sorted(collections.Counter(s.group_size for s in w4).items())),
        "w4a8_with_codebook": sum(s.has_codebook for s in w4),
        "fp8_full_precision": sum(s.full_precision for s in fp),
        "fp8_with_input_scale": sum(s.has_input_scale for s in fp),
        "marker_source": dict(sorted(collections.Counter(s.marker_source for s in quant.values()
                                                         if isinstance(s, SpikeSpec)).items())),
        "int8_blocks": int8_blocks if by_format.get(INT8, 0) and len(int8_blocks) < 50 else ("all" if by_format.get(INT8) else []),
        "refiner_quantized": sorted(n for n in quant if n.startswith("token_refiner.")),
        "unknown": list(unknown) + list(outside),
    }


def inspect_core(path) -> tuple:
    """(header, H3DiTConfig, summary). Body copied from h3_format.inspect_h3_dit (:295-402)."""
    header = h3_format._read_header(path)
    names = set(header.tensors)
    if not h3_format.is_h3_dit_header(header):
        raise _ng("video_patch_proj.weight and audio_patch_proj.weight are not both present")
    prefix = h3_format.detect_h3_prefix(names)
    t = {n[len(prefix):]: info for n, info in header.tensors.items() if n.startswith(prefix)}
    outside = sorted(n for n in names if not n.startswith(prefix))
    if "adaln_t_table" not in t:
        raise _ng("adaln_t_table is missing (not the pruned DiT)")
    missing = [r for r in h3_format._DIT_REQUIRED if r not in t]
    _require(not missing, f"required tensor {missing[:1]} is missing")
    stripped = set(t)
    num_layers = h3_format._count_contiguous(stripped, "blocks.", h3_format._BLOCK_REQUIRED, "本体のブロック")
    refiner_layers = h3_format._count_contiguous(stripped, "token_refiner.blocks.", h3_format._REFINER_REQUIRED,
                                                 "token_refiner のブロック")

    # 1-2: markers from both carriers, classified by marker format (NOT by weight dtype)
    meta, tensor = _read_markers(path, header, t, prefix)
    resolved = _resolve_markers(meta, tensor, t)
    leaves = _leaves_by_layer(t)
    spike: dict[str, SpikeSpec] = {}
    int8_marked: set[str] = set()
    for layer, (fmt, conf, source, full) in resolved.items():
        if fmt == FP8:
            spike[layer] = _check_fp8(layer, conf, source, full, t, leaves[layer])
        elif fmt == W4A8:
            spike[layer] = _check_w4a8(layer, conf, source, full, t, leaves[layer])
        else:
            int8_marked.add(layer)
    stray_f8 = sorted(n for n, info in t.items() if n.endswith(".weight") and info.dtype.startswith("F8_")
                      and n[: -len(".weight")] not in spike)
    _require(not stray_f8, f"'{stray_f8[:1]}' is an F8 weight without an fp8 marker")

    # 3: shape table on the logical view (w4a8 [N, K/2] -> [N, K])
    logical = dict(t)
    for layer, spec in spike.items():
        if spec.format == W4A8:
            info = t[f"{layer}.weight"]
            logical[f"{layer}.weight"] = dataclasses.replace(info, shape=(info.shape[0], 2 * info.shape[1]))
    c = _derive_config(logical, num_layers, refiner_layers)

    # 4: int8 layers through the ORIGINAL _quant_layers, on the table minus every fp8 / w4a8 tensor
    t_int8 = {n: info for n, info in t.items() if n.rpartition(".")[0] not in spike}
    int8 = h3_format._quant_layers(path, header, t_int8, lambda n: prefix + n)
    _require(set(int8) == int8_marked,
             f"int8 layers by marker {len(int8_marked)} != int8 layers by _quant_layers {len(int8)}"
             f" (only marker: {sorted(int8_marked - set(int8))[:3]}, only I8: {sorted(set(int8) - int8_marked)[:3]})")
    quant: dict = {**int8, **spike}

    # 5: unknown names, with the per-format auxiliary names known
    aux = set()
    for layer, spec in quant.items():
        fmt = INT8 if isinstance(spec, QuantSpec) else spec.format
        aux.update(f"{layer}.{leaf}" for leaf in _AUX[fmt])
    unknown = sorted(n for n in t if not (h3_format._DIT_KNOWN.match(n) or n in aux))
    cfg = H3DiTConfig(
        prefix=prefix, hidden_size=c["hidden"], num_layers=num_layers, token_refiner_layers=refiner_layers,
        num_heads=c["num_heads"], head_dim=c["head_dim"], ffn_hidden=c["ffn"], text_dim=c["text_dim"],
        latents_dim=c["latents_dim"], audio_latents_dim=c["audio_dim"], pruned=True, adaln_grid=c["grid"],
        adaln_rank=c["rank"], rope_inv_freq_len=c["inv"], quant=quant, unknown=tuple(unknown + outside),
    )
    summary = _summarize(prefix, quant, unknown, outside)
    summary["metadata_marker_layers"] = len(meta)
    summary["tensor_marker_layers"] = len(tensor)
    return header, cfg, summary


def inspect_h3_dit_spike(path: str):
    """Drop-in for h3_format.inspect_h3_dit (same signature and return)."""
    PATCH_HITS["inspect_h3_dit"] += 1
    header, cfg, summary = inspect_core(path)
    INSPECT_LAST.clear()
    INSPECT_LAST.update(summary)
    return header, cfg


# --------------------------------------------------------------------------- #
# Prototype layers (DESIGN §4)
# --------------------------------------------------------------------------- #

def _as_2d(x: torch.Tensor) -> tuple[tuple, torch.Tensor]:
    lead = tuple(x.shape[:-1])
    x2d = x.reshape(-1, x.shape[-1])
    if x2d.dtype != torch.bfloat16:
        x2d = x2d.to(torch.bfloat16)
    return lead, x2d.contiguous()


def _record_impl(name: str, kwargs: dict) -> None:
    """The backend the registry resolves for ``name`` with the real arguments (first call only)."""
    if name in BACKEND_SEEN:
        return
    try:
        BACKEND_SEEN[name] = CK_REGISTRY.get_implementation(name, kwargs=kwargs).__module__
    except Exception as exc:  # noqa: BLE001 -- recorded; the real call below fails loud by itself
        BACKEND_SEEN[name] = f"ERROR {exc!r}"


class _SpikeLinearBase(H3Int8Linear):
    """Common part. Subclasses H3Int8Linear so normalize_dtypes (dit.py:499) and lora.attach skip / accept it;
    ``__init__`` never calls the parent's (no int8 weight is created)."""

    def _setup(self, in_features: int, out_features: int, spec: SpikeSpec, name: str) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.spec = spec
        self.register_parameter("bias", None)
        self._spike_name = name
        self._leaf = name.rpartition(".")[2] or name
        self._region = "refiner:" if name.startswith("token_refiner.") else ""

    def _count(self, key: str, n: int = 1) -> None:
        COUNTS[f"{self._region}{key}"] += n
        COUNTS[f"{self._region}{key}@{self._leaf}"] += n

    def _no_lora(self, lora_xa=None) -> None:
        if lora_xa is not None or getattr(self, "_lora_specs", None):
            raise NotImplementedError(f"{type(self).__name__} ({self._spike_name}): LoRA is not supported in the spike")


class H3Fp8Linear(_SpikeLinearBase):
    """fp8 (E4M3) per-tensor Linear: ComfyUI's TensorCoreFP8Layout path with plain tensors."""

    def __init__(self, in_features: int, out_features: int, *, spec: SpikeSpec, name: str = "") -> None:
        nn.Module.__init__(self)
        self._setup(in_features, out_features, spec, name)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, dtype=torch.float8_e4m3fn),
                                   requires_grad=False)
        self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32), persistent=True)
        if spec.has_input_scale:
            self.register_buffer("input_scale", torch.empty((), dtype=torch.float32), persistent=True)
        else:
            self.register_buffer("input_scale", None)

    def _mode(self) -> str:
        if self.spec.full_precision or SPIKE.fp8_force_dequant:
            return "dequant"
        if self.weight.is_cuda and torch.cuda.get_device_capability(self.weight.device) < (8, 9):
            return "dequant"
        if self.input_scale is None:
            raise RuntimeError(f"H3Fp8Linear {self._spike_name}: scaled path without input_scale")
        return "scaled"

    def _quantize(self, x2d: torch.Tensor) -> torch.Tensor:
        _record_impl("quantize_per_tensor_fp8", {"x": x2d, "scale": self.input_scale, "output_type": torch.float8_e4m3fn})
        return ck.quantize_per_tensor_fp8(x2d, self.input_scale)

    def _mm(self, xq: torch.Tensor, w_rows: torch.Tensor, scale_a: torch.Tensor) -> torch.Tensor:
        """tensor/fp8.py _fp8_scaled_mm: scaled_mm_v2(input.contiguous(), weight.t(), ...)."""
        self._count("fp8_scaled_calls")
        return scaled_mm_v2(xq.contiguous(), w_rows.t(), scale_a=scale_a, scale_b=self.weight_scale,
                            bias=None, out_dtype=torch.bfloat16)

    def _dequant(self, w: torch.Tensor) -> torch.Tensor:
        _record_impl("dequantize_per_tensor_fp8", {"x": w, "scale": self.weight_scale, "output_type": torch.bfloat16})
        return ck.dequantize_per_tensor_fp8(w, self.weight_scale, torch.bfloat16)

    def _w_bf16(self) -> torch.Tensor:
        cache = _DEQ_CACHE
        if cache is None or self._leaf not in SPIKE.dequant_cache_leaves:
            self._count("fp8_deq_uncached")
            return self._dequant(self.weight)
        hit = cache.get(id(self))
        if hit is not None:
            ptr, w = hit
            if ptr != self.weight.data_ptr():
                raise RuntimeError(f"H3Fp8Linear {self._spike_name}: weight moved inside one block forward "
                                   f"({ptr:#x} -> {self.weight.data_ptr():#x})")
            self._count("fp8_deq_hit")
            return w
        w = self._dequant(self.weight)
        cache[id(self)] = (self.weight.data_ptr(), w)
        self._count("fp8_deq_miss")
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._no_lora()
        lead, x2d = _as_2d(x)
        if self._mode() == "scaled":
            out = self._mm(self._quantize(x2d), self.weight, self.input_scale)
        else:
            out = F.linear(x2d, self._w_bf16())
        return out.reshape(*lead, self.out_features)

    def forward_rows(self, x: torch.Tensor, n0: int, n1: int, lora_xa=None) -> torch.Tensor:
        self._no_lora(lora_xa)
        if n0 % 16 or (n1 - n0) % 16:
            raise ValueError(f"H3Fp8Linear {self._spike_name}: rows [{n0},{n1}) are not multiples of 16")
        lead, x2d = _as_2d(x)
        self._count("fp8_rows")
        if self._mode() == "scaled":
            out = self._mm(self._quantize(x2d), self.weight[n0:n1], self.input_scale)
        else:
            self._count("fp8_deq_uncached")
            out = F.linear(x2d, self._dequant(self.weight[n0:n1]))
        return out.reshape(*lead, n1 - n0)

    def shares_input(self, x: torch.Tensor) -> bool:
        return (self._mode() == "scaled" and x.is_cuda and x.device == self.weight.device
                and self.input_scale is not None and self.input_scale.device == x.device)

    def quantize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _lead, x2d = _as_2d(x)
        self._count("fp8_quantize_input")
        return self._quantize(x2d), self.input_scale

    def forward_rows_q(self, xq: torch.Tensor, xs: torch.Tensor, n0: int, n1: int, lora_xa=None) -> torch.Tensor:
        self._no_lora(lora_xa)
        if n0 % 16 or (n1 - n0) % 16:
            raise ValueError(f"H3Fp8Linear {self._spike_name}: rows [{n0},{n1}) are not multiples of 16")
        self._count("fp8_rows_q")
        return self._mm(xq, self.weight[n0:n1], xs)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, format={self.spec.format}, "
                f"full_precision={self.spec.full_precision}, input_scale={self.input_scale is not None}")


class H3W4A8Linear(_SpikeLinearBase):
    """Grouped W4A8: comfy_kitchen.w4a8_int8_linear on plain tensors (AsymW4A8Int8Layout's path)."""

    def __init__(self, in_features: int, out_features: int, *, spec: SpikeSpec, name: str = "") -> None:
        nn.Module.__init__(self)
        self._setup(in_features, out_features, spec, name)
        n, k, g = self.out_features, self.in_features, int(spec.group_size)
        self.weight = nn.Parameter(torch.empty(n, k // 2, dtype=torch.int8), requires_grad=False)
        self.register_buffer("weight_s_rel", torch.empty(n, k // g, dtype=torch.float8_e4m3fn), persistent=True)
        self.register_buffer("weight_s_channel", torch.empty(n, dtype=torch.float32), persistent=True)
        if spec.has_codebook:
            self.register_buffer("weight_codebook", torch.empty(16, dtype=torch.float32), persistent=True)
        else:
            self.register_buffer("weight_codebook", None)

    def _linear(self, x2d, qdata, s_rel, s_channel) -> torch.Tensor:
        kwargs = {"x": x2d, "qdata": qdata, "s_rel": s_rel, "s_channel": s_channel, "codebook": self.weight_codebook,
                  "correction": None, "bias": None, "group_size": int(self.spec.group_size),
                  "convrot_groupsize": int(self.spec.convrot_groupsize), "out_dtype": torch.bfloat16}
        _record_impl("w4a8_int8_linear", kwargs)
        return ck.w4a8_int8_linear(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._no_lora()
        lead, x2d = _as_2d(x)
        self._count("w4a8_calls")
        out = self._linear(x2d, self.weight, self.weight_s_rel, self.weight_s_channel)
        return out.reshape(*lead, self.out_features)

    def forward_rows(self, x: torch.Tensor, n0: int, n1: int, lora_xa=None) -> torch.Tensor:
        self._no_lora(lora_xa)
        if n0 % 16:
            raise ValueError(f"H3W4A8Linear {self._spike_name}: n0={n0} is not a multiple of 16")
        lead, x2d = _as_2d(x)
        self._count("w4a8_rows")
        out = self._linear(x2d, self.weight[n0:n1], self.weight_s_rel[n0:n1], self.weight_s_channel[n0:n1])
        return out.reshape(*lead, n1 - n0)

    def shares_input(self, x: torch.Tensor) -> bool:
        return False

    def quantize_input(self, x):
        raise RuntimeError("H3W4A8Linear.quantize_input must not be called (shares_input is False)")

    def forward_rows_q(self, xq, xs, n0, n1, lora_xa=None):
        raise RuntimeError("H3W4A8Linear.forward_rows_q must not be called (shares_input is False)")

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, format={self.spec.format}, "
                f"group_size={self.spec.group_size}, codebook={self.weight_codebook is not None}")


SPIKE_CLASSES = {FP8: H3Fp8Linear, W4A8: H3W4A8Linear}


# --------------------------------------------------------------------------- #
# Swap (DESIGN §5)
# --------------------------------------------------------------------------- #

_ORIG_INSPECT = h3_format.inspect_h3_dit
_ORIG_SWAP = QL.swap_in_int8_linears
_ORIG_BLOCK_FORWARD = LV.block_forward
_ORIG_NORMALIZE = D.H3DiT.__dict__["normalize_dtypes"]
_ORIG_EAGER_W4A8 = CB.eager_w4a8_int8_linear
_ORIG_TWOPASS = CB._dequant_int4_grouped_to_int8


def swap_spike(root: nn.Module, quant: dict, *, bias_dtype_of) -> int:
    """int8 entries -> the ORIGINAL swap_in_int8_linears; fp8 / w4a8 -> the prototype layers (same checks)."""
    PATCH_HITS["swap_in_int8_linears"] += 1
    int8 = {n: s for n, s in quant.items() if isinstance(s, QuantSpec)}
    spike = {n: s for n, s in quant.items() if isinstance(s, SpikeSpec)}
    other = sorted(set(quant) - set(int8) - set(spike))
    if other:
        raise RuntimeError(f"swap_spike: entries of unknown spec type: {other[:3]}")
    count = _ORIG_SWAP(root, int8, bias_dtype_of=bias_dtype_of)
    per = collections.Counter()
    for name in sorted(spike):
        spec = spike[name]
        parent_name, _, leaf = name.rpartition(".")
        try:
            parent = root.get_submodule(parent_name) if parent_name else root
        except AttributeError as exc:
            raise RuntimeError(f"spike layer '{name}': parent module missing: {exc}") from exc
        old = getattr(parent, leaf, None)
        if old is None or not isinstance(old, nn.Module):
            raise RuntimeError(f"spike layer '{name}' is not in the model")
        if not isinstance(old, nn.Linear):
            raise RuntimeError(f"spike layer '{name}' is not an nn.Linear ({type(old).__name__})")
        if old.bias is not None:
            raise RuntimeError(f"spike layer '{name}' replaces a Linear with a bias (not supported)")
        with torch.device("meta"):
            new = SPIKE_CLASSES[spec.format](old.in_features, old.out_features, spec=spec, name=name)
        setattr(parent, leaf, new)
        per[type(new).__name__] += 1
        count += 1
    SWAP_RECORD.clear()
    SWAP_RECORD.update({"H3Int8Linear": len(int8), **per, "total": count})
    return count


# --------------------------------------------------------------------------- #
# Patches (DESIGN §6)
# --------------------------------------------------------------------------- #

def _block_forward_cached(block, *args, **kwargs):
    """lowvram.block_forward with the per-block dequant cache (lifetime = one block forward)."""
    global _DEQ_CACHE
    PATCH_HITS["block_forward"] += 1
    prev, _DEQ_CACHE = _DEQ_CACHE, ({} if SPIKE.dequant_cache else None)
    try:
        return _ORIG_BLOCK_FORWARD(block, *args, **kwargs)
    finally:
        _DEQ_CACHE = prev


def _eager_w4a8_counting(*args, **kwargs):
    COUNTS["w4a8_eager"] += 1
    return _ORIG_EAGER_W4A8(*args, **kwargs)


def _twopass_counting(*args, **kwargs):
    COUNTS["w4a8_twopass"] += 1
    return _ORIG_TWOPASS(*args, **kwargs)


_EXPECTED_DTYPES = {
    "H3Int8Linear": {"weight": torch.int8, "weight_scale": torch.float32},
    "H3Fp8Linear": {"weight": torch.float8_e4m3fn, "weight_scale": torch.float32, "input_scale": torch.float32},
    "H3W4A8Linear": {"weight": torch.int8, "weight_s_rel": torch.float8_e4m3fn, "weight_s_channel": torch.float32,
                     "weight_codebook": torch.float32},
}


def _quant_dtypes(dit: nn.Module) -> dict[str, dict[str, str]]:
    out = {}
    for name, mod in dit.named_modules():
        if isinstance(mod, H3Int8Linear):
            kind = type(mod).__name__
            tensors = {"weight": mod.weight, **{k: v for k, v in mod._buffers.items()}}
            out[name] = {"class": kind, **{k: str(v.dtype).replace("torch.", "") for k, v in tensors.items() if v is not None}}
    return out


def _audit(dit: nn.Module, before: dict, after: dict, changed: int) -> dict:
    problems = []
    for name, rec in after.items():
        exp = _EXPECTED_DTYPES.get(rec["class"], {})
        for leaf, dt in rec.items():
            if leaf == "class":
                continue
            want = exp.get(leaf)
            if want is not None and dt != str(want).replace("torch.", ""):
                problems.append(f"{name}.{leaf}: {dt} (expected {want})")
            if before.get(name, {}).get(leaf) != dt:
                problems.append(f"{name}.{leaf}: changed {before.get(name, {}).get(leaf)} -> {dt}")
    table = str(dit.adaln_t_table.dtype).replace("torch.", "")
    if table != "float32":
        problems.append(f"adaln_t_table: {table} (expected float32)")
    modules = collections.Counter(type(m).__name__ for m in dit.modules()
                                  if isinstance(m, (H3Int8Linear, nn.Linear)))
    summary = collections.Counter(f"{r['class']}.{leaf}={dt}" for r in after.values() for leaf, dt in r.items() if leaf != "class")
    return {"passed": not problems, "problems": problems[:50], "n_problems": len(problems), "changed": changed,
            "adaln_t_table": table, "modules": dict(sorted(modules.items())),
            "dtype_histogram": dict(sorted(summary.items())),
            "per_layer": {n: ",".join(f"{k}={v}" for k, v in r.items() if k != "class") for n, r in after.items()}}


def _normalize_dtypes_audited(self) -> int:
    PATCH_HITS["normalize_dtypes"] += 1
    before = _quant_dtypes(self)
    changed = _ORIG_NORMALIZE(self)
    audit = _audit(self, before, _quant_dtypes(self), changed)
    AUDIT_LAST.clear()
    AUDIT_LAST.update(audit)
    LOG(f"dtype audit: passed={audit['passed']} changed={changed} adaln_t_table={audit['adaln_t_table']} "
        f"modules={audit['modules']}")
    if not audit["passed"]:
        raise RuntimeError(f"dtype audit after normalize_dtypes failed: {audit['problems'][:5]}")
    return changed


_ACTIVE = False


@contextlib.contextmanager
def spike_patches():
    """Install every patch (module-attribute assignment) and restore all of them in finally."""
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("spike_patches is already active")
    saved = [
        (h3_format, "inspect_h3_dit", h3_format.inspect_h3_dit, inspect_h3_dit_spike),
        (QL, "swap_in_int8_linears", QL.swap_in_int8_linears, swap_spike),
        (LV, "block_forward", LV.block_forward, _block_forward_cached),
        (CB, "eager_w4a8_int8_linear", CB.eager_w4a8_int8_linear, _eager_w4a8_counting),
        (CB, "_dequant_int4_grouped_to_int8", CB._dequant_int4_grouped_to_int8, _twopass_counting),
        (D.H3DiT, "normalize_dtypes", D.H3DiT.__dict__["normalize_dtypes"], _normalize_dtypes_audited),
    ]
    applied = []
    try:
        for owner, attr, orig, new in saved:
            setattr(owner, attr, new)
            applied.append((owner, attr, orig))
        _ACTIVE = True
        yield
    finally:
        for owner, attr, orig in reversed(applied):
            setattr(owner, attr, orig)
        _ACTIVE = False


def patches_restored() -> bool:
    return (h3_format.inspect_h3_dit is _ORIG_INSPECT and QL.swap_in_int8_linears is _ORIG_SWAP
            and LV.block_forward is _ORIG_BLOCK_FORWARD and CB.eager_w4a8_int8_linear is _ORIG_EAGER_W4A8
            and CB._dequant_int4_grouped_to_int8 is _ORIG_TWOPASS
            and D.H3DiT.__dict__["normalize_dtypes"] is _ORIG_NORMALIZE)


# --------------------------------------------------------------------------- #
# Dry run (DESIGN §7) and S4
# --------------------------------------------------------------------------- #

def _bias_dtype_of(header, cfg):
    def bias_dtype_of(name: str) -> torch.dtype:          # pipeline_h3.py:1549-1551
        info = header.tensors.get(f"{cfg.prefix}{name}.bias")
        return TORCH_DTYPES.get(info.dtype, torch.bfloat16) if info is not None else torch.bfloat16
    return bias_dtype_of


def original_verdict(path) -> dict:
    try:
        _h, cfg = _ORIG_INSPECT(str(path))
        return {"verdict": "accepted", "int8_layers": len(cfg.quant), "unknown": list(cfg.unknown)}
    except H3FormatError as exc:
        return {"verdict": "refused", "message": str(exc)}


def _owner_is_quant(key: str, quant: dict) -> bool:
    return key.rpartition(".")[0] in quant


def key_map_comparison(header, cfg, dit) -> tuple[dict, dict]:
    """Header keys mapped by dit_key_map_for vs dit.state_dict(keep_vars=True): (record, mapped infos)."""
    km = P.dit_key_map_for(cfg)
    sd = dit.state_dict(keep_vars=True)
    mapped, markers, dropped = {}, [], []
    for key, info in header.tensors.items():
        if key.endswith(".comfy_quant"):
            markers.append(key)
            continue
        target = km(key)
        if target is None:
            dropped.append(key)
            continue
        mapped[target] = info
    both = sorted(set(mapped) & set(sd))
    shape_mismatch = [f"{k}: file {list(mapped[k].shape)} model {list(sd[k].shape)}"
                      for k in both if tuple(mapped[k].shape) != tuple(sd[k].shape)]
    quant_dtype_mismatch = [f"{k}: file {mapped[k].dtype} model {sd[k].dtype}" for k in both
                            if _owner_is_quant(k, cfg.quant) and TORCH_DTYPES.get(mapped[k].dtype) != sd[k].dtype]
    float_dtype_diff = collections.Counter(f"{mapped[k].dtype}->{str(sd[k].dtype).replace('torch.', '')}"
                                           for k in both if not _owner_is_quant(k, cfg.quant)
                                           and TORCH_DTYPES.get(mapped[k].dtype) != sd[k].dtype)
    expected_dropped = sorted((cfg.prefix + u) if (cfg.prefix + u) in header.tensors else u for u in cfg.unknown)
    rec = {
        "header_tensors": len(header.tensors), "state_dict_keys": len(sd), "mapped": len(mapped),
        "skipped_comfy_quant": len(markers), "dropped_by_key_map": sorted(dropped),
        "dropped_equals_unknown": sorted(dropped) == expected_dropped,
        "missing_in_model (file key, no model key)": sorted(set(mapped) - set(sd)),
        "missing_in_file (model key, no file key)": sorted(set(sd) - set(mapped)),
        "shape_mismatch": shape_mismatch, "quant_dtype_mismatch": quant_dtype_mismatch,
        "float_dtype_diff_info (assign keeps the file dtype; normalize_dtypes fixes)": dict(float_dtype_diff),
    }
    return rec, mapped


def dry_one(key: str, path: Path) -> dict:
    rec: dict = {"file": rel(path), "status": "running"}
    try:
        LOG(f"dry-run [{key}]: original inspector")
        rec["original_inspector"] = original_verdict(path)
        LOG(f"dry-run [{key}]: {jdump(rec['original_inspector'])[:600]}")
        header, cfg, summary = inspect_core(str(path))
        rec["spike_inspector"] = summary
        LOG(f"dry-run [{key}]: spike inspector {jdump(summary)}")
        reset_counts()
        with spike_patches():
            reached = {"inspect_routes_to_spike": h3_format.inspect_h3_dit is inspect_h3_dit_spike}
            from engine_h3.quant_linear import swap_in_int8_linears as late_swap  # noqa: PLC0415 -- as _build_dit
            reached["late_import_swap_is_spike"] = late_swap is swap_spike
            reached["lowvram_block_forward_is_wrapper"] = LV.block_forward is _block_forward_cached
            reached["cuda_eager_w4a8_is_counter"] = CB.eager_w4a8_int8_linear is _eager_w4a8_counting
            reached["cuda_twopass_is_counter"] = CB._dequant_int4_grouped_to_int8 is _twopass_counting
            header2, cfg2 = h3_format.inspect_h3_dit(str(path))
            reached["patched_inspect_same_quant"] = cfg2.quant == cfg.quant and cfg2.unknown == cfg.unknown
            with torch.device("meta"):
                dit = D.H3DiT(cfg2)
                swapped = late_swap(dit, cfg2.quant, bias_dtype_of=_bias_dtype_of(header2, cfg2))
            rec["swap"] = {"returned": swapped, **SWAP_RECORD}
            cmp, mapped = key_map_comparison(header2, cfg2, dit)
            rec["key_map_comparison"] = cmp
            # (d): assign file-dtype meta tensors (what load_streaming does), then normalize_dtypes on meta
            state = {k: torch.empty(tuple(info.shape), dtype=TORCH_DTYPES[info.dtype], device="meta")
                     for k, info in mapped.items()}
            res = dit.load_state_dict(state, strict=False, assign=True)
            rec["meta_assign"] = {"missing_keys": list(res.missing_keys), "unexpected_keys": list(res.unexpected_keys)}
            try:
                dit.normalize_dtypes()
                audit_error = None
            except RuntimeError as exc:
                audit_error = str(exc)
            rec["dtype_audit"] = {**AUDIT_LAST, "error": audit_error}
            rec["patch_hits"] = dict(PATCH_HITS)
            rec["reached"] = reached
        rec["reached"]["restored"] = patches_restored()
        if key != "int8":
            rec["s4"] = s4_one(path, header, cfg.prefix)
        ok = (all(rec["reached"].values()) and not cmp["missing_in_model (file key, no model key)"]
              and not cmp["missing_in_file (model key, no file key)"] and not cmp["shape_mismatch"]
              and not cmp["quant_dtype_mismatch"] and cmp["dropped_equals_unknown"]
              and not rec["meta_assign"]["missing_keys"] and not rec["meta_assign"]["unexpected_keys"]
              and rec["dtype_audit"].get("passed") is True)
        rec["passed"] = bool(ok)
        rec["status"] = "done"
    except BaseException as exc:  # noqa: BLE001
        failure(rec, exc)
        rec["passed"] = False
    return rec


def _layer_groups(header, prefix: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for key in header.tensors:
        if key.startswith(prefix):
            groups[key.rpartition(".")[0]].append(key)
    return groups


def s4_one(path: Path, header, prefix: str) -> dict:
    """sft_quant_format.layer_schemes on the whole file, then on one layer at a time (filtered header)."""
    rec: dict = {}
    try:
        schemes = sft_quant_format.layer_schemes(str(path), header, prefix)
        rec["whole_file"] = {"verdict": "accepted", "per_scheme": dict(collections.Counter(schemes.values()))}
    except sft_quant_format.QuantFormatError as exc:
        rec["whole_file"] = {"verdict": "refused", "message": str(exc)}
    accepted = collections.Counter()
    refused = collections.Counter()
    examples: dict[str, str] = {}
    for layer, keys in sorted(_layer_groups(header, prefix).items()):
        wkey = f"{layer}.weight"
        if wkey not in header.tensors or header.tensors[wkey].dtype not in ("I8", "F8_E4M3", "F8_E5M2"):
            continue
        sub = dataclasses.replace(header, tensors={k: header.tensors[k] for k in keys})
        try:
            got = sft_quant_format.layer_schemes(str(path), sub, prefix)
            accepted[got.get(layer, "none")] += 1
        except sft_quant_format.QuantFormatError as exc:
            msg = str(exc).replace(layer, "<layer>")
            refused[msg] += 1
            examples.setdefault(msg, layer)
    rec["per_layer"] = {"accepted_per_scheme": dict(accepted), "refused_per_reason": dict(refused),
                        "refused_example_layer": examples}
    return rec


def dry_run(args) -> int:
    LOG.open(OUT_ROOT / "dry_run.log", mode="w")
    install_log_bridge()
    rec: dict = {"script": rel(Path(__file__)), "started_at": now(), "torch": torch.__version__,
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "files": {}}
    for key, f in FILES.items():
        rec["files"][key] = dry_one(key, ROOT / f)
        LOG(f"dry-run [{key}]: passed={rec['files'][key].get('passed')}")
    rec["all_passed"] = all(r.get("passed") for r in rec["files"].values())
    rec["status"] = "done"
    rec["finished_at"] = now()
    write_json(OUT_ROOT / "dry_run.json", rec)
    LOG(f"dry-run: all_passed={rec['all_passed']} -> {rel(OUT_ROOT / 'dry_run.json')}")
    LOG.close()
    return 0 if rec["all_passed"] else 1


def s4_only(args) -> int:
    LOG.open(OUT_ROOT / "s4.log", mode="w")
    rec: dict = {"started_at": now(), "files": {}, "status": "running"}
    try:
        for key in ("fp8", "w4a8_kijai", "w4a8_eros"):
            path = ROOT / FILES[key]
            header = sft_quant_format.read_header(str(path))
            prefix = h3_format.detect_h3_prefix(header.tensors)
            rec["files"][key] = s4_one(path, header, prefix)
            LOG(f"s4 [{key}]: {jdump(rec['files'][key])}")
        rec["status"] = "done"
    except BaseException as exc:  # noqa: BLE001
        failure(rec, exc)
    rec["finished_at"] = now()
    write_json(OUT_ROOT / "s4_result.json", rec)
    LOG.close()
    return 0 if rec["status"] == "done" else 1


# --------------------------------------------------------------------------- #
# GPU common: setup like load(), layers from the file
# --------------------------------------------------------------------------- #

def gpu_setup() -> dict:
    """require_cuda_int8 -> require_fp32_matmul -> pin_cublas_workspace (pipeline_h3.py:788-796), in that order."""
    dev = torch.device("cuda")
    backends = QL.require_cuda_int8()
    P.require_fp32_matmul()
    P.pin_cublas_workspace(dev)
    from importlib import metadata  # noqa: PLC0415

    cap = torch.cuda.get_device_capability(dev)
    return {
        "list_backends": backends, "shared_quant": list(QL.shared_quant_status()),
        "has_scaled_mm_v2": has_scaled_mm_v2(), "w4a8_chunked": bool(CB._W4A8_CHUNKED),
        "env_COMFY_KITCHEN_W4A8_CHUNKED": os.environ.get("COMFY_KITCHEN_W4A8_CHUNKED"),
        "comfy_kitchen": metadata.version("comfy-kitchen"), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(dev), "capability": list(cap),
    }


def _layer_keys(name: str, spec) -> list[str]:
    if isinstance(spec, QuantSpec):
        return ["weight", "weight_scale"]
    if spec.format == FP8:
        return ["weight", "weight_scale"] + (["input_scale"] if spec.has_input_scale else [])
    return ["weight", "weight_s_rel", "weight_s_channel"] + (["weight_codebook"] if spec.has_codebook else [])


def load_layer(path: Path, header, cfg, name: str, dev: torch.device) -> nn.Module:
    """One quantized Linear from the file (read_tensors: seek/readinto, no mmap), assigned, moved to dev."""
    spec = cfg.quant[name]
    leaves = _layer_keys(name, spec)
    full = [f"{cfg.prefix}{name}.{leaf}" for leaf in leaves]
    sd = {k[len(cfg.prefix) + len(name) + 1:]: v for k, v in read_tensors(str(path), full, header)}
    n = int(sd["weight"].shape[0])
    k = int(sd["weight"].shape[1]) * (2 if isinstance(spec, SpikeSpec) and spec.format == W4A8 else 1)
    with torch.device("meta"):
        if isinstance(spec, QuantSpec):
            mod = H3Int8Linear(k, n, bias=False, spec=spec)
        else:
            mod = SPIKE_CLASSES[spec.format](k, n, spec=spec, name=name)
    mod.load_state_dict(sd, strict=True, assign=True)
    return mod.to(dev)


def rand_input(m: int, k: int, seed: int, sigma: float, dev: torch.device) -> torch.Tensor:
    g = torch.Generator(device=dev).manual_seed(int(seed))
    return (torch.randn((m, k), generator=g, device=dev, dtype=torch.float32) * float(sigma)).to(torch.bfloat16)


def ulp_key(t: torch.Tensor) -> torch.Tensor:
    i = t.view(torch.int16).int()
    return torch.where(i < 0, -(i & 0x7FFF), i)


def diff_stats(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> dict:
    """Differing elements, max and p99.9 difference in bf16 ulps (row chunks; histogram of the integer diffs)."""
    hist = torch.zeros(65537, dtype=torch.int64, device=a.device)
    n_diff, mx = 0, 0
    for r0 in range(0, a.shape[0], chunk):
        d = (ulp_key(a[r0:r0 + chunk]) - ulp_key(b[r0:r0 + chunk])).abs()
        n_diff += int((d != 0).sum())
        mx = max(mx, int(d.max()))
        hist += torch.bincount(d.clamp(max=65536).flatten(), minlength=65537)
        del d
    cum = torch.cumsum(hist, 0)
    total = int(cum[-1])
    p999 = int(torch.searchsorted(cum, torch.tensor(math.ceil(0.999 * total), device=cum.device)))
    return {"elements": total, "differing": n_diff, "max_ulp": mx, "p99_9_ulp": p999}


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"equal": False, "why": f"shape/dtype {list(a.shape)} {a.dtype} vs {list(b.shape)} {b.dtype}"}
    if torch.equal(a, b):
        return {"equal": True}
    return {"equal": False, **diff_stats(a, b)}


@contextlib.contextmanager
def force_dequant(on: bool):
    prev = SPIKE.fp8_force_dequant
    SPIKE.fp8_force_dequant = bool(on)
    try:
        yield
    finally:
        SPIKE.fp8_force_dequant = prev


@contextlib.contextmanager
def deq_cache_scope(cache: dict | None):
    global _DEQ_CACHE
    prev, _DEQ_CACHE = _DEQ_CACHE, cache
    try:
        yield
    finally:
        _DEQ_CACHE = prev


class _WarnCounter(logging.Handler):
    """Any WARNING (or a 'dequantizing' fallback note) from comfy_kitchen's tensor layer = silent fallback."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if record.levelno >= logging.WARNING or "dequantizing" in msg:
            self.records.append(f"{record.name}:{record.levelname}:{msg[:300]}")


@contextlib.contextmanager
def watch_reference_fallbacks():
    h = _WarnCounter()
    names = ("comfy_kitchen.tensor.fp8", "comfy_kitchen.tensor.w4a8_int8", "comfy_kitchen.tensor.base")
    saved = []
    for n in names:
        lg = logging.getLogger(n)
        saved.append((lg, lg.level))
        lg.addHandler(h)
        lg.setLevel(logging.DEBUG)
    try:
        yield h
    finally:
        for lg, level in saved:
            lg.removeHandler(h)
            lg.setLevel(level)


# --------------------------------------------------------------------------- #
# S1 (DESIGN §8)
# --------------------------------------------------------------------------- #

def s1_layers(cfg, key: str) -> list[str]:
    spike = {n: s for n, s in cfg.quant.items() if isinstance(s, SpikeSpec)}
    if key in ("fp8", "w4a8_kijai"):
        return [f"blocks.{b}.{leaf}" for b in S1_BLOCKS for leaf in LEAVES if f"blocks.{b}.{leaf}" in spike]
    seen, out = set(), []
    for name in sorted(spike, key=lambda n: (int(n.split(".")[1]) if n.startswith("blocks.") else -1, n)):
        pair = (spike[name].group_size, name.rpartition(".")[2])
        if pair not in seen:
            seen.add(pair)
            out.append(name)
    return out


def _seed_for(name: str) -> int:
    block = int(name.split(".")[1]) if name.startswith("blocks.") else 0
    leaf_idx = [leaf.rpartition(".")[2] for leaf in LEAVES].index(name.rpartition(".")[2])
    return S1_SEED + 1000 * block + leaf_idx


def _sigmas(mod) -> list[tuple[str, float, bool]]:
    """(label, sigma, saturating)."""
    out = [("1.0", 1.0, False)]
    if isinstance(mod, H3Fp8Linear) and mod.input_scale is not None and not mod.spec.full_precision:
        s = float(mod.input_scale)
        out += [("448*is/4", 448.0 * s / 4.0, False), ("448*is", 448.0 * s, True)]
    return out


def _rows_concat(fn, n: int, step: int) -> torch.Tensor:
    return torch.cat([fn(n0, min(n0 + step, n)) for n0 in range(0, n, step)], dim=1)


def _rel_l2(out: torch.Tensor, x: torch.Tensor, w_deq_f32: torch.Tensor, rows: int) -> float:
    xs = x[:rows].float()
    ref = xs @ w_deq_f32.t()
    num = torch.linalg.vector_norm(out[:rows].float() - ref)
    den = torch.linalg.vector_norm(ref)
    return float(num / den)


def _fp8_wq(mod: H3Fp8Linear):
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout  # noqa: PLC0415

    return QuantizedTensor(mod.weight, "TensorCoreFP8Layout", TensorCoreFP8Layout.Params(
        scale=mod.weight_scale, orig_dtype=torch.bfloat16, orig_shape=tuple(mod.weight.shape)))


def _fp8_reference(mod: H3Fp8Linear, x: torch.Tensor, mode: str) -> torch.Tensor:
    """ComfyUI's path. scaled: both operands QuantizedTensor -> aten.linear handler (tensor/fp8.py:133-165).
    dequant (supervisor ruling after S1 attempt 1): comfy/ops.py dequantizes a full-precision layer's
    weight FIRST (``weight = weight.dequantize()``) and then calls the plain F.linear(x, w_bf16)."""
    from comfy_kitchen.tensor import QuantizedTensor  # noqa: PLC0415

    wq = _fp8_wq(mod)
    if mode == "scaled":
        xq = QuantizedTensor.from_float(x, "TensorCoreFP8Layout", scale=mod.input_scale)
        return F.linear(xq, wq)
    return F.linear(x, wq.dequantize())


def _fp8_reference_qt_operand(mod: H3Fp8Linear, x: torch.Tensor) -> torch.Tensor:
    """Information only (S1 attempt 1's reference): F.linear(plain x, QuantizedTensor) decomposes into
    aten.t + aten.mm and _handle_fp8_mm dequantizes the TRANSPOSED view (a different GEMM operand layout).
    Not ComfyUI's path for a full-precision layer."""
    return F.linear(x, _fp8_wq(mod))


def _w4a8_reference(mod: H3W4A8Linear, x: torch.Tensor) -> torch.Tensor:
    from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor  # noqa: PLC0415

    params = AsymW4A8Int8Layout.Params(
        scale=mod.weight_s_rel, s_channel=mod.weight_s_channel, codebook=mod.weight_codebook,
        orig_dtype=torch.bfloat16, orig_shape=(mod.out_features, mod.in_features),
        group_size=int(mod.spec.group_size), convrot_groupsize=W4A8_CONVROT)
    return F.linear(x, QuantizedTensor(mod.weight, "AsymW4A8Int8Layout", params))


def _ours_counts(fn):
    before = collections.Counter(COUNTS)
    out = fn()
    delta = {k: COUNTS[k] - before.get(k, 0) for k in ("w4a8_eager", "w4a8_twopass") if COUNTS[k] - before.get(k, 0)}
    return out, delta


def s1_fp8_case(mod: H3Fp8Linear, x: torch.Tensor, saturating: bool) -> dict:
    leaf = mod._leaf
    rec: dict = {}
    modes = ["dequant"] + (["scaled"] if not mod.spec.full_precision and mod.input_scale is not None else [])
    for mode in modes:
        with force_dequant(mode == "dequant"):
            r: dict = {}
            out = mod(x)
            r["1_full_vs_reference"] = compare(out, _fp8_reference(mod, x, mode))
            if mode == "dequant":
                r["info_QuantizedTensor を直接渡す別経路"] = compare(out, _fp8_reference_qt_operand(mod, x))
            if leaf in ("qkv_proj", "fc1"):
                sl = _rows_concat(lambda a, b: mod.forward_rows(x, a, b), mod.out_features, S2_SLICE)
                r["3_slices_vs_full_info"] = compare(sl, out)
                if mode == "scaled":
                    xq, xs = mod.quantize_input(x)
                    slq = _rows_concat(lambda a, b: mod.forward_rows_q(xq, xs, a, b), mod.out_features, S2_SLICE)
                    r["4_rows_q_vs_rows"] = compare(slq, sl)
                    del xq, xs, slq
                del sl
            if mode == "dequant" and leaf in SPIKE.dequant_cache_leaves:
                cache: dict = {}
                with deq_cache_scope(cache):
                    miss = mod(x)
                    hit = mod(x)
                r["5_cached_vs_uncached"] = {"miss": compare(miss, out), "hit": compare(hit, out),
                                             "cache_entries": len(cache)}
                del miss, hit, cache
            if not saturating:
                w = ck.dequantize_per_tensor_fp8(mod.weight, mod.weight_scale, torch.float32)
                r["6_rel_l2_vs_f32"] = _rel_l2(out, x, w, S1_SANITY_ROWS)
                del w
            del out
            rec[mode] = r
    return rec


def s1_w4a8_case(mod: H3W4A8Linear, x: torch.Tensor) -> dict:
    r: dict = {}
    out, ours = _ours_counts(lambda: mod(x))
    r["ours_counts_full"] = ours
    ref, refc = _ours_counts(lambda: _w4a8_reference(mod, x))
    r["reference_counts"] = refc
    r["1_full_vs_reference"] = compare(out, ref)
    del ref
    if mod._leaf in ("qkv_proj", "fc1"):
        sl, slc = _ours_counts(lambda: _rows_concat(lambda a, b: mod.forward_rows(x, a, b), mod.out_features, S2_SLICE))
        r["ours_counts_slices"] = slc
        r["2_slices_vs_full"] = compare(sl, out)
        del sl
    w = ck.dequantize_w4a8_int8_weight(mod.weight, mod.weight_s_rel, mod.weight_s_channel, mod.weight_codebook,
                                       None, int(mod.spec.group_size), W4A8_CONVROT, torch.float32)
    r["6_rel_l2_vs_f32"] = _rel_l2(out, x, w, S1_SANITY_ROWS)
    del w, out
    return r


def _s1_passed(case: dict, fmt: str, saturating: bool) -> list[str]:
    """Failures of one (layer, M, sigma) case (② and ③ are information)."""
    bad = []
    blocks = case.values() if fmt == FP8 else [case]
    for r in blocks:
        if not r["1_full_vs_reference"]["equal"]:
            bad.append("1_full_vs_reference")
        if "4_rows_q_vs_rows" in r and not r["4_rows_q_vs_rows"]["equal"]:
            bad.append("4_rows_q_vs_rows")
        c5 = r.get("5_cached_vs_uncached")
        if c5 and not (c5["miss"]["equal"] and c5["hit"]["equal"]):
            bad.append("5_cached_vs_uncached")
        if not saturating and not r["6_rel_l2_vs_f32"] < S1_SANITY_MAX_REL_L2:
            bad.append(f"6_rel_l2 {r['6_rel_l2_vs_f32']:.4f}")
        for k in ("ours_counts_full", "ours_counts_slices"):
            if r.get(k, {}).get("w4a8_eager"):
                bad.append(f"{k}: eager {r[k]['w4a8_eager']}")
    return bad


def _int8_impl_probe(dev) -> str:
    """int8_linear's backend for a synthetic call with the dtypes / shapes H3Int8Linear uses."""
    x = torch.zeros((4096, 5376), dtype=torch.bfloat16, device=dev)
    w = torch.zeros((1024, 5376), dtype=torch.int8, device=dev)
    s = torch.ones((1024, 1), dtype=torch.float32, device=dev)
    kwargs = {"x": x, "weight": w, "weight_scale": s, "bias": None, "out_dtype": torch.bfloat16, "convrot": True,
              "convrot_groupsize": 256, "input_act": None, "input_act_weight": None, "input_act_eps": 0.0,
              "residual": None, "residual_scale": None}
    try:
        return CK_REGISTRY.get_implementation("int8_linear", kwargs=kwargs).__module__ + " (synthetic kwargs)"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {exc!r}"


def run_s1(args) -> int:
    LOG.open(OUT_ROOT / "s1.log", mode="w")
    install_log_bridge()
    rec: dict = {"script": rel(Path(__file__)), "started_at": now(), "status": "running", "files": {}}
    out_path = OUT_ROOT / "s1_result.json"
    try:
        rec["gpu_check"] = gpu_busy_check()
        rec["setup"] = gpu_setup()
        dev = torch.device("cuda")
        reset_counts()
        failures: list[str] = []
        with spike_patches(), watch_reference_fallbacks() as warn:
            for key in ("fp8", "w4a8_kijai", "w4a8_eros"):
                path = ROOT / FILES[key]
                header, cfg, summary = inspect_core(str(path))
                frec: dict = {"layers": {}}
                rec["files"][key] = frec
                for name in s1_layers(cfg, key):
                    mod = load_layer(path, header, cfg, name, dev)
                    lrec: dict = {"spec": dataclasses.asdict(mod.spec), "shape": [mod.out_features, mod.in_features]}
                    if isinstance(mod, H3Fp8Linear):
                        lrec["input_scale"] = None if mod.input_scale is None else float(mod.input_scale)
                        lrec["weight_scale"] = float(mod.weight_scale)
                    for m in S1_M:
                        for label, sigma, sat in _sigmas(mod):
                            tag = f"M={m} sigma={label}"
                            x = rand_input(m, mod.in_features, _seed_for(name), sigma, dev)
                            w_before = len(warn.records)
                            case = (s1_fp8_case(mod, x, sat) if isinstance(mod, H3Fp8Linear) else s1_w4a8_case(mod, x))
                            case["sigma"] = sigma
                            case["reference_warnings"] = warn.records[w_before:]
                            bad = _s1_passed({k: v for k, v in case.items() if isinstance(v, dict) and k in ("dequant", "scaled")}
                                             if isinstance(mod, H3Fp8Linear) else case, mod.spec.format, sat)
                            if case["reference_warnings"]:
                                bad.append("reference fallback warning")
                            case["failures"] = bad
                            failures += [f"{key} {name} {tag}: {b}" for b in bad]
                            lrec[tag] = case
                            LOG(f"s1 [{key}] {name} {tag}: {'ok' if not bad else bad}")
                            del x
                            torch.cuda.empty_cache()
                    frec["layers"][name] = lrec
                    del mod
                    torch.cuda.empty_cache()
                    write_json(out_path, rec)
        rec["impl_modules"] = {**BACKEND_SEEN, "int8_linear": _int8_impl_probe(dev)}
        rec["counts_total"] = dict(COUNTS)
        rec["eager_total"] = COUNTS.get("w4a8_eager", 0)
        rec["failures"] = failures
        rec["passed"] = not failures and rec["eager_total"] == 0
        rec["status"] = "done"
    except BaseException as exc:  # noqa: BLE001
        failure(rec, exc)
        rec["passed"] = False
    rec["finished_at"] = now()
    write_json(out_path, rec)
    LOG(f"s1: passed={rec.get('passed')} -> {rel(out_path)}")
    LOG.close()
    return 0 if rec.get("passed") else 1


# --------------------------------------------------------------------------- #
# S2 (DESIGN §9)
# --------------------------------------------------------------------------- #

def time_ms(fn) -> float:
    for _ in range(S2_WARMUP):
        out = fn()
        del out
    torch.cuda.synchronize()
    ts = []
    for _ in range(S2_REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
        del out
    return float(statistics.median(ts))


def transient_mib(fn) -> float:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    out_b = sum(t.numel() * t.element_size() for t in (out if isinstance(out, tuple) else (out,)))
    del out
    return round((peak - base - out_b) / 2**20, 1)


def measure(fn) -> dict:
    return {"ms": round(time_ms(fn), 4), "transient_mib": transient_mib(fn)}


def s2_layer(mod, leaf: str, variant: str, dev) -> dict:
    """(a) normal M=39,764 and (b) VRAM 16GB mode shapes for one layer (``leaf``: qkv_proj / out_proj / fc1 / fc2)."""
    k = mod.in_features
    xn = rand_input(S2_M_NORMAL, k, S1_SEED, 1.0, dev)
    rec: dict = {"shape": [mod.out_features, k]}
    if variant == "fp8_dequant":
        w = mod._dequant(mod.weight)
        rec["a_normal"] = {"dequant": measure(lambda: mod._dequant(mod.weight)),
                           "linear": measure(lambda: F.linear(xn, w))}
    else:
        rec["a_normal"] = measure(lambda: mod(xn))
    if leaf == "qkv_proj":
        if variant == "fp8_dequant":
            rec["b_vram16_slice"] = {"dequant": measure(lambda: mod._dequant(mod.weight[0:S2_SLICE])),
                                     "linear": measure(lambda: F.linear(xn, w[0:S2_SLICE]))}
        elif mod.shares_input(xn):
            xq, xs = mod.quantize_input(xn)
            rec["b_quantize_input"] = measure(lambda: mod.quantize_input(xn))
            rec["b_vram16_slice"] = measure(lambda: mod.forward_rows_q(xq, xs, 0, S2_SLICE))
            del xq, xs
        else:
            rec["b_vram16_slice"] = measure(lambda: mod.forward_rows(xn, 0, S2_SLICE))
    elif leaf == "out_proj":
        rec["b_vram16"] = rec["a_normal"]
    else:
        xc = rand_input(S2_M_CHUNK, k, S1_SEED + 1, 1.0, dev)
        if variant == "fp8_dequant" or (isinstance(mod, H3Fp8Linear) and mod.spec.full_precision):
            w2 = w if variant == "fp8_dequant" else mod._dequant(mod.weight)
            rec["b_vram16"] = {"dequant": measure(lambda: mod._dequant(mod.weight)),
                               "linear": measure(lambda: F.linear(xc, w2))}
            del w2
        else:
            rec["b_vram16"] = measure(lambda: mod(xc))
        del xc
    if variant == "fp8_dequant":
        del w
    del xn
    torch.cuda.empty_cache()
    return rec


def _ms(entry) -> tuple[float, float]:
    """(compute ms, dequant ms) of one measured entry."""
    if entry is None:
        return float("nan"), 0.0
    if "dequant" in entry:
        return entry["linear"]["ms"], entry["dequant"]["ms"]
    return entry["ms"], 0.0


def estimate_step(layers: dict, s_tokens: int, n_blocks: int = 50) -> dict:
    """50·[21·t_qkv_slice + t_out + C·(t_fc1 + t_fc2)] + 50·t_deq(fc2) (+ 50·t_quantize_input, shown apart)."""
    c = len(LV.row_chunks(s_tokens, VRAM16_ROW_CHUNK))
    slices = 3 * math.ceil(56 / VRAM16_HEAD_GROUP)
    get = lambda leaf, key: (layers.get(leaf) or {}).get(key)  # noqa: E731
    qkv, qkv_deq = _ms(get("qkv_proj", "b_vram16_slice"))
    out, out_deq = _ms(get("out_proj", "b_vram16"))
    fc1, fc1_deq = _ms(get("fc1", "b_vram16"))
    fc2, fc2_deq = _ms(get("fc2", "b_vram16"))
    qin = (get("qkv_proj", "b_quantize_input") or {}).get("ms", 0.0)
    linear = n_blocks * (slices * (qkv + qkv_deq) + out + out_deq + c * (fc1 + fc2))
    deq_cached = n_blocks * (fc1_deq + fc2_deq)        # one dequant per block for each cached leaf
    total = linear + deq_cached + n_blocks * qin
    return {"S": s_tokens, "C": c, "slices": slices, "linear_ms": round(linear, 2), "dequant_ms": round(deq_cached, 2),
            "quantize_input_ms": round(n_blocks * qin, 2), "total_ms": round(total, 2),
            "formula": "50*[21*(t_qkv_slice+t_qkv_slice_deq) + t_out + t_out_deq + C*(t_fc1+t_fc2)] + 50*(t_deq(fc1)+t_deq(fc2)) + 50*t_quantize_input"}


def s2_variants() -> list[tuple[str, str, str, int | None]]:
    """(variant label, file key, mode, w4a8 G filter)."""
    return [("int8", "int8", "plain", None), ("fp8_scaled", "fp8", "plain", None), ("fp8_dequant", "fp8", "fp8_dequant", None),
            ("w4a8_kijai_G16", "w4a8_kijai", "plain", 16), ("w4a8_eros_G16", "w4a8_eros", "plain", 16),
            ("w4a8_eros_G8", "w4a8_eros", "plain", 8), ("w4a8_eros_G4", "w4a8_eros", "plain", 4)]


def s2_layer_names(cfg, key: str, g: int | None) -> dict[str, str]:
    """leaf -> layer name: block 10 for the uniform files; for Eros the first layer with that (leaf, G)."""
    if key != "w4a8_eros":
        return {leaf.rpartition(".")[2]: f"blocks.{S2_BLOCK}.{leaf}" for leaf in LEAVES}
    out = {}
    for name in sorted(cfg.quant, key=lambda n: (int(n.split(".")[1]) if n.startswith("blocks.") else -1, n)):
        spec = cfg.quant[name]
        leaf = name.rpartition(".")[2]
        if isinstance(spec, SpikeSpec) and spec.group_size == g and leaf not in out:
            out[leaf] = name
    return out


def run_s2(args) -> int:
    LOG.open(OUT_ROOT / "s2.log", mode="w")
    install_log_bridge()
    rec: dict = {"script": rel(Path(__file__)), "started_at": now(), "status": "running", "variants": {}}
    out_path = OUT_ROOT / "s2_result.json"
    try:
        rec["gpu_check"] = gpu_busy_check()
        rec["setup"] = gpu_setup()
        dev = torch.device("cuda")
        inspected = {}
        with spike_patches():
            for label, key, mode, g in s2_variants():
                if key not in inspected:
                    path = ROOT / FILES[key]
                    header, cfg = (_ORIG_INSPECT(str(path)) if key == "int8" else inspect_core(str(path))[:2])
                    inspected[key] = (path, header, cfg)
                path, header, cfg = inspected[key]
                vrec: dict = {"file": rel(path), "layers": {}}
                rec["variants"][label] = vrec
                with force_dequant(mode == "fp8_dequant"):
                    for leaf, name in s2_layer_names(cfg, key, g).items():
                        mod = load_layer(path, header, cfg, name, dev)
                        try:
                            vrec["layers"][leaf] = {"layer": name, **s2_layer(mod, leaf, mode, dev)}
                        except Exception as exc:  # noqa: BLE001 -- recorded per layer; the variant goes on
                            vrec["layers"][leaf] = {"layer": name, "status": "failed", "error": repr(exc),
                                                    "traceback": traceback.format_exc()}
                        LOG(f"s2 [{label}] {name}: {jdump(vrec['layers'][leaf])}")
                        del mod
                        torch.cuda.empty_cache()
                vrec["estimate_per_step_vram16"] = estimate_step(vrec["layers"], S2_M_NORMAL)
                write_json(out_path, rec)
        base = rec["variants"]["int8"]["estimate_per_step_vram16"]["total_ms"]
        rec["ratio_to_int8"] = {k: round(v["estimate_per_step_vram16"]["total_ms"] / base, 3)
                                for k, v in rec["variants"].items()}
        rec["flag_fp8_scaled_over_1_5x_int8"] = rec["ratio_to_int8"].get("fp8_scaled", 0) > S2_FLAG_RATIO
        rec["impl_modules"] = dict(BACKEND_SEEN)
        rec["counts_total"] = dict(COUNTS)
        rec["status"] = "done"
    except BaseException as exc:  # noqa: BLE001
        failure(rec, exc)
    rec["finished_at"] = now()
    write_json(out_path, rec)
    LOG(f"s2: status={rec['status']} -> {rel(out_path)}")
    LOG.close()
    return 0 if rec["status"] == "done" else 1


# --------------------------------------------------------------------------- #
# S3 (DESIGN §10): one condition per subprocess
# --------------------------------------------------------------------------- #

def build_request(cond: Condition, prompt: str, out_dir: Path) -> P.H3GenerateRequest:
    return P.H3GenerateRequest(
        prompt=prompt, seed=DEFAULT_SEED, width=WIDTH, height=HEIGHT, num_frames=NUM_FRAMES,
        output_path=str(out_dir / "output.mp4"),
        num_inference_steps=cond.steps, shift_video=cond.shift_video, shift_audio=cond.shift_audio,
        keep_resident=KEEP_RESIDENT, attention_backend=ATTENTION, block_swap_prefetch=PREFETCH,
        vram16_mode=VRAM16, loras=(),
    )


def conditions_record(cond: Condition, prompt: str) -> dict:
    return {"prompt": prompt, "negative_prompt": "", "seed": DEFAULT_SEED, "steps": cond.steps,
            "shift_video": cond.shift_video, "shift_audio": cond.shift_audio, "attention": ATTENTION,
            "vram16_mode": VRAM16, "blocks_on_gpu": BLOCKS_ON_GPU, "block_swap_prefetch": PREFETCH,
            "keep_resident": KEEP_RESIDENT, "deterministic": DETERMINISTIC, "width": WIDTH, "height": HEIGHT,
            "num_frames": NUM_FRAMES, "lora": None, "transformer": FILES[cond.file]}


class StepClock:
    """The progress callback: forwards to LOG and stamps each denoise step."""

    def __init__(self) -> None:
        self.stamps: dict[int, float] = {}

    def __call__(self, stage: str, index: int, total: int) -> None:
        LOG.progress(stage, index, total)
        if stage == "stage1_denoise":
            self.stamps[int(index)] = time.perf_counter()

    def sec_per_step(self, steps: int) -> dict:
        if 0 not in self.stamps or steps not in self.stamps:
            return {"mean": None, "median_after_first": None}
        diffs = [self.stamps[i + 1] - self.stamps[i] for i in range(steps) if i in self.stamps and i + 1 in self.stamps]
        return {"mean": round((self.stamps[steps] - self.stamps[0]) / steps, 3),
                "median_after_first": round(statistics.median(diffs[1:]), 3) if len(diffs) > 1 else None}


def expected_counts(quant: dict, num_heads: int, t: int, c: int, *, force: bool, cache: bool,
                    cache_leaves: tuple[str, ...]) -> collections.Counter:
    """Per key (and per key@leaf) the calls the VRAM 16GB mode path makes over T steps (lowvram.py:147-187)."""
    slices = 3 * math.ceil(num_heads / VRAM16_HEAD_GROUP)
    exp: collections.Counter = collections.Counter()

    def add(key: str, leaf: str, n: int) -> None:
        exp[key] += n * t
        exp[f"{key}@{leaf}"] += n * t

    for name, spec in quant.items():
        if not isinstance(spec, SpikeSpec) or not name.startswith("blocks."):
            continue
        leaf = name.rpartition(".")[2]
        calls = 1 if leaf == "out_proj" else c         # fc1 / fc2 once per row chunk
        if spec.format == W4A8:
            if leaf == "qkv_proj":
                add("w4a8_rows", leaf, slices)
            else:
                add("w4a8_calls", leaf, calls)
            continue
        dequant = spec.full_precision or force
        if leaf == "qkv_proj":
            if dequant:
                add("fp8_rows", leaf, slices)
                add("fp8_deq_uncached", leaf, slices)
            else:
                add("fp8_quantize_input", leaf, 1)
                add("fp8_rows_q", leaf, slices)
                add("fp8_scaled_calls", leaf, slices)
        elif not dequant:
            add("fp8_scaled_calls", leaf, calls)
        elif cache and leaf in cache_leaves:
            add("fp8_deq_miss", leaf, 1)
            if calls > 1:
                add("fp8_deq_hit", leaf, calls - 1)
        else:
            add("fp8_deq_uncached", leaf, calls)
    return exp


def plan_expectations(cond: Condition, t: int, c: int, b: int) -> dict[str, int]:
    """The plan's / design's own numbers, written out by hand (independent of expected_counts)."""
    if cond.name == "fp8":
        return {"fp8_quantize_input": 50 * t, "fp8_rows_q": 1050 * t, "fp8_deq_miss": 50 * t,
                "fp8_deq_hit": 50 * t * (c - 1), "fp8_scaled_calls": 50 * t * (22 + c)}
    if cond.name == "fp8_dequant":
        fc1 = ({"fp8_deq_miss@fc1": 50 * t} if "fc1" in SPIKE.dequant_cache_leaves
               else {"fp8_deq_uncached@fc1": 50 * t * c})          # --no-fc1-cache (risk §11)
        return {"fp8_scaled_calls": 0, "fp8_rows": 1050 * t, **fc1, "fp8_deq_miss@fc2": 50 * t}
    if cond.name.startswith("w4a8"):
        return {"w4a8_calls+w4a8_rows": b * t * (22 + 2 * c), "w4a8_eager": 0}
    return {"fp8_scaled_calls": 0, "w4a8_calls+w4a8_rows": 0, "w4a8_eager": 0}


def self_check(cond: Condition, cfg, engine: dict, counts: dict, audit: dict) -> dict:
    t = cond.steps
    seq = (((engine or {}).get("h3") or {}).get("phases") or {}).get("denoise", {}).get("sequence") or {}
    s = seq.get("total")
    if not isinstance(s, int):
        return {"passed": False, "why": f"no sequence total in the generate record ({seq!r})"}
    c = len(LV.row_chunks(s, VRAM16_ROW_CHUNK))
    exp = expected_counts(cfg.quant, cfg.num_heads, t, c, force=cond.fp8_force_dequant, cache=cond.dequant_cache,
                          cache_leaves=SPIKE.dequant_cache_leaves)
    keys = sorted((set(exp) | set(counts)) - {k for k in counts if k.startswith("w4a8_twopass")})
    per_key = {k: {"expected": exp.get(k, 0), "actual": counts.get(k, 0)} for k in keys}
    bad = [k for k, v in per_key.items() if v["expected"] != v["actual"]]
    fmt = W4A8 if cond.name.startswith("w4a8") else FP8
    blocks = sorted({int(n.split(".")[1]) for n, sp in cfg.quant.items()
                     if isinstance(sp, SpikeSpec) and n.startswith("blocks.") and sp.format == fmt})
    b = len(blocks)
    plan = plan_expectations(cond, t, c, b)
    combined = dict(counts)
    combined["w4a8_calls+w4a8_rows"] = counts.get("w4a8_calls", 0) + counts.get("w4a8_rows", 0)
    plan_rec = {k: {"expected": v, "actual": combined.get(k, 0)} for k, v in plan.items()}
    bad += [f"plan:{k}" for k, v in plan_rec.items() if v["expected"] != v["actual"]]
    fc2_calls = sum(v for k, v in counts.items() if k.endswith("@fc2") and not k.startswith("refiner:"))
    fc2_layers = sum(1 for n, sp in cfg.quant.items() if isinstance(sp, SpikeSpec) and n.startswith("blocks.") and n.endswith("fc2"))
    c_observed = fc2_calls / (fc2_layers * t) if fc2_layers else None
    if fc2_layers and c_observed != c:
        bad.append(f"C observed {c_observed} != {c}")
    if c != math.ceil(s / VRAM16_ROW_CHUNK):
        bad.append(f"row_chunks {c} != ceil(S/4096) {math.ceil(s / VRAM16_ROW_CHUNK)}")
    if PATCH_HITS.get("block_forward", 0) != cfg.num_layers * t:
        bad.append(f"block_forward wrapper hits {PATCH_HITS.get('block_forward', 0)} != {cfg.num_layers * t}")
    if not audit.get("passed") or audit.get("adaln_t_table") != "float32":
        bad.append("dtype audit")
    return {"passed": not bad, "failures": bad, "T": t, "S": s, "C": c, "B": b, "slices": 3 * math.ceil(cfg.num_heads / VRAM16_HEAD_GROUP),
            "expected": {k: v["expected"] for k, v in per_key.items()}, "actual": {k: v["actual"] for k, v in per_key.items()},
            "plan": plan_rec, "fc2_calls_observed": fc2_calls, "C_observed": c_observed,
            "block_forward_hits": PATCH_HITS.get("block_forward", 0)}


def expected_modules(cond: Condition) -> dict[str, int]:
    return {"int8": {"H3Int8Linear": 200}, "fp8": {"H3Fp8Linear": 200}, "fp8_dequant": {"H3Fp8Linear": 200},
            "w4a8_kijai": {"H3W4A8Linear": 200}, "w4a8_eros": {"H3Int8Linear": 20, "H3W4A8Linear": 180}}[cond.name]


def run_condition(name: str, prompt: str) -> int:
    from engine_h3.pipeline_h3 import H3LoadOptions, H3Paths, H3Pipeline  # noqa: PLC0415

    cond = CONDITIONS[name]
    out_dir = OUT_ROOT / S3_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.open(out_dir / "run.log", mode="w")
    install_log_bridge()
    rec: dict = {"condition": name, "file": FILES[cond.file], "conditions": conditions_record(cond, prompt),
                 "started_at": now(), "status": "running"}
    result_path = out_dir / "result.json"
    SPIKE.fp8_force_dequant = cond.fp8_force_dequant
    SPIKE.dequant_cache = cond.dequant_cache
    rec["spike"] = SPIKE.record()
    reset_counts()
    ok = False
    try:
        rec["gpu_check"] = gpu_busy_check()
        req = build_request(cond, prompt, out_dir)
        P.validate_request(req)
        rec["request"] = dataclasses.asdict(req)
        _h, cfg, summary = inspect_core(str(ROOT / FILES[cond.file]))
        rec["inspect"] = summary
        write_json(result_path, rec)
        clock = StepClock()
        paths = H3Paths(**{k: str(ROOT / v) for k, v in {**PATHS, "transformer": FILES[cond.file]}.items()})
        options = H3LoadOptions(blocks_on_gpu=BLOCKS_ON_GPU, attention_backend=ATTENTION,
                                block_swap_prefetch=PREFETCH, deterministic=DETERMINISTIC)
        with spike_patches():
            pipe = H3Pipeline(paths, options, device=torch.device("cuda"), progress=clock, log=LOG)
            try:
                t0 = time.perf_counter()
                build_report = pipe.load()
                rec["load_seconds"] = round(time.perf_counter() - t0, 2)
                rec["build_report"] = build_report
                write_json(result_path, rec)
                result = pipe.generate(req)
                rec["engine"] = {"seconds": result.get("seconds"), "peak_vram_mb": result.get("peak_vram_mb"),
                                 "peak_vram_reserved_mb": result.get("peak_vram_reserved_mb"),
                                 "h3": {"phases": (result.get("h3") or {}).get("phases")}, "full": result}
            finally:
                try:
                    pipe.close()
                except Exception as exc:  # noqa: BLE001
                    LOG(f"pipe.close failed (ignored): {exc!r}")
        rec["restored"] = patches_restored()
        rec["backends"] = {"list_backends": pipe.quant_backends, "impl_modules": dict(BACKEND_SEEN),
                           "has_scaled_mm_v2": has_scaled_mm_v2(), "w4a8_chunked": bool(CB._W4A8_CHUNKED),
                           "env_COMFY_KITCHEN_W4A8_CHUNKED": os.environ.get("COMFY_KITCHEN_W4A8_CHUNKED")}
        rec["modules"] = {"swap": dict(SWAP_RECORD), "expected": expected_modules(cond),
                          "from_audit": AUDIT_LAST.get("modules"), "dtype_audit": dict(AUDIT_LAST)}
        rec["sec_per_step"] = clock.sec_per_step(cond.steps)
        rec["counts"] = dict(COUNTS)
        rec["patch_hits"] = dict(PATCH_HITS)
        rec["self_check"] = self_check(cond, cfg, rec["engine"]["full"], dict(COUNTS), dict(AUDIT_LAST))
        swap_ok = all(SWAP_RECORD.get(k, 0) == v for k, v in expected_modules(cond).items())
        if not swap_ok:
            rec["self_check"]["passed"] = False
            rec["self_check"]["failures"].append(f"modules {dict(SWAP_RECORD)} != {expected_modules(cond)}")
        if PATCH_HITS.get("inspect_h3_dit", 0) < 2:
            rec["self_check"]["passed"] = False
            rec["self_check"]["failures"].append(f"spike inspector hits {PATCH_HITS.get('inspect_h3_dit', 0)} < 2 (load + build)")
        out_mp4 = out_dir / "output.mp4"
        rec["outputs"] = {"mp4": rel(out_mp4), "exists": out_mp4.is_file(),
                          "probe": ffprobe(out_mp4) if out_mp4.is_file() else None}
        rec["status"] = "done"
        ok = bool(rec["self_check"]["passed"] and rec["restored"] and out_mp4.is_file())
        LOG(f"s3 [{name}]: done; self_check={rec['self_check']['passed']} failures={rec['self_check'].get('failures')}")
    except BaseException as exc:  # noqa: BLE001 -- interruption is recorded too
        failure(rec, exc)
        rec["counts"] = dict(COUNTS)
        rec["patch_hits"] = dict(PATCH_HITS)
        rec["modules"] = {"swap": dict(SWAP_RECORD), "dtype_audit": dict(AUDIT_LAST)}
    finally:
        rec["finished_at"] = now()
        write_json(result_path, rec)
        LOG.close()
    return 0 if ok else 1


def run_s3_all(args) -> int:
    LOG.open(OUT_ROOT / "s3_all.log")
    rec: dict = {"started_at": now(), "runs": {}}
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT))
    env.setdefault("PYTHONUTF8", "1")
    env["H3_QUANT_SPIKE_PARENT_PID"] = str(os.getpid())

    def one(name: str) -> bool:
        LOG(f"s3-all: start {name}")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--s3", name, "--prompt", args.prompt]
        if args.no_fc1_cache:
            cmd.append("--no-fc1-cache")
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
        res = read_json(OUT_ROOT / S3_DIR / name / "result.json")
        passed = proc.returncode == 0 and res.get("status") == "done" and (res.get("self_check") or {}).get("passed")
        rec["runs"][name] = {"returncode": proc.returncode, "status": res.get("status"), "passed": bool(passed)}
        LOG(f"s3-all: {name} returncode={proc.returncode} passed={bool(passed)}")
        write_json(OUT_ROOT / "s3_all.json", rec)
        return bool(passed)

    first_ok = all([one(n) for n in S3_FIRST])
    if first_ok:
        one(S3_LAST)
    else:
        rec["runs"][S3_LAST] = {"skipped": "not every one of the first four conditions passed"}
    write_summary()
    rec["finished_at"] = now()
    write_json(OUT_ROOT / "s3_all.json", rec)
    LOG.close()
    return 0 if all(r.get("passed") for r in rec["runs"].values()) else 1


def _g(v, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_summary() -> Path:
    lines = ["# S3 summary (h3 fp8 / w4a8 spike, 2026-11-12)", "",
             "| condition | load s | total s | s/step (mean) | peak VRAM alloc / reserved MiB | self-check | notes |",
             "|---|---|---|---|---|---|---|"]
    for name in (*S3_FIRST, S3_LAST):
        r = read_json(OUT_ROOT / S3_DIR / name / "result.json")
        if not r:
            lines.append(f"| {name} | — | — | — | — | — | not run |")
            continue
        e = r.get("engine") or {}
        sc = r.get("self_check") or {}
        notes = r.get("error") or "; ".join(sc.get("failures") or []) or ""
        lines.append(f"| {name} | {_g(r.get('load_seconds'))} | {_g(e.get('seconds'))} | "
                     f"{_g((r.get('sec_per_step') or {}).get('mean'), 3)} | {_g(e.get('peak_vram_mb'))} / "
                     f"{_g(e.get('peak_vram_reserved_mb'))} | {sc.get('passed')} | {str(notes)[:200]} |")
    path = OUT_ROOT / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# --post: contact sheets, hf-round16/
# --------------------------------------------------------------------------- #

def make_sheet(frames_by_cond: dict[str, np.ndarray], path: Path) -> None:
    """Rows = conditions, columns = frames 0 / 36 / 72; each tile's long side 640."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    scale = SHEET_TILE_LONG / max(WIDTH, HEIGHT)
    tw, th = round(WIDTH * scale), round(HEIGHT * scale)
    head = 22
    sheet = Image.new("RGB", (tw * len(SHEET_FRAMES), (th + head) * len(frames_by_cond)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (name, frames) in enumerate(frames_by_cond.items()):
        y = i * (th + head)
        for j, f in enumerate(SHEET_FRAMES):
            if f < frames.shape[0]:
                sheet.paste(Image.fromarray(frames[f]).resize((tw, th), Image.LANCZOS), (j * tw, y + head))
            draw.text((j * tw + 6, y + 5), f"{name} f{f:03d}", fill=(235, 235, 235))
    sheet.save(path)


def post_process(args) -> int:
    LOG.open(OUT_ROOT / "post.log", mode="w")
    rec: dict = {"made_at": now(), "clips": {}, "status": "running"}
    try:
        hf = OUT_ROOT / HF_DIR
        hf.mkdir(parents=True, exist_ok=True)
        frames_all = {}
        for name in (*S3_FIRST, S3_LAST):
            mp4 = OUT_ROOT / S3_DIR / name / "output.mp4"
            if not mp4.is_file():
                rec["clips"][name] = {"missing": rel(mp4)}
                continue
            frames = read_frames(mp4, NUM_FRAMES)
            frames_all[name] = frames
            sheet = OUT_ROOT / S3_DIR / name / "sheet.png"
            make_sheet({name: frames}, sheet)
            steps = CONDITIONS[name].steps
            clip = f"h3_{name}_{WIDTH}x{HEIGHT}_{steps}steps_seed{DEFAULT_SEED}.mp4"
            shutil.copy2(mp4, hf / clip)
            shutil.copy2(sheet, hf / f"sheet_{name}.png")
            rec["clips"][name] = {"clip": clip, "probe": ffprobe(mp4), "sheet": f"sheet_{name}.png"}
            LOG(f"post: {name} -> {clip}")
        if frames_all:
            make_sheet(frames_all, hf / "sheet_all.png")
        summary = write_summary()
        shutil.copy2(summary, hf / "summary.md")
        write_readme(hf, rec)
        rec["status"] = "done"
    except BaseException as exc:  # noqa: BLE001
        failure(rec, exc)
    write_json(OUT_ROOT / "post.json", rec)
    LOG.close()
    return 0 if rec["status"] == "done" else 1


def write_readme(hf: Path, post: dict) -> None:
    """Skeleton for the owner (Japanese); the supervisor completes the findings after viewing."""
    L = [
        "# round16-h3-quant — MiniMax H3 の fp8・w4a8 の試作の検証（2026-11-12）",
        "",
        "台帳 §1-283 の第 0 段です。ComfyUI と同じ comfy-kitchen の関数を同じ引数で呼ぶ試作の層を、製品のコードを変えずに"
        "台本の中だけで差し込み、同じプロンプト・同じ種で 5 本を生成しました。基準は公式の int8 です。",
        "",
        "見ていただきたいこと: 映像と音声が破綻していないか（別のモデル・別の計算経路なので、絵が基準と違うこと自体は想定どおりです）。",
        "",
        f"プロンプト: {DEFAULT_PROMPT}",
        "",
        "## 条件と数値",
        "",
        "| 条件 | 重みのファイル | ステップ | 読み込み秒 | 全体の秒 | 1 ステップの秒 | VRAM の山（割り当て・予約 MiB） | 自己検査 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in (*S3_FIRST, S3_LAST):
        r = read_json(OUT_ROOT / S3_DIR / name / "result.json")
        e = r.get("engine") or {}
        L.append(f"| {name} | {Path(FILES[CONDITIONS[name].file]).name} | {CONDITIONS[name].steps} | "
                 f"{_g(r.get('load_seconds'))} | {_g(e.get('seconds'))} | {_g((r.get('sec_per_step') or {}).get('mean'), 3)} | "
                 f"{_g(e.get('peak_vram_mb'))}・{_g(e.get('peak_vram_reserved_mb'))} | {(r.get('self_check') or {}).get('passed')} |")
    L += [
        "",
        "共通: 1280×768・73 フレーム・種 1234・注意の実装 sage・VRAM 16GB モード on・先読み on・LoRA なし。",
        "fp8_dequant は fp8 の全層を bf16 へ戻して計算する経路（RTX 30 系で使う経路）を RTX 40 系で強制したものです。",
        "",
        "## ファイル",
        "",
    ]
    for name, c in post.get("clips", {}).items():
        if "clip" in c:
            L.append(f"- `{c['clip']}`・`{c['sheet']}`（{name}。並び画像はフレーム 0・36・72）")
    L += ["- `sheet_all.png`: 5 本の並び画像（行が条件）", "- `summary.md`: 数値の表", "",
          "## 所見", "", "（視聴の後に記入）", ""]
    (hf / "README.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="CPU: inspectors, meta build, key map, dtype audit, S4")
    g.add_argument("--s1", action="store_true", help="GPU: bit identity, backends, eager counting")
    g.add_argument("--s2", action="store_true", help="GPU: per-layer timing")
    g.add_argument("--s3", choices=sorted(CONDITIONS), help="GPU: one end-to-end condition (run by --s3-all)")
    g.add_argument("--s3-all", action="store_true", help="GPU: every condition, each in its own subprocess")
    g.add_argument("--s4", action="store_true", help="CPU: sft_quant_format.layer_schemes on the 3 new files")
    g.add_argument("--post", action="store_true", help="CPU: contact sheets, summary.md, hf-round16/")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--no-fc1-cache", action="store_true",
                    help="risk §11: keep only fc2's dequantized weight per block (fp8_dequant VRAM shortfall)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.no_fc1_cache:
        SPIKE.dequant_cache_leaves = ("fc2",)
    if args.dry_run:
        return dry_run(args)
    if args.s4:
        return s4_only(args)
    if args.post:
        return post_process(args)
    if args.s3_all:
        return run_s3_all(args)       # the parent never touches CUDA; each child checks it
    if not torch.cuda.is_available():
        print("CUDA is not available (the GPU stages need it)", file=sys.stderr)
        return 2
    if args.s1:
        return run_s1(args)
    if args.s2:
        return run_s2(args)
    return run_condition(args.s3, args.prompt)


if __name__ == "__main__":
    raise SystemExit(main())
