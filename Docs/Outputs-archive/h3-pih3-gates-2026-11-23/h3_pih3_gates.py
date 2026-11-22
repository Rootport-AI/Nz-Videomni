r"""pi-H3 stage 1: the GPU checks of the plan's "実機の確認" (MiniMax H3; product code read only).

Items (canonical order = the order they are run; group 1 -> 6):

  group 1  off path unchanged
    F0   worker selftest (engine venv, no backend): mp4 sha256 == the reference fingerprint 0da6111c...
    F1   API, pi_h3 absent: FL2VA, 20 steps, no LoRA, 1280x768x73 = the 2026-11-22 gate's R1 condition;
         video / audio stream MD5 == R1's recorded MD5 (and no pi-H3 key in metadata / worker log)
    M1   engine-direct rerun of the spike's I1c (Inpainting, Turbo LoRA 0.8, 4 steps, no pi-H3) with the
         same H3LoadOptions / request (engine_direct.py); stream MD5 == the spike's I1c output.mp4
    M2   (optional, only with --only) the same for O1c (Outpainting)
  group 2  the spike's P1..P4 through the product path (POST /generate with pi_h3)
    P1 FL2VA 4steps / P2 FL2VA 2steps / P3 Sulphur-3 4steps / P4 Sulphur-3 2steps
         stream MD5 vs the spike's runs/Pn/output.mp4; when they differ: per-frame PSNR (min / mean),
         ratio of differing values, max difference; audio sample difference. Record checks
         (pi_h3_used, h3.pi_h3: LoRA pairs, adapter counts, sigmas == the spike's, DWD chunks).
  group 3  keep_resident on: off(F1 condition) -> 4steps -> 2steps -> off in one backend session,
    R    each compared by stream MD5 with the standalone run (off: F1, else R1's record; 4: P1; 2: P2)
  group 4  large frames, FL2VA 4steps: L1 1920x1088x124, L2 2560x1472x73 (inside the comfort line
    L1 L2 checked against the descriptor's rows budget); DWD stage VRAM / time / chunks, worker RSS peak,
         20-minute job limit; an OOM is recorded with its stage
  group 5  other inputs, 4steps (finish + frames + audio): I2V1 first frame, I2V2 first + last,
    ...  REF Ref2VA (Ref2VA weights, the 2026-10-28 G-R1 material), NEG negative prompt (VSF),
         V16OFF VRAM 16GB mode off (falls back to 640x384x124 on GPU_OOM); Q1 (optional, --only) w4a8 weight
  group 6  A2V Strength 0.5 (2026-10-25 G3 run B material), pi-H3 4steps and Turbo LoRA 0.8 4 steps
    A2V  (for the owner's eyes and ears; numbers only here)

A result of "changed" in group 1 or "broken" in group 2 writes runs/_halt.json; later items refuse to run
until --continue-after-halt is given.

From the repo root (Git Bash; the APP venv; this script never imports torch):

    export PYTHONUTF8=1
    G=outputs/h3-pih3-gates-2026-11-23/h3_pih3_gates.py
    .venv/Scripts/python.exe -X utf8 $G --dry-run                # plan + file / header / request checks, no GPU
    .venv/Scripts/python.exe -X utf8 $G --gate-check             # the GPU gate only (port, workers, nvidia-smi query)
    .venv/Scripts/python.exe -X utf8 $G --live --only F0         # one item (GPU); done items are skipped
    .venv/Scripts/python.exe -X utf8 $G --post-only              # comparisons + summary from runs/

Precedent: Docs/Outputs-archive/h3-mixed-int8-2026-11-22/h3_mixed_int8_gate_api.py (temp yaml / state,
own port, gate, stop in finally, stream MD5, home-path scrub). Differences: plain HTTP (the MCP tools have no
pi_h3 yet: stage 4), one backend session per item, the gate records counts only (no process paths).

Exit codes: 0 = ran (verdicts are in the records), 1 = gate / setup refused, 2 = bad arguments,
3 = a job failed / timed out / an item was refused by the halt marker.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY_APP = REPO / ".venv" / "Scripts" / "python.exe"
PY_ENG = REPO / ".venv-engine-h3" / "Scripts" / "python.exe"
RUNS = HERE / "runs"
RT = HERE / "_rt"
LOG_DIR = HERE / "logs"
HALT = RUNS / "_halt.json"
SUMMARY_JSON = HERE / "summary.json"
SUMMARY_MD = HERE / "summary.md"
PORT_DEFAULT = 18698
OWNER_PORT = 18620
GPU_BUSY_MIB = 3072
TEMP_CONFIG_NAME = "config.h3pih3gates.yaml"
TEMP_STATE_NAME = "state.h3pih3gates.json"
H3 = "MiniMaxH3"
H3_CATEGORIES = ("transformer", "text_encoder", "video_vae", "audio")

SPIKE = REPO / "outputs" / "h3-pih3-spike-2026-11-22"
MIXED_SUMMARY = REPO / "Docs" / "Outputs-archive" / "h3-mixed-int8-2026-11-22" / "summary.json"
MIXED_R1_MP4 = REPO / "outputs" / "h3-mixed-int8-gate-2026-11-22" / "runs" / "R1" / "output.mp4"
FINGERPRINT = "0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9"

WEIGHTS = REPO / "models" / "MiniMaxH3" / "Weights"
FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
SULPHUR = "Sulphur-3_Ref2VA-hybrid_mid_int8_convrot_keepB0-1-49.safetensors"
REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
W4A8 = "10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors"
TURBO_LORA = "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
LORA_DIR = REPO / "models" / "MiniMaxH3" / "LoRA"
PI_FILES = {"4steps": REPO / "models/MiniMaxH3/VAE/pi-H3/pi_h3_4step_dwd_adapter_int8_convrot.safetensors",
            "2steps": REPO / "models/MiniMaxH3/VAE/pi-H3/pi_h3_2step_dwd_adapter_int8_convrot.safetensors"}

#: Same string as the spike's DEFAULT_PROMPT and the 2026-11-22 gate's R1.
CITY = ("A busy city street at dusk seen from the sidewalk. Pedestrians walk past the camera, their footsteps "
        "clicking on the pavement; cars and a bus drive by with engine hum and tire noise, and a car horn honks "
        "in the distance. Steady camera.")
#: The 2026-10-19 keyframe gates' prompt (g3_i2v_first*.cmd / g4_first_last.cmd), with their images.
RAINCOAT = ("A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles. "
            "She looks at the camera and says <d>[English] It finally stopped raining.</d>")
I2V_FIRST = REPO / "outputs" / "h3-gate2" / "i2v_input.png"
KF_FIRST = REPO / "outputs" / "h3-keyframes-2026-10-19" / "kf_first.png"
KF_LAST = REPO / "outputs" / "h3-keyframes-2026-10-19" / "kf_last.png"
#: The 2026-10-28 Ref2VA gate G-R1 (picture 1 = the elderly gentleman; hf-round10/README.md).
REF_PICTURE = REPO / "Docs" / "Outputs-archive" / "h3-ref2va-2026-10-28" / "17fde0c1-image.png"
REF_PROMPT = ("<Picture 1> is an elderly gentleman with white hair and a beard in a dark suit. He walks slowly "
              "through a sunlit park, turns to the camera and smiles. Natural ambient sound.")
#: The 2026-10-25 A2V gate G3 run B (outputs/h3_a2v_g3/drive.py).
A2V_IMAGE = REPO / "uploads" / "h3-a2v-development" / "00043-2168133585.jpg"
A2V_WAV = REPO / "uploads" / "h3-a2v-development" / "実験の成功、楽しみにしていますね.wav"
A2V_PROMPT = ("An anime-style girl speaks cheerfully to the camera, natural lip movement, gentle head motion, "
              "soft indoor light")
NEGATIVE = "blurry, low quality"   # the 2026-11-09 VSF gate's negative prompt (g1_on.cmd)

#: "broken" in group 2: a per-frame PSNR below this (dB), or a frame-count / audio mismatch.
BROKEN_PSNR_DB = 30.0
SIGMA_TOL = 1e-6
DEFAULT_JOB_TIMEOUT = 3600
LARGE_JOB_TIMEOUT = 1200
WORKER_RE = re.compile(r"-m\s+engine\w*\.worker|engine\w*[\\/]worker\.py|engine_direct\.py")

# ---------------------------------------------------------------------------------------------
# Home-folder scrubbing (copied from the precedent)
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
    changed = []
    for base in (LOG_DIR, RT, RUNS):
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".log", ".txt", ".json", ".yaml", ".yml"):
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                new = _HOME_RE.sub("<user>", text)
                if new != text:
                    p.write_text(new, encoding="utf-8", newline="\n")
                    changed.append(rel(p))
    return changed


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except (ValueError, OSError):
        return scrub(str(p))


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------


class Log:
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


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scrub(text.replace("\r\n", "\n")), encoding="utf-8", newline="\n")


def tool(name: str) -> str:
    local = REPO / "tools" / "ffmpeg" / "bin" / f"{name}.exe"
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found")
    return found


def run_cmd(cmd: list[str], timeout: float | None = None, cwd=None, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                          cwd=cwd, env=env)


def git_info() -> dict:
    out: dict[str, Any] = {}
    try:
        out["head"] = run_cmd(["git", "rev-parse", "HEAD"], 60, str(REPO)).stdout.strip() or None
        out["branch"] = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], 60, str(REPO)).stdout.strip() or None
        st = run_cmd(["git", "status", "--porcelain"], 120, str(REPO))
        out["status_porcelain"] = [ln for ln in (st.stdout or "").splitlines() if ln.strip()]
        out["dirty"] = bool(out["status_porcelain"])
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def get(d: Any, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


# ---------------------------------------------------------------------------------------------
# Media: probe, stream MD5, sha256, frame / audio differences (ffmpeg + numpy, no GPU)
# ---------------------------------------------------------------------------------------------


def ffprobe(path: Path) -> dict:
    exe = tool("ffprobe")
    v = run_cmd([exe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                 "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration", "-of", "json",
                 str(path)])
    a = run_cmd([exe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                 "stream=codec_name,sample_rate,channels,duration", "-of", "json", str(path)])
    vs = (json.loads(v.stdout or "{}").get("streams") or [{}])[0]
    as_ = (json.loads(a.stdout or "{}").get("streams") or [None])[0]
    nb = vs.get("nb_read_frames")
    vs["nb_read_frames"] = int(nb) if nb not in (None, "N/A") else None
    return {"path": rel(path), "video": vs, "audio": as_}


def stream_md5(mp4: Path, which: str) -> str | None:
    """MD5 of one stream's packets, the container left out (precedent's stream_md5)."""
    proc = subprocess.run([tool("ffmpeg"), "-v", "error", "-i", str(mp4), "-map", f"0:{which}:0", "-c", "copy",
                           "-f", "md5", "-"], capture_output=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        if "matches no streams" in (proc.stderr or ""):
            return None
        raise RuntimeError(f"ffmpeg md5 ({which}) failed: {proc.stderr.strip()[-300:]}")
    line = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    return line[0].split("=", 1)[1] if line[0].startswith("MD5=") else None


def stream_md5s(mp4: Path) -> dict:
    return {"video": stream_md5(mp4, "v"), "audio": stream_md5(mp4, "a")}


def sha256(path: Path) -> str:
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_diff(a: Path, b: Path) -> dict:
    """Per-frame PSNR (rgb24, 0-255) between two mp4 of the same size, frame by frame through two ffmpeg pipes."""
    import numpy as np  # noqa: PLC0415

    pa, pb = ffprobe(a)["video"], ffprobe(b)["video"]
    if (pa.get("width"), pa.get("height")) != (pb.get("width"), pb.get("height")):
        return {"error": f"size differs {pa.get('width')}x{pa.get('height')} vs {pb.get('width')}x{pb.get('height')}"}
    w, h = int(pa["width"]), int(pa["height"])
    n = w * h * 3
    cmd = lambda p: [tool("ffmpeg"), "-v", "error", "-i", str(p), "-map", "0:v:0", "-f", "rawvideo",  # noqa: E731
                     "-pix_fmt", "rgb24", "-"]
    ka = subprocess.Popen(cmd(a), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    kb = subprocess.Popen(cmd(b), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    psnr, ratio, maxd, frames = [], [], 0, 0
    try:
        while True:
            ra, rb = ka.stdout.read(n), kb.stdout.read(n)
            if len(ra) < n or len(rb) < n:
                extra = (len(ra) == n) != (len(rb) == n)
                break
            x = np.frombuffer(ra, np.uint8).astype(np.int16)
            y = np.frombuffer(rb, np.uint8).astype(np.int16)
            d = np.abs(x - y)
            mse = float((d.astype(np.float64) ** 2).mean())
            psnr.append(float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse)))
            ratio.append(float((d > 0).mean()))
            maxd = max(maxd, int(d.max()))
            frames += 1
    finally:
        for k in (ka, kb):
            k.kill()
            k.wait()
    fin = [p for p in psnr if p != float("inf")]
    return {"frames_compared": frames, "frame_count_differs": bool(extra) or pa.get("nb_read_frames") != pb.get("nb_read_frames"),
            "identical_frames": sum(1 for p in psnr if p == float("inf")),
            "psnr_min_db": round(min(fin), 3) if fin else None,
            "psnr_mean_db_finite": round(sum(fin) / len(fin), 3) if fin else None,
            "differing_value_ratio": round(sum(ratio) / len(ratio), 6) if ratio else None,
            "max_abs_diff": maxd, "psnr_per_frame": [None if p == float("inf") else round(p, 2) for p in psnr]}


def audio_diff(a: Path, b: Path) -> dict:
    import numpy as np  # noqa: PLC0415

    def pcm(p):
        r = subprocess.run([tool("ffmpeg"), "-v", "error", "-i", str(p), "-map", "0:a:0", "-f", "f32le", "-ac", "2",
                            "-"], capture_output=True)
        return np.frombuffer(r.stdout, np.float32) if r.returncode == 0 else None

    x, y = pcm(a), pcm(b)
    if x is None or y is None:
        return {"error": "no audio stream in one of the files"}
    m = min(len(x), len(y))
    d = np.abs(x[:m] - y[:m]) if m else np.zeros(1, np.float32)
    return {"samples": [int(len(x)), int(len(y))], "max_abs_diff": float(d.max()),
            "rms_diff": float(np.sqrt((d.astype(np.float64) ** 2).mean()))}


def compare(mp4: Path, ref: dict) -> dict:
    """``ref``: {"label", "mp4": Path | None, "md5": {"video", "audio"} | None}. MD5 first; when it differs and
    the reference mp4 exists, the frame / audio differences."""
    out: dict[str, Any] = {"label": ref["label"], "ref_mp4": rel(ref["mp4"]) if ref.get("mp4") else None}
    ref_md5 = ref.get("md5")
    if ref_md5 is None and ref.get("mp4") and Path(ref["mp4"]).is_file():
        ref_md5 = stream_md5s(Path(ref["mp4"]))
    if ref_md5 is None:
        out["error"] = "reference missing"
        return out
    md5 = stream_md5s(mp4)
    out.update(md5=md5, ref_md5=ref_md5, video_equal=md5["video"] == ref_md5.get("video"),
               audio_equal=md5["audio"] == ref_md5.get("audio"))
    out["equal"] = out["video_equal"] and out["audio_equal"]
    if not out["equal"] and ref.get("mp4") and Path(ref["mp4"]).is_file():
        if not out["video_equal"]:
            fd = frame_diff(mp4, Path(ref["mp4"]))
            out["frames"] = {k: v for k, v in fd.items() if k != "psnr_per_frame"}
            out["psnr_per_frame"] = fd.get("psnr_per_frame")
        if not out["audio_equal"]:
            out["audio"] = audio_diff(mp4, Path(ref["mp4"]))
    return out


# ---------------------------------------------------------------------------------------------
# The rows formula (Docs/COMFORT_LIMIT_TABLE.md §17.3) and the comfort line
# ---------------------------------------------------------------------------------------------


def latents(f: int) -> int:
    return (int(f) - 5) // 17 * 5 + 2


def body_frames(b: dict) -> int:
    """A single request's frames, or a chain's longest clip (the per-clip rows line, COMFORT_LIMIT_TABLE §17)."""
    return int(b["num_frames"]) if "num_frames" in b else max(int(c["num_frames"]) for c in b["clips"])


def rows_estimate(width: int, height: int, frames: int, *, text_tokens: int, keyframes=(), refs_rows: int = 0) -> dict:
    gh, gw = height // 32, width // 32
    video = gh * gw * latents(frames)
    cond = len(keyframes) * gh * gw
    audio = 2 * round(frames / 24 * 40)
    text = text_tokens + sum(gh * gw + 8 for k in keyframes if k in (0, frames - 1))
    return {"video": video, "cond": cond, "audio": audio, "text": text, "ref": refs_rows,
            "total": video + cond + audio + text + refs_rows}


def prompt_tokens(text: str) -> int | None:
    try:
        sys.path.insert(0, str(REPO))
        from services.engines.h3 import adapter as A  # noqa: PLC0415

        return A.count_prompt_tokens([text], assets=lambda k: REPO / "models" / descriptor()["assets"][k],
                                     text_encoder=lambda: None)[0]
    except Exception:  # noqa: BLE001
        return None


def descriptor() -> dict:
    return read_json(REPO / "scripts" / "manifests" / "40-minimaxh3.json")


def quant_class(file: str) -> str | None:
    try:
        sys.path.insert(0, str(REPO))
        import h3_format  # noqa: PLC0415

        return h3_format.quant_class(h3_format.inspect_h3_dit(str(WEIGHTS / file))[1])
    except Exception:  # noqa: BLE001
        return None


def comfort_budget(qc: str | None, comfort: dict | None = None) -> int | None:
    c = comfort if comfort is not None else (descriptor().get("comfort") or {})
    return (c.get("single_rows_budget_by_quant_class") or {}).get(qc or "", c.get("single_rows_budget"))


# ---------------------------------------------------------------------------------------------
# The items
# ---------------------------------------------------------------------------------------------


def base_body(**kw) -> dict:
    b = {"prompt": CITY, "width": 1280, "height": 768, "num_frames": 73, "frame_rate": 24.0, "seed": 1234,
         "attention_backend": "sage", "vram16_mode": True, "block_swap_prefetch": True, "keep_resident": False}
    b.update(kw)
    return b


def off20(**kw) -> dict:
    return base_body(num_inference_steps=20, shift_video=12.0, shift_audio=3.0, **kw)


def pi(choice: str, **kw) -> dict:
    # steps / shifts left out: the server fills the choice's values (adapter.prefill_request)
    return base_body(pi_h3=choice, **kw)


def turbo(**kw) -> dict:
    return base_body(num_inference_steps=4, shift_video=12.0, shift_audio=3.0, **kw)


def job(name: str, file: str, body: dict, *, uploads: dict | None = None, lora=None, compare_to=None,
        timeout: int = DEFAULT_JOB_TIMEOUT, fallback: dict | None = None, endpoint: str = "/generate",
        expect: dict | None = None) -> dict:
    """``body`` may hold "@image:<path>" / "@video:<path>" / "@audio:<path>" strings anywhere: each is
    uploaded once and replaced by its id (the chain's source_video / retake / clips[].source_audio, the
    outpaint / inpaint sources). ``expect``: {"width", "height", "frames"} of the output when it is not
    the body's own (a chain)."""
    return {"name": name, "file": file, "body": body, "uploads": uploads or {}, "lora": lora,
            "compare_to": compare_to, "timeout": timeout, "fallback": fallback, "endpoint": endpoint,
            "expect": expect}


def r1_record() -> dict | None:
    s = read_json(MIXED_SUMMARY)
    return get(s, "conds", "R1", "post", "stream_md5")


def ref_spike(item: str) -> dict:
    return {"label": f"spike {item}", "mp4": SPIKE / "runs" / item / "output.mp4"}


def ref_f1() -> dict:
    """The standalone off run: this script's F1 when present, else the 2026-11-22 gate's R1 record."""
    mp4 = RUNS / "F1" / "output.mp4"
    if mp4.is_file():
        return {"label": "F1 (this script)", "mp4": mp4}
    return {"label": "2026-11-22 gate R1 (record)", "mp4": MIXED_R1_MP4 if MIXED_R1_MP4.is_file() else None,
            "md5": r1_record()}


def ref_this_or_spike(item: str) -> dict:
    mp4 = RUNS / item / "output.mp4"
    return {"label": f"{item} (this script)", "mp4": mp4} if mp4.is_file() else ref_spike(item)


def items() -> dict[str, dict]:
    """Canonical order. ``default`` False = run only when named in --only."""
    I: dict[str, dict] = {}
    I["F0"] = {"group": 1, "kind": "selftest", "default": True,
               "title": "基準の出力の照合値（ワーカーの自己試験）",
               "rule": f"終了コード 0 で、mp4 の sha256 が {FINGERPRINT[:8]}… と一致"}
    I["F1"] = {"group": 1, "kind": "api", "default": True,
               "title": "API の off の単発（2026-11-22 の門の R1 と同じ条件）",
               "rule": "完了し、映像と音声のストリームの MD5 が R1 の記録と一致。記録に pi-H3 の鍵が無い",
               "jobs": [job("F1", FL2VA, off20(), compare_to={"label": "2026-11-22 gate R1 (record)",
                                                              "mp4": MIXED_R1_MP4, "md5": r1_record()})]}
    I["M1"] = {"group": 1, "kind": "engine_direct", "default": True, "source": "I1c",
               "title": "マスクのある off（昨日の I1c をエンジン直呼びで回し直す）",
               "rule": "完了し、映像と音声のストリームの MD5 が昨日の I1c の output.mp4 と一致"}
    I["M2"] = {"group": 1, "kind": "engine_direct", "default": False, "source": "O1c",
               "title": "マスクのある off（昨日の O1c。任意）",
               "rule": "完了し、映像と音声のストリームの MD5 が昨日の O1c の output.mp4 と一致"}
    for it, file, choice in (("P1", FL2VA, "4steps"), ("P2", FL2VA, "2steps"), ("P3", SULPHUR, "4steps"),
                             ("P4", SULPHUR, "2steps")):
        I[it] = {"group": 2, "kind": "api", "default": True, "pi": choice,
                 "title": f"実験の {it} の再現（{'FL2VA' if file == FL2VA else 'Sulphur-3'}・{choice}）",
                 "rule": "一致＝ストリームの MD5 が実験と一致。違えば「差あり」で数値。"
                         f"破綻＝フレームの PSNR の最小が {BROKEN_PSNR_DB:g} dB 未満か、フレーム数・音声が違う。"
                         "あわせて記録（pi_h3_used・h3.pi_h3）が揃うこと",
                 "jobs": [job(it, file, pi(choice), compare_to=ref_spike(it))]}
    I["R"] = {"group": 3, "kind": "api", "default": True, "resident": True,
              "title": "常駐 on での切り替え（off → 4steps → 2steps → off）",
              "rule": "4 本とも完了し、それぞれストリームの MD5 が単独の実行と一致（off は F1、4steps は P1、2steps は P2）",
              "jobs": [job("R1_off", FL2VA, off20(keep_resident=True), compare_to="F1"),
                       job("R2_4steps", FL2VA, pi("4steps", keep_resident=True), compare_to="P1"),
                       job("R3_2steps", FL2VA, pi("2steps", keep_resident=True), compare_to="P2"),
                       job("R4_off", FL2VA, off20(keep_resident=True), compare_to="F1")]}
    I["L1"] = {"group": 4, "kind": "api", "default": True, "pi": "4steps", "large": True,
               "title": "大きな画角 1920×1088・124 フレーム（FL2VA・4steps）",
               "rule": "快適上限の線の内側で、完了し、フレーム数と音声が期待どおり。DWD の段の VRAM・時間・塊の数を記録",
               "jobs": [job("L1", FL2VA, pi("4steps", width=1920, height=1088, num_frames=124),
                            timeout=LARGE_JOB_TIMEOUT)]}
    I["L2"] = {"group": 4, "kind": "api", "default": True, "pi": "4steps", "large": True,
               "title": "大きな画角 2560×1472・73 フレーム（FL2VA・4steps）",
               "rule": "L1 と同じ",
               "jobs": [job("L2", FL2VA, pi("4steps", width=2560, height=1472, num_frames=73),
                            timeout=LARGE_JOB_TIMEOUT)]}
    I["L3"] = {"group": 4, "kind": "api", "default": True, "pi": "4steps", "large": True,
               "title": "大きな画角 1920×1088・158 フレーム（FL2VA・4steps。DWD を分けて通す経路）",
               "rule": "L1 と同じ（DWD の塊が 2 個以上になること）",
               "jobs": [job("L3", FL2VA, pi("4steps", width=1920, height=1088, num_frames=158),
                            timeout=LARGE_JOB_TIMEOUT)]}
    I["I2V1"] = {"group": 5, "kind": "api", "default": True, "pi": "4steps",
                 "title": "I2V（先頭の画像。2026-10-19 の門の素材と文）",
                 "rule": "完了し、フレーム数と音声が期待どおり。images_used 1・pi_h3_used 4steps",
                 "jobs": [job("I2V1", FL2VA, pi("4steps", prompt=RAINCOAT),
                              uploads={"images": [(I2V_FIRST, 0)]})]}
    I["I2V2"] = {"group": 5, "kind": "api", "default": True, "pi": "4steps",
                 "title": "先頭と末尾の画像（2026-10-19 の門の素材と文）",
                 "rule": "完了し、フレーム数と音声が期待どおり。images_used 2・pi_h3_used 4steps",
                 "jobs": [job("I2V2", FL2VA, pi("4steps", prompt=RAINCOAT),
                              uploads={"images": [(KF_FIRST, 0), (KF_LAST, 72)]})]}
    I["REF"] = {"group": 5, "kind": "api", "default": True, "pi": "4steps",
                "title": "Ref2VA（Ref2VA の重み・2026-10-28 の G-R1 の素材と条件）",
                "rule": "完了し、フレーム数と音声が期待どおり。references の件数 1・pi_h3_used 4steps",
                "jobs": [job("REF", REF2VA, pi("4steps", prompt=REF_PROMPT, width=1344, height=768, num_frames=124,
                                               seed=42), uploads={"references": [("picture", REF_PICTURE, 1)]})]}
    I["NEG"] = {"group": 5, "kind": "api", "default": True, "pi": "4steps",
                "title": "ネガティブプロンプトつき（VSF・P1 の条件）",
                "rule": "完了し、フレーム数と音声が期待どおり。h3.vsf が記録され pi_h3_used 4steps",
                "jobs": [job("NEG", FL2VA, pi("4steps", nag_enabled=True, neg_method="vsf",
                                              negative_prompt=NEGATIVE))]}
    I["V16OFF"] = {"group": 5, "kind": "api", "default": True, "pi": "4steps",
                   "title": "VRAM 16GB mode off（P1 の条件。メモリ不足なら 640×384・124 フレーム）",
                   "rule": "完了し、フレーム数と音声が期待どおり。vram16_mode_used off・pi_h3_used 4steps",
                   "jobs": [job("V16OFF", FL2VA, pi("4steps", vram16_mode=False),
                                fallback=pi("4steps", vram16_mode=False, width=640, height=384, num_frames=124))]}
    I["Q1"] = {"group": 5, "kind": "api", "default": True, "pi": "4steps",
               "title": "w4a8 の層を含む重み（任意。Turbo を融合した重み）",
               "rule": "完了し、フレーム数と音声が期待どおり（受付で断られたら、その文を記録）",
               "jobs": [job("Q1", W4A8, pi("4steps"))]}
    a2v = dict(prompt=A2V_PROMPT, width=896, height=1152, num_frames=90, seed=12345)
    up = {"images": [(A2V_IMAGE, 0)], "source_audio": (A2V_WAV, 0.5)}
    I["A2V"] = {"group": 6, "kind": "api", "default": True,
                "title": "A2V の音声の Strength 0.5（2026-10-25 の G3 の run B の素材と条件）",
                "rule": "2 本とも完了し、フレーム数と音声が期待どおり（目と耳の確認はオーナー）",
                "jobs": [job("A2V_pi", FL2VA, pi("4steps", **a2v), uploads=up),
                         job("A2V_turbo", FL2VA, turbo(**a2v), uploads=up, lora=(TURBO_LORA, 0.8))]}
    # Follow-up (2026-11-23, the coordinator's request after looking at the sheets): run only with --only.
    I["X1"] = {"group": 7, "kind": "api", "default": False, "pi": "4steps",
               "title": "切り分け: A2V_pi と同じ要求で音声なし（先頭の画像だけの I2V）",
               "rule": "完了し、フレーム数と音声が期待どおり",
               "jobs": [job("X1", FL2VA, pi("4steps", **a2v), uploads={"images": [(A2V_IMAGE, 0)]})]}
    I["X2"] = {"group": 7, "kind": "api", "default": False, "pi": "4steps",
               "title": "切り分け: A2V_pi と同じ要求で音声の Strength 1.0",
               "rule": "完了し、フレーム数と音声が期待どおり",
               "jobs": [job("X2", FL2VA, pi("4steps", **a2v),
                            uploads={"images": [(A2V_IMAGE, 0)], "source_audio": (A2V_WAV, 1.0)})]}
    I["X3"] = {"group": 7, "kind": "api", "default": False, "pi": "2steps",
               "title": "切り分け: A2V_pi と同じ要求（Strength 0.5）で 2steps",
               "rule": "完了し、フレーム数と音声が期待どおり",
               "jobs": [job("X3", FL2VA, pi("2steps", **a2v), uploads=up)]}
    G_R1 = REPO / "outputs" / "h3-ref2va-2026-10-28" / "G-R1.mp4"
    I["REFc"] = {"group": 7, "kind": "api", "default": False,
                 "title": "切り分け: REF と同じ要求を pi-H3 なしで（G-R1 の条件: 20 ステップ・sdpa）",
                 "rule": "完了し、フレーム数と音声が期待どおり。G-R1 の出力と比べる（記録）",
                 "jobs": [job("REFc", REF2VA, off20(prompt=REF_PROMPT, width=1344, height=768, num_frames=124, seed=42,
                                                    attention_backend="sdpa"),
                              uploads={"references": [("picture", REF_PICTURE, 1)]},
                              compare_to={"label": "2026-10-28 G-R1 (selftest output)", "mp4": G_R1})]}
    # Follow-up 2 (2026-11-23, after dwd.CHUNK_MAX_ROWS 65,536 -> 131,072 in the working tree): --only.
    I["P1b"] = {"group": 7, "kind": "api", "default": False, "pi": "4steps",
                "title": "P1 の回し直し（分割の塊の上限を変えたあと）",
                "rule": "ストリームの MD5 が実験の P1 と一致",
                "jobs": [job("P1b", FL2VA, pi("4steps"), compare_to=ref_spike("P1"))]}
    I["REFt"] = {"group": 7, "kind": "api", "default": False,
                 "title": "切り分け: REF と同じ要求を pi-H3 なし・Turbo LoRA 0.8 の 4 ステップで",
                 "rule": "完了し、フレーム数と音声が期待どおり",
                 "jobs": [job("REFt", REF2VA, turbo(prompt=REF_PROMPT, width=1344, height=768, num_frames=124, seed=42),
                              uploads={"references": [("picture", REF_PICTURE, 1)]}, lora=(TURBO_LORA, 0.8))]}
    for it, w, h, f, t in (("L3b", 1920, 1088, 158, LARGE_JOB_TIMEOUT), ("L4", 1920, 1088, 243, 2700),
                           ("L5", 2560, 1472, 158, 2700)):
        I[it] = {"group": 7, "kind": "api", "default": False, "pi": "4steps", "large": True,
                 "title": f"大きな画角 {w}×{h}・{f} フレーム（FL2VA・4steps。塊の上限を変えたあと）",
                 "rule": "快適上限の線の内側で、完了し、フレーム数と音声が期待どおり",
                 "jobs": [job(it, FL2VA, pi("4steps", width=w, height=h, num_frames=f), timeout=t)]}
    # Follow-up 3 (after the per-chunk cache release in dwd.py, working tree): --only.
    I["P1c"] = {"group": 7, "kind": "api", "default": False, "pi": "4steps",
                "title": "P1 の回し直し（塊ごとのキャッシュ解放のあと）",
                "rule": "ストリームの MD5 が実験の P1 と一致",
                "jobs": [job("P1c", FL2VA, pi("4steps"), compare_to=ref_spike("P1"))]}
    for it, src, w, h, f in (("L4b", "L4", 1920, 1088, 243), ("L5b", "L5", 2560, 1472, 158)):
        I[it] = {"group": 7, "kind": "api", "default": False, "pi": "4steps", "large": True,
                 "title": f"{src} の回し直し（塊ごとのキャッシュ解放のあと）",
                 "rule": f"完了し、ストリームの MD5 が前回の {src} と一致",
                 "jobs": [job(it, FL2VA, pi("4steps", width=w, height=h, num_frames=f), timeout=2700,
                              compare_to={"label": f"{src} (this script, before the cache release)",
                                          "mp4": RUNS / src / "output.mp4"})]}
    # Follow-up 4 (h3_pih3_format moved to the repo root, working tree): --only.
    I["F1d"] = {"group": 7, "kind": "api", "default": False,
                "title": "F1 の回し直し（照合のモジュールを移したあと）",
                "rule": "ストリームの MD5 が R1 の記録と一致し、記録に pi-H3 の鍵が無い",
                "jobs": [job("F1d", FL2VA, off20(), compare_to={"label": "2026-11-22 gate R1 (record)",
                                                               "mp4": MIXED_R1_MP4, "md5": r1_record()})]}
    for it, src, file, choice in (("P1d", "P1", FL2VA, "4steps"), ("P3d", "P3", SULPHUR, "4steps")):
        I[it] = {"group": 7, "kind": "api", "default": False, "pi": choice,
                 "title": f"{src} の回し直し（照合のモジュールを移したあと）",
                 "rule": f"ストリームの MD5 が実験の {src} と一致",
                 "jobs": [job(it, file, pi(choice), compare_to=ref_spike(src))]}
    I.update(stage2_items())
    return I


# ---------------------------------------------------------------------------------------------
# Stage 2 (chain / V2V / Retake / Inpainting / Outpainting). Run with --stage2 (or --only).
# ---------------------------------------------------------------------------------------------

GA0_DIR = REPO / "Docs" / "Outputs-archive" / "h3-chain-a2v-2026-11-09" / "ga0"
G_C0_SHA256 = "4a98be0884ae7129e34f58578a3fe7ca3117ca5a764545cbac404e12a70cbec5"
CHAIN_SAME_PROMPT = (
    "A young girl in a white summer dress and a straw hat walks along a sandy beach at the water's edge on a sunny "
    "day. Gentle waves wash ashore with the sound of surf, seagulls call in the distance, and her footsteps are soft "
    "on the wet sand. Steady tracking camera.")      # the spike's C3b prompt (both clips)
MAID = ("A young woman in a black-and-white maid costume talks cheerfully to the camera in a pastel room decorated "
        "with sweets and flowers; natural head and hand movement, steady camera.")      # the spike's R1
MAID_WIDE = ("A wide shot of a young woman in a black-and-white maid costume talking cheerfully to the camera in a "
             "spacious pastel room decorated with sweets and flowers; natural head and hand movement, steady camera.")
GIRLS = ("Four young girls walking toward the camera on a sunny beach. The girl on the far right has blonde hair and "
         "wears a red dress, red outfit. Steady camera.")      # the spike's I1 (= the 2026-11-11 gate's)
COSPLAY = "コスプレイヤーの日本人女性が、こちらに向かって歩き続ける。彼女は片手を伸ばしたまま、こちらに微笑みかける。"
RETAKE_SRC = REPO / "uploads" / "h3-ref-calibration" / "H3-Ref-1080p-15s.mp4"
INPAINT_SRC = REPO / "uploads" / "h3-inpaint-development" / "inpaint-base.mp4"
INPAINT_MASK = REPO / "uploads" / "h3-inpaint-development" / "mask-768p.mp4"
OUTPAINT_SRC = SPIKE / "_material" / "O1_source_1024x576_124f.mp4"
V2V_SRC = REPO / "outputs" / "0b57ea7d-b494-43ea-a504-2a5637380581" / "output.mp4"
CLIP_AUDIO = REPO / "uploads" / "h3-a2v-development" / "ref-voice-patra001.wav"
LONG_SEAMS = (124, 226, 328, 430)


def gc0_record() -> dict:
    """The 2026-11-09 GA0 run of the 2026-10-29 G-C0 chain: its sent body, mp4 sha256 and clip latent hashes."""
    rq = read_json(GA0_DIR / "request.json").get("body") or {}
    res = read_json(GA0_DIR / "result.json")
    lat = [c.get("latents_sha256") for c in get(res, "metadata_excerpt", "h3_clips", default=[]) or []]
    return {"body": rq, "sha256": res.get("sha256") or G_C0_SHA256, "latents": lat}


def chain_job(name, body, *, frames, width, height, **kw):
    return job(name, FL2VA, body, endpoint="/generate/chain", expect={"frames": frames, "width": width,
                                                                       "height": height}, **kw)


def stage2_items() -> dict[str, dict]:
    I: dict[str, dict] = {}
    gc0 = gc0_record()
    I["C0"] = {"group": 1, "kind": "api", "default": False, "stage": 2, "gc0": True,
               "title": "段 2: off の連結（2026-10-29 の G-C0。2026-11-09 の GA0 と同じ本文）",
               "rule": f"完了し、mp4 の sha256 が G-C0 の {G_C0_SHA256[:8]}… と一致し、3 クリップの潜在の照合値が GA0 の記録と一致",
               "jobs": [chain_job("C0", dict(gc0["body"]), frames=328, width=640, height=384)]}
    I["M3"] = {"group": 1, "kind": "engine_direct", "default": False, "stage": 2, "source": "R1c",
               "title": "段 2: off の Retake（昨日の R1c をエンジン直呼びで回し直す）",
               "rule": "完了し、映像と音声のストリームの MD5 が昨日の R1c の output.mp4 と一致"}
    for it, src in (("E3b4", "C3b_4step"), ("E3b2", "C3b_2step"), ("ER1", "R1"), ("EI1", "I1"), ("EO1", "O1")):
        I[it] = {"group": 2, "kind": "engine_direct", "default": False, "stage": 2, "source": src, "pi": True,
                 "title": f"段 2: 実験の {src} の再現（エンジン直呼び・製品の pi-H3 の経路）",
                 "rule": "一致＝ストリームの MD5 が実験と一致。違えば「差あり」で数値。"
                         f"破綻＝フレームの PSNR の最小が {BROKEN_PSNR_DB:g} dB 未満か、フレーム数が違う"}
    common = {"attention_backend": "sage", "vram16_mode": True, "block_swap_prefetch": True, "keep_resident": False}
    I["ACH"] = {"group": 8, "kind": "api", "default": False, "stage": 2, "pi": "4steps",
                "title": "段 2: API の連結（実験の C3b と同じ条件・4steps）",
                "rule": "完了し、226 フレームと音声。記録（pi_h3_used・h3.pi_h3・クリップごとの pi_h3・DWD の塊）",
                "jobs": [chain_job("ACH", {"prompt": CHAIN_SAME_PROMPT, "width": 640, "height": 384, "seed": 1234,
                                           "clips": [{"num_frames": 124}, {"num_frames": 124}], "pi_h3": "4steps",
                                           **common}, frames=226, width=640, height=384)]}
    I["AV2V"] = {"group": 8, "kind": "api", "default": False, "stage": 2, "pi": "4steps",
                 "title": "段 2: API の V2V（2026-11-02 の門 ① の素材と文・56＋39・4steps）",
                 "rule": "完了し、51 フレームと音声。記録が揃う",
                 "jobs": [chain_job("AV2V", {"prompt": COSPLAY, "width": 640, "height": 384, "seed": 1234,
                                             "clips": [{"num_frames": 56}, {"num_frames": 39}],
                                             "source_video": {"video_id": f"@video:{V2V_SRC}"}, "pi_h3": "4steps",
                                             **common}, frames=51, width=640, height=384)]}
    rt = lambda s, **kw: {"prompt": MAID, "width": 1280, "height": 768, "seed": 1234,  # noqa: E731
                          "clips": [{"num_frames": 124}], **common, **kw,
                          "retake": {"video_id": f"@video:{RETAKE_SRC}", "window_start_sec": 4.0, "audio_strength": s}}
    I["ART"] = {"group": 8, "kind": "api", "default": False, "stage": 2, "pi": "4steps",
                "title": "段 2: API の Retake（実験の R1 と同じ素材・文・窓・Strength 1.0・4steps）",
                "rule": "完了し、124 フレームと音声。記録が揃う",
                "jobs": [chain_job("ART", rt(1.0, pi_h3="4steps"), frames=124, width=1280, height=768)]}
    I["AIN"] = {"group": 8, "kind": "api", "default": False, "stage": 2, "pi": "4steps",
                "title": "段 2: API の Inpainting（実験の I1 と同じ素材・文・Strength 1.0・4steps）",
                "rule": "完了し、73 フレームと音声。h3.inpaint と pi-H3 の記録が揃う",
                "jobs": [job("AIN", FL2VA, pi("4steps", prompt=GIRLS, inpaint={
                    "source_video_id": f"@video:{INPAINT_SRC}", "mask_video_id": f"@video:{INPAINT_MASK}",
                    "window_start_sec": 0.0, "audio_strength": 1.0}))]}
    I["AOUT"] = {"group": 8, "kind": "api", "default": False, "stage": 2, "pi": "4steps",
                 "title": "段 2: API の Outpainting（実験の O1 の切り出し・文・左右 128／上下 96・Strength 1.0・4steps）",
                 "rule": "完了し、124 フレームと音声。h3.outpaint と pi-H3 の記録が揃う",
                 "jobs": [job("AOUT", FL2VA, pi("4steps", prompt=MAID_WIDE, num_frames=124, outpaint={
                     "source_video_id": f"@video:{OUTPAINT_SRC}", "pad_left": 128, "pad_right": 128,
                     "pad_top": 96, "pad_bottom": 96, "audio_strength": 1.0}))]}
    au = f"@audio:{CLIP_AUDIO}"
    I["AAUD"] = {"group": 8, "kind": "api", "default": False, "stage": 2, "pi": "4steps",
                 "title": "段 2: API のクリップごとに音声を指定した連結（2026-11-09 の GA2 の本文・4steps）",
                 "rule": "完了し、345 フレームと音声。クリップごとの a2v と pi-H3 の記録",
                 "jobs": [chain_job("AAUD", {
                     "prompt": RAINCOAT, "width": 640, "height": 384, "seed": 42, "pi_h3": "4steps", **common,
                     "clips": [{"num_frames": 124, "source_audio": {"audio_id": au, "strength": 1.0, "offset_frames": 0}},
                               {"num_frames": 141, "prompt": "She turns and walks slowly along the street, humming a tune.",
                                "source_audio": {"audio_id": au, "strength": 1.0, "offset_frames": 102}},
                               {"num_frames": 124, "prompt": "She stops under a shop awning, looks up and says <d>[English] "
                                "Maybe I should go home now.</d>",
                                "source_audio": {"audio_id": au, "strength": 1.0, "offset_frames": 221}}]},
                     frames=345, width=640, height=384)]}
    I["R05"] = {"group": 8, "kind": "api", "default": False, "stage": 2,
                "title": "段 2: 音声の Strength 0.5 の Retake（R1 の条件・pi-H3 4steps と Turbo LoRA 0.8 の 4 ステップ）",
                "rule": "2 本とも完了し、124 フレームと音声（目と耳の確認はオーナー）",
                "jobs": [chain_job("R05_pi", rt(0.5, pi_h3="4steps"), frames=124, width=1280, height=768),
                         chain_job("R05_turbo", rt(0.5, num_inference_steps=4, shift_video=12.0, shift_audio=3.0),
                                   frames=124, width=1280, height=768, lora=(TURBO_LORA, 0.8))]}
    I["LONG"] = {"group": 8, "kind": "api", "default": False, "stage": 2, "pi": "4steps", "large": True,
                 "title": "段 2: 長い連結（5 本×124 フレーム・1280×768・全クリップ同じ文・4steps）",
                 "rule": "完了し、532 フレームと音声。主記憶と VRAM の山、復号の段、DWD の塊、クリップごとの行数を記録",
                 "jobs": [chain_job("LONG", {"prompt": CHAIN_SAME_PROMPT, "width": 1280, "height": 768, "seed": 1234,
                                             "clips": [{"num_frames": 124} for _ in range(5)], "pi_h3": "4steps",
                                             **common}, frames=532, width=1280, height=768, timeout=3600)]}
    return I


STAGE2_ORDER = ["C0", "M3", "E3b4", "E3b2", "ER1", "EI1", "EO1", "ACH", "AV2V", "ART", "AIN", "AOUT", "AAUD",
                "R05", "LONG"]


def gc0_compare(mp4: Path, p: dict) -> dict:
    rec = gc0_record()
    got = [c.get("latents_sha256") for c in p.get("chain_clips") or []]
    out = {"label": "2026-10-29 G-C0 / 2026-11-09 GA0 (record)", "sha256": sha256(mp4), "expect_sha256": rec["sha256"],
           "latents": got, "expect_latents": rec["latents"]}
    out["sha256_equal"] = out["sha256"] == rec["sha256"]
    out["latents_equal"] = bool(rec["latents"]) and got == rec["latents"]
    out["equal"] = out["video_equal"] = out["audio_equal"] = out["sha256_equal"] and out["latents_equal"]
    return out


ITEMS = items()
DEFAULT_ORDER = [k for k, v in ITEMS.items() if v["default"]]

# ---------------------------------------------------------------------------------------------
# The gate (port 18620, leftover workers, GPU). Counts only: no process path is recorded.
# ---------------------------------------------------------------------------------------------


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
    return [{"pid": d.get("ProcessId"), "ppid": d.get("ParentProcessId"), "cmdline": d.get("CommandLine") or ""}
            for d in data]


def descendants(root_pid: int, procs: list[dict]) -> list[dict]:
    out, seen, frontier = [], {root_pid}, {root_pid}
    while frontier:
        kids = [p for p in procs if p["ppid"] in frontier and p["pid"] not in seen]
        for p in kids:
            seen.add(p["pid"])
            out.append(p)
        frontier = {p["pid"] for p in kids}
    return out


def gpu_judge(app_lines: list[str], mem_line: str, my_pid: int, rc: tuple[int, int] = (0, 0)) -> dict:
    """Pure: nvidia-smi's compute-app lines and the memory line -> the record (counts only) + blocking reasons."""
    apps = [a.strip() for a in app_lines if a.strip()]
    others = [a for a in apps if a.split(",", 1)[0].strip() != str(my_pid)
              and any(w in a.lower() for w in ("python", "uvicorn"))]
    parts = [x.strip() for x in (mem_line or ",,").split(",")]
    try:
        used, total = int(parts[1]), int(parts[2])
    except (IndexError, ValueError):
        used, total = -1, -1
    rec = {"name": parts[0] if parts else None, "memory_used_mib": used, "memory_total_mib": total,
           "compute_apps_count": len(apps), "python_like_count": len(others), "returncodes": list(rc),
           "bound_mib": GPU_BUSY_MIB, "blocking": []}
    if rc != (0, 0) or used < 0:
        rec["blocking"].append(f"nvidia-smi が失敗（{rc[0]}/{rc[1]}）")
    if others:
        rec["blocking"].append(f"GPU を別の python／uvicorn が使っている（{len(others)} 件）")
    if used > GPU_BUSY_MIB:
        rec["blocking"].append(f"GPU の使用量 {used} MiB が {GPU_BUSY_MIB} MiB を超えている")
    return rec


def gpu_busy_check() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"blocking": ["nvidia-smi が見つからない"]}
    apps = run_cmd([exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], 60)
    mem = run_cmd([exe, "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"], 60)
    return gpu_judge((apps.stdout or "").splitlines(), ((mem.stdout or "").splitlines() or [""])[0], os.getpid(),
                     (apps.returncode, mem.returncode))


def preflight_checks() -> dict:
    rec: dict[str, Any] = {"at": now_iso(), "blocking": []}
    listening = port_listening(OWNER_PORT)
    rec["owner_server"] = {"port": OWNER_PORT, "listening": listening}
    if listening:
        rec["blocking"].append(f"ポート {OWNER_PORT} で何かが待ち受けている")
    rec["gpu"] = gpu_busy_check()
    rec["blocking"] += rec["gpu"]["blocking"]
    workers = [p["pid"] for p in all_processes() if WORKER_RE.search(p["cmdline"] or "")]
    rec["engine_workers"] = {"count": len(workers), "pids": workers}
    if workers:
        rec["blocking"].append(f"エンジンのワーカーのプロセスが残っている（{len(workers)} 件）")
    return rec


def log_gate(rec: dict) -> None:
    g = rec["gpu"]
    log(f"gate: port {OWNER_PORT} {'LISTENING' if rec['owner_server']['listening'] else 'free'}; "
        f"GPU {g.get('name')} used {g.get('memory_used_mib')} MiB (bound {GPU_BUSY_MIB}), compute apps "
        f"{g.get('compute_apps_count')}, python/uvicorn {g.get('python_like_count')}; "
        f"leftover workers {rec['engine_workers']['count']}")
    for b in rec["blocking"]:
        log(f"gate refuses: {b}")


# ---------------------------------------------------------------------------------------------
# The backend (temp yaml + temp state under _rt/, own port; stopped in finally)
# ---------------------------------------------------------------------------------------------


def write_temp_config(port: int) -> Path:
    import yaml  # noqa: PLC0415

    for sub in ("logs", "outputs", "uploads"):
        (RT / sub).mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = port
    raw["server"]["log_dir"] = str(RT / "logs")
    raw.setdefault("output", {})["dir"] = str(RT / "outputs")
    raw.setdefault("upload", {})["dir"] = str(RT / "uploads")
    raw["state_file"] = str(RT / TEMP_STATE_NAME)
    cfg = RT / TEMP_CONFIG_NAME
    cfg.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    selections: dict = {}
    current = read_json(REPO / "state.json")
    if isinstance(current.get("selections"), dict):
        selections = {k: dict(v) for k, v in current["selections"].items() if isinstance(v, dict)}
    for base in ("LTX23", "LTX25"):
        selections.setdefault(base, {c: "default" for c in H3_CATEGORIES})
    selections[H3] = {c: "default" for c in H3_CATEGORIES}
    (RT / TEMP_STATE_NAME).write_text(json.dumps({"schema": 1, "active_base_model": H3, "selections": selections},
                                                 indent=2), encoding="utf-8", newline="\n")
    return cfg


def api_key() -> str | None:
    try:
        import yaml  # noqa: PLC0415

        return (yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8")).get("server") or {}).get("api_key")
    except Exception:  # noqa: BLE001
        return None


def http(port: int, method: str, path: str, body=None, timeout: float = 60.0, raw: bytes | None = None,
         ctype: str = "application/json"):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode("utf-8"))
    headers = {"Content-Type": ctype} if data is not None else {}
    key = api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            txt = resp.read().decode("utf-8")
            return resp.status, (json.loads(txt) if txt else None)
    except urllib.error.HTTPError as exc:
        txt = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(txt)
        except Exception:  # noqa: BLE001
            return exc.code, {"raw": txt[:2000]}


def upload(port: int, endpoint: str, path: Path) -> dict:
    boundary = uuid.uuid4().hex
    body = (b"--" + boundary.encode() + b"\r\nContent-Disposition: form-data; name=\"file\"; filename=\""
            + path.name.encode("utf-8") + b"\"\r\nContent-Type: application/octet-stream\r\n\r\n" + path.read_bytes()
            + b"\r\n--" + boundary.encode() + b"--\r\n")
    st, r = http(port, "POST", endpoint, raw=body, ctype=f"multipart/form-data; boundary={boundary}", timeout=300)
    if st != 200:
        raise RuntimeError(f"upload {path.name} -> {endpoint}: {st} {r}")
    return r


class Server:
    def __init__(self, port: int, tag: str) -> None:
        self.port, self.tag = port, tag
        self.proc: subprocess.Popen | None = None
        self.fh = None
        self.record: dict[str, Any] = {"port": port}

    def alive(self) -> bool:
        try:
            return http(self.port, "GET", "/status", timeout=5)[0] == 200
        except Exception:  # noqa: BLE001
            return False

    def start(self) -> None:
        if port_listening(self.port):
            raise RuntimeError(f"port {self.port} is already in use; pick another with --port")
        cfg = write_temp_config(self.port)
        env = dict(os.environ)
        prefix = "".join(f"{d};" for d in (REPO / "tools" / "uv", REPO / "tools" / "ffmpeg" / "bin") if d.exists())
        env.update(PATH=prefix + env.get("PATH", ""), UV_PYTHON_INSTALL_DIR=str(REPO / ".python"),
                   PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.fh = (RT / "server.console.log").open("a", encoding="utf-8", errors="replace", newline="\n")
        self.fh.write(f"\n===== {now_iso()} h3_pih3_gates.py [{self.tag}] =====\n")
        self.fh.flush()
        self.proc = subprocess.Popen([str(PY_APP), "-u", str(REPO / "main.py"), "--config", str(cfg)], cwd=str(REPO),
                                     stdin=subprocess.DEVNULL, stdout=self.fh, stderr=subprocess.STDOUT, env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.record.update(pid=self.proc.pid, config=rel(cfg), started_at=now_iso())
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.alive():
                log(f"backend up (pid {self.proc.pid}, port {self.port})")
                return
            if self.proc.poll() is not None:
                raise RuntimeError("backend exited while starting (see _rt/server.console.log)")
            time.sleep(1.0)
        raise RuntimeError("/api/v1/status did not answer within 180 s")

    def stop(self) -> None:
        """unload -> terminate -> taskkill /T /F; then kill any descendant (worker) left."""
        if self.proc is None:
            return
        rec: dict[str, Any] = {"at": now_iso()}
        pid = self.proc.pid
        kids = descendants(pid, all_processes())
        try:
            rec["unload_http"] = http(self.port, "POST", "/pipeline/unload", timeout=600)[0]
        except Exception as exc:  # noqa: BLE001
            rec["unload_http"] = f"{type(exc).__name__}: {exc}"
        try:
            self.proc.terminate()
            self.proc.wait(timeout=60)
            rec["terminate"] = "ok"
        except Exception as exc:  # noqa: BLE001
            rec["terminate"] = f"{type(exc).__name__}: {exc}"
            run_cmd(["taskkill", "/PID", str(pid), "/T", "/F"])
        self.proc = None
        alive = {p["pid"] for p in all_processes()}
        left = [p["pid"] for p in kids if p["pid"] in alive]
        for p in left:
            run_cmd(["taskkill", "/PID", str(p), "/T", "/F"])
        rec["descendants_killed_after"] = left
        if self.fh is not None:
            self.fh.close()
            self.fh = None
        self.record["stop"] = rec
        log(f"backend stopped (pid {pid}; descendants killed afterwards {left})")


def file_offset(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def read_since(p: Path, off: int) -> str:
    if not p.exists():
        return ""
    with p.open("rb") as fh:
        fh.seek(off)
        return fh.read().decode("utf-8", errors="replace")


def resolve_transformer(port: int, file: str) -> str | None:
    st, models = http(port, "GET", "/models", timeout=120)
    for e in get(models, "categories", "transformer", "entries", default=[]) or []:
        if Path(str(e.get("path") or "")).name == file and e.get("exists", True):
            return e.get("name")
    return None


def load_weight(port: int, file: str, timeout_sec: float = 1800) -> dict:
    rec: dict[str, Any] = {"file": file}
    name = resolve_transformer(port, file)
    rec["name"] = name
    if name is None:
        rec["error"] = f"{file} is not in GET /models transformer entries"
        return rec
    t0 = time.monotonic()
    http(port, "POST", "/pipeline/unload", timeout=600)
    st, body = http(port, "POST", "/pipeline/load", {"base_model": H3, "models": {"transformer": name}},
                    timeout=timeout_sec)
    rec["load_http"] = st
    if st not in (200, 409):
        rec["error"] = f"load {st}: {body}"
        return rec
    while time.monotonic() - t0 < timeout_sec:
        _, s = http(port, "GET", "/status", timeout=30)
        if s and s.get("pipeline_loaded") and s.get("base_model") == H3 and s.get("state") in ("ready", "running"):
            rec.update(ok=True, seconds=round(time.monotonic() - t0, 1))
            return rec
        if s and s.get("state") == "error":
            rec["error"] = f"pipeline state error: {s.get('state')}"
            return rec
        time.sleep(5)
    rec["error"] = f"not loaded after {timeout_sec} s"
    return rec


def config_comfort(port: int) -> dict:
    st, cfg = http(port, "GET", "/config", timeout=60)
    return get(cfg, "limits", "comfort_budgets", "h3", default={}) or {}


def active_quant_class(port: int) -> str | None:
    st, models = http(port, "GET", "/models", timeout=120)
    for b in (models or {}).get("base_models") or []:
        if b.get("id") == H3:
            return b.get("transformer_quant_class")
    return None


def build_body(port: int, jb: dict, body: dict, cache: dict) -> dict:
    b = dict(body)
    up = jb["uploads"]

    def once(path: Path, endpoint: str, key: str) -> str:
        k = (str(path), endpoint)
        if k not in cache:
            cache[k] = upload(port, endpoint, path)[key]
        return cache[k]

    if up.get("images"):
        b["conditioning_images"] = [{"image_id": once(p, "/upload/image", "image_id"), "frame_idx": f}
                                    for p, f in up["images"]]
    if up.get("source_audio"):
        p, s = up["source_audio"]
        b["source_audio"] = {"audio_id": once(p, "/upload/audio", "audio_id"), "strength": s}
    if up.get("references"):
        endpoint = {"picture": ("/upload/image", "image_id"), "video": ("/upload/video", "video_id"),
                    "audio": ("/upload/audio", "audio_id")}
        b["references"] = [{"kind": k, "id": once(p, *endpoint[k]), "slot": s} for k, p, s in up["references"]]
    if jb["lora"]:
        b["loras"] = [{"name": Path(jb["lora"][0]).stem, "strength": jb["lora"][1]}]
    stores = {"@image:": ("/upload/image", "image_id"), "@video:": ("/upload/video", "video_id"),
              "@audio:": ("/upload/audio", "audio_id")}

    def sub(o):
        if isinstance(o, str):
            for pre, (ep, key) in stores.items():
                if o.startswith(pre):
                    return once(Path(o[len(pre):]), ep, key)
            return o
        if isinstance(o, dict):
            return {k: sub(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sub(v) for v in o]
        return o

    return sub(b)


def placeholder_paths(o) -> list[Path]:
    if isinstance(o, str):
        return [Path(o.split(":", 1)[1])] if o.startswith(("@image:", "@video:", "@audio:")) else []
    if isinstance(o, dict):
        return [p for v in o.values() for p in placeholder_paths(v)]
    if isinstance(o, list):
        return [p for v in o for p in placeholder_paths(v)]
    return []


def run_api_job(port: int, item: str, jb: dict, cache: dict, out_dir: Path, body: dict | None = None) -> dict:
    """POST /generate, poll GET /jobs/{id}, copy output.mp4 + metadata.json, slice the logs."""
    body = build_body(port, jb, body or jb["body"], cache)
    rec: dict[str, Any] = {"job": jb["name"], "file": jb["file"], "request": body, "started_at": now_iso()}
    wk_log, srv_log = RT / "logs" / "h3_worker.log", RT / "logs" / "server.log"
    wk_off, srv_off = file_offset(wk_log), file_offset(srv_log)
    t0 = time.monotonic()
    st, resp = http(port, "POST", jb.get("endpoint") or "/generate", body, timeout=300)
    rec["submit_http"] = st
    if st != 202:
        rec.update(status="refused", error=resp, finished_at=now_iso())
        log(f"  {jb['name']}: refused {st}: {json.dumps(resp, ensure_ascii=False)[:400]}")
        return rec
    job_id = resp["job_id"]
    rec["job_id"] = job_id
    last_key, stages = None, []
    while True:
        _, s = http(port, "GET", f"/jobs/{job_id}", timeout=60)
        s = s or {}
        key = (s.get("status"), s.get("stage"), s.get("current_step"))
        if key != last_key:
            stages.append({"t": round(time.monotonic() - t0, 1), "status": s.get("status"), "stage": s.get("stage"),
                           "step": s.get("current_step"), "total": s.get("total_steps")})
            log(f"  {jb['name']} {job_id[:8]} status={s.get('status')} stage={s.get('stage')} "
                f"step={s.get('current_step')}/{s.get('total_steps')}")
            last_key = key
        if s.get("status") in ("completed", "failed", "cancelled"):
            break
        if time.monotonic() - t0 > jb["timeout"]:
            rec["timed_out_after_sec"] = jb["timeout"]
            break
        time.sleep(5)
    rec["seconds_wall"] = round(time.monotonic() - t0, 1)
    rec["stages"] = stages
    rec["status"] = "timeout" if rec.get("timed_out_after_sec") else s.get("status")
    rec["error"] = s.get("error")
    rec["last_stage"] = stages[-1]["stage"] if stages else None
    if rec["status"] == "failed" and len(stages) > 1:
        rec["failed_stage"] = next((x["stage"] for x in reversed(stages) if x["stage"]), None)
    wk, srv = read_since(wk_log, wk_off), read_since(srv_log, srv_off)
    write_text(out_dir / f"{jb['name']}.worker_log.txt", wk)
    write_text(out_dir / f"{jb['name']}.server_log.txt", srv)
    rec["worker_generate_line"] = next((ln for ln in wk.splitlines() if "] generate:" in ln), None)
    src = RT / "outputs" / job_id
    if rec["status"] == "completed":
        mp4 = out_dir / ("output.mp4" if jb["name"] == item else f"{jb['name']}.mp4")
        shutil.copy2(src / "output.mp4", mp4)
        rec["mp4"] = rel(mp4)
        if (src / "metadata.json").is_file():
            write_text(out_dir / f"{jb['name']}.metadata.json", (src / "metadata.json").read_text(encoding="utf-8"))
            rec["metadata_file"] = rel(out_dir / f"{jb['name']}.metadata.json")
    rec["finished_at"] = now_iso()
    log(f"  {jb['name']}: {rec['status']} in {rec['seconds_wall']} s" + (f" ({rec['error']})" if rec["error"] else ""))
    return rec


# ---------------------------------------------------------------------------------------------
# Picks and checks
# ---------------------------------------------------------------------------------------------


def picks(meta: dict) -> dict:
    h3 = meta.get("h3") or {}
    ph = h3.get("phases") or {}
    den, dec = ph.get("denoise") or {}, ph.get("decode") or {}
    clip_phases = ph.get("clips") or []
    if clip_phases and not den:
        # a chain: phases = {clips: [{encode, denoise, ...}], decode}; the denoise peak is the clips' largest
        dens = [c.get("denoise") or {} for c in clip_phases]
        den = {"seconds": round(sum(d.get("seconds") or 0 for d in dens), 2),
               "peak_vram_mb": max((d.get("peak_vram_mb") or 0) for d in dens),
               "peak_vram_reserved_mb": max((d.get("peak_vram_reserved_mb") or 0) for d in dens),
               "vram16": dens[0].get("vram16") or {}, "sequence": dens[0].get("sequence"),
               "per_clip_seconds": [d.get("seconds") for d in dens]}
    clips_rec = h3.get("clips") or (meta.get("chain") or {}).get("clips") or []
    dwd = dec.get("dwd") or {}
    return {
        "transformer": get(meta, "models", "selection", "transformer", "file"),
        "generation_time_seconds": meta.get("generation_time_seconds"),
        "steps_shift": [h3.get("num_inference_steps"), h3.get("shift_video"), h3.get("shift_audio")],
        "pi_h3_used": meta.get("pi_h3_used"), "keep_resident_used": meta.get("keep_resident_used"),
        "vram16_mode_used": meta.get("vram16_mode_used"), "attention_used": meta.get("attention_used"),
        "images_used": h3.get("images_used"), "references": h3.get("references") or meta.get("references"),
        "seconds": {"encode": get(ph, "encode", "seconds"), "denoise": den.get("seconds"),
                    "decode": dec.get("seconds"), "dwd": dwd.get("seconds")},
        "vram_peak_mib": {"job": [meta.get("vram_optimization", {}).get("peak_vram_mb"), meta.get("peak_vram_reserved_mb")],
                          "denoise": [den.get("peak_vram_mb"), den.get("peak_vram_reserved_mb")],
                          "decode": [dec.get("peak_vram_mb"), dec.get("peak_vram_reserved_mb")]},
        "rss_peak_gib": h3.get("rss_peak_gib"), "commit_peak_gib": h3.get("commit_peak_gib"),
        "sequence": den.get("sequence"), "vram16": {k: get(den, "vram16", k) for k in ("cap_mib", "num_ooms")},
        "vae_source": dec.get("vae_source"),
        "dwd_chunks": [{k: c.get(k) for k in ("keep_blocks", "context_blocks", "latents", "frames", "seconds")}
                       for c in dwd.get("chunks") or []],
        "dwd_head": dwd.get("head"),
        "pi_h3": h3.get("pi_h3"), "a2v": h3.get("a2v") or meta.get("a2v"), "vsf": h3.get("vsf"),
        "loras": h3.get("loras"),
        "per_clip_denoise_seconds": den.get("per_clip_seconds"),
        "chain_clips": [{"latents_sha256": c.get("latents_sha256"), "head_equals_prev_tail": c.get("head_equals_prev_tail"),
                         "pi_h3": c.get("pi_h3"), "a2v": c.get("a2v")} for c in clips_rec],
        "chain_total_frames": (meta.get("chain") or {}).get("total_frames"),
        "retake": meta.get("retake") or h3.get("retake"), "v2v": meta.get("v2v") or h3.get("v2v"),
        "inpaint": h3.get("inpaint"), "outpaint": h3.get("outpaint"),
    }


def spike_sigmas(item: str) -> dict | None:
    s = (read_json(SPIKE / "runs" / item / "result.json").get("sessions") or [{}])[0]
    return {"video": s.get("video_sigmas"), "audio": s.get("audio_sigmas")} if s else None


def pi_checks(p: dict, choice: str, spike_item: str | None = None) -> dict:
    """The record checks of a pi-H3 job: what the product's done event carries."""
    rec = p.get("pi_h3") or {}
    steps = {"4steps": 4, "2steps": 2}[choice]
    out = {
        "pi_h3_used": p.get("pi_h3_used") == choice,
        "record_choice_steps": rec.get("choice") == choice and rec.get("steps") == steps,
        "lora_pairs_138": get(rec, "lora", "pairs_applied") == 138,
        "adapter_counts": bool(get(rec, "adapter", "counts")) or rec.get("dit_source") == "resident",
        "sigmas_lengths": len(get(rec, "sigmas", "video", default=[])) == steps + 1
        and len(get(rec, "sigmas", "audio", default=[])) == steps + 1,
        "dwd_chunks_recorded": isinstance(get(rec, "dwd", "chunks"), int) and rec["dwd"]["chunks"] >= 1,
        "decode_by_dwd": p.get("vae_source") == "dwd",
        "steps_shift": p.get("steps_shift") == [steps, 12.0, 3.0],
    }
    if spike_item:
        ref = spike_sigmas(spike_item)
        got = rec.get("sigmas") or {}
        if ref and ref.get("video"):
            d = max(max(abs(a - b) for a, b in zip(got.get(k) or [], ref[k] or [])) if got.get(k) else 1.0
                    for k in ("video", "audio"))
            out["sigmas_equal_spike"] = d <= SIGMA_TOL
    return out


def output_checks(rec: dict, body: dict, expect: dict | None = None) -> dict:
    mp4 = REPO / rec["mp4"] if rec.get("mp4") else None
    if mp4 is None or not mp4.is_file():
        return {"output": False}
    pr = ffprobe(mp4)
    v = pr["video"]
    e = expect or {}
    want_w, want_h = e.get("width", body.get("width")), e.get("height", body.get("height"))
    return {"probe": pr, "frames_ok": v.get("nb_read_frames") == e.get("frames", body.get("num_frames")),
            "size_ok": (v.get("width"), v.get("height")) == (want_w, want_h), "audio_ok": bool(pr.get("audio"))}


# ---------------------------------------------------------------------------------------------
# Item runners
# ---------------------------------------------------------------------------------------------


def run_selftest(item: str, out_dir: Path) -> dict:
    mp4, rep = out_dir / "f0.mp4", out_dir / "selftest.json"
    cmd = [str(PY_ENG), "-B", "-m", "engine_h3.worker", "--selftest-generate",
           "--transformer", f"models/MiniMaxH3/Weights/{FL2VA}",
           "--text-encoder", "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
           "--video-vae", "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
           "--audio-vae", "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
           "--tokenizer-dir", "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
           "--width", "640", "--height", "384", "--num-frames", "124", "--seed", "42", "--steps", "20",
           "--shift-video", "12", "--shift-audio", "3", "--attention-backend", "sdpa", "--vram16",
           "--prompt", RAINCOAT, "--output", str(mp4), "--report", str(rep)]
    rec: dict[str, Any] = {"cmd": [rel(c) if os.path.isabs(c) else c for c in cmd]}
    if item == "__plan__":
        return rec
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONUTF8": "1"}
    t0 = time.monotonic()
    with (out_dir / "console.log").open("w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            rec["exit_code"] = proc.wait(timeout=1800)
        finally:
            if proc.poll() is None:
                run_cmd(["taskkill", "/PID", str(proc.pid), "/T", "/F"])
                rec["killed"] = True
    rec["seconds_wall"] = round(time.monotonic() - t0, 1)
    rec["sha256"] = sha256(mp4) if mp4.is_file() else None
    rec["stream_md5"] = stream_md5s(mp4) if mp4.is_file() else None
    done = get(read_json(rep), "rounds", default=[{}])[0].get("done") or {}
    rec["done"] = {k: done.get(k) for k in ("seconds", "peak_vram_mb", "peak_vram_reserved_mb", "vram16_mode_used")}
    rec["fingerprint_equal"] = rec["sha256"] == FINGERPRINT
    rec["verdict"] = "pass" if rec.get("exit_code") == 0 and rec["fingerprint_equal"] else "changed"
    return rec


def run_engine_direct(item: str, out_dir: Path, dry: bool = False) -> dict:
    src = SPIKE / "runs" / ITEMS[item]["source"] / "result.json"
    cmd = [str(PY_ENG), "-I", "-B", str(HERE / "engine_direct.py"), "--source", str(src), "--out", str(out_dir)]
    if dry:
        cmd.append("--dry-run")
    env = {**os.environ, "PYTHONUTF8": "1"}
    if dry:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    rec: dict[str, Any] = {"cmd": [rel(c) for c in cmd], "source": rel(src)}
    t0 = time.monotonic()
    with (out_dir / ("dry_console.log" if dry else "console.log")).open("w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            rec["exit_code"] = proc.wait(timeout=600 if dry else 3600)
        finally:
            if proc.poll() is None:
                run_cmd(["taskkill", "/PID", str(proc.pid), "/T", "/F"])
                rec["killed"] = True
    rec["seconds_wall"] = round(time.monotonic() - t0, 1)
    if dry:
        plan = read_json(out_dir / "engine_direct_plan.json")
        rec["plan"] = {k: plan.get(k) for k in ("validate_request", "missing_paths", "cuda_visible_devices", "options")}
        return rec
    res = read_json(out_dir / "engine_direct.json")
    eng = res.get("engine") or {}
    rec["error"] = res.get("error")
    rec["engine"] = {k: eng.get(k) for k in ("seconds", "peak_vram_mb", "peak_vram_reserved_mb", "attention_used",
                                             "vram16_mode_used", "keep_resident_used", "rss_peak_gib",
                                             "commit_peak_gib", "num_frames")}
    dec = ((eng.get("h3") or {}).get("phases") or {}).get("decode") or {}
    rec["engine"]["decode"] = {k: dec.get(k) for k in ("seconds", "peak_vram_mb", "peak_vram_reserved_mb", "vae_source")}
    rec["engine"]["dwd_chunks"] = len((dec.get("dwd") or {}).get("chunks") or []) or None
    rec["pi_h3_keys"] = [k for k in ("pi_h3_used",) if k in eng] + (["h3.pi_h3"] if "pi_h3" in (eng.get("h3") or {}) else [])
    mp4 = out_dir / "output.mp4"
    if rec.get("exit_code") == 0 and mp4.is_file():
        rec["compare"] = c = compare(mp4, ref_spike(ITEMS[item]["source"]))
        rec["probe"] = ffprobe(mp4)
        if not ITEMS[item].get("pi"):
            rec["verdict"] = "pass" if c.get("equal") and not rec["pi_h3_keys"] else "changed"
        else:
            # the experiment's reproduction through the product's pi-H3 path (group 2's rule)
            rec["pi_h3"] = {"pi_h3_used": eng.get("pi_h3_used"), "record": (eng.get("h3") or {}).get("pi_h3"),
                            "clips": [cl.get("pi_h3") for cl in (eng.get("chain") or {}).get("clips") or []]}
            fr = c.get("frames") or {}
            if fr.get("frame_count_differs") or (fr.get("psnr_min_db") is not None and fr["psnr_min_db"] < BROKEN_PSNR_DB):
                rec["verdict"] = "broken"
            elif not eng.get("pi_h3_used"):
                rec["verdict"] = "record_missing"
            else:
                rec["verdict"] = "equal" if c.get("equal") else "diff"
    else:
        rec["verdict"] = "fail"
    return rec


def judge_api(item: str, spec: dict, jobs: list[dict]) -> str:
    """The item's verdict from its job records (each already holds checks / compare)."""
    if any(j.get("status") != "completed" for j in jobs):
        if spec.get("large") and any(j.get("status") in ("failed", "timeout") for j in jobs):
            return "oom" if any("OOM" in str(j.get("error") or "") for j in jobs) else jobs[-1].get("status")
        return "fail"
    out_ok = all(all(j.get("checks", {}).get(k, False) for k in ("frames_ok", "size_ok", "audio_ok")) for j in jobs)
    rec_ok = all(all((j.get("pi_checks") or {}).values()) for j in jobs) and all(j.get("extra_ok", True) for j in jobs)
    cmps = [j["compare"] for j in jobs if j.get("compare")]
    if spec["group"] == 1 and item != "F1":
        return "pass" if out_ok and cmps and cmps[0].get("equal") and all(j.get("extra_ok", True) for j in jobs) else "changed"
    if item == "F1":
        return "pass" if out_ok and cmps and cmps[0].get("equal") and jobs[0].get("extra_ok") else "changed"
    if spec["group"] == 2:
        c = cmps[0] if cmps else {}
        fr = c.get("frames") or {}
        if not out_ok or fr.get("frame_count_differs") or (fr.get("psnr_min_db") is not None
                                                           and fr["psnr_min_db"] < BROKEN_PSNR_DB):
            return "broken"
        if not rec_ok:
            return "record_missing"
        return "equal" if c.get("equal") else "diff"
    if cmps:
        if not out_ok or not rec_ok:
            return "fail"
        return "equal" if all(c.get("equal") for c in cmps) else "diff"
    return "pass" if out_ok and rec_ok else "fail"


def run_api_item(item: str, spec: dict, port: int, out_dir: Path) -> dict:
    rec: dict[str, Any] = {"jobs": []}
    srv = Server(port, item)
    cache: dict = {}
    try:
        srv.start()
        loaded = None
        comfort = config_comfort(port)
        for jb in spec["jobs"]:
            if loaded != jb["file"]:
                ld = load_weight(port, jb["file"])
                rec.setdefault("loads", []).append(ld)
                if not ld.get("ok"):
                    rec["jobs"].append({"job": jb["name"], "status": "load_error", "error": ld.get("error")})
                    break
                loaded = jb["file"]
            qc = active_quant_class(port)
            b = jb["body"]
            ntok = prompt_tokens(b["prompt"])
            kf = [f for _p, f in jb["uploads"].get("images") or []]
            est = rows_estimate(b["width"], b["height"], body_frames(b), text_tokens=ntok or 512, keyframes=kf)
            budget = comfort_budget(qc, comfort or None)
            comfort_rec = {"quant_class": qc, "budget_rows": budget, "rows_estimate": est, "text_tokens": ntok,
                           "inside": budget is None or est["total"] <= budget}
            log(f"  {jb['name']}: rows estimate {est['total']} vs line {budget} ({qc}) -> "
                f"{'inside' if comfort_rec['inside'] else 'OUTSIDE'}")
            j = run_api_job(port, item, jb, cache, out_dir)
            if j.get("status") == "failed" and jb.get("fallback") and "OOM" in str(j.get("error") or ""):
                log(f"  {jb['name']}: GPU_OOM -> fallback {jb['fallback']['width']}x{jb['fallback']['height']}"
                    f"x{jb['fallback']['num_frames']}")
                j = {"first_attempt": j, **run_api_job(port, item, jb, cache, out_dir, body=jb["fallback"])}
                b = jb["fallback"]
            j["comfort"] = comfort_rec
            if j.get("status") == "timeout":
                rec["jobs"].append(j)
                log(f"  {jb['name']}: timeout -> stopping the backend")
                break
            meta = read_json(REPO / j["metadata_file"]) if j.get("metadata_file") else {}
            j["picks"] = p = picks(meta)
            if j.get("status") == "completed":
                j["checks"] = output_checks(j, b, jb.get("expect"))
                choice = b.get("pi_h3")
                if choice:
                    j["pi_checks"] = pi_checks(p, choice, item if spec["group"] == 2 and spec["kind"] == "api" and item in ("P1", "P2", "P3", "P4") else None)
                extra = {}
                if not choice:
                    extra["no_pi_keys"] = p.get("pi_h3_used") is None and p.get("pi_h3") is None and \
                        "pi_h3=" not in str(j.get("worker_generate_line") or "")
                if jb["uploads"].get("images"):
                    extra["images_used"] = p.get("images_used") == len(jb["uploads"]["images"])
                if jb["uploads"].get("references"):
                    extra["references"] = bool(p.get("references"))
                if b.get("nag_enabled"):
                    extra["vsf_recorded"] = bool(p.get("vsf"))
                if b.get("vram16_mode") is False:
                    extra["vram16_off"] = p.get("vram16_mode_used") == "off"
                if b.get("keep_resident"):
                    extra["keep_resident_on"] = p.get("keep_resident_used") == "on"
                j["extra"] = extra
                j["extra_ok"] = all(extra.values())
                ct = jb.get("compare_to")
                if spec.get("gc0"):
                    j["compare"] = gc0_compare(REPO / j["mp4"], p)
                    log(f"  {jb['name']}: vs G-C0: sha256 {'=' if j['compare']['sha256_equal'] else '≠'} "
                        f"latents {'=' if j['compare']['latents_equal'] else '≠'}")
                elif ct:
                    ref = ref_f1() if ct == "F1" else ref_this_or_spike(ct) if isinstance(ct, str) else ct
                    j["compare"] = compare(REPO / j["mp4"], ref)
                    c = j["compare"]
                    log(f"  {jb['name']}: vs {c['label']}: video {'=' if c.get('video_equal') else '≠'} "
                        f"audio {'=' if c.get('audio_equal') else '≠'}"
                        + (f"; PSNR min {get(c, 'frames', 'psnr_min_db')} dB" if c.get("frames") else ""))
            rec["jobs"].append(j)
    except BaseException as exc:  # noqa: BLE001 -- recorded; the backend is stopped below
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        log(f"{item}: stopped: {rec['error']}")
    finally:
        try:
            srv.stop()
        except Exception as exc:  # noqa: BLE001
            srv.record["stop_error"] = f"{type(exc).__name__}: {exc}"
        rec["server"] = srv.record
    rec["verdict"] = "script_error" if rec.get("error") else judge_api(item, spec, rec["jobs"])
    return rec


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------

VERDICT_JA = {"pass": "合格", "equal": "一致", "diff": "差あり", "changed": "変わった（off の経路）",
              "broken": "破綻", "record_missing": "記録の不足", "fail": "不合格", "oom": "メモリ不足",
              "timeout": "時間切れ", "refused": "断った（関門）", "halted": "止めた（_halt.json）",
              "script_error": "台本の誤り", "failed": "失敗", "cancelled": "取り消し", None: "—"}


def summarize() -> dict:
    out: dict[str, Any] = {"made_at": now_iso(), "items": {}}
    for it, spec in ITEMS.items():
        r = read_json(RUNS / it / "result.json")
        if not r:
            continue
        row: dict[str, Any] = {"title": spec["title"], "group": spec["group"], "rule": spec["rule"],
                               "verdict": r.get("verdict"), "finished_at": r.get("finished_at")}
        if spec["kind"] == "selftest":
            row["numbers"] = {"sha256": r.get("sha256"), **(r.get("done") or {})}
        elif spec["kind"] == "engine_direct":
            row["numbers"] = {"engine": r.get("engine"), "compare": {k: v for k, v in (r.get("compare") or {}).items()
                                                                     if k != "psnr_per_frame"}}
        else:
            row["jobs"] = []
            for j in r.get("jobs") or []:
                p = j.get("picks") or {}
                c = j.get("compare") or {}
                row["jobs"].append({
                    "job": j.get("job"), "status": j.get("status"), "error": j.get("error"),
                    "failed_stage": j.get("failed_stage"), "seconds_wall": j.get("seconds_wall"),
                    "generation_time_seconds": p.get("generation_time_seconds"), "seconds": p.get("seconds"),
                    "vram_peak_mib": p.get("vram_peak_mib"), "rss_peak_gib": p.get("rss_peak_gib"),
                    "dwd_chunks": len(p.get("dwd_chunks") or []) or None,
                    "rows": get(p, "sequence", "total"), "comfort": {k: (j.get("comfort") or {}).get(k)
                                                                    for k in ("budget_rows", "inside")},
                    "compare": {k: c.get(k) for k in ("label", "video_equal", "audio_equal")} if c else None,
                    "frames": c.get("frames"), "audio": c.get("audio"),
                    "pi_checks": j.get("pi_checks"), "extra": j.get("extra"),
                    "pi_h3": {k: (p.get("pi_h3") or {}).get(k) for k in ("choice", "dit_source", "resident_dropped",
                                                                          "initial_state", "dwd")},
                    "a2v_mask_value": (p.get("a2v") or {}).get("mask_value")})
        out["items"][it] = row
    out["halt"] = read_json(HALT) or None
    return out


def write_summary(inv: dict | None = None) -> None:
    s = summarize()
    old = read_json(SUMMARY_JSON)
    s["invocations"] = (old.get("invocations") or []) + ([inv] if inv else [])
    write_json(SUMMARY_JSON, s)
    L = ["# pi-H3 段 1 の実機の確認（記録）", "", "台本 `outputs/h3-pih3-gates-2026-11-23/h3_pih3_gates.py`。", ""]
    if s.get("halt"):
        L += [f"**止めた印あり**（`runs/_halt.json`）: {s['halt'].get('item')} の {s['halt'].get('verdict')}。", ""]
    L += ["| 項目 | 内容 | 判定 | 主な数値 |", "|---|---|---|---|"]
    for it, row in s["items"].items():
        nums = ""
        if row.get("numbers"):
            n = row["numbers"]
            nums = (f"sha256 `{str(n.get('sha256'))[:12]}…` {n.get('seconds')} 秒" if "sha256" in n else
                    f"映像 {'=' if get(n, 'compare', 'video_equal') else '≠'} 音声 "
                    f"{'=' if get(n, 'compare', 'audio_equal') else '≠'}・{get(n, 'engine', 'seconds')} 秒")
        else:
            parts = []
            for j in row.get("jobs") or []:
                sec = j.get("seconds") or {}
                vr = j.get("vram_peak_mib") or {}
                t = (f"{j['job']}: {j['status']}・全体 {j.get('generation_time_seconds')} 秒・除ノイズ {sec.get('denoise')}・"
                     f"DWD {sec.get('dwd')}・復号の段の VRAM {vr.get('decode')}・RSS {j.get('rss_peak_gib')} GiB")
                if j.get("compare"):
                    c = j["compare"]
                    t += f"・vs {c['label']}: 映像 {'=' if c['video_equal'] else '≠'} 音声 {'=' if c['audio_equal'] else '≠'}"
                    if j.get("frames"):
                        t += f"（PSNR 最小 {j['frames'].get('psnr_min_db')} dB・平均 {j['frames'].get('psnr_mean_db_finite')}）"
                if j.get("failed_stage"):
                    t += f"・落ちた段 {j['failed_stage']}"
                parts.append(t)
            nums = "<br>".join(parts)
        L.append(f"| {it} | {row['title']} | {VERDICT_JA.get(row['verdict'], row['verdict'])} | {nums} |")
    L += ["", "合格の条件:", ""] + [f"- {it}: {row['rule']}" for it, row in s["items"].items()]
    write_text(SUMMARY_MD, "\n".join(L) + "\n")


# ---------------------------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------------------------


def dry_run(order: list[str]) -> int:
    sys.path.insert(0, str(REPO))
    problems: list[str] = []
    out: dict[str, Any] = {"made_at": now_iso(), "git": git_info(), "order": order, "items": {}}
    log(f"git {out['git'].get('branch')} {out['git'].get('head')}; changed paths {len(out['git'].get('status_porcelain') or [])}")
    # gate functions: the pure judge on synthetic lines, then the real read-only gate
    t = {"free": gpu_judge(["1234, [Insufficient Permissions], [N/A]"], "RTX, 1200, 16376", 99),
         "python_on_gpu": gpu_judge(["555, C:\\x\\python.exe, 900 MiB"], "RTX, 1200, 16376", 99),
         "busy": gpu_judge([], "RTX, 5000, 16376", 99), "smi_failed": gpu_judge([], "", 99, (1, 0))}
    gate_unit = {k: v["blocking"] for k, v in t.items()}
    gate_unit_ok = (not gate_unit["free"] and len(gate_unit["python_on_gpu"]) == 1 and len(gate_unit["busy"]) == 1
                    and len(gate_unit["smi_failed"]) >= 1 and "C:\\x" not in json.dumps(t))
    out["gate_unit"] = {"blocking": gate_unit, "ok": gate_unit_ok}
    log(f"gate judge on synthetic inputs: {'ok' if gate_unit_ok else 'FAILED'} {gate_unit}")
    problems += [] if gate_unit_ok else ["gate judge"]
    out["gate_now"] = pre = preflight_checks()
    log_gate(pre)
    out["halt_marker"] = read_json(HALT) or None
    # files
    files = {"spike runs": [SPIKE / "runs" / i / "output.mp4" for i in ("P1", "P2", "P3", "P4", "I1c", "O1c")],
             "spike materials": [SPIKE / "_material" / n for n in ("I1_source_1280x768_73f.mp4", "I1_mask_1280x768_73f.mp4",
                                                                  "O1_source_1024x576_124f.mp4")],
             "R1 record": [MIXED_SUMMARY, MIXED_R1_MP4],
             "weights": [WEIGHTS / f for f in (FL2VA, SULPHUR, REF2VA, W4A8)] + list(PI_FILES.values())
             + [LORA_DIR / TURBO_LORA],
             "materials": [I2V_FIRST, KF_FIRST, KF_LAST, REF_PICTURE, A2V_IMAGE, A2V_WAV],
             "venvs": [PY_APP, PY_ENG]}
    out["files"] = {g: {rel(p): p.exists() for p in ps} for g, ps in files.items()}
    for g, d in out["files"].items():
        miss = [k for k, v in d.items() if not v]
        log(f"files [{g}]: {'all present' if not miss else 'MISSING ' + str(miss)}")
        if miss and not (g == "weights" and miss == [rel(WEIGHTS / W4A8)]):
            problems.append(f"missing {miss}")
    r1 = r1_record()
    out["r1_md5"] = r1
    log(f"R1 recorded stream MD5: {r1}")
    problems += [] if r1 else ["R1 MD5 record"]
    # pi-H3 vs each base weight (the product's header match, torch-free)
    import h3_pih3_format as PF  # noqa: PLC0415 -- moved to the repo root (2026-11-23)

    out["pi_match"] = {}
    for f in (FL2VA, SULPHUR, REF2VA, W4A8):
        if not (WEIGHTS / f).is_file():
            continue
        for ch, pf in PI_FILES.items():
            pr = PF.model_problems(pf, WEIGHTS / f)
            out["pi_match"][f"{f} + {ch}"] = pr[:5]
            log(f"pi-H3 {ch} vs {f}: {'match' if not pr else str(len(pr)) + ' problem(s): ' + '; '.join(pr[:2])}")
    out["quant_class"] = {f: quant_class(f) for f in (FL2VA, SULPHUR, REF2VA, W4A8) if (WEIGHTS / f).is_file()}
    log(f"quant classes: {out['quant_class']}")
    # requests through the product's family rules (no backend; upload ids are placeholders)
    from api.models import GenerateChainRequest, GenerateRequest  # noqa: PLC0415
    from services import engines  # noqa: PLC0415

    def assets(k: str) -> Path:
        return REPO / "models" / descriptor()["assets"][k]

    for it in order:
        spec = ITEMS[it]
        entry: dict[str, Any] = {"title": spec["title"], "group": spec["group"], "kind": spec["kind"], "rule": spec["rule"],
                                 "done": bool(read_json(RUNS / it / "result.json").get("verdict"))}
        log(f"[{spec['group']}] {it}: {spec['title']}{'  (done; skipped in a live run)' if entry['done'] else ''}")
        if spec["kind"] == "selftest":
            entry["cmd"] = run_selftest("__plan__", RUNS / it)["cmd"]
            log(f"    selftest: {' '.join(entry['cmd'][:4])} ... (expect sha256 {FINGERPRINT[:12]}…)")
        elif spec["kind"] == "engine_direct":
            d = HERE / "_dry" / it
            d.mkdir(parents=True, exist_ok=True)
            r = run_engine_direct(it, d, dry=True)
            entry["engine_direct"] = r
            ok = r.get("exit_code") == 0 and get(r, "plan", "validate_request") == "passed"
            log(f"    engine-direct rebuild of {spec['source']}: {'ok' if ok else 'PROBLEM'} {r.get('plan')}")
            problems += [] if ok else [f"{it} engine-direct plan"]
        else:
            entry["jobs"] = []
            for jb in spec["jobs"]:
                for label, body in (("", jb["body"]),) + ((("fallback", jb["fallback"]),) if jb["fallback"] else ()):
                    b = dict(body)
                    if jb["uploads"].get("images"):
                        b["conditioning_images"] = [{"image_id": "dry", "frame_idx": f} for _p, f in jb["uploads"]["images"]]
                    if jb["uploads"].get("source_audio"):
                        b["source_audio"] = {"audio_id": "dry", "strength": jb["uploads"]["source_audio"][1]}
                    if jb["uploads"].get("references"):
                        b["references"] = [{"kind": k, "id": "dry", "slot": s} for k, _p, s in jb["uploads"]["references"]]
                    if jb["lora"]:
                        b["loras"] = [{"name": Path(jb["lora"][0]).stem, "strength": jb["lora"][1]}]
                    ph = placeholder_paths(b)
                    missing_up = [rel(x) for x in ph if not x.is_file()]
                    if missing_up:
                        problems.append(f"{it}/{jb['name']}: upload material missing {missing_up}")

                    def dry_ids(o):
                        if isinstance(o, str):
                            return "dry" if o.startswith(("@image:", "@video:", "@audio:")) else o
                        if isinstance(o, dict):
                            return {k: dry_ids(v) for k, v in o.items()}
                        if isinstance(o, list):
                            return [dry_ids(v) for v in o]
                        return o

                    b = dry_ids(b)
                    jr: dict[str, Any] = {"job": jb["name"] + (f" ({label})" if label else ""), "file": jb["file"],
                                          "endpoint": jb.get("endpoint"), "body": b,
                                          "uploads": [rel(x) for x in ph], "uploads_missing": missing_up}
                    try:
                        if jb.get("endpoint") == "/generate/chain":
                            req = GenerateChainRequest(**b)
                            engines.reject_chain("h3", req)
                            engines.validate_chain_for_family("h3", req)
                        else:
                            req = GenerateRequest(**b)
                            engines.reject_unsupported("h3", req)
                            engines.validate_for_family("h3", req)
                        if (WEIGHTS / jb["file"]).is_file():
                            engines.preflight_request("h3", req, assets=assets, transformer=lambda: WEIGHTS / jb["file"])
                        jr["family_rules"] = "passed"
                        jr["filled"] = {k: getattr(req, k) for k in ("num_inference_steps", "shift_video", "shift_audio",
                                                                     "pi_h3", "vram16_mode", "keep_resident")}
                    except Exception as exc:  # noqa: BLE001
                        jr["family_rules"] = f"{type(exc).__name__}: {getattr(exc, 'detail', None) or exc}"
                    ntok = prompt_tokens(b["prompt"])
                    kf = [f for _p, f in jb["uploads"].get("images") or []]
                    est = rows_estimate(b["width"], b["height"], body_frames(b), text_tokens=ntok or 512, keyframes=kf)
                    budget = comfort_budget(out["quant_class"].get(jb["file"]))
                    jr["comfort"] = {"rows_estimate": est["total"], "text_tokens": ntok, "budget_rows": budget,
                                     "inside": budget is None or est["total"] <= budget}
                    ct = jb.get("compare_to")
                    if ct:
                        ref = ref_f1() if ct == "F1" else ref_this_or_spike(ct) if isinstance(ct, str) else ct
                        jr["compare_to"] = {"label": ref["label"], "mp4": rel(ref["mp4"]) if ref.get("mp4") else None,
                                            "present": bool(ref.get("md5")) or bool(ref.get("mp4") and Path(ref["mp4"]).is_file())}
                    entry["jobs"].append(jr)
                    log(f"    {jr['job']}: {jb['file'][:40]} family rules {jr['family_rules']}"
                        + (f"; filled {jr.get('filled')}" if jr.get("filled") else "")
                        + f"; rows ~{est['total']} / line {budget} -> {'inside' if jr['comfort']['inside'] else 'OUTSIDE'}"
                        + (f"; compare {jr['compare_to']['label']} ({'present' if jr['compare_to']['present'] else 'MISSING'})"
                           if ct else ""))
                    want_ok = jr["family_rules"] == "passed" or (it == "Q1" and not (WEIGHTS / W4A8).is_file())
                    if not want_ok and it != "Q1":
                        problems.append(f"{it}/{jr['job']}: {jr['family_rules']}")
                    if ct and not jr["compare_to"]["present"] and ct not in ("F1", "P1", "P2"):
                        problems.append(f"{it}/{jr['job']}: compare reference missing")
        out["items"][it] = entry
    out["problems"] = problems
    out["ok"] = not problems
    write_json(HERE / "dry_run.json", out)
    log(f"dry-run record: {rel(HERE / 'dry_run.json')} ({'OK' if not problems else 'PROBLEMS ' + str(problems)})")
    return 0 if not problems else 1


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="pi-H3 stage 1 GPU checks")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="plan and checks; no GPU, no backend")
    mode.add_argument("--gate-check", action="store_true", help="the GPU gate only (read-only queries)")
    mode.add_argument("--live", action="store_true", help="run the items on the GPU")
    mode.add_argument("--post-only", action="store_true", help="rewrite summary.json / summary.md from runs/")
    ap.add_argument("--only", default=None, help=f"comma list of {','.join(ITEMS)} (default: {','.join(DEFAULT_ORDER)})")
    ap.add_argument("--stage2", action="store_true", help=f"the stage-2 items in order: {','.join(STAGE2_ORDER)}")
    ap.add_argument("--redo", action="store_true", help="run an item again even if its result exists (old one moved aside)")
    ap.add_argument("--continue-after-halt", action="store_true", help="run although runs/_halt.json exists")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT)
    a = ap.parse_args(argv)
    if a.only:
        names = [x.strip() for x in a.only.split(",") if x.strip()]
        bad = [x for x in names if x not in ITEMS]
        if bad or not names:
            ap.error(f"--only: unknown item(s) {bad}; choose from {','.join(ITEMS)}")
        a.order = [k for k in ITEMS if k in names]
    elif a.stage2:
        a.order = list(STAGE2_ORDER)
    else:
        a.order = list(DEFAULT_ORDER)
    if a.port == OWNER_PORT:
        ap.error(f"port {OWNER_PORT} is the owner's server")
    return a


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    a = parse_args(argv)
    mode = "dry_run" if a.dry_run else "gate_check" if a.gate_check else "post_only" if a.post_only else "live"
    log.open(LOG_DIR / f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    try:
        if a.dry_run:
            return dry_run(a.order)
        if a.gate_check:
            pre = preflight_checks()
            log_gate(pre)
            write_json(HERE / "gate_check.json", pre)
            return 0 if not pre["blocking"] else 1
        if a.post_only:
            write_summary()
            log(f"summary: {rel(SUMMARY_MD)}")
            return 0
        inv: dict[str, Any] = {"started_at": now_iso(), "argv": sys.argv[1:], "git": git_info(), "items": {}}
        code = 0
        for it in a.order:
            spec = ITEMS[it]
            out_dir = RUNS / it
            prev = read_json(out_dir / "result.json")
            if prev.get("verdict") and prev.get("verdict") not in ("refused", "halted") and not a.redo:
                log(f"{it}: done before ({prev['verdict']}); skipped (--redo to run again)")
                continue
            if HALT.exists() and not a.continue_after_halt:
                h = read_json(HALT)
                log(f"{it}: refused: runs/_halt.json ({h.get('item')}: {h.get('verdict')}); --continue-after-halt to go on")
                inv["items"][it] = "halted"
                code = 3
                break
            if out_dir.exists() and prev:
                out_dir.rename(out_dir.with_name(f"{it}.old-{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
            out_dir.mkdir(parents=True, exist_ok=True)
            rec: dict[str, Any] = {"item": it, "title": spec["title"], "group": spec["group"], "rule": spec["rule"],
                                   "started_at": now_iso(), "git": git_info()}
            log(f"=== {it}: {spec['title']} (git {str(rec['git'].get('head'))[:10]}, "
                f"{len(rec['git'].get('status_porcelain') or [])} changed path(s)) ===")
            rec["gate"] = pre = preflight_checks()
            log_gate(pre)
            if pre["blocking"]:
                rec.update(verdict="refused", finished_at=now_iso())
                write_json(out_dir / "result.json", rec)
                inv["items"][it] = "refused"
                code = 1
                break
            try:
                if spec["kind"] == "selftest":
                    rec.update(run_selftest(it, out_dir))
                elif spec["kind"] == "engine_direct":
                    rec.update(run_engine_direct(it, out_dir))
                else:
                    rec.update(run_api_item(it, spec, a.port, out_dir))
            except BaseException as exc:  # noqa: BLE001
                rec.update(verdict="script_error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            rec["finished_at"] = now_iso()
            rec["workers_after"] = [p["pid"] for p in all_processes() if WORKER_RE.search(p["cmdline"] or "")]
            write_json(out_dir / "result.json", rec)
            inv["items"][it] = rec["verdict"]
            log(f"{it}: verdict {rec['verdict']} ({VERDICT_JA.get(rec['verdict'], '')})")
            if rec["workers_after"]:
                log(f"WARNING: engine worker process(es) still alive after {it}: {rec['workers_after']}")
            if (spec["group"] == 1 and rec["verdict"] in ("changed", "fail", "script_error")) or \
                    (spec["group"] == 2 and rec["verdict"] in ("broken", "fail", "script_error")):
                write_json(HALT, {"item": it, "verdict": rec["verdict"], "at": now_iso()})
                log(f"{it}: {rec['verdict']} -> runs/_halt.json written; later items refuse until --continue-after-halt")
                code = 3
                break
            if rec["verdict"] in ("fail", "script_error", "timeout", "oom"):
                code = 3
        inv["finished_at"] = now_iso()
        inv["scrubbed"] = scrub_files()
        write_summary(inv)
        log(f"summary: {rel(SUMMARY_MD)} (exit {code})")
        return code
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
