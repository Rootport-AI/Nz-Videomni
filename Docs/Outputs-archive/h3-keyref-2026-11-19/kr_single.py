r"""§1-115 の比較実験（段階 A）: Reference とキーフレームの併用。1 条件＝1 プロセス（門だけ 2 本を 1 プロセスで）。
計画 glistening-splashing-hoare.md §4〜§6。製品コードは変えない（内部関数を呼ぶだけ）。

Three ways to run (plan §4):
  --mode product   H3Pipeline.generate() itself, untouched (runs 3, 4, 8). Its return value goes into the record.
  --mode harness   generate()'s steps in generate()'s order, from the engine's own stage functions, with the
                   key frame's three paths switched by --T / --C / --L (runs 1, 5-7, 9-11). Decode and mp4 by
                   the product's _decode. The final latents are saved (.pt).
  --mode gate      harness 010, then product 010, in ONE process; the two mp4s' video / audio stream MD5s
                   must match (runs 1 and 2). Exit 2 when they do not.

  T on : the key frame goes to the text encoder only, behind the references, as RefMedia("picture", N+1,
         image=<its canvas>) (SKEBA's form; no extra text). _encode_references / _denoise get the true
         references only.
  C off: cond_clean = None and cond_frame_idx = [] (both; _denoise checks the row count against the indices).
  L m  : the key frame's _encode_image rows -> unpatchify_video -> [1, 24, 1, H/16, W/16] at latent 5*(n/17)
         of a zero ``clean``; per latent frame m there, 1.0 elsewhere; _denoise(video_fix=..., video_inject_clean).

Run from the repo root (run_stageA.ps1 does the plan's list):

    set PYTHONPATH=%CD%
    set PYTHONUTF8=1
    .venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-keyref-2026-11-19\kr_single.py --name r05_110 ^
        --mode harness --transformer models\MiniMaxH3\Weights\minimax_h3_ref2va_pruned_int8_convrot.safetensors ^
        --prompt-file Docs\Outputs-archive\h3-keyref-2026-11-19\prompt_ref.txt ^
        --keyframe 0=uploads\key-ref-development\key-001.jpg --ref-image <picture> --T on --C on
    ... --dry-run   (no torch model, no weights read: the arguments, the files and the plan of the run only)

Exit code 0 = done; 1 = failed (exception, OOM, a refused argument, an L self-check that failed);
2 = the gate's streams did not match. The JSON report is always written.
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # no __pycache__ in the tracked archive

# Before torch is imported (the worker does the same, engine_h3/worker.py:203; the adapter sets it too).
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

# torch-free (engine_h3/constants.py imports nothing heavy): usable in --dry-run.
from engine_h3.constants import (  # noqa: E402
    DEFAULT_SHIFT_AUDIO,
    DEFAULT_SHIFT_VIDEO,
    DEFAULT_STEPS,
    SPATIAL_FACTOR,
    VAE_CLIP_LENGTH,
    latent_frames,
)

OUT_ROOT = ROOT / "outputs/h3-keyref-2026-11-19"
#: The other four files (the transformer comes from --transformer). Same as models/INSTALLED_PATHS.txt:22-25.
PATHS = {
    "text_encoder": "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "video_vae": "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
    "tokenizer_dir": "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
}
#: The repo's weight hash cache (written by the app; read only, never computed here).
IDENTITY_FILE = ROOT / "weight_identity.json"
FFMPEG = ROOT / "tools/ffmpeg/bin/ffmpeg.exe"

#: What the product's path sends (plan §4 共通の条件; checked against the code, see README):
#: load (services/engines/h3/adapter.py:962-975 -> engine_h3/worker.py:393-399)
LOAD_BLOCKS_ON_GPU = 8           # config.yaml:89 block_swap_blocks_on_gpu (adapter.py:969, DEFAULT 131)
LOAD_ATTENTION = "sdpa"          # adapter.py:970 = api/models.py:471 GenerateRequest default (load only)
LOAD_PREFETCH = True             # adapter.py:971 = api/models.py:145/472
LOAD_DETERMINISTIC = True        # adapter.py:974
#: generate (adapter.py:1114-1161 -> worker.py:690-713, 740-741)
JOB_ATTENTION = "sage"           # the owner's setting (per job, adapter.py:1129; worker.py:740-741)
JOB_PREFETCH = True              # adapter.py:1130
JOB_VRAM16 = True                # adapter.py:1134 (api/models.py:198 VRAM16_MODE_DEFAULT_H3)
JOB_KEEP_RESIDENT = False        # adapter.py:1160 (key absent) -> worker.py:702
JOB_FRAME_RATE = 24.0            # adapter.py:146 H3_FRAME_RATE, cast float by worker.py:697
DEFAULT_INJECT_CLEAN = 0.999     # plan §4 L (= engine_h3/denoise_mask.py VIDEO_INJECT_CLEAN)


def _log(msg: str) -> None:
    print(f"[kr_single] {msg}", flush=True)


def _progress(stage: str, index: int, total: int) -> None:
    print(f"[progress] {stage} {index}/{total}", flush=True)


def _rel(p) -> str:
    q = Path(p).resolve()
    try:
        return str(q.relative_to(ROOT))
    except ValueError:
        return str(q)


def _abs(p) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_tensor(t) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _json_safe(o, bad: list):
    """A copy of ``o`` with every non-finite float replaced by None (each one appended to ``bad``)."""
    if isinstance(o, float):
        if math.isfinite(o):
            return o
        bad.append(o)
        return None
    if isinstance(o, dict):
        return {str(k): _json_safe(v, bad) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v, bad) for v in o]
    return o


# -- weights -------------------------------------------------------------------------------------


def weight_identity(path: Path) -> str | None:
    """The cached sha256 of ``path`` from weight_identity.json when its entry (key = the lower-cased
    absolute path) has the file's size and mtime_ns; None otherwise."""
    try:
        data = json.loads(IDENTITY_FILE.read_text(encoding="utf-8-sig"))
        st = path.stat()
        entry = (data.get("entries") or {}).get(str(path.resolve()).lower())
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(entry, dict):
        return None
    if entry.get("size") != st.st_size or entry.get("mtime_ns") != st.st_mtime_ns:
        return None
    sha = entry.get("sha256")
    return sha if isinstance(sha, str) and sha else None


def weights_record(transformer) -> dict:
    tp = _abs(transformer)
    if not tp.is_file():
        raise ValueError(f"--transformer not found: {transformer}")
    out = {"transformer": {"path": Path(_rel(tp)).as_posix(), "bytes": tp.stat().st_size,
                           "sha256_cached": weight_identity(tp)}}
    for key, rel in PATHS.items():
        p = ROOT / rel
        if not p.exists():
            raise ValueError(f"{key} not found: {rel}")
        out[key] = {"path": rel, "bytes": p.stat().st_size if p.is_file() else None}
    return out


# -- arguments -----------------------------------------------------------------------------------


def parse_keyframes(items: list, num_frames: int) -> list:
    """``--keyframe F=PATH`` (repeatable) -> ``[{"frame", "path" (repo-relative), "abs"}]`` by frame."""
    out, seen = [], set()
    for raw in items or ():
        frame, sep, path = str(raw).partition("=")
        if not sep or not frame.strip().lstrip("-").isdigit() or not path.strip():
            raise ValueError(f"--keyframe must be FRAME=PATH ({raw!r})")
        n = int(frame)
        if not 0 <= n < num_frames:
            raise ValueError(f"--keyframe frame {n} is outside 0..{num_frames - 1}")
        if n in seen:
            raise ValueError(f"--keyframe frame {n} twice")
        seen.add(n)
        ap = _abs(path.strip()).resolve()
        if not ap.is_file():
            raise ValueError(f"--keyframe file not found: {path}")
        out.append({"frame": n, "path": Path(_rel(ap)).as_posix(), "abs": str(ap)})
    return sorted(out, key=lambda k: k["frame"])


def parse_refs(images: list, audios: list) -> list:
    """``--ref-image`` / ``--ref-audio`` (repeatable, in order) -> slots 1, 2, ... per kind, the engine's order."""
    out = []
    for kind, items in (("picture", images or ()), ("audio", audios or ())):
        for i, path in enumerate(items, start=1):
            ap = _abs(path).resolve()
            if not ap.is_file():
                raise ValueError(f"--ref-{'image' if kind == 'picture' else 'audio'} file not found: {path}")
            out.append({"kind": kind, "slot": i, "path": Path(_rel(ap)).as_posix(), "abs": str(ap),
                        "file_sha256": _sha_file(ap)})
    return out


def latent_index(frame: int) -> int:
    """Plan §2: only a latent that holds one picture frame: frame 17k -> latent 5k. Others are refused."""
    if frame % VAE_CLIP_LENGTH:
        raise ValueError(f"--L needs key frames on a multiple of 17 (frame {frame}: no latent holds it alone)")
    return 5 * (frame // VAE_CLIP_LENGTH)


def check_args(args, keyframes: list, refs: list) -> dict:
    """The refusals (ValueError) and what goes where. Returns the ``paths`` record: per key frame what the
    text encoder and the DiT get, for the record and the dry run."""
    t_on, c_on, l_m = args.T == "on", args.C == "on", args.L
    has_ref = bool(refs)
    pictures = [r for r in refs if r["kind"] == "picture"]
    last = int(args.num_frames) - 1
    if l_m is not None and not (math.isfinite(l_m) and 0.0 <= l_m <= 1.0):
        raise ValueError(f"--L must be in [0, 1] ({l_m})")
    if not (math.isfinite(args.inject_clean) and 0.0 <= args.inject_clean <= 1.0):
        raise ValueError(f"--inject-clean must be in [0, 1] ({args.inject_clean})")
    if args.mode == "product":
        if t_on or not c_on or l_m is not None:
            raise ValueError("--mode product takes no --T on / --C off / --L (generate() decides; plan §4)")
        if has_ref and keyframes:
            te, dit = "references only (key frames hidden from the text encoder, pipeline_h3.py:912-913)", "cond + ref"
        elif keyframes:
            te = f"key frames {[k['frame'] for k in keyframes if k['frame'] in (0, last)]} in front of the prompt"
            dit = "cond"
        else:
            te, dit = ("references" if has_ref else "prompt only"), ("ref" if has_ref else "-")
        return {"implied": {"T": bool(keyframes) and not has_ref, "C": bool(keyframes), "L": None},
                "text_encoder": te, "dit": dit}
    # harness / gate
    if not keyframes:
        raise ValueError("--mode harness needs a --keyframe (a job without one is --mode product)")
    if not has_ref:
        raise ValueError("--mode harness needs a reference (without one the key frame path is the product's I2V: "
                         "--mode product)")
    if args.mode == "gate" and (t_on or not c_on or l_m is not None):
        raise ValueError("--mode gate is 010 (no --T on / --C off / --L)")
    if t_on and len(keyframes) != 1:
        raise ValueError("--T on takes exactly one key frame in stage A (more is for the stage B / C plans)")
    lat = [latent_index(k["frame"]) for k in keyframes] if l_m is not None else []
    t_slot = (max((r["slot"] for r in pictures), default=0) + 1) if t_on else None
    te = "references" + (f" + key frame {keyframes[0]['frame']} as <Picture {t_slot}> (T item)" if t_on else "")
    dit = " + ".join(x for x in (("cond " + str([k["frame"] for k in keyframes])) if c_on else None,
                                 "ref",
                                 (f"video_fix latents {lat} m={l_m} inject_clean={args.inject_clean}"
                                  if l_m is not None else None)) if x)
    return {"T": t_on, "C": c_on, "L": l_m, "T_slot": t_slot, "L_latents": lat,
            "text_encoder": te, "dit": dit}


def condition_record(args, weights: dict, keyframes: list, refs: list, prompt: str, paths: dict) -> dict:
    return {
        "mode": args.mode, "width": int(args.width), "height": int(args.height), "num_frames": int(args.num_frames),
        "latent_frames": latent_frames(int(args.num_frames)), "seed": int(args.seed), "steps": int(args.steps),
        "shift_video": float(args.shift_video), "shift_audio": float(args.shift_audio),
        "frame_rate": JOB_FRAME_RATE,
        "load": {"blocks_on_gpu": LOAD_BLOCKS_ON_GPU, "attention_backend": LOAD_ATTENTION,
                 "block_swap_prefetch": LOAD_PREFETCH, "deterministic": LOAD_DETERMINISTIC},
        "job": {"attention_backend": JOB_ATTENTION, "block_swap_prefetch": JOB_PREFETCH,
                "vram16_mode": JOB_VRAM16, "keep_resident": JOB_KEEP_RESIDENT, "loras": [],
                "negative_prompt": "", "turbo": False},
        "env": {k: os.environ.get(k) for k in ("TORCH_COMPILE_DISABLE", "PYTHONPATH", "PYTHONUTF8",
                                               "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED")},
        "T": args.T, "C": args.C, "L": args.L, "inject_clean": float(args.inject_clean) if args.L is not None else None,
        "paths": paths, "weights": weights,
        "prompt_file": args.prompt_file, "prompt": prompt, "prompt_sha": _sha_text(prompt),
        "keyframes": [{k: v for k, v in kf.items() if k != "abs"} | {"file_sha256": _sha_file(kf["abs"])}
                      for kf in keyframes],
        "references": [{k: v for k, v in r.items() if k != "abs"} for r in refs],
    }


# -- media checks (the gate) -----------------------------------------------------------------------


def stream_md5(mp4: Path, which: str) -> str | None:
    """MD5 of one stream's packets, the container left out (the gate's measure):
    ``ffmpeg -v error -i <mp4> -map 0:<v|a>:0 -c copy -f md5 -`` -> ``MD5=<hex>``. None = no such stream."""
    exe = str(FFMPEG) if FFMPEG.is_file() else shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found (tools/ffmpeg/bin or PATH)")
    proc = subprocess.run([exe, "-v", "error", "-i", str(mp4), "-map", f"0:{which}:0", "-c", "copy", "-f", "md5", "-"],
                          capture_output=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        if "matches no streams" in (proc.stderr or ""):
            return None
        raise RuntimeError(f"ffmpeg md5 ({which}) failed: {proc.stderr.strip()}")
    line = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    return line[0].split("=", 1)[1] if line[0].startswith("MD5=") else None


def mp4_record(path: Path) -> dict:
    return {"path": _rel(path), "sha256": _sha_file(path), "bytes": path.stat().st_size,
            "video_md5": stream_md5(path, "v"), "audio_md5": stream_md5(path, "a")}


# -- the pipeline ------------------------------------------------------------------------------------


def make_pipeline(args, rec: dict):
    """worker._do_load's order (worker.py:384-410): probe_sage, the attention rule, H3LoadOptions, load()."""
    import torch

    from engine.transformer.sage_attention_service import probe_sage
    from engine_h3.pipeline_h3 import H3LoadOptions, H3Paths, H3Pipeline

    rec["sage_available"] = bool(probe_sage())
    if JOB_ATTENTION == "sage" and not rec["sage_available"]:
        # The worker would fall back to sdpa (worker.py:355-361); here that would be another condition.
        raise RuntimeError("SageAttention is not available in this venv (the plan's condition is sage)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    options = H3LoadOptions(blocks_on_gpu=LOAD_BLOCKS_ON_GPU, attention_backend=LOAD_ATTENTION,
                            block_swap_prefetch=LOAD_PREFETCH, deterministic=LOAD_DETERMINISTIC)
    paths = H3Paths(transformer=str(_abs(args.transformer)), **{key: str(ROOT / rel) for key, rel in PATHS.items()})
    pipe = H3Pipeline(paths, options, device=torch.device("cuda"), progress=_progress, log=_log)
    rec["build_report"] = pipe.load()
    rec["device_name"] = torch.cuda.get_device_name(0)
    rec["numerics"] = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "float32_matmul_precision": torch.get_float32_matmul_precision(),
                       "cudnn_deterministic": torch.backends.cudnn.deterministic,
                       "cudnn_benchmark": torch.backends.cudnn.benchmark}
    return pipe


def make_request(args, prompt: str, keyframes: list, refs: list, out_path: Path):
    """What worker._generate_request builds from the adapter's payload (worker.py:690-713, 740-741)."""
    from engine_h3 import pipeline_h3 as P

    return P.H3GenerateRequest(
        prompt=prompt, seed=int(args.seed), width=int(args.width), height=int(args.height),
        num_frames=int(args.num_frames), output_path=str(out_path), frame_rate=JOB_FRAME_RATE,
        num_inference_steps=int(args.steps), shift_video=float(args.shift_video),
        shift_audio=float(args.shift_audio),
        images=tuple(P.H3ImageInput(path=k["abs"], frame_idx=int(k["frame"])) for k in keyframes),
        keep_resident=JOB_KEEP_RESIDENT, attention_backend=JOB_ATTENTION, block_swap_prefetch=JOB_PREFETCH,
        vram16_mode=JOB_VRAM16, loras=(), source_audio=None,
        references=tuple(P.H3ReferenceInput(kind=r["kind"], path=r["abs"], slot=int(r["slot"])) for r in refs),
        negative_prompt="", vsf_scale=1.5, outpaint=None, inpaint=None,
    )


def run_product(pipe, req) -> dict:
    """generate() as the worker calls it (worker.py:768). Nothing wrapped, hooked or replaced."""
    t0 = time.perf_counter()
    res = pipe.generate(req)
    return {"result": res, "wall_seconds": round(time.perf_counter() - t0, 2)}


def build_fix(cond_clean, keyframes: list, lat: list, m: float, shape_v, gh: int, gw: int):
    """L: ``(clean, per_frame)``; ``clean`` = zeros of the video state's shape (CPU float32) with each key
    frame's own latent (its ``_encode_image`` rows, unpatchified) at its latent index."""
    import torch

    from engine_h3.dit import unpatchify_video

    grid = gh * gw
    clean = torch.zeros(tuple(shape_v), dtype=torch.float32)
    per = [1.0] * int(shape_v[2])
    for i, li in enumerate(lat):
        rows = cond_clean[i * grid:(i + 1) * grid].detach().to("cpu", torch.float32)
        z = unpatchify_video(rows, 1, gh, gw)               # [1, 24, 1, H/16, W/16]
        clean[:, :, li:li + 1] = z
        per[li] = float(m)
    return clean, per


def run_harness(pipe, req, args, keyframes: list, paths: dict, lat_path: Path) -> dict:
    """generate()'s steps (pipeline_h3.py:871-1192) for this job's case (key frames + references; no A2V,
    Outpainting, Inpainting, VSF or LoRA), in its order, with T / C / L switched."""
    import torch

    from engine_h3 import pipeline_h3 as P
    from engine_h3.constants import REFERENCE_KINDS, VAE_SPATIAL
    from engine_h3.denoise_mask import video_mask
    from engine_h3.image_cond import load_canvas_image
    from engine_h3.text_encoder import RefMedia

    t_on, c_on, l_m = paths["T"], paths["C"], paths["L"]
    keep = bool(req.keep_resident)
    started = time.perf_counter()
    phases = P._Phases(cuda=pipe.device.type == "cuda")
    out: dict = {}
    ok = False
    try:
        P.validate_request(req)                                                         # :888
        images = sorted(req.images or (), key=lambda img: int(img.frame_idx))           # :892
        cond_frame_idx = [int(img.frame_idx) for img in images]
        canvases, source_size = [], []
        for img in images:                                                              # :899-902
            canvas, size = load_canvas_image(img.path, int(req.width), int(req.height))
            canvases.append(canvas)
            source_size.append([int(size[0]), int(size[1])])
        last = int(req.num_frames) - 1
        vision_images = [c for n, c in zip(cond_frame_idx, canvases) if n in (0, last)]  # :906
        ref_media = pipe._read_references(req)                                          # :911
        if ref_media:
            vision_images = []                                                          # :912-913
        has_cond = bool(canvases)                                                       # :1008
        encode_total = P.encode_steps(has_cond=has_cond, has_audio=False, has_refs=bool(ref_media))  # :1011
        te_refs_in = pipe._te_references(ref_media)
        t_items = []
        if t_on:
            # T: the key frame behind the references, as the next picture slot (SKEBA's form).
            te_refs_in = list(te_refs_in) + [RefMedia("picture", int(paths["T_slot"]), image=canvases[0])]
            t_items = [{"kind": "picture", "slot": int(paths["T_slot"]), "frame": cond_frame_idx[0],
                        "T_item": True}]
        with torch.no_grad():
            with phases.phase("encode") as entry:                                       # :1018-1028
                text_hidden, text_tags = pipe._encode(
                    req.prompt, phases, images=vision_images, has_cond=has_cond,
                    has_audio=False, refs=te_refs_in,
                )
                entry["text_tokens"] = int(text_hidden.shape[0])
            del te_refs_in
            cond_clean = None
            ref_rows = None
            visual_refs = any(m["kind"] != "audio" for m in ref_media)                  # :1037
            video_enc = None
            if canvases:
                with phases.phase("image_encode") as entry:                             # :1040-1046
                    if visual_refs:
                        video_enc = pipe._build_video_encoder()
                    cond_clean = pipe._encode_image(canvases, entry, 4, encode_total, enc=video_enc)
                    entry["vision_tokens"] = int(phases.data["encode"].get("te_read", {}).get("vision_tokens", 0))
                    entry["canvas"] = [int(req.width), int(req.height)]
                    entry["source_size"] = source_size
            del canvases, vision_images
            references_summary = None
            if ref_media:
                with phases.phase("reference_encode") as entry:                         # :1048-1056
                    if visual_refs and video_enc is None:
                        video_enc = pipe._build_video_encoder()
                    te_refs = phases.data["encode"].get("te_read", {}).get("references", [])
                    index = 3 + int(bool(cond_clean is not None)) + 1
                    ref_rows = pipe._encode_references(ref_media, entry, index, encode_total, video_enc, te_refs)
                    video_enc = None
                    P._free_cuda()
                del ref_media
                counts = {kind: sum(1 for s, _r in ref_rows if s.kind == kind) for kind in REFERENCE_KINDS}
                references_summary = {"pictures": counts["picture"], "videos": counts["video"],
                                      "audios": counts["audio"]}
            # L: the fixed latents from the same key-frame rows (no new encode, no draw).
            t_lat = latent_frames(int(req.num_frames))
            gh, gw = int(req.height) // SPATIAL_FACTOR, int(req.width) // SPATIAL_FACTOR
            lh, lw = int(req.height) // VAE_SPATIAL, int(req.width) // VAE_SPATIAL
            video_fix = None
            l_check: dict = {}
            fix_kw: dict = {}
            if l_m is not None:
                shape_v = (1, 24, t_lat, lh, lw)
                video_fix = build_fix(cond_clean, keyframes, paths["L_latents"], l_m, shape_v, gh, gw)
                # The rows _denoise will hold at m = 0, counted by the engine's own builder on the CPU.
                probe = video_mask(video_fix[0], video_fix[1], grid=gh * gw, noise=torch.zeros(shape_v),
                                   inject_clean=float(args.inject_clean))
                l_check["mask_zero_rows"] = 0 if probe is None else int((probe.rows == 0).sum())
                l_check["mask_rows"] = 0 if probe is None else int(probe.rows.numel())
                l_check["per_frame"] = list(video_fix[1])
                del probe
                fix_kw = {"video_fix": video_fix, "video_inject_clean": float(args.inject_clean)}
            # C off: both empty (pipeline_h3.py:1807-1814 checks the rows against the indices).
            dit_cond = cond_clean if c_on else None
            dit_idx = cond_frame_idx if c_on else []
            with phases.phase("denoise") as entry:                                      # :1117-1124
                video_lat, audio_lat, prefetch_used = pipe._denoise(
                    req, text_hidden, text_tags, phases, entry,
                    cond_clean=dit_cond, cond_frame_idx=dit_idx, keep=keep,
                    lora_report=[], audio_fix=None, refs=ref_rows,
                    negative_hidden=None, vsf_scale=float(req.vsf_scale), vsf_report={},
                    **fix_kw,
                )
            if l_m is not None:
                clean = video_fix[0]
                for li in paths["L_latents"]:
                    got = video_lat[:, :, li:li + 1].detach().to("cpu", torch.float32)
                    want = clean[:, :, li:li + 1]
                    l_check.setdefault("latents", []).append({
                        "latent": li, "equal": bool(torch.equal(got, want)),
                        "max_abs": float((got - want).abs().max()),
                    })
                # The mask _denoise itself used: its record (pipeline_h3.py:2010-2017, mask.video.{rows,min,max}).
                used = ((phases.data.get("denoise") or {}).get("mask") or {}).get("video")
                l_check["denoise_mask_video"] = used
                used_min = None if not isinstance(used, dict) else used.get("min")
                # m = 0 (stage A): every key-frame latent's rows held, and they end as the key frame exactly.
                # m > 0 is not frozen: the values are recorded only.
                if l_m == 0.0:
                    l_check["ok"] = (l_check["mask_zero_rows"] == len(paths["L_latents"]) * gh * gw
                                     and used_min == 0.0
                                     and all(x["equal"] for x in l_check["latents"]))
                else:
                    l_check["ok"] = (l_check["mask_zero_rows"] == 0 and used_min is not None
                                     and abs(float(used_min) - float(l_m)) < 1e-6)   # float32 in the mask
            # The final latents, saved as they leave the denoise (a CPU copy; the decode gets the originals).
            v_cpu, a_cpu = video_lat.detach().cpu(), audio_lat.detach().cpu()
            torch.save({"video": v_cpu, "audio": a_cpu, "name": args.name}, lat_path)
            out["latents"] = {"path": _rel(lat_path), "v_shape": list(v_cpu.shape), "a_shape": list(a_cpu.shape),
                              "v_sha256": _sha_tensor(v_cpu), "a_sha256": _sha_tensor(a_cpu)}
            del v_cpu, a_cpu
            del text_hidden, text_tags, cond_clean, dit_cond, ref_rows, video_fix, fix_kw   # :1127-1128
            P._free_cuda()
            with phases.phase("decode") as entry:                                       # :1150-1158
                size_bytes = pipe._decode(req, video_lat, audio_lat, entry, keep=keep, source_wave=None)
            del video_lat, audio_lat
        ok = True
    finally:
        if not (ok and keep):                                                           # :1186-1192
            pipe.release_resident()
        P._free_cuda()
    mem = P._mem_gib()
    h3 = {"phases": phases.data}
    if references_summary is not None:
        h3["references"] = references_summary
    out.update({
        "seconds": round(time.perf_counter() - started, 2),
        "peak_vram_mb": phases.peak_alloc_mib if phases.cuda else None,
        "peak_vram_reserved_mb": phases.peak_reserved_mib if phases.cuda else None,
        "rss_peak_gib": mem.get("rss_peak"), "commit_peak_gib": mem.get("commit_peak"),
        "size_bytes": int(size_bytes), "attention_used": pipe._attention_for(req),
        "block_swap_prefetch_used": prefetch_used, "h3": h3, "T_items": t_items, "L_check": l_check or None,
    })
    return out


def summary(res: dict) -> dict:
    """The few fields the metrics read, from a generate() result or a harness record (the same shape)."""
    ph = ((res or {}).get("h3") or {}).get("phases") or {}
    return {
        "seconds": {k: (ph.get(k) or {}).get("seconds") for k in
                    ("encode", "image_encode", "reference_encode", "denoise", "decode")},
        "peak_vram_mb": {k: (ph.get(k) or {}).get("peak_vram_mb") for k in
                         ("encode", "image_encode", "reference_encode", "denoise", "decode")},
        "text_tokens": (ph.get("encode") or {}).get("text_tokens"),
        "te_references": ((ph.get("encode") or {}).get("te_read") or {}).get("references"),
        "sequence": (ph.get("denoise") or {}).get("sequence"),
        "mask": (ph.get("denoise") or {}).get("mask"),
        "dit_source": (ph.get("denoise") or {}).get("dit_source"),
        "peak_vram_mb_job": (res or {}).get("peak_vram_mb"),
    }


# -- the run -----------------------------------------------------------------------------------------


def one_run(pipe, args, mode: str, name: str, prompt: str, keyframes: list, refs: list, paths: dict) -> dict:
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{name}.mp4"
    req = make_request(args, prompt, keyframes, refs, mp4)
    rec: dict = {"name": name, "mode": mode, "paths": paths}
    _log(f"{name}: {mode} | TE: {paths['text_encoder']} | DiT: {paths['dit']}")
    if mode == "product":
        got = run_product(pipe, req)
        rec["result"] = got["result"]
        rec["wall_seconds"] = got["wall_seconds"]
        rec["summary"] = summary(got["result"])
    else:
        got = run_harness(pipe, req, args, keyframes, paths, out_dir / f"{name}.pt")
        rec["result"] = got
        rec["summary"] = summary(got)
        if got.get("L_check") is not None and not got["L_check"].get("ok"):
            rec["L_check_failed"] = True
    rec["mp4"] = mp4_record(mp4)
    return rec


def expected_rows(args, keyframes: list, refs: list, mode: str, paths: dict) -> dict:
    """--dry-run only, on the CPU (no model): the rows of each kind the DiT should get, by the engine's own
    size rules -- cond = key frames x grid (pipeline_h3.py:1808), a picture reference
    (H/32)*(W/32) of its ``match_size`` (reference.py:77-83, the rows of RefShape.picture), an audio reference
    2 * ceil(samples / 800) of the waveform ``_read_references`` would read (pipeline_h3.py:1381-1391,
    1466; RefShape.audio), video = T' x grid. The text rows need the text encoder: not counted."""
    from PIL import Image, ImageOps

    from engine_h3.constants import AUDIO_HOP, AUDIO_SAMPLE_RATE
    from engine_h3.reference import REF_CLIP_MAX_SEC, match_size
    from engine_h3.source_audio import load_source_waveform

    W, H = int(args.width), int(args.height)
    grid = (H // SPATIAL_FACTOR) * (W // SPATIAL_FACTOR)
    c_on = paths["implied"]["C"] if mode == "product" else paths["C"]
    ref_video = ref_audio = 0
    audio_info = []
    for r in refs:
        if r["kind"] == "picture":
            with Image.open(r["abs"]) as opened:
                w0, h0 = ImageOps.exif_transpose(opened).size
            w, h = match_size(w0, h0, W, H)
            ref_video += (h // SPATIAL_FACTOR) * (w // SPATIAL_FACTOR)
        else:
            read = max(1, int(REF_CLIP_MAX_SEC * AUDIO_SAMPLE_RATE))
            wave, _info = load_source_waveform(r["abs"], keep_samples=1, read_samples=read)
            samples = int(wave.shape[-1])
            ref_audio += 2 * math.ceil(samples / AUDIO_HOP)
            audio_info.append({"slot": r["slot"], "samples": samples, "seconds": round(samples / AUDIO_SAMPLE_RATE, 3)})
    out = {"cond": len(keyframes) * grid if c_on else 0, "ref_video": ref_video, "ref_audio": ref_audio,
           "video": latent_frames(int(args.num_frames)) * grid,
           "L_zero_rows": len(paths.get("L_latents") or []) * grid if paths.get("L") == 0.0 else None}
    if audio_info:
        out["ref_audio_read"] = audio_info
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=("product", "harness", "gate"), required=True)
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--keyframe", action="append", default=[], help="FRAME=PATH (repeatable)")
    ap.add_argument("--ref-image", action="append", default=[], help="PATH (repeatable; <Picture 1>, 2, ...)")
    ap.add_argument("--ref-audio", action="append", default=[], help="PATH (repeatable; <Audio 1>, 2, ...)")
    ap.add_argument("--T", choices=("on", "off"), default="off")
    ap.add_argument("--C", choices=("on", "off"), default="on")
    ap.add_argument("--L", type=float, default=None, help="mask value on the key frame's latent (off if absent)")
    ap.add_argument("--inject-clean", type=float, default=DEFAULT_INJECT_CLEAN)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--num-frames", type=int, default=124)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--shift-video", type=float, default=DEFAULT_SHIFT_VIDEO)
    ap.add_argument("--shift-audio", type=float, default=DEFAULT_SHIFT_AUDIO)
    ap.add_argument("--gate-names", default="r01_010_harness,r02_010_product",
                    help="--mode gate: the two runs' names (harness first)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def run(args) -> int:
    report = Path(args.report) if args.report else HERE / f"{args.name}{'.dryrun' if args.dry_run else ''}.json"
    rec: dict = {"name": args.name, "dry_run": bool(args.dry_run), "runs": []}
    ok, exit_code = False, 1
    pipe = None
    t_start = time.perf_counter()
    try:
        latent_frames(int(args.num_frames))
        if int(args.width) % SPATIAL_FACTOR or int(args.height) % SPATIAL_FACTOR:
            raise ValueError(f"width / height must be multiples of {SPATIAL_FACTOR}")
        prompt = _abs(args.prompt_file).read_text(encoding="utf-8-sig").strip()
        if not prompt:
            raise ValueError(f"empty prompt: {args.prompt_file}")
        keyframes = parse_keyframes(args.keyframe, int(args.num_frames))
        refs = parse_refs(args.ref_image, args.ref_audio)
        weights = weights_record(args.transformer)
        paths = check_args(args, keyframes, refs)
        rec["condition"] = condition_record(args, weights, keyframes, refs, prompt, paths)
        if args.mode == "gate":
            names = [s.strip() for s in args.gate_names.split(",")]
            if len(names) != 2 or not all(names):
                raise ValueError(f"--gate-names must be two names ({args.gate_names!r})")
            plan = [("harness", names[0]), ("product", names[1])]
            product_paths = check_args(argparse.Namespace(**{**vars(args), "mode": "product"}), keyframes, refs)
        else:
            plan = [(args.mode, args.name)]
            product_paths = paths
        rec["plan"] = [{"mode": m, "name": n, "paths": product_paths if m == "product" else paths} for m, n in plan]
        for item in rec["plan"]:
            _log(f"plan {item['name']}: {item['mode']} | weights {Path(weights['transformer']['path']).name} | "
                 f"keyframes {[(k['frame'], k['path']) for k in keyframes]} | "
                 f"refs {[(r['kind'], r['slot'], r['path']) for r in refs]}")
            _log(f"    text encoder: {item['paths']['text_encoder']}")
            _log(f"    DiT         : {item['paths']['dit']}")
        if args.dry_run:
            # Dry run only (a real run never reads the media here): the rows the DiT should get.
            for item in rec["plan"]:
                item["expected_rows"] = expected_rows(args, keyframes, refs, item["mode"], item["paths"])
                _log(f"    rows (expected): {item['expected_rows']}")
            ok, exit_code = True, 0
            return exit_code
        pipe = make_pipeline(args, rec)
        for (mode, name), item in zip(plan, rec["plan"]):
            r = one_run(pipe, args, mode, name, prompt, keyframes, refs, item["paths"])
            rec["runs"].append(r)
            # Per run, the archive also gets the run's own report (the metrics read these).
            if args.mode == "gate":
                rec.setdefault("run_reports", []).append(str(HERE / f"{name}.json"))
        problems = [r["name"] for r in rec["runs"] if r.get("L_check_failed")]
        if problems:
            rec["error"] = f"L self-check failed: {problems}"
            exit_code = 1
        elif args.mode == "gate":
            a, b = rec["runs"][0]["mp4"], rec["runs"][1]["mp4"]
            gate = {"video_md5_equal": a["video_md5"] == b["video_md5"] and a["video_md5"] is not None,
                    "audio_md5_equal": a["audio_md5"] == b["audio_md5"] and a["audio_md5"] is not None,
                    "file_sha256_equal": a["sha256"] == b["sha256"],
                    "sequence_equal": rec["runs"][0]["summary"]["sequence"] == rec["runs"][1]["summary"]["sequence"],
                    "text_tokens_equal": rec["runs"][0]["summary"]["text_tokens"]
                    == rec["runs"][1]["summary"]["text_tokens"]}
            gate["pass"] = gate["video_md5_equal"] and gate["audio_md5_equal"]
            rec["gate"] = gate
            _log(f"gate: {gate}")
            ok = gate["pass"]
            exit_code = 0 if ok else 2
        else:
            ok, exit_code = True, 0
    except BaseException as exc:  # noqa: BLE001 -- the reason goes into the JSON
        rec["oom"] = "out of memory" in str(exc).lower()
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        exit_code = 1
    finally:
        if pipe is not None:
            try:
                pipe.close()
            except Exception as exc:  # noqa: BLE001
                rec["close_error"] = repr(exc)
        rec["seconds_total"] = round(time.perf_counter() - t_start, 2)
        rec["ok"] = ok
        rec["exit_code"] = exit_code
        report.parent.mkdir(parents=True, exist_ok=True)
        bad: list = []
        safe = _json_safe(rec, bad)
        if bad:
            safe["nonfinite_values_nulled"] = len(bad)
        report.write_text(json.dumps(safe, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        if args.mode == "gate" and not args.dry_run:
            for r in rec["runs"]:
                one = {"name": r["name"], "gate_of": args.name, "ok": ok, "condition": rec.get("condition"),
                       "runs": [r]}
                (HERE / f"{r['name']}.json").write_text(
                    json.dumps(_json_safe(one, []), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(json.dumps({k: rec.get(k) for k in ("name", "ok", "exit_code", "error", "gate", "seconds_total")},
                         ensure_ascii=False, default=str), flush=True)
        if rec.get("traceback"):
            print(rec["traceback"], flush=True)
    return exit_code


def main() -> int:
    # The worker's root logger (worker.py:222-234): INFO to stderr, so the engine's logger lines show.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="[h3] %(name)s: %(message)s")
    return run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
