r"""§1-106 Chained (latent carry-over) measurement for MiniMax H3 FL2VA: one condition = one process.

Plan: velvety-hopping-hartmanis.md (§4 method, §5 conditions, §6.1 this script, §6.2 the JSON, §7 the
self-check). Measurement only: the product code is NOT changed. The script drives H3Pipeline's own
stages (``_encode`` / ``_denoise`` / ``_decode``) clip by clip and builds the carry-over itself.

Run from the repo root (run_c.ps1 does the list in the plan's order):

    set PYTHONPATH=%CD%
    set PYTHONUTF8=1
    .venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-chain-2026-10-29\c_chain.py --name c1_base ^
        --overlap 22 --strength 1.0 --inject noise --attention sage --prompts <archive>\prompts_c.json ^
        --decode both --save-latents --report <json>
    ... --selfcheck            (CPU only, no weights; the last line is SELFCHECK PASS)

Method (plan §4): clip i (F frames, T' = latent_frames(F), A = audio_latents(F)) is generated with seed
``seed + i`` and its own prompt; RoPE time starts at 0 in every clip. Clip i >= 1 gets
``video_fix = (clean_v, [1-s]*k + [1]*(T'-k))`` with the previous clip's last k = 5m+2 video latents in
``clean_v[:, :, :k]`` and ``audio_fix = (clean_a, [1-s]*ka + [1]*(A-ka))`` with the last ka latents of the
CONCATENATED audio in ``clean_a[:, :, :ka]`` (the rest is 0), where the overlap is O = 17m+5 frames,
S_i = i (F - O), J_i = S_i + O and ka_i = A_acc - round(S_i * 5/3) (A_acc = the concatenated audio's
actual length). Concatenation: video ``cat(prev, cur[:, :, k:])``, audio ``cat(prev, cur[:, :, ka:])``.
Strength 0 (C7, the control) = no mask, O = 0, the clips laid side by side.

Decodes (all through decode_frames / decode_wave, built like ``_decode``, then ``write_mp4``):
  once     (a) the concatenated latents decoded as one
  perclip  (b) every clip decoded alone; clip i >= 1 loses its first O frames and the matching audio
  xfade    (c) 0 < s < 1 only: the overlap latents (video k, audio ka) cross-faded with
               w_j = (j+1)/(n+1), then decoded as one
``pipe._decode`` is used only for the fingerprint (clip 0 alone, C0).

Inject (plan §4, H2): ``--inject clean`` sets ``engine_h3.denoise_mask.VIDEO_INJECT_CLEAN = 1.0``
(``video_mask`` reads the module variable when it is called; ``_denoise`` imports the function), checks
it on a small CPU tensor and records it.

Exit code 0 = done; 1 = failed (exception, OOM, a strength-1 head that is not the previous tail, a
refused argument); 2 = the fingerprint or the clip-0 recheck did not match. The JSON is always written.
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # the self-check must leave no __pycache__

# Before torch is imported (plan §6: same as the worker). The self-check hides the GPU itself.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
if "--selfcheck" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import analyze_chain as AC  # noqa: E402
from engine_h3 import denoise_mask as DM  # noqa: E402
from engine_h3 import pipeline_h3 as P  # noqa: E402
from engine_h3.constants import AUDIO_SAMPLE_RATE, FRAME_RATE, audio_latents, latent_frames  # noqa: E402

OUT_ROOT = ROOT / "outputs/h3-chain-2026-10-29"
#: Same files as Docs/Outputs-archive/h3-te-vram-2026-10-29/g_i5_t2v_fingerprint.cmd (FL2VA).
PATHS = {
    "transformer": "models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoder": "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "video_vae": "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
    "tokenizer_dir": "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
}
#: The fingerprint prompt (Docs/Outputs-archive/h3-gate3/g2_selftest_g5prompt.cmd) = the common text.
FP_PROMPT = (
    "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in "
    "puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>"
)
FP_SHA = "0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9"
#: Load-time values (plan §5): H3LoadOptions' defaults, as the selftest that took the fingerprint.
BLOCKS_ON_GPU = 8
PREFETCH = True
DETERMINISTIC = False
#: g_i5_t2v_fingerprint.cmd's conditions (plan §6.1 FP_COND).
FP_COND = {
    "width": 640, "height": 384, "clip_frames": 124, "seed": 42, "steps": 20, "shift_video": 12.0,
    "shift_audio": 3.0, "attention": "sdpa", "vram16": True, "prefetch": True, "blocks_on_gpu": 8,
    "deterministic": False,
}
#: What load_clip0 compares with the C1 report (plan §5).
REUSE_KEYS = ("width", "height", "clip_frames", "seed", "steps", "shift_video", "shift_audio", "attention",
              "vram16", "prefetch", "blocks_on_gpu", "deterministic", "weights")
LATENT_C, AUDIO_C = 24, 32
SAMPLES_PER_FRAME = Fraction(AUDIO_SAMPLE_RATE, FRAME_RATE)   # 4000/3

#: Seams for the self-check (fake decoders / writer); None = the real ones.
HOOKS: dict = {"video_decoder": None, "audio_vae": None, "write_mp4": None}


def _log(msg: str) -> None:
    print(f"[c_chain] {msg}", flush=True)


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


def _sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _round_frac(x: Fraction) -> int:
    """Python's round on an exact value (the thirds here never tie)."""
    return round(x)


def _mb(n) -> float:
    return round(n / 2**20, 1)


# -- geometry (plan §4) -------------------------------------------------------------------


def geometry(clip_frames: int, overlap: int | None, clips: int, strength: float) -> dict:
    """The chain's frame / latent bookkeeping (pure; plan §4's table is the expected value).

    Strength 0 is O = 0 (the clips side by side, no audio overlap); ``overlap`` is then ignored.
    The audio error of a seam (in latents) is ``start_i - S_i * 5/3`` with ``start_i`` = where clip i's
    audio begins in the concatenation (= round(S_i * 5/3) when s > 0): the placement error, recorded
    signed as the fixed head's shift as well (plan §4). ``fixed_end_vs_video`` (informative) is the
    end of the fixed audio head minus the end of the fixed video head, J_i * 5/3, in latents.
    """
    F, N, s = int(clip_frames), int(clips), float(strength)
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"strength must be in [0, 1] ({strength})")
    if N < 2:
        raise ValueError(f"clips must be 2 or more ({clips})")
    T = latent_frames(F)                  # refuses a frame count off 17n+5
    if F < P.MIN_NUM_FRAMES:
        raise ValueError(f"clip frames must be {P.MIN_NUM_FRAMES} or more ({F})")
    A = audio_latents(F)
    if s == 0.0:
        O, m, k = 0, None, 0
    else:
        O = 22 if overlap is None else int(overlap)
        if O < 5 or (O - 5) % 17:
            raise ValueError(f"overlap must be 17m+5 frames ({O})")
        m = (O - 5) // 17
        k = 5 * m + 2
        if O >= F:
            raise ValueError(f"overlap {O} must be shorter than the clip ({F})")
        if 2 * k > T:
            raise ValueError(f"overlap latents k={k}: 2k must be <= T'={T}")
    S = [i * (F - O) for i in range(1, N)]
    J = [x + O for x in S]
    ka, err, end_vs_video = [], [], []
    a_acc = A
    for si, ji in zip(S, J):
        start = a_acc if s == 0.0 else _round_frac(Fraction(5 * si, 3))
        kai = a_acc - start
        if not 0 <= kai <= A:
            raise ValueError(f"audio overlap ka={kai} out of [0, {A}] at S={si}")
        ka.append(kai)
        err.append(float(start - Fraction(5 * si, 3)))
        end_vs_video.append(None if s == 0.0 else float(a_acc - Fraction(5 * ji, 3)))
        a_acc += A - kai
    F_total = F + (N - 1) * (F - O)
    T_total = T + (N - 1) * (T - k)
    a_expected = _round_frac(Fraction(5 * F_total, 3))
    if s > 0.0 and T_total != latent_frames(F_total):
        raise AssertionError(f"T_total {T_total} != latent_frames({F_total})")
    return {
        "F": F, "T": T, "A": A, "N": N, "strength": s, "O": O, "m": m, "k": k, "S": S, "J": J, "ka": ka,
        "F_total": F_total, "T_total": T_total, "A_concat": a_acc, "A_expected": a_expected,
        "audio_latent_excess": a_acc - a_expected,
        # strength 0 has no overlap, so the three audio-error fields mean nothing there: None
        "audio_place_error_latents": None if s == 0.0 else [abs(e) for e in err],
        "audio_head_shift_latents": None if s == 0.0 else err,
        "audio_fixed_end_vs_video_latents": None if s == 0.0 else end_vs_video,
    }


# -- inject patch, masks, latents ----------------------------------------------------------


def apply_inject(mode: str) -> dict:
    """noise: nothing. clean: VIDEO_INJECT_CLEAN = 1.0. Then the real video_mask on a small CPU tensor."""
    before = float(DM.VIDEO_INJECT_CLEAN)
    if mode == "clean":
        DM.VIDEO_INJECT_CLEAN = 1.0
    elif mode != "noise":
        raise ValueError(mode)
    g = torch.Generator().manual_seed(7)
    clean = torch.randn((1, 2, 3, 2, 2), generator=g)
    noise = torch.randn((1, 2, 3, 2, 2), generator=g)
    mask = DM.video_mask(clean, [0.0, 1.0, 1.0], grid=4, noise=noise)
    equals_clean = bool(torch.equal(mask.inject, clean))
    expected = DM.VIDEO_INJECT_CLEAN * clean + (1.0 - DM.VIDEO_INJECT_CLEAN) * noise
    rec = {"mode": mode, "before": before, "after": float(DM.VIDEO_INJECT_CLEAN),
           "video_mask_reads_patched": bool(torch.equal(mask.inject, expected)),
           "inject_equals_clean": equals_clean}
    if mode == "clean" and not (equals_clean and rec["video_mask_reads_patched"]):
        raise RuntimeError(f"inject patch did not reach video_mask: {rec}")
    return rec


def build_fix(prev_v, prev_a_concat, shape_v, shape_a, k: int, ka: int, s: float):
    """``(video_fix, audio_fix)`` for a clip >= 1 (CPU float32 clean, zero-filled past the head);
    ``(None, None)`` at strength 0."""
    if float(s) == 0.0:
        return None, None
    m0 = 1.0 - float(s)
    T, A = int(shape_v[2]), int(shape_a[2])
    clean_v = torch.zeros(tuple(shape_v), dtype=torch.float32)
    clean_a = torch.zeros(tuple(shape_a), dtype=torch.float32)
    if k > 0:
        clean_v[:, :, :k] = prev_v[:, :, -k:].to(torch.float32)
    if ka > 0:
        clean_a[:, :, :ka] = prev_a_concat[:, :, -ka:].to(torch.float32)
    per_v = [m0] * k + [1.0] * (T - k)
    per_a = [m0] * ka + [1.0] * (A - ka)
    return (clean_v, per_v), (clean_a, per_a)


def concat_latents(vs: list, as_: list, geom: dict):
    """(a): video ``cat(prev, cur[:, :, k:])``, audio ``cat(prev, cur[:, :, ka_i:])``."""
    v, a = vs[0], as_[0]
    for i in range(1, len(vs)):
        v = torch.cat([v, vs[i][:, :, geom["k"]:]], dim=2)
        a = torch.cat([a, as_[i][:, :, geom["ka"][i - 1]:]], dim=2)
    return v, a


def _xfade(prev: torch.Tensor, cur: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0:
        return torch.cat([prev, cur], dim=2)
    shape = [1] * prev.dim()
    shape[2] = n
    w = (torch.arange(1, n + 1, dtype=torch.float32) / (n + 1)).view(shape)
    mid = (1.0 - w) * prev[:, :, -n:].float() + w * cur[:, :, :n].float()
    return torch.cat([prev[:, :, :-n], mid, cur[:, :, n:]], dim=2)


def xfade_latents(vs: list, as_: list, geom: dict):
    """(c): the overlap (video k, audio ka_i) cross-faded linearly, w_j = (j+1)/(n+1) towards the new clip."""
    v, a = vs[0], as_[0]
    for i in range(1, len(vs)):
        v = _xfade(v, vs[i], geom["k"])
        a = _xfade(a, as_[i], geom["ka"][i - 1])
    return v, a


def latent_record(v: torch.Tensor, a: torch.Tensor) -> dict:
    vf, af = v.float(), a.float()
    return {"v_shape": list(v.shape), "a_shape": list(a.shape), "v_dtype": str(v.dtype),
            "v_sha256": _sha_tensor(v), "a_sha256": _sha_tensor(a),
            "finite": bool(torch.isfinite(vf).all() and torch.isfinite(af).all()),
            "v_abs_mean": float(vf.abs().mean()), "v_std": float(vf.std()),
            "a_abs_mean": float(af.abs().mean()), "a_std": float(af.std())}


# -- frames and audio (b) ---------------------------------------------------------------------


def trim_concat_frames(frames: list, O: int) -> torch.Tensor:
    """(b): clip i >= 1 loses its first O frames."""
    return torch.cat([frames[0]] + [f[O:] for f in frames[1:]], dim=0)


def trim_concat_audio(waves: list, geom: dict) -> torch.Tensor:
    """(b): the chain so far is cut at round(J_i * 4000/3) samples and clip i continues from its local
    sample round(J_i * 4000/3) - round(S_i * 4000/3)."""
    out = waves[0]
    for i in range(1, len(waves)):
        S, J = geom["S"][i - 1], geom["J"][i - 1]
        cut = _round_frac(J * SAMPLES_PER_FRAME)
        start = cut - _round_frac(S * SAMPLES_PER_FRAME)
        if int(out.shape[-1]) < cut:
            raise RuntimeError(f"audio trim: {out.shape[-1]} samples before seam {i}, {cut} needed")
        out = torch.cat([out[:, :cut], waves[i][:, start:]], dim=1)
    return out


def overlap_agreement(frames: list, geom: dict) -> list:
    """(b), record only (plan §8): PSNR of clip i's first O frames against clip i-1's last O frames."""
    O, F = geom["O"], geom["F"]
    out = []
    if O <= 0:
        return out
    for i in range(1, len(frames)):
        psnr, identical = [], []
        for j in range(O):
            a = frames[i - 1][F - O + j].to(torch.float32)
            b = frames[i][j].to(torch.float32)
            mse = float(((a - b) ** 2).mean())
            identical.append(mse == 0.0)
            psnr.append(None if mse == 0.0 else 10.0 * float(torch.log10(torch.tensor(255.0 ** 2 / mse))))
        finite = [p for p in psnr if p is not None]
        out.append({"clip": i, "psnr": psnr, "identical": identical,
                    "min_psnr": min(finite) if finite else None,
                    "mean_psnr": sum(finite) / len(finite) if finite else None})
    return out


# -- seam metrics (in memory, before H.264; plan §6.1 / Minor 7) ---------------------------------


def frame_stats_torch(frames: torch.Tensor, batch: int = 8) -> dict:
    """Per-frame luma / chroma statistics and d(t) from uint8 RGB [F, H, W, 3] (analyze_chain's definitions)."""
    coef = torch.tensor([AC.Y_COEF, AC.CB_COEF, AC.CR_COEF], dtype=torch.float64).T   # [3 rgb, 3 ycc]
    out = {"frames": int(frames.shape[0]), "y_mean": [], "y_sq_mean": [], "chroma_mean": [], "d": []}
    prev = None
    for start in range(0, int(frames.shape[0]), batch):
        ycc = frames[start:start + batch].to(torch.float64) @ coef
        y = ycc[..., 0]
        chroma = ycc[..., 1:].pow(2).sum(-1).sqrt()
        for j in range(int(y.shape[0])):
            yj = y[j]
            out["y_mean"].append(float(yj.mean()))
            out["y_sq_mean"].append(float((yj * yj).mean()))
            out["chroma_mean"].append(float(chroma[j].mean()))
            if prev is not None:
                out["d"].append(float((yj - prev).abs().mean()))
            prev = yj
        del ycc, y, chroma
    return out


def seam_metrics(frames: torch.Tensor, geom: dict, *, cut: bool = False) -> dict:
    stats = frame_stats_torch(frames)
    return {"stats": stats, "summary": AC.summarize(stats, geom, cut=cut)}


# -- the pipeline ------------------------------------------------------------------------------


def weights_record() -> dict:
    out = {}
    for key, rel in PATHS.items():
        p = ROOT / rel
        out[key] = {"path": rel, "bytes": p.stat().st_size if p.is_file() else None}
    return out


def make_pipeline(args, rec: dict):
    from engine_h3.pipeline_h3 import H3LoadOptions, H3Paths, H3Pipeline

    options = H3LoadOptions(blocks_on_gpu=BLOCKS_ON_GPU, attention_backend=args.attention,
                            block_swap_prefetch=PREFETCH, deterministic=DETERMINISTIC)
    paths = H3Paths(**{key: str(ROOT / rel) for key, rel in PATHS.items()})
    pipe = H3Pipeline(paths, options, device=torch.device("cuda"), progress=_progress, log=_log)
    rec["build_report"] = pipe.load()
    return pipe


def clip_request(args, prompt: str, out_dir: Path):
    """Clip 0's request; clip i is ``dataclasses.replace(req, seed=seed+i, prompt=p_i)`` (plan §6.1)."""
    req = P.H3GenerateRequest(
        prompt=prompt, seed=int(args.seed), width=int(args.width), height=int(args.height),
        num_frames=int(args.clip_frames), output_path=str(out_dir / "clip.mp4"),
        num_inference_steps=int(args.steps), shift_video=float(args.shift_video),
        shift_audio=float(args.shift_audio), images=(), keep_resident=True, attention_backend=args.attention,
        block_swap_prefetch=PREFETCH, vram16_mode=bool(args.vram16),
    )
    P.validate_request(req)
    return req


def run_clip(pipe, req, video_fix, audio_fix):
    """One clip: encode (inside the "encode" phase, which ``_encode`` writes into) and denoise with
    keep=True; the latents come back on the CPU in float32."""
    phases = P._Phases(cuda=pipe.device.type == "cuda")
    with torch.no_grad():
        with phases.phase("encode") as e:
            hidden, tags = pipe._encode(req.prompt, phases, (), has_cond=False)
            e["text_tokens"] = int(hidden.shape[0])
        with phases.phase("denoise") as entry:
            v, a, used = pipe._denoise(req, hidden, tags, phases, entry, keep=True,
                                       video_fix=video_fix, audio_fix=audio_fix)
        v, a = v.cpu(), a.cpu()
    del hidden, tags
    P._free_cuda()
    enc, den = phases.data.get("encode", {}), phases.data.get("denoise", {})
    info = {
        "seconds": {"encode": enc.get("seconds"), "denoise": den.get("seconds")},
        "text_tokens": enc.get("text_tokens"), "dit_source": den.get("dit_source"),
        "sequence": den.get("sequence"), "m_per_step": den.get("m_per_step"), "mask": den.get("mask"),
        "vram16": den.get("vram16"), "block_swap_prefetch_used": used,
        "peak_vram_mb": {"encode": enc.get("peak_vram_mb"), "denoise": den.get("peak_vram_mb")},
        "peak_vram_reserved_mb": {"encode": enc.get("peak_vram_reserved_mb"),
                                  "denoise": den.get("peak_vram_reserved_mb")},
        "mem": P._mem_gib(), "phases": phases.data,
    }
    return v, a, info


def load_clip0(c1: dict, args, prompt0: str, weights: dict, shape_v, shape_a) -> dict:
    """``--reuse-clip0``: clip 0 from C1's saved latents, only if C1 ran under the same conditions
    and the file's latents hash to what C1 recorded (plan §5). Raises ValueError otherwise."""
    problems = []
    if not c1.get("ok"):
        problems.append("the C1 report is not ok")
    cond = c1.get("condition") or {}
    mine = condition_record(args, weights)
    for key in REUSE_KEYS:
        if cond.get(key) != mine.get(key):
            problems.append(f"{key}: C1 {cond.get(key)!r} != {mine.get(key)!r}")
    clips = c1.get("clips") or []
    c0 = clips[0] if clips else {}
    if c0.get("source") != "generated":
        problems.append(f"C1 clip 0 source {c0.get('source')!r} (generated expected)")
    if c0.get("prompt_sha") != _sha_text(prompt0):
        problems.append("clip 0 prompt sha differs from C1's")
    if problems:
        raise ValueError("reuse-clip0 refused: " + "; ".join(problems))
    path = c0.get("latents_path")
    if not path or not _abs(path).is_file():
        raise ValueError(f"reuse-clip0 refused: latents file missing ({path})")
    data = torch.load(_abs(path))
    v, a = data["video"], data["audio"]
    lat = c0.get("latents") or {}
    if _sha_tensor(v) != lat.get("v_sha256") or _sha_tensor(a) != lat.get("a_sha256"):
        raise ValueError("reuse-clip0 refused: the loaded latents' sha256 differs from the C1 report")
    if tuple(v.shape) != tuple(shape_v) or tuple(a.shape) != tuple(shape_a):
        raise ValueError(f"reuse-clip0 refused: shapes {list(v.shape)} / {list(a.shape)}")
    return {"video": v, "audio": a, "path": path, "seed": c0.get("seed")}


def condition_record(args, weights: dict) -> dict:
    return {
        "width": int(args.width), "height": int(args.height), "clip_frames": int(args.clip_frames),
        "clips": int(args.clips), "overlap_arg": args.overlap, "strength": float(args.strength),
        "inject": args.inject, "seed": int(args.seed), "steps": int(args.steps),
        "shift_video": float(args.shift_video), "shift_audio": float(args.shift_audio),
        "attention": args.attention, "vram16": bool(args.vram16), "prefetch": PREFETCH,
        "blocks_on_gpu": BLOCKS_ON_GPU, "deterministic": DETERMINISTIC, "decode": args.decode,
        "save_latents": bool(args.save_latents), "reuse_clip0": args.reuse_clip0,
        "fingerprint": bool(args.fingerprint), "recheck_clip0": bool(args.recheck_clip0),
        "prompts_file": args.prompts, "weights": weights,
    }


def fingerprint_problems(args, prompts: list) -> list:
    """``--fingerprint``: the run must be FP_COND with the fingerprint prompt for clip 0."""
    have = {"width": int(args.width), "height": int(args.height), "clip_frames": int(args.clip_frames),
            "seed": int(args.seed), "steps": int(args.steps), "shift_video": float(args.shift_video),
            "shift_audio": float(args.shift_audio), "attention": args.attention, "vram16": bool(args.vram16),
            "prefetch": PREFETCH, "blocks_on_gpu": BLOCKS_ON_GPU, "deterministic": DETERMINISTIC}
    out = [f"{k}: {have[k]!r} != {v!r}" for k, v in FP_COND.items() if have[k] != v]
    if prompts[0] != FP_PROMPT:
        out.append("clip 0 prompt is not the fingerprint prompt")
    if args.reuse_clip0:
        out.append("--reuse-clip0 with --fingerprint")
    return out


def load_prompts(path: str | None, n: int) -> list:
    """JSON list of strings; shorter than n repeats its last entry (plan §6.1)."""
    if path:
        items = json.loads(_abs(path).read_text(encoding="utf-8-sig"))
        if not isinstance(items, list) or not items or not all(isinstance(x, str) and x.strip() for x in items):
            raise ValueError(f"--prompts must be a JSON list of non-empty strings ({path})")
    else:
        items = [FP_PROMPT]
    return [items[min(i, len(items) - 1)] for i in range(n)]


# -- decoding -------------------------------------------------------------------------------------


def _new_video_decoder(pipe):
    if HOOKS["video_decoder"] is not None:
        return HOOKS["video_decoder"](pipe)
    from engine_h3.video_vae import H3VideoDecoder

    return H3VideoDecoder.from_file(pipe.paths.video_vae, device=pipe.device, dtype=torch.bfloat16)


def _new_audio_vae(pipe):
    if HOOKS["audio_vae"] is not None:
        return HOOKS["audio_vae"](pipe)
    from engine_h3.audio_vae import H3AudioVAE

    return H3AudioVAE.from_file(pipe.paths.audio_vae, device=pipe.device)


def _write(path: Path, frames, wave) -> int:
    if HOOKS["write_mp4"] is not None:
        return HOOKS["write_mp4"](str(path), frames, fps=FRAME_RATE, audio=wave, sample_rate=AUDIO_SAMPLE_RATE)
    from engine_h3.media_out import write_mp4

    return write_mp4(str(path), frames, fps=FRAME_RATE, audio=wave, sample_rate=AUDIO_SAMPLE_RATE)


def _cuda(pipe) -> bool:
    return pipe.device.type == "cuda"


def _peak_start(pipe) -> None:
    if _cuda(pipe):
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(pipe):
    return _mb(torch.cuda.max_memory_allocated()) if _cuda(pipe) else None


def decode_frames(pipe, v: torch.Tensor):
    """Video latents -> CPU uint8 frames, built as ``_decode`` does (resident or from_file -> device ->
    decode -> CPU -> kept as resident -> _free_cuda). Returns (frames, {seconds, peak_vram_mb})."""
    vae = (pipe._resident or {}).get("decoder")
    if vae is None:
        vae = _new_video_decoder(pipe)
    else:
        vae.to(pipe.device)
    _peak_start(pipe)
    t0 = time.perf_counter()
    try:
        frames = vae.decode(v.to(pipe.device), tiled=True)
    finally:
        vae.to("cpu")
        pipe._resident = {**(pipe._resident or {}), "decoder": vae}
        del vae
        P._free_cuda()
    return frames, {"seconds": round(time.perf_counter() - t0, 2), "peak_vram_mb": _peak_mb(pipe)}


def decode_wave(pipe, a: torch.Tensor):
    """Audio latents -> CPU float32 waveform [2, A*800], built as ``_decode`` does."""
    avae = (pipe._resident or {}).get("audio_vae")
    if avae is None:
        avae = _new_audio_vae(pipe)
    else:
        avae.to(pipe.device)
    _peak_start(pipe)
    t0 = time.perf_counter()
    try:
        wav = avae.decode(a.to(pipe.device)).float().cpu()
    finally:
        avae.to("cpu")
        pipe._resident = {**(pipe._resident or {}), "audio_vae": avae}
        del avae
        P._free_cuda()
    return wav, {"seconds": round(time.perf_counter() - t0, 2), "peak_vram_mb": _peak_mb(pipe)}


def _finish_decode(path: Path, frames, wave, geom: dict, rec: dict, *, cut: bool) -> dict:
    if int(frames.shape[0]) != geom["F_total"]:
        raise RuntimeError(f"{path.name}: {frames.shape[0]} frames, {geom['F_total']} expected")
    t0 = time.perf_counter()
    rec["seam_metrics"] = seam_metrics(frames, geom, cut=cut)
    rec["seam_seconds"] = round(time.perf_counter() - t0, 2)
    rec["size_bytes"] = int(_write(path, frames, wave))
    rec.update({"path": _rel(path), "sha256": _sha_file(path), "frames": int(frames.shape[0]),
                "audio_samples": int(wave.shape[-1]),
                "audio_samples_muxed": min(int(wave.shape[-1]), _round_frac(geom["F_total"] * SAMPLES_PER_FRAME))})
    return rec


def decode_once(pipe, v, a, geom: dict, path: Path) -> dict:
    """(a) (and (c), given the cross-faded latents): one decode of the whole chain."""
    t0 = time.perf_counter()
    frames, vi = decode_frames(pipe, v)
    wave, ai = decode_wave(pipe, a)
    rec = {"peak_vram_mb_video": vi["peak_vram_mb"], "peak_vram_mb_audio": ai["peak_vram_mb"],
           "decode_seconds": {"video": vi["seconds"], "audio": ai["seconds"]}}
    _finish_decode(path, frames, wave, geom, rec, cut=False)
    del frames, wave
    gc.collect()
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    return rec


def decode_perclip(pipe, vs: list, as_: list, geom: dict, path: Path) -> dict:
    """(b): every clip alone, the first O frames (and their audio) of clip i >= 1 cut."""
    t0 = time.perf_counter()
    frames_list, waves = [], []
    pv, pa, sv, sa = [], [], [], []
    for v, a in zip(vs, as_):
        f, vi = decode_frames(pipe, v)
        if int(f.shape[0]) != geom["F"]:
            raise RuntimeError(f"perclip: {f.shape[0]} frames, {geom['F']} expected")
        w, ai = decode_wave(pipe, a)
        frames_list.append(f)
        waves.append(w)
        pv.append(vi["peak_vram_mb"])
        pa.append(ai["peak_vram_mb"])
        sv.append(vi["seconds"])
        sa.append(ai["seconds"])
    rec = {"peak_vram_mb_video": max((x for x in pv if x is not None), default=None),
           "peak_vram_mb_audio": max((x for x in pa if x is not None), default=None),
           "decode_seconds": {"video": sv, "audio": sa},
           "overlap_agreement": overlap_agreement(frames_list, geom)}
    frames = trim_concat_frames(frames_list, geom["O"])
    wave = trim_concat_audio(waves, geom)
    del frames_list, waves
    _finish_decode(path, frames, wave, geom, rec, cut=True)
    del frames, wave
    gc.collect()
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    return rec


def fingerprint_check(pipe, v0, a0, req0, path: Path) -> dict:
    """Clip 0 alone through ``pipe._decode`` (the product's decode) -> mp4 sha256 vs FP_SHA."""
    req = dataclasses.replace(req0, output_path=str(path))
    entry: dict = {}
    t0 = time.perf_counter()
    with torch.no_grad():
        size = pipe._decode(req, v0.to(pipe.device), a0.to(pipe.device), entry, keep=True)
    sha = _sha_file(path)
    return {"path": _rel(path), "sha256": sha, "expected": FP_SHA, "match": sha == FP_SHA,
            "size_bytes": int(size), "entry": entry, "seconds": round(time.perf_counter() - t0, 2)}


def recheck_clip0(pipe, req0, v0, a0) -> dict:
    """C0: clip 0 again with the resident DiT; its latents must be torch.equal."""
    P._free_cuda()
    v, a, info = run_clip(pipe, req0, None, None)
    return {"equal_video": bool(torch.equal(v, v0)), "equal_audio": bool(torch.equal(a, a0)),
            "max_abs_video": float((v - v0).abs().max()), "max_abs_audio": float((a - a0).abs().max()),
            "dit_source": info["dit_source"], "seconds": info["seconds"]}


# -- the run ------------------------------------------------------------------------------------------


def run(args, pipe=None) -> int:
    rec: dict = {"name": args.name, "torch": torch.__version__, "clips": [], "decode": {}, "mem": {}}
    report = Path(args.report) if args.report else HERE / f"{args.name}.json"
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / args.name
    ok = False
    exit_code = 1
    mismatch = []
    inject_before = float(DM.VIDEO_INJECT_CLEAN)
    t_start = time.perf_counter()
    try:
        geom = geometry(args.clip_frames, args.overlap, args.clips, args.strength)
        rec["geometry"] = geom
        weights = weights_record()
        rec["condition"] = condition_record(args, weights)
        prompts = load_prompts(args.prompts, geom["N"])
        s = geom["strength"]
        if s == 0.0 and args.decode != "perclip":
            raise ValueError("strength 0 (the control) is decoded per clip only: --decode perclip")
        if args.fingerprint:
            problems = fingerprint_problems(args, prompts)
            if problems:
                raise ValueError("--fingerprint refused: " + "; ".join(problems))
        if args.recheck_clip0 and args.reuse_clip0:
            raise ValueError("--recheck-clip0 needs a generated clip 0 (not --reuse-clip0)")
        lh, lw = int(args.height) // 16, int(args.width) // 16
        shape_v = (1, LATENT_C, geom["T"], lh, lw)
        shape_a = (2, AUDIO_C, geom["A"])
        reuse = None
        if args.reuse_clip0:
            c1 = json.loads(_abs(args.reuse_clip0).read_text(encoding="utf-8-sig"))
            reuse = load_clip0(c1, args, prompts[0], weights, shape_v, shape_a)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. the inject patch (strength 0 has no mask: nothing to patch)
        rec["inject"] = apply_inject(args.inject if s > 0.0 else "noise")
        # 2. the pipeline
        if pipe is None:
            pipe = make_pipeline(args, rec)
        if _cuda(pipe):
            rec["device_name"] = torch.cuda.get_device_name(0)
        rec["numerics"] = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                           "float32_matmul_precision": torch.get_float32_matmul_precision(),
                           "cudnn_deterministic": torch.backends.cudnn.deterministic,
                           "cudnn_benchmark": torch.backends.cudnn.benchmark}
        rec["mem"]["start"] = P._mem_gib()
        req0 = clip_request(args, prompts[0], out_dir)

        # 3. the clips
        vs, as_ = [], []
        acc_a = None
        for i in range(geom["N"]):
            req = dataclasses.replace(req0, seed=int(args.seed) + i, prompt=prompts[i])
            P.validate_request(req)
            crec: dict = {"index": i, "seed": req.seed, "prompt": req.prompt, "prompt_sha": _sha_text(req.prompt)}
            rec["clips"].append(crec)
            if i == 0 and reuse is not None:
                v, a = reuse["video"], reuse["audio"]
                crec.update({"source": "reused", "reused_from": args.reuse_clip0, "latents_path": reuse["path"]})
                _log(f"clip 0 reused from {reuse['path']}")
            else:
                vfix, afix = (None, None)
                if i > 0:
                    k, ka = geom["k"], geom["ka"][i - 1]
                    vfix, afix = build_fix(vs[-1], acc_a, shape_v, shape_a, k, ka, s)
                    crec["fix"] = {"k": k, "ka": ka, "mask_value": None if vfix is None else 1.0 - s}
                P._free_cuda()
                _log(f"clip {i}: seed {req.seed}, fix {'none' if vfix is None else 'k=%d ka=%d' % (k, ka)}")
                v, a, info = run_clip(pipe, req, vfix, afix)
                del vfix, afix
                crec["source"] = "generated"
                crec.update(info)
            crec["latents"] = latent_record(v, a)
            if tuple(v.shape) != shape_v or tuple(a.shape) != shape_a:
                raise RuntimeError(f"clip {i}: latents {list(v.shape)} / {list(a.shape)}, "
                                   f"expected {list(shape_v)} / {list(shape_a)}")
            if i > 0 and s > 0.0:
                k, ka = geom["k"], geom["ka"][i - 1]
                hv = (v[:, :, :k] - vs[-1][:, :, -k:]).abs().max() if k else torch.tensor(0.0)
                ha = (a[:, :, :ka] - acc_a[:, :, -ka:]).abs().max() if ka else torch.tensor(0.0)
                eq_v = bool(torch.equal(v[:, :, :k], vs[-1][:, :, -k:]))
                eq_a = bool(torch.equal(a[:, :, :ka], acc_a[:, :, -ka:])) if ka else True
                crec["head_equals_prev_tail"] = eq_v and eq_a
                crec["head_equals_prev_tail_video"] = eq_v
                crec["head_equals_prev_tail_audio"] = eq_a
                crec["head_vs_prev_tail_max_abs"] = {"video": float(hv), "audio": float(ha)}
                if s == 1.0 and not crec["head_equals_prev_tail"]:
                    raise RuntimeError(f"clip {i}: strength 1.0 but the head is not the previous tail "
                                       f"(max |d| video {float(hv)}, audio {float(ha)})")
            if args.save_latents and crec.get("source") == "generated":
                lp = out_dir / f"clip{i}.pt"
                torch.save({"video": v, "audio": a, "name": args.name, "index": i, "seed": req.seed,
                            "prompt": req.prompt}, lp)
                crec["latents_path"] = _rel(lp)
            vs.append(v)
            as_.append(a)
            acc_a = a if i == 0 else torch.cat([acc_a, a[:, :, geom["ka"][i - 1]:]], dim=2)
            rec["mem"][f"after_clip{i}"] = P._mem_gib()
        if int(acc_a.shape[2]) != geom["A_concat"]:
            raise RuntimeError(f"concatenated audio {acc_a.shape[2]} latents, geometry says {geom['A_concat']}")

        # 4. recheck (C0)
        if args.recheck_clip0:
            rec["recheck"] = recheck_clip0(pipe, req0, vs[0], as_[0])
            if not (rec["recheck"]["equal_video"] and rec["recheck"]["equal_audio"]):
                mismatch.append("recheck")
        # 5. the DiT goes before any decode
        pipe.release_resident()
        P._free_cuda()
        rec["mem"]["after_release"] = P._mem_gib()
        # 6. fingerprint, then (a) (b) (c)
        if args.fingerprint:
            rec["decode"]["fingerprint"] = fingerprint_check(pipe, vs[0], as_[0], req0, out_dir / "clip0_fingerprint.mp4")
            if not rec["decode"]["fingerprint"]["match"]:
                mismatch.append("fingerprint")
        if args.decode in ("once", "both"):
            cv, ca = concat_latents(vs, as_, geom)
            rec["decode"]["once"] = decode_once(pipe, cv, ca, geom, out_dir / "once.mp4")
            del cv, ca
        if args.decode in ("perclip", "both"):
            rec["decode"]["perclip"] = decode_perclip(pipe, vs, as_, geom, out_dir / "perclip.mp4")
        if 0.0 < s < 1.0 and args.decode in ("once", "both"):
            xv, xa = xfade_latents(vs, as_, geom)
            rec["decode"]["xfade"] = decode_once(pipe, xv, xa, geom, out_dir / "xfade.mp4")
            del xv, xa
        rec["mem"]["end"] = P._mem_gib()
        ok = not mismatch
        exit_code = 2 if mismatch else 0
        rec["mismatch"] = mismatch
    except torch.OutOfMemoryError as exc:
        rec["oom"] = True
        rec["error"] = f"OutOfMemoryError: {exc}"
        rec["traceback"] = traceback.format_exc()
    except BaseException as exc:  # noqa: BLE001 -- the reason goes into the JSON
        rec["oom"] = "out of memory" in str(exc).lower()
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
    finally:
        DM.VIDEO_INJECT_CLEAN = inject_before
        if pipe is not None:
            try:
                pipe.close()
            except Exception as exc:  # noqa: BLE001
                rec["close_error"] = repr(exc)
        if torch.cuda.is_available():
            rec["peak_alloc_mb_at_exit"] = _mb(torch.cuda.max_memory_allocated())
        rec["seconds_total"] = round(time.perf_counter() - t_start, 2)
        rec.setdefault("oom", False)
        rec["ok"] = ok
        report.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(rec, indent=2, ensure_ascii=False, default=str)
        report.write_text(text, encoding="utf-8")
        print(json.dumps({k: rec.get(k) for k in ("name", "ok", "error", "mismatch", "seconds_total")},
                         ensure_ascii=False, default=str))
        if rec.get("traceback"):
            print(rec["traceback"])
    return exit_code


# -- self-check (CPU, no weights; plan §7) ---------------------------------------------------------

#: Plan §4's table (C0..C3 share one row). err = audio_head_shift_latents (C7: None, strength 0 has no overlap).
EXPECTED = {
    "C0-C3": dict(o=22, s=1.0, O=22, k=7, S=[102, 204], J=[124, 226], ka=[37, 37], F_total=328, T_total=97,
                  A_concat=547, A_expected=547, excess=0, err=[0.0, 0.0]),
    "C4": dict(o=5, s=1.0, O=5, k=2, S=[119, 238], J=[124, 243], ka=[9, 8], F_total=362, T_total=107,
               A_concat=604, A_expected=603, excess=1, err=[-1 / 3, 1 / 3]),
    "C5": dict(o=39, s=1.0, O=39, k=12, S=[85, 170], J=[124, 209], ka=[65, 66], F_total=294, T_total=87,
               A_concat=490, A_expected=490, excess=0, err=[1 / 3, -1 / 3]),
    "C7": dict(o=None, s=0.0, O=0, k=0, S=[124, 248], J=[124, 248], ka=[0, 0], F_total=372, T_total=111,
               A_concat=621, A_expected=620, excess=1, err=None),
}


def _line(results: list, ok: bool, what: str) -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {what}", flush=True)


def _raises(fn, *a, **kw) -> bool:
    try:
        fn(*a, **kw)
    except (ValueError, AssertionError):
        return True
    return False


def _args(**kw):
    base = dict(name="sc", clip_frames=124, clips=3, overlap=None, strength=1.0, inject="noise", prompts=None,
                seed=42, width=1344, height=768, steps=20, shift_video=12.0, shift_audio=3.0, attention="sage",
                vram16=True, decode="both", save_latents=False, reuse_clip0=None, fingerprint=False,
                recheck_clip0=False, report=None, out_dir=None)
    base.update(kw)
    return argparse.Namespace(**base)


class _FakeVideoDecoder:
    def __init__(self, log):
        self.log = log

    def to(self, _d):
        return self

    def decode(self, lat, tiled=True):
        T = int(lat.shape[2])
        F = (T - 2) // 5 * 17 + 5
        self.log.append(("decode_video", F))
        h, w = int(lat.shape[3]) * 16, int(lat.shape[4]) * 16
        base = (torch.arange(F, dtype=torch.int64) % 200).to(torch.uint8).view(F, 1, 1, 1)
        return base.expand(F, h, w, 3).clone()


class _FakeAudioVAE:
    def __init__(self, log):
        self.log = log

    def to(self, _d):
        return self

    def decode(self, lat):
        self.log.append(("decode_audio", int(lat.shape[2])))
        return torch.zeros((2, int(lat.shape[2]) * 800), dtype=torch.float32)


class _FakePipe:
    """Records what run() asks of the pipeline; the denoise keeps m = 0 rows clean via the real masks."""

    def __init__(self, log):
        self.device = torch.device("cpu")
        self._resident = None
        self.log = log
        self.prev_v = None
        self.acc_a = None

    def _encode(self, prompt, phases, images=(), *, has_cond, has_audio=False, refs=None):
        self.log.append(("encode", "encode" in phases.data, tuple(images), has_cond))
        return torch.full((5, 8), float(len(prompt))), torch.ones(5, dtype=torch.long)

    def _denoise(self, req, hidden, tags, phases, entry, cond_clean=None, cond_frame_idx=(), keep=False,
                 lora_report=None, audio_fix=None, video_fix=None, refs=None):
        T, A = latent_frames(req.num_frames), audio_latents(req.num_frames)
        lh, lw = req.height // 16, req.width // 16
        g = torch.Generator().manual_seed(int(req.seed))
        noise_v = torch.randn((1, LATENT_C, T, lh, lw), generator=g)
        noise_a = torch.randn((2, AUDIO_C, A), generator=g)
        v = noise_v * 0.5 + float(hidden.mean()) * 1e-3
        a = noise_a * 0.5
        head = None
        ka = 0
        if video_fix is not None:
            k = int(sum(1 for x in video_fix[1] if x < 1))
            ka = int(sum(1 for x in audio_fix[1] if x < 1))
            cv, ca = video_fix[0], audio_fix[0]
            if self.prev_v is not None:    # None after a reused clip 0 (this fake never saw it)
                head = (bool(torch.equal(cv[:, :, :k], self.prev_v[:, :, -k:])) and bool((cv[:, :, k:] == 0).all())
                        and bool(torch.equal(ca[:, :, :ka], self.acc_a[:, :, -ka:]))
                        and bool((ca[:, :, ka:] == 0).all()))
            vm = DM.video_mask(cv, video_fix[1], grid=(req.height // 32) * (req.width // 32), noise=noise_v)
            am = DM.audio_mask(ca, audio_fix[1])
            v, a = DM.finish(v, vm), DM.finish(a, am)
        self.log.append(("denoise", keep, int(req.seed), video_fix is None, audio_fix is None, head))
        entry["dit_source"] = "resident" if self._resident else "file"
        if keep:
            self._resident = {**(self._resident or {}), "dit": object()}
        self.prev_v = v
        if video_fix is None or self.acc_a is None:
            self.acc_a = a
        else:
            self.acc_a = torch.cat([self.acc_a, a[:, :, ka:]], dim=2)
        return v, a, "fake"

    def release_resident(self):
        self.log.append(("release",))
        self._resident = None

    def close(self):
        self.log.append(("close",))


#: The fake runs' clip length: 17n+5 with audio_latents(F) >= F * 5/3 (F = 1 mod 3, as 124), which (b)'s
#: audio cut needs (trim_concat_audio refuses a clip whose audio ends before its last frame).
FAKE_F = 73


def _fake_run(tmp: Path, log: list, refs: list, **kw):
    """run() with the fake pipeline, decoders and writer; returns (exit code, report dict)."""
    def writer(path, frames, *, fps, audio, sample_rate):
        refs.append(__import__("weakref").ref(frames))
        log.append(("write", Path(path).name, int(frames.shape[0]), int(audio.shape[1])))
        Path(path).write_bytes(b"fake")
        return 4

    HOOKS.update({"video_decoder": lambda pipe: _FakeVideoDecoder(log), "audio_vae": lambda pipe: _FakeAudioVAE(log),
                  "write_mp4": writer})
    try:
        args = _args(width=64, height=64, clip_frames=FAKE_F, report=str(tmp / f"{kw.get('name', 'sc')}.json"),
                     out_dir=str(tmp / kw.get("name", "sc")), **kw)
        code = run(args, pipe=_FakePipe(log))
    finally:
        HOOKS.update({"video_decoder": None, "audio_vae": None, "write_mp4": None})
    return code, json.loads(Path(args.report).read_text(encoding="utf-8"))


def selfcheck(report: str | None = None) -> int:
    import inspect
    import tempfile
    import weakref

    from engine_h3.media_out import write_mp4
    from engine_h3.scheduler import euler_step, sigma_schedule

    results: list[bool] = []
    _line(results, not torch.cuda.is_available(), f"CPU only (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")

    # 1. plan §4's table
    for label, e in EXPECTED.items():
        g = geometry(124, e["o"], 3, e["s"])
        same = (g["F"], g["T"], g["A"]) == (124, 37, 207) and all(
            g[key] == e[key] for key in ("O", "k", "S", "J", "ka", "F_total", "T_total", "A_concat", "A_expected"))
        same = same and g["audio_latent_excess"] == e["excess"]
        if e["err"] is None:
            same = same and all(g[key] is None for key in (
                "audio_head_shift_latents", "audio_place_error_latents", "audio_fixed_end_vs_video_latents"))
        else:
            same = same and all(abs(x - y) < 1e-12 for x, y in zip(g["audio_head_shift_latents"], e["err"]))
            same = same and all(abs(x) - abs(y) == 0 for x, y in zip(g["audio_place_error_latents"], g["audio_head_shift_latents"]))
        same = same and all(0 <= x <= g["A"] for x in g["ka"]) and 2 * g["k"] <= g["T"]
        if e["s"] > 0:
            same = same and g["T_total"] == latent_frames(g["F_total"])
        _line(results, same, f"1 geometry {label}: S={g['S']} J={g['J']} ka={g['ka']} F={g['F_total']} T'={g['T_total']} "
                             f"A={g['A_concat']} (round {g['A_expected']}, excess {g['audio_latent_excess']:+d}) "
                             f"shift={None if g['audio_head_shift_latents'] is None else [round(x, 4) for x in g['audio_head_shift_latents']]}")
    # integer rounding helper vs the product's audio_latents
    _line(results, all(_round_frac(Fraction(5 * n, 3)) == audio_latents(n) for n in range(0, 3000)),
          "1 round(n*5/3) (exact) == constants.audio_latents(n) for n < 3000")

    # 2. sweep
    valid = refused = 0
    excess: dict = {}
    bad = []
    for F in range(22, 482, 17):
        for O in (5, 22, 39):
            for N in range(2, 7):
                try:
                    g = geometry(F, O, N, 1.0)
                except ValueError:
                    refused += 1
                    continue
                valid += 1
                excess[g["audio_latent_excess"]] = excess.get(g["audio_latent_excess"], 0) + 1
                inv = (g["T_total"] == latent_frames(g["F_total"]) and all(0 <= x <= g["A"] for x in g["ka"])
                       and all(j - s_ == O for s_, j in zip(g["S"], g["J"]))
                       and g["S"] == [i * (F - O) for i in range(1, N)]
                       and abs(g["audio_latent_excess"]) <= 1
                       and all(abs(x) <= 1 / 3 + 1e-12 for x in g["audio_head_shift_latents"])
                       and g["A_concat"] == g["A"] + sum(g["A"] - x for x in g["ka"])
                       and 2 * g["k"] <= g["T"] and O < F)
                if not inv:
                    bad.append((F, O, N))
    _line(results, not bad and valid > 0,
          f"2 sweep F 22..481 x O 5/22/39 x N 2..6: {valid} valid, {refused} refused, "
          f"excess counts {dict(sorted(excess.items()))}, invariant failures {bad[:5]}")

    # 3. build_fix
    pv = torch.randn(1, 24, 37, 2, 3)
    pa = torch.randn(2, 32, 377)
    (cv, per_v), (ca, per_a) = build_fix(pv, pa, (1, 24, 37, 2, 3), (2, 32, 207), 7, 37, 1.0)
    ok3 = (tuple(cv.shape) == (1, 24, 37, 2, 3) and tuple(ca.shape) == (2, 32, 207) and cv.dtype == torch.float32
           and torch.equal(cv[:, :, :7], pv[:, :, -7:]) and bool((cv[:, :, 7:] == 0).all())
           and torch.equal(ca[:, :, :37], pa[:, :, -37:]) and bool((ca[:, :, 37:] == 0).all())
           and per_v == [0.0] * 7 + [1.0] * 30 and per_a == [0.0] * 37 + [1.0] * 170)
    (_cv, per_v5), (_ca, per_a5) = build_fix(pv, pa, (1, 24, 37, 2, 3), (2, 32, 207), 7, 37, 0.5)
    ok3 = ok3 and per_v5[:7] == [0.5] * 7 and per_v5[7:] == [1.0] * 30 and per_a5[:37] == [0.5] * 37
    (_cv0, per_v0), (ca0, per_a0) = build_fix(pv, pa, (1, 24, 37, 2, 3), (2, 32, 207), 7, 0, 1.0)
    ok3 = ok3 and bool((ca0 == 0).all()) and per_a0 == [1.0] * 207 and DM.audio_mask(ca0, per_a0) is None
    none_fix = build_fix(pv, pa, (1, 24, 37, 2, 3), (2, 32, 207), 0, 0, 0.0) == (None, None)
    ones_none = (DM.video_mask(torch.zeros(1, 24, 37, 2, 3), [1.0] * 37, grid=1, noise=torch.zeros(1, 24, 37, 2, 3))
                 is None and DM.audio_mask(torch.zeros(2, 32, 207), [1.0] * 207) is None)
    _line(results, ok3 and none_fix and ones_none,
          "3 build_fix: shapes, head = previous tail, zeros past k / ka, per values (s=1, 0.5), ka=0 -> no audio "
          "mask, s=0 -> (None, None) and all-ones masks are None")

    # 4. real masks: rows, default inject, the patch and its undo
    g = torch.Generator().manual_seed(3)
    clean = torch.randn(1, 4, 5, 2, 2, generator=g)
    noise = torch.randn(1, 4, 5, 2, 2, generator=g)
    per = [0.0, 0.0, 0.5, 1.0, 1.0]
    vm = DM.video_mask(clean, per, grid=3, noise=noise)
    am = DM.audio_mask(torch.randn(2, 32, 6, generator=g), [0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    ok4 = (torch.equal(vm.rows, torch.tensor(per).repeat_interleave(3)) and
           torch.equal(vm.inject, 0.999 * clean + (1.0 - 0.999) * noise) and
           torch.equal(am.rows, torch.tensor([0.0, 0.0, 1, 1, 1, 1]).repeat(2)) and DM.VIDEO_INJECT_CLEAN == 0.999)
    rec_clean = apply_inject("clean")
    vm_c = DM.video_mask(clean, per, grid=3, noise=noise)
    ok4 = ok4 and torch.equal(vm_c.inject, clean) and rec_clean["inject_equals_clean"] and rec_clean["after"] == 1.0
    DM.VIDEO_INJECT_CLEAN = rec_clean["before"]
    vm_back = DM.video_mask(clean, per, grid=3, noise=noise)
    rec_noise = apply_inject("noise")
    ok4 = ok4 and torch.equal(vm_back.inject, vm.inject) and not rec_noise["inject_equals_clean"] \
        and rec_noise["video_mask_reads_patched"] and DM.VIDEO_INJECT_CLEAN == 0.999
    _line(results, ok4, "4 video_mask / audio_mask: rows = per x grid, inject 0.999/0.001 by default; "
                        "patched -> inject == clean; undone -> as before")

    # 5. the product's step helpers with a dummy velocity
    gen = torch.Generator().manual_seed(5)
    clean = torch.randn(1, 4, 5, 2, 2, generator=gen)
    x = torch.randn(1, 4, 5, 2, 2, generator=gen)
    vm = DM.video_mask(clean, [0.0, 0.0, 0.5, 1.0, 1.0], grid=4, noise=x.clone())
    sig = sigma_schedule(8, 12.0)
    for i in range(8):
        xin = DM.dit_input(x, vm)
        v = torch.sin(xin) - 0.3 * xin          # any smooth "model"
        x = euler_step(x, DM.state_velocity(v, x, vm, float(sig[i])), float(sig[i]), float(sig[i + 1]))
    x = DM.finish(x, vm)
    _line(results, torch.equal(x[:, :, :2], clean[:, :, :2]) and bool(torch.isfinite(x).all()),
          "5 dit_input / state_velocity / euler_step / finish: m=0 rows end == clean, m=0.5 finite")

    # 6. concatenation at strength 1
    prev = torch.randn(1, 2, 10, 1, 1)
    cur = torch.randn(1, 2, 10, 1, 1)
    cur[:, :, :3] = prev[:, :, -3:]
    _line(results, torch.equal(torch.cat([prev, cur[:, :, 3:]], 2), torch.cat([prev[:, :, :-3], cur], 2)),
          "6 s=1: cat(prev, cur[k:]) == cat(prev[:-k], cur)")
    g22 = geometry(124, 22, 3, 1.0)
    vs = [torch.randn(1, 24, 37, 1, 1) for _ in range(3)]
    as_ = [torch.randn(2, 32, 207) for _ in range(3)]
    cvl, cal = concat_latents(vs, as_, g22)
    _line(results, cvl.shape[2] == g22["T_total"] and cal.shape[2] == g22["A_concat"],
          f"6 concat lengths: video {cvl.shape[2]} (T'_total), audio {cal.shape[2]} (A concat)")

    # 7. cross-fade
    p = torch.zeros(1, 1, 6, 1, 1)
    c = torch.ones(1, 1, 6, 1, 1)
    xf = _xfade(p, c, 4)
    w = xf[0, 0, 2:6, 0, 0]
    ok7 = (xf.shape[2] == 8 and abs(float(w[0]) - 1 / 5) < 1e-7 and abs(float(w[-1]) - 4 / 5) < 1e-7
           and bool((xf[0, 0, :2] == 0).all()) and bool((xf[0, 0, 6:] == 1).all()))
    same_in = torch.randn(1, 3, 9, 1, 1)
    ok7 = ok7 and torch.allclose(_xfade(same_in, same_in[:, :, -4:].repeat(1, 1, 1, 1, 1).clone(), 4),
                                 same_in, atol=1e-6)
    xv, xa = xfade_latents(vs, as_, g22)
    ok7 = ok7 and xv.shape == cvl.shape and xa.shape == cal.shape and torch.equal(xv, xfade_latents(vs, as_, g22)[0])
    a_p, a_c = torch.zeros(2, 32, 40), torch.ones(2, 32, 40)
    xa2 = _xfade(a_p, a_c, 37)
    ok7 = ok7 and xa2.shape[2] == 43 and abs(float(xa2[0, 0, 3]) - 1 / 38) < 1e-7 and abs(float(xa2[0, 0, 39]) - 37 / 38) < 1e-7
    _line(results, ok7, "7 cross-fade: end weights 1/(n+1), n/(n+1); same input -> same output; lengths "
                        "(video k, audio ka) = the concatenation's")

    # 8. trim (b)
    ok8 = True
    for label in ("C0-C3", "C4", "C5", "C7"):
        e = EXPECTED[label]
        g = geometry(124, e["o"], 3, e["s"])
        starts = [0] + g["S"]
        frames = [(torch.arange(124) + st).view(124, 1, 1, 1) for st in starts]
        got = trim_concat_frames(frames, g["O"]).view(-1)
        ramps = [(torch.arange(207 * 800, dtype=torch.float64) + float(st * SAMPLES_PER_FRAME)).repeat(2, 1)
                 for st in starts]
        wav = trim_concat_audio(ramps, g)
        n = min(int(wav.shape[1]), _round_frac(g["F_total"] * SAMPLES_PER_FRAME))
        dev = float((wav[0, :n] - torch.arange(n, dtype=torch.float64)).abs().max())
        ok8 = ok8 and torch.equal(got, torch.arange(g["F_total"])) and dev <= 1.0 and wav.shape[1] >= n
        print(f"      {label}: frames {got.numel()} = arange, audio {wav.shape[1]} samples, max |pos - value| {dev:.3f}")
    _line(results, ok8, "8 trim: frames = arange(F_total); audio ramp off by <= 1 sample")

    # 9. run() on a fake pipeline
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        log, refs = [], []
        code, rep = _fake_run(tmp, log, refs, name="fake1", overlap=22, strength=1.0, decode="both",
                              save_latents=True, recheck_clip0=True)
        gc.collect()
        den = [e for e in log if e[0] == "denoise"]
        enc = [e for e in log if e[0] == "encode"]
        rel = [i for i, e in enumerate(log) if e[0] == "release"]
        dec = [i for i, e in enumerate(log) if e[0].startswith("decode")]
        writes = {e[1]: e for e in log if e[0] == "write"}
        g = rep.get("geometry", {})
        ok9 = (code == 0 and rep.get("ok") and len(den) == 4 and all(e[1] for e in den)
               and [e[2] for e in den[:3]] == [42, 43, 44] and den[3][2] == 42
               and den[0][3] and den[0][4] and not den[1][3] and den[1][5] and den[2][5]
               and all(e[1] and e[2] == () and e[3] is False for e in enc)
               and rel and dec and rel[0] < dec[0]
               and writes.get("once.mp4", (0, 0, 0))[2] == g.get("F_total")
               and writes.get("perclip.mp4", (0, 0, 0))[2] == g.get("F_total")
               and "xfade.mp4" not in writes
               and all(r() is None for r in refs) and len(refs) == 2
               and all(c.get("head_equals_prev_tail") for c in rep["clips"][1:])
               and rep.get("recheck", {}).get("equal_video") and rep["inject"]["mode"] == "noise")
        _line(results, ok9, f"9 fake run s=1: exit {code}, keep all True, seeds {[e[2] for e in den]}, clip 0 fix None, "
                            f"heads = previous tails, encode inside its phase, release before decode, "
                            f"(a)/(b) {writes.get('once.mp4', (0,0,0))[2]}/{writes.get('perclip.mp4', (0,0,0))[2]} "
                            f"frames = F_total {g.get('F_total')}, frames let go after writing ({len(refs)} weakrefs dead)")
        # s = 0.5 with the clean inject: (a)(b)(c), and the patch is undone afterwards
        log2, refs2 = [], []
        code2, rep2 = _fake_run(tmp, log2, refs2, name="fake2", overlap=22, strength=0.5, inject="clean", decode="both")
        gc.collect()
        w2 = sorted(e[1] for e in log2 if e[0] == "write")
        _line(results, code2 == 0 and w2 == ["once.mp4", "perclip.mp4", "xfade.mp4"]
              and rep2["inject"]["inject_equals_clean"] and DM.VIDEO_INJECT_CLEAN == 0.999
              and all(r() is None for r in refs2)
              and all(c.get("head_equals_prev_tail") is False for c in rep2["clips"][1:]),
              f"9 fake run s=0.5 inject clean: writes {w2}, patch undone, heads not fixed bit-for-bit")
        # s = 0: per clip only, no fix
        log3, refs3 = [], []
        code3, rep3 = _fake_run(tmp, log3, refs3, name="fake3", strength=0.0, decode="perclip")
        den3 = [e for e in log3 if e[0] == "denoise"]
        w3 = [e for e in log3 if e[0] == "write"]
        _line(results, code3 == 0 and all(e[3] and e[4] for e in den3) and len(w3) == 1
              and w3[0][2] == rep3["geometry"]["F_total"] == 3 * FAKE_F,
              f"9 fake run s=0: no fix on any clip, one per-clip mp4 of {w3[0][2] if w3 else None} frames")
        # reuse clip 0 from fake1
        log4, refs4 = [], []
        code4, rep4 = _fake_run(tmp, log4, refs4, name="fake4", overlap=22, strength=1.0, decode="once",
                                reuse_clip0=str(tmp / "fake1.json"))
        den4 = [e for e in log4 if e[0] == "denoise"]
        _line(results, code4 == 0 and rep4["clips"][0]["source"] == "reused" and [e[2] for e in den4] == [43, 44]
              and rep4["clips"][0]["latents"]["v_sha256"] == rep["clips"][0]["latents"]["v_sha256"],
              f"9 fake run reuse: clip 0 reused (sha as C1), denoised seeds {[e[2] for e in den4]}")

        # 11. refusals
        ok11 = (_raises(geometry, 124, 23, 3, 1.0) and _raises(geometry, 125, 22, 3, 1.0)
                and _raises(geometry, 22, 22, 3, 1.0) and _raises(geometry, 56, 39, 3, 1.0)
                and _raises(geometry, 5, 5, 3, 1.0) and _raises(geometry, 124, 22, 1, 1.0)
                and _raises(geometry, 124, 22, 3, 1.5))
        fp_ok = fingerprint_problems(_args(width=640, height=384, attention="sdpa"), [FP_PROMPT] * 3) == []
        fp_bad = fingerprint_problems(_args(attention="sdpa"), [FP_PROMPT] * 3)
        fp_bad2 = fingerprint_problems(_args(width=640, height=384, attention="sdpa", seed=1), ["x"] * 3)
        ok11 = ok11 and fp_ok and len(fp_bad) == 2 and len(fp_bad2) == 2
        c1 = json.loads((tmp / "fake1.json").read_text(encoding="utf-8"))
        a1 = _args(width=64, height=64, clip_frames=FAKE_F)
        sv, sa = (1, 24, latent_frames(FAKE_F), 4, 4), (2, 32, audio_latents(FAKE_F))
        w = weights_record()
        accepted = load_clip0(c1, a1, FP_PROMPT, w, sv, sa)["seed"] == 42
        refuse_cond = _raises(load_clip0, c1, _args(width=64, height=64, clip_frames=FAKE_F, steps=8), FP_PROMPT, w, sv, sa)
        refuse_prompt = _raises(load_clip0, c1, a1, "another prompt", w, sv, sa)
        bad_sha = json.loads(json.dumps(c1))
        bad_sha["clips"][0]["latents"]["v_sha256"] = "0" * 64
        refuse_sha = _raises(load_clip0, bad_sha, a1, FP_PROMPT, w, sv, sa)
        code_fp, rep_fp = _fake_run(tmp, [], [], name="fake5", fingerprint=True)
        code_s0, rep_s0 = _fake_run(tmp, [], [], name="fake6", strength=0.0, decode="both")
        _line(results, ok11 and accepted and refuse_cond and refuse_prompt and refuse_sha
              and code_fp == 1 and "--fingerprint refused" in rep_fp.get("error", "")
              and code_s0 == 1 and "perclip" in rep_s0.get("error", ""),
              f"11 refusals: overlap off 17m+5 / F off 17n+5 / O >= F / 2k > T' / F < 22 / N < 2 / s > 1; "
              f"--fingerprint off FP_COND ({len(fp_bad)} problems); load_clip0 refuses a condition, a prompt and "
              f"a sha mismatch, accepts the match; strength 0 with --decode both refused")

    # 10. the product's signatures (a change in the product is detected here)
    pd = inspect.signature(P.H3Pipeline._denoise).parameters
    pe = inspect.signature(P.H3Pipeline._encode).parameters
    pdec = inspect.signature(P.H3Pipeline._decode).parameters
    pw = inspect.signature(write_mp4).parameters
    fields = {f.name for f in dataclasses.fields(P.H3GenerateRequest)}
    need = {"prompt", "seed", "width", "height", "num_frames", "output_path", "num_inference_steps", "shift_video",
            "shift_audio", "images", "keep_resident", "attention_backend", "block_swap_prefetch", "vram16_mode"}
    src_den = inspect.getsource(P.H3Pipeline._denoise)
    src_enc = inspect.getsource(P.H3Pipeline._encode)
    src_vm = inspect.getsource(DM.video_mask)
    ok10 = (list(pd)[:6] == ["self", "req", "text_hidden", "text_tags", "phases", "entry"]
            and all(k in pd for k in ("keep", "video_fix", "audio_fix", "cond_clean", "cond_frame_idx"))
            and list(pe)[:4] == ["self", "prompt", "phases", "images"] and pe["has_cond"].kind == inspect.Parameter.KEYWORD_ONLY
            and list(pdec)[:6] == ["self", "req", "video_lat", "audio_lat", "entry", "keep"]
            and list(inspect.signature(P._Phases.__init__).parameters) == ["self", "cuda"] and hasattr(P._Phases, "phase")
            and need <= fields
            and list(pw)[:2] == ["path", "frames"] and all(k in pw for k in ("fps", "audio", "sample_rate"))
            and DM.VIDEO_INJECT_CLEAN == 0.999
            and "from engine_h3.denoise_mask import" in src_den and "video_mask(" in src_den
            and "return video, audio, prefetch_used" in src_den and 'phases.data["encode"]["te"]' in src_enc
            and "VIDEO_INJECT_CLEAN * clean32" in src_vm
            and callable(P._free_cuda) and hasattr(P.H3Pipeline, "release_resident"))
    _line(results, ok10, "10 product contract: _denoise / _encode / _decode / _Phases / H3GenerateRequest fields / "
                         "write_mp4 signatures, _denoise imports video_mask at call time, video_mask reads "
                         "VIDEO_INJECT_CLEAN, _encode writes phases.data['encode'], VIDEO_INJECT_CLEAN == 0.999")

    # 12. a short synthetic mp4 through analyze_chain, and seam_metrics == analyze_chain's numbers
    import numpy as np

    Fs, Js, H, W = 46, 30, 96, 128
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    frames = []
    for t in range(Fs):
        r = ((xx + 2 * t) % 256).float() * 0.6 + 40
        gch = ((yy + t) % 256).float() * 0.5 + 50
        b = torch.full_like(r, 90.0)
        img = torch.stack([r, gch, b], -1) + (40.0 if t >= Js else 0.0)
        frames.append(img.clamp(0, 255).round().to(torch.uint8))
    frames = torch.stack(frames)
    tsec = torch.arange(_round_frac(Fs * SAMPLES_PER_FRAME) + 400, dtype=torch.float32) / AUDIO_SAMPLE_RATE
    wave = (0.2 * torch.sin(2 * 3.14159 * 440 * tsec)).repeat(2, 1)
    geom_s = {"S": [Js], "J": [Js]}
    with tempfile.TemporaryDirectory() as td:
        mp4 = Path(td) / "synthetic.mp4"
        write_mp4(str(mp4), frames, fps=FRAME_RATE, audio=wave, sample_rate=AUDIO_SAMPLE_RATE)
        st_mp4 = AC.read_mp4_video_stats(mp4)
        aw, rate = AC.read_mp4_audio(mp4)
    st_t = frame_stats_torch(frames)
    st_n = AC.stats_from_rgb(frames.numpy())
    rel = max(max(abs(a - b) / max(abs(b), 1e-9) for a, b in zip(st_t[key], st_n[key]))
              for key in ("y_mean", "y_sq_mean", "chroma_mean", "d"))
    sm = seam_metrics(frames, geom_s)
    sn = AC.summarize(st_n, geom_s)
    d = st_mp4["d"]
    argmax_t = int(np.argmax(d)) + 1
    ratio = AC.jump_ratio(d, Js)
    want_samples = _round_frac(Fs * SAMPLES_PER_FRAME)
    print(f"      mp4: {st_mp4['frames']} frames, audio {aw.shape} at {rate} Hz (muxed {want_samples}), "
          f"argmax d = {argmax_t}, ratio at J {ratio:.1f}, memory ratio {sm['summary']['seams'][0]['ratio_J']:.1f}, "
          f"torch vs numpy max rel {rel:.2e}")
    _line(results, st_mp4["frames"] == Fs and rate == AUDIO_SAMPLE_RATE and aw.shape[0] == 2
          and abs(aw.shape[1] - want_samples) <= 2048 and argmax_t == Js and ratio is not None and ratio > 5,
          f"12 analyze_chain reads the synthetic mp4: {st_mp4['frames']} frames, {aw.shape[1]} samples "
          f"(~{want_samples}), the planted jump found at t={argmax_t}")
    same12 = (rel < 1e-9 and sm["summary"]["seams"][0]["ratio_J"] is not None
              and abs(sm["summary"]["seams"][0]["ratio_J"] - sn["seams"][0]["ratio_J"]) < 1e-6
              and abs(sm["summary"]["regions"][1]["delta_y_mean"] - sn["regions"][1]["delta_y_mean"]) < 1e-6)
    _line(results, same12, f"12 seam_metrics (torch) == analyze_chain.stats_from_rgb (numpy) on the same frames "
                           f"(max rel {rel:.1e}; jump ratio {sn['seams'][0]['ratio_J']:.2f}, region dY "
                           f"{sn['regions'][1]['delta_y_mean']:.2f})")
    # the audio seam helper on a planted click
    click = np.zeros((2, 32000), dtype=np.float32)
    click[:, :] = 0.1 * np.sin(np.arange(32000) * 0.05)
    click[:, 16000:] += 0.5
    asx = AC.audio_seam(click, 32000, 0.5)
    _line(results, len(asx["rms"]) == 20 and asx["click_ratio"] is not None and asx["click_ratio"] > 5,
          f"12 audio_seam: 20 RMS windows, planted click ratio {asx['click_ratio']:.1f}")

    ok = all(results)
    line = f"SELFCHECK {'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)})"
    if report:
        Path(report).write_text(json.dumps({"selfcheck": "PASS" if ok else "FAIL", "passed": sum(results),
                                            "total": len(results)}, indent=2), encoding="utf-8")
    print(line)
    if ok:
        print("SELFCHECK PASS")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--clip-frames", type=int, default=124)
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--overlap", type=int, default=None, help="17m+5 frames (default 22); ignored at --strength 0")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--inject", choices=("noise", "clean"), default="noise")
    ap.add_argument("--prompts", help="JSON list of prompts (a shorter list repeats its last entry)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--shift-video", type=float, default=12.0)
    ap.add_argument("--shift-audio", type=float, default=3.0)
    ap.add_argument("--attention", choices=("sdpa", "sage"), default="sage")
    ap.add_argument("--vram16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--decode", choices=("once", "perclip", "both"), default="both")
    ap.add_argument("--save-latents", action="store_true")
    ap.add_argument("--reuse-clip0", metavar="C1_JSON", help="C1's report; clip 0 comes from its saved latents")
    ap.add_argument("--fingerprint", action="store_true")
    ap.add_argument("--recheck-clip0", action="store_true")
    ap.add_argument("--out-dir", help=f"mp4 / .pt folder (default {OUT_ROOT}\\<name>)")
    ap.add_argument("--report")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck(args.report)
    if not args.name:
        ap.error("--name is required (or --selfcheck)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
