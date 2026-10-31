r"""§1-109 の計測: Ref2VA の重みでの「固定行の連結＋クリップごとの参照」。1 条件＝1 プロセス。
複製元 c_chain.py（§183、Docs/Outputs-archive/h3-chain-2026-10-29/）との違い（計画 pure-knitting-shell.md §5 (1)-(9)）:
  (1) --transformer PATH（必須）で重みを選び、weight_identity.json の sha256 を sha256_cached に記録する
  (2) --refs REFS_JSON でクリップごとの参照（H3ReferenceInput）を渡す。clips[i].references に記録
  (3) run_clip は製品 generate() の参照の 4 手（_read_references／_encode(refs=)／reference_encode／_denoise(refs=)）
  (4) 強さ 1.0・注入 clean（_denoise(video_inject_clean=1.0)）・復号 once に固定（strength／inject／decode を CLI から外した）
  (5) クリップ 0 の再利用の鍵に、クリップ 0 の参照と重みの sha256_cached を足した
  (6) --product-recheck（門 G-R）: クリップ 0 の潜在の _decode と製品 generate() の mp4 の sha256 を比べる
  (7)(8) seam_metrics に継ぎ目 PSNR（psnr_J・psnr_window_median・psnr_drop）と潜在の継ぎ目の比 ratio_lat_J
  (9) 自己検査を差し替え（参照・再利用の鍵・製品の署名・PSNR／潜在の比・--transformer）。解析は analyze_refchain

Measurement only: the product code is NOT changed. The script drives H3Pipeline's own stages clip by clip
and builds the carry-over itself (plan §4).

Run from the repo root (run_cr.ps1 does the list in the plan's order):

    set PYTHONPATH=%CD%
    set PYTHONUTF8=1
    .venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-refchain-2026-10-31\cr_chain.py --name c2_ref2va_same ^
        --transformer models\MiniMaxH3\Weights\minimax_h3_ref2va_pruned_int8_convrot.safetensors ^
        --prompts <archive>\prompts_c2.json --refs <archive>\refs_c2.json --attention sage --save-latents --report <json>
    ... --selfcheck            (CPU only, no weights; the last line is SELFCHECK PASS)

Method (plan §4): clip i (F frames, T' = latent_frames(F), A = audio_latents(F)) is generated with seed
``seed + i``, its own prompt and its own references; RoPE time starts at 0 in every clip. Clip i >= 1 gets
``video_fix = (clean_v, [0]*k + [1]*(T'-k))`` with the previous clip's last k = 5m+2 video latents in
``clean_v[:, :, :k]`` and ``audio_fix = (clean_a, [0]*ka + [1]*(A-ka))`` with the last ka latents of the
CONCATENATED audio in ``clean_a[:, :, :ka]`` (the rest is 0), where the overlap is O = 17m+5 frames,
S_i = i (F - O), J_i = S_i + O and ka_i = A_acc - round(S_i * 5/3). ``_denoise`` gets
``video_inject_clean=1.0`` (the product chain's value, 裁定 84-2). Concatenation: video
``cat(prev, cur[:, :, k:])``, audio ``cat(prev, cur[:, :, ka:])``; one decode of the whole chain (once.mp4).

Exit code 0 = done; 1 = failed (exception, OOM, a head that is not the previous tail, a refused argument);
2 = the fingerprint, the clip-0 recheck, the product recheck or the references / fixed-rows check did not
match. The JSON is always written.
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # the self-check must leave no __pycache__

# Before torch is imported (same as the worker). The self-check hides the GPU itself.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
if "--selfcheck" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import analyze_refchain as AC  # noqa: E402
from engine_h3 import denoise_mask as DM  # noqa: E402
from engine_h3 import pipeline_h3 as P  # noqa: E402
from engine_h3.constants import (  # noqa: E402
    AUDIO_SAMPLE_RATE,
    FRAME_RATE,
    REFERENCE_KINDS,
    audio_latents,
    latent_frames,
)

OUT_ROOT = ROOT / "outputs/h3-refchain-2026-10-31"
#: The other four files (the transformer comes from --transformer, plan §5 (1)).
PATHS = {
    "text_encoder": "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "video_vae": "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
    "tokenizer_dir": "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
}
#: The fingerprint's weights (G-F; Docs/Outputs-archive/h3-te-vram-2026-10-29/g_i5_t2v_fingerprint.cmd).
FL2VA_TRANSFORMER = "models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
#: The repo's weight hash cache (written by the app; read only, never computed here, plan §5 (1)).
IDENTITY_FILE = ROOT / "weight_identity.json"
#: The fingerprint prompt (Docs/Outputs-archive/h3-gate3/g2_selftest_g5prompt.cmd) = G-F's text on all clips.
FP_PROMPT = (
    "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in "
    "puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>"
)
FP_SHA = "0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9"
#: Load-time values: H3LoadOptions' defaults, as the selftest that took the fingerprint.
BLOCKS_ON_GPU = 8
PREFETCH = True
DETERMINISTIC = False
#: Fixed by the plan (§5 (4)): strength 1.0 (mask value 0), the product chain's clean inject, one decode.
STRENGTH = 1.0
INJECT_CLEAN = 1.0
#: g_i5_t2v_fingerprint.cmd's conditions.
FP_COND = {
    "width": 640, "height": 384, "clip_frames": 124, "seed": 42, "steps": 20, "shift_video": 12.0,
    "shift_audio": 3.0, "attention": "sdpa", "vram16": True, "prefetch": True, "blocks_on_gpu": 8,
    "deterministic": False,
}
#: What load_clip0 compares with the source report (plan §5 (5)): weights = path / bytes / sha256_cached.
REUSE_KEYS = ("width", "height", "clip_frames", "seed", "steps", "shift_video", "shift_audio", "attention",
              "vram16", "prefetch", "blocks_on_gpu", "deterministic", "weights", "clip0_references")
LATENT_C, AUDIO_C = 24, 32
SAMPLES_PER_FRAME = Fraction(AUDIO_SAMPLE_RATE, FRAME_RATE)   # 4000/3
#: Seam PSNR (plan §5 (7)): identical frames are capped here instead of inf.
PSNR_CAP = 99.0

#: Seams for the self-check (fake decoders / writer); None = the real ones.
HOOKS: dict = {"video_decoder": None, "audio_vae": None, "write_mp4": None}


def _log(msg: str) -> None:
    print(f"[cr_chain] {msg}", flush=True)


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


def _json_safe(o, bad: list):
    """A copy of ``o`` with every non-finite float replaced by None (each one appended to ``bad``)."""
    if isinstance(o, float):
        if math.isfinite(o):
            return o
        bad.append(o)
        return None
    if isinstance(o, dict):
        return {k: _json_safe(v, bad) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v, bad) for v in o]
    return o


def _mb(n) -> float:
    return round(n / 2**20, 1)


# -- geometry (plan §4) -------------------------------------------------------------------


def geometry(clip_frames: int, overlap: int | None, clips: int, strength: float = STRENGTH) -> dict:
    """The chain's frame / latent bookkeeping (pure). Only strength 1.0 is run here (plan §5 (4)); the
    parameter stays so the copied arithmetic is unchanged. ``J_lat`` (plan §5 (8)): the index, in the
    concatenated video latents, of clip i's first new latent = i (T' - k) + k (the fixed rows' border).
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
    J_lat = [i * (T - k) + k for i in range(1, N)]
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
        "F": F, "T": T, "A": A, "N": N, "strength": s, "O": O, "m": m, "k": k, "S": S, "J": J, "J_lat": J_lat,
        "ka": ka, "F_total": F_total, "T_total": T_total, "A_concat": a_acc, "A_expected": a_expected,
        "audio_latent_excess": a_acc - a_expected,
        "audio_place_error_latents": None if s == 0.0 else [abs(e) for e in err],
        "audio_head_shift_latents": None if s == 0.0 else err,
        "audio_fixed_end_vs_video_latents": None if s == 0.0 else end_vs_video,
    }


# -- fixed rows, latents ---------------------------------------------------------------------


def build_fix(prev_v, prev_a_concat, shape_v, shape_a, k: int, ka: int, s: float = STRENGTH):
    """``(video_fix, audio_fix)`` for a clip >= 1 (CPU float32 clean, zero-filled past the head)."""
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


def latent_record(v: torch.Tensor, a: torch.Tensor) -> dict:
    vf, af = v.float(), a.float()
    return {"v_shape": list(v.shape), "a_shape": list(a.shape), "v_dtype": str(v.dtype),
            "v_sha256": _sha_tensor(v), "a_sha256": _sha_tensor(a),
            "finite": bool(torch.isfinite(vf).all() and torch.isfinite(af).all()),
            "v_abs_mean": float(vf.abs().mean()), "v_std": float(vf.std()),
            "a_abs_mean": float(af.abs().mean()), "a_std": float(af.std())}


# -- seam metrics (in memory, before H.264) --------------------------------------------------------


def _psnr_y(mse: float) -> float:
    """PSNR (dB, peak 255) of a luma MSE, capped at PSNR_CAP (identical frames -> PSNR_CAP, no inf)."""
    if mse <= 0.0:
        return PSNR_CAP
    return min(PSNR_CAP, 10.0 * math.log10(255.0 ** 2 / mse))


def frame_stats_torch(frames: torch.Tensor, batch: int = 8) -> dict:
    """Per-frame luma / chroma statistics, d(t) and the adjacent-frame luma PSNR(t) from uint8 RGB
    [F, H, W, 3] (analyze_refchain's definitions; ``psnr`` like ``d``: index 0 is t = 1)."""
    coef = torch.tensor([AC.Y_COEF, AC.CB_COEF, AC.CR_COEF], dtype=torch.float64).T   # [3 rgb, 3 ycc]
    out = {"frames": int(frames.shape[0]), "y_mean": [], "y_sq_mean": [], "chroma_mean": [], "d": [], "psnr": []}
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
                diff = yj - prev
                out["d"].append(float(diff.abs().mean()))
                out["psnr"].append(_psnr_y(float((diff * diff).mean())))
            prev = yj
        del ycc, y, chroma
    return out


def psnr_seam(psnr: list, j: int, half: int = AC.RATIO_HALF, exclude: int = AC.RATIO_EXCLUDE) -> dict:
    """Plan §5 (7): psnr_J = PSNR(frame[J-1], frame[J]); psnr_window_median = the median of PSNR(u) for u in
    [J-12, J+12] with |u - J| > 1 (inside 1 .. F-1); psnr_drop = median - psnr_J (dB). None when undefined."""
    here = AC.d_at(psnr, j)
    ref = [AC.d_at(psnr, u) for u in range(j - half, j + half + 1) if abs(u - j) > exclude]
    ref = [v for v in ref if v is not None]
    med = float(statistics.median(ref)) if ref else None
    return {"psnr_J": here, "psnr_window_median": med,
            "psnr_drop": None if here is None or med is None else med - here}


def latent_d(v: torch.Tensor) -> list:
    """Plan §5 (8): d_lat(t) = mean |z_t - z_(t-1)| over the video latents [1, 24, T, h, w] (index 0 is t = 1)."""
    z = v.detach().to("cpu", torch.float32)
    return [float(x) for x in (z[:, :, 1:] - z[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))]


def seam_metrics(frames: torch.Tensor, geom: dict, v_lat: torch.Tensor | None = None) -> dict:
    """analyze_refchain.summarize on the in-memory frames, plus per seam the PSNR fields (plan §5 (7)) and,
    from the concatenated video latents, ``J_lat`` / ``d_lat_J`` / ``ratio_lat_J`` (plan §5 (8): the same
    window as ratio_J, +-12 without |u - J_lat| <= 1). Record only: no criterion uses them."""
    stats = frame_stats_torch(frames)
    summary = AC.summarize(stats, geom)
    d_lat = latent_d(v_lat) if v_lat is not None else None
    for seam, jl in zip(summary["seams"], geom["J_lat"]):
        seam.update(psnr_seam(stats["psnr"], seam["J"]))
        seam["J_lat"] = jl
        seam["d_lat_J"] = None if d_lat is None else AC.d_at(d_lat, jl)
        seam["ratio_lat_J"] = None if d_lat is None else AC.jump_ratio(d_lat, jl)
    return {"stats": stats, "summary": summary, "d_lat": d_lat}


# -- weights and references (plan §5 (1)(2)) -------------------------------------------------------


def weight_identity(path: Path, identity_file: Path | None = None) -> str | None:
    """The cached sha256 of ``path`` from weight_identity.json when its entry (key = the lower-cased
    absolute path) has the file's size and mtime_ns; None otherwise (no file, no entry, stale entry)."""
    f = IDENTITY_FILE if identity_file is None else identity_file
    try:
        data = json.loads(Path(f).read_text(encoding="utf-8-sig"))
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


def weights_record(transformer, identity_file: Path | None = None) -> dict:
    """``condition.weights``: path (repo-relative) and bytes of every file, and the transformer's
    ``sha256_cached`` when weight_identity.json has a matching entry (plan §5 (1)). The transformer must exist."""
    tp = _abs(transformer)
    if not tp.is_file():
        raise ValueError(f"--transformer not found: {transformer}")
    out = {"transformer": {"path": Path(_rel(tp)).as_posix(), "bytes": tp.stat().st_size}}
    sha = weight_identity(tp, identity_file)
    if sha is not None:
        out["transformer"]["sha256_cached"] = sha
    for key, rel in PATHS.items():
        p = ROOT / rel
        out[key] = {"path": rel, "bytes": p.stat().st_size if p.is_file() else None}
    return out


def is_fl2va(transformer) -> bool:
    return _abs(transformer).resolve() == (ROOT / FL2VA_TRANSFORMER).resolve()


def load_refs(path: str | None, n: int) -> list:
    """``--refs``: a JSON list with one element per clip, each a list of ``{"kind", "slot", "path"}`` (path
    relative to the repo root; ``[]`` = no reference on that clip). Refused (ValueError): another element
    count, an unknown kind, a slot that is not an integer >= 1 or repeats a kind's slot in one clip, an empty
    or missing path. Returns per clip ``[{"kind", "slot", "path" (repo-relative, '/'), "abs", "file_sha256"}]``
    in the engine's order (pictures, videos, audios; each by slot). No file: n empty lists."""
    if not path:
        return [[] for _ in range(n)]
    items = json.loads(_abs(path).read_text(encoding="utf-8-sig"))
    if not isinstance(items, list) or len(items) != n:
        got = len(items) if isinstance(items, list) else type(items).__name__
        raise ValueError(f"--refs must be a JSON list of {n} elements, one per clip ({path}: {got})")
    shas: dict = {}
    out = []
    for i, clip in enumerate(items):
        if not isinstance(clip, list):
            raise ValueError(f"--refs clip {i}: a list of references expected ({clip!r})")
        seen = set()
        entries = []
        for ref in clip:
            if not isinstance(ref, dict):
                raise ValueError(f"--refs clip {i}: a reference must be an object ({ref!r})")
            kind, slot, rel = ref.get("kind"), ref.get("slot"), ref.get("path")
            if kind not in REFERENCE_KINDS:
                raise ValueError(f"--refs clip {i}: kind {kind!r} (one of {REFERENCE_KINDS})")
            if isinstance(slot, bool) or not isinstance(slot, int) or slot < 1:
                raise ValueError(f"--refs clip {i}: slot {slot!r} (an integer >= 1)")
            if (kind, slot) in seen:
                raise ValueError(f"--refs clip {i}: {kind} slot {slot} twice")
            seen.add((kind, slot))
            if not isinstance(rel, str) or not rel.strip():
                raise ValueError(f"--refs clip {i}: empty path ({kind} {slot})")
            ap = _abs(rel).resolve()
            if not ap.is_file():
                raise ValueError(f"--refs clip {i}: file not found ({kind} {slot}: {rel})")
            if str(ap) not in shas:
                shas[str(ap)] = _sha_file(ap)
            entries.append({"kind": kind, "slot": slot, "path": Path(_rel(ap)).as_posix(), "abs": str(ap),
                            "file_sha256": shas[str(ap)]})
        entries.sort(key=lambda e: (REFERENCE_KINDS.index(e["kind"]), e["slot"]))
        out.append(entries)
    return out


def ref_keys(entries: list) -> list:
    """What identifies a clip's references in the record and the reuse key (no absolute path, plan Nit 5)."""
    return [{"kind": e["kind"], "slot": e["slot"], "path": e["path"], "file_sha256": e["file_sha256"]}
            for e in entries]


def req_references(entries: list) -> tuple:
    """``H3ReferenceInput`` (absolute path, so validate_request finds the file) for one clip."""
    return tuple(P.H3ReferenceInput(e["kind"], e["abs"], int(e["slot"])) for e in entries)


def references_record(entries: list, info: dict) -> list:
    """``clips[i].references`` (plan §5 (2)): kind, slot, path, file_sha256 from --refs; source_size, size,
    vision_tokens, rows from the engine's ``reference_encode`` record (and the TE's ``te_read``)."""
    eng = {(r.get("kind"), int(r.get("slot", 0))): r for r in info.get("references_engine") or []}
    te = {(r.get("kind"), int(r.get("slot", 0))): r for r in info.get("te_references") or []}
    out = []
    for key in ref_keys(entries):
        r = eng.get((key["kind"], key["slot"]), {})
        t = te.get((key["kind"], key["slot"]), {})
        vt = r.get("vision_tokens")
        out.append({**key, "source_size": r.get("source_size"), "size": r.get("size"),
                    "vision_tokens": t.get("vision_tokens") if vt is None else vt, "rows": r.get("rows"),
                    "seconds": r.get("seconds")})
    return out


def refs_fix_problems(index: int, crec: dict, n_refs: int) -> dict:
    """The script's own check of a generated clip (plan §7, Major 3). With references:
    (1) sequence.ref_video == the sum of the picture / video references' rows and one record per reference.
    Without: no ``ref_video`` key in sequence. Clip >= 1 also: (2) mask.video.min == 0 and
    (3) mask.video.inject_clean == 1.0 (the fixed rows and the clean inject went in with the references)."""
    seq = crec.get("sequence") or {}
    mask_v = (crec.get("mask") or {}).get("video") or {}
    refs = crec.get("references") or []
    problems = []
    rec: dict = {"references": n_refs}
    if n_refs:
        rows = [r.get("rows") for r in refs if r.get("kind") != "audio"]
        total = None if any(x is None for x in rows) else sum(int(x) for x in rows)
        rec.update({"ref_video": seq.get("ref_video"), "ref_rows_sum": total})
        if len(refs) != n_refs:
            problems.append(f"{len(refs)} reference records for {n_refs} references")
        if total is None or seq.get("ref_video") != total:
            problems.append(f"sequence.ref_video {seq.get('ref_video')!r} != rows {total!r}")
    else:
        rec["ref_video_key"] = "ref_video" in seq
        if "ref_video" in seq:
            problems.append("sequence has ref_video without references")
    if index > 0:
        rec.update({"mask_video_min": mask_v.get("min"), "inject_clean": mask_v.get("inject_clean")})
        if mask_v.get("min") != 0.0:
            problems.append(f"mask.video.min {mask_v.get('min')!r} != 0")
        if mask_v.get("inject_clean") != INJECT_CLEAN:
            problems.append(f"mask.video.inject_clean {mask_v.get('inject_clean')!r} != {INJECT_CLEAN}")
    rec["problems"] = problems
    rec["ok"] = not problems
    return rec


# -- the pipeline ------------------------------------------------------------------------------


def make_pipeline(args, rec: dict):
    from engine_h3.pipeline_h3 import H3LoadOptions, H3Paths, H3Pipeline

    options = H3LoadOptions(blocks_on_gpu=BLOCKS_ON_GPU, attention_backend=args.attention,
                            block_swap_prefetch=PREFETCH, deterministic=DETERMINISTIC)
    paths = H3Paths(transformer=str(_abs(args.transformer)), **{key: str(ROOT / rel) for key, rel in PATHS.items()})
    pipe = H3Pipeline(paths, options, device=torch.device("cuda"), progress=_progress, log=_log)
    rec["build_report"] = pipe.load()
    return pipe


def clip_request(args, prompt: str, out_dir: Path):
    """Clip 0's request without references; clip i is ``dataclasses.replace(req, seed=seed+i, prompt=p_i,
    references=...)`` (plan §4 1.)."""
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
    """One clip, the product generate()'s reference steps in its order (pipeline_h3.py generate(); plan §4):
    ``_read_references(req)`` -> the "encode" phase (``_encode(..., refs=_te_references(media))``) -> with
    references the "reference_encode" phase (one video encoder, ``_encode_references``, dropped) -> the
    "denoise" phase (``_denoise(..., refs=rows, video_inject_clean=1.0)``, keep=True). The two encode phases
    are side by side, not nested; all inside ``torch.no_grad()``. The latents come back on the CPU in float32."""
    phases = P._Phases(cuda=pipe.device.type == "cuda")
    rows = None
    with torch.no_grad():
        media = pipe._read_references(req)
        with phases.phase("encode") as e:
            hidden, tags = pipe._encode(req.prompt, phases, (), has_cond=False, refs=pipe._te_references(media))
            e["text_tokens"] = int(hidden.shape[0])
        if media:
            with phases.phase("reference_encode") as e:
                visual = any(m["kind"] != "audio" for m in media)
                enc = pipe._build_video_encoder() if visual else None
                te = phases.data["encode"].get("te_read", {}).get("references", [])
                total = P.encode_steps(has_cond=False, has_audio=False, has_refs=True)
                rows = pipe._encode_references(media, e, total, total, enc, te)
                enc = None
                P._free_cuda()
        del media
        with phases.phase("denoise") as entry:
            v, a, used = pipe._denoise(req, hidden, tags, phases, entry, keep=True,
                                       video_fix=video_fix, audio_fix=audio_fix, refs=rows,
                                       video_inject_clean=INJECT_CLEAN)
        v, a = v.cpu(), a.cpu()
    del hidden, tags, rows
    P._free_cuda()
    enc_d, ref_d, den = (phases.data.get("encode", {}), phases.data.get("reference_encode", {}),
                         phases.data.get("denoise", {}))
    info = {
        "seconds": {"encode": enc_d.get("seconds"), "reference_encode": ref_d.get("seconds"),
                    "denoise": den.get("seconds")},
        "text_tokens": enc_d.get("text_tokens"), "dit_source": den.get("dit_source"),
        "sequence": den.get("sequence"), "m_per_step": den.get("m_per_step"), "mask": den.get("mask"),
        "vram16": den.get("vram16"), "block_swap_prefetch_used": used,
        "peak_vram_mb": {"encode": enc_d.get("peak_vram_mb"), "reference_encode": ref_d.get("peak_vram_mb"),
                         "denoise": den.get("peak_vram_mb")},
        "peak_vram_reserved_mb": {"encode": enc_d.get("peak_vram_reserved_mb"),
                                  "reference_encode": ref_d.get("peak_vram_reserved_mb"),
                                  "denoise": den.get("peak_vram_reserved_mb")},
        "references_engine": ref_d.get("references") or [],
        "te_references": (enc_d.get("te_read") or {}).get("references") or [],
        "mem": P._mem_gib(), "phases": phases.data,
    }
    return v, a, info


def load_clip0(src: dict, args, prompt0: str, weights: dict, refs: list, shape_v, shape_a) -> dict:
    """``--reuse-clip0``: clip 0 from the source report's saved latents, only if it ran under the same
    conditions (REUSE_KEYS: the weights' path / bytes / sha256_cached and clip 0's references included), the
    same clip-0 prompt, and the file's latents hash to what it recorded (plan §5 (5)). Raises ValueError
    otherwise. Returns the latents and the source's ``clips[0].references`` (copied into this report)."""
    problems = []
    if not src.get("ok"):
        problems.append("the source report is not ok")
    cond = src.get("condition") or {}
    mine = condition_record(args, weights, refs)
    for key in REUSE_KEYS:
        if cond.get(key) != mine.get(key):
            problems.append(f"{key}: source {cond.get(key)!r} != {mine.get(key)!r}")
    clips = src.get("clips") or []
    c0 = clips[0] if clips else {}
    if c0.get("source") != "generated":
        problems.append(f"source clip 0 is {c0.get('source')!r} (generated expected)")
    if c0.get("prompt_sha") != _sha_text(prompt0):
        problems.append("clip 0 prompt sha differs from the source's")
    src_refs = c0.get("references") or []
    if [{k: r.get(k) for k in ("kind", "slot", "path", "file_sha256")} for r in src_refs] != ref_keys(refs[0]):
        problems.append("the source's clips[0].references differ from this run's clip-0 references")
    if problems:
        raise ValueError("reuse-clip0 refused: " + "; ".join(problems))
    path = c0.get("latents_path")
    if not path or not _abs(path).is_file():
        raise ValueError(f"reuse-clip0 refused: latents file missing ({path})")
    data = torch.load(_abs(path))
    v, a = data["video"], data["audio"]
    lat = c0.get("latents") or {}
    if _sha_tensor(v) != lat.get("v_sha256") or _sha_tensor(a) != lat.get("a_sha256"):
        raise ValueError("reuse-clip0 refused: the loaded latents' sha256 differs from the source report")
    if tuple(v.shape) != tuple(shape_v) or tuple(a.shape) != tuple(shape_a):
        raise ValueError(f"reuse-clip0 refused: shapes {list(v.shape)} / {list(a.shape)}")
    return {"video": v, "audio": a, "path": path, "seed": c0.get("seed"), "references": copy.deepcopy(src_refs)}


def condition_record(args, weights: dict, refs: list) -> dict:
    return {
        "width": int(args.width), "height": int(args.height), "clip_frames": int(args.clip_frames),
        "clips": int(args.clips), "overlap_arg": args.overlap, "strength": STRENGTH,
        "inject": "clean", "video_inject_clean": INJECT_CLEAN, "seed": int(args.seed), "steps": int(args.steps),
        "shift_video": float(args.shift_video), "shift_audio": float(args.shift_audio),
        "attention": args.attention, "vram16": bool(args.vram16), "prefetch": PREFETCH,
        "blocks_on_gpu": BLOCKS_ON_GPU, "deterministic": DETERMINISTIC, "decode": "once",
        "save_latents": bool(args.save_latents), "reuse_clip0": args.reuse_clip0,
        "fingerprint": bool(args.fingerprint), "recheck_clip0": bool(args.recheck_clip0),
        "product_recheck": bool(args.product_recheck), "prompts_file": args.prompts, "refs_file": args.refs,
        "transformer_arg": args.transformer, "weights": weights,
        "clip0_references": ref_keys(refs[0]) if refs else [],
        "references_per_clip": [ref_keys(r) for r in refs],
    }


def fingerprint_problems(args, prompts: list, refs: list | None = None) -> list:
    """``--fingerprint``: FP_COND, the fingerprint prompt for clip 0, the FL2VA transformer (plan Nit 4), no
    reference, a generated clip 0."""
    have = {"width": int(args.width), "height": int(args.height), "clip_frames": int(args.clip_frames),
            "seed": int(args.seed), "steps": int(args.steps), "shift_video": float(args.shift_video),
            "shift_audio": float(args.shift_audio), "attention": args.attention, "vram16": bool(args.vram16),
            "prefetch": PREFETCH, "blocks_on_gpu": BLOCKS_ON_GPU, "deterministic": DETERMINISTIC}
    out = [f"{k}: {have[k]!r} != {v!r}" for k, v in FP_COND.items() if have[k] != v]
    if prompts[0] != FP_PROMPT:
        out.append("clip 0 prompt is not the fingerprint prompt")
    if not is_fl2va(args.transformer):
        out.append(f"transformer is not FL2VA ({args.transformer})")
    if refs and any(refs):
        out.append("--refs with --fingerprint (the fingerprint has no reference)")
    if args.reuse_clip0:
        out.append("--reuse-clip0 with --fingerprint")
    return out


def load_prompts(path: str | None, n: int) -> list:
    """JSON list of strings; shorter than n repeats its last entry."""
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


def _finish_decode(path: Path, frames, wave, geom: dict, rec: dict, v_lat=None) -> dict:
    if int(frames.shape[0]) != geom["F_total"]:
        raise RuntimeError(f"{path.name}: {frames.shape[0]} frames, {geom['F_total']} expected")
    t0 = time.perf_counter()
    rec["seam_metrics"] = seam_metrics(frames, geom, v_lat)
    rec["seam_seconds"] = round(time.perf_counter() - t0, 2)
    rec["size_bytes"] = int(_write(path, frames, wave))
    rec.update({"path": _rel(path), "sha256": _sha_file(path), "frames": int(frames.shape[0]),
                "audio_samples": int(wave.shape[-1]),
                "audio_samples_muxed": min(int(wave.shape[-1]), _round_frac(geom["F_total"] * SAMPLES_PER_FRAME))})
    return rec


def decode_once(pipe, v, a, geom: dict, path: Path) -> dict:
    """(a): one decode of the whole chain (the product's one decode, 裁定 84-4)."""
    t0 = time.perf_counter()
    frames, vi = decode_frames(pipe, v)
    wave, ai = decode_wave(pipe, a)
    rec = {"peak_vram_mb_video": vi["peak_vram_mb"], "peak_vram_mb_audio": ai["peak_vram_mb"],
           "decode_seconds": {"video": vi["seconds"], "audio": ai["seconds"]}}
    _finish_decode(path, frames, wave, geom, rec, v_lat=v)
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


def recheck_clip0(pipe, req_clip0, v0, a0) -> dict:
    """Clip 0 again with the resident DiT from ``req_clip0`` (its references included, plan Major 1);
    its latents must be torch.equal."""
    P._free_cuda()
    v, a, info = run_clip(pipe, req_clip0, None, None)
    return {"equal_video": bool(torch.equal(v, v0)), "equal_audio": bool(torch.equal(a, a0)),
            "max_abs_video": float((v - v0).abs().max()), "max_abs_audio": float((a - a0).abs().max()),
            "dit_source": info["dit_source"], "seconds": info["seconds"],
            "references": {"count": len(req_clip0.references), "engine": info["references_engine"]}}


def _audio_vae_source(pipe, entry: dict) -> str:
    """``_decode`` writes ``audio_vae_source`` only when it skips the audio VAE; otherwise it used a kept
    one ("resident") or built one ("file"), which the kept parts tell before the call."""
    return entry.get("audio_vae_source") or ("resident" if "audio_vae" in (pipe._resident or {}) else "file")


def product_recheck(pipe, req_clip0, v0, a0, crec0: dict, out_dir: Path) -> dict:
    """G-R (plan §5 (6)): before ``release_resident()``. First the script's clip-0 latents through
    ``pipe._decode(req_clip0, ...)`` -> clip0_script.mp4, then the product's own one-shot
    ``pipe.generate(replace(req_clip0, output_path=clip0_product.mp4))``; the two mp4 sha256 must match.
    Also kept for a mismatch's diagnosis: both decoders' sources, and the product's
    ``h3.phases.reference_encode.references`` / ``h3.phases.denoise.sequence`` against the script's clip 0."""
    script_path, prod_path = out_dir / "clip0_script.mp4", out_dir / "clip0_product.mp4"
    entry: dict = {}
    t0 = time.perf_counter()
    audio_src = _audio_vae_source(pipe, {})
    with torch.no_grad():
        size_s = pipe._decode(dataclasses.replace(req_clip0, output_path=str(script_path)),
                              v0.to(pipe.device), a0.to(pipe.device), entry, keep=False)
    script = {"path": _rel(script_path), "sha256": _sha_file(script_path), "size_bytes": int(size_s),
              "vae_source": entry.get("vae_source"), "audio_vae_source": entry.get("audio_vae_source") or audio_src,
              "entry": entry, "seconds": round(time.perf_counter() - t0, 2)}
    audio_src = _audio_vae_source(pipe, {})
    res = pipe.generate(dataclasses.replace(req_clip0, output_path=str(prod_path)))
    ph = ((res or {}).get("h3") or {}).get("phases") or {}
    dec = ph.get("decode") or {}
    prod_refs = (ph.get("reference_encode") or {}).get("references") or []
    prod_seq = (ph.get("denoise") or {}).get("sequence")
    product = {"path": _rel(prod_path), "sha256": _sha_file(prod_path), "size_bytes": res.get("size_bytes"),
               "vae_source": dec.get("vae_source"), "audio_vae_source": dec.get("audio_vae_source") or audio_src,
               "dit_source": (ph.get("denoise") or {}).get("dit_source"), "seconds": res.get("seconds"),
               "references": prod_refs, "sequence": prod_seq, "h3_references": (res.get("h3") or {}).get("references")}
    mine = {(r.get("kind"), r.get("slot")): r for r in crec0.get("references") or []}
    ref_cmp = []
    for r in prod_refs:
        m = mine.get((r.get("kind"), r.get("slot")), {})
        ref_cmp.append({"kind": r.get("kind"), "slot": r.get("slot"),
                        **{f"{k}_equal": r.get(k) == m.get(k) for k in ("rows", "vision_tokens", "size")}})
    refs_equal = len(prod_refs) == len(mine) and all(all(v for k, v in c.items() if k.endswith("_equal"))
                                                    for c in ref_cmp)
    return {"match": script["sha256"] == product["sha256"], "script": script, "product": product,
            "references_compare": ref_cmp, "references_equal": refs_equal,
            "sequence_equal": prod_seq == crec0.get("sequence"), "seconds": round(time.perf_counter() - t0, 2)}


# -- the run ------------------------------------------------------------------------------------------


def run(args, pipe=None) -> int:
    rec: dict = {"name": args.name, "torch": torch.__version__, "clips": [], "decode": {}, "mem": {}}
    report = Path(args.report) if args.report else HERE / f"{args.name}.json"
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / args.name
    ok = False
    exit_code = 1
    mismatch = []
    t_start = time.perf_counter()
    try:
        geom = geometry(args.clip_frames, args.overlap, args.clips, STRENGTH)
        rec["geometry"] = geom
        if not args.transformer:
            raise ValueError("--transformer is required")
        weights = weights_record(args.transformer)
        refs = load_refs(args.refs, geom["N"])
        rec["condition"] = condition_record(args, weights, refs)
        prompts = load_prompts(args.prompts, geom["N"])
        if args.fingerprint:
            problems = fingerprint_problems(args, prompts, refs)
            if problems:
                raise ValueError("--fingerprint refused: " + "; ".join(problems))
        if args.recheck_clip0 and args.reuse_clip0:
            raise ValueError("--recheck-clip0 needs a generated clip 0 (not --reuse-clip0)")
        if args.product_recheck and args.reuse_clip0:
            raise ValueError("--product-recheck needs a generated clip 0 (not --reuse-clip0)")
        lh, lw = int(args.height) // 16, int(args.width) // 16
        shape_v = (1, LATENT_C, geom["T"], lh, lw)
        shape_a = (2, AUDIO_C, geom["A"])
        reuse = None
        if args.reuse_clip0:
            src = json.loads(_abs(args.reuse_clip0).read_text(encoding="utf-8-sig"))
            reuse = load_clip0(src, args, prompts[0], weights, refs, shape_v, shape_a)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. the pipeline
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

        # 2. the clips
        vs, as_ = [], []
        acc_a = None
        req_clip0 = None
        for i in range(geom["N"]):
            req = dataclasses.replace(req0, seed=int(args.seed) + i, prompt=prompts[i],
                                      references=req_references(refs[i]))
            P.validate_request(req)
            if i == 0:
                req_clip0 = req      # the recheck, G-R's generate and _decode use this one (plan Major 1)
            crec: dict = {"index": i, "seed": req.seed, "prompt": req.prompt, "prompt_sha": _sha_text(req.prompt)}
            rec["clips"].append(crec)
            if i == 0 and reuse is not None:
                v, a = reuse["video"], reuse["audio"]
                crec.update({"source": "reused", "reused_from": args.reuse_clip0, "latents_path": reuse["path"],
                             "references": reuse["references"], "references_source": "reused"})
                _log(f"clip 0 reused from {reuse['path']}")
            else:
                vfix, afix = (None, None)
                if i > 0:
                    k, ka = geom["k"], geom["ka"][i - 1]
                    vfix, afix = build_fix(vs[-1], acc_a, shape_v, shape_a, k, ka, STRENGTH)
                    crec["fix"] = {"k": k, "ka": ka, "mask_value": 1.0 - STRENGTH}
                P._free_cuda()
                _log(f"clip {i}: seed {req.seed}, {len(refs[i])} reference(s), "
                     f"fix {'none' if vfix is None else 'k=%d ka=%d' % (k, ka)}")
                v, a, info = run_clip(pipe, req, vfix, afix)
                del vfix, afix
                crec["source"] = "generated"
                crec.update(info)
                crec["references"] = references_record(refs[i], info)
                crec["references_source"] = "generated"
                crec["refs_and_fix"] = refs_fix_problems(i, crec, len(refs[i]))
                if not crec["refs_and_fix"]["ok"] and "refs_and_fix" not in mismatch:
                    mismatch.append("refs_and_fix")
            crec["latents"] = latent_record(v, a)
            if tuple(v.shape) != shape_v or tuple(a.shape) != shape_a:
                raise RuntimeError(f"clip {i}: latents {list(v.shape)} / {list(a.shape)}, "
                                   f"expected {list(shape_v)} / {list(shape_a)}")
            if i > 0:
                k, ka = geom["k"], geom["ka"][i - 1]
                hv = (v[:, :, :k] - vs[-1][:, :, -k:]).abs().max() if k else torch.tensor(0.0)
                ha = (a[:, :, :ka] - acc_a[:, :, -ka:]).abs().max() if ka else torch.tensor(0.0)
                eq_v = bool(torch.equal(v[:, :, :k], vs[-1][:, :, -k:]))
                eq_a = bool(torch.equal(a[:, :, :ka], acc_a[:, :, -ka:])) if ka else True
                crec["head_equals_prev_tail"] = eq_v and eq_a
                crec["head_equals_prev_tail_video"] = eq_v
                crec["head_equals_prev_tail_audio"] = eq_a
                crec["head_vs_prev_tail_max_abs"] = {"video": float(hv), "audio": float(ha)}
                if not crec["head_equals_prev_tail"]:
                    raise RuntimeError(f"clip {i}: strength 1.0 but the head is not the previous tail "
                                       f"(max |d| video {float(hv)}, audio {float(ha)})")
            if args.save_latents and crec.get("source") == "generated":
                lp = out_dir / f"clip{i}.pt"
                torch.save({"video": v, "audio": a, "name": args.name, "index": i, "seed": req.seed,
                            "prompt": req.prompt, "references": crec.get("references")}, lp)
                crec["latents_path"] = _rel(lp)
            vs.append(v)
            as_.append(a)
            acc_a = a if i == 0 else torch.cat([acc_a, a[:, :, geom["ka"][i - 1]:]], dim=2)
            rec["mem"][f"after_clip{i}"] = P._mem_gib()
        if int(acc_a.shape[2]) != geom["A_concat"]:
            raise RuntimeError(f"concatenated audio {acc_a.shape[2]} latents, geometry says {geom['A_concat']}")

        # 3. recheck (G-F / G-R): clip 0 again from req_clip0
        if args.recheck_clip0:
            rec["recheck"] = recheck_clip0(pipe, req_clip0, vs[0], as_[0])
            if not (rec["recheck"]["equal_video"] and rec["recheck"]["equal_audio"]):
                mismatch.append("recheck")
        # 4. G-R: the product's one-shot generate, still with the resident DiT (plan §5 (6))
        if args.product_recheck:
            rec["product_recheck"] = product_recheck(pipe, req_clip0, vs[0], as_[0], rec["clips"][0], out_dir)
            if not rec["product_recheck"]["match"]:
                mismatch.append("product_recheck")
        # 5. the DiT goes before any decode
        pipe.release_resident()
        P._free_cuda()
        rec["mem"]["after_release"] = P._mem_gib()
        # 6. fingerprint (G-F), then (a)
        if args.fingerprint:
            rec["decode"]["fingerprint"] = fingerprint_check(pipe, vs[0], as_[0], req_clip0,
                                                             out_dir / "clip0_fingerprint.mp4")
            if not rec["decode"]["fingerprint"]["match"]:
                mismatch.append("fingerprint")
        cv, ca = concat_latents(vs, as_, geom)
        rec["decode"]["once"] = decode_once(pipe, cv, ca, geom, out_dir / "once.mp4")
        del cv, ca
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
        if pipe is not None:
            try:
                pipe.close()
            except Exception as exc:  # noqa: BLE001
                rec["close_error"] = repr(exc)
        if torch.cuda.is_available():
            rec["peak_alloc_mb_at_exit"] = _mb(torch.cuda.max_memory_allocated())
        rec["seconds_total"] = round(time.perf_counter() - t_start, 2)
        rec.setdefault("oom", False)
        rec.setdefault("mismatch", mismatch)
        rec["ok"] = ok
        report.parent.mkdir(parents=True, exist_ok=True)
        # No inf / NaN in the report (plan §5 (7)): any non-finite float becomes null and is counted.
        bad: list = []
        safe = _json_safe(rec, bad)
        if bad:
            safe["nonfinite_values_nulled"] = len(bad)
        text = json.dumps(safe, indent=2, ensure_ascii=False, default=str, allow_nan=False)
        report.write_text(text, encoding="utf-8")
        print(json.dumps({k: rec.get(k) for k in ("name", "ok", "error", "mismatch", "seconds_total")},
                         ensure_ascii=False, default=str))
        if rec.get("traceback"):
            print(rec["traceback"])
    return exit_code


# -- self-check (CPU, no weights; plan §5 (9)) ---------------------------------------------------------

#: The plan §4 geometry rows (strength 1.0). err = audio_head_shift_latents.
EXPECTED = {
    "O22": dict(o=22, O=22, k=7, S=[102, 204], J=[124, 226], J_lat=[37, 67], ka=[37, 37], F_total=328, T_total=97,
                A_concat=547, A_expected=547, excess=0, err=[0.0, 0.0]),
    "O5": dict(o=5, O=5, k=2, S=[119, 238], J=[124, 243], J_lat=[37, 72], ka=[9, 8], F_total=362, T_total=107,
               A_concat=604, A_expected=603, excess=1, err=[-1 / 3, 1 / 3]),
    "O39": dict(o=39, O=39, k=12, S=[85, 170], J=[124, 209], J_lat=[37, 62], ka=[65, 66], F_total=294, T_total=87,
                A_concat=490, A_expected=490, excess=0, err=[1 / 3, -1 / 3]),
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
    base = dict(name="sc", transformer=None, refs=None, clip_frames=124, clips=3, overlap=None, prompts=None,
                seed=42, width=1344, height=768, steps=20, shift_video=12.0, shift_audio=3.0, attention="sage",
                vram16=True, save_latents=False, reuse_clip0=None, fingerprint=False, recheck_clip0=False,
                product_recheck=False, report=None, out_dir=None)
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


class _FakeShape:
    def __init__(self, kind):
        self.kind = kind


class _FakePipe:
    """Records what run() asks of the pipeline: the reference steps (``_read_references`` /
    ``_te_references`` / ``_build_video_encoder`` / ``_encode_references``), ``_denoise`` (the real masks keep
    the m = 0 rows clean; ``sequence`` / ``mask`` written like the product's) and G-R's ``_decode`` /
    ``generate``. ``product_bytes``: what generate writes (the same as ``_decode`` = a match)."""

    def __init__(self, log, product_bytes=b"clip0"):
        self.device = torch.device("cpu")
        self._resident = None
        self.log = log
        self.prev_v = None
        self.acc_a = None
        self.product_bytes = product_bytes

    @staticmethod
    def _names(refs):
        return tuple((r.kind, int(r.slot), Path(r.path).name, Path(r.path).is_absolute()) for r in refs or ())

    def _read_references(self, req):
        refs = P.ordered_references(req.references)
        self.log.append(("read_refs", self._names(refs)))
        return [{"kind": r.kind, "slot": int(r.slot), "image": torch.zeros(1, 3, 1, 2, 2),
                 "info": {"source_size": [40, 20], "size": [32, 16]}} for r in refs]

    def _te_references(self, media):
        return [(m["kind"], m["slot"]) for m in media]

    def _build_video_encoder(self):
        self.log.append(("build_video_encoder",))
        return object()

    def _encode(self, prompt, phases, images=(), *, has_cond, has_audio=False, refs=None):
        refs = list(refs or ())
        self.log.append(("encode", "encode" in phases.data, tuple(images), has_cond, tuple(refs)))
        phases.data["encode"]["te_read"] = (
            {"references": [{"kind": k, "slot": s, "vision_tokens": 100 + s} for k, s in refs]} if refs else {})
        return torch.full((5, 8), float(len(prompt))), torch.ones(5, dtype=torch.long)

    def _encode_references(self, media, entry, index, total, video_enc, te_refs=()):
        tok = {(r["kind"], r["slot"]): r["vision_tokens"] for r in te_refs}
        self.log.append(("encode_refs", index, total, video_enc is not None, tuple(m["slot"] for m in media)))
        out, records = [], []
        for m in media:
            rows = torch.full((10 * m["slot"], 96), float(m["slot"]))
            records.append({"kind": m["kind"], "slot": m["slot"], "source_size": m["info"]["source_size"],
                            "size": m["info"]["size"], "vision_tokens": tok.get((m["kind"], m["slot"])),
                            "rows": int(rows.shape[0]), "seconds": 0.0})
            out.append((_FakeShape(m["kind"]), rows))
        entry["references"] = records
        return out

    def _denoise(self, req, hidden, tags, phases, entry, cond_clean=None, cond_frame_idx=(), keep=False,
                 lora_report=None, audio_fix=None, video_fix=None, refs=None, video_inject_clean=None):
        T, A = latent_frames(req.num_frames), audio_latents(req.num_frames)
        lh, lw = req.height // 16, req.width // 16
        g = torch.Generator().manual_seed(int(req.seed))
        noise_v = torch.randn((1, LATENT_C, T, lh, lw), generator=g)
        noise_a = torch.randn((2, AUDIO_C, A), generator=g)
        refs = list(refs or ())
        ref_n = sum(int(r.shape[0]) for s, r in refs if s.kind != "audio")
        v = noise_v * 0.5 + float(hidden.mean()) * 1e-3 + ref_n * 1e-4
        a = noise_a * 0.5
        head = None
        ka = 0
        entry["sequence"] = {"text": int(hidden.shape[0]), "video": T * lh * lw, "audio": 2 * A}
        if refs:
            entry["sequence"]["ref_video"] = ref_n
            entry["sequence"]["ref_audio"] = 0
        if video_fix is not None:
            k = int(sum(1 for x in video_fix[1] if x < 1))
            ka = int(sum(1 for x in audio_fix[1] if x < 1))
            cv, ca = video_fix[0], audio_fix[0]
            if self.prev_v is not None:    # None after a reused clip 0 (this fake never saw it)
                head = (bool(torch.equal(cv[:, :, :k], self.prev_v[:, :, -k:])) and bool((cv[:, :, k:] == 0).all())
                        and bool(torch.equal(ca[:, :, :ka], self.acc_a[:, :, -ka:]))
                        and bool((ca[:, :, ka:] == 0).all()))
            vm = DM.video_mask(cv, video_fix[1], grid=(req.height // 32) * (req.width // 32), noise=noise_v,
                               inject_clean=video_inject_clean)
            am = DM.audio_mask(ca, audio_fix[1])
            v, a = DM.finish(v, vm), DM.finish(a, am)
            entry["mask"] = {"audio": {"rows": int(am.rows.numel()), "min": float(am.rows.min()),
                                       "max": float(am.rows.max())},
                             "video": {"rows": int(vm.rows.numel()), "min": float(vm.rows.min()),
                                       "max": float(vm.rows.max())}}
            if video_inject_clean is not None:
                entry["mask"]["video"]["inject_clean"] = float(video_inject_clean)
        self.log.append(("denoise", keep, int(req.seed), video_fix is None, audio_fix is None, head,
                         tuple((s.kind, int(r.shape[0])) for s, r in refs), video_inject_clean,
                         self._names(req.references)))
        entry["dit_source"] = "resident" if self._resident else "file"
        if keep:
            self._resident = {**(self._resident or {}), "dit": object()}
        self.prev_v = v
        if video_fix is None or self.acc_a is None:
            self.acc_a = a
        else:
            self.acc_a = torch.cat([self.acc_a, a[:, :, ka:]], dim=2)
        return v, a, "fake"

    def _decode(self, req, video_lat, audio_lat, entry, keep=False, source_wave=None):
        self.log.append(("pdecode", Path(req.output_path).name, self._names(req.references),
                         "dit" in (self._resident or {})))
        entry["vae_source"] = "resident" if "decoder" in (self._resident or {}) else "file"
        Path(req.output_path).write_bytes(b"clip0")
        return 5

    def generate(self, req):
        self.log.append(("generate", Path(req.output_path).name, self._names(req.references), int(req.seed),
                         req.prompt, "dit" in (self._resident or {})))
        _v, _a, info = run_clip(self, req, None, None)
        Path(req.output_path).write_bytes(self.product_bytes)
        ph = info["phases"]
        ph["decode"] = {"vae_source": "file"}
        return {"size_bytes": len(self.product_bytes), "seconds": 0.0, "h3": {"phases": ph}}

    def release_resident(self):
        self.log.append(("release",))
        self._resident = None

    def close(self):
        self.log.append(("close",))


#: The fake runs' clip length: 17n+5 with audio_latents(F) >= F * 5/3 (F = 1 mod 3, as 124).
FAKE_F = 73


def _fake_run(tmp: Path, log: list, wrefs: list, pipe=None, **kw):
    """run() with the fake pipeline, decoders and writer; returns (exit code, report dict)."""
    def writer(path, frames, *, fps, audio, sample_rate):
        wrefs.append(__import__("weakref").ref(frames))
        log.append(("write", Path(path).name, int(frames.shape[0]), int(audio.shape[1])))
        Path(path).write_bytes(b"fake")
        return 4

    HOOKS.update({"video_decoder": lambda pipe: _FakeVideoDecoder(log), "audio_vae": lambda pipe: _FakeAudioVAE(log),
                  "write_mp4": writer})
    try:
        kw.setdefault("transformer", str(tmp / "fake_transformer.safetensors"))
        args = _args(width=64, height=64, clip_frames=FAKE_F, report=str(tmp / f"{kw.get('name', 'sc')}.json"),
                     out_dir=str(tmp / kw.get("name", "sc")), **kw)
        code = run(args, pipe=pipe if pipe is not None else _FakePipe(log))
    finally:
        HOOKS.update({"video_decoder": None, "audio_vae": None, "write_mp4": None})
    return code, json.loads(Path(args.report).read_text(encoding="utf-8"))


def selfcheck(report: str | None = None) -> int:
    import inspect
    import tempfile

    from engine_h3.media_out import write_mp4
    from engine_h3.scheduler import euler_step, sigma_schedule

    results: list[bool] = []
    _line(results, not torch.cuda.is_available(), f"0 CPU only (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    # the analysis module is this folder's, and the previous archive is not on sys.path (plan Nit 3)
    prev_arc = (ROOT / "Docs/Outputs-archive/h3-chain-2026-10-29").resolve()
    on_path = [p for p in sys.path if p and Path(p).resolve() == prev_arc]
    _line(results, Path(AC.__file__).resolve().parent == HERE and not on_path and "analyze_chain" not in sys.modules,
          f"0 analyze_refchain from this folder ({Path(AC.__file__).name}); h3-chain-2026-10-29 not on sys.path; "
          f"analyze_chain not imported")

    # 1. plan §4's table
    for label, e in EXPECTED.items():
        g = geometry(124, e["o"], 3)
        same = (g["F"], g["T"], g["A"]) == (124, 37, 207) and all(
            g[key] == e[key] for key in ("O", "k", "S", "J", "J_lat", "ka", "F_total", "T_total", "A_concat",
                                         "A_expected"))
        same = same and g["audio_latent_excess"] == e["excess"] and g["strength"] == 1.0
        same = same and all(abs(x - y) < 1e-12 for x, y in zip(g["audio_head_shift_latents"], e["err"]))
        same = same and all(0 <= x <= g["A"] for x in g["ka"]) and 2 * g["k"] <= g["T"]
        same = same and g["T_total"] == latent_frames(g["F_total"])
        _line(results, same, f"1 geometry {label}: S={g['S']} J={g['J']} J_lat={g['J_lat']} ka={g['ka']} "
                             f"F={g['F_total']} T'={g['T_total']} A={g['A_concat']} (excess {g['audio_latent_excess']:+d})")
    _line(results, all(_round_frac(Fraction(5 * n, 3)) == audio_latents(n) for n in range(0, 3000)),
          "1 round(n*5/3) (exact) == constants.audio_latents(n) for n < 3000")

    # 2. sweep
    valid = refused = 0
    bad = []
    for F in range(22, 482, 17):
        for O in (5, 22, 39):
            for N in range(2, 7):
                try:
                    g = geometry(F, O, N)
                except ValueError:
                    refused += 1
                    continue
                valid += 1
                T, k = g["T"], g["k"]
                inv = (g["T_total"] == latent_frames(g["F_total"]) and all(0 <= x <= g["A"] for x in g["ka"])
                       and all(j - s_ == O for s_, j in zip(g["S"], g["J"]))
                       and g["S"] == [i * (F - O) for i in range(1, N)]
                       and g["J_lat"] == [T + (i - 1) * (T - k) for i in range(1, N)]
                       and abs(g["audio_latent_excess"]) <= 1
                       and all(abs(x) <= 1 / 3 + 1e-12 for x in g["audio_head_shift_latents"])
                       and g["A_concat"] == g["A"] + sum(g["A"] - x for x in g["ka"])
                       and 2 * g["k"] <= g["T"] and O < F)
                if not inv:
                    bad.append((F, O, N))
    _line(results, not bad and valid > 0,
          f"2 sweep F 22..481 x O 5/22/39 x N 2..6: {valid} valid, {refused} refused, invariant failures {bad[:5]}")

    # 3. build_fix (strength 1.0)
    pv = torch.randn(1, 24, 37, 2, 3)
    pa = torch.randn(2, 32, 377)
    (cv, per_v), (ca, per_a) = build_fix(pv, pa, (1, 24, 37, 2, 3), (2, 32, 207), 7, 37)
    ok3 = (tuple(cv.shape) == (1, 24, 37, 2, 3) and tuple(ca.shape) == (2, 32, 207) and cv.dtype == torch.float32
           and torch.equal(cv[:, :, :7], pv[:, :, -7:]) and bool((cv[:, :, 7:] == 0).all())
           and torch.equal(ca[:, :, :37], pa[:, :, -37:]) and bool((ca[:, :, 37:] == 0).all())
           and per_v == [0.0] * 7 + [1.0] * 30 and per_a == [0.0] * 37 + [1.0] * 170)
    (_cv0, _pv0), (ca0, per_a0) = build_fix(pv, pa, (1, 24, 37, 2, 3), (2, 32, 207), 7, 0)
    ok3 = ok3 and bool((ca0 == 0).all()) and per_a0 == [1.0] * 207 and DM.audio_mask(ca0, per_a0) is None
    _line(results, ok3, "3 build_fix s=1: shapes, head = previous tail, zeros past k / ka, mask values 0 / 1, "
                        "ka=0 -> no audio mask")

    # 4. real masks: the default inject, inject_clean=1.0 by argument, the module value untouched
    g = torch.Generator().manual_seed(3)
    clean = torch.randn(1, 4, 5, 2, 2, generator=g)
    noise = torch.randn(1, 4, 5, 2, 2, generator=g)
    per = [0.0, 0.0, 0.5, 1.0, 1.0]
    vm = DM.video_mask(clean, per, grid=3, noise=noise)
    vm_c = DM.video_mask(clean, per, grid=3, noise=noise, inject_clean=INJECT_CLEAN)
    am = DM.audio_mask(torch.randn(2, 32, 6, generator=g), [0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    ok4 = (torch.equal(vm.rows, torch.tensor(per).repeat_interleave(3))
           and torch.equal(vm.inject, 0.999 * clean + (1.0 - 0.999) * noise)
           and torch.equal(vm_c.inject, clean) and torch.equal(vm_c.rows, vm.rows)
           and torch.equal(am.rows, torch.tensor([0.0, 0.0, 1, 1, 1, 1]).repeat(2)) and DM.VIDEO_INJECT_CLEAN == 0.999)
    _line(results, ok4, "4 video_mask: rows = per x grid; default inject 0.999/0.001; inject_clean=1.0 -> "
                        "inject == clean; DM.VIDEO_INJECT_CLEAN stays 0.999 (no module patch)")

    # 5. the product's step helpers with a dummy velocity (the clean inject)
    gen = torch.Generator().manual_seed(5)
    clean = torch.randn(1, 4, 5, 2, 2, generator=gen)
    x = torch.randn(1, 4, 5, 2, 2, generator=gen)
    vm = DM.video_mask(clean, [0.0, 0.0, 1.0, 1.0, 1.0], grid=4, noise=x.clone(), inject_clean=INJECT_CLEAN)
    sig = sigma_schedule(8, 12.0)
    for i in range(8):
        xin = DM.dit_input(x, vm)
        v = torch.sin(xin) - 0.3 * xin
        x = euler_step(x, DM.state_velocity(v, x, vm, float(sig[i])), float(sig[i]), float(sig[i + 1]))
    x = DM.finish(x, vm)
    _line(results, torch.equal(x[:, :, :2], clean[:, :, :2]) and bool(torch.isfinite(x).all()),
          "5 dit_input / state_velocity / euler_step / finish: m=0 rows end == clean")

    # 6. concatenation
    prev = torch.randn(1, 2, 10, 1, 1)
    cur = torch.randn(1, 2, 10, 1, 1)
    cur[:, :, :3] = prev[:, :, -3:]
    g22 = geometry(124, 22, 3)
    vs = [torch.randn(1, 24, 37, 1, 1) for _ in range(3)]
    as_ = [torch.randn(2, 32, 207) for _ in range(3)]
    cvl, cal = concat_latents(vs, as_, g22)
    _line(results, torch.equal(torch.cat([prev, cur[:, :, 3:]], 2), torch.cat([prev[:, :, :-3], cur], 2))
          and cvl.shape[2] == g22["T_total"] and cal.shape[2] == g22["A_concat"]
          and torch.equal(cvl[:, :, g22["J_lat"][0]], vs[1][:, :, g22["k"]])
          and torch.equal(cvl[:, :, g22["J_lat"][1]], vs[2][:, :, g22["k"]]),
          f"6 concat: cat(prev, cur[k:]) == cat(prev[:-k], cur); lengths {cvl.shape[2]} / {cal.shape[2]}; "
          f"concat[J_lat_i] = clip i's latent k (the first new one)")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        img1, img2 = tmp / "gentleman.png", tmp / "girl.png"
        img1.write_bytes(b"picture-1")
        img2.write_bytes(b"picture-2")
        (tmp / "fake_transformer.safetensors").write_bytes(b"weights")

        def refs_file(name, data):
            p = tmp / name
            p.write_text(json.dumps(data), encoding="utf-8")
            return str(p)

        r1 = {"kind": "picture", "slot": 1, "path": str(img1)}
        r2 = {"kind": "picture", "slot": 2, "path": str(img2)}
        same3 = refs_file("same3.json", [[r1], [r1], [r1]])
        per3 = refs_file("per3.json", [[r1], [r2, r1], [r1, r2]])

        # (a) --refs: reading and refusals
        got = load_refs(per3, 3)
        ok_a = ([[(e["kind"], e["slot"]) for e in c] for c in got] == [[("picture", 1)], [("picture", 1), ("picture", 2)],
                                                                         [("picture", 1), ("picture", 2)]]
                and got[0][0]["file_sha256"] == hashlib.sha256(b"picture-1").hexdigest()
                and Path(got[0][0]["abs"]).is_absolute() and "\\" not in got[0][0]["path"]
                and load_refs(None, 3) == [[], [], []])
        ok_a = ok_a and _raises(load_refs, refs_file("n2.json", [[r1], [r1]]), 3)
        ok_a = ok_a and _raises(load_refs, refs_file("missing.json", [[r1], [{**r1, "path": str(tmp / "no.png")}], []]), 3)
        ok_a = ok_a and _raises(load_refs, refs_file("dup.json", [[r1, {**r2, "slot": 1}], [], []]), 3)
        ok_a = ok_a and _raises(load_refs, refs_file("kind.json", [[{**r1, "kind": "image"}], [], []]), 3)
        ok_a = ok_a and _raises(load_refs, refs_file("slot0.json", [[{**r1, "slot": 0}], [], []]), 3)
        ok_a = ok_a and _raises(load_refs, refs_file("slotb.json", [[{**r1, "slot": True}], [], []]), 3)
        ok_a = ok_a and _raises(load_refs, refs_file("empty.json", [[{**r1, "path": " "}], [], []]), 3)
        _line(results, ok_a, "a --refs: per clip in the engine's order, file sha256, absolute path for the request "
                             "and repo-relative '/' path for the record, none -> []; refused: element count, missing "
                             "file, repeated slot, unknown kind, slot 0 / bool, empty path")

        # (a) this folder's inputs: refs_c2 / refs_c3 (the archived pictures), prompts_c0 / c2 / c3 (3 each)
        try:
            ar2, ar3 = load_refs(HERE / "refs_c2.json", 3), load_refs(HERE / "refs_c3.json", 3)
            pr = {n: load_prompts(str(HERE / f"prompts_{n}.json"), 3) for n in ("c0", "c2", "c3")}
            raw = {n: json.loads((HERE / f"prompts_{n}.json").read_text(encoding="utf-8-sig")) for n in pr}
            reqs = [dataclasses.replace(clip_request(_args(), "x", tmp), references=req_references(c)) for c in ar3]
            for q in reqs:
                P.validate_request(q)
            ok_in = ([len(c) for c in ar2] == [1, 1, 1] and [len(c) for c in ar3] == [1, 2, 2]
                     and all(e["path"].startswith("Docs/Outputs-archive/h3-ref2va-2026-10-28/") for c in ar3 for e in c)
                     and ar2[0] == ar3[0] and all(len(v) == 3 for v in raw.values())
                     and pr["c3"][0] == pr["c2"][0] and all(p_.startswith("<Picture 1> is the elderly gentleman. ")
                                                            for p_ in pr["c2"] + pr["c3"])
                     and not any("<Picture" in p_ for p_ in pr["c0"]))
        except (OSError, ValueError) as exc:
            ok_in, ar3 = False, [repr(exc)]
        _line(results, ok_in, f"a archive inputs: refs_c2 {[len(c) for c in ar2] if ok_in else '?'} / refs_c3 "
                              f"{[len(c) for c in ar3] if ok_in else ar3} pass validate_request; prompts 3 each, C3 "
                              f"clip 0 == C2 clip 0 (the reuse), the cards only in C2 / C3")

        # (b) run() with the references on the fake pipeline (C3's shape) + recheck + G-R
        log, wrefs = [], []
        code, rep = _fake_run(tmp, log, wrefs, name="fake_c3", prompts=None, refs=per3, save_latents=True,
                              recheck_clip0=True, product_recheck=True)
        gc.collect()
        den = [e for e in log if e[0] == "denoise"]
        enc = [e for e in log if e[0] == "encode"]
        rd = [e for e in log if e[0] == "read_refs"]
        er = [e for e in log if e[0] == "encode_refs"]
        idx = {e[0]: i for i, e in reversed(list(enumerate(log)))}
        gen_e = [e for e in log if e[0] == "generate"]
        pdec = [e for e in log if e[0] == "pdecode"]
        clips = rep.get("clips", [])
        want = [(("picture", 1, "gentleman.png", True),),
                (("picture", 1, "gentleman.png", True), ("picture", 2, "girl.png", True)),
                (("picture", 1, "gentleman.png", True), ("picture", 2, "girl.png", True))]
        ok_b = (code == 0 and rep.get("ok") and len(den) == 5   # 3 clips + recheck + G-R's generate
                and [e[2] for e in den] == [42, 43, 44, 42, 42]
                and [e[8] for e in den] == [want[0], want[1], want[2], want[0], want[0]]
                and [e[6] for e in den] == [(("picture", 10),), (("picture", 10), ("picture", 20)),
                                            (("picture", 10), ("picture", 20)), (("picture", 10),), (("picture", 10),)]
                and all(e[7] == INJECT_CLEAN for e in den[:4])
                and den[0][3] and not den[1][3] and den[1][5] and den[2][5]
                and len(rd) == 5 and len(er) == 5 and all(e[1] == e[2] == 4 and e[3] for e in er)
                and all(e[1] and e[2] == () and e[3] is False and len(e[4]) >= 1 for e in enc)
                and [len(c.get("references", [])) for c in clips] == [1, 2, 2]
                and all(set(r) >= {"kind", "slot", "path", "file_sha256", "source_size", "size", "vision_tokens", "rows"}
                        for c in clips for r in c["references"])
                and clips[1]["references"][1]["vision_tokens"] == 102 and clips[1]["references"][1]["rows"] == 20
                and all(c["refs_and_fix"]["ok"] for c in clips)
                and clips[1]["refs_and_fix"]["ref_video"] == 30 and clips[2]["refs_and_fix"]["inject_clean"] == 1.0
                and rep.get("recheck", {}).get("equal_video") and rep["recheck"]["references"]["count"] == 1
                and gen_e and gen_e[0][1] == "clip0_product.mp4" and gen_e[0][2] == want[0] and gen_e[0][3] == 42
                and gen_e[0][5] and pdec and pdec[0][1] == "clip0_script.mp4" and pdec[0][2] == want[0]
                and idx["pdecode"] < idx["generate"] < idx["release"]
                and rep["product_recheck"]["match"] and rep["product_recheck"]["sequence_equal"]
                and rep["product_recheck"]["references_equal"]
                and rep["product_recheck"]["script"]["vae_source"] == "file"
                and rep["product_recheck"]["product"]["audio_vae_source"] == "file"
                and all(c.get("head_equals_prev_tail") for c in clips[1:])
                and all(r() is None for r in wrefs) and len(wrefs) == 1)
        _line(results, ok_b, f"b fake run with per-clip references: exit {code}; _denoise got refs rows "
                             f"{[e[6] for e in den]} and video_inject_clean {sorted({e[7] for e in den[:4]})}; each clip "
                             f"its own references (absolute paths); recheck and G-R's generate / _decode from req_clip0 "
                             f"(with its reference); _decode -> generate -> release; refs_and_fix ok on every clip")
        # without references (C0 / C1 / G-F's shape): refs=[] to the TE, no reference phase, no ref_video
        log2 = []
        code2, rep2 = _fake_run(tmp, log2, [], name="fake_c0", save_latents=True)
        den2 = [e for e in log2 if e[0] == "denoise"]
        _line(results, code2 == 0 and not [e for e in log2 if e[0] in ("encode_refs", "build_video_encoder")]
              and all(e[6] == () and e[7] == INJECT_CLEAN for e in den2)
              and all(e[4] == () for e in log2 if e[0] == "encode")
              and all(c["references"] == [] and c["refs_and_fix"]["ok"] and not c["refs_and_fix"]["ref_video_key"]
                      for c in rep2["clips"]),
              "b fake run without references: no reference_encode, refs None to _denoise, no ref_video key, "
              "video_inject_clean 1.0")
        # G-R mismatch -> exit 2 / product_recheck
        code3, rep3 = _fake_run(tmp, [], [], pipe=_FakePipe([], product_bytes=b"other"), name="fake_gr_bad",
                                refs=same3, product_recheck=True)
        # refs_fix_problems on planted records
        good = {"sequence": {"ref_video": 30}, "mask": {"video": {"min": 0.0, "inject_clean": 1.0}},
                "references": [{"kind": "picture", "rows": 10}, {"kind": "picture", "rows": 20}]}
        bad_rows = copy.deepcopy(good)
        bad_rows["sequence"]["ref_video"] = 10
        bad_inj = copy.deepcopy(good)
        bad_inj["mask"]["video"]["inject_clean"] = 0.999
        bad_min = copy.deepcopy(good)
        bad_min["mask"]["video"]["min"] = 0.5
        noref_key = {"sequence": {"ref_video": 0}, "mask": {"video": {"min": 0.0, "inject_clean": 1.0}}, "references": []}
        _line(results, code3 == 2 and rep3["mismatch"] == ["product_recheck"] and not rep3["product_recheck"]["match"]
              and refs_fix_problems(1, good, 2)["ok"] and not refs_fix_problems(1, bad_rows, 2)["ok"]
              and not refs_fix_problems(1, bad_inj, 2)["ok"] and not refs_fix_problems(2, bad_min, 2)["ok"]
              and not refs_fix_problems(1, good, 3)["ok"] and refs_fix_problems(0, bad_min, 2)["ok"]
              and not refs_fix_problems(0, noref_key, 0)["ok"],
              "b G-R sha mismatch -> exit 2 'product_recheck'; refs_and_fix refuses ref_video != rows, "
              "inject_clean != 1, mask min != 0 (clip >= 1), a missing record, ref_video without references")

        # (c) the reuse key: C2 -> C3 accepted (references copied), other clip-0 references / weights refused
        code_c2, rep_c2 = _fake_run(tmp, [], [], name="fake_c2", refs=same3, save_latents=True)
        log4 = []
        code4, rep4 = _fake_run(tmp, log4, [], name="fake_c3r", refs=per3, save_latents=True,
                                reuse_clip0=str(tmp / "fake_c2.json"))
        den4 = [e for e in log4 if e[0] == "denoise"]
        c40 = rep4["clips"][0] if rep4.get("clips") else {}
        other0 = refs_file("other0.json", [[r2], [r1], [r1]])
        code5, rep5 = _fake_run(tmp, [], [], name="fake_c3x", refs=other0, reuse_clip0=str(tmp / "fake_c2.json"))
        (tmp / "other_transformer.safetensors").write_bytes(b"weights")    # same bytes, another path
        code6, rep6 = _fake_run(tmp, [], [], name="fake_c3w", refs=same3, reuse_clip0=str(tmp / "fake_c2.json"),
                                transformer=str(tmp / "other_transformer.safetensors"))
        code7, rep7 = _fake_run(tmp, [], [], name="fake_c3n", refs=None, reuse_clip0=str(tmp / "fake_c2.json"))
        _line(results, code_c2 == 0 and code4 == 0 and c40.get("source") == "reused"
              and c40.get("references_source") == "reused" and c40.get("references") == rep_c2["clips"][0]["references"]
              and [e[2] for e in den4] == [43, 44] and c40["latents"]["v_sha256"] == rep_c2["clips"][0]["latents"]["v_sha256"]
              and code5 == 1 and "clip0_references" in rep5.get("error", "")
              and code6 == 1 and "weights" in rep6.get("error", "")
              and code7 == 1 and "clip0_references" in rep7.get("error", ""),
              f"c reuse: C2 -> C3 accepted (clip 0 references copied, references_source reused, seeds "
              f"{[e[2] for e in den4]}); refused: other clip-0 references, other transformer path, none")
        # the old refusals (geometry, fingerprint, the reuse sha / prompt)
        ok_r = (_raises(geometry, 124, 23, 3) and _raises(geometry, 125, 22, 3) and _raises(geometry, 22, 22, 3)
                and _raises(geometry, 56, 39, 3) and _raises(geometry, 5, 5, 3) and _raises(geometry, 124, 22, 1))
        fl2va = str(ROOT / FL2VA_TRANSFORMER)
        fp_ok = fingerprint_problems(_args(width=640, height=384, attention="sdpa", transformer=fl2va),
                                     [FP_PROMPT] * 3, [[], [], []]) == []
        fp_bad = fingerprint_problems(_args(attention="sdpa", transformer=fl2va), [FP_PROMPT] * 3)
        fp_w = fingerprint_problems(_args(width=640, height=384, attention="sdpa",
                                          transformer="models/MiniMaxH3/Weights/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
                                    [FP_PROMPT] * 3, [[{"kind": "picture"}], [], []])
        src = json.loads((tmp / "fake_c2.json").read_text(encoding="utf-8"))
        a1 = _args(width=64, height=64, clip_frames=FAKE_F, transformer=str(tmp / "fake_transformer.safetensors"),
                   refs=same3)
        sv, sa = (1, 24, latent_frames(FAKE_F), 4, 4), (2, 32, audio_latents(FAKE_F))
        w = weights_record(a1.transformer)
        rs = load_refs(same3, 3)
        accepted = load_clip0(src, a1, FP_PROMPT, w, rs, sv, sa)["seed"] == 42
        refuse_prompt = _raises(load_clip0, src, a1, "another prompt", w, rs, sv, sa)
        bad_sha = copy.deepcopy(src)
        bad_sha["clips"][0]["latents"]["v_sha256"] = "0" * 64
        refuse_sha = _raises(load_clip0, bad_sha, a1, FP_PROMPT, w, rs, sv, sa)
        bad_ref = copy.deepcopy(src)
        bad_ref["clips"][0]["references"][0]["file_sha256"] = "0" * 64
        refuse_ref = _raises(load_clip0, bad_ref, a1, FP_PROMPT, w, rs, sv, sa)
        code_fp, rep_fp = _fake_run(tmp, [], [], name="fake_fp", fingerprint=True)
        _line(results, ok_r and fp_ok and len(fp_bad) == 2 and len(fp_w) == 2 and accepted and refuse_prompt
              and refuse_sha and refuse_ref and code_fp == 1 and "--fingerprint refused" in rep_fp.get("error", "")
              and "transformer is not FL2VA" in rep_fp.get("error", ""),
              "c refusals: geometry off the grid; --fingerprint off FP_COND / not FL2VA / with references; "
              "load_clip0 refuses another prompt, a latent sha and a clip-0 reference record mismatch, accepts the match")

        # (f) --transformer and weight_identity.json
        code_t, rep_t = _fake_run(tmp, [], [], name="fake_not", transformer=str(tmp / "missing.safetensors"))
        wt = tmp / "fake_transformer.safetensors"
        st = wt.stat()
        idf = tmp / "weight_identity.json"
        idf.write_text(json.dumps({"schema": 1, "entries": {str(wt.resolve()).lower(): {
            "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": "ab" * 32}}}), encoding="utf-8")
        hit = weights_record(str(wt), idf)["transformer"].get("sha256_cached")
        idf2 = tmp / "weight_identity_stale.json"
        idf2.write_text(json.dumps({"schema": 1, "entries": {str(wt.resolve()).lower(): {
            "size": st.st_size, "mtime_ns": st.st_mtime_ns + 1, "sha256": "ab" * 32}}}), encoding="utf-8")
        stale = "sha256_cached" in weights_record(str(wt), idf2)["transformer"]
        none = "sha256_cached" in weights_record(str(wt), tmp / "no_identity.json")["transformer"]
        real = weights_record(FL2VA_TRANSFORMER)["transformer"] if (ROOT / FL2VA_TRANSFORMER).is_file() else None
        _line(results, code_t == 1 and "--transformer not found" in rep_t.get("error", "") and hit == "ab" * 32
              and not stale and not none,
              f"f --transformer: a missing file refused; weight_identity.json: a matching entry -> sha256_cached, "
              f"a stale mtime or no file -> not recorded (no refusal); the repo's FL2VA entry: "
              f"{(real or {}).get('sha256_cached', 'not found')[:12]}")

    # (d) the product's signatures (a change in the product is detected here)
    pd = inspect.signature(P.H3Pipeline._denoise).parameters
    pe = inspect.signature(P.H3Pipeline._encode).parameters
    pdec = inspect.signature(P.H3Pipeline._decode).parameters
    per_ = inspect.signature(P.H3Pipeline._encode_references).parameters
    prr = inspect.signature(P.H3Pipeline._read_references).parameters
    pw = inspect.signature(write_mp4).parameters
    pvm = inspect.signature(DM.video_mask).parameters
    fields = {f.name for f in dataclasses.fields(P.H3GenerateRequest)}
    need = {"prompt", "seed", "width", "height", "num_frames", "output_path", "num_inference_steps", "shift_video",
            "shift_audio", "images", "keep_resident", "attention_backend", "block_swap_prefetch", "vram16_mode",
            "references"}
    src_den = inspect.getsource(P.H3Pipeline._denoise)
    src_enc = inspect.getsource(P.H3Pipeline._encode)
    src_dec = inspect.getsource(P.H3Pipeline._decode)
    src_gen = inspect.getsource(P.H3Pipeline.generate)
    ok_d = (list(pd)[:6] == ["self", "req", "text_hidden", "text_tags", "phases", "entry"]
            and all(k in pd for k in ("keep", "video_fix", "audio_fix", "cond_clean", "cond_frame_idx", "refs",
                                      "video_inject_clean"))
            and list(pe)[:4] == ["self", "prompt", "phases", "images"] and pe["has_cond"].kind == inspect.Parameter.KEYWORD_ONLY
            and "refs" in pe and pe["refs"].kind == inspect.Parameter.KEYWORD_ONLY
            and list(per_) == ["self", "media", "entry", "index", "total", "video_enc", "te_refs"]
            and list(prr) == ["self", "req"]
            and isinstance(inspect.getattr_static(P.H3Pipeline, "_te_references"), staticmethod)
            and callable(getattr(P.H3Pipeline, "_build_video_encoder", None))
            and callable(getattr(P.H3Pipeline, "generate", None))
            and list(inspect.signature(P.H3Pipeline.generate).parameters) == ["self", "req"]
            and [f.name for f in dataclasses.fields(P.H3ReferenceInput)] == ["kind", "path", "slot", "max_sec"]
            and list(pdec)[:6] == ["self", "req", "video_lat", "audio_lat", "entry", "keep"]
            and list(inspect.signature(P._Phases.__init__).parameters) == ["self", "cuda"] and hasattr(P._Phases, "phase")
            and need <= fields and "inject_clean" in pvm
            and P.encode_steps(has_cond=False, has_audio=False, has_refs=True) == 4
            and list(pw)[:2] == ["path", "frames"] and all(k in pw for k in ("fps", "audio", "sample_rate"))
            and DM.VIDEO_INJECT_CLEAN == 0.999
            and "inject_clean=video_inject_clean" in src_den and 'entry["sequence"]["ref_video"] = ref_video_n' in src_den
            and 'entry["mask"]["video"]["inject_clean"]' in src_den and "return video, audio, prefetch_used" in src_den
            and 'phases.data["encode"]["te_read"]' in src_enc and 'entry["vae_source"]' in src_dec
            and '"h3": h3' in src_gen and '"phases": phases.data' in src_gen
            and 'phases.phase("reference_encode")' in src_gen
            and callable(P._free_cuda) and hasattr(P.H3Pipeline, "release_resident"))
    _line(results, ok_d, "d product contract: _denoise (refs, video_inject_clean, sequence.ref_video, "
                         "mask.video.inject_clean), _encode (refs keyword), _encode_references (7 parameters), "
                         "_read_references / _te_references (static) / _build_video_encoder / generate(req), "
                         "H3GenerateRequest.references, H3ReferenceInput (kind, path, slot, max_sec), _decode's "
                         "vae_source, generate's h3.phases, encode_steps(refs) == 4")

    # (e) seam PSNR and the latent ratio
    Fp, H, W = 30, 8, 8
    fr = torch.zeros((Fp, H, W, 3), dtype=torch.uint8)
    for t in range(Fp):
        fr[t] = t % 2 * 2            # Y alternates by 2 -> PSNR = 10 log10(255^2 / 4)
    fr[16:] += 50                    # +50 at t = 16 except it also shifts the parity pattern
    st_ = frame_stats_torch(fr)
    exp2 = 10.0 * math.log10(255.0 ** 2 / 4.0)
    same_fr = torch.full((4, H, W, 3), 77, dtype=torch.uint8)
    st_same = frame_stats_torch(same_fr)
    # the planted seam at t = 16: |dY| = 50 - 2 = 48 (frame 15 is odd = 2, frame 16 = 0 + 50)
    exp_j = 10.0 * math.log10(255.0 ** 2 / 48.0 ** 2)
    ps = psnr_seam(st_["psnr"], 16)
    # the exclusion: values at J-1 and J+1 must not enter the median
    planted = [10.0] * 40
    planted[19 - 1] = 0.0            # t = 19 (J - 1)
    planted[21 - 1] = 0.0            # t = 21 (J + 1)
    planted[20 - 1] = 5.0            # t = 20 (J)
    ex = psnr_seam(planted, 20)
    edge = psnr_seam([30.0, 20.0, 30.0], 2)          # window clipped to 1 .. 3, and all of it excluded -> None
    edge2 = psnr_seam([30.0] * 5 + [1.0], 6)          # the last t: window [1, 4] only
    lat = torch.zeros((1, 24, 20, 2, 2))
    for t in range(20):
        lat[:, :, t] = 0.1 * t
    lat[:, :, 9:] += 1.0                               # jump at t = 9
    dl = latent_d(lat)
    rl = AC.jump_ratio(dl, 9)
    ok_e = (all(v == PSNR_CAP for v in st_same["psnr"]) and len(st_["psnr"]) == Fp - 1
            and abs(st_["psnr"][0] - exp2) < 1e-9 and abs(ps["psnr_J"] - exp_j) < 1e-9
            and abs(ps["psnr_window_median"] - exp2) < 1e-9 and abs(ps["psnr_drop"] - (exp2 - exp_j)) < 1e-9
            and ex["psnr_window_median"] == 10.0 and ex["psnr_drop"] == 5.0
            and edge["psnr_J"] == 20.0 and edge["psnr_window_median"] is None and edge["psnr_drop"] is None
            and edge2["psnr_window_median"] == 30.0 and psnr_seam([1.0], 5)["psnr_J"] is None
            and len(dl) == 19 and abs(dl[8] - 1.1) < 1e-6 and abs(dl[0] - 0.1) < 1e-6
            and rl is not None and abs(rl - 11.0) < 1e-4
            and json.dumps(seam_metrics(same_fr, {"S": [1], "J": [2], "J_lat": [1]}, lat[:, :, :2]), allow_nan=False)
            and _json_safe({"a": [float("inf"), 1.0], "b": float("nan")}, nb := []) == {"a": [None, 1.0], "b": None}
            and len(nb) == 2)
    _line(results, ok_e, f"e seam PSNR: identical frames -> {PSNR_CAP}; dY 2 -> {exp2:.3f} dB; planted seam "
                         f"psnr_J {ps['psnr_J']:.3f}, median {ps['psnr_window_median']:.3f}, drop {ps['psnr_drop']:.3f}; "
                         f"J-1 / J+1 excluded; window clipped at both ends (None when empty); latent ratio at the "
                         f"planted jump {rl:.3f} (= 1.1 / 0.1); no inf in the JSON")

    # 12. a short synthetic mp4 through analyze_refchain, seam_metrics == its numbers, and its CSV driver
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
    geom_s = {"S": [Js - 5], "J": [Js], "J_lat": [9], "O": 5, "N": 2}
    lat_s = (0.1 * torch.arange(14, dtype=torch.float32)).view(1, 1, 14, 1, 1).repeat(1, 24, 1, 1, 1)
    lat_s[:, :, 9:] += 1.0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mp4 = tmp / "synthetic.mp4"
        write_mp4(str(mp4), frames, fps=FRAME_RATE, audio=wave, sample_rate=AUDIO_SAMPLE_RATE)
        st_mp4 = AC.read_mp4_video_stats(mp4)
        aw, rate = AC.read_mp4_audio(mp4)
        sm = seam_metrics(frames, geom_s, lat_s)
        fake_rep = {"name": "c9_synthetic", "geometry": geom_s, "ok": True,
                    "condition": {"weights": {"transformer": {"path": "x/ref2va.safetensors"}}},
                    "clips": [{"references": []}, {"references": [{"kind": "picture"}]}],
                    "decode": {"once": {"path": str(mp4), "seam_metrics": sm}}}
        rp = tmp / "c9_synthetic.json"
        rp.write_text(json.dumps(fake_rep, allow_nan=False), encoding="utf-8")
        (tmp / "g_timeout.json").write_text('{"timeout": true}', encoding="utf-8")
        an_code = AC.analyze(AC.default_reports(tmp), tmp, tmp / "analysis.json")
        an = json.loads((tmp / "analysis.json").read_text(encoding="utf-8"))
        import csv as _csv

        with (tmp / "analysis_summary.csv").open(encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        curve_ok = (tmp / "analysis_curve_c9_synthetic.csv").is_file() and (tmp / "analysis_audio_c9_synthetic.csv").is_file()
    st_t = frame_stats_torch(frames)
    st_n = AC.stats_from_rgb(frames.numpy())
    rel = max(max(abs(a - b) / max(abs(b), 1e-9) for a, b in zip(st_t[key], st_n[key]))
              for key in ("y_mean", "y_sq_mean", "chroma_mean", "d"))
    sn = AC.summarize(st_n, geom_s)
    d = st_mp4["d"]
    argmax_t = int(np.argmax(d)) + 1
    ratio = AC.jump_ratio(d, Js)
    want_samples = _round_frac(Fs * SAMPLES_PER_FRAME)
    _line(results, st_mp4["frames"] == Fs and rate == AUDIO_SAMPLE_RATE and aw.shape[0] == 2
          and abs(aw.shape[1] - want_samples) <= 2048 and argmax_t == Js and ratio is not None and ratio > 5,
          f"12 analyze_refchain reads the synthetic mp4: {st_mp4['frames']} frames, {aw.shape[1]} samples "
          f"(~{want_samples}), the planted jump found at t={argmax_t}")
    same12 = (rel < 1e-9 and sm["summary"]["seams"][0]["ratio_J"] is not None
              and abs(sm["summary"]["seams"][0]["ratio_J"] - sn["seams"][0]["ratio_J"]) < 1e-6
              and abs(sm["summary"]["regions"][1]["delta_y_mean"] - sn["regions"][1]["delta_y_mean"]) < 1e-6)
    _line(results, same12, f"12 seam_metrics (torch) == analyze_refchain.stats_from_rgb (numpy) on the same frames "
                           f"(max rel {rel:.1e}; jump ratio {sn['seams'][0]['ratio_J']:.2f})")
    seam_row = rows[0] if rows else {}
    _line(results, an_code == 0 and an.get("ok") and an.get("summary_rows") == an.get("expected_rows") == 1
          and len(rows) == 1 and curve_ok and seam_row.get("name") == "c9_synthetic"
          and seam_row.get("refs_clip") == "1" and seam_row.get("transformer") == "ref2va.safetensors"
          and abs(float(seam_row["ratio_lat_J"]) - float(AC._fmt(sm["summary"]["seams"][0]["ratio_lat_J"]))) < 1e-9
          and abs(float(seam_row["psnr_drop"]) - float(AC._fmt(sm["summary"]["seams"][0]["psnr_drop"]))) < 1e-9
          and an.get("skipped") == ["g_timeout.json"],
          f"12 analyze_refchain.analyze: c*_*.json / g_*.json read (a timeout report skipped), "
          f"{len(rows)} row = conditions x seams; ratio_lat_J / psnr_drop copied from seam_metrics; curve and audio CSVs")
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
    ap.add_argument("--transformer", help="the DiT weights (FL2VA / Ref2VA), repo-relative or absolute (required)")
    ap.add_argument("--refs", help="JSON: one list of {kind, slot, path} per clip (plan §5 (2))")
    ap.add_argument("--clip-frames", type=int, default=124)
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--overlap", type=int, default=None, help="17m+5 frames (default 22)")
    ap.add_argument("--prompts", help="JSON list of prompts (a shorter list repeats its last entry)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--shift-video", type=float, default=12.0)
    ap.add_argument("--shift-audio", type=float, default=3.0)
    ap.add_argument("--attention", choices=("sdpa", "sage"), default="sage")
    ap.add_argument("--vram16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-latents", action="store_true")
    ap.add_argument("--reuse-clip0", metavar="SRC_JSON", help="a report whose saved clip 0 is reused")
    ap.add_argument("--fingerprint", action="store_true", help="G-F: clip 0's _decode vs the FL2VA fingerprint")
    ap.add_argument("--recheck-clip0", action="store_true")
    ap.add_argument("--product-recheck", action="store_true", help="G-R: clip 0's _decode vs the product's generate()")
    ap.add_argument("--out-dir", help=f"mp4 / .pt folder (default {OUT_ROOT}\\<name>)")
    ap.add_argument("--report")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck(args.report)
    if not args.name:
        ap.error("--name is required (or --selfcheck)")
    if not args.transformer:
        ap.error("--transformer is required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
