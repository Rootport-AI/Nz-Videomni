"""MiniMax H3 chain: several T2V clips joined by carrying latents over (台帳 §1-106).

A clip may carry its own Ref2VA references (台帳 §1-109): they go through the
single job's reference stage for that clip only; a clip without any runs as before.

The mechanism is Docs/MINIMAX_H3_DESIGN.md §3.3; the geometry is
``h3_chain_math`` at the repository root, shared with the app. Each clip's
fixed head is self-checked against the previous tail with ``torch.equal``.

``keep_resident`` means what it means for one job: kept after a successful
job that asks for it, released otherwise (and after any failure).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from engine_h3 import pipeline_h3 as P
from h3_chain_math import H3ChainLayout, compute_h3_chain_layout

#: The video fixed rows' injection coefficient (裁定 84-2: the latents themselves).
CHAIN_INJECT_CLEAN = 1.0


@dataclass(frozen=True)
class H3ChainClipSpec:
    """One clip: its whole prompt, its frame count and its own Ref2VA references
    (台帳 §1-109; empty = none)."""

    prompt: str
    num_frames: int
    references: tuple[P.H3ReferenceInput, ...] = ()


@dataclass
class H3ChainSpec:
    """``base``: the settings every clip shares; its ``prompt`` / ``num_frames``
    / ``seed`` are replaced per clip and its ``output_path`` is the chain's mp4.
    T2V only: ``base`` carries no images, source audio or references (a clip's
    references are its own, ``H3ChainClipSpec.references``)."""

    base: P.H3GenerateRequest
    clips: tuple[H3ChainClipSpec, ...]
    seed: int


def _sha_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def latents_sha256(t: torch.Tensor) -> str:
    """The latents' fingerprint: CPU float32 contiguous bytes (c_chain.py's ``_sha_tensor``)."""
    return hashlib.sha256(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()).hexdigest()


def carry_fix(prev_video: torch.Tensor, prev_audio: torch.Tensor, t_lat: int, a_lat: int, k: int, ka: int):
    """``(video_fix, audio_fix)`` for a clip of ``t_lat`` video / ``a_lat`` audio latents.

    The head is ``prev_video[:, :, -k:]`` / ``prev_audio[:, :, -ka:]`` at mask
    value 0, the rest generated (value 1; its clean part is zero and unused).
    Where the tail comes from is the caller's: the previous clip's video
    latents and the concatenated audio here (裁定 84-5 keeps the source
    replaceable). CPU float32.
    """
    shape_v = tuple(prev_video.shape[:2]) + (int(t_lat),) + tuple(prev_video.shape[3:])
    shape_a = tuple(prev_audio.shape[:2]) + (int(a_lat),)
    clean_v = torch.zeros(shape_v, dtype=torch.float32)
    clean_a = torch.zeros(shape_a, dtype=torch.float32)
    if k > 0:
        clean_v[:, :, :k] = prev_video[:, :, -k:].to(torch.float32)
    if ka > 0:
        clean_a[:, :, :ka] = prev_audio[:, :, -ka:].to(torch.float32)
    per_v = [0.0] * k + [1.0] * (int(t_lat) - k)
    per_a = [0.0] * ka + [1.0] * (int(a_lat) - ka)
    return (clean_v, per_v), (clean_a, per_a)


def clip_requests(spec: H3ChainSpec) -> list[P.H3GenerateRequest]:
    """Clip i's request: ``base`` with its prompt, frames, references, ``seed + i`` and ``keep_resident=True``."""
    return [
        dataclasses.replace(
            spec.base, prompt=clip.prompt, num_frames=int(clip.num_frames), seed=int(spec.seed) + i,
            keep_resident=True, references=tuple(clip.references),
        )
        for i, clip in enumerate(spec.clips)
    ]


def _check_t2v(base: P.H3GenerateRequest) -> None:
    if base.images or base.source_audio is not None or base.references:
        raise ValueError("連結は T2V だけです（共通設定（base）にはキーフレーム・音声・参照は付けられません。参照はクリップごとに付けます）")


def run_chain_h3(
    pipe: P.H3Pipeline,
    spec: H3ChainSpec,
    progress: Callable[..., None],
    log: Callable[[str], None],
) -> dict:
    """Run the chain on a loaded ``pipe`` and write ``spec.base.output_path``.

    ``progress(stage, index, total, *, outer_index=None, outer_total=None)``:
    the clip's stages carry ``outer_index`` = the clip (0-based) and
    ``outer_total`` = the clip count; the decode's progress goes through the
    pipeline's original ``_progress``, with no outer position. Returns the
    single job's ``done`` fields (``num_frames`` = the timeline's) plus ``chain``.

    A clip with references runs the single job's reference steps (``generate``
    without key frames): read before its text encoder, encoded after it in a
    ``reference_encode`` phase by a video encoder built for this clip only, and
    handed to its denoise next to the fixed head.
    """
    if pipe._dit_cfg is None:
        raise RuntimeError("generate before load")
    base = spec.base
    keep = bool(base.keep_resident)
    started = time.perf_counter()
    cuda = pipe.device.type == "cuda"
    ok = False
    lora_report: list[dict] = []
    clip_phases: list[P._Phases] = []
    clip_records: list[dict] = []
    decode_phases = P._Phases(cuda=cuda)
    prefetch_used = "off"
    size_bytes = 0
    layout: H3ChainLayout | None = None
    try:
        _check_t2v(base)
        layout = compute_h3_chain_layout([c.num_frames for c in spec.clips])
        requests = clip_requests(spec)
        for req in requests:
            P.validate_request(req)
        n = len(requests)
        k = layout.overlap_latents
        video_cat = audio_cat = prev_video = None
        for i, req in enumerate(requests):
            ka = layout.audio_overlap_latents[i - 1] if i else 0
            P._free_cuda()
            phases = P._Phases(cuda=cuda)
            clip_phases.append(phases)
            video_fix = audio_fix = None
            if i:
                video_fix, audio_fix = carry_fix(
                    prev_video, audio_cat, layout.clip_latent_frames[i], layout.clip_audio_latents[i], k, ka,
                )
            log(f"chain: clip {i + 1}/{n} seed={req.seed} frames={req.num_frames} "
                + (f"fixed head k={k} ka={ka}" if i else "no fixed head"))
            original_progress = pipe._progress
            pipe._progress = functools.partial(progress, outer_index=i, outer_total=n)
            try:
                with torch.no_grad():
                    # Ref2VA (台帳 §1-109): the clip's references are read on the CPU
                    # before its text encoder, as in a single job; none = as before.
                    media = pipe._read_references(req)
                    encode_total = P.encode_steps(has_cond=False, has_audio=False, has_refs=bool(media))
                    with phases.phase("encode") as entry:
                        hidden, tags = pipe._encode(
                            req.prompt, phases, images=(), has_cond=False, refs=pipe._te_references(media),
                        )
                        entry["text_tokens"] = int(hidden.shape[0])
                    ref_rows = None
                    ref_counts = None
                    if media:
                        with phases.phase("reference_encode") as entry:
                            # This clip's own video encoder (only for a picture / video
                            # reference), dropped once its references are encoded.
                            enc = pipe._build_video_encoder() if any(m["kind"] != "audio" for m in media) else None
                            te_refs = phases.data["encode"].get("te_read", {}).get("references", [])
                            ref_rows = pipe._encode_references(media, entry, encode_total, encode_total, enc, te_refs)
                            enc = None
                            P._free_cuda()
                        media = None   # encoded: drop the read media before the denoise, as a single job
                        counts = {kind: sum(1 for s, _r in ref_rows if s.kind == kind) for kind in P.REFERENCE_KINDS}
                        ref_counts = {"pictures": counts["picture"], "videos": counts["video"], "audios": counts["audio"]}
                    with phases.phase("denoise") as entry:
                        video, audio, prefetch_used = pipe._denoise(
                            req, hidden, tags, phases, entry, keep=True,
                            lora_report=lora_report if i == 0 else None,
                            video_fix=video_fix, audio_fix=audio_fix, refs=ref_rows,
                            video_inject_clean=CHAIN_INJECT_CLEAN,
                        )
                    video, audio = video.cpu(), audio.cpu()
            finally:
                pipe._progress = original_progress
            del hidden, tags, video_fix, audio_fix, media, ref_rows
            P._free_cuda()
            if int(video.shape[2]) != layout.clip_latent_frames[i] or int(audio.shape[2]) != layout.clip_audio_latents[i]:
                raise RuntimeError(
                    f"chain: clip {i + 1} latents {list(video.shape)} / {list(audio.shape)}, expected "
                    f"{layout.clip_latent_frames[i]} video / {layout.clip_audio_latents[i]} audio latents"
                )
            head_equal = None
            if i:
                # Self-check: the fixed head must come back as the previous tail bit for bit
                # (denoise_mask.finish puts the m = 0 rows back to their clean latents).
                head_equal = bool(torch.equal(video[:, :, :k], prev_video[:, :, -k:])) and (
                    ka == 0 or bool(torch.equal(audio[:, :, :ka], audio_cat[:, :, -ka:]))
                )
                if not head_equal:
                    raise RuntimeError(f"chain: clip {i + 1}'s fixed head is not the previous clip's tail")
            seconds = {"encode": phases.data.get("encode", {}).get("seconds")}
            if ref_counts is not None:
                seconds["reference_encode"] = phases.data.get("reference_encode", {}).get("seconds")
            seconds["denoise"] = phases.data.get("denoise", {}).get("seconds")
            record = {
                "prompt_sha256": _sha_text(req.prompt),
                "seconds": seconds,
                "latents_sha256": {"video": latents_sha256(video), "audio": latents_sha256(audio)},
                "head_equals_prev_tail": head_equal,
            }
            if ref_counts is not None:
                record["references"] = ref_counts   # only for a clip with references
            clip_records.append(record)
            if i == 0:
                video_cat, audio_cat = video, audio
            else:
                video_cat = torch.cat([video_cat, video[:, :, k:]], dim=2)
                audio_cat = torch.cat([audio_cat, audio[:, :, ka:]], dim=2)
            prev_video = video
            del video, audio
        if int(video_cat.shape[2]) != layout.total_latent_frames or int(audio_cat.shape[2]) != layout.total_audio_latents:
            raise RuntimeError(
                f"chain: concatenated latents {list(video_cat.shape)} / {list(audio_cat.shape)}, expected "
                f"{layout.total_latent_frames} video / {layout.total_audio_latents} audio latents"
            )
        del prev_video
        if not keep:
            # As a single job without the key: its denoise drops the DiT before
            # the decode; every clip here kept it (keep=True), so let it go now.
            pipe.release_resident()
        P._free_cuda()
        out_req = dataclasses.replace(base, num_frames=layout.total_frames, output_path=base.output_path)
        with torch.no_grad():
            with decode_phases.phase("decode") as entry:
                # The decoders compute on the latents' device: hand them over on the pipeline's.
                size_bytes = pipe._decode(
                    out_req, video_cat.to(pipe.device), audio_cat.to(pipe.device), entry, keep=keep,
                )
        del video_cat, audio_cat
        ok = True
    finally:
        if not (ok and keep):
            pipe.release_resident()
        P._free_cuda()
    mem = P._mem_gib()
    all_phases = clip_phases + [decode_phases]
    h3 = {
        "phases": {"clips": [p.data for p in clip_phases], "decode": decode_phases.data.get("decode", {})},
        "quant_backends": P._backend_summary(pipe.quant_backends),
    }
    if base.loras:
        h3["loras"] = lora_report   # clip 0's: every clip attaches the same files
    return {
        "seed_used": int(spec.seed),
        "sampler": pipe.sampler,
        "num_inference_steps": int(base.num_inference_steps),
        "shift_video": float(base.shift_video),
        "shift_audio": float(base.shift_audio),
        "peak_vram_mb": max(p.peak_alloc_mib for p in all_phases) if cuda else None,
        "peak_vram_reserved_mb": max(p.peak_reserved_mib for p in all_phases) if cuda else None,
        "rss_peak_gib": mem.get("rss_peak"),
        "commit_gib": mem.get("commit"),
        "commit_peak_gib": mem.get("commit_peak"),
        "seconds": round(time.perf_counter() - started, 2),
        "num_frames": int(layout.total_frames),
        "frame_rate": int(base.frame_rate),
        "width": int(base.width),
        "height": int(base.height),
        "size_bytes": int(size_bytes),
        "attention_used": pipe._attention_for(base),
        "block_swap_prefetch_used": prefetch_used,
        "keep_resident_used": "on" if keep else "off",
        "vram16_mode_used": "on" if base.vram16_mode else "off",
        "images_used": 0,
        "h3": h3,
        "chain": {
            **layout.to_dict(),
            "seeds": [int(spec.seed) + i for i in range(len(spec.clips))],
            "clips": clip_records,
            "h3": {"overlap_frames": layout.overlap_frames, "inject_clean": CHAIN_INJECT_CLEAN, "decode": "once"},
        },
    }
