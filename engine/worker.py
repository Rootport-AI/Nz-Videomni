"""Persistent LTX-2.3 generation worker (Phase 5, Approach W).

Runs inside the engine venv (.venv-engine), which is the only environment that
has torch + ltx_core/ltx_pipelines@00dc53d + gguf. The
app process (FastAPI, its own torch-less .venv) spawns ONE of these per loaded
model and talks to it over a tiny JSON-lines protocol on stdin/stdout. The mp4 is
written by the engine directly to a shared-disk path; only small control JSON
crosses the pipe.

Bootstrap mirrors outputs/phase4_gguf_gemma/run_t2v_bs8.py VERBATIM (proven to
run on this Windows + 16GB box):
  * TORCH_COMPILE_DISABLE=1 set before importing torch.
  * chdir(ROOT) + sys.path.insert(0, ROOT) so the first-party `engine.*` package
    (and the venv-installed `ltx_core` / `ltx_pipelines`) import.
  * `import ltx_core.loader` BEFORE the pipeline import (rev-00dc53d circular
    import gotcha — required).

Protocol (one JSON object per line; parent -> worker):
  {"op": "load", checkpoint_path, gemma_root, upsampler_path,
   gguf_transformer_path, gguf_gemma_path, gguf_per_layer_quant,
   block_swap_blocks_on_gpu, vae_spatial_tile_size, vae_temporal_tile_size}
  {"op": "generate", prompt, seed, height, width, num_frames, frame_rate,
   num_steps, images:[{path,frame_idx,strength}...], output_path,
   # Phase B/C IC-LoRA (forward-time weight patch); loras always present (may be []),
   # reference_video null unless a reference is supplied. preprocess (Phase C):
   # "none" -> raw reference used as-is (Phase B); "canny"/... -> converted to a
   # control-signal video via engine/preprocess/, path swapped to it:
   loras:[{path,strength}...], reference_video:{path,strength,preprocess}|null}
  # Phase 3 WP4 — masked AV-latent clip chaining (ONE decode, always-tiled stage2):
  {"op": "generate_chain", width, height, frame_rate, num_steps, seed,
   overlap_frames, overlap_strength, output_path,
   # chunked_upsample (optional, default False): when true the whole-timeline
   # spatial upsample runs in halo-padded temporal chunks (VRAM-bounded for long
   # 768p chains). Omitted/false -> the one-shot upsample path is byte-identical.
   chunked_upsample,
   clips:[{prompt, num_frames, images:[{path,frame_idx,strength}...]}...],
   # V2V continuation (optional; null unless continuing an uploaded video). When
   # present, clips may be length 1; clips[0].num_frames is the TOTAL clip-0
   # timeline (frozen context head + new tail). ``path`` is ALREADY the source
   # tail cut at the request fps (the app guarantees this — the engine does not
   # resample). The source tail is VAE-encoded (tiled) and frozen as clip-0's
   # head; the delivered mp4 is the NEW part only (context trimmed off the front):
   source:{path, context_frames}|null}
  {"op": "shutdown"}

Replies are framed with a unique prefix so library/tqdm stdout noise can be
ignored by the parent. Every protocol line: @@LTX@@<compact-json>, flushed. All
other logging goes to STDERR.
  @@LTX@@{"event":"ready"}
  @@LTX@@{"event":"done","seed_used":...,"peak_vram_mb":...}
  @@LTX@@{"event":"error","detail":...}

The generate_chain ``done`` event carries a ``chain`` dict (full junction
geometry). For a V2V run it additionally holds a ``chain.v2v`` sub-dict:
{context_frames, n_ctx_v, n_ctx_a, freeze_ka, trimmed_px, trimmed_audio_samples,
audio_fade_in_samples, source_had_audio, audio_head_frozen, new_frames_px,
decoded_frames_px, v2v_context_junction_px, source_context_px}.
``source_had_audio`` = the source FILE had an audio stream; ``audio_head_frozen``
(= freeze_ka > 0) = the frozen head actually carries audio continuity — these
can differ if the source's encoded audio latents ran out before n_ctx_a (a
warning is logged to STDERR in that case, freeze_ka is silently reduced).
"""

import os
import sys
import json
import gc
import logging
import traceback
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# import cwd = project root: make the first-party `engine.*` package (and the
# venv-installed `ltx_core`/`ltx_pipelines`) importable. Launched as
# `python -m engine.worker` with PYTHONPATH=<root>, but we also insert the root
# on sys.path + chdir(ROOT) here so a bare `python engine/worker.py` still works
# (mirrors the pre-relocation chdir/sys.path bootstrap).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Unique frame prefix for every protocol line the parent parses.
PREFIX = "@@LTX@@"


def _configure_worker_logging() -> None:
    """Route the engine's ``logging.getLogger(__name__)`` calls to STDERR.

    The engine modules (engine/gguf, engine/gemma, engine/transformer,
    engine/pipeline/*) log via module loggers that, until now, had no handler
    anywhere in the worker process -> everything below WARNING was silently
    dropped, so the GGUF/block-swap/model-load breakdown never reached
    logs/ltx_worker.log. We add ONE StreamHandler on the root logger pointed at
    STDERR (which the parent redirects to logs/ltx_worker.log — the peak_vram
    primary source) so those INFO lines become visible in the same file, framed
    with the familiar ``[ltx_worker]`` prefix + the emitting module name.

    This never touches STDOUT (the ``@@LTX@@`` protocol channel), and it does
    NOT duplicate the ``_log()`` print path: ``_log`` writes to STDERR directly
    (not through logging), so each diagnostic line is emitted exactly once. The
    handler is tagged + added idempotently so re-entry can't double it.
    """
    root = logging.getLogger()
    if not any(getattr(h, "_ltx_worker_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[ltx_worker] %(name)s: %(message)s"))
        handler._ltx_worker_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(logging.INFO)


_configure_worker_logging()


def _log(msg: str) -> None:
    """Human/diagnostic logging -> STDERR only (never the protocol channel)."""
    print(f"[ltx_worker] {msg}", file=sys.stderr, flush=True)


def _emit(event: str, **fields: object) -> None:
    """Write one framed protocol line to STDOUT and flush."""
    payload = {"event": event, **fields}
    print(PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def _detail(exc: BaseException) -> str:
    """repr + last ~2000 chars of traceback for an error reply."""
    tb = "".join(traceback.format_exc())
    return f"{exc!r}\n{tb[-2000:]}"


import torch  # noqa: E402

# Pre-warm to avoid the rev-00dc53d cold-import circular bug (MUST precede any
# quantization / pipeline import). Mirrors run_t2v_bs8.py L52-54.
import ltx_core.loader  # noqa: E402,F401

_log(f"pre-warmed ltx_core.loader; torch {torch.__version__} cuda={torch.cuda.is_available()}")

DEV = torch.device("cuda:0")
# init cuda context + warm + reset stats (mirrors run_t2v_bs8.py L57-61).
torch.cuda.set_device(DEV)
torch.cuda.init()
_ = torch.zeros(1, device=DEV)
torch.cuda.synchronize(DEV)
torch.cuda.reset_peak_memory_stats(DEV)

# Engine imports AFTER ltx_core.loader pre-warm + CUDA init.
from engine.pipeline.fast_video_pipeline import (  # noqa: E402
    LTXFastVideoPipeline,
)
from engine.api_types import ImageConditioningInput  # noqa: E402

_log("imported LTXFastVideoPipeline + ImageConditioningInput")

# Phase 5(B) VRAM fix: release the Windows-stranded caching-allocator reserved
# pool before each denoise. expandable_segments is a no-op on this box, so after
# block-swap loads the full GGUF transformer to GPU then evicts blocks to CPU,
# ~16GB of freed segments stay reserved and spill to WDDM shared during denoise.
# A gc+empty_cache right before denoise drops reserved ~17.5GB -> ~1.5GB (live
# unchanged), keeping denoise inside the 16GB card. Proven in outputs/phase5b_diag
# (tag EC): denoise shared 3969 -> 742MB (ambient), wall 210.9 -> 114.9s, output
# bit-identical.
import ltx_pipelines.distilled as _distilled  # noqa: E402

_orig_denoise_audio_video = _distilled.denoise_audio_video


def _denoise_with_cache_release(*args, **kwargs):
    gc.collect()
    torch.cuda.empty_cache()
    return _orig_denoise_audio_video(*args, **kwargs)


_distilled.denoise_audio_video = _denoise_with_cache_release
_log("installed pre-denoise empty_cache monkeypatch (Phase 5B VRAM fix)")

# F2 (G3 gate): per-step denoise progress. The wheel's denoising loops have no
# callback, but they all iterate via the samplers module's ``tqdm`` binding —
# engine/progress_shim.py swaps that for an observation-only wrapper that calls
# the emitter below after every completed step. STRICTLY additive: the shim
# yields the wheel's items unchanged (numbers/seeds/tensors untouched) and any
# emit failure is logged and swallowed so a progress hiccup can never fail a
# generation.
from engine import progress_shim  # noqa: E402


def _emit_step_progress(
    stage: str,
    index: int,
    total: int,
    it_s: float | None = None,
    outer_index: int | None = None,
    outer_total: int | None = None,
) -> None:
    fields: dict[str, object] = {
        "stage": str(stage),
        "index": int(index),
        "total": int(total),
    }
    if it_s is not None:
        fields["it_s"] = float(it_s)
    if outer_index is not None:
        fields["outer_index"] = int(outer_index)
    if outer_total is not None:
        fields["outer_total"] = int(outer_total)
    try:
        _emit("progress", **fields)
    except Exception as exc:  # noqa: BLE001 - observation must never kill a job
        _log(f"step-progress emit failed (ignored): {exc!r}")


progress_shim.install(_emit_step_progress)
_log("installed per-step tqdm progress shim (F2)")

# Module-global pipeline, built once on the first {"op":"load"}.
_PIPE: LTXFastVideoPipeline | None = None


def _do_load(msg: dict) -> None:
    """Build the pipeline ONCE. Mirrors run_t2v_bs8.py create() arg names/values."""
    global _PIPE
    if _PIPE is not None:
        _emit("ready")
        return

    _log("creating pipeline (GGUF transformer + GGUF Gemma)...")
    _PIPE = LTXFastVideoPipeline.create(
        checkpoint_path=msg["checkpoint_path"],
        gemma_root=msg["gemma_root"],
        upsampler_path=msg["upsampler_path"],
        device=DEV,
        block_swap_blocks_on_gpu=int(msg["block_swap_blocks_on_gpu"]),
        gguf_transformer_path=msg["gguf_transformer_path"],
        gguf_gemma_path=msg["gguf_gemma_path"],
        gguf_per_layer_quant=bool(msg["gguf_per_layer_quant"]),
        vae_spatial_tile_size=int(msg["vae_spatial_tile_size"]),
        vae_temporal_tile_size=int(msg["vae_temporal_tile_size"]),
        # Stage 3: load-once / keep-resident weights (StateDictRegistry). Default ON in
        # the resident worker; the reuse_loop spike toggles LTX_KEEP_RESIDENT=0 for an
        # A/B baseline against the per-job-rebuild known-good.
        keep_resident_weights=(os.environ.get("LTX_KEEP_RESIDENT", "1") == "1"),
        # TE per-layer CPU offload: stream the GGUF-quantized Gemma decoder layers
        # CPU->GPU one window at a time during text-encode (caps the ~15 GB encode
        # peak to a few GB). Default ON when the env var is ABSENT; LTX_TE_OFFLOAD=0
        # reproduces today's all-layers-GPU-resident behavior.
        te_offload_text_encoder=(os.environ.get("LTX_TE_OFFLOAD", "1") == "1"),
        # DiT CPU-resident build: build the transformer on CPU and move only the
        # non-block submodules to GPU, eliminating the ~16.9 GB load-time GPU
        # spike. Default ON when the env var is ABSENT; LTX_DIT_CPU_LOAD=0
        # reproduces today's build-on-GPU-then-evict behavior.
        dit_cpu_load=(os.environ.get("LTX_DIT_CPU_LOAD", "1") == "1"),
        # Phase 1: re-source VIDEO VAE + AUDIO VAE/vocoder from standalone files
        # (gate via LTX_COMPONENT_FILES, default OFF). Text projection path is
        # passed through but NOT wired (Phase 2).
        use_component_files=(os.environ.get("LTX_COMPONENT_FILES", "0") == "1"),
        component_video_vae_path=msg.get("component_video_vae_path", ""),
        component_audio_vae_path=msg.get("component_audio_vae_path", ""),
        component_text_projection_path=msg.get("component_text_projection_path", ""),
        # cpu_text_encode intentionally NOT set -> GGUF Gemma path wins.
    )
    _log("PIPELINE_CREATED_OK")
    _emit("ready")


def _resolve_ic_reference(
    ref: dict | None, output_path: str
) -> tuple[tuple[str, float] | None, float]:
    """Resolve a worker ``reference_video`` block -> (ic_reference, attn_strength).

    ``ref`` is the raw ``reference_video`` dict (or None when absent). Returns
    ``((ref_path, ref_strength), attention_strength)`` — or ``(None, 1.0)`` when
    no reference is supplied. ``attention_strength`` is the IC-LoRA
    control-adherence knob (conditioning_attention_strength, 0..1, default 1.0):
    at 1.0 no attention-strength wrapper is applied downstream (structurally
    byte-identical to before); < 1.0 relaxes how strongly the reference drives
    self-attention. For a Phase C control adapter (``preprocess`` != "none") the
    raw reference is converted to a control-signal video (edge map / skeleton)
    via engine/preprocess/ and the returned path is swapped to that control mp4
    (written next to ``output_path``); "none" leaves the raw video as-is and cv2
    is never imported. An unknown ``preprocess`` fails the job loud
    (``get_processor`` raises). Extracted verbatim from _do_generate so the
    single-generate and chain paths resolve the reference identically.
    """
    ic_reference = None
    attn_strength = 1.0
    if ref:
        ref_path = str(ref["path"])
        ref_strength = float(ref.get("strength", 1.0))
        attn_strength = float(ref.get("attention_strength", 1.0))
        preprocess = ref.get("preprocess", "none")
        if preprocess and preprocess != "none":
            import time

            from engine.preprocess import get_processor, preprocess_video

            processor = get_processor(preprocess)
            control_path = os.path.join(
                os.path.dirname(output_path), f"control_{preprocess}.mp4"
            )
            t0 = time.perf_counter()
            n_frames = preprocess_video(Path(ref_path), Path(control_path), processor)
            elapsed = time.perf_counter() - t0
            _log(
                f"PREPROCESS {preprocess} {ref_path} -> {control_path} "
                f"frames={n_frames} elapsed={elapsed:.2f}"
            )
            ref_path = control_path
        ic_reference = (ref_path, ref_strength)
    return ic_reference, attn_strength


def _do_generate(msg: dict) -> None:
    """Run one generation; mp4 is written by the engine to msg['output_path']."""
    assert _PIPE is not None, "generate before load"
    output_path = msg["output_path"]
    seed = int(msg["seed"])

    torch.cuda.reset_peak_memory_stats(DEV)

    images = [
        ImageConditioningInput(
            path=i["path"],
            frame_idx=int(i["frame_idx"]),
            strength=float(i["strength"]),
        )
        for i in msg.get("images", [])
    ]

    # Phase B IC-LoRA (forward-time weight patch on the per-layer-quant path).
    # ``ic_loras`` is always passed EXPLICITLY (even []): an explicit empty list is
    # the authoritative "no LoRA this job" -> clean detach, so a no-LoRA job after a
    # LoRA job is byte-identical to base (gate G3). ``ic_reference`` is the
    # Pixel-Spatial-Upscaler reference video (None when absent).
    ic_loras = [
        (str(lo["path"]), float(lo["strength"])) for lo in msg.get("loras", [])
    ]
    # ``reference_video`` -> (ic_reference, attn_strength). The Phase C control
    # preprocess (edge/pose) and the conditioning_attention_strength knob are
    # both resolved inside the shared helper (see _resolve_ic_reference); no
    # reference -> (None, 1.0), inert + byte-identical to before.
    ic_reference, attn_strength = _resolve_ic_reference(
        msg.get("reference_video"), output_path
    )

    _log(
        f"generating {msg['width']}x{msg['height']} / {msg['num_frames']} frames "
        f"/ {msg['num_steps']} steps seed={seed} images={len(images)} "
        f"ic_loras={len(ic_loras)} ic_reference={'yes' if ic_reference else 'no'}"
    )
    # F2: single-generate runs the wheel's two denoising loops back-to-back
    # inside __call__ (no seam to hook), so the shim infers stage1/stage2 from
    # the loop-invocation count within this op (see engine/progress_shim.py).
    progress_shim.begin_single_op()
    try:
        _PIPE.generate(
            prompt=msg["prompt"],
            seed=seed,
            height=int(msg["height"]),
            width=int(msg["width"]),
            num_frames=int(msg["num_frames"]),
            frame_rate=msg["frame_rate"],
            images=images,
            output_path=output_path,
            num_steps=int(msg["num_steps"]),
            ic_loras=ic_loras,
            ic_reference=ic_reference,
            ic_attention_strength=attn_strength,
        )
    finally:
        progress_shim.end_op()

    peak = torch.cuda.max_memory_allocated(DEV) // (1024 * 1024)

    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"engine produced no/empty output: {output_path}")

    _log(f"GENERATED_OK peak_vram_mb={peak} -> {output_path}")
    _emit("done", seed_used=seed, peak_vram_mb=int(peak))

    # Resident-reuse: free the just-finished job's transient allocations before
    # the next job. The block-swap transformer carries reference cycles
    # (swapped_forward closures capturing block lists) that plain refcounting
    # can't reclaim, so an explicit gc.collect() is required; empty_cache() then
    # returns the freed CUDA blocks to the driver. Pairs with the keep-latest
    # fix in BlockSwapService.install().
    gc.collect()
    torch.cuda.empty_cache()


def _do_generate_chain(msg: dict) -> None:
    """Run one masked AV-latent clip chain; ONE mp4 written to output_path.

    Emits ``progress`` events (stage1 per segment, tile per stage-2 tile, decode)
    and a terminal ``done`` with the peak VRAM + full junction metadata (segment
    seams AND tile seams) for the review harness.
    """
    assert _PIPE is not None, "generate_chain before load"
    from engine.pipeline.chain_pipeline import AudioSourceSpec, ChainClipSpec, SourceSpec

    output_path = msg["output_path"]
    seed = int(msg["seed"])
    torch.cuda.reset_peak_memory_stats(DEV)

    clips = [
        ChainClipSpec(
            prompt=c["prompt"],
            num_frames=int(c["num_frames"]),
            images=[
                ImageConditioningInput(
                    path=i["path"], frame_idx=int(i["frame_idx"]), strength=float(i["strength"])
                )
                for i in c.get("images", [])
            ],
        )
        for c in msg["clips"]
    ]

    # V2V continuation: optional source (an mp4 that is ALREADY the fps-correct
    # source tail). When present, a single clip is allowed.
    source = None
    src = msg.get("source")
    if src:
        try:
            source = SourceSpec(
                path=str(src["path"]),
                context_frames=int(src["context_frames"]),
            )
        except KeyError as exc:
            raise ValueError(
                "generate_chain: source requires path and context_frames "
                f"(missing key {exc})"
            ) from exc

    # A2V (audio-to-video): optional audio_source (a decodable audio/media file).
    # Mutually exclusive with source (also enforced in run_chain + the API layer).
    audio_source = None
    asrc = msg.get("audio_source")
    if asrc:
        try:
            audio_source = AudioSourceSpec(path=str(asrc["path"]))
        except KeyError as exc:
            raise ValueError(
                f"generate_chain: audio_source requires path (missing key {exc})"
            ) from exc

    # Style/character IC-LoRA (forward-time weight patch, applied across the whole
    # chain). Always parsed EXPLICITLY (even []): an explicit empty list is the
    # authoritative "no LoRA this chain" -> clean detach, clearing any stale LoRA
    # left on the resident pipeline by a prior single generate() (mirrors
    # _do_generate). α: control (reference) adapters are now accepted for
    # clips=1 chains — the reference_video block (present only when a reference
    # was supplied) is resolved below via the SAME helper as single generate()
    # and wired to run_chain's stage-1 clip-0 conditioning; without it the chain
    # is byte-identical to before.
    ic_loras = [
        (str(lo["path"]), float(lo["strength"])) for lo in msg.get("loras", [])
    ]
    ic_reference, ic_attn = _resolve_ic_reference(
        msg.get("reference_video"), output_path
    )

    # Opt-in memory-bounded spatial upsample (additive; default False keeps the
    # one-shot whole-timeline upsample byte-identical). When True the engine
    # upsamples the assembled stage-1 latent in halo-padded temporal chunks so a
    # long 768p chain fits in 16GB VRAM.
    chunked_upsample = bool(msg.get("chunked_upsample", False))

    _log(
        f"generate_chain {msg['width']}x{msg['height']} clips={len(clips)} "
        f"frames={[c.num_frames for c in clips]} seed={seed} "
        f"overlap={msg.get('overlap_frames')}/{msg.get('overlap_strength')} "
        f"source={'yes(ctx=' + str(source.context_frames) + ')' if source else 'no'} "
        f"audio_source={'yes' if audio_source else 'no'} "
        f"ic_loras={len(ic_loras)}"
    )

    def _progress(stage: str, index: int, total: int) -> None:
        _emit("progress", stage=stage, index=int(index), total=int(total))

    meta = _PIPE.generate_chain(
        clips=clips,
        width=int(msg["width"]),
        height=int(msg["height"]),
        frame_rate=msg["frame_rate"],
        num_steps=int(msg["num_steps"]),
        seed=seed,
        overlap_frames=int(msg["overlap_frames"]),
        overlap_strength=float(msg["overlap_strength"]),
        output_path=output_path,
        progress=_progress,
        source=source,
        audio_source=audio_source,
        ic_loras=ic_loras,
        ic_reference=ic_reference,
        ic_attention_strength=ic_attn,
        chunked_upsample=chunked_upsample,
    )

    peak = torch.cuda.max_memory_allocated(DEV) // (1024 * 1024)
    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"engine produced no/empty chain output: {output_path}")

    _log(f"CHAIN_OK peak_vram_mb={peak} -> {output_path}")
    _emit("done", seed_used=seed, peak_vram_mb=int(peak), chain=meta)

    gc.collect()
    torch.cuda.empty_cache()


def _shutdown() -> None:
    """Best-effort free + sync, then exit 0."""
    global _PIPE
    try:
        _PIPE = None
        if torch.cuda.is_available():
            torch.cuda.synchronize(DEV)
            torch.cuda.empty_cache()
    except Exception:
        pass
    sys.exit(0)


def main() -> None:
    # First, block for the load op (a fatal error here -> exit 1 so the parent
    # sees the worker die before 'ready').
    loaded = False
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            _log(f"ignoring non-JSON stdin line: {line[:120]!r}")
            continue

        op = msg.get("op")

        if not loaded:
            if op != "load":
                # Tolerate shutdown/EOF before load; ignore anything else.
                if op == "shutdown":
                    _shutdown()
                _log(f"ignoring op={op!r} before load")
                continue
            try:
                _do_load(msg)
                loaded = True
            except BaseException as exc:  # noqa: BLE001
                _log("LOAD_FAILED")
                _emit("error", detail=_detail(exc))
                sys.exit(1)
            continue

        # Serving loop (post-load).
        if op == "generate":
            try:
                _do_generate(msg)
            except BaseException as exc:  # noqa: BLE001 - keep serving on error
                _log("GENERATE_FAILED")
                _emit("error", detail=_detail(exc))
            continue
        if op == "generate_chain":
            try:
                _do_generate_chain(msg)
            except BaseException as exc:  # noqa: BLE001 - keep serving on error
                _log("GENERATE_CHAIN_FAILED")
                _emit("error", detail=_detail(exc))
            continue
        if op == "shutdown":
            _shutdown()
        _log(f"ignoring unknown op={op!r}")

    # EOF on stdin -> graceful shutdown.
    _shutdown()


if __name__ == "__main__":
    main()
