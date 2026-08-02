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
   # attention_backend (optional, default "sdpa"): "sdpa" | "sage". Present on
   # BOTH generate ops. Purely a speed knob — see _resolve_attention:
   attention_backend,
   # block_swap_prefetch (optional, default False): hides the block-swap
   # CPU<->GPU weight transfer behind computation on a dedicated CUDA stream.
   # Present on BOTH generate ops. Also purely a speed knob (unlike
   # attention_backend, output is bit-identical on/off) — see
   # _resolve_block_swap_prefetch:
   block_swap_prefetch,
   # keep_resident (optional, default False): keeps each submodel's CPU-side
   # state_dict resident BETWEEN jobs (wheel StateDictRegistry), cutting the
   # per-job preprocessing from 66-79s to 9-15s at the cost of ~20GB of main
   # memory. Present on BOTH generate ops. Output is bit-identical on/off.
   # A MISSING key means off AND is the explicit "free the cache" trigger —
   # see _resolve_keep_resident (which can also auto-downgrade it):
   keep_resident,
   # Phase B/C IC-LoRA (forward-time weight patch); loras always present (may be []),
   # reference_video null unless a reference is supplied. preprocess (Phase C):
   # "none" -> raw reference used as-is (Phase B); "canny"/... -> converted to a
   # control-signal video via engine/preprocess/, path swapped to it:
   # loras[].audio_strength is optional: absent -> the audio side follows
   # strength (byte-identical to before this feature existed); 0 -> skip the
   # audio-side weights entirely.
   loras:[{path,strength,audio_strength?}...], reference_video:{path,strength,preprocess}|null}
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
  @@LTX@@{"event":"ready","sage_available":true|false}
  @@LTX@@{"event":"done","seed_used":...,"peak_vram_mb":...,"attention_used":...,
          "block_swap_prefetch_used":...,"keep_resident_used":...,
          "peak_vram_reserved_mb":...}
  @@LTX@@{"event":"error","detail":...}

``ready.sage_available`` is this process's SageAttention probe (see
engine/transformer/sage_attention_service.probe_sage); the app publishes it as
``acceleration.sage_available`` on GET /status. ``done.attention_used`` is what
the finished job ACTUALLY ran on — "sdpa", "sage", or "sage->sdpa" when sage was
asked for but degraded (unavailable, or a kernel call raised mid-job). It rides
the same route as ``seed_used`` into metadata.json, and it — not any log line —
is the judging criterion for the sage real-device gates.

``done.block_swap_prefetch_used`` is the block-swap-prefetch analogue: "off",
"on", or "on->off" when prefetch was requested but degraded (block swap not
installed, pinned-memory allocation failed, etc.) — see
``FastVideoPipeline.block_swap_prefetch_used()``. Unlike attention, this is a
pure transfer-mechanism switch, so it never changes generated bytes.

``done.keep_resident_used`` is the same idea for the cross-job CPU-skeleton
cache: "off", "on", or "on->off" when the job asked for it but a worker-side
guard downgraded it (see ``_resolve_keep_resident`` — the reason is always
spelled out in a WARNING on STDERR, because "on->off" alone does not say
WHICH guard fired). Bit-identical output either way (§47.3 G9).

``done.peak_vram_reserved_mb`` is ``torch.cuda.max_memory_reserved()`` in MB,
reported ADDITIVELY alongside the existing ``peak_vram_mb`` (which is
``max_memory_allocated``-based and cannot see allocator-reserved-but-unused
growth from stream-separate pools) — it is the VRAM-risk signal for this
feature's real-device gate.

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
from engine.transformer.nag_service import NagParams  # noqa: E402
from engine.transformer.sage_attention_service import probe_sage  # noqa: E402
from engine.transformer.vsf_service import VsfParams  # noqa: E402

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
    # Probe BEFORE the (multi-minute, memory-hungry) pipeline build: probe_sage()
    # is fully guarded and cached, so this cannot fail the load, and doing it
    # first means the answer is already known no matter which branch below emits
    # ``ready``. The early-return branch is currently unreachable (the parent
    # never sends a second "load"), but it must carry the same field.
    sage_available = probe_sage()
    if _PIPE is not None:
        _emit("ready", sage_available=sage_available)
        return

    _log(f"sage_available={sage_available}")
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
        # Stage 3: load-once / keep-resident weights (StateDictRegistry).
        # ALWAYS False at create time — this is now a PER-JOB setting
        # (``keep_resident`` on the generate payload), armed/disarmed by
        # ``LTXFastVideoPipeline._set_keep_resident_job`` at each job's entry
        # point. The old ``LTX_KEEP_RESIDENT`` env var is gone entirely (it
        # defaulted to "1" here while services/ltx_runner.py forced "0" into the
        # child env — an asymmetry that only worked because both halves agreed
        # by accident). Starting False also means a worker that never receives
        # a keep_resident job behaves exactly as it does today.
        keep_resident_weights=False,
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
    _emit("ready", sage_available=sage_available)


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


def _resolve_nag(msg: dict) -> "NagParams | VsfParams | None":
    """Resolve a worker ``nag`` block -> NagParams / VsfParams, or None when
    absent/falsy.

    ``nag`` is present only when the app/API layer had a non-CFG negative
    prompt enabled for this job (payload is additive — absent for every
    pre-NAG caller and every disabled request, so this returns None and the
    job is byte-identical to before the feature existed).

    ``method`` selects between the two methods and defaults to ``"nag"`` when
    the key is missing: the app layer gained that key with VSF, so an older
    client (or a replayed pre-VSF payload) must keep resolving to exactly the
    NAG params it always did. An UNKNOWN method is a different situation
    entirely — it means the two layers disagree — and fails loudly rather than
    quietly falling back to the wrong algorithm. VSF's own knob defaults to
    the API's default (scale 1.5) for the same forward-compatibility reason.
    """
    blk = msg.get("nag")
    if not blk:
        return None
    method = str(blk.get("method", "nag"))
    if method == "nag":
        return NagParams(
            negative_prompt=str(blk["negative_prompt"]),
            scale=float(blk["scale"]),
            tau=float(blk["tau"]),
            alpha=float(blk["alpha"]),
        )
    if method == "vsf":
        return VsfParams(
            negative_prompt=str(blk["negative_prompt"]),
            scale=float(blk.get("vsf_scale", 1.5)),
        )
    raise RuntimeError(
        f"worker: unknown negative-prompt method {method!r} in the job's "
        "'nag' block — expected 'nag' or 'vsf'."
    )


def _neg_label(nag: "NagParams | VsfParams | None") -> str:
    """Job-log tag for the non-CFG negative-prompt method: off / nag / vsf.

    Replaces the old ``nag=on|off``: with two methods, "on" no longer says
    which algorithm actually ran, and that is the first thing anyone reading
    a log for a suspicious result needs to know.
    """
    if nag is None:
        return "off"
    return "vsf" if isinstance(nag, VsfParams) else "nag"


def _resolve_attention(msg: dict) -> tuple[str, bool]:
    """Resolve a job's ``attention_backend`` -> ``(effective_backend, degraded)``.

    ``degraded`` means "sage was asked for but this process cannot deliver it",
    which is what turns into ``attention_used="sage->sdpa"`` below.

    Missing key -> "sdpa": the payload is additive, so every pre-Acceleration
    caller (and every default request, which does not send the key at all)
    resolves to exactly the behaviour it always had.

    An UNKNOWN value fails the job loudly, exactly like ``_resolve_nag``'s
    unknown method: it can only mean the app and the engine disagree about the
    protocol, and quietly running the wrong backend would be recorded as a
    truthful-looking ``attention_used`` for a job that ignored the request.

    ``sage`` when SageAttention is unavailable is a DIFFERENT case and is
    deliberately not fatal: this is a speed knob, so "ran, just not faster"
    beats "failed". Warn, degrade, and make the degradation visible in the
    job's metadata rather than only in this log.
    """
    backend = str(msg.get("attention_backend", "sdpa"))
    if backend not in ("sdpa", "sage"):
        raise RuntimeError(
            f"worker: unknown attention_backend {backend!r} — expected "
            "'sdpa' or 'sage'."
        )
    if backend == "sage" and not probe_sage():
        # ASCII only, deliberately. This goes straight to STDERR, which on a
        # Japanese Windows is cp932 with errors="backslashreplace" - an em dash
        # here would land in logs/ltx_worker.log as a backslash-u2014 escape,
        # right in the middle of the one sentence an operator reads when
        # asking "why did my sage job run slow?". (Nothing crashes either way;
        # backslashreplace is exactly why it does not - this is only
        # legibility, which is why the RuntimeError above can keep its dash - that
        # one is JSON-escaped onto the protocol channel, never printed raw.)
        _log(
            "WARNING attention_backend='sage' requested but SageAttention is not "
            "available in this engine venv - falling back to sdpa for this job "
            "(reported as attention_used='sage->sdpa')."
        )
        return "sdpa", True
    return backend, False


def _attention_used(effective: str, degraded: bool) -> str:
    """The ``done`` event's ``attention_used``: what the job ACTUALLY ran on.

    Three sources, in order: the pre-job availability degrade (``degraded``),
    the backend that was handed to the pipeline, and — only when sage really did
    start — the pipeline's own record, which reports "sage->sdpa" if a kernel
    call raised mid-job and latched the rest of the job onto sdpa.
    """
    if degraded:
        return "sage->sdpa"
    if effective != "sage":
        return "sdpa"
    assert _PIPE is not None  # only reachable from a post-load generate op
    return _PIPE.attention_used()


def _resolve_block_swap_prefetch(msg: dict) -> bool:
    """Resolve a job's ``block_swap_prefetch``. Missing key -> False.

    Unlike ``_resolve_attention``, an invalid type is not fail-loud — this is a
    speed knob, not a correctness precondition (same relaxed regime as sage's
    unavailable-degrade). There is also no "unknown value" concept here since
    it is not an enum; ``bool()`` simply coerces whatever was sent.
    """
    return bool(msg.get("block_swap_prefetch", False))


def _block_swap_prefetch_used() -> str:
    """The ``done`` event's ``block_swap_prefetch_used``: "off" / "on" /
    "on->off" (requested but degraded — block swap not installed, pinned
    allocation failed, etc.). Reads the pipeline's per-job record, mirroring
    ``_attention_used`` above.
    """
    assert _PIPE is not None
    return _PIPE.block_swap_prefetch_used()


def _resolve_keep_resident(msg: dict, bs_prefetch: bool) -> tuple[bool, str | None]:
    """Resolve a job's ``keep_resident`` -> ``(effective, degraded_reason)``.

    Missing key -> ``(False, None)``: the payload is additive, so every
    pre-keep_resident caller resolves to today's behaviour. "Absent means off"
    is ALSO the explicit release trigger — the pipeline frees the ~20GB cache
    the first time a job resolves to False after a job that resolved to True.

    Three guards, all on the ON path only (an OFF job is never blocked):

    **G-A ``gguf_per_layer_quant=False`` -> RuntimeError (fail loud).** The
    bf16 fused path (engine/gguf/loader_service.py) applies IC-LoRA by
    ``weight.add_()`` — an IN-PLACE mutation of the state dict. With a
    persistent registry that mutation is written straight into the cached
    tensors, so every later job silently inherits the LoRA. This is a
    CORRECTNESS problem, not a speed one, hence the different regime from the
    two guards below. Unreachable through today's API (the app always loads
    with per-layer quant on), kept as the breakwater for whoever wires that
    switch up later.

    **G-B ``dit_cpu_load=False`` -> warn + auto-off.** Not a VRAM issue (the
    §47.3 G8 measurement showed peak VRAM unchanged within +-3MB): with the DiT
    built on the GPU, block swap keeps its own CPU eviction copies of the
    blocks, which DOUBLE UP with the same blocks living in the cache — ~11GB of
    main memory for nothing, on the exact axis this feature is already tight on.

    **G-C ``block_swap_prefetch=False`` -> warn + auto-off.** Same doubling,
    worse: the synchronous swap path makes a fresh GPU->CPU eviction copy every
    step while the cache holds the CPU master anyway (+11.4GB). §47.3 G7
    already measured commit at 96% of the ceiling with prefetch ON; stacking
    this on top lands in crash territory. And unlike G-B this combination is
    two adjacent checkboxes away in the UI, so it is a NORMAL thing to do by
    accident.

    Residual risk accepted (documented, not guarded): prefetch can still
    degrade LATER, inside install() (pinned allocation failure etc.), after
    this resolution has already said yes. Rare, and visible after the fact via
    ``block_swap_prefetch_used="on->off"`` next to ``keep_resident_used="on"``.

    Every auto-off logs WHY: ``keep_resident_used="on->off"`` in the metadata
    cannot tell G-B from G-C from a mid-job failure on its own.
    """
    if not bool(msg.get("keep_resident", False)):
        return False, None

    assert _PIPE is not None  # only reachable from a post-load generate op
    # 直接属性アクセス（getattrの既定値ではなく）：属性が消えたらガードが
    # 黙って素通りになるより AttributeError で落ちるほうがよい。
    if not _PIPE._gguf_per_layer_quant:
        raise RuntimeError(
            "worker: keep_resident=1 is not allowed with "
            "gguf_per_layer_quant=0 - the bf16 fused-LoRA path mutates the "
            "state dict in place, which would permanently contaminate the "
            "cross-job weight cache (every later job would silently inherit "
            "this job's LoRA). Load the model with per-layer quant, or run "
            "this job with keep_resident off."
        )
    # ASCII only in these two WARNINGs, deliberately: STDERR on a Japanese
    # Windows is cp932 + backslashreplace, and these lines are exactly what an
    # operator reads when asking "why does my metadata say on->off?".
    if not _PIPE._dit_cpu_load:
        _log(
            "WARNING keep_resident=1 requested but dit_cpu_load=0 - block swap "
            "would keep its own CPU eviction copies of the DiT blocks ON TOP OF "
            "the cached copies (about 11GB of main memory doubled up). Running "
            "this job with keep_resident OFF (reported as "
            "keep_resident_used='on->off')."
        )
        return False, "dit_cpu_load=0"
    if not bs_prefetch:
        _log(
            "WARNING keep_resident=1 requested but block_swap_prefetch=0 - the "
            "synchronous swap path writes a fresh GPU->CPU eviction copy every "
            "step while the cache already holds the CPU master (+11.4GB), and "
            "commit was already measured at 96 percent of the ceiling WITH "
            "prefetch on. Running this job with keep_resident OFF (reported as "
            "keep_resident_used='on->off')."
        )
        return False, "block_swap_prefetch=0"
    return True, None


def _keep_resident_used(msg: dict, effective: bool) -> str:
    """The ``done`` event's ``keep_resident_used``: "off" / "on" / "on->off".

    Unlike ``_block_swap_prefetch_used`` this does NOT read pipeline state:
    every downgrade for this feature happens up-front in
    ``_resolve_keep_resident`` (the arm itself never raises and never
    half-applies), so the request and the resolved value are the whole story.
    """
    if effective:
        return "on"
    return "on->off" if bool(msg.get("keep_resident", False)) else "off"


def _peak_vram_reserved_mb() -> int:
    """``torch.cuda.max_memory_reserved()`` in MB. The existing ``peak_vram_mb``
    (``max_memory_allocated``-based) cannot see allocator-reserved-but-unused
    growth from stream-separate pools, so this rides alongside it as an
    additional signal for this feature's VRAM real-device gate — it does not
    replace ``peak_vram_mb``.
    """
    return int(torch.cuda.max_memory_reserved(DEV) // (1024 * 1024))


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
        (
            str(lo["path"]),
            float(lo["strength"]),
            None if lo.get("audio_strength") is None else float(lo["audio_strength"]),
        )
        for lo in msg.get("loras", [])
    ]
    # ``reference_video`` -> (ic_reference, attn_strength). The Phase C control
    # preprocess (edge/pose) and the conditioning_attention_strength knob are
    # both resolved inside the shared helper (see _resolve_ic_reference); no
    # reference -> (None, 1.0), inert + byte-identical to before.
    ic_reference, attn_strength = _resolve_ic_reference(
        msg.get("reference_video"), output_path
    )
    # NAG (non-CFG negative prompt guidance): absent/falsy "nag" -> None, byte-
    # identical to before this feature existed.
    nag = _resolve_nag(msg)
    # Attention backend (speed only): absent -> "sdpa", byte-identical to before
    # this feature existed.
    attention, attn_degraded = _resolve_attention(msg)
    # Block-swap prefetch (speed only, output bit-identical): absent -> False,
    # byte-identical to before this feature existed.
    bs_prefetch = _resolve_block_swap_prefetch(msg)
    # Cross-job CPU-skeleton cache (preprocessing speed only, output
    # bit-identical): absent -> False, byte-identical to before. May raise
    # (G-A) or auto-off (G-B/G-C) — see _resolve_keep_resident.
    keep_res, _keep_res_reason = _resolve_keep_resident(msg, bs_prefetch)

    _log(
        f"generating {msg['width']}x{msg['height']} / {msg['num_frames']} frames "
        f"/ {msg['num_steps']} steps seed={seed} images={len(images)} "
        f"ic_loras={len(ic_loras)} ic_reference={'yes' if ic_reference else 'no'} "
        f"neg={_neg_label(nag)} attn={attention} bsprefetch={bs_prefetch} "
        f"keepresident={keep_res}"
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
            nag=nag,
            attention_backend=attention,
            block_swap_prefetch=bs_prefetch,
            # 常に明示的な bool を渡す（None="触らない"はスパイク互換用の
            # 既定であって、ワーカーからは使わない）。
            keep_resident=keep_res,
        )
    finally:
        progress_shim.end_op()

    peak = torch.cuda.max_memory_allocated(DEV) // (1024 * 1024)

    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"engine produced no/empty output: {output_path}")

    attention_used = _attention_used(attention, attn_degraded)
    bs_prefetch_used = _block_swap_prefetch_used()
    keep_res_used = _keep_resident_used(msg, keep_res)
    peak_reserved = _peak_vram_reserved_mb()
    _log(
        f"GENERATED_OK peak_vram_mb={peak} attention_used={attention_used} "
        f"block_swap_prefetch_used={bs_prefetch_used} "
        f"keep_resident_used={keep_res_used} "
        f"peak_vram_reserved_mb={peak_reserved} -> {output_path}"
    )
    _emit(
        "done",
        seed_used=seed,
        peak_vram_mb=int(peak),
        attention_used=attention_used,
        block_swap_prefetch_used=bs_prefetch_used,
        keep_resident_used=keep_res_used,
        peak_vram_reserved_mb=peak_reserved,
    )

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
        (
            str(lo["path"]),
            float(lo["strength"]),
            None if lo.get("audio_strength") is None else float(lo["audio_strength"]),
        )
        for lo in msg.get("loras", [])
    ]
    ic_reference, ic_attn = _resolve_ic_reference(
        msg.get("reference_video"), output_path
    )

    # Opt-in memory-bounded spatial upsample (additive; default False keeps the
    # one-shot whole-timeline upsample byte-identical). When True the engine
    # upsamples the assembled stage-1 latent in halo-padded temporal chunks so a
    # long 768p chain fits in 16GB VRAM.
    chunked_upsample = bool(msg.get("chunked_upsample", False))

    # NAG (non-CFG negative prompt guidance): absent/falsy "nag" -> None, byte-
    # identical to before this feature existed.
    nag = _resolve_nag(msg)
    # Attention backend (speed only): absent -> "sdpa", byte-identical to before.
    attention, attn_degraded = _resolve_attention(msg)
    # Block-swap prefetch (speed only, output bit-identical): absent -> False.
    bs_prefetch = _resolve_block_swap_prefetch(msg)
    # Cross-job CPU-skeleton cache: absent -> False; may raise (G-A) or
    # auto-off (G-B/G-C). Same helper as the single-generate path.
    keep_res, _keep_res_reason = _resolve_keep_resident(msg, bs_prefetch)

    _log(
        f"generate_chain {msg['width']}x{msg['height']} clips={len(clips)} "
        f"frames={[c.num_frames for c in clips]} seed={seed} "
        f"overlap={msg.get('overlap_frames')}/{msg.get('overlap_strength')} "
        f"source={'yes(ctx=' + str(source.context_frames) + ')' if source else 'no'} "
        f"audio_source={'yes' if audio_source else 'no'} "
        f"ic_loras={len(ic_loras)} neg={_neg_label(nag)} attn={attention} "
        f"bsprefetch={bs_prefetch} keepresident={keep_res}"
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
        nag=nag,
        attention_backend=attention,
        block_swap_prefetch=bs_prefetch,
        keep_resident=keep_res,
    )

    peak = torch.cuda.max_memory_allocated(DEV) // (1024 * 1024)
    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"engine produced no/empty chain output: {output_path}")

    attention_used = _attention_used(attention, attn_degraded)
    bs_prefetch_used = _block_swap_prefetch_used()
    keep_res_used = _keep_resident_used(msg, keep_res)
    peak_reserved = _peak_vram_reserved_mb()
    _log(
        f"CHAIN_OK peak_vram_mb={peak} attention_used={attention_used} "
        f"block_swap_prefetch_used={bs_prefetch_used} "
        f"keep_resident_used={keep_res_used} "
        f"peak_vram_reserved_mb={peak_reserved} -> {output_path}"
    )
    _emit(
        "done",
        seed_used=seed,
        peak_vram_mb=int(peak),
        chain=meta,
        attention_used=attention_used,
        block_swap_prefetch_used=bs_prefetch_used,
        keep_resident_used=keep_res_used,
        peak_vram_reserved_mb=peak_reserved,
    )

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
