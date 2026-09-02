"""Persistent LTX-2.5 generation worker (§3-98 Phase 1 skeleton).

Runs inside .venv-engine-ltx25 -- the ONLY environment that has torch 2.9.1+cu128
together with official LTX-2 v1.2.0 (ltx_core / ltx_pipelines @ d151147) and
transformers 5.x. It is a SIBLING of engine/worker.py, never a replacement: the
2.3 worker keeps its own venv (.venv-engine, transformers 4.57) and neither
process can import the other's stack.

The app process (FastAPI, its own torch-less .venv) spawns ONE of these per
loaded model and talks to it over the same tiny JSON-lines protocol the 2.3
worker uses, so the parent-side plumbing is shared. Only small control JSON
crosses the pipe; the mp4 is written by the engine directly to a shared-disk
path.

WHAT THIS FILE DELIBERATELY DOES NOT DO
    The pieces the 2.3 worker installs at import time -- the pre-denoise
    ``empty_cache`` monkeypatch, the ``progress_shim`` tqdm swap, the CUDA
    pre-warm -- are absent, and stay absent. Each is tuned to the 2.3 wheel's
    internals; re-applying them blind to a different pipeline is exactly the
    kind of borrowed-assumption bug this separate engine exists to avoid. The
    SageAttention PROBE is the one that came back, and it came back on this
    engine's own terms: 2.3 runs it at import time, here it is a lazy import
    inside ``load``, after the cheap path checks, so a typo'd model path is
    still reported in under a second and the module scope stays import-free.
    Per-step progress in particular needs no shim here: the official denoising
    loops call their ``denoiser`` once per step, so
    :mod:`engine25.pipeline25` counts steps at that seam instead of swapping
    tqdm out from under the library.

Bootstrap (mirrors engine/worker.py, minus the 2.3-specific pre-warm):
  * TORCH_COMPILE_DISABLE=1 set before importing torch.
  * chdir(ROOT) + sys.path.insert(0, ROOT) so the first-party ``engine25.*``
    package (and the venv-installed ``ltx_core`` / ``ltx_pipelines``) import,
    whether launched as ``python -m engine25.worker`` or as a bare script path.
  * torch is NOT imported at module scope; the heavy imports live inside
    ``load``, so a process that only has to answer ``shutdown`` never pays for
    them (``ltxcore_compat.verify`` alone drags in all of ltx_pipelines).

Protocol (one JSON object per line; parent -> worker):
  {"op": "load", transformer_path, text_encoder_path, video_vae_path,
   audio_vae_path, spatial_upsampler_path,
   [text_encoder_assets_path], [blocks_on_gpu], [te_layers_on_gpu],
   [cache_weights]}
      Every path present and non-empty is checked for existence first, then the
      pipeline is assembled (see :mod:`engine25.pipeline25`). The three optional
      numeric knobs are the documented 16 GB fallback ladder -- blocks 8/6/4 --
      exposed as payload fields so the ladder is a config change and not a code
      change. A failure here is fatal: ``error`` + exit 1, which is what the
      app's load-failure path expects.
  {"op": "generate", prompt, seed, width, height, num_frames, frame_rate,
   output_path, [images], loras, reference_video, [outpaint]}
      One two-stage generation, mp4 written by this process to ``output_path``.
      ``images`` empty/absent -> T2V; entries -> I2V. Fields the v1 contract
      ignores (negative_prompt, num_steps, vae_mode, ...) may ride along; each
      is logged as ignored and dropped. ``crop_output`` is NOT
      one of them -- it never reaches the worker in either engine, because it is
      an ffmpeg post-process the app applies to the finished mp4.
      ``loras`` = [{path, strength, [audio_strength]}...] is the job's Style /
      IC adapters and is ALWAYS a key, even as ``[]``: an explicit empty list is
      the authoritative "no adapter this job" that makes the previous job's
      attachment be cleared rather than inherited. ``reference_video`` =
      {path, strength, preprocess, [attention_strength]} | null is the IC-LoRA's
      reference video; ``preprocess`` != "none" turns it into a control signal
      (``control_<kind>.mp4`` next to the output) via engine/preprocess/ first.
      ``outpaint`` = {source_path, canvas_width, canvas_height, pad_*,
      blend_dilation_stage1, blend_dilation_stage2, freeze_source_audio} is
      ADDITIVE and ABSENT on every other job, so their payloads are unchanged.
      Its PRESENCE routes the op to :mod:`engine25.outpaint25` instead of the
      plain generation, which is why it carries the whole canvas geometry (the
      engine rebuilds the blend mask from it) rather than a flag.
      ``reference_video.path`` is then ALREADY the app-built green canvas, and
      ``source_path`` is the ORIGINAL upload, read only for its audio -- the
      canvas is written without an audio stream on purpose.
      The ``done`` reply adds ``vae_mode_used`` (which VAE decoder this
      checkpoint built -- "diff" or "conv"; a load-time fact, 台帳 §3-131) and
      ``ltx25`` (encode_fps/video_chunks/tiling/size_bytes/phases -- deliberately
      no ``sampler``, that is already ``ready.sampler``) to EVERY generate job,
      plain or outpaint. On an outpaint job it ALSO adds ``outpaint``: the job's
      metadata dict (geometry, blend, sigmas, the audio freeze proof, and its
      own nested ``ltx25`` sub-dict, a superset of the plain one -- the SAME
      dict object as the top-level ``ltx25`` key, not a second copy).
      ``outpaint`` itself rides the event and the ``GENERATE_REPORT`` line
      only; metadata.json does NOT carry it. ``vae_mode_used`` and ``ltx25``
      are different: metadata.json DOES carry both -- the app reads only the
      top-level ``ltx25``, never ``outpaint.ltx25``.
  {"op": "generate_chain", output_path, seed, clips, width, height, frame_rate,
   num_steps, overlap_frames, overlap_strength, [chunked_upsample],
   [stage2_window], [source], [audio_source], [retake], [end_source]}
      One masked AV-latent clip chain -> ONE mp4 (:mod:`engine25.chain25`). The
      body keys are the 2.3 chain op's, verbatim, because the app builds one
      payload shape for whichever engine is loaded. ``clips`` entries are
      {prompt, num_frames, images}; only clip 0 may carry images.
      ``source`` = {path, context_frames} continues an existing video (V2V):
      ``path`` is an mp4 the app has ALREADY cut to the requested frame rate,
      and its first ``context_frames`` pixel frames are frozen as the head of
      clip 0 and trimmed back off before delivery. ``audio_source`` = {path}
      renders a timeline against a given audio track (A2V) and muxes that
      waveform verbatim. The two are mutually exclusive. Both are optional and
      ABSENT on a plain T2V/I2V chain, whose output is byte-identical to the
      chain that shipped before they existed.
      ``retake`` = {path, head_px, tail_px, regenerate_audio} regenerates the
      MIDDLE of an existing clip: ``path`` is the frame-exact CFR window the app
      already cut, and the two glue widths are the pixel bands kept frozen at
      its two ends. Mutually exclusive with both source modes. Optional and
      ABSENT on every other chain.
      ``end_source`` = {path, context_frames, strength} makes the chain END on
      the user's material: ``path`` is the app-cut video (a still image was
      looped into one app-side) of ``context_frames + 1`` frames -- the extra
      one is the causal VAE's primer, dropped after the encode. Mutually
      exclusive with ``retake`` and ``audio_source``, COMBINABLE with
      ``source``. Optional and ABSENT on every other chain.
      ``loras`` / ``reference_video`` are the single op's blocks, applied
      uniformly across the chain -- the ONE reference is sliced per stage-1
      segment by ``run_chain`` -- and are ADDITIVE here (sent only when asked
      for), which is 2.3's chain payload shape verbatim.
      Unlike ``generate``, a field naming a feature this chain does not have
      (``nag``, ``vae_mode``) is REFUSED BY NAME rather than
      ignored -- see ``CHAIN_UNSUPPORTED_KEYS``. Both are engine-level knobs;
      no chain MODE is on that list any more.
      The four acceleration knobs are NOT on that list: all of them apply to
      the chain unchanged.
      The ``done`` reply adds ``vae_mode_used`` (same fact and vocabulary as the
      single op's, 台帳 §3-131) and ``chain``: the whole layout + metadata dict,
      in 2.3's shape -- including its ``v2v`` / ``a2v`` blocks when those modes
      ran, and its own ``ltx25`` sub-dict (``chain["ltx25"]``). Unlike the
      single op, no additive top-level ``ltx25`` key: ``chain`` already carries
      it, one level down.
  {"op": "shutdown"}

Replies are framed with the SAME unique prefix as the 2.3 worker so the shared
parent-side reader needs no branch, and so library/tqdm stdout noise stays
ignorable. Every protocol line: @@LTX@@<compact-json>, flushed. All other
logging goes to STDERR.
  @@LTX@@{"event":"ready","sampler":"euler_ancestral","sage_available":false}
  @@LTX@@{"event":"progress","stage":"stage1_denoise","index":3,"total":8}
  @@LTX@@{"event":"progress","stage":"stage1_denoise","index":3,"total":8,
          "outer_index":1,"outer_total":2}   <- chain only: which clip/tile
  @@LTX@@{"event":"done","seed_used":...,"peak_vram_mb":...,
          "vae_mode_used":"conv","ltx25":{...},...}
  @@LTX@@{"event":"error","detail":...}

``progress`` stage names are the 2.3 worker's names on purpose (``encode`` /
``stage1_denoise`` / ``stage2_denoise`` / ``decode``): the app-side receipt loop
already maps them to labels and job fractions, so a second vocabulary would buy
nothing and cost a branch.

``ready.sampler`` is the sampler this process actually holds. It carries the
resolved name, which is how the ``use_ancestral_sampler`` assertion becomes
observable to the app instead of living only in a log line -- the official code
silently downgrades that flag to False on GGUF paths and only WARNs.

``ready.sage_available`` is the MEASURED answer for this process, from
``engine.transformer.sage_attention_service.probe_sage`` (it used to be a
hard-coded false, from the days when this engine was SDPA-only and the wheel was
not installed in .venv-engine-ltx25). The app publishes it as
``acceleration.sage_available`` on GET /status for whichever engine is loaded,
which is what greys the option out when the wheel is missing or unusable.

Selftest (gate G4), run inside the venv without the app::

    python -m engine25.worker --selftest-generate \\
        --transformer <t.gguf> --text-encoder <te.gguf> \\
        --video-vae <conv.safetensors> --audio-vae <audio.safetensors> \\
        --spatial-upsampler <x2.safetensors> \\
        --output out.mp4 --width 320 --height 192 --num-frames 25 [--rounds 2] \\
        [--fused-dequant on|off]

It drives the SAME ``_do_load`` / ``_do_generate`` handlers the protocol uses --
it builds the JSON messages and feeds them in -- so a green selftest is evidence
about the shipped path, not about a parallel one. Its JSON report carries the
``done`` event verbatim, which is where the acceleration echoes live: run the
same command twice with ``--fused-dequant off`` and ``on`` and the two digests
must be IDENTICAL, because the fused kernels are a bit-exact substitution.
``--attention`` is the opposite kind of knob and its pair reads the opposite
way: sage is a QUANTIZED kernel, so ``sage`` and ``sdpa`` at one seed must
DIFFER, and two ``sage`` rounds at one seed must agree with each other.

The chain has its own (gate G2), same discipline, plus captured receipts::

    python -m engine25.worker --selftest-chain \\
        <the same five model paths> \\
        --output chain.mp4 --clips 2 --num-frames 25 \\
        --width 320 --height 192 [--chunked-upsample] [--rounds 2] \\
        [--source tail.mp4 --context 25 | --audio-source track.wav] \\
        [--fused-dequant on|off]

Its JSON report carries the mp4 digests, the ``done`` event verbatim (with its
``chain`` metadata) and the full ``progress`` series, so the event CONTRACT --
which stages fire, and which of them name an ``outer_index``/``outer_total`` --
is checked from the run instead of from the source.
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# import cwd = project root: make the first-party `engine25.*` package (and the
# venv-installed `ltx_core`/`ltx_pipelines`) importable. Launched as
# `python -m engine25.worker` with PYTHONPATH=<root>, but we also insert the
# root on sys.path + chdir(ROOT) here so a bare `python engine25/worker.py`
# still works.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Unique frame prefix for every protocol line the parent parses. Intentionally
# IDENTICAL to engine/worker.py's: the parent's line reader is shared, and the
# two workers never run in the same process, so a second prefix would buy
# nothing and cost a branch on the app side.
PREFIX = "@@LTX@@"


def _configure_worker_logging() -> None:
    """Route module-level ``logging`` calls to STDERR.

    Same shape as the 2.3 worker's: ONE StreamHandler on the root logger aimed
    at STDERR (which the parent redirects into the worker log file), tagged so
    re-entry cannot double it. STDOUT is never touched -- that channel belongs
    exclusively to the ``@@LTX@@`` protocol. The ``[ltx25_worker]`` prefix is
    what tells the two engines' log lines apart when both have run.
    """
    root = logging.getLogger()
    if not any(getattr(h, "_ltx25_worker_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[ltx25_worker] %(name)s: %(message)s"))
        handler._ltx25_worker_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(logging.INFO)


_configure_worker_logging()


def _log(msg: str) -> None:
    """Human/diagnostic logging -> STDERR only (never the protocol channel)."""
    print(f"[ltx25_worker] {msg}", file=sys.stderr, flush=True)


def _emit(event: str, **fields: object) -> None:
    """Write one framed protocol line to STDOUT and flush."""
    payload = {"event": event, **fields}
    print(PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def _detail(exc: BaseException) -> str:
    """repr + head/tail of the traceback for an error reply.

    Head AND tail: tail-only truncation drops the OUTER frames, which is where
    "which of our call sites raised" lives.
    """
    tb = "".join(traceback.format_exc())
    if len(tb) > 2800:
        tb = f"{tb[:1200]}\n...[traceback truncated]...\n{tb[-1600:]}"
    return f"{exc!r}\n{tb}"


# The load payload's path fields, in the order they are reported. Every one is
# OPTIONAL at the protocol level (an absent or empty value is simply not
# checked) so the adapter can grow the payload without this skeleton rejecting
# older or shorter messages; what the app actually guarantees is enforced on the
# app side by the manifest's REQUIRED_ASSETS.
_LOAD_PATH_FIELDS = (
    "transformer_path",
    "text_encoder_path",
    "video_vae_path",
    "audio_vae_path",
    "spatial_upsampler_path",
)

#: The one loaded pipeline. One model per process, exactly like the 2.3 worker:
#: the app spawns a worker per loaded model and unloads by killing it, so a
#: second ``load`` on a live process is a bug on the app side, not a mode.
_PIPE = None


def _emit_progress(
    stage: str,
    index: int,
    total: int,
    *,
    outer_index: int | None = None,
    outer_total: int | None = None,
) -> None:
    """One framed ``progress`` line. Never allowed to fail a job.

    The two keyword-only arguments are a CHAIN's position -- which clip is being
    denoised, which stage-2 tile -- and are emitted under the SAME field names
    the 2.3 worker uses (``engine/worker.py``'s ``_emit_step_progress``), because
    the app-side receipt loop that reads them is shared: without them a chain's
    job fraction rewinds to the start of stage 1 at every clip, and the "clip
    n/N" line the GUI shows has nothing to read.

    They are OMITTED from the line when absent, so a single generation's
    ``progress`` events stay byte-identical to what they were before the chain
    existed -- the single path calls this with three positional arguments and
    nothing else.
    """
    try:
        fields: dict[str, object] = {
            "stage": str(stage),
            "index": int(index),
            "total": int(total),
        }
        if outer_index is not None:
            fields["outer_index"] = int(outer_index)
        if outer_total is not None:
            fields["outer_total"] = int(outer_total)
        _emit("progress", **fields)
    except Exception as exc:  # noqa: BLE001
        _log(f"progress emit failed (ignored): {exc!r}")


def _do_load(msg: dict) -> None:
    """Assemble the LTX 2.5 pipeline from the handed-down paths, then answer ``ready``.

    Paths are existence-checked BEFORE the heavy imports: a typo'd manifest
    entry should be reported as a missing file in under a second, not after
    twenty seconds of importing ltx_pipelines. A failure at any point raises,
    and the caller turns that into ``error`` + exit 1 -- fail loud, because a
    half-loaded engine that answers ``ready`` would fail on the first job with a
    far less legible message.
    """
    global _PIPE
    if _PIPE is not None:
        # The probe is cached per process, so the re-answer cannot disagree with
        # the first ``ready``; importing it here rather than at module scope is
        # free on this branch, where the whole stack is already resident.
        from engine.transformer.sage_attention_service import probe_sage  # noqa: PLC0415

        _log("load: already loaded -- re-answering ready")
        _emit("ready", sampler=_PIPE.sampler, sage_available=probe_sage())
        return

    for field in _LOAD_PATH_FIELDS:
        raw = msg.get(field)
        if not raw:
            _log(f"load: {field} absent -- not checked")
            continue
        path = Path(str(raw))
        if not path.exists():
            raise FileNotFoundError(f"{field} does not exist: {path}")
        _log(f"load: {field} OK ({path.stat().st_size} bytes) {path}")

    # AFTER the path checks and BEFORE the pipeline: the checks above exist so a
    # typo'd manifest entry is reported in under a second, and this import pulls
    # in sageattention + triton (seconds, and a DLL load that can fail on a
    # broken wheel). ``probe_sage`` swallows every failure and returns False, so
    # an unusable wheel greys the option out in the UI instead of failing the
    # load -- but it must not get the chance to do either before a missing model
    # file has been named.
    from engine.transformer.sage_attention_service import probe_sage  # noqa: PLC0415

    sage_available = probe_sage()

    from engine25.pipeline25 import (  # noqa: PLC0415 -- deliberately lazy (see module docstring)
        DEFAULT_BLOCKS_ON_GPU,
        DEFAULT_TE_LAYERS_ON_GPU,
        Ltx25Pipeline,
        ModelFiles,
    )

    files = ModelFiles(
        transformer=str(msg["transformer_path"]),
        text_encoder=str(msg["text_encoder_path"]),
        video_vae=str(msg["video_vae_path"]),
        audio_vae=str(msg["audio_vae_path"]),
        spatial_upsampler=str(msg["spatial_upsampler_path"]),
        text_encoder_assets=(
            str(msg["text_encoder_assets_path"]) if msg.get("text_encoder_assets_path") else None
        ),
    )
    pipeline = Ltx25Pipeline(
        files,
        blocks_on_gpu=int(msg.get("blocks_on_gpu", DEFAULT_BLOCKS_ON_GPU)),
        te_layers_on_gpu=int(msg.get("te_layers_on_gpu", DEFAULT_TE_LAYERS_ON_GPU)),
        cache_weights=bool(msg.get("cache_weights", True)),
        deterministic=bool(msg.get("deterministic", True)),
        progress=_emit_progress,
    )
    _PIPE = pipeline
    _log(f"LOAD_OK {json.dumps(pipeline.build_report, ensure_ascii=False, default=str)}")
    # The MEASURED answer, from the probe above -- not a constant. See the
    # ``ready.sage_available`` paragraph in the module docstring.
    _emit("ready", sampler=pipeline.sampler, sage_available=sage_available)


def _ic_loras(msg: dict) -> list[tuple[str, float, float | None]]:
    """``msg['loras']`` -> the ``(path, strength, audio_strength)`` triples.

    ALWAYS a list, and a missing key gives ``[]`` rather than ``None``: an empty
    list is not "no opinion", it is the authoritative "no adapter this job",
    which is what makes ``Ltx25DiffusionStage.set_loras([])`` detach the previous
    job's attachment instead of letting it ride. The app's payload builder sends
    the key on every single generate for exactly that reason.

    The triple is built HERE, unconditionally, even though the normaliser
    downstream accepts 2-tuples -- the same shape 2.3's worker builds
    (``engine/worker.py``), so the two engines hand their (shared) normaliser the
    same thing and a missing ``audio_strength`` means one thing in both.
    """
    return [
        (
            str(lo["path"]),
            float(lo["strength"]),
            None if lo.get("audio_strength") is None else float(lo["audio_strength"]),
        )
        for lo in msg.get("loras") or []
    ]


def _preprocess_frame_cap(msg: dict) -> int | None:
    """Frames the reference preprocessor should decode: the generation length.

    Ported from 2.3's worker unchanged, because it is a statement about the
    GEOMETRY of a job and not about either engine's internals. Depth normalises
    over the whole clip it is given (and canny/dwpose simply waste decode time on
    footage the generation will never use), so this caps the preprocessor's input
    to exactly what stage 1 can consume.

    Single generate -> ``num_frames``. A chain -> the PIXEL TOTAL the stage-1
    ledger spans across every clip (a reference is attached to EVERY clip's
    stage-1 conditioning, sliced per segment by ``video_segment_windows``, so
    clip 0's length is not the cap). Neither key present -> ``None`` = decode
    everything, which is what a caller that supplies no length context gets.

    The arithmetic goes through ``chain_math`` -- the torch-free module BOTH
    engines share -- rather than being restated here, so the cap and the window
    ledger cannot drift apart.
    """
    if "num_frames" in msg:
        return int(msg["num_frames"])
    raw_clips = msg.get("clips") or []
    clip_frames = [int(c["num_frames"]) for c in raw_clips if "num_frames" in c]
    if not clip_frames or len(clip_frames) != len(raw_clips):
        return None

    from chain_math import (  # noqa: PLC0415 -- lazy like every other import here
        DEFAULT_OVERLAP_FRAMES,
        px_from_v_latent,
        v_latent_frames,
    )

    kv = int(msg.get("overlap_frames", DEFAULT_OVERLAP_FRAMES))
    seg_latent = [v_latent_frames(f) for f in clip_frames]
    f_total = sum(seg_latent) - (len(seg_latent) - 1) * kv
    return px_from_v_latent(f_total)


def _resolve_ic_reference(
    ref: dict | None,
    output_path: str,
    frame_cap: int | None = None,
    *,
    op: str = "generate",
) -> tuple[tuple[str, float] | None, float]:
    """A ``reference_video`` block -> ``((path, strength), attention_strength)``.

    ``None``/absent -> ``(None, 1.0)``, which is inert: no reference tokens are
    built and the job is byte-identical to one from before the feature existed.

    ``attention_strength`` (0..1, default 1.0) is the control-adherence knob; at
    1.0 the engine applies no wrapper at all, so the default stays structurally
    identical rather than merely numerically equal.

    When ``preprocess`` is anything but ``"none"`` the raw reference is first
    converted to a CONTROL SIGNAL (edge map / skeleton / depth map) by
    ``engine.preprocess`` -- the 2.3 package, imported here across the engine
    boundary on purpose: it is plain OpenCV/torch code with no ltx wheel in it,
    and a second copy of the same edge detector would be a second thing to keep
    honest. The result is written as ``control_<kind>.mp4`` NEXT TO THE OUTPUT
    (so a job's control signal is inspectable alongside what it produced) and the
    returned path is that file. An unknown kind fails the job loud
    (``get_processor`` raises); ``"none"`` never imports cv2 at all.

    Both entry points call this, so a reference resolves identically whether it
    arrived on a single generate or on a chain. The existence check is this
    engine's regime rather than 2.3's silence (see :func:`_existing_media_path`):
    without it a typo'd path surfaces as whatever the demuxer says two minutes
    into the job, or -- with a preprocess kind -- as an empty control mp4.
    """
    ic_reference = None
    attn_strength = 1.0
    if ref:
        ref_path = _existing_media_path(ref["path"], "reference_video", op=op)
        ref_strength = float(ref.get("strength", 1.0))
        attn_strength = float(ref.get("attention_strength", 1.0))
        preprocess = ref.get("preprocess", "none")
        if preprocess and preprocess != "none":
            import time  # noqa: PLC0415

            from engine.preprocess import (  # noqa: PLC0415
                get_processor,
                preprocess_video,
            )

            processor = get_processor(preprocess)
            control_path = os.path.join(
                os.path.dirname(output_path), f"control_{preprocess}.mp4"
            )
            t0 = time.perf_counter()
            n_frames = preprocess_video(
                Path(ref_path), Path(control_path), processor, frame_cap=frame_cap
            )
            elapsed = time.perf_counter() - t0
            _log(
                f"PREPROCESS {preprocess} {ref_path} -> {control_path} "
                f"frames={n_frames}"
                + ("" if frame_cap is None else f" cap={frame_cap}")
                + f" elapsed={elapsed:.2f}"
            )
            ref_path = control_path
        ic_reference = (ref_path, ref_strength)
    return ic_reference, attn_strength


def _preprocess_kind(msg: dict) -> str:
    """The reference's preprocess kind for the one parse log line ("-" = no reference)."""
    ref = msg.get("reference_video")
    return "-" if not ref else str(ref.get("preprocess", "none"))


def _resolve_block_swap_prefetch(msg: dict) -> bool:
    """Resolve a job's ``block_swap_prefetch``. Missing key -> False.

    2.3's reader verbatim (``engine/worker.py``): an invalid type is not
    fail-loud -- this is a speed knob, not a correctness precondition. There is
    also no "unknown value" concept here since it is not an enum; ``bool()``
    simply coerces whatever was sent. Written the same way in both workers on
    purpose: the app sends ONE payload shape for whichever engine is loaded, and
    two readers disagreeing about what ``"0"`` means would be a real bug that no
    test would catch.
    """
    return bool(msg.get("block_swap_prefetch", False))


def _resolve_fused_dequant(msg: dict) -> bool:
    """Resolve a job's ``fused_gguf_dequant_kernel``. Missing key -> False.

    Same relaxed regime as :func:`_resolve_block_swap_prefetch`, and 2.3's
    reader verbatim for the same reason.
    """
    return bool(msg.get("fused_gguf_dequant_kernel", False))


def _resolve_keep_resident(msg: dict) -> bool:
    """Resolve a job's ``keep_resident``. Missing key -> False.

    Absent-means-off carries MORE weight here than for the other two knobs, and
    it is 2.3's contract verbatim: because the retained weights survive the end
    of the job, a payload without the key is not merely "do not enable it", it is
    **an explicit instruction to release** whatever a previous job left resident.
    The app's payload builders are additive (they send the key only when the
    setting is on), so that release semantics is what an unchecked box produces,
    with no extra field and no extra code path.

    The CONTRACT is 2.3's word for word; the IMPLEMENTATION behind it is not.
    2.3 arms this through a two-argument call that returns a tuple and carries
    three internal degradation guards; here it is one flag on one registry with
    no degradation path at all (see
    :func:`engine25.pipeline25._swap_keep_resident`). Same regime as the other
    two readers otherwise: ``bool()`` coerces whatever arrived rather than
    failing loud, because this is a speed/RAM knob and not a correctness
    precondition.
    """
    return bool(msg.get("keep_resident", False))


def _resolve_nag(msg: dict):
    """Resolve a worker ``nag`` block -> NagParams / VsfParams, or None when
    absent/falsy.

    2.3's reader (``engine/worker.py``) WORD FOR WORD, and for the same reason
    :func:`_resolve_attention` is: the app builds ONE payload shape for whichever
    engine is loaded, so two readers that disagreed about what a block means
    would be a real bug no test would catch. The ONE difference is that the two
    params classes are imported inside the function -- this worker's module
    scope is deliberately import-free, and this only ever runs after ``load``
    pulled the stack in, so it is a ``sys.modules`` hit.

    ``nag`` is present only when the app/API layer had a non-CFG negative prompt
    enabled for this job (the payload is additive -- absent for every request
    that did not ask, so this returns None and the job is byte-identical to
    before the feature existed).

    ``method`` selects between the two methods and defaults to ``"nag"`` when
    the key is missing: the app layer gained that key with VSF, so an older
    client (or a replayed pre-VSF payload) must keep resolving to exactly the
    NAG params it always did. An UNKNOWN method is a different situation
    entirely -- it means the two layers disagree -- and fails loudly rather than
    quietly falling back to the wrong algorithm. VSF's own knob defaults to the
    API's default (scale 1.5) for the same forward-compatibility reason.
    """
    from engine25.neg_prompt25 import NagParams, VsfParams  # noqa: PLC0415

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
        f"engine25 worker: unknown negative-prompt method {method!r} in the "
        "job's 'nag' block — expected 'nag' or 'vsf'."
    )


def _neg_label(nag) -> str:
    """Job-log tag for the non-CFG negative-prompt method: off / nag / vsf.

    2.3's function verbatim. Not ``on|off``: with two methods, "on" no longer
    says which algorithm actually ran, and that is the first thing anyone
    reading a log for a suspicious result needs to know.

    The type import is function-local for the reason :func:`_resolve_nag`'s is,
    and it is a TYPE import only -- this function never constructs anything.
    """
    from engine25.neg_prompt25 import VsfParams  # noqa: PLC0415

    if nag is None:
        return "off"
    return "vsf" if isinstance(nag, VsfParams) else "nag"


def _resolve_attention(msg: dict) -> tuple[str, bool]:
    """Resolve a job's ``attention_backend`` -> ``(effective_backend, degraded)``.

    2.3's reader (``engine/worker.py``) word for word, and deliberately so: the
    app sends ONE payload shape for whichever engine is loaded, so two readers
    that disagreed about what an unknown value means would be a real bug no test
    would catch.

    ``degraded`` means "sage was asked for but this process cannot deliver it",
    which is what turns into ``attention_used="sage->sdpa"`` below.

    Missing key -> "sdpa": the payload is additive, so every caller written
    before this existed (and every default request, which does not send the key
    at all) resolves to exactly the behaviour it always had -- which is what
    keeps the frozen default-job digests valid.

    An UNKNOWN value fails the job loudly. It can only mean the app and the
    engine disagree about the protocol, and quietly running the wrong backend
    would be recorded as a truthful-looking ``attention_used`` for a job that
    ignored the request.

    ``sage`` when SageAttention is unavailable is a DIFFERENT case and is
    deliberately not fatal: this is a speed knob, so "ran, just not faster"
    beats "failed". Warn, degrade, and make the degradation visible in the job's
    metadata rather than only in this log.

    The ``probe_sage`` import is function-local, unlike 2.3's module-level one:
    this worker's module scope is deliberately import-free (see the module
    docstring), and this function only ever runs after ``load`` has already
    pulled the whole stack in, so the lookup is a ``sys.modules`` hit.
    """
    from engine.transformer.sage_attention_service import probe_sage  # noqa: PLC0415

    backend = str(msg.get("attention_backend", "sdpa"))
    if backend not in ("sdpa", "sage"):
        raise RuntimeError(
            f"engine25 worker: unknown attention_backend {backend!r} — expected "
            "'sdpa' or 'sage'."
        )
    if backend == "sage" and not probe_sage():
        # ASCII only, deliberately. This goes straight to STDERR, which on a
        # Japanese Windows is cp932 with errors="backslashreplace" - an em dash
        # here would land in logs/ltx_worker.log as a backslash-u2014 escape,
        # right in the middle of the one sentence an operator reads when asking
        # "why did my sage job run slow?". (Nothing crashes either way;
        # backslashreplace is exactly why it does not - this is only legibility,
        # which is why the RuntimeError above can keep its dash: that one is
        # JSON-escaped onto the protocol channel, never printed raw.)
        _log(
            "WARNING attention_backend='sage' requested but SageAttention is not "
            "available in this engine venv - falling back to sdpa for this job "
            "(reported as attention_used='sage->sdpa')."
        )
        return "sdpa", True
    return backend, False


def _attention_used(effective: str, degraded: bool) -> str:
    """The ``done`` event's ``attention_used``: what the job ACTUALLY ran on.

    2.3's function verbatim (``engine/worker.py``). Three sources, in order: the
    pre-job availability degrade (``degraded``), the backend that was handed to
    the pipeline, and -- only when sage really did start -- the pipeline's own
    record, which reports "sage->sdpa" if a kernel call raised mid-job and
    latched the rest of the job onto SDPA.

    Must be called AFTER ``reset_acceleration_job``: the pipeline's reset is
    what snapshots the live state into the value read here.
    """
    if degraded:
        return "sage->sdpa"
    if effective != "sage":
        return "sdpa"
    assert _PIPE is not None  # only reachable from a post-load generate op
    return _PIPE.attention_used()


def _do_generate(msg: dict) -> None:
    """Run one generation; the mp4 is written by this process to ``msg['output_path']``."""
    if _PIPE is None:
        raise RuntimeError("generate before load")

    from engine25.pipeline25 import (  # noqa: PLC0415
        IGNORED_FIELDS,
        image_conditionings,
    )

    seed = int(msg["seed"])
    images = image_conditionings(msg.get("images") or [])
    # Everything the request carried that this engine does not act on, gathered
    # here so ONE log line per job names them all. Membership, not value, is the
    # test: a caller that sends ``num_steps=8`` still gets told it had no effect.
    ignored = {name: msg[name] for name in IGNORED_FIELDS if name in msg}

    # Style/character LoRA and the IC-LoRA reference video (§3-102 third
    # increment). Both are resolved BEFORE the log line below, because a
    # preprocess kind can take a minute of its own and the line is what says the
    # job understood what it was asked for.
    ic_loras = _ic_loras(msg)
    ic_reference, attn_strength = _resolve_ic_reference(
        msg.get("reference_video"), str(msg["output_path"]), _preprocess_frame_cap(msg)
    )

    # Outpainting (§3-102). MEMBERSHIP is the switch, exactly as on 2.3: the
    # adapter puts this key on the payload only for an outpaint job, so its
    # presence routes the whole op to the two-stage driver below. It carries the
    # canvas GEOMETRY rather than a flag because the engine rebuilds the blend
    # mask from it, and a mask built from anything but the geometry the app
    # validated is the one thing this feature must never do.
    outpaint = msg.get("outpaint")

    # The four acceleration knobs. All are ABSENT-MEANS-OFF, so every payload
    # written before they existed resolves to today's behaviour, and all are
    # armed below rather than passed to ``generate``: they are per-job state on a
    # resident process, not generation parameters. For ``keep_resident`` the
    # absent case is an instruction rather than a default -- see its reader.
    # ``attention_backend`` is the only one that can REFUSE the job (an unknown
    # value is a protocol disagreement), and the only one that returns a pair:
    # the second half is "sage was asked for and this process cannot serve it".
    prefetch = _resolve_block_swap_prefetch(msg)
    fused = _resolve_fused_dequant(msg)
    keep_resident = _resolve_keep_resident(msg)
    attention, attention_degraded = _resolve_attention(msg)

    # The non-CFG negative prompt. ABSENT-MEANS-OFF like the four above, and
    # additive for the same reason: a payload written before this existed
    # resolves to None and the job runs exactly as it did. Unlike those four it
    # is not an acceleration knob -- it changes what the model computes -- which
    # is why it is armed through its own method below.
    nag = _resolve_nag(msg)

    # ONE line per job, after the parse: what the request ASKED for, in the
    # worker's own log, so "I attached a LoRA and nothing happened" can be told
    # apart from "the LoRA never reached the engine" without a rerun. The engine
    # logs what it DID (pipeline25's own line names the resolved downscale
    # factor); this names what arrived.
    _log(
        f"generate {msg['width']}x{msg['height']} / {msg['num_frames']} frames "
        f"seed={seed} images={len(images)} "
        f"ic_loras={len(ic_loras)} ic_reference={'yes' if ic_reference else 'no'} "
        f"preprocess={_preprocess_kind(msg)} attn={attn_strength:.3f} "
        # What was ASKED for. What was GOT is the pair of echo keys on the done
        # event below, which can differ ("on->off").
        f"fused={'on' if fused else 'off'} prefetch={'on' if prefetch else 'off'} "
        f"keep_resident={'on' if keep_resident else 'off'} "
        # The RESOLVED backend, not the raw field: an unavailable wheel has
        # already turned a 'sage' request into 'sdpa' by this line, and the
        # WARNING that says so is immediately above it in the same log.
        f"attention={attention} "
        # off / nag / vsf, in 2.3's spelling: the two engines' worker logs are
        # read side by side when a result is compared.
        f"neg={_neg_label(nag)}"
    )

    # OUTSIDE the try, and before the build: the fused kernels have to be armed
    # before the transformer is built, because the build is where the GGUF
    # weights are dequantized (see ``Ltx25Pipeline.set_acceleration_job``). Being
    # outside the try is the other half of the discipline -- an arm that threw
    # would skip the matching reset and leak this job's request into the next one
    # on a resident worker, so nothing that can throw may sit between the two.
    _PIPE.set_acceleration_job(
        block_swap_prefetch=prefetch,
        fused_gguf_dequant_kernel=fused,
        keep_resident=keep_resident,
        # Stated, not defaulted: the parameter has none. False until the request
        # key is wired up, which is the next step of §3-114.
        keep_resident_embeddings=False,
        attention_backend=attention,
    )
    # OUTSIDE the try for the same reason, and with an ordering constraint of
    # its own that is stricter than any of the four above: the negative prompt
    # is encoded during the PROMPT ENCODE, which is the first thing the job
    # does, and the patch reads ``state.requested`` at transformer-build time.
    # Armed on EVERY job, ``None`` included -- that is what clears a previous
    # job's negative prompt on a resident worker rather than inheriting it.
    _PIPE.set_nag_job(nag)
    try:
        if outpaint is not None:
            # Outpainting (§3-102). ``reference_video.path`` is ALREADY the
            # green canvas the app built, so ``ic_reference`` above points at
            # it and this branch only has to hand ``run_outpaint`` the geometry
            # it needs to rebuild the blend mask. Imported HERE rather than at
            # module scope for the same reason every other engine import in
            # this file is: a worker that only ever loads a model must not pay
            # for torch-heavy modules a job kind it is not running would need.
            from engine.outpaint.canvas import OutpaintGeometry  # noqa: PLC0415
            from engine25.outpaint25 import run_outpaint  # noqa: PLC0415

            # THE GEOMETRY IS THE ONLY SOURCE OF THE OUTPUT SIZE. ``msg`` also
            # carries ``width``/``height``, and they agree -- the adapter fills
            # ``canvas_width``/``canvas_height`` from exactly those two fields
            # -- but ``run_outpaint`` takes no width/height argument at all, so
            # there is no second place for them to be read from and disagree.
            geometry = OutpaintGeometry(
                canvas_width=int(outpaint["canvas_width"]),
                canvas_height=int(outpaint["canvas_height"]),
                pad_left=int(outpaint["pad_left"]),
                pad_right=int(outpaint["pad_right"]),
                pad_top=int(outpaint["pad_top"]),
                pad_bottom=int(outpaint["pad_bottom"]),
            )
            assert ic_reference is not None, (
                "outpaint requires a reference video (the green canvas); the API "
                "layer enforces this before the job is created"
            )
            result = run_outpaint(
                _PIPE,
                prompt=str(msg["prompt"]),
                canvas_path=ic_reference[0],
                source_path=outpaint.get("source_path"),
                geometry=geometry,
                num_frames=int(msg["num_frames"]),
                frame_rate=float(msg["frame_rate"]),
                # Carried and reported, never acted on: the distilled schedule
                # is fixed at 8 + 3 sigmas. The single-generate payload does not
                # carry the key at all (the adapter forwards only what is acted
                # upon), so 0 is the honest "not stated" -- the chain op reads
                # it exactly this way.
                num_steps=int(msg.get("num_steps", 0)),
                seed=seed,
                output_path=str(msg["output_path"]),
                # Passed on EVERY job, ``[]`` included: see :func:`_ic_loras`.
                # An outpaint job can never actually have none -- the official
                # workflow has no LoRA-free path and ``run_outpaint`` raises --
                # but the call shape is the plain generate's either way.
                ic_loras=ic_loras,
                ic_reference=ic_reference,
                ic_attention_strength=attn_strength,
                blend_dilation_stage1=int(outpaint.get("blend_dilation_stage1", 5)),
                blend_dilation_stage2=int(outpaint.get("blend_dilation_stage2", 2)),
                freeze_source_audio=bool(outpaint.get("freeze_source_audio", True)),
                # Reported through pipeline25's own ``_log_ignored``, so an
                # outpaint job's log names the dropped knobs in the same words a
                # plain one's does.
                ignored=ignored,
                progress=_emit_progress,
            )
        else:
            result = _PIPE.generate(
                prompt=str(msg["prompt"]),
                seed=seed,
                width=int(msg["width"]),
                height=int(msg["height"]),
                num_frames=int(msg["num_frames"]),
                frame_rate=float(msg["frame_rate"]),
                output_path=str(msg["output_path"]),
                images=images,
                # Passed on EVERY job, ``[]`` included: see :func:`_ic_loras`.
                ic_loras=ic_loras,
                ic_reference=ic_reference,
                ic_attention_strength=attn_strength,
                ignored=ignored,
            )

        # ONE report line for BOTH job kinds, and deliberately still the
        # ``GENERATE_REPORT`` marker: ``OutpaintResult`` is a strict superset of
        # ``GenerationResult`` (same field names, same ``as_dict`` order plus one
        # additive ``outpaint`` key), so every existing evidence collector that
        # greps for this marker keeps working on an outpaint job.
        report = result.as_dict()
        _log(f"GENERATE_REPORT {json.dumps(report, ensure_ascii=False, default=str)}")
    finally:
        # Runs on the failure path too: this is what stops a crashed job from
        # leaving the kernels armed -- or a negative prompt encoded -- for the
        # next one. Neither call raises. The negative prompt goes FIRST because
        # it is the one holding tensors.
        _PIPE.reset_nag_job()
        _PIPE.reset_acceleration_job()

    # ``outpaint`` is the additive ``done`` key, and it appears only on an
    # outpaint job -- unlike ``vae_mode_used`` and ``ltx25`` (below), which
    # ride EVERY job. The app reads this event by NAME (``event.get(...)``
    # per field), so an unknown key adds a fact without disturbing one. WHAT
    # IS IN IT (aside from its own ``ltx25`` sub-dict, see below) does not
    # reach metadata.json -- that would need an app-layer change, which this
    # theme does not make -- so the freeze proof, the audio branch and the
    # sampler's intentional differences live here and in the GENERATE_REPORT
    # line above. That is where the gates collect them from.
    extra: dict = {} if outpaint is None else {"outpaint": result.metadata}

    # 台帳 §3-131: ``ltx25``, a SECOND and SEPARATE ``done`` key, present on
    # EVERY generate job (plain or outpaint), unlike ``extra`` above. Its
    # content DOES reach metadata.json -- ``services/engines/ltx25/adapter.py``
    # lifts it onto ``GenerationOutcome.ltx25`` and
    # ``services/pipeline_manager.py`` writes it in verbatim when present.
    # ``outpaint25.py`` already builds its own ``ltx25`` sub-dict inside
    # ``result.metadata`` (the same dict that just went into ``extra``), so an
    # outpaint job reuses it rather than building a second, possibly-drifting
    # copy -- the app reads the TOP-LEVEL key only, never ``outpaint.ltx25``,
    # but the two are one and the same object here. A plain generation has no
    # such dict to reuse, so it is built fresh from the same fields the
    # ``done`` event below already reports (deliberately NO ``sampler`` -- a
    # constant already on ``ready.sampler``/LOAD_OK -- and no
    # ``stage1_eta``/``stage2_sampler``/``vram``, which only a two-stage
    # outpaint/chain build has).
    ltx25: dict = result.metadata["ltx25"] if outpaint is not None else {
        "encode_fps": result.encode_fps,
        "video_chunks": result.video_chunks,
        "tiling": result.tiling,
        "size_bytes": result.size_bytes,
        "phases": result.phases,
    }

    # Read the verdicts AFTER the reset -- that is where they are computed
    # ("asked for it and dequantized nothing eligible" is a degradation, and the
    # tally only closes at reset). Outside the try for the same reason the arm
    # is: a failed job has already raised and emits no ``done`` at all.
    _emit(
        "done",
        seed_used=seed,
        sampler=_PIPE.sampler,
        # peak_vram_mb / peak_vram_reserved_mb keep the 2.3 worker's key names
        # and units (MiB): the app already stores and displays both.
        peak_vram_mb=_gib_to_mb(result.peak_allocated_gib),
        peak_vram_reserved_mb=_gib_to_mb(result.peak_reserved_gib),
        rss_peak_gib=result.rss_peak_gib,
        seconds=round(result.seconds, 2),
        num_frames=result.num_frames,
        encode_fps=result.encode_fps,
        size_bytes=result.size_bytes,
        phases=result.phases,
        # 2.3's echo keys, same names and same three values ("off" / "on" /
        # "on->off"), so the app relays them from one code path per engine.
        block_swap_prefetch_used=_PIPE.block_swap_prefetch_used(),
        fused_gguf_dequant_kernel_used=_PIPE.fused_gguf_dequant_kernel_used(),
        # 2.3's third echo key, same name. The contract is the same three
        # values; this engine only ever emits two (see
        # ``Ltx25Pipeline.keep_resident_used``).
        keep_resident_used=_PIPE.keep_resident_used(),
        # 2.3's fourth echo key, same name and same three values. Read AFTER the
        # reset like the others, because the reset is what snapshots it -- and
        # computed from the pre-job degrade as well as the pipeline's record, so
        # a request that never reached the pipeline (no wheel) is still reported
        # as "sage->sdpa" rather than as a clean "sdpa".
        attention_used=_attention_used(attention, attention_degraded),
        # 台帳 §3-131: which VAE decoder this checkpoint built ("diff" / "conv").
        # A DIFFERENT vocabulary from 2.3's PrunaVAED echo above (that "off" /
        # "on" / "on->off" triad is whether the PRUNED decoder ran instead of
        # the stock one; this is the real decoder's own name) on purpose -- the
        # two answer different questions. Load-time, not per-job: it never
        # moves between jobs on one worker.
        vae_mode_used=_PIPE.video_vae_kind,
        # 台帳 §3-131: see the ``ltx25`` build above -- present on every job,
        # unlike ``extra``.
        ltx25=ltx25,
        # Appended LAST, and empty for every job but an outpaint one.
        **extra,
    )


#: Chain payload keys that name a feature this engine's chain does not have.
#:
#: REFUSED BY NAME, not ignored. Every one of them is already refused at the
#: endpoint (``services/engines/ltx25/adapter.py``'s ``CHAIN_REJECT_TABLE``), and
#: ``_RealBackend25.generate_chain`` builds its payload from a fixed literal --
#: plus the additive ``source`` / ``audio_source`` blocks below -- that contains
#: none of them, so an arrival here is not a stray field, it is the reject table
#: and the payload builder having drifted apart. Dropping it silently would hand
#: the user a video that quietly ignored the LoRA / the reference video / the
#: end source they asked for, which is the one failure this engine must never
#: produce.
#:
#: ``source`` (V2V) and ``audio_source`` (A2V) USED to be on this list and are
#: not any more: §3-102's second increment implemented both, so they are now
#: read below into :class:`~engine25.chain25.SourceSpec` /
#: :class:`~engine25.chain25.AudioSourceSpec`. ``loras`` and ``reference_video``
#: LEFT WITH THE THIRD, which implemented Style LoRA and the IC-LoRA reference
#: (the long-form one included: ONE reference video, sliced per stage-1 segment).
#: ``retake`` LEFT WITH THE RETAKE INCREMENT, which taught the chain to freeze
#: BOTH ends of a single window. What is left is what the engine still genuinely
#: does not have.
#:
#: The test is MEMBERSHIP, not truthiness: ``{"end_source": {}}`` is as much a sign
#: of drift as a populated block, and "the key was there but empty so we allowed
#: it" is exactly the kind of exception the single-rule principle exists to avoid.
#: ``fused_gguf_dequant_kernel`` LEFT WITH THE FUSED-KERNEL COMMIT: the chain
#: builds its transformer through the same GGUF loaders the single generate does,
#: so the Triton dequantization kernels apply to it unchanged and there is
#: nothing left to refuse. ``block_swap_prefetch`` LEFT WITH THE PREFETCH COMMIT
#: for the same reason, and the chain is where it matters most: it re-arms per
#: BUILD, and a chain builds the transformer once per stage-1 clip and once per
#: stage-2 tile. ``keep_resident`` LEFT WITH THE KEEP-RESIDENT COMMIT: it now
#: retains the text encoder's state dict between jobs, and a chain builds that
#: encoder exactly once per job like everything else does, so there is nothing
#: chain-shaped left to refuse. ``attention_backend`` LEFT WITH THE SAGE COMMIT,
#: and for the strongest version of that reason: the sage wrappers are installed
#: per transformer BUILD, and a chain is the path with the most builds (one per
#: stage-1 clip, one per stage-2 tile). Nothing about it is chain-shaped.
#:
#: The single-generate op differs deliberately for the knobs that remain here --
#: there they are ignored-and-logged (``IGNORED_FIELDS``) rather than refused,
#: because the app sends them on every single job.
#:
#: ``retake`` LEFT WITH THE RETAKE INCREMENT: the 2.5 chain freezes BOTH ends of
#: a single window now, so the block is READ below into
#: :class:`~engine25.chain25.RetakeSpec` rather than refused by name.
#: ``end_source`` LEFT WITH THE END-SOURCE INCREMENT for the same reason and it
#: was the last chain MODE on this list: the 2.5 chain runs the layout's own
#: stage-1 schedule and freezes the material's band at the timeline's tail, so
#: the block is READ below into :class:`~engine25.chain25.EndSourceSpec`.
#:
#: ``nag`` LEFT THIS TUPLE WITH THE NAG/VSF COMMIT, on exactly the argument
#: ``attention_backend`` left on: the patch is installed per transformer BUILD,
#: and a chain is the path with the MOST builds (one per stage-1 clip, one per
#: stage-2 tile). The negative prompt itself is encoded once per job by the
#: prompt encoder, which a chain calls once like everything else does. Nothing
#: about it is chain-shaped, so there was nothing chain-shaped left to refuse.
#: What remains is ONE engine-level field, and no mode of any kind.
CHAIN_UNSUPPORTED_KEYS = ("vae_mode",)


def _existing_media_path(raw: object, field: str, *, op: str = "generate_chain") -> str:
    """``str(raw)`` after checking it names a file that exists.

    2.3's worker does not check (:mod:`engine.worker`'s chain op reads the two
    blocks and hands the strings straight to the pipeline), and the app makes
    the check nearly moot -- it writes ``_source_tail.mp4`` immediately before
    the call and resolves the upload id for the audio. The check is here anyway
    because of what the FAILURE looks like without it: the path is first opened
    deep inside PyAV, and what the user is shown on the failed job is whatever
    the demuxer says about a name it could not open. One ``ValueError`` at the
    payload boundary names the field and the path instead, which is the same
    fail-loud regime the rest of this reader uses.

    A directory is rejected too (``is_file``, not ``exists``): a path that names
    a folder is the same mistake and would fail just as obscurely.

    ``op`` only names the op in the message. It exists because the reference
    video reaches this check from BOTH ops (see :func:`_resolve_ic_reference`),
    and a single generate whose reference path was mistyped should not be
    reported as a chain failure.
    """
    path = str(raw)
    if not Path(path).is_file():
        raise ValueError(f"{op}: {field} path is not an existing file: {path!r}")
    return path


def _do_generate_chain(msg: dict) -> None:
    """Run one masked AV-latent clip chain; ONE mp4 written to ``msg['output_path']``.

    Payload -> :class:`engine25.chain25.ChainSpec` -> ``run_chain``. The body
    keys are 2.3's chain op verbatim (``engine/worker.py``'s
    ``_do_generate_chain``): the two workers are unrelated code, but the body of
    a chain job is the same geometry in both, and the app's chain adapter builds
    one payload shape for whichever engine is loaded.

    The ``done`` event carries ``chain`` -- the full layout + metadata dict, in
    2.3's shape, so a client that already reads ``chain.total_px`` /
    ``chain.all_junctions`` needs no branch -- ALONGSIDE the same job-level keys
    the single path's ``done`` carries (``peak_vram_mb`` and friends), because
    the app stores those from one code path regardless of the job kind.
    """
    # BEFORE the load check: this is a ruling on the request, and a payload that
    # names a feature the engine does not have should say so whether or not a
    # model happens to be resident.
    refused = [key for key in CHAIN_UNSUPPORTED_KEYS if key in msg]
    if refused:
        raise RuntimeError(
            "engine25 worker: generate_chain received field(s) the LTX 2.5 chain does not "
            f"implement: {', '.join(refused)}. These are refused at the API "
            "(services/engines/ltx25/adapter.py CHAIN_REJECT_TABLE), so their arrival here "
            "means the reject table and the payload builder have drifted apart."
        )

    if _PIPE is None:
        raise RuntimeError("generate_chain before load")

    from engine25.chain25 import (  # noqa: PLC0415 -- deliberately lazy (see module docstring)
        AudioSourceSpec,
        ChainClipSpec,
        ChainSpec,
        EndSourceSpec,
        RetakeSpec,
        SourceSpec,
        run_chain,
    )
    from engine25.pipeline25 import image_conditionings  # noqa: PLC0415

    raw_clips = msg.get("clips") or []
    if not raw_clips:
        raise RuntimeError("engine25 worker: generate_chain needs a non-empty 'clips' list")

    seed = int(msg["seed"])
    clips = [
        ChainClipSpec(
            prompt=str(clip["prompt"]),
            num_frames=int(clip["num_frames"]),
            # Only clip 0 may carry them (the API schema enforces that); parsing
            # every clip's list anyway keeps this a payload reader rather than a
            # second place that knows the rule.
            images=image_conditionings(clip.get("images") or []),
        )
        for clip in raw_clips
    ]

    # V2V continuation: optional ``source``, an mp4 that is ALREADY the tail of
    # the user's video cut at the requested frame rate (the app's ``cut_tail_mp4``
    # runs before the engine is called; the engine checks the rate rather than
    # resampling). Read exactly as 2.3's chain op reads it -- same two keys, same
    # ``KeyError`` -> ``ValueError`` translation -- because the app builds ONE
    # chain payload shape for whichever engine is loaded.
    source = None
    src = msg.get("source")
    if src:
        try:
            source = SourceSpec(
                path=_existing_media_path(src["path"], "source"),
                context_frames=int(src["context_frames"]),
            )
        except KeyError as exc:
            raise ValueError(
                "generate_chain: source requires path and context_frames "
                f"(missing key {exc})"
            ) from exc

    # A2V: optional ``audio_source``, any media file with a decodable audio
    # stream. Mutually exclusive with ``source`` -- asserted in ``run_chain`` and
    # refused at the endpoint, so it is not re-stated here; this stays a payload
    # reader rather than a second place that knows the rule.
    #
    # ``if asrc:`` rather than ``is not None`` is 2.3's truthiness test kept
    # verbatim: an empty block is as much "no A2V" as an absent one, and the two
    # workers reading the same payload differently is exactly the drift the
    # refusal table above exists to catch.
    audio_source = None
    asrc = msg.get("audio_source")
    if asrc:
        try:
            audio_source = AudioSourceSpec(
                path=_existing_media_path(asrc["path"], "audio_source")
            )
        except KeyError as exc:
            raise ValueError(
                f"generate_chain: audio_source requires path (missing key {exc})"
            ) from exc

    # Retake (temporal inpainting): optional ``retake``, an mp4 that is ALREADY
    # the frame-exact, CFR window the app cut (``video_io.cut_window_mp4``), plus
    # the two glue widths in PIXEL frames. Mutually exclusive with ``source`` and
    # ``audio_source`` -- asserted in ``run_chain`` and refused at the endpoint,
    # so it is not re-stated here; this stays a payload reader.
    #
    # Read exactly as 2.3's chain op reads it (``engine/worker.py``): same four
    # keys, same truthiness test, same ``KeyError`` -> ``ValueError`` translation,
    # because the app builds ONE chain payload shape for whichever engine is
    # loaded. Absent -> byte-identical to before.
    retake = None
    rt = msg.get("retake")
    if rt:
        try:
            retake = RetakeSpec(
                path=_existing_media_path(rt["path"], "retake"),
                head_px=int(rt["head_px"]),
                tail_px=int(rt["tail_px"]),
                regenerate_audio=bool(rt["regenerate_audio"]),
            )
        except KeyError as exc:
            raise ValueError(
                "generate_chain: retake requires path, head_px, tail_px and "
                f"regenerate_audio (missing key {exc})"
            ) from exc

    # End source: optional ``end_source``, an mp4 the app has ALREADY cut to
    # ``context_frames + 1`` frames at the request fps (a still image was
    # looped into a video app-side, so the engine only ever sees a video —
    # ONE code path). The extra frame is the causal VAE's PRIMER: the encode
    # drops latent 0 so the remaining ones land on the timeline's own grid.
    # Mutually exclusive with ``retake`` and ``audio_source`` and COMBINABLE
    # with ``source`` (the one-clip interpolation the API allows) — asserted
    # in ``run_chain`` and refused at the endpoint, so it is not re-stated
    # here; this stays a payload reader.
    #
    # Read exactly as 2.3's chain op reads it (``engine/worker.py``): same
    # three keys, same truthiness test, same ``KeyError`` -> ``ValueError``
    # translation, and the same ``strength`` default of 1.0 for a payload
    # whose builder left it out. Absent -> byte-identical to before.
    end_source = None
    es = msg.get("end_source")
    if es:
        try:
            end_source = EndSourceSpec(
                path=_existing_media_path(es["path"], "end_source"),
                context_frames=int(es["context_frames"]),
                strength=float(es.get("strength", 1.0)),
            )
        except KeyError as exc:
            raise ValueError(
                "generate_chain: end_source requires path and context_frames "
                f"(missing key {exc})"
            ) from exc

    # Style/character LoRA + the ONE reference video, applied across the whole
    # chain and read with the SAME two helpers the single generate uses -- which
    # is the point: a reference must resolve (and preprocess) identically
    # whichever op it arrived on. The frame cap here is the chain's stage-1 pixel
    # TOTAL, not clip 0's length (see :func:`_preprocess_frame_cap`), because
    # every clip's stage-1 conditioning gets a slice of this reference.
    #
    # Unlike the single op these two are ADDITIVE in the payload (absent on a
    # chain that asked for neither), which is 2.3's chain shape verbatim: an
    # absent ``loras`` key gives ``[]``, and ``[]`` is still the explicit detach.
    ic_loras = _ic_loras(msg)
    ic_reference, ic_attn = _resolve_ic_reference(
        msg.get("reference_video"),
        str(msg["output_path"]),
        _preprocess_frame_cap(msg),
        op="generate_chain",
    )

    # The acceleration knobs, read with the SAME four helpers the single op
    # uses: one set of readers for the two entry points is what stops a knob from
    # being wired to one op and forgotten on the other. All are live on the
    # chain: it builds the transformer once per stage-1 clip and once per
    # stage-2 tile, and the stage re-arms the prefetch on every one of them.
    # ``keep_resident`` works differently but is just as live -- a chain builds
    # the TEXT encoder once per job, same as a single generate, so what it saves
    # is the same 7.7 GiB rebuild on the job after this one. ``attention_backend``
    # is live for the transformer reason: the stage strips and re-installs the
    # sage wrappers on EVERY one of those builds, so a chain's echo is true of
    # all of them or of none.
    prefetch = _resolve_block_swap_prefetch(msg)
    fused = _resolve_fused_dequant(msg)
    keep_resident = _resolve_keep_resident(msg)
    attention, attention_degraded = _resolve_attention(msg)
    # Same reader, same absent-means-off contract, as the single op's -- see
    # there. The chain is where the patch's per-build lifetime matters most.
    nag = _resolve_nag(msg)

    spec = ChainSpec(
        clips=clips,
        width=int(msg["width"]),
        height=int(msg["height"]),
        frame_rate=float(msg["frame_rate"]),
        seed=seed,
        overlap_frames=int(msg["overlap_frames"]),
        overlap_strength=float(msg["overlap_strength"]),
        output_path=str(msg["output_path"]),
        num_steps=int(msg.get("num_steps", 0)),
        chunked_upsample=bool(msg.get("chunked_upsample", False)),
        # Absent -> None -> chain_math's "standard" preset. The app sends the key
        # only when the request opted off standard, so the default payload and
        # the default geometry stay one thing.
        stage2_window=msg.get("stage2_window") or None,
        source=source,
        audio_source=audio_source,
    )

    _log(
        f"generate_chain {spec.width}x{spec.height} clips={len(clips)} "
        f"frames={[c.num_frames for c in clips]} seed={seed} "
        f"overlap={spec.overlap_frames}/{spec.overlap_strength} "
        f"chunked_upsample={spec.chunked_upsample} "
        # Same two fields, same spelling, as 2.3's chain log line: the two
        # engines' worker logs are read side by side when a chain is compared.
        f"source={'yes(ctx=' + str(source.context_frames) + ')' if source else 'no'} "
        f"audio_source={'yes' if audio_source else 'no'} "
        # The retake receipt, in the same line and the same spelling 2.3 uses:
        # the two glue widths and the audio ruling, as PARSED, before any of it
        # runs. ``regen`` is the one field that changes what is delivered (a
        # False re-muxes the window's own waveform and skips the vocoder), so it
        # is spelled out rather than folded into "yes".
        f"retake={'yes(head=' + str(retake.head_px) + ' tail=' + str(retake.tail_px)
                 + ' regen=' + ('on' if retake.regenerate_audio else 'off') + ')'
                 if retake else 'no'} "
        # The end source's parse receipt, in 2.3's spelling for the same
        # side-by-side reason: the band length as ASKED FOR (the file itself
        # carries one frame more) and the strength, which is the one field
        # that changes how hard stage 1 holds the video tail.
        f"end_source={'yes(ctx=' + str(end_source.context_frames)
                     + ' strength=' + format(end_source.strength, '.3f') + ')'
                     if end_source else 'no'} "
        # The parse receipt for the two §3-102 blocks, in the same line rather
        # than a second one: what the chain was ASKED for, before any of it runs.
        f"ic_loras={len(ic_loras)} ic_reference={'yes' if ic_reference else 'no'} "
        f"preprocess={_preprocess_kind(msg)} attn={ic_attn:.3f} "
        f"stage2win={spec.stage2_window or 'standard'} "
        # What was ASKED for; the done event's echo keys say what was GOT.
        f"fused={'on' if fused else 'off'} prefetch={'on' if prefetch else 'off'} "
        f"keep_resident={'on' if keep_resident else 'off'} "
        # RESOLVED, exactly as in the single op: an unavailable wheel has already
        # turned a 'sage' request into 'sdpa' by the time this line is written.
        f"attention={attention} "
        # off / nag / vsf, the single op's spelling and 2.3's.
        f"neg={_neg_label(nag)}"
    )

    # Armed OUTSIDE the try and before the first transformer build, disarmed in
    # the finally: the single op's discipline verbatim, and the reason it is
    # spelled out twice rather than factored into a helper is that the two ops'
    # bodies are what sits between the two calls. See ``_do_generate``.
    _PIPE.set_acceleration_job(
        block_swap_prefetch=prefetch,
        fused_gguf_dequant_kernel=fused,
        keep_resident=keep_resident,
        # The single op's line verbatim: stated because the parameter has no
        # default, and False until the request key is wired up (§3-114).
        keep_resident_embeddings=False,
        attention_backend=attention,
    )
    # The single op's discipline verbatim, for the same ordering reason: armed
    # outside the try, before the prompt encode, disarmed in the finally.
    _PIPE.set_nag_job(nag)
    try:
        result = run_chain(
            _PIPE,
            spec,
            _emit_progress,
            retake=retake,
            end_source=end_source,
            ic_loras=ic_loras,
            ic_reference=ic_reference,
            ic_attention_strength=ic_attn,
        )
        meta = result.metadata
        ltx25 = meta.get("ltx25") or {}
        vram = ltx25.get("vram") or {}

        # The chain's GENERATE_REPORT: one line, the phase ledger included, so a run
        # that was only ever watched through the worker log still says where its wall
        # clock and its VRAM peak went. The full metadata rides in the ``done`` event
        # rather than here -- this line is the human-readable summary.
        _log(
            "CHAIN_REPORT "
            + json.dumps(
                {
                    "output": result.output_path,
                    "n_clips": meta.get("n_clips"),
                    "n_tiles": meta.get("n_tiles"),
                    "total_px": meta.get("total_px"),
                    "seeds": meta.get("seeds"),
                    "wall_s": meta.get("wall_s"),
                    "vram_peak_mb": meta.get("vram_peak_mb"),
                    "stage1_sampler": ltx25.get("stage1_sampler"),
                    "chunked_upsample": ltx25.get("chunked_upsample"),
                    "stage2_window": ltx25.get("stage2_window"),
                    "vram": vram,
                    "phases": ltx25.get("phases"),
                },
                ensure_ascii=False,
                default=str,
            )
        )
    finally:
        # Two lines here, as in the single op, and in the same order: the
        # negative prompt goes first because it is the one holding tensors.
        _PIPE.reset_nag_job()
        _PIPE.reset_acceleration_job()

    _emit(
        "done",
        seed_used=seed,
        sampler=_PIPE.sampler,
        # Same key names and units (MiB) as the single path's done, from the
        # same per-phase peaks: the app reads these off one code path.
        peak_vram_mb=_gib_to_mb(vram.get("peak_allocated_gib")),
        peak_vram_reserved_mb=_gib_to_mb(vram.get("peak_reserved_gib")),
        rss_peak_gib=vram.get("rss_peak_gib"),
        seconds=meta.get("wall_s"),
        # The chain's frame count is the ASSEMBLED timeline, not any one clip's.
        num_frames=meta.get("total_px"),
        encode_fps=ltx25.get("encode_fps"),
        size_bytes=ltx25.get("size_bytes"),
        phases=ltx25.get("phases"),
        # 2.3's chain contract: the whole layout + metadata under one key.
        chain=meta,
        # The same echo keys, same values, as the single op's done: a chain runs
        # many builds, and the fused verdict is the whole JOB's ("on" only if
        # Triton really ran; "on->off" if any of them fell back).
        block_swap_prefetch_used=_PIPE.block_swap_prefetch_used(),
        fused_gguf_dequant_kernel_used=_PIPE.fused_gguf_dequant_kernel_used(),
        # 2.3's third echo key, same name. The contract is the same three
        # values; this engine only ever emits two (see
        # ``Ltx25Pipeline.keep_resident_used``).
        keep_resident_used=_PIPE.keep_resident_used(),
        # 2.3's fourth echo key. A chain's value covers the WHOLE job: the
        # kernel-failure latch lives on the pipeline's ``SageState``, which
        # outlives every one of the chain's transformer builds, so one failed
        # kernel call in the last stage-2 tile makes the whole chain
        # "sage->sdpa".
        attention_used=_attention_used(attention, attention_degraded),
        # 台帳 §3-131: same vocabulary and same load-time fact as the single
        # op's done -- see the comment there. No additive ``ltx25=`` key here:
        # ``chain=meta`` already carries ``meta["ltx25"]`` (built by
        # ``chain25.py``), and duplicating it at the top level would be two
        # copies of one dict to keep in sync instead of one.
        vae_mode_used=_PIPE.video_vae_kind,
    )


def _gib_to_mb(value: float | None) -> int | None:
    return None if value is None else int(round(value * 1024))


def _shutdown() -> None:
    """Best-effort free, then exit 0.

    torch is not imported at module scope, so the CUDA teardown the 2.3 worker
    performs is done only if torch actually came in along the way (it does, once
    ``load`` has run). Looking the module up in ``sys.modules`` rather than
    importing it keeps ``shutdown`` free of a multi-second import on a process
    that never touched the GPU.
    """
    global _PIPE
    if _PIPE is not None:
        try:
            _PIPE.close()
        except Exception as exc:  # noqa: BLE001 -- teardown is best effort
            _log(f"pipeline close failed (ignored): {exc!r}")
        _PIPE = None

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
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

        # Serving loop (post-load). An unsupported op is ANSWERED rather than
        # ignored, because the parent blocks waiting for a reply and silence
        # would read as a hang.
        if op == "shutdown":
            _shutdown()
        if op == "generate":
            try:
                _do_generate(msg)
            except BaseException as exc:  # noqa: BLE001
                # A failed job is reported and the process stays alive: the app
                # surfaces the detail on the job and the next request can still
                # be served (a VRAM shortfall on one geometry says nothing about
                # the next one). Only a failed LOAD is fatal.
                _log("GENERATE_FAILED")
                _emit("error", detail=_detail(exc))
            continue
        if op == "generate_chain":
            try:
                _do_generate_chain(msg)
            except BaseException as exc:  # noqa: BLE001
                # Same regime as ``generate``: a failed job is reported and the
                # process stays alive. Only a failed LOAD is fatal.
                _log("CHAIN_FAILED")
                _emit("error", detail=_detail(exc))
            continue
        _log(f"ignoring unknown op={op!r}")

    # EOF on stdin -> graceful shutdown.
    _shutdown()


# ---------------------------------------------------------------------------
# Selftest CLI (gate G4)
# ---------------------------------------------------------------------------


def _add_load_arguments(parser) -> None:
    """The model-path + load-knob arguments both selftests share.

    One definition, because a chain selftest that spelled ``--video-vae``
    differently from the single one would be a second contract to remember for
    no reason -- and because the two must build the SAME ``load`` message: that
    message is the thing under test in both.
    """
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--text-encoder-assets", default=None)
    parser.add_argument("--video-vae", required=True)
    parser.add_argument("--audio-vae", required=True)
    parser.add_argument("--spatial-upsampler", required=True)
    parser.add_argument("--blocks-on-gpu", type=int, default=None)
    parser.add_argument("--te-layers-on-gpu", type=int, default=None)
    parser.add_argument("--no-cache-weights", action="store_true")
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="leave cuDNN algorithm selection free (the audio vocoder then varies run to run)",
    )


def _load_message(args) -> dict:
    """Build the ``{"op":"load"}`` payload from the shared arguments."""
    load_msg = {
        "op": "load",
        "transformer_path": args.transformer,
        "text_encoder_path": args.text_encoder,
        "video_vae_path": args.video_vae,
        "audio_vae_path": args.audio_vae,
        "spatial_upsampler_path": args.spatial_upsampler,
    }
    if args.text_encoder_assets:
        load_msg["text_encoder_assets_path"] = args.text_encoder_assets
    if args.blocks_on_gpu is not None:
        load_msg["blocks_on_gpu"] = args.blocks_on_gpu
    if args.te_layers_on_gpu is not None:
        load_msg["te_layers_on_gpu"] = args.te_layers_on_gpu
    if args.no_cache_weights:
        load_msg["cache_weights"] = False
    if args.non_deterministic:
        load_msg["deterministic"] = False
    return load_msg


def _parse_image_arg(spec: str) -> dict:
    """``PATH[,FRAME_IDX[,STRENGTH]]`` -> one ``images`` entry."""
    parts = spec.split(",")
    return {
        "path": parts[0],
        "frame_idx": int(parts[1]) if len(parts) > 1 else 0,
        "strength": float(parts[2]) if len(parts) > 2 else 1.0,
    }


#: Where ``--lora NAME`` looks. Both engines share ONE adapter library (a LoRA
#: file has no base-model axis -- the 2.3 adapters are what 2.5 runs, which is
#: the whole finding §71 rests on), so this is the app's ``lora_dir`` and its
#: ``ic_loras`` config entries alike, resolved by filename rather than restated.
_LORA_SEARCH_ROOT = ROOT / "models" / "LTX23"


def _resolve_lora_path(name: str) -> str:
    """``NAME`` -> an existing adapter file, or fail loud.

    A path that already names a file wins outright; otherwise the name (with or
    without ``.safetensors``) is looked up under ``models/LTX23``. Searching by
    filename rather than taking a directory argument is what lets the selftest
    say ``--lora Pixar_Toon`` for a Style adapter and
    ``--lora ltx-2.3-22b-ic-lora-union-control-ref0.5`` for a control one without
    knowing which subdirectory each lives in -- the config's three logical
    control names all point at that ONE file, so the lookup stays unambiguous.

    Not-found and ambiguous both raise: a selftest that silently generated
    without the adapter it was told to use would be a green run that proves the
    opposite of what it claims.
    """
    direct = Path(name)
    if direct.is_file():
        return str(direct)
    stem = direct.name
    patterns = (stem, f"{stem}.safetensors")
    found = sorted(
        {p for pattern in patterns for p in _LORA_SEARCH_ROOT.rglob(pattern) if p.is_file()}
    )
    if not found:
        raise SystemExit(f"--lora {name!r}: no such adapter file under {_LORA_SEARCH_ROOT}")
    if len(found) > 1:
        raise SystemExit(
            f"--lora {name!r} matches {len(found)} files: {[str(p) for p in found]}"
        )
    return str(found[0])


def _parse_lora_arg(spec: str) -> dict:
    """``NAME[:STRENGTH[:AUDIO_STRENGTH]]`` -> one ``loras`` entry.

    The numeric suffixes are peeled off the RIGHT, and only while they parse as
    floats, so a Windows path (``S:\\models\\x.safetensors``) survives the split
    that its drive-letter colon would otherwise break.

    ``audio_strength`` is included ONLY when given, which is the app's payload
    shape (``_lora_payload_entry``): an absent key means the audio-side Linears
    follow the video strength, and sending ``null`` instead would be a different
    statement.
    """
    parts = spec.split(":")
    numbers: list[float] = []
    while len(parts) > 1 and len(numbers) < 2:
        try:
            value = float(parts[-1])
        except ValueError:
            break
        numbers.insert(0, value)
        parts.pop()
    entry = {
        "path": _resolve_lora_path(":".join(parts)),
        "strength": numbers[0] if numbers else 1.0,
    }
    if len(numbers) > 1:
        entry["audio_strength"] = numbers[1]
    return entry


def _add_ic_lora_arguments(parser) -> None:
    """The Style-LoRA / reference-video arguments BOTH selftests share.

    One definition for the same reason the load arguments have one: the two
    selftests must build the same blocks, because those blocks are the thing
    under test. The reference's two strengths are here (rather than left at their
    defaults) so the ``attention_strength`` branch -- the wrapper the engine
    applies only below 1.0 -- is reachable from the command line at all.
    """
    parser.add_argument(
        "--lora",
        action="append",
        default=[],
        help="Style/IC adapter (repeatable): NAME[:STRENGTH[:AUDIO_STRENGTH]]. NAME is a "
        "file under models/LTX23 (with or without .safetensors) or a path",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="IC-LoRA reference video. Needs a --lora that declares a reference downscale "
        "factor; %%128 width/height, as the API enforces for every reference job",
    )
    parser.add_argument("--reference-strength", type=float, default=1.0)
    parser.add_argument(
        "--preprocess",
        default="none",
        help="control-signal conversion for --reference: none (default) | canny | dwpose | "
        "depth. Anything but none writes control_<kind>.mp4 next to the output",
    )
    parser.add_argument(
        "--attention-strength",
        type=float,
        default=None,
        help="conditioning_attention_strength (0..1). Omitted -> the key is absent from the "
        "payload and the engine applies no wrapper at all",
    )


def _add_acceleration_arguments(parser) -> None:
    """The acceleration-knob arguments both selftests share.

    ``--fused-dequant on`` (the default) is what makes the selftest the thing
    that MEASURES the feature: the default has to be the accelerated path,
    because a knob whose selftest never turns it on is a knob nobody runs. The
    ``off`` side exists for the pair comparison the gate asks for -- same seed,
    same geometry, one bit different, digests compared.

    The four knobs have THREE different defaults between them, and each one is
    an argument rather than a convention -- see the comment above each.
    """
    parser.add_argument(
        "--fused-dequant",
        choices=("on", "off"),
        default="on",
        help="fused Triton GGUF dequantization kernels (default: on)",
    )
    parser.add_argument(
        "--block-swap-prefetch",
        choices=("on", "off"),
        default="on",
        help="asynchronous block-swap prefetching (default: on)",
    )
    # DEFAULT OFF -- the opposite direction from the two above, and not an
    # oversight. The other two are free (same output, less time), so their
    # selftest default is the accelerated path. This one BUYS TIME WITH RAM:
    # ~7.7 GiB stays resident between jobs. The app ships it off by default for
    # that reason, and a selftest whose default did not match would measure a
    # configuration nobody runs.
    parser.add_argument(
        "--keep-resident",
        choices=("on", "off"),
        default="off",
        help="retain the text encoder's state dict between jobs, ~7.7 GiB of resident RAM "
        "(default: off -- unlike the two knobs above, which default to on)",
    )
    # DEFAULT SDPA, and for a THIRD reason again. The first two knobs default to
    # on because they are free; keep-resident defaults to off because it buys
    # time with RAM. This one defaults to sdpa because it is the only knob that
    # CHANGES THE OUTPUT: sage is a quantized kernel, so a sage round's digest
    # cannot be compared against the frozen baseline the other gates use. The
    # app ships it off for the same reason (the default is a scope decision, not
    # a performance one), and a selftest whose default did not match would make
    # every digest comparison in this file a different measurement.
    parser.add_argument(
        "--attention",
        choices=("sdpa", "sage"),
        default="sdpa",
        help="attention kernel: sdpa (default) or sage (SageAttention -- faster, and "
        "quantized, so the mp4 differs from the sdpa one at the same seed)",
    )


def _acceleration_payload(args) -> dict:
    """The acceleration keys for a synthesised job message.

    ADDITIVE, exactly as the app's payload builder is: ``off`` omits the key
    entirely rather than sending ``False``, because absent-means-off is the
    contract the workers' readers implement and an ``off`` run has to exercise
    the same absent-key path a pre-acceleration payload would take. For
    ``keep_resident`` the omission is doubly the point: an absent key is the
    RELEASE instruction, so an ``off`` round on a warm worker is what proves the
    release path runs (see :func:`_resolve_keep_resident`).
    """
    payload: dict = {}
    if args.fused_dequant == "on":
        payload["fused_gguf_dequant_kernel"] = True
    if args.block_swap_prefetch == "on":
        payload["block_swap_prefetch"] = True
    if args.keep_resident == "on":
        payload["keep_resident"] = True
    if args.attention != "sdpa":
        # Additive like the three above: ``sdpa`` omits the key rather than
        # sending it, so an sdpa round exercises the same absent-key path a
        # pre-sage payload takes -- which is what makes its digest comparable to
        # the frozen baseline at all.
        payload["attention_backend"] = args.attention
    return payload


def _reference_payload(args) -> dict | None:
    """The ``reference_video`` block from the shared arguments (None = no reference)."""
    if args.reference is None:
        return None
    block: dict = {
        "path": args.reference,
        "strength": float(args.reference_strength),
        "preprocess": args.preprocess,
    }
    if args.attention_strength is not None:
        block["attention_strength"] = float(args.attention_strength)
    return block


def _selftest_generate(argv: list[str]) -> int:
    """Load + generate once (or twice) from the command line, then report JSON.

    Drives ``_do_load`` / ``_do_generate`` with synthesised protocol messages so
    the measured path is the shipped one. ``--rounds 2`` with an unchanged seed
    is the determinism probe: the two mp4 digests are compared and reported.

    ``--lora`` / ``--reference`` / ``--preprocess`` build the two §3-102 blocks
    the app sends. Both keys ride on EVERY message, ``[]``/``None`` included,
    which is what makes an unchanged digest on a no-LoRA run evidence that the
    explicit-detach path costs nothing rather than evidence that it never ran.
    """
    import argparse
    import hashlib
    import time

    parser = argparse.ArgumentParser(prog="engine25.worker --selftest-generate")
    _add_load_arguments(parser)
    parser.add_argument(
        "--output",
        required=True,
        help="mp4 path; with --rounds >1 the round number goes BEFORE the suffix (out.r1.mp4), "
        "because PyAV picks the container from the extension and would refuse 'out.mp4.1'",
    )
    parser.add_argument("--prompt", default="A calm sunlit kitchen, steam rising from a cup of tea.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--image", action="append", default=[], help="conditioning image (repeatable): PATH[,FRAME_IDX[,STRENGTH]]")
    _add_ic_lora_arguments(parser)
    _add_acceleration_arguments(parser)
    parser.add_argument("--report", default=None, help="write the JSON report here as well as to stdout")
    args = parser.parse_args(argv)

    load_msg = _load_message(args)
    images = [_parse_image_arg(spec) for spec in args.image]
    loras = [_parse_lora_arg(spec) for spec in args.lora]
    reference = _reference_payload(args)
    acceleration = _acceleration_payload(args)

    # Every framed event, captured on its way to stdout, exactly as the chain
    # selftest does it. Added with the acceleration knobs: this selftest used to
    # print its ``done`` event and keep nothing, so the echo keys -- the ONLY
    # statement of what the job actually got, as opposed to what it asked for --
    # were unavailable to anything reading the report. Patching the module global
    # is what leaves the handler itself untouched: the run under observation is
    # the shipped one, and the lines still reach stdout.
    events: list[dict] = []
    real_emit = _emit

    def _capturing_emit(event: str, **fields: object) -> None:
        events.append({"event": event, **fields})
        real_emit(event, **fields)

    globals()["_emit"] = _capturing_emit
    try:
        report: dict = {
            "load": load_msg,
            "images": images,
            # The RESOLVED blocks, not the raw arguments: which file a bare
            # ``--lora Pixar_Toon`` turned into is exactly what a report of a
            # LoRA run has to state.
            "loras": loras,
            "reference_video": reference,
            # What was ASKED for. Every round's ``done`` below says what was got.
            "acceleration": acceleration,
            "rounds": [],
        }
        started = time.perf_counter()
        _do_load(load_msg)
        report["load_seconds"] = round(time.perf_counter() - started, 2)
        assert _PIPE is not None
        report["build"] = _PIPE.build_report
        report["sampler"] = _PIPE.sampler

        digests = []
        for index in range(1, max(1, args.rounds) + 1):
            base = Path(args.output)
            out = str(base if args.rounds <= 1 else base.with_name(f"{base.stem}.r{index}{base.suffix}"))
            events.clear()
            _do_generate(
                {
                    "op": "generate",
                    "prompt": args.prompt,
                    "seed": args.seed,
                    "width": args.width,
                    "height": args.height,
                    "num_frames": args.num_frames,
                    "frame_rate": args.frame_rate,
                    "output_path": out,
                    "images": images,
                    # ALWAYS present, ``[]``/``None`` included -- the app's single
                    # generate sends both keys on every job, and an empty list is
                    # the explicit detach the worker must be told about.
                    "loras": loras,
                    "reference_video": reference,
                    # Deliberately present: proves the ignore-and-log path runs on
                    # the same messages the app will send.
                    "num_steps": 8,
                    "negative_prompt": "",
                    # Additive, exactly as the app builds it: absent entirely when
                    # --fused-dequant off.
                    **acceleration,
                }
            )
            digest = hashlib.sha256(Path(out).read_bytes()).hexdigest()
            digests.append(digest)
            done = next((e for e in events if e["event"] == "done"), None)
            if done is None:
                raise RuntimeError("the generation produced no terminal done event")
            round_report = {"round": index, "output": out, "sha256": digest,
                            "size_bytes": Path(out).stat().st_size,
                            # The done event VERBATIM, which is where the two
                            # acceleration echoes live. The chain selftest has
                            # always reported this; the single one now does too.
                            "done": done}
            # The control mp4 by MEASUREMENT: a preprocess kind that silently wrote
            # nothing would otherwise look like a clean run.
            if reference is not None and reference["preprocess"] != "none":
                control = Path(out).with_name(f"control_{reference['preprocess']}.mp4")
                round_report["control_video"] = {
                    "path": str(control),
                    "exists": control.is_file(),
                    "size_bytes": control.stat().st_size if control.is_file() else None,
                }
            report["rounds"].append(round_report)

        if len(digests) > 1:
            report["same_seed_sha_identical"] = len(set(digests)) == 1
    finally:
        globals()["_emit"] = real_emit

    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0


#: The chain selftest's default per-clip prompts. Two DISTINCT prompts, because
#: "one text encode per distinct prompt" is part of what the chain path does and
#: two identical ones would exercise only the dedup branch.
_SELFTEST_CHAIN_PROMPTS = (
    "A calm sunlit kitchen, steam rising from a cup of tea on a wooden table.",
    "The same kitchen as the afternoon light fades, the steam thinning to nothing.",
)


def _selftest_chain(argv: list[str]) -> int:
    """Load + run one clip chain from the command line, then report JSON (gate G2).

    Drives ``_do_load`` / :func:`_do_generate_chain` with synthesised protocol
    messages -- the SAME handlers the ``@@LTX@@`` protocol dispatches to -- so a
    green run is evidence about the shipped path. The framed events are captured
    on their way out, which is what makes the RECEIPTS checkable too: the report
    carries the ``done`` event verbatim (its ``chain`` block is the layout +
    metadata a client reads) and the whole ``progress`` series with its
    ``outer_index`` / ``outer_total`` positions, so "does a chain say which clip
    it is on" is answered by the run rather than by reading the code.

    ``--source`` (V2V) and ``--audio-source`` (A2V) add the one payload block
    each mode rides on, and are mutually exclusive here for the same reason they
    are at the endpoint: a chain is either a continuation of a video or a
    rendering of an audio track. Neither given, the payload's key set is
    byte-identical to the one that shipped before the two modes existed, which
    is what makes an unchanged digest evidence rather than a coincidence.

    ``--lora`` / ``--reference`` / ``--preprocess`` are the single selftest's,
    with the chain's ADDITIVE payload discipline: neither key is written unless
    asked for, so a plain chain's message is the dict it always was.

    ``stage2_window`` is deliberately not an argument: the default chain is what
    is under test, and the app sends that key only when a request opted off
    "standard". ``--rounds 2`` at an unchanged seed is the determinism probe.
    """
    import argparse
    import hashlib
    import time

    parser = argparse.ArgumentParser(prog="engine25.worker --selftest-chain")
    _add_load_arguments(parser)
    parser.add_argument(
        "--output",
        required=True,
        help="mp4 path; with --rounds >1 the round number goes BEFORE the suffix (out.r1.mp4), "
        "because PyAV picks the container from the extension and would refuse 'out.mp4.1'",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="per-clip prompt (repeatable); the default is two distinct prompts",
    )
    parser.add_argument("--clips", type=int, default=2, help="number of clips (default 2)")
    parser.add_argument("--num-frames", type=int, default=25, help="pixel frames PER CLIP")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--overlap-frames", type=int, default=2, help="K_v, video latent frames")
    parser.add_argument("--overlap-strength", type=float, default=0.5)
    chunked = parser.add_mutually_exclusive_group()
    chunked.add_argument("--chunked-upsample", dest="chunked_upsample", action="store_true")
    chunked.add_argument("--no-chunked-upsample", dest="chunked_upsample", action="store_false")
    parser.set_defaults(chunked_upsample=False)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="clip-0 conditioning image (repeatable): PATH[,FRAME_IDX[,STRENGTH]]",
    )
    # One mode or the other, never both -- argparse says so here, ``run_chain``
    # asserts it, and the API refuses it. Three layers because the two are not
    # merely unsupported together, they contradict each other.
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument(
        "--source",
        default=None,
        help="V2V: an mp4 that is ALREADY the fps-correct TAIL of the source video "
        "(what the app's cut_tail_mp4 writes). Its first --context frames are frozen "
        "as the head of clip 0 and trimmed back off before delivery",
    )
    origin.add_argument(
        "--audio-source",
        dest="audio_source",
        default=None,
        help="A2V: any media file with a decodable audio stream; its waveform is frozen "
        "over the whole timeline and muxed into the output verbatim",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=73,
        help="V2V context span in PIXEL frames (8n+1); read only with --source. "
        "The app's default is 73",
    )
    _add_ic_lora_arguments(parser)
    _add_acceleration_arguments(parser)
    parser.add_argument("--report", default=None, help="write the JSON report here as well as to stdout")
    args = parser.parse_args(argv)

    from chain_math import px_from_v_latent, v_latent_frames  # noqa: PLC0415

    load_msg = _load_message(args)
    images = [_parse_image_arg(spec) for spec in args.image]
    loras = [_parse_lora_arg(spec) for spec in args.lora]
    reference = _reference_payload(args)
    acceleration = _acceleration_payload(args)

    prompts = list(args.prompt) or list(_SELFTEST_CHAIN_PROMPTS)
    if len(prompts) < args.clips:
        # Cycle rather than repeat the last one: a chain whose clips all share a
        # prompt is a different (easier) job than one whose clips differ.
        prompts = [prompts[i % len(prompts)] for i in range(args.clips)]
    prompts = prompts[: args.clips]

    clip_frames = [args.num_frames] * args.clips
    expected_f_total = sum(v_latent_frames(f) for f in clip_frames) - args.overlap_frames * (
        args.clips - 1
    )

    # Every framed event, captured on its way to stdout. Patching the module
    # global is what leaves the handlers themselves untouched -- the run under
    # observation is the shipped one, and the lines still reach stdout.
    events: list[dict] = []
    real_emit = _emit

    def _capturing_emit(event: str, **fields: object) -> None:
        events.append({"event": event, **fields})
        real_emit(event, **fields)

    globals()["_emit"] = _capturing_emit
    try:
        report: dict = {
            "gate": "G2",
            "load": load_msg,
            "geometry": {
                "clips": clip_frames,
                "prompts": prompts,
                "width": args.width,
                "height": args.height,
                "frame_rate": args.frame_rate,
                "seed": args.seed,
                "overlap_frames": args.overlap_frames,
                "overlap_strength": args.overlap_strength,
                "chunked_upsample": args.chunked_upsample,
                "stage2_window": "standard (the key is omitted from the payload)",
                "images": images,
                # None/None is the plain-chain case whose digest gate G2(a)
                # compares against the pre-V2V baseline.
                "source": (
                    None
                    if args.source is None
                    else {"path": args.source, "context_frames": args.context}
                ),
                "audio_source": (
                    None if args.audio_source is None else {"path": args.audio_source}
                ),
                # The RESOLVED blocks (which file a bare --lora NAME became),
                # both None/[] on the plain chain whose digest gate G2(a)
                # compares against the pre-LoRA baseline.
                "loras": loras,
                "reference_video": reference,
                # What was ASKED for. Every round's ``done`` says what was got --
                # and a chain gets it across MANY transformer builds, which is
                # the thing this pair of numbers is here to show.
                "acceleration": acceleration,
            },
            "rounds": [],
        }

        started = time.perf_counter()
        _do_load(load_msg)
        report["load_seconds"] = round(time.perf_counter() - started, 2)
        assert _PIPE is not None
        report["build"] = _PIPE.build_report
        report["sampler"] = _PIPE.sampler

        digests = []
        for index in range(1, max(1, args.rounds) + 1):
            base = Path(args.output)
            out = str(base if args.rounds <= 1 else base.with_name(f"{base.stem}.r{index}{base.suffix}"))
            events.clear()
            round_started = time.perf_counter()
            payload: dict = {
                "op": "generate_chain",
                "width": args.width,
                "height": args.height,
                "frame_rate": args.frame_rate,
                "num_steps": 8,
                "seed": args.seed,
                "overlap_frames": args.overlap_frames,
                "overlap_strength": args.overlap_strength,
                "chunked_upsample": args.chunked_upsample,
                "output_path": out,
                "clips": [
                    {
                        "prompt": prompts[i],
                        "num_frames": args.num_frames,
                        "images": images if i == 0 else [],
                    }
                    for i in range(args.clips)
                ],
            }
            # ADDITIVE, exactly as the adapter builds them: absent unless asked
            # for, so the plain chain's payload is the same dict it always was.
            if args.source is not None:
                payload["source"] = {"path": args.source, "context_frames": args.context}
            if args.audio_source is not None:
                payload["audio_source"] = {"path": args.audio_source}
            # Additive here too, exactly as the chain adapter builds them: a
            # chain that asked for neither carries neither key.
            if loras:
                payload["loras"] = loras
            if reference is not None:
                payload["reference_video"] = reference
            # Additive here too: absent entirely when --fused-dequant off, which
            # is the same absent-key path a pre-acceleration payload took.
            payload.update(acceleration)
            _do_generate_chain(payload)
            seconds = time.perf_counter() - round_started

            done = next((e for e in events if e["event"] == "done"), None)
            if done is None:
                raise RuntimeError("the chain produced no terminal done event")
            chain_meta = done.get("chain") or {}
            progress = [e for e in events if e["event"] == "progress"]

            # Per-stage receipts: how many, and which outer positions were named.
            by_stage: dict[str, dict] = {}
            for event in progress:
                entry = by_stage.setdefault(str(event.get("stage")), {"count": 0, "outer": []})
                entry["count"] += 1
                pair = [event.get("outer_index"), event.get("outer_total")]
                if pair[1] is not None and pair not in entry["outer"]:
                    entry["outer"].append(pair)

            digest = hashlib.sha256(Path(out).read_bytes()).hexdigest()
            digests.append(digest)

            # The V2V audio HANDLE: the full, untrimmed timeline audio the
            # engine writes next to the mp4 for a later Join. Reported by
            # measurement (does the file exist, how big is it) rather than by
            # repeating the filename the metadata already states.
            v2v_meta = chain_meta.get("v2v") or {}
            handle = None
            handle_name = v2v_meta.get("audio_handle_filename")
            if handle_name:
                handle_path = Path(out).with_name(str(handle_name))
                exists = handle_path.is_file()
                handle = {
                    "path": str(handle_path),
                    "exists": exists,
                    "size_bytes": handle_path.stat().st_size if exists else None,
                }

            # Where a control-signal reference would have written its mp4 (next
            # to the output, as _resolve_ic_reference puts it).
            control_path = Path(out).with_name(
                f"control_{'none' if reference is None else reference['preprocess']}.mp4"
            )
            report["rounds"].append(
                {
                    "round": index,
                    "output": out,
                    "sha256": digest,
                    "size_bytes": Path(out).stat().st_size,
                    "seconds": round(seconds, 2),
                    # The done event VERBATIM: its chain block, its VRAM peaks
                    # and the job-level keys the app reads, exactly as framed.
                    "done": done,
                    # Independent arithmetic through the SAME shared pure module
                    # the engine used, so "total_px agrees with the layout" is a
                    # cross-check rather than a restatement of one number.
                    "layout_check": {
                        "total_px": chain_meta.get("total_px"),
                        "f_total_latent": chain_meta.get("f_total_latent"),
                        "seg_latent": chain_meta.get("seg_latent"),
                        "n_tiles": chain_meta.get("n_tiles"),
                        "all_junctions": chain_meta.get("all_junctions"),
                        "expected_f_total": expected_f_total,
                        "expected_total_px": px_from_v_latent(expected_f_total),
                    },
                    # The two mode blocks VERBATIM (``None`` on a plain chain,
                    # which is itself the check that a plain chain grew no new
                    # metadata): 16 keys for V2V, 8 for A2V, same names as 2.3's.
                    "v2v": chain_meta.get("v2v"),
                    "a2v": chain_meta.get("a2v"),
                    "v2v_key_count": len(v2v_meta) or None,
                    "a2v_key_count": len(chain_meta.get("a2v") or {}) or None,
                    "audio_handle": handle,
                    # Same measurement as the single selftest's: a preprocess
                    # kind that wrote nothing must not read as a clean run.
                    "control_video": (
                        None
                        if reference is None or reference["preprocess"] == "none"
                        else {
                            "path": str(control_path),
                            "exists": control_path.is_file(),
                            "size_bytes": (
                                control_path.stat().st_size
                                if control_path.is_file()
                                else None
                            ),
                        }
                    ),
                    "progress": {
                        "count": len(progress),
                        "with_outer": sum(1 for e in progress if e.get("outer_total") is not None),
                        "by_stage": by_stage,
                        "events": progress,
                    },
                }
            )

        if len(digests) > 1:
            report["same_seed_sha_identical"] = len(set(digests)) == 1
    finally:
        globals()["_emit"] = real_emit

    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    if "--selftest-generate" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--selftest-generate"]
        raise SystemExit(_selftest_generate(rest))
    if "--selftest-chain" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--selftest-chain"]
        raise SystemExit(_selftest_chain(rest))
    main()
