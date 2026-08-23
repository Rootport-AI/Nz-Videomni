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
    ``empty_cache`` monkeypatch, the ``progress_shim`` tqdm swap, the
    SageAttention probe, the CUDA pre-warm -- are absent, and stay absent. Each
    is tuned to the 2.3 wheel's internals; re-applying them blind to a different
    pipeline is exactly the kind of borrowed-assumption bug this separate engine
    exists to avoid. Per-step progress in particular needs no shim here: the
    official denoising loops call their ``denoiser`` once per step, so
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
   output_path, [images]}
      One two-stage generation, mp4 written by this process to ``output_path``.
      ``images`` empty/absent -> T2V; entries -> I2V. Fields the v1 contract
      ignores (negative_prompt, num_steps, the 2.3 acceleration knobs, ...) may
      ride along; each is logged as ignored and dropped. ``crop_output`` is NOT
      one of them -- it never reaches the worker in either engine, because it is
      an ffmpeg post-process the app applies to the finished mp4.
  {"op": "generate_chain", output_path, seed, clips, width, height, frame_rate,
   num_steps, overlap_frames, overlap_strength, [chunked_upsample],
   [stage2_window], [source], [audio_source]}
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
      Unlike ``generate``, a field naming a feature this chain does not have
      (``retake``, ``end_source``, ``loras``, ``reference_video``, ``nag``,
      ``attention_backend``, ``keep_resident``, ``vae_mode``,
      ``block_swap_prefetch``, ``fused_gguf_dequant_kernel``) is REFUSED BY NAME
      rather than ignored -- see ``CHAIN_UNSUPPORTED_KEYS``.
      The ``done`` reply adds ``chain``: the whole layout + metadata dict, in
      2.3's shape -- including its ``v2v`` / ``a2v`` blocks when those modes ran.
  {"op": "shutdown"}

Replies are framed with the SAME unique prefix as the 2.3 worker so the shared
parent-side reader needs no branch, and so library/tqdm stdout noise stays
ignorable. Every protocol line: @@LTX@@<compact-json>, flushed. All other
logging goes to STDERR.
  @@LTX@@{"event":"ready","sampler":"euler_ancestral","sage_available":false}
  @@LTX@@{"event":"progress","stage":"stage1_denoise","index":3,"total":8}
  @@LTX@@{"event":"progress","stage":"stage1_denoise","index":3,"total":8,
          "outer_index":1,"outer_total":2}   <- chain only: which clip/tile
  @@LTX@@{"event":"done","seed_used":...,"peak_vram_mb":...,...}
  @@LTX@@{"event":"error","detail":...}

``progress`` stage names are the 2.3 worker's names on purpose (``encode`` /
``stage1_denoise`` / ``stage2_denoise`` / ``decode``): the app-side receipt loop
already maps them to labels and job fractions, so a second vocabulary would buy
nothing and cost a branch.

``ready.sampler`` is the sampler this process actually holds. It carries the
resolved name, which is how the ``use_ancestral_sampler`` assertion becomes
observable to the app instead of living only in a log line -- the official code
silently downgrades that flag to False on GGUF paths and only WARNs.

``ready.sage_available`` is always false for this engine and is expected to stay
false: LTX 2.5 v1 is SDPA-only by scope decision, and sageattention is not even
installed in .venv-engine-ltx25. The field is present anyway because the app
publishes it as ``acceleration.sage_available`` on GET /status for whichever
engine is loaded -- an absent key would read as "unknown", which is wrong.

Selftest (gate G4), run inside the venv without the app::

    python -m engine25.worker --selftest-generate \\
        --transformer <t.gguf> --text-encoder <te.gguf> \\
        --video-vae <conv.safetensors> --audio-vae <audio.safetensors> \\
        --spatial-upsampler <x2.safetensors> \\
        --output out.mp4 --width 320 --height 192 --num-frames 25 [--rounds 2]

It drives the SAME ``_do_load`` / ``_do_generate`` handlers the protocol uses --
it builds the JSON messages and feeds them in -- so a green selftest is evidence
about the shipped path, not about a parallel one.

The chain has its own (gate G2), same discipline, plus captured receipts::

    python -m engine25.worker --selftest-chain \\
        <the same five model paths> \\
        --output chain.mp4 --clips 2 --num-frames 25 \\
        --width 320 --height 192 [--chunked-upsample] [--rounds 2] \\
        [--source tail.mp4 --context 25 | --audio-source track.wav]

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
        _log("load: already loaded -- re-answering ready")
        _emit("ready", sampler=_PIPE.sampler, sage_available=False)
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
    # sage_available=False is permanent for this engine (see the module docstring).
    _emit("ready", sampler=pipeline.sampler, sage_available=False)


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

    result = _PIPE.generate(
        prompt=str(msg["prompt"]),
        seed=seed,
        width=int(msg["width"]),
        height=int(msg["height"]),
        num_frames=int(msg["num_frames"]),
        frame_rate=float(msg["frame_rate"]),
        output_path=str(msg["output_path"]),
        images=images,
        ignored=ignored,
    )

    report = result.as_dict()
    _log(f"GENERATE_REPORT {json.dumps(report, ensure_ascii=False, default=str)}")
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
#: retake window they asked for, which is the one failure this engine must never
#: produce.
#:
#: ``source`` (V2V) and ``audio_source`` (A2V) USED to be on this list and are
#: not any more: §3-102's second increment implemented both, so they are now
#: read below into :class:`~engine25.chain25.SourceSpec` /
#: :class:`~engine25.chain25.AudioSourceSpec`. What is left is what the engine
#: still genuinely does not have.
#:
#: The test is MEMBERSHIP, not truthiness: ``{"loras": []}`` is as much a sign of
#: drift as a populated list, and "the key was there but empty so we allowed it"
#: is exactly the kind of exception the single-rule principle exists to avoid.
#: The single-generate op differs deliberately -- there the acceleration knobs
#: ARE part of the contract and are ignored-and-logged (``IGNORED_FIELDS``),
#: because the app sends them on every single job.
CHAIN_UNSUPPORTED_KEYS = (
    "retake",
    "end_source",
    "loras",
    "reference_video",
    "nag",
    "attention_backend",
    "keep_resident",
    "vae_mode",
    "block_swap_prefetch",
    "fused_gguf_dequant_kernel",
)


def _existing_media_path(raw: object, field: str) -> str:
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
    """
    path = str(raw)
    if not Path(path).is_file():
        raise ValueError(
            f"generate_chain: {field} path is not an existing file: {path!r}"
        )
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
        f"stage2win={spec.stage2_window or 'standard'}"
    )

    result = run_chain(_PIPE, spec, _emit_progress)
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


def _selftest_generate(argv: list[str]) -> int:
    """Load + generate once (or twice) from the command line, then report JSON.

    Drives ``_do_load`` / ``_do_generate`` with synthesised protocol messages so
    the measured path is the shipped one. ``--rounds 2`` with an unchanged seed
    is the determinism probe: the two mp4 digests are compared and reported.
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
    parser.add_argument("--report", default=None, help="write the JSON report here as well as to stdout")
    args = parser.parse_args(argv)

    load_msg = _load_message(args)
    images = [_parse_image_arg(spec) for spec in args.image]

    report: dict = {"load": load_msg, "images": images, "rounds": []}
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
                # Deliberately present: proves the ignore-and-log path runs on
                # the same messages the app will send.
                "num_steps": 8,
                "negative_prompt": "",
            }
        )
        digest = hashlib.sha256(Path(out).read_bytes()).hexdigest()
        digests.append(digest)
        report["rounds"].append({"round": index, "output": out, "sha256": digest,
                                 "size_bytes": Path(out).stat().st_size})

    if len(digests) > 1:
        report["same_seed_sha_identical"] = len(set(digests)) == 1

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
    parser.add_argument("--report", default=None, help="write the JSON report here as well as to stdout")
    args = parser.parse_args(argv)

    from chain_math import px_from_v_latent, v_latent_frames  # noqa: PLC0415

    load_msg = _load_message(args)
    images = [_parse_image_arg(spec) for spec in args.image]

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
