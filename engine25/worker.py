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
  {"op": "shutdown"}

Replies are framed with the SAME unique prefix as the 2.3 worker so the shared
parent-side reader needs no branch, and so library/tqdm stdout noise stays
ignorable. Every protocol line: @@LTX@@<compact-json>, flushed. All other
logging goes to STDERR.
  @@LTX@@{"event":"ready","sampler":"euler_ancestral","sage_available":false}
  @@LTX@@{"event":"progress","stage":"stage1_denoise","index":3,"total":8}
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


def _emit_progress(stage: str, index: int, total: int) -> None:
    """One framed ``progress`` line. Never allowed to fail a job."""
    try:
        _emit("progress", stage=stage, index=int(index), total=int(total))
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
            # Chain / retake / end-source are out of scope for LTX 2.5 v1 and
            # are rejected with a 422 at the API long before here; this is the
            # backstop for a payload that arrived another way.
            _log("op='generate_chain' is not supported by the LTX 2.5 engine (v1)")
            _emit(
                "error",
                detail=(
                    "engine25 worker: generate_chain is not supported by the LTX 2.5 engine "
                    "(v1 scope: single two-stage T2V/I2V generation)"
                ),
            )
            continue
        _log(f"ignoring unknown op={op!r}")

    # EOF on stdin -> graceful shutdown.
    _shutdown()


# ---------------------------------------------------------------------------
# Selftest CLI (gate G4)
# ---------------------------------------------------------------------------


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
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--text-encoder-assets", default=None)
    parser.add_argument("--video-vae", required=True)
    parser.add_argument("--audio-vae", required=True)
    parser.add_argument("--spatial-upsampler", required=True)
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
    parser.add_argument("--blocks-on-gpu", type=int, default=None)
    parser.add_argument("--te-layers-on-gpu", type=int, default=None)
    parser.add_argument("--no-cache-weights", action="store_true")
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="leave cuDNN algorithm selection free (the audio vocoder then varies run to run)",
    )
    parser.add_argument("--image", action="append", default=[], help="conditioning image (repeatable): PATH[,FRAME_IDX[,STRENGTH]]")
    parser.add_argument("--report", default=None, help="write the JSON report here as well as to stdout")
    args = parser.parse_args(argv)

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

    images = []
    for spec in args.image:
        parts = spec.split(",")
        images.append(
            {
                "path": parts[0],
                "frame_idx": int(parts[1]) if len(parts) > 1 else 0,
                "strength": float(parts[2]) if len(parts) > 2 else 1.0,
            }
        )

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


if __name__ == "__main__":
    if "--selftest-generate" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--selftest-generate"]
        raise SystemExit(_selftest_generate(rest))
    main()
