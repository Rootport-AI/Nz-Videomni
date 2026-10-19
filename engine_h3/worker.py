"""Persistent MiniMax H3 generation worker.

Runs inside ``.venv-engine-h3`` -- the environment with torch cu130,
comfy-kitchen (the int8 kernels), transformers 5 (Qwen3-VL) and PyAV (the exact
versions are engine_h3/venv-engine-h3.freeze.txt). It
is a SIBLING of ``engine/worker.py`` (LTX 2.3) and ``engine25/worker.py``
(LTX 2.5), never a replacement: each engine keeps its own venv and none of the
three processes can import another's stack.

The protocol is the one the LTX workers speak, so the app-side plumbing
(``@@LTX@@`` line reader, progress fractions, the error path) is shared. Only
small control JSON crosses the pipe; the mp4 is written by this process to
``output_path``. Nothing LTX-specific is carried over: no IC-LoRA / reference
video / chain / NAG / VSF branches. ``keep_resident`` follows the LTX contract
(kept while the key is true, released by a job without it).

Bootstrap (mirrors engine25/worker.py):
  * TORCH_COMPILE_DISABLE=1 set before anything can import torch.
  * chdir(ROOT) + sys.path.insert(0, ROOT), so ``engine_h3.*``, ``engine.*`` and
    the root modules (``h3_format``, ``sft_quant_format``) import whether
    launched as ``python -m engine_h3.worker`` or as a bare script path.
  * torch is NOT imported at module scope; the heavy imports live inside
    ``load``, so a process that only answers ``shutdown`` never pays for them
    and a typo'd model path is reported in under a second.

Protocol (one JSON object per line; parent -> worker):
  {"op": "load", transformer_path, text_encoder_path, video_vae_path,
   audio_vae_path, tokenizer_dir, [blocks_on_gpu],
   [attention_backend], [block_swap_prefetch], [deterministic]}
      Paths are existence-checked first, then the headers are inspected
      (``h3_format``) and comfy-kitchen's CUDA int8 path is pinned. No weight is
      read at load: every job builds each stage from the files and frees it
      (encode -> denoise -> decode), except what a ``keep_resident`` job kept
      (see ``generate``). A failure is fatal: ``error`` + exit 1.
      An optional key that is absent or null takes its default.
      ``blocks_on_gpu`` <= 0 (or >= 50) = the whole DiT on the GPU, no swap
      (see ``engine_h3.pipeline_h3.H3LoadOptions``). The text encoder has no
      knob: it is read layer by layer from the file onto the GPU.
      ``attention_backend`` / ``block_swap_prefetch`` here are the defaults of
      the jobs that do not send their own.
  {"op": "generate", prompt, seed, width, height, num_frames, frame_rate,
   [num_inference_steps], [shift_video], [shift_audio], output_path,
   [images], [keep_resident], [attention_backend], [block_swap_prefetch],
   [vram16_mode]}
      One T2V or I2V job. Omitted steps / shifts take the engine defaults
      (``engine_h3.constants``). ``images``: [] (T2V) or ``{"path", "frame_idx"}``
      key frames on distinct frames 0 .. num_frames-1 (any order); anything
      else is refused with a reason. Keys this
      engine does not read (top level, or inside an image entry) are logged
      once per job and ignored. ``keep_resident`` true keeps the DiT, the video
      decoder and the audio VAE on the CPU for the next job; a job without it
      releases them. ``vram16_mode`` true runs the denoise in VRAM 16GB mode
      (``engine_h3.lowvram``; absent / null = off here, the app sends it every
      job). A failure is reported with ``error`` and the process stays alive.
  {"op": "shutdown"}
  Before load an unknown op is only logged (an ``error`` there would be read
  as the reply to the next ``load``); after load it is answered with ``error``.

Events (worker -> parent; STDOUT, ``@@LTX@@`` + compact JSON):
  ready     {sampler, sage_available}
  progress  {stage, index, total}   stage: encode / stage1_denoise / decode
            (there is no stage2: the app adapter, services/engines/h3/adapter.py,
            reads these with ``_read_worker_events(one_stage=True)``, as Alpha
            Gen does; ``encode`` counts 4 steps when the job has any key
            frame, 3 otherwise)
  done      {seed_used, sampler, num_inference_steps, shift_video, shift_audio,
             peak_vram_mb, peak_vram_reserved_mb, rss_peak_gib, commit_gib,
             commit_peak_gib, seconds,
             num_frames, frame_rate, width, height, size_bytes, attention_used,
             block_swap_prefetch_used, keep_resident_used, vram16_mode_used,
             images_used, h3: {phases, quant_backends}}
  error     {detail}
STDERR carries the human log, prefixed ``[h3_worker]``.

Selftest:
  python -m engine_h3.worker --selftest-generate --transformer P --text-encoder P
      --video-vae P --audio-vae P --tokenizer-dir D --output out.mp4
      [--prompt S --width 640 --height 384 --num-frames 124 --seed 42 --steps 20
       --shift-video 12 --shift-audio 3 --rounds 1 --report r.json
       --attention-backend sdpa|sage --blocks-on-gpu 8 --prefetch|--no-prefetch
       --deterministic --image P --keyframe FRAME=PATH ... --keep-resident
       --vram16|--no-vram16]
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from engine_h3.constants import (  # noqa: E402 -- torch-free; needs ROOT on sys.path
    ATTENTION_BACKENDS,
    DEFAULT_SHIFT_AUDIO,
    DEFAULT_SHIFT_VIDEO,
    DEFAULT_STEPS,
    FRAME_RATE,
    LOG_PREFIX,
)

#: Same frame prefix as the LTX workers: the parent's line reader is shared.
PREFIX = "@@LTX@@"


def _configure_worker_logging() -> None:
    """ONE StreamHandler on the root logger, aimed at STDERR, tagged against re-entry.

    STDOUT is never touched -- it belongs to the ``@@LTX@@`` protocol. The
    ``[h3_worker]`` prefix tells this engine's lines apart in a shared log.
    """
    root = logging.getLogger()
    if not any(getattr(h, "_h3_worker_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(f"{LOG_PREFIX} %(name)s: %(message)s"))
        handler._h3_worker_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(logging.INFO)


_configure_worker_logging()


def _log(msg: str) -> None:
    """Human/diagnostic logging -> STDERR only (never the protocol channel)."""
    print(f"{LOG_PREFIX} {msg}", file=sys.stderr, flush=True)


def _emit(event: str, **fields: object) -> None:
    """Write one framed protocol line to STDOUT and flush."""
    payload = {"event": event, **fields}
    print(PREFIX + json.dumps(payload, separators=(",", ":"), default=str), flush=True)


def _detail(exc: BaseException) -> str:
    """repr + head/tail of the traceback (the outer frames name our call site)."""
    tb = "".join(traceback.format_exc())
    if len(tb) > 2800:
        tb = f"{tb[:1200]}\n...[traceback truncated]...\n{tb[-1600:]}"
    return f"{exc!r}\n{tb}"


def _emit_progress(stage: str, index: int, total: int) -> None:
    """One framed ``progress`` line. Never allowed to fail a job."""
    try:
        _emit("progress", stage=str(stage), index=int(index), total=int(total))
    except Exception as exc:  # noqa: BLE001
        _log(f"progress emit failed (ignored): {exc!r}")


#: The load payload's file fields (all required) and the tokenizer folder.
_LOAD_FILE_FIELDS = ("transformer_path", "text_encoder_path", "video_vae_path", "audio_vae_path")
_LOAD_DIR_FIELDS = ("tokenizer_dir",)

#: The generate payload's keys this engine reads; anything else is logged as ignored.
_GENERATE_FIELDS = frozenset({
    "op", "prompt", "seed", "width", "height", "num_frames", "frame_rate",
    "num_inference_steps", "shift_video", "shift_audio", "output_path",
    "images", "keep_resident", "attention_backend", "block_swap_prefetch",
    "vram16_mode",
})
#: The keys of one ``images`` entry this engine reads (``strength`` is an LTX notion).
_IMAGE_FIELDS = frozenset({"path", "frame_idx"})

#: The one loaded pipeline (one model per process, as in the LTX workers).
_PIPE = None
#: (effective backend, degraded) resolved at load: the default of the jobs that
#: do not send their own ``attention_backend``.
_ATTENTION: tuple[str, bool] = ("sdpa", False)


def _value_or(msg: dict, key: str, default, cast):
    """``msg[key]`` cast, or ``default`` when the key is absent OR an explicit null.

    One rule for load and generate: the app may send ``null`` for "not set",
    and ``int(None)`` / ``str(None) == "None"`` would turn that into a failed
    load or an unknown backend.
    """
    value = msg.get(key)
    return default if value is None else cast(value)


def _resolve_attention(msg: dict) -> tuple[str, bool]:
    """``attention_backend`` -> ``(effective, degraded)``, the LTX workers' rule.

    Missing -> "sdpa". An UNKNOWN value fails loudly (app and engine disagree on
    the protocol). "sage" without a usable SageAttention degrades to sdpa with a
    warning and is reported as ``attention_used="sage->sdpa"``: a speed knob,
    so "ran, just not faster" beats "failed".
    """
    from engine.transformer.sage_attention_service import probe_sage  # noqa: PLC0415

    backend = _value_or(msg, "attention_backend", "sdpa", str)
    if backend not in ATTENTION_BACKENDS:
        raise RuntimeError(f"h3 worker: unknown attention_backend {backend!r} - expected one of {ATTENTION_BACKENDS}.")
    if backend == "sage" and not probe_sage():
        # ASCII only: STDERR is cp932 with backslashreplace on a Japanese Windows.
        _log(
            "WARNING attention_backend='sage' requested but SageAttention is not available "
            "in this engine venv - falling back to sdpa (reported as attention_used='sage->sdpa')."
        )
        return "sdpa", True
    return backend, False


def _do_load(msg: dict) -> None:
    """Check the handed-down paths and headers, pin the int8 path, answer ``ready``."""
    global _PIPE, _ATTENTION
    if _PIPE is not None:
        from engine.transformer.sage_attention_service import probe_sage  # noqa: PLC0415

        _log("load: already loaded -- re-answering ready")
        _emit("ready", sampler=_PIPE.sampler, sage_available=probe_sage())
        return

    import h3_format  # noqa: PLC0415 -- torch-free; the one path check (the pipeline repeats it)

    for line in h3_format.check_model_paths(
        files={f: msg.get(f) for f in _LOAD_FILE_FIELDS},
        dirs={f: msg.get(f) for f in _LOAD_DIR_FIELDS},
    ):
        _log(line)

    # After the path checks: this pulls in sageattention + triton (seconds).
    from engine.transformer.sage_attention_service import probe_sage  # noqa: PLC0415

    sage_available = probe_sage()
    attention = _resolve_attention(msg)

    import torch  # noqa: PLC0415

    from engine_h3.pipeline_h3 import H3LoadOptions, H3Paths, H3Pipeline  # noqa: PLC0415

    defaults = H3LoadOptions()
    options = H3LoadOptions(
        blocks_on_gpu=_value_or(msg, "blocks_on_gpu", defaults.blocks_on_gpu, int),
        attention_backend=attention[0],
        block_swap_prefetch=_value_or(msg, "block_swap_prefetch", defaults.block_swap_prefetch, bool),
        deterministic=_value_or(msg, "deterministic", defaults.deterministic, bool),
    )
    paths = H3Paths(
        transformer=str(msg["transformer_path"]),
        text_encoder=str(msg["text_encoder_path"]),
        video_vae=str(msg["video_vae_path"]),
        audio_vae=str(msg["audio_vae_path"]),
        tokenizer_dir=str(msg["tokenizer_dir"]),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA が使えません（MiniMax H3 は NVIDIA GPU と CUDA 13 対応のドライバが必要です）")
    pipeline = H3Pipeline(paths, options, device=torch.device("cuda"), progress=_emit_progress, log=_log)
    report = pipeline.load()
    _PIPE = pipeline
    _ATTENTION = attention
    # ensure_ascii: STDERR is cp932 on a Japanese Windows and the parent reads
    # the log as UTF-8, so a non-ASCII path would land garbled.
    _log(f"LOAD_OK {json.dumps(report, ensure_ascii=True, default=str)}")
    _emit("ready", sampler=pipeline.sampler, sage_available=sage_available)


def _images(msg: dict) -> tuple[tuple, list[str]]:
    """``images`` -> ``(H3ImageInput, ...)`` and the ignored inner keys.

    The shape is checked here (a list of objects with a ``path``); which
    frames (0 .. num_frames-1, distinct) are the pipeline's ``validate_request``
    (one place for the rule).
    """
    from engine_h3.pipeline_h3 import H3ImageInput  # noqa: PLC0415

    raw = msg.get("images")
    if raw is None:
        return (), []
    if not isinstance(raw, list):
        raise ValueError(f"images はリストで渡してください（{type(raw).__name__}）")
    out, ignored = [], set()
    for entry in raw:
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"images の各要素は path を持つオブジェクトです（{entry!r}）")
        ignored |= {k for k in entry if k not in _IMAGE_FIELDS}
        out.append(H3ImageInput(path=str(entry["path"]), frame_idx=_value_or(entry, "frame_idx", 0, int)))
    return tuple(out), sorted(ignored)


def _generate_request(msg: dict, images: tuple | None = None):
    """``generate`` payload -> ``H3GenerateRequest`` (steps / shifts default when absent).

    ``attention_backend`` is left as sent (``None`` = the load value); the
    worker resolves it per job before the pipeline runs. ``images``: the
    already parsed ``images`` (``_do_generate`` parses them once, for the
    ignored-keys log too); parsed here when not given.
    """
    from engine_h3.pipeline_h3 import H3GenerateRequest  # noqa: PLC0415

    def _or(key, default, cast):
        return _value_or(msg, key, default, cast)

    if images is None:
        images, _ignored = _images(msg)
    return H3GenerateRequest(
        prompt=str(msg["prompt"]),
        seed=int(msg["seed"]),
        width=int(msg["width"]),
        height=int(msg["height"]),
        num_frames=int(msg["num_frames"]),
        output_path=str(msg["output_path"]),
        frame_rate=_or("frame_rate", FRAME_RATE, float),
        num_inference_steps=_or("num_inference_steps", DEFAULT_STEPS, int),
        shift_video=_or("shift_video", DEFAULT_SHIFT_VIDEO, float),
        shift_audio=_or("shift_audio", DEFAULT_SHIFT_AUDIO, float),
        images=images,
        keep_resident=_or("keep_resident", False, bool),
        attention_backend=_or("attention_backend", None, str),
        block_swap_prefetch=_or("block_swap_prefetch", None, bool),
        vram16_mode=_or("vram16_mode", False, bool),
    )


def _do_generate(msg: dict) -> None:
    """Run one T2V / I2V job; the mp4 is written by this process to ``msg['output_path']``."""
    if _PIPE is None:
        raise RuntimeError("generate before load")
    images, inner_ignored = _images(msg)
    ignored = sorted(k for k in msg if k not in _GENERATE_FIELDS)
    ignored += [f"images[].{k}" for k in inner_ignored]
    req = _generate_request(msg, images=images)
    # Per job: the job's own backend when it sends one, else the load value.
    effective, degraded = _resolve_attention(msg) if msg.get("attention_backend") is not None else _ATTENTION
    req.attention_backend = effective
    _log(
        f"generate: seed={req.seed} {req.width}x{req.height} frames={req.num_frames} "
        f"steps={req.num_inference_steps} shift={req.shift_video}/{req.shift_audio} "
        f"attention={effective} images={len(req.images)} keep_resident={req.keep_resident} "
        f"prefetch={req.block_swap_prefetch} vram16={req.vram16_mode} -> {req.output_path}"
        + (f" (ignored keys: {', '.join(ignored)})" if ignored else "")
    )
    result = _PIPE.generate(req)
    result["attention_used"] = "sage->sdpa" if degraded else effective
    _emit("done", **result)


def _release_after_failed_job() -> None:
    """Every failed job releases what ``keep_resident`` kept.

    The pipeline already does so for a failure inside ``generate``; this also
    covers a job refused before ``generate`` is reached (a malformed
    ``images``, an unknown ``attention_backend``, a missing key). Never raises.
    """
    if _PIPE is None:
        return
    try:
        _PIPE.release_resident()
    except Exception as exc:  # noqa: BLE001 -- the job's own error is what gets reported
        _log(f"releasing the kept models after a failed job failed (ignored): {exc!r}")


def _shutdown() -> None:
    """Best-effort free, then exit 0 (torch only if it was imported)."""
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
        except Exception:  # noqa: BLE001
            pass
    sys.exit(0)


def main() -> None:
    loaded = False
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:  # noqa: BLE001
            _log(f"ignoring non-JSON stdin line: {line[:120]!r}")
            continue
        if not isinstance(msg, dict):
            _log(f"ignoring non-object stdin line: {line[:120]!r}")
            continue

        op = msg.get("op")

        if not loaded:
            if op != "load":
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

        if op == "shutdown":
            _shutdown()
        if op == "generate":
            try:
                _do_generate(msg)
            except BaseException as exc:  # noqa: BLE001
                # A failed job is reported and the process stays alive.
                _log("GENERATE_FAILED")
                _release_after_failed_job()
                _emit("error", detail=_detail(exc))
            continue
        _log(f"unknown op={op!r} -- answered with an error event")
        _emit("error", detail=f"unknown op {op!r}")

    _shutdown()


# ---------------------------------------------------------------------------
# Selftest CLI
# ---------------------------------------------------------------------------

_SELFTEST_PROMPT = (
    "A woman in a sunlit kitchen looks up from her cup of tea and smiles. "
    "<d>[English] Good morning. The kettle is still warm.</d>"
)


def _selftest_parser():
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="python -m engine_h3.worker --selftest-generate")
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--video-vae", required=True)
    parser.add_argument("--audio-vae", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument(
        "--output", required=True,
        help="mp4 path; with --rounds >1 the round number goes before the suffix (out.r1.mp4)",
    )
    parser.add_argument("--prompt", default=_SELFTEST_PROMPT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--shift-video", type=float, default=DEFAULT_SHIFT_VIDEO)
    parser.add_argument("--shift-audio", type=float, default=DEFAULT_SHIFT_AUDIO)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--report", default=None, help="write the JSON report here as well as to stdout")
    parser.add_argument("--attention-backend", choices=("sdpa", "sage"), default="sdpa")
    parser.add_argument("--blocks-on-gpu", type=int, default=8)
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--image", default=None, help="I2V: the first frame (same as --keyframe 0=PATH)")
    parser.add_argument(
        "--keyframe", action="append", default=[], metavar="FRAME=PATH",
        help="I2V: a key frame at pixel frame FRAME (repeatable)",
    )
    parser.add_argument("--keep-resident", action="store_true")
    parser.add_argument("--vram16", action=argparse.BooleanOptionalAction, default=False,
                        help="VRAM 16GB mode (vram16_mode)")
    return parser


def _selftest_load_message(args) -> dict:
    """The ``load`` message the selftest sends: every key the app will send."""
    return {
        "op": "load",
        "transformer_path": args.transformer,
        "text_encoder_path": args.text_encoder,
        "video_vae_path": args.video_vae,
        "audio_vae_path": args.audio_vae,
        "tokenizer_dir": args.tokenizer_dir,
        "blocks_on_gpu": args.blocks_on_gpu,
        "attention_backend": args.attention_backend,
        "block_swap_prefetch": bool(args.prefetch),
        "deterministic": bool(args.deterministic),
    }


def _selftest_generate_message(args, out: str) -> dict:
    """The ``generate`` message the selftest sends (every key the app will send)."""
    return {
        "op": "generate",
        "prompt": args.prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "frame_rate": FRAME_RATE,
        "num_inference_steps": args.steps,
        "shift_video": args.shift_video,
        "shift_audio": args.shift_audio,
        "output_path": out,
        "images": _selftest_images(args),
        "keep_resident": bool(args.keep_resident),
        "attention_backend": args.attention_backend,
        "block_swap_prefetch": bool(args.prefetch),
        "vram16_mode": bool(args.vram16),
    }


def _selftest_images(args) -> list[dict]:
    """``--image P`` (frame 0) and every ``--keyframe FRAME=PATH`` -> the ``images`` entries.

    A malformed ``--keyframe`` or two key frames on one frame (``--image`` with
    ``--keyframe 0=...`` included) is refused here, before anything loads.
    """
    images: list[dict] = [{"path": args.image, "frame_idx": 0}] if args.image else []
    for spec in args.keyframe:
        frame, sep, path = str(spec).partition("=")
        if not sep or not frame.strip().isdigit() or not path.strip():
            raise ValueError(f"--keyframe は FRAME=PATH の形で渡してください（{spec!r}）")
        images.append({"path": path, "frame_idx": int(frame)})
    frames = [img["frame_idx"] for img in images]
    duplicated = sorted({n for n in frames if frames.count(n) > 1})
    if duplicated:
        raise ValueError(
            f"キーフレームの位置が重複しています（frame_idx={duplicated}。--image は --keyframe 0=PATH と同じです）"
        )
    return images


def _selftest_generate(argv: list[str]) -> int:
    """Load + generate once (or N times) through ``_do_load`` / ``_do_generate``, report JSON.

    The messages are synthesised exactly as the app will send them, so the run
    under test is the shipped path; ``_emit`` is wrapped to record every event
    (and still prints it). ``--rounds 2`` with one seed records whether the two
    mp4 digests agree.
    """
    import hashlib  # noqa: PLC0415
    import time  # noqa: PLC0415

    args = _selftest_parser().parse_args(argv)
    try:
        _selftest_images(args)  # a bad --keyframe is refused before the load
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    load_msg = _selftest_load_message(args)
    events: list[dict] = []
    real_emit = _emit

    def _capturing_emit(event: str, **fields: object) -> None:
        events.append({"event": event, **fields})
        real_emit(event, **fields)

    globals()["_emit"] = _capturing_emit
    try:
        report: dict = {"load": load_msg, "rounds": []}
        started = time.perf_counter()
        _do_load(load_msg)
        report["load_seconds"] = round(time.perf_counter() - started, 2)
        assert _PIPE is not None
        report["build"] = _PIPE.build_report
        report["ready"] = next((e for e in events if e["event"] == "ready"), None)

        digests = []
        for index in range(1, max(1, args.rounds) + 1):
            base = Path(args.output)
            out = str(base if args.rounds <= 1 else base.with_name(f"{base.stem}.r{index}{base.suffix}"))
            events.clear()
            _do_generate(_selftest_generate_message(args, out))
            digest = hashlib.sha256(Path(out).read_bytes()).hexdigest()
            digests.append(digest)
            done = next((e for e in events if e["event"] == "done"), None)
            if done is None:
                raise RuntimeError("the generation produced no terminal done event")
            report["rounds"].append({
                "round": index,
                "output": out,
                "sha256": digest,
                "size_bytes": Path(out).stat().st_size,
                "done": done,
                "progress": [
                    [e["stage"], e["index"], e["total"]] for e in events if e["event"] == "progress"
                ],
            })
        if len(digests) > 1:
            report["same_seed_sha_identical"] = len(set(digests)) == 1
    finally:
        globals()["_emit"] = real_emit

    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0


def _print_help() -> None:
    """``--help`` without a selftest: the protocol summary and the selftest options."""
    print((__doc__ or "").strip())
    print()
    _selftest_parser().print_help()


if __name__ == "__main__":
    if "--selftest-generate" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--selftest-generate"]
        raise SystemExit(_selftest_generate(rest))
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        _print_help()
        raise SystemExit(0)
    main()
