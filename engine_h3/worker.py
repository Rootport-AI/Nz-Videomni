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
``output_path``. Nothing LTX-specific is carried over: no IC-LoRA / NAG / VSF
branches; the chain (``generate_chain``) is this engine's own
(``engine_h3.chain_h3``, Docs/MINIMAX_H3_DESIGN.md 裁定 84), not the LTX
chain's. ``keep_resident`` follows the LTX contract (kept while the key is
true, released by a job without it).

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
   [vram16_mode], [loras], [source_audio], [references]}
      One T2V, I2V or A2V job. Omitted steps / shifts take the engine defaults
      (``engine_h3.constants``). ``images``: [] (T2V) or ``{"path", "frame_idx"}``
      key frames on distinct frames 0 .. num_frames-1 (any order); anything
      else is refused with a reason. Keys this
      engine does not read (top level, or inside an image entry) are logged
      once per job and ignored. ``keep_resident`` true keeps the DiT, the video
      decoder and the audio VAE on the CPU for the next job; a job without it
      releases them. ``vram16_mode`` true runs the denoise in VRAM 16GB mode
      (``engine_h3.lowvram``; absent / null = off here, the app sends it every
      job). ``loras``: [] or ``{"path", "strength"}`` entries (``engine_h3.lora``;
      strength defaults to 1.0; the app sends the key every job).
      ``source_audio`` (A2V, 台帳 §1-103): absent / null = T2V / I2V as before, else
      ``{"path", "strength"}`` (``strength`` 0..1, defaults to 0.5; the file's
      waveform is encoded to the audio latents and, at strength 1.0, muxed into
      the mp4; ``engine_h3.pipeline_h3.H3AudioInput``). Unknown inner keys are
      logged once and ignored.
      ``references`` (Ref2VA, 台帳 §1-107): absent / null / [] = no reference,
      else ``{"kind", "path", "slot", "max_sec"}`` entries (``kind`` picture /
      video / audio; ``slot`` the n of ``<Picture n>`` etc., 1 or more, distinct
      per kind; ``max_sec`` absent / null = only the 15 s cap;
      ``engine_h3.pipeline_h3.H3ReferenceInput``). Unknown inner keys are
      logged once and ignored. A failure is reported with ``error`` and the
      process stays alive.
  {"op": "generate_chain", seed, width, height, frame_rate, clips,
   [prompt], [num_inference_steps], [shift_video], [shift_audio], output_path,
   [keep_resident], [attention_backend], [block_swap_prefetch], [vram16_mode],
   [loras]}
      One T2V chain (台帳 §1-106; ``engine_h3.chain_h3``): ``clips`` is a list
      of ``{"prompt", "num_frames"}`` (2..24 clips; each prompt is the clip's
      whole text, the top-level ``prompt`` is not used), clip i runs with seed
      ``seed + i``, and the clips are joined on a fixed 22-frame overlap into
      ONE mp4 at ``output_path``. The other keys mean what they mean for
      ``generate`` and apply to every clip (the LoRA too). ``images`` /
      ``source_audio`` / ``references`` and any other key are logged and
      ignored, as are unknown keys inside a clip. A failure is reported with
      ``error`` and the process stays alive.
  {"op": "shutdown"}
  Before load an unknown op is only logged (an ``error`` there would be read
  as the reply to the next ``load``); after load it is answered with ``error``.

Events (worker -> parent; STDOUT, ``@@LTX@@`` + compact JSON):
  ready     {sampler, sage_available}
  progress  {stage, index, total[, outer_index, outer_total]}
            stage: encode / stage1_denoise / decode
            (``outer_index`` / ``outer_total``: a chain's clip, 0-based, and
            clip count, on its clips' encode / denoise lines only; absent
            otherwise, so a single job's lines are unchanged)
            (there is no stage2: the app adapter, services/engines/h3/adapter.py,
            reads these with ``_read_worker_events(one_stage=True)``, as Alpha
            Gen does; ``encode`` counts 3 steps, +1 when the job has any
            key frame, +1 when it has references, +1 when it encodes a source_audio
            (strength > 0); the reference and the audio encodings are one
            ``encode`` step each, and ``reference_encode`` / ``audio_encode`` are only phase names in
            ``done.h3.phases``, not a progress stage)
  done      {seed_used, sampler, num_inference_steps, shift_video, shift_audio,
             peak_vram_mb, peak_vram_reserved_mb, rss_peak_gib, commit_gib,
             commit_peak_gib, seconds,
             num_frames, frame_rate, width, height, size_bytes, attention_used,
             block_swap_prefetch_used, keep_resident_used, vram16_mode_used,
             images_used, h3: {phases, quant_backends, [loras], [a2v], [references]}}
            (``h3.loras`` only for a job with LoRA: one entry per file, its
            name, strength, layout, dtype, pairs_applied and skipped;
            ``h3.a2v`` only for an A2V job, passed through from the pipeline:
            mask_value, audio_latents, required_samples, encoded_samples,
            channels_used, resampler, encoder_tensors, audio_output,
            muxed_audio_samples; ``h3.references`` only for a Ref2VA job:
            {pictures, videos, audios} counts, with ``h3.phases.reference_encode``
            (its ``references`` list: kind, slot, rows, seconds, and per kind
            source_size / size / vision_tokens, source_frames / used_frames /
            latent_t / pairs, or samples / latent_t) and ``ref_video`` /
            ``ref_audio`` in ``h3.phases.denoise.sequence``)
            A chain's ``done`` has the same keys plus ``chain`` (the layout of
            ``h3_chain_math.H3ChainLayout.to_dict()``, ``seeds``, ``clips``:
            prompt_sha256 / seconds / latents_sha256 / head_equals_prev_tail,
            and ``h3``: overlap_frames / inject_clean / decode); its
            ``num_frames`` is the timeline's, ``h3.phases`` is
            ``{clips: [...], decode}`` and ``h3.loras`` is clip 0's.
  error     {detail}
STDERR carries the human log, prefixed ``[h3_worker]``.

Selftest:
  python -m engine_h3.worker --selftest-generate --transformer P --text-encoder P
      --video-vae P --audio-vae P --tokenizer-dir D --output out.mp4
      [--prompt S --width 640 --height 384 --num-frames 124 --seed 42 --steps 20
       --shift-video 12 --shift-audio 3 --rounds 1 --report r.json
       --attention-backend sdpa|sage --blocks-on-gpu 8 --prefetch|--no-prefetch
       --deterministic --image P --keyframe FRAME=PATH ... --keep-resident
       --vram16|--no-vram16 --lora PATH ... --lora-strength 1.0
       --source-audio PATH --audio-strength 0.5
       --ref-image PATH ... --ref-video PATH ... --ref-audio PATH ...]
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


def _emit_progress(
    stage: str, index: int, total: int, *, outer_index: int | None = None, outer_total: int | None = None,
) -> None:
    """One framed ``progress`` line. Never allowed to fail a job.

    ``outer_index`` / ``outer_total``: a chain's clip (0-based) and clip count,
    under the LTX workers' field names (the app's receipt loop is shared).
    Omitted from the line when absent, so a single job's lines are unchanged.
    """
    try:
        fields: dict[str, object] = {"stage": str(stage), "index": int(index), "total": int(total)}
        if outer_index is not None:
            fields["outer_index"] = int(outer_index)
        if outer_total is not None:
            fields["outer_total"] = int(outer_total)
        _emit("progress", **fields)
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
    "vram16_mode", "loras", "source_audio", "references",
})
#: The keys of one ``images`` entry this engine reads (``strength`` is an LTX notion).
_IMAGE_FIELDS = frozenset({"path", "frame_idx"})
#: The keys of one ``loras`` entry this engine reads (``audio_strength`` is an LTX notion).
_LORA_FIELDS = frozenset({"path", "strength"})
#: The keys of ``source_audio`` this engine reads.
_SOURCE_AUDIO_FIELDS = frozenset({"path", "strength"})
#: The keys of one ``references`` entry this engine reads (Ref2VA, 台帳 §1-107).
_REFERENCE_FIELDS = frozenset({"kind", "path", "slot", "max_sec"})
#: The generate_chain payload's keys this engine reads (台帳 §1-106): the single
#: job's, less what a T2V chain has no use for (the frame count is per clip),
#: plus ``clips``.
_CHAIN_FIELDS = (_GENERATE_FIELDS - {"num_frames", "images", "source_audio", "references"}) | {"clips"}
#: The keys of one ``clips`` entry this engine reads.
_CHAIN_CLIP_FIELDS = frozenset({"prompt", "num_frames"})

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


def _loras(msg: dict) -> tuple[tuple, list[str]]:
    """``loras`` -> ``(H3LoraInput, ...)`` and the ignored inner keys.

    The shape is checked here (a list of objects with a ``path``); the path,
    the strength and the file itself are the pipeline's ``validate_request``.
    """
    from engine_h3.pipeline_h3 import H3LoraInput  # noqa: PLC0415

    raw = msg.get("loras")
    if raw is None:
        return (), []
    if not isinstance(raw, list):
        raise ValueError(f"loras はリストで渡してください（{type(raw).__name__}）")
    out, ignored = [], set()
    for entry in raw:
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"loras の各要素は path を持つオブジェクトです（{entry!r}）")
        ignored |= {k for k in entry if k not in _LORA_FIELDS}
        out.append(H3LoraInput(path=str(entry["path"]), strength=_value_or(entry, "strength", 1.0, float)))
    return tuple(out), sorted(ignored)


def _source_audio(msg: dict) -> tuple:
    """``source_audio`` -> ``(H3AudioInput | None, ignored inner keys)``.

    Absent / null = no A2V. The shape (an object with a non-empty string
    ``path`` and a finite number ``strength``) is checked here; the 0..1 range
    and the file itself are the pipeline's ``validate_request``.
    """
    import math  # noqa: PLC0415

    from engine_h3.pipeline_h3 import H3AudioInput  # noqa: PLC0415

    raw = msg.get("source_audio")
    if raw is None:
        return None, []
    if not isinstance(raw, dict):
        raise ValueError(f"source_audio はオブジェクトで渡してください（{type(raw).__name__}）")
    path = raw.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"source_audio には空でない文字列の path が必要です（{path!r}）")
    strength = raw.get("strength")
    if strength is None:
        strength = 0.5
    elif isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength):
        raise ValueError(f"source_audio の strength は有限の数値で渡してください（{strength!r}）")
    ignored = sorted(k for k in raw if k not in _SOURCE_AUDIO_FIELDS)
    return H3AudioInput(path=path, strength=float(strength)), ignored


def _references(msg: dict) -> tuple[tuple, list[str]]:
    """``references`` -> ``(H3ReferenceInput, ...)`` and the ignored inner keys.

    Absent / null = none. The shape (a list of objects with a string ``kind``,
    a non-empty string ``path``, an integer ``slot`` and an absent / null /
    finite-number ``max_sec``) is checked here; the kind's name, the slot's
    range and uniqueness, the file and the ``max_sec`` range are the pipeline's
    ``validate_request``.
    """
    import math  # noqa: PLC0415

    from engine_h3.pipeline_h3 import H3ReferenceInput  # noqa: PLC0415

    raw = msg.get("references")
    if raw is None:
        return (), []
    if not isinstance(raw, list):
        raise ValueError(f"references はリストで渡してください（{type(raw).__name__}）")
    out, ignored = [], set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"references の各要素はオブジェクトです（{entry!r}）")
        kind, path, slot, max_sec = entry.get("kind"), entry.get("path"), entry.get("slot"), entry.get("max_sec")
        if not isinstance(kind, str):
            raise ValueError(f"references の kind は文字列で渡してください（{kind!r}）")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"references には空でない文字列の path が必要です（{path!r}）")
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise ValueError(f"references の slot は整数で渡してください（{slot!r}）")
        if max_sec is not None and (
            isinstance(max_sec, bool) or not isinstance(max_sec, (int, float)) or not math.isfinite(max_sec)
        ):
            raise ValueError(f"references の max_sec は有限の数値か null で渡してください（{max_sec!r}）")
        ignored |= {k for k in entry if k not in _REFERENCE_FIELDS}
        out.append(H3ReferenceInput(
            kind=kind, path=path, slot=slot, max_sec=None if max_sec is None else float(max_sec),
        ))
    return tuple(out), sorted(ignored)


def _generate_request(
    msg: dict, images: tuple | None = None, loras: tuple | None = None, source_audio=None,
    references: tuple | None = None,
):
    """``generate`` payload -> ``H3GenerateRequest`` (steps / shifts default when absent).

    ``attention_backend`` is left as sent (``None`` = the load value); the
    worker resolves it per job before the pipeline runs. ``images``: the
    already parsed ``images`` (``_do_generate`` parses them once, for the
    ignored-keys log too); parsed here when not given. ``loras``: the same.
    ``source_audio``: the already parsed ``H3AudioInput``; ``None`` falls back to
    parsing the message (a message without the key stays T2V / I2V).
    ``references``: the already parsed references; ``None`` parses the message.
    """
    from engine_h3.pipeline_h3 import H3GenerateRequest  # noqa: PLC0415

    def _or(key, default, cast):
        return _value_or(msg, key, default, cast)

    if images is None:
        images, _ignored = _images(msg)
    if loras is None:
        loras, _ignored = _loras(msg)
    if source_audio is None:
        source_audio, _ignored = _source_audio(msg)
    if references is None:
        references, _ignored = _references(msg)
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
        loras=loras,
        source_audio=source_audio,
        references=references,
    )


def _do_generate(msg: dict) -> None:
    """Run one T2V / I2V job; the mp4 is written by this process to ``msg['output_path']``."""
    if _PIPE is None:
        raise RuntimeError("generate before load")
    images, inner_ignored = _images(msg)
    loras, lora_ignored = _loras(msg)
    source_audio, audio_ignored = _source_audio(msg)
    references, ref_ignored = _references(msg)
    ignored = sorted(k for k in msg if k not in _GENERATE_FIELDS)
    ignored += [f"images[].{k}" for k in inner_ignored]
    ignored += [f"loras[].{k}" for k in lora_ignored]
    ignored += [f"source_audio.{k}" for k in audio_ignored]
    ignored += [f"references[].{k}" for k in ref_ignored]
    req = _generate_request(msg, images=images, loras=loras, source_audio=source_audio, references=references)
    # Per job: the job's own backend when it sends one, else the load value.
    effective, degraded = _resolve_attention(msg) if msg.get("attention_backend") is not None else _ATTENTION
    req.attention_backend = effective
    _log(
        f"generate: seed={req.seed} {req.width}x{req.height} frames={req.num_frames} "
        f"steps={req.num_inference_steps} shift={req.shift_video}/{req.shift_audio} "
        f"attention={effective} images={len(req.images)} keep_resident={req.keep_resident} "
        f"prefetch={req.block_swap_prefetch} vram16={req.vram16_mode} "
        f"loras={[(Path(item.path).name, item.strength) for item in req.loras]} "
        + (
            f"source_audio={(Path(req.source_audio.path).name, req.source_audio.strength)} "
            if req.source_audio is not None else ""
        )
        + (
            f"references={[(r.kind, r.slot, Path(r.path).name, r.max_sec) for r in req.references]} "
            if req.references else ""
        )
        + f"-> {req.output_path}"
        + (f" (ignored keys: {', '.join(ignored)})" if ignored else "")
    )
    result = _PIPE.generate(req)
    result["attention_used"] = "sage->sdpa" if degraded else effective
    _emit("done", **result)


def _chain_clips(msg: dict) -> tuple[list, list[str]]:
    """``clips`` -> ``[H3ChainClipSpec, ...]`` and the ignored inner keys.

    The shape (a list of objects with a string ``prompt`` and an integer
    ``num_frames``) is checked here; the count, the frame grid and the lengths
    are ``h3_chain_math``'s, the prompt the pipeline's ``validate_request``.
    """
    from engine_h3.chain_h3 import H3ChainClipSpec  # noqa: PLC0415

    raw = msg.get("clips")
    if not isinstance(raw, list):
        raise ValueError(f"clips はリストで渡してください（{type(raw).__name__}）")
    out, ignored = [], set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"clips の各要素はオブジェクトです（{entry!r}）")
        prompt, frames = entry.get("prompt"), entry.get("num_frames")
        if not isinstance(prompt, str):
            raise ValueError(f"clips の prompt は文字列で渡してください（{prompt!r}）")
        if isinstance(frames, bool) or not isinstance(frames, int):
            raise ValueError(f"clips の num_frames は整数で渡してください（{frames!r}）")
        ignored |= {k for k in entry if k not in _CHAIN_CLIP_FIELDS}
        out.append(H3ChainClipSpec(prompt=prompt, num_frames=frames))
    return out, sorted(ignored)


def _do_generate_chain(msg: dict) -> None:
    """Run one T2V chain (台帳 §1-106); the mp4 is written by this process to ``msg['output_path']``."""
    if _PIPE is None:
        raise RuntimeError("generate before load")
    from engine_h3.chain_h3 import H3ChainSpec, run_chain_h3  # noqa: PLC0415

    clips, clip_ignored = _chain_clips(msg)
    loras, lora_ignored = _loras(msg)
    ignored = sorted(k for k in msg if k not in _CHAIN_FIELDS)
    ignored += [f"clips[].{k}" for k in clip_ignored]
    ignored += [f"loras[].{k}" for k in lora_ignored]
    # The shared settings, read by the single job's rules. The prompt and the
    # frame count are each clip's; ``clips[0]``'s stand in for them here.
    base_msg = {k: v for k, v in msg.items() if k in _CHAIN_FIELDS}
    if clips:
        base_msg["prompt"], base_msg["num_frames"] = clips[0].prompt, clips[0].num_frames
    else:
        base_msg["prompt"], base_msg["num_frames"] = "", 0
    base = _generate_request(base_msg, images=(), loras=loras, source_audio=None, references=())
    # Per job: the job's own backend when it sends one, else the load value.
    effective, degraded = _resolve_attention(msg) if msg.get("attention_backend") is not None else _ATTENTION
    base.attention_backend = effective
    spec = H3ChainSpec(base=base, clips=tuple(clips), seed=base.seed)
    _log(
        f"generate_chain: seed={base.seed} {base.width}x{base.height} clips={[c.num_frames for c in clips]} "
        f"steps={base.num_inference_steps} shift={base.shift_video}/{base.shift_audio} "
        f"attention={effective} keep_resident={base.keep_resident} "
        f"prefetch={base.block_swap_prefetch} vram16={base.vram16_mode} "
        f"loras={[(Path(item.path).name, item.strength) for item in base.loras]} "
        f"-> {base.output_path}"
        + (f" (ignored keys: {', '.join(ignored)})" if ignored else "")
    )
    result = run_chain_h3(_PIPE, spec, _emit_progress, _log)
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
        if op == "generate_chain":
            try:
                _do_generate_chain(msg)
            except BaseException as exc:  # noqa: BLE001
                # A failed chain is reported and the process stays alive, as a job.
                _log("GENERATE_CHAIN_FAILED")
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
    parser.add_argument("--lora", action="append", default=[], metavar="PATH",
                        help="a LoRA file (repeatable)")
    parser.add_argument("--lora-strength", type=float, default=1.0,
                        help="the strength of every --lora (default 1.0)")
    parser.add_argument("--source-audio", default=None, metavar="PATH",
                        help="A2V: the source audio file (source_audio.path)")
    parser.add_argument("--audio-strength", type=float, default=0.5,
                        help="A2V: source_audio.strength (default 0.5)")
    parser.add_argument("--ref-image", action="append", default=[], metavar="PATH",
                        help="Ref2VA: a picture reference (repeatable; <Picture 1>, <Picture 2>, ... in order)")
    parser.add_argument("--ref-video", action="append", default=[], metavar="PATH",
                        help="Ref2VA: a video reference (repeatable; <Video 1>, ... in order)")
    parser.add_argument("--ref-audio", action="append", default=[], metavar="PATH",
                        help="Ref2VA: an audio reference (repeatable; <Audio 1>, ... in order)")
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
    msg = {
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
        "loras": [{"path": path, "strength": float(args.lora_strength)} for path in args.lora],
    }
    if args.source_audio:
        # Appended only when asked: a job without it is the same message as before.
        msg["source_audio"] = {"path": args.source_audio, "strength": float(args.audio_strength)}
    references = _selftest_references(args)
    if references:
        # The same way: a job without references is the same message as before.
        msg["references"] = references
    return msg


def _selftest_references(args) -> list[dict]:
    """``--ref-image`` / ``--ref-video`` / ``--ref-audio`` -> the ``references`` entries;
    the slot is numbered per kind in the order the options appear (from 1)."""
    out: list[dict] = []
    for kind, paths in (("picture", args.ref_image), ("video", args.ref_video), ("audio", args.ref_audio)):
        for slot, path in enumerate(paths, start=1):
            out.append({"kind": kind, "path": path, "slot": slot, "max_sec": None})
    return out


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
