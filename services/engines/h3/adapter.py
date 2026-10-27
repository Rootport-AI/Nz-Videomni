"""MiniMax H3 engine adapter — the app-side half of the ``engine_h3`` worker.

A SIBLING of :mod:`services.engines.ltx.adapter` and
:mod:`services.engines.ltx25.adapter`, built the way the 2.5 one is: the
app-side plumbing (the ``@@LTX@@`` JSON-lines protocol, the subprocess
spawn/read/kill loop, the progress-receipt loop, the synthetic mock backend) is
2.3's, reached through its override seams, and this module states only the
facts that are genuinely different:

* which venv and which ``python -m`` module the worker is (``engine_h3.worker``
  in ``.venv-engine-h3``, ``model.engine_python_h3``);
* which worker log file it writes;
* which load-payload field each model-management category feeds, plus the
  tokenizer folder (a descriptor ``assets`` entry, like 2.3's ``gemma_root``);
* the feature scope (the four tables below) and the numeric rules
  (:data:`REQUEST_RULES`: 17n+5 frames from 22, a 32 multiple, 24 fps fixed,
  1..50 steps, the two shifts, the keyframe grid (any frame 0..F-1: step 1 /
  offset 0, no strength), ``source_audio_strength`` (single-generate A2V), and ``flag_defaults``: VRAM 16GB mode on).

THE THIRD FAMILY DOES NOT YET EARN THE GENERALISATIONS
Docs/MULTI_ENGINE_DESIGN.md §8.1 deferred to it (a generic runner, a shared
mock, a variable category set): ``H3Runner`` subclasses ``LTXRunner``, the mock
is 2.3's plus the A2V record (``_MockBackendH3``); ``generate`` also forwards ``source_audio_path``, and the categories are the same four words.

THE WORKER PROTOCOL (Docs/MINIMAX_H3_DESIGN.md §8): ``load`` carries the four
paths, ``tokenizer_dir``, ``blocks_on_gpu``, ``attention_backend``,
``block_swap_prefetch`` and ``deterministic`` (no text-encoder knob: the text
encoder is read layer by layer straight from the file). ``generate`` carries the job, the
conditioning ``images`` (``{path, frame_idx}``; H3 has no strength), the
``loras`` (``{path, strength}``; always sent, ``[]`` = none), and the
per-job ``attention_backend`` / ``block_swap_prefetch`` (always sent: an absent
key would mean "the load value", not "off"), ``vram16_mode`` (always sent: the
worker's own default is off, the app's is on), ``source_audio``
(``{path, strength}``; single-generate A2V only — an absent key is "no source
audio") and ``keep_resident`` (only when asked for — an absent key is the
release request, the LTX contract). The progress stages are LTX's single-stage
ones (``encode`` / ``stage1_denoise`` / ``decode``), read with
``one_stage=True``; the A2V audio encode is a step of ``encode`` (its own name,
``audio_encode``, is a phase in ``done.h3.phases`` only), and a stage name the
reader does not know holds the bar where it is.

LIKE THE LTX ADAPTERS, THIS FILE NEVER IMPORTS torch. The engine lives only in
``.venv-engine-h3``; the app venv has none of it.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Callable

import h3_lora_format
from api.errors import feature_unsupported
from api.models import (
    GenerateChainRequest,
    GenerateRequest,
    VRAM16_MODE_DEFAULT_H3,
)
from config import AppConfig
from services import prompt_tokens, video_io
from services.engines.ltx.adapter import (
    GenerationOutcome,
    LTXRunner,
    ProgressCallback,
    _MockBackend,
    _RealBackend,
    resolve_seed,
)
from services.engines.ltx.request_rules import (
    FRAME_RATE_MAX,
    NUM_FRAMES_MAX,
    ReferenceLimits,
    RequestRules,
    SamplingRange,
)
from services.lora_registry import ResolvedLora

logger = logging.getLogger("h3.runner")

#: Model-management category -> the worker load-payload field it feeds. THE
#: SAME FOUR CATEGORY NAMES as both LTX families (the axes a user picks a file
#: on are a property of the product, and state.json is keyed by them); only the
#: payload fields are this worker's (engine_h3/worker.py ``_LOAD_FILE_FIELDS``).
SELECTION_FIELDS: dict[str, str] = {
    "transformer": "transformer_path",
    "text_encoder": "text_encoder_path",
    "video_vae": "video_vae_path",
    "audio": "audio_vae_path",
}

#: Fixed (non-selectable) descriptor assets: the Qwen3-VL tokenizer FOLDER.
#: A folder because the extra special tokens live only in
#: ``tokenizer_config.json`` (Docs/MINIMAX_H3_DESIGN.md §8 "tokenizer の渡し方"),
#: the same folder-asset form as 2.3's ``gemma_root``. ``_real_available``
#: checks it with ``.exists()``, which a folder satisfies.
REQUIRED_ASSETS: tuple[str, ...] = ("tokenizer_dir",)

#: Real / mock backend identifiers surfaced in metadata.json.
REAL_BACKEND_H3 = "minimax-h3"
MOCK_BACKEND_H3 = "mock-h3"

#: Worker load default for ``blocks_on_gpu`` (2.5's precedent, and the same
#: ``or`` rule: 0 in the setting means "unset"). ``GET /status``'s
#: ``block_swap_prefetch_available`` is computed app-side from the same setting.
DEFAULT_BLOCKS_ON_GPU = 8

# --------------------------------------------------------------------------- #
# numeric rules
# --------------------------------------------------------------------------- #

#: Frame grid: the video VAE's clip length 17 with 5 extra frames (``F = 17n +
#: 5``, ``T' = 5n + 2``; engine_h3/constants.py ``latent_frames``). ``n >= 1``:
#: F = 5 decodes poorly and is refused by the engine too (MIN_NUM_FRAMES = 22).
H3_FRAMES_STEP = 17
H3_FRAMES_OFFSET = 5
H3_FRAMES_MIN = 22
#: One DiT token = 32 px square (VAE 16 x patch 2).
H3_DIMENSION_MULTIPLE = 32
#: The model was trained at 24 fps only.
H3_FRAME_RATE = 24.0
#: The engine's own default (engine_h3/constants.py DEFAULT_STEPS); held once here,
#: agreement checked by test.
H3_DEFAULT_STEPS = 20
#: The largest 17n+5 count the schema ceiling allows (481 = 17*28+5).
H3_FRAMES_MAX = (NUM_FRAMES_MAX - H3_FRAMES_OFFSET) // H3_FRAMES_STEP * H3_FRAMES_STEP + H3_FRAMES_OFFSET

#: This family's numeric rules (Docs/MINIMAX_H3_DESIGN.md §9). The sampling
#: defaults are the engine's own (engine_h3/constants.py: 20 Euler steps,
#: shifts 12 / 3 -- the official scheduler pair). They are copied here because
#: the server side does not import the ``engine_h3`` package (a boundary, not
#: an inability). The step default is held once in ``H3_DEFAULT_STEPS``, and
#: tests/test_engine_h3_adapter.py checks the copies agree with
#: engine_h3/constants.py.
REQUEST_RULES = RequestRules(
    frames_step=H3_FRAMES_STEP,
    frames_offset=H3_FRAMES_OFFSET,
    frames_min=H3_FRAMES_MIN,
    frames_max=H3_FRAMES_MAX,
    dimension_multiple=H3_DIMENSION_MULTIPLE,
    frame_rate_min=H3_FRAME_RATE,
    frame_rate_max=H3_FRAME_RATE,
    default_steps=H3_DEFAULT_STEPS,
    sampling={
        "num_inference_steps": SamplingRange(min=1, max=50, step=1, default=H3_DEFAULT_STEPS),
        "shift_video": SamplingRange(min=1, max=20, step=0.1, default=12),
        "shift_audio": SamplingRange(min=1, max=20, step=0.1, default=3),
    },
    # vram16_mode's None means "this family's default" (裁定 44: on for H3).
    flag_defaults={"vram16_mode": VRAM16_MODE_DEFAULT_H3},
    # A keyframe may sit on any frame 0..F-1 (the RoPE time coordinate is the
    # pixel frame, not bound to a latent frame). No keyframe strength: an
    # omitted one stays None (keyframe_strength_default left at None).
    keyframe_step=1,
    keyframe_offset=0,
    # Single-generate A2V (``request.source_audio.strength``): mask value
    # m = 1 - strength on every audio latent; 1.0 keeps the source audio as is.
    source_audio_strength=SamplingRange(min=0.0, max=1.0, step=0.05, default=0.5),
    # Ref2VA's limits for the UI, always published whatever weights are loaded. The engine is not implemented yet (§1-107).
    reference=ReferenceLimits(
        max_images=9,
        max_videos=3,
        max_audios=3,
        max_total=12,
        video_total_sec=15.0,
        audio_total_sec=15.0,
        clip_min_sec=2.0,
    ),
)
assert H3_FRAME_RATE <= FRAME_RATE_MAX  # the schema ceiling must admit it

#: The seven special tokens the H3 text encoder adds to the Qwen3-VL
#: tokenizer.json, in id order (151669..). A copy of
#: ``engine_h3/text_encoder.py``'s ``EXTRA_SPECIAL_TOKENS`` (the server does not
#: import ``engine_h3``); tests/test_prompt_tokens.py checks it.
EXTRA_SPECIAL_TOKENS: tuple[str, ...] = (
    "<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>", "<|caption_start|>", "<|caption_end|>",
)


def count_prompt_tokens(
    texts: list[str],
    *,
    assets: Callable[[str], Path],
    text_encoder: Callable[[], Path],
) -> list[int]:
    """Token count of each prompt as the H3 text encoder sees it.

    The Qwen3-VL tokenizer (``assets.tokenizer_dir``) plus
    :data:`EXTRA_SPECIAL_TOKENS`, no special tokens added and no stripping
    (an empty prompt is 0). The text only: an I2V image's tokens are not
    counted, and there is no confirmed limit (``prompt_token_limit`` stays
    ``None``). ``text_encoder`` is unused. Dispatched by
    ``services.engines.count_prompt_tokens``.
    """
    tokenizer = prompt_tokens.tokenizer_file(
        assets("tokenizer_dir") / "tokenizer.json", EXTRA_SPECIAL_TOKENS
    )
    return [len(e.ids) for e in tokenizer.encode_batch(list(texts), add_special_tokens=False)]

# --------------------------------------------------------------------------- #
# feature scope (the four-way classification of GenerateRequest's fields)
# --------------------------------------------------------------------------- #


def _keyframe_strength(r: GenerateRequest) -> bool:
    """A conditioning image sets a strength (H3 has none; omitted = ``None``)."""
    return any(ci.strength is not None for ci in r.conditioning_images)


#: The 422 half: ``(field, feature, is_non_default)``, first offender wins. Every
#: predicate tests "differs from the default" (for ``strength`` the default is
#: ``None``: the operation panel omits it when the family has none, and any
#: value is refused), so a plain T2V / keyframe I2V request passes.
#:
#: ``conditioning_images`` appears ONCE here and is also HONOURED: the
#: keyframes (any frame, up to the schema's count cap) are what this engine
#: does with it, and this row refuses what it cannot (a strength). The audit
#: test allows exactly this overlap.
REJECT_TABLE: tuple[tuple[str, str, Callable[[GenerateRequest], bool]], ...] = (
    ("reference_video_id", "reference_video", lambda r: r.reference_video_id is not None),
    ("nag_enabled", "nag", lambda r: r.nag_enabled),
    ("pipeline", "two_stage_hq", lambda r: r.pipeline != "distilled"),
    ("guidance_scale", "guidance_scale", lambda r: r.guidance_scale != 1.0),
    ("outpaint", "outpaint", lambda r: r.outpaint is not None),
    ("inpaint", "inpaint", lambda r: r.inpaint is not None),
    ("alpha_gen", "alpha_gen", lambda r: r.alpha_gen is not None),
    ("vae_mode", "prune_vaed", lambda r: r.vae_mode != "default"),
    ("keep_resident_embeddings", "keep_resident_embeddings", lambda r: r.keep_resident_embeddings),
    ("conditioning_images", "keyframe_strength", _keyframe_strength),
)

#: The ignore-and-log half: a concept this engine does not have. Logged only
#: when the request sets a non-default value (2.5's discipline).
IGNORED_FIELDS: dict[str, str] = {
    "fused_gguf_dequant_kernel": "MiniMax H3 has no GGUF weights to dequantize",
}

#: Fields this engine ACTS ON: plain ``request.<field>`` reads in
#: :meth:`_RealBackendH3.generate`, except ``crop_output`` (the app-side ffmpeg
#: centre-crop of the finished mp4, as on LTX) and ``embed_mp4_metadata`` (the
#: app-side recipe embed, ``PipelineManager._embed_recipe``).
HONOURED_FIELDS: frozenset[str] = frozenset(
    {
        "prompt",
        "seed",
        "width",
        "height",
        "num_frames",
        "frame_rate",
        "num_inference_steps",
        "shift_video",
        "shift_audio",
        "conditioning_images",
        "loras",
        "keep_resident",
        "attention_backend",
        "block_swap_prefetch",
        "vram16_mode",
        # Single-generate A2V: the uploaded audio's path + strength reach the
        # worker as the ``source_audio`` payload key.
        "source_audio",
        "crop_output",
        "embed_mp4_metadata",
    }
)

#: ``field -> the field that governs it``: sub-parameters inert unless their
#: governor is non-default, and every governor here is refused above. The two
#: IC-LoRA strengths are the REFERENCE VIDEO's (what they scale), so
#: ``reference_video_id`` governs them: sent with ``loras`` alone they are
#: inert here.
GOVERNED_FIELDS: dict[str, str] = {
    "negative_prompt": "nag_enabled",
    "neg_method": "nag_enabled",
    "vsf_scale": "nag_enabled",
    "nag_scale": "nag_enabled",
    "nag_tau": "nag_enabled",
    "nag_alpha": "nag_enabled",
    "conditioning_attention_strength": "reference_video_id",
    "reference_video_strength": "reference_video_id",
}

#: Everything GET /models publishes as this engine's ``unsupported_features``:
#: the single-request reject table's feature names, plus ``chain``,
#: ``retake`` and ``fused_gguf_dequant_kernel``. ``chain`` and ``retake`` are
#: refused by :func:`reject_chain` (the chain endpoint is refused as a whole):
#: ``chain`` closes the Chained tab and Single's A2V card — unless the engine publishes ``constraints.source_audio`` (``a2vRouteFor`` then routes the card to ``/generate``) — and ``retake`` — a
#: chain request too — closes Retake's Edit sub-tab. With Outpainting,
#: Inpainting and Alpha Gen also refused, all four Edit sub-tabs close and the
#: operation panel greys the whole Edit tab (裁定 34).
#: ``fused_gguf_dequant_kernel`` is the one name here the server does NOT
#: refuse: it stays in :data:`IGNORED_FIELDS` (MiniMax H3 has no GGUF weights,
#: so a saved OFF must keep passing rather than 422), and is published only
#: so the UIs hide a control that can do nothing on this engine (台帳 §1-100).
#: ``lora_audio_strength`` is not refused either: H3 has no separate audio
#: strength for a LoRA, and ``services.engines.validate_for_family`` writes
#: every ``LoraSpec.audio_strength`` back to ``None`` (台帳 §1-104 裁定 4).
#: Nothing else — a name for a control that works here would close it.
UNSUPPORTED_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys(feature for _field, feature, _pred in REJECT_TABLE)
) + ("chain", "retake", "fused_gguf_dequant_kernel", "lora_audio_strength")


def _h3_lora_verdict(header: dict) -> tuple[bool, str | None]:
    """GET /loras' "supported" for one file, from its safetensors header alone
    (``h3_lora_format.map_lora_keys``: the same key mapping the engine uses).
    Display only: an unsupported file still runs (裁定 2)."""
    tensors = {
        name: spec for name, spec in header.items()
        if name != "__metadata__" and isinstance(spec, dict)
    }
    keymap = h3_lora_format.map_lora_keys(
        tensors,
        shapes={
            name: spec["shape"] if isinstance(spec.get("shape"), list) else ()
            for name, spec in tensors.items()
        },
        dtypes={name: str(spec.get("dtype")) for name, spec in tensors.items()},
    )
    return keymap.supported, keymap.reason


#: The family's LoRA format judge (``services.engines.lora_verdict``).
LORA_VERDICT = _h3_lora_verdict


def _refusal_detail(field: str, feature: str) -> str:
    if feature == "keyframe_strength":
        return "MiniMax H3 にはキーフレームの強度がありません（strength は指定しないでください）。"
    return (
        f"MiniMax H3 は {feature} に対応していません"
        f"（リクエストの {field} が既定値ではありません）。"
    )


def reject_unsupported(request: GenerateRequest) -> None:
    """422 the first out-of-scope field of ``request`` (first offender wins)."""
    for field, feature, is_non_default in REJECT_TABLE:
        if is_non_default(request):
            raise feature_unsupported(feature, detail=_refusal_detail(field, feature))


def reject_chain(request: GenerateChainRequest) -> None:
    """The chain endpoint is out of scope for this engine as a whole."""
    raise feature_unsupported(
        "chain",
        detail="MiniMax H3 は chain（連結生成）に対応していません。",
    )


def _log_ignored(request: GenerateRequest) -> None:
    """ONE log line naming every ignored field this request actually set."""
    fields = type(request).model_fields
    named = [
        f"{name}={getattr(request, name)!r} ({reason})"
        for name, reason in IGNORED_FIELDS.items()
        if name in fields and getattr(request, name) != fields[name].default
    ]
    if named:
        logger.info("MiniMax H3 ignores %d requested field(s): %s", len(named), "; ".join(named))


def check_kv(category: str, name: str, kv: dict[str, str]) -> None:
    """Nothing to rule on: H3 is judged by its tensor-name fingerprint in the
    precheck (``h3_format.inspect_h3_dit`` / ``inspect_h3_te``), which returns
    no KV."""
    return None


#: Top-level facts of the worker's ``done`` event that ``metadata["h3"]``
#: gathers next to the worker's own ``h3`` block (phases, quant backends).
#: The worker's ``h3`` block is merged in flat, so its A2V record
#: (``done.h3.a2v``, A2V jobs only) arrives as ``outcome.h3["a2v"]``;
#: ``PipelineManager._write_metadata`` lifts it out into ``metadata["a2v"]``.
_DONE_FACTS = (
    "sampler",
    "num_inference_steps",
    "shift_video",
    "shift_audio",
    "images_used",
    "rss_peak_gib",
    "commit_gib",
    "commit_peak_gib",
    "seconds",
)


def _h3_block(event: dict) -> dict:
    block = {key: event[key] for key in _DONE_FACTS if key in event}
    inner = event.get("h3")
    if isinstance(inner, dict):
        block.update(inner)
    return block


# --------------------------------------------------------------------------- #
# backend
# --------------------------------------------------------------------------- #


class _RealBackendH3(_RealBackend):
    """The ``engine_h3.worker`` subprocess, driven through 2.3's process plumbing."""

    SELECTION_FIELDS = SELECTION_FIELDS
    REQUIRED_ASSETS = REQUIRED_ASSETS
    _WORKER_MODULE = "engine_h3.worker"
    _LOG_NAME = "h3_worker.log"
    #: Fixed, not configurable: the package ships in this repository.
    _ENGINE_DIR_VALUE = "./engine_h3"
    _ENGINE_PYTHON_LABEL = "engine_python_h3"

    @classmethod
    def _engine_python_value(cls, config: AppConfig) -> str | None:
        return config.model.engine_python_h3

    # ------------------------------------------------------------------ load

    def _build_child_env(self, project_root: Path) -> dict[str, str]:
        """Minimal, like 2.5's: none of 2.3's ``LTX_*`` knobs (each names a
        code path inside ``engine/``); only the shared process hygiene."""
        env = dict(os.environ)
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("PYTHONPATH", None)
        env["PYTHONPATH"] = str(project_root)
        return env

    def _log_load_start(self, engine_python: str, engine_dir: Path, payload: dict) -> None:
        logger.info(
            "Loading pipeline (REAL engine_h3 worker). python=%s engine_dir=%s "
            "blocks_on_gpu=%s deterministic=%s",
            engine_python,
            engine_dir,
            payload["blocks_on_gpu"],
            payload["deterministic"],
        )

    def _build_load_payload(self, selection: dict[str, str] | None = None) -> dict:
        """The ``{"op": "load"}`` payload engine_h3's worker expects (pure).

        Key set AND order are pinned by the golden test in
        tests/test_engine_h3_adapter.py. ``attention_backend`` /
        ``block_swap_prefetch`` here are only the worker's defaults: every
        generate carries its own.
        """
        selection = selection or {}
        swapped = {
            field: (
                str(selection[category])
                if selection.get(category)
                else self._require_default_file(category)
            )
            for category, field in self.SELECTION_FIELDS.items()
        }
        return {
            "op": "load",
            "transformer_path": swapped["transformer_path"],
            "text_encoder_path": swapped["text_encoder_path"],
            "video_vae_path": swapped["video_vae_path"],
            "audio_vae_path": swapped["audio_vae_path"],
            "tokenizer_dir": self._require_asset("tokenizer_dir"),
            "blocks_on_gpu": self.low_vram.block_swap_blocks_on_gpu or DEFAULT_BLOCKS_ON_GPU,
            "attention_backend": GenerateRequest.model_fields["attention_backend"].default,
            "block_swap_prefetch": GenerateRequest.model_fields["block_swap_prefetch"].default,
            # cuDNN algorithm selection pinned, so a same-seed rerun is
            # bit-identical (2.5's precedent).
            "deterministic": True,
        }

    # -------------------------------------------------------------- generate

    def generate(
        self,
        request: GenerateRequest,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        conditioning_image_paths: list[Path] | None = None,
        lora_paths: list[ResolvedLora] | None = None,
        reference_video_path: Path | None = None,
        seed: int | None = None,
        outpaint_source_path: Path | None = None,
        inpaint_source_path: Path | None = None,
        inpaint_mask_path: Path | None = None,
        source_audio_path: Path | None = None,
    ) -> GenerationOutcome:
        """One T2V / keyframe I2V generation (keyframes on any frame 0..F-1),
        with the requested LoRAs and, for single-generate A2V, the source audio.

        ``source_audio_path`` is ``request.source_audio.audio_id`` resolved by
        the audio upload store (``PipelineManager.run_job``); ``None`` for every
        job without ``source_audio``.

        ``lora_paths`` is ``request.loras`` resolved by the registry (same
        order); only path and strength reach the worker (``audio_strength`` is
        ``None`` here, written back by ``validate_for_family``).
        ``reference_video_path`` and the outpaint and inpaint paths are
        accepted for the shared signature and never arrive filled: the fields
        that would produce them are refused by :func:`reject_unsupported`
        (re-checked here as the backstop).
        """
        if not self.loaded:
            self.load()

        reject_unsupported(request)
        _log_ignored(request)

        conditioning_image_paths = conditioning_image_paths or []
        mode = request.generation_mode

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "output.mp4"
        seed = int(seed) if seed is not None else resolve_seed(request.seed)

        images: list[dict] = []
        if mode == "i2v" and conditioning_image_paths:
            images = [
                {"path": str(path), "frame_idx": ci.frame_idx}
                for ci, path in zip(request.conditioning_images, conditioning_image_paths)
            ]

        lora_paths = lora_paths or []
        if len(lora_paths) != len(request.loras):
            raise RuntimeError(
                f"MiniMax H3: {len(request.loras)} LoRA(s) requested but "
                f"{len(lora_paths)} resolved"
            )
        loras = [
            {"path": str(lp.path), "strength": float(lp.strength)}
            for lp in lora_paths
        ]

        # crop_output: the worker writes the full-size mp4 to a temp file and
        # the app's ffmpeg helper centre-crops it into output.mp4 (as on LTX).
        target = output_dir / "_full.mp4" if request.crop_output is not None else output_path

        source_audio: dict | None = None
        if request.source_audio is not None:
            if source_audio_path is None:
                raise RuntimeError(
                    "MiniMax H3: source_audio が指定されていますが、音声ファイルのパスが"
                    "解決されていません（アプリ側の不具合です）。"
                )
            source_audio = {
                "path": str(source_audio_path),
                "strength": float(request.source_audio.strength),
            }

        if progress_callback:
            progress_callback(None, None, 0.05)

        payload: dict = {
            "op": "generate",
            "prompt": request.prompt,
            "seed": seed,
            "width": request.width,
            "height": request.height,
            "num_frames": request.num_frames,
            "frame_rate": request.frame_rate,
            "num_inference_steps": request.num_inference_steps,
            "shift_video": request.shift_video,
            "shift_audio": request.shift_audio,
            "images": images,
            # ALWAYS sent ([] = no LoRA).
            "loras": loras,
            # Per-job, ALWAYS sent: an absent key would mean "the load value".
            "attention_backend": request.attention_backend,
            "block_swap_prefetch": bool(request.block_swap_prefetch),
            # The default's source of truth is REQUEST_RULES.flag_defaults
            # (VRAM16_MODE_DEFAULT_H3), which validate_for_family writes back,
            # so None is not expected here; if it ever arrives it still means on.
            "vram16_mode": request.vram16_mode is not False,
        }
        # source_audio: only for single-generate A2V (an absent key = none).
        if source_audio is not None:
            payload["source_audio"] = source_audio
        payload["output_path"] = str(target)
        # keep_resident: the LTX contract — the key rides only when True, and
        # an absent key IS the release request.
        if request.keep_resident:
            payload["keep_resident"] = True

        with self._lock:
            try:
                self._send(payload)
            except Exception as exc:
                raise RuntimeError("MiniMax H3 worker died: " + self._stderr_tail()) from exc
            event = self._read_worker_events(
                progress_callback, chain=False, prefix="generate", one_stage=True
            )

        kind = event.get("event")
        if kind == "error":
            raise RuntimeError(event.get("detail", "MiniMax H3 worker generation failed"))
        if kind != "done":
            raise RuntimeError(f"MiniMax H3 worker returned unexpected event: {event!r}")

        if request.crop_output is not None:
            video_io.crop_mp4(
                target,
                output_path,
                request.crop_output.width,
                request.crop_output.height,
            )
            target.unlink(missing_ok=True)

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"MiniMax H3 worker produced no/empty output: {output_path}")

        if progress_callback:
            progress_callback(None, None, 0.90)
            progress_callback(None, None, 1.0)

        return GenerationOutcome(
            output_path=output_path,
            seed_used=event.get("seed_used", seed),
            peak_vram_mb=event.get("peak_vram_mb"),
            generation_mode=mode,
            backend=REAL_BACKEND_H3,
            attention_used=event.get("attention_used"),
            block_swap_prefetch_used=event.get("block_swap_prefetch_used"),
            keep_resident_used=event.get("keep_resident_used"),
            vram16_mode_used=event.get("vram16_mode_used"),
            peak_vram_reserved_mb=event.get("peak_vram_reserved_mb"),
            h3=_h3_block(event),
        )

    def generate_chain(self, chain_request, output_dir: Path, *args, **kwargs) -> GenerationOutcome:
        """Never reached through the API (``reject_chain`` refuses first)."""
        reject_chain(chain_request)
        raise AssertionError("unreachable")  # pragma: no cover


def _mock_channels_used(channels: int | None) -> str | None:
    """The worker's ``channels_used`` word (engine_h3/source_audio.py) for an
    ffprobed channel count: mono is duplicated, three or more keep the first two."""
    if not channels:
        return None
    if channels == 1:
        return "mono->stereo"
    if channels == 2:
        return "stereo"
    return "first2"


class _MockBackendH3(_MockBackend):
    """2.3's synthetic-clip mock plus the one H3-only input: ``source_audio``.

    The clip itself is the shared mock's. For an A2V job it adds the ``a2v``
    record the real worker reports in ``done.h3.a2v`` (so
    ``outcome.h3 == {"a2v": {...}}``, and a mock job's metadata carries an
    ``a2v`` block but no ``h3`` one). ``channels_used`` comes from an ffprobe of
    the upload; the latent-side keys the mock cannot know are ``None``. The mock
    does not mux audio (its mp4 is silent), so ``muxed_audio_samples`` is
    ``None``.
    """

    def generate(
        self,
        request: GenerateRequest,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        conditioning_image_paths: list[Path] | None = None,
        lora_paths: list[ResolvedLora] | None = None,
        reference_video_path: Path | None = None,
        seed: int | None = None,
        outpaint_source_path: Path | None = None,
        inpaint_source_path: Path | None = None,
        inpaint_mask_path: Path | None = None,
        source_audio_path: Path | None = None,
    ) -> GenerationOutcome:
        outcome = super().generate(
            request,
            output_dir=output_dir,
            progress_callback=progress_callback,
            conditioning_image_paths=conditioning_image_paths,
            lora_paths=lora_paths,
            reference_video_path=reference_video_path,
            seed=seed,
            outpaint_source_path=outpaint_source_path,
            inpaint_source_path=inpaint_source_path,
            inpaint_mask_path=inpaint_mask_path,
        )
        if request.source_audio is None:
            return outcome
        if source_audio_path is None:
            raise RuntimeError(
                "MiniMax H3 (mock): source_audio が指定されていますが、音声ファイルの"
                "パスが解決されていません（アプリ側の不具合です）。"
            )
        strength = float(request.source_audio.strength)
        _sample_rate, channels = video_io.probe_audio_stream(Path(source_audio_path))
        a2v = {
            "mask_value": 1.0 - strength,
            "audio_latents": None,
            "required_samples": None,
            "encoded_samples": None,
            "channels_used": _mock_channels_used(channels),
            "resampler": None,
            "encoder_tensors": None,
            "audio_output": "original" if strength == 1.0 else "decoded",
            "muxed_audio_samples": None,
            "mock": True,
        }
        return dataclasses.replace(outcome, h3={**(outcome.h3 or {}), "a2v": a2v})


class H3Runner(LTXRunner):
    """Facade for the MiniMax H3 engine family (the class attributes are the
    only per-family facts; everything else is ``LTXRunner``'s, except
    :meth:`generate`, which also forwards the H3-only ``source_audio_path``)."""

    _REAL_BACKEND_CLS = _RealBackendH3
    _MOCK_BACKEND_CLS = _MockBackendH3
    _MOCK_BACKEND_LABEL = MOCK_BACKEND_H3

    def generate(
        self,
        request: GenerateRequest,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        conditioning_image_paths: list[Path] | None = None,
        lora_paths: list[ResolvedLora] | None = None,
        reference_video_path: Path | None = None,
        seed: int | None = None,
        outpaint_source_path: Path | None = None,
        inpaint_source_path: Path | None = None,
        inpaint_mask_path: Path | None = None,
        source_audio_path: Path | None = None,
    ) -> GenerationOutcome:
        """``LTXRunner.generate`` plus ``source_audio_path`` (single-generate
        A2V's resolved upload; ``None`` otherwise), passed to the backend."""
        if self._backend is None or not self._backend.loaded:
            self.load()
        assert self._backend is not None
        return self._backend.generate(
            request,
            output_dir=output_dir,
            progress_callback=progress_callback,
            conditioning_image_paths=conditioning_image_paths,
            lora_paths=lora_paths,
            reference_video_path=reference_video_path,
            seed=seed,
            outpaint_source_path=outpaint_source_path,
            inpaint_source_path=inpaint_source_path,
            inpaint_mask_path=inpaint_mask_path,
            source_audio_path=source_audio_path,
        )
