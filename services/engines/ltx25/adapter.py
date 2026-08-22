"""LTX 2.5 engine adapter — the app-side half of the ``engine25`` worker (§3-98).

A SIBLING of :mod:`services.engines.ltx.adapter`, not a replacement. The two
engines share nothing at runtime: 2.5 runs official LTX-2 v1.2.0 with
transformers 5.x inside ``.venv-engine-ltx25``, which cannot coexist with 2.3's
transformers 4.57 in one environment. What they DO share is the app-side
plumbing — the ``@@LTX@@`` JSON-lines protocol, the subprocess spawn/read/kill
loop, the progress-receipt loop, the synthetic mock backend — so this module
subclasses the 2.3 adapter's classes through the seams introduced in P3a and
restates only the facts that are genuinely different:

* which venv and which ``python -m`` module the worker is (``engine25.worker``);
* which worker log file it writes (so BOTH logs survive a 2.3<->2.5 swap);
* which payload field each model-management category feeds;
* which child-process environment it gets (NONE of 2.3's ``LTX_*`` knobs — every
  one of them is read by 2.3's worker and would be a borrowed assumption here);
* the v1 feature scope: single two-stage T2V/I2V, everything else refused.

V1 SCOPE, STATED ONCE (owner ruling 2026-08-21). :data:`REJECT_TABLE` is the
422 half and :data:`IGNORED_FIELDS` the ignore-and-log half of the
``GenerateRequest`` field table; ``crop_output`` is deliberately in NEITHER —
it is an ffmpeg post-process the app applies to the finished mp4, so it is
engine-independent and simply works (see :meth:`_RealBackend25.generate`).

LIKE THE 2.3 ADAPTER, THIS FILE NEVER IMPORTS torch / ltx_* . They exist only
inside ``.venv-engine-ltx25``; the app venv has neither, and every engine
concern is deferred to the subprocess worker.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from api.errors import feature_unsupported, model_incompatible
from api.models import GenerateRequest
from config import AppConfig
from services import video_io
from services.engines.ltx.adapter import (
    _MockBackend,
    _RealBackend,
    GenerationOutcome,
    LTXRunner,
    LTX_ARCHITECTURE,
    ProgressCallback,
    _minor_version,
    resolve_seed,
)
from services.lora_registry import ResolvedLora

logger = logging.getLogger("ltx25.runner")

#: Model-management category -> the worker load-payload field it overrides.
#: THE SAME FOUR CATEGORY NAMES as LTX 2.3 (``transformer`` / ``text_encoder``
#: / ``video_vae`` / ``audio``) on purpose: the categories are the AXES a user
#: picks a file on, and those axes are a property of the product, not of an
#: engine generation. A second vocabulary would make "the transformer dropdown"
#: mean different keys per base model for no gain — and the runtime-state file
#: (state.json), which remembers a selection across a restart, is keyed by
#: exactly these names. Only the payload FIELDS differ, because engine25's
#: worker protocol is its own (see engine25/worker.py ``_LOAD_PATH_FIELDS``).
SELECTION_FIELDS: dict[str, str] = {
    "transformer": "transformer_path",
    "text_encoder": "text_encoder_path",
    "video_vae": "video_vae_path",
    "audio": "audio_vae_path",
}

#: Fixed (non-selectable) descriptor assets this engine needs.
#:
#: JUST THE SPATIAL UPSAMPLER. LTX 2.3 needs three (a tokenizer directory, the
#: upsampler and a separate text-projection file); 2.5 needs one, because the
#: other two have no counterpart here: the Gemma 4 tokenizer/processor metadata
#: travels INSIDE the text-encoder GGUF (engine25/assets_export.py exports it
#: to a sidecar next to the GGUF, and regenerates it when stale — nothing for
#: the app to point at), and 2.5 has no standalone text-projection file at all.
#:
#: THE AUDIO VAE IS DELIBERATELY NOT HERE. engine25's worker requires
#: ``audio_vae_path`` on every load, so it is unquestionably required — but it
#: arrives on the CATEGORY axis (``audio`` in :data:`SELECTION_FIELDS`), the
#: same way 2.3's does, which means it is already existence-checked by
#: ``LTXRunner._real_available`` through the descriptor's ``default_file`` and
#: is swappable from the models dropdown. Listing it here as well would demand
#: a SECOND declaration of the same file in the manifest (an ``assets`` entry
#: beside the category), i.e. two places to keep in sync for nothing.
REQUIRED_ASSETS: tuple[str, ...] = ("spatial_upsampler_path",)

#: LTX generation this adapter runs, as the first two segments of the
#: transformer GGUF's ``model_version`` KV. Mirror image of the 2.3 adapter's.
SUPPORTED_MODEL_VERSIONS: frozenset[str] = frozenset({"2.5"})

#: Real backend identifier surfaced in metadata.json. Distinct from 2.3's
#: ``"ltx-distilled"`` so an output file says which engine produced it.
REAL_BACKEND_25 = "ltx25-distilled"
#: Mock backend identifier. The mock CLASS is 2.3's (a synthetic gradient clip
#: says nothing about the engine that would have rendered it); only this label
#: differs, which is what lets the 2.3<->2.5 round-trip gate be checked from
#: metadata.json alone even when no GPU was involved.
MOCK_BACKEND_25 = "mock-ltx25"

#: Worker load defaults. ``blocks_on_gpu`` is the documented 16GB fallback
#: ladder's top rung (8 -> 6 -> 4) and is overridable through the existing
#: low-VRAM setting, so walking the ladder is a config change, not a code one.
DEFAULT_BLOCKS_ON_GPU = 8


# --------------------------------------------------------------------------- #
# v1 feature scope
# --------------------------------------------------------------------------- #

#: The 422 half of the GenerateRequest field table: ``(field, feature,
#: is_non_default)``. A machine-readable table rather than a chain of ``if``s
#: precisely so a pytest can hold it against ``GenerateRequest.model_fields``
#: and prove nothing was forgotten — a new request field that this engine
#: cannot honour must fail loudly, not ride through and be silently ignored.
#:
#: EVERY PREDICATE TESTS "DIFFERS FROM THE DEFAULT", never "is present": a
#: default-valued field was not asked for by the user (the frontend sends the
#: whole schema on every request), so rejecting it would make plain T2V
#: impossible.
REJECT_TABLE: tuple[tuple[str, str, Callable[[GenerateRequest], bool]], ...] = (
    ("pipeline", "two_stage_hq", lambda r: r.pipeline != "distilled"),
    ("outpaint", "outpaint", lambda r: r.outpaint is not None),
    ("loras", "loras", lambda r: bool(r.loras)),
    ("reference_video_id", "reference_video", lambda r: r.reference_video_id is not None),
    ("nag_enabled", "nag", lambda r: bool(r.nag_enabled)),
    ("vae_mode", "prune_vaed", lambda r: r.vae_mode != "default"),
    ("attention_backend", "sage_attention", lambda r: r.attention_backend != "sdpa"),
    ("keep_resident", "keep_resident", lambda r: bool(r.keep_resident)),
)

#: The ignore-and-log half: fields this engine cannot act on but that must NOT
#: fail a job, with the reason an operator sees in the log. Two kinds live here:
#: the distilled 2.5 schedule has no CFG and no step count to honour, and the
#: 2.3 acceleration knobs describe code paths engine25 simply does not have.
#: Refusing these would be hostile — they are ON by default (block_swap_prefetch,
#: fused_gguf_dequant_kernel), so a plain T2V request carries them.
IGNORED_FIELDS: dict[str, str] = {
    "negative_prompt": "LTX 2.5 distilled runs without classifier-free guidance",
    "guidance_scale": "LTX 2.5 distilled runs without classifier-free guidance",
    "num_inference_steps": "the distilled schedule is fixed at 8 + 3 sigmas",
    "neg_method": "no negative-prompt mechanism in the LTX 2.5 v1 scope",
    "vsf_scale": "no negative-prompt mechanism in the LTX 2.5 v1 scope",
    "fused_gguf_dequant_kernel": "2.3's Triton dequant kernel is not on this code path",
    "block_swap_prefetch": "engine25 uses its own block-swap window",
}

#: Everything GET /models publishes as this engine's ``unsupported_features``
#: (§3-98 Phase 5). The request-field features come from :data:`REJECT_TABLE`
#: so the two can never disagree; the chain-family names are added because they
#: are not request FIELDS at all — they are whole endpoints/modes, refused at
#: :meth:`_RealBackend25.generate_chain`, and the frontend needs their names to
#: grey out the Chained and Edit tabs.
UNSUPPORTED_FEATURES: tuple[str, ...] = (
    "chain",
    "retake",
    "end_source",
    "v2v",
    "a2v",
) + tuple(feature for _field, feature, _pred in REJECT_TABLE)


def reject_unsupported(request: GenerateRequest) -> None:
    """422 the first v1-out-of-scope field of ``request`` (§3-98 P3b).

    Called from :meth:`_RealBackend25.generate` — the adapter is the SOURCE OF
    TRUTH for what this engine can run, so the ruling lives next to the engine
    rather than in the endpoint. Phase 5 additionally calls it at the API layer
    so a rejection costs no worker round-trip and no job record; wiring it there
    changes nothing about the answer, only how early it arrives.

    First offender wins. Listing all of them would read as "fix these eight
    things" when in practice one control was left on.
    """
    for field, feature, is_non_default in REJECT_TABLE:
        if is_non_default(request):
            raise feature_unsupported(
                feature,
                detail=(
                    f"LTX 2.5(v1)は{feature}に対応していません"
                    f"(リクエストの{field}が既定値ではありません)。"
                    "この機能を使うにはベースモデルに「LTX 2.3」を選んでください。"
                ),
            )


def _log_ignored(request: GenerateRequest) -> None:
    """ONE log line naming every ignored field this request actually SET.

    Only non-default values are named: a default-valued field was not a choice
    the user made, and reporting all seven on every job would train the reader
    to skip the line. The default comes from the schema itself
    (``model_fields[...].default``) rather than a transcribed copy, so a
    changed default cannot make this lie.
    """
    fields = type(request).model_fields
    named = []
    for name, reason in IGNORED_FIELDS.items():
        spec = fields.get(name)
        if spec is None:
            continue
        value = getattr(request, name, None)
        if value != spec.default:
            named.append(f"{name}={value!r} ({reason})")
    if named:
        logger.info("LTX 2.5 ignores %d requested field(s): %s", len(named), "; ".join(named))


def check_kv(category: str, name: str, kv: dict[str, str]) -> None:
    """Rule on a transformer GGUF's KV header for the LTX 2.5 engine (§2.2).

    Same two-step contract, same missing-key-is-a-WARNING discipline and same
    "only the transformer is judged" rule as the 2.3 adapter's ``check_kv``;
    only :data:`SUPPORTED_MODEL_VERSIONS` differs. Reached only after
    ``services.engines.check_kv`` has already confirmed that the KV generation
    and the chosen base model's family AGREE, so the version branch below is a
    backstop for the cases that dispatcher cannot rule on (a file whose
    ``model_version`` is neither 2.3 nor 2.5 — a future generation, or a typo).
    """
    if category != "transformer":
        return
    architecture = (kv.get("general.architecture") or "").strip()
    if not architecture:
        logger.warning(
            "model '%s' declares no general.architecture in its GGUF header; "
            "loading it anyway (the engine's own loader has the last word).",
            name,
        )
    elif architecture != LTX_ARCHITECTURE:
        raise model_incompatible(
            category,
            name,
            detail=(
                f"'{architecture}'系のモデルです。LTXエンジンは"
                f"'{LTX_ARCHITECTURE}'のみ扱えます。"
            ),
        )
    version = (kv.get("model_version") or "").strip()
    if not version:
        logger.warning(
            "model '%s' declares no model_version in its GGUF header; loading "
            "it anyway (assuming it matches this engine's LTX generation).",
            name,
        )
        return
    if _minor_version(version) not in SUPPORTED_MODEL_VERSIONS:
        raise model_incompatible(
            category,
            name,
            detail=(
                f"このtransformerはltxv {version}です。LTX 2.5エンジンが扱えるのは"
                f"ltxv {'/'.join(sorted(SUPPORTED_MODEL_VERSIONS))}系のみです。"
            ),
        )


# --------------------------------------------------------------------------- #
# backend
# --------------------------------------------------------------------------- #


class _RealBackend25(_RealBackend):
    """The ``engine25.worker`` subprocess, driven through 2.3's process plumbing.

    Inherited unchanged: ``load`` (spawn -> send payload -> await ``ready``),
    ``unload`` (shutdown -> wait -> terminate -> kill), ``_send`` /
    ``_read_event`` / ``_read_worker_events`` / ``_stderr_tail``, and the
    descriptor-path helpers. Overridden below: everything that is a fact about
    WHICH engine is on the other end of the pipe.
    """

    SELECTION_FIELDS = SELECTION_FIELDS
    REQUIRED_ASSETS = REQUIRED_ASSETS
    _WORKER_MODULE = "engine25.worker"
    _LOG_NAME = "ltx25_worker.log"
    #: Fixed, not configurable: the engine25 package ships in this repository,
    #: so there is no installation choice to expose (unlike the interpreter).
    _ENGINE_DIR_VALUE = "./engine25"
    _ENGINE_PYTHON_LABEL = "engine_python_ltx25"

    @classmethod
    def _engine_python_value(cls, config: AppConfig) -> str | None:
        return config.model.engine_python_ltx25

    # ------------------------------------------------------------------ load

    def _build_child_env(self, project_root: Path) -> dict[str, str]:
        """The engine25 worker's environment — deliberately MINIMAL.

        NONE of the 2.3 worker's ``LTX_*`` variables are set here. Each of them
        (``LTX_COMPONENT_FILES``, ``LTX_TE_OFFLOAD``, ``LTX_DIT_CPU_LOAD``) names
        a code path inside ``engine/``; engine25 has its own offload and
        block-swap mechanics and reads none of them, so passing them would be
        decoration at best and a misleading log at worst. The 2.5 equivalents
        (``blocks_on_gpu`` / ``te_layers_on_gpu`` / ``cache_weights``) ride the
        LOAD PAYLOAD instead, where they are visible in the protocol.

        What IS kept is the process hygiene the two share: the expandable-
        segments allocator (16GB is the whole point), torch.compile off,
        unbuffered IO, and PYTHONPATH pinned to the project root so
        ``python -m engine25.worker`` resolves the first-party package.
        """
        env = dict(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("PYTHONPATH", None)
        env["PYTHONPATH"] = str(project_root)
        return env

    def _log_load_start(self, engine_python: str, engine_dir: Path, payload: dict) -> None:
        logger.info(
            "Loading pipeline (REAL engine25 worker). python=%s engine_dir=%s "
            "blocks_on_gpu=%s cache_weights=%s deterministic=%s",
            engine_python,
            engine_dir,
            payload["blocks_on_gpu"],
            payload["cache_weights"],
            payload["deterministic"],
        )

    def _build_load_payload(self, selection: dict[str, str] | None = None) -> dict:
        """Build the ``{"op": "load"}`` payload engine25's worker expects (pure).

        The five weight paths come from the base-model descriptor exactly as
        2.3's do — a selection override wins, otherwise the category's
        ``default_file`` — with :data:`SELECTION_FIELDS` again read rather than
        transcribed, so a category missing from that table cannot reach the
        worker at all.

        NOT SENT: ``text_encoder_assets_path``. The assets-only safetensors
        export (the ~30MB metadata twin the official ``GemmaAssets.load`` opens
        before the GGUF weights are swapped in) is DERIVED from the text-encoder
        GGUF and lives beside it; engine25's pipeline regenerates it whenever it
        is missing or stale (engine25/assets_export.ensure_assets_only). Naming
        it from here would add a path the app has to keep correct in order to
        express "the default location" — the field exists in the protocol only
        for a deployment that keeps the two apart.

        Key set AND insertion order are the byte contract, pinned by the golden
        snapshot in tests/test_ltx25_load_payload.py — do not reorder.
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
            "spatial_upsampler_path": self._require_asset("spatial_upsampler_path"),
            # The 16GB ladder's current rung. ``or`` (not a None check) matches
            # the 2.3 adapter: 0 means "unset" in this setting, not "zero blocks".
            "blocks_on_gpu": self.low_vram.block_swap_blocks_on_gpu or DEFAULT_BLOCKS_ON_GPU,
            # +14.7GB of host RAM to avoid re-reading the transformer from disk
            # for stage 2 (the official pipeline disposes the stage-1 weights,
            # meta-ing the buffers). Owner ruling H改: on by default, with the
            # RAM requirement documented; a RAM-tight machine turns it off.
            "cache_weights": True,
            # cuDNN algorithm selection pinned, so a same-seed rerun is
            # bit-identical (the audio vocoder is what varies otherwise). The
            # determinism gate H6 rests on this being ON by default.
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
    ) -> GenerationOutcome:
        """One two-stage T2V/I2V generation (the whole v1 scope).

        ``lora_paths`` / ``reference_video_path`` / ``outpaint_source_path`` are
        accepted so the call shape stays identical to 2.3's — the orchestrator
        passes them positionally — but reaching this method with any of them set
        is already impossible: the request fields behind them are refused by
        :func:`reject_unsupported` on the line above.

        ``crop_output`` IS honoured, and is the one v1-scope decision worth
        naming: it is an ffmpeg centre-crop the app performs on the finished
        mp4, so it is engine-independent and a 2.5 job gets the same non-64
        display sizes a 2.3 job does. Silently dropping it would have been the
        easy mistake.
        """
        if not self.loaded:
            self.load()

        reject_unsupported(request)
        _log_ignored(request)

        conditioning_image_paths = conditioning_image_paths or []
        mode = request.generation_mode

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "output.mp4"

        # Resolved in the PARENT so seed_used is deterministic regardless of the
        # worker, exactly as in the 2.3 backend.
        seed = int(seed) if seed is not None else resolve_seed(request.seed)

        images: list[dict] = []
        if mode == "i2v" and conditioning_image_paths:
            images = [
                {"path": str(path), "frame_idx": ci.frame_idx, "strength": ci.strength}
                for ci, path in zip(request.conditioning_images, conditioning_image_paths)
            ]

        # crop_output: the worker writes the full-size mp4 to a temp file and the
        # existing ffmpeg helper centre-crops it into output.mp4.
        target = output_dir / "_full.mp4" if request.crop_output is not None else output_path

        if progress_callback:
            progress_callback(None, None, 0.05)

        # The v1 contract, whole. Fields this engine ignores are NOT forwarded:
        # they were already logged above, and a payload that carries only what
        # is acted upon is a payload a golden snapshot can pin.
        payload: dict = {
            "op": "generate",
            "prompt": request.prompt,
            "seed": seed,
            "width": request.width,
            "height": request.height,
            "num_frames": request.num_frames,
            "frame_rate": request.frame_rate,
            "images": images,
            "output_path": str(target),
        }

        with self._lock:
            try:
                self._send(payload)
            except Exception as exc:
                raise RuntimeError("LTX 2.5 worker died: " + self._stderr_tail()) from exc
            event = self._read_worker_events(progress_callback, chain=False, prefix="generate")

        kind = event.get("event")
        if kind == "error":
            raise RuntimeError(event.get("detail", "LTX 2.5 worker generation failed"))
        if kind != "done":
            raise RuntimeError(f"LTX 2.5 worker returned unexpected event: {event!r}")

        if request.crop_output is not None:
            video_io.crop_mp4(
                target,
                output_path,
                request.crop_output.width,
                request.crop_output.height,
            )
            target.unlink(missing_ok=True)

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"LTX 2.5 worker produced no/empty output: {output_path}")

        if progress_callback:
            progress_callback(None, None, 0.90)
            progress_callback(None, None, 1.0)

        return GenerationOutcome(
            output_path=output_path,
            seed_used=event.get("seed_used", seed),
            peak_vram_mb=event.get("peak_vram_mb"),
            generation_mode=mode,
            backend=REAL_BACKEND_25,
            # The acceleration relay fields stay None: every one of them names a
            # 2.3 code path this engine does not have, and reporting "off" would
            # claim the knob exists here and was left alone. attention_used is
            # the exception worth stating positively — v1 is SDPA-only by scope.
            attention_used="sdpa",
            peak_vram_reserved_mb=event.get("peak_vram_reserved_mb"),
        )

    def generate_chain(self, chain_request, output_dir: Path, **kwargs) -> GenerationOutcome:
        """Refused: the whole chain family is out of the LTX 2.5 v1 scope.

        Chain, retake, end source, V2V continuation and A2V all arrive through
        this ONE method, so one refusal covers them; engine25's worker carries
        the same refusal as a backstop for a payload that arrives another way.
        """
        raise feature_unsupported(
            "chain",
            detail=(
                "LTX 2.5(v1)は連結生成(Chained・Retake・End source・V2V・A2V)に"
                "対応していません。ベースモデルに「LTX 2.3」を選んでください。"
            ),
        )


class LTX25Runner(LTXRunner):
    """Facade for the LTX 2.5 engine family.

    Everything the base facade does — lazy descriptor resolution, mock/real
    selection, ``set_descriptor`` teardown, the availability probe and its
    missing-file report — applies unchanged; the class attributes below are the
    only per-family facts, and they are exactly the seams P3a introduced.
    """

    _REAL_BACKEND_CLS = _RealBackend25
    _MOCK_BACKEND_CLS = _MockBackend
    _MOCK_BACKEND_LABEL = MOCK_BACKEND_25

    @property
    def sage_available(self) -> bool:
        """Always False, and expected to stay False.

        Not a probe result: SageAttention is not installed in
        ``.venv-engine-ltx25`` and v1 is SDPA-only by scope decision, so running
        the base class's file-existence probe would be asking a question whose
        answer is already fixed — and, worse, it would answer it by looking in
        the wrong venv's site-packages if the two ever diverged.
        """
        return False
