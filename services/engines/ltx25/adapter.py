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
* the feature scope: single two-stage T2V/I2V and the plain Chained clip
  chain; everything else refused.

V1 SCOPE, STATED ONCE (owner ruling 2026-08-21). :data:`REJECT_TABLE` is the
422 half and :data:`IGNORED_FIELDS` the ignore-and-log half of the
``GenerateRequest`` field table; ``crop_output`` is deliberately in NEITHER —
it is an ffmpeg post-process the app applies to the finished mp4, so it is
engine-independent and simply works (see :meth:`_RealBackend25.generate`).

THE CHAIN SCHEMA HAS ITS OWN FOUR TABLES (``CHAIN_*``, §3-102). The rules are
identical; the schemas are not, so one table forced to serve both would need a
per-schema exception list — the very thing the audit tests exist to prevent.

LIKE THE 2.3 ADAPTER, THIS FILE NEVER IMPORTS torch / ltx_* . They exist only
inside ``.venv-engine-ltx25``; the app venv has neither, and every engine
concern is deferred to the subprocess worker.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import chain_math
from api.errors import feature_unsupported, model_incompatible
from api.models import GenerateChainRequest, GenerateRequest
from config import AppConfig
from services import video_io
from services.engines.ltx.adapter import (
    _MockBackend,
    _RealBackend,
    GenerationOutcome,
    LTXRunner,
    LTX_ARCHITECTURE,
    ProgressCallback,
    _lora_payload_entry,
    _minor_version,
    _resolve_reference_preprocess,
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

#: Fields this engine ACTS ON. Six of them ride the generate payload verbatim
#: (see :meth:`_RealBackend25.generate`), ``conditioning_images`` becomes the
#: ``images`` list, and ``crop_output`` is the app-side ffmpeg centre-crop that
#: happens after the worker is done. Declared rather than merely implied so the
#: model_fields audit below can name a home for every field.
#:
#: ``loras`` / ``reference_video_id`` AND THEIR TWO STRENGTHS JOINED THIS SET
#: with §3-102's third increment (Style LoRA + IC-LoRA). None of the four is
#: read off the request the way the six above are: the orchestrator has already
#: resolved the adapter NAMES into files and the upload ID into a path, so what
#: :meth:`_RealBackend25.generate` acts on is the ``lora_paths`` /
#: ``reference_video_path`` keyword arguments carrying that material. The two
#: strengths ARE read from the request, and only shape the ``reference_video``
#: block. See the needle table in tests/test_ltx25_adapter.py.
HONOURED_FIELDS: frozenset[str] = frozenset(
    {
        "prompt",
        "width",
        "height",
        "num_frames",
        "frame_rate",
        "seed",
        "conditioning_images",
        "crop_output",
        "loras",
        "reference_video_id",
        "conditioning_attention_strength",
        "reference_video_strength",
    }
)

#: ``field -> the field that governs it``. These are SUB-PARAMETERS: each one is
#: inert unless its governor is non-default, and every governor here is in
#: :data:`REJECT_TABLE`. So they need no ruling of their own — a request that
#: makes one of them meaningful is already refused, by the governor, before this
#: field could have mattered.
#:
#: THE TWO REFERENCE STRENGTHS LEFT THIS TABLE with §3-102's third increment.
#: They were only ever here because their governor (``loras``) was a 422; now
#: that LoRA and the reference video run on this engine, a strength really does
#: change the job, so it is HONOURED — and leaving it classified "governed"
#: would have been the silent-drop the audit exists to catch.
#:
#: The distinction is worth keeping rather than folding into
#: :data:`IGNORED_FIELDS`: "ignored" promises a job runs anyway, which is false
#: here — these cannot reach a running job at all.
GOVERNED_FIELDS: dict[str, str] = {
    "nag_scale": "nag_enabled",
    "nag_tau": "nag_enabled",
    "nag_alpha": "nag_enabled",
}

# --------------------------------------------------------------------------- #
# chain feature scope (§3-102 第1段: Chained本体)
# --------------------------------------------------------------------------- #
#
# THE SAME FOUR-WAY CLASSIFICATION as the single-``/generate`` table above,
# applied to :class:`GenerateChainRequest`. A SECOND set of tables rather than a
# reuse of the first, because the two schemas answer differently for the same
# field name: ``num_inference_steps`` is ignored on both, but ``clips`` /
# ``overlap_frames`` / ``stage2_window`` exist only here, and ``outpaint``
# exists only there. One table forced to serve both would need a per-schema
# exception list, which is the thing the audit test is meant to make impossible.

#: The 422 half of the ``GenerateChainRequest`` field table: ``(field, feature,
#: is_non_default)``. Same predicate discipline as :data:`REJECT_TABLE` — every
#: one tests "DIFFERS FROM THE DEFAULT", never "is present", because the
#: frontend sends the whole schema on every request.
#:
#: The first two are whole MODES layered on top of a chain (Retake, End
#: source); the rest are the same engine-level features the single path
#: refuses. ``outpaint`` has no counterpart here — the chain schema has no such
#: field at all.
#:
#: ``source_video`` AND ``source_audio`` LEFT THIS TABLE with §3-102's second
#: increment: V2V continuation and A2V (Single, long and Batch alike) run on
#: this engine now, so both moved to :data:`CHAIN_HONOURED_FIELDS`. ``loras``
#: and ``reference_video_id`` LEFT WITH THE THIRD: Style/character LoRA and the
#: reference-video control IC-LoRA (long chains included) run here too. Retake
#: and End source stay — they are the modes engine25 still has no code path
#: for.
CHAIN_REJECT_TABLE: tuple[
    tuple[str, str, Callable[[GenerateChainRequest], bool]], ...
] = (
    ("retake", "retake", lambda r: r.retake is not None),
    ("end_source", "end_source", lambda r: r.end_source is not None),
    ("nag_enabled", "nag", lambda r: bool(r.nag_enabled)),
    ("pipeline", "two_stage_hq", lambda r: r.pipeline != "distilled"),
    ("vae_mode", "prune_vaed", lambda r: r.vae_mode != "default"),
    ("attention_backend", "sage_attention", lambda r: r.attention_backend != "sdpa"),
    ("keep_resident", "keep_resident", lambda r: bool(r.keep_resident)),
)

#: The ignore-and-log half. Field-for-field the same seven as
#: :data:`IGNORED_FIELDS` and for the same two reasons (the distilled schedule
#: has no CFG and no step count to honour; the 2.3 acceleration knobs name code
#: paths engine25 does not have) — spelled out rather than aliased so the audit
#: test reads one schema against one table.
CHAIN_IGNORED_FIELDS: dict[str, str] = {
    "negative_prompt": "LTX 2.5 distilled runs without classifier-free guidance",
    "guidance_scale": "LTX 2.5 distilled runs without classifier-free guidance",
    "num_inference_steps": "the distilled schedule is fixed at 8 + 3 sigmas",
    "neg_method": "no negative-prompt mechanism in the LTX 2.5 chain scope",
    "vsf_scale": "no negative-prompt mechanism in the LTX 2.5 chain scope",
    "fused_gguf_dequant_kernel": "2.3's Triton dequant kernel is not on this code path",
    "block_swap_prefetch": "engine25 uses its own block-swap window",
}

#: Fields the chain path ACTS ON. Eight ride the worker payload verbatim (see
#: :meth:`_RealBackend25.generate_chain`), ``prompt`` and ``clips`` together
#: become the per-clip list (effective prompt / num_frames / clip-0 images), and
#: ``crop_output`` is the app-side ffmpeg centre-crop applied to the finished
#: mp4 — the same engine-independent ruling the single path makes.
#:
#: ``source_video`` and ``source_audio`` are honoured INDIRECTLY, exactly as on
#: 2.3: the orchestrator resolves each upload id into material (the fps-correct
#: ``_source_tail.mp4`` cut for V2V, the uploaded wav for A2V) and hands it to
#: :meth:`_RealBackend25.generate_chain` as keyword arguments, which become the
#: additive ``source`` / ``audio_source`` payload blocks. Naming the REQUEST
#: fields here is what the audit needs — they are the fields the schema has.
#:
#: ``num_inference_steps`` is deliberately NOT here even though the payload
#: carries a ``num_steps`` key built from it: the distilled schedule is fixed,
#: so the engine records the number in metadata and denoises 8 + 3 steps
#: regardless. "Honoured" would claim it changes the output; it does not.
#:
#: ``loras`` / ``reference_video_id`` AND THEIR TWO STRENGTHS JOINED THIS SET
#: with §3-102's third increment, for the same reason and in the same shape as
#: the single path's :data:`HONOURED_FIELDS`: the orchestrator resolves the
#: adapter names and the upload id into material, and the payload carries the
#: additive ``loras`` / ``reference_video`` blocks below. A long reference is
#: sliced per stage-1 segment by the ENGINE; the app recomputes the same windows
#: from ``chain_math`` for metadata, so nothing about that geometry is decided
#: here.
CHAIN_HONOURED_FIELDS: frozenset[str] = frozenset(
    {
        "prompt",
        "clips",
        "width",
        "height",
        "crop_output",
        "frame_rate",
        "seed",
        "overlap_frames",
        "overlap_strength",
        "chunked_upsample",
        "stage2_window",
        "source_video",
        "source_audio",
        "loras",
        "reference_video_id",
        "conditioning_attention_strength",
        "reference_video_strength",
    }
)

#: ``field -> the field that governs it``, exactly as :data:`GOVERNED_FIELDS`:
#: each is inert unless its governor is non-default, and every governor here is
#: in :data:`CHAIN_REJECT_TABLE`, so a request that could make one of them
#: matter is already refused before this field is read.
#:
#: THE TWO REFERENCE STRENGTHS LEFT THIS TABLE with §3-102's third increment,
#: the chain twin of the single-path move: their governor is no longer a 422, so
#: "governed" would now be a promise that they cannot reach a running job — and
#: they can.
CHAIN_GOVERNED_FIELDS: dict[str, str] = {
    "nag_scale": "nag_enabled",
    "nag_tau": "nag_enabled",
    "nag_alpha": "nag_enabled",
}


#: Everything GET /models publishes as this engine's ``unsupported_features``
#: (§3-98 Phase 5). The request-field features come from :data:`REJECT_TABLE`
#: so the two can never disagree; the four chain-family MODE names are added
#: because they are not single-request FIELDS at all, and the frontend needs
#: their names to grey out the Edit tab and the Chained tab's mode panels.
#:
#: ``"chain"`` LEFT THIS TUPLE with §3-102's first increment (a plain Chained
#: job runs on this engine, so publishing "no chain" would grey out a tab that
#: works), and ``"v2v"`` / ``"a2v"`` left with the second: V2V continuation and
#: A2V run here too, which also lights the Single tab's A2V accordion and the
#: Batch tab's A2V rows, because the frontend greys all three by these names.
#: ``"loras"`` and ``"reference_video"`` left with the THIRD increment, which
#: re-opens the LoRA chips and the reference-video panel on both tabs.
#: The two modes that still cannot run — retake / end_source — stay, and they
#: are enforced field-by-field by :data:`CHAIN_REJECT_TABLE` rather than by one
#: blanket refusal.
UNSUPPORTED_FEATURES: tuple[str, ...] = (
    "retake",
    "end_source",
) + tuple(feature for _field, feature, _pred in REJECT_TABLE)


def reject_unsupported(request: GenerateRequest) -> None:
    """422 the first v1-out-of-scope field of ``request`` (§3-98 P3b).

    Called from :meth:`_RealBackend25.generate` — the adapter is the SOURCE OF
    TRUTH for what this engine can run, so the ruling lives next to the engine
    rather than in the endpoint. Phase 5 additionally calls it at the API layer
    so a rejection costs no worker round-trip and no job record; wiring it there
    changes nothing about the answer, only how early it arrives.

    First offender wins. Listing all of them would read as "fix these six
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


def reject_chain(request: GenerateChainRequest) -> None:
    """422 the first out-of-scope field of a chain ``request`` (§3-102).

    UNTIL §3-102 THIS TOOK NO ARGUMENT and refused every chain outright, on the
    grounds that nothing in the body could make this engine able to chain. That
    is no longer true: a plain Chained job runs here now, and what is left out
    of scope — Retake, End source, NAG and the acceleration knobs — is decided
    FIELD BY FIELD, exactly like the single path. So the request has to be
    read.

    Same first-offender-wins rule and same table discipline as
    :func:`reject_unsupported`; see :data:`CHAIN_REJECT_TABLE`.

    Lives at module level (like :func:`reject_unsupported`) so the API layer can
    refuse before a job record is created, while the backend method keeps it as
    the backstop for a payload that arrives another way.
    """
    for field, feature, is_non_default in CHAIN_REJECT_TABLE:
        if is_non_default(request):
            raise feature_unsupported(
                feature,
                detail=(
                    f"LTX 2.5の連結生成(Chained)は{feature}に対応していません"
                    f"(リクエストの{field}が既定値ではありません)。"
                    "この機能を使うにはベースモデルに「LTX 2.3」を選んでください。"
                ),
            )


def _log_ignored(request, table: dict[str, str] | None = None) -> None:
    """ONE log line naming every ignored field this request actually SET.

    Only non-default values are named: a default-valued field was not a choice
    the user made, and reporting all seven on every job would train the reader
    to skip the line. The default comes from the schema itself
    (``model_fields[...].default``) rather than a transcribed copy, so a
    changed default cannot make this lie.

    ``table`` selects which of the two ignore tables to read
    (:data:`IGNORED_FIELDS` for a single request, :data:`CHAIN_IGNORED_FIELDS`
    for a chain). One function for both because the RULE is identical and only
    the table differs — a second copy would be a second place to forget the
    "name only what was actually set" discipline.
    """
    fields = type(request).model_fields
    named = []
    for name, reason in (IGNORED_FIELDS if table is None else table).items():
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
        """One two-stage T2V/I2V generation, with Style / IC-LoRA (§3-102).

        ``lora_paths`` and ``reference_video_path`` ARE ACTED ON since §3-102's
        third increment: the orchestrator resolved the adapter names into
        safetensors files and the upload id into a path, and both ride the
        payload in 2.3's shape (see the ``loras`` / ``reference_video`` keys
        below). ``outpaint_source_path`` is still accepted-and-unused so the
        call shape stays identical to 2.3's — the orchestrator passes all of
        them positionally — but reaching this method with it set is impossible:
        ``outpaint`` is refused by :func:`reject_unsupported` on the line above.

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
        lora_paths = lora_paths or []
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

        # Style/character AND control IC-LoRA (§3-102 third increment), in 2.3's
        # shape verbatim — ``_lora_payload_entry`` is IMPORTED from the 2.3
        # adapter rather than restated, because the (path, strength[,
        # audio_strength]) triple is the app's contract with a resolved adapter,
        # not a fact about either engine. ``loras`` is ALWAYS a key (an empty
        # list means "detach whatever was attached", which the worker must be
        # told explicitly), and so is ``reference_video`` (None = no reference).
        #
        # ``preprocess`` is derived from the job's adapters; a conflict (>1
        # distinct kind) is already refused by api/generate.py, and this is the
        # defensive re-check at the runner hop, exactly as on 2.3. The reference
        # ``strength`` defaults to 1.0 (official guidance) and
        # ``attention_strength`` is spliced in ONLY when the request set it, so
        # an omitted-field job's block stays byte-identical.
        loras_payload = [_lora_payload_entry(lp) for lp in lora_paths]
        if reference_video_path is not None:
            reference_payload: dict | None = {
                "path": str(reference_video_path),
                "strength": (
                    1.0
                    if request.reference_video_strength is None
                    else float(request.reference_video_strength)
                ),
                "preprocess": _resolve_reference_preprocess(lora_paths),
            }
            if request.conditioning_attention_strength is not None:
                reference_payload["attention_strength"] = float(
                    request.conditioning_attention_strength
                )
        else:
            reference_payload = None

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
            "loras": loras_payload,
            "reference_video": reference_payload,
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

    def generate_chain(
        self,
        chain_request,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        clip0_conditioning_paths: list[Path] | None = None,
        source_tail_path: Path | None = None,
        source_context_frames: int | None = None,
        source_audio_path: Path | None = None,
        retake_window_path: Path | None = None,
        end_source_path: Path | None = None,
        end_source_context_frames: int | None = None,
        end_source_strength: float | None = None,
        lora_paths: list[ResolvedLora] | None = None,
        reference_video_path: Path | None = None,
        seed: int | None = None,
    ) -> GenerationOutcome:
        """One masked AV-latent clip chain via engine25's ``generate_chain`` op.

        THE CALL SHAPE IS 2.3's, VERBATIM — ``services/pipeline_manager.py``
        ``run_chain_job`` passes all thirteen keywords to whichever runner is
        active, and a signature that dropped the ones this engine cannot serve
        would make the orchestrator branch on the engine family. It does not,
        and must not: the ORCHESTRATOR's job is to prepare material, the
        ADAPTER's job is to rule on it.

        FOUR of those keywords name a mode outside this engine's scope (Retake
        and End source). Reaching this method with any of them set is already
        impossible — :func:`reject_chain` refuses the request fields behind them
        at the endpoint, and again on the line below — so a non-``None`` arrival
        means the two tables have drifted apart, which is a bug worth a loud
        ``RuntimeError`` rather than a silently ignored argument.

        HONOURED, and worth naming: ``clip0_conditioning_paths`` (clip 0's I2V
        keyframes — the only clip the schema lets carry them) and
        ``crop_output``, the app-side ffmpeg centre-crop of the finished mp4,
        which is engine-independent for a chain exactly as it is for a single
        job.

        ALSO HONOURED SINCE §3-102's SECOND INCREMENT: ``source_tail_path`` +
        ``source_context_frames`` (V2V continuation) and ``source_audio_path``
        (A2V). All three are material the ORCHESTRATOR prepared — the tail cut
        to the requested fps, the uploaded wav — and they ride the payload as
        the additive ``source`` / ``audio_source`` blocks below, in 2.3's shape
        so the two engines' chain payloads stay comparable.

        ALSO HONOURED SINCE THE THIRD: ``lora_paths`` (Style/character and
        control adapters, applied uniformly across the chain) and
        ``reference_video_path`` (the ONE reference video, sliced per stage-1
        segment by the engine). Both ride as ADDITIVE blocks — non-empty /
        non-``None`` only — so a plain chain's payload stays byte-identical to
        the golden, which is 2.3's discipline for the same two keys.
        """
        chain = chain_request
        # BEFORE the load, unlike :meth:`generate`. The ruling is a pure read of
        # the request, so spawning a worker first would only mean a doomed
        # request pays for a model load — and it keeps this refusal reachable
        # without a subprocess, which is what lets a pytest hold it.
        reject_chain(chain)
        _log_ignored(chain, CHAIN_IGNORED_FIELDS)

        # Fail loud, not silent: every one of these is refused above, so a value
        # here means the reject table and this signature disagree about what the
        # engine can do. Four entries, not six: ``lora_paths`` and
        # ``reference_video_path`` LEFT this guard with §3-102's third increment
        # — they are material this engine now consumes, so their arrival is a
        # job, not a drift.
        out_of_scope = {
            "retake_window_path": retake_window_path,
            "end_source_path": end_source_path,
            "end_source_context_frames": end_source_context_frames,
            "end_source_strength": end_source_strength,
        }
        supplied = [name for name, value in out_of_scope.items() if value is not None]
        if supplied:
            raise RuntimeError(
                "LTX 2.5 chain received out-of-scope material the feature table "
                f"should have refused: {', '.join(sorted(supplied))}"
            )

        if not self.loaded:
            self.load()

        clip0_conditioning_paths = clip0_conditioning_paths or []
        lora_paths = lora_paths or []
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "output.mp4"

        # Resolved in the PARENT so seed_used is deterministic regardless of the
        # worker, exactly as in the single path and in the 2.3 chain backend.
        seed = int(seed) if seed is not None else resolve_seed(chain.seed)

        # Only clip 0 may carry conditioning images (the schema validator
        # enforces that), so every other clip's list is empty by construction.
        clip0_images: list[dict] = []
        if chain.clips[0].conditioning_images and clip0_conditioning_paths:
            clip0_images = [
                {"path": str(path), "frame_idx": ci.frame_idx, "strength": ci.strength}
                for ci, path in zip(chain.clips[0].conditioning_images, clip0_conditioning_paths)
            ]
        clips_payload = [
            {
                "prompt": chain.clip_prompt(i),
                "num_frames": chain.clips[i].num_frames,
                "images": clip0_images if i == 0 else [],
            }
            for i in range(len(chain.clips))
        ]

        # crop_output: the worker writes the full-size mp4 to a temp file and the
        # existing ffmpeg helper centre-crops it into output.mp4.
        target = (output_dir / "_full.mp4") if chain.crop_output is not None else output_path

        if progress_callback:
            progress_callback(None, None, 0.03)

        # The chain contract, whole. The key SET and ORDER are the byte contract
        # (golden snapshot in tests/test_ltx25_adapter.py) and they deliberately
        # match 2.3's chain op: the two workers are unrelated code, but the body
        # of a chain job is the same geometry in both, and a needlessly different
        # key set would make the two engines' chain metadata incomparable.
        #
        # ``num_steps`` rides even though the distilled schedule is fixed — the
        # engine records it in metadata and denoises 8 + 3 regardless, which is
        # why num_inference_steps is classified ignore-and-log, not honoured.
        payload: dict = {
            "op": "generate_chain",
            "width": chain.width,
            "height": chain.height,
            "frame_rate": chain.frame_rate,
            "num_steps": chain.num_inference_steps,
            "seed": seed,
            "overlap_frames": int(chain.overlap_frames),
            "overlap_strength": float(chain.overlap_strength),
            "chunked_upsample": bool(chain.chunked_upsample),
            "output_path": str(target),
            "clips": clips_payload,
        }
        # V2V continuation (additive): the app-cut fps-correct source tail. The
        # ``is not None`` guard on BOTH values is what keeps a plain chain's
        # payload key set byte-identical to the golden above — the same
        # discipline, and the same key names and key ORDER, as 2.3's.
        if source_tail_path is not None and source_context_frames is not None:
            payload["source"] = {
                "path": str(source_tail_path),
                "context_frames": int(source_context_frames),
            }
        # A2V (additive): the uploaded audio path, passed as-is — the engine
        # truncates the encoded latent to the timeline. Absent for a plain or a
        # V2V chain (the schema makes the two source modes mutually exclusive).
        if source_audio_path is not None:
            payload["audio_source"] = {"path": str(source_audio_path)}
        # Style/character AND control IC-LoRA (additive): (path, strength[,
        # audio_strength]) per adapter, applied uniformly across the chain, and
        # sent ONLY when non-empty so a no-lora chain's payload is byte-identical
        # to the golden above. ``preprocess`` is deliberately NOT part of an
        # entry — it is derived once for the reference block below, and a control
        # adapter without a reference is already refused at the API layer.
        if lora_paths:
            payload["loras"] = [_lora_payload_entry(lp) for lp in lora_paths]
        # Reference-video CONTROL IC-LoRA (additive): the ONE reference video,
        # which the engine slices per stage-1 segment. Same block shape as the
        # single path's, and present only when a reference was requested.
        if reference_video_path is not None:
            reference_payload: dict = {
                "path": str(reference_video_path),
                "strength": (
                    1.0
                    if chain.reference_video_strength is None
                    else float(chain.reference_video_strength)
                ),
                "preprocess": _resolve_reference_preprocess(lora_paths),
            }
            if chain.conditioning_attention_strength is not None:
                reference_payload["attention_strength"] = float(
                    chain.conditioning_attention_strength
                )
            payload["reference_video"] = reference_payload
        # stage2_window: additive, sent ONLY when the request opted off
        # "standard", so a default chain's payload stays byte-identical to the
        # golden key set above (same contract as 2.3's).
        if chain.stage2_window != chain_math.STAGE2_WINDOW_DEFAULT:
            payload["stage2_window"] = chain.stage2_window

        with self._lock:
            try:
                self._send(payload)
            except Exception as exc:
                raise RuntimeError("LTX 2.5 worker died: " + self._stderr_tail()) from exc
            event = self._read_chain_events(progress_callback)

        kind = event.get("event")
        if kind == "error":
            raise RuntimeError(event.get("detail", "LTX 2.5 worker chain generation failed"))
        if kind != "done":
            raise RuntimeError(f"LTX 2.5 worker returned unexpected event: {event!r}")

        if chain.crop_output is not None:
            video_io.crop_mp4(
                target, output_path, chain.crop_output.width, chain.crop_output.height
            )
            target.unlink(missing_ok=True)

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"LTX 2.5 worker produced no/empty chain output: {output_path}")

        if progress_callback:
            progress_callback(None, None, 1.0)

        return GenerationOutcome(
            output_path=output_path,
            seed_used=event.get("seed_used", seed),
            peak_vram_mb=event.get("peak_vram_mb"),
            generation_mode="chain",
            backend=REAL_BACKEND_25,
            chain_metadata=event.get("chain"),
            # Same relay discipline as the single path: every acceleration field
            # names a 2.3 code path this engine does not have, so reporting "off"
            # would claim the knob exists here and was left alone. attention_used
            # is the one worth stating positively — the chain scope is SDPA-only.
            attention_used="sdpa",
            peak_vram_reserved_mb=event.get("peak_vram_reserved_mb"),
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
