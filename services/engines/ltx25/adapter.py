"""LTX 2.5 engine adapter — the app-side half of the ``engine25`` worker.

A SIBLING of :mod:`services.engines.ltx.adapter`, not a replacement. The two
engines share nothing at runtime: 2.5 runs official LTX-2 v1.2.0 with
transformers 5.x inside ``.venv-engine-ltx25``, which cannot coexist with 2.3's
transformers 4.57 in one environment. What they DO share is the app-side
plumbing — the ``@@LTX@@`` JSON-lines protocol, the subprocess spawn/read/kill
loop, the progress-receipt loop, the synthetic mock backend — so this module
subclasses the 2.3 adapter's classes through their override seams (class
attributes and hook methods) and restates only the facts that are genuinely
different:

* which venv and which ``python -m`` module the worker is (``engine25.worker``);
* which worker log file it writes (so BOTH logs survive a 2.3<->2.5 swap);
* which payload field each model-management category feeds;
* which child-process environment it gets (NONE of 2.3's ``LTX_*`` knobs — every
  one of them is read by 2.3's worker and would be a borrowed assumption here);
* the feature scope, field by field, in the tables below. What is refused is
  listed in :data:`REJECT_TABLE` (single ``/generate``) and
  :data:`CHAIN_REJECT_TABLE` (chain schema), and both hold engine-level
  features rather than modes.

THE SCOPE, STATED ONCE. :data:`REJECT_TABLE` is the
422 half and :data:`IGNORED_FIELDS` the ignore-and-log half of the
``GenerateRequest`` field table; ``crop_output`` is deliberately in NEITHER —
it is an ffmpeg post-process the app applies to the finished mp4, so it is
engine-independent and simply works (see :meth:`_RealBackend25.generate`).

THE CHAIN SCHEMA HAS ITS OWN FOUR TABLES (``CHAIN_*``). The rules are
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
#: JUST THE SPATIAL UPSAMPLER. LTX 2.3's
#: :data:`services.engines.ltx.adapter.REQUIRED_ASSETS` also names a tokenizer
#: directory and a separate text-projection file; 2.5 needs neither, because
#: they have no counterpart here: the Gemma 4 tokenizer/processor metadata
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
#: transformer's ``model_version`` KV (the GGUF header, or a quantized
#: safetensors' ``__metadata__``). Mirror image of the 2.3 adapter's.
SUPPORTED_MODEL_VERSIONS: frozenset[str] = frozenset({"2.5"})

#: Real backend identifier surfaced in metadata.json. Distinct from 2.3's
#: ``"ltx-distilled"`` so an output file says which engine produced it.
REAL_BACKEND_25 = "ltx25-distilled"
#: Mock backend identifier. The mock CLASS is 2.3's (a synthetic gradient clip
#: says nothing about the engine that would have rendered it); only this label
#: differs, which is what lets the 2.3<->2.5 round trip (Docs/VERIFICATION_LOG.md
#: §69.15) be checked from metadata.json alone even when no GPU was involved.
MOCK_BACKEND_25 = "mock-ltx25"

#: Worker load defaults. ``blocks_on_gpu`` is the top rung of the documented
#: 16GB fallback ladder (Videomni_Backend_Specification.md §6.10 (e)) and is
#: overridable through the low-VRAM setting, so walking the ladder is a config
#: change, not a code one.
DEFAULT_BLOCKS_ON_GPU = 8


# --------------------------------------------------------------------------- #
# feature scope
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
#:
#: Every row is an engine-level feature rather than a mode; outpaint, inpaint
#: and NAG/VSF are in :data:`HONOURED_FIELDS`.
REJECT_TABLE: tuple[tuple[str, str, Callable[[GenerateRequest], bool]], ...] = (
    ("pipeline", "two_stage_hq", lambda r: r.pipeline != "distilled"),
    ("vae_mode", "prune_vaed", lambda r: r.vae_mode != "default"),
)

#: The ignore-and-log half: fields this engine cannot act on but that must NOT
#: fail a job, with the reason an operator sees in the log. Refusing them would
#: be hostile — the frontend sends the whole schema on every request, so a plain
#: T2V request carries them all.
#:
#: Every entry has the same reason: the distilled 2.5 schedule is fixed and has
#: no classifier-free guidance, so a CFG scale and a step count have nothing to
#: attach to.
IGNORED_FIELDS: dict[str, str] = {
    "guidance_scale": "LTX 2.5 distilled runs without classifier-free guidance",
    "num_inference_steps": "the distilled schedule has a fixed step count",
}

#: Fields this engine ACTS ON. Six of them ride the generate payload verbatim
#: (see :meth:`_RealBackend25.generate`), the acceleration and residency flags
#: ride it as additive keys, ``conditioning_images`` becomes the ``images``
#: list, and ``crop_output`` / ``embed_mp4_metadata`` are app-side
#: post-processing of the finished mp4. Declared rather than merely implied so
#: the model_fields audit below can name a home for every field.
#:
#: ``loras`` / ``reference_video_id`` and their two strengths. ``loras`` and
#: ``reference_video_id`` are not read off the request the way the six above
#: are: the orchestrator has already resolved the adapter NAMES into files and
#: the upload ID into a path, so what :meth:`_RealBackend25.generate` acts on
#: is the ``lora_paths`` / ``reference_video_path`` keyword arguments carrying
#: that material. The two strengths ARE read from the request, and only shape
#: the ``reference_video`` block. See the needle table in
#: tests/test_ltx25_adapter.py.
#:
#: ``block_swap_prefetch`` / ``fused_gguf_dequant_kernel``. Unlike ``loras`` /
#: ``reference_video_id`` they ARE plain request reads: each rides the payload
#: as a bare ``True`` under the same additive contract 2.3 uses, and the
#: worker echoes back what actually happened
#: (``block_swap_prefetch_used`` / ``fused_gguf_dequant_kernel_used``) so a
#: degrade is visible in metadata.json rather than assumed.
#:
#: ``keep_resident``. THE CONTRACT IS 2.3's WORD FOR WORD — the key rides
#: only when asked for, an absent key IS the release request, and the worker
#: echoes "on"/"off" back — but THE IMPLEMENTATION IS A DIFFERENT THING: 2.3
#: keeps the skeletons of every sub-model resident, while 2.5 keeps ONE thing,
#: the Gemma 4 text encoder's state dict (size: Videomni_Backend_Specification.md
#: §6.10 (d)), and nothing else. Its default is ``KEEP_RESIDENT_DEFAULT``
#: (api/models.py). That fits this engine: the win is only ever on a SECOND
#: job in the same process, and the RAM it costs is real. So the two
#: engines answering to the same field name do not do the same amount of work,
#: and a reader comparing them should expect different numbers.
#:
#: ``keep_resident_embeddings`` keeps the EmbeddingsProcessor resident (it is
#: otherwise rebuilt from its own GGUF on every job), beside the text encoder
#: that ``keep_resident`` holds. THE CONTRACT IS ``keep_resident``'s, restated:
#: the key rides only when asked for, an absent key IS the release request, and
#: the worker echoes "on"/"off" back. The two are INDEPENDENT switches over two
#: different objects, so their RAM costs add rather than overlap.
#: This one names a component only LTX 2.5 has, so the 422 for it belongs to
#: the OTHER engine (``services/engines/ltx/adapter.py``'s ``REJECT_TABLE``).
#:
#: ``outpaint`` is not a knob but a whole job kind: its PRESENCE is what routes the
#: worker to ``engine25.outpaint25.run_outpaint`` instead of the plain
#: generation, so the payload block below carries the full canvas geometry
#: rather than a flag (the engine has to rebuild the blend mask from it). The
#: app has already substituted the green canvas for ``reference_video.path``,
#: which is why nothing else in this method changes — and the ORIGINAL upload
#: arrives separately as ``outpaint_source_path``, read only for its audio.
#: THE PAYLOAD SHAPE IS 2.3's, key for key (services/engines/ltx/adapter.py):
#: one app-side contract for one feature, so an operator comparing two engines'
#: worker logs is comparing the same names.
#:
#: ``attention_backend``. THE CONTRACT IS 2.3's WORD FOR WORD once more: the key
#: rides ONLY when the request asked for something other than the default
#: ``"sdpa"``, so a plain job's payload carries no ``attention_backend`` key,
#: and the worker echoes back what ACTUALLY ran
#: (``attention_used``: "sdpa" / "sage" / "sage->sdpa"). And here the
#: implementation really IS the same thing: engine25 shares 2.3's
#: ``services/sage_attention_service.py`` verbatim, because the two engines'
#: attention contract (flat ``(B, S, H*D)`` q/k/v, head_dim 128/64) is
#: identical. The ONE difference is the lifetime of the wrapper it installs —
#: 2.5 reuses one model shell across jobs, so every build strips and re-installs
#: rather than patching a fresh transformer.
#:
#: The seven negative-prompt fields are plain ``request.<field>`` reads that
#: ride the additive ``nag`` payload block below, key for key in 2.3's order, so
#: one app-side contract serves both engines.
#:
#: "Honoured" and not "governed" for the three NAG knobs specifically: governed
#: would promise that a field cannot reach a running job, and their governor
#: ``nag_enabled`` is itself honoured, so a scale really does change the
#: video. And not "ignored" either, for the obvious reason — this engine acts on
#: them. HONOURED is the only classification that is true.
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
        # The app-side recipe embed into the finished mp4
        # (``PipelineManager._embed_recipe``), engine-independent post-processing
        # like ``crop_output``.
        "embed_mp4_metadata",
        "loras",
        "reference_video_id",
        "conditioning_attention_strength",
        "reference_video_strength",
        "block_swap_prefetch",
        "fused_gguf_dequant_kernel",
        "keep_resident",
        "keep_resident_embeddings",
        "attention_backend",
        "outpaint",
        # Listed next to ``outpaint`` because the two are one mechanism seen
        # from two sides — the same IC-LoRA, the same canvas, the same blend.
        "inpaint",
        # Alpha Gen (RGB video -> grey matte). Its block is internal: only
        # POST /generate/alpha fills it. The single full-size stage rides the
        # payload as ``alpha_gen``; the light mode rides no key at all (it IS
        # the plain two-stage generate with the reference attached).
        "alpha_gen",
        # The non-CFG negative prompt (NAG / VSF), seven fields that travel as
        # one feature: the switch, the prompt, the method, and the two methods'
        # knobs. They are listed together rather than sorted in because that is
        # what they are — a request either carries the whole block or none of it.
        "nag_enabled",
        "negative_prompt",
        "nag_scale",
        "nag_tau",
        "nag_alpha",
        "neg_method",
        "vsf_scale",
    }
)

#: ``field -> the field that governs it``. These are SUB-PARAMETERS: each one is
#: inert unless its governor is non-default, and every governor here is in
#: :data:`REJECT_TABLE`. So they need no ruling of their own — a request that
#: makes one of them meaningful is already refused, by the governor, before this
#: field could have mattered.
#:
#: The distinction is worth keeping rather than folding into
#: :data:`IGNORED_FIELDS`: "ignored" promises a job runs anyway, which is false
#: for a governed field — it cannot reach a running job at all.
#:
#: The table may be empty, and is KEPT rather than deleted when it is.
#:
#: Empty is a real state and a checkable one: the classification audit reads
#: this table by name, ``test_every_governor_is_itself_refused`` iterates it (an
#: empty loop passes, which is the honest answer when nothing is governed), and
#: keeping the name means the day a NEW sub-parameter appears behind a NEW 422
#: there is somewhere obvious to put it.
GOVERNED_FIELDS: dict[str, str] = {}

# --------------------------------------------------------------------------- #
# chain feature scope
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
#: NO WHOLE MODE IS LEFT HERE. Every row is an engine-level feature the single
#: path refuses for the same reason it refuses it there; ``outpaint`` has no
#: counterpart at all, because the chain schema has no such field.
CHAIN_REJECT_TABLE: tuple[
    tuple[str, str, Callable[[GenerateChainRequest], bool]], ...
] = (
    ("pipeline", "two_stage_hq", lambda r: r.pipeline != "distilled"),
    ("vae_mode", "prune_vaed", lambda r: r.vae_mode != "default"),
)

#: The ignore-and-log half. Field-for-field the same as
#: :data:`IGNORED_FIELDS` and for the same one reason (the distilled schedule
#: has no CFG and no step count to honour) — spelled out rather than aliased so
#: the audit test reads one schema against one table.
CHAIN_IGNORED_FIELDS: dict[str, str] = {
    "guidance_scale": "LTX 2.5 distilled runs without classifier-free guidance",
    "num_inference_steps": "the distilled schedule has a fixed step count",
}

#: Fields the chain path ACTS ON. The scalar settings ride the worker payload
#: verbatim (``stage2_window`` only when it differs from the default) and the
#: acceleration and residency flags ride it as additive keys (see
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
#: so the engine records the number in metadata and denoises its fixed distilled
#: schedule regardless (the reason text is in :data:`CHAIN_IGNORED_FIELDS`).
#: "Honoured" would claim it changes the output; it does not.
#:
#: ``loras`` / ``reference_video_id`` and their two strengths are honoured for
#: the same reason and in the same shape as the single path's
#: :data:`HONOURED_FIELDS`: the orchestrator resolves the adapter names and the
#: upload id into material, and the payload carries the additive ``loras`` /
#: ``reference_video`` blocks below. A long reference is sliced per stage-1
#: segment by the ENGINE; the app recomputes the same windows from
#: ``chain_math`` for metadata, so nothing about that geometry is decided here.
#:
#: ``block_swap_prefetch`` / ``fused_gguf_dequant_kernel``: the same reason as
#: on the single path — and the chain is where the prefetch work is actually
#: visible, because a chain rebuilds the transformer once per stage and the
#: engine re-arms the prefetch engine on every one of those builds.
#:
#: ``keep_resident``, with the same warning as on the single path: THE CONTRACT
#: is 2.3's verbatim (key only when asked for, absent key = release, echo back
#: what happened) but THE IMPLEMENTATION IS A DIFFERENT THING — 2.3 holds every
#: sub-model's skeleton, 2.5 holds ONE state dict, the Gemma 4 text encoder's
#: (size: Videomni_Backend_Specification.md §6.10 (d)). Its default follows
#: ``KEEP_RESIDENT_DEFAULT`` (api/models.py). A chain builds the text encoder
#: once per JOB just as a single job does, so what is saved here is likewise
#: the SECOND job's rebuild, not anything inside the chain itself.
#:
#: ``keep_resident_embeddings``, with the same reading as on the single path:
#: what it saves is the NEXT job's rebuild of the EmbeddingsProcessor, because
#: a chain builds that component once per JOB exactly as a single generate
#: does. Same additive contract as ``keep_resident`` beside it, and the same
#: independence — two switches, two objects, two RAM costs that add.
#:
#: The component it names exists only on this engine, so the refusal it
#: produces is LTX 2.3's, not this engine's.
#:
#: ``attention_backend``, with the same contract as on the single path: the
#: key rides only when the request asked for something other than ``"sdpa"``,
#: and the worker echoes what ran. A chain is where the wrapper's lifetime
#: matters most — it rebuilds the transformer once per stage, and every one of
#: those builds strips the previous job's wrapper before installing a fresh
#: one, so the echo a chain returns is a fold over every build it made
#: ("sage->sdpa" when one of them fell back).
#:
#: The seven negative-prompt fields are plain ``chain.<field>`` reads riding the
#: additive ``nag`` block, key for key as on the single path and as on 2.3.
CHAIN_HONOURED_FIELDS: frozenset[str] = frozenset(
    {
        "prompt",
        "clips",
        "width",
        "height",
        "crop_output",
        # See the same entry in :data:`HONOURED_FIELDS`.
        "embed_mp4_metadata",
        "frame_rate",
        "seed",
        "overlap_frames",
        "overlap_strength",
        "chunked_upsample",
        "stage2_window",
        "source_video",
        "source_audio",
        # Retake, honoured INDIRECTLY exactly as the two source modes are: the
        # request field carries an upload id plus a window START TIME, and the
        # orchestrator has already turned that into material — the frame-exact,
        # CFR window mp4 it cut with ``video_io.cut_window_mp4``. What
        # :meth:`_RealBackend25.generate_chain` reads is therefore
        # ``retake_window_path`` (plus the glue widths off ``chain.retake``),
        # not the id. Naming the REQUEST field here is what the audit needs.
        "retake",
        # End source, honoured INDIRECTLY for the same reason retake and the
        # two source modes are: the request field carries an upload id (a
        # video OR a still image) plus the band length, and the ORCHESTRATOR
        # has already turned that into material — the mp4 it cut, or the
        # still it looped, to exactly ``context_frames + 1`` frames at the
        # request fps. What :meth:`_RealBackend25.generate_chain` reads is
        # therefore ``end_source_path`` and the two numbers beside it, not
        # the id. Naming the REQUEST field here is what the audit needs.
        "end_source",
        "loras",
        "reference_video_id",
        "conditioning_attention_strength",
        "reference_video_strength",
        "block_swap_prefetch",
        "fused_gguf_dequant_kernel",
        "keep_resident",
        "keep_resident_embeddings",
        "attention_backend",
        # The non-CFG negative prompt (NAG / VSF): the single path's seven, on
        # the chain schema, carrying the identical payload block.
        "nag_enabled",
        "negative_prompt",
        "nag_scale",
        "nag_tau",
        "nag_alpha",
        "neg_method",
        "vsf_scale",
    }
)

#: ``field -> the field that governs it``, exactly as :data:`GOVERNED_FIELDS`:
#: each is inert unless its governor is non-default, and every governor here is
#: in :data:`CHAIN_REJECT_TABLE`, so a request that could make one of them
#: matter is already refused before this field is read.
#:
#: Kept even when empty, for the same reasons as :data:`GOVERNED_FIELDS` — the
#: audit reads it by name, the governor test's loop passes when it is empty,
#: and a future sub-parameter behind a future 422 has an obvious home.
CHAIN_GOVERNED_FIELDS: dict[str, str] = {}


#: Everything GET /models publishes as this engine's ``unsupported_features``.
#: The request-field features come from :data:`REJECT_TABLE` so the two can
#: never disagree, and nothing is added to them: the published names are
#: exactly :data:`REJECT_TABLE`'s feature column. The frontend greys tabs and
#: panels by these names.
UNSUPPORTED_FEATURES: tuple[str, ...] = tuple(
    feature for _field, feature, _pred in REJECT_TABLE
)


def reject_unsupported(request: GenerateRequest) -> None:
    """422 the first out-of-scope field of ``request``.

    Called from :meth:`_RealBackend25.generate` — the adapter is the SOURCE OF
    TRUTH for what this engine can run, so the ruling lives next to the engine
    rather than in the endpoint. The API layer also calls it
    (``services.engines.reject_unsupported`` from api/generate.py) so a
    rejection costs no worker round-trip and no job record; calling it there
    changes nothing about the answer, only how early it arrives.

    First offender wins. Listing all of them would read as "fix all of these"
    when in practice one control was left on.
    """
    for field, feature, is_non_default in REJECT_TABLE:
        if is_non_default(request):
            raise feature_unsupported(
                feature,
                detail=(
                    f"LTX 2.5は{feature}に対応していません"
                    f"(リクエストの{field}が既定値ではありません)。"
                    "この機能を使うにはベースモデルに「LTX 2.3」を選んでください。"
                ),
            )


def reject_chain(request: GenerateChainRequest) -> None:
    """422 the first out-of-scope field of a chain ``request``.

    What is out of scope for a chain is decided FIELD BY FIELD, exactly like
    the single path, so the request has to be read. The refused fields are
    listed in :data:`CHAIN_REJECT_TABLE`.

    Same first-offender-wins rule and same table discipline as
    :func:`reject_unsupported`.

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
    the user made, and reporting every entry of the ignore table on every job
    would train the reader to skip the line. The default comes from the schema
    itself (``model_fields[...].default``) rather than a transcribed copy, so a
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
    """Rule on a transformer's KV header for the LTX 2.5 engine (§2.2).

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
            "model '%s' declares no general.architecture in its header; "
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
            "model '%s' declares no model_version in its header; loading "
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
        (``LTX_TE_OFFLOAD``, ``LTX_DIT_CPU_LOAD``) names
        a code path inside ``engine/``; engine25 has its own offload and
        block-swap mechanics and reads none of them, so passing them would be
        decoration at best and a misleading log at worst. The 2.5 equivalents
        (``blocks_on_gpu`` / ``te_layers_on_gpu`` / ``cache_weights``) ride the
        LOAD PAYLOAD instead, where they are visible in the protocol.

        What IS kept is the process hygiene the two share: torch.compile off,
        unbuffered IO, and PYTHONPATH pinned to the project root so
        ``python -m engine25.worker`` resolves the first-party package.

        ``PYTORCH_CUDA_ALLOC_CONF`` is deliberately not set. The
        ``expandable_segments:True`` option has no effect on this platform:
        torch refuses it on Windows ("expandable_segments not supported on this
        platform") and keeps the segmented caching allocator (measured on torch
        2.9.1; see Docs/VERIFICATION_LOG.md §75.7 and §89.3), and torch 2.9
        deprecates the variable's NAME in favour of ``PYTORCH_ALLOC_CONF``.
        Fragmentation is dealt with where it happens: the arena ring in
        ``engine/transformer/block_swap_prefetch.py``. The parent's own
        environment passes through (``dict(os.environ)`` below), so an operator
        who exports the variable by hand is not overridden.
        """
        env = dict(os.environ)
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

        The four category paths come from the base-model descriptor exactly as
        2.3's do — a selection override wins, otherwise the category's
        ``default_file`` — with :data:`SELECTION_FIELDS` again read rather than
        transcribed, so a category missing from that table cannot reach the
        worker at all. The spatial upsampler is a fixed descriptor asset
        (:data:`REQUIRED_ASSETS`) and takes no override.

        NOT SENT: ``text_encoder_assets_path``. The assets-only safetensors
        export (the ~30MB metadata twin the official ``GemmaAssets.load`` opens
        before the GGUF weights are swapped in) is DERIVED from the text-encoder
        GGUF and lives beside it; engine25's pipeline regenerates it whenever it
        is missing or stale (engine25/assets_export.ensure_assets_only). Naming
        it from here would add a path the app has to keep correct in order to
        express "the default location" — the field exists in the protocol only
        for a deployment that keeps the two apart.

        Key set AND insertion order are the byte contract, pinned by the golden
        snapshot in tests/test_ltx25_adapter.py — do not reorder.
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
            # Keeps the transformer weights in host RAM so stage 2 does not
            # re-read them from disk (the official pipeline disposes the stage-1
            # weights, meta-ing the buffers). The RAM cost is in
            # Videomni_Backend_Specification.md §6.10 (e). Always sent as True:
            # the app exposes no setting for it (the worker's own
            # ``--no-cache-weights`` is for standalone runs).
            "cache_weights": True,
            # cuDNN algorithm selection pinned, so a same-seed rerun is
            # bit-identical (the audio vocoder is what varies otherwise). Always
            # sent as True; the same-seed determinism check in
            # Docs/VERIFICATION_LOG.md §69.14 depends on it.
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
    ) -> GenerationOutcome:
        """One two-stage T2V/I2V generation, with Style / IC-LoRA, Outpainting
        and Inpainting.

        ``lora_paths`` and ``reference_video_path`` ARE ACTED ON: the
        orchestrator resolved the adapter names into safetensors files and the
        upload id into a path, and both ride the payload in 2.3's shape (see the
        ``loras`` / ``reference_video`` keys below).

        ``outpaint_source_path`` IS ACTED ON, and it is the one argument whose
        meaning is not obvious from its name: it is the ORIGINAL upload, not the
        material the engine extends. By the time the orchestrator calls this it
        has already built the green canvas and substituted it for
        ``reference_video_path``, so the canvas is what the IC-LoRA conditions
        on; the source travels separately because the canvas is written WITHOUT
        an audio stream on purpose, and the clip's own waveform is what the
        finished mp4 carries. It rides the ``outpaint`` block below, whose mere
        PRESENCE is what routes the worker to the two-stage outpaint driver.

        ``inpaint_source_path`` / ``inpaint_mask_path`` ARE ACTED ON. They are
        the orchestrator's two extra paths for an inpaint job and the canvas
        cannot supply either: the CUT WINDOW (``_inpaint_window.mp4`` — the
        source size is ffprobed from it and its waveform is what the finished
        mp4 carries, because the canvas is written without an audio stream on
        purpose) and the MASK VIDEO the engine decodes for the blend. Both ride
        the ``inpaint`` block below, whose mere PRESENCE routes the worker to
        ``engine25.inpaint25.run_inpaint``. The canvas itself rides in
        ``reference_video_path``, which the orchestrator has already
        substituted — so the IC-LoRA plumbing needs no notion of inpainting at
        all.

        ``crop_output`` IS honoured: it is an ffmpeg centre-crop the app
        performs on the finished mp4, so it is engine-independent and a 2.5 job
        gets the same non-64 display sizes a 2.3 job does. Silently dropping it
        is the easy mistake.
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

        # Style/character AND control IC-LoRA, in 2.3's
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
        # an omitted-field job's block carries no ``attention_strength`` key.
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

        # The base payload; the blocks below append to it. Fields this engine
        # ignores are NOT forwarded: they were already logged above, and a
        # payload that carries only what is acted upon is a payload a golden
        # snapshot can pin.
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
        # Acceleration (additive): each key rides ONLY when the request asked
        # for it, so a job with either knob turned off carries no key for it.
        # Whether a plain job carries them is decided by the pydantic defaults
        # (``BLOCK_SWAP_PREFETCH_DEFAULT`` / ``FUSED_GGUF_DEQUANT_KERNEL_DEFAULT``
        # in api/models.py) — an omitted key means off on the worker side, which
        # is 2.3's contract restated verbatim (services/engines/ltx/adapter.py).
        # Written as a literal ``request.<field>`` read on purpose: the needle
        # table in tests/test_ltx25_adapter.py proves an honoured field is
        # really read by searching this function's source for exactly that text.
        if request.block_swap_prefetch:
            payload["block_swap_prefetch"] = True
        if request.fused_gguf_dequant_kernel:
            payload["fused_gguf_dequant_kernel"] = True
        # keep_resident: the same additive contract, appended after the keys
        # above so their order does not move. The key rides only when True (the
        # default is ``KEEP_RESIDENT_DEFAULT`` in api/models.py) — and an absent
        # key is not merely "not asked for", it is the RELEASE request the
        # worker acts on (2.3's contract, restated verbatim). What the engine
        # then holds is only the text encoder's state dict, not 2.3's whole set
        # of sub-model skeletons.
        if request.keep_resident:
            payload["keep_resident"] = True
        # attention_backend, appended after the keys above so no existing key
        # order moves. The key rides only when the request picks something
        # other than "sdpa" (the pydantic default), so a plain job carries NO
        # key. An absent key means "sdpa" on the worker side (2.3's contract
        # restated verbatim), and the worker echoes back what actually ran
        # rather than what was asked for.
        if request.attention_backend != "sdpa":
            payload["attention_backend"] = request.attention_backend
        # Outpainting, appended after the blocks above so no existing key order
        # moves. The key is ABSENT from every non-outpaint job: the pydantic
        # default is None, and an omitted field and an explicit
        # ``outpaint: null`` both arrive as None.
        #
        # ITS PRESENCE IS ALSO THE SWITCH that routes the worker to
        # ``engine25.outpaint25.run_outpaint`` instead of the plain generate,
        # which is why it carries the full canvas geometry rather than a flag:
        # the engine rebuilds the blend mask from it, and a mask built from a
        # geometry the app did not validate is the one thing this feature must
        # never do. ``reference_video.path`` above is ALREADY the green canvas
        # (the orchestrator substituted it); the ``source_path`` here is the
        # original upload, read only for its audio.
        #
        # KEY FOR KEY 2.3's block (services/engines/ltx/adapter.py), same
        # order: one app-side contract per feature, not one per engine.
        if request.outpaint is not None:
            op = request.outpaint
            payload["outpaint"] = {
                "source_path": str(outpaint_source_path) if outpaint_source_path else None,
                "canvas_width": request.width,
                "canvas_height": request.height,
                "pad_left": op.pad_left,
                "pad_right": op.pad_right,
                "pad_top": op.pad_top,
                "pad_bottom": op.pad_bottom,
                "blend_dilation_stage1": op.blend_dilation_stage1,
                "blend_dilation_stage2": op.blend_dilation_stage2,
                "freeze_source_audio": op.freeze_source_audio,
            }
        # Inpainting, immediately after its sibling because the two are one
        # mechanism seen from two sides — and MUTUALLY EXCLUSIVE, which the
        # schema enforces and the worker asserts, so the two blocks can never
        # both be present. Additive like every block before it: absent from
        # every non-inpaint job, so their payloads carry no ``inpaint`` key.
        #
        # It carries the geometry because the engine has to rebuild the canvas
        # arithmetic, and the two FILE PATHS the canvas cannot supply: the cut
        # window (audio + the restore's original picture) and the mask video
        # itself. ``reference_video.path`` above is already the green-filled
        # canvas.
        #
        # Note what is NOT here: pads. The engine derives them from
        # ``canvas − source``, which is the same single-source-of-truth rule the
        # API enforces (see ``api.models.InpaintSpec``).
        #
        # KEY FOR KEY 2.3's block (services/engines/ltx/adapter.py), same order
        # and the same three guards: one app-side contract per feature, not one
        # per engine. Written out rather than shared with 2.3, as the outpaint
        # block above is: a common helper would change a shipped engine's code
        # path for nothing but line count.
        if request.inpaint is not None:
            ip = request.inpaint
            # The source size is READ FROM THE FILE, not taken from the request,
            # for exactly the reason the request does not carry it: the file is
            # the only thing that can be checked. The cut window preserves the
            # upload's resolution (``cut_window_mp4`` never rescales), so this
            # is the same number the endpoint validated the canvas against.
            if inpaint_mask_path is None:
                raise RuntimeError(
                    "inpaint: the mask video path never reached the backend; "
                    "run_inpaint cannot decode a mask it was not given"
                )
            if inpaint_source_path is None:
                raise RuntimeError(
                    "inpaint: the cut window's path never reached the backend; "
                    "the source size and the frozen audio both come from it"
                )
            source_size = video_io.probe_resolution(Path(inpaint_source_path))
            if source_size is None:
                raise RuntimeError(
                    "inpaint: could not probe the cut window's resolution "
                    f"({inpaint_source_path})"
                )
            source_width, source_height = source_size
            payload["inpaint"] = {
                "source_path": str(inpaint_source_path) if inpaint_source_path else None,
                "mask_path": str(inpaint_mask_path) if inpaint_mask_path else None,
                "canvas_width": request.width,
                "canvas_height": request.height,
                "source_width": source_width,
                "source_height": source_height,
                "blend_dilation_stage1": ip.blend_dilation_stage1,
                "blend_dilation_stage2": ip.blend_dilation_stage2,
            }
        # Alpha Gen, appended after the inpaint block so no existing key order
        # moves. The key rides ONLY for the single full-size stage, and its
        # presence is the switch that routes the worker to
        # ``engine25.alphagen25.run_alpha_gen``. The light mode carries NO key:
        # it is the plain two-stage generate with the reference attached, so its
        # payload is byte-for-byte a plain reference job's. Everything else the
        # engine needs (canvas, reference, LoRA) already rides above; the padding
        # is cut off app-side after the job.
        if request.alpha_gen is not None and request.alpha_gen.one_stage:
            payload["alpha_gen"] = {"mode": "one_stage"}
        # The non-CFG negative prompt (NAG / VSF), appended after the blocks
        # above so no existing key order moves. ADDITIVE — the block rides only
        # when the request actually enabled it, so a job without it carries no
        # ``nag`` key.
        #
        # KEY FOR KEY 2.3's BLOCK, in 2.3's order (services/engines/ltx/adapter.py):
        # one app-side contract per feature, not one per engine, so an operator
        # comparing two engines' worker logs is comparing the same names. The
        # worker key is ``method`` rather than ``neg_method``, and scale/tau/alpha
        # ride unconditionally because the engine only reads them when
        # ``method == "nag"``.
        if request.nag_enabled:
            payload["nag"] = {
                "negative_prompt": request.negative_prompt,
                "scale": float(request.nag_scale),
                "tau": float(request.nag_tau),
                "alpha": float(request.nag_alpha),
            }
            payload["nag"]["method"] = request.neg_method
            payload["nag"]["vsf_scale"] = request.vsf_scale
        # keep_resident_embeddings, appended after the blocks above so no
        # existing key order moves. Contract-for-contract ``keep_resident``'s
        # (the key rides only when True — the default is
        # ``KEEP_RESIDENT_EMBEDDINGS_DEFAULT`` in api/models.py — and an absent
        # key is not silence but the RELEASE request the worker acts on), over
        # a different object: the EmbeddingsProcessor's CPU state dict rather
        # than the text encoder's. Written as a literal
        # ``request.keep_resident_embeddings`` read for the needle table's sake,
        # same as the blocks above.
        if request.keep_resident_embeddings:
            payload["keep_resident_embeddings"] = True

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
            # The acceleration relays engine25 HAS: the
            # worker reports what actually happened, not what was asked for —
            # "off" / "on" / "on->off", the last being a degrade (no Triton, or a
            # build the prefetch engine could not be installed on).
            # pipeline_manager writes them into metadata.json unchanged. ``.get``
            # rather than a default because a worker that never spoke leaves
            # None, and None is the honest answer.
            block_swap_prefetch_used=event.get("block_swap_prefetch_used"),
            fused_gguf_dequant_kernel_used=event.get("fused_gguf_dequant_kernel_used"),
            # keep_resident echoes only "on"/"off" here — engine25 has no degrade
            # path for it, so "on->off" never appears even though the field's
            # contract allows it. A job that died before the text encoder was
            # built would have echoed "on" too, but then no done event arrives,
            # so nothing is relayed at all (2.3 behaves identically).
            keep_resident_used=event.get("keep_resident_used"),
            # The EmbeddingsProcessor's own echo, on exactly the
            # two-value contract of the line above ("on"/"off", never
            # "on->off" -- there is no degrade path here either). Separate from
            # ``keep_resident_used`` because the two switches are separate: a
            # job can hold one object resident and release the other.
            keep_resident_embeddings_used=event.get("keep_resident_embeddings_used"),
            # ``vae_mode_used`` is this engine's OWN decoder-name echo ("diff" /
            # "conv"): this engine has no PrunaVAED knob, and the echo answers a
            # different question with a different vocabulary from 2.3's
            # "off"/"on"/"on->off". ``ltx25`` is this engine's own additive facts
            # (encode_fps/video_chunks/tiling/size_bytes/phases, or the larger
            # block ``outpaint25._ltx25_block`` builds for an outpaint or inpaint
            # job) -- absent on 2.3 and on the mock, so it stays ``None`` there.
            # ``vae_mode_used`` is ``None`` only on the mock, though: on 2.3 it
            # is the PrunaVAED echo, not absent. Same ``.get`` discipline as the
            # relays above: a worker that never spoke leaves None.
            vae_mode_used=event.get("vae_mode_used"),
            ltx25=event.get("ltx25"),
            # The inpaint job's own facts, relayed verbatim. THIS
            # LINE IS THE ONLY ROUTE ``mask_proof`` TAKES TO metadata.json —
            # ``services/pipeline_manager.py`` writes ``metadata["inpaint"] =
            # {**outcome.inpaint, **provenance}``, and without the relay the
            # block would carry the app's provenance and none of the engine's
            # evidence. The worker sends the SUB-DICT (not the whole metadata
            # dict), which is what keeps that merge flat. ``None`` on every
            # other job, including outpaint: ``.get`` is the honest answer for a
            # key that is absent by design.
            inpaint=event.get("inpaint"),
            # attention_used is the worker's own echo — "sdpa", "sage", or
            # "sage->sdpa" for a build that asked for sage and could not have
            # it — so a degrade is visible in metadata.json rather than assumed.
            # ``.get`` for the same reason as the relays above: a worker that
            # never spoke leaves None.
            attention_used=event.get("attention_used"),
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

        None of those keywords names a mode outside this engine's scope; the
        request fields this engine refuses are ruled on by
        ``CHAIN_REJECT_TABLE``, not by this signature.

        HONOURED, and worth naming: ``clip0_conditioning_paths`` (clip 0's I2V
        keyframes — the only clip the schema lets carry them) and
        ``crop_output``, the app-side ffmpeg centre-crop of the finished mp4,
        which is engine-independent for a chain exactly as it is for a single
        job.

        ALSO HONOURED: ``source_tail_path`` +
        ``source_context_frames`` (V2V continuation) and ``source_audio_path``
        (A2V). All three are material the ORCHESTRATOR prepared — the tail cut
        to the requested fps, the uploaded wav — and they ride the payload as
        the additive ``source`` / ``audio_source`` blocks below, in 2.3's shape
        so the two engines' chain payloads stay comparable.

        ALSO HONOURED: ``lora_paths`` (Style/character and
        control adapters, applied uniformly across the chain) and
        ``reference_video_path`` (the ONE reference video, sliced per stage-1
        segment by the engine). Both ride as ADDITIVE blocks — non-empty /
        non-``None`` only — so a plain chain carries neither key
        (``GOLDEN_CHAIN_KEYS_25`` + ``GOLDEN_ACCEL_KEYS_25`` in
        tests/test_ltx25_adapter.py pin a plain chain's key set and order),
        which is 2.3's discipline for the same two keys.

        ALSO HONOURED: ``retake_window_path``, the
        frame-exact CFR window the orchestrator cut out of the user's material
        (``video_io.cut_window_mp4``). The glue widths ride with it off
        ``chain.retake``, in 2.3's block shape — the engine owns "what happens
        to those pixels", the app owns "which pixels", and neither engine cuts
        or resamples.

        ALSO HONOURED: ``end_source_path`` / ``end_source_context_frames`` /
        ``end_source_strength``. The same division of labour once more — the
        orchestrator prepared the material (a cut video, or a still image it
        looped into one; always ``context_frames + 1`` frames at the request
        fps, which is the primer the causal VAE needs) and the engine decides
        what happens to those latents. ``end_source_strength`` is the one of
        the three that is a KNOB rather than material. The orchestrator passes
        the schema's value (``EndSourceSpec.strength``); a ``None`` from a
        caller that omits the keyword falls back to 1.0 here, so the engine
        never sees ``None``.
        """
        chain = chain_request
        # BEFORE the load, unlike :meth:`generate`. The ruling is a pure read of
        # the request, so spawning a worker first would only mean a doomed
        # request pays for a model load — and it keeps this refusal reachable
        # without a subprocess, which is what lets a pytest hold it.
        reject_chain(chain)
        _log_ignored(chain, CHAIN_IGNORED_FIELDS)

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
        # engine records it in metadata and denoises the fixed schedule
        # regardless (the reason text is in ``CHAIN_IGNORED_FIELDS``), which is
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
        # ``is not None`` guard on BOTH values is what keeps a plain chain free
        # of a ``source`` key (``GOLDEN_CHAIN_KEYS_25`` + ``GOLDEN_ACCEL_KEYS_25``
        # in tests/test_ltx25_adapter.py pin a plain chain's key set and order) —
        # the same discipline, and the same key names and key ORDER, as 2.3's.
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
        # Retake (additive): the app-cut window plus the glue geometry, in 2.3's
        # block shape (services/engines/ltx/adapter.py) key for key and in the
        # same key ORDER, because the two workers are unrelated code but a
        # retake's geometry is the same in both. The ``is not None`` guard on
        # BOTH the path and the request block is what keeps a non-retake chain
        # free of a ``retake`` key (``GOLDEN_CHAIN_KEYS_25`` +
        # ``GOLDEN_ACCEL_KEYS_25`` in tests/test_ltx25_adapter.py pin a plain
        # chain's key set and order) — the same discipline as the two source
        # blocks, and the same one 2.3 keeps.
        if retake_window_path is not None and getattr(chain, "retake", None) is not None:
            payload["retake"] = {
                "path": str(retake_window_path),
                "head_px": int(chain.retake.head_px),
                "tail_px": int(chain.retake.tail_px),
                "regenerate_audio": bool(chain.retake.regenerate_audio),
            }
        # End source (additive): the app-prepared tail material — a cut video
        # or a looped still, so the engine only ever sees a video — plus the
        # band length and the user's strength. 2.3's block shape
        # (services/engines/ltx/adapter.py) key for key and in the same key
        # ORDER, for the reason the retake block above gives. The ``is not
        # None`` guard on BOTH values is what keeps a chain without an end
        # source free of an ``end_source`` key (``GOLDEN_CHAIN_KEYS_25`` +
        # ``GOLDEN_ACCEL_KEYS_25`` in tests/test_ltx25_adapter.py pin a plain
        # chain's key set and order).
        if end_source_path is not None and end_source_context_frames is not None:
            payload["end_source"] = {
                "path": str(end_source_path),
                "context_frames": int(end_source_context_frames),
                "strength": (
                    1.0 if end_source_strength is None else float(end_source_strength)
                ),
            }
        # Style/character AND control IC-LoRA (additive): (path, strength[,
        # audio_strength]) per adapter, applied uniformly across the chain, and
        # sent ONLY when non-empty, so a no-lora chain carries no ``loras`` key
        # (``GOLDEN_CHAIN_KEYS_25`` + ``GOLDEN_ACCEL_KEYS_25`` in
        # tests/test_ltx25_adapter.py pin a plain chain's key set and order).
        # ``preprocess`` is deliberately NOT part of an entry — it is derived once
        # for the reference block below, and a control adapter without a
        # reference is already refused at the API layer.
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
        # stage2_window: additive, sent ONLY when the request differs from
        # ``chain_math.STAGE2_WINDOW_DEFAULT``, so a default chain carries no
        # ``stage2_window`` key (same contract as 2.3's).
        if chain.stage2_window != chain_math.STAGE2_WINDOW_DEFAULT:
            payload["stage2_window"] = chain.stage2_window
        # Acceleration (additive): the same contract and the same literal-read
        # discipline as the single path above. Each block's position in this
        # function fixes its key's place in the payload; the conditional blocks
        # above are absent from a default chain. The key set and order of a
        # default chain are pinned by ``GOLDEN_CHAIN_KEYS_25`` +
        # ``GOLDEN_ACCEL_KEYS_25`` in tests/test_ltx25_adapter.py.
        if chain.block_swap_prefetch:
            payload["block_swap_prefetch"] = True
        if chain.fused_gguf_dequant_kernel:
            payload["fused_gguf_dequant_kernel"] = True
        # keep_resident, appended after the blocks above so no existing key
        # order moves. The key rides only when True; the default is
        # ``KEEP_RESIDENT_DEFAULT`` (api/models.py). An absent key is the release
        # request, not silence. The key set of a default chain job is pinned by
        # ``GOLDEN_CHAIN_KEYS_25`` + ``GOLDEN_ACCEL_KEYS_25`` in
        # tests/test_ltx25_adapter.py.
        if chain.keep_resident:
            payload["keep_resident"] = True
        # attention_backend, appended after the blocks above for the same reason.
        # Default "sdpa", so a plain chain carries no ``attention_backend`` key;
        # the key rides only when the user asked for something else, and an
        # absent key is "sdpa" on the worker side rather than silence.
        if chain.attention_backend != "sdpa":
            payload["attention_backend"] = chain.attention_backend
        # The non-CFG negative prompt (NAG / VSF), appended after the blocks
        # above for the reason they were, and byte-for-byte the single path's
        # block on the chain schema — which is also 2.3's chain block, key for
        # key.
        if chain.nag_enabled:
            payload["nag"] = {
                "negative_prompt": chain.negative_prompt,
                "scale": float(chain.nag_scale),
                "tau": float(chain.nag_tau),
                "alpha": float(chain.nag_alpha),
            }
            payload["nag"]["method"] = chain.neg_method
            payload["nag"]["vsf_scale"] = chain.vsf_scale
        # keep_resident_embeddings, appended last for the reason everything above
        # it was, and the chain twin of the single path's block key for key. The
        # key rides only when True; the default is
        # ``KEEP_RESIDENT_EMBEDDINGS_DEFAULT`` (api/models.py). An absent key is
        # the release request, not silence. The key set of a default chain job
        # is pinned by ``GOLDEN_CHAIN_KEYS_25`` + ``GOLDEN_ACCEL_KEYS_25`` in
        # tests/test_ltx25_adapter.py.
        if chain.keep_resident_embeddings:
            payload["keep_resident_embeddings"] = True

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
            # Same relay discipline as the single path: the knobs engine25
            # really has are echoed back from the worker's done event, and a
            # chain's echo is a fold over every build it made ("on->off" when one
            # of them fell back).
            block_swap_prefetch_used=event.get("block_swap_prefetch_used"),
            fused_gguf_dequant_kernel_used=event.get("fused_gguf_dequant_kernel_used"),
            # keep_resident is per-JOB, not per-build: the text encoder is built
            # once for the whole chain, so this echo is "on"/"off" and never the
            # folded "on->off" the two above can produce.
            keep_resident_used=event.get("keep_resident_used"),
            # The chain twin, and per-JOB for the same reason the line
            # above is -- the EmbeddingsProcessor is built once for the whole
            # chain, so this echo is "on"/"off" and never a folded "on->off".
            keep_resident_embeddings_used=event.get("keep_resident_embeddings_used"),
            # Same decoder-name echo as the single path -- see the comment
            # there. No ``ltx25=`` here: ``chain_metadata`` (built
            # above from ``event.get("chain")``) already carries
            # ``chain_metadata["ltx25"]``, so ``GenerationOutcome.ltx25`` stays
            # None on a chain outcome, single-job only by contract.
            vae_mode_used=event.get("vae_mode_used"),
            # attention_used is the worker's echo, and on a chain it is a FOLD
            # over every build the chain made — "sage->sdpa" when one of them
            # fell back, the same shape the block_swap_prefetch /
            # fused_gguf_dequant_kernel echoes above take here.
            attention_used=event.get("attention_used"),
            peak_vram_reserved_mb=event.get("peak_vram_reserved_mb"),
        )


class LTX25Runner(LTXRunner):
    """Facade for the LTX 2.5 engine family.

    Everything the base facade does — lazy descriptor resolution, mock/real
    selection, ``set_descriptor`` teardown, the availability probe and its
    missing-file report — applies unchanged; the class attributes below are the
    only per-family facts.

    ``sage_available`` is deliberately NOT overridden: the base class's
    file-existence probe reads ``_REAL_BACKEND_CLS._engine_python_value(config)``,
    which on this class is ``model.engine_python_ltx25`` — so it looks in the
    2.5 venv's own site-packages, and the honest answer is whatever it finds
    there.
    """

    _REAL_BACKEND_CLS = _RealBackend25
    _MOCK_BACKEND_CLS = _MockBackend
    _MOCK_BACKEND_LABEL = MOCK_BACKEND_25
