"""LTX 2.5 engine adapter — payload contract and v1 feature scope (§3-98 P3b).

Four things are pinned here, none of which needs a GPU or a weight file:

1. the ``{"op":"load"}`` payload engine25's worker receives — key SET, key
   ORDER and values, as a golden snapshot, the same discipline the 2.3 payload
   has had since model management (tests/test_model_swap_load.py). The 2.3
   golden is deliberately untouched by this file; the two engines' payloads are
   independent contracts;
2. the v1 feature scope: which ``GenerateRequest`` fields are a 422, which are
   ignored-and-logged, and — the part a table alone cannot state — that those
   two sets do not overlap and that ``crop_output`` is in NEITHER;
3. the CHAIN scope and the ``{"op":"generate_chain"}`` payload (§3-102): the
   same four-way field table and the same golden-snapshot discipline, applied
   to ``GenerateChainRequest``. Until §3-102 this engine refused every chain,
   so the chain schema needed no ruling at all; now a plain Chained job runs
   and each of its 34 fields must have exactly one home;
4. the seams: LTX 2.5 reuses the 2.3 MOCK backend class rather than declaring
   its own, which is a design ruling ("やらない" list) and therefore a test.
"""

from __future__ import annotations

import inspect
import logging
import threading
import types
from pathlib import Path

import pytest

from api.errors import APIError
from api.models import ChainClip, GenerateChainRequest, GenerateRequest
from config import AppConfig
from services.base_models import BaseModelDescriptor, CategoryDescriptor
from services.engines.ltx import adapter as ltx23
from services.engines.ltx25 import adapter as ltx25
from services.lora_registry import ResolvedLora
from services.low_vram import build_low_vram_settings

LTX23_KV = {"general.architecture": "ltxv", "model_version": "2.3.0"}
LTX25_KV = {"general.architecture": "ltxv", "model_version": "2.5.0"}

# The exact key ORDER of the load payload. JSON serialization preserves
# insertion order, so an order change IS a byte change.
GOLDEN_PAYLOAD_KEYS_25 = [
    "op",
    "transformer_path",
    "text_encoder_path",
    "video_vae_path",
    "audio_vae_path",
    "spatial_upsampler_path",
    "blocks_on_gpu",
    "cache_weights",
    "deterministic",
]


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _category(name: str, extension: str, default_file) -> CategoryDescriptor:
    return CategoryDescriptor(
        name=name,
        scan=(str(default_file.parent),),
        extensions=(extension,),
        default_file=str(default_file),
    )


@pytest.fixture()
def ltx25_paths(tmp_path):
    """An LTX 2.5 base model whose every payload path exists (dummy files).

    Built in Python rather than as JSON for the same reason the 2.3 fixture is:
    descriptor paths are ``models_dir``-relative, and joining ``models_dir``
    with an ABSOLUTE path yields that absolute path, so each file resolves
    exactly where the fixture put it.
    """
    m = tmp_path / "m"
    paths = {
        "tr": _touch(m / "Weights" / "ltx-2.5-transformer-Q4_K_M.gguf"),
        "te": _touch(m / "TextEncoder" / "gemma-4-Q6_K.gguf"),
        "vv": _touch(m / "VAE" / "ltx25_video_vae.safetensors"),
        "av": _touch(m / "VAE" / "ltx25_audio_vae.safetensors"),
        "ups": _touch(m / "Upscaler" / "ltx-2.5-spatial-upscaler-x2.safetensors"),
    }
    descriptor = BaseModelDescriptor(
        id="LTX25FIXTURE",
        display_name="LTX 2.5 fixture",
        engine_family="ltx25",
        categories={
            "transformer": _category("transformer", ".gguf", paths["tr"]),
            "text_encoder": _category("text_encoder", ".gguf", paths["te"]),
            "video_vae": _category("video_vae", ".safetensors", paths["vv"]),
            "audio": _category("audio", ".safetensors", paths["av"]),
        },
        assets={"spatial_upsampler_path": str(paths["ups"])},
    )
    cfg = AppConfig.model_validate({"model": {"backend": "mock"}})
    return cfg, paths, descriptor


def _backend(cfg, descriptor) -> ltx25._RealBackend25:
    return ltx25._RealBackend25(cfg, build_low_vram_settings(cfg), descriptor)


def _golden(paths) -> dict:
    return {
        "op": "load",
        "transformer_path": str(paths["tr"]),
        "text_encoder_path": str(paths["te"]),
        "video_vae_path": str(paths["vv"]),
        "audio_vae_path": str(paths["av"]),
        "spatial_upsampler_path": str(paths["ups"]),
        "blocks_on_gpu": 8,  # low_vram default None -> `or DEFAULT_BLOCKS_ON_GPU`
        "cache_weights": True,
        "deterministic": True,
    }


# --------------------------------------------------------------------------- #
# 1) golden load payload
# --------------------------------------------------------------------------- #


def test_load_payload_golden_without_selection(ltx25_paths):
    cfg, paths, descriptor = ltx25_paths
    backend = _backend(cfg, descriptor)
    p_none = backend._build_load_payload(None)
    p_empty = backend._build_load_payload({})
    assert p_none == p_empty == _golden(paths)
    assert list(p_none) == GOLDEN_PAYLOAD_KEYS_25


def test_load_payload_selection_overrides_only_the_named_field(ltx25_paths):
    cfg, paths, descriptor = ltx25_paths
    backend = _backend(cfg, descriptor)
    alt = _touch(paths["tr"].parent / "alt-Q6.gguf")
    p = backend._build_load_payload({"transformer": str(alt)})
    golden = _golden(paths)
    assert p["transformer_path"] == str(alt)
    for key in GOLDEN_PAYLOAD_KEYS_25:
        if key != "transformer_path":
            assert p[key] == golden[key]
    assert list(p) == GOLDEN_PAYLOAD_KEYS_25


def test_load_payload_all_four_categories_map_to_expected_fields(ltx25_paths):
    cfg, _paths, descriptor = ltx25_paths
    backend = _backend(cfg, descriptor)
    p = backend._build_load_payload(
        {"transformer": "X_TR", "text_encoder": "X_TE", "video_vae": "X_VV", "audio": "X_AU"}
    )
    assert p["transformer_path"] == "X_TR"
    assert p["text_encoder_path"] == "X_TE"
    assert p["video_vae_path"] == "X_VV"
    assert p["audio_vae_path"] == "X_AU"


def test_load_payload_fails_loud_on_a_missing_asset(tmp_path, ltx25_paths):
    """The spatial upsampler is the ONE required asset; a descriptor that does
    not declare it must stop the load in the app layer, not in the worker."""
    cfg, _paths, descriptor = ltx25_paths
    stripped = BaseModelDescriptor(
        id=descriptor.id,
        display_name=descriptor.display_name,
        engine_family=descriptor.engine_family,
        categories=descriptor.categories,
        assets={},
    )
    with pytest.raises(RuntimeError, match="spatial_upsampler_path"):
        _backend(cfg, stripped)._build_load_payload(None)


def test_category_names_are_the_same_axes_as_ltx23():
    """The four categories are the axes a user picks a file on — a property of
    the product, not of an engine generation. state.json remembers a selection
    by these names, so they must not diverge per base model."""
    assert set(ltx25.SELECTION_FIELDS) == set(ltx23.SELECTION_FIELDS)
    # ...while every payload FIELD differs (the protocols are unrelated).
    assert not set(ltx25.SELECTION_FIELDS.values()) & set(ltx23.SELECTION_FIELDS.values())


def test_required_assets_is_the_upsampler_alone():
    assert ltx25.REQUIRED_ASSETS == ("spatial_upsampler_path",)
    # The audio VAE is required by the worker but arrives on the CATEGORY axis,
    # so it must NOT be declared a second time as an asset.
    assert "audio" in ltx25.SELECTION_FIELDS


def test_2_3_golden_payload_is_untouched_by_this_engine():
    """The 2.3 field table is a frozen byte contract; adding a second engine
    must not have edited it."""
    assert ltx23.SELECTION_FIELDS == {
        "transformer": "gguf_transformer_path",
        "text_encoder": "gguf_gemma_path",
        "video_vae": "component_video_vae_path",
        "audio": "component_audio_vae_path",
    }


# --------------------------------------------------------------------------- #
# 2) KV ruling
# --------------------------------------------------------------------------- #


def test_check_kv_accepts_2_5_and_refuses_2_3():
    ltx25.check_kv("transformer", "default", LTX25_KV)  # no raise
    with pytest.raises(APIError) as ei:
        ltx25.check_kv("transformer", "old", LTX23_KV)
    assert ei.value.code == "MODEL_INCOMPATIBLE" and ei.value.status_code == 422
    assert "ltxv 2.3.0" in ei.value.detail


def test_check_kv_patch_segment_is_ignored():
    ltx25.check_kv("transformer", "respin", {**LTX25_KV, "model_version": "2.5.9"})


def test_check_kv_refuses_a_foreign_architecture():
    with pytest.raises(APIError) as ei:
        ltx25.check_kv("transformer", "wan", {"general.architecture": "wan"})
    assert "'wan'系のモデルです" in ei.value.detail


def test_check_kv_missing_keys_warn_but_pass(caplog):
    with caplog.at_level(logging.WARNING, logger="ltx25.runner"):
        ltx25.check_kv("transformer", "homebrew", {})
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


@pytest.mark.parametrize("category", ["text_encoder", "video_vae", "audio"])
def test_check_kv_judges_only_the_transformer(category):
    ltx25.check_kv(category, "default", LTX23_KV)
    ltx25.check_kv(category, "default", {"general.architecture": "gemma3"})


# --------------------------------------------------------------------------- #
# 3) v1 feature scope — the GenerateRequest field table
# --------------------------------------------------------------------------- #


def _request(**overrides) -> GenerateRequest:
    return GenerateRequest(prompt="a quiet harbour at first light", **overrides)


_LORAS = [{"name": "style-a", "strength": 1.0}]

#: One VALID request per rejected field that sets that field non-default, so
#: the table below drives real ``GenerateRequest`` objects rather than
#: asserting against itself.
#:
#: ``outpaint`` LEFT THIS TABLE with the Outpainting increment, and it was the
#: last entry that needed TWO companions (the video it extends and the IC-LoRA
#: that conditions on it). Its mirror image is
#: :data:`REQUEST_ACCEPTED_OUTPAINT` below.
#:
#: ``nag_enabled`` LEFT WITH THE NAG/VSF INCREMENT, and it was the last entry
#: that needed a companion at all (the schema couples it to a non-empty
#: negative prompt). Its mirror image is :data:`REQUEST_ACCEPTED_NAG`, which
#: inherits the coupling — this engine's ruling has to survive it, not dodge it.
#: Every row that remains is a bare one-field override.
REQUEST_OVERRIDES: dict[str, dict] = {
    "pipeline": {"pipeline": "two_stage_hq"},
    "vae_mode": {"vae_mode": "prune_vaed"},
}

#: The SCHEMA-REQUIRED companion for every NAG row below: ``nag_enabled`` is the
#: master switch and the schema refuses it without a non-empty negative prompt
#: (``api/models.py``). Spelt once and spread into each row, so a row says only
#: what it is actually testing.
_NAG_COMPANION = {"nag_enabled": True, "negative_prompt": "blurry, low quality"}

#: What the NAG/VSF increment turned on: the SEVEN negative-prompt fields, which
#: left THREE different tables at once — ``nag_enabled`` from
#: :data:`REQUEST_OVERRIDES`, the prompt/method/vsf_scale trio from
#: ``IGNORED_FIELDS``, and the three NAG knobs from ``GOVERNED_FIELDS`` (which
#: it emptied). A table of its own for the reason every accepted table before it
#: has one: it is THIS increment's evidence, and folding it into an earlier
#: increment's table would make both lie about what they cover.
#:
#: SEVEN ROWS FOR SEVEN FIELDS, each at a NON-DEFAULT value, because the class of
#: regression that matters here is a single knob being dropped on the way to the
#: payload while the switch still rides. The last two rows carry
#: ``neg_method="vsf"`` because ``vsf_scale`` is only meaningful under it — and
#: because VSF is the method whose payload key is spelt differently on the wire
#: (``method``, not ``neg_method``).
REQUEST_ACCEPTED_NAG: dict[str, dict] = {
    "nag_enabled": {**_NAG_COMPANION},
    "negative_prompt": {**_NAG_COMPANION},
    "nag_scale": {**_NAG_COMPANION, "nag_scale": 7.5},
    "nag_tau": {**_NAG_COMPANION, "nag_tau": 3.5},
    "nag_alpha": {**_NAG_COMPANION, "nag_alpha": 0.5},
    "neg_method": {**_NAG_COMPANION, "neg_method": "vsf"},
    "vsf_scale": {**_NAG_COMPANION, "neg_method": "vsf", "vsf_scale": 3.0},
}

#: What §3-102's THIRD increment turned on, as valid request bodies: the two
#: fields that left :data:`REJECT_TABLE` plus the two strength overrides that
#: left :data:`GOVERNED_FIELDS`. The mirror image of :data:`REQUEST_OVERRIDES`
#: — same shape, opposite verdict — so the acceptance test is table-driven
#: like the refusal one, and
#: tests/test_ltx25_api_guard.py can import it and drive the same bodies through
#: HTTP. A reference video is SCHEMA-coupled to an IC-LoRA, so the pair travels
#: together; the strength overrides are coupled the same way.
REQUEST_ACCEPTED_LORAS: dict[str, dict] = {
    "loras": {"loras": _LORAS},
    "reference_video_id": {"reference_video_id": "vid-123", "loras": _LORAS},
    "reference_video_strength": {
        "reference_video_id": "vid-123",
        "loras": _LORAS,
        "reference_video_strength": 0.8,
    },
    "conditioning_attention_strength": {
        "reference_video_id": "vid-123",
        "loras": _LORAS,
        "conditioning_attention_strength": 0.7,
    },
}

#: What 高速化第2弾 turned on: ``keep_resident``, which until this increment was
#: the last acceleration knob left in :data:`REJECT_TABLE`. A table of its OWN
#: rather than a row added to :data:`REQUEST_ACCEPTED_LORAS`, because that one
#: is the LoRA increment's evidence and says so in its name — mixing an
#: unrelated field into it would make both tables lie about what they cover.
#: One row, and a row it will stay: nothing else is coupled to it (no upload,
#: no adapter), so the request that turns it on is the plain request plus a
#: boolean.
#:
#: WHAT THIS BUYS on 2.5 is not what it buys on 2.3. The contract is 2.3's
#: verbatim — but the thing kept resident here is the Gemma 4 text encoder's
#: state dict alone (~7.7 GiB), not 2.3's whole set of sub-model skeletons.
REQUEST_ACCEPTED_KEEP_RESIDENT: dict[str, dict] = {
    "keep_resident": {"keep_resident": True},
}

#: What 高速化第3弾 turned on: ``attention_backend``, the LAST acceleration knob
#: left in :data:`REQUEST_OVERRIDES`. A table of its own for the same reason
#: :data:`REQUEST_ACCEPTED_KEEP_RESIDENT` is one — it is THIS increment's
#: evidence, and folding it into an earlier increment's table would make both
#: lie about what they cover.
#:
#: One row, and one row it stays: nothing is coupled to it (no upload, no
#: adapter), so the request that asks for sage is the plain request plus a
#: string. And unlike ``keep_resident``, what the field buys on 2.5 IS what it
#: buys on 2.3 — the two engines share
#: ``services/sage_attention_service.py`` verbatim, because their attention
#: contract is identical.
REQUEST_ACCEPTED_SAGE: dict[str, dict] = {
    "attention_backend": {"attention_backend": "sage"},
}

#: What the Outpainting increment turned on: ``outpaint``, the LAST whole MODE
#: :data:`REQUEST_OVERRIDES` refused. A table of its own for the reason the two
#: above are — it is THIS increment's evidence — and the ONLY one of the four
#: accepted tables whose row is not a knob but a job kind.
#:
#: THE COMPANIONS TRAVEL WITH IT, unchanged from when this row lived in
#: :data:`REQUEST_OVERRIDES`: the schema requires a reference video (the clip
#: whose canvas is extended) and the endpoint requires a CONTROL adapter to
#: consume it, so a bare ``outpaint`` block is not a valid request at all. Both
#: are honoured on this engine now, which is what makes the coupling harmless —
#: the point of the row is that the ruling lets the WHOLE combination through,
#: not that the block is reachable in isolation.
#:
#: The pads are one-sided and small on purpose: ``_request``'s canvas is the
#: schema default 512x320, so a 64px left pad leaves a 448x320 keep rectangle,
#: which clears the 256px floor the blend's mask dilation imposes. A test body
#: that tripped THAT rule would raise ValidationError before this engine's
#: ruling was ever consulted, and would prove nothing.
REQUEST_ACCEPTED_OUTPAINT: dict[str, dict] = {
    "outpaint": {
        "outpaint": {"pad_left": 64, "pad_right": 0, "pad_top": 0, "pad_bottom": 0},
        "reference_video_id": "vid-123",
        "loras": _LORAS,
    },
}


def test_reject_table_names_only_real_request_fields():
    fields = set(GenerateRequest.model_fields)
    assert {f for f, _feat, _p in ltx25.REJECT_TABLE} <= fields
    assert set(ltx25.IGNORED_FIELDS) <= fields


def test_reject_and_ignore_sets_are_disjoint():
    """A field is either refused or tolerated. Both would mean the answer
    depends on which check runs first."""
    assert not {f for f, _feat, _p in ltx25.REJECT_TABLE} & set(ltx25.IGNORED_FIELDS)


def test_crop_output_is_in_neither_set():
    """crop_output is an ffmpeg post-process the app applies to the finished
    mp4 — engine-independent, so it must be honoured, not refused or dropped."""
    assert "crop_output" not in {f for f, _feat, _p in ltx25.REJECT_TABLE}
    assert "crop_output" not in ltx25.IGNORED_FIELDS


def test_a_default_request_is_accepted():
    reqs = [_request(), _request(generation_mode="t2v"), _request(crop_output=None)]
    for r in reqs:
        ltx25.reject_unsupported(r)  # no raise


@pytest.mark.parametrize("field", list(REQUEST_OVERRIDES))
def test_every_rejected_field_raises_feature_unsupported(field):
    request = _request(**REQUEST_OVERRIDES[field])
    # The field under test really is non-default for this request...
    predicate = next(p for f, _feat, p in ltx25.REJECT_TABLE if f == field)
    assert predicate(request)
    # ...and the FIRST offender in table order is what the message names (a
    # schema-coupled companion field can legitimately be refused first).
    expected = next(feat for _f, feat, pred in ltx25.REJECT_TABLE if pred(request))
    with pytest.raises(APIError) as ei:
        ltx25.reject_unsupported(request)
    assert ei.value.code == "FEATURE_UNSUPPORTED" and ei.value.status_code == 422
    assert expected in ei.value.detail
    assert "LTX 2.3" in ei.value.detail


def test_reject_table_covers_every_field_the_fixture_names():
    """The parametrized test above is only as good as its request table."""
    assert set(REQUEST_OVERRIDES) == {f for f, _feat, _p in ltx25.REJECT_TABLE}


@pytest.mark.parametrize("case", list(REQUEST_ACCEPTED_LORAS))
def test_style_and_reference_are_no_longer_refused(case):
    """The headline of §3-102's third increment: Style/character LoRA and the
    reference-video control IC-LoRA pass the ruling instead of raising.
    Field-by-field, so a table that lost only ONE of the two rows fails here."""
    ltx25.reject_unsupported(_request(**REQUEST_ACCEPTED_LORAS[case]))  # no raise


def test_the_lora_fields_are_honoured_not_merely_unlisted():
    """Unlisted and honoured are different promises. A field dropped from every
    table would also stop raising — and would then be silently ignored."""
    lora_fields = {
        "loras",
        "reference_video_id",
        "conditioning_attention_strength",
        "reference_video_strength",
    }
    assert lora_fields <= ltx25.HONOURED_FIELDS
    assert not lora_fields & {f for f, _feat, _p in ltx25.REJECT_TABLE}
    assert not lora_fields & set(ltx25.IGNORED_FIELDS)
    assert not lora_fields & set(ltx25.GOVERNED_FIELDS)


@pytest.mark.parametrize("case", list(REQUEST_ACCEPTED_KEEP_RESIDENT))
def test_keep_resident_is_no_longer_refused(case):
    """The headline of 高速化第2弾: asking the 2.5 engine to keep the text
    encoder resident passes the ruling instead of raising 422. Until this
    increment it was the one acceleration knob that still refused a job — and
    refused it from the SETTINGS panel, i.e. for every job until the user found
    the setting again."""
    ltx25.reject_unsupported(_request(**REQUEST_ACCEPTED_KEEP_RESIDENT[case]))  # no raise


def test_keep_resident_is_honoured_not_merely_unlisted():
    """Unlisted and honoured are different promises. Dropping the field from
    every table would also stop the 422 — and would then keep the user's RAM
    setting a secret from the engine."""
    assert "keep_resident" in ltx25.HONOURED_FIELDS
    assert "keep_resident" not in {f for f, _feat, _p in ltx25.REJECT_TABLE}
    assert "keep_resident" not in ltx25.IGNORED_FIELDS
    assert "keep_resident" not in ltx25.GOVERNED_FIELDS


@pytest.mark.parametrize("case", list(REQUEST_ACCEPTED_SAGE))
def test_sage_attention_is_no_longer_refused(case):
    """The headline of 高速化第3弾: asking the 2.5 engine for the sage attention
    kernel passes the ruling instead of raising 422. Until this increment it was
    the LAST acceleration knob that still refused a job — and, like
    ``keep_resident`` before it, it refused from the SETTINGS panel, i.e. for
    every job until the user found the setting again."""
    ltx25.reject_unsupported(_request(**REQUEST_ACCEPTED_SAGE[case]))  # no raise


def test_sage_attention_is_honoured_not_merely_unlisted():
    """Unlisted and honoured are different promises. Dropping the field from
    every table would also stop the 422 — and would then run every job on SDPA
    while the Settings panel said sage."""
    assert "attention_backend" in ltx25.HONOURED_FIELDS
    assert "attention_backend" not in {f for f, _feat, _p in ltx25.REJECT_TABLE}
    assert "attention_backend" not in ltx25.IGNORED_FIELDS
    assert "attention_backend" not in ltx25.GOVERNED_FIELDS


def test_sage_attention_no_longer_names_a_published_limitation():
    """GET /models must stop publishing it, or the frontend greys out the
    attention control the server now accepts. This was the last acceleration
    name on the list, so the published set drops from seven names to six."""
    assert "sage_attention" not in ltx25.UNSUPPORTED_FEATURES
    # Six when this increment shipped; FIVE since Retake, FOUR since the End
    # source, THREE since Outpainting and TWO since NAG/VSF, each of which took
    # the next name off the list. The count is asserted rather than only the
    # absence because it is what catches a name being ADDED back by accident. At
    # two, every name left is an ENGINE-LEVEL feature -- no MODE of any kind is
    # published any more, chain or single.
    assert len(ltx25.UNSUPPORTED_FEATURES) == 2


def test_keep_resident_no_longer_names_a_published_limitation():
    """GET /models must stop publishing it, or the frontend greys out a setting
    the server now accepts — the exact trap this increment removes."""
    assert "keep_resident" not in ltx25.UNSUPPORTED_FEATURES


@pytest.mark.parametrize("case", list(REQUEST_ACCEPTED_OUTPAINT))
def test_outpaint_is_no_longer_refused(case):
    """The headline of the Outpainting increment: a canvas-extension request —
    the whole coupled combination, reference video and IC-LoRA included —
    passes the ruling instead of raising 422.

    THE COMBINATION IS THE TEST. Until this increment ``outpaint`` sat ABOVE the
    LoRA rows in the table precisely so that its companions could not steal the
    message; now that the row is gone, the same body has to survive the two
    rules that used to be underneath it as well. A table that dropped the
    ``outpaint`` row but re-refused it through ``loras`` would fail here."""
    ltx25.reject_unsupported(_request(**REQUEST_ACCEPTED_OUTPAINT[case]))  # no raise


def test_outpaint_is_honoured_not_merely_unlisted():
    """Unlisted and honoured are different promises, and nowhere more so than
    here: ``outpaint`` is not a knob whose loss degrades a picture, it is the
    SWITCH that routes the worker to the two-stage driver. A field dropped from
    every table would stop the 422 and then hand the user a plain generation at
    canvas size — a green border where their video should have been extended."""
    assert "outpaint" in ltx25.HONOURED_FIELDS
    assert "outpaint" not in {f for f, _feat, _p in ltx25.REJECT_TABLE}
    assert "outpaint" not in ltx25.IGNORED_FIELDS
    assert "outpaint" not in ltx25.GOVERNED_FIELDS


def test_outpaint_no_longer_names_a_published_limitation():
    """GET /models must stop publishing it, or the frontend greys out the Edit
    tab's 画角拡張 sub-tab for a mode the server now accepts."""
    assert "outpaint" not in ltx25.UNSUPPORTED_FEATURES


@pytest.mark.parametrize("case", list(REQUEST_ACCEPTED_NAG))
def test_the_negative_prompt_fields_are_no_longer_refused(case):
    """The headline of the NAG/VSF increment: a request that asks for a negative
    prompt passes the ruling instead of raising 422. Field by field, because the
    seven arrived from three different tables and a move that took only some of
    them would leave the rest silently dropped rather than loudly refused."""
    ltx25.reject_unsupported(_request(**REQUEST_ACCEPTED_NAG[case]))  # no raise


def test_the_negative_prompt_fields_are_honoured_not_merely_unlisted():
    """Unlisted, ignored and honoured are three different promises, and this is
    the increment where all three were in play at once: one field was a 422,
    three were "ignored", three were "governed". Dropping any of them from every
    table would stop the 422 and then hand the user a video generated without
    the negative prompt they typed — with nothing in the log to say so."""
    neg_fields = {
        "nag_enabled",
        "negative_prompt",
        "nag_scale",
        "nag_tau",
        "nag_alpha",
        "neg_method",
        "vsf_scale",
    }
    assert neg_fields <= ltx25.HONOURED_FIELDS
    assert not neg_fields & {f for f, _feat, _p in ltx25.REJECT_TABLE}
    assert not neg_fields & set(ltx25.IGNORED_FIELDS)
    assert not neg_fields & set(ltx25.GOVERNED_FIELDS)


def test_the_governed_table_is_empty_and_still_exists():
    """``GOVERNED_FIELDS`` lost its last three rows to the NAG/VSF increment.

    EMPTY IS A STATE, NOT AN ABSENCE. The name is still read by the
    classification audit and by ``test_every_governor_is_itself_refused``, whose
    loop is now honestly empty; deleting the table would make the audit
    incomplete rather than smaller, and would leave the next sub-parameter
    behind the next 422 with nowhere obvious to go."""
    assert ltx25.GOVERNED_FIELDS == {}
    assert ltx25.CHAIN_GOVERNED_FIELDS == {}


def test_nag_no_longer_names_a_published_limitation():
    """GET /models must stop publishing it, or the frontend greys out the
    negative-prompt panel — and the method switch and three knobs behind it —
    on both the Single and the Chained tab, for a feature the server runs."""
    assert "nag" not in ltx25.UNSUPPORTED_FEATURES


def test_unsupported_features_is_both_reject_tables_without_chain_itself():
    """Every feature name either table refuses must be published, or a control
    the server 422s stays lit in the frontend."""
    features = set(ltx25.UNSUPPORTED_FEATURES)
    assert {feat for _f, feat, _p in ltx25.REJECT_TABLE} <= features
    assert {feat for _f, feat, _p in ltx25.CHAIN_REJECT_TABLE} <= features
    # NO chain MODE is published any more. ``retake`` LEFT with the Retake
    # increment -- the engine regenerates the middle of a clip now, and the name
    # would go on greying out the Edit tab's 撮り直し sub-tab and the timeline
    # right-click route into it -- and ``end_source`` left with the End-source
    # increment, which was the LAST of them: the engine runs the layout's own
    # stage-1 schedule and freezes the material's band at the timeline's tail,
    # so publishing the name would grey the Chained tab's 素材（末尾） panel for a
    # mode that works.
    assert "retake" not in features
    assert "end_source" not in features
    # ...but "chain" itself LEFT with §3-102's first increment (a plain Chained
    # job runs on this engine now, so publishing "no chain" would grey out a tab
    # that works), and "v2v"/"a2v" left with the second — publishing either
    # would grey out the Chained tab's source panels, the Single tab's A2V
    # accordion and the Batch tab's A2V rows, all of which now work.
    assert "chain" not in features
    assert "v2v" not in features and "a2v" not in features
    assert len(ltx25.UNSUPPORTED_FEATURES) == len(features), "no duplicates"


def test_ignored_fields_are_logged_only_when_set(caplog):
    with caplog.at_level(logging.INFO, logger="ltx25.runner"):
        ltx25._log_ignored(_request())
    assert not [r for r in caplog.records if "ignores" in r.getMessage()]

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ltx25.runner"):
        # ``guidance_scale`` / ``num_inference_steps`` rather than the negative
        # prompt that used to stand here: the NAG/VSF increment moved those
        # three into HONOURED_FIELDS, so a needle naming them would now be
        # testing fields this table no longer has -- and would pass while
        # naming nothing, because ``_log_ignored`` skips what it cannot find.
        # These two are what the table is DOWN TO, so the needles cannot rot
        # again without the table itself becoming empty.
        #
        # ``pipeline="two_stage_hq"`` IS THE ONLY WAY TO BUILD SUCH A BODY, and
        # that is a fact about the schema rather than a trick: ``api/models.py``
        # pins both fields to their defaults whenever the pipeline is
        # ``distilled``. So on a 2.5 request that this engine would actually
        # ACCEPT, these two can only ever hold their defaults and this log line
        # cannot fire -- the table is a classification that stays honest, not a
        # message anyone will read. ``_log_ignored`` is a pure read of the
        # request, so calling it directly is what lets the rule be tested at all.
        ltx25._log_ignored(
            _request(pipeline="two_stage_hq", guidance_scale=7.5, num_inference_steps=30)
        )
    messages = [r.getMessage() for r in caplog.records if "ignores" in r.getMessage()]
    assert len(messages) == 1
    assert "guidance_scale" in messages[0] and "num_inference_steps" in messages[0]


# --------------------------------------------------------------------------- #
# 3c) chain feature scope — the GenerateChainRequest field table (§3-102)
# --------------------------------------------------------------------------- #
#
# A SECOND four-way table, for a SECOND schema. Chained used to be refused
# outright, so the whole of ``GenerateChainRequest`` needed no ruling at all;
# now a plain chain runs and each of its 34 fields has to have a home. The
# tests below are the chain twins of section 3 + 3b: the same coverage,
# exclusivity and reverse-direction audits, plus a golden payload, because the
# failure mode is the same one — a request field nobody classified is silently
# dropped or silently honoured, and either way the user is not told.


def _chain_request(**overrides) -> GenerateChainRequest:
    """A minimal VALID two-clip chain, the shape section 3c's tables ride on."""
    base: dict = {
        "prompt": "a quiet harbour at first light",
        "width": 512,
        "height": 320,
        "frame_rate": 24.0,
        "seed": 123,
        "overlap_frames": 2,
        "overlap_strength": 0.5,
        "clips": [{"num_frames": 25}, {"num_frames": 25}],
    }
    base.update(overrides)
    return GenerateChainRequest(**base)


#: One VALID chain request body per refused field, same discipline as
#: :data:`REQUEST_OVERRIDES`: the table drives real ``GenerateChainRequest``
#: objects rather than asserting against itself. Imported by
#: tests/test_ltx25_api_guard.py, which drives the same table through HTTP.
#:
#: ``source_video`` / ``source_audio`` LEFT this table with §3-102's second
#: increment — they are honoured now, and the requests that carry them live in
#: :data:`CHAIN_ACCEPTED_SOURCES` below instead. ``loras`` /
#: ``reference_video_id`` LEFT WITH THE THIRD, and live in
#: :data:`CHAIN_ACCEPTED_LORAS`. ``retake`` LEFT WITH THE RETAKE INCREMENT and
#: lives in :data:`CHAIN_ACCEPTED_RETAKE`. ``end_source`` LEFT WITH THE
#: END-SOURCE INCREMENT and lives in :data:`CHAIN_ACCEPTED_END_SOURCE` — it was
#: the last whole MODE here. ``nag_enabled`` LEFT WITH THE NAG/VSF INCREMENT
#: and lives in :data:`CHAIN_ACCEPTED_NAG`; it was the last row that carried a
#: schema companion, so every row that remains is a bare one-field override of
#: an engine-level field the single path refuses for the same reason.
CHAIN_OVERRIDES: dict[str, dict] = {
    "pipeline": {"pipeline": "two_stage_hq"},
    "vae_mode": {"vae_mode": "prune_vaed"},
}


#: The chain twin of :data:`REQUEST_ACCEPTED_NAG`, same seven fields and same
#: schema companion — a chain carries ONE negative prompt for every clip and
#: every stage, so the body shape is the single path's exactly. Imported by
#: tests/test_ltx25_api_guard.py, which drives the same bodies through HTTP.
CHAIN_ACCEPTED_NAG: dict[str, dict] = {
    "nag_enabled": {**_NAG_COMPANION},
    "negative_prompt": {**_NAG_COMPANION},
    "nag_scale": {**_NAG_COMPANION, "nag_scale": 7.5},
    "nag_tau": {**_NAG_COMPANION, "nag_tau": 3.5},
    "nag_alpha": {**_NAG_COMPANION, "nag_alpha": 0.5},
    "neg_method": {**_NAG_COMPANION, "neg_method": "vsf"},
    "vsf_scale": {**_NAG_COMPANION, "neg_method": "vsf", "vsf_scale": 3.0},
}


#: The two chain source modes that §3-102's second increment TURNED ON, as
#: valid request bodies. The mirror image of :data:`CHAIN_OVERRIDES`: same
#: shape, opposite verdict, so the acceptance test below is driven by a table
#: rather than by a hand-written pair — and tests/test_ltx25_api_guard.py can
#: import it and drive the same two bodies through HTTP.
CHAIN_ACCEPTED_SOURCES: dict[str, dict] = {
    "source_video": {
        "clips": [{"num_frames": 49}, {"num_frames": 25}],
        "source_video": {"video_id": "vid-1", "context_frames": 25},
    },
    "source_audio": {"source_audio": {"audio_id": "aud-1"}},
    # Single-tab A2V: the frontend sends a ONE-clip chain with the full-length
    # stage-2 window, which the schema only allows together with source_audio.
    "source_audio_full_length": {
        "clips": [{"num_frames": 121}],
        "source_audio": {"audio_id": "aud-1"},
        "stage2_window": "full_length",
    },
    # Long A2V: one uploaded audio across three clips.
    "source_audio_long": {
        "clips": [{"num_frames": 121}, {"num_frames": 121}, {"num_frames": 121}],
        "source_audio": {"audio_id": "aud-1"},
    },
}


#: The chain MODE the Retake increment turned on, as valid request bodies —
#: the mirror image of :data:`CHAIN_OVERRIDES`, same shape and opposite verdict,
#: so the acceptance test below is table-driven and tests/test_ltx25_api_guard.py
#: can drive the same bodies through HTTP.
#:
#: Retake owns the whole timeline, so every entry is ONE clip (the schema
#: enforces that). The three rows are the three geometries the app can ask for:
#: the default glue widths, an explicit non-default pair, and the "keep the
#: original sound" flag — which is the field that changes what is DELIVERED
#: rather than what is frozen, and therefore the one worth having in the table
#: of its own accord.
CHAIN_ACCEPTED_RETAKE: dict[str, dict] = {
    "retake": {
        "clips": [{"num_frames": 73}],
        "retake": {"video_id": "vid-1", "window_start_sec": 0.0},
    },
    "retake_explicit_glue": {
        "clips": [{"num_frames": 121}],
        "retake": {
            "video_id": "vid-1",
            "window_start_sec": 1.5,
            "head_px": 33,
            "tail_px": 32,
        },
    },
    "retake_keep_original_audio": {
        "clips": [{"num_frames": 73}],
        "retake": {
            "video_id": "vid-1",
            "window_start_sec": 0.0,
            "regenerate_audio": False,
        },
    },
}


#: The chain MODE the End-source increment turned on — the LAST row to leave
#: :data:`CHAIN_OVERRIDES`. Same discipline as the tables above: valid request
#: bodies, opposite verdict, imported by tests/test_ltx25_api_guard.py so the
#: same bodies go through HTTP.
#:
#: THE FIVE ROWS ARE THE FIVE THINGS THAT CAN VARY, and each is here because it
#: reaches a DIFFERENT part of the engine:
#:
#: * ``end_source`` — ONE clip, i.e. ``chain_math``'s ``in_window`` geometry
#:   (the band is that clip's own tail, and the owner's recommended usage).
#: * ``end_source_reverse`` — two clips, i.e. the ``reverse`` geometry, where
#:   stage 1 generates the clips LAST TO FIRST. It is the one row that changes
#:   the SHAPE of stage 1 rather than only what is frozen.
#: * ``end_source_strength`` — the knob, at a NON-DEFAULT value. A default-only
#:   table cannot see a strength that is dropped on the way to the payload,
#:   which is the class of regression the 1-ULP note in ``_freeze_strengths``
#:   is about.
#: * ``end_source_still`` — a still image rather than a video. The app loops it
#:   into a silent mp4, so the ENGINE sees one code path; what differs is which
#:   upload store the id comes from, which is a schema fact worth pinning.
#: * ``end_source_with_source_video`` — the INTERPOLATION the API explicitly
#:   allows: start material AND end material on one clip. It is the only
#:   combination the mode has, and the one a refusal keyed on the wrong field
#:   would break.
CHAIN_ACCEPTED_END_SOURCE: dict[str, dict] = {
    "end_source": {
        "clips": [{"num_frames": 121}],
        "end_source": {"video_id": "vid-2", "context_frames": 24},
    },
    "end_source_reverse": {
        "clips": [{"num_frames": 121}, {"num_frames": 121}],
        "end_source": {"video_id": "vid-2", "context_frames": 24},
    },
    "end_source_strength": {
        "clips": [{"num_frames": 121}],
        "end_source": {
            "video_id": "vid-2",
            "context_frames": 24,
            "strength": 0.7,
        },
    },
    "end_source_still": {
        "clips": [{"num_frames": 121}],
        "end_source": {"image_id": "img-1", "context_frames": 8},
    },
    "end_source_with_source_video": {
        "clips": [{"num_frames": 121}],
        "source_video": {"video_id": "vid-1", "context_frames": 25},
        "end_source": {"video_id": "vid-2", "context_frames": 24},
    },
}


#: What §3-102's THIRD increment turned on for a chain, same discipline again:
#: Style/character LoRA applied uniformly across the clips, and the ONE
#: reference video the engine slices per stage-1 segment. The reference entries
#: carry the IC-LoRA the schema couples them to, and the multi-clip one is the
#: long IC-LoRA case (§3-78's geometry, now on this engine too).
CHAIN_ACCEPTED_LORAS: dict[str, dict] = {
    "loras": {"loras": _LORAS},
    "reference_video_id": {
        "clips": [{"num_frames": 25}],
        "reference_video_id": "vid-123",
        "loras": _LORAS,
    },
    "reference_video_long": {"reference_video_id": "vid-123", "loras": _LORAS},
    "reference_video_strength": {
        "reference_video_id": "vid-123",
        "loras": _LORAS,
        "reference_video_strength": 0.8,
        "conditioning_attention_strength": 0.7,
    },
}


#: The chain twin of :data:`REQUEST_ACCEPTED_KEEP_RESIDENT` (高速化第2弾), and
#: a separate table for the same reason: it is this increment's evidence, not
#: the LoRA increment's. A chain builds the text encoder once per JOB exactly
#: as a single job does, so the field means the same thing on both endpoints —
#: what the SECOND job no longer has to rebuild.
CHAIN_ACCEPTED_KEEP_RESIDENT: dict[str, dict] = {
    "keep_resident": {"keep_resident": True},
}


#: The chain twin of :data:`REQUEST_ACCEPTED_SAGE` (高速化第3弾), a separate
#: table for the same reason. A chain rebuilds the transformer once per STAGE,
#: so the wrapper is stripped and re-installed on every one of those builds —
#: which is why the echo a chain returns is a fold, not a single build's answer.
CHAIN_ACCEPTED_SAGE: dict[str, dict] = {
    "attention_backend": {"attention_backend": "sage"},
}


@pytest.mark.parametrize("case", list(CHAIN_ACCEPTED_KEEP_RESIDENT))
def test_chain_keep_resident_is_no_longer_refused(case):
    """The chain twin of the single-path acceptance test (高速化第2弾)."""
    ltx25.reject_chain(_chain_request(**CHAIN_ACCEPTED_KEEP_RESIDENT[case]))  # no raise


@pytest.mark.parametrize("case", list(CHAIN_ACCEPTED_SAGE))
def test_chain_sage_attention_is_no_longer_refused(case):
    """The chain twin of the single-path acceptance test (高速化第3弾)."""
    ltx25.reject_chain(_chain_request(**CHAIN_ACCEPTED_SAGE[case]))  # no raise


def test_the_chain_sage_field_is_honoured_not_merely_unlisted():
    assert "attention_backend" in ltx25.CHAIN_HONOURED_FIELDS
    assert "attention_backend" not in {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE}
    assert "attention_backend" not in ltx25.CHAIN_IGNORED_FIELDS
    assert "attention_backend" not in ltx25.CHAIN_GOVERNED_FIELDS


def test_the_chain_keep_resident_field_is_honoured_not_merely_unlisted():
    assert "keep_resident" in ltx25.CHAIN_HONOURED_FIELDS
    assert "keep_resident" not in {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE}
    assert "keep_resident" not in ltx25.CHAIN_IGNORED_FIELDS
    assert "keep_resident" not in ltx25.CHAIN_GOVERNED_FIELDS


@pytest.mark.parametrize("case", list(CHAIN_ACCEPTED_LORAS))
def test_chain_style_and_reference_are_no_longer_refused(case):
    """The chain twin of the single-path acceptance test (§3-102 third
    increment). ``reference_video_long`` is the long IC-LoRA case: TWO clips
    driven by one reference video, which §3-78 established on 2.3."""
    ltx25.reject_chain(_chain_request(**CHAIN_ACCEPTED_LORAS[case]))  # no raise


def test_the_chain_lora_fields_are_honoured_not_merely_unlisted():
    lora_fields = {
        "loras",
        "reference_video_id",
        "conditioning_attention_strength",
        "reference_video_strength",
    }
    assert lora_fields <= ltx25.CHAIN_HONOURED_FIELDS
    assert not lora_fields & {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE}
    assert not lora_fields & set(ltx25.CHAIN_IGNORED_FIELDS)
    assert not lora_fields & set(ltx25.CHAIN_GOVERNED_FIELDS)


@pytest.mark.parametrize("case", list(CHAIN_ACCEPTED_RETAKE))
def test_chain_retake_is_no_longer_refused(case):
    """The headline of the Retake increment: a chain that regenerates the middle
    of an existing clip passes the ruling instead of raising. Until now this was
    the FIRST row of the chain reject table, and the one a user reached from the
    timeline's own right-click menu."""
    ltx25.reject_chain(_chain_request(**CHAIN_ACCEPTED_RETAKE[case]))  # no raise


def test_the_retake_field_is_honoured_not_merely_unlisted():
    """Unlisted and honoured are different promises. A field dropped from every
    table would also stop raising — and would then be silently ignored, i.e. the
    user would get their window back untouched in the middle and be told
    nothing."""
    assert "retake" in ltx25.CHAIN_HONOURED_FIELDS
    assert "retake" not in {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE}
    assert "retake" not in ltx25.CHAIN_IGNORED_FIELDS
    assert "retake" not in ltx25.CHAIN_GOVERNED_FIELDS


@pytest.mark.parametrize("case", list(CHAIN_ACCEPTED_END_SOURCE))
def test_chain_end_source_is_no_longer_refused(case):
    """The headline of the End-source increment: a chain that must END on the
    user's material passes the ruling instead of raising. It was the LAST row of
    the chain reject table, and the one the Chained tab's 素材（末尾） panel is
    greyed by."""
    ltx25.reject_chain(_chain_request(**CHAIN_ACCEPTED_END_SOURCE[case]))  # no raise


def test_the_end_source_field_is_honoured_not_merely_unlisted():
    """Unlisted and honoured are different promises. A field dropped from every
    table would also stop raising — and would then be silently ignored, i.e. the
    user would get a video that does NOT end on the material they handed in and
    be told nothing."""
    assert "end_source" in ltx25.CHAIN_HONOURED_FIELDS
    assert "end_source" not in {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE}
    assert "end_source" not in ltx25.CHAIN_IGNORED_FIELDS
    assert "end_source" not in ltx25.CHAIN_GOVERNED_FIELDS


def test_no_chain_mode_is_refused_any_more():
    """Each increment moved the rows it implemented and no others. With the End
    source's row gone, EVERY chain MODE the schema can express has a code path on
    this engine, and what is left in the table is engine-level fields only — so
    a MODE reappearing here would be a regression, not a scope decision.

    Spelled out as "none of these four" rather than as a count, because the
    failure this guards against is a specific name coming back, and a count
    would also fire for an unrelated engine-level field being added."""
    refused = {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE}
    assert not refused & {
        "end_source", "retake", "source_video", "source_audio",
        "loras", "reference_video_id",
    }
    # ...and what remains really is only the engine-level half. ``nag_enabled``
    # left it with the NAG/VSF increment, which is asserted by its own
    # acceptance test below rather than only by this set shrinking.
    assert refused == {"pipeline", "vae_mode"}


@pytest.mark.parametrize("case", list(CHAIN_ACCEPTED_SOURCES))
def test_v2v_and_a2v_are_no_longer_refused(case):
    """The headline of §3-102's second increment: the two source modes pass the
    ruling instead of raising. Field-by-field, so a table that lost only ONE of
    the two rows fails here."""
    ltx25.reject_chain(_chain_request(**CHAIN_ACCEPTED_SOURCES[case]))  # no raise


def test_the_two_source_modes_are_honoured_not_merely_unlisted():
    """Unlisted and honoured are different promises. A field dropped from every
    table would also stop raising — and would then be silently ignored."""
    assert {"source_video", "source_audio"} <= ltx25.CHAIN_HONOURED_FIELDS
    assert not {"source_video", "source_audio"} & {
        f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE
    }
    assert not {"source_video", "source_audio"} & set(ltx25.CHAIN_IGNORED_FIELDS)


@pytest.mark.parametrize("case", list(CHAIN_ACCEPTED_NAG))
def test_the_chain_negative_prompt_fields_are_no_longer_refused(case):
    """The chain twin of the single path's NAG acceptance, field by field."""
    ltx25.reject_chain(_chain_request(**CHAIN_ACCEPTED_NAG[case]))  # no raise


def test_the_chain_negative_prompt_fields_are_honoured_not_merely_unlisted():
    """Unlisted, ignored and honoured are three different promises — the chain
    twin of the single-path assertion, and the failure it guards is worse here:
    a chain is many clips, so a dropped negative prompt is many seconds of video
    generated without the thing the user asked to avoid."""
    neg_fields = {
        "nag_enabled",
        "negative_prompt",
        "nag_scale",
        "nag_tau",
        "nag_alpha",
        "neg_method",
        "vsf_scale",
    }
    assert neg_fields <= ltx25.CHAIN_HONOURED_FIELDS
    assert not neg_fields & {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE}
    assert not neg_fields & set(ltx25.CHAIN_IGNORED_FIELDS)
    assert not neg_fields & set(ltx25.CHAIN_GOVERNED_FIELDS)


def test_a_plain_chain_is_accepted():
    """The headline of §3-102: a chain with nothing layered on it passes."""
    for request in (
        _chain_request(),
        _chain_request(chunked_upsample=True),
        _chain_request(stage2_window="high_resolution"),
        _chain_request(crop_output={"width": 384, "height": 256}),
        _chain_request(clips=[{"num_frames": 25, "prompt": "dusk"}, {"num_frames": 25}]),
    ):
        ltx25.reject_chain(request)  # no raise


@pytest.mark.parametrize("field", list(CHAIN_OVERRIDES))
def test_every_rejected_chain_field_raises_feature_unsupported(field):
    request = _chain_request(**CHAIN_OVERRIDES[field])
    # The field under test really is non-default for this request...
    predicate = next(p for f, _feat, p in ltx25.CHAIN_REJECT_TABLE if f == field)
    assert predicate(request)
    # ...and the FIRST offender in table order is what the message names.
    expected = next(feat for _f, feat, pred in ltx25.CHAIN_REJECT_TABLE if pred(request))
    with pytest.raises(APIError) as ei:
        ltx25.reject_chain(request)
    assert ei.value.code == "FEATURE_UNSUPPORTED" and ei.value.status_code == 422
    assert expected in ei.value.detail
    assert "LTX 2.3" in ei.value.detail


def test_the_chain_reject_table_covers_every_field_the_fixture_names():
    """The parametrized test above is only as good as its request table."""
    assert set(CHAIN_OVERRIDES) == {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE}


def test_the_chain_ignore_table_is_logged_only_when_set(caplog):
    with caplog.at_level(logging.INFO, logger="ltx25.runner"):
        ltx25._log_ignored(_chain_request(), ltx25.CHAIN_IGNORED_FIELDS)
    assert not [r for r in caplog.records if "ignores" in r.getMessage()]

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ltx25.runner"):
        # ``guidance_scale`` / ``num_inference_steps`` for the same reason as
        # the single-path twin above, and with the same schema caveat: only a
        # ``two_stage_hq`` body can carry either at a non-default value, so on
        # an acceptable 2.5 chain this line cannot fire either.
        ltx25._log_ignored(
            _chain_request(
                pipeline="two_stage_hq", guidance_scale=7.5, num_inference_steps=30
            ),
            ltx25.CHAIN_IGNORED_FIELDS,
        )
    messages = [r.getMessage() for r in caplog.records if "ignores" in r.getMessage()]
    assert len(messages) == 1
    assert "guidance_scale" in messages[0] and "num_inference_steps" in messages[0]


def test_generate_chain_refuses_an_out_of_scope_chain(ltx25_paths):
    """UNTIL §3-102 this refused EVERY chain. Now the refusal is field-by-field,
    and it still happens without loading a worker — the ruling is a pure read of
    the request, so a doomed chain must not pay for a model load.

    The subject was ``end_source`` until that mode was implemented and NAG until
    the NAG/VSF increment; ``vae_mode`` is a field the engine still refuses on a
    chain, and it needs no schema companion, so the body under test is the plain
    chain plus one value."""
    cfg, _paths, descriptor = ltx25_paths
    backend = _backend(cfg, descriptor)
    with pytest.raises(APIError) as ei:
        backend.generate_chain(_chain_request(**CHAIN_OVERRIDES["vae_mode"]), output_dir=None)
    assert ei.value.code == "FEATURE_UNSUPPORTED" and ei.value.status_code == 422
    assert "LTX 2.3" in ei.value.detail


def test_the_backend_refuses_a_chain_through_the_shared_function(ltx25_paths):
    """The endpoint (P5) and the backend method must not be able to disagree:
    both go through ``reject_chain``, so there is one message and one code.

    The subject followed the test above off NAG and onto ``vae_mode`` when the
    NAG/VSF increment made a negative prompt a thing this engine runs."""
    cfg, _paths, descriptor = ltx25_paths
    request = _chain_request(**CHAIN_OVERRIDES["vae_mode"])
    with pytest.raises(APIError) as endpoint_side:
        ltx25.reject_chain(request)
    with pytest.raises(APIError) as backend_side:
        _backend(cfg, descriptor).generate_chain(request, output_dir=None)
    assert endpoint_side.value.code == backend_side.value.code
    assert endpoint_side.value.detail == backend_side.value.detail


def test_generate_chain_has_no_out_of_scope_material_left_to_refuse(ltx25_paths, tmp_path):
    """The orchestrator hands every runner the same thirteen keywords. THREE of
    them used to name material this engine could not use, and a value arriving
    here raised a loud ``RuntimeError`` because it could only mean the reject
    table and the signature had drifted apart.

    THE END-SOURCE INCREMENT EMPTIED THAT GUARD, so the assertion inverts: the
    LAST three keywords must now reach the payload rather than raise. Keeping
    the test rather than deleting it is what makes "the guard was removed
    because nothing was left" checkable against "the guard was removed and
    something is now silently ignored"."""
    cfg, _paths, descriptor = ltx25_paths
    _backend(cfg, descriptor)  # the fixture is what proves this path is loadable
    captured: list[dict] = []
    _capturing_chain_backend(captured).generate_chain(
        _chain_request(**CHAIN_ACCEPTED_END_SOURCE["end_source"]),
        output_dir=tmp_path / "es",
        end_source_path=tmp_path / "e.mp4",
        end_source_context_frames=24,
        end_source_strength=0.7,
    )
    assert captured[0]["end_source"]["path"] == str(tmp_path / "e.mp4")
    # ...while the EMPTY list run_chain_job always builds for a no-lora chain is
    # not "material" and must sail through. Checked on the no-subprocess harness
    # so the assertion is about the guard, not about a worker spawn.
    captured: list[dict] = []
    _capturing_chain_backend(captured).generate_chain(
        _chain_request(), output_dir=tmp_path / "ok", lora_paths=[]
    )
    assert captured[0]["op"] == "generate_chain"
    assert "loras" not in captured[0]


def test_the_lora_keywords_are_no_longer_out_of_scope_material(ltx25_paths, tmp_path):
    """The other half of the guard's §3-102-third change: adapters and a
    reference video ARRIVING here is a JOB now, not a drift between the table
    and the signature — so they must reach the payload rather than raise."""
    captured: list[dict] = []
    _capturing_chain_backend(captured).generate_chain(
        _chain_request(**CHAIN_ACCEPTED_LORAS["reference_video_long"]),
        output_dir=tmp_path / "ok",
        lora_paths=[(tmp_path / "union.safetensors", 1.0, "canny", None)],
        reference_video_path=tmp_path / "ref.mp4",
    )
    assert captured[0]["loras"][0]["strength"] == 1.0
    assert captured[0]["reference_video"]["preprocess"] == "canny"


# --------------------------------------------------------------------------- #
# 3d) golden chain payload — the worker contract, byte for byte
# --------------------------------------------------------------------------- #
#
# Same discipline as section 1's load payload: the key SET and the key ORDER
# are the contract (JSON preserves insertion order, so a reorder IS a byte
# change), and the payload is BUILT here rather than transcribed.

#: The two acceleration keys 高速化第1弾 added, in adapter order. They are the
#: TAIL of every payload rather than part of the golden lists below, because the
#: adapter appends them LAST — after every other additive block — so the older
#: blocks' positions are provably untouched. Both pydantic defaults are True, so
#: a plain request carries both and every key-order assertion below ends here.
GOLDEN_ACCEL_KEYS_25 = ["block_swap_prefetch", "fused_gguf_dequant_kernel"]

#: The default chain payload's key order. ``stage2_window`` is deliberately
#: absent: it is additive and rides only when the request opted off "standard".
GOLDEN_CHAIN_KEYS_25 = [
    "op",
    "width",
    "height",
    "frame_rate",
    "num_steps",
    "seed",
    "overlap_frames",
    "overlap_strength",
    "chunked_upsample",
    "output_path",
    "clips",
]


def _capturing_chain_backend(captured: list[dict]) -> ltx25._RealBackend25:
    """A ``_RealBackend25`` with no subprocess behind it.

    Mirrors the ``_RealBackend.__new__`` harness in
    tests/test_ltx_runner_payload.py: ``loaded`` is a read-only property over
    ``_proc.poll()``, so a live-looking fake proc makes it True and
    ``generate_chain`` skips the spawn.
    """
    be = ltx25._RealBackend25.__new__(ltx25._RealBackend25)
    be._proc = types.SimpleNamespace(poll=lambda: None)  # type: ignore[attr-defined]
    be._lock = threading.Lock()

    def _send(msg: dict) -> None:
        captured.append(msg)
        out = Path(msg["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 16)

    be._send = _send  # type: ignore[attr-defined]
    be._read_chain_events = lambda cb: {  # type: ignore[attr-defined]
        "event": "done",
        "seed_used": 123,
        "peak_vram_mb": 7000,
        # 台帳 §3-131: the chain's own ``ltx25`` sub-dict lives INSIDE "chain",
        # not as a sibling key — the worker never emits a top-level ``ltx25``
        # on a chain job (see the outcome test below).
        "chain": {
            "total_px": 41,
            "num_clips": 2,
            "ltx25": {
                "stage1_sampler": "ancestral",
                "stage1_eta": 1.0,
                "chunked_upsample": False,
                "tiling": None,
                "phases": {},
            },
        },
        # 高速化第1弾: the worker echoes what ACTUALLY happened for both knobs,
        # and the adapter relays it verbatim (see the outcome test below).
        "block_swap_prefetch_used": "on",
        "fused_gguf_dequant_kernel_used": "on",
        # 高速化第2弾: a third echo, and one that only ever says "on" or "off" —
        # engine25 has no degrade path for the resident text encoder.
        "keep_resident_used": "on",
        # 高速化第3弾: a FOURTH echo. It used to be hard-coded in the adapter
        # ("sdpa", because the field was a 422 on this engine); now it comes
        # from the worker like the three above, so the fake has to speak it or
        # the relay would answer None. "sdpa" here — a plain chain asks for
        # nothing else — and the positive "sage" case gets its own test below.
        "attention_used": "sdpa",
        # 台帳 §3-131: the decoder-name echo, same vocabulary as the single
        # path's fake above.
        "vae_mode_used": "conv",
    }
    return be


def test_chain_payload_golden_for_a_plain_two_clip_chain(tmp_path):
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(_chain_request(), output_dir=tmp_path / "out")

    assert len(captured) == 1
    payload = captured[0]
    assert list(payload) == GOLDEN_CHAIN_KEYS_25 + GOLDEN_ACCEL_KEYS_25
    assert payload == {
        "op": "generate_chain",
        "width": 512,
        "height": 320,
        "frame_rate": 24.0,
        "num_steps": 8,
        "seed": 123,
        "overlap_frames": 2,
        "overlap_strength": 0.5,
        "chunked_upsample": False,
        "output_path": str(tmp_path / "out" / "output.mp4"),
        "clips": [
            {"prompt": "a quiet harbour at first light", "num_frames": 25, "images": []},
            {"prompt": "a quiet harbour at first light", "num_frames": 25, "images": []},
        ],
        # Both pydantic defaults are True, so a PLAIN chain asks for both — the
        # additive contract is proved from the other side by
        # ``test_chain_payload_omits_the_acceleration_keys_when_explicitly_off``.
        "block_swap_prefetch": True,
        "fused_gguf_dequant_kernel": True,
    }


def test_chain_payload_omits_the_acceleration_keys_when_explicitly_off(tmp_path):
    """The ADDITIVE half of 高速化第1弾's contract, and the reason the adapter
    writes ``if chain.<field>:`` rather than sending the boolean.

    Turning a knob off must leave the payload byte-identical to the
    pre-acceleration golden, not send ``False``: an absent key already means off
    on the worker side, so a sent ``False`` would be a second spelling of the
    same silence — and the two spellings would drift."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(
        _chain_request(block_swap_prefetch=False, fused_gguf_dequant_kernel=False),
        output_dir=tmp_path / "off",
    )
    assert list(captured[0]) == GOLDEN_CHAIN_KEYS_25

    # ...and one knob off leaves exactly the OTHER key, in adapter order.
    be.generate_chain(
        _chain_request(block_swap_prefetch=False),
        output_dir=tmp_path / "half",
    )
    assert list(captured[1]) == GOLDEN_CHAIN_KEYS_25 + ["fused_gguf_dequant_kernel"]


def test_chain_payload_carries_keep_resident_only_when_asked(tmp_path):
    """高速化第2弾's half of the additive contract, and the direction that
    matters most here: this knob's pydantic default is FALSE, so the DEFAULT
    chain must look exactly as it did before the field was wired — the golden
    plus 第1弾's pair, and nothing else. Turning it on appends ONE key, and it
    lands at the very END, after the two 第1弾 keys, because the adapter appends
    the newest block last and every earlier block's position is the contract.

    Explicit ``False`` is tested beside the default on purpose: the frontend
    sends the whole schema on every request, so "the user left it off" arrives
    as a literal ``false`` far more often than as an omission, and the two must
    produce the same payload — an absent key IS the release request the worker
    acts on."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)

    be.generate_chain(_chain_request(), output_dir=tmp_path / "default")
    assert list(captured[0]) == GOLDEN_CHAIN_KEYS_25 + GOLDEN_ACCEL_KEYS_25

    be.generate_chain(_chain_request(keep_resident=False), output_dir=tmp_path / "explicit")
    assert list(captured[1]) == GOLDEN_CHAIN_KEYS_25 + GOLDEN_ACCEL_KEYS_25
    assert captured[1] == captured[0] | {"output_path": captured[1]["output_path"]}

    be.generate_chain(_chain_request(keep_resident=True), output_dir=tmp_path / "on")
    assert list(captured[2]) == GOLDEN_CHAIN_KEYS_25 + GOLDEN_ACCEL_KEYS_25 + ["keep_resident"]
    assert captured[2]["keep_resident"] is True


def test_chain_payload_carries_per_clip_prompts_and_clip0_images(tmp_path):
    """A per-clip prompt override wins over the global one, and ONLY clip 0 may
    carry conditioning images (the schema enforces that; the payload shows it)."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    request = _chain_request(
        clips=[
            {
                "num_frames": 25,
                "prompt": "a harbour at dusk",
                "conditioning_images": [{"image_id": "img-1", "frame_idx": 0, "strength": 0.9}],
            },
            {"num_frames": 25},
        ]
    )
    be.generate_chain(
        request,
        output_dir=tmp_path / "out",
        clip0_conditioning_paths=[tmp_path / "img-1.png"],
    )

    clips = captured[0]["clips"]
    assert clips[0]["prompt"] == "a harbour at dusk"
    assert clips[1]["prompt"] == "a quiet harbour at first light"
    assert clips[0]["images"] == [
        {"path": str(tmp_path / "img-1.png"), "frame_idx": 0, "strength": 0.9}
    ]
    assert clips[1]["images"] == []


def test_chain_payload_carries_stage2_window_only_when_non_default(tmp_path):
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(_chain_request(), output_dir=tmp_path / "a")
    assert "stage2_window" not in captured[0]

    be.generate_chain(
        _chain_request(stage2_window="high_resolution"), output_dir=tmp_path / "b"
    )
    assert captured[1]["stage2_window"] == "high_resolution"
    # ...appended AFTER the golden keys, so the default key order is untouched.
    assert list(captured[1]) == GOLDEN_CHAIN_KEYS_25 + ["stage2_window"] + GOLDEN_ACCEL_KEYS_25


def test_chain_payload_carries_chunked_upsample(tmp_path):
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(_chain_request(chunked_upsample=True), output_dir=tmp_path / "out")
    assert captured[0]["chunked_upsample"] is True


#: The V2V / A2V payload blocks are ADDITIVE and their key order is part of the
#: contract, exactly as the top-level key order is — and it is 2.3's key order
#: on purpose (services/engines/ltx/adapter.py), because the two workers are
#: unrelated code but the body of a chain job is the same geometry in both.
GOLDEN_SOURCE_KEYS_25 = ["path", "context_frames"]
GOLDEN_AUDIO_SOURCE_KEYS_25 = ["path"]

#: The retake block's key order, and 2.3's for the same reason as the two above
#: (services/engines/ltx/adapter.py): a retake's geometry is the same in both
#: engines, so its payload block is too.
GOLDEN_RETAKE_KEYS_25 = ["path", "head_px", "tail_px", "regenerate_audio"]

#: The end-source block's key order, 2.3's again and for the same reason.
#: ``strength`` is part of the BLOCK rather than resolved in the engine because
#: a ``None`` there means "the request left it at its default", which is a
#: statement about the schema and belongs on the app side of the wire.
GOLDEN_END_SOURCE_KEYS_25 = ["path", "context_frames", "strength"]


def test_chain_payload_carries_the_v2v_source_block(tmp_path):
    """V2V: the app-cut fps-correct tail plus the context length, appended AFTER
    the golden keys so a plain chain's payload is untouched."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    tail = tmp_path / "src" / "_source_tail.mp4"
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_SOURCES["source_video"]),
        output_dir=tmp_path / "out",
        source_tail_path=tail,
        source_context_frames=25,
    )

    payload = captured[0]
    assert list(payload) == GOLDEN_CHAIN_KEYS_25 + ["source"] + GOLDEN_ACCEL_KEYS_25
    assert list(payload["source"]) == GOLDEN_SOURCE_KEYS_25
    assert payload["source"] == {"path": str(tail), "context_frames": 25}
    # The two source modes are mutually exclusive in the schema; the payload
    # shows it rather than merely relying on it.
    assert "audio_source" not in payload


def test_the_v2v_source_block_needs_both_halves(tmp_path):
    """The guard is on BOTH values, like 2.3's: a tail with no context length is
    not a V2V job, and half a block reaching the worker would be worse than
    none — it would be a payload no golden pins."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(
        _chain_request(), output_dir=tmp_path / "a", source_tail_path=tmp_path / "t.mp4"
    )
    assert "source" not in captured[0]

    be.generate_chain(_chain_request(), output_dir=tmp_path / "b", source_context_frames=25)
    assert "source" not in captured[1]
    # ...and a plain chain is byte-identical to the golden either way.
    assert (
        list(captured[0])
        == list(captured[1])
        == GOLDEN_CHAIN_KEYS_25 + GOLDEN_ACCEL_KEYS_25
    )


def test_chain_payload_carries_the_a2v_audio_source_block(tmp_path):
    """A2V: the uploaded wav, passed as-is — the engine truncates the encoded
    latent to the timeline, so the app sends no geometry with it."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    wav = tmp_path / "up" / "input.wav"
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_SOURCES["source_audio"]),
        output_dir=tmp_path / "out",
        source_audio_path=wav,
    )

    payload = captured[0]
    assert list(payload) == GOLDEN_CHAIN_KEYS_25 + ["audio_source"] + GOLDEN_ACCEL_KEYS_25
    assert list(payload["audio_source"]) == GOLDEN_AUDIO_SOURCE_KEYS_25
    assert payload["audio_source"] == {"path": str(wav)}
    assert "source" not in payload


def test_chain_payload_carries_the_retake_block(tmp_path):
    """Retake: the app-cut window plus the glue geometry, appended AFTER the
    golden keys and AFTER the two source blocks, so a plain chain's payload is
    untouched and the older additive blocks keep their positions."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    window = tmp_path / "job" / "_retake_window.mp4"
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_RETAKE["retake"]),
        output_dir=tmp_path / "out",
        retake_window_path=window,
    )

    payload = captured[0]
    assert list(payload) == GOLDEN_CHAIN_KEYS_25 + ["retake"] + GOLDEN_ACCEL_KEYS_25
    assert list(payload["retake"]) == GOLDEN_RETAKE_KEYS_25
    assert payload["retake"] == {
        "path": str(window),
        # The schema's calibrated defaults (VERIFICATION_LOG §55.5), passed
        # through as INTEGERS — the engine indexes latents with them.
        "head_px": 25,
        "tail_px": 24,
        "regenerate_audio": True,
    }
    # A retake owns the whole timeline; the schema makes it exclusive with both
    # source modes, and the payload shows it rather than merely relying on it.
    assert "source" not in payload and "audio_source" not in payload


def test_the_retake_block_carries_non_default_geometry_and_the_audio_flag(tmp_path):
    """The three fields that are not the path really are read off the request,
    not defaulted a second time here. ``regenerate_audio=False`` is the one that
    changes what is DELIVERED (the window's own waveform, re-muxed, with the
    vocoder skipped), so a payload that dropped it would silently hand the user
    a re-synthesised soundtrack."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_RETAKE["retake_explicit_glue"]),
        output_dir=tmp_path / "a",
        retake_window_path=tmp_path / "w.mp4",
    )
    assert captured[0]["retake"]["head_px"] == 33
    assert captured[0]["retake"]["tail_px"] == 32

    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_RETAKE["retake_keep_original_audio"]),
        output_dir=tmp_path / "b",
        retake_window_path=tmp_path / "w.mp4",
    )
    assert captured[1]["retake"]["regenerate_audio"] is False


def test_the_retake_block_needs_both_the_window_and_the_request_field(tmp_path):
    """The guard is on BOTH, like 2.3's: a window path with no ``retake`` in the
    request is not a retake job, and half a block reaching the worker would be a
    payload no golden pins. Either half alone leaves the plain golden."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)

    # material without the request field...
    be.generate_chain(
        _chain_request(), output_dir=tmp_path / "a", retake_window_path=tmp_path / "w.mp4"
    )
    assert "retake" not in captured[0]

    # ...and the request field without material (the orchestrator failed to cut).
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_RETAKE["retake"]), output_dir=tmp_path / "b"
    )
    assert "retake" not in captured[1]
    assert (
        list(captured[0])
        == list(captured[1])
        == GOLDEN_CHAIN_KEYS_25 + GOLDEN_ACCEL_KEYS_25
    )


def test_the_retake_window_is_no_longer_out_of_scope_material(tmp_path):
    """The other half of the fail-loud guard's Retake change: a window ARRIVING
    here is a JOB now, not a drift between the table and the signature — so it
    must reach the payload rather than raise. The negative twin of
    ``test_generate_chain_fails_loud_on_out_of_scope_material``."""
    captured: list[dict] = []
    _capturing_chain_backend(captured).generate_chain(
        _chain_request(**CHAIN_ACCEPTED_RETAKE["retake"]),
        output_dir=tmp_path / "ok",
        retake_window_path=tmp_path / "w.mp4",
    )
    assert captured[0]["retake"]["path"] == str(tmp_path / "w.mp4")


def test_chain_payload_carries_the_end_source_block(tmp_path):
    """End source: the app-prepared tail material plus the band length and the
    strength, appended AFTER the golden keys and AFTER the retake block, so a
    plain chain's payload is untouched and every older additive block keeps its
    position."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    material = tmp_path / "job" / "_end_source.mp4"
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_END_SOURCE["end_source"]),
        output_dir=tmp_path / "out",
        end_source_path=material,
        end_source_context_frames=24,
    )

    payload = captured[0]
    assert list(payload) == GOLDEN_CHAIN_KEYS_25 + ["end_source"] + GOLDEN_ACCEL_KEYS_25
    assert list(payload["end_source"]) == GOLDEN_END_SOURCE_KEYS_25
    assert payload["end_source"] == {
        "path": str(material),
        "context_frames": 24,
        # NOT passed by the caller above: an absent strength is the schema's
        # default, resolved here so the engine never has to know what None means.
        "strength": 1.0,
    }
    assert "retake" not in payload


def test_the_end_source_block_carries_a_non_default_strength(tmp_path):
    """The knob really is read off the argument rather than defaulted a second
    time here. A payload that dropped it would silently HARD-freeze a band the
    user asked to hold softly — and at the default 1.0 no test could see it,
    which is why the arm is non-default."""
    captured: list[dict] = []
    _capturing_chain_backend(captured).generate_chain(
        _chain_request(**CHAIN_ACCEPTED_END_SOURCE["end_source_strength"]),
        output_dir=tmp_path / "a",
        end_source_path=tmp_path / "e.mp4",
        end_source_context_frames=24,
        end_source_strength=0.7,
    )
    assert captured[0]["end_source"]["strength"] == 0.7


def test_the_end_source_block_needs_both_the_material_and_the_band_length(tmp_path):
    """The guard is on BOTH values, like 2.3's and like the retake block's:
    half a block reaching the worker would be a payload no golden pins. Either
    half alone leaves the plain golden."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)

    # material without the band length...
    be.generate_chain(
        _chain_request(), output_dir=tmp_path / "a", end_source_path=tmp_path / "e.mp4"
    )
    assert "end_source" not in captured[0]

    # ...and the band length without material (the orchestrator failed to cut).
    be.generate_chain(
        _chain_request(), output_dir=tmp_path / "b", end_source_context_frames=24
    )
    assert "end_source" not in captured[1]
    assert (
        list(captured[0])
        == list(captured[1])
        == GOLDEN_CHAIN_KEYS_25 + GOLDEN_ACCEL_KEYS_25
    )


def test_the_end_source_block_rides_beside_a_v2v_source(tmp_path):
    """The one combination the mode has: start material AND end material on a
    single clip — the interpolation the API explicitly allows. BOTH additive
    blocks must ride, in the order the builder appends them (``source`` first,
    ``end_source`` after the retake slot), or one of the two is silently lost."""
    captured: list[dict] = []
    _capturing_chain_backend(captured).generate_chain(
        _chain_request(**CHAIN_ACCEPTED_END_SOURCE["end_source_with_source_video"]),
        output_dir=tmp_path / "out",
        source_tail_path=tmp_path / "_source_tail.mp4",
        source_context_frames=25,
        end_source_path=tmp_path / "_end_source.mp4",
        end_source_context_frames=24,
    )
    payload = captured[0]
    assert list(payload) == (
        GOLDEN_CHAIN_KEYS_25 + ["source", "end_source"] + GOLDEN_ACCEL_KEYS_25
    )
    assert payload["source"]["context_frames"] == 25
    assert payload["end_source"]["context_frames"] == 24


def test_chain_payload_carries_a2v_with_the_full_length_window(tmp_path):
    """Single-tab A2V on the wire: ONE clip plus the full-length stage-2 window.
    Both additive keys ride, and in the order the builder appends them — the
    source blocks first, ``stage2_window`` last, as 2.3 does."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_SOURCES["source_audio_full_length"]),
        output_dir=tmp_path / "out",
        source_audio_path=tmp_path / "input.wav",
    )

    payload = captured[0]
    assert (
        list(payload)
        == GOLDEN_CHAIN_KEYS_25 + ["audio_source", "stage2_window"] + GOLDEN_ACCEL_KEYS_25
    )
    assert payload["stage2_window"] == "full_length"
    assert len(payload["clips"]) == 1


def test_chain_payload_carries_a2v_across_a_long_chain(tmp_path):
    """Long A2V: ONE uploaded audio drives three clips. There is no per-clip
    audio — the engine tiles the single latent across the stage-1 segments — so
    the payload must not grow a per-clip audio key."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_SOURCES["source_audio_long"]),
        output_dir=tmp_path / "out",
        source_audio_path=tmp_path / "input.wav",
    )

    payload = captured[0]
    assert list(payload) == GOLDEN_CHAIN_KEYS_25 + ["audio_source"] + GOLDEN_ACCEL_KEYS_25
    assert len(payload["clips"]) == 3
    assert all(set(clip) == {"prompt", "num_frames", "images"} for clip in payload["clips"])


#: The IC-LoRA payload blocks' key order, also 2.3's verbatim (see
#: services/engines/ltx/adapter.py ``generate_chain``). ``attention_strength``
#: is spliced onto the end of the reference block only when the request set it.
GOLDEN_LORA_ENTRY_KEYS_25 = ["path", "strength"]
GOLDEN_REFERENCE_KEYS_25 = ["path", "strength", "preprocess"]


def _resolved(path, strength=1.0, preprocess="none", audio_strength=None):
    """A ``ResolvedLora`` the way services/lora_registry.py builds one."""
    return ResolvedLora(
        path=path, strength=strength, preprocess=preprocess, audio_strength=audio_strength
    )


def test_chain_payload_carries_the_style_lora_block(tmp_path):
    """Style/character LoRA on a chain (§3-102 third increment): additive,
    applied uniformly across the clips, and appended AFTER the golden keys so a
    no-lora chain's payload is untouched."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    adapter = tmp_path / "Pixar_Toon.safetensors"
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_LORAS["loras"]),
        output_dir=tmp_path / "out",
        lora_paths=[_resolved(adapter, 1.0)],
    )

    payload = captured[0]
    assert list(payload) == GOLDEN_CHAIN_KEYS_25 + ["loras"] + GOLDEN_ACCEL_KEYS_25
    assert list(payload["loras"][0]) == GOLDEN_LORA_ENTRY_KEYS_25
    assert payload["loras"] == [{"path": str(adapter), "strength": 1.0}]
    # A style-only chain asks for no reference video, so the key is absent —
    # additive on the chain path, unlike the single path where it is always there.
    assert "reference_video" not in payload


def test_chain_payload_carries_audio_strength_only_when_the_adapter_has_one(tmp_path):
    """§45's app-side half: ``audio_strength`` is spliced into an entry only when
    the resolved adapter carries one, so a no-audio job's payload is unchanged."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    a = tmp_path / "a.safetensors"
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_LORAS["loras"]),
        output_dir=tmp_path / "out",
        lora_paths=[_resolved(a, 1.0, audio_strength=0.4)],
    )
    assert captured[0]["loras"] == [
        {"path": str(a), "strength": 1.0, "audio_strength": 0.4}
    ]


def test_chain_payload_carries_the_reference_block_for_a_long_chain(tmp_path):
    """Long IC-LoRA (§3-78 geometry, now on 2.5): ONE reference video drives
    TWO clips. There is no per-clip reference — the engine slices the single
    stream per stage-1 segment — so the payload must not grow a per-clip key."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    adapter = tmp_path / "union-control.safetensors"
    ref = tmp_path / "ref.mp4"
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_LORAS["reference_video_long"]),
        output_dir=tmp_path / "out",
        lora_paths=[_resolved(adapter, 1.0, preprocess="canny")],
        reference_video_path=ref,
    )

    payload = captured[0]
    assert (
        list(payload)
        == GOLDEN_CHAIN_KEYS_25 + ["loras", "reference_video"] + GOLDEN_ACCEL_KEYS_25
    )
    assert list(payload["reference_video"]) == GOLDEN_REFERENCE_KEYS_25
    assert payload["reference_video"] == {
        "path": str(ref),
        "strength": 1.0,  # official guidance default when the request omits it
        "preprocess": "canny",
    }
    assert len(payload["clips"]) == 2
    assert all(set(clip) == {"prompt", "num_frames", "images"} for clip in payload["clips"])


def test_chain_reference_block_carries_the_two_strength_overrides(tmp_path):
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_LORAS["reference_video_strength"]),
        output_dir=tmp_path / "out",
        lora_paths=[_resolved(tmp_path / "u.safetensors", 1.0, preprocess="depth")],
        reference_video_path=tmp_path / "ref.mp4",
    )
    reference = captured[0]["reference_video"]
    assert reference["strength"] == 0.8
    assert reference["attention_strength"] == 0.7
    # ...appended AFTER the golden keys, so an omitted-field job's block order
    # is untouched.
    assert list(reference) == GOLDEN_REFERENCE_KEYS_25 + ["attention_strength"]


def test_chain_payload_omits_both_lora_blocks_by_default(tmp_path):
    """The additive contract, stated from the other side: a plain chain is
    byte-identical to the golden even though the two keywords were passed."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(_chain_request(), output_dir=tmp_path / "out", lora_paths=[])
    assert list(captured[0]) == GOLDEN_CHAIN_KEYS_25 + GOLDEN_ACCEL_KEYS_25


def test_chain_outcome_names_this_engine_and_relays_the_chain_metadata(tmp_path):
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    outcome = be.generate_chain(_chain_request(), output_dir=tmp_path / "out")

    assert outcome.generation_mode == "chain"
    assert outcome.backend == ltx25.REAL_BACKEND_25
    assert outcome.chain_metadata == {
        "total_px": 41,
        "num_clips": 2,
        "ltx25": {
            "stage1_sampler": "ancestral",
            "stage1_eta": 1.0,
            "chunked_upsample": False,
            "tiling": None,
            "phases": {},
        },
    }
    assert outcome.seed_used == 123
    assert outcome.peak_vram_mb == 7000
    # 高速化第3弾: "sdpa" is what the WORKER said (the fake echoes it), not a
    # constant the adapter writes — the positive "sage" case is the test below.
    # 台帳 §3-131: ``vae_mode_used`` is REPURPOSED as engine25's own
    # decoder-name echo ("diff" / "conv") — no longer a permanent None here.
    assert outcome.attention_used == "sdpa"
    assert outcome.vae_mode_used == "conv"
    # 高速化第1弾: these two DO name engine25 code paths now, so the worker's
    # echo rides through to metadata.json instead of being dropped.
    assert outcome.block_swap_prefetch_used == "on"
    assert outcome.fused_gguf_dequant_kernel_used == "on"
    # 高速化第2弾: and so does the third. It used to be None here, because the
    # field was a 422 on this engine.
    assert outcome.keep_resident_used == "on"


def test_chain_outcome_ltx25_field_stays_none_the_dict_lives_in_chain_metadata(tmp_path):
    """台帳 §3-131: ``GenerationOutcome.ltx25`` is single-job only by contract —
    a chain's engine25 facts live inside ``chain_metadata["ltx25"]`` instead
    (``chain=meta`` already carries them; the worker never emits a top-level
    ``ltx25`` on a chain ``done`` event, see ``_capturing_chain_backend``)."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    outcome = be.generate_chain(_chain_request(), output_dir=tmp_path / "out")

    assert outcome.ltx25 is None
    assert outcome.chain_metadata["ltx25"] == {
        "stage1_sampler": "ancestral",
        "stage1_eta": 1.0,
        "chunked_upsample": False,
        "tiling": None,
        "phases": {},
    }


def test_chain_payload_carries_the_attention_backend_only_when_asked(tmp_path):
    """高速化第3弾, the additive contract from both sides at once: a plain chain
    carries NO ``attention_backend`` key (so the golden above stays byte-exact),
    and a chain that asked for sage carries it — appended LAST, after the three
    acceleration keys that predate it."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(_chain_request(), output_dir=tmp_path / "plain")
    assert "attention_backend" not in captured[0]

    # ...nor does an EXPLICIT "sdpa", the shape the frontend actually sends.
    captured.clear()
    be = _capturing_chain_backend(captured)
    be.generate_chain(
        _chain_request(attention_backend="sdpa"), output_dir=tmp_path / "explicit"
    )
    assert "attention_backend" not in captured[0]

    captured.clear()
    be = _capturing_chain_backend(captured)
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_SAGE["attention_backend"]),
        output_dir=tmp_path / "sage",
    )
    assert captured[0]["attention_backend"] == "sage"
    assert list(captured[0])[-1] == "attention_backend"


def test_chain_payload_carries_the_nag_block_only_when_enabled(tmp_path):
    """The NAG/VSF increment's additive contract, from both sides at once.

    THE DEFAULT DIRECTION IS THE LOAD-BEARING ONE: ``nag_enabled``'s pydantic
    default is False and the frontend sends the whole schema every time, so a
    chain that merely CARRIES the seven fields at their defaults must produce a
    payload byte-identical to the golden -- which is why no golden needed an
    edit for this increment.

    And when it IS enabled, the block rides LAST, after every acceleration key,
    so no earlier key order moved."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(_chain_request(), output_dir=tmp_path / "plain")
    assert "nag" not in captured[0]

    # ...nor does a body that spells every negative-prompt field out at its
    # default, which is what the WebUI actually posts.
    captured.clear()
    be = _capturing_chain_backend(captured)
    be.generate_chain(
        _chain_request(
            nag_enabled=False,
            negative_prompt="",
            nag_scale=11.0,
            nag_tau=2.5,
            nag_alpha=0.25,
            neg_method="nag",
            vsf_scale=1.5,
        ),
        output_dir=tmp_path / "explicit",
    )
    assert "nag" not in captured[0]

    captured.clear()
    be = _capturing_chain_backend(captured)
    be.generate_chain(
        _chain_request(**CHAIN_ACCEPTED_NAG["nag_scale"]), output_dir=tmp_path / "nag"
    )
    assert list(captured[0])[-1] == "nag"
    # KEY FOR KEY 2.3's block, in 2.3's order, and the wire name for the method
    # is ``method`` rather than the request's ``neg_method``.
    assert list(captured[0]["nag"]) == [
        "negative_prompt",
        "scale",
        "tau",
        "alpha",
        "method",
        "vsf_scale",
    ]
    assert captured[0]["nag"] == {
        "negative_prompt": "blurry, low quality",
        "scale": 7.5,
        "tau": 2.5,
        "alpha": 0.25,
        "method": "nag",
        "vsf_scale": 1.5,
    }


def test_chain_payload_carries_the_vsf_method_and_scale(tmp_path):
    """VSF is a METHOD SWITCH inside the same block, not a second block: the
    NAG knobs keep riding unconditionally (the engine reads them only when
    ``method == "nag"``), and what changes is two values. A payload that dropped
    ``vsf_scale`` would run VSF at the reference default and no log would say
    the user's number had been lost."""
    captured: list[dict] = []
    _capturing_chain_backend(captured).generate_chain(
        _chain_request(**CHAIN_ACCEPTED_NAG["vsf_scale"]), output_dir=tmp_path / "vsf"
    )
    assert captured[0]["nag"]["method"] == "vsf"
    assert captured[0]["nag"]["vsf_scale"] == 3.0
    assert captured[0]["nag"]["scale"] == 11.0  # the NAG knob, still carried


def test_chain_outcome_relays_a_sage_echo_verbatim(tmp_path):
    """THE POSITIVE DIRECTION, which the "sdpa" assertions above cannot reach:
    a hard-coded ``attention_used="sdpa"`` would have passed every one of them.
    Only a worker that says "sage" tells the two apart — and the fold a chain
    returns ("sage->sdpa", one build fell back) must ride through untidied, or
    metadata.json would claim a kernel the job did not run."""
    for echoed in ("sage", "sage->sdpa"):
        captured: list[dict] = []
        be = _capturing_chain_backend(captured)
        base = be._read_chain_events(None)  # type: ignore[attr-defined]
        be._read_chain_events = lambda cb, e=echoed: {  # type: ignore[attr-defined]
            **base,
            "attention_used": e,
        }
        outcome = be.generate_chain(
            _chain_request(**CHAIN_ACCEPTED_SAGE["attention_backend"]),
            output_dir=tmp_path / echoed.replace(">", "_"),
        )
        assert outcome.attention_used == echoed


def test_chain_crop_output_is_an_app_side_post_process(tmp_path, monkeypatch):
    """crop_output is honoured on a chain for the same reason it is on a single
    job: it is ffmpeg, not the engine. The worker writes ``_full.mp4``."""
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    cropped: list[tuple] = []

    def _fake_crop(src, dst, width, height):
        cropped.append((Path(src).name, width, height))
        Path(dst).write_bytes(b"\x00" * 16)

    monkeypatch.setattr(ltx25.video_io, "crop_mp4", _fake_crop)
    be.generate_chain(
        _chain_request(crop_output={"width": 384, "height": 256}),
        output_dir=tmp_path / "out",
    )
    assert Path(captured[0]["output_path"]).name == "_full.mp4"
    assert cropped == [("_full.mp4", 384, 256)]


# --------------------------------------------------------------------------- #
# 3f) golden SINGLE-generate payload — the worker contract, byte for byte
# --------------------------------------------------------------------------- #
#
# Same discipline as the load and chain goldens: the key SET and the key ORDER
# are the contract, and the payload is BUILT here rather than transcribed.
# ``loras`` and ``reference_video`` are ALWAYS keys on this path — an empty list
# means "detach whatever was attached" and None means "no reference", and a
# worker that has to distinguish "absent" from "empty" is a worker with two
# meanings for one silence. (The CHAIN path is additive instead; both choices
# are 2.3's, restated per path.)

#: The single-generate payload's key order.
GOLDEN_GENERATE_KEYS_25 = [
    "op",
    "prompt",
    "seed",
    "width",
    "height",
    "num_frames",
    "frame_rate",
    "images",
    "loras",
    "reference_video",
    "output_path",
]


def _capturing_backend(captured: list[dict]) -> ltx25._RealBackend25:
    """A ``_RealBackend25`` with no subprocess behind it (single-generate twin
    of :func:`_capturing_chain_backend`)."""
    be = ltx25._RealBackend25.__new__(ltx25._RealBackend25)
    be._proc = types.SimpleNamespace(poll=lambda: None)  # type: ignore[attr-defined]
    be._lock = threading.Lock()

    def _send(msg: dict) -> None:
        captured.append(msg)
        out = Path(msg["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 16)

    be._send = _send  # type: ignore[attr-defined]
    be._read_worker_events = lambda cb, chain, prefix: {  # type: ignore[attr-defined]
        "event": "done",
        "seed_used": 4242,
        "peak_vram_mb": 7000,
        # 高速化第1弾: the single-path twin of the chain fake's echo.
        "block_swap_prefetch_used": "on",
        "fused_gguf_dequant_kernel_used": "on",
        # 高速化第2弾: the third echo, "on"/"off" only.
        "keep_resident_used": "on",
        # 高速化第3弾: the fourth, the single-path twin of the chain fake's.
        "attention_used": "sdpa",
        # 台帳 §3-131: the decoder-name echo and this engine's own additive
        # facts, single-generation shape (five keys, no sampler/vram).
        "vae_mode_used": "conv",
        "ltx25": {
            "encode_fps": 24,
            "video_chunks": 1,
            "tiling": None,
            "size_bytes": 16,
            "phases": {},
        },
    }
    return be


def test_generate_payload_golden_for_a_plain_t2v(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate(_request(width=512, height=320, num_frames=25, seed=123), tmp_path / "out")

    payload = captured[0]
    assert list(payload) == GOLDEN_GENERATE_KEYS_25 + GOLDEN_ACCEL_KEYS_25
    assert payload == {
        "op": "generate",
        "prompt": "a quiet harbour at first light",
        "seed": 123,
        "width": 512,
        "height": 320,
        "num_frames": 25,
        "frame_rate": 24.0,
        "images": [],
        # ALWAYS present, even with nothing to say: an empty list is the
        # explicit "detach whatever is attached" the worker acts on.
        "loras": [],
        "reference_video": None,
        "output_path": str(tmp_path / "out" / "output.mp4"),
        # Both pydantic defaults are True, so a plain T2V asks for both.
        "block_swap_prefetch": True,
        "fused_gguf_dequant_kernel": True,
    }


def test_generate_payload_omits_the_acceleration_keys_when_explicitly_off(tmp_path):
    """The single-path twin of the chain test above: both knobs off restores the
    pre-acceleration golden exactly, and one knob off leaves exactly one key."""
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate(
        _request(block_swap_prefetch=False, fused_gguf_dequant_kernel=False),
        tmp_path / "off",
    )
    assert list(captured[0]) == GOLDEN_GENERATE_KEYS_25

    be.generate(_request(fused_gguf_dequant_kernel=False), tmp_path / "half")
    assert list(captured[1]) == GOLDEN_GENERATE_KEYS_25 + ["block_swap_prefetch"]
    assert captured[1]["block_swap_prefetch"] is True


def test_generate_payload_carries_keep_resident_only_when_asked(tmp_path):
    """The single-path twin of the chain test in section 3d (高速化第2弾).

    The default direction is the load-bearing one: this knob's pydantic default
    is FALSE, so a plain T2V's payload is byte-identical to what it was before
    the field was wired — which is why the golden above needed no edit. Explicit
    ``False`` must produce that same payload, because the frontend sends the
    whole schema every time and an absent key is not silence: it is the release
    request the worker acts on."""
    captured: list[dict] = []
    be = _capturing_backend(captured)
    # A pinned seed, because the two payloads below are compared to EACH OTHER:
    # an omitted seed is drawn per job, which would differ for a reason that has
    # nothing to do with this field.
    fixed = {"seed": 123}

    be.generate(_request(**fixed), tmp_path / "default")
    assert list(captured[0]) == GOLDEN_GENERATE_KEYS_25 + GOLDEN_ACCEL_KEYS_25

    be.generate(_request(keep_resident=False, **fixed), tmp_path / "explicit")
    assert list(captured[1]) == GOLDEN_GENERATE_KEYS_25 + GOLDEN_ACCEL_KEYS_25
    assert captured[1] == captured[0] | {"output_path": captured[1]["output_path"]}

    be.generate(_request(keep_resident=True, **fixed), tmp_path / "on")
    # Appended LAST — after 第1弾's pair — so no earlier key moved.
    assert list(captured[2]) == GOLDEN_GENERATE_KEYS_25 + GOLDEN_ACCEL_KEYS_25 + ["keep_resident"]
    assert captured[2]["keep_resident"] is True


def test_generate_payload_carries_the_nag_block_only_when_enabled(tmp_path):
    """The single-path twin of the chain test in section 3d.

    Same two directions, same reason the default one is what matters: the seven
    fields ride the schema on every request, so a payload that gained a ``nag``
    key from a DEFAULT body would have broken the golden -- and would have armed
    a negative prompt on a job that never asked for one."""
    captured: list[dict] = []
    be = _capturing_backend(captured)
    fixed = {"seed": 123}

    be.generate(_request(**fixed), tmp_path / "plain")
    assert "nag" not in captured[0]

    be.generate(
        _request(
            nag_enabled=False,
            negative_prompt="",
            nag_scale=11.0,
            nag_tau=2.5,
            nag_alpha=0.25,
            neg_method="nag",
            vsf_scale=1.5,
            **fixed,
        ),
        tmp_path / "explicit",
    )
    assert "nag" not in captured[1]
    assert captured[1] == captured[0] | {"output_path": captured[1]["output_path"]}

    be.generate(_request(**REQUEST_ACCEPTED_NAG["nag_alpha"], **fixed), tmp_path / "nag")
    # Appended LAST -- after the acceleration keys and after outpaint -- so no
    # earlier key moved.
    assert list(captured[2])[-1] == "nag"
    assert list(captured[2]["nag"]) == [
        "negative_prompt",
        "scale",
        "tau",
        "alpha",
        "method",
        "vsf_scale",
    ]
    assert captured[2]["nag"] == {
        "negative_prompt": "blurry, low quality",
        "scale": 11.0,
        "tau": 2.5,
        "alpha": 0.5,
        "method": "nag",
        "vsf_scale": 1.5,
    }


def test_generate_payload_carries_the_vsf_method_and_scale(tmp_path):
    """The single-path twin: the method switch and its own scale, in the one
    block. See the chain test for why both are asserted."""
    captured: list[dict] = []
    _capturing_backend(captured).generate(
        _request(**REQUEST_ACCEPTED_NAG["vsf_scale"], seed=123), tmp_path / "vsf"
    )
    assert captured[0]["nag"]["method"] == "vsf"
    assert captured[0]["nag"]["vsf_scale"] == 3.0
    assert captured[0]["nag"]["scale"] == 11.0


def test_generate_payload_carries_the_style_lora_entries(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    a = tmp_path / "Pixar_Toon.safetensors"
    b = tmp_path / "second.safetensors"
    be.generate(
        _request(**REQUEST_ACCEPTED_LORAS["loras"]),
        tmp_path / "out",
        lora_paths=[_resolved(a, 1.0), _resolved(b, 0.6, audio_strength=0.4)],
    )

    payload = captured[0]
    assert list(payload) == GOLDEN_GENERATE_KEYS_25 + GOLDEN_ACCEL_KEYS_25
    assert payload["loras"] == [
        {"path": str(a), "strength": 1.0},
        # ``audio_strength`` spliced in only for the adapter that carries one.
        {"path": str(b), "strength": 0.6, "audio_strength": 0.4},
    ]
    # A style-only job still has no reference video to send.
    assert payload["reference_video"] is None


def test_generate_payload_carries_the_reference_block(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    ref = tmp_path / "ref.mp4"
    be.generate(
        _request(**REQUEST_ACCEPTED_LORAS["reference_video_id"], width=512, height=384),
        tmp_path / "out",
        lora_paths=[_resolved(tmp_path / "u.safetensors", 1.0, preprocess="canny")],
        reference_video_path=ref,
    )

    reference = captured[0]["reference_video"]
    assert list(reference) == GOLDEN_REFERENCE_KEYS_25
    assert reference == {
        "path": str(ref),
        "strength": 1.0,  # official guidance default when the request omits it
        "preprocess": "canny",
    }


def test_generate_reference_block_carries_the_two_strength_overrides(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate(
        _request(
            reference_video_id="vid-123",
            loras=_LORAS,
            reference_video_strength=0.8,
            conditioning_attention_strength=0.7,
            width=512,
            height=384,
        ),
        tmp_path / "out",
        lora_paths=[_resolved(tmp_path / "u.safetensors", 1.0, preprocess="depth")],
        reference_video_path=tmp_path / "ref.mp4",
    )
    reference = captured[0]["reference_video"]
    assert reference["strength"] == 0.8
    assert reference["attention_strength"] == 0.7
    assert list(reference) == GOLDEN_REFERENCE_KEYS_25 + ["attention_strength"]


def test_generate_refuses_two_conflicting_preprocess_kinds(tmp_path):
    """One uploaded reference can become ONE control signal. The API layer
    already refuses this; the runner hop re-checks, exactly as 2.3 does — the
    defensive half of a rule whose failure mode is a silently wrong control."""
    captured: list[dict] = []
    be = _capturing_backend(captured)
    with pytest.raises(APIError) as ei:
        be.generate(
            _request(**REQUEST_ACCEPTED_LORAS["reference_video_id"], width=512, height=384),
            tmp_path / "out",
            lora_paths=[
                _resolved(tmp_path / "a.safetensors", 1.0, preprocess="canny"),
                _resolved(tmp_path / "b.safetensors", 1.0, preprocess="depth"),
            ],
            reference_video_path=tmp_path / "ref.mp4",
        )
    assert ei.value.code == "LORA_PREPROCESS_CONFLICT"
    assert not captured, "no payload may be sent for a conflicting job"


def test_generate_outcome_names_this_engine(tmp_path):
    captured: list[dict] = []
    outcome = _capturing_backend(captured).generate(_request(), tmp_path / "out")
    assert outcome.backend == ltx25.REAL_BACKEND_25
    assert outcome.seed_used == 4242
    # 高速化第3弾: the worker's own echo, not a constant — see the positive
    # "sage" test below, which is the only one that can tell the two apart.
    assert outcome.attention_used == "sdpa"
    # 高速化第1弾+第2弾: the three acceleration echoes engine25 has are relayed;
    # the ONE it does not have stays None rather than claiming an untouched
    # "off".
    assert outcome.block_swap_prefetch_used == "on"
    assert outcome.fused_gguf_dequant_kernel_used == "on"
    assert outcome.keep_resident_used == "on"
    # 台帳 §3-131: ``vae_mode_used`` is REPURPOSED as engine25's own
    # decoder-name echo ("diff" / "conv") — no longer a permanent None here.
    # ``ltx25`` is single-job-only additive: the worker's five engine25 facts,
    # relayed verbatim (see ``_capturing_backend``'s fake ``done`` event).
    assert outcome.vae_mode_used == "conv"
    assert outcome.ltx25 == {
        "encode_fps": 24,
        "video_chunks": 1,
        "tiling": None,
        "size_bytes": 16,
        "phases": {},
    }


def test_generate_payload_carries_the_attention_backend_only_when_asked(tmp_path):
    """高速化第3弾, the single-path twin of the chain test: a plain T2V carries
    NO ``attention_backend`` key — which is what keeps the golden payload above
    byte-identical to the pre-sage one, and with it every frozen-SHA piece of
    evidence the earlier increments left behind."""
    captured: list[dict] = []
    _capturing_backend(captured).generate(_request(), tmp_path / "plain")
    assert "attention_backend" not in captured[0]

    # ...and neither does one that names "sdpa" EXPLICITLY, which is what the
    # frontend sends on every request: the predicate is "differs from the
    # default", never "is present", so the golden survives a full schema.
    captured.clear()
    _capturing_backend(captured).generate(
        _request(attention_backend="sdpa"), tmp_path / "explicit"
    )
    assert "attention_backend" not in captured[0]

    captured.clear()
    _capturing_backend(captured).generate(
        _request(**REQUEST_ACCEPTED_SAGE["attention_backend"]), tmp_path / "sage"
    )
    assert captured[0]["attention_backend"] == "sage"
    assert list(captured[0])[-1] == "attention_backend"


def test_generate_payload_is_untouched_when_outpaint_is_absent_or_None(tmp_path):
    """THE BEFORE PICTURE for the Outpainting increment (M-6), landed in commit
    C1 while the engine driver is still INACTIVE.

    C2 will add an ``outpaint`` block to this payload, guarded on
    ``request.outpaint is not None``. The risk that guard carries is not that it
    fails to fire — a gate that generated nothing would be obvious — but that it
    fires when it should not, or that adding it disturbs the key ORDER of the
    payload every other job sends. Either would change the bytes of a plain
    T2V's worker message, and with them every frozen-SHA piece of evidence the
    earlier increments left behind.

    So this pins the plain payload's key SET and key ORDER *now*, with
    ``outpaint`` named explicitly rather than merely omitted: the frontend sends
    the whole schema on every request, so ``outpaint=None`` is the shape that
    actually arrives, and "absent" and "explicitly None" must produce the same
    bytes. Written against the adapter AS IT IS — no adapter line changes in
    C1 — so a re-run after C2 is a real before/after comparison rather than a
    test written to fit the new code.
    """
    captured: list[dict] = []
    be = _capturing_backend(captured)
    # A pinned seed, because the two payloads are compared to EACH OTHER: an
    # omitted seed is drawn per job and would differ for a reason that has
    # nothing to do with this field.
    fixed = {"width": 512, "height": 320, "num_frames": 25, "seed": 123}

    be.generate(_request(**fixed), tmp_path / "absent")
    be.generate(_request(outpaint=None, **fixed), tmp_path / "explicit_none")

    for payload in captured:
        assert list(payload) == GOLDEN_GENERATE_KEYS_25 + GOLDEN_ACCEL_KEYS_25
        assert "outpaint" not in payload
    # Byte-identical but for the destination the two runs were given.
    assert captured[1] == captured[0] | {"output_path": captured[1]["output_path"]}


def test_generate_payload_carries_the_outpaint_block_when_asked(tmp_path):
    """THE AFTER PICTURE (the Outpainting increment). The test above pins that a
    plain job is untouched; this one pins that the block really rides, in 2.3's
    key order, appended LAST.

    The ORDER is the load-bearing half. ``outpaint`` is the newest key on this
    payload, so it goes at the end — every earlier increment's frozen-SHA
    evidence rests on no existing key moving. And the block's key order is 2.3's
    verbatim (services/engines/ltx/adapter.py), because the two engines answer
    to ONE app-side contract per feature: an operator comparing two worker logs
    is then comparing the same eleven names in the same places.

    ``outpaint_source_path`` is the ORIGINAL upload, not the canvas — by this
    point the orchestrator has already substituted the green canvas for
    ``reference_video_path`` — so the two paths in the payload are deliberately
    different files."""
    captured: list[dict] = []
    be = _capturing_backend(captured)
    source = tmp_path / "material.mp4"
    canvas = tmp_path / "outpaint_canvas.mp4"

    be.generate(
        _request(**REQUEST_ACCEPTED_OUTPAINT["outpaint"], seed=123),
        tmp_path / "out",
        lora_paths=[],
        reference_video_path=canvas,
        outpaint_source_path=source,
    )

    payload = captured[0]
    # Appended LAST: after the golden keys AND after the acceleration keys.
    assert list(payload) == GOLDEN_GENERATE_KEYS_25 + GOLDEN_ACCEL_KEYS_25 + ["outpaint"]
    assert payload["outpaint"] == {
        "source_path": str(source),
        "canvas_width": 512,
        "canvas_height": 320,
        "pad_left": 64,
        "pad_right": 0,
        "pad_top": 0,
        "pad_bottom": 0,
        # The workflow's own defaults, sent verbatim rather than defaulted
        # worker-side: the engine must not have a second opinion about the
        # parameter the official note calls "the most important" one.
        "blend_dilation_stage1": 5,
        "blend_dilation_stage2": 2,
        "freeze_source_audio": True,
    }
    # 2.3's key ORDER, not just its key set.
    assert list(payload["outpaint"]) == [
        "source_path", "canvas_width", "canvas_height",
        "pad_left", "pad_right", "pad_top", "pad_bottom",
        "blend_dilation_stage1", "blend_dilation_stage2", "freeze_source_audio",
    ]
    # The canvas rides where it always did — the reference block — so the
    # engine's IC-LoRA plumbing needs no notion of outpainting at all.
    assert payload["reference_video"]["path"] == str(canvas)


def test_generate_outcome_relays_a_sage_echo_verbatim(tmp_path):
    """THE POSITIVE DIRECTION (高速化第3弾). Every "sdpa" assertion in this file
    would also pass against the hard-coded ``attention_used="sdpa"`` the adapter
    used to write, so none of them proves the relay exists. This one does: only
    a worker saying "sage" — or "sage->sdpa", the degrade a build that could not
    load the kernel reports — can be distinguished from a constant."""
    for echoed in ("sage", "sage->sdpa"):
        captured: list[dict] = []
        be = _capturing_backend(captured)
        base = be._read_worker_events(None, False, "generate")  # type: ignore[attr-defined]
        be._read_worker_events = lambda cb, chain, prefix, e=echoed: {  # type: ignore[attr-defined]
            **base,
            "attention_used": e,
        }
        outcome = be.generate(
            _request(**REQUEST_ACCEPTED_SAGE["attention_backend"]),
            tmp_path / echoed.replace(">", "_"),
        )
        assert outcome.attention_used == echoed


def test_generate_outcome_relays_a_degrade_verbatim(tmp_path):
    """The echo is what HAPPENED, not what was asked for: a worker that fell
    back mid-job says so, and the adapter must not tidy that into "on"."""
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be._read_worker_events = lambda cb, chain, prefix: {  # type: ignore[attr-defined]
        "event": "done",
        "seed_used": 4242,
        "peak_vram_mb": 7000,
        "block_swap_prefetch_used": "on->off",
        "fused_gguf_dequant_kernel_used": "off",
    }
    outcome = be.generate(_request(), tmp_path / "out")
    assert outcome.block_swap_prefetch_used == "on->off"
    assert outcome.fused_gguf_dequant_kernel_used == "off"


def test_generate_outcome_leaves_the_echoes_none_when_the_worker_is_silent(tmp_path):
    """A worker too old to echo leaves None — the honest "not reported", which
    metadata.json shows as null rather than inventing an "off"."""
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be._read_worker_events = lambda cb, chain, prefix: {  # type: ignore[attr-defined]
        "event": "done",
        "seed_used": 4242,
        "peak_vram_mb": 7000,
    }
    outcome = be.generate(_request(), tmp_path / "out")
    assert outcome.block_swap_prefetch_used is None
    assert outcome.fused_gguf_dequant_kernel_used is None
    # 高速化第2弾's echo obeys the same rule: nobody reported, so nobody answers.
    assert outcome.keep_resident_used is None
    # 高速化第3弾: and so does the fourth, now that it is a relay. Before this
    # increment it was hard-coded, so a silent worker still produced "sdpa" —
    # an answer nobody had given.
    assert outcome.attention_used is None


# --------------------------------------------------------------------------- #
# 3b) the audit: every GenerateRequest field has exactly one home (M5)
# --------------------------------------------------------------------------- #
#
# THE POINT OF THIS SECTION is that it fails when someone ADDS a request field.
# A new field that nobody classified would otherwise be forwarded-or-not by
# accident: silently dropped if the generate payload does not name it, silently
# honoured if it does. Either way the user is not told. So the four declarations
# in the adapter must, together, account for the schema exactly — and the way to
# fix a failure here is to decide which of the four the new field belongs to,
# not to widen the test.

_CLASSIFICATIONS = {
    "422 (REJECT_TABLE)": lambda: {f for f, _feat, _p in ltx25.REJECT_TABLE},
    "ignore+log (IGNORED_FIELDS)": lambda: set(ltx25.IGNORED_FIELDS),
    "honoured (HONOURED_FIELDS)": lambda: set(ltx25.HONOURED_FIELDS),
    "governed by another field (GOVERNED_FIELDS)": lambda: set(ltx25.GOVERNED_FIELDS),
}


def test_every_generate_request_field_is_classified():
    classified: set[str] = set()
    for produce in _CLASSIFICATIONS.values():
        classified |= produce()
    unclassified = set(GenerateRequest.model_fields) - classified
    assert not unclassified, (
        "GenerateRequest gained field(s) the LTX 2.5 adapter says nothing about: "
        f"{sorted(unclassified)}. Put each one in REJECT_TABLE, IGNORED_FIELDS, "
        "HONOURED_FIELDS or GOVERNED_FIELDS in services/engines/ltx25/adapter.py "
        "(and update Docs/ の対応表), then re-run."
    )


def test_the_four_classifications_do_not_overlap():
    """Exactly one home each. Two homes means the answer depends on which check
    runs first, which is how a field ends up both refused and forwarded."""
    seen: dict[str, str] = {}
    for label, produce in _CLASSIFICATIONS.items():
        for field in produce():
            assert field not in seen, f"{field} is in both {seen[field]} and {label}"
            seen[field] = label


def test_no_classification_names_a_field_the_schema_does_not_have():
    """The reverse direction: a REMOVED or renamed request field must not sit in
    the adapter's tables pretending to be guarded."""
    fields = set(GenerateRequest.model_fields)
    for label, produce in _CLASSIFICATIONS.items():
        assert produce() <= fields, f"{label} names unknown field(s): {sorted(produce() - fields)}"


# --------------------------------------------------------------------------- #
# 3e) the same audit for the CHAIN schema (§3-102)
# --------------------------------------------------------------------------- #

_CHAIN_CLASSIFICATIONS = {
    "422 (CHAIN_REJECT_TABLE)": lambda: {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE},
    "ignore+log (CHAIN_IGNORED_FIELDS)": lambda: set(ltx25.CHAIN_IGNORED_FIELDS),
    "honoured (CHAIN_HONOURED_FIELDS)": lambda: set(ltx25.CHAIN_HONOURED_FIELDS),
    "governed (CHAIN_GOVERNED_FIELDS)": lambda: set(ltx25.CHAIN_GOVERNED_FIELDS),
}


def test_every_generate_chain_request_field_is_classified():
    classified: set[str] = set()
    for produce in _CHAIN_CLASSIFICATIONS.values():
        classified |= produce()
    unclassified = set(GenerateChainRequest.model_fields) - classified
    assert not unclassified, (
        "GenerateChainRequest gained field(s) the LTX 2.5 adapter says nothing "
        f"about: {sorted(unclassified)}. Put each one in CHAIN_REJECT_TABLE, "
        "CHAIN_IGNORED_FIELDS, CHAIN_HONOURED_FIELDS or CHAIN_GOVERNED_FIELDS "
        "in services/engines/ltx25/adapter.py (and update Docs/ の対応表)."
    )


def test_the_four_chain_classifications_do_not_overlap():
    seen: dict[str, str] = {}
    for label, produce in _CHAIN_CLASSIFICATIONS.items():
        for field in produce():
            assert field not in seen, f"{field} is in both {seen[field]} and {label}"
            seen[field] = label


def test_no_chain_classification_names_a_field_the_schema_does_not_have():
    fields = set(GenerateChainRequest.model_fields)
    for label, produce in _CHAIN_CLASSIFICATIONS.items():
        assert produce() <= fields, f"{label} names unknown field(s): {sorted(produce() - fields)}"


def test_every_chain_governor_is_itself_refused():
    """THE LOOP IS EMPTY SINCE THE NAG/VSF INCREMENT, and that is the correct
    result rather than a hole: ``CHAIN_GOVERNED_FIELDS`` has no rows left, so
    there is no governor to check. The test is kept because what it asserts is
    a RULE about any future row, not a fact about today's three."""
    refused = {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE}
    for field, governor in ltx25.CHAIN_GOVERNED_FIELDS.items():
        assert governor in refused, f"{field} is governed by {governor}, which is not refused"


def test_chain_crop_output_is_honoured_not_refused_nor_dropped():
    """The single path's ruling, restated for the chain schema: crop_output is
    an ffmpeg post-process on the finished mp4, so it is engine-independent."""
    assert "crop_output" in ltx25.CHAIN_HONOURED_FIELDS
    assert "crop_output" not in {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE}
    assert "crop_output" not in ltx25.CHAIN_IGNORED_FIELDS


def test_the_nested_chain_clip_fields_are_all_accounted_for():
    """THE AUDIT ABOVE ONLY SEES THE TOP LEVEL. ``clips`` is classified as
    honoured, which is a promise about ``ChainClip``'s OWN fields too — so they
    get named here, and a new one fails this test rather than being silently
    dropped from the payload the worker receives."""
    assert set(ChainClip.model_fields) == {"prompt", "num_frames", "conditioning_images"}
    # ...and each of the three really is read when the payload is built.
    source = inspect.getsource(ltx25._RealBackend25.generate_chain)
    assert "clip_prompt" in source  # prompt (override else the global one)
    assert "num_frames" in source
    assert "conditioning_images" in source


#: ``field -> the text that proves it is read``. Three fields need an entry,
#: and each for the same reason: the request field is not what ``generate_chain``
#: touches. The global ``prompt`` is reached THROUGH the schema's own helper (a
#: clip's override else the global one), and the two SOURCE fields are upload
#: IDS the orchestrator has already resolved into material — so what the method
#: reads is the keyword argument carrying that material, not ``chain.source_*``.
_CHAIN_HONOURED_READS = {
    "prompt": "chain.clip_prompt(",
    "source_video": "source_tail_path",
    "source_audio": "source_audio_path",
    # Retake, for the SAME reason as the two source fields: the request field
    # carries an upload id and a window start time, and the orchestrator has
    # already cut those into the frame-exact window mp4 — so what
    # ``generate_chain`` reads is the keyword argument carrying it. (The glue
    # widths ARE read off ``chain.retake``, but the default needle would look
    # for ``chain.retake`` as a whole, and the material is the load-bearing
    # half.)
    "retake": "retake_window_path",
    # End source, for the SAME reason: the request field carries an upload id
    # (a video OR a still) and the band length, and the orchestrator has already
    # turned that into the cut mp4 — so what ``generate_chain`` reads is the
    # keyword argument carrying it. (``context_frames`` and ``strength`` are
    # read off their own keywords too, for the same reason.)
    "end_source": "end_source_path",
    # §3-102 third increment, same reason as the source fields: the adapter
    # NAMES and the reference upload id are resolved into material by the
    # orchestrator, so what generate_chain reads is the keyword argument.
    "loras": "lora_paths",
    "reference_video_id": "reference_video_path",
}


def test_chain_honoured_fields_are_exactly_what_generate_chain_acts_on():
    """Not a transcription of the payload builder: a field declared honoured but
    never read fails here (the chain twin of the single-path test above)."""
    source = inspect.getsource(ltx25._RealBackend25.generate_chain)
    for field in ltx25.CHAIN_HONOURED_FIELDS:
        needle = _CHAIN_HONOURED_READS.get(field, f"chain.{field}")
        assert needle in source, f"{field} is declared honoured but never read"


def test_every_governor_is_itself_refused():
    """A sub-parameter is only safe to leave unruled because its GOVERNOR is a
    422 — otherwise a request could make it meaningful and this engine would act
    on a value it never reads.

    THE LOOP IS EMPTY SINCE THE NAG/VSF INCREMENT: ``GOVERNED_FIELDS`` lost its
    last three rows (the NAG knobs) when their governor stopped being a 422, so
    there is nothing to iterate. An empty loop passing is the honest answer to
    "is every governor refused?" when nothing is governed — and the test stays
    because it states a rule about the NEXT row, not a fact about the last
    three. ``test_the_governed_table_is_empty_and_still_exists`` above is what
    pins the emptiness itself, so this passing vacuously cannot hide a table
    that was deleted rather than emptied."""
    refused = {f for f, _feat, _p in ltx25.REJECT_TABLE}
    for field, governor in ltx25.GOVERNED_FIELDS.items():
        assert governor in refused, f"{field} is governed by {governor}, which is not refused"


#: ``field -> the text that proves it is read``, the single-path twin of
#: :data:`_CHAIN_HONOURED_READS`. It did not exist until §3-102's third
#: increment, because until then every honoured field WAS a ``request.<name>``
#: read. The two entries below are the exception the LoRA work introduced: the
#: adapter NAMES and the reference upload id are resolved into material by the
#: orchestrator, so what ``generate`` touches is the keyword argument carrying
#: that material, not the request field.
#:
#: The two strength overrides need NO entry: they really are read off the
#: request (``request.reference_video_strength`` /
#: ``request.conditioning_attention_strength``), which is exactly what the
#: default needle asserts.
_HONOURED_READS = {
    "loras": "lora_paths",
    "reference_video_id": "reference_video_path",
}


def test_honoured_fields_are_exactly_what_generate_acts_on(ltx25_paths):
    """Not a transcription of the payload builder: the payload is BUILT here and
    compared, so a field quietly dropped from ``generate`` fails this test."""
    import inspect

    source = inspect.getsource(ltx25._RealBackend25.generate)
    for field in ltx25.HONOURED_FIELDS:
        needle = _HONOURED_READS.get(field, f"request.{field}")
        assert needle in source, f"{field} is declared honoured but never read"


# --------------------------------------------------------------------------- #
# 4) seams
# --------------------------------------------------------------------------- #


def test_the_mock_backend_class_is_shared_with_ltx23():
    """Design ruling: no _MockBackend25. A synthetic gradient clip says nothing
    about which engine would have rendered it, so a second copy would be a copy
    with no content — only the reported label differs."""
    assert ltx25.LTX25Runner._MOCK_BACKEND_CLS is ltx23._MockBackend
    assert ltx23.LTXRunner._MOCK_BACKEND_CLS is ltx23._MockBackend
    assert ltx25.LTX25Runner._MOCK_BACKEND_LABEL != ltx23.LTXRunner._MOCK_BACKEND_LABEL


def test_mock_backend_label_reaches_the_outcome(tmp_path):
    """The label is not decoration: it is how metadata.json says WHICH engine's
    mock ran, which is what makes a GPU-free 2.3<->2.5 round trip checkable."""
    cfg = AppConfig.model_validate({"model": {"backend": "mock"}})
    runner = ltx25.LTX25Runner(cfg, build_low_vram_settings(cfg), _descriptor_stub())
    backend = runner._select_backend()
    assert isinstance(backend, ltx23._MockBackend)
    assert backend.backend_label == ltx25.MOCK_BACKEND_25


def _descriptor_stub() -> BaseModelDescriptor:
    return BaseModelDescriptor(
        id="LTX25", display_name="LTX 2.5", engine_family="ltx25",
        categories={"transformer": CategoryDescriptor("transformer", ("x",), (".gguf",))},
    )


def test_worker_seams_point_at_engine25():
    cls = ltx25._RealBackend25
    assert cls._WORKER_MODULE == "engine25.worker"
    assert cls._ENGINE_DIR_VALUE == "./engine25"
    # A SEPARATE log file: after a 2.3<->2.5 swap both logs must survive for the
    # round-trip gate to be checkable at all.
    assert cls._LOG_NAME != ltx23._RealBackend._LOG_NAME
    # The protocol frame prefix is deliberately SHARED (one parent-side reader).
    assert cls._PREFIX == ltx23._RealBackend._PREFIX


def test_engine_python_comes_from_its_own_config_key():
    cfg = AppConfig.model_validate({})
    assert ltx25._RealBackend25._engine_python_value(cfg).endswith(
        ".venv-engine-ltx25/Scripts/python.exe"
    )
    assert ltx23._RealBackend._engine_python_value(cfg).endswith(
        ".venv-engine/Scripts/python.exe"
    )
    cfg2 = AppConfig.model_validate({"model": {"engine_python_ltx25": "./other/python.exe"}})
    assert ltx25._RealBackend25._engine_python_value(cfg2) == "./other/python.exe"
    # ...and overriding 2.5's key leaves 2.3's alone.
    assert ltx23._RealBackend._engine_python_value(cfg2).endswith(
        ".venv-engine/Scripts/python.exe"
    )


def test_child_env_carries_no_2_3_engine_knobs(ltx25_paths, tmp_path):
    """Every LTX_* variable the 2.3 worker reads names a code path inside
    engine/. Inheriting them into engine25 would be the borrowed-assumption bug
    the separate engine exists to avoid.

    Measured as "which LTX_* keys this method ADDS to the inherited
    environment", not "are there any" — the parent process's own environment
    passes through by design (that is what makes a manual override possible),
    and the test process itself sets one (LTX_DISABLE_GRADIO)."""
    import os

    cfg, _paths, descriptor = ltx25_paths
    inherited = {k for k in os.environ if k.startswith("LTX_")}
    env = _backend(cfg, descriptor)._build_child_env(tmp_path)
    assert {k for k in env if k.startswith("LTX_")} == inherited
    assert env["PYTHONPATH"] == str(tmp_path)
    # The adapter deliberately sets NO allocator config: expandable_segments is
    # refused on Windows and torch 2.9 deprecates the variable's name, so the
    # line that used to set it was removed as a measured no-op (2026-08-24,
    # outputs/b4-vram-diag/). Nothing in the project sets the variable any more
    # either — run.ps1, scripts/install_ltx.ps1 and the LTX 2.3 adapter all
    # dropped their copies on 2026-09-02 (Docs/PENDING_TASKS_CLOSED.md §3-112).
    # Still asserted as "does not ADD it", for the same reason the LTX_*
    # assertion above is: the parent's environment passes through by design, so
    # an operator who exports the variable by hand is not overridden.
    added = {k: v for k, v in env.items() if os.environ.get(k) != v}
    assert "PYTORCH_CUDA_ALLOC_CONF" not in added

    ltx23_env = ltx23._RealBackend(cfg, build_low_vram_settings(cfg), descriptor)._build_child_env(
        tmp_path
    )
    assert {"LTX_COMPONENT_FILES", "LTX_TE_OFFLOAD", "LTX_DIT_CPU_LOAD"} <= set(ltx23_env)


def test_sage_availability_delegates_to_the_base_probe_on_the_2_5_venv(tmp_path):
    """高速化第3弾 REMOVED this class's ``sage_available`` override; what stands
    here now is the property it inherits.

    The old test asserted ``is False`` unconditionally, which was right while
    the answer was hard-coded and would be WRONG now — the base property is a
    file-existence probe, so on the owner's machine (where 高速化第3弾 installed
    the wheel into ``.venv-engine-ltx25``) the honest answer is True. Asserting
    a constant would therefore fail on exactly the machine the feature works on.

    So this pins the SHAPE instead, and the shape is the thing that could
    silently regress: the probe must read the LTX 2.5 venv's site-packages —
    ``_REAL_BACKEND_CLS._engine_python_value`` resolves to
    ``model.engine_python_ltx25`` on this class — and it must require BOTH
    packages, because the sage kernels are Triton-backed. A probe still pointed
    at ``.venv-engine`` would answer for the 2.3 engine's install, which is the
    wrong venv's answer given confidently. This is also what GET /status reports
    BEFORE any worker exists (see PipelineManager.acceleration_status_block);
    once a worker is up, its own import probe wins."""
    venv25 = tmp_path / ".venv-engine-ltx25"
    site = venv25 / "Lib" / "site-packages"
    site.mkdir(parents=True)
    (venv25 / "Scripts").mkdir()
    # A DIFFERENT venv for 2.3, fully stocked: if the probe read this one the
    # assertions below would come out backwards, which is the point of setting
    # it up rather than leaving the 2.3 key at its default.
    venv23 = tmp_path / ".venv-engine"
    site23 = venv23 / "Lib" / "site-packages"
    site23.mkdir(parents=True)
    (venv23 / "Scripts").mkdir()
    (site23 / "sageattention").mkdir()
    (site23 / "triton").mkdir()

    def _runner() -> ltx25.LTX25Runner:
        cfg = AppConfig.model_validate({})
        cfg.model.engine_python = (venv23 / "Scripts" / "python.exe").as_posix()
        cfg.model.engine_python_ltx25 = (venv25 / "Scripts" / "python.exe").as_posix()
        return ltx25.LTX25Runner(cfg, build_low_vram_settings(cfg), _descriptor_stub())

    # The property really is inherited now — no per-class override to go stale.
    assert "sage_available" not in vars(ltx25.LTX25Runner)
    assert ltx25.LTX25Runner.sage_available is ltx23.LTXRunner.sage_available

    assert _runner().sage_available is False  # the 2.5 venv is empty...
    (site / "sageattention").mkdir()
    assert _runner().sage_available is False  # ...triton missing -> still False
    (site / "triton").mkdir()
    assert _runner().sage_available is True  # ...and now the wheel is there

    # ...and THAT is the value GET /status publishes before any worker exists.
    # PipelineManager._sage_available is the truth table's single home
    # (tests/test_smoke.py walks all four of its rows on the 2.3 runner); what
    # is asserted here is only that the LTX 2.5 runner plugs into the pre-load
    # row of it, because that row is the one the removed override used to
    # answer with a constant. The 2.5 runner is REAL here (not the mock, whose
    # row returns False whatever the venv holds) and has no live worker, so the
    # file probe is what answers.
    from services.pipeline_manager import PipelineManager

    runner = _runner()
    runner.config.model.backend = "real"
    assert runner.is_mock is False and runner.worker_sage_available is None
    assert PipelineManager._sage_available(types.SimpleNamespace(runner=runner)) is True
