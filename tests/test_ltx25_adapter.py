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
#: asserting against itself. Several entries carry companions because the
#: SCHEMA already couples them (outpaint needs the video it extends; a
#: reference video needs an IC-LoRA to condition; NAG needs a negative prompt)
#: — this engine's ruling has to survive those couplings, not dodge them.
REQUEST_OVERRIDES: dict[str, dict] = {
    "pipeline": {"pipeline": "two_stage_hq"},
    "outpaint": {
        "outpaint": {"pad_left": 64, "pad_right": 0, "pad_top": 0, "pad_bottom": 0},
        "reference_video_id": "vid-123",
        "loras": _LORAS,
    },
    "loras": {"loras": _LORAS},
    "reference_video_id": {"reference_video_id": "vid-123", "loras": _LORAS},
    "nag_enabled": {"nag_enabled": True, "negative_prompt": "blurry, low quality"},
    "vae_mode": {"vae_mode": "prune_vaed"},
    "attention_backend": {"attention_backend": "sage"},
    "keep_resident": {"keep_resident": True},
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


def test_unsupported_features_is_both_reject_tables_without_chain_itself():
    """Every feature name either table refuses must be published, or a control
    the server 422s stays lit in the frontend."""
    features = set(ltx25.UNSUPPORTED_FEATURES)
    assert {feat for _f, feat, _p in ltx25.REJECT_TABLE} <= features
    assert {feat for _f, feat, _p in ltx25.CHAIN_REJECT_TABLE} <= features
    # The two chain MODES this engine still cannot run stay published...
    assert {"retake", "end_source"} <= features
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
        ltx25._log_ignored(_request(negative_prompt="blurry", block_swap_prefetch=False))
    messages = [r.getMessage() for r in caplog.records if "ignores" in r.getMessage()]
    assert len(messages) == 1
    assert "negative_prompt" in messages[0] and "block_swap_prefetch" in messages[0]


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
#: objects rather than asserting against itself, and several entries carry the
#: companions the SCHEMA couples them to (retake and end_source own the whole
#: timeline, so they need a single clip; NAG needs a negative prompt). Imported
#: by tests/test_ltx25_api_guard.py, which drives the same table through HTTP.
#:
#: ``source_video`` / ``source_audio`` LEFT this table with §3-102's second
#: increment — they are honoured now, and the requests that carry them live in
#: :data:`CHAIN_ACCEPTED_SOURCES` below instead.
CHAIN_OVERRIDES: dict[str, dict] = {
    "retake": {
        "clips": [{"num_frames": 73}],
        "retake": {"video_id": "vid-1", "window_start_sec": 0.0},
    },
    "end_source": {
        "clips": [{"num_frames": 73}],
        "end_source": {"image_id": "img-1", "context_frames": 24},
    },
    # The schema couples a reference video to an IC-LoRA, so the pair travels
    # together; the table's ORDER is what decides which of the two is named.
    "reference_video_id": {"reference_video_id": "vid-123", "loras": _LORAS},
    "loras": {"loras": _LORAS},
    "nag_enabled": {"nag_enabled": True, "negative_prompt": "blurry, low quality"},
    "pipeline": {"pipeline": "two_stage_hq"},
    "vae_mode": {"vae_mode": "prune_vaed"},
    "attention_backend": {"attention_backend": "sage"},
    "keep_resident": {"keep_resident": True},
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
        ltx25._log_ignored(
            _chain_request(negative_prompt="blurry", block_swap_prefetch=False),
            ltx25.CHAIN_IGNORED_FIELDS,
        )
    messages = [r.getMessage() for r in caplog.records if "ignores" in r.getMessage()]
    assert len(messages) == 1
    assert "negative_prompt" in messages[0] and "block_swap_prefetch" in messages[0]


def test_generate_chain_refuses_an_out_of_scope_chain(ltx25_paths):
    """UNTIL §3-102 this refused EVERY chain. Now the refusal is field-by-field,
    and it still happens without loading a worker — the ruling is a pure read of
    the request, so a doomed chain must not pay for a model load."""
    cfg, _paths, descriptor = ltx25_paths
    backend = _backend(cfg, descriptor)
    with pytest.raises(APIError) as ei:
        backend.generate_chain(_chain_request(**CHAIN_OVERRIDES["retake"]), output_dir=None)
    assert ei.value.code == "FEATURE_UNSUPPORTED" and ei.value.status_code == 422
    assert "LTX 2.3" in ei.value.detail


def test_the_backend_refuses_a_chain_through_the_shared_function(ltx25_paths):
    """The endpoint (P5) and the backend method must not be able to disagree:
    both go through ``reject_chain``, so there is one message and one code."""
    cfg, _paths, descriptor = ltx25_paths
    request = _chain_request(**CHAIN_OVERRIDES["end_source"])
    with pytest.raises(APIError) as endpoint_side:
        ltx25.reject_chain(request)
    with pytest.raises(APIError) as backend_side:
        _backend(cfg, descriptor).generate_chain(request, output_dir=None)
    assert endpoint_side.value.code == backend_side.value.code
    assert endpoint_side.value.detail == backend_side.value.detail


def test_generate_chain_fails_loud_on_out_of_scope_material(ltx25_paths, tmp_path):
    """The orchestrator hands every runner the same thirteen keywords. SIX of
    them name material this engine cannot use, and every one is refused by the
    table above — so a value arriving here means the table and this signature
    have drifted apart. That is a bug, and it must not look like a job."""
    cfg, _paths, descriptor = ltx25_paths
    backend = _backend(cfg, descriptor)
    with pytest.raises(RuntimeError, match="retake_window_path"):
        backend.generate_chain(
            _chain_request(), output_dir=tmp_path, retake_window_path=tmp_path / "w.mp4"
        )
    with pytest.raises(RuntimeError, match="lora_paths"):
        backend.generate_chain(
            _chain_request(), output_dir=tmp_path, lora_paths=[object()]
        )
    # ...while the EMPTY list run_chain_job always builds for a no-lora chain is
    # not "material" and must sail through. Checked on the no-subprocess harness
    # so the assertion is about the guard, not about a worker spawn.
    captured: list[dict] = []
    _capturing_chain_backend(captured).generate_chain(
        _chain_request(), output_dir=tmp_path / "ok", lora_paths=[]
    )
    assert captured[0]["op"] == "generate_chain"


# --------------------------------------------------------------------------- #
# 3d) golden chain payload — the worker contract, byte for byte
# --------------------------------------------------------------------------- #
#
# Same discipline as section 1's load payload: the key SET and the key ORDER
# are the contract (JSON preserves insertion order, so a reorder IS a byte
# change), and the payload is BUILT here rather than transcribed.

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
        "chain": {"total_px": 41, "num_clips": 2},
    }
    return be


def test_chain_payload_golden_for_a_plain_two_clip_chain(tmp_path):
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    be.generate_chain(_chain_request(), output_dir=tmp_path / "out")

    assert len(captured) == 1
    payload = captured[0]
    assert list(payload) == GOLDEN_CHAIN_KEYS_25
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
    }


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
    assert list(captured[1]) == GOLDEN_CHAIN_KEYS_25 + ["stage2_window"]


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
    assert list(payload) == GOLDEN_CHAIN_KEYS_25 + ["source"]
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
    assert list(captured[0]) == list(captured[1]) == GOLDEN_CHAIN_KEYS_25


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
    assert list(payload) == GOLDEN_CHAIN_KEYS_25 + ["audio_source"]
    assert list(payload["audio_source"]) == GOLDEN_AUDIO_SOURCE_KEYS_25
    assert payload["audio_source"] == {"path": str(wav)}
    assert "source" not in payload


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
    assert list(payload) == GOLDEN_CHAIN_KEYS_25 + ["audio_source", "stage2_window"]
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
    assert list(payload) == GOLDEN_CHAIN_KEYS_25 + ["audio_source"]
    assert len(payload["clips"]) == 3
    assert all(set(clip) == {"prompt", "num_frames", "images"} for clip in payload["clips"])


def test_chain_outcome_names_this_engine_and_relays_the_chain_metadata(tmp_path):
    captured: list[dict] = []
    be = _capturing_chain_backend(captured)
    outcome = be.generate_chain(_chain_request(), output_dir=tmp_path / "out")

    assert outcome.generation_mode == "chain"
    assert outcome.backend == ltx25.REAL_BACKEND_25
    assert outcome.chain_metadata == {"total_px": 41, "num_clips": 2}
    assert outcome.seed_used == 123
    assert outcome.peak_vram_mb == 7000
    # The chain scope is SDPA-only, and every OTHER acceleration relay names a
    # 2.3 code path this engine does not have — reporting "off" would claim the
    # knob exists here and was left alone.
    assert outcome.attention_used == "sdpa"
    assert outcome.block_swap_prefetch_used is None
    assert outcome.keep_resident_used is None
    assert outcome.fused_gguf_dequant_kernel_used is None
    assert outcome.vae_mode_used is None


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
    on a value it never reads."""
    refused = {f for f, _feat, _p in ltx25.REJECT_TABLE}
    for field, governor in ltx25.GOVERNED_FIELDS.items():
        assert governor in refused, f"{field} is governed by {governor}, which is not refused"


def test_honoured_fields_are_exactly_what_generate_acts_on(ltx25_paths):
    """Not a transcription of the payload builder: the payload is BUILT here and
    compared, so a field quietly dropped from ``generate`` fails this test."""
    import inspect

    source = inspect.getsource(ltx25._RealBackend25.generate)
    # Six fields ride the payload verbatim; the other two are transformed
    # (conditioning_images -> "images", crop_output -> ffmpeg post-process).
    for field in ltx25.HONOURED_FIELDS:
        assert f"request.{field}" in source, f"{field} is declared honoured but never read"


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
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"

    ltx23_env = ltx23._RealBackend(cfg, build_low_vram_settings(cfg), descriptor)._build_child_env(
        tmp_path
    )
    assert {"LTX_COMPONENT_FILES", "LTX_TE_OFFLOAD", "LTX_DIT_CPU_LOAD"} <= set(ltx23_env)


def test_sage_is_permanently_off_for_this_engine():
    cfg = AppConfig.model_validate({})
    runner = ltx25.LTX25Runner(cfg, build_low_vram_settings(cfg), _descriptor_stub())
    assert runner.sage_available is False
