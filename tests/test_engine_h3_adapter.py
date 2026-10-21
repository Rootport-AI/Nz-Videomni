"""MiniMax H3 engine adapter — scope, numeric rules, payloads, HTTP (第 2 弾 2b).

Named ``test_engine_h3_adapter`` (not ``test_h3_*``) on purpose: the
``test_h3_*`` files are the engine worker's own suites, run in the engine venv.
This one runs in the app venv, like every other adapter test.

Pinned here, none of it needing a GPU or a weight file:

1. the four-way audit of ``GenerateRequest`` against the adapter's tables
   (``conditioning_images`` is the one field that is both honoured and refused —
   key frames on any frame run, a strength does not);
2. the refusals and ``UNSUPPORTED_FEATURES`` (the reject table's names + ``chain`` + ``retake``
   + ``fused_gguf_dequant_kernel``);
3. the numeric rules and the ``constraints`` block GET /models publishes, for
   H3 and for LTX (the two JSON blocks of the stage-2 plan §3.3, verbatim);
4. the worker payloads: ``load`` (golden, no ``te_layers_on_gpu``) and
   ``generate`` (golden, ``images`` without strength, per-job acceleration,
   ``keep_resident`` only when asked) and the ``done`` -> outcome relay;
5. HTTP on three families (``three_family_client`` in conftest.py): GET /models,
   the switch to H3 (the H3 header precheck runs on every default file), mock
   T2V and I2V, the 422s, and the precheck refusing an LTX file.
"""

from __future__ import annotations

import inspect
import json
import threading
import types
from pathlib import Path

import pytest

from api.errors import APIError
from api.models import GenerateChainRequest, GenerateRequest
from config import AppConfig
from engine_h3 import constants as C
from services import engines
from services.base_models import BaseModelDescriptor, CategoryDescriptor
from services.engines.h3 import adapter as h3
from services.engines.ltx import request_rules as ltx_rules
from services.low_vram import build_low_vram_settings

PROMPT = "a lighthouse on a cliff at dusk, waves below"
_LORAS = [{"name": "style-a", "strength": 1.0}]
_IMG = {"image_id": "img-1", "frame_idx": 0}


def _request(**overrides) -> GenerateRequest:
    body = {"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 124}
    body.update(overrides)
    return GenerateRequest(**body)


# --------------------------------------------------------------------------- #
# 1) the audit
# --------------------------------------------------------------------------- #

_CLASSIFICATIONS = {
    "422 (REJECT_TABLE)": lambda: {f for f, _feat, _p in h3.REJECT_TABLE},
    "ignored (IGNORED_FIELDS)": lambda: set(h3.IGNORED_FIELDS),
    "honoured (HONOURED_FIELDS)": lambda: set(h3.HONOURED_FIELDS),
    "governed by another field (GOVERNED_FIELDS)": lambda: set(h3.GOVERNED_FIELDS),
}

#: The one field allowed in two tables: honoured (the key frames) AND refused
#: (a strength).
_ALLOWED_OVERLAP = {"conditioning_images"}


def test_the_four_tables_together_are_exactly_the_schema():
    classified: set[str] = set()
    for produce in _CLASSIFICATIONS.values():
        classified |= produce()
    fields = set(GenerateRequest.model_fields)
    assert not fields - classified, (
        "GenerateRequest gained field(s) the MiniMax H3 adapter says nothing about: "
        f"{sorted(fields - classified)}"
    )
    assert not classified - fields, f"unknown field(s): {sorted(classified - fields)}"


def test_the_four_tables_overlap_only_on_conditioning_images():
    seen: dict[str, str] = {}
    overlaps: set[str] = set()
    for label, produce in _CLASSIFICATIONS.items():
        for field in produce():
            if field in seen:
                overlaps.add(field)
            seen[field] = label
    assert overlaps == _ALLOWED_OVERLAP
    refused = {f for f, _feat, _p in h3.REJECT_TABLE}
    assert _ALLOWED_OVERLAP <= refused & set(h3.HONOURED_FIELDS)


def test_every_governor_is_itself_refused():
    refused = {f for f, _feat, _p in h3.REJECT_TABLE}
    for field, governor in h3.GOVERNED_FIELDS.items():
        assert governor in refused, f"{field} is governed by {governor}, which is not refused"


def test_honoured_fields_are_read_by_generate():
    source = inspect.getsource(h3._RealBackendH3.generate)
    for field in h3.HONOURED_FIELDS - {"embed_mp4_metadata"}:
        assert f"request.{field}" in source, f"{field} is declared honoured but never read"


def test_ignored_fields_are_really_not_read_by_generate():
    source = inspect.getsource(h3._RealBackendH3.generate)
    for field in h3.IGNORED_FIELDS:
        assert f"request.{field}" not in source


# --------------------------------------------------------------------------- #
# 2) the refusals
# --------------------------------------------------------------------------- #

#: One VALID request per reject-table row (keyed by FEATURE: one field has two).
REQUEST_OVERRIDES_H3: dict[str, dict] = {
    "loras": {"loras": _LORAS},
    "reference_video": {"reference_video_id": "vid-1", "loras": _LORAS},
    "nag": {"nag_enabled": True, "negative_prompt": "blurry"},
    "two_stage_hq": {"pipeline": "two_stage_hq"},
    "guidance_scale": {"guidance_scale": 3.0},
    "outpaint": {
        "outpaint": {"pad_left": 128},
        "reference_video_id": "vid-1",
        "loras": _LORAS,
        "width": 768,
    },
    "inpaint": {
        "inpaint": {"mask_video_id": "mask-1"},
        "reference_video_id": "vid-1",
        "loras": _LORAS,
        "width": 768,
        "height": 512,
    },
    "alpha_gen": {
        "alpha_gen": {"working_width": 512, "working_height": 384},
        "reference_video_id": "vid-1",
        "loras": _LORAS,
        "width": 512,
        "height": 384,
    },
    "prune_vaed": {"vae_mode": "prune_vaed"},
    "keep_resident_embeddings": {"keep_resident_embeddings": True},
    # 0.8 is LTX's default and still refused: H3 has no strength at all.
    "keyframe_strength": {"conditioning_images": [{**_IMG, "strength": 0.8}]},
}


def test_the_fixture_covers_every_refusal():
    assert set(REQUEST_OVERRIDES_H3) == {feat for _f, feat, _p in h3.REJECT_TABLE}


@pytest.mark.parametrize("feature", list(REQUEST_OVERRIDES_H3))
def test_every_refusal_raises_feature_unsupported_naming_h3(feature):
    request = _request(**REQUEST_OVERRIDES_H3[feature])
    # The row itself fires; the 422 names the FIRST row that fires (a reference
    # video, an outpaint ... carry their schema companion ``loras``, which is
    # refused before them).
    fired = [feat for _f, feat, pred in h3.REJECT_TABLE if pred(request)]
    assert feature in fired
    with pytest.raises(APIError) as ei:
        h3.reject_unsupported(request)
    err = ei.value
    assert err.code == "FEATURE_UNSUPPORTED" and err.status_code == 422
    assert f"'{fired[0]}'" in err.message
    assert err.detail.startswith("MiniMax H3 ")


def test_keyframe_strength_message_says_h3_has_none():
    with pytest.raises(APIError) as ei:
        h3.reject_unsupported(_request(conditioning_images=[{"image_id": "a", "frame_idx": 17, "strength": 0.5}]))
    assert ei.value.detail == "MiniMax H3 にはキーフレームの強度がありません（strength は指定しないでください）。"


def test_t2v_and_keyframe_i2v_are_accepted():
    h3.reject_unsupported(_request())
    h3.reject_unsupported(_request(conditioning_images=[_IMG]))
    # Several key frames, the last frame and a middle one: no refusal.
    h3.reject_unsupported(_request(conditioning_images=[
        _IMG, {"image_id": "img-2", "frame_idx": 60}, {"image_id": "img-3", "frame_idx": 123},
    ]))


def test_unsupported_features_is_the_reject_table_plus_chain_retake_and_fused_kernel():
    assert h3.UNSUPPORTED_FEATURES == (
        "loras",
        "reference_video",
        "nag",
        "two_stage_hq",
        "guidance_scale",
        "outpaint",
        "inpaint",
        "alpha_gen",
        "prune_vaed",
        "keep_resident_embeddings",
        "keyframe_strength",
        "chain",
        "retake",
        "fused_gguf_dequant_kernel",
    )
    assert engines.unsupported_features("h3") == h3.UNSUPPORTED_FEATURES


def test_fused_kernel_is_published_but_ignored_not_refused():
    # Published so the UIs hide the Settings row, yet never a 422: a saved OFF
    # must keep passing on H3 (台帳 §1-100).
    assert "fused_gguf_dequant_kernel" in h3.UNSUPPORTED_FEATURES
    # The name is both a request field and a feature name; neither column has it.
    assert "fused_gguf_dequant_kernel" not in {field for field, _feature, _pred in h3.REJECT_TABLE}
    assert "fused_gguf_dequant_kernel" not in {feature for _field, feature, _pred in h3.REJECT_TABLE}
    assert "fused_gguf_dequant_kernel" in h3.IGNORED_FIELDS
    h3.reject_unsupported(_request(fused_gguf_dequant_kernel=False))


def test_every_chain_is_refused():
    chain = GenerateChainRequest(prompt=PROMPT, clips=[{"num_frames": 49}, {"num_frames": 49}])
    with pytest.raises(APIError) as ei:
        engines.reject_chain("h3", chain)
    assert ei.value.code == "FEATURE_UNSUPPORTED"
    assert "chain" in ei.value.message


def test_registration():
    assert engines.FAMILY_BY_ID["h3"] == ("services.engines.h3.adapter", "H3Runner")
    assert engines.FAMILY_DISPLAY["h3"] == "MiniMax H3"
    assert "h3" not in engines.FAMILY_BY_KV.values()
    assert engines.runner_class_for("h3") is h3.H3Runner
    assert h3.H3Runner._MOCK_BACKEND_CLS is engines.runner_class_for("ltx")._MOCK_BACKEND_CLS


# --------------------------------------------------------------------------- #
# 3) numeric rules and constraints
# --------------------------------------------------------------------------- #

H3_CONSTRAINTS = {
    "num_frames": {"step": 17, "offset": 5, "min": 22, "max": 481},
    "dimension_multiple": 32,
    "frame_rate": {"min": 24, "max": 24},
    "sampling": {
        "num_inference_steps": {"min": 1, "max": 50, "step": 1, "default": 20},
        "shift_video": {"min": 1, "max": 20, "step": 0.1, "default": 12},
        "shift_audio": {"min": 1, "max": 20, "step": 0.1, "default": 3},
    },
    "keyframes": {"step": 1, "offset": 0},
}
LTX_CONSTRAINTS = {
    "num_frames": {"step": 8, "offset": 1, "min": 9, "max": 481},
    "dimension_multiple": 64,
    "frame_rate": {"min": 1, "max": 60},
    "sampling": None,
    "keyframes": {"step": 8, "offset": 1},
}


def test_constraints_blocks_are_the_plans_json():
    assert engines.request_constraints("h3") == H3_CONSTRAINTS
    assert engines.request_constraints("ltx") == LTX_CONSTRAINTS
    assert engines.request_constraints("ltx25") == LTX_CONSTRAINTS
    json.dumps(H3_CONSTRAINTS)  # serialisable as published


def test_schema_bounds_read_the_rules_constants():
    fields = GenerateRequest.model_fields
    meta = {type(m).__name__: m for m in fields["num_frames"].metadata}
    assert meta["Ge"].ge == ltx_rules.NUM_FRAMES_MIN and meta["Le"].le == ltx_rules.NUM_FRAMES_MAX
    meta = {type(m).__name__: m for m in fields["frame_rate"].metadata}
    assert meta["Ge"].ge == ltx_rules.FRAME_RATE_MIN and meta["Le"].le == ltx_rules.FRAME_RATE_MAX
    assert fields["num_inference_steps"].default is None
    assert fields["shift_video"].default is None and fields["shift_audio"].default is None


def test_h3_defaults_are_written_back():
    req = _request()
    engines.validate_for_family("h3", req)
    assert (req.num_inference_steps, req.shift_video, req.shift_audio) == (20, 12.0, 3.0)
    assert isinstance(req.num_inference_steps, int)
    assert req.vram16_mode is True  # 裁定 44: on by default on H3 (flag_defaults)


def test_default_steps_is_held_once_and_agrees_with_the_engine_constants():
    """台帳 §1-102 ③: one step default in the adapter, equal to the engine's; the shift defaults also match the engine's."""
    rules = h3.REQUEST_RULES
    assert rules.default_steps == rules.sampling["num_inference_steps"].default == h3.H3_DEFAULT_STEPS
    assert h3.H3_DEFAULT_STEPS == C.DEFAULT_STEPS
    assert rules.sampling["shift_video"].default == C.DEFAULT_SHIFT_VIDEO
    assert rules.sampling["shift_audio"].default == C.DEFAULT_SHIFT_AUDIO


def test_vram16_mode_default_is_none_and_an_explicit_false_is_kept():
    from api.models import VRAM16_MODE_DEFAULT_H3

    assert GenerateRequest.model_fields["vram16_mode"].default is None
    assert h3.REQUEST_RULES.flag_defaults == {"vram16_mode": VRAM16_MODE_DEFAULT_H3}
    assert "vram16_mode" not in json.dumps(engines.request_constraints("h3"))
    req = _request(vram16_mode=False)
    engines.validate_for_family("h3", req)
    assert req.vram16_mode is False


def test_ltx_defaults_are_written_back_and_shifts_stay_none():
    req = GenerateRequest(prompt=PROMPT)
    engines.validate_for_family("ltx", req)
    assert req.num_inference_steps == 8
    assert req.shift_video is None and req.shift_audio is None
    assert req.vram16_mode is None  # LTX declares no flag default


def test_explicit_values_are_kept():
    req = _request(num_inference_steps=30, shift_video=5.5, shift_audio=2.0)
    engines.validate_for_family("h3", req)
    assert (req.num_inference_steps, req.shift_video, req.shift_audio) == (30, 5.5, 2.0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"width": 650}, "width must be a multiple of 32"),
        ({"height": 400}, "height must be a multiple of 32"),
        ({"num_frames": 121}, "num_frames must be 17n+5"),
        ({"num_frames": 22 - 17}, None),  # 5 is below the schema's 9 -> pydantic
        ({"frame_rate": 30.0}, "frame_rate must be 24"),
        ({"num_inference_steps": 60}, "num_inference_steps must be between 1 and 50"),
        ({"shift_video": 25.0}, "shift_video must be between 1 and 20"),
        ({"shift_audio": 0.5}, "shift_audio must be between 1 and 20"),
    ],
)
def test_h3_violations(overrides, message):
    if message is None:
        with pytest.raises(ValueError):
            _request(**overrides)
        return
    with pytest.raises(APIError) as ei:
        engines.validate_for_family("h3", _request(**overrides))
    err = ei.value
    assert err.code == "VALIDATION_ERROR" and err.status_code == 422
    assert err.message == "Request validation failed"
    assert err.detail == [{"loc": ["body"], "msg": f"Value error, {message}", "type": "value_error"}]


def test_h3_frame_counts_on_the_grid_pass():
    for frames in (22, 124, 192, 243, 345, 481):
        engines.validate_for_family("h3", _request(num_frames=frames))


# --------------------------------------------------------------------------- #
# 4) worker payloads
# --------------------------------------------------------------------------- #


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _category(name: str, default_file: Path) -> CategoryDescriptor:
    return CategoryDescriptor(
        name=name, scan=(str(default_file.parent),), extensions=(".safetensors",),
        default_file=str(default_file),
    )


@pytest.fixture()
def h3_paths(tmp_path):
    m = tmp_path / "m"
    paths = {
        "tr": _touch(m / "Weights" / "dit.safetensors"),
        "te": _touch(m / "TextEncoder" / "te.safetensors"),
        "vv": _touch(m / "VAE" / "minimax_h3_video_vae.safetensors"),
        "av": _touch(m / "VAE" / "minimax_h3_audio_vae.safetensors"),
        "tok": m / "TextEncoder" / "qwen3vl_tokenizer",
    }
    paths["tok"].mkdir(parents=True)
    descriptor = BaseModelDescriptor(
        id="H3FIXTURE",
        display_name="MiniMax H3 fixture",
        engine_family="h3",
        categories={
            "transformer": _category("transformer", paths["tr"]),
            "text_encoder": _category("text_encoder", paths["te"]),
            "video_vae": _category("video_vae", paths["vv"]),
            "audio": _category("audio", paths["av"]),
        },
        assets={"tokenizer_dir": str(paths["tok"])},
    )
    cfg = AppConfig.model_validate({"model": {"backend": "mock"}})
    return cfg, paths, descriptor


GOLDEN_LOAD_KEYS_H3 = [
    "op",
    "transformer_path",
    "text_encoder_path",
    "video_vae_path",
    "audio_vae_path",
    "tokenizer_dir",
    "blocks_on_gpu",
    "attention_backend",
    "block_swap_prefetch",
    "deterministic",
]


def test_load_payload_golden(h3_paths):
    cfg, paths, descriptor = h3_paths
    backend = h3._RealBackendH3(cfg, build_low_vram_settings(cfg), descriptor)
    payload = backend._build_load_payload(None)
    assert list(payload) == GOLDEN_LOAD_KEYS_H3
    assert payload == {
        "op": "load",
        "transformer_path": str(paths["tr"]),
        "text_encoder_path": str(paths["te"]),
        "video_vae_path": str(paths["vv"]),
        "audio_vae_path": str(paths["av"]),
        "tokenizer_dir": str(paths["tok"]),
        "blocks_on_gpu": 8,
        "attention_backend": "sdpa",
        "block_swap_prefetch": True,
        "deterministic": True,
    }
    assert "te_layers_on_gpu" not in payload


def test_load_payload_selection_overrides_a_category(h3_paths, tmp_path):
    cfg, _paths, descriptor = h3_paths
    other = _touch(tmp_path / "other_dit.safetensors")
    backend = h3._RealBackendH3(cfg, build_low_vram_settings(cfg), descriptor)
    assert backend._build_load_payload({"transformer": str(other)})["transformer_path"] == str(other)


def test_backend_identity():
    assert h3._RealBackendH3._WORKER_MODULE == "engine_h3.worker"
    assert h3._RealBackendH3._ENGINE_DIR_VALUE == "./engine_h3"
    assert h3._RealBackendH3._ENGINE_PYTHON_LABEL == "engine_python_h3"
    cfg = AppConfig.model_validate({"model": {}})
    assert cfg.model.engine_python_h3 == "./.venv-engine-h3/Scripts/python.exe"
    assert h3.REQUIRED_ASSETS == ("tokenizer_dir",)
    assert h3.DEFAULT_BLOCKS_ON_GPU == 8


DONE_EVENT = {
    "event": "done",
    "seed_used": 42,
    "sampler": "euler",
    "num_inference_steps": 20,
    "shift_video": 12.0,
    "shift_audio": 3.0,
    "peak_vram_mb": 9000,
    "peak_vram_reserved_mb": 9500,
    "rss_peak_gib": 29.1,
    "commit_gib": 30.0,
    "commit_peak_gib": 40.2,
    "seconds": 300.5,
    "attention_used": "sdpa",
    "block_swap_prefetch_used": "on",
    "keep_resident_used": "on",
    "vram16_mode_used": "on",
    "images_used": 1,
    "h3": {"phases": {"encode": 1.0}, "quant_backends": ["cuda"]},
}


def _capturing_backend(captured: list[dict], event: dict | None = None) -> h3._RealBackendH3:
    be = h3._RealBackendH3.__new__(h3._RealBackendH3)
    be._proc = types.SimpleNamespace(poll=lambda: None)  # type: ignore[attr-defined]
    be._lock = threading.Lock()

    def _send(msg: dict) -> None:
        captured.append(msg)
        out = Path(msg["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 16)

    be._send = _send  # type: ignore[attr-defined]
    calls: list[dict] = []

    def _read(cb, *, chain, prefix, one_stage=False):
        calls.append({"chain": chain, "prefix": prefix, "one_stage": one_stage})
        return dict(event or DONE_EVENT)

    be._read_worker_events = _read  # type: ignore[attr-defined]
    be.read_calls = calls  # type: ignore[attr-defined]
    return be


GOLDEN_GENERATE_KEYS_H3 = [
    "op",
    "prompt",
    "seed",
    "width",
    "height",
    "num_frames",
    "frame_rate",
    "num_inference_steps",
    "shift_video",
    "shift_audio",
    "images",
    "attention_backend",
    "block_swap_prefetch",
    "vram16_mode",
    "output_path",
]


def _validated(**overrides) -> GenerateRequest:
    req = _request(**overrides)
    engines.validate_for_family("h3", req)
    return req


def test_generate_payload_t2v_golden(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    outcome = be.generate(_validated(), tmp_path / "job", seed=42)
    (payload,) = captured
    assert list(payload) == GOLDEN_GENERATE_KEYS_H3
    assert payload["num_inference_steps"] == 20
    assert (payload["shift_video"], payload["shift_audio"]) == (12.0, 3.0)
    assert payload["images"] == []
    assert payload["attention_backend"] == "sdpa"
    assert payload["block_swap_prefetch"] is True
    assert payload["vram16_mode"] is True  # the H3 default, sent every job
    assert "keep_resident" not in payload  # absent = the release request
    assert be.read_calls == [{"chain": False, "prefix": "generate", "one_stage": True}]
    assert outcome.backend == h3.REAL_BACKEND_H3
    assert outcome.keep_resident_used == "on"
    assert outcome.vram16_mode_used == "on"
    assert outcome.h3 == {
        "sampler": "euler",
        "num_inference_steps": 20,
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "images_used": 1,
        "rss_peak_gib": 29.1,
        "commit_gib": 30.0,
        "commit_peak_gib": 40.2,
        "seconds": 300.5,
        "phases": {"encode": 1.0},
        "quant_backends": ["cuda"],
    }


def test_generate_payload_i2v_keep_resident_and_per_job_acceleration(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    image = tmp_path / "first.png"
    image.write_bytes(b"png")
    req = _validated(
        conditioning_images=[_IMG],
        keep_resident=True,
        attention_backend="sage",
        block_swap_prefetch=False,
        vram16_mode=False,
    )
    be.generate(req, tmp_path / "job", conditioning_image_paths=[image], seed=1)
    (payload,) = captured
    assert payload["images"] == [{"path": str(image), "frame_idx": 0}]  # no strength
    assert payload["attention_backend"] == "sage"
    assert payload["block_swap_prefetch"] is False  # sent, not omitted
    assert payload["vram16_mode"] is False  # sent, not omitted
    assert list(payload)[-1] == "keep_resident" and payload["keep_resident"] is True


def test_generate_payload_sends_every_key_frame_with_its_position(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    paths = []
    for name in ("a.png", "b.png", "c.png"):
        (tmp_path / name).write_bytes(b"png")
        paths.append(tmp_path / name)
    req = _validated(conditioning_images=[
        _IMG, {"image_id": "img-2", "frame_idx": 123}, {"image_id": "img-3", "frame_idx": 60},
    ])
    be.generate(req, tmp_path / "job", conditioning_image_paths=paths, seed=1)
    (payload,) = captured
    # In request order (the engine sorts), each with its frame, no strength.
    assert payload["images"] == [
        {"path": str(paths[0]), "frame_idx": 0},
        {"path": str(paths[1]), "frame_idx": 123},
        {"path": str(paths[2]), "frame_idx": 60},
    ]


def test_generate_crops_on_the_app_side(tmp_path, monkeypatch):
    captured: list[dict] = []
    crops: list[tuple] = []

    def _crop(src, dst, w, h):
        crops.append((Path(src).name, w, h))
        Path(dst).write_bytes(b"\x00" * 8)

    monkeypatch.setattr(h3.video_io, "crop_mp4", _crop)
    be = _capturing_backend(captured)
    out = be.generate(_validated(crop_output={"width": 600, "height": 360}), tmp_path / "job", seed=1)
    assert Path(captured[0]["output_path"]).name == "_full.mp4"
    assert crops == [("_full.mp4", 600, 360)]
    assert out.output_path.name == "output.mp4"


def test_generate_refuses_as_the_backstop(tmp_path):
    be = _capturing_backend([])
    with pytest.raises(APIError):
        be.generate(_validated(nag_enabled=True, negative_prompt="x"), tmp_path / "job", seed=1)


# --------------------------------------------------------------------------- #
# 5) HTTP, three families
# --------------------------------------------------------------------------- #


def _activate(client, base_model: str) -> None:
    r = client.post("/api/v1/pipeline/load", json={"base_model": base_model})
    assert r.status_code == 200, r.text
    assert client.app_context.pipeline_manager.active_base_model == base_model


def _base(client, base_id: str) -> dict:
    listing = client.get("/api/v1/models").json()
    return next(b for b in listing["base_models"] if b["id"] == base_id)


def test_models_lists_h3_with_constraints(three_family_client):
    h3_entry = _base(three_family_client, "MiniMaxH3")
    assert h3_entry["engine_family"] == "h3"
    assert h3_entry["display_name"] == "MiniMax H3"
    assert h3_entry["installed"] is True
    assert h3_entry["constraints"] == H3_CONSTRAINTS
    assert h3_entry["unsupported_features"] == list(h3.UNSUPPORTED_FEATURES)
    assert h3_entry["category_order"] == ["transformer", "text_encoder", "video_vae", "audio"]
    assert _base(three_family_client, "LTX23")["constraints"] == LTX_CONSTRAINTS
    assert _base(three_family_client, "LTX25")["constraints"] == LTX_CONSTRAINTS
    # VRAM 16GB mode: open on H3, closed on both LTX.
    assert "vram16_mode" not in h3_entry["unsupported_features"]
    assert "vram16_mode" in _base(three_family_client, "LTX23")["unsupported_features"]
    assert "vram16_mode" in _base(three_family_client, "LTX25")["unsupported_features"]
    # Fused GGUF Dequantization Kernel: hidden on H3 only (台帳 §1-100).
    assert "fused_gguf_dequant_kernel" in h3_entry["unsupported_features"]
    assert "fused_gguf_dequant_kernel" not in _base(three_family_client, "LTX23")["unsupported_features"]


def test_switch_to_h3_runs_the_h3_precheck_and_weight_class_is_null(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    assert three_family_client.app_context.pipeline_manager.active_engine_family == "h3"
    assert _base(three_family_client, "MiniMaxH3")["transformer_weight_class"] is None
    limits = three_family_client.get("/api/v1/config").json()["limits"]
    assert "h3" not in (limits.get("comfort_budgets") or {})


def _metadata(client, job_id: str) -> dict:
    ctx = client.app_context
    return json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))


def test_mock_t2v_over_http(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = three_family_client.post(
        "/api/v1/generate",
        json={"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 124},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert three_family_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    meta = _metadata(three_family_client, job_id)
    assert meta["backend"] == h3.MOCK_BACKEND_H3
    assert meta["request"]["num_inference_steps"] == 20
    assert (meta["request"]["shift_video"], meta["request"]["shift_audio"]) == (12.0, 3.0)
    assert meta["request"]["vram16_mode"] is True  # the H3 default, written back
    assert meta["vram16_mode_used"] is None  # the mock reports no engine facts
    assert "h3" not in meta  # the mock reports no engine facts


def test_mock_i2v_over_http(three_family_client, png_bytes):
    _activate(three_family_client, "MiniMaxH3")
    up = three_family_client.post(
        "/api/v1/upload/image", files={"file": ("first.png", png_bytes, "image/png")}
    )
    image_id = up.json()["image_id"]
    r = three_family_client.post(
        "/api/v1/generate",
        json={
            "prompt": PROMPT,
            "width": 640,
            "height": 384,
            "num_frames": 22,
            "num_inference_steps": 4,
            "shift_video": 6.0,
            # The first frame and a position past the end (snapped to F-1 = 21).
            "conditioning_images": [
                {"image_id": image_id, "frame_idx": 0},
                {"image_id": image_id, "frame_idx": 30},
            ],
        },
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert three_family_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    meta = _metadata(three_family_client, job_id)
    assert meta["generation_mode"] == "i2v"
    assert meta["request"]["num_inference_steps"] == 4
    assert (meta["request"]["shift_video"], meta["request"]["shift_audio"]) == (6.0, 3.0)
    # The positions actually used, and no strength (H3 has none: null, not 1.0).
    assert [ci["frame_idx"] for ci in meta["request"]["conditioning_images"]] == [0, 21]
    assert [ci["strength"] for ci in meta["request"]["conditioning_images"]] == [None, None]


@pytest.mark.parametrize(
    "overrides, msg",
    [
        ({"num_frames": 49}, "Value error, num_frames must be 17n+5"),
        ({"frame_rate": 30.0}, "Value error, frame_rate must be 24"),
        ({"num_inference_steps": 60}, "Value error, num_inference_steps must be between 1 and 50"),
        ({"width": 650}, "Value error, width must be a multiple of 32"),
    ],
)
def test_h3_numeric_422_over_http(three_family_client, overrides, msg):
    _activate(three_family_client, "MiniMaxH3")
    body = {"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 124, **overrides}
    r = three_family_client.post("/api/v1/generate", json=body)
    assert r.status_code == 422, r.text
    assert r.json() == {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "detail": [{"loc": ["body"], "msg": msg, "type": "value_error"}],
        }
    }


@pytest.mark.parametrize("feature", ["nag", "keyframe_strength", "loras"])
def test_h3_refuses_over_http_before_any_lookup(three_family_client, feature):
    _activate(three_family_client, "MiniMaxH3")
    before = len(three_family_client.get("/api/v1/jobs").json())
    body = {"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 124}
    body.update(REQUEST_OVERRIDES_H3[feature])
    r = three_family_client.post("/api/v1/generate", json=body)
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert error["detail"].startswith("MiniMax H3 ")
    assert len(three_family_client.get("/api/v1/jobs").json()) == before


def test_h3_refuses_a_chain_over_http(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = three_family_client.post(
        "/api/v1/generate/chain",
        json={"prompt": PROMPT, "clips": [{"num_frames": 49}, {"num_frames": 49}]},
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "FEATURE_UNSUPPORTED"


def test_ltx_request_shapes_still_pass_on_ltx_after_h3(three_family_client):
    """A round trip H3 -> LTX 2.3: the LTX rules are back (8n+1), steps 8."""
    _activate(three_family_client, "MiniMaxH3")
    _activate(three_family_client, "LTX23")
    r = three_family_client.post(
        "/api/v1/generate", json={"prompt": PROMPT, "width": 672, "height": 384, "num_frames": 124}
    )
    assert r.status_code == 422
    assert r.json()["error"]["detail"][0]["msg"] == "Value error, width must be a multiple of 64"
    r = three_family_client.post(
        "/api/v1/generate", json={"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 124}
    )
    assert r.json()["error"]["detail"][0]["msg"] == "Value error, num_frames must be 8n+1"
    r = three_family_client.post("/api/v1/generate", json={"prompt": PROMPT, "num_frames": 49})
    assert r.status_code == 202, r.text
    meta = _metadata(three_family_client, r.json()["job_id"])
    assert meta["request"]["num_inference_steps"] == 8


def test_precheck_refuses_an_ltx_safetensors_as_the_h3_transformer(three_family_client):
    """A non-H3 safetensors selected as the H3 transformer is MODEL_INCOMPATIBLE
    (the H3FormatError is converted, not a 500)."""
    _activate(three_family_client, "MiniMaxH3")
    models_dir = Path(three_family_client.app_context.config.model.models_dir)
    stray = models_dir / "MiniMaxH3" / "Weights" / "not_h3.safetensors"
    stray.write_bytes(
        (lambda head: len(head).to_bytes(8, "little") + head)(
            json.dumps(
                {"transformer_blocks.0.attn.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
            ).encode()
        )
        + b"\x00\x00"
    )
    r = three_family_client.post(
        "/api/v1/pipeline/load",
        json={"base_model": "MiniMaxH3", "models": {"transformer": "not_h3"}},
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "MODEL_INCOMPATIBLE"
    assert "LTX の DiT です" in error["detail"]


def test_h3_block_reaches_metadata_json(three_family_client):
    """The ``outcome.h3`` -> ``metadata["h3"]`` wiring (pipeline_manager), end to
    end on the mock: the outcome is decorated after the mock runs, exactly as
    the LTX 2.5 twin of this test does (tests/test_ltx25_api_guard.py)."""
    _activate(three_family_client, "MiniMaxH3")
    pm = three_family_client.app_context.pipeline_manager
    orig = pm.runner.generate
    block = {"num_inference_steps": 20, "shift_video": 12.0, "phases": {"decode": 1.0}}

    def wrapped(*a, **k):
        outcome = orig(*a, **k)
        outcome.h3 = dict(block)
        return outcome

    pm.runner.generate = wrapped  # type: ignore[method-assign]
    r = three_family_client.post(
        "/api/v1/generate", json={"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 124}
    )
    assert r.status_code == 202, r.text
    meta = _metadata(three_family_client, r.json()["job_id"])
    assert meta["h3"] == block


def test_vram16_oom_guidance_reaches_the_job_error_as_gpu_oom(three_family_client):
    """裁定 47 (a): the engine's VRAM 16GB mode OOM is a RuntimeError whose text
    carries the guidance and "CUDA out of memory"; the app's ``_is_oom`` must
    classify it as ``GPU_OOM`` (503) and keep the guidance in the detail. The
    exception is raised from a wrapped runner, as in the test above."""
    from api.errors import gpu_oom
    from api.models import JobStatus
    from services.pipeline_manager import _is_oom

    guidance = (
        "VRAM 16GB mode でも GPU メモリが足りません (CUDA out of memory)。"
        "解像度かフレーム数を下げるか、VRAM 16GB mode を切ってください"
    )
    exc = RuntimeError(guidance)
    assert _is_oom(exc)
    err = gpu_oom(job_id="j", detail=str(exc))
    assert (err.code, err.status_code) == ("GPU_OOM", 503)
    assert "VRAM 16GB mode を切ってください" in err.detail

    _activate(three_family_client, "MiniMaxH3")
    pm = three_family_client.app_context.pipeline_manager

    def boom(*a, **k):
        raise RuntimeError(guidance)

    pm.runner.generate = boom  # type: ignore[method-assign]
    r = three_family_client.post(
        "/api/v1/generate", json={"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 124}
    )
    assert r.status_code == 202, r.text
    job = three_family_client.app_context.job_store.get(r.json()["job_id"])
    assert job.status == JobStatus.failed
    assert job.error.startswith("GPU_OOM:")
    assert "解像度かフレーム数を下げるか、VRAM 16GB mode を切ってください" in job.error
