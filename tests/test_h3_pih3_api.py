"""MiniMax H3 pi-H3 (``pi_h3``) on the app side: the request field, the step /
shift presets, the acceptance checks, the worker payload, the mock job and the
record, the LTX refusals and the installer descriptor.

Pinned here, none of it needing a GPU or a real pi-H3 file:

1. ``pi_h3`` off (omitted or ``"off"``) changes nothing: the payload is the
   golden one, the steps / shifts are the plain defaults, no file is looked at,
   and metadata.json has no ``pi_h3_used`` key;
2. with a choice, an omitted step count / shift becomes the choice's value
   (``constraints.pi_h3``, from ``h3_pih3_format``) and an explicit
   other value is a 422 that states the fact;
3. the acceptance check: a missing file is ``MODEL_FILE_MISSING``, a file that
   ``h3_pih3_format.model_problems`` finds fault with is ``MODEL_INCOMPATIBLE``,
   compared against the selected transformer;
4. the payload carries ``pi_h3 = {steps, path}`` (path not in the load
   payload), the ``done`` echo becomes ``outcome.pi_h3_used``;
5. the mock job runs to the end with ``pi_h3_used`` and ``h3.pi_h3``;
6. LTX 2.3 / 2.5 refuse it with the H3-only sentence and publish it;
7. the shipped descriptor: the two assets are downloads' files, under a map, and
   the video / audio VAE scans do not list them;
8. (段 2) the chain — Chained, V2V, Retake — by the same rules: the field, its
   copy into the stored per-clip request, the steps / shifts (an unsent chain
   step count is the choice's, not LTX's schema 8), the acceptance check, the
   payloads (off = today's goldens), the mock run and its record, the LTX
   chains' refusal; and the single path's Inpainting / Outpainting with pi-H3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import h3_pih3_format
import main
from api.errors import APIError
from api.models import GenerateRequest
from config import AppConfig
from conftest import (
    _make_args,
    base_model_descriptor,
    build_model_layout,
    h3_base_model_descriptor,
    write_gguf_with_kv,
    write_h3_model_files,
)
from services import engines
from services.base_models import load_base_models
from services.engines.h3 import adapter as h3
from services.engines.ltx import adapter as ltx23
from services.engines.ltx25 import adapter as ltx25
from test_engine_h3_adapter import (
    DONE_EVENT,
    GOLDEN_GENERATE_KEYS_H3,
    GOLDEN_LOAD_KEYS_H3,
    PROMPT,
    _activate,
    _capturing_backend,
    _metadata,
    _request,
    _validated,
)
from test_h3_format import _write

H3_ONLY_DETAIL = "pi-H3 は MiniMax H3 専用です。LTX では使えません。"
BODY = {"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 22}
PI_DIR = "MiniMaxH3/VAE/pi-H3"
PI_FILES = {
    "4steps": f"{PI_DIR}/pi_h3_4step_dwd_adapter_int8_convrot.safetensors",
    "2steps": f"{PI_DIR}/pi_h3_2step_dwd_adapter_int8_convrot.safetensors",
}


# --------------------------------------------------------------------------- #
# the field and the tables
# --------------------------------------------------------------------------- #


def test_the_field_defaults_to_off_and_takes_the_two_choices():
    assert GenerateRequest.model_fields["pi_h3"].default == "off"
    for choice in ("off", "4steps", "2steps"):
        assert _request(pi_h3=choice).pi_h3 == choice
    with pytest.raises(ValueError):
        _request(pi_h3="3steps")


def test_the_choices_agree_with_the_engine_module():
    assert set(h3.PI_H3_ASSETS) == set(h3_pih3_format.PI_H3_CHOICES)
    literal = set(GenerateRequest.model_fields["pi_h3"].annotation.__args__)
    assert literal == {"off"} | set(h3_pih3_format.PI_H3_CHOICES)


def test_h3_honours_it_and_does_not_publish_it():
    assert "pi_h3" in h3.HONOURED_FIELDS
    assert "pi_h3" not in h3.UNSUPPORTED_FEATURES
    h3.reject_unsupported(_request(pi_h3="4steps"))  # no raise


def test_constraints_publish_each_choice_from_the_engine_module():
    published = engines.request_constraints("h3")["pi_h3"]
    assert published == {
        choice: {
            "num_inference_steps": steps,
            "shift_video": h3_pih3_format.PI_H3_SHIFT_VIDEO,
            "shift_audio": h3_pih3_format.PI_H3_SHIFT_AUDIO,
        }
        for choice, steps in h3_pih3_format.PI_H3_CHOICES.items()
    }
    assert published["4steps"]["num_inference_steps"] == 4
    assert published["2steps"]["num_inference_steps"] == 2
    assert "pi_h3" not in engines.request_constraints("ltx")
    assert "pi_h3" not in engines.request_constraints("ltx25")


# --------------------------------------------------------------------------- #
# steps and shifts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("overrides", [{}, {"pi_h3": "off"}], ids=["omitted", "off"])
def test_off_keeps_the_plain_defaults(overrides):
    req = _validated(**overrides)
    assert (req.num_inference_steps, req.shift_video, req.shift_audio) == (20, 12.0, 3.0)


@pytest.mark.parametrize("choice, steps", [("4steps", 4), ("2steps", 2)])
def test_a_choice_fills_the_omitted_values(choice, steps):
    req = _validated(pi_h3=choice)
    assert (req.num_inference_steps, req.shift_video, req.shift_audio) == (steps, 12.0, 3.0)
    assert isinstance(req.num_inference_steps, int)
    assert isinstance(req.shift_video, float) and isinstance(req.shift_audio, float)


def test_explicit_values_equal_to_the_choice_pass():
    req = _validated(pi_h3="2steps", num_inference_steps=2, shift_video=12, shift_audio=3.0)
    assert (req.num_inference_steps, req.shift_video, req.shift_audio) == (2, 12.0, 3.0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"num_inference_steps": 20},
            'num_inference_steps=20 differs from the value pi_h3="4steps" uses (num_inference_steps=4)',
        ),
        (
            {"shift_video": 5.5},
            'shift_video=5.5 differs from the value pi_h3="4steps" uses (shift_video=12)',
        ),
        (
            {"shift_audio": 2.0},
            'shift_audio=2 differs from the value pi_h3="4steps" uses (shift_audio=3)',
        ),
    ],
)
def test_an_explicit_other_value_is_422_stating_the_fact(overrides, message):
    with pytest.raises(APIError) as ei:
        _validated(pi_h3="4steps", **overrides)
    err = ei.value
    assert err.code == "VALIDATION_ERROR" and err.status_code == 422
    assert err.detail == [{"loc": ["body"], "msg": f"Value error, {message}", "type": "value_error"}]


def test_the_choice_is_written_before_the_defaults():
    # A 2-step job that sends only shift_video: the steps come from the choice,
    # not from the family default 20.
    req = _validated(pi_h3="2steps", shift_video=12.0)
    assert req.num_inference_steps == 2


# --------------------------------------------------------------------------- #
# the acceptance check (unit)
# --------------------------------------------------------------------------- #


def _never(*_args, **_kwargs):
    raise AssertionError("must not be called")


@pytest.mark.parametrize("overrides", [{}, {"pi_h3": "off"}], ids=["omitted", "off"])
def test_preflight_looks_at_nothing_when_off(overrides):
    h3.preflight_request(_request(**overrides), assets=_never, transformer=_never)


def test_preflight_missing_file_is_model_file_missing(tmp_path):
    missing = tmp_path / "pi-H3" / "pi_h3_4step_dwd_adapter_int8_convrot.safetensors"
    with pytest.raises(APIError) as ei:
        h3.preflight_request(_request(pi_h3="4steps"), assets=lambda key: missing, transformer=_never)
    err = ei.value
    assert err.code == "MODEL_FILE_MISSING" and err.status_code == 422
    assert err.detail == "pi-H3 のファイルがありません（pi-H3/pi_h3_4step_dwd_adapter_int8_convrot.safetensors）。"
    assert str(tmp_path) not in err.detail  # no local folder in the sentence


def test_preflight_undeclared_asset_is_model_file_missing():
    def assets(key):
        raise KeyError(key)

    with pytest.raises(APIError) as ei:
        h3.preflight_request(_request(pi_h3="2steps"), assets=assets, transformer=_never)
    assert ei.value.code == "MODEL_FILE_MISSING"
    assert ei.value.detail == "このベースモデルの記述子に pi-H3 のファイル（assets.pi_h3_2steps_path）がありません。"


def test_preflight_asks_each_choice_for_its_own_asset(tmp_path, monkeypatch):
    asked: list[str] = []
    files = {}
    for choice, key in h3.PI_H3_ASSETS.items():
        files[key] = tmp_path / f"{choice}.safetensors"
        _write(files[key], {"x": ("F32", (1,))})
    monkeypatch.setattr(h3.h3_pih3_format, "model_problems", lambda pi, dit: [])
    for choice in h3.PI_H3_ASSETS:
        h3.preflight_request(
            _request(pi_h3=choice),
            assets=lambda key: (asked.append(key), files[key])[1],
            transformer=lambda: tmp_path / "dit.safetensors",
        )
    assert asked == ["pi_h3_4steps_path", "pi_h3_2steps_path"]


def test_preflight_compares_against_the_selected_transformer(tmp_path, monkeypatch):
    pi = tmp_path / "pi.safetensors"
    _write(pi, {"x": ("F32", (1,))})
    dit = tmp_path / "selected_dit.safetensors"
    calls: list[tuple] = []

    def problems(pi_path, dit_path):
        calls.append((Path(pi_path), Path(dit_path)))
        return []

    monkeypatch.setattr(h3.h3_pih3_format, "model_problems", problems)
    h3.preflight_request(_request(pi_h3="4steps"), assets=lambda key: pi, transformer=lambda: dit)
    assert calls == [(pi, dit)]


def test_preflight_mismatch_is_model_incompatible_with_the_findings(tmp_path, monkeypatch):
    pi = tmp_path / "pi_h3_4step.safetensors"
    _write(pi, {"x": ("F32", (1,))})
    findings = [f"finding {i}" for i in range(7)]
    monkeypatch.setattr(h3.h3_pih3_format, "model_problems", lambda pi_path, dit_path: findings)
    with pytest.raises(APIError) as ei:
        h3.preflight_request(
            _request(pi_h3="4steps"), assets=lambda key: pi, transformer=lambda: tmp_path / "dit.safetensors"
        )
    err = ei.value
    assert err.code == "MODEL_INCOMPATIBLE" and err.status_code == 422
    assert err.detail == (
        "pi-H3 のファイル（pi_h3_4step.safetensors）と本体の重み（dit.safetensors）を照合しました: "
        "finding 0；finding 1；finding 2；finding 3；finding 4（ほか 2 件）"
    )


def test_preflight_runs_the_real_module_on_a_wrong_file(tmp_path):
    """No stub: a safetensors file that is not a pi-H3 layout is refused by
    ``h3_pih3_format.model_problems``' own findings."""
    pi = tmp_path / "pi_h3_4step.safetensors"
    _write(pi, {"blocks.0.attn.out_proj.lora_A.weight": ("F32", (4, 8))})
    with pytest.raises(APIError) as ei:
        h3.preflight_request(
            _request(pi_h3="4steps"), assets=lambda key: pi, transformer=lambda: tmp_path / "dit.safetensors"
        )
    assert ei.value.code == "MODEL_INCOMPATIBLE"
    assert ei.value.detail.startswith("pi-H3 のファイル（pi_h3_4step.safetensors）と本体の重み（dit.safetensors）を照合しました: ")


def test_preflight_without_a_resolvable_transformer_checks_the_file_only(tmp_path, monkeypatch):
    pi = tmp_path / "pi.safetensors"
    _write(pi, {"x": ("F32", (1,))})
    monkeypatch.setattr(h3.h3_pih3_format, "model_problems", _never)
    h3.preflight_request(_request(pi_h3="4steps"), assets=lambda key: pi, transformer=lambda: None)


def test_the_dispatcher_passes_ltx_through():
    engines.preflight_request("ltx", _request(), assets=_never, transformer=_never)
    engines.preflight_request("ltx25", _request(), assets=_never, transformer=_never)


# --------------------------------------------------------------------------- #
# the worker payloads
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("overrides", [{}, {"pi_h3": "off"}], ids=["omitted", "off"])
def test_off_payload_is_todays_golden(tmp_path, overrides):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    outcome = be.generate(_validated(**overrides), tmp_path / "job", seed=42)
    (payload,) = captured
    assert list(payload) == GOLDEN_GENERATE_KEYS_H3
    assert "pi_h3" not in payload
    assert outcome.pi_h3_used is None


@pytest.mark.parametrize("choice, steps", [("4steps", 4), ("2steps", 2)])
def test_pi_h3_payload_carries_steps_and_path(tmp_path, choice, steps):
    files = {key: str(tmp_path / f"{key}.safetensors") for key in h3.PI_H3_ASSETS.values()}
    captured: list[dict] = []
    be = _capturing_backend(captured, {**DONE_EVENT, "num_inference_steps": steps, "pi_h3_used": choice})
    be._require_asset = lambda key: files[key]  # type: ignore[attr-defined]
    outcome = be.generate(_validated(pi_h3=choice, keep_resident=True), tmp_path / "job", seed=42)
    (payload,) = captured
    assert payload["pi_h3"] == {"steps": steps, "path": files[h3.PI_H3_ASSETS[choice]]}
    assert (payload["num_inference_steps"], payload["shift_video"], payload["shift_audio"]) == (steps, 12.0, 3.0)
    keys = list(payload)
    assert keys[-3:] == ["pi_h3", "output_path", "keep_resident"]
    assert [k for k in keys if k not in ("pi_h3", "keep_resident")] == GOLDEN_GENERATE_KEYS_H3
    assert outcome.pi_h3_used == choice


def test_the_load_payload_has_no_pi_h3_path(tmp_path):
    from test_engine_h3_adapter import _touch
    from services.base_models import BaseModelDescriptor
    from services.low_vram import build_low_vram_settings
    from test_engine_h3_adapter import _category

    m = tmp_path / "m"
    files = {c: _touch(m / f"{c}.safetensors") for c in ("tr", "te", "vv", "av")}
    tok = m / "tok"
    tok.mkdir()
    descriptor = BaseModelDescriptor(
        id="H3FIXTURE", display_name="fixture", engine_family="h3",
        categories={
            "transformer": _category("transformer", files["tr"]),
            "text_encoder": _category("text_encoder", files["te"]),
            "video_vae": _category("video_vae", files["vv"]),
            "audio": _category("audio", files["av"]),
        },
        assets={"tokenizer_dir": str(tok), "pi_h3_4steps_path": str(m / "pi4.safetensors")},
    )
    cfg = AppConfig.model_validate({"model": {"backend": "mock"}})
    payload = h3._RealBackendH3(cfg, build_low_vram_settings(cfg), descriptor)._build_load_payload(None)
    assert list(payload) == GOLDEN_LOAD_KEYS_H3
    assert "pi4" not in json.dumps(payload)


# --------------------------------------------------------------------------- #
# HTTP over the mock backend
# --------------------------------------------------------------------------- #


@pytest.fixture()
def pih3_client(tmp_path, monkeypatch):
    """``three_family_client``'s LTX 2.3 + MiniMax H3, with the H3 descriptor
    declaring the two pi-H3 assets and both files present (tiny safetensors;
    the transformer match is stubbed per test where it must pass)."""
    ltx = base_model_descriptor()
    h3d = h3_base_model_descriptor()
    h3d["assets"] = {
        **h3d["assets"],
        "pi_h3_4steps_path": PI_FILES["4steps"],
        "pi_h3_2steps_path": PI_FILES["2steps"],
    }
    fragment = build_model_layout(tmp_path, [ltx, h3d])
    models_dir = tmp_path / "models"
    write_gguf_with_kv(
        models_dir / ltx["categories"]["transformer"]["default_file"],
        **{"general.architecture": "ltxv", "model_version": "2.3.0"},
    )
    write_h3_model_files(models_dir, h3d, monkeypatch)
    for rel in PI_FILES.values():
        (models_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        _write(models_dir / rel, {"x": ("F32", (1,))})
    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {"backend": "mock", **fragment},
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
        "state_file": (tmp_path / "state.json").as_posix(),
        "tracking": {"backend": "mock"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    app = main.build_app(_make_args(cfg_path.as_posix()))
    with TestClient(app) as client:
        client.app_context = app.state.context  # type: ignore[attr-defined]
        client.models_dir = models_dir  # type: ignore[attr-defined]
        yield client


def _job_count(client) -> int:
    return len(client.get("/api/v1/jobs").json())


@pytest.mark.parametrize("choice, steps", [("4steps", 4), ("2steps", 2)])
def test_mock_pi_h3_job_runs_and_records(pih3_client, monkeypatch, choice, steps):
    seen: list[tuple] = []
    monkeypatch.setattr(
        h3.h3_pih3_format, "model_problems", lambda pi, dit: (seen.append((Path(pi), Path(dit))), [])[1]
    )
    _activate(pih3_client, "MiniMaxH3")
    r = pih3_client.post("/api/v1/generate", json={**BODY, "pi_h3": choice})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert pih3_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    meta = _metadata(pih3_client, job_id)
    assert meta["pi_h3_used"] == choice
    assert meta["h3"] == {"pi_h3": {"choice": choice, "steps": steps, "mock": True}}
    assert meta["request"]["pi_h3"] == choice
    assert (
        meta["request"]["num_inference_steps"], meta["request"]["shift_video"], meta["request"]["shift_audio"]
    ) == (steps, 12.0, 3.0)
    # compared once, against the selected (default) transformer
    ((pi, dit),) = seen
    assert pi.name == Path(PI_FILES[choice]).name and pi.parent.name == "pi-H3"
    assert dit.name == "minimax_h3_dit.safetensors"


@pytest.mark.parametrize("overrides", [{}, {"pi_h3": "off"}], ids=["omitted", "off"])
def test_mock_off_job_record_is_unchanged(pih3_client, monkeypatch, overrides):
    monkeypatch.setattr(h3.h3_pih3_format, "model_problems", _never)
    _activate(pih3_client, "MiniMaxH3")
    r = pih3_client.post("/api/v1/generate", json={**BODY, **overrides})
    assert r.status_code == 202, r.text
    meta = _metadata(pih3_client, r.json()["job_id"])
    assert "pi_h3_used" not in meta
    assert "h3" not in meta
    assert meta["request"]["num_inference_steps"] == 20


def test_http_missing_file_is_422_and_no_job(pih3_client):
    _activate(pih3_client, "MiniMaxH3")
    (pih3_client.models_dir / PI_FILES["2steps"]).unlink()
    before = _job_count(pih3_client)
    r = pih3_client.post("/api/v1/generate", json={**BODY, "pi_h3": "2steps"})
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "MODEL_FILE_MISSING"
    assert error["detail"] == "pi-H3 のファイルがありません（pi-H3/pi_h3_2step_dwd_adapter_int8_convrot.safetensors）。"
    assert _job_count(pih3_client) == before


def test_http_header_mismatch_is_422_and_no_job(pih3_client):
    """The real ``model_problems`` on the tiny stand-in file: its findings."""
    _activate(pih3_client, "MiniMaxH3")
    before = _job_count(pih3_client)
    r = pih3_client.post("/api/v1/generate", json={**BODY, "pi_h3": "4steps"})
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "MODEL_INCOMPATIBLE"
    assert error["detail"].startswith(
        "pi-H3 のファイル（pi_h3_4step_dwd_adapter_int8_convrot.safetensors）と"
        "本体の重み（minimax_h3_dit.safetensors）を照合しました: "
    )
    assert _job_count(pih3_client) == before


def test_http_explicit_other_steps_is_422(pih3_client, monkeypatch):
    monkeypatch.setattr(h3.h3_pih3_format, "model_problems", _never)
    _activate(pih3_client, "MiniMaxH3")
    r = pih3_client.post("/api/v1/generate", json={**BODY, "pi_h3": "4steps", "num_inference_steps": 8})
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["detail"][0]["msg"] == (
        'Value error, num_inference_steps=8 differs from the value pi_h3="4steps" uses (num_inference_steps=4)'
    )


def test_models_publishes_the_presets_and_not_the_feature(pih3_client):
    listing = pih3_client.get("/api/v1/models").json()
    entry = next(b for b in listing["base_models"] if b["id"] == "MiniMaxH3")
    assert entry["constraints"]["pi_h3"]["4steps"]["num_inference_steps"] == 4
    assert "pi_h3" not in entry["unsupported_features"]


def test_the_vae_scans_do_not_list_the_pi_h3_files(pih3_client):
    registry = pih3_client.app_context.model_registry
    for category in ("video_vae", "audio"):
        files = [Path(e.path).name for e in registry.entries(category, base_model="MiniMaxH3")]
        assert not [f for f in files if f.startswith("pi_h3_")], (category, files)


# --------------------------------------------------------------------------- #
# LTX refuses it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("adapter", [ltx23, ltx25], ids=["ltx23", "ltx25"])
@pytest.mark.parametrize("choice", ["4steps", "2steps"])
def test_ltx_refuses_with_the_h3_only_sentence(adapter, choice):
    with pytest.raises(APIError) as ei:
        adapter.reject_unsupported(GenerateRequest(prompt=PROMPT, pi_h3=choice))
    assert ei.value.code == "FEATURE_UNSUPPORTED" and ei.value.status_code == 422
    assert ei.value.detail == H3_ONLY_DETAIL
    adapter.reject_unsupported(GenerateRequest(prompt=PROMPT, pi_h3="off"))  # no raise
    assert "pi_h3" in adapter.UNSUPPORTED_FEATURES


@pytest.mark.parametrize("base_model", ["LTX23", "LTX25"])
def test_ltx_refuses_over_http(three_family_client, base_model):
    _activate(three_family_client, base_model)
    before = _job_count(three_family_client)
    r = three_family_client.post("/api/v1/generate", json={"prompt": PROMPT, "num_frames": 25, "pi_h3": "4steps"})
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert error["detail"] == H3_ONLY_DETAIL
    assert _job_count(three_family_client) == before


# --------------------------------------------------------------------------- #
# the shipped descriptor
# --------------------------------------------------------------------------- #


def _shipped_manifest() -> dict:
    path = AppConfig().manifest_dir / "40-minimaxh3.json"
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_shipped_descriptor_declares_the_two_files_as_assets():
    descriptor = load_base_models(AppConfig().manifest_dir)["MiniMaxH3"]
    assert {key: descriptor.assets[key] for key in h3.PI_H3_ASSETS.values()} == {
        "pi_h3_4steps_path": PI_FILES["4steps"],
        "pi_h3_2steps_path": PI_FILES["2steps"],
    }
    assert not set(h3.PI_H3_ASSETS.values()) & set(h3.REQUIRED_ASSETS)


def test_shipped_assets_are_download_files_under_a_map():
    """The installer's drift checks (scripts/install_ltx.ps1): every asset is a
    ``downloads[].files[].path``, and every file path lies under a ``map[].to``
    of its own download."""
    manifest = _shipped_manifest()
    file_paths = {fl["path"] for dl in manifest["downloads"] for fl in dl["files"]}
    for key, value in manifest["assets"].items():
        assert value in file_paths, key
    for dl in manifest["downloads"]:
        tos = [m["to"] for m in dl["map"]]
        for fl in dl["files"]:
            assert any(fl["path"] == to or fl["path"].startswith(to + "/") for to in tos), fl["path"]
    (pi,) = [dl for dl in manifest["downloads"] if dl["repo"] == "Lakonik/pi-H3"]
    assert sorted(pi["include"]) == sorted(Path(p).name for p in PI_FILES.values())
    assert not any("*" in pattern for pattern in pi["include"])  # one file per line
    for fl in pi["files"]:
        assert fl["min"] < 2_192_176_240
    # every downloaded name is claimed by a map entry (top-level files)
    froms = {m["from"] for m in pi["map"]}
    assert set(pi["include"]) == froms


# =========================================================================== #
# 段 2: the chain (Chained / V2V / Retake) and the single path's
# Inpainting / Outpainting together with pi-H3
# =========================================================================== #

from api.models import GenerateChainRequest  # noqa: E402
from test_engine_h3_adapter import (  # noqa: E402
    CHAIN_DONE_EVENT,
    GOLDEN_CHAIN_KEYS_H3,
    GOLDEN_CHAIN_KEYS_H3_RETAKE,
    GOLDEN_CHAIN_KEYS_H3_V2V,
    _chain,
    _validated_chain,
)

CHAIN_BODY = {"prompt": PROMPT, "width": 640, "height": 384}


def _pi_backend(captured: list[dict], event: dict, tmp_path):
    """``_capturing_backend`` with the two pi-H3 assets resolved to tmp paths."""
    files = {key: str(tmp_path / f"{key}.safetensors") for key in h3.PI_H3_ASSETS.values()}
    be = _capturing_backend(captured, event)
    be._require_asset = lambda key: files[key]  # type: ignore[attr-defined]
    return be, files


# ------------------------------------------------------------- the chain field


def test_the_chain_field_defaults_to_off_and_is_honoured_on_h3():
    assert GenerateChainRequest.model_fields["pi_h3"].default == "off"
    assert set(GenerateChainRequest.model_fields["pi_h3"].annotation.__args__) == set(
        GenerateRequest.model_fields["pi_h3"].annotation.__args__
    )
    assert "pi_h3" in h3.CHAIN_HONOURED_FIELDS
    h3.reject_chain(_chain(pi_h3="4steps"))  # no raise


@pytest.mark.parametrize("choice", ["off", "4steps", "2steps"])
def test_to_clip_request_keeps_the_choice(choice):
    chain = _validated_chain(pi_h3=choice)
    clip = chain.to_clip_request(0)
    assert clip.pi_h3 == choice
    assert (clip.num_inference_steps, clip.shift_video, clip.shift_audio) == (
        chain.num_inference_steps, chain.shift_video, chain.shift_audio,
    )


# ---------------------------------------------------- the chain steps / shifts


@pytest.mark.parametrize("overrides", [{}, {"pi_h3": "off"}], ids=["omitted", "off"])
def test_off_chain_keeps_the_plain_defaults(overrides):
    chain = _validated_chain(**overrides)
    assert (chain.num_inference_steps, chain.shift_video, chain.shift_audio) == (20, 12.0, 3.0)


@pytest.mark.parametrize("choice, steps", [("4steps", 4), ("2steps", 2)])
@pytest.mark.parametrize(
    "shape",
    [
        {},
        {"source_video": {"video_id": "vid-1"}, "clips": [{"num_frames": 56}, {"num_frames": 39}]},
        {"retake": {"video_id": "vid-1", "window_start_sec": 0.0}, "clips": [{"num_frames": 124}]},
    ],
    ids=["chain", "v2v", "retake"],
)
def test_a_chain_choice_fills_the_unsent_values(choice, steps, shape):
    chain = _validated_chain(pi_h3=choice, **shape)
    assert (chain.num_inference_steps, chain.shift_video, chain.shift_audio) == (steps, 12.0, 3.0)
    assert isinstance(chain.num_inference_steps, int)


def test_the_chain_schema_default_8_is_not_an_explicit_value():
    # GenerateChainRequest's num_inference_steps defaults to LTX's 8; an
    # unsent one must become the choice's 4, not be refused as "8".
    chain = _chain(pi_h3="4steps")
    assert chain.num_inference_steps == 8 and "num_inference_steps" not in chain.model_fields_set
    engines.validate_chain_for_family("h3", chain)
    assert chain.num_inference_steps == 4


def test_chain_explicit_values_equal_to_the_choice_pass():
    chain = _validated_chain(pi_h3="2steps", num_inference_steps=2, shift_video=12.0, shift_audio=3)
    assert (chain.num_inference_steps, chain.shift_video, chain.shift_audio) == (2, 12.0, 3.0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"num_inference_steps": 8},
            'num_inference_steps=8 differs from the value pi_h3="4steps" uses (num_inference_steps=4)',
        ),
        (
            {"shift_video": 5.5},
            'shift_video=5.5 differs from the value pi_h3="4steps" uses (shift_video=12)',
        ),
        (
            {"shift_audio": 2.0},
            'shift_audio=2 differs from the value pi_h3="4steps" uses (shift_audio=3)',
        ),
    ],
)
def test_chain_explicit_other_value_is_422_stating_the_fact(overrides, message):
    with pytest.raises(APIError) as ei:
        _validated_chain(pi_h3="4steps", **overrides)
    err = ei.value
    assert err.code == "VALIDATION_ERROR" and err.status_code == 422
    assert err.detail == [{"loc": ["body"], "msg": f"Value error, {message}", "type": "value_error"}]


# ------------------------------------------------------ the chain acceptance check


def test_chain_preflight_looks_at_nothing_when_off():
    h3.preflight_request(_validated_chain(), assets=_never, transformer=_never)


def test_chain_preflight_missing_file_is_model_file_missing(tmp_path):
    missing = tmp_path / "pi-H3" / "pi_h3_2step_dwd_adapter_int8_convrot.safetensors"
    with pytest.raises(APIError) as ei:
        h3.preflight_request(_validated_chain(pi_h3="2steps"), assets=lambda key: missing, transformer=_never)
    assert ei.value.code == "MODEL_FILE_MISSING"
    assert ei.value.detail == "pi-H3 のファイルがありません（pi-H3/pi_h3_2step_dwd_adapter_int8_convrot.safetensors）。"


def test_chain_preflight_mismatch_is_model_incompatible(tmp_path, monkeypatch):
    pi = tmp_path / "pi.safetensors"
    _write(pi, {"x": ("F32", (1,))})
    monkeypatch.setattr(h3.h3_pih3_format, "model_problems", lambda pi_path, dit_path: ["finding"])
    with pytest.raises(APIError) as ei:
        h3.preflight_request(
            _validated_chain(pi_h3="4steps"), assets=lambda key: pi, transformer=lambda: tmp_path / "dit.safetensors"
        )
    assert ei.value.code == "MODEL_INCOMPATIBLE"
    assert ei.value.detail == "pi-H3 のファイル（pi.safetensors）と本体の重み（dit.safetensors）を照合しました: finding"


# ------------------------------------------------------------ the chain payloads


@pytest.mark.parametrize("overrides", [{}, {"pi_h3": "off"}], ids=["omitted", "off"])
def test_off_chain_payloads_are_todays_goldens(tmp_path, overrides):
    captured: list[dict] = []
    be = _capturing_backend(captured, CHAIN_DONE_EVENT)
    outcome = be.generate_chain(_validated_chain(**overrides), tmp_path / "job", seed=42)
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3
    assert outcome.pi_h3_used is None
    tail = tmp_path / "job" / "_source_tail.mp4"
    be.generate_chain(
        _validated_chain(source_video={"video_id": "vid-1"}, clips=[{"num_frames": 56}, {"num_frames": 39}], **overrides),
        tmp_path / "job", seed=42, source_tail_path=tail, source_context_frames=22,
    )
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3_V2V
    window = tmp_path / "job" / "_retake_window.mp4"
    be.generate_chain(
        _validated_chain(retake={"video_id": "vid-1", "window_start_sec": 0.0}, clips=[{"num_frames": 124}], **overrides),
        tmp_path / "job", seed=42, retake_window_path=window,
    )
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3_RETAKE


@pytest.mark.parametrize("choice, steps", [("4steps", 4), ("2steps", 2)])
def test_pi_h3_chain_payload_carries_steps_and_path(tmp_path, choice, steps):
    captured: list[dict] = []
    be, files = _pi_backend(captured, {**CHAIN_DONE_EVENT, "pi_h3_used": choice}, tmp_path)
    outcome = be.generate_chain(
        _validated_chain(pi_h3=choice, nag_enabled=True, negative_prompt="blurry", neg_method="vsf", keep_resident=True),
        tmp_path / "job", seed=42,
    )
    (payload,) = captured
    assert payload["pi_h3"] == {"steps": steps, "path": files[h3.PI_H3_ASSETS[choice]]}
    assert (payload["num_inference_steps"], payload["shift_video"], payload["shift_audio"]) == (steps, 12.0, 3.0)
    assert list(payload) == GOLDEN_CHAIN_KEYS_H3 + ["nag", "pi_h3", "keep_resident"]
    assert outcome.pi_h3_used == choice


def test_pi_h3_v2v_and_retake_payloads(tmp_path):
    captured: list[dict] = []
    be, files = _pi_backend(captured, {**CHAIN_DONE_EVENT, "pi_h3_used": "4steps"}, tmp_path)
    expected = {"steps": 4, "path": files["pi_h3_4steps_path"]}
    tail = tmp_path / "job" / "_source_tail.mp4"
    be.generate_chain(
        _validated_chain(pi_h3="4steps", source_video={"video_id": "vid-1"},
                         clips=[{"num_frames": 56}, {"num_frames": 39}]),
        tmp_path / "job", seed=42, source_tail_path=tail, source_context_frames=22,
    )
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3_V2V + ["pi_h3"]
    assert captured[-1]["pi_h3"] == expected and captured[-1]["num_inference_steps"] == 4
    window = tmp_path / "job" / "_retake_window.mp4"
    be.generate_chain(
        _validated_chain(pi_h3="4steps", retake={"video_id": "vid-1", "window_start_sec": 0.0},
                         clips=[{"num_frames": 124}]),
        tmp_path / "job", seed=42, retake_window_path=window,
    )
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3_RETAKE + ["pi_h3"]
    assert captured[-1]["pi_h3"] == expected


def test_the_chain_relays_the_engine_pi_h3_record(tmp_path):
    record = {"choice": "4steps", "steps": 4, "file": "pi.safetensors"}
    event = {
        **CHAIN_DONE_EVENT,
        "pi_h3_used": "4steps",
        "h3": {**CHAIN_DONE_EVENT["h3"], "pi_h3": record},
    }
    captured: list[dict] = []
    be, _files = _pi_backend(captured, event, tmp_path)
    outcome = be.generate_chain(_validated_chain(pi_h3="4steps"), tmp_path / "job", seed=42)
    assert outcome.chain_metadata["h3"]["pi_h3"] == record
    assert outcome.pi_h3_used == "4steps"


# ------------------------------------------------------------------ LTX chains


@pytest.mark.parametrize("adapter", [ltx23, ltx25], ids=["ltx23", "ltx25"])
@pytest.mark.parametrize("choice", ["4steps", "2steps"])
def test_ltx_chains_refuse_with_the_h3_only_sentence(adapter, choice):
    with pytest.raises(APIError) as ei:
        adapter.reject_chain(_chain(pi_h3=choice))
    assert ei.value.code == "FEATURE_UNSUPPORTED" and ei.value.status_code == 422
    assert ei.value.detail == H3_ONLY_DETAIL
    adapter.reject_chain(_chain(pi_h3="off"))  # no raise


@pytest.mark.parametrize("base_model", ["LTX23", "LTX25"])
def test_ltx_chains_refuse_over_http(three_family_client, base_model):
    _activate(three_family_client, base_model)
    before = _job_count(three_family_client)
    r = three_family_client.post(
        "/api/v1/generate/chain",
        json={"prompt": PROMPT, "clips": [{"num_frames": 25}, {"num_frames": 25}], "pi_h3": "4steps"},
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert error["detail"] == H3_ONLY_DETAIL
    assert _job_count(three_family_client) == before


# --------------------------------------------------------- HTTP: chains on mock


def _stub_match(monkeypatch) -> list[tuple]:
    seen: list[tuple] = []
    monkeypatch.setattr(
        h3.h3_pih3_format, "model_problems", lambda pi, dit: (seen.append((Path(pi), Path(dit))), [])[1]
    )
    return seen


def _chain_completed(client, body: dict) -> str:
    r = client.post("/api/v1/generate/chain", json=body)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    return job_id


@pytest.mark.parametrize("choice, steps", [("4steps", 4), ("2steps", 2)])
def test_mock_pi_h3_chain_runs_and_records(pih3_client, monkeypatch, choice, steps):
    seen = _stub_match(monkeypatch)
    _activate(pih3_client, "MiniMaxH3")
    job_id = _chain_completed(
        pih3_client, {**CHAIN_BODY, "clips": [{"num_frames": 124}, {"num_frames": 124}], "pi_h3": choice}
    )
    meta = _metadata(pih3_client, job_id)
    assert meta["pi_h3_used"] == choice
    assert meta["h3"]["pi_h3"] == {"choice": choice, "steps": steps, "mock": True}
    assert meta["request"]["pi_h3"] == choice
    assert (
        meta["request"]["num_inference_steps"], meta["request"]["shift_video"], meta["request"]["shift_audio"]
    ) == (steps, 12.0, 3.0)
    # the stored per-clip record keeps the choice and its values
    stored = pih3_client.app_context.job_store.get(job_id).request
    assert (stored.pi_h3, stored.num_inference_steps) == (choice, steps)
    ((pi, dit),) = seen
    assert pi.name == Path(PI_FILES[choice]).name and dit.name == "minimax_h3_dit.safetensors"


@pytest.mark.parametrize("overrides", [{}, {"pi_h3": "off"}], ids=["omitted", "off"])
def test_mock_off_chain_record_is_unchanged(pih3_client, monkeypatch, overrides):
    monkeypatch.setattr(h3.h3_pih3_format, "model_problems", _never)
    _activate(pih3_client, "MiniMaxH3")
    job_id = _chain_completed(
        pih3_client, {**CHAIN_BODY, "clips": [{"num_frames": 124}, {"num_frames": 124}], **overrides}
    )
    meta = _metadata(pih3_client, job_id)
    assert "pi_h3_used" not in meta
    assert "pi_h3" not in meta["h3"]
    assert meta["request"]["num_inference_steps"] == 20


def test_http_chain_missing_file_is_422_and_no_job(pih3_client):
    _activate(pih3_client, "MiniMaxH3")
    (pih3_client.models_dir / PI_FILES["4steps"]).unlink()
    before = _job_count(pih3_client)
    r = pih3_client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BODY, "clips": [{"num_frames": 124}, {"num_frames": 124}], "pi_h3": "4steps"},
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "MODEL_FILE_MISSING"
    assert _job_count(pih3_client) == before


def test_http_chain_header_mismatch_is_422_and_no_job(pih3_client):
    """The real ``model_problems`` on the tiny stand-in file."""
    _activate(pih3_client, "MiniMaxH3")
    before = _job_count(pih3_client)
    r = pih3_client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BODY, "clips": [{"num_frames": 124}, {"num_frames": 124}], "pi_h3": "2steps"},
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "MODEL_INCOMPATIBLE"
    assert error["detail"].startswith("pi-H3 のファイル（pi_h3_2step_dwd_adapter_int8_convrot.safetensors）と")
    assert _job_count(pih3_client) == before


def test_http_chain_explicit_other_steps_is_422(pih3_client, monkeypatch):
    monkeypatch.setattr(h3.h3_pih3_format, "model_problems", _never)
    _activate(pih3_client, "MiniMaxH3")
    r = pih3_client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BODY, "clips": [{"num_frames": 124}], "pi_h3": "2steps", "num_inference_steps": 20,
              "retake": {"video_id": "not-looked-up", "window_start_sec": 0.0}},
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["detail"][0]["msg"] == (
        'Value error, num_inference_steps=20 differs from the value pi_h3="2steps" uses (num_inference_steps=2)'
    )


# ---------------------------------- HTTP: Retake / V2V / Inpainting / Outpainting

import shutil  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not on PATH"
)


@needs_ffmpeg
def test_mock_pi_h3_retake_runs_and_records(pih3_client, monkeypatch, tmp_path):
    from test_retake_api import _upload_window_source

    _stub_match(monkeypatch)
    _activate(pih3_client, "MiniMaxH3")
    vid = _upload_window_source(pih3_client, tmp_path)
    job_id = _chain_completed(pih3_client, {
        "prompt": PROMPT, "width": 384, "height": 256, "frame_rate": 24.0, "seed": 123,
        "clips": [{"num_frames": 124}], "pi_h3": "4steps",
        "retake": {"video_id": vid, "window_start_sec": 2.0},
    })
    meta = _metadata(pih3_client, job_id)
    assert meta["pi_h3_used"] == "4steps"
    assert meta["h3"]["pi_h3"] == {"choice": "4steps", "steps": 4, "mock": True}
    assert meta["retake"]["audio_strength"] == 0.5
    assert meta["request"]["num_inference_steps"] == 4


@needs_ffmpeg
def test_mock_pi_h3_v2v_runs_and_records(pih3_client, monkeypatch, tmp_path):
    from test_h3_chain_v2v_api import _upload_av_source

    _stub_match(monkeypatch)
    _activate(pih3_client, "MiniMaxH3")
    vid = _upload_av_source(pih3_client, tmp_path)
    job_id = _chain_completed(pih3_client, {
        **CHAIN_BODY, "clips": [{"num_frames": 56}, {"num_frames": 39}],
        "source_video": {"video_id": vid}, "pi_h3": "2steps",
    })
    meta = _metadata(pih3_client, job_id)
    assert meta["pi_h3_used"] == "2steps"
    assert meta["h3"]["pi_h3"] == {"choice": "2steps", "steps": 2, "mock": True}
    assert "v2v" in meta


@needs_ffmpeg
def test_mock_pi_h3_outpaint_passes_and_records(pih3_client, monkeypatch, tmp_path):
    from test_h3_outpaint_api import _body, _upload

    _stub_match(monkeypatch)
    _activate(pih3_client, "MiniMaxH3")
    vid = _upload(pih3_client, tmp_path)
    r = pih3_client.post("/api/v1/generate", json={**_body(vid), "pi_h3": "4steps"})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert pih3_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    meta = _metadata(pih3_client, job_id)
    assert meta["pi_h3_used"] == "4steps"
    assert meta["h3"]["pi_h3"] == {"choice": "4steps", "steps": 4, "mock": True}
    assert meta["h3"]["outpaint"]["mock"] is True
    assert meta["request"]["num_inference_steps"] == 4


@needs_ffmpeg
def test_mock_pi_h3_inpaint_passes_and_records(pih3_client, monkeypatch, tmp_path):
    from test_h3_inpaint_api import _body, _mask, _upload

    _stub_match(monkeypatch)
    _activate(pih3_client, "MiniMaxH3")
    vid = _upload(pih3_client, tmp_path)
    mask = _mask(pih3_client, tmp_path)
    r = pih3_client.post("/api/v1/generate", json={**_body(vid, mask), "pi_h3": "2steps"})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert pih3_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    meta = _metadata(pih3_client, job_id)
    assert meta["pi_h3_used"] == "2steps"
    assert meta["h3"]["pi_h3"] == {"choice": "2steps", "steps": 2, "mock": True}
    assert meta["inpaint"]["mock"] is True
    assert meta["request"]["num_inference_steps"] == 2


def test_outpaint_and_inpaint_payloads_with_pi_h3(tmp_path):
    captured: list[dict] = []
    be, files = _pi_backend(captured, {**DONE_EVENT, "pi_h3_used": "4steps"}, tmp_path)
    src = tmp_path / "src.mp4"
    op = _validated(
        pi_h3="4steps", width=640, height=384, num_frames=39,
        outpaint={"source_video_id": "vid", "pad_left": 64, "pad_right": 64, "pad_top": 64, "pad_bottom": 64},
    )
    be.generate(op, tmp_path / "job", seed=1, outpaint_source_path=src)
    keys = list(captured[-1])
    assert keys[-3:] == ["outpaint", "pi_h3", "output_path"]
    assert captured[-1]["pi_h3"] == {"steps": 4, "path": files["pi_h3_4steps_path"]}
    assert captured[-1]["num_inference_steps"] == 4
    ip = _validated(
        pi_h3="2steps", width=320, height=288, num_frames=39,
        inpaint={"source_video_id": "vid", "mask_video_id": "mask"},
    )
    be.generate(ip, tmp_path / "job", seed=1, inpaint_source_path=src, inpaint_mask_path=src)
    keys = list(captured[-1])
    assert keys[-3:] == ["inpaint", "pi_h3", "output_path"]
    assert captured[-1]["pi_h3"] == {"steps": 2, "path": files["pi_h3_2steps_path"]}
