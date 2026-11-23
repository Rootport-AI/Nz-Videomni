"""MiniMax H3 Sol-H3-VAE (``sol_vae``) on the app side: the request fields, the
refusal of ``pi_h3`` together with it, the worker payloads, the ``done`` echo,
the mock job and metadata.json, the LTX refusals and the installer descriptor.

Pinned here, none of it needing a GPU or a real Sol-H3-VAE file:

1. ``sol_vae`` off (omitted or ``false``) changes nothing on the wire: the
   payloads are the goldens, ``outcome.sol_vae_used`` is None and
   metadata.json has no ``sol_vae_used`` key (its request copy gains
   ``"sol_vae": false``, the only change);
2. ``pi_h3`` other than "off" together with ``sol_vae`` true is a 422 that
   states the fact (single and chain);
3. on, the payload carries ``sol_vae = {path}`` (the descriptor's asset,
   ``""`` when undeclared; the file is never required here), the ``done`` echo
   becomes ``outcome.sol_vae_used``;
4. the mock job: "on" when the asset's file is there, "on->off" when not;
5. LTX 2.3 / 2.5 refuse it with the H3-only sentence and publish it;
6. the shipped descriptor: the asset is a downloads' file under a map, and the
   VAE scans do not list it.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import main
from api.errors import APIError
from api.models import GenerateChainRequest, GenerateRequest
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
    CHAIN_DONE_EVENT,
    DONE_EVENT,
    GOLDEN_CHAIN_KEYS_H3,
    GOLDEN_CHAIN_KEYS_H3_RETAKE,
    GOLDEN_CHAIN_KEYS_H3_V2V,
    GOLDEN_GENERATE_KEYS_H3,
    PROMPT,
    _activate,
    _capturing_backend,
    _chain,
    _metadata,
    _request,
    _validated,
    _validated_chain,
)
from test_h3_format import _write

H3_ONLY_DETAIL = "Sol-H3-VAE は MiniMax H3 専用です。LTX では使えません。"
BODY = {"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 22}
CHAIN_BODY = {"prompt": PROMPT, "width": 640, "height": 384, "clips": [{"num_frames": 124}, {"num_frames": 124}]}
SOL_FILE = "MiniMaxH3/VAE/Sol/minimax_h3_sol_vae_int8_convrot.safetensors"


def _sol_backend(captured: list[dict], event: dict, tmp_path, *, declared: bool = True):
    """``_capturing_backend`` with a config and a descriptor whose ``sol_vae_path``
    asset (when ``declared``) names a file under ``tmp_path`` (not created)."""
    be = _capturing_backend(captured, event)
    be.config = AppConfig.model_validate({"model": {"models_dir": (tmp_path / "models").as_posix()}})
    be.descriptor = types.SimpleNamespace(assets={h3.SOL_VAE_ASSET: SOL_FILE} if declared else {})
    return be, str((tmp_path / "models" / SOL_FILE).resolve())


# --------------------------------------------------------------------------- #
# the fields and the tables
# --------------------------------------------------------------------------- #


def test_the_fields_default_to_false():
    assert GenerateRequest.model_fields["sol_vae"].default is False
    assert GenerateChainRequest.model_fields["sol_vae"].default is False
    assert _request(sol_vae=True).sol_vae is True


def test_h3_honours_it_and_does_not_publish_it():
    assert "sol_vae" in h3.HONOURED_FIELDS and "sol_vae" in h3.CHAIN_HONOURED_FIELDS
    assert "sol_vae" not in h3.UNSUPPORTED_FEATURES
    h3.reject_unsupported(_request(sol_vae=True))  # no raise
    h3.reject_chain(_chain(sol_vae=True))  # no raise


@pytest.mark.parametrize("value", [False, True])
def test_to_clip_request_keeps_it(value):
    assert _validated_chain(sol_vae=value).to_clip_request(0).sol_vae is value


def test_on_changes_no_step_or_shift():
    req = _validated(sol_vae=True)
    assert (req.num_inference_steps, req.shift_video, req.shift_audio) == (20, 12.0, 3.0)
    chain = _validated_chain(sol_vae=True)
    assert (chain.num_inference_steps, chain.shift_video, chain.shift_audio) == (20, 12.0, 3.0)


# --------------------------------------------------------------------------- #
# pi_h3 and sol_vae together
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("choice", ["4steps", "2steps"])
def test_pi_h3_with_sol_vae_is_refused_stating_the_fact(choice):
    message = f'pi_h3="{choice}" と sol_vae=true は同時に指定できません。'
    with pytest.raises(ValueError) as ei:
        h3.prefill_request(_request(pi_h3=choice, sol_vae=True))
    assert str(ei.value) == message
    with pytest.raises(APIError) as ei:
        _validated(pi_h3=choice, sol_vae=True)
    assert ei.value.code == "VALIDATION_ERROR" and ei.value.status_code == 422
    assert ei.value.detail == [{"loc": ["body"], "msg": f"Value error, {message}", "type": "value_error"}]
    with pytest.raises(APIError) as ei:
        _validated_chain(pi_h3=choice, sol_vae=True)
    assert ei.value.code == "VALIDATION_ERROR"
    assert ei.value.detail[0]["msg"] == f"Value error, {message}"


def test_either_alone_passes():
    _validated(pi_h3="4steps")
    _validated(sol_vae=True)
    _validated_chain(pi_h3="2steps")
    _validated_chain(sol_vae=True)


# --------------------------------------------------------------------------- #
# the worker payloads
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("overrides", [{}, {"sol_vae": False}], ids=["omitted", "off"])
def test_off_payloads_are_todays_goldens(tmp_path, overrides):
    captured: list[dict] = []
    be, _path = _sol_backend(captured, DONE_EVENT, tmp_path)
    outcome = be.generate(_validated(**overrides), tmp_path / "job", seed=42)
    assert list(captured[-1]) == GOLDEN_GENERATE_KEYS_H3
    assert outcome.sol_vae_used is None
    be, _path = _sol_backend(captured, CHAIN_DONE_EVENT, tmp_path)
    outcome = be.generate_chain(_validated_chain(**overrides), tmp_path / "job", seed=42)
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3
    assert outcome.sol_vae_used is None
    be.generate_chain(
        _validated_chain(source_video={"video_id": "vid-1"}, clips=[{"num_frames": 56}, {"num_frames": 39}], **overrides),
        tmp_path / "job", seed=42, source_tail_path=tmp_path / "t.mp4", source_context_frames=22,
    )
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3_V2V
    be.generate_chain(
        _validated_chain(retake={"video_id": "vid-1", "window_start_sec": 0.0}, clips=[{"num_frames": 124}], **overrides),
        tmp_path / "job", seed=42, retake_window_path=tmp_path / "w.mp4",
    )
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3_RETAKE


@pytest.mark.parametrize("used", ["on", "on->off"])
def test_sol_payload_carries_the_path_and_the_echo(tmp_path, used):
    captured: list[dict] = []
    be, path = _sol_backend(captured, {**DONE_EVENT, "sol_vae_used": used}, tmp_path)
    outcome = be.generate(_validated(sol_vae=True, keep_resident=True), tmp_path / "job", seed=42)
    (payload,) = captured
    assert payload["sol_vae"] == {"path": path}
    keys = list(payload)
    assert keys[-3:] == ["sol_vae", "output_path", "keep_resident"]
    assert [k for k in keys if k not in ("sol_vae", "keep_resident")] == GOLDEN_GENERATE_KEYS_H3
    assert (payload["num_inference_steps"], payload["shift_video"], payload["shift_audio"]) == (20, 12.0, 3.0)
    assert outcome.sol_vae_used == used


def test_an_undeclared_asset_sends_an_empty_path(tmp_path):
    captured: list[dict] = []
    be, _path = _sol_backend(captured, DONE_EVENT, tmp_path, declared=False)
    be.generate(_validated(sol_vae=True), tmp_path / "job", seed=42)
    assert captured[-1]["sol_vae"] == {"path": ""}


def test_sol_chain_payloads(tmp_path):
    captured: list[dict] = []
    be, path = _sol_backend(captured, {**CHAIN_DONE_EVENT, "sol_vae_used": "on"}, tmp_path)
    outcome = be.generate_chain(_validated_chain(sol_vae=True, keep_resident=True), tmp_path / "job", seed=42)
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3 + ["sol_vae", "keep_resident"]
    assert captured[-1]["sol_vae"] == {"path": path}
    assert outcome.sol_vae_used == "on"
    be.generate_chain(
        _validated_chain(sol_vae=True, source_video={"video_id": "vid-1"}, clips=[{"num_frames": 56}, {"num_frames": 39}]),
        tmp_path / "job", seed=42, source_tail_path=tmp_path / "t.mp4", source_context_frames=22,
    )
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3_V2V + ["sol_vae"]
    be.generate_chain(
        _validated_chain(sol_vae=True, retake={"video_id": "vid-1", "window_start_sec": 0.0},
                         clips=[{"num_frames": 124}]),
        tmp_path / "job", seed=42, retake_window_path=tmp_path / "w.mp4",
    )
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3_RETAKE + ["sol_vae"]


def test_the_engine_sol_record_rides_metadata_h3(tmp_path):
    record = {"used": "on->off", "reason": "too_large", "latent_tokens": 2_000_000}
    event = {**DONE_EVENT, "sol_vae_used": "on->off", "h3": {**DONE_EVENT.get("h3", {}), "sol_vae": record}}
    captured: list[dict] = []
    be, _path = _sol_backend(captured, event, tmp_path)
    outcome = be.generate(_validated(sol_vae=True), tmp_path / "job", seed=42)
    assert outcome.h3["sol_vae"] == record


# --------------------------------------------------------------------------- #
# HTTP over the mock backend
# --------------------------------------------------------------------------- #


@pytest.fixture()
def sol_client(tmp_path, monkeypatch):
    """``three_family_client``'s LTX 2.3 + MiniMax H3, with the H3 descriptor
    declaring the Sol-H3-VAE asset and a tiny stand-in file there."""
    ltx = base_model_descriptor()
    h3d = h3_base_model_descriptor()
    h3d["assets"] = {**h3d["assets"], h3.SOL_VAE_ASSET: SOL_FILE}
    fragment = build_model_layout(tmp_path, [ltx, h3d])
    models_dir = tmp_path / "models"
    write_gguf_with_kv(
        models_dir / ltx["categories"]["transformer"]["default_file"],
        **{"general.architecture": "ltxv", "model_version": "2.3.0"},
    )
    write_h3_model_files(models_dir, h3d, monkeypatch)
    (models_dir / SOL_FILE).parent.mkdir(parents=True, exist_ok=True)
    _write(models_dir / SOL_FILE, {"x": ("F32", (1,))})
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


def _completed(client, path: str, body: dict) -> str:
    r = client.post(path, json=body)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    return job_id


@pytest.mark.parametrize("present, used", [(True, "on"), (False, "on->off")])
def test_mock_sol_job_records_whether_the_file_is_there(sol_client, present, used):
    _activate(sol_client, "MiniMaxH3")
    if not present:
        (sol_client.models_dir / SOL_FILE).unlink()
    meta = _metadata(sol_client, _completed(sol_client, "/api/v1/generate", {**BODY, "sol_vae": True}))
    assert meta["sol_vae_used"] == used
    assert meta["request"]["sol_vae"] is True
    meta = _metadata(sol_client, _completed(sol_client, "/api/v1/generate/chain", {**CHAIN_BODY, "sol_vae": True}))
    assert meta["sol_vae_used"] == used
    assert meta["request"]["sol_vae"] is True


@pytest.mark.parametrize("overrides", [{}, {"sol_vae": False}], ids=["omitted", "off"])
def test_mock_off_records_have_no_sol_key(sol_client, overrides):
    _activate(sol_client, "MiniMaxH3")
    meta = _metadata(sol_client, _completed(sol_client, "/api/v1/generate", {**BODY, **overrides}))
    assert "sol_vae_used" not in meta and "h3" not in meta
    assert meta["request"]["sol_vae"] is False   # the request copy's one new key
    meta = _metadata(sol_client, _completed(sol_client, "/api/v1/generate/chain", {**CHAIN_BODY, **overrides}))
    assert "sol_vae_used" not in meta
    assert meta["request"]["sol_vae"] is False


def test_http_pi_h3_with_sol_vae_is_422_and_no_job(sol_client):
    _activate(sol_client, "MiniMaxH3")
    before = len(sol_client.get("/api/v1/jobs").json())
    for path, body in (("/api/v1/generate", BODY), ("/api/v1/generate/chain", CHAIN_BODY)):
        r = sol_client.post(path, json={**body, "pi_h3": "4steps", "sol_vae": True})
        assert r.status_code == 422, r.text
        error = r.json()["error"]
        assert error["code"] == "VALIDATION_ERROR"
        assert error["detail"][0]["msg"] == 'Value error, pi_h3="4steps" と sol_vae=true は同時に指定できません。'
    assert len(sol_client.get("/api/v1/jobs").json()) == before


def test_models_does_not_publish_it_on_h3(sol_client):
    listing = sol_client.get("/api/v1/models").json()
    entry = next(b for b in listing["base_models"] if b["id"] == "MiniMaxH3")
    assert "sol_vae" not in entry["unsupported_features"]
    assert "sol_vae" in next(b for b in listing["base_models"] if b["id"] == "LTX23")["unsupported_features"]


def test_the_vae_scans_do_not_list_the_sol_file(sol_client):
    registry = sol_client.app_context.model_registry
    for category in ("video_vae", "audio"):
        files = [Path(e.path).name for e in registry.entries(category, base_model="MiniMaxH3")]
        assert Path(SOL_FILE).name not in files, (category, files)


# --------------------------------------------------------------------------- #
# LTX refuses it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("adapter", [ltx23, ltx25], ids=["ltx23", "ltx25"])
def test_ltx_refuses_with_the_h3_only_sentence(adapter):
    for reject, request, off in (
        (adapter.reject_unsupported, GenerateRequest(prompt=PROMPT, sol_vae=True), GenerateRequest(prompt=PROMPT)),
        (adapter.reject_chain, _chain(sol_vae=True), _chain()),
    ):
        with pytest.raises(APIError) as ei:
            reject(request)
        assert ei.value.code == "FEATURE_UNSUPPORTED" and ei.value.status_code == 422
        assert ei.value.detail == H3_ONLY_DETAIL
        reject(off)  # no raise
    assert "sol_vae" in adapter.UNSUPPORTED_FEATURES


@pytest.mark.parametrize("base_model", ["LTX23", "LTX25"])
def test_ltx_refuses_over_http(three_family_client, base_model):
    _activate(three_family_client, base_model)
    for path, body in (
        ("/api/v1/generate", {"prompt": PROMPT, "num_frames": 25}),
        ("/api/v1/generate/chain", {"prompt": PROMPT, "clips": [{"num_frames": 25}, {"num_frames": 25}]}),
    ):
        r = three_family_client.post(path, json={**body, "sol_vae": True})
        assert r.status_code == 422, r.text
        error = r.json()["error"]
        assert error["code"] == "FEATURE_UNSUPPORTED"
        assert error["detail"] == H3_ONLY_DETAIL


# --------------------------------------------------------------------------- #
# the shipped descriptor
# --------------------------------------------------------------------------- #


def test_shipped_descriptor_declares_the_file_and_downloads_it():
    descriptor = load_base_models(AppConfig().manifest_dir)["MiniMaxH3"]
    assert descriptor.assets[h3.SOL_VAE_ASSET] == SOL_FILE
    assert h3.SOL_VAE_ASSET not in h3.REQUIRED_ASSETS
    manifest = json.loads((AppConfig().manifest_dir / "40-minimaxh3.json").read_text(encoding="utf-8-sig"))
    (sol,) = [dl for dl in manifest["downloads"] if any(fl["path"] == SOL_FILE for fl in dl["files"])]
    assert sol["repo"] == "Comfy-Org/MiniMax-H3"
    assert sol["include"] == ["vae/minimax_h3_sol_vae_int8_convrot.safetensors"]
    assert sol["map"] == [{"from": sol["include"][0], "to": SOL_FILE}]
    (fl,) = sol["files"]
    assert fl["min"] == 1_900_000_000 < 1_988_306_504
