"""§1-107: which known weight the loaded transformer is, by whole-file SHA-256
(services/weight_identity.py, descriptor ``known_weights``, GET /models
``base_models[].transformer_role``).

The hash is computed on the SUCCESS path of a pipeline load only, and only for
a base model whose descriptor lists ``known_weights`` and a file whose size
matches one of them. GET /models reads the cache and never hashes. The cache
survives a restart (a new instance reads the file) and a broken one is set
aside as ``.bad``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import main
from config import AppConfig
from conftest import (
    _make_args,
    base_model_descriptor,
    build_model_layout,
    h3_base_model_descriptor,
    write_h3_model_files,
    write_manifests,
)
from services import weight_identity
from services.base_models import KnownWeight, load_base_models
from services.weight_identity import CACHE_FILENAME, WeightIdentity

OTHER_SHA = "0" * 64


@pytest.fixture()
def hash_calls(monkeypatch) -> list[str]:
    """Records every real hash computation (the wrapped ``sha256_file``)."""
    calls: list[str] = []
    real = weight_identity.sha256_file

    def counting(path):
        calls.append(str(path))
        return real(path)

    monkeypatch.setattr(weight_identity, "sha256_file", counting)
    return calls


@pytest.fixture()
def identify_calls(monkeypatch) -> list[str]:
    """Records every ``WeightIdentity.identify`` call (hit or miss)."""
    calls: list[str] = []
    real = WeightIdentity.identify

    def counting(self, path, known):
        calls.append(str(path))
        return real(self, path, known)

    monkeypatch.setattr(WeightIdentity, "identify", counting)
    return calls


def _weight_file(tmp_path: Path, content: bytes = b"h3-dit-bytes" * 10) -> Path:
    path = tmp_path / "models" / "Weights" / "dit.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _known(path: Path, role: str = "fl2va", sha256: str | None = None) -> tuple[KnownWeight, ...]:
    data = path.read_bytes()
    return (
        KnownWeight(
            path="Weights/dit.safetensors",
            size=len(data),
            sha256=sha256 or hashlib.sha256(data).hexdigest(),
            role=role,
        ),
    )


# --------------------------------------------------------------------------- #
# WeightIdentity
# --------------------------------------------------------------------------- #


def test_matching_sha256_gives_the_role_and_writes_the_cache(tmp_path, hash_calls):
    dit = _weight_file(tmp_path)
    cache = tmp_path / CACHE_FILENAME
    identity = WeightIdentity.load(cache)

    assert identity.identify(dit, _known(dit)) == "fl2va"
    assert len(hash_calls) == 1
    raw = json.loads(cache.read_text(encoding="utf-8"))
    assert raw["schema"] == 1
    (entry,) = raw["entries"].values()
    assert entry["size"] == dit.stat().st_size
    assert entry["sha256"] == hashlib.sha256(dit.read_bytes()).hexdigest()


def test_same_size_other_sha256_is_none(tmp_path, hash_calls):
    dit = _weight_file(tmp_path)
    identity = WeightIdentity.load(tmp_path / CACHE_FILENAME)
    assert identity.identify(dit, _known(dit, sha256=OTHER_SHA)) is None
    assert len(hash_calls) == 1  # hashed (the size matched), then no match


def test_size_mismatch_is_not_hashed(tmp_path, hash_calls):
    dit = _weight_file(tmp_path)
    known = (KnownWeight("Weights/dit.safetensors", dit.stat().st_size + 1, OTHER_SHA, "fl2va"),)
    identity = WeightIdentity.load(tmp_path / CACHE_FILENAME)
    assert identity.identify(dit, known) is None
    assert hash_calls == []
    assert not (tmp_path / CACHE_FILENAME).exists()


def test_cache_hit_and_restart(tmp_path, hash_calls):
    dit = _weight_file(tmp_path)
    cache = tmp_path / CACHE_FILENAME
    known = _known(dit)
    first = WeightIdentity.load(cache)
    assert first.identify(dit, known) == "fl2va"
    assert first.identify(dit, known) == "fl2va"
    assert len(hash_calls) == 1

    restarted = WeightIdentity.load(cache)  # a new process reads the file
    assert restarted.cached_role(dit, known) == "fl2va"
    assert restarted.identify(dit, known) == "fl2va"
    assert len(hash_calls) == 1


def test_cached_role_never_hashes(tmp_path, hash_calls):
    dit = _weight_file(tmp_path)
    identity = WeightIdentity.load(tmp_path / CACHE_FILENAME)
    assert identity.cached_role(dit, _known(dit)) is None
    assert identity.cached_role(tmp_path / "missing.safetensors", _known(dit)) is None
    assert hash_calls == []


def test_changed_file_is_not_trusted_from_the_cache(tmp_path, hash_calls):
    dit = _weight_file(tmp_path)
    known = _known(dit)
    identity = WeightIdentity.load(tmp_path / CACHE_FILENAME)
    assert identity.identify(dit, known) == "fl2va"
    stat = dit.stat()
    # Same size, other bytes, other mtime: the cached hash no longer applies.
    dit.write_bytes(bytes(reversed(dit.read_bytes())))
    os.utime(dit, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))
    assert identity.cached_role(dit, known) is None
    assert identity.identify(dit, known) is None
    assert len(hash_calls) == 2


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"schema": 2, "entries": {}}),
        json.dumps({"schema": 1, "entries": []}),
        json.dumps({"schema": 1, "entries": {"x": {"size": 1, "mtime_ns": 1, "sha256": "zz"}}}),
    ],
)
def test_broken_cache_is_set_aside(tmp_path, content, caplog):
    cache = tmp_path / CACHE_FILENAME
    cache.write_text(content, encoding="utf-8")
    identity = WeightIdentity.load(cache)
    assert identity._entries == {}
    assert not cache.exists()
    assert (tmp_path / (CACHE_FILENAME + ".bad")).read_text(encoding="utf-8") == content
    assert any("キャッシュ" in r.getMessage() for r in caplog.records)


def test_unwritable_cache_is_only_a_warning(tmp_path, monkeypatch, caplog):
    dit = _weight_file(tmp_path)
    identity = WeightIdentity.load(tmp_path / CACHE_FILENAME)

    def refuse(self, text):
        raise OSError("read-only")

    monkeypatch.setattr(WeightIdentity, "_write_atomic", refuse)
    assert identity.identify(dit, _known(dit)) == "fl2va"
    assert identity.cached_role(dit, _known(dit)) == "fl2va"  # memory still holds it
    assert any("保存できませんでした" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# descriptor shape (services/base_models.py)
# --------------------------------------------------------------------------- #

_GOOD = {
    "path": "LTX23/Weights/x.safetensors",
    "size": 10,
    "sha256": "A" * 64,
    "role": "fl2va",
}


def _load_with(tmp_path, known_weights):
    descriptor = {**base_model_descriptor(), "known_weights": known_weights}
    return load_base_models(write_manifests(tmp_path / "manifests", [descriptor]))


def test_known_weights_absent_is_empty(tmp_path):
    (descriptor,) = load_base_models(write_manifests(tmp_path / "manifests")).values()
    assert descriptor.known_weights == ()


def test_known_weights_parsed_and_sha256_lowercased(tmp_path):
    (descriptor,) = _load_with(tmp_path, [_GOOD]).values()
    assert descriptor.known_weights == (
        KnownWeight("LTX23/Weights/x.safetensors", 10, "a" * 64, "fl2va"),
    )


@pytest.mark.parametrize(
    "known_weights, message",
    [
        ({}, "'known_weights' must be an array"),
        (["x"], r"known_weights\[0\] must be an object"),
        ([{**_GOOD, "path": "../x"}], "must be a relative path"),
        ([{**_GOOD, "path": ""}], "path must be a non-empty string"),
        ([{**_GOOD, "size": 0}], "size must be a positive integer"),
        ([{**_GOOD, "size": True}], "size must be a positive integer"),
        ([{**_GOOD, "size": "10"}], "size must be a positive integer"),
        ([{**_GOOD, "sha256": "abc"}], "sha256 must be 64 hex characters"),
        ([{**_GOOD, "sha256": "g" * 64}], "sha256 must be 64 hex characters"),
        ([{**_GOOD, "role": " "}], "role must be a non-empty string"),
    ],
)
def test_known_weights_broken_fails_loud(tmp_path, known_weights, message):
    with pytest.raises(RuntimeError, match=message):
        _load_with(tmp_path, known_weights)


def test_shipped_descriptors_known_weights():
    config = AppConfig()
    shipped = load_base_models(config.manifest_dir)
    roles = {w.role: w for w in shipped["MiniMaxH3"].known_weights}
    assert set(roles) == {"fl2va", "ref2va"}
    assert roles["fl2va"].sha256.startswith("e889202c")
    assert roles["ref2va"].sha256.startswith("9255f52b")
    assert shipped["LTX23"].known_weights == ()
    assert shipped["LTX25"].known_weights == ()
    # Every known weight is something the installer downloads.
    raw = json.loads((config.manifest_dir / "40-minimaxh3.json").read_text(encoding="utf-8-sig"))
    downloaded = {f["path"] for item in raw["downloads"] for f in item["files"]}
    assert {w.path for w in roles.values()} <= downloaded


# --------------------------------------------------------------------------- #
# pipeline load + GET /models
# --------------------------------------------------------------------------- #


@pytest.fixture()
def h3_known_client(tmp_path, monkeypatch):
    """LTX 2.3 (active at start) and MiniMax H3 on the mock backend, the H3
    descriptor listing its (small) default DiT under ``known_weights`` —
    written the way ``three_family_client`` writes the H3 files."""
    ltx23 = base_model_descriptor()
    h3 = h3_base_model_descriptor()
    fragment = build_model_layout(tmp_path, [ltx23, h3])
    models_dir = tmp_path / "models"
    write_h3_model_files(models_dir, h3, monkeypatch)
    dit_rel = h3["categories"]["transformer"]["default_file"]
    data = (models_dir / dit_rel).read_bytes()
    h3["known_weights"] = [
        {"path": dit_rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "role": "fl2va"},
        {"path": "MiniMaxH3/Weights/ref2va.safetensors", "size": len(data), "sha256": OTHER_SHA, "role": "ref2va"},
    ]
    write_manifests(tmp_path / "manifests", [ltx23, h3])

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
        yield client


def _base(client, base_id: str) -> dict:
    body = client.get("/api/v1/models").json()
    return next(b for b in body["base_models"] if b["id"] == base_id)


def _load(client, **body) -> None:
    r = client.post("/api/v1/pipeline/load", json=body)
    assert r.status_code == 200, r.text


def test_ltx_load_does_not_hash(client, hash_calls, identify_calls):
    _load(client)
    block = _base(client, "LTX23")
    assert block["transformer_role"] is None
    assert hash_calls == []
    assert identify_calls == []


def test_h3_load_hashes_once_and_models_reports_role(h3_known_client, tmp_path, hash_calls, identify_calls):
    client = h3_known_client
    assert _base(client, "MiniMaxH3")["transformer_role"] is None  # not active yet

    _load(client, base_model="MiniMaxH3")
    assert len(identify_calls) == 1
    assert len(hash_calls) == 1
    assert (tmp_path / CACHE_FILENAME).is_file()

    for _ in range(3):
        block = _base(client, "MiniMaxH3")
        assert block["transformer_role"] == "fl2va"
        assert _base(client, "LTX23")["transformer_role"] is None
    assert len(identify_calls) == 1  # GET /models never identifies

    pm = client.app_context.pipeline_manager
    pm.unload()
    pm.load()  # a second successful load: identified again, cache hit
    assert len(identify_calls) == 2
    assert len(hash_calls) == 1


def test_h3_hash_failure_does_not_fail_the_load(h3_known_client, monkeypatch, caplog):
    def broken(path):
        raise OSError("disk error")

    monkeypatch.setattr(weight_identity, "sha256_file", broken)
    _load(h3_known_client, base_model="MiniMaxH3")
    assert h3_known_client.app_context.pipeline_manager.active_base_model == "MiniMaxH3"
    assert _base(h3_known_client, "MiniMaxH3")["transformer_role"] is None
    assert any("判定できませんでした" in r.getMessage() for r in caplog.records)
