"""S1: LoraRegistry — config + lora_dir scan, kind/scale/thumbnail, resolve.

Torch-free. Synthetic safetensors headers are written in pure Python (8-byte LE
length + JSON header + a few data bytes) so kind detection (``__metadata__``
``reference_downscale_factor``) and the alpha/rank convolution scale
(``ss_network_alpha`` / ``ss_network_dim``, kohya metadata) are exercised without
any real weights. Mirrors the model_registry scan/collision precedent.
"""

from __future__ import annotations

import json
import struct

import pytest

from api.errors import APIError
from config import AppConfig
from services.lora_registry import LoraRegistry


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _write_safetensors(path, metadata=None, tensors=None, data_bytes=16):
    """Write a minimal valid-header safetensors file (header only matters)."""
    header: dict = {}
    if metadata is not None:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    for name, shape in (tensors or {}).items():
        header[name] = {"dtype": "F16", "shape": shape, "data_offsets": [0, 0]}
    blob = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * data_bytes)
    return path


def _config(tmp_path, ic_loras=None, lora_dir=None):
    """AppConfig whose lora_dir is a tmp dir (never the real
    ./models/LTX23/StyleLoRA)."""
    return AppConfig.model_validate(
        {
            "model": {
                "backend": "mock",
                "ic_loras": ic_loras or {},
                "lora_dir": str(lora_dir if lora_dir is not None else tmp_path / "loras"),
            }
        }
    )


# --------------------------------------------------------------------------- #
# directory scan + kind + scale
# --------------------------------------------------------------------------- #

def test_scan_discovers_style_lora(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "Pixar_Toon.safetensors",
        metadata={"ss_network_alpha": "16", "ss_network_dim": "32"},
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    assert "Pixar_Toon" in reg.names()
    info = reg.info("Pixar_Toon")
    assert info.kind == "style"
    assert info.source == "scan"
    assert info.exists
    assert info.scale == pytest.approx(0.5)  # 16 / 32


def test_scan_control_lora_detected_by_metadata(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "some-control.safetensors",
        metadata={"reference_downscale_factor": "2"},
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    assert reg.info("some-control").kind == "control"


def test_scale_defaults_to_one_without_alpha(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(lora_dir / "plain.safetensors", metadata={"foo": "bar"})
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    assert reg.info("plain").scale == 1.0


def test_scale_rank_falls_back_to_lora_a_tensor_shape(tmp_path):
    """No ss_network_dim -> rank read from a lora_A tensor's shape[0]."""
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "ranked.safetensors",
        metadata={"ss_network_alpha": "8"},  # dim absent
        tensors={"diffusion_model.blocks.0.attn.to_q.lora_A.weight": [16, 4096]},
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    assert reg.info("ranked").scale == pytest.approx(0.5)  # 8 / 16


def test_scan_skips_broken_safetensors(tmp_path):
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir(parents=True)
    (lora_dir / "broken.safetensors").write_bytes(b"\x00" * 8)  # header_len == 0
    _write_safetensors(lora_dir / "ok.safetensors", metadata={"x": "1"})
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    assert "ok" in reg.names()
    assert "broken" not in reg.names()  # silently dropped from the scan


def test_scan_ignores_non_safetensors_and_dotfiles(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(lora_dir / "keep.safetensors", metadata={"x": "1"})
    (lora_dir / "notes.txt").write_text("nope", encoding="utf-8")
    (lora_dir / ".hidden.safetensors").write_bytes(b"\x00" * 8)
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    assert reg.names() == ["keep"]


def test_has_thumbnail_sibling_png(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(lora_dir / "with_thumb.safetensors", metadata={"x": "1"})
    (lora_dir / "with_thumb.png").write_bytes(b"\x89PNG\r\n")
    _write_safetensors(lora_dir / "no_thumb.safetensors", metadata={"x": "1"})
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    assert reg.info("with_thumb").has_thumbnail is True
    assert reg.info("no_thumb").has_thumbnail is False


def test_missing_lora_dir_tolerated(tmp_path):
    reg = LoraRegistry(_config(tmp_path, lora_dir=tmp_path / "does-not-exist"))
    assert reg.names() == []


# --------------------------------------------------------------------------- #
# config authority + collisions
# --------------------------------------------------------------------------- #

def test_config_entry_wins_same_file_no_duplicate(tmp_path):
    """A file both registered in config AND present in lora_dir yields ONE entry
    (the config registration), not a duplicate scan entry."""
    lora_dir = tmp_path / "loras"
    weight = _write_safetensors(lora_dir / "my-style.safetensors", metadata={"x": "1"})
    reg = LoraRegistry(
        _config(tmp_path, ic_loras={"my-style": str(weight)}, lora_dir=lora_dir)
    )
    assert reg.names().count("my-style") == 1
    assert reg.info("my-style").source == "config"


def test_scan_name_collision_retreats_to_parent(tmp_path):
    """A scanned file whose stem equals a DIFFERENT config-registered name is
    exposed under a parent-qualified name (model_registry collision rule)."""
    lora_dir = tmp_path / "loras"
    cfg_weight = _write_safetensors(tmp_path / "elsewhere.safetensors", metadata={"x": "1"})
    _write_safetensors(lora_dir / "clash.safetensors", metadata={"x": "1"})
    reg = LoraRegistry(
        _config(tmp_path, ic_loras={"clash": str(cfg_weight)}, lora_dir=lora_dir)
    )
    names = reg.names()
    assert "clash" in names  # the config registration keeps the plain name
    assert "loras__clash" in names  # the scan file retreats to <parent>__<stem>
    assert reg.info("clash").source == "config"
    assert reg.info("loras__clash").source == "scan"


# --------------------------------------------------------------------------- #
# §1-15: reference_downscale_factor + preprocess exposed via as_dict()
# --------------------------------------------------------------------------- #

def test_as_dict_exposes_reference_downscale_factor_and_preprocess(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "union-control.safetensors",
        metadata={"reference_downscale_factor": "2"},
    )
    reg = LoraRegistry(
        _config(
            tmp_path,
            ic_loras={
                "canny-control": {
                    "path": str(lora_dir / "union-control.safetensors"),
                    "preprocess": "canny",
                }
            },
            lora_dir=lora_dir,
        )
    )
    row = reg.info("canny-control").as_dict()
    assert row["preprocess"] == "canny"
    assert row["reference_downscale_factor"] == pytest.approx(2.0)


def test_as_dict_reference_downscale_factor_none_for_style_lora(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(lora_dir / "Pixar_Toon.safetensors", metadata={"x": "1"})
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    row = reg.info("Pixar_Toon").as_dict()
    assert row["preprocess"] == "none"
    assert row["reference_downscale_factor"] is None


def test_as_dict_reference_downscale_factor_one_for_deblur_style_adapter(tmp_path):
    """Deblur needs no preprocessing (preprocess="none", a plain string config
    entry) but its header still carries reference_downscale_factor=1, and that
    value must survive even though it doesn't affect ``kind`` (already
    "control" via the metadata key's mere presence)."""
    lora_dir = tmp_path / "loras"
    weight = _write_safetensors(
        lora_dir / "deblur.safetensors", metadata={"reference_downscale_factor": "1"}
    )
    reg = LoraRegistry(_config(tmp_path, ic_loras={"deblur": str(weight)}, lora_dir=lora_dir))
    row = reg.info("deblur").as_dict()
    assert row["kind"] == "control"
    assert row["preprocess"] == "none"
    assert row["reference_downscale_factor"] == pytest.approx(1.0)


def test_reference_downscale_factor_unparsable_is_none(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "weird.safetensors",
        metadata={"reference_downscale_factor": "not-a-number"},
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    info = reg.info("weird")
    assert info.kind == "control"  # key presence alone still drives kind
    assert info.reference_downscale_factor is None  # but the value itself is unusable


def test_config_preprocess_is_control_even_without_metadata(tmp_path):
    """A config dict entry with preprocess != none is control regardless of what
    (if anything) the header says — and a broken-header config entry is KEPT."""
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir(parents=True)
    weight = lora_dir / "union.safetensors"
    weight.write_bytes(b"\x00" * 8)  # unreadable header
    reg = LoraRegistry(
        _config(
            tmp_path,
            ic_loras={"canny-control": {"path": str(weight), "preprocess": "canny"}},
            lora_dir=lora_dir,
        )
    )
    info = reg.info("canny-control")
    assert info.kind == "control"  # from preprocess, header unreadable
    assert info.scale == 1.0
    assert info.source == "config"
    assert info.preprocess == "canny"


# --------------------------------------------------------------------------- #
# resolve (scale folding + failure modes)
# --------------------------------------------------------------------------- #

def test_resolve_folds_alpha_scale_into_strength(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "scaled.safetensors",
        metadata={"ss_network_alpha": "16", "ss_network_dim": "32"},
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    path, strength, preprocess, audio_strength = reg.resolve("scaled", 1.0)
    assert strength == pytest.approx(0.5)  # 1.0 * (16/32)
    assert preprocess == "none"
    assert path.name == "scaled.safetensors"
    assert audio_strength is None  # not requested -> follows the video axis


def test_resolve_scan_entry_default_scale_unchanged(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(lora_dir / "plain.safetensors", metadata={"x": "1"})
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    _, strength, _, _ = reg.resolve("plain", 0.8)
    assert strength == pytest.approx(0.8)  # scale 1.0


def test_resolve_audio_strength_none_by_default(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(lora_dir / "plain.safetensors", metadata={"x": "1"})
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    resolved = reg.resolve("plain", 1.0)
    assert resolved.audio_strength is None


def test_resolve_audio_strength_zero_with_scaled_entry_stays_zero(tmp_path):
    # audio_strength=0.0 means "skip the audio-side patch entirely" — 0 * scale
    # must stay exactly 0.0, never folded away to None.
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "scaled.safetensors",
        metadata={"ss_network_alpha": "16", "ss_network_dim": "32"},
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    resolved = reg.resolve("scaled", 1.0, audio_strength=0.0)
    assert resolved.audio_strength == 0.0


def test_resolve_audio_strength_scaled_by_entry_scale(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "scaled.safetensors",
        metadata={"ss_network_alpha": "32", "ss_network_dim": "16"},  # scale=2.0
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    resolved = reg.resolve("scaled", 1.0, audio_strength=0.5)
    assert resolved.audio_strength == pytest.approx(1.0)  # 0.5 * 2.0


def test_resolve_pathlike_name_rejected(tmp_path):
    reg = LoraRegistry(_config(tmp_path, lora_dir=tmp_path / "loras"))
    for bad in ("../evil", "sub/x", "a\\b"):
        with pytest.raises(APIError) as ei:
            reg.resolve(bad, 1.0)
        assert ei.value.code == "LORA_NOT_FOUND"


def test_resolve_unknown_name(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(lora_dir / "known.safetensors", metadata={"x": "1"})
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    with pytest.raises(APIError) as ei:
        reg.resolve("nope", 1.0)
    assert ei.value.code == "LORA_NOT_FOUND" and ei.value.status_code == 404


def test_resolve_config_missing_file_fails_loud(tmp_path):
    reg = LoraRegistry(
        _config(
            tmp_path,
            ic_loras={"ghost": str(tmp_path / "gone.safetensors")},
            lora_dir=tmp_path / "loras",
        )
    )
    # Listed (config-authoritative) but resolve fails loud on the missing file.
    assert "ghost" in reg.names()
    assert reg.info("ghost").exists is False
    with pytest.raises(APIError) as ei:
        reg.resolve("ghost", 1.0)
    assert ei.value.code == "LORA_NOT_FOUND"


def test_empty_registry_resolve_message(tmp_path):
    reg = LoraRegistry(_config(tmp_path, lora_dir=tmp_path / "empty"))
    with pytest.raises(APIError) as ei:
        reg.resolve("anything", 1.0)
    assert ei.value.code == "LORA_NOT_FOUND"


def test_preprocess_for_backward_compat(tmp_path):
    lora_dir = tmp_path / "loras"
    scan_weight = _write_safetensors(lora_dir / "styleX.safetensors", metadata={"x": "1"})
    reg = LoraRegistry(
        _config(
            tmp_path,
            ic_loras={"pose": {"path": str(scan_weight), "preprocess": "dwpose"}},
            lora_dir=lora_dir,
        )
    )
    assert reg.preprocess_for("pose") == "dwpose"
    assert reg.preprocess_for("styleX") == "none"  # scan entry
    assert reg.preprocess_for("unknown") == "none"


def test_rescan_picks_up_new_file(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(lora_dir / "first.safetensors", metadata={"x": "1"})
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    assert "late" not in reg.names()
    _write_safetensors(lora_dir / "late.safetensors", metadata={"x": "1"})
    reg.rescan()
    assert "late" in reg.names()
