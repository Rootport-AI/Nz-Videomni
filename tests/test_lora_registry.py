"""S1: LoraRegistry — config + lora_dir scan, kind/layout/thumbnail, resolve.

Torch-free. Synthetic safetensors headers are written in pure Python (8-byte LE
length + JSON header + a few data bytes) so kind detection (``__metadata__``
``reference_downscale_factor``) and weight-layout detection (the tensor KEY
NAMES) are exercised without any real weights. Mirrors the model_registry
scan/collision precedent.

§3-108 (2026-09-02) replaced the ``scale`` field with ``layout``: the alpha/rank
metadata multiplier is gone (musubi-tuner already bakes alpha into the converted
weights and copies ``ss_network_alpha`` over verbatim, so multiplying again was
a double application), and an adapter whose key layout the engine loader cannot
read is now refused at ``resolve()`` with 422 ``LORA_FORMAT_UNSUPPORTED`` rather
than silently producing a LoRA-free video.
"""

from __future__ import annotations

import json
import struct

import pytest

from api.errors import APIError
from config import PROJECT_ROOT, AppConfig
from services.lora_registry import LoraRegistry, make_scope, shared_scope


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

#: One A/B key, enough to make a fixture look like a readable LoRA. Since
#: §3-108 a header with no recognisable weight keys is an UNSUPPORTED layout, so
#: every fixture that is not itself about layout detection needs one of these —
#: otherwise resolve() would 422 in tests about kind/thumbnail/collisions.
_DEFAULT_TENSORS = {
    "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": [4, 8],
    "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": [8, 4],
}


def _write_safetensors(path, metadata=None, tensors=None, data_bytes=16):
    """Write a minimal valid-header safetensors file (header only matters).

    ``tensors`` defaults to :data:`_DEFAULT_TENSORS` (an A/B pair); pass an
    explicit dict — ``{}`` included — to control the layout under test.
    """
    if tensors is None:
        tensors = _DEFAULT_TENSORS
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
# directory scan + kind + layout
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
    # The kohya alpha/dim metadata is a fossil of the pre-conversion training
    # run and no longer influences anything — only the KEY NAMES are read.
    assert info.layout == "ab"


def test_scan_control_lora_detected_by_metadata(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "some-control.safetensors",
        metadata={"reference_downscale_factor": "2"},
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    assert reg.info("some-control").kind == "control"


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
    # No header -> nothing to judge a layout on -> the pre-§3-108 "accept it,
    # let the engine speak" behaviour is kept. Every zero-byte dummy adapter in
    # the API test suites depends on this default.
    assert info.layout == "ab"
    assert info.source == "config"
    assert info.preprocess == "canny"


# --------------------------------------------------------------------------- #
# resolve (strength pass-through + failure modes)
# --------------------------------------------------------------------------- #

def test_resolve_passes_strength_through_despite_alpha_metadata(tmp_path):
    """§3-108 B案 (ComfyUI semantics): ``ss_network_alpha``/``ss_network_dim``
    in the header are NOT folded into the strength any more. musubi-tuner's A/B
    conversion already baked alpha into the weights and then copied the training
    metadata across verbatim, so the old ``* (alpha/dim)`` was a second, fossil
    application — ``Pixar_Toon`` (16/32) ran at half the requested strength.
    What the caller asks for is what the engine gets."""
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "with-alpha-meta.safetensors",
        metadata={"ss_network_alpha": "16", "ss_network_dim": "32"},
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    path, strength, preprocess, audio_strength = reg.resolve("with-alpha-meta", 0.8)
    assert strength == pytest.approx(0.8)  # NOT 0.4
    assert preprocess == "none"
    assert path.name == "with-alpha-meta.safetensors"
    assert audio_strength is None  # not requested -> follows the video axis


def test_resolve_audio_strength_none_by_default(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(lora_dir / "plain.safetensors", metadata={"x": "1"})
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    resolved = reg.resolve("plain", 1.0)
    assert resolved.audio_strength is None


def test_resolve_audio_strength_zero_stays_zero(tmp_path):
    # audio_strength=0.0 means "skip the audio-side patch entirely" — it must
    # stay exactly 0.0, never folded away to None (which would mean "follow the
    # video axis", the opposite instruction).
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "with-alpha-meta.safetensors",
        metadata={"ss_network_alpha": "16", "ss_network_dim": "32"},
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    resolved = reg.resolve("with-alpha-meta", 1.0, audio_strength=0.0)
    assert resolved.audio_strength == 0.0


def test_resolve_audio_strength_passed_through(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "with-alpha-meta.safetensors",
        metadata={"ss_network_alpha": "32", "ss_network_dim": "16"},  # old scale=2.0
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    resolved = reg.resolve("with-alpha-meta", 1.0, audio_strength=0.5)
    assert resolved.audio_strength == pytest.approx(0.5)  # NOT 1.0


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


# --------------------------------------------------------------------------- #
# §3-108: weight-layout detection + the 422 it feeds
#
# Only the tensor KEY NAMES are read (shapes and metadata are irrelevant here),
# because that is all a torch-free header parse can see — and all the engine
# loader itself keys off.
# --------------------------------------------------------------------------- #

_PREFIX = "diffusion_model.transformer_blocks.0.attn1.to_q"


def _layout_of(tmp_path, tensors, name="probe"):
    lora_dir = tmp_path / "loras"
    _write_safetensors(lora_dir / f"{name}.safetensors", tensors=tensors)
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    return reg.info(name).layout


def test_layout_ab(tmp_path):
    assert _layout_of(
        tmp_path,
        {
            f"{_PREFIX}.lora_A.weight": [16, 4096],
            f"{_PREFIX}.lora_B.weight": [4096, 16],
        },
    ) == "ab"


def test_layout_kohya(tmp_path):
    """The shape §3-108 unblocks: sd-scripts down/up/alpha with DOTTED paths
    (LTX2.3-MysticXXX, SynthPussy_01_rank32)."""
    assert _layout_of(
        tmp_path,
        {
            f"{_PREFIX}.lora_down.weight": [16, 4096],
            f"{_PREFIX}.lora_up.weight": [4096, 16],
            f"{_PREFIX}.alpha": [],
        },
    ) == "kohya"


def test_layout_unsupported_dora(tmp_path):
    layout = _layout_of(tmp_path, {f"{_PREFIX}.dora_scale": [4096]})
    assert layout.startswith("unsupported:")
    assert "DoRA" in layout


def test_layout_unsupported_loha(tmp_path):
    layout = _layout_of(
        tmp_path,
        {
            "lora_unet_transformer_blocks_0_attn1_to_q.hada_w1_a": [16, 4096],
            "lora_unet_transformer_blocks_0_attn1_to_q.hada_w1_b": [4096, 16],
        },
    )
    assert layout.startswith("unsupported:")
    assert "LoHa" in layout


def test_layout_unsupported_lokr(tmp_path):
    layout = _layout_of(
        tmp_path,
        {
            "lora_unet_transformer_blocks_0_attn1_to_q.lokr_w1": [16, 16],
            "lora_unet_transformer_blocks_0_attn1_to_q.lokr_w2": [256, 256],
        },
    )
    assert layout.startswith("unsupported:")
    assert "LoKr" in layout


def test_layout_unsupported_unknown_keys(tmp_path):
    """Neither A/B nor down/up nor a recognised exotic factorisation — including
    the degenerate 'valid header, no tensors at all' file."""
    layout = _layout_of(tmp_path, {"some.random.tensor": [1]})
    assert layout.startswith("unsupported:")
    assert "some.random.tensor" in layout  # the detail names what it did see
    assert _layout_of(tmp_path, {}, name="empty").startswith("unsupported:")


def test_layout_unsupported_kohya_underscore_keys(tmp_path):
    """Underscore-joined kohya names resolve to no ``named_modules()`` path, so
    they would attach to 0 Linears — exactly the silent no-op §3-108 ends. They
    are refused rather than accepted as "kohya"."""
    layout = _layout_of(
        tmp_path,
        {
            "lora_unet_transformer_blocks_0_attn1_to_q.lora_down.weight": [16, 4096],
            "lora_unet_transformer_blocks_0_attn1_to_q.lora_up.weight": [4096, 16],
        },
    )
    assert layout.startswith("unsupported:")
    assert "underscores" in layout


def test_layout_order_dora_beats_ab(tmp_path):
    """A DoRA file also carries ordinary A/B keys; reading it as plain A/B would
    silently drop the magnitude vector, so ``.dora_scale`` is checked first."""
    assert _layout_of(
        tmp_path,
        {
            f"{_PREFIX}.lora_A.weight": [16, 4096],
            f"{_PREFIX}.lora_B.weight": [4096, 16],
            f"{_PREFIX}.dora_scale": [4096],
        },
    ).startswith("unsupported:")


def test_layout_order_ab_beats_kohya(tmp_path):
    """One A/B key makes the whole file an A/B file — the same file-unit guard
    ``load_ic_lora_pairs`` applies, so a mixed file's delta is never doubled."""
    assert _layout_of(
        tmp_path,
        {
            f"{_PREFIX}.lora_A.weight": [16, 4096],
            f"{_PREFIX}.lora_B.weight": [4096, 16],
            "diffusion_model.transformer_blocks.1.attn1.to_q.lora_down.weight": [16, 4096],
            "diffusion_model.transformer_blocks.1.attn1.to_q.lora_up.weight": [4096, 16],
        },
    ) == "ab"


def test_resolve_unsupported_layout_is_422(tmp_path):
    """The whole point of the layout field: a file the engine loader cannot read
    is refused at resolve() — which BOTH endpoint validation loops (single and
    chain) already call per requested adapter — instead of running the job and
    returning a video indistinguishable from the LoRA-free one."""
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "loha-style.safetensors",
        tensors={"lora_unet_blocks_0_attn.hada_w1_a": [16, 4096]},
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    assert "loha-style" in reg.names()  # still LISTED — only using it is refused
    with pytest.raises(APIError) as ei:
        reg.resolve("loha-style", 1.0)
    assert ei.value.code == "LORA_FORMAT_UNSUPPORTED"
    assert ei.value.status_code == 422
    assert "LoHa" in ei.value.detail


def test_as_dict_key_set_unchanged_by_layout(tmp_path):
    """``layout`` is server-internal: GET /loras never shows it (the former
    ``scale`` was never exposed either). §1-104 added exactly the family's
    verdict, ``supported`` / ``unsupported_reason``."""
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "kohya-style.safetensors",
        tensors={
            f"{_PREFIX}.lora_down.weight": [16, 4096],
            f"{_PREFIX}.lora_up.weight": [4096, 16],
            f"{_PREFIX}.alpha": [],
        },
    )
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    assert reg.info("kohya-style").layout == "kohya"
    assert set(reg.info("kohya-style").as_dict()) == {
        "name",
        "kind",
        "has_thumbnail",
        "exists",
        "source",
        "preprocess",
        "reference_downscale_factor",
        "supported",
        "unsupported_reason",
    }
    assert reg.info("kohya-style").as_dict()["supported"] is True
    assert reg.info("kohya-style").as_dict()["unsupported_reason"] is None


# --------------------------------------------------------------------------- #
# §1-104: family-scoped listing (LoraScope / scope_provider)
# --------------------------------------------------------------------------- #

#: An H3 kohya-underscore pair (the Style 3dsrx file's naming): unsupported on
#: LTX (no dotted module path), supported on H3.
_H3_UNDERSCORE = {
    "lora_unet_blocks_0_attn_qkv_proj.lora_down.weight": [16, 5376],
    "lora_unet_blocks_0_attn_qkv_proj.lora_up.weight": [21504, 16],
    "lora_unet_blocks_0_attn_qkv_proj.alpha": [],
}
#: An AdaLN key no H3 target maps -> unsupported on H3 (display only).
_H3_FOREIGN = {
    **_H3_UNDERSCORE,
    "diffusion_model.blocks.0.adaLN_modulation.1.lora_A.weight": [16, 5376],
    "diffusion_model.blocks.0.adaLN_modulation.1.lora_B.weight": [32256, 16],
}


def _h3_verdict():
    pytest.importorskip("h3_lora_format")
    from services.engines.h3.adapter import LORA_VERDICT  # noqa: PLC0415

    return LORA_VERDICT


class _Switch:
    """A scope provider whose answer the test flips (a base-model switch)."""

    def __init__(self, scope):
        self.scope = scope

    def __call__(self):
        return self.scope


def test_default_scope_is_todays_config_lora_dir(tmp_path):
    cfg = _config(tmp_path)
    reg = LoraRegistry(cfg)
    assert reg.scope == shared_scope(cfg)
    assert reg.scope.include_config is True and reg.scope.verdict is None
    assert reg.lora_dir_display == (tmp_path / "loras").as_posix()  # outside the project


def test_make_scope_display_is_project_relative_posix():
    scope = make_scope("h3", PROJECT_ROOT / "models" / "MiniMaxH3" / "LoRA", include_config=False)
    assert scope.lora_dir_display == "models/MiniMaxH3/LoRA"


def test_scope_switch_rescans_before_resolve(tmp_path):
    ltx_dir, h3_dir = tmp_path / "ltx", tmp_path / "h3"
    _write_safetensors(ltx_dir / "ltx-style.safetensors")
    _write_safetensors(h3_dir / "h3-style.safetensors", tensors=_H3_UNDERSCORE)
    cfg = _config(tmp_path, lora_dir=ltx_dir)
    switch = _Switch(shared_scope(cfg, "ltx"))
    reg = LoraRegistry(cfg, scope_provider=switch)
    assert reg.names() == ["ltx-style"]
    switch.scope = make_scope("h3", h3_dir, include_config=False, verdict=lambda h: (True, None))
    # No explicit rescan(): the very next resolve sees the new family's listing.
    assert reg.resolve("h3-style", 1.0).path == h3_dir / "h3-style.safetensors"
    with pytest.raises(APIError) as ei:
        reg.resolve("ltx-style", 1.0)
    assert ei.value.code == "LORA_NOT_FOUND"
    assert reg.lora_dir_display == h3_dir.as_posix()


def test_h3_scope_excludes_config_ic_loras(tmp_path):
    control = _write_safetensors(tmp_path / "ic" / "pose.safetensors")
    h3_dir = tmp_path / "h3"
    _write_safetensors(h3_dir / "h3-style.safetensors", tensors=_H3_UNDERSCORE)
    cfg = _config(tmp_path, ic_loras={"pose": {"path": str(control), "preprocess": "dwpose"}})
    reg = LoraRegistry(
        cfg, scope_provider=lambda: make_scope("h3", h3_dir, include_config=False, verdict=_h3_verdict())
    )
    assert reg.names() == ["h3-style"]
    # preprocess_for reads the scoped entry: the config's dwpose is not in scope.
    assert reg.preprocess_for("pose") == "none"
    assert reg.preprocess_for("h3-style") == "none"


def test_preprocess_for_is_scoped(tmp_path):
    control = _write_safetensors(tmp_path / "ic" / "pose.safetensors")
    cfg = _config(tmp_path, ic_loras={"pose": {"path": str(control), "preprocess": "dwpose"}})
    switch = _Switch(shared_scope(cfg, "ltx"))
    reg = LoraRegistry(cfg, scope_provider=switch)
    assert reg.preprocess_for("pose") == "dwpose"
    switch.scope = make_scope("h3", tmp_path / "h3", include_config=False)
    assert reg.preprocess_for("pose") == "none"


def test_h3_underscore_kohya_is_supported(tmp_path):
    h3_dir = tmp_path / "h3"
    _write_safetensors(h3_dir / "style.safetensors", tensors=_H3_UNDERSCORE)
    reg = LoraRegistry(
        _config(tmp_path),
        scope_provider=lambda: make_scope("h3", h3_dir, include_config=False, verdict=_h3_verdict()),
    )
    entry = reg.info("style")
    assert entry.supported is True and entry.unsupported_reason is None
    # layout stays _detect_layout's (server-internal), the verdict decides.
    assert entry.layout.startswith("unsupported:")
    assert reg.resolve("style", 0.8).strength == 0.8


def test_h3_unsupported_file_is_listed_and_resolves_without_422(tmp_path):
    h3_dir = tmp_path / "h3"
    _write_safetensors(h3_dir / "foreign.safetensors", tensors=_H3_FOREIGN)
    reg = LoraRegistry(
        _config(tmp_path),
        scope_provider=lambda: make_scope("h3", h3_dir, include_config=False, verdict=_h3_verdict()),
    )
    row = reg.info("foreign").as_dict()
    assert row["supported"] is False
    assert row["unsupported_reason"]
    # Display only (裁定 2): the job may still run.
    assert reg.resolve("foreign", 1.0).path == h3_dir / "foreign.safetensors"


def test_h3_malformed_header_entry_is_listed_unsupported(tmp_path):
    """A verdict that cannot read a header (here ``"shape": 5``) lists the file
    as unsupported instead of failing the scan."""
    h3_dir = tmp_path / "h3"
    path = _write_safetensors(h3_dir / "bad.safetensors", tensors={})
    header = {
        "lora_unet_blocks_0_attn_qkv_proj.lora_down.weight": {"dtype": "F32", "shape": 5, "data_offsets": [0, 0]},
        "lora_unet_blocks_0_attn_qkv_proj.lora_up.weight": {"dtype": "F32", "shape": [21504, 16], "data_offsets": [0, 0]},
    }
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(16))

    def exploding(_header):
        raise TypeError("boom")

    reg = LoraRegistry(
        _config(tmp_path),
        scope_provider=lambda: make_scope("h3", h3_dir, include_config=False, verdict=_h3_verdict()),
    )
    assert reg.info("bad").as_dict()["supported"] is False
    reg = LoraRegistry(
        _config(tmp_path),
        scope_provider=lambda: make_scope("h3", h3_dir, include_config=False, verdict=exploding),
    )
    row = reg.info("bad").as_dict()
    assert row["supported"] is False
    assert row["unsupported_reason"] == "header unreadable: TypeError"


def test_empty_scope_404_names_the_directory(tmp_path):
    h3_dir = tmp_path / "h3"
    reg = LoraRegistry(_config(tmp_path), scope_provider=lambda: make_scope("h3", h3_dir, include_config=False))
    with pytest.raises(APIError) as ei:
        reg.resolve("x", 1.0)
    assert ei.value.detail == f"no LoRA files found in {h3_dir.as_posix()}"


def test_ltx_underscore_kohya_still_422(tmp_path):
    """The same underscore file in the LTX scope (no verdict): 422 as before."""
    lora_dir = tmp_path / "loras"
    _write_safetensors(lora_dir / "style.safetensors", tensors=_H3_UNDERSCORE)
    reg = LoraRegistry(_config(tmp_path, lora_dir=lora_dir))
    assert reg.info("style").supported is False
    assert "underscores" in reg.info("style").unsupported_reason
    with pytest.raises(APIError) as ei:
        reg.resolve("style", 1.0)
    assert ei.value.code == "LORA_FORMAT_UNSUPPORTED"
