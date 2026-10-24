"""``h3_lora_format``: the MiniMax H3 LoRA key mapping (torch-free; 台帳 §1-104).

Runs in either venv (no torch needed); the engine runner is the usual one::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_lora_format.py

Pinned: the dotted (with or without ``diffusion_model.``) and kohya underscore
names, both suffix families, the optional alpha, the token refiner, a file that
targets only part of the model, and every "not supported" reason (a key outside
the targets, DoRA / LoHa / LoKr, a key without its partner, a rank mismatch, a
dtype outside BF16 / F16 / F32). The headers of the three trial files are
checked when they are present (header only, no tensor is read).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h3_lora_format as F  # noqa: E402

LORA_DIR = ROOT / "models" / "MiniMaxH3" / "LoRA"


def _dotted(module, a="lora_A", b="lora_B", prefix="diffusion_model.", alpha=True):
    keys = [f"{prefix}{module}.{a}.weight", f"{prefix}{module}.{b}.weight"]
    return keys + ([f"{prefix}{module}.alpha"] if alpha else [])


def _underscore(module, a="lora_down", b="lora_up", alpha=True):
    stem = "lora_unet_" + module.replace(".", "_")
    keys = [f"{stem}.{a}.weight", f"{stem}.{b}.weight"]
    return keys + ([f"{stem}.alpha"] if alpha else [])


MODULES = [f"blocks.{i}.{t}" for i in range(3) for t in F.H3_LORA_TARGETS]
REFINER = [f"token_refiner.blocks.{j}.{t}" for j in range(2) for t in F.H3_LORA_TARGETS]


def test_targets():
    assert F.H3_LORA_TARGETS == ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


@pytest.mark.parametrize("prefix", ["diffusion_model.", ""])
@pytest.mark.parametrize("alpha", [True, False])
def test_dotted_ab(prefix, alpha):
    names = [k for m in MODULES + REFINER for k in _dotted(m, prefix=prefix, alpha=alpha)]
    km = F.map_lora_keys(names)
    assert km.layout == "ab" and km.supported and km.reason is None and km.unmapped == ()
    assert set(km.pairs) == set(MODULES + REFINER)
    p = km.pairs["token_refiner.blocks.1.mlp.fc2"]
    assert p.a == f"{prefix}token_refiner.blocks.1.mlp.fc2.lora_A.weight"
    assert p.b == f"{prefix}token_refiner.blocks.1.mlp.fc2.lora_B.weight"
    assert p.alpha == (f"{prefix}token_refiner.blocks.1.mlp.fc2.alpha" if alpha else None)


def test_dotted_kohya_suffixes():
    names = [k for m in MODULES for k in _dotted(m, a="lora_down", b="lora_up")]
    km = F.map_lora_keys(names)
    assert km.layout == "kohya" and km.supported and set(km.pairs) == set(MODULES)


@pytest.mark.parametrize("a, b", [("lora_down", "lora_up"), ("lora_A", "lora_B")])
def test_kohya_underscore_with_the_token_refiner(a, b):
    names = [k for m in MODULES + REFINER for k in _underscore(m, a=a, b=b)]
    km = F.map_lora_keys(names)
    assert km.layout == "kohya_underscore" and km.supported
    assert set(km.pairs) == set(MODULES + REFINER)
    assert km.pairs["blocks.2.attn.qkv_proj"] == F.LoraPairKeys(
        a=f"lora_unet_blocks_2_attn_qkv_proj.{a}.weight",
        b=f"lora_unet_blocks_2_attn_qkv_proj.{b}.weight",
        alpha="lora_unet_blocks_2_attn_qkv_proj.alpha",
    )
    assert "token_refiner.blocks.0.mlp.fc1" in km.pairs


def test_a_file_for_part_of_the_model_is_supported():
    names = _dotted("blocks.7.mlp.fc1") + _dotted("blocks.0.attn.out_proj")   # no token refiner, 2 modules
    km = F.map_lora_keys(names)
    assert km.supported and set(km.pairs) == {"blocks.7.mlp.fc1", "blocks.0.attn.out_proj"}


def test_mixed_and_none_layouts():
    km = F.map_lora_keys(_dotted("blocks.0.mlp.fc1") + _underscore("blocks.1.mlp.fc2"))
    assert km.layout == "mixed" and km.supported
    km = F.map_lora_keys([])
    assert km.layout == "none" and not km.supported and km.pairs == {} and km.reason


@pytest.mark.parametrize("extra", [
    "diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight",   # AdaLN: outside the targets
    "diffusion_model.blocks.0.norm1.lora_A.weight",
    "diffusion_model.video_patch_proj.lora_B.weight",
    "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",  # a text-encoder LoRA
    "something_else",
])
def test_a_key_outside_the_targets_is_not_supported(extra):
    km = F.map_lora_keys(_dotted("blocks.0.mlp.fc1") + [extra])
    assert not km.supported and extra in km.unmapped and "blocks.0.mlp.fc1" in km.pairs
    assert km.reason and km.reason.startswith("MiniMax H3 の LoRA")


@pytest.mark.parametrize("extra", [
    "diffusion_model.blocks.0.mlp.fc1.dora_scale",
    "lora_unet_blocks_0_mlp_fc1.hada_w1_a",
    "lora_unet_blocks_0_mlp_fc1.lokr_w1",
])
def test_dora_loha_lokr_are_not_supported(extra):
    km = F.map_lora_keys(_dotted("blocks.0.mlp.fc1") + [extra])
    assert not km.supported and "DoRA" in km.reason and extra in km.unmapped


def test_a_key_without_its_partner_is_not_supported():
    names = _dotted("blocks.0.mlp.fc1") + ["diffusion_model.blocks.1.mlp.fc2.lora_A.weight"]
    km = F.map_lora_keys(names)
    assert not km.supported and "blocks.1.mlp.fc2" not in km.pairs
    assert "diffusion_model.blocks.1.mlp.fc2.lora_A.weight" in km.unmapped
    km = F.map_lora_keys(_dotted("blocks.0.mlp.fc1") + ["diffusion_model.blocks.3.mlp.fc1.alpha"])   # alpha alone
    assert not km.supported and km.unmapped == ("diffusion_model.blocks.3.mlp.fc1.alpha",)


def test_the_same_module_twice_is_not_supported():
    km = F.map_lora_keys(_dotted("blocks.0.mlp.fc1") + _underscore("blocks.0.mlp.fc1"))
    assert not km.supported and len(km.unmapped) == 3 and "blocks.0.mlp.fc1" in km.pairs


def test_shapes_rank_mismatch_and_dtypes():
    names = _dotted("blocks.0.mlp.fc1")
    a, b, al = names
    shapes = {a: (4, 64), b: (96, 4), al: ()}
    assert F.map_lora_keys(names, shapes).supported
    km = F.map_lora_keys(names, {**shapes, b: (96, 8)})
    assert not km.supported and "rank" in km.reason
    for dt in ("BF16", "F16", "F32"):
        assert F.map_lora_keys(names, shapes, {a: dt, b: dt, al: "F32"}).supported
    km = F.map_lora_keys(names, shapes, {a: "F8_E4M3", b: "BF16", al: "F32"})
    assert not km.supported and "F8_E4M3" in km.reason


def test_module_is_torch_free():
    """Imported in a fresh, isolated interpreter (``-I``): no torch, no third-party module."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import h3_lora_format; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'numpy', 'safetensors')); "
        "assert not bad, bad; print('ok')"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


# the three trial files (header only; skipped when absent) ----------------------

_TRIAL = [
    # file, layout, pairs, with token refiner
    ("minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors", "ab", 208, True),
    ("3dsrx_minimax_h3.safetensors", "kohya_underscore", 200, False),
    ("camera_motion_h3_lora_v1_3000_pruned.safetensors", "ab", 208, True),
]


@pytest.mark.parametrize("name, layout, pairs, refiner", _TRIAL)
def test_trial_file_headers(name, layout, pairs, refiner):
    import sft_quant_format

    path = LORA_DIR / name
    if not path.is_file():
        pytest.skip(f"{path} is not present")
    t = sft_quant_format.read_header(path).tensors
    km = F.map_lora_keys(t, {k: v.shape for k, v in t.items()}, {k: v.dtype for k, v in t.items()})
    assert km.supported, km.reason
    assert (km.layout, len(km.pairs), km.unmapped) == (layout, pairs, ())
    assert ("token_refiner.blocks.1.mlp.fc2" in km.pairs) == refiner
    assert all(f"blocks.{i}.{tg}" in km.pairs for i in range(50) for tg in F.H3_LORA_TARGETS)
