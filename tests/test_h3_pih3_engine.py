"""pi-H3 in the MiniMax H3 engine, stage 1 (VERIFICATION_LOG §223; plan "pi-H3 を製品に実装する計画").

Runs in ``.venv-engine-h3`` through the engine-venv runner, without a GPU::

    CUDA_VISIBLE_DEVICES=-1 .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_pih3_engine.py

What is pinned:
  * the sigma lists against the closed form (2 / 4 steps, video / audio), other counts refused;
  * ``h3_pih3_format`` on the real pi-H3 files and base weights (skipped when absent)
    and on small synthetic headers (the facts it reports);
  * ``pi_h3.apply_adapter`` on a small 50-block DiT whose blocks 46-49 are int8, float, fp8 or
    w4a8, and a block mixing layer kinds running off and in VRAM 16GB mode;
  * ``pi_h3.initial_state`` (mask values 0, 1 and in between; the 0 / 1 case equals the
    experiment's whole-state scaling);
  * ``engine_h3.dwd``: the window-by-window layer against the experiment's all-rows layer (kept
    here only), the chunks with the 4-block margin against one pass, the per-block uint8
    writes, the frame counts, and a head read from a synthetic file;
  * ``H3DiT.forward``'s capture box (off and VRAM 16GB mode);
  * the pipeline: the resident DiT's tag, an off job touching none of the new code, the records;
  * the worker's ``pi_h3`` key.
The int8 matmul is faked in float64 (``test_h3_engine._fake_int8``); the synthetic markers are
``convrot: false`` for that reason (the real files are ConvRot, checked on their headers).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("comfy_kitchen")

import h3_format  # noqa: E402
from test_h3_engine import (  # noqa: E402
    FFN,
    HD,
    HID,
    LAT,
    TEXT,
    _fake_int8,
    _fake_quant_kernels,
    _fp8_lin,
    _tiny_dit_tensors,
    _tiny_forward_inputs,
    _tiny_vram16_dit,
    _w4a8_lin,
    _write_sft,
)

import h3_pih3_format as PF  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PI_DIR = ROOT / "models" / "MiniMaxH3" / "VAE" / "pi-H3"
PI_REAL = {2: PI_DIR / "pi_h3_2step_dwd_adapter_int8_convrot.safetensors",
           4: PI_DIR / "pi_h3_4step_dwd_adapter_int8_convrot.safetensors"}
WEIGHTS = ROOT / "models" / "MiniMaxH3" / "Weights"
BASES = (WEIGHTS / "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
         WEIGHTS / "Sulphur-3_Ref2VA-hybrid_mid_int8_convrot_keepB0-1-49.safetensors")

#: A small DWD head for the synthetic files and the CPU checks (the real constants of the
#: windows / blocks / shifts, tiny widths, a 2-px patch).
TINY_HEAD = PF.DWDHeadDims(in_channels=HID, t_dim=8, dim=16, heads=2, head_dim=8, ffn_mult=4, num_layers=8,
                           expansion=(2, 2), patch=2, patch_t=4, out_ch=3, num_register=4)


def _mark(fmt="int8_tensorwise", convrot=False):
    return torch.frombuffer(bytearray(json.dumps({"format": fmt, "convrot": convrot}).encode()), dtype=torch.uint8)


def _pi_tensors(seed: int, *, inner: int, head=TINY_HEAD, rank: int = 4) -> dict:
    """A synthetic pi-H3 file for the 50-block tiny DiT (``inner`` = heads x 128)."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=1.0, dtype=torch.float16):
        return (torch.randn(shape, generator=g) * scale).to(dtype)

    shapes = {"attn.out_proj": (HID, inner), "mlp.fc1": (2 * FFN, HID), "mlp.fc2": (HID, FFN)}
    t: dict = {}
    for i in PF.LORA_BLOCKS:
        for leaf, (o, k) in shapes.items():
            t[f"blocks.{i}.{leaf}.lora_A.weight"] = rnd(rank, k, scale=0.05)
            t[f"blocks.{i}.{leaf}.lora_B.weight"] = rnd(o, rank, scale=0.05)
    for i in PF.REPLACED_BLOCKS:
        for leaf, (o, k) in shapes.items():
            t[f"blocks.{i}.{leaf}.weight"] = torch.randint(-127, 128, (o, k), generator=g, dtype=torch.int8)
            t[f"blocks.{i}.{leaf}.weight_scale"] = torch.rand(o, 1, generator=g) / 100 + 1e-4
            t[f"blocks.{i}.{leaf}.comfy_quant"] = _mark()
    for i in range(PF.NUM_BLOCKS):
        t[f"blocks.{i}.adaln_proj.linear.weight"] = rnd(18 * HID, 8)
        if i in PF.REPLACED_BLOCKS:
            t[f"blocks.{i}.adaln_proj.linear.bias"] = rnd(18 * HID)
    t.update({
        "final_layer.norm.weight": rnd(HID, scale=0.3) + 1,
        "final_layer.adaln_proj.linear.weight": rnd(2 * HID, 8),
        "final_layer.adaln_proj.linear.bias": rnd(2 * HID),
        "final_layer.video_out.weight": rnd(LAT * 4, HID),
        "final_layer.video_out.bias": rnd(LAT * 4),
        "final_layer.audio_out.weight": rnd(32, HID),
        "final_layer.audio_out.bias": rnd(32),
    })
    for name, (kind, shape) in PF.dwd_tensor_table(head).items():
        if kind == "I8":
            t[name] = torch.randint(-127, 128, shape, generator=g, dtype=torch.int8)
        elif kind == "F32":
            t[name] = torch.rand(shape, generator=g) / 100 + 1e-3
        elif name.endswith(("norm1.weight", "norm2.weight")):
            t[name] = rnd(*shape, scale=0.1) + 1
        elif name.endswith(("scale1", "scale2")):
            t[name] = rnd(*shape, scale=0.1) + 0.5
        else:
            t[name] = rnd(*shape, scale=1 / math.sqrt(max(shape[-1], 1)))
    for layer in PF.dwd_int8_layers(head):
        t[f"{PF.PREFIX_REF}{layer}.comfy_quant"] = _mark()
    return t


@pytest.fixture
def tiny_head(monkeypatch):
    monkeypatch.setattr(PF, "DWD_HEAD", TINY_HEAD)
    return TINY_HEAD


def _files(tmp_path, *, heads=1, bf16_blocks=(), int8_adaln=False, seed=1, name="pi.safetensors"):
    """``(dit_path, pi_path)``: a 50-block tiny DiT file and a pi-H3 file that fits it."""
    dit_path = tmp_path / f"dit_h{heads}_{'-'.join(map(str, bf16_blocks))}_{int(int8_adaln)}.safetensors"
    if not dit_path.exists():
        _write_sft(dit_path, _tiny_dit_tensors(layers=50, refiner=1, grid=1025, rank=8, heads=heads,
                                               bf16_blocks=bf16_blocks, int8_adaln=int8_adaln))
    pi_path = tmp_path / name
    _write_sft(pi_path, _pi_tensors(seed, inner=heads * HD))
    return dit_path, pi_path


def _build(dit_path):
    from engine_h3 import pipeline_h3 as P

    pipe = P.H3Pipeline(P.H3Paths(str(dit_path), "te", "vv", "av", "tok"), P.H3LoadOptions(blocks_on_gpu=1),
                        device="cpu", progress=lambda *a: None, log=lambda m: None)
    dit, _cfg = pipe._build_dit({})
    return dit


def _file_tensors(path) -> dict:
    from engine.sft_quant.sft_reader import read_tensors

    import sft_quant_format

    header = sft_quant_format.read_header(path)
    return dict(read_tensors(str(path), list(header.tensors), header=header))


# --------------------------------------------------------------------------- #
# sigma lists
# --------------------------------------------------------------------------- #

def _closed_form(n: int, s: float) -> list[float]:
    max_t = {2: 0.93, 4: 0.96}[n]
    f = 0.6
    end = max_t * f / (n - 1 + f)
    raw = [max_t + (end - max_t) * k / (n - 1) for k in range(n)]
    return [s * r / (1 + (s - 1) * r) for r in raw] + [0.0]


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("shift", [12.0, 3.0])
def test_pi_h3_sigma_lists_match_the_closed_form(n, shift):
    from engine_h3.scheduler import pi_h3_sigma_schedule

    got = pi_h3_sigma_schedule(n, shift)
    assert got.dtype == torch.float32 and got.shape == (n + 1,) and float(got[-1]) == 0.0
    want = _closed_form(n, shift)
    assert max(abs(float(a) - b) for a, b in zip(got, want)) < 1e-6
    printed = {(4, 12.0): [0.99654, 0.964451, 0.899297, 0.695652], (4, 3.0): [0.986301, 0.871508, 0.690647, 0.363636],
               (2, 12.0): [0.993767, 0.865340], (2, 3.0): [0.975524, 0.616348]}[(n, shift)]   # research notes §18.4
    assert all(abs(float(a) - b) < 5e-6 for a, b in zip(got, printed))


@pytest.mark.parametrize("n", [1, 3, 5, 8, 20])
def test_pi_h3_sigma_list_refuses_other_step_counts(n):
    from engine_h3.scheduler import pi_h3_sigma_schedule

    with pytest.raises(ValueError, match="2 ステップと 4 ステップだけ"):
        pi_h3_sigma_schedule(n, 12.0)


def test_pi_h3_constants():
    assert PF.PI_H3_CHOICES == {"4steps": 4, "2steps": 2} and PF.PI_H3_STEPS == (2, 4)
    assert (PF.PI_H3_SHIFT_VIDEO, PF.PI_H3_SHIFT_AUDIO) == (12.0, 3.0)
    assert PF.choice_for_steps(4) == "4steps" and PF.choice_for_steps(2) == "2steps"
    for bad in (3, True, "4", 4.0):
        with pytest.raises(ValueError):
            PF.choice_for_steps(bad)


# --------------------------------------------------------------------------- #
# the torch-free header check
# --------------------------------------------------------------------------- #

def test_format_module_is_torch_free():
    import subprocess
    import sys

    code = ("import sys; sys.path.insert(0, sys.argv[1]); import h3_pih3_format; "
            "print('torch' in sys.modules, 'engine_h3' in sys.modules)")
    out = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False"


@pytest.mark.skipif(not all(p.is_file() for p in PI_REAL.values()), reason="the pi-H3 files are not in models/")
def test_real_pi_h3_files_fit_the_layout_and_the_meta_head():
    from engine_h3 import dwd

    for steps, path in PI_REAL.items():
        assert PF.file_problems(path) == [], steps
        info = PF.inspect_file(path)
        assert info.keys.counts() == PF.EXPECTED_COUNTS and info.keys.other == []
        assert {(s.format, s.convrot, s.convrot_groupsize) for s in info.quant_specs.values()} == {
            ("int8_tensorwise", True, 256)}
        assert len(info.dwd_specs) == 32
        with torch.device("meta"):
            head = dwd.DWDHead(make=dwd.int8_linear(info.dwd_specs))
        sd = head.state_dict()
        table = PF.dwd_tensor_table()
        assert set(sd) == set(table)
        assert all(tuple(sd[k].shape) == shape for k, (_kind, shape) in table.items())


@pytest.mark.skipif(not all(p.is_file() for p in (*PI_REAL.values(), *BASES)),
                    reason="the pi-H3 files or the base weights are not in models/")
def test_real_pi_h3_files_fit_both_base_weights():
    for path in PI_REAL.values():
        for base in BASES:
            assert PF.model_problems(path, base) == [], (path.name, base.name)


def test_synthetic_pi_h3_file_fits_and_each_misfit_is_stated(tmp_path, tiny_head):
    dit_path, pi_path = _files(tmp_path)
    assert PF.file_problems(pi_path) == [] and PF.model_problems(pi_path, dit_path) == []
    base = _pi_tensors(1, inner=HD)

    def problems(mutate, *, against=dit_path):
        t = dict(base)
        mutate(t)
        p = tmp_path / "bad.safetensors"
        _write_sft(p, t)
        return PF.model_problems(p, against)

    got = problems(lambda t: t.pop("blocks.3.mlp.fc1.lora_B.weight"))
    assert any("LoRA の組が 137 組" in s for s in got) and any("片方しか無い" in s for s in got)
    got = problems(lambda t: t.update({"blocks.47.mlp.fc1.weight": torch.zeros(2 * FFN, HID + 1, dtype=torch.int8)}))
    assert got == [f"置き換えの層 blocks.47.mlp.fc1 の重み [{2 * FFN}, {HID + 1}] が本体の層 [{2 * FFN}, {HID}] と合いません"]
    got = problems(lambda t: t.update({"blocks.46.attn.out_proj.comfy_quant": _mark("float8_e4m3fn")}))
    assert len(got) == 1 and "blocks.46.attn.out_proj.comfy_quant を int8 の量子化の印として読めません" in got[0]
    got = problems(lambda t: t.pop("dwd_pixel_refiner.blocks.7.ff.w2.weight_scale"))
    assert got == ["DWD のテンソルが 1 個足りません（dwd_pixel_refiner.blocks.7.ff.w2.weight_scale）"]
    got = problems(lambda t: t.update({"something.else": torch.zeros(1)}))
    assert got == ["pi-H3 のどの部分にも当たらないキーが 1 個あります（something.else）"]
    small = tmp_path / "dit3.safetensors"
    _write_sft(small, _tiny_dit_tensors(layers=3, refiner=1, grid=1025, rank=8))
    assert problems(lambda t: None, against=small) == ["本体のブロックが 3 個です（pi-H3 は 50 ブロックの本体に当てるものです）"]
    wide = tmp_path / "dit_wide.safetensors"
    _write_sft(wide, _tiny_dit_tensors(layers=50, refiner=1, grid=1025, rank=8, heads=2))
    got = problems(lambda t: None, against=wide)
    assert any(s.startswith("LoRA blocks.0.attn.out_proj の A") for s in got)
    assert PF.file_problems(tmp_path / "nope.safetensors") == [f"pi-H3 のファイルが見つかりません（{tmp_path / 'nope.safetensors'}）"]
    no_norm = dict(base)
    no_norm.pop("final_layer.norm.weight")
    bad_final = tmp_path / "bad_final.safetensors"
    _write_sft(bad_final, no_norm)
    with pytest.raises(PF.PiH3FormatError, match="中身が想定と違います.*出力層のテンソルが 6 個です"):
        PF.inspect_file(bad_final)


# --------------------------------------------------------------------------- #
# apply_adapter
# --------------------------------------------------------------------------- #

def test_apply_adapter_on_int8_layers(tmp_path, tiny_head):
    from engine_h3 import pi_h3
    from engine_h3.quant_linear import H3Int8Linear

    dit_path, pi_path = _files(tmp_path)
    dit = _build(dit_path)
    before = {n: m for n, m in dit.named_modules()}
    assert pi_h3.applied_tag(dit) is None
    rep = pi_h3.apply_adapter(dit, pi_path, "4steps")
    t = _file_tensors(pi_path)
    assert pi_h3.applied_tag(dit) == "4steps"
    for i in PF.REPLACED_BLOCKS:
        for leaf in PF.LEAVES:
            name = f"blocks.{i}.{leaf}"
            mod = dit.get_submodule(name)
            assert mod is before[name] and isinstance(mod, H3Int8Linear)      # values replaced in place
            assert torch.equal(mod.weight, t[name + ".weight"]) and torch.equal(mod.weight_scale, t[name + ".weight_scale"])
            assert mod.spec == h3_format.QuantSpec("int8_tensorwise", False, 256)
    for i in range(50):
        lin = dit.blocks[i].adaln_proj.linear
        assert lin.weight.dtype == torch.float32 and torch.equal(lin.weight, t[f"blocks.{i}.adaln_proj.linear.weight"].float())
        if i in PF.REPLACED_BLOCKS:
            assert torch.equal(lin.bias, t[f"blocks.{i}.adaln_proj.linear.bias"].float())
    fl = dit.final_layer
    assert torch.equal(fl.norm.weight, t["final_layer.norm.weight"].to(torch.bfloat16))
    assert torch.equal(fl.video_out.weight, t["final_layer.video_out.weight"].float())
    assert rep["counts"] == {"quant_layers": 12, "adaln_weights": 50, "adaln_biases": 4, "final_tensors": 7}
    assert [r["tensor"] for r in rep["rounding"]] == ["final_layer.norm.weight"]      # F16 -> BF16 only
    assert rep["rounding"][0]["changed_elements"] > 0 and rep["rounding"][0]["max_abs_change"] > 0


def test_apply_adapter_on_float_layers_and_int8_adaln(tmp_path, tiny_head):
    """Block 49 kept in BF16 (Sulphur-3's shape): the int8 values dequantized and cast to BF16;
    the int8 AdaLN of the other blocks becomes a float32 nn.Linear (pi-H3's bias on 46-48,
    the base's elsewhere)."""
    from engine.sft_quant.dequant import dequantize

    from engine_h3 import pi_h3
    from engine_h3.quant_linear import H3Int8Linear

    dit_path, pi_path = _files(tmp_path, bf16_blocks=(49,), int8_adaln=True)
    dit = _build(dit_path)
    base_bias_10 = dit.blocks[10].adaln_proj.linear.bias.detach().clone()
    assert isinstance(dit.blocks[10].adaln_proj.linear, H3Int8Linear)
    rep = pi_h3.apply_adapter(dit, pi_path, "2steps")
    t = _file_tensors(pi_path)
    for leaf in PF.LEAVES:
        name = f"blocks.49.{leaf}"
        mod = dit.get_submodule(name)
        assert type(mod) is torch.nn.Linear and mod.weight.dtype == torch.bfloat16
        want = dequantize("int8", t[name + ".weight"], {"weight_scale": t[name + ".weight_scale"]}, torch.float32)
        assert torch.equal(mod.weight, want.to(torch.bfloat16))
        assert rep["quant_layers"][name]["into"] == "nn.Linear bfloat16"
        assert isinstance(dit.get_submodule(f"blocks.48.{leaf}"), H3Int8Linear)
    for i in (10, 47):
        lin = dit.blocks[i].adaln_proj.linear
        assert type(lin) is torch.nn.Linear and lin.weight.dtype == torch.float32
        assert torch.equal(lin.weight, t[f"blocks.{i}.adaln_proj.linear.weight"].float())
    assert torch.equal(dit.blocks[10].adaln_proj.linear.bias, base_bias_10.float())
    assert torch.equal(dit.blocks[47].adaln_proj.linear.bias, t["blocks.47.adaln_proj.linear.bias"].float())
    assert rep["adaln"] == {"H3Int8Linear -> float32 nn.Linear": 49, "set into nn.Linear float32": 1}
    assert {r["tensor"] for r in rep["rounding"]} >= {f"quant dequantized (f32) -> blocks.49.{leaf}" for leaf in PF.LEAVES}


@pytest.mark.parametrize("fmt", ["fp8", "w4a8"])
def test_apply_adapter_replaces_fp8_and_w4a8_layers_by_int8(tmp_path, tiny_head, fmt):
    from engine_h3 import pi_h3
    from engine_h3.quant_linear import H3Int8Linear

    dit_path, pi_path = _files(tmp_path)
    dit = _build(dit_path)
    make = (lambda k, n, s: _fp8_lin(k, n, seed=s)) if fmt == "fp8" else (lambda k, n, s: _w4a8_lin(k, n, seed=s))
    for i in PF.REPLACED_BLOCKS:
        for j, leaf in enumerate(PF.LEAVES):
            old = dit.get_submodule(f"blocks.{i}.{leaf}")
            parent, _, attr = f"blocks.{i}.{leaf}".rpartition(".")
            setattr(dit.get_submodule(parent), attr, make(old.in_features, old.out_features, 10 * i + j))
    rep = pi_h3.apply_adapter(dit, pi_path, "4steps")
    t = _file_tensors(pi_path)
    for i in PF.REPLACED_BLOCKS:
        for leaf in PF.LEAVES:
            name = f"blocks.{i}.{leaf}"
            mod = dit.get_submodule(name)
            assert type(mod) is H3Int8Linear and mod.bias is None and not mod.weight.requires_grad
            assert torch.equal(mod.weight, t[name + ".weight"]) and torch.equal(mod.weight_scale, t[name + ".weight_scale"])
            assert rep["quant_layers"][name]["into"].startswith("H3Int8Linear (replaces H3")


@pytest.mark.parametrize("qkv_kind", ["fp8", "bf16"])
def test_a_block_mixing_layer_kinds_runs_off_and_in_vram16(monkeypatch, tmp_path, tiny_head, qkv_kind):
    """Block 47's qkv fp8 (or a plain BF16 nn.Linear) next to pi-H3's int8 out_proj / fc1 / fc2:
    the whole forward runs, and VRAM 16GB mode gives the off path's numbers bit for bit."""
    from engine_h3 import pi_h3
    from engine_h3.lowvram import Vram16Plan

    _fake_quant_kernels(monkeypatch)
    dit_path, pi_path = _files(tmp_path, heads=4)
    dit = _build(dit_path)
    qkv = dit.blocks[47].attn.qkv_proj
    if qkv_kind == "fp8":
        dit.blocks[47].attn.qkv_proj = _fp8_lin(qkv.in_features, qkv.out_features, seed=5)
    else:
        lin = torch.nn.Linear(qkv.in_features, qkv.out_features, bias=False, dtype=torch.bfloat16)
        lin.weight.data.copy_(torch.randn(lin.weight.shape, generator=torch.Generator().manual_seed(5)) * 0.05)
        lin.requires_grad_(False)
        dit.blocks[47].attn.qkv_proj = lin
    pi_h3.apply_adapter(dit, pi_path, "4steps")
    tokens, kw = _tiny_forward_inputs(dit)
    with torch.no_grad():
        want = dit(tokens, **kw)
        got = dit(None, tokens_box=[tokens.clone()], vram16=Vram16Plan(head_group=3, row_chunk=16), **kw)
    for g, w in zip(got, want):
        assert torch.isfinite(w).all() and torch.equal(g, w)


# --------------------------------------------------------------------------- #
# the initial state
# --------------------------------------------------------------------------- #

def test_initial_state_rows_by_mask_value():
    from engine_h3 import denoise_mask as DM
    from engine_h3 import pi_h3

    g = torch.Generator().manual_seed(3)
    clean_a, noise_a = torch.randn(2, 32, 6, generator=g), torch.randn(2, 32, 6, generator=g)
    am = DM.audio_mask(clean_a, [0.0, 1.0, 0.5, 0.25, 1.0, 0.0])
    s = 0.975524
    out = pi_h3.initial_state(noise_a, am, s)
    for j, m in enumerate([0.0, 1.0, 0.5, 0.25, 1.0, 0.0]):
        want = noise_a[:, :, j] * s + ((1 - s) * clean_a[:, :, j] if 0 < m < 1 else 0)
        assert torch.equal(out[:, :, j], want), j
    clean_v, noise_v = torch.randn(1, 24, 3, 4, 4, generator=g), torch.randn(1, 24, 3, 4, 4, generator=g)
    vm = DM.video_mask(clean_v, [0.0, 0.5, 1.0], grid=4, noise=noise_v)
    out = pi_h3.initial_state(noise_v, vm, 0.9)
    assert torch.equal(out[:, :, 0], noise_v[:, :, 0] * 0.9) and torch.equal(out[:, :, 2], noise_v[:, :, 2] * 0.9)
    assert torch.equal(out[:, :, 1], noise_v[:, :, 1] * 0.9 + (1 - 0.9) * clean_v[:, :, 1])
    assert torch.equal(pi_h3.initial_state(noise_v, None, 0.9), noise_v * 0.9)


def test_initial_state_with_masks_of_0_and_1_is_the_experiments_scaling():
    """The experiment scaled the whole state by sigma_0 at the first DiT input (VERIFICATION_LOG
    §223.4). With mask values 0 / 1 only, ``initial_state`` gives the same DiT inputs at every
    step and the same final latents (a fake mixing velocity, the real mask helpers)."""
    from engine_h3 import denoise_mask as DM
    from engine_h3 import pi_h3
    from engine_h3.scheduler import euler_step, pi_h3_sigma_schedule

    g = torch.Generator().manual_seed(7)
    clean, noise = torch.randn((1, 24, 6, 4, 4), generator=g), torch.randn((1, 24, 6, 4, 4), generator=g)
    vm = DM.video_mask(clean, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0], grid=4, noise=noise, inject_clean=1.0)
    sig = pi_h3_sigma_schedule(2, 12.0)

    def run(product: bool):
        x = noise.clone()
        if product:
            x = pi_h3.initial_state(x, vm, float(sig[0]))
        inputs = []
        for i in range(2):
            if not product and i == 0:
                x.mul_(float(sig[0]))           # the experiment: in place, at the first DiT input
            x_in = DM.dit_input(x, vm)
            inputs.append(x_in.clone())
            v = -x_in + x_in.mean(dim=(2, 3, 4), keepdim=True)
            v = DM.state_velocity(v, x, vm, float(sig[i]))
            x = euler_step(x, v, float(sig[i]), float(sig[i + 1]))
        return DM.finish(x, vm), inputs

    a, ia = run(True)
    b, ib = run(False)
    assert all(torch.equal(p, q) for p, q in zip(ia, ib)) and torch.equal(a, b)


# --------------------------------------------------------------------------- #
# DWD
# --------------------------------------------------------------------------- #

def _random_head(dtype=torch.float64, seed=20261122, cfg=TINY_HEAD):
    """Random-weight float head on the CPU (the experiment's chunking self-test)."""
    from engine_h3 import dwd

    g = torch.Generator().manual_seed(seed)
    head = dwd.DWDHead(cfg, make=dwd.float_linear(dtype), dtype=dtype)
    with torch.no_grad():
        for p in head.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=dtype) / math.sqrt(max(p.shape[-1], 1)))
        for blk in head.dwd_pixel_refiner.blocks:
            for p in (blk.norm1.weight, blk.norm2.weight):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g, dtype=dtype))
            for p in (blk.scale1, blk.scale2):
                p.copy_(0.5 + 0.1 * torch.randn(p.shape, generator=g, dtype=dtype))
    return head


def _inputs(t, gh, gw, dtype=torch.float64, seed=5, cfg=TINY_HEAD):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(t * gh * gw, cfg.in_channels, generator=g, dtype=dtype)
    t_emb = torch.randn(3, cfg.t_dim, generator=g, dtype=torch.float32)
    time_index = torch.randint(0, 3, (t * gh * gw,), generator=g)
    return hidden, t_emb, time_index


# The experiment's all-rows-at-once layer and refiner (outputs/h3-pih3-spike-2026-11-22/pih3_dwd.py),
# kept here only, as the reference of the window-by-window form.
def _old_layer(blk, X, geo, shifted, cfg, row_chunk, win_batch, rope_dev):
    import torch.nn.functional as F

    from engine_h3 import dwd

    R, D = X.shape
    nh, hd = cfg.heads, cfg.head_dim
    dev, dt = X.device, X.dtype
    q, k, v = (torch.empty(R, nh, hd, dtype=dt, device=dev) for _ in range(3))
    for r0, r1 in dwd._chunks(R, row_chunk):
        y = dwd._linear(blk.attn.to_qkv, dwd._rms(X[r0:r1], blk.norm1.weight)).view(r1 - r0, nh, 3 * hd)
        qq, kk, vv = y.chunk(3, dim=-1)
        q[r0:r1], k[r0:r1], v[r0:r1] = dwd._rms(qq, None), dwd._rms(kk, None), vv
    cos, sin = rope_dev[shifted]
    for r0, r1 in dwd._chunks(geo.N, row_chunk):
        q[r0:r1] = dwd._apply_rope(q[r0:r1], cos[r0:r1], sin[r0:r1])
        k[r0:r1] = dwd._apply_rope(k[r0:r1], cos[r0:r1], sin[r0:r1])
    out = torch.empty(R, nh, hd, dtype=dt, device=dev)
    for qi_all, ki_all in geo.windows[shifted]:
        for c0, c1 in dwd._chunks(qi_all.shape[0], win_batch):
            qi, ki = qi_all[c0:c1].to(dev), ki_all[c0:c1].to(dev)
            o = F.scaled_dot_product_attention(q[qi].transpose(1, 2), k[ki].transpose(1, 2), v[ki].transpose(1, 2))
            out[qi] = o.transpose(1, 2)
    for gq, gk in geo.glob:
        o = F.scaled_dot_product_attention(q[gq].transpose(0, 1)[None], k[gk].transpose(0, 1)[None],
                                           v[gk].transpose(0, 1)[None])
        out[gq] = o[0].transpose(0, 1)
    del q, k, v
    s1, s2 = blk.scale1.to(dt), blk.scale2.to(dt)
    for r0, r1 in dwd._chunks(R, row_chunk):
        X[r0:r1] = torch.addcmul(X[r0:r1], dwd._linear(blk.attn.to_out, out[r0:r1].reshape(r1 - r0, D)), s1)
    for r0, r1 in dwd._chunks(R, row_chunk):
        gate, val = dwd._linear(blk.ff.w1, dwd._rms(X[r0:r1], blk.norm2.weight)).chunk(2, dim=-1)
        X[r0:r1] = torch.addcmul(X[r0:r1], dwd._linear(blk.ff.w2, F.silu(gate) * val), s2)


def _old_run_refiner(head, feats, grid, *, keep_blocks=None, row_chunk=8192, win_batch=16):
    import torch.nn.functional as F

    from engine_h3 import dwd

    cfg, ref = head.cfg, head.dwd_pixel_refiner
    t, gh, gw = grid
    eh, ew = cfg.expansion
    H, W, D, P = gh * eh, gw * ew, cfg.dim, cfg.patch
    dev, dt = feats.device, feats.dtype
    geo = dwd.build_geometry(t, H, W, cfg.rope_dim, cfg.num_register + 1)
    X = torch.empty(geo.R, D, dtype=dt, device=dev)
    hw, HW = gh * gw, H * W
    step = max(1, row_chunk // hw)
    for t0, t1 in dwd._chunks(t, step):
        y = dwd._linear(ref.feature_expand, feats[t0 * hw:t1 * hw])
        X[t0 * HW:t1 * HW] = y.view(t1 - t0, gh, gw, eh, ew, D).permute(0, 1, 3, 2, 4, 5).reshape(-1, D)
    g = torch.cat([ref.register_tokens[0].to(dt), torch.zeros(1, D, dtype=dt, device=dev)])
    X[geo.N:] = g.repeat(geo.nb, 1)
    rope_dev = {k: (c.to(dev), s.to(dev)) for k, (c, s) in geo.rope.items()}
    for li, blk in enumerate(ref.blocks):
        _old_layer(blk, X, geo, li % 2 == 1, cfg, row_chunk, win_batch, rope_dev)
    b0, b1 = (0, geo.nb) if keep_blocks is None else keep_blocks
    oc, pt, ft = cfg.out_ch, cfg.patch_t, cfg.first_t
    nw, bw = ref.norm_out.weight.to(dt), ref.norm_out.bias.to(dt)
    odt = dt if dt in (torch.float32, torch.float64) else torch.float32
    pieces = []
    for b in range(b0, b1):
        r0, r1 = geo.block_rows[b]
        n = (r1 - r0) // HW
        y = F.layer_norm(X[r0:r1], (D,), nw, bw, dwd.EPS)
        first = dwd._linear(ref.first_frame_proj_out, y[:HW]).view(H, W, oc, ft, P, P)
        frames = [first.permute(3, 2, 0, 4, 1, 5).reshape(ft, oc, H * P, W * P).to("cpu", odt)]
        if n > 1:
            rest = dwd._linear(ref.proj_out, y[HW:]).view(n - 1, H, W, oc, pt, P, P)
            frames.append(rest.permute(0, 4, 3, 1, 5, 2, 6).reshape((n - 1) * pt, oc, H * P, W * P).to("cpu", odt))
        pieces.append(torch.cat(frames))
    return torch.cat(pieces)


def _old_to_uint8(pixels, frame_chunk=16):
    from engine_h3 import dwd

    mean = torch.tensor(dwd.COLOR_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(dwd.COLOR_STD, dtype=torch.float32).view(1, 3, 1, 1)
    out = torch.empty((pixels.shape[0], pixels.shape[2], pixels.shape[3], 3), dtype=torch.uint8)
    for f0 in range(0, pixels.shape[0], frame_chunk):
        p = pixels[f0:f0 + frame_chunk].float().cpu()
        c = p * std + mean
        x = ((c + 1.0) * 0.5).clamp_(0.0, 1.0).permute(0, 2, 3, 1)
        out[f0:f0 + frame_chunk] = (x * 255.0).round().to(torch.uint8)
    return out


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("win_batch, row_chunk", [(16, 8192), (3, 50)])
def test_window_by_window_layer_matches_the_all_rows_layer(dtype, win_batch, row_chunk):
    """The window-by-window attention gives the experiment's numbers (shifted windows over
    two blocks, 2 x 2 spatial windows, several global groups; row pieces smaller than a window)."""
    from engine_h3 import dwd

    head = _random_head(dtype)
    t, gh, gw = 12, 12, 12
    hidden, t_emb, tidx = _inputs(t, gh, gw, dtype)
    feats = dwd.normalize_features(head, hidden, t_emb, tidx)
    want = _old_run_refiner(head, feats, (t, gh, gw), row_chunk=row_chunk, win_batch=win_batch)
    got = dwd.run_refiner(head, feats.clone(), (t, gh, gw), row_chunk=row_chunk, win_batch=win_batch)
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    assert got.shape == want.shape == (dwd.frames_for_latents(t), 3, gh * 4, gw * 4)
    assert float((got - want).abs().max()) <= tol * float(want.abs().max())


def _chunked_float(head, hidden, t_emb, tidx, grid):
    """``dwd.decode``'s chunk loop with the float pixels kept (to compare before the codec)."""
    from engine_h3 import dwd

    t, gh, gw = grid
    hw = gh * gw
    pieces = []
    for ch in dwd.plan_chunks(t, gh, gw):
        (k0, k1), (c0, c1) = ch["keep"], ch["context"]
        l0, l1 = c0 * dwd.BLOCK, min(c1 * dwd.BLOCK, t)
        feats = dwd.normalize_features(head, hidden[l0 * hw:l1 * hw], t_emb, tidx[l0 * hw:l1 * hw])
        dwd.refine(head, feats, (l1 - l0, gh, gw), lambda b, px: pieces.append(px), keep_blocks=(k0 - c0, k1 - c0))
    return torch.cat(pieces)


@pytest.mark.parametrize("max_blocks, content", [(9, 1), (13, 5), (99, 14)])
def test_chunks_with_the_4_block_margin_match_one_pass(monkeypatch, max_blocks, content):
    """67 latents = 14 native blocks on a 12 x 12 DiT grid (float64): chunks of 1 or 5 kept
    blocks with 4 blocks of context give the whole pass's pixels; the uint8 written per block
    is the whole pass converted at once (the experiment's conversion)."""
    from engine_h3 import dwd

    t, gh, gw = 67, 12, 12
    monkeypatch.setattr(dwd, "CHUNK_MAX_ROWS", max_blocks * dwd.BLOCK * gh * gw)
    plan = dwd.plan_chunks(t, gh, gw)
    assert all(c["keep"][1] - c["keep"][0] == content for c in plan[:-1])
    head = _random_head()
    hidden, t_emb, tidx = _inputs(t, gh, gw)
    whole = dwd.run_refiner(head, dwd.normalize_features(head, hidden, t_emb, tidx), (t, gh, gw))
    got = _chunked_float(head, hidden, t_emb, tidx, (t, gh, gw))
    assert got.shape == whole.shape and whole.shape[0] == dwd.frames_for_latents(t) == 226
    assert float((got - whole).abs().max()) <= 1e-9
    frames, records = dwd.decode(head, dwd.DWDFeatures(hidden, t_emb, tidx, (t, gh, gw)), "cpu")
    assert len(records) == len(plan) and sum(r["frames"] for r in records) == 226
    assert frames.dtype == torch.uint8 and torch.equal(frames, _old_to_uint8(whole))


def test_plan_chunks_comes_from_the_shape_alone():
    from engine_h3 import dwd

    assert dwd.MARGIN_BLOCKS == 4
    # 1280 x 768, 1920 x 1088, 2560 x 1472: the longest one-pass clip and the split, from the constant
    for gh, gw in ((24, 40), (34, 60), (46, 80)):
        max_blocks = max(1, dwd.CHUNK_MAX_ROWS // (5 * gh * gw))
        content = max(1, max_blocks - 2 * dwd.MARGIN_BLOCKS)
        one = max(max_blocks, content + 2 * dwd.MARGIN_BLOCKS)
        assert dwd.plan_chunks(5 * one - 3, gh, gw) == [{"keep": (0, one), "context": (0, one)}], (gh, gw)
        plan = dwd.plan_chunks(5 * (one + 1) - 3, gh, gw)
        assert len(plan) > 1 and all(c["keep"][1] - c["keep"][0] == content for c in plan[:-1]), (gh, gw)
        assert max(c["context"][1] - c["context"][0] for c in plan) <= one
    assert len(dwd.plan_chunks(22, 24, 40)) == 1
    for t, gh, gw in ((67, 34, 60), (172, 24, 40), (377, 46, 80)):
        plan = dwd.plan_chunks(t, gh, gw)
        assert plan == dwd.plan_chunks(t, gh, gw)                      # deterministic
        nb = -(-t // 5)
        kept = [b for c in plan for b in range(*c["keep"])]
        assert kept == list(range(nb))                                  # every block once, in order
        for c in plan:
            (k0, k1), (c0, c1) = c["keep"], c["context"]
            assert k1 - k0 >= 1 and c0 == max(0, k0 - 4) and c1 == min(nb, k1 + 4)


def test_plan_chunks_does_not_split_a_clip_no_longer_than_a_split_chunk(monkeypatch):
    """The rule on a cap of 6 native blocks at 1920 x 1088 (34 x 60 = 2,040 DiT rows per latent):
    a split chunk keeps 1 and holds up to 1 + 2 x 4 = 9; so 8 and 9 blocks are one pass, 10 and
    more are split into chunks of 1 kept block (each at most 9 blocks with its context)."""
    from engine_h3 import dwd

    gh, gw = 34, 60
    monkeypatch.setattr(dwd, "CHUNK_MAX_ROWS", 6 * 5 * gh * gw + 1)
    for nb in (1, 6, 8, 9):
        assert dwd.plan_chunks(5 * nb - 3 if nb > 1 else 2, gh, gw) == [{"keep": (0, nb), "context": (0, nb)}], nb
    for nb in (10, 14):
        plan = dwd.plan_chunks(5 * nb - 3, gh, gw)
        assert len(plan) == nb and all(c["keep"][1] - c["keep"][0] == 1 for c in plan)
        assert max(c["context"][1] - c["context"][0] for c in plan) == 9


def test_refine_lets_the_features_go_before_the_layers():
    """``refine`` keeps no reference to the features once ``feature_expand`` has read them (no
    ``no_grad`` decorator holding its arguments); ``run_refiner`` records no autograd graph."""
    import weakref

    from engine_h3 import dwd

    head = _random_head(torch.float32)
    t, gh, gw = 7, 4, 4
    hidden, t_emb, tidx = _inputs(t, gh, gw, torch.float32)
    feats = dwd.normalize_features(head, hidden, t_emb, tidx)
    ref = weakref.ref(feats)
    box = [feats]
    del feats
    seen = []
    real_layer = dwd._layer

    def spy(*a, **k):
        seen.append(ref() is None)
        return real_layer(*a, **k)

    import unittest.mock as mock

    with mock.patch.object(dwd, "_layer", spy), torch.no_grad():
        dwd.refine(head, box.pop(), (t, gh, gw), lambda b, px: None)
    assert seen and all(seen)
    got = dwd.run_refiner(head, dwd.normalize_features(head, hidden, t_emb, tidx), (t, gh, gw))
    assert not got.requires_grad and got.grad_fn is None


@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_frame_counts(n):
    from engine_h3 import dwd

    assert dwd.frames_for_latents(5 * n + 2) == 17 * n + 5


def test_head_from_a_synthetic_file_decodes_whole_frames(monkeypatch, tmp_path, tiny_head):
    from engine_h3 import dwd

    _fake_int8(monkeypatch)
    _dit_path, pi_path = _files(tmp_path)
    head, rep = dwd.load_head(pi_path, "cpu")
    assert rep["loaded_tensors"] == len(PF.dwd_tensor_table(TINY_HEAD)) and rep["int8_layers"] == 32
    dec = dwd.DWDDecoder.from_file(pi_path, device="cpu")
    hidden, t_emb, tidx = _inputs(7, 2, 2, dtype=torch.bfloat16)
    frames = dec.decode(dwd.DWDFeatures(hidden, t_emb, tidx, (7, 2, 2)))
    assert frames.dtype == torch.uint8 and tuple(frames.shape) == (22, 8, 8, 3)
    assert len(dec.chunks) == 1 and dec.chunks[0]["frames"] == 22


# --------------------------------------------------------------------------- #
# the capture box
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("vram16", [False, True])
def test_forward_capture_is_the_hidden_state_before_final_rows(monkeypatch, tmp_path, vram16):
    from engine_h3.lowvram import Vram16Plan

    _fake_int8(monkeypatch)
    dit = _tiny_vram16_dit(tmp_path, layers=3)
    tokens, kw = _tiny_forward_inputs(dit)
    plan = Vram16Plan(head_group=3, row_chunk=16) if vram16 else None
    seen = {}
    real = dit.final_rows

    def spy(x, t_emb, time_index, layout, vram16=None):
        seen.update(hidden=x[layout.video].clone(), t_emb=t_emb.clone(), time_index=time_index[layout.video].clone())
        return real(x, t_emb, time_index, layout, vram16=vram16)

    with torch.no_grad():
        plain = dit(tokens.clone(), vram16=plan, **kw)
        box: dict = {}
        monkeypatch.setattr(dit, "final_rows", spy)
        got = dit(tokens.clone(), vram16=plan, capture=box, **kw)
    for a, b in zip(got, plain):
        assert torch.equal(a, b)
    assert set(box) == {"hidden", "t_emb", "time_index"}
    assert box["hidden"].device.type == "cpu" and box["hidden"].dtype == torch.bfloat16
    assert torch.equal(box["hidden"], seen["hidden"]) and torch.equal(box["t_emb"], seen["t_emb"].float())
    assert torch.equal(box["time_index"], seen["time_index"])
    assert box["hidden"].shape == (28, HID)


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

def _pi_job_pipe(monkeypatch, tmp_path):
    """The REAL generate / _denoise / _decode on the 50-block tiny DiT with two synthetic pi-H3
    files; returns ``(pipe, rec, job)``: ``rec`` gathers the builds, the video decoder's
    latents and the frames written to the mp4."""
    from engine_h3 import audio_vae as AV
    from engine_h3 import media_out as MO
    from engine_h3 import pipeline_h3 as P
    from engine_h3 import video_vae as VV

    _fake_int8(monkeypatch)
    monkeypatch.setattr(PF, "DWD_HEAD", TINY_HEAD)
    dit_path, pi4 = _files(tmp_path, seed=4, name="pi4.safetensors")
    _dit_path, pi2 = _files(tmp_path, seed=2, name="pi2.safetensors")
    _header, cfg = h3_format.inspect_h3_dit(str(dit_path))
    pipe = P.H3Pipeline(P.H3Paths(str(dit_path), "te", "vv", "av", "tok"), P.H3LoadOptions(blocks_on_gpu=1),
                        device="cpu", progress=lambda *a: None, log=lambda m: None)
    pipe._dit_cfg = cfg
    rec = {"builds": 0, "latents": [], "frames": [], "decoders": 0}
    real_build = pipe._build_dit

    def build(entry):
        rec["builds"] += 1
        return real_build(entry)

    monkeypatch.setattr(pipe, "_build_dit", build)
    monkeypatch.setattr(pipe, "_encode", lambda prompt, phases, **kw: (
        torch.randn(5, TEXT, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16),
        torch.ones(5, dtype=torch.long)))

    class FakeDecoder(torch.nn.Module):
        def decode(self, z, tiled=True):
            rec["latents"].append(z.clone())
            return torch.zeros((22, 64, 64, 3), dtype=torch.uint8)

    class FakeAudio(torch.nn.Module):
        def decode(self, a):
            return torch.zeros((2, 800 * a.shape[-1]))

    def decoder(cls, *a, **k):
        rec["decoders"] += 1
        return FakeDecoder()

    monkeypatch.setattr(VV.H3VideoDecoder, "from_file", classmethod(decoder))
    monkeypatch.setattr(AV.H3AudioVAE, "from_file", classmethod(lambda cls, *a, **k: FakeAudio()))
    monkeypatch.setattr(MO, "write_mp4", lambda path, frames, **k: rec["frames"].append(frames.clone()) or 1)

    def job(steps=None, keep=True, vram16=False):
        pi = None if steps is None else P.H3PiH3Input(steps=steps, path=str(pi4 if steps == 4 else pi2))
        req = P.H3GenerateRequest(prompt="p", seed=7, width=64, height=64, num_frames=22, output_path="o.mp4",
                                  num_inference_steps=3 if steps is None else steps, keep_resident=keep,
                                  vram16_mode=vram16, pi_h3=pi,
                                  shift_video=12.0, shift_audio=3.0)
        return pipe.generate(req)

    return pipe, rec, job


def test_pi_h3_job_records_and_never_touches_the_video_decoder(monkeypatch, tmp_path):
    from engine_h3.scheduler import pi_h3_sigma_schedule

    pipe, rec, job = _pi_job_pipe(monkeypatch, tmp_path)
    done = job(4, keep=False)
    assert rec["decoders"] == 0 and rec["latents"] == []
    assert done["pi_h3_used"] == "4steps" and done["num_inference_steps"] == 4
    assert list(done)[-2:] == ["pi_h3_used", "h3"] and "loras" not in done["h3"]
    r = done["h3"]["pi_h3"]
    assert (r["choice"], r["steps"], r["file"], r["dit_source"]) == ("4steps", 4, "pi4.safetensors", "file")
    assert r["adapter"]["counts"]["quant_layers"] == 12 and r["lora"]["pairs_applied"] == 138
    assert r["sigmas"]["video"] == [float(s) for s in pi_h3_sigma_schedule(4, 12.0)]
    assert r["initial_state"]["partial_rows"] == {"video": 0, "audio": 0}
    assert r["dwd"]["chunks"] == 1
    decode = done["h3"]["phases"]["decode"]
    assert decode["vae_source"] == "dwd" and decode["frames"] == 22
    frames = rec["frames"][-1]
    assert frames.dtype == torch.uint8 and tuple(frames.shape) == (22, 8, 8, 3)
    assert pipe._resident is None


@pytest.mark.parametrize("vram16", [False, True])
def test_resident_dit_is_rebuilt_when_its_pi_h3_tag_differs(monkeypatch, tmp_path, vram16):
    """keep_resident on: off -> 4 -> 4 -> 2 -> off. The DiT is built again whenever the kept
    one carries another pi-H3 (or none), reused when it carries the same, and every job gives
    what it gives alone."""
    _pipe, rec, job = _pi_job_pipe(monkeypatch, tmp_path)
    builds = []
    d = [job(None, vram16=vram16)]
    builds.append(rec["builds"])
    for steps in (4, 4, 2, None):
        d.append(job(steps, vram16=vram16))
        builds.append(rec["builds"])
    assert builds == [1, 2, 2, 3, 4]
    assert [x["h3"]["phases"]["denoise"]["dit_source"] for x in d] == ["file", "file", "resident", "file", "file"]
    assert d[1]["h3"]["pi_h3"]["resident_dropped"] is None and d[3]["h3"]["pi_h3"]["resident_dropped"] == "4steps"
    assert "pi_h3" not in d[4]["h3"] and "pi_h3_used" not in d[4]
    off0, off1 = rec["latents"]
    assert torch.equal(off0, off1)                         # off after pi-H3 == off alone
    f4a, f4b, f2 = rec["frames"][1], rec["frames"][2], rec["frames"][3]
    assert torch.equal(f4a, f4b)                           # reused == built
    (tmp_path / "fresh").mkdir()
    _pipe2, rec2, job2 = _pi_job_pipe(monkeypatch, tmp_path / "fresh")
    job2(4, keep=False, vram16=vram16)
    job2(2, keep=False, vram16=vram16)
    assert torch.equal(rec2["frames"][0], f4a) and torch.equal(rec2["frames"][1], f2)


def test_off_job_calls_none_of_the_pi_h3_code(monkeypatch, tmp_path):
    from engine_h3 import dwd, pi_h3
    from engine_h3 import pipeline_h3 as P
    from engine_h3.dit import H3DiT

    pipe, rec, job = _pi_job_pipe(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise AssertionError("an off job reached the pi-H3 code")

    for owner, name in ((pi_h3, "apply_adapter"), (pi_h3, "initial_state"), (P, "pi_h3_sigma_schedule"),
                        (PF, "file_problems"), (PF, "model_problems"), (PF, "inspect_file")):
        monkeypatch.setattr(owner, name, boom)
    monkeypatch.setattr(dwd.DWDDecoder, "from_file", classmethod(boom))
    real_forward = H3DiT.forward

    def forward(self, *a, **k):
        assert "capture" not in k
        return real_forward(self, *a, **k)

    monkeypatch.setattr(H3DiT, "forward", forward)
    done = job(None, keep=False)
    assert "pi_h3_used" not in done and "pi_h3" not in done["h3"]
    assert rec["decoders"] == 1 and done["h3"]["phases"]["decode"]["vae_source"] == "file"


def test_request_validation_pi_h3(monkeypatch, tmp_path, tiny_head):
    from engine_h3.pipeline_h3 import H3GenerateRequest, H3PiH3Input, validate_request

    dit_path, pi_path = _files(tmp_path)
    base = dict(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4")
    validate_request(H3GenerateRequest(**base, num_inference_steps=4, pi_h3=H3PiH3Input(4, str(pi_path))),
                     dit_path=str(dit_path))
    with pytest.raises(ValueError, match="4steps の要求のステップ数が 8 です（4 のはず）"):
        validate_request(H3GenerateRequest(**base, num_inference_steps=8, pi_h3=H3PiH3Input(4, str(pi_path))))
    with pytest.raises(ValueError, match="pi-H3 のステップ数は 2・4 のどれか"):
        validate_request(H3GenerateRequest(**base, num_inference_steps=3, pi_h3=H3PiH3Input(3, str(pi_path))))
    with pytest.raises(ValueError, match="pi-H3 のファイルが見つかりません"):
        validate_request(H3GenerateRequest(**base, num_inference_steps=2, pi_h3=H3PiH3Input(2, str(tmp_path / "x"))))
    small = tmp_path / "dit3.safetensors"
    _write_sft(small, _tiny_dit_tensors(layers=3, refiner=1, grid=1025, rank=8))
    with pytest.raises(ValueError, match="本体の重み（dit3.safetensors）の照合で、合わない所が 1 か所"):
        validate_request(H3GenerateRequest(**base, num_inference_steps=4, pi_h3=H3PiH3Input(4, str(pi_path))),
                         dit_path=str(small))


# --------------------------------------------------------------------------- #
# the worker
# --------------------------------------------------------------------------- #

def test_worker_pi_h3_key(monkeypatch):
    from engine_h3 import worker
    from engine_h3.pipeline_h3 import H3PiH3Input

    msg = {"op": "generate", "prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124,
           "output_path": "o.mp4"}
    assert worker._generate_request(msg).pi_h3 is None
    assert worker._generate_request({**msg, "pi_h3": None}).pi_h3 is None
    req = worker._generate_request({**msg, "num_inference_steps": 4, "pi_h3": {"steps": 4, "path": "C:/m/pi4.safetensors"}})
    assert req.pi_h3 == H3PiH3Input(steps=4, path="C:/m/pi4.safetensors")
    assert worker._pi_h3({"pi_h3": {"steps": 2, "path": "p", "extra": 1}})[1] == ["extra"]
    for bad in ({"steps": "4", "path": "p"}, {"steps": True, "path": "p"}, {"steps": 4, "path": ""}, [4, "p"]):
        with pytest.raises(ValueError, match="pi_h3"):
            worker._pi_h3({"pi_h3": bad})
    assert "pi_h3" in worker._GENERATE_FIELDS and "pi_h3" not in worker._CHAIN_FIELDS

    class FakePipe:
        seen = []

        def generate(self, req):
            FakePipe.seen.append(req)
            out = {"seed_used": 1}
            if req.pi_h3 is not None:
                out["pi_h3_used"] = "4steps"
            return out

    events, logs = [], []
    monkeypatch.setattr(worker, "_PIPE", FakePipe())
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(worker, "_log", logs.append)
    worker._do_generate({**msg, "num_inference_steps": 4, "pi_h3": {"steps": 4, "path": "C:/m/pi4.safetensors", "x": 1}})
    worker._do_generate(dict(msg))
    assert FakePipe.seen[0].pi_h3 == H3PiH3Input(4, "C:/m/pi4.safetensors") and FakePipe.seen[1].pi_h3 is None
    assert events[0]["pi_h3_used"] == "4steps" and "pi_h3_used" not in events[1]
    assert "pi_h3=(4, pi4.safetensors)" in logs[0] and "ignored keys: pi_h3.x" in logs[0]
    assert "pi_h3" not in logs[1] and "ignored keys" not in logs[1]
