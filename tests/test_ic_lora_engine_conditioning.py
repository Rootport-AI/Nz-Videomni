"""IC-LoRA conditioning_attention_strength — engine wiring unit tests (CPU-only).

Exercise the reference-conditioning wrapper branch added to
``LTXFastVideoPipeline._reference_conditioning_for_stage`` (fast_video_pipeline.py):
at attention_strength == 1.0 (or default) the reference conditioning is a bare
``VideoConditionByReferenceLatent`` (structurally byte-identical to before);
below 1.0 it is wrapped in ``ConditioningItemAttentionStrengthWrapper`` with the
scalar strength as its ``attention_mask`` (upstream ``iclora_utils`` parity).

Unit-level: no GPU, no model load, no LTX wheel. The pipeline module imports the
``ltx_core`` conditioning classes LAZILY inside the method under test (the video
loader is the repo's own ``engine.pipeline.common.load_video_conditioning_cpu``,
monkeypatched directly), so we inject tiny stand-in modules into ``sys.modules``
and drive the real method end-to-end.

Run this file in the ENGINE venv — the app ``.venv`` has no torch, so
``importorskip("torch")`` skips the whole module there:

  .venv-engine\\Scripts\\python.exe -m pytest tests\\test_ic_lora_engine_conditioning.py ^
      -p no:warnings -p no:cacheprovider --noconftest --rootdir .
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from engine.pipeline.fast_video_pipeline import (  # noqa: E402
    LTXFastVideoPipeline,
    _set_conv3d_memory_format,
)


# ── Stand-in conditioning classes mirroring the ltx_core signatures ──────────
class _StubVideoConditionByReferenceLatent:
    def __init__(self, latent, downscale_factor=1, strength=1.0):
        self.latent = latent
        self.downscale_factor = downscale_factor
        self.strength = strength


class _StubAttentionStrengthWrapper:
    def __init__(self, conditioning, attention_mask):
        self.conditioning = conditioning
        self.attention_mask = attention_mask


class _FakeVideoEncoder:
    """Stand-in for the wheel's ``VideoEncoder``: callable (plain encode) AND
    carrying ``tiled_encode`` — the method under test picks one by downscale
    factor, so both must exist and both must be observable."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.tiling_configs: list[object] = []
        self.inputs: list[object] = []

    def __call__(self, video):
        self.calls.append("plain")
        self.inputs.append(video)
        return torch.zeros(1, 128, 3, 4, 5)

    def tiled_encode(self, video, tiling_config=None):
        self.calls.append("tiled")
        self.tiling_configs.append(tiling_config)
        self.inputs.append(video)
        return torch.zeros(1, 128, 3, 4, 5)


@pytest.fixture
def stub_ltx(monkeypatch):
    """Inject stub ``ltx_core.conditioning`` + a stub reference-video loader.

    The method under test does ``from ltx_core.conditioning import (...)`` and
    ``from engine.pipeline.common import load_video_conditioning_cpu`` at call
    time, so pre-seeding sys.modules with the ltx stubs (parents included) makes
    the wheel imports resolve without the real wheel, and monkeypatching the
    loader on ``engine.pipeline.common`` is picked up by the lazy import.
    monkeypatch restores both.
    """
    cond_mod = types.ModuleType("ltx_core.conditioning")
    cond_mod.VideoConditionByReferenceLatent = _StubVideoConditionByReferenceLatent
    cond_mod.ConditioningItemAttentionStrengthWrapper = _StubAttentionStrengthWrapper

    import engine.pipeline.common as common_mod

    monkeypatch.setattr(
        common_mod,
        "load_video_conditioning_cpu",
        lambda **kw: torch.zeros(
            1, 3, int(kw["frame_cap"]), int(kw["height"]), int(kw["width"])
        ),
    )

    # cleanup_memory (gc + empty_cache + synchronize) is called before the
    # reference load; on CPU the real one is a no-op, the stub keeps the wheel out.
    helpers_mod = types.ModuleType("ltx_pipelines.utils.helpers")
    helpers_mod.cleanup_memory = lambda: None

    for name, mod in (
        ("ltx_core", types.ModuleType("ltx_core")),
        ("ltx_core.conditioning", cond_mod),
        ("ltx_pipelines", types.ModuleType("ltx_pipelines")),
        ("ltx_pipelines.utils", types.ModuleType("ltx_pipelines.utils")),
        ("ltx_pipelines.utils.helpers", helpers_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return cond_mod


class _FakePipe:
    """Minimal object exposing only the attributes the method reads."""

    def __init__(self, attn_strength: float, scale: int = 2) -> None:
        self._ic_reference = ("ref.mp4", 1.0)
        self._ic_reference_downscale_factor = scale
        self._ic_attention_strength = attn_strength


_TILING_SENTINEL = object()  # stands in for the TilingConfig the callers pass through


def _call(attn_strength: float, scale: int = 2, encoder=None, device="cpu"):
    """Invoke the real (unbound) method against a fake pipe + stage-1 cond_kwargs."""
    full_height = 768
    cond_height = full_height // 2  # 384 -> stage-1 gate passes
    cond_width = 640  # divisible by scale=2
    cond_kwargs = {
        "height": cond_height,
        "width": cond_width,
        "video_encoder": encoder if encoder is not None else _FakeVideoEncoder(),
        "dtype": torch.float32,
        "device": torch.device(device),
        "tiling_config": _TILING_SENTINEL,
    }
    fake = _FakePipe(attn_strength, scale=scale)
    method = LTXFastVideoPipeline._reference_conditioning_for_stage
    return method(fake, full_height=full_height, num_frames=9, cond_kwargs=cond_kwargs)


def test_reference_conditioning_no_wrap_at_one(stub_ltx):
    """strength 1.0 -> bare VideoConditionByReferenceLatent (no wrapper)."""
    conds = _call(1.0)
    assert len(conds) == 1
    assert isinstance(conds[0], _StubVideoConditionByReferenceLatent)
    assert not isinstance(conds[0], _StubAttentionStrengthWrapper)
    assert conds[0].downscale_factor == 2
    assert conds[0].strength == 1.0


def test_reference_conditioning_wraps_below_one(stub_ltx):
    """strength 0.6 -> ConditioningItemAttentionStrengthWrapper(mask == 0.6)."""
    conds = _call(0.6)
    assert len(conds) == 1
    wrapped = conds[0]
    assert isinstance(wrapped, _StubAttentionStrengthWrapper)
    assert wrapped.attention_mask == pytest.approx(0.6)
    assert isinstance(wrapped.conditioning, _StubVideoConditionByReferenceLatent)
    # underlying reference item is unchanged (strength + downscale preserved)
    assert wrapped.conditioning.strength == 1.0
    assert wrapped.conditioning.downscale_factor == 2


def test_reference_conditioning_at_scale_one(stub_ltx):
    """A same-resolution adapter (Deblur: reference_downscale_factor=1) feeds
    the reference at the FULL stage-1 resolution and reaches the conditioning
    item with downscale_factor=1."""
    conds = _call(1.0, scale=1)
    assert len(conds) == 1
    assert conds[0].downscale_factor == 1
    # stub encoder ignores its input, so assert on the loader call indirectly:
    # scale=1 leaves stage-1 dims untouched (no divisibility rejection either).
    assert isinstance(conds[0], _StubVideoConditionByReferenceLatent)


def test_scale_one_uses_tiled_encode(stub_ltx):
    """Factor 1 (deblur) carries 4x the reference pixels of factor 2 and OOM'd a
    16GB card on the untiled encode -> it must go through ``tiled_encode``, with
    the tiling config the caller threaded in via cond_kwargs."""
    enc = _FakeVideoEncoder()
    conds = _call(1.0, scale=1, encoder=enc)
    assert enc.calls == ["tiled"]
    assert enc.tiling_configs == [_TILING_SENTINEL]
    assert conds[0].downscale_factor == 1


def test_scale_two_stays_untiled(stub_ltx):
    """Factor 2 is deliberately NOT tiled: tiling splits the reference across
    temporal tiles and would break byte-identity with every existing output."""
    enc = _FakeVideoEncoder()
    conds = _call(1.0, scale=2, encoder=enc)
    assert enc.calls == ["plain"]
    assert enc.tiling_configs == []
    assert conds[0].downscale_factor == 2


def test_scale_two_moves_video_to_device(stub_ltx):
    """The loader now assembles the reference on the CPU, so the UNTILED factor-2
    path must ``.to(device)`` it before calling the encoder (VideoEncoder.forward
    expects device-resident input; tiled_encode moves tiles itself).

    ``device="meta"`` is the probe: it is not cuda (so the channels_last_3d /
    measurement branch stays off) yet it is distinguishable from the loader's
    "cpu" output, so a missing ``.to(device)`` shows up as a cpu tensor reaching
    the encoder — the regression that would silently kill every factor-2 job."""
    enc = _FakeVideoEncoder()
    _call(1.0, scale=2, encoder=enc, device="meta")
    assert enc.calls == ["plain"]
    assert enc.inputs[0].device.type == "meta"

    # ...and the tiled factor-1 path must NOT move it (tiles stream themselves).
    enc1 = _FakeVideoEncoder()
    _call(1.0, scale=1, encoder=enc1, device="meta")
    assert enc1.calls == ["tiled"]
    assert enc1.inputs[0].device.type == "cpu"


def test_reference_conditioning_unresolved_factor_raises(stub_ltx):
    """The factor is resolved by _set_ic_job; reaching the conditioning builder
    with it unset is a wiring bug. It must raise a real error (the previous
    ``assert`` vanished under ``python -O``)."""
    with pytest.raises(RuntimeError, match="not initialised"):
        _call(1.0, scale=None)


# ── Conv3d memory-format switch: the round trip must be lossless ────────────


class _MixedConvNet(torch.nn.Module):
    """Conv3d + Conv2d + Linear, i.e. the shape of module the helper must survive:
    a whole-module ``.to(channels_last_3d)`` would raise on the rank-4 Conv2d
    weight, which is why the helper filters on Conv3d."""

    def __init__(self) -> None:
        super().__init__()
        self.conv3d_a = torch.nn.Conv3d(2, 3, kernel_size=3)
        self.inner = torch.nn.Sequential(torch.nn.Conv3d(3, 4, kernel_size=3))
        self.conv2d = torch.nn.Conv2d(3, 4, kernel_size=3)
        self.linear = torch.nn.Linear(4, 5)


def test_conv3d_memory_format_round_trip_is_lossless():
    """channels_last_3d -> contiguous restores every tensor bit-for-bit and keeps
    the Parameter objects themselves (the encoder is rebuilt from a registry that
    aliases them), and non-Conv3d modules are never touched at all.

    This is what makes the reference-encode layout switch safe to undo in the
    ``finally``: stage 2 of the same job re-encodes keyframes with this encoder.
    """
    net = _MixedConvNet()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    params_before = {k: v for k, v in net.named_parameters()}
    non_conv3d = ("conv2d.weight", "conv2d.bias", "linear.weight", "linear.bias")

    touched = _set_conv3d_memory_format(net, torch.channels_last_3d)
    assert touched == 2  # both Conv3d, including the nested one; Conv2d/Linear skipped
    assert net.conv3d_a.weight.is_contiguous(memory_format=torch.channels_last_3d)
    assert net.inner[0].weight.is_contiguous(memory_format=torch.channels_last_3d)
    # non-Conv3d parameters keep their original (contiguous) layout
    for name in non_conv3d:
        assert net.state_dict()[name].is_contiguous()

    restored = _set_conv3d_memory_format(net, torch.contiguous_format)
    assert restored == 2

    after = net.state_dict()
    assert set(after) == set(before)
    for name, original in before.items():
        assert torch.equal(after[name], original), f"{name} changed value"
        assert after[name].is_contiguous(), f"{name} not back to contiguous"
    # ``Module.to(memory_format=)`` rewrites storages in place: same objects out
    for name, param in net.named_parameters():
        assert param is params_before[name], f"{name} was replaced, not converted"


# ── _set_ic_job: reference downscale factor resolution ──────────────────────
#
# These drive the REAL readers (safetensors + the wheel's private metadata
# reader), so they only run in the engine venv:
#   .venv-engine\Scripts\python.exe -m pytest tests\test_ic_lora_engine_conditioning.py --noconftest


@pytest.fixture
def write_lora(tmp_path):
    """Factory writing a one-tensor ``.safetensors`` carrying ``metadata``
    (safetensors requires str->str), returning its path."""
    save_file = pytest.importorskip("safetensors.torch").save_file
    pytest.importorskip("ltx_pipelines.ic_lora")

    def _write(name: str, metadata: dict | None = None) -> str:
        path = tmp_path / f"{name}.safetensors"
        save_file({"lora_A": torch.zeros(2, 2)}, str(path), metadata=metadata)
        return str(path)

    return _write


class _JobPipe:
    """Bare attribute bag — _set_ic_job only writes/reads its own _ic_* slots."""


def _set_ic_job(paths, reference=("ref.mp4", 1.0), attention_strength=1.0) -> _JobPipe:
    """Invoke the real (unbound) method with the positional call shape both
    ``generate()`` and ``chain_pipeline.run_chain`` use (they share it)."""
    pipe = _JobPipe()
    LTXFastVideoPipeline._set_ic_job(
        pipe, [(p, 1.0, None) for p in paths], reference, attention_strength
    )
    return pipe


def test_factor_one_accepted(write_lora):
    """A same-resolution control adapter (Deblur declares
    ``reference_downscale_factor="1"``) resolves to 1 instead of being rejected."""
    pipe = _set_ic_job([write_lora("deblur", {"reference_downscale_factor": "1"})])
    assert pipe._ic_reference_downscale_factor == 1


def test_factor_one_accepted_on_chain_call_shape(write_lora):
    """The chain route calls the very same method with an explicit attention
    strength (chain_pipeline.run_chain), so factor 1 must be accepted there too."""
    pipe = _set_ic_job(
        [write_lora("deblur", {"reference_downscale_factor": "1"})],
        attention_strength=0.6,
    )
    assert pipe._ic_reference_downscale_factor == 1
    assert pipe._ic_attention_strength == pytest.approx(0.6)


def test_factor_two_regression(write_lora):
    """Existing control adapters (union-control / x2 upscaler) declare 2 and are
    unaffected by the all-LoRA walk."""
    pipe = _set_ic_job([write_lora("union", {"reference_downscale_factor": "2"})])
    assert pipe._ic_reference_downscale_factor == 2


def test_style_lora_does_not_vote(write_lora):
    """A style/character LoRA carries no ``reference_downscale_factor`` key. It
    must neither vote nor be mistaken for a declared 1 (the wheel's reader
    returns 1 for both cases, which is why key presence is tested separately)."""
    style = write_lora("style", {"ss_network_dim": "64"})
    assert _set_ic_job(
        [style, write_lora("union", {"reference_downscale_factor": "2"})]
    )._ic_reference_downscale_factor == 2
    assert _set_ic_job(
        [style, write_lora("deblur", {"reference_downscale_factor": "1"})]
    )._ic_reference_downscale_factor == 1


def test_conflicting_one_and_two_rejected(write_lora):
    """One reference video is loaded at ONE resolution: a factor-1 adapter and a
    factor-2 adapter cannot both be served, so the combination is refused rather
    than silently feeding one of them an off-scale reference."""
    with pytest.raises(RuntimeError, match="Conflicting reference_downscale_factor"):
        _set_ic_job([
            write_lora("deblur", {"reference_downscale_factor": "1"}),
            write_lora("union", {"reference_downscale_factor": "2"}),
        ])


def test_conflicting_two_and_four_rejected(write_lora):
    with pytest.raises(RuntimeError, match="Conflicting reference_downscale_factor"):
        _set_ic_job([
            write_lora("union", {"reference_downscale_factor": "2"}),
            write_lora("other", {"reference_downscale_factor": "4"}),
        ])


def test_no_declaring_lora_rejected(write_lora):
    """A reference video with style-only adapters has no declared factor. The
    engine refuses rather than defaulting to 1 (the API rejects this earlier —
    this is the last line of defence)."""
    with pytest.raises(RuntimeError, match="declares"):
        _set_ic_job([write_lora("style", {"ss_network_dim": "64"})])


def test_metadata_absent_entirely_rejected(write_lora):
    """No ``__metadata__`` block at all reads back as None — same refusal."""
    with pytest.raises(RuntimeError, match="declares"):
        _set_ic_job([write_lora("bare")])


def test_non_positive_factor_rejected(write_lora):
    with pytest.raises(RuntimeError, match="expected >=1"):
        _set_ic_job([write_lora("broken", {"reference_downscale_factor": "0"})])


def test_reference_without_loras_rejected(write_lora):
    with pytest.raises(RuntimeError, match="no ic_loras"):
        _set_ic_job([])


def test_no_reference_leaves_factor_unset(write_lora):
    """No reference video -> no metadata read at all, factor stays None (the
    stale-clear path both routes rely on)."""
    pipe = _set_ic_job(
        [write_lora("union", {"reference_downscale_factor": "2"})], reference=None
    )
    assert pipe._ic_reference_downscale_factor is None
    assert pipe._ic_reference is None
