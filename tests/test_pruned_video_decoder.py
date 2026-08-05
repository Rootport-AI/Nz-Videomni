"""Structure of the pruned video VAE decoder (PrunaVAED, §3-50).

This is the machine half of gate G2: the module is built on the META device (no
weights, no GPU, no allocation) and every structural claim the design rests on
is checked against the workorder's flat-key table:

  * the parameter count is EXACTLY 345,006,256 — the number the upstream model
    card reports, reproduced here from an independent construction. It is the
    single most informative fingerprint available: get any width, any block
    count or any block KIND wrong and it moves;
  * the state_dict key set is exactly the 102 keys of ``PRUNAVAED_WORKORDER.md``
    §4.1, generated MECHANICALLY here from that table rather than read back off
    the module (a key set compared against itself proves nothing). The converter
    writes those very strings, so this is the backend half of the file-format
    contract;
  * ``norm3`` on both projection resnets is ``ChannelLayerNorm3d`` and NOT the
    ``nn.GroupNorm(num_groups=1)`` the wheel would have installed. Those two are
    DIFFERENT operators with the SAME parameter shape ``[C]``, so a mix-up loads
    without a single warning and shows up only as broken video (§2.5). Hence
    both a type assertion and a numerical one.

Run with ``.venv-engine`` and ``--noconftest``: the module imports ltx_core,
which the app venv does not have (there this whole file skips).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ltx_core")

from torch import nn  # noqa: E402

from ltx_core.model.video_vae.resnet import ResnetBlock3D, UNetMidBlock3D  # noqa: E402
from ltx_core.model.video_vae.sampling import DepthToSpaceUpsample  # noqa: E402

from engine.vae.pruned_video_decoder import (  # noqa: E402
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_STATE_DICT_KEYS,
    ChannelLayerNorm3d,
    PrunedVideoDecoder,
    PrunedVideoDecoderConfigurator,
)

# The metadata a converted PrunaVAED file carries (§4.2). Kept literal so the
# configurator's identity check is exercised on the real shape of the input.
GOOD_CONFIG = {
    "vae": {
        "_class_name": "PrunaVAEDDecoder",
        "latent_channels": 128,
        "patch_size": 4,
        "norm_layer": "pixel_norm",
        "causal_decoder": False,
        "timestep_conditioning": False,
        "decoder_base_channels": 128,
    }
}


def _expected_keys() -> set[str]:
    """The 102 keys of §4.1, generated from the table's own description.

    Written out block by block on purpose: the point is to encode the WORKORDER,
    independently of what the module happens to produce.
    """
    keys = {
        "conv_in.conv.weight",
        "conv_in.conv.bias",
        "conv_out.conv.weight",
        "conv_out.conv.bias",
        "per_channel_statistics.mean-of-means",
        "per_channel_statistics.std-of-means",
    }

    def res_x(flat_idx: int, num_layers: int) -> None:
        # UNetMidBlock3D: every resnet has in==out, so no conv_shortcut and no
        # norm3 (both are nn.Identity and therefore contribute no keys).
        for i in range(num_layers):
            for conv in ("conv1", "conv2"):
                for suffix in ("weight", "bias"):
                    keys.add(f"up_blocks.{flat_idx}.res_blocks.{i}.{conv}.conv.{suffix}")

    def upsample(flat_idx: int) -> None:
        # DepthToSpaceUpsample holds its CausalConv3d as ``conv``, whose real
        # convolution is one more level down (``conv.conv``).
        for suffix in ("weight", "bias"):
            keys.add(f"up_blocks.{flat_idx}.conv.conv.{suffix}")

    def projection(flat_idx: int) -> None:
        # conv1/conv2 go through CausalConv3d (``.conv``); conv_shortcut is a
        # BARE nn.Conv3d from make_linear_nd, so it does NOT. Mixing those two
        # up is the direct route to the silent-failure trap of §2.2.
        for suffix in ("weight", "bias"):
            keys.add(f"up_blocks.{flat_idx}.conv1.conv.{suffix}")
            keys.add(f"up_blocks.{flat_idx}.conv2.conv.{suffix}")
            keys.add(f"up_blocks.{flat_idx}.conv_shortcut.{suffix}")
            keys.add(f"up_blocks.{flat_idx}.norm3.{suffix}")

    res_x(0, 2)
    upsample(1)
    res_x(2, 2)
    projection(3)
    upsample(4)
    res_x(5, 4)
    projection(6)
    upsample(7)
    res_x(8, 6)
    upsample(9)
    res_x(10, 4)
    return keys


@pytest.fixture(scope="module")
def decoder() -> PrunedVideoDecoder:
    # META device: the module is 345M parameters, which would be 1.4GB of fp32
    # if it were ever materialised. Nothing here needs values.
    with torch.device("meta"):
        return PrunedVideoDecoderConfigurator.from_config(GOOD_CONFIG)


# --------------------------------------------------------------------------- #
# parameter count + key set (the two fingerprints)
# --------------------------------------------------------------------------- #

def test_parameter_count_is_exactly_the_upstream_number(decoder):
    # 345,006,256 is what the PrunaVAED model card reports and what §2.3's
    # independent hand summation produced. Reproducing it to the single
    # parameter is the decisive evidence that the block layout is right.
    total = sum(p.numel() for p in decoder.parameters())
    assert total == 345_006_256
    assert total == EXPECTED_PARAMETER_COUNT
    # The per-channel statistics are BUFFERS (2 x 128), counted separately —
    # they are part of the file and of the state dict, but not parameters.
    assert sum(b.numel() for b in decoder.buffers()) == 256


def test_state_dict_keys_match_the_workorder_table(decoder):
    got = set(decoder.state_dict())
    want = _expected_keys()
    assert len(want) == EXPECTED_STATE_DICT_KEYS == 102
    assert got == want, (
        f"missing={sorted(want - got)} unexpected={sorted(got - want)}"
    )


def test_load_state_dict_reports_no_missing_or_unexpected_keys(decoder):
    # The converter's output is loaded with strict=False + assign=True, which
    # SWALLOWS a key mismatch (§2.2) — only the RETURN VALUE lists it. A stand-in
    # state dict keyed off the workorder table is pushed through that same door
    # here (a fresh instance, so the shared fixture keeps its own tensors); G2
    # repeats the check against the real converted file.
    with torch.device("meta"):
        fresh = PrunedVideoDecoderConfigurator.from_config(GOOD_CONFIG)
    shapes = decoder.state_dict()
    sd = {key: torch.empty_like(shapes[key]) for key in _expected_keys()}
    result = fresh.load_state_dict(sd, strict=False, assign=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


# --------------------------------------------------------------------------- #
# block layout
# --------------------------------------------------------------------------- #

def test_up_blocks_have_the_expected_kinds_in_order(decoder):
    kinds = [type(b) for b in decoder.up_blocks]
    assert kinds == [
        UNetMidBlock3D,       # 0  res_x x2 @1024
        DepthToSpaceUpsample,  # 1  compress_all m=2
        UNetMidBlock3D,       # 2  res_x x2 @512
        ResnetBlock3D,        # 3  projection 512->384
        DepthToSpaceUpsample,  # 4  compress_all m=1
        UNetMidBlock3D,       # 5  res_x x4 @384
        ResnetBlock3D,        # 6  projection 384->256
        DepthToSpaceUpsample,  # 7  compress_time m=2
        UNetMidBlock3D,       # 8  res_x x6 @128
        DepthToSpaceUpsample,  # 9  compress_space m=2
        UNetMidBlock3D,       # 10 res_x x4 @64
    ]


@pytest.mark.parametrize(
    "key,shape",
    [
        ("conv_in.conv.weight", (1024, 128, 3, 3, 3)),
        ("up_blocks.1.conv.conv.weight", (4096, 1024, 3, 3, 3)),
        ("up_blocks.3.conv_shortcut.weight", (384, 512, 1, 1, 1)),
        ("up_blocks.3.norm3.weight", (512,)),
        ("up_blocks.4.conv.conv.weight", (3072, 384, 3, 3, 3)),
        ("up_blocks.6.conv_shortcut.weight", (256, 384, 1, 1, 1)),
        ("up_blocks.6.norm3.weight", (384,)),
        ("up_blocks.7.conv.conv.weight", (256, 256, 3, 3, 3)),
        ("up_blocks.9.conv.conv.weight", (256, 128, 3, 3, 3)),
        # conv_out is the load-bearing proof that the throwaway width-only
        # skeleton did its one job: it must take 64 channels, not the stock 128.
        ("conv_out.conv.weight", (48, 64, 3, 3, 3)),
    ],
)
def test_measured_tensor_shapes(decoder, key, shape):
    assert tuple(decoder.state_dict()[key].shape) == shape


def test_projection_resnets_use_channel_layer_norm_not_group_norm(decoder):
    # The single most dangerous mistake in this feature: nn.GroupNorm(1) and a
    # channel-only LayerNorm carry the SAME [C] weight/bias, so a mix-up loads
    # silently and only breaks the picture.
    for idx in (3, 6):
        norm3 = decoder.up_blocks[idx].norm3
        assert isinstance(norm3, ChannelLayerNorm3d)
        assert not isinstance(norm3, nn.GroupNorm)
        assert norm3.eps == 1e-6


def test_every_other_resnet_keeps_the_identity_norm3(decoder):
    # The stock blocks have in==out, so their norm3 stays nn.Identity — which is
    # why this whole trap never fired before PrunaVAED introduced projections.
    for idx in (0, 2, 5, 8, 10):
        for res in decoder.up_blocks[idx].res_blocks:
            assert isinstance(res.norm3, nn.Identity)
            assert isinstance(res.conv_shortcut, nn.Identity)


# --------------------------------------------------------------------------- #
# ChannelLayerNorm3d numerics
# --------------------------------------------------------------------------- #

def _reference_channelwise_layernorm(x, weight, bias, eps=1e-6):
    """diffusers' norm3, written out: ``nn.LayerNorm(C)`` on a channel-last view.

    Deliberately built from a SEPARATE nn.LayerNorm instance rather than from
    our own class, so the test compares two implementations instead of one.
    """
    channels = x.shape[1]
    ln = nn.LayerNorm(channels, eps=eps, elementwise_affine=True, bias=True)
    with torch.no_grad():
        ln.weight.copy_(weight)
        ln.bias.copy_(bias)
    return ln(x.movedim(1, -1)).movedim(-1, 1)


def test_channel_layer_norm_matches_a_reference_channelwise_layernorm():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 3, 4, 4)
    mod = ChannelLayerNorm3d(5)
    with torch.no_grad():
        mod.weight.copy_(torch.randn(5))
        mod.bias.copy_(torch.randn(5))

    got = mod(x)
    want = _reference_channelwise_layernorm(x, mod.weight, mod.bias)
    assert got.shape == x.shape
    torch.testing.assert_close(got, want)


def test_channel_layer_norm_normalises_over_channels_only():
    # Direct statement of the axis set: each (batch, frame, y, x) position has
    # zero mean and unit variance ACROSS CHANNELS after an affine-free pass.
    torch.manual_seed(1)
    x = torch.randn(2, 8, 3, 4, 4)
    mod = ChannelLayerNorm3d(8, eps=1e-12)
    with torch.no_grad():
        mod.weight.fill_(1.0)
        mod.bias.zero_()
    out = mod(x)
    torch.testing.assert_close(
        out.mean(dim=1), torch.zeros(2, 3, 4, 4), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        out.var(dim=1, unbiased=False), torch.ones(2, 3, 4, 4), atol=1e-4, rtol=0
    )


def test_group_norm_with_one_group_is_a_DIFFERENT_operator():
    # The wheel's comment claims GroupNorm(num_groups=1) "is equivalent to
    # LayerNorm" (resnet.py:89-90). True for LayerNorm([C,F,H,W]); FALSE for the
    # channel-only LayerNorm the weights were trained with. If this test ever
    # starts passing as "close", the whole justification for this module is gone
    # — so it asserts the two DISAGREE.
    torch.manual_seed(2)
    x = torch.randn(1, 6, 3, 4, 4)
    ours = ChannelLayerNorm3d(6)
    theirs = nn.GroupNorm(num_groups=1, num_channels=6, eps=1e-6, affine=True)
    with torch.no_grad():
        w, b = torch.randn(6), torch.randn(6)
        ours.weight.copy_(w)
        ours.bias.copy_(b)
        theirs.weight.copy_(w)
        theirs.bias.copy_(b)
    assert not torch.allclose(ours(x), theirs(x), atol=1e-3)


# --------------------------------------------------------------------------- #
# configurator: the file's metadata is checked, never trusted as authority
# --------------------------------------------------------------------------- #

def test_configurator_accepts_a_well_formed_config():
    with torch.device("meta"):
        model = PrunedVideoDecoderConfigurator.from_config(GOOD_CONFIG)
    assert isinstance(model, PrunedVideoDecoder)


@pytest.mark.parametrize(
    "key,value",
    [
        ("_class_name", "CausalVideoAutoencoder"),  # the STOCK decoder's file
        ("latent_channels", 64),
        ("patch_size", 2),
        ("norm_layer", "group_norm"),
        ("causal_decoder", True),
        ("timestep_conditioning", True),
        ("decoder_base_channels", 64),
    ],
)
def test_configurator_rejects_a_foreign_file(key, value):
    cfg = {"vae": dict(GOOD_CONFIG["vae"], **{key: value})}
    with pytest.raises(ValueError, match="config mismatch"):
        with torch.device("meta"):
            PrunedVideoDecoderConfigurator.from_config(cfg)


def test_configurator_rejects_a_missing_identity_key():
    vae = dict(GOOD_CONFIG["vae"])
    del vae["patch_size"]
    with pytest.raises(ValueError, match="config mismatch"):
        with torch.device("meta"):
            PrunedVideoDecoderConfigurator.from_config({"vae": vae})


def test_configurator_rejects_decoder_blocks():
    # Carrying decoder_blocks would let the STOCK VideoDecoderConfigurator read
    # this same file and build a silently wrong decoder (§4.2). Absent, it fails
    # loudly instead — so the converter must not emit it, and we say so here.
    cfg = {"vae": dict(GOOD_CONFIG["vae"], decoder_blocks=[["res_x", 2]])}
    with pytest.raises(ValueError, match="decoder_blocks"):
        with torch.device("meta"):
            PrunedVideoDecoderConfigurator.from_config(cfg)


def test_timestep_conditioning_is_refused_at_construction():
    # It would add last_time_embedder / scale_shift_table parameters that no
    # PrunaVAED file contains -> "Uninitialized parameters" and a meta-device
    # model returned as if nothing happened.
    with pytest.raises(ValueError, match="timestep_conditioning"):
        with torch.device("meta"):
            PrunedVideoDecoder(timestep_conditioning=True)
