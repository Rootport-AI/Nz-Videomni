"""§3-167 B-2 / §3-168: the LTX 2.5 engine reads a quantized safetensors transformer (CPU only).

Pins what engine25 adds on top of the shared pieces (``sft_quant_format`` and
``engine.sft_quant.quant_service``, tested on their own):

  * ``Ltx25DiffusionStage.from_safetensors`` builds a 4-block model from a fake
    quantized file -- prefixed and bare names; fp8 with scale shape ``()`` and
    ``[1]``, scaled and plain layers mixed; int8 with a scalar or a per-row
    scale; int8 ConvRot mixed with int8 and fp8 on a 256-wide config; the
    REDGraft mix (int8 ConvRot + w4a8 + fp8, §3-168 C-3) on the same config; F32
    tensors; a connector in fp8, int8 or w4a8 -- and its forward equals the forward
    of the same model loaded with the weights brought back to bf16 (by
    ``engine.sft_quant.dequant.dequantize``); ``dispose`` and a rebuild give
    the same numbers again;
  * ``Ltx25SftStateDictLoader.metadata()`` has ltx_core 1.2's shape
    (``metadata()["config"]["transformer"]``) and is parsed once;
  * the EmbeddingsProcessor loader takes the connectors from the safetensors and
    the projections from a GGUF, and still refuses overlapping files;
  * ``ancestral_detection_skipped`` rebinds the probe the constructor looks up
    and restores it in ``finally``; ``verify`` pins the global-name lookup;
  * ``Ltx25Pipeline`` picks ``from_safetensors`` / ``from_gguf`` by extension and
    reports ``detected`` as None.

Run with ``.venv-engine-ltx25`` and ``--noconftest`` (through the runner that
borrows the app venv's pytest).
"""

from __future__ import annotations

import functools
import json
import types

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

import torch  # noqa: E402
from torch import nn  # noqa: E402

import sft_quant_format  # noqa: E402
from engine.sft_quant.dequant import dequantize, hadamard  # noqa: E402
from engine25 import ltxcore_compat, pipeline25  # noqa: E402
from engine25.gguf_gemma4 import (  # noqa: E402
    Ltx25SftConnectorLoader,
    Ltx25GemmaError,
    Ltx25MultiGgufStateDictLoader,
)
from engine25.gguf_transformer import (  # noqa: E402
    LTX25_EMBEDDINGS_PROCESSOR_KEY_OPS,
    Ltx25CpuModelBuilder,
    Ltx25DiffusionStage,
    Ltx25SftStateDictLoader,
    _dummy_video_modality,
)
from engine25.ltxcore_compat import (  # noqa: E402
    DistilledPipeline,
    LTXModelConfigurator,
    ancestral_detection_skipped,
    ltx_distilled,
)

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
CPU = torch.device("cpu")
P = "model.diffusion_model."
N_BLOCKS = 4

#: The real LTX 2.5 transformer config (from the distilled GGUF's KV block),
#: shrunk to 4 blocks and toy widths. The flags are the real ones so the
#: skeleton has the real 2.5 layout (gated attention, cross-attention AdaLN, no
#: FF bias, 84 tensors per block).
TRANSFORMER_CONFIG = {
    "activation_fn": "gelu-approximate", "attention_bias": True, "attention_type": "default",
    "caption_channels": 3840, "double_self_attention": False, "dropout": 0.0,
    "norm_elementwise_affine": False, "norm_eps": 1e-06, "num_embeds_ada_norm": 1000,
    "num_vector_embeds": None, "only_cross_attention": False, "cross_attention_norm": True,
    "upcast_attention": False, "use_linear_projection": False, "qk_norm": "rms_norm",
    "standardization_norm": "rms_norm", "positional_embedding_type": "rope",
    "positional_embedding_theta": 10000.0, "positional_embedding_max_pos": [20, 2048, 2048],
    "timestep_scale_multiplier": 1000, "av_ca_timestep_scale_multiplier": 1000.0,
    "use_audio_video_cross_attention": True, "ff_bias": False, "share_ff": False,
    "audio_positional_embedding_max_pos": [20], "av_cross_ada_norm": True,
    "use_middle_indices_grid": True, "apply_gated_attention": True,
    "caption_proj_before_connector": True, "cross_attention_adaln": True, "rope_type": "split",
    "frequencies_precision": "float64", "use_keyframes_abs_pos_embedding": True,
    # toy sizes
    "num_layers": N_BLOCKS, "num_attention_heads": 2, "attention_head_dim": 8,
    "in_channels": 8, "out_channels": 8, "cross_attention_dim": 16,
    "audio_num_attention_heads": 2, "audio_attention_head_dim": 4,
    "audio_in_channels": 8, "audio_out_channels": 8, "audio_cross_attention_dim": 8,
}
CONFIG = {"transformer": TRANSFORMER_CONFIG, "scheduler": {}}
#: The same, 256 wide (2 heads x 128): the video Linears' in_features become
#: multiples of 256, which ConvRot needs (the audio side stays 8 wide).
CONFIG_256 = {
    "transformer": {
        **TRANSFORMER_CONFIG, "attention_head_dim": 128, "cross_attention_dim": 256,
    },
    "scheduler": {},
}
GEMMA_SOURCE = {"ltx_version": "2.5.0", "gemma_version": "gemma4-12b-ltx-v1"}


def _metadata(config: dict) -> dict:
    return {
        "config": json.dumps(config),
        "model_version": "2.5.0",
        "gemma_source_checkpoint": json.dumps(GEMMA_SOURCE),
        "license": "not json",
    }

#: A few connector tensors (names as in the real file, sizes toy).
CONNECTOR_SHAPES = {
    "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight": (16, 16),
    "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.bias": (16,),
    "video_embeddings_connector.learnable_registers": (4, 16),
    "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight": (8, 8),
    "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.bias": (8,),
    "audio_embeddings_connector.learnable_registers": (4, 8),
}


# --------------------------------------------------------------------------- #
# fake file
# --------------------------------------------------------------------------- #


def _skeleton(config: dict = CONFIG) -> nn.Module:
    return LTXModelConfigurator.from_metadata({"config": config})


def _int8_marker(convrot: bool) -> torch.Tensor:
    conf = {"format": "int8_tensorwise", "convrot": convrot, "convrot_groupsize": 256}
    return torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)


def _w4a8_marker() -> torch.Tensor:
    conf = {"format": "asym_w4a8_int8", "group_size": 16, "convrot": True, "convrot_groupsize": 256}
    return torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)


def _quantize_w4a8(val: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """A w4a8 encoding of ``val`` (rotate, per-row s_channel onto +/-127, per
    16-column s_rel in fp8, nearest code of a non-uniform codebook, two codes
    per byte, low nibble first). s_rel is stored F8_E4M3, as REDGraft's file
    does (a U8 s_rel is covered by test_sft_quant_loader_service).
    The reference is the engine's own ``dequantize``."""
    o, i = val.shape
    rotated = (val.view(o, i // 256, 256) @ hadamard("cpu")).view(o, i)
    s_channel = (rotated.abs().amax(dim=1) / 127).clamp_min(1e-8).to(torch.float32)
    target = (rotated / s_channel[:, None]).view(o, i // 16, 16)
    codebook = torch.tensor(
        [-1.0, -0.8, -0.62, -0.47, -0.34, -0.22, -0.11, -0.03,
         0.03, 0.11, 0.22, 0.34, 0.47, 0.62, 0.8, 1.0], dtype=torch.float32,
    )
    s_rel = target.abs().amax(dim=2).clamp_min(1.0).to(FP8)
    level = target / s_rel.to(torch.float32)[:, :, None]
    codes = (level.unsqueeze(-1) - codebook).abs().argmin(-1).view(o, i).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
    leaves = {
        "weight": packed,
        "weight_s_rel": s_rel,
        "weight_s_channel": s_channel,
        "weight_codebook": codebook,
        "comfy_quant": _w4a8_marker(),
    }
    aux = {"weight_s_rel": s_rel, "weight_s_channel": s_channel, "weight_codebook": codebook}
    return leaves, dequantize("w4a8", packed, aux, BF16)


def _quantize(scheme: str, val: torch.Tensor, scale_shape) -> tuple[dict, torch.Tensor]:
    """One Linear weight in ``scheme``: (leaf -> stored tensor, bf16 reference).

    ``scale_shape`` is ``()`` / ``(1,)`` (one scale) or ``"row"`` (int8 only:
    one per output row, stored ``(o, 1)``). The int8 references come from the
    engine's own ``dequantize`` (the rules are pinned in test_sft_quant_dequant).
    """
    o, i = val.shape
    if scheme == "w4a8":
        return _quantize_w4a8(val)
    if scheme == "fp8":
        w8 = val.to(FP8)
        return {"weight": w8}, w8.to(BF16)
    if scheme == "fp8_scaled":
        scale = torch.tensor(0.01, dtype=torch.float32)
        w8 = (val / scale).to(FP8)
        leaves = {
            "weight": w8,
            "weight_scale": scale.reshape(scale_shape),
            "input_scale": torch.ones((), dtype=torch.float32),  # ignored by the loader
        }
        return leaves, (w8.to(torch.float32) * scale).to(BF16)
    convrot = scheme == "int8_convrot"
    if convrot:
        val = (val.view(o, i // 256, 256) @ hadamard("cpu")).view(o, i)
    if scale_shape == "row":
        scale = (val.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8)
        stored_scale = scale
    else:
        scale = (val.abs().amax() / 127).clamp_min(1e-8).reshape(1, 1).expand(o, 1)
        stored_scale = scale[0, 0].reshape(scale_shape).clone()
    q = torch.clamp(torch.round(val / scale), -127, 127).to(torch.int8)
    leaves = {"weight": q, "weight_scale": stored_scale, "comfy_quant": _int8_marker(convrot)}
    return leaves, dequantize(scheme, q, {"weight_scale": scale.contiguous()}, BF16)


def _scheme_for(pattern: str, n: int, in_features: int) -> str:
    """The n-th block Linear's scheme under a CASE's pattern."""
    if pattern == "fp8":
        return "fp8_scaled" if n % 2 else "fp8"
    if pattern == "int8":
        return "int8"
    if pattern == "redgraft":
        # REDGraft LTX 2.5: ConvRot and w4a8 wherever the width allows, else fp8
        if in_features % 256 == 0:
            return "w4a8" if n % 2 else "int8_convrot"
        return "fp8_scaled"
    # "mixed": ConvRot wherever the width allows, else int8 and fp8 in turn
    if in_features % 256 == 0:
        return "int8_convrot"
    return "int8" if n % 2 else "fp8_scaled"


def _write_quant(
    path, *, prefix: str, pattern: str = "fp8", scale_shape=(), connector: str = "bf16",
    config: dict = CONFIG,
):
    """Write a 4-block quantized transformer; return (reference bf16 sd, reference connectors).

    Block Linear weights follow ``pattern`` (see :func:`_scheme_for`). Every
    ``scale_shift_table`` is stored F32 (the loader brings F32 to bf16), the
    rest bf16. The connector's weights are bf16, scaled fp8, int8 (per-row
    scale) or w4a8 (REDGraft's; widened to 256 inputs, which w4a8 needs). The
    references are what the model should end up holding.
    """
    from safetensors.torch import save_file

    model = _skeleton(config)
    linears = {name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
    gen = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {}
    ref: dict[str, torch.Tensor] = {}
    n_quant = 0
    for key, value in model.state_dict().items():
        val = torch.randn(value.shape, generator=gen) * 0.1
        layer = key[: -len(".weight")] if key.endswith(".weight") else None
        if key.startswith("transformer_blocks.") and layer in linears:
            n_quant += 1
            scheme = _scheme_for(pattern, n_quant, val.shape[1])
            shape = scale_shape if scheme.startswith("int8") or scale_shape != "row" else ()
            leaves, ref[key] = _quantize(scheme, val, shape)
            for leaf, tensor in leaves.items():
                tensors[prefix + layer + "." + leaf] = tensor
        elif key.endswith("scale_shift_table"):
            tensors[prefix + key] = val.to(torch.float32)
            ref[key] = val.to(BF16)
        else:
            tensors[prefix + key] = val.to(BF16)
            ref[key] = tensors[prefix + key]

    connectors: dict[str, torch.Tensor] = {}
    for key, shape in CONNECTOR_SHAPES.items():
        if connector == "w4a8" and key.endswith(".weight"):
            shape = (shape[0], 256)
        val = torch.randn(shape, generator=gen) * 0.1
        layer = key[: -len(".weight")]
        if connector != "bf16" and key.endswith(".weight"):
            scheme = {"fp8": "fp8_scaled", "int8": "int8", "w4a8": "w4a8"}[connector]
            leaves, connectors[key] = _quantize(
                scheme, val, "row" if connector == "int8" else scale_shape
            )
            for leaf, tensor in leaves.items():
                tensors[prefix + layer + "." + leaf] = tensor
        else:
            tensors[prefix + key] = val.to(BF16)
            connectors[key] = tensors[prefix + key]

    # The TE-side projection some files carry: inside the prefix for a bare
    # file, outside it for a prefixed one. Never part of the transformer.
    tensors["text_embedding_projection.video_aggregate_embed.weight"] = torch.zeros(4, 4, dtype=BF16)

    save_file(tensors, str(path), metadata=_metadata(config))
    return ref, connectors


@pytest.fixture
def inspect_4_blocks(monkeypatch):
    """``inspect`` expects the real 48 blocks; the fake has 4."""
    monkeypatch.setattr(
        sft_quant_format, "inspect", functools.partial(sft_quant_format.inspect, expected_blocks=N_BLOCKS)
    )


def _reference_forward(ref_sd: dict[str, torch.Tensor], modality, config: dict = CONFIG) -> torch.Tensor:
    model = _skeleton(config)
    model.load_state_dict(ref_sd, strict=True, assign=True)
    with torch.no_grad():
        return model.eval()(modality, None, None)[0]


def _modality(config: dict = CONFIG):
    return _dummy_video_modality(config, device=CPU, dtype=BF16, width=64, height=64, frames=9)


#: (prefix, pattern, scale_shape, connector, config)
CASES = [
    pytest.param(P, "fp8", (), "bf16", CONFIG, id="prefixed-scale0d"),
    pytest.param("", "fp8", (1,), "fp8", CONFIG, id="bare-scale1-fp8connector"),
    pytest.param(P, "fp8", (1,), "fp8", CONFIG, id="prefixed-scale1-fp8connector"),
    pytest.param(P, "int8", (), "bf16", CONFIG, id="int8-prefixed-scalar"),
    pytest.param(P, "int8", "row", "int8", CONFIG, id="int8-prefixed-row-int8connector"),
    pytest.param("", "int8", (1,), "int8", CONFIG, id="int8-bare-scale1-int8connector"),
    pytest.param(P, "mixed", "row", "int8", CONFIG_256, id="convrot-mixed-256"),
    pytest.param(P, "redgraft", "row", "w4a8", CONFIG_256, id="w4a8-redgraft-mix-256-w4a8connector"),
]
CASE_ARGS = ("prefix", "pattern", "scale_shape", "connector", "config")
_SCALE_SHAPE_OF = {"fp8_scaled": lambda o: (), "int8": lambda o: (o, 1), "int8_convrot": lambda o: (o, 1)}
#: w4a8's auxiliary tensors after normalization: leaf -> (dtype, shape of (o, i)).
_W4A8_AUX_OF = {
    "weight_s_rel": (FP8, lambda o, i: (o, i // 16)),
    "weight_s_channel": (torch.float32, lambda o, i: (o,)),
    "weight_codebook": (torch.float32, lambda o, i: (16,)),
}


# --------------------------------------------------------------------------- #
# transformer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(CASE_ARGS, CASES)
def test_from_safetensors_forward_matches_bf16_reference_and_survives_dispose(
    tmp_path, inspect_4_blocks, prefix, pattern, scale_shape, connector, config
):
    path = tmp_path / "t.safetensors"
    ref_sd, _ = _write_quant(
        path, prefix=prefix, pattern=pattern, scale_shape=scale_shape, connector=connector, config=config
    )
    modality = _modality(config)
    expected = _reference_forward(ref_sd, modality, config)

    stage = Ltx25DiffusionStage.from_safetensors(str(path), device=CPU, blocks_on_gpu=0, cache_weights=True)
    assert isinstance(stage._transformer_builder, Ltx25CpuModelBuilder)
    assert isinstance(stage._transformer_builder.model_loader, Ltx25SftStateDictLoader)
    layers = stage._transformer_builder.model_loader.layout.layers
    wanted = {"fp8": {"fp8", "fp8_scaled"}, "int8": {"int8"},
              "mixed": {"int8_convrot", "int8", "fp8_scaled"},
              "redgraft": {"int8_convrot", "w4a8", "fp8_scaled"}}[pattern]
    assert set(layers.values()) == wanted

    outs = []
    for _ in range(2):  # build, forward, dispose -- then again from the cache
        x0 = stage._build_transformer(device=CPU)
        velocity = x0.velocity_model
        state = velocity.state_dict()
        stored = {t.dtype for t in state.values()}
        assert (FP8 in stored) == (pattern != "int8")
        assert (torch.int8 in stored) == (pattern != "fp8")
        scales = [k for k in state if k.endswith(".weight_scale")]
        assert scales
        for k in scales:
            layer = k[: -len(".weight_scale")]
            o = state[layer + ".weight"].shape[0]
            assert state[k].dtype == torch.float32
            assert state[k].shape == _SCALE_SHAPE_OF[layers[layer]](o), k
        # the quantized weights are Parameters that do not require grad (int8 cannot)
        for layer in layers:
            w = velocity.get_submodule(layer).weight
            assert isinstance(w, nn.Parameter) and not w.requires_grad
        # w4a8: packed weight (o, i/2) and its three auxiliary tensors
        for layer in (n for n, sch in layers.items() if sch == "w4a8"):
            mod = velocity.get_submodule(layer)
            o, i = mod.out_features, mod.in_features
            assert mod.weight.dtype == torch.int8 and mod.weight.shape == (o, i // 2)
            for leaf, (dtype, shape) in _W4A8_AUX_OF.items():
                t = state[f"{layer}.{leaf}"]
                assert (t.dtype, tuple(t.shape)) == (dtype, shape(o, i)), f"{layer}.{leaf}"
        assert not any(k.startswith(("text_embedding_projection.", "video_embeddings_connector."))
                       for k in state)
        with torch.no_grad():
            outs.append(velocity(modality, None, None)[0])
        x0.dispose()
        assert all(t.is_meta for t in velocity.state_dict().values())

    assert torch.equal(outs[0], expected)
    assert torch.equal(outs[1], expected)


def test_fp8_loader_metadata_has_ltx_core_1_2_shape_and_is_parsed_once(tmp_path, inspect_4_blocks, monkeypatch):
    path = tmp_path / "t.safetensors"
    _write_quant(path, prefix=P)
    loader = Ltx25SftStateDictLoader(str(path), sft_quant_format.inspect(str(path)))

    def _no_second_read(_path):
        raise AssertionError("metadata() re-read the header")

    monkeypatch.setattr(sft_quant_format, "read_header", _no_second_read)
    for _ in range(2):
        meta = loader.metadata(str(path))
        assert meta["config"]["transformer"]["num_layers"] == N_BLOCKS
        assert meta["model_version"] == "2.5.0"
        assert meta["gemma_source_checkpoint"] == GEMMA_SOURCE
        assert meta["license"] == "not json"


def test_build_calls_the_fp8_placement_check(tmp_path, inspect_4_blocks, monkeypatch):
    from engine25 import gguf_transformer

    seen = []
    monkeypatch.setattr(gguf_transformer, "_assert_quant_only_in_linears", seen.append)
    path = tmp_path / "t.safetensors"
    _write_quant(path, prefix=P)
    stage = Ltx25DiffusionStage.from_safetensors(str(path), device=CPU, blocks_on_gpu=0)
    x0 = stage._build_transformer(device=CPU)
    assert seen == [x0.velocity_model]


def test_from_safetensors_refuses_what_inspect_refuses(tmp_path):
    # 4 blocks against the real 48: the one acceptance check runs in from_safetensors.
    path = tmp_path / "t.safetensors"
    _write_quant(path, prefix=P)
    with pytest.raises(sft_quant_format.QuantFormatError, match="transformer_blocks"):
        Ltx25DiffusionStage.from_safetensors(str(path), device=CPU)


# --------------------------------------------------------------------------- #
# EmbeddingsProcessor loader
# --------------------------------------------------------------------------- #


def _write_te_gguf(path) -> dict[str, torch.Tensor]:
    import gguf
    import numpy as np

    arrays = {
        "text_embedding_projection.video_aggregate_embed.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "text_embedding_projection.audio_aggregate_embed.weight": np.ones((2, 4), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((4, 4), dtype=np.float32),  # Gemma: dropped
    }
    writer = gguf.GGUFWriter(str(path), arch="gemma4")
    writer.add_string("config", json.dumps({"text_encoder": {}}))
    for name, array in arrays.items():
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {
        "feature_extractor.video_aggregate_embed.weight": torch.from_numpy(arrays[
            "text_embedding_projection.video_aggregate_embed.weight"]),
        "feature_extractor.audio_aggregate_embed.weight": torch.from_numpy(arrays[
            "text_embedding_projection.audio_aggregate_embed.weight"]),
    }


@pytest.mark.parametrize(CASE_ARGS, CASES)
def test_embeddings_loader_mixes_quantized_safetensors_and_gguf(
    tmp_path, prefix, pattern, scale_shape, connector, config
):
    sft = tmp_path / "t.safetensors"
    te = tmp_path / "te.gguf"
    _, connectors = _write_quant(
        sft, prefix=prefix, pattern=pattern, scale_shape=scale_shape, connector=connector, config=config
    )
    projections = _write_te_gguf(te)

    loader = Ltx25MultiGgufStateDictLoader((str(sft), str(te)))
    assert isinstance(loader._loaders[0], Ltx25SftConnectorLoader)
    meta = loader.metadata()
    assert meta["config"]["transformer"]["num_layers"] == N_BLOCKS
    assert meta["gemma_source_checkpoint"] == GEMMA_SOURCE

    sd = loader.load((str(sft), str(te)), sd_ops=LTX25_EMBEDDINGS_PROCESSOR_KEY_OPS, device=CPU).sd
    expected_connectors = {
        key.replace("video_embeddings_connector.", "video_connector.")
        .replace("audio_embeddings_connector.", "audio_connector."): value
        for key, value in connectors.items()
    }
    assert set(sd) == set(expected_connectors) | set(projections)
    for key, value in expected_connectors.items():
        assert sd[key].dtype == BF16
        assert torch.equal(sd[key], value), key
    for key, value in projections.items():
        assert torch.equal(sd[key], value), key


def test_embeddings_loader_refuses_overlapping_files(tmp_path):
    sft = tmp_path / "t.safetensors"
    _write_quant(sft, prefix=P)
    loader = Ltx25MultiGgufStateDictLoader((str(sft), str(sft)))
    with pytest.raises(Ltx25GemmaError, match=r"t\.safetensors redefines"):
        loader.load((str(sft), str(sft)), sd_ops=LTX25_EMBEDDINGS_PROCESSOR_KEY_OPS, device=CPU)


# --------------------------------------------------------------------------- #
# the ancestral-sampler probe and the pipeline
# --------------------------------------------------------------------------- #


def test_ancestral_detection_skipped_rebinds_and_restores():
    original = ltx_distilled.should_use_ancestral_sampler
    # The constructor reads the name from this very namespace.
    assert DistilledPipeline.__init__.__globals__ is vars(ltx_distilled)
    with ancestral_detection_skipped():
        assert ltx_distilled.should_use_ancestral_sampler("S:/never/opened.safetensors") is True
    assert ltx_distilled.should_use_ancestral_sampler is original

    with pytest.raises(RuntimeError, match="boom"):
        with ancestral_detection_skipped():
            raise RuntimeError("boom")
    assert ltx_distilled.should_use_ancestral_sampler is original


def test_constructor_calls_the_probe_by_global_name_and_verify_pins_it():
    assert "should_use_ancestral_sampler" in DistilledPipeline.__init__.__code__.co_names
    ltxcore_compat.verify()


class _Stop(Exception):
    pass


@pytest.mark.parametrize(("suffix", "expected"), [(".safetensors", "from_safetensors"), (".gguf", "from_gguf")])
def test_pipeline_picks_the_loader_by_extension_and_reports_detected_none(
    tmp_path, monkeypatch, suffix, expected
):
    files = {}
    for name, filename in (
        ("transformer", "transformer" + suffix),
        ("text_encoder", "te.gguf"),
        ("video_vae", "vv.safetensors"),
        ("audio_vae", "av.safetensors"),
        ("spatial_upsampler", "up.safetensors"),
    ):
        (tmp_path / filename).write_bytes(b"")
        files[name] = str(tmp_path / filename)

    monkeypatch.setattr(
        pipeline25.assets_export,
        "ensure_assets_only",
        lambda *_a, **_k: types.SimpleNamespace(path=tmp_path / "assets.safetensors", as_dict=dict),
    )
    probes = []

    class _FakeDistilled:
        def __init__(self, *, model_paths, **_kwargs):
            # What the real constructor does, through the same module global.
            probes.append(ltx_distilled.should_use_ancestral_sampler(model_paths.transformer()))
            self.use_ancestral_sampler = probes[-1]

    monkeypatch.setattr(pipeline25, "DistilledPipeline", _FakeDistilled)
    called = []

    def _stage(name):
        def build(path, **kwargs):
            called.append((name, path, sorted(kwargs)))
            raise _Stop

        return staticmethod(build)

    monkeypatch.setattr(pipeline25.Ltx25ProgressStage, "from_safetensors", _stage("from_safetensors"))
    monkeypatch.setattr(pipeline25.Ltx25ProgressStage, "from_gguf", _stage("from_gguf"))

    pipe = object.__new__(pipeline25.Ltx25Pipeline)
    with pytest.raises(_Stop):
        pipeline25.Ltx25Pipeline.__init__(
            pipe, pipeline25.ModelFiles(**files), device=CPU, deterministic=False
        )

    assert probes == [True]  # the rebound probe; the file (empty) was never opened
    assert ltx_distilled.should_use_ancestral_sampler is ltxcore_compat.should_use_ancestral_sampler
    assert pipe.build_report["use_ancestral_sampler"] == {"detected": None, "forced": True}
    assert called == [
        (expected, files["transformer"], ["blocks_on_gpu", "cache_weights", "device", "dtype"])
    ]
