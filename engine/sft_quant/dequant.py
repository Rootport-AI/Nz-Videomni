"""Dequantization of a quantized safetensors Linear weight (§3-168 C-1b).

The GGUF side's ``dequantize_ggml_tensor`` counterpart for the quantization
schemes ``sft_quant_format`` accepts (its ``SCHEMES``): one ``if``/``elif`` per
scheme, computed in float32 and cast to the activation dtype. Used by the
``sft_quant_linear`` forward (every call) and by ``load_connector_bf16`` (once,
on the CPU).

  * :func:`dequantize` — the weight in ``dtype`` from the stored weight and its
    auxiliary tensors (already normalized).
  * :func:`normalize_aux` — an auxiliary tensor as read from the file, brought
    to the dtype and shape ``sft_quant_format.aux_specs`` prescribes (the loader
    and the connector share it, so the forward sees one shape per scheme).
  * :func:`hadamard` — the 256 x 256 ConvRot matrix, cached per device.

The int8 / ConvRot / w4a8 maths follows the converter's NumPy implementation
(``Nz-GGUF-Converter-LTX23/src/converter/comfy_dequant.py``, the cross-check
reference; not imported): ``q.float() * scale`` per output row, then each
contiguous group of 256 input columns times ``H``. w4a8 (§3-168 C-3) first
unpacks two 4-bit codes per byte (even column in the low nibble), looks them
up in the layer's 16-entry codebook, multiplies each group of 16 columns by
its ``s_rel``, rounds onto the int8 grid (round half to even, clamp +/-127),
multiplies by ``s_channel`` per row and rotates like ConvRot. No row chunking:
the intermediates are released in order so the peak for the largest layer
([16384, 4096]) stays at two float32-sized copies (~512 MiB), as for fp8
today. w4a8 too: the int32 codes plus the float32 lookup, then the lookup
plus the rotated copy (``index_select`` reads the int32 index as is;
``codebook[idx]`` would first copy it to int64, doubling the peak). CPU
profiler, [4096, 4096]: 128 MiB = two float32 copies.

Depends on torch and the torch-free ``sft_quant_format`` only.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

import sft_quant_format

#: ConvRot group size (the only one ``sft_quant_format`` accepts).
CONVROT_GROUPSIZE = 256
#: The int8 grid w4a8's intermediate is rounded onto.
_INT8_GRID = 127.0

#: The 4 x 4 regular Hadamard seed ComfyUI's ConvRot is built from (symmetric,
#: -1 on the anti-diagonal — not the Sylvester H4), the converter's ``_H4``.
_H4 = (
    (1.0, 1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0, 1.0),
)

_HADAMARD: dict[torch.device, torch.Tensor] = {}


def hadamard(device: torch.device | str) -> torch.Tensor:
    """The normalized 256 x 256 Hadamard matrix in float32 on ``device``.

    Four Kronecker powers of :data:`_H4` divided by 16 (= sqrt(256), exact in
    float32): symmetric, orthogonal and involutory (``H @ H == I``), every
    element ``+/- 1/16``. Built outside inference mode so the cached tensor is
    an ordinary one, usable from any later mode. Cached per device.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    h = _HADAMARD.get(device)
    if h is None:
        with torch.inference_mode(False):
            h4 = torch.tensor(_H4, dtype=torch.float32)
            m = h4
            while m.shape[0] < CONVROT_GROUPSIZE:
                m = torch.kron(m, h4)
            h = (m / 16.0).to(device)
        _HADAMARD[device] = h
    return h


def dequantize(
    scheme: str | None,
    weight: torch.Tensor,
    aux: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """``weight`` (stored form) back in ``dtype``; ``aux`` holds its normalized
    auxiliary tensors by name (``weight_scale`` ...). ``scheme`` None is a
    non-quantized layer (a plain cast)."""
    if scheme is None or scheme == "fp8":
        return weight.to(dtype)
    if scheme == "fp8_scaled":
        # The §3-167 expression, unchanged (bit-identical fp8 output).
        return (weight.to(torch.float32) * aux["weight_scale"]).to(dtype)
    if scheme == "int8":
        wf = weight.to(torch.float32)
        wf.mul_(aux["weight_scale"])  # (o, 1): one scale per output row
        return wf.to(dtype)
    if scheme == "int8_convrot":
        o, i = weight.shape
        wf = weight.to(torch.float32)
        wf.mul_(aux["weight_scale"])
        rotated = torch.matmul(
            wf.view(o, i // CONVROT_GROUPSIZE, CONVROT_GROUPSIZE), hadamard(weight.device)
        )
        del wf
        return rotated.view(o, i).to(dtype)
    if scheme == "w4a8":
        o, half = weight.shape
        i = half * 2
        # Two 4-bit codes per byte: element 2k in bits 0-3, 2k+1 in bits 4-7.
        p = weight.view(torch.uint8)
        codes = torch.stack((p & 0x0F, p >> 4), dim=-1).view(o, i)
        idx = codes.to(torch.int32)
        del codes
        # index_select takes the int32 index as is; codebook[idx] would make an
        # int64 copy of it first (+2 float32 copies of peak).
        v = aux["weight_codebook"].index_select(0, idx.view(-1)).view(o, i)  # float32
        del idx
        # Per group of 16 columns: x s_rel, then onto the int8 grid.
        gs = sft_quant_format.W4A8_GROUP_SIZE
        g = v.view(o, i // gs, gs)
        g.mul_(aux["weight_s_rel"].to(torch.float32).unsqueeze(-1))
        g.round_()  # half to even, as np.rint
        g.clamp_(-_INT8_GRID, _INT8_GRID)
        v.mul_(aux["weight_s_channel"].unsqueeze(1))
        rotated = torch.matmul(
            v.view(o, i // CONVROT_GROUPSIZE, CONVROT_GROUPSIZE), hadamard(weight.device)
        )
        del g, v
        return rotated.view(o, i).to(dtype)
    raise RuntimeError(f"quantized safetensors: unknown scheme {scheme!r}")


def normalize_aux(
    scheme: str, leaf: str, value: torch.Tensor, o: int, i: int
) -> torch.Tensor:
    """An auxiliary tensor of a ``scheme`` layer with weight shape ``(o, i)``,
    brought to the dtype and shape ``sft_quant_format.aux_specs`` prescribes.

    Shapes: ``"(o,1)"`` — a scalar or one value per row, ``(o, 1)`` either
    way (the int8 scale); every other rule — ``"()"`` (the fp8 scale, as
    §3-167), w4a8's ``"(o,)"`` / ``"(16,)"`` / ``"(o,i/16)"`` — a reshape to
    ``sft_quant_format.aux_shape`` (which raises on an unknown rule). A U8 ``weight_s_rel`` is
    the same bytes as F8_E4M3 and is read as such (``view``). A value of any
    other dtype raises: the ``inspect`` check has already ruled on what the
    file may hold.
    """
    from engine.sft_quant.sft_reader import TORCH_DTYPES

    specs = sft_quant_format.aux_specs(scheme)
    if leaf not in specs:
        raise RuntimeError(
            f"quantized safetensors: '{leaf}' is not an auxiliary tensor of scheme {scheme!r}"
        )
    dtype_name, rule = specs[leaf]
    dtype = TORCH_DTYPES[dtype_name]
    if leaf == "weight_s_rel" and value.dtype == torch.uint8:
        value = value.view(dtype)  # byte-identical reinterpretation
    if value.dtype != dtype:
        raise RuntimeError(
            f"quantized safetensors: '{leaf}' is {value.dtype}, expected {dtype}"
        )
    if rule == "()":
        return value.reshape(())
    if rule == "(o,1)":
        return value.reshape(-1, 1).expand(o, 1).contiguous()
    return value.reshape(sft_quant_format.aux_shape(rule, o, i))
