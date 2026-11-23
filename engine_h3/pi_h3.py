"""pi-H3 on the DiT: the non-LoRA part of the difference, and the initial state.

pi-H3 (Docs/MINIMAX_H3_RESEARCH_NOTES.md §18; the experiment and the owner's
rulings: Docs/VERIFICATION_LOG.md §223) is applied without re-quantizing
anything (§223.3). This module holds what the experiment's script did by
replacing product functions at run time, moved here unchanged:

* :func:`apply_adapter` -- right after ``H3Pipeline._build_dit`` (CPU DiT):
  - blocks 46-49 ``attn.out_proj`` / ``mlp.fc1`` / ``mlp.fc2``: on a quantized
    layer (any ``H3QuantLinear``: int8, fp8 or w4a8) the file's int8 ``weight``
    and F32 ``weight_scale`` go in as stored -- into the layer itself when it is
    an ``H3Int8Linear``, else into a new ``H3Int8Linear`` that takes its place;
    on a float ``nn.Linear`` (Sulphur-3's block 49) the weight is dequantized to
    float32 (``engine.sft_quant.dequant``) and cast to the layer's dtype;
  - every block's ``adaln_proj.linear.weight``, blocks 46-49's bias and the
    ``final_layer`` tensors: the file's values cast to the layer's dtype; a
    quantized AdaLN layer becomes a float32 ``nn.Linear`` with the file's
    weight (and the file's bias on 46-49, the layer's own elsewhere);
  - every cast that changed a value is recorded (``rounding``).
  The LoRA part (blocks 0-45) is the product's LoRA path (``engine_h3.lora``),
  attached by the pipeline as one more file at strength 1.0; the ``dwd_`` part
  is ``engine_h3.dwd``'s.
* :func:`initial_state` -- after the masks are built: the state times the
  first sigma (ComfyUI's ``noise * sigma_0``), and on the rows whose mask value
  is strictly between 0 and 1, ``(1 - sigma_0) * clean`` added.

The DiT keeps which pi-H3 it carries (:func:`applied_tag`), so a resident DiT
is reused only for the same choice (the pipeline rebuilds it otherwise; there
is no undo).
"""

from __future__ import annotations

import collections
import time
from pathlib import Path

import torch
import torch.nn as nn

import h3_pih3_format as PF
from engine_h3.quant_linear import H3Int8Linear, H3QuantLinear

#: The DiT attribute naming the pi-H3 applied to it (None / absent = none).
_TAG = "_pi_h3_applied"


def applied_tag(dit) -> str | None:
    """``"4steps"`` / ``"2steps"`` for a DiT :func:`apply_adapter` changed, else None."""
    return None if dit is None else getattr(dit, _TAG, None)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _cast(value: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, dict]:
    out = value.to(dtype).contiguous()
    exact = bool(torch.equal(out.to(value.dtype), value))
    rec = {"from": _dtype_name(value.dtype), "to": _dtype_name(dtype), "exact": exact}
    if not exact:
        rec["changed_elements"] = int((out.to(value.dtype) != value).sum())
        rec["max_abs_change"] = float((out.to(torch.float64) - value.to(torch.float64)).abs().max())
    return out, rec


def _set_param(module: nn.Module, leaf: str, value: torch.Tensor) -> None:
    setattr(module, leaf, nn.Parameter(value, requires_grad=False))


def _set_parent(dit: nn.Module, name: str, new: nn.Module) -> None:
    parent, _, leaf = name.rpartition(".")
    setattr(dit.get_submodule(parent), leaf, new)


@torch.no_grad()
def apply_adapter(dit: nn.Module, path, choice: str, *, log=None) -> dict:
    """Apply the non-LoRA, non-DWD part of the pi-H3 file ``path`` to ``dit`` (CPU); returns the record.

    ``choice``: ``"4steps"`` / ``"2steps"``, kept on the DiT (:func:`applied_tag`).
    The file is checked first (``h3_pih3_format.inspect_file``); a layer the
    file does not fit raises ``RuntimeError``.
    """
    from engine.sft_quant.dequant import dequantize, normalize_aux  # noqa: PLC0415
    from engine.sft_quant.sft_reader import read_tensors  # noqa: PLC0415

    started = time.perf_counter()
    info = PF.inspect_file(path)
    keys = info.keys
    names = [k for leaves in keys.quant.values() for k in (leaves["weight"], leaves["weight_scale"])]
    names += list(keys.adaln_w.values()) + list(keys.adaln_b.values()) + keys.final
    t = dict(read_tensors(str(path), names, header=info.header))
    conv: collections.Counter = collections.Counter()
    rounding: list[dict] = []
    rep: dict = {"quant_layers": {}, "adaln": collections.Counter(), "final": {}}

    def note(where: str, rec: dict) -> None:
        conv[f"{where}: {rec['from']}->{rec['to']}{'' if rec['exact'] else ' (rounded)'}"] += 1
        if not rec["exact"]:
            rounding.append({"tensor": where, **rec})

    for layer, leaves in sorted(keys.quant.items()):
        mod = dit.get_submodule(layer)
        w, s = t[leaves["weight"]], t[leaves["weight_scale"]]
        spec = info.quant_specs[layer]
        if w.dtype != torch.int8 or s.dtype != torch.float32:
            raise RuntimeError(f"pi-H3 {layer}: {w.dtype} / {s.dtype}（int8 / float32 のはず）")
        if isinstance(mod, H3QuantLinear):
            if (int(mod.out_features), int(mod.in_features)) != tuple(w.shape):
                raise RuntimeError(f"pi-H3 {layer}: {list(w.shape)} が層 {mod.out_features}x{mod.in_features} と合いません")
            if isinstance(mod, H3Int8Linear):
                target, into = mod, "H3Int8Linear (values replaced)"
                spec_changed = mod.spec != spec
            else:
                # fp8 / w4a8 (no bias: quant_linear refuses a biased fp8 / w4a8 layer)
                target = H3Int8Linear(int(mod.in_features), int(mod.out_features), bias=False, spec=spec)
                into, spec_changed = f"H3Int8Linear (replaces {type(mod).__name__})", True
            _set_param(target, "weight", w.contiguous())
            target.weight_scale = s.reshape(-1, 1).contiguous()
            target.spec = spec
            if target is not mod:
                target.requires_grad_(False)
                _set_parent(dit, layer, target)
            rep["quant_layers"][layer] = {
                "into": into, "as_stored": "I8 weight + F32 weight_scale",
                "spec": f"{spec.format}/convrot={spec.convrot}/g{spec.convrot_groupsize}",
                "spec_changed": spec_changed,
            }
            conv[f"quant: I8+F32 -> {into}"] += 1
        elif isinstance(mod, nn.Linear):
            o, i = (int(v) for v in w.shape)
            if (mod.out_features, mod.in_features) != (o, i):
                raise RuntimeError(f"pi-H3 {layer}: {list(w.shape)} が nn.Linear {mod.out_features}x{mod.in_features} と合いません")
            scheme = "int8_convrot" if spec.convrot else "int8"
            aux = {"weight_scale": normalize_aux(scheme, "weight_scale", s, o, i)}
            deq = dequantize(scheme, w, aux, torch.float32)
            value, rec = _cast(deq, mod.weight.dtype)
            _set_param(mod, "weight", value)
            rep["quant_layers"][layer] = {"into": f"nn.Linear {rec['to']}", "dequantized": scheme, **rec}
            note(f"quant dequantized (f32) -> {layer}", rec)
        else:
            raise RuntimeError(f"pi-H3 {layer}: {type(mod).__name__} は Linear ではありません")
    for i in range(PF.NUM_BLOCKS):
        proj = dit.blocks[i].adaln_proj
        lin = proj.linear
        w = t[keys.adaln_w[i]]
        b_pi = t[keys.adaln_b[i]] if i in keys.adaln_b else None
        if isinstance(lin, H3QuantLinear):
            bias_src = b_pi if b_pi is not None else (None if lin.bias is None else lin.bias.detach())
            if bias_src is None:
                raise RuntimeError(f"pi-H3 blocks.{i}.adaln_proj.linear: 偏りがありません")
            new = nn.Linear(int(lin.in_features), int(lin.out_features), bias=True, dtype=torch.float32, device="cpu")
            wv, wr = _cast(w, torch.float32)
            bv, br = _cast(bias_src, torch.float32)
            _set_param(new, "weight", wv)
            _set_param(new, "bias", bv)
            new.requires_grad_(False)
            proj.linear = new
            rep["adaln"][f"{type(lin).__name__} -> float32 nn.Linear"] += 1
            note("adaln weight (into a new float32 nn.Linear)", wr)
            note(f"adaln bias ({'pi-H3' if b_pi is not None else 'base'}) (into the new nn.Linear)", br)
        elif isinstance(lin, nn.Linear):
            if tuple(lin.weight.shape) != tuple(w.shape):
                raise RuntimeError(f"pi-H3 blocks.{i}.adaln_proj.linear: {list(w.shape)} が {list(lin.weight.shape)} と合いません")
            wv, wr = _cast(w, lin.weight.dtype)
            _set_param(lin, "weight", wv)
            note("adaln weight", wr)
            rep["adaln"][f"set into nn.Linear {wr['to']}"] += 1
            if b_pi is not None:
                bv, br = _cast(b_pi, lin.bias.dtype)
                _set_param(lin, "bias", bv)
                note("adaln bias (pi-H3)", br)
        else:
            raise RuntimeError(f"pi-H3 blocks.{i}.adaln_proj.linear: {type(lin).__name__}")
    for k in keys.final:
        parent, _, leaf = k.rpartition(".")
        mod = dit.get_submodule(parent)
        old = getattr(mod, leaf)
        value, rec = _cast(t[k], old.dtype)
        if tuple(value.shape) != tuple(old.shape):
            raise RuntimeError(f"pi-H3 {k}: {list(value.shape)} が {list(old.shape)} と合いません")
        _set_param(mod, leaf, value)
        rep["final"][k] = rec
        note(k, rec)
    del t
    setattr(dit, _TAG, choice)
    rep["adaln"] = dict(rep["adaln"])
    rep["counts"] = {"quant_layers": len(rep["quant_layers"]), "adaln_weights": PF.NUM_BLOCKS,
                     "adaln_biases": len(keys.adaln_b), "final_tensors": len(rep["final"])}
    rep["conversions"] = dict(conv)
    rep["rounding"] = rounding
    rep["seconds"] = round(time.perf_counter() - started, 2)
    if log is not None:
        log(f"pi-H3 adapter ({Path(str(path)).name}): {rep['counts']} rounding={len(rounding)} "
            f"in {rep['seconds']}s")
    return rep


def partial_rows(mask) -> torch.Tensor | None:
    """bool, broadcastable like ``mask.m``: the rows whose value is strictly between 0 and 1 (None without a mask)."""
    if mask is None:
        return None
    return (mask.m > 0) & (mask.m < 1)


def initial_state(x: torch.Tensor, mask, sigma0: float) -> torch.Tensor:
    """The state a pi-H3 denoise starts from: ``x * sigma0``, plus ``(1 - sigma0) * clean`` on the partial rows.

    ``x``: the stream's initial noise (the state, float32); ``mask``: its
    ``denoise_mask.StreamMask`` or None. Rows with mask value 0 or 1 (and every
    row without a mask) get the scaling only -- the experiment's whole-state
    scaling (VERIFICATION_LOG §223.4); a row with ``0 < m < 1`` then starts at
    ``sigma0 * noise + (1 - sigma0) * clean``.
    """
    s = float(sigma0)
    out = x * s
    part = partial_rows(mask)
    if part is None or not bool(part.any()):
        return out
    return torch.where(part, out + (1.0 - s) * mask.clean, out)
