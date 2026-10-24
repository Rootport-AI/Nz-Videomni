"""MiniMax H3 LoRA at run time: a low-rank path added to the output, in float32 (台帳 §1-104).

What it computes, per target Linear (``blocks.{i}`` and ``token_refiner.blocks.{j}``
x ``attn.qkv_proj`` / ``attn.out_proj`` / ``mlp.fc1`` / ``mlp.fc2``)::

    y = bf16( f32(base(x)) + sum_k s_k * (f32(x) @ A_k^T) @ B_k^T ),   s_k = strength_k * alpha_k / rank_k

(1.0 instead of ``alpha / rank`` when the file has no alpha). The int8 weights
are never touched; A and B stay on the module as CPU buffers in the file's
dtype (BF16 / F16 / F32) and are lifted to float32 at each call (exact). The
LoRA reads the RAW ``x`` (ConvRot's rotation lives inside the int8 kernel).

How it is wired (Docs/MINIMAX_H3_DESIGN.md §6.6):

* ``attach`` registers ``_lora_A_{k}`` / ``_lora_B_{k}`` as NON-persistent
  buffers on each target (so block swap, the prefetch arenas, ``_park_dit``
  and ``drop_to_meta`` carry them with the block, and the state dict does not
  change), a ``_lora_specs`` list, and ONE forward hook per module that calls
  :func:`apply` on every ``Module.__call__``;
* ``H3Int8Linear.forward_rows`` (VRAM 16GB mode's qkv, which does not go
  through ``__call__``) calls :func:`apply` itself with its ``[n0, n1)`` rows of
  B, and ``lowvram.block_forward`` computes ``f32(h) @ A^T`` once per block
  (:func:`xa_for`) instead of once per head group;
* ``detach`` removes all of it (``delattr``, so ``_non_persistent_buffers_set``
  is cleaned too, and ``handle.remove()``). It MUST run after the prefetch
  teardown: ``_release`` writes ``mod._buffers[name]`` back by name.

Same shapes on both paths (bit identity off == VRAM 16GB mode by construction):
rows in ``lowvram.row_chunks(M, VRAM16_ROW_CHUNK)``, output columns in
``LORA_COL_CHUNK`` (= one head group's q / k / v rows of the stacked qkv
weight). The row chunks read ``constants.VRAM16_ROW_CHUNK``, not the job's
``Vram16Plan.row_chunk``, so the two must stay equal for off == on (the tests
that shrink one shrink both). A job without LoRA never reaches this module: no
buffer, no hook.

No ``except`` in :func:`apply`: a failure is a failed job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

import h3_lora_format
import sft_quant_format
from engine_h3 import constants as _C
from engine_h3.quant_linear import H3Int8Linear

_TORCH_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
assert tuple(_TORCH_DTYPES) == h3_lora_format.H3_LORA_DTYPES   # one list of accepted dtypes

_SPECS = "_lora_specs"
_HOOK = "_lora_hook"


@dataclass
class H3LoraFile:
    """One LoRA file read into CPU tensors (the file's dtype)."""

    name: str                       # file name only (the report goes into the mp4's metadata)
    layout: str                     # h3_lora_format.H3LoraKeyMap.layout
    dtype: str                      # "BF16" / "F16" / "F32", or "+"-joined when mixed, "" when no pair
    #: canonical module name -> (A [r, in], B [out, r], alpha / r or 1.0)
    modules: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = field(default_factory=dict)
    skipped: list[dict] = field(default_factory=list)   # [{"module", "reason"}]


@dataclass(frozen=True)
class _Spec:
    k: int
    a: str              # buffer names on the module
    b: str
    scale: float          # strength * alpha / rank
    dtype_a: torch.dtype  # the file's dtypes; the buffers must still hold them at every call
    dtype_b: torch.dtype
    rank: int


# --------------------------------------------------------------------------- #
# the file
# --------------------------------------------------------------------------- #

def _header(path):
    p = Path(str(path or ""))
    if not str(path or "").strip() or not p.is_file():
        raise ValueError(f"LoRA のファイルがありません（{path}）")
    try:
        return sft_quant_format.read_header(p)
    except (sft_quant_format.QuantFormatError, OSError) as exc:
        raise ValueError(f"LoRA のファイルを読めません（{p.name}）: {exc}") from exc


def _key_map(header) -> h3_lora_format.H3LoraKeyMap:
    t = header.tensors
    return h3_lora_format.map_lora_keys(
        t, {k: v.shape for k, v in t.items()}, {k: v.dtype for k, v in t.items()}
    )


def inspect_lora_file(path) -> h3_lora_format.H3LoraKeyMap:
    """The key map of ``path`` from its header only; a missing / unreadable file is a Japanese ValueError."""
    return _key_map(_header(path))


def load_lora_file(path) -> H3LoraFile:
    """Read the A / B (and alpha) tensors of every mapped pair (CPU, the file's dtype).

    A pair whose A or B is not BF16 / F16 / F32, or whose alpha is not a
    one-element BF16 / F16 / F32 tensor, is skipped and recorded (owner ruling 2:
    an odd file still runs). Shapes are checked against the model by ``attach``.
    """
    from engine.sft_quant.sft_reader import read_tensors  # noqa: PLC0415

    header = _header(path)
    kmap = _key_map(header)
    info = header.tensors
    out = H3LoraFile(name=Path(str(path)).name, layout=kmap.layout, dtype="")
    if kmap.unmapped:
        out.skipped.append({
            "module": "(unmapped keys)",
            "reason": f"{len(kmap.unmapped)} key(s) outside the H3 targets or without a partner, e.g. {kmap.unmapped[0]}",
        })
    wanted: dict[str, h3_lora_format.LoraPairKeys] = {}
    for module, keys in kmap.pairs.items():
        bad = [info[k].dtype for k in (keys.a, keys.b) if info[k].dtype not in _TORCH_DTYPES]
        if bad:
            out.skipped.append({"module": module, "reason": f"dtype {bad[0]} (BF16 / F16 / F32 only)"})
            continue
        if keys.alpha is not None:
            a_info = info[keys.alpha]
            numel = 1
            for d in a_info.shape:
                numel *= d
            if a_info.dtype not in _TORCH_DTYPES or numel != 1:
                out.skipped.append({
                    "module": module,
                    "reason": f"alpha {a_info.dtype}{list(a_info.shape)} is not a one-element float",
                })
                continue
        wanted[module] = keys
    names = [k for keys in wanted.values() for k in (keys.a, keys.b, keys.alpha) if k is not None]
    tensors = dict(read_tensors(str(path), names, header=header))
    dtypes: set[str] = set()
    for module, keys in wanted.items():
        a, b = tensors[keys.a], tensors[keys.b]
        rank = int(a.shape[0]) if a.dim() >= 1 else 0
        if keys.alpha is not None and rank > 0:
            scale = float(tensors[keys.alpha].float().reshape(-1)[0].item()) / rank
        else:
            scale = 1.0
        out.modules[module] = (a, b, scale)
        dtypes |= {info[keys.a].dtype, info[keys.b].dtype}
    out.dtype = "+".join(sorted(dtypes))
    return out


# --------------------------------------------------------------------------- #
# attach / detach
# --------------------------------------------------------------------------- #

def _hook(module, args, out):
    return apply(module, args[0], out)


def attach(dit: nn.Module, files) -> list[dict]:
    """Attach every ``(H3LoraFile, strength)`` to ``dit``; returns one report per file.

    A pair whose module is not in the model, is not an ``H3Int8Linear`` /
    ``nn.Linear``, or whose shapes are not ``A [r, in]`` / ``B [out, r]`` is
    skipped and recorded. Buffers keep the file's dtype and device (CPU).
    """
    reports = []
    for lora_file, strength in files:
        report = {
            "name": lora_file.name,
            "strength": float(strength),
            "layout": lora_file.layout,
            "dtype": lora_file.dtype,
            "pairs_applied": 0,
            "skipped": list(lora_file.skipped),
        }
        for module_name, (a, b, alpha_scale) in lora_file.modules.items():
            try:
                module = dit.get_submodule(module_name)
            except AttributeError:
                report["skipped"].append({"module": module_name, "reason": "not in this model"})
                continue
            if not isinstance(module, (H3Int8Linear, nn.Linear)):
                report["skipped"].append({"module": module_name, "reason": f"not a Linear ({type(module).__name__})"})
                continue
            r = int(a.shape[0]) if a.dim() == 2 else -1
            if (b.dim() != 2 or r < 1
                    or tuple(a.shape) != (r, int(module.in_features))
                    or tuple(b.shape) != (int(module.out_features), r)):
                report["skipped"].append({
                    "module": module_name,
                    "reason": (
                        f"shape A {list(a.shape)} / B {list(b.shape)} does not fit "
                        f"in {module.in_features} -> out {module.out_features}"
                    ),
                })
                continue
            specs = getattr(module, _SPECS, None)
            if specs is None:
                specs = []
                setattr(module, _SPECS, specs)
                setattr(module, _HOOK, module.register_forward_hook(_hook))
            k = len(specs)
            spec = _Spec(k=k, a=f"_lora_A_{k}", b=f"_lora_B_{k}",
                         scale=float(strength) * float(alpha_scale), dtype_a=a.dtype, dtype_b=b.dtype, rank=r)
            module.register_buffer(spec.a, a, persistent=False)
            module.register_buffer(spec.b, b, persistent=False)
            specs.append(spec)
            report["pairs_applied"] += 1
        reports.append(report)
    return reports


def detach(dit: nn.Module) -> int:
    """Remove every LoRA buffer, spec list and hook from ``dit``; returns the modules cleaned.

    Idempotent; works on meta / CPU / GPU tensors alike (only Python state is
    touched). Must run AFTER the block swap prefetch teardown (see the module
    docstring).
    """
    cleaned = 0
    for module in dit.modules():
        specs = module.__dict__.get(_SPECS)
        if specs is None:
            continue
        for spec in specs:
            for name in (spec.a, spec.b):
                if name in module._buffers:
                    delattr(module, name)
        handle = module.__dict__.get(_HOOK)
        if handle is not None:
            handle.remove()
            del module.__dict__[_HOOK]
        del module.__dict__[_SPECS]
        cleaned += 1
    return cleaned


# --------------------------------------------------------------------------- #
# the maths
# --------------------------------------------------------------------------- #

def _buffer(module, name: str, dtype: torch.dtype) -> torch.Tensor:
    """The buffer ``name``, looked up by name at every call (block swap rebinds it)."""
    t = getattr(module, name)
    if t.dtype != dtype:
        raise RuntimeError(f"LoRA buffer {name} is {t.dtype}, attached as {dtype}")
    return t


def xa_for(module, x: torch.Tensor):
    """``[f32(x) @ A_k^T]`` per spec (unscaled, ``[M, r_k]`` float32), or None without LoRA.

    Row chunks as in :func:`apply`, so the vram16 path hands ``apply`` the same
    numbers the off path computes inside it.
    """
    specs = getattr(module, _SPECS, None)
    if not specs:
        return None
    from engine_h3.lowvram import row_chunks  # noqa: PLC0415 -- lowvram imports this module

    x2d = x.reshape(-1, x.shape[-1])
    m = x2d.shape[0]
    chunks = row_chunks(m, _C.VRAM16_ROW_CHUNK)
    out = []
    for spec in specs:
        a = _buffer(module, spec.a, spec.dtype_a).float()
        xa = torch.empty((m, spec.rank), dtype=torch.float32, device=x2d.device)
        for r0, r1 in chunks:
            xa[r0:r1] = torch.mm(x2d[r0:r1].float(), a.t())
        out.append(xa)
    return out


def apply(module, x: torch.Tensor, out: torch.Tensor, n0: int = 0, n1: int | None = None, xa=None) -> torch.Tensor:
    """Add the LoRA of ``module`` to ``out`` in place and return it.

    ``out`` holds output features ``[n0, n1)`` (``forward_rows``) or all of them;
    ``xa``: :func:`xa_for`'s result for this ``x`` (VRAM 16GB mode), else it is
    computed here. Every matmul operand is float32; the f32 sum is added into
    the bf16 ``out`` once per element (the in-place add computes in f32 and
    rounds once).
    """
    from engine_h3.lowvram import row_chunks  # noqa: PLC0415 -- lowvram imports this module

    specs = getattr(module, _SPECS)
    x2d = x.reshape(-1, x.shape[-1])
    out2d = out.view(-1, out.shape[-1])
    if n1 is None:
        n1 = n0 + out2d.shape[1]
    if out2d.shape[1] != n1 - n0 or out2d.shape[0] != x2d.shape[0]:
        raise RuntimeError(f"LoRA: output {list(out2d.shape)} does not match rows {x2d.shape[0]} x [{n0}, {n1})")
    col = int(_C.LORA_COL_CHUNK)
    a_s = None if xa is not None else [_buffer(module, s.a, s.dtype_a).float() for s in specs]
    b_s = [_buffer(module, s.b, s.dtype_b) for s in specs]
    for r0, r1 in row_chunks(x2d.shape[0], _C.VRAM16_ROW_CHUNK):
        if xa is not None:
            xs = [xa[i][r0:r1] * s.scale for i, s in enumerate(specs)]
        else:
            xr = x2d[r0:r1].float()
            xs = [torch.mm(xr, a_s[i].t()) * s.scale for i, s in enumerate(specs)]
        for c0 in range(n0, n1, col):
            c1 = min(c0 + col, n1)
            delta = torch.mm(xs[0], b_s[0][c0:c1].float().t())
            for i in range(1, len(specs)):
                delta.addmm_(xs[i], b_s[i][c0:c1].float().t())
            out2d[r0:r1, c0 - n0:c1 - n0].add_(delta)
    return out
