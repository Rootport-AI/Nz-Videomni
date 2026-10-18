"""Stream a safetensors file into a model that was built on the meta device.

Why this shape (plan section 1, review C-1): building the DiT or the text
encoder the ordinary way would first allocate them at full precision -- about
77 GB of float32 for the DiT and 49 GB of bf16 for the TE -- before a single
weight is read, which is more than the 64 GB machine has. So both are built
under ``with torch.device("meta")``, the int8 Linears are swapped in (still on
meta), and this module fills them:

* every tensor is read on its own with ``engine.sft_quant.sft_reader.read_tensors``
  (seek + readinto into a fresh buffer; no memory map, which on Windows would be
  charged to the commit limit for the whole file);
* the tensors are collected into one dict and handed to
  ``load_state_dict(strict=False, assign=True)``, so the module ADOPTS the
  read buffers -- nothing is copied and nothing is allocated twice. ``assign``
  also keeps the file's dtype per tensor (int8 / bf16 / f16 / f32), which is
  what the precision islands of the H3 DiT rely on;
* afterwards no parameter or buffer may still be on meta. One that is would
  fail much later as an obscure "Cannot copy out of meta tensor" in the middle
  of a forward, so it is refused here with every name listed.

Names the model does not have (``unexpected keys``) are logged and dropped, as
are the keys ``key_map`` maps to ``None`` and those with a skipped suffix
(``.comfy_quant``: the quantization marker is read from the header by
``h3_format``, and the spec lives as a module attribute, not as a tensor).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from engine.sft_quant.sft_reader import read_tensors

logger = logging.getLogger(__name__)


@dataclass
class LoadReport:
    loaded: int
    skipped: tuple[str, ...]
    seconds: float
    bytes: int


def meta_tensor_names(model: nn.Module) -> list[str]:
    """Every parameter / buffer (persistent or not) that is still on the meta device."""
    names = [n for n, p in model.named_parameters(remove_duplicate=False) if p.is_meta]
    names += [n for n, b in model.named_buffers(remove_duplicate=False) if b is not None and b.is_meta]
    return names


def load_streaming(
    model: nn.Module,
    path: str,
    header,
    *,
    key_map: Callable[[str], str | None],
    skip_suffixes: tuple[str, ...] = (".comfy_quant",),
) -> LoadReport:
    """Fill a meta-built ``model`` from ``path``; returns what was loaded and skipped."""
    started = time.perf_counter()
    mapping: dict[str, str] = {}
    targets: set[str] = set()
    skipped: list[str] = []
    for key in header.tensors:
        if key.endswith(tuple(skip_suffixes)):
            skipped.append(key)
            continue
        target = key_map(key)
        if target is None:
            skipped.append(key)
            continue
        if target in targets:
            raise RuntimeError(f"sft_load: two file tensors map to the same model key {target!r}")
        targets.add(target)
        mapping[key] = target

    state: dict[str, torch.Tensor] = {}
    nbytes = 0
    for key, tensor in read_tensors(path, list(mapping), header):
        state[mapping[key]] = tensor
        nbytes += tensor.numel() * tensor.element_size()

    result = model.load_state_dict(state, strict=False, assign=True)
    del state
    if result.unexpected_keys:
        logger.warning(
            "sft_load: %d tensor(s) of %s have no place in %s and were dropped: %s",
            len(result.unexpected_keys), path, type(model).__name__,
            ", ".join(sorted(result.unexpected_keys)[:20]),
        )
    still_meta = meta_tensor_names(model)
    if still_meta:
        shown = ", ".join(still_meta[:40])
        more = f" (+{len(still_meta) - 40} more)" if len(still_meta) > 40 else ""
        raise RuntimeError(
            f"sft_load: {len(still_meta)} tensor(s) of {type(model).__name__} were not in "
            f"{path} and are still on meta: {shown}{more}"
        )
    seconds = time.perf_counter() - started
    if skipped:
        logger.info("sft_load: skipped %d tensor(s) of %s (markers / unused)", len(skipped), path)
    logger.info(
        "sft_load: %d tensors, %.2f GiB from %s in %.1fs",
        len(mapping) - len(result.unexpected_keys), nbytes / 2**30, path, seconds,
    )
    return LoadReport(
        loaded=len(mapping) - len(result.unexpected_keys),
        skipped=tuple(skipped) + tuple(sorted(result.unexpected_keys)),
        seconds=round(seconds, 2),
        bytes=nbytes,
    )


def move_module_tree(root: nn.Module, device: torch.device, skip: set[int]) -> int:
    """Move every parameter / buffer of ``root`` to ``device``, skipping module ids in ``skip``.

    The engine25 shape (``engine25/gguf_transformer.py:_move_module_tree``),
    rewritten here because that module imports ltx_core: written by hand
    rather than as ``root.to(device)`` because the point is to leave one
    subtree (the DiT ``blocks``) on the CPU for the
    swap service. Parameters are rebound as fresh ``nn.Parameter``s with the
    same ``requires_grad`` (int8 ones stay ``False``); buffers go through their
    own ``.to()``. A tensor still on meta is a load bug and is refused by name.
    """
    moved = 0
    for module in root.modules():
        if id(module) in skip:
            continue
        for name, param in list(module._parameters.items()):
            if param is None or param.device == device:
                continue
            if param.is_meta:
                raise RuntimeError(f"parameter {name!r} of {type(module).__name__} is still on meta")
            module._parameters[name] = nn.Parameter(param.data.to(device), requires_grad=param.requires_grad)
            moved += 1
        for name, buf in list(module._buffers.items()):
            if buf is None or buf.device == device:
                continue
            if buf.is_meta:
                raise RuntimeError(f"buffer {name!r} of {type(module).__name__} is still on meta")
            module._buffers[name] = buf.to(device)
            moved += 1
    return moved


def load_mapped_to(
    module: nn.Module,
    path: str,
    header,
    mapping: dict[str, str],
    device: torch.device,
) -> int:
    """Read ``mapping``'s file tensors straight onto ``device`` into a meta ``module``; returns bytes.

    The direct read of the text encoder: one decoder layer (or the embedding
    table, or the I2V vision tower) at a time, file -> CPU buffer -> ``device``,
    so the CPU never holds more than one tensor of it. ``mapping`` is file name ->
    the module's own state-dict key (relative to ``module``). The tensors keep
    the file's dtype (``assign=True``), exactly as :func:`load_streaming` does,
    so a layer read this way computes bit for bit what a streamed-in layer
    moved to the GPU computes. Every tensor of ``module`` must come from the
    file: a key the module does not have, or a tensor left on meta, is refused.
    """
    state: dict[str, torch.Tensor] = {}
    nbytes = 0
    for key, tensor in read_tensors(path, list(mapping), header):
        state[mapping[key]] = tensor.to(device)
        nbytes += tensor.numel() * tensor.element_size()
        del tensor
    result = module.load_state_dict(state, strict=False, assign=True)
    del state
    if result.unexpected_keys:
        raise RuntimeError(
            f"sft_load: {type(module).__name__} has no place for {sorted(result.unexpected_keys)[:20]} of {path}"
        )
    still_meta = meta_tensor_names(module)
    if still_meta:
        raise RuntimeError(
            f"sft_load: {len(still_meta)} tensor(s) of {type(module).__name__} were not read from "
            f"{path} and are still on meta: {', '.join(still_meta[:40])}"
        )
    return nbytes


def drop_to_meta(module: nn.Module) -> None:
    """Rebind every parameter / buffer of ``module`` to an empty meta tensor of the same shape and dtype.

    The other half of :func:`load_mapped_to`: the module keeps its structure
    (and its hooks) and can be read in again, but holds no memory. Parameters
    keep their ``requires_grad`` (int8 ones stay ``False``).
    """
    for sub in module.modules():
        for name, param in list(sub._parameters.items()):
            if param is None or param.is_meta:
                continue
            sub._parameters[name] = nn.Parameter(
                torch.empty_like(param, device="meta"), requires_grad=param.requires_grad
            )
        for name, buf in list(sub._buffers.items()):
            if buf is None or buf.is_meta:
                continue
            sub._buffers[name] = torch.empty_like(buf, device="meta")
