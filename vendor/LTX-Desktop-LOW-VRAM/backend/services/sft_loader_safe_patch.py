"""Portable runtime monkeypatch for ltx_core's safetensors weight loader.

Root cause (confirmed): ltx_core/loader/sft_loader.py
`SafetensorsStateDictLoader.load` reads each tensor with

    with safetensors.safe_open(shard_path, framework="pt", device=str(device)) as f:
        ...
        value = f.get_tensor(name).to(device=device, non_blocking=True, copy=False)

`safe_open` memory-maps the shard and `get_tensor` materializes tensors that
alias the mmap'd backing storage. On Windows, materializing these mmap tensors
inside a long-lived/persistent worker that rebuilds the Gemma text encoder
per-job corrupts the heap and surfaces as a native
`Windows fatal exception: access violation` (exit 139) — deterministically at
`f.get_tensor(...)` for the base-Gemma CPU load, and intermittently downstream
(e.g. in the GGUF dequant path) as the corruption propagates.

A prior attempt kept `safe_open` + `get_tensor` and only flipped the transfer to
`non_blocking=False, copy=True`. That did NOT fix it: `get_tensor` still
materializes the mmap, so the corruption source remained.

This patch replaces `SafetensorsStateDictLoader.load` with a fully mmap-free
loader, mirroring the ComfyUI `--disable-mmap` recipe that is proven to work on
safetensors 0.7.0 (which has NO `backend=` kwarg). Per shard it reads the entire
file into RAM and parses it with `safetensors.torch.load(bytes)` (no `safe_open`,
no mmap at all), then performs an EAGER synchronous owning transfer to the target
device (`tensor.to(device=device, copy=True)`; no `non_blocking=True`,
no `copy=False`). It reproduces the original load()'s `sd_ops` application,
size/dtype accumulation, sharded-list handling, and `StateDict` return structure
exactly, so both CPU targets (Gemma) and GPU targets (VAE/upsampler/transformer)
behave identically — only the mmap is removed.

Apply once, after `import ltx_core.loader` is warmed and before building the
pipeline:

    from services.sft_loader_safe_patch import apply_sft_loader_safe_patch
    apply_sft_loader_safe_patch()
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_sft_loader_safe_patch() -> bool:
    """Monkeypatch SafetensorsStateDictLoader.load to a non-mmap load(bytes) loader.

    Replaces the mmap-based `safe_open`/`get_tensor` path with a fully
    mmap-free `safetensors.torch.load(bytes)` read plus an eager owning
    `.to(device, copy=True)` transfer, preserving the original load() semantics.

    Returns True if applied (or already applied), False on failure.
    """
    global _APPLIED
    if _APPLIED:
        return True
    try:
        import safetensors.torch as safetensors_torch
        import torch
        from ltx_core.loader import sft_loader
        from ltx_core.loader.primitives import StateDict

        def _safe_load(self, path, sd_ops, device=None):  # type: ignore[no-untyped-def]
            # Non-mmap reimplementation of SafetensorsStateDictLoader.load.
            # Reads each shard fully into RAM and parses with
            # safetensors.torch.load(bytes) — no safe_open, no mmap, so nothing
            # can alias mmap'd backing storage. Then moves each tensor to the
            # target device with an eager, synchronous, owning copy.
            sd = {}
            size = 0
            dtype = set()
            device = device or torch.device("cpu")
            model_paths = path if isinstance(path, list) else [path]
            for shard_path in model_paths:
                # Read the whole shard into RAM (mmap-free), parse to CPU tensors.
                with open(shard_path, "rb") as fh:
                    data = fh.read()
                shard_sd = safetensors_torch.load(data)  # CPU tensors, no mmap
                del data
                for name, tensor in shard_sd.items():
                    expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if expected_name is None:
                        continue
                    # copy=True: own the memory (never alias any backing buffer).
                    # No non_blocking: synchronous, CUDA-safe transfer.
                    value = tensor.to(device=device, copy=True)
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, value in key_value_pairs:
                        size += value.nbytes
                        dtype.add(value.dtype)
                        sd[key] = value
                del shard_sd
            return StateDict(sd=sd, device=device, size=size, dtype=dtype)

        sft_loader.SafetensorsStateDictLoader.load = _safe_load  # type: ignore[assignment]
        _APPLIED = True
        logger.info(
            "sft_loader safe patch applied (non-mmap load(bytes); eager copy=True transfer)"
        )
        return True
    except Exception as exc:  # pragma: no cover
        logger.warning("sft_loader safe patch failed to apply: %s", exc)
        return False
