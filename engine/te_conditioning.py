"""Small parts shared by the three engines (H3, LTX 2.3, LTX 2.5) for keeping the
text encoder's conditioning in main memory between jobs: the hash of a tensor's
values and the copies to and from main memory. This is the only torch-loading
file of the cache parts (``engine/ref_latent_cache.py`` stays standard-library only)."""

from __future__ import annotations

import hashlib

import torch


def tensor_sha256(t: torch.Tensor) -> str:
    """SHA-256 of a tensor's own bytes, without a copy: a CPU contiguous tensor is
    viewed as the integer type of its element size (bf16 -> int16) and hashed in
    place. Anything else is made CPU contiguous first, so its bytes are never misread."""
    t = t.detach()
    if t.device.type != "cpu" or not t.is_contiguous():
        t = t.cpu().contiguous()
    as_int = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return hashlib.sha256(t.view(as_int).numpy()).hexdigest()


def _map(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple) and hasattr(value, "_fields"):  # NamedTuple
        return type(value)(*[_map(v, fn) for v in value])
    if isinstance(value, tuple):
        return tuple(_map(v, fn) for v in value)
    if isinstance(value, list):
        return [_map(v, fn) for v in value]
    return value  # None, int, str, torch.device, ...: as it is


def host_copy(value):
    """A fresh copy in main memory (CPU), detached from the graph; dtype and shape are kept.
    Tuples, lists and NamedTuples are copied element by element and keep their type;
    None and anything that is not a tensor or such a container is returned as it is."""
    return _map(value, lambda t: t.detach().to("cpu", copy=True))


def device_copy(value, device):
    """A fresh copy on ``device`` (dtype not given); containers and other values as in :func:`host_copy`."""
    return _map(value, lambda t: t.to(device, copy=True))
