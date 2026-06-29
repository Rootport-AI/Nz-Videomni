"""GPU cleanup helper service."""

from __future__ import annotations

import gc

import torch

from services.services_utils import empty_device_cache, sync_device


class TorchCleaner:
    """Wraps GPU memory cleanup operations."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self._device = device

    def cleanup(self) -> None:
        # Correct order: collect Python cyclic refs first so their tensors enter
        # the allocator's free list, then drain the GPU queue, then return those
        # free blocks to the driver.  Reversing the order (empty_cache then gc)
        # misses tensors only freed by the GC sweep.
        gc.collect()
        sync_device(self._device)
        empty_device_cache(self._device)
