"""Block swapping service for low-VRAM inference.

Keeps only N transformer blocks resident on GPU at any time, swapping
the rest to CPU RAM during the forward pass. Works with the LTX-2
dual-stream DiT (48 blocks: video + audio streams share the same
transformer block list).

Usage:
    service = BlockSwapService(blocks_on_gpu=20, device=torch.device("cuda:0"))
    service.install(transformer)   # call once after model load
    service.uninstall(transformer) # call to restore original behaviour
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_BLOCK_SWAP_ATTR = "_block_swap_original_forward"


def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in args/kwargs to the target device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        moved = [_move_to_device(x, device) for x in obj]
        return type(obj)(moved)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


class BlockSwapService:
    """Installs forward-pass hooks on a transformer to swap blocks on/off GPU.

    Args:
        blocks_on_gpu: How many blocks to keep on GPU simultaneously.
                       0 disables block swapping entirely.
        device:        The GPU device blocks run on during their forward pass.
    """

    def __init__(self, blocks_on_gpu: int, device: torch.device) -> None:
        self.blocks_on_gpu = blocks_on_gpu
        self.device = device
        self._installed_transformers: list[nn.Module] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def install(self, transformer: nn.Module) -> None:
        """Patch transformer blocks with swap hooks. Idempotent."""
        if self.blocks_on_gpu == 0:
            logger.info("BlockSwap disabled (blocks_on_gpu=0)")
            return

        blocks = self._get_blocks(transformer)
        if not blocks:
            logger.warning("BlockSwap: could not find transformer blocks — skipping")
            return

        total = len(blocks)
        if self.blocks_on_gpu >= total:
            logger.info(
                "BlockSwap: blocks_on_gpu=%d >= total=%d — no swapping needed",
                self.blocks_on_gpu, total,
            )
            return

        logger.info(
            "BlockSwap: installing on %d blocks, keeping %d/%d on %s",
            total, self.blocks_on_gpu, total, self.device,
        )

        # Move all blocks to CPU initially.
        for block in blocks:
            block.to("cpu")

        # Patch each block with a swap-in / swap-out forward wrapper.
        for idx, block in enumerate(blocks):
            self._patch_block(block, idx, blocks)

        # Resident-reuse leak fix: the BlockSwapService is a single resident
        # instance, and install() runs on a freshly-built transformer on EVERY
        # generate (model_ledger never caches the model). The previous job's
        # transformer was already del'd by DistilledPipeline.__call__, so this
        # list held its last live reference — retaining it leaked one resident
        # transformer per job (~1GB GPU window + ~18GB CPU/commit of CPU-evicted
        # blocks), ratcheting to the job-5 native crash. Keep ONLY the current
        # transformer so the prior one becomes collectable (gc.collect() breaks
        # its swapped_forward reference cycles; see the between-job cleanup in
        # engine.worker._do_generate).
        self._installed_transformers.clear()
        self._installed_transformers.append(transformer)

    def uninstall(self, transformer: nn.Module) -> None:
        """Remove swap hooks and move all blocks back to GPU."""
        blocks = self._get_blocks(transformer)
        if not blocks:
            return

        for block in blocks:
            orig = getattr(block, _BLOCK_SWAP_ATTR, None)
            if orig is not None:
                block.forward = orig  # type: ignore[method-assign]
                delattr(block, _BLOCK_SWAP_ATTR)
            block.to(self.device)

        if transformer in self._installed_transformers:
            self._installed_transformers.remove(transformer)

        logger.info("BlockSwap: uninstalled, all blocks moved to %s", self.device)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_blocks(self, transformer: nn.Module) -> list[nn.Module]:
        """Find the transformer block list by trying common attribute names.

        LTX model_ledger.transformer() returns X0Model which wraps LTXModel
        as self.velocity_model — so we traverse one level deeper if needed.
        """
        candidates = [transformer]
        # LTX-specific: X0Model/LegacyX0Model wrap the real model in velocity_model
        inner = getattr(transformer, "velocity_model", None)
        if isinstance(inner, nn.Module):
            candidates.append(inner)

        for module in candidates:
            for attr in ("transformer_blocks", "blocks", "layers", "model_blocks"):
                blocks = getattr(module, attr, None)
                if blocks is not None and len(blocks) > 0:
                    return list(blocks)

        # Fallback: find any ModuleList with >4 children.
        for module in candidates:
            for _, child in module.named_children():
                if isinstance(child, nn.ModuleList) and len(child) > 4:
                    return list(child)

        return []

    def _patch_block(
        self,
        block: nn.Module,
        idx: int,
        all_blocks: list[nn.Module],
    ) -> None:
        """Replace block.forward with a version that swaps neighbours."""
        original_forward = block.forward
        setattr(block, _BLOCK_SWAP_ATTR, original_forward)

        blocks_on_gpu = self.blocks_on_gpu
        device = self.device
        total = len(all_blocks)

        def swapped_forward(*args: Any, **kwargs: Any) -> Any:
            # Window: keep blocks [idx .. idx+blocks_on_gpu-1] on GPU.
            window_start = idx
            window_end = min(idx + blocks_on_gpu, total)

            # Ensure every block in the current window is on GPU.
            # (On the first block this loads the full initial window;
            #  on subsequent blocks it incrementally loads the new tail.)
            #
            # IMPORTANT: use any() over all parameters, not just next() on the
            # first one.  FP8 (float8_e4m3fn) parameters can end up in a mixed
            # CPU/GPU state after the two-stage distilled pipeline transitions
            # from Stage 1 (half-res) to Stage 2 (full-res): a non-FP8 norm
            # weight may already be on GPU while the FP8 attention weights are
            # still on CPU.  Checking only the first parameter misses this and
            # skips the .to(device) call, causing a device-mismatch inside
            # fp8_cast.new_linear_forward when x is on CUDA but w_up is CPU.
            for load_idx in range(window_start, window_end):
                blk = all_blocks[load_idx]
                params = list(blk.parameters())
                if not params:
                    continue
                if any(p.device.type == "cpu" for p in params):
                    blk.to(device)
                    # Belt-and-suspenders: explicitly move any parameter that
                    # Module.to() may have silently skipped (observed with FP8
                    # dtypes on some Windows/CUDA builds).
                    for p in params:
                        if p.device.type == "cpu":
                            p.data = p.data.to(device)

            # Evict the block that just left the window (idx - 1).
            # It has already finished its forward pass.
            evict_idx = idx - 1
            if evict_idx >= 0:
                prev = all_blocks[evict_idx]
                prev_params = list(prev.parameters())
                if prev_params and any(p.device.type != "cpu" for p in prev_params):
                    prev.to("cpu")

            # BUG FIX: move input tensors to GPU before calling forward.
            # block.to(device) moves the weights, but args/kwargs still
            # hold tensors on CPU (output of the previous evicted block).
            # PyTorch dispatches ops to the device of the *input tensors*,
            # not the module — so without this the entire forward runs on
            # CPU, pinning utilisation at 100% and timing out.
            args = _move_to_device(args, device)
            kwargs = _move_to_device(kwargs, device)

            return original_forward(*args, **kwargs)

        block.forward = swapped_forward  # type: ignore[method-assign]


def build_block_swap_service(
    blocks_on_gpu: int,
    device: torch.device,
) -> BlockSwapService | None:
    """Factory: returns None when swapping is disabled (blocks_on_gpu=0)."""
    if blocks_on_gpu <= 0:
        return None
    return BlockSwapService(blocks_on_gpu=blocks_on_gpu, device=device)