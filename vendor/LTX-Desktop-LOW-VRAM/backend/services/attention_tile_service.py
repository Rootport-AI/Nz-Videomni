"""Attention tiling service for low-VRAM inference.

Replaces the global scaled_dot_product_attention with a chunked version
that processes the query sequence in tiles, keeping peak attention memory
bounded regardless of sequence length or resolution.

This is particularly effective for long videos and high resolutions where
the full attention matrix (seq_len x seq_len) would OOM.

Usage:
    service = AttentionTileService(tile_size=512)
    service.install()    # patches torch.nn.functional globally
    service.uninstall()  # restores original attention
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_ORIGINAL_SDPA_ATTR = "_attn_tile_original_sdpa"


class AttentionTileService:
    """Patches F.scaled_dot_product_attention with a memory-efficient tiled version.

    Args:
        tile_size: Number of query tokens processed per tile.
                   Smaller = less VRAM, more compute overhead.
                   0 disables tiling.
                   Recommended values: 256, 512, 1024, 2048.
    """

    def __init__(self, tile_size: int) -> None:
        self.tile_size = tile_size
        self._installed = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def install(self) -> None:
        """Patch F.scaled_dot_product_attention globally. Idempotent."""
        if self.tile_size <= 0:
            logger.info("AttentionTiling disabled (tile_size=0)")
            return

        if self._installed:
            return

        # Preserve the original (may already be SageAttention-patched).
        original = F.scaled_dot_product_attention
        setattr(F, _ORIGINAL_SDPA_ATTR, original)

        tile_size = self.tile_size

        def tiled_sdpa(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: float | None = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            # Only tile when sequence length exceeds tile_size and no mask.
            # Fall back to original for masked / causal / short sequences.
            q_len = query.shape[-2]
            if q_len <= tile_size or attn_mask is not None or is_causal:
                return original(
                    query, key, value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    **kwargs,
                )

            return _tiled_attention(
                query, key, value,
                tile_size=tile_size,
                dropout_p=dropout_p,
                scale=scale,
            )

        F.scaled_dot_product_attention = tiled_sdpa  # type: ignore[assignment]
        self._installed = True
        logger.info("AttentionTiling installed (tile_size=%d)", self.tile_size)

    def uninstall(self) -> None:
        """Restore original F.scaled_dot_product_attention."""
        if not self._installed:
            return

        original = getattr(F, _ORIGINAL_SDPA_ATTR, None)
        if original is not None:
            F.scaled_dot_product_attention = original  # type: ignore[assignment]
            delattr(F, _ORIGINAL_SDPA_ATTR)

        self._installed = False
        logger.info("AttentionTiling uninstalled")


# ------------------------------------------------------------------ #
# Core tiled attention implementation                                 #
# ------------------------------------------------------------------ #

def _tiled_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tile_size: int,
    dropout_p: float = 0.0,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute attention in query-sequence tiles to bound peak VRAM.

    Supports both 3D (batch*heads, seq, dim) and 4D (batch, heads, seq, dim)
    layouts as used by LTX-2's dual-stream DiT.
    """
    # Determine scale factor.
    head_dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    q_len = query.shape[-2]
    num_tiles = math.ceil(q_len / tile_size)

    output_chunks: list[torch.Tensor] = []

    for tile_idx in range(num_tiles):
        q_start = tile_idx * tile_size
        q_end = min(q_start + tile_size, q_len)

        # Slice the query tile — all dims except seq are kept whole.
        q_tile = query[..., q_start:q_end, :]

        # Use PyTorch's own SDPA for each tile — this lets it use flash
        # attention internally if available, while we control VRAM via tiling.
        tile_out = F.scaled_dot_product_attention(
            q_tile, key, value,
            attn_mask=None,
            dropout_p=dropout_p if query.requires_grad else 0.0,
            is_causal=False,
            scale=scale,
        )
        output_chunks.append(tile_out)

    return torch.cat(output_chunks, dim=-2)


def build_attention_tile_service(tile_size: int) -> AttentionTileService | None:
    """Factory: returns None when tiling is disabled (tile_size=0)."""
    if tile_size <= 0:
        return None
    return AttentionTileService(tile_size=tile_size)