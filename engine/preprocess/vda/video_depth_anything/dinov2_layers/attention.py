# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

import torch.nn.functional as F
from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


# VENDORING CHANGE: the xFormers import guard is gone. xFormers is not installed
# in the engine venv, and upstream's no-xFormers fallback materialises the full
# NxN score matrix (reserved 14.7 GB at 1920x1088 in the G0 smoke). Both forward
# passes below now call ``F.scaled_dot_product_attention`` unconditionally
# (~4.4 GB, faster, numerical delta well inside fp16 tolerance).


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        # VENDORING CHANGE: SDPA instead of the materialised q@k^T + softmax.
        # ``self.scale`` is dropped on purpose -- SDPA applies 1/sqrt(head_dim)
        # internally, which is the same value (see __init__ above). Inference
        # only, so ``attn_drop`` (p=0.0) is a no-op and is not wired in.
        assert attn_bias is None, "attn_bias (nested tensors) is not supported"
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    # VENDORING CHANGE: upstream's xFormers ``memory_efficient_attention`` body is
    # gone -- the inherited SDPA forward is already memory-efficient. The class is
    # KEPT (name + no new parameters) because dinov2_layers/__init__.py and
    # block.py reference it and the checkpoint loads with strict=True.
    pass
