"""Stand-in for diffsynth.core.attention.attention_forward (SDPA only).

minimax_h3_dit._sdpa_varlen_attention calls it with q/k/v in "b n s d"
(the default patterns) and ``scale=head_dim**-0.5``.
"""
import torch.nn.functional as F


def attention_forward(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d",
                      out_pattern="b n s d", dims=None, attn_mask=None, scale=None, is_causal=False, **_):
    assert q_pattern == k_pattern == v_pattern == out_pattern == "b n s d"
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale)
