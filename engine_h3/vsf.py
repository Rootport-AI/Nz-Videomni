"""MiniMax H3 negative prompt by Value Sign Flip (VSF, 台帳 §1-127).

The negative prompt's rows sit at the end of the text range of the packed
sequence, ``[positive (L_pos) | negative (n)]``; in every DiT block their
attention values V are multiplied by ``-scale`` right after V is projected
(before RoPE, which V never gets), so every query that attends to them is
pushed away from them in one attention call. The token refiner is not touched.

``flip_values`` is the one line both attention paths call (the off path
``H3Attention.forward`` and VRAM 16GB mode's ``lowvram.block_forward``), so
the two stay the same maths.

``H3VsfProbe`` measures, once per job (step 0, block 0), how much of the
softmax weight of a sample of video queries lands on the negative rows
(``m_neg``) and on the positive text rows (``m_text``). It only reads q and k.

Depends on torch only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class H3Vsf:
    """The negative rows ``rows`` of the packed sequence and the flip ``scale``."""

    rows: slice
    scale: float


def flip_values(v: torch.Tensor, vsf: H3Vsf) -> None:
    """``v[rows] *= -scale`` in place (v is ``[S, heads, D]``)."""
    v[vsf.rows].mul_(-float(vsf.scale))


def probe_query_rows(video: slice, n: int = 256) -> torch.Tensor:
    """``min(n, video rows)`` row indices spread evenly over the video range."""
    start, stop = int(video.start), int(video.stop)
    rows = stop - start
    count = min(int(n), rows)
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor([start + (i * rows) // count for i in range(count)], dtype=torch.long)


class H3VsfProbe:
    """Softmax mass of the sampled queries on the negative and the text rows.

    ``observe`` is called with each head group's q, k (``[S, g, D]``, after
    RoPE) and the group's first head number; the values are kept per head
    number and ``result`` averages them in ascending head order, so a single
    call over all heads (off path) and per-group calls (VRAM 16GB mode) give
    the same numbers.
    """

    def __init__(self, query_rows: torch.Tensor, neg_rows: slice, text_rows: slice) -> None:
        self.query_rows = query_rows
        self.neg_rows = neg_rows
        self.text_rows = text_rows
        self._neg: dict[int, float] = {}
        self._text: dict[int, float] = {}

    def observe(self, q: torch.Tensor, k: torch.Tensor, head0: int) -> None:
        g, d = int(q.shape[1]), int(q.shape[2])
        idx = self.query_rows.to(q.device)
        for j in range(g):
            qs = q[idx, j].detach().float()
            kh = k[:, j].detach().float()
            s = (qs @ kh.T) * (d ** -0.5)
            p = torch.exp(s - torch.logsumexp(s, dim=-1, keepdim=True))
            h = int(head0) + j
            if h in self._neg:
                raise RuntimeError(f"H3VsfProbe: ヘッド {h} が 2 回観測されました")
            self._neg[h] = float(p[:, self.neg_rows].sum(-1).mean())
            self._text[h] = float(p[:, self.text_rows].sum(-1).mean())

    def result(self) -> dict[str, float]:
        heads = sorted(self._neg)
        if not heads or heads != list(range(len(heads))):
            raise RuntimeError(f"H3VsfProbe: 観測したヘッドが揃っていません（{heads}）")
        n = len(heads)
        m_neg = sum(self._neg[h] for h in heads) / n
        m_text = sum(self._text[h] for h in heads) / n
        return {"m_neg": m_neg, "m_text": m_text}
