r"""A-5 (台帳 §1-105 ①, R3): g2_block_equal.py unchanged, wrapped.

1. require_cuda_int8() first, then assert shared_quant_status() == (True, None).
2. Count H3Int8Linear.forward_rows / forward_rows_q / quantize_input / forward
   per (attention backend, on|off), where on|off is read from the H3DiTBlock's
   ``vram16`` at its forward (g2 toggles block.vram16 = None / Vram16Plan()).
   The wrappers only count; they call the original with the same arguments.
3. runpy g2_block_equal.py with --report <new path> (never the old default).
4. Print and save the counts; check on: forward_rows == 0, forward_rows_q == blocks*21.

    set PYTHONUTF8=1
    .venv-engine-h3\Scripts\python.exe outputs\h3-speedup-2026-10-24\a5_g2_wrapped.py
"""
from __future__ import annotations

import collections
import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
G2 = HERE / "g2_block_equal_r3.py"   # copy of outputs/h3-vram16-gates/g2_block_equal.py (build_coords argument renamed only)
REPORT = HERE / "a5_g2_block_equal.json"
COUNTS = HERE / "a5_counts.json"
BLOCKS = "0-49"

from engine_h3 import quant_linear  # noqa: E402
from engine_h3.dit import H3DiTBlock  # noqa: E402
from engine_h3.quant_linear import H3Int8Linear  # noqa: E402

quant_linear.require_cuda_int8()
status = quant_linear.shared_quant_status()
print(f"[a5] shared_quant_status() = {status}", flush=True)
if status != (True, None):
    print(f"[a5] STOP: shared quantization is not enabled: {status[1]}", flush=True)
    sys.exit(3)

counts: dict = collections.defaultdict(int)
ctx = {"mode": "outside"}


def counted(name, fn):
    def wrapper(*a, **k):
        counts[f"{ctx['mode']}|{name}"] += 1
        return fn(*a, **k)
    wrapper.__wrapped__ = fn
    return wrapper


for name in ("forward_rows", "forward_rows_q", "quantize_input", "forward"):
    setattr(H3Int8Linear, name, counted(name, getattr(H3Int8Linear, name)))

_block_forward = H3DiTBlock.forward


def block_forward(self, *a, **k):
    prev = ctx["mode"]
    ctx["mode"] = f"{self.attn.attention_backend}/{'on' if self.vram16 is not None else 'off'}"
    counts[f"{ctx['mode']}|block"] += 1
    try:
        return _block_forward(self, *a, **k)
    finally:
        ctx["mode"] = prev


H3DiTBlock.forward = block_forward

sys.argv = [str(G2), "--blocks", BLOCKS, "--report", str(REPORT)]
try:
    runpy.run_path(str(G2), run_name="__main__")
except SystemExit as e:
    if e.code not in (0, None):
        raise

a, _, b = BLOCKS.partition("-")
n_blocks = int(b) - int(a) + 1
res = {"status": list(status), "status_after": list(quant_linear.shared_quant_status()),
       "blocks": n_blocks, "counts": dict(sorted(counts.items())), "checks": {}}
for be in ("sage", "sdpa"):
    on_rows = counts[f"{be}/on|forward_rows"]
    on_q = counts[f"{be}/on|forward_rows_q"]
    on_quant = counts[f"{be}/on|quantize_input"]
    res["checks"][be] = {
        "on_forward_rows": on_rows, "on_forward_rows_q": on_q, "on_quantize_input": on_quant,
        "expect_forward_rows_q": n_blocks * 21,
        "ok": on_rows == 0 and on_q == n_blocks * 21 and on_quant == n_blocks,
    }
COUNTS.write_text(json.dumps(res, indent=2), encoding="utf-8")
print("[a5] " + json.dumps(res, indent=1), flush=True)
sys.exit(0 if all(c["ok"] for c in res["checks"].values()) else 4)
