"""Gate G3: H3Int8Linear blocks under the product's BlockSwapService (GPU).

Run (only after the supervisor's GPU go-ahead):
    set PYTHONPATH=<repo root>
    .venv-engine-h3\\Scripts\\python.exe outputs\\h3-gate1\\g3_blockswap.py [--report P.json]

A dummy transformer with ``blocks`` = 50 x (RMSNorm -> H3Int8Linear -> SiLU ->
H3Int8Linear, residual), hidden 1024, random int8 weights on the CPU. Three
modes, two passes each, from the SAME CPU state dict (cloned per mode):

  resident  every block on the GPU, no block swap
  sync      BlockSwapService(blocks_on_gpu=8, hold_arenas=True), prefetch_requested=False
  prefetch  same, prefetch_requested=True -> must report last_prefetch_used == "on"

Pass: the prefetch echo is "on", and all six outputs are bit-identical
(torch.equal). require_cuda_int8() has disabled comfy-kitchen's triton and
eager backends first, so a block whose weight or scale is not on the GPU at
forward time RAISES NoCapableBackendError instead of quietly taking a slow
path — getting through all six passes is itself the "CUDA path" check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from engine.transformer.block_swap_service import BlockSwapService  # noqa: E402
from engine_h3.quant_linear import H3Int8Linear, require_cuda_int8  # noqa: E402
from h3_format import QuantSpec  # noqa: E402

HIDDEN, FFN, N_BLOCKS, BLOCKS_ON_GPU, SEQ = 1024, 2048, 50, 8, 2048
SEED = 4321
SPEC = QuantSpec(format="int8_tensorwise", convrot=True, convrot_groupsize=256)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(HIDDEN, eps=1e-5, dtype=torch.bfloat16)
        self.fc1 = H3Int8Linear(HIDDEN, FFN, bias=False, spec=SPEC)
        self.fc2 = H3Int8Linear(FFN, HIDDEN, bias=True, spec=SPEC)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # tensors only, like H3DiTBlock
        return x + self.fc2(F.silu(self.fc1(self.norm(x))))


class _Dummy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(N_BLOCKS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


def _state_dict() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(SEED)
    sd: dict[str, torch.Tensor] = {}
    for i in range(N_BLOCKS):
        p = f"blocks.{i}."
        sd[p + "norm.weight"] = (1.0 + 0.1 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16)
        sd[p + "fc1.weight"] = torch.randint(-127, 128, (FFN, HIDDEN), generator=g, dtype=torch.int8)
        sd[p + "fc1.weight_scale"] = (torch.rand(FFN, 1, generator=g) + 0.5) * (0.5 / 127 / HIDDEN**0.5)
        sd[p + "fc2.weight"] = torch.randint(-127, 128, (HIDDEN, FFN), generator=g, dtype=torch.int8)
        sd[p + "fc2.weight_scale"] = (torch.rand(HIDDEN, 1, generator=g) + 0.5) * (0.5 / 127 / FFN**0.5)
        sd[p + "fc2.bias"] = (0.01 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16)
    return sd


def _build(sd: dict[str, torch.Tensor]) -> _Dummy:
    with torch.device("meta"):
        m = _Dummy()
    missing, unexpected = m.load_state_dict({k: v.clone() for k, v in sd.items()}, strict=True, assign=True)
    assert not missing and not unexpected
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(Path(__file__).with_name("g3_result.json")))
    args = ap.parse_args()

    dev = torch.device("cuda", 0)
    report: dict = {
        "gate": "G3",
        "config": {"hidden": HIDDEN, "ffn": FFN, "blocks": N_BLOCKS, "blocks_on_gpu": BLOCKS_ON_GPU, "seq": SEQ},
    }
    t0 = time.perf_counter()
    try:
        report["backends"] = require_cuda_int8()
        sd = _state_dict()
        x = torch.randn(SEQ, HIDDEN, generator=torch.Generator().manual_seed(SEED + 1)).to(torch.bfloat16).to(dev)
        outs: dict[str, list[torch.Tensor]] = {}
        modes: dict[str, dict] = {}

        with torch.no_grad():
            # resident
            m = _build(sd).to(dev)
            outs["resident"] = [m(x).clone() for _ in range(2)]
            modes["resident"] = {"last_prefetch_used": None}
            del m
            torch.cuda.empty_cache()

            for name, prefetch in (("sync", False), ("prefetch", True)):
                m = _build(sd)  # every block on the CPU
                svc = BlockSwapService(blocks_on_gpu=BLOCKS_ON_GPU, device=dev, hold_arenas=True)
                svc.prefetch_requested = prefetch
                svc.install(m)
                info = {"last_prefetch_used": svc.last_prefetch_used}
                try:
                    passes = []
                    for _ in range(2):
                        torch.cuda.synchronize(dev)
                        t1 = time.perf_counter()
                        passes.append(m(x).clone())
                        torch.cuda.synchronize(dev)
                        info.setdefault("pass_seconds", []).append(time.perf_counter() - t1)
                    outs[name] = passes
                finally:
                    svc.teardown_prefetch()
                    svc.release_installed()
                info["peak_vram_mb"] = torch.cuda.max_memory_allocated(dev) / 2**20
                modes[name] = info
                del m, svc
                torch.cuda.empty_cache()

        ref = outs["resident"][0]
        eq = {f"{k}[{i}]": bool(torch.equal(ref, o)) for k, v in outs.items() for i, o in enumerate(v)}
        maxdiff = {
            f"{k}[{i}]": float((ref.float() - o.float()).abs().max()) for k, v in outs.items() for i, o in enumerate(v)
        }
        report.update(
            modes=modes,
            bit_identical=eq,
            max_abs_diff=maxdiff,
            out_finite=bool(torch.isfinite(ref).all()),
            out_std=float(ref.float().std()),
        )
        report["pass"] = bool(
            modes["prefetch"]["last_prefetch_used"] == "on"
            and modes["sync"]["last_prefetch_used"] == "off"
            and all(eq.values())
            and report["out_finite"]
        )
    except Exception as exc:  # noqa: BLE001 — the report must say why
        import traceback

        report["pass"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    report["seconds"] = time.perf_counter() - t0
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k not in ("traceback", "backends")}, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
