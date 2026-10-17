r"""Gate 2a-G2 (and its step-0 baseline): the H3 text encoder's T2V output, bit for bit.

    set PYTHONPATH=<repo root>
    .venv-engine-h3\Scripts\python.exe outputs\h3-gate2\g2_te_direct.py --mode baseline
        (BEFORE the direct-read change: the first-stage code, GemmaLayerOffloadService,
         te_layers_on_gpu=0 = one layer at a time. Writes te_baseline_t2v.pt + te_baseline_t2v.json)
    .venv-engine-h3\Scripts\python.exe outputs\h3-gate2\g2_te_direct.py --mode compare
        (AFTER the change: the direct-read encoder. torch.equal against the baseline
         hidden state [L, 5120] and the token ids; writes g2_te_direct.json)

The prompt is the first-stage G5 prompt (the one g4_te_compare.py names "the
G5 prompt"; 41 tokens). Each mode runs in a FRESH process, so the process
peaks (working set / commit) are the encoder's own: load -> encode -> release.
comfy-kitchen's CUDA path is pinned first (require_cuda_int8), as the pipeline's
load does, so the int8 matmuls run on the same backend in both modes.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from engine_h3.pipeline_h3 import _mem_gib  # noqa: E402
from engine_h3.quant_linear import require_cuda_int8  # noqa: E402
from engine_h3.text_encoder import H3TextEncoder  # noqa: E402

HERE = Path(__file__).resolve().parent
TE = ROOT / "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
TOK = ROOT / "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer"
PROMPT = (
    "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in "
    "puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>"
)
DEV = torch.device("cuda")


def run_encoder() -> tuple[dict, torch.Tensor, list[int], torch.Tensor]:
    kwargs = {"device": DEV}
    if "layers_on_gpu" in inspect.signature(H3TextEncoder.__init__).parameters:
        kwargs["layers_on_gpu"] = 0  # the first stage's shipped value (window of one layer)
    te = H3TextEncoder(str(TE), str(TOK), **kwargs)
    rec: dict = {"encoder_kwargs": {k: str(v) for k, v in kwargs.items()}, "mem": {}}
    rec["mem"]["start"] = _mem_gib()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    info = te.load()
    t1 = time.perf_counter()
    rec["mem"]["after_load"] = _mem_gib()
    with torch.no_grad():
        hidden, tags = te.encode(PROMPT)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    rec["mem"]["after_encode"] = _mem_gib()
    ids = te.tokenize(PROMPT)
    hidden_cpu, tags_cpu = hidden.detach().cpu(), tags.detach().cpu()
    del hidden, tags
    te.release()
    del te
    t3 = time.perf_counter()
    rec["mem"]["after_release"] = _mem_gib()
    rec["load_info"] = info
    rec["seconds"] = {"load": round(t1 - t0, 2), "encode": round(t2 - t1, 2), "release": round(t3 - t2, 2),
                      "total": round(t3 - t0, 2)}
    rec["peak_vram_mb"] = int(round(torch.cuda.max_memory_allocated() / 2**20))
    rec["peak_vram_reserved_mb"] = int(round(torch.cuda.max_memory_reserved() / 2**20))
    rec["tokens"] = len(ids)
    rec["hidden"] = {"shape": list(hidden_cpu.shape), "dtype": str(hidden_cpu.dtype),
                     "finite": bool(torch.isfinite(hidden_cpu.float()).all()),
                     "abs_mean": float(hidden_cpu.float().abs().mean())}
    return rec, hidden_cpu, ids, tags_cpu


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("baseline", "compare"), required=True)
    ap.add_argument("--baseline", default=str(HERE / "te_baseline_t2v.pt"))
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    backends = require_cuda_int8()
    rec, hidden, ids, tags = run_encoder()
    rec["mode"] = args.mode
    rec["prompt"] = PROMPT
    rec["cuda_backend"] = {k: bool(v.get("available")) and not bool(v.get("disabled"))
                           for k, v in backends.items() if isinstance(v, dict)}
    rec["mem"]["process_end"] = _mem_gib()

    if args.mode == "baseline":
        torch.save({"hidden": hidden, "ids": ids, "tags": tags, "prompt": PROMPT}, args.baseline)
        report = args.report or str(HERE / "te_baseline_t2v.json")
        rec["saved"] = args.baseline
        ok = rec["tokens"] == 41 and rec["hidden"]["finite"]
    else:
        base = torch.load(args.baseline)
        same_ids = list(base["ids"]) == list(ids)
        equal = bool(torch.equal(base["hidden"], hidden))
        same_tags = bool(torch.equal(base["tags"], tags))
        diff = float((base["hidden"].float() - hidden.float()).abs().max()) if base["hidden"].shape == hidden.shape else None
        rec["verdict"] = {"bit_identical": equal, "ids_identical": same_ids, "tags_identical": same_tags,
                          "max_abs_diff": diff}
        report = args.report or str(HERE / "g2_te_direct.json")
        ok = equal and same_ids and same_tags
    rec["ok"] = ok
    text = json.dumps(rec, indent=2, ensure_ascii=False, default=str)
    Path(report).write_text(text, encoding="utf-8")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
