r"""Gate 2a-G5: the video decode's VRAM at 1344x768 x 345 frames (plan §6). GPU.

    set PYTHONPATH=<repo root>
    .venv-engine-h3\Scripts\python.exe outputs\h3-gate2\g5_decode_vram.py [--report P]

Random normalised latents [1, 24, 102, 48, 84] (float32, seed 0) through
``H3VideoDecoder.decode`` with the real weights (bf16 + float32 islands, tiled).
Pass: the reserved-VRAM peak of the whole process (decoder weights + decode)
below 12.8 GB (8/10 of 16 GB: room for AviUtl2's preview), and the output CPU
uint8 [345, 768, 1344, 3]. The decode-only peak (after the weights are in) is
recorded as well.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from engine_h3.constants import latent_frames  # noqa: E402
from engine_h3.pipeline_h3 import _mem_gib  # noqa: E402
from engine_h3.video_vae import H3VideoDecoder  # noqa: E402

VAE = ROOT / "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors"
LINE_BYTES = 12.8e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(HERE / "g5_decode_vram.json"))
    args = ap.parse_args()
    dev = torch.device("cuda")
    w, h, f = 1344, 768, 345
    t_lat = latent_frames(f)
    gen = torch.Generator(device="cpu").manual_seed(0)
    lat = torch.randn((1, 24, t_lat, h // 16, w // 16), generator=gen, dtype=torch.float32).to(dev)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    dec = H3VideoDecoder.from_file(str(VAE), device=dev, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    after_load_alloc = torch.cuda.memory_allocated()
    after_load_reserved = torch.cuda.memory_reserved()
    load_peak_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        frames = dec.decode(lat, tiled=True)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    decode_peak_reserved = torch.cuda.max_memory_reserved()
    decode_peak_alloc = torch.cuda.max_memory_allocated()
    whole_peak_reserved = max(load_peak_reserved, decode_peak_reserved)
    rec = {
        "latent_shape": list(lat.shape),
        "output": {"shape": list(frames.shape), "dtype": str(frames.dtype), "device": str(frames.device),
                   "mean": round(float(frames.float().mean()), 2)},
        "decoder_weights_alloc_mb": round(after_load_alloc / 2**20),
        "after_load_reserved_mb": round(after_load_reserved / 2**20),
        "decode_peak_alloc_mb": round(decode_peak_alloc / 2**20),
        "decode_peak_reserved_mb": round(decode_peak_reserved / 2**20),
        "whole_peak_reserved_mb": round(whole_peak_reserved / 2**20),
        "whole_peak_reserved_gb": round(whole_peak_reserved / 1e9, 2),
        "line_gb": LINE_BYTES / 1e9,
        "seconds": {"load": round(t1 - t0, 1), "decode": round(t2 - t1, 1)},
        "mem": _mem_gib(),
    }
    ok = (whole_peak_reserved < LINE_BYTES and frames.dtype == torch.uint8 and frames.device.type == "cpu"
          and tuple(frames.shape) == (f, h, w, 3))
    rec["ok"] = ok
    text = json.dumps(rec, indent=2)
    Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
