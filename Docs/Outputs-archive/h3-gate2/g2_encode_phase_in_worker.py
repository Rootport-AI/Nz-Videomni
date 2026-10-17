r"""2a-G2 side measurement: the worker selftest (1 denoise step, encode phase is the
point) with timers wrapped around the encode stage's pieces, to see where the
encode phase's seconds go inside the worker process. GPU; ~60 s.
"""
import functools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0]]
from engine_h3 import worker  # noqa: E402

T = []


def timed(owner, name, label):
    real = getattr(owner, name)

    @functools.wraps(real)
    def wrap(*a, **k):
        t0 = time.perf_counter()
        try:
            return real(*a, **k)
        finally:
            T.append([label, round(time.perf_counter() - t0, 2)])

    setattr(owner, name, wrap)


import engine_h3.pipeline_h3 as P  # noqa: E402
import engine_h3.text_encoder as TE  # noqa: E402

timed(P.H3Pipeline, "_encode", "pipeline._encode (whole)")
timed(TE.H3TextEncoder, "load", "te.load")
timed(TE.H3TextEncoder, "encode", "te.encode")
timed(TE.H3TextEncoder, "release", "te.release")
timed(P, "_free_cuda", "_free_cuda")
p = ("A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles. "
     "She looks at the camera and says <d>[English] It finally stopped raining.</d>")
out = ROOT / "outputs/h3-gate2/g2_encode_phase_in_worker.mp4"
rc = worker._selftest_generate([
    "--transformer", "models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "--text-encoder", "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "--video-vae", "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
    "--audio-vae", "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
    "--tokenizer-dir", "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
    "--output", str(out), "--prompt", p, "--steps", "1",
    "--report", str(ROOT / "outputs/h3-gate2/g2_encode_phase_in_worker.report.json"),
])
rep = json.loads((ROOT / "outputs/h3-gate2/g2_encode_phase_in_worker.report.json").read_text(encoding="utf-8"))
enc = rep["rounds"][0]["done"]["h3"]["phases"]["encode"]
res = {"timers": T, "encode_phase_seconds": enc["seconds"], "te_read": enc.get("te_read")}
(ROOT / "outputs/h3-gate2/g2_encode_phase_in_worker.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("RESULT", json.dumps(res), file=sys.stderr)
