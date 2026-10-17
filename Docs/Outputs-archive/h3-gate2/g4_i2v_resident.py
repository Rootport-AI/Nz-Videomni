r"""Gate 2a-G4: one I2V, the T2V regression, and keep_resident across jobs (plan §6, INTERFACE2 §8). GPU.

    set PYTHONPATH=<repo root>
    .venv-engine-h3\Scripts\python.exe outputs\h3-gate2\g4_i2v_resident.py [--report P]

ONE worker process (the shipped ``_do_load`` / ``_do_generate`` with synthesised
messages, as the selftest does), six jobs, 640x384 / 124 f / 24 fps / 20 steps /
seed 42 / sdpa, the G5 prompt:

  J0  T2V  no key              (regression: sha256 must be 0da6111c...)
  J1  I2V  no key, prefetch off (the reference for J3; PSNR of the first frame vs the canvas input)
  J2  T2V  keep on,  prefetch on
  J3  I2V  keep on,  prefetch off
  J4  T2V  keep on,  prefetch on
  J5  T2V  no key              (reuses what is kept, then releases it)

After every job: torch.cuda.memory_allocated / memory_reserved, the process
commit / working set, the sha256, where the DiT and the VAEs came from.
Pass (plan §6 2a-G4): J0, J2, J4, J5 = 0da6111c; J3 = J1; J3..J5 skip the DiT and
VAE reads; keeping adds NOTHING to what is allocated between jobs (the value
after every job equals the value after J0, a job that keeps nothing -- the
remainder is torch's cuBLAS workspace, see g4_allocated_probe.py; wording of
the gate changed by the supervisor after review m-13, 2026-10-18); the commit
after J5 is back to the level before keeping (after J1) -- if not, recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine_h3 import worker  # noqa: E402  (chdir(ROOT) + logging, as the real worker)

T2V_SHA = "0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9"
PROMPT = (
    "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in "
    "puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>"
)
JOBS = [
    ("J0", "t2v", None, True),
    ("J1", "i2v", None, False),
    ("J2", "t2v", True, True),
    ("J3", "i2v", True, False),
    ("J4", "t2v", True, True),
    ("J5", "t2v", None, True),
]


def psnr_first_frame(mp4: str, png: str, width: int, height: int) -> dict:
    import av
    import numpy as np

    from engine_h3.image_cond import load_canvas_image

    canvas, _ = load_canvas_image(png, width, height)
    ref = np.asarray(canvas, dtype=np.float64)
    with av.open(mp4) as c:
        frame = next(c.decode(video=0)).to_ndarray(format="rgb24").astype(np.float64)
    mse = float(((frame - ref) ** 2).mean())
    return {"psnr_db": round(10 * np.log10(255.0 ** 2 / mse), 2) if mse > 0 else None, "mse": round(mse, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(HERE / "i2v_input.png"))
    ap.add_argument("--report", default=str(HERE / "g4_i2v_resident.json"))
    args = ap.parse_args()

    import torch

    from engine_h3.pipeline_h3 import _mem_gib

    sel = worker._selftest_parser().parse_args([
        "--transformer", "models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "--text-encoder", "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "--video-vae", "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
        "--audio-vae", "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
        "--tokenizer-dir", "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
        "--output", "unused.mp4", "--prompt", PROMPT, "--attention-backend", "sdpa",
    ])
    events: list[dict] = []
    real_emit = worker._emit

    def capture(event, **fields):
        events.append({"event": event, **fields})
        real_emit(event, **fields)

    worker._emit = capture
    report: dict = {"jobs": [], "image": args.image}
    worker._do_load(worker._selftest_load_message(sel))
    report["after_load"] = {"mem": _mem_gib(), "allocated": torch.cuda.memory_allocated()}
    for name, kind, keep, prefetch in JOBS:
        out = str(HERE / f"g4_{name}_{kind}.mp4")
        msg = worker._selftest_generate_message(sel, out)
        msg["block_swap_prefetch"] = prefetch
        if kind == "i2v":
            msg["images"] = [{"path": args.image, "frame_idx": 0}]
        if keep is None:
            msg.pop("keep_resident")          # "no key"
        else:
            msg["keep_resident"] = keep
        events.clear()
        t0 = time.perf_counter()
        worker._do_generate(msg)
        seconds = round(time.perf_counter() - t0, 2)
        done = next(e for e in events if e["event"] == "done")
        ph = done["h3"]["phases"]
        rec = {
            "job": name, "kind": kind, "keep_resident": keep, "prefetch": prefetch, "seconds": seconds,
            "sha256": hashlib.sha256(Path(out).read_bytes()).hexdigest(),
            "keep_resident_used": done["keep_resident_used"], "images_used": done["images_used"],
            "block_swap_prefetch_used": done["block_swap_prefetch_used"], "attention_used": done["attention_used"],
            "dit_source": ph["denoise"].get("dit_source"), "vae_source": ph["decode"].get("vae_source"),
            "resident_parts_after": worker._PIPE.resident_parts,
            "allocated_after": torch.cuda.memory_allocated(),
            "reserved_after": torch.cuda.memory_reserved(),
            "mem_after": _mem_gib(),
            "done_peak_vram_mb": done["peak_vram_mb"], "done_peak_vram_reserved_mb": done["peak_vram_reserved_mb"],
            "phase_seconds": {k: v.get("seconds") for k, v in ph.items()},
            "denoise_read_seconds": (ph["denoise"].get("dit") or {}).get("read_seconds"),
            "phases": ph,
        }
        if kind == "i2v":
            rec["first_frame_vs_input"] = psnr_first_frame(out, args.image, 640, 384)
        report["jobs"].append(rec)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    worker._emit = real_emit

    j = {r["job"]: r for r in report["jobs"]}
    commit_before = j["J1"]["mem_after"].get("commit")
    commit_after = j["J5"]["mem_after"].get("commit")
    report["verdict"] = {
        "t2v_fingerprint_J0": j["J0"]["sha256"] == T2V_SHA,
        "t2v_fingerprints_resident_J2_J4_J5": all(j[k]["sha256"] == T2V_SHA for k in ("J2", "J4", "J5")),
        "i2v_resident_J3_equals_J1": j["J3"]["sha256"] == j["J1"]["sha256"],
        "reads_skipped_J3_J4_J5": all(j[k]["dit_source"] == "resident" and j[k]["vae_source"] == "resident"
                                      for k in ("J3", "J4", "J5")),
        "resident_adds_no_allocation_between_jobs": all(
            r["allocated_after"] == j["J0"]["allocated_after"] for r in report["jobs"]),
        "allocated_between_jobs_bytes": j["J0"]["allocated_after"],
        "released_after_J5": j["J5"]["resident_parts_after"] == [],
        "commit_after_J1_gib": commit_before,
        "commit_after_J5_gib": commit_after,
        "commit_process_peak_gib": j["J5"]["mem_after"].get("commit_peak"),
        "working_set_process_peak_gib": j["J5"]["mem_after"].get("rss_peak"),
        "commit_while_kept_J2_J3_J4_gib": [j[k]["mem_after"].get("commit") for k in ("J2", "J3", "J4")],
        "i2v_psnr_db": j["J1"]["first_frame_vs_input"]["psnr_db"],
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    Path(args.report).write_text(text, encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    v = report["verdict"]
    ok = all(v[k] for k in ("t2v_fingerprint_J0", "t2v_fingerprints_resident_J2_J4_J5", "i2v_resident_J3_equals_J1",
                            "reads_skipped_J3_J4_J5", "resident_adds_no_allocation_between_jobs",
                            "released_after_J5"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
