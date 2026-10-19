r"""CPU only: the first and the last frame of outputs/h3-gate1/g5_t2v_after_review.mp4
-> kf_first.png / kf_last.png (gate 1-G2 / 1-G4 test images, plan §5), and the
frames at TEN_POSITIONS -> kf10_NNN.png (the 10-keyframe VRAM run of 1-G4).

    .venv-engine-h3\Scripts\python.exe -I outputs\h3-keyframes-2026-10-19\extract_keyframes.py

PyAV decodes every frame (rgb24, the same call as Docs/Outputs-archive/h3-gate2/
g4_i2v_resident.py psnr_first_frame); nothing is generated.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import av
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "outputs/h3-gate1/g5_t2v_after_review.mp4"
#: 10 distinct pixel frames of F = 124 incl. 0 and F-1 (about every 13-14 frames).
TEN_POSITIONS = (0, 13, 27, 41, 55, 68, 82, 96, 110, 123)


def main() -> int:
    first = last = None
    n = 0
    ten = {}
    with av.open(str(SRC)) as c:
        for frame in c.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")
            if first is None:
                first = arr
            if n in TEN_POSITIONS:
                ten[n] = arr
            last = arr
            n += 1
    rec = {"source": str(SRC), "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(), "frames": n}
    for name, arr in (("kf_first.png", first), ("kf_last.png", last)):
        p = HERE / name
        Image.fromarray(arr).save(p)
        rec[name] = {"width": int(arr.shape[1]), "height": int(arr.shape[0]),
                     "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    rec["kf10"] = {}
    for pos, arr in sorted(ten.items()):
        p = HERE / f"kf10_{pos:03d}.png"
        Image.fromarray(arr).save(p)
        rec["kf10"][p.name] = {"frame": pos, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (HERE / "extract_keyframes.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
