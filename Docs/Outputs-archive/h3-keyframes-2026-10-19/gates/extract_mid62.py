"""CPU only: frame F//2 of g5_t2v_after_review.mp4 -> kf_mid62.png (same decode/save as extract_keyframes.py)."""
import hashlib, json, sys
from pathlib import Path
import av
from PIL import Image
OUT = Path(r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-keyframes-2026-10-19")
SRC = Path(r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-gate1\g5_t2v_after_review.mp4")
frames = []
with av.open(str(SRC)) as c:
    for f in c.decode(video=0):
        frames.append(f.to_ndarray(format="rgb24"))
F = len(frames); mid = F // 2
arr = frames[mid]
p = OUT / f"kf_mid{mid}.png"
Image.fromarray(arr).save(p)
rec = {"source": str(SRC), "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(), "frames": F, "frame": mid,
       p.name: {"width": int(arr.shape[1]), "height": int(arr.shape[0]), "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}}
(OUT / "extract_kf_mid62.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(rec, indent=2))
