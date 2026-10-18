"""Print one row per arm from runs/<label>/run.json (gates A-G5 / A-G4)."""
import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
labels = sys.argv[1:] or sorted(p.name for p in RUNS.iterdir() if (p / "run.json").exists())
rows = []
for lab in labels:
    f = RUNS / lab / "run.json"
    if not f.exists():
        print(lab, "missing")
        continue
    r = json.loads(f.read_text(encoding="utf-8"))
    d = r.get("done") or {}
    den = ((d.get("h3") or {}).get("phases") or {}).get("denoise") or {}
    w = (r.get("wddm") or {}).get("denoise_window") or {}
    idle = (r.get("wddm") or {}).get("idle") or {}
    t = r.get("timing") or {}
    row = {
        "label": lab,
        "denoise_alloc_peak": den.get("peak_vram_mb"), "denoise_reserved_peak": den.get("peak_vram_reserved_mb"),
        "step_seconds": t.get("step_seconds"), "step2": t.get("step2_seconds"),
        "shared_median": w.get("shared_mb_median"), "shared_max": w.get("shared_mb_max"),
        "shared_idle": idle.get("shared_mb_median"),
        "commit_peak_gib": d.get("commit_peak_gib"),
        "prefetch": d.get("block_swap_prefetch_used"), "vram16_used": d.get("vram16_mode_used"),
        "vram16": den.get("vram16"), "seq": (den.get("sequence") or {}).get("total"),
        "error": (r.get("error") or {}).get("message", "")[:200] if r.get("error") else None,
        "mismatch": r.get("effective_mismatch"), "overrides": r.get("vram16_overrides"),
    }
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False))
