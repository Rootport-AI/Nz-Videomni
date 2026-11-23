import json, sys
for it in sys.argv[1:]:
    d = json.load(open(f"outputs/h3-sol-spike-2026-11-23/runs/gen_{it}/result.json", encoding="utf-8"))
    e = d.get("engine") or {}
    full = e.get("full") or {}
    print(it, d["status"], d.get("self_check"), "load", d.get("load_seconds"), "seconds", e.get("seconds"),
          "peak", e.get("peak_vram_mb"), e.get("peak_vram_reserved_mb"), "std_dec", d.get("standard_decoder"),
          "latent", (d.get("latent") or {}).get("stats"), "probe", ((d.get("outputs") or {}).get("probe") or {}).get("video"),
          d.get("started_at"), d.get("finished_at"), d.get("error"))
