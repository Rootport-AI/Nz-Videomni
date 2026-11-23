import json
for it in "abcde":
    d = json.load(open(f"outputs/h3-sol-spike-2026-11-23/runs/compare_{it}/result.json", encoding="utf-8"))
    print("=====", it, d["status"], d.get("error"), d["started_at"], d["finished_at"])
    for n, r in d["decodes"].items():
        st = r.get("stats") or {}
        print(f"  dec {n}: {r['result']} {r.get('seconds')}s load {(r.get('load') or {}).get('seconds')}s peak {r.get('peak')} frames {r.get('frames')} sec {st.get('seconds')} maxsec {st.get('max_sections_alive')}")
    m = d["metrics"]
    for n, v in m["vs_standard"].items():
        print(f"  vs std {n}: psnr {v['psnr_db']} min {v['per_frame_min_db']} ssim {v['ssim']} maxabs {v['max_abs']}")
    for n, v in m["pairs"].items():
        print(f"  pair {n}: psnr {v['psnr_db']} min {v['per_frame_min_db']} maxabs {v['max_abs']}")
    for n, v in m["grid_edges"].items():
        p = v["p128"]; q = v["p16"]; r2 = v["p256"]
        print(f"  grid {n}: p128 x {p['x']['ratio']}/{p['x']['ratio_excluding_16']} y {p['y']['ratio']}/{p['y']['ratio_excluding_16']} | p256 x {r2['x']['ratio']}/{r2['x']['ratio_excluding_16']} y {r2['y']['ratio']}/{r2['y']['ratio_excluding_16']} | p16 x {q['x']['ratio']} y {q['y']['ratio']}")
    for n, v in m["flicker"].items():
        print(f"  flicker {n}: boundary {v['boundary_mean']} inside {v['inside_mean']} ratio {v['ratio']}")
    print("  sheet", d.get("sheet"))
