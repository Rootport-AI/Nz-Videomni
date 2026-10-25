"""各腕の各ステップの壁時計の窓で、nvidia-smi dmon の PCIe 受信（rxpci）の中央値と 300〜2000 MB/s の標本数を出す。
外部の GPU の負荷（約 950 MB/s の定常の受信と SM 約 35%）の有無を見るため。使い方: python step_rx.py [label ...]"""
import json, statistics, sys, time
from pathlib import Path
RUNS = Path(__file__).resolve().parent / "runs"
labels = sys.argv[1:] or sorted(p.name for p in RUNS.iterdir() if (p / "run.json").exists())
for r in labels:
    d = json.loads((RUNS / r / "run.json").read_text(encoding="utf-8"))
    dn = [p for p in d["progress"] if p["stage"] == "stage1_denoise"]
    rows = []
    for ln in (RUNS / r / "dmon.txt").read_text(encoding="ascii", errors="replace").splitlines():
        f = ln.split()
        if ln.startswith("#") or len(f) < 11:
            continue
        try:
            rows.append((time.mktime(time.strptime(f[0] + " " + f[1], "%Y%m%d %H:%M:%S")), int(f[3]), int(f[9])))
        except Exception:
            pass
    out = []
    for k in range(1, len(dn)):
        rx = [x[2] for x in rows if dn[k - 1]["t_wall"] <= x[0] <= dn[k]["t_wall"]]
        out.append({"step": k, "step_seconds": d["timing"]["step_seconds"][k - 1], "samples": len(rx),
                    "rx_median_MBps": statistics.median(rx) if rx else None,
                    "rx_300_2000_samples": sum(1 for v in rx if 300 < v < 2000)})
    iw = d["idle_window"]
    idle = [(x[1], x[2]) for x in rows if iw["start_wall"] <= x[0] <= iw["end_wall"]]
    print(json.dumps({"label": r, "steps": out, "idle_sm_rx": idle}, ensure_ascii=False))
