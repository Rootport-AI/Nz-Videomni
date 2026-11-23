"""S8 (Sol) vs S8_off (the same body without sol_vae): the WDDM time series at the key moments, the step times,
the dedicated peaks and the decode stage. Reads runs/S8 and runs/S8_off; writes s8_compare.json. No GPU."""
import datetime
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def series(item):
    rows = []
    for line in (HERE / "runs" / item / "wddm.csv").read_text(encoding="ascii", errors="replace").splitlines():
        p = line.strip().split(",")
        if len(p) < 4 or p[0].startswith("timestamp"):
            continue
        try:
            rows.append((datetime.datetime.strptime(p[0], "%Y-%m-%d %H:%M:%S.%f").timestamp(), float(p[1]), float(p[2])))
        except ValueError:
            pass
    return rows


def one(item):
    r = json.loads((HERE / "runs" / item / "result.json").read_text(encoding="utf-8"))
    j = r["jobs"][0]
    rows = series(item)
    t_job = rows[0][0] + 3.0                        # the sampler's 3 s baseline, then the job is submitted
    st = j["stages"]
    den0 = next(s["t"] for s in st if s["stage"] == "stage1_denoise")
    dec = next(s["t"] for s in st if s["stage"] == "decode")
    end = st[-1]["t"]
    steps = [s["t"] for s in st if s["stage"] == "stage1_denoise"]
    step_gaps = [round(b - a, 1) for a, b in zip(steps, steps[1:]) if b - a > 0]

    def med(a, b, k=2):
        v = [x[k] for x in rows if t_job + a <= x[0] < t_job + b]
        return round(statistics.median(v), 1) if v else None

    def mx(a, b, k=1):
        v = [x[k] for x in rows if t_job + a <= x[0] < t_job + b]
        return round(max(v), 1) if v else None

    p = j["picks"]
    return {
        "status": j["status"], "sol": j["sol"], "generation_time_seconds": p["generation_time_seconds"],
        "seconds": p["seconds"], "vram_peak_mib": p["vram_peak_mib"],
        "stage_times": {"denoise_start": den0, "decode_start": dec, "end": end},
        "denoise_step_intervals_s": step_gaps,
        "shared_mb": {"before_job_median": med(-3, 0), "encode_median": med(0, den0),
                      "first_10s_of_denoise_median": med(den0, den0 + 10), "denoise_median": med(den0, dec),
                      "pre_decode_60s_median": med(dec - 60, dec), "decode_median": med(dec, end + 1),
                      "decode_max": mx(dec, end + 1, 2)},
        "dedicated_mb": {"encode_max": mx(0, den0), "denoise_max": mx(den0, dec), "decode_max": mx(dec, end + 1)},
        "wddm_judgement": {k: {kk: v.get(kk) for kk in ("samples", "shared_median_rise_mb", "overflow")}
                           for k, v in (j.get("wddm") or {}).items()},
        "denoise_jump_at_s": next((round(t - t_job, 1) for t, d, s in rows if s - rows[0][2] >= 300), None),
    }


out = {"S8": one("S8"), "S8_off": one("S8_off")}
(HERE / "s8_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
