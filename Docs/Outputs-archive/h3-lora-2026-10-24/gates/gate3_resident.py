"""Gate 3 (plan §5.2): resident cleanliness through the app's HTTP API.

J0 none (no keep) -> J1 Turbo (keep) -> J2 Style (keep) -> J3 none (keep) -> J4 Style (no keep).
Pass: sha256(J3) == sha256(J0) and sha256(J2) == sha256(J4). All jobs 640x384x124f, seed 42,
G5 prompt, embed_mp4_metadata=false (so the mp4 bytes depend only on the generation).
Run with the app venv python; the backend must already be running (run.bat) on port 18620.
"""
import hashlib, json, sys, time, urllib.request, urllib.error
from pathlib import Path

BASE = "http://127.0.0.1:18620/api/v1"
REPO = Path(r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni")
OUT = REPO / "outputs" / "h3-gate-lora"
PROMPT = ("A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles. "
          "She looks at the camera and says <d>[English] It finally stopped raining.</d>")
TURBO = "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16"
STYLE = "3dsrx_minimax_h3"


def http(method, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wait_ready():
    for _ in range(600):
        st, s = http("GET", "/status")
        if st == 200 and s.get("state") in ("ready", "loaded") and s.get("base_model") == "MiniMaxH3":
            return s
        time.sleep(2)
    raise SystemExit(f"pipeline not ready: {s}")


def job(name, loras, keep, steps=20, shift_v=12.0, shift_a=3.0):
    body = {"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 124, "frame_rate": 24,
            "seed": 42, "num_inference_steps": steps, "shift_video": shift_v, "shift_audio": shift_a,
            "keep_resident": keep, "embed_mp4_metadata": False}
    if loras:
        body["loras"] = [{"name": n, "strength": 1.0} for n in loras]
    t0 = time.time()
    st, r = http("POST", "/generate", body)
    if st != 202 and st != 200:
        raise SystemExit(f"{name}: submit failed {st} {r}")
    jid = r["job_id"]
    while True:
        time.sleep(3)
        st, j = http("GET", f"/jobs/{jid}")
        if j["status"] in ("completed", "failed", "cancelled"):
            break
    secs = time.time() - t0
    if j["status"] != "completed":
        raise SystemExit(f"{name}: {j['status']} {j.get('error')}")
    res = j["result"]
    md = json.loads(Path(res["metadata_path"]).read_text(encoding="utf-8"))
    h3 = md.get("h3", {})
    row = {"name": name, "job_id": jid, "loras": loras, "keep": keep, "seconds": round(secs, 1),
           "gen_seconds": res.get("generation_time_seconds"), "sha256": sha(res["output_path"]),
           "dit_source": h3.get("phases", {}).get("denoise", {}).get("dit_source"),
           "h3_loras": h3.get("loras"), "ic_lora": md.get("ic_lora"),
           "peak_vram_mb": md.get("peak_vram_mb") or h3.get("peak_vram_mb"), "output": res["output_path"]}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st, s = http("GET", "/status")
    print("status:", st, json.dumps(s)[:300], flush=True)
    if s.get("base_model") != "MiniMaxH3" or s.get("state") not in ("ready", "loaded"):
        st, r = http("POST", "/pipeline/load", {"base_model": "MiniMaxH3"}, timeout=120)
        print("load:", st, json.dumps(r)[:200], flush=True)
        wait_ready()
    st, l = http("GET", "/loras")
    print("loras:", l.get("lora_dir"), [(e["name"], e["supported"]) for e in l["loras"]], flush=True)
    rows = []
    rows.append(job("J0_none_nokeep", [], False))
    rows.append(job("J1_turbo_keep", [TURBO], True, steps=4, shift_v=6.0, shift_a=3.0))
    rows.append(job("J2_style_keep", [STYLE], True))
    rows.append(job("J3_none_keep", [], True))
    rows.append(job("J4_style_nokeep", [STYLE], False))
    by = {r["name"]: r for r in rows}
    verdict = {"J3==J0": by["J3_none_keep"]["sha256"] == by["J0_none_nokeep"]["sha256"],
               "J2==J4": by["J2_style_keep"]["sha256"] == by["J4_style_nokeep"]["sha256"],
               "J1!=J0": by["J1_turbo_keep"]["sha256"] != by["J0_none_nokeep"]["sha256"],
               "J2_resident": by["J2_style_keep"]["dit_source"] == "resident",
               "J3_resident": by["J3_none_keep"]["dit_source"] == "resident"}
    report = {"rows": rows, "verdict": verdict, "pass": all(verdict.values())}
    (OUT / "g3_resident.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("VERDICT:", json.dumps(verdict), "PASS" if report["pass"] else "FAIL", flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
