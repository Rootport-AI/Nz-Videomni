r"""Sol-H3-VAE stage 1: the GPU checks of the plan's 6.2 (MiniMax H3; product code read only).

Items (run in this order by default; each one alone with --only):

  S1   product decoder == the experiment's configuration B, the 5 latents of the experiment (engine_items.py)
  S2   the product path with Sol (API): T2V, a 2-clip chain, V2V, Inpainting -- each completes, sol_vae_used "on",
       vae_source "sol", the frame count and the size
  S3   the ordinary use (API): Turbo LoRA 0.8, 4 steps, 1280 x 768, 124 frames x 5 clips, one shared prompt;
       Off and Sol once each (completion, the decode stage's seconds and VRAM peaks)
  S4   the fallback (API, a temporary descriptor whose sol_vae_path names no file): the Sol job completes with
       "on->off" / reason file_missing, and its video stream equals the same-seed Off job's
  S5   the Off path unchanged: F0 = the worker selftest's mp4 sha256 == the reference fingerprint 0da6111c...
       (design canon 裁定 45), F1 = the 2026-11-22 gate's R1 condition over the API, stream MD5 == R1's record
       (the pi-H3 gates' F0 / F1)
  S6   a broken Sol file fails the job at the start, before the text encoder, with the facts only (engine_items.py)
  S7   keep_resident on: Off -> Sol -> Off in one backend session; the 3rd vae_source is "resident"
  S8   the product path's VRAM (API): 2560 x 1472, 192 frames x 2 clips with Sol; the decode stage's peaks and the
       WDDM shared GPU memory while the job runs (the experiment's sampler)
  S9   the Sol weights' read time (engine_items.py; warm from_file x3 + one FILE_FLAG_NO_BUFFERING read)
  S10  the size limit at 2560 x 1472 (engine_items.py; WDDM sampled here): T'=112 inside -> Sol, T'=117 above ->
       the stock decoder with reason too_large

From the repo root (Git Bash; the APP venv; this script never imports torch):

    export PYTHONUTF8=1
    G=outputs/h3-sol-gates-2026-11-23/h3_sol_gates.py
    .venv/Scripts/python.exe -X utf8 $G --dry-run            # plans, files, requests, engine dry runs; no GPU
    .venv/Scripts/python.exe -X utf8 $G --gate-check         # the GPU gate only
    .venv/Scripts/python.exe -X utf8 $G --live --only S1     # one item (GPU); a done item is skipped (--redo)
    .venv/Scripts/python.exe -X utf8 $G --post-only          # summary.json / summary.md from runs/

The backend plumbing (temporary yaml / state under _rt/, own port 18699, the gate before every item, the stop in
``finally``, the stream MD5, the home-path scrub) is the pi-H3 gates' (outputs/h3-pih3-gates-2026-11-23/
h3_pih3_gates.py, imported and pointed at this folder). The owner's config.yaml / state.json / manifests are only
read: S4's descriptor change is a copy of the manifests under _rt/ named by the temporary yaml.

Exit codes: 0 = ran (verdicts in the records), 1 = gate / setup refused, 2 = bad arguments, 3 = an item failed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PIH3_GATES = REPO / "outputs" / "h3-pih3-gates-2026-11-23" / "h3_pih3_gates.py"
SPIKE = REPO / "outputs" / "h3-sol-spike-2026-11-23"
SAMPLER = HERE / "wddm_sampler.ps1"
PORT_DEFAULT = 18699


def _load_pg():
    spec = importlib.util.spec_from_file_location("h3_pih3_gates", PIH3_GATES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # pointed at this folder (the functions read these module globals at call time)
    mod.HERE, mod.RUNS, mod.RT, mod.LOG_DIR = HERE, HERE / "runs", HERE / "_rt", HERE / "logs"
    mod.HALT = HERE / "runs" / "_halt.json"
    mod.SUMMARY_JSON, mod.SUMMARY_MD = HERE / "summary.json", HERE / "summary.md"
    mod.TEMP_CONFIG_NAME, mod.TEMP_STATE_NAME = "config.h3solgates.yaml", "state.h3solgates.json"
    mod.PORT_DEFAULT = PORT_DEFAULT
    return mod


pg = _load_pg()
log = pg.log
RUNS, RT, LOG_DIR = pg.RUNS, pg.RT, pg.LOG_DIR
FL2VA = pg.FL2VA
TURBO = (pg.TURBO_LORA, 0.8)

#: S4: the temporary descriptor's sol_vae_path (a path inside models_dir where no file is; nothing is created).
ABSENT_SOL = "MiniMaxH3/VAE/Sol/__h3_sol_gates_absent__.safetensors"
_MANIFEST_OVERRIDE: dict = {"sol_vae_path": None}
_real_write_temp_config = pg.write_temp_config


def write_temp_config(port: int) -> Path:
    """The pi-H3 gates' temporary yaml, plus (S4 only) a copy of the manifests with another sol_vae_path."""
    import yaml  # noqa: PLC0415

    cfg = _real_write_temp_config(port)
    if _MANIFEST_OVERRIDE["sol_vae_path"] is None:
        return cfg
    mdir = RT / "manifests"
    shutil.rmtree(mdir, ignore_errors=True)
    shutil.copytree(REPO / "scripts" / "manifests", mdir)
    m = json.loads((mdir / "40-minimaxh3.json").read_text(encoding="utf-8-sig"))
    m["assets"]["sol_vae_path"] = _MANIFEST_OVERRIDE["sol_vae_path"]
    (mdir / "40-minimaxh3.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    raw.setdefault("model", {})["manifest_dir"] = str(mdir)
    cfg.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    return cfg


pg.write_temp_config = write_temp_config

# ---------------------------------------------------------------------------------------------
# Bodies
# ---------------------------------------------------------------------------------------------

COMMON = {"attention_backend": "sage", "vram16_mode": True, "block_swap_prefetch": True, "keep_resident": False}
STEPS4 = {"num_inference_steps": 4, "shift_video": 12.0, "shift_audio": 3.0}


def t2v(**kw) -> dict:
    return {"prompt": pg.CITY, "width": 1280, "height": 768, "num_frames": 73, "frame_rate": 24.0, "seed": 1234,
            **COMMON, **STEPS4, **kw}


def items() -> dict[str, dict]:
    J, CJ = pg.job, pg.chain_job
    I: dict[str, dict] = {}
    I["S1"] = {"kind": "engine", "title": "製品の復号器と実験の構成 B の一致（実験の潜在 5 本）",
               "rule": "5 本とも uint8 が 1 ビットまで一致し、製品の映像の MD5 が実験の B.mp4 と一致"}
    I["S2"] = {"kind": "api", "sol": True, "title": "製品の経路で Sol（T2V・連結 2 本・V2V・Inpainting。Turbo LoRA 0.8・4 ステップ）",
               "rule": "4 本とも完了し、sol_vae_used が on、vae_source が sol、フレーム数と大きさが合う",
               "jobs": [J("S2_t2v", FL2VA, t2v(sol_vae=True), lora=TURBO),
                        CJ("S2_chain", {"prompt": pg.CHAIN_SAME_PROMPT, "width": 1280, "height": 768, "seed": 1234,
                                        "clips": [{"num_frames": 124}, {"num_frames": 124}], "sol_vae": True,
                                        **COMMON, **STEPS4}, frames=226, width=1280, height=768, lora=TURBO),
                        CJ("S2_v2v", {"prompt": pg.COSPLAY, "width": 640, "height": 384, "seed": 1234,
                                      "clips": [{"num_frames": 56}, {"num_frames": 39}], "sol_vae": True,
                                      "source_video": {"video_id": f"@video:{pg.V2V_SRC}"}, **COMMON, **STEPS4},
                           frames=51, width=640, height=384, lora=TURBO),
                        J("S2_inpaint", FL2VA, t2v(prompt=pg.GIRLS, sol_vae=True, inpaint={
                            "source_video_id": f"@video:{pg.INPAINT_SRC}", "mask_video_id": f"@video:{pg.INPAINT_MASK}",
                            "window_start_sec": 0.0, "audio_strength": 1.0}), lora=TURBO)]}
    long_body = lambda **kw: {"prompt": pg.CHAIN_SAME_PROMPT, "width": 1280, "height": 768, "seed": 1234,  # noqa: E731
                              "clips": [{"num_frames": 124} for _ in range(5)], **COMMON, **STEPS4, **kw}
    I["S3"] = {"kind": "api", "title": "ふだんの使い方（Turbo LoRA 0.8・4 ステップ・1280×768・124 フレーム × 5 本）の Off と Sol",
               "rule": "2 本とも完了（532 フレーム・音声）。Sol は sol_vae_used on。復号の段の秒と VRAM の山を記録",
               "jobs": [CJ("S3_off", long_body(), frames=532, width=1280, height=768, lora=TURBO, timeout=3600),
                        CJ("S3_sol", long_body(sol_vae=True), frames=532, width=1280, height=768, lora=TURBO,
                           timeout=3600)]}
    I["S4"] = {"kind": "api", "absent_sol": True, "title": "切り替え（資産のパスを無いファイルに向けた記述子の写し）",
               "rule": "2 本とも完了し、Sol のジョブは on->off（理由 file_missing）、映像のストリームが同じシードの Off と一致",
               "jobs": [J("S4_sol", FL2VA, t2v(sol_vae=True), lora=TURBO), J("S4_off", FL2VA, t2v(), lora=TURBO)]}
    I["S5"] = {"kind": "off_path", "title": "Off の経路（F0: ワーカーの自己試験の照合値／F1: R1 の条件の API）",
               "rule": f"F0 の mp4 の sha256 が {pg.FINGERPRINT[:8]}…、F1 の映像と音声のストリームが R1 の記録と一致し、"
                       "記録に sol_vae の欄が無い",
               "jobs": [J("S5_F1", FL2VA, pg.off20(), compare_to="F1")]}
    I["S6"] = {"kind": "engine", "title": "壊れたファイル（実物の先頭 64 MiB の写し）",
               "rule": "テキストエンコーダーより前に失敗し、文が「Sol-H3-VAE のファイル（…）の照合で、合わない所が 1 か所…」"}
    res = lambda **kw: t2v(keep_resident=True, **kw)  # noqa: E731
    I["S7"] = {"kind": "api", "title": "常駐 on で Off → Sol → Off",
               "rule": "3 本とも完了し、vae_source が file → sol → resident、Sol は on",
               "jobs": [J("S7_1off", FL2VA, res(), lora=TURBO), J("S7_2sol", FL2VA, res(sol_vae=True), lora=TURBO),
                        J("S7_3off", FL2VA, res(), lora=TURBO)]}
    I["S8"] = {"kind": "api", "wddm": True, "title": "製品の経路での VRAM（2560×1472・192 フレーム × 2 本・Sol）",
               "rule": "完了し、sol_vae_used on。復号の段の VRAM の山（確保と予約）と、共有 GPU メモリの増分が 0 付近",
               "jobs": [CJ("S8", {"prompt": pg.CHAIN_SAME_PROMPT, "width": 2560, "height": 1472, "seed": 1234,
                                  "clips": [{"num_frames": 192}, {"num_frames": 192}], "sol_vae": True,
                                  **COMMON, **STEPS4}, frames=362, width=2560, height=1472, lora=TURBO,
                           timeout=5400)]}
    # The control of S8 (2026-11-24, the supervisor's request): the same body without sol_vae. Named only
    # (--only S8_off); not in the default order.
    I["S8_off"] = {"kind": "api", "wddm": True, "default": False,
                   "title": "S8 の対照（同じ条件で sol_vae を指定しない Off）",
                   "rule": "完了し、sol_vae_used が無い。共有 GPU メモリの時系列を S8 と比べる（判定は記録だけ）",
                   "jobs": [CJ("S8_off", {"prompt": pg.CHAIN_SAME_PROMPT, "width": 2560, "height": 1472, "seed": 1234,
                                          "clips": [{"num_frames": 192}, {"num_frames": 192}],
                                          **COMMON, **STEPS4}, frames=362, width=2560, height=1472, lora=TURBO,
                               timeout=5400)]}
    I["S9"] = {"kind": "engine", "title": "Sol の重みの読み込みの時間",
               "rule": "記録だけ（OS のファイルのキャッシュを通さない読み取りの秒と、from_file の 3 回の秒）"}
    I["S10"] = {"kind": "engine", "wddm": True, "title": "上限の境目（2560×1472。T'=112 は Sol、T'=117 は too_large）",
                "rule": "内側は Sol で完走し共有 GPU メモリの増分が 0 付近、超えたほうは通常の VAE（reason too_large）で完走"}
    return I


ITEMS = items()
ORDER = [k for k, v in ITEMS.items() if v.get("default", True)]

# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------


def sol_picks(meta: dict) -> dict:
    h3 = meta.get("h3") or {}
    dec = (h3.get("phases") or {}).get("decode") or {}
    return {"sol_vae_used": meta.get("sol_vae_used"), "sol_vae": h3.get("sol_vae"), "vae_source": dec.get("vae_source"),
            "decode_seconds": dec.get("seconds"),
            "decode_vram_mib": [dec.get("peak_vram_mb"), dec.get("peak_vram_reserved_mb")],
            "request_sol_vae": (meta.get("request") or {}).get("sol_vae")}


def judge_jobs(item: str, jobs: list[dict]) -> str:
    if not jobs or any(j.get("status") != "completed" for j in jobs):
        return "fail"
    out_ok = all(all(j.get("checks", {}).get(k, False) for k in ("frames_ok", "size_ok", "audio_ok")) for j in jobs)
    s = {j["job"]: j.get("sol") or {} for j in jobs}
    if item == "S2":
        ok = all(v["sol_vae_used"] == "on" and v["vae_source"] == "sol" for v in s.values())
    elif item == "S3":
        ok = s["S3_sol"]["sol_vae_used"] == "on" and s["S3_off"]["sol_vae_used"] is None
    elif item == "S4":
        sol = s["S4_sol"]
        eq = (jobs[0].get("compare_off") or {}).get("video_equal")
        ok = sol["sol_vae_used"] == "on->off" and (sol["sol_vae"] or {}).get("reason") == "file_missing" and bool(eq)
    elif item == "S5":
        c = jobs[0].get("compare") or {}
        ok = bool(c.get("equal")) and s["S5_F1"]["sol_vae_used"] is None and not s["S5_F1"]["sol_vae"]
    elif item == "S7":
        ok = ([v["vae_source"] for v in s.values()] == ["file", "sol", "resident"]
              and s["S7_2sol"]["sol_vae_used"] == "on")
    elif item == "S8_off":
        ok = s["S8_off"]["sol_vae_used"] is None and not s["S8_off"]["sol_vae"]
    elif item == "S8":
        w = (jobs[0].get("wddm") or {})
        ok = s["S8"]["sol_vae_used"] == "on" and all((w.get(k) or {}).get("overflow") is False for k in ("job", "decode"))
    else:
        ok = True
    return "pass" if out_ok and ok else "fail"


# ---------------------------------------------------------------------------------------------
# WDDM sampling (the experiment's sampler; the shared GPU memory while a stretch of work runs)
# ---------------------------------------------------------------------------------------------

LUID_PS = r"""
$s = Get-Counter -Counter "\GPU Adapter Memory(*)\Dedicated Usage" -MaxSamples 1
$rows = @()
foreach ($smp in $s.CounterSamples) {
  if ($smp.Path -match '\(([^)]+)\)') {
    $rows += [pscustomobject]@{ instance = $Matches[1]; mb = [math]::Round($smp.CookedValue / 1MB, 1) }
  }
}
ConvertTo-Json -Compress -InputObject @($rows | Sort-Object mb -Descending)
"""


OVERFLOW_MB = 300.0


class Wddm:
    def __init__(self, csv: Path) -> None:
        self.csv, self.stop = csv, csv.with_suffix(".stop")
        self.proc, self.luid, self.t_start = None, None, None

    def __enter__(self):
        r = pg.run_cmd(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", LUID_PS], timeout=180)
        try:
            rows = json.loads(r.stdout.strip()) if r.stdout.strip() else []
            rows = [rows] if isinstance(rows, dict) else rows
            self.luid = max(rows, key=lambda x: float(x.get("mb") or 0))["instance"] if rows else None
        except (json.JSONDecodeError, ValueError):
            self.luid = None
        if self.luid:
            self.stop.unlink(missing_ok=True)
            self.proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SAMPLER), "-Luid", self.luid,
                 "-Csv", str(self.csv), "-Stop", str(self.stop)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            time.sleep(3.0)
            self.t_start = time.time()      # the samples before this are the baseline
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            time.sleep(2.0)
            self.stop.write_text("stop", encoding="ascii")
            try:
                self.proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                self.proc.kill()

    def window(self, t0: float, t1: float) -> dict:
        import statistics  # noqa: PLC0415

        rows = []
        if self.csv.is_file():
            for line in self.csv.read_text(encoding="ascii", errors="replace").splitlines():
                p = line.strip().split(",")
                if len(p) < 4 or p[0].startswith("timestamp"):
                    continue
                try:
                    t = datetime.strptime(p[0], "%Y-%m-%d %H:%M:%S.%f").timestamp()
                    rows.append({"t": t, "dedicated_mb": float(p[1]), "shared_mb": float(p[2])})
                except ValueError:
                    continue
        base = [r["shared_mb"] for r in rows if r["t"] < self.t_start] if self.t_start else []
        win = [r for r in rows if t0 <= r["t"] <= t1]
        b = statistics.median(base) if base else None
        if not win:
            return {"samples": 0, "shared_baseline_mb": b, "luid_found": bool(self.luid), "overflow": None}
        med = statistics.median(r["shared_mb"] for r in win)
        rise = None if b is None else round(med - b, 1)
        # The comfort calibration's rule (Docs/COMFORT_LIMIT_TABLE.md, v3'): the window's WDDM shared-memory
        # median +300 MB or more over the state before the measurement = overflow; fewer than 10 samples: not judged.
        overflow = None if (rise is None or len(win) < 10) else rise >= OVERFLOW_MB
        return {"samples": len(win), "dedicated_peak_mb": max(r["dedicated_mb"] for r in win),
                "shared_peak_mb": max(r["shared_mb"] for r in win), "shared_median_mb": round(med, 1),
                "shared_baseline_mb": b, "shared_median_rise_mb": rise,
                "shared_peak_rise_mb": None if b is None else round(max(r["shared_mb"] for r in win) - b, 1),
                "rule": f"median rise >= {OVERFLOW_MB:g} MB over the baseline (>= 10 samples)", "overflow": overflow}


# ---------------------------------------------------------------------------------------------
# Item runners
# ---------------------------------------------------------------------------------------------


def run_engine(item: str, out_dir: Path, dry: bool = False) -> dict:
    cmd = [str(pg.PY_ENG), "-B", str(HERE / "engine_items.py"), "--item", item, "--out", str(out_dir)]
    if dry:
        cmd.append("--dry-run")
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONUTF8": "1"}
    rec: dict[str, Any] = {"cmd": [pg.rel(c) if os.path.isabs(c) else c for c in cmd]}
    t0 = time.monotonic()
    wd = Wddm(out_dir / "wddm.csv") if ITEMS[item].get("wddm") and not dry else None
    with (out_dir / "console.log").open("w", encoding="utf-8", errors="replace") as fh:
        if wd:
            wd.__enter__()
        try:
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT)
            rec["exit_code"] = proc.wait(timeout=600 if dry else 5400)
        finally:
            if wd:
                wd.__exit__()
    rec["seconds_wall"] = round(time.monotonic() - t0, 1)
    eng = pg.read_json(out_dir / "engine.json")
    rec["engine"] = eng
    if wd:
        marks = eng.get("marks") or {}
        rec["wddm"] = {k: wd.window(v["t0"], v["t1"]) for k, v in marks.items() if "t1" in v}
    if item == "S1" and not dry and eng.get("items"):
        rec["mp4_vs_experiment_B"] = {}
        for it, r in eng["items"].items():
            ours = REPO / r["product_mp4"]
            theirs = SPIKE / "runs" / f"compare_{it}" / "B.mp4"
            rec["mp4_vs_experiment_B"][it] = pg.stream_md5(ours, "v") == pg.stream_md5(theirs, "v")
    ok = bool(eng.get("ok")) and (item != "S1" or dry or all((rec.get("mp4_vs_experiment_B") or {}).values()))
    if item == "S10" and not dry:
        rec["shared_overflow_inside"] = (rec.get("wddm") or {}).get("inside", {}).get("overflow")
        ok = ok and rec["shared_overflow_inside"] is False
    rec["verdict"] = "pass" if ok else "fail"
    return rec


def run_api(item: str, spec: dict, port: int, out_dir: Path) -> dict:
    """The pi-H3 gates' backend session; the Sol picks and checks added to each job."""
    rec: dict[str, Any] = {"jobs": []}
    _MANIFEST_OVERRIDE["sol_vae_path"] = ABSENT_SOL if spec.get("absent_sol") else None
    srv = pg.Server(port, item)
    cache: dict = {}
    wd = Wddm(out_dir / "wddm.csv") if spec.get("wddm") else None
    try:
        srv.start()
        loaded = None
        for jb in spec["jobs"]:
            if loaded != jb["file"]:
                ld = pg.load_weight(port, jb["file"])
                rec.setdefault("loads", []).append(ld)
                if not ld.get("ok"):
                    rec["jobs"].append({"job": jb["name"], "status": "load_error", "error": ld.get("error")})
                    break
                loaded = jb["file"]
            if wd:
                wd.__enter__()
            t0 = time.time()
            try:
                j = pg.run_api_job(port, item, jb, cache, out_dir)
            finally:
                if wd:
                    wd.__exit__()
            if wd:
                t_end = time.time()
                dec_t = next((s["t"] for s in j.get("stages") or [] if s.get("stage") == "decode"), None)
                j["wddm"] = {"job": wd.window(t0, t_end),
                             "decode": wd.window(t0 + dec_t, t_end) if dec_t is not None else {"samples": 0}}
            meta = pg.read_json(REPO / j["metadata_file"]) if j.get("metadata_file") else {}
            j["picks"] = pg.picks(meta)
            j["sol"] = sol_picks(meta)
            if j.get("status") == "completed":
                j["checks"] = pg.output_checks(j, jb["body"], jb.get("expect"))
                if jb.get("compare_to"):
                    j["compare"] = pg.compare(REPO / j["mp4"], pg.ref_f1())
            rec["jobs"].append(j)
            log(f"  {jb['name']}: {j.get('status')} sol_vae_used={j['sol']['sol_vae_used']} "
                f"vae_source={j['sol']['vae_source']} decode {j['sol']['decode_seconds']} s {j['sol']['decode_vram_mib']}")
            if j.get("status") != "completed":
                break
        if item == "S4" and len(rec["jobs"]) == 2 and all(j.get("mp4") for j in rec["jobs"]):
            a, b = (REPO / rec["jobs"][0]["mp4"], REPO / rec["jobs"][1]["mp4"])
            rec["jobs"][0]["compare_off"] = {"video_equal": pg.stream_md5(a, "v") == pg.stream_md5(b, "v")}
    except BaseException as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        log(f"{item}: stopped: {rec['error']}")
    finally:
        _MANIFEST_OVERRIDE["sol_vae_path"] = None
        try:
            srv.stop()
        except Exception as exc:  # noqa: BLE001
            srv.record["stop_error"] = f"{type(exc).__name__}: {exc}"
        rec["server"] = srv.record
    rec["verdict"] = "script_error" if rec.get("error") else judge_jobs(item, rec["jobs"])
    return rec


def run_off_path(item: str, spec: dict, port: int, out_dir: Path) -> dict:
    f0 = pg.run_selftest("S5_F0", out_dir)
    rec = {"F0": f0}
    if f0.get("verdict") != "pass":
        rec["verdict"] = "fail"
        return rec
    api = run_api(item, spec, port, out_dir)
    rec.update(api)
    return rec


# ---------------------------------------------------------------------------------------------
# Dry run, summary, main
# ---------------------------------------------------------------------------------------------


def dry_run() -> int:
    sys.path.insert(0, str(REPO))
    problems: list[str] = []
    out: dict[str, Any] = {"made_at": pg.now_iso(), "git": pg.git_info(), "items": {}}
    out["gate_now"] = pre = pg.preflight_checks()
    pg.log_gate(pre)
    files = [SPIKE / "runs" / f"gen_{x}" / "latent.safetensors" for x in "abcde"] + \
            [SPIKE / "runs" / f"compare_{x}" / "B.mp4" for x in "abcde"] + \
            [SPIKE / "sol_decoder.py", SAMPLER, pg.V2V_SRC, pg.INPAINT_SRC, pg.INPAINT_MASK, pg.LORA_DIR / pg.TURBO_LORA,
             pg.WEIGHTS / FL2VA, REPO / "models/MiniMaxH3/VAE/Sol/minimax_h3_sol_vae_int8_convrot.safetensors",
             pg.MIXED_SUMMARY, pg.PY_APP, pg.PY_ENG]
    out["files"] = {pg.rel(p): p.exists() for p in files}
    miss = [k for k, v in out["files"].items() if not v]
    log(f"files: {'all present' if not miss else 'MISSING ' + str(miss)}")
    problems += [f"missing {miss}"] if miss else []
    out["r1_md5"] = pg.r1_record()
    problems += [] if out["r1_md5"] else ["R1 MD5 record"]
    from api.models import GenerateChainRequest, GenerateRequest  # noqa: PLC0415
    from services import engines  # noqa: PLC0415

    for it, spec in ITEMS.items():
        e: dict[str, Any] = {"title": spec["title"], "kind": spec["kind"]}
        if spec["kind"] == "engine":
            d = HERE / "_dry" / it
            d.mkdir(parents=True, exist_ok=True)
            r = run_engine(it, d, dry=True)
            e["engine"] = r["engine"]
            ok = r["verdict"] == "pass"
            log(f"{it}: engine dry run {'ok' if ok else 'PROBLEM ' + str(r['engine'].get('error'))}")
            problems += [] if ok else [f"{it} engine dry run"]
        if spec["kind"] == "off_path":
            e["selftest_cmd"] = pg.run_selftest("__plan__", RUNS / it)["cmd"]
        for jb in spec.get("jobs") or []:
            b = json.loads(json.dumps(jb["body"]).replace('"@video:', '"dry:'))
            b = {k: v for k, v in b.items()}
            if jb["lora"]:
                b["loras"] = [{"name": Path(jb["lora"][0]).stem, "strength": jb["lora"][1]}]
            ph = [p for p in pg.placeholder_paths(jb["body"]) if not p.is_file()]
            try:
                if jb.get("endpoint") == "/generate/chain":
                    req = GenerateChainRequest(**b)
                    engines.reject_chain("h3", req)
                    engines.validate_chain_for_family("h3", req)
                else:
                    req = GenerateRequest(**b)
                    engines.reject_unsupported("h3", req)
                    engines.validate_for_family("h3", req)
                rules = "passed"
            except Exception as exc:  # noqa: BLE001
                rules = f"{type(exc).__name__}: {getattr(exc, 'detail', None) or exc}"
            e.setdefault("jobs", []).append({"job": jb["name"], "family_rules": rules, "uploads_missing": [pg.rel(p) for p in ph],
                                             "sol_vae": b.get("sol_vae", False)})
            log(f"{it}/{jb['name']}: family rules {rules}; sol_vae={b.get('sol_vae', False)}"
                + (f"; MISSING uploads {ph}" if ph else ""))
            problems += [] if rules == "passed" and not ph else [f"{it}/{jb['name']}"]
        out["items"][it] = e
    # S4's temporary descriptor parses and names a path with no file behind it
    _MANIFEST_OVERRIDE["sol_vae_path"] = ABSENT_SOL
    try:
        cfg = write_temp_config(PORT_DEFAULT)
        from services.base_models import load_base_models  # noqa: PLC0415

        d = load_base_models(RT / "manifests")["MiniMaxH3"]
        out["s4_descriptor"] = {"sol_vae_path": d.assets["sol_vae_path"],
                                "file_exists": (REPO / "models" / d.assets["sol_vae_path"]).exists(),
                                "temp_config": pg.rel(cfg)}
        ok = d.assets["sol_vae_path"] == ABSENT_SOL and not out["s4_descriptor"]["file_exists"]
    except Exception as exc:  # noqa: BLE001
        out["s4_descriptor"] = {"error": f"{type(exc).__name__}: {exc}"}
        ok = False
    finally:
        _MANIFEST_OVERRIDE["sol_vae_path"] = None
    log(f"S4 temporary descriptor: {out['s4_descriptor']}")
    problems += [] if ok else ["S4 descriptor"]
    out["problems"], out["ok"] = problems, not problems
    pg.write_json(HERE / "dry_run.json", out)
    log(f"dry-run: {'OK' if not problems else 'PROBLEMS ' + str(problems)}")
    return 0 if not problems else 1


def write_summary() -> None:
    rows, L = {}, ["# Sol-H3-VAE 段 1 の実機の確認（記録）", "", "| 項目 | 内容 | 判定 |", "|---|---|---|"]
    for it, spec in ITEMS.items():
        r = pg.read_json(RUNS / it / "result.json")
        if not r:
            continue
        rows[it] = {"title": spec["title"], "rule": spec["rule"], "verdict": r.get("verdict"),
                    "finished_at": r.get("finished_at")}
        L.append(f"| {it} | {spec['title']} | {r.get('verdict')} |")
    pg.write_json(HERE / "summary.json", {"made_at": pg.now_iso(), "items": rows})
    pg.write_text(HERE / "summary.md", "\n".join(L + ["", "合格の条件:", ""]
                                                  + [f"- {k}: {v['rule']}" for k, v in rows.items()]) + "\n")


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="Sol-H3-VAE stage 1 GPU checks")
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--gate-check", action="store_true")
    m.add_argument("--live", action="store_true")
    m.add_argument("--post-only", action="store_true")
    ap.add_argument("--only", default=None, help=f"comma list of {','.join(ORDER)}")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT)
    a = ap.parse_args(argv)
    order = ORDER
    if a.only:
        order = [x.strip() for x in a.only.split(",") if x.strip()]
        if [x for x in order if x not in ITEMS]:
            ap.error(f"--only: choose from {','.join(ITEMS)}")
    if a.port == pg.OWNER_PORT:
        ap.error(f"port {pg.OWNER_PORT} is the owner's server")
    mode = "dry_run" if a.dry_run else "gate_check" if a.gate_check else "post_only" if a.post_only else "live"
    log.open(LOG_DIR / f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    try:
        if a.dry_run:
            return dry_run()
        if a.gate_check:
            pre = pg.preflight_checks()
            pg.log_gate(pre)
            pg.write_json(HERE / "gate_check.json", pre)
            return 0 if not pre["blocking"] else 1
        if a.post_only:
            write_summary()
            return 0
        code = 0
        for it in order:
            spec, out_dir = ITEMS[it], RUNS / it
            prev = pg.read_json(out_dir / "result.json")
            if prev.get("verdict") and prev["verdict"] != "refused" and not a.redo:
                log(f"{it}: done before ({prev['verdict']}); skipped (--redo to run again)")
                continue
            if out_dir.exists() and prev:
                out_dir.rename(out_dir.with_name(f"{it}.old-{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
            out_dir.mkdir(parents=True, exist_ok=True)
            rec: dict[str, Any] = {"item": it, "title": spec["title"], "rule": spec["rule"], "started_at": pg.now_iso(),
                                   "git": pg.git_info()}
            log(f"=== {it}: {spec['title']} ===")
            rec["gate"] = pre = pg.preflight_checks()
            pg.log_gate(pre)
            if pre["blocking"]:
                rec.update(verdict="refused", finished_at=pg.now_iso())
                pg.write_json(out_dir / "result.json", rec)
                code = 1
                break
            try:
                if spec["kind"] == "engine":
                    rec.update(run_engine(it, out_dir))
                elif spec["kind"] == "off_path":
                    rec.update(run_off_path(it, spec, a.port, out_dir))
                else:
                    rec.update(run_api(it, spec, a.port, out_dir))
            except BaseException as exc:  # noqa: BLE001
                rec.update(verdict="script_error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            rec["finished_at"] = pg.now_iso()
            rec["workers_after"] = [p["pid"] for p in pg.all_processes() if pg.WORKER_RE.search(p["cmdline"] or "")]
            pg.write_json(out_dir / "result.json", rec)
            log(f"{it}: verdict {rec['verdict']}")
            if rec["verdict"] != "pass":
                code = 3
        pg.scrub_files()
        write_summary()
        return code
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
