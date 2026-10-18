"""MiniMax H3 の VRAM 溢れの計測台（台帳 §1-101 第 1 段・計画 PLAN_measure.md）。

腕（計測条件の組）1 本を 1 プロセスで回す。製品コードは変えず、
``engine_h3.worker`` の ``_do_load`` / ``_do_generate`` をセルフテストと同じ
合成メッセージで呼び、割り込みはワーカーの関数とインスタンスの属性を包むだけで行う。

出力（すべて ``runs/<label>/``。この台本のフォルダからの絶対パス）:
  run.json                    引数・実際に効いた値・全イベント・progress の時刻・index 1/2 の採取・done・所要時間
  memory_stats_step1.json     ノイズ除去の 1 ステップ目の終わり（progress index 1）の memory_stats ほか
  memory_stats_step2.json     2 ステップ目の終わり（index 2）
  snapshot_step1.pickle       --record-history のとき。index 1 のメモリの確保と解放の履歴
  snapshot_error.pickle       --record-history で例外（OOM など）が起きたとき
  wddm.csv                    WDDM（Windows の GPU メモリ管理）の専用／共有メモリ（vram_sampler2.ps1）
  dmon.txt                    nvidia-smi dmon -s ut -o DT -d 1（PCIe の受信・送信と使用率）
  gpu_process_memory_preload.csv / _step1.csv  プロセスごとの GPU メモリ（読み込み前・index 1）
  luid.json                   この回に取り直した LUID（アダプタの識別子）の記録（キャッシュではない）
  stdout.log                  この台本の標準出力と標準エラーの写し
  out.mp4                     生成物

終了コード: 0＝完了（OOM などの例外も「結果」として記録して 0）／2＝実際に効いた値が引数と合わない
（run.json の effective_mismatch）／1＝台本自体の失敗（引数・パス・採取器）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# --- 出力先は engine_h3.worker の import より前に絶対パスで確定する --------------------
# （worker の import で os.chdir(ROOT) が走る。worker.py:91。相対パスは作業ツリーを汚す）
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # Nz-Videomni
RUNS = HERE / "runs"
SAMPLER = HERE / "vram_sampler2.ps1"
MODELS = ROOT / "models"

# 既定のモデルのパス（scripts/manifests/40-minimaxh3.json の default_file / assets が正本）
DEFAULT_TRANSFORMER = MODELS / "MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DEFAULT_TEXT_ENCODER = MODELS / "MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
DEFAULT_VIDEO_VAE = MODELS / "MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = MODELS / "MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors"
DEFAULT_TOKENIZER_DIR = MODELS / "MiniMaxH3/TextEncoder/qwen3vl_tokenizer"

MIB = 2 ** 20
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def now_iso(t: float | None = None) -> str:
    t = time.time() if t is None else t
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + ".%03d" % int((t % 1) * 1000)


def say(msg: str) -> None:
    print(f"[h3_vram {now_iso()}] {msg}", file=sys.stderr, flush=True)


# === 標準出力・標準エラーの写し ==============================================
class Tee:
    def __init__(self, stream, fh, lock):
        self.stream, self.fh, self.lock = stream, fh, lock

    def write(self, text):
        with self.lock:
            try:
                self.stream.write(text)
            except Exception:  # noqa: BLE001 -- 端末の文字コードで落ちても記録は続ける
                pass
            self.fh.write(text)
            self.fh.flush()
        return len(text)

    def flush(self):
        with self.lock:
            try:
                self.stream.flush()
            except Exception:  # noqa: BLE001
                pass
            self.fh.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


def install_tee(out_dir: Path):
    fh = (out_dir / "stdout.log").open("a", encoding="utf-8", errors="replace")
    lock = threading.Lock()
    sys.stdout = Tee(sys.__stdout__, fh, lock)
    sys.stderr = Tee(sys.__stderr__, fh, lock)
    return fh


# === LUID の取得（outputs/comfort-calib-2026-10-04/calib.py の写し。毎回取り直す） =====
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


def discover_luid(out_dir: Path) -> dict:
    """キャッシュは使わない（古い LUID だと採取器が例外にならず 0 を書き続ける）。"""
    out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", LUID_PS],
                         capture_output=True, text=True, timeout=180,
                         creationflags=CREATE_NO_WINDOW).stdout.strip()
    rows = json.loads(out) if out else []
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        raise RuntimeError("GPU Adapter Memory のインスタンスが見つかりません（LUID を取得できない）")
    rows.sort(key=lambda r: float(r.get("mb") or 0.0), reverse=True)
    info = {"chosen_luid": rows[0]["instance"], "chosen_dedicated_mb": rows[0]["mb"],
            "candidates": rows, "discovered_at": now_iso(), "note": "この回に取り直した値の記録。次の回では使わない"}
    (out_dir / "luid.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    say(f"LUID: {info['chosen_luid']} (dedicated {rows[0]['mb']} MB); candidates={[(r['instance'], r['mb']) for r in rows]}")
    return info


# === プロセスごとの GPU メモリ ================================================
PROC_PS = r"""
$s = Get-Counter -Counter "\GPU Process Memory(*)\Dedicated Usage", "\GPU Process Memory(*)\Shared Usage" -MaxSamples 1
$names = @{}
foreach ($p in Get-Process) { $names[[int]$p.Id] = $p.ProcessName }
$rows = @()
foreach ($smp in $s.CounterSamples) {
  if ($smp.Path -match '\(pid_(\d+)_(luid_[^)]+)\)\\(dedicated|shared) usage') {
    $procId = [int]$Matches[1]
    $rows += [pscustomobject]@{ pid = $procId; name = $names[$procId]; luid = $Matches[2]; kind = $Matches[3]; mb = [math]::Round($smp.CookedValue / 1MB, 1) }
  }
}
ConvertTo-Json -Compress -InputObject @($rows)
"""


def sample_process_memory(out_dir: Path, tag: str, luid: str) -> dict:
    """``\\GPU Process Memory(*)`` の専用・共有を 1 回採って CSV に書き、要約を返す。失敗しても止めない。"""
    t0 = time.time()
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", PROC_PS],
                             capture_output=True, text=True, timeout=180,
                             creationflags=CREATE_NO_WINDOW).stdout.strip()
        rows = json.loads(out) if out else []
        if isinstance(rows, dict):
            rows = [rows]
    except Exception as exc:  # noqa: BLE001
        return {"tag": tag, "error": repr(exc), "t_wall": t0}
    me = os.getpid()
    per: dict[int, dict] = {}
    for r in rows:
        if str(r.get("luid", "")).lower() != luid.lower():
            continue
        e = per.setdefault(int(r["pid"]), {"pid": int(r["pid"]), "name": r.get("name") or "",
                                           "dedicated_mb": 0.0, "shared_mb": 0.0})
        e["dedicated_mb" if r["kind"].lower() == "dedicated" else "shared_mb"] += float(r["mb"] or 0.0)
    path = out_dir / f"gpu_process_memory_{tag}.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "pid", "name", "is_self", "dedicated_MB", "shared_MB"])
        for e in sorted(per.values(), key=lambda x: -x["dedicated_mb"]):
            w.writerow([now_iso(t0), e["pid"], e["name"], int(e["pid"] == me),
                        round(e["dedicated_mb"], 1), round(e["shared_mb"], 1)])
    mine = per.get(me, {"dedicated_mb": 0.0, "shared_mb": 0.0})
    others = [e for p, e in per.items() if p != me]
    return {
        "tag": tag, "t_wall": t0, "iso": now_iso(t0), "seconds_taken": round(time.time() - t0, 2),
        "csv": path.name, "processes": len(per),
        "self_dedicated_mb": round(mine["dedicated_mb"], 1), "self_shared_mb": round(mine["shared_mb"], 1),
        "others_dedicated_mb": round(sum(e["dedicated_mb"] for e in others), 1),
        "others_shared_mb": round(sum(e["shared_mb"] for e in others), 1),
    }


# === 採取器（calib.py の Monitors の写し。WDDM 採取器と dmon を 1 単位で起動・停止） =====
class Monitors:
    def __init__(self, out_dir: Path, luid: str):
        self.out_dir = out_dir
        self.luid = luid
        self.wddm_csv = out_dir / "wddm.csv"
        self.dmon_txt = out_dir / "dmon.txt"
        self.stop_file = out_dir / "stop_sampler.txt"
        self._sampler = None
        self._dmon = None
        self._dmon_fh = None

    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file.unlink(missing_ok=True)
        self._sampler = subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SAMPLER),
             "-Luid", self.luid, "-Csv", str(self.wddm_csv), "-Stop", str(self.stop_file)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        self._dmon_fh = self.dmon_txt.open("w", encoding="ascii", errors="replace")
        self._dmon = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "ut", "-o", "DT", "-d", "1"],
            stdout=self._dmon_fh, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW)
        time.sleep(2.0)

    def stop(self, grace: float = 2.0) -> None:
        time.sleep(grace)
        if self._dmon is not None:
            try:
                self._dmon.terminate()
                self._dmon.wait(timeout=15)
            except Exception:  # noqa: BLE001
                try:
                    self._dmon.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._dmon = None
        if self._dmon_fh is not None:
            try:
                self._dmon_fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._dmon_fh = None
        if self._sampler is not None:
            self.stop_file.write_text("stop", encoding="ascii")
            try:
                self._sampler.wait(timeout=30)
            except Exception:  # noqa: BLE001
                try:
                    self._sampler.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._sampler = None
        self.stop_file.unlink(missing_ok=True)


def read_wddm(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="ascii", errors="replace") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({"ts": r["timestamp"], "ded": float(r["dedicated_MB"]), "shr": float(r["shared_MB"]),
                             "dt": float(r["dt_ms"])})
            except Exception:  # noqa: BLE001
                continue
    return rows


def wddm_summary(path: Path) -> dict:
    rows = read_wddm(path)
    ded = [r["ded"] for r in rows]
    shr = [r["shr"] for r in rows]
    dts = sorted(r["dt"] for r in rows[1:])
    return {
        "samples": len(rows),
        "nonzero_dedicated_samples": sum(1 for v in ded if v > 0),
        "dedicated_mb_min": min(ded) if ded else None, "dedicated_mb_max": max(ded) if ded else None,
        "shared_mb_min": min(shr) if shr else None, "shared_mb_max": max(shr) if shr else None,
        "dt_ms_median": dts[len(dts) // 2] if dts else None,
    }


def wddm_window(path: Path, t0: float, t1: float) -> dict:
    """壁時計 t0〜t1 の間の WDDM の採取の中央値と最大（採取器の時刻はローカル時刻の文字列）。"""
    rows = []
    for r in read_wddm(path):
        try:
            tt = time.mktime(time.strptime(r["ts"][:19], "%Y-%m-%d %H:%M:%S")) + float("0" + r["ts"][19:])
        except Exception:  # noqa: BLE001
            continue
        if t0 <= tt <= t1:
            rows.append(r)
    ded = sorted(r["ded"] for r in rows)
    shr = sorted(r["shr"] for r in rows)
    med = lambda v: v[len(v) // 2] if v else None  # noqa: E731
    return {"samples": len(rows), "dedicated_mb_median": med(ded), "shared_mb_median": med(shr),
            "dedicated_mb_max": ded[-1] if ded else None, "shared_mb_max": shr[-1] if shr else None}


# === torch まわり =============================================================
STAT_KEYS = (
    "allocated_bytes.all.current", "allocated_bytes.all.peak",
    "reserved_bytes.all.current", "reserved_bytes.all.peak",
    "inactive_split_bytes.all.current", "inactive_split_bytes.all.peak",
    "num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free",
)


def mem_stats(torch) -> dict:
    s = torch.cuda.memory_stats()
    out = {k: s.get(k) for k in STAT_KEYS}
    for k in STAT_KEYS:
        if k.endswith((".current", ".peak")) and out.get(k) is not None:
            out[k.replace("_bytes", "_mib", 1)] = round(out[k] / MIB, 1)
    return out


def mem_info(torch) -> dict:
    free, total = torch.cuda.mem_get_info()
    return {"free_bytes": free, "total_bytes": total, "free_mib": round(free / MIB, 1),
            "total_mib": round(total / MIB, 1), "used_mib": round((total - free) / MIB, 1)}


def sdpa_paths(torch) -> dict:
    out = {}
    for name in ("is_flash_attention_available", "mem_efficient_sdp_enabled", "flash_sdp_enabled",
                 "math_sdp_enabled", "cudnn_sdp_enabled"):
        fn = getattr(torch.backends.cuda, name, None)
        try:
            out[name] = fn() if fn is not None else "absent"
        except Exception as exc:  # noqa: BLE001
            out[name] = f"error: {exc!r}"
    return out


def memory_fraction(torch):
    try:
        return float(torch.cuda.get_per_process_memory_fraction())
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc!r}"


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# === 引数 =====================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MiniMax H3 の VRAM 計測台（腕 1 本＝1 プロセス）")
    p.add_argument("--label", required=True, help="runs/<label>/ に書く（例: T1_A_sage_b8_nocap_hist）")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=1152)
    p.add_argument("--num-frames", type=int, default=124)
    p.add_argument("--attention-backend", choices=("sdpa", "sage"), default="sdpa")
    p.add_argument("--blocks-on-gpu", type=int, default=8)
    p.add_argument("--memory-fraction", type=float, default=None,
                   help="torch.cuda.set_per_process_memory_fraction の値。省略で上限なし")
    p.add_argument("--record-history", action=argparse.BooleanOptionalAction, default=False,
                   help="メモリの確保と解放の履歴を _do_generate の直前から採り、index 1 でスナップショットを書く")
    p.add_argument("--history-entries", type=int, default=2_000_000, help="履歴の件数の上限（max_entries）")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--idle-seconds", type=float, default=10.0, help="読み込み前の待機の採取の秒数")
    p.add_argument("--image", default=None, help="I2V: 先頭フレームの画像（images=[{path, frame_idx: 0}]）")
    p.add_argument("--prompt-file", default=None, help="UTF-8 のファイルの中身をプロンプトにする（省略でセルフテストの既定）")
    p.add_argument("--keep-resident", action="store_true", help="generate の keep_resident を True にする")
    p.add_argument("--deterministic", action="store_true", help="load の deterministic を True にする")
    # VRAM 16GB mode（台帳 §1-101 弾 A の門。既定では従来の振る舞いのまま）
    p.add_argument("--vram16", action="store_true", help="generate の vram16_mode を True にする")
    p.add_argument("--vram16-head-group", type=int, default=None, help="門 A-G5: Vram16Plan.head_group を台本から上書き")
    p.add_argument("--vram16-row-chunk", type=int, default=None, help="門 A-G5: Vram16Plan.row_chunk を台本から上書き")
    p.add_argument("--vram16-blocks", type=int, default=None,
                   help="門 A-G5: lowvram の VRAM16_BLOCKS_ON_GPU を台本から上書き（on のときの常駐数の上限）")
    p.add_argument("--runs-dir", default=None, help="runs/ の代わりの出力の親フォルダ（省略で従来の runs/）")
    p.add_argument("--dry-run", action="store_true",
                   help="モデルを読まず、パス・LUID・採取器・小さな確保とスナップショットの書き出しだけを確かめる")
    # モデルのパス（engine_h3/worker.py の _selftest_parser と同じ名前）
    p.add_argument("--transformer", default=str(DEFAULT_TRANSFORMER))
    p.add_argument("--text-encoder", default=str(DEFAULT_TEXT_ENCODER))
    p.add_argument("--video-vae", default=str(DEFAULT_VIDEO_VAE))
    p.add_argument("--audio-vae", default=str(DEFAULT_AUDIO_VAE))
    p.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    args = p.parse_args(argv)
    for name in ("transformer", "text_encoder", "video_vae", "audio_vae", "tokenizer_dir"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))   # chdir の前に絶対パス化
    for name in ("image", "prompt_file"):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if args.runs_dir:
        args.runs_dir = str(Path(args.runs_dir).resolve())
    if args.memory_fraction is not None and not (0.0 < args.memory_fraction <= 1.0):
        p.error("--memory-fraction は 0 より大きく 1 以下")
    return args


def check_paths(args) -> dict:
    out = {}
    for name in ("transformer", "text_encoder", "video_vae", "audio_vae"):
        f = Path(getattr(args, name))
        out[name] = {"path": str(f), "exists": f.is_file(), "bytes": f.stat().st_size if f.is_file() else None}
    d = Path(args.tokenizer_dir)
    out["tokenizer_dir"] = {"path": str(d), "exists": d.is_dir(),
                            "files": sorted(x.name for x in d.iterdir()) if d.is_dir() else None}
    return out


# === 試運転 ===================================================================
def dry_run(args, out_dir: Path) -> int:
    result: dict = {"label": args.label, "started": now_iso(), "args": vars(args), "checks": {}}
    ok = True
    paths = check_paths(args)
    result["paths"] = paths
    result["checks"]["model_files_exist"] = all(v["exists"] for v in paths.values())
    ok &= result["checks"]["model_files_exist"]

    luid = discover_luid(out_dir)
    result["luid"] = luid
    mon = Monitors(out_dir, luid["chosen_luid"])
    mon.start()
    t_start = time.time()
    try:
        # worker の import（chdir が走っても出力先が動かないことの確認。torch は読まない）
        import engine_h3.worker as worker  # noqa: PLC0415
        result["worker_import"] = {"module": worker.__file__, "cwd_after_import": os.getcwd(),
                                   "out_dir": str(out_dir), "selftest_parser_options": sorted(
                                       a.dest for a in worker._selftest_parser()._actions)}
        import torch  # noqa: PLC0415

        result["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                           "device": torch.cuda.get_device_name(0)}
        result["sdpa_paths"] = sdpa_paths(torch)
        result["mem_info_before"] = mem_info(torch)
        if args.memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        result["per_process_memory_fraction"] = memory_fraction(torch)
        result["process_memory"] = sample_process_memory(out_dir, "preload", luid["chosen_luid"])
        torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python",
                                                 max_entries=args.history_entries)
        x = torch.empty(64 * MIB, dtype=torch.uint8, device="cuda")
        x.fill_(1)
        y = torch.empty(3 * MIB + 123, dtype=torch.uint8, device="cuda")   # 丸めの確認用
        torch.cuda.synchronize()
        result["memory_stats_after_alloc"] = mem_stats(torch)
        snap = out_dir / "snapshot_step1.pickle"
        torch.cuda.memory._dump_snapshot(str(snap))
        del y
        torch.cuda.memory._record_memory_history(enabled=None)
        result["snapshot"] = {"path": str(snap), "bytes": snap.stat().st_size}
        result["checks"]["snapshot_written"] = snap.stat().st_size > 0
        del x
        # 5 秒の採取（採取器の起動 2 秒を含めて 5 秒以上）
        while time.time() - t_start < 5.0:
            time.sleep(0.25)
    except Exception as exc:  # noqa: BLE001
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        ok = False
    finally:
        mon.stop()
    w = wddm_summary(mon.wddm_csv)
    result["wddm"] = w
    result["checks"]["wddm_dedicated_nonzero"] = bool(w["samples"] and w["nonzero_dedicated_samples"] == w["samples"])
    dmon_lines = [ln for ln in mon.dmon_txt.read_text(encoding="ascii", errors="replace").splitlines()
                  if ln.strip() and not ln.startswith("#")] if mon.dmon_txt.exists() else []
    result["dmon"] = {"data_lines": len(dmon_lines), "head": mon.dmon_txt.read_text(
        encoding="ascii", errors="replace").splitlines()[:4] if mon.dmon_txt.exists() else []}
    result["checks"]["dmon_has_rows"] = len(dmon_lines) > 0
    for k in ("wddm_dedicated_nonzero", "dmon_has_rows", "snapshot_written"):
        ok &= bool(result["checks"].get(k))
    result["ok"] = bool(ok)
    result["finished"] = now_iso()
    write_json(HERE / "dryrun.json", result)
    write_json(out_dir / "dryrun.json", result)
    say(f"dry-run {'OK' if ok else 'NG'}: checks={result['checks']}")
    return 0 if ok else 1


def build_opts_det_check(rec, args, mismatch, eff) -> bool:
    opts = (rec.get("build_report") or {}).get("options") or {}
    if not opts:
        return False
    eff["build_deterministic"] = opts.get("deterministic")
    if bool(opts.get("deterministic")) != bool(args.deterministic):
        mismatch.append(f"build_report.deterministic={opts.get('deterministic')} != {bool(args.deterministic)}")
    return True


# === 本番 =====================================================================
def run(args, out_dir: Path) -> int:
    t_script = time.perf_counter()
    rec: dict = {"label": args.label, "started": now_iso(), "pid": os.getpid(), "args": vars(args),
                 "events": [], "progress": [], "samples": {}, "error": None}
    run_json = out_dir / "run.json"
    paths = check_paths(args)
    rec["paths"] = paths
    if not all(v["exists"] for v in paths.values()):
        rec["error"] = {"type": "MissingModel", "message": "モデルのファイルが見つかりません"}
        write_json(run_json, rec)
        say("model files missing; see run.json")
        return 1

    luid = discover_luid(out_dir)
    rec["luid"] = luid
    mon = Monitors(out_dir, luid["chosen_luid"])
    mon.start()
    rec["monitors_started"] = now_iso()
    hist_on = False
    torch = None
    try:
        import engine_h3.worker as worker  # noqa: PLC0415 -- ここで chdir(ROOT) が走る

        overrides = {}
        if args.vram16_head_group is not None or args.vram16_row_chunk is not None or args.vram16_blocks is not None:
            # 製品の既定は変えず、このプロセスの中だけで差し替える（門 A-G5）
            from engine_h3 import lowvram  # noqa: PLC0415

            base_plan = lowvram.Vram16Plan
            kw = {}
            if args.vram16_head_group is not None:
                kw["head_group"] = args.vram16_head_group
            if args.vram16_row_chunk is not None:
                kw["row_chunk"] = args.vram16_row_chunk
            if kw:
                lowvram.Vram16Plan = lambda: base_plan(**kw)
            if args.vram16_blocks is not None:
                lowvram.VRAM16_BLOCKS_ON_GPU = args.vram16_blocks
            overrides = {**kw, "VRAM16_BLOCKS_ON_GPU": lowvram.VRAM16_BLOCKS_ON_GPU,
                         "VRAM16_BLOCKS_S_THRESHOLD": getattr(lowvram, "VRAM16_BLOCKS_S_THRESHOLD", None)}
        rec["vram16_overrides"] = overrides

        # --- 読み込み前の待機（他プロセスとこのプロセスの土台の採取） ---
        t_idle0 = time.time()
        while time.time() - t_idle0 < args.idle_seconds:
            time.sleep(0.25)
        rec["idle_window"] = {"start_wall": t_idle0, "end_wall": time.time(),
                              "start_iso": now_iso(t_idle0), "end_iso": now_iso()}
        import torch  # noqa: PLC0415

        rec["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0)}
        rec["sdpa_paths"] = sdpa_paths(torch)
        rec["mem_info_preload"] = mem_info(torch)
        rec["samples"]["process_memory_preload"] = sample_process_memory(out_dir, "preload", luid["chosen_luid"])
        if args.memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
            cap = args.memory_fraction * rec["mem_info_preload"]["total_mib"]
            rec["memory_cap"] = {"fraction": args.memory_fraction, "cap_mib": round(cap, 1),
                                 "preload_free_mib": rec["mem_info_preload"]["free_mib"],
                                 "cap_exceeds_preload_free_minus_800mib":
                                     cap > rec["mem_info_preload"]["free_mib"] - 800}
        rec["per_process_memory_fraction"] = memory_fraction(torch)

        # --- _emit を包んで全イベントを記録（_selftest_generate と同じ方法） ---
        real_emit = worker._emit

        def capturing_emit(event: str, **fields):
            rec["events"].append({"t_mono": time.perf_counter(), "t_wall": time.time(), "event": event, **fields})
            real_emit(event, **fields)

        worker._emit = capturing_emit

        load_msg = {
            "op": "load", "transformer_path": args.transformer, "text_encoder_path": args.text_encoder,
            "video_vae_path": args.video_vae, "audio_vae_path": args.audio_vae,
            "tokenizer_dir": args.tokenizer_dir, "blocks_on_gpu": args.blocks_on_gpu,
            "attention_backend": args.attention_backend, "block_swap_prefetch": True, "deterministic": bool(args.deterministic),
        }
        rec["load_msg"] = load_msg
        t0 = time.perf_counter()
        worker._do_load(load_msg)
        rec["load_seconds"] = round(time.perf_counter() - t0, 2)
        pipe = worker._PIPE
        rec["build_report"] = pipe.build_report

        # --- _progress を包む（pipeline_h3.py:363 で保持している呼び出し先） ---
        orig_progress = pipe._progress

        def progress_hook(stage, index, total):
            entry = {"stage": stage, "index": int(index), "total": int(total)}
            if stage == "stage1_denoise":
                torch.cuda.synchronize()
            entry["t_mono"] = time.perf_counter()
            entry["t_wall"] = time.time()
            entry["iso"] = now_iso(entry["t_wall"])
            if stage == "stage1_denoise" and int(index) >= 1:
                i = int(index)
                s: dict = {"index": i, "t_mono_enter": entry["t_mono"], "t_wall_enter": entry["t_wall"]}
                s["memory_stats"] = mem_stats(torch)
                s["mem_get_info"] = mem_info(torch)
                if i == 1:
                    svc = getattr(pipe, "_swap_service", None)
                    eng = getattr(svc, "_prefetch_engine", None) if svc is not None else None
                    s["swap"] = {
                        "service": svc is not None,
                        "blocks_on_gpu": getattr(svc, "blocks_on_gpu", None),
                        "last_prefetch_used": getattr(svc, "last_prefetch_used", None),
                        "prefetch_engine": eng is not None,
                        "arenas": len(getattr(eng, "_arenas", []) or []) if eng is not None else None,
                        "blocks": len(getattr(eng, "_blocks", []) or []) if eng is not None else None,
                    }
                    s["process_memory"] = sample_process_memory(out_dir, "step1", luid["chosen_luid"])
                    if hist_on:
                        snap = out_dir / "snapshot_step1.pickle"
                        ts = time.perf_counter()
                        torch.cuda.memory._dump_snapshot(str(snap))
                        s["snapshot"] = {"path": snap.name, "bytes": snap.stat().st_size,
                                         "seconds": round(time.perf_counter() - ts, 2)}
                s["t_mono_done"] = time.perf_counter()
                s["t_wall_done"] = time.time()
                rec["samples"][f"step{i}"] = s
                write_json(out_dir / f"memory_stats_step{i}.json", s)
            rec["progress"].append(entry)
            return orig_progress(stage, index, total)

        pipe._progress = progress_hook

        prompt = (Path(args.prompt_file).read_text(encoding="utf-8").strip() if args.prompt_file
                  else worker._SELFTEST_PROMPT)
        gen_msg = {
            "op": "generate", "prompt": prompt, "seed": args.seed,
            "width": args.width, "height": args.height, "num_frames": args.num_frames,
            "frame_rate": worker.FRAME_RATE, "num_inference_steps": args.steps,
            "shift_video": worker.DEFAULT_SHIFT_VIDEO, "shift_audio": worker.DEFAULT_SHIFT_AUDIO,
            "output_path": str(out_dir / "out.mp4"), "images": [{"path": args.image, "frame_idx": 0}] if args.image else [],
            "keep_resident": bool(args.keep_resident),
            "attention_backend": args.attention_backend, "block_swap_prefetch": True,
        }
        if args.vram16:
            gen_msg["vram16_mode"] = True
        rec["generate_msg"] = gen_msg
        if args.record_history:
            torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python",
                                                     max_entries=args.history_entries)
            hist_on = True
        t1 = time.perf_counter()
        rec["generate_started_iso"] = now_iso()
        worker._do_generate(gen_msg)
        rec["generate_seconds"] = round(time.perf_counter() - t1, 2)
    except BaseException as exc:  # noqa: BLE001 -- OOM なども「結果」として記録する
        rec["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                        "iso": now_iso()}
        say(f"exception recorded: {type(exc).__name__}: {str(exc)[:300]}")
        if hist_on and torch is not None:
            try:
                snap = out_dir / "snapshot_error.pickle"
                torch.cuda.memory._dump_snapshot(str(snap))
                rec["error"]["snapshot"] = snap.name
            except Exception as exc2:  # noqa: BLE001
                rec["error"]["snapshot_error"] = repr(exc2)
        if torch is not None:
            try:
                rec["error"]["memory_stats"] = mem_stats(torch)
            except Exception:  # noqa: BLE001
                pass
    finally:
        if hist_on and torch is not None:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:  # noqa: BLE001
                pass
        mon.stop()
        rec["monitors_stopped"] = now_iso()

    # --- 集計 ---
    done = next((e for e in rec["events"] if e.get("event") == "done"), None)
    rec["done"] = done
    s1, s2 = rec["samples"].get("step1"), rec["samples"].get("step2")
    if s1 and s2:
        rec["timing"] = {
            "step2_seconds": round(s2["t_mono_enter"] - s1["t_mono_done"], 3),
            "step1_hook_seconds": round(s1["t_mono_done"] - s1["t_mono_enter"], 3),
            "rule": "2 ステップ目＝index 2 の時刻 − index 1 の割り込みを終えた時刻（synchronize 後）",
        }
        p0 = next((p for p in rec["progress"] if p["stage"] == "stage1_denoise" and p["index"] == 0), None)
        if p0:
            rec["timing"]["step1_seconds_incl_prefetch_startup"] = round(s1["t_mono_enter"] - p0["t_mono"], 3)
        d = {k: (s2["memory_stats"].get(k) or 0) - (s1["memory_stats"].get(k) or 0)
             for k in ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_ooms")}
        rec["timing"]["counts_step1_to_step2"] = d
    steps_s = [rec["samples"][f"step{i}"] for i in range(1, args.steps + 1) if f"step{i}" in rec["samples"]]
    if steps_s:
        p0 = next((p for p in rec["progress"] if p["stage"] == "stage1_denoise" and p["index"] == 0), None)
        secs, counts = [], []
        for j, s in enumerate(steps_s):
            prev_done = (steps_s[j - 1]["t_mono_done"] if j else (p0["t_mono"] if p0 else None))
            secs.append(round(s["t_mono_enter"] - prev_done, 3) if prev_done is not None else None)
            ms = s["memory_stats"]
            counts.append({k: ms.get(k) for k in ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_ooms")})
        rec.setdefault("timing", {})["step_seconds"] = secs
        rec["timing"]["step_seconds_rule"] = "index k の時刻 − index k−1 の割り込みを終えた時刻（k=1 は index 0 の時刻から。先読みの立ち上がりを含む）"
        rec["timing"]["counts_cumulative_by_index"] = counts
    if s1 and torch is not None:
        pm_pre = rec["samples"].get("process_memory_preload") or {}
        mi = s1["mem_get_info"]
        reserved_mib = (s1["memory_stats"].get("reserved_bytes.all.current") or 0) / MIB
        if "others_dedicated_mb" in pm_pre:
            rec["non_torch_estimate_mib"] = {
                "formula": "(total-free)@index1 - reserved@index1 - others_dedicated@preload",
                "value": round(mi["used_mib"] - reserved_mib - pm_pre["others_dedicated_mb"], 1),
            }
    rec["wddm"] = wddm_summary(out_dir / "wddm.csv")
    dn = [p for p in rec["progress"] if p["stage"] == "stage1_denoise"]
    if len(dn) >= 2:
        rec["wddm"]["denoise_window"] = wddm_window(out_dir / "wddm.csv", dn[0]["t_wall"], dn[-1]["t_wall"])
        rec["wddm"]["denoise_window"]["rule"] = "stage1_denoise の index 0 から最後の index までの壁時計"
    if done:
        rec["vram16"] = {"vram16_mode_used": done.get("vram16_mode_used"),
                         "record": ((done.get("h3") or {}).get("phases") or {}).get("denoise", {}).get("vram16")}
    iw = rec.get("idle_window")
    if iw:
        rec["wddm"]["idle"] = wddm_window(out_dir / "wddm.csv", iw["start_wall"], iw["end_wall"])
        if s1 and torch is not None and rec["wddm"]["idle"].get("dedicated_mb_median") is not None:
            # 第 2 の見積もり: 他プロセスの持ち分＝待機中（このプロセスの CUDA の文脈を作る前）のアダプタ全体の専用
            # （プロセスごとのカウンタは dwm などの共有の面を重ねて数えるらしく、合計がアダプタ全体を超えるため）
            reserved_mib = (s1["memory_stats"].get("reserved_bytes.all.current") or 0) / MIB
            rec["non_torch_estimate_mib_adapter_idle"] = {
                "formula": "(total-free)@index1 - reserved@index1 - wddm_dedicated_median@idle",
                "value": round(s1["mem_get_info"]["used_mib"] - reserved_mib
                               - rec["wddm"]["idle"]["dedicated_mb_median"], 1),
            }

    # --- 実際に効いた値と引数の突き合わせ ---
    eff: dict = {}
    mismatch: list[str] = []
    opts = (rec.get("build_report") or {}).get("options") or {}
    if opts:
        eff["build_blocks_on_gpu"] = opts.get("blocks_on_gpu")
        eff["build_attention_backend"] = opts.get("attention_backend")
        if opts.get("blocks_on_gpu") != args.blocks_on_gpu:
            mismatch.append(f"build_report.blocks_on_gpu={opts.get('blocks_on_gpu')} != {args.blocks_on_gpu}")
        if opts.get("attention_backend") != args.attention_backend:
            mismatch.append(f"build_report.attention_backend={opts.get('attention_backend')} != {args.attention_backend}")
    if build_opts_det_check(rec, args, mismatch, eff):
        pass
    if done:
        if args.keep_resident or done.get("keep_resident_used") is not None:
            eff["keep_resident_used"] = done.get("keep_resident_used")
            want_kr = "on" if args.keep_resident else "off"
            if done.get("keep_resident_used") != want_kr:
                mismatch.append(f"done.keep_resident_used={done.get('keep_resident_used')} != {want_kr}")
        eff["images_used"] = done.get("images_used")
        want_im = 1 if args.image else 0
        if done.get("images_used") != want_im:
            mismatch.append(f"done.images_used={done.get('images_used')} != {want_im}")
        eff["attention_used"] = done.get("attention_used")
        eff["block_swap_prefetch_used"] = done.get("block_swap_prefetch_used")
        if done.get("attention_used") != args.attention_backend:
            mismatch.append(f"done.attention_used={done.get('attention_used')} != {args.attention_backend}")
        if done.get("block_swap_prefetch_used") != "on":
            mismatch.append(f"done.block_swap_prefetch_used={done.get('block_swap_prefetch_used')} != on")
        eff["vram16_mode_used"] = done.get("vram16_mode_used")
        want_v = "on" if args.vram16 else "off"
        if done.get("vram16_mode_used") != want_v:
            mismatch.append(f"done.vram16_mode_used={done.get('vram16_mode_used')} != {want_v}")
    if s1 and "swap" in s1:
        sw = s1["swap"]
        eff["arenas"] = sw.get("arenas")
        eff["swap_blocks_on_gpu"] = sw.get("blocks_on_gpu")
        n = args.blocks_on_gpu
        v16 = (rec.get("vram16") or {}).get("record") or {}
        if args.vram16 and v16.get("blocks_on_gpu") is not None:
            n = v16["blocks_on_gpu"]          # on: blocks_on_gpu_for の結果が効く
            eff["vram16_blocks_on_gpu"] = n
        nblocks = sw.get("blocks")
        if nblocks and 0 < n < nblocks:
            expected = min(nblocks, n + 2)
            eff["arenas_expected"] = expected
            if sw.get("arenas") != expected:
                mismatch.append(f"arenas={sw.get('arenas')} != expected {expected}")
        if sw.get("blocks_on_gpu") is not None and sw.get("blocks_on_gpu") != n:
            mismatch.append(f"swap_service.blocks_on_gpu={sw.get('blocks_on_gpu')} != {n}")
    frac = rec.get("per_process_memory_fraction")
    eff["per_process_memory_fraction"] = frac
    want = 1.0 if args.memory_fraction is None else args.memory_fraction
    if isinstance(frac, float) and abs(frac - want) > 1e-6:
        mismatch.append(f"per_process_memory_fraction={frac} != {want}")
    rec["effective"] = eff
    if mismatch:
        rec["effective_mismatch"] = mismatch
    rec["script_seconds"] = round(time.perf_counter() - t_script, 2)
    rec["finished"] = now_iso()
    write_json(run_json, rec)
    say(f"finished: error={rec['error']['type'] if rec['error'] else None} mismatch={mismatch or None} "
        f"step2={rec.get('timing', {}).get('step2_seconds')}s -> {run_json}")
    return 2 if mismatch else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = ((Path(args.runs_dir) if args.runs_dir else RUNS) / args.label).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.path.insert(0, str(ROOT))
    fh = install_tee(out_dir)
    try:
        say(f"label={args.label} out_dir={out_dir} dry_run={args.dry_run}")
        return dry_run(args, out_dir) if args.dry_run else run(args, out_dir)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
