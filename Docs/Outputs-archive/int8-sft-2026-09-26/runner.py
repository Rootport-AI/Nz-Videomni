#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""§3-168 実機検証の MCP 実機ランナー。標準ライブラリ＋mcp パッケージのみ。

`stage3_25\stage3_runner25.py`（§3-167 B-2・REST 直叩き版）の構造を手本にし、
**REST 直叩きを MCP ツール呼び出しに置き換えた**もの。副命令の構成・結果 JSON
の形・VRAM／コミットの計測（1 秒間隔）・`psnr-ssim`（同梱 ffmpeg）はほぼその
まま踏襲している。

MCP の呼び方は確立済みの経路（`Docs/VERIFICATION_LOG.md` §39.5・§39.6）と同じ:
リポジトリの `.venv\Scripts\python.exe -m mcp_server` を stdio トランスポート
の子プロセスとして起動し、`mcp` パッケージのクライアントで
`initialize → tools/list → tools/call` する。1 コマンドごとにセッションを
張って閉じる（`Docs/Outputs-archive/start-end-bridge-2026-09-07/mcp_call.py`
と同じパターン）。

このツール作成担当からは GPU を使う副命令（load / single / iclora / chain /
keep2 / bad-header の load 部分 / restore）を一度も実行していない
（作成者への指示で禁止されている）。動作確認は `status` と `models` のみ
（`record` は job_status 等の読み取りだけで GPU を使わないため、既存ジョブの
記録復元に限り実行している）。

**注意（single はベースモデル／transformerを切り替えない）**: `single` /
`iclora` / `chain` は現在ロード済みのパイプラインへそのまま投げるだけで、
`base_model` / `models.transformer` を自動では切り替えない（切替に使うのは
`load` だけ）。切り替え忘れたまま `single` を呼ぶと、狙ったのと違う
transformer で静かに完走してしまう（実例: `load --base-model LTX25` を
省略したまま `single --base-model LTX25 ...` を呼び、実際には直前に
ロードされていた LTX23 の fp8 で生成された。`results/invalid_*.json` 参照）。
**必ず `single`/`iclora`/`chain` の前に `load --base-model X --transformer N`
を呼ぶこと。** `--base-model` 引数は記録用のラベルに過ぎず、実際にどの
パイプラインで生成されるかは `load` 済みの状態で決まる。

リポジトリには何も書かない（例外: bad-header が --weights-dir に置く数 KB の
偽ファイル。--cleanup で消す）。結果は本ファイルと同じ場所の
results/<tag>.json（＋<tag>.samples.jsonl）へ追記保存する。

副命令:
  status                                    backend_status・list_models・list_jobs
  models --base-model X [--transformer N ...]  名前の登録確認
  load --base-model X --transformer N       load_pipeline→応答とlist_modelsの両方で確認
  single --base-model X --transformer N --tag T [...]   submit_generate→wait→動画+meta
  iclora --base-model X --transformer N --tag T --reference PATH --ic-lora NAME
  chain --base-model X --transformer N --tag T          submit_chain 2クリップ
  keep2 --base-model X --transformer N --tag T [--reuse]
                                             keep_resident 2本→映像/音声ストリームMD5一致
                                             ＋PSNR∞（ファイル全体SHA256は参考値）。
                                             --reuse で生成をやり直さず既存の
                                             <tag>_r1/_r2 から再判定のみ（GPU不使用）
  bad-header --base-model X --weights-dir PATH [--case NAME] [--cleanup]
  psnr-ssim --a X --b Y                     同梱 ffmpeg の psnr/ssim フィルタ
  save --job-id ID --dest PATH              save_job_video
  record --job-id ID --tag T [--kind K] [--from-outputs]
                                             job_status等（既定）または
                                             outputs/<job-id>/直読み（--from-outputs、
                                             バックエンド停止中でも使える）から
                                             結果を事後復元（どちらもGPU不使用）
  restore --base-model X --transformer N    load と同じ（名前だけ変える）
  report                                    results/*.json（invalid_*/_* を除く） → RESULTS.md
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPORT_MD = HERE / "RESULTS.md"
ROOT = Path(r"S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni")
PY = ROOT / ".venv" / "Scripts" / "python.exe"

PROMPT = ("a cinematic tracking shot of a red sports car driving along a "
          "coastal road at sunset")  # 前回ランナー（stage3/stage3_25）と同じ文（比較のため）


# === 小物 ==================================================================
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def say(msg: str) -> None:
    print("[%s] %s" % (now(), msg), flush=True)


def save(tag: str, data: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / ("%s.json" % tag)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    say("saved %s" % path)
    return path


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def gib(b):
    return None if b is None else round(b / 1024 ** 3, 2)


def cell(v):
    return "—" if v is None or v == "" else str(v)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# === MCP クライアント（stdio、1コマンド1セッション） =========================
class RunnerToolError(Exception):
    """MCP ツールが isError:true を返した（サーバー側の ToolError の翻訳文言）。"""


def server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=str(PY),
        args=["-m", "mcp_server"],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUTF8": "1"},
    )


class Mcp:
    def __init__(self, session: ClientSession) -> None:
        self.session = session

    async def call(self, name: str, args: dict | None = None) -> dict:
        res = await self.session.call_tool(name, args or {})
        if res.isError:
            parts = [getattr(c, "text", None) or str(c) for c in (res.content or [])]
            raise RunnerToolError("; ".join(p for p in parts if p) or ("%s failed" % name))
        sc = res.structuredContent
        if sc is not None:
            return sc
        for c in res.content or []:
            text = getattr(c, "text", None)
            if text:
                try:
                    return json.loads(text)
                except Exception:
                    return {"_text": text}
        return {}


@asynccontextmanager
async def mcp_session():
    async with stdio_client(server_params()) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            yield Mcp(s)


def run_async(coro_fn):
    """``async def run(mcp): ...`` を1セッションの中で実行するヘルパー。

    ``stdio_client`` / ``ClientSession`` は anyio の TaskGroup 上に乗っている
    ため、内側で送出した ``RunnerToolError`` はそのままでは
    ``BaseExceptionGroup`` に包まれて出てくる（Python 3.11+ の except* 対象）。
    呼び出し側は素の ``except RunnerToolError`` だけを書けば済むよう、ここで
    1段だけ剥がす。
    """

    async def _runner():
        async with mcp_session() as mcp:
            return await coro_fn(mcp)

    try:
        return asyncio.run(_runner())
    except* RunnerToolError as eg:
        # ``eg.exceptions[0]`` can itself still be a nested BaseExceptionGroup
        # (anyio's TaskGroup wraps once per nested async-with), so unwrap all
        # the way down to the leaf RunnerToolError before re-raising it.
        leaf = eg.exceptions[0]
        while isinstance(leaf, BaseExceptionGroup):
            leaf = leaf.exceptions[0]
        raise leaf from None


# === 計測（VRAM 専有・コミット・参考で WDDM 共有） ============================
# stage3 / stage3_25 と同一の方式（Get-Counter 1秒間隔・nvidia-smi 1秒間隔）。
_PS_SAMPLER = r"""
$c = @('\Memory\Committed Bytes','\Memory\Commit Limit','\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage')
while ($true) {
  $s = Get-Counter -Counter $c -ErrorAction SilentlyContinue
  $o = [ordered]@{ t = [DateTimeOffset]::Now.ToUnixTimeMilliseconds() / 1000.0; commit = $null; limit = $null; ded = @{}; sh = @{} }
  foreach ($x in $s.CounterSamples) {
    $p = $x.Path.ToLower()
    if ($p.EndsWith('\committed bytes')) { $o.commit = $x.CookedValue }
    elseif ($p.EndsWith('\commit limit')) { $o.limit = $x.CookedValue }
    elseif ($p -match '\(([^)]+)\)\\dedicated usage$') { $o.ded[$Matches[1]] = $x.CookedValue }
    elseif ($p -match '\(([^)]+)\)\\shared usage$') { $o.sh[$Matches[1]] = $x.CookedValue }
  }
  [Console]::Out.WriteLine(($o | ConvertTo-Json -Compress -Depth 4))
  [Console]::Out.Flush()
  Start-Sleep -Milliseconds 800
}
"""


class Sampler:
    """nvidia-smi（専有 VRAM・MiB）と PowerShell（コミット・参考の WDDM 共有）を並走。

    stage3 / stage3_25 の Sampler と無変更（MCP 化はトランスポートだけの話で、
    計測は subprocess ベースのため独立している）。
    """

    def __init__(self):
        self.vram = []    # (t, MiB)
        self.commit = []  # (t, bytes, limit)
        self.shared = []  # (t, bytes)
        self.procs = []
        self.threads = []

    def _reader(self, proc, handler):
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                handler(time.time(), line)
            except Exception:
                pass

    def _on_smi(self, t, line):
        self.vram.append((t, float(line.split(",")[0])))

    def _on_ps(self, t, line):
        o = json.loads(line)
        if o.get("commit") is not None:
            self.commit.append((t, float(o["commit"]), o.get("limit")))
        ded = o.get("ded") or {}
        if ded:
            main = max(ded, key=lambda k: ded[k])
            sh = (o.get("sh") or {}).get(main)
            if sh is not None:
                self.shared.append((t, float(sh)))

    def start(self):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        smi = subprocess.Popen(
            ["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-l", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, creationflags=flags)
        enc = base64.b64encode(_PS_SAMPLER.encode("utf-16-le")).decode("ascii")
        ps = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", enc],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, creationflags=flags)
        for proc, handler in ((smi, self._on_smi), (ps, self._on_ps)):
            th = threading.Thread(target=self._reader, args=(proc, handler), daemon=True)
            th.start()
            self.procs.append(proc)
            self.threads.append(th)
        deadline = time.time() + 15
        while time.time() < deadline and not (self.vram and self.commit):
            time.sleep(0.2)

    def stop(self):
        time.sleep(1.5)
        for proc in self.procs:
            try:
                proc.kill()
            except Exception:
                pass
        for th in self.threads:
            th.join(timeout=3)

    def summary(self, t0=None, t1=None) -> dict:
        def pick(rows):
            return [r for r in rows if (t0 is None or r[0] >= t0) and (t1 is None or r[0] <= t1)]

        vr, cm, sh = pick(self.vram), pick(self.commit), pick(self.shared)
        vv = [r[1] for r in vr]
        cv = [r[1] for r in cm]
        sv = [r[1] for r in sh]
        return {
            "samples": {"vram": len(vv), "commit": len(cv), "shared": len(sv)},
            "vram_mib_before": vv[0] if vv else None,
            "vram_mib_max": max(vv) if vv else None,
            "vram_mib_median": statistics.median(vv) if vv else None,
            "vram_mib_end": vv[-1] if vv else None,
            "commit_gib_before": gib(cv[0]) if cv else None,
            "commit_gib_max": gib(max(cv)) if cv else None,
            "commit_gib_median": gib(statistics.median(cv)) if cv else None,
            "commit_gib_end": gib(cv[-1]) if cv else None,
            "commit_gib_delta_max": gib(max(cv) - cv[0]) if cv else None,
            "commit_gib_delta_end": gib(cv[-1] - cv[0]) if cv else None,
            "commit_limit_gib": gib(cm[-1][2]) if cm and cm[-1][2] else None,
            "shared_gib_max_wddm": gib(max(sv)) if sv else None,
        }

    def write_raw(self, tag: str):
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / ("%s.samples.jsonl" % tag)
        with path.open("w", encoding="utf-8") as fh:
            for t, v in self.vram:
                fh.write(json.dumps({"t": t, "vram_mib": v}) + "\n")
            for t, c, lim in self.commit:
                fh.write(json.dumps({"t": t, "commit": c, "limit": lim}) + "\n")
            for t, s in self.shared:
                fh.write(json.dumps({"t": t, "shared": s}) + "\n")


def print_meas(m: dict):
    say("  VRAM(nvidia-smi 専有 MiB): 開始前=%s 最大=%s 中央値=%s 終了=%s"
        % (m["vram_mib_before"], m["vram_mib_max"], m["vram_mib_median"], m["vram_mib_end"]))
    say("  コミット(GiB): 開始前=%s 最大=%s 中央値=%s 終了=%s 増分最大=%s 上限=%s"
        % (m["commit_gib_before"], m["commit_gib_max"], m["commit_gib_median"],
           m["commit_gib_end"], m["commit_gib_delta_max"], m["commit_limit_gib"]))
    say("  参考: WDDM 共有 GPU メモリ最大=%s GiB（サンプル数 %s）"
        % (m["shared_gib_max_wddm"], m["samples"]))


# === models 応答の読み解き（REST版のbase_block/transformer_entriesと同形） ===
def base_block(models_body: dict, base: str) -> dict:
    for bm in (models_body or {}).get("base_models") or []:
        if bm.get("id") == base:
            return bm
    return {}


def transformer_entries(models_body: dict, base: str) -> list:
    bm = base_block(models_body, base)
    return ((bm.get("categories") or {}).get("transformer") or {}).get("entries") or []


def active_transformer_for(models_body: dict, base: str):
    bm = base_block(models_body, base)
    return ((bm.get("categories") or {}).get("transformer") or {}).get("active")


# === status =================================================================
async def _status(mcp: Mcp) -> dict:
    st = await mcp.call("backend_status")
    models: dict | None = None
    jobs: dict | None = None
    if st.get("reachable"):
        # backend_status never raises, but list_models/list_jobs can (e.g. the
        # backend answers /status but then drops the connection) -- keep the
        # status report itself crash-proof.
        try:
            models = await mcp.call("list_models")
        except RunnerToolError as exc:
            models = {"_error": str(exc)}
        try:
            jobs = await mcp.call("list_jobs")
        except RunnerToolError as exc:
            jobs = {"_error": str(exc)}
    return {"backend_status": st, "models": models, "jobs": jobs}


def cmd_status(args) -> int:
    rec = run_async(_status)
    st, models, jobs = rec["backend_status"], rec["models"], rec["jobs"]
    say("backend_status: reachable=%s base_url=%s api_key_configured=%s pipeline_load_in_flight=%s"
        % (st.get("reachable"), st.get("base_url"), st.get("api_key_configured"),
           st.get("pipeline_load_in_flight")))
    if not st.get("reachable"):
        say("  error=%s" % st.get("error"))
        say("バックエンド未稼働のため list_models / list_jobs はスキップしました（run.bat で起動してください）")
        save("status", {"at": now(), "kind": "status", **rec})
        return 1
    s = st.get("status") or {}
    say("  status.state=%s status.base_model=%s" % (s.get("state"), s.get("base_model")))
    if models and not models.get("_error"):
        say("active_base_model=%s" % models.get("active_base_model"))
        for bm in models.get("base_models") or []:
            cat = (bm.get("categories") or {}).get("transformer") or {}
            say("  base_model id=%-8s active=%-5s installed=%-5s missing=%s entries=%d active_transformer=%s"
                % (bm.get("id"), bm.get("active"), bm.get("installed"), bm.get("missing_categories"),
                   len(cat.get("entries") or []), cat.get("active")))
    else:
        say("list_models 失敗: %s" % (models or {}).get("_error"))
    if jobs and not jobs.get("_error"):
        counts = jobs.get("counts") or {}
        running = [j for j in jobs.get("jobs") or [] if j.get("status") in ("queued", "running")]
        say("list_jobs counts=%s running/queued=%d" % (counts, len(running)))
        for j in running:
            say("  running job_id=%s status=%s progress=%s" % (j.get("job_id"), j.get("status"), j.get("progress")))
    else:
        say("list_jobs 失敗: %s" % (jobs or {}).get("_error"))
    save("status", {"at": now(), "kind": "status", **rec})
    return 0


# === models ==================================================================
def cmd_models(args) -> int:
    async def go(mcp: Mcp) -> dict:
        return await mcp.call("list_models")

    try:
        body = run_async(go)
    except RunnerToolError as exc:
        say("ToolError: %s" % exc)
        return 1
    entries = transformer_entries(body, args.base_model)
    active = active_transformer_for(body, args.base_model)
    say("base_model=%s active_transformer=%s entries=%d" % (args.base_model, active, len(entries)))
    for e in entries:
        say("  entry name=%-55s exists=%s source=%s path=%s"
            % (e.get("name"), e.get("exists"), e.get("source"), e.get("path")))
    wanted = args.transformer or []
    found_all = {}
    for want in wanted:
        hit = [e for e in entries if e.get("name") == want]
        found_all[want] = hit[0] if hit else None
        if hit:
            say("登録あり: %s exists=%s path=%s" % (want, hit[0].get("exists"), hit[0].get("path")))
        else:
            say("未登録: %s は %s の transformer に出ていません" % (want, args.base_model))
    save(args.tag or "models", {"at": now(), "kind": "models", "base_model": args.base_model,
                                "wanted": wanted, "found": found_all, "active": active,
                                "transformer_entries": entries})
    return 0 if all(found_all.values()) else (0 if not wanted else 1)


# === load / restore ==========================================================
async def _do_load(mcp: Mcp, transformer: str, base_model: str, tag: str) -> dict:
    body_sent = {"models": {"transformer": transformer}, "base_model": base_model}
    sampler = Sampler()
    sampler.start()
    t0 = time.time()
    say("load_pipeline %s" % json.dumps(body_sent, ensure_ascii=False))
    resp = await mcp.call("load_pipeline", {
        "models": {"transformer": transformer}, "base_model": base_model, "wait_sec": 45,
    })
    say("  1回目の応答: %s" % dump(resp))
    deadline = t0 + 1800
    while not resp.get("finished", True) and time.time() < deadline:
        time.sleep(2)
        st = await mcp.call("backend_status")
        state = ((st.get("status") or {}).get("state"))
        say("  ... backend_status.state=%s" % state)
        if state in ("ready", "error"):
            resp["finished"] = True
            resp["polled_state"] = state
            break
    t1 = time.time()
    sampler.stop()
    sampler.write_raw(tag)
    meas = sampler.summary()

    models_after = await mcp.call("list_models")
    published = active_transformer_for(models_after, base_model)
    record = {
        "at": now(), "tag": tag, "kind": "load", "request": body_sent,
        "response": resp, "seconds": round(t1 - t0, 1),
        "models_active_after": published, "measure": meas,
    }
    say("-> finished=%s models_active_after=%s（要求=%s） 所要 %.1f 秒"
        % (resp.get("finished"), published, transformer, t1 - t0))
    print_meas(meas)
    return record


def cmd_load(args) -> int:
    tag = args.tag or ("load_%s" % args.transformer)
    rec = run_async(lambda mcp: _do_load(mcp, args.transformer, args.base_model, tag))
    save(tag, rec)
    ok = rec.get("response", {}).get("finished") and rec.get("models_active_after") == args.transformer
    if not ok:
        say("ERROR: 選択が確認できません（応答未完了、または list_models の active が要求と不一致）")
        return 1
    return 0


def cmd_restore(args) -> int:
    """load と同じ（名前だけ変える）。"""
    tag = args.tag or "restore"
    rec = run_async(lambda mcp: _do_load(mcp, args.transformer, args.base_model, tag))
    save(tag, rec)
    ok = rec.get("response", {}).get("finished") and rec.get("models_active_after") == args.transformer
    return 0 if ok else 1


# === single / iclora / chain（共通: submit→wait→path→mp4info） ===============
def parse_lora(text: str) -> dict:
    name, _, strength = text.partition(":")
    return {"name": name, "strength": float(strength) if strength else 1.0}


#: wait_for_job 呼び出し自体が失敗したとき、再試行してよい一時的な失敗の目印。
#: 実例(2026-09-27): LoRA読み込み中にサーバーが一時的に応答せず
#: BACKEND_UNREACHABLEになったが、ジョブ自体は82秒後にcompletedしていた
#: （job 4f04197d-e67e-40a6-a4af-16565f89db93）。即座に失敗にせず、
#: 5秒待って再試行する（連続6回＝約30秒まで。それでも駄目なら諦める）。
_TRANSIENT_WAIT_MARKERS = ("BACKEND_UNREACHABLE", "ReadTimeout", "TimeoutError", "timed out")
_TRANSIENT_WAIT_MAX_RETRIES = 6
_TRANSIENT_WAIT_RETRY_SEC = 5.0


def _is_transient_wait_error(text: str) -> bool:
    low = text.lower()
    return any(marker.lower() in low for marker in _TRANSIENT_WAIT_MARKERS)


async def _wait_job(mcp: Mcp, job_id: str, overall_timeout: float = 7200) -> dict:
    """ジョブが終端状態になるまで ``wait_for_job`` を呼び続ける。

    ``wait_for_job`` 自体の呼び出しが ``RunnerToolError``（BACKEND_UNREACHABLE
    やタイムアウト系）になっても即座には失敗にせず、5秒待って再試行する
    （``_TRANSIENT_WAIT_MAX_RETRIES`` 回＝連続約30秒まで）。1回でも成功したら
    連続失敗カウントはリセットする。それでも連続で上限に達したら例外を
    そのまま再送出する（サーバーが本当に落ちている場合まで無限に粘らない）。
    """
    t0 = time.time()
    last_note = 0.0
    transient_streak = 0
    while True:
        try:
            status = await mcp.call(
                "wait_for_job", {"job_id": job_id, "timeout_sec": 45, "poll_interval_sec": 5})
        except RunnerToolError as exc:
            text = str(exc)
            if _is_transient_wait_error(text) and transient_streak < _TRANSIENT_WAIT_MAX_RETRIES:
                transient_streak += 1
                say("  ... wait_for_job が一時的に失敗（%d/%d回目、%.0f秒後に再試行）: %s"
                    % (transient_streak, _TRANSIENT_WAIT_MAX_RETRIES, _TRANSIENT_WAIT_RETRY_SEC, text))
                await asyncio.sleep(_TRANSIENT_WAIT_RETRY_SEC)
                continue
            raise
        transient_streak = 0
        if status.get("status") in ("completed", "failed", "cancelled"):
            return status
        if time.time() - last_note > 30:
            last_note = time.time()
            say("  ... %s progress=%s stage=%s timed_out=%s"
                % (status.get("status"), status.get("progress"), status.get("stage"), status.get("timed_out")))
        if time.time() - t0 > overall_timeout:
            status["status"] = status.get("status") or "timeout"
            return status


async def _fetch_result(mcp: Mcp, job_id: str, which: str = "output") -> dict:
    """job_id の動画パス・存在確認・埋め込みメタデータ（get_mp4_info）をまとめて取る。"""
    if which == "joined":
        vp = await mcp.call("get_joined_video_path", {"job_id": job_id})
    else:
        vp = await mcp.call("get_job_video_path", {"job_id": job_id})
    meta = {}
    if vp.get("exists"):
        info = await mcp.call("get_mp4_info", {"path": vp["path"]})
        comment = info.get("comment")
        if comment:
            try:
                meta = json.loads(comment)
            except Exception as exc:  # noqa: BLE001
                meta = {"_parse_error": str(exc), "_raw": comment[:2000]}
    return {"video_path": vp, "meta": meta}


def _meta_summary(meta: dict) -> dict:
    sel = (((meta or {}).get("models") or {}).get("selection") or {}).get("transformer") or {}
    return {
        "transformer_name": sel.get("name"), "transformer_file": sel.get("file"),
        "peak_vram_mb": ((meta or {}).get("vram_optimization") or {}).get("peak_vram_mb"),
        "peak_vram_reserved_mb": (meta or {}).get("peak_vram_reserved_mb"),
        "generation_time_seconds": (meta or {}).get("generation_time_seconds"),
        "keep_resident_used": (meta or {}).get("keep_resident_used"),
        "fused_gguf_dequant_kernel_used": (meta or {}).get("fused_gguf_dequant_kernel_used"),
        "ic_lora": (meta or {}).get("ic_lora"),
        "request_loras": ((meta or {}).get("request") or {}).get("loras"),
        "output": ((meta or {}).get("output") or {}).get("path"),
    }


async def _submit_and_wait(mcp: Mcp, tool: str, payload: dict, tag: str, kind: str) -> dict:
    sampler = Sampler()
    sampler.start()
    t0 = time.time()
    say("%s %s" % (tool, json.dumps(payload, ensure_ascii=False)))
    submit = await mcp.call(tool, payload)
    job_id = submit.get("job_id")
    if not job_id:
        sampler.stop()
        rec = {"at": now(), "tag": tag, "kind": kind, "request": payload, "submit_response": submit,
               "error": "no job_id in submit response"}
        save(tag, rec)
        return rec
    say("  job_id=%s" % job_id)
    status = await _wait_job(mcp, job_id)
    t1 = time.time()
    sampler.stop()
    sampler.write_raw(tag)
    meas = sampler.summary(t0, t1 + 2)

    fetched = {}
    if status.get("status") == "completed":
        fetched = await _fetch_result(mcp, job_id, "output")
    m = _meta_summary(fetched.get("meta") or {})

    rec = {
        "at": now(), "tag": tag, "kind": kind, "request": payload, "submit_response": submit,
        "job_id": job_id, "status": status.get("status"), "error": status.get("error"),
        "wall_seconds": round(t1 - t0, 1), "waited_sec": status.get("waited_sec"),
        "video_path": (fetched.get("video_path") or {}).get("path"),
        "video_exists": (fetched.get("video_path") or {}).get("exists"),
        "meta": m, "measure": meas,
    }
    save(tag, rec)
    say("-> status=%s 経過 %.1f 秒 job_id=%s" % (status.get("status"), t1 - t0, job_id))
    if status.get("status") != "completed":
        say("失敗（本文全文）\n%s" % dump(status))
    say("  metadata: transformer.file=%s peak_vram_mb=%s generation_time_seconds=%s "
        "keep_resident_used=%s fused_gguf_dequant_kernel_used=%s"
        % (m["transformer_file"], m["peak_vram_mb"], m["generation_time_seconds"],
           m["keep_resident_used"], m["fused_gguf_dequant_kernel_used"]))
    say("  LoRA: ic_lora=%s request_loras=%s" % (dump(m["ic_lora"]), dump(m["request_loras"])))
    say("  出力: %s（exists=%s）" % (rec["video_path"], rec["video_exists"]))
    print_meas(meas)
    return rec


def _single_payload(args) -> dict:
    payload = {
        "prompt": args.prompt, "width": args.width, "height": args.height,
        "num_frames": args.frames, "frame_rate": 24.0, "seed": args.seed,
    }
    if args.lora:
        payload["loras"] = [parse_lora(x) for x in args.lora]
    if args.keep_resident:
        payload["keep_resident"] = True
    # --prefetch: submit_generate の block_swap_prefetch をそのまま渡す
    # （既定 on＝先読み。off で同期スワップ経路になる。G3「先読みオフの
    # single 1本」用。attention_backend/sage も加速設定として存在するが
    # このフラグでは扱わない -- 今のところ先読みだけが要る）。
    if getattr(args, "prefetch", None) is not None:
        payload["block_swap_prefetch"] = (args.prefetch == "on")
    return payload


def cmd_single(args) -> int:
    payload = _single_payload(args)

    async def go(mcp: Mcp) -> dict:
        if args.reference_video:
            if args.width % 128 or args.height % 128:
                raise SystemExit(
                    "ERROR: 参照動画つきは幅・高さが128の倍数でないと422です（例: --width 512 --height 384）")
            src = Path(args.reference_video)
            if not src.is_absolute():
                src = ROOT / src
            say("upload_video %s" % src)
            up = await mcp.call("upload_video", {"file_path": str(src)})
            payload["reference_video_id"] = up["video_id"]
            say("  video_id=%s" % up["video_id"])
        return await _submit_and_wait(mcp, "submit_generate", payload, args.tag, "single")

    try:
        rec = run_async(go)
    except SystemExit as exc:
        say(str(exc))
        return 2
    return 0 if rec.get("status") == "completed" else 1


def cmd_iclora(args) -> int:
    payload = {
        "prompt": args.prompt, "width": args.width, "height": args.height,
        "num_frames": args.frames, "frame_rate": 24.0, "seed": args.seed,
        "loras": [{"name": args.ic_lora, "strength": args.strength}],
    }

    async def go(mcp: Mcp) -> dict:
        src = Path(args.reference)
        if not src.is_absolute():
            src = ROOT / src
        say("upload_video %s" % src)
        up = await mcp.call("upload_video", {"file_path": str(src)})
        payload["reference_video_id"] = up["video_id"]
        say("  video_id=%s frame_count=%s fps=%s" % (up["video_id"], up.get("frame_count"), up.get("fps")))
        return await _submit_and_wait(mcp, "submit_generate", payload, args.tag, "iclora")

    if args.width % 128 or args.height % 128:
        say("ERROR: IC-LoRA（参照動画あり）は幅・高さが128の倍数でないと422です（例: --width 512 --height 384）")
        return 2
    rec = run_async(go)
    return 0 if rec.get("status") == "completed" else 1


def cmd_chain(args) -> int:
    clips = [{"num_frames": n} for n in args.clip_frames]
    payload = {
        "prompt": args.prompt, "clips": clips,
        "width": args.width, "height": args.height, "frame_rate": 24.0, "seed": args.seed,
    }
    if args.lora:
        payload["loras"] = [parse_lora(x) for x in args.lora]

    rec = run_async(lambda mcp: _submit_and_wait(mcp, "submit_chain", payload, args.tag, "chain"))
    return 0 if rec.get("status") == "completed" else 1


# === keep2（keep_resident 2本→映像/音声ストリームMD5一致＋PSNR∞） ============
# 2026-09-27 実機発見: keep_resident 2本のファイル全体 SHA-256 は、映像・
# 音声そのものが完全一致していても、mp4 コンテナのメタデータ（§3-164 で
# 埋め込む job_id 等が2本で異なる）のせいで一致しない。合否は
# 「映像ストリームMD5一致 かつ 音声ストリームMD5一致（同梱ffmpegの
# `-map 0:v:0 -c copy -f md5 -`／`-map 0:a:0 ...`）かつ PSNR∞」に変更し、
# ファイル全体のSHA-256は参考値として記録するだけにする（合否には使わない）。
def _load_existing_job_rec(tag: str) -> dict | None:
    p = RESULTS / ("%s.json" % tag)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def cmd_keep2(args) -> int:
    if args.reuse:
        r1, r2 = _load_existing_job_rec("%s_r1" % args.tag), _load_existing_job_rec("%s_r2" % args.tag)
        if r1 is None or r2 is None:
            say("ERROR: --reuse には既存の results/%s_r1.json と results/%s_r2.json の両方が必要です"
                % (args.tag, args.tag))
            return 2
        say("--reuse: 既存の記録から再判定します（%s_r1.json / %s_r2.json、生成のやり直しはしません）"
            % (args.tag, args.tag))
    else:
        base_payload = {
            "prompt": args.prompt, "width": args.width, "height": args.height,
            "num_frames": args.frames, "frame_rate": 24.0, "seed": args.seed, "keep_resident": True,
        }

        async def go(mcp: Mcp) -> tuple:
            r1 = await _submit_and_wait(mcp, "submit_generate", dict(base_payload), "%s_r1" % args.tag, "keep2")
            r2 = await _submit_and_wait(mcp, "submit_generate", dict(base_payload), "%s_r2" % args.tag, "keep2")
            return r1, r2

        r1, r2 = run_async(go)

    if r1.get("status") != "completed" or r2.get("status") != "completed":
        say("ERROR: keep2 の2本のうち少なくとも1本が completed ではありません")
        save(args.tag, {"at": now(), "kind": "keep2", "r1": r1, "r2": r2, "match": None})
        return 1

    p1, p2 = Path(r1["video_path"]), Path(r2["video_path"])

    # 参考値（コンテナメタデータの違いで不一致になり得る。合否には使わない）。
    h1, h2 = sha256_file(p1), sha256_file(p2)
    sha256_match = h1 == h2
    say("参考: ファイル全体SHA256: r1=%s r2=%s match=%s（コンテナメタデータの違いで不一致もあり得る）"
        % (h1, h2, sha256_match))

    v1, v2 = _stream_md5(p1, "0:v:0"), _stream_md5(p2, "0:v:0")
    a1, a2 = _stream_md5(p1, "0:a:0"), _stream_md5(p2, "0:a:0")
    video_match = v1 is not None and v1 == v2
    audio_match = a1 == a2  # 両方 None（音声トラック無し）も一致扱い
    say("映像ストリームMD5: r1=%s r2=%s match=%s" % (v1, v2, video_match))
    say("音声ストリームMD5: r1=%s r2=%s match=%s" % (a1, a2, audio_match))

    ps_rec = None
    try:
        ps_rec = _psnr_ssim(p1, p2, "%s_psnr_ssim" % args.tag)
    except Exception as exc:  # noqa: BLE001
        say("psnr-ssim 計測に失敗（続行）: %s" % exc)
    psnr_y = ((ps_rec or {}).get("psnr") or {}).get("y")
    psnr_inf = psnr_y is not None and math.isinf(psnr_y)
    say("PSNR y=%s（∞=%s）" % (psnr_y, psnr_inf))

    match = video_match and audio_match and psnr_inf
    rec = {
        "at": now(), "tag": args.tag, "kind": "keep2", "r1": r1, "r2": r2,
        "sha256_r1": h1, "sha256_r2": h2, "sha256_match": sha256_match,
        "video_md5_r1": v1, "video_md5_r2": v2, "video_md5_match": video_match,
        "audio_md5_r1": a1, "audio_md5_r2": a2, "audio_md5_match": audio_match,
        "psnr_y": psnr_y, "psnr_inf": psnr_inf, "psnr_ssim": ps_rec,
        "match": match, "reused": bool(args.reuse),
    }
    save(args.tag, rec)
    say("-> keep2 判定: %s" % ("OK" if match else "MISMATCH"))
    return 0 if match else 1


# === bad-header（ケース辞書は1箇所・追加しやすく） ============================
_BAD_HEADER_PREFIX = "model.diffusion_model."


def _zeros(itemsize: int, count: int) -> bytes:
    """中身を読まれないダミーテンソルの生バイト列（全ゼロ）。"""
    return b"\x00" * (itemsize * count)


def _placeholder_blocks(start: int = 1, end: int = 48) -> list[tuple[str, str, list[int], bytes]]:
    """block ``start``..``end-1`` を ``scale_shift_table`` の BF16[1] ダミーで
    埋める（transformer_blocks の個数（48個）だけを満たすための最小テンソル。
    本文は読まれない）。全ケース共通。
    """
    return [
        (_BAD_HEADER_PREFIX + "transformer_blocks.%d.scale_shift_table" % i, "BF16", [1], _zeros(2, 1))
        for i in range(start, end)
    ]


def _pack_header(tensors: list[tuple[str, str, list[int], bytes]], *, note: str) -> bytes:
    """(key, dtype, shape, raw_bytes) のリストから safetensors 1本ぶんの
    バイト列を組み立てる（ヘッダ＋データ領域。raw_bytes がそのままそのテン
    ソルのデータになる。長さは shape・dtype の itemsize と一致させること）。
    全ケース共通の配線（1箇所にまとめて重複を無くす）。
    """
    header = {"__metadata__": {"config": json.dumps({"transformer": {}}), "note": note}}
    off = 0
    blobs = []
    for key, dtype, shape, raw in tensors:
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    raw_header = json.dumps(header).encode("utf-8")
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    return struct.pack("<Q", len(raw_header)) + raw_header + b"".join(blobs)


def _case_fp8_per_row_scale() -> bytes:
    """前回（stage3 / stage3_25）と同一内容: per-row 倍率の fp8。必ず拒否される。

    ①__metadata__.config（transformer あり）②transformer_blocks が 0..47 揃う
    ③dtype（F8 は 2 次元 .weight か 1 次元 .bias・それ以外は BF16/F32）④fp8 の
    重みが1本以上 ⑤weight_scale が F32 のスカラー（shape () か (1,)）でなければ
    拒否 -- ①〜④を満たし、⑤の per-row [64] で落とす。
    """
    blk0 = _BAD_HEADER_PREFIX + "transformer_blocks.0.attn1.to_q"
    tensors = [
        (blk0 + ".weight", "F8_E4M3", [64, 64], _zeros(1, 64 * 64)),
        (blk0 + ".weight_scale", "F32", [64], _zeros(4, 64)),
    ] + _placeholder_blocks()
    return _pack_header(tensors, note="int8 runner fake header (per-row scale; must be refused)")


#: G7（§3-168・計画§4）向けに追加した5ケース。すべて計画§2.1の拒否条件表の
#: いずれかに当たる想定（製品側は §3-168 の worktree 実装＝将来のサーバー。
#: 現行サーバーでも「未対応の dtype/量子化」等の別理由で拒否されるはずだが、
#: 期待する拒否理由そのものは §3-168 実装後に確認すること）。
def _case_nvfp4_format() -> bytes:
    """量子化 format が受理集合の外（nvfp4）。計画§2.1「format が受理集合の
    外（nvfp4・mxfp8・convrot_w4a4）」で拒否される想定。"""
    blk0 = _BAD_HEADER_PREFIX + "transformer_blocks.0.attn1.to_q"
    marker = json.dumps({"format": "nvfp4"}).encode("utf-8")
    tensors = [
        (blk0 + ".weight", "I8", [64, 64], _zeros(1, 64 * 64)),
        (blk0 + ".comfy_quant", "U8", [len(marker)], marker),
    ] + _placeholder_blocks()
    return _pack_header(tensors, note="G7: nvfp4 marker (format outside accepted set)")


def _case_i8_no_marker() -> bytes:
    """I8 の重みに印（comfy_quant／メタデータ）が一切無い。計画§2.1「I8の重み
    に印が無い」で拒否される想定。"""
    blk0 = _BAD_HEADER_PREFIX + "transformer_blocks.0.attn1.to_q"
    tensors = [
        (blk0 + ".weight", "I8", [64, 64], _zeros(1, 64 * 64)),
        # .comfy_quant を意図的に置かない（メタデータ側にも層エントリを作らない）。
    ] + _placeholder_blocks()
    return _pack_header(tensors, note="G7: I8 weight with no comfy_quant marker and no metadata entry")


def _case_w4a8_no_codebook() -> bytes:
    """asym_w4a8_int8 の印はあるが weight_codebook が無い。計画§2.1「補助の
    不足・余り」で拒否される想定。in_features=256（packed [64,128]）・
    group_size=16（groups=16）。w4a8 は convrot 常時True。"""
    blk0 = _BAD_HEADER_PREFIX + "transformer_blocks.0.attn1.to_q"
    marker = json.dumps({"format": "asym_w4a8_int8"}).encode("utf-8")
    tensors = [
        (blk0 + ".weight", "I8", [64, 128], _zeros(1, 64 * 128)),
        (blk0 + ".weight_s_channel", "F32", [64], _zeros(4, 64)),
        (blk0 + ".weight_s_rel", "F8_E4M3", [64, 16], _zeros(1, 64 * 16)),
        # .weight_codebook を意図的に置かない。
        (blk0 + ".comfy_quant", "U8", [len(marker)], marker),
    ] + _placeholder_blocks()
    return _pack_header(tensors, note="G7: asym_w4a8_int8 marker missing weight_codebook")


def _case_quanto_data_placement() -> bytes:
    """optimum-quanto 風の '_data' 置き場所違反。量子化重みを正規の `.weight`
    ではなく `._data` に置く（quanto の内部命名の模倣）。計画§2.1「I8・U8の
    置き場所違反（optimum-quantoの_data等）」で拒否される想定。"""
    blk0 = _BAD_HEADER_PREFIX + "transformer_blocks.0.attn1.to_q"
    tensors = [
        (blk0 + "._data", "I8", [64, 64], _zeros(1, 64 * 64)),
        # 正規の .weight は置かない -- _data だけが I8 として存在する状態。
    ] + _placeholder_blocks()
    return _pack_header(tensors, note="G7: optimum-quanto style '_data' placement (not '.weight')")


def _case_weight_scale_wrong_shape() -> bytes:
    """int8_tensorwise の weight_scale が [o,2]（受理される ()/(1,)/(o,1) の
    どれでもない）。計画§2.1「倍率の形違い」で拒否される想定。"""
    blk0 = _BAD_HEADER_PREFIX + "transformer_blocks.0.attn1.to_q"
    marker = json.dumps({"format": "int8_tensorwise", "convrot": False}).encode("utf-8")
    tensors = [
        (blk0 + ".weight", "I8", [64, 64], _zeros(1, 64 * 64)),
        (blk0 + ".weight_scale", "F32", [64, 2], _zeros(4, 64 * 2)),
        (blk0 + ".comfy_quant", "U8", [len(marker)], marker),
    ] + _placeholder_blocks()
    return _pack_header(tensors, note="G7: weight_scale shape [o,2] (not ()/(1,)/(o,1))")


#: ケース名 -> バイト列を作る関数。後から追加しやすいよう1箇所にまとめる。
#: 計画§2.1の「params 入れ子の convrot（受理側）」は意図的にここへ含めていない
#: -- 拒否ではなく受理を確かめるケースなので bad-header／load_pipeline の
#: 422フローに乗らない（README「G7」節に理由と代替の確認手順を記載）。
BAD_HEADER_CASES = {
    "fp8_per_row_scale": _case_fp8_per_row_scale,
    "nvfp4_format": _case_nvfp4_format,
    "i8_no_marker": _case_i8_no_marker,
    "w4a8_no_codebook": _case_w4a8_no_codebook,
    "quanto_data_placement": _case_quanto_data_placement,
    "weight_scale_wrong_shape": _case_weight_scale_wrong_shape,
}


def cmd_bad_header(args) -> int:
    if args.case not in BAD_HEADER_CASES:
        say("ERROR: 未知のケース %r（選べるのは %s）" % (args.case, sorted(BAD_HEADER_CASES)))
        return 2
    weights_dir = Path(args.weights_dir)
    if not weights_dir.is_absolute():
        weights_dir = ROOT / weights_dir
    target = weights_dir / ("zz_bad_%s.safetensors" % args.case)
    stem = target.stem

    if args.cleanup:
        if target.exists():
            target.unlink()
            say("削除しました: %s" % target)
        else:
            say("ありません: %s" % target)

        async def go_cleanup(mcp: Mcp) -> dict:
            return await mcp.call("list_models")

        body = run_async(go_cleanup)
        still = [e for e in transformer_entries(body, args.base_model) if stem in (e.get("name") or "")]
        say("list_models 上の残存: %s" % still)
        return 0

    weights_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        say("既に存在: %s（上書きします）" % target)
    target.write_bytes(BAD_HEADER_CASES[args.case]())
    say("作成: %s（%d バイト）" % (target, target.stat().st_size))

    async def go(mcp: Mcp) -> dict:
        before_body = await mcp.call("list_models")
        active_before = active_transformer_for(before_body, args.base_model)
        entries = transformer_entries(before_body, args.base_model)
        hit = [e for e in entries if e.get("name") in (stem, "%s__%s" % (target.parent.name, stem))]
        name = hit[0]["name"] if hit else None
        say("list_models 登録名: %s" % (hit[0] if hit else "未登録"))

        rec = {"at": now(), "tag": args.tag or ("bad_header_%s" % args.case), "kind": "bad_header",
               "case": args.case, "file": str(target), "size": target.stat().st_size,
               "registered": hit[0] if hit else None, "active_before": active_before}
        if name:
            say("load_pipeline %s" % json.dumps({"base_model": args.base_model, "models": {"transformer": name}},
                                                 ensure_ascii=False))
            try:
                resp = await mcp.call("load_pipeline", {"models": {"transformer": name},
                                                         "base_model": args.base_model})
                rec["load_response"] = resp
                rec["tool_error"] = None
            except RunnerToolError as exc:
                rec["load_response"] = None
                rec["tool_error"] = str(exc)
            after_body = await mcp.call("list_models")
            rec["active_after"] = active_transformer_for(after_body, args.base_model)
            say("  active transformer: 前=%s 後=%s（拒否なら変わらないはず）" % (active_before, rec["active_after"]))
            say("  ToolError: %s" % rec["tool_error"])
        return rec

    rec = run_async(go)
    save(rec["tag"], rec)
    say("後片付けは: bad-header --case %s --cleanup" % args.case)
    unchanged = rec.get("active_after") == rec.get("active_before")
    rejected = rec.get("tool_error") is not None
    return 0 if (rejected and unchanged) else 1


# === psnr-ssim（同梱 ffmpeg。stage3_25 と同じ手法） ===========================
_PSNR_RE = re.compile(r"PSNR\s+y:(?P<y>\S+)\s+u:(?P<u>\S+)\s+v:(?P<v>\S+)\s+average:(?P<avg>\S+)")
_SSIM_RE = re.compile(r"SSIM\s+Y:(?P<y>\S+).*?U:(?P<u>\S+).*?V:(?P<v>\S+).*?All:(?P<all>\S+)")


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    bundled = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if bundled.exists():
        return str(bundled)
    raise FileNotFoundError("ffmpeg が PATH にも %s にもありません" % bundled)


_STREAM_MD5_RE = re.compile(r"MD5=([0-9a-fA-F]+)")


def _stream_md5(path: Path, map_selector: str) -> str | None:
    """1本のストリーム（映像/音声）だけの MD5（同梱 ffmpeg の md5 muxer）。

    2026-09-27 の実機発見: keep_resident 2本のファイル全体 SHA-256 は、
    映像・音声そのものが完全一致していても、mp4 コンテナのメタデータ
    （§3-164 で埋め込む job_id 等）が2本で異なるため一致しない。
    ``-map <selector> -c copy -f md5 -`` で対象ストリームだけを再エンコード
    無しで取り出し、その中身だけの MD5 を取ればコンテナメタデータの影響を
    受けない（真の内容一致判定）。指定したストリームが無ければ None を返す
    （例: 音声トラックの無い動画への `0:a:0`）。
    """
    ffmpeg = _ffmpeg_bin()
    cmd = [ffmpeg, "-i", str(path), "-map", map_selector, "-c", "copy", "-f", "md5", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")
    m = _STREAM_MD5_RE.search(proc.stdout or "")
    return m.group(1) if m else None


def _resolve_video(token: str) -> Path:
    cand = Path(token)
    if not cand.is_absolute():
        cand = ROOT / token if (ROOT / token).exists() else cand
    if cand.exists():
        return cand
    rec_path = RESULTS / ("%s.json" % token)
    if not rec_path.exists():
        raise FileNotFoundError("mp4 でも results/%s.json でもありません: %s" % (token, token))
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    out = rec.get("video_path") or (rec.get("meta") or {}).get("output")
    if not out:
        raise FileNotFoundError("results/%s.json に video_path も meta.output もありません" % token)
    p = Path(out)
    return p if p.is_absolute() else ROOT / out


def _psnr_ssim(a: Path, b: Path, tag: str) -> dict:
    ffmpeg = _ffmpeg_bin()
    cmd = [ffmpeg, "-i", str(a), "-i", str(b), "-lavfi",
           "[0:v][1:v]ssim;[0:v][1:v]psnr", "-f", "null", "-"]
    say("実行: %s" % " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")
    text = proc.stdout or ""
    psnr_m, ssim_m = _PSNR_RE.search(text), _SSIM_RE.search(text)
    rec = {
        "at": now(), "tag": tag, "kind": "psnr_ssim", "a": str(a), "b": str(b),
        "returncode": proc.returncode,
        "psnr": {k: float(v) for k, v in psnr_m.groupdict().items()} if psnr_m else None,
        "ssim": {k: float(v) for k, v in ssim_m.groupdict().items()} if ssim_m else None,
        "raw_tail": text[-4000:],
    }
    save(tag, rec)
    return rec


def cmd_psnr_ssim(args) -> int:
    a, b = _resolve_video(args.a), _resolve_video(args.b)
    tag = args.tag or ("psnr_ssim_%s_vs_%s" % (Path(args.a).stem[:20], Path(args.b).stem[:20]))
    rec = _psnr_ssim(a, b, tag)
    if not rec["psnr"] or not rec["ssim"]:
        say("ffmpeg の出力から PSNR/SSIM を抽出できませんでした（raw_tail を results/%s.json に保存）" % tag)
        return 1
    say("  PSNR: y=%s average=%s / SSIM: Y=%s All=%s"
        % (rec["psnr"]["y"], rec["psnr"]["avg"], rec["ssim"]["y"], rec["ssim"]["all"]))
    return 0


# === save ====================================================================
def cmd_save(args) -> int:
    async def go(mcp: Mcp) -> dict:
        return await mcp.call("save_job_video", {
            "job_id": args.job_id, "dest_dir": args.dest,
            "filename": args.filename, "no_clobber": not args.overwrite,
            "which": args.which,
        })

    rec = run_async(go)
    say("saved_path=%s size_bytes=%s" % (rec.get("saved_path"), rec.get("size_bytes")))
    save(args.tag or ("save_%s" % args.job_id), {"at": now(), "kind": "save", "job_id": args.job_id, **rec})
    return 0


# === record（既存ジョブから results/<tag>.json を事後復元。GPU不使用） ========
def _parse_iso(ts: str):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _record_common(job_id: str, tag: str, kind: str, status_src: dict, meta: dict,
                   video_path, video_exists, recovered_label) -> dict:
    """MCP経由／--from-outputs のどちらでも使う共通の組み立て。

    ``status_src`` は status/error/created_at/completed_at/request を持つ辞書
    （MCP経路では ``job_status`` の応答、--from-outputs 経路では
    ``metadata.json`` 自身 -- どちらも同じキー名を持つので共通化できる）。
    ``meta`` は ``_meta_summary`` に渡す辞書（get_mp4_info の comment、または
    --from-outputs では同じ metadata.json）。``recovered_label`` は
    ``True``（MCP経由）か ``"from_outputs"``（ローカル直読み）。
    """
    m = _meta_summary(meta)
    wall_seconds = None
    created_at, completed_at = status_src.get("created_at"), status_src.get("completed_at")
    if created_at and completed_at:
        try:
            wall_seconds = round((_parse_iso(completed_at) - _parse_iso(created_at)).total_seconds(), 1)
        except Exception:  # noqa: BLE001
            wall_seconds = None

    return {
        "at": now(), "tag": tag, "kind": kind, "request": status_src.get("request"),
        "submit_response": {
            "job_id": job_id, "status": status_src.get("status"), "created_at": status_src.get("created_at"),
            "note": "record で事後復元（元の submit 応答そのものではない）",
        },
        "job_id": job_id, "status": status_src.get("status"), "error": status_src.get("error"),
        "wall_seconds": wall_seconds, "waited_sec": None,
        "video_path": str(video_path) if video_path is not None else None, "video_exists": video_exists,
        "meta": m, "measure": None, "samples": None, "recovered": recovered_label,
    }


async def _record_via_mcp(mcp: Mcp, job_id: str, tag: str, kind: str) -> dict:
    """``job_status``・``get_job_video_path``・``get_mp4_info`` だけを読み、
    ``submit_generate``/``submit_chain`` は一切呼ばない（GPUは使わない）。"""
    job = await mcp.call("job_status", {"job_id": job_id})
    fetched: dict = {}
    if job.get("status") == "completed":
        fetched = await _fetch_result(mcp, job_id, "output")
    vp = fetched.get("video_path") or {}
    return _record_common(job_id, tag, kind, job, fetched.get("meta") or {},
                          vp.get("path"), vp.get("exists"), True)


def _record_from_outputs(job_id: str, tag: str, kind: str) -> dict:
    """MCP・バックエンドを一切使わず、ローカルの
    ``outputs/<job_id>/{metadata.json,output.mp4}`` だけから直接読む
    （バックエンド停止中でも使える）。``get_mp4_info`` の ``comment`` が返す
    JSON は ``metadata.json`` と同一（README参照）なので、同じ辞書を
    status_src・meta の両方として使い回せる（metadata.json はジョブの状態
    ―status/created_at/completed_at/request― と生成条件のメタデータの両方を
    同じ1ファイルのトップレベルに持っている）。
    """
    job_dir = ROOT / "outputs" / job_id
    meta_path = job_dir / "metadata.json"
    video_path = job_dir / "output.mp4"
    if not meta_path.exists():
        raise FileNotFoundError("metadata.json がありません: %s" % meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return _record_common(job_id, tag, kind, meta, meta, video_path, video_path.exists(), "from_outputs")


def cmd_record(args) -> int:
    if args.from_outputs:
        rec = _record_from_outputs(args.job_id, args.tag, args.kind)
    else:
        rec = run_async(lambda mcp: _record_via_mcp(mcp, args.job_id, args.tag, args.kind))
    save(rec["tag"], rec)
    m = rec["meta"]
    say("復元(recovered=%s): status=%s transformer.file=%s loras=%s 生成秒=%s wall_seconds=%s"
        % (rec["recovered"], rec["status"], m.get("transformer_file"), dump(m.get("request_loras")),
           m.get("generation_time_seconds"), rec.get("wall_seconds")))
    say("  出力: %s（exists=%s）" % (rec["video_path"], rec["video_exists"]))
    return 0 if rec["status"] == "completed" else 1


# === report ==================================================================
def cmd_report(args) -> int:
    recs = []
    for path in sorted(RESULTS.glob("*.json")):
        # "_"始まりは内部用の一時記録（例: _last_fp8_single 的なもの）、
        # "invalid_"始まりは集計から外すべき無効な記録（例: 切替忘れで
        # 意図と違うtransformerで走ったジョブ。台帳側でファイル名を
        # invalid_<元tag>_<理由> に改名して除外する運用）。
        if path.name.startswith("_") or path.name.startswith("invalid_"):
            continue
        try:
            recs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass

    L = ["# §3-168 実機検証の結果（MCP 実機ランナー）", "",
         "作成 %s。VRAM は nvidia-smi の専有（MiB）、コミットは `\\Memory\\Committed Bytes`（GiB）、"
         "共有は WDDM の `GPU Adapter Memory\\Shared Usage`（参考値）。`invalid_*.json` は"
         "集計から除外している（無効と判定された記録。ファイル名の接頭辞で判定）。" % now(), ""]

    loads = [r for r in recs if r.get("kind") == "load"]
    if loads:
        L += ["## ロード", "",
              "| tag | transformer | finished | 所要秒 | コミット開始前→最大（増分） | 上限 | VRAM 最大 | active_after |",
              "|---|---|---|---|---|---|---|---|"]
        for r in loads:
            m = r.get("measure") or {}
            L.append("| %s | %s | %s | %s | %s→%s（+%s） | %s | %s | %s |" % (
                r.get("tag"), (r.get("request") or {}).get("models", {}).get("transformer"),
                (r.get("response") or {}).get("finished"), r.get("seconds"),
                cell(m.get("commit_gib_before")), cell(m.get("commit_gib_max")),
                cell(m.get("commit_gib_delta_max")), cell(m.get("commit_limit_gib")),
                cell(m.get("vram_mib_max")), r.get("models_active_after")))
        L.append("")

    gens = [r for r in recs if r.get("kind") in ("single", "iclora", "chain", "keep2")]
    if gens:
        L += ["## 生成（single / iclora / chain / keep2）", "",
              "| tag | 種別 | 状態 | transformer file | 生成秒 | 経過秒 | peak_vram_mb | smi 最大 | コミット最大 | 復元 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in gens:
            m, s = r.get("meta") or {}, r.get("measure") or {}
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r.get("tag"), r.get("kind"), r.get("status"), cell(m.get("transformer_file")),
                cell(m.get("generation_time_seconds")), cell(r.get("wall_seconds")),
                cell(m.get("peak_vram_mb")), cell(s.get("vram_mib_max")), cell(s.get("commit_gib_max")),
                str(r.get("recovered")) if r.get("recovered") else ""))
        L.append("")
        fails = [r for r in gens if r.get("status") != "completed"]
        for r in fails:
            L += ["失敗 `%s`: %s" % (r.get("tag"), cell(r.get("error"))), ""]

    keep2 = [r for r in recs if r.get("kind") == "keep2" and "sha256_r1" in r]
    for r in keep2:
        reused = "（--reuse で既存の記録から再判定）" if r.get("reused") else ""
        L += ["## keep_resident 2本の同一性（`%s`）%s" % (r.get("tag"), reused), "",
              "- 判定（映像/音声ストリームMD5一致＋PSNR∞）: **%s**" % r.get("match"),
              "- 映像ストリームMD5一致: %s（r1=%s… r2=%s…）" % (
                  r.get("video_md5_match"), (r.get("video_md5_r1") or "")[:12], (r.get("video_md5_r2") or "")[:12]),
              "- 音声ストリームMD5一致: %s（r1=%s… r2=%s…）" % (
                  r.get("audio_md5_match"), (r.get("audio_md5_r1") or "")[:12], (r.get("audio_md5_r2") or "")[:12]),
              "- 参考（合否には使わない）: ファイル全体SHA256一致: %s（r1=%s… r2=%s…。"
              "mp4コンテナのメタデータ（§3-164のjob_id等）の違いで不一致になり得る）" % (
                  r.get("sha256_match"), r.get("sha256_r1", "")[:12], r.get("sha256_r2", "")[:12]), ""]
        ps = r.get("psnr_ssim")
        if ps and ps.get("psnr"):
            L.append("- PSNR y=%s average=%s / SSIM Y=%s All=%s" % (
                ps["psnr"]["y"], ps["psnr"]["avg"], ps["ssim"]["y"], ps["ssim"]["all"]))
        L.append("")

    bad = [r for r in recs if r.get("kind") == "bad_header"]
    for r in bad:
        L += ["## 不合格ファイル（ケース `%s`）" % r.get("case"), "",
              "- ファイル: `%s`（%s バイト）・登録名: %s" % (r.get("file"), r.get("size"),
                                                         cell((r.get("registered") or {}).get("name"))),
              "- active: 前=%s 後=%s（拒否なら不変のはず）" % (r.get("active_before"), r.get("active_after")),
              "- ToolError: %s" % cell(r.get("tool_error")), ""]

    ps_list = [r for r in recs if r.get("kind") == "psnr_ssim"]
    if ps_list:
        L += ["## PSNR／SSIM（単独実行分）", "",
              "| tag | a | b | PSNR y | PSNR average | SSIM Y | SSIM All |",
              "|---|---|---|---|---|---|---|"]
        for r in ps_list:
            psnr, ssim = r.get("psnr") or {}, r.get("ssim") or {}
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                r.get("tag"), cell(r.get("a")), cell(r.get("b")), cell(psnr.get("y")),
                cell(psnr.get("avg")), cell(ssim.get("y")), cell(ssim.get("all"))))
        L.append("")

    REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    say("wrote %s" % REPORT_MD)
    print("\n".join(L))
    return 0


# === main ====================================================================
def main() -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(func=cmd_status)

    s = sub.add_parser("models")
    s.add_argument("--base-model", required=True, choices=["LTX23", "LTX25"])
    s.add_argument("--transformer", action="append", default=None)
    s.add_argument("--tag", default=None)
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("load")
    s.add_argument("--base-model", required=True, choices=["LTX23", "LTX25"])
    s.add_argument("--transformer", required=True)
    s.add_argument("--tag", default=None)
    s.set_defaults(func=cmd_load)

    s = sub.add_parser("restore")
    s.add_argument("--base-model", required=True, choices=["LTX23", "LTX25"])
    s.add_argument("--transformer", required=True)
    s.add_argument("--tag", default=None)
    s.set_defaults(func=cmd_restore)

    s = sub.add_parser("single")
    s.add_argument("--base-model", required=True, choices=["LTX23", "LTX25"])
    s.add_argument("--transformer", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--width", type=int, default=512)
    s.add_argument("--height", type=int, default=320)
    s.add_argument("--frames", type=int, default=49)
    s.add_argument("--seed", type=int, default=12345)
    s.add_argument("--prompt", default=PROMPT)
    s.add_argument("--lora", action="append", default=None, help="name[:strength]（複数可）")
    s.add_argument("--reference-video", default=None)
    s.add_argument("--keep-resident", action="store_true")
    s.add_argument("--prefetch", choices=["on", "off"], default=None,
                   help="block_swap_prefetch を明示指定（既定は省略=サーバー既定on）。"
                        "off で先読みなしの同期スワップ経路になる（G3）")
    s.set_defaults(func=cmd_single)

    s = sub.add_parser("iclora")
    s.add_argument("--base-model", required=True, choices=["LTX23", "LTX25"])
    s.add_argument("--transformer", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--reference", required=True, help="参照動画（絶対 or ROOT 相対）")
    s.add_argument("--ic-lora", required=True, dest="ic_lora")
    s.add_argument("--strength", type=float, default=1.0)
    s.add_argument("--width", type=int, default=512)
    s.add_argument("--height", type=int, default=384)
    s.add_argument("--frames", type=int, default=49)
    s.add_argument("--seed", type=int, default=12345)
    s.add_argument("--prompt", default=PROMPT)
    s.set_defaults(func=cmd_iclora)

    s = sub.add_parser("chain")
    s.add_argument("--base-model", required=True, choices=["LTX23", "LTX25"])
    s.add_argument("--transformer", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--width", type=int, default=512)
    s.add_argument("--height", type=int, default=320)
    s.add_argument("--clip-frames", type=int, nargs="+", default=[49, 49])
    s.add_argument("--seed", type=int, default=12345)
    s.add_argument("--prompt", default=PROMPT)
    s.add_argument("--lora", action="append", default=None)
    s.set_defaults(func=cmd_chain)

    s = sub.add_parser("keep2")
    s.add_argument("--base-model", required=True, choices=["LTX23", "LTX25"])
    s.add_argument("--transformer", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--width", type=int, default=512)
    s.add_argument("--height", type=int, default=320)
    s.add_argument("--frames", type=int, default=49)
    s.add_argument("--seed", type=int, default=12345)
    s.add_argument("--prompt", default=PROMPT)
    s.add_argument("--reuse", action="store_true",
                   help="生成をやり直さず、既存の results/<tag>_r1.json / _r2.json から新基準で再判定するだけ"
                        "（GPU不使用。gates.py --resume 用）")
    s.set_defaults(func=cmd_keep2)

    s = sub.add_parser("bad-header")
    s.add_argument("--base-model", required=True, choices=["LTX23", "LTX25"])
    s.add_argument("--weights-dir", required=True)
    s.add_argument("--case", default=next(iter(BAD_HEADER_CASES)))
    s.add_argument("--tag", default=None)
    s.add_argument("--cleanup", action="store_true")
    s.set_defaults(func=cmd_bad_header)

    s = sub.add_parser("psnr-ssim")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--tag", default=None)
    s.set_defaults(func=cmd_psnr_ssim)

    s = sub.add_parser("save")
    s.add_argument("--job-id", required=True, dest="job_id")
    s.add_argument("--dest", required=True)
    s.add_argument("--filename", default=None)
    s.add_argument("--which", choices=["output", "joined"], default="output")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--tag", default=None)
    s.set_defaults(func=cmd_save)

    s = sub.add_parser("record")
    s.add_argument("--job-id", required=True, dest="job_id")
    s.add_argument("--tag", required=True)
    s.add_argument("--kind", choices=["single", "chain", "iclora"], default="single")
    s.add_argument("--from-outputs", action="store_true", dest="from_outputs",
                   help="MCP/バックエンドを使わず、ローカルの outputs/<job-id>/"
                        "{metadata.json,output.mp4} だけから直接読む（バックエンド停止中でも使える）")
    s.set_defaults(func=cmd_record)

    sub.add_parser("report").set_defaults(func=cmd_report)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
