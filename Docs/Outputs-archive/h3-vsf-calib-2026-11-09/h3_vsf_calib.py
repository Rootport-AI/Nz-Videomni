r"""MiniMax H3 の VSF（Value Sign Flip。負のプロンプトの行の値の符号を反転して負の要素を遠ざける方式）の
実用域を較正する 18 本を、MCP サーバー経由で回す台本（台帳 ``PENDING_TASKS.md`` §3-278・記録簿
``VERIFICATION_LOG.md`` §203）。

18 本の条件は同じフォルダの ``calib_plan.json`` を正本とし、この台本はそれを読んで引数を組み立てる
（段 A・B・A′〔名前は R_*〕・C・D。回す順は A → B → R → C → D）。生成しない対照 ``A_off`` は第 1 弾の
``../h3-vsf-2026-11-09/visual/T0`` を ``jobs/A_off/`` に写す。製品コードは import しない。部品は第 1 弾の台本
``Docs/Outputs-archive/h3-vsf-2026-11-09/h3_vsf_gate.py`` から写した（一時 yaml・``Server``・MCP の stdio
クライアント・``check_tool_schemas``・``ensure_pipeline``・``wait_job``・``run_job``・``vsf_block``）。

流れ:
  1. ``A_off`` が ``jobs/A_off/`` に無ければ、元のフォルダから mp4・``metadata.json``・``request.json`` を写す。
  2. ``config.yaml`` を読むだけで、``server.port``・``server.log_dir``・``output.dir``・``upload.dir``・``state_file``
     だけを ``_rt\`` に向けた一時 yaml（``config.h3vsfcalib.yaml``）と、MiniMax H3 を選んだ一時 state
     （``state.h3vsfcalib.json``）を ``_rt\`` に書く。
  3. ``.venv\Scripts\python.exe main.py --config <一時 yaml>`` で別ポート（既定 18699）に実バックエンドを起こし、
     ``GET /api/v1/status`` が 200 を返すまで待つ。
  4. MCP サーバー（``python -m mcp_server``）を子プロセスとして起こす。環境に ``PYTHONUTF8=1``・
     ``LTX_MCP_BASE_URL=http://127.0.0.1:<port>``（``/api/v1`` を含まない形）・``LTX_MCP_CONFIG=<一時 yaml>``・
     ``LTX_MCP_OUTPUT_DIR=_rt\outputs`` を渡す。``tools/list`` の引数名を照合してから回す。
  5. ``load_pipeline(base_model="MiniMaxH3", models={"transformer": "default"}, wait_sec=45)`` を 1 回、
     以後 ``backend_status`` を 10 秒おき。
  6. 選んだ本を順に回す（1 本ごとに ``jobs/<名前>/`` へ request.json・metadata.json・mp4・worker_log_excerpt.txt）。
     待ちきれなかった本（``wait_gave_up``）が出たら残りは送らない。
  7. ``--keep-server`` でなければ ``unload_pipeline`` → サーバーをプロセスの木ごと停止。
  8. ``summary.json`` を書く（前の回の結果に、名前ごとに上書きで足す）。

終了コード: 0＝回した本がすべて completed、3＝どれかが失敗（先へは進む）、
1＝準備の段で止まった（サーバーが起きない・道具の引数名が違う・読み込みの失敗など）、
``--selfcheck`` は 0＝すべて PASS・1＝どれかが FAIL。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # Docs\Outputs-archive\<このフォルダ> の 3 つ上＝リポジトリ直下
RT = HERE / "_rt"
JOBS = HERE / "jobs"
PLAN_PATH = HERE / "calib_plan.json"
PY = REPO / ".venv" / "Scripts" / "python.exe"
PORT_DEFAULT = 18699
TEMP_CONFIG_NAME = "config.h3vsfcalib.yaml"
TEMP_STATE_NAME = "state.h3vsfcalib.json"

H3_BASE_MODEL = "MiniMaxH3"
H3_CATEGORIES = ("transformer", "text_encoder", "video_vae", "audio")

VSF_KEYS = ("rows", "scale", "m_neg", "m_text")
STAGE_ORDER = ("A", "B", "R", "C", "D")

TOOL_READ_TIMEOUT = timedelta(seconds=360)
WAIT_TIMEOUT_SEC = 45
WAIT_POLL_SEC = 5.0
MAX_CONSECUTIVE_TRANSPORT_ERRORS = 12

# tools/list の inputSchema で照合する引数名。台本が実際に渡す名前だけを並べる（無ければ止まる）。
REQUIRED_TOOL_ARGS: dict[str, set[str]] = {
    "backend_status": set(),
    "load_pipeline": {"base_model", "models", "wait_sec"},
    "unload_pipeline": set(),
    "submit_generate": {
        "prompt", "negative_prompt", "nag_enabled", "neg_method", "vsf_scale", "width", "height", "num_frames",
        "frame_rate", "seed", "loras", "attention_backend", "num_inference_steps", "shift_video", "shift_audio",
        "vram16_mode",
    },
    "wait_for_job": {"job_id", "timeout_sec", "poll_interval_sec"},
    "save_job_video": {"job_id", "dest_dir", "no_clobber", "which"},
    "get_job_video_path": {"job_id"},
}

# --selfcheck の突き合わせ先。計画書 §3 の表をそのまま写したもの（calib_plan.json とは独立に持つ）。
#: (名前, 段, seed, 幅, 高さ, フレーム, 負の文の記号, scale)
N1 = "blue clothes, cars"
N2 = "blue denim shirt, blue jacket, blue clothing, white car, vehicle, automobile"
EXPECTED_JOBS: list[tuple[str, str, int, int, int, int, str, float]] = (
    [(f"A_s{s:g}", "A", 42, 1344, 768, 56, "N1", s) for s in (0, 1, 1.5, 2, 2.5, 3, 4, 5)]
    + [(f"B_s{s:g}", "B", 42, 1344, 768, 56, "N1", s) for s in (6, 8)]
    + [(f"R_s{s:g}", "R", 43, 1344, 768, 56, "N1", s) for s in (0, 1, 1.5, 2)]
    + [("C_s1.5_N2", "C", 42, 1344, 768, 56, "N2", 1.5)]
    + [(f"L_s{s:g}", "D", 42, 1344, 768, 124, "N1", s) for s in (0, 1.5, 3)]
)
EXPECTED_COMMON = {
    "prompt": ("A Japanese woman in casual clothes walks toward the camera along a city street and smiles, "
               "reaching one hand toward the viewer; steady camera, natural daylight."),
    "frame_rate": 24.0, "attention_backend": "sage", "vram16_mode": True, "shift_video": 12.0, "shift_audio": 3.0,
    "num_inference_steps": 4,
    "loras": [{"name": "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16", "strength": 0.8}],
}
FORBIDDEN_KEYS = ("embed_mp4_metadata", "keep_resident")  # 既定のまま（第 1 弾の T0 と同じ）。送らない

# ---------------------------------------------------------------------------------------------
# 小道具
# ---------------------------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str | None:
    if not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vsf_block(meta: dict) -> tuple[dict | None, str | None]:
    """metadata.json の VSF の記録と、その在りか。計画では ``metadata.h3.vsf``。念のため最上位も見る。"""
    h3 = meta.get("h3")
    if isinstance(h3, dict) and "vsf" in h3:
        return h3.get("vsf"), "h3.vsf"
    if "vsf" in meta:
        return meta.get("vsf"), "vsf"
    return None, None


# ---------------------------------------------------------------------------------------------
# 一時 yaml とサーバー（g5.py の write_temp_config／Server に倣う）
# ---------------------------------------------------------------------------------------------

def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def http(port: int, method: str, path: str, body=None, timeout: float = 60.0):
    url = base_url(port) + "/api/v1" + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except Exception:  # noqa: BLE001
            return exc.code, {"raw": raw}


def write_temp_config(port: int) -> tuple[Path, Path]:
    """``config.yaml`` は読むだけ。5 つの鍵だけを ``_rt\\`` に向けた一時 yaml と、H3 を選んだ一時 state を書く。"""
    import yaml  # アプリの仮想環境の依存（config.py が使う）

    RT.mkdir(parents=True, exist_ok=True)
    for sub in ("logs", "outputs", "uploads"):
        (RT / sub).mkdir(exist_ok=True)

    raw = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = port
    raw["server"]["log_dir"] = str(RT / "logs")
    raw.setdefault("output", {})["dir"] = str(RT / "outputs")
    raw.setdefault("upload", {})["dir"] = str(RT / "uploads")
    raw["state_file"] = str(RT / TEMP_STATE_NAME)
    cfg = RT / TEMP_CONFIG_NAME
    cfg.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # 今の state.json の形（schema・selections）を写し、MiniMax H3 の 4 カテゴリを default にして H3 を選ぶ。
    selections: dict[str, dict[str, str]] = {}
    try:
        current = json.loads((REPO / "state.json").read_text(encoding="utf-8"))
        if isinstance(current.get("selections"), dict):
            selections = {k: dict(v) for k, v in current["selections"].items() if isinstance(v, dict)}
    except Exception:  # noqa: BLE001 -- state.json はキャッシュ。無くても既定で組む
        pass
    for base in ("LTX23", "LTX25"):
        selections.setdefault(base, {c: "default" for c in H3_CATEGORIES})
    selections[H3_BASE_MODEL] = {c: "default" for c in H3_CATEGORIES}
    state = RT / TEMP_STATE_NAME
    state.write_text(json.dumps({"schema": 1, "active_base_model": H3_BASE_MODEL, "selections": selections},
                                indent=2), encoding="utf-8")
    return cfg, state


def all_processes() -> list[dict]:
    """全プロセスの PID・親 PID・名前・コマンド行（``Get-CimInstance Win32_Process``）。"""
    ps = ("Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
          "ConvertTo-Json -Depth 3")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    try:
        data = json.loads(r.stdout or "null")
    except json.JSONDecodeError:
        return []
    if data is None:
        return []
    if isinstance(data, dict):
        data = [data]
    return [{"pid": d.get("ProcessId"), "ppid": d.get("ParentProcessId"), "name": d.get("Name"),
             "cmdline": d.get("CommandLine") or ""} for d in data]


def descendants(root_pid: int, procs: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen = {root_pid}
    frontier = {root_pid}
    while frontier:
        children = [p for p in procs if p["ppid"] in frontier and p["pid"] not in seen]
        for p in children:
            seen.add(p["pid"])
            out.append(p)
        frontier = {p["pid"] for p in children}
    return out


class Server:
    def __init__(self, port: int) -> None:
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.reused_pid: int | None = None
        self.log_path = RT / "server.log"
        self.config_path: Path | None = None
        self._fh = None
        self.record: dict[str, Any] = {"port": port}

    def alive(self) -> bool:
        try:
            status, _ = http(self.port, "GET", "/status", timeout=5)
            return status == 200
        except Exception:  # noqa: BLE001
            return False

    def start(self) -> None:
        self.config_path = RT / TEMP_CONFIG_NAME
        if self.alive():
            # 前の回の --keep-server で残したサーバーだけを使い回す。コマンド行に一時 yaml の名前が無ければ他人のもの。
            ours = [p for p in all_processes() if TEMP_CONFIG_NAME in p["cmdline"] and "main.py" in p["cmdline"]]
            if not ours:
                raise RuntimeError(f"ポート {self.port} で別のサーバーが応答している（{TEMP_CONFIG_NAME} で起こしたもの"
                                   "ではない）。--port で別のポートを選ぶか、そのサーバーを止めてから回す")
            self.reused_pid = ours[0]["pid"]
            self.record.update(reused=True, pid=self.reused_pid, cmdline=ours[0]["cmdline"])
            log(f"ポート {self.port} の前の回のサーバー（pid {self.reused_pid}）を使い回す")
            return
        cfg, state = write_temp_config(self.port)
        env = dict(os.environ)
        # run.ps1 と同じ環境（tools\uv と tools\ffmpeg\bin を PATH の先頭へ・uv の python の置き場）。
        # g5.py が足していた PYTORCH_CUDA_ALLOC_CONF は利用者の起動（run.ps1）に無いので足さない。
        prefix = ""
        for d in (REPO / "tools" / "uv", REPO / "tools" / "ffmpeg" / "bin"):
            if d.exists():
                prefix += f"{d};"
        env["PATH"] = prefix + env.get("PATH", "")
        env["UV_PYTHON_INSTALL_DIR"] = str(REPO / ".python")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        self._fh = self.log_path.open("a", encoding="utf-8", errors="replace")
        self._fh.write(f"\n===== {now_iso()} h3_vsf_calib.py: main.py --config {cfg} =====\n")
        self._fh.flush()
        cmd = [str(PY), "-u", str(REPO / "main.py"), "--config", str(cfg)]
        self.proc = subprocess.Popen(cmd, cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=self._fh,
                                     stderr=subprocess.STDOUT, env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.record.update(reused=False, pid=self.proc.pid, cmd=cmd, config=str(cfg), state=str(state),
                           log=str(self.log_path), started_at=now_iso())
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.alive():
                log(f"サーバーが起きた（pid {self.proc.pid}・ポート {self.port}）")
                return
            if self.proc.poll() is not None:
                raise RuntimeError("サーバーが起動中に終わった:\n"
                                   + self.log_path.read_text(encoding="utf-8", errors="replace")[-3000:])
            time.sleep(1.0)
        raise RuntimeError("180 秒待っても /api/v1/status が応答しない")

    def stop(self) -> None:
        """unload → terminate → だめなら taskkill /T /F。最後に子孫（ワーカー）が残っていれば木ごと止める。"""
        if self.proc is None and self.reused_pid is None:
            return  # 起こしても使い回してもいない（起動前に止まった・他人のサーバーだった）。何もしない
        rec: dict[str, Any] = {"at": now_iso()}
        try:
            rec["unload_http"] = http(self.port, "POST", "/pipeline/unload", timeout=600)[0]
        except Exception as exc:  # noqa: BLE001
            rec["unload_http"] = f"{type(exc).__name__}: {exc}"
        pid = self.proc.pid if self.proc is not None else self.reused_pid
        kids = descendants(pid, all_processes()) if pid else []
        rec["descendants_before"] = [{"pid": p["pid"], "cmdline": p["cmdline"][:300]} for p in kids]
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=60)
                rec["terminate"] = "ok"
            except Exception as exc:  # noqa: BLE001
                rec["terminate"] = f"{type(exc).__name__}: {exc}"
                rec["taskkill"] = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                                                 text=True, errors="replace").stdout.strip()
            self.proc = None
        elif pid:
            rec["taskkill"] = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                                             text=True, errors="replace").stdout.strip()
        alive_pids = {p["pid"] for p in all_processes()}
        left = [p for p in kids if p["pid"] in alive_pids]
        for p in left:
            subprocess.run(["taskkill", "/PID", str(p["pid"]), "/T", "/F"], capture_output=True)
        rec["descendants_killed_after"] = [p["pid"] for p in left]
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._fh = None
        self.record["stop"] = rec
        log(f"サーバーを止めた（pid {pid}・残っていた子孫 {rec['descendants_killed_after']}）")


def worker_log_path() -> Path | None:
    p = RT / "logs" / "h3_worker.log"
    if p.exists():
        return p
    found = sorted((RT / "logs").glob("*h3*worker*.log"))
    return found[0] if found else p


def log_offset() -> int:
    p = worker_log_path()
    return p.stat().st_size if p is not None and p.exists() else 0


def log_since(offset: int) -> str:
    p = worker_log_path()
    if p is None or not p.exists():
        return ""
    with p.open("rb") as fh:
        fh.seek(offset)
        return fh.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------------------------
# MCP の呼び出し（h3_retake_gate.py の写し）
# ---------------------------------------------------------------------------------------------

class ToolCallError(Exception):
    """道具がエラーを返した（``isError``）か、呼び出しそのものが失敗した。``text`` は全文。"""

    def __init__(self, tool: str, text: str) -> None:
        super().__init__(f"{tool}: {text}")
        self.tool = tool
        self.text = text


class Mcp:
    def __init__(self, session, calls_log: list) -> None:
        self.session = session
        self.calls_log = calls_log

    async def call(self, tool: str, args: dict[str, Any] | None = None) -> dict:
        t0 = time.monotonic()
        entry: dict[str, Any] = {"at": now_iso(), "tool": tool, "args": args or {}}
        try:
            result = await self.session.call_tool(tool, args or {}, read_timeout_seconds=TOOL_READ_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 -- 転送の失敗も全文で残す
            entry.update(ok=False, seconds=round(time.monotonic() - t0, 2),
                         error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            self.calls_log.append(entry)
            raise ToolCallError(tool, entry["error"]) from exc
        entry["seconds"] = round(time.monotonic() - t0, 2)
        text = "\n".join(getattr(c, "text", "") for c in (result.content or []) if getattr(c, "text", None))
        if result.isError:
            entry.update(ok=False, error=text)
            self.calls_log.append(entry)
            raise ToolCallError(tool, text)
        if isinstance(result.structuredContent, dict):
            data = result.structuredContent
        else:
            try:
                data = json.loads(text) if text else {}
            except json.JSONDecodeError:
                data = {"_text": text}
        entry["ok"] = True
        if tool not in ("wait_for_job", "backend_status"):
            entry["result"] = data
        self.calls_log.append(entry)
        return data


def check_tool_schemas(tools: list) -> dict:
    by_name = {t.name: t for t in tools}
    report: dict[str, Any] = {"tool_count": len(tools), "problems": []}
    for name, needed in REQUIRED_TOOL_ARGS.items():
        tool = by_name.get(name)
        if tool is None:
            report["problems"].append(f"道具 {name} が tools/list に無い")
            continue
        props = set((tool.inputSchema or {}).get("properties", {}))
        missing = needed - props
        if missing:
            report["problems"].append(f"{name} の inputSchema に引数 {sorted(missing)} が無い（あるのは {sorted(props)}）")
    report["ok"] = not report["problems"]
    return report


async def ensure_pipeline(mcp: Mcp, args, run_rec: dict) -> bool:
    """H3 を載せる。load_pipeline は 1 回だけ（読み込み中の呼び直しは 409）。以後 backend_status を 10 秒おき。"""
    load: dict[str, Any] = {"requested": {"base_model": H3_BASE_MODEL, "models": {"transformer": "default"},
                                          "wait_sec": 45}, "responses": [], "status_polls": 0}
    run_rec["load"] = load
    t0 = time.monotonic()
    try:
        resp = await mcp.call("load_pipeline", load["requested"])
        load["responses"].append(resp)
        log(f"load_pipeline: finished={resp.get('finished')} pipeline_loaded={resp.get('pipeline_loaded')}")
    except ToolCallError as exc:
        load["responses"].append({"error": exc.text})
        if "PIPELINE_LOADING" not in exc.text:
            load["error"] = exc.text
            log(f"load_pipeline が失敗: {exc.text}")
            return False
        log("既に読み込み中（PIPELINE_LOADING）。終わるのを待つ")
    deadline = t0 + args.load_timeout_sec
    while True:
        st = await mcp.call("backend_status")
        load["status_polls"] += 1
        s = st.get("status") or {}
        load["last_status"] = s
        if not st.get("reachable"):
            load["error"] = f"バックエンドに届かない: {st.get('error')}"
            return False
        if s.get("pipeline_loaded") and s.get("base_model") == H3_BASE_MODEL and s.get("state") in ("ready", "running"):
            break
        if s.get("state") == "error":
            load["error"] = f"パイプラインの状態が error: {s}"
            return False
        if time.monotonic() > deadline:
            load["error"] = f"{args.load_timeout_sec} 秒待っても読み込みが終わらない（最後の状態 {s}）"
            return False
        await asyncio.sleep(10)
    load["seconds"] = round(time.monotonic() - t0, 1)
    log(f"H3 が載った（{load['seconds']} 秒）")
    return True


async def wait_job(mcp: Mcp, job_id: str, timeout_sec: float) -> dict:
    """wait_for_job を繰り返す。timed_out:true は失敗ではない。転送の失敗は連続 12 回まで許す。"""
    t0 = time.monotonic()
    calls = 0
    errors_in_row = 0
    samples: list[dict] = []
    last_key = None
    last: dict = {}
    while True:
        try:
            st = await mcp.call("wait_for_job", {"job_id": job_id, "timeout_sec": WAIT_TIMEOUT_SEC,
                                                 "poll_interval_sec": WAIT_POLL_SEC})
            errors_in_row = 0
        except ToolCallError as exc:
            errors_in_row += 1
            samples.append({"t": round(time.monotonic() - t0, 1), "error": exc.text})
            if errors_in_row >= MAX_CONSECUTIVE_TRANSPORT_ERRORS:
                return {"final": last, "calls": calls, "seconds": round(time.monotonic() - t0, 1),
                        "samples": samples, "gave_up": f"wait_for_job が {errors_in_row} 回続けて失敗: {exc.text}"}
            await asyncio.sleep(5)
            continue
        calls += 1
        last = st
        key = (st.get("status"), st.get("stage"), st.get("current_step"))
        if key != last_key:
            samples.append({"t": round(time.monotonic() - t0, 1), "status": st.get("status"),
                            "progress": st.get("progress"), "stage": st.get("stage"),
                            "current_step": st.get("current_step")})
            log(f"  job {job_id[:8]} status={st.get('status')} progress={st.get('progress')} stage={st.get('stage')}")
            last_key = key
        if not st.get("timed_out"):
            return {"final": st, "calls": calls, "seconds": round(time.monotonic() - t0, 1), "samples": samples}
        if time.monotonic() - t0 > timeout_sec:
            return {"final": st, "calls": calls, "seconds": round(time.monotonic() - t0, 1), "samples": samples,
                    "gave_up": f"{timeout_sec} 秒待っても終わらない（ジョブはバックエンドで続いている）"}



# ---------------------------------------------------------------------------------------------
# 計画（calib_plan.json）と引数の組み立て
# ---------------------------------------------------------------------------------------------

def load_plan(path: Path = PLAN_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generated_jobs(plan: dict) -> list[dict]:
    """生成する本（``generate: false`` を除く）を、段の順（A → B → R → C → D）・計画の中の順に並べる。"""
    order = {s: i for i, s in enumerate(plan.get("stage_order") or STAGE_ORDER)}
    jobs = [j for j in plan["jobs"] if j.get("generate", True)]
    return sorted(jobs, key=lambda j: (order.get(j["stage"], 99), plan["jobs"].index(j)))


def build_args(plan: dict, job: dict) -> dict:
    """1 本の submit_generate の引数。共通の引数に、寸法・seed・負の文・scale を足す（scale 0 も送る）。"""
    c = plan["common"]
    a: dict[str, Any] = {"prompt": c["prompt"], "width": job["width"], "height": job["height"],
                         "num_frames": job["num_frames"], "frame_rate": c["frame_rate"], "seed": job["seed"],
                         "attention_backend": c["attention_backend"], "vram16_mode": c["vram16_mode"],
                         "shift_video": c["shift_video"], "shift_audio": c["shift_audio"],
                         "loras": [dict(x) for x in c["loras"]], "num_inference_steps": c["num_inference_steps"]}
    if job.get("negative") is not None:
        a.update(nag_enabled=True, neg_method="vsf", negative_prompt=job["negative"],
                 vsf_scale=float(job["vsf_scale"]))
    return a


def job_note(job: dict) -> str:
    neg = "負の文なし" if job.get("negative") is None else f"負の文 `{job['negative']}`"
    return (f"段 {job['stage']}・seed {job['seed']}・{job['width']}×{job['height']}×{job['num_frames']}・{neg}・"
            f"scale {job['vsf_scale']}・{job.get('note', '')}")


def select_jobs(plan: dict, stage: str, only: str | None) -> list[dict]:
    jobs = generated_jobs(plan)
    if stage != "all":
        jobs = [j for j in jobs if j["stage"] == stage]
    if only:
        names = [n.strip() for n in only.split(",") if n.strip()]
        known = {j["name"] for j in generated_jobs(plan)}
        unknown = [n for n in names if n not in known]
        if unknown:
            raise SystemExit(f"--only に計画に無い名前（生成しない A_off を含む）: {unknown}")
        jobs = [j for j in jobs if j["name"] in names]
    return jobs


# ---------------------------------------------------------------------------------------------
# A_off（第 1 弾の T0）の写し
# ---------------------------------------------------------------------------------------------

def copy_a_off(plan: dict, summary: dict) -> None:
    """``generate: false`` の本を ``jobs/<名前>/`` に写す（既にあれば写さない）。summary にも 1 行を足す。"""
    for job in plan["jobs"]:
        if job.get("generate", True):
            continue
        src = (HERE / job["source"]).resolve()
        dst = JOBS / job["name"]
        dst.mkdir(parents=True, exist_ok=True)
        copied = []
        for f in sorted(src.glob("*.mp4")) + [src / "metadata.json", src / "request.json"]:
            if f.is_file() and not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
                copied.append(f.name)
        if copied:
            log(f"{job['name']}: {src} から {copied} を写した")
        if job["name"] in summary["jobs"]:
            continue
        meta: dict = {}
        try:
            meta = json.loads((dst / "metadata.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        phase1: dict = {}
        try:  # 待ちの秒は第 1 弾の summary.json にだけある（無くても困らない）
            phase1 = json.loads((src.parents[1] / "summary.json").read_text(encoding="utf-8"))["jobs"][src.name]
        except Exception:  # noqa: BLE001
            pass
        mp4 = sorted(dst.glob("*.mp4"))
        summary["jobs"][job["name"]] = {
            "label": job["name"], "stage": job["stage"], "scale": None, "seed": job["seed"],
            "size": [job["width"], job["height"], job["num_frames"]], "negative": None,
            "status": "copied" if mp4 else "missing", "source": str(src),
            "seconds_wall": phase1.get("seconds_wall"),
            "generation_time_seconds": meta.get("generation_time_seconds"),
            "peak_vram_reserved_mb": meta.get("peak_vram_reserved_mb"),
            "peak_vram_mb": (meta.get("vram_optimization") or {}).get("peak_vram_mb"),
            "attention_used": meta.get("attention_used"),
            "vsf": None, "negative_tokens": None,
            "sha256": sha256(mp4[0]) if mp4 else None,
            "error": None if mp4 else f"{src} に mp4 が無い",
        }


# ---------------------------------------------------------------------------------------------
# 1 本のジョブ（h3_vsf_gate.py の run_job の写し）
# ---------------------------------------------------------------------------------------------

async def run_job(mcp: Mcp, args, label: str, job_dir: Path, tool_args: dict, note: str) -> dict:
    """submit_generate → 待つ → save_job_video → metadata.json を写す → sha256・h3.vsf・ワーカーのログを記録。"""
    job_dir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"label": label, "dir": str(job_dir), "note": note, "tool": "submit_generate",
                           "tool_args": tool_args, "started_at": now_iso(), "status": None, "error": None}
    write_json(job_dir / "request.json", {"tool": "submit_generate", "arguments": tool_args, "note": note,
                                          "remark": "MCP の道具に渡した引数。サーバーが受けた本文は metadata.json の request"})
    log(f"=== {label}: {note} ===")
    offset = log_offset()
    t0 = time.monotonic()
    try:
        sub = await mcp.call("submit_generate", tool_args)
    except ToolCallError as exc:
        rec.update(status="submit_error", error=exc.text, finished_at=now_iso())
        log(f"submit_generate が拒否: {exc.text}")
        return rec
    job_id = sub.get("job_id")
    rec["job_id"] = job_id
    log(f"job_id={job_id}")
    waited = await wait_job(mcp, job_id, args.job_timeout_sec)
    final = waited.pop("final") or {}
    rec["wait"] = waited
    rec["job_final"] = final
    rec["status"] = final.get("status") or "unknown"
    rec["seconds_wall"] = round(time.monotonic() - t0, 1)
    excerpt = log_since(offset)
    if excerpt:
        (job_dir / "worker_log_excerpt.txt").write_text(excerpt, encoding="utf-8")
    rec["worker_log_lines"] = [ln for ln in excerpt.splitlines() if "neg=" in ln or "vsf" in ln.lower()
                               or "m_neg" in ln]
    if waited.get("gave_up"):
        rec.update(status="wait_gave_up", error=waited["gave_up"], finished_at=now_iso())
        return rec
    if rec["status"] != "completed":
        rec["error"] = final.get("error")
        rec["finished_at"] = now_iso()
        log(f"ジョブが {rec['status']}: {final.get('error')}")
        return rec
    try:
        saved = await mcp.call("save_job_video", {"job_id": job_id, "dest_dir": str(job_dir),
                                                  "no_clobber": True, "which": "output"})
        rec["saved"] = saved
        saved_path = Path(saved["saved_path"])
        rec["sha256"] = sha256(saved_path)
    except ToolCallError as exc:
        rec.update(error=f"save_job_video: {exc.text}", finished_at=now_iso())
        return rec
    meta: dict = {}
    try:
        vp = await mcp.call("get_job_video_path", {"job_id": job_id})
        src = Path(vp["path"]).parent / "metadata.json"
        if src.exists():
            shutil.copy2(src, job_dir / "metadata.json")
            meta = json.loads(src.read_text(encoding="utf-8"))
        else:
            rec["metadata_missing"] = str(src)
    except ToolCallError as exc:
        rec["metadata_missing"] = exc.text
    vsf, where = vsf_block(meta)
    rec["vsf"] = vsf
    rec["vsf_where"] = where
    rec["metadata_summary"] = {
        "attention_used": meta.get("attention_used"),
        "vram16_mode_used": meta.get("vram16_mode_used"),
        "keep_resident_used": meta.get("keep_resident_used"),
        "generation_time_seconds": meta.get("generation_time_seconds"),
        "peak_vram_reserved_mb": meta.get("peak_vram_reserved_mb"),
        "peak_vram_mb": (meta.get("vram_optimization") or {}).get("peak_vram_mb"),
        "seed_used": meta.get("seed_used"),
        "num_inference_steps": (meta.get("h3") or {}).get("num_inference_steps"),
        "shift_video": (meta.get("h3") or {}).get("shift_video"),
        "shift_audio": (meta.get("h3") or {}).get("shift_audio"),
        "negative_tokens": (((meta.get("h3") or {}).get("phases") or {}).get("encode") or {}).get("negative_tokens"),
    }
    rec["finished_at"] = now_iso()
    log(f"完了: sha256={(rec['sha256'] or '')[:16]}… vsf={vsf}")
    return rec


def flatten_record(rec: dict, job: dict) -> dict:
    """summary.json の 1 本に、表で使う欄を平らに足す（metadata_summary の値を上に出す）。"""
    ms = rec.get("metadata_summary") or {}
    rec.update(stage=job["stage"], scale=job["vsf_scale"], seed=job["seed"],
               size=[job["width"], job["height"], job["num_frames"]], negative=job.get("negative"),
               generation_time_seconds=ms.get("generation_time_seconds"),
               peak_vram_reserved_mb=ms.get("peak_vram_reserved_mb"), peak_vram_mb=ms.get("peak_vram_mb"),
               attention_used=ms.get("attention_used"), negative_tokens=ms.get("negative_tokens"))
    for k in ("seconds_wall", "vsf", "sha256", "error"):
        rec.setdefault(k, None)
    return rec


async def run_calib(mcp: Mcp, args, plan: dict, summary: dict) -> bool:
    jobs = select_jobs(plan, args.stage, args.only)
    log(f"##### 較正（段 {args.stage}・{len(jobs)} 本: {[j['name'] for j in jobs]}） #####")
    ok = True
    for job in jobs:
        label = job["name"]
        tool_args = build_args(plan, job)
        note = job_note(job)
        try:
            rec = await run_job(mcp, args, label, JOBS / label, tool_args, note)
        except Exception as exc:  # noqa: BLE001 -- 1 本の例外で残りを止めない
            rec = {"label": label, "status": "script_error", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(), "tool_args": tool_args}
        summary["jobs"][label] = flatten_record(rec, job)
        flush(summary)
        if rec.get("status") != "completed" or rec.get("error"):
            ok = False
        if rec.get("status") == "wait_gave_up":
            log("待ちきれなかったジョブがバックエンドで走っているため、残りは送らない（JOB_BUSY になる）")
            break
    key = args.stage if not args.only else f"{args.stage}/only:{args.only}"
    summary["stages"][key] = {"pass": ok, "at": now_iso()}
    return ok


# ---------------------------------------------------------------------------------------------
# まとめ
# ---------------------------------------------------------------------------------------------

def load_summary() -> dict:
    p = HERE / "summary.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("jobs", {})
                data.setdefault("stages", {})
                data.setdefault("runs", [])
                return data
        except Exception:  # noqa: BLE001
            pass
    return {"script": "h3_vsf_calib.py", "plan": "台帳 §3-278・記録簿 §203（条件の正本は calib_plan.json）",
            "jobs": {}, "stages": {}, "runs": []}


def flush(summary: dict) -> None:
    write_json(HERE / "summary.json", summary)


async def run(args, plan: dict, summary: dict, run_rec: dict) -> int:
    from mcp import ClientSession, StdioServerParameters  # アプリの仮想環境の依存
    from mcp.client.stdio import stdio_client

    calls: list = []
    calls_path = HERE / "logs" / f"mcp_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    run_rec["mcp_calls"] = str(calls_path)
    env = {**os.environ, "PYTHONUTF8": "1", "LTX_MCP_BASE_URL": base_url(args.port),
           "LTX_MCP_CONFIG": str(RT / TEMP_CONFIG_NAME), "LTX_MCP_OUTPUT_DIR": str(RT / "outputs")}
    run_rec["mcp_env"] = {k: env[k] for k in ("PYTHONUTF8", "LTX_MCP_BASE_URL", "LTX_MCP_CONFIG", "LTX_MCP_OUTPUT_DIR")}
    server = StdioServerParameters(command=str(PY), args=["-m", "mcp_server"], cwd=str(REPO), env=env,
                                   encoding="utf-8")
    stderr_log = HERE / "logs" / "mcp_server.stderr.log"
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    code = 0
    try:
        with open(stderr_log, "a", encoding="utf-8") as errlog:
            errlog.write(f"\n===== {now_iso()} h3_vsf_calib.py --stage {args.stage} --only {args.only} =====\n")
            errlog.flush()
            async with stdio_client(server, errlog=errlog) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=TOOL_READ_TIMEOUT) as session:
                    await session.initialize()
                    schema = check_tool_schemas((await session.list_tools()).tools)
                    run_rec["tool_schema_check"] = schema
                    if not schema["ok"]:
                        for p in schema["problems"]:
                            log(f"引数名の照合に失敗: {p}")
                        return 1
                    mcp = Mcp(session, calls)
                    st = await mcp.call("backend_status")
                    run_rec["status_before"] = st
                    if not st.get("reachable"):
                        log(f"バックエンドに届かない: {st.get('error')}")
                        return 1
                    if not await ensure_pipeline(mcp, args, run_rec):
                        log(f"読み込みで止まる: {run_rec['load'].get('error')}")
                        return 1
                    flush(summary)
                    try:
                        ok = await run_calib(mcp, args, plan, summary)
                    except Exception as exc:  # noqa: BLE001
                        summary["stages"][args.stage] = {"pass": False, "error": f"{type(exc).__name__}: {exc}",
                                                         "traceback": traceback.format_exc()}
                        ok = False
                    if not ok:
                        code = 3
                    flush(summary)
                    if not args.keep_server:
                        try:
                            run_rec["unload"] = await mcp.call("unload_pipeline")
                        except ToolCallError as exc:
                            run_rec["unload"] = {"error": exc.text}
    finally:
        write_json(calls_path, calls)
    return code


# ---------------------------------------------------------------------------------------------
# 乾いた確認（--dry-run・--selfcheck）
# ---------------------------------------------------------------------------------------------

def dry_run(args, plan: dict) -> int:
    jobs = select_jobs(plan, args.stage, args.only)
    out: dict[str, Any] = {
        "port": args.port, "temp_config": str(RT / TEMP_CONFIG_NAME), "temp_state": str(RT / TEMP_STATE_NAME),
        "server_cmd": [str(PY), "-u", str(REPO / "main.py"), "--config", str(RT / TEMP_CONFIG_NAME)],
        "mcp_env": {"PYTHONUTF8": "1", "LTX_MCP_BASE_URL": base_url(args.port),
                    "LTX_MCP_CONFIG": str(RT / TEMP_CONFIG_NAME), "LTX_MCP_OUTPUT_DIR": str(RT / "outputs")},
        "load_pipeline": {"base_model": H3_BASE_MODEL, "models": {"transformer": "default"}, "wait_sec": 45},
        "copy_not_generated": {j["name"]: {"from": str((HERE / j["source"]).resolve()),
                                           "to": str(JOBS / j["name"])}
                               for j in plan["jobs"] if not j.get("generate", True)},
        "order": [j["name"] for j in jobs],
        "submit_generate": {j["name"]: {"dir": str(JOBS / j["name"]), "note": job_note(j),
                                        "args": build_args(plan, j)} for j in jobs},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def selfcheck(plan: dict) -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    gen = generated_jobs(plan)
    by_name = {j["name"]: j for j in gen}
    negs = {"N1": N1, "N2": N2}
    check("生成する本が 18 本", len(gen) == 18, f"{len(gen)} 本")
    check("名前と順が計画書 §3 の表と一致（A → B → R → C → D）",
          [j["name"] for j in gen] == [e[0] for e in EXPECTED_JOBS], f"{[j['name'] for j in gen]}")
    check("名前に重複が無い", len({j["name"] for j in plan["jobs"]}) == len(plan["jobs"]))
    check("段の順が A → B → R → C → D", list(plan.get("stage_order") or []) == list(STAGE_ORDER),
          str(plan.get("stage_order")))
    check("負の文の N1・N2 が計画書のとおり", plan.get("negatives") == negs, str(plan.get("negatives")))
    for name, stage, seed, w, h, nf, neg, scale in EXPECTED_JOBS:
        j = by_name.get(name)
        if j is None:
            check(f"{name}: 計画にある", False, "無い")
            continue
        problems = []
        if j["stage"] != stage:
            problems.append(f"段 {j['stage']}≠{stage}")
        if j["seed"] != seed:
            problems.append(f"seed {j['seed']}≠{seed}")
        if (j["width"], j["height"], j["num_frames"]) != (w, h, nf):
            problems.append(f"寸法 {j['width']}×{j['height']}×{j['num_frames']}≠{w}×{h}×{nf}")
        if j["width"] % 32 or j["height"] % 32:
            problems.append("寸法が 32 の倍数でない")
        if (j["num_frames"] - 5) % 17 or not (22 <= j["num_frames"] <= 481):
            problems.append("フレームが 17n+5（22〜481）でない")
        if j.get("negative") != negs[neg]:
            problems.append(f"負の文 {j.get('negative')!r}≠{neg}")
        if not isinstance(j.get("vsf_scale"), (int, float)) or float(j["vsf_scale"]) != float(scale):
            problems.append(f"scale {j.get('vsf_scale')}≠{scale}")
        elif not (0.0 <= float(j["vsf_scale"]) <= 10.0):
            problems.append("scale が API の範囲 0〜10 の外")
        a = build_args(plan, j)
        for k, v in EXPECTED_COMMON.items():
            if a.get(k) != v:
                problems.append(f"共通の引数 {k}={a.get(k)!r}≠{v!r}")
        if not (a.get("nag_enabled") is True and a.get("neg_method") == "vsf"
                and a.get("negative_prompt") == negs[neg] and a.get("vsf_scale") == float(scale)):
            problems.append("負の鍵（nag_enabled・neg_method・negative_prompt・vsf_scale）が揃わない")
        bad = [k for k in FORBIDDEN_KEYS if k in a]
        if bad:
            problems.append(f"送らないはずの鍵 {bad}")
        extra = set(a) - REQUIRED_TOOL_ARGS["submit_generate"]
        if extra:
            problems.append(f"照合の一覧に無い引数 {sorted(extra)}")
        check(f"{name}: 段 {stage}・seed {seed}・{w}×{h}×{nf}・{neg}・scale {scale:g}", not problems,
              "; ".join(problems))
    off = [j for j in plan["jobs"] if not j.get("generate", True)]
    check("生成しない本は A_off の 1 本", [j["name"] for j in off] == ["A_off"], str([j["name"] for j in off]))
    for j in off:
        src = (HERE / j["source"]).resolve()
        check(f"{j['name']}: 元のフォルダ {src} がある", src.is_dir())
        mp4s = sorted(src.glob("*.mp4")) if src.is_dir() else []
        check(f"{j['name']}: 元に mp4 が 1 本ある（{', '.join(p.name for p in mp4s)}）", len(mp4s) == 1)
        check(f"{j['name']}: 元に metadata.json がある", (src / "metadata.json").is_file())
        check(f"{j['name']}: 元に request.json がある", (src / "request.json").is_file())
        try:
            req = json.loads((src / "request.json").read_text(encoding="utf-8"))["arguments"]
            same = (all(req.get(k) == v for k, v in EXPECTED_COMMON.items())
                    and (req.get("width"), req.get("height"), req.get("num_frames"), req.get("seed"))
                    == (j["width"], j["height"], j["num_frames"], j["seed"])
                    and "negative_prompt" not in req and not req.get("nag_enabled")
                    and not any(k in req for k in FORBIDDEN_KEYS))
            check(f"{j['name']}: 元の引数が共通の条件と同じで、負の文なし", same)
        except Exception as exc:  # noqa: BLE001
            check(f"{j['name']}: 元の request.json を読める", False, f"{type(exc).__name__}: {exc}")
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  （{detail}）" if detail and not ok else ""))
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"--- {len(results) - failed}/{len(results)} PASS" + (f"・{failed} FAIL" if failed else ""))
    return 1 if failed else 0


# ---------------------------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        description="MiniMax H3 の VSF の実用域の較正 18 本（calib_plan.json）を MCP 経由・別ポートの実バックエンドで回す")
    ap.add_argument("--stage", choices=("A", "B", "R", "C", "D", "all"), default="all",
                    help="回す段（R＝計画書の A′〔seed 43〕・D＝長い尺）。all は A → B → R → C → D（既定 all）")
    ap.add_argument("--only", default=None, help="名前をカンマ区切りで（例: A_s0,A_s1.5）。--stage と重ねて絞る")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT, help=f"一時サーバーのポート（既定 {PORT_DEFAULT}）")
    ap.add_argument("--keep-server", action="store_true",
                    help="終わりに unload もサーバーの停止もしない（次の回で同じポートのサーバーを使い回す）")
    ap.add_argument("--load-timeout-sec", type=float, default=1200.0, help="H3 の読み込みを待つ上限（既定 1200）")
    ap.add_argument("--job-timeout-sec", type=float, default=3600.0,
                    help="1 本を待つ上限（既定 3600）。超えたら残りは送らない")
    ap.add_argument("--dry-run", action="store_true",
                    help="計画と送る引数を表示して終わる（サーバーも MCP も起こさない・何も写さない）")
    ap.add_argument("--selfcheck", action="store_true",
                    help="calib_plan.json を計画書 §3 の表と突き合わせ、A_off の元を確かめる（何も起こさない）")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args(argv)
    plan = load_plan()
    if args.selfcheck:
        return selfcheck(plan)
    if args.dry_run:
        return dry_run(args, plan)
    select_jobs(plan, args.stage, args.only)  # 名前の誤りはサーバーを起こす前に止める
    summary = load_summary()
    copy_a_off(plan, summary)
    run_rec: dict[str, Any] = {"started_at": now_iso(), "args": vars(args).copy()}
    summary["runs"].append(run_rec)
    flush(summary)
    srv = Server(args.port)
    code = 1
    try:
        srv.start()
        run_rec["server"] = srv.record
        flush(summary)
        code = asyncio.run(run(args, plan, summary, run_rec))
    except BaseException as exc:  # noqa: BLE001 -- 例外も中断も記録して終わる
        run_rec["error"] = f"{type(exc).__name__}: {exc}"
        run_rec["traceback"] = traceback.format_exc()
        log(f"止まった: {run_rec['error']}")
        code = 1
    finally:
        if args.keep_server:
            log("--keep-server: サーバーを残す（読み込んだ H3 もそのまま）")
        else:
            try:
                srv.stop()
            except Exception as exc:  # noqa: BLE001
                srv.record["stop_error"] = f"{type(exc).__name__}: {exc}"
        run_rec["server"] = srv.record
        run_rec["finished_at"] = now_iso()
        run_rec["exit_code"] = code
        flush(summary)
        log(f"記録: {HERE / 'summary.json'}（終了コード {code}）")
    return code


if __name__ == "__main__":
    sys.exit(main())
