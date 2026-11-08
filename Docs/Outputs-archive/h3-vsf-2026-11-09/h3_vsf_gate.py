"""MiniMax H3 のネガティブプロンプト（VSF）の機械の門 G2・G3 と目視の実験を、MCP サーバー経由で回す台本（台帳 §1-127）。

計画の正本: ``C:/Users/phenotype nezu/.claude/plans/agile-dazzling-catmull.md`` の §4（機械の門）と §5（目視の実験）。
製品コードは import しない。部品は次の 2 つの雛形から写した。

- ``Docs/Outputs-archive/h3-retake-2026-11-08/h3_retake_gate.py``: MCP の stdio クライアント（``Mcp``・
  ``check_tool_schemas``・``ensure_pipeline``・``wait_job``）と request.json／metadata.json の保存。
- ``outputs/ltx25-chain-g5/g5.py``: 一時 yaml（``write_temp_config``）と ``Server``（別ポートで実バックエンドを
  起こし、終わりに unload → 停止）。

流れ:
  1. ``config.yaml`` を読むだけで、``server.port``・``server.log_dir``・``output.dir``・``upload.dir``・``state_file``
     だけを ``_rt\\`` に向けた一時 yaml と、MiniMax H3 を選んだ一時 state を ``_rt\\`` に書く。
  2. ``.venv\\Scripts\\python.exe main.py --config <一時 yaml>`` で別ポート（既定 18699）に実バックエンドを起こし、
     ``GET /api/v1/status`` が 200 を返すまで待つ。
  3. MCP サーバー（``python -m mcp_server``）を子プロセスとして起こす。環境に ``PYTHONUTF8=1``・
     ``LTX_MCP_BASE_URL=http://127.0.0.1:<port>``（MCP の設定は ``/api/v1`` を含まない形。``mcp_server/settings.py``）・
     ``LTX_MCP_CONFIG=<一時 yaml>``・``LTX_MCP_OUTPUT_DIR=_rt\\outputs`` を渡す（後の 2 つが無いと MCP は
     ``config.yaml`` の出力先 ``outputs\\`` からジョブの動画を探してしまう）。
  4. ``load_pipeline(base_model="MiniMaxH3", models={"transformer": "default"}, wait_sec=45)`` を 1 回、
     以後 ``backend_status`` を 10 秒おき。
  5. ``--mode`` の門を回す（g2 → g3 → visual の順。all は 3 つとも）。
  6. ``--keep-server`` でなければ ``unload_pipeline`` → サーバーをプロセスの木ごと停止。
  7. ``summary.json`` を書く（前の回の結果に、ジョブの名前ごとに上書きで足す）。

終了コード: 0＝回した門がすべて合格（visual はすべて completed）、3＝どれかが不合格・失敗（先へは進む）、
1＝準備の段で止まった（サーバーが起きない・道具の引数名が違う・読み込みの失敗など）。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
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
PY = REPO / ".venv" / "Scripts" / "python.exe"
PORT_DEFAULT = 18699
TEMP_CONFIG_NAME = "config.h3vsf.yaml"
TEMP_STATE_NAME = "state.h3vsf.json"

H3_BASE_MODEL = "MiniMaxH3"
H3_CATEGORIES = ("transformer", "text_encoder", "video_vae", "audio")

#: G0 の指紋（VSF なし・640×384×124・seed 42・20 ステップ・shift 12/3・sdpa）。正本は記録簿
#: ``Docs/VERIFICATION_LOG.md`` §180.4 G-I5 と ``Docs/Outputs-archive/h3-te-vram-2026-10-29/g_i5_t2v_fingerprint.json``。
G0_FINGERPRINT = "0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9"
G0_PROMPT = ("A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles. "
             "She looks at the camera and says <d>[English] It finally stopped raining.</d>")

#: 目視の実験の文（計画書 §2 の表のとおり）。
VISUAL_PROMPT = ("A Japanese woman in casual clothes walks toward the camera along a city street and smiles, "
                 "reaching one hand toward the viewer; steady camera, natural daylight.")
VISUAL_NEGATIVE = "black hair, long hair, eyeglasses, cars"
TURBO_LORA = "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16"
TURBO_LORA_STRENGTH = 0.8
VISUAL_SCALES = (None, 1.5, 3.0, 5.0, 8.0)  # None＝off（nag なし）。番号 0〜4

VSF_KEYS = ("rows", "scale", "m_neg", "m_text")

TOOL_READ_TIMEOUT = timedelta(seconds=360)
WAIT_TIMEOUT_SEC = 45
WAIT_POLL_SEC = 5.0
MAX_CONSECUTIVE_TRANSPORT_ERRORS = 12

# tools/list の inputSchema で照合する引数名。台本が実際に渡す名前だけを並べる（無ければ止まる）。
REQUIRED_TOOL_ARGS: dict[str, set[str]] = {
    "backend_status": set(),
    "list_models": set(),
    "load_pipeline": {"base_model", "models", "wait_sec"},
    "unload_pipeline": set(),
    "submit_generate": {
        "prompt", "negative_prompt", "nag_enabled", "neg_method", "vsf_scale", "width", "height", "num_frames",
        "frame_rate", "seed", "loras", "attention_backend", "keep_resident", "embed_mp4_metadata",
        "num_inference_steps", "shift_video", "shift_audio", "vram16_mode",
    },
    "submit_chain": {"prompt", "clips", "negative_prompt", "nag_enabled", "neg_method", "vsf_scale", "width",
                     "height", "frame_rate", "seed", "attention_backend"},
    "wait_for_job": {"job_id", "timeout_sec", "poll_interval_sec"},
    "cancel_job": {"job_id"},
    "save_job_video": {"job_id", "dest_dir", "no_clobber", "which"},
    "get_job_video_path": {"job_id"},
}

_NAG_WORD = re.compile(r"(?<![A-Za-z_])nag(?![A-Za-z_])")


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


def mentions_nag(text: str) -> bool:
    """エラー文が機能名 ``nag`` を語として含むか（``nag_enabled`` の部分一致は数えない）。"""
    return bool(_NAG_WORD.search(text or ""))


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
        self._fh.write(f"\n===== {now_iso()} h3_vsf_gate.py: main.py --config {cfg} =====\n")
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
# 1 本のジョブ
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


async def expect_refusal(mcp: Mcp, tool: str, tool_args: dict, need_feature_unsupported: bool) -> dict:
    """道具が ToolCallError で断ることを確かめる。万一受理されたらジョブを取り消して待つ。"""
    rec: dict[str, Any] = {"tool": tool, "tool_args": tool_args}
    try:
        sub = await mcp.call(tool, tool_args)
    except ToolCallError as exc:
        rec["error_text"] = exc.text
        rec["has_feature_unsupported"] = "FEATURE_UNSUPPORTED" in exc.text
        rec["mentions_nag"] = mentions_nag(exc.text)
        rec["pass"] = rec["mentions_nag"] and (rec["has_feature_unsupported"] or not need_feature_unsupported)
        return rec
    rec["accepted_unexpectedly"] = sub
    rec["pass"] = False
    job_id = sub.get("job_id")
    if job_id:
        try:
            rec["cancel"] = await mcp.call("cancel_job", {"job_id": job_id})
        except ToolCallError as exc:
            rec["cancel"] = {"error": exc.text}
        waited = await wait_job(mcp, job_id, 1800)
        rec["cancel_wait_final_status"] = (waited.get("final") or {}).get("status")
    return rec


# ---------------------------------------------------------------------------------------------
# 引数の組み立て
# ---------------------------------------------------------------------------------------------

def g0_shape_args() -> dict:
    """G0 と同じ形（640×384×124・seed 42・20 ステップ・shift 12/3・sdpa・VRAM 16GB mode）。"""
    return {"prompt": G0_PROMPT, "width": 640, "height": 384, "num_frames": 124, "frame_rate": 24.0, "seed": 42,
            "num_inference_steps": 20, "shift_video": 12.0, "shift_audio": 3.0, "attention_backend": "sdpa",
            "vram16_mode": True}


def g2_nag_args() -> dict:
    a = {"prompt": G0_PROMPT, "width": 640, "height": 384, "num_frames": 124, "frame_rate": 24.0, "seed": 42,
         "nag_enabled": True, "negative_prompt": "blurry", "neg_method": "nag", "attention_backend": "sdpa",
         "vram16_mode": True}
    return a


def g2_vsf_args() -> dict:
    a = g2_nag_args()
    a.update(neg_method="vsf", vsf_scale=3.0, num_inference_steps=4)
    return a


def g2_chain_args() -> dict:
    return {"prompt": G0_PROMPT, "clips": [{"num_frames": 124}, {"num_frames": 124}], "width": 640, "height": 384,
            "frame_rate": 24.0, "seed": 42, "nag_enabled": True, "negative_prompt": "blurry", "neg_method": "vsf",
            "vsf_scale": 3.0, "attention_backend": "sdpa"}


def g3_vsf_args() -> dict:
    a = g0_shape_args()
    a.update(nag_enabled=True, negative_prompt="blurry, low quality", neg_method="vsf", vsf_scale=3.0,
             keep_resident=True, embed_mp4_metadata=False)
    return a


def g3_off_args() -> dict:
    a = g0_shape_args()
    a.update(keep_resident=True, embed_mp4_metadata=False)
    return a


def visual_plan(series: str) -> list[tuple[str, dict, str]]:
    """(名前, 引数, 説明) の並び。T 系列 → B 系列の順。"""
    out = []
    for s in (("T", "B") if series == "both" else (series,)):
        for i, scale in enumerate(VISUAL_SCALES):
            a: dict[str, Any] = {"prompt": VISUAL_PROMPT, "width": 1344, "height": 768, "num_frames": 56,
                                 "frame_rate": 24.0, "seed": 42, "attention_backend": "sage", "vram16_mode": True,
                                 "shift_video": 12.0, "shift_audio": 3.0}
            if s == "T":
                a["loras"] = [{"name": TURBO_LORA, "strength": TURBO_LORA_STRENGTH}]
                a["num_inference_steps"] = 4
                kind = f"Turbo LoRA {TURBO_LORA_STRENGTH}・4 ステップ"
            else:
                a["num_inference_steps"] = 20
                kind = "LoRA なし・20 ステップ"
            if scale is None:
                vs = "VSF off"
            else:
                a.update(nag_enabled=True, neg_method="vsf", negative_prompt=VISUAL_NEGATIVE, vsf_scale=scale)
                vs = f"VSF {scale:g}"
            out.append((f"{s}{i}", a, f"{kind}・{vs}"))
    return out


# ---------------------------------------------------------------------------------------------
# 門
# ---------------------------------------------------------------------------------------------

async def gate_g2(mcp: Mcp, args, summary: dict) -> bool:
    log("##### G2: 受理と拒否 #####")
    res: dict[str, Any] = {"started_at": now_iso(), "checks": {}}
    # ③ list_models の unsupported_features に nag が残る
    try:
        models = await mcp.call("list_models")
        h3 = next((b for b in models.get("base_models", []) if b.get("id") == H3_BASE_MODEL), None) or {}
        feats = h3.get("unsupported_features")
        res["checks"]["3_list_models"] = {"expected": "MiniMaxH3 の unsupported_features に 'nag' がある",
                                          "actual": feats, "pass": isinstance(feats, list) and "nag" in feats}
    except ToolCallError as exc:
        res["checks"]["3_list_models"] = {"error_text": exc.text, "pass": False}
    # ① neg_method="nag" は 422 FEATURE_UNSUPPORTED / nag
    c1 = await expect_refusal(mcp, "submit_generate", g2_nag_args(), need_feature_unsupported=True)
    c1["expected"] = "ToolCallError・文に FEATURE_UNSUPPORTED と nag"
    res["checks"]["1_generate_nag"] = c1
    # ④ submit_chain は vsf でも nag で断る
    c4 = await expect_refusal(mcp, "submit_chain", g2_chain_args(), need_feature_unsupported=True)
    c4["expected"] = "ToolCallError・文に FEATURE_UNSUPPORTED と nag（Chained は方式に関係なく断る）"
    res["checks"]["4_chain_vsf"] = c4
    # ② neg_method="vsf" は受理して completed、metadata に h3.vsf の 4 鍵
    job = await run_job(mcp, args, "g2_vsf", HERE / "g2" / "vsf", g2_vsf_args(),
                        "G2 ②: VSF 3.0・4 ステップ（受理されて completed になること）")
    summary["jobs"]["g2_vsf"] = job
    vsf = job.get("vsf")
    keys_ok = isinstance(vsf, dict) and all(k in vsf for k in VSF_KEYS)
    res["checks"]["2_generate_vsf"] = {"expected": f"completed・metadata の h3.vsf に {list(VSF_KEYS)}",
                                       "status": job.get("status"), "error": job.get("error"), "vsf": vsf,
                                       "vsf_where": job.get("vsf_where"),
                                       "pass": job.get("status") == "completed" and keys_ok}
    res["pass"] = all(c.get("pass") for c in res["checks"].values())
    res["finished_at"] = now_iso()
    write_json(HERE / "g2" / "result.json", res)
    summary["gates"]["g2"] = {"pass": res["pass"], "result": str(HERE / "g2" / "result.json"),
                              "checks": {k: v.get("pass") for k, v in res["checks"].items()}}
    log(f"G2: {'合格' if res['pass'] else '不合格'} {summary['gates']['g2']['checks']}")
    return res["pass"]


async def gate_g3(mcp: Mcp, args, summary: dict) -> bool:
    log("##### G3: VSF の直後の off に持ち越しが無いこと #####")
    on = await run_job(mcp, args, "g3_vsf", HERE / "g3" / "vsf", g3_vsf_args(),
                       "G3 ①: VSF 3.0・keep_resident（G0 と同じ形）")
    summary["jobs"]["g3_vsf"] = on
    off = await run_job(mcp, args, "g3_off", HERE / "g3" / "off", g3_off_args(),
                        "G3 ②: ①の直後に VSF なし（指紋が G0 と同じこと）")
    summary["jobs"]["g3_off"] = off
    vsf_on = on.get("vsf")
    checks = {
        "vsf_job_completed": on.get("status") == "completed",
        "vsf_job_has_vsf_keys": isinstance(vsf_on, dict) and all(k in vsf_on for k in VSF_KEYS),
        "vsf_job_log_neg_vsf": any("neg=vsf" in ln for ln in on.get("worker_log_lines") or []),
        "off_job_completed": off.get("status") == "completed",
        "off_job_sha_is_g0": (off.get("sha256") or "") == G0_FINGERPRINT,
        "off_job_has_no_vsf_key": off.get("status") == "completed" and off.get("vsf_where") is None,
        "off_job_log_neg_off": any("neg=off" in ln for ln in off.get("worker_log_lines") or []),
    }
    res = {"expected_off_sha256": G0_FINGERPRINT, "vsf_job": {"sha256": on.get("sha256"), "vsf": vsf_on,
                                                              "log_lines": on.get("worker_log_lines")},
           "off_job": {"sha256": off.get("sha256"), "vsf_where": off.get("vsf_where"),
                       "log_lines": off.get("worker_log_lines")},
           "checks": checks, "pass": all(checks.values()), "at": now_iso()}
    write_json(HERE / "g3" / "result.json", res)
    summary["gates"]["g3"] = {"pass": res["pass"], "result": str(HERE / "g3" / "result.json"), "checks": checks}
    log(f"G3: {'合格' if res['pass'] else '不合格'} {checks}")
    return res["pass"]


async def run_visual(mcp: Mcp, args, summary: dict) -> bool:
    log(f"##### 目視の実験（系列 {args.series}） #####")
    ok = True
    for label, tool_args, note in visual_plan(args.series):
        try:
            rec = await run_job(mcp, args, label, HERE / "visual" / label, tool_args, note)
        except Exception as exc:  # noqa: BLE001 -- 1 本の例外で残りを止めない
            rec = {"label": label, "status": "script_error", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(), "tool_args": tool_args}
        rec["series"] = label[0]
        rec["scale"] = tool_args.get("vsf_scale") if tool_args.get("nag_enabled") else None
        summary["jobs"][label] = rec
        flush(summary)
        if rec.get("status") != "completed" or rec.get("error"):
            ok = False
        if rec.get("status") == "wait_gave_up":
            log("待ちきれなかったジョブがバックエンドで走っているため、残りは送らない（JOB_BUSY になる）")
            break
    summary["gates"][f"visual_{args.series}"] = {"pass": ok, "at": now_iso()}
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
                data.setdefault("gates", {})
                data.setdefault("runs", [])
                return data
        except Exception:  # noqa: BLE001
            pass
    return {"script": "h3_vsf_gate.py", "plan": "agile-dazzling-catmull.md §4・§5", "jobs": {}, "gates": {}, "runs": []}


def flush(summary: dict) -> None:
    write_json(HERE / "summary.json", summary)


async def run(args, summary: dict, run_rec: dict) -> int:
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
            errlog.write(f"\n===== {now_iso()} h3_vsf_gate.py --mode {args.mode} =====\n")
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
                    modes = ["g2", "g3", "visual"] if args.mode == "all" else [args.mode]
                    for m in modes:
                        try:
                            if m == "g2":
                                ok = await gate_g2(mcp, args, summary)
                            elif m == "g3":
                                ok = await gate_g3(mcp, args, summary)
                            else:
                                ok = await run_visual(mcp, args, summary)
                        except Exception as exc:  # noqa: BLE001 -- 1 つの門の例外で残りを止めない
                            summary["gates"][m] = {"pass": False, "error": f"{type(exc).__name__}: {exc}",
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


def dry_run(args) -> int:
    plan: dict[str, Any] = {"port": args.port, "temp_config": str(RT / TEMP_CONFIG_NAME),
                            "server_cmd": [str(PY), "-u", str(REPO / "main.py"), "--config",
                                           str(RT / TEMP_CONFIG_NAME)],
                            "mcp_env": {"PYTHONUTF8": "1", "LTX_MCP_BASE_URL": base_url(args.port),
                                        "LTX_MCP_CONFIG": str(RT / TEMP_CONFIG_NAME),
                                        "LTX_MCP_OUTPUT_DIR": str(RT / "outputs")},
                            "load_pipeline": {"base_model": H3_BASE_MODEL, "models": {"transformer": "default"},
                                              "wait_sec": 45}}
    modes = ["g2", "g3", "visual"] if args.mode == "all" else [args.mode]
    if "g2" in modes:
        plan["g2"] = {"3_list_models": "list_models", "1_generate_nag": g2_nag_args(),
                      "4_chain_vsf": g2_chain_args(), "2_generate_vsf": g2_vsf_args()}
    if "g3" in modes:
        plan["g3"] = {"vsf": g3_vsf_args(), "off": g3_off_args(), "expected_off_sha256": G0_FINGERPRINT}
    if "visual" in modes:
        plan["visual"] = {label: {"note": note, "args": a} for label, a, note in visual_plan(args.series)}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="MiniMax H3 の VSF の門 G2・G3 と目視の実験（MCP 経由・別ポートの実バックエンド）")
    ap.add_argument("--mode", choices=("g2", "g3", "visual", "all"), required=True)
    ap.add_argument("--port", type=int, default=PORT_DEFAULT, help=f"一時サーバーのポート（既定 {PORT_DEFAULT}）")
    ap.add_argument("--keep-server", action="store_true",
                    help="終わりに unload もサーバーの停止もしない（次の回で同じポートのサーバーを使い回す）")
    ap.add_argument("--series", choices=("T", "B", "both"), default="both",
                    help="visual の系列（T＝Turbo 4 ステップ・B＝素の 20 ステップ・both＝T → B。既定 both）")
    ap.add_argument("--load-timeout-sec", type=float, default=1200.0)
    ap.add_argument("--job-timeout-sec", type=float, default=3600.0)
    ap.add_argument("--dry-run", action="store_true", help="送る引数を表示して終わる（サーバーも MCP も起こさない）")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    summary = load_summary()
    run_rec: dict[str, Any] = {"started_at": now_iso(), "args": vars(args).copy()}
    summary["runs"].append(run_rec)
    srv = Server(args.port)
    code = 1
    try:
        srv.start()
        run_rec["server"] = srv.record
        flush(summary)
        code = asyncio.run(run(args, summary, run_rec))
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
