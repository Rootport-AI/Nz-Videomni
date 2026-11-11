r"""MiniMax H3 の Inpainting（台帳 §1-124）弾 1 の GPU の門 I1・I2 を、API 経由（MCP サーバー → バックエンド →
エンジン）で回す台本。

計画: ``enumerated-questing-cupcake.md`` §6.2（合否 1〜7）・§4.1・§4.7・§4.9・§4.10。製品コードは変えない（読むだけ）。
部品は前例の台本から写した:
  - 一時 yaml・一時 state・別ポートのサーバーの起動と停止、MCP の stdio クライアント・引数名の照合・
    読み込みの待ち・ジョブの待ち: ``Docs/Outputs-archive/h3-outpaint-2026-11-10/h3_outpaint_gate_api.py``
  - PSNR（全フレームまとめた MSE）と、対照（切り出しを ``engine_h3.media_out.write_mp4`` で 1 回書き直したもの）を
    エンジンの仮想環境の子プロセス（GPU を隠す）で作る作法: 同じフォルダの ``h3_outpaint_gate.py`` と
    ``h3_outpaint_gate_api.py`` の ``measure_helper_main``
  - マスクの格子化・sheet・升目の重ね描き・マスクの中の平均絶対差:
    ``outputs/h3-inpaint-spike-2026-11-11/h3_inpaint_spike.py``
  - 幾何・塊・なじませの重み: 製品の ``engine_h3/inpaint_geometry.py``（numpy だけ。import する）、
    マスクの読み込み: 製品の ``engine_h3/inpaint_h3.read_inpaint_mask``（測定の子プロセスだけが import する）

使い方（リポジトリの根で。アプリの仮想環境の python。Git Bash）:

    export PYTHONUTF8=1
    .venv/Scripts/python.exe Docs/Outputs-archive/h3-inpaint-2026-11-11/h3_inpaint_gate_api.py --gate all --dry-run
    .venv/Scripts/python.exe Docs/Outputs-archive/h3-inpaint-2026-11-11/h3_inpaint_gate_api.py --gate all
    .venv/Scripts/python.exe Docs/Outputs-archive/h3-inpaint-2026-11-11/h3_inpaint_gate_api.py --gate all --post-only

流れ（本番）:
  1. 前の確かめ（ログに残す）: オーナーのサーバー（18620）が無応答か・GPU の使用量（nvidia-smi）・
     ``engine_h3.worker`` などのワーカーの残り。18620 が応答するかワーカーが残っていれば止まる（GPU の取り合いを避ける）。
  2. ``config.yaml`` は読むだけで、5 つの鍵を ``outputs/h3-inpaint-gates-2026-11-12/_rt/`` に向けた一時 yaml と、
     MiniMax H3 を選んだ一時 state を書き、別ポート（既定 18699）で実バックエンドを起こす（``backend: auto`` のまま）。
  3. MCP サーバー（``.venv/Scripts/python.exe -m mcp_server``）を stdio の子プロセスで起こし、
     ``initialize → tools/list`` で引数名（``inpaint_*`` の 4 つを含む）を照合してから ``tools/call``。
  4. ``backend_status`` → ``load_pipeline(MiniMaxH3, transformer=default, wait_sec=45)`` → 本ごとに
     ``upload_video``（元動画とマスクをそのまま）→ ``submit_generate``（``reference_video_id``＝元動画・
     ``inpaint_mask_video_id``・``inpaint_window_start_sec=0.0``）→ ``wait_for_job`` → ``save_job_video``
     （``<本>/output.mp4``）→ ジョブのフォルダから ``metadata.json``・``_inpaint_window.mp4``・``_inpaint_mask.mp4`` を写す。
  5. ``unload_pipeline`` → サーバーをプロセスの木ごと止める（子のワーカーも）。
  6. 後処理（``--post-only`` と同じ）: 出力の ffprobe・``metadata.inpaint`` の検査・芯の PSNR・マスクの中の平均絶対差・
     I2 の下端・塊の境目・sheet・升目の図 → ``<本>/result.json``、``hf-round15/``、``summary.json``・``summary.md``。

終了コード: 0＝回した本がすべて completed で後処理まで済んだ（合否は記録に書くだけで終了コードに入れない）、
1＝前の確かめ・準備で止まった、2＝引数の誤り、3＝ジョブの失敗・送信の拒否・待ちきれない・測定の失敗・出力が無い。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import statistics
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
REPO = HERE.parents[2]  # Docs/Outputs-archive/<このフォルダ> の 3 つ上＝リポジトリ直下
PY = REPO / ".venv" / "Scripts" / "python.exe"
ENGINE_PY = REPO / ".venv-engine-h3" / "Scripts" / "python.exe"
OUT_ROOT = REPO / "outputs" / "h3-inpaint-gates-2026-11-12"
RT = OUT_ROOT / "_rt"
LOG_DIR = OUT_ROOT / "logs"
HF_DIR = OUT_ROOT / "hf-round15"
HF_NAME = "round15-h3-inpaint"
PORT_DEFAULT = 18699
OWNER_PORT = 18620
TEMP_CONFIG_NAME = "config.h3inpaintgate.yaml"
TEMP_STATE_NAME = "state.h3inpaintgate.json"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"

H3_BASE_MODEL = "MiniMaxH3"
H3_CATEGORIES = ("transformer", "text_encoder", "video_vae", "audio")
H3_FPS = 24
NUM_FRAMES = 73
SEED = 1234
ATTENTION = "sage"
WINDOW_START_SEC = 0.0
BLEND_DILATION_DEFAULT = 1  # サーバーの既定（REQUEST_RULES の inpaint_blend_dilation）。送らない。feather_px の期待値に使う
#: スパイクと同じ文（outputs/h3-inpaint-spike-2026-11-11/h3_inpaint_spike.py の DEFAULT_PROMPT）
DEFAULT_PROMPT = (
    "Four young girls walking toward the camera on a sunny beach. The girl on the far right has blonde hair "
    "and wears a red dress, red outfit. Steady camera."
)

DEV = REPO / "uploads" / "h3-inpaint-development"
GATES: dict[str, dict[str, Any]] = {
    "I1": {"source": DEV / "inpaint-base.mp4", "mask": DEV / "mask-768p.mp4", "source_size": (1280, 768),
           "expect_canvas": (1280, 768), "expect_pad_bottom": 0, "expect_feather_px": 20,
           "purpose": "768p（キャンバス＝元の実寸。pad 0）"},
    "I2": {"source": DEV / "inpaint-base-1080p.mp4", "mask": DEV / "mask-1080p.mp4", "source_size": (1920, 1080),
           "expect_canvas": (1920, 1088), "expect_pad_bottom": 8, "expect_feather_px": 30,
           "purpose": "1080p（キャンバス 1920×1088。下に 8 画素の pad を足して生成し、切り戻す）"},
}
SOURCE_FRAMES_EXPECTED = 124
MASK_FRAMES_EXPECTED = 121

TOOL_READ_TIMEOUT = timedelta(seconds=360)
WAIT_TIMEOUT_SEC = 45
WAIT_POLL_SEC = 5.0
MAX_CONSECUTIVE_TRANSPORT_ERRORS = 12
UPLOAD_MAX_FRAMES = 100000  # 尺を測るためだけの値（素材の枚数より確実に大きい。切り詰めは起きない）

# 合否の基準（計画 §6.2）
PSNR_MARGIN_DB = 1.0
MASK_MAD_MIN = 10.0
BOUNDARY_RATIO_MAX = 2.0
BOTTOM_ROWS = 8
JUDGED_BOUNDARIES = ((16, 17), (33, 34))
SHEET_FRAMES = (0, 36, 72)
SHEET_TILE_LONG = 640
OVERLAY_FRAME = 36

#: metadata.inpaint に在るはずの鍵。エンジンの記録（engine_h3/inpaint_h3.inpaint_record）と
#: アプリの provenance（services/pipeline_manager.py の H3 の枝）。
ENGINE_KEYS = ("canvas", "pads", "source", "mask_rule", "chunk_frames", "chunks", "latent_frames",
               "fixed_tokens_total", "generated_tokens_total", "generated_per_latent_frame", "audio_strength",
               "source_had_audio", "audio_latents_encoded", "audio_output", "feather_px", "inject_clean",
               "self_check", "mask_proof", "cropped_to")
PROVENANCE_KEYS = ("source_video_id", "mask_video_id", "source_fps", "mask_fps", "resampled", "mask_resampled",
                   "window_start_sec", "window_start_frame", "window_written_frames", "mask_written_frames",
                   "window_has_audio")

# tools/list の inputSchema で照合する引数名（台本が実際に渡す名前と、計画 §4.7 の inpaint の 4 引数）。無ければ止まる。
REQUIRED_TOOL_ARGS: dict[str, set[str]] = {
    "backend_status": set(),
    "load_pipeline": {"base_model", "models", "wait_sec"},
    "unload_pipeline": set(),
    "list_models": set(),
    "upload_video": {"file_path", "max_frames"},
    "submit_generate": {
        "prompt", "width", "height", "num_frames", "frame_rate", "seed", "attention_backend",
        "reference_video_id", "inpaint_mask_video_id", "inpaint_audio_strength",
        "inpaint_blend_dilation_stage1", "inpaint_window_start_sec",
    },
    "wait_for_job": {"job_id", "timeout_sec", "poll_interval_sec"},
    "save_job_video": {"job_id", "dest_dir", "filename", "no_clobber", "which"},
    "get_job_video_path": {"job_id"},
}

NOT_SENT = ["num_inference_steps（既定 20）", "shift_video／shift_audio（既定 12／3）",
            "inpaint_audio_strength（既定 0.5）", "inpaint_blend_dilation_stage1（既定 1）",
            "vram16_mode（既定 on）", "block_swap_prefetch（既定 on）", "keep_resident（既定 off）", "loras（なし）"]


# ---------------------------------------------------------------------------------------------
# 小道具
# ---------------------------------------------------------------------------------------------

class Log:
    """標準出力と、その回のログファイルへ同じ行を書く。"""

    def __init__(self) -> None:
        self.file = None
        self.path: Path | None = None

    def open(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = open(path, "a", encoding="utf-8", newline="\n")

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


log = Log()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


def tool(name: str) -> str:
    local = REPO / "tools" / "ffmpeg" / "bin" / f"{name}.exe"
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} が見つかりません（<repo>/tools/ffmpeg/bin か PATH）")
    return found


def run_cmd(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def ffprobe(path: Path) -> dict:
    """映像（枚数は -count_frames の実測）と音声の要点（写し元と同じ形）。"""
    exe = tool("ffprobe")
    v = run_cmd([exe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                 "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration",
                 "-of", "json", str(path)])
    a = run_cmd([exe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                 "stream=codec_name,sample_rate,channels,duration", "-of", "json", str(path)])
    vs = (json.loads(v.stdout or "{}").get("streams") or [{}])[0]
    as_ = (json.loads(a.stdout or "{}").get("streams") or [None])[0]
    nb = vs.get("nb_read_frames")
    fps = None
    if vs.get("r_frame_rate"):
        num, _, den = str(vs["r_frame_rate"]).partition("/")
        try:
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            fps = None
    return {"path": str(path), "video": {**vs, "fps": fps, "nb_read_frames": int(nb) if nb not in (None, "N/A") else None},
            "audio": as_}


def geometry_module():
    """製品の engine_h3/inpaint_geometry.py（numpy だけ）。"""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from engine_h3 import inpaint_geometry as IG  # noqa: PLC0415

    return IG


def feather_px_of(r: int, w: int, h: int) -> dict:
    """なじませの画素数。製品の ``services/engines/h3/adapter._feather_px`` を呼べればその値、
    呼べなくても同じ式 ``floor(r·max(W,H)/64+0.5)`` の値を並べて記録する。"""
    import math  # noqa: PLC0415

    formula = int(math.floor(int(r) * max(int(w), int(h)) / 64 + 0.5))
    out: dict[str, Any] = {"formula": formula}
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from services.engines.h3.adapter import _feather_px  # noqa: PLC0415

        out["adapter"] = int(_feather_px(r, w, h))
    except Exception as exc:  # noqa: BLE001
        out["adapter_error"] = f"{type(exc).__name__}: {exc}"
    out["value"] = out.get("adapter", formula)
    return out


# ---------------------------------------------------------------------------------------------
# 前の確かめ（オーナーのサーバー・GPU・ワーカーの残り）
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


def all_processes() -> list[dict]:
    ps = ("Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
          "ConvertTo-Json -Depth 3")
    r = run_cmd(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=120)
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


WORKER_RE = re.compile(r"-m\s+engine\w*\.worker|engine\w*[\\/]worker\.py")


def preflight_checks() -> dict:
    """18620 の応答・GPU の使用量・ワーカーの残り。``blocking`` が空でなければ本番は止まる。"""
    rec: dict[str, Any] = {"at": now_iso(), "blocking": []}
    try:
        status, body = http(OWNER_PORT, "GET", "/status", timeout=3)
        rec["owner_server"] = {"port": OWNER_PORT, "responded": True, "http": status,
                               "body": body if isinstance(body, dict) else str(body)[:300]}
        rec["blocking"].append(f"オーナーのサーバー（{OWNER_PORT}）が応答している（HTTP {status}）")
    except Exception as exc:  # noqa: BLE001
        rec["owner_server"] = {"port": OWNER_PORT, "responded": False, "error": f"{type(exc).__name__}: {exc}"}
    q = run_cmd(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"], timeout=60)
    apps = run_cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits"], timeout=60)
    rec["gpu"] = {"returncode": q.returncode, "query": q.stdout.strip(), "compute_apps": apps.stdout.strip(),
                  "stderr": (q.stderr or "").strip()[-300:]}
    procs = all_processes()
    workers = [{"pid": p["pid"], "ppid": p["ppid"], "cmdline": p["cmdline"][:300]}
               for p in procs if WORKER_RE.search(p["cmdline"] or "")]
    alive = {p["pid"] for p in procs}
    for w in workers:
        w["orphan"] = w["ppid"] not in alive
    rec["engine_workers"] = workers
    if workers:
        rec["blocking"].append(f"ワーカーのプロセスが残っている（{[w['pid'] for w in workers]}）")
    rec["process_count"] = len(procs)
    return rec


def log_preflight(rec: dict) -> None:
    o = rec["owner_server"]
    log(f"前の確かめ: {OWNER_PORT} は {'応答あり（HTTP ' + str(o.get('http')) + '）' if o['responded'] else '無応答'}")
    log(f"前の確かめ: GPU {rec['gpu']['query'] or '（nvidia-smi の値なし）'}（名前, 使用 MiB, 全体 MiB, 使用率 %）")
    apps = [ln for ln in (rec["gpu"]["compute_apps"] or "").splitlines() if ln.strip()]
    py = [ln for ln in apps if "python" in ln.lower()]
    log(f"前の確かめ: GPU を使うプロセス {len(apps)} 個（うち python {py or 'なし'}。全件は記録の preflight.gpu）")
    log(f"前の確かめ: ワーカーの残り {rec['engine_workers'] or 'なし'}")


# ---------------------------------------------------------------------------------------------
# 本の組み立て
# ---------------------------------------------------------------------------------------------

def gate_list(arg: str) -> list[str]:
    return list(GATES) if arg == "all" else [arg]


def build_job(gate: str, prompt: str) -> dict:
    IG = geometry_module()
    g = GATES[gate]
    sw, sh = g["source_size"]
    cw, ch = IG.round_up_32(sw), IG.round_up_32(sh)
    if (cw, ch) != g["expect_canvas"]:
        raise SystemExit(f"{gate}: round_up_32 のキャンバス {cw}x{ch} が想定 {g['expect_canvas']} と違う")
    geom = IG.compute_inpaint_geometry(sw, sh, cw, ch)
    fpx = feather_px_of(BLEND_DILATION_DEFAULT, cw, ch)
    tool_args = {
        "prompt": prompt, "width": cw, "height": ch, "num_frames": NUM_FRAMES, "frame_rate": float(H3_FPS),
        "seed": SEED, "attention_backend": ATTENTION, "inpaint_window_start_sec": WINDOW_START_SEC,
        # reference_video_id・inpaint_mask_video_id は送る直前に入れる（upload_video の ID）
    }
    return {
        "gate": gate, "label": gate, "purpose": g["purpose"], "frames": NUM_FRAMES,
        "source": str(g["source"]), "mask": str(g["mask"]), "source_size": [sw, sh], "canvas": [cw, ch],
        "geometry": {"pad_left": geom.pad_left, "pad_right": geom.pad_right, "pad_top": geom.pad_top,
                     "pad_bottom": geom.pad_bottom, "grid": [geom.grid_w, geom.grid_h]},
        "chunks": [list(c) for c in IG.chunk_ranges(NUM_FRAMES)],
        "feather_px": fpx, "expect_feather_px": g["expect_feather_px"], "tool_args": tool_args,
    }


def material_check(job: dict) -> dict:
    """素材 2 本の ffprobe と、受付の規則（寸法の一致・24fps 換算の枚数・音声）に照らした確かめ。"""
    src = ffprobe(Path(job["source"]))
    msk = ffprobe(Path(job["mask"]))
    sv, mv = src["video"], msk["video"]
    sw, sh = job["source_size"]
    mfps = mv.get("fps") or 0.0
    mask_frames_24 = int(round((mv.get("nb_read_frames") or 0) * H3_FPS / mfps)) if mfps else None
    problems = []
    if (sv.get("width"), sv.get("height")) != (sw, sh):
        problems.append(f"元動画の寸法 {sv.get('width')}x{sv.get('height')} が {sw}x{sh} ではない")
    if sv.get("nb_read_frames") != SOURCE_FRAMES_EXPECTED:
        problems.append(f"元動画の枚数 {sv.get('nb_read_frames')} が {SOURCE_FRAMES_EXPECTED} ではない")
    if not src.get("audio"):
        problems.append("元動画に音声が無い（音声の強さ 0.5 の既定では 422 になる）")
    if (mv.get("width"), mv.get("height")) != (sw, sh):
        problems.append(f"マスクの寸法 {mv.get('width')}x{mv.get('height')} が元動画 {sw}x{sh} と違う")
    if mv.get("nb_read_frames") != MASK_FRAMES_EXPECTED:
        problems.append(f"マスクの枚数 {mv.get('nb_read_frames')} が {MASK_FRAMES_EXPECTED} ではない")
    if mask_frames_24 is None or mask_frames_24 < NUM_FRAMES:
        problems.append(f"マスクの 24fps 換算の枚数 {mask_frames_24} が {NUM_FRAMES} 未満")
    src_secs = (sv.get("nb_read_frames") or 0) / (sv.get("fps") or H3_FPS)
    if src_secs + 1e-6 < WINDOW_START_SEC + NUM_FRAMES / H3_FPS:
        problems.append(f"元動画の尺 {src_secs:.3f} 秒が窓 {NUM_FRAMES / H3_FPS:.3f} 秒に足りない")
    return {"source_probe": src, "mask_probe": msk, "mask_frames_at_24fps": mask_frames_24,
            "problems": problems, "ok": not problems}


# ---------------------------------------------------------------------------------------------
# 一時 yaml とサーバー（写し元の写し。置き場だけ outputs/h3-inpaint-gates-2026-11-12/_rt）
# ---------------------------------------------------------------------------------------------

def write_temp_config(port: int) -> tuple[Path, Path]:
    """``config.yaml`` は読むだけ。5 つの鍵だけを ``_rt`` に向けた一時 yaml と、H3 を選んだ一時 state を書く。"""
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
    write_text(cfg, yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
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
    write_text(state, json.dumps({"schema": 1, "active_base_model": H3_BASE_MODEL, "selections": selections},
                                 indent=2))
    return cfg, state


def server_cmd() -> list[str]:
    return [str(PY), "-u", str(REPO / "main.py"), "--config", str(RT / TEMP_CONFIG_NAME)]


class Server:
    def __init__(self, port: int) -> None:
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.log_path = RT / "server.console.log"
        self._fh = None
        self.record: dict[str, Any] = {"port": port}

    def alive(self) -> bool:
        try:
            status, _ = http(self.port, "GET", "/status", timeout=5)
            return status == 200
        except Exception:  # noqa: BLE001
            return False

    def start(self) -> None:
        if self.alive():
            raise RuntimeError(f"ポート {self.port} で別のサーバーが応答している。--port で別のポートを選ぶか、"
                               "そのサーバーを止めてから回す")
        cfg, state = write_temp_config(self.port)
        env = dict(os.environ)
        # run.ps1 と同じ環境（tools/uv と tools/ffmpeg/bin を PATH の先頭へ・uv の python の置き場）。
        prefix = ""
        for d in (REPO / "tools" / "uv", REPO / "tools" / "ffmpeg" / "bin"):
            if d.exists():
                prefix += f"{d};"
        env["PATH"] = prefix + env.get("PATH", "")
        env["UV_PYTHON_INSTALL_DIR"] = str(REPO / ".python")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        RT.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8", errors="replace", newline="\n")
        self._fh.write(f"\n===== {now_iso()} h3_inpaint_gate_api.py: main.py --config {cfg} =====\n")
        self._fh.flush()
        cmd = server_cmd()
        self.proc = subprocess.Popen(cmd, cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=self._fh,
                                     stderr=subprocess.STDOUT, env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.record.update(pid=self.proc.pid, cmd=cmd, config=str(cfg), state=str(state),
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
        if self.proc is None:
            return
        rec: dict[str, Any] = {"at": now_iso()}
        try:
            rec["unload_http"] = http(self.port, "POST", "/pipeline/unload", timeout=600)[0]
        except Exception as exc:  # noqa: BLE001
            rec["unload_http"] = f"{type(exc).__name__}: {exc}"
        pid = self.proc.pid
        kids = descendants(pid, all_processes())
        rec["descendants_before"] = [{"pid": p["pid"], "cmdline": p["cmdline"][:300]} for p in kids]
        try:
            self.proc.terminate()
            self.proc.wait(timeout=60)
            rec["terminate"] = "ok"
        except Exception as exc:  # noqa: BLE001
            rec["terminate"] = f"{type(exc).__name__}: {exc}"
            rec["taskkill"] = run_cmd(["taskkill", "/PID", str(pid), "/T", "/F"]).stdout.strip()
        self.proc = None
        alive_pids = {p["pid"] for p in all_processes()}
        left = [p for p in kids if p["pid"] in alive_pids]
        for p in left:
            run_cmd(["taskkill", "/PID", str(p["pid"]), "/T", "/F"])
        rec["descendants_killed_after"] = [p["pid"] for p in left]
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._fh = None
        self.record["stop"] = rec
        log(f"サーバーを止めた（pid {pid}・残っていた子孫 {rec['descendants_killed_after']}）")


def file_offset(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def read_since(p: Path, offset: int) -> str:
    if not p.exists():
        return ""
    with p.open("rb") as fh:
        fh.seek(offset)
        return fh.read().decode("utf-8", errors="replace")


def worker_log_path() -> Path:
    p = RT / "logs" / "h3_worker.log"
    if p.exists():
        return p
    found = sorted((RT / "logs").glob("*h3*worker*.log"))
    return found[0] if found else p


# ---------------------------------------------------------------------------------------------
# MCP の呼び出し（写し元の写し）
# ---------------------------------------------------------------------------------------------

class ToolCallError(Exception):
    def __init__(self, tool_name: str, text: str) -> None:
        super().__init__(f"{tool_name}: {text}")
        self.tool = tool_name
        self.text = text


class Mcp:
    def __init__(self, session, calls_log: list) -> None:
        self.session = session
        self.calls_log = calls_log

    async def call(self, tool_name: str, args: dict[str, Any] | None = None) -> dict:
        t0 = time.monotonic()
        entry: dict[str, Any] = {"at": now_iso(), "tool": tool_name, "args": args or {}}
        try:
            result = await self.session.call_tool(tool_name, args or {}, read_timeout_seconds=TOOL_READ_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 -- 転送の失敗も全文で残す
            entry.update(ok=False, seconds=round(time.monotonic() - t0, 2),
                         error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            self.calls_log.append(entry)
            raise ToolCallError(tool_name, entry["error"]) from exc
        entry["seconds"] = round(time.monotonic() - t0, 2)
        text = "\n".join(getattr(c, "text", "") for c in (result.content or []) if getattr(c, "text", None))
        if result.isError:
            entry.update(ok=False, error=text)
            self.calls_log.append(entry)
            raise ToolCallError(tool_name, text)
        if isinstance(result.structuredContent, dict):
            data = result.structuredContent
        else:
            try:
                data = json.loads(text) if text else {}
            except json.JSONDecodeError:
                data = {"_text": text}
        entry["ok"] = True
        if tool_name not in ("wait_for_job", "backend_status"):
            entry["result"] = data
        self.calls_log.append(entry)
        return data


def check_tool_schemas(tools: list) -> dict:
    by_name = {t.name: t for t in tools}
    report: dict[str, Any] = {"tool_count": len(tools), "problems": []}
    for name, needed in REQUIRED_TOOL_ARGS.items():
        t = by_name.get(name)
        if t is None:
            report["problems"].append(f"道具 {name} が tools/list に無い")
            continue
        props = set((t.inputSchema or {}).get("properties", {}))
        missing = needed - props
        if missing:
            report["problems"].append(f"{name} の inputSchema に引数 {sorted(missing)} が無い（あるのは {sorted(props)}）")
    sg = by_name.get("submit_generate")
    if sg is not None:
        report["submit_generate_inpaint_args"] = sorted(
            k for k in (sg.inputSchema or {}).get("properties", {}) if k.startswith("inpaint_"))
    report["ok"] = not report["problems"]
    return report


def mcp_env(port: int, with_config: bool = True) -> dict:
    env = {"PYTHONUTF8": "1", "LTX_MCP_BASE_URL": base_url(port), "LTX_MCP_OUTPUT_DIR": str(RT / "outputs")}
    if with_config:
        env["LTX_MCP_CONFIG"] = str(RT / TEMP_CONFIG_NAME)
    return env


def mcp_params(port: int, with_config: bool = True):
    from mcp import StdioServerParameters  # アプリの仮想環境の依存

    env = {**os.environ, **mcp_env(port, with_config)}
    return StdioServerParameters(command=str(PY), args=["-m", "mcp_server"], cwd=str(REPO), env=env, encoding="utf-8")


async def list_tools_only(port: int) -> dict:
    """dry-run 用: MCP サーバーを stdio で起こし、initialize と tools/list だけを呼ぶ（tools/call はしない）。
    一時 yaml はまだ書かないので、LTX_MCP_CONFIG は渡さない（config.yaml を読むだけ）。"""
    from mcp import ClientSession  # noqa: PLC0415
    from mcp.client.stdio import stdio_client  # noqa: PLC0415

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "mcp_server.stderr.log", "a", encoding="utf-8", newline="\n") as errlog:
        errlog.write(f"\n===== {now_iso()} h3_inpaint_gate_api.py --dry-run（tools/list だけ） =====\n")
        errlog.flush()
        async with stdio_client(mcp_params(port, with_config=False), errlog=errlog) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=TOOL_READ_TIMEOUT) as session:
                init = await session.initialize()
                report = check_tool_schemas((await session.list_tools()).tools)
                report["server_info"] = {"name": init.serverInfo.name, "version": init.serverInfo.version,
                                         "protocol": init.protocolVersion}
                return report


async def ensure_pipeline(mcp: Mcp, args, inv: dict) -> bool:
    """H3 を載せる。load_pipeline は 1 回だけ（読み込み中の呼び直しは 409）。以後 backend_status を 10 秒おき。"""
    load: dict[str, Any] = {"requested": {"base_model": H3_BASE_MODEL, "models": {"transformer": "default"},
                                          "wait_sec": 45}, "responses": [], "status_polls": 0}
    inv["load"] = load
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
    try:
        models = await mcp.call("list_models")
        h3 = next((b for b in models.get("base_models", []) if b.get("id") == H3_BASE_MODEL), None) or {}
        inv["models_after_load"] = {k: h3.get(k) for k in ("transformer_role", "transformer_weight_class",
                                                           "unsupported_features", "constraints")}
    except ToolCallError as exc:
        inv["models_after_load"] = {"error": exc.text}
    return True


async def wait_job(mcp: Mcp, job_id: str, timeout_sec: float) -> dict:
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
            log(f"  job {job_id[:8]} status={st.get('status')} progress={st.get('progress')} stage={st.get('stage')} "
                f"step={st.get('current_step')}")
            last_key = key
        if not st.get("timed_out"):
            return {"final": st, "calls": calls, "seconds": round(time.monotonic() - t0, 1), "samples": samples}
        if time.monotonic() - t0 > timeout_sec:
            return {"final": st, "calls": calls, "seconds": round(time.monotonic() - t0, 1), "samples": samples,
                    "gave_up": f"{timeout_sec} 秒待っても終わらない（ジョブはバックエンドで続いている）"}


# ---------------------------------------------------------------------------------------------
# 1 本のジョブ
# ---------------------------------------------------------------------------------------------

def denoise_picks(meta: dict) -> dict:
    den = (((meta.get("h3") or {}).get("phases") or {}).get("denoise")) or {}
    v16 = den.get("vram16") or {}
    return {"peak_vram_mb": den.get("peak_vram_mb"), "peak_vram_reserved_mb": den.get("peak_vram_reserved_mb"),
            "seconds": den.get("seconds"), "block_swap_prefetch_used": den.get("block_swap_prefetch_used"),
            "sequence_total": (den.get("sequence") or {}).get("total"),
            "vram16": {k: v16.get(k) for k in ("cap_mib", "num_alloc_retries", "num_ooms", "blocks_on_gpu")}}


async def run_job(mcp: Mcp, args, job: dict) -> dict:
    gate = job["gate"]
    job_dir = OUT_ROOT / gate
    job_dir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {k: job[k] for k in ("gate", "purpose", "frames", "source", "mask", "source_size",
                                               "canvas", "geometry", "feather_px")}
    rec.update(started_at=now_iso(), status=None, error=None, dir=str(job_dir))
    log(f"=== {gate}: {job['purpose']} ===")
    ups: dict[str, Any] = {}
    for kind in ("source", "mask"):
        try:
            up = await mcp.call("upload_video", {"file_path": job[kind], "max_frames": UPLOAD_MAX_FRAMES})
        except ToolCallError as exc:
            rec.update(status="upload_error", error=f"{kind}: {exc.text}", finished_at=now_iso())
            return rec
        ups[kind] = up
        log(f"upload {kind}: video_id={up.get('video_id')} frame_count={up.get('frame_count')} fps={up.get('fps')}")
    rec["upload"] = ups
    tool_args = {**job["tool_args"], "reference_video_id": ups["source"]["video_id"],
                 "inpaint_mask_video_id": ups["mask"]["video_id"]}
    rec["tool_args"] = tool_args
    write_json(job_dir / "request.json", {
        "tool": "submit_generate", "arguments": tool_args, "gate": gate, "not_sent": NOT_SENT,
        "note": "MCP の道具に渡した引数。H3 では MCP が reference_video_id を inpaint.source_video_id へ移す。"
                "サーバーが受けた本文（既定の書き戻し後）は metadata.json の request"})
    server_log = RT / "logs" / "server.log"
    srv_off = file_offset(server_log)
    wk_off = file_offset(worker_log_path())
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
    wk = read_since(worker_log_path(), wk_off)
    if wk:
        write_text(job_dir / "worker_log_excerpt.txt", wk)
    srv = read_since(server_log, srv_off)
    if srv:
        write_text(job_dir / "server_log_excerpt.txt", srv)
    if waited.get("gave_up"):
        rec.update(status="wait_gave_up", error=waited["gave_up"], finished_at=now_iso())
        return rec
    if rec["status"] != "completed":
        rec["error"] = final.get("error")
        log(f"ジョブが {rec['status']}: {(final.get('error') or '')[:300]}")
        rec["finished_at"] = now_iso()
        return rec
    try:
        saved = await mcp.call("save_job_video", {"job_id": job_id, "dest_dir": str(job_dir),
                                                  "filename": "output.mp4", "no_clobber": True, "which": "output"})
        rec["saved_path"] = saved["saved_path"]
    except ToolCallError as exc:
        rec["error"] = f"save_job_video: {exc.text}"
    try:
        vp = await mcp.call("get_job_video_path", {"job_id": job_id})
        src_dir = Path(vp["path"]).parent
        rec["job_dir"] = str(src_dir)
        copied, missing = [], []
        for name in ("metadata.json", "_inpaint_window.mp4", "_inpaint_mask.mp4"):
            if (src_dir / name).exists():
                shutil.copy2(src_dir / name, job_dir / name)
                copied.append(name)
            else:
                missing.append(name)
        rec["copied_from_job_dir"] = copied
        if missing:
            rec["missing_in_job_dir"] = missing
            log(f"注意: ジョブのフォルダに無い: {missing}（{src_dir}）")
    except ToolCallError as exc:
        rec["missing_in_job_dir"] = exc.text
    meta: dict = {}
    if (job_dir / "metadata.json").exists():
        meta = json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))
    h3 = meta.get("h3") or {}
    rec["metadata_summary"] = {k: meta.get(k) for k in (
        "resolution", "attention_used", "vram16_mode_used", "keep_resident_used", "block_swap_prefetch_used",
        "generation_time_seconds", "peak_vram_reserved_mb", "seed_used")}
    rec["metadata_summary"]["num_inference_steps"] = h3.get("num_inference_steps")
    rec["metadata_summary"]["request_inpaint"] = (meta.get("request") or {}).get("inpaint")
    rec["denoise"] = denoise_picks(meta)
    rec["finished_at"] = now_iso()
    write_json(job_dir / "result.json", {"gate": gate, "run": rec, "metadata": meta})
    return rec


# ---------------------------------------------------------------------------------------------
# 測定（エンジンの仮想環境の子プロセス。GPU を隠す。CPU だけ）
# ---------------------------------------------------------------------------------------------

def measure_helper_main(req_path: str, res_path: str) -> int:
    """（.venv-engine-h3 で走る）出力・切り出し・対照を製品と同じ復号（PyAV の rgb24）で読み、
    マスクを製品の ``read_inpaint_mask`` で塊のブロックにし、芯・マスクの中・境目の量と図を作る。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(REPO))
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    from engine_h3.constants import FRAME_RATE  # noqa: PLC0415
    from engine_h3.inpaint_geometry import (  # noqa: PLC0415
        TOKEN_PX, chunk_block_masks, chunk_ranges, compute_inpaint_geometry, feather_weights_from_mask)
    from engine_h3.inpaint_h3 import read_inpaint_mask  # noqa: PLC0415
    from engine_h3.media_out import write_mp4  # noqa: PLC0415
    from engine_h3.source_tail import read_source_frames  # noqa: PLC0415

    req = json.loads(Path(req_path).read_text(encoding="utf-8"))
    nf = int(req["frames"])
    sw, sh = (int(v) for v in req["source"])
    cw, ch = (int(v) for v in req["canvas"])
    fpx = int(req["feather_px"])
    geom = compute_inpaint_geometry(sw, sh, cw, ch)
    pt, pl = geom.pad_top, geom.pad_left

    cut = read_source_frames(req["cut"], nf)
    if cut.shape[1:3] != (sh, sw):
        raise RuntimeError(f"切り出しの寸法 {cut.shape[2]}x{cut.shape[1]} が元の実寸 {sw}x{sh} と違う")
    control = Path(req["control"])
    made_control = False
    if not control.exists():
        # 写し元（h3_outpaint_gate_api.measure_helper_main）と同じ: 切り出しの画素を write_mp4 で 1 回書き直す（音声なし）
        write_mp4(str(control), torch.from_numpy(cut), fps=FRAME_RATE, audio=None)
        made_control = True
    ctrl = read_source_frames(str(control), nf)
    out = read_source_frames(req["output"], nf)
    if out.shape[1:3] != (sh, sw):
        raise RuntimeError(f"出力の寸法 {out.shape[2]}x{out.shape[1]} が元の実寸 {sw}x{sh} と違う")
    blocks, proof = read_inpaint_mask(req["mask"], nf, (sw, sh), geom)
    cblocks = chunk_block_masks(blocks, nf)
    ranges = chunk_ranges(nf)
    weights = [feather_weights_from_mask(cblocks[k], geom, fpx) for k in range(len(ranges))]
    core = [w == 1.0 for w in weights]                       # 芯（元の画素がそのまま貼り戻される所）
    band = [w < 1.0 for w in weights]                        # マスク ∪ なじませ帯
    tokpix = [np.repeat(np.repeat(cblocks[k], TOKEN_PX, 0), TOKEN_PX, 1)[pt:pt + sh, pl:pl + sw]
              for k in range(len(ranges))]                   # 塊のトークンマスクの画素
    chunk_of = np.zeros(nf, dtype=int)
    for k, (a, b) in enumerate(ranges):
        chunk_of[a:b] = k
    bottom = np.zeros((sh, sw), dtype=bool)
    bottom[sh - BOTTOM_ROWS:, :] = True

    def psnr(total: float, count: int):
        if count == 0:
            return None
        mse = total / count
        return 100.0 if mse == 0 else round(float(10.0 * np.log10(255.0 ** 2 / mse)), 4)

    nk = len(ranges)
    acc = {name: {"out": [0.0] * nk, "ctrl": [0.0] * nk, "n": [0] * nk} for name in ("core", "core_bottom", "full")}
    mad_sum, mad_n = [0.0] * nk, [0] * nk
    for i in range(nf):
        k = int(chunk_of[i])
        c = cut[i].astype(np.float32)
        do = out[i].astype(np.float32) - c
        dc = ctrl[i].astype(np.float32) - c
        for name, m in (("core", core[k]), ("core_bottom", core[k] & bottom), ("full", None)):
            so = do if m is None else do[m]
            sc = dc if m is None else dc[m]
            acc[name]["out"][k] += float(np.square(so, dtype=np.float32).sum(dtype=np.float64))
            acc[name]["ctrl"][k] += float(np.square(sc, dtype=np.float32).sum(dtype=np.float64))
            acc[name]["n"][k] += int(so.size)
        ad = np.abs(do).mean(axis=2)
        mad_sum[k] += float(ad[tokpix[k]].sum(dtype=np.float64))
        mad_n[k] += int(tokpix[k].sum())
    psnr_rec: dict[str, Any] = {}
    for name, a in acc.items():
        psnr_rec[name] = {
            "output_vs_cut_db": psnr(sum(a["out"]), sum(a["n"])),
            "control_vs_cut_db": psnr(sum(a["ctrl"]), sum(a["n"])),
            "values": sum(a["n"]),
            "per_chunk": [{"chunk": list(ranges[k]), "output_vs_cut_db": psnr(a["out"][k], a["n"][k]),
                           "control_vs_cut_db": psnr(a["ctrl"][k], a["n"][k]), "values": a["n"][k]}
                          for k in range(nk)],
        }
    mask_mad = {"all": round(sum(mad_sum) / sum(mad_n), 4) if sum(mad_n) else None, "pixels": sum(mad_n),
                "per_chunk": [{"chunk": list(ranges[k]),
                               "mad": round(mad_sum[k] / mad_n[k], 4) if mad_n[k] else None, "pixels": mad_n[k]}
                              for k in range(nk)]}

    # 塊の境目: 「マスク ∪ なじませ帯」（その対の 2 枚の塊の和集合）の画素で、隣り合う枚の平均絶対差
    adj = []
    for i in range(nf - 1):
        region = band[int(chunk_of[i])] | band[int(chunk_of[i + 1])]
        n = int(region.sum())
        if n == 0:
            adj.append({"pair": [i, i + 1], "output_mad": None, "cut_mad": None, "pixels": 0})
            continue
        d_out = np.abs(out[i + 1].astype(np.int16) - out[i].astype(np.int16)).mean(axis=2)
        d_cut = np.abs(cut[i + 1].astype(np.int16) - cut[i].astype(np.int16)).mean(axis=2)
        adj.append({"pair": [i, i + 1], "output_mad": round(float(d_out[region].mean()), 4),
                    "cut_mad": round(float(d_cut[region].mean()), 4), "pixels": n,
                    "chunk_boundary": bool(chunk_of[i] != chunk_of[i + 1])})
    xor = []
    for k in range(nk - 1):
        x = cblocks[k] ^ cblocks[k + 1]
        xp = np.repeat(np.repeat(x, TOKEN_PX, 0), TOKEN_PX, 1)[pt:pt + sh, pl:pl + sw]
        xor.append({"chunks": [list(ranges[k]), list(ranges[k + 1])], "tokens": int(x.sum()),
                    "pixels": int(xp.sum()), "tokens_a": int(cblocks[k].sum()), "tokens_b": int(cblocks[k + 1].sum())})

    # 図（スパイクの描き方: マスクの升目を赤 (255,0,0,100)、32 画素の格子を白 (255,255,255,70)）
    def red_overlay(img, pix, grid=False):
        ov = np.zeros(img.shape[:2] + (4,), np.uint8)
        ov[pix] = (255, 0, 0, 100)
        if grid:
            for r in range(0, sh + 1, TOKEN_PX):
                ov[min(r, sh - 1), :] = (255, 255, 255, 70)
            for q in range(0, sw + 1, TOKEN_PX):
                ov[:, min(q, sw - 1)] = (255, 255, 255, 70)
        base = Image.fromarray(img).convert("RGBA")
        return Image.alpha_composite(base, Image.fromarray(ov, "RGBA")).convert("RGB")

    scale = SHEET_TILE_LONG / max(sw, sh)
    tw, th = round(sw * scale), round(sh * scale)
    head = 22
    frames_for_sheet = [f for f in SHEET_FRAMES if f < nf]
    sheet = Image.new("RGB", (tw * 3, (th + head) * len(frames_for_sheet)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for r, f in enumerate(frames_for_sheet):
        k = int(chunk_of[f])
        tiles = [("source (cut)", Image.fromarray(cut[f])), ("output", Image.fromarray(out[f])),
                 ("output + token mask (red)", red_overlay(out[f], tokpix[k]))]
        y = r * (th + head)
        for j, (label, im) in enumerate(tiles):
            sheet.paste(im.resize((tw, th), Image.LANCZOS), (j * tw, y + head))
            draw.text((j * tw + 6, y + 5), f"f{f:03d} chunk {k} {label}", fill=(235, 235, 235))
    sheet.save(req["sheet"])
    fo = min(OVERLAY_FRAME, nf - 1)
    red_overlay(cut[fo], tokpix[int(chunk_of[fo])], grid=True).save(req["overlay"])

    res = {
        "made_at": now_iso(), "frames": nf, "source": [sw, sh], "canvas": [cw, ch], "feather_px": fpx,
        "pads": {"left": geom.pad_left, "right": geom.pad_right, "top": geom.pad_top, "bottom": geom.pad_bottom},
        "control": {"path": str(control), "made_now": made_control,
                    "note": "切り出し（_inpaint_window.mp4）の画素を engine_h3.media_out.write_mp4 で 1 回書き直したもの（音声なし）"},
        "mask_proof_reread": {k: (float(v) if not isinstance(v, int) else v) for k, v in proof.items()},
        "chunks": [list(r) for r in ranges],
        "chunk_tokens": [int(cblocks[k].sum()) for k in range(nk)],
        "core_pixels_per_chunk": [int(c.sum()) for c in core],
        "band_pixels_per_chunk": [int(b.sum()) for b in band],
        "psnr": psnr_rec, "mask_mad": mask_mad, "adjacent": adj, "chunk_xor": xor,
        "sheet": req["sheet"], "overlay": req["overlay"],
        "decoder": "engine_h3.source_tail.read_source_frames（PyAV・rgb24。エンジンが元動画を読むのと同じ）",
    }
    write_json(Path(res_path), res)
    return 0


def run_measure_helper(req: dict, req_path: Path, res_path: Path) -> dict:
    write_json(req_path, req)
    if res_path.exists():
        res_path.unlink()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": "-1",
           "TORCH_COMPILE_DISABLE": "1"}
    cmd = [str(ENGINE_PY), str(Path(__file__).resolve()), "--measure-helper", str(req_path), str(res_path)]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=3600)
    if proc.returncode != 0 or not res_path.exists():
        return {"error": f"測定の子プロセスが失敗（{proc.returncode}）", "stderr_tail": proc.stderr[-3000:], "cmd": cmd}
    return json.loads(res_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# 後処理（合否 1〜6・図・HF のフォルダ）
# ---------------------------------------------------------------------------------------------

def gate_files(gate: str) -> dict[str, Path]:
    d = OUT_ROOT / gate
    return {"output": d / "output.mp4", "metadata": d / "metadata.json", "cut": d / "_inpaint_window.mp4",
            "mask": d / "_inpaint_mask.mp4"}


def check_probe(gate: str, probe: dict) -> dict:
    sw, sh = GATES[gate]["source_size"]
    v = probe["video"]
    items = {
        "size": {"value": [v.get("width"), v.get("height")], "expect": [sw, sh],
                 "ok": (v.get("width"), v.get("height")) == (sw, sh)},
        "frames": {"value": v.get("nb_read_frames"), "expect": NUM_FRAMES, "ok": v.get("nb_read_frames") == NUM_FRAMES},
        "fps": {"value": v.get("fps"), "expect": H3_FPS,
                "ok": v.get("fps") is not None and abs(v["fps"] - H3_FPS) < 1e-3},
        "audio": {"value": (probe.get("audio") or {}).get("codec_name"), "ok": bool(probe.get("audio"))},
    }
    return {"items": items, "ok": all(i["ok"] for i in items.values())}


def check_metadata(gate: str, meta: dict) -> dict:
    g = GATES[gate]
    sw, sh = g["source_size"]
    cw, ch = g["expect_canvas"]
    ip = meta.get("inpaint")
    if not isinstance(ip, dict):
        return {"ok": False, "error": "metadata.json に inpaint が無い"}
    pads_expect = {"left": 0, "right": cw - sw, "top": 0, "bottom": g["expect_pad_bottom"]}
    sc = ip.get("self_check") or {}
    mp = ip.get("mask_proof") or {}
    items = {
        "canvas": {"value": ip.get("canvas"), "expect": [cw, ch], "ok": ip.get("canvas") == [cw, ch]},
        "pads": {"value": ip.get("pads"), "expect": pads_expect, "ok": ip.get("pads") == pads_expect},
        "self_check.fixed_latents_equal": {"value": sc.get("fixed_latents_equal"),
                                           "ok": sc.get("fixed_latents_equal") is True},
        "feather_px": {"value": ip.get("feather_px"), "expect": g["expect_feather_px"],
                       "ok": ip.get("feather_px") == g["expect_feather_px"]},
        "mask_proof.white_ratio": {"value": mp.get("white_ratio"),
                                   "ok": isinstance(mp.get("white_ratio"), (int, float)) and mp["white_ratio"] > 0},
        "provenance_keys": {"missing": [k for k in PROVENANCE_KEYS if k not in ip],
                            "ok": all(k in ip for k in PROVENANCE_KEYS)},
        "resolution": {"value": (meta.get("output") or {}).get("resolution"), "expect": f"{sw}x{sh}", "ok": (meta.get("output") or {}).get("resolution") == f"{sw}x{sh}"},
    }
    return {"items": items, "engine_keys_missing": [k for k in ENGINE_KEYS if k not in ip],
            "extra_keys": sorted(set(ip) - set(ENGINE_KEYS) - set(PROVENANCE_KEYS)),
            "ok": all(i["ok"] for i in items.values())}


def judge(gate: str, probe_chk: dict, meta_chk: dict, m: dict) -> dict:
    """計画 §6.2 の合否 1〜6（7 はオーナーの目視）。"""
    def ge(a, b):
        return a is not None and b is not None and a >= b - PSNR_MARGIN_DB

    core = m["psnr"]["core"]
    bot = m["psnr"]["core_bottom"]
    adj = m["adjacent"]
    judged_pairs = {tuple(p) for p in JUDGED_BOUNDARIES}
    boundary_pairs = {tuple(a["pair"]) for a in adj if a.get("chunk_boundary")}
    others = [a["output_mad"] for a in adj
              if tuple(a["pair"]) not in boundary_pairs and a["output_mad"] is not None]
    med = statistics.median(others) if others else None
    bnd = []
    for a in adj:
        if tuple(a["pair"]) in boundary_pairs:
            ratio = None if (med in (None, 0) or a["output_mad"] is None) else round(a["output_mad"] / med, 4)
            bnd.append({"pair": a["pair"], "output_mad": a["output_mad"], "cut_mad": a["cut_mad"], "ratio": ratio,
                        "judged": tuple(a["pair"]) in judged_pairs,
                        "ok": ratio is not None and ratio <= BOUNDARY_RATIO_MAX})
    items = {
        "1_output_probe": {"ok": probe_chk["ok"]},
        "2_metadata_inpaint": {"ok": meta_chk["ok"]},
        "3_core_psnr": {"output_vs_cut_db": core["output_vs_cut_db"], "control_vs_cut_db": core["control_vs_cut_db"],
                        "threshold_db": None if core["control_vs_cut_db"] is None
                        else round(core["control_vs_cut_db"] - PSNR_MARGIN_DB, 4),
                        "per_chunk_ok": [ge(c["output_vs_cut_db"], c["control_vs_cut_db"]) for c in core["per_chunk"]],
                        "ok": ge(core["output_vs_cut_db"], core["control_vs_cut_db"])},
        "4_mask_mad": {"value": m["mask_mad"]["all"], "threshold": MASK_MAD_MIN,
                       "ok": m["mask_mad"]["all"] is not None and m["mask_mad"]["all"] >= MASK_MAD_MIN},
        "6_chunk_boundary": {"median_other_pairs": None if med is None else round(med, 4),
                             "median_excludes": "塊の境目の対すべて", "boundaries": bnd,
                             "ok": all(b["ok"] for b in bnd if b["judged"]) and any(b["judged"] for b in bnd)},
    }
    if gate == "I2":
        items["5_bottom_rows_core_psnr"] = {
            "rows": BOTTOM_ROWS, "output_vs_cut_db": bot["output_vs_cut_db"], "control_vs_cut_db": bot["control_vs_cut_db"],
            "values": bot["values"], "ok": ge(bot["output_vs_cut_db"], bot["control_vs_cut_db"])}
    return {"items": items, "ok": all(i["ok"] for i in items.values())}


def post_gate(gate: str, summary: dict) -> dict:
    """1 本の後処理。出力が無ければ ``status: no_output`` を返す（測らない）。"""
    files = gate_files(gate)
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        log(f"{gate}: 出力が無い（{len(missing)} 件）: {missing}")
        return {"gate": gate, "status": "no_output", "missing": missing, "at": now_iso()}
    d = OUT_ROOT / gate
    log(f"{gate}: 後処理を始める")
    probe = ffprobe(files["output"])
    meta = json.loads(files["metadata"].read_text(encoding="utf-8"))
    probe_chk = check_probe(gate, probe)
    meta_chk = check_metadata(gate, meta)
    ip = meta.get("inpaint") or {}
    fpx = ip.get("feather_px")
    if not isinstance(fpx, int):
        return {"gate": gate, "status": "measure_error", "error": "metadata.inpaint.feather_px が無い（芯が決まらない）",
                "probe_check": probe_chk, "metadata_check": meta_chk}
    sw, sh = GATES[gate]["source_size"]
    req = {"frames": NUM_FRAMES, "source": [sw, sh], "canvas": list(GATES[gate]["expect_canvas"]), "feather_px": fpx,
           "output": str(files["output"]), "cut": str(files["cut"]), "mask": str(files["mask"]),
           "control": str(d / "_control_write_mp4.mp4"), "sheet": str(d / "sheet.png"),
           "overlay": str(d / f"token_overlay_f{OVERLAY_FRAME:03d}.png")}
    m = run_measure_helper(req, d / "measure_request.json", d / "measure_result.json")
    if m.get("error"):
        log(f"{gate}: 測定に失敗: {m['error']}\n{m.get('stderr_tail', '')}")
        return {"gate": gate, "status": "measure_error", "error": m["error"], "measure": m,
                "probe_check": probe_chk, "metadata_check": meta_chk}
    verdict = judge(gate, probe_chk, meta_chk, m)
    post = {"gate": gate, "status": "measured", "at": now_iso(), "output_probe": probe, "probe_check": probe_chk,
            "metadata_check": meta_chk, "measure": m, "judge": verdict}
    log(f"{gate}: 合否 {'すべて合格' if verdict['ok'] else '不合格あり'} "
        f"{ {k: v['ok'] for k, v in verdict['items'].items()} }")
    # result.json に post を足す（本番の記録と metadata は残す）
    rpath = d / "result.json"
    rj = json.loads(rpath.read_text(encoding="utf-8")) if rpath.exists() else {"gate": gate}
    rj.setdefault("metadata", meta)
    rj["post"] = post
    write_json(rpath, rj)
    # HF のフォルダ
    HF_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(files["output"], HF_DIR / hf_names(gate)["output"])
    shutil.copy2(files["cut"], HF_DIR / hf_names(gate)["source"])
    shutil.copy2(d / "sheet.png", HF_DIR / hf_names(gate)["sheet"])
    shutil.copy2(Path(req["overlay"]), HF_DIR / hf_names(gate)["overlay"])
    return post


def hf_names(gate: str) -> dict[str, str]:
    sw, sh = GATES[gate]["source_size"]
    return {"output": f"{gate}_{sw}x{sh}_{NUM_FRAMES}f_seed{SEED}.mp4",
            "source": f"source_{gate}_{sw}x{sh}_{NUM_FRAMES}f.mp4",
            "sheet": f"sheet_{gate}.png", "overlay": f"token_overlay_{gate}_f{OVERLAY_FRAME:03d}.png"}


# ---------------------------------------------------------------------------------------------
# まとめ（summary.json・summary.md・hf-round15/README.md）
# ---------------------------------------------------------------------------------------------

def load_summary() -> dict:
    if SUMMARY_JSON.exists():
        try:
            data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("gates", {})
                data.setdefault("invocations", [])
                return data
        except Exception:  # noqa: BLE001
            pass
    return {"script": "h3_inpaint_gate_api.py", "plan": "enumerated-questing-cupcake.md §6.2（I1・I2）",
            "gates": {}, "invocations": []}


def _p(x, nd: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "合格" if x else "不合格"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def gate_rows(summary: dict) -> list[tuple[str, dict, dict]]:
    out = []
    for gate in GATES:
        g = summary["gates"].get(gate)
        if g:
            out.append((gate, g.get("run") or {}, g.get("post") or {}))
    return out


def metric_lines(gate: str, run: dict, post: dict) -> list[str]:
    """数値の表（合否 1〜6）。"""
    j = (post.get("judge") or {}).get("items") or {}
    m = post.get("measure") or {}
    pc = (post.get("probe_check") or {}).get("items") or {}
    mc = (post.get("metadata_check") or {}).get("items") or {}
    if post.get("status") != "measured":
        return [f"後処理の状態: {post.get('status', '未実行')}（{post.get('error') or post.get('missing') or ''}）", ""]
    i3, i4, i6 = j["3_core_psnr"], j["4_mask_mad"], j["6_chunk_boundary"]
    L = ["| 番号 | 項目 | 値 | 基準 | 合否 |", "|---|---|---|---|---|",
         f"| 1 | 出力の寸法・枚数・fps（1 秒あたりの枚数）・音声 | {pc['size']['value']}・{pc['frames']['value']} 枚・"
         f"{_p(pc['fps']['value'])}・{pc['audio']['value'] or '無し'} | {pc['size']['expect']}・{NUM_FRAMES} 枚・24・あり | "
         f"{_p(j['1_output_probe']['ok'])} |",
         f"| 2 | metadata の inpaint（キャンバス・pad・固定した潜在のビット一致・なじませ画素・マスクの白の割合・由来の鍵・結果の解像度） | "
         f"{mc['canvas']['value']}・{mc['pads']['value']}・{json.dumps(mc['self_check.fixed_latents_equal']['value'])}・"
         f"{mc['feather_px']['value']}・{_p(mc['mask_proof.white_ratio']['value'], 4)}・欠け {mc['provenance_keys']['missing']}・"
         f"{mc['resolution']['value']} | {mc['canvas']['expect']}・{mc['pads']['expect']}・true・{mc['feather_px']['expect']}・"
         f"0 より大・欠け無し・{mc['resolution']['expect']} | {_p(j['2_metadata_inpaint']['ok'])} |",
         f"| 3 | 芯の PSNR（出力 対 切り出し、全フレーム、dB） | {_p(i3['output_vs_cut_db'])} | 対照 {_p(i3['control_vs_cut_db'])} − 1.0 = "
         f"{_p(i3['threshold_db'])} 以上 | {_p(i3['ok'])} |",
         f"| 4 | マスクの中の平均絶対差（0〜255） | {_p(i4['value'])} | {MASK_MAD_MIN:g} 以上 | {_p(i4['ok'])} |"]
    if "5_bottom_rows_core_psnr" in j:
        i5 = j["5_bottom_rows_core_psnr"]
        L.append(f"| 5 | 下端 {BOTTOM_ROWS} 行の芯の PSNR（dB） | {_p(i5['output_vs_cut_db'])} | 対照 {_p(i5['control_vs_cut_db'])} − 1.0 "
                 f"以上 | {_p(i5['ok'])} |")
    for b in i6["boundaries"]:
        tag = "6" if b["judged"] else "6（参考）"
        L.append(f"| {tag} | 塊の境目 {b['pair'][0]}→{b['pair'][1]} 枚目の隣り合う枚の差 | {_p(b['output_mad'])}"
                 f"（中央値の {_p(b['ratio'])} 倍。元動画では {_p(b['cut_mad'])}） | 他の枚の中央値 {_p(i6['median_other_pairs'])} の "
                 f"{BOUNDARY_RATIO_MAX:g} 倍以内 | {_p(b['ok'])} |")
    L.append("")
    pcs = (m.get("psnr") or {}).get("core", {}).get("per_chunk") or []
    mads = (m.get("mask_mad") or {}).get("per_chunk") or []
    L += ["塊（17 枚ずつ）ごとの値:", "", "| 塊（枚） | 芯の PSNR 出力／対照（dB） | マスクの中の平均絶対差 | 升目の数 |", "|---|---|---|---|"]
    for k, c in enumerate(pcs):
        md = mads[k]["mad"] if k < len(mads) else None
        tok = (m.get("chunk_tokens") or [None] * len(pcs))[k]
        L.append(f"| {c['chunk'][0]}〜{c['chunk'][1] - 1} | {_p(c['output_vs_cut_db'])}／{_p(c['control_vs_cut_db'])} | "
                 f"{_p(md)} | {tok} |")
    L.append("")
    xs = m.get("chunk_xor") or []
    if xs:
        L.append("隣り合う塊でマスクの升目が入れ替わる数（排他的論理和〔XOR〕。升目／画素）: "
                 + "、".join(f"{x['chunks'][0][1] - 1}→{x['chunks'][1][0]}: {x['tokens']}／{x['pixels']}" for x in xs))
        L.append("")
    return L


def write_summary_md(summary: dict) -> None:
    L = ["# MiniMax H3 の Inpainting 弾 1 の GPU の門（I1・I2。API 経由。記録）", "",
         "台本 `Docs/Outputs-archive/h3-inpaint-2026-11-11/h3_inpaint_gate_api.py`。計画 §6.2。数値の正本は記録簿 "
         "`Docs/VERIFICATION_LOG.md`。", "",
         "| 本 | 状態 | ジョブ ID | 生成秒 | 壁秒 | VRAM の山 割り当て／予約（MiB） | 合否 1〜6 |", "|---|---|---|---|---|---|---|"]
    for gate, run, post in gate_rows(summary):
        d = run.get("denoise") or {}
        ok = (post.get("judge") or {}).get("ok")
        L.append(f"| {gate} | {run.get('status', '—')}／後処理 {post.get('status', '—')} | {run.get('job_id', '—')} | "
                 f"{_p((run.get('metadata_summary') or {}).get('generation_time_seconds'), 1)} | {_p(run.get('seconds_wall'), 1)} | "
                 f"{_p(d.get('peak_vram_mb'), 0)}／{_p(d.get('peak_vram_reserved_mb'), 0)} | "
                 f"{'すべて合格' if ok else ('不合格あり' if ok is False else '—')} |")
    L.append("")
    for gate, run, post in gate_rows(summary):
        L += [f"## {gate}", ""] + metric_lines(gate, run, post)
    write_text(SUMMARY_MD, "\n".join(L))


ITEM_LABELS = {"1_output_probe": "1 出力の寸法・枚数・音声", "2_metadata_inpaint": "2 metadata の inpaint",
               "3_core_psnr": "3 芯の PSNR", "4_mask_mad": "4 マスクの中の差", "5_bottom_rows_core_psnr": "5 下端 8 行の芯",
               "6_chunk_boundary": "6 塊の境目"}


def write_hf_readme(summary: dict, prompt: str) -> None:
    rows = gate_rows(summary)
    measured = [(g, r, p) for g, r, p in rows if p.get("status") == "measured"]
    if not measured:
        return
    concl = []
    for g, r, p in measured:
        ok = (p.get("judge") or {}).get("ok")
        bad = [k for k, v in ((p.get("judge") or {}).get("items") or {}).items() if not v.get("ok")]
        bad = [ITEM_LABELS.get(k, k) for k in bad]
        concl.append(f"- {g}: 機械で測れる合否 1〜6 は{'すべて合格' if ok else '不合格あり（' + '、'.join(bad) + '）'}。")
    L = [f"# {HF_NAME} — MiniMax H3 の Inpainting の GPU の門（2026-11-12）", "",
         "## 結論", ""] + concl + [
         "- 残るのは目視（合否 7）: 継ぎ目・色の段差・動きの連続。下の「見る順」で見てください。", "",
         "## 目的", "",
         "製品に入れた MiniMax H3 の Inpainting（部分の描き直し。マスク動画の白い所だけを描き直し、外は元の動画に戻す。"
         "台帳 PENDING_TASKS.md §1-124）を、利用者と同じ経路（MCP〔Model Context Protocol。外部の道具からアプリを操作する仕組み〕"
         "→ アプリのサーバー → エンジン）で 2 本回し、元の寸法のまま正しく描き直せるかを確かめました。", "",
         "## 条件", "",
         "| 欄 | I1 | I2 |", "|---|---|---|",
         "| 元の動画 | `inpaint-base.mp4`（1280×768） | `inpaint-base-1080p.mp4`（1920×1080） |",
         "| マスク動画 | `mask-768p.mp4` | `mask-1080p.mp4` |",
         "| キャンバス（生成する寸法） | 1280×768（足し分なし） | 1920×1088（下に 8 画素足して生成し、最後に切り戻す） |",
         f"| 枚数・fps | {NUM_FRAMES} 枚・24 | 同じ |",
         f"| 種（seed）・注意の実装（attention） | {SEED}・{ATTENTION} | 同じ |",
         "| ステップ・shift（ノイズの減らし方の偏り）・音声の強さ・なじませ段数 | サーバーの既定（20・12／3・0.5・1） | 同じ |",
         "| Turbo LoRA（少ないステップ用の追加の重み） | なし | なし |",
         f"| プロンプト | {prompt} | 同じ |", "",
         "## 数値", "",
         "用語: 芯＝マスクとなじませ帯の外で、元の画素がそのまま貼り戻される画素。PSNR（ピーク信号対雑音比。dB〔デシベル〕。"
         "大きいほど元に近い）。対照＝元の切り出しをエンジンと同じ書き出し（libx264・yuv420p・BT.709）で 1 回書き直したもの"
         "（再符号化 1 回ぶんの劣化の天井）。塊＝映像の VAE（画素と潜在を行き来する部品）が時間方向にまとめて扱う 17 枚。", ""]
    for g, r, p in measured:
        L += [f"### {g}", ""] + metric_lines(g, r, p)
    L += ["## ファイル", ""]
    for g, r, p in measured:
        n = hf_names(g)
        sw, sh = GATES[g]["source_size"]
        L += [f"- `{n['output']}`: {g} の出力（{sw}×{sh}・{NUM_FRAMES} 枚・音声あり）",
              f"- `{n['source']}`: {g} の元の動画の切り出し（サーバーが切った窓そのもの。見比べ用）",
              f"- `{n['sheet']}`: フレーム 0・36・72 の 3 行 × 「元」「出力」「出力にマスクの升目を赤で重ねたもの」の 3 列",
              f"- `{n['overlay']}`: フレーム {OVERLAY_FRAME} の元の映像に、その塊のマスクの升目（赤）と 32 画素の格子（白）を重ねた図"]
    L += ["", "## 見る順", "",
          "1. `sheet_*.png` で、マスクの中に破綻の無い絵が出て、周りとつながっているか。",
          "2. 出力の動画と元の切り出しを並べて再生し、マスクの縁（なじませ帯）に継ぎ目や色の段差が無いか。",
          "3. 16→17 枚目・33→34 枚目（塊の境目）で、マスクの中の動きが飛ばないか。",
          "4. I2 は画面の下端（足して生成し、切り戻した 8 画素の近く）に乱れが無いか。", ""]
    write_text(HF_DIR / "README.md", "\n".join(L))


def flush(summary: dict) -> None:
    write_json(SUMMARY_JSON, summary)
    write_summary_md(summary)


# ---------------------------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------------------------

async def run_live(args, jobs: list[dict], summary: dict, inv: dict) -> int:
    from mcp import ClientSession  # noqa: PLC0415
    from mcp.client.stdio import stdio_client  # noqa: PLC0415

    calls: list = []
    calls_path = LOG_DIR / f"mcp_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    inv["mcp_calls"] = str(calls_path)
    inv["mcp_env"] = mcp_env(args.port)
    code = 0
    try:
        with open(LOG_DIR / "mcp_server.stderr.log", "a", encoding="utf-8", newline="\n") as errlog:
            errlog.write(f"\n===== {now_iso()} h3_inpaint_gate_api.py {[j['gate'] for j in jobs]} =====\n")
            errlog.flush()
            async with stdio_client(mcp_params(args.port), errlog=errlog) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=TOOL_READ_TIMEOUT) as session:
                    init = await session.initialize()
                    inv["server_info"] = {"name": init.serverInfo.name, "version": init.serverInfo.version,
                                          "protocol": init.protocolVersion}
                    schema = check_tool_schemas((await session.list_tools()).tools)
                    inv["tool_schema_check"] = schema
                    if not schema["ok"]:
                        for p in schema["problems"]:
                            log(f"引数名の照合に失敗: {p}")
                        return 1
                    log(f"引数名の照合: 合格（submit_generate の inpaint_*: {schema.get('submit_generate_inpaint_args')}）")
                    mcp = Mcp(session, calls)
                    st = await mcp.call("backend_status")
                    inv["status_before"] = st
                    if not st.get("reachable"):
                        log(f"バックエンドに届かない: {st.get('error')}")
                        return 1
                    if not await ensure_pipeline(mcp, args, inv):
                        log(f"読み込みで止まる: {inv['load'].get('error')}")
                        return 1
                    flush(summary)
                    for job in jobs:
                        try:
                            rec = await run_job(mcp, args, job)
                        except Exception as exc:  # noqa: BLE001
                            rec = {"gate": job["gate"], "status": "script_error",
                                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                        summary["gates"][job["gate"]] = {"conditions": job, "run": rec}
                        flush(summary)
                        if rec.get("status") != "completed" or rec.get("error"):
                            code = 3
                    try:
                        inv["unload"] = await mcp.call("unload_pipeline")
                    except ToolCallError as exc:
                        inv["unload"] = {"error": exc.text}
    finally:
        write_json(calls_path, calls)
    return code


def do_post(gates: list[str], summary: dict, prompt: str) -> int:
    code = 0
    for gate in gates:
        try:
            post = post_gate(gate, summary)
        except Exception as exc:  # noqa: BLE001
            post = {"gate": gate, "status": "measure_error", "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
            log(f"{gate}: 後処理で例外: {post['error']}")
        summary["gates"].setdefault(gate, {})["post"] = post
        if post.get("status") != "measured":
            code = 3
        flush(summary)
    write_hf_readme(summary, prompt)
    if (HF_DIR / "README.md").exists():
        log(f"HF のフォルダ: {HF_DIR}")
    return code


def dry_run(args, gates: list[str], jobs: list[dict]) -> int:
    out: dict[str, Any] = {
        "made_at": now_iso(), "argv": sys.argv[1:],
        "server": {"port": args.port, "temp_config": str(RT / TEMP_CONFIG_NAME), "temp_state": str(RT / TEMP_STATE_NAME),
                   "cmd": server_cmd(), "note": "config.yaml は読むだけ。port・log_dir・output.dir・upload.dir・state_file "
                                                 "を _rt へ（dry-run では書かない）。backend: auto のまま"},
        "mcp_server": {"command": str(PY), "args": ["-m", "mcp_server"], "cwd": str(REPO), "env": mcp_env(args.port)},
        "load_pipeline": {"base_model": H3_BASE_MODEL, "models": {"transformer": "default"}, "wait_sec": 45},
        "not_sent": NOT_SENT,
        "preflight": preflight_checks(),
        "gates": {},
    }
    log_preflight(out["preflight"])
    ok = True
    for job in jobs:
        gate = job["gate"]
        mat = material_check(job)
        existing = [str(p) for p in gate_files(gate).values() if p.exists()]
        fp = job["feather_px"]
        entry = {
            "purpose": job["purpose"], "source_size": job["source_size"], "canvas_round_up_32": job["canvas"],
            "geometry": job["geometry"], "chunks": job["chunks"],
            "feather_px": fp, "feather_px_expect": job["expect_feather_px"],
            "feather_px_ok": fp["value"] == job["expect_feather_px"] and fp["formula"] == job["expect_feather_px"],
            "material": mat,
            "upload_video": [{"file_path": job["source"], "max_frames": UPLOAD_MAX_FRAMES},
                             {"file_path": job["mask"], "max_frames": UPLOAD_MAX_FRAMES}],
            "submit_generate": {**job["tool_args"], "reference_video_id": "<元動画の upload_video の video_id>",
                                "inpaint_mask_video_id": "<マスクの upload_video の video_id>"},
            "save_to": str(OUT_ROOT / gate / "output.mp4"),
            "existing_outputs": existing,
        }
        out["gates"][gate] = entry
        g = job["geometry"]
        log(f"{gate}: 元 {job['source_size']} → キャンバス {job['canvas']}・pad 右 {g['pad_right']} 下 {g['pad_bottom']}・"
            f"格子 {g['grid']}・塊 {job['chunks']}・feather_px {fp}（期待 {job['expect_feather_px']}）")
        sv, mv = mat["source_probe"]["video"], mat["mask_probe"]["video"]
        log(f"{gate}: 元動画 {sv.get('width')}x{sv.get('height')} {sv.get('nb_read_frames')} 枚 {_p(sv.get('fps'))}fps "
            f"音声 {(mat['source_probe'].get('audio') or {}).get('codec_name')}／マスク {mv.get('width')}x{mv.get('height')} "
            f"{mv.get('nb_read_frames')} 枚 {_p(mv.get('fps'))}fps（24fps 換算 {mat['mask_frames_at_24fps']} 枚）")
        if not mat["ok"]:
            ok = False
            for p in mat["problems"]:
                log(f"{gate}: 素材の問題: {p}")
        if not entry["feather_px_ok"]:
            ok = False
            log(f"{gate}: feather_px が期待と違う")
        if existing:
            log(f"{gate}: 注意: 既に出力がある（本番は止まる。別の場所へ移してから回す）: {existing}")
    try:
        out["tools_list"] = asyncio.run(asyncio.wait_for(list_tools_only(args.port), timeout=120))
        log(f"tools/list の照合: {'合格' if out['tools_list']['ok'] else '失敗 ' + str(out['tools_list']['problems'])}"
            f"（道具 {out['tools_list']['tool_count']} 個・submit_generate の inpaint_*: "
            f"{out['tools_list'].get('submit_generate_inpaint_args')}）")
        ok = ok and out["tools_list"]["ok"]
    except Exception as exc:  # noqa: BLE001
        out["tools_list"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        log(f"tools/list を呼べなかった: {out['tools_list']['error']}")
        ok = False
    out["ok"] = ok
    write_json(OUT_ROOT / "dry_run.json", out)
    log(f"dry-run の記録: {OUT_ROOT / 'dry_run.json'}（{'問題なし' if ok else '問題あり'}）")
    return 0 if ok else 1


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="MiniMax H3 の Inpainting 弾 1 の GPU の門 I1・I2（MCP 経由・別ポートの実バックエンド）")
    ap.add_argument("--gate", choices=("I1", "I2", "all"), help="回す門（all は 1 回の起動で I1 → I2）")
    ap.add_argument("--dry-run", action="store_true",
                    help="素材の ffprobe・幾何・引数の組み立て・tools/list の照合だけ（サーバーを起こさない。GPU を使わない）")
    ap.add_argument("--post-only", action="store_true", help="既にある出力から指標と HF のフォルダを作る（GPU を使わない）")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT, help=f"一時サーバーのポート（既定 {PORT_DEFAULT}）")
    ap.add_argument("--prompt", default=None, help="プロンプトの差し替え（既定はスパイクと同じ文）")
    ap.add_argument("--load-timeout-sec", type=float, default=1200.0)
    ap.add_argument("--job-timeout-sec", type=float, default=7200.0, help="1 本を待つ上限（既定 7200 秒）")
    ap.add_argument("--measure-helper", nargs=2, metavar=("REQ", "RES"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.measure_helper:
        return args
    if not args.gate:
        ap.error("--gate を指定する（I1・I2・all）")
    if args.dry_run and args.post_only:
        ap.error("--dry-run と --post-only は同時に指定できない")
    if args.port == OWNER_PORT:
        ap.error(f"ポート {OWNER_PORT} はオーナーのサーバーの番号。別の番号にする")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args(argv)
    if args.measure_helper:
        return measure_helper_main(*args.measure_helper)
    mode = "dry_run" if args.dry_run else ("post_only" if args.post_only else "live")
    log.open(LOG_DIR / f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    prompt = args.prompt or DEFAULT_PROMPT
    gates = gate_list(args.gate)
    try:
        jobs = [build_job(g, prompt) for g in gates]
        if args.dry_run:
            return dry_run(args, gates, jobs)
        summary = load_summary()
        inv: dict[str, Any] = {"started_at": now_iso(), "argv": sys.argv[1:], "mode": mode, "gates": gates,
                               "log": str(log.path)}
        summary["invocations"].append(inv)
        if args.post_only:
            code = do_post(gates, summary, prompt)
            inv.update(finished_at=now_iso(), exit_code=code)
            flush(summary)
            log(f"記録: {SUMMARY_JSON}・{SUMMARY_MD}（終了コード {code}）")
            return code
        # 本番
        for job in jobs:
            mat = material_check(job)
            if not mat["ok"]:
                log(f"{job['gate']}: 素材の問題で止まる: {mat['problems']}")
                return 1
            existing = [str(p) for p in gate_files(job["gate"]).values() if p.exists()]
            if existing:
                log(f"{job['gate']}: 既に出力がある（上書きしない）。別の場所へ移してから回す: {existing}")
                return 1
        pre = preflight_checks()
        inv["preflight"] = pre
        log_preflight(pre)
        if pre["blocking"]:
            for b in pre["blocking"]:
                log(f"前の確かめで止まる: {b}")
            flush(summary)
            return 1
        flush(summary)
        srv = Server(args.port)
        code = 1
        try:
            srv.start()
            inv["server"] = srv.record
            flush(summary)
            code = asyncio.run(run_live(args, jobs, summary, inv))
        except BaseException as exc:  # noqa: BLE001 -- 例外も中断も記録して終わる
            inv["error"] = f"{type(exc).__name__}: {exc}"
            inv["traceback"] = traceback.format_exc()
            log(f"止まった: {inv['error']}")
            code = 1
        finally:
            try:
                srv.stop()
            except Exception as exc:  # noqa: BLE001
                srv.record["stop_error"] = f"{type(exc).__name__}: {exc}"
            inv["server"] = srv.record
            inv["workers_after_stop"] = [{"pid": p["pid"], "cmdline": p["cmdline"][:300]}
                                         for p in all_processes() if WORKER_RE.search(p["cmdline"] or "")]
            if inv["workers_after_stop"]:
                log(f"注意: 止めた後もワーカーが残っている: {inv['workers_after_stop']}")
            flush(summary)
        done = [j["gate"] for j in jobs
                if ((summary["gates"].get(j["gate"]) or {}).get("run") or {}).get("status") == "completed"]
        if done:
            pcode = do_post(done, summary, prompt)
            code = code or pcode
        inv.update(finished_at=now_iso(), exit_code=code)
        flush(summary)
        log(f"記録: {SUMMARY_JSON}・{SUMMARY_MD}（終了コード {code}）")
        return code
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
