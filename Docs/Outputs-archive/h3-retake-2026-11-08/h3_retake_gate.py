"""MiniMax H3 の Retake（入力動画の指定区間だけを描き直す）の GPU の門を、MCP サーバー経由で回す台本。

計画の正本: ``C:/Users/phenotype nezu/.claude/plans/foamy-napping-kettle.md`` の §7「検証の門」の 3。
製品コードは import しない。MCP サーバー（``.venv\\Scripts\\python.exe -m mcp_server``）を stdio の子プロセスとして
起こし、``initialize → tools/list → tools/call`` の実プロトコルで道具を呼ぶ（記録簿 VERIFICATION_LOG §39.6 と同じ経路）。

流れ:
  1. ``backend_status`` で到達を確認
  2. ``load_pipeline(base_model="MiniMaxH3", models={"transformer": <transformer>}, wait_sec=45)``。
     応答待ちが時間切れなら ``backend_status`` で ``pipeline_loaded`` になるまで待つ（``--skip-load`` で省略）
  3. ``upload_video(material, max_frames=<十分大きい値>)`` を 1 回だけ（``frame_count``・``fps`` を記録）
  4. strength ごとに ``submit_chain``（clips 1 本・のりしろは渡さない）→ ``wait_for_job`` を繰り返す
     → ``save_job_video`` → ``metadata.json`` を写す → 送った引数を ``request.json`` に書く
  5. 記録だけ取る（判定は付けない）: ``metadata.retake``、出力のフレーム数、頭と尾の 22 フレームの PSNR
  6. ``<out-dir>/summary.json`` と ``<out-dir>/summary.md``

終了コード: 0＝全条件のジョブが completed、3＝どれかの条件が失敗（failed／送信の拒否／時間切れなど。先へは進む）、
1＝準備の段で止まった（到達できない・道具の引数名が違う・読み込みの失敗など）。

標準出力は UTF-8（``PYTHONUTF8=1`` で起動する前提。念のため台本でも ``reconfigure`` する）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------------------------

REPO_DEFAULT = Path(r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni")
OUT_DIR_DEFAULT = REPO_DEFAULT / "outputs" / "h3-retake-gates-2026-11-08"
BASE_URL_DEFAULT = "http://127.0.0.1:18620"

H3_BASE_MODEL = "MiniMaxH3"
H3_FPS = 24.0
GLUE_FRAMES = 22          # H3 の頭と尾ののりしろ（固定）。PSNR の帯の幅
DIM_MULTIPLE = 32         # H3 の幅と高さの倍数
LONG_SIDE_DEFAULT = 1280  # 既定の解像度の長辺（README「解像度の既定」）
UPLOAD_MAX_FRAMES = 100000  # 尺を測るためだけの max_frames（元の尺より確実に大きい値）

TOOL_READ_TIMEOUT = timedelta(seconds=360)  # 道具 1 回の応答待ち（upload_video は最大 300 秒）
WAIT_TIMEOUT_SEC = 45                       # wait_for_job の 1 回の待ち（道具側で 1〜45 にクランプ）
WAIT_POLL_SEC = 5.0
MAX_CONSECUTIVE_TRANSPORT_ERRORS = 12

# tools/list の inputSchema で照合する引数名。台本が実際に渡す名前だけを並べる（名前が無ければ止まる）。
REQUIRED_TOOL_ARGS: dict[str, set[str]] = {
    "backend_status": set(),
    "list_models": set(),
    "load_pipeline": {"base_model", "models", "wait_sec"},
    "upload_video": {"file_path", "max_frames"},
    "submit_chain": {
        "prompt", "clips", "width", "height", "frame_rate", "seed", "attention_backend",
        "retake_video_id", "retake_window_start_sec", "retake_audio_strength",
    },
    "wait_for_job": {"job_id", "timeout_sec", "poll_interval_sec"},
    "save_job_video": {"job_id", "dest_dir", "no_clobber", "which"},
    "get_job_video_path": {"job_id"},
}
# 渡さないが、存在することだけ確かめる名前（H3 では渡さないのが契約。弾 5 の作業ツリーの版で在るはず）。
EXPECTED_BUT_UNSENT_ARGS: dict[str, set[str]] = {
    "submit_chain": {"retake_head_px", "retake_tail_px", "retake_regenerate_audio"},
}

# metadata.retake から表に出す鍵（全体は summary.json に丸ごと残す）。
RETAKE_PICK_KEYS = (
    "audio_strength", "muxed_original_waveform", "decoded_frames_px", "source_had_audio", "audio_frozen",
    "n_head_v", "n_tail_v", "n_head_a", "n_tail_a", "window_start_frame", "written_frames",
    "upload_has_audio", "source_fps", "resampled",
)


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


def fmt_strength(s: float) -> str:
    """0 → "0"、0.5 → "0.5"、1.0 → "1.0"（ラベル用。1 と 1.0 を区別しない程度に整える）。"""
    if float(s).is_integer():
        return "0" if s == 0 else f"{s:.1f}"
    return f"{s:g}"


def parse_strengths(text: str) -> list[float]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if not (0.0 <= v <= 1.0) or not math.isfinite(v):
            raise argparse.ArgumentTypeError(f"strength は 0〜1 の数値: {part!r}")
        vals.append(v)
    if not vals:
        raise argparse.ArgumentTypeError("strength が 1 つもありません")
    return vals


def ffbin(repo: Path, name: str) -> str:
    local = repo / "tools" / "ffmpeg" / "bin" / f"{name}.exe"
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} が見つかりません（{local} も PATH も）")


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------------------------
# ffprobe／ffmpeg（記録用）
# ---------------------------------------------------------------------------------------------

def probe_media(repo: Path, path: Path, count_frames: bool = False) -> dict:
    """映像の縦横・fps・（count_frames のとき）復号できたフレーム数、音声の有無と長さ。"""
    cmd = [ffbin(repo, "ffprobe"), "-v", "error"]
    if count_frames:
        cmd += ["-count_frames"]
    cmd += ["-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames,nb_read_frames,"
            "duration,sample_rate,channels:format=duration",
            "-of", "json", str(path)]
    r = run_cmd(cmd)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr.strip()[-2000:], "cmd": cmd}
    data = json.loads(r.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    out: dict[str, Any] = {"ok": True, "path": str(path), "format_duration": _f(data.get("format", {}).get("duration"))}
    if v:
        num, _, den = str(v.get("r_frame_rate", "0/1")).partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else None
        out["video"] = {
            "codec": v.get("codec_name"), "width": v.get("width"), "height": v.get("height"), "fps": fps,
            "nb_frames": _i(v.get("nb_frames")), "nb_read_frames": _i(v.get("nb_read_frames")),
            "duration": _f(v.get("duration")),
        }
    else:
        out["video"] = None
    out["audio"] = ({"codec": a.get("codec_name"), "sample_rate": _i(a.get("sample_rate")),
                     "channels": a.get("channels"), "duration": _f(a.get("duration"))} if a else None)
    return out


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def default_canvas(width: int, height: int, long_side: int = LONG_SIDE_DEFAULT) -> tuple[int, int]:
    """素材の縦横比に近い 32 の倍数。長辺を ``long_side`` にし、短辺は比から求めて 32 の倍数へ切り上げる
    （切り上げなので製品の中央切り抜き〔cover_crop〕が削るのは長辺側の端だけ。16:9 → 1280×736）。"""
    if width >= height:
        short = math.ceil(long_side * height / width / DIM_MULTIPLE) * DIM_MULTIPLE
        return long_side, max(DIM_MULTIPLE, short)
    short = math.ceil(long_side * width / height / DIM_MULTIPLE) * DIM_MULTIPLE
    return max(DIM_MULTIPLE, short), long_side


def cut_window_from_material(repo: Path, material: Path, out: Path, start_sec: float, frames: int) -> dict:
    """``_retake_window.mp4`` が無いときの代わり。製品の ``services/video_io.cut_window_mp4`` と同じ規則
    （24fps へ揃えてから ``round(start_sec*24)`` 番目から ``frames`` 枚を select）で映像だけを切り出す。"""
    start = max(0, round(start_sec * H3_FPS))
    end = start + frames - 1
    vf = f"fps={H3_FPS:g},select='between(n\\,{start}\\,{end})',setpts=PTS-STARTPTS"
    cmd = [ffbin(repo, "ffmpeg"), "-hide_banner", "-y", "-i", str(material), "-vf", vf,
           "-frames:v", str(frames), "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
           "-fps_mode", "cfr", "-r", f"{H3_FPS:g}", str(out)]
    out.parent.mkdir(parents=True, exist_ok=True)
    r = run_cmd(cmd)
    return {"ok": r.returncode == 0, "path": str(out), "start_frame": start, "end_frame": end,
            "cmd": cmd, "stderr_tail": r.stderr.strip()[-1500:] if r.returncode != 0 else None}


_PSNR_SUMMARY_RE = re.compile(
    r"PSNR y:(?P<y>\S+) u:(?P<u>\S+) v:(?P<v>\S+) average:(?P<avg>\S+) min:(?P<min>\S+) max:(?P<max>\S+)")


def band_psnr(repo: Path, output: Path, ref: Path, out_range: tuple[int, int], ref_range: tuple[int, int],
              width: int, height: int) -> dict:
    """出力の ``out_range``（[始, 終) のフレーム番号）と参照の ``ref_range`` の PSNR（ffmpeg の psnr フィルタ）。
    参照は製品の cover_crop と同じ形（縦横比を保って覆うまで拡大→中央を切り抜き。LANCZOS）に合わせる。
    製品の拡大は ceil・ffmpeg の拡大は四捨五入なので 1 画素ずれうる（記録のみで判定に使わない）。"""
    o0, o1 = out_range
    r0, r1 = ref_range
    fc = (
        f"[0:v]trim=start_frame={o0}:end_frame={o1},setpts=PTS-STARTPTS,format=yuv420p[o];"
        f"[1:v]trim=start_frame={r0}:end_frame={r1},setpts=PTS-STARTPTS,"
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1,format=yuv420p[r];"
        f"[o][r]psnr=stats_file=-"
    )
    cmd = [ffbin(repo, "ffmpeg"), "-hide_banner", "-nostats", "-i", str(output), "-i", str(ref),
           "-filter_complex", fc, "-f", "null", "-"]
    r = run_cmd(cmd)
    res: dict[str, Any] = {"out_frames": [o0, o1], "ref_frames": [r0, r1], "ok": r.returncode == 0}
    if r.returncode != 0:
        res["error"] = r.stderr.strip()[-2000:]
        res["cmd"] = cmd
        return res
    per_frame = []
    for line in r.stdout.splitlines():
        m = re.search(r"psnr_avg:(\S+)", line)
        if m:
            per_frame.append(_psnr_val(m.group(1)))
    res["frames_compared"] = len(per_frame)
    res["per_frame_psnr_avg"] = per_frame
    m = _PSNR_SUMMARY_RE.search(r.stderr)
    if m:
        res["summary"] = {k: _psnr_val(v) for k, v in m.groupdict().items()}
    finite = [v for v in per_frame if v is not None and math.isfinite(v)]
    res["min_psnr_avg"] = min(finite) if finite else None
    res["identical_frames"] = sum(1 for v in per_frame if v is not None and math.isinf(v))
    return res


def _psnr_val(text: str) -> float | None:
    if text.lower() == "inf":
        return math.inf
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------------
# MCP の呼び出し
# ---------------------------------------------------------------------------------------------

class ToolCallError(Exception):
    """道具がエラーを返した（``isError``）か、呼び出しそのものが失敗した。``text`` は全文。"""

    def __init__(self, tool: str, text: str) -> None:
        super().__init__(f"{tool}: {text}")
        self.tool = tool
        self.text = text


class Mcp:
    def __init__(self, session: ClientSession, calls_log: list) -> None:
        self.session = session
        self.calls_log = calls_log  # 全呼び出しの控え（道具名・引数・秒・成否・エラー全文）

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
        # wait_for_job と backend_status は回数が多いので応答の本体は控えに入れない（条件の記録に残る）
        if tool not in ("wait_for_job", "backend_status"):
            entry["result"] = data
        self.calls_log.append(entry)
        return data


def check_tool_schemas(tools: list) -> dict:
    """tools/list の inputSchema と、台本が渡す引数名を照合する。"""
    by_name = {t.name: t for t in tools}
    report: dict[str, Any] = {"tool_count": len(tools), "tools": sorted(by_name), "problems": [], "warnings": [],
                              "submit_chain_properties": None}
    for name, needed in REQUIRED_TOOL_ARGS.items():
        tool = by_name.get(name)
        if tool is None:
            report["problems"].append(f"道具 {name} が tools/list に無い")
            continue
        props = set((tool.inputSchema or {}).get("properties", {}))
        missing = needed - props
        if missing:
            report["problems"].append(f"{name} の inputSchema に引数 {sorted(missing)} が無い（あるのは {sorted(props)}）")
        if name == "submit_chain":
            report["submit_chain_properties"] = sorted(props)
            report["submit_chain_required"] = (tool.inputSchema or {}).get("required")
    for name, names in EXPECTED_BUT_UNSENT_ARGS.items():
        tool = by_name.get(name)
        if tool is None:
            continue
        props = set((tool.inputSchema or {}).get("properties", {}))
        missing = names - props
        if missing:
            report["warnings"].append(f"{name} に {sorted(missing)} が無い（台本は渡さないので止めない）")
    report["ok"] = not report["problems"]
    return report


# ---------------------------------------------------------------------------------------------
# 段ごとの処理
# ---------------------------------------------------------------------------------------------

async def ensure_pipeline(mcp: Mcp, args, summary: dict) -> bool:
    """H3 を載せる。応答待ちが時間切れなら backend_status で pipeline_loaded を待つ。"""
    load: dict[str, Any] = {"requested": {"base_model": H3_BASE_MODEL, "models": {"transformer": args.transformer},
                                          "wait_sec": 45},
                            "responses": [], "status_polls": 0}
    summary["load"] = load
    t0 = time.monotonic()
    if args.skip_load:
        load["skipped"] = True
    else:
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
        if args.skip_load and not s.get("pipeline_loaded"):
            load["error"] = f"--skip-load だが H3 が載っていない（state={s.get('state')} base_model={s.get('base_model')}）"
            return False
        if args.skip_load and s.get("base_model") != H3_BASE_MODEL:
            load["error"] = f"--skip-load だが選択中のベースモデルが {s.get('base_model')}"
            return False
        if time.monotonic() > deadline:
            load["error"] = f"{args.load_timeout_sec} 秒待っても読み込みが終わらない（最後の状態 {s}）"
            return False
        await asyncio.sleep(10)
    load["seconds"] = round(time.monotonic() - t0, 1)
    log(f"H3 が載った（{load['seconds']} 秒）")
    try:
        models = await mcp.call("list_models")
        h3 = next((b for b in models.get("base_models", []) if b.get("id") == H3_BASE_MODEL), None)
        summary["models_after_load"] = {
            "base_model_entry": h3,
            "transformer_role": (h3 or {}).get("transformer_role"),
            "transformer_weight_class": (h3 or {}).get("transformer_weight_class"),
        }
        log(f"transformer_role={summary['models_after_load']['transformer_role']}")
    except ToolCallError as exc:
        summary["models_after_load"] = {"error": exc.text}
    return True


def build_submit_args(args, video_id: str, strength: float) -> dict:
    """submit_chain に渡す引数。H3 の Retake では retake_head_px／retake_tail_px を渡さない（22 固定・
    サーバーが決める）。retake_regenerate_audio は道具の既定（True）のまま。clips は 1 本で、クリップの
    prompt は渡さない（渡すと共通の prompt の後ろに追記される）。"""
    return {
        "prompt": args.prompt,
        "clips": [{"num_frames": args.frames}],
        "width": args.width,
        "height": args.height,
        "frame_rate": H3_FPS,
        "seed": args.seed,
        "attention_backend": args.attention,
        "retake_video_id": video_id,
        "retake_window_start_sec": args.start_sec,
        "retake_audio_strength": strength,
    }


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
        key = (st.get("status"), st.get("stage"), st.get("clip"), st.get("current_step"))
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


def pick_retake(meta: dict) -> dict:
    rt = meta.get("retake") or {}
    return {k: rt.get(k) for k in RETAKE_PICK_KEYS}


def record_psnr(repo: Path, cond: dict, output: Path, ref: Path, width: int, height: int, window_frames: int) -> None:
    """出力の頭と尾の 22 フレームを、参照（窓）の頭と尾の 22 フレームと比べる。"""
    out_probe = cond.get("output_probe") or {}
    n_out = ((out_probe.get("video") or {}).get("nb_read_frames")) or window_frames
    ref_probe = probe_media(repo, ref, count_frames=True)
    cond["window_ref_probe"] = ref_probe
    n_ref = ((ref_probe.get("video") or {}).get("nb_read_frames")) or window_frames
    g = GLUE_FRAMES
    cond["psnr"] = {
        "band_frames": g,
        "output_frames": n_out,
        "ref_frames": n_ref,
        "head": band_psnr(repo, output, ref, (0, g), (0, g), width, height),
        # 尾は各自の最後の 22 枚（フレーム数が窓と違ったときも端どうしで揃える。違いは output_frames で分かる）
        "tail": band_psnr(repo, output, ref, (n_out - g, n_out), (n_ref - g, n_ref), width, height),
    }


async def run_condition(mcp: Mcp, args, repo: Path, out_dir: Path, video_id: str, strength: float,
                        label: str, shared_ref: dict) -> dict:
    cond_dir = out_dir / label
    cond_dir.mkdir(parents=True, exist_ok=True)
    tool_args = build_submit_args(args, video_id, strength)
    cond: dict[str, Any] = {"label": label, "strength": strength, "dir": str(cond_dir), "started_at": now_iso(),
                            "tool_args": tool_args, "status": None, "error": None}
    write_json(cond_dir / "request.json", {"tool": "submit_chain", "arguments": tool_args,
                                           "note": "MCP の道具に渡した引数。サーバーが受けた本文（書き戻し後）は "
                                                   "metadata.json の request"})
    log(f"=== 条件 {label}（retake_audio_strength={strength}）===")
    try:
        sub = await mcp.call("submit_chain", tool_args)
    except ToolCallError as exc:
        cond.update(status="submit_error", error=exc.text, finished_at=now_iso())
        log(f"submit_chain が拒否: {exc.text}")
        return cond
    cond["submit_response"] = sub
    job_id = sub.get("job_id")
    cond["job_id"] = job_id
    log(f"job_id={job_id}")
    waited = await wait_job(mcp, job_id, args.job_timeout_sec)
    final = waited.pop("final") or {}
    cond["wait"] = waited
    cond["job_final"] = final
    cond["status"] = final.get("status") or "unknown"
    if waited.get("gave_up"):
        cond.update(status="wait_gave_up", error=waited["gave_up"], finished_at=now_iso())
        return cond
    if cond["status"] != "completed":
        cond["error"] = final.get("error")
        cond["finished_at"] = now_iso()
        log(f"ジョブが {cond['status']}: {final.get('error')}")
        return cond

    # 保存と写し
    try:
        cond["saved"] = await mcp.call("save_job_video", {"job_id": job_id, "dest_dir": str(cond_dir),
                                                          "no_clobber": True, "which": "output"})
        saved_path = Path(cond["saved"]["saved_path"])
    except ToolCallError as exc:
        cond.update(error=f"save_job_video: {exc.text}", finished_at=now_iso())
        return cond
    job_dir = None
    try:
        vp = await mcp.call("get_job_video_path", {"job_id": job_id})
        cond["job_video_path"] = vp
        job_dir = Path(vp["path"]).parent
    except ToolCallError as exc:
        cond["job_video_path"] = {"error": exc.text}
    meta: dict = {}
    if job_dir is not None and (job_dir / "metadata.json").exists():
        shutil.copy2(job_dir / "metadata.json", cond_dir / "metadata.json")
        meta = json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))
        cond["metadata_path"] = str(job_dir / "metadata.json")
    else:
        cond["metadata_path"] = None
        cond["metadata_missing"] = f"{job_dir}\\metadata.json が無い"
    cond["metadata_summary"] = {k: meta.get(k) for k in (
        "backend", "attention_used", "vram16_mode_used", "generation_time_seconds", "seed", "models")}
    cond["metadata_retake"] = meta.get("retake")
    cond["retake_picks"] = pick_retake(meta)

    # 記録（判定は付けない）
    cond["output_probe"] = probe_media(repo, saved_path, count_frames=True)
    window = job_dir / "_retake_window.mp4" if job_dir is not None else None
    if window is not None and window.exists():
        ref, ref_kind = window, "job_retake_window"
    else:
        if "fallback" not in shared_ref:
            shared_ref["fallback"] = cut_window_from_material(
                repo, Path(args.material), out_dir / "_window_ref" / "window_from_material.mp4",
                args.start_sec, args.frames)
        fb = shared_ref["fallback"]
        ref, ref_kind = (Path(fb["path"]), "material_cut_by_script") if fb.get("ok") else (None, "none")
        cond["window_fallback"] = fb
    cond["window_ref"] = {"kind": ref_kind, "path": str(ref) if ref else None}
    if ref is not None:
        try:
            record_psnr(repo, cond, saved_path, ref, args.width, args.height, args.frames)
        except Exception as exc:  # noqa: BLE001 -- 記録の失敗で門を止めない
            cond["psnr"] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    cond["finished_at"] = now_iso()
    log(f"完了: 出力 {((cond['output_probe'].get('video') or {}).get('nb_read_frames'))} フレーム・"
        f"muxed_original_waveform={cond['retake_picks'].get('muxed_original_waveform')}")
    return cond


# ---------------------------------------------------------------------------------------------
# まとめ
# ---------------------------------------------------------------------------------------------

def _p(x) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        if math.isinf(x):
            return "inf"
        return f"{x:.2f}"
    return str(x)


def write_summary_md(path: Path, summary: dict) -> None:
    g = summary.get("geometry", {})
    lines = [
        "# MiniMax H3 の Retake の GPU の門（記録）",
        "",
        f"- 開始: {summary.get('started_at')}／終了: {summary.get('finished_at')}",
        f"- 素材: `{summary.get('material', {}).get('path')}`",
        f"- 窓: 開始 {g.get('start_sec')} 秒（{g.get('start_frame')} フレーム目）・{g.get('frames')} フレーム・"
        f"{g.get('width')}×{g.get('height')}・種 {g.get('seed')}・注意の計算 {g.get('attention')}",
        f"- transformer: `{summary.get('args', {}).get('transformer')}`（transformer_role="
        f"{(summary.get('models_after_load') or {}).get('transformer_role')}）",
        "- 判定は付けない（PSNR は記録のみ）。数値の正本は記録簿 §200。",
        "",
        "| 条件 | strength | 状態 | job_id | 秒 | 出力フレーム | muxed_original_waveform | decoded_frames_px | "
        "n_tail_a | 音声の有無（元） | 頭 PSNR 平均／最小 | 尾 PSNR 平均／最小 | 参照 | エラー |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in summary.get("conditions", []):
        rp = c.get("retake_picks") or {}
        ps = c.get("psnr") or {}
        head = ps.get("head") or {}
        tail = ps.get("tail") or {}
        out_frames = ((c.get("output_probe") or {}).get("video") or {}).get("nb_read_frames")
        err = (c.get("error") or "").replace("|", "\\|").replace("\n", " ")
        if len(err) > 160:
            err = err[:160] + "…（全文は summary.json）"
        lines.append(
            f"| {c.get('label')} | {c.get('strength')} | {c.get('status')} | `{(c.get('job_id') or '–')}` | "
            f"{_p((c.get('wait') or {}).get('seconds'))} | {_p(out_frames)} | {_p(rp.get('muxed_original_waveform'))} | "
            f"{_p(rp.get('decoded_frames_px'))} | {_p(rp.get('n_tail_a'))} | {_p(rp.get('source_had_audio'))} | "
            f"{_p((head.get('summary') or {}).get('avg'))}／{_p(head.get('min_psnr_avg'))} | "
            f"{_p((tail.get('summary') or {}).get('avg'))}／{_p(tail.get('min_psnr_avg'))} | "
            f"{(c.get('window_ref') or {}).get('kind', '–')} | {err or '–'} |"
        )
    lines += ["", "欄の意味は台本の README.md「JSON の欄」。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def flush(out_dir: Path, summary: dict) -> None:
    write_json(out_dir / "summary.json", summary)
    write_summary_md(out_dir / "summary.md", summary)


# ---------------------------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------------------------

def labels_for(args) -> list[tuple[float, str]]:
    out = []
    for s in args.strengths:
        tag = f"s{fmt_strength(s)}"
        if args.label is None:
            out.append((s, tag))
        elif len(args.strengths) == 1:
            out.append((s, args.label))
        else:
            out.append((s, f"{args.label}-{tag}"))
    names = [n for _, n in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"条件の名前が重複: {names}")
    return out


async def run(args, summary: dict) -> int:
    repo = Path(args.repo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stderr_log = out_dir / "mcp_server.stderr.log"
    env = {**os.environ, "PYTHONUTF8": "1", "LTX_MCP_BASE_URL": args.base_url}
    server = StdioServerParameters(command=args.mcp_python, args=["-m", "mcp_server"], cwd=str(repo), env=env,
                                   encoding="utf-8")
    summary["mcp"] = {"python": args.mcp_python, "cwd": str(repo), "LTX_MCP_BASE_URL": args.base_url,
                      "stderr_log": str(stderr_log), "calls": []}
    calls = summary["mcp"]["calls"]
    exit_code = 0
    with open(stderr_log, "a", encoding="utf-8") as errlog:
        errlog.write(f"\n===== {now_iso()} h3_retake_gate.py =====\n")
        errlog.flush()
        async with stdio_client(server, errlog=errlog) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=TOOL_READ_TIMEOUT) as session:
                init = await session.initialize()
                summary["mcp"]["server_info"] = {"name": init.serverInfo.name, "version": init.serverInfo.version,
                                                 "protocol": init.protocolVersion}
                tools = (await session.list_tools()).tools
                schema = check_tool_schemas(tools)
                summary["tool_schema_check"] = schema
                flush(out_dir, summary)
                if not schema["ok"]:
                    for p in schema["problems"]:
                        log(f"引数名の照合に失敗: {p}")
                    return 1
                for w in schema["warnings"]:
                    log(f"注意: {w}")
                mcp = Mcp(session, calls)

                # (1) 到達
                st = await mcp.call("backend_status")
                summary["status_before"] = st
                if not st.get("reachable"):
                    log(f"バックエンドに届かない: {st.get('error')}")
                    return 1
                log(f"到達: base_url={st.get('base_url')} state={(st.get('status') or {}).get('state')} "
                    f"base_model={(st.get('status') or {}).get('base_model')}")

                # (2) 読み込み
                if not await ensure_pipeline(mcp, args, summary):
                    log(f"読み込みで止まる: {summary['load'].get('error')}")
                    flush(out_dir, summary)
                    return 1
                flush(out_dir, summary)

                # (3) 素材のアップロード（1 回だけ）
                try:
                    up = await mcp.call("upload_video", {"file_path": str(Path(args.material).resolve()),
                                                         "max_frames": UPLOAD_MAX_FRAMES})
                except ToolCallError as exc:
                    summary["upload"] = {"error": exc.text}
                    log(f"upload_video が失敗: {exc.text}")
                    flush(out_dir, summary)
                    return 1
                summary["upload"] = up
                log(f"upload: video_id={up.get('video_id')} frame_count={up.get('frame_count')} fps={up.get('fps')}")
                fc, fps = up.get("frame_count"), up.get("fps")
                if fc and fps:
                    need = args.start_sec + args.frames / H3_FPS
                    have = fc / fps
                    summary["geometry"]["upload_seconds"] = have
                    summary["geometry"]["window_end_sec"] = need
                    if need > have + 1e-6:
                        log(f"注意: 窓の終わり {need:.3f} 秒が素材の尺 {have:.3f} 秒を超える（サーバーが断るはず）")
                flush(out_dir, summary)

                # (4)(5) 条件ごと
                shared_ref: dict = {}
                for strength, label in labels_for(args):
                    try:
                        cond = await run_condition(mcp, args, repo, out_dir, up["video_id"], strength, label,
                                                   shared_ref)
                    except Exception as exc:  # noqa: BLE001 -- 1 条件の例外で残りを止めない
                        cond = {"label": label, "strength": strength, "status": "script_error",
                                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                    summary["conditions"].append(cond)
                    if cond.get("status") != "completed" or cond.get("error"):
                        exit_code = 3
                    flush(out_dir, summary)
                    if cond.get("status") == "wait_gave_up":
                        log("待ちきれなかったジョブがバックエンドで走っているため、残りの条件は送らない（JOB_BUSY になる）")
                        break
    return exit_code


def dry_run(args, summary: dict) -> int:
    print(json.dumps({
        "base_url": args.base_url,
        "mcp_server": {"command": args.mcp_python, "args": ["-m", "mcp_server"], "cwd": args.repo},
        "load_pipeline": None if args.skip_load else {"base_model": H3_BASE_MODEL,
                                                      "models": {"transformer": args.transformer}, "wait_sec": 45},
        "upload_video": {"file_path": str(Path(args.material).resolve()), "max_frames": UPLOAD_MAX_FRAMES},
        "geometry": summary["geometry"],
        "submit_chain": {label: build_submit_args(args, "<upload_video の video_id>", s)
                         for s, label in labels_for(args)},
        "out_dir": args.out_dir,
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="MiniMax H3 の Retake の GPU の門（MCP 経由）")
    ap.add_argument("--material", required=True, help="話し声のある素材の mp4（MCP サーバーのマシン上のパス）")
    ap.add_argument("--start-sec", type=float, required=True, help="窓の開始秒（0 以上）")
    ap.add_argument("--frames", type=int, default=124, help="窓のフレーム数（17n+5・73〜481。既定 124）")
    ap.add_argument("--width", type=int, default=None, help="幅（32 の倍数。既定は README「解像度の既定」）")
    ap.add_argument("--height", type=int, default=None, help="高さ（32 の倍数）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt", default="A person talking to the camera, natural lip movement, steady shot.")
    ap.add_argument("--strengths", type=parse_strengths, default=parse_strengths("0,0.5,1.0"),
                    help="retake_audio_strength の並び（カンマ区切り。既定 0,0.5,1.0）")
    ap.add_argument("--base-url", default=os.environ.get("LTX_MCP_BASE_URL") or BASE_URL_DEFAULT,
                    help="バックエンドの URL（MCP サーバーへ LTX_MCP_BASE_URL として渡す）")
    ap.add_argument("--transformer", default="default", help="H3 の transformer の登録名（既定 default＝FL2VA）")
    ap.add_argument("--attention", default="sdpa", choices=["sdpa", "sage"], help="attention_backend（既定 sdpa）")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--label", default=None, help="条件の名前（既定 s<strength>。strength が複数なら接頭辞）")
    ap.add_argument("--skip-load", action="store_true", help="load_pipeline を呼ばない（既に H3 が載っているとき）")
    ap.add_argument("--dry-run", action="store_true", help="送る引数を表示して終わる（MCP サーバーを起こさない）")
    ap.add_argument("--load-timeout-sec", type=float, default=900.0, help="読み込みを待つ上限（既定 900 秒）")
    ap.add_argument("--job-timeout-sec", type=float, default=3600.0, help="1 条件のジョブを待つ上限（既定 3600 秒）")
    ap.add_argument("--repo", default=str(REPO_DEFAULT), help="Nz-Videomni のリポジトリ直下（MCP サーバーの cwd）")
    ap.add_argument("--mcp-python", default=None,
                    help="MCP サーバーを起こす python（既定 <repo>\\.venv\\Scripts\\python.exe）")
    args = ap.parse_args(argv)
    if args.start_sec < 0:
        ap.error("--start-sec は 0 以上")
    if (args.width is None) != (args.height is None):
        ap.error("--width と --height は両方指定するか両方省略する")
    if args.mcp_python is None:
        args.mcp_python = str(Path(args.repo) / ".venv" / "Scripts" / "python.exe")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args(argv)
    repo = Path(args.repo)
    material = Path(args.material)
    if not material.exists():
        print(f"素材が無い: {material}", file=sys.stderr)
        return 1
    mat_probe = probe_media(repo, material)
    mv = mat_probe.get("video") or {}
    if args.width is None:
        if not mv.get("width") or not mv.get("height"):
            print(f"素材の縦横が取れない（--width／--height を指定）: {mat_probe}", file=sys.stderr)
            return 1
        args.width, args.height = default_canvas(int(mv["width"]), int(mv["height"]))
        size_source = f"素材 {mv['width']}×{mv['height']} から長辺 {LONG_SIDE_DEFAULT}・32 の倍数へ切り上げ"
    else:
        size_source = "引数"
    warnings = []
    if args.width % DIM_MULTIPLE or args.height % DIM_MULTIPLE:
        warnings.append(f"{args.width}×{args.height} は 32 の倍数でない（サーバーが断るはず）")
    if (args.frames - 5) % 17 or not (73 <= args.frames <= 481):
        warnings.append(f"--frames {args.frames} は 17n+5・73〜481 でない（サーバーが断るはず）")
    if not mat_probe.get("audio"):
        warnings.append("素材に音声が無い（s>0 の条件は断られうる。無音の素材の門は s=0 だけで回す）")
    summary: dict[str, Any] = {
        "script": "h3_retake_gate.py",
        "started_at": now_iso(),
        "args": {k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items()},
        "material": {"path": str(material.resolve()), "probe": mat_probe},
        "geometry": {"start_sec": args.start_sec, "start_frame": max(0, round(args.start_sec * H3_FPS)),
                     "frames": args.frames, "width": args.width, "height": args.height, "size_source": size_source,
                     "seed": args.seed, "attention": args.attention, "fps": H3_FPS, "glue_frames": GLUE_FRAMES},
        "warnings": warnings,
        "conditions": [],
    }
    for w in warnings:
        log(f"注意: {w}")
    if args.dry_run:
        return dry_run(args, summary)
    out_dir = Path(args.out_dir)
    code = 1
    try:
        code = asyncio.run(run(args, summary))
    except BaseException as exc:  # noqa: BLE001 -- 例外も中断も記録して終わる
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        log(f"止まった: {summary['error']}")
        if isinstance(exc, KeyboardInterrupt):
            log("注意: 送ったジョブはバックエンドで走り続ける（連結には協調的な停止が無い）")
        code = 1
    finally:
        summary["finished_at"] = now_iso()
        summary["exit_code"] = code
        flush(out_dir, summary)
        log(f"記録: {out_dir / 'summary.json'}・{out_dir / 'summary.md'}（終了コード {code}）")
    return code


if __name__ == "__main__":
    sys.exit(main())
