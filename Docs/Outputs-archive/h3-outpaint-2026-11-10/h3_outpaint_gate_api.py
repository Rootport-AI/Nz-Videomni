r"""MiniMax H3 の Outpainting（台帳 §1-125）弾 2 の GPU の門 G5（快適上限の点）と G6（20 ステップの目視用）を、
API 経由（MCP サーバー → バックエンド → エンジン）で回す台本。

計画: ``rosy-jingling-stonebraker.md`` §2.6・§3「弾 2」・§7。説明書は同じフォルダの README.md の「弾 2」の節。
製品コードは import しない（変えもしない）。部品は前例の台本から写した:
  - MCP の stdio クライアント・引数名の照合・読み込みの待ち・ジョブの待ち:
    ``Docs/Outputs-archive/h3-retake-2026-11-08/h3_retake_gate.py``
  - 一時 yaml・一時 state・別ポートのサーバーの起動と停止（``_rt``）:
    ``Docs/Outputs-archive/h3-vsf-calib-2026-11-09/h3_vsf_calib.py``
  - 素材の切り出しの作法と、保持矩形の PSNR・境界の勾配比（判定の関数そのもの）:
    同じフォルダの ``h3_outpaint_gate.py``（弾 1）。G6 の測定だけはエンジンの仮想環境の子プロセスで、
    弾 1 の ``measure_run`` をそのまま呼ぶ（``--measure-helper``）。
  - 退避の窓（WDDM の共有メモリの中央値）の測り方: 較正台 ``outputs/comfort-calib-h3-chain-2026-11-08``
    の ``vram_sampler2.ps1`` と ``calib.py`` の server.log の区切り方（快適上限表 §17.2 の規則 v3″）。

使い方（リポジトリの根で。アプリの仮想環境の python）:

    set PYTHONUTF8=1
    .venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate_api.py --prepare
    .venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate_api.py --gate G5 --frames 345 --dry-run
    .venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate_api.py --gate G5 --frames 345
    .venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate_api.py --gate G6

流れ（--prepare・--dry-run 以外）:
  1. 素材が無ければ止まる（先に --prepare）。
  2. ``config.yaml`` は読むだけで、5 つの鍵を ``outputs\h3-outpaint-gates-2026-11-10\_rt\`` に向けた一時 yaml と、
     MiniMax H3 を選んだ一時 state を書き、別ポート（既定 18699）で実バックエンドを起こす。
  3. MCP サーバー（``.venv\Scripts\python.exe -m mcp_server``）を stdio の子プロセスで起こし、
     ``initialize → tools/list`` で引数名を照合してから ``tools/call``。
  4. ``load_pipeline(MiniMaxH3, transformer=default)``（``--skip-load`` なら呼ばず、載っていることだけ確かめる）
     → ``upload_video`` → ``submit_generate``（``outpaint_pad_*``・``outpaint_audio_strength``・``reference_video_id``。
     なじませは送らない＝サーバーの既定）→ ``wait_for_job`` → ``save_job_video`` → ``metadata.json`` の写し。
  5. G5: ジョブの間 WDDM の共有メモリを標本化し、server.log でノイズ除去と復号の窓を区切って記録し、
     §17.2 の規則で判定する。G6: 保持矩形の PSNR・境界の勾配比（弾 1 と同じ量）と ``h3.outpaint`` の 12 鍵。
  6. ``--keep-server`` でなければ ``unload_pipeline`` → サーバーをプロセスの木ごと止める。
  7. ``<label>\request.json``・``metadata.json``・``summary_row.json`` と、``summary_api.json``・``summary_api.md``。

終了コード: 0＝回した本が completed（G5 の OOM は門の記録として 0 に数える。下の「判定」）、
3＝ジョブが OOM 以外で失敗・送信の拒否・待ちきれない・測定の失敗、1＝準備の段で止まった。
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
REPO = HERE.parents[2]  # Docs\Outputs-archive\<このフォルダ> の 3 つ上＝リポジトリ直下
PY = REPO / ".venv" / "Scripts" / "python.exe"
ENGINE_PY = REPO / ".venv-engine-h3" / "Scripts" / "python.exe"
OUT_ROOT = REPO / "outputs" / "h3-outpaint-gates-2026-11-10"
MAT_DIR = OUT_ROOT / "_material"
RT = OUT_ROOT / "_rt"
PORT_DEFAULT = 18699
TEMP_CONFIG_NAME = "config.h3outpaintapi.yaml"
TEMP_STATE_NAME = "state.h3outpaintapi.json"
SUMMARY_JSON = OUT_ROOT / "summary_api.json"
SUMMARY_MD = OUT_ROOT / "summary_api.md"

H3_BASE_MODEL = "MiniMaxH3"
H3_CATEGORIES = ("transformer", "text_encoder", "video_vae", "audio")
H3_FPS = 24.0
TOKEN_PX = 32

TOOL_READ_TIMEOUT = timedelta(seconds=360)
WAIT_TIMEOUT_SEC = 45
WAIT_POLL_SEC = 5.0
MAX_CONSECUTIVE_TRANSPORT_ERRORS = 12
UPLOAD_MAX_FRAMES = 100000  # 尺を測るためだけの値（素材の枚数より確実に大きい。切り詰めは起きない）

# --- G5（快適上限表 §17.1 の条件に合わせる。ステップ数だけ 4〔§17.1 の ① と同じ扱い〕） ---
G5_SOURCE = REPO / "outputs" / "h3-retake-gates-2026-11-08" / "_material" / "H3-Ref-1080p-30s.mp4"
G5_SRC_W, G5_SRC_H = 1280, 720
G5_PADS = {"left": 320, "right": 320, "top": 184, "bottom": 184}   # 1280x720 -> 1920x1088
G5_FRAMES_DEFAULT = (345, 362)
G5_BASE_FRAMES = 56  # 規則 v3″ の基準点（較正の b1080 と同じ長さ）。退避の判定の比べる相手
#: 較正の文（outputs/h3-vram16-gates/f0_prompt.txt。text の行は 30。快適上限表 §19 の「プロンプト 30」）
G5_PROMPT = "コスプレイヤーの日本人女性が、こちらに向かって歩いてくる。彼女はこちらに片手を伸ばして微笑む。"
G5_PROMPT_ROWS = 30
G5_SEED = 12345
G5_STEPS = 4
G5_ATTENTION = "sage"  # §17.1「attention は利用者の既定の sage」
G5_AUDIO_STRENGTH = 0.5
SINGLE_ROWS_BUDGET = 219516  # scripts/manifests/40-minimaxh3.json の comfort.single_rows_budget（§17.5 (a)）

# --- G6 ---
G6_SOURCE = REPO / "outputs" / "h3-vsf-calib-2026-11-09" / "hf-round12" / "L_s0_1344x768_seed42.mp4"
G6_MATERIAL_NAME = "G6_src_1344x768_124f.mp4"
G6_SRC_W, G6_SRC_H, G6_FRAMES = 1344, 768, 124
G6_PADS = {"left": 288, "right": 288, "top": 160, "bottom": 160}   # 1344x768 -> 1920x1088
G6_LABEL = "G6_20steps"
G6_SEED = 1234
G6_STEPS = 20
G6_ATTENTION = "sage"
G6_AUDIO_STRENGTH = 0.5
#: 弾 1 の既定の文（h3_outpaint_gate.py の DEFAULT_PROMPT）。同じ素材・同じ pad・同じ種の E2c_r1（Turbo 4 ステップ）と
#: 比べられるように同じ文にした。--prompt で差し替えられる。
G6_PROMPT = (
    "A wide shot of a young woman in a black-and-white maid costume talking cheerfully to the camera in a "
    "spacious pastel room decorated with sweets and flowers; natural head and hand movement, steady camera."
)

OUTPAINT_KEYS = ("canvas", "pads", "source", "aligned", "fixed_tokens", "generated_tokens", "feather_px",
                 "audio_strength", "source_had_audio", "audio_latents_encoded", "audio_output", "inject_clean")

# 規則 v3″（快適上限表 §17.2）
SHARED_MARGIN_MB = 300.0
MIN_WINDOW_SAMPLES = 10
JUDGED_WINDOWS = ("denoise", "decode")
STAGE_LABELS = {"encode": "text encode", "denoise": "stage-1 denoise", "decode": "VAE decode"}

# tools/list の inputSchema で照合する引数名。台本が実際に渡す名前だけ（無ければ止まる）。
REQUIRED_TOOL_ARGS: dict[str, set[str]] = {
    "backend_status": set(),
    "load_pipeline": {"base_model", "models", "wait_sec"},
    "unload_pipeline": set(),
    "list_models": set(),
    "upload_video": {"file_path", "max_frames"},
    "submit_generate": {
        "prompt", "width", "height", "num_frames", "frame_rate", "seed", "attention_backend",
        "num_inference_steps", "reference_video_id", "outpaint_pad_left", "outpaint_pad_right",
        "outpaint_pad_top", "outpaint_pad_bottom", "outpaint_audio_strength",
    },
    "wait_for_job": {"job_id", "timeout_sec", "poll_interval_sec"},
    "save_job_video": {"job_id", "dest_dir", "no_clobber", "which"},
    "get_job_video_path": {"job_id"},
}
#: 送らないが、在ることだけ確かめる名前（弾 2 の MCP の版で在るはず。無くても止めない）。
EXPECTED_BUT_UNSENT_ARGS = {"submit_generate": {"outpaint_blend_dilation_stage1", "outpaint_freeze_source_audio",
                                                "loras", "vram16_mode", "keep_resident"}}


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


def tool(name: str) -> str:
    local = REPO / "tools" / "ffmpeg" / "bin" / f"{name}.exe"
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} が見つかりません（<repo>\\tools\\ffmpeg\\bin か PATH）")
    return found


def run_cmd(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def ffprobe(path: Path) -> dict:
    """映像（枚数は -count_frames の実測）と音声の要点（弾 1 の ffprobe と同じ形）。"""
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


def latent_frames(n: int) -> int:
    """17n+5 のフレーム数の潜在フレーム数（快適上限表 §17.3: (F−5)÷17×5＋2）。"""
    if n < 22 or (n - 5) % 17:
        raise SystemExit(f"フレーム数 {n} は 17n+5（22 以上）ではありません")
    return (n - 5) // 17 * 5 + 2


def predicted_rows(width: int, height: int, frames: int, prompt_rows: int | None) -> dict:
    """S（DiT に入る列の総行数）の予測（快適上限表 §17.3）。Outpainting は cond 0・キーフレームなし・参照なし。
    固定行は映像の行の中のマスクで、行を増やさない（計画 §8・快適上限表 §19.3）。"""
    gh, gw = height // TOKEN_PX, width // TOKEN_PX
    video = gh * gw * latent_frames(frames)
    audio = 2 * round(frames / 24 * 40)
    out = {"video": video, "cond": 0, "audio": audio, "text": prompt_rows,
           "total": None if prompt_rows is None else video + audio + prompt_rows}
    if out["total"] is not None:
        out["vs_single_rows_budget"] = out["total"] - SINGLE_ROWS_BUDGET
    return out


# ---------------------------------------------------------------------------------------------
# 素材（--prepare。CPU・ffmpeg）
# ---------------------------------------------------------------------------------------------

def g5_material_name(frames: int) -> str:
    return f"G5_src_{G5_SRC_W}x{G5_SRC_H}_{frames}f.mp4"


def prepare_g5(frames: int, record: dict) -> dict:
    """G5 の素材: 元の 1080p（24fps・720 枚・音声あり）の先頭から ``frames`` 枚を 1280x720 へ縮めて切り出す。
    作法は弾 1 の ``_prepare_main`` と同じ（select の窓・atrim・lanczos・libx264 crf 12・yuv420p・24fps CFR・
    音声は PCM、断られたら AAC 192k）。枚数・寸法・音声を ffprobe で確かめ、合わなければ止まる。"""
    out = MAT_DIR / g5_material_name(frames)
    key = f"g5:{frames}"
    if out.is_file() and key in record:
        log(f"prepare: 既存の {out.name} を使う（作り直さない）")
        return record[key]
    if not G5_SOURCE.is_file():
        raise SystemExit(f"G5 の元の素材がありません: {G5_SOURCE}")
    sprobe = ffprobe(G5_SOURCE)
    have = (sprobe["video"] or {}).get("nb_read_frames") or 0
    if frames > have:
        raise SystemExit(f"G5 の {frames} 枚は元の素材の {have} 枚を超えます")
    sfps = (sprobe["video"] or {}).get("fps")
    start, end = 0, frames - 1
    vchain = ([f"fps={H3_FPS:g}"] if sfps is None or abs(sfps - H3_FPS) > 1e-3 else []) + [
        f"select='between(n\\,{start}\\,{end})'", "setpts=PTS-STARTPTS", f"scale={G5_SRC_W}:{G5_SRC_H}:flags=lanczos"]
    fc = ("[0:v]" + ",".join(vchain) + "[v];"
          f"[0:a]atrim=start={start / H3_FPS}:end={(end + 1) / H3_FPS},asetpts=PTS-STARTPTS[a]")

    def cut(audio_args):
        cmd = [tool("ffmpeg"), "-y", "-v", "error", "-i", str(G5_SOURCE), "-filter_complex", fc,
               "-map", "[v]", "-map", "[a]", *audio_args,
               "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", f"{H3_FPS:g}",
               str(out)]
        return cmd, run_cmd(cmd)

    MAT_DIR.mkdir(parents=True, exist_ok=True)
    cmd, proc = cut(["-c:a", "pcm_s16le"])
    audio_codec = "pcm_s16le"
    if proc.returncode != 0:
        cmd, proc = cut(["-c:a", "aac", "-b:a", "192k"])
        audio_codec = "aac 192k"
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg の G5 の切り出しに失敗しました（{proc.returncode}）: {proc.stderr[-2000:]}")
    probe = ffprobe(out)
    vid = probe["video"]
    if vid.get("nb_read_frames") != frames or (vid.get("width"), vid.get("height")) != (G5_SRC_W, G5_SRC_H):
        raise SystemExit(f"G5 の切り出しの枚数か寸法が合いません（{vid.get('nb_read_frames')} 枚・"
                         f"{vid.get('width')}x{vid.get('height')}。{frames} 枚・{G5_SRC_W}x{G5_SRC_H} が必要）")
    if not probe.get("audio"):
        raise SystemExit(f"G5 の切り出しに音声がありません: {out}")
    entry = {"material": out.name, "source": str(G5_SOURCE), "source_fps": sfps, "frames": frames,
             "start_frame": start, "end_frame": end, "ffmpeg_cmd": cmd, "audio_codec": audio_codec, "probe": probe}
    record[key] = entry
    log(f"prepare: {out.name} {vid.get('width')}x{vid.get('height')} {vid.get('nb_read_frames')} 枚・"
        f"音声 {probe['audio'].get('codec_name')} {probe['audio'].get('sample_rate')}Hz")
    return entry


def prepare_g6(record: dict) -> dict:
    """G6 の素材: hf-round12 の L_s0（1344x768・124 枚・24fps・AAC 32kHz）を ``_material`` へそのまま写す
    （再符号化しない。サーバーが ``_outpaint_source.mp4`` へ切り出し直す）。枚数・寸法・fps・音声を確かめる。"""
    out = MAT_DIR / G6_MATERIAL_NAME
    if out.is_file() and "g6" in record:
        log(f"prepare: 既存の {out.name} を使う（作り直さない）")
        return record["g6"]
    if not G6_SOURCE.is_file():
        raise SystemExit(f"G6 の元の素材がありません: {G6_SOURCE}")
    MAT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(G6_SOURCE, out)
    probe = ffprobe(out)
    vid = probe["video"]
    if (vid.get("nb_read_frames") != G6_FRAMES or (vid.get("width"), vid.get("height")) != (G6_SRC_W, G6_SRC_H)
            or vid.get("fps") is None or abs(vid["fps"] - H3_FPS) > 1e-3 or not probe.get("audio")):
        raise SystemExit(f"G6 の素材が想定（{G6_FRAMES} 枚・{G6_SRC_W}x{G6_SRC_H}・24fps・音声あり）と違います: {probe}")
    entry = {"material": out.name, "source": str(G6_SOURCE), "copied": True, "probe": probe}
    record["g6"] = entry
    log(f"prepare: {out.name} {vid.get('width')}x{vid.get('height')} {vid.get('nb_read_frames')} 枚・"
        f"音声 {probe['audio'].get('codec_name')} {probe['audio'].get('sample_rate')}Hz（写しただけ）")
    return entry


def load_prepare_record() -> dict:
    p = MAT_DIR / "prepare_api.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


def do_prepare(frames_list: list[int]) -> dict:
    record = load_prepare_record()
    for f in frames_list:
        prepare_g5(f, record)
    prepare_g6(record)
    write_json(MAT_DIR / "prepare_api.json", record)
    return record


# ---------------------------------------------------------------------------------------------
# 本の組み立て
# ---------------------------------------------------------------------------------------------

def build_job(args) -> dict:
    """1 回の起動で回す 1 本。``tool_args`` が submit_generate に渡す引数（reference_video_id は送る直前に入れる）。"""
    if args.gate == "G5":
        frames = int(args.frames)
        latent_frames(frames)
        pads, (sw, sh) = G5_PADS, (G5_SRC_W, G5_SRC_H)
        label = args.label or f"G5_{frames}f"
        material = MAT_DIR / g5_material_name(frames)
        prompt = args.prompt or G5_PROMPT
        prompt_rows = G5_PROMPT_ROWS if args.prompt is None else None
        seed, steps, attention, strength = G5_SEED, G5_STEPS, G5_ATTENTION, G5_AUDIO_STRENGTH
        purpose = (f"快適上限の点（1920x1088・{frames} 枚・4 ステップ・Turbo なし・{attention}）" if frames != G5_BASE_FRAMES
                   else "規則 v3″ の基準点（退避の判定の比べる相手）")
    else:
        frames = G6_FRAMES
        pads, (sw, sh) = G6_PADS, (G6_SRC_W, G6_SRC_H)
        label = args.label or G6_LABEL
        material = MAT_DIR / G6_MATERIAL_NAME
        prompt = args.prompt or G6_PROMPT
        prompt_rows = None
        seed, steps, attention, strength = G6_SEED, G6_STEPS, G6_ATTENTION, G6_AUDIO_STRENGTH
        purpose = "20 ステップ・Turbo なしの完走（目視用）"
    width = sw + pads["left"] + pads["right"]
    height = sh + pads["top"] + pads["bottom"]
    if width % TOKEN_PX or height % TOKEN_PX:
        raise SystemExit(f"キャンバス {width}x{height} が {TOKEN_PX} の倍数ではありません")
    tool_args = {
        "prompt": prompt, "width": width, "height": height, "num_frames": frames, "frame_rate": H3_FPS,
        "seed": seed, "attention_backend": attention, "num_inference_steps": steps,
        "outpaint_pad_left": pads["left"], "outpaint_pad_right": pads["right"],
        "outpaint_pad_top": pads["top"], "outpaint_pad_bottom": pads["bottom"],
        "outpaint_audio_strength": strength,
        # 送らないもの: loras（Turbo なし）・outpaint_blend_dilation_stage1（サーバーの既定）・
        # outpaint_freeze_source_audio（H3 では 422）・vram16_mode（サーバーの既定 on）・keep_resident（既定 off）・
        # block_swap_prefetch（既定 on）。shift も既定（12／3）。
    }
    return {"gate": args.gate, "label": label, "purpose": purpose, "frames": frames, "source_size": [sw, sh],
            "canvas": [width, height], "pads": dict(pads), "material": str(material), "tool_args": tool_args,
            "predicted_rows": predicted_rows(width, height, frames, prompt_rows),
            "prompt_rows_note": ("較正の文（text 30 行）" if prompt_rows is not None
                                 else "text の行はプロンプトのトークン数（予測しない。metadata の sequence.text を見る）")}


# ---------------------------------------------------------------------------------------------
# 一時 yaml とサーバー（h3_vsf_calib.py の写し。置き場だけ outputs\...\_rt）
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


def server_cmd() -> list[str]:
    return [str(PY), "-u", str(REPO / "main.py"), "--config", str(RT / TEMP_CONFIG_NAME)]


class Server:
    def __init__(self, port: int) -> None:
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.reused_pid: int | None = None
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
        prefix = ""
        for d in (REPO / "tools" / "uv", REPO / "tools" / "ffmpeg" / "bin"):
            if d.exists():
                prefix += f"{d};"
        env["PATH"] = prefix + env.get("PATH", "")
        env["UV_PYTHON_INSTALL_DIR"] = str(REPO / ".python")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        self._fh = self.log_path.open("a", encoding="utf-8", errors="replace")
        self._fh.write(f"\n===== {now_iso()} h3_outpaint_gate_api.py: main.py --config {cfg} =====\n")
        self._fh.flush()
        cmd = server_cmd()
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
            return
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
                rec["taskkill"] = run_cmd(["taskkill", "/PID", str(pid), "/T", "/F"]).stdout.strip()
            self.proc = None
        elif pid:
            rec["taskkill"] = run_cmd(["taskkill", "/PID", str(pid), "/T", "/F"]).stdout.strip()
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
# WDDM の共有メモリの標本（較正台の vram_sampler2.ps1 の写し）と server.log の窓
# ---------------------------------------------------------------------------------------------

SAMPLER_PS1 = r"""param(
    [Parameter(Mandatory = $true)][string]$Luid,
    [Parameter(Mandatory = $true)][string]$Csv,
    [Parameter(Mandatory = $true)][string]$Stop
)
# outputs/comfort-calib-h3-chain-2026-11-08/vram_sampler2.ps1 の写し（列: timestamp,dedicated_MB,shared_MB,dt_ms）
"timestamp,dedicated_MB,shared_MB,dt_ms" | Out-File -FilePath $Csv -Encoding ascii
$prev = $null
while (-not (Test-Path -LiteralPath $Stop)) {
    try {
        $s = Get-Counter -Counter "\GPU Adapter Memory(*)\Dedicated Usage", "\GPU Adapter Memory(*)\Shared Usage" -MaxSamples 1 -ErrorAction Stop
        $ded = 0.0
        $shr = 0.0
        foreach ($smp in $s.CounterSamples) {
            if ($smp.Path -match [regex]::Escape($Luid)) {
                $mb = [math]::Round($smp.CookedValue / 1MB, 1)
                if ($smp.Path -match 'dedicated usage') { $ded = $mb }
                elseif ($smp.Path -match 'shared usage') { $shr = $mb }
            }
        }
        $now = Get-Date
        $dt = if ($null -eq $prev) { 0 } else { [math]::Round(($now - $prev).TotalMilliseconds, 0) }
        $prev = $now
        "$($now.ToString('yyyy-MM-dd HH:mm:ss.fff')),$ded,$shr,$dt" | Out-File -FilePath $Csv -Append -Encoding ascii
    }
    catch {
        Start-Sleep -Milliseconds 200
    }
}
"""

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


def discover_luid() -> dict:
    """GPU のアダプタのうち専用メモリの使用量が最も大きいもの（較正台の discover_luid と同じ選び方）。"""
    out = run_cmd(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", LUID_PS], timeout=180)
    rows = json.loads(out.stdout.strip()) if out.stdout.strip() else []
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        raise RuntimeError("GPU Adapter Memory のインスタンスが見つからない")
    rows.sort(key=lambda r: float(r.get("mb") or 0.0), reverse=True)
    return {"chosen_luid": rows[0]["instance"], "chosen_dedicated_mb": rows[0]["mb"], "candidates": rows,
            "discovered_at": now_iso()}


class Sampler:
    def __init__(self, job_dir: Path, luid: str) -> None:
        self.csv = job_dir / "wddm.csv"
        self.stop_file = job_dir / "wddm.stop"
        self.luid = luid
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        ps1 = RT / "wddm_sampler.ps1"
        ps1.parent.mkdir(parents=True, exist_ok=True)
        ps1.write_text(SAMPLER_PS1, encoding="utf-8")
        if self.stop_file.exists():
            self.stop_file.unlink()
        self.proc = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1),
                                      "-Luid", self.luid, "-Csv", str(self.csv), "-Stop", str(self.stop_file)],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def stop(self) -> None:
        if self.proc is None:
            return
        self.stop_file.write_text("stop", encoding="ascii")
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None

    def rows(self) -> list[dict]:
        if not self.csv.exists():
            return []
        out = []
        for line in self.csv.read_text(encoding="ascii", errors="replace").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                t = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S.%f").timestamp()
                out.append({"t": t, "dedicated_mb": float(parts[1]), "shared_mb": float(parts[2])})
            except ValueError:
                continue
        return out


SERVERLOG_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) +(\w+) +([\w.]+): (.*)$")
STAGE_START_RE = re.compile(r"^generate (?P<label>.+?) started(?: \((?P<total>\d+) \w+\))?$")
STAGE_PROG_RE = re.compile(r"^generate (?P<label>.+?) (?P<done>\d+)/(?P<total>\d+) \w+ \([\d.]+ [\w/]+, "
                           r"(?P<elapsed>[\d.]+)s elapsed\)$")


def stage_spans(text: str) -> dict:
    """server.log の切れ端から、工程ごとの窓 [started の行 → 最後の行] を取る（較正台の _h3_stage_span と同じ区切り）。"""
    events = []
    for line in text.splitlines():
        m = SERVERLOG_RE.match(line.rstrip())
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group(2)) / 1000.0
        events.append((t, m.group(5)))
    spans: dict[str, dict] = {}
    for name, label in STAGE_LABELS.items():
        start = end = None
        last_done = total = elapsed = None
        for t, msg in events:
            s = STAGE_START_RE.match(msg)
            if s and s.group("label") == label:
                start, end, last_done, total, elapsed = t, t, None, None, None
                continue
            p = STAGE_PROG_RE.match(msg)
            if p and p.group("label") == label and start is not None:
                end = t
                last_done, total, elapsed = int(p.group("done")), int(p.group("total")), float(p.group("elapsed"))
        spans[name] = {"label": label, "found": start is not None, "start": start, "end": end,
                       "start_iso": datetime.fromtimestamp(start).isoformat(timespec="milliseconds") if start else None,
                       "end_iso": datetime.fromtimestamp(end).isoformat(timespec="milliseconds") if end else None,
                       "last_done": last_done, "total": total, "log_elapsed_s": elapsed}
    return spans


def window_stats(rows: list[dict], t0, t1) -> dict:
    if t0 is None or t1 is None or t1 < t0:
        return {"evaluated": False, "reason": "窓が見つからない"}
    w = [r for r in rows if t0 <= r["t"] <= t1]
    shared = [r["shared_mb"] for r in w]
    ded = [r["dedicated_mb"] for r in w]
    return {"evaluated": True, "seconds": round(t1 - t0, 2), "wddm_samples": len(w),
            "enough_samples": len(w) >= MIN_WINDOW_SAMPLES,
            "shared_median_mb": round(statistics.median(shared), 1) if shared else None,
            "shared_peak_mb": round(max(shared), 1) if shared else None,
            "dedicated_median_mb": round(statistics.median(ded), 1) if ded else None,
            "dedicated_peak_mb": round(max(ded), 1) if ded else None}


# ---------------------------------------------------------------------------------------------
# MCP の呼び出し（h3_retake_gate.py の写し）
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
    report: dict[str, Any] = {"tool_count": len(tools), "problems": [], "warnings": []}
    for name, needed in REQUIRED_TOOL_ARGS.items():
        t = by_name.get(name)
        if t is None:
            report["problems"].append(f"道具 {name} が tools/list に無い")
            continue
        props = set((t.inputSchema or {}).get("properties", {}))
        missing = needed - props
        if missing:
            report["problems"].append(f"{name} の inputSchema に引数 {sorted(missing)} が無い（あるのは {sorted(props)}）")
    for name, names in EXPECTED_BUT_UNSENT_ARGS.items():
        t = by_name.get(name)
        if t is None:
            continue
        missing = names - set((t.inputSchema or {}).get("properties", {}))
        if missing:
            report["warnings"].append(f"{name} に {sorted(missing)} が無い（台本は渡さないので止めない）")
    report["ok"] = not report["problems"]
    return report


async def ensure_pipeline(mcp: Mcp, args, run_rec: dict) -> bool:
    """H3 を載せる。load_pipeline は 1 回だけ（読み込み中の呼び直しは 409）。以後 backend_status を 10 秒おき。"""
    load: dict[str, Any] = {"requested": {"base_model": H3_BASE_MODEL, "models": {"transformer": "default"},
                                          "wait_sec": 45}, "responses": [], "status_polls": 0}
    run_rec["load"] = load
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
        if args.skip_load and not (s.get("pipeline_loaded") and s.get("base_model") == H3_BASE_MODEL):
            load["error"] = (f"--skip-load だが H3 が載っていない（state={s.get('state')} "
                             f"base_model={s.get('base_model')} pipeline_loaded={s.get('pipeline_loaded')}）")
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
        run_rec["models_after_load"] = {k: h3.get(k) for k in ("transformer_role", "transformer_weight_class",
                                                               "unsupported_features", "constraints")}
    except ToolCallError as exc:
        run_rec["models_after_load"] = {"error": exc.text}
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
# 記録と判定
# ---------------------------------------------------------------------------------------------

def denoise_picks(meta: dict) -> dict:
    """metadata.json の h3.phases.denoise から G5 の記録の鍵を抜く。"""
    den = (((meta.get("h3") or {}).get("phases") or {}).get("denoise")) or {}
    v16 = den.get("vram16") or {}
    return {"peak_vram_mb": den.get("peak_vram_mb"), "peak_vram_reserved_mb": den.get("peak_vram_reserved_mb"),
            "seconds": den.get("seconds"), "block_swap_prefetch_used": den.get("block_swap_prefetch_used"),
            "sequence": den.get("sequence"), "sequence_total": (den.get("sequence") or {}).get("total"),
            "vram16": {k: v16.get(k) for k in ("cap_mib", "cap_fraction", "num_alloc_retries", "num_ooms",
                                               "inactive_split_peak_mib", "blocks_on_gpu", "row_chunk")},
            "m_per_step": den.get("m_per_step")}


def oom_stage(error: str | None) -> str | None:
    """job.error の最初のトレースバックの段（較正台の h3_oom_location の簡略版。denoise／encode／decode／other）。"""
    if not error or "Traceback" not in error:
        return None
    first = error.split("Traceback (most recent call last):")[1]
    files = re.findall(r'File "([^"]+)", line \d+, in (\S+)', first)
    names = [(f.replace("\\", "/").rsplit("/", 1)[-1], fn) for f, fn in files]
    if any(f == "text_encoder.py" for f, _ in names):
        return "encode"
    if any(f in ("dit.py", "lowvram.py") or fn == "_denoise" for f, fn in names):
        return "denoise"
    if any("decode" in f"{f}:{fn}".lower() for f, fn in names):
        return "decode"
    return "other"


def find_baseline(summary: dict, label: str) -> dict | None:
    """規則 v3″ の基準点（G5 の 56 枚）。同じ summary_api.json にある最後の completed の本。"""
    for rec in reversed(summary.get("runs", [])):
        if (rec.get("gate") == "G5" and rec.get("frames") == G5_BASE_FRAMES and rec.get("status") == "completed"
                and rec.get("label") != label):
            return rec
    return None


def judge_g5(rec: dict, baseline: dict | None) -> dict:
    """快適上限表 §17.2 の規則 v3″: 快適 = completed で、① OOM・② 退避・③ 先読みの降格のどれにも当たらない。
    ② は denoise と decode の窓で、標本 10 以上・共有メモリの中央値が基準点の同じ窓より +300MB 以上なら退避。"""
    err = rec.get("error") or ""
    status = rec.get("status")
    oom = status == "failed" and err.startswith("GPU_OOM:")
    out: dict[str, Any] = {"rule": "快適上限表 §17.2 規則 v3″", "status": status, "oom": oom,
                           "oom_stage": oom_stage(err) if oom else None}
    prefetch = (rec.get("metadata_summary") or {}).get("block_swap_prefetch_used")
    out["prefetch_used"] = prefetch
    out["prefetch_demoted"] = None if status != "completed" else (prefetch != "on")
    spill: dict[str, Any] = {}
    undecided = []
    windows = rec.get("windows") or {}
    base_windows = (baseline or {}).get("windows") or {}
    for name in JUDGED_WINDOWS:
        w = windows.get(name) or {}
        b = base_windows.get(name) or {}
        item = {"shared_median_mb": w.get("shared_median_mb"), "wddm_samples": w.get("wddm_samples"),
                "baseline_shared_median_mb": b.get("shared_median_mb"),
                "baseline_label": (baseline or {}).get("label")}
        if not w.get("evaluated"):
            item["judged"] = None
            item["reason"] = w.get("reason") or "窓なし"
        elif not w.get("enough_samples"):
            item["judged"] = None
            item["reason"] = f"標本が {MIN_WINDOW_SAMPLES} 未満"
        elif b.get("shared_median_mb") is None:
            item["judged"] = None
            item["reason"] = f"基準点（G5 の {G5_BASE_FRAMES} 枚）の同じ窓が無い"
        else:
            item["delta_mb"] = round(w["shared_median_mb"] - b["shared_median_mb"], 1)
            item["judged"] = item["delta_mb"] >= SHARED_MARGIN_MB
        if item["judged"] is None and status == "completed":
            undecided.append(name)
        spill[name] = item
    out["spill"] = spill
    if oom:
        out["verdict"] = "非快適（① OOM）"
    elif status != "completed":
        out["verdict"] = f"判定なし（ジョブが {status}）"
    elif out["prefetch_demoted"]:
        out["verdict"] = "非快適（③ 先読みの降格）"
    elif any(v.get("judged") for v in spill.values()):
        out["verdict"] = "非快適（② 退避）"
    elif undecided:
        out["verdict"] = f"快適（①③ なし。② は未判定: {', '.join(undecided)}）"
    else:
        out["verdict"] = "快適"
    return out


def measure_g6(rec: dict, job_dir: Path) -> dict:
    """エンジンの仮想環境の子プロセスで、弾 1 の measure_run（保持矩形の PSNR・境界の勾配比・pad の帯）を呼ぶ。
    比べる相手は弾 1 と同じ作り: 切り出し＝サーバーが切った ``_outpaint_source.mp4``、対照＝それを
    ``engine_h3.media_out.write_mp4`` で書き直したもの（子プロセスが作る）。"""
    op = rec.get("outpaint_record") or {}
    src = job_dir / "_outpaint_source.mp4"
    if not src.exists():
        return {"error": f"{src} が無い（ジョブのフォルダから写せなかった）"}
    req = {
        "out_root": str(OUT_ROOT),
        "output": str(Path(rec["saved_path"])),
        "cut": str(src),
        "control": str(job_dir / "_control_write_mp4.mp4"),
        "run": {"canvas": rec["canvas"], "source": rec["source_size"], "frames": rec["frames"],
                "pads": {f"pad_{k}": v for k, v in rec["pads"].items()},
                "feather_px": op.get("feather_px"),
                "material": str(src), "control": str(job_dir / "_control_write_mp4.mp4")},
    }
    if req["run"]["feather_px"] is None:
        return {"error": "metadata の h3.outpaint.feather_px が無い（なじませの芯が決まらない）"}
    return run_measure_helper(req, job_dir / "measure_request.json", job_dir / "measure_result.json")


def run_measure_helper(req: dict, req_path: Path, res_path: Path) -> dict:
    write_json(req_path, req)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": "-1",
           "TORCH_COMPILE_DISABLE": "1"}
    cmd = [str(ENGINE_PY), str(Path(__file__).resolve()), "--measure-helper", str(req_path), str(res_path)]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=1800)
    if proc.returncode != 0 or not res_path.exists():
        return {"error": f"測定の子プロセスが失敗（{proc.returncode}）", "stderr_tail": proc.stderr[-3000:], "cmd": cmd}
    return json.loads(res_path.read_text(encoding="utf-8"))


def measure_helper_main(req_path: str, res_path: str) -> int:
    """（エンジンの仮想環境で走る）弾 1 の台本を import し、対照を作って measure_run を呼ぶ。CPU だけ。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO))
    import h3_outpaint_gate as G  # noqa: PLC0415 -- 弾 1 の台本（判定の関数の正本）
    import torch  # noqa: PLC0415
    from engine_h3.media_out import write_mp4  # noqa: PLC0415

    req = json.loads(Path(req_path).read_text(encoding="utf-8"))
    run = req["run"]
    nf = int(run["frames"])
    control = Path(req["control"])
    cut_frames = G.read_frames(Path(req["cut"]), nf)
    if not control.exists():
        write_mp4(str(control), torch.from_numpy(cut_frames), fps=G.FRAME_RATE, audio=None)
    ctrl_psnr = G.psnr_frames(G.read_frames(control, nf), cut_frames)
    metrics = G.measure_run(run, Path(req["output"]), Path(req["out_root"]))
    metrics["control_psnr_vs_cut_db"] = round(ctrl_psnr, 4)
    metrics["judge"] = "h3_outpaint_gate.measure_run（弾 1 と同じ関数）"
    write_json(Path(res_path), metrics)
    return 0


# ---------------------------------------------------------------------------------------------
# 1 本のジョブ
# ---------------------------------------------------------------------------------------------

async def run_job(mcp: Mcp, args, job: dict, summary: dict, luid: dict | None) -> dict:
    label = job["label"]
    job_dir = OUT_ROOT / label
    job_dir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {k: job[k] for k in ("gate", "label", "purpose", "frames", "source_size", "canvas", "pads",
                                               "material", "predicted_rows")}
    rec.update(started_at=now_iso(), status=None, error=None, dir=str(job_dir))
    log(f"=== {label}: {job['purpose']} ===")
    # (a) 素材のアップロード
    try:
        up = await mcp.call("upload_video", {"file_path": job["material"], "max_frames": UPLOAD_MAX_FRAMES})
    except ToolCallError as exc:
        rec.update(status="upload_error", error=exc.text, finished_at=now_iso())
        return rec
    rec["upload"] = up
    log(f"upload: video_id={up.get('video_id')} frame_count={up.get('frame_count')} fps={up.get('fps')}")
    tool_args = {**job["tool_args"], "reference_video_id": up["video_id"]}
    rec["tool_args"] = tool_args
    write_json(job_dir / "request.json", {"tool": "submit_generate", "arguments": tool_args, "gate": job["gate"],
                                          "note": "MCP の道具に渡した引数。H3 では MCP が reference_video_id を "
                                                  "outpaint.source_video_id へ移す。サーバーが受けた本文（書き戻し後）は "
                                                  "metadata.json の request"})
    # (b) 標本と server.log の起点（G5 だけ標本を取る）
    server_log = RT / "logs" / "server.log"
    srv_off = file_offset(server_log)
    wk_off = file_offset(worker_log_path())
    sampler = Sampler(job_dir, luid["chosen_luid"]) if (job["gate"] == "G5" and luid) else None
    if sampler:
        sampler.start()
        await asyncio.sleep(args.idle_sec)  # 送る前の静かな区間（記録だけ）
    t_submit = time.time()
    t0 = time.monotonic()
    try:
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
    finally:
        if sampler:
            sampler.stop()
    final = waited.pop("final") or {}
    rec["wait"] = waited
    rec["job_final"] = final
    rec["status"] = final.get("status") or "unknown"
    rec["seconds_wall"] = round(time.monotonic() - t0, 1)
    wk = read_since(worker_log_path(), wk_off)
    if wk:
        (job_dir / "worker_log_excerpt.txt").write_text(wk, encoding="utf-8")
    srv = read_since(server_log, srv_off)
    if srv:
        (job_dir / "server_log_excerpt.txt").write_text(srv, encoding="utf-8")
    if sampler:
        rows = sampler.rows()
        spans = stage_spans(srv)
        rec["spans"] = spans
        rec["wddm"] = {"luid": luid["chosen_luid"], "samples": len(rows), "csv": str(sampler.csv)}
        rec["windows"] = {name: window_stats(rows, s.get("start"), s.get("end")) for name, s in spans.items()}
        rec["windows"]["idle_before_submit"] = window_stats(rows, rows[0]["t"] if rows else None, t_submit)
    if waited.get("gave_up"):
        rec.update(status="wait_gave_up", error=waited["gave_up"], finished_at=now_iso())
        return rec
    if rec["status"] != "completed":
        rec["error"] = final.get("error")
        log(f"ジョブが {rec['status']}: {(final.get('error') or '')[:300]}")
    else:
        try:
            saved = await mcp.call("save_job_video", {"job_id": job_id, "dest_dir": str(job_dir),
                                                      "no_clobber": True, "which": "output"})
            rec["saved_path"] = saved["saved_path"]
        except ToolCallError as exc:
            rec["error"] = f"save_job_video: {exc.text}"
        meta: dict = {}
        try:
            vp = await mcp.call("get_job_video_path", {"job_id": job_id})
            src_dir = Path(vp["path"]).parent
            if (src_dir / "metadata.json").exists():
                shutil.copy2(src_dir / "metadata.json", job_dir / "metadata.json")
                meta = json.loads((src_dir / "metadata.json").read_text(encoding="utf-8"))
            else:
                rec["metadata_missing"] = str(src_dir / "metadata.json")
            if (src_dir / "_outpaint_source.mp4").exists():
                shutil.copy2(src_dir / "_outpaint_source.mp4", job_dir / "_outpaint_source.mp4")
        except ToolCallError as exc:
            rec["metadata_missing"] = exc.text
        h3 = meta.get("h3") or {}
        rec["metadata_summary"] = {k: meta.get(k) for k in (
            "attention_used", "vram16_mode_used", "keep_resident_used", "block_swap_prefetch_used",
            "generation_time_seconds", "peak_vram_reserved_mb", "seed_used")}
        rec["metadata_summary"]["num_inference_steps"] = h3.get("num_inference_steps")
        rec["metadata_summary"]["loras"] = h3.get("loras")
        rec["metadata_summary"]["request_outpaint"] = (meta.get("request") or {}).get("outpaint")
        rec["denoise"] = denoise_picks(meta)
        op = h3.get("outpaint")
        rec["outpaint_record"] = op
        rec["outpaint_keys_check"] = {
            "present": isinstance(op, dict),
            "count": len(op) if isinstance(op, dict) else 0,
            "missing": [k for k in OUTPAINT_KEYS if not isinstance(op, dict) or k not in op],
            "extra": sorted(set(op) - set(OUTPAINT_KEYS)) if isinstance(op, dict) else [],
        }
        if job["gate"] == "G6" and rec.get("saved_path"):
            try:
                rec["metrics"] = measure_g6(rec, job_dir)
                if rec["metrics"].get("error"):
                    rec["error"] = f"測定: {rec['metrics']['error']}"
            except Exception as exc:  # noqa: BLE001
                rec["metrics"] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                rec["error"] = f"測定: {type(exc).__name__}: {exc}"
    if job["gate"] == "G5":
        rec["judgement"] = judge_g5(rec, find_baseline(summary, label))
        log(f"判定: {rec['judgement']['verdict']}")
    rec["finished_at"] = now_iso()
    return rec


# ---------------------------------------------------------------------------------------------
# まとめ
# ---------------------------------------------------------------------------------------------

def load_summary() -> dict:
    if SUMMARY_JSON.exists():
        try:
            data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("runs", [])
                data.setdefault("invocations", [])
                return data
        except Exception:  # noqa: BLE001
            pass
    return {"script": "h3_outpaint_gate_api.py", "plan": "rosy-jingling-stonebraker.md §2.6・§3 弾 2・§7（G5・G6）",
            "runs": [], "invocations": []}


def put_run(summary: dict, rec: dict) -> None:
    """同じ名前の本は今回のもので置き換える（前のものは invocations の側に時刻だけ残る）。"""
    summary["runs"] = [r for r in summary["runs"] if r.get("label") != rec.get("label")] + [rec]


def _p(x) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def write_summary_md(summary: dict) -> None:
    lines = ["# MiniMax H3 の Outpainting 弾 2 の GPU の門（API 経由。記録）", "",
             "数値の正本は記録簿 `Docs/VERIFICATION_LOG.md`。欄の意味は台本の README.md「弾 2」の節。", "",
             "## G5（快適上限の点。画質は見ない）", "",
             "| 本 | 枚数 | 状態 | 判定 | S（実測／予測） | 山 alloc／reserved（MiB） | cap | 再試行／OOM | inactive split 山 | "
             "denoise 共有 中央値（基準） | decode 共有 中央値（基準） | 生成秒 | 壁秒 | エラー |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["runs"]:
        if r.get("gate") != "G5":
            continue
        d = r.get("denoise") or {}
        v = d.get("vram16") or {}
        j = r.get("judgement") or {}
        sp = j.get("spill") or {}
        err = (r.get("error") or "").replace("|", "\\|").replace("\n", " ")[:120]

        def win(n):
            s = sp.get(n) or {}
            return f"{_p(s.get('shared_median_mb'))}（{_p(s.get('baseline_shared_median_mb'))}）"
        lines.append(
            f"| {r.get('label')} | {r.get('frames')} | {r.get('status')} | {j.get('verdict', '–')} | "
            f"{_p(d.get('sequence_total'))}／{_p((r.get('predicted_rows') or {}).get('total'))} | "
            f"{_p(d.get('peak_vram_mb'))}／{_p(d.get('peak_vram_reserved_mb'))} | {_p(v.get('cap_mib'))} | "
            f"{_p(v.get('num_alloc_retries'))}／{_p(v.get('num_ooms'))} | {_p(v.get('inactive_split_peak_mib'))} | "
            f"{win('denoise')} | {win('decode')} | "
            f"{_p((r.get('metadata_summary') or {}).get('generation_time_seconds'))} | {_p(r.get('seconds_wall'))} | "
            f"{err or '–'} |")
    lines += ["", "## G6（20 ステップの目視用）", "",
              "| 本 | 状態 | 保持矩形 PSNR 対照／芯（dB） | 勾配比 縦 max／横 max | h3.outpaint の鍵 | 生成秒 | "
              "山 alloc／reserved（MiB） | 注意 | ステップ | エラー |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["runs"]:
        if r.get("gate") != "G6":
            continue
        m = r.get("metrics") or {}
        b = m.get("boundary_gradient_ratio") or {}
        d = r.get("denoise") or {}
        ms = r.get("metadata_summary") or {}
        kc = r.get("outpaint_keys_check") or {}
        err = (r.get("error") or "").replace("|", "\\|").replace("\n", " ")[:120]
        lines.append(
            f"| {r.get('label')} | {r.get('status')} | {_p(m.get('keep_psnr_vs_control_db'))}／"
            f"{_p(m.get('keep_core_psnr_vs_control_db'))} | {_p(b.get('vertical_max'))}／{_p(b.get('horizontal_max'))} | "
            f"{kc.get('count', '–')}（欠け {kc.get('missing', '–')}） | {_p(ms.get('generation_time_seconds'))} | "
            f"{_p(d.get('peak_vram_mb'))}／{_p(d.get('peak_vram_reserved_mb'))} | {ms.get('attention_used', '–')} | "
            f"{ms.get('num_inference_steps', '–')} | {err or '–'} |")
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def flush(summary: dict) -> None:
    write_json(SUMMARY_JSON, summary)
    write_summary_md(summary)


# ---------------------------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------------------------

def mcp_env(port: int) -> dict:
    return {"PYTHONUTF8": "1", "LTX_MCP_BASE_URL": base_url(port), "LTX_MCP_CONFIG": str(RT / TEMP_CONFIG_NAME),
            "LTX_MCP_OUTPUT_DIR": str(RT / "outputs")}


async def run(args, job: dict, summary: dict, inv: dict) -> int:
    from mcp import ClientSession, StdioServerParameters  # アプリの仮想環境の依存
    from mcp.client.stdio import stdio_client

    calls: list = []
    calls_path = OUT_ROOT / "logs_api" / f"mcp_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job['label']}.json"
    inv["mcp_calls"] = str(calls_path)
    env = {**os.environ, **mcp_env(args.port)}
    inv["mcp_env"] = mcp_env(args.port)
    server = StdioServerParameters(command=str(PY), args=["-m", "mcp_server"], cwd=str(REPO), env=env, encoding="utf-8")
    stderr_log = OUT_ROOT / "logs_api" / "mcp_server.stderr.log"
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    code = 0
    luid = None
    if job["gate"] == "G5":
        try:
            luid = discover_luid()
            inv["luid"] = luid
            log(f"LUID: {luid['chosen_luid']}（専用 {luid['chosen_dedicated_mb']}MB）")
        except Exception as exc:  # noqa: BLE001 -- 標本が取れなくても①③は判定できる。②は未判定になる
            inv["luid_error"] = f"{type(exc).__name__}: {exc}"
            log(f"注意: LUID が取れない（② 退避は未判定になる）: {exc}")
    try:
        with open(stderr_log, "a", encoding="utf-8") as errlog:
            errlog.write(f"\n===== {now_iso()} h3_outpaint_gate_api.py {job['label']} =====\n")
            errlog.flush()
            async with stdio_client(server, errlog=errlog) as (read, write):
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
                    for w in schema["warnings"]:
                        log(f"注意: {w}")
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
                    try:
                        rec = await run_job(mcp, args, job, summary, luid)
                    except Exception as exc:  # noqa: BLE001
                        rec = {**{k: job[k] for k in ("gate", "label", "frames")}, "status": "script_error",
                               "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                    put_run(summary, rec)
                    write_json(OUT_ROOT / job["label"] / "summary_row.json", rec)
                    flush(summary)
                    g5_oom = job["gate"] == "G5" and (rec.get("judgement") or {}).get("oom")
                    if (rec.get("status") != "completed" and not g5_oom) or (rec.get("status") == "completed"
                                                                             and rec.get("error")):
                        code = 3
                    if not args.keep_server:
                        try:
                            inv["unload"] = await mcp.call("unload_pipeline")
                        except ToolCallError as exc:
                            inv["unload"] = {"error": exc.text}
    finally:
        write_json(calls_path, calls)
    return code


def dry_run(args, job: dict) -> int:
    material = Path(job["material"])
    out = {
        "gate": job["gate"], "label": job["label"], "purpose": job["purpose"],
        "material": {"path": str(material), "exists": material.is_file()},
        "server": {"port": args.port, "temp_config": str(RT / TEMP_CONFIG_NAME), "temp_state": str(RT / TEMP_STATE_NAME),
                   "cmd": server_cmd(), "note": "config.yaml は読むだけ。port・log_dir・output.dir・upload.dir・state_file を _rt へ"},
        "mcp_server": {"command": str(PY), "args": ["-m", "mcp_server"], "cwd": str(REPO), "env": mcp_env(args.port)},
        "load_pipeline": None if args.skip_load else {"base_model": H3_BASE_MODEL, "models": {"transformer": "default"},
                                                      "wait_sec": 45},
        "upload_video": {"file_path": str(material), "max_frames": UPLOAD_MAX_FRAMES},
        "submit_generate": {**job["tool_args"], "reference_video_id": "<upload_video の video_id>"},
        "not_sent": ["loras（Turbo なし）", "outpaint_blend_dilation_stage1（サーバーの既定）",
                     "outpaint_freeze_source_audio（H3 では 422）", "vram16_mode（既定 on）", "keep_resident（既定 off）",
                     "block_swap_prefetch（既定 on）", "shift_video／shift_audio（既定 12／3）"],
        "canvas": job["canvas"], "pads": job["pads"], "source_size": job["source_size"], "frames": job["frames"],
        "predicted_rows": job["predicted_rows"], "prompt_rows_note": job["prompt_rows_note"],
        "single_rows_budget": SINGLE_ROWS_BUDGET,
        "records": (["h3.phases.denoise.peak_vram_mb／peak_vram_reserved_mb", "h3.phases.denoise.vram16（cap_mib・"
                     "num_alloc_retries・num_ooms・inactive_split_peak_mib）", "h3.phases.denoise.sequence.total",
                     "成否・生成秒・壁秒", "WDDM の共有メモリ（denoise・decode の窓。server.log で区切る）",
                     f"判定: 規則 v3″（② の基準点は G5 の {G5_BASE_FRAMES} 枚）"]
                    if job["gate"] == "G5" else
                    ["保持矩形の PSNR（対照＝_outpaint_source.mp4 を write_mp4 で書き直したもの）・境界の勾配比・pad の帯",
                     "h3.outpaint の 12 鍵", "生成秒・VRAM の山（h3.phases.denoise）"]),
        "out_dir": str(OUT_ROOT / job["label"]),
        "summary": [str(SUMMARY_JSON), str(SUMMARY_MD)],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not material.is_file():
        log(f"注意: 素材が無い（先に --prepare）: {material}")
    return 0


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="MiniMax H3 の Outpainting 弾 2 の GPU の門 G5・G6（MCP 経由・別ポートの実バックエンド）")
    ap.add_argument("--gate", choices=("G5", "G6"), help="回す門")
    ap.add_argument("--frames", type=int, default=None,
                    help=f"G5 のフレーム数（17n+5。既定の 2 点は {G5_FRAMES_DEFAULT}、基準点は {G5_BASE_FRAMES}）。"
                         "--prepare では切り出す枚数（カンマ区切り可は --prepare-frames）")
    ap.add_argument("--prepare", action="store_true", help="素材を作って終わる（CPU・ffmpeg）")
    ap.add_argument("--prepare-frames", default=",".join(str(f) for f in G5_FRAMES_DEFAULT),
                    help="--prepare で作る G5 の枚数（カンマ区切り。既定 345,362）")
    ap.add_argument("--dry-run", action="store_true", help="サーバーを起こさず、送る引数の一覧を出して終わる")
    ap.add_argument("--skip-load", action="store_true", help="load_pipeline を呼ばない（--keep-server で残した H3 を使う）")
    ap.add_argument("--keep-server", action="store_true", help="終わりに unload もサーバーの停止もしない")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT, help=f"一時サーバーのポート（既定 {PORT_DEFAULT}）")
    ap.add_argument("--label", default=None, help="本の名前（既定 G5_<枚数>f／G6_20steps）")
    ap.add_argument("--prompt", default=None, help="プロンプトの差し替え（既定 G5 は較正の文、G6 は弾 1 の文）")
    ap.add_argument("--idle-sec", type=float, default=15.0, help="G5: 標本を始めてから送るまでの静かな区間（記録だけ）")
    ap.add_argument("--load-timeout-sec", type=float, default=1200.0)
    ap.add_argument("--job-timeout-sec", type=float, default=7200.0, help="1 本を待つ上限（既定 7200 秒）")
    ap.add_argument("--measure-helper", nargs=2, metavar=("REQ", "RES"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.measure_helper:
        return args
    if not args.prepare and not args.gate:
        ap.error("--gate か --prepare を指定する")
    if args.gate == "G5" and args.frames is None and not args.prepare:
        ap.error("G5 は --frames が要る（例: --frames 345）")
    if args.gate == "G6" and args.frames not in (None, G6_FRAMES):
        ap.error(f"G6 のフレーム数は {G6_FRAMES} 固定")
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
    if args.prepare:
        frames_list = [int(x) for x in args.prepare_frames.split(",") if x.strip()]
        if args.gate == "G5" and args.frames is not None and args.frames not in frames_list:
            frames_list.append(args.frames)
        for f in frames_list:
            latent_frames(f)
        do_prepare(frames_list)
        log(f"素材: {MAT_DIR}（記録 {MAT_DIR / 'prepare_api.json'}）")
        if not args.gate:
            return 0
    job = build_job(args)
    if args.dry_run:
        return dry_run(args, job)
    if not Path(job["material"]).is_file():
        print(f"素材が無い（先に --prepare）: {job['material']}", file=sys.stderr)
        return 1
    summary = load_summary()
    inv: dict[str, Any] = {"started_at": now_iso(), "argv": sys.argv[1:], "label": job["label"]}
    summary["invocations"].append(inv)
    flush(summary)
    srv = Server(args.port)
    code = 1
    try:
        srv.start()
        inv["server"] = srv.record
        flush(summary)
        code = asyncio.run(run(args, job, summary, inv))
    except BaseException as exc:  # noqa: BLE001 -- 例外も中断も記録して終わる
        inv["error"] = f"{type(exc).__name__}: {exc}"
        inv["traceback"] = traceback.format_exc()
        log(f"止まった: {inv['error']}")
        code = 1
    finally:
        if args.keep_server:
            log("--keep-server: サーバーを残す（読み込んだ H3 もそのまま）")
        else:
            try:
                srv.stop()
            except Exception as exc:  # noqa: BLE001
                srv.record["stop_error"] = f"{type(exc).__name__}: {exc}"
        inv["server"] = srv.record
        inv["finished_at"] = now_iso()
        inv["exit_code"] = code
        flush(summary)
        log(f"記録: {SUMMARY_JSON}・{SUMMARY_MD}（終了コード {code}）")
    return code


if __name__ == "__main__":
    sys.exit(main())
