"""MiniMax H3 の連結生成（Chained）に「クリップごとの A2V」（``clips[i].source_audio``）を足した実装に対する
GPU の門 GA0〜GA3 を、別のポートに起こした実バックエンドへ HTTP で直接送って回す台本（台帳 §1-120、
計画 ``enumerated-percolating-lovelace.md`` §6.12）。

製品コードは import しない（例外: 突き合わせの記録のためだけに、torch を使わない純関数の
``h3_chain_math.clip_audio_start_sample`` と ``compute_h3_chain_layout`` を読む。期待値そのものは
この台本の中で独立に計算する）。部品は次から写した。

- ``Docs/Outputs-archive/h3-vsf-chain-2026-11-09/h3_vsf_chain_gate.py``: 一時 yaml（``build_temp_config``・
  ``write_temp_config``）・``Server``（別ポートで起こし、終わりに unload → プロセスの木ごと停止）・ワーカーの
  ログの差分・プロンプトの組み立て（``compose_clip_prompt``・``split_prompts``・``expected_clip_prompts``。逐語）・
  G-C0 の本文との突き合わせ（``request_diff``）。
- ``Docs/Outputs-archive/h3-chain-gates-2026-10-29/g_c0.json``: G-C0 の要求本文（``request_body``）・mp4 の指紋・
  各クリップの潜在の指紋。GA0 はこの本文をそのまま送る。

前例との違い: 前例は MCP の ``submit_chain`` で送ったが、MCP の ``ChainClipArg`` には ``source_audio`` が無い
（計画 §6.11「MCP と Gradio は送らない」）。そこで全部の門を ``POST /api/v1/generate/chain`` へ直接送る
（G-C0 そのものも HTTP 直送で回した。本文は ``g_c0.json`` の ``request_body``）。音声は
``POST /api/v1/upload/audio`` で上げる。

門:
  GA0  不変。G-C0 の本文（FL2VA・640×384・124×3・seed 42・sdpa・20 ステップ・shift 12/3・
       ``embed_mp4_metadata=false``）をそのまま送り、mp4 の sha256 と各クリップの ``latents_sha256`` が G-C0 と全桁一致。
  GA1  Strength 0 は効き目なし。GA0 の本文の各クリップに ``source_audio={audio_id, strength: 0.0, offset_frames: S_i}``
       （S = 0・102・204）を足す。指紋が G-C0 と一致・``metadata.a2v.clips[i].encoded_samples == 0``。
  GA2  Strength 1.0・継ぎ目が格子に乗らない形。[124, 141, 124]・全クリップ 1.0・offset 0・102・221。
       完走・``h3.clips[1..].head_equals_prev_tail`` が真・``mask_value == 0``・``audio_output == "decoded"``・
       総フレーム 345・``start_sample`` が 0・136000・294400。
  GA3  Strength 0.5・目視用。3×209・1344×768・sage・0.5・offset 0・187・374・20 ステップ・shift 12/3。
       機械が見るのは完走と記録の形だけ。VRAM の山・所要・主記憶は記録だけ。目視はオーナー。
  GA3T GA3 と同じ条件で Turbo LoRA（``minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16``・強度 0.8）・
       4 ステップ・shift 12/3（前例 ``h3-vsf-2026-11-09/h3_vsf_gate.py`` の T 系列の写し）。目視用。結果は ``ga3t\\``。
  all  GA0 → GA1 → GA2 → GA3 → GA3T を 1 つのサーバーで続けて回す。どれかが不合格・失敗なら、そこで止まる。

流れ:
  1. ``config.yaml`` を読むだけで、``server.port``・``server.log_dir``・``output.dir``・``upload.dir``・``state_file``
     を ``_rt\\`` に向け、``vram.block_swap: true``・``vram.block_swap_blocks_on_gpu: 8`` を明示した一時 yaml と、
     MiniMax H3 を選んだ一時 state を ``_rt\\`` に書く（前例と同じ）。
  2. ``.venv\\Scripts\\python.exe main.py --config <一時 yaml>`` で別ポート（既定 18699）に実バックエンドを起こす。
  3. ``POST /pipeline/load {"base_model": "MiniMaxH3", "models": {"transformer": "default"}}`` を 1 回、以後
     ``GET /status`` を 10 秒おき。
  4. GA1〜GA3 を回すときは、音声を ``POST /upload/audio`` で 1 回だけ上げ、返った ``audio_id`` を全クリップで使う。
  5. 門を回す。ジョブは ``POST /generate/chain`` → ``GET /jobs/{id}`` を 10 秒おき。終わったら
     ``_rt\\outputs\\<job_id>\\metadata.json`` を門のフォルダへ写し、``output.mp4`` の sha256 を取る
     （mp4 は ``_rt\\`` に置いたまま。保管庫には写さない）。
  6. ``--keep-server`` でなければ ``POST /pipeline/unload`` → サーバーをプロセスの木ごと停止。
  7. 門ごとの ``ga<N>\\result.json`` と、全体の ``summary.json`` を書く。

終了コード: 0＝回した門がすべて合格（GA3 は記録の形の基準が合格）、3＝どれかが不合格・失敗（そこで止まる）、
1＝準備の段で止まった（サーバーが起きない・読み込みの失敗・音声のアップロードの失敗など）。
``--dry-run`` は送る本文と期待値を表示するだけ（サーバーも GPU も使わない）。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # Docs\Outputs-archive\<このフォルダ> の 3 つ上＝リポジトリ直下
ARCHIVE = REPO / "Docs" / "Outputs-archive"
RT = HERE / "_rt"
PY = REPO / ".venv" / "Scripts" / "python.exe"
PORT_DEFAULT = 18699
TEMP_CONFIG_NAME = "config.h3chaina2v.yaml"
TEMP_STATE_NAME = "state.h3chaina2v.json"

H3_BASE_MODEL = "MiniMaxH3"
H3_CATEGORIES = ("transformer", "text_encoder", "video_vae", "audio")
REAL_BACKEND_H3 = "minimax-h3"
FL2VA_NAME = "default"
FL2VA_FILE = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
H3_FPS = 24.0

#: 一時 yaml で明示する vram の鍵（前例と同じ。G-C0 を回した 2026-10-29 の config.yaml の値）。
VRAM_OVERRIDES = {"block_swap": True, "block_swap_blocks_on_gpu": 8}

#: G-C0（2026-10-29）の記録。正本は記録簿 ``Docs/VERIFICATION_LOG.md`` §184.6。
G_C0_JSON = ARCHIVE / "h3-chain-gates-2026-10-29" / "g_c0.json"
G_C0_MP4_SHA = "4a98be0884ae7129e34f58578a3fe7ca3117ca5a764545cbac404e12a70cbec5"
PROMPTS_C = ARCHIVE / "h3-chain-2026-10-29" / "prompts_c.json"
OVERLAP_FRAMES = 22  # H3 の連結の重なり（固定）

#: 音声（30.9 秒）。GA1〜GA3 で全クリップが同じ audio_id を指す。
AUDIO_DEFAULT = REPO / "uploads" / "h3-a2v-development" / "ref-voice-patra001.wav"

#: H3 の音声の格子（h3_chain_math と同じ値を独立に持つ）: 32 kHz・1 潜在 800 標本・1 フレーム 5/3 潜在。
AUDIO_SR = 32000
AUDIO_HOP = 800
AUDIO_PER_FRAME = Fraction(5, 3)

#: 門の形。
GA0_FRAMES = [124, 124, 124]
GA2_FRAMES = [124, 141, 124]
GA3_FRAMES = [209, 209, 209]
GA1_STRENGTH, GA2_STRENGTH, GA3_STRENGTH = 0.0, 1.0, 0.5
GA3_WIDTH, GA3_HEIGHT = 1344, 768
GA3_ATTENTION = "sage"
#: GA3 のプロンプト（共通の文だけ。クリップのプロンプトは付けない）。台詞のタグ ``<d>`` は入れない
#: （音声の中身と食い違わないように）。``--ga3-prompt`` で替えられる。
GA3_PROMPT_DEFAULT = ("A young woman sits in a softly lit room and talks to the camera with natural lip movement "
                      "and an expressive face. The camera is steady.")
#: GA3T（GA3 と同じ条件で Turbo LoRA・4 ステップ）。LoRA の名前・強度・指定の形は前例
#: ``h3-vsf-2026-11-09/h3_vsf_gate.py`` の T 系列（``TURBO_LORA``・``TURBO_LORA_STRENGTH``・``visual_plan``）の写し。
#: 名前は ``models/MiniMaxH3/LoRA/`` のファイル名から拡張子を除いたもの（前例の ``visual/T0/metadata.json`` の
#: ``request.loras`` と同じ）。
TURBO_LORA = "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16"
TURBO_LORA_STRENGTH = 0.8
TURBO_STEPS = 4

#: 計画 §6.12 に書かれた期待値（台本の独立の計算と、製品の純関数の両方と突き合わせる）。
PLAN_OFFSETS = {"ga1": [0, 102, 204], "ga2": [0, 102, 221], "ga3": [0, 187, 374]}
PLAN_START_SAMPLES_GA2 = [0, 136000, 294400]
PLAN_TOTAL_FRAMES = {"ga0": 328, "ga1": 328, "ga2": 345, "ga3": 583}

#: GA0 で、サーバーが受けた本文（metadata.request）が G-C0 の本文と同じでなければならない欄（前例と同じ）。
GA0_REQUEST_CRITICAL = (
    "prompt", "negative_prompt", "nag_enabled", "attention_backend", "block_swap_prefetch", "embed_mp4_metadata",
    "vram16_mode", "width", "height", "crop_output", "frame_rate", "num_inference_steps", "shift_video",
    "shift_audio", "seed", "clips", "source_video", "source_audio", "retake", "end_source", "loras",
    "reference_video_id",
)
GA0_REQUEST_IGNORED_BY_H3 = ("overlap_frames", "overlap_strength", "chunked_upsample", "stage2_window")

POLL_SEC = 10.0
GPU_SAMPLE_SEC = 3.0


# ---------------------------------------------------------------------------------------------
# 小道具（前例の写し）
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


def sha_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def check(checks: dict, name: str, ok: bool, actual: Any = None, expected: Any = None) -> None:
    checks[name] = {"pass": bool(ok), "actual": actual, "expected": expected}


def all_pass(checks: dict) -> bool:
    return bool(checks) and all(c.get("pass") for c in checks.values())


# ---------------------------------------------------------------------------------------------
# プロンプトの組み立て（h3_vsf_chain_gate.py の写し。逐語）
# ---------------------------------------------------------------------------------------------

def compose_clip_prompt(shared: str, own: str | None) -> str:
    """api/models.py ``compose_clip_prompt``: both ends trimmed, non-empty parts joined by one space."""
    parts = (shared.strip(), (own or "").strip())
    return " ".join(part for part in parts if part)


def split_prompts(prompts: list[str]) -> tuple[str, list[str]]:
    """prompts_c.json (the full 3 clip prompts) -> (shared prompt, appended part per clip)."""
    if not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError("the prompts file must be a non-empty JSON list of non-empty strings")
    shared = prompts[0]
    owns = [""]
    for i, text in enumerate(prompts[1:], start=1):
        if not text.startswith(shared):
            raise ValueError(f"prompts[{i}] does not start with prompts[0] (the shared prompt)")
        own = text[len(shared):]
        owns.append(own.strip())
    for i, text in enumerate(prompts):
        if compose_clip_prompt(shared, owns[i]) != text:
            raise ValueError(f"prompts[{i}] cannot be recomposed from the shared part and its appended part")
    return shared, owns


def expected_clip_prompts(prompts: list[str], n_clips: int) -> tuple[str, list[str], list[str]]:
    shared, owns = split_prompts(prompts)
    appended = [o for o in owns[1:] if o] or [""]
    own_per_clip = [""] + [appended[(i - 1) % len(appended)] for i in range(1, n_clips)]
    return shared, own_per_clip, [compose_clip_prompt(shared, o) for o in own_per_clip]


def load_prompts() -> list[str]:
    return json.loads(PROMPTS_C.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# 幾何（台本の中で独立に計算する）
# ---------------------------------------------------------------------------------------------

def seam_starts(frames: list[int]) -> list[int]:
    """各クリップの連結の時間軸上の開始フレーム S_i（S_0 = 0、S_i = S_{i-1} + F_{i-1} − 22）。"""
    out = [0]
    for f in frames[:-1]:
        out.append(out[-1] + f - OVERLAP_FRAMES)
    return out


def total_frames(frames: list[int]) -> int:
    return sum(frames) - OVERLAP_FRAMES * (len(frames) - 1)


def start_sample(offset_frames: int) -> int:
    """``round(O × 5/3) × 800``（計画 §3・§6.8 ①。三分の一はちょうど半分にならないので丸めの向きで迷わない）。"""
    return round(Fraction(offset_frames) * AUDIO_PER_FRAME) * AUDIO_HOP


def required_samples(num_frames: int, fps: float = H3_FPS) -> int:
    return round(num_frames * AUDIO_SR / fps)


def product_math_crosscheck(frames: list[int], offsets: list[int]) -> dict:
    """製品の ``h3_chain_math``（torch を使わない純関数）と台本の計算の突き合わせ。記録だけ（読めなければその旨）。"""
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import h3_chain_math as hm  # noqa: PLC0415 -- 標準ライブラリだけの純関数（torch を import しない）
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    out: dict[str, Any] = {"available": True}
    try:
        layout = hm.compute_h3_chain_layout(frames)
        # 製品の seam_start_frames は継ぎ目ごと（クリップ 1 から）。クリップ 0 の 0 を前に足して比べる。
        out["seam_start_frames"] = [0] + list(layout.seam_start_frames)
        out["total_frames"] = int(layout.total_frames)
        out["clip_audio_latents"] = list(layout.clip_audio_latents)
    except Exception as exc:  # noqa: BLE001
        out["layout_error"] = f"{type(exc).__name__}: {exc}"
    try:
        out["clip_audio_start_sample"] = [int(hm.clip_audio_start_sample(o)) for o in offsets]
    except Exception as exc:  # noqa: BLE001
        out["start_sample_error"] = f"{type(exc).__name__}: {exc}"
    out["equals_script"] = {
        "seam_start_frames": out.get("seam_start_frames") == seam_starts(frames),
        "total_frames": out.get("total_frames") == total_frames(frames),
        "start_sample": out.get("clip_audio_start_sample") == [start_sample(o) for o in offsets],
    }
    return out


# ---------------------------------------------------------------------------------------------
# 要求の組み立て（POST /generate/chain の本文）
# ---------------------------------------------------------------------------------------------

def load_archived_g_c0() -> dict:
    try:
        d = json.loads(G_C0_JSON.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001 -- 記録が無くても主の基準（指紋）は判定できる
        return {}
    return {"request": d.get("metadata_request") or {},
            "latents": [c.get("latents_sha256") for c in (d.get("metadata_h3") or {}).get("clips", [])],
            "mp4_sha256": (d.get("mp4") or {}).get("sha256"), "request_body": d.get("request_body") or {}}


def chain_clips(prompts: list[str], frames: list[int]) -> tuple[str, list[dict]]:
    """``prompt`` と ``clips``（g_chain.py の build_request と同じ形。クリップ 0 は共通の文だけ）。"""
    shared, owns, _ = expected_clip_prompts(prompts, len(frames))
    clips = []
    for own, f in zip(owns, frames):
        clip: dict = {"num_frames": f}
        if own:
            clip["prompt"] = own
        clips.append(clip)
    return shared, clips


def g_c0_body(prompts: list[str], frames: list[int]) -> dict:
    """G-C0 の ``request_body`` と同じ鍵と順の本文（``frames`` だけを替えられる）。"""
    shared, clips = chain_clips(prompts, frames)
    return {"prompt": shared, "width": 640, "height": 384, "seed": 42, "attention_backend": "sdpa",
            "num_inference_steps": 20, "shift_video": 12.0, "shift_audio": 3.0, "clips": clips,
            "embed_mp4_metadata": False}


def with_clip_audio(body: dict, audio_id: str, strength: float, offsets: list[int]) -> dict:
    out = copy.deepcopy(body)
    for clip, off in zip(out["clips"], offsets):
        clip["source_audio"] = {"audio_id": audio_id, "strength": strength, "offset_frames": off}
    return out


def ga0_body(prompts: list[str]) -> dict:
    return g_c0_body(prompts, GA0_FRAMES)


def ga1_body(prompts: list[str], audio_id: str) -> dict:
    return with_clip_audio(ga0_body(prompts), audio_id, GA1_STRENGTH, seam_starts(GA0_FRAMES))


def ga2_body(prompts: list[str], audio_id: str) -> dict:
    return with_clip_audio(g_c0_body(prompts, GA2_FRAMES), audio_id, GA2_STRENGTH, seam_starts(GA2_FRAMES))


def ga3_body(prompt: str, audio_id: str) -> dict:
    body = {"prompt": prompt, "width": GA3_WIDTH, "height": GA3_HEIGHT, "seed": 42,
            "attention_backend": GA3_ATTENTION, "num_inference_steps": 20, "shift_video": 12.0, "shift_audio": 3.0,
            "clips": [{"num_frames": f} for f in GA3_FRAMES], "embed_mp4_metadata": False}
    return with_clip_audio(body, audio_id, GA3_STRENGTH, seam_starts(GA3_FRAMES))


def ga3t_body(prompt: str, audio_id: str) -> dict:
    """GA3 と同じ本文に、前例の T 系列と同じ ``loras``（Turbo 0.8）と 4 ステップ（shift 12/3 はそのまま）。"""
    body = ga3_body(prompt, audio_id)
    body["loras"] = [{"name": TURBO_LORA, "strength": TURBO_LORA_STRENGTH}]
    body["num_inference_steps"] = TURBO_STEPS
    return body


def expected_a2v(frames: list[int], offsets: list[int], strength: float) -> list[dict]:
    """クリップごとの ``metadata.a2v.clips[i]`` の、幾何から決まる値（台本の独立の計算）。"""
    return [{"strength": strength, "offset_frames": o, "mask_value": 1.0 - strength, "start_sample": start_sample(o),
             "required_samples": required_samples(f), "audio_output": "decoded"} for f, o in zip(frames, offsets)]


# ---------------------------------------------------------------------------------------------
# metadata.json の読み方と判定（純関数）
# ---------------------------------------------------------------------------------------------

def h3_block(meta: dict) -> dict:
    h3 = meta.get("h3")
    return h3 if isinstance(h3, dict) else {}


def h3_clips(meta: dict) -> list[dict]:
    clips = h3_block(meta).get("clips")
    return [c if isinstance(c, dict) else {} for c in clips] if isinstance(clips, list) else []


def a2v_clips(meta: dict) -> list[dict | None]:
    """``metadata.a2v.clips``（無ければ空のリスト）。"""
    a2v = meta.get("a2v")
    clips = a2v.get("clips") if isinstance(a2v, dict) else None
    return list(clips) if isinstance(clips, list) else []


def latents_shas(meta: dict) -> list[dict | None]:
    return [c.get("latents_sha256") for c in h3_clips(meta)]


def chain_log_lines(excerpt: str) -> list[str]:
    return [ln for ln in (excerpt or "").splitlines() if "generate_chain:" in ln]


def _norm_clips(clips: Any) -> Any:
    """クリップの欄のうち空（None・空のリスト）のものを落とす（G-C0 の後に増えた ``references``・``source_audio``）。"""
    if not isinstance(clips, list):
        return clips
    return [{k: v for k, v in c.items() if v not in (None, [])} if isinstance(c, dict) else c for c in clips]


def request_diff(actual: dict, archived: dict) -> dict:
    out: dict[str, dict] = {"critical": {}, "ignored_by_h3": {}, "other": {}}
    for key in sorted(set(actual) | set(archived)):
        a, b = actual.get(key, "<無い>"), archived.get(key, "<無い>")
        if key == "clips":
            a, b = _norm_clips(a), _norm_clips(b)
        if a == b:
            continue
        bucket = ("critical" if key in GA0_REQUEST_CRITICAL else
                  "ignored_by_h3" if key in GA0_REQUEST_IGNORED_BY_H3 else "other")
        out[bucket][key] = {"now": a, "g_c0": b}
    return out


def judge_common(job: dict, meta: dict, *, attention: str, frames: list[int], width: int, height: int,
                 steps: int = 20) -> dict:
    """どの門にも共通の条件（ジョブ・エンジン・重み・注意の実装・ステップ・shift・クリップ・総フレーム）。"""
    checks: dict[str, dict] = {}
    h3 = h3_block(meta)
    sel = ((meta.get("models") or {}).get("selection") or {}).get("transformer") or {}
    req = meta.get("request") or {}
    check(checks, "job_completed", job.get("status") == "completed", job.get("status"), "completed")
    check(checks, "backend_is_minimax_h3", meta.get("backend") == REAL_BACKEND_H3, meta.get("backend"),
          REAL_BACKEND_H3)
    check(checks, "transformer_is_fl2va_default", sel.get("name") == FL2VA_NAME and sel.get("file") == FL2VA_FILE,
          sel, {"name": FL2VA_NAME, "file": FL2VA_FILE})
    check(checks, f"attention_used_{attention}", meta.get("attention_used") == attention, meta.get("attention_used"),
          attention)
    check(checks, f"steps_{steps}_shift_12_3",
          (h3.get("num_inference_steps"), h3.get("shift_video"), h3.get("shift_audio")) == (steps, 12.0, 3.0),
          [h3.get("num_inference_steps"), h3.get("shift_video"), h3.get("shift_audio")], [steps, 12.0, 3.0])
    check(checks, "clip_num_frames", (meta.get("chain") or {}).get("clip_num_frames") == frames,
          (meta.get("chain") or {}).get("clip_num_frames"), frames)
    check(checks, "total_frames", (meta.get("chain") or {}).get("total_frames") == total_frames(frames),
          (meta.get("chain") or {}).get("total_frames"), total_frames(frames))
    res = (meta.get("output") or {}).get("resolution")
    check(checks, "resolution", res == f"{width}x{height}", res, f"{width}x{height}")
    check(checks, "embed_mp4_metadata_false", req.get("embed_mp4_metadata") is False, req.get("embed_mp4_metadata"),
          False)
    return checks


def judge_fingerprint(checks: dict, job: dict, meta: dict, archived: dict, expect_sha: str) -> dict:
    """mp4 と各クリップの潜在の指紋を G-C0 と比べる（GA0・GA1）。"""
    lat = latents_shas(meta)
    check(checks, "mp4_sha256_matches_g_c0", (job.get("sha256") or "") == expect_sha, job.get("sha256"), expect_sha)
    arch = archived.get("latents") or []
    check(checks, "latents_sha256_match_g_c0", bool(arch) and lat == arch, lat, arch or "<g_c0.json が読めない>")
    return {"latents_sha256": lat, "latents_equal_g_c0_per_clip": [a == b for a, b in zip(lat, arch)] if arch else None}


def judge_a2v_shape(checks: dict, meta: dict, frames: list[int], offsets: list[int], strength: float,
                    audio_id: str | None) -> dict:
    """``metadata.a2v.clips[i]`` が全クリップにあり、幾何から決まる値が台本の計算と同じこと（GA1〜GA3 共通）。"""
    clips = a2v_clips(meta)
    exp = expected_a2v(frames, offsets, strength)
    check(checks, "a2v.clips_one_per_clip", len(clips) == len(frames) and all(isinstance(c, dict) for c in clips),
          [None if c is None else "dict" for c in clips], ["dict"] * len(frames))
    keys = ("strength", "offset_frames", "mask_value", "start_sample", "required_samples", "audio_output")
    actual = [{k: (c or {}).get(k) for k in keys} for c in clips]
    check(checks, "a2v.clips_geometry", actual == exp, actual, exp)
    if audio_id is not None:
        ids = [(c or {}).get("source_audio_id") for c in clips]
        check(checks, "a2v.clips_source_audio_id", ids == [audio_id] * len(frames), ids, [audio_id] * len(frames))
    h3_has = ["a2v" in c for c in h3_clips(meta)]
    return {"a2v_clips": clips, "h3_clips_have_a2v": h3_has,
            "encoded_samples": [(c or {}).get("encoded_samples") for c in clips],
            "audio_encode_seconds": [(c.get("seconds") or {}).get("audio_encode") for c in h3_clips(meta)]}


def judge_ga0(job: dict, meta: dict, archived: dict, prompts: list[str], expect_sha: str) -> dict:
    checks = judge_common(job, meta, attention="sdpa", frames=GA0_FRAMES, width=640, height=384)
    records: dict[str, Any] = {}
    h3 = h3_block(meta)
    req = meta.get("request") or {}
    _, _, composed = expected_clip_prompts(prompts, len(GA0_FRAMES))
    check(checks, "seeds_42_43_44", h3.get("seeds") == [42, 43, 44], h3.get("seeds"), [42, 43, 44])
    psha = [c.get("prompt_sha256") for c in h3_clips(meta)]
    check(checks, "clip_prompts_composed_as_g_c0", psha == [sha_text(t) for t in composed], psha,
          [sha_text(t) for t in composed])
    check(checks, "no_lora_no_crop_no_negative",
          req.get("loras") in ([], None) and req.get("crop_output") is None and not req.get("nag_enabled")
          and not req.get("negative_prompt"),
          {k: req.get(k) for k in ("loras", "crop_output", "nag_enabled", "negative_prompt")},
          {"loras": [], "crop_output": None, "nag_enabled": False, "negative_prompt": ""})
    check(checks, "no_a2v_block", "a2v" not in meta and not any("a2v" in c for c in h3_clips(meta)),
          {"a2v": "a2v" in meta, "h3.clips[i].a2v": [("a2v" in c) for c in h3_clips(meta)]}, "どちらも無い")
    if archived.get("request"):
        diff = request_diff(req, archived["request"])
        records["request_diff_vs_g_c0"] = diff
        check(checks, "request_critical_fields_equal_g_c0", not diff["critical"], diff["critical"], {})
    records.update(judge_fingerprint(checks, job, meta, archived, expect_sha))
    return {"checks": checks, "records": records}


def judge_ga1(job: dict, meta: dict, archived: dict, expect_sha: str, audio_id: str | None) -> dict:
    checks = judge_common(job, meta, attention="sdpa", frames=GA0_FRAMES, width=640, height=384)
    records = judge_fingerprint(checks, job, meta, archived, expect_sha)
    records.update(judge_a2v_shape(checks, meta, GA0_FRAMES, seam_starts(GA0_FRAMES), GA1_STRENGTH, audio_id))
    enc = records["encoded_samples"]
    check(checks, "a2v.clips_encoded_samples_0", len(enc) == len(GA0_FRAMES) and all(e == 0 for e in enc), enc,
          [0] * len(GA0_FRAMES))
    return {"checks": checks, "records": records}


def judge_ga2(job: dict, meta: dict, audio_id: str | None) -> dict:
    checks = judge_common(job, meta, attention="sdpa", frames=GA2_FRAMES, width=640, height=384)
    offsets = seam_starts(GA2_FRAMES)
    records = judge_a2v_shape(checks, meta, GA2_FRAMES, offsets, GA2_STRENGTH, audio_id)
    clips = a2v_clips(meta)
    heads = [c.get("head_equals_prev_tail") for c in h3_clips(meta)]
    check(checks, "head_equals_prev_tail_clips_1_on", len(heads) == len(GA2_FRAMES) and all(h is True for h in heads[1:]),
          heads, [None] + [True] * (len(GA2_FRAMES) - 1))
    masks = [(c or {}).get("mask_value") for c in clips]
    check(checks, "a2v.mask_value_0", len(masks) == len(GA2_FRAMES) and all(m == 0 for m in masks), masks,
          [0.0] * len(GA2_FRAMES))
    outs = [(c or {}).get("audio_output") for c in clips]
    check(checks, "a2v.audio_output_decoded", outs == ["decoded"] * len(GA2_FRAMES), outs,
          ["decoded"] * len(GA2_FRAMES))
    starts = [(c or {}).get("start_sample") for c in clips]
    check(checks, "a2v.start_sample_0_136000_294400", starts == PLAN_START_SAMPLES_GA2, starts, PLAN_START_SAMPLES_GA2)
    xc = product_math_crosscheck(GA2_FRAMES, offsets)
    prod = xc.get("clip_audio_start_sample")
    check(checks, "a2v.start_sample_equals_clip_audio_start_sample", prod is not None and starts == prod, starts,
          prod if prod is not None else xc)
    records["head_equals_prev_tail"] = heads
    records["latents_sha256"] = latents_shas(meta)
    return {"checks": checks, "records": records}


def judge_ga3(job: dict, meta: dict, audio_id: str | None, turbo: bool = False) -> dict:
    """完走と記録の形だけ（目視の合否はオーナー）。``turbo``＝GA3T（4 ステップ・Turbo LoRA が当たったこと）。"""
    checks = judge_common(job, meta, attention=GA3_ATTENTION, frames=GA3_FRAMES, width=GA3_WIDTH, height=GA3_HEIGHT,
                          steps=TURBO_STEPS if turbo else 20)
    req_loras = [(lo.get("name"), lo.get("strength")) for lo in ((meta.get("request") or {}).get("loras") or [])
                 if isinstance(lo, dict)]
    want = [(TURBO_LORA, TURBO_LORA_STRENGTH)] if turbo else []
    check(checks, "request_loras", req_loras == want, req_loras, want)
    if turbo:
        applied = h3_block(meta).get("loras") or []
        names = [lo.get("name") for lo in applied if isinstance(lo, dict)]
        check(checks, "h3.loras_turbo_applied", names == [TURBO_LORA + ".safetensors"]
              and all((lo.get("pairs_applied") or 0) > 0 for lo in applied), applied,
              f"{TURBO_LORA}.safetensors が当たっている（pairs_applied > 0）")
    records = judge_a2v_shape(checks, meta, GA3_FRAMES, seam_starts(GA3_FRAMES), GA3_STRENGTH, audio_id)
    heads = [c.get("head_equals_prev_tail") for c in h3_clips(meta)]
    check(checks, "head_equals_prev_tail_clips_1_on", len(heads) == len(GA3_FRAMES) and all(h is True for h in heads[1:]),
          heads, [None] + [True] * (len(GA3_FRAMES) - 1))
    records["head_equals_prev_tail"] = heads
    records["latents_sha256"] = latents_shas(meta)
    return {"checks": checks, "records": records}


def memory_record(meta: dict) -> dict:
    """VRAM の山・所要・主記憶（記録だけ）。出どころは metadata.json（前例・G3 の報告と同じ鍵）。"""
    h3 = h3_block(meta)
    phases = h3.get("phases") or {}
    per_clip = []
    for c in (phases.get("clips") or []) if isinstance(phases, dict) else []:
        row = {}
        for name in ("encode", "reference_encode", "audio_encode", "denoise"):
            ph = c.get(name) if isinstance(c, dict) else None
            if isinstance(ph, dict):
                row[name] = {k: ph.get(k) for k in ("seconds", "peak_vram_mb", "peak_vram_reserved_mb",
                                                    "rss_process_peak_gib", "commit_process_peak_gib")
                             if k in ph}
        per_clip.append(row)
    dec = phases.get("decode") if isinstance(phases, dict) else None
    return {
        "generation_time_seconds": meta.get("generation_time_seconds"),
        "peak_vram_reserved_mb": meta.get("peak_vram_reserved_mb"),
        "peak_vram_mb": (meta.get("vram_optimization") or {}).get("peak_vram_mb"),
        "rss_peak_gib": h3.get("rss_peak_gib"),
        "commit_peak_gib": h3.get("commit_peak_gib"),
        "h3_seconds": h3.get("seconds"),
        "clip_seconds": [c.get("seconds") for c in h3_clips(meta)],
        "phases_per_clip": per_clip,
        "phase_decode": {k: dec.get(k) for k in ("seconds", "peak_vram_mb", "peak_vram_reserved_mb",
                                                 "rss_process_peak_gib", "commit_process_peak_gib")
                         if k in dec} if isinstance(dec, dict) else None,
    }


# ---------------------------------------------------------------------------------------------
# 一時 yaml とサーバー（h3_vsf_chain_gate.py の写し。HTTP に api_key の見出しを足した）
# ---------------------------------------------------------------------------------------------

API_KEY: str | None = None  # config.yaml の server.api_key（null なら送らない）


def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _headers(extra: dict | None = None) -> dict:
    h = dict(extra or {})
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _read_response(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def http(port: int, method: str, path: str, body=None, timeout: float = 60.0):
    url = base_url(port) + "/api/v1" + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=_headers(headers), method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _read_response(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _read_response(exc.read())


def upload_audio(port: int, path: Path) -> tuple[int, Any]:
    """``POST /upload/audio``（multipart/form-data、欄の名前は ``file``）。"""
    boundary = uuid.uuid4().hex
    data = path.read_bytes()
    name = path.name.replace('"', "_")
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode("utf-8")
    payload = head + data + f"\r\n--{boundary}--\r\n".encode("ascii")
    req = urllib.request.Request(base_url(port) + "/api/v1/upload/audio", data=payload, method="POST",
                                 headers=_headers({"Content-Type": f"multipart/form-data; boundary={boundary}"}))
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, _read_response(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _read_response(exc.read())


def build_temp_config(raw: dict, port: int) -> dict:
    out = copy.deepcopy(raw)
    out.setdefault("server", {})["port"] = port
    out["server"]["log_dir"] = str(RT / "logs")
    out.setdefault("output", {})["dir"] = str(RT / "outputs")
    out.setdefault("upload", {})["dir"] = str(RT / "uploads")
    out["state_file"] = str(RT / TEMP_STATE_NAME)
    vram = out.get("vram")
    if not isinstance(vram, dict):
        vram = out["vram"] = {}
    vram.update(VRAM_OVERRIDES)
    return out


def read_live_config() -> dict:
    import yaml  # アプリの仮想環境の依存（config.py が使う）

    return yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8")) or {}


def write_temp_config(port: int) -> tuple[Path, Path, dict]:
    import yaml

    RT.mkdir(parents=True, exist_ok=True)
    for sub in ("logs", "outputs", "uploads"):
        (RT / sub).mkdir(exist_ok=True)

    live = read_live_config()
    raw = build_temp_config(live, port)
    cfg = RT / TEMP_CONFIG_NAME
    cfg.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    overridden = {"vram_live": {k: (live.get("vram") or {}).get(k, "<無い>") for k in VRAM_OVERRIDES},
                  "vram_temp": dict(VRAM_OVERRIDES)}

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
    return cfg, state, overridden


def all_processes() -> list[dict]:
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
        cfg, state, overridden = write_temp_config(self.port)
        env = dict(os.environ)
        prefix = ""
        for d in (REPO / "tools" / "uv", REPO / "tools" / "ffmpeg" / "bin"):
            if d.exists():
                prefix += f"{d};"
        env["PATH"] = prefix + env.get("PATH", "")
        env["UV_PYTHON_INSTALL_DIR"] = str(REPO / ".python")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        RT.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8", errors="replace")
        self._fh.write(f"\n===== {now_iso()} h3_chain_a2v_gate.py: main.py --config {cfg} =====\n")
        self._fh.flush()
        cmd = [str(PY), "-u", str(REPO / "main.py"), "--config", str(cfg)]
        self.proc = subprocess.Popen(cmd, cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=self._fh,
                                     stderr=subprocess.STDOUT, env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.record.update(reused=False, pid=self.proc.pid, cmd=cmd, config=str(cfg), state=str(state),
                           config_overrides=overridden, log=str(self.log_path), started_at=now_iso())
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


def worker_log_path() -> Path:
    p = RT / "logs" / "h3_worker.log"
    if p.exists():
        return p
    found = sorted((RT / "logs").glob("*h3*worker*.log")) if (RT / "logs").exists() else []
    return found[0] if found else p


def log_offset() -> int:
    p = worker_log_path()
    return p.stat().st_size if p.exists() else 0


def log_since(offset: int) -> str:
    p = worker_log_path()
    if not p.exists():
        return ""
    with p.open("rb") as fh:
        fh.seek(offset)
        return fh.read().decode("utf-8", errors="replace")


class GpuSampler:
    """``nvidia-smi`` の memory.used を数秒おきに読む（記録だけ。GPU に計算はさせない）。"""

    def __init__(self, interval: float = GPU_SAMPLE_SEC) -> None:
        self.interval = interval
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.error: str | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20)
                val = int((r.stdout or "").strip().splitlines()[0])
                self.samples.append((round(time.monotonic() - self._t0, 1), val))
            except Exception as exc:  # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.interval)

    def start(self) -> "GpuSampler":
        self._thread.start()
        return self

    def finish(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=30)
        vals = [v for _, v in self.samples]
        return {"interval_s": self.interval, "samples": len(vals), "max_memory_used_mib": max(vals) if vals else None,
                "first_mib": vals[0] if vals else None, "error": self.error}


# ---------------------------------------------------------------------------------------------
# 読み込み・音声・1 本のジョブ
# ---------------------------------------------------------------------------------------------

def ensure_pipeline(port: int, args, run_rec: dict) -> bool:
    """H3（FL2VA＝登録名 default）を載せる。load は 1 回だけ。以後 /status を 10 秒おき。"""
    body = {"base_model": H3_BASE_MODEL, "models": {"transformer": FL2VA_NAME}}
    load: dict[str, Any] = {"requested": body, "status_polls": 0}
    run_rec["load"] = load
    t0 = time.monotonic()
    try:
        code, resp = http(port, "POST", "/pipeline/load", body, timeout=args.load_timeout_sec)
        load["response"] = {"http": code, "body": resp}
        if code not in (200, 409):
            load["error"] = f"/pipeline/load が {code}: {json.dumps(resp, ensure_ascii=False)}"
            log(load["error"])
            return False
    except Exception as exc:  # noqa: BLE001 -- 読み込みが長く応答が切れても /status で待つ
        load["response"] = {"error": f"{type(exc).__name__}: {exc}"}
    deadline = t0 + args.load_timeout_sec
    while True:
        try:
            code, st = http(port, "GET", "/status", timeout=30)
        except Exception as exc:  # noqa: BLE001
            code, st = None, {"error": f"{type(exc).__name__}: {exc}"}
        load["status_polls"] += 1
        load["last_status"] = {k: (st or {}).get(k) for k in ("pipeline_loaded", "state", "base_model",
                                                              "acceleration")} if isinstance(st, dict) else st
        if code == 200 and st.get("pipeline_loaded") and st.get("base_model") == H3_BASE_MODEL \
                and st.get("state") in ("ready", "running"):
            break
        if code == 200 and st.get("state") == "error":
            load["error"] = f"パイプラインの状態が error: {load['last_status']}"
            return False
        if time.monotonic() > deadline:
            load["error"] = f"{args.load_timeout_sec} 秒待っても読み込みが終わらない（最後の状態 {load['last_status']}）"
            return False
        time.sleep(10)
    load["seconds"] = round(time.monotonic() - t0, 1)
    log(f"H3 が載った（{load['seconds']} 秒）")
    return True


def ensure_audio(port: int, args, run_rec: dict, ctx: dict) -> bool:
    if ctx.get("audio_id"):
        return True
    path = Path(args.audio).resolve()
    rec: dict[str, Any] = {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size
                           if path.is_file() else None}
    run_rec["audio_upload"] = rec
    if not path.is_file():
        rec["error"] = "音声のファイルが無い"
        log(f"音声が無い: {path}")
        return False
    code, resp = upload_audio(port, path)
    rec["http"] = code
    rec["response"] = resp
    if code != 200 or not isinstance(resp, dict) or not resp.get("audio_id"):
        rec["error"] = f"/upload/audio が {code}"
        log(f"音声のアップロードが失敗: {code} {json.dumps(resp, ensure_ascii=False)}")
        return False
    ctx["audio_id"] = resp["audio_id"]
    log(f"音声を上げた: audio_id={ctx['audio_id']}")
    return True


def run_chain_job(port: int, args, label: str, job_dir: Path, body: dict, note: str) -> tuple[dict, dict]:
    """POST /generate/chain → 待つ → metadata.json を写す → sha256・ワーカーのログ。(記録, metadata)。"""
    job_dir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"label": label, "dir": str(job_dir), "note": note, "endpoint": "POST /api/v1/generate/chain",
                           "body": body, "started_at": now_iso(), "status": None, "error": None}
    write_json(job_dir / "request.json", {"endpoint": "POST /api/v1/generate/chain", "body": body, "note": note,
                                          "remark": "送った本文。サーバーが受けた本文は metadata.json の request"})
    log(f"=== {label}: {note} ===")
    offset = log_offset()
    sampler = GpuSampler().start()
    t0 = time.monotonic()
    meta: dict = {}
    try:
        code, sub = http(port, "POST", "/generate/chain", body, timeout=600)
        rec["submit"] = {"http": code, "body": sub}
        if code != 202 or not isinstance(sub, dict) or not sub.get("job_id"):
            rec.update(status="submit_error", error=f"HTTP {code}: {json.dumps(sub, ensure_ascii=False)}",
                       finished_at=now_iso())
            log(f"/generate/chain が拒否: {rec['error']}")
            return rec, meta
        job_id = sub["job_id"]
        rec["job_id"] = job_id
        log(f"job_id={job_id}")
        samples: list[dict] = []
        last_key = None
        final: dict = {}
        while True:
            try:
                code, st = http(port, "GET", f"/jobs/{job_id}", timeout=60)
            except Exception as exc:  # noqa: BLE001
                samples.append({"t": round(time.monotonic() - t0, 1), "error": f"{type(exc).__name__}: {exc}"})
                time.sleep(POLL_SEC)
                continue
            if code != 200 or not isinstance(st, dict):
                samples.append({"t": round(time.monotonic() - t0, 1), "http": code, "body": st})
            else:
                final = st
                key = (st.get("status"), st.get("stage"), st.get("clip"), st.get("current_step"))
                if key != last_key:
                    samples.append({"t": round(time.monotonic() - t0, 1), **{k: st.get(k) for k in (
                        "status", "progress", "stage", "clip", "clip_count", "current_step", "total_steps")}})
                    log(f"  job {job_id[:8]} status={st.get('status')} progress={st.get('progress')} "
                        f"stage={st.get('stage')} clip={st.get('clip')}/{st.get('clip_count')}")
                    last_key = key
                if st.get("status") in ("completed", "failed", "cancelled"):
                    break
            if time.monotonic() - t0 > args.job_timeout_sec:
                rec["wait_gave_up"] = f"{args.job_timeout_sec} 秒待っても終わらない（ジョブはバックエンドで続いている）"
                break
            time.sleep(POLL_SEC)
        rec["poll_samples"] = samples
        rec["job_final"] = final
        rec["status"] = "wait_gave_up" if rec.get("wait_gave_up") else (final.get("status") or "unknown")
        rec["seconds_wall"] = round(time.monotonic() - t0, 1)
    finally:
        rec["gpu_sampler"] = sampler.finish()
    excerpt = log_since(offset)
    if excerpt:
        (job_dir / "worker_log_excerpt.txt").write_text(excerpt, encoding="utf-8")
    rec["chain_log_lines"] = chain_log_lines(excerpt)
    if rec["status"] != "completed":
        rec["error"] = rec.get("wait_gave_up") or (rec.get("job_final") or {}).get("error")
        rec["finished_at"] = now_iso()
        log(f"ジョブが {rec['status']}: {rec['error']}")
        return rec, meta
    out_dir = RT / "outputs" / rec["job_id"]
    mp4 = out_dir / "output.mp4"
    rec["mp4"] = str(mp4)  # 保管庫には写さない（_rt に置いたまま）
    rec["sha256"] = sha256(mp4)
    src = out_dir / "metadata.json"
    if src.exists():
        shutil.copy2(src, job_dir / "metadata.json")
        meta = json.loads(src.read_text(encoding="utf-8"))
    else:
        rec["metadata_missing"] = str(src)
    rec["finished_at"] = now_iso()
    log(f"完了: sha256={(rec.get('sha256') or '')[:16]}…")
    return rec, meta


# ---------------------------------------------------------------------------------------------
# 門
# ---------------------------------------------------------------------------------------------

def expected_sha(args) -> str:
    return (args.expect_sha or G_C0_MP4_SHA).lower()


def job_result_fields(job: dict, meta: dict) -> dict:
    return {"job_id": job.get("job_id"), "status": job.get("status"), "error": job.get("error"),
            "submit": job.get("submit"), "sha256": job.get("sha256"), "mp4": job.get("mp4"),
            "seconds_wall": job.get("seconds_wall"), "chain_log_lines": job.get("chain_log_lines"),
            "gpu_sampler": job.get("gpu_sampler"), "memory": memory_record(meta) if meta else None,
            "metadata_excerpt": metadata_excerpt(meta) if meta else None}


def metadata_excerpt(meta: dict) -> dict:
    h3 = h3_block(meta)
    return {k: meta.get(k) for k in ("backend", "attention_used", "vram16_mode_used", "block_swap_prefetch_used",
                                     "keep_resident_used", "seed_used", "generation_time_seconds",
                                     "peak_vram_reserved_mb", "chain", "output", "a2v")} | {
        "h3": {k: h3.get(k) for k in ("num_inference_steps", "shift_video", "shift_audio", "seeds", "decode",
                                      "audio_drift_flags", "audio_latent_excess")},
        "h3_clips": [{k: c.get(k) for k in ("prompt_sha256", "seconds", "latents_sha256", "head_equals_prev_tail",
                                            "a2v")} for c in h3_clips(meta)],
    }


def finish_gate(name: str, path: Path, res: dict, summary: dict) -> bool:
    res["pass"] = all_pass(res.get("checks") or {})
    res["finished_at"] = now_iso()
    write_json(path, res)
    failed = [k for k, v in (res.get("checks") or {}).items() if not v.get("pass")]
    summary["gates"][name] = {"pass": res["pass"], "result": str(path), "failed": failed}
    log(f"{name}: {'合格' if res['pass'] else '不合格'}" + (f"（落ちた基準: {failed}）" if failed else ""))
    return res["pass"]


def gate_ga0(port: int, args, summary: dict, ctx: dict, run_rec: dict) -> bool:
    log("##### GA0: 不変（G-C0 の本文をそのまま） #####")
    prompts = load_prompts()
    archived = load_archived_g_c0()
    gdir = HERE / "ga0"
    job, meta = run_chain_job(port, args, "ga0", gdir, ga0_body(prompts), "GA0: G-C0 と同じ連結生成（音声なし）")
    summary["jobs"]["ga0"] = job
    judged = judge_ga0(job, meta, archived, prompts, expected_sha(args)) if job.get("status") == "completed" \
        else {"checks": {"job_completed": {"pass": False, "actual": job.get("status"), "expected": "completed"}}}
    res = {"gate": "GA0", "label": args.label, "started_at": job.get("started_at"),
           "expected_sha256": expected_sha(args), "expected_sha256_source": "--expect-sha" if args.expect_sha
           else "G-C0（h3-chain-gates-2026-10-29/g_c0.json）", **job_result_fields(job, meta), **judged}
    ctx["ga0_sha"] = job.get("sha256")
    ctx["ga0_latents"] = latents_shas(meta) if meta else None
    return finish_gate("GA0", gdir / "result.json", res, summary)


def gate_ga1(port: int, args, summary: dict, ctx: dict, run_rec: dict) -> bool:
    log("##### GA1: Strength 0 は効き目なし #####")
    gdir = HERE / "ga1"
    if not ensure_audio(port, args, run_rec, ctx):
        res = {"gate": "GA1", "label": args.label, "audio_upload": run_rec.get("audio_upload"),
               "checks": {"audio_uploaded": {"pass": False, "actual": run_rec.get("audio_upload"),
                                             "expected": "audio_id が返る"}}}
        return finish_gate("GA1", gdir / "result.json", res, summary)
    prompts = load_prompts()
    archived = load_archived_g_c0()
    body = ga1_body(prompts, ctx["audio_id"])
    job, meta = run_chain_job(port, args, "ga1", gdir, body,
                              "GA1: GA0 と同じ本文の全クリップに source_audio（strength 0.0・offset S_i）")
    summary["jobs"]["ga1"] = job
    judged = judge_ga1(job, meta, archived, expected_sha(args), ctx["audio_id"]) \
        if job.get("status") == "completed" else \
        {"checks": {"job_completed": {"pass": False, "actual": job.get("status"), "expected": "completed"}}}
    res = {"gate": "GA1", "label": args.label, "started_at": job.get("started_at"), "audio_id": ctx["audio_id"],
           "audio_upload": run_rec.get("audio_upload"), "expected_sha256": expected_sha(args),
           "sha256_equals_ga0_this_run": (job.get("sha256") == ctx["ga0_sha"]) if ctx.get("ga0_sha") else None,
           "latents_equal_ga0_this_run": (latents_shas(meta) == ctx["ga0_latents"])
           if ctx.get("ga0_latents") and meta else None,
           "math_crosscheck": product_math_crosscheck(GA0_FRAMES, seam_starts(GA0_FRAMES)),
           **job_result_fields(job, meta), **judged}
    return finish_gate("GA1", gdir / "result.json", res, summary)


def gate_ga2(port: int, args, summary: dict, ctx: dict, run_rec: dict) -> bool:
    log("##### GA2: Strength 1.0・継ぎ目が格子に乗らない形 #####")
    gdir = HERE / "ga2"
    if not ensure_audio(port, args, run_rec, ctx):
        res = {"gate": "GA2", "label": args.label, "audio_upload": run_rec.get("audio_upload"),
               "checks": {"audio_uploaded": {"pass": False, "actual": run_rec.get("audio_upload"),
                                             "expected": "audio_id が返る"}}}
        return finish_gate("GA2", gdir / "result.json", res, summary)
    body = ga2_body(load_prompts(), ctx["audio_id"])
    job, meta = run_chain_job(port, args, "ga2", gdir, body,
                              "GA2: [124, 141, 124]・strength 1.0・offset 0・102・221")
    summary["jobs"]["ga2"] = job
    judged = judge_ga2(job, meta, ctx["audio_id"]) if job.get("status") == "completed" else \
        {"checks": {"job_completed": {"pass": False, "actual": job.get("status"), "expected": "completed"}}}
    res = {"gate": "GA2", "label": args.label, "started_at": job.get("started_at"), "audio_id": ctx["audio_id"],
           "math_crosscheck": product_math_crosscheck(GA2_FRAMES, seam_starts(GA2_FRAMES)),
           **job_result_fields(job, meta), **judged}
    return finish_gate("GA2", gdir / "result.json", res, summary)


def _gate_ga3_like(port: int, args, summary: dict, ctx: dict, run_rec: dict, turbo: bool) -> bool:
    name, key = ("GA3T", "ga3t") if turbo else ("GA3", "ga3")
    kind = f"Turbo LoRA {TURBO_LORA_STRENGTH}・{TURBO_STEPS} ステップ" if turbo else "LoRA なし・20 ステップ"
    log(f"##### {name}: Strength 0.5・{kind}・目視用（機械は完走と記録の形だけ） #####")
    gdir = HERE / key
    if not ensure_audio(port, args, run_rec, ctx):
        res = {"gate": name, "label": args.label, "audio_upload": run_rec.get("audio_upload"),
               "checks": {"audio_uploaded": {"pass": False, "actual": run_rec.get("audio_upload"),
                                             "expected": "audio_id が返る"}}}
        return finish_gate(name, gdir / "result.json", res, summary)
    body = (ga3t_body if turbo else ga3_body)(args.ga3_prompt, ctx["audio_id"])
    job, meta = run_chain_job(port, args, key, gdir, body,
                              f"{name}: 3×209・1344×768・sage・strength 0.5・offset 0・187・374・{kind}（目視用）")
    summary["jobs"][key] = job
    judged = judge_ga3(job, meta, ctx["audio_id"], turbo=turbo) if job.get("status") == "completed" else \
        {"checks": {"job_completed": {"pass": False, "actual": job.get("status"), "expected": "completed"}}}
    res = {"gate": name, "label": args.label, "started_at": job.get("started_at"), "audio_id": ctx["audio_id"],
           "visual_verdict": "オーナーの目視（台本は合否を付けない。pass は完走と記録の形の基準だけ）",
           "math_crosscheck": product_math_crosscheck(GA3_FRAMES, seam_starts(GA3_FRAMES)),
           **job_result_fields(job, meta), **judged}
    if meta:
        res["h3_loras"] = h3_block(meta).get("loras")
    return finish_gate(name, gdir / "result.json", res, summary)


def gate_ga3(port: int, args, summary: dict, ctx: dict, run_rec: dict) -> bool:
    return _gate_ga3_like(port, args, summary, ctx, run_rec, turbo=False)


def gate_ga3t(port: int, args, summary: dict, ctx: dict, run_rec: dict) -> bool:
    return _gate_ga3_like(port, args, summary, ctx, run_rec, turbo=True)


GATES = {"GA0": gate_ga0, "GA1": gate_ga1, "GA2": gate_ga2, "GA3": gate_ga3, "GA3T": gate_ga3t}


# ---------------------------------------------------------------------------------------------
# まとめ・実行
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
    return {"script": "h3_chain_a2v_gate.py", "jobs": {}, "gates": {}, "runs": []}


def flush(summary: dict) -> None:
    write_json(HERE / "summary.json", summary)


def gates_of(args) -> list[str]:
    return list(GATES) if args.gate == "all" else [args.gate]


def run(args, summary: dict, run_rec: dict) -> int:
    port = args.port
    code, st = http(port, "GET", "/status", timeout=30)
    run_rec["status_before"] = st
    if code != 200:
        log(f"/status が {code}")
        return 1
    if not ensure_pipeline(port, args, run_rec):
        log(f"読み込みで止まる: {run_rec['load'].get('error')}")
        return 1
    flush(summary)
    ctx: dict[str, Any] = {}
    for g in gates_of(args):
        try:
            ok = GATES[g](port, args, summary, ctx, run_rec)
        except Exception as exc:  # noqa: BLE001 -- 例外も門の失敗として記録して止まる
            summary["gates"][g] = {"pass": False, "error": f"{type(exc).__name__}: {exc}",
                                   "traceback": traceback.format_exc()}
            ok = False
        flush(summary)
        if not ok:
            log(f"{g} が合格しなかったので、ここで止まる（次の門へは進まない）")
            return 3
    return 0


def plan_for_dry_run(args) -> dict:
    prompts = load_prompts()
    archived = load_archived_g_c0()
    aid = "<POST /upload/audio が返す audio_id>"
    plan: dict[str, Any] = {
        "port": args.port, "label": args.label,
        "temp_config": str(RT / TEMP_CONFIG_NAME),
        "temp_config_overrides": {"server.port": args.port, "server.log_dir": str(RT / "logs"),
                                  "output.dir": str(RT / "outputs"), "upload.dir": str(RT / "uploads"),
                                  "state_file": str(RT / TEMP_STATE_NAME),
                                  **{f"vram.{k}": v for k, v in VRAM_OVERRIDES.items()}},
        "server_cmd": [str(PY), "-u", str(REPO / "main.py"), "--config", str(RT / TEMP_CONFIG_NAME)],
        "load": {"POST /api/v1/pipeline/load": {"base_model": H3_BASE_MODEL, "models": {"transformer": FL2VA_NAME}}},
        "audio": {"POST /api/v1/upload/audio": str(Path(args.audio).resolve()),
                  "exists": Path(args.audio).is_file(),
                  "size_bytes": Path(args.audio).stat().st_size if Path(args.audio).is_file() else None,
                  "used_by": ["GA1", "GA2", "GA3"]},
        "stop_rule": "どれかの門が合格しなければ（422 などの拒否・ジョブの失敗・基準の不合格）そこで止まる",
    }
    gates = gates_of(args)
    if "GA0" in gates:
        body = ga0_body(prompts)
        plan["GA0"] = {"POST /api/v1/generate/chain": body,
                       "body_equals_g_c0_request_body": body == archived.get("request_body"),
                       "expect": {"mp4_sha256": expected_sha(args), "latents_sha256": archived.get("latents"),
                                  "total_frames": total_frames(GA0_FRAMES), "a2v_block": "無い"},
                       "result": str(HERE / "ga0" / "result.json")}
    if "GA1" in gates:
        offs = seam_starts(GA0_FRAMES)
        plan["GA1"] = {"POST /api/v1/generate/chain": ga1_body(prompts, aid),
                       "expect": {"mp4_sha256": expected_sha(args), "latents_sha256": archived.get("latents"),
                                  "a2v.clips[i].encoded_samples": [0, 0, 0],
                                  "a2v.clips[i]": expected_a2v(GA0_FRAMES, offs, GA1_STRENGTH)},
                       "offsets_equal_plan": offs == PLAN_OFFSETS["ga1"],
                       "math_crosscheck": product_math_crosscheck(GA0_FRAMES, offs),
                       "result": str(HERE / "ga1" / "result.json")}
    if "GA2" in gates:
        offs = seam_starts(GA2_FRAMES)
        plan["GA2"] = {"POST /api/v1/generate/chain": ga2_body(prompts, aid),
                       "expect": {"total_frames": total_frames(GA2_FRAMES),
                                  "head_equals_prev_tail": [None, True, True],
                                  "a2v.clips[i]": expected_a2v(GA2_FRAMES, offs, GA2_STRENGTH),
                                  "start_sample_plan": PLAN_START_SAMPLES_GA2},
                       "offsets_equal_plan": offs == PLAN_OFFSETS["ga2"],
                       "start_samples_equal_plan": [start_sample(o) for o in offs] == PLAN_START_SAMPLES_GA2,
                       "total_frames_equal_plan": total_frames(GA2_FRAMES) == PLAN_TOTAL_FRAMES["ga2"],
                       "math_crosscheck": product_math_crosscheck(GA2_FRAMES, offs),
                       "result": str(HERE / "ga2" / "result.json")}
    if "GA3" in gates:
        offs = seam_starts(GA3_FRAMES)
        last_end_sec = (start_sample(offs[-1]) + required_samples(GA3_FRAMES[-1])) / AUDIO_SR
        plan["GA3"] = {"POST /api/v1/generate/chain": ga3_body(args.ga3_prompt, aid),
                       "expect": {"total_frames": total_frames(GA3_FRAMES),
                                  "head_equals_prev_tail": [None, True, True],
                                  "a2v.clips[i]": expected_a2v(GA3_FRAMES, offs, GA3_STRENGTH),
                                  "attention_used": GA3_ATTENTION, "resolution": f"{GA3_WIDTH}x{GA3_HEIGHT}"},
                       "audio_needed_sec_for_last_clip": round(last_end_sec, 3),
                       "offsets_equal_plan": offs == PLAN_OFFSETS["ga3"],
                       "total_frames_equal_plan": total_frames(GA3_FRAMES) == PLAN_TOTAL_FRAMES["ga3"],
                       "math_crosscheck": product_math_crosscheck(GA3_FRAMES, offs),
                       "records_only": ["VRAM の山（metadata・nvidia-smi）", "所要秒", "主記憶（rss・commit の山）"],
                       "result": str(HERE / "ga3" / "result.json")}
    if "GA3T" in gates:
        offs = seam_starts(GA3_FRAMES)
        body = ga3t_body(args.ga3_prompt, aid)
        diff_vs_ga3 = {k: {"ga3t": body.get(k), "ga3": ga3_body(args.ga3_prompt, aid).get(k, "<無い>")}
                       for k in sorted(set(body) | set(ga3_body(args.ga3_prompt, aid)))
                       if body.get(k, "<無い>") != ga3_body(args.ga3_prompt, aid).get(k, "<無い>")}
        plan["GA3T"] = {"POST /api/v1/generate/chain": body,
                        "diff_vs_ga3_body": diff_vs_ga3,
                        "lora_file_exists": (REPO / "models" / "MiniMaxH3" / "LoRA" / f"{TURBO_LORA}.safetensors").is_file(),
                        "expect": {"total_frames": total_frames(GA3_FRAMES),
                                   "head_equals_prev_tail": [None, True, True],
                                   "a2v.clips[i]": expected_a2v(GA3_FRAMES, offs, GA3_STRENGTH),
                                   "attention_used": GA3_ATTENTION, "resolution": f"{GA3_WIDTH}x{GA3_HEIGHT}",
                                   "steps_shift": [TURBO_STEPS, 12.0, 3.0],
                                   "h3.loras": f"{TURBO_LORA}.safetensors・pairs_applied > 0"},
                        "records_only": ["VRAM の山（metadata・nvidia-smi）", "所要秒", "主記憶（rss・commit の山）"],
                        "result": str(HERE / "ga3t" / "result.json")}
    return plan


def dry_run(args) -> int:
    print(json.dumps(plan_for_dry_run(args), ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        description="MiniMax H3 の連結生成のクリップごとの A2V の門 GA0〜GA3（HTTP 直送・別ポートの実バックエンド）")
    ap.add_argument("--gate", required=True, choices=("GA0", "GA1", "GA2", "GA3", "GA3T", "all"),
                    help="回す門（all＝GA0 → GA1 → GA2 → GA3 → GA3T を 1 つのサーバーで。不合格で止まる）")
    ap.add_argument("--label", default="run", help="結果の JSON に残す名前（既定 run）")
    ap.add_argument("--expect-sha", default=None,
                    help=f"GA0／GA1 で比べる mp4 の sha256（既定は G-C0 の {G_C0_MP4_SHA[:16]}…）")
    ap.add_argument("--audio", default=str(AUDIO_DEFAULT), help="GA1〜GA3 で上げる音声（既定 ref-voice-patra001.wav）")
    ap.add_argument("--ga3-prompt", default=GA3_PROMPT_DEFAULT, help="GA3 の共通のプロンプト")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT, help=f"一時サーバーのポート（既定 {PORT_DEFAULT}）")
    ap.add_argument("--keep-server", action="store_true",
                    help="終わりに unload もサーバーの停止もしない（次の回で同じポートのサーバーを使い回す）")
    ap.add_argument("--load-timeout-sec", type=float, default=1200.0)
    ap.add_argument("--job-timeout-sec", type=float, default=7200.0)
    ap.add_argument("--dry-run", action="store_true", help="送る本文と期待値を表示して終わる（サーバーも GPU も使わない）")
    args = ap.parse_args(argv)
    if args.expect_sha is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", args.expect_sha):
        ap.error("--expect-sha は 16 進 64 文字")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.label):
        ap.error("--label は英数字と . _ - だけ")
    return args


def print_compact(summary: dict, gates: list[str]) -> None:
    print("---- まとめ ----")
    for g in gates:
        r = summary["gates"].get(g)
        if r is None:
            print(f"{g}: 回していない")
            continue
        mark = "合格" if r.get("pass") else "不合格"
        extra = f" 落ちた基準={r.get('failed')}" if r.get("failed") else ""
        if r.get("error"):
            extra += f" 例外={r['error']}"
        print(f"{g}: {mark}{extra}  ({r.get('result', '-')})")


def main(argv: list[str] | None = None) -> int:
    global API_KEY
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    API_KEY = ((read_live_config().get("server") or {}).get("api_key") or None)
    summary = load_summary()
    run_rec: dict[str, Any] = {"started_at": now_iso(), "args": vars(args).copy(), "api_key_sent": bool(API_KEY)}
    # 前の回の門の結果は、この回で回す門の分だけ消す（回していない門を「回した」と読まないように）。
    for g in gates_of(args):
        summary["gates"].pop(g, None)
    summary["runs"].append(run_rec)
    srv = Server(args.port)
    code = 1
    try:
        srv.start()
        run_rec["server"] = srv.record
        flush(summary)
        code = run(args, summary, run_rec)
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
        print_compact(summary, gates_of(args))
        log(f"記録: {HERE / 'summary.json'}（終了コード {code}）")
    return code


if __name__ == "__main__":
    sys.exit(main())
