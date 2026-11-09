"""MiniMax H3 のネガティブプロンプト（VSF）を連結生成（Chained）の経路へつないだことを確かめる門 GC0〜GC3 を、
MCP サーバー経由で回す台本（台帳 §1-127 の続き）。

製品コードは import しない。部品は次の雛形から写した。

- ``Docs/Outputs-archive/h3-vsf-2026-11-09/h3_vsf_gate.py``: 一時 yaml（``write_temp_config``）・``Server``（別ポートで
  実バックエンドを起こし、終わりに unload → 停止）・MCP の stdio クライアント（``Mcp``・``check_tool_schemas``・
  ``ensure_pipeline``・``wait_job``）・ワーカーのログの読み方（``_rt\\logs\\h3_worker.log`` の差分）。
- ``Docs/Outputs-archive/h3-chain-gates-2026-10-29/g_chain.py``: 門 G-C0 の条件とプロンプトの組み立て
  （``compose_clip_prompt``・``split_prompts``・``expected_clip_prompts``。写しは逐語）。
- ``Docs/Outputs-archive/h3-retake-2026-11-08/h3_retake_gate.py``: Retake の送り方（``upload_video`` に大きい
  ``max_frames`` を渡して尺を測る → ``submit_chain`` に ``retake_video_id``・``retake_window_start_sec``）。

門:
  gc0  ネガティブなしの連結生成の回帰。条件は保管庫の G-C0 と同じ（FL2VA＝登録名 ``default``・640×384・124×3・
       ``prompts_c.json`` の組み立て・sdpa・20 ステップ・shift 12/3・VRAM 16GB mode on・先読み on・GPU に残す
       ブロック 8・LoRA なし・seed 42・``embed_mp4_metadata=false``・切り抜きなし）。mp4 の sha256 を G-C0 の値
       （または ``--expect-sha``）と比べる。結果は ``gc0\\<label>\\result.json``。
  gc1  VSF つきの連結生成（2 クリップ・512×288・56 フレームずつ・倍率 3・``keep_resident=true``）。各クリップの
       ``metadata.h3.clips[i].vsf``・負のプロンプトの符号化が 1 回だけであること・ワーカーのログの
       ``neg=vsf(scale=3.0)``。続けて ``neg_method="nag"`` が 422 で断られること、``list_models`` の
       ``unsupported_features`` に ``nag`` が残ること。
  gc2  VSF つきの Retake（素材は gc0 か gc1 の出力。``--material`` で替えられる）。
  gc3  持ち越しが無いこと。VSF のジョブの直後（同じサーバー・``keep_resident=true`` の後）に gc0 と同じ要求を送り、
       指紋が基準と同じ・``vsf`` の鍵が無い・ログが ``neg=off``。
  all  gc0 → gc1 → gc2 → gc3 を 1 つのサーバーで続けて回す。

流れ:
  1. ``config.yaml`` を読むだけで、``server.port``・``server.log_dir``・``output.dir``・``upload.dir``・``state_file``
     を ``_rt\\`` に向け、さらに ``vram.block_swap: true``・``vram.block_swap_blocks_on_gpu: 8`` を明示した一時 yaml と、
     MiniMax H3 を選んだ一時 state を ``_rt\\`` に書く（今の ``config.yaml`` の値に門が左右されないように）。
  2. ``.venv\\Scripts\\python.exe main.py --config <一時 yaml>`` で別ポート（既定 18699）に実バックエンドを起こす。
  3. MCP サーバー（``python -m mcp_server``）を子プロセスとして起こす。環境に ``PYTHONUTF8=1``・
     ``LTX_MCP_BASE_URL=http://127.0.0.1:<port>``（``/api/v1`` を含まない形）・``LTX_MCP_CONFIG=<一時 yaml>``・
     ``LTX_MCP_OUTPUT_DIR=_rt\\outputs`` を渡す。
  4. ``load_pipeline(base_model="MiniMaxH3", models={"transformer": "default"}, wait_sec=45)`` を 1 回、
     以後 ``backend_status`` を 10 秒おき。
  5. ``--mode`` の門を回す。
  6. ``--keep-server`` でなければ ``unload_pipeline`` → サーバーをプロセスの木ごと停止。
  7. 門ごとの ``result.json`` と、全体の ``summary.json`` を書く。

終了コード: 0＝回した門がすべて合格、3＝どれかが不合格・失敗（先へは進む）、
1＝準備の段で止まった（サーバーが起きない・道具の引数名が違う・読み込みの失敗など）。
``--dry-run`` は送る引数を表示するだけ、``--selfcheck`` は GPU もサーバーも使わない自己検査。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
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
ARCHIVE = REPO / "Docs" / "Outputs-archive"
RT = HERE / "_rt"
PY = REPO / ".venv" / "Scripts" / "python.exe"
PORT_DEFAULT = 18699
TEMP_CONFIG_NAME = "config.h3vsfchain.yaml"
TEMP_STATE_NAME = "state.h3vsfchain.json"

H3_BASE_MODEL = "MiniMaxH3"
H3_CATEGORIES = ("transformer", "text_encoder", "video_vae", "audio")
REAL_BACKEND_H3 = "minimax-h3"
FL2VA_NAME = "default"
FL2VA_FILE = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
H3_FPS = 24.0

#: 一時 yaml で明示する vram の鍵（G-C0 を回した 2026-10-29 の config.yaml の値。サーバーの記録
#: ``h3-chain-gates-2026-10-29/backend.console.log`` の読み込みの行に ``blocks_on_gpu=8``）。
#: 先読み（block swap prefetch）は yaml の鍵ではなく要求ごとの ``block_swap_prefetch``（既定 true）。
#: ``vram.block_swap`` は表示用だけの鍵（config.yaml.example の注記）だが、G-C0 のときの値に揃えて書く。
VRAM_OVERRIDES = {"block_swap": True, "block_swap_blocks_on_gpu": 8}

#: G-C0（2026-10-29）の記録。正本は記録簿 ``Docs/VERIFICATION_LOG.md`` §184.6。
G_C0_JSON = ARCHIVE / "h3-chain-gates-2026-10-29" / "g_c0.json"
G_C0_MP4_SHA = "4a98be0884ae7129e34f58578a3fe7ca3117ca5a764545cbac404e12a70cbec5"
PROMPTS_C = ARCHIVE / "h3-chain-2026-10-29" / "prompts_c.json"
CLIP_FRAMES_GC0 = 124
N_CLIPS_GC0 = 3
OVERLAP_FRAMES = 22  # H3 の連結の重なり（固定）

#: gc1 の形（第 1 弾の G1 と同じ小さな形を 2 クリップに）。
GC1_WIDTH, GC1_HEIGHT, GC1_FRAMES, GC1_CLIPS = 512, 288, 56, 2
NEGATIVE = "blurry, low quality"
VSF_SCALE = 3.0
VSF_KEYS = ("rows", "scale", "m_neg", "m_text")
NAG_REFUSAL_DETAIL = "VSF 方式だけ"

#: gc2 の既定（素材が gc1 の出力〔90 フレーム・3.75 秒〕でも窓が収まる値。17n+5 かつ 73 以上）。
RETAKE_FRAMES_DEFAULT = 73
RETAKE_START_DEFAULT = 0.5
UPLOAD_MAX_FRAMES = 100000  # 尺を測るためだけの max_frames（元の尺より確実に大きい値）

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
    "upload_video": {"file_path", "max_frames"},
    "submit_chain": {
        "prompt", "clips", "width", "height", "frame_rate", "seed", "attention_backend", "block_swap_prefetch",
        "keep_resident", "embed_mp4_metadata", "nag_enabled", "neg_method", "negative_prompt", "vsf_scale",
        "retake_video_id", "retake_window_start_sec",
    },
    "wait_for_job": {"job_id", "timeout_sec", "poll_interval_sec"},
    "cancel_job": {"job_id"},
    "save_job_video": {"job_id", "dest_dir", "no_clobber", "which"},
    "get_job_video_path": {"job_id"},
}
#: submit_chain に無いと分かっていて、渡さない引数（H3 の既定で動く。metadata.json で確かめる）。
#: tools/list に現れたら記録だけする（配線の作業で足されたかどうかが分かるように）。
CHAIN_ARGS_NOT_SENT = ("num_inference_steps", "shift_video", "shift_audio", "vram16_mode")

#: gc0 で、サーバーが受けた本文（metadata.request）が G-C0 の本文と同じでなければならない欄。
GC0_REQUEST_CRITICAL = (
    "prompt", "negative_prompt", "nag_enabled", "attention_backend", "block_swap_prefetch", "embed_mp4_metadata",
    "vram16_mode", "width", "height", "crop_output", "frame_rate", "num_inference_steps", "shift_video",
    "shift_audio", "seed", "clips", "source_video", "source_audio", "retake", "end_source", "loras",
    "reference_video_id",
)
#: H3 が無視する欄（MCP の submit_chain は常に送るか、既定が API と違う）。違っても記録だけ。
GC0_REQUEST_IGNORED_BY_H3 = ("overlap_frames", "overlap_strength", "chunked_upsample", "stage2_window")


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


def sha_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


_NAG_WORD = re.compile(r"(?<![A-Za-z_])nag(?![A-Za-z_])")


def mentions_nag(text: str) -> bool:
    """エラー文が機能名 ``nag`` を語として含むか（``nag_enabled`` の部分一致は数えない）。"""
    return bool(_NAG_WORD.search(text or ""))


def check(checks: dict, name: str, ok: bool, actual: Any = None, expected: Any = None) -> None:
    checks[name] = {"pass": bool(ok), "actual": actual, "expected": expected}


def all_pass(checks: dict) -> bool:
    return bool(checks) and all(c.get("pass") for c in checks.values())


# ---------------------------------------------------------------------------------------------
# プロンプトの組み立て（g_chain.py の写し。逐語）
# ---------------------------------------------------------------------------------------------

def compose_clip_prompt(shared: str, own: str | None) -> str:
    """api/models.py ``compose_clip_prompt``: both ends trimmed, non-empty parts joined by one space."""
    parts = (shared.strip(), (own or "").strip())
    return " ".join(part for part in parts if part)


def split_prompts(prompts: list[str]) -> tuple[str, list[str]]:
    """prompts_c.json (the full 3 clip prompts) -> (shared prompt, appended part per clip).

    Every later prompt must be the first one plus an appended sentence; the appended part is what
    ``clips[i].prompt`` carries, so the product composes the same strings back.
    """
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
    """(shared, own per clip, the composed prompt per clip) for ``n_clips`` clips.

    Clip 0 is the shared prompt alone. Clips 1.. cycle through the appended parts of the prompts file
    (3 clips = prompts_c.json as is; 13 clips alternate the two appended sentences).
    """
    shared, owns = split_prompts(prompts)
    appended = [o for o in owns[1:] if o] or [""]
    own_per_clip = [""] + [appended[(i - 1) % len(appended)] for i in range(1, n_clips)]
    return shared, own_per_clip, [compose_clip_prompt(shared, o) for o in own_per_clip]


def load_prompts() -> list[str]:
    return json.loads(PROMPTS_C.read_text(encoding="utf-8"))


def chain_clips(prompts: list[str], n_clips: int, frames: int) -> tuple[str, list[dict]]:
    """``submit_chain`` の ``prompt`` と ``clips``（g_chain.py の build_request と同じ形。クリップ 0 は共通の文だけ）。"""
    shared, owns, _ = expected_clip_prompts(prompts, n_clips)
    clips = []
    for own in owns:
        clip: dict = {"num_frames": frames}
        if own:
            clip["prompt"] = own
        clips.append(clip)
    return shared, clips


# ---------------------------------------------------------------------------------------------
# 要求の組み立て（MCP の submit_chain に渡す引数）
# ---------------------------------------------------------------------------------------------

def gc0_args(prompts: list[str]) -> dict:
    """G-C0 と同じ連結生成。ステップ数・shift・VRAM 16GB mode は submit_chain に引数が無いので H3 の既定
    （20・12/3・on）で動く。metadata.json で確かめる。切り抜き・LoRA・ネガティブは送らない。"""
    shared, clips = chain_clips(prompts, N_CLIPS_GC0, CLIP_FRAMES_GC0)
    return {"prompt": shared, "clips": clips, "width": 640, "height": 384, "frame_rate": H3_FPS, "seed": 42,
            "attention_backend": "sdpa", "block_swap_prefetch": True, "keep_resident": False,
            "embed_mp4_metadata": False}


def gc1_args(prompts: list[str], neg_method: str = "vsf") -> dict:
    shared, clips = chain_clips(prompts, GC1_CLIPS, GC1_FRAMES)
    return {"prompt": shared, "clips": clips, "width": GC1_WIDTH, "height": GC1_HEIGHT, "frame_rate": H3_FPS,
            "seed": 42, "attention_backend": "sdpa", "block_swap_prefetch": True, "keep_resident": True,
            "embed_mp4_metadata": False, "nag_enabled": True, "neg_method": neg_method,
            "negative_prompt": NEGATIVE, "vsf_scale": VSF_SCALE}


def gc2_args(prompts: list[str], video_id: str, width: int, height: int, frames: int, start_sec: float) -> dict:
    """VSF つきの Retake。のりしろ（22 固定）と音声の強さ（既定 0.5）は渡さない（h3_retake_gate.py と同じ）。
    ``keep_resident=true`` は、all で gc3 の直前のジョブになるため（gc3 は VSF のジョブの直後を見る）。"""
    return {"prompt": prompts[0], "clips": [{"num_frames": frames}], "width": width, "height": height,
            "frame_rate": H3_FPS, "seed": 42, "attention_backend": "sdpa", "block_swap_prefetch": True,
            "keep_resident": True, "embed_mp4_metadata": False, "retake_video_id": video_id,
            "retake_window_start_sec": start_sec, "nag_enabled": True, "neg_method": "vsf",
            "negative_prompt": NEGATIVE, "vsf_scale": VSF_SCALE}


# ---------------------------------------------------------------------------------------------
# metadata.json とログの読み方（純粋な関数。--selfcheck が確かめる）
# ---------------------------------------------------------------------------------------------

def h3_block(meta: dict) -> dict:
    h3 = meta.get("h3")
    return h3 if isinstance(h3, dict) else {}


def h3_clips(meta: dict) -> list[dict]:
    clips = h3_block(meta).get("clips")
    return [c if isinstance(c, dict) else {} for c in clips] if isinstance(clips, list) else []


def phase_encode(meta: dict, i: int) -> dict:
    """``metadata.h3.phases.clips[i].encode``（無ければ空）。"""
    phases = h3_block(meta).get("phases") or {}
    clips = phases.get("clips") if isinstance(phases, dict) else None
    if not isinstance(clips, list) or i >= len(clips) or not isinstance(clips[i], dict):
        return {}
    enc = clips[i].get("encode")
    return enc if isinstance(enc, dict) else {}


def latents_shas(meta: dict) -> list[dict | None]:
    return [c.get("latents_sha256") for c in h3_clips(meta)]


def chain_log_lines(excerpt: str) -> list[str]:
    """ワーカーのログの差分から、連結生成の 1 行（``generate_chain: ...``）だけを抜く。"""
    return [ln for ln in (excerpt or "").splitlines() if "generate_chain:" in ln]


def vsf_ok(vsf: Any, scale: float = VSF_SCALE) -> bool:
    return isinstance(vsf, dict) and all(k in vsf for k in VSF_KEYS) and vsf.get("scale") == scale


def _norm_clips(clips: Any) -> Any:
    """クリップの欄のうち空（None・空のリスト）のものを落とす。G-C0 の後に増えた ``clips[].references``（空の
    リスト）のような欄で、中身の同じクリップを違うと数えないため。"""
    if not isinstance(clips, list):
        return clips
    return [{k: v for k, v in c.items() if v not in (None, [])} if isinstance(c, dict) else c for c in clips]


def request_diff(actual: dict, archived: dict) -> dict:
    """サーバーが受けた本文（metadata.request）と G-C0 のときの本文の差。``critical`` は合否に使う欄、
    ``ignored_by_h3`` は H3 が無視する欄、``other`` はそれ以外（G-C0 の後に増えた欄など）。
    ``clips`` は空の欄を落としてから比べる（``_norm_clips``）。"""
    out: dict[str, dict] = {"critical": {}, "ignored_by_h3": {}, "other": {}}
    for key in sorted(set(actual) | set(archived)):
        a, b = actual.get(key, "<無い>"), archived.get(key, "<無い>")
        if key == "clips":
            a, b = _norm_clips(a), _norm_clips(b)
        if a == b:
            continue
        bucket = ("critical" if key in GC0_REQUEST_CRITICAL else
                  "ignored_by_h3" if key in GC0_REQUEST_IGNORED_BY_H3 else "other")
        out[bucket][key] = {"now": a, "g_c0": b}
    return out


def load_archived_g_c0() -> dict:
    try:
        d = json.loads(G_C0_JSON.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001 -- 記録が無くても門の主の基準（指紋）は判定できる
        return {}
    return {"request": d.get("metadata_request") or {}, "latents": [c.get("latents_sha256") for c in
                                                                     (d.get("metadata_h3") or {}).get("clips", [])],
            "mp4_sha256": (d.get("mp4") or {}).get("sha256"), "request_body": d.get("request_body") or {}}


def judge_gc0_like(job: dict, meta: dict, expect_sha: str, archived: dict, prompts: list[str]) -> dict:
    """gc0 と gc3 に共通の基準（ジョブ・条件・指紋）。"""
    checks: dict[str, dict] = {}
    req = meta.get("request") or {}
    h3 = h3_block(meta)
    sel = ((meta.get("models") or {}).get("selection") or {}).get("transformer") or {}
    _, _, composed = expected_clip_prompts(prompts, N_CLIPS_GC0)
    check(checks, "job_completed", job.get("status") == "completed", job.get("status"), "completed")
    check(checks, "backend_is_minimax_h3", meta.get("backend") == REAL_BACKEND_H3, meta.get("backend"),
          REAL_BACKEND_H3)
    check(checks, "transformer_is_fl2va_default", sel.get("name") == FL2VA_NAME and sel.get("file") == FL2VA_FILE,
          sel, {"name": FL2VA_NAME, "file": FL2VA_FILE})
    check(checks, "attention_used_sdpa", meta.get("attention_used") == "sdpa", meta.get("attention_used"), "sdpa")
    check(checks, "vram16_mode_used_on", meta.get("vram16_mode_used") == "on", meta.get("vram16_mode_used"), "on")
    check(checks, "block_swap_prefetch_used_on", meta.get("block_swap_prefetch_used") == "on",
          meta.get("block_swap_prefetch_used"), "on")
    check(checks, "steps_20_shift_12_3",
          (h3.get("num_inference_steps"), h3.get("shift_video"), h3.get("shift_audio")) == (20, 12.0, 3.0),
          [h3.get("num_inference_steps"), h3.get("shift_video"), h3.get("shift_audio")], [20, 12.0, 3.0])
    check(checks, "seeds_42_43_44", h3.get("seeds") == [42, 43, 44], h3.get("seeds"), [42, 43, 44])
    check(checks, "three_clips", len(h3_clips(meta)) == N_CLIPS_GC0, len(h3_clips(meta)), N_CLIPS_GC0)
    psha = [c.get("prompt_sha256") for c in h3_clips(meta)]
    check(checks, "clip_prompts_composed_as_g_c0", psha == [sha_text(t) for t in composed], psha,
          [sha_text(t) for t in composed])
    check(checks, "no_lora_no_crop_no_negative",
          req.get("loras") in ([], None) and req.get("crop_output") is None and not req.get("nag_enabled")
          and not req.get("negative_prompt"),
          {k: req.get(k) for k in ("loras", "crop_output", "nag_enabled", "negative_prompt")},
          {"loras": [], "crop_output": None, "nag_enabled": False, "negative_prompt": ""})
    check(checks, "embed_mp4_metadata_false", req.get("embed_mp4_metadata") is False, req.get("embed_mp4_metadata"),
          False)
    check(checks, "mp4_sha256_matches", (job.get("sha256") or "") == expect_sha, job.get("sha256"), expect_sha)
    records: dict[str, Any] = {}
    if archived.get("request"):
        diff = request_diff(req, archived["request"])
        records["request_diff_vs_g_c0"] = diff
        check(checks, "request_critical_fields_equal_g_c0", not diff["critical"], diff["critical"], {})
    lat = latents_shas(meta)
    records["latents_sha256"] = lat
    if archived.get("latents"):
        records["latents_equal_g_c0"] = [a == b for a, b in zip(lat, archived["latents"])] if lat else None
    return {"checks": checks, "records": records}


def judge_gc1(job: dict, meta: dict) -> dict:
    checks: dict[str, dict] = {}
    clips = h3_clips(meta)
    check(checks, "job_completed", job.get("status") == "completed", job.get("status"), "completed")
    check(checks, "two_clips", len(clips) == GC1_CLIPS, len(clips), GC1_CLIPS)
    for i in range(GC1_CLIPS):
        vsf = clips[i].get("vsf") if i < len(clips) else None
        check(checks, f"clips[{i}].vsf_has_keys_and_scale_3", vsf_ok(vsf), vsf,
              {"keys": list(VSF_KEYS), "scale": VSF_SCALE})
    e0, e1 = phase_encode(meta, 0), phase_encode(meta, 1)
    check(checks, "phases.clips[0].encode_has_negative", "te_read_negative" in e0 and "negative_tokens" in e0,
          {k: e0.get(k, "<無い>") for k in ("te_read_negative", "negative_tokens")}, "両方ある")
    check(checks, "phases.clips[1].encode_has_no_negative",
          bool(e1) and "te_read_negative" not in e1 and "negative_tokens" not in e1,
          {"encode_present": bool(e1), "keys": sorted(k for k in e1 if "neg" in k)}, "encode はあり、負の鍵は無い")
    lines = job.get("chain_log_lines") or []
    check(checks, "log_neg_vsf_scale_3", any("neg=vsf(scale=3.0)" in ln for ln in lines), lines,
          "generate_chain: の行に neg=vsf(scale=3.0)")
    return {"checks": checks}


def judge_refusal(rec: dict) -> bool:
    text = rec.get("error_text") or ""
    return ("FEATURE_UNSUPPORTED" in text and mentions_nag(text) and NAG_REFUSAL_DETAIL in text
            and "accepted_unexpectedly" not in rec)


def judge_gc2(job: dict, meta: dict) -> dict:
    checks: dict[str, dict] = {}
    clips = h3_clips(meta)
    vsf = clips[0].get("vsf") if clips else None
    check(checks, "job_completed", job.get("status") == "completed", job.get("status"), "completed")
    check(checks, "is_retake", isinstance(meta.get("retake"), dict) or isinstance((meta.get("request") or {})
                                                                                   .get("retake"), dict),
          {"retake": meta.get("retake") is not None, "request.retake": (meta.get("request") or {}).get("retake")},
          "metadata.retake か request.retake がある")
    check(checks, "clips[0].vsf_has_keys_and_scale_3", vsf_ok(vsf), vsf, {"keys": list(VSF_KEYS), "scale": VSF_SCALE})
    lines = job.get("chain_log_lines") or []
    check(checks, "log_neg_vsf_scale_3", any("neg=vsf(scale=3.0)" in ln for ln in lines), lines,
          "generate_chain: の行に neg=vsf(scale=3.0)")
    return {"checks": checks}


def judge_gc3_extra(job: dict, meta: dict) -> dict:
    checks: dict[str, dict] = {}
    with_vsf = [i for i, c in enumerate(h3_clips(meta)) if "vsf" in c]
    check(checks, "no_vsf_key_in_any_clip", job.get("status") == "completed" and not with_vsf, with_vsf, [])
    check(checks, "no_top_level_h3_vsf", "vsf" not in h3_block(meta) and "vsf" not in meta,
          {"h3.vsf": "vsf" in h3_block(meta), "vsf": "vsf" in meta}, "どちらも無い")
    lines = job.get("chain_log_lines") or []
    check(checks, "log_neg_off", bool(lines) and all("neg=off" in ln for ln in lines), lines,
          "generate_chain: の行に neg=off")
    return {"checks": checks}


# ---------------------------------------------------------------------------------------------
# 一時 yaml とサーバー（h3_vsf_gate.py の写し。一時 yaml に vram の 2 鍵を足した）
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


def build_temp_config(raw: dict, port: int) -> dict:
    """``config.yaml`` を読んだ辞書から一時 yaml の辞書を作る（純粋。--selfcheck が確かめる）。
    5 つの置き場所を ``_rt\\`` に向け、``vram`` の 2 鍵（``VRAM_OVERRIDES``）を明示する。"""
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


def write_temp_config(port: int) -> tuple[Path, Path, dict]:
    import yaml  # アプリの仮想環境の依存（config.py が使う）

    RT.mkdir(parents=True, exist_ok=True)
    for sub in ("logs", "outputs", "uploads"):
        (RT / sub).mkdir(exist_ok=True)

    live = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
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
        self._fh.write(f"\n===== {now_iso()} h3_vsf_chain_gate.py: main.py --config {cfg} =====\n")
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


def blocks_on_gpu_from_server_logs() -> list[str]:
    """サーバーのログに残る読み込みの行（``Loading pipeline (REAL engine_h3 worker) ... blocks_on_gpu=N``）。記録だけ。"""
    out: list[str] = []
    paths = [RT / "server.log"] + (sorted((RT / "logs").glob("*.log")) if (RT / "logs").exists() else [])
    for p in paths:
        if not p.exists() or p.name.startswith("h3_worker"):
            continue
        try:
            for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if "engine_h3 worker" in ln and "blocks_on_gpu=" in ln:
                    out.append(f"{p.name}: {ln.strip()}")
        except OSError:
            continue
    return out[-5:]


# ---------------------------------------------------------------------------------------------
# MCP の呼び出し（h3_vsf_gate.py の写し）
# ---------------------------------------------------------------------------------------------

class ToolCallError(Exception):
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
        except Exception as exc:  # noqa: BLE001
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
        if name == "submit_chain":
            report["submit_chain_args_not_sent_but_present"] = sorted(set(CHAIN_ARGS_NOT_SENT) & props)
    report["ok"] = not report["problems"]
    return report


async def ensure_pipeline(mcp: Mcp, args, run_rec: dict) -> bool:
    """H3（FL2VA＝登録名 default）を載せる。load_pipeline は 1 回だけ。以後 backend_status を 10 秒おき。"""
    load: dict[str, Any] = {"requested": {"base_model": H3_BASE_MODEL, "models": {"transformer": FL2VA_NAME},
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
    load["blocks_on_gpu_log_lines"] = blocks_on_gpu_from_server_logs()
    log(f"H3 が載った（{load['seconds']} 秒）")
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

async def run_chain_job(mcp: Mcp, args, label: str, job_dir: Path, tool_args: dict, note: str) -> tuple[dict, dict]:
    """submit_chain → 待つ → save_job_video → metadata.json を写す → sha256・ワーカーのログ。(記録, metadata)。"""
    job_dir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"label": label, "dir": str(job_dir), "note": note, "tool": "submit_chain",
                           "tool_args": tool_args, "started_at": now_iso(), "status": None, "error": None}
    write_json(job_dir / "request.json", {"tool": "submit_chain", "arguments": tool_args, "note": note,
                                          "remark": "MCP の道具に渡した引数。サーバーが受けた本文は metadata.json の request"})
    log(f"=== {label}: {note} ===")
    offset = log_offset()
    t0 = time.monotonic()
    meta: dict = {}
    try:
        sub = await mcp.call("submit_chain", tool_args)
    except ToolCallError as exc:
        rec.update(status="submit_error", error=exc.text, finished_at=now_iso())
        log(f"submit_chain が拒否: {exc.text}")
        return rec, meta
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
    rec["chain_log_lines"] = chain_log_lines(excerpt)
    rec["vsf_log_lines"] = [ln for ln in excerpt.splitlines() if "H3 VSF:" in ln]
    if waited.get("gave_up"):
        rec.update(status="wait_gave_up", error=waited["gave_up"], finished_at=now_iso())
        return rec, meta
    if rec["status"] != "completed":
        rec["error"] = final.get("error")
        rec["finished_at"] = now_iso()
        log(f"ジョブが {rec['status']}: {final.get('error')}")
        return rec, meta
    try:
        saved = await mcp.call("save_job_video", {"job_id": job_id, "dest_dir": str(job_dir),
                                                  "no_clobber": True, "which": "output"})
        rec["saved"] = saved
        rec["mp4"] = str(saved["saved_path"])
        rec["sha256"] = sha256(Path(saved["saved_path"]))
    except ToolCallError as exc:
        rec.update(error=f"save_job_video: {exc.text}", finished_at=now_iso())
        return rec, meta
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
    rec["metadata_summary"] = {
        "backend": meta.get("backend"),
        "attention_used": meta.get("attention_used"),
        "vram16_mode_used": meta.get("vram16_mode_used"),
        "block_swap_prefetch_used": meta.get("block_swap_prefetch_used"),
        "keep_resident_used": meta.get("keep_resident_used"),
        "generation_time_seconds": meta.get("generation_time_seconds"),
        "peak_vram_reserved_mb": meta.get("peak_vram_reserved_mb"),
        "peak_vram_mb": (meta.get("vram_optimization") or {}).get("peak_vram_mb"),
        "seed_used": meta.get("seed_used"),
        "num_inference_steps": h3_block(meta).get("num_inference_steps"),
        "shift_video": h3_block(meta).get("shift_video"),
        "shift_audio": h3_block(meta).get("shift_audio"),
        "clips_vsf": [c.get("vsf") for c in h3_clips(meta)],
    }
    rec["finished_at"] = now_iso()
    log(f"完了: sha256={(rec.get('sha256') or '')[:16]}…")
    return rec, meta


async def expect_refusal(mcp: Mcp, tool: str, tool_args: dict) -> dict:
    """道具が ToolCallError で断ることを確かめる。万一受理されたらジョブを取り消して待つ。"""
    rec: dict[str, Any] = {"tool": tool, "tool_args": tool_args}
    try:
        sub = await mcp.call(tool, tool_args)
    except ToolCallError as exc:
        rec["error_text"] = exc.text
        rec["has_feature_unsupported"] = "FEATURE_UNSUPPORTED" in exc.text
        rec["mentions_nag"] = mentions_nag(exc.text)
        rec["has_detail"] = NAG_REFUSAL_DETAIL in exc.text
        rec["pass"] = judge_refusal(rec)
        return rec
    rec["accepted_unexpectedly"] = sub
    rec["pass"] = False
    job_id = sub.get("job_id")
    if job_id:
        try:
            rec["cancel"] = await mcp.call("cancel_job", {"job_id": job_id})
        except ToolCallError as exc:
            rec["cancel"] = {"error": exc.text}
        waited = await wait_job(mcp, job_id, 3600)
        rec["cancel_wait_final_status"] = (waited.get("final") or {}).get("status")
    return rec


# ---------------------------------------------------------------------------------------------
# 門
# ---------------------------------------------------------------------------------------------

def expected_sha(args) -> str:
    return (args.expect_sha or G_C0_MP4_SHA).lower()


def job_result_fields(job: dict) -> dict:
    ms = job.get("metadata_summary") or {}
    return {"job_id": job.get("job_id"), "status": job.get("status"), "error": job.get("error"),
            "sha256": job.get("sha256"), "mp4": job.get("mp4"), "seconds_wall": job.get("seconds_wall"),
            "generation_time_seconds": ms.get("generation_time_seconds"),
            "peak_vram_reserved_mb": ms.get("peak_vram_reserved_mb"), "peak_vram_mb": ms.get("peak_vram_mb"),
            "chain_log_lines": job.get("chain_log_lines")}


def finish_gate(name: str, path: Path, res: dict, summary: dict) -> bool:
    res["pass"] = all_pass(res.get("checks") or {})
    res["finished_at"] = now_iso()
    write_json(path, res)
    failed = [k for k, v in (res.get("checks") or {}).items() if not v.get("pass")]
    summary["gates"][name] = {"pass": res["pass"], "result": str(path), "failed": failed}
    log(f"{name}: {'合格' if res['pass'] else '不合格'}" + (f"（落ちた基準: {failed}）" if failed else ""))
    return res["pass"]


async def gate_gc0(mcp: Mcp, args, summary: dict, ctx: dict) -> bool:
    log(f"##### GC0: ネガティブなしの連結生成の回帰（label={args.label}） #####")
    prompts = load_prompts()
    archived = load_archived_g_c0()
    gdir = HERE / "gc0" / args.label
    job, meta = await run_chain_job(mcp, args, f"gc0_{args.label}", gdir, gc0_args(prompts),
                                    "GC0: G-C0 と同じ連結生成（ネガティブなし）")
    summary["jobs"][f"gc0_{args.label}"] = job
    judged = judge_gc0_like(job, meta, expected_sha(args), archived, prompts)
    res = {"gate": "gc0", "label": args.label, "started_at": job.get("started_at"),
           "expected_sha256": expected_sha(args), "expected_sha256_source": "--expect-sha" if args.expect_sha
           else "G-C0（h3-chain-gates-2026-10-29/g_c0.json）",
           "sha256_match": (job.get("sha256") or "") == expected_sha(args), **job_result_fields(job),
           "vsf_in_clips": [i for i, c in enumerate(h3_clips(meta)) if "vsf" in c],
           "blocks_on_gpu_log_lines": blocks_on_gpu_from_server_logs(), **judged}
    ctx["gc0_mp4"] = job.get("mp4")
    ctx["gc0_sha"] = job.get("sha256")
    return finish_gate("gc0", gdir / "result.json", res, summary)


async def gate_gc1(mcp: Mcp, args, summary: dict, ctx: dict) -> bool:
    log("##### GC1: VSF つきの連結生成 #####")
    prompts = load_prompts()
    gdir = HERE / "gc1"
    job, meta = await run_chain_job(mcp, args, "gc1_vsf", gdir / "vsf", gc1_args(prompts, "vsf"),
                                    "GC1: 2 クリップ・512×288・56 フレームずつ・VSF 3.0・keep_resident")
    summary["jobs"]["gc1_vsf"] = job
    judged = judge_gc1(job, meta)
    checks = judged["checks"]
    refusal = await expect_refusal(mcp, "submit_chain", gc1_args(prompts, "nag"))
    refusal["expected"] = f"ToolCallError・文に FEATURE_UNSUPPORTED と nag と「{NAG_REFUSAL_DETAIL}」"
    check(checks, "chain_neg_method_nag_refused", refusal["pass"], refusal.get("error_text")
          or refusal.get("accepted_unexpectedly"), refusal["expected"])
    try:
        models = await mcp.call("list_models")
        h3 = next((b for b in models.get("base_models", []) if b.get("id") == H3_BASE_MODEL), None) or {}
        feats = h3.get("unsupported_features")
        check(checks, "list_models_still_has_nag", isinstance(feats, list) and "nag" in feats, feats,
              "MiniMaxH3 の unsupported_features に nag")
        ctx["transformer_role"] = h3.get("transformer_role")
    except ToolCallError as exc:
        check(checks, "list_models_still_has_nag", False, exc.text, "MiniMaxH3 の unsupported_features に nag")
    res = {"gate": "gc1", "label": args.label, "started_at": job.get("started_at"), **job_result_fields(job),
           "clips_vsf": [c.get("vsf") for c in h3_clips(meta)],
           "encode_negative": [{k: phase_encode(meta, i).get(k) for k in ("te_read_negative", "negative_tokens")}
                               for i in range(GC1_CLIPS)],
           "vsf_log_lines": job.get("vsf_log_lines"), "refusal": refusal, "checks": checks}
    ctx["gc1_mp4"] = job.get("mp4")
    ctx["last_vsf_job"] = "gc1_vsf"
    return finish_gate("gc1", gdir / "result.json", res, summary)


def pick_material(args, ctx: dict) -> tuple[Path | None, str]:
    """gc2 の素材。--material → この回の gc0 → 保管庫の gc0\\*\\ の最新 → この回の gc1 → gc1\\vsf\\ の最新。"""
    if args.material:
        return Path(args.material).resolve(), "--material"
    if ctx.get("gc0_mp4"):
        return Path(ctx["gc0_mp4"]), "gc0（この回）"
    found = sorted((HERE / "gc0").glob("*/*.mp4"), key=lambda p: p.stat().st_mtime) if (HERE / "gc0").exists() else []
    if found:
        return found[-1], "gc0（前の回）"
    if ctx.get("gc1_mp4"):
        return Path(ctx["gc1_mp4"]), "gc1（この回）"
    found = sorted((HERE / "gc1" / "vsf").glob("*.mp4"), key=lambda p: p.stat().st_mtime) \
        if (HERE / "gc1" / "vsf").exists() else []
    if found:
        return found[-1], "gc1（前の回）"
    return None, "なし"


def retake_canvas(args, source: str) -> tuple[int, int]:
    if args.retake_width and args.retake_height:
        return args.retake_width, args.retake_height
    return (GC1_WIDTH, GC1_HEIGHT) if source.startswith("gc1") else (640, 384)


async def gate_gc2(mcp: Mcp, args, summary: dict, ctx: dict) -> bool:
    log("##### GC2: VSF つきの Retake #####")
    gdir = HERE / "gc2"
    res: dict[str, Any] = {"gate": "gc2", "label": args.label, "started_at": now_iso(), "checks": {}}
    material, source = pick_material(args, ctx)
    res["material"] = {"path": str(material) if material else None, "source": source,
                       "sha256": sha256(material) if material else None}
    if material is None or not material.is_file():
        check(res["checks"], "material_found", False, str(material), "gc0 か gc1 の出力、または --material")
        return finish_gate("gc2", gdir / "result.json", res, summary)
    try:
        up = await mcp.call("upload_video", {"file_path": str(material), "max_frames": UPLOAD_MAX_FRAMES})
    except ToolCallError as exc:
        check(res["checks"], "upload_ok", False, exc.text, "video_id が返る")
        return finish_gate("gc2", gdir / "result.json", res, summary)
    res["upload"] = up
    fc, fps = up.get("frame_count"), up.get("fps")
    need = args.retake_start_sec + args.retake_frames / H3_FPS
    res["geometry"] = {"window_end_sec": need, "material_seconds": (fc / fps) if fc and fps else None}
    if fc and fps and need > fc / fps + 1e-6:
        log(f"注意: 窓の終わり {need:.3f} 秒が素材の尺 {fc / fps:.3f} 秒を超える（サーバーが断るはず）")
    w, h = retake_canvas(args, source)
    tool_args = gc2_args(load_prompts(), up["video_id"], w, h, args.retake_frames, args.retake_start_sec)
    job, meta = await run_chain_job(mcp, args, "gc2_retake_vsf", gdir / "retake", tool_args,
                                    f"GC2: Retake（{args.retake_frames} フレーム・{args.retake_start_sec} 秒から）・VSF 3.0")
    summary["jobs"]["gc2_retake_vsf"] = job
    judged = judge_gc2(job, meta)
    res.update({**job_result_fields(job), "clips_vsf": [c.get("vsf") for c in h3_clips(meta)],
                "metadata_retake": meta.get("retake"), "vsf_log_lines": job.get("vsf_log_lines")})
    res["checks"].update(judged["checks"])
    ctx["last_vsf_job"] = "gc2_retake_vsf"
    return finish_gate("gc2", gdir / "result.json", res, summary)


async def gate_gc3(mcp: Mcp, args, summary: dict, ctx: dict) -> bool:
    log("##### GC3: VSF のジョブの直後に持ち越しが無いこと #####")
    prompts = load_prompts()
    archived = load_archived_g_c0()
    gdir = HERE / "gc3"
    job, meta = await run_chain_job(mcp, args, "gc3_off", gdir / "off", gc0_args(prompts),
                                    "GC3: VSF のジョブの直後に GC0 と同じ連結生成（ネガティブなし）")
    summary["jobs"]["gc3_off"] = job
    judged = judge_gc0_like(job, meta, expected_sha(args), archived, prompts)
    judged["checks"].update(judge_gc3_extra(job, meta)["checks"])
    res = {"gate": "gc3", "label": args.label, "started_at": job.get("started_at"),
           "preceding_vsf_job_this_run": ctx.get("last_vsf_job"),
           "expected_sha256": expected_sha(args), "sha256_match": (job.get("sha256") or "") == expected_sha(args),
           "sha256_equals_gc0_this_run": (job.get("sha256") == ctx["gc0_sha"]) if ctx.get("gc0_sha") else None,
           **job_result_fields(job), **judged}
    return finish_gate("gc3", gdir / "result.json", res, summary)


GATES = {"gc0": gate_gc0, "gc1": gate_gc1, "gc2": gate_gc2, "gc3": gate_gc3}


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
    return {"script": "h3_vsf_chain_gate.py", "jobs": {}, "gates": {}, "runs": []}


def flush(summary: dict) -> None:
    write_json(HERE / "summary.json", summary)


def modes_of(args) -> list[str]:
    return ["gc0", "gc1", "gc2", "gc3"] if args.mode == "all" else [args.mode]


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
    ctx: dict[str, Any] = {}
    try:
        with open(stderr_log, "a", encoding="utf-8") as errlog:
            errlog.write(f"\n===== {now_iso()} h3_vsf_chain_gate.py --mode {args.mode} --label {args.label} =====\n")
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
                    if schema.get("submit_chain_args_not_sent_but_present"):
                        log(f"記録: submit_chain に {schema['submit_chain_args_not_sent_but_present']} がある"
                            "（この台本は渡さない。H3 の既定で動くことを metadata.json で確かめる）")
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
                    for m in modes_of(args):
                        try:
                            ok = await GATES[m](mcp, args, summary, ctx)
                        except Exception as exc:  # noqa: BLE001 -- 1 つの門の例外で残りを止めない
                            summary["gates"][m] = {"pass": False, "error": f"{type(exc).__name__}: {exc}",
                                                   "traceback": traceback.format_exc()}
                            ok = False
                        if not ok:
                            code = 3
                        flush(summary)
                        gave_up = [j for j in summary["jobs"].values()
                                   if isinstance(j, dict) and j.get("status") == "wait_gave_up"
                                   and j.get("started_at", "") >= run_rec["started_at"]]
                        if gave_up:
                            log("待ちきれなかったジョブがバックエンドで走っているため、残りの門は回さない（JOB_BUSY になる）")
                            break
                    if not args.keep_server:
                        try:
                            run_rec["unload"] = await mcp.call("unload_pipeline")
                        except ToolCallError as exc:
                            run_rec["unload"] = {"error": exc.text}
    finally:
        write_json(calls_path, calls)
    return code


def dry_run(args) -> int:
    prompts = load_prompts()
    plan: dict[str, Any] = {"port": args.port, "label": args.label, "temp_config": str(RT / TEMP_CONFIG_NAME),
                            "temp_config_overrides": {"server.port": args.port, "server.log_dir": str(RT / "logs"),
                                                      "output.dir": str(RT / "outputs"),
                                                      "upload.dir": str(RT / "uploads"),
                                                      "state_file": str(RT / TEMP_STATE_NAME),
                                                      **{f"vram.{k}": v for k, v in VRAM_OVERRIDES.items()}},
                            "server_cmd": [str(PY), "-u", str(REPO / "main.py"), "--config",
                                           str(RT / TEMP_CONFIG_NAME)],
                            "mcp_env": {"PYTHONUTF8": "1", "LTX_MCP_BASE_URL": base_url(args.port),
                                        "LTX_MCP_CONFIG": str(RT / TEMP_CONFIG_NAME),
                                        "LTX_MCP_OUTPUT_DIR": str(RT / "outputs")},
                            "load_pipeline": {"base_model": H3_BASE_MODEL, "models": {"transformer": FL2VA_NAME},
                                              "wait_sec": 45},
                            "expected_sha256": expected_sha(args)}
    modes = modes_of(args)
    if "gc0" in modes:
        plan["gc0"] = {"submit_chain": gc0_args(prompts), "result": str(HERE / "gc0" / args.label / "result.json")}
    if "gc1" in modes:
        plan["gc1"] = {"submit_chain_vsf": gc1_args(prompts, "vsf"), "submit_chain_nag_expect_422": gc1_args(prompts, "nag"),
                       "then": "list_models"}
    if "gc2" in modes:
        w, h = (args.retake_width or 640, args.retake_height or 384)
        plan["gc2"] = {"material": args.material or "gc0（この回か前の回）→ gc1 の順に探す",
                       "upload_video": {"file_path": "<素材>", "max_frames": UPLOAD_MAX_FRAMES},
                       "submit_chain": gc2_args(prompts, "<upload_video の video_id>", w, h, args.retake_frames,
                                                args.retake_start_sec),
                       "canvas_note": "素材が gc1 の出力なら 512×288（--retake-width/--retake-height で替えられる）"}
    if "gc3" in modes:
        plan["gc3"] = {"submit_chain": gc0_args(prompts), "expected_sha256": expected_sha(args)}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------------------------
# 自己検査（GPU もサーバーも使わない）
# ---------------------------------------------------------------------------------------------

def selfcheck() -> int:
    ok_all = True

    def line(ok: bool, text: str) -> None:
        nonlocal ok_all
        ok_all = ok_all and bool(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {text}")

    prompts = load_prompts()
    shared, owns, composed = expected_clip_prompts(prompts, N_CLIPS_GC0)
    line(composed == prompts, "3 クリップの合成が prompts_c.json と文字どおり同じ")
    line(owns[0] == "" and shared == prompts[0], "クリップ 0 は共通の文だけ（clips[0] に prompt を付けない）")

    archived = load_archived_g_c0()
    body = archived.get("request_body") or {}
    a = gc0_args(prompts)
    # 同じであるべき欄（MCP の引数名 = HTTP の本文の名前）
    for k in ("prompt", "width", "height", "seed", "attention_backend", "clips", "embed_mp4_metadata"):
        line(a.get(k) == body.get(k), f"gc0 の {k} が G-C0 の本文と同じ（{json.dumps(a.get(k), ensure_ascii=False)[:80]}）")
    # G-C0 の本文にあって submit_chain に引数が無い欄（H3 の既定と同じ値であること）
    for k, default in (("num_inference_steps", 20), ("shift_video", 12.0), ("shift_audio", 3.0)):
        line(body.get(k) == default and k not in a, f"G-C0 の {k}={body.get(k)} は H3 の既定と同じで、gc0 は送らない")
    line("vram16_mode" not in body and "vram16_mode" not in a, "vram16_mode は G-C0 も gc0 も送らない（H3 の既定 on）")
    line(archived.get("request", {}).get("vram16_mode") is True and archived["request"].get("block_swap_prefetch") is True,
         "G-C0 のサーバー側の本文は vram16_mode=true・block_swap_prefetch=true")
    line(a.get("block_swap_prefetch") is True and a.get("keep_resident") is False and "loras" not in a
         and "crop_width" not in a and "nag_enabled" not in a, "gc0 は先読み on・常駐 off・LoRA なし・切り抜きなし・ネガティブなし")
    line(archived.get("mp4_sha256") == G_C0_MP4_SHA, "g_c0.json の mp4 sha256 が G_C0_MP4_SHA と同じ")

    # 一時 yaml の組み立て
    t = build_temp_config({"server": {"port": 18620}, "vram": {"block_swap": False, "block_swap_blocks_on_gpu": 3}}, 18699)
    line(t["vram"]["block_swap"] is True and t["vram"]["block_swap_blocks_on_gpu"] == 8 and t["server"]["port"] == 18699
         and t["state_file"].endswith(TEMP_STATE_NAME), "一時 yaml は vram の 2 鍵を上書きし、ポートと state を _rt へ向ける")
    t2 = build_temp_config({"server": {}}, 1)
    line(t2["vram"] == VRAM_OVERRIDES, "config.yaml に vram が無くても 2 鍵を書く")

    # 本文の差（G-C0 の本文そのものなら差なし。H3 が無視する欄だけ違えば critical は空）
    req = copy.deepcopy(archived.get("request") or {})
    line(request_diff(req, archived.get("request") or {}) == {"critical": {}, "ignored_by_h3": {}, "other": {}},
         "G-C0 の本文どうしの差は無い")
    req2 = dict(req, chunked_upsample=True, overlap_frames=3, new_field=1)
    d2 = request_diff(req2, archived.get("request") or {})
    line(not d2["critical"] and "chunked_upsample" in d2["ignored_by_h3"] and "new_field" in d2["other"],
         "MCP が常に送る chunked_upsample=True は無視される欄に入り、合否に使わない")
    req4 = copy.deepcopy(req)
    for c in req4.get("clips") or []:
        c["references"] = []
    line(not request_diff(req4, archived.get("request") or {})["critical"],
         "後から増えた clips[].references（空）は差に数えない")
    req5 = copy.deepcopy(req)
    req5["clips"][1]["prompt"] = "other"
    line("clips" in request_diff(req5, archived.get("request") or {})["critical"], "クリップのプロンプトの違いは合否に使う")
    req3 = dict(req, vram16_mode=False)
    line("vram16_mode" in request_diff(req3, archived.get("request") or {})["critical"], "vram16_mode の違いは合否に使う欄")

    # 合成の metadata で判定を確かめる
    gc0_meta = {"backend": REAL_BACKEND_H3, "attention_used": "sdpa", "vram16_mode_used": "on",
                "block_swap_prefetch_used": "on",
                "models": {"selection": {"transformer": {"name": FL2VA_NAME, "file": FL2VA_FILE}}},
                "request": copy.deepcopy(archived.get("request") or {}),
                "h3": {"num_inference_steps": 20, "shift_video": 12.0, "shift_audio": 3.0, "seeds": [42, 43, 44],
                       "clips": [{"prompt_sha256": sha_text(t), "latents_sha256": lat}
                                 for t, lat in zip(composed, archived.get("latents") or [None] * 3)]}}
    good_job = {"status": "completed", "sha256": G_C0_MP4_SHA}
    j = judge_gc0_like(good_job, gc0_meta, G_C0_MP4_SHA, archived, prompts)
    line(all_pass(j["checks"]), f"gc0 の判定: 正しい metadata で全基準が合格（{len(j['checks'])} 基準）")
    line(j["records"].get("latents_equal_g_c0") == [True, True, True], "gc0 の記録: 潜在の sha256 を G-C0 と比べる")
    j_bad = judge_gc0_like(dict(good_job, sha256="0" * 64), gc0_meta, G_C0_MP4_SHA, archived, prompts)
    line(not j_bad["checks"]["mp4_sha256_matches"]["pass"], "gc0 の判定: 指紋が違えば不合格")

    vsf = {"rows": [41, 46], "scale": 3.0, "m_neg": 0.004, "m_text": 0.01}
    gc1_meta = {"h3": {"clips": [{"vsf": vsf}, {"vsf": vsf}],
                       "phases": {"clips": [{"encode": {"te_read_negative": {"read_gib": 1}, "negative_tokens": 6}},
                                            {"encode": {"te_read": {}}}]}}}
    gc1_job = {"status": "completed", "chain_log_lines": [
        "[engine_h3.worker] generate_chain: seed=42 512x288 clips=[56, 56] ... neg=vsf(scale=3.0) -> x.mp4"]}
    line(all_pass(judge_gc1(gc1_job, gc1_meta)["checks"]), "gc1 の判定: 正しい metadata で全基準が合格")
    bad1 = copy.deepcopy(gc1_meta)
    bad1["h3"]["phases"]["clips"][1]["encode"]["negative_tokens"] = 6
    line(not judge_gc1(gc1_job, bad1)["checks"]["phases.clips[1].encode_has_no_negative"]["pass"],
         "gc1 の判定: クリップ 1 でも負のプロンプトを符号化していたら不合格")
    bad2 = copy.deepcopy(gc1_meta)
    bad2["h3"]["clips"][1]["vsf"] = dict(vsf, scale=1.5)
    line(not judge_gc1(gc1_job, bad2)["checks"]["clips[1].vsf_has_keys_and_scale_3"]["pass"], "gc1 の判定: 倍率が違えば不合格")
    line(not judge_gc1(dict(gc1_job, chain_log_lines=["generate_chain: ... -> x.mp4"]), gc1_meta)["checks"]
         ["log_neg_vsf_scale_3"]["pass"], "gc1 の判定: 連結の行に neg=vsf(scale=3.0) が無ければ不合格")
    line(chain_log_lines("a\n[w] generate: x neg=vsf\n[w] generate_chain: y neg=off\n") == ["[w] generate_chain: y neg=off"],
         "ログの抜き出しは generate_chain: の行だけ（単発の generate: の行を数えない）")

    refusal_text = ("422 FEATURE_UNSUPPORTED: nag — MiniMax H3 のネガティブプロンプトは VSF 方式だけです"
                    "（neg_method を \"vsf\" にしてください）。")
    line(judge_refusal({"error_text": refusal_text}), "拒否の判定: FEATURE_UNSUPPORTED・nag・「VSF 方式だけ」で合格")
    line(not judge_refusal({"error_text": "422 FEATURE_UNSUPPORTED: nag_enabled"}),
         "拒否の判定: 機能名が nag_enabled だけ・説明なしは不合格")

    gc2_meta = {"retake": {"audio_strength": 0.5}, "h3": {"clips": [{"vsf": vsf}]}}
    gc2_job = {"status": "completed", "chain_log_lines": ["generate_chain: ... retake=('a.mp4', 0.5) neg=vsf(scale=3.0) -> x"]}
    line(all_pass(judge_gc2(gc2_job, gc2_meta)["checks"]), "gc2 の判定: 正しい metadata で全基準が合格")
    g2a = gc2_args(prompts, "vid", 640, 384, RETAKE_FRAMES_DEFAULT, RETAKE_START_DEFAULT)
    line(len(g2a["clips"]) == 1 and (g2a["clips"][0]["num_frames"] - 5) % 17 == 0 and g2a["clips"][0]["num_frames"] >= 73
         and "retake_head_px" not in g2a and "retake_tail_px" not in g2a and "retake_audio_strength" not in g2a,
         "gc2 の引数: クリップ 1 本・窓は 17n+5 かつ 73 以上・のりしろと音声の強さは渡さない")
    gc1_len = GC1_CLIPS * GC1_FRAMES - (GC1_CLIPS - 1) * OVERLAP_FRAMES
    line(RETAKE_START_DEFAULT + RETAKE_FRAMES_DEFAULT / H3_FPS <= gc1_len / H3_FPS,
         f"gc2 の既定の窓は gc1 の出力（{gc1_len} フレーム）にも収まる")

    off_meta = copy.deepcopy(gc0_meta)
    off_job = dict(good_job, chain_log_lines=["generate_chain: ... neg=off -> x"])
    line(all_pass(judge_gc3_extra(off_job, off_meta)["checks"]), "gc3 の判定: vsf の鍵が無く neg=off なら合格")
    on_meta = copy.deepcopy(gc0_meta)
    on_meta["h3"]["clips"][2]["vsf"] = vsf
    line(not judge_gc3_extra(off_job, on_meta)["checks"]["no_vsf_key_in_any_clip"]["pass"],
         "gc3 の判定: どれかのクリップに vsf があれば不合格")
    line(not judge_gc3_extra(dict(off_job, chain_log_lines=[]), off_meta)["checks"]["log_neg_off"]["pass"],
         "gc3 の判定: 連結の行が無ければ不合格")

    print("SELFCHECK PASS" if ok_all else "SELFCHECK FAIL")
    return 0 if ok_all else 3


# ---------------------------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        description="MiniMax H3 の VSF を連結生成へつないだことを確かめる門 GC0〜GC3（MCP 経由・別ポートの実バックエンド）")
    ap.add_argument("--mode", choices=("gc0", "gc1", "gc2", "gc3", "all"),
                    help="回す門（all＝gc0 → gc1 → gc2 → gc3 を 1 つのサーバーで）。--selfcheck のときは不要")
    ap.add_argument("--label", default="run", help="結果の JSON に残す名前。gc0 は gc0\\<label>\\ に書く（既定 run）")
    ap.add_argument("--expect-sha", default=None,
                    help=f"gc0／gc3 で比べる mp4 の sha256（既定は G-C0 の {G_C0_MP4_SHA[:16]}…）")
    ap.add_argument("--material", default=None, help="gc2 の素材の mp4（既定は gc0 の出力、無ければ gc1 の出力）")
    ap.add_argument("--retake-frames", type=int, default=RETAKE_FRAMES_DEFAULT,
                    help=f"gc2 の窓のフレーム数（17n+5・73〜481。既定 {RETAKE_FRAMES_DEFAULT}）")
    ap.add_argument("--retake-start-sec", type=float, default=RETAKE_START_DEFAULT,
                    help=f"gc2 の窓の開始秒（既定 {RETAKE_START_DEFAULT}）")
    ap.add_argument("--retake-width", type=int, default=None, help="gc2 の幅（既定は素材に合わせて 640 か 512）")
    ap.add_argument("--retake-height", type=int, default=None, help="gc2 の高さ（既定は素材に合わせて 384 か 288）")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT, help=f"一時サーバーのポート（既定 {PORT_DEFAULT}）")
    ap.add_argument("--keep-server", action="store_true",
                    help="終わりに unload もサーバーの停止もしない（次の回で同じポートのサーバーを使い回す）")
    ap.add_argument("--load-timeout-sec", type=float, default=1200.0)
    ap.add_argument("--job-timeout-sec", type=float, default=3600.0)
    ap.add_argument("--dry-run", action="store_true", help="送る引数を表示して終わる（サーバーも MCP も起こさない）")
    ap.add_argument("--selfcheck", action="store_true", help="GPU もサーバーも使わない自己検査をして終わる")
    args = ap.parse_args(argv)
    if not args.selfcheck and not args.mode:
        ap.error("--mode が要る（--selfcheck のときを除く）")
    if args.expect_sha is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", args.expect_sha):
        ap.error("--expect-sha は 16 進 64 文字")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.label):
        ap.error("--label は英数字と . _ - だけ（フォルダの名前になる）")
    if (args.retake_width is None) != (args.retake_height is None):
        ap.error("--retake-width と --retake-height は両方指定するか両方省略する")
    return args


def print_compact(summary: dict, modes: list[str]) -> None:
    print("---- まとめ ----")
    for m in modes:
        g = summary["gates"].get(m) or {}
        mark = "合格" if g.get("pass") else "不合格"
        extra = f" 落ちた基準={g.get('failed')}" if g.get("failed") else ""
        if g.get("error"):
            extra += f" 例外={g['error']}"
        print(f"{m}: {mark}{extra}  ({g.get('result', '-')})")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args(argv)
    if args.selfcheck:
        return selfcheck()
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
        print_compact(summary, modes_of(args))
        log(f"記録: {HERE / 'summary.json'}（終了コード {code}）")
    return code


if __name__ == "__main__":
    sys.exit(main())
