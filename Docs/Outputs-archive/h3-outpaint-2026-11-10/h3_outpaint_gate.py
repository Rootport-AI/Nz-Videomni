r"""MiniMax H3 Outpainting (台帳 §1-125) の GPU の門 G0〜G4: エンジンの venv で H3Pipeline を直接呼ぶ台本。

計画: ``rosy-jingling-stonebraker.md`` §3「弾 1」・§7「GPU の門」。製品コードは変えない（読むだけ）。
説明書は同じフォルダの README.md。数値の正本は記録簿 ``Docs/VERIFICATION_LOG.md``。

リポジトリの根から（cmd）:

    set PYTHONPATH=%CD%
    set PYTHONUTF8=1
    .venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --prepare
    .venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --dry-run
    .venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --gate G0
    .venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --gate G1,G2,G3,G4

``--prepare`` と ``--dry-run`` は GPU を隠して（CUDA_VISIBLE_DEVICES=-1）CPU だけで動く。

門:
  G0  DiT なし。VAE の往復だけ（単独符号化＝外接矩形／灰色のキャンバスごと符号化 2 色／元の単独の往復）。
  G1  1280x720 -> 1920x1088（pad L320 R320 T184 B184）・s=0.5・r=1。奇数 pad（L13 R19 T8 B8 -> 1312x736）も 1 本。
  G2  主条件で r=0・r=3（r=1 は G1 と共用）。
  G3  主条件で注入係数 0.999 の対照（engine_h3.outpaint_h3.OUTPAINT_INJECT_CLEAN を差し替える）。
  G4  主条件で s=1.0（s=0.5 は G1 と共用）。
  E1  実験: 切り出しの上下 2 行を落とした 1280x716 の素材で、pad L320 R320 T186 B186（1920x1088）・r=1 と r=0。
      --gate E1 で明示したときだけ回す（既定の --gate は G0〜G4）。
  E2  汎用の実験: --material の動画を 24fps・ちょうど --frames 枚・偶数寸法に切り出し（--scale で縮小と中央の切り抜き）、
      --pads・--canvas・--feathers（r の列）で 1 本ずつ回す。本の名前は <--label>_r<r>。既定の注意の実装は sage。
記録と判定は数値を取るだけで止めない。終了コード: 0＝全部完走、1＝失敗した本か門がある、2＝引数・素材の誤り。
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

# torch を読む前に（製品のワーカーと同じ）。--prepare／--dry-run は GPU を隠す。
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
CPU_ONLY = "--prepare" in sys.argv or "--dry-run" in sys.argv
if CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from engine_h3 import outpaint_h3 as OH  # noqa: E402
from engine_h3 import pipeline_h3 as P  # noqa: E402
from engine_h3.constants import AUDIO_SAMPLE_RATE, FRAME_RATE, latent_frames  # noqa: E402
from engine_h3.outpaint_geometry import (  # noqa: E402
    LATENT_PX, TOKEN_PX, aligned_frames, compute_outpaint_geometry, feather_weights,
)

# --------------------------------------------------------------------------- #
# 固定の条件
# --------------------------------------------------------------------------- #

OUT_ROOT = ROOT / "outputs" / "h3-outpaint-gates-2026-11-10"
MATERIAL_SRC = ROOT / "outputs" / "h3-retake-gates-2026-11-08" / "_material" / "H3-Ref-1080p-30s.mp4"
MATERIAL_NAME = "src_1280x720_124f.mp4"
CONTROL_NAME = "control_write_mp4_1280x720_124f.mp4"
SRC_W, SRC_H = 1280, 720
#: 実験 E1: 切り出しの上 2 行と下 2 行（素材の縁の暗い行）を落とした素材とその対照。
TRIM_NAME = "src_1280x716_124f_trim2.mp4"
TRIM_CONTROL_NAME = "control_write_mp4_1280x716_124f_trim2.mp4"
TRIM_ROWS = 2
TRIM_W, TRIM_H = SRC_W, SRC_H - 2 * TRIM_ROWS
NUM_FRAMES = 124                  # 17x7+5
START_SEC = 2.0                   # 開始フレームは製品の窓と同じく round(開始秒x24)

#: 前例 Docs/Outputs-archive/h3-chain-2026-10-29/c_chain.py の PATHS と同じ（FL2VA）。
PATHS = {
    "transformer": "models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoder": "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "video_vae": "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
    "tokenizer_dir": "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
}
#: Turbo LoRA。置き場は scripts/manifests/40-minimaxh3.json の "lora_dir": "MiniMaxH3/LoRA"（models/ からの相対）。
#: 名前・強度 0.8・4 ステップ・shift 12/3 据え置きは前例 h3-vsf-2026-11-09/h3_vsf_gate.py の T 系列と
#: h3-chain-a2v-2026-11-09/h3_chain_a2v_gate.py の GA3T の写し。エンジンには worker.py と同じく
#: H3LoraInput(path=<ファイルの絶対パス>, strength) で渡す（アプリが名前をパスへ直す役を台本が持つ）。
TURBO_LORA_FILE = "models/MiniMaxH3/LoRA/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
TURBO_LORA_STRENGTH = 0.8
TURBO_STEPS = 4
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 3.0
#: 読み込み時の値は H3LoadOptions の既定（前例 c_chain.py と同じ）。
BLOCKS_ON_GPU = 8
PREFETCH = True
DETERMINISTIC = False

DEFAULT_SEED = 1234
DEFAULT_PROMPT = (
    "A wide shot of a young woman in a black-and-white maid costume talking cheerfully to the camera in a "
    "spacious pastel room decorated with sweets and flowers; natural head and hand movement, steady camera."
)

#: G0 の灰色（ImageNet の平均 (0.485, 0.456, 0.406)x255 の丸めと、128 の灰色）。
G0_GRAYS = {"imagenet": (124, 116, 104), "gray128": (128, 128, 128)}
G0_BAND_PX = 32
#: 境界の勾配比の近傍の幅（前例 outputs/ltx25-outpaint-gate/analyze_w2.py の _BOUNDARY_WINDOW と同じ）。
BOUNDARY_WINDOW = 8
PSNR_IDENTICAL = 100.0

MAIN_PADS = {"pad_left": 320, "pad_right": 320, "pad_top": 184, "pad_bottom": 184}
ODD_PADS = {"pad_left": 13, "pad_right": 19, "pad_top": 8, "pad_bottom": 8}
TRIM_PADS = {"pad_left": 320, "pad_right": 320, "pad_top": 186, "pad_bottom": 186}   # 716 + 372 = 1088
GATES = ("G0", "G1", "G2", "G3", "G4", "E1", "E2")
DEFAULT_GATES = ("G0", "G1", "G2", "G3", "G4")   # E1・E2 は --gate で明示したときだけ


def feather_px_of(r: int, canvas_w: int, canvas_h: int) -> int:
    """計画 §2.2: F = floor(r x max(W_c, H_c) / 64 + 0.5)。"""
    return int(np.floor(r * max(canvas_w, canvas_h) / 64.0 + 0.5))


def run_plan() -> list[dict]:
    """生成の本の一覧（G0 は別）。r=1・s=0.5・注入 1.0 の主条件は G1 の 1 本を G2／G4 と共用する。"""
    def one(label, gate, pads, r=1, s=0.5, inject=1.0, purpose="", trim=False):
        sw, sh = (TRIM_W, TRIM_H) if trim else (SRC_W, SRC_H)
        cw = sw + pads["pad_left"] + pads["pad_right"]
        ch = sh + pads["pad_top"] + pads["pad_bottom"]
        return {"label": label, "gate": gate, "pads": dict(pads), "canvas": [cw, ch], "r": r,
                "feather_px": feather_px_of(r, cw, ch), "audio_strength": s, "inject_clean": inject,
                "purpose": purpose, "source": [sw, sh],
                "material": TRIM_NAME if trim else MATERIAL_NAME,
                "control": TRIM_CONTROL_NAME if trim else CONTROL_NAME}
    return [
        one("G1_main", "G1", MAIN_PADS, purpose="主条件（G2 の r=1・G4 の s=0.5 と共用）"),
        one("G1_odd", "G1", ODD_PADS, purpose="奇数 pad・32 に乗らない"),
        one("G2_r0", "G2", MAIN_PADS, r=0, purpose="なじませ 0（硬い貼り付け）"),
        one("G2_r3", "G2", MAIN_PADS, r=3, purpose="なじませ r=3"),
        one("G3_inject0999", "G3", MAIN_PADS, inject=0.999, purpose="注入係数 0.999 の対照"),
        one("G4_s1", "G4", MAIN_PADS, s=1.0, purpose="音声 s=1.0（元の波形）"),
        one("E1_trim_r1", "E1", TRIM_PADS, trim=True, purpose="素材の上下 2 行を落とした素材・なじませ r=1"),
        one("E1_trim_r0", "E1", TRIM_PADS, r=0, trim=True, purpose="素材の上下 2 行を落とした素材・なじませ 0"),
    ]


# --------------------------------------------------------------------------- #
# 小物
# --------------------------------------------------------------------------- #

class Log:
    """標準出力と、そのときの本のログファイルへ同じ行を書く。"""

    def __init__(self) -> None:
        self.file = None

    def __call__(self, msg: str) -> None:
        line = f"[h3_outpaint_gate] {msg}"
        print(line, flush=True)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()

    def progress(self, stage: str, index: int, total: int) -> None:
        line = f"[progress] {stage} {index}/{total}"
        print(line, flush=True)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()

    def open(self, path: Path) -> None:
        self.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


LOG = Log()


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def rel(p) -> str:
    q = Path(p).resolve()
    try:
        return str(q.relative_to(ROOT))
    except ValueError:
        return str(q)


def jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jdump(obj) + "\n", encoding="utf-8")


def tool(name: str) -> str:
    local = ROOT / "tools" / "ffmpeg" / "bin" / f"{name}.exe"
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} が見つかりません（<repo>\\tools\\ffmpeg\\bin か PATH）")
    return found


def ffprobe(path: Path) -> dict:
    """映像（枚数は -count_frames の実測）と音声の要点。"""
    exe = tool("ffprobe")
    v = subprocess.run(
        [exe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,duration",
         "-of", "json", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    a = subprocess.run(
        [exe, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels,duration", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    vs = (json.loads(v.stdout or "{}").get("streams") or [{}])[0]
    as_ = (json.loads(a.stdout or "{}").get("streams") or [None])[0]
    nb = vs.get("nb_read_frames")
    return {"path": rel(path), "video": {**vs, "nb_read_frames": int(nb) if nb not in (None, "N/A") else None},
            "audio": as_}


# --------------------------------------------------------------------------- #
# 判定の量（前例 analyze_w2.py の psnr・keep・boundary_gradient_ratio と同じ量を、最小の形で書き直したもの）
# --------------------------------------------------------------------------- #

def psnr_from_mse(mse: float) -> float:
    return PSNR_IDENTICAL if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


def psnr_frames(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """uint8 [F, H, W, 3] 同士の PSNR（全フレームまとめた MSE）。mask は bool [H, W]（True の画素だけ）。"""
    if a.shape != b.shape:
        raise ValueError(f"psnr: 形が違います {a.shape} と {b.shape}")
    total, count = 0.0, 0
    for i in range(a.shape[0]):
        d = a[i].astype(np.float32) - b[i].astype(np.float32)
        if mask is not None:
            d = d[mask]
        total += float(np.square(d, dtype=np.float32).sum(dtype=np.float64))
        count += d.size
    return psnr_from_mse(total / max(count, 1))


def edge_band_mask(h: int, w: int, band: int, sides=("left", "right", "top", "bottom")) -> np.ndarray:
    """bool [h, w]: 指定した辺から band 画素以内の画素。"""
    m = np.zeros((h, w), dtype=bool)
    if "left" in sides:
        m[:, :band] = True
    if "right" in sides:
        m[:, w - band:] = True
    if "top" in sides:
        m[:band, :] = True
    if "bottom" in sides:
        m[h - band:, :] = True
    return m


def boundary_gradients(frames: np.ndarray, x: int, y: int, w: int, h: int) -> dict:
    """保持矩形の 4 辺の境界の勾配比（境界をまたぐ差の平均 / 両側 8 画素の近傍の差の平均）。

    analyze_w2.boundary_gradient_ratio と同じ量（縦の境界＝左右の列）に、横の境界（上下の行）を足した。
    差は輝度の代わりに RGB の平均。全フレームを足し合わせてから比にする。キャンバスの端に接する辺
    （pad 0）は測らない（None）。1.0 は近傍と見分けがつかない、大きいほど継ぎ目が立つ。
    """
    F, H, W, _ = frames.shape
    sides = {}
    if x > 0:
        sides["left"] = ("v", x - 1)
    if W - (x + w) > 0:
        sides["right"] = ("v", x + w - 1)
    if y > 0:
        sides["top"] = ("h", y - 1)
    if H - (y + h) > 0:
        sides["bottom"] = ("h", y + h - 1)
    acc = {k: [0.0, 0, 0.0, 0] for k in sides}
    for i in range(F):
        g = frames[i].astype(np.float32).mean(axis=2)
        dx = np.abs(np.diff(g, axis=1))   # dx[:, c] = |g[:, c+1] - g[:, c]|
        dy = np.abs(np.diff(g, axis=0))   # dy[r, :] = |g[r+1, :] - g[r, :]|
        for name, (kind, k) in sides.items():
            if kind == "v":
                lo, hi = max(0, k - BOUNDARY_WINDOW), min(dx.shape[1], k + BOUNDARY_WINDOW + 1)
                b = dx[y:y + h, k]
                win = np.concatenate([dx[y:y + h, lo:k], dx[y:y + h, k + 1:hi]], axis=1)
            else:
                lo, hi = max(0, k - BOUNDARY_WINDOW), min(dy.shape[0], k + BOUNDARY_WINDOW + 1)
                b = dy[k, x:x + w]
                win = np.concatenate([dy[lo:k, x:x + w], dy[k + 1:hi, x:x + w]], axis=0)
            a = acc[name]
            a[0] += float(b.sum(dtype=np.float64))
            a[1] += b.size
            a[2] += float(win.sum(dtype=np.float64))
            a[3] += win.size
    out = {k: None for k in ("left", "right", "top", "bottom")}
    for name, (sb, nb, sw, nw) in acc.items():
        local = sw / max(nw, 1)
        out[name] = None if local <= 1e-9 else round((sb / max(nb, 1)) / local, 4)
    vert = [out[k] for k in ("left", "right") if out[k] is not None]
    horz = [out[k] for k in ("top", "bottom") if out[k] is not None]
    out["vertical_max"] = max(vert) if vert else None
    out["horizontal_max"] = max(horz) if horz else None
    return out


def pad_band_std(frames: np.ndarray, x: int, y: int, w: int, h: int, step: int = 8) -> dict:
    """pad の帯（保持矩形の外）の画素の標準偏差（チャンネル平均）。平らな帯＝絵が出ていない目安。"""
    F, H, W, _ = frames.shape
    regions = {
        "left": (slice(y, y + h), slice(0, x)), "right": (slice(y, y + h), slice(x + w, W)),
        "top": (slice(0, y), slice(0, W)), "bottom": (slice(y + h, H), slice(0, W)),
    }
    out = {}
    for name, (rs, cs) in regions.items():
        sub = frames[::step, rs, cs]
        out[name] = None if sub.size == 0 else round(float(sub.reshape(-1, 3).astype(np.float32).std(axis=0).mean()), 3)
    return out


def read_frames(path: Path, n: int) -> np.ndarray:
    from engine_h3.source_tail import read_source_frames  # noqa: PLC0415

    return read_source_frames(str(path), n)


# --------------------------------------------------------------------------- #
# --prepare
# --------------------------------------------------------------------------- #

def prepare(out_root: Path, record: dict) -> None:
    """素材と対照を用意する（CPU）。既にある素材は作り直さない（無いものだけ作り、prepare.json に追記する）。"""
    mat_dir = out_root / "_material"
    pj = mat_dir / "prepare.json"
    have_main = pj.is_file() and (mat_dir / MATERIAL_NAME).is_file() and (mat_dir / CONTROL_NAME).is_file()
    if have_main:
        record.update(json.loads(pj.read_text(encoding="utf-8")))
        LOG(f"prepare: 既存の {MATERIAL_NAME} と {CONTROL_NAME} を使う（作り直さない）")
    else:
        _prepare_main(out_root, record)
    if "material_trim2" in record and (mat_dir / TRIM_NAME).is_file() and (mat_dir / TRIM_CONTROL_NAME).is_file():
        LOG(f"prepare: 既存の {TRIM_NAME} と {TRIM_CONTROL_NAME} を使う（作り直さない）")
    else:
        _prepare_trim2(out_root, record)
    pdata = json.loads(pj.read_text(encoding="utf-8")) if pj.is_file() else {}
    pdata.update({k: record[k] for k in ("material", "control", "material_trim2", "control_trim2") if k in record})
    write_json(pj, pdata)   # E2 の項目（e2:<label>）は残す


def _prepare_trim2(out_root: Path, record: dict) -> None:
    """実験 E1: 切り出し 1280x720 から上 2 行と下 2 行を落とした 1280x716 と、その write_mp4 の対照。"""
    from engine_h3.media_out import write_mp4  # noqa: PLC0415

    mat_dir = out_root / "_material"
    src = mat_dir / MATERIAL_NAME
    out = mat_dir / TRIM_NAME
    # 再符号化は切り出しと同じ条件（crf 12・yuv420p・CFR 24fps）。音声はそのまま写す。
    cmd = [tool("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0",
           "-vf", f"crop={TRIM_W}:{TRIM_H}:0:{TRIM_ROWS}",
           "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", str(FRAME_RATE),
           "-c:a", "copy", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg の trim2 に失敗しました（{proc.returncode}）: {proc.stderr[-2000:]}")
    probe = ffprobe(out)
    LOG(f"prepare: {rel(out)} ffprobe={jdump(probe)}")
    vid = probe["video"]
    if vid.get("nb_read_frames") != NUM_FRAMES or (vid.get("width"), vid.get("height")) != (TRIM_W, TRIM_H):
        raise SystemExit(f"trim2 の枚数か寸法が合いません（{vid.get('nb_read_frames')} 枚・"
                         f"{vid.get('width')}x{vid.get('height')}。{NUM_FRAMES} 枚・{TRIM_W}x{TRIM_H} が必要）")
    frames, wave, info = OH.read_outpaint_source(str(out), NUM_FRAMES, FRAME_RATE, 0.5, log=LOG)
    LOG(f"prepare: trim2 read_outpaint_source(s=0.5) frames={list(frames.shape)} "
        f"wave={None if wave is None else list(wave.shape)} info={info}")
    control = mat_dir / TRIM_CONTROL_NAME
    write_mp4(str(control), torch.from_numpy(frames), fps=FRAME_RATE, audio=None)
    ctrl_psnr = psnr_frames(read_frames(control, NUM_FRAMES), frames)
    LOG(f"prepare: control {rel(control)} PSNR(control vs trim2)={ctrl_psnr:.3f} dB")
    record["material_trim2"] = {"source": rel(src), "cut": rel(out), "ffmpeg_cmd": cmd, "audio_codec": "copy",
                                "crop": f"crop={TRIM_W}:{TRIM_H}:0:{TRIM_ROWS}", "probe": probe,
                                "read_outpaint_source_s05": info}
    record["control_trim2"] = {"path": rel(control), "psnr_vs_cut_db": round(ctrl_psnr, 4),
                               "note": "trim2 の画素を engine_h3.media_out.write_mp4 で書き直したもの（音声なし）"}


def _prepare_main(out_root: Path, record: dict) -> None:
    """素材を 1280x720・24fps・ちょうど 124 枚に切り出し、write_mp4 で書き直した対照も作る（CPU）。"""
    mat_dir = out_root / "_material"
    mat_dir.mkdir(parents=True, exist_ok=True)
    src = MATERIAL_SRC
    if not src.is_file():
        raise SystemExit(f"元の素材がありません: {src}")
    out = mat_dir / MATERIAL_NAME
    start = round(START_SEC * FRAME_RATE)
    end = start + NUM_FRAMES - 1
    # services/video_io.py の cut_window_mp4 と同じ作法（select の窓・atrim・crf 12・yuv420p・CFR・
    # 音声は PCM、断られたら AAC 192k）。素材は 24fps なので fps の変換はしない。縮小だけを足す。
    fc = (f"[0:v]select='between(n\\,{start}\\,{end})',setpts=PTS-STARTPTS,"
          f"scale={SRC_W}:{SRC_H}:flags=lanczos[v];"
          f"[0:a]atrim=start={start / FRAME_RATE}:end={(end + 1) / FRAME_RATE},asetpts=PTS-STARTPTS[a]")

    def cut(audio_args):
        cmd = [tool("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-filter_complex", fc,
               "-map", "[v]", "-map", "[a]", *audio_args,
               "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", str(FRAME_RATE),
               str(out)]
        return cmd, subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    cmd, proc = cut(["-c:a", "pcm_s16le"])
    audio_codec = "pcm_s16le"
    if proc.returncode != 0:
        cmd, proc = cut(["-c:a", "aac", "-b:a", "192k"])
        audio_codec = "aac 192k"
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg の切り出しに失敗しました（{proc.returncode}）: {proc.stderr[-2000:]}")
    probe = ffprobe(out)
    LOG(f"prepare: {rel(out)} ffprobe={jdump(probe)}")
    vid = probe["video"]
    if vid.get("nb_read_frames") != NUM_FRAMES or (vid.get("width"), vid.get("height")) != (SRC_W, SRC_H):
        raise SystemExit(f"切り出しの枚数か寸法が合いません（{vid.get('nb_read_frames')} 枚・"
                         f"{vid.get('width')}x{vid.get('height')}。{NUM_FRAMES} 枚・{SRC_W}x{SRC_H} が必要）")
    # s=1.0 は窓の全長 round(F x 32000 / 24) サンプルが必須（engine_h3.outpaint_h3.read_outpaint_source）。
    frames, wave, info = OH.read_outpaint_source(str(out), NUM_FRAMES, FRAME_RATE, 1.0, log=LOG)
    LOG(f"prepare: read_outpaint_source(s=1.0) frames={list(frames.shape)} wave={list(wave.shape)} info={info}")
    # 対照: 切り出しの画素を、出力と同じ write_mp4（libx264・yuv420p・BT.709）で書き直す。
    from engine_h3.media_out import write_mp4  # noqa: PLC0415

    control = mat_dir / CONTROL_NAME
    write_mp4(str(control), torch.from_numpy(frames), fps=FRAME_RATE, audio=None)
    ctrl = read_frames(control, NUM_FRAMES)
    ctrl_psnr = psnr_frames(ctrl, frames)
    LOG(f"prepare: control {rel(control)} PSNR(control vs cut)={ctrl_psnr:.3f} dB")
    record["material"] = {
        "source": rel(src), "source_probe": ffprobe(src), "cut": rel(out), "ffmpeg_cmd": cmd,
        "audio_codec": audio_codec, "start_frame": start, "end_frame": end, "probe": probe,
        "read_outpaint_source_s1": info, "wave_shape_s1": list(wave.shape),
    }
    record["control"] = {"path": rel(control), "psnr_vs_cut_db": round(ctrl_psnr, 4),
                         "note": "切り出しの画素を engine_h3.media_out.write_mp4 で書き直したもの（音声なし）"}


def load_prepared(out_root: Path, record: dict, runs: list[dict]) -> None:
    p = out_root / "_material" / "prepare.json"
    needed = {MATERIAL_NAME, CONTROL_NAME} | {r["material"] for r in runs} | {r["control"] for r in runs}
    missing = [n for n in sorted(needed) if not (out_root / "_material" / n).is_file()]
    if not p.is_file() or missing:
        raise SystemExit(f"素材がありません（{missing or 'prepare.json'}）。先に --prepare を実行してください")
    record.update(json.loads(p.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# 要求の組み立て
# --------------------------------------------------------------------------- #

def run_attention(run: dict, args) -> str:
    """注意の実装: --attention があればそれ、無ければ E2 は sage（オーナー了承の時短）、他は sdpa（従来どおり）。"""
    if args.attention:
        return args.attention
    return "sage" if run.get("gate") == "E2" else "sdpa"


# --------------------------------------------------------------------------- #
# E2（汎用の実験）
# --------------------------------------------------------------------------- #

def _int_list(text: str, n: int | None, what: str) -> list[int]:
    try:
        vals = [int(v.strip()) for v in str(text).split(",") if v.strip() != ""]
    except ValueError:
        raise SystemExit(f"{what} は整数のカンマ区切りです（{text!r}）") from None
    if n is not None and len(vals) != n:
        raise SystemExit(f"{what} は {n} 個の整数です（{text!r}）")
    return vals


def probe_fps(path: Path) -> float | None:
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate", "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()
    try:
        num, _, den = out.partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None


def prepare_e2(args, out_root: Path, record: dict) -> dict:
    """E2 の素材: --material を 24fps・ちょうど --frames 枚・偶数寸法に切り出し、write_mp4 の対照も作る（CPU）。

    作法は --prepare と同じ（select の窓・atrim・crf 12・yuv420p・CFR・音声は PCM、断られたら AAC 192k）。
    素材が 24fps でなければ先に fps=24 で揃えてから窓を取る（cut_window_mp4 と同じ順）。--scale W:H は
    縦横比を保って W:H を覆うまで lanczos で縮め、中央で W:H に切り抜く。--scale が無いときは奇数の辺だけ
    1 画素を切り落とす。同じ名前と同じ引数の素材が既にあれば作り直さない。
    """
    from engine_h3.media_out import write_mp4  # noqa: PLC0415

    if not args.material:
        raise SystemExit("E2 には --material が要ります")
    src = Path(args.material).resolve()
    if not src.is_file():
        raise SystemExit(f"素材がありません: {src}")
    frames_n = int(args.frames)
    latent_frames(frames_n)   # 17n+5 でなければここで止まる
    sprobe = ffprobe(src)
    sw0, sh0 = int(sprobe["video"]["width"]), int(sprobe["video"]["height"])
    if args.scale:
        tw, th = _int_list(args.scale.replace(":", ","), 2, "--scale")
        if tw % 2 or th % 2 or tw <= 0 or th <= 0:
            raise SystemExit(f"--scale は正の偶数の W:H にしてください（{args.scale}）")
        geom_vf = (f"scale={tw}:{th}:flags=lanczos:force_original_aspect_ratio=increase,"
                   f"crop={tw}:{th}:(iw-{tw})/2:(ih-{th})/2")
    else:
        tw, th = sw0 - sw0 % 2, sh0 - sh0 % 2
        geom_vf = f"crop={tw}:{th}:{(sw0 - tw) // 2}:{(sh0 - th) // 2}" if (tw, th) != (sw0, sh0) else ""
    name = f"{args.label}_{tw}x{th}_{frames_n}f.mp4"
    cname = f"{args.label}_{tw}x{th}_{frames_n}f_control.mp4"
    key = f"e2:{args.label}"
    params = {"source": str(src), "frames": frames_n, "start_sec": float(args.start_sec), "scale": args.scale,
              "size": [tw, th]}
    mat_dir = out_root / "_material"
    mat_dir.mkdir(parents=True, exist_ok=True)
    pj = mat_dir / "prepare.json"
    pdata = json.loads(pj.read_text(encoding="utf-8")) if pj.is_file() else {}
    old = pdata.get(key) or {}
    if old.get("params") == params and (mat_dir / name).is_file() and (mat_dir / cname).is_file():
        LOG(f"prepare E2: 既存の {name} と {cname} を使う（作り直さない）")
        record[key] = old
        return old
    fps = probe_fps(src)
    if fps is None or fps <= 0:
        raise SystemExit(f"素材の fps が読めません: {src}")
    has_audio = sprobe.get("audio") is not None
    start = max(0, round(float(args.start_sec) * FRAME_RATE))
    end = start + frames_n - 1
    vchain = ([f"fps={FRAME_RATE}"] if abs(fps - FRAME_RATE) > 1e-3 else []) + [
        f"select='between(n\\,{start}\\,{end})'", "setpts=PTS-STARTPTS"] + ([geom_vf] if geom_vf else [])
    fc = "[0:v]" + ",".join(vchain) + "[v]"
    if has_audio:
        fc += f";[0:a]atrim=start={start / FRAME_RATE}:end={(end + 1) / FRAME_RATE},asetpts=PTS-STARTPTS[a]"
    out = mat_dir / name

    def cut(audio_args):
        cmd = [tool("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-filter_complex", fc, "-map", "[v]"]
        if has_audio:
            cmd += ["-map", "[a]", *audio_args]
        cmd += ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", str(FRAME_RATE),
                str(out)]
        return cmd, subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    cmd, proc = cut(["-c:a", "pcm_s16le"])
    audio_codec = "pcm_s16le" if has_audio else None
    if proc.returncode != 0 and has_audio:
        cmd, proc = cut(["-c:a", "aac", "-b:a", "192k"])
        audio_codec = "aac 192k"
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg の E2 の切り出しに失敗しました（{proc.returncode}）: {proc.stderr[-2000:]}")
    probe = ffprobe(out)
    vid = probe["video"]
    LOG(f"prepare E2: {rel(out)} {vid.get('width')}x{vid.get('height')} {vid.get('nb_read_frames')} 枚 "
        f"audio={probe.get('audio')}")
    if vid.get("nb_read_frames") != frames_n or (vid.get("width"), vid.get("height")) != (tw, th):
        raise SystemExit(f"E2 の切り出しの枚数か寸法が合いません（{vid.get('nb_read_frames')} 枚・"
                         f"{vid.get('width')}x{vid.get('height')}。{frames_n} 枚・{tw}x{th} が必要）")
    s = float(args.audio_strength)
    frames, wave, info = OH.read_outpaint_source(str(out), frames_n, FRAME_RATE, s, log=LOG)
    control = mat_dir / cname
    write_mp4(str(control), torch.from_numpy(frames), fps=FRAME_RATE, audio=None)
    ctrl_psnr = psnr_frames(read_frames(control, frames_n), frames)
    LOG(f"prepare E2: control {rel(control)} PSNR(control vs cut)={ctrl_psnr:.3f} dB")
    entry = {"params": params, "material": name, "control": cname, "source_probe": sprobe, "source_fps": fps,
             "resampled": abs(fps - FRAME_RATE) > 1e-3, "start_frame": start, "end_frame": end, "ffmpeg_cmd": cmd,
             "audio_codec": audio_codec, "probe": probe, "read_outpaint_source": info,
             "control_psnr_vs_cut_db": round(ctrl_psnr, 4)}
    pdata[key] = entry
    write_json(pj, pdata)
    record[key] = entry
    return entry


def e2_runs(args, entry: dict) -> list[dict]:
    """E2 の本の一覧（r ごとに 1 本）。キャンバスは素材＋pad（--canvas があれば一致を確かめる）。"""
    sw, sh = entry["params"]["size"]
    pl, pr, pt, pb = _int_list(args.pads, 4, "--pads")
    cw, ch = sw + pl + pr, sh + pt + pb
    if args.canvas:
        gw, gh = _int_list(args.canvas, 2, "--canvas")
        if (gw, gh) != (cw, ch):
            raise SystemExit(f"--canvas {gw}x{gh} が素材 {sw}x{sh} と pad の和 {cw}x{ch} に一致しません")
    if cw % TOKEN_PX or ch % TOKEN_PX:
        raise SystemExit(f"キャンバス {cw}x{ch} が {TOKEN_PX} の倍数ではありません（素材 {sw}x{sh}・pad {args.pads}）")
    pads = {"pad_left": pl, "pad_right": pr, "pad_top": pt, "pad_bottom": pb}
    out = []
    for r in _int_list(args.feathers, None, "--feathers"):
        out.append({"label": f"{args.label}_r{r}", "gate": "E2", "pads": dict(pads), "canvas": [cw, ch], "r": r,
                    "feather_px": feather_px_of(r, cw, ch), "audio_strength": float(args.audio_strength),
                    "inject_clean": 1.0, "purpose": f"汎用の実験（素材 {Path(entry['params']['source']).name}）",
                    "source": [sw, sh], "material": entry["material"], "control": entry["control"],
                    "frames": int(entry["params"]["frames"])})
    return out


def build_request(run: dict, args, out_root: Path) -> P.H3GenerateRequest:
    cw, ch = run["canvas"]
    return P.H3GenerateRequest(
        prompt=args.prompt, seed=int(args.seed), width=cw, height=ch, num_frames=int(run.get("frames", NUM_FRAMES)),
        output_path=str(out_root / run["label"] / "output.mp4"),
        num_inference_steps=int(args.steps), shift_video=SHIFT_VIDEO, shift_audio=SHIFT_AUDIO,
        keep_resident=True, attention_backend=run_attention(run, args), block_swap_prefetch=PREFETCH,
        vram16_mode=not args.vram16_off,
        loras=() if args.no_lora else (P.H3LoraInput(path=str(ROOT / TURBO_LORA_FILE), strength=TURBO_LORA_STRENGTH),),
        outpaint=P.H3OutpaintInput(
            path=str(out_root / "_material" / run.get("material", MATERIAL_NAME)),
            audio_strength=float(run["audio_strength"]),
            feather_px=int(run["feather_px"]), **run["pads"],
        ),
    )


def geometry_record(run: dict) -> dict:
    cw, ch = run["canvas"]
    sw, sh = run.get("source", [SRC_W, SRC_H])
    g = compute_outpaint_geometry(sw, sh, cw, ch, **run["pads"])
    return {"canvas": [cw, ch], "source": [sw, sh], "aligned": [g.x0, g.y0, g.x1, g.y1],
            "aligned_size": [g.aligned_w, g.aligned_h], "grid": [g.grid_w, g.grid_h],
            "fixed_tokens": g.fixed_count, "generated_tokens": g.generated_count,
            "latent_offset_yx": list(g.latent_offset), "source_offset_in_aligned_yx": list(g.source_offset_in_aligned),
            "latent_frames": latent_frames(int(run.get("frames", NUM_FRAMES)))}


# --------------------------------------------------------------------------- #
# G0（DiT なし）
# --------------------------------------------------------------------------- #

def run_g0(out_root: Path, record: dict) -> bool:
    from engine_h3.media_out import write_mp4  # noqa: PLC0415
    from engine_h3.video_vae import H3VideoDecoder, H3VideoEncoder  # noqa: PLC0415

    g0_dir = out_root / "G0"
    LOG.open(g0_dir / "g0.log")
    rec: dict = {"label": "G0", "started_at": now(), "status": "running"}
    record["g0"] = rec
    dev = torch.device("cuda")
    try:
        torch.set_grad_enabled(False)
        P.require_fp32_matmul()
        P.pin_cublas_workspace(dev)
        pads = MAIN_PADS
        cw, ch = SRC_W + pads["pad_left"] + pads["pad_right"], SRC_H + pads["pad_top"] + pads["pad_bottom"]
        g = compute_outpaint_geometry(SRC_W, SRC_H, cw, ch, **pads)
        rec["geometry"] = geometry_record({"canvas": [cw, ch], "pads": pads})
        src = read_frames(out_root / "_material" / MATERIAL_NAME, NUM_FRAMES)
        # 外接矩形の中の固定トークンの潜在マス（2x2）と、そのうち生成トークンに隣り合う「縁」。
        ty0, tx0, ty1, tx1 = g.y0 // TOKEN_PX, g.x0 // TOKEN_PX, g.y1 // TOKEN_PX, g.x1 // TOKEN_PX
        fixed_box = np.asarray(g.fixed[ty0:ty1, tx0:tx1])
        padded = np.pad(fixed_box, 1, constant_values=False)
        inner = fixed_box.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                inner &= padded[1 + dy:1 + dy + fixed_box.shape[0], 1 + dx:1 + dx + fixed_box.shape[1]]
        edge = fixed_box & ~inner
        cells = {name: torch.from_numpy(m.repeat(2, 0).repeat(2, 1)) for name, m in
                 (("fixed", fixed_box), ("fixed_edge", edge), ("fixed_inner", inner))}
        rec["fixed_tokens_in_box"] = {k: int(v.sum()) // 4 for k, v in cells.items()}

        enc_t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        enc = H3VideoEncoder.from_file(str(ROOT / PATHS["video_vae"]), device=dev, dtype=torch.float32)
        lat: dict[str, torch.Tensor] = {}
        # (i) 単独: 外接矩形（鏡写しのすき間）を符号化。製品の _encode_source_tail と同じ encode_clips(tiled=True)。
        lat["single"] = enc.encode_clips(aligned_frames(src, g), device=dev, tiled=True).float().cpu()
        # (ii) キャンバスごと: 余白を灰色で埋めて 1920x1088 を符号化し、外接矩形の潜在を切り出す。
        ly0, lx0, ly1, lx1 = g.y0 // LATENT_PX, g.x0 // LATENT_PX, g.y1 // LATENT_PX, g.x1 // LATENT_PX
        for name, rgb in G0_GRAYS.items():
            canvas = np.empty((NUM_FRAMES, ch, cw, 3), dtype=np.uint8)
            canvas[...] = np.asarray(rgb, dtype=np.uint8)
            canvas[:, g.pad_top:g.pad_top + SRC_H, g.pad_left:g.pad_left + SRC_W] = src
            z = enc.encode_clips(canvas, device=dev, tiled=True)
            lat[f"canvas_{name}"] = z[:, :, :, ly0:ly1, lx0:lx1].float().cpu().contiguous()
            del canvas, z
        # (iii) 対照: 元の 1280x720 を単独でそのまま。
        lat["control_source"] = enc.encode_clips(src, device=dev, tiled=True).float().cpu()
        del enc
        P._free_cuda()
        rec["encode_seconds"] = round(time.perf_counter() - enc_t0, 2)
        rec["encode_peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        rec["latent_shapes"] = {k: list(v.shape) for k, v in lat.items()}

        diffs = {}
        ref = lat["single"]
        for name in G0_GRAYS:
            other = lat[f"canvas_{name}"]
            d = {}
            for cname, m in cells.items():
                a, b = ref[:, :, :, m], other[:, :, :, m]
                den = float(torch.linalg.vector_norm(a))
                d[cname] = None if a.numel() == 0 else round(float(torch.linalg.vector_norm(a - b)) / max(den, 1e-12), 6)
            d["whole_box"] = round(float(torch.linalg.vector_norm(ref - other)) / float(torch.linalg.vector_norm(ref)), 6)
            diffs[f"single_vs_canvas_{name}"] = d
        rec["latent_rel_l2"] = diffs

        dec_t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        dec = H3VideoDecoder.from_file(str(ROOT / PATHS["video_vae"]), device=dev, dtype=torch.bfloat16)
        oy, ox = g.source_offset_in_aligned
        band = edge_band_mask(SRC_H, SRC_W, G0_BAND_PX)
        psnr = {}
        for name, z in lat.items():
            frames = dec.decode(z.to(dev), tiled=True)   # CPU uint8 [F, h, w, 3]
            arr = frames.numpy()
            keep = arr if name == "control_source" else arr[:, oy:oy + SRC_H, ox:ox + SRC_W]
            if keep.shape != src.shape:
                raise RuntimeError(f"G0 {name}: 復号の形 {list(arr.shape)} から保持矩形を取れません")
            full_db = psnr_frames(keep, src)
            band_db = psnr_frames(keep, src, band)
            inner_db = psnr_frames(keep, src, ~band)
            psnr[name] = {"keep_db": round(full_db, 4), "edge32_db": round(band_db, 4),
                          "interior_db": round(inner_db, 4), "edge_minus_interior_db": round(band_db - inner_db, 4),
                          "decoded_shape": list(arr.shape)}
            write_mp4(str(g0_dir / f"{name}.mp4"), frames, fps=FRAME_RATE, audio=None)
            LOG(f"G0 {name}: {psnr[name]}")
            del frames, arr, keep
        del dec
        P._free_cuda()
        rec["decode_seconds"] = round(time.perf_counter() - dec_t0, 2)
        rec["decode_peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        rec["psnr_vs_cut"] = psnr
        rec["notes"] = {
            "single": "外接矩形 [x0,y0)-[x1,y1) を aligned_frames（鏡写し）で組んで単独符号化（製品の経路）",
            "canvas_*": "1920x1088 の余白を灰色で埋めてキャンバスごと符号化し、外接矩形の潜在を切り出した",
            "control_source": "元の 1280x720 を単独で符号化→復号（VAE の往復の下限）",
            "edge32": f"保持矩形の 4 辺から内側 {G0_BAND_PX} 画素の帯。interior はその内側",
            "latent_rel_l2": "single を基準にした相対 L2（全チャンネル・全潜在フレーム）。fixed_edge は生成トークンに"
                             "8 近傍で接する固定トークン、fixed_inner はその内側",
        }
        rec["status"] = "done"
        return True
    except Exception as exc:  # noqa: BLE001 -- 記録して次へ
        rec["status"] = "failed"
        rec["error"] = repr(exc)
        rec["traceback"] = traceback.format_exc()
        LOG(f"G0 failed: {exc!r}\n{rec['traceback']}")
        return False
    finally:
        rec["finished_at"] = now()
        write_json(g0_dir / "result.json", rec)
        P._free_cuda()
        LOG.close()


# --------------------------------------------------------------------------- #
# G1〜G4（生成）
# --------------------------------------------------------------------------- #

def measure_run(run: dict, out_path: Path, out_root: Path) -> dict:
    """出力 mp4 の保持矩形を、対照（write_mp4 で書き直した切り出し）と切り出しそのものと比べる。"""
    cw, ch = run["canvas"]
    pads = run["pads"]
    x, y = pads["pad_left"], pads["pad_top"]
    sw, sh = run["source"]
    nf = int(run.get("frames", NUM_FRAMES))
    out = read_frames(out_path, nf)
    if out.shape[1:3] != (ch, cw):
        raise RuntimeError(f"出力の寸法 {out.shape[2]}x{out.shape[1]} がキャンバス {cw}x{ch} と違います")
    keep = out[:, y:y + sh, x:x + sw]
    control = read_frames(out_root / "_material" / run["control"], nf)
    cut = read_frames(out_root / "_material" / run["material"], nf)
    g = compute_outpaint_geometry(sw, sh, cw, ch, **pads)
    core = feather_weights(g, int(run["feather_px"])) == 1.0
    m = {
        "keep_psnr_vs_control_db": round(psnr_frames(keep, control), 4),
        "keep_core_psnr_vs_control_db": round(psnr_frames(keep, control, core), 4),
        "keep_psnr_vs_cut_db": round(psnr_frames(keep, cut), 4),
        "keep_core_psnr_vs_cut_db": round(psnr_frames(keep, cut, core), 4),
        "core_pixels": int(core.sum()),
        "boundary_gradient_ratio": boundary_gradients(out, x, y, sw, sh),
        "pad_band_std": pad_band_std(out, x, y, sw, sh),
        "output_probe": ffprobe(out_path),
    }
    del out, keep, control, cut
    return m


def run_generations(runs: list[dict], args, out_root: Path, record: dict, save) -> bool:
    from engine_h3.pipeline_h3 import H3LoadOptions, H3Paths, H3Pipeline  # noqa: PLC0415

    all_ok = True
    load_rec: dict = {"started_at": now()}
    record["load"] = load_rec
    LOG.open(out_root / "load.log")
    t0 = time.perf_counter()
    try:
        options = H3LoadOptions(blocks_on_gpu=BLOCKS_ON_GPU, attention_backend=run_attention(runs[0], args),
                                block_swap_prefetch=PREFETCH, deterministic=DETERMINISTIC)
        paths = H3Paths(**{k: str(ROOT / v) for k, v in PATHS.items()})
        pipe = H3Pipeline(paths, options, device=torch.device("cuda"), progress=LOG.progress, log=LOG)
        load_rec["build_report"] = pipe.load()
        load_rec["seconds"] = round(time.perf_counter() - t0, 2)
    except Exception as exc:  # noqa: BLE001
        load_rec.update({"error": repr(exc), "traceback": traceback.format_exc()})
        LOG(f"load failed: {exc!r}")
        LOG.close()
        for run in runs:
            record["runs"].append({**run, "status": "skipped", "error": "読み込みに失敗したため回していない"})
        save()
        return False
    LOG.close()
    save()
    try:
        for run in runs:
            run_dir = out_root / run["label"]
            run_dir.mkdir(parents=True, exist_ok=True)
            LOG.open(run_dir / "run.log")
            rec = {**run, "started_at": now(), "status": "running", "geometry": geometry_record(run),
                   "conditions": {**record["conditions"], "attention": run_attention(run, args),
                                  "num_frames": int(run.get("frames", NUM_FRAMES))}}
            record["runs"].append(rec)
            save()
            wall = time.perf_counter()
            try:
                req = build_request(run, args, out_root)
                rec["request"] = dataclasses.asdict(req)
                OH.OUTPAINT_INJECT_CLEAN = float(run["inject_clean"])
                LOG(f"run {run['label']}: inject_clean={OH.OUTPAINT_INJECT_CLEAN} request={jdump(rec['request'])}")
                result = pipe.generate(req)
                write_json(run_dir / "result.json", result)
                rec["status"] = "done"
                h3 = result.get("h3", {})
                rec["outpaint_record"] = h3.get("outpaint")
                rec["loras_record"] = h3.get("loras")
                rec["decode_record"] = {k: (h3.get("phases", {}).get("decode") or {}).get(k)
                                        for k in ("audio_vae_source", "audio_samples", "audio_samples_source",
                                                  "audio_samples_decoded", "frames")}
                rec["peak_vram_mb"] = result.get("peak_vram_mb")
                rec["peak_vram_reserved_mb"] = result.get("peak_vram_reserved_mb")
                rec["seconds"] = result.get("seconds")
                rec["rss_peak_gib"] = result.get("rss_peak_gib")
                rec["commit_peak_gib"] = result.get("commit_peak_gib")
                LOG(f"run {run['label']}: done in {rec['seconds']} s, peak {rec['peak_vram_mb']} MiB; measuring")
                rec["metrics"] = measure_run(run, Path(req.output_path), out_root)
                LOG(f"run {run['label']}: metrics={jdump(rec['metrics'])}")
            except Exception as exc:  # noqa: BLE001 -- 記録して次の本へ
                all_ok = False
                rec["status"] = "failed" if rec["status"] == "running" else rec["status"] + "+measure_failed"
                rec["error"] = repr(exc)
                rec["traceback"] = traceback.format_exc()
                LOG(f"run {run['label']} failed: {exc!r}\n{rec['traceback']}")
            finally:
                OH.OUTPAINT_INJECT_CLEAN = 1.0
                rec["wall_seconds"] = round(time.perf_counter() - wall, 2)
                rec["finished_at"] = now()
                P._free_cuda()
                LOG.close()
                save()
    finally:
        try:
            pipe.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[h3_outpaint_gate] pipe.close failed (ignored): {exc!r}", flush=True)
    return all_ok


# --------------------------------------------------------------------------- #
# まとめ
# --------------------------------------------------------------------------- #

def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_summary(out_root: Path, record: dict) -> None:
    write_json(out_root / "summary.json", record)
    lines = [f"# H3 Outpainting の GPU の門（{record.get('started_at')}）", ""]
    c = record.get("conditions", {})
    lines += [f"- 条件: {c.get('lora_file')}（強度 {c.get('lora_strength')}）・{c.get('steps')} ステップ・"
              f"shift {c.get('shift_video')}/{c.get('shift_audio')}・seed {c.get('seed')}・{c.get('attention')}・"
              f"VRAM 16GB mode {c.get('vram16_mode')}・{NUM_FRAMES} フレーム",
              f"- 対照の PSNR（write_mp4 の書き直し vs 切り出し）: {_fmt((record.get('control') or {}).get('psnr_vs_cut_db'))} dB"
              f"（E1 の trim2: {_fmt((record.get('control_trim2') or {}).get('psnr_vs_cut_db'))} dB）",
              ""]
    g0 = record.get("g0")
    if g0:
        lines += ["## G0（VAE の往復）", "", f"状態: {g0.get('status')} {g0.get('error') or ''}", ""]
        if g0.get("latent_rel_l2"):
            lines += ["| 比較 | fixed | fixed_edge | fixed_inner | whole_box |", "|---|---|---|---|---|"]
            for k, d in g0["latent_rel_l2"].items():
                lines.append(f"| {k} | {_fmt(d.get('fixed'))} | {_fmt(d.get('fixed_edge'))} | "
                             f"{_fmt(d.get('fixed_inner'))} | {_fmt(d.get('whole_box'))} |")
            lines.append("")
        if g0.get("psnr_vs_cut"):
            lines += ["| 符号化 | 保持矩形 dB | 縁 32px dB | 内部 dB | 縁−内部 dB |", "|---|---|---|---|---|"]
            for k, d in g0["psnr_vs_cut"].items():
                lines.append(f"| {k} | {_fmt(d['keep_db'])} | {_fmt(d['edge32_db'])} | {_fmt(d['interior_db'])} | "
                             f"{_fmt(d['edge_minus_interior_db'])} |")
            lines.append("")
    runs = record.get("runs") or []
    if runs:
        lines += ["## G1〜G4・E1（生成）", "",
                  "| 本 | 門 | キャンバス | pad L/R/T/B | なじませ px | s | 注入 | 固定トークン | 状態 | 保持 PSNR 対照 dB | "
                  "芯 PSNR 対照 dB | 勾配比 縦 | 勾配比 横 | audio_output | inject_clean 記録 | 秒 | VRAM 山 MiB |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in runs:
            m = r.get("metrics") or {}
            bg = m.get("boundary_gradient_ratio") or {}
            op = r.get("outpaint_record") or {}
            p = r["pads"]
            lines.append(
                f"| {r['label']} | {r['gate']} | {r['canvas'][0]}x{r['canvas'][1]} | "
                f"{p['pad_left']}/{p['pad_right']}/{p['pad_top']}/{p['pad_bottom']} | {r['feather_px']} | "
                f"{r['audio_strength']} | {r['inject_clean']} | {(r.get('geometry') or {}).get('fixed_tokens')} | "
                f"{r.get('status')} | {_fmt(m.get('keep_psnr_vs_control_db'))} | "
                f"{_fmt(m.get('keep_core_psnr_vs_control_db'))} | {_fmt(bg.get('vertical_max'))} | "
                f"{_fmt(bg.get('horizontal_max'))} | {_fmt(op.get('audio_output'))} | {_fmt(op.get('inject_clean'))} | "
                f"{_fmt(r.get('seconds'))} | {_fmt(r.get('peak_vram_mb'))} |")
        lines.append("")
        failed = [r for r in runs if r.get("error")]
        for r in failed:
            lines.append(f"- {r['label']}: {r.get('error')}")
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", default=",".join(DEFAULT_GATES),
                    help="回す門（カンマ区切り。既定は G0〜G4。E1 は明示したときだけ）")
    ap.add_argument("--prepare", action="store_true", help="素材と対照を作って終わる（CPU）")
    ap.add_argument("--dry-run", action="store_true", help="モデルを読まずに幾何と要求の一覧を出す（CPU）")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int, default=TURBO_STEPS)
    ap.add_argument("--attention", choices=("sdpa", "sage"), default=None,
                    help="注意の実装（既定: E2 は sage、G0〜G4・E1 は sdpa）")
    ap.add_argument("--no-lora", action="store_true", help="Turbo LoRA を付けない（--steps も合わせて変えること）")
    ap.add_argument("--vram16-off", action="store_true", help="VRAM 16GB mode を切る（既定は on＝アプリの H3 の既定）")
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    e2 = ap.add_argument_group("E2（汎用の実験）")
    e2.add_argument("--material", default=None, help="元の動画（E2 で必須）")
    e2.add_argument("--pads", default=None, help="pad の 左,右,上,下（E2 で必須）")
    e2.add_argument("--canvas", default=None, help="キャンバスの 幅,高さ（省略なら素材＋pad。指定すれば一致を確かめる）")
    e2.add_argument("--feathers", default="1,0", help="なじませの段数 r の列（既定 1,0。1 つにつき 1 本）")
    e2.add_argument("--label", default="E2", help="本の名前の接頭辞（既定 E2。本は <label>_r<r>）")
    e2.add_argument("--frames", type=int, default=NUM_FRAMES, help="フレーム数（17n+5。既定 124）")
    e2.add_argument("--start-sec", type=float, default=0.0, help="窓の開始秒（開始フレームは round(秒x24)）")
    e2.add_argument("--scale", default=None, help="W:H。縦横比を保って覆うまで lanczos で縮め、中央で切り抜く（偶数）")
    e2.add_argument("--audio-strength", type=float, default=0.5, help="音声の Strength s（既定 0.5。音声の無い素材は 0）")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    gates = [s.strip().upper() for s in args.gate.split(",") if s.strip()]
    bad = [s for s in gates if s not in GATES]
    if bad:
        print(f"未知の門: {bad}（{GATES} のどれか）", file=sys.stderr)
        return 2
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "script": rel(Path(__file__)), "started_at": now(), "args": vars(args), "gates": gates,
        "conditions": {
            "transformer": PATHS["transformer"], "lora_file": None if args.no_lora else TURBO_LORA_FILE,
            "lora_strength": None if args.no_lora else TURBO_LORA_STRENGTH, "steps": args.steps,
            "shift_video": SHIFT_VIDEO, "shift_audio": SHIFT_AUDIO, "seed": args.seed,
            "attention": args.attention or "本ごと（E2 は sage・他は sdpa。runs[].conditions.attention）",
            "vram16_mode": not args.vram16_off, "blocks_on_gpu": BLOCKS_ON_GPU, "block_swap_prefetch": PREFETCH,
            "deterministic": DETERMINISTIC, "keep_resident": True, "num_frames": NUM_FRAMES,
            "prompt": args.prompt, "negative_prompt": "",
        },
        "runs": [],
    }

    if args.prepare:
        prepare(out_root, record)
        if "E2" in gates:
            prepare_e2(args, out_root, record)
        print(jdump({k: v for k, v in record.items()
                     if k in ("material", "control", "material_trim2", "control_trim2") or k.startswith("e2:")}))
        return 0

    runs = [r for r in run_plan() if r["gate"] in gates]
    if args.dry_run:
        prepare(out_root, record)   # 無い素材だけ作る（既存は作り直さない）
    if "E2" in gates:
        if not args.pads:
            print("E2 には --pads が要ります", file=sys.stderr)
            return 2
        runs += e2_runs(args, prepare_e2(args, out_root, record))   # CPU（無ければ切り出しを作る）
    load_prepared(out_root, record, runs)

    if args.dry_run:
        rows = []
        if "G0" in gates:
            g0g = geometry_record({"canvas": [SRC_W + 640, SRC_H + 368], "pads": MAIN_PADS})
            print(f"G0: geometry={jdump(g0g)}  grays={G0_GRAYS}  band={G0_BAND_PX}px")
            (out_root / "G0").mkdir(parents=True, exist_ok=True)
        for run in runs:
            req = build_request(run, args, out_root)
            P.validate_request(req)   # LoRA のヘッダと元動画の有無もここで確かめる（CPU）
            geo = geometry_record(run)
            (out_root / run["label"]).mkdir(parents=True, exist_ok=True)
            rows.append({**run, "geometry": geo, "request": dataclasses.asdict(req)})
        print("| 本 | 門 | キャンバス | pad L/R/T/B | 外接矩形 | 固定/生成トークン | なじませ px | s | 注入 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            p, geo = r["pads"], r["geometry"]
            print(f"| {r['label']} | {r['gate']} | {r['canvas'][0]}x{r['canvas'][1]} | "
                  f"{p['pad_left']}/{p['pad_right']}/{p['pad_top']}/{p['pad_bottom']} | "
                  f"[{geo['aligned'][0]},{geo['aligned'][1]})-[{geo['aligned'][2]},{geo['aligned'][3]}) | "
                  f"{geo['fixed_tokens']}/{geo['generated_tokens']} | {r['feather_px']} | {r['audio_strength']} | "
                  f"{r['inject_clean']} |")
        for r in rows:
            print(f"--- {r['label']} request ---")
            print(jdump(r["request"]))
        return 0

    if not torch.cuda.is_available():
        print("CUDA が使えません（GPU の門は GPU が要ります）", file=sys.stderr)
        return 2

    # 既存の summary.json があれば読み込んで足す（今回回さない本と G0 の記録は残す。同じ本は今回のもので置き換える）。
    prev_path = out_root / "summary.json"
    if prev_path.is_file():
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        labels = {r["label"] for r in runs}
        prev_conditions = prev.get("conditions")
        for r in prev.get("runs") or []:
            if r.get("label") not in labels:
                r.setdefault("conditions", prev_conditions)
                record["runs"].append(r)
        if "G0" not in gates and prev.get("g0"):
            record["g0"] = prev["g0"]
        record["previous_invocations"] = (prev.get("previous_invocations") or []) + [
            {k: prev.get(k) for k in ("started_at", "finished_at", "gates", "exit_code", "args", "load")}]
    if "G0" not in gates and "g0" not in record and (out_root / "G0" / "result.json").is_file():
        record["g0"] = json.loads((out_root / "G0" / "result.json").read_text(encoding="utf-8"))

    def save():
        write_summary(out_root, record)

    ok = True
    try:
        if "G0" in gates:
            ok = run_g0(out_root, record) and ok
            save()
        if runs:
            ok = run_generations(runs, args, out_root, record, save) and ok
    except BaseException as exc:  # noqa: BLE001 -- 中断も記録する
        ok = False
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
        raise
    finally:
        record["finished_at"] = now()
        record["exit_code"] = 0 if ok else 1
        save()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
