r"""MiniMax H3 Inpainting のスパイク（台帳 §1-124・2026-11-11）: エンジンの venv で H3Pipeline を直接呼ぶ使い捨ての台本。

計画: ``enumerated-questing-cupcake.md``（§3 条件表・§4 手順・§5 判定・§6 成果物）。製品コードは変えない
（Outpainting のトークン単位の固定マスクに、利用者のマスク動画から作った値を、このプロセスの中だけで差し込む）。
写し元: ``Docs/Outputs-archive/h3-outpaint-2026-11-10/h3_outpaint_gate.py``（前置き・PATHS・Log・小道具・
素材の切り出し・要求・読み込みと generate の形）。

リポジトリの根から（Git Bash）:

    export PYTHONPATH="$PWD" PYTHONUTF8=1
    ./.venv-engine-h3/Scripts/python.exe outputs/h3-inpaint-spike-2026-11-11/h3_inpaint_spike.py --dry-run
    ./.venv-engine-h3/Scripts/python.exe outputs/h3-inpaint-spike-2026-11-11/h3_inpaint_spike.py
    ./.venv-engine-h3/Scripts/python.exe outputs/h3-inpaint-spike-2026-11-11/h3_inpaint_spike.py --post-only

``--dry-run`` と ``--post-only`` は GPU を隠して（CUDA_VISIBLE_DEVICES=-1）CPU だけで動く。
差し替え 2 箇所（どちらもモジュール属性への代入。try で入れ finally で戻す）:
  (a) engine_h3.outpaint_h3.outpaint_fix を包む: 元の値（幾何。下 1 行だけ 1）と Inpainting のマスクの max を返す。
  (b) engine_h3.outpaint_geometry.composite を素通しにする（貼り戻しをしない）。
終了コード: 0＝完走、1＝失敗（result.json に記録）、2＝引数・素材の誤り。
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

# torch を読む前に（製品のワーカーと同じ）。--dry-run／--post-only は GPU を隠す。
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
CPU_ONLY = "--dry-run" in sys.argv or "--post-only" in sys.argv
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from engine_h3 import denoise_mask as DM  # noqa: E402
from engine_h3 import outpaint_geometry as OG  # noqa: E402
from engine_h3 import outpaint_h3 as OH  # noqa: E402
from engine_h3 import pipeline_h3 as P  # noqa: E402
from engine_h3.constants import FRAME_RATE, audio_latents, latent_frames  # noqa: E402
from engine_h3.outpaint_geometry import compute_outpaint_geometry  # noqa: E402

# --------------------------------------------------------------------------- #
# 固定の条件（計画 §3）
# --------------------------------------------------------------------------- #

OUT_ROOT = HERE
MATERIAL_SRC = ROOT / "uploads" / "h3-inpaint-development" / "inpaint-base.mp4"
MASK_SRC = ROOT / "uploads" / "h3-inpaint-development" / "mask-768p.mp4"
MATERIAL_NAME = "src_1280x768_73f.mp4"
SRC_W, SRC_H = 1280, 768
NUM_FRAMES = 73                   # 17x4+5
START_FRAME = 0                   # アプリの窓も先頭から（pipeline_manager の cut_window_mp4(src, out, 0.0, F, fps)）
CANVAS_W, CANVAS_H = 1280, 800
PADS = {"pad_left": 0, "pad_right": 0, "pad_top": 0, "pad_bottom": 32}
AUDIO_STRENGTH = 0.5
FEATHER_PX = 0
STEPS = 20
ATTENTION = "sage"
VRAM16 = True

#: 門の台本と同じ（FL2VA）。
PATHS = {
    "transformer": "models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoder": "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "video_vae": "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
    "tokenizer_dir": "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
}
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 3.0
BLOCKS_ON_GPU = 8
PREFETCH = True
DETERMINISTIC = False

DEFAULT_SEED = 1234
DEFAULT_PROMPT = (
    "Four young girls walking toward the camera on a sunny beach. The girl on the far right has blonde hair "
    "and wears a red dress, red outfit. Steady camera."
)

# マスクの規則（計画 §3。scratchpad の試作 make_token_mask.py と同じ）
TOKEN = 32
MASK_RED_MIN = 128
CHUNK = 17                        # 映像 VAE の時間方向の塊（17 枚）
LATENTS_PER_CHUNK = 5
DROP_TAIL_LATENTS = 3             # 73 枚は 85 枚に埋めて 5 塊・25 潜在、末尾 3 を捨てて 22
GRID_W, GRID_H = SRC_W // TOKEN, SRC_H // TOKEN          # 40 x 24（元の領域）
CANVAS_GRID_H = CANVAS_H // TOKEN                         # 25
SHEET_FRAMES = (0, 36, 72)
SHEET_TILE_LONG = 640

OUT_NAME = "output.mp4"
CROP_NAME = "output_crop768.mp4"
HF_DIR = "hf-round14"
HF_OUT = "inpaint_spike_1280x800_20steps_seed1234.mp4"
HF_CROP = "inpaint_spike_crop1280x768_20steps_seed1234.mp4"
HF_SRC = "source_1280x768_73f.mp4"


# --------------------------------------------------------------------------- #
# 小物（門の台本の写し）
# --------------------------------------------------------------------------- #

class Log:
    """標準出力と、ログファイルへ同じ行を書く。"""

    def __init__(self) -> None:
        self.file = None

    def __call__(self, msg: str) -> None:
        line = f"[h3_inpaint_spike] {msg}"
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


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def ffprobe(path: Path) -> dict:
    """映像（枚数は -count_frames の実測）と音声の要点。"""
    exe = tool("ffprobe")
    v = run_cmd(
        [exe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,duration",
         "-of", "json", str(path)])
    a = run_cmd(
        [exe, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels,duration", "-of", "json", str(path)])
    vs = (json.loads(v.stdout or "{}").get("streams") or [{}])[0]
    as_ = (json.loads(a.stdout or "{}").get("streams") or [None])[0]
    nb = vs.get("nb_read_frames")
    return {"path": rel(path), "video": {**vs, "nb_read_frames": int(nb) if nb not in (None, "N/A") else None},
            "audio": as_}


def read_frames(path: Path, n: int) -> np.ndarray:
    from engine_h3.source_tail import read_source_frames  # noqa: PLC0415

    return read_source_frames(str(path), n)


# --------------------------------------------------------------------------- #
# 素材の切り出し（門の台本の _prepare_main を写し、scale を外す）
# --------------------------------------------------------------------------- #

def prepare_material(out_root: Path) -> dict:
    """inpaint-base.mp4 の先頭 73 枚を 1280x768・24fps・音声つきで切り出す（CPU）。既にあれば作り直さない。"""
    mat_dir = out_root / "_material"
    pj = mat_dir / "prepare.json"
    out = mat_dir / MATERIAL_NAME
    if pj.is_file() and out.is_file():
        LOG(f"prepare: 既存の {MATERIAL_NAME} を使う（作り直さない）")
        return json.loads(pj.read_text(encoding="utf-8"))
    mat_dir.mkdir(parents=True, exist_ok=True)
    src = MATERIAL_SRC
    if not src.is_file():
        raise SystemExit(f"元の素材がありません: {src}")
    start = START_FRAME
    end = start + NUM_FRAMES - 1
    # services/video_io.py の cut_window_mp4 と同じ作法（select の窓・atrim・crf 12・yuv420p・CFR・
    # 音声は PCM、断られたら AAC 192k）。素材は 1280x768・24fps なので縮小も fps の変換もしない。
    fc = (f"[0:v]select='between(n\\,{start}\\,{end})',setpts=PTS-STARTPTS[v];"
          f"[0:a]atrim=start={start / FRAME_RATE}:end={(end + 1) / FRAME_RATE},asetpts=PTS-STARTPTS[a]")

    def cut(audio_args):
        cmd = [tool("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-filter_complex", fc,
               "-map", "[v]", "-map", "[a]", *audio_args,
               "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", str(FRAME_RATE),
               str(out)]
        return cmd, run_cmd(cmd)

    cmd, proc = cut(["-c:a", "pcm_s16le"])
    audio_codec = "pcm_s16le"
    first_error = None
    if proc.returncode != 0:
        first_error = proc.stderr[-2000:]
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
    if not probe["audio"]:
        raise SystemExit(f"切り出しに音声がありません: {rel(out)}")
    frames, wave, info = OH.read_outpaint_source(str(out), NUM_FRAMES, FRAME_RATE, AUDIO_STRENGTH, log=LOG)
    LOG(f"prepare: read_outpaint_source(s={AUDIO_STRENGTH}) frames={list(frames.shape)} "
        f"wave={None if wave is None else list(wave.shape)} info={info}")
    if wave is None:
        raise SystemExit("read_outpaint_source が音声を返しませんでした（s=0.5 では音声が要る）")
    rec = {
        "source": rel(src), "source_probe": ffprobe(src), "cut": rel(out), "ffmpeg_cmd": cmd,
        "audio_codec": audio_codec, "pcm_error": first_error, "start_frame": start, "end_frame": end,
        "probe": probe, "read_outpaint_source_s05": info, "frames_shape": list(frames.shape),
        "wave_shape_s05": list(wave.shape), "made_at": now(),
    }
    write_json(pj, rec)
    return rec


# --------------------------------------------------------------------------- #
# マスクの読み込みと格子化（新規。規則は試作 make_token_mask.py と同じ）
# --------------------------------------------------------------------------- #

def read_rgb_head(path: Path, n: int, w: int, h: int) -> np.ndarray:
    """ffmpeg で先頭 n 枚を rawvideo rgb24 に流して読む -> uint8 [n, h, w, 3]（ちょうど n 枚を要求する）。"""
    cmd = [tool("ffmpeg"), "-v", "error", "-i", str(path), "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = p.stderr.decode("utf-8", errors="replace")
    if p.returncode != 0:
        raise SystemExit(f"ffmpeg でマスクを読めませんでした（{p.returncode}）: {err[-2000:]}")
    if err.strip():
        LOG(f"mask: ffmpeg stderr: {err.strip()[-500:]}")
    a = np.frombuffer(p.stdout, np.uint8)
    per = w * h * 3
    if a.size != n * per:
        raise SystemExit(f"マスクの読み込みの量が合いません（{a.size} バイト。{n} 枚 x {w}x{h}x3 = {n * per} が必要）")
    return a.reshape(n, h, w, 3)


def build_token_mask(out_root: Path) -> dict:
    """mask-768p.mp4 -> token_mask_73.npy（float32 [22, 24, 40]）と、下に全 1 の 1 行を足した [22, 25, 40]。"""
    probe = ffprobe(MASK_SRC)
    vid = probe["video"]
    if (vid.get("width"), vid.get("height")) != (SRC_W, SRC_H):
        raise SystemExit(f"マスクの寸法が {SRC_W}x{SRC_H} ではありません（{vid.get('width')}x{vid.get('height')}）")
    if (vid.get("nb_read_frames") or 0) < NUM_FRAMES:
        raise SystemExit(f"マスクの枚数が {NUM_FRAMES} 未満です（{vid.get('nb_read_frames')}）")
    m = read_rgb_head(MASK_SRC, NUM_FRAMES, SRC_W, SRC_H)
    pix = m[..., 0] >= MASK_RED_MIN                                                  # [73, 768, 1280]
    blk = pix.reshape(NUM_FRAMES, GRID_H, TOKEN, GRID_W, TOKEN).any(axis=(2, 4))     # [73, 24, 40]
    n_chunks = -(-NUM_FRAMES // CHUNK)                                               # 5
    chunks = [(CHUNK * c, min(CHUNK * c + CHUNK, NUM_FRAMES)) for c in range(n_chunks)]
    chunk_mask = np.stack([blk[a:b].any(axis=0) for a, b in chunks])                 # [5, 24, 40]
    t_lat = latent_frames(NUM_FRAMES)
    full = np.repeat(chunk_mask, LATENTS_PER_CHUNK, axis=0)                          # [25, 24, 40]
    if full.shape[0] - DROP_TAIL_LATENTS != t_lat:
        raise SystemExit(f"潜在の数が合いません（{full.shape[0]} - {DROP_TAIL_LATENTS} != {t_lat}）")
    lat = full[:t_lat].astype(np.float32)                                            # [22, 24, 40]
    np.save(out_root / "token_mask_73.npy", lat)
    padded = np.concatenate([lat, np.ones((t_lat, 1, GRID_W), np.float32)], axis=1)  # [22, 25, 40]
    return {
        "pixel": pix, "blocks": blk, "chunk_mask": chunk_mask, "token_mask": lat,
        "padded": torch.from_numpy(np.ascontiguousarray(padded)),
        "record": {
            "source": rel(MASK_SRC), "probe": probe, "frames_read": int(m.shape[0]),
            "rule": (f"赤 >= {MASK_RED_MIN} を白・{TOKEN}x{TOKEN} のブロックに 1 画素でも白なら 1・"
                     f"時間は {CHUNK} 枚の塊ごとの和集合・潜在は塊ごとに {LATENTS_PER_CHUNK} 個で末尾 "
                     f"{DROP_TAIL_LATENTS} 個を捨てる"),
            "chunks": [list(c) for c in chunks],
            "token_mask_shape": list(lat.shape), "padded_shape": list(padded.shape),
            "chunk_token_counts": chunk_mask.reshape(n_chunks, -1).sum(1).astype(int).tolist(),
            "per_latent_mask_tokens": lat.reshape(t_lat, -1).sum(1).astype(int).tolist(),
            "pixel_white_ratio_f": {str(f): round(float(pix[f].mean()), 5) for f in SHEET_FRAMES},
            "saved": rel(out_root / "token_mask_73.npy"),
        },
    }


def chunk_of(frame: int) -> int:
    return min(frame // CHUNK, -(-NUM_FRAMES // CHUNK) - 1)


# --------------------------------------------------------------------------- #
# 差し替え (a)・(b)（新規）
# --------------------------------------------------------------------------- #

_ORIG_FIX = OH.outpaint_fix
_ORIG_COMPOSITE = OG.composite
SPIKE: dict = {"fix_calls": 0, "composite_calls": 0, "fix_record": None}


def reset_counts() -> None:
    SPIKE.update({"fix_calls": 0, "composite_calls": 0, "fix_record": None})


def make_fix_wrapper(mask_padded: torch.Tensor):
    expect = tuple(int(s) for s in mask_padded.shape)

    def spike_outpaint_fix(clean_bbox, clean_a, geom, s, t_lat, a_lat):
        video_fix, audio_fix = _ORIG_FIX(clean_bbox, clean_a, geom, s, t_lat, a_lat)
        clean_v, values = video_fix
        if tuple(int(x) for x in values.shape) != expect:
            raise RuntimeError(f"spike: outpaint_fix の値の形 {list(values.shape)} が {list(expect)} ではありません")
        new = torch.maximum(values, mask_padded.to(device=values.device, dtype=values.dtype)).contiguous()
        gen_per = (new == 1).reshape(new.shape[0], -1).sum(1).to(torch.int64).tolist()
        fixed = int((new == 0).sum())
        SPIKE["fix_calls"] += 1
        SPIKE["fix_record"] = {
            "values_shape": list(new.shape), "geometry_generated_tokens_total": int((values == 1).sum()),
            "fixed_tokens_total": fixed, "generated_tokens_total": int((new == 1).sum()),
            "other_values": int(((new != 0) & (new != 1)).sum()),
            "generated_per_latent_frame": [int(x) for x in gen_per],
            "fixed_per_latent_frame": [int(new.shape[1] * new.shape[2] - x) for x in gen_per],
            "clean_v_shape": list(clean_v.shape),
            "audio_fix": None if audio_fix is None else {
                "clean_shape": list(audio_fix[0].shape),
                "values_unique": sorted({round(float(x), 6) for x in audio_fix[1].unique().tolist()}),
                "values_count_lt1": int((audio_fix[1] < 1).sum())},
        }
        return (clean_v, new), audio_fix

    return spike_outpaint_fix


def spike_composite(decoded, source, weights, geom):
    """貼り戻しをしない: decoded をそのまま返す（回数だけ数える）。"""
    SPIKE["composite_calls"] += 1
    return decoded


def install(mask_padded: torch.Tensor) -> None:
    OH.outpaint_fix = make_fix_wrapper(mask_padded)
    OG.composite = spike_composite


def uninstall() -> None:
    OH.outpaint_fix = _ORIG_FIX
    OG.composite = _ORIG_COMPOSITE


# --------------------------------------------------------------------------- #
# 要求と幾何（門の台本の build_request／geometry_record を変えたもの）
# --------------------------------------------------------------------------- #

def build_request(args, out_root: Path) -> P.H3GenerateRequest:
    return P.H3GenerateRequest(
        prompt=args.prompt, seed=DEFAULT_SEED, width=CANVAS_W, height=CANVAS_H, num_frames=NUM_FRAMES,
        output_path=str(out_root / OUT_NAME),
        num_inference_steps=STEPS, shift_video=SHIFT_VIDEO, shift_audio=SHIFT_AUDIO,
        keep_resident=True, attention_backend=ATTENTION, block_swap_prefetch=PREFETCH,
        vram16_mode=VRAM16, loras=(),
        outpaint=P.H3OutpaintInput(
            path=str(out_root / "_material" / MATERIAL_NAME),
            audio_strength=AUDIO_STRENGTH, feather_px=FEATHER_PX, **PADS,
        ),
    )


def geometry_record() -> dict:
    g = compute_outpaint_geometry(SRC_W, SRC_H, CANVAS_W, CANVAS_H, **PADS)
    return {"canvas": [CANVAS_W, CANVAS_H], "source": [SRC_W, SRC_H], "aligned": [g.x0, g.y0, g.x1, g.y1],
            "aligned_size": [g.aligned_w, g.aligned_h], "grid": [g.grid_w, g.grid_h],
            "fixed_tokens": g.fixed_count, "generated_tokens": g.generated_count,
            "latent_offset_yx": list(g.latent_offset), "source_offset_in_aligned_yx": list(g.source_offset_in_aligned),
            "latent_frames": latent_frames(NUM_FRAMES)}


def conditions(args) -> dict:
    return {
        "prompt": args.prompt, "negative_prompt": "", "seed": DEFAULT_SEED, "steps": STEPS, "lora": None,
        "shift_video": SHIFT_VIDEO, "shift_audio": SHIFT_AUDIO, "attention": ATTENTION, "vram16_mode": VRAM16,
        "blocks_on_gpu": BLOCKS_ON_GPU, "block_swap_prefetch": PREFETCH, "deterministic": DETERMINISTIC,
        "keep_resident": True, "num_frames": NUM_FRAMES, "canvas": [CANVAS_W, CANVAS_H], "pads": dict(PADS),
        "feather_px": FEATHER_PX, "audio_strength": AUDIO_STRENGTH, "inject_clean": float(OH.OUTPAINT_INJECT_CLEAN),
        "paste_back": "しない（engine_h3.outpaint_geometry.composite を素通しに差し替え）",
        "token_values": "max(幾何の値, Inpainting マスク) -> [22, 25, 40]",
        "transformer": PATHS["transformer"],
    }


# --------------------------------------------------------------------------- #
# --dry-run（計画 §4.1）
# --------------------------------------------------------------------------- #

def dry_run(args, out_root: Path) -> int:
    rec: dict = {"script": rel(Path(__file__)), "started_at": now(), "conditions": conditions(args), "checks": {}}
    checks = rec["checks"]
    ok = True

    def check(name: str, cond: bool, detail) -> None:
        nonlocal ok
        checks[name] = {"passed": bool(cond), "detail": detail}
        ok = ok and bool(cond)
        LOG(f"dry-run: {name}: {'ok' if cond else 'NG'} {jdump(detail) if not isinstance(detail, str) else detail}")

    try:
        mat = prepare_material(out_root)
        pv, pa = mat["probe"]["video"], mat["probe"]["audio"]
        check("material", pv.get("nb_read_frames") == NUM_FRAMES and (pv.get("width"), pv.get("height")) == (SRC_W, SRC_H)
              and bool(pa), {"frames": pv.get("nb_read_frames"), "size": [pv.get("width"), pv.get("height")],
                             "audio": pa, "audio_codec": mat.get("audio_codec"),
                             "read_outpaint_source_s05": mat.get("read_outpaint_source_s05"),
                             "wave_shape_s05": mat.get("wave_shape_s05")})
        mk = build_token_mask(out_root)
        mrec = mk["record"]
        check("mask", mrec["token_mask_shape"] == [22, 24, 40] and mrec["padded_shape"] == [22, 25, 40]
              and bool((mk["padded"][:, -1, :] == 1).all()), mrec)
        rec["mask"] = mrec
        geo = geometry_record()
        check("geometry", geo["grid"] == [40, 25] and geo["fixed_tokens"] == 960, geo)
        t_lat = latent_frames(NUM_FRAMES)
        a_lat = audio_latents(NUM_FRAMES)
        check("latent_frames", t_lat == 22, {"latent_frames": t_lat, "audio_latents": a_lat})

        g = compute_outpaint_geometry(SRC_W, SRC_H, CANVAS_W, CANVAS_H, **PADS)
        te_a = int(mat["wave_shape_s05"][-1]) // 800 if mat.get("wave_shape_s05") else a_lat - 1
        te_a = min(te_a, a_lat)
        clean_bbox = torch.zeros(1, 24, t_lat, g.aligned_h // 16, g.aligned_w // 16)
        clean_a = torch.zeros(2, 32, te_a)
        reset_counts()
        install(mk["padded"])
        try:
            reached = {"OH.outpaint_fix_is_wrapper": OH.outpaint_fix is not _ORIG_FIX}
            from engine_h3.outpaint_geometry import composite as late_composite  # noqa: PLC0415 -- generate と同じ読み方
            reached["late_import_composite_is_stub"] = late_composite is spike_composite
            video_fix, audio_fix = OH.outpaint_fix(clean_bbox, clean_a, g, AUDIO_STRENGTH, t_lat, a_lat)
            dummy = np.zeros((NUM_FRAMES, CANVAS_H, CANVAS_W, 3), np.uint8)
            late_composite(dummy, np.zeros((NUM_FRAMES, SRC_H, SRC_W, 3), np.uint8),
                           OG.feather_weights(g, FEATHER_PX), g)
        finally:
            uninstall()
        reached["restored"] = OH.outpaint_fix is _ORIG_FIX and OG.composite is _ORIG_COMPOSITE
        fr = SPIKE["fix_record"]
        expect_gen = [int(x) + GRID_W for x in mrec["per_latent_mask_tokens"]]
        check("wrapper", all(reached.values()) and SPIKE["fix_calls"] == 1 and SPIKE["composite_calls"] == 1
              and fr["values_shape"] == [22, 25, 40] and fr["other_values"] == 0
              and fr["generated_per_latent_frame"] == expect_gen
              and list(video_fix[0].shape) == [1, 24, t_lat, CANVAS_H // 16, CANVAS_W // 16],
              {"reached": reached, "fix_calls": SPIKE["fix_calls"], "composite_calls": SPIKE["composite_calls"],
               "clean_bbox_shape": list(clean_bbox.shape), "clean_a_shape": list(clean_a.shape),
               "expected_generated_per_latent_frame（マスク+下 1 行 40）": expect_gen, "fix_record": fr})
        rec["spike_counts_per_latent"] = fr["generated_per_latent_frame"]

        clean_v, values = video_fix
        noise = torch.zeros_like(clean_v)
        sm = DM.video_mask(clean_v, values, grid=g.grid_w * g.grid_h, noise=noise,
                           inject_clean=float(OH.OUTPAINT_INJECT_CLEAN))
        check("video_mask", sm is not None and list(sm.m.shape) == [1, 1, t_lat, CANVAS_H // 16, CANVAS_W // 16]
              and int(sm.rows.numel()) == t_lat * g.grid_w * g.grid_h
              and int(sm.zero.sum()) == fr["fixed_tokens_total"] * 4,
              None if sm is None else {"m_shape": list(sm.m.shape), "rows": int(sm.rows.numel()),
                                       "zero_cells": int(sm.zero.sum()), "fixed_tokens_x4": fr["fixed_tokens_total"] * 4})
        audio_out = torch.zeros(2, 32, a_lat)
        try:
            OH.check_outpaint_fixed(clean_v, video_fix, audio_out, audio_fix, AUDIO_STRENGTH, 0)
            cf = "passed"
        except Exception as exc:  # noqa: BLE001
            cf = repr(exc)
        check("check_outpaint_fixed", cf == "passed", cf)

        req = build_request(args, out_root)
        P.validate_request(req)
        rec["request"] = dataclasses.asdict(req)
        check("validate_request", True, "passed")
    except BaseException as exc:  # noqa: BLE001
        ok = False
        rec["error"] = repr(exc)
        rec["traceback"] = traceback.format_exc()
        LOG(f"dry-run failed: {exc!r}\n{rec['traceback']}")
    rec["all_passed"] = ok
    rec["finished_at"] = now()
    write_json(out_root / "dry_run.json", rec)
    LOG(f"dry-run: all_passed={ok} -> {rel(out_root / 'dry_run.json')}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# 本番（門の台本の run_generations を 1 本に写したもの）
# --------------------------------------------------------------------------- #

def run_generation(args, out_root: Path) -> int:
    from engine_h3.pipeline_h3 import H3LoadOptions, H3Paths, H3Pipeline  # noqa: PLC0415

    LOG.open(out_root / "run.log")
    rec: dict = {"script": rel(Path(__file__)), "started_at": now(), "conditions": conditions(args),
                 "geometry": geometry_record(), "status": "running", "spike_valid": False}
    result_path = out_root / "result.json"

    def save():
        write_json(result_path, rec)

    ok = False
    try:
        rec["material"] = prepare_material(out_root)
        mk = build_token_mask(out_root)
        rec["mask"] = mk["record"]
        req = build_request(args, out_root)
        P.validate_request(req)
        rec["request"] = dataclasses.asdict(req)
        save()
        t0 = time.perf_counter()
        options = H3LoadOptions(blocks_on_gpu=BLOCKS_ON_GPU, attention_backend=ATTENTION,
                                block_swap_prefetch=PREFETCH, deterministic=DETERMINISTIC)
        paths = H3Paths(**{k: str(ROOT / v) for k, v in PATHS.items()})
        pipe = H3Pipeline(paths, options, device=torch.device("cuda"), progress=LOG.progress, log=LOG)
        rec["load"] = {"build_report": pipe.load(), "seconds": round(time.perf_counter() - t0, 2)}
        save()
        try:
            wall = time.perf_counter()
            install(mk["padded"])
            try:
                reset_counts()
                LOG(f"run: inject_clean={OH.OUTPAINT_INJECT_CLEAN} request={jdump(rec['request'])}")
                try:
                    result = pipe.generate(req)
                    rec["self_check"] = "passed"
                except Exception as exc:  # noqa: BLE001
                    rec["generate_error"] = repr(exc)
                    msg = str(exc)
                    if "自己検査" in msg or "Outpainting: 固定したトークン" in msg:
                        rec["self_check"] = repr(exc)
                    else:
                        rec["self_check"] = "not_reached"
                    raise
            finally:
                uninstall()
                rec["spike_counts"] = {"outpaint_fix_calls": SPIKE["fix_calls"],
                                       "composite_calls": SPIKE["composite_calls"]}
                rec["tokens_actual"] = SPIKE["fix_record"]
                fr_ = SPIKE.get("fix_record")
                expect_gen_ = [int(x) + GRID_W for x in mk["record"]["per_latent_mask_tokens"]]
                rec["spike_valid"] = bool(
                    SPIKE["fix_calls"] == 1 and SPIKE["composite_calls"] == 1
                    and fr_ is not None and fr_.get("other_values") == 0
                    and fr_.get("generated_per_latent_frame") == expect_gen_)
                rec["wall_seconds"] = round(time.perf_counter() - wall, 2)
            rec["engine"] = result
            h3 = result.get("h3", {}) if isinstance(result, dict) else {}
            rec["tokens_engine_record"] = {k: (h3.get("outpaint") or {}).get(k)
                                           for k in ("fixed_tokens", "generated_tokens", "inject_clean")}
            rec["seconds"] = result.get("seconds")
            rec["peak_vram_mb"] = result.get("peak_vram_mb")
            rec["peak_vram_reserved_mb"] = result.get("peak_vram_reserved_mb")
            rec["status"] = "done"
            ok = bool(rec["spike_valid"])
            LOG(f"run: done in {rec['seconds']} s, peak {rec['peak_vram_mb']} MiB, spike_valid={rec['spike_valid']} "
                f"counts={rec['spike_counts']}")
        finally:
            try:
                pipe.close()
            except Exception as exc:  # noqa: BLE001
                LOG(f"pipe.close failed (ignored): {exc!r}")
    except BaseException as exc:  # noqa: BLE001 -- 中断も記録する
        rec["status"] = "failed"
        rec["error"] = repr(exc)
        rec["traceback"] = traceback.format_exc()
        LOG(f"run failed: {exc!r}\n{rec['traceback']}")
        ok = False
    finally:
        rec["finished_at"] = now()
        save()
    if rec.get("status") == "done":
        try:
            post_process(out_root)
        except BaseException as exc:  # noqa: BLE001 -- 後処理の SystemExit も記録する
            rec["post_error"] = repr(exc)
            save()
            LOG(f"post failed (generation itself is recorded): {exc!r}\n{traceback.format_exc()}")
            ok = False
    LOG.close()
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# 後処理（新規。GPU 不要）
# --------------------------------------------------------------------------- #

def mask_pixels(chunk_mask_c: np.ndarray) -> np.ndarray:
    """[24, 40] -> bool [768, 1280]。"""
    return np.repeat(np.repeat(chunk_mask_c, TOKEN, 0), TOKEN, 1)


def dilate1(g: np.ndarray) -> np.ndarray:
    """トークン格子で 1 個ぶん膨らませる（8 近傍）。"""
    p = np.pad(g, 1)
    out = np.zeros_like(g)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= p[dy:dy + g.shape[0], dx:dx + g.shape[1]]
    return out


def token_overlay(img_rgb: np.ndarray, grid_mask: np.ndarray):
    """試作と同じ描き方: マスクのブロックを赤 (255,0,0,100)、32 画素の格子線を白 (255,255,255,70)。"""
    from PIL import Image  # noqa: PLC0415

    h, w = img_rgb.shape[:2]
    ov = np.zeros((h, w, 4), np.uint8)
    ov[mask_pixels(grid_mask)] = (255, 0, 0, 100)
    for r in range(h // TOKEN + 1):
        ov[min(r * TOKEN, h - 1), :] = (255, 255, 255, 70)
    for k in range(w // TOKEN + 1):
        ov[:, min(k * TOKEN, w - 1)] = (255, 255, 255, 70)
    base = Image.fromarray(img_rgb).convert("RGBA")
    return Image.alpha_composite(base, Image.fromarray(ov, "RGBA")).convert("RGB")


def red_overlay(img_rgb: np.ndarray, grid_mask: np.ndarray):
    from PIL import Image  # noqa: PLC0415

    ov = np.zeros(img_rgb.shape[:2] + (4,), np.uint8)
    ov[mask_pixels(grid_mask)] = (255, 0, 0, 100)
    base = Image.fromarray(img_rgb).convert("RGBA")
    return Image.alpha_composite(base, Image.fromarray(ov, "RGBA")).convert("RGB")


def make_sheet(src: np.ndarray, out768: np.ndarray, chunk_mask: np.ndarray, path: Path) -> None:
    """フレーム 0・36・72 x 「元」「出力の上 768 行」「出力にマスクを赤」。各コマを長辺 640 に縮める。"""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    scale = SHEET_TILE_LONG / max(SRC_W, SRC_H)
    tw, th = round(SRC_W * scale), round(SRC_H * scale)
    head = 22
    sheet = Image.new("RGB", (tw * 3, (th + head) * len(SHEET_FRAMES)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, f in enumerate(SHEET_FRAMES):
        g = chunk_mask[chunk_of(f)]
        tiles = [("source", Image.fromarray(src[f])), ("output top 768", Image.fromarray(out768[f])),
                 ("output + token mask (red)", red_overlay(out768[f], g))]
        y = i * (th + head)
        for j, (label, im) in enumerate(tiles):
            sheet.paste(im.resize((tw, th), Image.LANCZOS), (j * tw, y + head))
            draw.text((j * tw + 6, y + 5), f"f{f:03d} {label}", fill=(235, 235, 235))
    sheet.save(path)


def metrics(src: np.ndarray, out768: np.ndarray, chunk_mask: np.ndarray) -> dict:
    """上 768 行の、元動画との平均絶対差（RGB の 0〜255）。内＝そのフレームの塊のトークンマスク、外＝1 トークン膨らませた外。"""
    def one(f: int) -> dict:
        g = chunk_mask[chunk_of(f)]
        inside = mask_pixels(g)
        outside = ~mask_pixels(dilate1(g))
        d = np.abs(out768[f].astype(np.int16) - src[f].astype(np.int16)).astype(np.float32)
        return {"inside_mad": round(float(d[inside].mean()), 3) if inside.any() else None,
                "outside_mad": round(float(d[outside].mean()), 3) if outside.any() else None,
                "inside_px": int(inside.sum()), "outside_px": int(outside.sum())}

    per = {str(f): one(f) for f in SHEET_FRAMES}
    ins_sum = ins_n = out_sum = out_n = 0.0
    for f in range(src.shape[0]):
        g = chunk_mask[chunk_of(f)]
        inside = mask_pixels(g)
        outside = ~mask_pixels(dilate1(g))
        d = np.abs(out768[f].astype(np.int16) - src[f].astype(np.int16)).astype(np.float32).mean(axis=2)
        ins_sum += float(d[inside].sum())
        ins_n += float(inside.sum())
        out_sum += float(d[outside].sum())
        out_n += float(outside.sum())
    per["all"] = {"inside_mad": round(ins_sum / ins_n, 3) if ins_n else None,
                  "outside_mad": round(out_sum / out_n, 3) if out_n else None}
    return {"note": "上 768 行・元動画（_material の切り出し）との平均絶対差。内＝その塊のトークンマスクの画素、"
                    "外＝マスクを 1 トークン（8 近傍）膨らませた範囲の外", "frames": per}


def post_process(out_root: Path) -> dict:
    LOG("post: start")
    out_mp4 = out_root / OUT_NAME
    material = out_root / "_material" / MATERIAL_NAME
    if not out_mp4.is_file():
        raise SystemExit(f"出力がありません: {out_mp4}")
    crop = out_root / CROP_NAME
    cmd = [tool("ffmpeg"), "-y", "-v", "error", "-i", str(out_mp4), "-map", "0:v:0", "-map", "0:a:0?",
           "-vf", f"crop={SRC_W}:{SRC_H}:0:0", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
           "-c:a", "copy", str(crop)]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg の切り抜きに失敗しました（{proc.returncode}）: {proc.stderr[-2000:]}")
    mk = build_token_mask(out_root)
    chunk_mask = mk["chunk_mask"]
    src = read_frames(material, NUM_FRAMES)
    out = read_frames(out_mp4, NUM_FRAMES)
    if out.shape[1:3] != (CANVAS_H, CANVAS_W):
        raise SystemExit(f"出力の寸法が {CANVAS_W}x{CANVAS_H} ではありません（{out.shape[2]}x{out.shape[1]}）")
    out768 = np.ascontiguousarray(out[:, :SRC_H])
    make_sheet(src, out768, chunk_mask, out_root / "sheet.png")
    f36 = SHEET_FRAMES[1]
    token_overlay(src[f36], chunk_mask[chunk_of(f36)]).save(out_root / "token_overlay_f036.png")
    m = metrics(src, out768, chunk_mask)
    post = {"made_at": now(), "crop": {"path": rel(crop), "ffmpeg_cmd": cmd, "probe": ffprobe(crop)},
            "output_probe": ffprobe(out_mp4), "metrics": m,
            "sheet": rel(out_root / "sheet.png"), "token_overlay": rel(out_root / "token_overlay_f036.png")}
    LOG(f"post: metrics={jdump(m)}")
    result_path = out_root / "result.json"
    rec = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    rec["post"] = post
    write_json(result_path, rec)
    hf = out_root / HF_DIR
    hf.mkdir(parents=True, exist_ok=True)
    for a, b in ((out_mp4, HF_OUT), (crop, HF_CROP), (material, HF_SRC),
                 (out_root / "sheet.png", "sheet.png"), (out_root / "token_overlay_f036.png", "token_overlay_f036.png")):
        shutil.copy2(a, hf / b)
    write_readme(hf, rec, mk["record"])
    LOG(f"post: done -> {rel(hf)}")
    return post


def _g(v, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_readme(hf: Path, rec: dict, mrec: dict) -> None:
    c = rec.get("conditions") or {}
    tok = rec.get("tokens_actual") or {}
    eng = rec.get("tokens_engine_record") or {}
    m = ((rec.get("post") or {}).get("metrics") or {}).get("frames") or {}
    L = [
        "# round14-h3-inpaint — MiniMax H3 の Inpainting のスパイク（2026-11-11）",
        "",
        "元の動画（1280×768・73 フレーム・24fps）のうち、一番右の少女の領域だけを描き直せるかを確かめる 1 本の実験です"
        "（台帳 PENDING_TASKS.md §1-124）。Outpainting で作った「トークン（32×32 画素の升目）ごとに固定するか生成するかを決める仕組み」に、"
        "利用者のマスク動画から作った値を、台本の中だけで差し込みました。製品のコードは変えていません。",
        "",
        "見る順: (1) マスクの中に破綻の無い絵が出て、周りとつながっているか、(2) 髪の色と服がプロンプト（金髪・赤い服）に寄ったか、"
        "(3) 縁に継ぎ目や元の少女の残像が無いか。マスクの中の升目には元の映像の情報が一切入らないので、「元と違う」こと自体は仕組みで決まっています。",
        "",
        "## 条件",
        "",
        "| 欄 | 値 |",
        "|---|---|",
        f"| プロンプト | {c.get('prompt', '')} |",
        f"| 種 | {c.get('seed')} |",
        f"| ステップ | {c.get('steps')}（Turbo LoRA なし） |",
        f"| キャンバス | {CANVAS_W}×{CANVAS_H}（下に 32 画素足した。この 1 行は Outpainting として生成される） |",
        f"| 音声の強さ（Strength） | {c.get('audio_strength')}（元の音声の潜在をどれだけ残すか。製品の既定と同じ） |",
        f"| 注入係数 | {c.get('inject_clean')}（固定する升目に元の潜在をそのまま入れる） |",
        "| 貼り戻し | しない（生成後に元の画素を上書きする処理を止めた。固定した領域も映像 VAE の往復ぶん少しぼける） |",
        f"| 注意の実装（attention） | {c.get('attention')} |",
        f"| VRAM 16GB モード | {'on' if c.get('vram16_mode') else 'off'} |",
        "",
        "## マスクの作り方",
        "",
        f"- 規則: {mrec.get('rule')}。",
        "- 映像 VAE は時間方向に 17 枚を 1 塊として潜在にするので、塊の中で 1 度でもマスクにかかった升目は、その塊の間ずっと生成します。",
        f"- 塊ごとのマスクの升目の数（1 枚 960 升中）: {mrec.get('chunk_token_counts')}",
        "- `token_overlay_f036.png` はフレーム 36 の元の映像に、その塊のマスク（赤）と 32 画素の格子（白）を重ねたものです。",
        "",
        "## 数値",
        "",
        "| 項目 | 値 |",
        "|---|---|",
        f"| 妥当性（差し替えが 1 回ずつ効いたか） | {rec.get('spike_valid')}（{rec.get('spike_counts')}） |",
        f"| 固定した升目の自己検査（元の潜在とビット一致） | {rec.get('self_check')}（生成の例外: {rec.get('generate_error')}） |",
        f"| 固定・生成の升目の総数（台本が数えた実数） | 固定 {tok.get('fixed_tokens_total')}・生成 {tok.get('generated_tokens_total')} |",
        f"| 同（エンジンの記録。幾何のみの値） | 固定 {eng.get('fixed_tokens')}・生成 {eng.get('generated_tokens')}（1 潜在フレームあたり） |",
        f"| 潜在フレームごとの生成の升目（下 1 行の 40 を含む） | {tok.get('generated_per_latent_frame')} |",
        f"| 所要秒 | {_g(rec.get('seconds'), 1)} |",
        f"| VRAM の山（割り当て・予約 MiB） | {_g(rec.get('peak_vram_mb'), 0)}・{_g(rec.get('peak_vram_reserved_mb'), 0)} |",
        "",
        "元の動画との平均絶対差（上 768 行・RGB の 0〜255。内＝マスクの升目、外＝マスクを 1 升ぶん広げた範囲の外）。"
        "外の値は VAE の往復による劣化の大きさ、内の値は描き直しの大きさの目安です（合否の基準ではありません）。",
        "",
        "| フレーム | マスクの内 | マスクの外 |",
        "|---|---|---|",
    ]
    for k, v in m.items():
        name = "全フレーム" if k == "all" else k
        L.append(f"| {name} | {_g(v.get('inside_mad'))} | {_g(v.get('outside_mad'))} |")
    L += [
        "",
        "## ファイル",
        "",
        f"- `{HF_OUT}`: 出力そのもの（1280×800。下 32 画素は Outpainting で生成した帯）",
        f"- `{HF_CROP}`: 出力の上 768 行（元の動画と同じ寸法。見比べ用）",
        f"- `{HF_SRC}`: 元の動画（先頭 73 フレームの切り出し）",
        "- `sheet.png`: フレーム 0・36・72 の 3 行 × 「元」「出力の上 768 行」「出力にマスクを赤で重ねたもの」の 3 列",
        "- `token_overlay_f036.png`: マスクの升目の図（上の「マスクの作り方」）",
        "",
        "比べる基準は、同じ素材・同じ少女で今日回した LTX 2.5 の Inpainting（金髪・薄緑の服に変わった）です。",
        "",
    ]
    (hf / "README.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="モデルを読まずに素材・マスク・幾何・差し替え・要求を確かめる（CPU）")
    ap.add_argument("--post-only", action="store_true", help="既にある output.mp4 から後処理だけを行う（CPU）")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_root = OUT_ROOT
    if args.dry_run and args.post_only:
        print("--dry-run と --post-only は同時に指定できません", file=sys.stderr)
        return 2
    if args.dry_run:
        return dry_run(args, out_root)
    if args.post_only:
        try:
            post_process(out_root)
        except BaseException as exc:  # noqa: BLE001 -- 後処理の SystemExit も記録する
            LOG(f"post failed: {exc!r}\n{traceback.format_exc()}")
            return 1
        return 0
    if not torch.cuda.is_available():
        print("CUDA が使えません（本番は GPU が要ります）", file=sys.stderr)
        return 2
    return run_generation(args, out_root)


if __name__ == "__main__":
    raise SystemExit(main())
