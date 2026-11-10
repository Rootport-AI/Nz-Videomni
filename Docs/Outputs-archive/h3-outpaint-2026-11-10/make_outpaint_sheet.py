r"""MiniMax H3 の Outpainting の門（台帳 §1-125）の結果を、遠隔で目視できる形（Hugging Face に上げるフォルダ）に整える補助。

``h3_outpaint_gate.py`` で回した後に、監督が実行する（GPU もサーバーも使わない。読んで書くだけ）::

    .venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\make_outpaint_sheet.py --hf-dir outputs\h3-outpaint-gates-2026-11-10\hf-round13

読むもの: ``<out-root>\summary.json``（``runs[]`` と ``conditions.seed``。無ければ各本の ``result.json`` と台本と同じ既定の条件）・
``<out-root>\<本>\output.mp4``・``<out-root>\G0\<名前>.mp4``・``<out-root>\G0\result.json``・``<out-root>\_material\src_1280x720_124f.mp4``。

書くもの（``--hf-dir``）: 各本の mp4（``<本>_<幅>x<高さ>_r<なじませ段数>_s<音声Strength>_seed<seed>.mp4``）・``source_1280x720_124f.mp4``・
``sheet_all.png``・``sheet_<本>.png``・G0 の mp4 と ``sheet_G0.png``・``README.md``。何度でも回してよい（既存のファイルは消さない）。
動画の無い本は飛ばし、README の表に「未実行」と書く。画像は ``tools\ffmpeg\bin\ffmpeg.exe`` で 3 コマ（先頭・中央・末尾）を切り出し、
Pillow で縮めて並べる（保持矩形の枠を 1 画素の線で描く）。

``--no-sheets`` は画像を作らない。``--out-root`` は入力の根の差し替え（試走用）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FFMPEG = REPO / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
DEFAULT_ROOT = REPO / "outputs" / "h3-outpaint-gates-2026-11-10"

SRC_W, SRC_H, NUM_FRAMES = 1280, 720, 124
FRAME_IDX = (0, NUM_FRAMES // 2, NUM_FRAMES - 1)
LONG_SIDE = 640
GAP = 8
LABEL_W = 250
HEADER_H = 40
FRAME_COLOR = (255, 0, 255)
DEFAULT_SEED = 1234
SOURCE_NAME = "src_1280x720_124f.mp4"

MAIN = {"pad_left": 320, "pad_right": 320, "pad_top": 184, "pad_bottom": 184}
ODD = {"pad_left": 13, "pad_right": 19, "pad_top": 8, "pad_bottom": 8}


def _plan(label, pads, r=1, s=0.5, inject=1.0):
    cw = SRC_W + pads["pad_left"] + pads["pad_right"]
    ch = SRC_H + pads["pad_top"] + pads["pad_bottom"]
    feather = int(((r * max(cw, ch)) / 64.0) + 0.5)
    return {"label": label, "pads": dict(pads), "canvas": [cw, ch], "r": r, "feather_px": feather,
            "audio_strength": s, "inject_clean": inject}


# 台本 h3_outpaint_gate.py の run_plan と同じ条件（summary.json の runs[] が無いときの代わり）
PLAN = [_plan("G1_main", MAIN), _plan("G1_odd", ODD), _plan("G2_r0", MAIN, r=0), _plan("G2_r3", MAIN, r=3),
        _plan("G3_inject0999", MAIN, inject=0.999), _plan("G4_s1", MAIN, s=1.0)]
G0_NAMES = ["single", "canvas_imagenet", "canvas_gray128", "control_source"]


def load_json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def g(x, digits=2) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def collect(root: Path, summary: dict) -> list[dict]:
    sruns = {r.get("label"): r for r in (summary.get("runs") or []) if isinstance(r, dict)}
    seed = (summary.get("conditions") or {}).get("seed", (summary.get("args") or {}).get("seed", DEFAULT_SEED))
    rows = []
    for base in PLAN:
        label = base["label"]
        sr = sruns.get(label) or {}
        res = load_json(root / label / "result.json")
        cond = {k: sr.get(k, base[k]) for k in ("pads", "canvas", "r", "feather_px", "audio_strength", "inject_clean")}
        mp4 = root / label / "output.mp4"
        m = sr.get("metrics") or {}
        bg = m.get("boundary_gradient_ratio") or {}
        rows.append({
            "label": label, **cond, "seed": seed, "mp4": mp4 if mp4.is_file() else None,
            "status": sr.get("status") or ("done" if mp4.is_file() else "未実行"),
            "keep": m.get("keep_psnr_vs_control_db"), "core": m.get("keep_core_psnr_vs_control_db"),
            "vmax": bg.get("vertical_max"), "hmax": bg.get("horizontal_max"),
            "seconds": sr.get("seconds", res.get("seconds")),
            "peak": sr.get("peak_vram_mb", res.get("peak_vram_mb")),
            "reserved": sr.get("peak_vram_reserved_mb", res.get("peak_vram_reserved_mb")),
        })
    return rows


def hf_name(r: dict) -> str:
    cw, ch = r["canvas"]
    return f"{r['label']}_{cw}x{ch}_r{r['r']}_s{r['audio_strength']:g}_seed{r['seed']}.mp4"


def copy_if_changed(src: Path, dst: Path) -> bool:
    if dst.is_file() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
        return False
    shutil.copy2(src, dst)
    return True


# ---------------------------------------------------------------------------------------------
# 一覧の画像
# ---------------------------------------------------------------------------------------------

def load_font(size: int):
    from PIL import ImageFont
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def grab(mp4: Path, tmp: Path, tag: str) -> list:
    from PIL import Image
    out = []
    for i, n in enumerate(FRAME_IDX):
        png = tmp / f"{tag}_{i}.png"
        cmd = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(mp4), "-vf", f"select=eq(n\\,{n})",
               "-vsync", "0", "-frames:v", "1", "-y", str(png)]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if r.returncode != 0 or not png.is_file():
            print(f"  注意: {mp4.name} の {n} コマ目を切り出せない: {(r.stderr or '').strip()[:200]}")
            out.append(None)
            continue
        with Image.open(png) as im:
            out.append(im.convert("RGB").copy())
    return out


def thumb(im, keep_rect):
    """長辺を LONG_SIDE に縮め、保持矩形（元の座標 x, y, w, h）があれば 1 画素の枠を描く。"""
    from PIL import Image, ImageDraw
    k = LONG_SIDE / max(im.size)
    t = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    if keep_rect:
        x, y, w, h = keep_rect
        x0, y0 = round(x * k), round(y * k)
        x1, y1 = min(t.width, round((x + w) * k)) - 1, min(t.height, round((y + h) * k)) - 1
        ImageDraw.Draw(t).rectangle([x0, y0, x1, y1], outline=FRAME_COLOR, width=1)
    return t


def make_sheet(entries: list[dict], key: str, out: Path) -> bool:
    """entries: {name, lines[], mp4, rect|None}。1 本 1 行・3 コマ。"""
    from PIL import Image, ImageDraw
    entries = [e for e in entries if e["mp4"] is not None]
    if not entries:
        print(f"  sheet_{key}: 並べる動画が無いので作らない")
        return False
    font, small = load_font(20), load_font(16)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for e in entries:
            ims = grab(e["mp4"], Path(td), e["name"].replace(" ", "_"))
            rows.append([thumb(im, e["rect"]) if im is not None else None for im in ims])
    row_h = [max([t.height for t in r if t is not None] or [LONG_SIDE * 9 // 16]) for r in rows]
    col_w = max([t.width for r in rows for t in r if t is not None] or [LONG_SIDE])
    width = GAP + LABEL_W + GAP + 3 * (col_w + GAP)
    height = GAP + HEADER_H + GAP + sum(h + GAP for h in row_h)
    sheet = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(sheet)
    d.text((GAP, GAP + 8), f"sheet_{key}", fill="black", font=font)
    for c, n in enumerate(FRAME_IDX):
        d.text((GAP + LABEL_W + GAP + c * (col_w + GAP) + 4, GAP + 8), f"frame {n}", fill="black", font=font)
    y = GAP + HEADER_H + GAP
    for e, r, h in zip(entries, rows, row_h):
        ty = y + 4
        d.text((GAP + 4, ty), e["name"], fill="black", font=font)
        for line in e["lines"]:
            ty += 24
            d.text((GAP + 4, ty), line, fill="black", font=small)
        for c, t in enumerate(r):
            x = GAP + LABEL_W + GAP + c * (col_w + GAP)
            if t is None:
                d.rectangle([x, y, x + col_w - 1, y + h - 1], fill=(200, 200, 200))
            else:
                sheet.paste(t, (x, y))
        y += h + GAP
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"書いた: {out}（{sheet.width}×{sheet.height}・{len(entries)} 行）")
    return True


def run_entry(r: dict) -> dict:
    p = r["pads"]
    cw, ch = r["canvas"]
    return {"name": r["label"], "mp4": r["mp4"], "rect": (p["pad_left"], p["pad_top"], SRC_W, SRC_H),
            "lines": [f"{cw}x{ch}  r{r['r']} ({r['feather_px']}px)",
                      f"s {r['audio_strength']:g}  inject {r['inject_clean']:g}", f"seed {r['seed']}"]}


# ---------------------------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------------------------

def write_readme(hf: Path, rows: list[dict], names: dict[str, str], sheets: list[str], source_ok: bool,
                 g0: dict, g0_files: list[str]) -> None:
    L = [
        "# round13-h3-outpaint — MiniMax H3 の Outpainting の GPU の門（2026-11-10）",
        "",
        "元の動画（1280×720・124 フレーム・24fps）の外側へキャンバスを広げ、外側を生成で描き足す機能（Outpainting。"
        "台帳 PENDING_TASKS.md §1-125）の門の動画です。保持矩形（元の動画を置いた位置）の画質、継ぎ目、動きの連続を目で確かめます。"
        "数値の正本は Nz-Videomni の VERIFICATION_LOG.md、台本は `Docs/Outputs-archive/h3-outpaint-2026-11-10/` です。",
        "",
        "## 各動画の条件",
        "",
        "なじませ幅は継ぎ目をぼかす幅（px）、音声 Strength は元の音声を残す割合（1.0 で元の波形）、"
        "注入係数は元の映像の潜在を入れる強さです。",
        "",
        "| 本 | キャンバス | pad 左/右/上/下 | なじませ r（px） | 音声 Strength | 注入係数 | ファイル |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        p = r["pads"]
        fn = f"`{names[r['label']]}`" if r["label"] in names else "未実行"
        L.append(f"| {r['label']} | {r['canvas'][0]}×{r['canvas'][1]} | "
                 f"{p['pad_left']}/{p['pad_right']}/{p['pad_top']}/{p['pad_bottom']} "
                 f"| {r['r']}（{r['feather_px']}） | {r['audio_strength']:g} | {r['inject_clean']:g} | {fn} |")
    L += ["", f"種は全本 {rows[0]['seed']}。" + ("比べる元は `source_1280x720_124f.mp4`（元の動画の切り出し）です。" if source_ok else ""),
          "", "## 数値", "",
          "PSNR は保持矩形を、同じ再符号化を経た対照と比べた値（dB。大きいほど元に近い）で、芯はなじませの重みが 1 の画素だけです。"
          "境界の勾配比は保持矩形の辺をまたぐ画素差を近傍の差で割った値（1.0 で近傍と同じ、大きいほど継ぎ目が立つ。"
          "縦は左右の辺、横は上下の辺の大きいほう）です。",
          "",
          "| 本 | 状態 | 保持矩形 PSNR | 芯 PSNR | 勾配比（縦） | 勾配比（横） | 所要秒 | VRAM の山（割り当て MiB） | 同（予約 MiB） |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        status = r["status"] if r["mp4"] is not None else "未実行"
        L.append(f"| {r['label']} | {status} | {g(r['keep'])} | {g(r['core'])} "
                 f"| {g(r['vmax'])} | {g(r['hmax'])} | {g(r['seconds'], 1)} | {g(r['peak'], 0)} | {g(r['reserved'], 0)} |")
    L += ["", "## 見方", "",
          "- 各画像は 1 本 1 行、先頭・中央・末尾の 3 コマです。**マゼンタの 1 画素の枠の内側が元の映像、外側が生成**です。",
          "- 見るのは、枠の辺に沿った継ぎ目、色や明るさの段差、時間方向の動きの連続（枠をまたいで人物や背景の動きが途切れないか）です。",
          "- `G2_r0` はなじませなし、`G2_r3` は広いなじませ、`G1_odd` は 32 に乗らない pad の例です。",
          "- `G3_inject0999` は注入係数だけ 0.999 の対照、`G4_s1` は音声 Strength 1.0（元の波形が入る）です。",
          ""]
    if sheets:
        L += ["画像: " + "・".join(f"`{s}`" for s in sheets), ""]
    L += ["## G0（VAE の往復だけ。生成ではありません）", "",
          "キャンバスごと符号化したとき、余白の詰め物が元の領域をどれだけ汚すかを測る門です。4 本は元の動画を符号化して復号しただけで、"
          "描き足しはありません。違いは符号化の仕方だけです。",
          "",
          "- `single`: 外接矩形だけを単独で符号化（設計の採用案）",
          "- `canvas_imagenet`・`canvas_gray128`: キャンバスに置き、余白を ImageNet の平均色・灰色 128 で埋めて符号化",
          "- `control_source`: 元の動画をそのまま往復（下限）",
          "",
          "見るのは、`single` と `canvas_*` の、保持矩形の辺の近くの汚れの違いです。"]
    if g0_files:
        L += ["", "ファイル: " + "・".join(f"`{f}`" for f in g0_files)]
    ps = g0.get("psnr_vs_cut") if isinstance(g0.get("psnr_vs_cut"), dict) else {}
    if ps:
        L += ["", "| 名前 | 保持矩形 PSNR | 辺 32px の帯 | その内側 | 帯 − 内側 |", "|---|---|---|---|---|"]
        for n in G0_NAMES:
            v = ps.get(n) or {}
            L.append(f"| {n} | {g(v.get('keep_db'))} | {g(v.get('edge32_db'))} | {g(v.get('interior_db'))} | "
                     f"{g(v.get('edge_minus_interior_db'))} |")
    L.append("")
    (hf / "README.md").write_text("\n".join(L), encoding="utf-8")
    print(f"書いた: {hf / 'README.md'}")


# ---------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="Outpainting の門の遠隔目視用フォルダを作る（GPU は使わない）")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_ROOT, help="門の出力の根（既定 outputs\\h3-outpaint-gates-2026-11-10）")
    ap.add_argument("--hf-dir", type=Path, default=None, help="書き先（既定は <out-root>\\hf-round13）")
    ap.add_argument("--no-sheets", action="store_true", help="一覧の画像を作らない")
    a = ap.parse_args(argv)
    root = a.out_root
    hf = a.hf_dir or (root / "hf-round13")
    summary = load_json(root / "summary.json")
    rows = collect(root, summary)
    hf.mkdir(parents=True, exist_ok=True)
    names: dict[str, str] = {}
    for r in rows:
        if r["mp4"] is None:
            print(f"  {r['label']}: 未実行（output.mp4 が無い）")
            continue
        names[r["label"]] = hf_name(r)
        if copy_if_changed(r["mp4"], hf / names[r["label"]]):
            print(f"  写した: {names[r['label']]}")
    src = root / "_material" / SOURCE_NAME
    source_ok = src.is_file()
    if source_ok:
        copy_if_changed(src, hf / "source_1280x720_124f.mp4")
    g0_dir = root / "G0"
    g0_files, g0_entries = [], []
    for n in G0_NAMES:
        p = g0_dir / f"{n}.mp4"
        if p.is_file():
            copy_if_changed(p, hf / f"G0_{n}.mp4")
            g0_files.append(f"G0_{n}.mp4")
            g0_entries.append({"name": f"G0 {n}", "mp4": p, "rect": None, "lines": ["VAE round trip only"]})
    g0 = load_json(g0_dir / "result.json") or (summary.get("g0") or {})
    sheets: list[str] = []
    if not a.no_sheets:
        if not FFMPEG.is_file():
            print(f"ffmpeg が無い: {FFMPEG}", file=sys.stderr)
            return 1
        if make_sheet([run_entry(r) for r in rows], "all", hf / "sheet_all.png"):
            sheets.append("sheet_all.png")
        for r in rows:
            if r["mp4"] is not None and make_sheet([run_entry(r)], r["label"], hf / f"sheet_{r['label']}.png"):
                sheets.append(f"sheet_{r['label']}.png")
        if g0_entries and make_sheet(g0_entries, "G0", hf / "sheet_G0.png"):
            sheets.append("sheet_G0.png")
    write_readme(hf, rows, names, sheets, source_ok, g0, g0_files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
