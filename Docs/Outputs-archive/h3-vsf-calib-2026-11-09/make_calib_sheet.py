r"""MiniMax H3 の VSF の実用域の較正（台帳 §3-278・記録簿 §203）の表・一覧の画像・Hugging Face に上げるフォルダを作る補助。

``h3_vsf_calib.py`` で回した後に、監督が実行する（GPU もサーバーも使わない。読んで書くだけ）::

    .venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-calib-2026-11-09\make_calib_sheet.py
    .venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-calib-2026-11-09\make_calib_sheet.py --hf-dir <フォルダ>

読むもの: ``calib_plan.json``（条件の正本）・``jobs\<名前>\metadata.json``（無ければ ``summary.json`` の記録）・
``jobs\<名前>\*.mp4``。

書くもの:
  - ``VISUAL_TABLE.md``: 1 本 1 行の表（A_off を先頭に、計画の順）。m は有効数字 3 桁。
  - ``sheets\sheet_<A|R|B|C|D>.png``: 段ごとの一覧の画像。1 本につき 3 コマ（56 フレームは 0.5・1.2・2.0 秒、
    124 フレームは 0.5・2.5・4.5 秒）を ``tools\ffmpeg\bin\ffmpeg.exe`` で切り出し、幅 448 に縮めて並べる
    （行＝本、列＝時刻、左端に名前・scale・seed・寸法）。動画がまだ無い本は行を省く（段 A だけの後でも作れる）。
  - ``--hf-dir``: Hugging Face に上げるフォルダ。mp4 を ``<名前>_<幅>x<高さ>_seed<seed>.mp4`` で写し、一覧の画像と
    ``README.md`` を置く。何度でも回してよい（後の段の分が足される。フォルダの中の既存のファイルは消さない）。

``--plan``・``--jobs-dir``・``--summary``・``--out-dir`` は試しに別の入力・出力先へ向けるためのもの（既定はこのフォルダ）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # Docs\Outputs-archive\<このフォルダ> の 3 つ上＝リポジトリ直下
FFMPEG = REPO / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"

THUMB_W = 448
GAP = 8
LABEL_W = 230
HEADER_H = 44
TIMES_SHORT = (0.5, 1.2, 2.0)  # 56 フレーム＝2.33 秒
TIMES_LONG = (0.5, 2.5, 4.5)  # 124 フレーム＝5.17 秒

SHEETS: dict[str, list[str]] = {
    "A": ["A_off", "A_s0", "A_s1", "A_s1.5", "A_s2", "A_s2.5", "A_s3", "A_s4", "A_s5"],
    "R": ["R_s0", "R_s1", "R_s1.5", "R_s2", "A_s0", "A_s1", "A_s1.5", "A_s2"],
    "B": ["A_s5", "B_s6", "B_s8"],
    # 比べる相手は押す力の目安（scale × m_neg）が C_s1.5_N2 に近い A_s2・A_s2.5（実測で決めた。A_s4 は数値を上げた側の参考）
    "C": ["A_s0", "A_s1.5", "C_s1.5_N2", "A_s2", "A_s2.5", "A_s4"],
    "D": ["L_s0", "L_s1.5", "L_s3", "A_s0", "A_s1.5", "A_s3"],
}
STAGE_LABEL = {"A": "A", "R": "A′", "B": "B", "C": "C", "D": "D"}

LOOK_POINTS = [
    "**消えるか**: 青い服（デニムのシャツ）と車が、どの scale から見えなくなるか。scale 0（対照）に写っていることが前提です。"
    "元の動画では車は 0.5 秒の時点で右に写り、2.0 秒では画面から出ているので、**車は 0.5 秒のコマで見てください**。",
    "**正の場面が保たれるか**: 日本らしい街路・こちらへ歩く・微笑む・手を伸ばす。",
    "**崩れ・逆転**: 強い scale で絵が壊れるか、負に挙げた要素がかえって強まるか（先行事例の negpip〔画像生成 AI 向けのネガティブプロンプトの拡張〕で報告された現象）。",
    "{C_POINT}",
    "**長い尺**: 段 D（124 フレーム）の 1.5 と 3 が、段 A（56 フレーム）の 1.5 と 3 より弱く効くか。",
]
DECIDE_POINTS = [
    "**既定値**: seed 42 と 43 の両方で 2 要素が消え、正の場面が保たれる最小の scale（見込みは 1.5 か 2）。",
    "**スライダーの最大値**: seed 42 で正の場面が保たれる最大の scale（5 が保たれて 6〜8 で崩れるなら 5。先行事例と揃う）。"
    "API の上限 10 は触らない。",
    "段 D で長い尺のほうが明らかに弱く効くなら、ヒント文に「長い動画ほど効きが弱い（scale を上げる）」を足すか、"
    "既定値を長い尺でも効く側に寄せる（どちらにするかは結果を見て決める。文言は後の計画）。{D_MEASURED}",
    "段 C で同じ押す力でも言い換えのほうが場面を保つなら、ヒント文に「言い換えのほうが数値を上げるより効く」を足す"
    "（negpip の作者の助言と同じ）。",
    "値が LTX と違えば、操作パネルにベースモデルごとの範囲の仕組みが要る。要るかどうかはこの結果を見てから別に相談する。",
]


# ---------------------------------------------------------------------------------------------
# 読む
# ---------------------------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sig3(x) -> str:
    """有効数字 3 桁（指数の表記にしない。例 0.000873）。"""
    if x is None:
        return "–"
    if not isinstance(x, (int, float)):
        return str(x)
    if x == 0:
        return "0"
    s = f"{float(x):.3g}"
    if "e" in s:
        s = f"{Decimal(s):f}"
    return s


def fmt(x, digits: int | None = None) -> str:
    if x is None:
        return "–"
    if isinstance(x, float) and digits is not None:
        return f"{x:.{digits}f}"
    return str(x)


def fmt_scale(s) -> str:
    return "off" if s is None else f"{float(s):g}"


def find_mp4(job_dir: Path, meta: dict) -> Path | None:
    """そのジョブの mp4。metadata の job_id と同じ名前を優先し、無ければ一番新しいもの。"""
    if not job_dir.is_dir():
        return None
    jid = meta.get("job_id")
    if jid and (job_dir / f"{jid}.mp4").is_file():
        return job_dir / f"{jid}.mp4"
    mp4s = sorted(job_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return mp4s[0] if mp4s else None


def ordered_jobs(plan: dict) -> list[dict]:
    """表の順: 生成しない本（A_off）を先頭に、あとは計画の順。"""
    jobs = plan["jobs"]
    return [j for j in jobs if not j.get("generate", True)] + [j for j in jobs if j.get("generate", True)]


def neg_label(plan: dict, neg: str | None) -> str:
    if neg is None:
        return "なし"
    for k, v in (plan.get("negatives") or {}).items():
        if v == neg:
            return k
    return "?"


def collect(plan: dict, jobs_dir: Path, summary: dict) -> dict[str, dict]:
    sjobs = summary.get("jobs") or {}
    rows: dict[str, dict] = {}
    for job in ordered_jobs(plan):
        name = job["name"]
        jd = jobs_dir / name
        meta = load_json(jd / "metadata.json")
        sj = sjobs.get(name) or {}
        ms = sj.get("metadata_summary") or {}
        h3 = meta.get("h3") if isinstance(meta.get("h3"), dict) else {}
        if meta:
            vsf = h3.get("vsf") if "vsf" in h3 else meta.get("vsf")
        else:
            vsf = sj.get("vsf")
        vsf = vsf if isinstance(vsf, dict) else None
        enc = ((h3.get("phases") or {}).get("encode") or {})
        mp4 = find_mp4(jd, meta)
        status = sj.get("status")
        if not status:
            status = ("copied" if not job.get("generate", True) else "completed") if (meta and mp4) else "未実行"
        scale = job.get("vsf_scale")
        m_neg = vsf.get("m_neg") if vsf else None
        rows[name] = {
            "job": job, "status": status, "mp4": mp4, "meta": meta,
            "vsf": vsf, "rows": vsf.get("rows") if vsf else None, "m_neg": m_neg,
            "m_text": vsf.get("m_text") if vsf else None, "vsf_scale_used": vsf.get("scale") if vsf else None,
            "push": (float(scale) * m_neg) if (scale is not None and m_neg is not None) else None,
            "gen_sec": meta.get("generation_time_seconds", sj.get("generation_time_seconds",
                                                                  ms.get("generation_time_seconds"))),
            "wall_sec": sj.get("seconds_wall"),
            "peak_reserved": meta.get("peak_vram_reserved_mb", sj.get("peak_vram_reserved_mb",
                                                                      ms.get("peak_vram_reserved_mb"))),
            "peak_alloc": (meta.get("vram_optimization") or {}).get("peak_vram_mb",
                                                                    sj.get("peak_vram_mb", ms.get("peak_vram_mb"))),
            "attention": meta.get("attention_used", sj.get("attention_used", ms.get("attention_used"))),
            "negative_tokens": enc.get("negative_tokens", sj.get("negative_tokens")),
            "sha": sj.get("sha256") or (sha256(mp4) if mp4 else None),
            "error": (str(sj.get("error") or "")).replace("|", "\\|").replace("\n", " ")[:200],
        }
    return rows


# ---------------------------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------------------------

def common_text(plan: dict) -> list[str]:
    c = plan["common"]
    lora = c["loras"][0]
    return [
        f"- 正のプロンプト: `{c['prompt']}`",
        f"- Turbo LoRA `{lora['name']}` 強度 {lora['strength']}・{c['num_inference_steps']} ステップ・"
        f"shift は映像 {c['shift_video']:g}／音声 {c['shift_audio']:g}・注意の計算 {c['attention_backend']}・"
        f"VRAM 16GB mode {'on' if c['vram16_mode'] else 'off'}・{c['frame_rate']:g}fps・transformer は `default`（FL2VA）。",
        "- 生成した本はすべて `nag_enabled=true`・`neg_method=\"vsf\"`・負の文・`vsf_scale`（0 を含む）を送った。"
        "`embed_mp4_metadata` と `keep_resident` は送っていない（既定のまま。mp4 に生成条件が埋め込まれる）。",
        f"- 負の文: N1 = `{plan['negatives']['N1']}`、N2 = `{plan['negatives']['N2']}`。",
        "- `A_off` は負の文なしの対照で、第 1 弾の T0（`../h3-vsf-2026-11-09/visual/T0`）をそのまま写したもの（生成していない）。",
    ]


def write_table(plan: dict, rows: dict[str, dict], out: Path) -> None:
    lines = [
        "# VSF の実用域の較正の表（make_calib_sheet.py が書いた）",
        "",
        "台帳 §3-278・記録簿 §203。条件の正本は `calib_plan.json`。",
        "",
        *common_text(plan),
        "",
        "`m_neg` は負の文の行が注意の配分を奪った割合、`m_text` は正の文の行に乗った割合（ステップ 0・最初のブロック・"
        "映像の行 256 本の平均。sdpa の定義で計算した値で、sage の実際の重みとはわずかに違いうる）。"
        "押す力の目安は scale × m_neg。scale 0 でも負の行は同じ配置で置かれるので、負の行と m は scale によらず同じになるはず。"
        "sha256 は記録だけ（mp4 に生成条件が埋め込まれるので、重なりの確認には使えない）。",
        "",
        "| 名前 | 段 | seed | 寸法×フレーム | 負の文 | scale | 状態 | 負の行（始, 終） | m_neg | m_text | 押す力の目安 "
        "| 生成の秒 | 待ちの秒 | VRAM の山（予約 MiB） | VRAM の山（割り当て MiB） | 注意の計算 | sha256（先頭 16） | エラー |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        j = r["job"]
        sha = f"`{r['sha'][:16]}`" if r["sha"] else "–"
        lines.append(
            f"| {name} | {STAGE_LABEL.get(j['stage'], j['stage'])} | {j['seed']} | "
            f"{j['width']}×{j['height']}×{j['num_frames']} | {neg_label(plan, j.get('negative'))} | "
            f"{fmt_scale(j.get('vsf_scale'))} | {r['status']} | {fmt(r['rows'])} | {sig3(r['m_neg'])} | "
            f"{sig3(r['m_text'])} | {sig3(r['push'])} | {fmt(r['gen_sec'], 1)} | {fmt(r['wall_sec'], 1)} | "
            f"{fmt(r['peak_reserved'])} | {fmt(r['peak_alloc'])} | {fmt(r['attention'])} | {sha} | "
            f"{r['error'] or '–'} |")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"書いた: {out}")


# ---------------------------------------------------------------------------------------------
# 一覧の画像
# ---------------------------------------------------------------------------------------------

def times_for(job: dict) -> tuple[float, float, float]:
    return TIMES_LONG if job["num_frames"] > 56 else TIMES_SHORT


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


def grab_frames(mp4: Path, times: tuple[float, ...], tmp: Path, tag: str) -> list[Path | None]:
    out: list[Path | None] = []
    for i, t in enumerate(times):
        png = tmp / f"{tag}_{i}.png"
        cmd = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-ss", f"{t}", "-i", str(mp4), "-frames:v", "1",
               "-y", str(png)]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if r.returncode != 0 or not png.is_file():
            print(f"  注意: {mp4.name} の {t} 秒のコマを切り出せない: {(r.stderr or '').strip()[:300]}")
            out.append(None)
        else:
            out.append(png)
    return out


def make_sheet(plan: dict, rows: dict[str, dict], key: str, names: list[str], out: Path) -> bool:
    from PIL import Image, ImageDraw

    avail = []
    for n in names:
        r = rows.get(n)
        if r is None or r["mp4"] is None:
            print(f"  sheet_{key}: {n} の動画が無いので行を省く")
            continue
        avail.append(n)
    if not avail:
        print(f"  sheet_{key}: 並べる動画が 1 本も無いので作らない")
        return False
    font = load_font(20)
    small = load_font(17)
    thumbs: dict[str, list] = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for n in avail:
            job = rows[n]["job"]
            ims = []
            for p in grab_frames(rows[n]["mp4"], times_for(job), tmp, n.replace(".", "_")):
                if p is None:
                    ims.append(None)
                    continue
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    h = round(im.height * THUMB_W / im.width)
                    ims.append(im.resize((THUMB_W, h), Image.LANCZOS))
            thumbs[n] = ims
    row_h = {n: max([im.height for im in thumbs[n] if im is not None] or [round(THUMB_W * 768 / 1344)])
             for n in avail}
    width = GAP + LABEL_W + GAP + 3 * (THUMB_W + GAP)
    height = GAP + HEADER_H + GAP + sum(row_h[n] + GAP for n in avail)
    sheet = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(sheet)
    d.text((GAP, GAP + 10), f"sheet_{key}", fill="black", font=font)
    for c in range(3):
        ts = sorted({times_for(rows[n]["job"])[c] for n in avail})
        label = "t = " + " / ".join(f"{t:g} s" for t in ts)
        if len(ts) > 1:
            label += "  (56f / 124f)"
        x = GAP + LABEL_W + GAP + c * (THUMB_W + GAP)
        d.text((x + 4, GAP + 10), label, fill="black", font=font)
    y = GAP + HEADER_H + GAP
    for n in avail:
        job = rows[n]["job"]
        neg = neg_label(plan, job.get("negative"))
        lines = [(n, font), (f"scale {fmt_scale(job.get('vsf_scale'))}", small),
                 (f"neg {neg if neg != 'なし' else 'none'}", small), (f"seed {job['seed']}", small),
                 (f"{job['width']}x{job['height']}x{job['num_frames']}", small)]
        ty = y + 4
        for text, f in lines:
            d.text((GAP + 4, ty), text, fill="black", font=f)
            ty += 28 if f is font else 24
        for c, im in enumerate(thumbs[n]):
            x = GAP + LABEL_W + GAP + c * (THUMB_W + GAP)
            if im is None:
                d.rectangle([x, y, x + THUMB_W - 1, y + row_h[n] - 1], fill=(200, 200, 200))
                d.text((x + 8, y + 8), "no frame", fill="black", font=small)
            else:
                sheet.paste(im, (x, y))
        y += row_h[n] + GAP
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"書いた: {out}（{sheet.width}×{sheet.height}・{len(avail)} 行）")
    return True


# ---------------------------------------------------------------------------------------------
# Hugging Face に上げるフォルダ
# ---------------------------------------------------------------------------------------------

def hf_name(job: dict) -> str:
    return f"{job['name']}_{job['width']}x{job['height']}_seed{job['seed']}.mp4"


def copy_if_changed(src: Path, dst: Path) -> bool:
    if dst.is_file() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
        return False
    shutil.copy2(src, dst)
    return True


def machine_checks(plan: dict, rows: dict[str, dict]) -> str:
    gen = [j for j in plan["jobs"] if j.get("generate", True)]
    done = [j for j in gen if rows[j["name"]]["status"] == "completed"]
    with_vsf = [j for j in done if rows[j["name"]]["vsf"] is not None]
    groups: dict[tuple, list[str]] = {}
    for j in done:
        groups.setdefault((j["seed"], j["width"], j["height"], j["num_frames"], j.get("negative")), []).append(j["name"])
    same, differ = [], []
    for key, names in groups.items():
        if len(names) < 2:
            continue
        vals = {(json.dumps(rows[n]["rows"]), rows[n]["m_neg"], rows[n]["m_text"]) for n in names}
        (same if len(vals) == 1 else differ).append(names)
    parts = [f"生成する 18 本のうち {len(done)} 本が完了（`summary.json` の status が completed）",
             f"完了した本のうち {len(with_vsf)} 本の `metadata.json` に `h3.vsf`（負の行・scale・m_neg・m_text）がある"
             "（scale 0 を含む）"]
    if same or differ:
        txt = (f"seed・寸法・フレーム・負の文が同じ本の組 {len(same) + len(differ)} 組のうち {len(same)} 組で、"
               "負の行と m_neg・m_text が scale によらず同じ（反転した値に掛ける倍率に依らない値なので、同じになるのが正しい）")
        if differ:
            txt += "。違った組: " + "、".join("・".join(f"`{n}`" for n in g) for g in differ)
        parts.append(txt)
    return "機械で確かめたこと: " + "。".join(parts) + "。"


def n_rows(r: dict | None) -> int | None:
    rr = (r or {}).get("rows")
    return rr[1] - rr[0] if isinstance(rr, list) and len(rr) == 2 else None


def measured_texts(rows: dict[str, dict]) -> dict[str, str]:
    """言い換え（段 C）と長い尺（段 D）について、実測の m から README の文を組む（値が無ければ控えめな文）。"""
    a, c, l15 = rows.get("A_s1.5"), rows.get("C_s1.5_N2"), rows.get("L_s1.5")
    partners = "押す力の目安が近い `A_s2`・`A_s2.5`"
    ratio = ""
    if a and c and a["m_neg"] and c["m_neg"]:
        na, nc = n_rows(a), n_rows(c)
        times_rows = f"{nc / na:g} 倍" if na and nc else "数倍"
        ratio = (f"N2 の行は N1 の {times_rows}（{nc} 行と {na} 行）だが、m_neg は約 {c['m_neg'] / a['m_neg']:.2f} 倍"
                 f"（{sig3(c['m_neg'])} と {sig3(a['m_neg'])}）。1 行あたりの配分は行が増えるほど薄まる")
    c_point = (f"**言い換え**: `C_s1.5_N2` を、{partners}（N1 の scale 2 と 2.5）と見比べ、同じ力でも言い換えのほうが"
               "場面を保ったまま消すか。" + (f"{ratio}ため、計画の当初の見込み（`A_s4`・`A_s5` が相手）より相手の scale が"
                                           "低くなりました。`sheet_C.png` には数値を上げた側の参考に `A_s4` も並べています。" if ratio else ""))
    c_note = f"言い換え（同義語で厚くした負の文）。{partners}と比べる。" + (f"{ratio}。" if ratio else "")
    d_measured = ""
    if a and l15 and a["m_neg"] and l15["m_neg"]:
        d_measured = (f"実測の m_neg は 124 フレームで {sig3(l15['m_neg'])}、56 フレームで {sig3(a['m_neg'])} と、"
                      f"長い尺は約 {l15['m_neg'] / a['m_neg']:.2f} 倍です（同じ scale なら押す力の目安も同じ割合で小さい）。")
    return {"C_POINT": c_point, "C_NOTE": c_note, "D_MEASURED": d_measured}


def write_hf(plan: dict, rows: dict[str, dict], sheets_dir: Path, hf: Path) -> None:
    mt = measured_texts(rows)
    hf.mkdir(parents=True, exist_ok=True)
    files: list[tuple[dict, str]] = []
    for name, r in rows.items():
        if r["mp4"] is None:
            continue
        dst = hf / hf_name(r["job"])
        if copy_if_changed(r["mp4"], dst):
            print(f"  写した: {dst.name}")
        files.append((r["job"], dst.name))
    sheet_names = []
    for key in SHEETS:
        p = sheets_dir / f"sheet_{key}.png"
        if p.is_file():
            if copy_if_changed(p, hf / p.name):
                print(f"  写した: {p.name}")
            sheet_names.append(p.name)
    c = plan["common"]
    lora = c["loras"][0]
    lines = [
        "# round12-h3-vsf-calib — MiniMax H3 の VSF の実用域の較正（2026-11-09）",
        "",
        "MiniMax H3 のネガティブプロンプトの強さ（scale）の既定値と、操作パネルのスライダーの最大値を決めるための動画です。"
        "台帳 PENDING_TASKS.md §3-278。数値と条件の正本は Nz-Videomni の VERIFICATION_LOG.md §203 です。",
        "",
        "## 前提",
        "",
        "- **VSF**（Value Sign Flip）: 負のプロンプトの行の値の符号を反転し、1 回の注意の計算で負に挙げた要素を遠ざける方式です。"
        "MiniMax H3 のネガティブプロンプトはこの方式だけです。scale はその押す強さです。",
        "- **scale 0 は対照です**: 負の文の行を 1.5 や 3 と同じ配置で置き、反転した値だけをゼロにしたものです。"
        "負の文なし（`A_off`）と比べると行の配置そのものが違うので、scale の効きだけを見るには scale 0 と比べます。",
        "- **構図や服装の変化は受け入れます**: オーナーの事前の考察どおり、負の語に引っ張られて構図や服装が変わるのは"
        "ネガティブプロンプトの性質です（画像生成 AI の経験と同じ。部分だけを直す Inpainting とは違います）。"
        "見るのは「挙げた要素が消えるか」と「正の場面が保たれるか」で、「ほかが変わらないか」ではありません。",
        "- **押す力の目安**は scale × m_neg です。m_neg は負の文の行が注意の配分を奪った割合で、負の文の行数が多いほど、"
        "また動画の系列が長いほど値が変わります。",
        "",
        "## 共通の条件",
        "",
        f"- プロンプト: {c['prompt']}",
        f"- Turbo LoRA `{lora['name']}` 強度 {lora['strength']}・{c['num_inference_steps']} ステップ・"
        f"shift は映像 {c['shift_video']:g}／音声 {c['shift_audio']:g}・注意の計算 {c['attention_backend']}・"
        f"VRAM 16GB mode {'on' if c['vram16_mode'] else 'off'}・{c['frame_rate']:g}fps・FL2VA の重み。",
        f"- 負の文 N1: `{plan['negatives']['N1']}`（元の動画に写っている 2 つの要素）",
        f"- 負の文 N2: `{plan['negatives']['N2']}`（N1 を同義語で厚くした言い換え）",
        "- `A_off` は負の文なしの元の動画です（前の round の T0 と同じもの。生成し直していません）。",
        "",
        "## 各動画の条件",
        "",
        "| ファイル | 段 | seed | 寸法×フレーム | 負の文 | scale | m_neg | 押す力の目安 | 目的 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for job, fname in files:
        r = rows[job["name"]]
        lines.append(f"| `{fname}` | {STAGE_LABEL.get(job['stage'], job['stage'])} | {job['seed']} | "
                     f"{job['width']}×{job['height']}×{job['num_frames']} | {neg_label(plan, job.get('negative'))} | "
                     f"{fmt_scale(job.get('vsf_scale'))} | {sig3(r['m_neg'])} | {sig3(r['push'])} | "
                     f"{mt['C_NOTE'] if job['name'] == 'C_s1.5_N2' else job.get('note', '')} |")
    missing = [n for n, r in rows.items() if r["mp4"] is None]
    lines.append("")
    if missing:
        lines += ["まだ上げていない本（後で同じフォルダに足します）: " + "・".join(f"`{n}`" for n in missing), ""]
    lines += ["段 A′ の本の名前は `R_` で始まります（2 つ目の seed で確かめる段）。段 D の本の名前は `L_` で始まります（長い尺）。", ""]
    if sheet_names:
        lines += ["## 一覧の画像", "",
                  "各本の 3 つの時刻のコマを横に並べた画像です（56 フレームの本は 0.5・1.2・2.0 秒、124 フレームの本は 0.5・2.5・4.5 秒）。"
                  "行が 1 本、左端に名前・scale・負の文・seed・寸法があります。", ""]
        lines += [f"- `{s}`" for s in sheet_names]
        lines.append("")
    lines += ["## 見ていただきたい点", ""]
    lines += [f"{i}. {t.replace('{C_POINT}', mt['C_POINT'])}" for i, t in enumerate(LOOK_POINTS, 1)]
    lines += ["", "## 決め方の案", "", "数値と動画を見てから決めます。", ""]
    lines += [f"- {t.replace('{D_MEASURED}', mt['D_MEASURED'])}" for t in DECIDE_POINTS]
    lines += ["", machine_checks(plan, rows), ""]
    (hf / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"書いた: {hf / 'README.md'}（動画 {len(files)} 本・画像 {len(sheet_names)} 枚）")


# ---------------------------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="VSF の較正の表・一覧の画像・Hugging Face に上げるフォルダを作る（GPU は使わない）")
    ap.add_argument("--plan", type=Path, default=HERE / "calib_plan.json", help="計画（既定はこのフォルダの calib_plan.json）")
    ap.add_argument("--jobs-dir", type=Path, default=HERE / "jobs", help="1 本ごとのフォルダの親（既定 jobs）")
    ap.add_argument("--summary", type=Path, default=HERE / "summary.json", help="台本の記録（既定 summary.json）")
    ap.add_argument("--out-dir", type=Path, default=HERE, help="VISUAL_TABLE.md と sheets の書き先（既定はこのフォルダ）")
    ap.add_argument("--hf-dir", type=Path, default=None, help="Hugging Face に上げるフォルダ（指定したときだけ作る）")
    ap.add_argument("--no-sheets", action="store_true", help="一覧の画像を作らない（表だけ）")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args(argv)
    plan = load_json(args.plan)
    if not plan.get("jobs"):
        print(f"計画を読めない: {args.plan}", file=sys.stderr)
        return 1
    rows = collect(plan, args.jobs_dir, load_json(args.summary))
    write_table(plan, rows, args.out_dir / "VISUAL_TABLE.md")
    sheets_dir = args.out_dir / "sheets"
    if not args.no_sheets:
        if not FFMPEG.is_file():
            print(f"ffmpeg が無い: {FFMPEG}", file=sys.stderr)
            return 1
        for key, names in SHEETS.items():
            make_sheet(plan, rows, key, names, sheets_dir / f"sheet_{key}.png")
    if args.hf_dir is not None:
        write_hf(plan, rows, sheets_dir, args.hf_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
