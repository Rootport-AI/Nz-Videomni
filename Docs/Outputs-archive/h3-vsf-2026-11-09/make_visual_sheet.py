"""目視の実験 10 本の表（``VISUAL_TABLE.md``）を、``summary.json`` と ``visual\\<名前>\\metadata.json`` から書く補助。

``h3_vsf_gate.py --mode visual`` を回した後に、監督が実行する（GPU もサーバーも使わない。読むだけ）::

    .venv\\Scripts\\python.exe Docs\\Outputs-archive\\h3-vsf-2026-11-09\\make_visual_sheet.py

埋める欄は README の「目視の実験の 10 本」の表と同じ（``m_neg``・``m_text``・秒・VRAM の山）。
metadata.json があればそちらを正とし、無ければ summary.json の記録を使う。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = [f"{s}{i}" for s in ("T", "B") for i in range(5)]
SCALES = ["off", "1.5", "3", "5", "8"]
PAIRS = [
    ("0", "1", "off 対 1.5", "効くか"),
    ("1", "3", "1.5 対 5", "つまみが効くか"),
    ("3", "4", "5 対 8", "崩れや逆転が出るか"),
]


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def fmt(x, digits: int = 4) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def row_for(label: str, job: dict) -> dict:
    meta = load_json(HERE / "visual" / label / "metadata.json")
    h3 = meta.get("h3") if isinstance(meta.get("h3"), dict) else {}
    vsf = h3.get("vsf") if "vsf" in h3 else meta.get("vsf", job.get("vsf"))
    ms = job.get("metadata_summary") or {}
    vram_opt = meta.get("vram_optimization") or {}
    return {
        "label": label,
        "status": job.get("status") or ("completed" if meta else "未実行"),
        "note": job.get("note") or "–",
        "m_neg": (vsf or {}).get("m_neg") if isinstance(vsf, dict) else None,
        "m_text": (vsf or {}).get("m_text") if isinstance(vsf, dict) else None,
        "rows": (vsf or {}).get("rows") if isinstance(vsf, dict) else None,
        "gen_sec": meta.get("generation_time_seconds", ms.get("generation_time_seconds")),
        "wall_sec": job.get("seconds_wall"),
        "peak_reserved": meta.get("peak_vram_reserved_mb", ms.get("peak_vram_reserved_mb")),
        "peak_alloc": vram_opt.get("peak_vram_mb", ms.get("peak_vram_mb")),
        "attention": meta.get("attention_used", ms.get("attention_used")),
        "sha": (job.get("sha256") or "")[:12] or "–",
        "error": (job.get("error") or "").replace("|", "\\|").replace("\n", " ")[:160],
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    summary = load_json(HERE / "summary.json")
    jobs = summary.get("jobs") or {}
    rows = {label: row_for(label, jobs.get(label) or {}) for label in LABELS}

    lines = [
        "# 目視の実験 10 本の表（make_visual_sheet.py が書いた）",
        "",
        "共通: 1344×768・56 フレーム・24fps・seed 42・注意の計算 sage・VRAM 16GB mode・shift 12/3。"
        "T は Turbo LoRA 0.8・4 ステップ、B は LoRA なし・20 ステップ。負の文は `black hair, long hair, eyeglasses, cars`。",
        "",
        "`m_neg` は負の行が注意の配分を奪った割合、`m_text` は正の文の行に乗った割合（ステップ 0・ブロック 0・映像の行 256 本の平均）。"
        "sdpa の softmax の定義で計算した値で、sage の実際の重みとはわずかに違いうる。",
        "",
        "| 名前 | 条件 | 状態 | m_neg | m_text | 負の行（始, 終） | 生成の秒 | 待ちの秒 | VRAM の山（予約 MiB） | VRAM の山（割り当て MiB） | 注意の計算 | sha256（先頭） | エラー |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label in LABELS:
        r = rows[label]
        lines.append(
            f"| {label} | {r['note']} | {r['status']} | {fmt(r['m_neg'])} | {fmt(r['m_text'])} | {fmt(r['rows'])} | "
            f"{fmt(r['gen_sec'], 1)} | {fmt(r['wall_sec'], 1)} | {fmt(r['peak_reserved'])} | {fmt(r['peak_alloc'])} | "
            f"{fmt(r['attention'])} | `{r['sha']}` | {r['error'] or '–'} |")

    lines += ["", "## 比較の対", "",
              "| 対 | 見ること | 系列 | 左 | 右 | 左の m_neg | 右の m_neg |",
              "|---|---|---|---|---|---|---|"]
    for series in ("T", "B"):
        for a, b, name, what in PAIRS:
            la, lb = f"{series}{a}", f"{series}{b}"
            lines.append(f"| {name} | {what} | {series} | {la} | {lb} | {fmt(rows[la]['m_neg'])} | "
                         f"{fmt(rows[lb]['m_neg'])} |")
    for i, s in enumerate(SCALES):
        lines.append(f"| T 対 B（{s}） | ステップ数で変わるか | – | T{i} | B{i} | {fmt(rows[f'T{i}']['m_neg'])} | "
                     f"{fmt(rows[f'B{i}']['m_neg'])} |")
    lines.append("")

    out = HERE / "VISUAL_TABLE.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"書いた: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
