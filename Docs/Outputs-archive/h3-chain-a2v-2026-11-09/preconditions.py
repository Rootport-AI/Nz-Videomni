"""MiniMax H3 の連結生成にクリップごとの A2V を足した門 GA0〜GA3 の、前後の点検（台帳 §1-120）。

写し元: ``Docs/Outputs-archive/h3-vsf-chain-2026-11-09/preconditions.py``（動きは同じ。置き場所の名前だけを替えた）。
様式は ``Docs/Outputs-archive/h3-frag-fix-2026-11-04/preconditions.txt`` に倣う
（git・nvidia-smi・python のプロセス・``engine.worker``／``engine_h3.worker`` の孤児・AviUtl2）。

使い方（リポジトリ直下で）::

    .venv\\Scripts\\python.exe Docs\\Outputs-archive\\h3-chain-a2v-2026-11-09\\preconditions.py before
    .venv\\Scripts\\python.exe Docs\\Outputs-archive\\h3-chain-a2v-2026-11-09\\preconditions.py after

結果は標準出力と、この台本と同じフォルダの ``preconditions_<before|after>.txt`` に書く。
読むだけで、何も止めない・殺さない（判断は監督が行う）。

終了コード: 0＝問題なし、2＝AviUtl2（``aviutl2.exe``）が動いている、
3＝``engine.worker``／``engine_h3.worker`` のプロセスが残っている（AviUtl2 が無いとき）。
両方に当たるときは 2 を返す（どちらも本文に出る）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # Docs\Outputs-archive\<このフォルダ> の 3 つ上＝リポジトリ直下

WORKER_MARKERS = ("engine.worker", "engine_h3.worker")
SERVER_MARKER = "main.py"


def run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
    except Exception as exc:  # noqa: BLE001 -- 点検の失敗も本文に残す
        return f"<失敗: {type(exc).__name__}: {exc}>"
    out = (r.stdout or "").rstrip()
    if r.returncode != 0:
        out += f"\n<終了コード {r.returncode}: {(r.stderr or '').strip()[-1000:]}>"
    return out


def processes(name_like: str) -> list[dict]:
    """``Get-CimInstance Win32_Process`` で名前が ``name_like`` に合うプロセスの一覧（PID・名前・コマンド行）。
    PowerShell 自身は名前が合わないので一覧に入らない（検索の文字列を含むシェルを数えてしまう事故を避ける）。"""
    ps = (f"Get-CimInstance Win32_Process -Filter \"Name like '{name_like}'\" | "
          "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Depth 3")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    text = (r.stdout or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [{"error": text[:1000]}]
    if isinstance(data, dict):
        data = [data]
    return [{"pid": d.get("ProcessId"), "ppid": d.get("ParentProcessId"), "name": d.get("Name"),
             "cmdline": d.get("CommandLine") or ""} for d in data]


def main(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if len(argv) != 1 or argv[0] not in ("before", "after"):
        print("使い方: preconditions.py before|after", file=sys.stderr)
        return 1
    which = argv[0]
    lines: list[str] = [f"点検: {which}", f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]

    lines += ["== git branch --show-current", run(["git", "branch", "--show-current"], REPO)]
    lines += ["== git rev-parse HEAD", run(["git", "rev-parse", "HEAD"], REPO)]
    lines += ["== git log --oneline -1", run(["git", "log", "--oneline", "-1"], REPO)]
    lines += ["== git status --short (empty below = clean)", run(["git", "status", "--short"], REPO)]

    lines += ["== nvidia-smi --query-gpu=memory.used,memory.total --format=csv",
              run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"])]

    me = os.getpid()
    py = processes("%python%")
    lines += ["== python processes (Get-CimInstance; empty = none)"]
    workers: list[dict] = []
    for p in py:
        if "error" in p:
            lines.append(f"<一覧の読み取りに失敗: {p['error']}>")
            continue
        tags = []
        if p["pid"] == me:
            tags.append("この点検そのもの")
        if any(m in p["cmdline"] for m in WORKER_MARKERS):
            tags.append("ワーカー")
            if p["pid"] != me:
                workers.append(p)
        if SERVER_MARKER in p["cmdline"]:
            tags.append("main.py（バックエンド）")
        tag = f"  [{'・'.join(tags)}]" if tags else ""
        lines.append(f"pid={p['pid']} ppid={p['ppid']} {p['name']}{tag}")
        lines.append(f"    {p['cmdline']}")
    lines += ["== worker processes (engine.worker / engine_h3.worker; empty = none)"]
    for p in workers:
        lines.append(f"pid={p['pid']} ppid={p['ppid']} {p['cmdline']}")

    av = processes("aviutl2%")
    lines += ["== AviUtl2 (aviutl2.exe; empty = not running)"]
    for p in av:
        lines.append(f"pid={p.get('pid')} {p.get('name')} {p.get('cmdline') or p.get('error', '')}")

    lines += ["== nvidia-smi (full)", run(["nvidia-smi"])]

    code = 0
    if av:
        code = 2
    elif workers:
        code = 3
    verdict = {0: "問題なし", 2: "AviUtl2 が動いている（閉じてから回す）",
               3: "ワーカーのプロセスが残っている（監督が判断。この台本は止めない）"}[code]
    lines += ["", f"判定: {verdict}（終了コード {code}）"]

    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / f"preconditions_{which}.txt").write_text(text, encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
