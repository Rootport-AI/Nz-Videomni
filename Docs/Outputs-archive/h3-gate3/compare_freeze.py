"""3-G1: compare `uv pip freeze` of the installer-built .venv-engine-h3 with the
committed freeze (plan s3.1 / review M-2).

Both sides are reduced to a set of (normalised name, version). A direct-URL line
`name @ https://.../name-<ver>-....whl` is reduced to name + the version parsed
from the wheel filename (URL-decoded), so sageattention compares by name and
version only. Writes a JSON report.

usage: python compare_freeze.py <committed_freeze.txt> <new_freeze.txt> <report.json>
"""
import json
import re
import sys
import urllib.parse


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse(path: str) -> dict:
    out = {}
    for raw in open(path, encoding="utf-8-sig").read().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " @ " in line:
            name, url = line.split(" @ ", 1)
            fname = urllib.parse.unquote(url.rsplit("/", 1)[-1])
            parts = fname.split("-")
            ver = parts[1] if fname.endswith(".whl") and len(parts) > 2 else url
            out[norm(name)] = ver
        elif "==" in line:
            name, ver = line.split("==", 1)
            out[norm(name)] = ver.strip()
        else:
            out[norm(line)] = "?"
    return out


def main() -> int:
    old, new, report = sys.argv[1:4]
    a, b = parse(old), parse(new)
    only_old = sorted(f"{k}=={a[k]}" for k in a.keys() - b.keys())
    only_new = sorted(f"{k}=={b[k]}" for k in b.keys() - a.keys())
    changed = sorted(f"{k}: {a[k]} -> {b[k]}" for k in a.keys() & b.keys() if a[k] != b[k])
    res = {
        "committed_count": len(a),
        "new_count": len(b),
        "only_in_committed": only_old,
        "only_in_new": only_new,
        "version_changed": changed,
        "identical": not (only_old or only_new or changed),
    }
    json.dump(res, open(report, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
