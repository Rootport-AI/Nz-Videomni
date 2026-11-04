# -*- coding: utf-8 -*-
"""S3-a summary: one row per t4_n<N>.json (product.equal / product.differing_row_count / ok).

    python summarize_t4.py            (Markdown to stdout; s3a_t4_all.sh tees it to t4_summary.md)

tail_rows_of_last_block, fold_check and position_check are old-rule fields and are NOT judged
(position_check measures a 4-row block, which never matches stock SDPA: VERIFICATION_LOG §182.6).
Exit 0 = all 13 present and passing.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NS = [513, 544, 575, 576, 998, 1087, 1088, 1100, 2078, 3123, 3718, 4100, 6148]


def main() -> int:
    bad = 0
    print("| N | product.equal | differing_row_count | max_abs_diff | ok | verdict |")
    print("|---|---|---|---|---|---|")
    for n in NS:
        f = HERE / f"t4_n{n}.json"
        if not f.exists():
            print(f"| {n} | - | - | - | - | MISSING |")
            bad += 1
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = d.get("product") or {}
        passed = p.get("equal") is True and p.get("differing_row_count") == 0 and d.get("ok") is True
        bad += not passed
        note = "" if not d.get("error") else f" ({d['error']})"
        print(f"| {n} | {p.get('equal')} | {p.get('differing_row_count')} | {p.get('max_abs_diff')} "
              f"| {d.get('ok')} | {'PASS' if passed else 'FAIL'}{note} |")
    print()
    print(f"T4: {len(NS) - bad}/{len(NS)} pass (not judged: tail_rows_of_last_block, fold_check, position_check)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
