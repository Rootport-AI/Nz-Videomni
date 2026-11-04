# -*- coding: utf-8 -*-
"""Plan §5.4 pass / fail table from the json files in this folder (Markdown to stdout and gates.md).

    python compare_gates.py

Keys (read from the scripts that write them):
  t4_n<N>.json      t4_tail_block.py: product.equal, product.differing_row_count, ok
  post_<refs>.json  t2_te_ab.py: vs_baseline.bit_identical, tokens_match_expected, allocator.num_ooms,
                    allocator.num_alloc_retries, attention_calls (0 in product mode), tokens,
                    decoder_peak_reserved_mb, seconds.encode   (pre_<refs>.json = the S1 values)
  <pre|post>_i1080.json  worker --selftest-generate: rounds[0].sha256,
                    rounds[0].done.h3.phases.encode.{peak_vram_reserved_mb, peak_vram_mb, seconds, te_read.tokens}
  g5_on.json        worker: rounds[0].sha256
  logs/<name>.meta.log  the exit code (last "exit=" line) written by common.sh run_logged
Not judged: t4 tail_rows_of_last_block / fold_check / position_check; the TE seconds are recorded only.
Exit 0 = every gate in the table passes.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NS = [513, 544, 575, 576, 998, 1087, 1088, 1100, 2078, 3123, 3718, 4100, 6148]
G5_SHA = "0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9"
TOKENS_I1080 = 18462
ENC = "rounds.0.done.h3.phases.encode."


def load(name):
    f = HERE / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def get(d, path, default=None):
    """Dotted lookup; a numeric part indexes a list."""
    for key in path.split("."):
        if isinstance(d, list) and key.isdigit():
            d = d[int(key)] if len(d) > int(key) else None
        elif isinstance(d, dict):
            d = d.get(key)
        else:
            return default
        if d is None:
            return default
    return d


def exit_code(name):
    f = HERE / "logs" / f"{name}.meta.log"
    if not f.exists():
        return None
    codes = [line.split("=", 1)[1].strip() for line in f.read_text(encoding="utf-8").splitlines()
             if line.startswith("exit=")]
    return int(codes[-1]) if codes else None


def mark(ok):
    return "PASS" if ok else "FAIL"


def main() -> int:
    out, fails = ["# Gate table (plan §5.4)", ""], 0

    out += ["## Gate 1: T4 (product vs stock SDPA, all rows)", "",
            "| N | product.equal | differing_row_count | verdict |", "|---|---|---|---|"]
    t4_ok = 0
    for n in NS:
        d = load(f"t4_n{n}.json")
        ok = bool(d) and get(d, "product.equal") is True and get(d, "product.differing_row_count") == 0
        t4_ok += ok
        out.append(f"| {n} | {get(d, 'product.equal')} | {get(d, 'product.differing_row_count')} | "
                   f"{mark(ok) if d else 'MISSING'} |")
    fails += t4_ok != len(NS)
    out += ["", f"T4: {t4_ok}/{len(NS)} (tail_rows_of_last_block / fold_check / position_check not judged)", ""]

    out += ["## t2 (whole TE, post-fix vs the S1 pre-fix .pt, --cap)", "",
            "| refs | bit_identical | tokens_match_expected | tokens | num_ooms | attention_calls | "
            "num_alloc_retries pre→post | decoder_peak_reserved_mb pre→post | encode s pre→post | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for refs in ("pictures9", "video15"):
        pre, post = load(f"pre_{refs}.json"), load(f"post_{refs}.json")
        ok = (bool(post) and get(post, "vs_baseline.bit_identical") is True
              and get(post, "tokens_match_expected") is True and get(post, "allocator.num_ooms") == 0
              and get(post, "attention_calls") == 0)
        fails += not ok
        out.append(f"| {refs} | {get(post, 'vs_baseline.bit_identical')} | {get(post, 'tokens_match_expected')} | "
                   f"{get(post, 'tokens')} | {get(post, 'allocator.num_ooms')} | {get(post, 'attention_calls')} | "
                   f"{get(pre, 'allocator.num_alloc_retries')}→{get(post, 'allocator.num_alloc_retries')} | "
                   f"{get(pre, 'decoder_peak_reserved_mb')}→{get(post, 'decoder_peak_reserved_mb')} | "
                   f"{get(pre, 'seconds.encode')}→{get(post, 'seconds.encode')} | "
                   f"{mark(ok) if post else 'MISSING'} |")
    out.append("")

    out += ["## Gate 2: worker, the i1080 TE shape (N=18,462, 56 frames, sdpa, --vram16)", ""]
    pre, post = load("pre_i1080.json"), load("post_i1080.json")
    x_pre, x_post = exit_code("s1c_worker_i1080_pre"), exit_code("s3d_worker_i1080_post")
    sha_pre, sha_post = get(pre, "rounds.0.sha256"), get(post, "rounds.0.sha256")
    tok_pre, tok_post = get(pre, ENC + "te_read.tokens"), get(post, ENC + "te_read.tokens")
    r_pre, r_post = get(pre, ENC + "peak_vram_reserved_mb"), get(post, ENC + "peak_vram_reserved_mb")
    out += ["| item | pre (S1-c) | post (S3-d) |", "|---|---|---|",
            f"| exit code | {x_pre} | {x_post} |",
            f"| mp4 sha256 | {sha_pre} | {sha_post} |",
            f"| te_read.tokens | {tok_pre} | {tok_post} |",
            f"| encode peak_vram_reserved_mb | {r_pre} | {r_post} |",
            f"| encode peak_vram_mb | {get(pre, ENC + 'peak_vram_mb')} | {get(post, ENC + 'peak_vram_mb')} |",
            f"| encode seconds (recorded only) | {get(pre, ENC + 'seconds')} | {get(post, ENC + 'seconds')} |", ""]
    checks = {
        "exit 0 (both)": x_pre == 0 and x_post == 0,
        "mp4 sha256 identical": sha_pre is not None and sha_pre == sha_post,
        "tokens 18,462 (both)": tok_pre == TOKENS_I1080 and tok_post == TOKENS_I1080,
        "TE stage reserved lower after the fix": r_pre is not None and r_post is not None and r_post < r_pre,
    }
    for k, v in checks.items():
        out.append(f"- {k}: {mark(v)}")
        fails += not v
    out.append("")

    out += ["## G5 (T2V fingerprint, --vram16)", ""]
    g5 = load("g5_on.json")
    x_g5, sha_g5 = exit_code("s3e_g5_fingerprint"), get(g5, "rounds.0.sha256")
    g5_ok = x_g5 == 0 and sha_g5 == G5_SHA
    fails += not g5_ok
    out += [f"- exit {x_g5}, sha256 {sha_g5} (expected {G5_SHA}): {mark(g5_ok)}", "",
            "Machine gate (pytest, S2 / S4): not read here (the supervisor's records).", "",
            f"**Overall: {'ALL PASS' if fails == 0 else f'{fails} gate(s) FAIL or missing'}**"]
    text = "\n".join(out) + "\n"
    (HERE / "gates.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
