> **このファイルは実機の `outputs/h3-te-equal-chunks-2026-11-04/gates.md`（git追跡外）のスナップショットである（2026-11-05複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# Gate table (plan §5.4)

## Gate 1: T4 (product vs stock SDPA, all rows)

| N | product.equal | differing_row_count | verdict |
|---|---|---|---|
| 513 | True | 0 | PASS |
| 544 | True | 0 | PASS |
| 575 | True | 0 | PASS |
| 576 | True | 0 | PASS |
| 998 | True | 0 | PASS |
| 1087 | True | 0 | PASS |
| 1088 | True | 0 | PASS |
| 1100 | True | 0 | PASS |
| 2078 | True | 0 | PASS |
| 3123 | True | 0 | PASS |
| 3718 | True | 0 | PASS |
| 4100 | True | 0 | PASS |
| 6148 | True | 0 | PASS |

T4: 13/13 (tail_rows_of_last_block / fold_check / position_check not judged)

## t2 (whole TE, post-fix vs the S1 pre-fix .pt, --cap)

| refs | bit_identical | tokens_match_expected | tokens | num_ooms | attention_calls | num_alloc_retries pre→post | decoder_peak_reserved_mb pre→post | encode s pre→post | verdict |
|---|---|---|---|---|---|---|---|---|---|
| pictures9 | True | True | 9258 | 0 | 0 | 0→0 | 8914.0→6912.0 | 45.33→38.49 | PASS |
| video15 | True | True | 16314 | 0 | 0 | 0→0 | 10384.0→10384.0 | 97.03→96.36 | PASS |

## Gate 2: worker, the i1080 TE shape (N=18,462, 56 frames, sdpa, --vram16)

| item | pre (S1-c) | post (S3-d) |
|---|---|---|
| exit code | 0 | 0 |
| mp4 sha256 | e5c6ac9e4c9c3bffc0dc370ab6e3184c3e7aa273ffb9b2aacf3024e945c7abbe | e5c6ac9e4c9c3bffc0dc370ab6e3184c3e7aa273ffb9b2aacf3024e945c7abbe |
| te_read.tokens | 18462 | 18462 |
| encode peak_vram_reserved_mb | 16074 | 12070 |
| encode peak_vram_mb | 9978 | 9698 |
| encode seconds (recorded only) | 152.49 | 124.24 |

- exit 0 (both): PASS
- mp4 sha256 identical: PASS
- tokens 18,462 (both): PASS
- TE stage reserved lower after the fix: PASS

## G5 (T2V fingerprint, --vram16)

- exit 0, sha256 0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9 (expected 0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9): PASS

Machine gate (pytest, S2 / S4): not read here (the supervisor's records).

**Overall: ALL PASS**
