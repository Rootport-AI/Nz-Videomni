# このファイルは `outputs/h3-frag-fix-2026-11-04/analysis/judge_segments.py` の複写です（2026-11-04）。

"""Judge 4d (i)-(iv) from analyze_snapshot.py's <prefix>.json (segments[] and blocks[]). Read-only; prints markdown."""
import json, sys
M = 2 ** 20
d = json.load(open(sys.argv[1], encoding="utf-8"))
segs = d["segments"]
print("segment keys:", sorted(segs[0].keys()) if segs else None)
print("block keys:", sorted({k for s in segs for b in s["blocks"] for k in b}))
tot = sum(s["total_size"] for s in segs); al = sum(s["allocated_size"] for s in segs)
print(f"\nsegments={len(segs)} reserved={tot/M:.1f}MiB allocated={al/M:.1f}MiB reserved_free={(tot-al)/M:.1f}MiB\n")
def used_blocks(s): return [b for b in s["blocks"] if b["state"] != "inactive"]
print("| # | アドレス | 大きさ MiB | 使用中 MiB | 空き MiB | 最大の連続した空き MiB | 使用中の塊の数 | 起点 (origin_key) | 種別 |")
print("|---|---|---|---|---|---|---|---|---|")
for i, s in enumerate(segs):
    print(f"| {i} | {s['address']} | {s['total_size']/M:.1f} | {s['allocated_size']/M:.1f} | {s['free']/M:.1f} | {s['max_free_block']/M:.1f} | {len(used_blocks(s))} | `{s['origin_key']}` | {s['segment_type']} |")
print("\n(i) 3,300-3,500MiB segments:", [s["address"] for s in segs if 3300 * M <= s["total_size"] <= 3500 * M] or "none")
te = [s["address"] + " " + s["origin_key"] for s in segs if "sft_load.py:174" in s["origin_key"] or "text_encoder.py" in s["origin_key"]
      or any("text_encoder.py" in (c if isinstance(c, str) else json.dumps(c)) for c in (s.get("origin_chain") or []))]
print("(ii) TE-stage segments (origin_key or origin_chain mentions sft_load.py:174 / text_encoder.py):", te or "none")
for s in segs:
    if s["total_size"] == 20 * M:
        ub = used_blocks(s)
        print(f"(iii) 20MiB segment {s['address']} origin={s['origin_key']!r} creator_source={s['creator_source']} used blocks:",
              [(b["size"], b["state"], b.get("origin")) for b in ub])
print("\n(iv) segments with max contiguous free > 100MiB")
print("| アドレス | 大きさ MiB | 使用中 MiB | 空き MiB | 最大の連続した空き MiB | 起点 | 使用中の塊（大きさ MiB・起点） |")
print("|---|---|---|---|---|---|---|")
for s in segs:
    if s["max_free_block"] > 100 * M:
        ub = "; ".join(f"{b['size_mib']}@{b['offset_mib']} `{b.get('origin')}`" for b in used_blocks(s)) or "なし"
        print(f"| {s['address']} | {s['total_size']/M:.1f} | {s['allocated_size']/M:.1f} | {s['free']/M:.1f} | {s['max_free_block']/M:.1f} | `{s['origin_key']}` | {ub} |")
print("\nS·U-class (1,700-1,760MiB) segments, all blocks:")
for s in segs:
    if 1700 * M <= s["total_size"] <= 1760 * M:
        print(s["address"], s["origin_key"], [(b["size_mib"], b["offset_mib"], b["state"], b.get("origin") or b.get("last_alloc_here")) for b in s["blocks"]])
print("\nAll segments, blocks (size_mib, offset_mib, state, origin/last_alloc_here):")
for s in segs:
    print(f"- {s['address']} {s['total_size']/M:.1f}MiB:", [(b["size_mib"], b["offset_mib"], b["state"], b.get("origin") or b.get("last_alloc_here")) for b in s["blocks"]])
