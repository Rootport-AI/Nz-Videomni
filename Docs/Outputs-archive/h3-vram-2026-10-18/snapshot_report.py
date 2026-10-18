"""スナップショット（torch.cuda.memory._dump_snapshot の pickle）の集計。

使い方:  python snapshot_report.py runs/<label>/snapshot_step1.pickle [--out-prefix PATH] [--device 0]
出力:    <pickle と同じ場所>/<stem>_report.md と <stem>_report.json（--out-prefix で変更）

中身（計画 PLAN_measure.md §4）:
  (1) device_traces（確保と解放の履歴）を再生し、「割り当て合計が最大になった瞬間」を見つける。
      その瞬間に生きているブロックをセグメント（cudaMalloc で取った塊）に当て、
      セグメントごとの予約・使用中・未使用、未使用の連続した隙間の大きさの等級別の合計、
      「新しい segment_alloc のうち、直前の空きの合計が要求以上だったもの」の件数と量を出す。
  (2) その瞬間の最大の確保 20 件の大きさと呼び出し元（python のフレームの filename:line）。
  (3) 書き出し時点のセグメントの一覧（segments）。

再生の方法（torch 2.13 の torch/cuda/_memory_viz.py と MemoryViz.js の考え方に合わせた）:
  履歴は有効化した時点から（または max_entries の環のぶんだけ）しか無いので、書き出し時点の
  segments と blocks を「終わりの状態」とし、履歴を逆にたどって「履歴の頭の状態」を復元してから、
  頭から順に再生する。割り当て合計は alloc で増え free_completed で減る（_memory_viz.trace と同じ）。
  履歴の alloc / free_completed の size は「要求した大きさ」（丸める前）であることを試運転で確かめた
  （3MiB+123B の確保で size=3,145,851・ブロックは 3,146,240）。そこで占有と割り当て合計は、
  割り当て器と同じく 512 バイトの倍数に切り上げた大きさで数える（memory_stats の allocated_bytes と同じ数え方）。
  分割されずに余りごと渡されたブロックの余りは、ここでは「隙間」に数えられる（小さい誤差）。
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import pickle
import sys
from pathlib import Path

MIB = 2 ** 20
SMALL_SEGMENT = 2 * MIB          # small プール（1MiB 以下の確保）のセグメントは 2MiB
CLASSES = [("<64MiB", 0, 64 * MIB), ("64-256MiB", 64 * MIB, 256 * MIB), ("256-1024MiB", 256 * MIB, 1024 * MIB),
           ("1024-2048MiB", 1024 * MIB, 2048 * MIB), (">2048MiB", 2048 * MIB, float("inf"))]


def blk(n: int) -> int:
    """割り当て器の丸め（512 バイトの倍数に切り上げ）。"""
    return ((int(n) + 511) // 512) * 512


def mib(n) -> float:
    return round(n / MIB, 1)


def frame_str(frames, limit=4) -> str:
    """python のフレームを内側から最大 limit 個、filename:line:name で。torch の内部は省く。"""
    out = []
    for f in frames or []:
        fn = str(f.get("filename", ""))
        norm = fn.replace("\\", "/")
        if "/torch/" in norm and "/site-packages/" in norm:
            continue
        out.append(f"{os.path.basename(norm)}:{f.get('line')}:{f.get('name')}")
        if len(out) >= limit:
            break
    if not out and frames:
        f = frames[0]
        out.append(f"{os.path.basename(str(f.get('filename', '')))}:{f.get('line')}:{f.get('name')}")
    return " <- ".join(out)


class State:
    """セグメント（addr -> size, stream）と生きているブロック（addr -> size, stream, frames）。"""

    def __init__(self):
        self.segs: dict[int, dict] = {}
        self.live: dict[int, dict] = {}

    def copy(self):
        s = State()
        s.segs = {k: dict(v) for k, v in self.segs.items()}
        s.live = {k: dict(v) for k, v in self.live.items()}
        return s

    def allocated(self) -> int:
        return sum(b["size"] for b in self.live.values())

    def reserved(self) -> int:
        return sum(s["size"] for s in self.segs.values())


def final_state(snap, device: int) -> State:
    st = State()
    for seg in snap.get("segments", []):
        if seg.get("device", device) != device:
            continue
        st.segs[seg["address"]] = {"size": seg["total_size"], "stream": seg.get("stream", 0),
                                   "type": seg.get("segment_type")}
        addr = seg["address"]
        for b in seg.get("blocks", []):
            if b.get("state") in ("active_allocated", "active_awaiting_free"):
                st.live[addr] = {"size": b["size"], "stream": seg.get("stream", 0), "frames": b.get("frames", []),
                                 "requested": b.get("requested_size")}
            addr += b["size"]
    return st


def unwind(st: State, trace) -> tuple[State, list[str]]:
    """終わりの状態から履歴を逆にたどり、履歴の頭の状態を返す。"""
    s = st.copy()
    notes = []
    for e in reversed(trace):
        a = e.get("action")
        if a == "alloc":
            if s.live.pop(e["addr"], None) is None:
                notes.append(f"unwind: alloc {e['addr']:#x} not live at end")
        elif a == "free_completed":
            s.live[e["addr"]] = {"size": blk(e["size"]), "stream": e.get("stream", 0), "frames": e.get("frames", []),
                                 "requested": e["size"]}
        elif a == "segment_alloc":
            s.segs.pop(e["addr"], None)
        elif a == "segment_free":
            s.segs[e["addr"]] = {"size": e["size"], "stream": e.get("stream", 0), "type": None}
    return s, notes[:20]


def apply(s: State, e) -> None:
    a = e.get("action")
    if a == "alloc":
        s.live[e["addr"]] = {"size": blk(e["size"]), "stream": e.get("stream", 0), "frames": e.get("frames", []),
                             "requested": e["size"]}
    elif a == "free_completed":
        s.live.pop(e["addr"], None)
    elif a == "segment_alloc":
        s.segs[e["addr"]] = {"size": e["size"], "stream": e.get("stream", 0), "type": None}
    elif a == "segment_free":
        s.segs.pop(e["addr"], None)


def is_small(seg) -> bool:
    return seg.get("type") == "small" or (seg.get("type") is None and seg["size"] == SMALL_SEGMENT)


def layout(s: State) -> list[dict]:
    """セグメントごとに、使用中のブロックと連続した隙間を並べる。"""
    starts = sorted(s.segs)
    per = {a: [] for a in starts}
    orphan = 0
    for addr, b in s.live.items():
        i = bisect.bisect_right(starts, addr) - 1
        if i >= 0 and addr < starts[i] + s.segs[starts[i]]["size"]:
            per[starts[i]].append((addr, b))
        else:
            orphan += b["size"]
    out = []
    for a in starts:
        seg = s.segs[a]
        blocks = sorted(per[a], key=lambda x: x[0])
        gaps, cur, used = [], a, 0
        for addr, b in blocks:
            if addr > cur:
                gaps.append(addr - cur)
            cur = max(cur, addr + b["size"])
            used += b["size"]
        end = a + seg["size"]
        if end > cur:
            gaps.append(end - cur)
        out.append({"addr": a, "size": seg["size"], "stream": seg["stream"], "small": is_small(seg),
                    "used": used, "free": seg["size"] - used, "gaps": gaps, "nblocks": len(blocks)})
    if orphan:
        out.append({"addr": None, "size": 0, "stream": None, "small": False, "used": orphan, "free": -orphan,
                    "gaps": [], "nblocks": -1, "note": "セグメントに当たらないブロック（履歴の欠けの疑い）"})
    return out


def classify(gaps) -> dict:
    res = {name: {"count": 0, "mib": 0.0} for name, _, _ in CLASSES}
    for g in gaps:
        for name, lo, hi in CLASSES:
            if lo <= g < hi:
                res[name]["count"] += 1
                res[name]["mib"] = round(res[name]["mib"] + g / MIB, 1)
                break
    return res


def analyse(snap, device: int = 0) -> dict:
    traces = snap.get("device_traces", [])
    trace = traces[device] if device < len(traces) else []
    end = final_state(snap, device)
    head, notes = unwind(end, trace)

    # 1 回目の再生: 割り当て合計が最大になった位置と、新しいセグメントの確保の時点の空き
    s = head.copy()
    alloc_total = s.allocated()
    best, best_i = alloc_total, -1
    seg_events = []
    counts = {}
    for i, e in enumerate(trace):
        a = e.get("action")
        counts[a] = counts.get(a, 0) + 1
        if a == "segment_alloc":
            # 要求＝この segment_alloc を起こした次の alloc（同じストリーム）
            req = None
            for e2 in trace[i + 1:i + 50]:
                if e2.get("action") == "alloc" and e2.get("stream", 0) == e.get("stream", 0):
                    req = blk(e2["size"])
                    break
            lay = layout(s)
            same = [L for L in lay if L["addr"] is not None and L["stream"] == e.get("stream", 0)]
            big = [L for L in same if not L["small"]]
            free_total = sum(L["free"] for L in big)
            max_gap = max((g for L in big for g in L["gaps"]), default=0)
            need = req if req is not None else e["size"]
            seg_events.append({
                "index": i, "segment_mib": mib(e["size"]), "request_mib": mib(req) if req is not None else None,
                "free_total_before_mib": mib(free_total), "max_gap_before_mib": mib(max_gap),
                "free_total_was_enough": free_total >= need and not (need <= 1 * MIB),
                "frames": frame_str(e.get("frames") or (trace[i + 1].get("frames") if i + 1 < len(trace) else [])),
            })
        apply(s, e)
        if a in ("alloc", "free_completed"):
            alloc_total += blk(e["size"]) if a == "alloc" else -blk(e["size"])
            if alloc_total > best:
                best, best_i = alloc_total, i

    # 2 回目の再生: その瞬間の状態
    s = head.copy()
    for e in trace[:best_i + 1]:
        apply(s, e)
    lay = layout(s)
    all_gaps = [g for L in lay if L["addr"] is not None and not L["small"] for g in L["gaps"]]
    top = sorted(s.live.items(), key=lambda kv: -kv[1]["size"])[:20]
    enough = [x for x in seg_events if x["free_total_was_enough"]]

    peak = {
        "trace_index": best_i,
        "trace_entries": len(trace),
        "allocated_mib": mib(s.allocated()),
        "reserved_mib": mib(s.reserved()),
        "unused_reserved_mib": mib(s.reserved() - s.allocated()),
        "segments": len(s.segs),
        "unused_gap_classes_large_pool": classify(all_gaps),
        "unused_small_pool_mib": mib(sum(L["free"] for L in lay if L["small"])),
        "largest_gap_mib": mib(max(all_gaps, default=0)),
        "per_segment": [
            {"addr": hex(L["addr"]) if L["addr"] is not None else None, "stream": L["stream"],
             "pool": "small" if L["small"] else "large", "reserved_mib": mib(L["size"]), "used_mib": mib(L["used"]),
             "unused_mib": mib(L["free"]), "largest_gap_mib": mib(max(L["gaps"], default=0)),
             "blocks": L["nblocks"], **({"note": L["note"]} if "note" in L else {})}
            for L in sorted(lay, key=lambda L: -L["size"])
        ],
        "top20_live": [{"size_mib": mib(b["size"]), "addr": hex(a), "stream": b["stream"],
                        "frames": frame_str(b.get("frames"))} for a, b in top],
    }
    at_dump = {
        "segments": len(end.segs),
        "reserved_mib": mib(end.reserved()),
        "allocated_mib": mib(end.allocated()),
        "list": [
            {"addr": hex(seg["address"]), "total_mib": mib(seg["total_size"]),
             "allocated_mib": mib(seg.get("allocated_size", 0)), "active_mib": mib(seg.get("active_size", 0)),
             "type": seg.get("segment_type"), "stream": seg.get("stream", 0), "blocks": len(seg.get("blocks", [])),
             "largest_inactive_mib": mib(max((b["size"] for b in seg.get("blocks", []) if b.get("state") == "inactive"),
                                             default=0))}
            for seg in sorted(snap.get("segments", []), key=lambda x: -x["total_size"])
            if seg.get("device", device) == device
        ],
    }
    ooms = [{"index": i, "size_mib": mib(e.get("size", 0)), "device_free_mib": mib(e.get("device_free", 0)),
             "frames": frame_str(e.get("frames"))}
            for i, e in enumerate(trace) if e.get("action") == "oom"]
    return {
        "action_counts": counts,
        "head_state": {"segments": len(head.segs), "reserved_mib": mib(head.reserved()),
                       "allocated_mib": mib(head.allocated())},
        "unwind_notes": notes,
        "peak": peak,
        "segment_allocs": {
            "count": len(seg_events),
            "free_total_was_enough_count": len(enough),
            "free_total_was_enough_mib": round(sum(x["segment_mib"] for x in enough), 1),
            "rule": "large プールで同じストリームの空きの合計 ≥ 要求（次の alloc の大きさ）なのに新しいセグメントを確保した",
            "events_enough": enough[:200],
        },
        "oom_events": ooms,
        "at_dump": at_dump,
    }


def to_markdown(r: dict, src: str) -> str:
    p = r["peak"]
    L = [f"# スナップショットの集計: `{src}`", ""]
    L.append(f"- 履歴の件数: {p['trace_entries']}（種類別 {r['action_counts']}）")
    L.append(f"- 履歴の頭の状態（逆にたどって復元）: セグメント {r['head_state']['segments']} 個・"
             f"予約 {r['head_state']['reserved_mib']} MiB・割り当て {r['head_state']['allocated_mib']} MiB")
    if r["unwind_notes"]:
        L.append(f"- 逆にたどる際の注意（先頭 20 件）: {r['unwind_notes']}")
    L += ["", "## (1) 割り当て合計が最大になった瞬間", "",
          f"- 位置: 履歴の {p['trace_index']} 番目",
          f"- 割り当て {p['allocated_mib']} MiB／予約 {p['reserved_mib']} MiB／未使用の予約 {p['unused_reserved_mib']} MiB"
          f"（セグメント {p['segments']} 個・最大の連続した隙間 {p['largest_gap_mib']} MiB・small プールの未使用 {p['unused_small_pool_mib']} MiB）",
          "", "未使用の連続した隙間の大きさの等級別（large プール）:", "", "| 等級 | 個数 | 合計 MiB |", "|---|---:|---:|"]
    for name, v in p["unused_gap_classes_large_pool"].items():
        L.append(f"| {name} | {v['count']} | {v['mib']} |")
    sa = r["segment_allocs"]
    L += ["", f"新しいセグメントの確保（segment_alloc）: {sa['count']} 件。うち「直前の空きの合計が要求以上だった」もの "
              f"{sa['free_total_was_enough_count']} 件・{sa['free_total_was_enough_mib']} MiB（規則: {sa['rule']}）", ""]
    if sa["events_enough"]:
        L += ["| 履歴の位置 | セグメント MiB | 要求 MiB | 直前の空きの合計 MiB | 直前の最大の隙間 MiB | 呼び出し元 |",
              "|---:|---:|---:|---:|---:|---|"]
        for x in sa["events_enough"][:40]:
            L.append(f"| {x['index']} | {x['segment_mib']} | {x['request_mib']} | {x['free_total_before_mib']} | "
                     f"{x['max_gap_before_mib']} | `{x['frames']}` |")
        L.append("")
    L += ["セグメントごと（その瞬間。大きい順に先頭 40 個）:", "",
          "| 番地 | プール | ストリーム | 予約 MiB | 使用中 MiB | 未使用 MiB | 最大の隙間 MiB | ブロック数 |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in p["per_segment"][:40]:
        L.append(f"| {s['addr']} | {s['pool']} | {s['stream']} | {s['reserved_mib']} | {s['used_mib']} | "
                 f"{s['unused_mib']} | {s['largest_gap_mib']} | {s['blocks']} |")
    L += ["", "## (2) その瞬間の最大の確保 20 件", "", "| MiB | ストリーム | 呼び出し元（内側から） |", "|---:|---:|---|"]
    for t in p["top20_live"]:
        L.append(f"| {t['size_mib']} | {t['stream']} | `{t['frames']}` |")
    if r["oom_events"]:
        L += ["", "## OOM（メモリ不足）の記録", ""]
        for o in r["oom_events"]:
            L.append(f"- 履歴の {o['index']} 番目: 要求 {o['size_mib']} MiB・CUDA の空き {o['device_free_mib']} MiB `{o['frames']}`")
    d = r["at_dump"]
    L += ["", "## (3) 書き出し時点のセグメントの一覧", "",
          f"セグメント {d['segments']} 個・予約 {d['reserved_mib']} MiB・割り当て {d['allocated_mib']} MiB", "",
          "| 番地 | 種別 | ストリーム | 予約 MiB | 使用中 MiB | active MiB | 最大の空きブロック MiB | ブロック数 |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in d["list"][:60]:
        L.append(f"| {s['addr']} | {s['type']} | {s['stream']} | {s['total_mib']} | {s['allocated_mib']} | "
                 f"{s['active_mib']} | {s['largest_inactive_mib']} | {s['blocks']} |")
    if len(d["list"]) > 60:
        L.append(f"| …ほか {len(d['list']) - 60} 個（JSON を参照） | | | | | | | |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="スナップショット（pickle）の集計")
    ap.add_argument("pickle")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out-prefix", default=None, help="既定は <pickle の stem>_report")
    a = ap.parse_args(argv)
    src = Path(a.pickle).resolve()
    with src.open("rb") as fh:
        snap = pickle.load(fh)
    r = analyse(snap, a.device)
    prefix = Path(a.out_prefix) if a.out_prefix else src.with_name(src.stem + "_report")
    prefix.with_suffix(".json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(r, src.name)
    prefix.with_suffix(".md").write_text(md, encoding="utf-8")
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
