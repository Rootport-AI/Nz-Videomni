# -*- coding: utf-8 -*-
"""gates.md（compare_gates.py の出力）の各値を、元の json（t4_n*.json・post_/pre_*.json・*_i1080.json・g5）と突き合わせる。
あわせて第 1 版（v1_equal_split/）の不一致の値と、主メモリのコミットの山を出す。読むだけ。
  python -I -X utf8 analysis/check_gates_te.py > analysis/check_gates_te_output.txt"""
import io, json, os, re, csv, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.dirname(HERE)
OUTS = os.path.dirname(G)
J = lambda p: json.load(io.open(os.path.join(G, p), encoding="utf-8"))
md = io.open(os.path.join(G, "gates.md"), encoding="utf-8").read()
checks = []
def chk(item, pv, rv):
    checks.append((item, pv, rv, str(pv) == str(rv)))

# T4
for m in re.finditer(r"^\| (\d+) \| (True|False) \| (\d+) \| (PASS|FAIL) \|$", md, re.M):
    n = m.group(1); d = J("t4_n%s.json" % n)
    chk("T4 N=%s equal" % n, m.group(2), str(d["product"]["equal"]))
    chk("T4 N=%s differing_row_count" % n, m.group(3), str(d["product"]["differing_row_count"]))
    chk("T4 N=%s max_abs_diff" % n, "0.0", str(d["product"]["max_abs_diff"]))
# t2
for m in re.finditer(r"^\| (pictures9|video15) \| (\w+) \| (\w+) \| (\d+) \| (\d+) \| (\d+) \| (\d+)→(\d+) \| ([\d.]+)→([\d.]+) \| ([\d.]+)→([\d.]+) \| (\w+) \|$", md, re.M):
    r = m.group(1); pre = J("pre_%s.json" % r); post = J("post_%s.json" % r)
    chk(r + " bit_identical", m.group(2), str(post["vs_baseline"]["bit_identical"]))
    chk(r + " tokens_match_expected", m.group(3), str(post["tokens_match_expected"]))
    chk(r + " tokens", m.group(4), str(post["tokens"]))
    chk(r + " num_ooms", m.group(5), str(post["allocator"]["num_ooms"]))
    chk(r + " attention_calls", m.group(6), str(post["attention_calls"]))
    chk(r + " retries pre", m.group(7), str(pre["allocator"]["num_alloc_retries"]))
    chk(r + " retries post", m.group(8), str(post["allocator"]["num_alloc_retries"]))
    chk(r + " reserved pre", m.group(9), str(pre["decoder_peak_reserved_mb"]))
    chk(r + " reserved post", m.group(10), str(post["decoder_peak_reserved_mb"]))
    chk(r + " encode s pre", m.group(11), str(pre["seconds"]["encode"]))
    chk(r + " encode s post", m.group(12), str(post["seconds"]["encode"]))
# worker
pre, post = J("pre_i1080.json"), J("post_i1080.json")
def enc(d): return d["rounds"][0]["done"]["h3"]["phases"]["encode"]
for key, lab in (("peak_vram_reserved_mb", "encode peak_vram_reserved_mb"), ("peak_vram_mb", "encode peak_vram_mb"), ("seconds", r"encode seconds \(recorded only\)")):
    m = re.search(r"^\| %s \| ([\d.]+) \| ([\d.]+) \|$" % lab, md, re.M)
    chk("worker pre " + key, m.group(1), str(enc(pre)[key])); chk("worker post " + key, m.group(2), str(enc(post)[key]))
m = re.search(r"^\| te_read.tokens \| (\d+) \| (\d+) \|$", md, re.M)
chk("worker pre tokens", m.group(1), str(enc(pre)["te_read"]["tokens"])); chk("worker post tokens", m.group(2), str(enc(post)["te_read"]["tokens"]))
m = re.search(r"^\| mp4 sha256 \| ([0-9a-f]{64}) \| ([0-9a-f]{64}) \|$", md, re.M)
for k, f in ((1, "pre_i1080.mp4"), (2, "post_i1080.mp4")):
    h = hashlib.sha256(open(os.path.join(G, f), "rb").read()).hexdigest()
    chk("sha256 " + f + "（ファイルを読み直して計算）", m.group(k), h)
h = hashlib.sha256(open(os.path.join(G, "g5_on.mp4"), "rb").read()).hexdigest()
chk("G5 sha256（ファイルを読み直して計算）", "0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9", h)
g5 = J("g5_on.json"); e = g5["rounds"][0]["done"]["h3"]["phases"]["encode"]
print("G5: tokens %s・TE 予約／山 %s／%s MiB・TE 秒 %s" % (e["te_read"]["tokens"], e["peak_vram_reserved_mb"], e["peak_vram_mb"], e["seconds"]))
for d, nm in ((pre, "pre"), (post, "post")):
    x = enc(d)
    print("worker %s: read_s %s・encode_s(te_read) %s・commit_process_peak %s GiB" % (nm, x["te_read"]["read_seconds"], x["te_read"]["encode_seconds"], x["commit_process_peak_gib"]))
for c in checks:
    print(("OK  " if c[3] else "NG  ") + " | ".join(str(x) for x in c[:3]))
print("\n項目数 %d・不一致 %d" % (len(checks), sum(not c[3] for c in checks)))

print("\n## 第 1 版（v1_equal_split/）の t2")
for r in ("pictures9", "video15"):
    d = J("v1_equal_split/post_%s.json" % r); v = d["vs_baseline"]
    print(r, "tokens", d["tokens"], "bit_identical", v["bit_identical"], "tokens_not_identical", v["tokens_not_identical"],
          "max_abs_diff", v["max_abs_diff"], "mean_abs_diff", round(v["mean_abs_diff"], 4), "min_cos", round(v["min_token_cosine"], 4), "@", v["min_token_cosine_index"],
          "reserved", d["decoder_peak_reserved_mb"], "alloc", d["decoder_peak_alloc_mb"], "encode_s", d["seconds"]["encode"])
for r in ("pictures9", "video15"):
    d = J("pre_%s.json" % r)
    print("pre", r, "reserved", d["decoder_peak_reserved_mb"], "alloc", d["decoder_peak_alloc_mb"], "pre_decoder reserved", d["hooks"]["pre_decoder_peak_reserved_mb"], "min_free_pre", d["hooks"]["min_free_pre_mb"])
for r in ("pictures9", "video15"):
    d = J("post_%s.json" % r)
    print("post", r, "reserved", d["decoder_peak_reserved_mb"], "alloc", d["decoder_peak_alloc_mb"], "min_free_pre", d["hooks"]["min_free_pre_mb"], "min_cos", d["vs_baseline"]["min_token_cosine"])
t4v1 = [J("v1_equal_split/t4_n%d.json" % n)["product"] for n in (513, 544, 575, 576, 998, 1087, 1088, 1100, 2078, 3123, 3718, 4100, 6148)]
print("第 1 版 T4: equal 真 %d／13・differing_row_count の和 %d" % (sum(p["equal"] for p in t4v1), sum(p["differing_row_count"] for p in t4v1)))

print("\n## 主メモリのコミットの山")
for p in (os.path.join(G, "commit_log.csv"), os.path.join(OUTS, "comfort-calib-h3-2026-11-04-te", "commit_log.csv")):
    rows = list(csv.DictReader(io.open(p, encoding="utf-8-sig")))
    mx = max(rows, key=lambda r: float(r["pct"]))
    print(os.path.relpath(p, OUTS), "行数", len(rows), "山", mx)
