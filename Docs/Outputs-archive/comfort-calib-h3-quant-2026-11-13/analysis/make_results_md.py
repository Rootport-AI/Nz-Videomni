# -*- coding: utf-8 -*-
"""RESULTS.md を analysis/results.json（check_results_quant.py の出力）から組み立てる。
表の数値はすべて results.json から写す（手で打たない）。11-04 の int8 の値は、記録簿 §196 の数値を
定数で持つ（11-04 の runs/<label>/run.json の値。check_results_quant.py の INT8_REF と同じ出所）。

  set PYTHONUTF8=1 & ..\\..\\..\\.venv\\Scripts\\python.exe -B make_results_md.py
"""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = json.load(io.open(os.path.join(HERE, "results.json"), encoding="utf-8"))
P = {p["label"]: p for p in R["points"]}
C = R["conclusions"]
PRED = R["predictions"]

# 11-04（int8・FL2VA default・sage・4 ステップ・K=1）の run.json の値（記録簿 §196）
I04 = {
    "s1080_328": dict(S=203092, res=12964, alloc=12169, retries=9, sps=297.68),
    "s1080_345": dict(S=213348, res=13486, alloc=12669, retries=9, sps=326.248),
    "s1440_192": dict(S=217798, res=13764, alloc=12905, retries=9, sps=338.335),
    "b1080": dict(S=38984, res=5866, alloc=4340, sps=19.902),
    "b1440": dict(S=70144, res=8214, alloc=5709, sps=48.405),
}
I04_CONTROL = "2.99 GiB／6.94 GiB／3.84 GiB／13.67 GiB・free 3.89 GiB"  # 11-04 s1080_362（runs/s1080_362/run.json）


def n(v, d=None):
    if v is None:
        return "—"
    if isinstance(v, float):
        if d is not None:
            return format(v, ",.%df" % d)
        return format(int(v), ",") if v == int(v) else format(v, ",.3f").rstrip("0")
    if isinstance(v, int):
        return format(v, ",")
    return str(v)


def oom_cells(p):
    o = p.get("OOM")
    if not o:
        return "—", "—", "—"
    t = (o.get("数") or {}).get("表記") or {}
    four = "%s／%s／%s／%s・free %s" % (t.get("要求"), t.get("割り当て"), t.get("予約済みの空き"), t.get("許可"), t.get("free"))
    return o["場所"], "%s／%s" % (o["落ちたステップ"], o["total_steps"]), four


def delta(p):
    if p["役割"] != "点" or p["status"] != "completed":
        return "—"
    return "／".join(n(p["窓"][k].get("delta_mb")) for k in ("denoise", "decode"))


ROLE_NOTE = {"b1080_w4a8": "事故（二重起動）・共有メモリの標本なし・残置。木の基準点は b1080_w4a8_r2"}


def point_table(fmt):
    rows = [p for p in R["points"] if p["形式"] == fmt]
    o = ["| ラベル | 役割 | 形 | 判定 | S | cap | 予約の山（ノイズ除去／復号） | 割り当ての山（ノイズ除去／復号） | 余裕 | retries | 1 ステップの秒 | Δ共有（ノイズ除去／復号 MB） | OOM の場所 | 落ちたステップ | OOM の 4 つの数（要求／割り当て／予約済みの空き／許可） | 快適の回数 |",
         "|" + "---|" * 16]
    for p in rows:
        place, step, four = oom_cells(p)
        role = p["役割"] + ("（復帰の暖機）" if "_rec" in p["label"] else "")
        if p["label"] in ROLE_NOTE:
            role += "・" + ROLE_NOTE[p["label"]]
        sps = p["1ステップ秒_meta"] if p["1ステップ秒_meta"] is not None else p["1ステップ秒_log"]
        o.append("| %s | %s | %d×%d×%d | %s | %s | %s | %s／%s | %s／%s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            p["label"], role, p["W"], p["H"], p["F"], p["判定"], n(p["S_実測"]), n(p["cap_MiB"]),
            n(p["予約の山_denoise"]), n(p["予約の山_decode"]), n(p["割り当ての山_denoise"]), n(p["割り当ての山_decode"]),
            n(p["余裕_MiB"]), n(p["retries"]), n(sps), delta(p), place, step, four, n(p.get("快適の回数"))))
    return "\n".join(o)


def same_r2(a, b):
    keys = ("予約の山_denoise", "予約の山_decode", "割り当ての山_denoise", "割り当ての山_decode", "retries", "inactive_split_MiB", "cap_MiB")
    return all(P[a][k] == P[b][k] for k in keys)


def ratio(a, b):
    return P[a]["1ステップ秒_meta"] / I04[b]["sps"]


def main():
    caps = [p["cap_MiB"] for p in R["points"] if p["cap_MiB"] is not None]
    viol = R["checks"]["violations"]
    w, f, e = C["w4a8"], C["fp8"], C["eros"]
    wb = w["帯の重なり"]["共通の帯_S"]
    ctl = P["s1080_362_int8"]
    _, ctl_step, ctl_four = oom_cells(ctl)
    out = []
    A = out.append
    A("# H3 の fp8・w4a8 の重みでの快適上限の較正 — 結果（台帳 §1-285・2026-11-13）\n")
    A("この文書は結果の正本です。数値はすべて `analysis/results.json`（`check_results_quant.py` が runs/ の記録から組み直したもの）から写しています。パスはこの較正台フォルダ（`outputs/comfort-calib-h3-quant-2026-11-13/`）からの相対です。\n")
    A("言葉: **S** ＝ DiT（映像と音声を生成する本体のネットワーク）に入る総行数。**cap** ＝ VRAM 16GB mode の割り当ての上限（約 14,000MiB）。**c** ＝ 快適の上端の点。**U** ＝ c の 1 段上で快適でなかった点。**予約の山** ＝ PyTorch が GPU から確保した領域（reserved）の最大。**割り当ての山** ＝ そのうち実際に使っていた量（allocated）の最大。**余裕** ＝ cap − ノイズ除去の段の予約の山。**retries** ＝ 上限に当たって、空いたセグメント（確保済みの区画）を返してから確保し直した回数。**Δ共有** ＝ 共有メモリ（GPU が主メモリへ退避した量）の中央値の、同じ形式の基準点との差。\n")
    A("## 0. 結論（先に）\n")
    A("| 形式（重み） | 解像度 | c（S） | U（S） | S の帯 | int8 との比較 | 結論の形 |")
    A("|---|---|---|---|---|---|---|")
    names = {"fp8": "fp8（Comfy-Org `minimax_h3_fl2va_pruned_fp8_scaled`）", "w4a8": "w4a8（Kijai `minimax_h3_fl2va_pruned_w4a8_mixed`）",
             "eros": "Eros（`10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized`）"}
    for fmt in ("fp8", "w4a8", "eros"):
        for res, v in C[fmt]["解像度"].items():
            byF = {p["F"]: p["S_実測"] for p in R["points"] if p["形式"] == fmt and p["役割"] == "点"}
            A("| %s | %s | %s（%s） | %s（%s） | %s〜%s | %s（int8 は c=%d／U=%d） | %s |" % (
                names[fmt], {"1080": "1920×1088", "1440": "2560×1472"}[res], n(v["c"]), n(byF.get(v["c"])), n(v["U"]), n(byF.get(v["U"])),
                n(v["帯_S"][0]), n(v["帯_S"][1]), v["向き"] if fmt != "eros" else "c は測らない（2 点とも OOM）", v["int8 の現行"]["c"], v["int8 の現行"]["U"],
                C[fmt]["結論の形"] if fmt != "eros" else "345・328 とも OOM（量は未定）"))
    A("")
    A("- **fp8 は int8 と同じ**です（結論の形 %s）。c・U が同じだけでなく、c の点の予約の山（13,486／13,764MiB）・retries 9・OOM の場所・OOM の文の 4 つの数まで int8 とほぼ一致しました（割り当ての山は −8／−7MiB）。" % f["結論の形"])
    A("- **Kijai の w4a8 は 1080・1440 とも 1 段下**です（結論の形 %s）。1080 と 1440 の帯は S で重なり、共通部分は %s〜%s です。c は 2 回快適で確定しました（`_r2` が 1 回目と MiB 単位で一致）。" % (w["結論の形"], n(wb[0]), n(wb[1])))
    A("- **Eros は 1920×1088 の 345 と 328 がどちらも OOM** → %s" % e["結論の形"].split("→ ")[-1] + "。311 は打っていません。")
    A("- **現行の線 219,516 との関係**: 操作パネルは形式によらず S ≤ 219,516 を快適の印にしています。Kijai の w4a8 では、印の内側（S %s を超え 219,516 以下）で OOM する形があります（1080×345＝S 213,348、1440×192＝S 217,798）。fp8 では印と結果が合っています（線 219,516 は共通の帯 %s〜%s の中）。Eros も 328（S 203,092）・345（S 213,348）が印の内側で OOM しました。" % (
        n(wb[0]), n(f["帯の重なり"]["共通の帯_S"][0]), n(f["帯の重なり"]["共通の帯_S"][1])))
    A("- 線を形式で分ける受け皿（記述子・印・種別判定）は、この計測では設計しません（オーナー裁定 3）。判断が要る点は §8 にまとめました。")
    A("- 突き合わせ: `check_results_quant.py` は %d 本の run.json で違反 %d 件です。注記 %d 件は §3 の末尾に載せました。\n" % (len(R["points"]), len(viol), len(R["checks"]["notes"])))
    A("## 1. 条件\n")
    A("- 2026-11-04 の再較正と同じです（計画 §3 の突き合わせの表）。Single・FL2VA・キーフレーム 1 枚（K=1）・VRAM 16GB mode on・先読み on・keep_resident off・attention は sage・4 ステップ・種 12345・LoRA なし・shift は既定の 12／3 です（Eros も 12／3。利用者の既定 10／3 とは違いますが、shift は VRAM に効かないので許容）。")
    A("- 今回の違い: AviUtl2 は起動していません。cap は %s〜%sMiB の帯に入りました（11-04 の帯 13,999〜14,025 の中）。" % (n(min(caps)), n(max(caps))))
    A("- サーバーのコードは dev `fcccdcf`（＝main `50f4b65`）です（`git_state_*.txt`）。作業ツリーの差は台帳の起票 1 件だけでした。")
    A("- 較正台は 11-04 の写しに、次の差分を加えたものです: 形式を run.json に残す・基準点の重みの照合・復帰の重みを点の計画から読む（`calib_quant.diff`・`drive2_quant.diff`）、煙の計画を重みごとに 4 本。selftest は差分の前後で回しました（`selftest/`・`selftest_before.txt`）。")
    A("- 弾 X（計画外の 2 点。オーナー了承）: int8 の 2560×1472×175（Kijai との比較用。基準点 b1440_int8 を今回取り直し、11-04 の b1440 と一致）と Eros の 1920×1088×328。煙はラベルに `_x` を付けました。")
    A("- 判定規則は v3″（計画 §5）です。快適 ＝ 完走し、OOM・退避（ノイズ除去／復号の窓で Δ共有 が +300MB 超）・先読みの降格のどれでもないこと。\n")
    A("## 2. 対照（弾 I）\n")
    A("- s1080_362_int8 は 11-04 と同じく OOM しました。場所は %s、%s ステップ目です。OOM の文の 4 つの数 %s は、11-04（%s）と一致しました → 環境は 11-04 と同じです。\n" % (ctl["OOM"]["場所"], ctl_step, ctl_four, I04_CONTROL))
    A("## 3. 点の表\n")
    for fmt, title in (("int8", "int8（対照）"), ("w4a8", "Kijai w4a8"), ("fp8", "fp8"), ("eros", "Eros")):
        A("### %s\n" % title)
        A(point_table(fmt))
        A("")
    A("- 事故の `b1080_w4a8`: 計測担当が基準点を二重に起動しました。GPU のジョブは 1 本で、2 本目は 409 JOB_BUSY で拒まれています（`manifest.jsonl` の submit_failed の行）。2 本目が同じフォルダに止めの印を置いたため、共有メモリの標本が欠けました（ノイズ除去・復号とも n=0）。数値は 1 本目のジョブの値で、`b1080_w4a8_r2` と一致しています。記録は残し、木の 1080 の点は `b1080_w4a8_r2` を基準点にしました（`progress.md`）。")
    A("- 「1 ステップの秒」は完走した点では metadata の値（DiT の読み込みを除いたノイズ除去の秒 ÷ ステップ数）です。OOM の点ではサーバーのログにある完走したステップの秒で、完走したステップが無い点は「—」です。")
    A("- `_x` の煙は弾 X の重みの読み込み直後の煙です。")
    A("- 復帰の暖機（`_rec<時刻>`）は、OOM の後にパイプラインを読み直したときの煙です。結論には使っていません。")
    A("- OOM の場所の「線形層の入力の量子化・out_proj／qkv」は、要求の大きさが S×7,168（out_proj の入力の幅）または S×5,376（qkv の入力の幅）の int8 と一致したことから名指しました（§6）。")
    A("- 全点の詳細（重みのファイル・load_ok_dit・S の予測との差・shift）は `analysis/results_table.md` にあります。")
    A("- 突き合わせの注記:")
    for x in R["checks"]["notes"]:
        A("  - %s: %s" % (x["label"], x["msg"]))
    A("")
    A("## 4. 木の結論と帯\n")
    A("| 形式 | 解像度 | c | 快適の回数 | U | U の判定 | S の帯 | 注記 |")
    A("|---|---|---|---|---|---|---|---|")
    for fmt in ("w4a8", "fp8", "eros"):
        for res, v in C[fmt]["解像度"].items():
            pts = [p for p in R["points"] if p["形式"] == fmt and p["役割"] == "点" and p["res"] == res]
            cp = [p for p in pts if p["F"] == v["c"]]
            up = [p for p in pts if p["F"] == v["U"]]
            A("| %s | %s | %s | %s | %s | %s | %s〜%s | %s |" % (
                fmt, res, n(v["c"]), n(cp[0]["快適の回数"]) if cp else "—", n(v["U"]),
                "・".join("%s（%s）" % (p["判定"], p["OOM"]["場所"] if p["OOM"] else "") for p in up) or "—",
                n(v["帯_S"][0]), n(v["帯_S"][1]), "；".join(v["注記"]) or "—"))
    A("")
    r2 = [("s1080_328_w4a8", "s1080_328_w4a8_r2"), ("s1440_175_w4a8", "s1440_175_w4a8_r2")]
    A("- Kijai の c の 2 回目（`_r2`）は、1 回目と予約の山・割り当ての山・retries・inactive_split・cap が%s一致しました。" % ("すべて" if all(same_r2(a, b) for a, b in r2) else "**一致しませんでした（要確認）**"))
    A("- fp8 の c は 1 回だけです（弾 Z-② は打っていません）。根拠は 2 つあります。Kijai の `_r2` が MiB 単位で一致したこと、11-04 の int8 の `_r2` も 1 回目と予約・割り当て・retries まで一致していたこと（1080 13,486／12,669／9、1440 13,764／12,905／9）です（計画 §5）。")
    A("- U はすべて cap での OOM なので、1 回で決まります（v3″）。")
    A("- int8 の 2560×1472×175（弾 X）は快適でした（快適の回数 1）。Kijai との比較のための点で、int8 の線の判定には使いません。int8 の c・U は現行（1080 345／362、1440 192／209）のままです。\n")
    A("## 5. 予測（計画 §4.2）との照合\n")
    A("予測は「形式の予約の山 ＝ int8 の同じ形の予約の山 ＋ a×S ＋ b」です。int8 の山は、今回測った点（弾 X の 1440×175）があればその値、無ければ 11-04 の値を使います。a・b は、fp8 が −4 バイト／行・−2MiB、Kijai が +5,372・−920、Eros が +5,372・0 です。\n")
    A("| 点 | 形式 | int8 の点 | 予測の余裕（MiB） | 予測 | 実測 | 実測の予約の山 − 予測（MiB） | 照合 | 当て直した a での余裕（参考） |")
    A("|---|---|---|---|---|---|---|---|---|")
    for q in PRED:
        A("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (q["label"], q["形式"], q["int8 の点"], n(q["予測の余裕"]), q["予測"], q["実測の判定"],
                                               n(q["差_MiB（実測−予測）"]), q["照合"],
                                               "・".join("%s: %s" % (k, n(v)) for k, v in (q.get("別の読みの余裕") or {}).items()) or "—"))
    A("")
    A("- fp8 は一致しました（差 +2.8MiB）。Kijai は 345・192 の 2 点で外れました（予測は快適、実測は OOM＝山が高い側）。快適だった 328・175 も、山は予測より高い側でした（+360／+1,037MiB）。Eros は 345・328 とも一致しました（OOM。328 は計画の予測が余裕 −5.5 の境目）。")
    A("- 参考の列: 割り当ての読み（a=+5,116）は Kijai 345・192 と Eros 328 を快適と外します。予約の読み（a=+7,228）は OOM の点を OOM 側に置きますが、1440×175 で否定されました（§6）。どちらの式でも、予約で決まる OOM は予測できません。\n")
    A("## 6. 仕組みの考察（要約。詳細は `analysis/mechanism_w4a8.md`）\n")
    A("**事実（コードと記録）**")
    A("- Kijai の 1440×192 は、out_proj（注意の出力 7,168 列を 5,376 列へ戻す線形層）の入力の量子化で OOM しました。要求 1.46GiB は S×7,168 の int8 と一致します（qkv の幅 5,376 なら 1.09GiB）。Eros の 1080×328 は同じ関数の qkv 側で、要求 1.02GiB は S×5,376 の int8 と一致します。")
    A("- comfy-kitchen の二重確保: `w4a8_int8_linear` は 2391 行で `xq`（M×K の int8）を無条件に確保します。`fast_act` 経路（M≥512・groupsize 256）では 2446 行で `quantize_int8_rowwise_convrot64` の戻り値に差し替えるため、元の `xq` は一度も使われません（死んだ確保）。ほかの経路は `xq` に直接書きます。")
    A("- この二重確保は上流（PyPI 0.2.41・GitHub main 2026-11-04）でも残っていて、issue・PR はありません（別の調査担当の確認）。int8 の `int8_linear` は置き場を 1 本だけ確保します。")
    a328, a175, i175 = P["s1080_328_w4a8"], P["s1440_175_w4a8"], P["s1440_175_int8"]
    A("- int8 との差（Kijai − int8）: 1080×328 は予約 %+d・割り当て %+d・retries %d 対 %d（int8 は 11-04）。1440×175 は予約 %+d・割り当て %+d・retries %d 対 %d（int8 は弾 X）。" % (
        a328["予約の山_denoise"] - I04["s1080_328"]["res"], a328["割り当ての山_denoise"] - I04["s1080_328"]["alloc"], a328["retries"], I04["s1080_328"]["retries"],
        a175["予約の山_denoise"] - i175["予約の山_denoise"], a175["割り当ての山_denoise"] - i175["割り当ての山_denoise"], a175["retries"], i175["retries"]))
    A("- 割り当ての読み（b＝−920MiB は置き場の環が小さい分、a は 1080×328 で当てた +5,116 バイト／行）は、1440×175 の差を +52.6MiB と予測し、実測は %+dMiB でした（独立な 1 点で一致）。" % (a175["割り当ての山_denoise"] - i175["割り当ての山_denoise"]))
    A("- 予約の読み（a＝+7,228 バイト／行）は否定されました。1 行あたりの予約の超過は 1080×328 で 7,228、1440×175 で 10,830 で、S に比例しません。前回の「7,228 は out_proj の S×7,168 の 1 本と近い」は偶然でした。")
    A("- int8 の傾き: 割り当ては 1080（328→345）で約 51,120、1440（175→192）で約 51,020 バイト／行と揃います。予約は約 53,370 と約 60,450 で揃いません。")
    A("- OOM した 4 点のうち 3 点は、予約済みの空きが要求より大きいのに 1 塊で取れませんでした（断片化）。Kijai 345 は空き 1.42GiB に 418MiB、Kijai 192 は 2.23GiB に 1.46GiB、Eros 328 は 1.26GiB に 1.02GiB です。この 3 点では、落ちた瞬間の割り当てが int8 の同じ形の割り当ての山より小さくなっていました（Eros 328: 11.40GiB 対 12,169MiB）。")
    A("- Eros は置き場の予約が int8 と同じ値でした（煙 3,848・基準点 5,866MiB）。Eros 345 は 1 ステップ目の sage の中で落ち、落ちた瞬間の割り当て（12.83GiB）が int8 345 の割り当ての山 12,669MiB を超えていました。")
    A("- fp8 は qkv の入力を 1 回だけ量子化して共有し、out_proj も 1 本確保するだけです。w4a8 の上乗せは無く、実測も int8 と同じでした。\n")
    A("**読み（推測）**")
    A("- Kijai の「使っている量」の上乗せは、bf16 の入力 h を注意の段の間持ち続ける分（+5,376 バイト／行）と、置き場の環が小さい分（−920MiB）で説明できます。")
    A("- 線を下げているのは予約の側の超過、つまり caching allocator（PyTorch のメモリの再利用の仕組み）の断片化だと読んでいます。呼び出しごとに S×7,168・S×5,376 の int8 の一時テンソルが 2 本並ぶため、予約の中に使えない穴が残ります。ただし量は形（解像度・一時テンソルの大きさ）で変わり、1 本の式にはなりません。")
    A("- Eros は置き場の環が小さくならない（b＝0）ので、w4a8 の層の上乗せがそのまま効きます（345 は使っている量の側、328 は断片化の側で落ちた）。")
    A("- 1 ステップの秒（int8 との比）:")
    A("  - fp8: 1080×345 が %.3f 倍、1440×192 が %.3f 倍。" % (ratio("s1080_345_fp8", "s1080_345"), ratio("s1440_192_fp8", "s1440_192")))
    A("  - Kijai: 基準点で %.3f 倍（1080）・%.3f 倍（1440）、1080×328 で %.3f 倍、1440×175 で %.3f 倍（int8 は弾 X）。" % (
        P["b1080_w4a8_r2"]["1ステップ秒_meta"] / I04["b1080"]["sps"], P["b1440_w4a8"]["1ステップ秒_meta"] / I04["b1440"]["sps"],
        ratio("s1080_328_w4a8", "s1080_328"), P["s1440_175_w4a8"]["1ステップ秒_meta"] / P["s1440_175_int8"]["1ステップ秒_meta"]))
    A("  - Kijai は大きな S ほど比が 1 に近づきます。大きな S ではステップ時間の多くを注意の項が占め、注意の項は形式によらないためと読んでいます（推測）。\n")
    A("## 7. 留保\n")
    A("- 20 ステップでの確認は int8 だけです（11-04 の s1080_345_20。計画 §3）。fp8・w4a8 の 20 ステップは確かめていません。")
    A("- fp8 の RTX 30 系の経路（計算能力 8.9 未満で bf16 に戻す経路）は測っていません（オーナー裁定 2。開発機は 8.9 で `scaled` 経路）。")
    A("- 弾 Z-②（fp8 の c の `_r2`）と弾 Z-③（1280×768×481）は打っていません。根拠: Kijai の `_r2` が完全に一致したこと、fp8 と int8 が同じ値だったこと、481 は int8 の余裕が 2,647MiB あり、今回見た Kijai の予約の超過の最大（1440×175 の +1,138MiB）でも届かないこと（割り当ての読みでは 5,116×139,882 行 ≒ 682MiB − 920 ＝ −238MiB）。")
    A("- Eros は 328・345 とも OOM で、線は S 203,092 より下としか言えません（量は未定。311 以下は打たずに打ち止め＝オーナー了承）。")
    A("- 予約の読み（a≈+7,228 バイト／行）は弾 X の 1440×175 で否定されました。予約の側の超過を 1 本の式で表す読みは、今のところありません。")
    A("- int8 の 2560×1472×175 は快適 1 回だけです（比較用の点）。")
    A("- shift は全点 12／3 です（OOM の点は metadata が無いので、ワーカーの generate 行から読みました）。\n")
    A("## 8. オーナーの判断が要る点（選択肢を並べるだけ）\n")
    A("- (a) 線が形式で違う事実の受け皿は、台帳 §1-286 として起票済みです。")
    A("- (b) comfy-kitchen の w4a8 カーネルの二重確保（§6）を上流へ報告するか、しないか。\n")
    A("## 9. 記録の所在（このフォルダからの相対パス）\n")
    A("- 点ごとの記録: `runs/<ラベル>/`（run.json・metadata.json・point.json・request.json・job.json・server_slice.log・wddm.csv ほか）")
    A("- 集計: `analysis/results.json`・`analysis/results_table.md`（`check_results_quant.py` の出力）、`analysis/mechanism_w4a8.md`（仕組みの考察）、`analysis/make_results_md.py`（この文書の生成）")
    A("- 運転の記録: `progress.md`・`manifest.jsonl`・`commit_log.csv`・`drive2_*.log`・`rp_*.txt`・`run_p_*_output.txt`・`preflight_*`")
    A("- 重みと環境: `load_*.txt`（LOAD_OK 行。弾 X は `load_int8_x.txt`・`load_eros_x.txt`）・`load_*_output.txt`・`idle_*_output.txt`・`git_state_*.txt`（弾 X は `git_state_x.txt`・`git_state_load_eros_x.txt`）・`original_state.json`・`restore_state_output.txt`・`restore_state_x_output.txt`・`unload_output.txt`・`unload_x_output.txt`・`luid.json`")
    A("- 較正台の差分: `calib_quant.diff`・`drive2_quant.diff`（元は `calib_before_quant.py`・`drive2_before_quant.sh`）、`selftest/`・`selftest_before.txt`")
    A("- 計画: 監督の計画ファイル（graceful-fluttering-reddy。§3 条件・§4 予測・§5 判定・§7 点・§10 結論の形）")
    io.open(os.path.join(ROOT, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("wrote RESULTS.md")


if __name__ == "__main__":
    main()
