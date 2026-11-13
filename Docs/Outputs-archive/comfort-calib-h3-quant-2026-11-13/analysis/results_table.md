> 複写（スナップショット）: 原本 `outputs/comfort-calib-h3-quant-2026-11-13/analysis/results_table.md`（git 管理外）・複写日 2026-11-13

# H3 形式別の快適上限 — 突き合わせの表（check_results_quant.py が生成）

runs/ の記録だけから組み直した。判定は計画 §5 の v3''（ここで独立に組み直し、run.json と突き合わせた）。

## 1. 点ごとの行

| ラベル | 形式 | 重み（名前／ファイル） | load_ok_dit | W×H×F | K | S（predict／実測・差） | status | 判定 | cap | 予約の山（denoise／decode） | 割り当ての山（denoise／decode） | 余裕 | retries | 1 ステップ秒（meta／log） | OOM の場所 | OOM の 4 つの数（要求／割り当て／予約済みの空き／許可）・free | 落ちたステップ | shift | 快適の回数 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm__default | int8 | default／`minimax_h3_fl2va_pruned_int8_convrot.safetensors` | {"int8_tensorwise": 200}（load_int8.txt） | 640×384×56 | 1 | 4,784／4,784・0 | completed | **煙** | 14,023 | 3,848／5,382 | 3,816／5,062 | 10,175 | 0 | 1.867／1.675 | — | — | — | 12／3 | — |
| warm__default_x | int8 | default／`minimax_h3_fl2va_pruned_int8_convrot.safetensors` | {"int8_tensorwise": 200} | 640×384×56 | 1 | 4,784／4,784・0 | completed | **煙** | 14,023 | 3,848／5,382 | 3,816／5,062 | 10,175 | 0 | 1.875／1.675 | — | — | — | 12／3 | — |
| b1080_int8 | int8 | default／`minimax_h3_fl2va_pruned_int8_convrot.safetensors` | {"int8_tensorwise": 200}（load_int8.txt） | 1920×1088×56 | 1 | 38,984／38,984・0 | completed | **基準点** | 13,999 | 5,866／10,486 | 4,340／8,143 | 8,133 | 0 | 19.865／19.675 | — | — | — | 12／3 | — |
| b1440_int8 | int8 | default／`minimax_h3_fl2va_pruned_int8_convrot.safetensors` | {"int8_tensorwise": 200} | 2560×1472×56 | 1 | 70,144／70,144・0 | completed | **基準点** | 14,001 | 8,214／14,800 | 5,709／10,768 | 5,787 | 0 | 48.513／48.325 | — | — | — | 12／3 | — |
| s1080_362_int8 | int8 | 計画 default／`minimax_h3_fl2va_pruned_int8_convrot.safetensors` | {"int8_tensorwise": 200}（load_int8.txt） | 1920×1088×362 | 1 | 223,604／223,604・0 | failed | **OOM** | — | —／— | —／— | — | — | —／352.9 | o の確保（denoise・`o = torch.empty((s, inner), dtype=torch.bfloat16, device=x.device)`） | 2.99 GiB／6.94 GiB／3.84 GiB／13.67 GiB・free 3.89 GiB | 2／4 | 12／3（ワーカーの generate 行（metadata なし）） | 0 |
| s1440_175_int8 | int8 | default／`minimax_h3_fl2va_pruned_int8_convrot.safetensors` | {"int8_tensorwise": 200} | 2560×1472×175 | 1 | 199,342／199,342・0 | completed | **快適** | 14,001 | 12,700／14,846 | 12,007／10,861 | 1,301 | 9 | 286.517／286.3 | — | — | — | 12／3 | 1 |
| warm__minimax_h3_fl2va_pruned_w4a8_mixed | w4a8 | minimax_h3_fl2va_pruned_w4a8_mixed／`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | {"asym_w4a8_int8": 200} | 640×384×56 | 1 | 4,784／4,784・0 | completed | **煙** | 14,023 | 2,888／5,374 | 2,846／5,062 | 11,135 | 0 | 2.013／1.825 | — | — | — | 12／3 | — |
| warm__minimax_h3_fl2va_pruned_w4a8_mixed_rec1113102714 | w4a8 | minimax_h3_fl2va_pruned_w4a8_mixed／`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | {"asym_w4a8_int8": 200} | 640×384×56 | 1 | 4,784／4,784・0 | completed | **煙** | 14,023 | 2,888／5,374 | 2,846／5,062 | 11,135 | 0 | 2.005／1.85 | — | — | — | 12／3 | — |
| warm__minimax_h3_fl2va_pruned_w4a8_mixed_rec1113111425 | w4a8 | minimax_h3_fl2va_pruned_w4a8_mixed／`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | {"asym_w4a8_int8": 200} | 640×384×56 | 1 | 4,784／4,784・0 | completed | **煙** | 14,023 | 2,888／5,374 | 2,846／5,062 | 11,135 | 0 | 2／1.85 | — | — | — | 12／3 | — |
| b1080_w4a8 | w4a8 | minimax_h3_fl2va_pruned_w4a8_mixed／`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | {"asym_w4a8_int8": 200} | 1920×1088×56 | 1 | 38,984／38,984・0 | completed | **基準点** | 13,999 | 5,052／10,486 | 3,465／8,143 | 8,947 | 0 | 21.545／21.4 | — | — | — | 12／3 | — |
| b1080_w4a8_r2 | w4a8 | minimax_h3_fl2va_pruned_w4a8_mixed／`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | {"asym_w4a8_int8": 200} | 1920×1088×56 | 1 | 38,984／38,984・0 | completed | **基準点** | 13,999 | 5,052／10,486 | 3,465／8,143 | 8,947 | 0 | 21.5／21.325 | — | — | — | 12／3 | — |
| b1440_w4a8 | w4a8 | minimax_h3_fl2va_pruned_w4a8_mixed／`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | {"asym_w4a8_int8": 200} | 2560×1472×56 | 1 | 70,144／70,144・0 | completed | **基準点** | 13,999 | 8,178／14,800 | 5,097／10,768 | 5,821 | 0 | 51.288／51.1 | — | — | — | 12／3 | — |
| s1080_328_w4a8 | w4a8 | minimax_h3_fl2va_pruned_w4a8_mixed／`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | {"asym_w4a8_int8": 200} | 1920×1088×328 | 1 | 203,092／203,092・0 | completed | **快適** | 13,999 | 13,444／10,540 | 12,240／8,261 | 555 | 16 | 305.285／305.1 | — | — | — | 12／3 | 2 |
| s1080_328_w4a8_r2 | w4a8 | minimax_h3_fl2va_pruned_w4a8_mixed／`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | {"asym_w4a8_int8": 200} | 1920×1088×328 | 1 | 203,092／203,092・0 | completed | **快適** | 13,999 | 13,444／10,540 | 12,240／8,261 | 555 | 16 | 305.787／305.575 | — | — | — | 12／3 | 2 |
| s1080_345_w4a8 | w4a8 | 計画 minimax_h3_fl2va_pruned_w4a8_mixed／`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | {"asym_w4a8_int8": 200} | 1920×1088×345 | 1 | 213,348／213,348・0 | failed | **OOM** | — | —／— | —／— | — | — | —／333.9 | sage の中（denoise・`b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)`） | 418.00 MiB／11.88 GiB／1.42 GiB／13.67 GiB・free 1.33 GiB | 3／4 | 12／3（ワーカーの generate 行（metadata なし）） | 0 |
| s1440_175_w4a8 | w4a8 | minimax_h3_fl2va_pruned_w4a8_mixed／`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | {"asym_w4a8_int8": 200} | 2560×1472×175 | 1 | 199,342／199,342・0 | completed | **快適** | 13,999 | 13,838／14,846 | 12,058／10,861 | 161 | 13 | 294.773／294.55 | — | — | — | 12／3 | 2 |
| s1440_175_w4a8_r2 | w4a8 | minimax_h3_fl2va_pruned_w4a8_mixed／`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | {"asym_w4a8_int8": 200} | 2560×1472×175 | 1 | 199,342／199,342・0 | completed | **快適** | 13,999 | 13,838／14,846 | 12,058／10,861 | 161 | 13 | 294.852／294.675 | — | — | — | 12／3 | 2 |
| s1440_192_w4a8 | w4a8 | 計画 minimax_h3_fl2va_pruned_w4a8_mixed／`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | {"asym_w4a8_int8": 200} | 2560×1472×192 | 1 | 217,798／217,798・0 | failed | **OOM** | — | —／— | —／— | — | — | —／346.6 | 線形層の入力の量子化・out_proj（要求＝S×7,168 の int8）（denoise・`q_2d = torch.empty(`） | 1.46 GiB／10.28 GiB／2.23 GiB／13.67 GiB・free 2.13 GiB | 2／4 | 12／3（ワーカーの generate 行（metadata なし）） | 0 |
| warm__minimax_h3_fl2va_pruned_fp8_scaled | fp8 | minimax_h3_fl2va_pruned_fp8_scaled／`minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | {"float8_e4m3fn": 200} fp8_mode=scaled | 640×384×56 | 1 | 4,784／4,784・0 | completed | **煙** | 14,023 | 3,848／5,382 | 3,809／5,062 | 10,175 | 0 | 2.545／2.35 | — | — | — | 12／3 | — |
| warm__minimax_h3_fl2va_pruned_fp8_scaled_rec1113132820 | fp8 | minimax_h3_fl2va_pruned_fp8_scaled／`minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | {"float8_e4m3fn": 200} fp8_mode=scaled | 640×384×56 | 1 | 4,784／4,784・0 | completed | **煙** | 14,023 | 3,848／5,382 | 3,809／5,062 | 10,175 | 0 | 2.53／2.325 | — | — | — | 12／3 | — |
| b1080_fp8 | fp8 | minimax_h3_fl2va_pruned_fp8_scaled／`minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | {"float8_e4m3fn": 200} fp8_mode=scaled | 1920×1088×56 | 1 | 38,984／38,984・0 | completed | **基準点** | 13,999 | 6,014／10,486 | 4,481／8,143 | 7,985 | 0 | 25.352／25.15 | — | — | — | 12／3 | — |
| b1440_fp8 | fp8 | minimax_h3_fl2va_pruned_fp8_scaled／`minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | {"float8_e4m3fn": 200} fp8_mode=scaled | 2560×1472×56 | 1 | 70,144／70,144・0 | completed | **基準点** | 13,999 | 8,362／14,800 | 5,811／10,768 | 5,637 | 0 | 58.207／58 | — | — | — | 12／3 | — |
| s1080_345_fp8 | fp8 | minimax_h3_fl2va_pruned_fp8_scaled／`minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | {"float8_e4m3fn": 200} fp8_mode=scaled | 1920×1088×345 | 1 | 213,348／213,348・0 | completed | **快適** | 13,999 | 13,486／10,548 | 12,661／8,271 | 513 | 9 | 355.155／354.95 | — | — | — | 12／3 | 1 |
| s1080_362_fp8 | fp8 | 計画 minimax_h3_fl2va_pruned_fp8_scaled／`minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | {"float8_e4m3fn": 200} fp8_mode=scaled | 1920×1088×362 | 1 | 223,604／223,604・0 | failed | **OOM** | — | —／— | —／— | — | — | —／384.5 | o の確保（denoise・`o = torch.empty((s, inner), dtype=torch.bfloat16, device=x.device)`） | 2.99 GiB／6.93 GiB／3.85 GiB／13.67 GiB・free 3.89 GiB | 2／4 | 12／3（ワーカーの generate 行（metadata なし）） | 0 |
| s1440_192_fp8 | fp8 | minimax_h3_fl2va_pruned_fp8_scaled／`minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | {"float8_e4m3fn": 200} fp8_mode=scaled | 2560×1472×192 | 1 | 217,798／217,798・0 | completed | **快適** | 13,999 | 13,764／14,858 | 12,898／10,874 | 235 | 9 | 367.975／367.775 | — | — | — | 12／3 | 1 |
| s1440_209_fp8 | fp8 | 計画 minimax_h3_fl2va_pruned_fp8_scaled／`minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | {"float8_e4m3fn": 200} fp8_mode=scaled | 2560×1472×209 | 1 | 236,254／236,254・0 | failed | **OOM** | — | —／— | —／— | — | — | —／— | sage の中（denoise・`h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)`） | 462.00 MiB／12.79 GiB／560.07 MiB／13.67 GiB・free 1.29 GiB | 1／4 | 12／3（ワーカーの generate 行（metadata なし）） | 0 |
| warm__10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized | eros | 10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized／`10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors` | {"asym_w4a8_int8": 180, "int8_tensorwise": 20} | 640×384×56 | 1 | 4,784／4,784・0 | completed | **煙** | 14,023 | 3,848／5,382 | 3,816／5,062 | 10,175 | 0 | 1.988／1.75 | — | — | — | 12／3 | — |
| warm__10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized_x | eros | 10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized／`10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors` | {"asym_w4a8_int8": 180, "int8_tensorwise": 20} | 640×384×56 | 1 | 4,784／4,784・0 | completed | **煙** | 14,023 | 3,848／5,382 | 3,816／5,062 | 10,175 | 0 | 1.988／1.75 | — | — | — | 12／3 | — |
| b1080_eros | eros | 10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized／`10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors` | {"asym_w4a8_int8": 180, "int8_tensorwise": 20} | 1920×1088×56 | 1 | 38,984／38,984・0 | completed | **基準点** | 13,999 | 5,866／10,486 | 4,436／8,143 | 8,133 | 0 | 21.378／21.175 | — | — | — | 12／3 | — |
| s1080_328_eros | eros | 計画 10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized／`10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors` | {"asym_w4a8_int8": 180, "int8_tensorwise": 20} | 1920×1088×328 | 1 | 203,092／203,092・0 | failed | **OOM** | — | —／— | —／— | — | — | —／304.7 | 線形層の入力の量子化・qkv（要求＝S×5,376 の int8）（denoise・`q_2d = torch.empty(`） | 1.02 GiB／11.40 GiB／1.26 GiB／13.67 GiB・free 1.96 GiB | 2／4 | 12／3（ワーカーの generate 行（metadata なし）） | 0 |
| s1080_345_eros | eros | 計画 10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized／`10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors` | {"asym_w4a8_int8": 180, "int8_tensorwise": 20} | 1920×1088×345 | 1 | 213,348／213,348・0 | failed | **OOM** | — | —／— | —／— | — | — | —／— | sage の中（denoise・`h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)`） | 418.00 MiB／12.83 GiB／799.40 MiB／13.67 GiB・free 1.02 GiB | 1／4 | 12／3（ワーカーの generate 行（metadata なし）） | 0 |

- 余裕 ＝ cap − ノイズ除去の予約の山（計画 §4.2 の int8 の余裕 513／235 と同じ取り方）。retries が 0 でない点では OOM までの距離そのものではない（§5）。
- 退避の窓（共有メモリの中央値 MB・wddm.csv から再計算。Δ は同じ形式の基準点との差）:
  - s1440_175_int8（基準 b1440_int8）: denoise 1,165.9（Δ -26.1・n=1142）・decode 1,162.1（Δ -18.3・n=176）
  - s1080_328_w4a8（基準 b1080_w4a8_r2）: denoise 651.2（Δ -11.3・n=1216）・decode 657.8（Δ -3.7・n=192）
  - s1080_328_w4a8_r2（基準 b1080_w4a8_r2）: denoise 630.6（Δ -31.9・n=1219）・decode 648.4（Δ -13.1・n=192）
  - s1440_175_w4a8（基準 b1440_w4a8）: denoise 647.6（Δ -14.7・n=1175）・decode 649.6（Δ -0.5・n=176）
  - s1440_175_w4a8_r2（基準 b1440_w4a8）: denoise 657.9（Δ -4.4・n=1175）・decode 653.4（Δ 3.3・n=177）
  - s1080_345_fp8（基準 b1080_fp8）: denoise 1,163（Δ -3.7・n=1415）・decode 1,170（Δ 12・n=203）
  - s1440_192_fp8（基準 b1440_fp8）: denoise 1,141.1（Δ -35.4・n=1466）・decode 1,163.3（Δ -12.4・n=194）

## 2. 形式ごとの結論（計画 §10。Eros は §5 の 2 択）

int8 の現行: 1080 c=345／U=362（S 213,348／223,604）、1440 c=192／U=209（S 217,798／236,254）、線 219,516。

| 形式 | 解像度 | c | U | 帯（S: c の S〜U の S） | int8 との比較 | 注記 |
|---|---|---|---|---|---|---|
| int8 | 1080 | — | 362 | —〜223,604 | 未確定（int8 c=345／U=362） | — |
| int8 | 1440 | 175 | — | 199,342〜— | 未確定（int8 c=192／U=209） | c=175 は快適 1 回（_r2 は弾 Z。§5 の根拠を添える） |
| w4a8 | 1080 | 328 | 345 | 203,092〜213,348 | 下（int8 c=345／U=362） | — |
| w4a8 | 1440 | 175 | 192 | 199,342〜217,798 | 下（int8 c=192／U=209） | — |
| fp8 | 1080 | 345 | 362 | 213,348〜223,604 | 同（int8 c=345／U=362） | c=345 は快適 1 回（_r2 は弾 Z。§5 の根拠を添える） |
| fp8 | 1440 | 192 | 209 | 217,798〜236,254 | 同（int8 c=192／U=209） | c=192 は快適 1 回（_r2 は弾 Z。§5 の根拠を添える） |
| eros | 1080 | — | 328 | —〜203,092 | 未確定（int8 c=345／U=362） | — |

- **int8**: 結論の形 対照（弾 I・§7.2）: 1080 の U=362 が OOM（環境は 11-04 と同じかは §5 の 4 つの数で見る）。追加の点（弾 X）: 1440×175 が 快適（Kijai との比較用。線の判定には使わない）。1080 と 1440 の帯の重なり: —（両解像度の c・U がそろっていない）
- **w4a8**: 結論の形 (下)。1080 と 1440 の帯の重なり: 重なる（共通 203,092〜213,348・219,516 は外）
- **fp8**: 結論の形 (同)。1080 と 1440 の帯の重なり: 重なる（共通 217,798〜223,604・219,516 は中）
- **eros**: 結論の形 345・328 とも OOM → Eros の線は S 203,092 より下。量は未定（オーナー裁定で打ち止め）。1080 と 1440 の帯の重なり: —（両解像度の c・U がそろっていない）

## 3. 予測との照合（計画 §4.2: 形式の山 ＝ int8 の山 ＋ a×S ＋ b）

int8 の山は同じ形の int8 の完走点（今回の runs にあればそれ、無ければ 11-04・記録簿 §196）。a・b: fp8 −4 B/行・−2MiB、w4a8 +5,372・−920、Eros +5,372・0。

| 点 | 形式 | int8 の点 | S | int8 の山／余裕 | a×S＋b | 予測の山／余裕 | 予測 | 実測の判定 | 実測の山 | 差（実測−予測） | 実測の cap／余裕／retries | 照合 | 別の読みの余裕 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1080_328_w4a8 | w4a8 | s1080_328 | 203,092 | 12,964／1,035 | 120.5 | 13,084.5／914.5 | 快適 | 快適 | 13,444 | 359.5 | 13,999／555／16 | 判定は一致・山は予測より高い側（差 +359.5 MiB） | 割り当ての読み a=+5,116: 964.1・予約の読み a=+7,228（弾 X で否定）: 555.1 |
| s1080_328_w4a8_r2 | w4a8 | s1080_328 | 203,092 | 12,964／1,035 | 120.5 | 13,084.5／914.5 | 快適 | 快適 | 13,444 | 359.5 | 13,999／555／16 | 判定は一致・山は予測より高い側（差 +359.5 MiB） | 割り当ての読み a=+5,116: 964.1・予約の読み a=+7,228（弾 X で否定）: 555.1 |
| s1080_345_w4a8 | w4a8 | s1080_345 | 213,348 | 13,486／513 | 173 | 13,659／340 | 快適 | OOM | — | — | —／—／— | 外れ（予測は快適・実測は OOM＝山が高い側） | 割り当ての読み a=+5,116: 392.1・予約の読み a=+7,228（弾 X で否定）: -37.6 |
| s1440_175_w4a8 | w4a8 | s1440_175_int8（今回） | 199,342 | 12,700／1,301 | 101.3 | 12,801.3／1,199.7 | 快適 | 快適 | 13,838 | 1,036.7 | 13,999／161／13 | 判定は一致・山は予測より高い側（差 +1036.7 MiB） | 割り当ての読み a=+5,116: 1,248.4・予約の読み a=+7,228（弾 X で否定）: 846.9 |
| s1440_175_w4a8_r2 | w4a8 | s1440_175_int8（今回） | 199,342 | 12,700／1,301 | 101.3 | 12,801.3／1,199.7 | 快適 | 快適 | 13,838 | 1,036.7 | 13,999／161／13 | 判定は一致・山は予測より高い側（差 +1036.7 MiB） | 割り当ての読み a=+5,116: 1,248.4・予約の読み a=+7,228（弾 X で否定）: 846.9 |
| s1440_192_w4a8 | w4a8 | s1440_192 | 217,798 | 13,764／235 | 195.8 | 13,959.8／39.2 | 快適 | OOM | — | — | —／—／— | 外れ（予測は快適・実測は OOM＝山が高い側） | 割り当ての読み a=+5,116: 92.4・予約の読み a=+7,228（弾 X で否定）: -346.3 |
| s1080_345_fp8 | fp8 | s1080_345 | 213,348 | 13,486／513 | -2.8 | 13,483.2／515.8 | 快適 | 快適 | 13,486 | 2.8 | 13,999／513／9 | 一致（差 +2.8 MiB） | — |
| s1440_192_fp8 | fp8 | s1440_192 | 217,798 | 13,764／235 | -2.8 | 13,761.2／237.8 | 快適 | 快適 | 13,764 | 2.8 | 13,999／235／9 | 一致（差 +2.8 MiB） | — |
| s1080_328_eros | eros | s1080_328 | 203,092 | 12,964／1,035 | 1,040.5 | 14,004.5／-5.5 | OOM | OOM | — | — | —／—／— | 一致（OOM） | 割り当ての読み a=+5,116: 44.1・予約の読み a=+7,228（弾 X で否定）: -364.9 |
| s1080_345_eros | eros | s1080_345 | 213,348 | 13,486／513 | 1,093 | 14,579／-580 | OOM | OOM | — | — | —／—／— | 一致（OOM） | 割り当ての読み a=+5,116: -527.9・予約の読み a=+7,228（弾 X で否定）: -957.6 |

## 4. 整合の検査

- 違反: 0 件
- 注記: 19 件
  - warm__default: load_ok_dit は load_int8.txt（run.json に欄なし＝差分 1 より前の点） から
  - b1080_int8: load_ok_dit は load_int8.txt（run.json に欄なし＝差分 1 より前の点） から
  - s1080_362_int8: used_check_h3 は対象外（failed・metadata なし）
  - s1080_362_int8: load_ok_dit は load_int8.txt（run.json に欄なし＝差分 1 より前の点） から
  - s1080_362_int8: shift 12.0／3.0 は ワーカーの generate 行（metadata なし）
  - b1080_w4a8: 共有メモリの標本なし（二重起動の事故。progress.md 参照）・数値は 1 本目のジョブの値
  - b1080_w4a8: manifest.jsonl に submit_failed の行 1 件（HTTP 409）
  - s1080_345_w4a8: used_check_h3 は対象外（failed・metadata なし）
  - s1080_345_w4a8: shift 12.0／3.0 は ワーカーの generate 行（metadata なし）
  - s1440_192_w4a8: used_check_h3 は対象外（failed・metadata なし）
  - s1440_192_w4a8: shift 12.0／3.0 は ワーカーの generate 行（metadata なし）
  - s1080_362_fp8: used_check_h3 は対象外（failed・metadata なし）
  - s1080_362_fp8: shift 12.0／3.0 は ワーカーの generate 行（metadata なし）
  - s1440_209_fp8: used_check_h3 は対象外（failed・metadata なし）
  - s1440_209_fp8: shift 12.0／3.0 は ワーカーの generate 行（metadata なし）
  - s1080_328_eros: used_check_h3 は対象外（failed・metadata なし）
  - s1080_328_eros: shift 12.0／3.0 は ワーカーの generate 行（metadata なし）
  - s1080_345_eros: used_check_h3 は対象外（failed・metadata なし）
  - s1080_345_eros: shift 12.0／3.0 は ワーカーの generate 行（metadata なし）
