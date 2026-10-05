# 快適上限の計測・第 2 回の結果（第 8 弾・§1-79・計測日 2026-10-05）

> フォルダ名は準備した日（2026-10-04）のまま。10-04 に準備したが、自動許可の判定が `calib.py server start --pid` を拒否して計測 0 本で中断し（オーナー裁定で中止）、10-05 にサーバーと監視を起こし直して計測した。

独立裏取り（Opus・2026-10-05）: 表は 84 本すべて一次記録から再現。直すべき 1・注意 5・参考 6 は第 3〜6 節に反映。

**結論（先に）**
- **84 本すべて完了・失敗 0・`used_matches_request` は全点一致・連結 36 本はすべて `chunked_upsample: true` で送り、`server_slice.log` に `chain upsample started (3 units)` が 1 行ずつ出たことを確認**。コミットの最大は **101.66 GiB（71.0%・`ha_s_720_289`）**で、90% の警戒線には一度も届かなかった。
- 各セルの定数候補（規則 6「既知の溢れ点より下で、どれかの幾何で 2 回快適だった最も高い段」。詳細は第 3 節）:

| セル | 定数候補（トークン） | その点 | 1280×768 換算（単発） |
|---|---:|---|---|
| 2.3 標準・連結（全 on・cu=true） | **42,240** | 1920×1024 w22 | — |
| 2.5 標準・連結（全 on と既定構成の小さいほう） | **46,376** | 1984×1088 w22 | — |
| 2.3 重い（fp8mixed）・単発 | **32,640** | 1080p 121／1280×768 265 | **265 コマ** |
| 2.3 重い（fp8mixed）・連結 | **32,384** | 1472×1024 w22 | — |
| 2.5 重い・単発（REDGraft 既定構成・fp8 照合） | **38,760** | 1080p 145 | **313 コマ**（38,400） |
| 2.5 重い・連結（既定構成と全 on の小さいほう） | **39,424** | 1792×1024 w22 | — |
| 2.3 Q6_K・単発 | **43,200** | 1280×768 353 | **353 コマ** |
| 2.3 Q6_K・連結 | **40,832** | 1856×1024 w22 | — |
| 2.5 Q6_K・単発 | **43,344**（M を過去の割れ 44,160 とした場合）／**44,160**（今回の 2 回快適を採った場合） | 896×1152 337／1280×768 361 | **353 コマ**／**361 コマ** |
| 2.5 Q6_K・連結 | **43,648** | 1984×1024 w22 | — |

- **cu=true にすると連結の境界が動いた**: 2.3 標準は第 7 弾（cu=false）で `s23_gap_mid` に 6〜7 GB の退避が出て「未確定」だったが、cu=true では `s23_gap_mid` が 25〜27 秒に縮んで退避が消え、境界は stage-2 の窓で決まった（1920×1024 快適×2・1984×1024 溢れ×2）。2.5 標準は逆に、cu=false で 2 回快適だった 2048×1088 が cu=true では**割れ**（溢れ×1・快適×1）になり、c が 1 段下がった（46,376）。
- **オーナーの判断事項**: (1) 2.5 Q6_K の単発の M（09-14 の 361 割れを採るか、今回の 361 快適×2 を採るか。どちらも delta が閾値 300 MB の際）／(2) 2.5 重いは連結の c（39,424）が単発の c（38,760）より高い。2.5 の配信表に単発と連結を同じ 1 値で書くなら小さいほう／(3) 連結の U で「割れ」が 2 腕（sb・ra）に出た（cu=true の連結は U の段が不安定になりやすい）。

## 1. 条件

| 項目 | 値 |
|---|---|
| 日付 | 2026-10-05 08:19〜20:12（84 本の実測の合計 28,742 秒＝約 7 時間 59 分。冷却・読み込み込みの経過は約 11 時間 53 分） |
| リポジトリ | Nz-Videomni dev・HEAD `c83f7f0`（main `dc93279` に含まれる） |
| サーバー | 監督が `.venv\Scripts\python.exe main.py` を `Start-Process` で起動（PID 32708・`http://127.0.0.1:18620`）。`calib.py server start --pid 32708` と `calib.py luid` は監督が打った（LUID `luid_0x00000000_0x0000f29f_phys_0`） |
| 変換器（登録名＝実ファイル） | LTX23 `default`＝`LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`（sa）／LTX23 `Sulphur-2-base-distil-Q6_K`＝`Sulphur-2-base-distil-Q6_K.gguf`（qa）／LTX23 `sulphur_distil_fp8mixed`＝`sulphur_distil_fp8mixed.safetensors`（ha）／LTX25 `default`＝`LTX-2.5-22B-distilled-transformer.gguf`（sb・sd）／LTX25 `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K`＝同名 `.gguf`（qb）／LTX25 `redgraftLTX25Fast2K_ltx25RedgraftNSFW`＝同名 `.safetensors`（ra・rb）／LTX25 `ltx25_uncensored_v1.1-fp8_scaled`＝同名 `.safetensors`（ua）。各点の `metadata.json` の `models` と `transformer_used` で確認 |
| 構成 | 全 on＝`ACCEL_ALLON_LTX23`（sage・prefetch・keep_resident・fused・`prune_vaed`・埋め込み常駐 off）／`ACCEL_ALLON_LTX25`（sage・prefetch・keep_resident・fused・vae `default`・埋め込み常駐 on）。既定構成（sd・ra・ua）＝`DEFAULT_ACCEL`（sdpa・prefetch on・keep_resident off・fused on・vae `default`・埋め込み常駐 off。2.5 の vae の実体は `conv`） |
| 判定規則 | 規則 v3′（基準点と比べて、判定可能な窓のどれか 1 つで WDDM 共有メモリの中央値が +300 MB 以上なら溢れ。標本 10 未満の窓は判定しない）。窓は 2.3 が `stage2`・`restore`・`s23_stage1`・`s23_gap_mid`・`s23_gap_tail`、2.5 が `stage2`・`restore`・`p25_30_decode_encode`。今回の溢れ 18 本のうち `stage2` 以外の窓で閾値を超えたのは qa 連結の 2 本（`s23_gap_mid` +357.0／+359.3。`stage2` も +351.3 で両方超え）だけ |
| 手順 | 2 回一致（c が 2 回快適・c の 1 段上が 2 回快適でない。割れは溢れに数える）・計画の規則 1〜9・冷却 150 秒（`cool.sh`）・腕ごとの基準点（同じ幾何・変換器・構成の 89 コマ。連結は同じクリップ形の 1280×768）・変換器と構成の切り替えごとに捨てラン（768×512×121）・時間切れ 3,600 秒 |
| 連結の形 | クリップ `[361,361]`・`stage2_window: standard`（w22）・**`chunked_upsample: true`**（`mk.sh` で明示・`calib.py preflight` で検査・各点の `request.json` と `server_slice.log` で確認）。overlap は送らない（3／0.5） |
| 既定値の突き合わせ | 計画の「既定値の突き合わせ表」のとおり。第 7 弾（10-03）と意図して違うのは連結の `chunked_upsample=true` だけ |
| コミット上限 | 計画時の実測 143.22 GiB・今回の監視の `limit_gib` **143.26 GiB**（監視 PID 18276）。90%＝128.9 GiB・95%＝136.1 GiB。**最大 101.66 GiB（71.0%）** |
| 基準点の異常の条件 | 監督裁定（10-05 09:38）で「溢れている／stage-2 の窓の標本が 10 未満／同じファイルの過去の基準点の stage-2 中央値と 300 MB 以上違う」に改めた（`idle-calib` の床＋300 MB は撤回。重みのファイルごとに常駐量が違うため）。基準点 18 本はすべて正常（捨てラン 9 本は判定なし） |

## 2. 腕ごとの表

列: 判定は v3′、delta は判定可能な窓のうち最大の delta（その窓の名前。stage-2 以外なら stage-2 の delta を添える）、標本はその窓の WDDM 標本数、所要は `wall.seconds`／`stage2.seconds`（秒）、コミットは点の最中の最大（GiB・上限比）。表は `manifest.jsonl` と `commit_table.md` から機械的に作った。
### sa

| 点 | 幾何 | トークン | 判定 | delta（効いた窓） | 標本 | 所要（総／stage-2） | コミット最大 |
|---|---|---:|---|---|---:|---|---|
| `sa_warm` | 768×512×121 | 6,144 | 捨てラン | — | — | 80／9 | 75.04（52.4%） |
| `sa_c_b1280x768` | 1280×768 `[361,361]` | 21,120 | 基準点 | stage2 中央値 703.1 | 137 | 319／148 | 78.00（54.4%） |
| `sa_c_1920x1024` | 1920×1024 `[361,361]` | 42,240 | 快適 | -22.5（s23_stage1）・stage2 -43.7 | 162 | 675／323 | 83.29（58.1%） |
| `sa_c_1984x1024` | 1984×1024 `[361,361]` | 43,648 | **溢れ** | +711.7（stage2） | 349 | 728／360 | 81.10（56.6%） |
| `sa_c_1920x1024_r2` | 1920×1024 `[361,361]` | 42,240 | 快適 | -56.6（s23_stage1）・stage2 -64.6 | 157 | 664／319 | 80.24（56.0%） |
| `sa_c_1984x1024_r2` | 1984×1024 `[361,361]` | 43,648 | **溢れ** | +553.4（stage2） | 344 | 724／357 | 80.87（56.4%） |

### qa

| 点 | 幾何 | トークン | 判定 | delta（効いた窓） | 標本 | 所要（総／stage-2） | コミット最大 |
|---|---|---:|---|---|---:|---|---|
| `qa_warm` | 768×512×121 | 6,144 | 捨てラン | — | — | 82／8 | 76.36（53.3%） |
| `qa_s_720_b89` | 1280×768×89 | 11,520 | 基準点 | stage2 中央値 1,148.8 | 11 | 50／16 | 75.92（53.0%） |
| `qa_s_720_353` | 1280×768×353 | 43,200 | 快適 | +10.6（stage2） | 46 | 151／70 | 82.83（57.8%） |
| `qa_s_720_353_r2` | 1280×768×353 | 43,200 | 快適 | +9.0（s23_gap_tail）・stage2 +3.3 | 36 | 150／70 | 83.07（58.0%） |
| `qa_s_720_361` | 1280×768×361 | 44,160 | **溢れ** | +947.2（stage2） | 49 | 156／74 | 85.24（59.5%） |
| `qa_s_1080_b89` | 1920×1088×89 | 24,480 | 基準点 | stage2 中央値 1,161.5 | 24 | 82／36 | 80.63（56.3%） |
| `qa_s_1080_161` | 1920×1088×161 | 42,840 | 快適 | +12.5（s23_stage1）・stage2 -0.3 | 34 | 152／69 | 83.62（58.4%） |
| `qa_s_1080_161_r2` | 1920×1088×161 | 42,840 | 快適 | +27.4（s23_stage1）・stage2 -25.5 | 34 | 152／69 | 83.54（58.3%） |
| `qa_c_b1280x768` | 1280×768 `[361,361]` | 21,120 | 基準点 | stage2 中央値 1,161.7 | 132 | 303／142 | 79.48（55.5%） |
| `qa_c_1920x1024` | 1920×1024 `[361,361]` | 42,240 | **溢れ** | +357.0（s23_gap_mid）・stage2 +351.3 | 29 | 670／321 | 84.49（59.0%） |
| `qa_c_1856x1024` | 1856×1024 `[361,361]` | 40,832 | 快適 | +3.3（s23_stage1）・stage2 -0.4 | 153 | 626／307 | 84.21（58.8%） |
| `qa_c_1856x1024_r2` | 1856×1024 `[361,361]` | 40,832 | 快適 | +77.0（s23_gap_mid）・stage2 +63.2 | 24 | 626／308 | 84.21（58.8%） |
| `qa_c_1920x1024_r2` | 1920×1024 `[361,361]` | 42,240 | **溢れ** | +359.3（s23_gap_mid）・stage2 +351.3 | 29 | 670／322 | 84.68（59.1%） |

### sb

| 点 | 幾何 | トークン | 判定 | delta（効いた窓） | 標本 | 所要（総／stage-2） | コミット最大 |
|---|---|---:|---|---|---:|---|---|
| `sb_warm` | 768×512×121 | 6,144 | 捨てラン | — | — | 56／9 | 59.30（41.4%） |
| `sb_c_b1280x768` | 1280×768 `[361,361]` | 21,120 | 基準点 | stage2 中央値 651.3 | 139 | 295／146 | 63.37（44.2%） |
| `sb_c_2048x1088` | 2048×1088 `[361,361]` | 47,872 | **溢れ** | +412.0（stage2） | 520 | 888／553 | 71.15（49.7%） |
| `sb_c_1984x1088` | 1984×1088 `[361,361]` | 46,376 | 快適 | +37.0（stage2） | 351 | 700／378 | 70.40（49.1%） |
| `sb_c_1984x1088_r2` | 1984×1088 `[361,361]` | 46,376 | 快適 | -3.0（p25_30_decode_encode）・stage2 -4.8 | 116 | 694／369 | 70.28（49.1%） |
| `sb_c_2048x1088_r2` | 2048×1088 `[361,361]` | 47,872 | 快適 | +97.5（stage2） | 401 | 760／426 | 70.65（49.3%） |

### sd

| 点 | 幾何 | トークン | 判定 | delta（効いた窓） | 標本 | 所要（総／stage-2） | コミット最大 |
|---|---|---:|---|---|---:|---|---|
| `sd_warm` | 768×512×121 | 6,144 | 捨てラン | — | — | 40／9 | 55.28（38.6%） |
| `sd_c_b1280x768` | 1280×768 `[361,361]` | 21,120 | 基準点 | stage2 中央値 654.3 | 174 | 353／183 | 51.22（35.8%） |
| `sd_c_1984x1088` | 1984×1088 `[361,361]` | 46,376 | 快適 | +10.2（stage2） | 516 | 941／552 | 58.22（40.6%） |

### qb

| 点 | 幾何 | トークン | 判定 | delta（効いた窓） | 標本 | 所要（総／stage-2） | コミット最大 |
|---|---|---:|---|---|---:|---|---|
| `qb_warm` | 768×512×121 | 6,144 | 捨てラン | — | — | 58／9 | 64.72（45.2%） |
| `qb_s_896_b89` | 896×1152×89 | 12,096 | 基準点 | stage2 中央値 1,177.3 | 11 | 46／17 | 66.96（46.7%） |
| `qb_s_896_337` | 896×1152×337 | 43,344 | 快適 | -9.9（stage2） | 47 | 147／70 | 75.36（52.6%） |
| `qb_s_896_337_r2` | 896×1152×337 | 43,344 | 快適 | -25.9（stage2） | 48 | 146／72 | 75.40（52.6%） |
| `qb_s_720_b89` | 1280×768×89 | 11,520 | 基準点 | stage2 中央値 1,184.4 | 10 | 42／16 | 67.02（46.8%） |
| `qb_s_720_353` | 1280×768×353 | 43,200 | 快適 | -32.7（p25_30_decode_encode）・stage2 -32.7 | 24 | 146／72 | 75.42（52.6%） |
| `qb_s_720_353_r2` | 1280×768×353 | 43,200 | 快適 | -32.7（p25_30_decode_encode）・stage2 -32.7 | 24 | 146／72 | 75.43（52.6%） |
| `qb_s_720_361` | 1280×768×361 | 44,160 | 快適 | +243.5（stage2） | 68 | 181／102 | 75.75（52.9%） |
| `qb_s_720_361_r2` | 1280×768×361 | 44,160 | 快適 | +287.3（stage2） | 66 | 177／99 | 75.77（52.9%） |
| `qb_s_1080_b89` | 1920×1088×89 | 24,480 | 基準点 | stage2 中央値 1,175.9 | 24 | 80／37 | 70.64（49.3%） |
| `qb_s_1080_161` | 1920×1088×161 | 42,840 | 快適 | -9.3（p25_30_decode_encode）・stage2 -41.0 | 24 | 144／69 | 75.42（52.6%） |
| `qb_s_1080_161_r2` | 1920×1088×161 | 42,840 | 快適 | -18.6（p25_30_decode_encode）・stage2 -41.0 | 24 | 144／71 | 75.38（52.6%） |
| `qb_c_b1280x768` | 1280×768 `[361,361]` | 21,120 | 基準点 | stage2 中央値 1,157.6 | 140 | 297／146 | 69.83（48.7%） |
| `qb_c_1920x1024` | 1920×1024 `[361,361]` | 42,240 | 快適 | +0.0（p25_30_decode_encode）・stage2 -21.7 | 107 | 622／327 | 75.26（52.5%） |
| `qb_c_1984x1024` | 1984×1024 `[361,361]` | 43,648 | 快適 | +42.3（stage2） | 333 | 660／354 | 75.68（52.8%） |
| `qb_c_2048x1024` | 2048×1024 `[361,361]` | 45,056 | **溢れ** | +514.1（stage2） | 623 | 972／659 | 76.55（53.4%） |
| `qb_c_1984x1024_r2` | 1984×1024 `[361,361]` | 43,648 | 快適 | +122.1（stage2） | 345 | 674／366 | 75.75（52.9%） |
| `qb_c_2048x1024_r2` | 2048×1024 `[361,361]` | 45,056 | **溢れ** | +608.4（stage2） | 595 | 947／632 | 76.10（53.1%） |

### ra

| 点 | 幾何 | トークン | 判定 | delta（効いた窓） | 標本 | 所要（総／stage-2） | コミット最大 |
|---|---|---:|---|---|---:|---|---|
| `ra_warm` | 768×512×121 | 6,144 | 捨てラン | — | — | 66／13 | 51.79（36.2%） |
| `ra_s_1080_b89` | 1920×1088×89 | 24,480 | 基準点 | stage2 中央値 1,176.7 | 34 | 122／51 | 67.41（47.1%） |
| `ra_s_1080_145` | 1920×1088×145 | 38,760 | 快適 | -16.5（p25_30_decode_encode）・stage2 -16.5 | 22 | 187／93 | 70.78（49.4%） |
| `ra_s_1080_153` | 1920×1088×153 | 40,800 | **溢れ** | +338.2（stage2） | 88 | 227／131 | 71.40（49.8%） |
| `ra_s_1080_145_r2` | 1920×1088×145 | 38,760 | 快適 | -16.2（stage2） | 62 | 185／93 | 71.20（49.7%） |
| `ra_s_1080_153_r2` | 1920×1088×153 | 40,800 | **溢れ** | +346.5（stage2） | 87 | 225／131 | 71.54（49.9%） |
| `ra_s_720_b89` | 1280×768×89 | 11,520 | 基準点 | stage2 中央値 1,177.0 | 15 | 70／22 | 64.09（44.7%） |
| `ra_s_720_313` | 1280×768×313 | 38,400 | 快適 | -8.7（p25_30_decode_encode）・stage2 -41.1 | 22 | 187／92 | 70.92（49.5%） |
| `ra_s_720_313_r2` | 1280×768×313 | 38,400 | 快適 | -17.2（p25_30_decode_encode）・stage2 -28.7 | 22 | 183／92 | 70.97（49.5%） |
| `ra_s_720_337` | 1280×768×337 | 41,280 | **溢れ** | +367.2（stage2） | 94 | 237／140 | 71.80（50.1%） |
| `ra_c_b1280x768` | 1280×768 `[361,361]` | 21,120 | 基準点 | stage2 中央値 1,157.4 | 191 | 395／202 | 66.84（46.7%） |
| `ra_c_1728x1024` | 1728×1024 `[361,361]` | 38,016 | 快適 | +15.9（p25_30_decode_encode）・stage2 -5.7 | 98 | 758／426 | 71.25（49.7%） |
| `ra_c_1792x1024` | 1792×1024 `[361,361]` | 39,424 | 快適 | +13.7（p25_30_decode_encode）・stage2 -21.5 | 101 | 796／447 | 71.46（49.9%） |
| `ra_c_1856x1024` | 1856×1024 `[361,361]` | 40,832 | **溢れ** | +421.1（stage2） | 575 | 975／615 | 72.01（50.3%） |
| `ra_c_1792x1024_r2` | 1792×1024 `[361,361]` | 39,424 | 快適 | +0.0（p25_30_decode_encode）・stage2 -5.7 | 102 | 794／447 | 71.67（50.0%） |
| `ra_c_1856x1024_r2` | 1856×1024 `[361,361]` | 40,832 | 快適 | +230.5（stage2） | 507 | 904／550 | 72.46（50.6%） |

### rb

| 点 | 幾何 | トークン | 判定 | delta（効いた窓） | 標本 | 所要（総／stage-2） | コミット最大 |
|---|---|---:|---|---|---:|---|---|
| `rb_warm` | 768×512×121 | 6,144 | 捨てラン | — | — | 50／13 | 71.92（50.2%） |
| `rb_c_b1280x768` | 1280×768 `[361,361]` | 21,120 | 基準点 | stage2 中央値 1,149.7 | 156 | 335／166 | 81.94（57.2%） |
| `rb_c_1792x1024` | 1792×1024 `[361,361]` | 39,424 | 快適 | +15.9（p25_30_decode_encode）・stage2 -14.7 | 102 | 615／321 | 86.09（60.1%） |

### ua

| 点 | 幾何 | トークン | 判定 | delta（効いた窓） | 標本 | 所要（総／stage-2） | コミット最大 |
|---|---|---:|---|---|---:|---|---|
| `ua_warm` | 768×512×121 | 6,144 | 捨てラン | — | — | 56／11 | 59.22（41.3%） |
| `ua_s_1080_b89` | 1920×1088×89 | 24,480 | 基準点 | stage2 中央値 1,159.7 | 32 | 112／48 | 74.98（52.3%） |
| `ua_s_1080_145` | 1920×1088×145 | 38,760 | 快適 | -9.0（stage2） | 60 | 183／90 | 78.13（54.5%） |
| `ua_s_1080_153` | 1920×1088×153 | 40,800 | **溢れ** | +634.9（stage2） | 99 | 235／148 | 78.51（54.8%） |

### ha

| 点 | 幾何 | トークン | 判定 | delta（効いた窓） | 標本 | 所要（総／stage-2） | コミット最大 |
|---|---|---:|---|---|---:|---|---|
| `ha_warm` | 768×512×121 | 6,144 | 捨てラン | — | — | 66／10 | 84.47（59.0%） |
| `ha_s_1080_b89` | 1920×1088×89 | 24,480 | 基準点 | stage2 中央値 2,184.1 | 25 | 90／37 | 97.40（68.0%） |
| `ha_s_1080_129` | 1920×1088×129 | 34,680 | **溢れ** | +929.6（stage2） | 39 | 130／58 | 101.16（70.6%） |
| `ha_s_1080_121` | 1920×1088×121 | 32,640 | 快適 | +127.0（stage2） | 34 | 118／51 | 100.41（70.1%） |
| `ha_s_1080_121_r2` | 1920×1088×121 | 32,640 | 快適 | +124.0（stage2） | 34 | 118／51 | 100.39（70.1%） |
| `ha_s_1080_129_r2` | 1920×1088×129 | 34,680 | **溢れ** | +753.9（stage2） | 38 | 128／58 | 101.25（70.7%） |
| `ha_s_720_b89` | 1280×768×89 | 11,520 | 基準点 | stage2 中央値 2,183.0 | 11 | 46／17 | 94.88（66.2%） |
| `ha_s_720_265` | 1280×768×265 | 32,640 | 快適 | +1.9（s23_stage1）・stage2 -29.3 | 29 | 118／51 | 100.36（70.1%） |
| `ha_s_720_265_r2` | 1280×768×265 | 32,640 | 快適 | +16.4（s23_stage1）・stage2 -27.4 | 30 | 118／51 | 100.44（70.1%） |
| `ha_s_720_289` | 1280×768×289 | 35,520 | **溢れ** | +1,063.6（stage2） | 39 | 128／58 | 101.66（71.0%） |
| `ha_c_b1280x768` | 1280×768 `[361,361]` | 21,120 | 基準点 | stage2 中央値 2,181.1 | 138 | 317／149 | 97.76（68.2%） |
| `ha_c_1472x1024` | 1472×1024 `[361,361]` | 32,384 | 快適 | +2.7（stage2） | 223 | 499／240 | 100.69（70.3%） |
| `ha_c_1536x1024` | 1536×1024 `[361,361]` | 33,792 | **溢れ** | +410.3（stage2） | 239 | 524／257 | 101.21（70.6%） |
| `ha_c_1472x1024_r2` | 1472×1024 `[361,361]` | 32,384 | 快適 | -15.9（s23_stage1）・stage2 -21.5 | 126 | 497／240 | 100.72（70.3%） |
| `ha_c_1536x1024_r2` | 1536×1024 `[361,361]` | 33,792 | **溢れ** | +418.3（stage2） | 239 | 524／257 | 101.24（70.7%） |

**腕のまとめ**（c＝2 回快適の上端・U＝c の 1 段上）:

| 腕 | セル | 変換器・構成 | 結果 |
|---|---|---|---|
| sa | 2.3 標準・連結 | `default`・全 on | c＝1920×1024（42,240）快適×2・U＝1984×1024（43,648）溢れ×2 |
| qa | 2.3 Q6_K・単発 | Sulphur Q6_K・全 on | 1280×768 353（43,200）快適×2・361（44,160）溢れ×1／1080p 161（42,840）快適×2 |
| qa | 2.3 Q6_K・連結 | 同上 | c＝1856×1024（40,832）快適×2・U＝1920×1024（42,240）溢れ×2 |
| sb | 2.5 標準・連結 | `default`・全 on | c＝1984×1088（46,376）快適×2・U＝2048×1088（47,872）**割れ**（+412.0 溢れ・+97.5 快適） |
| sd | 2.5 標準・連結の照合 | `default`・既定構成 | 1984×1088 快適×1（照合は終わり） |
| qb | 2.5 Q6_K・単発 | REDGraft Q6_K・全 on | 896×1152 337（43,344）快適×2／1280×768 353（43,200）快適×2・361（44,160）快適×2（+243.5／+287.3・閾値の際）／1080p 161（42,840）快適×2 |
| qb | 2.5 Q6_K・連結 | 同上 | c＝1984×1024（43,648）快適×2・U＝2048×1024（45,056）溢れ×2 |
| ra | 2.5 重い・単発 | REDGraft・既定構成 | 1080p c＝145（38,760）快適×2・U＝153（40,800）溢れ×2／1280×768 313（38,400）快適×2・337（41,280）溢れ×1 |
| ra | 2.5 重い・連結 | 同上 | c＝1792×1024（39,424）快適×2・U＝1856×1024（40,832）**割れ**（+421.1 溢れ・+230.5 快適） |
| rb | 2.5 重い・連結の照合 | REDGraft・全 on | 1792×1024 快適×1（照合は終わり） |
| ua | 2.5 重い・単発の照合 | uncensored fp8・既定構成 | 1080p 145 快適×1・153 溢れ×1（ra と同じ境界） |
| ha | 2.3 重い・単発 | fp8mixed・全 on | 1080p c＝121（32,640）快適×2・U＝129（34,680）溢れ×2／1280×768 265（32,640）快適×2・289（35,520）溢れ×1 |
| ha | 2.3 重い・連結 | 同上 | c＝1472×1024（32,384）快適×2・U＝1536×1024（33,792）溢れ×2 |

腕 ha の補足: 計画の出発点 S＝129 が溢れたので規則 2 で下った（121 快適）。c＝121 は計画の例示（c＝129）と違うので、1280×768 は規則 5 を読み替えて「c（32,640）以下で最大＝265 コマ（34 潜在×960）」「c＋1 段（34,680）以上で最小＝289 コマ（37×960＝35,520）」とし、連結の出発点は「c 以下で最も高い連結の段」＝1472×1024（46×32×22＝32,384。計画の早見表の 1536 の 1 段下）とした。

## 3. 新規則「既知の溢れ点より下の最も高い段」での各セルの定数候補

規則 6: セルの定数候補＝全幾何の既知の溢れ（割れを含む・過去日の記録を含む）のうち最小のもの M より下で、どれかの幾何で 2 回快適だった最も高い段。過去日の点は計画の棚卸し（一次記録の `request.json` で条件を確認済み）から引いた。

| セル | M（既知の溢れの最小）と出典 | 定数の候補 | 1280×768 換算（単発） | 備考 |
|---|---|---:|---|---|
| 2.3 標準・連結 | 43,648（1984×1024・今回 溢れ×2） | **42,240**（1920×1024 快適×2） | — | 第 7 弾（cu=false）の点は条件が違うので M に入れない |
| 2.5 標準・連結 | 47,872（2048×1088・今回 割れ） | **46,376**（1984×1088。全 on 快適×2・既定構成 快適×1） | — | 全 on と既定構成の小さいほう＝同じ値 |
| 2.3 重い・単発（fp8mixed） | 34,680（1080p 129・今回 溢れ×2）。ほかに 35,520（1280×768 289・今回 溢れ×1）・42,840（1080p 161・09-26 溢れ） | **32,640**（1080p 121 快適×2・1280×768 265 快適×2） | 34 潜在コマ＝**265 コマ** | 09-26 の既定構成（診断値）121 快適×2・129 溢れ×2 と同じ段 |
| 2.3 重い・連結（fp8mixed） | 33,792（1536×1024・今回 溢れ×2）。ほかに 参考（w46・cu=false）の 38,456（w46 1216×704・09-26 cu=false 溢れ×1。窓と cu が今回と違う参考点） | **32,384**（1472×1024 快適×2） | — | 単発の c（32,640）より 256 トークン低い |
| 2.5 重い・単発 | 40,800（1080p 153。ra 溢れ×2・ua 溢れ×1・09-27 全 on 溢れ×2）。ほかに 40,320（896×1152 313・09-26 `fp8_e4m3fn` 全 on 溢れ×2） | **38,760**（1080p 145。ra 快適×2・ua 快適×1・09-27 全 on 快適×2） | 40 潜在コマ＝**313 コマ**（38,400。ra 快適×2） | REDGraft と fp8 で同じ境界。既定構成と全 on でも同じ。候補 38,760 は 40,320 の追加で変わらない |
| 2.5 重い・連結 | 40,832（1856×1024・今回 ra 割れ。09-26 `fp8_e4m3fn`・cu=false・`[25,161]` で溢れ×2） | **39,424**（1792×1024。ra 既定構成 快適×2・rb 全 on 快適×1） | — | **単発の c（38,760）より高い**。2.5 の 1 行に単発と連結を同じ値で書くなら 38,760 |
| 2.3 Q6_K・単発 | 44,160（1280×768 361。今回 溢れ×1・09-17 溢れ×2・768×1280 361 も 09-17 溢れ×2） | **43,200**（1280×768 353 快適×2） | **353 コマ** | 1080p は 161（42,840）快適×2。09-17 の 1080p 169（44,880）快適×2 は M より上なので採らない |
| 2.3 Q6_K・連結 | 42,240（1920×1024・今回 溢れ×2）。ほかに 38,456（w46・1216×704・09-25 `a_w46`／`a_w46_r2` 溢れ×2・cu=false・クリップ `[361,361]`） | **40,832**（1856×1024 快適×2） | — | 単発の c より低い。**判断事項**: 38,456 の点は窓が w46 で cu=false（今回の連結は w22・cu=true）。同じ条件でないので M から外すか、含めて M＝38,456 とするかはオーナーの判断。含める場合、38,456 より下で 2 回快適の連結の点は今回は無い（09-25 の w43 35,948・w40 33,440 は快適×1）ので補足の計測が要る |
| 2.5 Q6_K・単発 | **判断事項**: (a) 44,160（1280×768 361・09-14 全 on の割れ +304.1／+289.1）を採る／(b) 今回の 361 快適×2（+243.5／+287.3）を採り、M を 45,120（全 on・09-14 369 溢れ×1）とする。既定構成の 09-14 の 896×1152 345（44,352）の割れを入れるなら 44,352 | (a) **43,344**（896×1152 337 快適×2）／(b) **44,160**（1280×768 361 快適×2） | (a) **353 コマ**（43,200・快適×2）／(b) **361 コマ** | (b) の M を 44,352 にしても候補 44,160 は変わらない（(a) の 43,344 も変わらない）。M の版の違いは第 5 節 |
| 2.5 Q6_K・連結 | 45,056（2048×1024・今回 溢れ×2） | **43,648**（1984×1024 快適×2） | — | 09-14 の既定構成・cu=false の 1984×1024 快適×2 と同じ c |

> **裁定（§148・2026-10-05）**: 09-25 の w46・1216×704・38,456（cu=false）は、窓（w46 対 w22）と分割アップサンプル（なし対あり）の両方が今回の条件と違うため M から外し、2.3 Q6_K・連結の候補は **40,832**（1856×1024）とする。cu=true での w46 は未計測（補足計測 W5 は自動許可の判定に拒否され 0 本で中止。経緯は §148）。

注:
- 単発と連結のどちらが低いかはセルで違う。連結のほうが低いのは 2.3 Q6_K（40,832＜43,200）・2.3 重い（32,384＜32,640）・2.3 標準（42,240＜第 15 節の単発 42,840）・2.5 標準（46,376＜第 15 節の単発 46,920）。連結のほうが高いのは 2.5 重い（39,424＞38,760）。2.5 Q6_K は同程度（43,648 と 43,344／44,160）。
- 2.3 重い（fp8mixed）の候補は、2.3 標準の単発（42,840）より約 1 万トークン低い。fp8mixed の基準点の stage-2 中央値（2,181〜2,184 MB）は Q4_K_M（650〜703 MB）より約 1.5 GB 多い。

## 4. 既存の較正値との整合

- **2.3 重い（fp8mixed）・09-26**: 09-26 の全 on は 4 本だけで 2 回一致の点は 0（1080p 161 溢れ・連結 w46 1216×704 溢れ）。今回、全 on の境界は 1080p 121 快適×2・129 溢れ×2 で、09-26 の既定構成（診断値）の 121／129 と同じ段だった（計画が出発点 129 を選んだ読み「全 on の境界も 121〜129 付近」は当たり、境界は 129 の下）。09-27 の C-4 較正（§1-32）の「2.3 既定は fp8・silveroxides が同じ段（121f）」とも整合。**コミットは 09-26 より約 6〜7.5 GiB 低い**（捨てラン 90.51→84.47・単発基準点 103.53→97.40・連結基準点 105.31→97.76 GiB。連結の基準点は 09-26 が 1216×704 で今回は 1280×768 と幾何が違う）。上限も 113.82→143.26 GiB に増えたので、上限比は 95.6〜97.2%（09-26・停止）→ 68〜71%（今回）。
- **2.3 Q6_K・09-17**: 1280×768 361 は 09-17 の溢れ×2（基準点 185 コマ・当時の規則の版）と今回（基準点 89 コマ・溢れ×1・+947.2）で一致。353（43,200）は今回はじめて測り快適×2。1080p 161 は 09-26 の対照（快適×1）と今回（快適×2）で一致。09-17 の連結はすべて cu=false で、今回の cu=true の連結（c＝1856×1024）とは比べない。
- **2.5 Q6_K・09-14**: 全 on の 1080p 161 は 09-14 快適×1・今回 快適×2 で一致。**1280×768 361 は食い違う**: 09-14 全 on は割れ（+304.1／+289.1・基準点 169 コマ）、今回は快適×2（+243.5／+287.3・基準点 89 コマ）。4 本とも delta が閾値 300 MB の ±60 MB に入り、今回の 2 本は stage-2 の時間も 353 より 37〜42% 長い（部分的な退避の兆し）。規則上の判定は過去と同日で食い違うので併記し、オーナーの判断事項とする（第 3 節）。369 は今回走らせていない。連結は 09-14 の既定構成・cu=false・`[25,161]` の c＝1984×1024 と、今回の全 on・cu=true・`[361,361]` の c が同じ。
- **2.5 重い・09-27（連結の `fp8_e4m3fn` だけ 09-26）**: 09-27 の全 on（uncensored fp8・REDGraft 混在・int8 ConvRot。`COMFORT_LIMIT_TABLE.md` 第 14.1 節）の 1080p 145 快適×2・153 溢れ×2、REDGraft の 1280×768 313 快適×2・337 溢れ×1 と、今回の既定構成（REDGraft・fp8 照合）が同じ境界。連結は 09-26 の主役が `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn`（いま手元に無いファイル）・cu=false・`[25,161]` で 1792 快適×2・1856 溢れ×2、今回は REDGraft・cu=true・`[361,361]` で 1792 快適×2・1856 割れ。c は同じ位置。（裏取りで日付を訂正）
- **09-27（§1-32・C-4）**: 「2.5 全 on は fp8・REDGraft・int8 ConvRot が同じ段（145f）」は、既定構成でも同じ段であることが今回分かった（ra・ua）。
- **第 15 節（第 7 弾・10-03・cu=false）**: 2.3 標準の連結は cu=false で 1920×1024 溢れ×2（`s23_gap_mid`）・1856 快適×2・1792 割れ、今回の cu=true では 1920×1024 快適×2・1984×1024 溢れ×2（stage-2 の窓）で、c が 1 段上がり、`s23_gap_mid` の退避が消えた（計測点の gap_mid は 25〜27 秒・delta は負）。2.5 標準の連結は cu=false で 2048×1088 快適×2・2112 溢れ×2、今回の cu=true では 2048×1088 割れ・1984×1088 快適×2 で、c が 1 段下がった。2.3 Q6_K の連結も動いた: 09-17 は cu=false・クリップ `[25,161]` で 1920×1088（44,880）が快適×1、今回は cu=true・`[361,361]` で 1920×1024（42,240）が溢れ×2。クリップの形も違うので単純比較はできない。単発は第 15 節の値のまま（今回は測っていない）。

## 5. 残る空きセル・留保

- **2.3 重いの代表は fp8mixed だけ**（オーナー決定 2）。int8 系（silveroxides・Kijai）の全 on は未計測（ブロック常駐量は fp8 と同じ帯 17.4〜18.8 GiB という記録はある）。
- **Q6_K の M は過去日の記録を含む**: 2.3 Q6_K の M（44,160）は今回の溢れ×1 と 09-17 の溢れ×2（基準点 185 コマ・当時の規則の版）。2.5 Q6_K の M の候補（44,160）は 09-14 全 on の割れ（基準点 169 コマ）。今回の基準点はいずれも 89 コマ。判定は「同じ解像度の基準点との相対」なので、基準点の長さの違いが delta に効く可能性がある（どの程度かは測っていない）。
- **cu=true での `s23_gap_mid` の判定可否**: 2.3 の連結の基準点 3 本（sa・qa・ha）の `s23_gap_mid` の標本はどれも 12（10 以上なので判定できた・長さ 11.5〜12.0 秒）。計測点は 19〜29 標本。2.5 には `s23_*` の窓が無い。
- **連結の U の割れ**: sb（2048×1088）と ra（1856×1024）で U が割れた。規則 3 どおり c はそのまま・U より上へは進んでいない。cu=true の連結の境界は 1 段の幅で揺れやすい。
- **単発の基準点の標本が際どい**: `qb_s_720_b89` は stage-2 の標本がちょうど 10、`qa_s_720_b89`・`qb_s_896_b89`・`ha_s_720_b89` は 11。
- 単発の基準点の stage-2 以外の窓（`restore`・`s23_*`・`p25_*`）は標本 5〜9 のものがあり、計測点の判定の比較相手として使われている（今回これで判定が変わった点は無い）。
- コミット監視の実際の間隔は約 3 秒（計画は 2 秒）で、14:43 に 36 秒の欠けがある（判定には関係しない）。
- LTX 2.3 では `keep_resident_embeddings_used` が `metadata.json` に無く照合していない（2.3 は埋め込み常駐を 422 で断る設計のため影響なし）。
- 閾値はコードでは「+300 を超える」、文書では「+300 以上」（300.0 ちょうどの点は無い）。
- 2.5 Q6_K は今回、既定構成を 1 本も照合していない（2.5 の 1 行は両構成で成り立つ値が要る）。
- 2.5 標準・既定構成の連結は照合の 1 点だけ（c の 1 段上は測っていない＝計画どおり）。2.3 Q6_K の 1080p 169 は M より上なので走らせていない。
- 速度の兆し（stage-2 の時間が隣の段より大きく伸びる）は判定に使っていない。時間の集計は工程 3（別担当）。

## 6. 逸脱と事故

1. **10-04 の中断**: 担当 W1 の `calib.py server start --pid 10244` が自動許可の判定に拒否され（理由: Interfere With Workloads）、計測 0 本で中断・オーナー裁定で中止。10-05 に監督がサーバー（PID 32708）と監視（PID 18276）を起こし、`server start --pid`・`luid` を監督が打って再開した。`original_state.json`（10-05 08:14:52）は LTX23 の選択が `sulphur_distil_fp8mixed`（10-04 の `default` とは違う。オーナーの操作後の状態）。
2. **基準点の異常の条件の変更**: `qa_s_720_b89`（stage-2 1,148.8 MB）が計画の文言「`idle-calib` の床＋300 MB 超」に当たり W1 が止めて報告→監督裁定（09:38）で「同じファイルの過去の基準点の stage-2 中央値と 300 MB 以上違う」に改め、以後の全腕に適用。
3. **ha の読み替え**: 計画の例示（c＝129）と違って c＝121 になったので、1280×768 の点（265・289）と連結の出発点（1472×1024・早見表に無い段）を規則 5・8 から読み替えた（第 2 節の補足）。
4. **qa の 361 は溢れ×1 で止め、qb の 361 は快適だったので 2 回目を走らせた**（指示書どおり。同日の照合は「快適なら ×2 にして過去の記録と併記」）。
5. 進行記録 `progress.md` の W3 の行「ra の本数 14（捨 1・基 3・測 10）」は、`manifest.jsonl` では 16（捨 1・基 3・測 12）。W3 の合計 23 は正しい（ra 16・rb 3・ua 4）。
6. LTX23 の全点で `preflight` が「prompt not pinned」の警告を出したが、第 7 弾と同じで、送った `prompt` は較正台の固定文（`request_sent.prompt`）だった。
7. `sd_warm` のコミット最大 55.28 GiB は点の開始 2 秒前の値（点の最中の最大は 52.73 GiB）。
8. `one.sh` は較正台フォルダではなくセッションの scratchpad に置いた（`mk.sh`→`cool.sh`→`rp.sh` を順に呼ぶだけ）。
9. 事故はなし。失敗 0・422 なし・時間切れなし・`used_matches_request` の不一致なし・コミット 90% 到達なし・監視の停滞（exit 6）なし・`drive2.sh` の停止（exit 4）なし。

## 7. 原状復帰

- 20:12 `restore-state --base-model LTX25`（LTX25 の transformer を `redgraftLTX25Fast2K_ltx25RedgraftNSFW` に）→ 20:12 `restore-state --base-model LTX23`（有効ベースモデル LTX23・transformer `sulphur_distil_fp8mixed`）。どちらも応答と `GET /models` が一致し、`state.json verified`。自動許可の判定による拒否はなかった。
- `state.json` は `original_state.json` の `active_base_model`・`selections` と完全一致。
- 20:13 `POST /api/v1/pipeline/unload` → `{"pipeline_loaded":false,"state":"unloaded"}`。`GET /status` は `unloaded`・`base_model: LTX23`・待ち行列 0（completed 84・failed 0）。
- サーバー（PID 32708）と監視（PID 18276）は稼働のまま（停止は監督）。裏で動く `rp.sh`／`drive2.sh` は無い。

## 付録 A: `make_results.py` の出力

### 全計測点

| ラベル | 系統 | 幾何 | トークン | 状態 | v1 | v2 | **v3′** | 加速照合 |
|---|---|---|---:|---|---|---|---|---|
| `sa_warm` | LTX23 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `sa_c_b1280x768` | LTX23 | 1280x768/722f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `sa_c_1920x1024` | LTX23 | 1920x1024/722f | 42,240 | completed | inconclusive | organic | **clean** | ok |
| `sa_c_1984x1024` | LTX23 | 1984x1024/722f | 43,648 | completed | inconclusive | organic | **plateau** | ok |
| `sa_c_1920x1024_r2` | LTX23 | 1920x1024/722f | 42,240 | completed | inconclusive | organic | **clean** | ok |
| `sa_c_1984x1024_r2` | LTX23 | 1984x1024/722f | 43,648 | completed | inconclusive | organic | **plateau** | ok |
| `qa_warm` | LTX23 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `qa_s_720_b89` | LTX23 | 1280x768/89f | 11,520 | completed | organic | baseline | **baseline** | ok |
| `qa_s_720_353` | LTX23 | 1280x768/353f | 43,200 | completed | inconclusive | organic | **clean** | ok |
| `qa_s_720_353_r2` | LTX23 | 1280x768/353f | 43,200 | completed | inconclusive | organic | **clean** | ok |
| `qa_s_720_361` | LTX23 | 1280x768/361f | 44,160 | completed | harmful | harmful | **plateau** | ok |
| `qa_s_1080_b89` | LTX23 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `qa_s_1080_161` | LTX23 | 1920x1088/161f | 42,840 | completed | inconclusive | organic | **clean** | ok |
| `qa_s_1080_161_r2` | LTX23 | 1920x1088/161f | 42,840 | completed | inconclusive | organic | **clean** | ok |
| `qa_c_b1280x768` | LTX23 | 1280x768/722f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `qa_c_1920x1024` | LTX23 | 1920x1024/722f | 42,240 | completed | harmful | organic | **plateau** | ok |
| `qa_c_1856x1024` | LTX23 | 1856x1024/722f | 40,832 | completed | inconclusive | organic | **clean** | ok |
| `qa_c_1856x1024_r2` | LTX23 | 1856x1024/722f | 40,832 | completed | harmful | organic | **clean** | ok |
| `qa_c_1920x1024_r2` | LTX23 | 1920x1024/722f | 42,240 | completed | harmful | organic | **plateau** | ok |
| `sb_warm` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `sb_c_b1280x768` | LTX25 | 1280x768/722f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `sb_c_2048x1088` | LTX25 | 2048x1088/722f | 47,872 | completed | harmful | harmful | **plateau** | ok |
| `sb_c_1984x1088` | LTX25 | 1984x1088/722f | 46,376 | completed | inconclusive | organic | **clean** | ok |
| `sb_c_1984x1088_r2` | LTX25 | 1984x1088/722f | 46,376 | completed | inconclusive | organic | **clean** | ok |
| `sb_c_2048x1088_r2` | LTX25 | 2048x1088/722f | 47,872 | completed | inconclusive | organic | **clean** | ok |
| `sd_warm` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `sd_c_b1280x768` | LTX25 | 1280x768/722f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `sd_c_1984x1088` | LTX25 | 1984x1088/722f | 46,376 | completed | inconclusive | organic | **clean** | ok |
| `qb_warm` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `qb_s_896_b89` | LTX25 | 896x1152/89f | 12,096 | completed | organic | baseline | **baseline** | ok |
| `qb_s_896_337` | LTX25 | 896x1152/337f | 43,344 | completed | inconclusive | organic | **clean** | ok |
| `qb_s_896_337_r2` | LTX25 | 896x1152/337f | 43,344 | completed | harmful | organic | **clean** | ok |
| `qb_s_720_b89` | LTX25 | 1280x768/89f | 11,520 | completed | organic | baseline | **baseline** | ok |
| `qb_s_720_353` | LTX25 | 1280x768/353f | 43,200 | completed | inconclusive | organic | **clean** | ok |
| `qb_s_720_353_r2` | LTX25 | 1280x768/353f | 43,200 | completed | inconclusive | organic | **clean** | ok |
| `qb_s_720_361` | LTX25 | 1280x768/361f | 44,160 | completed | harmful | organic | **clean** | ok |
| `qb_s_720_361_r2` | LTX25 | 1280x768/361f | 44,160 | completed | harmful | organic | **clean** | ok |
| `qb_s_1080_b89` | LTX25 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `qb_s_1080_161` | LTX25 | 1920x1088/161f | 42,840 | completed | inconclusive | organic | **clean** | ok |
| `qb_s_1080_161_r2` | LTX25 | 1920x1088/161f | 42,840 | completed | harmful | organic | **clean** | ok |
| `qb_c_b1280x768` | LTX25 | 1280x768/722f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `qb_c_1920x1024` | LTX25 | 1920x1024/722f | 42,240 | completed | inconclusive | organic | **clean** | ok |
| `qb_c_1984x1024` | LTX25 | 1984x1024/722f | 43,648 | completed | harmful | organic | **clean** | ok |
| `qb_c_2048x1024` | LTX25 | 2048x1024/722f | 45,056 | completed | harmful | harmful | **plateau** | ok |
| `qb_c_1984x1024_r2` | LTX25 | 1984x1024/722f | 43,648 | completed | harmful | organic | **clean** | ok |
| `qb_c_2048x1024_r2` | LTX25 | 2048x1024/722f | 45,056 | completed | harmful | harmful | **plateau** | ok |
| `ra_warm` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `ra_s_1080_b89` | LTX25 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `ra_s_1080_145` | LTX25 | 1920x1088/145f | 38,760 | completed | inconclusive | organic | **clean** | ok |
| `ra_s_1080_153` | LTX25 | 1920x1088/153f | 40,800 | completed | inconclusive | harmful | **plateau** | ok |
| `ra_s_1080_145_r2` | LTX25 | 1920x1088/145f | 38,760 | completed | inconclusive | organic | **clean** | ok |
| `ra_s_1080_153_r2` | LTX25 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `ra_s_720_b89` | LTX25 | 1280x768/89f | 11,520 | completed | organic | baseline | **baseline** | ok |
| `ra_s_720_313` | LTX25 | 1280x768/313f | 38,400 | completed | inconclusive | organic | **clean** | ok |
| `ra_s_720_313_r2` | LTX25 | 1280x768/313f | 38,400 | completed | inconclusive | organic | **clean** | ok |
| `ra_s_720_337` | LTX25 | 1280x768/337f | 41,280 | completed | inconclusive | harmful | **plateau** | ok |
| `ra_c_b1280x768` | LTX25 | 1280x768/722f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `ra_c_1728x1024` | LTX25 | 1728x1024/722f | 38,016 | completed | inconclusive | organic | **clean** | ok |
| `ra_c_1792x1024` | LTX25 | 1792x1024/722f | 39,424 | completed | inconclusive | organic | **clean** | ok |
| `ra_c_1856x1024` | LTX25 | 1856x1024/722f | 40,832 | completed | inconclusive | harmful | **plateau** | ok |
| `ra_c_1792x1024_r2` | LTX25 | 1792x1024/722f | 39,424 | completed | inconclusive | organic | **clean** | ok |
| `ra_c_1856x1024_r2` | LTX25 | 1856x1024/722f | 40,832 | completed | inconclusive | organic | **clean** | ok |
| `rb_warm` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `rb_c_b1280x768` | LTX25 | 1280x768/722f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `rb_c_1792x1024` | LTX25 | 1792x1024/722f | 39,424 | completed | inconclusive | organic | **clean** | ok |
| `ua_warm` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `ua_s_1080_b89` | LTX25 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `ua_s_1080_145` | LTX25 | 1920x1088/145f | 38,760 | completed | inconclusive | organic | **clean** | ok |
| `ua_s_1080_153` | LTX25 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `ha_warm` | LTX23 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `ha_s_1080_b89` | LTX23 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `ha_s_1080_129` | LTX23 | 1920x1088/129f | 34,680 | completed | organic | organic | **plateau** | ok |
| `ha_s_1080_121` | LTX23 | 1920x1088/121f | 32,640 | completed | harmful | organic | **clean** | ok |
| `ha_s_1080_121_r2` | LTX23 | 1920x1088/121f | 32,640 | completed | harmful | organic | **clean** | ok |
| `ha_s_1080_129_r2` | LTX23 | 1920x1088/129f | 34,680 | completed | organic | organic | **plateau** | ok |
| `ha_s_720_b89` | LTX23 | 1280x768/89f | 11,520 | completed | organic | baseline | **baseline** | ok |
| `ha_s_720_265` | LTX23 | 1280x768/265f | 32,640 | completed | harmful | organic | **clean** | ok |
| `ha_s_720_265_r2` | LTX23 | 1280x768/265f | 32,640 | completed | inconclusive | organic | **clean** | ok |
| `ha_s_720_289` | LTX23 | 1280x768/289f | 35,520 | completed | inconclusive | organic | **plateau** | ok |
| `ha_c_b1280x768` | LTX23 | 1280x768/722f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `ha_c_1472x1024` | LTX23 | 1472x1024/722f | 32,384 | completed | inconclusive | organic | **clean** | ok |
| `ha_c_1536x1024` | LTX23 | 1536x1024/722f | 33,792 | completed | inconclusive | organic | **plateau** | ok |
| `ha_c_1472x1024_r2` | LTX23 | 1472x1024/722f | 32,384 | completed | inconclusive | organic | **clean** | ok |
| `ha_c_1536x1024_r2` | LTX23 | 1536x1024/722f | 33,792 | completed | inconclusive | organic | **plateau** | ok |

### 加速が実際に効いたか（`metadata.json` の `*_used`）

| ラベル | transformer | attention | block_swap_prefetch | keep_resident | fused_gguf_dequant_kernel | vae_mode | keep_resident_embeddings | ok |
|---|---|---|---|---|---|---|---|---|
| `sa_warm` | `default` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `sa_c_b1280x768` | `default` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `sa_c_1920x1024` | `default` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `sa_c_1984x1024` | `default` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `sa_c_1920x1024_r2` | `default` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `sa_c_1984x1024_r2` | `default` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_warm` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_s_720_b89` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_s_720_353` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_s_720_353_r2` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_s_720_361` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_s_1080_b89` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_s_1080_161` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_s_1080_161_r2` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_c_b1280x768` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_c_1920x1024` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_c_1856x1024` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_c_1856x1024_r2` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `qa_c_1920x1024_r2` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `sb_warm` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `sb_c_b1280x768` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `sb_c_2048x1088` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `sb_c_1984x1088` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `sb_c_1984x1088_r2` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `sb_c_2048x1088_r2` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `sd_warm` | `default` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `sd_c_b1280x768` | `default` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `sd_c_1984x1088` | `default` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `qb_warm` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_s_896_b89` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_s_896_337` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_s_896_337_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_s_720_b89` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_s_720_353` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_s_720_353_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_s_720_361` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_s_720_361_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_s_1080_b89` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_s_1080_161` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_s_1080_161_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_c_b1280x768` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_c_1920x1024` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_c_1984x1024` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_c_2048x1024` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_c_1984x1024_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `qb_c_2048x1024_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `ra_warm` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_s_1080_b89` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_s_1080_145` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_s_1080_153` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_s_1080_145_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_s_1080_153_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_s_720_b89` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_s_720_313` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_s_720_313_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_s_720_337` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_c_b1280x768` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_c_1728x1024` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_c_1792x1024` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_c_1856x1024` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_c_1792x1024_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ra_c_1856x1024_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `rb_warm` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `rb_c_b1280x768` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `rb_c_1792x1024` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `ua_warm` | `ltx25_uncensored_v1.1-fp8_scaled` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ua_s_1080_b89` | `ltx25_uncensored_v1.1-fp8_scaled` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ua_s_1080_145` | `ltx25_uncensored_v1.1-fp8_scaled` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ua_s_1080_153` | `ltx25_uncensored_v1.1-fp8_scaled` | `sdpa` | `on` | `off` | `on` | `conv` | `off` | ok |
| `ha_warm` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_s_1080_b89` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_s_1080_129` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_s_1080_121` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_s_1080_121_r2` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_s_1080_129_r2` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_s_720_b89` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_s_720_265` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_s_720_265_r2` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_s_720_289` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_c_b1280x768` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_c_1472x1024` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_c_1536x1024` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_c_1472x1024_r2` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `ha_c_1536x1024_r2` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |

`keep_resident_embeddings` が 2.3 の点で *null* なのは仕様です（`services/engines/ltx/adapter.py:481`。2.3 に埋め込み処理器が無い）。

### v3′ の窓ごとの判定（基準比・+300MB が線）

**`sa_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 7 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 4 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 7 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 8 | × fewer than 10 WDDM samples | — | — |
| stage2 | 5 | × fewer than 10 WDDM samples | — | — |

**`sa_c_b1280x768`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 74 | × no baseline for this window | — | — |
| s23_gap_mid | 12 | × no baseline for this window | — | — |
| s23_gap_tail | 74 | × no baseline for this window | — | — |
| s23_stage1 | 80 | × no baseline for this window | — | — |
| stage2 | 137 | × no baseline for this window | — | — |

**`sa_c_1920x1024`** — v3′=`clean`（基準 `sa_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 169 | ○ | -72.3 | 下 |
| s23_gap_mid | 26 | ○ | -22.6 | 下 |
| s23_gap_tail | 169 | ○ | -72.3 | 下 |
| s23_stage1 | 162 | ○ | -22.5 | 下 |
| stage2 | 300 | ○ | -43.7 | 下 |

**`sa_c_1984x1024`** — v3′=`plateau`（基準 `sa_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 171 | ○ | -66.6 | 下 |
| s23_gap_mid | 27 | ○ | -56.5 | 下 |
| s23_gap_tail | 171 | ○ | -66.6 | 下 |
| s23_stage1 | 164 | ○ | -56.5 | 下 |
| stage2 | 349 | ○ | 711.7 | **超** |

**`sa_c_1920x1024_r2`** — v3′=`clean`（基準 `sa_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 167 | ○ | -71.3 | 下 |
| s23_gap_mid | 25 | ○ | -56.6 | 下 |
| s23_gap_tail | 167 | ○ | -71.3 | 下 |
| s23_stage1 | 157 | ○ | -56.6 | 下 |
| stage2 | 297 | ○ | -64.6 | 下 |

**`sa_c_1984x1024_r2`** — v3′=`plateau`（基準 `sa_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 171 | ○ | -82.4 | 下 |
| s23_gap_mid | 27 | ○ | -59.1 | 下 |
| s23_gap_tail | 171 | ○ | -82.4 | 下 |
| s23_stage1 | 164 | ○ | -56.3 | 下 |
| stage2 | 344 | ○ | 553.4 | **超** |

**`qa_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 6 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 4 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 6 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 8 | × fewer than 10 WDDM samples | — | — |
| stage2 | 6 | × fewer than 10 WDDM samples | — | — |

**`qa_s_720_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 8 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 5 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 8 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 11 | × no baseline for this window | — | — |
| stage2 | 11 | × no baseline for this window | — | — |

**`qa_s_720_353`** — v3′=`clean`（基準 `qa_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 36 | ○ | 2.5 | 下 |
| s23_gap_mid | 24 | ○ | -15.3 | 下 |
| s23_gap_tail | 36 | ○ | 2.5 | 下 |
| s23_stage1 | 35 | ○ | -15.3 | 下 |
| stage2 | 46 | ○ | 10.6 | 下 |

**`qa_s_720_353_r2`** — v3′=`clean`（基準 `qa_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 36 | ○ | 9.0 | 下 |
| s23_gap_mid | 24 | ○ | -12.5 | 下 |
| s23_gap_tail | 36 | ○ | 9.0 | 下 |
| s23_stage1 | 35 | ○ | -14.3 | 下 |
| stage2 | 46 | ○ | 3.3 | 下 |

**`qa_s_720_361`** — v3′=`plateau`（基準 `qa_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 36 | ○ | -21.4 | 下 |
| s23_gap_mid | 25 | ○ | -7.1 | 下 |
| s23_gap_tail | 36 | ○ | -21.4 | 下 |
| s23_stage1 | 36 | ○ | -7.1 | 下 |
| stage2 | 49 | ○ | 947.2 | **超** |

**`qa_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 19 | × no baseline for this window | — | — |
| s23_gap_mid | 12 | × no baseline for this window | — | — |
| s23_gap_tail | 19 | × no baseline for this window | — | — |
| s23_stage1 | 20 | × no baseline for this window | — | — |
| stage2 | 24 | × no baseline for this window | — | — |

**`qa_s_1080_161`** — v3′=`clean`（基準 `qa_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 38 | ○ | -8.5 | 下 |
| s23_gap_mid | 24 | ○ | 10.4 | 下 |
| s23_gap_tail | 38 | ○ | -8.5 | 下 |
| s23_stage1 | 34 | ○ | 12.5 | 下 |
| stage2 | 46 | ○ | -0.3 | 下 |

**`qa_s_1080_161_r2`** — v3′=`clean`（基準 `qa_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 38 | ○ | -25.4 | 下 |
| s23_gap_mid | 24 | ○ | 27.4 | 下 |
| s23_gap_tail | 38 | ○ | -25.4 | 下 |
| s23_stage1 | 34 | ○ | 27.4 | 下 |
| stage2 | 46 | ○ | -25.5 | 下 |

**`qa_c_b1280x768`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 71 | × no baseline for this window | — | — |
| s23_gap_mid | 12 | × no baseline for this window | — | — |
| s23_gap_tail | 71 | × no baseline for this window | — | — |
| s23_stage1 | 77 | × no baseline for this window | — | — |
| stage2 | 132 | × no baseline for this window | — | — |

**`qa_c_1920x1024`** — v3′=`plateau`（基準 `qa_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 167 | ○ | -15.9 | 下 |
| s23_gap_mid | 29 | ○ | 357.0 | **超** |
| s23_gap_tail | 167 | ○ | -15.9 | 下 |
| s23_stage1 | 157 | ○ | -4.7 | 下 |
| stage2 | 300 | ○ | 351.3 | **超** |

**`qa_c_1856x1024`** — v3′=`clean`（基準 `qa_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 147 | ○ | -10.5 | 下 |
| s23_gap_mid | 24 | ○ | -0.5 | 下 |
| s23_gap_tail | 147 | ○ | -10.5 | 下 |
| s23_stage1 | 153 | ○ | 3.3 | 下 |
| stage2 | 286 | ○ | -0.4 | 下 |

**`qa_c_1856x1024_r2`** — v3′=`clean`（基準 `qa_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 148 | ○ | -7.1 | 下 |
| s23_gap_mid | 24 | ○ | 77.0 | 下 |
| s23_gap_tail | 148 | ○ | -7.1 | 下 |
| s23_stage1 | 153 | ○ | 11.2 | 下 |
| stage2 | 285 | ○ | 63.2 | 下 |

**`qa_c_1920x1024_r2`** — v3′=`plateau`（基準 `qa_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 167 | ○ | -24.1 | 下 |
| s23_gap_mid | 29 | ○ | 359.3 | **超** |
| s23_gap_tail | 167 | ○ | -24.1 | 下 |
| s23_stage1 | 158 | ○ | 11.2 | 下 |
| stage2 | 299 | ○ | 351.3 | **超** |

**`sb_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 5 | × fewer than 10 WDDM samples | — | — |

**`sb_c_b1280x768`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 53 | × no baseline for this window | — | — |
| restore | 53 | × no baseline for this window | — | — |
| stage2 | 139 | × no baseline for this window | — | — |

**`sb_c_2048x1088`** — v3′=`plateau`（基準 `sb_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 121 | ○ | -3.0 | 下 |
| restore | 121 | ○ | -3.0 | 下 |
| stage2 | 520 | ○ | 412.0 | **超** |

**`sb_c_1984x1088`** — v3′=`clean`（基準 `sb_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 117 | ○ | 5.2 | 下 |
| restore | 117 | ○ | 5.2 | 下 |
| stage2 | 351 | ○ | 37.0 | 下 |

**`sb_c_1984x1088_r2`** — v3′=`clean`（基準 `sb_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 116 | ○ | -3.0 | 下 |
| restore | 116 | ○ | -3.0 | 下 |
| stage2 | 344 | ○ | -4.8 | 下 |

**`sb_c_2048x1088_r2`** — v3′=`clean`（基準 `sb_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 120 | ○ | -11.3 | 下 |
| restore | 120 | ○ | -11.3 | 下 |
| stage2 | 401 | ○ | 97.5 | 下 |

**`sd_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 6 | × fewer than 10 WDDM samples | — | — |

**`sd_c_b1280x768`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 52 | × no baseline for this window | — | — |
| restore | 52 | × no baseline for this window | — | — |
| stage2 | 174 | × no baseline for this window | — | — |

**`sd_c_1984x1088`** — v3′=`clean`（基準 `sd_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 116 | ○ | -6.9 | 下 |
| restore | 116 | ○ | -6.9 | 下 |
| stage2 | 516 | ○ | 10.2 | 下 |

**`qb_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 6 | × fewer than 10 WDDM samples | — | — |

**`qb_s_896_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 6 | × fewer than 10 WDDM samples | — | — |
| restore | 6 | × fewer than 10 WDDM samples | — | — |
| stage2 | 11 | × no baseline for this window | — | — |

**`qb_s_896_337`** — v3′=`clean`（基準 `qb_s_896_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 25 | ○ | -42.8 | 下 |
| restore | 25 | ○ | -42.8 | 下 |
| stage2 | 47 | ○ | -9.9 | 下 |

**`qb_s_896_337_r2`** — v3′=`clean`（基準 `qb_s_896_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 25 | ○ | -49.9 | 下 |
| restore | 25 | ○ | -49.9 | 下 |
| stage2 | 48 | ○ | -25.9 | 下 |

**`qb_s_720_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 6 | × fewer than 10 WDDM samples | — | — |
| restore | 6 | × fewer than 10 WDDM samples | — | — |
| stage2 | 10 | × no baseline for this window | — | — |

**`qb_s_720_353`** — v3′=`clean`（基準 `qb_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -32.7 | 下 |
| restore | 24 | ○ | -32.7 | 下 |
| stage2 | 48 | ○ | -32.7 | 下 |

**`qb_s_720_353_r2`** — v3′=`clean`（基準 `qb_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -32.7 | 下 |
| restore | 24 | ○ | -32.7 | 下 |
| stage2 | 48 | ○ | -32.7 | 下 |

**`qb_s_720_361`** — v3′=`clean`（基準 `qb_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 26 | ○ | -48.5 | 下 |
| restore | 26 | ○ | -48.5 | 下 |
| stage2 | 68 | ○ | 243.5 | 下 |

**`qb_s_720_361_r2`** — v3′=`clean`（基準 `qb_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 26 | ○ | -32.5 | 下 |
| restore | 26 | ○ | -32.5 | 下 |
| stage2 | 66 | ○ | 287.3 | 下 |

**`qb_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 13 | × no baseline for this window | — | — |
| restore | 13 | × no baseline for this window | — | — |
| stage2 | 24 | × no baseline for this window | — | — |

**`qb_s_1080_161`** — v3′=`clean`（基準 `qb_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -9.3 | 下 |
| restore | 24 | ○ | -9.3 | 下 |
| stage2 | 46 | ○ | -41.0 | 下 |

**`qb_s_1080_161_r2`** — v3′=`clean`（基準 `qb_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -18.6 | 下 |
| restore | 24 | ○ | -18.6 | 下 |
| stage2 | 47 | ○ | -41.0 | 下 |

**`qb_c_b1280x768`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 52 | × no baseline for this window | — | — |
| restore | 52 | × no baseline for this window | — | — |
| stage2 | 140 | × no baseline for this window | — | — |

**`qb_c_1920x1024`** — v3′=`clean`（基準 `qb_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 107 | ○ | 0.0 | 下 |
| restore | 107 | ○ | 0.0 | 下 |
| stage2 | 308 | ○ | -21.7 | 下 |

**`qb_c_1984x1024`** — v3′=`clean`（基準 `qb_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 111 | ○ | 8.5 | 下 |
| restore | 111 | ○ | 8.5 | 下 |
| stage2 | 333 | ○ | 42.3 | 下 |

**`qb_c_2048x1024`** — v3′=`plateau`（基準 `qb_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 114 | ○ | 0.0 | 下 |
| restore | 114 | ○ | 0.0 | 下 |
| stage2 | 623 | ○ | 514.1 | **超** |

**`qb_c_1984x1024_r2`** — v3′=`clean`（基準 `qb_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 111 | ○ | 9.6 | 下 |
| restore | 111 | ○ | 9.6 | 下 |
| stage2 | 345 | ○ | 122.1 | 下 |

**`qb_c_2048x1024_r2`** — v3′=`plateau`（基準 `qb_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 114 | ○ | -7.7 | 下 |
| restore | 114 | ○ | -7.7 | 下 |
| stage2 | 595 | ○ | 608.4 | **超** |

**`ra_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 8 | × fewer than 10 WDDM samples | — | — |

**`ra_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 13 | × no baseline for this window | — | — |
| restore | 13 | × no baseline for this window | — | — |
| stage2 | 34 | × no baseline for this window | — | — |

**`ra_s_1080_145`** — v3′=`clean`（基準 `ra_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -16.5 | 下 |
| restore | 22 | ○ | -16.5 | 下 |
| stage2 | 62 | ○ | -16.5 | 下 |

**`ra_s_1080_153`** — v3′=`plateau`（基準 `ra_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -42.7 | 下 |
| restore | 23 | ○ | -42.7 | 下 |
| stage2 | 88 | ○ | 338.2 | **超** |

**`ra_s_1080_145_r2`** — v3′=`clean`（基準 `ra_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -16.5 | 下 |
| restore | 22 | ○ | -16.5 | 下 |
| stage2 | 62 | ○ | -16.2 | 下 |

**`ra_s_1080_153_r2`** — v3′=`plateau`（基準 `ra_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -45.7 | 下 |
| restore | 23 | ○ | -45.7 | 下 |
| stage2 | 87 | ○ | 346.5 | **超** |

**`ra_s_720_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 6 | × fewer than 10 WDDM samples | — | — |
| restore | 6 | × fewer than 10 WDDM samples | — | — |
| stage2 | 15 | × no baseline for this window | — | — |

**`ra_s_720_313`** — v3′=`clean`（基準 `ra_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -8.7 | 下 |
| restore | 22 | ○ | -8.7 | 下 |
| stage2 | 61 | ○ | -41.1 | 下 |

**`ra_s_720_313_r2`** — v3′=`clean`（基準 `ra_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -17.2 | 下 |
| restore | 22 | ○ | -17.2 | 下 |
| stage2 | 62 | ○ | -28.7 | 下 |

**`ra_s_720_337`** — v3′=`plateau`（基準 `ra_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -0.2 | 下 |
| restore | 23 | ○ | -0.2 | 下 |
| stage2 | 94 | ○ | 367.2 | **超** |

**`ra_c_b1280x768`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 53 | × no baseline for this window | — | — |
| restore | 53 | × no baseline for this window | — | — |
| stage2 | 191 | × no baseline for this window | — | — |

**`ra_c_1728x1024`** — v3′=`clean`（基準 `ra_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 98 | ○ | 15.9 | 下 |
| restore | 98 | ○ | 15.9 | 下 |
| stage2 | 399 | ○ | -5.7 | 下 |

**`ra_c_1792x1024`** — v3′=`clean`（基準 `ra_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 101 | ○ | 13.7 | 下 |
| restore | 101 | ○ | 13.7 | 下 |
| stage2 | 417 | ○ | -21.5 | 下 |

**`ra_c_1856x1024`** — v3′=`plateau`（基準 `ra_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 103 | ○ | 0.0 | 下 |
| restore | 103 | ○ | 0.0 | 下 |
| stage2 | 575 | ○ | 421.1 | **超** |

**`ra_c_1792x1024_r2`** — v3′=`clean`（基準 `ra_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 102 | ○ | 0.0 | 下 |
| restore | 102 | ○ | 0.0 | 下 |
| stage2 | 418 | ○ | -5.7 | 下 |

**`ra_c_1856x1024_r2`** — v3′=`clean`（基準 `ra_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 104 | ○ | -15.7 | 下 |
| restore | 104 | ○ | -15.7 | 下 |
| stage2 | 507 | ○ | 230.5 | 下 |

**`rb_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 8 | × fewer than 10 WDDM samples | — | — |

**`rb_c_b1280x768`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 52 | × no baseline for this window | — | — |
| restore | 52 | × no baseline for this window | — | — |
| stage2 | 156 | × no baseline for this window | — | — |

**`rb_c_1792x1024`** — v3′=`clean`（基準 `rb_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 102 | ○ | 15.9 | 下 |
| restore | 102 | ○ | 15.9 | 下 |
| stage2 | 300 | ○ | -14.7 | 下 |

**`ua_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 7 | × fewer than 10 WDDM samples | — | — |

**`ua_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 13 | × no baseline for this window | — | — |
| restore | 13 | × no baseline for this window | — | — |
| stage2 | 32 | × no baseline for this window | — | — |

**`ua_s_1080_145`** — v3′=`clean`（基準 `ua_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -33.7 | 下 |
| restore | 22 | ○ | -33.7 | 下 |
| stage2 | 60 | ○ | -9.0 | 下 |

**`ua_s_1080_153`** — v3′=`plateau`（基準 `ua_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -9.0 | 下 |
| restore | 23 | ○ | -9.0 | 下 |
| stage2 | 99 | ○ | 634.9 | **超** |

**`ha_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 7 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 5 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 7 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 10 | × no baseline for this window | — | — |
| stage2 | 6 | × fewer than 10 WDDM samples | — | — |

**`ha_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 19 | × no baseline for this window | — | — |
| s23_gap_mid | 13 | × no baseline for this window | — | — |
| s23_gap_tail | 19 | × no baseline for this window | — | — |
| s23_stage1 | 22 | × no baseline for this window | — | — |
| stage2 | 25 | × no baseline for this window | — | — |

**`ha_s_1080_129`** — v3′=`plateau`（基準 `ha_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 29 | ○ | 0.0 | 下 |
| s23_gap_mid | 20 | ○ | 3.0 | 下 |
| s23_gap_tail | 29 | ○ | 0.0 | 下 |
| s23_stage1 | 31 | ○ | 3.0 | 下 |
| stage2 | 39 | ○ | 929.6 | **超** |

**`ha_s_1080_121`** — v3′=`clean`（基準 `ha_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 28 | ○ | 10.6 | 下 |
| s23_gap_mid | 17 | ○ | -0.1 | 下 |
| s23_gap_tail | 28 | ○ | 10.6 | 下 |
| s23_stage1 | 29 | ○ | -0.1 | 下 |
| stage2 | 34 | ○ | 127.0 | 下 |

**`ha_s_1080_121_r2`** — v3′=`clean`（基準 `ha_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 28 | ○ | 9.5 | 下 |
| s23_gap_mid | 18 | ○ | -4.0 | 下 |
| s23_gap_tail | 28 | ○ | 9.5 | 下 |
| s23_stage1 | 29 | ○ | 0.9 | 下 |
| stage2 | 34 | ○ | 124.0 | 下 |

**`ha_s_1080_129_r2`** — v3′=`plateau`（基準 `ha_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 29 | ○ | 1.5 | 下 |
| s23_gap_mid | 20 | ○ | 16.8 | 下 |
| s23_gap_tail | 29 | ○ | 1.5 | 下 |
| s23_stage1 | 31 | ○ | 16.8 | 下 |
| stage2 | 38 | ○ | 753.9 | **超** |

**`ha_s_720_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 9 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 6 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 9 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 14 | × no baseline for this window | — | — |
| stage2 | 11 | × no baseline for this window | — | — |

**`ha_s_720_265`** — v3′=`clean`（基準 `ha_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 28 | ○ | -25.6 | 下 |
| s23_gap_mid | 18 | ○ | 1.9 | 下 |
| s23_gap_tail | 28 | ○ | -25.6 | 下 |
| s23_stage1 | 29 | ○ | 1.9 | 下 |
| stage2 | 34 | ○ | -29.3 | 下 |

**`ha_s_720_265_r2`** — v3′=`clean`（基準 `ha_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 28 | ○ | -0.1 | 下 |
| s23_gap_mid | 17 | ○ | 16.4 | 下 |
| s23_gap_tail | 28 | ○ | -0.1 | 下 |
| s23_stage1 | 30 | ○ | 16.4 | 下 |
| stage2 | 34 | ○ | -27.4 | 下 |

**`ha_s_720_289`** — v3′=`plateau`（基準 `ha_s_720_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 28 | ○ | 0.9 | 下 |
| s23_gap_mid | 21 | ○ | 18.8 | 下 |
| s23_gap_tail | 28 | ○ | 0.9 | 下 |
| s23_stage1 | 31 | ○ | 18.8 | 下 |
| stage2 | 39 | ○ | 1063.6 | **超** |

**`ha_c_b1280x768`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 72 | × no baseline for this window | — | — |
| s23_gap_mid | 12 | × no baseline for this window | — | — |
| s23_gap_tail | 72 | × no baseline for this window | — | — |
| s23_stage1 | 83 | × no baseline for this window | — | — |
| stage2 | 138 | × no baseline for this window | — | — |

**`ha_c_1472x1024`** — v3′=`clean`（基準 `ha_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 117 | ○ | -13.6 | 下 |
| s23_gap_mid | 19 | ○ | -13.2 | 下 |
| s23_gap_tail | 117 | ○ | -13.6 | 下 |
| s23_stage1 | 126 | ○ | -13.2 | 下 |
| stage2 | 223 | ○ | 2.7 | 下 |

**`ha_c_1536x1024`** — v3′=`plateau`（基準 `ha_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 119 | ○ | -15.9 | 下 |
| s23_gap_mid | 20 | ○ | -45.0 | 下 |
| s23_gap_tail | 119 | ○ | -15.9 | 下 |
| s23_stage1 | 130 | ○ | -5.7 | 下 |
| stage2 | 239 | ○ | 410.3 | **超** |

**`ha_c_1472x1024_r2`** — v3′=`clean`（基準 `ha_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 117 | ○ | -21.6 | 下 |
| s23_gap_mid | 19 | ○ | -15.9 | 下 |
| s23_gap_tail | 117 | ○ | -21.6 | 下 |
| s23_stage1 | 126 | ○ | -15.9 | 下 |
| stage2 | 223 | ○ | -21.5 | 下 |

**`ha_c_1536x1024_r2`** — v3′=`plateau`（基準 `ha_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 120 | ○ | -5.5 | 下 |
| s23_gap_mid | 20 | ○ | -45.7 | 下 |
| s23_gap_tail | 120 | ○ | -5.5 | 下 |
| s23_stage1 | 130 | ○ | -5.7 | 下 |
| stage2 | 239 | ○ | 418.3 | **超** |

### 速度と共有メモリ

| ラベル | 生成秒 | stage2秒 | 秒/トークン | 復元秒 | 予約ピークMB | 共有ピーク(job)MB | 参照符号化 予約MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sa_warm` | 78.0 | 8.6 | 0.001392 | 6.7 | 9,008 | 689.7 | — |
| `sa_c_b1280x768` | 318.0 | 148.1 | 0.007010 | 74.3 | 10,090 | 734.4 | — |
| `sa_c_1920x1024` | 673.8 | 323.1 | 0.007649 | 169.2 | 15,474 | 843.9 | — |
| `sa_c_1984x1024` | 727.9 | 359.7 | 0.008241 | 171.3 | 15,910 | 1422.6 | — |
| `sa_c_1920x1024_r2` | 662.1 | 319.1 | 0.007553 | 167.6 | 15,474 | 774.6 | — |
| `sa_c_1984x1024_r2` | 723.4 | 356.9 | 0.008176 | 171.1 | 15,910 | 1256.5 | — |
| `qa_warm` | 80.3 | 8.4 | 0.001367 | 6.6 | 9,064 | 1175.8 | — |
| `qa_s_720_b89` | 49.0 | 15.6 | 0.001354 | 8.6 | 8,906 | 1176.3 | — |
| `qa_s_720_353` | 150.3 | 69.6 | 0.001611 | 36.1 | 14,446 | 1161.2 | — |
| `qa_s_720_353_r2` | 150.4 | 69.6 | 0.001611 | 36.3 | 14,446 | 1165.7 | — |
| `qa_s_720_361` | 155.4 | 74.4 | 0.001685 | 36.2 | 16,142 | 2121.2 | — |
| `qa_s_1080_b89` | 82.1 | 35.5 | 0.001452 | 19.3 | 11,428 | 1162.3 | — |
| `qa_s_1080_161` | 151.5 | 69.0 | 0.001611 | 38.4 | 14,420 | 1163.3 | — |
| `qa_s_1080_161_r2` | 150.8 | 69.0 | 0.001611 | 37.8 | 14,420 | 1178.2 | — |
| `qa_c_b1280x768` | 300.9 | 142.2 | 0.006733 | 71.2 | 10,286 | 1177.1 | — |
| `qa_c_1920x1024` | 668.5 | 321.4 | 0.007610 | 167.0 | 15,598 | 1521.1 | — |
| `qa_c_1856x1024` | 625.2 | 307.4 | 0.007527 | 148.0 | 15,144 | 1184.4 | — |
| `qa_c_1856x1024_r2` | 624.9 | 307.5 | 0.007531 | 147.7 | 15,144 | 1238.7 | — |
| `qa_c_1920x1024_r2` | 668.9 | 321.8 | 0.007617 | 166.7 | 15,598 | 1528.9 | — |
| `sb_warm` | 54.9 | 8.6 | 0.001392 | 3.0 | 7,180 | 1178.3 | — |
| `sb_c_b1280x768` | 294.3 | 146.1 | 0.006918 | 52.7 | 8,456 | 1161.7 | — |
| `sb_c_2048x1088` | 886.9 | 553.2 | 0.011556 | 120.4 | 15,912 | 1385.3 | — |
| `sb_c_1984x1088` | 699.0 | 377.6 | 0.008141 | 117.0 | 15,580 | 1166.9 | — |
| `sb_c_1984x1088_r2` | 692.3 | 368.6 | 0.007947 | 116.9 | 15,430 | 1159.3 | — |
| `sb_c_2048x1088_r2` | 759.7 | 426.1 | 0.008902 | 120.3 | 15,638 | 1142.5 | — |
| `sd_warm` | 38.1 | 9.2 | 0.001489 | 2.9 | 6,836 | 1138.6 | — |
| `sd_c_b1280x768` | 351.9 | 182.6 | 0.008643 | 52.6 | 8,456 | 1166.3 | — |
| `sd_c_1984x1088` | 938.8 | 551.9 | 0.011899 | 116.9 | 15,568 | 1142.5 | — |
| `qb_warm` | 56.6 | 8.7 | 0.001416 | 3.0 | 7,180 | 1652.6 | — |
| `qb_s_896_b89` | 44.2 | 17.2 | 0.001426 | 6.2 | 6,918 | 1689.3 | — |
| `qb_s_896_337` | 146.0 | 70.5 | 0.001627 | 24.7 | 15,466 | 1640.6 | — |
| `qb_s_896_337_r2` | 145.9 | 72.0 | 0.001661 | 24.6 | 15,466 | 1635.2 | — |
| `qb_s_720_b89` | 41.9 | 16.1 | 0.001393 | 6.0 | 6,796 | 1696.4 | — |
| `qb_s_720_353` | 145.0 | 72.3 | 0.001674 | 24.3 | 15,416 | 1604.4 | — |
| `qb_s_720_353_r2` | 144.9 | 72.0 | 0.001667 | 24.3 | 15,416 | 1659.7 | — |
| `qb_s_720_361` | 178.5 | 102.3 | 0.002317 | 25.9 | 15,678 | 1647.9 | — |
| `qb_s_720_361_r2` | 175.1 | 98.7 | 0.002235 | 25.9 | 15,678 | 1663.9 | — |
| `qb_s_1080_b89` | 80.1 | 36.6 | 0.001495 | 13.1 | 10,488 | 1623.8 | — |
| `qb_s_1080_161` | 143.0 | 68.7 | 0.001604 | 23.9 | 15,336 | 1663.7 | — |
| `qb_s_1080_161_r2` | 142.9 | 70.8 | 0.001653 | 23.9 | 15,336 | 1663.7 | — |
| `qb_c_b1280x768` | 294.8 | 146.4 | 0.006932 | 52.6 | 9,602 | 1663.7 | — |
| `qb_c_1920x1024` | 621.0 | 327.3 | 0.007749 | 107.5 | 15,150 | 1655.8 | — |
| `qb_c_1984x1024` | 658.7 | 354.0 | 0.008110 | 111.0 | 15,528 | 1655.8 | — |
| `qb_c_2048x1024` | 971.6 | 658.6 | 0.014618 | 114.4 | 15,894 | 1680.9 | — |
| `qb_c_1984x1024_r2` | 672.5 | 365.9 | 0.008382 | 111.0 | 15,560 | 1647.9 | — |
| `qb_c_2048x1024_r2` | 945.7 | 632.4 | 0.014036 | 114.4 | 15,904 | 1800.9 | — |
| `ra_warm` | 65.2 | 13.2 | 0.002148 | 3.0 | 7,180 | 1656.7 | — |
| `ra_s_1080_b89` | 120.5 | 50.9 | 0.002077 | 13.1 | 11,554 | 1652.7 | — |
| `ra_s_1080_145` | 186.2 | 92.8 | 0.002396 | 22.1 | 15,084 | 1672.2 | — |
| `ra_s_1080_153` | 226.5 | 131.2 | 0.003217 | 23.0 | 15,644 | 1605.0 | — |
| `ra_s_1080_145_r2` | 183.0 | 92.8 | 0.002396 | 22.1 | 15,084 | 1380.7 | — |
| `ra_s_1080_153_r2` | 224.3 | 131.1 | 0.003213 | 23.0 | 15,644 | 1639.1 | — |
| `ra_s_720_b89` | 68.8 | 22.2 | 0.001927 | 6.0 | 8,076 | 1672.4 | — |
| `ra_s_720_313` | 184.7 | 92.2 | 0.002402 | 22.3 | 14,964 | 1656.3 | — |
| `ra_s_720_313_r2` | 182.4 | 92.2 | 0.002402 | 22.3 | 15,006 | 1646.9 | — |
| `ra_s_720_337` | 236.7 | 140.2 | 0.003398 | 23.5 | 15,812 | 1672.2 | — |
| `ra_c_b1280x768` | 393.3 | 201.9 | 0.009560 | 52.6 | 10,690 | 1663.7 | — |
| `ra_c_1728x1024` | 758.1 | 425.6 | 0.011194 | 97.9 | 14,980 | 1655.8 | — |
| `ra_c_1792x1024` | 794.7 | 447.4 | 0.011350 | 101.6 | 15,268 | 1673.0 | — |
| `ra_c_1856x1024` | 973.2 | 615.5 | 0.015073 | 104.1 | 15,742 | 1642.4 | — |
| `ra_c_1792x1024_r2` | 793.0 | 447.1 | 0.011342 | 101.5 | 15,330 | 1663.7 | — |
| `ra_c_1856x1024_r2` | 904.2 | 549.9 | 0.013467 | 104.0 | 15,666 | 1639.5 | — |
| `rb_warm` | 49.6 | 12.6 | 0.002051 | 3.0 | 6,808 | 1252.3 | — |
| `rb_c_b1280x768` | 332.8 | 165.8 | 0.007848 | 52.6 | 10,690 | 1663.7 | — |
| `rb_c_1792x1024` | 615.0 | 320.9 | 0.008138 | 101.7 | 15,262 | 1663.8 | — |
| `ua_warm` | 54.3 | 10.7 | 0.001733 | 3.0 | 7,180 | 1631.9 | — |
| `ua_s_1080_b89` | 110.4 | 48.5 | 0.001979 | 13.0 | 11,904 | 1643.6 | — |
| `ua_s_1080_145` | 180.9 | 90.5 | 0.002334 | 22.1 | 15,298 | 1245.4 | — |
| `ua_s_1080_153` | 234.3 | 147.9 | 0.003625 | 23.0 | 15,850 | 1794.6 | — |
| `ha_warm` | 64.8 | 9.8 | 0.001587 | 6.7 | 9,498 | 2200.4 | — |
| `ha_s_1080_b89` | 88.4 | 36.9 | 0.001507 | 19.6 | 12,412 | 2184.1 | — |
| `ha_s_1080_129` | 128.4 | 58.2 | 0.001678 | 29.8 | 16,250 | 3141.4 | — |
| `ha_s_1080_121` | 117.4 | 51.1 | 0.001567 | 28.3 | 15,486 | 2312.0 | — |
| `ha_s_1080_121_r2` | 116.6 | 50.9 | 0.001558 | 27.6 | 15,486 | 2310.8 | — |
| `ha_s_1080_129_r2` | 127.3 | 57.6 | 0.001661 | 29.1 | 16,250 | 2964.9 | — |
| `ha_s_720_b89` | 45.7 | 16.8 | 0.001458 | 8.6 | 9,838 | 2199.3 | — |
| `ha_s_720_265` | 116.6 | 51.0 | 0.001563 | 27.7 | 15,484 | 2190.7 | — |
| `ha_s_720_265_r2` | 116.7 | 51.0 | 0.001563 | 27.6 | 15,484 | 2199.4 | — |
| `ha_s_720_289` | 128.0 | 58.5 | 0.001647 | 28.4 | 16,554 | 3289.8 | — |
| `ha_c_b1280x768` | 315.0 | 149.1 | 0.007060 | 71.8 | 12,540 | 2201.5 | — |
| `ha_c_1472x1024` | 498.1 | 240.0 | 0.007411 | 117.7 | 15,422 | 2191.3 | — |
| `ha_c_1536x1024` | 522.1 | 257.2 | 0.007613 | 119.5 | 15,938 | 2591.6 | — |
| `ha_c_1472x1024_r2` | 497.1 | 240.2 | 0.007416 | 116.6 | 15,422 | 2184.9 | — |
| `ha_c_1536x1024_r2` | 523.4 | 257.2 | 0.007613 | 120.5 | 15,938 | 2599.4 | — |

### 所要時間（`manifest.jsonl` の実測）

| ラベル | 開始 | 終了 | 秒 |
|---|---|---|---:|
| `sa_warm` | 2026-10-05 08:19:30.877 | 2026-10-05 08:20:51.081 | 80.2 |
| `sa_c_b1280x768` | 2026-10-05 08:23:31.890 | 2026-10-05 08:28:50.442 | 318.6 |
| `sa_c_1920x1024` | 2026-10-05 08:31:33.868 | 2026-10-05 08:42:49.348 | 675.5 |
| `sa_c_1984x1024` | 2026-10-05 08:45:35.561 | 2026-10-05 08:57:43.801 | 728.2 |
| `sa_c_1920x1024_r2` | 2026-10-05 09:00:30.240 | 2026-10-05 09:11:34.245 | 664.0 |
| `sa_c_1984x1024_r2` | 2026-10-05 09:14:19.916 | 2026-10-05 09:26:24.072 | 724.2 |
| `qa_warm` | 2026-10-05 09:30:13.406 | 2026-10-05 09:31:35.623 | 82.2 |
| `qa_s_720_b89` | 2026-10-05 09:34:20.497 | 2026-10-05 09:35:10.728 | 50.2 |
| `qa_s_720_353` | 2026-10-05 09:38:45.349 | 2026-10-05 09:41:15.865 | 150.5 |
| `qa_s_720_353_r2` | 2026-10-05 09:43:54.615 | 2026-10-05 09:46:25.104 | 150.5 |
| `qa_s_720_361` | 2026-10-05 09:49:03.124 | 2026-10-05 09:51:39.592 | 156.5 |
| `qa_s_1080_b89` | 2026-10-05 09:54:13.737 | 2026-10-05 09:55:35.989 | 82.2 |
| `qa_s_1080_161` | 2026-10-05 09:58:13.574 | 2026-10-05 10:00:46.009 | 152.4 |
| `qa_s_1080_161_r2` | 2026-10-05 10:03:21.740 | 2026-10-05 10:05:54.179 | 152.4 |
| `qa_c_b1280x768` | 2026-10-05 10:08:45.574 | 2026-10-05 10:13:48.493 | 302.9 |
| `qa_c_1920x1024` | 2026-10-05 10:16:28.736 | 2026-10-05 10:27:38.909 | 670.2 |
| `qa_c_1856x1024` | 2026-10-05 10:30:20.112 | 2026-10-05 10:40:46.073 | 626.0 |
| `qa_c_1856x1024_r2` | 2026-10-05 10:43:32.213 | 2026-10-05 10:53:58.130 | 625.9 |
| `qa_c_1920x1024_r2` | 2026-10-05 10:56:45.340 | 2026-10-05 11:07:55.338 | 670.0 |
| `sb_warm` | 2026-10-05 11:13:42.066 | 2026-10-05 11:14:38.228 | 56.2 |
| `sb_c_b1280x768` | 2026-10-05 11:17:18.235 | 2026-10-05 11:22:13.192 | 295.0 |
| `sb_c_2048x1088` | 2026-10-05 11:24:54.444 | 2026-10-05 11:39:42.664 | 888.2 |
| `sb_c_1984x1088` | 2026-10-05 11:42:23.413 | 2026-10-05 11:54:03.539 | 700.1 |
| `sb_c_1984x1088_r2` | 2026-10-05 11:56:45.780 | 2026-10-05 12:08:19.771 | 694.0 |
| `sb_c_2048x1088_r2` | 2026-10-05 12:11:02.658 | 2026-10-05 12:23:42.769 | 760.1 |
| `sd_warm` | 2026-10-05 12:26:19.902 | 2026-10-05 12:27:00.064 | 40.2 |
| `sd_c_b1280x768` | 2026-10-05 12:29:41.332 | 2026-10-05 12:35:34.620 | 353.3 |
| `sd_c_1984x1088` | 2026-10-05 12:38:16.359 | 2026-10-05 12:53:56.946 | 940.6 |
| `qb_warm` | 2026-10-05 12:57:42.174 | 2026-10-05 12:58:40.384 | 58.2 |
| `qb_s_896_b89` | 2026-10-05 13:01:14.794 | 2026-10-05 13:02:00.952 | 46.2 |
| `qb_s_896_337` | 2026-10-05 13:04:36.209 | 2026-10-05 13:07:02.725 | 146.5 |
| `qb_s_896_337_r2` | 2026-10-05 13:09:38.421 | 2026-10-05 13:12:04.885 | 146.5 |
| `qb_s_720_b89` | 2026-10-05 13:14:42.633 | 2026-10-05 13:15:24.788 | 42.2 |
| `qb_s_720_353` | 2026-10-05 13:18:00.473 | 2026-10-05 13:20:26.931 | 146.5 |
| `qb_s_720_353_r2` | 2026-10-05 13:23:02.048 | 2026-10-05 13:25:28.548 | 146.5 |
| `qb_s_720_361` | 2026-10-05 13:28:04.817 | 2026-10-05 13:31:05.323 | 180.5 |
| `qb_s_720_361_r2` | 2026-10-05 13:33:41.791 | 2026-10-05 13:36:38.356 | 176.6 |
| `qb_s_1080_b89` | 2026-10-05 13:39:13.882 | 2026-10-05 13:40:34.188 | 80.3 |
| `qb_s_1080_161` | 2026-10-05 13:43:09.530 | 2026-10-05 13:45:33.961 | 144.4 |
| `qb_s_1080_161_r2` | 2026-10-05 13:48:08.912 | 2026-10-05 13:50:33.403 | 144.5 |
| `qb_c_b1280x768` | 2026-10-05 13:53:14.149 | 2026-10-05 13:58:10.974 | 296.8 |
| `qb_c_1920x1024` | 2026-10-05 14:00:53.582 | 2026-10-05 14:11:15.531 | 622.0 |
| `qb_c_1984x1024` | 2026-10-05 14:13:58.403 | 2026-10-05 14:24:58.514 | 660.1 |
| `qb_c_2048x1024` | 2026-10-05 14:27:40.615 | 2026-10-05 14:43:52.858 | 972.2 |
| `qb_c_1984x1024_r2` | 2026-10-05 14:46:33.456 | 2026-10-05 14:57:47.457 | 674.0 |
| `qb_c_2048x1024_r2` | 2026-10-05 15:00:29.310 | 2026-10-05 15:16:16.062 | 946.8 |
| `ra_warm` | 2026-10-05 15:22:01.320 | 2026-10-05 15:23:07.570 | 66.2 |
| `ra_s_1080_b89` | 2026-10-05 15:25:44.318 | 2026-10-05 15:27:46.775 | 122.5 |
| `ra_s_1080_145` | 2026-10-05 15:30:23.144 | 2026-10-05 15:33:29.808 | 186.7 |
| `ra_s_1080_153` | 2026-10-05 15:36:07.049 | 2026-10-05 15:39:53.814 | 226.8 |
| `ra_s_1080_145_r2` | 2026-10-05 15:42:30.595 | 2026-10-05 15:45:35.113 | 184.5 |
| `ra_s_1080_153_r2` | 2026-10-05 15:48:11.312 | 2026-10-05 15:51:56.053 | 224.7 |
| `ra_s_720_b89` | 2026-10-05 15:54:32.178 | 2026-10-05 15:55:42.469 | 70.3 |
| `ra_s_720_313` | 2026-10-05 15:58:18.387 | 2026-10-05 16:01:25.039 | 186.7 |
| `ra_s_720_313_r2` | 2026-10-05 16:04:02.689 | 2026-10-05 16:07:05.267 | 182.6 |
| `ra_s_720_337` | 2026-10-05 16:09:41.524 | 2026-10-05 16:13:38.257 | 236.7 |
| `ra_c_b1280x768` | 2026-10-05 16:16:18.575 | 2026-10-05 16:22:53.654 | 395.1 |
| `ra_c_1728x1024` | 2026-10-05 16:25:37.478 | 2026-10-05 16:38:15.742 | 758.3 |
| `ra_c_1792x1024` | 2026-10-05 16:40:57.676 | 2026-10-05 16:54:14.097 | 796.4 |
| `ra_c_1856x1024` | 2026-10-05 16:56:55.380 | 2026-10-05 17:13:10.368 | 975.0 |
| `ra_c_1792x1024_r2` | 2026-10-05 17:15:50.022 | 2026-10-05 17:29:04.264 | 794.2 |
| `ra_c_1856x1024_r2` | 2026-10-05 17:31:44.818 | 2026-10-05 17:46:49.229 | 904.4 |
| `rb_warm` | 2026-10-05 17:49:23.897 | 2026-10-05 17:50:14.048 | 50.1 |
| `rb_c_b1280x768` | 2026-10-05 17:52:55.726 | 2026-10-05 17:58:30.598 | 334.9 |
| `rb_c_1792x1024` | 2026-10-05 18:01:09.999 | 2026-10-05 18:11:25.486 | 615.5 |
| `ua_warm` | 2026-10-05 18:14:59.666 | 2026-10-05 18:15:55.860 | 56.2 |
| `ua_s_1080_b89` | 2026-10-05 18:18:32.278 | 2026-10-05 18:20:24.721 | 112.4 |
| `ua_s_1080_145` | 2026-10-05 18:23:02.152 | 2026-10-05 18:26:04.871 | 182.7 |
| `ua_s_1080_153` | 2026-10-05 18:28:42.732 | 2026-10-05 18:32:37.413 | 234.7 |
| `ha_warm` | 2026-10-05 18:38:19.022 | 2026-10-05 18:39:25.255 | 66.2 |
| `ha_s_1080_b89` | 2026-10-05 18:42:01.587 | 2026-10-05 18:43:31.849 | 90.3 |
| `ha_s_1080_129` | 2026-10-05 18:46:07.717 | 2026-10-05 18:48:18.041 | 130.3 |
| `ha_s_1080_121` | 2026-10-05 18:50:53.393 | 2026-10-05 18:52:51.686 | 118.3 |
| `ha_s_1080_121_r2` | 2026-10-05 18:55:26.802 | 2026-10-05 18:57:25.145 | 118.3 |
| `ha_s_1080_129_r2` | 2026-10-05 18:59:59.890 | 2026-10-05 19:02:08.245 | 128.4 |
| `ha_s_720_b89` | 2026-10-05 19:04:42.661 | 2026-10-05 19:05:28.872 | 46.2 |
| `ha_s_720_265` | 2026-10-05 19:08:03.986 | 2026-10-05 19:10:02.352 | 118.4 |
| `ha_s_720_265_r2` | 2026-10-05 19:12:36.712 | 2026-10-05 19:14:35.036 | 118.3 |
| `ha_s_720_289` | 2026-10-05 19:17:12.487 | 2026-10-05 19:19:20.909 | 128.4 |
| `ha_c_b1280x768` | 2026-10-05 19:22:07.056 | 2026-10-05 19:27:23.915 | 316.9 |
| `ha_c_1472x1024` | 2026-10-05 19:30:05.623 | 2026-10-05 19:38:24.975 | 499.4 |
| `ha_c_1536x1024` | 2026-10-05 19:41:02.684 | 2026-10-05 19:49:46.371 | 523.7 |
| `ha_c_1472x1024_r2` | 2026-10-05 19:52:21.746 | 2026-10-05 20:00:39.110 | 497.4 |
| `ha_c_1536x1024_r2` | 2026-10-05 20:03:14.986 | 2026-10-05 20:11:58.586 | 523.6 |

- 計測点の実行時間の合計: **479.0 分**
- 最初の点の開始から最後の点の終了まで（冷却込み）: **712.5 分**
- 実行した点の数（再走を含む延べ）: **84**


## 付録 B: commit_table.py の出力

| plan | label | v3' | stage2 Δ MB | restore Δ MB | used | commit peak GiB | % of limit |
|---|---|---|---|---|---|---|---|
| p_ha_warm | ha_warm | baseline |  |  | ok | 84.47 | 59.0 |
| p_ha_s_1080_b89 | ha_s_1080_b89 | baseline |  |  | ok | 97.40 | 68.0 |
| p_ha_s_1080_129 | ha_s_1080_129 | plateau | +929.6 | +0.0 | ok | 101.16 | 70.6 |
| p_ha_s_1080_121 | ha_s_1080_121 | clean | +127.0 | +10.6 | ok | 100.41 | 70.1 |
| p_ha_s_1080_121_r2 | ha_s_1080_121_r2 | clean | +124.0 | +9.5 | ok | 100.39 | 70.1 |
| p_ha_s_1080_129_r2 | ha_s_1080_129_r2 | plateau | +753.9 | +1.5 | ok | 101.25 | 70.7 |
| p_ha_s_720_b89 | ha_s_720_b89 | baseline |  |  | ok | 94.88 | 66.2 |
| p_ha_s_720_265 | ha_s_720_265 | clean | -29.3 | -25.6 | ok | 100.36 | 70.1 |
| p_ha_s_720_265_r2 | ha_s_720_265_r2 | clean | -27.4 | -0.1 | ok | 100.44 | 70.1 |
| p_ha_s_720_289 | ha_s_720_289 | plateau | +1063.6 | +0.9 | ok | 101.66 | 71.0 |
| p_ha_c_b1280x768 | ha_c_b1280x768 | baseline |  |  | ok | 97.76 | 68.2 |
| p_ha_c_1472x1024 | ha_c_1472x1024 | clean | +2.7 | -13.6 | ok | 100.69 | 70.3 |
| p_ha_c_1536x1024 | ha_c_1536x1024 | plateau | +410.3 | -15.9 | ok | 101.21 | 70.6 |
| p_ha_c_1472x1024_r2 | ha_c_1472x1024_r2 | clean | -21.5 | -21.6 | ok | 100.72 | 70.3 |
| p_ha_c_1536x1024_r2 | ha_c_1536x1024_r2 | plateau | +418.3 | -5.5 | ok | 101.24 | 70.7 |
| p_qa_warm | qa_warm | baseline |  |  | ok | 76.36 | 53.3 |
| p_qa_s_720_b89 | qa_s_720_b89 | baseline |  |  | ok | 75.92 | 53.0 |
| p_qa_s_720_353 | qa_s_720_353 | clean | +10.6 | +2.5 | ok | 82.83 | 57.8 |
| p_qa_s_720_353_r2 | qa_s_720_353_r2 | clean | +3.3 | +9.0 | ok | 83.07 | 58.0 |
| p_qa_s_720_361 | qa_s_720_361 | plateau | +947.2 | -21.4 | ok | 85.24 | 59.5 |
| p_qa_s_1080_b89 | qa_s_1080_b89 | baseline |  |  | ok | 80.63 | 56.3 |
| p_qa_s_1080_161 | qa_s_1080_161 | clean | -0.3 | -8.5 | ok | 83.62 | 58.4 |
| p_qa_s_1080_161_r2 | qa_s_1080_161_r2 | clean | -25.5 | -25.4 | ok | 83.54 | 58.3 |
| p_qa_c_b1280x768 | qa_c_b1280x768 | baseline |  |  | ok | 79.48 | 55.5 |
| p_qa_c_1920x1024 | qa_c_1920x1024 | plateau | +351.3 | -15.9 | ok | 84.49 | 59.0 |
| p_qa_c_1856x1024 | qa_c_1856x1024 | clean | -0.4 | -10.5 | ok | 84.21 | 58.8 |
| p_qa_c_1856x1024_r2 | qa_c_1856x1024_r2 | clean | +63.2 | -7.1 | ok | 84.21 | 58.8 |
| p_qa_c_1920x1024_r2 | qa_c_1920x1024_r2 | plateau | +351.3 | -24.1 | ok | 84.68 | 59.1 |
| p_qb_warm | qb_warm | baseline |  |  | ok | 64.72 | 45.2 |
| p_qb_s_896_b89 | qb_s_896_b89 | baseline |  |  | ok | 66.96 | 46.7 |
| p_qb_s_896_337 | qb_s_896_337 | clean | -9.9 | -42.8 | ok | 75.36 | 52.6 |
| p_qb_s_896_337_r2 | qb_s_896_337_r2 | clean | -25.9 | -49.9 | ok | 75.40 | 52.6 |
| p_qb_s_720_b89 | qb_s_720_b89 | baseline |  |  | ok | 67.02 | 46.8 |
| p_qb_s_720_353 | qb_s_720_353 | clean | -32.7 | -32.7 | ok | 75.42 | 52.6 |
| p_qb_s_720_353_r2 | qb_s_720_353_r2 | clean | -32.7 | -32.7 | ok | 75.43 | 52.6 |
| p_qb_s_720_361 | qb_s_720_361 | clean | +243.5 | -48.5 | ok | 75.75 | 52.9 |
| p_qb_s_720_361_r2 | qb_s_720_361_r2 | clean | +287.3 | -32.5 | ok | 75.77 | 52.9 |
| p_qb_s_1080_b89 | qb_s_1080_b89 | baseline |  |  | ok | 70.64 | 49.3 |
| p_qb_s_1080_161 | qb_s_1080_161 | clean | -41.0 | -9.3 | ok | 75.42 | 52.6 |
| p_qb_s_1080_161_r2 | qb_s_1080_161_r2 | clean | -41.0 | -18.6 | ok | 75.38 | 52.6 |
| p_qb_c_b1280x768 | qb_c_b1280x768 | baseline |  |  | ok | 69.83 | 48.7 |
| p_qb_c_1920x1024 | qb_c_1920x1024 | clean | -21.7 | +0.0 | ok | 75.26 | 52.5 |
| p_qb_c_1984x1024 | qb_c_1984x1024 | clean | +42.3 | +8.5 | ok | 75.68 | 52.8 |
| p_qb_c_2048x1024 | qb_c_2048x1024 | plateau | +514.1 | +0.0 | ok | 76.55 | 53.4 |
| p_qb_c_1984x1024_r2 | qb_c_1984x1024_r2 | clean | +122.1 | +9.6 | ok | 75.75 | 52.9 |
| p_qb_c_2048x1024_r2 | qb_c_2048x1024_r2 | plateau | +608.4 | -7.7 | ok | 76.10 | 53.1 |
| p_ra_warm | ra_warm | baseline |  |  | ok | 51.79 | 36.2 |
| p_ra_s_1080_b89 | ra_s_1080_b89 | baseline |  |  | ok | 67.41 | 47.1 |
| p_ra_s_1080_145 | ra_s_1080_145 | clean | -16.5 | -16.5 | ok | 70.78 | 49.4 |
| p_ra_s_1080_153 | ra_s_1080_153 | plateau | +338.2 | -42.7 | ok | 71.40 | 49.8 |
| p_ra_s_1080_145_r2 | ra_s_1080_145_r2 | clean | -16.2 | -16.5 | ok | 71.20 | 49.7 |
| p_ra_s_1080_153_r2 | ra_s_1080_153_r2 | plateau | +346.5 | -45.7 | ok | 71.54 | 49.9 |
| p_ra_s_720_b89 | ra_s_720_b89 | baseline |  |  | ok | 64.09 | 44.7 |
| p_ra_s_720_313 | ra_s_720_313 | clean | -41.1 | -8.7 | ok | 70.92 | 49.5 |
| p_ra_s_720_313_r2 | ra_s_720_313_r2 | clean | -28.7 | -17.2 | ok | 70.97 | 49.5 |
| p_ra_s_720_337 | ra_s_720_337 | plateau | +367.2 | -0.2 | ok | 71.80 | 50.1 |
| p_ra_c_b1280x768 | ra_c_b1280x768 | baseline |  |  | ok | 66.84 | 46.7 |
| p_ra_c_1728x1024 | ra_c_1728x1024 | clean | -5.7 | +15.9 | ok | 71.25 | 49.7 |
| p_ra_c_1792x1024 | ra_c_1792x1024 | clean | -21.5 | +13.7 | ok | 71.46 | 49.9 |
| p_ra_c_1856x1024 | ra_c_1856x1024 | plateau | +421.1 | +0.0 | ok | 72.01 | 50.3 |
| p_ra_c_1792x1024_r2 | ra_c_1792x1024_r2 | clean | -5.7 | +0.0 | ok | 71.67 | 50.0 |
| p_ra_c_1856x1024_r2 | ra_c_1856x1024_r2 | clean | +230.5 | -15.7 | ok | 72.46 | 50.6 |
| p_rb_warm | rb_warm | baseline |  |  | ok | 71.92 | 50.2 |
| p_rb_c_b1280x768 | rb_c_b1280x768 | baseline |  |  | ok | 81.94 | 57.2 |
| p_rb_c_1792x1024 | rb_c_1792x1024 | clean | -14.7 | +15.9 | ok | 86.09 | 60.1 |
| p_sa_warm | sa_warm | baseline |  |  | ok | 75.04 | 52.4 |
| p_sa_c_b1280x768 | sa_c_b1280x768 | baseline |  |  | ok | 78.00 | 54.4 |
| p_sa_c_1920x1024 | sa_c_1920x1024 | clean | -43.7 | -72.3 | ok | 83.29 | 58.1 |
| p_sa_c_1984x1024 | sa_c_1984x1024 | plateau | +711.7 | -66.6 | ok | 81.10 | 56.6 |
| p_sa_c_1920x1024_r2 | sa_c_1920x1024_r2 | clean | -64.6 | -71.3 | ok | 80.24 | 56.0 |
| p_sa_c_1984x1024_r2 | sa_c_1984x1024_r2 | plateau | +553.4 | -82.4 | ok | 80.87 | 56.4 |
| p_sb_warm | sb_warm | baseline |  |  | ok | 59.30 | 41.4 |
| p_sb_c_b1280x768 | sb_c_b1280x768 | baseline |  |  | ok | 63.37 | 44.2 |
| p_sb_c_2048x1088 | sb_c_2048x1088 | plateau | +412.0 | -3.0 | ok | 71.15 | 49.7 |
| p_sb_c_1984x1088 | sb_c_1984x1088 | clean | +37.0 | +5.2 | ok | 70.40 | 49.1 |
| p_sb_c_1984x1088_r2 | sb_c_1984x1088_r2 | clean | -4.8 | -3.0 | ok | 70.28 | 49.1 |
| p_sb_c_2048x1088_r2 | sb_c_2048x1088_r2 | clean | +97.5 | -11.3 | ok | 70.65 | 49.3 |
| p_sd_warm | sd_warm | baseline |  |  | ok | 55.28 | 38.6 |
| p_sd_c_b1280x768 | sd_c_b1280x768 | baseline |  |  | ok | 51.22 | 35.8 |
| p_sd_c_1984x1088 | sd_c_1984x1088 | clean | +10.2 | -6.9 | ok | 58.22 | 40.6 |
| p_ua_warm | ua_warm | baseline |  |  | ok | 59.22 | 41.3 |
| p_ua_s_1080_b89 | ua_s_1080_b89 | baseline |  |  | ok | 74.98 | 52.3 |
| p_ua_s_1080_145 | ua_s_1080_145 | clean | -9.0 | -33.7 | ok | 78.13 | 54.5 |
| p_ua_s_1080_153 | ua_s_1080_153 | plateau | +634.9 | -9.0 | ok | 78.51 | 54.8 |
