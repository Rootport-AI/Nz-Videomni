> **このファイルは実機の `outputs/comfort-calib-2026-09-27/RESULTS.md`（git追跡外）のスナップショットである（2026-09-28複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# int8／w4a8 safetensors の transformer での快適上限の較正 — 一次記録（2026-09-27〜28・C-4）

> **第1節から第4節は、結論・条件・実際に走らせた順番・判定の読み方です。** 手順と判定規則の正本は `Docs/VERIFICATION_LOG.md` §121.11（手順 v2）、今夜の運用計画は承認済みの実施計画（`floofy-enchanting-sundae.md`）、点の設計は同じフォルダの `README.md` 第3節です。
> **第5節は `make_results.py` の出力の逐語です。** 人が書き写した数字は 1 つもありません。第6節の表は、`runs/<ラベル>/run.json`・`commit_table.md`（`commit_log.csv` から機械で集計したもの）から、判定と比較に使う値だけを抜き出したものです。
> 台帳は `Docs/PENDING_TASKS.md` §1-32（この較正）・§1-31（配信値などへの反映の受け皿）・§1-33（ConvRot の高速化の Go／No-go）です。

---

## 1. 結論

### 1.1 何を測ったのか

**transformer（変換器）を ComfyUI 標準の int8／w4a8 safetensors（重みを 8 ビット整数、または 4 ビット重み＋8 ビット活性で持つ形式）にしたとき、快適上限（VRAM 溢れが起きない生成規模の目安。1 タイルあたりのトークン数）がどこにあるかを、同じサーバープロセスの fp8 と並べて実測しました。** 直接読みの実装は §3-168（`Docs/VERIFICATION_LOG.md` §121）です。

| 頭 | ベースモデル | 変換器の登録名 | 重みの種別 | 構成 | 役割 |
|---|---|---|---|---|---|
| `u_` | LTX 2.5 | `ltx25_uncensored_v1.1-fp8_scaled` | fp8 | 全 on | LTX 2.5 の対照（同じプロセスの fp8） |
| `r_` | LTX 2.5 | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | int8＋w4a8 の混在（REDGraft） | 全 on | 主役（Q1・Q3） |
| `h_` | LTX 2.5 | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot` | int8 ConvRot（アダマール回転つき） | 全 on | 主役（Q2） |
| `f_` | LTX 2.3 | `sulphur_distil_fp8mixed` | fp8 | 既定構成 | LTX 2.3 の対照（同じプロセスの fp8） |
| `x_` | LTX 2.3 | `ltx-2.3-22b-distilled-1.1_int8mixedtensorwise` | int8（silveroxides・ConvRot なし） | 既定構成 | 主役（Q4） |
| `j_` | LTX 2.3 | `ltx-2.3-22b-distilled-1.1_w4a8` | w4a8 | 全 on（入口判定で決定） | 主役（Q5） |
| `k_` | LTX 2.3 | `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot` | int8 ConvRot（Kijai・transformer のみ） | 既定構成 | 任意（Q4'） |

「全 on」は線（快適上限の目安）を定義している構成（sage attention・keep_resident・VAE の conv 復元などをすべて有効）、「既定構成」はサーバーの既定（sdpa・keep_resident なし・VAE の既定）です。**LTX 2.3 の既定構成には製品の線が無いので、`f_`・`x_`・`k_` の値は fp8 と比べるための診断値です。**

**スコープは「測って結論を出す」までです。** 配信値（`config.py` の `_default_comfort_budgets()`）・UI・マニフェストには触れていません。Nz-Videomni のコードも `Docs/` の本文も変えていません（`Docs/Outputs-archive/` への複写だけです）。**反映は別途検討（台帳 §1-31）で、判断はオーナーです。**

### 1.2 問いへの答え

刻みは 1920×1088 の潜在 1 コマ（2,040 トークン）、比べるのは「2 回とも快適だった上端 c」だけです（第4節）。線比は現行の線 44,880 との比です。

| 問い | 答え | 根拠（1920×1088・第2段階の窓の差は基準点比） |
|---|---|---|
| **Q1** REDGraft 混在（2.5）は fp8 より上か。Q6_K GGUF（42,840 快適・44,880 溢れ）まで届くか | **fp8 と同じ段。Q6_K GGUF には届かない** | `r_` の c＝145 コマ（38,760・線比 86%）で、同じプロセスの fp8（`u_`）の c と同じ段。153 コマ（40,800）は 2 回とも溢れ（+558.2／+618.5）、161 コマ（42,840）も溢れ（+1,191.0・1 回）。REDGraft の Q6_K GGUF（§106）が快適だった 42,840 は、この safetensors では溢れ |
| **Q2** 2.5 int8 ConvRot（公式複製）は fp8 と同じか | **同じ** | `h_` の c＝145 コマ（38,760）で `u_` と同じ段。145 コマは 2 回とも快適（+153.6／+140.3）、153 コマは 2 回とも溢れ（+814.2／+695.0） |
| **Q3** 2.5 の 2 種は単一値で表せるか | **REDGraft は表せる。int8 ConvRot は 720p を測っていない** | `r_` の 1280×768 は、1080p の c（38,760）以下で最大の 313 コマ（38,400）が 2 回とも快適（−34.0／−25.5）、c の 1 段上（40,800）以上で最小の 337 コマ（41,280）が溢れ（+678.5・1 回）。規則の 2 条件がどちらも成り立つ。`h_` は 1080p の c が `u_` と同じだったので、規則により 720p は走らせていない |
| **Q4** 2.3 int8（silveroxides）は既定構成で fp8 と同じか | **同じ** | `x_` の c＝121 コマ（32,640・線比 73%）で、同じプロセスの fp8（`f_`）の c と同じ段。129 コマ（34,680）は両方とも 2 回溢れ（`x_` +728.8／+748.5、`f_` +726.3／+747.4） |
| **Q5** 2.3 w4a8 は fp8 より上か。全 on で測れるか | **全 on で測れた。線 44,880 まで成り立つ** | 入口判定（全 on の捨てランと 89 コマの基準点のコミット最大がどちらも 88% 未満）はそれぞれ 59.8%・70.0% で通過。145 コマ（+19.0）・161 コマ（−3.9）・169 コマ＝線上の 44,880（+27.1／+12.8、2 回）がすべて快適。**ただし同じ構成（全 on）の fp8 の境界は今夜測っていない**ので「同じ段」の比較はできない。参考として、fp8 の既定構成の c（121 コマ）より、また前回（2026-09-26）の fp8 全 on で溢れた 161 コマ（+3,864.8・1 回）より上で快適 |
| **Q4'**（任意）2.3 Kijai int8 ConvRot は silveroxides と同じか | **違う（上）。c は 137 コマ（36,720・線比 82%）以上** | 既定構成で 121 コマ（−8.3）・129 コマ（−41.6）が各 1 回快適、137 コマが 2 回とも快適（−37.9／−31.4）。`x_`・`f_` が 2 回とも溢れた 129 コマが快適。**梯子の上限（fp8 の c＋2 段）で止めたので、真の境界は測っていない** |

### 1.3 まとめ

- **LTX 2.5 の全 on では、測った 3 種（fp8・REDGraft 混在・int8 ConvRot）の 1080p の c がすべて 145 コマ（38,760・線比 86%）で揃いました。** 3 種とも 153 コマ（40,800）が 2 回とも溢れです。今夜の fp8（uncensored）の境界は、前回（2026-09-26）の別の fp8 ファイル（guillaume）の境界（145 コマ 快適×2・153 コマ 溢れ×2）と同じ段でした。
- **LTX 2.3 の既定構成では、fp8mixed と silveroxides int8 の c が 121 コマ（32,640）で揃い、Kijai int8 ConvRot だけが 137 コマ（36,720）以上でした。** fp8mixed の境界は前回の診断（121 コマ 快適×2・129 コマ 溢れ）と同じ段です。
- **LTX 2.3 の w4a8 は、全 on のまま線上（44,880）まで 2 回とも快適でした。** 今夜測った量子化 safetensors のうち、LTX 2.3 を全 on で測れたのはこの 1 本だけです。
- 失敗・打ち切り・取り直しは 0 回、47 ランすべてで `transformer_used` が要求と一致しました。コミット（Windows の仮想メモリの予約）の最大は上限の 84.1% で、停止線（95%）にも警戒線（90%）にも達していません。
- なぜ Kijai だけ上に出たのか、なぜ REDGraft の safetensors が同じ系統の Q6_K GGUF より下に出たのかは、**この較正では切り分けていません**（観測事実は第6節 表3）。

### 1.4 生成時間の比（§1-33 の Go／No-go の材料）

**比は、両方の腕で基準点か「2 回とも快適」だった点だけで取りました**（溢れた点は遅くなるので混ぜていません）。数値は第6節 表5です。

- **LTX 2.5（全 on）の int8 ConvRot（`h_`）対 fp8（`u_`）**: 第2段階の窓の秒で 89 コマ ×1.033・145 コマ ×1.019、生成全体の秒で ×1.045・×1.030。
- **LTX 2.3（既定構成）の silveroxides int8（`x_`・ConvRot なし）対 fp8（`f_`）**: 第2段階の窓で ×0.999・×1.000、生成全体で ×1.024・×0.981。
- **LTX 2.3（既定構成）の Kijai int8 ConvRot（`k_`）対 fp8（`f_`）**: 第2段階の窓で 89 コマ ×1.020、生成全体で ×0.973（121 コマは `k_` が 1 回だけなので参考: ×1.014・×0.987）。
- 揺れの目安: 同じ点の 1 回目と 2 回目の差は、第2段階の窓で最大 0.70 秒（1.6%）、生成全体で最大 5.93 秒（2.7%・`f_` の 121 コマ）でした。窓の秒は約 1 秒おきの標本から測るので、約 1 秒の刻みがあります。

**ここから Go／No-go は判断していません。** 判断は §1-33 で行います。

---

## 2. 環境と記録

### 2.1 機体・サーバー・コード

| 項目 | 値 |
|---|---|
| GPU | VRAM 16 GB のカード 1 枚（WDDM の LUID `luid_0x00000000_0x0000f316_phys_0`・`luid.json`） |
| コミットの上限 | **112.11 GiB**（`\Memory\Commit Limit`。計測の間ずっと動かず・`commit_log.csv` 5,243 行）。警戒線 90%＝約 100.9 GiB、停止線 95%＝約 106.5 GiB |
| 開始時のコミット | 19.37 GiB（17.3%・`commit_log.csv` の最初の行 23:32:56）。門 P0 の条件は 28 GiB 以下 |
| 開始時の GPU | 専有 785 MiB（`nvidia-smi`・門 P0。条件は 2,000 MiB 以下） |
| サーバー | 計測の前から稼働していたもの（計画 §2 でオーナーに `run.bat` での起動をお願いしたもの）。較正台は `server start`（`--pid` なし）で登録しただけで、起動も再起動もしていない |
| 動いているコード | 本体 dev **`fc9a285`**（作業ツリー clean・`git_state_start.txt`）。計測中に本体のコードは変えていない |
| 開始時の選択 | `active_base_model` が LTX23、LTX23 は `sulphur_distil_fp8mixed`、LTX25 は `redgraftLTX25Fast2K_ltx25RedgraftNSFW`（`original_state.json`・23:33:11） |
| 着手前の行列 | 空（門 P0・`progress.md` 23:32） |

### 2.2 較正台

**2026-09-26 の較正台のコードを複製し、`drive2.sh`・`commit_table.py`・`digest.py` のパス、`README.md`、自己検査の節 H だけを変えています**（`harness_diff.txt`）。`calib.py`・`mkplan.py`・`make_results.py`・`batch_table.py`・`vram_sampler2.ps1`・`commit_monitor.ps1`・`analyse_selftest.py` は 1 バイトも変えていません。

### 2.3 GPU を使う前に通した検査

| 検査 | 結果 | 記録 |
|---|---|---|
| `selftest\parser_selftest.py` | 299 件・失敗 0 | `selftest_output.txt` |
| `selftest\analyse_selftest.py` | 43 件・失敗 0 | `analyse_selftest_output.txt` |
| `calib.py preflight`（オフライン・Tier A の 33 点） | 失敗 0（準備時 2026-09-27 02:52 と、門 P0 の 23:32 の 2 回） | `preflight_output.txt`（準備時）・`preflight_output_start.txt`（門 P0） |
| 門 P0-1〜P0-8 | すべて合格（サーバー稼働・7 本の変換器が実在・dev `fc9a285` clean・GPU 専有 785 MiB・コミット 19.2 GiB・監視の起動・スリープと休止なし） | `progress.md` 23:32 |

Tier B として足した 18 本の計画ファイルは、1 本ずつ `preflight --plan` を通してから流しています。`preflight.json` は最後の 1 本（`p_k_s_1080_137_d_r2.json`・03:48・失敗 0）で上書きされています。

### 2.4 測定の実績

| 項目 | 実測 |
|---|---|
| 走らせた点 | **47 ラン**。全点 `completed`・失敗 0・待ち時間切れ 0・422 は 0・無効 0 |
| 内訳 | 捨てラン 7（各腕 1）／基準点 8（1080p 89 コマ 7・720p 169 コマ 1）／判定した点 32 |
| 時間帯 | 最初の読み込み（起点）23:33:11 から最後の復元 03:55:18 まで（4 時間 22 分）。計測点そのものは 23:37:13〜03:54:16（冷却込み 257.0 分・実行だけの合計 128.7 分。第5節末尾） |
| 冷却 | 点と点のあいだ 150 秒（`drive2.sh` が各点の前に入れた） |
| 待ち時間切れの上限 | 1 点あたり 3,600 秒 |
| 変換器の切り替え | 7 回（`load_*_output.txt`）。どれも読み込みの応答と `GET /models` の照合が要求した名前と一致。切り替えのたびに待機時の床を 3 分取り直した（`idle_*_output.txt`） |
| 後片づけ | 最後に `restore-state` を 2 回（LTX 2.5 を明示 → 明示なし）。最後の `state.json` は開始時の控えと一致（`restore_state_output.txt`） |

---

## 3. 点の設計と、実際に走らせた順番

### 3.1 点の設計（要点）

**点の一覧（ラベル・幾何・トークン・基準点）は `README.md` 第3節が正本です。**

- トークン数は (幅÷32)×(高さ÷32)×潜在フレーム数、潜在フレーム数は (フレーム数−1)÷8＋1 です。
- ラベルは `<頭>_s_<幾何>_<コマ数>`。`b<数>` が基準点、`_r2` が 2 回目、`_d` が既定構成、`*_warm` が捨てラン（768×512・121 コマ・判定しない）です。
- 基準点は同じ幾何・同じ変換器・同じ構成で取りました（1080p は 89 コマ、`r_` の 720p は 169 コマ）。
- 全点に共通: T2V（文字から動画）・シード 12345・24 fps・指示文は較正台の既定。参照動画なし。

### 3.2 実際に走らせた順番

時刻は `progress.md`（監督の日誌）と `manifest.jsonl`（`make_results.py` の「所要時間」の表）によります。1 弾は `drive2.sh` の 1 回の起動です。

| 弾 | 点（走らせた順） | 時刻 | 結果の要点 |
|---|---|---|---|
| （腕 `u_`） | 読み込み・待機時の床 3 分 | 23:33〜23:37 | 起点 T0＝23:33:11 |
| `u_A` | `u_warm`・`u_s_1080_b89`・`u_s_1080_145`・`u_s_1080_153` | 23:37〜23:53 | 145 快適・153 溢れ |
| `u_B` | `u_s_1080_145_r2`・`u_s_1080_153_r2` | 23:56〜00:04 | 145 快適×2・153 溢れ×2 → c＝145 |
| （腕 `r_`） | 読み込み・床 3 分 | 00:04〜00:08 | |
| `r_A` | `r_warm`・`r_s_1080_b89`・`r_s_1080_153`・`r_s_1080_161` | 00:08〜00:26 | 153・161 とも溢れ → `u_` の c の段（145）へ降りる |
| `r_B` | `r_s_1080_145`・`r_s_1080_145_r2` | 00:29〜00:36 | 145 快適×2 |
| `r_C` | `r_s_1080_153_r2`・`r_s_720_b169`・`r_s_720_313`・`r_s_720_313_r2` | 00:39〜00:56 | 153 溢れ×2 → c＝145。720p 313 快適×2 |
| `r_D` | `r_s_720_337` | 00:59〜01:03 | 720p 337 溢れ |
| （腕 `h_`） | 読み込み・床 3 分 | 01:03〜01:06 | |
| `h_A` | `h_warm`・`h_s_1080_b89`・`h_s_1080_145`・`h_s_1080_153` | 01:06〜01:23 | 145 快適・153 溢れ |
| `h_B` | `h_s_1080_145_r2`・`h_s_1080_153_r2` | 01:25〜01:34 | c＝145（`u_` と同じ）→ 規則により 720p なし |
| （腕 `f_`・LTX 2.3 へ切り替え） | 読み込み・床 3 分 | 01:34〜01:38 | |
| `f_A` | `f_warm_d`・`f_s_1080_b89_d`・`f_s_1080_121_d`・`f_s_1080_129_d` | 01:38〜01:57 | 121 快適・129 溢れ |
| `f_B` | `f_s_1080_121_d_r2`・`f_s_1080_129_d_r2` | 02:00〜02:10 | c＝121 |
| （腕 `x_`） | 読み込み・床 3 分 | 02:10〜02:14 | |
| `x_A` | `x_warm_d`・`x_s_1080_b89_d`・`x_s_1080_121_d`・`x_s_1080_129_d` | 02:14〜02:33 | 121 快適・129 溢れ |
| `x_B` | `x_s_1080_121_d_r2`・`x_s_1080_129_d_r2` | 02:36〜02:46 | c＝121（`f_` と同じ） |
| （腕 `j_`） | 読み込み・床 3 分 | 02:46〜02:50 | |
| `j_G`（入口判定） | `j_warm`・`j_s_1080_b89`（全 on） | 02:50〜02:55 | コミット最大 59.8%・70.0% → 全 on で測る |
| `j_A` | `j_s_1080_145`・`j_s_1080_161` | 02:58〜03:06 | 両方快適 → 169 へ |
| `j_B` | `j_s_1080_169` | 03:09〜03:12 | 快適 |
| `j_C` | `j_s_1080_169_r2` | 03:14〜03:17 | 快適×2 →「線まで成り立つ」 |
| （腕 `k_`） | 読み込み・床 3 分 | 03:18〜03:21 | 経過 3:45（k_ の関門 7:00 の内側） |
| `k_A` | `k_warm_d`・`k_s_1080_b89_d`・`k_s_1080_121_d`・`k_s_1080_129_d` | 03:21〜03:40 | 121・129 とも快適 → 137 へ |
| `k_B` | `k_s_1080_137_d` | 03:43〜03:47 | 快適。梯子の上限 |
| `k_C` | `k_s_1080_137_d_r2` | 03:50〜03:54 | 快適×2 → c は 137 以上 |
| 復元 | `restore-state` ×2 | 03:54:46〜03:55:18 | 開始時の状態へ |

### 3.3 計画から変えたところ

- **腕も点の規則も計画どおりです。** 計画の見込み（6〜7 時間・約 48〜53 ラン）に対し、実績は 4 時間 22 分・47 ランでした。
- 時間の関門（`r_` 720p は 1:50・`h_` 720p は 3:15・`k_` は 7:00）はどれにもかかっていません。**`h_` の 720p を走らせていないのは時間ではなく規則による**ものです（1080p の c が `u_` と同じだったため）。
- `j_` は入口判定を通ったので、既定構成版の計画ファイル 4 本（`p_j_warm_d`・`p_j_s_1080_b89_d`・`p_j_s_1080_129_d`・`p_j_s_1080_145_d`）は使っていません。
- `k_` は計画の「`x_` と同じ規則（fp8 の c から上下 2 段まで）」に従い、137 コマ（121＋2 段）で上に登るのを止めました。

---

## 4. 判定の読み方

**規則 v3′ です。同じ幾何・同じ変換器・同じ構成の基準点と比べて、第2段階（Stage-2）の窓と復元（VAE でのデコード）の窓の両方で、共有 GPU メモリ（VRAM に入り切らない分をシステムメモリへ逃がす領域）の中央値が +300 メガバイト以上は上がらないことを「快適」とします。** 較正台の出力では、快適が `clean`、溢れが `plateau` です。規則の正本は `Docs/COMFORT_LIMIT_TABLE.md` 第4.2節、較正台の規約は `outputs/comfort-calib-2026-09-04/README.md` 第5節〜第7節です。

- 判定は `runs/<ラベル>/run.json` の `verdict_v3prime` と、`v3prime.windows.stage2`・`v3prime.windows.restore` の `delta_mb` で読みます。10 標本未満の窓は判定不能です（**今回、判定した 32 点で判定不能の窓はありません**）。
- LTX 2.3 の点の `verdict_v3prime` は 5 つの窓の複合（`s23_stage1`・`s23_gap_mid`・`s23_gap_tail` を足したもの）、LTX 2.5 の点は 3 つの窓の複合（`p25_30_decode_encode` を足したもの）です。**今回、2 窓の判定と複合の判定が食い違った点はありません。**
- **復元の窓は、判定した 32 点のすべてで +300 未満でした**（最大は `f_s_1080_121_d` の +6.9）。溢れはすべて第2段階の窓で起きています。
- **2 回一致の規則**: 境界の材料にするのは、2 回とも同じ判定になった点だけです。**比べるのは「2 回とも快適だった上端 c」だけ**で、溢れ側は「c の 1 段上が 2 回快適ではない」ことだけを要件にします。「同じ」は、c が同じプロセスの fp8 の c と同じ段であることです。1 段でもずれれば「上／下」とします。今夜は、2 回で割れた点も、逆行（下の段が溢れて上の段が快適）もありませんでした。
- 刻みは 1920×1088 の潜在 1 コマ＝2,040 トークンです。
- **専有の値は、第2段階の窓での WDDM の専有ピーク（カード全体・約 1 秒に 1 回の採取・`vram.stage2_window.dedicated_peak_mb`）です。** torch の割当ピークは `metadata.json` の `peak_vram_mb` です（全点が参照動画なしなので、点どうしで比べられます）。
- **最初のジョブの所要は比べません。** 読み込みと keep_resident の写しが混ざるので、各読み込みの後に捨てランを置いています。

---

## 5. 結果表（測定後・`make_results.py` の機械生成）

**この節は `make_results.py` の出力（`make_results_output.md`）をそのまま取り込んだものです。** もとの数値は `runs\<ラベル>\run.json` と `manifest.jsonl` にあり、そこから機械が読み直して表にしています。

**読むときの注意**（機械が出す表そのものには書いていないこと）:

- 表の `v1`・`v2` 列は**古い判定規則の出力**です。今回の判定は `v3′` 列です。
- 捨てラン（`*_warm`・`*_warm_d`）は形の上では基準点（`baseline`）ですが、どの点もこれを基準に参照していません。
- 「速度と共有メモリ」の表の `stage2秒` は、サーバーのログの 1 段あたりの秒×段数から出した値で、第6節 表5 の主に使う「第2段階の窓の秒」（`v3prime.windows.stage2.seconds`）とは別の値です。
- `keep_resident_embeddings` が LTX 2.3 の点で *null* なのは仕様です（表の下の注記のとおり）。既定構成の点（`_d`）の `attention` が `sdpa`・`keep_resident` と `vae_mode` が `off` なのは、サーバー既定で走らせたためで、照合は「既定構成を宣言した点が既定構成で走った」ことを確かめています。
- fused カーネル（`fused_gguf_dequant_kernel`）は safetensors の変換器には効きませんが、記録上は `on` と出ます。

---

<!-- ここから `make_results_output.md` の逐語 -->
### 全計測点

| ラベル | 系統 | 幾何 | トークン | 状態 | v1 | v2 | **v3′** | 加速照合 |
|---|---|---|---:|---|---|---|---|---|
| `u_warm` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `u_s_1080_b89` | LTX25 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `u_s_1080_145` | LTX25 | 1920x1088/145f | 38,760 | completed | inconclusive | organic | **clean** | ok |
| `u_s_1080_153` | LTX25 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `u_s_1080_145_r2` | LTX25 | 1920x1088/145f | 38,760 | completed | inconclusive | organic | **clean** | ok |
| `u_s_1080_153_r2` | LTX25 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `r_warm` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `r_s_1080_b89` | LTX25 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `r_s_1080_153` | LTX25 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `r_s_1080_161` | LTX25 | 1920x1088/161f | 42,840 | completed | harmful | harmful | **plateau** | ok |
| `r_s_1080_145` | LTX25 | 1920x1088/145f | 38,760 | completed | harmful | organic | **clean** | ok |
| `r_s_1080_145_r2` | LTX25 | 1920x1088/145f | 38,760 | completed | harmful | organic | **clean** | ok |
| `r_s_1080_153_r2` | LTX25 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `r_s_720_b169` | LTX25 | 1280x768/169f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `r_s_720_313` | LTX25 | 1280x768/313f | 38,400 | completed | harmful | organic | **clean** | ok |
| `r_s_720_313_r2` | LTX25 | 1280x768/313f | 38,400 | completed | harmful | organic | **clean** | ok |
| `r_s_720_337` | LTX25 | 1280x768/337f | 41,280 | completed | harmful | harmful | **plateau** | ok |
| `h_warm` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `h_s_1080_b89` | LTX25 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `h_s_1080_145` | LTX25 | 1920x1088/145f | 38,760 | completed | harmful | organic | **clean** | ok |
| `h_s_1080_153` | LTX25 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `h_s_1080_145_r2` | LTX25 | 1920x1088/145f | 38,760 | completed | harmful | organic | **clean** | ok |
| `h_s_1080_153_r2` | LTX25 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `f_warm_d` | LTX23 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `f_s_1080_b89_d` | LTX23 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `f_s_1080_121_d` | LTX23 | 1920x1088/121f | 32,640 | completed | harmful | organic | **clean** | ok |
| `f_s_1080_129_d` | LTX23 | 1920x1088/129f | 34,680 | completed | inconclusive | harmful | **plateau** | ok |
| `f_s_1080_121_d_r2` | LTX23 | 1920x1088/121f | 32,640 | completed | inconclusive | organic | **clean** | ok |
| `f_s_1080_129_d_r2` | LTX23 | 1920x1088/129f | 34,680 | completed | inconclusive | harmful | **plateau** | ok |
| `x_warm_d` | LTX23 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `x_s_1080_b89_d` | LTX23 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `x_s_1080_121_d` | LTX23 | 1920x1088/121f | 32,640 | completed | inconclusive | organic | **clean** | ok |
| `x_s_1080_129_d` | LTX23 | 1920x1088/129f | 34,680 | completed | inconclusive | harmful | **plateau** | ok |
| `x_s_1080_121_d_r2` | LTX23 | 1920x1088/121f | 32,640 | completed | inconclusive | organic | **clean** | ok |
| `x_s_1080_129_d_r2` | LTX23 | 1920x1088/129f | 34,680 | completed | inconclusive | harmful | **plateau** | ok |
| `j_warm` | LTX23 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `j_s_1080_b89` | LTX23 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `j_s_1080_145` | LTX23 | 1920x1088/145f | 38,760 | completed | harmful | organic | **clean** | ok |
| `j_s_1080_161` | LTX23 | 1920x1088/161f | 42,840 | completed | harmful | organic | **clean** | ok |
| `j_s_1080_169` | LTX23 | 1920x1088/169f | 44,880 | completed | organic | organic | **clean** | ok |
| `j_s_1080_169_r2` | LTX23 | 1920x1088/169f | 44,880 | completed | organic | organic | **clean** | ok |
| `k_warm_d` | LTX23 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `k_s_1080_b89_d` | LTX23 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `k_s_1080_121_d` | LTX23 | 1920x1088/121f | 32,640 | completed | inconclusive | organic | **clean** | ok |
| `k_s_1080_129_d` | LTX23 | 1920x1088/129f | 34,680 | completed | organic | organic | **clean** | ok |
| `k_s_1080_137_d` | LTX23 | 1920x1088/137f | 36,720 | completed | organic | organic | **clean** | ok |
| `k_s_1080_137_d_r2` | LTX23 | 1920x1088/137f | 36,720 | completed | organic | organic | **clean** | ok |

### 加速が実際に効いたか（`metadata.json` の `*_used`）

| ラベル | transformer | attention | block_swap_prefetch | keep_resident | fused_gguf_dequant_kernel | vae_mode | keep_resident_embeddings | ok |
|---|---|---|---|---|---|---|---|---|
| `u_warm` | `ltx25_uncensored_v1.1-fp8_scaled` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `u_s_1080_b89` | `ltx25_uncensored_v1.1-fp8_scaled` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `u_s_1080_145` | `ltx25_uncensored_v1.1-fp8_scaled` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `u_s_1080_153` | `ltx25_uncensored_v1.1-fp8_scaled` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `u_s_1080_145_r2` | `ltx25_uncensored_v1.1-fp8_scaled` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `u_s_1080_153_r2` | `ltx25_uncensored_v1.1-fp8_scaled` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `r_warm` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `r_s_1080_b89` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `r_s_1080_153` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `r_s_1080_161` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `r_s_1080_145` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `r_s_1080_145_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `r_s_1080_153_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `r_s_720_b169` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `r_s_720_313` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `r_s_720_313_r2` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `r_s_720_337` | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `h_warm` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `h_s_1080_b89` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `h_s_1080_145` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `h_s_1080_153` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `h_s_1080_145_r2` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `h_s_1080_153_r2` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `f_warm_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_b89_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_121_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_129_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_121_d_r2` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_129_d_r2` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `x_warm_d` | `ltx-2.3-22b-distilled-1.1_int8mixedtensorwise` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `x_s_1080_b89_d` | `ltx-2.3-22b-distilled-1.1_int8mixedtensorwise` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `x_s_1080_121_d` | `ltx-2.3-22b-distilled-1.1_int8mixedtensorwise` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `x_s_1080_129_d` | `ltx-2.3-22b-distilled-1.1_int8mixedtensorwise` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `x_s_1080_121_d_r2` | `ltx-2.3-22b-distilled-1.1_int8mixedtensorwise` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `x_s_1080_129_d_r2` | `ltx-2.3-22b-distilled-1.1_int8mixedtensorwise` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `j_warm` | `ltx-2.3-22b-distilled-1.1_w4a8` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `j_s_1080_b89` | `ltx-2.3-22b-distilled-1.1_w4a8` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `j_s_1080_145` | `ltx-2.3-22b-distilled-1.1_w4a8` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `j_s_1080_161` | `ltx-2.3-22b-distilled-1.1_w4a8` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `j_s_1080_169` | `ltx-2.3-22b-distilled-1.1_w4a8` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `j_s_1080_169_r2` | `ltx-2.3-22b-distilled-1.1_w4a8` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `k_warm_d` | `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `k_s_1080_b89_d` | `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `k_s_1080_121_d` | `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `k_s_1080_129_d` | `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `k_s_1080_137_d` | `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `k_s_1080_137_d_r2` | `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |

`keep_resident_embeddings` が 2.3 の点で *null* なのは仕様です（`services/engines/ltx/adapter.py:481`。2.3 に埋め込み処理器が無い）。

### v3′ の窓ごとの判定（基準比・+300MB が線）

**`u_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 7 | × fewer than 10 WDDM samples | — | — |

**`u_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 13 | × no baseline for this window | — | — |
| restore | 13 | × no baseline for this window | — | — |
| stage2 | 25 | × no baseline for this window | — | — |

**`u_s_1080_145`** — v3′=`clean`（基準 `u_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -31.6 | 下 |
| restore | 22 | ○ | -31.6 | 下 |
| stage2 | 44 | ○ | 82.6 | 下 |

**`u_s_1080_153`** — v3′=`plateau`（基準 `u_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -39.1 | 下 |
| restore | 23 | ○ | -39.1 | 下 |
| stage2 | 109 | ○ | 903.3 | **超** |

**`u_s_1080_145_r2`** — v3′=`clean`（基準 `u_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -31.8 | 下 |
| restore | 22 | ○ | -31.8 | 下 |
| stage2 | 43 | ○ | 113.6 | 下 |

**`u_s_1080_153_r2`** — v3′=`plateau`（基準 `u_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -31.8 | 下 |
| restore | 23 | ○ | -31.8 | 下 |
| stage2 | 83 | ○ | 622.5 | **超** |

**`r_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 8 | × fewer than 10 WDDM samples | — | — |

**`r_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 13 | × no baseline for this window | — | — |
| restore | 13 | × no baseline for this window | — | — |
| stage2 | 27 | × no baseline for this window | — | — |

**`r_s_1080_153`** — v3′=`plateau`（基準 `r_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -26.7 | 下 |
| restore | 23 | ○ | -26.7 | 下 |
| stage2 | 76 | ○ | 558.2 | **超** |

**`r_s_1080_161`** — v3′=`plateau`（基準 `r_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -33.6 | 下 |
| restore | 24 | ○ | -33.6 | 下 |
| stage2 | 130 | ○ | 1191.0 | **超** |

**`r_s_1080_145`** — v3′=`clean`（基準 `r_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -35.7 | 下 |
| restore | 22 | ○ | -35.7 | 下 |
| stage2 | 44 | ○ | 58.0 | 下 |

**`r_s_1080_145_r2`** — v3′=`clean`（基準 `r_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -47.3 | 下 |
| restore | 22 | ○ | -47.3 | 下 |
| stage2 | 44 | ○ | -35.1 | 下 |

**`r_s_1080_153_r2`** — v3′=`plateau`（基準 `r_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -36.4 | 下 |
| restore | 23 | ○ | -36.4 | 下 |
| stage2 | 89 | ○ | 618.5 | **超** |

**`r_s_720_b169`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 11 | × no baseline for this window | — | — |
| restore | 11 | × no baseline for this window | — | — |
| stage2 | 23 | × no baseline for this window | — | — |

**`r_s_720_313`** — v3′=`clean`（基準 `r_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -23.7 | 下 |
| restore | 22 | ○ | -23.7 | 下 |
| stage2 | 44 | ○ | -34.0 | 下 |

**`r_s_720_313_r2`** — v3′=`clean`（基準 `r_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -15.4 | 下 |
| restore | 23 | ○ | -15.4 | 下 |
| stage2 | 44 | ○ | -25.5 | 下 |

**`r_s_720_337`** — v3′=`plateau`（基準 `r_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -41.4 | 下 |
| restore | 23 | ○ | -41.4 | 下 |
| stage2 | 86 | ○ | 678.5 | **超** |

**`h_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 7 | × fewer than 10 WDDM samples | — | — |

**`h_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 13 | × no baseline for this window | — | — |
| restore | 13 | × no baseline for this window | — | — |
| stage2 | 26 | × no baseline for this window | — | — |

**`h_s_1080_145`** — v3′=`clean`（基準 `h_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -7.0 | 下 |
| restore | 22 | ○ | -7.0 | 下 |
| stage2 | 44 | ○ | 153.6 | 下 |

**`h_s_1080_153`** — v3′=`plateau`（基準 `h_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -36.6 | 下 |
| restore | 23 | ○ | -36.6 | 下 |
| stage2 | 96 | ○ | 814.2 | **超** |

**`h_s_1080_145_r2`** — v3′=`clean`（基準 `h_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -31.4 | 下 |
| restore | 22 | ○ | -31.4 | 下 |
| stage2 | 44 | ○ | 140.3 | 下 |

**`h_s_1080_153_r2`** — v3′=`plateau`（基準 `h_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -15.4 | 下 |
| restore | 23 | ○ | -15.4 | 下 |
| stage2 | 84 | ○ | 695.0 | **超** |

**`f_warm_d`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 9 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 5 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 9 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 10 | × no baseline for this window | — | — |
| stage2 | 7 | × fewer than 10 WDDM samples | — | — |

**`f_s_1080_b89_d`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 28 | × no baseline for this window | — | — |
| s23_gap_mid | 20 | × no baseline for this window | — | — |
| s23_gap_tail | 28 | × no baseline for this window | — | — |
| s23_stage1 | 26 | × no baseline for this window | — | — |
| stage2 | 39 | × no baseline for this window | — | — |

**`f_s_1080_121_d`** — v3′=`clean`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 41 | ○ | 6.9 | 下 |
| s23_gap_mid | 31 | ○ | -27.1 | 下 |
| s23_gap_tail | 41 | ○ | 6.9 | 下 |
| s23_stage1 | 34 | ○ | -36.9 | 下 |
| stage2 | 58 | ○ | 108.3 | 下 |

**`f_s_1080_129_d`** — v3′=`plateau`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 43 | ○ | -27.2 | 下 |
| s23_gap_mid | 34 | ○ | 5.0 | 下 |
| s23_gap_tail | 43 | ○ | -27.2 | 下 |
| s23_stage1 | 37 | ○ | -36.4 | 下 |
| stage2 | 66 | ○ | 726.3 | **超** |

**`f_s_1080_121_d_r2`** — v3′=`clean`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 42 | ○ | -0.1 | 下 |
| s23_gap_mid | 30 | ○ | -0.1 | 下 |
| s23_gap_tail | 42 | ○ | -0.1 | 下 |
| s23_stage1 | 35 | ○ | -36.1 | 下 |
| stage2 | 58 | ○ | 28.6 | 下 |

**`f_s_1080_129_d_r2`** — v3′=`plateau`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 44 | ○ | -0.1 | 下 |
| s23_gap_mid | 35 | ○ | 16.5 | 下 |
| s23_gap_tail | 44 | ○ | -0.1 | 下 |
| s23_stage1 | 37 | ○ | -35.9 | 下 |
| stage2 | 65 | ○ | 747.4 | **超** |

**`x_warm_d`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 9 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 5 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 9 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 11 | × no baseline for this window | — | — |
| stage2 | 7 | × fewer than 10 WDDM samples | — | — |

**`x_s_1080_b89_d`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 29 | × no baseline for this window | — | — |
| s23_gap_mid | 20 | × no baseline for this window | — | — |
| s23_gap_tail | 29 | × no baseline for this window | — | — |
| s23_stage1 | 26 | × no baseline for this window | — | — |
| stage2 | 38 | × no baseline for this window | — | — |

**`x_s_1080_121_d`** — v3′=`clean`（基準 `x_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 42 | ○ | -18.6 | 下 |
| s23_gap_mid | 30 | ○ | -38.1 | 下 |
| s23_gap_tail | 42 | ○ | -18.6 | 下 |
| s23_stage1 | 34 | ○ | 15.5 | 下 |
| stage2 | 58 | ○ | 21.2 | 下 |

**`x_s_1080_129_d`** — v3′=`plateau`（基準 `x_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 44 | ○ | -18.1 | 下 |
| s23_gap_mid | 34 | ○ | -11.3 | 下 |
| s23_gap_tail | 44 | ○ | -18.1 | 下 |
| s23_stage1 | 37 | ○ | 15.5 | 下 |
| stage2 | 65 | ○ | 728.8 | **超** |

**`x_s_1080_121_d_r2`** — v3′=`clean`（基準 `x_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 42 | ○ | 0.7 | 下 |
| s23_gap_mid | 30 | ○ | -9.7 | 下 |
| s23_gap_tail | 42 | ○ | 0.7 | 下 |
| s23_stage1 | 34 | ○ | -0.7 | 下 |
| stage2 | 58 | ○ | 46.8 | 下 |

**`x_s_1080_129_d_r2`** — v3′=`plateau`（基準 `x_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 43 | ○ | 1.7 | 下 |
| s23_gap_mid | 34 | ○ | 8.5 | 下 |
| s23_gap_tail | 43 | ○ | 1.7 | 下 |
| s23_stage1 | 37 | ○ | -0.2 | 下 |
| stage2 | 65 | ○ | 748.5 | **超** |

**`j_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 6 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 5 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 6 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 19 | × no baseline for this window | — | — |
| stage2 | 9 | × fewer than 10 WDDM samples | — | — |

**`j_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 19 | × no baseline for this window | — | — |
| s23_gap_mid | 14 | × no baseline for this window | — | — |
| s23_gap_tail | 19 | × no baseline for this window | — | — |
| s23_stage1 | 32 | × no baseline for this window | — | — |
| stage2 | 28 | × no baseline for this window | — | — |

**`j_s_1080_145`** — v3′=`clean`（基準 `j_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 36 | ○ | -9.4 | 下 |
| s23_gap_mid | 22 | ○ | 27.4 | 下 |
| s23_gap_tail | 36 | ○ | -9.4 | 下 |
| s23_stage1 | 44 | ○ | 16.1 | 下 |
| stage2 | 44 | ○ | 19.0 | 下 |

**`j_s_1080_161`** — v3′=`clean`（基準 `j_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 38 | ○ | -7.8 | 下 |
| s23_gap_mid | 25 | ○ | -25.7 | 下 |
| s23_gap_tail | 38 | ○ | -7.8 | 下 |
| s23_stage1 | 47 | ○ | 1.4 | 下 |
| stage2 | 49 | ○ | -3.9 | 下 |

**`j_s_1080_169`** — v3′=`clean`（基準 `j_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 38 | ○ | -26.7 | 下 |
| s23_gap_mid | 27 | ○ | 35.5 | 下 |
| s23_gap_tail | 38 | ○ | -26.7 | 下 |
| s23_stage1 | 48 | ○ | 19.6 | 下 |
| stage2 | 52 | ○ | 27.1 | 下 |

**`j_s_1080_169_r2`** — v3′=`clean`（基準 `j_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 39 | ○ | -16.0 | 下 |
| s23_gap_mid | 27 | ○ | 12.8 | 下 |
| s23_gap_tail | 39 | ○ | -16.0 | 下 |
| s23_stage1 | 48 | ○ | 0.3 | 下 |
| stage2 | 52 | ○ | 12.8 | 下 |

**`k_warm_d`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 9 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 5 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 9 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 13 | × no baseline for this window | — | — |
| stage2 | 8 | × fewer than 10 WDDM samples | — | — |

**`k_s_1080_b89_d`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 28 | × no baseline for this window | — | — |
| s23_gap_mid | 21 | × no baseline for this window | — | — |
| s23_gap_tail | 28 | × no baseline for this window | — | — |
| s23_stage1 | 28 | × no baseline for this window | — | — |
| stage2 | 39 | × no baseline for this window | — | — |

**`k_s_1080_121_d`** — v3′=`clean`（基準 `k_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 41 | ○ | -8.1 | 下 |
| s23_gap_mid | 31 | ○ | 35.7 | 下 |
| s23_gap_tail | 41 | ○ | -8.1 | 下 |
| s23_stage1 | 37 | ○ | -17.1 | 下 |
| stage2 | 59 | ○ | -8.3 | 下 |

**`k_s_1080_129_d`** — v3′=`clean`（基準 `k_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 43 | ○ | -8.2 | 下 |
| s23_gap_mid | 33 | ○ | 37.4 | 下 |
| s23_gap_tail | 43 | ○ | -8.2 | 下 |
| s23_stage1 | 40 | ○ | 1.6 | 下 |
| stage2 | 64 | ○ | -41.6 | 下 |

**`k_s_1080_137_d`** — v3′=`clean`（基準 `k_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 44 | ○ | -33.2 | 下 |
| s23_gap_mid | 36 | ○ | 13.3 | 下 |
| s23_gap_tail | 44 | ○ | -33.2 | 下 |
| s23_stage1 | 42 | ○ | -23.1 | 下 |
| stage2 | 71 | ○ | -37.9 | 下 |

**`k_s_1080_137_d_r2`** — v3′=`clean`（基準 `k_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 44 | ○ | -27.8 | 下 |
| s23_gap_mid | 37 | ○ | 22.1 | 下 |
| s23_gap_tail | 44 | ○ | -27.8 | 下 |
| s23_stage1 | 42 | ○ | -14.4 | 下 |
| stage2 | 70 | ○ | -31.4 | 下 |

### 速度と共有メモリ

| ラベル | 生成秒 | stage2秒 | 秒/トークン | 復元秒 | 予約ピークMB | 共有ピーク(job)MB | 参照符号化 予約MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `u_warm` | 58.4 | 10.5 | 0.001709 | 3.1 | 7,180 | 1724.4 | — |
| `u_s_1080_b89` | 85.5 | 37.8 | 0.001544 | 13.1 | 11,906 | 1646.5 | — |
| `u_s_1080_145` | 134.0 | 65.1 | 0.001680 | 22.1 | 15,302 | 1729.9 | — |
| `u_s_1080_153` | 236.2 | 163.2 | 0.004000 | 23.1 | 15,846 | 2136.8 | — |
| `u_s_1080_145_r2` | 134.0 | 64.0 | 0.001652 | 22.1 | 15,302 | 1793.7 | — |
| `u_s_1080_153_r2` | 197.6 | 124.5 | 0.003051 | 23.1 | 15,846 | 1872.9 | — |
| `r_warm` | 63.0 | 12.4 | 0.002026 | 3.0 | 7,180 | 1692.5 | — |
| `r_s_1080_b89` | 94.3 | 40.4 | 0.001648 | 13.1 | 11,554 | 1643.1 | — |
| `r_s_1080_153` | 193.5 | 114.3 | 0.002801 | 23.1 | 15,644 | 1743.6 | — |
| `r_s_1080_161` | 277.8 | 195.4 | 0.004562 | 24.0 | 16,270 | 2374.4 | — |
| `r_s_1080_145` | 142.5 | 66.3 | 0.001711 | 22.2 | 15,084 | 1630.9 | — |
| `r_s_1080_145_r2` | 142.9 | 66.3 | 0.001711 | 22.1 | 15,084 | 1499.8 | — |
| `r_s_1080_153_r2` | 212.1 | 133.5 | 0.003272 | 23.1 | 15,644 | 1790.1 | — |
| `r_s_720_b169` | 83.3 | 34.5 | 0.001634 | 11.3 | 10,694 | 1681.3 | — |
| `r_s_720_313` | 141.7 | 65.5 | 0.001707 | 22.2 | 14,966 | 1657.6 | — |
| `r_s_720_313_r2` | 141.8 | 65.5 | 0.001707 | 22.2 | 14,966 | 1665.9 | — |
| `r_s_720_337` | 208.8 | 128.8 | 0.003121 | 23.4 | 15,836 | 1882.9 | — |
| `h_warm` | 59.0 | 11.2 | 0.001831 | 3.0 | 7,180 | 1667.8 | — |
| `h_s_1080_b89` | 89.3 | 39.0 | 0.001593 | 13.1 | 11,906 | 1639.2 | — |
| `h_s_1080_145` | 137.8 | 65.4 | 0.001687 | 22.1 | 15,288 | 1668.2 | — |
| `h_s_1080_153` | 220.9 | 145.2 | 0.003559 | 23.1 | 15,848 | 2008.4 | — |
| `h_s_1080_145_r2` | 138.3 | 66.2 | 0.001707 | 22.2 | 15,288 | 1705.8 | — |
| `h_s_1080_153_r2` | 201.8 | 125.8 | 0.003085 | 23.0 | 15,848 | 1879.4 | — |
| `f_warm_d` | 66.6 | 11.1 | 0.001807 | 8.6 | 14,972 | 2204.3 | — |
| `f_s_1080_b89_d` | 167.8 | 57.5 | 0.002347 | 28.2 | 12,318 | 2240.2 | — |
| `f_s_1080_121_d` | 222.1 | 87.0 | 0.002665 | 42.0 | 15,190 | 2297.9 | — |
| `f_s_1080_129_d` | 226.4 | 98.2 | 0.002833 | 43.6 | 15,986 | 2960.3 | — |
| `f_s_1080_121_d_r2` | 216.1 | 86.8 | 0.002661 | 42.1 | 15,190 | 2266.2 | — |
| `f_s_1080_129_d_r2` | 235.8 | 98.2 | 0.002833 | 43.6 | 15,986 | 2966.9 | — |
| `x_warm_d` | 68.5 | 11.1 | 0.001807 | 8.7 | 14,972 | 2203.9 | — |
| `x_s_1080_b89_d` | 171.9 | 57.3 | 0.002341 | 28.4 | 12,318 | 2188.3 | — |
| `x_s_1080_121_d` | 215.2 | 87.0 | 0.002665 | 41.9 | 15,190 | 2241.9 | — |
| `x_s_1080_129_d` | 230.1 | 98.2 | 0.002833 | 43.6 | 15,986 | 2950.8 | — |
| `x_s_1080_121_d_r2` | 214.8 | 86.8 | 0.002661 | 41.7 | 15,190 | 2249.9 | — |
| `x_s_1080_129_d_r2` | 225.1 | 98.2 | 0.002833 | 42.8 | 15,986 | 2952.5 | — |
| `j_warm` | 76.2 | 13.8 | 0.002246 | 6.1 | 8,972 | 653.8 | — |
| `j_s_1080_b89` | 103.0 | 40.8 | 0.001667 | 19.5 | 10,210 | 651.5 | — |
| `j_s_1080_145` | 157.2 | 66.2 | 0.001707 | 35.7 | 13,114 | 667.7 | — |
| `j_s_1080_161` | 169.9 | 74.2 | 0.001733 | 37.5 | 14,334 | 653.9 | — |
| `j_s_1080_169` | 176.5 | 78.5 | 0.001748 | 38.1 | 14,926 | 671.1 | — |
| `j_s_1080_169_r2` | 176.8 | 78.5 | 0.001748 | 38.2 | 14,926 | 652.6 | — |
| `k_warm_d` | 68.8 | 12.3 | 0.002002 | 8.5 | 14,604 | 1164.8 | — |
| `k_s_1080_b89_d` | 163.4 | 58.6 | 0.002396 | 28.4 | 12,054 | 1178.6 | — |
| `k_s_1080_121_d` | 216.2 | 88.2 | 0.002702 | 41.5 | 13,574 | 1181.0 | — |
| `k_s_1080_129_d` | 229.3 | 96.6 | 0.002785 | 42.8 | 14,168 | 1180.2 | — |
| `k_s_1080_137_d` | 239.1 | 105.5 | 0.002872 | 44.4 | 14,850 | 1164.2 | — |
| `k_s_1080_137_d_r2` | 239.2 | 105.5 | 0.002872 | 44.1 | 14,850 | 1164.8 | — |

### 所要時間（`manifest.jsonl` の実測）

| ラベル | 開始 | 終了 | 秒 |
|---|---|---|---:|
| `u_warm` | 2026-09-27 23:37:13.023 | 2026-09-27 23:38:13.140 | 60.1 |
| `u_s_1080_b89` | 2026-09-27 23:40:48.236 | 2026-09-27 23:42:14.475 | 86.2 |
| `u_s_1080_145` | 2026-09-27 23:44:49.478 | 2026-09-27 23:47:03.869 | 134.4 |
| `u_s_1080_153` | 2026-09-27 23:49:38.835 | 2026-09-27 23:53:35.470 | 236.6 |
| `u_s_1080_145_r2` | 2026-09-27 23:56:15.830 | 2026-09-27 23:58:30.259 | 134.4 |
| `u_s_1080_153_r2` | 2026-09-28 00:01:05.132 | 2026-09-28 00:04:23.722 | 198.6 |
| `r_warm` | 2026-09-28 00:08:13.690 | 2026-09-28 00:09:17.852 | 64.2 |
| `r_s_1080_b89` | 2026-09-28 00:11:52.893 | 2026-09-28 00:13:29.234 | 96.3 |
| `r_s_1080_153` | 2026-09-28 00:16:04.193 | 2026-09-28 00:19:18.914 | 194.7 |
| `r_s_1080_161` | 2026-09-28 00:21:54.661 | 2026-09-28 00:26:33.555 | 278.9 |
| `r_s_1080_145` | 2026-09-28 00:29:17.746 | 2026-09-28 00:31:42.214 | 144.5 |
| `r_s_1080_145_r2` | 2026-09-28 00:34:17.050 | 2026-09-28 00:36:41.486 | 144.4 |
| `r_s_1080_153_r2` | 2026-09-28 00:39:21.610 | 2026-09-28 00:42:54.266 | 212.7 |
| `r_s_720_b169` | 2026-09-28 00:45:30.077 | 2026-09-28 00:46:54.356 | 84.3 |
| `r_s_720_313` | 2026-09-28 00:49:29.295 | 2026-09-28 00:51:51.762 | 142.5 |
| `r_s_720_313_r2` | 2026-09-28 00:54:26.616 | 2026-09-28 00:56:49.033 | 142.4 |
| `r_s_720_337` | 2026-09-28 00:59:31.643 | 2026-09-28 01:03:02.344 | 210.7 |
| `h_warm` | 2026-09-28 01:06:45.182 | 2026-09-28 01:07:45.346 | 60.2 |
| `h_s_1080_b89` | 2026-09-28 01:10:20.387 | 2026-09-28 01:11:50.648 | 90.3 |
| `h_s_1080_145` | 2026-09-28 01:14:25.669 | 2026-09-28 01:16:44.044 | 138.4 |
| `h_s_1080_153` | 2026-09-28 01:19:19.006 | 2026-09-28 01:23:01.609 | 222.6 |
| `h_s_1080_145_r2` | 2026-09-28 01:25:46.106 | 2026-09-28 01:28:04.525 | 138.4 |
| `h_s_1080_153_r2` | 2026-09-28 01:30:39.474 | 2026-09-28 01:34:02.065 | 202.6 |
| `f_warm_d` | 2026-09-28 01:38:11.968 | 2026-09-28 01:39:20.184 | 68.2 |
| `f_s_1080_b89_d` | 2026-09-28 01:41:55.173 | 2026-09-28 01:44:43.619 | 168.4 |
| `f_s_1080_121_d` | 2026-09-28 01:47:18.915 | 2026-09-28 01:51:01.598 | 222.7 |
| `f_s_1080_129_d` | 2026-09-28 01:53:36.879 | 2026-09-28 01:57:23.597 | 226.7 |
| `f_s_1080_121_d_r2` | 2026-09-28 02:00:06.767 | 2026-09-28 02:03:43.314 | 216.6 |
| `f_s_1080_129_d_r2` | 2026-09-28 02:06:18.329 | 2026-09-28 02:10:15.020 | 236.7 |
| `x_warm_d` | 2026-09-28 02:14:19.286 | 2026-09-28 02:15:29.479 | 70.2 |
| `x_s_1080_b89_d` | 2026-09-28 02:18:04.506 | 2026-09-28 02:20:56.984 | 172.5 |
| `x_s_1080_121_d` | 2026-09-28 02:23:31.975 | 2026-09-28 02:27:08.697 | 216.7 |
| `x_s_1080_129_d` | 2026-09-28 02:29:44.532 | 2026-09-28 02:33:35.183 | 230.7 |
| `x_s_1080_121_d_r2` | 2026-09-28 02:36:16.473 | 2026-09-28 02:39:52.991 | 216.5 |
| `x_s_1080_129_d_r2` | 2026-09-28 02:42:27.950 | 2026-09-28 02:46:14.436 | 226.5 |
| `j_warm` | 2026-09-28 02:50:05.531 | 2026-09-28 02:51:21.812 | 76.3 |
| `j_s_1080_b89` | 2026-09-28 02:53:56.713 | 2026-09-28 02:55:41.102 | 104.4 |
| `j_s_1080_145` | 2026-09-28 02:58:21.947 | 2026-09-28 03:01:00.448 | 158.5 |
| `j_s_1080_161` | 2026-09-28 03:03:35.314 | 2026-09-28 03:06:25.789 | 170.5 |
| `j_s_1080_169` | 2026-09-28 03:09:02.953 | 2026-09-28 03:11:59.508 | 176.6 |
| `j_s_1080_169_r2` | 2026-09-28 03:14:42.449 | 2026-09-28 03:17:41.028 | 178.6 |
| `k_warm_d` | 2026-09-28 03:21:39.811 | 2026-09-28 03:22:50.026 | 70.2 |
| `k_s_1080_b89_d` | 2026-09-28 03:25:25.080 | 2026-09-28 03:28:09.574 | 164.5 |
| `k_s_1080_121_d` | 2026-09-28 03:30:44.528 | 2026-09-28 03:34:21.309 | 216.8 |
| `k_s_1080_129_d` | 2026-09-28 03:36:57.111 | 2026-09-28 03:40:47.867 | 230.8 |
| `k_s_1080_137_d` | 2026-09-28 03:43:32.339 | 2026-09-28 03:47:33.222 | 240.9 |
| `k_s_1080_137_d_r2` | 2026-09-28 03:50:15.141 | 2026-09-28 03:54:15.811 | 240.7 |

- 計測点の実行時間の合計: **128.7 分**
- 最初の点の開始から最後の点の終了まで（冷却込み）: **257.0 分**
- 実行した点の数（再走を含む延べ）: **47**

<!-- `make_results_output.md` の逐語ここまで -->

---

## 6. 種別ごとの判定と比較（表1〜表5）

**この節の表は、第5節の生の表・`run.json`・`commit_table.md` から、判定と比較に使う値だけを抜き出したものです。** 差はすべて**その点自身の基準点と比べた第2段階の窓の共有 GPU メモリ中央値の差**（メガバイト）で、快適と溢れを分ける線は +300 です。2 回走らせた点は「1 回目／2 回目」の順に並べています。過去の記録から引いた値には出典（前回＝`outputs/comfort-calib-2026-09-26/`・§119、§113＝`outputs/comfort-calib-2026-09-17/`、§106＝`outputs/comfort-calib-2026-09-14/`）を付けています。

### 表1 — 種別ごとの 1920×1088 の境界（同じプロセスの fp8 と並べる）

**LTX 2.5・全 on**（基準点は各腕の 89 コマ: `u_s_1080_b89`・`r_s_1080_b89`・`h_s_1080_b89`）

| コマ数・トークン（線比） | `u_` fp8（対照） | `r_` REDGraft 混在 | `h_` int8 ConvRot |
|---|---|---|---|
| 161・42,840（95%） | — | +1,191.0・溢れ（1 回） | — |
| 153・40,800（91%） | +903.3／+622.5・**溢れ×2** | +558.2／+618.5・**溢れ×2** | +814.2／+695.0・**溢れ×2** |
| 145・38,760（86%） | +82.6／+113.6・**快適×2** | +58.0／−35.1・**快適×2** | +153.6／+140.3・**快適×2** |
| **2 回快適の上端 c** | **145** | **145** | **145** |
| fp8 との比較 | （基準） | **同じ** | **同じ** |

- 参考（別のファイル・別の日）: 前回の LTX 2.5 fp8（guillaume・`ltx-2.5-22b-distilled-transformer-fp8_e4m3fn`）は、145 コマ −35.9／−35.2 で快適×2、153 コマ +536.5／+536.6 で溢れ×2 でした。今夜の `u_` と同じ段です。
- 参考（§106）: REDGraft LTX 2.5 の Q6_K GGUF は、1920×1088 で 42,840 が既定構成でも全 on でも快適（全 on −14.7）、既定構成で 44,880 が最初の溢れでした。今夜の REDGraft safetensors（`r_`）は 42,840 が溢れ（+1,191.0・1 回）です。

**LTX 2.3・既定構成（診断）**（基準点は各腕の 89 コマ: `f_s_1080_b89_d`・`x_s_1080_b89_d`・`k_s_1080_b89_d`）

| コマ数・トークン（線比） | `f_` fp8mixed（対照） | `x_` silveroxides int8 | `k_` Kijai int8 ConvRot |
|---|---|---|---|
| 137・36,720（82%） | — | — | −37.9／−31.4・**快適×2** |
| 129・34,680（77%） | +726.3／+747.4・**溢れ×2** | +728.8／+748.5・**溢れ×2** | −41.6・快適（1 回） |
| 121・32,640（73%） | +108.3／+28.6・**快適×2** | +21.2／+46.8・**快適×2** | −8.3・快適（1 回） |
| **2 回快適の上端 c** | **121** | **121** | **137 以上**（上を測っていない） |
| fp8 との比較 | （基準） | **同じ** | **上**（2 段以上） |

- 参考（前回の診断）: 同じ `sulphur_distil_fp8mixed` の既定構成は、121 コマ −42.2／−24.1 で快適×2、129 コマ +483.3 で溢れ（1 回）でした。今夜の `f_` と同じ段です。
- `x_` と `f_` は、同じ点の差がほぼ同じ大きさでした（129 コマで +726〜+749）。

**LTX 2.3・全 on**（基準点は `j_s_1080_b89`）

| コマ数・トークン（線比） | `j_` w4a8 | 参考: 他の変換器の全 on（出典） |
|---|---|---|
| 169・44,880（100%・線上） | +27.1／+12.8・**快適×2** | Q6_K GGUF −49.1／−22.8 で快適×2（§113） |
| 161・42,840（95%） | −3.9・快適（1 回） | fp8mixed +3,864.8 で溢れ（1 回・前回） |
| 145・38,760（86%） | +19.0・快適（1 回） | — |
| **2 回快適の上端 c** | **169（線上）** | |

- 入口判定（G-C）: 全 on の捨てラン `j_warm` のコミット最大 67.09 GiB（59.8%）、基準点 `j_s_1080_b89` 78.44 GiB（70.0%）で、どちらも 88% 未満だったので全 on で測りました。
- **同じプロセスで fp8 の全 on は測っていません**（今夜の `f_` は既定構成）。「fp8 より上か」は構成違いの参考比較です。
- 梯子は線上（169 コマ）で止めているので、線より上は測っていません。

### 表2 — LTX 2.5 の幾何ごと（`r_` の 1280×768）

**全 on・基準点は `r_s_720_b169`（1280×768・169 コマ・21,120 トークン）**

| 幾何・コマ数 | トークン（線比） | 第2段階の窓の差 | 判定 |
|---|---|---:|---|
| 1280×768・337 | 41,280（92%） | +678.5 | 溢れ（1 回） |
| 1280×768・313 | 38,400（86%） | −34.0／−25.5 | **快適×2** |
| （対照）1920×1088・153 | 40,800（91%） | +558.2／+618.5 | 溢れ×2 |
| （対照）1920×1088・145 | 38,760（86%） | +58.0／−35.1 | 快適×2 |

- **単一値の判定（規則）**: 1080p の c（38,760）以下で最大の 720p 点＝313 コマ（38,400）が 2 回とも快適、c の 1 段上（40,800）以上で最小の 720p 点＝337 コマ（41,280）が溢れ。**2 条件とも成り立つので、`r_` は「表せる」**です。1280×768 の刻みは 960 トークン（潜在 1 コマ）で、40,800 以上の最小点が 337 コマです。
- **`h_` の 720p は走らせていません。** 規則（1080p の c が `u_` と違ったときだけ測る）によるもので、時間の関門によるものではありません。
- 溢れた点は第2段階が延びています。第2段階の窓の秒は、313 コマ 43.69／43.71 秒に対し、337 コマ 85.87 秒でした。
- 参考（前回・別の fp8 ファイル）: guillaume fp8 の 1280×768 は 313 コマ（38,400）が快適×2、329 コマ（40,320）が 2 回で割れ（+297.0／+305.7）でした。

### 表3 — 基準点（89 コマ）の torch の割当ピークと WDDM の専有ピーク

**ブロックの常駐量は、変換器ファイルのヘッダから数えた `transformer_blocks` の重さです**（`Docs/VERIFICATION_LOG.md` §121.11 の前提・`README.md` 第0節）。ほかの列は今夜の実測です。

| 腕・構成 | ブロック常駐量（ヘッダ実測） | torch 割当ピーク `peak_vram_mb` | torch 予約ピーク | WDDM 専有ピーク（第2段階の窓） | 共有 GPU メモリの底（第2段階の窓の中央値） |
|---|---:|---:|---:|---:|---:|
| `u_` 2.5 fp8・全 on | 17.32 GiB | 10,420 MB | 11,906 MB | 13,006.9 MB | 1,198.9 MB |
| `r_` 2.5 REDGraft 混在・全 on | int8 831 層＋w4a8 609 層（中間） | 10,087 MB | 11,554 MB | 12,297.2 MB | 1,163.1 MB |
| `h_` 2.5 int8 ConvRot・全 on | 17.35 GiB | 10,420 MB | 11,906 MB | 12,630.3 MB | 1,159.2 MB |
| `f_` 2.3 fp8mixed・既定 | 18.76 GiB | 10,259 MB | 12,318 MB | 13,034.5 MB | 2,186.3 MB |
| `x_` 2.3 silveroxides int8・既定 | 18.77 GiB | 10,261 MB | 12,318 MB | 13,018.8 MB | 2,186.3 MB |
| `k_` 2.3 Kijai int8 ConvRot・既定 | 17.35 GiB | 9,526 MB | 12,054 MB | 12,754.3 MB | 1,168.8 MB |
| `j_` 2.3 w4a8・**全 on** | 9.8 GiB | 8,442 MB | 10,210 MB | 10,916.6 MB | 639.4 MB |

**同じプロセスの fp8 との差**（同じベースモデル・同じ構成の対どうし）

| 対 | torch 割当 | torch 予約 | WDDM 専有 | 共有の底 |
|---|---:|---:|---:|---:|
| `h_` − `u_` | 0 MB | 0 MB | −376.6 MB | −39.7 MB |
| `r_` − `u_` | −333 MB | −352 MB | −709.7 MB | −35.8 MB |
| `x_` − `f_` | +2 MB | 0 MB | −15.7 MB | 0.0 MB |
| `k_` − `f_` | −733 MB | −264 MB | −280.2 MB | −1,017.5 MB |

- **観測事実**: LTX 2.5 では、fp8 と int8 ConvRot の torch の割当ピークが同じ値（10,420 MB）で、境界も同じ段でした。LTX 2.3 では、fp8mixed と silveroxides int8 の値がほぼ同じ（差 2 MB）で境界も同じ段、**Kijai だけが torch の割当で 733 MB、共有の底で 1,017.5 MB 低く、境界が 2 段以上上**でした。ブロック常駐量のヘッダ実測は、Kijai 17.35 GiB に対し fp8mixed 18.76 GiB・silveroxides 18.77 GiB です。
- 同じ差は重い点でも続いています。121 コマの torch の割当ピークは `f_` 11,960 MB・`x_` 11,962 MB・`k_` 11,226 MB（`k_` − `f_` ＝ −734 MB）、145 コマは `u_` 13,840 MB・`h_` 13,842 MB・`r_` 13,510 MB でした。
- 共有 GPU メモリのジョブ全体のピーク（第5節「速度と共有メモリ」の表）も、`f_`・`x_` が約 2,190〜2,970 MB、`k_` が約 1,164〜1,181 MB でした。
- **なぜ Kijai だけ下がったのか、なぜ REDGraft が Q6_K GGUF より下に出たのかは、この較正では切り分けていません。** ここに並べたのは観測値だけです。
- **境界の点では比べられません。** 境界のまわりでは、快適な点も溢れた点も専有ピークがカードの天井の近くにあります（2.5 の 145 コマ 15,756.5〜15,993.8 MB・153 コマ 15,727.8〜15,883.3 MB、2.3 の `f_`・`x_` の 121 コマ 15,769.5〜15,826.8 MB、`k_` の 137 コマ 15,552.1／15,558.3 MB、`j_` の 169 コマ 15,636.2／15,641.1 MB）。
- `j_` は全 on なので、既定構成の LTX 2.3 の 3 本とは同じ列で比べられません。

### 表4 — 点ごとのコミットの最大と上限比

Windows のコミット（`\Memory\Committed Bytes`）を 2 秒おきに `commit_log.csv` に残し、`commit_table.py` が各点の開始から終了までの最大を取りました（`drive2_*.log` の各点の `peak` と同じ値です）。上限は計測の間ずっと 112.11 GiB、警戒線 90% は約 100.9 GiB、停止線 95% は約 106.5 GiB です。**計測全体の最大は 94.25 GiB（84.1%・01:31:33・`h_s_1080_153_r2`）で、警戒線にも停止線にも一度も達していません。**

**腕ごとの幅**

| 腕・構成 | 捨てラン | 基準点と判定した点の最大の幅 | 腕の最大 |
|---|---|---|---|
| `u_` 2.5 fp8・全 on | 67.4% | 80.6〜83.8% | 93.94 GiB（83.8%） |
| `r_` 2.5 REDGraft 混在・全 on | 59.8% | 73.2〜78.0% | 87.45 GiB（78.0%） |
| `h_` 2.5 int8 ConvRot・全 on | 66.2% | 80.3〜84.1% | 94.25 GiB（84.1%） |
| `f_` 2.3 fp8mixed・既定 | 65.4% | 73.0〜73.6% | 82.55 GiB（73.6%） |
| `x_` 2.3 silveroxides int8・既定 | 65.2% | 72.8〜74.0% | 83.00 GiB（74.0%） |
| `j_` 2.3 w4a8・全 on | 59.8% | 70.0〜74.3% | 83.32 GiB（74.3%） |
| `k_` 2.3 Kijai int8 ConvRot・既定 | 62.1% | 73.4〜74.0% | 82.96 GiB（74.0%） |

**点ごとの値**（走らせた順）

| ラベル | 開始 | 判定 | コミットの最大（GiB） | 上限比 |
|---|---|---|---:|---:|
| `u_warm` | 09-27 23:37 | 捨てラン | 75.58 | 67.4% |
| `u_s_1080_b89` | 09-27 23:40 | 基準点 | 90.38 | 80.6% |
| `u_s_1080_145` | 09-27 23:44 | 快適 | 93.31 | 83.2% |
| `u_s_1080_153` | 09-27 23:49 | 溢れ | 93.92 | 83.8% |
| `u_s_1080_145_r2` | 09-27 23:56 | 快適 | 93.43 | 83.3% |
| `u_s_1080_153_r2` | 09-28 00:01 | 溢れ | 93.94 | 83.8% |
| `r_warm` | 09-28 00:08 | 捨てラン | 67.06 | 59.8% |
| `r_s_1080_b89` | 09-28 00:11 | 基準点 | 82.80 | 73.9% |
| `r_s_1080_153` | 09-28 00:16 | 溢れ | 86.77 | 77.4% |
| `r_s_1080_161` | 09-28 00:21 | 溢れ | 87.45 | 78.0% |
| `r_s_1080_145` | 09-28 00:29 | 快適 | 86.30 | 77.0% |
| `r_s_1080_145_r2` | 09-28 00:34 | 快適 | 86.31 | 77.0% |
| `r_s_1080_153_r2` | 09-28 00:39 | 溢れ | 86.90 | 77.5% |
| `r_s_720_b169` | 09-28 00:45 | 基準点 | 82.05 | 73.2% |
| `r_s_720_313` | 09-28 00:49 | 快適 | 86.22 | 76.9% |
| `r_s_720_313_r2` | 09-28 00:54 | 快適 | 86.26 | 76.9% |
| `r_s_720_337` | 09-28 00:59 | 溢れ | 87.15 | 77.7% |
| `h_warm` | 09-28 01:06 | 捨てラン | 74.26 | 66.2% |
| `h_s_1080_b89` | 09-28 01:10 | 基準点 | 90.04 | 80.3% |
| `h_s_1080_145` | 09-28 01:14 | 快適 | 93.37 | 83.3% |
| `h_s_1080_153` | 09-28 01:19 | 溢れ | 94.09 | 83.9% |
| `h_s_1080_145_r2` | 09-28 01:25 | 快適 | 93.57 | 83.5% |
| `h_s_1080_153_r2` | 09-28 01:30 | 溢れ | 94.25 | 84.1% |
| `f_warm_d` | 09-28 01:38 | 捨てラン | 73.29 | 65.4% |
| `f_s_1080_b89_d` | 09-28 01:41 | 基準点 | 81.79 | 73.0% |
| `f_s_1080_121_d` | 09-28 01:47 | 快適 | 82.39 | 73.5% |
| `f_s_1080_129_d` | 09-28 01:53 | 溢れ | 82.55 | 73.6% |
| `f_s_1080_121_d_r2` | 09-28 02:00 | 快適 | 82.46 | 73.6% |
| `f_s_1080_129_d_r2` | 09-28 02:06 | 溢れ | 82.48 | 73.6% |
| `x_warm_d` | 09-28 02:14 | 捨てラン | 73.11 | 65.2% |
| `x_s_1080_b89_d` | 09-28 02:18 | 基準点 | 81.59 | 72.8% |
| `x_s_1080_121_d` | 09-28 02:23 | 快適 | 82.14 | 73.3% |
| `x_s_1080_129_d` | 09-28 02:29 | 溢れ | 82.89 | 73.9% |
| `x_s_1080_121_d_r2` | 09-28 02:36 | 快適 | 82.83 | 73.9% |
| `x_s_1080_129_d_r2` | 09-28 02:42 | 溢れ | 83.00 | 74.0% |
| `j_warm` | 09-28 02:50 | 捨てラン | 67.09 | 59.8% |
| `j_s_1080_b89` | 09-28 02:53 | 基準点 | 78.44 | 70.0% |
| `j_s_1080_145` | 09-28 02:58 | 快適 | 81.31 | 72.5% |
| `j_s_1080_161` | 09-28 03:03 | 快適 | 82.57 | 73.7% |
| `j_s_1080_169` | 09-28 03:09 | 快適 | 83.11 | 74.1% |
| `j_s_1080_169_r2` | 09-28 03:14 | 快適 | 83.32 | 74.3% |
| `k_warm_d` | 09-28 03:21 | 捨てラン | 69.66 | 62.1% |
| `k_s_1080_b89_d` | 09-28 03:25 | 基準点 | 82.28 | 73.4% |
| `k_s_1080_121_d` | 09-28 03:30 | 快適 | 82.88 | 73.9% |
| `k_s_1080_129_d` | 09-28 03:36 | 快適 | 82.92 | 74.0% |
| `k_s_1080_137_d` | 09-28 03:43 | 快適 | 82.62 | 73.7% |
| `k_s_1080_137_d_r2` | 09-28 03:50 | 快適 | 82.96 | 74.0% |

### 表5 — 生成時間の比（§1-33 の Go／No-go の材料）

**取り方**: 両方の腕で基準点か「2 回とも快適」だった点だけで比べます（溢れた点は第2段階が延びるので混ぜません）。主に使う列は `run.json` の第2段階の窓の秒（`v3prime.windows.stage2.seconds`・WDDM の標本を取った窓の長さ）、併記するのは `metadata.json` の生成全体の秒（`generation_time_seconds`）です。2 回走らせた点は 1 回目／2 回目を並べ、比は 2 回の平均どうしで取りました。

**LTX 2.5・全 on — int8 ConvRot（`h_`）対 fp8（`u_`）**

| 点 | `u_` 窓の秒 | `h_` 窓の秒 | 窓の比 `h_`÷`u_` | `u_` 生成秒 | `h_` 生成秒 | 生成の比 |
|---|---:|---:|---:|---:|---:|---:|
| 89 コマ（基準点） | 25.17 | 26.00 | ×1.033 | 85.51 | 89.33 | ×1.045 |
| 145 コマ（快適×2） | 43.39／42.69 | 43.58／44.14 | ×1.019 | 134.02／134.04 | 137.85／138.33 | ×1.030 |

**LTX 2.3・既定構成 — silveroxides int8（`x_`・ConvRot なし）と Kijai int8 ConvRot（`k_`）対 fp8（`f_`）**

| 点 | `f_` 窓の秒 | `x_` 窓の秒 | 窓の比 `x_`÷`f_` | `f_` 生成秒 | `x_` 生成秒 | 生成の比 |
|---|---:|---:|---:|---:|---:|---:|
| 89 コマ（基準点） | 38.29 | 38.24 | ×0.999 | 167.84 | 171.88 | ×1.024 |
| 121 コマ（快適×2） | 57.98／57.92 | 57.98／57.93 | ×1.000 | 222.06／216.13 | 215.25／214.79 | ×0.981 |

| 点 | `f_` 窓の秒 | `k_` 窓の秒 | 窓の比 `k_`÷`f_` | `f_` 生成秒 | `k_` 生成秒 | 生成の比 |
|---|---:|---:|---:|---:|---:|---:|
| 89 コマ（基準点） | 38.29 | 39.07 | ×1.020 | 167.84 | 163.38 | ×0.973 |
| （参考）121 コマ（`k_` は快適 1 回） | 57.98／57.92 | 58.75 | ×1.014 | 222.06／216.13 | 216.25 | ×0.987 |

`k_` の 121 コマは 1 回しか走らせていないので、取り方の規則（2 回とも快適）を満たしません。参考として別の行に置きました。`k_` が 2 回とも快適だった 137 コマは、`f_` が溢れる側なので対になりません。

**参考 — REDGraft 混在（`r_`）対 fp8（`u_`）**（LTX 2.5・全 on）

| 点 | `u_` 窓の秒 | `r_` 窓の秒 | 窓の比 | `u_` 生成秒 | `r_` 生成秒 | 生成の比 |
|---|---:|---:|---:|---:|---:|---:|
| 89 コマ（基準点） | 25.17 | 26.89 | ×1.068 | 85.51 | 94.27 | ×1.102 |
| 145 コマ（快適×2） | 43.39／42.69 | 44.18／44.19 | ×1.027 | 134.02／134.04 | 142.47／142.91 | ×1.065 |

**揺れの見積もり**

- **今夜の 1 回目と 2 回目の差**（同じ点・同じ読み込みのまま）: 第2段階の窓の秒は `u_` 145 コマ 0.70 秒（1.6%）・`h_` 145 コマ 0.56 秒（1.3%）・`r_` 145 コマ 0.01 秒・`r_` 720p 313 コマ 0.02 秒・`f_` 121 コマ 0.06 秒・`x_` 121 コマ 0.05 秒・`j_` 169 コマ 0.01 秒・`k_` 137 コマ 0.02 秒。生成全体の秒は `f_` 121 コマの 5.93 秒（2.7%）が最大で、ほかは 0.02〜0.48 秒（0.4% 以下）でした。
- **前回の実測**: 読み直しをまたいでも、同じ点の所要（`manifest.jsonl` の実測）は一致していました。LTX 2.5 fp8 の 1280×768・313 コマは、05:30 の 1 回目と、変換器を読み直した後の 07:11 の `g_s_720_313_r2b` がどちらも 130.4 秒、1920×1088・145 コマは 1 回目と 2 回目がどちらも 130.3 秒です（`outputs/comfort-calib-2026-09-26/`）。
- **窓の秒には約 1 秒の刻みがあります**（WDDM の標本が約 1 秒おきのため）。89 コマ（窓 25〜39 秒）では 1 秒が 2.5〜4%、145 コマ（窓 約 43 秒）では約 2.3% にあたります。**`h_`÷`u_` の窓の比（×1.019〜×1.033）は、この刻みと同じくらいの大きさです。**
- 参考に、サーバーのログの 1 段あたりの秒×段数（第5節の `stage2秒`）で比べても同じ向きでした: `h_`÷`u_` は 89 コマ ×1.032・145 コマ ×1.019、`x_`÷`f_` は ×0.997・×1.000、`k_`÷`f_` は 89 コマ ×1.021（121 コマは参考 ×1.015）。
- LTX 2.3 の比は既定構成での値です。製品の線を定義している全 on での比は測っていません。
- `j_`（w4a8・全 on）には同じ構成の fp8 の対が無いので、この表には入れていません（生成秒は第5節「速度と共有メモリ」の表のとおり）。

---

## 7. 運用上の出来事

時刻と経過の正本は `progress.md` です。

1. **失敗・打ち切り・取り直しは 0 回でした。** 47 ランすべて `completed`、`transformer_used` は全点で要求と一致し、ほかの `*_used` の照合もすべて `ok` でした。`drive2.sh` の終了コード 3（行列が空かない）・4（コミット 95%）は一度も出ていません。
2. **時間の関門にはどれもかかっていません**（起点 23:33:11 からの経過で、`r_` 1080p の最後の点の終了 00:42 が 1:10、`h_` 1080p の終了 01:34 が 2:01、`j_` の終了 03:17 が 3:44）。`k_`（任意の P4）まで走らせ、全腕の計測を経過 4:21 で終えました。
3. **`j_` の入口判定は全 on 側に出ました**（59.8%・70.0%）。既定構成版の計画ファイル 4 本は使っていません。
4. **`k_` は梯子の上限（137 コマ）で止めました。** 121 コマと 129 コマで快適が続いたため規則どおり 137 コマへ上がり、137 コマの 2 回目だけ走らせて腕を終えました。145 コマ以上は測っていません。
5. **復元**: 03:54:46 に `restore-state --base-model LTX25`（REDGraft を読み直し）、続いて `--base-model` なしで LTX 2.3（`sulphur_distil_fp8mixed`）を読み直しました。最後の `state.json` は `active_base_model` が LTX23、LTX23 が `sulphur_distil_fp8mixed`、LTX25 が `redgraftLTX25Fast2K_ltx25RedgraftNSFW` で、`original_state.json` と一致しています（`restore_state_output.txt`）。なお §121.11 の安全弁 7 は開始時の LTX 2.5 の選択を `default` と書いていますが、実際の開始時の選択は REDGraft の safetensors でした。復元先は計画 §2（P0-5）どおり「開始時の状態」（`original_state.json`）にしています。
6. **較正台のコードは変えていません**（第2.2節）。本体のコードも dev `fc9a285` のままです。計測後の状態は `git_state_end.txt` に記録しました（2026-09-28 04:06・dev `fc9a285`・未追跡は保管用フォルダのみ）。
7. `progress.md` の 01:03 の行（`r_D` の終了）は、実施中は時刻が伏せ字で、`r_s_720_337` の機械出力が 1 行貼り付いていました。2026-09-28 朝に `run.json` の値のまま整形しました（`r_D` の終了時刻は `manifest.jsonl` で 01:03:02）。

---

## 8. ファイル一覧とジョブID

### 8.1 このキャンペーンが残したファイル

**一次記録の正本は本書（`outputs/comfort-calib-2026-09-27/RESULTS.md`）です。** 同じフォルダの中身は `README.md` 第8節の見取り図にあり、実施の後に `runs/`・`manifest.jsonl`・`commit_log.csv`・`drive2_*.log`・`load_*_output.txt`・`idle_*_output.txt`・`run_*_output.txt`・`restore_state_output.txt`・`original_state.json`・`git_state_start.txt`・`preflight_output_start.txt`・`calib/` と、集計の出力 `make_results_output.md`・`report_output.txt`・`commit_table.md`・`digest.md` が加わっています。

**このフォルダは git の追跡外です。** 読者が辿れるように、**本書・`README.md`・`progress.md`・`digest.md`（全 47 ランの一覧）・`commit_table.md`（点ごとのコミット）・`make_results_output.md`** を `Docs/Outputs-archive/comfort-calib-2026-09-27/` へ複写してあります。`runs/` の生データ・`manifest.jsonl`・`commit_log.csv`・`plans/`・`calib/`・ログ類・スクリプト・出力の動画は複写していません。

### 8.2 ジョブID（オーナーの削除判断用）

**47 ランぶんの出力が `Nz-Videomni/outputs/<ジョブID>/` に残っています。** 一覧は `runs/<ラベル>/run.json` の `job_id` から機械的に書き出したものです（走らせた順）。全点 `completed` で、失敗・待ち時間切れ・422 はありません。

| ラベル | ジョブID |
|---|---|
| `u_warm` | `0f11263d-20a5-4147-b276-2dba42b4edba` |
| `u_s_1080_b89` | `327d401f-f985-4599-9b6f-f74eac222f12` |
| `u_s_1080_145` | `1e02bc5a-1080-4594-81ae-5aad7628660d` |
| `u_s_1080_153` | `68672815-0345-404b-bf8c-cc743916da56` |
| `u_s_1080_145_r2` | `d339e948-0471-4eb7-906a-3be58ef84131` |
| `u_s_1080_153_r2` | `a6d14138-2d33-4515-a043-a9369e08d397` |
| `r_warm` | `286387b7-132e-4507-9ae1-6b6bdd42f7df` |
| `r_s_1080_b89` | `466d17f0-9f8c-44c8-9af1-a9c28d9afa27` |
| `r_s_1080_153` | `f680e8d7-119e-466a-bdca-fe403e03e485` |
| `r_s_1080_161` | `836fd83d-73a9-472c-8f7e-cd236ec73383` |
| `r_s_1080_145` | `fc19f7a0-b3dc-4e0e-bb59-d685e9b23320` |
| `r_s_1080_145_r2` | `fe66087a-5629-417a-b8d6-5619e7626b96` |
| `r_s_1080_153_r2` | `029734ec-c2f7-49df-8f85-86a43f66a076` |
| `r_s_720_b169` | `3bf967b8-7d1c-4dd0-8270-bbf7e0abc963` |
| `r_s_720_313` | `96a33a7f-ab07-479f-9e9f-54b274d01a12` |
| `r_s_720_313_r2` | `123425e5-1633-4732-bee5-2b20e0e49064` |
| `r_s_720_337` | `e95b3faa-95f6-424a-9d80-8358e25ea739` |
| `h_warm` | `9beec9a5-fefd-4687-8e2b-0ce2e0ded053` |
| `h_s_1080_b89` | `a031ab5d-4609-469d-9350-3284b911177d` |
| `h_s_1080_145` | `ceba44ce-112a-4278-8af8-2fac574c7f52` |
| `h_s_1080_153` | `7bcf128e-7a75-4b1a-83e7-7fe00fc0c94a` |
| `h_s_1080_145_r2` | `9b0b123c-bc04-4d33-af2e-28129c4d26e7` |
| `h_s_1080_153_r2` | `d94e33f8-3c18-4dbd-8654-3269b37b2e8f` |
| `f_warm_d` | `0e45ef8b-4879-4a31-9cab-c613ee9d7297` |
| `f_s_1080_b89_d` | `bcbfc36c-a24d-416a-a2c3-e4c7ef2a5c94` |
| `f_s_1080_121_d` | `8931e1db-642d-44ac-8dbb-eb175d609ea6` |
| `f_s_1080_129_d` | `d53bd7ce-d823-4745-b12a-eb64931b8f04` |
| `f_s_1080_121_d_r2` | `f6564064-9ce5-44b7-ae94-f527b8570136` |
| `f_s_1080_129_d_r2` | `053d34fa-adf4-493e-9f37-4ade05881b4c` |
| `x_warm_d` | `55aca97a-6ea2-4068-82c6-a60cca91dc2f` |
| `x_s_1080_b89_d` | `c3859f22-b6c6-4442-910c-2a47880c3a86` |
| `x_s_1080_121_d` | `55ca1897-a2b3-42c6-8a57-50c5f6fea139` |
| `x_s_1080_129_d` | `955b80c4-874b-44a0-afc2-466ad0263771` |
| `x_s_1080_121_d_r2` | `2db26622-88fd-45a6-a4f5-c91c15856592` |
| `x_s_1080_129_d_r2` | `d332b0a3-f5ff-434c-b536-9b6d89ee0279` |
| `j_warm` | `172c5c09-ab9b-43ca-b4c3-a460b3635eb2` |
| `j_s_1080_b89` | `203bfeae-cf93-4b4c-bfcd-5926158c5cef` |
| `j_s_1080_145` | `08702d81-e5e0-41f3-9350-97bb7af983f5` |
| `j_s_1080_161` | `8467cd0c-8787-4155-b06d-b94795fcc407` |
| `j_s_1080_169` | `635aaf1e-cbe1-4a74-9a2c-891aef1f36e0` |
| `j_s_1080_169_r2` | `51e114e0-05bc-4ae4-8754-b7df80c12d08` |
| `k_warm_d` | `c7d74076-e7b9-49dd-b51d-baa718fd23ed` |
| `k_s_1080_b89_d` | `8a21f501-83d1-490c-a875-f10344518f28` |
| `k_s_1080_121_d` | `7a3b588f-e6db-43bf-aaa3-160db17f4a36` |
| `k_s_1080_129_d` | `0a894881-a0ca-4f66-8354-35da251f8dc7` |
| `k_s_1080_137_d` | `1a212bfd-24fa-4787-9b4e-0a2989042415` |
| `k_s_1080_137_d_r2` | `26877fde-424b-4991-89a5-a3973e11d8ad` |

---

## 9. 留保

1. **測ったのは各種別 1 ファイルだけです。** 一般則ではなく、そのファイルでの実測です。別の配布元の int8・w4a8・fp8 や、同じ種別の別のファイルは測っていません。今夜の LTX 2.5 の fp8 対照（`u_`）は前回の fp8（guillaume）とは別のファイルです。
2. **LTX 2.3 の `f_`・`x_`・`k_` は既定構成での値です。** 製品の LTX 2.3 の線は全 on で定義されているので、これらは fp8 と比べるための診断値で、全 on の境界ではありません。表5 の LTX 2.3 の時間比も既定構成での値です。
3. **`k_` は梯子の上限（fp8 の c＋2 段＝137 コマ）で止めたので、真の境界は測っていません。** c は「137 コマ以上」としか言えません。
4. **`k_` の 121 コマと 129 コマは各 1 回だけです。** c（137 コマ・2 回快適）より下の段なので境界の判定には影響しませんが、2 回一致の確認はしていません。
5. **`j_` は、全 on で測れた唯一の LTX 2.3 の量子化 safetensors です。** 同じプロセスで全 on の fp8 を測っていないので、Q5 の「fp8 より上」は構成違いの参考比較です。また梯子は線上（169 コマ）で止めているので、線より上は測っていません。145 コマと 161 コマは各 1 回です。
6. **`h_` の 720p は規則により走らせていません。** Q3 の「単一値で表せるか」は `r_` についてだけ答えています。
7. **ConvRot（アダマール回転）の復元は生成を遅くするので、判定を甘くする向きがありえます。** `h_`・`k_` の境界が fp8 と同じか上に出たことには、この向きが混ざっている可能性があります（今夜の第2段階の窓の秒は、`h_` が `u_` の ×1.019〜×1.033、`k_` が `f_` の ×1.020 でした）。
8. `r_` の 1080p 161 コマと 720p 337 コマは 1 回だけです。どちらも c の 1 段上以上の溢れ側で、規則上は 1 回で足ります。
9. 原因（なぜ Kijai だけ上か、なぜ REDGraft の safetensors が Q6_K GGUF より下か）は断定していません。第6節 表3 に観測値を並べただけです。
10. 画像の出来・参照動画あり・48 fps・連結生成・896×1152 は測っていません。見たのはメモリと時間だけです。
11. **スコープは「測って結論」までです。** 配信値・UI・マニフェストへの反映は別途検討（台帳 §1-31）です。
