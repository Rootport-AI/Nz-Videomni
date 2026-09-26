> **このファイルは実機の `outputs/comfort-calib-2026-09-26/RESULTS.md`（git追跡外）のスナップショットである（2026-09-26複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# fp8 safetensors の transformer での快適上限の較正 — 一次記録（2026-09-26）

> **第1節から第4節は、条件・点の設計・実際に走らせた順番・判定の規約です。** 規約の正本は第2次（2026-09-04）の `README.md` 第5節〜第7節に、今回の適用条件と点の設計は同じフォルダの `README.md` に、計測の計画の正本は承認済みの計画書（§3-167 B-3・2026-09-26）にあります。
> **第5節以降が測定後の記録です。** 第5節の表は `make_results.py` の出力の逐語で、人が書き写した数字は1つもありません。第6節の表は、`runs/<ラベル>/run.json` と `commit_log.csv` から判定と比較に使う値だけを抜き出したものです。
> **計画と実際が違うところは、隠さずに第3節と第7節に書いてあります。**

---

## 1. 結論

### 1.1 何を測ったのか

**transformer（変換器）を fp8 safetensors（重みを 8 ビットの浮動小数点で持つ形式）にしたとき、快適上限の線（VRAM 溢れが起きない生成規模の目安。1 タイルあたりのトークン数）が GGUF のときと同じく成り立つかを実測しました。** fp8 を置くだけで使えるようにした改修は `Docs/PENDING_TASKS.md` §3-167 の B-1（LTX 2.3）と B-2（LTX 2.5）で、その記録の正本は `Docs/VERIFICATION_LOG.md` §117・§118 です。

| ベースモデル | 変換器の登録名 | 役割 | 線 |
|---|---|---|---|
| LTX 2.3 | `sulphur_distil_fp8mixed` | 主役 | 単発 44,880・連結 40,000（全 on） |
| LTX 2.3 | `Sulphur-2-base-distil-Q6_K` | 同じプロセスの対照 | 同上 |
| LTX 2.5 | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | 主役 | 単発・連結とも 44,880 |
| LTX 2.5 | `default`（公式 GGUF） | 同じプロセスの対照 | 同上 |

**スコープは「測って結論を出す」までです。** 配信値（`config.py` の `_default_comfort_budgets()`）は変更していません。Nz-Videomni のコードも 1 バイトも触っていません。**配信値への反映は別途検討で、判断はオーナーです。**

### 1.2 LTX 2.5 fp8 — 線が下がる（単一の低い値で表せる）

**計画で決めておいた結論の 3 つの形（①線は動かない ②線が下がる ③単一の線では表せない）のうち、② にあたります。** 4 つの幾何すべてで、線の 91〜100% の点が 2 回とも（または 1 回で大きく）VRAM 溢れになり、その下に 2 回とも快適な段があり、さらに下に溢れはありませんでした。逆行（下の段が溢れて上の段が快適）はありません。

- **2 回とも快適だった上端**は、4 つの幾何で 38,304〜39,424 トークン（現行の線 44,880 の 85〜88%）です。
- **2 回とも溢れた下端**は、3 つの幾何で 40,320〜40,832 トークンです（1280×768 の 40,320 は 2 回で割れたので境界に採っていません）。
- 幾何ごとの境界の差は、快適側の上端どうしで 1,120 トークンです。これは 1920×1088 の潜在 1 コマ分（2,040 トークン）より小さい差です。**したがって「解像度ごとに割れる」ではなく「単一の低い値」で表せる形です。** 2 回とも快適だった上端の最小は 38,304（896×1152・297 コマ）です。
- **同じプロセスの公式 GGUF は、同じ条件で線上（1920×1088・169 コマ）まで快適でした**（第6節 表2）。溢れは fp8 に固有です。

### 1.3 LTX 2.3 fp8 — 全 on は測り切れず。既定構成では線が大きく下がる

**全 on（線を定義している構成）では、境界を決め切れませんでした。** Windows のコミット（仮想メモリの予約）が停止線の 95% を 2 回超えたため、計画どおり全 on の計測をそこで打ち切ったからです（第7節）。それまでに取れた 2 点は、どちらも VRAM 溢れでした。

- 1216×704・窓 w46（38,456 トークン・連結の線 40,000 の 96%）: **+2,126.7 MB の溢れ**（1 回）。
- 1920×1088・161 コマ（42,840 トークン・単発の線の 95%）: **+3,864.8 MB の溢れ**（1 回）。同じ点は同じプロセスの Q6_K では快適（+1.2）でした。

**代わりに、サーバーの既定構成（sdpa・keep_resident なし・prune_vaed なし）で 1920×1088 を降りて境界を測りました。** 線の定義（全 on）とは別の構成なので、**診断**として別枠に置きます。

- 既定構成の境界は **121 コマ（32,640 トークン）が 2 回とも快適、129 コマ（34,680）が溢れ**です。現行の線 44,880 の 73〜77% にあたります。
- 全 on の 161 コマの溢れ（+3,864.8）は、既定構成の同じ点（+3,529.5）とほぼ同じ形でした。既定構成ではコミットは上限の 85.6% にとどまっていたので、**2.3 fp8 の溢れはコミットの逼迫の巻き添えではなく、fp8 の transformer そのものによるもの**と読めます（専有 VRAM が 16 GB の天井に張り付いています）。
- **全 on の境界は、既定構成の境界と同じか、それより下にあると読んでいます**（推定です。全 on は速いぶん共有 GPU メモリが上がりやすい側だからで、`Docs/COMFORT_LIMIT_TABLE.md` 第10.2節の観察と同じ向きです）。

形としては ② の側（線が下がる）ですが、**全 on での境界の値は確定していません。**

### 1.4 スコープ外（測っていないこと）

- **LTX 2.3 fp8 の全 on で、計画した点のほとんど**（1920×1088 の 169・177 コマ、1280×768 と 768×1280 の全点、連結 [25,161] の全点、`f_w46` の 2 回目、Tier C の w43・w40）。理由は第7節のコミットの停止線です。
- LTX 2.3 fp8 の既定構成で測ったのは 1920×1088 だけです。
- 各変換器とも、測ったのは 1 つのファイルだけです。別の fp8 の配布物や、別の量子化は測っていません。
- 参照動画あり・48 fps・撮り直し（Retake）・画像の出来は測っていません。見たのはメモリと時間だけです。

---

## 2. 環境と記録

### 2.1 サーバー

| 項目 | 値 |
|---|---|
| サーバー | 監督役が B-2 の実機確認のために 2026-09-26 0 時 2 分に起動したもの（shell の PID 49264・ポート 18620）。オーナーが起動したサーバーではないので `server start --pid 49264` で登録した |
| 動いているコード | dev `755f5ff`（B-2 の実装）。開始時の HEAD は `246d358` だが、`755f5ff..246d358` の差は文書 2 本だけ（`git_state_start.txt`・`git_state_end.txt`） |
| 開始時の選択 | `active_base_model` が LTX25、LTX23 は `sulphur_distil_fp8mixed`、LTX25 は `default`（`original_state.json`） |
| 着手前の行列 | `pending` 0・`running` 0（`status_at_p0.json`） |

### 2.2 較正台

**第6次（2026-09-25）の較正台のコードだけを複製し、変えたのは `README.md`（T15）と自己検査の節 H（T16）の 2 か所だけです。** `calib.py`・`mkplan.py`・`make_results.py`・`batch_table.py`・`vram_sampler2.ps1`・`analyse_selftest.py` は 1 バイトも変えていません（`harness_diff.txt`）。

**較正台の外に、監督役が駆動と監視のスクリプトを足しました**（コミットの監視 `commit_monitor.ps1`、1 点ずつ流す `drive2.sh` など。一覧は `README.md` 第6節）。較正台そのものの判定や記録には手を入れていません。

### 2.3 GPU を使う前に通した検査

| 検査 | 件数 | 結果 |
|---|---|---|
| `selftest\parser_selftest.py` | 231 件（節 A〜G の 72＋節 H の 159） | 失敗 0 |
| `selftest\analyse_selftest.py` | 43 件 | 失敗 0 |
| `calib.py preflight`（オフライン） | 着手前は 15 ファイル・35 点 | 失敗 0 |

記録は `selftest_output.txt` と `preflight_output.txt` です。**`preflight_output.txt` は、計測の終わり（07:44）に、実際に使った 1 点 1 ファイルの計画すべて（85 点）へ走らせ直した記録です**（`preflight.json` は最後の実行で上書きされる作りのため）。途中で足した計画ファイルの検算は、`preflight_perpoint_f_output.txt`・`preflight_perpoint_g_output.txt` と `progress.md` にあります。

### 2.4 測定の実績

| 項目 | 実測 |
|---|---|
| 走らせた点 | **57 ラン**。全点 `completed`・失敗 0・待ち時間切れ 0・422 は 0。**うち 1 ランは無効**（`g_s_720_313_r2`。要求は fp8 なのに `default` で走った。第7節） |
| 内訳 | 捨てラン 5（`f_warm`・`g_warm`・`g_warm2`・`a_warm`・`b_warm`）／LTX 2.3 fp8 全 on 4／LTX 2.3 fp8 既定構成 9／LTX 2.5 fp8 33（うち無効 1）／同じプロセスの対照 6（Q6_K 3・公式 GGUF 3） |
| 時間帯 | サーバーの登録 02:06 から最終の復元 07:45 まで（5 時間 40 分）。計測点そのものの時間は第5節末尾の機械出力にあります |
| 冷却 | 点と点のあいだ 150 秒（1 点 1 ファイルの駆動が各点の前に入れた） |
| 待ち時間切れの上限 | 1 点あたり 3,600 秒 |
| 変換器の切り替え | 復元を除いて 8 回（最初の 2 回は `progress.md`、残り 6 回は `load_*_output.txt`）。どれも読み込みの応答と `GET /models` の照合が要求した名前と一致。切り替えのたびに待機時の床を 3 分取り直した（`idle_*_output.txt`・`calib/`） |
| 後片づけ | 制御・修正・追加診断の各段の後に 2 段階で復元（LTX 2.3 → LTX 2.5。07:03・07:14・07:29・07:45 の 4 回）。最後の `state.json` は開始時の控えと一致（`restore_state_output.txt`） |

**待機時の床は、v3′ の判定には使っていません**（v3′ は基準点の窓中央値との差だけを使うため）。取り直したのは、較正台が `calib/calibration.json` の無いときに `run` を拒む作りだからです。

---

## 3. 点の設計と、実際に走らせた順番

### 3.1 点の設計（要点）

**点の一覧（ラベル・幾何・トークン・基準点）は `README.md` 第3節が正本です。** 要点だけ書きます。

- トークン数は **(幅÷32)×(高さ÷32)×潜在フレーム数**です。単発の潜在フレーム数は (フレーム数−1)÷8＋1、連結は第 2 段（Stage-2）の窓の潜在フレーム数（`standard`＝22）です。
- ラベルの頭は、`f_` が LTX 2.3 fp8、`g_` が LTX 2.5 fp8、`a_` が LTX 2.3 Q6_K、`b_` が LTX 2.5 公式 GGUF です。`_s_` が単発、`_c_` が連結、`b<数>` が基準点、`_r2` が 2 回目、**`_d` が既定構成の診断点**です。
- **基準点は、同じ幾何・同じ変換器・同じ構成で取りました。** 単発は低いフレーム数の点、連結 [25,161] は 1280×768・窓 22、Tier D（クリップ 361×2 本）は同じ解像度の窓 22 です。既定構成の診断点には既定構成の基準点 `f_s_1080_b89_d` を別に取り、対照にもそれぞれ自前の基準点を取りました。
- 全点に共通: T2V・シード 12345・24 fps・指示文は較正台の既定（LTX 2.3 は `PROMPT_23`、LTX 2.5 は `PROMPT_25`）。構成は、診断点（`_d`）を除いて**全 on** です（中身は `README.md` 第1節）。

### 3.2 実際に走らせた順番

**計画は「計画ファイルを幾何ごとに流す」形でしたが、途中から「1 点 1 計画ファイル」を 1 点ずつ流す形に組み替えました**（第7節の 2）。以下、時刻は `progress.md` と `manifest.jsonl` によります。

1. **LTX 2.3 fp8（全 on）**: 02:06 読み込み・待機時の床 → 02:10 捨てラン `f_warm` → **Tier D**（`f_b1216` 02:12・`f_w46` 02:20）。`f_w46` でコミットが 95% を超えたので、`f_w46_r2` と以降の連続投入を中止。
2. **LTX 2.3 fp8 の Tier A（1 点ずつ）**: `f_s_1080_b89`（02:28）→ `f_s_1080_161`（02:32）。ここでもコミットが 95% を超えたので、**LTX 2.3 fp8 の全 on はここで打ち切り**。
3. **計画外の追加診断（既定構成）**: `f_s_1080_b89_d`（02:41）→ `f_s_1080_161_d`（02:46）。溢れがコミットの巻き添えかどうかを切り分けるため。
4. **LTX 2.5 fp8**: 02:54 読み込み・待機時の床 → 捨てラン `g_warm`（02:57）→ **Tier A** の 12 点（03:01〜04:11。1080p → 720p → 896×1152 → 連結）。内側の点はすべて溢れ。
5. **LTX 2.5 fp8 の Tier B（降りる段）**: 第 1 弾（1080p 153・145 コマ、720p 353・345・329 コマ。04:14〜04:40）→ 第 2 弾（896×1152 329・313・297 コマ、連結 1792×1024・1728×1024・1664×960。04:43〜05:11）→ 第 3 弾（1080p 145・153 コマの 2 回目）→ 第 4 弾（720p 329 コマの 2 回目・313 コマ）→ 第 5 弾（896×1152 297・313 コマの 2 回目）→ 第 6 弾（連結 1792×1024・1856×1024 の 2 回目。05:53 まで）→ 第 7 弾（720p 313 コマの 2 回目 `g_s_720_313_r2`。05:56。**無効**）。
6. **制御段階**: (1) 公式 GGUF の対照（`b_warm`・`b_s_1080_b89`・`_161`・`_169`。05:59〜06:13）→ (2) Q6_K の対照（`a_warm`・`a_s_1080_b89`・`_161`・`_169`。06:19〜06:35）→ (3) LTX 2.3 fp8 既定構成の降りる段（153・145・137 コマ。06:41〜07:02）→ (4) 復元（07:03）。
7. **修正段階**: LTX 2.5 fp8 を読み直し、捨てラン `g_warm2` → **`g_s_720_313_r2b`**（720p 313 コマの有効な 2 回目。07:11）→ 復元（07:14）。
8. **追加診断 X3**: 既定構成の 121・105 コマ（07:19〜07:28）→ 復元（07:29）。
9. **追加診断 X4**: 既定構成の 129 コマと 121 コマの 2 回目（07:34〜07:44）→ 最終の復元（07:45）。

### 3.3 計画から変えたところ

- **降りる段**: LTX 2.5 の 1080p は計画どおり 3 段（161 → 153 → 145 コマ）。720p は計画の 3 段（353・345・329 コマ）に **4 段目 313 コマ**を足しました（329 コマが快適側すれすれだったため）。896×1152 は計画どおり 3 段（329・313・297 コマ）。**連結は計画の 1984×1024・1920×1024（どちらも線の内側の上のほう）をやめ、1792×1024・1728×1024・1664×960 に差し替えました**（線の 91% の 1856×1024 で溢れたため）。
- **LTX 2.3 fp8 の既定構成の降りる段**（161 → 153 → 145 → 137 → 129 → 121 コマ、105 コマ）は、計画に無い診断です。
- **Tier Q（同じプロセスの対照）**は、計画では「基準点＋1 点」でしたが、「基準点＋161 コマ＋169 コマ」の 3 点ずつ走らせました（fp8 が溢れた 161 コマと、線上の 169 コマの両方を見るため）。
- **走らせなかった点**: LTX 2.3 fp8 の全 on の残り（第1.4節）です。LTX 2.3 の 768×1280 は、計画の「削る順番」の 1 番目としてもともと後回しにしていたものです。LTX 2.5 fp8 は Tier A の 12 点をすべて走らせました。

---

## 4. 判定の読み方

**規則 v3 です。同じ幾何・同じ変換器・同じ構成の基準点と比べて、Stage-2 の窓と復元（VAE でのデコード）の窓の両方で、共有 GPU メモリ（VRAM に入り切らない分をシステムメモリへ逃がす領域）の中央値が +300 メガバイト以上は上がらないことを「快適」とします。** 規則の正本は `Docs/COMFORT_LIMIT_TABLE.md` 第4.2節です。

- 境界は `runs/<ラベル>/run.json` の `v3prime.windows.stage2` と `v3prime.windows.restore` の `delta_mb` だけで決めています。10 標本未満の窓は判定不能です（今回、判定に使う 2 窓で判定不能になった点はありません）。
- **LTX 2.3 の点の `verdict_v3prime` は 5 つの窓の複合**（`s23_stage1`・`s23_gap_mid`・`s23_gap_tail` を足したもの）、**LTX 2.5 の点は 3 つの窓の複合**（`p25_30_decode_encode` を足したもの）です。**今回は、2 窓の判定と複合の判定が食い違ったランはありません。**
- **復元の窓は、有効な 42 点（基準点 9・捨てラン 5・無効 1 を除く）のすべてで +300 未満でした**（最大は `f_s_1080_153_d` の +145.0）。溢れはすべて Stage-2 の窓で起きています。
- **2 回一致の規則**: 境界の材料にするのは、2 回とも同じ判定になった点だけです。2 回で割れた点は境界に採らず、3 回目も走らせません。
- **専有の値は、Stage-2 の窓での WDDM の専有ピーク（グラフィックスカード全体・約 1 秒に 1 回の採取）です**（`vram.stage2_window.dedicated_peak_mb`）。「torch が確保した量」ではありません。参考に `metadata.json` の `peak_vram_mb`（torch の割当ピーク。今回は全点が参照動画なしなので、点どうしで比べられます）も第6節 表3 に並べます。
- **Stage-2 の速さは、同じ幾何の基準点や同じ点の別の変換器の `stage2.seconds` と並べて比べます。**
- **最初のジョブの所要は比べません。** fp8 の読み込みと keep_resident の写しが混ざるので、各読み込みの後に捨てランを置いています。

---

## 5. 結果表（測定後・`make_results.py` の機械生成）

**この節は `make_results.py` の出力（`make_results_output.md`）をそのまま取り込んだものです。** もとの数値は `runs\<ラベル>\run.json` と `manifest.jsonl` にあり、そこから機械が読み直して表にしています。

**読むときの注意**（機械が出す表そのものには書いていないこと）:

- 表の `v1`・`v2` 列は**古い判定規則の出力**です。今回の判定は `v3′` 列と、第6節で使う規則 v3（2 窓）です。
- **`g_s_720_313_r2` の行は無効です**（加速照合が `NG`。要求は fp8、実際は `default`）。判定の `clean` と差 −500.8 は公式 GGUF で走った値なので、fp8 の判定には使いません。有効な 2 回目は `g_s_720_313_r2b` です。
- 捨てラン（`*_warm`・`g_warm2`）は形の上では基準点（`baseline`）ですが、どの点もこれを基準に参照していません。
- 連結の「幾何」列のフレーム数（186f）はクリップ 25＋161 の合計、Tier D の 722f は 361＋361 の合計です。
- `keep_resident_embeddings` が LTX 2.3 の点で *null* なのは仕様です（表の下の注記のとおり）。既定構成の診断点（`_d`）の `attention` が `sdpa`・`keep_resident` と `vae_mode` が `off` なのは、サーバー既定で走らせたためで、照合は「既定構成を宣言した点が既定構成で走った」ことを確かめています。
- fused カーネル（`fused_gguf_dequant_kernel`）は fp8 の transformer には効きませんが、記録上は `on` と出ます。

---

<!-- ここから `make_results_output.md` の逐語 -->
### 全計測点

| ラベル | 系統 | 幾何 | トークン | 状態 | v1 | v2 | **v3′** | 加速照合 |
|---|---|---|---:|---|---|---|---|---|
| `f_warm` | LTX23 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `f_b1216` | LTX23 | 1216x704/722f | 18,392 | completed | organic | baseline | **baseline** | ok |
| `f_w46` | LTX23 | 1216x704/722f | 38,456 | completed | organic | organic | **plateau** | ok |
| `f_s_1080_b89` | LTX23 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `f_s_1080_161` | LTX23 | 1920x1088/161f | 42,840 | completed | harmful | harmful | **plateau** | ok |
| `f_s_1080_b89_d` | LTX23 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `f_s_1080_161_d` | LTX23 | 1920x1088/161f | 42,840 | completed | harmful | harmful | **plateau** | ok |
| `g_warm` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `g_s_1080_b89` | LTX25 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `g_s_1080_161` | LTX25 | 1920x1088/161f | 42,840 | completed | harmful | harmful | **plateau** | ok |
| `g_s_1080_169` | LTX25 | 1920x1088/169f | 44,880 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_b169` | LTX25 | 1280x768/169f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `g_s_720_361` | LTX25 | 1280x768/361f | 44,160 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_369` | LTX25 | 1280x768/369f | 45,120 | completed | harmful | harmful | **plateau** | ok |
| `g_s_896_b161` | LTX25 | 896x1152/161f | 21,168 | completed | organic | baseline | **baseline** | ok |
| `g_s_896_345` | LTX25 | 896x1152/345f | 44,352 | completed | harmful | harmful | **plateau** | ok |
| `g_s_896_353` | LTX25 | 896x1152/353f | 45,360 | completed | harmful | harmful | **plateau** | ok |
| `g_c_b1280x768` | LTX25 | 1280x768/186f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `g_c_1856x1024` | LTX25 | 1856x1024/186f | 40,832 | completed | harmful | harmful | **plateau** | ok |
| `g_c_1920x1088` | LTX25 | 1920x1088/186f | 44,880 | completed | harmful | harmful | **plateau** | ok |
| `g_s_1080_153` | LTX25 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `g_s_1080_145` | LTX25 | 1920x1088/145f | 38,760 | completed | harmful | organic | **clean** | ok |
| `g_s_720_353` | LTX25 | 1280x768/353f | 43,200 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_345` | LTX25 | 1280x768/345f | 42,240 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_329` | LTX25 | 1280x768/329f | 40,320 | completed | harmful | organic | **clean** | ok |
| `g_s_896_329` | LTX25 | 896x1152/329f | 42,336 | completed | harmful | harmful | **plateau** | ok |
| `g_s_896_313` | LTX25 | 896x1152/313f | 40,320 | completed | harmful | harmful | **plateau** | ok |
| `g_s_896_297` | LTX25 | 896x1152/297f | 38,304 | completed | organic | organic | **clean** | ok |
| `g_c_1792x1024` | LTX25 | 1792x1024/186f | 39,424 | completed | harmful | organic | **clean** | ok |
| `g_c_1728x1024` | LTX25 | 1728x1024/186f | 38,016 | completed | harmful | organic | **clean** | ok |
| `g_c_1664x960` | LTX25 | 1664x960/186f | 34,320 | completed | harmful | organic | **clean** | ok |
| `g_s_1080_145_r2` | LTX25 | 1920x1088/145f | 38,760 | completed | harmful | organic | **clean** | ok |
| `g_s_1080_153_r2` | LTX25 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_329_r2` | LTX25 | 1280x768/329f | 40,320 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_313` | LTX25 | 1280x768/313f | 38,400 | completed | harmful | organic | **clean** | ok |
| `g_s_896_297_r2` | LTX25 | 896x1152/297f | 38,304 | completed | organic | organic | **clean** | ok |
| `g_s_896_313_r2` | LTX25 | 896x1152/313f | 40,320 | completed | harmful | harmful | **plateau** | ok |
| `g_c_1792x1024_r2` | LTX25 | 1792x1024/186f | 39,424 | completed | harmful | organic | **clean** | ok |
| `g_c_1856x1024_r2` | LTX25 | 1856x1024/186f | 40,832 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_313_r2` | LTX25 | 1280x768/313f | 38,400 | completed | harmful | organic | **clean** | **NG {"transformer_used": {"requested": "ltx-2.5-22b-distilled-transformer-fp8_e4m3fn", "used": "default"}}** |
| `b_warm` | LTX25 | 768x512/121f | 6,144 | completed | clean | baseline | **baseline** | ok |
| `b_s_1080_b89` | LTX25 | 1920x1088/89f | 24,480 | completed | clean | baseline | **baseline** | ok |
| `b_s_1080_161` | LTX25 | 1920x1088/161f | 42,840 | completed | clean | clean | **clean** | ok |
| `b_s_1080_169` | LTX25 | 1920x1088/169f | 44,880 | completed | clean | clean | **clean** | ok |
| `a_warm` | LTX23 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `a_s_1080_b89` | LTX23 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `a_s_1080_161` | LTX23 | 1920x1088/161f | 42,840 | completed | harmful | organic | **clean** | ok |
| `a_s_1080_169` | LTX23 | 1920x1088/169f | 44,880 | completed | harmful | organic | **clean** | ok |
| `f_s_1080_153_d` | LTX23 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `f_s_1080_145_d` | LTX23 | 1920x1088/145f | 38,760 | completed | harmful | harmful | **plateau** | ok |
| `f_s_1080_137_d` | LTX23 | 1920x1088/137f | 36,720 | completed | inconclusive | harmful | **plateau** | ok |
| `g_warm2` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `g_s_720_313_r2b` | LTX25 | 1280x768/313f | 38,400 | completed | harmful | organic | **clean** | ok |
| `f_s_1080_121_d` | LTX23 | 1920x1088/121f | 32,640 | completed | inconclusive | organic | **clean** | ok |
| `f_s_1080_105_d` | LTX23 | 1920x1088/105f | 28,560 | completed | inconclusive | organic | **clean** | ok |
| `f_s_1080_129_d` | LTX23 | 1920x1088/129f | 34,680 | completed | organic | organic | **plateau** | ok |
| `f_s_1080_121_d_r2` | LTX23 | 1920x1088/121f | 32,640 | completed | inconclusive | organic | **clean** | ok |

### 加速が実際に効いたか（`metadata.json` の `*_used`）

| ラベル | transformer | attention | block_swap_prefetch | keep_resident | fused_gguf_dequant_kernel | vae_mode | keep_resident_embeddings | ok |
|---|---|---|---|---|---|---|---|---|
| `f_warm` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `f_b1216` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `f_w46` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `f_s_1080_b89` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `f_s_1080_161` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `f_s_1080_b89_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_161_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `g_warm` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_b89` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_161` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_169` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_b169` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_361` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_369` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_b161` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_345` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_353` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_b1280x768` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1856x1024` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1920x1088` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_153` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_145` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_353` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_345` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_329` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_329` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_313` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_297` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1792x1024` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1728x1024` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1664x960` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_145_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_153_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_329_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_313` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_297_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_313_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1792x1024_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1856x1024_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_313_r2` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | **NG** |
| `b_warm` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `b_s_1080_b89` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `b_s_1080_161` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `b_s_1080_169` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `a_warm` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_s_1080_b89` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_s_1080_161` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_s_1080_169` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `f_s_1080_153_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_145_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_137_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `g_warm2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_313_r2b` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `f_s_1080_121_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_105_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_129_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_121_d_r2` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |

`keep_resident_embeddings` が 2.3 の点で *null* なのは仕様です（`services/engines/ltx/adapter.py:481`。2.3 に埋め込み処理器が無い）。

### v3′ の窓ごとの判定（基準比・+300MB が線）

**`f_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 6 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 4 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 6 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 10 | × no baseline for this window | — | — |
| stage2 | 7 | × fewer than 10 WDDM samples | — | — |

**`f_b1216`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 64 | × no baseline for this window | — | — |
| s23_gap_mid | 10 | × no baseline for this window | — | — |
| s23_gap_tail | 64 | × no baseline for this window | — | — |
| s23_stage1 | 74 | × no baseline for this window | — | — |
| stage2 | 121 | × no baseline for this window | — | — |

**`f_w46`** — v3′=`plateau`（基準 `f_b1216`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 65 | ○ | -18.7 | 下 |
| s23_gap_mid | 43 | ○ | 872.5 | **超** |
| s23_gap_tail | 65 | ○ | -18.7 | 下 |
| s23_stage1 | 74 | ○ | 13.4 | 下 |
| stage2 | 220 | ○ | 2126.7 | **超** |

**`f_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 18 | × no baseline for this window | — | — |
| s23_gap_mid | 13 | × no baseline for this window | — | — |
| s23_gap_tail | 18 | × no baseline for this window | — | — |
| s23_stage1 | 23 | × no baseline for this window | — | — |
| stage2 | 25 | × no baseline for this window | — | — |

**`f_s_1080_161`** — v3′=`plateau`（基準 `f_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 36 | ○ | -14.2 | 下 |
| s23_gap_mid | 68 | ○ | 2538.1 | **超** |
| s23_gap_tail | 36 | ○ | -14.2 | 下 |
| s23_stage1 | 37 | ○ | -16.3 | 下 |
| stage2 | 277 | ○ | 3864.8 | **超** |

**`f_s_1080_b89_d`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 27 | × no baseline for this window | — | — |
| s23_gap_mid | 20 | × no baseline for this window | — | — |
| s23_gap_tail | 27 | × no baseline for this window | — | — |
| s23_stage1 | 26 | × no baseline for this window | — | — |
| stage2 | 39 | × no baseline for this window | — | — |

**`f_s_1080_161_d`** — v3′=`plateau`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 57 | ○ | -42.5 | 下 |
| s23_gap_mid | 86 | ○ | 2233.4 | **超** |
| s23_gap_tail | 57 | ○ | -42.5 | 下 |
| s23_stage1 | 47 | ○ | -3.2 | 下 |
| stage2 | 147 | ○ | 3529.5 | **超** |

**`g_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 6 | × fewer than 10 WDDM samples | — | — |

**`g_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 13 | × no baseline for this window | — | — |
| restore | 13 | × no baseline for this window | — | — |
| stage2 | 24 | × no baseline for this window | — | — |

**`g_s_1080_161`** — v3′=`plateau`（基準 `g_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -17.9 | 下 |
| restore | 24 | ○ | -17.9 | 下 |
| stage2 | 111 | ○ | 920.2 | **超** |

**`g_s_1080_169`** — v3′=`plateau`（基準 `g_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 25 | ○ | -8.9 | 下 |
| restore | 25 | ○ | -8.9 | 下 |
| stage2 | 145 | ○ | 1574.7 | **超** |

**`g_s_720_b169`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 12 | × no baseline for this window | — | — |
| restore | 12 | × no baseline for this window | — | — |
| stage2 | 21 | × no baseline for this window | — | — |

**`g_s_720_361`** — v3′=`plateau`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 26 | ○ | -31.6 | 下 |
| restore | 26 | ○ | -31.6 | 下 |
| stage2 | 139 | ○ | 1303.5 | **超** |

**`g_s_720_369`** — v3′=`plateau`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 26 | ○ | -13.8 | 下 |
| restore | 26 | ○ | -13.8 | 下 |
| stage2 | 145 | ○ | 1623.4 | **超** |

**`g_s_896_b161`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 11 | × no baseline for this window | — | — |
| restore | 11 | × no baseline for this window | — | — |
| stage2 | 21 | × no baseline for this window | — | — |

**`g_s_896_345`** — v3′=`plateau`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 25 | ○ | 0.1 | 下 |
| restore | 25 | ○ | 0.1 | 下 |
| stage2 | 141 | ○ | 1376.4 | **超** |

**`g_s_896_353`** — v3′=`plateau`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 26 | ○ | -26.7 | 下 |
| restore | 26 | ○ | -26.7 | 下 |
| stage2 | 146 | ○ | 1692.2 | **超** |

**`g_c_b1280x768`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 12 | × no baseline for this window | — | — |
| restore | 12 | × no baseline for this window | — | — |
| stage2 | 21 | × no baseline for this window | — | — |

**`g_c_1856x1024`** — v3′=`plateau`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -60.9 | 下 |
| restore | 24 | ○ | -60.9 | 下 |
| stage2 | 79 | ○ | 533.7 | **超** |

**`g_c_1920x1088`** — v3′=`plateau`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 26 | ○ | -60.8 | 下 |
| restore | 26 | ○ | -60.8 | 下 |
| stage2 | 145 | ○ | 1527.5 | **超** |

**`g_s_1080_153`** — v3′=`plateau`（基準 `g_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -35.6 | 下 |
| restore | 23 | ○ | -35.6 | 下 |
| stage2 | 78 | ○ | 536.5 | **超** |

**`g_s_1080_145`** — v3′=`clean`（基準 `g_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -9.1 | 下 |
| restore | 22 | ○ | -9.1 | 下 |
| stage2 | 42 | ○ | -35.9 | 下 |

**`g_s_720_353`** — v3′=`plateau`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -4.3 | 下 |
| restore | 24 | ○ | -4.3 | 下 |
| stage2 | 116 | ○ | 1001.4 | **超** |

**`g_s_720_345`** — v3′=`plateau`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -13.3 | 下 |
| restore | 24 | ○ | -13.3 | 下 |
| stage2 | 100 | ○ | 785.4 | **超** |

**`g_s_720_329`** — v3′=`clean`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -4.7 | 下 |
| restore | 23 | ○ | -4.7 | 下 |
| stage2 | 61 | ○ | 297.0 | 下 |

**`g_s_896_329`** — v3′=`plateau`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -26.4 | 下 |
| restore | 24 | ○ | -26.4 | 下 |
| stage2 | 101 | ○ | 776.6 | **超** |

**`g_s_896_313`** — v3′=`plateau`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -0.1 | 下 |
| restore | 23 | ○ | -0.1 | 下 |
| stage2 | 66 | ○ | 302.3 | **超** |

**`g_s_896_297`** — v3′=`clean`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 21 | ○ | -8.9 | 下 |
| restore | 21 | ○ | -8.9 | 下 |
| stage2 | 41 | ○ | -35.8 | 下 |

**`g_c_1792x1024`** — v3′=`clean`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -53.2 | 下 |
| restore | 24 | ○ | -53.2 | 下 |
| stage2 | 43 | ○ | 88.7 | 下 |

**`g_c_1728x1024`** — v3′=`clean`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -64.7 | 下 |
| restore | 22 | ○ | -64.7 | 下 |
| stage2 | 41 | ○ | -37.6 | 下 |

**`g_c_1664x960`** — v3′=`clean`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 20 | ○ | -78.2 | 下 |
| restore | 20 | ○ | -78.2 | 下 |
| stage2 | 36 | ○ | -21.4 | 下 |

**`g_s_1080_145_r2`** — v3′=`clean`（基準 `g_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -33.9 | 下 |
| restore | 22 | ○ | -33.9 | 下 |
| stage2 | 42 | ○ | -35.2 | 下 |

**`g_s_1080_153_r2`** — v3′=`plateau`（基準 `g_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -10.6 | 下 |
| restore | 23 | ○ | -10.6 | 下 |
| stage2 | 78 | ○ | 536.6 | **超** |

**`g_s_720_329_r2`** — v3′=`plateau`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -13.0 | 下 |
| restore | 23 | ○ | -13.0 | 下 |
| stage2 | 61 | ○ | 305.7 | **超** |

**`g_s_720_313`** — v3′=`clean`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -17.8 | 下 |
| restore | 22 | ○ | -17.8 | 下 |
| stage2 | 41 | ○ | -40.6 | 下 |

**`g_s_896_297_r2`** — v3′=`clean`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 21 | ○ | -3.3 | 下 |
| restore | 21 | ○ | -3.3 | 下 |
| stage2 | 41 | ○ | -35.4 | 下 |

**`g_s_896_313_r2`** — v3′=`plateau`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | 0.2 | 下 |
| restore | 23 | ○ | 0.2 | 下 |
| stage2 | 61 | ○ | 303.2 | **超** |

**`g_c_1792x1024_r2`** — v3′=`clean`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -78.5 | 下 |
| restore | 23 | ○ | -78.5 | 下 |
| stage2 | 44 | ○ | 40.0 | 下 |

**`g_c_1856x1024_r2`** — v3′=`plateau`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -78.2 | 下 |
| restore | 24 | ○ | -78.2 | 下 |
| stage2 | 78 | ○ | 512.5 | **超** |

**`g_s_720_313_r2`** — v3′=`clean`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -525.4 | 下 |
| restore | 22 | ○ | -525.4 | 下 |
| stage2 | 42 | ○ | -500.8 | 下 |

**`b_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 5 | × fewer than 10 WDDM samples | — | — |

**`b_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 13 | × no baseline for this window | — | — |
| restore | 13 | × no baseline for this window | — | — |
| stage2 | 24 | × no baseline for this window | — | — |

**`b_s_1080_161`** — v3′=`clean`（基準 `b_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -43.7 | 下 |
| restore | 24 | ○ | -43.7 | 下 |
| stage2 | 47 | ○ | -0.6 | 下 |

**`b_s_1080_169`** — v3′=`clean`（基準 `b_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 25 | ○ | -43.6 | 下 |
| restore | 25 | ○ | -43.6 | 下 |
| stage2 | 50 | ○ | -43.3 | 下 |

**`a_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 6 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 4 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 6 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 7 | × fewer than 10 WDDM samples | — | — |
| stage2 | 6 | × fewer than 10 WDDM samples | — | — |

**`a_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 18 | × no baseline for this window | — | — |
| s23_gap_mid | 12 | × no baseline for this window | — | — |
| s23_gap_tail | 18 | × no baseline for this window | — | — |
| s23_stage1 | 20 | × no baseline for this window | — | — |
| stage2 | 24 | × no baseline for this window | — | — |

**`a_s_1080_161`** — v3′=`clean`（基準 `a_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 35 | ○ | -17.3 | 下 |
| s23_gap_mid | 24 | ○ | 1.4 | 下 |
| s23_gap_tail | 35 | ○ | -17.3 | 下 |
| s23_stage1 | 35 | ○ | 1.0 | 下 |
| stage2 | 46 | ○ | 1.2 | 下 |

**`a_s_1080_169`** — v3′=`clean`（基準 `a_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 37 | ○ | 0.3 | 下 |
| s23_gap_mid | 26 | ○ | -18.5 | 下 |
| s23_gap_tail | 37 | ○ | 0.3 | 下 |
| s23_stage1 | 36 | ○ | -23.1 | 下 |
| stage2 | 48 | ○ | -23.8 | 下 |

**`f_s_1080_153_d`** — v3′=`plateau`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 54 | ○ | 145.0 | 下 |
| s23_gap_mid | 80 | ○ | 1664.5 | **超** |
| s23_gap_tail | 54 | ○ | 145.0 | 下 |
| s23_stage1 | 44 | ○ | 17.0 | 下 |
| stage2 | 156 | ○ | 2909.0 | **超** |

**`f_s_1080_145_d`** — v3′=`plateau`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 55 | ○ | -42.0 | 下 |
| s23_gap_mid | 59 | ○ | 798.0 | **超** |
| s23_gap_tail | 55 | ○ | -42.0 | 下 |
| s23_stage1 | 42 | ○ | -17.2 | 下 |
| stage2 | 116 | ○ | 2001.9 | **超** |

**`f_s_1080_137_d`** — v3′=`plateau`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 45 | ○ | -16.9 | 下 |
| s23_gap_mid | 37 | ○ | 206.0 | 下 |
| s23_gap_tail | 45 | ○ | -16.9 | 下 |
| s23_stage1 | 40 | ○ | 16.7 | 下 |
| stage2 | 72 | ○ | 1316.7 | **超** |

**`g_warm2`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 6 | × fewer than 10 WDDM samples | — | — |

**`g_s_720_313_r2b`** — v3′=`clean`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -14.1 | 下 |
| restore | 22 | ○ | -14.1 | 下 |
| stage2 | 41 | ○ | -41.4 | 下 |

**`f_s_1080_121_d`** — v3′=`clean`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 40 | ○ | -42.2 | 下 |
| s23_gap_mid | 31 | ○ | -3.2 | 下 |
| s23_gap_tail | 40 | ○ | -42.2 | 下 |
| s23_stage1 | 34 | ○ | -3.1 | 下 |
| stage2 | 58 | ○ | -42.2 | 下 |

**`f_s_1080_105_d`** — v3′=`clean`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 38 | ○ | -17.2 | 下 |
| s23_gap_mid | 26 | ○ | -17.5 | 下 |
| s23_gap_tail | 38 | ○ | -17.2 | 下 |
| s23_stage1 | 30 | ○ | 16.8 | 下 |
| stage2 | 48 | ○ | -15.8 | 下 |

**`f_s_1080_129_d`** — v3′=`plateau`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 42 | ○ | -42.4 | 下 |
| s23_gap_mid | 34 | ○ | -17.0 | 下 |
| s23_gap_tail | 42 | ○ | -42.4 | 下 |
| s23_stage1 | 38 | ○ | -0.2 | 下 |
| stage2 | 65 | ○ | 483.3 | **超** |

**`f_s_1080_121_d_r2`** — v3′=`clean`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 41 | ○ | -24.1 | 下 |
| s23_gap_mid | 31 | ○ | 21.0 | 下 |
| s23_gap_tail | 41 | ○ | -24.1 | 下 |
| s23_stage1 | 35 | ○ | 19.8 | 下 |
| stage2 | 58 | ○ | -24.1 | 下 |

### 速度と共有メモリ

| ラベル | 生成秒 | stage2秒 | 秒/トークン | 復元秒 | 予約ピークMB | 共有ピーク(job)MB | 参照符号化 予約MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `f_warm` | 68.1 | 9.9 | 0.001611 | 6.3 | 9,498 | 2196.0 | — |
| `f_b1216` | 280.0 | 130.2 | 0.007079 | 63.5 | 14,310 | 2212.0 | — |
| `f_w46` | 412.0 | 261.3 | 0.006795 | 64.8 | 17,700 | 4318.3 | — |
| `f_s_1080_b89` | 87.2 | 37.0 | 0.001513 | 18.6 | 12,412 | 2204.3 | — |
| `f_s_1080_161` | 428.4 | 415.9 | 0.009709 | 36.1 | 19,310 | 6078.7 | — |
| `f_s_1080_b89_d` | 169.0 | 57.8 | 0.002359 | 27.5 | 12,320 | 2202.2 | — |
| `f_s_1080_161_d` | 390.9 | 221.2 | 0.005165 | 57.0 | 18,952 | 5749.5 | — |
| `g_warm` | 53.8 | 9.0 | 0.001465 | 3.0 | 7,180 | 1694.5 | — |
| `g_s_1080_b89` | 81.5 | 36.8 | 0.001501 | 13.1 | 11,326 | 1508.0 | — |
| `g_s_1080_161` | 240.2 | 166.9 | 0.003897 | 24.1 | 16,414 | 2119.9 | — |
| `g_s_1080_169` | 293.6 | 217.3 | 0.004843 | 25.0 | 16,982 | 2756.4 | — |
| `g_s_720_b169` | 70.6 | 30.9 | 0.001463 | 11.3 | 10,414 | 1637.5 | — |
| `g_s_720_361` | 278.9 | 208.8 | 0.004728 | 26.0 | 16,774 | 2525.0 | — |
| `g_s_720_369` | 296.8 | 219.0 | 0.004854 | 26.4 | 17,048 | 2828.5 | — |
| `g_s_896_b161` | 71.4 | 31.4 | 0.001481 | 11.4 | 10,424 | 1657.7 | — |
| `g_s_896_345` | 283.2 | 212.1 | 0.004782 | 25.1 | 16,833 | 2567.1 | — |
| `g_s_896_353` | 297.1 | 219.9 | 0.004848 | 25.6 | 17,116 | 2908.9 | — |
| `g_c_b1280x768` | 79.5 | 30.9 | 0.001463 | 12.4 | 10,414 | 1672.4 | — |
| `g_c_1856x1024` | 195.4 | 117.8 | 0.002884 | 23.9 | 15,870 | 1744.4 | — |
| `g_c_1920x1088` | 301.4 | 217.3 | 0.004843 | 25.9 | 16,994 | 2756.5 | — |
| `g_s_1080_153` | 187.1 | 117.8 | 0.002886 | 23.2 | 15,856 | 1752.6 | — |
| `g_s_1080_145` | 129.1 | 63.1 | 0.001629 | 22.2 | 15,286 | 1639.7 | — |
| `g_s_720_353` | 248.3 | 174.6 | 0.004042 | 24.6 | 16,496 | 2209.1 | — |
| `g_s_720_345` | 223.1 | 150.6 | 0.003565 | 24.0 | 16,226 | 1976.8 | — |
| `g_s_720_329` | 161.3 | 91.7 | 0.002273 | 23.1 | 15,708 | 1667.2 | — |
| `g_s_896_329` | 224.2 | 151.5 | 0.003579 | 24.3 | 16,272 | 1975.9 | — |
| `g_s_896_313` | 167.3 | 98.0 | 0.002429 | 23.4 | 15,706 | 1667.0 | — |
| `g_s_896_297` | 127.5 | 62.0 | 0.001617 | 21.4 | 15,142 | 1640.0 | — |
| `g_c_1792x1024` | 139.7 | 64.5 | 0.001636 | 23.3 | 15,452 | 1689.4 | — |
| `g_c_1728x1024` | 135.1 | 61.2 | 0.001610 | 22.5 | 15,080 | 1483.0 | — |
| `g_c_1664x960` | 122.3 | 54.5 | 0.001587 | 20.6 | 14,224 | 1683.6 | — |
| `g_s_1080_145_r2` | 129.0 | 63.0 | 0.001625 | 22.2 | 15,297 | 1667.3 | — |
| `g_s_1080_153_r2` | 186.6 | 117.6 | 0.002882 | 23.2 | 15,846 | 1736.4 | — |
| `g_s_720_329_r2` | 161.0 | 91.7 | 0.002273 | 23.1 | 15,708 | 1658.4 | — |
| `g_s_720_313` | 129.5 | 62.1 | 0.001617 | 22.3 | 15,158 | 1667.1 | — |
| `g_s_896_297_r2` | 127.4 | 62.0 | 0.001617 | 21.4 | 15,142 | 1658.4 | — |
| `g_s_896_313_r2` | 161.1 | 91.7 | 0.002273 | 23.4 | 15,706 | 1694.9 | — |
| `g_c_1792x1024_r2` | 139.2 | 65.2 | 0.001655 | 23.2 | 15,452 | 1640.2 | — |
| `g_c_1856x1024_r2` | 195.9 | 117.8 | 0.002884 | 23.9 | 15,870 | 1752.5 | — |
| `g_s_720_313_r2` | 156.4 | 62.2 | 0.001621 | 22.5 | 13,060 | 1145.5 | — |
| `b_warm` | 26.8 | 8.7 | 0.001416 | 3.0 | 6,796 | 1155.3 | — |
| `b_s_1080_b89` | 80.2 | 36.6 | 0.001495 | 13.1 | 9,206 | 1139.7 | — |
| `b_s_1080_161` | 143.1 | 70.8 | 0.001653 | 24.0 | 14,290 | 1128.5 | — |
| `b_s_1080_169` | 150.2 | 75.2 | 0.001674 | 24.9 | 14,830 | 1096.4 | — |
| `a_warm` | 84.1 | 8.4 | 0.001367 | 6.1 | 9,064 | 1168.5 | — |
| `a_s_1080_b89` | 85.2 | 35.7 | 0.001458 | 18.1 | 11,428 | 1179.0 | — |
| `a_s_1080_161` | 149.1 | 69.3 | 0.001618 | 35.5 | 14,420 | 1180.2 | — |
| `a_s_1080_169` | 155.7 | 73.3 | 0.001634 | 36.3 | 15,004 | 1159.7 | — |
| `f_s_1080_153_d` | 373.8 | 234.4 | 0.005746 | 54.7 | 18,244 | 5112.4 | — |
| `f_s_1080_145_d` | 330.9 | 174.4 | 0.004501 | 54.2 | 17,478 | 4231.3 | — |
| `f_s_1080_137_d` | 249.7 | 108.0 | 0.002941 | 44.2 | 16,674 | 3541.9 | — |
| `g_warm2` | 52.2 | 8.8 | 0.001440 | 3.0 | 7,180 | 1675.2 | — |
| `g_s_720_313_r2b` | 129.5 | 62.1 | 0.001617 | 22.3 | 15,158 | 1657.4 | — |
| `f_s_1080_121_d` | 203.4 | 87.5 | 0.002679 | 40.4 | 15,220 | 2208.5 | — |
| `f_s_1080_105_d` | 192.1 | 72.0 | 0.002521 | 38.2 | 13,874 | 2204.8 | — |
| `f_s_1080_129_d` | 219.7 | 98.0 | 0.002824 | 41.8 | 15,968 | 2695.6 | — |
| `f_s_1080_121_d_r2` | 219.0 | 87.3 | 0.002675 | 41.0 | 15,190 | 2208.5 | — |

### 所要時間（`manifest.jsonl` の実測）

| ラベル | 開始 | 終了 | 秒 |
|---|---|---|---:|
| `f_warm` | 2026-09-26 02:10:05.863 | 2026-09-26 02:11:16.031 | 70.2 |
| `f_b1216` | 2026-09-26 02:12:46.045 | 2026-09-26 02:17:26.782 | 280.7 |
| `f_w46` | 2026-09-26 02:20:02.634 | 2026-09-26 02:26:55.559 | 412.9 |
| `f_s_1080_b89` | 2026-09-26 02:28:01.581 | 2026-09-26 02:29:29.765 | 88.2 |
| `f_s_1080_161` | 2026-09-26 02:32:04.786 | 2026-09-26 02:39:13.660 | 428.9 |
| `f_s_1080_b89_d` | 2026-09-26 02:41:14.691 | 2026-09-26 02:44:05.039 | 170.3 |
| `f_s_1080_161_d` | 2026-09-26 02:46:40.252 | 2026-09-26 02:53:11.235 | 391.0 |
| `g_warm` | 2026-09-26 02:57:46.677 | 2026-09-26 02:58:40.832 | 54.1 |
| `g_s_1080_b89` | 2026-09-26 03:01:15.899 | 2026-09-26 03:02:38.054 | 82.2 |
| `g_s_1080_161` | 2026-09-26 03:05:13.129 | 2026-09-26 03:09:13.809 | 240.7 |
| `g_s_1080_169` | 2026-09-26 03:11:49.563 | 2026-09-26 03:16:44.352 | 294.8 |
| `g_s_720_b169` | 2026-09-26 03:19:20.144 | 2026-09-26 03:20:32.343 | 72.2 |
| `g_s_720_361` | 2026-09-26 03:23:07.337 | 2026-09-26 03:27:48.030 | 280.7 |
| `g_s_720_369` | 2026-09-26 03:30:23.880 | 2026-09-26 03:35:22.662 | 298.8 |
| `g_s_896_b161` | 2026-09-26 03:37:58.482 | 2026-09-26 03:39:10.704 | 72.2 |
| `g_s_896_345` | 2026-09-26 03:41:45.660 | 2026-09-26 03:46:30.412 | 284.8 |
| `g_s_896_353` | 2026-09-26 03:49:06.180 | 2026-09-26 03:54:04.947 | 298.8 |
| `g_c_b1280x768` | 2026-09-26 03:56:40.793 | 2026-09-26 03:58:01.039 | 80.2 |
| `g_c_1856x1024` | 2026-09-26 04:00:35.997 | 2026-09-26 04:03:52.424 | 196.4 |
| `g_c_1920x1088` | 2026-09-26 04:06:27.460 | 2026-09-26 04:11:30.144 | 302.7 |
| `g_s_1080_153` | 2026-09-26 04:14:22.416 | 2026-09-26 04:17:30.848 | 188.4 |
| `g_s_1080_145` | 2026-09-26 04:20:05.803 | 2026-09-26 04:22:16.150 | 130.3 |
| `g_s_720_353` | 2026-09-26 04:24:51.058 | 2026-09-26 04:28:59.712 | 248.7 |
| `g_s_720_345` | 2026-09-26 04:31:35.503 | 2026-09-26 04:35:20.018 | 224.5 |
| `g_s_720_329` | 2026-09-26 04:37:54.929 | 2026-09-26 04:40:37.357 | 162.4 |
| `g_s_896_329` | 2026-09-26 04:43:24.845 | 2026-09-26 04:47:09.436 | 224.6 |
| `g_s_896_313` | 2026-09-26 04:49:45.288 | 2026-09-26 04:52:33.761 | 168.5 |
| `g_s_896_297` | 2026-09-26 04:55:09.600 | 2026-09-26 04:57:17.974 | 128.4 |
| `g_c_1792x1024` | 2026-09-26 04:59:52.896 | 2026-09-26 05:02:13.348 | 140.4 |
| `g_c_1728x1024` | 2026-09-26 05:04:48.164 | 2026-09-26 05:07:04.542 | 136.4 |
| `g_c_1664x960` | 2026-09-26 05:09:39.472 | 2026-09-26 05:11:43.796 | 124.3 |
| `g_s_1080_145_r2` | 2026-09-26 05:14:24.580 | 2026-09-26 05:16:34.910 | 130.3 |
| `g_s_1080_153_r2` | 2026-09-26 05:19:09.858 | 2026-09-26 05:22:18.416 | 188.6 |
| `g_s_720_329_r2` | 2026-09-26 05:25:01.284 | 2026-09-26 05:27:43.770 | 162.5 |
| `g_s_720_313` | 2026-09-26 05:30:19.583 | 2026-09-26 05:32:29.975 | 130.4 |
| `g_s_896_297_r2` | 2026-09-26 05:35:15.409 | 2026-09-26 05:37:23.712 | 128.3 |
| `g_s_896_313_r2` | 2026-09-26 05:39:58.647 | 2026-09-26 05:42:40.997 | 162.3 |
| `g_c_1792x1024_r2` | 2026-09-26 05:45:24.937 | 2026-09-26 05:47:45.367 | 140.4 |
| `g_c_1856x1024_r2` | 2026-09-26 05:50:20.188 | 2026-09-26 05:53:36.719 | 196.5 |
| `g_s_720_313_r2` | 2026-09-26 05:56:18.282 | 2026-09-26 05:58:56.695 | 158.4 |
| `b_warm` | 2026-09-26 05:59:23.195 | 2026-09-26 05:59:51.269 | 28.1 |
| `b_s_1080_b89` | 2026-09-26 06:02:26.283 | 2026-09-26 06:03:48.544 | 82.3 |
| `b_s_1080_161` | 2026-09-26 06:06:23.481 | 2026-09-26 06:08:47.877 | 144.4 |
| `b_s_1080_169` | 2026-09-26 06:11:22.900 | 2026-09-26 06:13:53.412 | 150.5 |
| `a_warm` | 2026-09-26 06:19:41.372 | 2026-09-26 06:21:07.673 | 86.3 |
| `a_s_1080_b89` | 2026-09-26 06:23:42.630 | 2026-09-26 06:25:08.828 | 86.2 |
| `a_s_1080_161` | 2026-09-26 06:27:43.950 | 2026-09-26 06:30:14.348 | 150.4 |
| `a_s_1080_169` | 2026-09-26 06:32:50.257 | 2026-09-26 06:35:26.684 | 156.4 |
| `f_s_1080_153_d` | 2026-09-26 06:41:12.764 | 2026-09-26 06:47:27.640 | 374.9 |
| `f_s_1080_145_d` | 2026-09-26 06:50:03.476 | 2026-09-26 06:55:36.234 | 332.8 |
| `f_s_1080_137_d` | 2026-09-26 06:58:11.254 | 2026-09-26 07:02:21.922 | 250.7 |
| `g_warm2` | 2026-09-26 07:07:52.010 | 2026-09-26 07:08:46.153 | 54.1 |
| `g_s_720_313_r2b` | 2026-09-26 07:11:21.197 | 2026-09-26 07:13:31.626 | 130.4 |
| `f_s_1080_121_d` | 2026-09-26 07:19:19.359 | 2026-09-26 07:22:43.877 | 204.5 |
| `f_s_1080_105_d` | 2026-09-26 07:25:19.778 | 2026-09-26 07:28:32.200 | 192.4 |
| `f_s_1080_129_d` | 2026-09-26 07:34:20.503 | 2026-09-26 07:38:01.080 | 220.6 |
| `f_s_1080_121_d_r2` | 2026-09-26 07:40:35.949 | 2026-09-26 07:44:16.589 | 220.6 |

- 計測点の実行時間の合計: **175.7 分**
- 最初の点の開始から最後の点の終了まで（冷却込み）: **334.2 分**
- 実行した点の数（再走を含む延べ）: **57**

<!-- `make_results_output.md` の逐語ここまで -->

---

## 6. 変換器ごとの判定と比較（表1〜表5）

**この節の表は、第5節の生の表と `run.json` から判定と比較に使う値だけを抜き出したものです。** 差はすべて**その点自身の基準点と比べた Stage-2 の窓の共有 GPU メモリ中央値の差**（メガバイト）で、快適と VRAM 溢れを分ける線は +300 です。2 回走らせた点は「1 回目／2 回目」の順に並べています。過去の記録から引いた値には出典（§113＝`outputs/comfort-calib-2026-09-17/`、§3-165＝`outputs/comfort-calib-2026-09-25/`、§106＝`outputs/comfort-calib-2026-09-14/`）を付けています。

### 表1 — LTX 2.3・1920×1088 の単発（線 44,880）

| コマ数・トークン（線比） | 変換器・構成（出典） | Stage-2 窓の差 | 判定 |
|---|---|---:|---|
| 185・48,960（109%） | Q6_K・全 on（§113） | +965.6／+1,049.7 | 溢れ×2 |
| 177・46,920（105%） | Q6_K・全 on（§113） | +201.7／+337.1 | 割れ |
| 177・46,920（105%） | Q4_K_M・全 on（§113） | +334.7 | 溢れ（1 回） |
| 169・44,880（100%） | Q6_K・全 on（§113） | −49.1／−22.8 | 快適×2 |
| 169・44,880（100%） | Q4_K_M・全 on（§113） | +919.0／−27.2 | 割れ |
| 169・44,880（100%） | Q6_K・全 on（今回・同じプロセス） | −23.8 | 快適（1 回） |
| 161・42,840（95%） | Q6_K・全 on（今回・同じプロセス） | +1.2 | 快適（1 回） |
| 161・42,840（95%） | **fp8・全 on** | **+3,864.8** | **溢れ（1 回）** |
| 161・42,840（95%） | fp8・既定構成（診断） | +3,529.5 | 溢れ（1 回） |
| 153・40,800（91%） | fp8・既定構成（診断） | +2,909.0 | 溢れ（1 回） |
| 145・38,760（86%） | fp8・既定構成（診断） | +2,001.9 | 溢れ（1 回） |
| 137・36,720（82%） | fp8・既定構成（診断） | +1,316.7 | 溢れ（1 回） |
| 129・34,680（77%） | fp8・既定構成（診断） | +483.3 | 溢れ（1 回） |
| 121・32,640（73%） | fp8・既定構成（診断） | −42.2／−24.1 | **快適×2** |
| 105・28,560（64%） | fp8・既定構成（診断） | −15.8 | 快適（1 回） |

- 基準点は、今回の fp8 全 on が `f_s_1080_b89`、fp8 既定構成が `f_s_1080_b89_d`、Q6_K が `a_s_1080_b89` です（いずれも 89 コマ）。
- **fp8 の既定構成は、降りるほど差が単調に小さくなり、129 コマと 121 コマのあいだで快適に転じます。** 逆行はありません。
- **同じ 161 コマでの Stage-2 の所要**: fp8 全 on 415.95 秒（基準点 37.05 秒）、fp8 既定構成 221.25 秒（同 57.75 秒）、Q6_K 69.3 秒（同 35.7 秒）。fp8 全 on の Stage-2 の窓は、PCIe 受信の中央値が 27,128.5 MB/s でした。
- **補助の窓 `s23_gap_mid`**（Stage-1 の終わりから Stage-2 の始まりまで）も、fp8 全 on の 161 コマで +2,538.1、既定構成の 161 コマで +2,233.4 でした。2 窓の判定と 5 窓の複合の判定は、全点で一致しています。
- **走っているときの共有 GPU メモリの底**（基準点の Stage-2 の窓の中央値）は、fp8 が 2,193.6〜2,202.7 MB、Q6_K が 1,177.6 MB で、fp8 のほうが約 1 GB 高い位置にあります。判定はどちらも自分の基準点との差で行っているので、この底の差は判定に入りません。
- **LTX 2.3 fp8 の全 on で、この幾何の境界は決まっていません**（第1.3節）。

### 表2 — LTX 2.5（線 44,880）

**fp8・全 on（今回）**

| 幾何・コマ数 | トークン（線比） | Stage-2 窓の差 | 判定 |
|---|---|---:|---|
| 1920×1088・169 | 44,880（100%） | +1,574.7 | 溢れ（1 回） |
| 1920×1088・161 | 42,840（95%） | +920.2 | 溢れ（1 回） |
| 1920×1088・153 | 40,800（91%） | +536.5／+536.6 | **溢れ×2** |
| 1920×1088・145 | 38,760（86%） | −35.9／−35.2 | **快適×2** |
| 1280×768・369 | 45,120（101%） | +1,623.4 | 溢れ（1 回） |
| 1280×768・361 | 44,160（98%） | +1,303.5 | 溢れ（1 回） |
| 1280×768・353 | 43,200（96%） | +1,001.4 | 溢れ（1 回） |
| 1280×768・345 | 42,240（94%） | +785.4 | 溢れ（1 回） |
| 1280×768・329 | 40,320（90%） | +297.0／+305.7 | 割れ |
| 1280×768・313 | 38,400（86%） | −40.6／−41.4 | **快適×2** |
| 896×1152・353 | 45,360（101%） | +1,692.2 | 溢れ（1 回） |
| 896×1152・345 | 44,352（99%） | +1,376.4 | 溢れ（1 回） |
| 896×1152・329 | 42,336（94%） | +776.6 | 溢れ（1 回） |
| 896×1152・313 | 40,320（90%） | +302.3／+303.2 | **溢れ×2** |
| 896×1152・297 | 38,304（85%） | −35.8／−35.4 | **快適×2** |
| 連結 1920×1088 | 44,880（100%） | +1,527.5 | 溢れ（1 回） |
| 連結 1856×1024 | 40,832（91%） | +533.7／+512.5 | **溢れ×2** |
| 連結 1792×1024 | 39,424（88%） | +88.7／+40.0 | **快適×2** |
| 連結 1728×1024 | 38,016（85%） | −37.6 | 快適（1 回） |
| 連結 1664×960 | 34,320（76%） | −21.4 | 快適（1 回） |

- 連結はクリップ 25＋161・窓 22（`standard`）です。基準点は、1920×1088 が `g_s_1080_b89`（89 コマ）、1280×768 が `g_s_720_b169`（169 コマ）、896×1152 が `g_s_896_b161`（161 コマ）、連結が `g_c_b1280x768` です。
- **1280×768・313 コマの 2 回目は `g_s_720_313_r2b` です。** 先に走った `g_s_720_313_r2` は `default` で走った無効ランなので、この表には入れていません（**無効（変換器不一致）**。第7節の 3）。
- **1280×768・329 コマと 896×1152・313 コマは、同じ 40,320 トークンでどちらもしきい +300 のすれすれ**（+297.0〜+305.7）でした。1280×768 は 2 回で割れ、896×1152 は 2 回とも溢れです。
- **溢れた点は Stage-2 の所要が延びています。** 1920×1088 では、快適な 145 コマが 63.15 秒、溢れた 153 コマが 117.75 秒、161 コマが 166.95 秒、169 コマが 217.35 秒です。

**同じプロセスの公式 GGUF（`default`・全 on）**

| 点 | Stage-2 窓の差 | 判定 | Stage-2 の所要（同じ点の fp8） |
|---|---:|---|---:|
| 1920×1088・161（42,840） | −0.6 | 快適（1 回） | 70.8 秒（166.95 秒） |
| 1920×1088・169（44,880） | −43.3 | 快適（1 回） | 75.15 秒（217.35 秒） |

基準点は自前の `b_s_1080_b89` です。**変えたのは載せた変換器のファイルだけで、同じサーバープロセス・同じシード・同じ加速設定です。**

**参考 1 — §3-165 の公式 GGUF（全 on・連結のクリップ 361×2 本。幾何が違うので直接は並べない）**: 1280×768・w46（44,160）が −49.5／−32.9 で快適×2、1152×640・w61（43,920）が −33.6／−33.2 で快適×2、1280×768・w52（49,920）が +1,551.3 で溢れでした。

**参考 2 — §106 の REDGraft LTX 2.5 Q6_K（構成を混ぜないよう分けて並べる）**

| 幾何 | 既定構成（快適側の最大／最初の溢れ） | 全 on の確認点 |
|---|---|---|
| 1280×768 | 44,160／45,120 | 44,160 は割れ（+304.1／+289.1）・45,120 は溢れ（+545.1） |
| 1920×1088 | 42,840／44,880 | 42,840 は快適（−14.7） |
| 896×1152 | 40,320／45,360 | 測っていない |
| 連結（窓 22） | 43,648／44,880 | 測っていない |

- **fp8 の境界は、どの幾何でも REDGraft Q6_K より低い位置にあります。** 1920×1088 でいえば、REDGraft Q6_K は 42,840 が既定構成でも全 on でも快適、fp8 は 40,800 が 2 回とも溢れです。
- §106 の全 on は、全 on のほうが基準比が上がる側（`Docs/COMFORT_LIMIT_TABLE.md` 第10.2節）なので、既定構成の値と同じ列には並べていません。

### 表3 — ブロックの重さの見立ての検証（専有ピーク）

**見立て**（`Docs/VERIFICATION_LOG.md` §117.8・計画書 §1）: fp8 の 1 ブロックは約 0.386 GB（Q4_K_M 254 MiB・Q6_K 303 MiB）。GPU に常駐させるブロックは既定 8 本なので、fp8 は **Q6_K より約 +0.5 GiB、Q4_K_M より約 +0.9 GiB** 増える見込みでした（9 本で換算すると +0.6／+1.0 GiB）。

**比べる値は 2 つです。** 「専有ピーク」は Stage-2 の窓での WDDM の専有ピーク（カード全体）、「torch の割当ピーク」は `metadata.json` の `peak_vram_mb` です。どちらも、天井（約 16 GB）に届いていない基準点で比べます。

**LTX 2.3・1920×1088・89 コマ（基準点）**

| 変換器・構成（出典） | 専有ピーク | torch の割当ピーク |
|---|---:|---:|
| Q4_K_M・全 on（§113） | 11,092.5 MB | 8,484 MB |
| Q6_K・全 on（§113） | 11,932.1 MB | 8,988 MB |
| Q6_K・全 on（今回・同じプロセス） | 11,925.8 MB | 8,988 MB |
| **fp8・全 on** | **12,889.9 MB** | **10,259 MB** |
| fp8・既定構成（診断） | 12,798.1 MB | 10,256 MB |

- **fp8 − Q6_K（同じプロセス）**: 専有 +964.1 MB、torch の割当 +1,271 MB（約 1.2 GiB）。**見立て（+0.5〜0.6 GiB）の約 2 倍です。**
- **fp8 − Q4_K_M（§113・別の日）**: 専有 +1,797.4 MB、torch の割当 +1,775 MB。見立て（+0.9〜1.0 GiB）の約 2 倍です。
- 参考: **Q6_K − Q4_K_M の torch の割当は +504 MB で、ブロックの重さからの見立て（8 本で約 +0.4 GiB）とほぼ合います。** 同じ見立て方が fp8 では合いません。
- **torch の割当の差は、点が重くなっても変わりません。** 161 コマでも fp8 全 on 14,083 MB・Q6_K 12,812 MB で差は +1,271 MB と、89 コマと同じでした。解像度や尺に比例する分ではなく、変換器に付いて回る一定の分です。

**LTX 2.3・1216×704・クリップ 361×2 本・窓 22（基準点）**

| 変換器（出典） | 専有ピーク | torch の割当ピーク |
|---|---:|---:|
| Q4_K_M（§3-165） | 9,804.5 MB | 8,442 MB |
| Q6_K（§3-165） | 11,257.0 MB | 8,442 MB |
| **fp8（今回）** | **12,407.2 MB** | **9,047 MB** |

- fp8 − Q6_K は専有 +1,150.2 MB、fp8 − Q4_K_M は専有 +2,602.7 MB です。
- この点の torch の割当ピークは、Q4_K_M と Q6_K で同じ 8,442 MB で、Stage-2 の重さを表していません（ジョブ全体のピークが別の場面で決まっているためと読んでいます）。

**LTX 2.5・1920×1088・89 コマ（基準点・同じプロセス）**

| 変換器 | 専有ピーク | torch の割当ピーク |
|---|---:|---:|
| 公式 GGUF（`default`） | 9,714.3 MB | 8,216 MB |
| **fp8** | **11,810.8 MB** | **10,419 MB** |

- fp8 − 公式 GGUF は、専有 +2,096.5 MB、torch の割当 +2,203 MB（約 2.1 GiB）です。161 コマ（+2,201 MB）・169 コマ（+2,202 MB）でも差は同じでした。LTX 2.5 には常駐ブロックの重さの見立てがありません（計画書 §9 の 9 で「差はさらに大きい可能性」とだけ書いていました）。

**境界の点では比べられません。** 境界のまわりでは、溢れた点もすぐ下の快適な点も、専有ピークがカードの天井の近くにあります（LTX 2.5 fp8 の溢れた点は 15,770.0〜15,968.7 MB、2 回とも快適だった上端は 15,626.9〜15,902.0 MB。LTX 2.3 fp8 は全 on の 161 コマが 15,904.1 MB、既定構成の 121 コマが 15,720.5 MB）。

**読み**: **fp8 で快適上限が下がったのは、変換器に付いて回る一定の VRAM の増え方（LTX 2.3 で Q6_K 比 約 +1.2 GiB、LTX 2.5 で公式 GGUF 比 約 +2.1 GiB）と向きが合っています。** ただし増え方の大きさは、常駐ブロックの重さの差からの見立ての約 2 倍で、**ブロックの重さだけでは説明できません。** 何が上乗せされているのか（§117.8 が「さらに増える」と書いた、bf16 のまま残るブロックが常駐の窓に入る時間帯など）は測っていません。なお、512×320（§117.8）や 768×512 の捨てランでは、ジョブ全体の torch の割当ピークが fp8 と GGUF で同じ値（LTX 2.3 で 8,442 MB）になり、差は表に出ません。

### 表4 — 1216×704・窓 w46（§3-165 の申し送り向け）

クリップ 361×2 本（つなぐと 705 フレーム）・38,456 トークン（連結の線 40,000 の 96%）・全 on です。

| 変換器（出典） | Stage-2 窓の差 | 専有ピーク | Stage-2 の所要（基準点） |
|---|---|---:|---:|
| Q4_K_M（§3-165） | +8.3・快適（1 回） | 13,838.8 MB | 127.05 秒（123.15） |
| Q6_K（§3-165） | +945.3／+914.8・**溢れ×2** | 15,484.1／15,506.7 MB | 255.75／253.65 秒（130.95） |
| **fp8（今回）** | **+2,126.7・溢れ（1 回）** | **16,059.7 MB** | **261.3 秒（130.2）** |

- **fp8 でも、この点（w46 の目安解像度そのもの）は VRAM 溢れになりました。** 差は Q6_K の 2 倍を超え、専有ピークはカードの天井に張り付いています。
- **fp8 は 1 回だけです。** 2 回目（`f_w46_r2`）は、1 回目でコミットが 95.6% に達したため保留しました（第7節の 1）。計画の Tier C（w43・w40 へ降りる段）も走らせていません。
- **この表は「重い変換器では、w46 の目安解像度が溢れる側に出る」材料です**（Q4_K_M は快適・Q6_K は 2 回とも溢れ・fp8 も溢れ）。§3-165 の申し送りの最終判断はオーナーです。

### 表5 — 点ごとのコミットの最大と上限比

Windows のコミット（`\Memory\Committed Bytes`）を 2 秒おきに `commit_log.csv` に残し、**各点の開始から終了まで（終了の 4 秒後まで）の最大**を取りました。上限（`\Memory\Commit Limit`）は計測の間ずっと 113.82 GiB で動きませんでした。警戒線は 90%（約 102.4 GiB）、停止線は 95%（約 108.1 GiB）です。

**変換器と構成ごとの幅**

| 変換器・構成 | 捨てラン | 計測点の最大の幅 |
|---|---|---|
| LTX 2.3 fp8・全 on | 79.5% | 91.0〜97.2%（停止線超え 2 点） |
| LTX 2.3 fp8・既定構成 | — | 69.1〜85.6% |
| LTX 2.5 fp8・全 on | 69.6%・69.7% | 82.6〜88.6% |
| LTX 2.3 Q6_K・全 on | 73.5% | 76.6〜79.6% |
| LTX 2.5 公式 GGUF・全 on | 58.0% | 62.3〜67.3% |

- **LTX 2.3 fp8・全 on は、捨てランの後の落ち着いた値が 81.15 GiB（71.3%）で、どの点でも一時的に 20 GiB 前後上がりました**（`progress.md` 02:10・02:29）。02:30 の時点で、ワーカーの python 1 本のページングされるメモリ（`PagedMemorySize`）が 67.67 GiB でした。
- **LTX 2.3 fp8 の既定構成は、同じ変換器でも 69.1〜85.6% と幅があります。** 既定構成は keep_resident（常駐の写し）を使わないぶん軽いのですが、どの時点で何が効いて上下したかは切り分けていません。
- **先に作った `commit_table.md` の `f_b1216`・`f_w46` の行（80.2%・80.4%）は、完了した後の値で、最大ではありません**（Tier D は連続駆動 `drive.sh` で流したため、その記録の形が違いました）。この表では `commit_log.csv` から取り直しています。

**点ごとの値**（走らせた順）

| ラベル | 判定 | コミットの最大（GiB） | 上限比 |
|---|---|---:|---:|
| `f_warm` | 捨てラン | 90.51 | 79.5% |
| `f_b1216` | 基準点 | 105.31 | 92.5% |
| `f_w46` | 溢れ | 108.85 | 95.6% |
| `f_s_1080_b89` | 基準点 | 103.53 | 91.0% |
| `f_s_1080_161` | 溢れ | 110.59 | 97.2% |
| `f_s_1080_b89_d` | 基準点 | 94.80 | 83.3% |
| `f_s_1080_161_d` | 溢れ | 97.44 | 85.6% |
| `g_warm` | 捨てラン | 79.20 | 69.6% |
| `g_s_1080_b89` | 基準点 | 94.91 | 83.4% |
| `g_s_1080_161` | 溢れ | 100.04 | 87.9% |
| `g_s_1080_169` | 溢れ | 100.59 | 88.4% |
| `g_s_720_b169` | 基準点 | 93.99 | 82.6% |
| `g_s_720_361` | 溢れ | 100.35 | 88.2% |
| `g_s_720_369` | 溢れ | 100.65 | 88.4% |
| `g_s_896_b161` | 基準点 | 94.04 | 82.6% |
| `g_s_896_345` | 溢れ | 100.51 | 88.3% |
| `g_s_896_353` | 溢れ | 100.88 | 88.6% |
| `g_c_b1280x768` | 基準点 | 94.23 | 82.8% |
| `g_c_1856x1024` | 溢れ | 99.67 | 87.6% |
| `g_c_1920x1088` | 溢れ | 100.83 | 88.6% |
| `g_s_1080_153` | 溢れ | 99.54 | 87.5% |
| `g_s_1080_145` | 快適 | 98.91 | 86.9% |
| `g_s_720_353` | 溢れ | 100.32 | 88.1% |
| `g_s_720_345` | 溢れ | 100.05 | 87.9% |
| `g_s_720_329` | 快適 | 99.55 | 87.5% |
| `g_s_896_329` | 溢れ | 100.12 | 88.0% |
| `g_s_896_313` | 溢れ | 99.59 | 87.5% |
| `g_s_896_297` | 快適 | 98.90 | 86.9% |
| `g_c_1792x1024` | 快適 | 99.39 | 87.3% |
| `g_c_1728x1024` | 快適 | 98.87 | 86.9% |
| `g_c_1664x960` | 快適 | 98.13 | 86.2% |
| `g_s_1080_145_r2` | 快適 | 99.09 | 87.1% |
| `g_s_1080_153_r2` | 溢れ | 99.79 | 87.7% |
| `g_s_720_329_r2` | 溢れ | 99.61 | 87.5% |
| `g_s_720_313` | 快適 | 98.98 | 87.0% |
| `g_s_896_297_r2` | 快適 | 99.42 | 87.3% |
| `g_s_896_313_r2` | 溢れ | 100.13 | 88.0% |
| `g_c_1792x1024_r2` | 快適 | 99.25 | 87.2% |
| `g_c_1856x1024_r2` | 溢れ | 99.78 | 87.7% |
| `g_s_720_313_r2` | 無効（変換器不一致） | 74.60 | 65.5% |
| `b_warm` | 捨てラン | 66.07 | 58.0% |
| `b_s_1080_b89` | 基準点 | 70.90 | 62.3% |
| `b_s_1080_161` | 快適 | 75.89 | 66.7% |
| `b_s_1080_169` | 快適 | 76.60 | 67.3% |
| `a_warm` | 捨てラン | 83.61 | 73.5% |
| `a_s_1080_b89` | 基準点 | 87.19 | 76.6% |
| `a_s_1080_161` | 快適 | 90.07 | 79.1% |
| `a_s_1080_169` | 快適 | 90.62 | 79.6% |
| `f_s_1080_153_d` | 溢れ | 81.49 | 71.6% |
| `f_s_1080_145_d` | 溢れ | 89.25 | 78.4% |
| `f_s_1080_137_d` | 溢れ | 88.50 | 77.8% |
| `g_warm2` | 捨てラン | 79.32 | 69.7% |
| `g_s_720_313_r2b` | 快適 | 98.80 | 86.8% |
| `f_s_1080_121_d` | 快適 | 78.61 | 69.1% |
| `f_s_1080_105_d` | 快適 | 87.90 | 77.2% |
| `f_s_1080_129_d` | 溢れ | 79.27 | 69.6% |
| `f_s_1080_121_d_r2` | 快適 | 87.88 | 77.2% |

`g_s_720_329` の「快適」は 1 回目の判定で、この点は 2 回目と割れています（表2）。

---

## 7. 運用上の出来事（計画と違ったところ）

時刻と経過の正本は `progress.md` です。

1. **コミットの停止線（95%）を 2 回超えました。** 1 回目は Tier D の `f_w46`（02:24 に 108.85 GiB・95.6%）、2 回目は `f_s_1080_161`（02:33 に 110.59 GiB・97.2%）です。どちらも計画 §5-9 どおり、**実行中のジョブは完走させ、次の点は投入しませんでした。** `DELETE /jobs/{id}` は取り消しの要求を受け付けるだけで、実行中の推論は止めないことを着手前に確かめてあったので、取り消しは使っていません。**LTX 2.3 fp8 の全 on は、2 回目の超過でここで打ち切り、代わりに既定構成の診断で境界を測りました**（第3.2節の 3・6・8・9）。この診断は計画に無い追加で、`README.md` 第3節の「追加診断」にあたります。**`f_w46` の 2 回目は保留のままです**（オーナーへの報告事項）。
2. **駆動を「1 点 1 計画ファイル」に組み替えました。** 最初は計画ファイルを連続駆動 `drive.sh` で順に流していましたが、その関門は「次のファイルを始める直前のコミット」を見る形でした（`drive_DA_f.log`）。コミットは点が終わると 80% 前後に戻るので、点の最中に 95% を超えても関門を素通りします。また、幾何ごとの計画ファイルの中の点は較正台が続けて投入するので、途中で止められません。そこで点ごとに計画ファイルを分け（`plans/p_*.json`）、**1 点ずつ流して、その点の間のコミットの最大を確かめ、95% 以上なら次を投入しない**駆動 `drive2.sh` に替えました。使わなかった幾何ごとの計画ファイルは `plans/unused/` に移してあります。**連続駆動 `drive.sh` を止めるとき、Git Bash に `pkill` が無かったので、PowerShell でプロセス番号を特定し `taskkill /T` で子ごと止めました。**
3. **05:56 の事故で、1 ランが無効になりました。** 05:28 に止めたはずの制御段階の待ち受け（第 6 弾の完了を待つもの）の子プロセス `phase_controls.sh` が生き残っていて、第 6 弾の完了（05:53:39）の 150 秒後に制御段階を始め、`load --base-model LTX25 --transformer default` を 05:56:17 に実行しました。同じ時刻（05:56:18）に第 7 弾の `g_s_720_313_r2` が投入されたため、**この点は `default`（公式 GGUF）で走りました。** `used_matches_request` が不一致（要求 fp8／実際 `default`）になったので**無効**とし、重複していた正規の待ち受けはプロセスの木ごと止め、生き残った 1 本だけを進めました。制御段階の後の修正段階で LTX 2.5 fp8 を読み直し、**`g_s_720_313_r2b` として fp8 で取り直しています**（快適・−41.4）。**教訓: 待ち受けのスクリプトを止めるときは、道具の停止だけでは子プロセスが残ることがあるので、`taskkill /T` で木ごと止めること。**
4. **降りる段を計画から変えました**（第3.3節）。連結の降りる段の差し替え、720p の 4 段目の追加、LTX 2.3 fp8 の既定構成の 7 段です。
5. **復元は各段の後に 2 段階で行いました**（LTX 2.3 → LTX 2.5 の順。07:03・07:14・07:29・07:45）。最後の復元の後の `state.json` は、`active_base_model` が LTX25、LTX23 の選択が `sulphur_distil_fp8mixed`、LTX25 の選択が `default` で、開始時の状態と同じです。
6. **所要**: 02:06 から 07:45 まで（5 時間 40 分）・57 ラン・失敗 0（無効 1）。
7. **較正台のコードは 1 バイトも変えていません**（`harness_diff.txt`。変えたのは `README.md` と自己検査の節 H だけ）。計画ファイルは `plans/p_*.json`（1 点 1 ファイル）と捨てランの `plans/plan_W_*.json`、見本の `plans/plan_B1_f_1080_169_r2.json`（使っていない）、Tier D の断片 `plans/fragments/` です。
8. **`preflight --plan` は 1 回に 1 ファイルしか受け付けません。** 1 点 1 ファイルに組み替えた後の検算は、ファイルごとに走らせています（`preflight_perpoint_*_output.txt`）。

---

## 8. ファイル一覧とジョブID

### 8.1 このキャンペーンが残したファイル

**一次記録の正本は本書（`outputs/comfort-calib-2026-09-26/RESULTS.md`）です。** 同じフォルダの中身は `README.md` 第6節の見取り図にあります。

**このフォルダは git の追跡外です。** 読者が辿れるように、**本書・`README.md`・`progress.md`・`make_results_output.md`・`report_output.txt`・`preflight_output.txt`・`git_state_start.txt`・`git_state_end.txt`・`restore_state_output.txt`・`harness_diff.txt`** と、`tables/` の代わりに **`digest.md`（全 57 ランの一覧）・`commit_table.md`（点ごとのコミット）** を `Docs/Outputs-archive/comfort-calib-2026-09-26/` へ複写してあります（複写の作法の正本は `Docs/Outputs-archive/README.md`）。**`commit_log.csv`・`runs/` の生データ・`manifest.jsonl`・`plans/`・`calib/`・`run_*_output.txt`・`idle_*_output.txt`・駆動と監視のスクリプト・較正台のスクリプトは複写していません**——実機側にしかありません。`commit_log.csv` の点ごとの最大は第6節 表5にあります。

**結論の要約は `Docs/COMFORT_LIMIT_TABLE.md` 第13節、較正そのものの記録は `Docs/VERIFICATION_LOG.md` §119 にあります。数値の正本は本書です。**

### 8.2 ジョブID（オーナーの削除判断用）

**57 ランぶんの出力が `Nz-Videomni/outputs/<ジョブID>/` に残っています。** 一覧は `runs/<ラベル>/run.json` の `job_id` から機械的に書き出したものです。全点 `completed` で、失敗・待ち時間切れ・422 はありません。

| ラベル | ジョブID |
|---|---|
| `a_s_1080_161` | `68cb404c-7659-40df-904f-8870484730d5` |
| `a_s_1080_169` | `4a439039-163c-448d-ac27-47a3d210fed9` |
| `a_s_1080_b89` | `deebc0ac-dd60-46b7-9f68-c94487106718` |
| `a_warm` | `ea02c0a8-be18-458d-95b4-daf9219419f7` |
| `b_s_1080_161` | `431a1744-528b-4f7c-bc2b-ed3423e4448e` |
| `b_s_1080_169` | `8f2d0b16-9711-49b3-a9ff-6aa71906d81b` |
| `b_s_1080_b89` | `08050388-a444-4f70-af5f-8d8ef50e915f` |
| `b_warm` | `bdba0fd4-5064-47cd-9692-851f79b7a15b` |
| `f_b1216` | `bfa6caaa-f3df-407a-a4fb-21ed349f6a3c` |
| `f_s_1080_105_d` | `d18b03e2-1dec-4591-b71c-57a7c385bb01` |
| `f_s_1080_121_d` | `e8c1aced-ecfe-48ee-b066-0031ee77c5b9` |
| `f_s_1080_121_d_r2` | `eb2bfe25-346e-4d87-8b4a-75ac05b875de` |
| `f_s_1080_129_d` | `20adfad6-3e50-44cc-9722-80e06ade6774` |
| `f_s_1080_137_d` | `92a571ca-2598-4935-986a-4bf6048fdefd` |
| `f_s_1080_145_d` | `ea85935b-aaa7-4e40-ba03-8120b2b1cfc5` |
| `f_s_1080_153_d` | `ee2a2e70-78c0-433b-a5ef-afd4de1d7ab6` |
| `f_s_1080_161` | `504ca4d8-a112-4274-a456-44713b153ef5` |
| `f_s_1080_161_d` | `620f91da-ca27-4abd-b0ec-674fffd41dd9` |
| `f_s_1080_b89` | `89d90c8e-7ab0-432b-b5a7-c7346dcf3880` |
| `f_s_1080_b89_d` | `e79e42dd-1761-4b0c-ae79-5f913b1ba969` |
| `f_w46` | `4dbd4ec0-4747-4f8c-9dce-8ce1ef4179b1` |
| `f_warm` | `e109aa63-4119-4ef3-b591-a07f85c53ab6` |
| `g_c_1664x960` | `11ab3b4c-6464-4513-8532-b4fcb64a422a` |
| `g_c_1728x1024` | `8e588e8b-8d25-49c9-91b6-5b90bcc9c5bc` |
| `g_c_1792x1024` | `31bc6f05-f0e4-4c51-a438-ba7d1483dfbf` |
| `g_c_1792x1024_r2` | `6ded5852-4d48-47e2-88b8-00e89b0d8986` |
| `g_c_1856x1024` | `d4de62d3-dd9a-4d1d-b72d-23225252e7b9` |
| `g_c_1856x1024_r2` | `b3919b10-a16e-4afa-8ff6-8a2f26101e54` |
| `g_c_1920x1088` | `2a130318-4d98-4ab3-abf2-1d7eaa36c48e` |
| `g_c_b1280x768` | `b2fa26c2-23f4-499a-bce5-6089414717d4` |
| `g_s_1080_145` | `27300192-bad4-4d62-83db-ff9d5469e5e2` |
| `g_s_1080_145_r2` | `54fa5967-c81b-4dc5-a498-c6e17e59d768` |
| `g_s_1080_153` | `252ef7c7-88ba-435c-84f9-6e5109998210` |
| `g_s_1080_153_r2` | `1e350cdf-ff2c-4914-a973-209e310edef9` |
| `g_s_1080_161` | `12db9a9f-ba3a-414a-90a5-09c2151b7cb2` |
| `g_s_1080_169` | `7336dc5c-612a-4a0c-bd89-58c089722a8f` |
| `g_s_1080_b89` | `282fc659-3823-4458-ab5c-6d035ad18b12` |
| `g_s_720_313` | `e36b3642-0b59-4014-8abe-2f02f61b184e` |
| `g_s_720_313_r2`（無効・変換器不一致） | `a1028f79-6fc1-46b7-be5b-c633f83e2902` |
| `g_s_720_313_r2b` | `8c11f0f6-5f48-4614-8962-3724809642d7` |
| `g_s_720_329` | `01a2f8cb-0c9c-4ecb-a5e5-7c0dccd1dff2` |
| `g_s_720_329_r2` | `5ca8a334-c6e6-4cca-810f-21d46b2727f7` |
| `g_s_720_345` | `bd4f9853-7145-4277-8e55-a09fd4656229` |
| `g_s_720_353` | `606c1519-b67b-40a6-90fe-49ed9f5f2e19` |
| `g_s_720_361` | `a02f0f49-8a44-42f0-9a5f-d5da09d1edc8` |
| `g_s_720_369` | `4a0e27db-35f2-4aa2-93c6-6859ed02042a` |
| `g_s_720_b169` | `b39fcdfe-acfd-4826-97df-8ee9cdfec114` |
| `g_s_896_297` | `4c08e9a2-1a3e-491b-826d-46ce14809369` |
| `g_s_896_297_r2` | `0fbbe78a-5abf-4fb3-9de2-e0eddad0fd44` |
| `g_s_896_313` | `2bc8279e-d68e-4244-98c9-7b558def00a3` |
| `g_s_896_313_r2` | `44816e94-54dc-43ea-824f-5612f5b661ab` |
| `g_s_896_329` | `c3031d53-df78-4bb8-8bcd-99d89b4084ed` |
| `g_s_896_345` | `3b2d5c58-afd7-4274-b1de-c2a23d31fcef` |
| `g_s_896_353` | `29cb5282-8a54-4a90-b632-612d28eee8f0` |
| `g_s_896_b161` | `17e5656c-f74c-431a-8dac-17a62a55d0ab` |
| `g_warm` | `ea79f378-d98e-45ba-8858-fba27f68b275` |
| `g_warm2` | `1d3c83a3-af9d-41b3-9e49-5bde68d578a6` |
