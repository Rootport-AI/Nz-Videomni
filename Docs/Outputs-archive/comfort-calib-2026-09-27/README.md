> **このファイルは実機の `outputs/comfort-calib-2026-09-27/README.md`（git追跡外）のスナップショットである（2026-09-28複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# 快適上限キャリブレーション計測台（§3-168 C-4・2026-09-27／重みの種別ごとの快適上限の較正・準備フェーズ）

> **この文書は準備フェーズ（2026-09-27）の記録です。GPU・サーバー・ジョブ投入はまだ行っていません。**
> やったのは、較正台の複製、Tier A の計画ファイル 33 本の作成、`selftest` 2 本と
> オフラインの `preflight`（`--online` なし）の実行だけです。実施（GPU を使う本番）は別セッションです。
>
> **手順の正本は `Docs/VERIFICATION_LOG.md` §121.11 です。** この README は実施のための写し（開発機のローカルのみ・git 追跡外）です。
> 以下の第 1〜7 節は実施のときに読みやすいようにここへ写したものであり、手順そのものの正本ではありません。
> 本文と §121.11 に食い違いがあれば **§121.11 が勝ちます**。
> 判定規約 v3′、`metadata.json` の注意、副命令の一覧、サーバーの起動のしかたなど計測台そのものの規約は
> `outputs/comfort-calib-2026-09-04/README.md` 第5節〜第7節が正本です（第2次からの言い伝え、第6次 README 冒頭より）。
>
> 第2次から第6次までのフォルダは読み取り専用の履歴です。1バイトも書き換えていません。

---

## 0. 前提（c4_procedure_v2.md §0 の写し）

1. 1 ランの周期（実行＋冷却 150 秒）: 2.5 全 on＝基準点 約 4 分・重い点 約 7 分／2.3 既定構成＝基準点 約 5.5 分・重い点 約 8.5 分／変換器の切り替え（load＋idle-calib 3 分＋捨てラン）約 8 分（§119・RESULTS 第5節）。
2. §13.1 の 2.5 fp8（guillaume）は実機に無い。2.5 の fp8 の基準は `ltx25_uncensored_v1.1-fp8_scaled`（fp8 1,440 層・connector も fp8・ブロック 17.32 GiB）で、このプロセスで新たに 2 回一致の境界を取る。
3. ブロック（transformer_blocks）の常駐量（ヘッダ実測）: Kijai 17.35 GiB／fp8mixed 18.76／silveroxides 18.77／2.5 int8_convrot 17.35／2.5 uncensored fp8 17.32／2.3 w4a8 9.8（Q4_K_M の約 11.9 より軽い）／REDGraft は int8 831 層＋w4a8 609 層で中間。Kijai と fp8mixed のブロック差は約 1.4 GiB（上限比 約 1.2 ポイント）。
4. LTX 2.3 の既定構成には製品の行が無い。2.3 の既定構成の値は診断値（fp8 との比較材料）。製品の 2.3 の線は全 on で定義。
5. 判定は規則 v3（stage2 窓と restore 窓・基準点比 +300 MB・10 標本未満は判定不能）。境界の材料は 2 回一致した点だけ。

## 1. 問い（c4_procedure_v2.md §1 の写し）

| # | 問い | 答えの形 |
|---|---|---|
| Q1 | REDGraft 混在（2.5）の線は fp8（u_）より上か。Q6_K GGUF（1080p 42,840 快適・44,880 溢れ）まで届くか | 同じ／間／線まで成り立つ |
| Q2 | 2.5 int8_convrot（公式複製）の線は fp8（u_）と同じか | 同じ／上／下 |
| Q3 | 2.5 の 2 種（REDGraft・int8_convrot）は単一値で表せるか | 表せる／幾何で割れる |
| Q4 | 2.3 int8（silveroxides＝fp8mixed とブロックが同じ重さ・同じ 1,232 層）は既定構成で fp8（f_）と同じか | 同じ／違う |
| Q5 | 2.3 w4a8 は fp8 より上か。全 on で測れるか | 全 on 入口判定（G-C）→ 全 on の線 または 既定構成の診断値 |
| （任意）Q4' | 2.3 Kijai（ConvRot・transformer のみ）は silveroxides と同じか | 時間が残れば |

判定規則（計測前に固定）: 刻み＝1080p の潜在 1 コマ＝2,040 トークン。比較は「2 回快適の上端 c」だけで行う。「同じ」＝int8 の c が同一プロセスの fp8 の c と同じ段。1 段以上ずれたら「上／下」。逆行は両点を `_r2` で取り直し、それでも残れば「非単調」と記録してその腕を止める。単一値で表せるかは各幾何の c で判定する。詳細は手順書 v2 §1。

## 2. 複製元（第7次・2026-09-26）との差分

**差分そのものは `harness_diff.txt` に保存してあります。** 要旨:

- `calib.py`・`mkplan.py`・`make_results.py`・`batch_table.py`・`vram_sampler2.ps1`・`selftest/analyse_selftest.py`・`selftest/fake_runs/`・`selftest/plans_fixture/`・`commit_monitor.ps1` は **1 バイトも変えていません**。
- `drive2.sh`: cd 先の 2026-09-27 フォルダへの変更と、`commit_log.csv` のパスの 2 行。CSV の置き場所は、前回の較正台がそのセッションのスクラッチパッド下に置いていたのを、**このフォルダ自身**（`outputs/comfort-calib-2026-09-27/commit_log.csv`）に変えました。準備（今回のセッション）と実施（別セッション、手順書 v2 §7 の「2 日目」）が分かれるため、セッション固有のスクラッチパッドに置くと実施のときには無くなっているからです。`outputs/` は git 管理外なので、フォルダ自身に置いても問題ありません。
- `commit_table.py`・`digest.py`: パス定数 `D = r"...outputs/comfort-calib-2026-09-27"` の 1 行ずつ。
- `README.md`: この文書（毎回書き直す。第7次の `harness_diff.txt` でも同じ扱い）。
- `selftest/parser_selftest.py`: **節 H と、冒頭の 1 文だけ**書き直しました。節 A〜G は 1 文字も変えていません。節 H の変更点は同ファイル内の節 H 冒頭のコメントに書いてあります（要旨: 7 本の腕・連結なし・旧 H(5) のジョブ ID 照合を削除し T5a と `used_check` に一本化）。
- 複製していないもの（手順書 v2 §5）: `original_state.json`・`calib/`・`plans/`（中身。フォルダ自体は再作成）・`runs/`・`manifest.jsonl`・`preflight.json`・`luid.json`・`server.pid`・ログ類・`drive.sh`・`phase_controls.sh` など、前回の実行に固有のファイル一式。

## 3. Tier A の計画ファイル（今回作ったもの・33 本・1 点 1 ファイル）

トークン数は (幅÷32)×(高さ÷32)×潜在フレーム数、潜在フレーム数は単発で (フレーム数−1)÷8+1。線比は 1080p/720p とも 2.5 全 on の線 44,880 との比（2.3 既定構成の点は製品の線が無いため診断値としての比。前提4）。

各腕とも「捨てラン W（768×512×121f・6,144 トークン・判定しない）→ 基準点（1080p は 89f、r_ の 720p は 169f）→ Tier A の 2 点」の構成です（手順書 v2 §2）。j_ だけ、全 on 版と既定版の両方を作ってあります（入口判定 G-C で片方だけ使う）。

| ラベルの頭 | 腕 | 変換器（登録名） | 構成 | ファイル |
|---|---|---|---|---|
| `u_` | LTX 2.5 uncensored fp8 | `ltx25_uncensored_v1.1-fp8_scaled` | 全 on | `p_u_warm.json`・`p_u_s_1080_b89.json`・`p_u_s_1080_145.json`・`p_u_s_1080_153.json` |
| `r_` | REDGraft LTX 2.5 混在(int8+w4a8) | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` **（`.part`・未ダウンロード）** | 全 on | `p_r_warm.json`・`p_r_s_1080_b89.json`・`p_r_s_1080_153.json`・`p_r_s_1080_161.json`・`p_r_s_720_b169.json` |
| `h_` | LTX 2.5 int8_convrot（公式複製） | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot` **（`.part`・未ダウンロード）** | 全 on | `p_h_warm.json`・`p_h_s_1080_b89.json`・`p_h_s_1080_145.json`・`p_h_s_1080_153.json` |
| `f_` | LTX 2.3 fp8mixed | `sulphur_distil_fp8mixed` | 既定（`_d`） | `p_f_warm_d.json`・`p_f_s_1080_b89_d.json`・`p_f_s_1080_121_d.json`・`p_f_s_1080_129_d.json` |
| `x_` | LTX 2.3 silveroxides int8 | `ltx-2.3-22b-distilled-1.1_int8mixedtensorwise` | 既定（`_d`） | `p_x_warm_d.json`・`p_x_s_1080_b89_d.json`・`p_x_s_1080_121_d.json`・`p_x_s_1080_129_d.json` |
| `j_`（全 on 版） | LTX 2.3 w4a8 | `ltx-2.3-22b-distilled-1.1_w4a8` | 全 on（入口判定 G-C の候補） | `p_j_warm.json`・`p_j_s_1080_b89.json`・`p_j_s_1080_145.json`・`p_j_s_1080_161.json` |
| `j_`（既定版） | 同上 | 同上 | 既定（`_d`。G-C の候補） | `p_j_warm_d.json`・`p_j_s_1080_b89_d.json`・`p_j_s_1080_129_d.json`・`p_j_s_1080_145_d.json` |
| `k_`（P4・任意） | LTX 2.3 Kijai int8_convrot（transformer のみ） | `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot` | 既定（`_d`） | `p_k_warm_d.json`・`p_k_s_1080_b89_d.json`・`p_k_s_1080_121_d.json`・`p_k_s_1080_129_d.json` |

幾何・フレーム・トークンの内訳:

| ラベル | 幾何 | フレーム | 潜在 | トークン | 線比(44,880) | 基準点 |
|---|---|---|---|---|---|---|
| `*_warm` / `*_warm_d` | 768×512 | 121f | 16 | 6,144 | 14% | （自身が基準点・判定しない） |
| `u_s_1080_b89` / `r_s_1080_b89` / `h_s_1080_b89` / `f_s_1080_b89_d` / `x_s_1080_b89_d` / `j_s_1080_b89` / `j_s_1080_b89_d` / `k_s_1080_b89_d` | 1920×1088 | 89f | 12 | 24,480 | 55% | （自身が基準点） |
| `r_s_720_b169` | 1280×768 | 169f | 22 | 21,120 | 47% | （自身が基準点） |
| `u_s_1080_145` / `h_s_1080_145` | 1920×1088 | 145f | 19 | 38,760 | 86% | `u_s_1080_b89` / `h_s_1080_b89` |
| `u_s_1080_153` / `r_s_1080_153` / `h_s_1080_153` | 1920×1088 | 153f | 20 | 40,800 | 91% | 各腕の `*_b89` |
| `r_s_1080_161` / `j_s_1080_161` | 1920×1088 | 161f | 21 | 42,840 | 95% | `r_s_1080_b89` / `j_s_1080_b89` |
| `f_s_1080_121_d` / `x_s_1080_121_d` / `k_s_1080_121_d` | 1920×1088 | 121f | 16 | 32,640 | 73% | 各腕の `*_b89_d` |
| `f_s_1080_129_d` / `x_s_1080_129_d` / `k_s_1080_129_d` / `j_s_1080_129_d` | 1920×1088 | 129f | 17 | 34,680 | 77% | 各腕の `*_b89_d` |
| `j_s_1080_145` | 1920×1088 | 145f | 19 | 38,760 | 86% | `j_s_1080_b89` |
| `j_s_1080_145_d` | 1920×1088 | 145f | 19 | 38,760 | 86% | `j_s_1080_b89_d` |

作ったコマンド列は `mkplan_commands.sh`、出力は `mkplan_output.txt` にあります。全て `--base-model`・`--transformer`・`--accel` を省略していません（手順書 v2 §5・§6）。

**登録名の確認**: 7 本の変換器名は、`models/LTX23/Weights/`・`models/LTX25/Weights/` に実在するファイル名から拡張子を除いたもの（`GET /models` の慣習）と確認済みです（2026-09-27 時点）。`redgraftLTX25Fast2K_ltx25RedgraftNSFW`（safetensors・`.part`）と既存の GGUF `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` は別名で衝突しません。`r_`・`h_` の 2 本は準備時点で `.part`（ダウンロード中）です。**実施の前に `calib.py preflight --online` で確認し、まだ `.part` のままなら該当の腕は延期してください**（手順書 v2 では「未ダウンロードの計画を作ってよい」とされているだけで、実施可否は改めての確認が要ります）。

Tier B・C（境界の絞り込み・`_r2` の 2 回目）は、Tier A の結果を読んでから監督役が作ります（手順書 v2 §3・§5）。

## 4. 実行順・入口判定・安全弁（c4_procedure_v2.md §3・§4 の写し）

### 実行順

2.5 → 2.3 の順（ベースモデルの切り替えは 1 回）: u_ → r_（1080p→720p）→ h_（→条件付き 720p）→〔LTX23〕→ f_ → x_ → j_ →（k_）→ 復元。

各腕: `calib.py load`（前面・出力を `load_<頭>_output.txt` へ。応答と `GET /models` の二重照合）→ **別の呼び出しで** `idle-calib --minutes 3` → 捨てランは**次の弾の drive2 の先頭の計画に入れる**（基準点の前に冷却 150 秒が入る）。弾と弾のあいだは 150 秒以上あける。

j_ の入口判定（G-C）: 全 on の捨てラン＋89f の後に `commit_log` の最大を見て、**88% 未満なら全 on の Tier A（`p_j_s_1080_145.json`・`p_j_s_1080_161.json`）へ**、**88% 以上なら `load` し直さずに既定構成の計画（`p_j_warm_d.json` から）へ切り替える**（構成の切り替え後は捨てランを 1 本置く）。

### 安全弁（毎回必ず守る）

1. **`server start` は `calib.py server status` が 0（稼働中）のときだけ**、`--pid` なしで打つ。ポートが閉じていると較正台が自分で `run.ps1` を起動してしまう。閉じていたら止めてオーナーへ報告する。
2. **コミット監視の生存確認**: 各弾の起動前に `commit_log.csv` の最終行の時刻が 10 秒以内であることを確かめる（監視が死ぬと `drive2.sh` の停止線が素通りになる）。90% で警戒（記録して続行）・95% で停止（実行中のジョブは完走させ、次を投入しない）。
3. 1 点 1 計画ファイル＋`drive2.sh`。1 回の起動は Tier A か B の 1 弾。**`load` と `idle-calib` は別の呼び出し**（1 回のコマンドにまとめない）。待ち受けスクリプトは使わない（監督が 1 弾ずつ手で起動・終了通知の後に結果を読む）。起動前に `bash`・`python calib.py` の残りプロセスが無いことを確認する。止めるときは `taskkill /T /PID`（Git Bash に `pkill` は無い）。
4. **止める条件は `transformer_used` が要求と不一致（05:56 の事故の型）だけではなく**、ワーカー落ち／422／待ち時間切れ（`--job-timeout 3600`）／コミット 95% も含む。ただし **`fused_gguf_dequant_kernel_used` 等の他の `*_used` の不一致は記録して続行する**（止めない）。
5. 占有: 計測中はオーナーが Create・Batch を使わない前提（開始前に了承）。各弾の前に `GET /status` の行列が空であることを確認する（`drive2` が確認する）。
6. サーバーは止めず再起動もしない。
7. **復元は最後に `restore-state` を 2 回**（開始時の `active_base_model` を最後に）。**復元先は C-4 開始時の選択（`original_state.json`）と一致することを、開始前にオーナーの普段の選択（LTX23＝fp8 Sulphur・LTX25＝default）と突き合わせて確認する。**

## 5. 較正台の使い方（c4_procedure_v2.md §5 の写し・監督役が打つコマンド）

作業ディレクトリはこのフォルダ、Python は `..\..\.venv\Scripts\python.exe`、環境変数 `PYTHONIOENCODING=utf-8`。

```
# 着手前（GPU なし・今回はここまで実施済み）
python selftest/parser_selftest.py     # 299 check(s), 0 failure(s)
python selftest/analyse_selftest.py    # 43 check(s), 0 failure(s)
python calib.py preflight              # --online なし。33 point(s), 0 failed, ok=True

# ここから先は実施（GPU を使う本番セッション）で行う ------------------------

# preflight --online で r_・h_ の .part が解消しているか確認してから着手
python calib.py preflight --online

# サーバーの登録（GET /status が 200・queue.pending と queue.running が 0 のとき、
# server status が 0 のときだけ --pid なしで打つ）
python calib.py server start

# 例: u_ 腕
python calib.py load --base-model LTX25 --transformer ltx25_uncensored_v1.1-fp8_scaled
python calib.py idle-calib --minutes 3
bash drive2.sh drive2_u.log plans/p_u_warm.json plans/p_u_s_1080_b89.json plans/p_u_s_1080_145.json plans/p_u_s_1080_153.json

# 集計と復元
python calib.py report
python calib.py restore-state --base-model LTX23
python calib.py restore-state --base-model LTX25
```

`commit_monitor.ps1` は較正台の外で別窓に回す（`powershell -NoProfile -ExecutionPolicy Bypass -File commit_monitor.ps1 -Out commit_log.csv -IntervalSec 2`、このフォルダで）。

## 6. 成果物（実施後に作るもの・c4_procedure_v2.md §6 の写し）

- `RESULTS.md`（一次記録）。
- `Docs/COMFORT_LIMIT_TABLE.md` 第14節「int8 系 safetensors での実測」。
- `Docs/VERIFICATION_LOG.md` §121 に C-4 の節。C-4 の完了は台帳 §1-32 のクローズ（CLOSED への移送）で表す。
- `Docs/Outputs-archive/comfort-calib-2026-09-27/` へ複写。

**今回の準備フェーズでは `Docs/` にも `README.md`（リポジトリ直下）にも一切触れていません。**

## 7. 実施の門と日程（c4_procedure_v2.md §7 の写し）

- 1 日目（サーバー起動後）: C-1〜C-3 の実機の門 G3〜G8。
- 2 日目（GPU を約 6〜8 時間占有できる了承を得てから）: C-4 を一括（同一プロセス）。**今回の準備フェーズはこの前段階です。**
- 較正の出力と検証用 safetensors は残す（削除はオーナー判断）。

---

## 8. ファイルの見取り図（今回作ったもの）

| 名前 | 中身 |
|---|---|
| `calib.py`・`mkplan.py`・`batch_table.py`・`make_results.py`・`vram_sampler2.ps1`・`commit_monitor.ps1` | 複製元と無変更（第2節） |
| `selftest/parser_selftest.py` | 節 H を今回の 33 点へ書き直し（節 A〜G は無変更）。299 check(s), 0 failure(s) |
| `selftest/analyse_selftest.py`・`selftest/fake_runs/`・`selftest/plans_fixture/` | 複製元と無変更。43 check(s), 0 failure(s) |
| `drive2.sh`・`commit_table.py`・`digest.py` | パスだけ今回のフォルダへ書き換え |
| `plans/p_*.json` | Tier A の計画ファイル 33 本（1 点 1 ファイル） |
| `mkplan_commands.sh`・`mkplan_output.txt` | 計画ファイルを作ったコマンド列と出力 |
| `selftest_output.txt`・`analyse_selftest_output.txt` | 上記 selftest 2 本の実行結果 |
| `preflight_output.txt`・`preflight.json` | オフライン `preflight`（`--online` なし）の結果。33 point(s), 0 failed, ok=true |
| `harness_diff.txt` | 複製元（第7次・2026-09-26）との差分 |
| `progress.md` | 実施セッション用の空の雛形（日時・弾・結果・判断） |
| `README.md` | この文書 |

`original_state.json`・`calib/`・`runs/`・`manifest.jsonl`・`server.pid`・`luid.json` はまだありません（実施セッションの `server start`／`load`／`idle-calib`／`run` で作られます）。

---

## 実施結果（2026-09-28 追記）

較正は 2026-09-27 23:33〜2026-09-28 03:55 に実施しました。結果は同じフォルダの `RESULTS.md`、結論は `Docs/COMFORT_LIMIT_TABLE.md` 第14節、実施記録は `Docs/VERIFICATION_LOG.md` §121.12 です。
