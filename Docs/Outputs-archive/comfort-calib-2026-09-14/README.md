> **このファイルは実機の `outputs/comfort-calib-2026-09-14/README.md`（git追跡外）のスナップショットである（2026-09-14複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# 快適上限キャリブレーション計測台（第4次・2026-09-14／別GGUF〔Q6_K〕での実測）

> **規約の正本は `outputs/comfort-calib-2026-09-04/README.md` の第5節〜第7節です。**
> 判定規約 v3′、`metadata.json` の `peak_vram_*` を参照つきの点と参照なしの点のあいだで
> 比べてはいけないという注意、副命令の一覧、サーバーの起動のしかた、冷却150秒、
> 実行の順番 —— これらはすべて第2次（2026-09-04）の README が正本であり、
> **この文書では書き写しません。** ここに書くのは**複製元（第3次・2026-09-05）との差分だけ**です。
>
> 第2次・第3次のフォルダは**読み取り専用の履歴**です。1バイトも書き換えません。

---

## 1. 何を測るのか

**REDGraft LTX 2.5 を Q6_K で GGUF 化した transformer（変換器）での快適上限を、実測で求めます。**

現在配信している線（`ltx25` の無条件行＝単発生成・連結生成とも 44,880 トークン）は、
公式の Q4_K で較正した値です。2026-09-13 の実機ゲートで、Q6_K は公式 Q4_K に比べて
除去（denoise）の場面で約 1.03 ギビバイト多く使い、ブロック差し替えでやり取りする
データ量が約 1.46 倍になることが分かっています。**線が下がる見込みがあるので測り直します。**

**スコープは「測って結論を出す」までです。** 配信値（`config.py` の `_default_comfort_budgets()`）は
変更しません。フロントエンド・バックエンドへの反映方法も設計しません。

計測の一次記録は `RESULTS_comfort-calib-2026-09-14.md` です。

---

## 2. 複製元（第3次・2026-09-05）との差分

複製したのは `calib.py`・`vram_sampler2.ps1`・`batch_table.py`・`make_results.py`・
`mkplan.py`・`selftest/` です。`runs/`・`calib/`・`manifest.jsonl`・`luid.json`・
`original_state.json`・`reference_uploads.json`・`preflight.json` は複製していません
（それらは第3次の測定結果そのものだからです）。

`calib.py` への変更は次の8か所だけで、ほかは1文字も変えていません。
**改修前後の差分そのものは `harness_diff.txt` に保存してあります**（どの塊がどの項目かの
対応表も同ファイルの冒頭に付けてあります）。

| # | 変えた場所 | 何を変えたか |
|---|---|---|
| T1 | `active_transformer()`（新設）・`cmd_load` | `--transformer` を要求本体に載せる。読み込みの応答と `GET /models` の**両方**が要求どおりでなければ非零で止まる |
| T2 | `main` | `load`／`restore-state` に `--transformer`（既定は指定なし） |
| T3 | `record_original_state` | `state.json` の `selections`（土台モデルごとの選択）も控える |
| T4 | `cmd_restore_state` | 控えた選択で戻し、戻したあと `state.json` を読み直して確かめる。**ファイルを手で書かない** |
| T5a | `cmd_run` | 点ごとに、いま載っている transformer が点の宣言と違えば `skipped_wrong_transformer` で記録して飛ばす |
| T5b | `check_point` | 点は transformer を必ず宣言する。**宣言の無い点は誤りとして止める** |
| T6 | `used_check`・`analyse`・`cmd_report`・`make_results` | どの transformer で走ったかを `metadata.json` から拾い、要求と突き合わせ、表に載せる |
| T7 | `cmd_idle_calib` | 待機時の計測を transformer ごとの別フォルダに分け、記録にも transformer を残す |
| T8 | `check_point`ほか | **連結の点**を製品の規則で検算する（従来は必ず赤になっていた）。フレーム数の表示は各クリップの合計へ |

`mkplan.py` は今回の梯子（Tier B／Tier C）用に書き直しました。`selftest/parser_selftest.py` の
節Hも今回の計画ファイルで書き直しています（節A〜Gは複製元のままです）。

---

## 3. 計画ファイル

`plans/` に11本あります。共通の条件は seed 12345・長い車の指示文（`PROMPT_25`）・24fps・
**加速はサーバー既定**（点に `accel` を書きません）。連結は `clips: [25, 161]`・
窓 `standard`・チャンク化アップサンプルなしで、第1次（2026-08-31）の E3 と同じ形です。

| ファイル | 点 | 何の腕か |
|---|---|---|
| `plan_W_warm_q6.json` | 1 | 暖機（768×512／121コマ） |
| `plan_A1a_single_720_q6.json` / `plan_A1b_…` | 4 / 1 | 単発 1280×768 |
| `plan_A2a_single_1080_q6.json` / `plan_A2b_…` | 4 / 1 | 単発 1920×1088 |
| `plan_A3a_single_896_q6.json` / `plan_A3b_…` | 4 / 1 | 単発 896×1152（縦長） |
| `plan_A4a_chain_q6.json` / `plan_A4b_…` | 5 / 1 | 連結（窓22） |
| `plan_W_warm_q4.json` | 1 | 公式 Q4_K へ戻したあとの暖機 |
| `plan_Q_control_q4.json` | 4 | 公式 Q4_K の対照 |

**各腕が「下段」と「最上段」の2本に分かれているのは、打ち切りのためです。**
`run` は計画ファイルを最後まで回しますし、背景で走らせている以上こちらから
Ctrl-C を押せません。下段の結果を `batch_table.py` で見てから、最上段を走らせるかどうかを
決めます。

---

## 4. この計測台を使うときに踏みやすい落とし穴（3つだけ）

1. **計画ファイルの境目には冷却が入りません。** `cmd_run` の冷却は1つのファイルの中の
   2点目以降にだけ効きます。次の `run` を打つ前に、手で150秒空けてください。
2. **公式 Q4_K の対照（P8）で `idle-calib` を回したあとは、Q6_K の点を `reanalyse` しないでください。**
   待機時の計測結果は `calib/calibration.json` 1本を上書きするので、上書き後に解析をやり直すと
   別の基準で判定し直すことになります。確定した点は判定に使った値を記録の中に焼き込んであるので、
   やり直す必要もありません。
3. **`preflight` に `--online` は付けません。** `--online` が確かめるのは制御アダプタ2種の
   有無で、今回の計測はそれを使いません。

---

## 5. ファイルの見取り図

| 名前 | 中身 |
|---|---|
| `calib.py` | 計測台の本体（副命令の一覧は第2次 README 第7節） |
| `mkplan.py` | Tier B／Tier C の計画ファイルを作る（同じラベルの2度目を拒む） |
| `batch_table.py` | 腕ごとの表を出す（GPU不要） |
| `make_results.py` | 一次記録の結果表を機械生成する（手打ちしない） |
| `vram_sampler2.ps1` | GPUメモリの採取（複製元のまま） |
| `selftest/` | GPUを使わない自己検査2本（`selftest_output.txt` が実行結果） |
| `plans/` | 計画ファイル11本 |
| `harness_diff.txt` | 複製元との差分（T1〜T8） |
| `preflight_output.txt` | 全計画の机上検算の結果（27点・失敗0） |
| `git_state_start.txt` | 開始時の作業ツリー・GPU・ポート・空き容量の記録 |
| `RESULTS_comfort-calib-2026-09-14.md` | 一次記録（これが正本） |
