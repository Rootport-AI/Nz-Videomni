> **このファイルは実機の `outputs/comfort-calib-2026-09-26/README.md`（git追跡外）のスナップショットである（2026-09-26複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# 快適上限キャリブレーション計測台（第7次・2026-09-26／fp8 safetensors の transformer での快適上限の確認較正）

> **規約の正本は `outputs/comfort-calib-2026-09-04/README.md` の第5節〜第7節です。**
> 判定規約 v3′、`metadata.json` の `peak_vram_*` を参照つきの点と参照なしの点のあいだで
> 比べてはいけないという注意、副命令の一覧、サーバーの起動のしかた、冷却150秒、
> 実行の順番 —— これらはすべて第2次（2026-09-04）の README が正本であり、
> **この文書では書き写しません。** ここに書くのは**複製元（第6次・2026-09-25）との差分と、今回の点の設計だけ**です。
> 計測の計画の正本は、承認済みの計画書（§3-167 B-3・2026-09-26）です。
>
> 第2次から第6次までのフォルダは**読み取り専用の履歴**です。1バイトも書き換えません。

---

## 1. 何を測るのか

**transformer（変換器）を fp8 safetensors にしたとき、快適上限の線（VRAM 溢れが起きない生成規模の目安）が
GGUF のときと同じく成り立つかを確かめます。** fp8 を置くだけで使えるようにしたのは
`Docs/PENDING_TASKS.md` §3-167 の B-1（LTX 2.3）と B-2（LTX 2.5）で、記録の正本は
`Docs/VERIFICATION_LOG.md` §117・§118 です。fp8 は GPU に常駐させるブロックが GGUF より重いので、
線が下がる可能性があります（見立ては計画書 §1）。

- 線は、LTX 2.3 の全 on が単発 44,880・連結 40,000、LTX 2.5 が単発・連結とも 44,880 です
  （`Docs/COMFORT_LIMIT_TABLE.md` 第1.1節）。
- 対象の変換器は、LTX 2.3 が `sulphur_distil_fp8mixed`、LTX 2.5 が
  `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` です。同じプロセスの対照として、LTX 2.3 は
  `Sulphur-2-base-distil-Q6_K`、LTX 2.5 は公式の `default` を走らせます（Tier Q。計画は 2 ランずつ、実際は基準点＋161 コマ＋169 コマの 3 ランずつ）。
- 構成は**全 on**（すべての加速を入れた構成）に固定します。LTX 2.3 は
  `attention_backend=sage`・`block_swap_prefetch=true`・`keep_resident=true`・
  `fused_gguf_dequant_kernel=true`・`vae_mode=prune_vaed`・`keep_resident_embeddings=false`、
  LTX 2.5 は `vae_mode` を既定のままにし、`keep_resident_embeddings=true` にします
  （`mkplan.py` の `ACCEL_ALLON_LTX23`・`ACCEL_ALLON_LTX25`。定義の正本は `Docs/COMFORT_LIMIT_TABLE.md` 末尾の付記の1項目目）。
  fused カーネルは fp8 の transformer には効きませんが、記録上は `on` と出ます。
- 全点に共通の形: T2V・シード 12345・24 fps・指示文は較正台の既定（LTX 2.3 は `PROMPT_23`、LTX 2.5 は `PROMPT_25`）。
- 判定は**規則 v3′**です（第4節）。
- 併せて、§3-165 の申し送り（Sulphur-2 Q6_K の 1216×704・w46 の VRAM 溢れを配信値に反映するか）に材料を出します（Tier D）。判断はオーナーです。
- **スコープは「測って結論を出す」までです。** 結論は「線は動かない」「線が下がる（解像度ごとに割れる場合を含む）」
  「単一の線では表せない」のどれでもよく、どれも「配信値への反映は別途検討」の形で閉じます。
  配信値（`config.py` の `_default_comfort_budgets()`）は変更しません。

計測の一次記録は `RESULTS.md` です（結果表は `make_results.py` の機械生成）。
**計測は 2026-09-26 02:06〜07:45 に終わりました（57 ラン・失敗 0・無効 1）。結論と、計画と実際の違いは `RESULTS.md` 第1節・第3節・第7節にあります。**

### サーバーについて

**今回は、監督役が B-2 の段階 3 のために 2026-09-26 の 0 時 2 分に起動したサーバー（shell PID 49264）を使います。**
オーナーが起動したサーバーではないので、`server start --pid 49264` で登録してかまいません（第5節 2 項）。
着手前の `GET /status` は `status_at_p0.json`、作業ツリーの状態は `git_state_start.txt` にあります
（手順 0 の後の dev・HEAD `246d358`。サーバーが動かしているコード `755f5ff` との差は文書 2 本だけです）。
開始時の `state.json` は、`active_base_model` が LTX25、LTX23 の選択が `sulphur_distil_fp8mixed`、LTX25 の選択が `default` です。

---

## 2. 複製元（第6次・2026-09-25）との差分

複製したのは `calib.py`・`mkplan.py`・`batch_table.py`・`make_results.py`・
`vram_sampler2.ps1`・`selftest/`（`parser_selftest.py`・`analyse_selftest.py`・`fake_runs/`・`plans_fixture/`）・`README.md` の**コードだけ**です。
**`original_state.json` は複製していません。** `record_original_state()` はこのファイルがあると
何もせずに戻るので、第6次の控えを持ち込むと、終了時の復元が別の変換器へ戻ってしまいます。
`server.pid`・`luid.json`・`preflight.json`・`manifest.jsonl`・`progress.md`・`runs/`・`plans/`・`calib/`・`tables/`・
`reference_uploads.json`・`RESULTS.md`・`*_output.txt`・`harness_diff.txt`・`git_state_*`・`server_console*.log` も複製していません。

変更は次の **2 か所だけ**です。**差分そのものは `harness_diff.txt` に保存してあります。**
`calib.py` と `mkplan.py` は **1 バイトも変えていません**（第6次で入った T12 の窓の写しと T14 の `--label` で足ります）。

| # | 変えた場所 | 何を変えたか |
|---|---|---|
| T15 | `README.md` | この文書。今回の計画・複製元との差分・点の設計・落とし穴と見取り図の引き継ぎ・コミット上限の監視 |
| T16 | `selftest/parser_selftest.py` の**節 H と冒頭の 1 文だけ** | 今回の Tier D・Tier A の計画ファイル 9 本（30 ラン）に向けて書き直した。節 A〜G は 1 文字も変えていない |

`analyse_selftest.py`・`make_results.py`・`batch_table.py`・`vram_sampler2.ps1` も変えていません。

---

## 3. 点の設計

トークン数の数え方は **(幅÷32) ×(高さ÷32) ×潜在フレーム数** です。単発の潜在フレーム数は (フレーム数−1)÷8＋1、
連結は Stage-2 の窓の潜在フレーム数（`standard`＝22）です。
ラベルの頭は、`f_` が LTX 2.3 fp8、`g_` が LTX 2.5 fp8、同じプロセスの対照が既存の規約どおり `a_`（LTX 2.3 Q6_K）と `b_`（LTX 2.5 公式）です。
`_s_` が単発、`_c_` が連結、`b<数>` が基準点、`_r2` が 2 回目です。
着手前の計画ファイルは**幾何ごとに小さく分けて**ありました。作ったときのコマンドは `mkplan_commands.sh`、出力は `mkplan_output.txt`、
全ファイルの宣言（ベースモデル・変換器・加速の組）とトークン数の機械照合は `plan_check_output.txt` にあります。

> **実際の運用では、計測の途中で「1 点 1 計画ファイル」に組み替えました**（下の「実際に流した形」）。
> 下の Tier D・Tier A の表にある幾何ごとの計画ファイル（`plan_D1_*`・`plan_A*`）は、使わずに `plans/unused/` へ移してあります。
> 点の設計（ラベル・幾何・トークン・基準点）そのものは変えていません。

### 捨てラン（各 `load` の直後に 1 本・判定しない）

| 計画ファイル | ラベル | 変換器 |
|---|---|---|
| `plan_W_f_warm.json` | `f_warm` | `sulphur_distil_fp8mixed` |
| `plan_W_a_warm.json` | `a_warm` | `Sulphur-2-base-distil-Q6_K` |
| `plan_W_g_warm.json` | `g_warm` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` |
| `plan_W_b_warm.json` | `b_warm` | `default` |

どれも 768×512×121 フレーム（6,144 トークン）・全 on です。重みの読み込みと keep_resident の写しをここで済ませ、
`used_matches_request` とコミットの初回確認を兼ねます。形の上では `is_baseline` ですが、どの点もこれを基準に参照しません。

### Tier D（LTX 2.3 fp8・§3-165 の申し送り向け・最初に走らせる・`plans/plan_D1_f_w46.json`・3 ラン）

クリップは 361 フレーム×2 本です。`plans/fragments/` の断片 3 本（`f_b1216`・`f_w46`・`f_w46_r2`）を 1 本に連結しました。

| ラベル | 解像度・窓 | 潜在 | トークン（線 40,000 比） |
|---|---|---|---|
| `f_b1216` | 1216×704・22 | 22 | 18,392（46%・基準点） |
| `f_w46`・`f_w46_r2` | 1216×704・w46 | 46 | 38,456（96%） |

### Tier A（LTX 2.3 fp8・15 ラン）

| 計画ファイル | ラベル | 幾何 | 潜在 | トークン（線比） |
|---|---|---|---|---|
| `plan_A1_f_1080.json` | `f_s_1080_b89` | 1920×1088・89f | 12 | 24,480（55%・基準点） |
| | `f_s_1080_161` | 1920×1088・161f | 21 | 42,840（95%） |
| | `f_s_1080_169` | 1920×1088・169f | 22 | 44,880（100%・内側の最上段） |
| | `f_s_1080_177` | 1920×1088・177f | 23 | 46,920（105%・外側） |
| `plan_A2_f_720.json` | `f_s_720_b185` | 1280×768・185f | 24 | 23,040（51%・基準点） |
| | `f_s_720_345` | 1280×768・345f | 44 | 42,240（94%） |
| | `f_s_720_361` | 1280×768・361f | 46 | 44,160（98%・内側の最上段。Q6_K の孤立溢れ点） |
| | `f_s_720_369` | 1280×768・369f | 47 | 45,120（101%・外側） |
| `plan_A3_f_p1280.json` | `f_s_p1280_b185` | 768×1280・185f | 24 | 23,040（51%・基準点） |
| | `f_s_p1280_361` | 768×1280・361f | 46 | 44,160（98%・内側の最上段） |
| | `f_s_p1280_369` | 768×1280・369f | 47 | 45,120（101%・外側） |
| `plan_A4_f_chain.json` | `f_c_b1280x768` | 連結 [25,161]・1280×768・窓 22 | 22 | 21,120（53%・基準点。§113 の先例どおり共用） |
| | `f_c_1792x1024` | 連結・1792×1024 | 22 | 39,424（99%・内側の最上段） |
| | `f_c_1856x1024` | 連結・1856×1024 | 22 | 40,832（102%・外側） |
| | `f_c_1920x1088` | 連結・1920×1088 | 22 | 44,880（112%・外側） |

### Tier A（LTX 2.5 fp8・12 ラン）

| 計画ファイル | ラベル | 幾何 | 潜在 | トークン（線 44,880 比） |
|---|---|---|---|---|
| `plan_A5_g_1080.json` | `g_s_1080_b89` | 1920×1088・89f | 12 | 24,480（55%・基準点） |
| | `g_s_1080_161` | 1920×1088・161f | 21 | 42,840（95%） |
| | `g_s_1080_169` | 1920×1088・169f | 22 | 44,880（100%・内側の最上段） |
| `plan_A6_g_720.json` | `g_s_720_b169` | 1280×768・169f | 22 | 21,120（47%・基準点） |
| | `g_s_720_361` | 1280×768・361f | 46 | 44,160（98%・内側の最上段） |
| | `g_s_720_369` | 1280×768・369f | 47 | 45,120（101%・外側） |
| `plan_A7_g_896.json` | `g_s_896_b161` | 896×1152・161f | 21 | 21,168（47%・基準点） |
| | `g_s_896_345` | 896×1152・345f | 44 | 44,352（99%・内側の最上段） |
| | `g_s_896_353` | 896×1152・353f | 45 | 45,360（101%・外側） |
| `plan_A8_g_chain.json` | `g_c_b1280x768` | 連結 [25,161]・1280×768・窓 22 | 22 | 21,120（47%・基準点） |
| | `g_c_1856x1024` | 連結・1856×1024 | 22 | 40,832（91%） |
| | `g_c_1920x1088` | 連結・1920×1088 | 22 | 44,880（100%・内側の最上段） |

### Tier B・C・Q（結果を見てから監督役が作る）

降りる段・条件つきの段・同一プロセスの対照は、計画書 §4-1〜§4-3 の規則に従い、Tier A の結果を読んでから作ります。
形の見本として `plans/plan_B1_f_1080_169_r2.json`（`f_s_1080_169_r2`・基準は `f_s_1080_b89`）だけを先に作り、机上検算に通してあります。
Tier B 以降のファイルは節 H の対象外なので、`mkplan` の出力行（`transformer=`・`accel=allon`）を目で確かめてください。

### 実際に流した形（1 点 1 計画ファイル・2026-09-26 02:27 から）

- **計画ファイルは 1 点ずつ `plans/p_<ラベルの頭>_<幾何>_<コマ数>[_d][_r2].json`** に分けました（例: `plans/p_g_1080_145_r2.json`）。
  捨てランは `plans/plan_W_*.json`、Tier D は断片 `plans/fragments/f_b1216.json`・`f_w46.json` を 1 本ずつ流しました（`f_w46_r2` は保留）。
- **駆動は `drive2.sh`**（較正台の外）です。1 点を流すごとに、その点の最中のコミットの最大を `commit_log.csv` から読み、
  **95% 以上なら次の点を投入せずに止まります**。点と点のあいだで行列が空になるのを待ち、150 秒の冷却を入れます。記録は `drive2_*.log` です。
- **止まったのは 2 回です**（`f_w46`・`f_s_1080_161`）。LTX 2.3 fp8 の全 on はここで打ち切り、既定構成の診断点（ラベルの末尾 `_d`）で代えました。
- **Tier B・Q と追加診断**は、結果を見ながら監督役が 1 点ずつ作りました。制御段階（公式 GGUF と Q6_K の対照、LTX 2.3 fp8 の既定構成の降りる段、復元）は `phase_controls.sh`、
  修正段階と追加診断 X3・X4 は監督役の待ち受け（`fixup.log`・`x3.log`・`x4.log`）で流しました。
- 実際の順番の正本は `RESULTS.md` 第3.2節、時刻は `progress.md` です。

### 実行の順番と、変換器の切り替え（計画時点のもの）

1. 2.3 fp8: `load --base-model LTX23 --transformer sulphur_distil_fp8mixed` → `idle-calib --minutes 3` → `f_warm` → Tier D → Tier A（1080p → 720p → 768×1280 → 連結）→ Tier B → Tier C
2. Q6_K 対照: `load --base-model LTX23 --transformer Sulphur-2-base-distil-Q6_K` → `idle-calib` → `a_warm` → Tier Q
3. 2.5 fp8: `load --base-model LTX25 --transformer ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` → `idle-calib` → `g_warm` → Tier A（1080p → 720p → 896×1152 → 連結）→ Tier B
4. 公式対照: `load --base-model LTX25 --transformer default` → `idle-calib` → `b_warm` → Tier Q
5. 復元（2 回・順番固定）: `restore-state --base-model LTX23`（控え＝`sulphur_distil_fp8mixed`）→ `restore-state --base-model LTX25`（控え＝`default`。最後にして `active_base_model` を LTX25 に戻す）。出力は `restore_state_output.txt` へ。

所要の見込みと、時間が足りないときに削る順番は計画書 §4-4 が正本です。
**実際には、2 の Q6_K の対照と 4 の公式の対照を制御段階にまとめて 3 の後に流し、復元は各段の後に 4 回行いました**（`RESULTS.md` 第3.2節）。

---

## 4. 判定の読み方

**境界は `runs/<ラベル>/run.json` の `v3prime.windows.stage2` と `v3prime.windows.restore` の
2 つの窓だけで決めます。** 同じ幾何・同じ変換器の基準点（単発は低フレームの点、連結は窓 22 の点）の窓中央値と比べ、
共有 GPU メモリの中央値が **+300 メガバイト**以上上がれば VRAM 溢れとみなします。10 標本未満の窓は判定不能です。

- **LTX 2.3 の点**には、この 2 つに加えて `s23_stage1`・`s23_gap_mid`・`s23_gap_tail` の 3 つの窓が付きます。
  `verdict_v3prime` は**5 つの窓の複合**で、どれか 1 つでも +300 メガバイトを超えれば `plateau` になります。
  2 窓の判定と食い違うことがあるので、一次記録では両方を並べ、食い違ったランは名前を挙げます。
  LTX 2.3 では復元の窓と `s23_gap_tail` は同じ区間で、値は常に一致します。
- **LTX 2.5 の点**には、`stage2`・`restore`・`p25_30_decode_encode` の 3 つの窓が付きます。
- **2 回一致の規則**: 各幾何の内側の最上段は結果によらず 2 回走らせます。境界の材料にするのは 2 回一致した点だけで、
  割れた点は境界に採らず 3 回目も走らせません。外側の点は 1 回だけです（逆行したときの例外は計画書 §4-3）。
- **結論の 3 つの形の見分け方**と比較表の組み方は計画書 §7 が正本です。
- **LTX 2.3 のワーカーは PHASE 行を出しません。** 専有の値は WDDM のカード全体の値（約 1 秒に 1 回の採取）と、
  ジョブ全体のピークだけです。`Docs/COMFORT_LIMIT_TABLE.md` 第10.3節の「torch が確保した量」と直接比べてはいけません。
- **Stage-2 の速さは `sec/token` 列では比べません。** 比べるときは、同じ幾何の基準点の `stage2.seconds` と並べます。
- **最初のジョブの所要は比べません。** fp8 の読み込み（約 29 GB／23.5 GB）と keep_resident の写しが混ざるからで、そのために捨てランを置いています。

---

## 5. この計測台を使うときに踏みやすい落とし穴

### 第6次から引き継ぐもの

1. **`original_state.json` を前のキャンペーンから持ち込まないでください。** あると `server start` が控えを取り直さず、
   終了時の復元が前のキャンペーンの選択へ戻ります（第2節）。
2. **`server start --pid` は、較正のために監督役が起動したサーバーにだけ付けます。** オーナーが起動したサーバーに付けると、
   万一 `server stop` を打ったときにオーナーのサーバーへ届きます。今回の 49264 は監督役が起動したものです。
   また、ポートが開いていないと `server start` は 2 つ目のサーバーを起動するので、`GET /status` が 200 を返すのを確かめてから打ちます。
3. **`server stop` は打ちません。** サーバーを止めるかどうかは監督役が決めます。
4. **変換器を替えた点には、必ず自前の基準点を持たせてください。** 基準点の参照は `manifest.jsonl` 全体から
   `baseline_label` の最後のレコードを拾う仕組みで、**変換器を照合しません。** 今回は `f_`・`g_`・`a_`・`b_` でラベルを分けてあります。
5. **切り替えのあとに `reanalyse` を打たないでください。** `idle-calib` は `calib/calibration.json` を 1 本上書きします。
   v3′ の判定は基準点の窓中央値だけを使うので、上書き自体は確定済みの判定に影響しません。
6. **`idle-calib` はモデルを読み込んでから取ってください。** 読み込み前に取ると `pipeline_loaded=False` の警告が出て、
   待機時の床が低めに出ます（v3′ には効きません）。
7. **各計画ファイルを始める前に `GET /status` の行列が空であることを確認してください**
   （`queue.pending` と `queue.running` がともに 0）。
8. **計画ファイルの境目には冷却が入りません。** `cmd_run` の冷却 150 秒は 1 つのファイルの中の 2 点目以降にだけ効きます。
   次の `run` を打つ前に、手で 150 秒空けてください。
9. **`preflight.json` は最後に走らせた `preflight` の結果で上書きされます。** 今回の `preflight_output.txt` は、計測の終わり（07:44）に
   実際に使った計画ファイルすべて（85 点）へ走らせ直した記録です。途中で足したファイルの検算は `preflight_perpoint_*_output.txt` と `progress.md` にあります。
10. **`preflight` に `--online` は付けません。** `--online` が確かめるのは制御アダプタ 2 種の有無で、今回の計測はそれを使いません。
    LTX 2.3 の点に出る「prompt not pinned」の警告は想定どおりです（`build_request` が `PROMPT_23` を入れます）。
11. **`selftest/parser_selftest.py` の節 H はキャンペーンごとに書き直す節です。** 今回は T16 で Tier D・Tier A の 9 本（30 ラン）へ向け直してあり、
    `parser_selftest.py` は **231 項目・失敗 0**（節 A〜G の 72 項目＋節 H の 159 項目）、
    `analyse_selftest.py` は **43 項目・失敗 0** で通ります。
12. **節 H を書き直すときは、期待値を新しい入力に合わせて直すのではなく、その入力で本当に成り立つことだけを書いてください。**
    前者をやると、見張りが見張らなくなります。
13. **計測担当のエージェントが計画ファイルを作れない場合があります**（第5次で、計画ファイルを作る命令が道具の許可判定に
    繰り返し拒まれました）。そのときは監督役が計画ファイルを起こし、`calib.py run` まで直接実行してください。
    Tier B 以降は監督役が `calib.py run` を直接バックグラウンドで実行します。
14. **出力をファイルへ流すときは、標準出力の文字コードを UTF-8 に指定してください**（`PYTHONIOENCODING=utf-8`。
    `make_results.py` は `PYTHONUTF8=1`）。既定の cp932 では、全角の文字のところで異常終了します。

### 今回に固有のもの

15. **`calib/calibration.json` が無いと `run` は即座に終わります**（`calib.py:88-89`・`:2086-2088`「missing … run `idle-calib` first」）。
    複製先には `calib/` がありません。**各 `load` の直後に必ず `idle-calib --minutes 3` を打ってください。**
16. **各 `load` のあとに捨てランを 1 本走らせます**（768×512×121・判定しない）。切り替えの直後の最初のジョブを基準点にすると、
    重みの読み込みで基準の窓が歪みます。
17. **`mkplan.py` を呼ぶたびに `--base-model`・`--transformer`・`--accel allon` を省略しないでください。**
    `--base-model` の既定は `LTX25`、`--accel` の既定は `default` です（`mkplan.py:140-141`・`:160-171`）。
    `--accel` を忘れると点が既定構成を宣言し、`used_check` が構成の違いを検出できなくなります。
    窓だけを変える点には `--label` を付けてください（付けないと同じ解像度の基準点と名前がぶつかります）。
18. **コミット上限を監視してください（較正台の外）。** fp8 は Windows のコミット（確約済みメモリ）を GGUF より多く使います
    （LTX 2.3 fp8＋keep_resident は低解像度でも上限の 87% 前後）。監督役が PowerShell の
    `Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit' -SampleInterval 2 -Continuous` を裏で回して `commit_log.csv` に残します。
    **上限比 90%（約 102.4 GiB）で警戒＝記録して続行、95%（約 108.1 GiB）で停止**＝実行中のジョブを打ち切り（打ち切る手段が無ければ完了を待つ）、
    冷却中に計測を止めて次へ進まずに報告します。コミット不足の異常終了やワーカー落ちも即停止です。
    捨てランのあとの落ち着いた値を記録してから本計測に入ります。見極め点は各捨てラン・`f_b1216`・`f_s_1080_177`・`f_s_720_369` です。
    **実際の監視は `commit_monitor.ps1`（2 秒おき）で、記録はこのフォルダの `commit_log.csv` に複写してあります。**
19. **`GET /models` の非アクティブなベースモデルの `active` は、`state.json` の選択と一致しないことがあります**（P0 の時点で、
    LTX23 の選択は `state.json` では `sulphur_distil_fp8mixed` なのに、`GET /models` の LTX23 側は `default` と出ていました）。
    変換器の確認は `load` の応答と、読み込み後の `GET /models`（アクティブ側）で行います（`cmd_load` はそうしています）。

### 今回の計測で分かったこと（次に使う人へ）

20. **`DELETE /jobs/{id}` は、実行中のジョブの推論を止めません**（取り消しの要求を受け付けるだけです）。停止線に達したら、
    **実行中のジョブは完走させ、次の点を投入しない**という止め方しかありません。
21. **停止線の関門は「点の最中の最大」で見てください。** 最初に使った連続駆動 `drive.sh` は「次のファイルを始める直前のコミット」を見ていたので、
    点が終わって 80% 前後に戻った値で関門を素通りしました（`drive_DA_f.log`）。幾何ごとの計画ファイルの中の点は較正台が続けて投入するので、
    1 点ごとに止めたいときは **1 点 1 計画ファイル＋`drive2.sh`** の形にしてください。
22. **`calib.py preflight --plan` は 1 回に 1 ファイルしか受け付けません。** 1 点 1 ファイルにしたら、ファイルごとに走らせるか、
    `--plan` を付けずに `plans/` 全体へ走らせてください。
23. **Git Bash には `pkill` がありません。** 裏で回した駆動や待ち受けのスクリプトを止めるときは、PowerShell でプロセス番号を特定し、
    **`taskkill /T /PID <番号>` で子プロセスごと**止めてください。道具の停止だけでは子の `bash` が生き残ることがあります。
24. **待ち受けのスクリプトが 2 本残ると、変換器の切り替えと次の点の投入がぶつかります。** 今回、止めたはずの制御段階の待ち受け
    `phase_controls.sh` の子が生き残り、第 6 弾の完了後に `load … default` を実行した同じ秒に、第 7 弾の点が投入されました。
    その点は `default` で走り、`used_matches_request` の不一致で無効になりました（`RESULTS.md` 第7節の 3）。
    **待ち受けを張り替えるときは、古いものを木ごと止めてから張り直し、`used_matches_request` を点ごとに確かめてください。**
25. **LTX 2.3 fp8＋keep_resident（全 on）は、この機体（コミット上限 113.82 GiB）では 1080p 級の点でコミットが 95% を超えます**（基準点の 89 コマでも 91.0%）。
    同じ構成で測り直すなら、ページファイルを広げるなどコミットの上限を上げる手当てが要ります。
26. **`commit_table.md` の Tier D の 2 行（`f_b1216`・`f_w46`）は完了後の値です**（`drive.sh` の記録の形が違うため）。点の最大は `RESULTS.md` 表5 が正です。

---

## 6. ファイルの見取り図

| 名前 | 中身 |
|---|---|
| `RESULTS.md` | 一次記録（これが正本） |
| `calib.py` | 計測台の本体（副命令の一覧は第2次 README 第7節。今回は変更なし） |
| `mkplan.py` | 計画ファイルを作る（同じラベルの 2 度目を拒む。今回は変更なし） |
| `batch_table.py` | 計画ファイルごとの表を出す（GPU 不要） |
| `make_results.py` | 一次記録の結果表を機械生成する（手打ちしない） |
| `vram_sampler2.ps1` | GPU メモリの採取（複製元のまま） |
| `selftest/` | GPU を使わない自己検査 2 本（231 項目＋43 項目・失敗 0。結果は `selftest_output.txt`。変更は T16） |
| `plans/` | 実際に流した 1 点 1 計画ファイル（`p_*.json`）・捨てラン（`plan_W_*.json`）・Tier B の見本 1（未使用）・Tier D の断片（`plans/fragments/`）・使わなかった幾何ごとの計画ファイル（`plans/unused/`） |
| `mkplan_commands.sh`・`mkplan_output.txt` | 計画ファイルを作ったコマンド列とその出力 |
| `plan_check_output.txt` | 全計画ファイルの宣言（ベースモデル・変換器・加速）とトークン数の機械照合（失敗 0） |
| `preflight_output.txt`・`preflight.json` | 計測の終わりに走らせ直した机上検算（85 点・失敗 0）。着手前は 15 ファイル・35 点・失敗 0 |
| `preflight_perpoint_*_output.txt` | 途中で足した 1 点 1 ファイルの検算 |
| `harness_diff.txt` | 複製元との差分（T15・T16） |
| `git_state_start.txt` | 開始時の作業ツリー（dev・HEAD・未コミットのファイル） |
| `status_at_p0.json` | 着手前の `GET /status`（行列が空・`base_model` LTX25） |
| `git_state_end.txt` | 終了時の作業ツリー（監督役が最後に作る） |
| `runs/<ラベル>/`・`manifest.jsonl` | 点ごとの生データと、全点の要求と応答の記録（計測で作られる） |
| `calib/` | 待機時の床（`idle-calib` で作られる） |
| `commit_log.csv` | コミット上限の監視記録（2 秒おき・`commit_monitor.ps1` が作る。文書への複写はせず、点ごとの最大値を RESULTS 表5に） |
| `drive.sh`・`drive2.sh`・`phase_controls.sh`・`commit_monitor.ps1` | 較正台の外の駆動と監視のスクリプト（連続駆動・1 点ずつの駆動・制御段階・コミットの監視） |
| `drive*.log`・`phase_controls.log`・`fixup.log`・`x3.log`・`x4.log` | 駆動と待ち受けの記録 |
| `digest.py`・`digest.md`・`commit_table.py`・`commit_table.md` | 全 57 ランの一覧と、点ごとのコミットの表（読み取り専用の集計） |

| `load_*_output.txt`・`idle_*_output.txt`・`run_*_output.txt` | 変換器の切り替え・待機時の床・各点の実行の出力 |
| `report_output.txt`・`make_results_output.md`・`progress.md`・`restore_state_output.txt` | 計測後の集計・実行の日誌・復元の記録 |
| `original_state.json`／`luid.json`／`server.pid` | 開始時の選択の控え・GPU の識別子・登録したサーバーの PID（`server start` で作られる） |

**`tables/`（計画ファイルごとの表）は今回は作っていません。** `Docs/Outputs-archive/comfort-calib-2026-09-26/` への複写では、第6次の先例の `tables/` の代わりに `digest.md` と `commit_table.md` を複写しています（`commit_table.md` の Tier D の 2 行は完了後の値。第5節の 26）。

---

## 7. 使い方（監督役が打つコマンド）

作業ディレクトリはこのフォルダ、Python は `..\..\.venv\Scripts\python.exe`、環境変数 `PYTHONIOENCODING=utf-8` です。

```
# 着手前（GPU なし）
python selftest/parser_selftest.py
python selftest/analyse_selftest.py
python calib.py preflight

# サーバーの登録（GET /status が 200・queue.pending と queue.running が 0 のとき）
python calib.py server start --pid 49264

# 2.3 fp8
python calib.py load --base-model LTX23 --transformer sulphur_distil_fp8mixed
python calib.py idle-calib --minutes 3
python calib.py run --plan plans/plan_W_f_warm.json --job-timeout 3600 --cooldown 150
#   実際には 1 点 1 ファイルを drive2.sh で流した（点ごとにコミットの最大を確かめ、95% 以上なら次を投入しない）
bash drive2.sh drive2_A_f.log plans/p_f_1080_b89.json plans/p_f_1080_161.json

# Tier B 以降の計画ファイルの作り方（毎回 3 つの引数を省略しない）
python mkplan.py --out plan_B1_f_1080_169_r2.json --base-model LTX23 --transformer sulphur_distil_fp8mixed \
  --label-prefix f_s_1080 --width 1920 --height 1088 --frames 169 --suffix _r2 --baseline f_s_1080_b89 --accel allon --tier B
python calib.py preflight --plan plans/<新しいファイル>

# 集計と復元
python calib.py report
python calib.py restore-state --base-model LTX23
python calib.py restore-state --base-model LTX25
```
