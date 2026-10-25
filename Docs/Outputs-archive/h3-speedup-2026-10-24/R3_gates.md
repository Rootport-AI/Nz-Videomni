# R3 台帳 §1-105 ①「QKV の活性化量子化の共有」GPU の門 A-4〜A-10

計測日 2026-10-25（08:40〜09:46）。対象はブランチ dev の作業ツリー（実装は未コミット。`git diff` の sha256 `b6e964da…`、変更 5 ファイル: `engine_h3/lora.py`・`lowvram.py`・`pipeline_h3.py`・`quant_linear.py`・`tests/test_h3_engine.py`）。GPU は RTX 4070 Ti SUPER 16GB、comfy-kitchen 0.2.37。製品のコードは変えていない。新しく作ったファイルと書き換えた計測台は、すべて `outputs/h3-speedup-2026-10-24/` の中にある。各回の前に `nvidia-smi --query-compute-apps` で Python のプロセスが無いことを確かめた（全回 0 個。`r3/*.precheck_compute_apps.csv`・`runs/R3_*.precheck_compute_apps.csv`）。

## 結論（全門の合否）

| 門 | 内容 | 結果 | 合否 |
|---|---|---|---|
| A-4 | `H3_GPU_TESTS=1` の `tests/test_h3_engine.py` 全件 | 162 passed・失敗 0（G1 の 8 通りを含む） | **合格** |
| A-5 | 製品の形の 50 ブロック比較＋呼び出しの計数 | sdpa 50/50 一致。sage 48/50 で、違うのはブロック 22・40 だけ、値は前回と全桁同じ。on の回で `forward_rows` 0 回・`forward_rows_q` 1,050 回（sage・sdpa それぞれ） | **合格**（台本の写しの修正 1 か所あり。下の注） |
| A-6 | 指紋 g3 の off（sdpa・640×384・124 フレーム・20 ステップ・seed 42） | sha256 `0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9` 全桁一致。on は A-7 の g2 で同じ値 | **合格** |
| A-7 | LoRA の門の on 側 6 本 | 6 本とも sha256 が前回と全桁一致。6 本ともログに共有の行、report の `qkv_quant_shared` が `true` | **合格** |
| A-8 | 前後の速度（I2V 896×1152×124・3 ステップ） | 負荷の無い 2・3 ステップ目の平均は 前 21.868 秒（4 個）・後 20.337 秒（2 個）、差 **1.531 秒**。同じ静かな時間帯の前 1 回（pre3）だけと比べると差 1.398 秒。mp4 の md5 は 6 回とも `eb2a53ae…` | **合格** |
| A-9 | VRAM のピーク（A-8 の記録） | ノイズ除去の割り当て／予約のピーク: 前 4,411／7,262 MiB、後 4,379／7,262 MiB。OOM・再試行 0 | **合格** |
| A-10 | 長尺 T2V 1280×768×311（門の外・記録だけ） | 2・3 ステップ目の平均 前 74.435 秒 → 後 71.360 秒（−3.075 秒）。md5 は前後で同じ | 記録 |

---

## A-4 GPU テスト

- **条件**: `H3_GPU_TESTS=1 PYTHONUTF8=1 .venv-engine-h3\Scripts\python.exe scripts\run_h3_pytest.py tests\test_h3_engine.py -rA`。
- **結果**: 162 passed・失敗 0・34.1 秒。G1 `test_gpu_shared_quant_join_equals_forward` は 8 通り（bias の有無 × `_DISABLE_CUTLASS_INT8` の真偽 × m 4,096／39,764）がすべて合格。C1〜C6（`test_shared_quant_symbols_of_the_pinned_comfy_kitchen`・`test_shared_quant_probe_never_raises_and_says_why`・`test_vram16_falls_back_when_not_shared`・`test_lora_vram16_shared_matches_off` の 3 通りほか）も合格。
- ログ行 `int8: qkv quantization shared across head groups` の確認は監督の決定どおり A-4 から外し、A-6／A-7 で確かめた（下）。
- **合否**: 合格。
- **成果物**: `r3/a4_gpu_pytest.txt`。

## A-5 製品の形の 50 ブロック比較

- **条件**: 包みの台本 `a5_g2_wrapped.py`。(1) `require_cuda_int8()` を呼んでから `shared_quant_status() == (True, None)` を確かめる（実測で `(True, None)`。終わりの時点でも同じ）。(2) `H3Int8Linear.forward_rows`・`forward_rows_q`・`quantize_input`・`forward` を、数えるだけで同じ引数で元を呼ぶ包みに差し替える。数えるときは `H3DiTBlock.forward` も包み、そのブロックの `attn.attention_backend` と `vram16` が None かどうかから「バックエンド/on か off」を読んで分けた（g2 は 1 ブロックごとに sage と sdpa の両方で `block.vram16 = None`（off）→ `Vram16Plan()`（on）の順に回す。台本を読んで確認した）。(3) `runpy.run_path` で g2 を `--blocks 0-49 --report outputs/h3-speedup-2026-10-24/a5_g2_block_equal.json` で呼ぶ。S＝39,764、ヘッド群 8・行の塊 4,096。
- **注（台本の写しの修正）**: 元の `outputs/h3-vram16-gates/g2_block_equal.py` は、今の製品のコードでは動かなかった。`TypeError: build_coords() got an unexpected keyword argument 'cond_frames'`（`r3/a5_g2_wrapped.try1_build_coords_error.log`）。`engine_h3/rope.build_coords` の引数は §1-97（コミット `66761ec`）で `cond_frames` から `cond_frame_idx`（キーフレームの画素番号の並び）に変わっている。製品の不具合ではなく台本が古いだけなので、元は変えずに写し `g2_block_equal_r3.py` を作った。変えたのは `cond_frames=1` → `cond_frame_idx=(0,)` と、`--report` を必須にした（既定のままだと前回の正本を上書きする）ことの 2 点だけ。先頭フレーム 1 枚の条件の行の RoPE の時刻は `keyframe_t(0, …)`＝L で、前と同じ（下の sage の数値が前回と全桁一致したことでも裏付けられる）。取り直しはこの 1 回だけ。
- **結果**:

| 項目 | 今回 | 前回 `g2_block_equal.json` |
|---|---|---|
| sdpa | 50/50 一致 | 50/50 |
| sage | 48/50 一致。違うのはブロック 22・40 | 同じ |
| ブロック 22（sage） | max_abs 256.0・elements_differ 115,388,441・rel_out 0.00659228097488087・rel_delta 0.006592545742312935 | 全桁同じ |
| ブロック 40（sage） | max_abs 1792.0・elements_differ 6,974,901・rel_out 0.0030670819750614436・rel_delta 0.0030670851576144968 | 全桁同じ |
| 全ブロックの有限性 | 両方とも真 | 同じ |
| 割り当てのピーク | 9,994 MB | 9,994 MB |
| 所要時間 | 216.3 秒 | 209.2 秒 |

呼び出しの回数（50 ブロック。`a5_counts.json`）:

| 区分 | `forward_rows` | `forward_rows_q` | `quantize_input` | `forward` |
|---|---|---|---|---|
| sage/on | **0** | **1,050**（＝50 × 21） | 50 | 1,050 |
| sdpa/on | **0** | **1,050** | 50 | 1,050 |
| sage/off | 0 | 0 | 0 | 200 |
| sdpa/off | 0 | 0 | 0 | 200 |

`forward` の on の 1,050 回は、1 ブロックで out_proj 1＋fc1 の塊 10＋fc2 の塊 10＝21 回（qkv の 21 回とたまたま同じ数）。off の 200 回は 1 ブロック 4 回（qkv・out_proj・fc1・fc2）。

- **合否**: 合格。
- **成果物**: `a5_g2_wrapped.py`・`g2_block_equal_r3.py`・`a5_g2_block_equal.json`・`a5_counts.json`・`r3/a5_g2_wrapped.log`・`r3/a5_g2_wrapped.try1_build_coords_error.log`。

## A-6 指紋 g3（off）

- **条件**: `a6_g3_off.sh`（`outputs/h3-vram16-gates/g3.sh` の写しで、off だけ・出力先を `r3/` に変えた）。ワーカーのセルフテスト、sdpa・640×384・124 フレーム・20 ステップ・seed 42。on は A-7 の g2（`--vram16`）と同じ計算なのでそちらを使った。
- **結果**: off の sha256 `0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9`（全桁一致）。ノイズ除去 109.01 秒。on（A-7 の g2）も同じ sha256、report の `qkv_quant_shared` は `true`。
- off の回のログにも `int8: qkv quantization shared across head groups (comfy-kitchen internals found)` が出る（検査は load の時に VRAM 16GB mode の on／off と関係なく走るため。コードの事実どおり）。off の report には `vram16` の記録自体が無い（`vram16` が None）。
- **合否**: 合格。
- **成果物**: `a6_g3_off.sh`・`r3/a6_g3_off.json`・`r3/a6_g3_off.log`・`r3/a6_g3_off.mp4`。

## A-7 LoRA の門の on 側 6 本

- **条件**: `a7_gates.sh`（`Docs/Outputs-archive/h3-lora-2026-10-24/gates/gates.sh` の写し。変えたのは出力先 `OUT=outputs/h3-speedup-2026-10-24/r3/a7_gates` と 2 行目の説明だけ）を `a67_chain.sh` から 1 本ずつ回した。sdpa・決定性 off・seed 42。
- **結果**（sha256 は全桁を前回の `gates/*.json` の `rounds[0].sha256` と突き合わせた。ノイズ除去の秒数は参考）:

| 門 | 名前 | sha256（全桁一致） | ログの共有の行 | `qkv_quant_shared` | ノイズ除去 今回／前回（秒） | 割り当て／予約のピーク（MiB。今回＝前回） |
|---|---|---|---|---|---|---|
| g2 | g2_nolora_on | `0da6111cdca853ce…` 一致 | あり | true | 107.32／113.54 | 3,779／3,790 |
| g4b | g4_turbo_640_on | `0caa05dcde6e48a8…` 一致 | あり | true | 39.31／39.29 | 4,071／4,098 |
| g5b | g5_style_on | `c05bca7bcbbc803a…` 一致 | あり | true | 126.34／129.40 | 3,807／3,826 |
| g5d | g5_motion_on | `a9df669270612d50…` 一致 | あり | true | 125.98／130.44 | 3,799／3,808 |
| g5f | g5_mixed_on | `f48a4c2a064fc861…` 一致 | あり | true | 133.03／137.97 | 3,835／3,844 |
| g5g | g5_style_on_noprefetch | `c05bca7bcbbc803a…` 一致 | あり | true | 229.34／231.67 | 2,781／3,646 |

「ログの共有の行」は `int8: qkv quantization shared across head groups (comfy-kitchen internals found)`（各 `*.console.log`）。どの回にも `NOT shared` の行は無い。割り当て／予約のピークは 6 本とも前回と同じ値。

- **合否**: 合格。
- **成果物**: `a7_gates.sh`・`a67_chain.sh`・`r3/a7_gates/*.json`・`*.console.log`・`*.mp4`・`r3/a67_chain.log`。

## A-8 前後の速度

- **条件**: 計測台 `measure_h3_speedup.py` に `--no-qkv-share` を足した（下の「計測台の変更」）。腕の台本 `a8_runs.sh`（`arm_n3.sh` と同じ引数: I2V 896×1152×124・sage・`--vram16`・`--keep-resident`・`--deterministic`・先読み on・`--blocks-on-gpu 8`・`--steps 3`・seed 42・`f0_prompt.txt`・`--vram16-head-group 8 --vram16-row-chunk 4096 --vram16-blocks 4`）。腕の間の冷まし方は `arm.sh` と同じ（`arm_r3.sh`。30 秒待ち＋使用量 1,200 MiB 未満まで待つ。足したのは直前の Python プロセスの確認だけ）。同じセッションで 前1→後1→前2→後2。負荷の判定は `step_rx.py`（そのステップの壁時計の窓の dmon の受信の中央値。300 MB/s 超を「負荷あり」）。
- **取り直し**: 最初の 4 回のうち 後1・前2・後2 は全ステップで外部の負荷（受信の中央値 約 470 MB/s）があった。後の側の負荷の無い 2・3 ステップ目が 0 個になったので、計画どおり後を取り直した（後3）。外部の負荷が消えるのを待つ台本 `r3/wait_quiet.sh`（20 秒の dmon で受信の中央値 300 MB/s 未満かつ SM の中央値 10% 未満）で約 5 分待ってから回した。続けて、同じ静かな時間帯で前も 1 回足した（前3。前の側は前1 だけで 2 個あり計画上は不要だが、隣り合う時間帯どうしで比べるための追加）。

ステップごとの秒数（秒）と受信の中央値（MB/s）:

| 回（ラベル `R3_A8_<回>_h8_r4096_b4`） | 種類 | 1 ステップ目 | 2 ステップ目 | 3 ステップ目 | 受信の中央値（1／2／3） | 負荷 |
|---|---|---|---|---|---|---|
| pre1 | 前 | 22.167 | **22.034** | **21.968** | 126／23／5 | なし（ただし 2・3 ステップ目に 300〜2,000 MB/s の標本が 3 個ずつ。下の注） |
| post1 | 後 | 21.547 | 21.374 | 21.385 | 472／474／471 | **あり** |
| pre2 | 前 | 23.020 | 22.823 | 22.853 | 469／475／474 | **あり** |
| post2 | 後 | 21.545 | 21.381 | 21.436 | 470／479／473 | **あり** |
| post3（取り直し） | 後 | 20.512 | **20.326** | **20.348** | 6／5／6 | なし |
| pre3（追加） | 前 | 21.893 | **21.719** | **21.750** | 5／6／6 | なし |
| 参考: R2 の N3 | 前と同じ製品のコード | 21.858 | 21.641 | 21.684 | 5／5／5 | なし |

比較（負荷の無い 2・3 ステップ目の平均。太字の値）:

| 比べ方 | 前 | 後 | 差（前 − 後） |
|---|---|---|---|
| 計画どおり（前は pre1・pre3 の 4 個、後は post3 の 2 個） | 21.868 秒 | 20.337 秒 | **1.531 秒** |
| 同じ静かな時間帯の隣どうし（pre3 と post3） | 21.735 秒 | 20.337 秒 | 1.398 秒 |
| 参考: R2 の N3 と post3 | 21.663 秒 | 20.337 秒 | 1.326 秒 |
| 参考: 負荷ありどうし（pre2 と post1・post2 の 2・3 ステップ目） | 22.838 秒 | 21.394 秒 | 1.444 秒 |

- 注: pre1 は判定の規則では「負荷なし」だが、2・3 ステップ目の窓に 300〜2,000 MB/s の標本が 3 個ずつ、読み込み前の待機の窓にも 115〜314 MB/s の標本があり、弱い負荷が混じっていた可能性がある（推測）。pre3 より 0.27 秒遅い。そのため計画どおりの差 1.531 秒は R2 の上限 1.447 秒を少し超えている。pre3 だけと比べた 1.398 秒は上限の内側で、こちらが実態に近いと見られる（推測）。どちらの比べ方でも合格条件（1.0 秒以上）は満たす。
- 生成物の md5: 6 回とも `eb2a53ae336e4eb99dc6ef8ddcad6bb7`（R2 と一致）。
- report の `h3.phases.denoise.vram16.qkv_quant_shared`: 後（post1〜3）は 3 回とも `true`、前（pre1〜3）は 3 回とも `false`。ログは後に `int8: qkv quantization shared across head groups (comfy-kitchen internals found)`、前に `NOT shared (disabled by --no-qkv-share)`（差し替えが効いたことの確認）。
- 実際に効いた値の食い違い（`effective_mismatch`）・例外は 6 回とも無し。
- **合否**: 合格（(1) 負荷のある回を除いて前 4 個・後 2 個、(2) 差 1.531 秒（隣どうしでは 1.398 秒）≥ 1.0 秒、(3) md5 が全回で R2 と一致、(4) 記録の真偽が前後どおり）。
- **成果物**: `measure_h3_speedup.py`（変更後）・`r3/measure_h3_speedup.before_r3.py`（変更前の写し）・`arm_r3.sh`・`a8_runs.sh`・`r3/wait_quiet.sh`・`runs/R3_A8_*`（run.json・dmon.txt・stdout.log・out.mp4 ほか）・`r3/a8_summary.py`・`r3/a8_summary.json`・`r3/a8_step_rx.jsonl`・`r3/a8_runs.log`・`r3/a8_retake.log`。

### 計測台の変更（`measure_h3_speedup.py`）

`--no-qkv-share` を足した。読み込み（`worker._do_load`）の前に、`engine_h3.quant_linear._probe_shared_quant` を「`_SHARED_CB = None`・`_SHARED_REASON = "disabled by --no-qkv-share"` にするだけの関数」に差し替える。`require_cuda_int8` は末尾で `_probe_shared_quant()` をモジュールの大域の名前で呼ぶ（`quant_linear.py:359`）ので、モジュールの属性の差し替えで効く。ログの `NOT shared (disabled by --no-qkv-share)` と report の `false` で効いたことを確かめた。フラグを付けなければ元と同じ動き。

## A-9 VRAM（A-8 の記録）

| 回 | 割り当てのピーク（MiB） | 予約のピーク（MiB） | `inactive_split` のピーク（MiB） | OOM | 再試行 | WDDM 共有メモリのノイズ除去中の最大（MB） | コミットのピーク（GiB） |
|---|---|---|---|---|---|---|---|
| 前（pre1・pre2・pre3） | 4,411 | 7,262 | 952 | 0 | 0 | 1,212〜1,272 | 35.25 |
| 後（post1・post2・post3） | **4,379** | **7,262** | 964 | 0 | 0 | 1,208〜1,254 | 35.25〜35.26 |
| 参考: R2（5 回） | 4,411 | 7,262 | 952 | 0 | 0 | — | — |

- 後の割り当てのピークは前より 32 MiB 低い。予約のピークは同じ。前の値は R2 と全桁同じ。
- 「退避」は、この計測台の記録にある OOM（`num_ooms`）・再試行（`num_alloc_retries`）と、WDDM の共有メモリ（GPU のメモリからシステムのメモリへの溢れ）で見た。共有メモリは前後で同じ水準（約 1.2 GB。R2 の N3 の窓の最大 1,194 MB とも同じ水準で、溢れの増加は無い）。
- `inactive_split` のピーク（記録だけ）は 952 → 964 MiB。
- **合否**: 合格。
- **成果物**: A-8 と同じ（`r3/a8_summary.json`）。

## A-10 長尺 T2V（門の外・記録）

- **条件**: `a10_runs.sh`（`outputs/h3-vram16-gates/g4.sh` の 2 行目の写し。違いは `--steps 3` と前の `--no-qkv-share`）。T2V 1280×768×311・S＝89,388・sage・`--vram16`・常駐・決定性・先読み・`--blocks-on-gpu 8`（実際は 4）。前 → 後の 1 回ずつ。どちらの回も外部の負荷なし（受信の中央値 2〜5 MB/s）。
- **結果**:

| 回 | 1 ステップ目 | 2 ステップ目 | 3 ステップ目 | 割り当て／予約のピーク（MiB） | `inactive_split`（MiB） | OOM・再試行 | md5 |
|---|---|---|---|---|---|---|---|
| 前 | 75.116 | 74.444 | 74.426 | 7,062／11,382 | 1,634 | 0・0 | `a7bfab231e1863bea36d3e5383cdb8e0` |
| 後 | 71.592 | 71.372 | 71.347 | 6,605／11,382 | 1,665 | 0・0 | `a7bfab231e1863bea36d3e5383cdb8e0` |

- 2・3 ステップ目の平均で 74.435 → 71.360 秒、**1 ステップあたり 3.075 秒の短縮**（見込み 約 3.1 秒。M0 の 3.138 ms × 1,000 回の 20/21＝2.99 秒とほぼ同じ）。生成物は前後でビット一致。割り当てのピークは 457 MiB 下がり、予約のピークは同じ。
- 前の回の割り当て／予約のピーク・`inactive_split` は前回の A-G4（2 ステップ）と全桁同じ。2 ステップ目の秒数は A-G4 の 76.237 秒より 1.8 秒速い（理由は未確認。A-G4 の時に弱い外部の負荷があったかは今回は確かめていない）。md5 はステップ数が違う（A-G4 は 2 ステップ）ので A-G4 と比べられない。
- **成果物**: `a10_runs.sh`・`runs/R3_A10_T2V1280x768x311_{pre,post}/`・`r3/a10_summary.json`・`r3/a10_runs.log`。

---

## 計測中に気づいたこと

**計画・R1・R2 と食い違う点、新しい事実**

1. **`outputs/h3-vram16-gates/g2_block_equal.py` は今の製品のコードでは動かない。** `rope.build_coords` の引数が §1-97（`66761ec`）で `cond_frames` → `cond_frame_idx` に変わったため。計画の「`g2_block_equal.py` は変えず」は、元を変えず写し（`g2_block_equal_r3.py`。引数名の 1 か所と `--report` の必須化だけ）で守った。今後この台本を使う門は同じ写しを使うこと。製品の不具合ではない。
2. **A-8 の差は計画どおりの集計で 1.531 秒で、R2 の上限 1.447 秒をわずかに超えた。** 原因は前の側の pre1 に弱い外部の負荷が混じっていたためと見られる（推測。標本の根拠は A-8 の注）。同じ静かな時間帯の隣どうし（pre3 と post3）では 1.398 秒（上限の 97%）。負荷ありどうしでも 1.444 秒。① の短縮は、R2 の上限（量子化の核 20/21 回分）をほぼ全部取れている。
3. 割り当てのピークは前より下がった（I2V 896×1152 で −32 MiB、長尺 T2V で −457 MiB）。予約のピークは変わらない。計画 A-9 の「ピークは動かない可能性が高い（推測）」に対して、予約は推測どおり、割り当ては少し下がった。
4. 20 ステップの g2（640×384）のノイズ除去は 113.54 → 107.32 秒（−6.2 秒。約 0.31 秒／ステップ。小さい画角なので量子化の分も小さい）。Style・Motion・混合の LoRA の on も 3〜5 秒短くなった。Turbo 4 ステップ（g4b）は 39.31／39.29 秒で差は見えない。
5. ログ行 `int8: qkv quantization shared across head groups (comfy-kitchen internals found)` は、VRAM 16GB mode が off の回（A-6）にも出る。検査が load の時に on／off と関係なく走るため（コードの事実どおり。計画の書き方と矛盾はしないが、「on の証拠」にはならず、on の証拠は report の `qkv_quant_shared` と A-5 の計数）。
6. 実装担当の `a4_pre_gpu_tests.txt` にもすでに GPU テストの合格（162 passed）が記録されていた。今回の A-4 も同じ 162 件。

**次のセッションのための注意**

7. **外部の GPU の負荷が今回も出た。** 09:09〜09:24 ごろ、SM 約 25%・PCIe の受信 約 440〜830 MB/s の定常の流れが続き、その間のステップは前で約 1.1 秒、後で約 1.0 秒遅くなった。09:20 ごろ Windows の性能カウンター（`\GPU Engine(*)\Utilization Percentage`）を見たところ、負荷の大きい順に dwm（3D 14.7%）と chrome（PID 26284。videoprocessing 9.9%・videodecode 7.7%・3D 5.6%）だった。ブラウザーの動画の再生と見られるが、これが受信の流れの出どころかは確かめていない（推測）。その時の `nvidia-smi --query-compute-apps` に Python のプロセスは無かった。速さの A/B では `r3/wait_quiet.sh` で負荷が消えるのを待ってから回し、`step_rx.py` で各ステップを確かめるのがよい。
8. 孤児プロセス: 全回の前後で Python のプロセスは 0 個（最後の回の後も 0 個）。計測台を止めずに残したプロセスも無い。
9. Python の `Path.write_text` で書いた台本は Windows では CRLF になり、bash の台本は CRLF だと動かない（`arm_r3.sh` は LF に直してから使った）。元の `gates.sh` は CRLF のまま動いている形跡があるが、写しは LF で作った。

## 成果物の一覧（すべて `outputs/h3-speedup-2026-10-24/`。git 管理外）

| ファイル | 中身 |
|---|---|
| `R3_gates.md` | 本書 |
| `a5_g2_wrapped.py`・`g2_block_equal_r3.py`・`a5_g2_block_equal.json`・`a5_counts.json` | A-5 の包みの台本・g2 の写し・結果・呼び出しの計数 |
| `a6_g3_off.sh`・`a7_gates.sh`・`a67_chain.sh` | A-6・A-7 の台本（写し）と連続実行 |
| `measure_h3_speedup.py` | 計測台（`--no-qkv-share` を足した）。変更前は `r3/measure_h3_speedup.before_r3.py` |
| `arm_r3.sh`・`a8_runs.sh`・`a10_runs.sh` | A-8・A-10 の腕の台本 |
| `runs/R3_A8_*`・`runs/R3_A10_*` | 各回の run.json・dmon.txt・wddm.csv・stdout.log・memory_stats・out.mp4・console.log・precheck |
| `r3/` | ログ（`r3.log`・`a4_gpu_pytest.txt`・`a5_g2_wrapped*.log`・`a67_chain.log`・`a8_runs.log`・`a8_retake.log`・`a10_runs.log`）、A-6 の json／log／mp4、A-7 の `a7_gates/`、集計（`a8_summary.py`・`a8_summary.json`・`a8_step_rx.jsonl`・`a10_summary.json`）、補助（`precheck.sh`・`wait_quiet.sh`）、各回の precheck の csv |
