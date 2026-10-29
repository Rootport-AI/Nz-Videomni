# §1-116 テキストエンコーダーの VRAM 計測（2026-10-29）

計画書 `clever-squishing-lake.md`（§3〜§6）の台本と記録です。T1〜T3（計測）は製品コードを変更しません。その結果を製品に取り込んだ実装後の門（G-I1〜G-I4）と実験（T4・T4b）の台本・記録も同じ場所に置きます。数値の正本は記録簿 `Docs/VERIFICATION_LOG.md` §181（計測）と §182（実装と門）に書きます（ここには置きません）。

## 中身

| ファイル | 内容 |
|---|---|
| `t1_sdpa_kernels.py` | T1。注意の演算 1 回（q (1,64,N,128)、k・v (1,8,N,128)、bf16、乱数 seed 0）をカーネルごとに 1 プロセスで測る。重みは読まない |
| `t2_te_ab.py` | T2・T3。テキストエンコーダー全体（50 層・直読み）を、テキスト側の注意関数だけ差し替えて走らせる |
| `run_common.ps1` | 起動の共通部（1 条件＝1 プロセス・時間切れでプロセスの木ごと停止・コンソールを `<名前>.console.log` へ） |
| `run_t1.ps1`・`run_t2.ps1`・`run_t3.ps1` | 計画の順に条件を並べたもの。条件つきの行はコメントにしてある |
| `g_r2_repro.cmd` | §1-116 実装後の GPU の門 G-I4。G-R2（画像 3 枚＋音声 1 本の Ref2VA）を同じ引数で生成し直す。画像は `Docs/Outputs-archive/h3-ref2va-2026-10-28/` に複製したもの |
| `run_gi.ps1` | §1-116 実装後の GPU の門 G-I1〜G-I3（第 1 回。端数の併合なし）。G-I1 → G-I2a（今の方式で動画 6 組の基準 `today_video6.pt` を作る）→ G-I2b → G-I3（画像 9 枚・動画 15 秒）の順に並べたもの。`run_common.ps1` を使う |
| `t4_tail_block.py` | T4・T4b。ダミーの注意（bf16・q＝(1,64,N,128)・k／v＝(1,8,N,128)）で、最後の塊の高さだけを変えて今の方式と `torch.equal` で比べる。重みは読まない。`--n`（N）・`--tails`（測る高さ。例 `17-23,64-575`）・`--skip-product`（製品の関数の測定を省く）。既定の高さの一覧は 1・2・4・8・12・16・24・32・48・51・64・96・128・256・512 |
| `run_gi2.ps1`・`g_r2_repro2.cmd` | 短い端数の塊の併合（計画書 §10）の後の GPU の門のやり直し（G-I1'〜G-I4'）。`run_gi.ps1`・`g_r2_repro.cmd` の複製で、条件名と出力名だけを `g_i1b_*`・`g_i2c_*`・`g_i3b_*`・`g_r2_repro2.*` に変えたもの（基準の `today_video6.pt` は作り直さない） |

## 実装後の結果ファイル（§182 の記録。数値は記録簿にだけ書く）

| ファイル | 内容 |
|---|---|
| `g_i1_pictures3_product.*`・`g_i2a_video6_today.*`・`g_i2b_video6_product.*`・`g_i3_pictures9_product_cap.*`・`g_i3_video15_product_cap.*` | 第 1 回の門（併合なし）。`g_i2b` は今の方式と一致しなかった記録で、不一致の証拠として残す |
| `g_i1b_pictures3_product.*`・`g_i2c_video6_product.*`・`g_i3b_pictures9_product_cap.*`・`g_i3b_video15_product_cap.*` | 第 2 回の門（64 行未満の端数を併合した後）。`g_i2c` は G-I2a の基準と一致 |
| `g_r2_repro.json`・`g_r2_repro.console.log` | G-I4。G-R2 の再現（第 1 回）。mp4 の sha256 を記録 |
| `g_r2_repro2.json`・`g_r2_repro2.console.log` | G-I4'。同（第 2 回） |
| `g_i5_t2v_fingerprint.cmd`・`.json`・`.console.log` | G-I5。参照なしの T2V の回帰指紋（設計正本 §8 の条件に `--vram16`）。sha256 が指紋と一致（記録簿 §182.7） |
| `t4_tail_block_n3123.*`・`t4_tail_block_n4100.*`・`t4_tail_block_n6148.*` | T4。塊の高さ（既定の一覧）ごとの一致。`product`（製品の関数の結果）・`position_check`（位置が無関係か）・`fold_check`（最後の 516 行を 1 つの塊にしたもの）の欄を持つ |
| `t4b_tail_block_n3123.*`・`t4b_tail_block_n4100.*`・`t4b_tail_block_n6148.*` | T4b。17〜23 行と 64〜575 行の全ての高さ（`tail_sweep` に 519 通り） |

- 各 `.json` は `.console.log` と対になる。`g_i*` は T2 と同じ欄を持つ（下の「JSON の主な欄」）。`g_r2_repro*.json` は `rounds[0]` に `sha256`・`size_bytes`・`done`（`h3.phases` の段ごとの記録）を持つ。
- G-I4 の画像 3 枚（`17fde0c1-image.png`・`d80320dd-image.png`・`31d79f30-image.png`）の保管庫への複製は、このフォルダーではなく **`Docs/Outputs-archive/h3-ref2va-2026-10-28/`** にある。音声（`outputs\h3-ref2va-2026-10-28\ref_audio_raincoat_speech.wav`）は git 管理外で、sha256 は記録簿 §182.8。
- 基準の隠れ状態（`today_a.pt`・`today_video6.pt`）は `outputs\h3-te-vram-2026-10-29\`（追跡外）。

## 走らせ方

リポジトリ直下で（場所はどこからでもよい。台本がリポジトリ直下へ移動する）:

```
powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-te-vram-2026-10-29\run_t1.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-te-vram-2026-10-29\run_t2.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-te-vram-2026-10-29\run_t3.ps1
```

- 先頭で `nvidia-smi` を表示し、AviUtl2 が起動していれば止まる（`-SkipPreflight` で無視）。
- 結果: このフォルダーに `<名前>.json` と `<名前>.console.log`。基準の出力（`.pt`）は `outputs\h3-te-vram-2026-10-29\`（追跡外）。
- 時間切れ: T1 は 600 秒（f0 は 300 秒）、T2 は 900 秒、T3 は 1,800 秒。時間切れのときは `.console.log` の末尾に `TIMEOUT`、`.json` は `{"timeout": true}`。
- 自己検査（CPU のみ・GPU と重みを使わない）: `set CUDA_VISIBLE_DEVICES=` の後で `t1_sdpa_kernels.py --selfcheck`、`t2_te_ab.py --selfcheck`。最後の行が `SELFCHECK PASS`。

単独で 1 条件を走らせる例（§1-116 以後、製品の `encode` は早期解放のフックを常に入れるので、`--drop-early` は実質つねに有効である。そのため §181 の「J なし」の行〔`t2_pictures3_f1`、`today_a` の復号器より前の山など〕は、今の HEAD からは測り直せない）:

```
set PYTHONPATH=%CD%
set PYTHONUTF8=1
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-te-vram-2026-10-29\t1_sdpa_kernels.py --n 3130 --kernel f1 --baseline outputs\h3-te-vram-2026-10-29\t1_today_n3130.pt
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-te-vram-2026-10-29\t2_te_ab.py --mode f1 --drop-early --refs pictures9 --cap
```

## カーネル・モード

| 名前 | 呼び方 |
|---|---|
| today | `F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)`（既定の選択。T2 では何も登録せず、テキスト側を標準の `sdpa` に戻す。製品は §1-116 からテキスト側を `H3_TEXT_ATTENTION` で組むため。製品の早期解放のフックは残るが、数値は変わらない） |
| math | today と同じ呼び方を `sdpa_kernel([MATH])` で（T1 のみ） |
| probe | T1: `can_use_efficient_attention`・`can_use_cudnn_attention` を `SDPAParams(q,k,v,None,0.0,True,True)`・debug で。T2: 素通しの関数で最初の呼び出しの引数を記録 |
| f0 | `sdpa_kernel([CUDNN_ATTENTION])`・`is_causal=True`・`enable_gqa=True` |
| f1 | k・v を `repeat_interleave(8, dim=1)` → `sdpa_kernel([EFFICIENT_ATTENTION])`・`is_causal=True` |
| g256・g512 | math のまま問い合わせを 256／512 行ずつ。全長の k・v に bool マスク片 `arange(s,e)[:,None] >= arange(N)[None,:]` |
| sage | `sageattn(q,k,v,tensor_layout="HND",is_causal=True,sm_scale=module.scaling)`（T2 のみ） |
| product | 何も登録しない。製品のテキストエンコーダーそのまま（§1-116 実装後の注意関数と早期解放）。T2 のみ |

`--pairs K`（`--refs video15` のときだけ）: 参照動画の組を先頭 K 組に絞る（既定は全 16 組）。

T2 の参照（`--refs`）と、自己検査で数えたトークン数 N: `pictures3`（画像 3 枚＋音声 1 本）N=3,123、`pictures9`（同じ 3 枚を 3 周）N=9,258、`video15`（16 組 × 1,008）N=16,314。

## JSON の主な欄

**T1（`t1_*.json`）**

| 欄 | 意味 |
|---|---|
| `ms`・`ms_all` | 時間（ミリ秒）。`--warmup` 回の空回しの後、`--repeat` 回の中央値と全部 |
| `inputs_alloc_mb` | 入力 q・k・v の割り当て |
| `peak_alloc_mb`・`peak_reserved_mb` | 入力を置いた後に山をリセットしてからの山（割り当て／予約） |
| `peak_alloc_over_inputs_mb` | 山から入力の分を引いたもの（演算そのものの山） |
| `vs_baseline` | `torch_equal`・`max_abs_diff`・`mean_abs_diff`・`mismatch_elements` |
| `probe.efficient`・`probe.cudnn` | `can_use`（真偽）・`warnings`（不適格の理由） |
| `sdp` | `is_flash_attention_available`・`mem_efficient_sdp_enabled`・`cudnn_sdp_enabled`・`fp16_bf16_reduction_math_sdp_allowed`・`sdp_priority_order(_names)`・TF32 の設定 |
| `error`・`ok` | 例外の理由。走れば `ok: true` |

**T2・T3（`t2_*.json`・`t3_*.json`）**

| 欄 | 意味 |
|---|---|
| `tokens`・`expected.tokens` | 実際の N と、encode 前に CPU で数えた N（`tokens_match_expected`） |
| `hooks.pre_decoder_peak_alloc_mb`・`_reserved_mb` | デコーダー前（埋め込み表＋視覚の塔）の山。層 0 の pre-hook で読み、そこで山をリセット |
| `decoder_peak_alloc_mb`・`decoder_peak_reserved_mb` | 層 0 以降（デコーダー）の山。層 0 に届かずに落ちたときは全体の山 |
| `hooks.min_free_pre_mb`・`min_free_post_mb`・`min_free_attn_mb` | `mem_get_info` の空きの最小値（各層の前・後、注意関数の直後。`min_free_attn_mb` は today では null） |
| `hooks.dropped` | J のフックで降ろしたもの（`embed_tokens`・`visual`） |
| `cap_info` | `--cap` の上限（`fraction`・`cap_mib`・`free_mib`・`reserved_mib`・`margin_mib`） |
| `allocator.num_ooms`・`num_alloc_retries` | encode の間に増えた回数 |
| `seconds` | `load`・`precount`・`encode`・`release`・`total` |
| `read_report` | テキストエンコーダーの記録（`read_gib`・`read_seconds`・`tokens`・`references` など） |
| `mem` | `_mem_gib`（プロセスの作業セット／コミット）。参考値 |
| `attention_calls` | 差し替えた関数が呼ばれた回数（50 のはず。today は 0） |
| `probe` | probe モードの記録（`attention_mask_is_none`・`kwargs_is_causal`・`module_is_causal`・形・`use_gqa_in_sdpa` など） |
| `vs_baseline` | `bit_identical`・`max_abs_diff`・`mean_abs_diff`・`min_token_cosine`（トークンごとのコサイン類似度の最小・float32）・`ids_identical`・`tags_identical` |
| `oom`・`error`・`ok` | 割り当て器の上限超過、その他の例外、完走して隠れ状態が有限なら `ok: true` |
