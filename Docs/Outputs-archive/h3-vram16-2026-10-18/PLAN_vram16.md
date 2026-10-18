# 計画: MiniMax H3「VRAM 16GB mode」の設計と実装（台帳 §1-101 の第 2 段〜）

作成 2026-10-18 夜。監督＝Fable 5.1。前段（第 1 段・計測）は `Docs/VERIFICATION_LOG.md` §168 と `Docs/Outputs-archive/h3-vram-2026-10-18/`。棚卸し E1（エンジン）・E2（アプリ側）・E3（操作パネル）、設計担当の検討（Plan agent・Opus）、敵対的レビュー（Opus。Critical 1・Major 8・minor 13・**全件採用**。観点＝過剰設計の危険／裏付けの有無）を反映済み。プランモードのため報告書ファイルは無く、実装の前に本文の要点を scratchpad\h3vram\design\ と保管庫へ写す。

## 0. 文脈

- 製品の条件（I2V 896×1152×124f・系列 S＝39,764 行）でノイズ除去の予約が 16,828MiB と物理 16,376MiB を超え、約 3.3GB が共有メモリへ退避し、2 ステップ目から 1 ステップ約 200 秒（正常は 20〜30 秒）。原因は実測で確定（§168）: 割り当て器が返さない空きセグメント＋ sage 呼び出し時の活性化の山（U＝bf16 [S,5376]≈408MiB。山の内訳の表は実装時に設計正本へ置く）。固定費は先読みの受け皿 10 本 3,711MiB・token_refiner 1,470MiB。
- 目標（オーナー）: **退避ゼロ**＝予約＋他プロセス ≤ 物理 16GB・余裕つき。精度を落とす手段は不可。既存経路は維持し、新経路を「VRAM 16GB mode」として on／off。
- 先行実装（Zironic/H3-Optimizations〔LICENSE なし〕・kijai/ComfyUI-KJNodes〔GPL-3.0〕）は**考え方のみ**。コードは写さない（オーナー指示）。外部レビューの指摘は全件採用済み。

## 1. オーナー裁定（番号は研究ノート §10 の続番 44〜52。設計の規則にあたる 44〜47・51 は設計正本 §2 へ、進め方の 48〜50・52 は研究ノート §10 へ）

- 44: 名称「VRAM 16GB mode」。Settings「Acceleration／生成の高速化」のトグル。**H3 では既定 on**、off も選べる。Gradio と MCP にも出す。値はジョブごとに毎回送る。
- 45: 同じワーカーの中の切り替え。off は現行経路に一切触れず（基準の指紋 `0da6111c…` の不変で担保）、on が新経路。ワーカーや仮想環境は複製しない。
- 46: 数値の約束。目標はビット一致。門で最終桁の差が出たときは、ブロックごとの最大差がゆらぎ級であることとオーナーの目視で品質差が無いことの 2 つで採否を決める。
- 47: OOM は (a)＝ジョブを失敗させ「解像度かフレーム数を下げるか、VRAM 16GB mode を切る」と案内（「画角」と書かない）。黙って退避を許す経路は作らない。既知の限界: 上限はこのプロセスの割り当てだけを縛るので、ジョブ中に他プロセスが VRAM を増やした場合の退避は防げない。
- 48: 目標の尺度は台帳 §1-101 のコマ数のまま。生成時間は本テーマの対象外。
- 49: 実装担当は設計に疑問があれば作業を止めてエスカレーションする権利を持つ。設計の変更は禁止。
- 50: 計画の敵対的レビューは「過剰設計やスパゲッティ化の危険」「調査で裏付けが取れるか」の 2 点で採否を決める。
- 51: 分割の単位（ヘッド群・行の塊）と常駐数は形状だけから決める決定的な規則。割り当て器の上限だけは実行時の値から導いてよい。
- 52: PinkCherry Hybrid が置くだけで動いた事実を研究ノート §13.3 と設計正本に 1 行ずつ記録。

## 2. on の経路で行うこと（削減は「予約」と「割り当て」を分けて評価し、合計は目標値であって予算ではない。#5・#6 は山の区間ではないので山を下げる効果は持たない）

| # | 単位 | 何をするか | 等価性の見込み |
|---|---|---|---|
| 1 | 割り当て器の上限 | ノイズ除去の段の頭（`_place_dit` の前・`try` の内側）で `mem_get_info` の総量と空きから上限を導き `set_per_process_memory_fraction` で課す。`finally` の先頭で 1.0 に戻す。再試行・OOM の回数の**差分**と `inactive_split` のピークを `h3.phases.denoise.vram16` に記録。OOM は裁定 47 | ビット一致（数値に触れない） |
| 2 | 早期解放 | q・k・v を重みの行で切って射影（融合 qkv を作らない）。`tokens` は容れ物で渡し、bf16 化の直後に `del`（参照の持ち主: `_denoise` の局所変数・`nn.Module._call_impl` の args・forward の引数。swap の包みの args はそのブロックの呼び出しの間だけ）。norm 後の `h` は最後のヘッド群の後に捨てる | ビット一致 |
| 3 | ヘッド分割 | 注意計算をヘッド群（`VRAM16_HEAD_GROUP`＝8）ごとに回し、既存の `_attention(q_g, k_g, v_g, backend)` を呼ぶ（写し・sage／SDPA の形は off と同じ） | ビット一致の見込み。**既知の危険**: sage の `k.mean`（`sageattention/core.py:724-725`）は出力数が変わると集約の順序が変わりうる（推測）→ A-G2 で判定（裁定 46） |
| 4 | FFN の行の塊化 | `row_chunks(S)`＝塊の数 `ceil(S/4096)` で**等分**（端の塊も 129 行以上になり、`int8_linear` の m≤2 と Turing の m≤128 の別経路を避ける。S ≤ 4,096 なら分割しない）。塊ごとに既存の `block.mlp(h2[r0:r1])` を呼び受け皿へ | ビット一致の見込み（行ごとの量子化・int32 累積。CUTLASS が引き受けるかは形依存なので製品の形で門） |
| 5 | AdaLN の塊化 | 既存の `_modulate`／`_gate` を行の塊ごとに呼ぶだけ（書き写さない） | ビット一致（同じ関数） |
| 6 | 最終層 | 行の塊で `fl.norm`→`_modulate`（bf16）→ f32 の連続バッファへ `copy_`、`_linear_f32` を 1 回（行列積は分割しない） | ビット一致 |
| 7 | 退避 | `refine_text` の直後に token_refiner と `condition_proj` を **常駐ジョブなら CPU へ（`move_module_tree`）、非常駐ジョブなら meta へ落として捨てる（`drop_to_meta` の流用）**。主メモリのコミット量の増加（常駐時 約 1.5GB・推測）は A-G4 で F0 と比べる | ビット一致 |
| 8 | 常駐数 | `_place_dit(dit, prefetch, blocks_on_gpu)` に引数を足し、on／off を問わず毎回 `service.blocks_on_gpu = n` を `install` の前に代入（反映は `block_swap_service.py:233, 317` で確認済み）。`n = blocks_on_gpu_for(S, n_load)`: `n_load ≤ 0` か `≥ 50`（全常駐）ならそのまま、off ならそのまま、on で S ≥ しきい値なら `min(n_load, 4)`。**A-G5 で 4 対 8 に時間差が無ければしきい値を削除し、on は常に `min(n_load, 4)`** | ビット一致 |
| 9 | 範囲 | on の q/k/v は群ごとに連続で最大オフセットは S·g·D（群 8 で S≈209 万行まで安全）。防御関数は作らない。off 経路の 10 万行超の限界は設計正本 §13 に記録 | — |

- 採らないもの: Q だけの流し込み・q,k の strided 渡し・2-slice MLP・融合 SwiGLU・RMSNorm＋RoPE 融合・`smooth_k` 無効化・VAE の fp16 累積・CUTLASS 設定の強制。

## 3. エンジン側の設計

### 3.1 置き方（スパゲッティ化を避ける: 新しい計算を書かず、既存の関数を範囲ごとに呼ぶ）

- 新モジュール **`engine_h3/lowvram.py`**: `Vram16Plan`（dataclass: `head_group`・`row_chunk`。attention の方式は `block.attn.attention_backend` を読む）・`row_chunks(S, chunk) -> list[(r0,r1)]`（等分）・`blocks_on_gpu_for(S, n_load)`・`derive_memory_cap(free, total, reserved, margin) -> fraction`（純粋関数。clamp は `min(1.0, max(0.0, ·))` だけ）・`block_forward(block, x, t_emb, adaln_index, rope_cos, rope_sin)`・`final_rows_chunked(...)`。先頭で `from engine_h3 import dit as _dit` とし、`_dit._modulate`／`_dit._gate`／`_dit._attention`／`_dit._linear_f32`／`_dit.apply_rope` を呼ぶ（テストでの差し替えが効く）。`torch.no_grad()` の中で呼ばれる前提（`inference_mode` という語はコメントにも書かない）。
- `dit.py` の変更は分岐と引数だけ: `H3DiTBlock.vram16 = None`（クラス属性）、`forward` の先頭 `if self.vram16 is not None: from engine_h3 import lowvram; return lowvram.block_forward(self, x, t_emb, adaln_index, rope_cos, rope_sin)`（遅延 import はこの片側だけ）。`H3DiT.forward(tokens, *, ..., vram16=None, tokens_box=None)`: `tokens is None` なら `tokens_box.pop()`（形の検査より前）、ループで `block.vram16 = vram16` を**毎回**代入（None も。常駐 DiT の後始末）、`x = tokens.to(bf16)` の直後に on なら `del tokens`。`final_rows(..., vram16=None)` の入口で分岐。off の経路の行は変えない。
- `H3Int8Linear.forward_rows(x, n0, n1)`: `self.weight[n0:n1]`・`self.weight_scale[n0:n1]`（連続ビュー。n0 は 128 の倍数）・`self.bias[n0:n1]`（あれば）を `int8_linear` に渡す。毎回 `self.weight` から切る。
- 定数は `engine_h3/constants.py`: `VRAM16_HEAD_GROUP=8`・`VRAM16_ROW_CHUNK=4096`（FFN・AdaLN・最終層で共通）・`VRAM16_BLOCKS_ON_GPU=4`・`VRAM16_BLOCKS_S_THRESHOLD`（初期値 16,384。A-G5 で削除の可能性）・`VRAM16_CAP_MARGIN_MIB=1024`。

### 3.2 ブロックの on 経路（`lowvram.block_forward`）

1. `shift…gate_mlp = block.adaln_proj(t_emb)`。
2. `n1 = block.norm1(x)`；`h = empty_like(n1)`；塊 r ごとに `h[r] = _modulate(n1[r], shift_msa, scale_msa, adaln_index[r])`；`del n1`。
3. `inner = attn.heads*attn.head_dim`、受け皿 `o = empty(S, inner)` bf16。ヘッド群（h0..h1、g 個）ごとに `q = attn.qkv_proj.forward_rows(h, 0*inner + h0*D, 0*inner + h1*D).view(S, g, D)`、k・v も同様（`part*inner` の基点）→ `attn.q_norm(q)`／`attn.k_norm(k)` → `apply_rope` → `out = _attention(q, k, v, attn.attention_backend)`（`[S, g, D]`）→ `o[:, h0*D:h1*D].copy_(out.reshape(S, g*D))`。最後の群の後に `del h`。`a = attn.out_proj(o)`。
4. `x2 = empty_like(x)`；塊ごとに `x2[r] = _gate(x[r], gate_msa, a[r], adaln_index[r])`。
5. `n2 = block.norm2(x2)`；`h2` を 2 と同じく塊で；`m = empty_like(x2)`；塊ごとに `m[r] = block.mlp(h2[r])`。
6. 塊ごとに `_gate(x2[r], gate_mlp, m[r], adaln_index[r])` を新しい受け皿へ書いて返す。
- 時間の注記: qkv の量子化が 1 ブロックで 3×7 回走る（A-G4 の 2 ステップ目の時間に現れうる要因として記録）。

### 3.3 `final_rows`／`_denoise` の on 経路

- `final_rows_chunked`: 映像行・音声行それぞれで `buf = empty(N, hidden, f32)`、塊ごとに `buf[r].copy_(_modulate(fl.norm(rows[r]), shift, scale, time_index[rows][r]))`（bf16→f32 は厳密）→ `_linear_f32(buf, out)` を 1 回。
- `_denoise`: ① `S = text_hidden.shape[0] + cond + 2·A + T'·gh·gw` を事前計算 ② on かつ cuda なら上限を課す（`try` の内側・`_place_dit` の前）③ `_place_dit(dit, prefetch=..., blocks_on_gpu=blocks_on_gpu_for(S, n_load) if on else n_load)` ④ `refine_text` → on なら token_refiner と condition_proj を退避（#7）⑤ ループで on なら `box = [dit.embed_rows(...)]; dit(None, tokens_box=box, vram16=plan, ...)`（off は従来の行のまま）⑥ `finally` の先頭で上限を 1.0 に戻し（例外を出さない）、`entry["vram16"]`（上限の fraction と MiB・常駐数・分割単位・`num_alloc_retries`／`num_ooms` の差分・`inactive_split_bytes.all.peak`）。
- OOM: on のときだけ `except torch.OutOfMemoryError as exc: raise RuntimeError("VRAM 16GB mode でも GPU メモリが足りません (CUDA out of memory)。解像度かフレーム数を下げるか、VRAM 16GB mode を切ってください") from exc`。アプリ側の `_is_oom`（`services/pipeline_manager.py:65-68`）が GPU_OOM 503 に分類し既存の方針でワーカーを unload（626-632）。案内文は `detail` に載る。
- `vram16_mode_used`: `generate` の戻り値で `keep_resident_used` の直後に `"on"|"off"`（`_denoise` の戻り値は変えない）。

### 3.4 ワーカー

- `H3GenerateRequest.vram16_mode: bool = False`（ワーカーの既定は off。API の既定 on はアダプタ）。`_GENERATE_FIELDS`・`_generate_request` の `_or("vram16_mode", False, bool)`・セルフテストの `--vram16`（`BooleanOptionalAction`）・golden の**末尾**・docstring（43・66）と 328 のログ行。`load` の項目は増やさない。

### 3.5 テスト（`tests/test_h3_engine.py`）

- `_tiny_cfg`／`_tiny_dit_tensors` に `heads=` 引数（既定 1）。ヘッド 4・群 3 で on の attention が `H3Attention.forward`（SDPA・CPU）と `torch.equal`。
- 等価テストの偽の int8 は float64 で計算してから bf16 に戻す（`forward` と `forward_rows` の両方を同じ偽物で）。
- 足すテスト: `row_chunks` の等分（S=4,097→2 塊・S=4,096→1 塊）／`block_forward` が on／off で `torch.equal`／`final_rows` の on／off 一致／`H3DiT.forward` の on／off 一致・`tokens_box` の取り出し・weakref で埋め込みがブロック 0 の後に解放されること・`block.vram16` が off の後に None・off の forward が `lowvram` を呼ばない／`derive_memory_cap`・`blocks_on_gpu_for` の算術（全常駐・off・しきい値）／常駐の遷移テストを vram16 で parametrize（`move_module_tree`／`drop_to_meta` が呼ばれたことを spy で・J1 と J2 の潜在が一致・on と off の潜在が `torch.equal`）／golden（generate・done）の更新。`inference_mode` 不在は現状のまま。
- GPU（`H3_GPU_TESTS=1`）: `forward_rows` の行の結合が `forward` と `torch.equal`、`forward(x[r])` の縦結合が `forward(x)` と `torch.equal`（m＝S の実寸と 4,096 と 129。(n,k) は qkv の群幅・out・fc1・fc2 の 4 種）、`k.mean` の [1,8,S,128] 対 [1,56,S,128] の切り出し比較（M1 の先行確認）。

### 3.6 実装の順序（C1。各手順で CPU の検査を通してから次へ）

1. `constants.py` の定数・`lowvram.py` の `Vram16Plan`・`row_chunks`・`blocks_on_gpu_for`・`derive_memory_cap`。
2. `quant_linear.forward_rows`。
3. `lowvram.block_forward`・`final_rows_chunked`（既存関数を範囲で呼ぶ）。
4. `dit.py` の分岐（ブロックの先頭・`tokens_box`／`vram16`／`del tokens`・`final_rows` の引数・遅延 import）。
5. `pipeline_h3`: `H3GenerateRequest.vram16_mode`・S の事前計算・`_place_dit` の引数と毎回の代入・上限・退避・`tokens_box`・OOM・`vram16_mode_used`。
6. `worker`（§3.4）と golden → `run_h3_pytest.py` 全件（A-G1）。
7. コードレビュー（A-G6）→ 修正 → GPU の門（A-G5 → A-G2 → A-G3 → A-G3b → A-G4。監督が了承を取ってから）。

## 4. アプリ側（機能の有無の正本はアダプタの `REJECT_TABLE`。記述子に機能の欄は無い）

- **API** `api/models.py`: `VRAM16_MODE_DEFAULT_H3 = True`（`*_DEFAULT` の作法。コメントに「api の正本・Gradio のミラー・MCP の import 元の 3 点を同時に」）と `GenerateRequest.vram16_mode: bool | None = Field(None, ...)`（Acceleration 区画。None＝系統の既定）。検証は型だけ。
- **既定の書き戻し**: `RequestRules`（`services/engines/ltx/request_rules.py`）に真偽値の既定の小さな表 `flag_defaults: dict[str, bool]` を足し、`fill_defaults`（`services/engines/__init__.py:215-238`）が None のときだけ書き戻す。H3 の `REQUEST_RULES` に `{"vram16_mode": True}`。LTX は宣言しないので None のまま。`constraints()` には出さない。仕様書 §6.2b (c) に 1 文。
- **H3 アダプタ** `services/engines/h3/adapter.py`: `HONOURED_FIELDS` に追加。generate の payload は `attention_backend`・`block_swap_prefetch` の直後に `vram16_mode: bool(...)` を**毎回**送る。`GOLDEN_GENERATE_KEYS_H3`（447）と 2 つのゴールデン（471・501）を更新。`done.vram16_mode_used` を `GenerationOutcome`（`ltx/adapter.py:558-640`）と `_write_metadata`（`pipeline_manager.py:1860-1893`。`keep_resident_used` の隣）へ。`_DONE_FACTS` は変えない。
- **LTX 2.3／2.5 アダプタ**: `REJECT_TABLE` に `("vram16_mode", "vram16_mode", lambda r: r.vram16_mode is True)`（None と False は通す）。**422 の文言は機能名で分岐**（両アダプタの `reject_unsupported` の雛形は「もう片方の LTX を選んでください」と誘導するため。`ltx/adapter.py:255-264`・`ltx25/adapter.py:494-503`）: 「VRAM 16GB mode は MiniMax H3 専用です。LTX では使えません」。更新するテスト: 4 表の網羅監査 3 か所・`REQUEST_OVERRIDES_23`／`REQUEST_OVERRIDES`・完全一致タプル（`test_ltx_adapter_audit.py:144`・`test_engine_dispatch.py:222`・`test_ltx25_api_guard.py:1609/1655`・`test_mcp_tools_system.py:313`・`test_ltx25_adapter.py:502/576/612`）・`test_engine_h3_adapter.py:558`・文言のテスト。連結スキーマには足さない。
- **OOM の表示**: pytest で「案内文を含む RuntimeError → `_is_oom` → 503 `GPU_OOM` の `detail` に案内文が残る」を固定。注入の仕組みは作らない。
- **Gradio**: `ui.py` の Acceleration に `accel_vram16` の `gr.Checkbox`（既定 True）・`feature_scope.py` の `FEATURE_UI`／`RESET_VALUES`・`_gated_components`・`handlers.py` のミラー定数と末尾のキーワード専用引数・「既定と違うときだけ送る」（False のときだけ）・`i18n.py` EN／JA。単発の Generate タブだけ。テスト: `test_gradio_models.py:398/457/510`・`test_gradio_ui.py`・`test_gradio_handlers.py`。
- **MCP** `mcp_server/tools/generate.py`: `vram16_mode: bool | None = None`（`is not None` のとき送る）。docstring と `server.py` の `INSTRUCTIONS`・`MCP_SERVER_DESIGN.md` D28・`test_mcp_tools_generate.py`。
- **仕様書** `Videomni_Backend_Specification.md`: §6.2 の表・§6.2b (c)(f)・§6.6・§6.10 (d) の件数表・§12.2・版上げ。`MULTI_ENGINE_DESIGN.md` §5.6。

## 5. 操作パネル（aux2・契約・Language は変更不要。配置は埋め込み再ビルド＋deploy.ps1 の 2 か所）

- 前例＝Fused GGUF 行（既定 on・ON の間だけ注記・`false` のときだけ要求に載せる）。
- 触る場所: `shell/accelerationSettings.ts`（型・2 つの defaults・`withServerDefaults`・読み取り 2 分岐・`AccelerationRequestFields`・`effectiveAccelerationFields`・`accelerationRequestFields`・冒頭コメント）、`shell/useAccelerationSettings.ts`、`shell/SettingsPanel.tsx`（`!vram16Unsupported && (...)`。注記「MiniMax H3 専用。ノイズ除去の活性化を分割して VRAM 16GB に収める。精度は変えない」）、`i18n/strings.ts`（en／ja）、`shell/featureScope.ts`（`vram16_mode: { disables: ["settingsRow.vram16"], resets: "vram16Mode" }`）、`shell/comfortDisplayTable.ts:140-149`、`api/types.ts` の `GenerateRequest`（**連結要求の型と `buildA2vChainPayload` には足さない**。バックエンドの連結スキーマと揃える）、`modes/batch/useBatchForm.ts:868-876` の OR-list は Batch i2v の経路だけ確認、`bridge/mockBridge.ts`（LTX 2.3／2.5 の `MOCK_UNSUPPORTED_FEATURES`）。GPU_OOM の表示に `detail` が出ることの vitest を 1 本。
- テスト: 完全リテラルの 7 ファイル・`featureScope.test`・`AppShell.featureScope.test`・`SettingsPanel.test`・`mockBridge.test`／`mockBridge.h3.test`・新規 `vram16Mode.paths.test.ts`。
- 文書: `FE/Docs/API_REFERENCE.md`（送信規約の表）・`FE/Docs/DEVLOG.md` 新節＋最終更新行。

## 6. 弾と門

| 弾 | 範囲 | 担当 | 門（順序どおり） |
|---|---|---|---|
| A: エンジン | §3 | C1（Opus） | **A-G1** CPU テスト全件（`run_h3_pytest.py`）／**A-G6** コードレビュー（Opus）→修正／**A-G5** 分割単位と常駐の比較（**判定規則を事前登録**: 候補の中で 2 ステップ目が最速から +10% 以内のもののうち割り当てが最小のもの。同点なら大きい塊。ヘッド 28／14／8・行 8,192／4,096／2,048・常駐 4 対 8〔640×384 と 896×1152〕を 2 ステップで 1 本ずつ）→ 定数を確定／**A-G2** GPU: `forward_rows`・行の結合・`k.mean` のテスト、製品の形（S≈39,764・実重み・1 ブロックずつ 50 ブロック）での on／off のブロック比較（`torch.equal`。差が出たら最大差と相対差を記録し裁定 46）／**A-G3** off の指紋 `0da6111c…` 不変／**A-G3b** セルフテストを `--vram16` で回し（G5 の条件・S＝9,326 で FFN は 3 塊）指紋が `0da6111c…` と**一致**（端から端までのビット一致）／**A-G4** 計測台の F0 相当（I2V 896×1152×124・3 ステップ・常駐 on・決定性 on・`--vram16`）: 共有が基準並み・2 ステップ目・割り当て／予約・コミット量の山（F0 と比較）・`vram16` の記録。**台帳の目標の形 2 本**（1280×768×311f・1920×1088×141f・2 ステップ・退避ゼロの判定） |
| B: 配線 | §4・§5・Gradio・MCP・metadata・テスト・文書（API_REFERENCE・DEVLOG） | C2 アプリ（Opus）・C3 操作パネル（Opus） | B-G1 pytest 全件・MCP テスト／B-G2 `npm run typecheck`・vitest（`--exclude backend.integration`）・lint／B-G3 mock での往復（H3 で行が出る・LTX で隠れる・要求に載る）／B-G4 コードレビュー／B-G5 配置（埋め込みビルド→deploy.ps1。AviUtl2 停止時）→ **オーナーの操作パネルからの確認**: I2V 896×1152×124f で 2 ステップ目の時間を見てから 20 ステップ 1 本（タスクマネージャーの共有メモリ）・off での 1 本 |
| C: 文書 | 設計正本（§2 に裁定 44〜47・51／§4〜§6・§8〔`lowvram.py`・メッセージの `vram16_mode`・`--vram16`〕・§9・§11・§13 に「VRAM 16GB mode」と山の内訳の表・off の 10 万行超の限界・PinkCherry）・MULTI_ENGINE §5.6・仕様書・研究ノート §10（48〜50・52）／§13.3・記録簿 §169・台帳 §1-101・README §7.6（凍結ゾーンの規則を確認してから） | 文書係（Sonnet） | 読み合わせ・敵対的レビュー |

- GPU を使う門は弾ごとにまとめてオーナーの了承を取る（A-G5〜A-G4 で約 15 本・B-G5 はオーナー実行）。GPU は同時に 1 本。
- 実装担当は裁定 49。設計に疑問が出たら止めて監督へ。
- コミットは弾ごとに dev へ（監督）。main merge はオーナーの合図。

## 7. 範囲外

- 快適上限の較正（§1-95）・Turbo／LoRA・Ref2V・CUTLASS 設定・VAE の融合・T1 の非再現の原因・off 経路の 10 万行超の防御（既知の限界として記録）。
