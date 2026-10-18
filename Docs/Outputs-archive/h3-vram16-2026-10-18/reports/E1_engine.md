# E1_engine.md — engine_h3 の現行構造（VRAM 16GB mode 設計用の棚卸し。Explore・Opus・2026-10-18。監督が本文から保存）

ルートは `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\`。以下のパスはこのルートからの相対パス。行番号は dev `aebc3d7` 時点。

寸法（`h3_format.py:145-161`）: hidden 5376・heads 56・head_dim 128・inner 7168（qkv の出力は 3×7168＝21,504）・ffn 14336（fc1 の出力は 28,672）・text_dim 5120・層 50・refiner 2・AdaLN の格子 G＝1025・rank 8。

## 1. `engine_h3/dit.py`

### 1-1. `H3DiT.forward` の流れ（533-571）
- 引数（533-544）: `tokens` は位置引数。ほかはキーワード専用（`layout`・`t_unique`・`time_index`・`modality`・`rope_cos`・`rope_sin`・`attention_backend="sdpa"`）。
- 形の検査（547-557）。backend の配り方（558-559）: `for block in self.blocks: block.attn.attention_backend = attention_backend`。token_refiner には配らない（refiner は常に `"sdpa"`、222）。
- 変調の準備（561-563）: `t_emb = adaln_embed(t_unique)` は `[M, rank]` f32。`adaln_index = time_index*3 + modality`。
- ブロックのループ（565-567）: `x = tokens.to(torch.bfloat16)`（tokens は bf16 なので同じオブジェクト＝x と tokens は同一）。`for block in self.blocks: x = block(x, t_emb, adaln_index, rope_cos, rope_sin)`。
- 出口（570-571）: `final_rows(...)`。
- `tokens` の参照が切れる位置: forward の引数名がループの最後まで残る。`_denoise` の局所変数も `dit(...)` が戻るまで持つ（`del tokens` は `pipeline_h3.py:767`）。各ブロックの入力 x は、そのブロックの間は呼び出し側の `x` と swap の包みの `args`（`block_swap_service.py:283-286`・`356-359`）が持つ。
- token_refiner と condition_proj は forward の中では使われない。`refine_text`（465-472）だけが使い、`_denoise` がループの前に 1 回呼ぶ（`pipeline_h3.py:714`）。

### 1-2. `embed_rows`（474-495）
`parts = [text_refined.to(bf16)]` → cond があれば `_linear_f32(cond_rows, video_patch_proj).to(bf16)` → audio → video（射影は f32 の島、結果だけ bf16）→ `torch.cat(parts, 0)` で新しい `[S, hidden]` bf16。`_linear_f32`（374-377）は入力・重み・bias を `.to(float32)`。

### 1-3. `final_rows`（515-531）
`shift, scale = final_layer.adaln_proj(t_emb)`（`[M, hidden]` f32 ×2。`modality_num=1`・`expand=2`、361）。`head(rows, out)`: `h = fl.norm(x[rows])`（bf16）→ `_modulate(h, shift, scale, time_index[rows])`（添字は time_index だけ）→ `_linear_f32(h, out)`（f32 出力）。映像行→`video_out`（`[Nv, 96]`）、音声行→`audio_out`（`[Na, 32]`）。text と cond の行は計算しない。

### 1-4. `H3DiTBlock.forward`（310-325）
| 行 | 処理 | 生きているもの |
|---|---|---|
| 319 | `adaln_proj(t_emb)` → 6 本の `[3M, hidden]` f32（1 本の `[3M, 6·hidden]` の chunk のビュー、273） | |
| 320 | `h = _modulate(self.norm1(x), shift_msa, scale_msa, adaln_index)` | norm1(x) の bf16 一時・f32 `[S,hidden]` 2 本・h は bf16 |
| 321 | `h = self.attn(h, rope_cos, rope_sin)` | 1-6 |
| 322 | `x = _gate(x, gate_msa, h, adaln_index)` | f32 `[S,hidden]` 1 本・新しい x は bf16・入力 x は呼び出し側が持つ |
| 323 | `h = _modulate(self.norm2(x), ...)` | 320 と同じ |
| 324 | `h = self.mlp(h)` | fc1 の出力 `[S, 28672]` bf16 など |
| 325 | `return _gate(x, gate_mlp, h, adaln_index)` | 322 と同じ |
引数はテンソルだけを位置で受ける約束（コメント 318）。

### 1-5. `_modulate`・`_gate`
- `_modulate`（276-288）: `out = scale.index_select(0, idx)`（新しい f32 `[S,hidden]`）→ `out.add_(1.0)` → `out.mul_(x)`（bf16 を f32 に昇格）→ `out.add_(shift.index_select(0, idx))`（2 本目の f32）→ `return out.to(x.dtype)`。docstring（280-282）「f32 の一時は 2 本。1344x768 x 124f では 1 本約 0.8GB」。
- `_gate`（291-296）: `out = gate.index_select(0, idx)` → `out.mul_(other)` → `out.add_(x)` → `out.to(x.dtype)`。
- どちらも行ごとに独立。

### 1-6. `H3Attention`（207-236）
- 層（217-220）: `qkv_proj = Linear(hidden, 3*inner)`（int8 に差し替わる）、`q_norm`／`k_norm = RMSNorm(head_dim)`、`out_proj = Linear(inner, hidden)`。bias 無し。
- qkv の並び（228-229）: `qkv_proj(x).view(s, 3, heads, head_dim)` → `unbind(dim=1)`。列は `[q(56×128); k(56×128); v(56×128)]`、各ブロックの中はヘッド外側・チャネル内側。ヘッド h の q は列 `h·128…h·128+127`、k は `7168 + h·128…`、v は `14336 + h·128…`。q・k・v は非連続ビュー。
- q_norm／k_norm（230-231）: 最後の軸 D だけ。(行, ヘッド) ごとに独立。新しいテンソルが返る。
- RoPE（232-234、`rope.py:175-184`）: cos／sin `[S,96]` を `.to(x.dtype).unsqueeze(1)` で全ヘッドに放送。先頭 96 チャネルを回転、残り 32 は素通し。`torch.cat` で新しいテンソル。ヘッドをまたぐ計算は無い。
- `_attention`（190-199）: `[1, H, S, D]`（HND）のビュー。sage は `.contiguous()` してから `sageattn(..., tensor_layout="HND", is_causal=False)`（196）。SDPA は非連続のまま `F.scaled_dot_product_attention(qh, kh, vh)`（マスク無し・既定 scale）。戻りは `out.squeeze(0).transpose(0,1)` の `[S,H,D]` 非連続ビュー。
- out_proj（236）: `out_proj(out.reshape(s, heads*head_dim))`（非連続なのでコピー）。入力列 `h·128…` がヘッド h。
- sage の取り込み（175-187）: 初回 import。

### 1-7. `H3MLP`（239-249）
`gate, value = self.fc1(x).chunk(2, dim=-1)` → `self.fc2(F.silu(gate) * value)`。fc1 の行は `[gate; value]`。行ごとに独立。fc1 の出力 `[S, 28672]` bf16。

### 1-8. `H3AdaLNProj`・`adaln_embed`・`adaln_index`
- `H3AdaLNProj`（252-273）: `Linear(rank, expand*hidden*modality_num)`（ブロックは 96,768）。f32 で `F.linear` → `view(M*3, 6*hidden)` → `chunk(6)`。行 `m*3 + tag`（タグ 0 映像・1 テキスト・2 音声）。SiLU 無し。
- `adaln_embed`（501-513）: `pos = clamp(t,0,1)*(G-1)`、`i0 = floor(pos)`（上限 G-2）、`lerp` で `[M, rank]` f32。
- `adaln_index`（563）: `time_index*3 + modality`。M は 1〜3（`pipeline_h3.py:188-201` の `time_rows`）。

### 1-9. 設定を forward に渡す既存の仕組み
docstring 51-58: ブロックの forward はテンソルだけを位置で受ける（swap の包みが全引数を `_move_to_device` で GPU へ送るため）。非テンソルは素通しされるが罠になる → attention backend は引数でなく forward ごとに設定するモジュール属性（221-222・558-559）。先頭コメント 10 にも明記。

### 1-10. そのほか
構築（385-416）: meta の上で組み `requires_grad_(False)`。`normalize_dtypes`（418-454）: 島（patch_proj・video_out／audio_out・全 `adaln_proj.linear`・adaln_t_table・rope）は f32、`H3Int8Linear` は触らず、残り bf16。`ATTENTION_BACKENDS` の import（75）はテストが確認（`tests/test_h3_engine.py:803-810`）。

## 2. `engine_h3/quant_linear.py`
- `H3Int8Linear`（59-99）: `in_features`・`out_features`・`weight`（int8 `[N,K]` Parameter・`requires_grad=False`、84-87）・`weight_scale`（buffer f32 `[N,1]`・persistent、88-92）・`bias`（任意、93-98）・`spec`（`QuantSpec(format, convrot, convrot_groupsize)`。普通の属性、65-67・99）。
- `forward`（101-121）: `reshape(-1, K)`・bf16 化・`.contiguous()` → bias も bf16 → `int8_linear(x2d, weight, weight_scale, bias, out_dtype=bf16, convrot=..., convrot_groupsize=...)` → `reshape(*lead, N)`。
- 行で切った部分行列で呼べるか（comfy-kitchen は読むだけ）: 公開関数は registry に振り分けるだけ（`comfy_kitchen/__init__.py:997-1054`）。CUDA の制約（`backends/cuda/__init__.py:4464-4488`）は「weight は int8 で 2 次元」「weight_scale は f32」だけ。CUDA 実装（2035-2289）: 活性化の量子化は行ごと（x だけで決まる）。weight は非連続なら `.contiguous()`（2053-2054）。スケールは `numel()==n` ならそのまま（2223）。`_int8_weight_scale_arg`（528-531）は連続ならそのまま。`[N,K]` 連続の行スライスと `weight_scale[n0:n1]` は連続ビュー。出力は要素ごとの逆量子化（`x_scale[m]·w_scale[n]·acc + bias`）で int32 累積は厳密。ただし CUTLASS と cuBLAS の分岐は m・n・k で決まる（2195-2256）→ 部分行列でビット一致するかは未実測。m==2 の GEMV は 16B 境界（2171-2172）。ConvRot の量子化カーネルの選択は k で決まる（469-476。`5120 < K < 8192` は融合しない＝qkv の 5376・out_proj の 7168。fc2 の 14336 は融合）。m に依存するのは m==1 のみ（418-430）。列（K）方向を切ると ConvRot の 256 グループに関わる。行（N）方向は関わらない。
- block swap との制約: 同期経路は `blk.to(device)` と `p.data = ...`（`block_swap_service.py:259-266`）。先読みは `_parameters`／`_buffers` を歩き（`block_swap_prefetch.py:282-299`）、Parameter は `.data = view`、buffer は辞書の差し替え（646-651）。登録されていない tensor 属性は swap の対象外 → forward の中で毎回 `self.weight`／`self.weight_scale` を読む。docstring 3-14: ラッパー型を使わない理由。
- `require_cuda_int8`（131-168）: triton と eager を無効化。`swap_in_int8_linears`（171-210）。

## 3. `engine_h3/pipeline_h3.py`
- `H3LoadOptions`（105-121）: `blocks_on_gpu=8`・`attention_backend="sdpa"`・`block_swap_prefetch=True`・`deterministic=False`。`<=0` か `>=50` は swap しない。
- `H3GenerateRequest`（132-150）: 必須 `prompt`・`seed`・`width`・`height`・`num_frames`・`output_path`。既定あり `frame_rate=24`・`num_inference_steps=20`・`shift_video=12.0`・`shift_audio=3.0`・`images=()`・`keep_resident=False`・`attention_backend: str|None`・`block_swap_prefetch: bool|None`（None なら load の値。`_attention_for` 519-520・`_prefetch_for` 522-525）。`validate_request`（157-185）。
- `generate`（448-517）: 452 keep・453 `_Phases`・459 validate・461-464 画像・465 `with torch.no_grad():`・466-468 encode・470-475 image_encode・476-479 denoise・480-481 del と `_free_cuda`・482-484 decode・finally（486-493）: 成功かつ keep 以外は `release_resident()`、常に `_free_cuda()`。戻り値 dict（495-517）＝`done`（キー順序はテストで固定。`h3: {"phases", "quant_backends"}`、516）。
- `_denoise`（678-796）: 699-705 DiT の入手（常駐 or `_build_dit`）・706 mark・711 `_place_dit(dit, prefetch=...)`・714 `refine_text`・716-725 layout／coords／rope_tables／modality・731-738 雑音（CPU 生成器・f32・映像→音声→cond）・740-741 σ・742-745 `entry["sequence"]`・748 progress 0・ループ（749-773）: `time_rows` → 756 `tokens = dit.embed_rows(...)` → 757-766 `dit(tokens, layout=..., ..., attention_backend=attention)` → 767 `del tokens` → 768-771 unpatchify／rows_to_audio／euler_step → 772 del → 773 progress。ループ内に `_free_cuda` 無し。774-777 `m_per_step`・`block_swap_prefetch_used`・NaN 検査。finally（780-796）: `teardown_prefetch()` → `release_installed()` → 成功かつ keep なら `_park_dit` して `_resident`、そうでなければ外す → `del dit`・`_free_cuda()`・mark。
- `h3.phases.denoise` の中身: `dit`（file のとき 602-607）・`dit_source`・`rss_/commit_{after_load,after_place,after_release}_gib`・`sequence`・`m_per_step`・`block_swap_prefetch_used`・`dit_parked_tensors`（keep）・phase の終わりの `seconds`／`rss_*`／`commit_*`／`peak_vram_mb`／`peak_vram_reserved_mb`（321-335）。
- `_Phases`（296-335）: 入口で `reset_peak_memory_stats()`（315）、出口で `max_memory_*`。phase 単位のみ。
- `_place_dit`（634-662）: 647 `_unpatch_block_swap`・649 `n = int(self.options.blocks_on_gpu)`（load の値だけ）・650-653 全常駐なら `move_module_tree(dit, device, skip=set())` → `(None, "off")`・654-655 `skip = {id(m) for block in blocks for m in block.modules()}` でブロック以外を GPU へ・657-658 `if self._swap_service is None: BlockSwapService(blocks_on_gpu=n, device, hold_arenas=True)`（最初の 1 回だけ）・660 `prefetch_requested`・661 `install(dit)`・662 return。`hold_arenas=True` の理由（640）: 頂点が denoise の中にある。
- そのほか: `_unpatch_block_swap`（610-632）・`_park_dit`（664-676: `move_module_tree(dit, cpu, skip=set())` のあと unpatch）・`_build_dit`（578-608）・常駐（371-374・787-792・817-821・836-840・857-866・868-871・873-882）・`_free_cuda`（338-341。呼ぶ場所 481・493・547・568・795・823・842・882）。

## 4. `engine_h3/worker.py`
- `_value_or`（171-179）: 無いときも `null` のときも default。
- `_do_load`（205-256）: 233-239 `H3LoadOptions(...)`・249 `H3Pipeline`・250 `load()`・256 ready。load のオプションは `build_report["options"]`（`pipeline_h3.py:435-440`）。
- `_GENERATE_FIELDS`（156-160）: 無いキーは「無視」とログ（319-320）。`_generate_request`（281-311）: `_or(key, default, cast)`。`_do_generate`（314-334）: 323-324 attention を解決・332 `generate(req)`・333 `attention_used`・334 `_emit("done", **result)`。
- 新しい項目を足すとき: load 側 docstring 26-40・233-239・`H3LoadOptions`・`build_report`・`_selftest_parser`（431-460）・`_selftest_load_message`（463-476）。generate 側 docstring 41-51・`_GENERATE_FIELDS`・`_generate_request`・`_selftest_generate_message`（479-497）。done 側 `pipeline_h3.py:495-517`。golden に当たる。アダプタ（`services/engines/h3/adapter.py:372-377, 443`）。
- セルフテストの引数（431-460）: `--transformer --text-encoder --video-vae --audio-vae --tokenizer-dir --output`（必須）、`--prompt --width --height --num-frames --seed --steps --shift-video --shift-audio --rounds --report --attention-backend --blocks-on-gpu --prefetch/--no-prefetch --deterministic --image --keep-resident`。
- 最上位で torch を import しない（docstring 22-24・テスト 729-732）。

## 5. `engine/transformer/block_swap_service.py`・`block_swap_prefetch.py`
- `BlockSwapService`（49-361）: 生成引数 `blocks_on_gpu`・`device`・`hold_arenas=False`（68-91）。`install`（97-165）: `teardown_prefetch()` → `blocks_on_gpu==0` なら何もしない（106）・`>= total` なら swap しない（116）→ 全ブロック `.to("cpu")`（129-130）→ `prefetch_requested` なら `_build_prefetch_engine`（失敗なら同期経路）→ 包み → `_installed_transformers`（147-165）。`release_installed`（167-174）。`teardown_prefetch`（176-191）。
- `blocks_on_gpu` を後から変える手段: 専用 API 無し。属性（74）で `install` の中で読まれる（106・116・233・317）。同期経路の閉包は install 時点で取り込み（233）、PrefetchEngine は構築時の値（325）→ **次の install より前に代入すれば反映される（推論。そういう呼び出し側は現在無い）**。
- 包み: 同期 `swapped_forward`（237-286）: 窓 `[idx, idx+n)` を `.to(device)`（254-266）、`idx-1` を `.to("cpu")`（270-275）、`_move_to_device(args/kwargs)`（283-284）。先読み（353-359）: `engine.on_block_forward(idx)` → `_move_to_device` → 元の forward。`_move_to_device`（37-46）は tuple・list・dict を再帰。同期経路は最後のブロック（49）を次のパスまで GPU に残す（推論。design §4 132 行と整合）。
- `_get_blocks`（197-221）: `transformer_blocks`／`blocks`／`layers`／`model_blocks`。
- 「プロセスに 1 つ」の前提: service 147-165・86-91、prefetch 234-243、`pipeline_h3.py:367-371`・657-658・873-882、テスト 921-945。
- `PrefetchEngine`（302-740）: `prepare()`（354-417）で CPU 側の正本と配置（`_plan` 419-477。512B 境界で 1 本の uint8）、pinned pool（最大ブロック×2 枠、391）、`_validate`（479-509）、`hold_arenas` なら受け皿を `min(total, n+2)` 個（401-407）。`hold_arenas` の意味（docstring 43-96・service 56-65）: `teardown()` まで手放さない。`empty_cache()` では解放されない（86-90）。off なら `_issue`／`_release`（592-593・679-685）。`teardown()`（691-720）。常駐の量 n+1、輪は n+2。`_install_views`／`_release`（646-685）。

## 6. `sft_load.py`・`rope.py`・`constants.py`
- `move_module_tree(root, device, skip)`（121-150）: `root.modules()` をたどり skip を飛ばし、各 Parameter を `nn.Parameter(param.data.to(device), requires_grad=...)` として作り直す（同一性は変わる）。buffer は `.to(device)`。既にその device のものは数えない。meta は例外。
- meta 関連: `meta_tensor_names`（51-55）・`load_streaming`（58-118、`assign=True`）・`load_mapped_to`／`drop_to_meta`（153-209、TE 用）。
- RoPE: `build_coords`（78-149）`[S,3]` f64・`rope_tables`（152-167）f32 `[S,96]`・`apply_rope`（175-184）新しいテンソル。
- `constants.py`: torch 非依存（20）。「数値は 1 か所」（1-7）。`ATTENTION_BACKENDS`（40-41）・タグ（38）・`LOG_PREFIX`（39）。

## 7. テストと門の道具
- `tests/test_h3_engine.py`: 実行は `scripts/run_h3_pytest.py`（1-7）。GPU テストは `skipif(not (cuda.is_available() and H3_GPU_TESTS=="1"))`（684-687。`test_int8_linear_matches_dequantized_bf16`・相対誤差 < 3e-2）。小さな DiT（51-62・563-608）: HID 64・HEADS 1・HD 128・FFN 48。CPU で固定: AdaLN 両端（69-87）・フレーム規約（94-103）・σ（113-140）・M（147-163）・modality（166-172）・patchify（175-185）・RoPE（192-）・TE 群・meta 残り（626-656）・`inference_mode` 禁止（720-726。全行走査。pipeline の docstring は「inference-mode」）・worker の torch（729-732）・golden（820-897）・`_free_cuda` finally（900-918）・service 常駐（921-945）・包みの二重防止（1042-1063）・常駐の遷移（1086-1103・1190-1269）。
- golden: `GOLDEN_LOAD_KEYS`（820-823）完全一致（853）・`GOLDEN_GENERATE_KEYS`（826-830）は `_GENERATE_FIELDS` とも一致（863-864）・`GOLDEN_DONE_KEYS`（831-836）順序込み（881・896）。`h3` のキー `["phases","quant_backends"]`・phases は集合（882-883）。`phases.denoise` の中は検査しない。スタブは `_denoise` が 3 つ組を返す前提（877・1081）。常駐の遷移テストは `H3Int8Linear.forward` を monkeypatch（1208-1213）。
- `scripts/run_h3_pytest.py`（1-24）: アプリ venv の site-packages を末尾に足す。`--noconftest -p no:cacheprovider --rootdir ROOT`。
- `Docs/Outputs-archive/h3-gate1/g4_compare.py`: A 座標と RoPE・B adaln_embed・C refine_text／embed_rows／final_rows・D 2 ブロック・E ブロックごと（421-479: meta 上で `H3DiTBlock` を組み int8「mine」と bf16「mine_deq」、オラクルは DiffSynth。`x = randn(S=512, hidden)`。残差の `_rel`（93-96）と `_maxabs`（99-100）。閾値 WIRING 1e-2・QUANT 5e-2。S＝512〔L64・A24・T'4・10×10〕）。流用できる部品: `_Reader`（103-125）・`_step_indices`（132-144）・`_nonblock_dit`（212-225）・`_quant_of`（314-316）。注意: 最上位で oracle を import（67-79）。`ROOT = parents[2]`（50）はアーカイブ先では `Docs/` を指す（`WEIGHTS` の解決がずれる）。
- `h3-gate2/g4_allocated_probe.py`: cuBLAS 作業領域 8.25MiB の切り分け。

## 8. 設計で守るべき約束（文書）
1. `no_grad` だけ・`inference_mode` は 1 か所も書かない（MULTI §5.8 477-499・H3 §6 157）。
2. int8 は素の int8 Parameter と f32 `[N,1]` buffer・`int8_linear` を CUDA 経路に固定（H3 §5 143-149）。
3. block swap と先読みの共有部品は変更しない。`BlockSwapService` は常駐（H3 §6 155・§4 131・133）。
4. 常駐のジョブの終わりに GPU に何も残さない。包みを二重にしない（H3 §4 130-135）。
5. 1 度に GPU に置くのは 1 段ぶん（H3 §4 118）。
6. 精度の島: `normalize_dtypes`（H3 §6 158）。
7. meta で組んで `assign=True`・読み残しを検査（H3 §6 159）。
8. キー名は §8 で固定（H3 §8 212-223。指紋の条件は 244）。
9. 定数の正本は `constants.py`、API の正本は `REQUEST_RULES`（H3 §9 250-254）。
10. worker の最上位で torch を import しない。ComfyUI のコードは写さない（H3 §8 183・206）。

## 要点
1. off を守る鍵は golden（load／generate 完全一致・done 順序込み）。`h3.phases.denoise` の中は自由。
2. `inference_mode` はコメントでも書けない。
3. 設定は forward ごとのモジュール属性（`attention_backend` の型）。token_refiner には配られない。
4. `tokens` は `_denoise` と forward の引数名が 50 ブロックの間持つ。x は呼び出し側と swap の包みの args がそのブロックの間持つ。
5. attention 以外は全部行ごとに独立。
6. q_norm／k_norm／RoPE はヘッドごとに独立。qkv の列は `[q;k;v]`×ヘッド外側。
7. `int8_linear` の制約は dtype と次元だけ。行スライスは連続ビュー。ビット一致は未実測。
8. swap は Parameter／buffer だけ → スライスを保存しない。
9. 常駐の遷移テストは `H3Int8Linear.forward` を monkeypatch。
10. `BlockSwapService` はプロセスに 1 つ。`blocks_on_gpu` は install の前の代入で効く（推論）。
11. 受け皿は `min(50, n+2)`×最大ブロック。`empty_cache` では返らない。
12. VRAM の頂点は phase 単位。ループ内に `_free_cuda` 無し。
13. `g4_compare.py` part E は on／off 比較に流用できる（`ROOT` の修正が要る）。
14. `move_module_tree` の skip。
15. worker で触る箇所: `_do_load` 233-239・`_GENERATE_FIELDS` 156-160・`_generate_request` 296-311・セルフテスト 431-497・アダプタ 372-377/443。
