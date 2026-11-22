# A: 製品のコードの棚卸し（MiniMax H3 エンジンと Sulphur-3 int8 の 2 本）

調べた版: `Nz-Videomni` の HEAD `c773dd6d`（2026-11-22）。読むだけで、ファイルの編集・テスト・Python の実行はしていない。
行番号はすべてこの版のもの。「推測」と書いたもの以外は、コードか文書で確かめた事実。

---

## 1. プロジェクトの概要（H3 エンジンの構成と、対応している重みの形式）

### 構成（アプリ → アダプタ → ワーカー → パイプライン → DiT）
- アプリ（torch の無い `.venv`）: `services/model_registry.py:201-214` の事前検査が `h3_format.inspect_h3_dit` と `inspect_h3_te` を呼び、`H3FormatError` を 422 `MODEL_INCOMPATIBLE` にする。`services/pipeline_manager.py:685-708` が `h3_format.quant_class` を読み、`services/weight_identity.py` が sha256 で FL2VA か Ref2VA かを判定する。
- アダプタ: `services/engines/h3/adapter.py`。
- ワーカー: `engine_h3/worker.py`。専用の仮想環境は `.venv-engine-h3`（設計正本 `Docs/MINIMAX_H3_DESIGN.md:29`）。
- パイプライン: `engine_h3/pipeline_h3.py`
  - `load()`（:764-856）は検査だけを行う。`inspect_h3_dit` → `inspect_h3_te` → `require_cuda_quant(formats, fp8_mode)` → `shared_quant_status`。
  - `_build_dit`（:1572-1602）がノイズ除去の段で DiT を組む。meta の上で `H3DiT(cfg)` を組み、`swap_in_int8_linears` で差し替え、`load_streaming` で読み、`normalize_dtypes` で dtype を揃える。
  - `_place_dit`（:1628-1667）が、ブロック以外を GPU へ移し、ブロックは `BlockSwapService` に任せる。
- DiT: `engine_h3/dit.py`
  - `H3DiTBlock.forward`（:362-381）の先頭で、`vram16` が設定されていれば `lowvram.block_forward` へ分岐する。
  - 量子化層は `engine_h3/quant_linear.py`・`fp8_linear.py`・`w4a8_linear.py`。

### 対応している重みの形式（設計正本 §5・§7、記録簿 §210・§211）
- DiT の量子化の形式は `DIT_FORMATS = (int8_tensorwise, float8_e4m3fn, asym_w4a8_int8)`（`h3_format.py:54-59`）。1 ファイルに混在してもよい（設計正本 :30・:1462）。テキストエンコーダは int8 だけ。
- 量子化の印（その層が量子化されていることと形式を示す記述）は 2 か所から読む。メタデータ式（旧式）の `__metadata__._quantization_metadata.layers` を優先し、そこに載っていない層だけ層ごとの `<層>.comfy_quant` を読む（`h3_format.py:321-363`）。
- 実機で確かめた重み（記録簿 §211.6 の G1〜G6）:
  - 公式 int8 ConvRot
  - 公式 fp8
  - Kijai w4a8（印はメタデータ式だけ）
  - h3ErosMax w4a8（int8 20 層＋w4a8 180 層の混在。ブロック 0・1・47・48・49 は int8 の per_row〔行ごとの倍率〕）
  - 旧式の印の int8
  - fp8＋Turbo LoRA
- 実機で確かめた int8 は、すべて `convrot:true`。研究ノートの記録では、ヘッダを実測した int8 は 14 本とも ConvRot つき（`MINIMAX_H3_RESEARCH_NOTES.md:484`）。
- 枝刈りしていない版は対象外で、読み込みを断る（`h3_format.py:513-519`、記録簿 §210.2 の裁定 1）。

---

## 2. 検査（`h3_format.inspect_h3_dit`、`h3_format.py:502-611`）

### H3 の DiT と判定する条件と、断るもの
1. 判定の条件: 同じ接頭辞の下に `video_patch_proj.weight` と `audio_patch_proj.weight` が両方あること（:207-210）。接頭辞は 4 通り（:74）。
2. `adaln_t_table` が無いファイルは断る。`time_embedder.proj_in.weight` があれば「枝刈りしていない版」という理由の文を付ける（:513-519）。
3. 必須のテンソル（`_DIT_REQUIRED` :258-262）と、ブロックごとの必須のテンソル（`_BLOCK_REQUIRED` :253-256）を確かめる。ブロックの番号は 0 から連続していなければならない（:283-293）。
4. 印を読み、印の無い層を検査する（:305-318）。その後、形式ごとに dtype と形を厳密に確かめる。
   - int8（:405-412）: 重み I8 `[N,K]`、`weight_scale` F32 `[N,1]`。0 次元（ファイル全体で 1 つ）の倍率は断る。
5. 形の表（:569-589）で、全ブロックと token_refiner の全ブロックの重みを確かめる。

### 印の読み方（`parse_marker` :134-181）
- `format`・`convrot`・`convrot_groupsize`・`group_size` は、最上位か `params` の中から読む。両方にあれば最上位が勝つ。
- **`convrot` が無ければ False になる**（:159 `merged.get("convrot", False)`）。真偽値でなければ断る（:160-161）。
- `convrot_groupsize` の既定は 256（:162）。int8 では 256 以外の値もそのまま通す（:142-144）。
- `per_row`・`orig_dtype` などの鍵は読まずに捨てる（:141-142）。

### 印の無い浮動小数点の層（BF16・F16・F32）が混じるファイル
**受け入れる。何層あっても、どの層でもよい。**
- `_refuse_unmarked`（:366-384）が断るのは次の 2 つだけ。
  - 印の無い I8 または F8_* の重み
  - 印の無い層にある補助テンソル（`weight_scale`・`input_scale`・`weight_s_*`・`weight_codebook`）
- 印の無い BF16 の層は、どれも `quant` に入らない。そのため、浮動小数点の `nn.Linear` のまま残る。
- テストも、この振る舞いを固定している。
  - 全層が浮動小数点で印が 1 つも無い DiT を受け、`cfg.quant == {}` になること（`tests/test_h3_format.py:292-295`）
  - token_refiner が BF16 のまま混在すること（`test_mixed_formats_in_one_file` :400-415）
- ただし、本体のブロックの一部だけが BF16 という組み合わせを直接試すテストは見当たらない。
- 浮動小数点の重みに int8 の印が付いた層は断る（:399-400、`test_marker_on_a_float_weight_is_refused` :571）。

### 知らないテンソル
- 知っている名前の一覧は `_DIT_KNOWN`（:238-249）。どの Linear も `weight|bias|weight_scale|comfy_quant` を持てる。
- 量子化層の補助テンソルは、知っている名前として扱う（:497-499）。
- それ以外は `unknown` に入る（:592）。`pipeline_h3.py:779-783` が警告を出し、`dit_key_map_for`（:561-574）が読み捨てる。断りはしない。

---

## 3. 層の差し替え（`quant_linear.py`・`fp8_linear.py`・`w4a8_linear.py`）

### 差し替える条件
- `swap_in_int8_linears`（`quant_linear.py:611-651`）が差し替えるのは、`cfg.quant` に名前がある層だけ。
- **印の無い層は、`H3DiT` が組んだ素の `nn.Linear(..., dtype=bfloat16)` のまま残る**（`dit.py:262,265,293-294`）。
- 読み込みは `load_state_dict(assign=True)` で、ファイルの dtype のまま入る（`sft_load.py:13-17,90`）。その後、`normalize_dtypes` が bf16 に揃える（6 を参照）。
- fp8 と w4a8 の層の差し替え先にバイアスがあれば、読み込みを断る（:668-669）。

### convrot の旗と groupsize
- `H3Int8Linear` は、`forward`（:245-265）でも `forward_rows`（:267-304）でも、`int8_linear(..., convrot=bool(spec.convrot), convrot_groupsize=...)` に印の値をそのまま渡す。
- comfy-kitchen 0.2.37 の CUDA 経路の `int8_linear`（`.venv-engine-h3/.../comfy_kitchen/backends/cuda/__init__.py`）:
  - `convrot=False` のときは、`_C.quantize_int8_rowwise` で入力を回転せず行ごとに量子化し（:2155-2165 付近）、同じ GEMM（行列積の処理）に渡す。
  - ComfyUI 側の処理（`comfy_kitchen/tensor/int8.py:283-294`）も、`params.convrot`（既定 False）を `int8_linear` にそのまま渡す。
- **したがって、ConvRot の無い素の int8（重みが行ごとの倍率 `[N,1]`）も、H3 エンジンは式の上では正しく計算する実装になっている。ConvRot つきを前提にした処理は無い。**
  - 例外は VRAM 16GB mode の入力量子化の共有で、ConvRot つきと groupsize 256 のときだけ使う（`shares_input` :321-322）。
  - ConvRot が無いときは、この共有を使わずに `forward_rows` を切片ごとに呼ぶ（`lowvram.py:154`）。
- `convrot_groupsize` は 256 以外の値もそのまま渡す。groupsize 256 の速い経路（融合した核）に乗らないだけ。
- `per_row` の鍵は読まない。倍率の形が `[N,1]` であれば、行ごとの倍率として扱う。

### 未確認・前提が崩れうる点
- **ConvRot の無い int8 を H3 で実際の GPU で回した記録は無い。**
  - CPU テストは、偽の核（`_int8_linear_as_float64`、`tests/test_h3_engine.py:1764-1772`）が `convrot=False` を前提にしているので、配線だけを確かめている。
  - 記録簿の「回転なし」の int8 は、すべて LTX 2.3 の別エンジン（読み込みのときに bf16 に戻す方式）の話（`VERIFICATION_LOG.md:15710` 付近、`COMFORT_LIMIT_TABLE.md:750`）。
- ノイズの仮説（推測）: 印の `convrot` と重みの実態が食い違うと、エラーを出さずに完走してノイズになる。
  - 印に `convrot` の鍵が無い（既定 False）、または false なのに、重みが実際には回転済み
  - 逆に、`convrot:true` なのに重みが回転されていない
  - 倍率の意味（掛けるのか、割るのか）がファイルの作り手と違う
  - どれに当たるかは、ヘッダの実測と重みの値の検査で切り分けられる（別の担当の範囲）。

---

## 4. `lowvram.block_forward`（`lowvram.py:119-197`）

### 各層で呼ぶもの
- **qkv**（:148-161）:
  - `_lora.xa_for(qkv, h)`（:149）。素の Linear でも LoRA が無ければ None を返し、落ちない。
  - `qkv.shares_input(h)`（:150）。**素の `nn.Linear` はここで `AttributeError` になる（報告の症状と一致）。**
  - 共有するときは `qkv.quantize_input(h)` と `qkv.forward_rows_q`（:152）、共有しないときは `qkv.forward_rows`（:154）を呼ぶ。どちらも素の Linear には無い。
- **out_proj**（:176）: `attn.out_proj(o)` を `Module.__call__` で呼ぶ。素の Linear で動く。
- **fc1・fc2**（:191-192）: `_dequant_cache(fc1)` と `_dequant_cache(fc2)` の文脈の中で `block.mlp(...)` を `__call__` で呼ぶ。素の Linear で動く。
  - `_dequant_cache`（:200-203）は `getattr(layer, "dequant_cache", contextlib.nullcontext)()` で、注釈に「一部を量子化したファイルで、fc1 と fc2 が bf16 のまま残った場合」と明記されている。
  - この想定は弾 1 のコミット `7d1c6f34` で入った。
- その他:
  - `final_rows_chunked`（:233-252）と `embed_rows_packed`（:206-230）は `_dit._linear_f32` を使う（素の Linear が前提）。
  - AdaLN と norm は素のモジュール。

### 素の `nn.Linear` が来ると落ちる箇所
- `engine_h3` と `services` と `engine` の全体を grep した結果、量子化層だけのメソッドを呼ぶのは **`lowvram.py:150` と :152・:154 の qkv の 3 行だけ**。qkv 以外には無い。
- つまり、qkv が量子化されていれば、out_proj・fc1・fc2 が素の Linear でも VRAM 16GB mode は通る（コードの読解による。素の層を混ぜて実際に動かしたテストは無い）。

### VRAM 16GB mode が off のとき
- `H3Attention.forward`（`dit.py:269-285`）は `self.qkv_proj(x)` と `self.out_proj(...)` を `__call__` で呼ぶ。
- `H3MLP.forward`（:296-298）も `__call__` で呼ぶ。
- 量子化層だけのメソッドは一切呼ばないので、**off なら素の bf16 の Linear で動く**（コードの読解。素の層が混ざったブロックを off で動かした記録は無い）。

---

## 5. LoRA（`lora.py`）

- `attach`（:170-221）は、`H3QuantLinear` か `nn.Linear` を対象として受ける（:193）。forward hook（:166-167, 212）を掛け、`__call__` を通る呼び出しのたびに `apply` を呼ぶ。
- **素の `nn.Linear` に LoRA を当てる経路は既にあり、実際に使われている。** 公式の重みでは token_refiner の 2 ブロックが BF16 の素の Linear で（研究ノート :426）、G6 の LoRA は 208 組を当てた。208 は「50 ブロック×4＋refiner 2 ブロック×4」と一致する（記録簿 :24008）。
- `xa_for`（:261-282）は `_lora_specs` を `getattr` で引くだけなので、素の Linear でも動く。
- `forward_rows(..., lora_xa)` は、量子化層の中で `add_lora`（`quant_linear.py:189-200`）を通って `lora.apply(..., xa=)` を呼ぶ。
- VRAM 16GB mode で qkv が素の Linear のときは、LoRA があっても無くても :150 で落ちる。

---

## 6. 素の Linear（BF16）の重みが混じる場合の周辺

### dtype の扱い
- `normalize_dtypes`（`dit.py:474-510`）は `H3QuantLinear` を飛ばす。それ以外の浮動小数点の Parameter と buffer は、精度の島（下の一覧）なら f32、それ以外は bf16 に揃える。
  - 精度の島: patch_proj、video_out と audio_out、全 `adaln_proj.linear`、`adaln_t_table`、`rope`
  - ブロックの中の素の Linear は bf16 になる。BF16 のファイルなら何も変わらない。
  - F16 や F32 で保存されたファイルなら bf16 への丸めが 1 回増える。これは事実で、それが忠実さの原則に合うかの評価はしていない。
- 設計正本 :1457「量子化されていない層は、読んだ後に島の外側を bf16、内側を f32 に揃える」。

### block swap による運び
- 先読みの経路は、テンソルを `("plain", dtype, shape)` として uint8 の arena（ブロックの重みを詰める置き場）に詰める。dtype は問わない（`block_swap_prefetch.py:458-460`）。番地は 512 バイトに揃える（:139-142）。
- 同期の経路は `blk.to(device)` で運ぶ（`block_swap_service.py:256-268`）。
- bf16 の 2 次元の重みも、そのまま運べる。

### ブロックの大きさ（推測を含む）
- int8 のブロックは 371.1MB（記録簿 §210.7）。
- BF16 のブロックは、qkv・out_proj・fc1・fc2 の合計で約 770MB と見積もった（形からの計算で、実測ではない）。
- 置き場の環は最大のブロックの大きさで作られる（記録簿 :23882）。そのため BF16 のブロックが 1 つでもあると、固定メモリの中継（2 枠）と VRAM の環が約 2 倍になる（推測）。

### ブロック以外の線形層の dtype（公式のファイル。研究ノート :426）
| 層 | dtype |
|---|---|
| token_refiner の 2 ブロックの 4 層 | BF16（素の `nn.Linear`） |
| condition_proj | BF16 |
| patch_proj、final の出力 | F32 |
| AdaLN | F16（読み込み後に F32） |

- 素の `nn.Linear` は既に実績がある。ただしブロック swap の外（GPU に常駐。VRAM 16GB mode では `refine_text` の後に降ろす。`pipeline_h3.py:1670-1686`）での実績で、ブロック swap の中で運んだ実績は無い。
- 量子化が 1 層も無い DiT は、`quant_class` が "int8" を返す（`h3_format.py:619-633`）。

---

## 7. Ref2VA と FL2VA の判定、枝刈りの判定

### Ref2VA と FL2VA
- ファイル全体の SHA-256 を、記述子 `scripts/manifests/40-minimaxh3.json:48-61` の `known_weights`（公式の fl2va と ref2va の 2 本。どちらも 20,970,379,616 バイト）と照らし合わせる（設計正本 :624-628、`services/weight_identity.py:153-184`）。
- 大きさが一致しないファイルは、ハッシュを計算しない。したがって Sulphur-3 は `transformer_role = null`（種別なし）になる。
- 種別なしのときの扱い（設計正本 :632-633）:
  - 操作パネルは参照素材を送る。
  - 注意文の ① と ② は出さない。
- エンジン（`engine_h3`）は種別を見ない。参照素材が来れば行を足すだけ。

### 枝刈り版とそうでない版
- 枝刈り版の判定は `adaln_t_table` があるかどうか（`h3_format.py:513-519`）。
- 枝刈りしていない版は「この版はまだ使えません」と断る（:516-517）。`H3DiT.__init__` でも二重に断る（`dit.py:446-449`）。

---

## 8. 過去の文書の記述

- **「Sulphur」**: すべて LTX 2.3 の Sulphur-2（GGUF の Q4_K_M と Q6_K、`sulphur_distil_fp8mixed`）の話。H3 の Sulphur-3 の記述は無い（`COMFORT_LIMIT_TABLE.md` §11 ほか）。
- **ブロックの一部を BF16 で残した H3 の配布物**（研究ノート §13.3）:
  - :464 H3 Eros Max「int8 ConvRot（per_row・一部ブロックは bf16）」
  - :491「4〜5 ブロックを bf16 で残す版（Eros beta6 は 4 ブロック）」
  - いずれも製品で試した記録は無い。
  - 試したのは h3ErosMax w4a8 beta5 で、ブロック 0・1・47・48・49 は bf16 ではなく int8 の per_row だった（記録簿 :23780、:23989）。
- **Hybrid**: FL2VA を土台に、後段ブロックの `adaln_proj` を Ref2VA から取る、などの作り方がある（研究ノート :501-508）。PinkCherry Hybrid は置くだけで生成できた（同 :516、設計正本 §12）。Hybrid とコミュニティ版の対応の保証は後回し（設計正本 :1634、:1654）。
- **ConvRot の無い int8（「raw int8」）**: H3 での記述は無い。ConvRot の無い int8（silveroxides `int8mixedtensorwise`）を測ったのは LTX 2.3 の側だけ（`COMFORT_LIMIT_TABLE.md:750,777`、記録簿 :15710）。
- **「keep」「keepB」**: 該当する記述は無い。
- **コード中の手がかり**: 「一部を量子化したファイル」（`lowvram.py:201-202`）と「印の無い I8・F8 は bf16 の Linear に生の符号が入るので断る」（`h3_format.py:313-315`、設計正本 :1472）の 2 か所だけ。

---

## 9. H3 の量子化層・lowvram・LoRA を扱うテスト

| ファイル | 内容 |
|---|---|
| `tests/test_h3_format.py` | 検査器（49 件）。印、形式ごとの厳密な検査、印の無い I8・F8 の拒否、全層浮動小数点の受理、形式の混在、旧式の印 |
| `tests/test_h3_engine.py` | エンジンの CPU テスト（162 件。GPU のテストは `H3_GPU_TESTS=1` のときだけ）。int8・fp8・w4a8 の層、差し替え、`normalize_dtypes`、VRAM 16GB mode と off の一致（`test_vram16_*` :1958-、偽の核は `convrot=False`）、qkv の共有、`dequant_cache`、LoRA、常駐 |
| `tests/test_h3_quant_class.py` | `quant_class` と `transformer_quant_class`（8 件） |
| `tests/test_h3_lora_format.py` | LoRA の鍵の写像（`h3_lora_format`。13 件） |
| `tests/test_engine_h3_adapter.py` | アプリ側のアダプタと事前検査の文（115 件） |
| `tests/test_block_swap_release.py` | `BlockSwapService.release_installed`（共通部品） |
| `tests/test_sft_quant_format.py` | LTX 側の判定器（`metadata_layers` を H3 と共用） |
| `tests/test_chain_lora.py` | LTX の連結の LoRA（H3 は対象外） |

---

## まとめ: 2 本の症状とコードの対応

- **keepB0-1-49**:
  - 検査器は、印の無い BF16 の層を何層でも受ける。差し替えでは素の `nn.Linear` のまま残る。
  - VRAM 16GB mode（既定 on）の `lowvram.py:150` で、qkv が素の Linear だと落ちる。報告の症状と一致する。
  - 落ちるのは qkv だけ。off なら、コードの上では通る（実際に動かしてはいない）。
- **s3_raw_int8**:
  - エンジンは印の `convrot` の値（無ければ False）をそのまま `int8_linear` に渡す。ConvRot の無い int8 を計算する実装は備わっている。
  - 実際の GPU で確かめたことは無い。
  - ノイズの原因は、印と重みの実態の食い違いの可能性があるが、推測。印の内容と重みの値の検査が要る。
