# §1-104 MiniMax H3 の Style LoRA と Turbo LoRA（同時実装）— 実装計画

- 作成: 2026-10-24（プランモード）／状態: 敵対的レビュー済み（Critical 0・Major 7・Minor 14 → Major 7 全採用・Minor 13 採用・1 件はオーナー判断〔§7-6〕）。了承待ち
- 入口: 台帳 `Docs/PENDING_TASKS.md` §1-104・設計正本 `Docs/MINIMAX_H3_DESIGN.md`（裁定 10）・オーナーの事前調査レポート（2026-10-24）
- 棚卸し報告（セッション限り。scratchpad\h3lora\）: R1 既存 LoRA 機構／R2 engine_h3 内部／R3 試験 LoRA 3 本のヘッダ。E1〜E3（操作パネル・バックエンド配管・エンジン接続点）・設計案・レビューは本文に反映済み（プランモードのためファイル化できず。保管は実装時に `Docs/Outputs-archive/h3-lora-2026-10-24/`）。
- 次の番号: VERIFICATION_LOG §176・フロントエンド DEVLOG §148・CLOSED 3-251・台帳 §3 の新項目 3-249〜・研究ノート §10 の裁定 62〜

## 1. Context（なぜやるか）

MiniMax H3 は α版で LoRA を後回しにした（裁定 10）。CivitAI などで H3 用の Style／Motion LoRA と、LightX2V 系の Turbo LoRA（4〜8 ステップで生成するための蒸留の追加重み）が普及しており、利用者が手で置いた LoRA を操作パネル・Gradio・MCP から使えるようにする。Turbo が入ると 1 本の生成時間が下がり、後続の較正（§1-95）・Chained（§1-106）の実験費用も下がる。

### オーナー裁定（2026-10-24・質問 11 件への回答。研究ノート §10 に裁定 62〜72 として転記）
1. 形式＝「ComfyUI が読める名前」。点区切り `diffusion_model.` と下線区切り kohya `lora_unet_` の両方・接尾辞 `lora_A/lora_B` と `lora_down/lora_up` の両方・`.alpha` は任意（無ければ倍率 1.0・有れば alpha÷rank）。
2. 判定＝一覧で「非対応」と表示するが選択も実行もできる。生成ボタンの下に黄色い注意文「LoRAが非対応の形式です。効果がない可能性があります。」。一部のモジュールだけを狙うファイル（token_refiner 無し等）は許容。
3. 同時使用の本数制限なし。
4. 強度＝既存のタグとチップ（0.05〜2.0・既定 1.0）を流用。H3 ではチップの音声トグルと絵文字を隠し、`audio_strength` は送らない（422 は不要）。台帳 §3 に「H3 の LoRA で動画・音声に個別の強度」を起票。
5. ステップ数・shift の自動変更なし。
6. 一覧は選択中の系統で切替（H3→`models/MiniMaxH3/LoRA`・LTX→`models/LTX23/StyleLoRA`）。系統とディレクトリの対応はサーバー側 1 か所。操作パネルは系統を知らない。
7. `install-MiniMaxH3.bat` で空の `LoRA` フォルダと案内ファイルを作る（ダウンロード無し）。
8. Gradio と MCP も同時対応。
9. LTX 側のコードは触らない（`engine/` は凍結）。
10. 試験ファイル 3 本の移動は了承後の最初の一手として監督が行う。
11. CPU 正本は保存 dtype のまま・GPU で使うときに FP32 化（可逆・数値同一）。

### 事前調査レポート（オーナー）の要件
LoRA の演算（A/B の行列積と差分の加算）は FP32。int8 重みへ融合しない。rank の固定値を書かない。qkv の行分割では B も同じ行で切る。形の合わないテンソルは黙って無視しない。全 50 ブロック分を GPU に常駐させない（block swap・先読み・VRAM 16GB mode と連動）。メモリ不足時に黙って bf16 へ落とさない。LoRA 有効時の快適上限低下は許容・実測値を記録。当面対象外＝分離 QKV の diffusers/PEFT・量子化 LoRA・LoHa/LoKr/DoRA・Text Encoder 用。

## 2. 確定事実（計画の根拠）

- **試験ファイル 3 本**（R3）: Turbo（bf16・1.96GB）＝ComfyUI 名、qkv は **rank 384 のブロック対角**（非対角 0 を実測）・他 rank 128、alpha 24／8＝倍率 0.0625、token_refiner 2 ブロックも対象。Style 3dsrx（fp32・298MB）＝**kohya 下線名** `lora_unet_blocks_0_attn_qkv_proj.lora_down.weight`、rank 16、alpha 16＝倍率 1.0、token_refiner なし。Motion（fp16・155MB）＝ComfyUI 名、rank 16、alpha なし、token_refiner あり、AdaLN 分は変換器が除去済み。3 本とも AdaLN・norm・patch・音声固有キーなし。ComfyUI 本体の loader は両方の名前と両方の接尾辞を 1 つで受ける（alpha なし→1.0）。
- **H3 エンジンの形**（R2・E3・レビューで裏取り）: int8 層は 50 ブロック×4（qkv_proj 5376→21504 `[q;k;v]`、out_proj 7168→5376、fc1 5376→28672 `[gate;value]`、fc2 14336→5376）＝`H3Int8Linear`（`engine_h3/quant_linear.py:59-154`。`forward_rows(x,n0,n1)` は毎回 `self.weight[n0:n1]` を切る。呼び出し元は `lowvram.py:139-141` とテスト `test_h3_engine.py:1881` だけ）。token_refiner 2 ブロックは **素の bf16 `nn.Linear`**（同じ名前。`refine_text` で 1 ジョブ 1 回。VRAM 16GB mode では後で CPU か meta に降ろす）。`lowvram.block_forward`（`:107-170`）は qkv を 7 群×3 本＝**21 回 `forward_rows`**（同じ bf16 の `h`）、out_proj は 1 回、FFN は `block.mlp(h2[r0:r1])` を 4096 行の塊で呼ぶ。`_build_dit`（`pipeline_h3.py:677-707`）は meta で組み→int8 差し替え→`load_streaming(assign=True)`→`normalize_dtypes()`。`generate` は `validate_request`→画像→**文章の符号化（重い）**→`_denoise`。`_denoise`（`:881-997`）は常駐の再利用か build→`_vram16_begin`→`_place_dit`（**`install()` は毎ジョブ先読み計画を作り直す**。`_plan` は Parameter と buffer〔非永続も〕を全部詰め、dtype は何でも可・非連続は自分で連続化・CPU 上のテンソルしか受けない）→`refine_text`→ステップ→finally（`teardown_prefetch`〔`_release` が `mod._buffers[name] = cpu_t` を**名前で書き戻す**〕・`release_installed`〔テンソルに触らない〕・keep なら `_park_dit`）。`move_module_tree`／`drop_to_meta`（`sft_load.py:144-160,195-209`）は非永続 buffer も動かし、hook は残る。全経路 `torch.no_grad()`（`inference_mode` 不使用）。TF32 の設定はどこにも無い（torch 2.13 既定＝FP32 行列積は厳密。新 API `torch.backends.cuda.matmul.fp32_precision` もある）。
- **既存の LoRA 機構**（R1・E2）: `services/lora_registry.py` は `config.model.lora_dir`（既定 `models/LTX23/StyleLoRA`）1 か所だけを平らに走査＋`ic_loras` 設定。`as_dict` は 7 項目（`layout` は非公開・鍵集合はテストで固定）。`_detect_layout` は **下線 kohya 名を `unsupported:`** と判定し `resolve()` が 422 `LORA_FORMAT_UNSUPPORTED`。`GET /loras` は毎回再走査。`preprocess_for` は設定の `ic_loras` だけを見る。H3 アダプタは `REJECT_TABLE` 先頭行で `loras` を 422（`services/engines/h3/adapter.py:201`）。`lora_paths` は generate に届くが読まれない。`test_honoured_fields_are_read_by_generate`（`tests/test_engine_h3_adapter.py:102-105`）は generate のソースに `request.<欄>` の読みがあることを要求する。記述子（`services/base_models.py`）に LoRA の欄は無く、`h3_format.py` は torch 非依存でアプリ側も import 済み（`services/model_registry.py:46`）。`/generate` の検証は `pipeline_manager.active_engine_family`＝**runner の記述子**を見る（`pipeline_manager.py:251-261`。`model_registry` は読み込み成功後の `_remember` でしか変わらない）。`metadata.json` の `ic_lora` 欄は系統非依存で `audio_strength` と `preprocess_for` の値をそのまま写す。`recipe_text` は metadata 全体を mp4 のコメントに埋める。
- **操作パネル**（E1）: チップは `lora/LoraChips.tsx:94-133`（🎥 `:98-100`・音声トグル 🔊/🔇 `:118-126`・音声値 `:127-129`）。`AppShell.tsx:1589-1600`→`PromptBar.tsx:37,57`→`LoraChips`。`useLoras` の呼び出しは 4 か所（`AppShell.tsx:317`・`InventoryScreen.tsx:42`・`useOutpaintForm.ts:404`・`useBatchI2vLongForm.ts:436`）でマウント時に 1 回取得。Inventory は `kind==="style"` だけ表示・系統を知らない。空表示文言は `models/LTX23/StyleLoRA` を直書き（`i18n/strings.ts:1294-1307`・`:3070-3079`）。生成ボタン直下は `GenerateReasonsNote`（送信を止める）→エラーカード。黄色の非ブロック表示は `.warning-banner.warning-banner-mild`（`SingleScreen.css:203-220`。`ChainedScreen.tsx:1005` に前例）。`keyframe_strength` の前例＝FEATURE_UI の行で隠す＋`SingleScreen.buildSubmission` の `omitStrength`（`:1011-1017`）。モック `bridge/mockBridge.ts:490-505` の H3 非対応一覧に `loras` がある。
- **Gradio／MCP**（E1）: Gradio の Style タブは `demo.load` と Reload だけで更新（モデル切替後は古いまま）。タグ解析 `handlers.py:384-386` は 3 つ目の数（音声）を受け、呼び出しは 3 か所（`handlers.py:891,1467`・`batch.py:353`）。MCP `list_loras`・`LoraArg(name, strength, audio_strength)`・`server.py:48-52` に「H3 は loras を 422」の説明文。
- **インストーラ**（E2）: `put_*_here.txt` は git 管理（`.gitignore:36-39` の許可。`git check-ignore` で新ファイルも許可を確認）で、作るスクリプトは無い。`install-MiniMaxH3.bat`→`scripts/install_model.ps1 -BaseModel MiniMaxH3`。
- **Git**: dev＝origin/dev `65c49ae`・main＝origin/main `aceda4f`・作業ツリー空。
- **容量**: rank 128 の A/B は bf16 で約 39MB／ブロック（Turbo）、Style f32 約 6MB、Motion f16 約 3MB。CPU 正本は保存 dtype のままなので Turbo 50 ブロックで約 1.9GB。先読みの受け皿は 16GB mode（6 本）で約 +235MB、既定（10 本）で約 +390MB。固定メモリの中継（2 本）も最大ブロックに合わせて育ち、**縮まない**（`PinnedStagingPool.ensure`）。

## 3. 設計

### 3.1 境界の仕様（INTERFACE。C1／C2／C3 が共有する）

```text
# h3_lora_format.py（リポジトリ直下・torch 非依存・アプリとワーカーの両方が import）
H3_LORA_TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
LoraPairKeys(a: str, b: str, alpha: str | None)                      # frozen dataclass
H3LoraKeyMap(layout: "ab"|"kohya"|"kohya_underscore"|"mixed"|"none",
             pairs: dict[module_name, LoraPairKeys],                 # 正規名 → ファイルのキー
             unmapped: tuple[str, ...],                              # H3 の対象に写らない LoRA キー・相方の無いキー
             supported: bool, reason: str | None)
map_lora_keys(names, shapes=None, dtypes=None) -> H3LoraKeyMap
#  正規名: blocks.{i}.attn.qkv_proj … token_refiner.blocks.{j}.mlp.fc2
#  点区切り: 先頭の "diffusion_model." は任意・接尾辞 .lora_A/.lora_B.weight と .lora_down/.lora_up.weight・.alpha
#  下線: ^lora_unet_(token_refiner_)?blocks_(\d+)_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)$
#  supported := pairs が空でない AND DoRA/LoHa/LoKr のキーが無い AND unmapped が空
#               AND（shapes があれば）A.shape[0]==B.shape[1] AND（dtypes があれば）BF16/F16/F32 のみ
#  倍率はここでは計算しない（alpha の値が要る）。エンジン側で alpha÷A.shape[0]・無ければ 1.0

# 記述子（manifest）
"lora_dir": "MiniMaxH3/LoRA"          # 任意・models_dir 相対・_require_rel_path を通す
BaseModelDescriptor.lora_dir: str | None = None     # None → config.model.lora_dir（LTX は無変更）

# services/lora_registry.py
LoraScope(family, lora_dir: Path(絶対), lora_dir_display: "models/MiniMaxH3/LoRA", include_config: bool,
          verdict: Callable[[header_dict], (bool, str|None)] | None)
#   verdict が None ＝ 今日の規則（_detect_layout）で、非対応なら resolve() が 422（LTX）
#   verdict が有る ＝ 判定は表示だけで resolve() は 422 しない（H3）。strict の旗は持たない
LoraRegistry(config, scope_provider: Callable[[], LoraScope] | None = None)        # None → 今日と同じ範囲
LoraEntryInfo += supported: bool, unsupported_reason: str | None
registry.lora_dir_display -> str

# services/engines/__init__.py
lora_verdict(family) -> Callable | None                # getattr(adapter, "LORA_VERDICT", None)
lora_scope_for(descriptor, config) -> LoraScope        # ★系統→ディレクトリの対応はここ 1 か所

# GET /api/v1/loras（加算のみ。POST /loras/reload は件数だけで変更なし）
{ "loras": [ {…既存 7 項目…, "supported": bool, "unsupported_reason": str|null} ],
  "lora_dir": "models/MiniMaxH3/LoRA" }

# 機能名（H3 の UNSUPPORTED_FEATURES に追加・REJECT_TABLE には入れない）
"lora_audio_strength"

# H3 の generate ペイロード（アダプタ→ワーカー。常に送る）
"loras": [ {"path": "<絶対パス>", "strength": <float>} ]        # [] = 無し

# ワーカーの done（LoRA が 1 本以上あるジョブだけ。GOLDEN_H3_KEYS は不変）
done["h3"]["loras"] = [ {"name": "<ファイル名>", "strength", "layout", "dtype",
                          "pairs_applied": int, "skipped": [{"module","reason"}]} ]   # 絶対パスは書かない（mp4 に埋まる）

# engine_h3/lora.py
H3LoraInput(path: str, strength: float = 1.0)                # pipeline_h3.py・frozen（H3ImageInput と同型）
H3GenerateRequest.loras: tuple[H3LoraInput, ...] = ()
inspect_lora_file(path) -> H3LoraKeyMap                     # ヘッダだけ。validate_request から呼ぶ
load_lora_file(path) -> H3LoraFile(name, layout, dtype, modules: dict[name, (A, B, alpha_scale)], skipped)
attach(dit, [(H3LoraFile, strength), ...]) -> list[report]  ／ detach(dit) -> int（冪等・delattr で buffer と非永続集合を消し hook も外す）
xa_for(module, x) -> list[f32 [M, r_k]] | None               # spec が無ければ None
apply(module, x, out, n0=0, n1=None, xa=None) -> out         # bf16 の out をその場で更新
モジュール属性: buffer "_lora_A_{k}" / "_lora_B_{k}"（persistent=False・CPU・保存 dtype）、"_lora_specs"、"_lora_hook"
H3Int8Linear.forward_rows(x, n0, n1, lora_xa=None)
constants.py: LORA_COL_CHUNK = 1024   # = VRAM16_HEAD_GROUP × head_dim(128)＝16GB mode の q/k/v の行切り幅。テストで一致を固定

# 操作パネル
featureScope: UiTarget "promptBar.loraAudioStrength"・FEATURE_UI.lora_audio_strength・loraAudioStrengthHiddenFor(unsupported) -> boolean
PromptBar／LoraChips の prop: audioStrengthHidden?: boolean
SingleScreen の prop: loraAudioHidden?: boolean・loraUnsupportedNotice?: boolean
useLoras({ refreshKey }) ・ ready 状態に loraDir?: string
LoraEntry.supported?: boolean ・ LoraEntry.unsupported_reason?: string | null ・ 一覧応答に lora_dir?: string
loraTags.ts: omitLoraAudioStrength(specs) -> specs
```

### 3.2 エンジン（C1）— 数式と接続点

- **計算**: `y = bf16( f32(base(x)) + Σ_k s_k · ( f32(x) @ A_kᵀ ) @ B_kᵀ )`。`s_k = strength × (alpha ÷ rank)`（alpha 無し→1.0）。A/B は CPU に保存 dtype のまま置き、呼び出し時に `.float()`（bf16/f16→f32 は可逆）。LoRA は**生の x** を見る（ConvRot の回転はカーネル内部の話で、`forward(x) ≈ x@Wᵀ` は回転前の空間）。
- **接続は forward hook**。`forward()` を通る呼び出し（off 経路の qkv・out_proj・fc1・fc2・token_refiner の 8 本・16GB mode の out_proj と `block.mlp`）はすべて `Module.__call__` を通るので、`register_forward_hook` で `apply(module, x, out)` を掛ける。**`dit.py` は無変更・`H3Int8Linear.forward` も無変更**・token_refiner の `nn.Linear` も差し替え不要。`int8_linear` と `F.linear` の出力は新しいテンソルなので、その場で更新してよい（`H3Attention.forward` の `.view` は hook の後。レビューで確認）。LoRA が無いジョブでは hook も buffer も無く、今日と同じ経路（指紋 `0da6111c…` の根拠）。
- **`forward_rows`（16GB mode の qkv）だけは `__call__` を通らない**ので、引数 `lora_xa` を足し、カーネルの後に `if getattr(self, "_lora_specs", None): out = lora.apply(self, x, out, n0, n1, xa=lora_xa)`。無指定・spec 無しなら属性の参照 1 回だけで算術は増えない。`B[n0:n1]` の行で切る（事前調査レポートの要件）。
- **`lowvram.block_forward` の変更は 1 行＋引数**: ヘッド群のループの前に `xa = _lora.xa_for(attn.qkv_proj, h)`、3 つの `forward_rows` に `lora_xa=xa`。LoRA 無しなら `None` で算術ゼロ（「新しい計算を書かない」規則を守る）。これで `f32(h)@Aᵀ` を 21 回でなく 1 回にする（Turbo で約 +4 秒／ステップの無駄を避ける）。
- **塊の切り方は両経路で同じ形にする（ビット一致を形で保証する。レビュー Major 7 採用）**: `apply` は off 経路でも 16GB mode でも、行は `lowvram.row_chunks(M, VRAM16_ROW_CHUNK)`、**列は `LORA_COL_CHUNK`（1024）** の塊で回す。16GB mode の `forward_rows` が渡す `[n0,n1)` は幅 1024 の q/k/v の 1 切片なので、off 経路の列の塊と同じ形になる。fc1（28×1024）・fc2・out_proj（5.25×1024。端の塊 256 は両経路で同じ）も同じ規則で切るので、cuBLAS に渡る形は off と on で全モジュール一致する。qkv だけの特別扱いは作らない。`LORA_COL_CHUNK == VRAM16_HEAD_GROUP × head_dim` を CPU テストで固定。
- **加算の形（レビュー Major 6 採用。1 要素 20→12 バイト）**: 行の塊 `[r0,r1)` ごとに `xs_k = f32(x[r0:r1]) @ A_kᵀ × s_k`（xa を 16GB mode で渡された場合はその切片に s_k を掛ける）、列の塊 `[c0,c1)` ごとに `delta = mm(xs_0, B_0[c0:c1]ᵀ)`（再利用する f32 の塊 `[行, 1024]`≈16MB）、2 本目以降は `delta.addmm_(xs_k, B_k[c0:c1]ᵀ)`、最後に **`out[r0:r1, c0:c1].add_(delta)`**（bf16 の受け皿に f32 を足す in-place は共通 dtype f32 で計算して 1 回だけ bf16 に丸める。`torch.can_cast(f32, bf16)` は真）。読み書きは bf16 読 2＋f32 書 4＋f32 読 4＋bf16 書 2＝12 バイト／要素。
- **attach／detach の位置**（`pipeline_h3.py`）: 
  - **早い段階の検査（レビュー Major 2 採用）**: `validate_request` で各 LoRA の存在・ヘッダ・`inspect_lora_file`（鍵の写像）を確かめ、ファイルが無い・壊れている場合は**文章の符号化の前**に日本語で拒否（画像の前例と同じ。常駐 DiT を失わない）。
  - `_denoise` の `try:` 先頭（`_vram16_begin`・`_place_dit` の前。`:897-898`）で `load_lora_file`→`attach`（CPU のまま。先読みの `_plan` が buffer を一緒に詰める）。attach 後は `H3LoraFile` の辞書を捨てる（token_refiner 分の重複を避ける。Minor 15）。
  - `finally` の `service.release_installed()` の直後・`if ok and keep:` の前（`:987-988`）で `detach(dit)` を**無条件**に呼ぶ。**順序の制約（Major 5）**: `teardown_prefetch` の `_release` は `mod._buffers[name]` を名前で書き戻すので、detach は必ず teardown の後。この制約をコードのコメントと設計正本 §6.6 に書き、CPU テスト（teardown→detach の呼び出し順）で固定する。
  - `detach` は `delattr(module, name)`（`_buffers` と `_non_persistent_buffers_set` の両方が消える。Minor 13）＋`handle.remove()`。meta／CPU／GPU のどれでも安全・冪等。
- **形の検査**: `attach` で `A:[r, in_features]`・`B:[out_features, r]` を照合。合わない組は**飛ばして記録**（裁定 2＝実行できる）。対象がモデルに無い名前（ブロック番号超過等）・dtype が BF16/F16/F32 以外の組も同じ。`H3Int8Linear` と `nn.Linear` だけを対象にする（他は記録）。`apply` に `except` を書かない（失敗は失敗）。
- **FP32 の保証（レビュー Major 3・Minor 20 採用。条件は実装時に訂正）**: `H3Pipeline.load` の `require_cuda_int8()` の隣で 1 回 `require_fp32_matmul()`: `allow_tf32 is False`・`torch.get_float32_matmul_precision() == "highest"`・（属性があれば）`torch.backends.cuda.matmul.fp32_precision in ("ieee", "none")` の 3 条件を確かめ、違えば `RuntimeError`（fail loud。done への記録はしない）。torch 2.13 の既定は `fp32_precision="none"`（＝親設定を継承）で、効いているのは旧来の旗なので、計画の当初の「`== "ieee"`」は全ロードを拒否する誤りだった（C1 の実測でエスカレーション→監督が訂正）。CPU テストで、bf16 の行列積なら必ず違う値になる入力（大きな K・打ち消し合う値）を使って f32 の明示的な参照と `torch.equal`、さらに `torch.mm`／`addmm_` を差し替えて LoRA の全オペランドが f32 であることを確かめる。
- **ファイルの読み**: `sft_quant_format.read_header`＋`engine.sft_quant.sft_reader.read_tensors`（連続・保存 dtype）。**ジョブ間のキャッシュは作らない**（Turbo 約 2GB の読みは NVMe で 1〜3 秒。常駐 DiT を清潔に保つ設計と相性がよい）。
- **要求とワーカー**: `H3GenerateRequest.loras`、`validate_request` でパス空・`strength<=0`・ファイル無し・ヘッダ不良を日本語で拒否。`worker._GENERATE_FIELDS += "loras"`、`_LORA_FIELDS={"path","strength"}`、`_images` と同型の `_loras(msg)`（内側の知らないキーは記録）。セルフテストに `--lora PATH`（繰り返し可・パスのみ）と `--lora-strength X`（全 LoRA に共通・既定 1.0。Minor 12）。`_selftest_generate_message` は常に `"loras"` を送る（GOLDEN_GENERATE_KEYS を更新）。
- **全ブロック常駐（`blocks_on_gpu` が 0 以下か 50 以上）のとき**は LoRA の buffer も全部 GPU に載る（Turbo で約 +1.9GB）。ベースの重みも全部 GPU にある条件なので整合するが、「50 ブロック分を常駐させない」要件の例外として設計正本に明記（Major 4）。

### 3.3 バックエンド（C2）

- **記述子**: `scripts/manifests/40-minimaxh3.json` に `"lora_dir": "MiniMaxH3/LoRA"`。`services/base_models.py` に欄 `lora_dir: str | None`（`_require_rel_path`）。LTX の manifest は触らない（無ければ `config.model.lora_dir`＝今日の挙動）。
- **系統→ディレクトリの対応は `services/engines/__init__.py::lora_scope_for(descriptor, config)` の 1 か所**: 記述子に `lora_dir` があればそれ（`include_config=False`＝`ic_loras` の設定登録を含めない。H3 の一覧に LTX の IC-LoRA を出さない）、無ければ `config.model.lora_dir`（`include_config=True`）。`verdict` は `lora_verdict(family)`（アダプタの `LORA_VERDICT`。LTX は無し＝`_detect_layout`・422 あり）。
- **登録簿**: `LoraRegistry(config, scope_provider)`。`rescan()` は provider の範囲で走査し `(scope_key, entries, scope)` を 1 回の代入で差し替える。`resolve()`／`info()`／`preprocess_for()` は先に `_ensure_scope()`（provider の鍵が前回と違えば再走査）＝**切替直後のジョブが古い系統の一覧から名前解決することを防ぐ**。`resolve()` の 422 は `not entry.supported and scope.verdict is None`（LTX は今日どおり・H3 は 422 しない）。`preprocess_for` は範囲内の entry から返す（設定を直接見ない。Minor 10）。`_detect_layout` は無変更（`test_lora_registry.py:466` はそのまま）。`as_dict` に 2 項目追加（鍵集合のテストを更新）。
- **系統の出どころは 1 つ（レビュー Major 1 採用）**: provider は `/generate` の検証と同じ **runner の記述子**（`pipeline_manager.runner.descriptor`）を見る。構築時（`api/context.py`）は `LoraRegistry` を `set_active_base_model(...)` の後・`PipelineManager` の前へ動かし、provider は `pipeline_manager` が無い間だけ `model_registry.descriptor()` で代用（`__init__` が即 `rescan()` するため）。
- **`api/loras.py`**: 応答に `lora_dir` を足す。`api/generate.py`・`pipeline_manager.py` の呼び出しは**無変更**。
- **H3 アダプタ**（`services/engines/h3/adapter.py`）: REJECT_TABLE から `loras` 行を削除・`HONOURED_FIELDS` に `loras`・`GOVERNED_FIELDS` の 2 つの強度は `reference_video_id` が統治（参照動画の強度なので意味も合う。`test_every_governor_is_itself_refused` も成り立つ。H3 で `loras` と一緒に強度が来ても黙って無効＝GOVERNED の定義どおり。仕様書に明記）・`UNSUPPORTED_FEATURES` の末尾に `"lora_audio_strength"`・`LORA_VERDICT = _h3_lora_verdict`（`h3_lora_format.map_lora_keys(names, shapes, dtypes)`）・**`validate_for_family` で各 `LoraSpec.audio_strength` を `None` に正規化**（ステップ数の既定値の書き戻しと同じ前例。metadata の `ic_lora` にも無視した値が残らない。Gradio／MCP 側の落とし込みは不要になる。Minor 10・11）・generate ペイロードに `"loras": [{"path","strength"}]` を常に載せる（`request.loras` と `lora_paths` を一緒に辿る＝ソース監査テストも満たす。Minor 8）。
- **metadata.json**: 無変更（`ic_lora` 欄は系統非依存。`audio_strength` は上の正規化で None になる）。エンジン側の記録は `metadata.h3.loras`（既存の `_h3_block` 経由・ファイル名のみ）。
- **Gradio**: 非対応の行は caption を `f"{name} (非対応)"`。Load 後の `.then` にギャラリー更新を足す（モデル切替で古いままの件）。`style_note` の文言を更新。**`drop_audio` の配管は作らない**（サーバーが正規化する）。
- **MCP**: `list_loras` の説明（系統ごと・`supported`・`lora_dir`）、`server.py:48-52`（H3 は `loras` 可・`audio_strength` は無視・`reference_video_id`／`submit_chain`／`nag_enabled` は引き続き 422）、`params.py` の LoraArg、`tools/generate.py:205-207`。
- **インストーラ**: git 管理の `models/MiniMaxH3/LoRA/put_LoRA_safetensors_here.txt`（LTX23 と同じ様式。ComfyUI 名／kohya 名と `.png` サムネイルの説明）＋`scripts/install_model.ps1` の H3 分岐に `New-Item -ItemType Directory -Force` 1 行＋`install_model.ps1:262-268`／`setup.ps1:306-310` の利用者資産の一覧に `models\MiniMaxH3\LoRA`。

### 3.4 操作パネル（C3）

- `shell/featureScope.ts`: UiTarget `"promptBar.loraAudioStrength"`・行 `lora_audio_strength`・`loraAudioStrengthHiddenFor()`。`:172` の「`loras` に受け口は無い」注記を直す。
- `lora/LoraChips.tsx`: prop `audioStrengthHidden`。真なら 🎥・トグル・音声値の 3 つを描かない。`PromptBar.tsx` は素通し。
- `shell/AppShell.tsx`: `loraAudioHidden` を導出して PromptBar と SingleScreen へ。**既存の** `useLoras()`（`:317`）に `refreshKey: baseModels.active` を渡す。`loraUnsupportedNotice`＝`parseLoraPrompt(prompt).loras` の名前が一覧の `supported===false` に当たるか（useMemo）。InventoryScreen に `refreshKey`。
- `useLoras` の **4 か所すべて**に `refreshKey` を渡す（Inventory・AppShell・`useOutpaintForm.ts:404`・`useBatchI2vLongForm.ts:436`。例外を作らない。Minor 16）。
- `modes/single/SingleScreen.tsx`: `buildSubmission` で `loraAudioHidden` なら `omitLoraAudioStrength(request.loras)`（`omitStrength` の隣・同型）。`<GenerateReasonsNote/>` の直後に `<p className="warning-banner warning-banner-mild" role="status">`（`generateReasonCodes` には入れない＝送信を止めない）。文言は §7-6。
- `lora/loraTags.ts`: `omitLoraAudioStrength`（純関数）。
- `modes/inventory/useLoras.ts`／`InventoryScreen.tsx`／`LoraCard.tsx`: `refreshKey` で再取得・`loraDir` を保持・`supported===false` に「非対応」バッジ（`title` に理由）・空表示文言はサーバーの `lora_dir` を使う。
- `api/types.ts`・`i18n/strings.ts`（EN／JA。注意文・バッジ）・`bridge/mockBridge.ts`（H3 の非対応一覧から `loras` を外し `lora_audio_strength` を足す・`MOCK_LORAS` に `supported:false` を 1 件・モックの一覧に `lora_dir`）。
- Library のカードを押したときのタグは **`<lora:名前:1.0:1.0>` のまま**（例外を増やさない。H3 では 3 つ目は送られずチップにも出ない）。
- バッチ・Chained・Edit は H3 で閉じているので `audio_strength` の省略は Single だけに置く（keyframe の前例と同じ。サーバー側の正規化が二重の守り）。

### 3.5 数値の約束と費用

- **LoRA 無しはビット一致**: attach が呼ばれず hook も buffer も無い。`forward_rows` は属性参照 1 回だけ。先読みの `_plan` が見るテンソルも今日と同じ（受け皿の大きさも同じ）。J1（LoRA あり・常駐）→J2（LoRA なし）は detach が `_park_dit` の前に走るので、J2 の DiT は attach 前と同一（buffer・非永続集合・hook の集合をテストで照合）。固定メモリの中継だけは大きいまま残る（数値には無関係。設計正本に記す。Minor 14）。**門＝指紋 `0da6111c…` の不変（off／on）**。
- **LoRA ありの off と 16GB mode はビット一致を形で保証する**: xa（`x@Aᵀ`）は両経路で同じ 4096 行の塊・同じ形で cuBLAS に渡る。B 段は両経路で同じ 1024 列の塊（§3.2）。ベースの出力は裁定 46 で一致済み。よって門 4・5 は **off と on の mp4 の sha256 一致**を合格条件にする（sage は今日どおり 2 ブロックで値が違うので門は sdpa）。
- **費用の見積もり**（S≈4 万行・50 ブロック・推定。実測は門で取る）: 行列積は Turbo（rank 384／128）で約 7.5e13 FLOP／ステップ≈1.3 秒、rank 16 で≈0.1 秒。f32 の受け皿の往復は 12 バイト／要素で約 1.5TB／ステップ≈1.5〜2 秒。小さな GEMM の起動は約 3.5 万回／ステップ／LoRA≈0.3 秒。合計で Turbo 約 +3 秒／ステップ、rank 16 で約 +2 秒（現状約 22 秒／ステップに対し +10〜15%）。これは「LoRA の加算は FP32 で」という要件に固有の費用。後続課題（台帳 §3）: `comfy_kitchen.int8_linear` が持つ `residual`／`residual_scale`（エピローグで差分を足す既存の口。dtype の扱いは未確認）の利用・Turbo のブロック対角の利用（約 −30% FLOPs）。
- **VRAM**: 受け皿は保存 dtype の分だけ増える（Turbo bf16 約 39MB／ブロック→16GB mode で約 +235MB）。f32 の一時の塊は `[≤4096, 1024]`≈16MB。qkv の xa は `[S, 384]` f32≈61MB／Turbo 1 本。主メモリは Turbo で約 +1.9GB（コミット量を門で記録）。

## 4. 作業分割と担当（1 ファイル 1 所有者・サブエージェントは Fable 禁止）

- **最初の一手（監督）**: `uploads/LoRA-MiniMax-H3/` の 3 本を `models/MiniMaxH3/LoRA/` へ移動（Uploads は一時置き場）。
- **C1 エンジン（Opus）**: `h3_lora_format.py`（新・最初に書く。C2 が import する）・`engine_h3/lora.py`（新）・`engine_h3/constants.py`（`LORA_COL_CHUNK`）・`quant_linear.py`（`forward_rows` の引数）・`lowvram.py`（1 行＋引数）・`pipeline_h3.py`・`worker.py`・`tests/test_h3_lora_format.py`（新）・`tests/test_h3_engine.py`。CPU テストは `_fake_int8`＋tiny DiT。GPU の門は監督が実行（§5）。
- **C2 バックエンド（Opus）**: `scripts/manifests/40-minimaxh3.json`・`services/base_models.py`・`services/lora_registry.py`・`services/engines/__init__.py`・`api/context.py`・`api/loras.py`・`services/engines/h3/adapter.py`・Gradio（`adapters.py`・`ui.py`・`i18n.py`）・MCP（`tools/system.py`・`server.py`・`params.py`・`tools/generate.py`）・インストーラ（`install_model.ps1`・`setup.ps1`・案内ファイル）・テスト（`test_base_models.py`・`test_lora_registry.py`・`test_loras_endpoint.py`・`test_engine_h3_adapter.py`・`test_engine_dispatch.py`・Gradio・MCP）。
- **C3 操作パネル（Opus）**: §3.4 の列挙ファイルとテスト（`featureScope.test.ts`・`LoraChips.test.tsx`・`loraTags.test.ts`・`AppShell.minimaxH3.test.tsx`・`useLoras.test.ts`・`InventoryScreen.test.tsx`）。`npm run typecheck`→`npx vitest run --exclude "**/backend.integration.test.ts"`。aux2 は `build.ps1`／`deploy.ps1`（配置は AviUtl2 停止時）。
- **文書係（Sonnet）**: コードの後（§6）。
- **コードレビュー（Opus）**: C1〜C3 の完了後にそれぞれ 1 本。
- 全担当への明記: 勝手な設計変更は禁止。設計に疑問を抱いたら手を止め、根拠（実測値・エラー全文）を添えてエスカレーションする権限を与える。担当外のファイルとバックエンドの凍結部分（`engine/`・`/api/v1` のスキーマ）は変更しない。
- 並行の順序: C1 は `h3_lora_format.py` を最初に書き、C2 はそれが置かれてから verdict を結線（それまでは registry／記述子／アダプタ側を進める）。C3 は INTERFACE だけで独立に進める。

## 5. 門（ゲート）と検証

### 5.1 機械の門（担当が回す。CPU）
- エンジン: `.venv-engine-h3\Scripts\python.exe scripts\run_h3_pytest.py tests\test_h3_lora_format.py tests\test_h3_engine.py …`。主なテスト: 鍵の写像（点区切り・下線・token_refiner・一部欠け＝対応・AdaLN キー＝非対応・DoRA/LoHa/LoKr・相方欠け・rank 不一致・torch 非依存を `-I` の子プロセスで確認）／LoRA 無しの forward が無変更（hook・buffer が無い・`torch.equal`）／attach→detach で buffer・非永続集合・hook・state_dict の鍵が元に戻る／**FP32 の証明**（bf16 なら必ず違う入力で f32 参照と `torch.equal`・`mm`/`addmm_` の差し替えで全オペランド f32）／alpha の倍率と alpha 無し／kohya 名と A/B 名で同じ出力／形の不一致は飛ばして記録／複数 LoRA の累積／token_refiner に hook 経由で当たる／`test_vram16_block_forward_matches_off` の LoRA 付き版（CPU で `torch.equal`）／`LORA_COL_CHUNK == VRAM16_HEAD_GROUP×head_dim`／qkv の xa が 1 ブロック 1 回／先読みの `_enumerate_slots` に `_lora_A_0` が並ぶ／**同期 swap 経路**（`_patch_block`）で buffer がブロックと一緒に動き detach 後に残らない／teardown→detach の順序／常駐 J1（LoRA）→J2（なし）＝基準／失敗ジョブでも detach／buffer の dtype 監視／`validate_request`（パス・強度・ファイル無し・ヘッダ不良が符号化の前に出る）／ワーカーの `loras` 解析と内側の無視キーの記録／`--lora`・`--lora-strength`／GOLDEN_GENERATE_KEYS 更新・GOLDEN_H3_KEYS 不変・`done.h3.loras` は LoRA ありのときだけ。GPU opt-in（`H3_GPU_TESTS=1`）: 実ファイル 3 本の attach 件数（Turbo 208・Style 200・Motion 208）。
- アプリ: `.venv` で `python -m pytest tests`（§175 の条件: `tools\ffmpeg\bin` を PATH 先頭・`-o addopts=""`・MCP の 1 件 deselect）。主なテスト: 記述子の `lora_dir` 解析（絶対パス・`..` 拒否）／範囲の切替で `resolve` 前に再走査／H3 で下線 kohya が supported／H3 の非対応ファイルが 422 にならない／LTX は従来どおり 422（`:466` 不変）／H3 の範囲に `ic_loras` が出ない／`preprocess_for` が範囲内の entry を見る／`as_dict` の鍵集合／`GET /loras` の `lora_dir`／H3 アダプタ: `REQUEST_OVERRIDES_H3` から `loras` を外す・`:734` の parametrize を `reference_video` に差し替え・`UNSUPPORTED_FEATURES` の golden・`validate_for_family` が `audio_strength` を None にする・ペイロードに常に `loras`・`GOLDEN_GENERATE_KEYS_H3` の順序・HTTP で 202・未知の LoRA は 404／`lora_verdict("ltx") is None`／Gradio（caption・Load 後の更新）／MCP（`supported`・`lora_dir` の素通し）。
- 操作パネル: `npm run typecheck`→vitest 全件（`backend.integration.test.ts` 除外）。

### 5.2 GPU の門（監督が実行。各段でオーナーの了承。sdpa・deterministic）
1. LoRA 無し・16GB mode off: `g2_selftest_g5prompt.cmd` で指紋 `0da6111c…` 不変。
2. LoRA 無し・16GB mode on（`--vram16`）: 同じ指紋。
3. 常駐の清潔さ（アプリ経由・`keep_resident`）: J1 Turbo → J2 Style → J3 LoRA なし。J3 の sha256 がアプリ側の LoRA 無し基準と一致・J2 が J1 の LoRA を引き継がない（J2 の sha256 が「Style 単独・常駐なし」の sha256 と一致）。
4. Turbo T2V 4 ステップ: セルフテスト `--lora models\MiniMaxH3\LoRA\minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors --steps 4 --shift-video 6 --shift-audio 3`、768p（例 1344×768×124f）、off と on の 2 本。**合格＝off と on の sha256 一致**。秒／ステップ・VRAM 山・RSS 山・コミット量の山・`done.h3.loras` を記録。目視はオーナー。
5. Style 3dsrx と Motion（20 ステップ・G5 の条件）＋混合（Style＋Motion）、off と on（sha256 一致）。Style はさらに **先読み off（同期 swap）** で 1 本。門 1 との秒数・山の比較。目視はオーナー。
6. アプリの端から端（オーナー目視）: H3 の Inventory 一覧とモデル切替での更新・「非対応」バッジ（壊したファイルを 1 つ置く）・黄色の注意文・チップに音声トグルと絵文字が無い・LTX 側が無変更・Gradio・MCP。aux2 の配置は AviUtl2 停止時。

### 5.3 完了の条件
全テスト合格・門 1〜6 合格・文書（§6）・コードレビュー 3 本の指摘を処理・dev にコミット（エンジン／バックエンド／操作パネル／文書の分割）→オーナーの合図で `--no-ff` で main へ。

## 6. 文書（文書係・Sonnet。コードの後）

- `Docs/MINIMAX_H3_DESIGN.md`: 新節「6.6 LoRA（実行時の低ランク経路）」（式・hook・`forward_rows` の行切り・1024 列の塊と一致の根拠・attach/detach の位置と teardown との順序・全ブロック常駐の例外・固定メモリの中継が育つこと・費用・非対応の判定）、§12 から「LoRA・Turbo」を外す、§13 に後続課題。
- `Docs/MINIMAX_H3_RESEARCH_NOTES.md` §10: 裁定 62〜72（本計画 §1 の 11 件）＋設計上の判断（§7）。
- `Videomni_Backend_Specification.md` v0.5.99: `:524`（木に `MiniMaxH3/LoRA/`）・`:739`（`loras` の登録は記述子の `lora_dir` か `lora_dir` 設定）・`:759-761`・`:891`（H3 の非対応一覧から `loras` を外し `lora_audio_strength` を足す）・`:915-920`（LoraSpec の `audio_strength` は H3 で None に正規化・2 つの強度は `reference_video_id` 統治）・`GET /loras` の `supported`／`unsupported_reason`／`lora_dir`・`:1425-1432`（H3 は書式で 422 しない）・`:1979`。
- `Docs/MULTI_ENGINE_DESIGN.md:382-384`: 「LoRA はベースモデル軸を持たない」を記述子の `lora_dir` の説明に改める。
- `README.md`（凍結ゾーン 1〜131 行は触らない）: `:382-386` に H3 の置き場所と対応形式・`:413-418` の木に `LoRA/`・`:981` の表の行（H3 で LoRA 可・IC-LoRA は不可）。
- フロントエンド `Docs/API_REFERENCE.md` §3.6（系統ごとの一覧・新項目・`:262` の「書式で絞らない・422」の記述を H3 の扱いに合わせる）／§3.7（reload は件数だけ）／§5.3（H3 の `audio_strength`）・フロントエンド README・`DEVLOG.md` §148。
- `mcp_server/server.py` の INSTRUCTIONS（コードと一緒に C2）。
- `Docs/VERIFICATION_LOG.md` §176（門 1〜6 の記録・費用の実測）。台帳: §1-104 → `PENDING_TASKS_CLOSED.md` 3-251・§3-249「H3 の LoRA で動画・音声に個別の強度」・§3-250「H3 の LoRA 加算の高速化（`int8_linear` の `residual` エピローグ・Turbo のブロック対角）」・Library のタグを H3 で 2 引数にするかは §3-249 の留意に 1 行。`§1-105` の前提文は「§1-104 の後」のまま成立。
- 保管: 棚卸し R1〜R3・設計案・レビュー・本計画を `Docs/Outputs-archive/h3-lora-2026-10-24/` に置く。

## 7. 設計上の判断（レビューの問いへの監督の回答。了承時にオーナーが覆せる）

1. **「非対応」の意味**: H3 の対象に写らないキーが 1 つでもあれば非対応（AdaLN・norm 等）。対象の一部が無いだけ（token_refiner なし）は対応。
2. **IC-LoRA（`config.yaml` の `ic_loras`）は H3 の一覧に出さない**（記述子が `lora_dir` を持つ系統は設定登録を含めない）。
3. **off と 16GB mode の一致は形で保証**（1024 列の塊を両経路で使う）。測って判断する分岐は作らない。
4. **費用**: v1 はこの費用を受け入れ、高速化は台帳 §3-250 に起票。ジョブ間の LoRA キャッシュは作らない。
5. **Library のカードのタグ**: `<lora:名前:1.0:1.0>` のまま（UI の例外を増やさない）。
6. **注意文の文言（オーナー裁定 73・2026-10-24 了承後）**: **原文のまま**「LoRAが非対応の形式です。効果がない可能性があります。」（EN: "This LoRA's format is not supported. It may have no effect."）。LTX でも同じ条件で同じ文言が出る（LTX では送信時に 422 のエラーカードも並ぶ）。応答に旗は足さない。

## 8. 範囲外・後回し
diffusers/PEFT の分離 QKV・量子化 LoRA・LoHa/LoKr/DoRA・Text Encoder 用 LoRA・Ref2VA 用の重みとの互換の確認（Ref2VA 自体が未実装）・LoRA 融合済み配布物の扱い（既存どおり普通の重みとして読む）・快適上限の較正（§1-95）・LoRA 加算の高速化（§3-250）。

## 9. 敵対的レビューの取捨（記録）
- 採用（Major）: 1 系統の出どころを runner の記述子に統一／2 ファイルの検査を文章の符号化の前へ／3 FP32 の証明テスト／4 同期 swap 経路の試験と全ブロック常駐の明記／5 teardown→detach の順序の明文化と試験／6 加算を `mm`＋`add_` の 12 バイト形へ／7 qkv の B 段を 1024 列の塊に（両経路で同形＝一致を形で保証）。
- 採用（Minor）: 8 `request.loras` をアダプタで読む・golden 更新／9 統治の変更を仕様書に／10 `audio_strength` をサーバーで None に正規化・`preprocess_for` を範囲内に／11 Gradio の `drop_audio` を作らない／12 `strict` 旗・`lora_fp32_matmul` の記録・`--lora` の `:強度` 構文をやめる／13 `delattr` で消す／14 固定メモリの中継の注記／15 attach 後にファイルの辞書を捨てる／16 `useLoras` 4 か所に `refreshKey`／18 done にパスでなくファイル名／19 コミット量の記録と J1 Turbo→J2 Style／20 新 API の TF32 確認／21 reload は件数のみと明記。
- オーナー判断: 17 注意文の文言（§7-6）。
- 棄却: なし。
