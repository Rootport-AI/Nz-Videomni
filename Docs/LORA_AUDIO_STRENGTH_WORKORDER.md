# Style LoRA 音声強度制御（audio_strength）ワークオーダー

> **【ステータス追記 2026-08-02】実装完了・テーマ完結。** バックエンドAPI＋MCP（Model Context Protocol。Claudeなどのエージェントがツールを呼び出すための標準プロトコル）＋Gradio（バックエンド検証用のWeb UI）＋AviUtl2フロントエンドまで一気通貫で実装済み・自動ゲート全緑・オーナー実機A/Bゲート全項目合格・LoRAチップUIの目視確認も合格（詳細は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §45.2/§45.3）。**本文は実装内容の記録として運用する**（今後この機能の仕様を確認したいときはここを読む）。
>
> 併読: [`STYLE_LORA_WORKORDER.md`](STYLE_LORA_WORKORDER.md)（LoRA適用機構そのものの起票元。書式はこれに倣っている）／フロントエンド[`API_REFERENCE.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/API_REFERENCE.md) §5.3（API利用者向けの正本）。

- 作成: 2026-08-02（実装完了と同時に起票）
- 正本ポインタ: API仕様＝フロントエンド`API_REFERENCE.md` §5.3／機械検証＝本リポジトリ`VERIFICATION_LOG.md`／実機A/B結果＝同書の該当節

## 1. 背景

LTX 2.3では、Style LoRA（画風・キャラクターLoRA。映像の見た目を変える目的で訓練された追加学習の差分重み）を適用した状態で動画を生成すると、**音声出力が壊れる**（雑音・音割れ）症状がコミュニティで複数報告されていた。原因の仮説は、「映像目的で訓練されたLoRAの重みファイルの中に、音声側の層に対応する差分重みも同時に含まれており、それが音声生成を汚染している」というもの。LTX 2.3のトランスフォーマーは映像ストリームと音声ストリームを持つデュアルストリーム構造で、LoRAは通常どちらのストリームの層にも無差別にキーを持つため、映像用に調整されたLoRAでも音声側の層にランダムに近い差分が乗ってしまう。

対処として、LoRAごとに音声側の適用強度を映像側とは独立に制御できる`audio_strength`をAPIに追加した。軸は2つ（映像軸・音声軸）に確定した（4軸案の経緯と再訪条件はフロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §4-24）。

計画段階でPlanエージェントが起案した計画に敵対的レビューを実施し、BLOCKER 2件（うち1件は`parseLoraPrompt`のコールバック署名変更漏れ——タグの引数が2個から3個に増えると文字列中の出現位置がズレ、チップの±/×ボタンが誤った位置を編集する不具合）を含む指摘を実装前にすべて計画へ反映してから着手した。

## 2. 仕様

### 2.1 API（`api/models.py::LoraSpec`）

```
audio_strength: float | None = Field(None, ge=0.0, le=2.0)
```

- 型: `float | None`。範囲: 0.0〜2.0（**0を許容**）。既定: `None`（省略）。
- 既存の`strength`は`gt=0.0`（0拒否）のまま変更していない。`audio_strength`だけが0を許容するのは、「0=音声側の重みを一切適用しない」を表現するため。
- 省略（`None`）時は音声側も`strength`の値にそのまま追従し、`audio_strength`を持たない従来のリクエストと完全同一の挙動になる（後方互換。詳細は§6）。

### 2.2 タグ構文（Gradio UI／AviUtl2フロントエンド共通の慣習）

```
<lora:名前:映像の強さ:音声の強さ>
```

第3位置引数が音声の強さ。省略時は映像の強さにそのまま追従する。Gradio側は`gradio_ui/handlers.py::_LORA_TOKEN_RE`、フロントエンド側は`webui/src/lora/loraTags.ts::LORA_TAG_RE`がそれぞれ対応する正規表現を持つが、許容する引数の文字種が異なる——Gradio側は数値限定マッチ、フロントエンド側は任意文字列にマッチしたうえで`normalizeStrength`が数値へ正規化する。この差が§2.5の非対応構文で挙動の食い違いを生む。

### 2.3 クランプ範囲

| 軸 | クランプ範囲 | 備考 |
|---|---|---|
| 映像(`strength`) | 0.05〜2.0 | 既存のまま変更なし |
| 音声(`audio_strength`) | 0.0〜2.0 | 下限が映像側と異なる（0を許容） |

### 2.4 既定挿入トークンの変更

画風LoRAのカード（Library・Gradioギャラリー共通）をクリックしたときの既定挿入トークンは、`<lora:名前:1.0>`（2引数）から**`<lora:名前:1.0:1.0>`（3引数）**へ変更した。第3引数を最初から見せることで、音声強度制御という機能の存在に気づけるようにする狙い。音声側`1.0`は映像追従時の実効値と数値が同一なので、この変更自体は生成結果に影響しない。

### 2.5 `<lora:名前::0>`（映像側省略）の扱い

`<lora:名前::0>`のように映像側の引数だけを省略する記法は、**Gradio側とフロントエンド側で解釈が食い違うため使用しないこと**。Gradio側の`_LORA_TOKEN_RE`は数値限定マッチのため空の第2引数にはマッチせず、タグはLoRAとして解釈されずプロンプト本文にそのまま残る。フロントエンド側の`LORA_TAG_RE`は任意文字列にマッチするため空の第2引数でもタグとして認識され、`normalizeStrength`が既定値`1.0`を返して`{strength: 1.0, audio_strength: 0}`としてAPIへ送信される（映像側は既定強度で適用、音声側だけ0）。音声側だけを指定したい場合も、映像側の強さを明示すること。フロントエンド側の`formatLoraTag`は音声強度を出力するときに必ず映像強度も明示するよう設計されており、UI操作からこの記法が生成されることはない（詳細は§6）。

## 3. 分類ルール（音声軸／映像軸判定）

LoRAの重みキーのモジュール接頭辞（例`transformer_blocks.0.audio_attn1.to_q`）をドット分割し、コンポーネント名を次の順序で判定する（`engine/gguf/ic_lora_common.py::classify_lora_axis`）。**判定順が仕様であり、順序を入れ替えてはならない**（`audio_to_video_attn`は文字列として`audio_`で始まるため、先に完全一致の特例を評価しないと誤って音声軸に分類されてしまう）。

| 判定順 | コンポーネント | 軸 | 根拠（`engine/transformer.py`での書き込み先ストリーム） |
|---|---|---|---|
| 1 | `video_to_audio_attn` | 音声 | `ax = ax + ...`（音声ストリームへ加算、L385-395） |
| 2 | `audio_to_video_attn` | 映像 | `vx = vx + ...`（映像ストリームへ加算、L353-363） |
| 3 | `av_ca_v2a_gate_adaln_single` ／ `av_ca_audio_scale_shift_adaln_single` | 音声 | v2a（映像→音声）項のゲート／音声側scale-shift |
| 4 | `av_ca_a2v_gate_adaln_single` ／ `av_ca_video_scale_shift_adaln_single` | 映像 | a2v（音声→映像）項のゲート／映像側scale-shift |
| 5 | `audio_`で始まる | 音声 | `audio_attn1`／`audio_attn2`／`audio_ff`／`audio_adaln_single`／`audio_patchify_proj`等 |
| 6 | それ以外 | 映像 | `attn1`／`attn2`／`ff`／`adaln_single`／`patchify_proj`等 |

未知のトップレベル接頭辞（`_KNOWN_VIDEO_TOP_LEVEL`に無いもの）は映像扱いにしたうえでDEBUGログを出す。判定順1・2はクロス注意の**書き込み先ストリーム**で軸を決める特例で、名前の由来ストリームとは逆になる点に注意（`video_to_audio_attn`は映像を参照して音声を更新する層なので音声軸、`audio_to_video_attn`はその逆で映像軸）。判定順3・4も同様に、v2a（映像→音声方向）の変調項は音声軸、a2v（音声→映像方向）の変調項は映像軸として扱う。

実在のLoRAファイル（`ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors`、3320キー）を用いて`av_ca_*`4種の存在を実測確認済み。

## 4. `audio_strength=0`をスキップ実装にした理由

`audio_strength=0`のキーは、LoRAバッファの登録・forward時の適用specsへの追加を**行わない**（`continue`でスキップする）。数式上は「delta（差分重み）に0を掛けて加算する」ことと**数学的に等価**だが、あえて計算そのものをスキップする実装にしたのは次の理由による。

- **VRAM（forward時attach経路のみ）**: スキップした分のLoRAバッファ（A/B行列）をGPUへ載せずに済む。bf16融合経路はもともとA/B行列をバッファとして保持しないため、この効果はない。
- **速度（両経路）**: フォワード時にdelta計算（`fp32`でのmatmul→キャスト→in-place加算）自体を丸ごと省略できる。bf16融合経路でも、省けるのはロード時のfp32一時確保とmatmulの計算コストである。

このスキップ判定は、forward時attach経路（`attach_ic_loras`）とbf16融合経路（`_fuse_ic_loras`）の両方で同じ`strength_for_prefix`ヘルパーの戻り値`0.0`をトリガーに行う。

## 5. 2経路対応（forward時attachとbf16融合）

エンジンは2つの独立したLoRA適用経路を持つ（既存の設計、§8参照）。

- **forward時attach経路**（per-layer-quant、`engine/gguf/ic_lora_common.py::attach_ic_loras`）: LoRAのA/B行列をバッファとして対象`nn.Linear`に登録しておき、forward呼び出しのたびに、その場でdequantした重みへdeltaを加算する。
- **bf16融合経路**（`engine/gguf/loader_service.py::_fuse_ic_loras`）: モデルロード時に一度だけ、BF16の状態辞書へdeltaを直接足し込んでしまう（load-time fuse）。

`audio_strength`はこの両方の経路に同一の`strength_for_prefix(prefix, strength, audio_strength)`ヘルパーを通す形で配線した。`audio_strength is None`のときはこのヘルパーが分類を一切行わずに`strength`をそのまま返す（G-BC、§6参照）。`audio_strength`が非Noneのときだけ`classify_lora_axis`を呼び、音声軸キーには`audio_strength`を、映像軸キーには`strength`を適用する。delta計算式（fp32 matmul→単一キャスト→in-place加算）自体は両経路とも不変で、`audio_strength`はどの強さを使うかの入力を差し替えるだけである。

## 6. 後方互換の不変条件（G-BC）

`audio_strength`を指定しない（省略する）リクエストについて、以下は現行と**完全に同一**であることをゲートしている。

- ワーカーメッセージの`loras`ペイロード（`audio_strength`キー自体が出ない）。
- Gradio UI／MCP（Model Context Protocol）がつくるHTTPボディ（dictに`audio_strength`キーが入らない。MCP側は`model_dump(exclude_none=True)`）。
- エンジンのdelta計算経路（`audio_strength is None`のとき`strength_for_prefix`は分類を経由せず`strength`を即座に返す設計。「Noneならstrengthをそのまま返す」という分岐自体はあるが、分類ロジックへは進まない）。
- 生成結果（同一シードでビット同一）。

**唯一の例外（許容する差分）**:

- `outputs/{job_id}/metadata.json`の`request`ブロックは`req.model_dump()`を素通しで書き出すため、LoRAを使うジョブでは`"audio_strength": null`が必ず出現する。
- `two_stage_hq`の`pipeline`フィールドと同じ既存前例で、意図的な差分（`api/models.py`にコメントで明記）。
- 既存テスト`tests/test_chain_lora.py`の辞書完全一致アサーションはこの`null`キーを織り込んで期待値を更新済み。

## 7. 実装ファイル一覧

WP1〜WP6（バックエンドエンジン中核・エンジン配線・サービス層・API/MCP・Gradio UI・AviUtl2フロントエンド）で変更したファイルは以下（`git status`で確認可能。バックエンドリポジトリの一覧）。

- `api/models.py`（`LoraSpec.audio_strength`フィールド追加）
- `engine/gguf/ic_lora_common.py`（`IcLoraEntry`・`classify_lora_axis`・`strength_for_prefix`・`normalize_ic_loras`新設）
- `engine/gguf/loader_service.py`（bf16融合経路の配線）
- `engine/pipeline/chain_pipeline.py`（型注釈の更新）
- `engine/pipeline/fast_video_pipeline.py`（型注釈の更新）
- `engine/worker.py`（ワーカーメッセージへの`audio_strength`配線）
- `gradio_ui/handlers.py`（`_LORA_TOKEN_RE`・`parse_prompt_loras`の3引数対応）
- `gradio_ui/i18n.py`（音声クランプ警告の新規i18nキー）
- `gradio_ui/ui.py`（画風LoRAカードの既定挿入トークン変更）
- `mcp_server/params.py`（`LoraArg.audio_strength`）
- `mcp_server/tools/generate.py`（`exclude_none=True`化）
- `services/lora_registry.py`（`resolve()`の`audio_strength`引数・`ResolvedLora`の4要素化）
- `services/ltx_runner.py`（`_lora_payload_entry`ヘルパー・添字＋`getattr`方式での互換維持）
- `services/pipeline_manager.py`（`_loras_for_log`・metadata出力の配線）
- `tests/test_chain_lora.py`・`tests/test_gradio_lora_parse.py`・`tests/test_gradio_style_tab.py`・`tests/test_ic_lora_api.py`・`tests/test_ic_lora_forward.py`・`tests/test_lora_registry.py`・`tests/test_ltx_runner_payload.py`・`tests/test_mcp_tools_generate.py`（新規・更新テスト）

フロントエンドリポジトリ（`Nz-LTX23-frontend-AviUtl2`）側は`webui/src/api/types.ts`・`webui/src/lora/loraTags.ts`・`webui/src/lora/LoraChips.tsx`・`webui/src/lora/lora.css`・`webui/src/i18n/strings.ts`と各テストファイル（詳細はフロントエンドDocsの記録を参照）。

## 8. 参照

- KJNodes `nodes/ltxv_nodes.py`の`LTX2LoraLoaderAdvanced`（4軸＋その他の5スライダー露出、4軸化を見送った際の比較対象）。
- フロントエンド台帳[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §4-24（4軸化「Style LoRA Advanced mode」のスコープ外起票）。
- [`IC_LORA_PHASE_B_STATUS.md`](IC_LORA_PHASE_B_STATUS.md)（LoRA適用機構そのものの完成物）。
