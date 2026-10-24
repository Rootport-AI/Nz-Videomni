# トークンカウンター（台帳 §1-111）の研究ノート

- 作成: 2026-10-23／最終更新: 2026-10-24
- 位置づけ: 台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §1-111（トークンカウンター）と §1-112（2,000 文字制限の撤廃）の判断材料の正本だった（§1-111 は [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) 3-249 でクローズ済み。§1-112 の判断材料としては引き続き使う）。第 1 節以下はオーナーの事前調査レポート（2026-10-23）をそのまま収めたもので、第 4 節のアーキテクチャと第 6 節の検証事項は未確定の推奨案である。
- 末尾の「補足: 事実確認の結果」は、レポートをコードと照合した結果（2026-10-23）。レポートの事実に誤りは無かった。
- **更新（2026-10-24）**: 第 4 節と第 6 節の未確定の事項は、末尾の「G0 の実測」と「決まったことと実装」で決着した。台帳 §1-111 は [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) 3-249 でクローズした。§1-112（2,000 文字制限の撤廃）は台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) に残る。
- **更新（2026-10-24、§1-112 のクローズ）**: §1-112（2,000 文字制限の撤廃）も [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) 3-250 でクローズした（記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §175）。本書の中で 2,000 文字の制限を「現行」「残る」と述べる箇所（第 3 節・第 7 節など）と、上の「引き続き使う」「台帳に残る」は、すべてクローズ前の状態である。

## Nz-Videomni §1-111 トークンカウンター機能・事前調査レポート

対象：`Docs/PENDING_TASKS.md` §1-111  
リポジトリ：https://github.com/Rootport-AI/Nz-Videomni/tree/dev

### 1. 目的

A1111 SD WebUI／Forge／Forge neoのプロンプト欄を参考に、Nz-Videomniの操作パネルに軽量なトークンカウンターを追加する。

現在選択されているベースモデルの**実際のトークナイザーに基づくトークン数**を表示し、モデルの上限を超過した場合に視覚的な注意を促す。

現在の対象はLTX 2.3、LTX 2.5、MiniMax H3。今後のベースモデル追加にも容易に対応できる構造とする。

### 2. UI/UXの確定方針

- プロンプト入力欄の右上に、小さく `876 / 1024` のような形式で表示する。
- 上限超過時はカウンターの文字色と枠を赤くする。
- 説明文、警告ダイアログ、Generateボタンのロックは不要。
- `[]`や`()`の対応関係の検査も不要。
- 上限超過そのものを422エラーの条件にはしない。LTXが超過部分を切り捨てて生成できるなら、その動作を許容する。
- カウンターの計数失敗も生成を妨げない。
- **入力時の軽快さを最優先**する。Forge neoと同程度の操作感を目標とする。

Chainedでは、共通プロンプトに加えて各クリップの個別プロンプトがある。各クリップの表示には**共通＋個別を合成したStage-1の実トークン数**を使用する。Stage-2は共通プロンプトのみを使用する。

上限が未確認のモデルでは、確証のない上限値を表示せず、トークン数だけを表示する。

### 3. 現行コードで確認できた事実

| ベースモデル | トークナイザー | 資産の所在 |
|---|---|---|
| LTX 2.3 | Gemma 3 | `LTX23/TextEncoder/tokenizer/` |
| LTX 2.5 | Gemma 4 | テキストエンコーダーGGUF内に同梱 |
| MiniMax H3 | Qwen3-VL | `MiniMaxH3/TextEncoder/qwen3vl_tokenizer/` |

- LTX 2.3：`10-ltx23.json` の `assets.gemma_root` に登録済み。
- LTX 2.5：`engine25/assets_export.py` がGGUFからトークナイザー等の資産を抽出し、`*.assets.safetensors`を生成する仕組みを持つ。
- MiniMax H3：`40-minimaxh3.json` の `assets.tokenizer_dir` に登録済み。
- 現在のAPIサーバーと3系統の推論エンジンは、それぞれ独立したPython仮想環境を使用している。
- 現行のAPIには、共通・個別の2,000文字制限およびChainedの合成後2,000文字制限が存在する。

モデル本体の巨大な重みをロードすることなく、トークナイザー資産だけで計数することは原理的に可能。

#### モデル固有の前処理

- LTX系：GemmaのBOSトークン処理を再現する。公式実装の最大長は1,024トークン。計数時は切り捨て・パディング前の実トークン数を取得する。
- H3：`add_special_tokens=False`を使用。`tokenizer_config.json`の追加特殊トークン7個を正しく扱う必要がある。
- H3のI2Vでは画像由来のトークンも追加される。初期実装では文章部分の計数を対象とし、画像トークンとの関係および表示すべき上限は別途確認する。

### 4. 推奨アーキテクチャ（設計案・未確定）

**APIサーバー内に共通のトークン計数サービスを設け、モデル固有処理を交換可能にする構成**を第一候補とする。

1. AviUtl2のWebView2からローカルAPIへ非同期に計数要求を送る。
2. APIサーバーは選択中のベースモデルとテキストエンコーダーの構成を取得する。
3. モデル別のトークナイザープロバイダーが実トークン数を返す。
4. UIは数値と超過時の赤色表示だけを更新する。

候補ライブラリはRust実装の高速なHugging Face `tokenizers`。PyTorchや巨大なモデル重みへの依存は避ける。

トークナイザーの種類、資産の所在、前処理プロファイル、上限値は、できるだけベースモデルの記述子（manifest）と関連付ける。UI側にモデル名ごとの分岐を増やさない。

将来のモデル追加では、既存方式なら記述子の登録、未対応方式なら小さなプロバイダーの追加で済む構成を目指す。

なお、LTX 2.5の既存`assets_export.py`はPyTorch等に依存するため、APIサーバーからそのまま利用できるとは限らない。資産の事前抽出・キャッシュ・専用の軽量抽出処理を比較して決定する。

### 5. 性能要件と実装上の注意

- WebView2のUIスレッドで重い計数処理を実行しない。
- 入力停止後に計数するデバウンス方式を第一候補とする。A1111には800msの実装例があるが、Nz-Videomniの値は実測で決める。
- トークナイザーはキャッシュして再利用する。
- Chainedでは複数クリップを一括計数し、個別のHTTP要求を乱発しない。
- ベースモデル・テキストエンコーダーを切り替えた場合は、正しいトークナイザーへ切り替える。
- 古い非同期応答で新しいカウンターを上書きしない。
- 計数に失敗した場合でも、既存の動画生成機能は正常に使用できる。

### 6. 実装前の検証事項

1. 3モデルの実際の推論経路と、軽量計数処理が生成するトークンID列が一致するかを確認する。特にGemmaのBOSおよびH3の追加特殊トークン。
2. LTX 2.3／2.5の実際の上限と切り捨て条件を、Nz-Videomniが固定している依存バージョンで再確認する。
3. H3の実用上のテキストトークン上限を調査する。1,024を一律に適用しない。
4. 各トークナイザーの初回ロード時間、常駐RAM、計数遅延を実測する。
5. WebView2から非同期計数した場合の入力レスポンスを検証する。軽快さを満たせなければ、ブラウザ側計数・Web Worker・概算方式も比較対象とする。

正確な計数を第一候補とするが、実測前に方式を確定しない。

### 7. §1-111に含めない事項

**既存の2,000文字制限の撤廃は別タスクとして§1に起票する。**（→ 台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §1-112 として起票済み）

この別タスクでは、API・AviUtl2・Gradio・MCP等に残る文字数制限の整合性を確認する。ただし、代わりに1,024トークンで422を返す制限を設けるものではない。

§1-111は、あくまでトークン数の計算と表示、および超過時の色変更を対象とする。

### 8. 主要情報源

- 現行タスク：https://github.com/Rootport-AI/Nz-Videomni/blob/dev/Docs/PENDING_TASKS.md
- LTX 2.3記述子：https://github.com/Rootport-AI/Nz-Videomni/blob/dev/scripts/manifests/10-ltx23.json
- LTX 2.5記述子：https://github.com/Rootport-AI/Nz-Videomni/blob/dev/scripts/manifests/20-ltx25.json
- H3記述子：https://github.com/Rootport-AI/Nz-Videomni/blob/dev/scripts/manifests/40-minimaxh3.json
- LTX 2.5資産抽出：https://github.com/Rootport-AI/Nz-Videomni/blob/dev/engine25/assets_export.py
- H3の実装：https://github.com/Rootport-AI/Nz-Videomni/blob/dev/engine_h3/text_encoder.py
- APIの文字数制限：https://github.com/Rootport-AI/Nz-Videomni/blob/dev/api/models.py
- 共通プロンプトUI：https://github.com/Rootport-AI/Nz-Videomni/blob/dev/AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/webui/src/shell/PromptBar.tsx
- LTX公式Gemmaトークナイザー：https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/text_encoders/gemma/tokenizer.py
- LTX公式最大トークン数：https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/text_encoders/gemma/gemma_assets.py
- A1111のカウンター更新処理：https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/javascript/token-counters.js

## 補足: 事実確認の結果（2026-10-23）

レポート第 3 節の事実は、作業ツリー（dev）・`models/` 配下の実物・各仮想環境に入っている依存のソースと照合して、すべて一致した。以下は照合で分かった追加の事実である。

### 最大長と切り捨て（固定している依存のソース）

- LTX 2.3（`.venv-engine` の ltx-core、commit `00dc53d`）: 最大長 1,024 は `encoders/base_encoder.py` の `LTXVGemmaTokenizer(tokenizer_root, 1024)` にある。`tokenizer.py` は `text.strip()` のあと `padding="max_length"`・`truncation=True` で、BOS は Gemma 3 のトークナイザー自身が付ける（`tokenizer_config.json` の `add_bos_token: true`）。
- LTX 2.5（`.venv-engine-ltx25` の ltx-core、commit `d151147`）: `gemma_assets.py` の `TOKENIZER_MAX_LENGTH = 1024`。`tokenizer.py` は `strip()` のあと `truncation=True` で、BOS が無ければ先頭に足してから 1,024 で切る（Gemma 4 は BOS を付けないため）。
- レポート第 8 節の Lightricks の URL は main ブランチを指し、内容は LTX 2.5 の固定版に近い。LTX 2.3 の固定版は別の実装（上記）である。
- Nz-Videomni 側でこの長さを上書きしている箇所は無い。切り捨て時に警告が出るかは確かめていない。
- MiniMax H3: `engine_h3` に文章の長さの上限や切り捨ては見当たらない（レポートの「上限は未確認」と一致）。

### トークナイザーの資産

- LTX 2.5 の記述子 `scripts/manifests/20-ltx25.json` には、トークナイザーの登録欄が無い（GGUF への同梱は `_note` にだけ書かれている）。
- `engine25/assets_export.py` のうち、GGUF からの読み出し（`read_payload`）は `gguf` と numpy だけで動き、torch を使わない。torch に依存するのは書き込み（`_write`）と読み返し（`_read_existing`）である。
- MiniMax H3 の `tokenizer_config.json` の `additional_special_tokens` は 20 個あり、そのうち `tokenizer.json` に無いのが `<d>`・`</d>`・`<|cutoff|>`・`<|lyrics_start|>`・`<|lyrics_end|>`・`<|caption_start|>`・`<|caption_end|>` の 7 個（`engine_h3/text_encoder.py` の `EXTRA_SPECIAL_TOKENS` と一致）。`tokenizer.json` だけを読むと、この 7 個は 1 個のトークンにならない。

### 仮想環境と依存

- API サーバーの `.venv` には `tokenizers`・`transformers`・`gguf`・`safetensors`・`torch` のどれも入っていない。API サーバーで数える案では依存の追加が要る。
- 3 つのエンジンの仮想環境（`.venv-engine`・`.venv-engine-ltx25`・`.venv-engine-h3`）には、どれも `tokenizers==0.22.2` が入っている。

### 操作パネルの現状

- 共通プロンプト欄（`webui/src/shell/PromptBar.tsx`）には、`maxLength={2000}` と `{value.length}/2000` の文字数表示が既にある。トークンカウンターと置き換えるか並べるかは判断が要る。

### 2,000 文字制限が残っている場所（台帳 §1-112 の着手材料）

- API（`api/models.py`）: `GenerateRequest.prompt`・`GenerateRequest.negative_prompt`・`ChainClip.prompt`・`GenerateChainRequest.prompt`・`GenerateChainRequest.negative_prompt` の `max_length=2000` と、`GenerateChainRequest.validate_chain_constraints` の合成後の検査。正本は仕様書 §6.2。
- 操作パネル（`webui/src/`）: `shell/PromptBar.tsx`、`shell/NagAccordion.tsx`（ネガティブプロンプト）、`modes/single/paramUtils.ts` の `isValidPrompt` とその利用先（`PromptBar.tsx`・`modes/chained/useChainForm.ts`・`modes/single/useGenerationForm.ts`）、`modes/batch-i2v-long/useBatchI2vLongForm.ts` の `MAX_PROMPT_LENGTH` と `BatchI2vLongTable.tsx`、`i18n/strings.ts` の文言、`bridge/mockBridge.ts`、`api/types.ts` のコメント、関連テスト。
- Gradio: 独自の制限は無い（API の 422 に任せている）。
- MCP: 入力の検査は無い。説明文に 2,000 文字の記載がある（`mcp_server/params.py`・`mcp_server/tools/generate.py`）。
- 文書: README、仕様書 §6.2、フロントエンドの `Docs/API_REFERENCE.md` と `Docs/BATCH_I2V_WORKORDER.md`。

## オーナーの裁定（2026-10-23）

- 共通プロンプト欄の文字数表示（`n/2000`）は撤廃し、トークンカウンターに置き換える。
- ネガティブプロンプトも同様に扱う。ネガティブプロンプト欄にもトークンカウンターを付け、その 2,000 文字の上限も台帳 §1-112 で撤廃する。
- ベースモデルごとのトークナイザーの情報は、エンジンのアダプタ（`services/engines/<エンジン>/adapter.py`）に置く。
  - アダプタに置くもの: トークナイザーの種類・前処理（Gemma の BOS・H3 の追加特殊トークン）・上限値・資産の在りかの決め方。
  - 資産の在りか: LTX 2.3（`assets.gemma_root`）と MiniMax H3（`assets.tokenizer_dir`）は、いまどおり記述子（`scripts/manifests/*.json`）の `assets` にあり、アダプタが読む（生成のための読み込みと同じ経路）。LTX 2.5 はトークナイザーが選択式の `text_encoder` の GGUF の中にあるので、選択中のファイルから取り出す処理をアダプタが持つ。記述子に新しい欄や書式は足さない。
  - 上限値: アダプタの数値の規則（`REQUEST_RULES`）と同じく `GET /models` の `constraints` で配る。確かめていないモデルは空（null）にしてトークン数だけを出す（H3 に快適上限の線を出さないのと同じ考え方）。
  - フロントの `webui/src/shell/featureScope.ts` は「サーバーの機能名 → 閉じる UI 部品」の翻訳表でモデルの軸を持たないので、トークナイザーの情報は置かない。UI が決めるのはカウンターをどの欄に出すかだけ。
  - 理由: 当初は記述子に置くとしたが、LTX 2.5 の扱いを考えてアダプタに改めた。トークナイザーはベースモデルではなくエンジンに属する知識であり（`Docs/MULTI_ENGINE_DESIGN.md` §4.4）、記述子に集約すると、LTX 2.5 の「選択中のファイルから取る」規則を表す書式と解釈のコード、記述子とエンジンのコードの一致を縛るテストが余分に要るため。
  - どちらの案でも要るもの: 軽量な計数がエンジンの実際の処理と同じトークン列を返すかの確認（第 6 節の 1）。

## G0 の実測（2026-10-24）

実装の前に、3 つのエンジンの仮想環境（どれも `tokenizers` 0.22.2 入り）で、CPU だけ・重みを読まずに測った。スクリプトはセッションの作業用フォルダに置き、残していない。

### E1: 推論経路とトークン列が一致するか

- **3 系統 × 20 入力のすべてで、推論経路のトークナイザーと「`tokenizers` 単体＋下の規則」のトークン ID の列が一致した**。入力は英語・日本語（NFC と NFD）・絵文字・CRLF・前後の空白・特殊トークンの文字列・`<d>…</d>`・`<bos>` で始まる文・1,023／1,024／1,025 トークン前後・1 万文字など。
- LTX 2.3 と 2.5 は、切り捨てのあとに開始トークン 1 個と利用者のトークン 1,023 個が残る。上限 1,024 は開始トークン込みの数である。
- 空の文字列は、LTX が 1（開始トークンだけ）、MiniMax H3 が 0。
- MiniMax H3 の追加の特殊トークン 7 個の ID は 151669〜151675 で、`tokenizer.json` の語彙の後ろに、宣言の順に並ぶ。
- LTX 2.5 のテキストエンコーダーの GGUF から取り出した `tokenizer.json` は、`*.assets.safetensors` にあるものとバイト単位で同じだった。
- エンジンが読む `tokenizer_config.json` を加えても、ID は変わらない。

| 系統 | 資産 | 数える文字列 | 数え方 |
|---|---|---|---|
| LTX 2.3 | 記述子の `assets.gemma_root` の `tokenizer.json` | 前後の空白を除く | 特殊トークンを付けて（`add_special_tokens=True`）数えた長さ。開始トークン込み |
| LTX 2.5 | 選択中のテキストエンコーダーの GGUF のテンソル `tokenizer_json` | 前後の空白を除く | 特殊トークンを付けずに数えた長さ＋1（先頭がすでに `<bos>` なら＋0） |
| MiniMax H3 | 記述子の `assets.tokenizer_dir` の `tokenizer.json`＋追加の特殊トークン 7 個 | そのまま | 特殊トークンを付けずに（`add_special_tokens=False`）数えた長さ |

### E2: 読み込みの時間・常駐メモリ・数える時間

数値は、新しいプロセスを 5 回起こした中央値。

| 系統 | 初回の読み込み | 増えた常駐メモリ | 2,000 文字 | 1 万文字 | 25 本をまとめて |
|---|---|---|---|---|---|
| LTX 2.3 | 612 ms | 約 71 MB | 0.31 ms | 1.57 ms | 1.71 ms |
| LTX 2.5 | 650 ms（GGUF の読み出しはキャッシュが温まった状態で 8 ms、文字列からの組み立てが 631 ms） | 約 68 MB | 0.35 ms | 1.99 ms | 1.77 ms |
| MiniMax H3 | 165 ms | 約 31 MB | 0.60 ms | 3.70 ms | 3.51 ms |

- 1 つのトークナイザーを 4 つのスレッドから同時に使っても問題は出なかった。
- 計画で決めた中止の目安（初回 2 秒超・クリップ 24 本で数十 ms 超・常駐で数百 MB 超）には、どれも届かなかった。数える時間が数 ms なので、入力が止まってから送るまでの待ち（デバウンス）は使い心地で決めた（250 ms）。

## 決まったことと実装（2026-10-24）

第 4 節の推奨案と第 6 節の検証事項は、上の実測と次の実装で決着した。実装の記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §174、窓口の契約は [`Videomni_Backend_Specification.md`](../Videomni_Backend_Specification.md) §6.1 の補足と §6.2b (e)、アダプタの仕組みは [`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §5.2・§7 を正本とする。

- **方式**: API サーバーが Hugging Face の `tokenizers` で数える（第 4 節の第一候補）。アプリの仮想環境（`.venv`）に `tokenizers==0.22.2` を足した。torch も重みも読まない。ブラウザ側で数える案・Web Worker・概算は要らなかった。
- **第 6 節の 1（トークン列の一致）**: E1 のとおり全件一致。
- **第 6 節の 2（LTX の上限と切り捨て）**: 1,024（開始トークン込み）。超えた分は黙って切り捨てられる。この値を `GET /models` の `constraints.prompt_token_limit` で配る。サーバーは検査しない。
- **第 6 節の 3（H3 の上限）**: 確かめていないので `null`。画面はトークン数だけを出し、赤くしない。I2V の画像のトークンは数えない（文章だけ）。
- **第 6 節の 4（読み込み・メモリ・遅延）**: E2 のとおり。トークナイザーはプロセスに 1 つだけ（最後に使ったもの）持つ。
- **第 6 節の 5（入力の軽さ）**: オーナーの目視で合格（2026-10-24）。
- **LTX 2.5 のトークナイザーの取り出し**: `engine25/assets_export.py` は使わず、自前の GGUF 読み取り器（`services/gguf_kv.py`）に、小さなテンソルを 1 本だけ読む関数を足した。選択中のテキストエンコーダーから毎回取り出す（ファイルが変わらなければ、持っているものを使う）。
- **数える文字列**: 共通のプロンプトは `<lora:>` のタグを除いた後（実際に送る文字列）。Chained の各クリップは、サーバーの合成規則（`api/models.py` の `compose_clip_prompt`）で共通と結んだ後。ネガティブプロンプトはそのまま。
- **画面**: 操作パネルの共通プロンプト欄・ネガティブプロンプト欄（全エンジン。MiniMax H3 では欄が使えない状態だが、後の NAG／VSF に備えて出す）・Chained の各クリップに、`876 / 1024 tokens`（上限を確かめていない〔`null`〕モデルは `876 tokens`）を出す。超えたら数字と枠を赤くするだけで、生成は妨げない。`n/2000` の表示は消した。2,000 文字の制限とエラー文は台帳 §1-112 まで残る。Gradio・MCP・Batch の画面には付けていない。
- **補足の「操作パネルの現状」の問い（置き換えるか並べるか）**: オーナーの裁定どおり置き換えた。
