# プロンプト入力欄の一本化 ＋ ネガティブ欄グレーアウト — 次セッションワークオーダー（起票 2026-07-06・設計ユーザー合意済み）

> **本書は着手前のワークオーダー。** 次セッションはこの入口から開始する。**設計は本セッション（2026-07-06）でユーザーと合意済み**——結論を先取りするのではなく、ここに書かれた確定事項どおりに「スライス実装→回帰ゲート（pytest・省略時 payload byte 同一）→GUI 目視/実機確認（ユーザー）」の型で進める。GUI のみの改修（API・engine・services は不可触）。

- 作成: 2026-07-06（設計合意・**実装は次セッション**）
- 併読: [`VERIFICATION_LOG.md` §30](VERIFICATION_LOG.md)（画風/キャラ LoRA S2 GUI＝プロンプト内 `<lora:...>` パースの実装元）／[`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md) 冒頭

---

## 1. 目的と背景

現状、メインのプロンプト入力欄が **2 箇所**に分かれている:

- **Generate タブ**の `prompt`（`gradio_ui/ui.py` L180 付近）＝単発 T2V/I2V 生成用。
- **Clip Chain タブ**の `chain_prompt`（同 L361 付近・ラベル「Prompt (shared)」）＝クリップ連結の**共通ベースプロンプト**（各クリップ枠の個別プロンプトがこれを上書きする。空クリップはこの共通プロンプトを流用）。

ユーザー要望＝この 2 つを **1 つの「下書き」プロンプト欄に統合**し、`gr.Tabs()` の外＝**タブ群の上**（上部共通バーの下）に常時表示する。どのタブを見ていても同じ欄が見え、Style LoRA タブでサムネイルを選ぶと上部の下書き欄にタグが貼られる、という **Forge Neo 風 UX** を実現する。

## 2. 確定した設計（2026-07-06 ユーザー合意）

**方式＝「本物の欄を隠して残し、生成時にコピーする」ではなく、「下書き欄を唯一の本物にして両ハンドラへ直結する」。** 直結にすることで、コピーの実行順序に依存する事故や、隠し欄への残留プロンプトといった懸念が**構造的に発生しない**（Gradio のハンドラは inputs に並べたコンポーネントの値をクリック時に atomically に受け取るため）。

1. **単一の下書きプロンプト欄を新設**（3 行）。`with gr.Blocks` 内の上部共通バー（status／refresh／load／unload の `gr.Row`・L164 付近）の**直後**、`with gr.Tabs():` の**直前**に配置する。全タブで常時表示される。
2. **旧 `prompt`（Generate）と旧 `chain_prompt`（Clip Chain）は削除**する（`visible=False` で隠して残すのではなく、下書き欄がこの 2 欄を置き換える）。
3. **両生成ボタンに直結**: `generate_btn.click` と `chain_generate_btn.click` の `inputs` 先頭を、両方ともこの下書き欄にする。ハンドラは第 1 引数に文字列を受け取るだけなので、`gradio_ui/handlers.py` の `make_generate_handler` / `make_chain_handler` の**シグネチャ変更は不要**（要確認のうえ、不要であることを担保）。
4. **LoRA サムネイル選択の貼付け先を下書き欄へ**: `style_gallery.select`（`gradio_ui/ui.py` L918-920 付近・現状 `inputs=[prompt,…], outputs=prompt`）を**下書き欄**に向け直す。これで LoRA 選択→上部の下書き欄にタグ追記、が全タブから見える。**バックエンド改修は不要**（純フロントエンドの Gradio イベント）。
5. **連結ハンドラでの LoRA タグ除去（＋警告）**: `make_chain_handler`（`gradio_ui/handlers.py` L429〜）は現状 `<lora:...>` を素通しする。ここで、受け取った共通プロンプトから `<lora:...>` トークンを**除去してからエンドポイントへ送る**（適用はしない）。除去が発生したら `gr.Warning`（例「クリップ連結では LoRA は未対応のためタグを無視しました」）を出す。除去には既存の `_LORA_TOKEN_RE`（`handlers.py` L136）を再利用する。
   - **スコープ外**: 各クリップ**個別**プロンプト欄（`c1_prompt`／`cN_prompt`）に手書きされた LoRA タグは除去しない（ユーザー決定・稀なケースにつき無視）。除去対象は共通プロンプトのみ。
6. **ネガティブプロンプト欄のグレーアウト**: `negative`（Generate・L182 付近）と `chain_negative`（Clip Chain・L364 付近）を **`interactive=False`（グレーアウト）** にする。現状 CFG=1 固定＝蒸留モデル前提でネガティブは常に無効なため、**常時**グレーアウトでよい。ただし**欄自体は削除せず、将来の非蒸留モデル対応時のモックとして残す**（ユーザー決定）。
   - **ペイロード不変の担保**: `interactive=False` にするだけで、既定値・送信ペイロード（`"negative_prompt": …`）は現状のまま変えない＝**回帰 SHA 不変**。値やキーの送出条件を触らないこと。
   - 可能なら info 注記「蒸留モデル（CFG=1）ではネガティブプロンプトは無効です」を EN/JA で添える（任意）。

## 3. i18n / registry の扱い

- 下書き欄のラベル／プレースホルダを **EN/JA 両方**に用意（既存 `lbl_prompt`／`ph_prompt` を流用するか、新規 `lbl_prompt_draft` 等を起こすかは実装判断。文言は「単発生成にもクリップ連結の共通ベースにも使う」ことが伝わるものにする）。
- `chain_prompt` 削除に伴い `lbl_prompt_shared` / `ph_prompt2`（EN/JA・計 4 箇所）が未使用になる → **grep で他参照が無いことを確認してから対で削除**（片側だけ消すと言語切替のキー不整合になる）。
- `reg()` 登録・`switch_language` は、削除分が自然に消え、新設の下書き欄が 1 つ登録される形に更新。EN/JA はロックステップで。

## 4. 回帰・検証ゲート

- **pytest**（app venv・現行基準 **418 passed + 1 skipped**）赤化ゼロ。GUI ユニットテスト（mock transport）を更新／追加:
  - 下書き欄 → Generate 生成（既存 generate 経路が下書き欄の値を第 1 引数で受ける）。
  - 下書き欄 → Clip Chain 生成で `<lora:...>` が除去され、警告が出て、除去後プロンプトが送られる。
  - Style 選択 → 下書き欄にトークン追記。
  - **トークン無しプロンプトの payload byte 同一**（S2 で担保した性質を維持: `<lora:…>` を含まない場合、送信 payload が従来と完全一致）。
- **GUI 目視/実機確認（ユーザー）**: ①上部に下書き欄が 1 つ・全タブで見える ②Style LoRA タブでサムネイル選択→上部欄にタグ ③連結で LoRA タグ除去の警告 ④ネガティブ欄がグレーアウト。

## 5. スコープ外・将来の研究課題

- **連結タブで LoRA を実際に効かせるバックエンド改修**＝**将来の研究課題**。実体＝`GenerateChainRequest` に `loras` フィールドを加算（凍結 API の加算的変更の型）＋ worker 配線（全体共通適用か、クリップ毎適用かの設計判断を含む）。今回は「除去＋警告」でフロントのみに留める。
- 各クリップ個別プロンプト欄の LoRA タグ除去（無視）。
- ネガティブ／CFG の worker 配線＆API 露出（別課題・非蒸留 dev 対応時）。

## 6. 触るファイル

- `gradio_ui/ui.py`（下書き欄の新設・旧 2 欄の削除・イベント配線変更・ネガティブ欄 `interactive=False`）
- `gradio_ui/handlers.py`（`make_chain_handler` に LoRA トークン除去＋警告）
- `gradio_ui/i18n.py`（下書き欄ラベルの EN/JA 追加・`lbl_prompt_shared`/`ph_prompt2` の対削除）
- `tests/test_gradio_*.py`（更新／追加）
- **不可触**: API・engine・services（今回は GUI のみ）／`_LORA_TOKEN_RE`・`parse_prompt_loras` のパース仕様（再利用するが変更しない）

## 7. 参照

- 設計合意のログ＝**本ワークオーダー**（2026-07-06 セッション）。
- LoRA パース既存実装＝`gradio_ui/handlers.py` L127-214（`parse_prompt_loras` / `_LORA_TOKEN_RE`）。
- Style LoRA タブ＝`gradio_ui/ui.py` L582-601 ＋ `on_style_select`（L899-913）。
- 現行プロンプト欄＝Generate `prompt`（L180 付近）／ Clip Chain `chain_prompt`（L361 付近）。
- 連結ハンドラ＝`gradio_ui/handlers.py` `make_chain_handler`（L429〜・第 1 引数が共通プロンプト）。
