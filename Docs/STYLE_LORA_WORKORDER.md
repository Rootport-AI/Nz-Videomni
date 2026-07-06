# 画風/キャラクター LoRA 対応 — 次セッションワークオーダー（起票 2026-07-06・ユーザー要件確定）

> **本書は着手前のワークオーダー。** 次セッションはこの入口から開始する。設計を先取りせず「リサーチ→スパイク→設計→ユーザー合意→スライス実装→回帰ゲート→目視ゲート」の型（本プロジェクト標準）で解く。**特に D（キー互換）は実物検証なしに約束しない**。

- 作成: 2026-07-06（ユーザー要件を確定・**設計/実装は次セッション**）
- 併読: [`IC_LORA_PHASE_B_STATUS.md`](IC_LORA_PHASE_B_STATUS.md)（LoRA 適用機構の完成物・不可触）／[`MODEL_MANAGEMENT_DESIGN.md`](MODEL_MANAGEMENT_DESIGN.md)（「一覧を返す API＋選択適用」の型）／[`VERIFICATION_LOG.md` §28](VERIFICATION_LOG.md)（strength 可変化）・[§23](VERIFICATION_LOG.md)（GUI モック合意の型）
- 正本ポインタ: 機構＝`IC_LORA_PHASE_B_STATUS.md`／strength＝VERIFICATION_LOG §28／モデル管理の型＝`MODEL_MANAGEMENT_DESIGN.md`／GUI の型＝VERIFICATION_LOG §23

---

## 1. ユーザー要件（2026-07-06 確定・3点）

1. **ディレクトリ配置＋リロードで使える**: Forge Neo のように、特定ディレクトリに LoRA ファイル（safetensors）を置き、GUI の「LoRA リロード」ボタン（またはサーバー再起動）だけで使えるようにする。
2. **プロンプト内コマンドで適用＋weight 指定**: Forge Neo のように、プロンプトに簡単なコマンド（例: `<lora:名前:0.8>` 形式）を書くだけで LoRA 適用と weight 指定ができる。
3. **GUI「Style LoRA」タブ**: 登録済み画風/キャラ LoRA の一覧を**サムネイル付き**で表示。サムネイルをクリックすると上記コマンドがプロンプトに自動入力される。UI/UX は Forge Neo を参考。

**ユーザーの問い**＝「中身の配線を変えず、表面的な軽い実装だけでできるか？」（この問いへの正直な見立ては §3-G。）

## 2. 現状の確定事実（2026-07-06 セッションで検証済み・出発点）

- **LoRA 適用機構は完成済み・汎用**: forward 時 GPU LoRA 適用（per-layer-quant 経路・`engine/gguf/ic_lora_common.py`＋`quant_service.py`）、weight は API `LoraSpec.strength`（0〜2.0）で全層可変（§21/§28）。**この機構は不可触**（Phase B の注意を継承）。
- **現状ブロッカー3点**（コード確認済み）:
  1. `api/models.py` の loras⇔reference_video_id **all-or-nothing 検証**＝参照動画なしの LoRA 単独指定は 422。
  2. LoRA は `config.yaml` の `model.ic_loras` の**レジストリ登録制**（`services/lora_registry.py` が論理名を解決）＝任意ファイルの投入経路が無い。
  3. **キー命名の互換未検証**＝我々の attach は Union-Control/upscaler のキー集合でのみ実証。CivitAI 配布（ComfyUI 命名）は要キーマップ。上流 `ltx_core` に `LTXV_LORA_COMFY_RENAMING_MAP` が存在＝先行事例。
- LoRA ファイルは safetensors のまま扱う設計（GGUF 化不要）。ファインチューン**本体**（重み丸ごとの差し替え）は別問題（16GB では GGUF 化必須）＝**本ワークオーダーのスコープ外**。
- GUI は凍結 API の薄いクライアント（`gradio_ui/` 9モジュール・i18n EN/JA）。アダプタドロップダウンを `/config` の `model.ic_loras` から動的構築する先行事例あり（`adapters.py`）。
- モデル管理の先行事例: `GET /models`＋`POST /pipeline/load`（§26・正本＝`MODEL_MANAGEMENT_DESIGN.md`）＝「一覧を返す API＋選択適用」の型。リロード系 API の参考になる。

## 3. 設計上の論点（次セッションで リサーチ→設計→ユーザー合意 の型で解く・結論を先取りしない）

- **A. プロンプト内コマンドの解釈層**: 凍結 API 的に最も軽いのは **GUI 側でパース**して既存 `loras:[{name,strength}]` へ変換（API 無変更・プロンプトからコマンドを除去して送信）。API 側パースにする場合は加算的変更の定型ゲートが要る。Forge Neo/A1111 の `<lora:name:weight>` 記法の互換性をどこまで持つか。
- **B. ディレクトリスキャン＋リロード**: 新ディレクトリ（例: `models/loras/`）のスキャンを誰がやるか（サーバー起動時＋リロード API? 例: `GET /loras` 一覧＋`POST /loras/reload`＝加算的新設）。`config.yaml` レジストリとの関係（登録制と自動スキャンの共存・名前衝突時の規則）。
- **C. スタイル LoRA と制御 LoRA の区別**: all-or-nothing 検証（ブロッカー①）の緩和方法（参照動画不要の LoRA を型として区別するか・preprocess=none との関係・既存 IC-LoRA 経路の byte 同一維持が定型ゲート）。
- **D. キー互換スパイク（最初に置く・GO/NO-GO ゲート）**: 実物の CivitAI LTX-2.3 LoRA（例: ユーザー提示の Pixar CGI Toon Style / Transformation）を1本入手し、キー集合が我々の attach 機構で解決できるか・`LTXV_LORA_COMFY_RENAMING_MAP` 相当が必要かを最初に検証（S0 スパイク相当）。distilled 8step/CFG1.0 経路での効きも要確認（CivitAI 側の推奨設定・トリガーワードの扱いを含む）。
- **E. VRAM**: LoRA バッファの常駐コスト（rank×次元×層数）と複数 LoRA 同時適用時の上限。peak_vram_mb 計測が定型ゲート。
- **F. サムネイル**: A1111/Forge 系の慣習（`<name>.safetensors` と同名の `.png`/`.preview.png` を並置）を踏襲するか。CivitAI のプレビュー画像を手動配置する運用で始めるのが最軽量。Gradio は `gr.Gallery` の select イベント→プロンプト Textbox 更新で「クリック→コマンド自動入力」が素直に組める。
- **G. ユーザーの問い「表面的な軽い実装で足りるか」への現時点の見立て（正直に）**: **GUI タブ自体（ギャラリー＋クリック挿入）とプロンプトパースは軽い**（GUI 内で完結可能）。ただし「LoRA が実際に効く」ためのバックエンド側（B/C/D）は軽くない実装が必要＝**特に D のキー互換は実物検証なしに約束できない**。次セッションは **D スパイクを最初に置くこと**。

## 4. 進め方の型（本プロジェクト標準）

リサーチ（Web＋コード）→ **D スパイク（GO/NO-GO）** → 設計 → ユーザー合意（GUI はモック合意の型・§23 前例）→ スライス実装 → 回帰ゲート（既存 SHA byte 一致・pytest・省略時 byte 同一・peak_vram_mb）→ 目視ゲート（720p 級 1280×768・映画トレイラー風・賑やかな題材）。

## 5. スコープ外（本ワークオーダーでやらない）

- ファインチューン本体（safetensors→GGUF 化）対応。
- IC-LoRA 重ね掛け（多重制御・研究課題・別メモ＝`IC_LORA_PHASE_C_STATUS.md` 末尾）。
- negative/CFG 露出（worker 未配線・継続）。

## 6. 参照

- ユーザー提示の実物例（本体対象・スタイル LoRA）:
  - https://civitai.red/models/2536130/ltx-23-pixar-cgi-toon-style
  - https://civitai.red/models/2487612/ltx-23-transformation
- スコープ外の本体例（ファインチューン本体・参考）: https://civitai.red/models/2601098/sulphur-2-base
- 正本ポインタ: 機構＝`IC_LORA_PHASE_B_STATUS.md`／strength＝VERIFICATION_LOG §28／モデル管理の型＝`MODEL_MANAGEMENT_DESIGN.md`／GUI の型＝VERIFICATION_LOG §23
