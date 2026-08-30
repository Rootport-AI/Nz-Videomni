# 次セッション引き継ぎ書 — Nz-Videomni

> **過去の引き継ぎ（日付ごとに積み上がっていた「最新ステータス」ブロック）は[`HANDOFF_ARCHIVE.md`](HANDOFF_ARCHIVE.md)へ移した。** 本書には現在有効な引き継ぎだけを置く。

---

## 1. このリポジトリの形

**`Nz-Videomni` はモノレポ（1つのリポジトリに複数の成果物を入れる方式）である。** バックエンドとフロントエンドが同じリポジトリに同居しており、片方だけを別リポジトリで開発することはもう無い。

| 場所 | 中身 |
|------|------|
| リポジトリ直下 | バックエンド（`main.py` / `api/` / `services/` / `engine/`〔LTX 2.3〕/ `engine25/`〔LTX 2.5〕/ `gradio_ui/` / `mcp_server/`）。REST API サーバー本体 |
| `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/` | AviUtl2 拡張フロントエンドのソース（C++ プラグイン `native/` ＋ React/TypeScript の `webui/`） |
| `AviUtl2-Plugin/NzVideomni.aux2` | ビルド済みのプラグイン本体（配布物・git 追跡。利用者はこれを AviUtl2 へドラッグ＆ドロップする） |
| `Docs/` | プロジェクト全体の文書と課題台帳 |
| `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/` | フロントエンド固有の文書（ブリッジ契約・API リファレンス・実装ログほか） |

**旧フロントエンドリポジトリ `Nz-LTX23-frontend-AviUtl2`（GitHub 上の名称は不変）は凍結してある。** ディスク上のローカルフォルダは `_frozen_Nz-LTX23-frontend-AviUtl2` へリネーム済み（旧相対リンクが本機でだけ解決してしまう事故を防ぐため）。以後の変更はすべて本リポジトリで行うこと。

**名前の使い分け**: 製品ブランドは **Nz-Videomni**（識別子は `NzVideomni` / `nz-videomni`）。**LTX 2.3・LTX23 はモデルの名前**であり、こちらは今後も残る（`models/LTX23/`・HuggingFace の `Rootport/Nz-LTX23-weights`・`Nz-GGUF-Converter-LTX23` など）。将来 LTX 2.5 や Wan 2.x といった別のモデルも載せられる基盤を目指しているため、製品名からモデル名を外してある。

---

## 2. 文書の地図（どれを読み、どれを直すか）

文書は3つに分ける。**①生きた文書**（現状を現在形で書く。随時更新する）／**②追記専用の記録簿**（過去の記述は書き換えない。新しい記録を必ず追記する）／**③凍結文書**（一切触らない）。

### ① 生きた文書（現在の仕様・手順を書くもの。読む人は必ずここを見る）

| 文書 | 役割 |
|------|------|
| [`../README.md`](../README.md) | 利用者向けの入口。セットアップ導線・起動・`models/` の構成・API 概要・MCP サーバー |
| [`../Videomni_Backend_Specification.md`](../Videomni_Backend_Specification.md) | バックエンド仕様書。**凍結 API 契約（§6）はこの文書が正本** |
| [`PENDING_TASKS.md`](PENDING_TASKS.md) | **プロジェクト全体の課題台帳。**「次に何をすべきか」はここで確認する |
| [`STORAGE_POLICY.md`](STORAGE_POLICY.md) | `outputs/` / `uploads/` の設計原則（Outputs は宝物・Uploads は一時置き場） |
| [`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) | クリップ連結（Clip Chain）の内部構造と現行アーキテクチャの設計正本 |
| [`RESOLUTION_DURATION_CAPABILITY.md`](RESOLUTION_DURATION_CAPABILITY.md) / [`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) | 解像度×尺の能力（spill-free 閾値・生成時間）と快適上限の各正本 |
| [`ACCELERATION_RESEARCH_NOTES.md`](ACCELERATION_RESEARCH_NOTES.md) / [`LTX23_REFERENCE.md`](LTX23_REFERENCE.md) / [`MCP_SERVER_DESIGN.md`](MCP_SERVER_DESIGN.md) | 高速化候補の整理・LTX 2.3 の一般知識・MCP サーバー設計の各正本 |
| [`VSF_README_NOTES.md`](VSF_README_NOTES.md) | 非CFGネガティブプロンプト（NAG／VSF）の**使い分けの正本**（どちらを選ぶか・つまみの目安・LTX 2.5 での実測）。README が利用者向けに指している先である |
| [`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) | **マルチエンジン化（複数の動画生成AIをドロップダウンで切り替える）の設計正本。第1段階（土台）・第2段階（LTX 2.5）とも実装済み**（起票とクローズ記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-97・§3-98。**生きている後続課題の一覧は[`PENDING_TASKS.md`](PENDING_TASKS.md) §3 が唯一の正本**なので、件数と項目番号はそちらで確認すること——本書を含む他の文書には数を書かない） |
| フロントエンド [`API_REFERENCE.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/API_REFERENCE.md) | API 利用者（フロントエンド実装者）向けの正本 |
| フロントエンド [`BRIDGE_CONTRACT.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/BRIDGE_CONTRACT.md) | ネイティブ ↔ Web UI の JSON-RPC 契約 |
| フロントエンド [`REAL_BACKEND_CHECKLIST.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/REAL_BACKEND_CHECKLIST.md) | 実バックエンド接続時の確認手順 |
| フロントエンド [`Mock/AVIUTL2_DESIGN_BRIEF.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Mock/AVIUTL2_DESIGN_BRIEF.md) | **フロントエンドのデザインブリーフの正本。** バックエンド側の[`AVIUTL2_DESIGN_BRIEF.md`](AVIUTL2_DESIGN_BRIEF.md)は別物の凍結スナップショットであり、これとは区別すること |

### ② 追記専用の記録簿（過去の記述は書き換えない。新しい記録を必ず追記する）

- [`HANDOFF_ARCHIVE.md`](HANDOFF_ARCHIVE.md) — 過去の引き継ぎエントリ
- [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) — 実機検証の全経緯。**各テーマの最新状態は必ずここの該当節（節末尾の「状態」表記）で確認する**
- [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) — クローズ済み項目の完了記録
- フロントエンド `Docs/DEVLOG.md` — フロントエンド側の実装ログ

### ③ 凍結文書（一切触らない。当時の記述のままなので、旧ディレクトリ構成や旧モデル配置が残っている）

- 各 `*_WORKORDER.md` / `*_STATUS.md`
- 完了済み・凍結済みの設計書（[`MODEL_MANAGEMENT_DESIGN.md`](MODEL_MANAGEMENT_DESIGN.md)、バックエンド側の[`AVIUTL2_DESIGN_BRIEF.md`](AVIUTL2_DESIGN_BRIEF.md) など）

### 参考資料（③の一種。当時のまま収蔵する下調べで、設計の正本ではない）

上の3分類のうち **③凍結文書の一種**として扱う。設計の正本と取り違えないよう、ここに個別に挙げておく。

| 文書 | 備考 |
|------|------|
| [`LTX25_RESEARCH_NOTES.md`](LTX25_RESEARCH_NOTES.md) | オーナーによる 2026-08-20 共有の LTX 2.5 事前調査。**マルチエンジン設計の議論より前に行われた調査であり、設計の正本ではない。**設計の正本は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md)（①生きた文書）。本文は書き換えず、失効した箇所には日付つきの追記を足す運用である（実際に本文へ日付つきの追記が入っている）。新しい知見は設計正本側へ書くこと。**末尾に「実装してみた結果との差分」の章が2つある**（「§3-98 v1実装完了」と「§3-102 第3段 実装完了」）——調査時の想定と実装結果の差分を一覧にしたもので、本文を読む前にこちらを見ると当時の記述のどこが失効しているか分かる |

> **歴史は②・③側に書く（②は追記、③は不変）。** 生きた文書（①）には現在の姿だけを現在形で書き、経緯は②・③の文書に委ねること。
>
> **凡例**: ここに挙がっていない `Docs/` の文書は原則アーカイブ扱い（②または③のいずれか）。例外は[`../Videomni_Backend_Specification.md`](../Videomni_Backend_Specification.md) §0.3 の SSOT 表に載っているもの（アーカイブ形式の文書が仕様の正本を兼ねる場合がある。例: [`ICLORA_DEPTH_DEBLUR_WORKORDER.md`](ICLORA_DEPTH_DEBLUR_WORKORDER.md)）。

---

## 3. 開発の基本操作

### バックエンド

- 導入は `setup.bat`（中身は `scripts/setup.ps1` → `scripts/install_ltx.ps1`）、起動は `run.bat`（中身は `run.ps1`）。どちらもリポジトリ直下固定。
- **アプリ1プロセス＋エンジン系統ごとのワーカー構成**。アプリは `./.venv`（torch 無し）、LTX 2.3 の worker は `./.venv-engine`（torch+cu128）と `engine/`、LTX 2.5 の worker は `./.venv-engine-ltx25` と `engine25/`。**エンジン系統ごとに仮想環境とワーカーが1つずつある**（同居できない依存を持つため。理由は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §3.3・§5.3）。ワーカーは同時に1つだけ生き、ベースモデルを切り替えると旧ワーカーを落としてから新ワーカーを起こす。
- テストは `.\.venv\Scripts\python.exe -m pytest -q`。
- `config.yaml` は **git 追跡外**。配布されるのは `config.yaml.example` で、`setup.bat` / `run.bat` が無いときだけ複製する。**両者は同期させること。**
- `config.yaml` を書き換えたときの反映は**バックエンド再起動後**である。

### フロントエンド

作業ディレクトリは `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/`。

- ビルド: `powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Config Release`。Web UI（`webui/`）の `npm run build:single` を先に走らせ、その単一 HTML を `.aux2` へ埋め込む（**埋め込みが既定**）。出力は `build\ninja-release\NzVideomni.aux2`。
- デプロイ: `powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1 -Config Release`。**2箇所へ配る**——実機の AviUtl2 インストール先（既定 `D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin` の `NzVideomni\NzVideomni.aux2`。`Language\*.NzVideomni.aul2` は同じAviUtl2インストールの`data\Language\`〔`Plugin\`の兄弟〕へ）と、**本リポジトリの配布用コピー `AviUtl2-Plugin\NzVideomni.aux2`**。後者を忘れると、利用者が受け取るプラグインだけが古いままになる。
- 機械検証は3点セット（いずれも `webui/` で実行）: 型検査 `npm run typecheck`（`npx tsc --noEmit` は偽の合格を出すので使わないこと）・テスト `npm run test -- --run`（vitest）・静的検査 `npm run lint`。
- `.aux2` はビルドスクリプト経由でのみ作る（ninja を直接叩くと Web UI の埋め込みが更新されない）。

### MCP サーバー

`setup.bat` を一度実行すると、リポジトリ直下に `.mcp.json` が自動生成される（サーバーキーは `nz-videomni`、ツール22個）。マシン固有の絶対パスを含むため git 追跡外。

---

## 4. `models/` の配置とインストーラ

**ベースモデル優先レイアウト**——「置き場所がそのまま所属の宣言になる」形にしてある。

```text
models/
├─ LTX23/                LTX 2.3 一式
│   ├─ Weights/          transformer の GGUF（利用者の自家変換 GGUF もここへ同居する）
│   ├─ TextEncoder/      Gemma GGUF ＋ text projection ＋ tokenizer/
│   ├─ VAE/              映像・音声 VAE ＋ prunavaed/（枝刈り版デコーダ）
│   ├─ Upscaler/         空間アップスケーラ
│   ├─ StyleLoRA/        利用者が持ち込む画風・キャラクター LoRA
│   └─ IC-LoRA/          deblur / in-outpainting / pixel-spatial-upscaler / union-control
├─ LTX25/                LTX 2.5 一式
│   ├─ Weights/          transformer の GGUF
│   ├─ TextEncoder/      Gemma 4 GGUF（tokenizer は GGUF の中。隣の *.assets.safetensors は自動生成物）
│   ├─ VAE/              映像 VAE（畳み込みデコーダ版）＋音声 VAE ＋ diffvae/（拡散デコーダ版・退避先）
│   └─ Upscaler/         空間アップスケーラ
└─ Preprocessors/        ベースモデルに依存しない前処理器: DWPose/ ・ VDA/
```

- 各フォルダの `put_*_here.txt` は git 追跡。空フォルダのプレースホルダと「そこへ置ける形式」の掲示を兼ねる。
- **導入口は「本体＋ベースモデル別」の2階建てである。** `setup.bat` は本体のインストーラで、最初に試すAIとして **LTX 2.3 と共用前処理器だけ**を導入する。**別のベースモデルは専用のバッチで足す**——LTX 2.5 は `install-LTX25.bat`（リポジトリ直下）をダブルクリックすると、`scripts/install_model.ps1` を経て `scripts/install_ltx.ps1 -BaseModel LTX25 -SkipVenv -SkipMigrate` が走り、記述子が宣言するファイル一式が `models/LTX25/<カテゴリ>/` へ置かれる。設計正本は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §6.2、実装と実測は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §82（**開発機ゲートA0〜A8とサブマシンゲートG0〜G7の全合格をもって 2026-08-30 に完結**。台帳の項目は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-111へ移した）。
- **`setup.bat` の取得対象を分けているのは `install_ltx.ps1` の `-BaseModel` である**（記述子の `id` を並べる配列。既定値が「`setup.bat` が同梱する集合」そのもの）。**引数なしで呼ばれた `setup.bat` の検証テーブルに LTX 2.5 の行は出ない**——テーブルの中身を知りたいときは、数を書き写した文書ではなく `setup.bat` の出力そのものを見ること。**新しいベースモデルを足すときに `-BaseModel` の既定値へ手を入れる必要はない**（バッチ1本と記述子1本で足りる。既定値が伸びるのは、本体インストーラの同梱物そのものが増えるときだけである）。
- **LTX 2.5 の重みは HuggingFace で公開済みである**（Public・非 Gated）。[`Rootport/Nz-LTX25-weights`](https://huggingface.co/Rootport/Nz-LTX25-weights)（transformer GGUF・映像/音声 VAE・空間アップスケーラ）と[`Rootport/Nz-Gemma4-12B-LTX25`](https://huggingface.co/Rootport/Nz-Gemma4-12B-LTX25)（Gemma 4 テキストエンコーダ GGUF）の2本立てで、**ライセンス確認も完了している**。`install-LTX25.bat` が取りにいくのはこの2本である（手で置く場合も、記述子の期待名のまま `models/LTX25/<カテゴリ>/` へ置けばそのまま認識される）。ファイル一覧と SHA-256 の正本は[`LTX25_RESEARCH_NOTES.md`](LTX25_RESEARCH_NOTES.md) 10節。期待するファイル名は記述子と[`../README.md`](../README.md) §1「LTX 2.5 を追加する」の表にも書いてある。
- インストーラ（`scripts/install_ltx.ps1`）は **`scripts/manifests/*.json` に駆動される**。manifest が取得元リポジトリ・展開先・期待ファイルを宣言し、スクリプト自体はモデル名を持たない。新しいモデルを足すときは manifest を足す。
- **ガードは期待ファイル単位**である（ディレクトリ合計サイズではない）。`TextEncoder` が2つのリポジトリから供給されること、`Weights` に利用者の自家変換 GGUF が同居することの2点で、合計方式は破綻するため。
- **旧レイアウトからの自動移行を持つ。** 旧配置のファイルを新配置へ移動し、`config.yaml` 内のモデルパスも自動で書き換える（書き換え前に `config.yaml.bak` を作る）。移動は上書きしない方式で、実行前に安全性チェック（シンボリックリンク・衝突）を通る。
- 検証に失敗した場合は、**足りない1ファイルだけを消して再実行する**。`models\LTX23`・`models\LTX25` をフォルダごと消してはいけない（`StyleLoRA` や自家変換 GGUF は再取得できない利用者資産のため）。`install-LTX25.bat` の失敗案内も同じことを言う。

---

## 5. 直近の状況（2026-08-30 現在）

### 5.1 いま何が動くか

**LTX 2.5 で使えない機能は2つである**——`GET /models` の `unsupported_features` は現在2件（**非蒸留モデル**＝`two_stage_hq`、**PrunaVAED**＝`prune_vaed`）で、要求すると 422 で断る。**これは「エンジンが持たない機能」の件数であって、台帳に残っている課題の件数ではない**（課題の件数と番号の正本は台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §3 の表）。**それ以外はすべて動く**——基本生成（テキストから動画・画像から動画）・クリップ連結（Chained）・V2V継続（素材（冒頭）に動画を使って続きを作る）・A2V（音声から動画。Single・長尺・バッチのいずれも）・撮り直し（Retake）・素材（末尾）（End source）・スタイル LoRA・IC-LoRA（クリップ別の参照窓＝長尺 IC-LoRA を含む）・高速化4つ（GGUF逆量子化の1カーネル化・先読み block swap〔いずれも既定 on〕・モデル骨格の常駐＝`keep_resident`〔既定 off のオプトイン〕・SageAttention＝`attention_backend`〔既定は `"sdpa"` のまま〕）・画角拡張（Outpainting）・非CFGネガティブプロンプト（NAG／VSF）である。

**画面で灰色になるタブ・サブタブ・カードは1つも無い**（Edit タブの2つのサブタブも両方とも生きている）。フロントエンドが機能名を読んで灰色にしているのは `webui/src/shell/useBaseModels.ts` の3つの表だけで、そこに現れる語に `two_stage_hq` と `prune_vaed` は無い——**非蒸留の選択は `unsupported_features` を見ずに描かれ、押せば 422 が返る。**

**LTX 2.5 について「いま着手できる課題」は1つも無い。** 残るこの2機能はどちらも外部要因待ちである——**非蒸留モデルはそれを回せる計算機が、PrunaVAED は上流の LTX 2.5 版の枝刈り済み重みが、それぞれ手元に無い。** **次に動くのはこの前提が外から変わったときなので、次のテーマは台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §3-102 の外から選ぶことになる。** 台帳の「1. 近日中の改修項目」に残っているのも §1-4（オーナー自身が README を書く作業）だけなので、**次のテーマは台帳「3. 将来の研究課題」から選ぶ。**

### 5.2 テーマごとの正本の索引（どの節を読めばよいか）

**段ごとの経過を本書では年表にしない。** 設計上の要点は [`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §8.5 が、実測と既知の制限は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) の各節が正本である。下表は索引だけを持つ。

| テーマ | 開通 | 正本（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md)） |
|---|---|---|
| 基本生成（v1） | 2026-08-22 | §69（オーナーの実機確認は §69.21） |
| クリップ連結（Chained） | 2026-08-23 | §72（同 §72.10） |
| V2V 継続・A2V〔Single・長尺・バッチ〕 | 2026-08-23 | §73（同 §73.10。長尺 A2V はオーナー裁定で合格扱い＝同 §73.10(2)） |
| スタイル LoRA・IC-LoRA〔長尺を含む〕 | 2026-08-24 | §74（同 §74.12） |
| 高速化第1弾＝fused カーネルと先読み block swap〔既定 on〕 | 2026-08-24 | §75 |
| 高速化第2弾＝モデル骨格の常駐（`keep_resident`）〔既定 off〕 | 2026-08-25 | §76 |
| 高速化第3弾＝SageAttention（`attention_backend`）〔既定 `"sdpa"`〕 | 2026-08-25 | §77（目視は §77.10） |
| 撮り直し（Retake）・素材（末尾）（End source） | 2026-08-26 | §78（目視は §78.14・監督裁定の追認は §78.15） |
| 画角拡張（Outpainting）と「脱緑ブレンド」 | 2026-08-29 | §79（脱緑は §79.10・目視は §79.11・既定値2点の裁定は §79.12） |
| 非CFGネガティブプロンプト（NAG／VSF） | 2026-08-30 | §80（目視は §80.10） |
| 推論モードの統一（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-124） | 2026-08-30 | §81 |
| ベースモデル別インストールバッチ（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-111・同 §3-126） | 2026-08-30 | §82（サブマシンゲートは §82.4・再試行は §82.7） |

**技術的な教訓と設計上の要点も本書には写さない**——設計上の要点は [`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §8.5 の各段に、検証の教訓は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §75.11・§78.11・§81.9 にある。**同じ内容を本書へ写すと、正本が直っても本書だけが古くなる。**

### 5.3 あとから読む人が取り違えやすい点

1. **「速くなった」と「絵が変わった」を混ぜないこと。** 高速化3弾のうち**絵が変わるのは第3弾（SageAttention）だけ**で、第1弾・第2弾は固定ベンチマークの SHA-256 が従来と一致する。**目視ゲートが第3弾にだけあるのはこの違いによる。**
2. **「断っていた機能を開ける」作業と「無視していた項目を効かせる」作業とでは、動く表が違う。** `unsupported_features` が動くのは前者だけである。**さらに「この一覧が減った」と「画面のグレーアウトが解けた」も独立である**（設定画面は `unsupported_features` ではなく `GET /status` の能力フラグを見る箇所がある）。正本は [`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §8.5 の高速化第3弾の段と [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §77.9(5)・§77.4。
3. **実機ゲートの「LTX 2.3 が 108.2秒・LTX 2.5 が 77.2秒」を「2.5 のほうが速い」と読まないこと。** 交互対比較ではなく、**2.3 側はワーカーを載せ替えた直後の1本目（コールドキャッシュ込み）**である。速度計測の作法は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §43.6。
4. **画角拡張の目視ゲート（G8′）の正本は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §79.11 である**——同じ 2026-08-29 に脱緑ブレンドより前の合格宣言もあるが、脱緑で 8 腕すべての mp4 の SHA-256 が変わったため無効になっている（同 §79.10(12)）。
5. **「G8 に合格した」と「監督裁定が追認された」は別の出来事である**（撮り直し・素材（末尾）の裁定2件＝R-1・M8。合格は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §78.14(4)、追認は同 §78.15）。**どちらもいまは決着している。**

**既知の制限で、いちばん誤解されやすいものを1つだけ挙げておく。** **素材（末尾）をクリップ2本以上で使う逆順 Chained は、音声の継ぎ目が LTX 2.3 より劣化する**（数値の正本は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §78.9①）。**これはオーナーが試聴のうえ「想定の範囲内」と裁定した受容済みの仕様であり**（同 §78.14(3)。同じ裁定で「動画の継ぎ目で数フレームの早期収束」も受容されている）、**新しい不具合として再発見しないこと。** 残っている課題は原因の究明だけで、台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §3-117 にある。

### 5.4 残っているオーナー作業

**残っているのは README のスピードガイド執筆（台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §1-4）の1件だけである。** α版公開前に残っている唯一の作業で、**オーナー自身が手書きするもの**である（AI エージェントが実装するタスクではない）。書く前に読み返すものと、他に記録の無い覚書は同 §1-4 にまとめてある。

**目視ゲートも監督裁定の追認も、全テーマ分が決着している**（テーマごとの正本は §5.2 の表）。台帳の「2. 実装済み・ユーザーのテスト待ち」は、全項目が合格して空になったので節ごと削除してある。

いま効力を持っている裁定のうち、次の3つは**「オーナーが今のかたちを選んだ」状態**なので、見直すならその裁定を覆す話として起票すること。

- **画角拡張の既定値2点**——C1＝ぼかしの既定値は **5/2**、C2＝stage-2 は **2ステップ**のまま（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §79.12）。**この裁定には「経緯が文書で振り返れることを確認したうえでの了承」という条件が付いている**ので、**判断材料（同 §79.10 の測定と §79.11(3) の利害の整理）を要約で潰したり削ったりしてはならない。**
- **監督裁定2件（R-1・M8）**——R-1 はのりしろの一致度を判定する天井の取り方（**同一の処理鎖を通った天井＝読みB**）、M8 は先頭フレームのキーフレーム印の述語（**キャリー基準**）。追認で出荷構成は動いていない（同 §78.15）。
- **逆順 Chained（複数クリップ）は推奨外で、継ぎ目の劣化は仕様として受容する**（同 §78.14(3)）。

**押し込み（`git push`）の状態は、この行を読んで判断せず、必ず自分の目で測ること。** **確認は2つのコマンドだけである**——**`git status`（作業ツリーが空か）と `git log --oneline @{u}..HEAD`（未押し込みのコミットの一覧。`git log origin/main..main` でも同じ）。** **モノレポなので確認は1回で済む。** 本数もハッシュも本書には書かない（書いた端から古くなるため）。**押し込みはオーナー承認のうえで行う運用なので、エージェントは以後もコミットまでで止めること。**

**素材の置き場について1つ注意がある。`outputs/` は git の追跡外である。** 目視ゲートの素材（`outputs/ltx25-rtes-gate/g8_material/` や `outputs/ltx25-outpaint-gate/g8_material/`）も、その読み方を書いた `README.md` も、**リポジトリには入っていないのでオーナーの実機にしか存在しない。** 文書からこれらのパスを参照するときは「実機にあるもの」という前提で書くこと。**リポジトリに残る正本は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) の該当節のほうである。** また**目視の判定はどれも「オーナー自身の実機生成」で行われている**ので、素材フォルダの1本1本に個別の判定が付いた、とは読まないこと（同 §78.14(2)）。素材は消さずに残してあるので、台帳 §3-117 に着手するときの比較材料になる。

### 5.5 回帰の物差しと計測の作法

**回帰の物差しは揃っている**——固定ベンチマーク B1〜B4（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §72.7）・B5〜B8（同 §73.7）・B9〜B13（同 §74.8）を同じ条件で走らせ直せば、「速くなったが絵が変わっていないか」を mp4 の SHA-256 一致で判定できる。**基準 SHA は一度も動いていない**（同 §75.9・§76.5）。**第3弾（SageAttention）は絵が変わるので、この物差しは既定ジョブ（`sdpa`）の回帰確認にしか使えない**——`sage` 側の合否は PSNR と SSIM の相対の帯＋オーナー目視で出す（同 §77.2(3)・§77.5 の R2）。

- **速度を比べるときは交互対比較で測ること**（同一セッションで A・B・A・B と交互に走らせて対で差を取る。同 §43.6・§73.11(2)）。
- **コードを変えたあとの再ゲートは、必ずサーバーを再起動してから行うこと**——`POST /pipeline/load` は Python のモジュールを読み込み直さないので、上げたまま測ると古いコードを測ってしまう（同 §75.11）。
- **新しい品質指標を導入するときは、まず既知良品を測ってから線を引くこと。** 第3弾では当初置いた SSIM の絶対値の線を LTX 2.3 自身が割り、オーナー再裁定で作り直している（同 §77.2(3)）。

### 5.6 新しく起票された課題

**課題の一覧と状態は台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) が正本**で、件数も項目番号も本書には書かない。**2026-08-30 に起票した §3-127（`scripts/setup.ps1` の完了案内が実在しない `.au2pkg.zip` を案内している）と §3-128（`engine/venv-engine.freeze.txt` が `hf-xet` を固定していない）の2件**は、どちらもインストール導線を実機で通したときに見つかったもので（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §82.4・§82.7）、**同日中に実装しクローズ済みである**（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-127・§3-128）。**サブマシンでの `setup.bat` 再実行確認も同日中に合格し、両項目とも完全に決着している**（実測は同ログ §82.8）。

**MCP サーバーの引数の欠落は、性質の同じものが2件立っている**——§3-115（`submit_chain` に撮り直しの引数が無い）と §3-122（`submit_generate` に画角拡張の引数が無い）で、**着手するなら2件まとめて扱うのが自然である。**

LTX 2.5 まわりに着手する場合の読む順序: **①本書§2（文書の地図） → ②[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-98（v1 で何を作り何を作らなかったか）と[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-102（残っている作業） → ③[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §3.3・§5.3・§5.6（設計正本） → ④[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) の該当節（§5.2 の索引で引く） → ⑤[`../Videomni_Backend_Specification.md`](../Videomni_Backend_Specification.md) §6.10（API 契約）**。

---

## 6. 作業の進め方（恒久ルール）

- **コードに着手する前に実装計画を提示して合意を得る。** 承認は規模によらず必要。
- **テストが落ちたら勝手に直さず、原因を分析して報告する。** 先行事例を複製し、独自発明をしない。
- **push・マージはオーナー承認のうえで行う。** エージェントは原則コミットまで。
- **環境隔離を厳守する。** システム Python は触らない。依存はすべてプロジェクト配下の venv に閉じ込める。
- **人間向けの説明は普通のまともな日本語で書く。** 内部の略語は避け、使うときは短い解説を添える。結論を先に書く。
- 検証は交互対比較・機械検証・実機ゲートの順で積み上げ、結果は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md)へ、課題の増減は[`PENDING_TASKS.md`](PENDING_TASKS.md)へ記録する。
- **目視検証**: 720p級（1280×768）以上＋映画トレイラー風プロンプト＋「賑やかな町＋セリフ」題材で行う（512×320級は顔溶けで判断不能）。客観PASSとユーザー目視ゲートを混同しない。実験前に仮説→裏取り（手当たり次第の実験禁止）。
- **サブエージェント**: Opus以下を使う（Fable5禁止）・非破壊・能動ポーリング監視（ウォッチャー待ち停止禁止）・異常時は続行せず報告。GPU計測の一次ソースは**そのエンジン系統のワーカーログ**の`peak_vram_mb`——LTX 2.3なら`logs/ltx_worker.log`、LTX 2.5なら`logs/ltx25_worker.log`（アプリ側の`logs/server.log`とは別物）。
