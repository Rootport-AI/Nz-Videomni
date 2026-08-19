# 次セッション引き継ぎ書 — Nz-Videomni

> **過去の引き継ぎ（日付ごとに積み上がっていた「最新ステータス」ブロック）は[`HANDOFF_ARCHIVE.md`](HANDOFF_ARCHIVE.md)へ移した。** 本書には現在有効な引き継ぎだけを置く。

---

## 1. このリポジトリの形

**`Nz-Videomni` はモノレポ（1つのリポジトリに複数の成果物を入れる方式）である。** バックエンドとフロントエンドが同じリポジトリに同居しており、片方だけを別リポジトリで開発することはもう無い。

| 場所 | 中身 |
|------|------|
| リポジトリ直下 | バックエンド（`main.py` / `api/` / `services/` / `engine/` / `gradio_ui/` / `mcp_server/`）。REST API サーバー本体 |
| `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/` | AviUtl2 拡張フロントエンドのソース（C++ プラグイン `native/` ＋ React/TypeScript の `webui/`） |
| `AviUtl2-Plugin/NzVideomni.aux2` | ビルド済みのプラグイン本体（配布物・git 追跡。利用者はこれを AviUtl2 へドラッグ＆ドロップする） |
| `Docs/` | プロジェクト全体の文書と課題台帳 |
| `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/` | フロントエンド固有の文書（ブリッジ契約・API リファレンス・実装ログほか） |

**旧フロントエンドリポジトリ `Nz-LTX23-frontend-AviUtl2` は凍結してある。** ディスク上には残っているが、以後の変更はすべて本リポジトリで行うこと。

**名前の使い分け**: 製品ブランドは **Nz-Videomni**（識別子は `NzVideomni` / `nz-videomni`）。**LTX 2.3・LTX23 はモデルの名前**であり、こちらは今後も残る（`models/LTX23/`・HuggingFace の `Rootport/Nz-LTX23-weights`・`Nz-GGUF-Converter-LTX23` など）。将来 LTX 2.5 や Wan 2.x といった別のモデルも載せられる基盤を目指しているため、製品名からモデル名を外してある。

---

## 2. 文書の地図（どれを読み、どれを直すか）

**生きた文書**（現在の仕様・手順を書くもの。読む人は必ずここを見る）

| 文書 | 役割 |
|------|------|
| [`../README.md`](../README.md) | 利用者向けの入口。セットアップ導線・起動・`models/` の構成・API 概要・MCP サーバー |
| [`../Videomni_Backend_Specification.md`](../Videomni_Backend_Specification.md) | バックエンド仕様書。**凍結 API 契約（§6）はこの文書が正本** |
| [`PENDING_TASKS.md`](PENDING_TASKS.md) | **プロジェクト全体の課題台帳。**「次に何をすべきか」はここで確認する |
| [`STORAGE_POLICY.md`](STORAGE_POLICY.md) | `outputs/` / `uploads/` の設計原則（Outputs は宝物・Uploads は一時置き場） |
| [`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) | クリップ連結（Clip Chain）の内部構造と現行アーキテクチャの設計正本 |
| [`RESOLUTION_DURATION_CAPABILITY.md`](RESOLUTION_DURATION_CAPABILITY.md) / [`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) | 解像度×尺の能力（spill-free 閾値・生成時間）と快適上限の各正本 |
| [`ACCELERATION_RESEARCH_NOTES.md`](ACCELERATION_RESEARCH_NOTES.md) / [`LTX23_REFERENCE.md`](LTX23_REFERENCE.md) / [`MCP_SERVER_DESIGN.md`](MCP_SERVER_DESIGN.md) | 高速化候補の整理・LTX 2.3 の一般知識・MCP サーバー設計の各正本 |
| フロントエンド [`API_REFERENCE.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/API_REFERENCE.md) | API 利用者（フロントエンド実装者）向けの正本 |
| フロントエンド [`BRIDGE_CONTRACT.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/BRIDGE_CONTRACT.md) | ネイティブ ↔ Web UI の JSON-RPC 契約 |
| フロントエンド [`REAL_BACKEND_CHECKLIST.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/REAL_BACKEND_CHECKLIST.md) | 実バックエンド接続時の確認手順 |

**アーカイブ文書**（歴史の保管庫。内容は当時のままで、更新しない）

- [`HANDOFF_ARCHIVE.md`](HANDOFF_ARCHIVE.md) — 過去の引き継ぎエントリ
- [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) — 実機検証の全経緯。**各テーマの最新状態は必ずここの該当節（節末尾の「状態」表記）で確認する**
- [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) — クローズ済み項目の完了記録
- フロントエンド `Docs/DEVLOG.md` — フロントエンド側の実装ログ
- 各 `*_WORKORDER.md` / `*_STATUS.md` / `*_RESEARCH.md` と、完了済み・凍結済みの設計書（[`MODEL_MANAGEMENT_DESIGN.md`](MODEL_MANAGEMENT_DESIGN.md)、[`AVIUTL2_DESIGN_BRIEF.md`](AVIUTL2_DESIGN_BRIEF.md) など。**当時の記述のままなので、旧ディレクトリ構成や旧モデル配置が残っている**）

> **歴史はアーカイブ側に書く。** 生きた文書には現在の姿だけを現在形で書き、経緯はアーカイブ文書に委ねること。

---

## 3. 開発の基本操作

### バックエンド

- 導入は `setup.bat`（中身は `scripts/setup.ps1` → `scripts/install_ltx.ps1`）、起動は `run.bat`（中身は `run.ps1`）。どちらもリポジトリ直下固定。
- **2プロセス・2venv 構成**。アプリは `./.venv`（torch 無し）、実推論 worker は `./.venv-engine`（torch+cu128）。
- テストは `.\.venv\Scripts\python.exe -m pytest -q`。
- `config.yaml` は **git 追跡外**。配布されるのは `config.yaml.example` で、`setup.bat` / `run.bat` が無いときだけ複製する。**両者は同期させること。**
- `config.yaml` を書き換えたときの反映は**バックエンド再起動後**である。

### フロントエンド

作業ディレクトリは `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/`。

- ビルド: `powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Config Release`。Web UI（`webui/`）の `npm run build:single` を先に走らせ、その単一 HTML を `.aux2` へ埋め込む（**埋め込みが既定**）。出力は `build\ninja-release\NzVideomni.aux2`。
- デプロイ: `powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1 -Config Release`。**2箇所へ配る**——実機の AviUtl2 インストール先（既定 `D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin` の `NzVideomni\NzVideomni.aux2`。`Language\*.NzVideomni.aul2` も同じインストール先へ）と、**本リポジトリの配布用コピー `AviUtl2-Plugin\NzVideomni.aux2`**。後者を忘れると、利用者が受け取るプラグインだけが古いままになる。
- 型検査は `npm run typecheck`（`webui/` で実行）。`npx tsc --noEmit` は偽の合格を出すので使わないこと。
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
└─ Preprocessors/        ベースモデルに依存しない前処理器: DWPose/ ・ VDA/
```

- 各フォルダの `put_*_here.txt`（全13本）は git 追跡。空フォルダのプレースホルダと「そこへ置ける形式」の掲示を兼ねる。
- インストーラ（`scripts/install_ltx.ps1`）は **`scripts/manifests/*.json` に駆動される**。manifest が取得元リポジトリ・展開先・期待ファイルを宣言し、スクリプト自体はモデル名を持たない。新しいモデルを足すときは manifest を足す。
- **ガードは期待ファイル単位**である（ディレクトリ合計サイズではない）。`TextEncoder` が2つのリポジトリから供給されること、`Weights` に利用者の自家変換 GGUF が同居することの2点で、合計方式は破綻するため。
- **旧レイアウトからの自動移行を持つ。** 旧配置のファイルを新配置へ移動し、`config.yaml` 内のモデルパスも自動で書き換える（書き換え前に `config.yaml.bak` を作る）。移動は上書きしない方式で、実行前に安全性チェック（シンボリックリンク・衝突）を通る。
- 検証に失敗した場合は、**足りない1ファイルだけを消して再実行する**。`models\LTX23` ごと消してはいけない（`StyleLoRA` や自家変換 GGUF は再取得できない利用者資産のため）。

---

## 5. 直近の状況（2026-08-19 現在）

- **End source（素材（末尾）＝添付した画像・動画へ繋がる動画の生成）テーマは完結している。** オーナー裁定により、**クリップ1本での使用が推奨**、複数クリップは受理されるが推奨外（品質劣化は仕様として許容）。品質重視で複数クリップを繋ぐ場合の実用手順は「正順 Chained ＋最終クリップだけ補間仕上げ」で、[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) §11 と `README.md` の手動リレー節に記載がある（**実機検証は未実施**）。
- **Single タブの賢い快適上限マーカーも完結している。** 5つの高速化トグルが全て on のときだけ `limits.single_comfort_token_budget`（44,880）からの逆算式でマーカーを引き、1つでも off なら従来の `spill_free_frames` へフォールバックする。
- **台帳の「1. 近日中の改修項目」に残っているのは §1-4 だけ**——オーナー自身が README のスピードガイドを書く作業であり、AI エージェントが実装するタスクではない。次に着手する候補は「3. 将来の研究課題」から選ぶ（直近の起票は §3-96 End source 付き連結クリップの改善研究、§3-95 快適上限の適用拡大、§3-54／§3-55 の軽量ユーティリティAI と Inpainting）。
- **プリセットのフレーム数を引き上げ済み**（720p 361 / FHD 169 / WQHD 89、既定 361）。`config.yaml`・`config.yaml.example`・フロントエンドの `defaultConfig.ts` の3点を一致させてある。反映はバックエンド再起動後。
- **ヘッダーのモデル名ドロップダウン（「LTX 2.3」既定 ／「LTX 2.5」はモック）を新設済み。** オーナーの目視確認は次回セッション以降。

---

## 6. 作業の進め方（恒久ルール）

- **コードに着手する前に実装計画を提示して合意を得る。** 承認は規模によらず必要。
- **テストが落ちたら勝手に直さず、原因を分析して報告する。** 先行事例を複製し、独自発明をしない。
- **push・マージはオーナー承認のうえで行う。** エージェントは原則コミットまで。
- **環境隔離を厳守する。** システム Python は触らない。依存はすべてプロジェクト配下の venv に閉じ込める。
- **人間向けの説明は普通のまともな日本語で書く。** 内部の略語は避け、使うときは短い解説を添える。結論を先に書く。
- 検証は交互対比較・機械検証・実機ゲートの順で積み上げ、結果は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md)へ、課題の増減は[`PENDING_TASKS.md`](PENDING_TASKS.md)へ記録する。
