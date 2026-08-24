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
| [`LTX25_RESEARCH_NOTES.md`](LTX25_RESEARCH_NOTES.md) | オーナーによる 2026-08-20 共有の LTX 2.5 事前調査。**マルチエンジン設計の議論より前に行われた調査であり、設計の正本ではない。**設計の正本は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md)（①生きた文書）。本文は改変せず収蔵してあるので、更新するのではなく、新しい知見は設計正本側へ書く。**末尾に「実装してみた結果との差分」の章を2つだけ足してある**（「§3-98 v1実装完了」と「§3-102 第3段 実装完了」）——調査時の想定と実装結果の差分を一覧にしたもので、本文を読む前にこちらを見ると当時の記述のどこが失効しているか分かる |

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

- 各フォルダの `put_*_here.txt`（**全18本**。`git ls-files` 実測）は git 追跡。空フォルダのプレースホルダと「そこへ置ける形式」の掲示を兼ねる。
- **LTX 2.5 の記述子（`scripts/manifests/20-ltx25.json`）は `downloads` が空**で、インストーラは LTX 2.5 のぶんを単に飛ばす（最後の検証テーブル 15 行にも LTX 2.5 は出ない）。**これは意図した状態である**——`setup.bat` は本体のインストーラで、最初に試すAIとして LTX 2.3 だけを導入し、**別のベースモデルは専用のバッチ（`install-LTX25.bat` 等）で導入する**という設計である（[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §6.2）。バッチ群の整備までは `downloads` を空のまま置くこと（埋めると `setup.bat` が LTX 2.5 まで取りにいく）。整備は[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-111。
- **LTX 2.5 の重みは HuggingFace で公開済みである**（Public・非 Gated）。[`Rootport/Nz-LTX25-weights`](https://huggingface.co/Rootport/Nz-LTX25-weights)（transformer GGUF・映像/音声 VAE・空間アップスケーラ）と[`Rootport/Nz-Gemma4-12B-LTX25`](https://huggingface.co/Rootport/Nz-Gemma4-12B-LTX25)（Gemma 4 テキストエンコーダ GGUF）の2本立てで、**ライセンス確認も完了している**。ダウンロードして `models/LTX25/<カテゴリ>/` へ記述子の期待名のまま置けば、そのまま認識される。ファイル一覧と SHA-256 の正本は[`LTX25_RESEARCH_NOTES.md`](LTX25_RESEARCH_NOTES.md) 10節。期待するファイル名 5 本は記述子と[`../README.md`](../README.md) §1「models フォルダの構成」にも書いてある。
- インストーラ（`scripts/install_ltx.ps1`）は **`scripts/manifests/*.json` に駆動される**。manifest が取得元リポジトリ・展開先・期待ファイルを宣言し、スクリプト自体はモデル名を持たない。新しいモデルを足すときは manifest を足す。
- **ガードは期待ファイル単位**である（ディレクトリ合計サイズではない）。`TextEncoder` が2つのリポジトリから供給されること、`Weights` に利用者の自家変換 GGUF が同居することの2点で、合計方式は破綻するため。
- **旧レイアウトからの自動移行を持つ。** 旧配置のファイルを新配置へ移動し、`config.yaml` 内のモデルパスも自動で書き換える（書き換え前に `config.yaml.bak` を作る）。移動は上書きしない方式で、実行前に安全性チェック（シンボリックリンク・衝突）を通る。
- 検証に失敗した場合は、**足りない1ファイルだけを消して再実行する**。`models\LTX23` ごと消してはいけない（`StyleLoRA` や自家変換 GGUF は再取得できない利用者資産のため）。

---

## 5. 直近の状況（2026-08-25 現在）

**LTX 2.5 で使えるのは、基本生成（テキストから動画・画像から動画）・クリップ連結（Chained）・V2V継続（素材（冒頭）に動画を使って続きを作る）・A2V（音声から動画。Single・長尺・バッチのいずれも）・スタイル LoRA・IC-LoRA（参照動画による制御。クリップ別の参照窓＝長尺 IC-LoRA を含む）・高速化2つ（GGUF逆量子化の1カーネル化・先読み block swap。いずれも既定 on）である。** まだ無いのは Retake・素材（末尾）・Outpainting・NAG／VSF・残る高速化3つ（`keep_resident`・SageAttention・PrunaVAED）で、要求すると 422 で断り、画面側でもパネルが灰色になる。**Chained タブは LTX 2.5 でも開き、その中で灰色のまま残るのは素材（末尾）の1カードだけ**である（`unsupported_features` は8件）。後続は台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-102。

**ここまでの機能は、すべてオーナーの実機確認に合格している。** 基本生成（v1）は 2026-08-22（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §69.21）、クリップ連結は 2026-08-23（同 §72.10）、V2V継続と A2V は 2026-08-24（同 §73.10）、**スタイル LoRA と IC-LoRA は 2026-08-25**（同 §74.12。AviUtl2 の操作パネルから両方を試し、いずれも効果があることを確認して合格）。**長尺A2V はオーナー裁定で合格扱い**（裁定の全文と理由は同 §73.10(2)）。**使うときの注意が2つある**——スタイル LoRA は**トリガー語（そのLoRAの合言葉）をプロンプトに入れないと効きが穏やか**で、輪郭線制御（canny）は**鮮明な参照素材でないとエッジがほとんど出ない**（同 §74.9）。16GB の回避策は 1 段も使っていない（1920×1088・169フレームの V2V まで確認済み）。実測の正本は同 §69（v1）・§72（連結生成＋固定ベンチマーク B1〜B4）・§73（V2V・A2V＋B5〜B8）・§74（LoRA・参照動画＋B9〜B13）、API 契約は[`../Videomni_Backend_Specification.md`](../Videomni_Backend_Specification.md) §6.10、設計正本は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §5.6・§8.5、方式の差分は[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) 12節である。

**高速化第1弾（2026-08-24 実装）は全ゲート合格で完了しており、オーナーの目視作業も無い。** GGUF逆量子化の1カーネル化と先読み block swap を LTX 2.5 でも開通させ、**両方とも既定 on** にした。**2つ併用で生成時間が 61.8%短縮**（実測 102.86秒 → 39.28秒）で、**固定ベンチマークB系17本すべて mp4 の SHA-256 が従来と一致**——速くなったが絵は1バイトも変わっていないので、目視ゲートは設けていない。**`unsupported_features` は8件のまま動いておらず、フロントエンドのコードも1行も触っていない**（この2つは 422 で断っていた機能ではなく「無視＋ログ」の側にあった項目のため）。実測の正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §75（B4のVRAM超過とアリーナリングの顛末は同 §75.7、検証の教訓は同 §75.11）。

### 残っているオーナー作業（無し）

**2026-08-25 の時点で、オーナー側に持ち越している作業は無い。**

- **目視は全て決着した。** 最後まで残っていたスタイル LoRA と IC-LoRA の目視は 2026-08-25 に合格した（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §74.12）。**高速化第1弾にはもともと目視作業が無い**（出力がビット単位で不変であることを機械で確認済みのため）。
- **押し込みも済んだ。** 最後まで未押し込みだった高速化第1弾のぶん7本（実装5本＝`d4311ca` / `ae21067` / `0ed330b` / `22aebf7` / `6027180`＋文書2本＝`fe1afb2` / `5e35016`）は 2026-08-25 に押し込まれ、`git status -sb` は `## main...origin/main`（`ahead` の表示なし＝未押し込み0本）になっている。**本数の正本は引き続き `git status -sb` の `ahead` 表示**なので、次に押し込む前にもそれを見ること。押し込みはオーナー承認のうえで行う運用なので、**エージェントは以後もコミットまでで止めること。**

### 次に着手する候補

台帳の「1. 近日中の改修項目」に残っているのは §1-4 だけ（オーナー自身が README のスピードガイドを書く作業であり、AI エージェントが実装するタスクではない）。したがって次のテーマは「3. 将来の研究課題」から選ぶ。

**次のテーマは §3-102 の「高速化技術の LTX 2.5 前倒し」の第2弾である。** 順番はオーナー裁定（2026-08-23）の段階分け「必須級 → 高速化技術の前倒し → 準必須級 → 実験的」で決まっており、**必須級（クリップ連結・V2V・A2V・スタイル LoRA・IC-LoRA）と、高速化の第1弾（fused カーネル・先読み block swap）は 2026-08-24 で終わっている**。

**第2弾はモデル骨格の常駐（`keep_resident`）のテキストエンコーダ側**で、**既定 OFF のオプトイン**として入れる想定である。**第3弾は SageAttention（`attention_backend`）**で、こちらは出力の細部が変わるので **PSNR（ピーク信号対雑音比。元の絵とどれだけ違うかを1つの数値で表す指標で、単位は dB。数値が大きいほど元の絵に近い）を基準にした判定**が要る（§43 の注意書きと同じ話である）。**PrunaVAED（`vae_mode`）は LTX 2.5 版の枝刈り済み重みが存在しないため先送り**である。第2弾・第3弾はいずれもアダプタの `REJECT_TABLE` / `CHAIN_REJECT_TABLE` から該当行が外れ、`unsupported_features` から名前が1つ消えることが完了の合図になる——**第1弾はここが動かない種類の作業だった**（もともと「無視＋ログ」の側にあった項目を「動作」へ移しただけなので、422 の表も `unsupported_features` も無変更である）。

**回帰の物差しは揃っている**——固定ベンチマーク B1〜B4（同 §72.7）・B5〜B8（同 §73.7）・B9〜B13（同 §74.8）を同じ条件で走らせ直せば、「速くなったが絵が変わっていないか」を mp4 の SHA-256 一致で判定できる。**第1弾でもこの物差しで判定しており、基準 SHA は1つも動いていない**（同 §75.9）。**速度を比べるときは交互対比較で測ること**（同一セッションで A・B・A・B と交互に走らせて対で差を取る。理由は同 §43.6・§73.11(2)）。**コードを変えたあとの再ゲートは、必ずサーバーを再起動してから行うこと**——`POST /pipeline/load` は Python のモジュールを読み込み直さないので、上げたまま測ると古いコードを測ってしまう（同 §75.11）。

そのほかの直近の起票は §3-103（拡散デコーダ版 VAE と決定性）・§3-104（インストーラの `-ResolveLatest` の不具合）・§3-105（2.3 ワーカーの 2 ジョブ目以降のせり上がり）・§3-107（ストレージ必要容量の再実測）・§3-108（kohya 形式 LoRA をローダーが読めない）・§3-109（チャンク化 upsample の廃止）・§3-111（ベースモデル別インストールバッチの整備）・**§3-112（CUDA アロケータ設定の見直し。高速化第1弾の原因究明から派生）**・**§3-113（先読み block swap の解放処理をスロット単位で囲う）**・§3-96・§3-95・§3-54／§3-55 である。

LTX 2.5 まわりに着手する場合の読む順序: **①本書§2（文書の地図） → ②[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-98（v1 で何を作り何を作らなかったか）と[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-102（残っている作業） → ③[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §3.3・§5.3・§5.6（設計正本） → ④[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §69・§71〜§75（実測。**高速化は §75**） → ⑤[`../Videomni_Backend_Specification.md`](../Videomni_Backend_Specification.md) §6.10（API 契約）**。
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
