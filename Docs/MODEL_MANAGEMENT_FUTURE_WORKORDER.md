# モデル管理機能（A1111 風）将来ワークオーダー — 現状記録と要件の入口

> **注記（2026-07-11）**: 2026-07-11 の UI 改修でモデル配置（GGUF 直下配置がデフォルトに）とプリセット名
> （`phase1_default`→`minimal`、`phase1_target`→`small`）が変更された。最新の配置・名称は
> [`README.md`](../README.md) を参照。§1 のレイアウト表は当時（2026-07-04）の記録のため変更しない。

> ## ✅ 実装済み（2026-07-05）
> **最小スコープ（下記★絞り込み）を実装・main マージ済み。** 4カテゴリレジストリ（transformer／text_encoder／video_vae／audio）＋`GET /models`＋`POST /pipeline/load` の optional モデル指定（省略時＝golden スナップショットで byte 同一固定）＋Settings タブのドロップダウン GUI。**正本＝[`MODEL_MANAGEMENT_DESIGN.md`](MODEL_MANAGEMENT_DESIGN.md)＋VERIFICATION_LOG §26**（S4 切替検証＝別名二重登録 swap で **SHA 一致×3** PASS・未知名 404）。**実代替モデルの DL は将来ユーザー任意**（本セッションでは切替配線の実証のみ・検証後に余計な登録は削除済み）。以下は着手前の記録として残す。
>
> **やらないと確定した項目**: §2-1 のディレクトリ再編（移行スクリプト・インストーラ書き換え・マニフェスト化）＝スコープ外（既存レイアウトのまま追加モデル置き場を走査する方式を採用）。

- 作成: 2026-07-04（ユーザー要望の記録・**設計/実装は別セッション**。ユーザー決定=今回は記録のみ）
- 要望の原文旨: モデルをドロップダウンで選択してロードできるようにしたい。`models/` を Text encoder・VAE・LTX2.3 などモデル種類ごとのディレクトリで管理する想定（A1111 SD WebUI / Forge neo に近い仕様）。現状の「LTX 2.3 だけでも蒸留・非蒸留などの種類ごとに個別ディレクトリ」は冗長。

> **★スコープ絞り込み（2026-07-05・ユーザー決定・こちらが正）**: 「すでに完成している内部の配線を壊したくない」。要件は**「複数のファインチューニングモデルをダウンロードして使うとき、ドロップダウンから選択できる」ことのみ**。→ §2-1 のディレクトリ再編（移行スクリプト・インストーラ書き換え・マニフェスト化）は**やらない**。最小スコープ＝①既存レイアウトのまま追加モデル置き場の走査 ②モデル列挙 API（加算・`GET /models` 等）③`pipeline/load` への optional モデル指定（省略時=現行既定・byte 同一）④GUI ドロップダウン（IC-LoRA ドロップダウン `gradio_ui/adapters.py:26-41` の型を踏襲）。検証＝既定構成の byte-match 回帰＋選択モデルの 16GB fit。**同枠の同時改修＝コンソールログ改修**（[`CONSOLE_LOG_FORGE_NEO_RESEARCH.md`](CONSOLE_LOG_FORGE_NEO_RESEARCH.md)・httpx ポーリング行の抑制＋工程名つき進捗/it 速度/所要時間/VRAM ピークのコンソール出力）。
>
> **★選択対象（2026-07-05 追記・ユーザー決定）**: LTX 本体（transformer GGUF）だけでなく、**テキストエンコーダ（Gemma）・VAE・音声系（音声 VAE＋vocoder。A2V の音声エンコードと通常生成の音声デコードが使う別重み）も種類ごとのドロップダウンで選択可能に**。設計上の注意＝組み合わせの互換性（既定の組み合わせを常に安全な既定値とし、差し替え時の shape/互換検査と失敗時の分かりやすいエラーを設計セッションで詰める）。

## 1. 現状のレイアウトと結合点（調査済み・2026-07-04 時点）

`models/` 直下（すべて `config.yaml` の `model:` ブロックで固定パス指定・`scripts/install_ltx.ps1` が配置）:

| ディレクトリ | 中身 | config キー |
|---|---|---|
| `ltx-2.3-gguf/LTX-2.3-distilled-1.1/` | GGUF transformer（Q4_K_M） | `gguf_transformer_path` |
| `gemma-3-12b-it-gguf/` | GGUF Gemma テキストエンコーダ | `gguf_gemma_path` |
| `gemma-3-12b-it-tokenizer/` | tokenizer のみ（~40MB） | `gemma_root` |
| `ltx-2.3-components/` | vae/（video+audio VAE）・text_encoders/（projection） | component 系パス |
| `ltx-2.3/` | spatial upsampler ＋ 参照専用 checkpoint パス | `spatial_upsampler_path`・`checkpoint_path`（未使用） |
| `ltx-2.3-ic-lora/` | IC-LoRA アダプタ（pixel-spatial-upscaler/・union-control/） | `model.ic_loras`（名前→パスの map） |
| `preprocessors/` | DWPose 等 TorchScript | （model: 外・前処理段が参照） |

**結合点（変更時に触る場所）**:
- `config.py:44-101 ModelConfig` ＝ 全アーティファクトが単一の固定パス文字列。
- `services/ltx_runner.py:688-782` ＝ config のパスを直接読んで worker ペイロードに焼き込む。「利用可能なモデルの列挙」「リクエスト毎のモデル選択」という概念が存在しない。切替＝config.yaml 編集＋サーバー再起動。
- `models/INSTALLED_PATHS.txt` ＝ インストーラの人間向けログでありアプリは読まない（機械可読マニフェストではない）。
- **既存の「選択 UI」前例は IC-LoRA アダプタのドロップダウンのみ**（`gradio_ui/adapters.py:26-41`・`model.ic_loras` の名前レジストリ駆動）＝種類別レジストリ＋ドロップダウンの最小の型がここにある。
- Load/Unload の現状: `POST /api/v1/pipeline/load|unload`（`api/pipeline.py`→`services/pipeline_manager.py:81-107`）＝config で固定された一式をロード/解放するだけ。

## 2. 将来設計セッションへの要件の入口（未確定・ユーザーと詰める）

1. **ディレクトリ再編**: 種類別（`models/transformers/`・`models/text_encoders/`・`models/vae/`・`models/upscalers/`・`models/loras/`・`models/preprocessors/`）へ移行。移行スクリプトと `install_ltx.ps1` の書き換え・`INSTALLED_PATHS.txt` の機械可読マニフェスト化（またはディレクトリ走査による発見）を伴う。
2. **API**: モデル列挙（`GET /models` 等・凍結 API への加算）と、`pipeline/load` へのモデル指定（optional・省略時は現行既定=後方互換）。
3. **GUI**: 種類ごとのドロップダウン（IC-LoRA ドロップダウンの型を踏襲）＋ Load ボタンとの結線・ロード中の進捗表示（現状は数分かかるロードにスピナーのみ）。
4. **検証観点**: 既定構成の byte-match 回帰（再編後も同一出力）・16GB fit の維持・dev(非蒸留) GGUF 等の将来重み（[`FEATURE_RESEARCH_2026-07-04.md`](FEATURE_RESEARCH_2026-07-04.md) D節）が同じ仕組みに乗ること。
5. **制約の継承**: 2プロセス/2venv・凍結 API は加算的拡張のみ・巨大ディスク要求をしない（[[no-large-pagefile-disk-requirement]] 相当の製品制約）。

## 3. 着手時の注意

- `checkpoint_path`（削除済み 43GB モノリスの参照専用キー）の扱いを整理する好機。
- モデル切替は VRAM/commit の再検証を伴う（keep=0 前提のロード/アンロードシーケンス）。
- スコープを「レイアウト再編＋列挙＋選択ロード」に絞り、モデルの自動ダウンロード等へ広げない。
