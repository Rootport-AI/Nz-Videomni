# 調査メモ: Forge Neo の生成時コンソールログ

作成日: 2026-07-05
目的: 我々のLTX-2.3動画生成バックエンドは、生成中のコンソールが `GET /jobs/{id} 200 OK` のポーリング行で埋まり、進捗・速度・所要時間などの意味ある情報が出ない。将来のログ改修の参考として、画像生成ツール「Forge Neo」が生成実行中にコンソールへ何を出しているかを実装ソースから調査した。

---

## 1. 「Forge Neo」の同定

- 本体: **Haoming02/sd-webui-forge-classic** リポジトリの **`neo` ブランチ**。
  - リポジトリ: https://github.com/Haoming02/sd-webui-forge-classic
  - Neoブランチ: https://github.com/Haoming02/sd-webui-forge-classic/tree/neo
  - README: https://github.com/Haoming02/sd-webui-forge-classic/blob/neo/README.md
- 系譜: `AUTOMATIC1111/stable-diffusion-webui`(元祖) → `lllyasviel/stable-diffusion-webui-forge`(Forge、メモリ管理最適化フォーク) → 開発停滞後、`Haoming02/sd-webui-forge-classic` がメンテナンスを継続 → 同リポジトリ内で「Classic」(旧Gradio3系)と「Neo」(Gradio4.40系、Forge2/最新機能追随)の2ブランチに分岐。
- Neoブランチは `lllyasviel/stable-diffusion-webui-forge:main` に対して500コミット以上先行しており、Flux/GGUF/Qwen-Image/Wan2.2/Nunchakuなど新モデル対応を継続追加する形で維持されている。
- 派生・関連プロジェクト(参考、本調査の主対象ではない): `6Morpheus6/forge-neo`(独立フォーク)、`Panchovix/stable-diffusion-webui-reForge` など。本調査は最も直系である Haoming02 版 `neo` ブランチのソースコードを直接読んで検証した。

**重要な前提**: Forge/Forge Neoはコンソール出力の大半をA1111由来のロジック(`modules/`配下)からほぼそのまま継承している。Forge固有の追加は主に「メモリ管理(モデルのGPU/CPU移動)」と「モデルロード時の設定テーブル表示」。したがって以下のリストは「A1111由来(共通)」と「Forge固有」を明記する。

---

## 2. 生成時コンソールログの実例リスト

各項目: **出力内容** / **出典ファイル・行(neoブランチ)** / **系譜(共通 or Forge固有)**

### 2.1 モデルロード関連(生成リクエスト受信時、遅延ロードで発火)

| # | ログ内容(実際のフォーマット文字列) | 出典 | 系譜 |
|---|---|---|---|
| 1 | `Loading Model: {forge_loading_parameters}` — チェックポイント名/VAE/CLIP skip等の読み込みパラメータ一式を1行で表示 | `modules/sd_models.py` `forge_model_reload()` | Forge固有 |
| 2 | `Model loaded in {timer.summary()}.` — 例: `Model loaded in 4.2s (calculate hash: 0.3s, load config: 0.1s, ...)` のように総時間+内訳(0.1s以上のみ)を括弧内に列挙 | `modules/sd_models.py` / `modules/timer.py` `Timer.summary()` | 共通(A1111由来のTimerクラス) |
| 3 | `Requested to load {ModelClassName}` — メモリ管理レイヤーがどのモデル(UNet/VAE/CLIP等)をロード対象にしたか | `backend/memory_management.py:650` | Forge固有 |
| 4 | `Moving model(s) has taken {X:.2f} seconds` — モデルをGPU/CPU間で移動(オフロード)した実測時間 | `backend/memory_management.py:703` | Forge固有 |
| 5 | `Unloading {ModelClassName}` / `{N} models unloaded.` — 旧モデルのアンロード | `backend/memory_management.py:599,676` | Forge固有 |
| 6 | `Potential memory leak detected with model {X}...` / `Memory Leak with model {X} !` — メモリリーク検知警告 | `backend/memory_management.py:731,746` | Forge固有 |
| 7 | 起動時1回: `Total VRAM {X} MB, total RAM {Y} MB` / `VRAM State: {STATE}` / `Device: {device_name}` | `backend/memory_management.py:189,372,399` | Forge固有(起動時だが把握の参考として記載) |
| 8 | `list_loaded_weights()` によるロード済みモデル一覧の**リッチテーブル表示**(`rich`ライブラリ使用、タイトル "Currently Loaded Weights"、列: Model / VRAM(MB) / Device) | `modules/sd_models.py` `list_loaded_weights()` | Forge固有 |

### 2.2 LoRA/追加ネットワークのロード

| # | ログ内容 | 出典 | 系譜 |
|---|---|---|---|
| 9 | `[LORA] Loading {filename} for {model_flag} with {N} unmatched keys` | `extensions-builtin/sd_forge_lora/networks.py:62` | Forge固有 |
| 10 | `[LORA] Loaded {filename} for {model_flag}-UNet with {N} keys at weight {strength} (skipped {M} keys) with on_the_fly = {bool}` (CLIP版も同様) | `extensions-builtin/sd_forge_lora/networks.py:71,81` | Forge固有 |
| 11 | `[LORA] Mismatch {filename} for {model_flag}-UNet with {N} keys mismatched...` (警告) / `Failed to load LoRA: "{name}"` (エラー) | 同上 :69,79,122 | Forge固有 |

我々のプロジェクトはic-loraを扱っているため、この「ロード成功時に適用強度・キー数・不一致数を1行で吐く」形式はそのまま参考になる。

### 2.3 サンプリング進捗(生成本体、ステップ単位)

| # | ログ内容 | 出典 | 系譜 |
|---|---|---|---|
| 12 | **ステップ進捗バー**(tqdm) — サンプラー本体(k-diffusion由来の`sample_euler`等の関数内)が標準tqdmで `it/s` 付きバーを出す。実例: `50%|█████     | 10/20 [00:05<00:05, 2.00it/s]` | `modules/sd_samplers_common.py`(`launch_sampling`が呼ぶサンプラー関数内部、k-diffusionライブラリのtqdm使用) | 共通(A1111由来、k-diffusion外部ライブラリのtqdm) |
| 13 | **"Total progress" バー** — `shared.total_tqdm`(`TotalTQDM`クラス)が全ジョブ×全ステップ数を`desc="Total progress"`で表示。実例(ユーザー実測ログより): `Total progress: 100%|████████████████████████████████████████████████████| 20/20 [00:02<00:00, 8.44it/s]` | `modules/shared_total_tqdm.py` `TotalTQDM` / `modules/sd_samplers_common.py:389 shared.total_tqdm.update()` | 共通(A1111由来の仕組み。`opts.multiple_tqdm`設定でON/OFF可、`cmd_opts.disable_console_progressbars`でも無効化可) |
| 14 | ジョブ開始/終了のログ: `Starting job {job}` / `Ending job {job} ({duration:.2f} seconds)` (Python `logging`経由、コンソールに出る) | `modules/shared_state.py` `State.begin()`/`end()` | 共通(A1111由来のStateクラス) |
| 15 | 割り込み系: `Received skip request` / `Received interrupt request` / `Received stop generating request` | `modules/shared_state.py` | 共通 |
| 16 | サンプリング中の例外時: `Encountered RecursionError during sampling; try to use a smaller rho value instead` | `modules/sd_samplers_common.py:401` | 共通 |

### 2.4 完了時のサマリ情報

| # | ログ内容 | 出典 | 系譜 |
|---|---|---|---|
| 17 | **所要時間表示**: `Time taken: {elapsed_text}` (例: `12.3 sec.` または `1 min. 5.2 sec.`) | `modules/call_queue.py`(`wrap_gradio_call`) | 共通(A1111由来) — **注: これはUIのHTMLパネルへの表示であり、標準出力/ロガー経由のコンソール出力ではない**。ただし内部でロガーに出す設定にも転用しやすい形。 |
| 18 | **VRAM統計**(有効時): Active peak `{X:.2f} GB` / Reserved peak `{X:.2f} GB` / System peak `{X:.1f}/{total:g} GB ({pct:.1f}%)` — `shared.mem_mon`(バックグラウンドでVRAM使用量をポーリングするメモリモニタスレッド)が生成中ずっと監視し、生成後にピーク値を取得 | `modules/call_queue.py`(`shared.mem_mon.stop()`の戻り値を整形) | 共通(A1111由来のMemUsageMonitor) — 同上、UI表示だがログ転用の参考価値大 |
| 19 | Timer内訳の再掲: `{category}: done in {X:.3f}s`(`--log-startup`相当の詳細モード時、各サブカテゴリの所要時間) | `modules/timer.py` `Timer.record()` | 共通 |

---

## 3. 動画生成バックエンドへの転用適性メモ

我々の構成は「2プロセス(APIサーバ+ワーカー)・2段パイプライン(base→upsample相当のステージ)・VAEデコード(セグメント/タイル処理あり)・ジョブポーリングAPI」という前提がある。各ログ種別の転用適性:

- **#1 `Loading Model:` 相当** — 適性高。ワーカーがモデルをロードする瞬間に「どのcheckpoint/LoRA/解像度/フレーム数で開始するか」を1行で吐くのは即転用できる。現状ジョブ投入時のパラメータ echo が弱いなら参考になる。
- **#2 `Model loaded in Xs (内訳)`** — 適性高。ロード時間の内訳表示は、ワーカー起動直後の「モデルロード完了まで待たされている無音状態」の可視化に直結する。
- **#3〜#6 メモリ管理系(Requested to load / Moving model(s) has taken / Unloading)** — 適性中〜高。我々はVRAM 16GB制約が最重要不変条件なので、「どのモデルをいつVRAMへ乗せ替えたか」「移動に何秒かかったか」をログ化するのはVRAM溢れ対策(WDDM是正等、既存作業と関連)のデバッグに直接役立つ。
- **#8 ロード済みモデル一覧テーブル** — 適性中。`rich`ライブラリでの整形テーブルは視認性が良い実装パターンとして参考になるが、必須ではない。
- **#9〜#11 LoRAロードログ** — 適性高。ic-lora機能と直結。「適用強度・不一致キー数・on_the_fly判定」を1行で出す形式はそのまま流用できる。
- **#12 ステップ進捗バー(it/s)** — 適性は**動画生成特有の事情で要調整**。画像は1本のサンプリングループでステップ進捗=itそのものだが、動画(特に2段パイプライン)は「ステージ1のデノイズステップ」「ステージ2(upsample/refine)のデノイズステップ」「VAEデコード(セグメント/タイル単位)」がそれぞれ別の進捗軸になる。単純な1本のtqdmバーではなく、**ステージ名+ステップ進捗+it/sを併記する多段バー**(Forgeの`TotalTQDM`のように「全体」と「現在ステージ」の2階層)が必要。ここはForgeのアーキテクチャ(1本の画像=1本のバー)をそのまま持ち込めない部分。
- **#13 "Total progress" バー** — 適性中。バッチ画像生成の「Nバッチ×Mステップ」進捗の思想は、我々の「セグメント数×ステップ数」や「チェーン内クリップ数×ステップ数」に構造的に近い。マッピングして流用可能。
- **#14 `Starting job` / `Ending job (Xs)`** — 適性高。httpxポーリング行の代わりに、ジョブの開始・終了だけをロガーで明示するだけでも現状の「意味のある情報が出ない」問題はかなり改善する。最小コストで効果が大きい候補。
- **#16 例外時メッセージ** — 適性高。動画生成でも「メモリ不足でタイルVAEにフォールバック」等の類似分岐があるはずで、Forgeの`Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding`のような一言警告パターンはそのまま真似できる(検索結果より、本家Forgeの既知メッセージ)。
- **#17 `Time taken:`** — 適性高。ただし転用時はUIパネルではなく**ロガー経由でコンソールに直接出す**必要がある(Forgeは元々UI表示用でコンソールには出していない点に注意)。動画生成は1ジョブが長時間(分単位)になりやすいので、完了時の合計時間表示は重要度が高い。
- **#18 VRAM peak統計** — 適性高。「一次ソース=ltx_worker.logのpeak_vram_mb」という既存の計測方針(運用ルールに記載)と親和性が高い。Forgeの`mem_mon`のように生成中バックグラウンドでポーリングし、完了時にピークをログへ出す設計は、既存のVRAM計測の仕組みと統合しやすい。

**総括**: httpxポーリングログを埋もれさせないための最小改修としては、(a) ジョブ開始/終了をロガーで明示(#14相当)、(b) ステージ単位の進捗+it/s相当の速度表示(#12/#13を動画の2段パイプライン用に多段化)、(c) 完了時にVRAM peakと合計時間をコンソールへ直接出す(#17/#18をUIでなくログとして)、の3点が費用対効果が高い。

---

## 4. 出典URL一覧

- リポジトリ本体: https://github.com/Haoming02/sd-webui-forge-classic
- Neoブランチ: https://github.com/Haoming02/sd-webui-forge-classic/tree/neo
- README(neo): https://github.com/Haoming02/sd-webui-forge-classic/blob/neo/README.md
- ソース(直接取得・grep確認済み、neoブランチ):
  - `backend/memory_management.py` https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/backend/memory_management.py
  - `modules/sd_models.py` https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/modules/sd_models.py
  - `modules/sd_samplers_common.py` https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/modules/sd_samplers_common.py
  - `modules/shared_total_tqdm.py` https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/modules/shared_total_tqdm.py
  - `modules/shared_state.py` https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/modules/shared_state.py
  - `modules/call_queue.py` https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/modules/call_queue.py
  - `modules/timer.py` https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/modules/timer.py
  - `extensions-builtin/sd_forge_lora/networks.py` https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/extensions-builtin/sd_forge_lora/networks.py
- 実測コンソールログ例の引用元(GitHub Issues/Discussions、"Total progress"表示の実例確認):
  - lllyasviel/stable-diffusion-webui-forge の各種Issue/Discussion(検索経由で確認。個別URL特定は不可、"Total progress: 100%|...| 20/20 [00:02<00:00, 8.44it/s]" 形式のログ断片がユーザー投稿として複数存在)
- Forge Neoの系譜・素性に関する補足情報:
  - Grokipedia解説ページ: https://grokipedia.com/page/Stable_Diffusion_WebUI_Forge_Neo
  - インストール解説記事: https://www.stablediffusiontutorials.com/2025/11/forge-neo-installation.html
