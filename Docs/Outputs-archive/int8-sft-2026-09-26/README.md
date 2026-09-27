> **このフォルダは、§3-168（ComfyUI 標準の int8 safetensors を置くだけで使えるようにするテーマ）の実機検証で使った道具と結果一式のスナップショットである（2026-09-27 複写）。** 原本はセッションの scratchpad（`...\scratchpad\int8\`。git 追跡外・セッション終了後は消える一時領域）にあり、正本の結論は [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §121（§121.4・§121.6〜§121.8 に実機の門 G3〜G8 の結果表）と台帳 [`../../PENDING_TASKS_CLOSED.md`](../../PENDING_TASKS_CLOSED.md) §3-168 に転記済みである。本フォルダは、その転記の元になったツールと生データを git 追跡下に残し、読者が結論の裏取りや道具の再利用をできるようにするために複写した。**mp4 本体は複写していない**（結果の数値・ジョブ ID・ハッシュ先頭 8 桁だけで足りるため）。

# §3-168 int8／w4a8 safetensors 実機検証ツール一式（2026-09-27 実施分）

## 何がここにあるか

| ファイル | 内容 |
|---|---|
| `runner.py` | MCP（Model Context Protocol）経由で実機を叩く実機ランナー本体。副命令: `status`／`models`／`load`／`restore`／`single`／`iclora`／`chain`／`keep2`／`bad-header`／`psnr-ssim`／`save`／`record`／`report`。使い方の詳細は `README_runner.md` を参照。 |
| `gates.py` | `runner.py` の副命令を正しい順序（`load`→…→`restore`）で呼ぶ薄いオーケストレーター。G3〜G8（計画 `floofy-enchanting-sundae.md` 第4節）を変換器5本ぶん一気に回す。`--dry-run`／`--only`／`--stage`／`--skip`／`--resume` に対応。 |
| `g1_census.py` | G1 (ii)。safetensors のヘッダと量子化の印だけを seek+read で読み、層ごとの方式判定を集計する（mmap・全読み禁止）。 |
| `g1_equiv.py` | G1 (i)(iii)。変換ツール `Nz-GGUF-Converter-LTX23/src/converter/comfy_dequant.py`（NumPy 実装）と製品側 `engine.sft_quant.dequant`（torch 実装）を、乱数テンソルと実ファイルの層で突き合わせる。engine 側の venv（torch 入り）で実行する。 |
| `README_runner.md` | 上記4本の使い方・落とし穴・動作確認記録・2026-09-27 の改修点（`wait_for_job` の耐性・`record` 副命令新設）の一次記録。 |
| `RESULTS.md` | `runner.py report` が `results/*.json` を集計して書いた最終結果表（ロード・生成・keep_resident 同一性・不合格ファイル・PSNR／SSIM）。VERIFICATION_LOG §121.4・§121.6〜§121.8 の実機結果表は、すべてこの表から転記した。 |
| `c4_procedure_v2.md` | C-4（重みの種別ごとの快適上限の較正）実施手順 v2 の一次記録。正本は VERIFICATION_LOG §121.11。 |
| `baseline/stream_md5.json` | G6（fp8 回帰）の基準側（C-0 の3本の mp4）の映像／音声ストリーム MD5。`gates.py --baseline-md5` で事前計算したもの。 |

## 再利用の仕方

- **同種の実機検証をもう一度回すとき**は `runner.py`／`gates.py` をそのまま複製して使える。対象の変換器・基準 GGUF・手順は `gates.py` 冒頭の `CONVERTERS` 定義（README_runner.md の「対象の変換器5本と基準」表と対応）を差し替えるだけで良い。
- **判定規則を読むとき**は `g1_census.py`・`g1_equiv.py` が、製品側 `sft_quant_format.py`／`engine/sft_quant/dequant.py` の実装をどう検証したかの実例になる。
- **keep_resident・fp8 回帰の合否判定は、ファイル全体の SHA-256 ではなく映像／音声のストリーム MD5 で行う**（`runner.py` の `_stream_md5()`）。理由は README_runner.md の「2026-09-27 の修正3点」節、および VERIFICATION_LOG §121.10 の申し送りを参照。mp4 コンテナのメタデータ（§3-164 で埋め込む job_id 等）は 2 本の生成で異なるため、ファイル全体のハッシュでは同一の生成結果でも不一致になる。
- **`single`／`iclora`／`chain` は transformer を切り替えない**（`load` が最後に選んだものへそのまま投げる）。切り替えを伴うときは必ず直前に `load` を呼ぶこと（README_runner.md 冒頭の「重要」節）。

## 対応関係

- 実機の門 G3〜G8 の結果と判定: [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §121.4（総括・G6/G7/G8）・§121.6（C-1＝(a)Kijai・(b)silveroxides）・§121.7（C-2＝(d)LTX 2.5 公式複製）・§121.8（C-3＝(c)JoaoZaokk w4a8・(e)REDGraft）
- 台帳の状態: [`../../PENDING_TASKS_CLOSED.md`](../../PENDING_TASKS_CLOSED.md) §3-168（C-4 は [`../../PENDING_TASKS.md`](../../PENDING_TASKS.md) §1-32）
- C-4（較正）の手順の正本: [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §121.11
