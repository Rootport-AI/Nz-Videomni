# I2 取得元の確定（MiniMax H3 第 3 弾 install-MiniMaxH3.bat 用）

調査日 2026-10-18。Hugging Face の公開 API（`/api/models/...`・`/tree`・`/paths-info`・`/commits`）と、ログイン無しの HEAD 要求、小さなテキストの取得だけで確かめた。重みはダウンロードしていない。取得した小さなファイル（LICENSE・README・木の JSON）は同じフォルダの `raw/` にある。

## ① 結論の表

| 取得元 | ファイル（リポジトリ内の正式なパス） | バイト数（HF） | 手元（`models/MiniMaxH3/` 下） | 一致 | 現在のリビジョン（main） | このファイルを最後に変えたコミット | ゲート |
|---|---|---|---|---|---|---|---|
| Comfy-Org/MiniMax-H3 | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `Weights/…` 20,970,379,616 | サイズ一致 | `e5eb578a89295337b8ff433a035929ce0279e0b6`（2026-09-29） | `3f57e8291d`（Initial commit, 2026-07-30） | 無し |
| Comfy-Org/MiniMax-H3 | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27,141,342,152 | `TextEncoder/…` 27,141,342,152 | サイズ一致 | 同上 | `3f57e8291d` | 無し |
| Comfy-Org/MiniMax-H3 | `vae/minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 | `VAE/…` 5,207,808,496 | サイズ一致 | 同上 | `3f57e8291d` | 無し |
| Comfy-Org/MiniMax-H3 | `vae/minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 | `VAE/…` 605,254,808 | サイズ一致＋sha256 一致（`8e505d95dd1561d4…`） | 同上 | `3f57e8291d` | 無し |
| MiniMaxAI/MiniMax-H3 | `FL2VA/text_encoder/config.json` | 1,474 | 1,474 | git の blob ハッシュ一致 | `42ed227ee7df40d41602854ae760620d6eb651fe`（2026-08-13） | `5d9b308a59`（2026-08-03） | 無し |
| 同 | `FL2VA/text_encoder/merges.txt` | 1,671,839 | 1,671,839 | blob 一致 | 同上 | `5d9b308a59` | 無し |
| 同 | `FL2VA/text_encoder/preprocessor_config.json` | 390 | 390 | blob 一致 | 同上 | `5d9b308a59` | 無し |
| 同 | `FL2VA/text_encoder/tokenizer.json` | 7,032,403 | 7,032,403 | blob 一致 | 同上 | `5d9b308a59` | 無し |
| 同 | `FL2VA/text_encoder/tokenizer_config.json` | 11,003 | 11,003 | blob 一致 | 同上 | `5d9b308a59` | 無し |
| 同 | `FL2VA/text_encoder/vocab.json` | 2,776,833 | 2,776,833 | blob 一致 | 同上 | `5d9b308a59` | 無し |

- tokenizer 6 ファイルの合計は 11,493,942 バイト。記述子の `_min` の注記（11,493,942B）と一致する。
- 記述子の `include` に書いた 10 個のパスは、すべて一字一句そのまま実在する。
- 2 つのリビジョンは `VERIFICATION_LOG.md` §165.4 の記録（`e5eb578…`・`42ed227…`）と同じ。第 1 弾以降、どちらのリポジトリも main が動いていない。

## ② ゲートとライセンスの事実

### ゲート（事実）
- `https://huggingface.co/api/models/Comfy-Org/MiniMax-H3` → `"gated": false`・`"private": false`・`"disabled": false`。
- `https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3` → `"gated": false`・`"private": false`・`"disabled": false`。
- ログイン無しの HEAD 要求: Comfy-Org の大きなファイルは 302（Xet の配信網へ転送。`x-linked-size` が上の表のバイト数と一致・`x-repo-commit` は `e5eb578…`）、MiniMaxAI の小さなファイルは 307（同じサイト内の resolve-cache へ転送・`x-repo-commit` は `42ed227…`）。401／403 は出ない。転送を追った取得は `config.json` が 200・1,474 バイト、音声 VAE の先頭 100 バイトの範囲要求が 206。
- よって**トークン無しの `hf download` で両方とも取れる見込み**（実際の `hf download` は走らせていない。⑤）。インストーラの既存の文言「The repo is public and needs no token」はこの 2 リポジトリにもそのまま当てはまる。

### ライセンス（事実）
- 本文: `MiniMaxAI/MiniMax-H3` のルートの `LICENSE`（17,604 バイト・blob `c389b4585533`）。URL `https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE`。補足の Q&A は `docs/QA-about-License.md`（`https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md`）。
- 名称: **MiniMax H3 COMMUNITY LICENSE AGREEMENT**（HF のメタデータでは `license: other`・`license_name: minimax-h3-community-license-agreement`）。公開日／ライセンス日 2026-08-02。許諾者は Nanonoble Pte. Ltd.（本文でいう「MiniMax」）。準拠法と管轄は香港。
- 地域条項: 許諾は「Applicable Territory」＝**全世界から Excluded Territories を除いた地域**に限られる。Excluded Territories は **EU・英国・韓国・米国**（I.3／I.5）。地域外での使用・複製・改変・配布・表示は許諾されない（V.4）。Exhibit A（利用規定）の 1 番目も「地域外での使用」。地域外の組織は申請フォーム `https://platform.minimax.io/h3-license` から個別の許諾を求められる（Q&A・README）。日本は定義上 Applicable Territory に入る。
  - 注意: Q&A の見出しは「limited to the EU, UK, South Korea, and US」と書いているが、本文の定義ではこの 4 地域は**除外**される側。正本は LICENSE 本文。
- 量子化派生物: 「Model Derivatives」は MiniMax H3 の改変や、それに基づく著作物を含む（I.11）。「MiniMax H3 Works」は素材・派生物・その派生物のすべて（I.7）。int8 ConvRot 版のような量子化物は「改変」に当たると読める（明文で「量子化」とは書いていないので、この当てはめは推測）。
- 第三者への配布の条件（III）:
  1. 受け取る第三者全員に**本ライセンスの写し**を渡す（III.1）。
  2. 改変したファイルには、改変した旨を目立つように表示する（III.2）。
  3. 推奨（義務ではない）: 「Powered by MiniMax H3」の表示、生成物への AI 生成の識別子、体験を書いた公開記事（III.3）。
  4. Hosted Services 以外の配布には、次の文を書いた **`NOTICE` テキストファイル**を添える（III.4）: `MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.`
  - 統合された最終製品の一部として受け取った者には III が適用されない。ただし V と Exhibit A は適用される（III の末尾）。
- その他の要点: 年商 2,000 万米ドル超の商用は事前の書面許諾が要る（IV.1）。商用の製品・サービスでは UI に「MiniMax H3」を目立つように表示する（IV.2）。他の AI モデルの改善に使ってはならない（V.3）。利用者には V と Exhibit A 以上の制限を課して通知する（V.2）。第三者に生成機能を提供するなら、防護措置と違反報告の窓口を維持する（V.5）。利用規定は全 20 項で、軍事目的の禁止（19）や公開環境での機械生成の明示（12）を含む。テキストエンコーダは Qwen3-VL-32B で Apache 2.0（本文末尾の Additional Note）。
- Comfy-Org 側: README の先頭のメタデータが `license: other`・`license_name: minimax-h3-community-license-agreement`・`license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE`・`base_model: MiniMaxAI/MiniMax-H3`。本文は「Repackaged model files for ComfyUI.」と元リポジトリへのリンクだけ。**ライセンス本文の写しも NOTICE ファイルも置いていない**（ファイルは `.gitattributes`・`README.md` と重みだけ）。

### README に書くべき要点（推測。法的な判断はオーナー）
- ライセンスの名称と本文の URL。利用は EU・英国・韓国・米国を除く地域に限られること。利用規定（Exhibit A）に従うこと。商用なら UI に「MiniMax H3」の表示が要ること。テキストエンコーダは Qwen3-VL-32B（Apache 2.0）であること。
- 本製品は重みを再ホストせず（研究ノートの裁定）、利用者の機械が HF から直接取得する。この形で本製品が III の「配布」をしていることになるかどうかは解釈の問題で、未確認。安全側に倒すなら、インストーラで `LICENSE` も一緒に取得して `models/MiniMaxH3/` に置き、NOTICE の文を README に載せる案がある（③の 5）。

## ③ 記述子 `40-minimaxh3.json` との差（直すべき点）

1. **パス・ファイル名・サイズに直す点は無い**。`include` の 10 個は実在する。`files[].path` は `map` の写し先と整合する。`min` は実物の約 4% 下で妥当。`_note` の「Comfy-Org/MiniMax-H3 のゲートの有無とファイル名は未検証」は**検証済みに書き換えてよい**（ゲート無し・ファイル名一致）。
2. **リビジョンを固定する欄が無い**。記述子の書式にも、`install_ltx.ps1` の実行部（`$argv = @("download", $dl.repo)` に `--include` と `--local-dir` を足すだけ）にも `revision` は無い（`scripts/*.ps1` と `scripts/manifests/*.json` を grep して 0 件）。固定したいなら、`downloads[]` に `"revision": "<コミット>"` を足し、実行部で `--revision` を渡す改修が要る。LTX の既存項目も固定していないので、固定するかどうかは設計判断。固定するなら値は Comfy-Org `e5eb578a89295337b8ff433a035929ce0279e0b6`、MiniMaxAI `42ed227ee7df40d41602854ae760620d6eb651fe`。
3. **tokenizer の `include` は、ファイル名を 1 つずつ並べたままにする必要がある**（事実に基づく注意）。`install_ltx.ps1` の注記どおり `--include` は fnmatch で、`*` が `/` をまたぐ。`FL2VA/text_encoder/*` と書くと、bf16 のテキストエンコーダの 14 分割（約 65GB）まで落ちてくる。今の 6 個の列挙は正しい。
4. **`map` は今のままで動く**（コードを読んで確かめた事実）。`Resolve-MapTarget` は最も長く一致する接頭辞を置き換えるので、`FL2VA/text_encoder/config.json` → `MiniMaxH3/TextEncoder/qwen3vl_tokenizer/config.json`、`diffusion_models/…` → `MiniMaxH3/Weights/…` になる。`Test-UnderPrefix` は完全一致も受け付けるので、ファイル単位の `map`（例 `{"from":"LICENSE","to":"MiniMaxH3/LICENSE"}`）も書ける。
5. （任意）ライセンスの写しを置くなら、2 番目の項目の `include` に `LICENSE` を、`map` に `{ "from": "LICENSE", "to": "MiniMaxH3/LICENSE" }` を足す。ただし `files[]` に検査の行を足さないと、`LICENSE` だけが欠けても再取得されない。逆に足すと、導入済みの人の再実行でその項目全体（tokenizer）を取り直す。AlphaGen の LoRA を独立した項目にした前例（`20-ltx25.json` の `_note`）と同じ論点。採るかどうかはオーナー判断。
6. tokenizer の代わりの取得元: 同じ 6 ファイルは、MiniMaxAI リポジトリのルートの `text_encoder/` と `Ref2VA/text_encoder/` にも同一の blob で存在する。`processor/`・`FL2VA/processor/` には `config.json` が無い（他の 5 個は同一の blob）。`tokenizer/` 系は 4 個だけ。**Comfy-Org 側には tokenizer のファイルが 1 つも無い**。どちらもゲート無しなので、ゲートを理由に取得元を変える必要は無い。

## ④ `hf download` の使い方の事実

- 本製品の `.venv-engine` の版: `huggingface_hub==0.36.2`・`hf-xet==1.6.0`（`engine/venv-engine.freeze.txt`。`hf version` でも 0.36.2）。Comfy-Org の重みは Xet ストレージ（HEAD の転送先が `xet-bridge-us`）なので、hf-xet が入っていることが前提になる（すでに固定済み）。
- `hf download --help`（0.36.2）の引数: `repo_id [filenames ...]`・`--revision REVISION`（ブランチ名・タグ・コミットのハッシュ）・`--include [INCLUDE ...]`・`--exclude`・`--local-dir`・`--token`・`--max-workers`（既定 8）など。**`--include` でサブフォルダのファイルだけを取れ、`--revision` でコミットに固定できる**（ヘルプの記述という事実。この 2 リポジトリでの実走は未確認）。
- `install_ltx.ps1` の既知の落とし穴（コードの注記・§112）: `--include` は nargs="*" なので、1 個の `--include` の後に全パターンを並べる（繰り返すと最後の組だけが残る）。1.x ではこの形が黙って無視されるので、版を上げないこと。fnmatch では `*` が `/` をまたぐ。実行は `hf.exe` ではなく `& $enginePy -P -m huggingface_hub.cli.hf`。
- 取得先の構造: `--local-dir $stage` の下に、リポジトリ内のパスのまま置かれる（例 `$stage/FL2VA/text_encoder/tokenizer.json`、`$stage/diffusion_models/…`）。`$stage/.cache/` は再開用の記録で、写し替えの対象外。その後 `map` の最長一致で `models/` の下へ移す。`kind:"dir"` の検査は、写し先フォルダ直下のファイルの合計で行う（`Get-DirectChildSize`）。
- 前例: `20-ltx25.json` は `include` にフォルダの glob（`Weights/*` など）、`map` にフォルダの対応を書く。`10-ltx23.json` の `gemma_root` は `files` に `"kind": "dir"`・`min` 39,000,000。`40-minimaxh3.json` はこの両方の書式に沿っている。
- 現状、`scripts/install_ltx.ps1`・`scripts/install_model.ps1` に MiniMaxH3 への言及は無く、`install-MiniMaxH3.bat` もまだ無い（リポジトリのルートにあるのは `install-LTX25.bat`・`install-UETrack.bat` だけ）。

## ⑤ 未確認

- この 2 リポジトリで実際に `hf download --include … --revision …` を走らせた結果（重みを落とさない制約のため未実施。小さな tokenizer 6 個だけなら試せる）。
- DiT・テキストエンコーダ・映像 VAE の手元の実物の sha256（サイズは一致。音声 VAE だけ sha256 を計算し、HF の LFS の oid と一致することを確かめた）。HF 側の sha256（LFS の oid）は、DiT `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a`、TE `bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923`、映像 VAE `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`、音声 VAE `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48`。
- 利用者の機械が HF から直接取る形で、本製品が LICENSE の III（写しと NOTICE の義務）や V.2／V.5 の義務を負うかどうか（法的な解釈）。
- int8 ConvRot の量子化が III.2 の「改変したファイルへの表示」の対象になるか。Comfy-Org がその表示をしているか（ファイルのメタデータは読んでいない）。
