# 利用者に見える文言の直し案（オーナー判断待ち・未適用。2026-10-18）

MiniMax H3 第 2 弾のあとのコード内コメントの現行化（記録は `Docs/VERIFICATION_LOG.md` §166.10）で見つかった、利用者や AI エージェントに見える文言の直し案である。コメントと違って外に出る文なので、直し担当は触らず、オーナーの判断を待つ。行番号は 2026-10-18 の dev（`cc38a67`）の作業ツリーで数えた。直すときは現在の文で探すこと。

## Python（例外文 1・MCP 向けの文 4）

| 番号 | ファイル | 現在の文 | 案 | 理由 |
|---|---|---|---|---|
| P-070 | `engine_h3/dit.py`（392 行付近。例外文で、ジョブの error に出る） | 「MiniMax H3: 枝刈り版（adaln_t_table を持つ）の DiT だけを扱います。非枝刈り版は第 1 弾の対象外です」 | 「MiniMax H3: 枝刈り版（adaln_t_table を持つ）の DiT だけを扱います。非枝刈り版には対応していません」 | 「第 1 弾の対象外」は作業の段階の言い方で、第 2 弾でも非枝刈り版は扱わない（`h3_format` の事前検査でも拒否する） |
| P-071 | `mcp_server/server.py`（54〜56 行付近。`INSTRUCTIONS`） | 「keep_resident（モデル骨格の常駐）は LTX 2.5 でも使えます（既定 off。ただし LTX 2.3 とは常駐する中身が違い、2.5 が抱えるのはテキストエンコーダの重みだけです。常駐ぶんのメモリを多く使います）。」 | 「keep_resident（モデル骨格の常駐）は LTX 2.5 と MiniMax H3 でも使えます（既定 off。ただし LTX 2.3 とは常駐する中身が違い、2.5 が抱えるのはテキストエンコーダの重みだけ、H3 が抱えるのは DiT・映像の復号器・音声 VAE です〔すべて CPU 上〕。常駐ぶんのメモリを多く使います）。」 | H3 も `keep_resident` を受ける（アダプタの `HONOURED_FIELDS`・設計正本 §4）。今の文は LTX 2.5 だけを挙げる |
| P-072 | `mcp_server/tools/generate.py`（313〜315 行付近。`submit_generate` の `num_frames` の説明） | 「LTX は 8n+1、MiniMax H3 は 17n+5 かつ 22 以上・32 の倍数・24fps。」 | 「LTX は 8n+1、MiniMax H3 は 17n+5 かつ 22 以上。」 | 「32 の倍数・24fps」はフレーム数ではなく幅・高さと fps の規約で、同じ説明の上の段に正しく書いてある |
| P-073 | 同（321〜323 行付近。`conditioning_images` の説明） | 「I2V用のキーフレーム画像（最大10件＝`config.py` の `MAX_CONDITIONING_IMAGES`、`upload_image` で得た `image_id` を使う）。」 | 末尾に「MiniMax H3 は `frame_idx=0` の 1 件だけで、`strength` は既定のまま（それ以外は 422 FEATURE_UNSUPPORTED）。」を足す | H3 の制限は `params.py` の `ConditioningImageArg` の説明にはあるが、ツールの引数の説明に無い |
| P-074 | 同（365〜370 行付近。`keep_resident` の説明） | LTX 2.3 と LTX 2.5 の常駐の中身だけを説明している | 末尾に「MiniMax H3 でも使え、常駐するのは DiT・映像の復号器・音声 VAE（すべて CPU 上。GPU には残しません）です。併用の制限も自動降格も無く、エコーは `"on"` / `"off"` の2値です（実測は `Docs/VERIFICATION_LOG.md` §166）。」を足す | H3 の常駐の中身が書かれていない（`pipeline_h3` は `keep_resident_used` を `"on"`／`"off"` で返す） |

補足（案は書いていない）: `mcp_server/server.py` の `INSTRUCTIONS` の `attention_backend` の段は LTX 2.5 だけを挙げるが、H3 もジョブごとに `attention_backend` を受け、SageAttention が無ければ SDPA へ落とす（`engine_h3/worker.py` の `_resolve_attention`）。必要なら「LTX 2.5 と MiniMax H3 で使えます」に。

## 操作パネル（webui）の文言 2

| 番号 | ファイル | 現在の文 | 案 | 理由 |
|---|---|---|---|---|
| U-1 | `webui/src/i18n/strings.ts` の `single.reservedNote`（英・日） | 英 "CFG is not supported by the current backend (reserved for future use)"／日「CFGは現在のバックエンドでは未対応です（将来の使用のために予約されています）」 | 英 "The selected base model does not use CFG (reserved for future use)"／日「選択中のベースモデルはCFGを使いません（将来の使用のために予約されています）」 | この欄はいま、サンプリングの 3 本を持たないベースモデル（LTX）のときだけ出る。「現在のバックエンド」はベースモデルを切り替える今の画面の語彙と合わない。優先度 低 |
| U-2 | 同 `single.unavailableOnBaseModel.keyframesMulti`（英・日） | 「選択中のベースモデルが使うキーフレームは先頭の1枚だけです」 | （コードの疑い C-1 を直さない場合）「先頭フレーム（0 フレーム目）に置いた 1 枚だけです」と位置まで言う | 制限されるのは枚数だけで、その 1 枚は先頭以外へ動かせる（動かすと送信時に 422。`VERIFICATION_LOG.md` §166.10 の webui の C-1）。C-1 と組で判断する。優先度 低 |

## オーナー判断（2026-10-18）

- **P-071: 適用済み**（オーナーの文面に、常駐に対応している LTX 2.3 の一句を監督が足した形。`mcp_server/server.py` の `INSTRUCTIONS`）: 「keep_resident（モデル骨格の常駐）が使えるのは LTX 2.3・LTX 2.5・MiniMax H3 です（既定 off。ただし、それぞれ常駐する中身が違い、2.3 が抱えるのは全部品の骨格、2.5 が抱えるのはテキストエンコーダの重みだけ、H3 が抱えるのは DiT・映像の復号器・音声 VAE です〔すべて CPU 上〕。常駐ぶんのメモリを多く使います）。」
- **U-2: 現状のまま**（案は説明が詳細すぎる、とのオーナー判断）。
- P-070・P-072〜P-074・U-1: 適用済み（案のとおり。オーナー了承）。
