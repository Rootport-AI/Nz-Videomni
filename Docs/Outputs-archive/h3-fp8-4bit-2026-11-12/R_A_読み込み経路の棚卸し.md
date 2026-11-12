# MiniMax H3: 量子化した重みの読み込み経路の棚卸し（調査 A）

- 調査日: 2026-11-12
- 方法: コードと文書を読み、重みファイルはヘッダ（ファイル先頭の目録部分）だけを読んだ（読み取り専用）。GPU・git・パッケージ導入は使っていない。
- 読んだコードの版（ここに書く行番号はこの版のもの）
  - 本製品のバックエンド: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni` の dev ブランチ `abc8eea`（2026-11-12）
- 印の読み方: 原文の「observed（観測した事実）」は印なしで書き、原文の「Inference」は【推測】と書く。
- ヘッダの生の書き出し（`survey.py`・`survey_out.txt`・`qmeta.py`・`qmeta_out.txt`）は、調査時の作業フォルダにあったもので、保管していない。

用語:
- **量子化の印（`comfy_quant`）**: 量子化（重みの数値を少ないビット数で表して容量を減らすこと）した層ごとに、その層の形式を書いた小さなテンソル `<層の名前>.comfy_quant`。中身は JSON の文字列を uint8 の並びにしたもの。
- **旧式のメタデータ（`__metadata__._quantization_metadata`）**: 層ごとのテンソルの代わりに、safetensors のファイル先頭にある目録（ヘッダ）の `__metadata__` 欄へ、全層の形式をまとめて JSON で書く方式。
- **枝刈り版**: AdaLN（時刻に応じて各層の出力を調整する仕組み）の枝を、先に計算した表 `adaln_t_table` に畳み込んで小さくした版。非枝刈り版は、時刻の埋め込みの枝をそのまま持つ。
- **fp8**: 8 ビットの浮動小数点数。ComfyUI では `float8_e4m3fn`（指数 4 ビット）と `float8_e5m2`（指数 5 ビット）がある。
- **w4a8**: 重みを 4 ビット、計算時の入力（活性化）を 8 ビットで扱う方式。ComfyUI の形式名は `asym_w4a8_int8`。同様に w6a8 は重み 6 ビット、w4a4 は重みと活性化がともに 4 ビット。
- **nvfp4**: NVIDIA の 4 ビット浮動小数点の形式。
- **カーネル**: GPU の上で実際に計算を行う中核の処理（計算核）。
- **Ampere／Ada／Blackwell**: NVIDIA の GPU の世代。それぞれ RTX 30 系／RTX 40 系／RTX 50 系。本文の「SM」や「compute capability（計算能力）」は、GPU の世代を表す NVIDIA の版番号である（数字が大きいほど新しい）。

## (a) Eros の beta5 の 2 本が拒否される理由（観測した事実）

`h3ErosMax_beta5_3178732_int8_TURBO_Hybrid` と `h3ErosMax_beta5_3185154` は、どちらも `h3_format.py:269-274` で止まる。表示される文言は次のとおり。

> MiniMax H3 のファイルとして使えません: int8 の層 'blocks.0.attn.out_proj' に量子化の印（comfy_quant）がありません（旧式の __metadata__._quantization_metadata には対応していません）

- ヘッダの中身: I8（8 ビット整数）の重みが 200 本あり、それぞれに正しい F32 `[o,1]` の `weight_scale`（倍率）がついている。**`.comfy_quant` のテンソルは 1 本もない。** 量子化の印は `__metadata__._quantization_metadata` にある（書いたのは「Star Ultimate Model Converter」）。項目は 200 件で、すべて `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`。
- 読み込める beta4 との違い: テンソルの名前は同じで、200 本の `comfy_quant` がないだけ。ほかに 10 本のテンソルが BF16 から F32 に変わっているが、害はない（`normalize_dtypes`、`engine_h3/dit.py:474-511`）。どちらも枝刈り版で、`adaln_t_table` は [1025,8]、名前の接頭辞はなし（""）、ブロックは 50 個。`adaln_basis` と `adaln_mean` は未知のテンソルとして捨てられる。
- 拒否は意図したもの: `Docs/MINIMAX_H3_DESIGN.md` §7（1378 行）、§12（1538 行）。研究ノート `MINIMAX_H3_RESEARCH_NOTES.md:490`、ComfyUI の `convert_old_quants`（546 行）。
- LTX 側はメタデータ式の印を受け付ける: `sft_quant_format._resolve_markers`（587-609 行）は、まずメタデータを見て、次に `comfy_quant` を見る。【推測】これを `h3_format._quant_layers` で使い回せば足りる。エンジンは印のテンソルを一度も読まず、`load_streaming` は `.comfy_quant` を読み飛ばす。
- オーナーには理由が見えない: 事前検査 `services/model_registry.py:206-216` → `api/errors.py:341`（422 応答）は理由を `error.detail` に入れるが、`main.py:404-406` はそれをログに書かない。Settings の欄 `webui/src/shell/ModelsPanel.tsx:110-113` は定型の案内文を出し、`useModels.ts:175-177` は `detail` を捨てる。ヘッダーのベースモデルの選択欄 `useBaseModels.ts:229` は `detail` を表示する。

ほかの 5 本は通る（beta4、公式の 2 本、PinkCherry v18 Hybrid、PulpCut turbo）。7 本すべてが int8_tensorwise＋ConvRot 256 で、量子化されたブロックの線形層（Linear）が 200 本あり、LoRA のキーは含まない。`weight_identity.json` がハッシュを記憶しているのは公式の 2 本だけで、事前検査には記憶の仕組みがない。

## 現行の H3 の経路と fp8／4 ビット（観測した事実）

- `ACCEPTED_FORMATS = ("int8_tensorwise",)`（`h3_format.py:42`）。量子化された層として数えるのは I8 の `.weight` だけ（245-247 行）。
- I8 でない層に `weight_scale` や `comfy_quant` があると拒否する（248-259 行）→ fp8_scaled はここで引っかかる。
- 詰め込む前の形どおりかを厳密に確かめる検査（350-379 行）が先に走る → 4 ビットを詰め込んだ重みは、形の矛盾として落ちる（【推測】）。
- 【推測】倍率も印もない fp8 のファイルなら通り、`normalize_dtypes` で bf16 に広げられる。
- `H3Int8Linear`（`engine_h3/quant_linear.py:143-319`）: int8 の生の Parameter と F32 [N,1] のバッファを持ち、`comfy_kitchen.int8_linear` を呼ぶ。`require_cuda_int8` が triton と eager（PyTorch の素の計算）の経路を無効にする。`QuantizedTensor` は使わない（block swap の置き場は、ふつうのテンソルを uint8 として詰める）。engine_h3 には、bf16 に戻して計算する経路がない。
- VRAM 16GB mode（`lowvram.py` の 146-152 行あたり）は `qkv.shares_input`／`forward_rows`／`forward_rows_q` を呼ぶ。新しい Linear のクラスは `forward_rows` を備えなければならない。
- LoRA（`lora.py`）は出力の側に足す。`attach` はどの Linear でも受け付ける（193 行）→ そのまま引き継げる（【推測】）。

## (c) LTX 側の fp8（観測した事実）

- `sft_quant_format.SCHEME_TABLE`（132-138 行）: `fp8`、`fp8_scaled`（F32 の倍率 1 つ）、`int8`、`int8_convrot`、`w4a8`。形式は float8_e4m3fn と e5m2 を受け付ける。fp8 の印はなくてもよく、メタデータ式の印も受け付ける。
- 実行の仕方 `engine/sft_quant/quant_service.py:274-293` と `dequant.py:83-136`: 重みは CPU 上で fp8／int8 のまま持ち、呼ばれるたびに活性化の型へ戻して（逆量子化して）、ふつうの `F.linear` で計算する。comfy_kitchen も `_scaled_mm` も使わない。engine25 も同じものを使う。
- 規則の正本は VERIFICATION_LOG §121.3（13782 行）。nvfp4、mxfp8、convrot_w4a4 はそこで拒否する。

## (b) `.venv-engine-h3` の comfy_kitchen 0.2.37

torch は 2.13.0+cu130。triton-windows 3.7.1 は入っているが、読み込み時に切ってある。

形式（レイアウト）ごとに必要な最低の compute capability:

| 形式 | 最低の compute capability |
|---|---|
| TensorWiseINT8 | 7.5 |
| TensorCoreFP8（E4M3、テンソル全体で倍率 1 つ。torch の `_scaled_mm`、使えないときは戻して計算） | 8.9（Ada） |
| MXFP8 | 10.0 |
| NVFP4 | 10.0 |
| AsymW4A8Int8（W6A8 も） | 8.0 |
| ConvRotW4A4 | 7.5 |
| SVDQuantW4A4 | 8.0 |
| AWQ W4A16 | eager のみ |

- 関数として呼べる `fp8_linear` はない。fp8 の行列積は `QuantizedTensor` の包みを通してしか使えない。`int8_linear` は CUDA でしか登録されず、それも cuBLASLt（NVIDIA の行列積ライブラリ）があるときだけ。

## (d) 文書に残る過去の判断

- 裁定 6: int8 ConvRot（コミュニティの主流）。裁定 8: nvfp4／bf16 のテキストエンコーダは「将来」。
- 研究ノート 121-130 行: FP8 は Ampere で使えない → 「FP8とNVFP4は選ばない」（Ada の開発機と Ampere の予備機の両方で動く必要がある）。
- α 版では fp8、w6a8、int4、nvfp4、混合精度を後回しにした（研究ノート 600-608 行）。§14.6（770 行）では、高速化の手段として FP8／NVFP4 を対象外にした。
- Comfy-Org は `*_pruned_fp8_scaled` と `*_pruned_w6a8` も配っている（813-821 行）。調査時点の CivitAI の上位 30 件: int8 が 22、fp8 が 6、int4 が 5、nvfp4 が 4（483 行）。
- 台帳: H3 の fp8／4 ビット／nvfp4／`_quantization_metadata` の項目はない。近いのは §3-246 と §4-43。LTX の int8／w4a8 の作業は CLOSED の §3-168／169。
