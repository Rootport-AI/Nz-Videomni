# MiniMax H3: 枝刈り版と非枝刈り版の流通、検証用のファイル、Ampere での対応（調査 C）

- 調査日: 2026-11-12
- 方法: ウェブと一次資料を読んだだけ（読み取り専用）。GPU・git・パッケージ導入は使っていない。safetensors のヘッダは HTTP Range（ファイルの先頭部分だけを取る要求）で読んだ。CivitAI はダウンロードにログインが要るので、そのファイルは大きさだけで判断した。HuggingFace（以下 HF）のファイルはヘッダで確かめた。
- 読んだコードの版
  - ComfyUI 本体: `Comfy-Org/ComfyUI` の master（2026-11-12 時点の写し）
  - comfy-kitchen: `Comfy-Org/comfy-kitchen` の main（0.2.41。2026-11-12 時点の写し）
- スクリプトと結果は調査時の作業フォルダにあったもので、保管していない。
- 印の読み方: 【一次資料で確認】は一次資料で確かめたこと（以下 [P]）、【推測】は推し量ったこと（以下 [I]）。

用語:
- **量子化の印（`comfy_quant`）**: 量子化（重みの数値を少ないビット数で表して容量を減らすこと）した層ごとに、その層の形式を書いた小さなテンソル `<層の名前>.comfy_quant`。中身は JSON の文字列を uint8 の並びにしたもの。
- **旧式のメタデータ（`__metadata__._quantization_metadata`）**: 層ごとのテンソルの代わりに、safetensors のファイル先頭にある目録（ヘッダ）の `__metadata__` 欄へ、全層の形式をまとめて JSON で書く方式。
- **枝刈り版**: AdaLN（時刻に応じて各層の出力を調整する仕組み）の枝を、先に計算した表 `adaln_t_table` に畳み込んで小さくした版。非枝刈り版は、時刻の埋め込みの枝をそのまま持つ。
- **fp8**: 8 ビットの浮動小数点数。ComfyUI では `float8_e4m3fn`（指数 4 ビット）と `float8_e5m2`（指数 5 ビット）がある。
- **w4a8**: 重みを 4 ビット、計算時の入力（活性化）を 8 ビットで扱う方式。ComfyUI の形式名は `asym_w4a8_int8`。同様に w6a8 は重み 6 ビット、w4a4 は重みと活性化がともに 4 ビット。
- **nvfp4**: NVIDIA の 4 ビット浮動小数点の形式。
- **カーネル**: GPU の上で実際に計算を行う中核の処理（計算核）。
- **Ampere／Ada／Blackwell**: NVIDIA の GPU の世代。それぞれ RTX 30 系／RTX 40 系／RTX 50 系。本文の「SM」や「compute capability（計算能力）」は、GPU の世代を表す NVIDIA の版番号である（数字が大きいほど新しい）。

## 結論

枝刈り版がふつうの形である。非枝刈り版の量子化ファイルは少数派で、主に int8 と nvfp4、それに初期のものや Hybrid の合成がいくつかある。読み込み処理が扱わなければならない点は 2 つ。枝刈り版の一部は、すべてのキーの頭に `model.diffusion_model.` がつく。h3ErosMax は `adaln_t_table` を F32 でなく BF16 で持つ。

## 問 1 枝刈り版を始めたのは誰か

- [P] Kijai（Comfy-Org）の ComfyUI PR #15224「Support MiniMax-H3」（2026-08-03 に取り込み）: 「Supports both original time-embedder and our pruned (40% smaller) precomputed adaln-curve-table checkpoint variants.」（元の時刻の埋め込みと、私たちの枝刈り版〔40% 小さい、先に計算した AdaLN の曲線の表〕の両方の重みに対応する）。https://github.com/Comfy-Org/ComfyUI/pull/15224
- [P] `*_pruned_fp8_scaled` は 2026-08-03 に Comfy-Org へ上げられた。枝刈り版の bf16 は 08-04。https://huggingface.co/api/models/Comfy-Org/MiniMax-H3/commits/main
- [P] ComfyUI は `adaln_t_table` があるかどうかで枝刈り版を見分ける。ない場合は時刻の埋め込みを組み立てる。https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py
- [P] 品質: Comfy のブログは「functionally equivalent … no loss in output quality」（機能として同等で、出力の品質は落ちない）と書く（比較は示していない）。https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui
  - multimodalart のモデルカード（2026-08-09）: 畳み込んだ射影それぞれの誤差は約 1.7e-3 で、公開された重みよりわずかに正確な値に近い。同じシードで最後まで生成すると、枝刈りによる結果のずれは、注意機構（attention）のカーネルを替えたときと同程度だった（映像の潜在のコサイン類似度で 0.991。比較対象は 0.992）。https://huggingface.co/multimodalart/MiniMax-H3-Pruned

## 問 2 量子化ファイルの中の枝刈り版と非枝刈り版

### CivitAI（H3 の重みのページ 40 件を API で取得し、大きさで判断）[I]

https://civitai.com/api/v1/models?baseModels=MiniMax%20H3&types=Checkpoint

| 形式 | 枝刈り版 | 非枝刈り版 |
|---|---|---|
| int8 | 57 | 9 |
| fp8 | 7 | 0 |
| mxfp8 | 6 | 0 |
| w4a8 | 18 | 0 |
| w6a8 | 8 | 0 |
| nvfp4 | 6 | 0 |
| int4／w4a4 | 6 | 4 |

非枝刈り版の int8（約 34 GB。FeiHou には 49 GB のものもある）:

- CivitaiOfficial の fl2va／ref2va int8_convrot https://civitai.com/models/2821932?modelVersionId=3216500
- PinkCherry beta-0.6 https://civitai.com/models/2838593?modelVersionId=3221782
- FeiHou Remix v0.6／v0.8 https://civitai.com/models/2879272
- Felldude「unpruned」 https://civitai.com/models/2951893?modelVersionId=3344032
- XUELUO の 2 本 https://civitai.com/models/2864405

非枝刈り版の 4 ビット: MM744 の int4 convrot 17.9 GB https://civitai.com/models/2830162

大きさから見てすべて枝刈り版のもの: DaSiWa、h3ErosMax（beta2〜7。int8 も w4a8 もすべて）、PinkCherry v1.5／v1.8、joeygambino、Liant、Aki、RedCraft。

### HuggingFace（ヘッダで確認。置き直しも別に数えた）[P]

- int8: 枝刈り版 約 125、非枝刈り版 32。非枝刈り版の例は Comfy-Org、Gluttony10、OxBee、DeepBeepMeep、Singularity、PinkCherry の alpha 版、DmitryDB 22.5 GB。
- fp8: 枝刈り版 13、非枝刈り版 2。非枝刈り版の例は lightx2v の Turbo 34 GB、rzgar 47 GB。
- w4a8: 枝刈り版 26、非枝刈り版 2。非枝刈り版の例は tsolful/Minimax_H3_W4A8 25.6 GB。
- nvfp4: 枝刈り版 約 25、非枝刈り版 11。非枝刈り版の例は lilcheaty、ModelsLab、rockerBOO、cccian091、Simplismart。
- w6a8: 枝刈り版 8、非枝刈り版 0。

nvfp4 は大きさだけでは判断できない。lilcheaty の `nvfp4_full` 18.75 GB は非枝刈り版で、AdaLN を 4 ビットに詰めている（`adaln_proj` [96768,1344]）。多数派はどの形式でも枝刈り版。HF での非枝刈り版の割合は int8 で約 20%、nvfp4 で約 30%。fp8、w4a8、w6a8 では小さいかゼロ。

## 問 3 VRAM 12〜24 GB の GPU 向けの案内

- [P] Comfy の公式の雛形（テンプレート）は、すべて `*_pruned_int8_convrot` を読み込む。https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json
- [P] Comfy のブログは、RTX 3060 で動くのは枝刈り＋int8＋動的なオフロード（重みを主記憶へ逃がすこと。dynamic offloading）のおかげだとしている。https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui
- [P] Comfy-Org のモデルカード: int8_convrot を優先し、fp8_scaled は int8_convrot が使えないときだけ。https://huggingface.co/Comfy-Org/MiniMax-H3
- [I] 一般向けの GPU に非枝刈り版の量子化ファイルを勧める案内はない。実際上は枝刈り版が標準である。

## 問 4 検証用にダウンロードするファイル（どれも利用申請は不要）

### a. Comfy-Org の fp8 [P]

- `Comfy-Org/MiniMax-H3` の `diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors`
- 20,958,205,608 バイト、sha256 `12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993`
- ライセンス: MiniMax H3 Community License
- ヘッダ: `adaln_t_table` は F32 [1025,8]、`comfy_quant` が 200 本。

### b. Kijai の w4a8 [P]

- `Kijai/MiniMax-H3-experimental` の `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`（リポジトリの最上位）
- 12,540,858,008 バイト、sha256 `01aa7b92c007c599890461c325f9b7e3c96fb06c36f242f95b62f7f20e538dec`
- ライセンス: 表明なし
- FL2VA のファイルは `__metadata__._quantization_metadata` を持つ。200 層すべてが `asym_w4a8_int8`（group 16、convrot 256）。`comfy_quant` のテンソルはなく、`adaln_t_table` は F32。
- Ref2VA のファイル `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors`（11,770,657,048 バイト）は、ヘッダのメタデータを**持たない**。設定は層ごとの `comfy_quant` のテンソル 208 本にある（調査 B の訂正）。

### c. TenStrip の h3ErosMax [P]

- `TenStrip/10Eros-Max`（リポジトリの最上位）。ライセンスは MiniMax H3 Community License（モデルカードによると、LTX 2.3、Wan 2.2、Krea 2 のライセンスもかかる）。CivitAI の beta5 は TURBO のものを載せている。https://civitai.com/models/2851079?modelVersionId=3294059
- `10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors`
  - 13,997,668,774 バイト、sha256 `a8067999c65594b462d581c02b8a0573dd42d9812a14c586150a947c13388f2e`
- `10Eros_Max_h3_hybrid_beta5_w4a8_14gb_optimized.safetensors`（Turbo なし）
  - 13,997,668,758 バイト、sha256 `16249794bc0d4627a3960a6c9f32631bad267ce887c716f8e77c5356d2757ca2`
- 2 本とも枝刈り版。`adaln_t_table` は **BF16** [1025,8]。ほかに `adaln_basis` [8,2688] と `adaln_mean` [2688] がある。
- 層の内訳: `asym_w4a8_int8` が 180 層（group 16／8／4 がそれぞれ 82／53／45）と、行ごとの convrot の `int8_tensorwise` が 20 層（ブロック 0、1、47、48、49 の 4 つの線形層すべて）。設定はヘッダとテンソルの両方にある。

## 問 5 Ampere（RTX 30 系）での fp8 と w4a8

- [P] fp8 は Ampere ではそのまま動かない。ComfyUI の fp8 の判定は SM 8.9 以上を求める。古い GPU では float8 を「emulated ops」（模擬の演算）として扱い、計算時は元の精度に戻す（`get_disabled_quant_formats`、`supports_fp8_compute`）。https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ops.py
- [P] w4a8 は Ampere で使える。comfy-kitchen の `AsymW4A8Int8Layout` は `MIN_SM_VERSION = (8, 0)`。Windows 向けの CUDA の配布物（wheel）は sm 75-virtual、80、89、120f 向けにビルドされている。https://github.com/Comfy-Org/comfy-kitchen 。CUDA の経路には PyTorch の cu130 以上が要る。そうでなければ eager（PyTorch の素の計算）か Triton の経路になる。Kijai のモデルカード: 「Requires ComfyUI 0.31.0」（ComfyUI 0.31.0 が必要）。
- [I] Ampere に限った Comfy-Org や Kijai の言明はない。RTX 30 系での w4a8 の速度の計測もない。
