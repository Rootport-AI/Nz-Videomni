> **このファイルは実機の `outputs/diffvae-research-2026-09-01/RESULTS.md`（git追跡外）のスナップショットである（2026-09-02複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# DiffVAE調査 一次記録（2026-09-01）

本書は2026-09-01に実施した調査の一次記録である。**採否の判断・評価は含まない**（採否の判断は台帳
[`PENDING_TASKS.md`](../../Docs/PENDING_TASKS.md) §3-103で扱う）。

DiffVAE（Diffusion VAE：LTX 2.5の映像VAE〈映像の圧縮展開器〉のうち、反復的なノイズ除去を伴う拡散
デコーダ版。既定の畳み込みデコーダ版〈Conv VAE〉と対をなす）について、実機での動作可否・所要時間・
VRAM消費・決定性（同じシードなら同じ結果になる性質）の機序を調べた結果を記す。

---

## 1. 測定条件

- **実施日**: 2026-09-01。
- **実機**: NVIDIA GeForce RTX 4070 Ti SUPER（16GB）。Windows。GPUの他負荷なし（計測対象のプロセス
  以外にGPUを使うプロセスを起動していない状態）。
- **ソフトウェア**: `ltx_core` 1.2.0（`.venv-engine-ltx25` 仮想環境）。同環境の `torch` は
  2.9.1+cu128（`python -c "import torch; print(torch.__version__)"` で確認）。
- **重み**: `models/LTX25/VAE/diffvae/ltx-2.5-video-vae-bf16.safetensors`。
- **潜在の生成方法**: `torch.randn(generator=...)` によるランダム潜在を使用した（実際の生成潜在
  ではない）。**したがって本記録は動作可否・所要時間・VRAM消費・決定性のみを対象とし、画質は一切
  評価していない。**
- **計測スクリプト**（本フォルダに複製、§9参照）:
  - `diffvae_smoke.py` — ランダム潜在を実際にDiffVAEでデコードし、所要時間とピークVRAM確保量を
    測る読み取り専用スクリプト（製品コード・設定には一切書き込まない）。
  - `diffvae_probe.py` — GPUを使わず、公式のタイル推奨器（`recommended_decode_tiling_config`）へ
    このプロジェクトの目標解像度・尺と申告空きVRAMを渡し、推奨タイル寸法または算出拒否
    （`ValueError`）を照会するスクリプト。

---

## 2. 実測表

| 解像度・尺 | 申告空きVRAM | 推奨タイル(frames/height/width) | 結果 | 所要 | ピーク確保 |
|---|---|---|---|---|---|
| 768×512/41f | 14.73GiB | 80f/512/768 | 成功 | 6.0秒（0.15秒/f） | 11.79GiB |
| 1280×768/49f | 14.73GiB | 80f/768/736 | 成功（ピーク確保が物理VRAM 16GBを超過） | 250.7秒（5.12秒/f） | 34.59GiB |
| 1280×768/121f | 14.73GiB | 128f/768/544 | メモリ不足で失敗 | 562秒で失敗 | — |
| 1280×768/121f | 6GiB申告（最小級タイルを強制） | 88f/480/320 | メモリ不足で失敗 | 649秒で失敗 | — |
| 1280×768/121f | 4GiB申告 | 推奨器が算出を拒否（`ValueError`） | — | — | — |
| 1920×1088/121f | 14.73GiB | 128f/640/608 | メモリ不足で失敗 | 360秒で失敗 | — |
| 1920×1088/121f | 8GiB申告（最小級タイルを強制） | 88f/352/384 | メモリ不足で失敗 | 550秒で失敗 | — |
| 1920×1088/121f | 5GiB申告 | 推奨器が算出を拒否（`ValueError`） | — | — | — |

「申告空きVRAM」は `diffvae_probe.py`/`diffvae_smoke.py` が推奨器へ渡す `free_bytes` 引数の値
（実機の実際の空きVRAMを都度読む代わりに固定値を指定することで、タイル寸法がどう変わるかを
切り分けている）。「14.73GiB」は実機で `cuda_activation_budget_bytes()` が返した実測の活性化
予算値。

### 補足観測

- 768×512/41fでは、推奨器の内部見積り（約3GiB相当）に対し、実測ピーク確保量は11.79GiBだった。
  推奨器がこの見積りに使う安全係数（`coef=5`）はnatten（後述）経由の実行を想定したものである。
- 公式推奨器が返す最小タイルは80f×320×320であり、これより小さいタイルは算出されない（4GiB
  申告・5GiB申告で拒否されたのはこの下限に起因する）。
- 比較値: 現行の畳み込みデコーダ版（Conv VAE）のデコードは、解像度・尺によらず約1.9GB横ばいである
  （[`Docs/VERIFICATION_LOG.md`](../../Docs/VERIFICATION_LOG.md) 340行目「VAE decode は
  ~1.9GB 横ばい」）。

---

## 3. 動作モードの実機確認

DiffVAEには4つの最適化モード（`DiffVAEMode`）があり、実機で確認した結果は次のとおり。

| モード | 実機確認結果 |
|---|---|
| `CHUNKED_EAGER`（既定） | 可。natten（後述）が環境に無いため、Tritonによる `na3d` 実装へフォールバックして動作した（ログ: `"DiffVAE NA fallback: using Triton na3d."`）。 |
| `CHUNKED_COMPILE` | `CHUNKED_EAGER` と同一の設定へ縮退する。NATTENを前提とするcompile経路がこの環境では使えないため。 |
| `COMBINED_COMPILE` | natten必須のため、例外を送出して拒否される。 |
| `BLACKWELL_DSL` | 対象GPU外。データセンター向けBlackwell世代GPU専用のCuTe DSLカーネルであり、RTX 4070 Ti SUPER（Ada Lovelace世代）は対象に含まれない。 |

---

## 4. natten（近傍注意を計算する専用ライブラリ）

- **公式最適化ガイド**（`packages/ltx-pipelines/docs/optimization.md`）は、非B200 GPUでの本番用
  DiffVAEデコードに `uv sync --package ltx-core --extra natten` を推奨する一方、natten不在時の
  フォールバック順は「Triton → eager」と明記している。
- **`ltx_core` 1.2.0自体の依存指定**（`.venv-engine-ltx25\Lib\site-packages\ltx_core-1.2.0.dist-info\METADATA`
  15〜16行目、ローカルで実機確認）: `natten==0.21.7+torch2130cu132` と `torch==2.13.0` は、
  environment markerが `(platform_machine == 'aarch64' and sys_platform == 'linux' and extra == 'natten') or (platform_machine == 'x86_64' and sys_platform == 'linux' and extra == 'natten')`
  の場合のみ要求される。**すなわち `natten` extraはLinux（x86_64/aarch64）限定であり、Windows上の
  pipインストールでは条件が成立せずextraの依存関係自体が要求されない。** 同ファイル360行目には
  「`natten` extraは `natten==0.21.7+torch2130cu132` と `torch==2.13.0`（cu132）を固定する。
  それより古いPyTorch/NVIDIAスタックでは、大きなstage-5ボリュームでNATTEN TokPerm内部にCUDA
  illegal memory accessが起きうる」という注記もある。
- **本プロジェクトの実行環境**は `torch` 2.9.1+cu128（`.venv-engine-ltx25`）であり、上記の
  `natten==0.21.7+torch2130cu132`（`torch==2.13.0`要求）とは異なる。
- **NATTEN公式サイト**（natten.org）: 「NOTE: Windows builds are experimental and not regularly
  tested.」（Windowsビルドは実験的で継続的なテスト対象外）と明記。公式wheel配布
  （whl.natten.org）はLinux（x86_64/aarch64）向けのみで、Windows向けの公式wheelは無い。
- **有志による非公式wheel**（`wildminder/AI-windows-whl`、GitHub）は、torch 2.12/2.13＋CUDA 13系
  向けのビルドのみを配布している。

---

## 5. 決定性の機序

- **DiffVAEのデコード呼び出し**（`ltx_pipelines.utils.blocks.VideoDecoder.__call__`、
  `diffvae_smoke.py` での実際の呼び出し形は `dec(latent, tiling_config, generator)`）は、
  第3引数として `torch.Generator` を受け取る。ノイズはこの `generator` を渡した
  `torch.randn(generator=...)` を通じて生成過程全体に伝わる。
- **公式Diffusers文書**（`Lightricks/LTX-2.5-Diffusers` README）: 「The decoder draws its own
  noise, so pass a generator for reproducible decoding.」「decoding is only reproducible with a
  `generator`.」（デコーダ自身がノイズを引くため、再現性のためには generator を渡す必要がある）。
- **タイル寸法はDiffVAEのときのみ、実行時の空きVRAM（`torch.cuda.mem_get_info` 相当、本プロジェ
  クトでは `ltx_core.devices.cuda_activation_budget_bytes()`）から決まる**（`diffvae_probe.py`
  の `free_bytes` 引数がこれに相当する）。畳み込みデコーダ版（Conv VAE）の自動タイル化は縦横比
  のみで決まり空きVRAM量を読まないため、この点がDiffVAE固有の非決定要因になる。
- **公式Diffusers文書**: 「Each tile is denoised separately, so a tiled decode does not
  reproduce an untiled one exactly.」（各タイルは個別にノイズ除去されるため、タイル分割デコード
  は非分割デコードを厳密には再現しない）。

---

## 6. 実装コード読解（アーキテクチャ・タイル化・メモリ見積り・決定性の追補）

本章は2026-09-01に実施した実装コードの読解結果を記す。対象は `.venv-engine-ltx25` の
`ltx_core` 1.2.0（Pythonソース）と実機重み `ltx-2.5-video-vae-bf16.safetensors` のsafetensors
ヘッダ（メタデータ）のみであり、**モデルロード・GPU実行は行っていない**（実測ではなくコード
読解のみに基づく）。以下の数値・既定値は実機重みに埋め込まれたconfigを正典として記載しており、
Pythonクラスのコンストラクタ既定値とは異なる場合がある。

### 6.1 アーキテクチャ

- デコーダのクラスは `NADiffusionDecoder`。5段構成。
  - ステージ1〜4: 決定的な近傍注意（Neighborhood Attention、近傍のトークンのみを参照する
    注意機構）Transformerアップサンプラ（`stage_channels` は2048/1024/512/512）＋
    PixelShuffle（チャンネル方向の値を空間方向へ並べ替えて解像度を上げる演算）アップサンプル。
    重み778MiBで、デコーダ全体（795.6MiB、§6.4参照）の98%を占める。
  - ステージ5: 拡散ブロック8個（`dim`=256・`head_dim`=64・SwiGLU（Swish-Gated Linear Unit、
    ゲート付き線形層を用いるFFN構成）`hidden`=1024・約890万パラメータ＝17.1MiB、デコーダ全体
    の2%）。ステージ5の格子は「フルフレームレート×画素の1/4解像度」で、近傍注意窓は
    `(11, 11, 11)`（画素換算で±5フレーム・±20画素に相当）。
- **実機configは `default_num_inference_steps=1`・`model_output_type="x0"`。** `_decode_one_tile`
  のステップループはこの既定値では0回転となり、σ=1の純雑音からx0を1発予測する（Euler更新も
  通らない）。ステップ数はコンストラクタ／config経由のみで指定され、実行時引数としては渡せない。
  ステップ数を増やすと、`x_t_init` が全長分（1280×768×169fで約0.93GiB）常駐する。
- 条件付けはcross-attentionではなく、concat（連結）／線形射影による加算注入。時刻埋め込みは
  AdaLN（Adaptive Layer Normalization、時刻埋め込みに応じてscale/shiftを調整する正規化）で
  行われ、scale/shiftのみが動的でgateはロード時に畳み込まれる。
- 乱数消費は2箇所のみ。1ステップ時はタイルごとに独立に `torch.randn(generator=...)` を呼ぶ。
  複数ステップ時は全長を一度に生成する。いずれもCPU generatorを渡せばGPU非依存になる。

### 6.2 タイル化の対象と下限の由来

- タイル化されるのはステージ4〜5のみ（時間・高さ・幅の3軸直積に分割し、台形マスクで単位分割
  をブレンドする）。**ステージ1〜3は全体ボリュームで1回だけ**実行され、その出力 `feat_s4` は
  復号処理の全期間にわたり常駐する（1280×768×169fで1.362GiB）。画素加算器はフル画面H×W分
  常駐する（同条件で0.879GiB）。この2つはタイル寸法に依存しない固定費であり、1280×768条件で
  合計約2.24GiBになる。
- 公式推奨器（§2で使用した `recommended_decode_tiling_config`）が返す下限80f×320×320は、
  受容野（あるトークンの出力に影響し得る入力範囲）に由来するoverlap（時間40フレーム・空間
  160画素＝ステージ5の8ブロック×半径5トークン）の2倍として決まっている。明示 `TilingConfig`
  はAPI上frames≥42・空間≥168まで受理するが、overlapは40/160のまま `_validate_overlap` で
  強制される。再計算倍率は幅256で2.67倍/軸、168で21倍/軸になる。
- タイル1枚のピークを支配するのはステージ5の256ch特徴（80f×512×768タイルで0.9375GiB）と、
  その注意入口のQKV／RoPE中間テンソル。CHUNKEDモードのピーク概算は≈4.4U≈4.1GiB（natten前提。
  §2の安全係数`coef=5`＝4.69GiBと整合する）。

### 6.3 メモリ見積り係数が外れた機序（コード上の候補）

- Tritonカーネル自体は余計なテンソルを実体化しない（出力1本のみ・オンラインsoftmax、softmax
  の正規化定数をブロック単位で逐次更新し中間全体を保持しない計算方式）。
- 推奨器の見積り係数はステージ5のみをモデル化しており、次の3項が未計上と読める。
  ① ステージ4のタイル毎注意作業域（約1.5〜2U相当）
  ② ステージ1〜3の一時領域（1280×768×169fで約1GiB）
  ③ RoPEがfp32で回り同時に6本生存する（`rope_compute_dtype` 既定`float32`。chunked経路
     では `rope_num_tiles=1` 固定でチャンク全体を一括回転する）
- natten不在時はモード再解決により、係数が7→5、安全マージンが2GiB→1GiBへ黙って緩む。
- `DiffusionVideoDecoder.recommended_tiling_config`（クラスメソッド）は必須kwargs欠落で
  TypeErrorとなり単体では動作しない。§2で実際に使用した `recommended_decode_tiling_config`
  （`ltx_pipelines/helpers.py` 経由）はこの制約を回避しており動作する。

### 6.4 コード上に存在する設定の口（挙動を変えるもの）

- `configure_abs_rope(compute_dtype=...)`（RoPEの計算精度）
- `configure_w_chunks`（既定4）
- `SwiGLUTileSpec`（既定16,384トークン）
- `free_bytes` の明示指定・`set_per_process_memory_fraction`（推奨器の予算入力）
- 明示 `TileSizeConfig`（`ensure_tiling_config` は明示指定があれば再導出せず検証のみ行う）
- `DiffVAEMode` はnatten不在環境では実質 `CHUNKED_EAGER` 一択（`COMBINED_COMPILE` は例外を
  送出して拒否・`CHUNKED_COMPILE` は `CHUNKED_EAGER` へ縮退・`BLACKWELL_DSL` は対象GPU外。
  §3参照）。
- ステップ間で保持されるのは `context`／`x_t` のみで、QKV等の作業域はブロック内で生成・破棄
  される。`feat_s4` は書き込みが1回のみで、以後はスライス読み出しのみ。
- 重み内訳: デコーダ795.6MiB（うちステージ1〜3系が736MiBで、`forward_stages_1_to_3` で
  1回のみ使用される）・エンコーダ608.3MiB。段別オフロードのフックはltx_core／ltx_pipelines
  いずれにも存在しない。

### 6.5 決定性の機序（追補）

第5章「決定性の機序」の追補として、コード読解で判明した内訳を記す（矛盾する内容はない）。

- 1ステップ時、雑音はタイルごとに独立に引かれる（§6.1）。そのため**タイル表（タイル数・
  順序・各タイルの形状）が乱数の消費順序を決める**。タイル表が変わると、継ぎ目付近だけで
  なく画面全体が変わる。
- `AUTO_TILING` は `mem_get_info` 相当の実測（実機では `cuda_activation_budget_bytes()`）を
  読むため、タイル表そのものが実行時の空きVRAMに依存する。これが第5章で述べた「タイル寸法
  はDiffVAEのときのみ実行時の空きVRAMから決まる」の内部機序にあたる。
- 明示 `TileSizeConfig` と固定 `generator` を組み合わせれば、タイル表・乱数消費順序とも
  実行時の空きVRAMに依存しなくなり、環境非依存になる。
- ブレンド加算器はfp16で計算され、タイル順が固定であれば再現的である。

### 主要ファイル一覧（本章の引用元）

| ファイル | 内容 |
|---|---|
| `diffusion_video_decoder.py` | `NADiffusionDecoder` 本体・ステージ構成・推論ステップループ |
| `diffusion_tiling.py` | タイル分割・overlap検証（`_validate_overlap`）・`TileSizeConfig`／`TilingConfig` |
| `attention.py` | 近傍注意の実装（Triton `na3d` 経路・natten経路） |
| `rope_math.py` | RoPE計算（`configure_abs_rope`・`rope_compute_dtype`） |
| `apply.py` | 推奨タイル寸法の算出（`recommended_decode_tiling_config` 等） |
| `transformer/config.py` | ステージ別チャンネル数・ブロック構成などのconfig定義 |

---

## 7. コミュニティ・公式の参照情報

伝聞（コミュニティ報告）と一次情報（公式ドキュメント・本プロジェクトの実測）を区別して転記する。

### 7.1 公式ドキュメント・一次情報

- `packages/ltx-pipelines/docs/optimization.md`（Lightricks/LTX-2公式）: DiffVAEの4モード
  （`chunked_eager`/`chunked_compile`/`combined_compile`/`blackwell_dsl`）の定義、natten
  フォールバック順、int32アドレッシング境界に伴うcompileコスト増（例: 736×1024×120fの構成で
  約70秒以上のコンパイル時間）について記述。
- `Lightricks/LTX-2.5-Diffusers` README（Hugging Face）: §5に転記した generator・タイル再現性
  の記述元。`LTX2VideoDiffusionDecodePipeline` による拡散デコーダの使用例も記載。
- Diffusers公式ドキュメント（Hugging Face・GitHub両方に同内容）: LTX-2パイプラインAPIの一般的な
  記述。
- SGLang cookbook（LTX2.5ページ）: LTX-2.5のデプロイ手順集。DiffVAEに特化した記述は確認できな
  かった。
- natten.org（NATTEN公式サイト）・`SHI-Labs/NATTEN` `docs/install.md`: §4に転記したWindows
  実験的対応・wheel配布範囲の記述元。

### 7.2 コミュニティ報告（伝聞・非一次）

- **LTX-2 GitHub Issue #277**（`Lightricks/LTX-2`、2026-09-01時点でOpen・未マージ）: 解像度
  1920×1088・24fps、フレーム数145/169/193/241で、AUTOタイリングが選ぶstage-5タイルの平坦化
  活性化テンソルが `2**32` 要素境界を跨ぐと、それ以降が一様グレーになる現象を報告。報告者の解析
  では、幅480×高さ272・チャンネル256のフルワイドタイルで「128フレーム目の活性化が
  4,278,190,080要素（`2**32` の49.8%地点）から始まる」ことが、観測されたグレー破損領域と一致
  するとしている。実機環境はNVIDIA A100-SXM4-80GB、`ti2vid_two_stages_hq` パイプライン。
- **ComfyUI Issue #15606**（`Comfy-Org/ComfyUI`）: Windows・RTX 5090（32GB）・ComfyUI 0.33.0
  （0.32.0でも再現）で、サンプリング完了後にLTX-2.5のDiffVAEを構築する際、Pythonの例外を出さ
  ないままプロセスごと終了する（アクセス違反 `0xC0000005`）事例。スタックトレースは
  `na_diffusion_decoder.py` 内の `torch.nn.Linear` 初期化位置を指している。
- **note.com記事①**（sepiablue「【LTX-2.5】ComfyUIでLTX-2.5をVRAM 12GBで動画生成してみた
  (i2v・簡易版)」）: RTX 4070（12GB）、736×1280・24fps・120フレーム（5秒）で総生成時間6分55秒。
  VRAMピーク11.4GB、メインメモリ（システムRAM）ピーク45.3GB（サンプラー実行時）。VAE処理が総時間
  の約半分を占めるとの記述。
- **note.com記事②**（黒箱AI実測録アニキ「LTX-2.5...replacing just one VAE file made it 22.5%
  faster [RTX 5090 measured results]」）: RTX 5090（32GB）、ComfyUI 0.33.0、1280×704・121フレ
  ーム。デコード単独の時間は拡散デコーダ8.87〜8.91秒 対 Conv VAE 4.68秒（Conv側が47.4%短い）。
  総生成時間は拡散デコーダ41.82秒 対 Conv VAE 32.43秒（Conv側が22.5%短い）。VRAM使用量はほぼ
  同等（約31,692MB 対 31,728MB）。
- **note.com記事③**（同著者「The Secret of the LTX-2.5 VAE: Only the Decoder Was Updated」）:
  RTX 5090（32GB）。画像→潜在→画像の往復テスト（拡散サンプリングを伴わない、9フレーム・
  768×1152）でConv VAEはPSNR 32.6dB（平均絶対差4.42）・デコード2.0秒、DiffVAEはPSNR 30.2dB
  （平均絶対差6.00）・デコード4.1秒。別途、img2vidの1フレーム目忠実度比較（1280×704・121フレ
  ーム系列）ではConv VAE（img_compression=0）がMAD 3.36・PSNR 34.5dB、DiffVAE（同条件）がMAD
  4.69・PSNR 32.0dB。
- **16GB級GPUでの拡散デコーダ成功報告**は、本調査の時点（2026-09-01）でのWeb検索では発見でき
  なかった。見つかった実測報告はRTX 4070（12GB、①）とRTX 5090（32GB、②③）のみである。

---

## 8. 出典URL一覧

- Lightricks/LTX-2（GitHub本体）: https://github.com/Lightricks/LTX-2
- LTX公式最適化ガイド（optimization.md）: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/optimization.md
- LTX-2.5公式モデルページ（Hugging Face）: https://huggingface.co/Lightricks/LTX-2.5
- Lightricks/LTX-2.5-Diffusers README（Hugging Face）: https://huggingface.co/Lightricks/LTX-2.5-Diffusers/blob/main/README.md
- Diffusers公式ドキュメント（Hugging Face、LTX-2ページ）: https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2
- Diffusers公式ドキュメント（GitHubソース）: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx_video.md
- SGLang cookbook（LTX2.5）: https://lmsysorg.mintlify.app/cookbook/diffusion/LTX/LTX2.5
- NATTEN公式サイト（インストールガイド）: https://natten.org/install/
- NATTEN GitHub install.md: https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md
- wildminder/AI-windows-whl（有志wheel配布）: https://github.com/wildminder/AI-windows-whl
- LTX-2 GitHub Issue #277: https://github.com/Lightricks/LTX-2/issues/277
- ComfyUI Issue #15606: https://github.com/Comfy-Org/ComfyUI/issues/15606
- note.com記事①（sepiablue、RTX 4070 12GB）: https://note.com/sepiablue/n/n85331fa19947
- note.com記事②（黒箱AI実測録アニキ、RTX 5090・22.5%高速化）: https://note.com/ai_drive/n/n7de65cfd5f20
- note.com記事③（同著者、RTX 5090・PSNR往復テスト）: https://note.com/ai_drive/n/n583ef17a9b6a

---

## 9. ファイル一覧

`outputs/diffvae-research-2026-09-01/` 直下。

| ファイル | 内容 |
|---|---|
| `RESULTS.md` | 本ファイル（一次記録） |
| `diffvae_smoke.py` | 実デコード計測スクリプト（読み取り専用。scratchpadから複製・再現性のため保存） |
| `diffvae_probe.py` | タイル推奨器の照会スクリプト（GPU未使用・読み取り専用。同上） |

両スクリプトとも製品のコード・設定ファイルへは一切書き込まない（標準出力へ結果を表示するのみ）。
