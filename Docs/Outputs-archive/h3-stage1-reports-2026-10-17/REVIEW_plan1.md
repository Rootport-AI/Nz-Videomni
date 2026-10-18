# 計画 `cached-doodling-hare.md`（MiniMax H3 α版 第 1 弾）の事実確認レビュー

レビュー日 2026-10-17。読むだけで、GPU・インストール・編集はしていない。
（注: このセッションはプランモードで、書き込みが許されたのは本ファイルだけだった。指定の `scratchpad\h3\REVIEW_plan1.md` には書けていない。中身は同じなので、必要なら監督が写す。）

## (1) 総評

- 骨格（素の int8 Parameter＋f32 スケールのバッファ＋`ck.int8_linear`、既存の block swap と先読み、T2V の系列の組み方）は、現物のコードと照らして成り立つ。先読みの待ち合わせの順序にも穴は見つからなかった。
- ただし、モデルを組む手順が書かれていない。DiT も TE も「普通に組んでから差し替える」と書いてあり、このとおりに作ると CPU の RAM が溢れて第 1 弾が成立しない。ほかに、門の観測手段の欠落（eager に落ちたことを見分けられない、TE の RoPE 設定を確かめる手段がない）と、第 2 弾で手戻りになる名前の決め方が 3 点ある。
- 過剰な設計は軽い（G2 の m=80,000、視覚塔の読み込み）。文書の規約との食い違いは小さい。

## (2) 指摘の一覧（重さ順）

### C-1【Critical】DiT と TE の組み方が「普通に組んで差し替える」で、CPU の RAM が溢れる
- **指摘**: 計画 §3.3 では、`dit.py` は DiffSynth の移植で、`text_encoder.py` は「`Qwen3VLModel` を `num_hidden_layers=50` で組み…量子化 350 本は `H3Int8Linear` に差し替え」、§6.4 も「DiffSynth の方法そのまま」と書いている。移植元の `MiniMaxH3DiT.__init__` は `nn.Linear` を既定の dtype（f32）で作る。DiT 本体はパラメータ約 193 億（ブロックあたり qkv 115M・out 38.5M・fc1 154M・fc2 77M、これが 50 個）で、f32 なら約 77GB になる。TE は 50 層で約 244 億（q 42M・k/v 各 5.2M・o 42M・gate/up/down 各 131M、これが 50 層）で、config の dtype の bf16 でも約 49GB、f32 なら約 98GB になる。開発機の 64GB では組み立ての時点で溢れる。§6.2 の「山は約 27GB＋α」は、この割り当てが無い前提でしか成り立たない。`H3Int8Linear` が int8 の空テンソルを自分で確保する作りでも、読み込み時に reader の分がもう 1 本でき、山が 2 倍になる。
- **根拠**: `DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py:330-345`（`nn.Linear` を直に作る）・`minimax_h3_text_encoder.py:20-72`（`Qwen3VLModel(config)` を直に作る）・transformers 5.14.1 `modeling_qwen3_vl.py:751-753`（`Qwen3VLTextModel` が 50 層の `nn.Linear` を作る）。既存の作法は `engine/sft_quant/quant_service.py` の `load_state_dict(strict=False, assign=True)`（R3 項目 2）。
- **直し方の案**: §3.3 と §5 に「DiT も TE も `torch.device("meta")` の中で組む → 量子化層を `H3Int8Linear`（これも meta）に差し替える → `sft_reader` で 1 本ずつ読んだテンソルを `load_state_dict(assign=True)` で入れる → 読み残しの meta テンソルが無いことを検査する」と書く。RAM の山（TE の読み込み後・解放後・DiT の読み込み後）を G5 の `done` に出す（`rss_peak_gib` はすでに予定にある）。

### M-1【Major】「eager に落ちていない」ことを観測する手段がない（G3）
- **指摘**: G3 は「forward が CUDA 経路で走ること（こだまが `on->on`・eager に落ちていないこと）」を合否にしている。しかし `ck.int8_linear` は呼び出しごとに制約を照合し、合わなければ黙って次のバックエンドへ移る。例えば重みやスケールが CPU に残っている、スケールが f32 でない、などの場合である。triton を切っておけば落ち先は eager になり、eager は `weight.to(x.device)` で毎回転送するので結果も正しい。だから出力を見ても区別できない。`ck.use_backend("cuda")` も、照合に失敗すると優先順のリストへ戻るだけで、例外にはならない。また、こだまの値は `"off"`／`"on"`／`"on->off"` の 3 種で、`on->on` という値は存在しない。
- **根拠**: `comfy-kitchen/comfy_kitchen/registry.py:180-233`（override に失敗すると `self._priority` の巡回へ進む）・`__init__.py:1056-1058`・`backends/eager/quantization.py:1028`（R1 2-2 追加 (b)）・`engine/transformer/block_swap_service.py:136-138`。
- **直し方の案**: `require_cuda_int8()` で `ck.disable_backend("triton")` に加えて `ck.disable_backend("eager")` も切る。こうすれば、CUDA の制約から外れた呼び出しは `NoCapableBackendError` で止まる（fail loud）。H3 の経路で comfy-kitchen を使うのは `int8_linear` だけなので副作用はない。G3 の合格条件は `last_prefetch_used == "on"` に直す。あわせて、同じ入力について「全ブロック常駐」「同期の block swap」「先読み」の 3 通りを 2 パスずつ回し、出力がビット単位で一致することも条件に加える。先読みの説明文が記録している不具合の型、すなわち落ちずに digest だけが変わる型（`block_swap_prefetch.py:105-118`）を、ここで捕まえるためである。

### M-2【Major】TE の RoPE の設定（`rope_parameters` の読み替え）を確かめる門が無い
- **指摘**: §6.4 は「5 系の設定の読み替え（`rope_parameters`）で引っかかる可能性 → G4 のトークン列の一致で確かめる」としている。しかしトークン列は TE の forward を通らないので、mrope の section・interleaved・theta が誤って読まれても検出できない。G4 の TE 欄も「トークン列と提示文の一致だけ」で、50 層の数値（350 本の名前の写し・層オフロードの包み・norm=Identity）を照合する門が無い。
- **根拠**: 計画 §3.5 G4・§6.4。DiffSynth の設定は `rope_scaling: {mrope_interleaved: True, mrope_section: [24,20,20]}`・`rope_theta 5000000`（`minimax_h3_text_encoder.py:38-44`）。R1 2-6。
- **直し方の案**: (1) CPU のテストで、組んだ後の `config.text_config` の rope の各値（section・interleaved・theta・head_dim）を固定値と照合する。(2) G4 の TE 欄に「先頭 2 層と最後の 1 層を、`engine/sft_quant/dequant.py` で bf16 に戻した素の Linear の版と `H3Int8Linear` の版で比べる」と、「層オフロード on と off で 50 層後の出力がビット単位で一致する」を足す。

### M-3【Major】追加特殊トークン `<d>` などが transformers 5.14.1 で語彙に入るかを、番号で確かめていない
- **指摘**: 計画は「追加特殊トークンを読み込む」と書くだけである。transformers 5 は `additional_special_tokens` を `extra_special_tokens` に読み替える（`tokenization_utils_base.py:1007-1009, 1822-1824`）。4 系と同じく 151669〜151675 として語彙に足されるかは確かめていない。入らなかった場合、`<d>` は普通の BPE に割れる。落ちずに音声の台詞の扱いだけが変わるので、G5 の目視でも気づきにくい。
- **根拠**: R2 D・ComfyUI `text_encoders/minimax.py:32-34, 132`（番号を明示したうえで `add_special_tokens` で足している）。
- **直し方の案**: CPU テストに「`<d>`→151669 … `<|caption_end|>`→151675、`"<d>[English] hi</d>"` の先頭と末尾がその番号になる」を足す。通らなければ、ComfyUI と同じように明示的に `add_special_tokens` を呼ぶ。

### M-4【Major】G4 の「DiffSynth のファイルを写して数行の shim」は、見積もりが小さすぎる
- **指摘**: `minimax_h3_dit.py` は `..core.attention.attention_forward` と `..core.gradient.gradient_checkpoint_forward` を相対 import する（`:8-9`）。だから shim は「SDPA への差し替え数行」では済まず、パッケージの形を真似たダミーの `core` が要る。さらに、正解側の int8 Linear をどう作るかが書かれていない。DiffSynth と同じ `QuantizedTensor` 経路で作るなら `core/quant/backends/comfy_kitchen.py`（318 行）と `base.py`・`config.py`（計約 800 行）が要る。その場合は同じカーネルを通るので、照合で確かめられるのは配線だけになる。VAE も同じで、映像は `.weight` を付ける変換、音声は weight_norm の g/v への分解が、正解側にも必要になる。
- **根拠**: `DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py:8-9`・`utils/state_dict_converters/minimax_h3_video_vae.py`・`minimax_h3_audio_vae.py:25-60`。R2 A。
- **直し方の案**: 正解側の Linear は「`engine/sft_quant/dequant.py`（int8_convrot・groupsize 256 に対応済み）で bf16／f32 に戻した `nn.Linear`」に決める。許容誤差は活性化の量子化誤差を見込んで決め直す（相対誤差 1e-2 級。1e-3 は当てにしない）。そのうえで「`H3Int8Linear` と、逆量子化してから `F.linear` の版を、Linear 1 本ずつ比べる」試験を別に置き、カーネルの正しさとモデルの配線を分けて確かめる。shim の作業量は C1 の担当に明記する。なお、16GB でブロック 1 個ずつ載るかについては、int8 0.39GB に逆量子化した bf16 の 0.77GB を足し、短い系列で動かせば十分に載る。

### M-5【Major】ヘッダの判定（指紋）が torch を要るモジュールに入っており、第 2 弾のアプリ側の事前検査から使えない
- **指摘**: §3.3 は、H3 の判定・接頭辞の剥がし・形状から構成を導く処理・`comfy_quant` の読みを `engine_h3/sft_load.py` に入れ、同じファイルで `sft_reader`（torch）も使う。第 2 弾では `services/model_registry.py` の `precheck_model_file` に H3 の判定を足す必要がある（R3 項目 4）。しかしアプリの venv には torch が無く、エンジン側のパッケージをアプリから import する前例もない。ここで torch と混ぜると、第 2 弾で判定部分を切り出し直すことになる。
- **根拠**: R3 項目 4・(3)-4。`MULTI_ENGINE_DESIGN.md` §2.3（`sft_quant_format.py` は torch に依存しないので、アプリ側とエンジン側の両方から呼べる）。
- **直し方の案**: 判定と構成の導出だけを、torch を使わないモジュールに分ける。置き場は `sft_quant_format.py` と同格のリポジトリ直下（例 `h3_format.py`）か、torch を import しない `engine_h3/h3_header.py`。`sft_load.py` はそれを呼ぶだけにする。テスト（合成ヘッダ）は torch 無しで回せるので、アプリの venv でも走る。

### M-6【Major】`progress` の段階名 `denoise` は、親の割合計算が知らない語である
- **指摘**: 計画の `progress` は `stage`＝`encode`／`denoise`／`decode` になっている。ところが親の `_progress_frac` が単発生成で知っている語は `encode`・`stage1_denoise`・`stage2_denoise` だけで、知らない語には None を返す。そのため、第 2 弾で `_RealBackend` を継承して `_read_worker_events` を使い回すと、ノイズ除去の間ずっと進捗バーが止まる。単段の生成には、`one_stage=True` で `stage1_denoise` が 0.06〜0.90 を埋める前例（Alpha Gen）がすでにある。
- **根拠**: `services/engines/ltx/adapter.py:431-484`・R3 項目 1・項目 6 の注意。
- **直し方の案**: ワーカーは最初から `stage1_denoise`（`index`／`total`）を出す。第 2 弾のアダプタは `one_stage` の経路を使う。設計正本には「H3 は単段で、LTX の語彙を借りる」と 1 行書く。共有の親を触らずに済む。

### m-1【Minor】`done` に、第 2 弾で親が読むこだまのキーが足りない
- **指摘**: 親は `attention_used`・`block_swap_prefetch_used`・`peak_vram_reserved_mb` を `.get` で読む。だから欠けても落ちはしないが、`metadata.json` に None が入る。G3 と G5 自体も、先読みが効いたかを `done` から確かめる必要がある。
- **根拠**: `services/engines/ltx/adapter.py:2395-2417`・`ltx25/adapter.py:1053-1106`。
- **直し方の案**: `done` に `attention_used`・`block_swap_prefetch_used`・`peak_vram_reserved_mb` を足す。`h3.phases` と同じ階層に `rss` の段ごとの値を入れてもよい。

### m-2【Minor】セルフテストの引数に `--prompt`・`--attention-backend`・`--blocks-on-gpu`・`--prefetch` が無い
- **指摘**: G5 は「`<d>[English] …</d>` を含む文章」で回し、§6.6 は sage を 1 回回し、G3 と G5 は先読みを on にする。しかし §3.3 の引数の一覧には、それを指定する手段が無い。
- **根拠**: 計画 §3.3 `worker.py` の行・`engine25/worker.py:1990-1999`（`--prompt` と高速化の引数がある）。
- **直し方の案**: 上の 4 つを足す。あわせて、`load` と `generate` のメッセージのキー名（`transformer_path`・`text_encoder_path`・`video_vae_path`・`audio_vae_path`・`tokenizer_dir`・`blocks_on_gpu`・`te_layers_on_gpu`・`attention_backend`・`block_swap_prefetch`）をここで決めておく。そうすれば第 2 弾の `_build_load_payload` がそのまま写せる。tokenizer はカテゴリではないので、第 2 弾では記述子の `assets` と `REQUIRED_ASSETS` で渡すことになる。この点も設計正本に書いておく。

### m-3【Minor】推論モード（`no_grad` か `inference_mode` か）が決まっていない
- **指摘**: 設計正本 §5.8 は、エンジン系統ごとに推論の囲みを 1 つに揃える規約を持つ。先読みのページ固定プールはジョブをまたいで生き残り、囲みが混ざると次のジョブが落ちる（§3-124 の実例）。計画にはこの指定が無い。
- **根拠**: `Docs/MULTI_ENGINE_DESIGN.md` §5.8・`block_swap_service.py:87-91`（プールはサービスが持ち続ける）。
- **直し方の案**: 「`engine_h3` は全経路 `torch.no_grad()`」と決め、engine25 と同じ形の CPU テスト（`engine_h3` 配下に `inference_mode` が無いこと）を足す。§5.8 の表にも 1 行足す。

### m-4【Minor】mp4 の BT.709 は「タグを付ける」だけでは色がずれる
- **指摘**: LTX 2.5 の `encode_video` は、RGB→YUV の変換を自前の BT.709 行列（`yuv420p_bt709_converter_`）で行ってからタグを付けている。PyAV の `VideoFrame.from_ndarray(rgb24)` を libx264 の yuv420p にそのまま渡すと、変換は swscale の既定（BT.601）で行われ、タグだけが BT.709 になる。
- **根拠**: `.venv-engine-ltx25/.../ltx_pipelines/utils/media_io/encode.py:127-201`・`ltx_core/color/audio_mux.py`。
- **直し方の案**: `media_out.py` で変換の行列を明示する（`reformat(format="yuv420p", dst_colorspace=…, dst_color_range=…)` を使うか、BT.709 の式で自前に変換する）。音声は LTX の手順をそのまま写せば足りる。手順は、s16 のパックした AudioFrame → `AudioResampler`（エンコーダの fltp・stereo・32000）→ pts を振る → 最後にフラッシュ、の順である。`ltx_core` は import しない。

### m-5【Minor】VAE の計算精度が書かれていない
- **指摘**: 映像 VAE は fp16 のファイル（5.21GB）、音声 VAE は fp32 のファイルである。計算を bf16・fp16・fp32 のどれで行うかで、VRAM（映像を fp32 にすると重みだけで 10.4GB）と結果が変わる。DiffSynth は `torch_dtype`（bf16）で解いている。
- **根拠**: `minimax_h3_video_vae.py:540-552`・計画 §3.3。
- **直し方の案**: 「映像 VAE は bf16、音声 VAE は f32（BigVGAN と Snake は精度に敏感なため）」のように決めて書く。`latents_mean`／`latents_std` を F16 のテンソルからではなく F32 の定数かメタデータの JSON から取ることも、正本に書く（R2 (3)-16）。

### m-6【Minor】`weight_scale` を `[N]` に変える必然性がない
- **指摘**: comfy-kitchen の CUDA 経路は、`[N,1]` でも連続な f32 ならそのまま使う。CUTLASS には `numel()==n` で渡り、cuBLAS の後の逆量子化にも縦ベクトルとして通る。`[N]` にするには読み込み時に形を変える必要があり、`load_state_dict(assign=True)` は形が違うと例外になる。さらに、既存の検査（`sft_quant_format._ROW_SCALE`）と、M-4 の正解側で使う `dequant.py` は `(o,1)` を前提にしている。
- **根拠**: `comfy-kitchen/.../backends/cuda/__init__.py:526-529, 2228`・`sft_quant_format.py:122`。
- **直し方の案**: ファイルどおり `[N,1]` で持つ。§5 の「`[N]` で持つ」を消す。

### m-7【Minor】RoPE の「float64」の範囲を誤読しやすい
- **指摘**: 座標は float64 で作るが、DiffSynth も diffusers も、角度の計算の前に float32 に落としている（`pos.to(torch.float32)`・`position_ids.to(torch.float32)`）。§5 の「RoPE は float64」を角度の計算まで float64 と読むと、正解側とずれる。
- **根拠**: `minimax_h3_dit.py:95-102`・diffusers `transformer_minimax_h3.py:96-104`。
- **直し方の案**: 「座標は float64 で作り、角度と cos／sin は float32（参照実装と同じ）」と書く。

### m-8【Minor】AdaLN の表の格子数を `1023`／`1024` と決め打ちしている
- **指摘**: §3.3 は「`pos=clamp(t)·1024`、`i0=min(floor,1023)`」と書く一方で、「形状から格子数を導く」とも書いており、両者が食い違う（コミュニティ版には格子 4097 がある。研究ノート §13.3 の 6）。
- **根拠**: `minimax_h3_dit_comfy.py:38-41`（`table.shape[0]-1`・`-2`）。
- **直し方の案**: `G-1`・`G-2` で書く。テストに「step 0 では t_v=t_a=0 なので unique が 1 個になる（M=1）」と「step 1 以降は M=2」を足す（T2V では t=1.0 に届かないので、表の端の試験は I2V の条件行を想定した合成値で行う）。

### m-9【Minor】仮想環境のピンの抜けと、dry-run の主張の範囲
- **指摘**: (1) 移植する映像 VAE は `einops`（`rearrange`・`repeat`）を import するが、ピンの一覧に無い（`minimax_h3_video_vae.py:7, 11`）。transformers 5.14.1 も torchvision も einops を引き込まない。(2) `av` に版の指定が無い（R1 の解決では 19.0.1、既存の freeze は 18.1.0）。(3) 「uv の解決は dry-run で成功済み」は、diffsynth と accelerate を含めた B2 の組での話である。diffsynth を除いた組（C）を試したのは torch 2.14.1 の側だけである（R1 2-7 の表）。(4) エンジンの venv には pytest が無く、`tests/conftest.py` は FastAPI を import する。
- **根拠**: 上記の各箇所・`Docs/Outputs-archive/start-end-bridge-2026-09-07/implA_engine_runner/run_ltx25_pytest.py`。
- **直し方の案**: `einops==0.8.2` と `av` の版（18.1.0 か 19.0.1 のどちらか）を明記する。dry-run は本番のピンの組でもう一度回してから構築する（GPU は不要）。§3.4 には「既存の runner の形（アプリ venv の site-packages を末尾に足し、`--noconftest` を付ける）で回す」と書く。

### m-10【Minor】G1 で sage を必須にしている
- **指摘**: 既定は sdpa で、sage は §6.6 で任意に 1 回試すだけである。それなのに `probe_sage()` が True であることを G1 の合否に入れている。
- **直し方の案**: G1 の合否は `int8_linear` だけで決め、sage の可否は記録に回す。

### m-11【Minor】ステップ 20 は Comfy の雛形の値だが、サンプラが違う
- **指摘**: Comfy の公式雛形の 20 は `res_multistep`（2 次の多段法）での値で、参照実装の Euler の既定は 50 である（R2 G・I）。Euler で 20 にすると、雛形より画質が落ちる可能性がある。裁定 18 が決めたのは「定数にして画面に出さない」ことだけで、20 という値はまだオーナーの裁定になっていない。
- **直し方の案**: 計画の §1 に「20 は監督の選択、サンプラの違いあり」と明記して了承を得る。G5 で絵が甘ければ、まず 50 で 1 本比べる、と申し送る。

### m-12【Minor】過剰な設計 2 点
- **指摘**: (1) G2 の m=80,000 は #136 の境界を CUDA 経路で確かめる試験である。しかし第 1 弾の系列は約 9.5k 行（640×384×124f）で、境界の約 75k に遠く、triton を切るので #136 自体にも当たらない。§1-95（快適上限の較正）で大きな尺を測るときに要る試験である。(2) 視覚塔（1.19GB）は T2V では使わないのに読み込む。
- **直し方の案**: (1) G2 は形 `[9.5k,5376]×[28672,5376]` の煙試験と、eager の結果との一致の確認にとどめ、m=80,000 は §1-95 へ送る。残す場合も「落ちない」だけでなく、末尾の行の値を eager と比べる（int32 のあふれは不正アクセスでなく、静かに書き間違える形でも出うる）。(2) 第 1 弾では視覚塔を meta のまま残すか CPU に置き、GPU には載せない。

### m-13【Minor】文書の規約との整合
- **指摘**:
  - (1) 裁定 1〜21 には、設計ではなく進め方の裁定（12・13・20 など、セッションの範囲や了承の単位）が混ざっている。これを設計正本へ「移管」すると、正本に歴史が入る。
  - (2) `MULTI_ENGINE_DESIGN.md` で現行化が要るのは §5.3 だけではない。§5.3 の venv の一覧と「容量は心配しなくてよい」の段（H3 では torch が違うので約 3.5〜4GB 増える）、§5.8 の推論モードの表、§9.2 の「推論パイプライン一式」の行、§2.2・§2.4 の「ComfyUI 方式のテンソル名の推測は採っていない」も、H3 が指紋の判定を採る時点で訂正の対象になる。
  - (3) ピンの版が、設計正本・pyproject・freeze の 3 箇所に書き写されるおそれがある。
- **直し方の案**:
  - (1) 設計正本へは設計の裁定だけを移し、進め方の裁定は研究ノート §10 に残して参照で結ぶ。
  - (2) 上記の各節に 1 行ずつの訂正の引用を添える。中身は H3 の設計正本の節を指すだけにする。
  - (3) 版の正本は freeze（と pyproject の「なぜこの版か」）の 1 箇所にし、設計正本は参照にとどめる。

### m-14【Minor】load と generate の分担と、ジョブごとの読み直し
- **指摘**: §3.3 の pipeline は、ジョブごとに「TE を組む→解放→DiT を組む→解放」を行う。この形だと、1 ジョブごとに約 48GB をディスクから読み直す。第 1 弾のセルフテストでは問題ないが、`_do_load` が何をするのか（パスの確認だけか、何かを常駐させるか）は書かれていない。
- **直し方の案**: 設計正本に「第 1 弾の load はパスの検査とヘッダの判定だけ、generate が毎回組む」と明記する。読み込みの秒数は `h3.phases` に出す。常駐（`cache_weights` 相当）を採るかどうかは第 2 弾の判断事項として残す。

## (3) 確認したが問題なしと判断した点

- **(a) 先読みと `int8_linear` の待ち合わせ**: 先読みは、forward の先頭の `on_block_forward` で S2（計算側のストリームがコピーのイベントを待つ）を積んでから、元の forward を呼ぶ（`block_swap_prefetch.py:557-561`）。comfy-kitchen は、量子化・GEMM・逆量子化のすべてを `torch.cuda.current_stream(x.device)` に積む（`backends/cuda/__init__.py:2055`）。したがって、カーネルはデータが届いた後に走る。アリーナの境界は 512 バイトにそろっているので、CUTLASS が要る 16 バイト境界（`_aligned16`）も、f32 の view も合法である。view は連続なので、毎回の `.contiguous()` のコピーも起きない。スケールはバッファとしてブロックと一緒にアリーナに載る。同期経路でも `blk.to(device)` でバッファまで移る。ポインタをキーにした重みのキャッシュは comfy-kitchen に無く（キャッシュは機器の性質と RoPE の表だけ）、アリーナの再利用で古い値を読む心配は無い。
- **(b) `GemmaLayerOffloadService` を Qwen3-VL に使うこと**: `Qwen3VLModel` を渡すと、`language_model.layers`（50 個の `nn.ModuleList`）が候補として見つかる（`layer_offload_service.py:214-223`）。層の forward の署名（`hidden_states, position_embeddings, attention_mask, position_ids, past_key_values, use_cache, **kwargs`）は位置引数 1 つとキーワード引数だけで、包みの `_move_to_device` がタプルの `position_embeddings` まで移す。`layers_on_gpu` の最小値は 1（0 だと無効）。`compute_device` は明示して渡すこと、`use_cache=False` で呼ぶこと（DiffSynth と同じ）の 2 点を申し送れば足りる。eviction のたびに GPU→CPU のコピーが起きるが、エンコードは 1 回だけなので許容できる。
- **(c) T2V の系列・時刻・RoPE**: `[text|audio|video]`・テキストと映像は t_v・音声は t_a・タグは 0／1／2・AdaLN の添字＝番号×3＋タグ・最終層の添字は番号だけ、のいずれも DiffSynth の `_build_packed_fl2va` と `model_fn` に一致する。ステップ 20・shift 12/3 では、step 0 で M=1、それ以降は M=2 で、t が 1.0 に届くことはない。
- **(d) フレームと潜在の数・タイル**: n=7 では T'=37・A=207、n=20 では T'=102・A=575 で、R2 の表と一致する。タイルの分割は、高さ 384 で開始位置 [0,128]・重なり 128、幅 640 で開始位置 [0,192,384]・重なり 64 になり、`split_tiles` は端まで覆う。
- **(f) wheel**: torch 2.13.0+cu130・torchvision 0.28.0+cu130・triton-windows 3.7.1.post27・sage post6 cu130・comfy-kitchen 0.2.37 cp312-abi3 は、いずれも R1 が URL と公開日まで確かめている。torchaudio を入れない副作用も見当たらない（H3 の各ファイル・`sft_reader`・`av` のいずれも torchaudio を使わない。`sft_reader` が使う numpy は、transformers と torchvision 経由で入る）。
- **CUDA 経路の制約**: x は bf16 で 2 次元、weight は int8 で 2 次元、scale は f32、DiT と TE の int8 Linear に bias は無い。いずれも `int8_linear` の CUDA の制約（`backends/cuda/__init__.py:4463-4488`）を満たす。`convrot` は Python の bool で渡すこと（制約は `{bool}`）。
- `_get_blocks` は、DiT の最上位の `blocks` で本体の 50 個を取る。`token_refiner.blocks` は別の階層なので巻き込まない。
- 重みの容量と VRAM: 1 ブロックは int8 で約 0.39GB、アリーナの輪は 10 個で約 3.9GB、常駐部は token_refiner の 1.54GB ほか、活性化は 9.5k 行で数百 MB 級。16GB に収まる見込みである。
