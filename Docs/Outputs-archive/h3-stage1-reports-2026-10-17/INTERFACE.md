# engine_h3 の境界の仕様（第 1 弾・担当 3 体の共通契約）

監督が 2026-10-17 に書いた。**この文書に書いた名前・形・dtype は、担当が勝手に変えない。** 変える必要があると判断したら、作業を止めて監督に理由を添えて報告する（監督が全担当に周知してから変える）。計画の本文は `C:\Users\phenotype nezu\.claude\plans\cached-doodling-hare.md`。

## 0. 全員の約束

- 置き場: リポジトリ `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`。新設は `h3_format.py`（直下）と `engine_h3/`（新しいパッケージ）。`tests/test_h3_*.py` は C3 だけが書く。
- **変更禁止**: `engine/`・`engine25/`・`services/`・`api/`・`scripts/`・`Docs/`・`sft_quant_format.py`・`chain_math.py`・既存の `tests/*`・フロントエンド。共有部品（`engine/transformer/*`・`engine/sft_quant/*`・`engine/gemma/*`）は **import して使うだけ**。噛み合わなければ `engine_h3/` 側で包む。どうしても共有部品を変えたくなったら止めて報告。
- 担当外のファイルは編集しない（1 ファイル 1 所有者。所有者は §9）。
- 推論の囲みは **`torch.no_grad()` だけ**。`torch.inference_mode` は `engine_h3/` に 1 箇所も書かない（設計書 §5.8。CPU テストで固定する）。
- モデルは **`with torch.device("meta"):` の中で組む**。重みは `engine_h3/sft_load.py` が 1 本ずつ読んで `load_state_dict(assign=True)` で入れる。普通の精度で先に確保しない（DiT f32 約 77GB・TE bf16 約 49GB で 64GB を超える）。
- `engine_h3/*.py` はモジュール最上位で torch を import してよい。**`engine_h3/worker.py` だけは最上位で torch を import しない**（engine25 と同じ。`load` の中で遅延 import）。
- 移植元は DiffSynth-Studio（Apache-2.0。`...\scratchpad\h3\src\DiffSynth-Studio\diffsynth\`）。移植したファイルの先頭に、由来（ファイルパスとコミット `4e34969`）と Apache-2.0 の表示を書く。**ComfyUI（GPL-3.0）のコードは写さない**（振る舞いを読んで理解するのは可）。diffusers（`...\scratchpad\h3\src\dx\`）は照合先。
- 語調と文体: コード内の説明文は英語でも日本語でもよいが、既存の `engine25/*.py` と同じ流儀（なぜそうするかを書く）。文字コード UTF-8・改行は既存の `engine25/*.py` に合わせる。
- 仮想環境 `.venv-engine-h3\Scripts\python.exe` は監督が構築中。存在すれば CPU の検査に使ってよい。**GPU は使わない**（門は監督が統制する。GPU を使う試験は監督から「GPU の使用を許可」の連絡があってから）。
- 既存の venv（`.venv*`）・本体の Python・`.uv_cache` には触らない。何もインストールしない。
- 判断に迷ったら止めて報告（根拠＝エラー全文・ソースの行）。設計の勝手な変更は禁止。

## 1. 定数（置き場は `engine_h3/constants.py`。C3 が作る。他は import して使う）

```python
DEFAULT_STEPS = 20
DEFAULT_SHIFT_VIDEO = 12.0
DEFAULT_SHIFT_AUDIO = 3.0
FRAME_RATE = 24                 # 固定
SPATIAL_FACTOR = 32             # 1 トークン = 32px 四方(VAE 16 × patch 2)
VAE_SPATIAL = 16
VAE_TEMPORAL = 4
VAE_CLIP_LENGTH = 17
VAE_TOKEN_DROP = 3
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_SAMPLE_RATE = 32000
AUDIO_HOP = 800                 # 1 音声潜在 = 800 サンプル
KEYFRAME_NOISE_AUG = 0.999      # 第 2 弾で使う
MODALITY_VIDEO, MODALITY_TEXT, MODALITY_AUDIO = 0, 1, 2
LOG_PREFIX = "[h3_worker]"
def latent_frames(num_frames: int) -> int        # (F - 5) // 17 * 5 + 2 。F % 17 != 5 は ValueError
def audio_latents(num_frames: int) -> int        # round(F / 24 * 40)
```

## 2. `h3_format.py`（リポジトリ直下・**torch 非依存**・C3）

`sft_quant_format.read_header` / `read_tensor_bytes` / `TensorInfo` / `Header` の上に書く。アプリ側（torch の無い venv）からも import される。

```python
class H3FormatError(ValueError): ...

@dataclass(frozen=True)
class QuantSpec:
    format: str                 # "int8_tensorwise" だけを受理。他は H3FormatError
    convrot: bool
    convrot_groupsize: int      # 既定 256。256 以外も受理(そのまま int8_linear に渡る)

def parse_comfy_quant(raw: bytes) -> QuantSpec
    # JSON。読むキーは format / convrot / convrot_groupsize だけ(最上位か "params" の下)。他は無視

H3_PREFIXES = ("", "model.diffusion_model.", "model.model.", "net.")
def detect_h3_prefix(names: Iterable[str]) -> str       # video_patch_proj.weight を持つ接頭辞。無ければ H3FormatError
def is_h3_dit_header(header) -> bool                    # {prefix}video_patch_proj.weight と {prefix}audio_patch_proj.weight の同時存在

@dataclass(frozen=True)
class H3DiTConfig:
    prefix: str
    hidden_size: int            # video_patch_proj.weight.shape[0] = 5376
    num_layers: int             # blocks.N の数 = 50
    token_refiner_layers: int   # token_refiner.blocks.N の数 = 2
    num_heads: int              # qkv_proj.weight.shape[0] / (3*head_dim) = 56
    head_dim: int               # blocks.0.attn.q_norm.weight.shape[0] = 128
    ffn_hidden: int             # fc1.weight.shape[0] / 2 = 14336
    text_dim: int               # condition_proj.weight.shape[1] = 5120
    latents_dim: int            # final_layer.video_out.weight.shape[0] / 4 = 24
    audio_latents_dim: int      # final_layer.audio_out.weight.shape[0] = 32
    pruned: bool                # adaln_t_table がある
    adaln_grid: int | None      # adaln_t_table.shape[0] (1025)
    adaln_rank: int | None      # adaln_t_table.shape[1] (8)
    rope_inv_freq_len: int      # rope.inv_freq.shape[0] = 16
    quant: dict[str, QuantSpec] # 接頭辞を剥がした層名("blocks.0.attn.qkv_proj") → spec
    unknown: tuple[str, ...]    # 知らないテンソル名(剥がした後)。読み込み側が警告して捨てる

def inspect_h3_dit(path: str) -> tuple[Header, H3DiTConfig]
    # H3 でない / 非枝刈り(adaln_t_table なし) / 必須テンソル欠落 / 形の矛盾 は H3FormatError(日本語の理由つき)
    # 非枝刈りは第 1 弾では拒否(理由を文で返す)

@dataclass(frozen=True)
class H3TEConfig:
    num_layers: int             # model.layers.N の数 = 50
    quant: dict[str, QuantSpec] # "model.layers.0.self_attn.q_proj" → spec
    has_visual: bool            # visual.* がある
    metadata: dict              # __metadata__ (minimax_h3_te など)

def inspect_h3_te(path: str) -> tuple[Header, H3TEConfig]
```

「知っているテンソル名」の正規表現の一覧（DiT）: `blocks.\d+.(attn.(qkv_proj|out_proj).(weight|bias|weight_scale|comfy_quant)|attn.(q_norm|k_norm).weight|mlp.(fc1|fc2).(weight|bias|weight_scale|comfy_quant)|norm1.weight|norm2.weight|adaln_proj.linear.(weight|bias))`、`token_refiner.blocks.\d+.*`、`token_refiner.final_norm.weight`、`condition_proj.(weight|bias)`、`(video|audio)_patch_proj.(weight|bias)`、`final_layer.(norm.weight|adaln_proj.linear.(weight|bias)|(video|audio)_out.(weight|bias))`、`rope.inv_freq`、`adaln_t_table`。token_refiner の中身の名前は Comfy-Org のヘッダ（R2 §B・ノート §13.2）に合わせる。実ファイルのヘッダは `...\scratchpad\h3\dl\hdr\` に取得済み（R2 が置いた。無ければ `sft_quant_format.read_header` で `models/MiniMaxH3/...` から読む。取得が終わるまでは合成ヘッダで作業）。

## 3. `engine_h3/quant_linear.py`（C1）

```python
class H3Int8Linear(nn.Module):
    """重みは素の int8 の Parameter、スケールは f32 [N,1] のバッファ。forward は comfy_kitchen.int8_linear。"""
    def __init__(self, in_features: int, out_features: int, *, bias: bool, spec: QuantSpec, bias_dtype: torch.dtype = torch.bfloat16)
        # 呼び出し側が with torch.device("meta") の中で作る
        self.weight = nn.Parameter(torch.empty(out, in, dtype=torch.int8), requires_grad=False)
        self.register_buffer("weight_scale", torch.empty(out, 1, dtype=torch.float32))   # 永続バッファ
        self.bias = nn.Parameter(torch.empty(out, dtype=bias_dtype), requires_grad=False) if bias else None
        self.spec = spec   # 属性(テンソルではない)
    def forward(self, x: Tensor) -> Tensor      # x [..., K] bf16(他の浮動小数は bf16 に揃える) → [..., N] bf16
        # comfy_kitchen.int8_linear(x2d, self.weight, self.weight_scale, self.bias, out_dtype=torch.bfloat16,
        #                           convrot=self.spec.convrot, convrot_groupsize=self.spec.convrot_groupsize)
        # bias の dtype が bf16 でなければ bf16 に揃えて渡す

def require_cuda_int8() -> dict
    # comfy_kitchen.list_backends()["cuda"] が available かつ capabilities に "int8_linear" を含むことを確かめる。
    # 満たさなければ RuntimeError(日本語の理由。unavailable_reason を含める)。
    # 続けて comfy_kitchen.disable_backend("triton") と disable_backend("eager") を呼ぶ(fail loud)。
    # 返り値は list_backends() の辞書(ログと done に使う)

def swap_in_int8_linears(root: nn.Module, quant: dict[str, QuantSpec], *, bias_dtype_of: Callable[[str], torch.dtype]) -> int
    # root の中の dotted 名(例 "blocks.0.attn.qkv_proj")の nn.Linear を H3Int8Linear(meta) に差し替える。返り値は差し替えた数。
    # 名前が見つからない / nn.Linear でない は RuntimeError
```

state_dict のキー: `{層}.weight`（I8）・`{層}.weight_scale`（F32 [N,1]）・`{層}.bias`。ファイルの `{層}.comfy_quant` は読み込み側が捨てる。

## 4. `engine_h3/rope.py`（C1）

```python
def build_coords(*, text_len: int, latent_frames: int, grid_h: int, grid_w: int, audio_latents: int, cond_frames: int = 0) -> Tensor
    # 返り値 [S, 3] float64。行の並びは系列と同じ [text | cond | audio | video]。
    # text: t=0..L-1, h=w=0 / video: t は L から 5/3×(1,4,4,4,4) 周期で累積、h,w は縦横比で正規化した格子 ×32 /
    # audio: t = L + k (k=0..A-1) を ch0, ch1 の順に 2 回、h=0、w は格子の最小(ch0)と最大(ch1) /
    # cond(第 2 弾): t = L、h,w は video と同じ格子。DiffSynth pipelines/minimax_h3_audio_video.py:586-709 と diffusers before_denoise.py:36-133 を照合する
def rope_tables(coords: Tensor, inv_freq: Tensor) -> tuple[Tensor, Tensor]
    # cos, sin: float32 [S, 96]。角度は float32 で計算(座標の float64 は作るときだけ)
def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor
    # x [S, heads, 128]。先頭 96 次元を rotate_half で回し、残り 32 はそのまま
```

## 5. `engine_h3/dit.py`（C1）

```python
@dataclass(frozen=True)
class H3SeqLayout:
    text: slice; cond: slice; audio: slice; video: slice; total: int
    # [text | cond | audio | video]。T2V は cond = slice(L, L)(空)

class H3DiT(nn.Module):
    def __init__(self, cfg: H3DiTConfig)      # 呼び出し側が meta の中で作る。枝刈り版だけ(cfg.pruned が False なら NotImplementedError)
    # 子: blocks(nn.ModuleList, 名前は必ず "blocks")・token_refiner・condition_proj・video_patch_proj・audio_patch_proj・final_layer・
    #     adaln_t_table(buffer F32 [G,R])・rope_inv_freq(buffer F32 [16]、ファイルの rope.inv_freq を入れる)
    # state_dict のキーは Comfy-Org のテンソル名(接頭辞を剥がしたもの)と一致させる(名前の写しを要らなくする)
    def refine_text(self, text_hidden: Tensor) -> Tensor
        # [L, 5120] bf16 → condition_proj(bf16) → token_refiner 2 ブロック → final_norm → [L, 5376] bf16。ループの前に 1 回だけ呼ぶ
    def embed_rows(self, text_refined: Tensor, audio_rows: Tensor, video_rows: Tensor, cond_rows: Tensor | None = None) -> Tensor
        # text [L,5376] bf16 / audio [2A,32] f32 / video [Nv,96] f32 / cond [Nc,96] f32 → tokens [S,5376] bf16
        # patch_proj は f32 で計算してから bf16 に落とす(精度の島)
    def forward(self, tokens: Tensor, *, layout: H3SeqLayout, t_unique: Tensor, time_index: Tensor, modality: Tensor,
                rope_cos: Tensor, rope_sin: Tensor, attention_backend: str = "sdpa") -> tuple[Tensor, Tensor]
        # tokens [S,5376] bf16 / t_unique [M] f32 (t = 1-σ の値。M は 1〜3) / time_index [S] long (0..M-1) /
        # modality [S] long (0 video / 1 text / 2 audio) / rope_cos, rope_sin [S,96] f32
        # 返り値 (v_video_rows [Nv,96] f32, v_audio_rows [Na,32] f32)。出力は「データ方向の速度」(x0 = x + σ v)
        # AdaLN: pos = clamp(t,0,1)*(G-1), i0 = min(floor(pos), G-2), w = pos - i0, e = lerp(table[i0], table[i0+1], w) → adaln_proj.linear(e) (SiLU なし, f32)
        #        行の添字 = time_index*3 + modality。最終層の添字は time_index だけ
        # attention: 1 本の系列・マスクなし。"sdpa" は F.scaled_dot_product_attention、"sage" は sageattention.sageattn(tensor_layout="HND")
        # 各ブロックの forward の引数はテンソルだけにする(block swap のラッパーが引数を GPU に移すため)。ブロック列は逐次呼び出し

# 行の並べ替えの道具(C1 が所有。pipeline が使う)
def patchify_video(latent: Tensor) -> Tensor     # [1,24,T',H/16,W/16] → [T'·(H/32)·(W/32), 96]。並びは DiffSynth に合わせる(t → h → w)
def unpatchify_video(rows: Tensor, latent_frames: int, grid_h: int, grid_w: int) -> Tensor   # 逆
def audio_to_rows(latent: Tensor) -> Tensor      # [2,32,A] → [2A, 32](ch0 の全時刻、続いて ch1)
def rows_to_audio(rows: Tensor, audio_latents: int) -> Tensor   # 逆
```

## 6. `engine_h3/scheduler.py`（C3）

```python
def sigma_schedule(n_steps: int, shift: float) -> Tensor    # f32 [n_steps+1]。base = linspace(1,0,n+1)[:-1] に σ = s·b/(1+(s-1)·b) を掛け、末尾に 0 を足す
def euler_step(x: Tensor, v: Tensor, sigma: float, sigma_next: float) -> Tensor   # x + (σ - σ_next)·v を f32 で計算し x の dtype で返す
```

## 7. VAE と出力（C2）

```python
class H3VideoVAE(nn.Module):
    @classmethod
    def from_file(cls, path: str, *, device, dtype=torch.bfloat16) -> "H3VideoVAE"   # meta で組み sft_load で入れ、device/dtype に移す
    latents_mean, latents_std: buffer f32 [24]       # __metadata__ の JSON から(F16 テンソルからではない)
    def encode(self, video: Tensor, *, tiled: bool = True) -> Tensor
        # [1,3,F,H,W] f32 [0,1] → ImageNet 正規化 → [1,24,T',H/16,W/16] f32 (mean/std で正規化済み・事後分布の平均)。F=1(先頭フレーム)も通す。第 1 弾の門では使わない
    def decode(self, latents: Tensor, *, tiled: bool = True) -> Tensor
        # [1,24,T',H/16,W/16] f32(正規化済み) → [1,3,17n+5,H,W] f32 [0,1]。T'=1 のときは 5 つ(tokens_chunk_size)に複製して 3 番目(frame_pre_padding の位置)の 1 フレームを返す(現行の ComfyUI と同じ。実測 PSNR 31.1 dB・2 つ複製は 21.3 dB。2026-10-17 変更)
class H3AudioVAE(nn.Module):
    @classmethod
    def from_file(cls, path: str, *, device) -> "H3AudioVAE"      # f32
    latents_mean, latents_std: buffer f32 [32]
    def decode(self, latents: Tensor) -> Tensor      # [2,32,A] f32(正規化済み) → [2, A*800] f32 (-1..1)
def write_mp4(path: str, frames: Tensor, *, fps: int, audio: Tensor | None, sample_rate: int = 32000) -> int
    # frames [F,H,W,3] uint8 CPU / audio [2,N] f32 CPU (-1..1) または None。libx264・crf 19・yuv420p・BT.709 を明示して変換(タグだけでない)・
    # 音声は s16 の AudioFrame → AudioResampler(fltp, stereo, 32000) → pts → flush の順で AAC。音声は映像の尺に切る。返り値は書いたバイト数
```

## 8. 読み込み・TE・パイプライン・ワーカー（C3）

```python
# engine_h3/sft_load.py
@dataclass
class LoadReport: loaded: int; skipped: tuple[str, ...]; seconds: float; bytes: int
def load_streaming(model: nn.Module, path: str, header, *, key_map: Callable[[str], str | None], skip_suffixes=(".comfy_quant",)) -> LoadReport
    # header.tensors の各キーを key_map でモデルのキーに写す(None は捨てる)。engine/sft_quant/sft_reader.read_tensors で 1 本ずつ読み、
    # 辞書に集めて model.load_state_dict(sd, strict=False, assign=True)。終わったら meta のまま残った Parameter/Buffer が無いことを検査し、
    # あれば RuntimeError(名前を列挙)。unexpected keys は警告ログ

# engine_h3/text_encoder.py
class H3TextEncoder:
    def __init__(self, te_path: str, tokenizer_dir: str, *, layers_on_gpu: int, device)
    def load(self) -> dict          # inspect_h3_te → Qwen3VLTextModel(config) を meta で組む(50 層, norm=Identity) → swap_in_int8_linears → load_streaming → GemmaLayerOffloadService(layers_on_gpu, device).install
    def encode(self, prompt: str) -> tuple[Tensor, Tensor]     # ([L,5120] bf16 on device, [L] long のモダリティタグ。T2V は全部 1)
    def release(self) -> None
    # トークナイズ: AutoTokenizer.from_pretrained(tokenizer_dir) を chat template なし・add_special_tokens=False で。<d> 等 7 語が 151669..151675 で語彙に入っていることを load で検査(入らなければ add_special_tokens で足す)
    # 設定: DiffSynth minimax_h3_text_encoder.py:20-72 の値(rope_scaling mrope_interleaved True / mrope_section [24,20,20] / rope_theta 5e6 / head_dim 128 / 64 heads / 8 kv heads / hidden 5120 / intermediate 25600 / rms 1e-6)
    # 名前の写し: "model.layers.N.*" → "layers.N.*"、"model.embed_tokens.*" → "embed_tokens.*"、"visual.*" と "lm_head"/"model.norm" は捨てる(第 1 弾)

# engine_h3/pipeline_h3.py
@dataclass
class H3Paths: transformer: str; text_encoder: str; video_vae: str; audio_vae: str; tokenizer_dir: str
@dataclass
class H3LoadOptions: blocks_on_gpu: int = 8; te_layers_on_gpu: int = 0; attention_backend: str = "sdpa"; block_swap_prefetch: bool = True; deterministic: bool = False
@dataclass
class H3GenerateRequest: prompt: str; seed: int; width: int; height: int; num_frames: int; output_path: str;
                         frame_rate: int = 24; num_inference_steps: int = DEFAULT_STEPS; shift_video: float = DEFAULT_SHIFT_VIDEO; shift_audio: float = DEFAULT_SHIFT_AUDIO
class H3Pipeline:
    def __init__(self, paths: H3Paths, options: H3LoadOptions, *, device, progress: Callable[[str, int, int], None], log: Callable[[str], None])
    def load(self) -> dict      # パスの存在 → inspect_h3_dit / inspect_h3_te / VAE のヘッダ → require_cuda_int8() → 構成の要約を返す(LOAD_OK のログと ready に使う)。重みは読まない
    def generate(self, req: H3GenerateRequest) -> dict
        # 検査: width,height % 32 == 0 / num_frames % 17 == 5 / frame_rate == 24 / steps >= 1 / shift > 0。違えば ValueError(日本語)
        # 段 1 encode: TE.load → encode → release(GPU と CPU を解放)。progress("encode", i, n)
        # 段 2 denoise: H3DiT を meta で組む → swap_in_int8_linears → load_streaming → ブロック以外を GPU へ、blocks は CPU のまま →
        #        BlockSwapService(blocks_on_gpu, device, hold_arenas=True); service.prefetch_requested = options.block_swap_prefetch; service.install(dit)
        #        ノイズ: torch.Generator(device="cpu").manual_seed(seed) で video (1,24,T',h,w) f32 → audio (2,32,A) f32 の順
        #        σ 列 2 本、各ステップ: t_v = 1-σ_v[i], t_a = 1-σ_a[i] → unique → time_index → forward → euler_step を映像と音声で別々に。progress("stage1_denoise", i+1, steps)
        #        終わったら service.teardown_prefetch() / release_installed() → DiT を del → gc / empty_cache
        # 段 3 decode: H3VideoVAE.from_file → decode → H3AudioVAE.from_file → decode → write_mp4。progress("decode", k, 3)
        # 返り値 = done の中身(§8 ワーカー)。phases は段ごとの秒数・RSS(psutil が無ければ省く)・VRAM ピーク

# engine_h3/worker.py  (engine25/worker.py の作法を写す。LTX 固有の分岐は持ち込まない)
# 起動: python -m engine_h3.worker (cwd = リポジトリ直下、PYTHONPATH=<root>)。ROOT を sys.path に挿入し os.chdir(ROOT)
# stdin: 1 行 1 JSON。op = "load" / "generate" / "shutdown"。load 前の未知 op は無視(ログ)、load 後の未知 op は error を返して生存
#   load:     {"op":"load","transformer_path","text_encoder_path","video_vae_path","audio_vae_path","tokenizer_dir","blocks_on_gpu","te_layers_on_gpu","attention_backend","block_swap_prefetch","deterministic"}
#   generate: {"op":"generate","prompt","seed","width","height","num_frames","frame_rate","num_inference_steps","shift_video","shift_audio","output_path"}  (steps/shift は省略可 → 既定値)
# stdout: "@@LTX@@" + JSON(1 行)。event = "ready" {sage_available: bool} / "progress" {stage, index, total} / "done" {...} / "error" {detail}
#   done: seed_used, sampler("euler"), num_inference_steps, shift_video, shift_audio, peak_vram_mb, peak_vram_reserved_mb (MiB), rss_peak_gib, seconds,
#         num_frames, frame_rate, width, height, size_bytes, attention_used, block_swap_prefetch_used ("off"/"on"/"on->off"), h3: {phases: {...}, quant_backends: {...}}
# stderr: 人向けログ。接頭辞 "[h3_worker]"
# --selftest-generate --transformer P --text-encoder P --video-vae P --audio-vae P --tokenizer-dir D --output P.mp4
#   [--prompt S --width 640 --height 384 --num-frames 124 --seed 42 --steps 20 --shift-video 12 --shift-audio 3 --rounds 1 --report P.json
#    --attention-backend sdpa|sage --blocks-on-gpu 8 --prefetch/--no-prefetch --te-layers-on-gpu 0]
#   _do_load / _do_generate を合成メッセージで直接呼び、_emit を差し替えて全イベントを記録し、mp4 の sha256 と done を JSON で書く。--rounds 2 で同一シードの一致を記録
```

## 9. 所有者

| 所有者 | ファイル |
|---|---|
| C1 | `engine_h3/quant_linear.py`・`engine_h3/dit.py`・`engine_h3/rope.py`・`outputs/h3-gate1/`（門の試験スクリプトと oracle の shim。git 管理外） |
| C2 | `engine_h3/video_vae.py`・`engine_h3/audio_vae.py`・`engine_h3/media_out.py` |
| C3 | `h3_format.py`・`engine_h3/__init__.py`（空）・`engine_h3/constants.py`・`engine_h3/sft_load.py`・`engine_h3/text_encoder.py`・`engine_h3/scheduler.py`・`engine_h3/pipeline_h3.py`・`engine_h3/worker.py`・`tests/test_h3_*.py` |
| 監督 | 仮想環境・重みの取得・門の実行の統制・コミット |

## 10. 重みと参考資料の所在

- 重み（取得中。監督が置く）: `models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors`・`models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors`・`models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer/`（tokenizer.json ほか 6 ファイル）・`models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors`・`models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors`
- 調査報告: `...\scratchpad\h3\R1_stack.md`（comfy-kitchen の API・2-2 追加）・`R2_model.md`（モデル構造・(3) 間違えやすい点 19 件）・`R3_inventory.md`（本製品の受け皿）
- 参照実装: `...\scratchpad\h3\src\DiffSynth-Studio\`・`...\scratchpad\h3\src\dx\`（diffusers の H3）・`...\scratchpad\h3\src\comfy-kitchen\`（公開 API の確認用）・`...\scratchpad\h3\src\ComfyUI\`（GPL。読むだけ）
- 本製品の前例: `engine25/worker.py`（プロトコルとセルフテスト）・`engine25/gguf_transformer.py:559-660, 1070-1130`（block swap の配線）・`engine/sft_quant/quant_service.py:274-330`（既存の量子化 Linear の形）・`engine/sft_quant/sft_reader.py`・`engine/transformer/block_swap_service.py`・`engine/gemma/layer_offload_service.py`・`engine/transformer/sage_attention_service.py`（`probe_sage`・`_sage_callable`）
