"""Pydantic schemas — the external API contract (spec ch.6).

Conditioning is now Phase-3 UNFROZEN: multiple keyframes (cap 5), arbitrary
``frame_idx`` (snapped server-side to the official ``0``-or-``8n+1`` latent grid
and clamped into range), and per-item ``strength``. ``num_pixel_frames`` and reference-video conditioning
remain out of scope. The OTHER constraints stay FROZEN as the final-form API so
that future frontends (AviUtl2, DaVinci Resolve) and later phases do not break:
÷64 generation resolution, 8n+1 frame counts, and the distilled 8-step / CFG=1.0
requirement. Do not relax those validators without revisiting the spec.

Resolution note: ``width``/``height`` are the *generation* size and must be a
multiple of **64** — the two-stage distilled pipeline generates stage-1 at half
resolution then 2x-upsamples, so the requested size must be divisible by 64
(``ltx_pipelines`` ``assert_resolution(is_two_stage=True)``). This is a
deliberate tightening from the earlier 32 rule (the mock never enforced it). Any
non-64 final display size (e.g. 960x540) is obtained via ``crop_output``.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from config import LimitsConfig

# Pydantic-declared DEFAULTS only (no config.yaml file I/O) — the single place
# api/models.py sources the V2V context_frames bounds from, so they can never
# silently drift from what config.py / GET /config advertise. See
# SourceVideoSpec.validate_context_frames.
_LIMITS_DEFAULTS = LimitsConfig()

# block_swap_prefetch's own default (S4, 2026-08-01: real-device gate G1-G7
# passed, owner confirmed "gate green -> default on"). Named (unlike most
# Field defaults in this file) because it is the SINGLE SOURCE this module's
# two Field(...) defaults below both read, AND the value mcp_server/tools/
# generate.py imports directly (mcp_server already sits in the same process/
# repo as api/, same precedent as services/*.py's existing `from api.models
# import ...`). gradio_ui/handlers.py deliberately does NOT import this — it
# keeps its own mirrored constant (`BLOCK_SWAP_PREFETCH_DEFAULT`) with an
# explicit cross-reference comment instead, since gradio_ui talks to the
# backend purely over HTTP. If this value ever changes again, update it here
# and move gradio_ui's mirror + mcp_server's import target in the same change.
BLOCK_SWAP_PREFETCH_DEFAULT = True


class CropOutput(BaseModel):
    width: int = Field(..., ge=32)
    height: int = Field(..., ge=32)


class ConditioningImage(BaseModel):
    image_id: str
    frame_idx: int = Field(0, ge=0)
    strength: float = Field(0.8, ge=0.0, le=1.0)
    crf: int | None = None


class LoraSpec(BaseModel):
    """One IC-LoRA adapter reference (Phase B, additive).

    ``name`` is a SERVER-SIDE registered adapter name (resolved to a safetensors
    path via ``config.model.ic_loras``) — NOT a filesystem path. Path-like names
    (containing ``/``, ``\\`` or ``..``) are rejected so a client can never point
    the server at an arbitrary file. ``strength`` is bounded like the other
    conditioning strengths (0 < s <= 2).
    """

    name: str = Field(..., min_length=1, max_length=200)
    strength: float = Field(1.0, gt=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_name_is_not_a_path(self) -> "LoraSpec":
        if "/" in self.name or "\\" in self.name or ".." in self.name:
            raise ValueError(
                "lora name must be a registered adapter name, not a path "
                "(no '/', '\\' or '..')"
            )
        return self


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field("", max_length=2000)

    # NAG（Normalized Attention Guidance, arXiv:2505.21179）— CFG を使わない
    # ネガティブプロンプト手法。蒸留パイプラインは guidance_scale=1.0 固定
    # （CFG 2パス denoise は不可）のため、代わりに cross-attention の出力を
    # 正プロンプト出力と負プロンプト出力から外挿・正規化してブレンドする。1本の
    # negative_prompt が映像・音声どちらの text cross-attention にも効く。既定値
    # 11.0 / 2.5 / 0.25 は kijai/ComfyUI-KJNodes の LTX2_NAG 実装と同一。
    nag_enabled: bool = False
    nag_scale: float = Field(11.0, ge=1.0, le=20.0)
    nag_tau: float = Field(2.5, ge=1.0, le=10.0)
    nag_alpha: float = Field(0.25, ge=0.0, le=1.0)

    # VSF（Value Sign Flip, arXiv:2508.10931）— NAG に続く第2の非CFGネガティブ
    # プロンプト手法。正負のコンテキストを連結し1回の attention で処理、負側の
    # V（value）だけを −α倍する（NAGのような正規化・ゲートは無い）。排除力は
    # NAGより強く、正プロンプト忠実度はNAGが上——性格違いのため方式を選べる。
    # neg_method で NAG/VSF を切り替える（nag_enabled が非CFGネガの共通マスター
    # トグルで、両方式をカバーする）。
    neg_method: Literal["nag", "vsf"] = "nag"
    # vsf_scale（α）既定1.5はWan実測1.7に近い論文準拠値。実機検証（2026-07-29）
    # でscale 15以上は8ステップ蒸留で収束崩壊と確定したため上限10（論文の実証
    # レンジ相当）。実用域は1.5〜5。
    vsf_scale: float = Field(1.5, ge=0.0, le=10.0)

    # ─── Acceleration（生成高速化）— ADDITIVE/optional。3つとも既定値のまま
    # 省略したリクエストは、ワーカーペイロードが導入前とバイト単位で同一になる。
    #
    # attention_backend: attention の実装を差し替える。"sdpa"（既定）は
    # PyTorch の scaled_dot_product_attention、"sage" は SageAttention 2.2.0
    # （INT8/FP8 量子化 attention カーネル）。実測で 720p end-to-end 約1.17倍・
    # stage2 約1.56倍、VRAM 増なし。
    # 【重要】sage は数値精度が sdpa と異なるため、**同一シードでも生成結果の
    # 細部が変わる**（バグではなく仕様）。厳密な再現性が要る場合は sdpa のまま
    # にすること。実際に使われた backend は metadata.json の attention_used に
    # 記録される（「設定したのに効いていない」を検出するため）。
    # 未導入環境で "sage" を指定した場合はジョブを落とさず sdpa へ降格する
    # （速度最適化であって生成結果の正しさの前提ではないので、NAG のような
    # fail-loud とは規律を変えている）。利用可否は GET /status の
    # acceleration.sage_available で確認できる。
    attention_backend: Literal["sdpa", "sage"] = "sdpa"

    # block_swap_prefetch: block swap（VRAM を節約するために transformer の
    # ブロックを CPU と GPU のあいだで出し入れする仕組み）の転送を、計算とは
    # 別の CUDA stream で先回りさせて待ち時間を隠す。あわせて GPU→CPU の
    # 退避コピーを廃止する（重みは推論中に一切変化しないので、CPU 側の正本を
    # 保持して GPU 側は捨てるだけでよい）。実測 768p/257f で約11〜13%短縮。
    # 【重要】attention_backend と違い、**生成結果は変わらない**（転送の
    # 方式だけを変えるので、同一シードならビット単位で同一になる）。
    # block swap が無効な設定（vram.block_swap=false / blocks_on_gpu=0 /
    # blocks_on_gpu が全ブロック数以上）では黙って no-op になる。実際に
    # 効いたかどうかは metadata.json の block_swap_prefetch_used で確認できる。
    # 利用可否は GET /status の acceleration.block_swap_prefetch_available。
    # 既定on（S4, 2026-08-01）: 実機ゲート（ビット一致＋VRAM）G1〜G7全PASSを
    # 条件にオーナーが確定した既定反転。offにすると従来の同期スワップになる。
    block_swap_prefetch: bool = BLOCK_SWAP_PREFETCH_DEFAULT

    # ─── モック2件（受理のみ・エンジン未消費）───
    # 以下2つは UI/API の枠だけ先に確定させたもので、**エンジンは一切読まない**。
    # 受け取っても生成は何も変わらない。ワーカーペイロードにも GET /status にも
    # 載せない（載せると「設定したのに効いていない」罠になる）。一方、
    # model_dump() 経由の metadata.json / GET /jobs の request には自然に現れる
    # ——two_stage_hq の pipeline と同じ既存前例で、exclude 等の細工はしない。
    #
    # fused_gguf_dequant_gemm: GGUF の逆量子化と GEMM を1カーネルに融合する案。
    fused_gguf_dequant_gemm: bool = False
    # vae_mode: VAE の実装選択（"prune_vaed" は枝刈り版 VAE デコーダ）。
    # 既存の vram.vae_tiling（VRAM 節約のためのタイル分割）とは**無関係**——
    # 名前が似ているだけで、こちらは VAE 実装そのものの差し替えを指す。
    vae_mode: Literal["default", "prune_vaed"] = "default"

    # 生成サイズ。必ず64の倍数（two-stage distilled）。最終表示サイズは crop_output で。
    width: int = Field(512, ge=256, le=4096)
    height: int = Field(320, ge=128, le=4096)

    # 最終MP4のクロップサイズ。None ならクロップしない。
    crop_output: CropOutput | None = None

    # 尺 cap は 20s(481f=8×60+1)@24fps まで許容。溢れ/低速/非実用は
    # クライアント UI 警告に委ねる（解像度別 spill-free は /config の
    # limits.spill_free_frames、実測根拠は RESOLUTION_DURATION_CAPABILITY.md §8.4/§8.6）。
    num_frames: int = Field(49, ge=9, le=481)
    frame_rate: float = Field(24.0, ge=1.0, le=60.0)
    num_inference_steps: int = Field(8, ge=1, le=100)
    guidance_scale: float = Field(1.0, ge=0.0, le=20.0)
    seed: int = -1
    pipeline: Literal["distilled", "two_stage_hq"] = "distilled"

    # 空配列なら T2V。1件以上なら I2V（マルチキーフレーム対応、cap 5）。
    # 各 frame_idx は validator で 0-or-8n+1 グリッドへスナップ＋範囲クランプされる。
    conditioning_images: list[ConditioningImage] = Field(default_factory=list)

    # IC-LoRA / style-LoRA (Phase B + S1, ADDITIVE/optional — a request omitting
    # both fields is byte-identical to before). ``loras`` are registered adapter
    # names (resolved server-side, never paths). A CONTROL adapter (union-control /
    # pixel-spatial-upscaler) requires a reference video; a STYLE/character adapter
    # does not — so the reference requirement is enforced per-adapter-kind at the
    # endpoint (api/generate.py), not as a shape-only cross-validation here. The
    # reverse still holds: ``reference_video_id`` set => at least one lora (below).
    # ``reference_video_id`` is obtained from POST /upload/video.
    loras: list[LoraSpec] = Field(default_factory=list)
    reference_video_id: str | None = None

    # IC-LoRA control adjustability (ADDITIVE/optional — a request omitting both
    # fields is byte-identical to before). Both only apply to a lora job (cross-
    # validated below to require ``loras``).
    #
    # ``conditioning_attention_strength`` — control adherence: how strictly the
    # output follows the IC-LoRA control signal (canny edges / pose skeleton).
    # Upstream name kept. None ⇒ omitted ⇒ the engine builds no attention wrapper
    # ⇒ byte-identical to today.
    conditioning_attention_strength: float | None = Field(None, ge=0.0, le=1.0)
    # ``reference_video_strength`` — the reference conditioning strength
    # (denoise_mask = 1 − s). Official guidance keeps this at 1.0; values < 1.0
    # can cause the reference to pop/bleed through into the output (official
    # tutorial warning) — exposed deliberately per user decision, default
    # unchanged (the runner still emits strength=1.0 when this is None).
    reference_video_strength: float | None = Field(None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_ltx_constraints(self) -> "GenerateRequest":
        if self.width % 64 != 0:
            raise ValueError("width must be a multiple of 64")
        if self.height % 64 != 0:
            raise ValueError("height must be a multiple of 64")
        if (self.num_frames - 1) % 8 != 0:
            raise ValueError("num_frames must be 8n+1")
        if self.crop_output is not None:
            if self.crop_output.width > self.width:
                raise ValueError("crop_output.width must be <= width")
            if self.crop_output.height > self.height:
                raise ValueError("crop_output.height must be <= height")
        if self.pipeline == "distilled":
            if self.num_inference_steps != 8:
                raise ValueError(
                    "distilled pipeline requires num_inference_steps=8 in Phase 1"
                )
            if self.guidance_scale != 1.0:
                raise ValueError(
                    "distilled pipeline requires guidance_scale=1.0 in Phase 1"
                )

        # Conditioning (Phase 3): multi-keyframe I2V, cap 5.
        if len(self.conditioning_images) > 5:
            raise ValueError("at most 5 conditioning images are supported")
        # frame_idx is fed to the engine's guide path (VideoConditionByKeyframeIndex)
        # as a RAW PIXEL RoPE offset (positions[:,0] += frame_idx, no ÷8), so it must
        # be a latent-aligned pixel. Two on-grid cases, per the official LTX-2 /
        # ComfyUI (LTXVAddGuide) convention:
        #   * frame_idx == 0  -> start frame, routed to the latent-replace path;
        #     left byte-identical (the engine also special-cases idx==0 causal_fix).
        #   * frame_idx  > 0  -> a keyframe/guide that must sit on a latent-frame
        #     START pixel = the 8n+1 grid. Snap via (f-1)//8*8+1 (ComfyUI
        #     get_latent_index) and clamp to [1, num_frames-8] — for num_frames=8m+1
        #     the last latent-frame start is num_frames-8 (e.g. 49 -> 41).
        # Snapping is a safety net — the UI may still send natural values. The ÷64
        # rule is spatial-only and unrelated.
        for image in self.conditioning_images:
            if image.frame_idx == 0:
                continue  # latent-replace path (start frame); byte-identical to today
            snapped = (image.frame_idx - 1) // 8 * 8 + 1
            image.frame_idx = max(1, min(snapped, self.num_frames - 8))

        # IC-LoRA reference requirement is now KIND-dependent (S1) and enforced at
        # the endpoint layer (api/generate.py), not here: a CONTROL adapter
        # (union-control / pixel-spatial-upscaler) needs a reference video, but a
        # STYLE/character adapter does not — so "loras require reference_video_id"
        # can no longer be decided from the request shape alone (it needs the
        # registry's per-adapter kind). The reverse still holds unconditionally: a
        # reference video only ever conditions a lora.
        if self.reference_video_id and not self.loras:
            raise ValueError(
                "reference_video_id requires at least one lora (the reference "
                "video only conditions an IC-LoRA)"
            )
        # IC-LoRA control-adjustability fields only apply to a lora job.
        if self.conditioning_attention_strength is not None and not self.loras:
            raise ValueError(
                "conditioning_attention_strength requires at least one lora "
                "(it only adjusts an IC-LoRA control signal)"
            )
        if self.reference_video_strength is not None and not self.loras:
            raise ValueError(
                "reference_video_strength requires at least one lora "
                "(it only adjusts an IC-LoRA reference conditioning)"
            )
        if self.nag_enabled and not self.negative_prompt.strip():
            raise ValueError("nag_enabled requires a non-empty negative_prompt")
        return self

    @property
    def generation_mode(self) -> Literal["t2v", "i2v"]:
        return "i2v" if self.conditioning_images else "t2v"


# Total-timeline pixel-frame cap. The masked AV-latent chain's stage-2 DECODE is
# always tiled, so that step stays VRAM-flat at any length. The UPSAMPLE step,
# however, runs over the WHOLE timeline in one GPU pass and does not free its
# intermediate latents, so it carries a VRAM term that DOES grow with total
# length. This cap bounds that term: it is a sanity ceiling = 24 clips × 481f
# (the max clip count × the frozen per-clip cap), documented so a UI cannot
# request an unbounded timeline.
MAX_CHAIN_TOTAL_PIXEL_FRAMES = 24 * 481  # 11544


class ChainClip(BaseModel):
    """One clip in a generate-chain request.

    ``prompt`` is optional: when absent the chain's base ``prompt`` is used
    (prompt propagation). ``num_frames`` must be 8n+1. Only clip 0 may carry
    conditioning images (minimal I2V start / keyframes) — clips 1..N are the
    later segments of one continuous masked AV-latent timeline (they inherit
    continuity from the previous segment's frozen overlap latents, not images).
    """

    prompt: str | None = Field(None, max_length=2000)
    num_frames: int = Field(49, ge=9, le=481)
    conditioning_images: list[ConditioningImage] = Field(default_factory=list)


class SourceVideoSpec(BaseModel):
    """Video-to-video continuation source (Phase V2V, ADDITIVE/optional).

    ``video_id`` is an existing upload from POST /upload/video (reuses the
    reference-video store). ``context_frames`` is the source *tail* span (pixel
    frames, 8n+1) that is VAE-encoded and frozen as clip-0's head; the delivered
    mp4 is the NEW part only (the context is trimmed off the front server-side).

    ``context_frames`` is bounded [25, config.limits.v2v_context_frames_max]. The
    max (currently 145, sourced from :class:`config.LimitsConfig` so this stays
    in lockstep with the value advertised via ``GET /config``) is a conservative
    v1 ceiling well inside the HARD invariant enforced by
    :func:`chain_math.compute_chain_layout`: the frozen video head
    (``n_ctx_v = (context_frames-1)//8+1``) must fit inside stage-2 TILE 0
    (``chain_math.STAGE2_V_TILE`` = 22 latents, i.e. <= 169 pixel frames /
    ``chain_math.px_from_v_latent(chain_math.STAGE2_V_TILE)``) because the
    variant-B hard-freeze only covers tile 0 — ``compute_chain_layout`` raises
    ValueError if that invariant is ever violated. Do NOT raise this cap without
    re-checking multi-tile freeze behaviour first.
    ``context_frames < clips[0].num_frames`` is cross-validated on the request.
    """

    video_id: str = Field(..., min_length=1)
    context_frames: int = Field(73)

    @model_validator(mode="after")
    def validate_context_frames(self) -> "SourceVideoSpec":
        cf = self.context_frames
        cf_min = _LIMITS_DEFAULTS.v2v_context_frames_min
        cf_max = _LIMITS_DEFAULTS.v2v_context_frames_max
        if cf < cf_min:
            raise ValueError(f"source_video.context_frames must be >= {cf_min}")
        if cf > cf_max:
            raise ValueError(
                f"source_video.context_frames must be <= {cf_max} "
                "(conservative v1 cap, config.limits.v2v_context_frames_max; "
                "see chain_math's stage-2 tile-fit invariant)"
            )
        if (cf - 1) % 8 != 0:
            raise ValueError("source_video.context_frames must be 8n+1")
        return self


class SourceAudioSpec(BaseModel):
    """Audio-to-video source (Phase A2V, ADDITIVE/optional).

    ``audio_id`` is an existing upload from POST /upload/audio. The uploaded
    waveform is VAE-encoded to audio latents that are HARD-frozen (full length,
    mask_value=0) across the chain timeline, so the generated video is driven to
    match the audio (lip-sync) while the ORIGINAL waveform is muxed back onto the
    output (no vocoder). Video length is authoritative: the audio is truncated to
    the timeline, never padded — a too-short upload is rejected up front (422
    SOURCE_AUDIO_TOO_SHORT). v1 exposes no trimming controls (audio_start_time /
    audio_max_duration) and is limited to a single clip (see the request
    validator). Mutually exclusive with ``source_video`` (A2V + V2V is out of
    v1 scope).
    """

    audio_id: str = Field(..., min_length=1)


class GenerateChainRequest(BaseModel):
    """A chain of clips assembled into ONE continuous masked AV-latent timeline.

    Additive to the frozen single-``/generate`` contract. Shares
    width/height/seed/frame_rate/pipeline across clips; each clip's effective
    prompt is its override if present else the global ``prompt``. Reuses the
    same FROZEN validators (÷64 resolution, 8n+1 frames, distilled 8-step /
    CFG=1.0) as :class:`GenerateRequest`.

    ARCHITECTURE (Phase 3 WP4): every clip is a stage-1 SEGMENT of one timeline;
    the segments are stage-1 generated with a video+audio latent tail carry over
    a ``overlap_frames`` (= K_v LATENT-frame) overlap, crossfaded into one latent,
    then refined in temporal tiles and decoded ONCE. There is no per-clip mp4 and
    no pixel-domain concat/trim — boundaries live inside the single decode, so
    the seams are continuous.
    """

    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field("", max_length=2000)

    # NAG（Normalized Attention Guidance, arXiv:2505.21179）— CFG を使わない
    # ネガティブプロンプト手法。詳細は GenerateRequest の同名フィールドを参照。
    # チェーンでは全クリップ・全ステージ共通で1本の negative_prompt が効く。
    nag_enabled: bool = False
    nag_scale: float = Field(11.0, ge=1.0, le=20.0)
    nag_tau: float = Field(2.5, ge=1.0, le=10.0)
    nag_alpha: float = Field(0.25, ge=0.0, le=1.0)

    # VSF（Value Sign Flip, arXiv:2508.10931）— 詳細は GenerateRequest の同名
    # フィールドを参照。チェーンでは全クリップ・全ステージ共通で1本の設定が効く。
    neg_method: Literal["nag", "vsf"] = "nag"
    vsf_scale: float = Field(1.5, ge=0.0, le=10.0)

    # Acceleration（生成高速化）— 詳細は GenerateRequest の同名フィールドを参照。
    # チェーンでは全クリップ・全ステージ共通で1つの設定が効く。sage 有効時は
    # 同一シードでも生成結果の細部が変わる点、モック2件（fused_gguf_dequant_gemm /
    # vae_mode）が受理のみでエンジン未消費である点、vae_mode が既存 vae_tiling と
    # 無関係である点も、すべて GenerateRequest と同じ。
    attention_backend: Literal["sdpa", "sage"] = "sdpa"
    # block_swap_prefetch: 詳細は GenerateRequest の同名フィールドを参照。
    # 既定on（S4, 2026-08-01）。offにすると従来の同期スワップになる。
    block_swap_prefetch: bool = BLOCK_SWAP_PREFETCH_DEFAULT
    fused_gguf_dequant_gemm: bool = False
    vae_mode: Literal["default", "prune_vaed"] = "default"

    width: int = Field(512, ge=256, le=4096)
    height: int = Field(320, ge=128, le=4096)
    crop_output: CropOutput | None = None

    frame_rate: float = Field(24.0, ge=1.0, le=60.0)
    num_inference_steps: int = Field(8, ge=1, le=100)
    guidance_scale: float = Field(1.0, ge=0.0, le=20.0)
    seed: int = -1
    pipeline: Literal["distilled", "two_stage_hq"] = "distilled"

    # Continuity (Phase 3 WP4): overlap = K_v LATENT frames shared between
    # consecutive stage-1 segments (the previous segment's tail is copied into
    # the next segment's head and frozen at ``overlap_strength``). K_v=3 is the
    # spike-validated default; must be < every clip's stage-1 latent-frame count.
    overlap_frames: int = Field(3, ge=1, le=8)
    overlap_strength: float = Field(0.5, ge=0.0, le=1.0)

    # 1..24 clips. WITHOUT source_video the floor is 2 (a single clip is just
    # /generate) — enforced explicitly in the model_validator so the old
    # rejection is preserved. WITH source_video a single clip is allowed (the
    # frozen source head IS the "previous segment"). Field floor is 1 so the
    # source path validates; capped so the timeline stays within
    # MAX_CHAIN_TOTAL_PIXEL_FRAMES.
    clips: list[ChainClip] = Field(..., min_length=1, max_length=24)

    # Video-to-video continuation (Phase V2V, ADDITIVE/optional — a request
    # omitting this field is byte-identical to before). When set, the tail of an
    # uploaded source video is frozen as clip-0's head; see :class:`SourceVideoSpec`.
    source_video: SourceVideoSpec | None = None

    # Audio-to-video (Phase A2V, ADDITIVE/optional — a request omitting this field
    # is byte-identical to before). When set, an uploaded audio track is frozen as
    # the chain's audio latent and the video is generated to match it; see
    # :class:`SourceAudioSpec`. Mutually exclusive with ``source_video``.
    source_audio: SourceAudioSpec | None = None

    # Style/character IC-LoRA (ADDITIVE/optional — a request omitting this field is
    # byte-identical to before). Same ``LoraSpec`` type/validation as
    # ``GenerateRequest.loras``; the strengths apply uniformly to EVERY clip and
    # every stage of the chain (no per-clip strengths in v1 — owner decision).
    # A2V (source_audio) and V2V continuation (source_video) may be combined with
    # loras (no exclusivity guard).
    loras: list[LoraSpec] = Field(default_factory=list)

    # Reference-video CONTROL IC-LoRA (Phase C chain support, ADDITIVE/optional,
    # ALPHA scope — owner decision 2026-07-11): a chain MAY now carry a
    # ``reference_video_id`` like a single ``/generate``, but ONLY when the chain
    # is exactly 1 clip in v1 (a per-clip reference video is out of scope, demoted
    # to a later research item). Mutually exclusive with ``source_video``: the
    # frozen V2V source head and a reference-conditioned control adapter would
    # otherwise compete for clip 0's head. Same type/bounds as
    # ``GenerateRequest.reference_video_id`` / ``conditioning_attention_strength``
    # / ``reference_video_strength`` (see there for field-level rationale); the
    # control-vs-style adapter kind check still needs the registry, so it stays at
    # the endpoint (api/generate_chain.py), mirroring the single-generate check.
    reference_video_id: str | None = None
    conditioning_attention_strength: float | None = Field(None, ge=0.0, le=1.0)
    reference_video_strength: float | None = Field(None, ge=0.0, le=1.0)

    # Chunked-upsample opt-in (ADDITIVE/optional — a request omitting this field
    # is byte-identical to before). When True the engine upsamples the assembled
    # stage-1 timeline in temporal chunks (halo overlap + CPU offload) instead of
    # one whole-timeline GPU pass, trading time for a flat VRAM ceiling so long
    # 768p chains fit in 16GB; the default (False) keeps the existing one-pass
    # path untouched (owner decision: off = zero regression).
    chunked_upsample: bool = False

    @model_validator(mode="after")
    def validate_chain_constraints(self) -> "GenerateChainRequest":
        if self.width % 64 != 0:
            raise ValueError("width must be a multiple of 64")
        if self.height % 64 != 0:
            raise ValueError("height must be a multiple of 64")
        if self.crop_output is not None:
            if self.crop_output.width > self.width:
                raise ValueError("crop_output.width must be <= width")
            if self.crop_output.height > self.height:
                raise ValueError("crop_output.height must be <= height")
        if self.pipeline == "distilled":
            if self.num_inference_steps != 8:
                raise ValueError(
                    "distilled pipeline requires num_inference_steps=8 in Phase 1"
                )
            if self.guidance_scale != 1.0:
                raise ValueError(
                    "distilled pipeline requires guidance_scale=1.0 in Phase 1"
                )

        # A2V + V2V are mutually exclusive (v1 scope — do not mix an uploaded
        # continuation video with an uploaded driving audio). Rejected up front.
        if self.source_audio is not None and self.source_video is not None:
            raise ValueError(
                "source_audio and source_video are mutually exclusive "
                "(A2V and V2V cannot be combined in v1)"
            )

        # A2V is limited to EXACTLY one clip in v1 (multi-clip audio window split
        # is out of scope). The single frozen audio latent spans the one clip.
        if self.source_audio is not None and len(self.clips) != 1:
            raise ValueError("source_audio requires exactly 1 clip in v1")

        # Clip-count floor: WITHOUT a source (video OR audio) OR a reference-video
        # control adapter, a chain needs >= 2 clips (a single clip is just
        # /generate) — preserve the pre-V2V rejection. WITH a source_video the
        # frozen source head IS the prior segment, WITH a source_audio a single
        # clip is the whole timeline, and WITH reference_video_id the chain is
        # ALPHA-scoped to exactly 1 clip (enforced below) — so 1 clip is OK in all
        # three cases.
        if (
            self.source_video is None
            and self.source_audio is None
            and self.reference_video_id is None
            and len(self.clips) < 2
        ):
            raise ValueError("chain requires at least 2 clips")

        # V2V continuation cross-validation (all 422 at request time):
        if self.source_video is not None:
            cf = self.source_video.context_frames
            clip0 = self.clips[0].num_frames
            if cf >= clip0:
                raise ValueError(
                    f"source_video.context_frames ({cf}) must be < clips[0].num_frames "
                    f"({clip0}) so a NEW tail remains to generate"
                )
            # The frozen source head occupies clip-0's first latent frames, so the
            # start-frame slot is taken by the source — clip 0 cannot also carry
            # conditioning images.
            if self.clips[0].conditioning_images:
                raise ValueError(
                    "source_video is mutually exclusive with clips[0].conditioning_images "
                    "(the source tail already occupies clip 0's frozen head)"
                )

        for i, clip in enumerate(self.clips):
            if (clip.num_frames - 1) % 8 != 0:
                raise ValueError(f"clips[{i}].num_frames must be 8n+1")
            # overlap_frames (K_v LATENT) must fit inside EVERY clip's stage-1
            # latent-frame count = (num_frames - 1)//8 + 1 (each segment either
            # provides or receives the K_v-frame overlap).
            stage1_frames = (clip.num_frames - 1) // 8 + 1
            if self.overlap_frames >= stage1_frames:
                raise ValueError(
                    f"overlap_frames ({self.overlap_frames}) must be < "
                    f"clips[{i}] stage-1 latent frames ({stage1_frames})"
                )
            # Only clip 0 may carry conditioning images.
            if i > 0 and clip.conditioning_images:
                raise ValueError(
                    "only clip 0 may carry conditioning_images (later clips are "
                    "later segments of one continuous timeline)"
                )
            if len(clip.conditioning_images) > 5:
                raise ValueError(
                    f"clips[{i}]: at most 5 conditioning images are supported"
                )
            for image in clip.conditioning_images:
                if image.frame_idx == 0:
                    continue
                snapped = (image.frame_idx - 1) // 8 * 8 + 1
                image.frame_idx = max(1, min(snapped, clip.num_frames - 8))

        # Total-timeline geometry: sum of pixel frames minus the shared overlaps.
        # Delegated to the shared pure-Python chain_math so the validator, the
        # engine and the metadata agree; it also raises on a degenerate audio
        # overlap (clips too short for a continuous crossfade).
        import chain_math
        try:
            layout = chain_math.compute_chain_layout(
                [c.num_frames for c in self.clips], self.frame_rate,
                kv=self.overlap_frames,
                source_context_px=(
                    self.source_video.context_frames if self.source_video else None
                ),
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if layout.total_px > MAX_CHAIN_TOTAL_PIXEL_FRAMES:
            raise ValueError(
                f"chain total timeline {layout.total_px} pixel frames exceeds the "
                f"cap {MAX_CHAIN_TOTAL_PIXEL_FRAMES} (reduce clip count or lengths)"
            )

        # Reference-video CONTROL IC-LoRA (alpha, clips=1 only): mirrors
        # GenerateRequest.validate_ltx_constraints (api/models.py:173-188) plus the
        # v1 clip-count cap and the V2V exclusivity below. The reverse still holds
        # unconditionally: a reference video only ever conditions a lora.
        if self.reference_video_id and not self.loras:
            raise ValueError(
                "reference_video_id requires at least one lora (the reference "
                "video only conditions an IC-LoRA)"
            )
        if self.reference_video_id and len(self.clips) != 1:
            raise ValueError("reference_video_id requires exactly 1 clip in v1")
        # IC-LoRA control-adjustability fields only apply to a lora job.
        if self.conditioning_attention_strength is not None and not self.loras:
            raise ValueError(
                "conditioning_attention_strength requires at least one lora "
                "(it only adjusts an IC-LoRA control signal)"
            )
        if self.reference_video_strength is not None and not self.loras:
            raise ValueError(
                "reference_video_strength requires at least one lora "
                "(it only adjusts an IC-LoRA reference conditioning)"
            )
        # V2V continuation freezes clip 0's head from the source tail; a
        # reference-conditioned control adapter also drives conditioning at clip 0.
        # Disallow combining them (mirrors the source_audio/source_video
        # exclusivity above) rather than defining a precedence rule between the
        # two clip-0 heads.
        if self.reference_video_id and self.source_video is not None:
            raise ValueError(
                "reference_video_id and source_video are mutually exclusive "
                "(a control adapter's reference conditioning would compete with "
                "the frozen V2V source head at clip 0)"
            )
        if self.nag_enabled and not self.negative_prompt.strip():
            raise ValueError("nag_enabled requires a non-empty negative_prompt")
        return self

    def clip_prompt(self, index: int) -> str:
        """Effective prompt for clip ``index`` (override else global base)."""
        override = self.clips[index].prompt
        return override if override else self.prompt

    def to_clip_request(self, index: int) -> "GenerateRequest":
        """Build the per-clip :class:`GenerateRequest` (clip 0 keeps its images).

        LIVE PATH: ``services/job_store.py:135`` (``create_chain_if_idle``) calls
        this on every chain job creation, and the result is re-validated as a
        ``GenerateRequest`` and stored as ``JobRecord.request``. Any field added
        to ``GenerateChainRequest`` that is not transcribed here silently drops
        out of the stored/serialized request for a chain job — omitting the nag
        fields would make chain creation 500 (nag_enabled True + empty
        negative_prompt would fail GenerateRequest's own validator).

        The acceleration fields (``attention_backend``, ``block_swap_prefetch``,
        and the two mock fields ``fused_gguf_dequant_gemm`` / ``vae_mode``) are
        transcribed for the same reason: they do not fail validation when
        dropped, so an omission would silently mis-report a chain job's
        reproducibility metadata (GET /jobs' ``request`` and metadata.json would
        claim sdpa/default for a sage chain).
        The chain's OWN worker payload is built from the chain request, not from
        this per-clip copy — this transcription only feeds the stored record.
        """
        clip = self.clips[index]
        return GenerateRequest(
            prompt=self.clip_prompt(index),
            negative_prompt=self.negative_prompt,
            nag_enabled=self.nag_enabled,
            nag_scale=self.nag_scale,
            nag_tau=self.nag_tau,
            nag_alpha=self.nag_alpha,
            neg_method=self.neg_method,
            vsf_scale=self.vsf_scale,
            attention_backend=self.attention_backend,
            block_swap_prefetch=self.block_swap_prefetch,
            fused_gguf_dequant_gemm=self.fused_gguf_dequant_gemm,
            vae_mode=self.vae_mode,
            width=self.width,
            height=self.height,
            crop_output=None,  # crop is applied once, on the final concat.
            num_frames=clip.num_frames,
            frame_rate=self.frame_rate,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            seed=self.seed,
            pipeline=self.pipeline,
            conditioning_images=clip.conditioning_images if index == 0 else [],
        )


class UploadImageResponse(BaseModel):
    image_id: str
    original_filename: str
    stored_path: str
    width: int
    height: int
    content_type: str


class UploadVideoResponse(BaseModel):
    video_id: str
    original_filename: str
    stored_path: str
    content_type: str
    size_bytes: int
    # True only when the optional trim_start_sec/trim_duration_sec query
    # arguments were supplied AND the cut actually succeeded. Additive: older
    # clients simply ignore it, and an upload without trim arguments always
    # reports False.
    trimmed: bool = False


class UploadAudioResponse(BaseModel):
    audio_id: str
    original_filename: str
    stored_path: str
    content_type: str
    size_bytes: int


class JoinRequest(BaseModel):
    """POST /jobs/{job_id}/join body (V2V, ADDITIVE — new endpoint only).

    Server-side join of a completed V2V job's continuation (``output.mp4``, the
    NEW part only) back onto its uploaded source video, producing ``joined.mp4``
    next to the job output. GPU-free (ffmpeg only) and independent of the
    single-GPU-job guard.

    ``audio_smoothing`` selects the audio treatment at the junction:

    * ``True`` (default) — crossfade: a true overlapped equal-power crossfade
      via the engine's ``<stem>_audio_handle.wav`` sidecar when the job has one,
      else the no-handle fade-pair (see ``services/video_io.join_v2v`` and
      Docs/V2V_AUDIO_JOIN_RESEARCH.md).
    * ``False`` — hard concat (no fades). Kept for parity/testing; the GUI only
      exposes the smoothed path.

    ``handle_crossfade_ms`` applies to the handle true-crossfade only; it is an
    audio-only acrossfade (the video is always a hard cut at the seam). The
    default is 300 ms (F5, G3 visual/audition gate: 150 ms — the original
    VERIFICATION_LOG §24.7 sweet spot — left the seam slightly audible on real
    content; the GUI offers 150/300/500). All fields are optional; an empty
    body ``{}`` gives the default smoothed join.
    """

    audio_smoothing: bool = True
    handle_crossfade_ms: int = Field(300, ge=0, le=2000)
    # V2V Join tail-keep: keep only this many trailing seconds of the uploaded
    # source before concatenating the continuation, so the join delivers "the
    # last N seconds of the original + the new part". 0 is the explicit request
    # to join the source at full length. When the source is already this short
    # (or shorter) it is joined at full length unchanged.
    source_tail_seconds: float = Field(5.0, ge=0.0)


class JoinResponse(BaseModel):
    """POST /jobs/{job_id}/join result (synchronous 200).

    ``join_mode`` is the treatment actually applied: ``handle_crossfade`` |
    ``fade_pair`` | ``hard_concat`` | ``video_only`` (either side had no audio).
    ``source_normalized`` reports whether the uploaded source needed an ffmpeg
    normalization pass (scale + center-crop + fps resample) to match the
    continuation's resolution/fps before joining.
    """

    job_id: str
    joined_path: str
    join_mode: str
    source_normalized: bool
    source_lufs: float | None = None
    continuation_lufs_before: float | None = None
    fade_ms_applied: int = 0
    handle_crossfade_ms_applied: int = 0
    handle_context_seconds: float | None = None
    loudness_matched: bool = False
    # V2V Join tail-keep: seconds of the source dropped from its head by the
    # tail-keep trim (= source full duration - kept duration), i.e. the offset at
    # which the continuation begins in the joined timeline. 0.0 when no trim ran
    # (source_tail_seconds=0 or source already at/under that length).
    trimmed_source_seconds: float = 0.0
    # The source's measured fps (ffprobe). None when it could not be probed.
    source_fps: float | None = None


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str


class GenerateChainResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    num_clips: int


class JobResult(BaseModel):
    video_url: str
    duration_seconds: float
    resolution: str
    file_size_bytes: int
    generation_time_seconds: float
    seed_used: int
    output_path: str
    metadata_path: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    current_step: int | None
    total_steps: int | None
    # F3 (G3 feedback, ADDITIVE): pipeline phase of the latest progress event
    # ("encode" / "stage1_denoise" / "stage2_denoise" / "stage1" / "tile" /
    # "decode" / "denoise"). None when the backend has not reported one (mock
    # milestones, queued jobs, pre-F2 workers) — consumers must treat unknown
    # values as "no label".
    stage: str | None = None
    # Chain clip progress (ADDITIVE, same discipline as ``stage``): 1-based
    # index of the clip (stage-1 segment) the chain is currently denoising and
    # the total clip count. Set only by chain jobs on real backends (the worker
    # reports the segment position); ``clip`` retains its last value through
    # the later whole-timeline stages (stage-2 tiles / decode), so clip ==
    # clip_count reads as "all clips are through stage 1". None for single
    # generates, queued jobs, mock milestones, and pre-F2 workers.
    clip: int | None = None
    clip_count: int | None = None
    # V2V (ADDITIVE): ``is_v2v`` is True when the job is a chain continuation of
    # an uploaded source video (has a ``source_video`` spec) — the only jobs
    # POST /jobs/{id}/join accepts. ``joined`` reports whether a ``joined.mp4``
    # currently exists next to the job output (i.e. a join has been run and not
    # deleted); it is False unless the responder resolved the output dir.
    is_v2v: bool = False
    joined: bool = False
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None
    request: GenerateRequest
    result: JobResult | None
