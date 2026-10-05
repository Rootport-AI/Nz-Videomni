"""Pydantic schemas — the external API contract (spec §6).

Conditioning accepts multiple keyframes (cap ``MAX_CONDITIONING_IMAGES``),
arbitrary ``frame_idx`` (snapped server-side to the official ``0``-or-``8n+1``
latent grid and clamped into range), and per-item ``strength``.
``num_pixel_frames`` (multi-frame keyframe conditioning) is not part of the
request. The OTHER constraints are FROZEN as the final-form API so that
frontends (the AviUtl2 plugin, and any other client of the same API such as
a DaVinci Resolve script) do not break: ÷64 generation resolution, 8n+1 frame
counts, and the distilled 8-step / CFG=1.0 requirement. Do not relax those
validators without revisiting the spec.

Resolution note: ``width``/``height`` are the *generation* size and must be a
multiple of **64** — the two-stage distilled pipeline generates stage-1 at half
resolution then 2x-upsamples, so the requested size must be divisible by 64
(``ltx_pipelines`` ``assert_resolution(is_two_stage=True)``). Any non-64 final
display size (e.g. 960x540) is obtained via ``crop_output``.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from config import LimitsConfig, MAX_CONDITIONING_IMAGES

# Pydantic-declared DEFAULTS only (no config.yaml file I/O) — the single place
# api/models.py sources the V2V and end-source context_frames bounds from, so
# these validators judge by config.py's defaults. GET /config advertises the
# values read from config.yaml, so a config.yaml override of these keys changes
# what is advertised but not what is validated. See
# SourceVideoSpec.validate_context_frames and EndSourceSpec.validate_end_source.
_LIMITS_DEFAULTS = LimitsConfig()

# Outpainting: the smallest keep-rectangle side we accept.
#
# The Laplacian-pyramid blend dilates its mask AFTER shrinking it to a 64px long
# side, so a radius of r pixels there costs ``r * canvas_long_side / 64`` real
# pixels — about 150px at a 1920px canvas with the official r=5. That dilation
# grows the *generated* region inward from every padded side, so a keep
# rectangle can be replaced by generated content even at 256: with bands on
# both the left and the right, a 256-wide keep rectangle on a 1920-wide canvas
# keeps 0 of its 256 columns at the stage-1 default r=5 (120 at the stage-2
# default r=2; 96 at r=5 on a 1024-wide canvas; 0 at r=2 on a 4096-wide one).
# 256 is therefore a flat floor that only stops the pathological case (a keep
# rectangle of a few dozen pixels); it is NOT a bound derived from the canvas
# size and the radius, and it does not guarantee that the kept pixels survive.
#
# ``engine/outpaint/canvas.py`` carries the same number as a defence-in-depth
# check. It is duplicated rather than shared because the app venv and the engine
# venv never import each other (services/engines/ltx/adapter.py's module
# docstring); the API is the enforcing layer, so drift can only ever make the
# engine stricter — a loud ValueError, never a silently wrong result.
OUTPAINT_MIN_KEEP_SIDE = 256

# Inpainting (``Docs/INPAINTING_DESIGN.md`` §6.2): the smallest side we accept
# for the SOURCE video.
#
# Numerically the same 256 as ``OUTPAINT_MIN_KEEP_SIDE`` above, and deliberately
# a DIFFERENT NAME with a different reason, because the two rules protect
# different things and could legitimately move apart. Outpainting's floor
# protects the KEPT rectangle from a mask dilation that eats inwards from the
# canvas edge. This one protects the CANVAS ITSELF: the canvas is the source
# rounded UP to a multiple of 128, and stage 1 runs at half of it through a VAE
# with a 32px patch stride — below 256 the stage-1 frame is 128px or less and
# the model has no context around the mask to work from at all.
#
# ``engine/inpaint/canvas.py`` carries the same number as a defence-in-depth
# check, duplicated rather than shared for the reason stated above: the app venv
# and the engine venv never import each other.
INPAINT_MIN_SOURCE_SIDE = 256

# The canvas grid. Both canvas sides must be multiples of 128. The video VAE's
# spatial stride is 32 in both engines and stage 1 runs at half the canvas
# size, which alone needs 64; 128 is the grid ``reference_resolution_invalid``
# already enforces for every reference-video job.
INPAINT_CANVAS_MULTIPLE = 128


def round_up_128(value: int) -> int:
    """Smallest multiple of :data:`INPAINT_CANVAS_MULTIPLE` that is >= ``value``.

    The app-side twin of ``engine.inpaint.canvas.round_up_128``. Duplicated for
    the reason ``INPAINT_MIN_SOURCE_SIDE`` is: the app venv and the engine venv
    never import each other. The API is the enforcing layer — it ffprobes the
    source and refuses a request whose canvas does not match — so drift could
    only ever make the engine stricter, where it surfaces as a loud ValueError
    rather than a silently wrong picture.

    >>> round_up_128(1280), round_up_128(1920), round_up_128(1080)
    (1280, 1920, 1152)
    """
    if value <= 0:
        raise ValueError(f"round_up_128 needs a positive size, got {value}")
    return -(-int(value) // INPAINT_CANVAS_MULTIPLE) * INPAINT_CANVAS_MULTIPLE

# block_swap_prefetch's own default (the real-device gate and the default-on
# decision are recorded in Docs/VERIFICATION_LOG.md §44). Named (unlike most
# Field defaults in this file) because it is the SINGLE SOURCE this module's
# two Field(...) defaults below both read, AND the value mcp_server/tools/
# generate.py imports directly (mcp_server already sits in the same process/
# repo as api/, same precedent as services/*.py's existing `from api.models
# import ...`). gradio_ui/handlers.py deliberately does NOT import this — it
# keeps its own mirrored constant (`BLOCK_SWAP_PREFETCH_DEFAULT`) with an
# explicit cross-reference comment instead, since gradio_ui talks to the
# backend purely over HTTP. If this value changes, update it here and move
# gradio_ui's mirror + mcp_server's import target in the same change.
BLOCK_SWAP_PREFETCH_DEFAULT = True

# keep_resident（モデル骨格のジョブ間常駐）の既定値。クライアントはこの定数と
# 違う値のときだけキーを送る（block_swap_prefetch と同じ扱い）。既定で有効に
# しないのはオーナー確定（メインメモリを大きく常駐で持つ機能を、メモリ量の
# 分からない環境で勝手に有効化しない。常駐量の実測は Docs/VERIFICATION_LOG.md
# §48）。名前付き定数にしている理由は BLOCK_SWAP_PREFETCH_DEFAULT と同じ：
# この2つの Field 既定と mcp_server/tools/generate.py の import 元を1箇所に
# 集約するため。gradio_ui/handlers.py は（HTTP越しのクライアントなので）
# import せず自前のミラー定数を持つ——変えるときは両方＋MCPを同じ変更で動かすこと。
KEEP_RESIDENT_DEFAULT = False

# keep_resident_embeddings（LTX 2.5 の埋め込み処理器のジョブ間常駐）の既定値。
# クライアントはこの定数と違う値のときだけキーを送る（上の定数と同じ扱い）。
# 名前付き定数にしている理由は上の2つと同じ：このファイルの2つの Field 既定と、
# mcp_server/tools/generate.py の import 元、そして gradio_ui/handlers.py の
# ミラー定数（Gradio は HTTP 越しのクライアントなので import せず自前で持つ）を
# 1箇所に集約するため——**変えるときは正本（ここ）＋gradio_ui のミラー＋MCP の
# import 元の3点を同じ変更で動かすこと**。
KEEP_RESIDENT_EMBEDDINGS_DEFAULT = False

# fused_gguf_dequant_kernel（GGUF 逆量子化の Triton 1カーネル化）の既定値。
# 実機ゲートと既定を有効にした判断は Docs/VERIFICATION_LOG.md §51。
# クライアントはこの定数と違う値のときだけキーを送る（block_swap_prefetch と
# 同じ扱い）。名前付き定数にしている理由は BLOCK_SWAP_PREFETCH_DEFAULT /
# KEEP_RESIDENT_DEFAULT と同じ：この2つの Field 既定と
# mcp_server/tools/generate.py の import 元を1箇所に集約するため。
# gradio_ui/handlers.py だけは（HTTP越しのクライアントなので）import せず自前の
# ミラー定数を持つ——**変えるときは正本（ここ）＋gradio_ui のミラー＋MCP の
# import 元の3点を同じ変更で動かすこと**。
FUSED_GGUF_DEQUANT_KERNEL_DEFAULT = True

# embed_mp4_metadata（完成した mp4 への生成条件の埋め込み）の既定値。
# A1111 の enable_pnginfo と同じく既定で書き込み、利用者が明示的に外す設計。
# クライアントはこの定数と違う値のときだけキーを送る（block_swap_prefetch と
# 同じ扱い）。名前付き定数にしている理由は上の定数群と同じ：このファイルの
# 2つの Field 既定と mcp_server/tools/generate.py の import 元を1箇所に集約する
# ため。gradio_ui/handlers.py は（HTTP越しのクライアントなので）import せず
# 自前のミラー定数を持つ——**変えるときは正本（ここ）＋gradio_ui のミラー＋
# MCP の import 元の3点を同じ変更で動かすこと**。
EMBED_MP4_METADATA_DEFAULT = True


class CropOutput(BaseModel):
    width: int = Field(..., ge=32)
    height: int = Field(..., ge=32)


class ConditioningImage(BaseModel):
    image_id: str
    frame_idx: int = Field(0, ge=0)
    strength: float = Field(0.8, ge=0.0, le=1.0)
    crf: int | None = None


def _normalize_conditioning_images(
    images: list[ConditioningImage], num_frames: int, *, where: str = ""
) -> None:
    """Validate one request's keyframe list and snap it onto the latent grid, in place.

    frame_idx is fed to the engine's guide path (VideoConditionByKeyframeIndex)
    as a RAW PIXEL RoPE offset (positions[:,0] += frame_idx, no ÷8), so it must
    be a latent-aligned pixel. Two on-grid cases, per the official LTX-2 /
    ComfyUI (LTXVAddGuide) convention:

      * frame_idx == 0  -> start frame, routed to the latent-replace path;
        left byte-identical (the engine also special-cases idx==0 causal_fix).
      * frame_idx  > 0  -> a keyframe/guide that must sit on a latent-frame
        START pixel = the 8n+1 grid. Snap via (f-1)//8*8+1 (ComfyUI
        get_latent_index) and clamp to [1, num_frames-8] — for num_frames=8m+1
        the last latent-frame start is num_frames-8 (e.g. 49 -> 41).

    Snapping is a safety net — the UI may still send natural values. The ÷64
    rule is spatial-only and unrelated.

    門番は3段: 枚数（``MAX_CONDITIONING_IMAGES`` まで）→ スナップ（上のグリッド）
    → 重複（スナップ後に同じ位置へ落ちた2枚を弾く）。``where`` は連結リクエストが
    ``clips[i]: `` を前置するための接頭辞。
    """
    if len(images) > MAX_CONDITIONING_IMAGES:
        raise ValueError(
            f"{where}at most {MAX_CONDITIONING_IMAGES} conditioning images are supported"
        )

    claimed_by: dict[int, int] = {}  # スナップ後の位置 -> それを先に取った入力 frame_idx
    for image in images:
        raw = image.frame_idx
        if raw == 0:
            snapped = 0
        else:
            snapped = max(1, min((raw - 1) // 8 * 8 + 1, num_frames - 8))

        if snapped in claimed_by:
            if snapped == 0:
                raise ValueError(
                    f"{where}two conditioning images at frame_idx 0 "
                    "(the start frame can only be set once)"
                )
            raise ValueError(
                f"{where}conditioning_images: frame_idx {claimed_by[snapped]} and "
                f"{raw} both snap to frame {snapped} (positions must be distinct)"
            )

        claimed_by[snapped] = raw
        if raw != 0:
            image.frame_idx = snapped


class LoraSpec(BaseModel):
    """One IC-LoRA adapter reference.

    ``name`` is a SERVER-SIDE registered adapter name (resolved to a safetensors
    path via ``config.model.ic_loras`` registrations or files scanned from
    ``config.model.lora_dir``) — NOT a filesystem path. Path-like names
    (containing ``/``, ``\\`` or ``..``) are rejected so a client can never point
    the server at an arbitrary file. The bounds of ``strength`` are declared on
    its Field.
    """

    name: str = Field(..., min_length=1, max_length=200)
    strength: float = Field(1.0, gt=0.0, le=2.0)
    # audio_strength: 映像軸（strength）とは独立した音声軸の適用強度。省略
    # （None）なら音声側も strength に追従する。
    # 0 は音声側の重みを一切適用しない（=スキップ）——映像目的で訓練された
    # style LoRA の音声側差分が生成音声を壊す事例（雑音・音割れ）への対処。
    # strength と異なり 0 を許容する（gt=0.0 ではなく ge=0.0）。
    audio_strength: float | None = Field(None, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_name_is_not_a_path(self) -> "LoraSpec":
        if "/" in self.name or "\\" in self.name or ".." in self.name:
            raise ValueError(
                "lora name must be a registered adapter name, not a path "
                "(no '/', '\\' or '..')"
            )
        return self


class OutpaintSpec(BaseModel):
    """Canvas extension (outpainting). Reproduces the official Lightricks
    ``LTX-2.3_ICLoRA_Outpaint_Two_Stage_Distilled`` ComfyUI workflow.

    Geometry contract: ``GenerateRequest.width`` / ``height`` are the FINAL
    CANVAS, and the four pads are cut out of it — the keep rectangle is
    ``width - pad_left - pad_right`` by ``height - pad_top - pad_bottom`` and
    must equal the reference video's own resolution (the endpoint verifies this
    with ffprobe and rejects a mismatch). Deriving the canvas from the pads
    instead would give geometry two sources of truth, and the existing
    128-multiple rule already applies to ``width``/``height``, so the canvas is
    the natural place to anchor it. The source video is never resized.

    ``align_h`` / ``align_v`` are deliberately NOT part of this contract: the
    alignment choice is a UI affordance for *distributing* the pads, and pads do
    not determine an alignment uniquely, so the four numbers are both the more
    precise and the only necessary record.
    """

    pad_left: int = Field(0, ge=0, le=4096)
    pad_right: int = Field(0, ge=0, le=4096)
    pad_top: int = Field(0, ge=0, le=4096)
    pad_bottom: int = Field(0, ge=0, le=4096)

    # Laplacian-pyramid blend dilation for the two blends (after stage 1 at half
    # resolution, and after stage 2 at full resolution). The official workflow's
    # own values are 5 and 2 (nodes 5266 / 5226) and its note calls this "the
    # most important parameter".
    #
    # This IS surfaced in the UI. The AviUtl2 WebUI's Outpainting panel shows it
    # as a single "マスクブラー" (mask blur) slider over this very 0-15 stage-1
    # range, with stage 2 following at the workflow's own 5:2 ratio so the two
    # can never drift apart. Two things stay hidden behind that one control: the
    # numbers the user reads are the resulting band in full-resolution pixels
    # (the step count means nothing without a canvas size), and both keys are
    # omitted from the request entirely while they sit at these defaults — so a
    # server that predates the fields still accepts a default outpaint request.
    blend_dilation_stage1: int = Field(5, ge=0, le=15)
    blend_dilation_stage2: int = Field(2, ge=0, le=15)

    # Freeze the source video's own audio latent across both stages. The
    # official workflow does this and notes "Frozen audio helps guiding
    # outpainting to be consistent with the sounds in the video" (node 5392);
    # turning it off lets the model invent audio for the widened frame instead.
    freeze_source_audio: bool = True

    @property
    def total_pad(self) -> int:
        return self.pad_left + self.pad_right + self.pad_top + self.pad_bottom


class InpaintSpec(BaseModel):
    """Masked partial regeneration (inpainting). Design canon:
    ``Docs/INPAINTING_DESIGN.md`` §6.

    Repaints the white region of a mask video inside an existing clip, using the
    same In-Outpainting IC-LoRA and the same two-stage green-canvas machinery
    outpainting uses — the geometry is what differs.

    GEOMETRY CONTRACT. ``GenerateRequest.width`` / ``height`` are the CANVAS:
    the source video's own resolution rounded UP to a multiple of 128. The
    source sits at the canvas' top-left (0, 0) and the right/bottom bands are
    sentinel green, cut off again losslessly at the end of the job.

    **The pads are NOT fields here, deliberately.** The server derives them by
    ffprobing the reference video (``canvas − source``), which means the source
    file is the single source of truth for its own size. A pad field would be a
    second one, checkable against nothing, and a client that computed it wrong
    would get a silently mis-framed canvas instead of a 422. The same reasoning
    is why the delivered resolution is the source's rather than a request field:
    there is nothing for a caller to get wrong.

    THE WINDOW IS TWO VALUES, exactly as retake's is: ``window_start_sec`` plus
    ``GenerateRequest.num_frames``. Length is not defined twice. The start is on
    the MATERIAL's timeline (an AviUtl2 ribbon trimmed off the front has already
    had its offset subtracted by the caller), and the server cuts the window out
    of the upload before anything else happens.
    """

    # The mask video from POST /upload/video. White (>= 128) marks the pixels to
    # repaint; the binarisation happens on the receiving side, twice — once in
    # the ffmpeg filtergraph that paints the canvas and once when the engine
    # decodes the mask for the blend — at the same threshold both times.
    mask_video_id: str = Field(..., min_length=1)

    # Where the window starts on the SOURCE MATERIAL's own timeline.
    window_start_sec: float = Field(0.0, ge=0.0)

    # Laplacian-pyramid blend dilation for the two blends, the same knob and the
    # same 0-15 range outpainting exposes (``OutpaintSpec``). The AviUtl2 panel
    # exposes both as integer fields ("Stage-1" / "Stage-2", defaults from the
    # panel's ``DEFAULT_BLEND_DILATION_STAGE1`` / ``_STAGE2``) and always sends
    # them; the defaults here stand for any other client. They exist as fields
    # so a sweep on real hardware can vary them without a code change.
    blend_dilation_stage1: int = Field(5, ge=0, le=15)
    blend_dilation_stage2: int = Field(2, ge=0, le=15)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field("", max_length=2000)

    # NAG（Normalized Attention Guidance, arXiv:2505.21179）— CFG を使わない
    # ネガティブプロンプト手法。蒸留パイプラインは guidance_scale=1.0 固定
    # （CFG 2パス denoise は不可）のため、代わりに cross-attention の出力を
    # 正プロンプト出力と負プロンプト出力から外挿・正規化してブレンドする。1本の
    # negative_prompt が映像・音声どちらの text cross-attention にも効く。既定値
    # （下の nag_scale / nag_tau / nag_alpha の Field 既定）は kijai/ComfyUI-KJNodes
    # の LTX2_NAG 実装と同一。
    nag_enabled: bool = False
    nag_scale: float = Field(11.0, ge=1.0, le=20.0)
    nag_tau: float = Field(2.5, ge=1.0, le=10.0)
    nag_alpha: float = Field(0.25, ge=0.0, le=1.0)

    # VSF（Value Sign Flip, arXiv:2508.10931）— NAG と並ぶもう1つの非CFGネガティブ
    # プロンプト手法。正負のコンテキストを連結し1回の attention で処理、負側の
    # V（value）だけを −α倍する（NAGのような正規化・ゲートは無い）。排除力は
    # NAGより強く、正プロンプト忠実度はNAGが上——性格違いのため方式を選べる。
    # neg_method で NAG/VSF を切り替える（nag_enabled が非CFGネガの共通マスター
    # トグルで、両方式をカバーする）。
    neg_method: Literal["nag", "vsf"] = "nag"
    # vsf_scale（α）の既定はWan実測1.7に近い論文準拠値。上限は、8ステップ蒸留で
    # 大きな scale が収束崩壊した実機検証の記録に基づく（論文の実証レンジ相当）。
    # 既定と上限の値は下の Field が持つ。
    vsf_scale: float = Field(1.5, ge=0.0, le=10.0)

    # ─── Acceleration（生成高速化）— ADDITIVE/optional。各キーをワーカー
    # ペイロードへ載せる条件はエンジンのアダプタ（LTX 2.3 は
    # services/engines/ltx/adapter.py）が持ち、既定のリクエストが運ぶキーの集合は
    # tests/test_ltx_runner_payload.py の
    # test_default_payload_key_set_is_unchanged_by_acceleration が固定している。
    #
    # attention_backend: attention の実装を差し替える。"sdpa"（既定）は
    # PyTorch の scaled_dot_product_attention、"sage" は SageAttention
    # （INT8/FP8 量子化 attention カーネル。版はエンジンの仮想環境の固定ファイル
    # が持つ。LTX 2.3 は engine/venv-engine.freeze.txt）。速度と VRAM の実測は
    # Docs/VERIFICATION_LOG.md §43。
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
    # 退避コピーを行わない（重みは推論中に一切変化しないので、CPU 側の正本を
    # 保持して GPU 側は捨てるだけでよい）。短縮の実測は Docs/VERIFICATION_LOG.md §44。
    # 【重要】attention_backend と違い、**生成結果は変わらない**（転送の
    # 方式だけを変えるので、同一シードならビット単位で同一になる）。
    # block swap が組まれない構成（GPU に置くブロック数
    # vram.block_swap_blocks_on_gpu が全ブロック数以上）では黙って no-op になる。
    # 実際に効いたかどうかは metadata.json の block_swap_prefetch_used で確認できる。
    # 利用可否は GET /status の acceleration.block_swap_prefetch_available。
    # 既定は BLOCK_SWAP_PREFETCH_DEFAULT（既定を決めた実機ゲートの記録は
    # Docs/VERIFICATION_LOG.md §44）。off にすると同期スワップになる。
    block_swap_prefetch: bool = BLOCK_SWAP_PREFETCH_DEFAULT

    # keep_resident: モデルのCPU側「骨格」（各サブモデルの state_dict）を
    # ジョブ間で常駐させ、2回目以降の生成でディスクからの再マテリアライズを
    # 省く。wheel の StateDictRegistry をワーカーの ModelLedger に差し込む
    # 実装で、**生成結果は変わらない**（同一シードでのビット単位一致は
    # Docs/VERIFICATION_LOG.md §47.3）。効果は前処理（骨格の組み立て）の短縮で、
    # 実測は §47.3・§48。
    # 【重要】代償はメインメモリ：キャッシュ実体が常駐する（常駐量の実測は §48）。
    # **メモリ64GB以上を推奨**。足りない環境ではページアウトで denoise が逆に
    # 遅くなりうる。既定は KEEP_RESIDENT_DEFAULT（理由はその定数のコメント）。
    # ON→OFF→ON と戻した場合、OFF の時点でキャッシュを解放するので次の ON は
    # 全サブモデルの再ロードを1回だけ払い直す（仕様）。
    # 実際に効いたかどうかは metadata.json の keep_resident_used で確認できる
    # （"off" / "on" / "on->off"）。ワーカー側には安全ガードがあり、
    # 組み合わせによっては自動的に off へ降格する（engine/worker.py の
    # _resolve_keep_resident を参照）。GET /status には載せない（利用可否は
    # 環境依存ではなくメモリ量の問題で、サーバーからは判定できないため）。
    # 【エンジン差】上の実測・併用制限・自動降格はすべて LTX 2.3 のものである。
    # LTX 2.5（engine_family="ltx25"）でも効くが、**契約が同じだけで実装は別物**
    # ——2.3 が全サブモデルの骨格を抱えるのに対し、2.5 が常駐させるのは Gemma 4
    # テキストエンコーダの state dict ただ1つで、2本目以降のジョブが短くなる
    # （常駐量と短縮の実測は Docs/VERIFICATION_LOG.md §76）。2.5 側には併用の制限も
    # 自動降格も存在しないため、keep_resident_used は "on" / "off" の2値しか
    # 出ない（契約は3値のまま）。
    keep_resident: bool = KEEP_RESIDENT_DEFAULT

    # keep_resident_embeddings: LTX 2.5 の**埋め込み処理器**（プロンプトを読み
    # 取ったあとの内部表現を整える部品）の、CPU側 state dict ただ1本をジョブ間
    # で常駐させ、毎ジョブの GGUF 読み直しを省く。
    # 【重要】keep_resident と同じく、**生成結果は1バイトも変わらない**——
    # 同じ部品を作り直さずに使い回すだけなので、構築が速くなるだけである。
    # 代償はメインメモリの常駐（常駐量の実測の正本は Docs/VERIFICATION_LOG.md
    # §92）。LTX 2.5 の keep_resident（テキストエンコーダの state dict。常駐量は
    # §76）とは別々のスイッチで、併用したときのメインメモリ増分は**加算的**になる。
    # 実際に効いたかどうかは metadata.json の keep_resident_embeddings_used で
    # 確認できる（"on" / "off" の2値。2.5 側に自動降格の経路が無いため、
    # keep_resident と違って "on->off" は出ない）。GET /status には載せない
    # （keep_resident と同じ理由——利用可否は環境の能力ではなくメモリ量の
    # 問題で、サーバーからは判定できない）。
    # 【エンジン差】**向きがここまでのフィールドと逆である**。このフィールドが
    # 指す埋め込み処理器は LTX 2.5 にしか無い部品なので、非対応を宣言するのは
    # LTX 2.3 の側になる：engine_family="ltx" では unsupported_features に
    # keep_resident_embeddings が載り、true を送ると 422 FEATURE_UNSUPPORTED に
    # なる（断る表は services/engines/ltx/adapter.py の REJECT_TABLE・
    # CHAIN_REJECT_TABLE）。
    keep_resident_embeddings: bool = KEEP_RESIDENT_EMBEDDINGS_DEFAULT

    # fused_gguf_dequant_kernel: GGUF（K量子化 Q4_K/Q5_K/Q6_K）の逆量子化を
    # Triton の1カーネルに融合し、純 PyTorch 実装の多段テンソル演算を置き換える。
    # 逆量子化が1ジョブに占める時間の分解計測は Docs/VERIFICATION_LOG.md §50。
    # 【重要】block_swap_prefetch と同じく **生成結果は変わらない**（純 PyTorch
    # 実装とのビット一致を必須要件として実装・検証している）。Triton 不在・
    # カーネル例外・自己検証不一致のいずれでも黙って純 PyTorch 実装へ降格し、
    # 生成は落とさない。
    # 実際に効いたかどうかは metadata.json の fused_gguf_dequant_kernel_used で
    # 確認できる（"off" / "on" / "on->off"。"on->off" は「要求したが実際には
    # 適用されなかった」）。GET /status には載せない（keep_resident と同じ規律）。
    fused_gguf_dequant_kernel: bool = FUSED_GGUF_DEQUANT_KERNEL_DEFAULT

    # vae_mode: 映像VAE**デコーダ**の実装選択。"prune_vaed" は枝刈り版
    # （PrunaVAED）で、映像の復元が速くなる代わりに**出力品質がわずかに低下
    # する可能性がある**——そのため既定は恒久的に "default"（枝刈りしない
    # デコーダ）とする。既定と違うときだけワーカーへ送る加算的
    # コントラクトで、実際に何で復元したかは metadata.json の vae_mode_used
    # （LTX 2.3 では "off" / "on" / "on->off"）で確認できる。"on->off" は
    # 「枝刈りを頼んだが重みファイルが無かったので既定デコーダで完走した」。
    # LTX 2.5 は "prune_vaed" を断り（services/engines/ltx25/adapter.py の
    # REJECT_TABLE）、vae_mode_used には読み込んだデコーダの種類が入る。
    # GET /status には載せない（keep_resident と同じ規律——重みの有無は
    # 「環境の能力」とは性質が違う）。
    # 既存の vram.vae_tiling（VRAM 節約のためのタイル分割）とは**無関係**——
    # 名前が似ているだけで、こちらは VAE 実装そのものの差し替えを指す。
    # タイル設定はチャンネル幅に依存しないので枝刈り版でも一切変わらない。
    vae_mode: Literal["default", "prune_vaed"] = "default"

    # embed_mp4_metadata: 完成した output.mp4（と、この動画を結合した joined.mp4）
    # に、metadata.json と同じ JSON 文字列をコンテナの ``comment`` タグとして
    # 刻む。再多重化（再エンコード無し）で後付けする。埋め込みに
    # 失敗しても警告ログのみで、元の mp4 とジョブの完了は保たれる。mp4 を
    # 再エンコードするとタグは消える。読み出しは POST /utils/mp4-info。
    # false のときは何も刻まない（結合でも元動画のタグを引き継がない）。
    embed_mp4_metadata: bool = EMBED_MP4_METADATA_DEFAULT

    # 生成サイズ。必ず64の倍数（two-stage distilled）。最終表示サイズは crop_output で。
    width: int = Field(512, ge=256, le=4096)
    height: int = Field(320, ge=128, le=4096)

    # 最終MP4のクロップサイズ。None ならクロップしない。
    crop_output: CropOutput | None = None

    # 尺 cap は 20s(481f=8×60+1)@24fps まで許容。溢れ/低速/非実用は
    # クライアント UI 警告に委ねる（解像度別 spill-free は /config の
    # limits.spill_free_frames、実測の記録は Docs/COMFORT_LIMIT_TABLE.md）。
    num_frames: int = Field(49, ge=9, le=481)
    frame_rate: float = Field(24.0, ge=1.0, le=60.0)
    num_inference_steps: int = Field(8, ge=1, le=100)
    guidance_scale: float = Field(1.0, ge=0.0, le=20.0)
    seed: int = -1
    pipeline: Literal["distilled", "two_stage_hq"] = "distilled"

    # 空配列なら T2V。1件以上なら I2V（マルチキーフレーム対応、cap ``MAX_CONDITIONING_IMAGES``）。
    # 各 frame_idx は validator で 0-or-8n+1 グリッドへスナップ＋範囲クランプされる。
    conditioning_images: list[ConditioningImage] = Field(default_factory=list)

    # IC-LoRA / style-LoRA (ADDITIVE/optional). ``loras`` are registered adapter
    # names (resolved server-side, never paths). A CONTROL adapter (kind
    # "control" in services/lora_registry.py) requires a reference video; a
    # STYLE/character adapter does not — so the reference requirement is
    # enforced per-adapter-kind at the endpoint (api/generate.py), not as a
    # shape-only cross-validation here. The reverse still holds:
    # ``reference_video_id`` set => at least one lora (below).
    # ``reference_video_id`` is obtained from POST /upload/video.
    loras: list[LoraSpec] = Field(default_factory=list)
    reference_video_id: str | None = None

    # IC-LoRA control adjustability (ADDITIVE/optional). Both only apply to a
    # lora job (cross-validated below to require ``loras``).
    #
    # ``conditioning_attention_strength`` — control adherence: how strictly the
    # output follows the IC-LoRA control signal (canny edges / pose skeleton /
    # depth map).
    # Upstream name kept. None ⇒ omitted from the worker payload ⇒ the engine
    # builds no attention wrapper.
    conditioning_attention_strength: float | None = Field(None, ge=0.0, le=1.0)
    # ``reference_video_strength`` — the reference conditioning strength
    # (denoise_mask = 1 − s). Official guidance keeps this at 1.0; values < 1.0
    # can cause the reference to pop/bleed through into the output (official
    # tutorial warning) — exposed deliberately per user decision; when this is
    # None the runner emits strength=1.0.
    reference_video_strength: float | None = Field(None, ge=0.0, le=1.0)

    # Outpainting (ADDITIVE/optional). ``None`` ⇒ the worker payload carries no
    # ``outpaint`` key. See OutpaintSpec for the geometry contract.
    outpaint: OutpaintSpec | None = None

    # Inpainting (ADDITIVE/optional). ``None`` ⇒ the worker payload carries no
    # ``inpaint`` key. See InpaintSpec for the geometry contract.
    inpaint: InpaintSpec | None = None

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
                    "distilled pipeline requires num_inference_steps=8"
                )
            if self.guidance_scale != 1.0:
                raise ValueError(
                    "distilled pipeline requires guidance_scale=1.0"
                )

        # Conditioning: multi-keyframe I2V — 枚数 / スナップ / 重複の3段。
        _normalize_conditioning_images(self.conditioning_images, self.num_frames)

        # The IC-LoRA reference requirement is KIND-dependent and enforced at the
        # endpoint layer (api/generate.py), not here: a CONTROL adapter (kind
        # "control" in services/lora_registry.py) needs a reference video, but a
        # STYLE/character adapter does not — so "loras require reference_video_id"
        # cannot be decided from the request shape alone (it needs the registry's
        # per-adapter kind). The reverse still holds unconditionally: a reference
        # video only ever conditions a lora.
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

        # ── Outpainting ───────────────────────────────────────────────────────
        # One flat check per rule (no nesting): every failure names exactly what
        # the caller got wrong.
        if self.outpaint is not None:
            op = self.outpaint
            if not self.reference_video_id:
                raise ValueError(
                    "outpaint requires reference_video_id (the video whose "
                    "canvas is being extended)"
                )
            if op.total_pad == 0:
                raise ValueError(
                    "outpaint requires at least one non-zero pad (nothing would "
                    "be extended otherwise)"
                )
            if self.conditioning_images:
                raise ValueError(
                    "outpaint and conditioning_images are mutually exclusive "
                    "(the source video already fixes the frame content)"
                )
            if self.crop_output is not None:
                raise ValueError(
                    "outpaint and crop_output are mutually exclusive (cropping "
                    "the result would cut off the region that was just extended)"
                )
            keep_width = self.width - op.pad_left - op.pad_right
            keep_height = self.height - op.pad_top - op.pad_bottom
            if keep_width < OUTPAINT_MIN_KEEP_SIDE or keep_height < OUTPAINT_MIN_KEEP_SIDE:
                raise ValueError(
                    f"outpaint keep region {keep_width}x{keep_height} is smaller "
                    f"than {OUTPAINT_MIN_KEEP_SIDE}px on a side; the blend's mask "
                    "dilation reaches roughly a tenth of the canvas's long side "
                    "inward and would consume it entirely"
                )

        # ── Inpainting ────────────────────────────────────────────────────────
        # One flat check per rule, same discipline as the outpaint block above.
        # Everything that needs the file on disk (the source's real resolution,
        # the mask's resolution and frame count, the window fitting) lives at
        # the endpoint instead — see api/generate.py.
        if self.inpaint is not None:
            if not self.reference_video_id:
                raise ValueError(
                    "inpaint requires reference_video_id (the video whose masked "
                    "region is being repainted)"
                )
            if self.outpaint is not None:
                raise ValueError(
                    "inpaint and outpaint are mutually exclusive (one repaints "
                    "inside the frame, the other invents a band around it)"
                )
            if self.conditioning_images:
                raise ValueError(
                    "inpaint and conditioning_images are mutually exclusive "
                    "(the source video already fixes the frame content)"
                )
            if self.crop_output is not None:
                raise ValueError(
                    "inpaint and crop_output are mutually exclusive (the result "
                    "is already cropped back to the source's own resolution)"
                )
            if self.width % INPAINT_CANVAS_MULTIPLE != 0:
                raise ValueError(
                    f"inpaint width must be a multiple of {INPAINT_CANVAS_MULTIPLE} "
                    "(it is the canvas: the source's width rounded up)"
                )
            if self.height % INPAINT_CANVAS_MULTIPLE != 0:
                raise ValueError(
                    f"inpaint height must be a multiple of {INPAINT_CANVAS_MULTIPLE} "
                    "(it is the canvas: the source's height rounded up)"
                )
        return self

    @property
    def generation_mode(self) -> Literal["t2v", "i2v"]:
        return "i2v" if self.conditioning_images else "t2v"


# Total-timeline pixel-frame cap. The masked AV-latent chain's stage-2 DECODE is
# always tiled, so that step stays VRAM-flat at any length. The UPSAMPLE step,
# however, runs over the WHOLE timeline in one GPU pass unless the request opts
# into ``chunked_upsample``, and that pass does not free its intermediate
# latents, so it carries a VRAM term that DOES grow with total length. This cap
# bounds that term: it is a sanity ceiling = 24 clips × 481f (the max clip count
# × the frozen per-clip cap), documented so a UI cannot request an unbounded
# timeline.
#
# It is charged on what the CLIPS assemble to (``ChainLayout.clips_total_px``),
# not on the delivered mp4 — see the comparison in GenerateChainRequest's
# validator. An end source's frozen band is never counted on top of the clips:
# in the reachable modes it is the last clip's own tail, and the API-unreachable
# ``internal_segment`` mode, which appends it, has it subtracted back out. So
# attaching one can never push a chain over this cap.
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
    """Video-to-video continuation source (optional).

    ``video_id`` is an existing upload from POST /upload/video (reuses the
    reference-video store). ``context_frames`` is the source *tail* span (pixel
    frames, 8n+1) that is VAE-encoded and frozen as clip-0's head; the delivered
    mp4 is the NEW part only (the context is trimmed off the front server-side).

    ``context_frames`` is bounded [``v2v_context_frames_min``,
    ``v2v_context_frames_max``] of :class:`config.LimitsConfig`'s defaults
    (``_LIMITS_DEFAULTS``, which does not read ``config.yaml``; ``GET /config``
    advertises the ``config.yaml`` values, so the two diverge if
    ``config.yaml`` overrides these keys). The max is a conservative ceiling
    for the default stage-2 window, inside the HARD invariant enforced by
    :func:`chain_math.compute_chain_layout`: the frozen video head
    (``n_ctx_v = (context_frames-1)//8+1``) must fit inside stage-2 TILE 0
    (``v_tile`` latents of the chosen ``stage2_window``, i.e.
    ``chain_math.px_from_v_latent(v_tile)`` pixel frames) because the
    variant-B hard-freeze only covers tile 0 — ``compute_chain_layout`` raises
    ValueError if that invariant is ever violated. A narrower window can bind
    below the max; GenerateChainRequest's validator checks
    ``chain_math.stage2_max_context_px(v_tile)`` per request. Do NOT raise this
    cap without re-checking multi-tile freeze behaviour first.
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
                "(the cap is LimitsConfig.v2v_context_frames_max in config.py)"
            )
        if (cf - 1) % 8 != 0:
            raise ValueError("source_video.context_frames must be 8n+1")
        return self


class SourceAudioSpec(BaseModel):
    """Audio-to-video source (optional).

    ``audio_id`` is an existing upload from POST /upload/audio. The uploaded
    waveform is VAE-encoded to audio latents that are HARD-frozen (full length,
    mask_value=0) across the chain timeline, so the generated video is driven to
    match the audio (lip-sync) while the ORIGINAL waveform is muxed back onto the
    output (no vocoder). Video length is authoritative: the audio is truncated to
    the timeline, never padded — a too-short upload is rejected up front (422
    SOURCE_AUDIO_TOO_SHORT). No trimming controls are exposed (audio_start_time /
    audio_max_duration). ONE audio spans every clip of the chain (long A2V): the
    encoded latent covers the whole assembled timeline and
    ``chain_math.audio_segment_windows`` hands each stage-1 segment its own
    window on it. Mutually exclusive with ``source_video`` (A2V + V2V is out of
    scope), with ``retake`` (both would own the chain's audio latent) and with
    ``end_source`` (its band freezes the material's own audio).
    """

    audio_id: str = Field(..., min_length=1)


class RetakeSpec(BaseModel):
    """Retake — regenerate the MIDDLE of an existing clip (temporal inpainting).

    ``video_id`` is an existing upload from POST /upload/video. The server cuts
    the window ``[window_start_sec, window_start_sec + clips[0].num_frames)`` out
    of it (``video_io.cut_window_mp4``, frame-exact, resampled to ``frame_rate``
    if needed) and hands ONLY that window to the engine, which regenerates its
    middle while holding ``head_px`` frames at the front and ``tail_px`` frames at
    the back frozen through both stages. The delivered mp4 is the WHOLE window,
    untrimmed: the glue bands are the overlap material the timeline lays back
    over the original, which is what puts the quality seam at the window's outer
    edge instead of at the edit point.

    ``clips[0].num_frames`` is the SINGLE source for the window length — there is
    deliberately no second length field here to disagree with it. Its legal range
    ([``chain_math.RETAKE_WINDOW_MIN_PX``,
    ``chain_math.retake_max_window_px(v_tile)``] for the chosen stage-2 window)
    is enforced by ``chain_math.compute_chain_layout``;
    ``config.limits.retake_window_min_frames`` / ``retake_window_max_frames``
    publish the default-window numbers.

    ``head_px`` must be 8n+1 and ``tail_px`` a multiple of 8 — the two ends sit on
    DIFFERENT latent grids because the video VAE is causal
    (``chain_math.v_tail_latents``). The 25/24 defaults are the calibrated
    recommendation from VERIFICATION_LOG §55.5; wider is not monotonically
    better (49/48 measurably weakened the audio seam).

    ``regenerate_audio`` False keeps the window's ORIGINAL waveform and muxes it
    back verbatim (the vocoder is skipped); it then requires the upload to
    actually have an audio stream (422 up front if not).

    Mutually exclusive with ``source_video``, ``source_audio``,
    ``reference_video_id``, ``clips[0].conditioning_images`` and ``end_source``
    — all of them want to own the ends of the one clip.
    """

    video_id: str = Field(..., min_length=1)
    window_start_sec: float = Field(..., ge=0.0)
    head_px: int = Field(25, ge=9)
    tail_px: int = Field(24, ge=8)
    regenerate_audio: bool = True

    @model_validator(mode="after")
    def validate_glue_grids(self) -> "RetakeSpec":
        if (self.head_px - 1) % 8 != 0:
            raise ValueError(
                "retake.head_px must be 8n+1 (the head band starts at the causal "
                "VAE's lone keyframe latent)"
            )
        if self.tail_px % 8 != 0:
            raise ValueError(
                "retake.tail_px must be a multiple of 8 (the tail band is whole "
                "latent groups counted back from the end)"
            )
        return self


class EndSourceSpec(BaseModel):
    """End source — the chain ENDS with an uploaded video / still image.

    The mirror of :class:`SourceVideoSpec` at the other end of the timeline.
    Exactly ONE of ``video_id`` (an upload from POST /upload/video, the same
    store the continuation source uses) or ``image_id`` (POST /upload/image) is
    given; the app turns either into one ``_end_source.mp4`` — a still is looped
    into a silent video server-side — so the engine sees a SINGLE code path.
    Its frames are VAE-encoded and hard-frozen as the last ``context_frames``
    pixel frames of the whole chain, exactly the way a retake freezes its glue
    bands.

    THE CLIP COUNT AND ``source_video`` SELECT THE GEOMETRY. ``chain_math``
    decides it in one place and publishes it as the layout's (and the
    metadata's) ``end_source.mode``; no field of the request names it. IN EVERY
    CASE THE OUTPUT LENGTH IS THE CLIPS' OWN TOTAL — the band is always the LAST
    clip's own tail, never an addition to the timeline.

    ONE CLIP -> ``"in_window"``. The band is the clip's OWN last
    ``context_frames`` pixel frames, so THE OUTPUT LENGTH IS THE CLIP LENGTH,
    UNCHANGED: ``num_frames`` is the total, of which the last ``context_frames``
    are the material and the rest is newly generated. A one-clip chain is
    denoised by stage 1 as a SINGLE window, which is the point of the mode — the
    frozen band is inside the window every generated latent attends over, so the
    motion can steer towards the material over the whole clip. The band eats
    into the clip's own latents, so a clip with nothing left to generate once the
    band (and a ``source_video`` head, if any) is frozen is REJECTED by
    ``chain_math.compute_chain_layout`` with a 422 naming the clip length that
    would work. The upload must still hold ``context_frames + 1`` frames.

    TWO OR MORE CLIPS, NO ``source_video`` -> ``"reverse"``. THE SAME
    OUTPUT-LENGTH PROMISE, ON A CHAIN: ``num_frames`` still means what it says,
    the band is the LAST clip's own tail, and the delivered length is
    ``sum(clip frames) - overlaps`` exactly as it is without an end source. What
    changes is invisible from the request — stage 1 generates the clips LAST TO
    FIRST, each one freezing the next one's opening as its own ending, so the
    whole chain is generated towards the material instead of being crossfaded
    onto it at the end. ``overlap_strength`` governs those reverse seams exactly
    as it governs forward ones. One combination is refused in this mode; see the
    422 list below.

    TWO OR MORE CLIPS *WITH* A ``source_video`` -> ``"bridge"``. The chain fills
    the span between two uploads: it begins on the start source's tail (trimmed
    off the delivery, as on any V2V continuation) and ends on the end source's
    head. THE SCHEDULE IS AN ORDINARY FORWARD CHAIN'S — nothing is generated
    backwards — and the only segment that differs from a plain chain is the LAST
    one, which is conditioned at BOTH ends at once: its head by the ordinary
    ``overlap_frames`` のり代, its tail by the band. The output-length promise is
    unchanged (the clips' own total, minus the trimmed start-source head).

    WHAT ``bridge`` COSTS is inside that last clip: when the two uploads are far
    apart in content, the transition between them shows there as a crossfade or
    a morph. THAT IS ACCEPTED BEHAVIOUR (owner ruling), not a defect —
    the mode is for material that is already similar, e.g. two takes of the same
    scene with the span between them missing. The one guarantee is the 422 below:
    the last clip must have free latents BETWEEN its two frozen ends.

    ONE CLIP IS THE RECOMMENDED USAGE FOR ``end_source`` ALONE (owner ruling).
    ``reverse`` (two or more clips, no start source) IS ACCEPTED BUT
    NOT RECOMMENDED: real-run gates showed a systematic morph at clip seams and
    at the tail (just before the anchor), which is accepted as a spec-level
    trade-off rather than treated as a defect. See Docs/VERIFICATION_LOG.md
    §64.7 / §65.8.

    (A fourth mode, ``"internal_segment"``, in which the band is APPENDED and
    the delivered length grows by ``context_frames``, is kept in ``chain_math``
    behind a test/rollback override that no request reaches: a request that
    would fit its shape gets ``"reverse"``, so a caller's length prediction
    does not add the band.)

    ``context_frames`` is a MULTIPLE OF 8, not 8n+1. The two ends sit on
    DIFFERENT latent grids because the video VAE is causal: latent 0 is a lone
    keyframe covering pixel 0 only, so a HEAD band is 8n+1 while a TAIL band is
    whole groups of 8 counted back from the end and never touches that keyframe
    (``chain_math.v_tail_latents``). Bounded [``end_context_frames_min``,
    ``end_context_frames_max``] of :class:`config.LimitsConfig`'s defaults
    (``_LIMITS_DEFAULTS``, which does not read ``config.yaml``).

    8 IS THE RECOMMENDED VALUE, AND THE DEFAULT OF 72 IS NOT. 72 is what the
    contract defaults to; the real-run comparison in Docs/VERIFICATION_LOG.md
    §61 settled on an 8-frame anchor, because a longer band spends the
    denoising window re-rendering the material and costs the generator its
    invention. The frontend sends its own value explicitly
    (``END_SOURCE_CONTEXT_FRAMES`` in its ``timeline/tailAlign.ts``) rather
    than letting the default apply.

    ``end_context_frames_max`` IS AN OPERATIONAL CAP, NOT A GEOMETRIC ONE. The
    band is free to span several stage-2 tiles (``ChainLayout.end_tile_bands``
    is the per-tile freeze plan), so no window geometry limits it; the cap is
    simply where measurement stops, kept to avoid shipping an unvalidated
    region. Raising it is an edit to :class:`config.LimitsConfig` plus a
    real-run quality gate, not a geometry change. There is likewise NO
    per-window cross-check on the request.

    OVERLAP >= 2 IS REQUIRED IN ``in_window`` MODE, AND NOT IN ``reverse`` OR
    ``bridge``. The rule was written for the ``internal_segment`` geometry, whose
    extra segment cost one more audio crossfade and made ``overlap_frames == 1``
    marginal — an exhaustive sweep found every degenerate case confined to kv=1.
    Neither ``reverse`` nor ``bridge`` appends a segment, so both spend exactly
    the audio budget an end-source-less chain spends; kv=1 is in fact
    ``reverse``'s intended value (one shared latent per reverse seam).
    ``in_window`` keeps the rule as a conservative choice (widening the accepted
    range is a behaviour change nothing needs). The rejections live in
    ``chain_math.compute_chain_layout`` and surface as 422s with a message
    telling the caller what to change; they are not re-checked here.

    ``strength`` (0.0..1.0, default 1.0) SOFTENS ONLY STAGE 1. 1.0 is a hard
    freeze at stage 1 too (mask ``0.0``). Below 1.0 the stage-1 mask value
    for the tail becomes ``1.0 - strength`` instead of ``0.0``, so stage 1 is
    allowed to drift from the material by degrees rather than being pinned to
    it outright — ``overlap_strength`` is the analogous knob for the seam
    BETWEEN generated segments, this one is for the seam onto the MATERIAL
    itself. STAGE 2 ALWAYS HARD-FREEZES (mask 0.0) REGARDLESS OF ``strength``,
    so the delivered last frame is the material either way: in the returned
    ``freeze_proof``, ``s2_video_tail`` is therefore always 0.0, while
    ``s1_video_tail`` being non-zero is EXPECTED (not a bug) whenever
    ``strength < 1.0`` — see ``freeze_proof.s1_expected_zero``. IT IS A
    VIDEO-ONLY KNOB: the material's audio band is hard-frozen in both stages at
    every strength, so ``s1_audio_tail`` / ``s2_audio_tail`` are 0.0 regardless
    (and ``s1_expected_zero`` speaks for the video pair alone).

    THE +1 PRIMER: the app cuts (or synthesises) ``context_frames + 1`` frames,
    not ``context_frames``. The causal video VAE spends the material's FIRST
    frame on its lone keyframe latent, which is not part of the tail band — so
    frame 0 of the upload never appears in the output and the delivered mp4 ends
    with the upload's frames 1..context_frames.

    OUTPUT LENGTH — ONE ANSWER FOR EVERY REACHABLE MODE: ``total_px ==
    clips_total_px``, i.e. the length the same clips would have produced with no
    end source at all. Asking for one 169-frame clip with a 24-frame band yields
    a 169-frame mp4 whose last 24 frames are the material; asking for three
    169-frame clips at ``overlap_frames == 1`` yields the 505 frames those clips
    assemble to, of which the last 24 are the material. Nothing is TRIMMED
    either, which is the reverse of ``source_video``: that one cuts its frozen
    head OFF the delivered mp4.

    THE MATERIAL'S AUDIO COMES WITH IT. A video end source with an audio track
    has that track frozen over the band alongside the video — no field, no
    toggle; a still image (or a video with no audio) simply has none and the
    tail's audio is generated freely. ``strength`` governs the VIDEO band only:
    the audio band is hard-frozen at every strength.

    Mutually exclusive with ``retake`` (it already owns both ends of its one
    window), ``source_audio`` (a driving audio track owns the whole timeline's
    audio and would collide with the band's frozen audio) and
    ``reference_video_id`` (a control adapter's per-segment conditioning
    competes with the frozen band).

    THE 422 THAT BELONGS TO THE TWO MULTI-CLIP MODES (``reverse`` and
    ``bridge``, from ``chain_math.compute_chain_layout``): A LAST CLIP too short
    to hold ``overlap_frames + the band`` is refused, naming the clip length that
    would work. Below that there is nothing free between the clip's two spoken-
    for ends — under ``reverse`` the latents handed backwards would be the frozen
    material rather than newly generated content; under ``bridge`` the denoiser
    would have no latent of its own to make the transition in.

    ``source_video`` + ``end_source`` + TWO OR MORE CLIPS selects ``bridge``;
    on ONE clip the same pair is the interpolation case, ``in_window``.

    It may also be combined with ``clips[0].conditioning_images`` — those
    keyframes condition the TIMELINE's first clip, which in ``reverse`` mode is
    the one generated LAST — and there is deliberately NO collision check between
    a keyframe and the band. In ``reverse`` and ``bridge`` modes the band is in
    the LAST clip and the keyframes are in the FIRST, so with two or more clips
    they cannot meet at all. In ``in_window`` mode a keyframe COULD
    land inside the band, where the freeze would simply overwrite it — an
    accepted gap, left unrejected here because adding the rejection means
    reversing three existing statements (a test, this comment and
    VERIFICATION_LOG) at once.
    """

    video_id: str | None = Field(None, min_length=1)
    image_id: str | None = Field(None, min_length=1)
    context_frames: int = Field(72)
    strength: float = Field(1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_end_source(self) -> "EndSourceSpec":
        if (self.video_id is None) == (self.image_id is None):
            raise ValueError(
                "end_source requires exactly one of video_id / image_id "
                "(a still is turned into a video server-side, so the two are "
                "alternatives, never a pair)"
            )
        cf = self.context_frames
        cf_min = _LIMITS_DEFAULTS.end_context_frames_min
        cf_max = _LIMITS_DEFAULTS.end_context_frames_max
        if cf < cf_min:
            raise ValueError(f"end_source.context_frames must be >= {cf_min}")
        if cf > cf_max:
            raise ValueError(
                f"end_source.context_frames must be <= {cf_max} "
                "(the cap is LimitsConfig.end_context_frames_max in config.py "
                "— an operational cap on the measured range, not a geometric "
                "limit)"
            )
        if cf % 8 != 0:
            raise ValueError(
                "end_source.context_frames must be a multiple of 8 (a tail band "
                "is whole latent groups counted back from the end — the head "
                "grid's 8n+1 does not apply)"
            )
        return self


class GenerateChainRequest(BaseModel):
    """A chain of clips assembled into ONE continuous masked AV-latent timeline.

    Additive to the frozen single-``/generate`` contract. Shares
    width/height/seed/frame_rate/pipeline across clips; each clip's effective
    prompt is its override if present else the global ``prompt``. Reuses the
    same FROZEN validators (÷64 resolution, 8n+1 frames, distilled 8-step /
    CFG=1.0) as :class:`GenerateRequest`.

    ARCHITECTURE: every clip is a stage-1 SEGMENT of one timeline;
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
    # 同一シードでも生成結果の細部が変わる点、vae_mode="prune_vaed" も同様に
    # 絵が変わる点、vae_mode が既存 vae_tiling と無関係である点も、すべて
    # GenerateRequest と同じ。
    attention_backend: Literal["sdpa", "sage"] = "sdpa"
    # block_swap_prefetch: 詳細は GenerateRequest の同名フィールドを参照。
    # 既定は BLOCK_SWAP_PREFETCH_DEFAULT。off にすると同期スワップになる。
    block_swap_prefetch: bool = BLOCK_SWAP_PREFETCH_DEFAULT
    # keep_resident: 詳細は GenerateRequest の同名フィールドを参照。既定は
    # KEEP_RESIDENT_DEFAULT（常駐するメインメモリ量の実測は LTX 2.3 が
    # VERIFICATION_LOG §48、テキストエンコーダを常駐させる LTX 2.5 が §76）。
    # チェーンでも1つの設定がチェーン全体に効く（骨格キャッシュはジョブ単位
    # ではなくワーカー単位）。LTX 2.5 でも同じで、構築はジョブあたり1回だけで
    # ある（§76）。
    keep_resident: bool = KEEP_RESIDENT_DEFAULT
    # keep_resident_embeddings: 詳細は GenerateRequest の同名フィールドを参照。
    # 既定は KEEP_RESIDENT_EMBEDDINGS_DEFAULT。チェーンでも1つの設定がチェーン
    # 全体に効き、埋め込み処理器の構築はジョブあたり1回である（したがって節約
    # されるのは「次のジョブの構築」であって、チェーンの内側ではない）。LTX 2.3
    # で 422 になる点も単発と同じ（連結で断る項目は
    # services/engines/ltx/adapter.py の CHAIN_REJECT_TABLE が正本）。
    keep_resident_embeddings: bool = KEEP_RESIDENT_EMBEDDINGS_DEFAULT
    # fused_gguf_dequant_kernel: 詳細は GenerateRequest の同名フィールドを参照。
    # 既定は FUSED_GGUF_DEQUANT_KERNEL_DEFAULT（GenerateRequest と同じ）。
    # チェーンでも1つの設定がチェーン全体に効く。
    fused_gguf_dequant_kernel: bool = FUSED_GGUF_DEQUANT_KERNEL_DEFAULT
    vae_mode: Literal["default", "prune_vaed"] = "default"
    # embed_mp4_metadata: 詳細は GenerateRequest の同名フィールドを参照。
    # 既定は EMBED_MP4_METADATA_DEFAULT。チェーンでは完成した連結動画
    # （output.mp4）に刻む。
    embed_mp4_metadata: bool = EMBED_MP4_METADATA_DEFAULT

    width: int = Field(512, ge=256, le=4096)
    height: int = Field(320, ge=128, le=4096)
    crop_output: CropOutput | None = None

    frame_rate: float = Field(24.0, ge=1.0, le=60.0)
    num_inference_steps: int = Field(8, ge=1, le=100)
    guidance_scale: float = Field(1.0, ge=0.0, le=20.0)
    seed: int = -1
    pipeline: Literal["distilled", "two_stage_hq"] = "distilled"

    # Continuity: overlap = K_v LATENT frames shared between consecutive
    # stage-1 segments (the previous segment's tail is copied into the next
    # segment's head and frozen at ``overlap_strength``). The Field default
    # below is the spike-validated K_v; it must be < every clip's stage-1
    # latent-frame count.
    overlap_frames: int = Field(3, ge=1, le=8)
    overlap_strength: float = Field(0.5, ge=0.0, le=1.0)

    # 1..24 clips. A plain chain needs at least 2 (a single clip is just
    # /generate); the clip-count floor in the model_validator lists the inputs
    # that make a single clip legal. Field floor is 1 so those paths validate;
    # capped so the timeline stays within MAX_CHAIN_TOTAL_PIXEL_FRAMES.
    clips: list[ChainClip] = Field(..., min_length=1, max_length=24)

    # Video-to-video continuation (optional). When set, the tail of an
    # uploaded source video is frozen as clip-0's head; see :class:`SourceVideoSpec`.
    source_video: SourceVideoSpec | None = None

    # Audio-to-video (optional). When set, an uploaded audio track is frozen as
    # the chain's audio latent and the video is generated to match it; see
    # :class:`SourceAudioSpec`. Mutually exclusive with ``source_video``.
    source_audio: SourceAudioSpec | None = None

    # Retake — temporal inpainting (optional; a request omitting this field
    # sends no "retake" key in the worker payload — the adapters' payload tests
    # pin the default chain key set). See :class:`RetakeSpec`.
    retake: RetakeSpec | None = None

    # End source — the chain ENDS with an uploaded video / still (optional; a
    # request omitting this field sends no "end_source" key in the worker
    # payload — the adapters' payload tests pin the default chain key set).
    # See :class:`EndSourceSpec`.
    end_source: EndSourceSpec | None = None

    # Style/character IC-LoRA (optional). Same ``LoraSpec`` type/validation as
    # ``GenerateRequest.loras``; the strengths apply uniformly to EVERY clip and
    # every stage of the chain (no per-clip strengths — owner decision).
    # A2V (source_audio) and V2V continuation (source_video) may be combined with
    # loras (no exclusivity guard).
    loras: list[LoraSpec] = Field(default_factory=list)

    # Reference-video CONTROL IC-LoRA (optional): a chain MAY carry a
    # ``reference_video_id`` like a single ``/generate``, on ANY clip count the
    # ``clips`` field allows (owner decision). One long reference video covers
    # the WHOLE assembled timeline; the server auto-slices it into per-clip
    # windows for each stage-1 segment (chain_math.video_segment_windows), so
    # no per-clip upload exists or is needed. A reference shorter than the
    # timeline is not an error — segments past the end of the reference simply
    # generate without one. A depth-preprocess control adapter is the exception
    # (Video-Depth-Anything is a whole-clip design that cannot be windowed):
    # that combination is rejected on >1 clip, at the endpoint
    # (LORA_DEPTH_CHAIN_UNSUPPORTED). Mutually exclusive with ``source_video``:
    # the frozen V2V source head and a reference-conditioned control adapter
    # would otherwise compete for clip 0's head. Same type/bounds as
    # ``GenerateRequest.reference_video_id`` /
    # ``conditioning_attention_strength`` / ``reference_video_strength`` (see
    # there for field-level rationale); the control-vs-style adapter kind check
    # needs the registry, so it lives at the endpoint (api/generate_chain.py),
    # mirroring the single-generate check.
    reference_video_id: str | None = None
    conditioning_attention_strength: float | None = Field(None, ge=0.0, le=1.0)
    reference_video_strength: float | None = Field(None, ge=0.0, le=1.0)

    # Chunked-upsample opt-in (optional). When True the engine upsamples the
    # assembled stage-1 timeline in temporal chunks (halo overlap + CPU
    # offload) instead of one whole-timeline GPU pass, trading time for a flat
    # VRAM ceiling so long high-resolution chains fit a smaller GPU (see the
    # measured record); when False the engine keeps the one-pass path (owner
    # decision: the chunked path is opt-in, so the one-pass path is left
    # untouched).
    chunked_upsample: bool = False

    # Stage-2 window preset (optional; a request omitting this field sends no
    # "stage2_window" key in the worker payload). "standard" is the default
    # tile layout. The other tiled windows trade seams against per-tile
    # weight: "high_resolution" (a shorter window with a wider overlap) costs
    # fewer attention tokens per tile but gets MORE seams (owner decision); the
    # "w25".."w61" ladder (named for the window length) gets FEWER seams at the
    # cost of a heavier tile against the comfortable per-tile budget (the chain
    # budgets in the comfort_budgets table, served from the base-model
    # manifests' ``comfort`` block).
    #
    # Named for the geometry, NOT for a duration: the window's advance is a
    # LATENT-frame count, so its wall-clock length depends on frame_rate. The UI
    # is what renders it as seconds for the frame rate actually chosen.
    # chain_math.STAGE2_WINDOW_PRESETS is the single source of truth for the
    # numbers behind each name.
    #
    # "full_length" (no overlap between tiles) is the a2v (audio-to-video)
    # window: its length in pixel frames equals the ChainClip.num_frames
    # ceiling, so a ONE-clip chain always fits in a SINGLE stage-2 tile and its
    # stage-2 becomes exactly what plain POST /generate does — no tile seam
    # anywhere on the timeline. It is restricted below to 1 clip + source_audio,
    # the shape the Single/Batch a2v flow builds. It deliberately does NOT bound
    # how long that clip may comfortably be: that axis is
    # config.limits.spill_free_frames (the per-resolution comfortable frame cap
    # the server publishes), not this one.
    stage2_window: Literal[
        "standard", "high_resolution", "full_length",
        "w25", "w28", "w31", "w34", "w37", "w40", "w43",
        "w46", "w49", "w52", "w55", "w58", "w61",
    ] = "standard"

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
                    "distilled pipeline requires num_inference_steps=8"
                )
            if self.guidance_scale != 1.0:
                raise ValueError(
                    "distilled pipeline requires guidance_scale=1.0"
                )

        # A2V + V2V are mutually exclusive (do not mix an uploaded continuation
        # video with an uploaded driving audio). Rejected up front.
        if self.source_audio is not None and self.source_video is not None:
            raise ValueError(
                "source_audio and source_video are mutually exclusive "
                "(A2V and V2V cannot be combined)"
            )

        # A2V accepts any clip count the ``clips`` field allows (long A2V). The
        # uploaded audio is ONE track spanning the whole assembled timeline;
        # chain_math.audio_segment_windows hands each stage-1 segment its own
        # window on that global audio latent (consecutive windows overlap by the
        # same per-join K_a the assembler crossfades with), so no per-clip audio
        # upload exists — and none is needed.

        # Retake owns BOTH ends of the one and only clip, so it cannot share the
        # timeline with any other head/end claimant. Rejected up front, in the
        # same style as the source_audio/source_video pair above.
        if self.retake is not None:
            if self.source_video is not None:
                raise ValueError(
                    "retake and source_video are mutually exclusive (a retake "
                    "freezes both ends of an existing window; a V2V continuation "
                    "freezes clip 0's head and generates onward)"
                )
            if self.source_audio is not None:
                raise ValueError(
                    "retake and source_audio are mutually exclusive (both want to "
                    "own the chain's audio latent)"
                )
            if self.reference_video_id is not None:
                raise ValueError(
                    "retake and reference_video_id are mutually exclusive (a "
                    "control adapter's reference conditioning would compete with "
                    "the frozen retake glue bands)"
                )
            if self.clips[0].conditioning_images:
                raise ValueError(
                    "retake is mutually exclusive with clips[0].conditioning_images "
                    "(the window's own frames already occupy the frozen ends)"
                )
            # The window IS the clip; clips[0].num_frames is its length.
            if len(self.clips) != 1:
                raise ValueError("retake requires exactly 1 clip (the window itself)")

        # An end source freezes the tail of the WHOLE chain, so it cannot share
        # the timeline with anything else that claims an end or the audio track.
        # Rejected up front, in the same style as the retake block above. NOT
        # exclusive with source_video (start + end IS the interpolation use case)
        # nor with clips[0].conditioning_images (a keyframe is legal; there is
        # deliberately no keyframe x band collision check — see the note after
        # the layout below).
        if self.end_source is not None:
            if self.retake is not None:
                raise ValueError(
                    "retake and end_source are mutually exclusive (a retake "
                    "window already freezes the tail of the single clip it "
                    "regenerates; an end source freezes the tail of the whole "
                    "chain)"
                )
            if self.source_audio is not None:
                raise ValueError(
                    "end_source and source_audio are mutually exclusive (a "
                    "driving audio track owns the whole timeline's audio, while "
                    "an end source freezes the material's own audio over the "
                    "band — the two would collide at the end)"
                )
            if self.reference_video_id is not None:
                raise ValueError(
                    "end_source and reference_video_id are mutually exclusive (a "
                    "control adapter's per-segment reference conditioning would "
                    "compete with the frozen end-source band)"
                )

        # Clip-count floor: WITHOUT a source (video OR audio), a reference video,
        # a retake or an end_source, a chain needs >= 2 clips (a single clip is
        # just /generate). WITH a source_video the frozen source head IS the prior
        # segment, WITH a source_audio a single clip is the whole timeline, WITH
        # reference_video_id a single clip is a 1-clip reference-conditioned
        # chain, WITH a retake the single clip IS the window being repaired, and
        # WITH an end_source a single clip is "a 5-second video that ends with
        # this" (owner decision) — so 1 clip is OK in all five cases. (Forgetting
        # the retake term here would 422 EVERY retake request before it reached
        # any of its own validation.)
        if (
            self.source_video is None
            and self.source_audio is None
            and self.reference_video_id is None
            and self.retake is None
            and self.end_source is None
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
            _normalize_conditioning_images(
                clip.conditioning_images, clip.num_frames, where=f"clips[{i}]: "
            )

        # Total-timeline geometry: sum of pixel frames minus the shared overlaps.
        # Delegated to the shared pure-Python chain_math so the validator, the
        # engine and the metadata agree; it also raises on a degenerate audio
        # overlap (clips too short for a continuous crossfade).
        import chain_math
        # Stage-2 window preset -> the (v_tile, v_adv) the engine will actually
        # tile with. Passed EXPLICITLY so this validator's geometry matches the
        # engine's even when the request opted into a non-default window; the
        # default preset resolves to the same pair compute_chain_layout's own
        # defaults are bound to (chain_math.STAGE2_V_TILE / STAGE2_V_ADV), so an
        # omitted stage2_window yields the same layout either way.
        stage2_v_tile, stage2_v_adv = chain_math.resolve_stage2_window(self.stage2_window)

        # "full_length" (the zero-overlap, single-tile a2v window) is only
        # geometrically valid on the shape the a2v flow builds. Two independent
        # checks, each with its own message, so the 422 says which one failed.
        if self.stage2_window == chain_math.STAGE2_WINDOW_FULL_LENGTH:
            if len(self.clips) != 1:
                raise ValueError(
                    f'stage2_window="full_length" requires exactly 1 clip (got '
                    f"{len(self.clips)}): the window spans the WHOLE timeline as "
                    "one stage-2 tile, which only fits inside the 481-frame "
                    "per-clip ceiling when there is a single clip"
                )
            if self.source_audio is None:
                # Scope limit, not geometry: a 1-clip chain without audio would
                # tile fine, but the window is unlocked for the a2v flow only
                # (owner decision). THIS is the line to delete if a
                # non-A2V single-clip chain ever wants the same whole-timeline
                # stage-2.
                raise ValueError(
                    'stage2_window="full_length" requires source_audio (it is '
                    "the audio-to-video window; every other chain shape uses "
                    "one of the tiled windows (standard / high_resolution / "
                    "w25..w61))"
                )
            # No source_video / retake exclusivity check here on purpose: both
            # are ALREADY mutually exclusive with source_audio further up (the
            # source_audio x source_video pair and the retake x source_audio
            # pair), so those combinations 422 before reaching this block.

        # V2V x non-default window: the frozen source head must leave stage-2
        # TILE 0 something to generate. The V2V context cap
        # (v2v_context_frames_max; SourceVideoSpec checks config.py's
        # LimitsConfig value) is sized for the standard window, whose
        # chain_math.stage2_max_context_px ceiling is above it (as is every wider
        # "w*" window's), and is deliberately NOT changed here — a narrower window
        # such as "high_resolution" has a ceiling below the cap and needs its own
        # check, or a context at the cap would freeze tile 0 completely (an
        # untested degenerate that compute_chain_layout's own `n_ctx_v > v_tile`
        # guard would still wave through, since it only rejects a head LONGER
        # than the tile).
        if self.source_video is not None:
            max_ctx = chain_math.stage2_max_context_px(stage2_v_tile)
            if self.source_video.context_frames > max_ctx:
                raise ValueError(
                    f"source_video.context_frames ({self.source_video.context_frames}) "
                    f"must be <= {max_ctx} when stage2_window={self.stage2_window!r} "
                    "(a longer frozen head would fill the whole first stage-2 "
                    "window, leaving nothing to generate). Shorten context_frames "
                    "or choose a wider stage2_window."
                )

        # NO end-source x window cross-validation here. The band may span as many
        # stage-2 tiles as it needs (``chain_math`` publishes the per-tile plan in
        # ``ChainLayout.end_tile_bands``), so there is no window-derived ceiling
        # to check: end_context_frames_max (EndSourceSpec checks config.py's
        # LimitsConfig value) is an OPERATIONAL cap, not a geometric one.

        # Retake x non-default window: ALLOWED. A retake window must be refined
        # as ONE stage-2 tile, which is the only geometry the both-side freeze was
        # validated under (VERIFICATION_LOG §55.2/§55.3); that is enforced by
        # BOUNDING the window rather than refusing the combination:
        # compute_chain_layout below checks the window against
        # chain_math.retake_max_window_px(v_tile). ``stage2_v_tile`` is resolved
        # above and passed in, so that bound follows the request's own preset. A
        # window past the preset's ceiling (it would split into 2 tiles and freeze
        # only the last one) is therefore a 422 that names the ceiling.
        # config.limits.retake_window_{min,max}_frames publishes the
        # STANDARD-preset numbers; a client that offers other windows is
        # responsible for mirroring retake_max_window_px(v_tile) (see config.py).

        try:
            layout = chain_math.compute_chain_layout(
                [c.num_frames for c in self.clips], self.frame_rate,
                kv=self.overlap_frames,
                v_tile=stage2_v_tile,
                v_adv=stage2_v_adv,
                source_context_px=(
                    self.source_video.context_frames if self.source_video else None
                ),
                # Window length + glue-band geometry (8n+1 window in
                # [chain_math.RETAKE_WINDOW_MIN_PX, retake_max_window_px(v_tile)],
                # head/tail grids, a free middle in BOTH latent domains) is
                # validated THERE, so the validator, the engine and the mock
                # cannot disagree. Its ValueError surfaces as 422.
                retake_glue_px=(
                    None if self.retake is None
                    else (self.retake.head_px, self.retake.tail_px)
                ),
                # The frozen tail band's own geometry (multiple of 8, >= 8, and
                # free stage-1 latents left in the FINAL clip) is validated
                # THERE, so the validator, the engine and the mock cannot
                # disagree. Its ValueError — which names the concrete frame count
                # the final clip would need — surfaces as 422. NO MODE IS PASSED:
                # the clip count already in ``clip_frames`` plus the
                # ``source_context_px`` passed just above are the whole input to
                # that decision, and passing a mode is what would let this
                # validator and the engine disagree about which one a request is
                # in.
                end_context_px=(
                    None if self.end_source is None
                    else self.end_source.context_frames
                ),
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        # The cap is on the USER'S OWN CLIPS, not on the delivered timeline.
        # ``ChainLayout.clips_total_px`` is the SINGLE definition of that quantity
        # — the same property ``to_dict`` publishes, so the validator and the
        # metadata can never disagree. In the three modes a request can reach
        # (``in_window``, ``reverse`` and ``bridge``) the band is part of a clip,
        # so it equals ``total_px``, exactly as on a chain without an end source;
        # only the API-unreachable ``internal_segment`` mode appends a band
        # segment AFTER the clips and makes the two differ.
        clips_total_px = layout.clips_total_px
        if clips_total_px > MAX_CHAIN_TOTAL_PIXEL_FRAMES:
            raise ValueError(
                f"chain total timeline {clips_total_px} pixel frames exceeds the "
                f"cap {MAX_CHAIN_TOTAL_PIXEL_FRAMES} (reduce clip count or lengths)"
            )

        # NO end-source x clip-0-keyframe collision check here. In the
        # ``reverse`` and ``bridge`` modes the band is the LAST clip's tail while
        # the keyframes belong to the FIRST, so with two or more clips they
        # cannot reach each other and the test would be identically false. (Under
        # ``bridge`` a start source occupies clip 0's head as well, but that pair
        # is mutually exclusive with keyframes at the request level.) In the
        # ``in_window`` mode a keyframe CAN land inside the band (it is the clip's
        # own tail) and the freeze would overwrite it; that gap is knowingly left
        # open — see EndSourceSpec's docstring.

        # Reference-video CONTROL IC-LoRA: mirrors the same check in
        # GenerateRequest.validate_ltx_constraints plus the V2V exclusivity below.
        # The reverse still holds unconditionally: a reference video only ever
        # conditions a lora.
        if self.reference_video_id and not self.loras:
            raise ValueError(
                "reference_video_id requires at least one lora (the reference "
                "video only conditions an IC-LoRA)"
            )

        # A reference video accepts any clip count the ``clips`` field allows
        # (owner decision). A long reference video is ONE upload spanning the
        # whole assembled timeline; chain_math.video_segment_windows hands each
        # stage-1 segment its own pixel-frame window on that reference (segments
        # past the end of a too-short reference just generate without one — no
        # error). A depth-preprocess control adapter is refused on >1 clip, at
        # the endpoint (api/generate_chain.py) where the adapter's registry entry
        # is resolved, not here.

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

        LIVE PATH: ``services/job_store.py`` (``create_chain_if_idle``) calls
        this on every chain job creation, and the result is re-validated as a
        ``GenerateRequest`` and stored as ``JobRecord.request``. Any field added
        to ``GenerateChainRequest`` that is not transcribed here silently drops
        out of the stored/serialized request for a chain job — omitting the nag
        fields would make chain creation 500 (nag_enabled True + empty
        negative_prompt would fail GenerateRequest's own validator).

        The acceleration fields (``attention_backend``, ``block_swap_prefetch``,
        ``keep_resident``, ``keep_resident_embeddings``,
        ``fused_gguf_dequant_kernel`` and ``vae_mode``) are
        transcribed for the same reason: they do not fail validation when
        dropped, so an omission would silently mis-report a chain job's
        settings in the stored record (GET /jobs' ``request`` would claim
        sdpa/default for a sage chain; a chain's metadata.json is written from
        the chain request itself).
        The chain's OWN worker payload is built from the chain request, not from
        this per-clip copy — this transcription only feeds the stored record.
        ``embed_mp4_metadata`` is transcribed for the same reason (the stored
        record would otherwise claim the default).

        ``retake`` is deliberately NOT transcribed, following the same precedent
        as ``source_video`` / ``source_audio``: it has no counterpart on
        ``GenerateRequest``, so there is nothing to drop it INTO, and dropping it
        changes no validation outcome. The authoritative copy is
        ``JobRecord.chain_request``, which ``run_chain_job`` reads and which the
        worker payload is built from.
        ``end_source`` follows that same precedent for the same reasons.
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
            keep_resident=self.keep_resident,
            keep_resident_embeddings=self.keep_resident_embeddings,
            fused_gguf_dequant_kernel=self.fused_gguf_dequant_kernel,
            vae_mode=self.vae_mode,
            embed_mp4_metadata=self.embed_mp4_metadata,
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
    # True only when a cut actually ran and succeeded: either the window of the
    # optional trim_start_sec/trim_duration_sec query arguments or the
    # max_frames ceiling (see services/video_upload_store.VideoUploadStore.save).
    # Additive: older clients simply ignore it, and an upload that sends
    # neither always reports False.
    trimmed: bool = False
    # What the STORED file measures (post-cut when a cut ran). Both None unless
    # the upload asked to be measured by sending ``max_frames`` -- an ordinary
    # upload spends no extra ffprobe. Also None whenever a probe or cut failed:
    # "unknown" is a legitimate answer and the client is expected to fall back
    # (the end source estimates the band length from the media duration
    # instead) rather than treat it as an error.
    # The end source's automatic band length is computed FROM these numbers, so
    # they must describe the file the server actually kept, never the original
    # upload -- see services/video_upload_store.VideoUploadStore.save.
    frame_count: int | None = None
    fps: float | None = None


class UploadAudioResponse(BaseModel):
    audio_id: str
    original_filename: str
    stored_path: str
    content_type: str
    size_bytes: int


class JoinRequest(BaseModel):
    """POST /jobs/{job_id}/join body (V2V, ADDITIVE — an endpoint of its own).

    Server-side join of a completed V2V job's continuation (``output.mp4``, the
    NEW part only) back onto its uploaded source video, producing ``joined.mp4``
    next to the job output. GPU-free (ffmpeg only) and independent of the
    single-GPU-job guard.

    ``audio_smoothing`` selects the audio treatment at the junction:

    * ``True`` — crossfade: a true overlapped equal-power crossfade
      via the engine's ``<stem>_audio_handle.wav`` sidecar when the job has one,
      else the no-handle fade-pair (see ``services/video_io.join_v2v`` and
      Docs/V2V_AUDIO_JOIN_RESEARCH.md).
    * ``False`` — hard concat (no fades). Kept for parity/testing; the GUI only
      exposes the smoothed path.

    ``handle_crossfade_ms`` applies to the handle true-crossfade only; it is an
    audio-only acrossfade (the video is always a hard cut at the seam). Its
    default is longer than the VERIFICATION_LOG §24.7 sweet spot, which left
    the seam slightly audible on real content (VERIFICATION_LOG §26.3); the
    GUI's crossfade dropdown offers alternatives. All fields are optional; an
    empty body ``{}`` uses every field's default.
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
    # ADDITIVE: pipeline phase of the latest progress event, as the engine
    # worker names it ("encode" / "stage1_denoise" / "tile" / "decode" and so
    # on). None when the backend has not reported one (mock milestones, queued
    # jobs) — consumers must treat unknown values as "no label".
    stage: str | None = None
    # Chain clip progress (ADDITIVE, same discipline as ``stage``): 1-based
    # index of the clip (stage-1 segment) the chain is currently denoising and
    # the total clip count. Set only by chain jobs on real backends (the worker
    # reports the segment position); ``clip`` retains its last value through
    # the later whole-timeline stages (stage-2 tiles / decode), so clip ==
    # clip_count reads as "all clips are through stage 1". None for single
    # generates, queued jobs and mock milestones.
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
