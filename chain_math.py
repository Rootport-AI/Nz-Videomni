"""Pure-Python chain (clip-concatenation) timeline & tile/junction math.

SINGLE SOURCE OF TRUTH for the masked AV-latent chaining architecture geometry,
shared by:
  * engine/pipeline/chain_pipeline.py  (the torch orchestration — uses this to
    lay out stage-1 segments, per-join audio overlaps, and the always-tiled
    stage-2 windows so the engine and the app agree byte-for-byte on geometry),
  * services/ltx_runner.py _MockBackend  (synthesises a single mp4 of the right
    total length + the same junction indices, GPU-free),
  * services/pipeline_manager.py         (metadata.json junction frames),
  * api/models.py                        (total-length validation).

ZERO heavy deps (no torch / no ltx_core) so it imports in BOTH the app venv
(.venv, torch-less) and the engine venv (.venv-engine). The latent-frame math is
a faithful pure-Python replica of the installed wheel:

  * video latent frames for P pixel frames:  (P - 1) // 8 + 1
      (ltx_core.types.VideoLatentShape.from_pixel_shape, temporal scale 8, causal +1)
  * pixel frames for N video latent frames:   (N - 1) * 8 + 1   (inverse)
  * audio latent frames for P pixel frames @ fps:
        round(P / fps * 25.0)
      where 25.0 = sample_rate/hop_length/audio_latent_downsample_factor
      = 16000/160/4  (ltx_core.types.AudioLatentShape.from_duration).

Architecture (validated by outputs/phase3_clip_concat_spike/s1+s2 spikes):
  per-segment STAGE 1 (half-res) with video+audio latent tail carry+freeze
    -> assemble ONE continuous stage-1 AV latent (linear crossfade at overlaps;
       per-join audio K_a chosen so the assembled audio lands EXACTLY on the
       stage-2 audio target length)
    -> ONE upsample over the whole timeline
    -> STAGE 2 refine in TEMPORAL TILES (always tiled; a short chain degenerates
       to a single tile), tile i>=1 leading overlap hard-frozen then blended
    -> ONE VAE decode -> ONE mp4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Latent geometry constants (from the installed wheel) ─────────────────────
VIDEO_TIME_FACTOR = 8            # temporal VAE compression (causal +1 keyframe)
AUDIO_LATENTS_PER_SEC = 25.0     # 16000 / 160 / 4

# ── Stage-2 window presets (video-latent domain) ─────────────────────────────
# ``(v_tile, v_adv)`` per selectable stage-2 window; the overlap ("のり代") is
# always the derived ``kt_v = v_tile - v_adv``.
#
#   "standard"        (22, 18) -> kt_v 4.  The S2-spike default, unchanged since
#       Phase 3 WP4. 22 latent frames per tile keeps BOTH video and audio
#       temporal RoPE positions well under the trained 20s ceiling (each tile
#       restarts local positions at 0); advance 18 -> 4-frame overlap between
#       consecutive stage-2 tiles.
#   "high_resolution" (19, 12) -> kt_v 7.  Opt-in only (§3-57 sweep + follow-up,
#       owner decision 2026-08-09). A 19-frame window costs ~14% fewer attention
#       tokens per tile, which keeps high resolutions (>= ~1216x1664) inside the
#       comfortable budget below instead of spilling; the wider 7-frame overlap
#       is what the follow-up run added to suppress the "morphing" the bare
#       19/4 window showed on owner review. It advances less per tile, so a
#       chain of the same length gets MORE seams — hence opt-in, not default.
#       See Docs/CHAIN_STAGE2_RESEARCH_NOTES.md §1 and
#       outputs/stage2_window_sweep/SWEEP_RESULTS.md.
#
# BOTH advances are multiples of 3, which is what keeps the 24fps video/audio
# advance rounding exact. A future preset MUST honour that too.
STAGE2_WINDOW_PRESETS: dict[str, tuple[int, int]] = {
    "standard": (22, 18),
    "high_resolution": (19, 12),
}
STAGE2_WINDOW_DEFAULT = "standard"

STAGE2_V_TILE, STAGE2_V_ADV = STAGE2_WINDOW_PRESETS[STAGE2_WINDOW_DEFAULT]
STAGE2_KT_V = STAGE2_V_TILE - STAGE2_V_ADV   # 4


# ── ORDERING CONSTRAINT (do not move this block below `def compute_chain_layout`)
# ``compute_chain_layout``'s ``v_tile: int = STAGE2_V_TILE`` / ``v_adv: int =
# STAGE2_V_ADV`` default arguments are bound to whatever those names hold at
# `def`-EVALUATION time (Python evaluates default-argument expressions once, at
# function definition, not at call time). So the preset table, the two derived
# STAGE2_* constants and everything below MUST stay ABOVE that `def`:
# reassigning STAGE2_V_TILE afterwards — or monkeypatching it from outside the
# module — would NOT change the function's defaults. ``tests/test_stage2_window.py``
# pins this by asserting the bound defaults equal the "standard" preset.
def resolve_stage2_window(name: str | None) -> tuple[int, int]:
    """Resolve a stage-2 window preset NAME to its ``(v_tile, v_adv)``.

    ``None`` / ``""`` -> the default preset (so every caller that simply has no
    opinion produces byte-identical geometry to before this knob existed). An
    unknown name raises ValueError: the window changes the OUTPUT, so a typo
    must fail loudly rather than silently fall back to the default.
    """
    key = name or STAGE2_WINDOW_DEFAULT
    try:
        return STAGE2_WINDOW_PRESETS[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown stage2_window {name!r}; expected one of "
            f"{sorted(STAGE2_WINDOW_PRESETS)}"
        ) from exc


# ── Comfortable per-tile attention-token budget (§1-14) ──────────────────────
# ONE stage-2 tile attends over ``(width/32) * (height/32) * v_tile`` latent
# tokens (32 = the video VAE's spatial compression factor). Past roughly this
# many tokens the attention working set stops fitting the comfortable VRAM
# envelope and the run starts paying for it — measured on the §3-57 spill arm
# (outputs/stage2_window_sweep/SWEEP_RESULTS.md §2b): 44,880 tokens cost
# +1,277MB peak VRAM and +27% wall clock against 38,760 at the same resolution;
# 47,840 cost +2,309MB and +32% against 36,800. 40,000 sits between the clean
# and spilling measurements of both pairs.
#
# This is a SEPARATE axis from ``config.limits.spill_free_frames`` (the
# server-published comfortable per-clip FRAME cap for a resolution): that one
# bounds how LONG one clip may be, this one bounds how WIDE one stage-2 window
# is. A chain can satisfy one and violate the other.
CHAIN_COMFORT_TOKEN_BUDGET = 40_000


def chain_window_tokens(width: int, height: int, v_tile: int) -> int:
    """Attention tokens ONE stage-2 tile spans at ``width x height``.

    ``(width // 32) * (height // 32) * v_tile``. Compare against
    :data:`CHAIN_COMFORT_TOKEN_BUDGET`. Chain-specific: a single ``/generate``
    refines the whole clip in one window, so its budget is a different number.
    """
    return (width // 32) * (height // 32) * v_tile


# ── Comfortable stage-1 attention-token budget with a reference (§1-15) ──────
# The sibling of CHAIN_COMFORT_TOKEN_BUDGET above, on a DIFFERENT axis: that one
# bounds ONE stage-2 tile, this one bounds ONE stage-1 segment. Stage 1 runs at
# HALF resolution and denoises a whole clip in a single pass (no tiling), so
# without a reference it is never the bottleneck — but an IC-LoRA reference is
# patchified alongside the clip and ADDS tokens to that same single pass, which
# is what makes stage 1 bind first on a referenced chain.
#
# 25,000 is PROVISIONAL, derived from the two measured statements in
# Docs/CHAIN_STAGE2_RESEARCH_NOTES.md §6: at 1152x1536 a referenced clip tops out
# around 361 frames (= 24,840 tokens by the formula below at scale 2), and
# keeping 481 frames requires dropping to roughly 328 stage-1 patches
# (= 25,010 tokens). Both land just either side of 25,000. Calibrate it on the
# real-hardware gate (plan G4) and update this constant with the measured knee.
CHAIN_STAGE1_COMFORT_TOKEN_BUDGET = 25_000


def chain_stage1_tokens(
    width: int, height: int, v_latent: int, ref_scale: int | None = None
) -> int:
    """Attention tokens ONE stage-1 segment spans at ``width x height``.

    ``width``/``height`` are the OUTPUT (full) resolution; stage 1 runs at half
    of it, and the video VAE plus patchifier put one latent token per 32 output
    pixels of that halved frame — hence ``(width // 2 // 32) * (height // 2 // 32)``
    spatial patches, times ``v_latent`` (the segment's stage-1 video latent
    frames, ``v_latent_frames(clip_frames[i])``).

    ``ref_scale`` is the adapter's ``reference_downscale_factor`` (``None`` = no
    IC-LoRA reference on this segment). A reference is patchified from its OWN
    shape (``engine/pipeline/reference_video_cond.py``) and appended to the same
    attention sequence, so it adds ``ref_spatial * v_latent`` tokens on top:
    ``scale=2`` (the union-control adapters) contributes a quarter of the spatial
    patches, ``scale=1`` (deblur) contributes exactly as many as the clip itself
    — i.e. it doubles the segment.

    The integer divisions are applied in that exact order and are NOT simplified
    away on the "resolutions are multiples of 128 anyway" assumption: the budget
    is also consulted while the user is still dragging a resolution slider.
    Compare the result against :data:`CHAIN_STAGE1_COMFORT_TOKEN_BUDGET`.
    ``webui/src/shell/tokenBudget.ts`` mirrors this function verbatim, so any
    change here must be mirrored there.
    """
    half_w = width // 2
    half_h = height // 2
    spatial = (half_w // 32) * (half_h // 32)
    tokens = spatial * v_latent
    if ref_scale:
        ref_spatial = ((half_w // ref_scale) // 32) * ((half_h // ref_scale) // 32)
        tokens += ref_spatial * v_latent
    return tokens


def stage2_max_context_px(v_tile: int) -> int:
    """Largest 8n+1 V2V context span that leaves tile 0 something to generate.

    The variant-B hard-freeze covers stage-2 TILE 0 only, so the frozen head
    must fit inside it — but a head that fills it EXACTLY (``n_ctx_v ==
    v_tile``) leaves tile 0 100% frozen, an untested degenerate. This is the
    ceiling for ``n_ctx_v <= v_tile - 1``: 161 for the standard window (above
    ``config.limits.v2v_context_frames_max`` = 145, so it never binds there)
    and 137 for "high_resolution" (below 145, so it DOES bind — see
    ``api/models.py``'s GenerateChainRequest cross-validation).
    """
    return px_from_v_latent(v_tile - 1)


# ── Retake (temporal inpainting) window bounds ───────────────────────────────
# Retake regenerates the MIDDLE of an existing clip while both ends stay frozen.
# The whole window is refined as ONE stage-2 tile (the both-side freeze was only
# ever validated in that degenerate geometry — VERIFICATION_LOG §55.2/§55.3), so
# the window must fit a single tile: ``n_tiles == 1``. That is exactly
# ``px_from_v_latent(v_tile)`` = 169 px frames for the standard (22, 18) window
# (177 already splits into 2 tiles). Note this is the sibling of
# :func:`stage2_max_context_px` above but WITHOUT its ``- 1``: V2V needs tile 0
# to have something left to generate after the frozen head, whereas retake
# deliberately fills the whole tile — its free middle is carved out INSIDE the
# window by the glue bands, not by leaving tile space over.
#
# The floor is a quality bound, not a geometric one: below ~73 px frames the
# free middle left between the 25/24 default glue bands stops being enough
# material to regenerate anything meaningful (owner decision 2026-08-09,
# PENDING_TASKS.md §1-17).
RETAKE_WINDOW_MIN_PX = 73


def retake_max_window_px(v_tile: int) -> int:
    """Largest retake window (8n+1 px frames) that stays ONE stage-2 tile.

    169 for the standard (22, 18) preset. ``config.limits.retake_window_max_frames``
    publishes the standard-preset number to clients; this function is the
    geometry-truth an engine/validator uses for whatever ``v_tile`` is in play.
    """
    return px_from_v_latent(v_tile)


# Default continuity params (new semantics: overlap_frames == K_v LATENT frames).
DEFAULT_OVERLAP_FRAMES = 3       # K_v (video latent overlap), S1-validated
DEFAULT_OVERLAP_STRENGTH = 0.5   # stage-1 carry overlap strength, S1-validated

# ── Chunked spatial-upsample layout (video-latent domain) ────────────────────
# The whole-timeline stage-1 -> stage-2 spatial upsample is memory-bound on the
# full assembled latent (VRAM grows with total length -> OOM on long 768p
# chains). The optional chunked path upsamples in temporal chunks of
# UPSAMPLE_CHUNK_FRAMES output-core frames, each padded by UPSAMPLE_HALO_FRAMES
# frames on both sides so the spatial upsampler's temporal convolutions see the
# same neighbourhood as the one-shot pass (halo=18 reproduces it exactly on the
# convolution interior; halo=17 breaks). The halo is discarded after upsampling
# — only the core is kept — and the physical timeline ends keep the same
# zero-padding the one-shot upsample sees.
UPSAMPLE_CHUNK_FRAMES = 32
UPSAMPLE_HALO_FRAMES = 18


@dataclass
class UpsampleChunk:
    """One temporal chunk of the chunked spatial upsample (video-latent frames).

    * ``in_start`` / ``in_len`` — slice ``[in_start, in_start+in_len)`` of the
      assembled latent fed to the upsampler (output core + physical-clipped halo),
    * ``keep_lo`` / ``keep_len`` — sub-slice of the UPSAMPLED chunk to keep
      (drops the halo): ``[keep_lo, keep_lo+keep_len)``,
    * ``out_start`` — where the kept core lands on the output timeline (== the
      core start ``s``); consecutive chunks tile ``[0, f_total)`` with no gap and
      no overlap.
    """

    in_start: int
    in_len: int
    keep_lo: int
    keep_len: int
    out_start: int


def plan_upsample_chunks(
    f_total: int,
    chunk: int = UPSAMPLE_CHUNK_FRAMES,
    halo: int = UPSAMPLE_HALO_FRAMES,
) -> list[UpsampleChunk]:
    """Tile ``[0, f_total)`` output-core video-latent frames into halo-padded chunks.

    The output cores ``[s, e)`` advance by ``chunk`` (s = 0, chunk, 2*chunk, …),
    with ``e = min(s+chunk, f_total)``. Each chunk reads ``[in_start, in_end)``
    with ``halo`` frames of context on both sides, clipped to the physical
    timeline ends. ``keep_lo`` is the core's offset inside the upsampled chunk
    (== ``halo`` for interior chunks, 0 at the head); ``keep_len`` is the core
    length. The kept cores tile the whole timeline with neither gap nor overlap.
    Pure geometry (no torch); ``f_total`` is the assembled video-latent frame
    count and must be >= 1.
    """
    if f_total <= 0:
        raise ValueError(f"f_total must be >= 1 (got {f_total})")
    chunks: list[UpsampleChunk] = []
    for s in range(0, f_total, chunk):
        e = min(s + chunk, f_total)
        in_start = max(0, s - halo)
        in_end = min(f_total, e + halo)
        chunks.append(
            UpsampleChunk(
                in_start=in_start,
                in_len=in_end - in_start,
                keep_lo=s - in_start,
                keep_len=e - s,
                out_start=s,
            )
        )
    return chunks


def v_latent_frames(pixel_frames: int) -> int:
    """Video latent-frame count for a pixel frame count (8n+1)."""
    return (pixel_frames - 1) // VIDEO_TIME_FACTOR + 1


def px_from_v_latent(n_latent: int) -> int:
    """Inverse of :func:`v_latent_frames` (returns 8n+1 pixel frames)."""
    return (n_latent - 1) * VIDEO_TIME_FACTOR + 1


def v_tail_latents(tail_px: int) -> int:
    """Video latent frames a ``tail_px``-pixel TAIL band occupies (``tail_px // 8``).

    Deliberately NOT :func:`v_latent_frames` — the head/tail grids are different
    because the video VAE is CAUSAL. Latent 0 is a lone keyframe covering pixel
    0 only, and latent k>=1 covers pixels ``8k-7 .. 8k``. So:

      * a HEAD band must be ``8n+1`` px (0, then whole groups of 8) and occupies
        ``v_latent_frames(head_px)`` = ``(head_px - 1)//8 + 1`` latents,
      * a TAIL band must be a MULTIPLE OF 8 px (whole groups of 8 counted back
        from the end, never touching the keyframe) and occupies ``tail_px // 8``
        latents.

    169/25/24 -> 4 head latents and 3 tail latents. The asymmetry is why the
    default glue is 25/24 rather than 25/25 (VERIFICATION_LOG §55.5).
    """
    return tail_px // VIDEO_TIME_FACTOR


def a_frames_for_px(pixel_frames: int, fps: float) -> int:
    """Audio latent-frame count for a pixel span at ``fps``."""
    return round(pixel_frames / float(fps) * AUDIO_LATENTS_PER_SEC)


@dataclass
class ChainLayout:
    """Fully-resolved geometry for one chain (all deterministic from inputs)."""

    # inputs (echoed)
    clip_frames: list[int]
    fps: float
    kv: int
    v_tile: int
    v_adv: int
    kt_v: int

    # stage-1
    seg_latent: list[int]           # per-segment stage-1 video latent frames
    seg_audio: list[int]            # per-segment stage-1 audio latent frames
    f_total: int                    # assembled video latent frames
    total_px: int                   # assembled pixel frames
    a_total: int                    # assembled audio latent frames
    ka_list: list[int]              # per-join audio overlap (len = n_clips - 1)

    # stage-2 tiles
    v_tiles: list[tuple[int, int]]  # (start_latent, len_latent)
    a_tiles: list[tuple[int, int]]  # (start_latent, len_latent)
    audio_adv: int
    kt_a: int
    n_tiles: int

    # junction indices (0-based LAST-frame-of-segment; boundary is J / J+1)
    segment_seam_junctions: list[int] = field(default_factory=list)
    tile_seam_junctions: list[int] = field(default_factory=list)
    all_junctions: list[int] = field(default_factory=list)

    duration_sec: float = 0.0

    # ── video-to-video continuation (source head) geometry ───────────────────
    # Populated ONLY when ``compute_chain_layout`` is called with
    # ``source_context_px`` (a source video's tail is frozen as the head of
    # clip-0). All None/0 for a normal chain (source-less path unchanged).
    source_context_px: int | None = None   # frozen context pixel-frame span
    n_ctx_v: int = 0                        # frozen video-latent head frames
    n_ctx_a: int = 0                        # frozen audio-latent head frames
    trim_px: int = 0                        # pixel frames trimmed off the front
    # 0-based pixel index (UNTRIMMED timeline) of the last frozen context frame
    # (source->new boundary is v2v_context_junction_px / +1); the delivered
    # (trimmed) mp4 starts the NEW content at pixel index 0.
    v2v_context_junction_px: int | None = None
    new_frames_px: int = 0                  # delivered new pixel frames (post-trim)

    # ── retake (temporal inpainting) geometry ────────────────────────────────
    # PRIMARY DATA ONLY — deliberately just the two inputs. Every derived number
    # (n_head_v / n_tail_v / n_head_a / n_tail_a / free_middle) is recomputed by
    # a pure function inside :meth:`to_dict` rather than stored here, so there is
    # exactly ONE definition of each and no way for a stored copy to drift out of
    # step with the function the engine calls. Both None on every non-retake
    # chain (the ``retake`` sub-dict is then absent from ``to_dict``).
    retake_window_px: int | None = None      # == clip_frames[0] (the whole window)
    retake_glue_px: tuple[int, int] | None = None   # (head_px, tail_px)

    def to_dict(self) -> dict:
        d = {
            "clip_frames": self.clip_frames,
            "fps": self.fps,
            "kv": self.kv,
            "v_tile": self.v_tile,
            "v_adv": self.v_adv,
            "kt_v": self.kt_v,
            "seg_latent": self.seg_latent,
            "seg_audio": self.seg_audio,
            "f_total_latent": self.f_total,
            "total_px": self.total_px,
            "a_total": self.a_total,
            "ka_list": self.ka_list,
            "video_tiles": [list(t) for t in self.v_tiles],
            "audio_tiles": [list(t) for t in self.a_tiles],
            "audio_adv": self.audio_adv,
            "kt_a": self.kt_a,
            "n_tiles": self.n_tiles,
            "segment_seam_junctions": self.segment_seam_junctions,
            "tile_seam_junctions": self.tile_seam_junctions,
            "all_junctions": self.all_junctions,
            "duration_sec": self.duration_sec,
        }
        if self.source_context_px is not None:
            d["v2v"] = {
                "source_context_px": self.source_context_px,
                "n_ctx_v": self.n_ctx_v,
                "n_ctx_a": self.n_ctx_a,
                "trim_px": self.trim_px,
                "v2v_context_junction_px": self.v2v_context_junction_px,
                "new_frames_px": self.new_frames_px,
            }
        if self.retake_glue_px is not None:
            head_px, tail_px = self.retake_glue_px
            window_px = int(self.retake_window_px or 0)
            n_head_a, n_tail_a = retake_audio_glue_latents(
                window_px=window_px, head_px=head_px, tail_px=tail_px,
                fps=self.fps, a_win=self.a_total,
            )
            d["retake"] = {
                "window_px": window_px,
                "head_px": head_px,
                "tail_px": tail_px,
                "n_head_v": v_latent_frames(head_px),
                "n_tail_v": v_tail_latents(tail_px),
                "n_head_a": n_head_a,
                "n_tail_a": n_tail_a,
                # PIXEL-domain span that is actually regenerated: [head_px,
                # window_px - tail_px). Reported as a 2-list for JSON.
                "free_middle_px": [head_px, window_px - tail_px],
            }
        return d


def compute_chain_layout(
    clip_frames: list[int],
    fps: float,
    kv: int = DEFAULT_OVERLAP_FRAMES,
    *,
    v_tile: int = STAGE2_V_TILE,
    v_adv: int = STAGE2_V_ADV,
    source_context_px: int | None = None,
    retake_glue_px: tuple[int, int] | None = None,
) -> ChainLayout:
    """Resolve the full chain geometry from clip pixel-frame counts + fps + K_v.

    Faithful generalisation of the S2 spike (reduces to it for uniform clips).
    Raises ValueError on geometrically impossible inputs (surfaced by the API
    validator / engine before any GPU work).

    ``source_context_px`` (video-to-video continuation): when given, the tail of
    an uploaded source video (``source_context_px`` pixel frames, 8n+1) is
    VAE-encoded and frozen as the HEAD of clip-0's timeline. This is the SINGLE
    SOURCE OF TRUTH for:
      * ``n_ctx_v`` = frozen video-latent head frames = ``v_latent_frames(px)``,
      * ``n_ctx_a`` = frozen audio-latent head frames = ``a_frames_for_px(px)``,
      * ``trim_px`` = pixel frames (and the matching audio samples, derived at
        runtime from the decoded sample-rate) cut off the FRONT of the decoded
        output so the delivered mp4 is the NEW part only,
      * the source->new junction index (untrimmed timeline) for metadata/harness.
    clip_frames[0] is the TOTAL clip-0 timeline (context head + new tail); the
    frozen context occupies its first ``n_ctx_v`` stage-1 latent frames.

    ``retake_glue_px`` (temporal inpainting, mutually exclusive with
    ``source_context_px``): ``(head_px, tail_px)`` glue bands frozen at BOTH ends
    of a SINGLE-clip window whose length is ``clip_frames[0]``. This is the
    SINGLE SOURCE OF TRUTH for the whole retake geometry — the app validator, the
    engine and the mock all resolve the frozen band sizes from here, so they
    cannot disagree. All the geometric rejections live here (window bounds and
    8n+1 grid, head 8n+1 / tail multiple-of-8, a free middle in BOTH the video
    and the audio latent domains, and the ``n_tiles == 1`` invariant the both-side
    freeze was validated under). ``None`` -> byte-identical to before.
    """
    n = len(clip_frames)
    if n < 1:
        raise ValueError("chain requires at least one clip")
    kt_v = v_tile - v_adv

    n_ctx_v = n_ctx_a = trim_px = new_frames_px = 0
    v2v_context_junction_px: int | None = None
    if source_context_px is not None:
        if source_context_px < 1:
            raise ValueError(f"source_context_px must be >= 1 (got {source_context_px})")
        if (source_context_px - 1) % VIDEO_TIME_FACTOR != 0:
            raise ValueError(
                f"source_context_px must be 8n+1 (got {source_context_px})"
            )
        if source_context_px >= clip_frames[0]:
            raise ValueError(
                f"source_context_px ({source_context_px}) must be < clip 0 "
                f"total frames ({clip_frames[0]}) so a NEW tail remains"
            )
        n_ctx_v = v_latent_frames(source_context_px)
        if n_ctx_v > v_tile:
            max_ctx_px = px_from_v_latent(v_tile)
            raise ValueError(
                f"source_context_px ({source_context_px}) -> frozen video-latent "
                f"head n_ctx_v={n_ctx_v} exceeds stage-2 tile size v_tile={v_tile}: "
                "the variant-B hard-freeze only covers stage-2 TILE 0, so the "
                "frozen head must fit entirely inside the first tile. Max allowed "
                f"source_context_px for this v_tile is {max_ctx_px}."
            )
        n_ctx_a = a_frames_for_px(source_context_px, fps)
        trim_px = source_context_px
        new_frames_px = clip_frames[0] - source_context_px
        v2v_context_junction_px = source_context_px - 1

    # ── retake: window + glue-band validation (video-latent domain) ──────────
    retake_window_px: int | None = None
    if retake_glue_px is not None:
        if source_context_px is not None:
            raise ValueError(
                "retake_glue_px and source_context_px are mutually exclusive "
                "(a retake window freezes BOTH ends of one clip; a V2V "
                "continuation freezes the head of clip 0 and generates onward)"
            )
        if n != 1:
            raise ValueError(
                f"retake requires exactly 1 clip (the window itself); got {n}"
            )
        retake_window_px = int(clip_frames[0])
        head_px, tail_px = (int(retake_glue_px[0]), int(retake_glue_px[1]))
        max_window_px = retake_max_window_px(v_tile)
        if (retake_window_px - 1) % VIDEO_TIME_FACTOR != 0:
            raise ValueError(
                f"retake window must be 8n+1 pixel frames (got {retake_window_px})"
            )
        if not (RETAKE_WINDOW_MIN_PX <= retake_window_px <= max_window_px):
            raise ValueError(
                f"retake window ({retake_window_px}) must be within "
                f"[{RETAKE_WINDOW_MIN_PX}, {max_window_px}] for v_tile={v_tile}: "
                "the whole window is refined as ONE stage-2 tile (that is the "
                "only geometry the both-side freeze was validated under)"
            )
        # Head on the 8n+1 grid, tail on the multiple-of-8 grid — see
        # :func:`v_tail_latents` for why the two ends differ (causal VAE).
        if (head_px - 1) % VIDEO_TIME_FACTOR != 0:
            raise ValueError(f"retake head_px must be 8n+1 (got {head_px})")
        if head_px < 9:
            raise ValueError(
                f"retake head_px must be >= 9 (got {head_px}): a 1-frame head "
                "freezes only the lone keyframe latent"
            )
        if tail_px % VIDEO_TIME_FACTOR != 0:
            raise ValueError(
                f"retake tail_px must be a multiple of 8 (got {tail_px})"
            )
        if tail_px < VIDEO_TIME_FACTOR:
            raise ValueError(f"retake tail_px must be >= 8 (got {tail_px})")
        if head_px + tail_px >= retake_window_px:
            raise ValueError(
                f"retake glue bands ({head_px} + {tail_px}) must leave a free "
                f"middle inside the {retake_window_px}-frame window"
            )

    seg_latent = [v_latent_frames(f) for f in clip_frames]
    seg_audio = [a_frames_for_px(f, fps) for f in clip_frames]

    if any(kv >= L for L in seg_latent):
        raise ValueError(
            f"overlap_frames (K_v={kv}) must be < every clip's stage-1 latent "
            f"frames {seg_latent}"
        )

    # ── assembled stage-1 length ─────────────────────────────────────────────
    f_total = sum(seg_latent) - (n - 1) * kv
    total_px = px_from_v_latent(f_total)
    a_total = a_frames_for_px(total_px, fps)

    # ── retake: free-middle checks in BOTH latent domains ────────────────────
    # The pixel-domain ``head_px + tail_px < window_px`` check above is NOT
    # sufficient: the two latent grids quantise differently (video 8:1 with a
    # causal keyframe, audio 25 latents/sec), so a pixel middle that looks free
    # can still leave zero free latents on one side. Both are checked here,
    # after ``f_total`` / ``a_total`` are known.
    if retake_glue_px is not None:
        head_px, tail_px = (int(retake_glue_px[0]), int(retake_glue_px[1]))
        n_head_v = v_latent_frames(head_px)
        n_tail_v = v_tail_latents(tail_px)
        if n_head_v + n_tail_v >= f_total:
            raise ValueError(
                f"retake glue leaves no free VIDEO middle: n_head_v={n_head_v} + "
                f"n_tail_v={n_tail_v} >= f_total={f_total}"
            )
        n_head_a, n_tail_a = retake_audio_glue_latents(
            window_px=int(retake_window_px or 0), head_px=head_px,
            tail_px=tail_px, fps=fps, a_win=a_total,
        )
        if n_head_a + n_tail_a >= a_total:
            raise ValueError(
                f"retake glue leaves no free AUDIO middle: n_head_a={n_head_a} + "
                f"n_tail_a={n_tail_a} >= a_total={a_total}"
            )

    # Per-join audio overlap K_a so assembled audio == a_total EXACTLY:
    #   sum(a_seg) - sum(ka) = a_total  =>  sum(ka) = sum(a_seg) - a_total.
    ka_list: list[int] = []
    if n > 1:
        sum_ka = sum(seg_audio) - a_total
        n_join = n - 1
        if sum_ka < n_join:
            raise ValueError(
                f"degenerate audio overlap (sum_ka={sum_ka} < joins={n_join}); "
                "clips too short for a continuous audio crossfade"
            )
        base_ka = sum_ka // n_join
        rem_ka = sum_ka % n_join
        ka_list = [base_ka + (1 if j < rem_ka else 0) for j in range(n_join)]
        assert sum(ka_list) == sum_ka

    # ── stage-2 tile layout (video latent domain) ────────────────────────────
    v_starts = list(range(0, max(1, f_total - kt_v), v_adv))
    if not v_starts or v_starts[-1] + v_tile < f_total:
        v_starts.append(max(0, f_total - v_tile))
        v_starts = sorted(set(v_starts))
    v_tiles: list[tuple[int, int]] = []
    for vs in v_starts:
        ve = min(vs + v_tile, f_total)
        v_tiles.append((vs, ve - vs))
    n_tiles = len(v_tiles)
    # Retake's whole reason for a window cap: the both-side freeze was only ever
    # validated on a SINGLE stage-2 tile (VERIFICATION_LOG §55.2). The window
    # bound above is derived from exactly this, so a failure here means the
    # bound and the tiler have drifted apart — an assert, not a user-facing
    # ValueError.
    assert retake_glue_px is None or n_tiles == 1, (n_tiles, retake_window_px, v_tile)

    # audio tiles, time-aligned to the video advance
    audio_adv = round(v_adv * VIDEO_TIME_FACTOR / float(fps) * AUDIO_LATENTS_PER_SEC)
    a_len_full = a_frames_for_px(px_from_v_latent(v_tile), fps)
    kt_a = a_len_full - audio_adv
    a_tiles: list[tuple[int, int]] = []
    for i, (vs, vlen) in enumerate(v_tiles):
        tile_px = px_from_v_latent(vlen)
        alen = a_frames_for_px(tile_px, fps)
        as_ = i * audio_adv
        a_tiles.append((as_, alen))

    # verify audio tiling reassembles to a_total and stays in-bounds
    if n_tiles > 1:
        a_reassembled = a_tiles[0][1]
        for i in range(1, n_tiles):
            a_reassembled = a_reassembled + a_tiles[i][1] - kt_a
            if a_tiles[i][0] + a_tiles[i][1] > a_total:
                raise ValueError(f"audio tile {i} out of bounds vs a_total={a_total}")
            if (a_tiles[i - 1][0] + a_tiles[i - 1][1]) - a_tiles[i][0] != kt_a:
                raise ValueError(f"audio overlap mismatch at tile join {i}")
        if a_reassembled != a_total:
            raise ValueError(f"audio reassembly {a_reassembled} != a_total {a_total}")
        if a_tiles[-1][0] + a_tiles[-1][1] != a_total:
            raise ValueError("last audio tile does not reach a_total")

    # ── junctions (0-based last-frame-of-segment; boundary J / J+1) ───────────
    # Segment i's first NEW latent frame = sum_{k<i} L[k] - (i-1)*K_v.
    segment_seam_junctions: list[int] = []
    for i in range(1, n):
        new_latent = sum(seg_latent[:i]) - (i - 1) * kv
        segment_seam_junctions.append(px_from_v_latent(new_latent) - 1)

    # Tile i>=1 fresh content begins at global latent vs_i + kt_v.
    tile_seam_junctions: list[int] = []
    for i in range(1, n_tiles):
        new_latent = v_tiles[i][0] + kt_v
        tile_seam_junctions.append(px_from_v_latent(new_latent) - 1)

    def _spread(js: list[int]) -> list[int]:
        out: list[int] = []
        for j in js:
            out += [j - 1, j, j + 1]
        return sorted({x for x in out if 0 <= x < total_px - 1})

    all_junctions = sorted(
        set(_spread(segment_seam_junctions) + _spread(tile_seam_junctions))
    )

    return ChainLayout(
        clip_frames=list(clip_frames),
        fps=float(fps),
        kv=kv,
        v_tile=v_tile,
        v_adv=v_adv,
        kt_v=kt_v,
        seg_latent=seg_latent,
        seg_audio=seg_audio,
        f_total=f_total,
        total_px=total_px,
        a_total=a_total,
        ka_list=ka_list,
        v_tiles=v_tiles,
        a_tiles=a_tiles,
        audio_adv=audio_adv,
        kt_a=kt_a,
        n_tiles=n_tiles,
        segment_seam_junctions=segment_seam_junctions,
        tile_seam_junctions=tile_seam_junctions,
        all_junctions=all_junctions,
        duration_sec=round(total_px / float(fps), 3),
        source_context_px=source_context_px,
        n_ctx_v=n_ctx_v,
        n_ctx_a=n_ctx_a,
        trim_px=trim_px,
        v2v_context_junction_px=v2v_context_junction_px,
        new_frames_px=new_frames_px,
        retake_window_px=retake_window_px,
        retake_glue_px=(
            None if retake_glue_px is None
            else (int(retake_glue_px[0]), int(retake_glue_px[1]))
        ),
    )


# ── audio-to-video (A2V) geometry — uploaded-audio freeze windows ────────────
# These are the SINGLE SOURCE OF TRUTH shared by the app (preflight: is the
# uploaded audio long enough?) and the engine (how many audio-latent frames to
# slice off the VAE-encoded upload, and which global window each stage-1 segment
# freezes). Both are pure functions of the same geometry ``compute_chain_layout``
# resolves — ``audio_latents_required`` recomputes a_total from the same three
# lines, ``audio_segment_windows`` reads a resolved layout — so app and engine
# agree byte-for-byte. They cover 1..24 clips: with several clips the uploaded
# audio is still ONE track over the whole assembled timeline, and each stage-1
# segment freezes its own window on it.
def audio_latents_required(
    clip_frames: list[int], fps: float, kv: int = DEFAULT_OVERLAP_FRAMES
) -> int:
    """Total audio-latent frames the assembled chain timeline requires.

    Equal to ``compute_chain_layout(...).a_total`` — the audio-latent count for
    the assembled pixel timeline (per-join K_v overlaps already folded into
    ``total_px``), computed via :func:`a_frames_for_px`. An uploaded audio track
    that VAE-encodes to fewer than this many latent frames is a truncation error
    at the caller (video length is authoritative; audio is truncated, never
    padded — matches upstream a2vid).

    Computes a_total DIRECTLY (seg_latent -> f_total -> total_px -> a_total)
    instead of resolving a whole :class:`ChainLayout`. a_total depends only on
    the clip lengths, fps and K_v — never on the stage-2 window (``v_tile`` /
    ``v_adv``) — so this function has NO business running the layout's stage-2
    audio-tile reassembly check. Routing it through ``compute_chain_layout``
    used to drag that window-dependent check into the A2V length PREFLIGHT,
    where a config the request's own window accepts could still raise here and
    surface as a 500 (measured: a single 321-frame clip @ 23.976fps, which the
    DEFAULT window rejects and ``high_resolution`` accepts). The stage-2
    reassembly check still runs — in the request validator, on the window the
    request actually asked for.

    The K_v floor is kept verbatim (same ValueError text as
    ``compute_chain_layout``): it is pure clip geometry, window-independent, and
    callers already expect a chain with K_v >= a clip's latent length to fail.
    """
    if len(clip_frames) < 1:
        raise ValueError("chain requires at least one clip")
    seg_latent = [v_latent_frames(f) for f in clip_frames]
    if any(kv >= L for L in seg_latent):
        raise ValueError(
            f"overlap_frames (K_v={kv}) must be < every clip's stage-1 latent "
            f"frames {seg_latent}"
        )
    f_total = sum(seg_latent) - (len(seg_latent) - 1) * kv
    return a_frames_for_px(px_from_v_latent(f_total), fps)


def audio_segment_windows(layout: ChainLayout) -> list[tuple[int, int]]:
    """Per stage-1 segment ``(start, len)`` window on the GLOBAL audio timeline.

    For audio-to-video each stage-1 segment hard-freezes the slice of the
    uploaded audio latent that lands under it. Segment ``i`` occupies
    ``seg_audio[i]`` audio-latent frames and consecutive segments overlap by
    ``ka_list[i]`` (the same per-join audio crossfade the engine assembles with),
    so segment ``i`` starts at ``sum(seg_audio[:i]) - sum(ka_list[:i])`` and the
    last window ends exactly at ``a_total``. For a single clip this reduces to
    ``[(0, a_total)]``. Pure function of a resolved :class:`ChainLayout`, and
    clip-count agnostic — this is what long A2V (2..24 clips) rides on.
    """
    windows: list[tuple[int, int]] = []
    start = 0
    for i, alen in enumerate(layout.seg_audio):
        windows.append((start, alen))
        if i < len(layout.ka_list):
            start += alen - layout.ka_list[i]
    return windows


def video_segment_windows(layout: ChainLayout) -> list[tuple[int, int]]:
    """Per stage-1 segment ``(start_px, len_px)`` window on the GLOBAL reference
    timeline.

    The clip-wise IC-LoRA reference (§1-15) is ONE long uploaded video laid over
    the assembled chain timeline; this says which pixel-frame slice of it belongs
    to stage-1 segment ``i``. Segment ``i`` begins at global stage-1 latent
    ``s_i = sum(seg_latent[:i]) - i * kv`` (the same accumulation
    ``segment_seam_junctions`` uses, one K_v earlier — that one reports the first
    NEW latent, i.e. ``s_i + kv``). The video VAE is CAUSAL: latent 0 covers
    pixel 0 alone and latent ``f >= 1`` covers pixels ``8f-7 .. 8f``, so a
    segment's LOCAL pixel 0 sits at global pixel ``8 * s_i`` and the segment
    spans its own ``clip_frames[i]`` pixel frames from there.

    Windows are per STAGE-1 SEGMENT because that is the only place a reference is
    injected: LoRAs are applied to the stage-1 ledger only, and stage 2 refines
    the already-assembled timeline with no reference conditioning at all.

    No V2V / retake terms appear here on purpose. A reference is mutually
    exclusive with ``source_video`` and with retake at the API level
    (``api/models.py``), so a layout that carries ``source_context_px`` /
    ``retake_glue_px`` can never reach this function with a reference attached —
    there is no trimmed head to compensate for.

    Adjacent windows overlap by exactly ``8 * kv - 7`` pixel frames (17 at the
    default K_v=3), which is precisely the pixel span the K_v latent carry-over
    covers — so the reference the two neighbours see across a seam is the same
    footage, and the seam stays consistent for free. For a single clip this
    reduces to ``[(0, clip_frames[0])]``, identical to the single-clip
    ``/generate`` path. With 8n+1 clip lengths the last window ends exactly at
    ``total_px - 1``, i.e. the windows need exactly ``total_px`` reference frames.
    Pure function of a resolved :class:`ChainLayout`, clip-count agnostic (1..24).
    """
    windows: list[tuple[int, int]] = []
    for i, clip_px in enumerate(layout.clip_frames):
        s_i = sum(layout.seg_latent[:i]) - i * layout.kv
        windows.append((VIDEO_TIME_FACTOR * s_i, clip_px))
    return windows


def freeze_mask_values(
    mask_value: float,
    tail_mask_value: float | None = None,
    audio_mask_value: float | None = None,
) -> tuple[float, float, float, float]:
    """Resolve the four frozen-band strengths a chain denoise step writes.

    Returns ``(video_head, video_tail, audio_head, audio_tail)`` for
    ``engine.pipeline.chain_pipeline._denoise_av_with_carry``. A "strength" here
    is the denoise-mask value the frozen band is pinned to: ``1 - overlap_strength``
    for an ordinary carry-over seam, ``0.0`` for a hard freeze.

    Two independent overrides, both defaulting to ``None`` = "same as
    ``mask_value``", so every pre-override call site is bit-identical:

    * ``tail_mask_value`` — the retake two-sided freeze; splits HEAD from TAIL.
    * ``audio_mask_value`` — long A2V; splits VIDEO from AUDIO. The uploaded
      audio window must be hard-frozen (0.0) while the video seam keeps carrying
      over at the user's ``overlap_strength``. Collapsing the two into one
      ``mask_value=0.0`` (what pre-long-A2V A2V did, harmlessly, because a
      single clip freezes no video head at all) would weld every segment seam
      shut from clip 2 onward and discard ``overlap_strength`` without a word.

    Given BOTH overrides the TAIL one wins on the audio tail. That pairing is
    retake + A2V, which the API rejects outright, so this is a definition rather
    than a behaviour anything relies on.

    Pure, so the app venv can test it without torch.
    """
    v_head = float(mask_value)
    a_head = v_head if audio_mask_value is None else float(audio_mask_value)
    if tail_mask_value is None:
        return v_head, v_head, a_head, a_head
    tail = float(tail_mask_value)
    return v_head, tail, a_head, tail


# ── retake: audio glue rounding ("H-A1′") + tail token addressing ────────────
# The audio patchifier is CAUSAL (``is_causal=True``), so audio latent ``i`` does
# NOT cover the naive ``[i/25, (i+1)/25)`` second — it covers
# ``[max(4i-3,0)/100, (4i+1)/100)``, i.e. the same 1/25s width shifted 0.75
# frames EARLIER. Machine-checked against the wheel's own
# ``get_patch_grid_bounds`` (VERIFICATION_LOG §55.2).
#
# RESIDUAL UNCERTAINTY, recorded on purpose (VERIFICATION_LOG §55.6): what is
# verified is the PATCHIFIER's declared support, NOT that the audio VAE ENCODER
# actually responds over that same span. The alignment probe (``runs/align/``)
# came back INDETERMINATE — all 7 probes missed both the naive and the causal
# prediction by 1-3 frames. Do not write "the causal model was confirmed".
# The scanning rule below is chosen precisely so that swapping
# :func:`audio_latent_support_sec` for a better time-support model automatically
# updates the frozen band sizes with no other change.
def audio_latent_support_sec(i: int) -> tuple[float, float]:
    """Time span (seconds) audio latent frame ``i`` actually supports.

    ``[max(4i - 3, 0)/100, (4i + 1)/100)`` — the wheel's causal patch grid.
    """
    return max(4 * i - 3, 0) / 100.0, (4 * i + 1) / 100.0


def retake_audio_glue_latents(
    *, window_px: int, head_px: int, tail_px: int, fps: float, a_win: int
) -> tuple[int, int]:
    """Frozen audio-latent counts ``(n_head_a, n_tail_a)`` for the glue bands.

    The rule ("H-A1′") is stated as an INTENT and solved by SCANNING rather than
    by a rounding formula: freeze only those audio latents whose whole time
    support lies inside the frozen VIDEO band. Nothing that overlaps the
    regenerated middle is ever frozen.

    Why not the obvious "floor the head, ceil the tail" formula: measured against
    the real causal support that rule overshoots into the regenerated middle by
    up to 30 ms in 379 of 508 grid combinations (75%); symmetric rounding
    overshoots in 100% of them, by up to 46.7 ms. The scan overshoots in none,
    at the price of under-freezing by at most 35 ms — under one audio latent
    frame and shorter than one 24fps video frame (VERIFICATION_LOG §55.2).

    Pure geometry; ``a_win`` is the window's total audio-latent count
    (``ChainLayout.a_total``).
    """
    eps = 1e-9
    head_s = head_px / float(fps)
    tail_start_s = (window_px - tail_px) / float(fps)

    # Head: the longest run of leading latents that ENDS inside the head band.
    n_head = 0
    for i in range(a_win):
        if audio_latent_support_sec(i)[1] <= head_s + eps:
            n_head = i + 1
        else:
            break
    # Tail: the longest run of trailing latents that STARTS inside the tail band.
    first_tail = a_win
    for i in range(a_win - 1, -1, -1):
        if audio_latent_support_sec(i)[0] >= tail_start_s - eps:
            first_tail = i
        else:
            break
    n_head = max(0, min(n_head, a_win))
    first_tail = max(0, min(first_tail, a_win))
    return n_head, a_win - first_tail


def retake_tail_token_range(latent_frames: int, hw: int, n_tail: int) -> tuple[int, int]:
    """ABSOLUTE ``[lo, hi)`` video token range of the trailing ``n_tail`` latents.

    ``hw`` is the LATENT height*width (patch size 1), so latent frame ``f``
    occupies tokens ``[f*hw, (f+1)*hw)``.

    NEGATIVE INDEXING IS FORBIDDEN HERE. Conditioning items APPEND tokens at the
    end of the token dimension, so ``mask[:, -n_tail*hw:]`` grabs conditioning
    tokens instead of the last video latents and silently under-freezes the tail.
    For the same reason the caller's bound check must be against
    ``latent_frames * hw`` and never against ``mask.shape[1]`` (which may already
    be longer, so it would not defend anything). VERIFICATION_LOG §55.2.
    """
    if not (0 < n_tail <= latent_frames):
        raise ValueError(
            f"n_tail must be within (0, latent_frames={latent_frames}]; got {n_tail}"
        )
    return (latent_frames - n_tail) * hw, latent_frames * hw
