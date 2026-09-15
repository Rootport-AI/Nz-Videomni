"""Inpainting for LTX 2.5 (台帳 §3-150): repaint a masked region, two stages.

What this is
------------
The 2.5 half of "Inpainting", and the sibling of :mod:`engine25.outpaint25`:
same two-stage skeleton, same 2.3-shipped In-Outpainting IC-LoRA, same
Laplacian pyramid blend -- with the geometry INVERTED. Outpainting invents a
band AROUND the footage; inpainting repaints a region INSIDE it.

    text encode
      -> freeze the SOURCE WINDOW's audio latent (the cut window carries the
         audio; the canvas deliberately does not)
      -> encode the GREEN-FILLED CANVAS as the IC-LoRA reference conditioning
      -> STAGE 1 at half resolution, with that reference attached
      -> decode stage 1 to pixels (half res)
      -> decode the MASK, paste it on the canvas, halve it
      -> DE-GREEN, TWICE: the pad bands AND the masked region become this
         stage's own generated pixels
      -> BLEND 1: Laplacian pyramid, dilation 5, against that canvas
      -> 2x PIXEL upscale
      -> tiled VAE re-encode at full resolution
      -> STAGE 2 at full resolution, no reference conditioning
      -> decode stage 2 to pixels (full res)
      -> decode the MASK again, at full resolution; DE-GREEN twice again
      -> BLEND 2: Laplacian pyramid, dilation 2
      -> RESTORE: every pixel the stage-2 dilated mask never reached goes back
         to the canvas' own bytes -- and ``mask_proof`` records what was touched
      -> CROP off the right/bottom pad bands (uint8, lossless)
      -> mux the source window's ORIGINAL waveform -> one mp4.

Design canon: ``Docs/INPAINTING_DESIGN.md`` (§6 is the mask contract, §7 the
procedure). The PROCEDURE is 2.3's, which gate G6 validated on
``engine/pipeline/inpaint_pipeline.py``; the CODE is not, because 2.5 drives
official 1.2.0 blocks (``prompt_encoder`` / ``audio_conditioner`` /
``image_conditioner`` / ``stage`` / ``video_decoder`` / ``audio_decoder``)
instead of a ``ModelLedger``.

WHAT IS BORROWED, AND WHY NOTHING IS COPIED
-------------------------------------------
This module defines exactly TWO things of its own -- the mask decode and the
result class -- and imports everything else. Each borrowing says what a second
copy would cost:

* ``engine.inpaint.canvas`` (``InpaintGeometry`` / ``place_mask_on_canvas`` /
  ``half_res_mask`` / ``fill_pad_bands_with_generated_`` /
  ``fill_mask_with_generated_`` / ``restore_and_measure_``) -- the geometry and
  the restore are torch-only arithmetic with no wheel in them, so BOTH engines
  import the same module. The API validator builds the same geometry, so a copy
  here would let the canvas the app validated and the canvas this engine blends
  disagree. ``restore_and_measure_`` in particular is what produces
  ``mask_proof``, the one number that proves a real mask reached a real engine.
* ``engine.outpaint.pyramid_blend.blend_video_u8`` -- the blend IS the feature,
  shared by all four drivers (2.3 in/outpaint, 2.5 in/outpaint). A second
  implementation would be a second set of seams to gate.
* :mod:`engine25.outpaint25`'s twenty-odd helpers -- the pixel boundaries, the
  audio freeze and its proof, the mux plan, the reference encode, the stage-2
  re-encode and the ``ltx25`` sub-dict. Those six in particular were lifted out
  of ``run_outpaint`` INTO module scope for this driver to call (see that
  module's own §"The two ImageConditioner calls"), because each is either a
  MEASURED path (the ``channels_last_3d`` re-layout: 27188 MB -> 6174 MB) or a
  user-visible CONTRACT (the ``ltx25`` sub-dict lands in ``metadata.json``
  verbatim). A copy of either would be a second answer to one question.
* ``engine25.chain25`` / ``engine25.pipeline25`` / ``engine25.ltxcore_compat``
  -- engine25's standing rules: one dtype, one sampler vocabulary, one entry
  point to the wheel.

What this module OWNS
---------------------
* :func:`_decode_mask_u8` -- 2.3's ``engine.pipeline.common.decode_mask_video``
  cannot be imported here: that module is bound to 2.3's wheel at import time
  (``ltx_core`` names that differ in 1.2.0) and its decoder is
  ``decode_video_from_file``, whose signature is not the one
  :mod:`engine25.ltxcore_compat` re-exports. The CONTRACT is identical, line for
  line, and the two decoders yield the same ``(1, H, W, C)`` uint8 RGB frame, so
  the same "red channel >= 128" produces the same plane. ``tests/
  test_ltx25_inpaint.py`` rounds an H.264 mask through this one and holds it
  against that contract.
* :class:`InpaintResult` -- :class:`~engine25.outpaint25.OutpaintResult` with
  one ``ClassVar`` changed, so ``as_dict``'s additive key is named for THIS job
  kind. Nothing else differs, which is why it is a subclass rather than a twin.

Deliberate differences from 2.3's inpaint driver, all recorded
--------------------------------------------------------------
The same three :mod:`engine25.outpaint25` carries, for the same reasons, and
they are recorded in the ``ltx25`` sub-dict rather than left to be inferred:

* **SAMPLER.** Stage 1 runs the ANCESTRAL euler this checkpoint was distilled
  for (:data:`engine25.chain25.STAGE1_SAMPLER`); 2.3 runs plain euler in both
  stages. Stage 2 is deterministic euler in both engines.
* **STAGE-2 AUDIO INIT.** Stage 2 CARRIES stage 1's audio latent and writes the
  frozen head back over it, then re-freezes -- ``"stage1_carry"``. 2.3 rebuilds
  from zeros. Identical under a full freeze; they differ only over a PARTIAL
  one, where 2.3 hands stage 2 zeros for the un-frozen tail.
* **RE-ENCODE TILING.** The decode's resolved tiling config plus
  ``channels_last_3d`` for the duration, rather than 2.3's narrowed spatial
  tiles.

What this module deliberately does NOT do
-----------------------------------------
* **It restores ONCE, after the FINAL blend only.** A restore between the
  stages would change stage 2's input (it re-encodes the blended pixels as its
  initial latent), so the promise "outside the mask is the original" is made
  where it can be kept and nowhere else.
* **It decodes the mask TWICE rather than keeping both resolutions.** A
  full-resolution 1920x1088x481 mask is 1.0 GB as uint8 and the half-resolution
  one another 0.25 GB; decoding again costs a few seconds of ffmpeg and keeps
  only one of them resident at a time. 2.3 ruled the same way.
* **It de-greens with TWO calls, not one.** Outpainting's pad bands are the
  whole generated region, so one call covers it. Here the pad bands and the
  mask are disjoint regions with different shapes -- the bands come from the
  geometry, the mask from the video.
* **It adds no progress phase.** The vocabulary is the same four names the app
  already maps, with the same KNOWN CONSEQUENCE ``outpaint25`` documents (the
  single-generate bar stands still at 0.50 through decode 1 + blend 1 + upscale
  + re-encode) -- and here that stretch is longer still, by the two mask
  decodes and the third canvas decode.

Run notes
---------
One job at a time, like everything else in this engine. The mode is
``with torch.no_grad():`` -- engine25's ONE mode, end to end (VERIFICATION_LOG
§81; ``tests/test_ltx25_outpaint.py`` scans every module here for the other
one, comments included).
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from engine.inpaint.canvas import (
    InpaintGeometry,
    fill_mask_with_generated_,
    fill_pad_bands_with_generated_,
    half_res_mask,
    place_mask_on_canvas,
    restore_and_measure_,
)
from engine.outpaint.pyramid_blend import blend_video_u8
from engine25.chain25 import (
    DTYPE,
    STAGE1_ANCESTRAL_ETA,
    STAGE1_SAMPLER,
    STAGE2_SEED_OFFSET,
    _stage1_sampler_kwargs,
    _vram_summary,
)
from engine25.ltxcore_compat import (
    AUTO_TILING,
    DISTILLED_SIGMAS,
    AudioLatentShape,
    GaussianNoiser,
    ModalitySpec,
    SimpleDenoiser,
    VideoPixelShape,
    cleanup_memory,
    decode_video_by_frame,
    encode_video,
    ensure_tiling_config,
    get_video_chunks_number,
    tiling_scale_factors_for_vae,
)
from engine25.outpaint25 import (
    _PIXEL_CHUNK_FRAMES,
    OutpaintResult,
    ProgressFn,
    _audio_freeze,
    _audio_init,
    _blend_chunk_size,
    _canvas_u8,
    _decoded_to_u8,
    _encode_chunk_count,
    _encode_reference_conditionings,
    _freeze_proof,
    _freeze_source_audio,
    _iter_encode_chunks,
    _ltx25_block,
    _mux_audio,
    _mux_plan,
    _phase_peak_mb,
    _reencode_stage2,
    _require_frames,
    _resolve_stage2_sigmas,
    _upscale_u8,
)
from engine25.pipeline25 import (
    STAGE_1_DENOISE,
    STAGE_2_DENOISE,
    _log_ignored,
    _peaks,
    validate_geometry,
)
from engine25.reference25 import resolve_reference_downscale_factor

logger = logging.getLogger(__name__)

__all__ = ["InpaintError", "InpaintResult", "run_inpaint"]

#: The job kind, said once: it names the additive ``as_dict`` key, the worker's
#: ``done`` key, the metadata sub-dict and every log line's prefix.
_LABEL = "inpaint"

class InpaintError(RuntimeError):
    """An inpaint request the LTX 2.5 engine cannot run."""


# ---------------------------------------------------------------------------
# The mask: this module's ONE decode
# ---------------------------------------------------------------------------


def _decode_mask_u8(
    *,
    mask_path: str,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
    threshold: int = 128,
) -> torch.Tensor:
    """Decode the mask video to ``(F, 1, height, width)`` uint8 (0/255) on the CPU.

    The 2.5 twin of ``engine.pipeline.common.decode_mask_video``, and
    deliberately its sibling rather than a call into it: that module binds 2.3's
    wheel at import time and reaches for ``decode_video_from_file``, whose
    signature differs in 1.2.0. What this DOES call is the one decoder
    :mod:`engine25.ltxcore_compat` re-exports, which yields the same
    ``(1, H, W, C)`` uint8 RGB frame (``frame.to_rgb().to_ndarray()``) -- so the
    same red-channel rule produces the same plane, which
    ``tests/test_ltx25_inpaint.py`` rounds through real H.264 to show.

    Sits next to :func:`~engine25.outpaint25._canvas_u8`'s import, at module
    level, for TWO reasons: it is the same uint8 boundary as that decode, and
    ``decode_video_by_frame`` is a module global here so a test can substitute a
    double for it without a model or a file.

    The list of things it deliberately does NOT do is the whole specification
    (``Docs/INPAINTING_DESIGN.md`` §6.2):

    * **it does not resize.** A mask is a statement about which pixels of a
      SPECIFIC picture get repainted; rescaling it would move the boundary by a
      sub-pixel amount the caller cannot see and cannot correct. The contract is
      that the mask arrives at the source's own resolution or the request is
      refused with a 422, so a size mismatch here means the API guard was
      bypassed -- and this raises rather than papering over it;
    * **it does not normalise.** There is no VAE in this path. The values that
      come out are 0 and 255 and nothing else;
    * **it does not interpolate between the two levels.** The plugin writes the
      mask through Media Foundation and the app re-encodes nothing, but H.264's
      4:2:0 chroma and its deblocking filter still leave a soft grey ring around
      every edge; ``threshold`` is what turns that ring back into a decision.

    The RED channel is taken and the other two dropped: for a genuinely grey
    mask all three are equal, and for one that somehow is not, red is the
    channel ``video_io.fill_mask_green_mp4``'s ``lut`` reads too.

    ``num_frames`` is both the decode cap and an assertion. A short mask would
    silently leave the tail of the window unrepainted, so a shortfall raises --
    in :func:`~engine25.outpaint25._require_frames`' words, under this driver's
    own label -- and zero frames gets its own message, because "unreadable" and
    "short" are diagnosed differently.
    """
    if int(num_frames) <= 0:
        raise ValueError(f"inpaint mask: num_frames must be >= 1, got {num_frames}")

    planes: list[torch.Tensor] = []
    for frame in decode_video_by_frame(
        path=str(mask_path), device=device, frame_cap=int(num_frames)
    ):
        if frame.ndim != 4 or int(frame.shape[0]) != 1:
            raise ValueError(
                f"inpaint mask video {mask_path}: expected one (1, H, W, C) frame per "
                f"step, got {tuple(frame.shape)}"
            )
        got_h, got_w = int(frame.shape[1]), int(frame.shape[2])
        if (got_h, got_w) != (int(height), int(width)):
            raise ValueError(
                f"inpaint mask video {mask_path} is {got_w}x{got_h} but the source is "
                f"{int(width)}x{int(height)}; the mask is never resized "
                "(Docs/INPAINTING_DESIGN.md §6.2)"
            )
        # (1, H, W, C) -> (H, W), red channel only, thresholded to 0/255.
        red = frame[0, :, :, 0]
        planes.append((red >= int(threshold)).to(torch.uint8).mul_(255).cpu())
        del frame, red

    if not planes:
        raise ValueError(f"inpaint mask decoded to 0 frames: {mask_path}")
    _require_frames(
        "the mask video", len(planes), int(num_frames),
        source=str(mask_path), label=_LABEL,
    )
    return torch.stack(planes, dim=0).unsqueeze(1)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InpaintResult(OutpaintResult):
    """What one :func:`run_inpaint` produced.

    :class:`~engine25.outpaint25.OutpaintResult` with ONE thing changed: the
    name ``as_dict`` gives its single additive key. Everything the worker's
    ``done`` builder reads -- the eight attributes by name, plus
    ``as_dict()`` for the ``GENERATE_REPORT`` line -- is inherited unchanged, so
    the worker needs no branch and the app still finds ``peak_vram_mb`` for the
    job-VRAM gate.

    A SUBCLASS rather than a twin dataclass, and ``_JOB_KEY`` a ``ClassVar``
    rather than a field, because a ``ClassVar`` is not a dataclass field: the
    field-derivation test keeps seeing exactly the generation contract plus
    ``metadata``, for this class as for its parent.

    :attr:`width` / :attr:`height` are the DELIVERED size -- the source's, not
    the canvas' -- because the pad bands are cropped off before the encode and
    these two name what the mp4 actually comes out at. The canvas dimensions are
    in the ``inpaint`` metadata block, where the geometry belongs.
    """

    _JOB_KEY: ClassVar[str] = _LABEL


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


# No decorator: the mode is the ``with torch.no_grad():`` below, as on every
# engine25 path -- the other mode is banned here (VERIFICATION_LOG §81).
def run_inpaint(  # noqa: PLR0913, PLR0915 -- one linear procedure; splitting it would hide the order
    pipeline: Any,
    *,
    prompt: str,
    canvas_path: str,
    source_path: str | None,
    mask_path: str,
    geometry: InpaintGeometry,
    num_frames: int,
    frame_rate: float,
    num_steps: int,
    seed: int,
    output_path: str,
    ic_loras: list[tuple] | None = None,
    ic_reference: tuple[str, float] | None = None,
    ic_attention_strength: float = 1.0,
    blend_dilation_stage1: int = 5,
    blend_dilation_stage2: int = 2,
    freeze_source_audio: bool = True,
    stage2_sigmas: list[float] | None = None,
    ignored: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
) -> InpaintResult:
    """Repaint ``mask_path``'s white region inside ``canvas_path`` -> one mp4.

    ``pipeline`` is an :class:`engine25.pipeline25.Ltx25Pipeline` -- the loaded
    model, whose official blocks this function drives directly instead of going
    through ``DistilledPipeline.__call__``.

    ``canvas_path`` is the lossless, video-only canvas built by
    ``services.video_io.fill_mask_green_mp4``: the cut window with the mask's
    white region painted #66FF00 and the right/bottom bands padded out to the
    128-multiple canvas. ``ic_reference`` points at that SAME file (the app
    substitutes it for the uploaded reference, so the IC-LoRA plumbing needs no
    changes at all). ``source_path`` is the CUT WINDOW -- the same footage the
    canvas was built from, still carrying the audio the canvas deliberately does
    not.

    ``mask_path`` is the uploaded mask video at the SOURCE resolution. It is
    decoded here rather than read off the canvas because the blend needs the
    region as NUMBERS, not as green pixels -- and it is decoded twice, once per
    stage, rather than held at both resolutions (see the module docstring).

    ``geometry`` describes the canvas and the source rectangle anchored at its
    top-left corner, and is the ONLY source of both resolutions -- there is no
    ``width``/``height`` argument, so the canvas the app validated and the
    canvas this generates cannot disagree. ``num_frames`` / ``frame_rate`` are
    the generation timeline, which the app has already checked the source and
    the mask both cover.

    ``num_steps`` is carried and reported but never acted on, exactly as in
    outpainting and in the chain: the distilled schedule is fixed at 8 + 3
    sigmas, and stage 2's is further fixed at
    :data:`~engine25.outpaint25.OUTPAINT_STAGE2_SIGMAS` -- the SAME schedule,
    imported rather than restated, because the two features start stage 2 from
    the same kind of re-encode.

    ``stage2_sigmas`` is an ENGINE-INTERNAL experiment knob, not a request
    field. ``None`` -- every real job -- uses that constant.

    Returns an :class:`InpaintResult` -- a superset of the plain generation's
    result, so the worker's ``done`` builder needs no branch.
    """
    # ── Geometry: the app's rules, restated by the two owners of them ─────────
    # ``validate_geometry`` is the two-stage pipeline's own backstop (multiples
    # of 64, 8n+1 frames) and ``geometry.validate()`` is the canvas module's
    # (multiples of 128, an even source, a source no smaller than 256 a side).
    # BOTH, because they check different things and each is the last line of
    # defence for its own: a payload that reached the engine another way -- the
    # selftest CLI, a future MCP tool -- has passed neither.
    width = int(geometry.canvas_width)
    height = int(geometry.canvas_height)
    validate_geometry(width, height, int(num_frames))
    geometry.validate()
    # The ONE place stage 1's geometry is computed, so the half-resolution
    # de-green and the half-resolution mask can never disagree about where the
    # source rectangle ends.
    half_w, half_h, half_sw, half_sh = geometry.half_dims()
    _log_ignored(ignored)

    dp = pipeline.pipeline            # the official DistilledPipeline
    stage = pipeline.stage            # Ltx25ProgressStage
    device: torch.device = pipeline.device
    vram = pipeline.vram

    encode_fps = int(round(float(frame_rate)))
    if encode_fps < 1:
        raise InpaintError(f"frame_rate={frame_rate} rounds to {encode_fps} fps")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lora_entries = list(ic_loras or [])
    # Resolved BEFORE any model is built: it is a header read, and discovering
    # "this LoRA declares no reference factor" two minutes into a job would be a
    # poor trade. It also RAISES when a reference arrives with no adapters,
    # which is the one combination the in/outpainting contract cannot run -- the
    # official workflow has no LoRA-free path at all.
    reference_factor = resolve_reference_downscale_factor(lora_entries, ic_reference)

    stage2_values = _resolve_stage2_sigmas(stage2_sigmas)
    chunk_size = _blend_chunk_size()
    # THE INTEGER, not ``chunk_size``. ``_blend_chunk_size()`` returns None for
    # "the default", which ``blend_video_u8`` understands and the four SHARED
    # helpers below do NOT -- ``half_res_mask`` / ``fill_mask_with_generated_``
    # / ``restore_and_measure_`` and the pixel chunking all take a plain int and
    # would raise ``TypeError`` on None. 2.3 spells the same resolution
    # ``blend_chunk = chunk_size or 8``; here the default already has a name.
    pixel_step = chunk_size or _PIXEL_CHUNK_FRAMES

    full_shape = VideoPixelShape(1, int(num_frames), height, width, float(frame_rate))
    a_total = int(AudioLatentShape.from_video_pixel_shape(full_shape).to_torch_shape()[2])

    logger.info(
        "inpaint %dx%d canvas (source %dx%d, pads r/b=%d/%d) / %d frames @ %.3f fps "
        "(encode %d fps) seed=%d loras=%d reference=%s mask=%s dilation=%d/%d "
        "freeze_audio=%s stage2_sigmas=%s -> %s",
        width, height, geometry.source_width, geometry.source_height,
        geometry.pad_right, geometry.pad_bottom, int(num_frames), float(frame_rate),
        encode_fps, int(seed), len(lora_entries),
        "no" if ic_reference is None
        else f"{Path(str(ic_reference[0])).name} strength={float(ic_reference[1]):.3f} "
             f"factor={reference_factor} attn={float(ic_attention_strength):.3f}",
        Path(str(mask_path)).name,
        int(blend_dilation_stage1), int(blend_dilation_stage2),
        bool(freeze_source_audio), stage2_values, out_path,
    )

    # set_loras BEFORE begin_job, and unconditionally: begin_job releases the
    # PREVIOUS job's attachment, and a job that asks for no adapter must clear
    # rather than inherit one. Same two lines, same order, as every other path.
    stage.set_loras(lora_entries)
    stage.begin_job()
    stage.progress = progress
    pipeline.prompt_encoder.progress = progress
    vram.phases.clear()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.time()

    with torch.no_grad():
        # ── 10_prompt_encode ──────────────────────────────────────────────────
        # ONE prompt, ONE call. The phase, its two sub-phases and the ``encode``
        # progress events are all emitted by ``Ltx25PromptEncoder`` itself.
        (encoded,) = pipeline.prompt_encoder([str(prompt)])
        video_ctx, audio_ctx = encoded.video_encoding, encoded.audio_encoding
        del encoded

        # ── 12_audio_conditioning: freeze the SOURCE WINDOW's audio ───────────
        # BEFORE the video encoder and the transformer, because the audio
        # encoder is the cheapest model in the job (~46 MB) and doing it here
        # means it is never resident alongside either of them.
        #
        # THE WINDOW, not the upload: ``source_path`` is ``_inpaint_window.mp4``,
        # the cut the app made before it painted the canvas. UNDERRUN IS NOT
        # FATAL, for the reason outpainting gives -- the audio is guidance for a
        # video that is already fully specified.
        audio = _freeze_source_audio(
            audio_conditioner=pipeline.audio_conditioner,
            device=device,
            vram=vram,
            source_path=source_path,
            enabled=freeze_source_audio,
            a_total=a_total,
            label=_LABEL,
        )
        frozen_audio = audio.latent
        source_waveform = audio.waveform
        audio_sr = audio.sampling_rate
        audio_available = audio.available_frames
        n_frozen = audio.frozen_frames
        source_had_audio = audio.source_had_audio

        # ── 11_reference_encode: the green-filled canvas, at HALF resolution ──
        # THE TILING CONFIG IS THE DECODE'S, HANDED OVER RESOLVED. Resolving it
        # here, with no models built, is where the free-memory branch would be
        # at its most optimistic -- the same hoist ``run_outpaint`` makes.
        tiling_config = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=tiling_scale_factors_for_vae(dp.video_decoder.checkpoint_path),
            video_shape=full_shape,
            vae_checkpoint_path=dp.video_decoder.checkpoint_path,
            diffvae_optimization=dp.video_decoder.diffvae_optimization,
            device=device,
        )

        conds_ref, reference_frames = _encode_reference_conditionings(
            image_conditioner=dp.image_conditioner,
            ic_reference=ic_reference,
            reference_factor=reference_factor,
            height=height,
            width=width,
            num_frames=num_frames,
            tiling_config=tiling_config,
            ic_attention_strength=ic_attention_strength,
            device=device,
            vram=vram,
        )

        # ── 21_stage1_denoise: half resolution, green canvas attached ─────────
        # NO INITIAL VIDEO LATENT and no video band: stage 1 generates the whole
        # canvas from noise, guided by the reference. The audio is the only
        # modality with anything frozen.
        stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
        init_a1 = _audio_init(
            frozen_audio, frames_frozen=n_frozen, full_shape=full_shape, device=device
        )
        band_a1 = _audio_freeze(init_a1, n_frozen)

        # NO ``outer_index`` / ``outer_total``: inpainting is one clip, so the
        # per-step events go out as THREE POSITIONAL ARGUMENTS and a receiver
        # written for a single generation is unchanged.
        stage.announce(STAGE_1_DENOISE, vram_phase="21_stage1_denoise")
        vstate, astate = stage(
            denoiser=SimpleDenoiser(video_ctx, audio_ctx),
            sigmas=stage_1_sigmas,
            noiser=GaussianNoiser(
                generator=torch.Generator(device=device).manual_seed(int(seed))
            ),
            width=half_w,
            height=half_h,
            frames=int(num_frames),
            fps=float(frame_rate),
            video=ModalitySpec(
                context=video_ctx, conditionings=conds_ref, initial_latent=None
            ),
            audio=ModalitySpec(
                context=audio_ctx, conditionings=band_a1, initial_latent=init_a1
            ),
            # The ancestral stage-1 sampler, with the ``eta > 0`` assertion the
            # frozen band rests on carried inside it.
            **_stage1_sampler_kwargs(
                STAGE1_SAMPLER, STAGE1_ANCESTRAL_ETA, int(seed), pipeline.dtype
            ),
        )
        stage1_latent = vstate.latent.detach().clone()
        stage1_audio = astate.latent.detach().clone()
        s1_audio_head = (
            stage1_audio[:, :, :n_frozen].detach().clone() if frozen_audio is not None else None
        )
        del vstate, astate, conds_ref, band_a1, init_a1
        cleanup_memory()

        # ── 24_stage1_decode ─────────────────────────────────────────────────
        # THIS IS THE WHOLE REASON inpainting cannot ride the official two-stage
        # call: that call upsamples in LATENT space between the stages, and the
        # blend has to happen on PIXELS.
        vram.reset()
        decode1_started = time.perf_counter()
        gen1 = torch.Generator(device=device).manual_seed(int(seed))
        stage1_pixels = _decoded_to_u8(
            dp.video_decoder(stage1_latent, tiling_config, gen1),
            progress=progress,
            chunks=get_video_chunks_number(int(num_frames), tiling_config),
        )
        vram.record("24_stage1_decode", time.perf_counter() - decode1_started)
        del stage1_latent, gen1
        cleanup_memory()

        canvas_half = _canvas_u8(
            video_path=str(canvas_path),
            height=half_h,
            width=half_w,
            frame_cap=int(num_frames),
            device=device,
        )
        _require_frames(
            "the stage-1 decode", int(stage1_pixels.shape[0]), int(num_frames),
            source="the half-resolution VAE decode", label=_LABEL,
        )
        canvas_half = canvas_half[: int(num_frames)]
        stage1_pixels = stage1_pixels[: int(num_frames)]

        # ── 25_blend1: Laplacian pyramid against the green-filled canvas ─────
        vram.reset()
        blend1_started = time.perf_counter()
        # PASTE, THEN HALVE. Downscaling before the paste would area-average the
        # mask across the pad boundary and bleed it into a band that is cropped
        # off at the end.
        mask_half = half_res_mask(
            place_mask_on_canvas(
                _decode_mask_u8(
                    mask_path=str(mask_path),
                    num_frames=int(num_frames),
                    height=int(geometry.source_height),
                    width=int(geometry.source_width),
                    device=device,
                ),
                geometry,
            ),
            half_h,
            half_w,
            chunk_size=pixel_step,
        )
        # DE-GREEN, TWICE. The coarse pyramid levels mix the canvas' DC
        # component into the generated area whatever the mask says, so what
        # bleeds outwards must be the picture the model just drew rather than a
        # flat #66FF00. Two calls because the pad bands and the mask are
        # DISJOINT regions with different shapes -- the bands come from the
        # geometry, the mask from the video.
        fill_pad_bands_with_generated_(
            canvas_half,
            generated=stage1_pixels,
            source_height=half_sh,
            source_width=half_sw,
        )
        fill_mask_with_generated_(
            canvas_half, generated=stage1_pixels, mask=mask_half, chunk_size=pixel_step
        )
        blended_half = blend_video_u8(
            stage1_pixels,
            canvas_half,
            mask_half,
            mask_low_res_dilation=int(blend_dilation_stage1),
            chunk_size=chunk_size,
            device=device,
        )
        vram.record("25_blend1", time.perf_counter() - blend1_started)
        del stage1_pixels, canvas_half, mask_half
        cleanup_memory()

        # ── 26_pixel_upscale ─────────────────────────────────────────────────
        vram.reset()
        upscale_started = time.perf_counter()
        upscaled = _upscale_u8(
            blended_half, height=height, width=width, device=device, step=pixel_step
        )
        vram.record("26_pixel_upscale", time.perf_counter() - upscale_started)
        del blended_half
        cleanup_memory()

        # ── 27_stage2_encode: the SECOND ImageConditioner build ──────────────
        # The measured path, called rather than copied: ``channels_last_3d`` for
        # the duration and a ``finally`` that restores the contiguous layout,
        # which is a CORRECTNESS requirement (torch's bf16 Conv3d produces
        # different latents under the two layouts and the same encoder object is
        # reachable again within the process).
        vram.reset()
        encode2_started = time.perf_counter()

        stage2_init = _reencode_stage2(
            dp.image_conditioner, upscaled, tiling_config, device
        )
        vram.record("27_stage2_encode", time.perf_counter() - encode2_started)
        del upscaled
        cleanup_memory()

        # ── 22_stage2_denoise: full resolution, NO reference conditioning ────
        stage2_sigma_tensor = torch.tensor(
            stage2_values, dtype=torch.float32, device=device
        )
        # Computed ONCE and handed to BOTH modalities. Stage 2 starts from a
        # re-encode rather than from noise, so its noise scale must be the
        # schedule's own first sigma -- and if a caller overrides the schedule,
        # the scale has to follow it.
        noise_scale2 = float(stage2_sigma_tensor[0].item())

        # STAGE 1's AUDIO, CARRIED, with the frozen head written back over it
        # and re-frozen. Re-freezing is mandatory rather than belt-and-braces:
        # stage 2 re-noises at ``noise_scale2``, which would destroy a
        # stage-1-only freeze outright. The CARRY (rather than 2.3's zeros) is
        # the intentional difference recorded in the ``ltx25`` sub-dict.
        init_a2 = stage1_audio.detach().clone()
        if frozen_audio is not None:
            init_a2[:, :, :n_frozen] = frozen_audio
        band_a2 = _audio_freeze(init_a2, n_frozen)

        stage.announce(STAGE_2_DENOISE, vram_phase="22_stage2_denoise")
        vstate2, astate2 = stage(
            denoiser=SimpleDenoiser(video_ctx, audio_ctx),
            sigmas=stage2_sigma_tensor,
            noiser=GaussianNoiser(
                generator=torch.Generator(device=device).manual_seed(
                    int(seed) + STAGE2_SEED_OFFSET
                )
            ),
            width=width,
            height=height,
            frames=int(num_frames),
            fps=float(frame_rate),
            video=ModalitySpec(
                context=video_ctx,
                conditionings=[],
                noise_scale=noise_scale2,
                initial_latent=stage2_init.to(DTYPE),
            ),
            audio=ModalitySpec(
                context=audio_ctx,
                conditionings=band_a2,
                noise_scale=noise_scale2,
                initial_latent=init_a2,
            ),
            # Neither ``stepper`` nor ``loop``: stage 2 is DETERMINISTIC Euler in
            # both engines and upstream, and passing nothing is how
            # ``DiffusionStage``'s own default is said rather than restated.
        )
        final_v = vstate2.latent.detach().clone()
        final_a = astate2.latent.detach().clone()
        s2_audio_head = (
            final_a[:, :, :n_frozen].detach().clone() if frozen_audio is not None else None
        )
        del vstate2, astate2, stage2_init, init_a2, band_a2, stage1_audio
        cleanup_memory()

        # ── 28_stage2_decode ─────────────────────────────────────────────────
        vram.reset()
        decode2_started = time.perf_counter()
        gen2 = torch.Generator(device=device).manual_seed(int(seed))
        stage2_pixels = _decoded_to_u8(
            dp.video_decoder(final_v, tiling_config, gen2),
            progress=progress,
            chunks=get_video_chunks_number(int(num_frames), tiling_config),
        )[: int(num_frames)]
        vram.record("28_stage2_decode", time.perf_counter() - decode2_started)
        del final_v, gen2
        cleanup_memory()

        # ── 29_blend2: the final blend, at full resolution ───────────────────
        vram.reset()
        blend2_started = time.perf_counter()
        canvas_full = _canvas_u8(
            video_path=str(canvas_path),
            height=height,
            width=width,
            frame_cap=int(num_frames),
            device=device,
        )[: int(num_frames)]
        _require_frames(
            "the stage-2 decode", int(stage2_pixels.shape[0]), int(num_frames),
            source="the full-resolution VAE decode", label=_LABEL,
        )
        # The SECOND decode of the mask, at full resolution this time. Holding
        # both resolutions at once buys nothing and costs 1.25 GB at 1920x1088.
        mask_full = place_mask_on_canvas(
            _decode_mask_u8(
                mask_path=str(mask_path),
                num_frames=int(num_frames),
                height=int(geometry.source_height),
                width=int(geometry.source_width),
                device=device,
            ),
            geometry,
        )
        fill_pad_bands_with_generated_(
            canvas_full,
            generated=stage2_pixels,
            source_height=int(geometry.source_height),
            source_width=int(geometry.source_width),
        )
        fill_mask_with_generated_(
            canvas_full, generated=stage2_pixels, mask=mask_full, chunk_size=pixel_step
        )
        final_pixels = blend_video_u8(
            stage2_pixels,
            canvas_full,
            mask_full,
            mask_low_res_dilation=int(blend_dilation_stage2),
            chunk_size=chunk_size,
            device=device,
        )
        vram.record("29_blend2", time.perf_counter() - blend2_started)
        del stage2_pixels, canvas_full
        cleanup_memory()

        # ── 31_restore: outside the dilated mask, the original bytes ─────────
        # A FRESH decode of the canvas, not the de-greened copy above: that one
        # had its mask region AND its pad bands overwritten with generated
        # pixels, so it is no longer the original anywhere it matters. Outside
        # the mask the canvas IS the source window (that is what
        # ``fill_mask_green_mp4`` writes), which is why the source does not have
        # to be decoded separately here. The pad bands come back green and are
        # cropped off two steps later.
        #
        # ITS OWN VRAM PHASE, numbered above the blends because it is a new
        # measurement this driver adds and ``30_decode_encode`` already names
        # the mux. Recorded HERE, in execution order, so the report reads in the
        # order the work happened.
        vram.reset()
        restore_started = time.perf_counter()
        canvas_restore = _canvas_u8(
            video_path=str(canvas_path),
            height=height,
            width=width,
            frame_cap=int(num_frames),
            device=device,
        )[: int(num_frames)]
        mask_proof = restore_and_measure_(
            blended=final_pixels,
            source=canvas_restore,
            mask=mask_full,
            dilation=int(blend_dilation_stage2),
            chunk_size=pixel_step,
            device=device,
        )
        vram.record("31_restore", time.perf_counter() - restore_started)
        logger.info("inpaint mask proof: %s", mask_proof)
        del canvas_restore, mask_full
        cleanup_memory()

        # ── CROP back to the source resolution ───────────────────────────────
        # The source is anchored at (0, 0), so this is a slice of the right and
        # bottom bands and nothing else -- lossless, at uint8, before the encode.
        final_pixels = final_pixels[
            :, : int(geometry.source_height), : int(geometry.source_width), :
        ].contiguous()
        logger.info(
            "inpaint crop: %dx%d canvas -> %dx%d delivered (pads r/b=%d/%d)",
            width, height, geometry.source_width, geometry.source_height,
            geometry.pad_right, geometry.pad_bottom,
        )

        # ── 30_decode_encode: the freeze proof, the mux, the mp4 ─────────────
        freeze_proof = _freeze_proof(
            frozen_audio=frozen_audio,
            s1_audio_head=s1_audio_head,
            s2_audio_head=s2_audio_head,
        )
        logger.info(
            "inpaint: audio freeze proof %s %s",
            {None: "N/A", True: "PASS", False: "FAIL"}[freeze_proof["pass"]],
            {k: v for k, v in freeze_proof.items() if k != "pass"},
        )
        del s1_audio_head, s2_audio_head

        mux_meta = _mux_plan(
            freeze_source_audio=bool(freeze_source_audio),
            source_waveform=source_waveform,
            audio_frozen_frames=n_frozen,
        )
        vram.reset()
        encode_started = time.perf_counter()
        mux_audio, muxed_samples = _mux_audio(
            mux_meta=mux_meta,
            source_waveform=source_waveform,
            audio_latent=final_a,
            audio_decoder=dp.audio_decoder,
            num_frames=num_frames,
            frame_rate=frame_rate,
            audio_sr=audio_sr,
            label=_LABEL,
        )
        del final_a
        cleanup_memory()

        chunks = _encode_chunk_count(int(num_frames), pixel_step)
        encode_video(
            video=_iter_encode_chunks(final_pixels, pixel_step),
            fps=encode_fps,
            audio=mux_audio,
            output_path=str(out_path),
            video_chunks_number=chunks,
        )
        vram.record("30_decode_encode", time.perf_counter() - encode_started)
        del final_pixels, mux_audio

    wall = time.time() - started

    if not out_path.is_file() or out_path.stat().st_size <= 0:
        raise InpaintError(f"the inpaint produced no/empty output: {out_path}")

    gc.collect()
    cleanup_memory()

    # The job's RESTING footprint after the collector has run -- the same marker
    # every other path records, so all of them are read the same way.
    vram.record("40_job_end", 0.0)

    summary = _vram_summary(vram.phases)
    peak_gib = summary.get("peak_allocated_gib")
    peak_mb = round(float(peak_gib) * 2**30 / 1e6, 1) if peak_gib is not None else 0.0

    metadata: dict[str, Any] = {
        # 2.3's ``inpaint`` sub-dict, key for key and in its order, so a client
        # that already parses a 2.3 inpaint job parses this one.
        _LABEL: {
            **geometry.as_dict(),
            "blend_dilation_stage1": int(blend_dilation_stage1),
            "blend_dilation_stage2": int(blend_dilation_stage2),
            "blend_chunk_size": chunk_size,
            "blend1_peak_vram_mb": _phase_peak_mb(vram.phases, "25_blend1"),
            "blend2_peak_vram_mb": _phase_peak_mb(vram.phases, "29_blend2"),
            "stage2_sigmas": list(stage2_values),
            "freeze_source_audio": bool(freeze_source_audio),
            "audio_frozen_latent_frames": int(n_frozen),
            "audio_latent_frames_required": int(a_total),
            "muxed_original_waveform": bool(mux_meta["muxed_original_waveform"]),
            "muxed_audio_samples": int(muxed_samples),
            "audio_sampling_rate": int(audio_sr),
            "canvas_path": str(canvas_path),
            "mask_path": str(mask_path),
            # The evidence that a mask was really decoded and really used. The
            # mock backend cannot fabricate these -- it never opens a video --
            # so a metadata.json carrying them is proof the real engine ran the
            # real mask (Docs/INPAINTING_DESIGN.md §7.6).
            "mask_proof": mask_proof,
            # ── 2.5-only additions ───────────────────────────────────────────
            # The same six facts the 2.5 outpaint block carries, for the same
            # reasons: the proof that the audio band actually held (None, never
            # 0.0, when nothing was frozen), which of the three audio outcomes
            # ran, and whether the vocoder did. ``source_had_audio`` is DISTINCT
            # from ``muxed_original_waveform``: a source can have a track and
            # still freeze nothing.
            "freeze_proof": freeze_proof,
            "audio_branch": mux_meta["audio_branch"],
            "vocoder_skipped": bool(mux_meta["vocoder_skipped"]),
            "source_had_audio": bool(source_had_audio),
            "encoded_audio_frames_available": int(audio_available),
            "source_path": None if source_path is None else str(source_path),
            # ...plus the one measurement 2.3 does not make. The restore is this
            # driver's third full-resolution canvas decode and its own float32
            # dilation, so it is the step most likely to set the job's peak on a
            # 16 GB card -- and it is the step an operator would narrow
            # ``LTX_OUTPAINT_BLEND_CHUNK`` for. Read out of the phase record
            # rather than measured again, exactly like the two blends above.
            "restore_peak_vram_mb": _phase_peak_mb(vram.phases, "31_restore"),
        },
        # The flat 2.3 keys. ``width``/``height`` are the DELIVERED size, not
        # the canvas: this is what the mp4 comes out at.
        "seed": int(seed),
        "width": int(geometry.source_width),
        "height": int(geometry.source_height),
        "num_frames": int(num_frames),
        "num_steps": int(num_steps),
        "wall_s": round(wall, 2),
        "vram_peak_mb": peak_mb,
        "vram_within_16gb": peak_mb < 16000,
        "output_mp4": str(out_path),
        # engine25-only facts, in their own sub-dict so the 2.3-shaped keys
        # above stay exactly the 2.3 set a client already parses. The SAME
        # builder outpainting uses -- the worker re-sends this dict as a
        # top-level ``ltx25`` key and the app writes it into metadata.json
        # verbatim, so two copies would be two answers to "what did 2.5 do".
        "ltx25": _ltx25_block(
            seed=seed,
            noise_scale2=noise_scale2,
            reference_frames=reference_frames,
            reference_factor=reference_factor,
            ic_attention_strength=ic_attention_strength,
            encode_fps=encode_fps,
            chunks=chunks,
            pixel_step=pixel_step,
            tiling_config=tiling_config,
            out_path=out_path,
            summary=summary,
            phases=vram.phases,
        ),
    }

    result = InpaintResult(
        output_path=str(out_path),
        seed=int(seed),
        # The DELIVERED size again, for the same reason: these two are what the
        # worker reports and what the app shows next to the finished file.
        width=int(geometry.source_width),
        height=int(geometry.source_height),
        num_frames=int(num_frames),
        frame_rate=float(frame_rate),
        encode_fps=encode_fps,
        # Inpainting takes no conditioning images: the canvas arrives as the
        # IC-LoRA REFERENCE, not as a keyframe. Reported as 0 rather than
        # omitted, because the field is part of the generation contract.
        num_images=0,
        size_bytes=out_path.stat().st_size,
        seconds=wall,
        metadata=metadata,
        phases=dict(vram.phases),
        video_chunks=chunks,
        tiling=None if tiling_config is None else repr(tiling_config),
        **_peaks(vram, device),
    )
    # ``GENERATED_OK`` DELIBERATELY, not a new marker: the worker's own
    # machine-readable line stays ``GENERATE_REPORT`` (which is what the
    # existing evidence collectors grep for) and this human summary keeps the
    # single path's marker, so an inpaint job reads like the generation it is.
    logger.info(
        "GENERATED_OK %.1fs peak_allocated=%sGiB peak_reserved=%sGiB rss_peak=%sGiB "
        "inpaint %dx%d -> %s",
        result.seconds, result.peak_allocated_gib, result.peak_reserved_gib,
        result.rss_peak_gib, result.width, result.height, result.output_path,
    )
    return result
