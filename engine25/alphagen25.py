"""AlphaGen for LTX 2.5: an RGB clip in, a grey matte clip of the same size out.

What this is
------------
The engine half of the "one-stage full-size" AlphaGen mode (台帳 §1-83; the
gate that justified it is ``Docs/VERIFICATION_LOG.md`` §151). AlphaGen is an
IC-LoRA (``alpha-gen``) that reads an RGB reference video and generates the
matte for it -- white opaque, black transparent, grey in between -- at the
reference's own size and frame count.

    text encode (the official empty prompt; the app sends one space)
      -> encode the reference at FULL size (factor 1), as the IC-LoRA
         reference conditioning
      -> STAGE 1 at FULL size, with that reference attached
      -> decode that latent straight to pixels
      -> encode one video-only mp4 at crf 0.

There is NO stage 2, NO latent upsampler and NO audio decode. The phases a job
records are ``10_*`` (written by the prompt encoder itself),
``11_reference_encode``, ``21_stage1_denoise``, ``30_decode_encode`` and
``40_job_end`` -- the absence of any ``22_*`` is the record that the job ran
one stage.

Why it is its own driver, and how it differs from the official one
-------------------------------------------------------------------
The official ``ICLoraPipeline`` has a ``skip_stage_2`` switch, but that runs
stage 1 at HALF the requested size and decodes the half-size latent as it is.
This product runs stage 1 at the FULL requested size instead -- the same load
as calling the official one-stage path with twice the dimensions, which is
row B of the gate in ``Docs/VERIFICATION_LOG.md`` §151. Neither
``Ltx25Pipeline.generate`` nor ``DistilledPipeline.__call__`` can express
that (both are two-stage with stage 1 fixed at half size), so this module
drives the official blocks directly, exactly as
:func:`engine25.inpaint25.run_inpaint` does. The light mode of the feature
needs none of this: it is the plain two-stage generation with a reference, and
the app sends no ``alpha_gen`` block for it.

The sampler is this product's 2.5 stage-1 ancestral euler
(:data:`engine25.chain25.STAGE1_SAMPLER`); the official ``ICLoraPipeline`` uses
plain euler. The gate's mattes were produced with the ancestral one. The audio
context goes into stage 1 as the official pipeline does, but the audio latent
is never decoded (``encode_video(audio=None)``).

What the shared helper is borrowed for
--------------------------------------
The reference encode is :func:`engine25.outpaint25._encode_reference_conditionings`
with ``full_resolution=True`` (read at the full size, not half) and
``require_frames=True`` (a reference that yields no frames is an error, not the
in/outpainting "generate without one" fallback -- a matte with no reference is
meaningless). It is called rather than copied because it encodes inside the
``image_conditioner`` loan, which is the measured memory path.

A note on "lossless"
--------------------
The output is written at ``crf=0``, but the official ``encode_video`` converts
to BT.709 LIMITED-range yuv420p (luma 16..235) before it writes, so even at
crf 0 the luma is quantised to 220 levels. When the app stretches the matte
back to full range, unused code values appear at regular gaps. "Lossless" here
means "the crf 19 lossy compression is avoided", not "every 8-bit level
survives". The light mode's output is the plain path's crf 19.

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
from pathlib import Path
from typing import Any

import torch

from engine25.chain25 import (
    STAGE1_ANCESTRAL_ETA,
    STAGE1_SAMPLER,
    _stage1_sampler_kwargs,
)
from engine25.ltxcore_compat import (
    AUTO_TILING,
    DISTILLED_SIGMAS,
    GaussianNoiser,
    ModalitySpec,
    SimpleDenoiser,
    VideoPixelShape,
    cleanup_memory,
    encode_video,
    ensure_tiling_config,
    get_video_chunks_number,
    tiling_scale_factors_for_vae,
)
from engine25.outpaint25 import ProgressFn, _encode_reference_conditionings
from engine25.pipeline25 import (
    STAGE_1_DENOISE,
    GenerationResult,
    _log_ignored,
    _peaks,
    validate_geometry,
)
from engine25.reference25 import resolve_reference_downscale_factor

logger = logging.getLogger(__name__)

__all__ = ["AlphaGenError", "run_alpha_gen"]

#: The crf the one-stage output is written at (see "A note on lossless").
ALPHA_GEN_CRF = 0


class AlphaGenError(RuntimeError):
    """An AlphaGen request the LTX 2.5 engine cannot run."""


# No decorator: the mode is the ``with torch.no_grad():`` below, as on every
# engine25 path -- the other mode is banned here (VERIFICATION_LOG §81).
def run_alpha_gen(  # noqa: PLR0913, PLR0915 -- one linear procedure; splitting it would hide the order
    pipeline: Any,
    *,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    output_path: str,
    ic_loras: list[tuple] | None,
    ic_reference: tuple[str, float] | None,
    ic_attention_strength: float = 1.0,
    ignored: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
) -> GenerationResult:
    """Generate the matte for ``ic_reference`` at ``width`` x ``height`` -> one mp4.

    ``pipeline`` is an :class:`engine25.pipeline25.Ltx25Pipeline`, whose official
    blocks this function drives directly. ``width`` / ``height`` are the CANVAS
    the app built (multiples of 64 -- ``validate_geometry``'s rule), and the ONE
    stage runs at exactly that size. ``ic_reference`` is the app-built,
    canvas-sized reference video; it is required.

    Returns a plain :class:`~engine25.pipeline25.GenerationResult`, so the
    worker's ``done`` builder treats the job as a plain generation.
    """
    width = int(width)
    height = int(height)
    num_frames = int(num_frames)
    validate_geometry(width, height, num_frames)
    if ic_reference is None:
        raise AlphaGenError(
            "alpha_gen requires a reference video: the matte is generated FROM it"
        )
    _log_ignored(ignored)

    dp = pipeline.pipeline            # the official DistilledPipeline
    stage = pipeline.stage            # Ltx25ProgressStage
    device: torch.device = pipeline.device
    vram = pipeline.vram

    encode_fps = int(round(float(frame_rate)))
    if encode_fps < 1:
        raise AlphaGenError(f"frame_rate={frame_rate} rounds to {encode_fps} fps")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lora_entries = list(ic_loras or [])
    # Resolved BEFORE any model is built: it is a header read. It also RAISES
    # when a reference arrives with no adapters.
    reference_factor = resolve_reference_downscale_factor(lora_entries, ic_reference)

    full_shape = VideoPixelShape(1, num_frames, height, width, float(frame_rate))

    logger.info(
        "alpha_gen one-stage %dx%d / %d frames @ %.3f fps (encode %d fps, crf %d) "
        "seed=%d loras=%d reference=%s strength=%.3f factor=%s attn=%.3f -> %s",
        width, height, num_frames, float(frame_rate), encode_fps, ALPHA_GEN_CRF,
        int(seed), len(lora_entries), Path(str(ic_reference[0])).name,
        float(ic_reference[1]), reference_factor, float(ic_attention_strength),
        out_path,
    )

    # set_loras BEFORE begin_job, and unconditionally: begin_job releases the
    # PREVIOUS job's attachment. Same lines, same order, as every other path.
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
        # The phase and its progress events are emitted by the encoder itself.
        (encoded,) = pipeline.prompt_encoder([str(prompt)])
        video_ctx, audio_ctx = encoded.video_encoding, encoded.audio_encoding
        del encoded

        # ── 11_reference_encode: the reference at FULL size ───────────────────
        # The tiling config is the decode's, resolved here and reused for both
        # the reference encode and the final decode -- the same hoist
        # ``run_inpaint`` makes.
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
            full_resolution=True,
            require_frames=True,
        )

        # ── 21_stage1_denoise: FULL size, reference attached ──────────────────
        # NO initial latent in either modality: the matte is generated from
        # noise, guided by the reference.
        stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
        stage.announce(STAGE_1_DENOISE, vram_phase="21_stage1_denoise")
        vstate, astate = stage(
            denoiser=SimpleDenoiser(video_ctx, audio_ctx),
            sigmas=stage_1_sigmas,
            noiser=GaussianNoiser(
                generator=torch.Generator(device=device).manual_seed(int(seed))
            ),
            width=width,
            height=height,
            frames=num_frames,
            fps=float(frame_rate),
            video=ModalitySpec(context=video_ctx, conditionings=conds_ref),
            audio=ModalitySpec(context=audio_ctx),
            **_stage1_sampler_kwargs(
                STAGE1_SAMPLER, STAGE1_ANCESTRAL_ETA, int(seed), pipeline.dtype
            ),
        )
        latent = vstate.latent
        # The audio latent is never decoded: the matte is video-only.
        del astate, vstate, conds_ref, video_ctx, audio_ctx
        cleanup_memory()

        # ── 30_decode_encode: the stage-1 latent straight to the mp4 ──────────
        chunks = get_video_chunks_number(num_frames, tiling_config)
        vram.reset()
        encode_started = time.perf_counter()
        decode_gen = torch.Generator(device=device).manual_seed(int(seed))
        encode_video(
            video=pipeline._with_decode_progress(
                dp.video_decoder(latent, tiling_config, decode_gen), chunks
            ),
            fps=encode_fps,
            audio=None,
            output_path=str(out_path),
            video_chunks_number=chunks,
            crf=ALPHA_GEN_CRF,
        )
        vram.record("30_decode_encode", time.perf_counter() - encode_started)
        del latent, decode_gen

    seconds = time.time() - started

    if not out_path.is_file() or out_path.stat().st_size <= 0:
        raise AlphaGenError(f"alpha_gen produced no/empty output: {out_path}")

    gc.collect()
    cleanup_memory()

    # The job's RESTING footprint after the collector has run -- the same
    # marker every other path records.
    vram.record("40_job_end", 0.0)

    result = GenerationResult(
        output_path=str(out_path),
        seed=int(seed),
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=float(frame_rate),
        encode_fps=encode_fps,
        # The reference rides as the IC-LoRA REFERENCE, not as a keyframe.
        num_images=0,
        size_bytes=out_path.stat().st_size,
        seconds=seconds,
        phases=dict(vram.phases),
        video_chunks=chunks,
        tiling=None if tiling_config is None else repr(tiling_config),
        **_peaks(vram, device),
    )
    logger.info(
        "GENERATED_OK %.1fs peak_allocated=%sGiB peak_reserved=%sGiB rss_peak=%sGiB "
        "alpha_gen %dx%d reference_frames=%d -> %s",
        result.seconds, result.peak_allocated_gib, result.peak_reserved_gib,
        result.rss_peak_gib, result.width, result.height, int(reference_frames),
        result.output_path,
    )
    return result
