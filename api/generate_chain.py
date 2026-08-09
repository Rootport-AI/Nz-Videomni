"""POST /generate/chain — start a clip-concatenation chain job (Phase 3).

Additive to the frozen single-``/generate`` contract: a base prompt + a list of
clip specs are generated sequentially (each seeded from the previous clip's
carry latent) and concatenated into one continuous ``output.mp4``. Reuses the
single-job guard (busy -> 409).
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import (
    APIError,
    job_busy,
    lora_control_unsupported_in_chain,
    lora_preprocess_conflict,
    lora_requires_reference,
    reference_requires_control_lora,
    reference_resolution_invalid,
    retake_video_not_found,
    source_audio_not_found,
    source_video_not_found,
)
from api.generate import spawn_job_thread
from api.models import GenerateChainRequest, GenerateChainResponse

router = APIRouter()


@router.post(
    "/generate/chain",
    response_model=GenerateChainResponse,
    status_code=202,
    dependencies=[Depends(require_auth)],
)
def generate_chain(
    request: GenerateChainRequest,
    background_tasks: BackgroundTasks,
    context: AppContext = Depends(get_context),
) -> GenerateChainResponse:
    # Validate clip-0 conditioning images exist up front (only clip 0 may carry
    # them; the model validator already enforces that).
    for ci in request.clips[0].conditioning_images:
        context.upload_store.path_for(ci.image_id)  # raises IMAGE_NOT_FOUND

    # V2V continuation: resolve the source video (404) and preflight it (422 for
    # too-short) BEFORE reserving a job — same up-front-failure discipline as the
    # conditioning-image check above (precedent api/generate.py).
    if request.source_video is not None:
        try:
            context.video_upload_store.path_for(request.source_video.video_id)
        except APIError:
            raise source_video_not_found(request.source_video.video_id)
        context.pipeline_manager.preflight_source_video(
            request.source_video, request.frame_rate
        )

    # A2V: resolve the source audio (404) and preflight it (422 for too-short)
    # BEFORE reserving a job — same up-front-failure discipline as source_video.
    if request.source_audio is not None:
        try:
            context.audio_upload_store.path_for(request.source_audio.audio_id)
        except APIError:
            raise source_audio_not_found(request.source_audio.audio_id)
        context.pipeline_manager.preflight_source_audio(
            request.source_audio,
            [c.num_frames for c in request.clips],
            request.frame_rate,
            request.overlap_frames,
        )

    # Retake: resolve the source video (404) and preflight the window (422 when
    # it runs past the upload, or when regenerate_audio=False was asked for on a
    # silent upload) BEFORE reserving a job — same up-front-failure discipline as
    # source_video/source_audio above. The window's GEOMETRY was already settled
    # by the schema + chain_math; what is checked here is only whether the
    # uploaded material actually contains it.
    if request.retake is not None:
        try:
            context.video_upload_store.path_for(request.retake.video_id)
        except APIError:
            raise retake_video_not_found(request.retake.video_id)
        context.pipeline_manager.preflight_retake_window(
            request.retake, request.clips[0].num_frames, request.frame_rate
        )

    # Reference-video CONTROL IC-LoRA (Phase C chain support, ADDITIVE, ALPHA
    # scope — clips=1 only, schema-enforced): validate the reference video up
    # front, mirroring api/generate.py 57-63.
    if request.reference_video_id is not None:
        context.video_upload_store.path_for(request.reference_video_id)  # 404 if missing
        # All CONTROL adapters use reference_downscale_factor=2, so the reference
        # is consumed at half output resolution on the 64-grid -- width/height not
        # divisible by 128 crashes the worker's VAE encode (mirrors api/generate.py).
        if request.width % 128 != 0 or request.height % 128 != 0:
            raise reference_resolution_invalid(request.width, request.height)

    # Style/character + reference-video CONTROL IC-LoRA (ADDITIVE): resolve every
    # requested adapter (404 unknown/missing) up front — same discipline as
    # api/generate.py — and inspect its kind:
    #   * a CONTROL adapter (union-control / pixel-spatial-upscaler) derives its
    #     conditioning from a reference video. In v1 a chain only carries a
    #     reference_video_id when it is exactly 1 clip (schema-enforced), so a
    #     control adapter on a >1-clip chain is still rejected outright
    #     (LORA_CONTROL_UNSUPPORTED_IN_CHAIN, unchanged pre-alpha behaviour); on a
    #     1-clip chain it instead needs the reference_video_id, exactly like the
    #     single-generate check (LORA_REQUIRES_REFERENCE);
    #   * conversely a reference video is ONLY consumable through a control
    #     adapter (its downscale factor comes from that adapter's metadata), so a
    #     reference + style-only chain is rejected here
    #     (REFERENCE_REQUIRES_CONTROL_LORA, mirrors api/generate.py);
    #   * a single reference video can only be turned into ONE control signal, so
    #     >1 distinct non-"none" preprocess kind is a conflict (Phase C, mirrors
    #     api/generate.py).
    preprocess_kinds: set[str] = set()
    control_names: list[str] = []
    for spec in request.loras:
        context.lora_registry.resolve(spec.name, spec.strength)  # 404 if unknown/missing
        entry = context.lora_registry.info(spec.name)
        if entry.kind == "control":
            control_names.append(spec.name)
        if entry.preprocess != "none":
            preprocess_kinds.add(entry.preprocess)
    if control_names:
        if len(request.clips) != 1:
            raise lora_control_unsupported_in_chain(control_names)
        if request.reference_video_id is None:
            raise lora_requires_reference(control_names)
    if request.reference_video_id is not None and not control_names:
        raise reference_requires_control_lora([spec.name for spec in request.loras])
    if len(preprocess_kinds) > 1:
        raise lora_preprocess_conflict(sorted(preprocess_kinds))

    # Single-job guard: atomically reserve, else 409 JOB_BUSY.
    job = context.job_store.create_chain_if_idle(request)
    if job is None:
        raise job_busy()

    # Mock backend -> BackgroundTasks (preserves TestClient sync semantics the
    # suite relies on); real backend -> dedicated daemon thread so a long chain
    # job doesn't starve the polling GET /jobs / self-POSTs (see api/generate.py,
    # including the start-failure guard that fails the job instead of orphaning it).
    if (context.config.model.backend or "auto").strip().lower() == "mock":
        background_tasks.add_task(context.pipeline_manager.run_chain_job, job)
    else:
        spawn_job_thread(context.pipeline_manager.run_chain_job, job)

    return GenerateChainResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        num_clips=len(request.clips),
    )
