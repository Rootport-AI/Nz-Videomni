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
from api.errors import APIError, job_busy, source_audio_not_found, source_video_not_found
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

    # Single-job guard: atomically reserve, else 409 JOB_BUSY.
    job = context.job_store.create_chain_if_idle(request)
    if job is None:
        raise job_busy()

    background_tasks.add_task(context.pipeline_manager.run_chain_job, job)

    return GenerateChainResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        num_clips=len(request.clips),
    )
