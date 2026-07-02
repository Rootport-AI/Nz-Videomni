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
from api.errors import job_busy
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
