"""POST /generate — start a generation job (spec 7.2 / 8.1)."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import job_busy, lora_preprocess_conflict
from api.models import GenerateRequest, GenerateResponse

router = APIRouter()


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=202,
    dependencies=[Depends(require_auth)],
)
def generate(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    context: AppContext = Depends(get_context),
) -> GenerateResponse:
    # Validate conditioning images exist up front (minimal I2V).
    for ci in request.conditioning_images:
        context.upload_store.path_for(ci.image_id)  # raises IMAGE_NOT_FOUND

    # Phase B IC-LoRA: validate the reference video + adapter names up front, the
    # same way conditioning image_ids are checked (fail at job creation, not deep
    # in the worker). The GenerateRequest validator already enforced the
    # loras<->reference_video_id all-or-nothing rule.
    if request.reference_video_id is not None:
        context.video_upload_store.path_for(request.reference_video_id)  # 404 if missing
    # Phase C: a single reference video can only be turned into ONE kind of
    # control signal, so >1 distinct non-"none" preprocess kind among the
    # requested loras is rejected up front (fail loud, minimal implementation).
    preprocess_kinds: set[str] = set()
    for spec in request.loras:
        _, _, preprocess = context.lora_registry.resolve(spec.name, spec.strength)  # 404 if unknown/missing
        if preprocess != "none":
            preprocess_kinds.add(preprocess)
    if len(preprocess_kinds) > 1:
        raise lora_preprocess_conflict(sorted(preprocess_kinds))

    # Single-job guard: atomically reserve, else 409 JOB_BUSY.
    job = context.job_store.create_if_idle(request)
    if job is None:
        raise job_busy()

    # Run the generation off the request path (Starlette runs sync tasks in a
    # threadpool, so ffmpeg / inference won't block the event loop).
    background_tasks.add_task(context.pipeline_manager.run_job, job)

    return GenerateResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
    )
