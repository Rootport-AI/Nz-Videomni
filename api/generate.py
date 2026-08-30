"""POST /generate — start a generation job (spec 7.2 / 8.1)."""

from __future__ import annotations

import threading

from fastapi import APIRouter, BackgroundTasks, Depends

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import (
    job_busy,
    lora_preprocess_conflict,
    lora_requires_reference,
    outpaint_preprocess_conflict,
    outpaint_source_mismatch,
    outpaint_source_too_short,
    reference_requires_control_lora,
    reference_resolution_invalid,
)
from api.models import GenerateRequest, GenerateResponse, JobStatus
from services import engines
from services.job_store import JobRecord, now_iso

router = APIRouter()


def spawn_job_thread(target, job: JobRecord) -> None:
    """Start the real-backend worker thread for ``job`` (shared with
    /generate/chain). If the thread cannot be started, the job must NOT stay
    queued: an is_active orphan would 409-block every later request until a
    server restart. Fail it (terminal, guard released) and re-raise so the
    request surfaces the error as a 500."""
    worker = threading.Thread(target=target, args=(job,), daemon=True)
    try:
        worker.start()
    except Exception:
        job.status = JobStatus.failed
        job.error = "Failed to start the generation worker thread"
        job.completed_at = now_iso()
        raise


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
    # ── Engine feature scope (§3-98 P5) ─────────────────────────────────────
    # FIRST, before every other check. What the ACTIVE base model's engine can
    # do is a property of the server, not of this request, so it is answered
    # without touching the upload stores or the lora registry: telling a user
    # "that reference video does not exist" for a request whose engine cannot
    # consume reference videos at all would send them to fix the wrong thing.
    # A no-op for LTX 2.3 (it declares no unsupported features), which is why
    # every pre-existing test is unaffected — deliberately, not by luck.
    engines.reject_unsupported(context.pipeline_manager.active_engine_family, request)

    # Validate conditioning images exist up front (minimal I2V).
    for ci in request.conditioning_images:
        context.upload_store.path_for(ci.image_id)  # raises IMAGE_NOT_FOUND

    # Phase B/C/S1 IC-LoRA: validate the reference video + adapter names up front,
    # the same way conditioning image_ids are checked (fail at job creation, not
    # deep in the worker).
    if request.reference_video_id is not None:
        context.video_upload_store.path_for(request.reference_video_id)  # 404 if missing
        # CONTROL adapters declare reference_downscale_factor=2 (union-control
        # family) or 1 (deblur). Under factor 2 the reference is consumed at half
        # output resolution on the 64-grid, so width/height not divisible by 128
        # crashes the worker's VAE encode. The check is applied to every reference
        # request (merely conservative for factor 1, which needs only 64).
        if request.width % 128 != 0 or request.height % 128 != 0:
            raise reference_resolution_invalid(request.width, request.height)
    # Resolve every requested adapter (404 unknown/missing) and inspect its kind:
    #   * a CONTROL adapter derives its conditioning from a reference video, so it
    #     requires reference_video_id (S1: replaces the old all-or-nothing rule,
    #     which is now kind-aware -- a STYLE/character adapter needs no reference);
    #   * conversely a reference video is ONLY consumable through a control
    #     adapter (its downscale factor comes from that adapter's metadata), so a
    #     reference + style-only request is rejected here. Until the factor guard
    #     was relaxed to accept factor 1, the engine happened to catch this misuse
    #     deep in the job; this endpoint check is now the only one;
    #   * a single reference video can only be turned into ONE control signal, so
    #     >1 distinct non-"none" preprocess kind is a conflict (Phase C).
    preprocess_kinds: set[str] = set()
    control_names: list[str] = []
    for spec in request.loras:
        context.lora_registry.resolve(spec.name, spec.strength)  # 404 if unknown/missing
        entry = context.lora_registry.info(spec.name)
        if entry.kind == "control":
            control_names.append(spec.name)
        if entry.preprocess != "none":
            preprocess_kinds.add(entry.preprocess)
    if control_names and request.reference_video_id is None:
        raise lora_requires_reference(control_names)
    if request.reference_video_id is not None and not control_names:
        raise reference_requires_control_lora([spec.name for spec in request.loras])
    if len(preprocess_kinds) > 1:
        raise lora_preprocess_conflict(sorted(preprocess_kinds))

    # ── Outpainting (§1-13) ──────────────────────────────────────────────────
    # The shape-only rules (pads, exclusivity, keep-region floor) live in the
    # pydantic validator; the three below need the registry or the file on disk,
    # so they belong here — the same split the control-adapter checks above use.
    if request.outpaint is not None:
        if preprocess_kinds:
            raise outpaint_preprocess_conflict(sorted(preprocess_kinds))
        from services import video_io

        op = request.outpaint
        keep = (
            request.width - op.pad_left - op.pad_right,
            request.height - op.pad_top - op.pad_bottom,
        )
        ref_path = context.video_upload_store.path_for(request.reference_video_id)
        actual = video_io.probe_resolution(ref_path)
        if actual != keep:
            raise outpaint_source_mismatch(keep, actual)
        # Frame count is authoritative for the blend pairing (see
        # outpaint_source_too_short). ffprobe is required for outpainting — it was
        # already required a line above, so this raises rather than degrading.
        available = video_io.frame_count(ref_path)
        if available < request.num_frames:
            raise outpaint_source_too_short(available, request.num_frames)

    # Loading guard: without this, a load in flight would still return 202 and
    # the job would only die later, inside the job thread, as GENERATION_FAILED.
    # Checked BEFORE the job-busy guard because loading is the stronger claim
    # (same order the frontend's status badge uses).
    context.pipeline_manager.reject_if_loading()

    # Single-job guard: atomically reserve, else 409 JOB_BUSY.
    job = context.job_store.create_if_idle(request)
    if job is None:
        raise job_busy()

    # Run the generation off the request path. The mock backend keeps using
    # BackgroundTasks (Starlette's threadpool) so the TestClient's synchronous
    # after-response semantics — which the whole test suite relies on — are
    # preserved. The real backend instead gets its OWN daemon thread: a real job
    # can run for many minutes, and parking it in the shared anyio worker-thread
    # pool would let long jobs starve the polling GET /jobs and self-issued POSTs
    # that the UI depends on. Dedicated threads sidestep that entirely.
    if (context.config.model.backend or "auto").strip().lower() == "mock":
        background_tasks.add_task(context.pipeline_manager.run_job, job)
    else:
        spawn_job_thread(context.pipeline_manager.run_job, job)

    return GenerateResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
    )
