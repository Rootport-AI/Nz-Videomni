"""POST /generate — start a generation job (spec 7.2 / 8.1)."""

from __future__ import annotations

import threading

from fastapi import APIRouter, BackgroundTasks, Depends

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import (
    APIError,
    inpaint_lora_invalid,
    inpaint_mask_frame_mismatch,
    inpaint_mask_not_found,
    inpaint_mask_resolution_mismatch,
    inpaint_preprocess_conflict,
    inpaint_source_mismatch,
    job_busy,
    lora_preprocess_conflict,
    lora_requires_reference,
    outpaint_preprocess_conflict,
    outpaint_source_mismatch,
    outpaint_source_too_short,
    reference_requires_control_lora,
    reference_resolution_invalid,
)
from api.models import (
    INPAINT_MIN_SOURCE_SIDE,
    GenerateRequest,
    GenerateResponse,
    JobStatus,
    round_up_128,
)
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
    # NOT a no-op for either engine any more. LTX 2.3 declared nothing until
    # §3-114 gave it one refusal of its own (``keep_resident_embeddings``, which
    # names a component only 2.5 has), so the guard now answers on both sides.
    # What is still true is that a DEFAULT request passes on both: every
    # predicate in both tables tests "differs from the default", which is why
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

    # ── Outpainting (Docs/PENDING_TASKS_CLOSED.md §3-70, filed as §1-13 at the time) ──
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

    # ── Inpainting (台帳 §3-55) ─────────────────────────────────────────────
    # The same split the outpaint block above uses: the shape-only rules
    # (exclusivity, the 128 grid, the window's 8n+1) are in the pydantic
    # validator; everything that needs the registry or a file on disk is here.
    # Five checks, in the order a user can act on them — what is wrong with the
    # adapters, then with the source, then with the mask, then with the window.
    if request.inpaint is not None:
        ip = request.inpaint
        from services import video_io

        # (a) A control adapter that PREPROCESSES its reference would hand the
        # model an edge map of sentinel green. Same ruling as outpainting's.
        if preprocess_kinds:
            raise inpaint_preprocess_conflict(sorted(preprocess_kinds))
        # (b) ...and there must be exactly one control adapter to consume the
        # canvas through. The name is deliberately NOT fixed here (outpainting
        # does not fix it either): which adapter does in/out-painting is a
        # registry fact, not an API constant.
        if len(control_names) != 1:
            raise inpaint_lora_invalid([spec.name for spec in request.loras])

        # (c) The canvas must be the SOURCE rounded up. The source file is the
        # single source of truth for its own size (InpaintSpec's docstring), so
        # this is the check that makes the geometry safe end to end.
        ref_path = context.video_upload_store.path_for(request.reference_video_id)
        source_size = video_io.probe_resolution(ref_path)
        if source_size is None:
            # ``expected`` is the REQUESTED canvas here, not a canvas derived
            # from the source: there is no source size to derive one from, and
            # the message must not imply otherwise (see the factory's docstring).
            raise inpaint_source_mismatch((request.width, request.height), None)
        src_w, src_h = source_size
        if src_w < INPAINT_MIN_SOURCE_SIDE or src_h < INPAINT_MIN_SOURCE_SIDE:
            raise inpaint_source_mismatch(
                (request.width, request.height),
                source_size,
                reason=(
                    f"each source side must be at least {INPAINT_MIN_SOURCE_SIDE}px"
                ),
            )
        canvas = (round_up_128(src_w), round_up_128(src_h))
        if canvas != (request.width, request.height):
            raise inpaint_source_mismatch(canvas, source_size)

        # (d) The mask: it must exist, be exactly the source's resolution (it is
        # never resized), and carry exactly num_frames frames. The frame count is
        # checked HERE rather than left to ffmpeg because this build's
        # ``maskedmerge`` has no ``shortest`` option — a short mask would have its
        # last frame repeated and the tail of the window would go silently
        # unrepainted (services/video_io.fill_mask_green_mp4).
        try:
            mask_path = context.video_upload_store.path_for(ip.mask_video_id)
        except APIError as exc:  # the store raises REFERENCE_VIDEO_NOT_FOUND
            raise inpaint_mask_not_found(ip.mask_video_id) from exc
        mask_size = video_io.probe_resolution(mask_path)
        if mask_size != source_size:
            raise inpaint_mask_resolution_mismatch(source_size, mask_size)
        mask_frames = video_io.frame_count(mask_path)
        if mask_frames != request.num_frames:
            raise inpaint_mask_frame_mismatch(mask_frames, request.num_frames)

        # (e) The window has to fit the material. Delegated to the pipeline
        # manager exactly as retake's is — it owns the upload store and the
        # fps/frame-count arithmetic, and putting the rule anywhere else would
        # give it two homes.
        context.pipeline_manager.preflight_inpaint_window(
            ip, request.reference_video_id, request.num_frames, request.frame_rate
        )

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
