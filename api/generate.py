"""POST /generate — start a generation job (spec §6.1 / §6.2)."""

from __future__ import annotations

import threading

from fastapi import APIRouter, BackgroundTasks, Depends

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import (
    APIError,
    alpha_gen_invalid,
    inpaint_lora_invalid,
    inpaint_mask_frame_mismatch,
    inpaint_mask_not_found,
    inpaint_mask_resolution_mismatch,
    inpaint_preprocess_conflict,
    inpaint_source_mismatch,
    inpaint_source_not_found,
    inpaint_video_no_audio,
    job_busy,
    lora_preprocess_conflict,
    lora_requires_reference,
    outpaint_preprocess_conflict,
    outpaint_source_mismatch,
    outpaint_source_not_found,
    outpaint_source_odd_size,
    outpaint_source_too_short,
    outpaint_video_no_audio,
    reference_audio_no_stream,
    reference_audio_silent,
    reference_not_found,
    reference_requires_control_lora,
    reference_resolution_invalid,
    source_audio_not_found,
)
from api.models import (
    INPAINT_MIN_SOURCE_SIDE,
    GenerateRequest,
    GenerateResponse,
    JobStatus,
    round_up_32,
    round_up_128,
)
from services import engines, video_io
from services.job_store import JobRecord, now_iso

router = APIRouter()

# The fps the app cuts MiniMax H3's inpaint source and mask to
# (services/pipeline_manager.py, ``cut_window_mp4(..., 24)``), so the mask's
# frame count is judged on that grid.
_H3_INPAINT_FPS = 24.0


def _preflight_h3_inpaint(
    context: AppContext, request: GenerateRequest, preprocess_kinds: set[str] | list[str]
) -> None:
    """The MiniMax H3 shape of ``inpaint`` (``inpaint.source_video_id``;
    台帳 §1-124), checked before a job exists, in the order a user can act on
    them: adapters, source, canvas, mask, window, audio. The pydantic validator
    has already enforced the shape-only rules (exactly one source route,
    exclusivity with outpaint / conditioning_images / crop_output).
    """
    ip = request.inpaint
    # (1) A preprocessing control adapter has nothing to read (shared rule).
    if preprocess_kinds:
        raise inpaint_preprocess_conflict(sorted(preprocess_kinds))

    # (2) The source upload, under its own field's 404.
    try:
        src_path = context.video_upload_store.path_for(ip.source_video_id)
    except APIError:
        raise inpaint_source_not_found(ip.source_video_id)

    # (3) The source must be measurable, even-sized (the app's window cut is
    # libx264 yuv420p) and no smaller than the shared floor.
    source_size = video_io.probe_resolution(src_path)
    if source_size is None:
        raise inpaint_source_mismatch((request.width, request.height), None, multiple=32)
    src_w, src_h = source_size
    canvas = (round_up_32(src_w), round_up_32(src_h))
    if src_w % 2 or src_h % 2:
        raise inpaint_source_mismatch(
            canvas,
            source_size,
            multiple=32,
            reason="the source's width and height must both be even",
        )
    if src_w < INPAINT_MIN_SOURCE_SIDE or src_h < INPAINT_MIN_SOURCE_SIDE:
        raise inpaint_source_mismatch(
            canvas,
            source_size,
            multiple=32,
            reason=f"each source side must be at least {INPAINT_MIN_SOURCE_SIDE}px",
        )

    # (4) The canvas is the source rounded up to 32 (the engine mirrors the
    # 0-31px right/bottom pad and crops it off again after decode).
    if canvas != (request.width, request.height):
        raise inpaint_source_mismatch(canvas, source_size, multiple=32)

    # (5) The mask: present, and exactly the source's own resolution.
    try:
        mask_path = context.video_upload_store.path_for(ip.mask_video_id)
    except APIError as exc:  # the store raises REFERENCE_VIDEO_NOT_FOUND
        raise inpaint_mask_not_found(ip.mask_video_id) from exc
    mask_size = video_io.probe_resolution(mask_path)
    if mask_size != source_size:
        raise inpaint_mask_resolution_mismatch(source_size, mask_size)

    # (6) The mask's frame count at 24 fps (the app cuts its first num_frames
    # frames at 24 fps): the same resampling expression outpainting's source
    # route uses. At least num_frames, not exactly.
    mask_frames = video_io.frame_count(mask_path)
    mask_fps = video_io.probe_fps(mask_path)
    if mask_fps and abs(mask_fps - _H3_INPAINT_FPS) > 1e-3:
        mask_frames = int(round(mask_frames * _H3_INPAINT_FPS / mask_fps))
    if mask_frames < request.num_frames:
        raise inpaint_mask_frame_mismatch(mask_frames, request.num_frames, rule="at_least")

    # (7) The window has to fit the material (the pipeline manager owns the rule).
    context.pipeline_manager.preflight_inpaint_window(
        ip, ip.source_video_id, request.num_frames, request.frame_rate
    )

    # (8) Audio: an omitted Strength becomes the family's default 0.5, so None
    # counts as "> 0" here. Same refusal as outpainting's source route.
    uses_audio = ip.audio_strength is None or ip.audio_strength > 0
    if uses_audio and not video_io.has_audio_stream(src_path):
        shown = "default" if ip.audio_strength is None else f"{ip.audio_strength:g}"
        raise inpaint_video_no_audio(
            ip.source_video_id,
            detail=(
                "the inpaint uses the source video's audio "
                f"(audio_strength={shown} > 0; send 0 to generate the audio "
                f"freely), but upload {ip.source_video_id} has no audio stream"
            ),
        )


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


def check_references_exist(context: AppContext, refs, *, clip_index: int | None = None) -> None:
    """Reference (Ref2VA, MiniMax H3): every ``refs[j].id`` must name an upload
    of its own kind's store; the first miss raises REFERENCE_NOT_FOUND (404)
    naming the entry. A check only: no path is resolved for later use (the
    job resolves its own from the stored request). Shared with
    /generate/chain, which passes ``clip_index`` so the message names
    ``clips[i].references[j]``; the single /generate leaves it out."""
    reference_stores = {
        "picture": context.upload_store,
        "video": context.video_upload_store,
        "audio": context.audio_upload_store,
    }
    for index, ref in enumerate(refs):
        try:
            reference_stores[ref.kind].path_for(ref.id)
        except APIError:
            raise reference_not_found(index, ref.id, ref.kind, ref.slot, clip_index=clip_index)
        # An audio upload may be a video container (POST /upload/audio): its
        # audio stream is all the engine uses, so a file without one is
        # refused here (422) rather than late inside the engine; so is one
        # whose audio track is digitally silent (every sample zero).
        if ref.kind == "audio":
            path = reference_stores["audio"].path_for(ref.id)
            if not video_io.has_audio_stream(path):
                raise reference_audio_no_stream(index, ref.id, clip_index=clip_index)
            if video_io.audio_is_digitally_silent(path):
                raise reference_audio_silent(index, ref.id, clip_index=clip_index)


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
    # ── Engine feature scope ────────────────────────────────────────────────
    # FIRST, before every other check. What the ACTIVE base model's engine can
    # do is a property of the server, not of this request, so it is answered
    # without touching the upload stores or the lora registry: telling a user
    # "that reference video does not exist" for a request whose engine cannot
    # consume reference videos at all would send them to fix the wrong thing.
    # Each engine declares its refusals in its adapter's ``REJECT_TABLE``
    # (services/engines/<family>/adapter.py), so the guard answers on both
    # engines. A DEFAULT request passes on both: every predicate in those
    # tables tests "differs from the default" — deliberately, not by luck.
    engines.reject_unsupported(context.pipeline_manager.active_engine_family, request)

    # ── Engine numeric rules ────────────────────────────────────────────────
    # Right AFTER the feature scope, so a refused feature is named before a
    # number is (LTX 2.5 with two_stage_hq AND an 8n+1 violation answers
    # FEATURE_UNSUPPORTED). The frame grid, the spatial multiple, the fps and
    # the step / shift ranges are the active family's (its adapter's
    # ``REQUEST_RULES``); the 422 is shaped like pydantic's. This is also the
    # ONE place an omitted ``num_inference_steps`` / shift becomes the family's
    # default, so everything downstream (the job log, the runner, metadata.json)
    # sees a number. The keyframe positions are also snapped onto the family's
    # grid here (and an omitted keyframe strength gets the family's default).
    engines.validate_for_family(context.pipeline_manager.active_engine_family, request)

    # Alpha Gen's block is INTERNAL: only POST /generate/alpha builds it (from a
    # source it has measured itself), so a /generate that carries one is refused
    # rather than trusted. Placed right after the engine scope so LTX 2.3 still
    # answers FEATURE_UNSUPPORTED first.
    if request.alpha_gen is not None:
        raise alpha_gen_invalid("alpha_gen is set by POST /generate/alpha only")

    # Validate conditioning images exist up front.
    for ci in request.conditioning_images:
        context.upload_store.path_for(ci.image_id)  # raises IMAGE_NOT_FOUND

    # Single-generate A2V (MiniMax H3): resolve the source audio (404) and
    # preflight it (422 for too-short / no audio stream) BEFORE reserving a
    # job — the same order and the same errors as /generate/chain's
    # source_audio. An LTX engine never gets here: its REJECT_TABLE refused the
    # field above (422 FEATURE_UNSUPPORTED), ahead of any upload lookup.
    if request.source_audio is not None:
        try:
            context.audio_upload_store.path_for(request.source_audio.audio_id)
        except APIError:
            raise source_audio_not_found(request.source_audio.audio_id)
        context.pipeline_manager.preflight_single_source_audio(
            request.source_audio, request.num_frames, request.frame_rate
        )

    # Reference (Ref2VA, MiniMax H3): every id must name an upload of its own
    # kind's store (404 otherwise), before a job is reserved. The stores' own
    # 404s name other fields (image_id, reference_video_id, source_audio), so
    # the miss is answered as REFERENCE_NOT_FOUND naming the entry. An LTX
    # engine never gets here (its REJECT_TABLE refused the field above), and
    # the counts and slots were checked by validate_for_family.
    check_references_exist(context, request.references)

    # IC-LoRA: validate the reference video + adapter names up front, the same
    # way conditioning image_ids are checked (fail at job creation, not deep in
    # the worker).
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
    #     requires reference_video_id (a STYLE/character adapter needs no
    #     reference);
    #   * conversely a reference video is ONLY consumable through a control
    #     adapter (its downscale factor comes from that adapter's metadata), so a
    #     reference + style-only request is rejected here, at job creation rather
    #     than deep in the job;
    #   * a single reference video can only be turned into ONE control signal, so
    #     >1 distinct non-"none" preprocess kind is a conflict.
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

    # ── Outpainting ───────────────────────────────────────────────────────
    # The shape-only rules (pads, exclusivity, keep-region floor) live in the
    # pydantic validator; the three below need the registry or the file on disk,
    # so they belong here — the same split the control-adapter checks above use.
    #
    # The source arrives by exactly one route (the schema's rule): LTX's
    # ``reference_video_id`` or MiniMax H3's ``outpaint.source_video_id``
    # (台帳 §1-125). The branch is on the REQUEST'S SHAPE, never on the family
    # (each family's rules refuse the other's route). The source_video_id route
    # adds three checks the app's cut needs: its own 404 name, an even size
    # (the cut is libx264 yuv420p), and the frame count at the REQUEST's fps
    # (the cut resamples to it). Then the audio preflight: validate_for_family
    # has already written H3's default Strength back, so ``audio_strength`` is
    # a number here on that route and None on LTX's (which refuses it).
    if request.outpaint is not None:
        if preprocess_kinds:
            raise outpaint_preprocess_conflict(sorted(preprocess_kinds))
        from services import video_io

        op = request.outpaint
        keep = (
            request.width - op.pad_left - op.pad_right,
            request.height - op.pad_top - op.pad_bottom,
        )
        if op.source_video_id is not None:
            try:
                ref_path = context.video_upload_store.path_for(op.source_video_id)
            except APIError:
                raise outpaint_source_not_found(op.source_video_id)
        else:
            ref_path = context.video_upload_store.path_for(request.reference_video_id)
        actual = video_io.probe_resolution(ref_path)
        if op.source_video_id is not None and actual is not None and (actual[0] % 2 or actual[1] % 2):
            raise outpaint_source_odd_size(actual)
        if actual != keep:
            raise outpaint_source_mismatch(keep, actual)
        # Frame count is authoritative for the blend pairing (see
        # outpaint_source_too_short). ffprobe is required for outpainting — it was
        # already required a line above, so this raises rather than degrading.
        available = video_io.frame_count(ref_path)
        if op.source_video_id is not None:
            # The count at the request's fps (the cut resamples the source to
            # it): the same expression as PipelineManager.preflight_retake_window.
            src_fps = video_io.probe_fps(ref_path)
            if src_fps and abs(src_fps - float(request.frame_rate)) > 1e-3:
                available = int(round(available * float(request.frame_rate) / src_fps))
        if available < request.num_frames:
            raise outpaint_source_too_short(available, request.num_frames)
        if (op.audio_strength or 0) > 0 and not video_io.has_audio_stream(ref_path):
            raise outpaint_video_no_audio(
                op.source_video_id,
                detail=(
                    "the outpaint uses the source video's audio "
                    f"(audio_strength={op.audio_strength:g} > 0; send 0 to generate "
                    f"the audio freely), but upload {op.source_video_id} has no audio stream"
                ),
            )

    # ── Inpainting ────────────────────────────────────────────────────────
    # The same split the outpaint block above uses: the shape-only rules
    # (exclusivity, the 128 grid, the window's 8n+1) are in the pydantic
    # validator; everything that needs the registry or a file on disk is here.
    # Five checks, in the order a user can act on them — what is wrong with the
    # adapters, then with the source, then with the mask, then with the window.
    #
    # The branch is on the REQUEST'S SHAPE, never on the family (台帳 §1-124),
    # exactly as outpainting's is: ``inpaint.source_video_id`` is MiniMax H3's
    # shape (32 grid, no control LoRA, mask counted at 24 fps, audio Strength),
    # ``reference_video_id`` is LTX's (the block after this one, unchanged).
    if request.inpaint is not None and request.inpaint.source_video_id is not None:
        _preflight_h3_inpaint(context, request, preprocess_kinds)

    if request.inpaint is not None and request.inpaint.source_video_id is None:
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

    # Run the generation off the request path. The mock backend uses
    # BackgroundTasks (Starlette's threadpool) so the TestClient's synchronous
    # after-response semantics — which the test suite relies on — hold. The
    # real backend instead gets its OWN daemon thread: a real job can run for
    # many minutes, and parking it in the shared anyio worker-thread pool would
    # let long jobs starve the polling GET /jobs and self-issued POSTs that the
    # UI depends on. Dedicated threads sidestep that entirely.
    if (context.config.model.backend or "auto").strip().lower() == "mock":
        background_tasks.add_task(context.pipeline_manager.run_job, job)
    else:
        spawn_job_thread(context.pipeline_manager.run_job, job)

    return GenerateResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
    )
