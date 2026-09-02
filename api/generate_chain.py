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
    end_source_not_found,
    job_busy,
    lora_depth_chain_unsupported,
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
from services import engines

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
    # ── Engine feature scope (§3-98 P5, widened §3-102) ─────────────────────
    # Chained, Retake, End source, V2V continuation and A2V are all shapes of
    # THIS request, and which of them an engine can serve is a FIELD-BY-FIELD
    # answer, not one blanket yes/no — LTX 2.5 runs a plain Chained job but none
    # of the four modes layered on it. Hence the body is passed. LTX 2.3 serves
    # all of them — but it is no longer a no-op there either: §3-114 gave 2.3
    # one refusal of its own (``keep_resident_embeddings``, which names a
    # component only 2.5 has). A DEFAULT chain still passes on both engines,
    # because every predicate in both tables tests "differs from the default".
    #
    # STAYS AHEAD OF THE UPLOAD LOOKUPS BELOW. "This engine cannot do that" is a
    # fact about the server; "that video does not exist" is a fact about the
    # request. Answering the second first would send the user to fix something
    # that is not the problem.
    engines.reject_chain(context.pipeline_manager.active_engine_family, request)

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

    # End source: resolve the upload (404) and, for a VIDEO, preflight its length
    # (422 when it cannot supply context_frames + 1 frames) BEFORE reserving a
    # job — same up-front-failure discipline as the blocks above. The id resolves
    # against a DIFFERENT store depending on which kind was sent (the schema
    # guarantees exactly one of the two is set): videos live in the reference/
    # continuation video store, stills in the conditioning-image store. An image
    # needs no length check at all — it is looped to whatever length the band
    # asks for. The tail band's GEOMETRY was already settled by the schema +
    # chain_math; what is checked here is only whether the uploaded material
    # actually contains it.
    if request.end_source is not None:
        if request.end_source.video_id is not None:
            try:
                context.video_upload_store.path_for(request.end_source.video_id)
            except APIError:
                raise end_source_not_found(request.end_source.video_id)
            context.pipeline_manager.preflight_end_source(
                request.end_source, request.frame_rate
            )
        else:
            try:
                context.upload_store.path_for(request.end_source.image_id)
            except APIError:
                raise end_source_not_found(request.end_source.image_id)

    # Reference-video CONTROL IC-LoRA (Phase C chain support, ADDITIVE, 1..24
    # clips — owner decision 2026-08-11): validate the reference video up front,
    # mirroring api/generate.py 57-63.
    if request.reference_video_id is not None:
        context.video_upload_store.path_for(request.reference_video_id)  # 404 if missing
        # CONTROL adapters declare reference_downscale_factor=2 (union-control
        # family) or 1 (deblur). Under factor 2 the reference is consumed at half
        # output resolution on the 64-grid, so width/height not divisible by 128
        # crashes the worker's VAE encode. The check is applied to every reference
        # request (merely conservative for factor 1, which needs only 64) --
        # mirrors api/generate.py.
        if request.width % 128 != 0 or request.height % 128 != 0:
            raise reference_resolution_invalid(request.width, request.height)

    # Style/character + reference-video CONTROL IC-LoRA (ADDITIVE): resolve every
    # requested adapter (404 unknown/missing) up front — same discipline as
    # api/generate.py — and inspect its kind:
    #   * a CONTROL adapter (union-control / pixel-spatial-upscaler) derives its
    #     conditioning from a reference video, on any clip count (1..24 — owner
    #     decision 2026-08-11 lifted the old 1-clip-only ALPHA scope): it needs
    #     the reference_video_id, exactly like the single-generate check
    #     (LORA_REQUIRES_REFERENCE);
    #   * a depth-preprocess CONTROL adapter is the one exception: it remains
    #     rejected outright on a >1-clip chain (LORA_DEPTH_CHAIN_UNSUPPORTED) —
    #     the depth preprocessor (Video-Depth-Anything) is a whole-clip design
    #     that cannot process a chain-length reference (owner decision
    #     2026-08-11; the engine-side chunking to lift this is a later item);
    #   * conversely a reference video is ONLY consumable through a control
    #     adapter (its downscale factor comes from that adapter's metadata), so a
    #     reference + style-only chain is rejected here
    #     (REFERENCE_REQUIRES_CONTROL_LORA, mirrors api/generate.py);
    #   * a single reference video can only be turned into ONE control signal, so
    #     >1 distinct non-"none" preprocess kind is a conflict (Phase C, mirrors
    #     api/generate.py).
    preprocess_kinds: set[str] = set()
    control_names: list[str] = []
    depth_names: list[str] = []
    for spec in request.loras:
        context.lora_registry.resolve(spec.name, spec.strength)  # 404 if unknown/missing
        entry = context.lora_registry.info(spec.name)
        if entry.kind == "control":
            control_names.append(spec.name)
        if entry.preprocess != "none":
            preprocess_kinds.add(entry.preprocess)
        if entry.preprocess == "depth":
            depth_names.append(spec.name)
    if depth_names and len(request.clips) > 1:
        raise lora_depth_chain_unsupported(depth_names)
    if control_names:
        if request.reference_video_id is None:
            raise lora_requires_reference(control_names)
    if request.reference_video_id is not None and not control_names:
        raise reference_requires_control_lora([spec.name for spec in request.loras])
    if len(preprocess_kinds) > 1:
        raise lora_preprocess_conflict(sorted(preprocess_kinds))

    # Loading guard: without this, a load in flight would still return 202 and
    # the job would only die later, inside the job thread, as GENERATION_FAILED.
    # Checked BEFORE the job-busy guard because loading is the stronger claim
    # (same order the frontend's status badge uses).
    context.pipeline_manager.reject_if_loading()

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
