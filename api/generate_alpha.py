"""POST /generate/alpha — start an Alpha Gen job (RGB video -> grey alpha matte).

Additive to the frozen single-``/generate`` contract, in the same way
``POST /generate/chain`` is: a thin public body (:class:`AlphaGenRequest`) from
which the SERVER builds an ordinary internal :class:`GenerateRequest` — fixed
prompt (one space), the ``alpha-gen`` IC-LoRA at strength 1.0, the uploaded
video as the reference, the canvas as width/height, and the internal
:class:`AlphaGenSpec` — and runs it through the same single-job path
(``create_if_idle`` -> ``PipelineManager.run_job``). LTX 2.5 only.

Design canon: ``Docs/ALPHAGEN_DESIGN.md``. The check order follows
api/generate.py's rule "engine scope first", then the upload and the adapter,
then the source, the window and the geometry:

  (a) the active engine lists ``alpha_gen`` in ``unsupported_features``
      -> 422 FEATURE_UNSUPPORTED (this endpoint's own wording), BEFORE the
      source is measured;
  (b) the body's shape (8n+1, <= 145, integer fps) — pydantic;
  (c) the reference upload (404 REFERENCE_VIDEO_NOT_FOUND) and the ``alpha-gen``
      adapter (404 LORA_NOT_FOUND; registered as anything but a control
      adapter without preprocess -> 422 ALPHA_GEN_INVALID);
  (d) the source's size: measurable, even sides, >= INPAINT_MIN_SOURCE_SIDE;
  (e) the window fits the material;
  (f) the working size / canvas fit the GenerateRequest bounds;
  (g) the internal GenerateRequest is built (a rejection -> 422);
  (h) the loading / job-busy guards (409);
  (i) on the ``/generate`` side, a public body carrying ``alpha_gen`` is
      refused (api/generate.py), so this endpoint is its only producer.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends

import chain_math
from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import alpha_gen_invalid, alpha_gen_unsupported, job_busy
from api.generate import spawn_job_thread
from api.models import (
    INPAINT_MIN_SOURCE_SIDE,
    AlphaGenRequest,
    AlphaGenSpec,
    GenerateRequest,
    GenerateResponse,
    LoraSpec,
)
from api.models_registry import transformer_weight_class
from services import engines, video_io

router = APIRouter()

#: The registered IC-LoRA name Alpha Gen runs with (fixed by the server; the
#: user never picks it). Registered like any other adapter in config.yaml's
#: ``model.ic_loras``; the operation panel and Gradio hide it from their lists.
ALPHA_GEN_LORA_NAME = "alpha-gen"
ALPHA_GEN_LORA_STRENGTH = 1.0
#: The official AlphaGen prompt is empty; GenerateRequest.prompt needs at least
#: one character, so the server sends a single space (gate 0, VERIFICATION_LOG
#: §151, confirmed the generation runs with it).
ALPHA_GEN_PROMPT = " "

# GenerateRequest's own bounds (api/models.py ``width``/``height`` Fields). A
# working size or canvas outside them would fail the internal request's
# validation and surface as a 500, so they are checked here first.
_WIDTH_RANGE = (256, 4096)
_HEIGHT_RANGE = (128, 4096)

_MISSING = object()


def _requested_value(request: AlphaGenRequest, key: str):
    """The value a comfort row's ``requires`` key compares against: the
    pass-through field on the Alpha Gen body when it has one, otherwise the
    internal GenerateRequest's default for that field (what the job will run
    with), otherwise "absent"."""
    if key in AlphaGenRequest.model_fields:
        return getattr(request, key)
    field = GenerateRequest.model_fields.get(key)
    if field is not None and not field.is_required():
        return field.default
    return _MISSING


def alpha_gen_budget(context: AppContext, family: str, request: AlphaGenRequest) -> int | None:
    """The comfort budget (tokens) the Alpha Gen canvas is fitted under.

    The row is picked by the existing rule (``config.ComfortRow``: rows are
    tried top-down and the FIRST row whose ``requires`` keys ALL match wins),
    with ``weight_class`` taken from the selected transformer — the same class
    ``GET /models`` reports. From that row: ``single_budget`` in light mode
    (the ordinary two-stage path), ``alpha_gen_budget`` otherwise.

    EXCEPTION (written down in Docs/ALPHAGEN_DESIGN.md): when the weight class
    cannot be determined — or no row matches — the SMALLEST value across the
    rows is used, so an unclassifiable file never gets a larger canvas than
    any known class would. ``None`` (no shrink) when the family has no comfort
    rows or none of them carries a value.
    """
    profile = context.config.limits.comfort_budgets.get(family)
    if profile is None or not profile.rows:
        return None
    key = "single_budget" if request.light_mode else "alpha_gen_budget"

    pm = context.pipeline_manager
    weight_class = transformer_weight_class(
        context.model_registry, pm.active_base_model, pm.active_models
    )
    if weight_class is not None:
        for row in profile.rows:
            matched = True
            for req_key, req_value in row.requires.items():
                actual = (
                    weight_class
                    if req_key == "weight_class"
                    else _requested_value(request, req_key)
                )
                if actual is _MISSING or actual != req_value:
                    matched = False
                    break
            if matched:
                return getattr(row, key)

    values = [getattr(row, key) for row in profile.rows if getattr(row, key) is not None]
    return min(values) if values else None


@router.post(
    "/generate/alpha",
    response_model=GenerateResponse,
    status_code=202,
    dependencies=[Depends(require_auth)],
)
def generate_alpha(
    request: AlphaGenRequest,
    background_tasks: BackgroundTasks,
    context: AppContext = Depends(get_context),
) -> GenerateResponse:
    family = context.pipeline_manager.active_engine_family

    # (a) Engine scope FIRST — a fact about the server, answered before the
    # upload is looked up or measured (same rule as api/generate.py). Read
    # from the adapter's own table, so no second list lives here.
    if "alpha_gen" in engines.unsupported_features(family):
        raise alpha_gen_unsupported()

    # (c) The reference upload (404 REFERENCE_VIDEO_NOT_FOUND) and the fixed
    # adapter (404 LORA_NOT_FOUND when unregistered or its file is missing).
    source_path = context.video_upload_store.path_for(request.reference_video_id)
    context.lora_registry.resolve(ALPHA_GEN_LORA_NAME, ALPHA_GEN_LORA_STRENGTH)
    adapter = context.lora_registry.info(ALPHA_GEN_LORA_NAME)
    if adapter.kind != "control" or adapter.preprocess != "none":
        raise alpha_gen_invalid(
            "alpha-gen must be registered as a control adapter without preprocess "
            f"(kind={adapter.kind}, preprocess={adapter.preprocess})"
        )

    # (d) The source's own size. Even sides because the matte is written back
    # at exactly this size in yuv420p; the floor is inpainting's
    # (INPAINT_MIN_SOURCE_SIDE), reused rather than invented.
    source_size = video_io.probe_resolution(source_path)
    if source_size is None:
        raise alpha_gen_invalid(
            f"could not measure the resolution of upload {request.reference_video_id}"
        )
    src_w, src_h = source_size
    if src_w % 2 or src_h % 2:
        raise alpha_gen_invalid(
            f"the source is {src_w}x{src_h}; both sides must be even"
        )
    if src_w < INPAINT_MIN_SOURCE_SIDE or src_h < INPAINT_MIN_SOURCE_SIDE:
        raise alpha_gen_invalid(
            f"the source is {src_w}x{src_h}; each side must be at least "
            f"{INPAINT_MIN_SOURCE_SIDE}px"
        )

    # (e) The window has to fit the material (the rule inpainting uses, shared
    # through the pipeline manager; this endpoint wraps it in its own code).
    detail = context.pipeline_manager._window_fit_detail(
        request.reference_video_id,
        request.window_start_sec,
        request.num_frames,
        request.frame_rate,
    )
    if detail is not None:
        raise alpha_gen_invalid(detail)

    # (f) Working size and canvas under the comfort budget.
    budget = alpha_gen_budget(context, family, request)
    try:
        work_w, work_h, canvas_w, canvas_h = chain_math.alpha_gen_geometry(
            src_w, src_h, request.num_frames, budget
        )
    except ValueError as exc:
        raise alpha_gen_invalid(str(exc)) from exc
    for label, (w, h) in (("working size", (work_w, work_h)), ("canvas", (canvas_w, canvas_h))):
        if not (
            _WIDTH_RANGE[0] <= w <= _WIDTH_RANGE[1]
            and _HEIGHT_RANGE[0] <= h <= _HEIGHT_RANGE[1]
        ):
            raise alpha_gen_invalid(
                f"the {label} {w}x{h} (source {src_w}x{src_h}, budget {budget}) is "
                f"outside {_WIDTH_RANGE[0]}-{_WIDTH_RANGE[1]} x "
                f"{_HEIGHT_RANGE[0]}-{_HEIGHT_RANGE[1]}"
            )

    # (g) The internal request. pipeline / guidance_scale stay at their
    # defaults; the pass-through fields are copied verbatim so a matte job
    # keeps the user's residency / acceleration settings (an absent
    # keep_resident key is the RELEASE request on LTX 2.5).
    try:
        internal = GenerateRequest(
            prompt=ALPHA_GEN_PROMPT,
            loras=[LoraSpec(name=ALPHA_GEN_LORA_NAME, strength=ALPHA_GEN_LORA_STRENGTH)],
            reference_video_id=request.reference_video_id,
            width=canvas_w,
            height=canvas_h,
            num_frames=request.num_frames,
            frame_rate=float(request.frame_rate),
            seed=request.seed,
            attention_backend=request.attention_backend,
            block_swap_prefetch=request.block_swap_prefetch,
            keep_resident=request.keep_resident,
            keep_resident_embeddings=request.keep_resident_embeddings,
            fused_gguf_dequant_kernel=request.fused_gguf_dequant_kernel,
            alpha_gen=AlphaGenSpec(
                window_start_sec=request.window_start_sec,
                one_stage=not request.light_mode,
                working_width=work_w,
                working_height=work_h,
                budget_tokens=budget,
            ),
        )
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise alpha_gen_invalid(f"internal request rejected: {exc}") from exc

    # (h) Loading guard, then the single-job guard (409), exactly as
    # api/generate.py orders them.
    context.pipeline_manager.reject_if_loading()
    job = context.job_store.create_if_idle(internal)
    if job is None:
        raise job_busy()

    # Mock backend -> BackgroundTasks (TestClient sync semantics); real backend
    # -> a dedicated daemon thread (see api/generate.py).
    if (context.config.model.backend or "auto").strip().lower() == "mock":
        background_tasks.add_task(context.pipeline_manager.run_job, job)
    else:
        spawn_job_thread(context.pipeline_manager.run_job, job)

    return GenerateResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
    )
