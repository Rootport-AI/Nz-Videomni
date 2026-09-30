"""POST /upload/image, POST /upload/video, POST /upload/audio (spec §6.1).

Each endpoint stores one uploaded file and returns its id; later requests to
POST /generate and POST /generate/chain refer to the file by that id (a video
id goes into fields such as ``reference_video_id``). The three share the same
storage/naming/error pattern.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import upload_invalid_type
from api.models import UploadAudioResponse, UploadImageResponse, UploadVideoResponse

router = APIRouter()


@router.post("/upload/image", response_model=UploadImageResponse, dependencies=[Depends(require_auth)])
async def upload_image(
    file: UploadFile = File(...),
    context: AppContext = Depends(get_context),
) -> UploadImageResponse:
    if not file.filename:
        raise upload_invalid_type(detail="missing filename")
    data = await file.read()
    stored = context.upload_store.save(data=data, filename=file.filename)
    return UploadImageResponse(
        image_id=stored.image_id,
        original_filename=stored.original_filename,
        stored_path=context.upload_store.stored_relpath(stored.image_id),
        width=stored.width,
        height=stored.height,
        content_type=stored.content_type,
    )


@router.post("/upload/video", response_model=UploadVideoResponse, dependencies=[Depends(require_auth)])
async def upload_video(
    file: UploadFile = File(...),
    # Optional ribbon trim window: keep only [start, start+duration) of the
    # uploaded video. Deliberately UNCONSTRAINED (no ge=/le=): out-of-range or
    # non-finite values must fall through to a normal untrimmed upload rather
    # than turn an upload that succeeds without them into a 422. The store
    # decides.
    trim_start_sec: float | None = Query(None),
    trim_duration_sec: float | None = Query(None),
    # Keep only the first max_frames frames of the uploaded video. The caller
    # picks the cap (the chain reference uses the chain ceiling,
    # api/models.py's MAX_CHAIN_TOTAL_PIXEL_FRAMES), and sending it also asks
    # for frame_count/fps to be measured. Same unconstrained (no ge=/le=)
    # convention as the trim window above: an out-of-range value falls through
    # to a normal untrimmed upload rather than a 422, and an explicit trim
    # window (when both are somehow sent) always wins -- see
    # VideoUploadStore.save's docstring.
    max_frames: int | None = Query(None),
    context: AppContext = Depends(get_context),
) -> UploadVideoResponse:
    if not file.filename:
        raise upload_invalid_type(detail="missing filename")
    data = await file.read()
    # save() may shell out to ffmpeg (trim / max_frames path), which would
    # otherwise block the event loop -- and with it every other API call --
    # for the whole cut.
    stored = await run_in_threadpool(
        context.video_upload_store.save,
        data=data,
        filename=file.filename,
        trim_start_sec=trim_start_sec,
        trim_duration_sec=trim_duration_sec,
        max_frames=max_frames,
    )
    return UploadVideoResponse(
        video_id=stored.video_id,
        original_filename=stored.original_filename,
        stored_path=context.video_upload_store.stored_relpath(stored),
        content_type=stored.content_type,
        size_bytes=stored.size_bytes,
        trimmed=stored.trimmed,
        # Measured only when max_frames was sent, else None on both (see
        # VideoUploadStore.save). Pure transcription -- the store owns the rules.
        frame_count=stored.frame_count,
        fps=stored.fps,
    )


@router.post("/upload/audio", response_model=UploadAudioResponse, dependencies=[Depends(require_auth)])
async def upload_audio(
    file: UploadFile = File(...),
    context: AppContext = Depends(get_context),
) -> UploadAudioResponse:
    if not file.filename:
        raise upload_invalid_type(detail="missing filename")
    data = await file.read()
    stored = context.audio_upload_store.save(data=data, filename=file.filename)
    return UploadAudioResponse(
        audio_id=stored.audio_id,
        original_filename=stored.original_filename,
        stored_path=context.audio_upload_store.stored_relpath(stored),
        content_type=stored.content_type,
        size_bytes=stored.size_bytes,
    )
