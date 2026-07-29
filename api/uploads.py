"""POST /upload/image — minimal-I2V image upload (spec 7.1).

POST /upload/video — reference-video upload for the Phase B IC-LoRA
(Pixel-Spatial-Upscaler). Same storage/naming/error pattern as the image
endpoint; the returned ``video_id`` is passed as ``reference_video_id`` to
POST /generate.
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
    # Optional ribbon trim window (V2V): keep only [start, start+duration) of the
    # uploaded video. Deliberately UNCONSTRAINED (no ge=/le=): out-of-range or
    # non-finite values must fall through to a normal untrimmed upload rather
    # than turn a previously-successful request into a 422. The store decides.
    trim_start_sec: float | None = Query(None),
    trim_duration_sec: float | None = Query(None),
    context: AppContext = Depends(get_context),
) -> UploadVideoResponse:
    if not file.filename:
        raise upload_invalid_type(detail="missing filename")
    data = await file.read()
    # save() may shell out to ffmpeg (trim path), which would otherwise block the
    # event loop -- and with it every other API call -- for the whole cut.
    stored = await run_in_threadpool(
        context.video_upload_store.save,
        data=data,
        filename=file.filename,
        trim_start_sec=trim_start_sec,
        trim_duration_sec=trim_duration_sec,
    )
    return UploadVideoResponse(
        video_id=stored.video_id,
        original_filename=stored.original_filename,
        stored_path=context.video_upload_store.stored_relpath(stored),
        content_type=stored.content_type,
        size_bytes=stored.size_bytes,
        trimmed=stored.trimmed,
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
