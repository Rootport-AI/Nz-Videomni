"""Object tracking: the three ``/utils/track/...`` endpoints (§3-54).

    POST   /utils/track/sessions                  -> 201 {session_id, width, height, frame_bytes}
    POST   /utils/track/sessions/{sid}/frame      -> 200 {frame, box, score}
    DELETE /utils/track/sessions/{sid}            -> 200 {closed: true}

A SEPARATE NAMESPACE FROM GENERATION, ON PURPOSE. Tracking creates no job, takes
no queue slot, touches no GPU and stores nothing. ``/status`` gets its own
``tracking`` block for the same reason, and the request models below live here
rather than in ``api/models.py`` (which is the generation vocabulary).

THE SERVER RETURNS A RAW BOX AND A RAW SCORE AND NOTHING ELSE. "Lost", smoothing,
size-following and keyframe thinning are all pure functions in the plugin, so a
change to any of them is a plugin change and this contract does not move. That
is the whole reason the settings panel can grow to seven items without an API
version bump.

See Docs/OBJECT_TRACKING_DESIGN.md §4.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import track_frame_invalid
from services import tracking_manager

router = APIRouter()


class TrackBox(BaseModel):
    """A rectangle, top-left corner plus size, in frame pixels.

    Floats, not ints: the plugin derives this from a partial filter's X/Y/size,
    which are themselves sub-pixel, and rounding on the way in would mean the
    box that comes back can never equal the box that went out.
    """

    x: float
    y: float
    w: float = Field(..., gt=0)
    h: float = Field(..., gt=0)


class OpenTrackSessionRequest(BaseModel):
    """Body of ``POST /utils/track/sessions``.

    ``width``/``height`` are the ACTUAL size of the frame the plugin rendered,
    not anything read from the project settings — they fix ``frame_bytes``, and
    every later frame is checked against that one number.

    ``search_factor`` omitted means "the tracker's own default"
    (``tracking.uetrack_runtime.BASE_SEARCH_FACTOR``). It is not restated here:
    a second copy of a default is a second thing that can be wrong. The CEILING
    is restated, because it is a server-side limit rather than a default: the
    search window is cropped at this multiple of the box, so an unbounded value
    is an unbounded crop and resize on the worker's CPU. The settings panel's
    own range (2.0-6.0) sits inside it; this is the outer edge, not the UI's.
    """

    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    box: TrackBox
    search_factor: float | None = Field(default=None, gt=0, le=8.0)


@router.post(
    "/utils/track/sessions",
    status_code=201,
    dependencies=[Depends(require_auth)],
)
async def open_track_session(
    body: OpenTrackSessionRequest,
    context: AppContext = Depends(get_context),
) -> dict:
    """Open the tracking session. 409 if one is already open, 503 if not installed.

    Threadpooled because opening the session is what STARTS the worker on a cold
    server: a ~107 MB checkpoint read plus model construction is seconds of
    blocking work, and it must not sit on the event loop.
    """
    return await run_in_threadpool(
        context.tracking_manager.open,
        width=body.width,
        height=body.height,
        box=(body.box.x, body.box.y, body.box.w, body.box.h),
        search_factor=body.search_factor,
    )


@router.post(
    "/utils/track/sessions/{session_id}/frame",
    dependencies=[Depends(require_auth)],
)
async def push_track_frame(
    session_id: str,
    request: Request,
    # AviUtl2's frame number, echoed back untouched. The server does not
    # interpret it: it exists so the plugin can pair an answer with the frame it
    # drew without keeping its own queue.
    frame: int = Query(...),
    context: AppContext = Depends(get_context),
) -> dict:
    """Send one raw RGBA frame, get one box back.

    The body is read whole (``await request.body()``) rather than streamed: the
    manager needs the complete frame before it can do anything with it, and at
    8.3 MB for 1080p over loopback the read is not the expensive part — the
    inference behind ``run_in_threadpool`` is.

    A body bigger than the ceiling is refused from its ``Content-Length``, i.e.
    BEFORE it is buffered: reading a claimed 200 MB first and then measuring it
    would be paying the whole cost of the mistake to find out. The exact
    per-session length check still happens in the manager — this is only the
    ceiling, and a request without the header is read as before.
    """
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit():
        if int(declared) > tracking_manager.MAX_FRAME_BYTES:
            raise track_frame_invalid(
                detail=(
                    f"Content-Length is {declared} bytes, "
                    f"limit is {tracking_manager.MAX_FRAME_BYTES}"
                )
            )
    data = await request.body()
    box, score = await run_in_threadpool(
        context.tracking_manager.frame, session_id, frame, data
    )
    return {"frame": frame, "box": box, "score": score}


@router.delete(
    "/utils/track/sessions/{session_id}",
    dependencies=[Depends(require_auth)],
)
async def close_track_session(
    session_id: str,
    context: AppContext = Depends(get_context),
) -> dict:
    """Close the session. The plugin sends this on success, cancel and error alike.

    The worker process stays resident — the model load is per server, not per
    session. Only the tracker's per-object state goes away.
    """
    return await run_in_threadpool(context.tracking_manager.close, session_id)
