"""mp4 recipe read-out: ``POST /utils/mp4-info`` (台帳 §3-164).

    POST /utils/mp4-info  {"path": "<absolute local path>"}  -> 200 {"comment": str | null}

Returns the container ``comment`` tag of a video file on this machine — for a
file this server generated, the same JSON text as its ``metadata.json``
(``services.video_io.embed_comment_tag``). Reading is ffprobe's job
(``services.video_io.probe_comment_tag``); there is no parser here.

LOCAL USE ONLY. The body names a path on the SERVER's disk, so the endpoint
answers loopback requests only (403 ``LOCAL_ONLY`` otherwise, including when
the server was started with ``--listen``), and a UNC path is refused before
any filesystem access (422 ``MEDIA_UNREADABLE``) so a request cannot make the
server open a network share. A client connected to another machine cannot
use this endpoint for its own local files.

The request model lives here rather than in ``api/models.py`` (the generation
vocabulary), following ``api/tracking.py``.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api.deps import require_auth
from api.errors import local_only, media_not_found, media_unreadable
from services import video_io

router = APIRouter()

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})


class Mp4InfoRequest(BaseModel):
    """Body of ``POST /utils/mp4-info``: a local path on the server's machine."""

    path: str


class Mp4InfoResponse(BaseModel):
    """The container ``comment`` tag, or ``None`` when the file has none."""

    comment: str | None


@router.post(
    "/utils/mp4-info",
    response_model=Mp4InfoResponse,
    dependencies=[Depends(require_auth)],
)
async def mp4_info(body: Mp4InfoRequest, request: Request) -> Mp4InfoResponse:
    """Read the ``comment`` tag of a local video file.

    403 ``LOCAL_ONLY`` for a non-loopback caller, 422 ``MEDIA_UNREADABLE`` for a
    UNC path or a file ffprobe cannot read, 404 ``MEDIA_NOT_FOUND`` when the path
    does not exist or is not a regular file.
    """
    client_host = request.client.host if request.client is not None else None
    if client_host not in _LOOPBACK_HOSTS:
        raise local_only(detail=f"client: {client_host}")
    # UNC and other ``\\``-drive forms (``//server/share``, ``/\server\share``,
    # ``\\?\UNC\...``, ``\\?\C:\...``, ``\\.\...``) are recognised by the parsed
    # drive rather than by the string prefix, and refused before any
    # filesystem access so a request cannot make the server open a share.
    if PureWindowsPath(body.path).drive.startswith("\\\\"):
        raise media_unreadable(detail="network (UNC) and device paths are not accepted")
    comment = await run_in_threadpool(_read_comment, body.path)
    return Mp4InfoResponse(comment=comment)


def _read_comment(raw_path: str) -> str | None:
    """File check and ffprobe, both blocking, run together off the event loop."""
    path = Path(raw_path)
    if not path.is_file():
        raise media_not_found(raw_path)
    try:
        return video_io.probe_comment_tag(path)
    except video_io.FFmpegError as exc:
        raise media_unreadable(detail=str(exc)[-1000:])
