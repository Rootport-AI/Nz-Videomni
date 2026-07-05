"""LTX-AviUtl2-Bridge — FastAPI entrypoint (spec 3 / 4 / 12.4).

Boots the FastAPI app, applies CLI overrides, registers the API router under
/api/v1, mounts the Gradio test UI at /ui, and configures CORS + logging.

Environment isolation (spec 2.5): this process never touches the system Python.
Run it via ``run.ps1`` or the project venv's interpreter.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import re
import socket
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response

from api.context import RuntimeInfo, build_context
from api.errors import APIError
from api.router import api_router
from config import load_config

logger = logging.getLogger("ltx")

LOCALHOST_CORS_REGEX = r"^http://(127\.0\.0\.1|localhost)(:\d+)?$"


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(log_dir / "server.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    # S0: the Gradio test UI polls GET /jobs/{id} once per second via an httpx
    # client (gradio_ui/handlers.py::_poll_job_until_done). httpx/httpcore log
    # every request at INFO ("HTTP Request: GET ... 200 OK"), and because those
    # loggers propagate to the root logger configured above, that one line per
    # second floods BOTH the console and server.log — drowning the meaningful
    # job/progress/VRAM lines (historically ~69% of server.log). Raise their
    # threshold to WARNING so routine polling is silent while genuine transport
    # errors still surface. uvicorn's access log is left at its default (it is a
    # separate logger with its own handler and far lower volume — one line per
    # request, not the httpx "HTTP Request:" echo).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


class JobPollingAccessFilter(logging.Filter):
    """Drop uvicorn access-log lines for SUCCESSFUL job-status polling GETs.

    The Gradio UI polls ``GET /api/v1/jobs/{job_id}`` once per second while a
    job runs; with uvicorn's default access log that is one console line per
    second for minutes (the "access-log flood" from the G3 gate). This filter
    suppresses exactly that traffic and nothing else:

    * only ``GET`` (POST /generate, DELETE /jobs/... always show),
    * only the job-status resource itself — subpaths like
      ``/api/v1/jobs/{id}/video`` or ``/joined`` and the job LIST
      ``/api/v1/jobs`` still show,
    * only status 200 — a 404/500 on the polling URL still shows.

    uvicorn's access LogRecord carries ``record.args ==
    (client_addr, method, full_path, http_version, status_code)``
    (uvicorn.protocols.utils / logging AccessFormatter contract). The tuple is
    handled defensively: any unexpected shape lets the record through
    (fail-open — we would rather log too much than eat a real access line).
    """

    _JOB_STATUS_GET = re.compile(r"^/api/v1/jobs/[^/?#]+/?(?:[?#].*)?$")

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            args = record.args
            if not isinstance(args, tuple) or len(args) != 5:
                return True
            _client, method, path, _http_version, status = args
            if method != "GET" or status != 200:
                return True
            if not isinstance(path, str):
                return True
            return not self._JOB_STATUS_GET.match(path)
        except Exception:
            return True  # fail-open: never let log filtering break logging


def build_uvicorn_log_config() -> dict:
    """uvicorn's default logging dictConfig + the job-polling access filter.

    Deep-copied so the module-level ``uvicorn.config.LOGGING_CONFIG`` template
    is never mutated. Everything else (formatters, levels, handlers) stays
    byte-identical to uvicorn's defaults; only the ``access`` handler gains the
    :class:`JobPollingAccessFilter`.
    """
    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    filters = log_config.setdefault("filters", {})
    # "()" as a callable is the documented dictConfig custom-factory hook; the
    # class object (not a dotted path) keeps this immune to __main__ vs main
    # module-name ambiguity.
    filters["job_polling_access"] = {"()": JobPollingAccessFilter}
    access_handler = log_config.get("handlers", {}).get("access")
    if isinstance(access_handler, dict):
        handler_filters = access_handler.setdefault("filters", [])
        if "job_polling_access" not in handler_filters:
            handler_filters.append("job_polling_access")
    return log_config


def build_app(args: argparse.Namespace) -> FastAPI:
    config = load_config(args.config)

    # Apply CLI overrides on top of config.yaml.
    if args.port is not None:
        config.server.port = args.port
    if args.allow_all_cors:
        config.server.allow_all_cors = True
    if args.api_key is not None:
        config.server.api_key = args.api_key
    if args.te_offload is not None:
        config.vram.te_offload_text_encoder = args.te_offload
    if args.dit_cpu_load is not None:
        config.vram.dit_cpu_load = args.dit_cpu_load

    host = "0.0.0.0" if args.listen else config.server.host

    configure_logging(config.log_dir)

    if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        logger.warning(
            "PYTORCH_CUDA_ALLOC_CONF is not set. "
            'Consider $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" to reduce OOM.'
        )

    app = FastAPI(title="LTX-AviUtl2-Bridge", version="0.4.0")

    runtime = RuntimeInfo(
        host=host,
        port=config.server.port,
        listen=args.listen,
        api_key=config.server.api_key,
    )
    app.state.context = build_context(config, runtime)

    # CORS (spec 3.1 / 3.2)
    if config.server.allow_all_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=LOCALHOST_CORS_REGEX,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api/v1")
    ui_mounted = mount_gradio(app, runtime)
    register_root_route(app, ui_mounted=ui_mounted)

    return app


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error_handler(_request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_envelope())

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # Keep FastAPI's 422 but wrap in our error envelope shape. exc.errors()
        # can carry a non-serializable ctx (the raised ValueError); keep only
        # JSON-safe fields.
        detail = [
            {"loc": list(e.get("loc", [])), "msg": str(e.get("msg", "")), "type": str(e.get("type", ""))}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "detail": detail,
                }
            },
        )


def mount_gradio(app: FastAPI, runtime: RuntimeInfo) -> bool:
    """Mount the Gradio test UI at /ui. Returns True iff it actually mounted."""
    if os.environ.get("LTX_DISABLE_GRADIO"):
        logger.info("Gradio UI disabled via LTX_DISABLE_GRADIO")
        return False
    try:
        import gradio as gr

        from gradio_ui import build_ui

        base_url = f"http://127.0.0.1:{runtime.port}"
        blocks = build_ui(base_url, api_key=runtime.api_key)
        # Dark theme is the default (spec: Settings->Theme switches to light).
        # gr.Blocks(js=) is deprecated in gradio 6, so the startup js is passed
        # at the mount site (mount_gradio_app accepts js=; routes.py ~L2472).
        gr.mount_gradio_app(
            app, blocks, path="/ui",
            js="() => { document.body.classList.add('dark'); }",
        )
        return True
    except Exception:
        logger.exception("Failed to mount Gradio UI; continuing with API only")
        return False


def register_root_route(app: FastAPI, *, ui_mounted: bool) -> None:
    """GET / -- redirect to the UI when it's mounted, else a tiny API landing JSON.

    Without this, visiting http://host:port/ (the natural first thing to try)
    hits FastAPI's default 404 even though the UI is happily running at /ui.
    """
    if ui_mounted:

        @app.get("/", include_in_schema=False)
        async def _root() -> RedirectResponse:
            return RedirectResponse(url="/ui", status_code=307)
    else:

        @app.get("/", include_in_schema=False)
        async def _root() -> JSONResponse:
            return JSONResponse({"service": "LTX-AviUtl2-Bridge", "ui": None, "docs": "/docs"})

    @app.get("/favicon.ico", include_in_schema=False)
    async def _favicon() -> Response:
        return Response(status_code=204)


def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LTX-AviUtl2-Bridge backend")
    parser.add_argument("--listen", action="store_true", help="bind 0.0.0.0 (home LAN)")
    parser.add_argument("--port", type=int, default=None, help="override server port")
    parser.add_argument("--api-key", type=str, default=None, help="require Bearer api-key")
    parser.add_argument("--allow-all-cors", action="store_true", help="allow all CORS origins")
    parser.add_argument("--config", type=str, default=None, help="path to config.yaml")
    parser.add_argument(
        "--te-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="CPU-offload (sequential per-layer stream) the Gemma text encoder during encode (default: config / ON)",
    )
    parser.add_argument(
        "--dit-cpu-load",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Build the DiT (transformer) on CPU and stream blocks to GPU during denoise, avoiding the ~16.9GB load-time GPU spike (default: config / ON)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args)
    runtime: RuntimeInfo = app.state.context.runtime

    if args.listen:
        logger.warning("--listen enabled: server is reachable on your LAN (no internet exposure intended).")
        logger.warning("UI: http://%s:%d/ui", local_ip(), runtime.port)
    logger.info("Starting server on %s:%d  (UI: http://127.0.0.1:%d/ui)", runtime.host, runtime.port, runtime.port)

    uvicorn.run(
        app,
        host=runtime.host,
        port=runtime.port,
        log_level="info",
        # F1 (G3 feedback): uvicorn defaults + a filter that mutes the 1-line/s
        # GET /api/v1/jobs/{id} 200 polling flood (see JobPollingAccessFilter).
        log_config=build_uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
