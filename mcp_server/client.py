"""HTTP client wrapping the existing FastAPI backend for MCP tools.

One :class:`BackendClient` instance lives for the MCP server process
(``get_client()`` / ``set_client()`` below are the only access points -- tools
never construct a client themselves). Its ``httpx.AsyncClient`` is built lazily
on first use so importing this module never opens a socket.

Error translation (plan D7): every non-2xx response is translated to a single
``mcp.server.fastmcp.exceptions.ToolError`` carrying ``"CODE: message —
detail"`` from the backend's ``{"error": {code, message, detail}}`` envelope
(``api/errors.py``). A transport-level failure (backend not running) becomes a
``ToolError`` telling the user to run ``run.bat`` -- EXCEPT ``httpx.ReadTimeout``,
which passes through UNTRANSLATED: ``/pipeline/load`` and
``/jobs/{id}/join`` keep running server-side after the client socket times out
(plan D5), so callers that care (``tools/system.py::load_pipeline`` etc.) catch
``httpx.ReadTimeout`` themselves and report a ``finished: false`` status
instead of a hard error.
"""

from __future__ import annotations

from typing import Any

import httpx
from mcp.server.fastmcp.exceptions import ToolError

from mcp_server.settings import Settings, load_settings

# Every backend route lives under this prefix (main.py: `app.include_router(
# api_router, prefix="/api/v1")`). Settings.base_url deliberately excludes it
# so backend_status can show the plain "http://host:port" a human would type
# into a browser.
API_PREFIX = "/api/v1"


class BackendClient:
    """Thin async wrapper over the backend's HTTP API.

    ``transport`` is test-only: production always uses ``settings.base_url``
    over real TCP; tests pass an ``httpx.ASGITransport`` bound to the mock
    FastAPI app so no server process is needed.
    """

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport
        self._http: httpx.AsyncClient | None = None
        # Set/cleared by tools/system.py::load_pipeline around its POST
        # /pipeline/load call. Guards against a second agent turn issuing a
        # concurrent load while the first is still in flight (pipeline_manager
        # has a real double-load race -- see the plan's "重要な事実" section).
        self.pipeline_load_in_flight: bool = False

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    @property
    def api_key_configured(self) -> bool:
        return self._settings.api_key is not None

    @property
    def output_dir(self) -> Any:
        return self._settings.output_dir

    @property
    def upload(self) -> Any:
        """The local ``config.yaml`` ``upload:`` section (W2 tools/uploads.py
        precheck source -- see Settings.upload's docstring note)."""
        return self._settings.upload

    def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None:
            headers: dict[str, str] = {}
            if self._settings.api_key:
                headers["Authorization"] = f"Bearer {self._settings.api_key}"
            self._http = httpx.AsyncClient(
                base_url=self._settings.base_url,
                headers=headers,
                transport=self._transport,
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ----- request helpers -------------------------------------------------

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        http = self._ensure_http()
        try:
            resp = await http.request(method, API_PREFIX + path, **kwargs)
        except httpx.ReadTimeout:
            raise  # deliberately untranslated -- see module docstring / plan D5
        except httpx.RequestError as exc:
            raise self._unreachable_error(exc) from exc
        return self._parse_response(resp)

    async def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request("GET", path, params=params)

    async def post_json(
        self, path: str, json: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"json": json}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await self._request("POST", path, **kwargs)

    async def post_file(
        self,
        path: str,
        *,
        files: dict[str, Any],
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        # ``params`` is the QUERY string, not the multipart body: POST
        # /upload/video declares trim_start_sec / trim_duration_sec / max_frames
        # as ``Query(...)`` (api/uploads.py), so they must ride on the URL. Sent
        # only when a caller actually passes one, so an ordinary upload's
        # request line is byte-identical to what it was before this parameter
        # existed.
        kwargs: dict[str, Any] = {"files": files, "data": data}
        if params is not None:
            kwargs["params"] = params
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await self._request("POST", path, **kwargs)

    async def delete_json(self, path: str) -> dict[str, Any]:
        return await self._request("DELETE", path)

    # ----- error translation -------------------------------------------------

    def _parse_response(self, resp: httpx.Response) -> dict[str, Any]:
        if resp.status_code < 400:
            if not resp.content:
                return {}
            return resp.json()
        self._raise_for_error(resp)
        raise AssertionError("_raise_for_error must always raise")  # pragma: no cover

    def _raise_for_error(self, resp: httpx.Response) -> None:
        """Translate the ``{"error": {code, message, detail}}`` envelope (or any
        non-conforming error body) into one ``ToolError``."""
        try:
            payload = resp.json()
        except ValueError:
            raise ToolError(
                f"HTTP_{resp.status_code}: バックエンドが不正なエラー応答を返しました（JSON以外）"
            ) from None
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            raise ToolError(f"HTTP_{resp.status_code}: バックエンドが想定外のエラー形式を返しました")
        code = error.get("code") or f"HTTP_{resp.status_code}"
        message = error.get("message", "unknown error")
        detail = error.get("detail")
        text = f"{code}: {message}"
        if detail:
            text += f" — {detail}"
        raise ToolError(text)

    def _unreachable_error(self, exc: httpx.RequestError) -> ToolError:
        return ToolError(
            "BACKEND_UNREACHABLE: バックエンドに接続できません。"
            "run.bat でバックエンドを起動してください "
            f"(base_url={self._settings.base_url}) — {exc}"
        )


# ----- process-wide singleton -------------------------------------------------

_client: BackendClient | None = None


def get_client() -> BackendClient:
    """Return the process-wide :class:`BackendClient`, building it from
    :func:`load_settings` on first use."""
    global _client
    if _client is None:
        _client = BackendClient(load_settings())
    return _client


def set_client(client: BackendClient | None) -> None:
    """Install (or clear, with ``None``) the process-wide client.

    Test-only hook: tests build a :class:`BackendClient` pointed at an
    ``httpx.ASGITransport`` wrapping the mock FastAPI app and install it here so
    tool functions (which always call :func:`get_client`) talk to the mock
    without any real socket.
    """
    global _client
    _client = client
