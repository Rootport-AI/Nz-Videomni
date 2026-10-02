"""REST client — the ONLY place the UI's own /api/v1/* requests + auth
headers live. (The Style LoRA gallery's thumbnail URL, fetched by the
browser without auth headers, is built in ``adapters.build_style_gallery``.)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import httpx


class ApiClient:
    """Thin wrapper over the frozen REST API.

    ``client`` is injectable so tests can pass
    ``httpx.Client(transport=httpx.MockTransport(handler))``. When omitted a
    real client is created lazily (never at import/build time).
    """

    def __init__(self, base_url: str, api_key: str | None = None,
                 client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=60)
        return self._client

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # --- read ---
    def get_status(self) -> dict:
        r = self.client.get(self._url("/api/v1/status"), headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_config(self) -> dict:
        r = self.client.get(self._url("/api/v1/config"), headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_job(self, job_id: str) -> dict:
        r = self.client.get(self._url(f"/api/v1/jobs/{job_id}"), headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def list_jobs(self) -> list[dict]:
        """GET /jobs -> the full job list (each item is a JobResponse dict)."""
        r = self.client.get(self._url("/api/v1/jobs"), headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_models(self) -> dict:
        """GET /models — category-scoped model enumeration (model management)."""
        r = self.client.get(self._url("/api/v1/models"), headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def list_loras(self) -> list[dict]:
        """GET /loras -> the LoRA list (each item is a
        ``services.lora_registry.LoraEntryInfo.as_dict()`` row). The server
        rescans on every call, so a freshly dropped-in file shows up without a
        restart."""
        r = self.client.get(self._url("/api/v1/loras"), headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json().get("loras", [])

    def reload_loras(self) -> dict:
        """POST /loras/reload -> ``{total, styles, controls}`` after an explicit
        registry rescan (config + lora_dir)."""
        r = self.client.post(self._url("/api/v1/loras/reload"), headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def delete_job(self, job_id: str) -> dict:
        """DELETE /jobs/{id}. The server cancels the job if it is still active
        (``{"cancelled": True, ...}`` for a queued job, ``{"cancel_requested":
        True, ...}`` for a running one) or drops it + its output dir if it is
        terminal (``{"deleted": True, ...}``). Returns the parsed envelope."""
        r = self.client.delete(self._url(f"/api/v1/jobs/{job_id}"),
                               headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    # --- lifecycle ---
    def load_pipeline(self) -> dict:
        # Model load can be slow; give it a generous timeout.
        r = self.client.post(self._url("/api/v1/pipeline/load"), headers=self.headers, timeout=600)
        r.raise_for_status()
        return r.json()

    def load_pipeline_models(self, models: dict, base_model: str | None = None) -> dict:
        """POST /pipeline/load with a ``models`` selection block (category ->
        registered NAME). A swap restarts the engine worker and can take
        minutes, so reuse the generous load timeout.

        ``base_model`` (multi-engine axis) is added to the body ONLY when the
        caller passes one, so a caller that does not know about base models
        sends a body without the ``base_model`` key."""
        body: dict = {"models": models}
        if base_model is not None:
            body["base_model"] = base_model
        r = self.client.post(self._url("/api/v1/pipeline/load"),
                             json=body,
                             headers=self.headers, timeout=600)
        r.raise_for_status()
        return r.json()

    def unload_pipeline(self) -> dict:
        r = self.client.post(self._url("/api/v1/pipeline/unload"), headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()

    # --- generate flow ---
    def upload_image(self, path: str) -> str:
        with open(path, "rb") as fh:
            files = {"file": (Path(path).name, fh.read())}
        r = self.client.post(self._url("/api/v1/upload/image"), files=files,
                             headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()["image_id"]

    def upload_video(self, path: str) -> str:
        # Video upload (POST /upload/video): the IC-LoRA control adapters'
        # reference video and the Clip Chain V2V source video. Videos can be
        # large (the server caps them at ``upload.max_video_size_mb``), so
        # use a longer timeout than image uploads. Returns the server's
        # ``video_id`` (sent as ``reference_video_id`` or as
        # ``source_video.video_id``).
        with open(path, "rb") as fh:
            files = {"file": (Path(path).name, fh.read())}
        r = self.client.post(self._url("/api/v1/upload/video"), files=files,
                             headers=self.headers, timeout=300)
        r.raise_for_status()
        return r.json()["video_id"]

    def upload_audio(self, path: str) -> str:
        # Source-audio upload (POST /upload/audio) for A2V (audio-driven
        # generation). The server caps audio at ``upload.max_audio_size_mb``,
        # so a 120s timeout is plenty. Returns the server's ``audio_id``
        # (passed to /generate/chain as ``source_audio.audio_id``).
        with open(path, "rb") as fh:
            files = {"file": (Path(path).name, fh.read())}
        r = self.client.post(self._url("/api/v1/upload/audio"), files=files,
                             headers=self.headers, timeout=120)
        r.raise_for_status()
        return r.json()["audio_id"]

    def generate(self, payload: dict) -> httpx.Response:
        # Return the raw response so the caller can branch on 409 / >=400 while
        # keeping the client thin.
        return self.client.post(self._url("/api/v1/generate"), json=payload,
                                headers=self.headers, timeout=60)

    def generate_chain(self, payload: dict) -> httpx.Response:
        # Clip-chain start (POST /generate/chain). Same thin style as
        # ``generate``: return the raw response so the caller branches on
        # 409 / >=400. The endpoint responds 202 with the job envelope.
        return self.client.post(self._url("/api/v1/generate/chain"), json=payload,
                                headers=self.headers, timeout=60)

    def fetch_video(self, job_id: str) -> str:
        """Download the finished mp4 to a temp file and return its path."""
        r = self.client.get(self._url(f"/api/v1/jobs/{job_id}/video"),
                            headers=self.headers, timeout=60)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(prefix=f"{job_id}_", suffix=".mp4", delete=False)
        tmp.write(r.content)
        tmp.close()
        return tmp.name

    # --- V2V join flow ---
    def join_job(self, job_id: str, payload: dict | None = None) -> httpx.Response:
        # Server-side V2V join (POST /jobs/{id}/join). Synchronous on the server
        # (ffmpeg: loudnorm two-pass + re-encode) so give it a generous 120s
        # timeout. Same thin style as ``generate``: return the raw response so
        # the caller branches on >=400 and formats the error envelope.
        return self.client.post(self._url(f"/api/v1/jobs/{job_id}/join"),
                                json=payload or {}, headers=self.headers, timeout=120)

    def fetch_joined(self, job_id: str) -> str:
        """Download joined.mp4 (GET /jobs/{id}/joined) to a temp file and return
        its path. Mirrors :meth:`fetch_video`; long timeout for big outputs."""
        r = self.client.get(self._url(f"/api/v1/jobs/{job_id}/joined"),
                            headers=self.headers, timeout=120)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(prefix=f"{job_id}_joined_", suffix=".mp4",
                                          delete=False)
        tmp.write(r.content)
        tmp.close()
        return tmp.name

    # --- utils ---
    def mp4_info(self, path: str) -> dict:
        """POST /utils/mp4-info -> ``{"comment": str | None}`` for a
        server-local mp4 path (the generation conditions written into the
        ``comment`` tag). Errors (MEDIA_NOT_FOUND 404 / MEDIA_UNREADABLE 422 /
        LOCAL_ONLY 403) raise ``httpx.HTTPStatusError`` like the other read
        methods."""
        r = self.client.post(self._url("/api/v1/utils/mp4-info"), json={"path": path},
                             headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()
