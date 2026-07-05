"""REST client — the ONLY place /api/v1/* paths + auth headers live."""

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

    def delete_job(self, job_id: str) -> dict:
        """DELETE /jobs/{id}. The server cancels the job if it is still active
        (``{"cancel_requested": True, ...}``) or drops it + its output dir if it
        is terminal (``{"deleted": True, ...}``). Returns the parsed envelope."""
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

    def load_pipeline_models(self, models: dict) -> dict:
        """POST /pipeline/load with a ``models`` selection block (additive S2
        extension: category -> registered NAME). A swap restarts the engine
        worker and can take minutes, so reuse the generous load timeout."""
        r = self.client.post(self._url("/api/v1/pipeline/load"),
                             json={"models": models},
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
        # Reference-video upload (POST /upload/video) for the IC-LoRA control
        # adapters. Videos can be large (up to 200MB), so use a longer timeout
        # than image uploads. Returns the server's ``video_id`` (passed to
        # /generate as ``reference_video_id``).
        with open(path, "rb") as fh:
            files = {"file": (Path(path).name, fh.read())}
        r = self.client.post(self._url("/api/v1/upload/video"), files=files,
                             headers=self.headers, timeout=300)
        r.raise_for_status()
        return r.json()["video_id"]

    def upload_audio(self, path: str) -> str:
        # Source-audio upload (POST /upload/audio) for A2V (audio-driven
        # generation). Audio caps at 50MB (server default), so a 120s timeout
        # is plenty. Returns the server's ``audio_id`` (passed to
        # /generate/chain as ``source_audio.audio_id``).
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
