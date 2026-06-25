"""In-memory single-job store (spec ch.8).

Phase 1 keeps one job in flight. There is no DB and no queue: history lives in
a dict and is lost on restart (the ``outputs/{job_id}/metadata.json`` files
remain on disk). The generate endpoint enforces single-concurrency by checking
:meth:`JobStore.has_active` and returning 409 (JOB_BUSY) otherwise.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

from api.models import GenerateRequest, JobResponse, JobResult, JobStatus


def now_iso() -> str:
    """UTC timestamp like ``2026-06-25T12:00:00Z`` (spec examples)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class JobRecord:
    def __init__(self, job_id: str, request: GenerateRequest):
        self.job_id = job_id
        self.request = request
        self.status: JobStatus = JobStatus.queued
        self.progress: float = 0.0
        self.current_step: int | None = None
        self.total_steps: int | None = None
        self.created_at: str = now_iso()
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.error: str | None = None
        self.result: JobResult | None = None
        self.cancel_requested: bool = False

    @property
    def is_active(self) -> bool:
        return self.status in (JobStatus.queued, JobStatus.running)

    def to_response(self) -> JobResponse:
        return JobResponse(
            job_id=self.job_id,
            status=self.status,
            progress=self.progress,
            current_step=self.current_step,
            total_steps=self.total_steps,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            error=self.error,
            request=self.request,
            result=self.result,
        )


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self, request: GenerateRequest) -> JobRecord:
        record = JobRecord(str(uuid.uuid4()), request)
        with self._lock:
            self._jobs[record.job_id] = record
        return record

    def create_if_idle(self, request: GenerateRequest) -> JobRecord | None:
        """Atomically create a job only if none is active (single-job guard).

        Returns the new record, or ``None`` if a job is already queued/running
        (the caller should respond 409 JOB_BUSY).
        """
        with self._lock:
            if any(r.is_active for r in self._jobs.values()):
                return None
            record = JobRecord(str(uuid.uuid4()), request)
            self._jobs[record.job_id] = record
            return record

    def has_active(self) -> bool:
        with self._lock:
            return any(r.is_active for r in self._jobs.values())

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[JobRecord]:
        with self._lock:
            return list(self._jobs.values())

    def remove(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def counts(self) -> dict[str, int]:
        with self._lock:
            records = list(self._jobs.values())
        return {
            "pending": sum(1 for r in records if r.status == JobStatus.queued),
            "running": sum(1 for r in records if r.status == JobStatus.running),
            "completed": sum(1 for r in records if r.status == JobStatus.completed),
            "failed": sum(1 for r in records if r.status == JobStatus.failed),
        }
