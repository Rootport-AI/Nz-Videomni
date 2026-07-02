"""Pipeline manager (spec ch.9).

Owns pipeline load/unload lifecycle, auto-load on first generation, OOM cleanup,
and runs a job to completion on a background thread: updates the JobRecord,
encodes the video (via the runner), and writes ``metadata.json`` (spec 10.2).
"""

from __future__ import annotations

import logging
import platform
import sys
import threading
import time
from pathlib import Path

from api.errors import gpu_oom, generation_failed, pipeline_load_failed
from api.models import JobResult, JobStatus
from config import AppConfig
from services import gpu_info, video_io
from services.job_store import JobRecord, JobStore, now_iso
from services.low_vram import build_low_vram_settings, safe_memory_cleanup
from services.ltx_runner import LTXRunner
from services.upload_store import UploadStore

logger = logging.getLogger("ltx.pipeline")


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "outofmemory" in name or "out of memory" in msg or "cuda oom" in msg


class PipelineManager:
    STATE_UNLOADED = "unloaded"
    STATE_LOADING = "loading"
    STATE_READY = "ready"
    STATE_RUNNING = "running"
    STATE_ERROR = "error"

    def __init__(self, config: AppConfig, job_store: JobStore, upload_store: UploadStore):
        self.config = config
        self.job_store = job_store
        self.upload_store = upload_store
        self.low_vram = build_low_vram_settings(config)
        self.runner = LTXRunner(config, self.low_vram)
        self.state = self.STATE_UNLOADED
        self._lock = threading.Lock()

    # --------------------------------------------------------------- status

    @property
    def loaded(self) -> bool:
        return self.runner.loaded

    @property
    def pipeline_type(self) -> str:
        return self.runner.pipeline_type

    def vram_status_block(self) -> dict:
        return self.low_vram.status_block(
            low_vram_disabled_required=self.config.limits.low_vram_disabled_required
        )

    # ------------------------------------------------------------ lifecycle

    def load(self) -> None:
        with self._lock:
            if self.runner.loaded:
                self.state = self.STATE_READY
                return
            self.state = self.STATE_LOADING
        try:
            self.runner.load()
            self.state = self.STATE_READY
        except Exception as exc:
            self.state = self.STATE_ERROR
            self._cleanup_after_error()
            logger.exception("Pipeline load failed")
            raise pipeline_load_failed(detail=str(exc)) from exc

    def unload(self) -> None:
        with self._lock:
            self.runner.unload()
            self.state = self.STATE_UNLOADED

    def _cleanup_after_error(self) -> None:
        try:
            self.runner.unload()
        except Exception:
            pass
        safe_memory_cleanup()
        self.state = self.STATE_UNLOADED

    # ------------------------------------------------------------- run job

    def run_job(self, job: JobRecord) -> None:
        """Run one generation job to terminal state. Intended for a worker thread."""
        job.status = JobStatus.running
        job.started_at = now_iso()
        job.progress = 0.05
        started = time.time()
        output_dir = self.config.output_dir / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Job %s start mode=%s %dx%d frames=%d seed=%d",
            job.job_id,
            job.request.generation_mode,
            job.request.width,
            job.request.height,
            job.request.num_frames,
            job.request.seed,
        )

        try:
            if not self.runner.loaded:
                if not self.config.model.auto_load_on_generate:
                    raise pipeline_load_failed(detail="auto_load_on_generate is disabled")
                self.load()

            self.state = self.STATE_RUNNING

            cond_paths = [
                self.upload_store.path_for(ci.image_id)
                for ci in job.request.conditioning_images
            ]

            def on_progress(step, total, progress):
                job.current_step = step
                job.total_steps = total
                job.progress = progress

            outcome = self.runner.generate(
                job.request,
                output_dir=output_dir,
                progress_callback=on_progress,
                conditioning_image_paths=cond_paths,
            )

            elapsed = time.time() - started
            result = self._finalize(job, outcome, output_dir, elapsed)

            if job.cancel_requested:
                job.status = JobStatus.cancelled
            else:
                job.status = JobStatus.completed
                job.result = result
            job.progress = 1.0
            job.completed_at = now_iso()
            self.state = self.STATE_READY
            logger.info("Job %s done in %.1fs -> %s", job.job_id, elapsed, job.status.value)

        except Exception as exc:
            job.completed_at = now_iso()
            job.status = JobStatus.failed
            if _is_oom(exc):
                err = gpu_oom(job_id=job.job_id, detail=str(exc))
                logger.error("Job %s OOM: %s", job.job_id, exc)
                self._cleanup_after_error()
            else:
                err = generation_failed(job_id=job.job_id, detail=str(exc))
                logger.exception("Job %s failed", job.job_id)
                self.state = self.STATE_READY if self.runner.loaded else self.STATE_UNLOADED
            job.error = f"{err.code}: {err.message} ({err.detail})" if err.detail else f"{err.code}: {err.message}"
        finally:
            safe_memory_cleanup()

    # -------------------------------------------------------- run chain job

    @staticmethod
    def _overlap_leading_pixels(overlap_frames: int) -> int:
        """Leading PIXEL frames a non-first clip shares with its predecessor.

        The overlap is ``K = overlap_frames`` LATENT frames at the clip head.
        Under the causal VAE temporal mapping (latent frame 0 -> 1 pixel, each
        subsequent latent frame -> 8 pixels; time factor 8) those K latent frames
        decode to ``1 + (K-1)*8`` pixel frames, which is exactly what must be
        trimmed from each non-first clip so the timeline is continuous.
        """
        k = max(1, int(overlap_frames))
        return 1 + (k - 1) * 8

    def run_chain_job(self, job: JobRecord) -> None:
        """Run a multi-clip chain to a single concatenated output.mp4.

        Clip 0 is a normal T2V/I2V generate; clips 1..N-1 are seeded from the
        previous clip's Stage-1 carry latent (+ overlap params) and persist their
        own carry for the next clip. Per-clip mp4s are concatenated with the
        leading-overlap trim on clips 1..N-1. The existing single-clip
        :meth:`run_job` is untouched.
        """
        chain = job.chain_request
        assert chain is not None, "run_chain_job requires job.chain_request"

        job.status = JobStatus.running
        job.started_at = now_iso()
        job.progress = 0.02
        started = time.time()
        output_dir = self.config.output_dir / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        n = len(chain.clips)

        logger.info(
            "Chain job %s start clips=%d %dx%d overlap=%d/%.2f seed=%d",
            job.job_id, n, chain.width, chain.height,
            chain.overlap_frames, chain.overlap_strength, chain.seed,
        )

        try:
            if not self.runner.loaded:
                if not self.config.model.auto_load_on_generate:
                    raise pipeline_load_failed(detail="auto_load_on_generate is disabled")
                self.load()
            self.state = self.STATE_RUNNING

            clip_paths: list[Path] = []
            per_clip_seconds: list[float] = []
            prev_carry: Path | None = None
            seed_used = chain.seed
            backend = None
            peak_vram_mb = None

            for i in range(n):
                if job.cancel_requested:
                    break
                clip_req = chain.to_clip_request(i)
                clip_dir = output_dir / f"clip_{i:02d}"
                clip_dir.mkdir(parents=True, exist_ok=True)

                cond_paths = [
                    self.upload_store.path_for(ci.image_id)
                    for ci in clip_req.conditioning_images
                ]

                # Non-last clips persist a carry tail for the next clip; the last
                # clip needs none. Clips after the first are seeded from prev.
                carry_out = (clip_dir / "carry.pt") if i < n - 1 else None

                def on_progress(step, total, progress, _i=i):
                    job.current_step = step
                    job.total_steps = total
                    job.progress = round((_i + max(0.0, min(1.0, progress))) / n, 3)

                clip_started = time.time()
                outcome = self.runner.generate(
                    clip_req,
                    output_dir=clip_dir,
                    progress_callback=on_progress,
                    conditioning_image_paths=cond_paths,
                    prev_clip_latent_path=prev_carry,
                    overlap_frames=chain.overlap_frames,
                    overlap_strength=chain.overlap_strength,
                    carry_latent_out_path=carry_out,
                )
                per_clip_seconds.append(round(time.time() - clip_started, 2))
                clip_paths.append(outcome.output_path)
                prev_carry = outcome.carry_latent_path
                seed_used = outcome.seed_used
                backend = outcome.backend
                if outcome.peak_vram_mb is not None:
                    peak_vram_mb = max(peak_vram_mb or 0, outcome.peak_vram_mb)

                logger.info(
                    "Chain job %s clip %d/%d done in %.1fs carry=%s",
                    job.job_id, i + 1, n, per_clip_seconds[-1], prev_carry,
                )

            # Concatenate with the leading-overlap trim on clips 1..N-1.
            output_path = output_dir / "output.mp4"
            trim_px = self._overlap_leading_pixels(chain.overlap_frames)
            crop = None
            if chain.crop_output is not None:
                crop = (chain.crop_output.width, chain.crop_output.height)
            video_io.concat_mp4s(
                clip_paths,
                output_path,
                frame_rate=chain.frame_rate,
                trim_leading_pixels_per_nonfirst_clip=trim_px,
                crop=crop,
            )

            elapsed = time.time() - started
            total_frames = chain.clips[0].num_frames + sum(
                c.num_frames - trim_px for c in chain.clips[1:]
            )
            duration = round(total_frames / chain.frame_rate, 3)
            if crop is not None:
                resolution = f"{crop[0]}x{crop[1]}"
            else:
                resolution = f"{chain.width}x{chain.height}"
            file_size = output_path.stat().st_size if output_path.exists() else 0

            metadata_path = output_dir / "metadata.json"
            if self.config.output.save_metadata_json:
                self._write_chain_metadata(
                    job=job, chain=chain, metadata_path=metadata_path,
                    resolution=resolution, duration=duration, file_size=file_size,
                    elapsed=elapsed, seed_used=seed_used, backend=backend or "mock",
                    peak_vram_mb=peak_vram_mb, per_clip_seconds=per_clip_seconds,
                    total_frames=total_frames, trim_px=trim_px,
                )

            result = JobResult(
                video_url=f"/api/v1/jobs/{job.job_id}/video",
                duration_seconds=duration,
                resolution=resolution,
                file_size_bytes=file_size,
                generation_time_seconds=round(elapsed, 2),
                seed_used=seed_used,
                output_path=f"outputs/{job.job_id}/output.mp4",
                metadata_path=f"outputs/{job.job_id}/metadata.json",
            )

            if job.cancel_requested:
                job.status = JobStatus.cancelled
            else:
                job.status = JobStatus.completed
                job.result = result
            job.progress = 1.0
            job.completed_at = now_iso()
            self.state = self.STATE_READY
            logger.info(
                "Chain job %s done in %.1fs -> %s (frames=%d)",
                job.job_id, elapsed, job.status.value, total_frames,
            )

        except Exception as exc:
            job.completed_at = now_iso()
            job.status = JobStatus.failed
            if _is_oom(exc):
                err = gpu_oom(job_id=job.job_id, detail=str(exc))
                logger.error("Chain job %s OOM: %s", job.job_id, exc)
                self._cleanup_after_error()
            else:
                err = generation_failed(job_id=job.job_id, detail=str(exc))
                logger.exception("Chain job %s failed", job.job_id)
                self.state = self.STATE_READY if self.runner.loaded else self.STATE_UNLOADED
            job.error = (
                f"{err.code}: {err.message} ({err.detail})" if err.detail
                else f"{err.code}: {err.message}"
            )
        finally:
            safe_memory_cleanup()

    def _write_chain_metadata(
        self, *, job, chain, metadata_path, resolution, duration, file_size,
        elapsed, seed_used, backend, peak_vram_mb, per_clip_seconds,
        total_frames, trim_px,
    ) -> None:
        metadata = {
            "job_id": job.job_id,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": now_iso(),
            "status": "completed",
            "kind": "chain",
            "request": chain.model_dump(),
            "generation_mode": "chain",
            "seed_used": seed_used,
            "generation_time_seconds": round(elapsed, 2),
            "backend": backend,
            "chain": {
                "num_clips": len(chain.clips),
                "overlap_frames": chain.overlap_frames,
                "overlap_strength": chain.overlap_strength,
                "trim_leading_pixels_per_nonfirst_clip": trim_px,
                "per_clip_seconds": per_clip_seconds,
                "total_frames": total_frames,
                "clip_num_frames": [c.num_frames for c in chain.clips],
            },
            "output": {
                "path": f"outputs/{job.job_id}/output.mp4",
                "resolution": resolution,
                "duration_seconds": duration,
                "frame_rate": chain.frame_rate,
                "file_size_bytes": file_size,
            },
            "vram_optimization": self.low_vram.metadata_block(peak_vram_mb=peak_vram_mb),
            "environment": self._environment_block(),
        }
        video_io.save_metadata(metadata_path, metadata)

    # ------------------------------------------------------------ finalize

    def _finalize(self, job: JobRecord, outcome, output_dir: Path, elapsed: float) -> JobResult:
        req = job.request
        if req.crop_output is not None:
            res_w, res_h = req.crop_output.width, req.crop_output.height
        else:
            res_w, res_h = req.width, req.height
        resolution = f"{res_w}x{res_h}"
        duration = round(req.num_frames / req.frame_rate, 3)
        file_size = outcome.output_path.stat().st_size if outcome.output_path.exists() else 0

        metadata_path = output_dir / "metadata.json"
        if self.config.output.save_metadata_json:
            self._write_metadata(
                job=job,
                outcome=outcome,
                metadata_path=metadata_path,
                resolution=resolution,
                duration=duration,
                file_size=file_size,
                elapsed=elapsed,
            )

        return JobResult(
            video_url=f"/api/v1/jobs/{job.job_id}/video",
            duration_seconds=duration,
            resolution=resolution,
            file_size_bytes=file_size,
            generation_time_seconds=round(elapsed, 2),
            seed_used=outcome.seed_used,
            output_path=f"outputs/{job.job_id}/output.mp4",
            metadata_path=f"outputs/{job.job_id}/metadata.json",
        )

    def _write_metadata(self, *, job, outcome, metadata_path, resolution, duration, file_size, elapsed) -> None:
        req = job.request
        metadata = {
            "job_id": job.job_id,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": now_iso(),
            "status": "completed",
            "request": req.model_dump(),
            "generation_mode": outcome.generation_mode,
            "seed_used": outcome.seed_used,
            "generation_time_seconds": round(elapsed, 2),
            "backend": outcome.backend,
            "output": {
                "path": f"outputs/{job.job_id}/output.mp4",
                "resolution": resolution,
                "duration_seconds": duration,
                "frame_rate": req.frame_rate,
                "file_size_bytes": file_size,
            },
            "vram_optimization": self.low_vram.metadata_block(peak_vram_mb=outcome.peak_vram_mb),
            "environment": self._environment_block(),
        }
        video_io.save_metadata(metadata_path, metadata)

    def _environment_block(self) -> dict:
        torch_version = None
        cuda_version = None
        try:
            import torch  # type: ignore

            torch_version = torch.__version__
            cuda_version = getattr(torch.version, "cuda", None)
        except Exception:
            pass
        gpu = gpu_info.get_gpu_info()
        return {
            "python": platform.python_version(),
            "torch": torch_version,
            "cuda": cuda_version,
            "gpu": gpu.get("name"),
            "platform": sys.platform,
        }
