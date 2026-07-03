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

    def run_chain_job(self, job: JobRecord) -> None:
        """Run a multi-clip chain to ONE continuous output.mp4 (Phase 3 WP4).

        Masked AV-latent concatenation: the whole chain runs INSIDE ONE worker
        invocation (latents resident across segments) — per-segment stage-1 with
        AV carry+freeze, crossfade assembly, always-tiled stage-2, ONE VAE decode.
        Replaces the old per-clip generate + carry.pt + ffmpeg-concat + trim path
        (which cut hard at every boundary). Junction pixel-frame indices (segment
        seams AND stage-2 tile seams) are recorded in metadata for the review
        harness. The single-clip :meth:`run_job` is untouched.

        Cancellation: the chain is now one atomic worker op, so cancel is honored
        at the job boundary (before dispatch) — matching that a single generate is
        also not interruptible mid-run.
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
            if job.cancel_requested:
                job.status = JobStatus.cancelled
                job.progress = 1.0
                job.completed_at = now_iso()
                self.state = self.STATE_READY
                logger.info("Chain job %s cancelled before dispatch", job.job_id)
                return

            if not self.runner.loaded:
                if not self.config.model.auto_load_on_generate:
                    raise pipeline_load_failed(detail="auto_load_on_generate is disabled")
                self.load()
            self.state = self.STATE_RUNNING

            # Only clip 0 may carry conditioning images (validator enforces this).
            clip0_cond_paths = [
                self.upload_store.path_for(ci.image_id)
                for ci in chain.clips[0].conditioning_images
            ]

            def on_progress(step, total, progress):
                job.current_step = step
                job.total_steps = total
                job.progress = round(max(0.0, min(1.0, progress)), 3)

            outcome = self.runner.generate_chain(
                chain,
                output_dir=output_dir,
                progress_callback=on_progress,
                clip0_conditioning_paths=clip0_cond_paths,
            )

            elapsed = time.time() - started
            meta = outcome.chain_metadata or {}
            total_frames = int(meta.get("total_px", 0))
            duration = round(total_frames / chain.frame_rate, 3) if total_frames else 0.0
            if chain.crop_output is not None:
                resolution = f"{chain.crop_output.width}x{chain.crop_output.height}"
            else:
                resolution = f"{chain.width}x{chain.height}"
            output_path = outcome.output_path
            file_size = output_path.stat().st_size if output_path.exists() else 0

            metadata_path = output_dir / "metadata.json"
            if self.config.output.save_metadata_json:
                self._write_chain_metadata(
                    job=job, chain=chain, metadata_path=metadata_path,
                    resolution=resolution, duration=duration, file_size=file_size,
                    elapsed=elapsed, seed_used=outcome.seed_used,
                    backend=outcome.backend or "mock",
                    peak_vram_mb=outcome.peak_vram_mb, total_frames=total_frames,
                    chain_meta=meta,
                )

            result = JobResult(
                video_url=f"/api/v1/jobs/{job.job_id}/video",
                duration_seconds=duration,
                resolution=resolution,
                file_size_bytes=file_size,
                generation_time_seconds=round(elapsed, 2),
                seed_used=outcome.seed_used,
                output_path=f"outputs/{job.job_id}/output.mp4",
                metadata_path=f"outputs/{job.job_id}/metadata.json",
            )

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
        elapsed, seed_used, backend, peak_vram_mb, total_frames, chain_meta,
    ) -> None:
        cm = chain_meta or {}
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
                "architecture": "masked_av_latent_concat",
                "overlap_frames": chain.overlap_frames,
                "overlap_strength": chain.overlap_strength,
                "total_frames": total_frames,
                "clip_num_frames": [c.num_frames for c in chain.clips],
                # Junction pixel-frame indices (0-based last-frame-of-segment; the
                # boundary is J / J+1) for the review harness — segment seams AND
                # stage-2 tile seams, plus the ±1 spread that was actually probed.
                "segment_seam_junctions": cm.get("segment_seam_junctions", []),
                "tile_seam_junctions": cm.get("tile_seam_junctions", []),
                "all_junctions": cm.get("all_junctions", []),
                "video_tiles": cm.get("video_tiles", []),
                "n_tiles": cm.get("n_tiles"),
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
