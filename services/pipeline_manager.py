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

import chain_math
from api.errors import (
    gpu_oom,
    generation_failed,
    pipeline_load_failed,
    source_audio_too_short,
    source_video_too_short,
)
from api.models import JobResult, JobStatus, SourceAudioSpec, SourceVideoSpec
from config import AppConfig
from services import gpu_info, video_io
from services.audio_upload_store import AudioUploadStore
from services.job_store import JobRecord, JobStore, now_iso
from services.low_vram import build_low_vram_settings, safe_memory_cleanup
from services.lora_registry import LoraRegistry
from services.ltx_runner import LTXRunner
from services.model_registry import CATEGORIES, DEFAULT_NAME
from services.upload_store import UploadStore
from services.video_upload_store import VideoUploadStore

logger = logging.getLogger("ltx.pipeline")


def _peak_vram_suffix(peak_vram_mb) -> str:
    """`` peak_vram=NNNNMB`` for a truthy peak, else ``""`` (mock reports 0/None)."""
    try:
        peak = int(peak_vram_mb)
    except (TypeError, ValueError):
        return ""
    return f" peak_vram={peak}MB" if peak > 0 else ""


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

    def __init__(
        self,
        config: AppConfig,
        job_store: JobStore,
        upload_store: UploadStore,
        video_upload_store: VideoUploadStore | None = None,
        lora_registry: LoraRegistry | None = None,
        audio_upload_store: AudioUploadStore | None = None,
    ):
        self.config = config
        self.job_store = job_store
        self.upload_store = upload_store
        # Phase B: reference-video store + IC-LoRA name registry. Defaulted so
        # existing constructions (tests) still work; the app always injects them.
        self.video_upload_store = video_upload_store or VideoUploadStore(config)
        # A2V: source-audio store. Defaulted like the video store so existing
        # constructions keep working; the app always injects it.
        self.audio_upload_store = audio_upload_store or AudioUploadStore(config)
        self.lora_registry = lora_registry or LoraRegistry(config)
        self.low_vram = build_low_vram_settings(config)
        self.runner = LTXRunner(config, self.low_vram)
        self.state = self.STATE_UNLOADED
        self._lock = threading.Lock()
        # Model management: the category NAME used by the last successful load
        # ("default" until an explicit selection succeeds). Read by GET /models;
        # retained while the worker is unloaded — load-state questions belong to
        # ``pipeline_loaded`` (design ruling §9-6). Never updated on a failed
        # swap-load (the previous successful selection stays authoritative).
        self.active_models: dict[str, str] = {c: DEFAULT_NAME for c in CATEGORIES}

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
            "Job %s start mode=%s %dx%d frames=%d steps=%d fps=%g seed=%d",
            job.job_id,
            job.request.generation_mode,
            job.request.width,
            job.request.height,
            job.request.num_frames,
            job.request.num_inference_steps,
            job.request.frame_rate,
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

            # Phase B/C IC-LoRA: resolve adapter names -> (path, strength,
            # preprocess) via the registry and reference_video_id -> path via the
            # video store. The API layer already validated existence + preprocess-
            # kind conflicts (mirroring conditioning images), so these re-resolve
            # the same objects for the runner hop.
            lora_paths = [
                self.lora_registry.resolve(spec.name, spec.strength)
                for spec in job.request.loras
            ]
            reference_video_path = (
                self.video_upload_store.path_for(job.request.reference_video_id)
                if job.request.reference_video_id
                else None
            )

            def on_progress(step, total, progress):
                job.current_step = step
                job.total_steps = total
                job.progress = progress

            outcome = self.runner.generate(
                job.request,
                output_dir=output_dir,
                progress_callback=on_progress,
                conditioning_image_paths=cond_paths,
                lora_paths=lora_paths,
                reference_video_path=reference_video_path,
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
            logger.info(
                "Job %s done in %.1fs%s -> %s",
                job.job_id, elapsed, _peak_vram_suffix(outcome.peak_vram_mb), job.status.value,
            )

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

    # --------------------------------------------------- V2V source preflight

    def preflight_source_video(
        self, source_video: SourceVideoSpec, request_frame_rate: float
    ) -> None:
        """Validate an uploaded V2V continuation source BEFORE a job is created.

        ffprobes the stored source for fps + frame count and rejects (422
        SOURCE_VIDEO_TOO_SHORT) when it cannot supply the requested
        ``context_frames`` tail after resampling to ``request_frame_rate``. The
        video_id is assumed already resolved (the endpoint 404s first). Geometry
        bounds (8n+1, [25,145], context < clip-0) are enforced by the schema.
        """
        src_path = self.video_upload_store.path_for(source_video.video_id)
        n_src = video_io.frame_count(src_path)
        src_fps = video_io.probe_fps(src_path)
        # Frame count after resampling to the request fps. Exact for the no-resample
        # case; a duration-based estimate for the resample case (cut_tail_mp4 is the
        # frame-exact backstop, which raises if the resampled source is still short).
        if src_fps and abs(src_fps - float(request_frame_rate)) > 1e-3:
            effective = int(round(n_src * float(request_frame_rate) / src_fps))
        else:
            effective = n_src
        if effective < source_video.context_frames:
            raise source_video_too_short(
                detail=(
                    f"source has {n_src} frames @ {src_fps} fps "
                    f"(~{effective} @ {request_frame_rate} fps) < "
                    f"context_frames={source_video.context_frames}"
                )
            )

    # --------------------------------------------------- A2V source preflight

    def preflight_source_audio(
        self,
        source_audio: SourceAudioSpec,
        clip_frames: list[int],
        frame_rate: float,
        overlap_frames: int,
    ) -> None:
        """Validate an uploaded A2V source audio BEFORE a job is created.

        Resolves the ``audio_id`` (the endpoint 404s first) and ffprobes the
        stored file for an audio stream + duration. The chain timeline needs
        ``chain_math.audio_latents_required(...)`` audio-latent frames (25/sec —
        :data:`chain_math.AUDIO_LATENTS_PER_SEC`, the SINGLE SOURCE OF TRUTH the
        engine also encodes against); an upload whose duration VAE-encodes to
        fewer than that is rejected (422 SOURCE_AUDIO_TOO_SHORT). Video length is
        authoritative — audio is truncated, never padded (matches upstream a2vid
        and the engine's ``a2v_avail < a_total`` guard). No fps resample is done.
        """
        src_path = self.audio_upload_store.path_for(source_audio.audio_id)
        required = chain_math.audio_latents_required(
            clip_frames, frame_rate, kv=overlap_frames
        )
        if not video_io.has_audio_stream(src_path):
            raise source_audio_too_short(
                detail=f"no decodable audio stream in upload {source_audio.audio_id}"
            )
        duration = video_io.probe_duration(src_path)
        if duration is None:
            # ffprobe unavailable / unreadable duration: cannot verify length here.
            # The engine's a2v_avail < a_total guard is the frame-exact backstop.
            return
        available = round(duration * chain_math.AUDIO_LATENTS_PER_SEC)
        if available < required:
            raise source_audio_too_short(
                detail=(
                    f"audio is {duration:.3f}s (~{available} audio-latent frames) "
                    f"< required {required} frames for the {sum(clip_frames)}-frame "
                    f"timeline @ {frame_rate} fps"
                )
            )

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
            "Chain job %s start clips=%d %dx%d steps=%d fps=%g overlap=%d/%.2f seed=%d",
            job.job_id, n, chain.width, chain.height,
            chain.num_inference_steps, chain.frame_rate,
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

            # V2V continuation: cut the fps-correct source tail the engine needs
            # (last context_frames frames at the request fps; resampled if the
            # source fps differs). The engine does NOT resample. Provenance
            # (source_fps, resampled) is recorded in the v2v metadata block.
            source_tail_path = None
            source_context_frames = None
            v2v_provenance = None
            if chain.source_video is not None:
                src_path = self.video_upload_store.path_for(chain.source_video.video_id)
                source_context_frames = chain.source_video.context_frames
                source_tail_path = output_dir / "_source_tail.mp4"
                cut = video_io.cut_tail_mp4(
                    src_path, source_tail_path, source_context_frames, chain.frame_rate,
                )
                v2v_provenance = {
                    "source_video_id": chain.source_video.video_id,
                    "source_fps": cut["source_fps"],
                    "resampled": cut["resampled"],
                }

            # A2V: resolve the uploaded audio path (byte-passed to the engine as-is
            # — no cut/resample; the engine truncates to the timeline). Mutually
            # exclusive with source_video (validator enforces this), so only one of
            # source_tail_path / source_audio_path is ever set.
            source_audio_path = None
            a2v_provenance = None
            if chain.source_audio is not None:
                source_audio_path = self.audio_upload_store.path_for(
                    chain.source_audio.audio_id
                )
                a2v_provenance = {"source_audio_id": chain.source_audio.audio_id}

            def on_progress(step, total, progress):
                job.current_step = step
                job.total_steps = total
                job.progress = round(max(0.0, min(1.0, progress)), 3)

            outcome = self.runner.generate_chain(
                chain,
                output_dir=output_dir,
                progress_callback=on_progress,
                clip0_conditioning_paths=clip0_cond_paths,
                source_tail_path=source_tail_path,
                source_context_frames=source_context_frames,
                source_audio_path=source_audio_path,
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
                    chain_meta=meta, v2v_provenance=v2v_provenance,
                    a2v_provenance=a2v_provenance,
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
                "Chain job %s done in %.1fs%s -> %s (frames=%d)",
                job.job_id, elapsed, _peak_vram_suffix(outcome.peak_vram_mb),
                job.status.value, total_frames,
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
        v2v_provenance=None, a2v_provenance=None,
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
        # V2V continuation (additive): only present when a source_video was used,
        # so a normal chain's metadata key set is byte-unchanged. The engine's
        # (or mock's) chain.v2v sub-dict + the app-side provenance (source_video_id,
        # source_fps, resampled).
        v2v = cm.get("v2v")
        if v2v is not None:
            metadata["v2v"] = {**v2v, **(v2v_provenance or {})}
        # A2V continuation (additive): only present when a source_audio was used,
        # so a normal chain's metadata key set is byte-unchanged. The engine's (or
        # mock's) chain.a2v sub-dict + the app-side provenance (source_audio_id).
        a2v = cm.get("a2v")
        if a2v is not None:
            metadata["a2v"] = {**a2v, **(a2v_provenance or {})}
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
        # Phase B IC-LoRA: additive block, only present for lora jobs so non-lora
        # metadata keeps its exact prior key set. (The two new GenerateRequest
        # fields also appear inside the frozen-additive ``request`` dump.)
        if req.loras:
            metadata["ic_lora"] = {
                # Phase C: additive ``preprocess`` field (control-signal kind per
                # adapter). Existing ``name``/``strength``/``reference_video_id``
                # keys are unchanged so Phase B metadata parsers keep working.
                "loras": [
                    {
                        "name": spec.name,
                        "strength": spec.strength,
                        "preprocess": self.lora_registry.preprocess_for(spec.name),
                    }
                    for spec in req.loras
                ],
                "reference_video_id": req.reference_video_id,
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
