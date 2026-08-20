"""Shared application context.

A single :class:`AppContext` bundles config and the service singletons. It is
built in ``main.py`` and stored on ``app.state.context`` so routers can reach it
via the ``get_context`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import AppConfig
from services.audio_upload_store import AudioUploadStore
from services.base_models import BaseModelDescriptor, load_base_models
from services.job_store import JobStore
from services.join_manager import JoinManager
from services.lora_registry import LoraRegistry
from services.model_registry import ModelRegistry
from services.pipeline_manager import PipelineManager
from services.upload_store import UploadStore
from services.video_upload_store import VideoUploadStore


@dataclass
class RuntimeInfo:
    host: str
    port: int
    listen: bool = False
    api_key: str | None = None


@dataclass
class AppContext:
    config: AppConfig
    runtime: RuntimeInfo
    job_store: JobStore = field(default_factory=JobStore)
    upload_store: UploadStore = field(init=False)
    video_upload_store: VideoUploadStore = field(init=False)
    audio_upload_store: AudioUploadStore = field(init=False)
    lora_registry: LoraRegistry = field(init=False)
    #: Base-model descriptors, loaded ONCE at startup from
    #: ``config.model.manifest_dir`` and shared by everything that needs them
    #: (the model registry and the pipeline/engine layer alike). A broken or
    #: missing descriptor fails the server at boot, not per request.
    base_models: dict[str, BaseModelDescriptor] = field(init=False)
    model_registry: ModelRegistry = field(init=False)
    pipeline_manager: PipelineManager = field(init=False)
    join_manager: JoinManager = field(init=False)

    def __post_init__(self) -> None:
        self.base_models = load_base_models(self.config.manifest_dir)
        self.upload_store = UploadStore(self.config)
        self.video_upload_store = VideoUploadStore(self.config)
        self.audio_upload_store = AudioUploadStore(self.config)
        self.join_manager = JoinManager(
            self.config, self.job_store, self.video_upload_store
        )
        self.lora_registry = LoraRegistry(self.config)
        self.model_registry = ModelRegistry(self.config, base_models=self.base_models)
        self.pipeline_manager = PipelineManager(
            self.config,
            self.job_store,
            self.upload_store,
            self.video_upload_store,
            self.lora_registry,
            audio_upload_store=self.audio_upload_store,
            # The engine layer builds its worker payload from the base model's
            # descriptor (§3-97 P3b). P3b pins that to the FIRST declared
            # descriptor — the same one ``ModelRegistry.default_base_model``
            # picks; making it switchable per request is the API axis (P6).
            descriptor=next(iter(self.base_models.values())),
        )


def build_context(config: AppConfig, runtime: RuntimeInfo) -> AppContext:
    return AppContext(config=config, runtime=runtime)
