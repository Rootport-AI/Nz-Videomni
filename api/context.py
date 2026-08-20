"""Shared application context.

A single :class:`AppContext` bundles config and the service singletons. It is
built in ``main.py`` and stored on ``app.state.context`` so routers can reach it
via the ``get_context`` dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import AppConfig
from services.audio_upload_store import AudioUploadStore
from services.base_models import BaseModelDescriptor, load_base_models
from services.job_store import JobStore
from services.join_manager import JoinManager
from services.lora_registry import LoraRegistry
from services.model_registry import ModelRegistry
from services.pipeline_manager import PipelineManager
from services.runtime_state import RuntimeState
from services.upload_store import UploadStore
from services.video_upload_store import VideoUploadStore

logger = logging.getLogger("ltx.state")


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
    #: Last active base model + per-base selection, read ONCE here at startup
    #: (§3-97 P5). Unusable/absent -> the shipped defaults, never a boot error.
    runtime_state: RuntimeState = field(init=False)
    model_registry: ModelRegistry = field(init=False)
    pipeline_manager: PipelineManager = field(init=False)
    join_manager: JoinManager = field(init=False)

    def __post_init__(self) -> None:
        self.base_models = load_base_models(self.config.manifest_dir)
        self.runtime_state = RuntimeState.load(self.config.state_path)
        self.upload_store = UploadStore(self.config)
        self.video_upload_store = VideoUploadStore(self.config)
        self.audio_upload_store = AudioUploadStore(self.config)
        self.join_manager = JoinManager(
            self.config, self.job_store, self.video_upload_store
        )
        self.lora_registry = LoraRegistry(self.config)
        self.model_registry = ModelRegistry(self.config, base_models=self.base_models)
        active_base = self._restore_active_base_model()
        # The registry answers base-less calls with the ACTIVE base model, so
        # it has to start on the restored one too, not on the first descriptor.
        self.model_registry.set_active_base_model(active_base)
        self.pipeline_manager = PipelineManager(
            self.config,
            self.job_store,
            self.upload_store,
            self.video_upload_store,
            self.lora_registry,
            audio_upload_store=self.audio_upload_store,
            # The engine layer builds its worker payload from the base model's
            # descriptor (§3-97 P3b). Which base model that is comes from the
            # runtime state (P5) — the one the operator last loaded — and
            # changes per request through POST /pipeline/load's ``base_model``
            # axis (P6).
            descriptor=self.base_models[active_base],
            runtime_state=self.runtime_state,
            active_base_model=active_base,
            # The remembered category NAMES of THAT base model. Not resolved to
            # paths here on purpose: a name that no longer exists is caught by
            # ``ModelRegistry.resolve``'s 404 when the next load actually asks
            # for it, which keeps startup free of model-store I/O and of a
            # second, divergent copy of the resolution rules.
            active_models=self.runtime_state.selection_for(active_base),
            # Needed to look up the descriptor of a base model a load switches
            # TO, and to publish the new active base back (P6).
            model_registry=self.model_registry,
        )

    def _restore_active_base_model(self) -> str:
        """The base model to start on: the remembered one, else the first.

        A remembered id that no longer exists (the descriptor was removed, or
        the operator downgraded) is NOT an error — it is exactly what the
        fallback is for. It is logged at INFO, because "you are not on the base
        model you left off on" is something the operator should be able to see
        in the console without it reading as a fault.
        """
        remembered = self.runtime_state.active_base_model
        if remembered is not None and remembered in self.base_models:
            return remembered
        first = next(iter(self.base_models))
        if remembered is not None:
            logger.info(
                "前回のベースモデル '%s' は現在の記述子にありません。"
                "'%s' で起動します。",
                remembered,
                first,
            )
        return first


def build_context(config: AppConfig, runtime: RuntimeInfo) -> AppContext:
    return AppContext(config=config, runtime=runtime)
