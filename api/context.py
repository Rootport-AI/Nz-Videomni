"""Shared application context.

A single :class:`AppContext` bundles config and the service singletons. It is
built in ``main.py`` and stored on ``app.state.context`` so routers can reach it
via the ``get_context`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import AppConfig
from services.job_store import JobStore
from services.pipeline_manager import PipelineManager
from services.upload_store import UploadStore


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
    pipeline_manager: PipelineManager = field(init=False)

    def __post_init__(self) -> None:
        self.upload_store = UploadStore(self.config)
        self.pipeline_manager = PipelineManager(self.config, self.job_store, self.upload_store)


def build_context(config: AppConfig, runtime: RuntimeInfo) -> AppContext:
    return AppContext(config=config, runtime=runtime)
