"""Runtime policy query handler."""

from __future__ import annotations

from api_types import RuntimePolicyResponse
from runtime_config.runtime_config import RuntimeConfig


class RuntimePolicyHandler:
    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    def get_runtime_policy(self) -> RuntimePolicyResponse:
        # Server-side single source of truth for forced API mode.
        return RuntimePolicyResponse(force_api_generations=self._config.force_api_generations)
