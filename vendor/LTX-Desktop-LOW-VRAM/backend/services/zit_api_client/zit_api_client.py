"""Z-Image Turbo API client protocol for FAL endpoints."""

from __future__ import annotations

from typing import Protocol


class ZitAPIClient(Protocol):
    def generate_text_to_image(
        self,
        *,
        api_key: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
    ) -> bytes:
        ...
