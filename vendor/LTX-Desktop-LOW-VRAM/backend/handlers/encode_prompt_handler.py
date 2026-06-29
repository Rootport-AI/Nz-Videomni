"""Handler for manually encoding a prompt on GPU (single-GPU mode)."""

from __future__ import annotations

import gc
import logging
import torch
from threading import RLock
from typing import TYPE_CHECKING

from handlers.base import StateHandlerBase
from runtime_config.model_download_specs import resolve_model_path
from services.services_utils import sync_device
from state.app_state_types import TextEncodingResult

if TYPE_CHECKING:
    from handlers.pipelines_handler import PipelinesHandler
    from handlers.text_handler import TextHandler
    from runtime_config.runtime_config import RuntimeConfig
    from state.app_state_types import AppState

logger = logging.getLogger(__name__)


class EncodePromptHandler(StateHandlerBase):
    """Encodes a prompt on GPU with only Gemma loaded — no transformer in VRAM."""

    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
    ) -> None:
        super().__init__(state, lock, config)
        self._pipelines = pipelines_handler
        self._text = text_handler

    def is_single_gpu_local_mode(self) -> bool:
        """True when manual prompt encoding is relevant.

        Only needed in single-GPU + local encoder mode.  Multi-GPU keeps
        Gemma on cuda:1 permanently so the workflow is automatic.
        """
        if self.state.app_settings.use_multi_gpu:
            return False
        return self._text.should_use_local_encoding()

    def get_encoded_prompt(self) -> str | None:
        te = self.state.text_encoder
        return te.encoded_prompt if te is not None else None

    def _get_or_load_encoder(self, gemma_root: str) -> object:
        """Return the cached encoder, cold-loading from disk if necessary."""
        te = self.state.text_encoder
        assert te is not None
        encoder = te.cached_encoder
        if encoder is None:
            logger.info("Cold-start: loading Gemma text encoder from disk to CPU…")
            self._pipelines._install_text_patches_if_needed()
            checkpoint_path = str(
                resolve_model_path(
                    self.models_dir, self.config.model_download_specs, "checkpoint"
                )
            )
            from ltx_pipelines.utils.model_ledger import ModelLedger
            ledger = ModelLedger(
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
                gemma_root_path=gemma_root,
            )
            encoder = ledger.text_encoder()  # type: ignore[assignment]
            if encoder is None or type(encoder).__name__ == "DummyTextEncoder":
                raise RuntimeError(
                    "Failed to load Gemma text encoder — "
                    "DummyTextEncoder returned unexpectedly during cold-start"
                )
        return encoder

    def encode_prompt(self, prompt: str) -> None:
        """Encode a prompt on GPU and cache the embeddings.

        Single-GPU: ejects the video pipeline, encodes on cuda:0, returns encoder
        to CPU.  Multi-GPU: the text encoder already lives on cuda:1 — encode
        there without touching the video pipeline.

        Always stores the result under the key (prompt.strip(), False) because
        local encoding never applies prompt enhancement.
        """
        te = self.state.text_encoder
        if te is None:
            raise RuntimeError("Text encoder state not initialised")

        gemma_root = self._text.resolve_gemma_root()
        if not gemma_root:
            raise RuntimeError(
                "Local text encoder not available. "
                "Download the text encoder or check Settings."
            )

        is_multi_gpu = self.state.app_settings.use_multi_gpu

        # ── 1. Free VRAM (single-GPU only) ──────────────────────────────────
        if not is_multi_gpu:
            self._pipelines.unload_gpu_pipeline()
            logger.info("GPU pipeline ejected; VRAM freed for prompt encoding")

        # ── 2. Obtain GemmaTextEncoder ──────────────────────────────────────
        # Clear api_embeddings so the pipeline uses DummyTextEncoder after caching.
        with self._lock:
            te.api_embeddings = None

        encoder = self._get_or_load_encoder(gemma_root)

        # ── 3. Ensure encoder is on the correct GPU ──────────────────────────
        device = torch.device("cuda:1") if is_multi_gpu else self.config.device

        # Always verify and move — device state may have changed since last call
        # (e.g. mode switched between single/multi-GPU, or encoder returned to CPU).
        try:
            current_device = next(iter(encoder.parameters())).device
        except StopIteration:
            current_device = torch.device("cpu")

        if current_device != device:
            logger.info("Moving Gemma from %s to %s for encoding", current_device, device)
            encoder.to(device)
            sync_device(device)
        else:
            logger.info("Gemma already on %s, encoding in-place", device)

        v: torch.Tensor
        a: torch.Tensor | None
        try:
            # ── 4. Encode using the unpatched encode_text directly ───────────
            # We bypass patched_encode_text (which is designed for the pipeline
            # call-site) and call the original ltx_core function ourselves.
            from ltx_core.text_encoders.gemma.encoders.base_encoder import (
                encode_text as _raw_encode_text,
            )

            with torch.inference_mode():
                raw = _raw_encode_text(encoder, [prompt])

            v_raw, a_raw = raw[0]
            v = v_raw.to(device)
            a = a_raw.to(device) if a_raw is not None else None

        finally:
            # ── 5. Return Gemma to CPU (single-GPU only) ─────────────────────
            # Multi-GPU: leave encoder resident on cuda:1 for subsequent calls.
            if not is_multi_gpu:
                logger.info("Returning Gemma to CPU; freeing VRAM")
                encoder.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # ── 6. Cache the result ──────────────────────────────────────────────
        encoded = TextEncodingResult(video_context=v, audio_context=a)
        prompt_key = (prompt.strip(), False)   # False: no prompt enhancement

        with self._lock:
            max_size = self.state.app_settings.prompt_cache_size
            if max_size > 0:
                if prompt_key in te.prompt_cache:
                    del te.prompt_cache[prompt_key]
                elif len(te.prompt_cache) >= max_size:
                    oldest = next(iter(te.prompt_cache))
                    del te.prompt_cache[oldest]
                te.prompt_cache[prompt_key] = encoded

            # api_embeddings set so that if generate runs immediately the
            # DummyTextEncoder path is taken and the cached result is used.
            te.api_embeddings = encoded
            te.encoded_prompt = prompt.strip()

        if is_multi_gpu:
            logger.info("Prompt encoded on cuda:1 and cached (multi-GPU mode).")
        else:
            logger.info("Prompt encoded on GPU and cached. Gemma returned to CPU. VRAM freed.")

    def enhance_prompt(self, prompt: str) -> str:
        """Enhance a prompt using a locally running LM Studio model.

        Calls the OpenAI-compatible endpoint exposed by LM Studio
        (default: http://localhost:1234/v1/chat/completions).
        No GPU involvement — pure HTTP call to the LM Studio process.
        """
        import json
        import urllib.error
        import urllib.request

        lm_studio_url = "http://localhost:1234/v1/chat/completions"

        system_prompt = (
            "You are a prompt enhancer for LTX-Video, a text-to-video AI model. "
            "Rewrite the user's prompt to be richer and more cinematic. "
            "Rules: "
            "1. Preserve all dialogue exactly as written — do not alter or remove any spoken words. "
            "2. Add specific physical motion and action to characters and the environment — gestures, movement, background activity. "
            "3. Describe lighting, camera angle, atmosphere, and texture with precision. "
            "4. No filler words, no generic phrases — every addition must be concrete and visual. "
            "Output only the enhanced prompt — no explanations, no preamble, no labels."
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 400,
            "stream": True,  # stream tokens to avoid timeout on slow models
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            lm_studio_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            chunks: list[str] = []
            # timeout applies per socket read — resets with each incoming chunk,
            # so slow generation won't timeout as long as tokens keep arriving.
            with urllib.request.urlopen(req, timeout=60) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:"):].strip()
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk_obj = json.loads(payload_str)
                        content = chunk_obj["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            chunks.append(content)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"LM Studio not reachable at {lm_studio_url}. "
                "Make sure LM Studio is running with a model loaded and the "
                "local server is enabled (port 1234)."
            ) from exc

        enhanced = "".join(chunks).strip()
        if not enhanced:
            raise RuntimeError("LM Studio returned an empty response — is a model loaded?")
        logger.info("Prompt enhanced via LM Studio: %r -> %r", prompt[:60], enhanced[:60])
        return enhanced
