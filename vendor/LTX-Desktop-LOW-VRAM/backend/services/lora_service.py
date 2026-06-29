"""CivitAI and standard LoRA loading service for LTX-2.

Handles loading .safetensors LoRA files, detecting their key format
(official Lightricks vs CivitAI/diffusers-style), remapping keys as
needed, and applying them to the transformer via the ModelLedger.

Supports multiple LoRAs simultaneously with per-LoRA strength control.

Usage:
    service = LoraService(device=torch.device("cuda:0"))
    loras = service.load_loras([
        LoraEntry(path="/path/to/lora.safetensors", strength=0.8),
    ])
    service.apply_to_model_ledger(model_ledger, loras)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Data models                                                          #
# ------------------------------------------------------------------ #

@dataclass
class LoraEntry:
    """A single LoRA to load and apply."""
    path: str
    strength: float = 1.0
    enabled: bool = True


@dataclass
class LoadedLora:
    """A loaded and key-remapped LoRA ready to apply."""
    path: str
    strength: float
    state_dict: dict[str, torch.Tensor]
    format: str  # "lightricks" | "civitai" | "diffusers" | "unknown"


# ------------------------------------------------------------------ #
# Key format detection and remapping                                   #
# ------------------------------------------------------------------ #

# CivitAI / diffusers LoRAs use these key prefixes for the transformer.
_CIVITAI_TRANSFORMER_PREFIXES = (
    "lora_unet_",
    "transformer.",
    "lora_transformer_",
    "diffusion_model.",  # ComfyUI convention for LTX LoRAs
)

# Official Lightricks LoRAs use these prefixes.
_LIGHTRICKS_PREFIXES = (
    "transformer_blocks.",
    "single_transformer_blocks.",
)

# Diffusers lora suffix patterns.
_DIFFUSERS_LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
_CIVITAI_LORA_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".alpha")


def _detect_format(state_dict: dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if not keys:
        return "unknown"

    sample = keys[0]

    if any(sample.startswith(p) for p in _LIGHTRICKS_PREFIXES):
        return "lightricks"

    if any(s in sample for s in _CIVITAI_LORA_SUFFIXES):
        if any(sample.startswith(p) for p in _CIVITAI_TRANSFORMER_PREFIXES):
            return "civitai"

    if any(s in sample for s in _DIFFUSERS_LORA_SUFFIXES):
        return "diffusers"

    # Check a broader sample.
    for key in keys[:20]:
        if ".lora_down." in key or ".lora_up." in key:
            return "civitai"
        if ".lora_A." in key or ".lora_B." in key:
            return "diffusers"

    return "unknown"


def _remap_civitai_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap CivitAI-style keys to Lightricks transformer key format.

    CivitAI pattern:  lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight
    Lightricks pattern: transformer_blocks.0.attn.to_q.lora_down.weight
    """
    remapped: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        new_key = key

        # Strip common CivitAI prefixes.
        for prefix in ("lora_unet_", "lora_transformer_"):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break

        # Replace underscores-as-dots back to dots for known block patterns.
        # e.g. transformer_blocks_0_attn -> transformer_blocks.0.attn
        new_key = _underscores_to_dots(new_key)

        remapped[new_key] = tensor

    return remapped


def _remap_diffusers_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap diffusers / ComfyUI lora_A/lora_B keys to lora_down/lora_up convention.

    Two prefix conventions exist in the wild:
    - Diffusers: ``transformer.transformer_blocks.0.attn1.to_q.lora_A.weight``
    - ComfyUI:   ``diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight``

    The LTX transformer's ``named_parameters()`` (from velocity_model) does NOT
    include either outer prefix — keys start with ``transformer_blocks.`` directly.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        new_key = key
        # Strip leading module wrappers added by diffusers or ComfyUI.
        if new_key.startswith("transformer."):
            new_key = new_key[len("transformer."):]
        elif new_key.startswith("diffusion_model."):
            new_key = new_key[len("diffusion_model."):]
        new_key = new_key.replace(".lora_A.weight", ".lora_down.weight")
        new_key = new_key.replace(".lora_B.weight", ".lora_up.weight")
        remapped[new_key] = tensor
    return remapped


def _underscores_to_dots(key: str) -> str:
    """Convert CivitAI underscore-separated key segments to dot notation.

    Handles patterns like:
      transformer_blocks_0_attn_to_q  ->  transformer_blocks.0.attn.to_q
    """
    # Known structural segments that use underscores legitimately.
    # We convert numeric segments and known sub-module names.
    import re
    # Insert dots before numeric segments (block indices).
    key = re.sub(r'_(\d+)_', r'.\1.', key)
    key = re.sub(r'_(\d+)$', r'.\1', key)
    # Convert remaining structural underscores for known patterns.
    for segment in (
        "transformer_blocks",
        "single_transformer_blocks",
        "attn",
        "ff",
        "norm",
        "proj",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "net",
    ):
        # Only replace if it appears as a standalone underscore-joined segment.
        key = key.replace(f"_{segment}_", f".{segment}.")
        key = key.replace(f"_{segment}.", f".{segment}.")
        if key.endswith(f"_{segment}"):
            key = key[: -len(f"_{segment}")] + f".{segment}"

    return key


# ------------------------------------------------------------------ #
# LoRA service                                                         #
# ------------------------------------------------------------------ #

class LoraService:
    """Loads and applies LoRA weights to the LTX-2 transformer."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def load_loras(self, entries: list[LoraEntry]) -> list[LoadedLora]:
        """Load and remap a list of LoRA entries from disk."""
        loaded: list[LoadedLora] = []
        for entry in entries:
            if not entry.enabled:
                continue
            lora = self._load_single(entry)
            if lora is not None:
                loaded.append(lora)
        return loaded

    def apply_to_model_ledger(
        self,
        model_ledger: Any,
        loras: list[LoadedLora],
    ) -> None:
        """Apply loaded LoRAs to a ModelLedger instance.

        Attempts to use the ModelLedger's native loras= parameter first.
        Falls back to forward-hook patching if unavailable.
        """
        if not loras:
            return

        try:
            from ltx_core.loader.lora import apply_lora_to_model
            transformer = model_ledger.transformer()
            for lora in loras:
                logger.info(
                    "Applying LoRA %s (format=%s, strength=%.2f)",
                    Path(lora.path).name,
                    lora.format,
                    lora.strength,
                )
                apply_lora_to_model(
                    model=transformer,
                    lora_state_dict=lora.state_dict,
                    scale=lora.strength,
                )
            logger.info("Applied %d LoRA(s) via ltx_core loader", len(loras))
        except Exception as exc:
            logger.warning(
                "ltx_core LoRA loader unavailable (%s), falling back to direct merge",
                exc,
            )
            transformer = model_ledger.transformer()
            self.apply_hooks_to_transformer(transformer, loras)

    def apply_hooks_to_transformer(
        self,
        transformer: Any,
        loras: list[LoadedLora],
    ) -> None:
        """Apply LoRAs via runtime forward hooks on an already-built transformer.

        This is preferable to weight merging because:
        - Works correctly with FP8 quantized weights (no re-quantization loss)
        - Works with block swap (hooks survive device moves, weights loaded on demand)
        - The LoRA delta is applied in the computation dtype (bf16) not storage dtype (fp8)

        For each matched Linear, we patch its forward to compute:
            out = base_forward(x) + scale * F.linear(F.linear(x, down), up)
        """
        import torch.nn.functional as F

        try:
            # transformer() returns X0Model wrapping velocity_model.
            # LoRA keys are in the velocity_model namespace.
            lookup_module = getattr(transformer, "velocity_model", transformer)

            # Build a map: module_name -> list of (down, up, scale)
            lora_map: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] = {}

            for lora in loras:
                sd = lora.state_dict
                keys = list(sd.keys())
                down_keys = [k for k in keys if k.endswith(".lora_down.weight")]

                matched = 0
                for down_key in down_keys:
                    up_key = down_key.replace(".lora_down.weight", ".lora_up.weight")
                    if up_key not in sd:
                        continue
                    # module name = key without ".lora_down.weight"
                    module_name = down_key[: -len(".lora_down.weight")]

                    alpha_key = module_name + ".alpha"
                    down_t = sd[down_key].float()
                    alpha = sd[alpha_key].item() if alpha_key in sd else down_t.shape[0]
                    scale = float(lora.strength * (alpha / down_t.shape[0]))

                    entry = (down_t, sd[up_key].float(), scale)
                    lora_map.setdefault(module_name, []).append(entry)
                    matched += 1

                logger.info(
                    "LoRA hook: matched %d layers for %s",
                    matched,
                    Path(lora.path).name,
                )

            # Install forward hooks on matched Linear layers.
            hooked = 0
            for name, module in lookup_module.named_modules():
                entries = lora_map.get(name)
                if not entries or not isinstance(module, torch.nn.Linear):
                    continue

                original_forward = module.forward
                # Capture entries and original_forward in closure.
                def _make_hooked_forward(
                    orig_fwd: Any,
                    _entries: list[tuple[torch.Tensor, torch.Tensor, float]],
                ) -> Any:
                    def hooked_forward(
                        x: torch.Tensor, **kwargs: Any
                    ) -> torch.Tensor:
                        out = orig_fwd(x, **kwargs)
                        for down, up, sc in _entries:
                            d = down.to(device=x.device, dtype=x.dtype)
                            u = up.to(device=x.device, dtype=x.dtype)
                            # LoRA: out += up @ (down @ x^T) scaled
                            out = out + F.linear(F.linear(x, d), u) * sc
                        return out
                    return hooked_forward

                module.forward = _make_hooked_forward(original_forward, entries)
                hooked += 1

            logger.info(
                "LoRA hooks installed on %d Linear layers (%d LoRAs)",
                hooked,
                len(loras),
            )
        except Exception as exc:
            logger.error("LoRA hook install failed: %s", exc, exc_info=True)

    def _load_single(self, entry: LoraEntry) -> LoadedLora | None:
        path = Path(entry.path)
        if not path.exists():
            logger.warning("LoRA file not found: %s", path)
            return None
        if path.suffix.lower() != ".safetensors":
            logger.warning("LoRA file must be .safetensors format: %s", path)
            return None

        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(path), device="cpu")
        except Exception as exc:
            logger.error("Failed to load LoRA %s: %s", path, exc, exc_info=True)
            return None

        fmt = _detect_format(state_dict)
        logger.info("Loaded LoRA %s: %d keys, format=%s", path.name, len(state_dict), fmt)

        if fmt == "civitai":
            state_dict = _remap_civitai_keys(state_dict)
        elif fmt == "diffusers":
            state_dict = _remap_diffusers_keys(state_dict)

        return LoadedLora(
            path=str(path),
            strength=entry.strength,
            state_dict=state_dict,
            format=fmt,
        )


def build_lora_service(device: torch.device) -> LoraService:
    """Factory for LoraService."""
    return LoraService(device=device)
