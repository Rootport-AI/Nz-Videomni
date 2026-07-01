"""Sequential per-layer CPU offload for the GGUF-quantized Gemma-3 text encoder.

Problem
-------
Even with the Gemma transformer kept *compressed* in VRAM (Q4_K_M GGUF, ~7.3 GB
file) the text-encode forward still has a ~15 GB VRAM peak on the 16 GB card,
because all 48 GGUF-quantized decoder layers are GPU-resident simultaneously
(their dequant intermediates and activations stack up across the stack).

This service applies the SAME sliding-window streaming pattern already proven for
the LTX DiT in ``services.block_swap_service``, but for the Gemma decoder layers:
the 48 quantized decoder layers are kept CPU-resident and streamed to the GPU one
window at a time during the text-encode forward (compute stays on the GPU). Only
``layers_on_gpu`` layers are GPU-resident at any instant, capping the ~15 GB encode
peak to a few GB.

Key difference vs ``BlockSwapService``
--------------------------------------
The Gemma decoder Linear weights are GGUF-quantized and registered as **buffers**
(see ``gguf_quant_service._patch_linear_for_ggml_dequant``), NOT as parameters.
``module.parameters()`` therefore does NOT see them, so we must NOT gate the
device moves on ``layer.parameters()`` device checks the way BlockSwapService does
(that would miss the quantized weights entirely). Instead we call
``layer.to(device)`` / ``layer.to("cpu")`` unconditionally for window members and
evictions: the ``GGMLQuantizedTensor.to()`` override is out-of-place, cheap, and
preserves the dequant metadata (``_ggml_type`` / ``_float_shape``).

Usage
-----
    service = GemmaLayerOffloadService(layers_on_gpu=2, compute_device=dev)
    service.install(inner_gemma_model)   # the module holding language_model.layers
    service.uninstall(inner_gemma_model) # optional, restore original forwards
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_LAYER_OFFLOAD_ATTR = "_gemma_layer_offload_original_forward"


def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in args/kwargs to the target device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        moved = [_move_to_device(x, device) for x in obj]
        return type(obj)(moved)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


class GemmaLayerOffloadService:
    """Installs forward-pass wrappers on the Gemma decoder layers to stream them
    CPU<->GPU one window at a time during the text-encode forward.

    Args:
        layers_on_gpu:  How many decoder layers to keep on the GPU simultaneously
                        (the sliding window size). Compute always runs on the GPU.
        compute_device: The GPU device layers run on during their forward pass.
                        When None, it is inferred at install() time from the
                        model's final language-model norm weight device (matching
                        Lever-3's ``compute_dev = lang.norm.weight.device``).
    """

    def __init__(
        self,
        layers_on_gpu: int = 2,
        compute_device: torch.device | None = None,
    ) -> None:
        self.layers_on_gpu = layers_on_gpu
        self.compute_device = compute_device
        self._installed_models: list[nn.Module] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def install(self, text_encoder_model: nn.Module) -> None:
        """Patch the Gemma decoder layers with sliding-window swap wrappers.

        ``text_encoder_model`` should be the module that holds the decoder
        ``language_model.layers`` ModuleList (the inner Gemma model / a parent of
        it). The layer-container lookup is robust: it tries the documented nesting
        and falls back to the largest ModuleList of decoder layers. No-op (warn +
        return) if the layer container cannot be located.
        """
        if self.layers_on_gpu <= 0:
            logger.info("GemmaLayerOffload disabled (layers_on_gpu=%d)", self.layers_on_gpu)
            return

        layers, lang = self._get_layers(text_encoder_model)
        if not layers:
            logger.warning(
                "GemmaLayerOffload: could not find Gemma decoder layers — skipping "
                "(text encode stays fully GPU-resident)"
            )
            return

        # Resolve the compute device. Prefer the explicit one; else infer from the
        # final language-model norm weight (match Lever-3's compute_dev).
        device = self.compute_device
        if device is None:
            device = self._infer_compute_device(lang, layers)
        if device is None:
            logger.warning(
                "GemmaLayerOffload: could not infer a compute device — skipping"
            )
            return
        self.compute_device = device

        total = len(layers)
        if self.layers_on_gpu >= total:
            logger.info(
                "GemmaLayerOffload: layers_on_gpu=%d >= total=%d — no streaming needed",
                self.layers_on_gpu, total,
            )
            return

        logger.info(
            "GemmaLayerOffload: installing on %d Gemma decoder layers, keeping %d/%d "
            "on %s (streaming the GGUF-quantized layers CPU->GPU per window)",
            total, self.layers_on_gpu, total, device,
        )

        # Move ALL decoder layers to CPU initially. GGML weights are buffers, so a
        # plain layer.to("cpu") moves them (the GGMLQuantizedTensor.to override
        # preserves the quant metadata).
        for layer in layers:
            layer.to("cpu")

        # Patch each layer with a swap-in / swap-out forward wrapper.
        for idx, layer in enumerate(layers):
            self._patch_layer(layer, idx, layers, device)

        # Keep ONLY the current model reference (mirrors BlockSwapService's
        # resident-reuse leak fix: avoid retaining a previous build's layers).
        self._installed_models.clear()
        self._installed_models.append(text_encoder_model)

    def uninstall(self, text_encoder_model: nn.Module) -> None:
        """Remove the swap wrappers and move all decoder layers back to the GPU."""
        layers, _lang = self._get_layers(text_encoder_model)
        if not layers:
            return

        device = self.compute_device or torch.device("cpu")
        for layer in layers:
            orig = getattr(layer, _LAYER_OFFLOAD_ATTR, None)
            if orig is not None:
                layer.forward = orig  # type: ignore[method-assign]
                delattr(layer, _LAYER_OFFLOAD_ATTR)
            layer.to(device)

        if text_encoder_model in self._installed_models:
            self._installed_models.remove(text_encoder_model)

        logger.info("GemmaLayerOffload: uninstalled, all layers moved to %s", device)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_layers(
        self, text_encoder_model: nn.Module
    ) -> tuple[list[nn.Module], nn.Module | None]:
        """Locate the 48-layer Gemma decoder ModuleList (robust lookup).

        Documented nesting (verified against modeling_gemma3 + the GGUF key remap
        ``model.model.language_model.layers.N...``):
            text_encoder.model            (Gemma3ForConditionalGeneration)
              .model                      (Gemma3Model)
                .language_model           (Gemma3TextModel) -> .layers / .norm

        ``text_encoder_model`` may be passed in at any of these levels, so we probe
        a few candidate roots. Returns ``(layers, lang_module)`` where
        ``lang_module`` (the Gemma3TextModel, holding ``.norm``) is used to infer
        the compute device; ``lang_module`` may be None on the fallback path.
        """
        # Build candidate "language model" modules (the thing holding .layers/.norm).
        candidates: list[nn.Module] = []

        def _add(mod: Any) -> None:
            if isinstance(mod, nn.Module) and mod not in candidates:
                candidates.append(mod)

        m = text_encoder_model
        _add(m)
        # text_encoder.model -> Gemma3ForConditionalGeneration
        outer = getattr(m, "model", None)
        _add(outer)
        # .model.model -> Gemma3Model
        inner = getattr(outer, "model", None) if outer is not None else None
        _add(inner)
        # .model.model.language_model -> Gemma3TextModel
        for parent in (m, outer, inner):
            lang = getattr(parent, "language_model", None) if parent is not None else None
            _add(lang)
        # Also a direct .language_model off the inner-most candidate already covered.

        # Documented path: a candidate with a `.layers` ModuleList of decoder layers.
        for cand in candidates:
            layers = getattr(cand, "layers", None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 4:
                return list(layers), cand

        # Fallback: search the whole module tree for the largest ModuleList with
        # >4 children whose container path mentions the language model.
        best: tuple[int, nn.ModuleList, nn.Module | None] | None = None
        for name, child in text_encoder_model.named_modules():
            if isinstance(child, nn.ModuleList) and len(child) > 4:
                if best is None or len(child) > best[0]:
                    # Try to recover the parent (Gemma3TextModel) for norm lookup.
                    parent = self._resolve_parent(text_encoder_model, name)
                    best = (len(child), child, parent)
        if best is not None:
            return list(best[1]), best[2]

        return [], None

    @staticmethod
    def _resolve_parent(root: nn.Module, dotted_name: str) -> nn.Module | None:
        """Resolve the parent module of a dotted submodule name (best-effort)."""
        if "." not in dotted_name:
            return root
        parent_name = dotted_name.rsplit(".", 1)[0]
        mod: Any = root
        for part in parent_name.split("."):
            mod = getattr(mod, part, None)
            if mod is None:
                return None
        return mod if isinstance(mod, nn.Module) else None

    @staticmethod
    def _infer_compute_device(
        lang: nn.Module | None, layers: list[nn.Module]
    ) -> torch.device | None:
        """Infer the GPU compute device from the final norm weight (Lever-3 style),
        falling back to any tensor found among the layers."""
        norm = getattr(lang, "norm", None) if lang is not None else None
        w = getattr(norm, "weight", None) if norm is not None else None
        if isinstance(w, torch.Tensor):
            return w.device
        # Fallback: any parameter or buffer on the layers reveals where compute runs.
        for layer in layers:
            for p in layer.parameters():
                return p.device
            for b in layer.buffers():
                return b.device
        return None

    def _patch_layer(
        self,
        layer: nn.Module,
        idx: int,
        all_layers: list[nn.Module],
        device: torch.device,
    ) -> None:
        """Replace layer.forward with a version that streams neighbours CPU<->GPU."""
        original_forward = layer.forward
        setattr(layer, _LAYER_OFFLOAD_ATTR, original_forward)

        layers_on_gpu = self.layers_on_gpu
        total = len(all_layers)

        def swapped_forward(*args: Any, **kwargs: Any) -> Any:
            # Window: keep layers [idx .. idx+layers_on_gpu-1] on the GPU.
            window_start = idx
            window_end = min(idx + layers_on_gpu, total)

            # Ensure every layer in the current window is on the GPU. GGML weights
            # are BUFFERS (not parameters), so we do NOT gate on .parameters()
            # device checks — we call .to(device) unconditionally. The
            # GGMLQuantizedTensor.to override is out-of-place + cheap and preserves
            # the dequant metadata.
            for load_idx in range(window_start, window_end):
                all_layers[load_idx].to(device)

            # Evict the layer that just left the window (idx - 1); it has already
            # finished its forward pass.
            evict_idx = idx - 1
            if evict_idx >= 0:
                all_layers[evict_idx].to("cpu")

            # Move input tensors to the GPU before calling forward: layer.to(device)
            # moves the weights, but args/kwargs still hold tensors on CPU (output of
            # the previous evicted layer). PyTorch dispatches ops to the device of
            # the INPUT tensors, not the module, so without this the forward would
            # run on CPU.
            args = _move_to_device(args, device)
            kwargs = _move_to_device(kwargs, device)

            return original_forward(*args, **kwargs)

        layer.forward = swapped_forward  # type: ignore[method-assign]


def build_gemma_layer_offload_service(
    layers_on_gpu: int,
    compute_device: torch.device | None = None,
) -> GemmaLayerOffloadService | None:
    """Factory: returns None when offloading is disabled (layers_on_gpu<=0)."""
    if layers_on_gpu <= 0:
        return None
    return GemmaLayerOffloadService(
        layers_on_gpu=layers_on_gpu, compute_device=compute_device
    )
