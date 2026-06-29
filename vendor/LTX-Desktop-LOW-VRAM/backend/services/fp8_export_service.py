"""Export a pre-quantized FP8 transformer safetensors file.

The pre-quantized file contains the inner LTXModel state dict with fp8_e4m3fn
linear weights and bf16 everything else.  Subsequent pipeline loads read
~2–3 GB instead of streaming transformer keys out of the full 43 GB checkpoint,
and no per-load downcast CPU work is needed.

Key format matches what SingleGPUModelBuilder expects (LTX/post-rename keys,
no ComfyUI prefix) so the builder can be given model_sd_ops=None.

UPCAST_DURING_INFERENCE still needs to be applied at load time (patches
nn.Linear.forward to upcast fp8→bf16 on the fly during inference).
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Fixed filename inside models_dir.
FP8_TRANSFORMER_FILENAME = "fp8_transformer.safetensors"


def fp8_transformer_path(models_dir: Path) -> Path:
    return models_dir / FP8_TRANSFORMER_FILENAME


def fp8_transformer_exists(models_dir: Path) -> bool:
    return fp8_transformer_path(models_dir).exists()


def export_fp8_transformer(
    checkpoint_path: str,
    output_path: str,
    progress_cb: Callable[[float], None] | None = None,
) -> None:
    """Load the transformer with fp8_cast, save the inner LTXModel state dict.

    Args:
        checkpoint_path: Path to the full model checkpoint (.safetensors).
        output_path:     Destination for the pre-quantized fp8 file.
        progress_cb:     Optional callback(fraction: float) — 0.0 … 1.0.
    """
    import safetensors
    from safetensors.torch import save_file

    # ── 1. Copy config metadata from the original checkpoint ─────────────────
    # SingleGPUModelBuilder.model_config() reads {"config": json_str} from the
    # file's safetensors metadata.  Preserving it lets the builder reconstruct
    # the model architecture when loading the pre-quantized file.
    config_json = "{}"
    try:
        with safetensors.safe_open(checkpoint_path, framework="pt") as f:
            meta = f.metadata() or {}
            config_json = meta.get("config", "{}")
    except Exception as exc:
        logger.warning("Could not read metadata from checkpoint: %s", exc)

    if progress_cb:
        progress_cb(0.05)

    # ── 2. Load transformer with fp8_cast to CPU ──────────────────────────────
    # ModelLedger with no gemma_root / upsampler_path builds only the
    # transformer (and VAE/audio) builders — but only the transformer is
    # actually loaded when we call ledger.transformer().
    from ltx_core.quantization import QuantizationPolicy
    from ltx_pipelines.utils.model_ledger import ModelLedger

    logger.info("Loading transformer from %s with fp8_cast (CPU)…", checkpoint_path)
    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        quantization=QuantizationPolicy.fp8_cast(),
    )

    if progress_cb:
        progress_cb(0.10)

    # ledger.transformer() → X0Model(LTXModel(...)) with fp8 linear weights on CPU.
    # This reads transformer-relevant keys from the mmap'd checkpoint and
    # applies: ComfyUI rename + fp8 downcast.  Other model components are NOT
    # loaded (vae_decoder_builder etc. exist but .build() is never called).
    x0_model: torch.nn.Module = ledger.transformer()

    if progress_cb:
        progress_cb(0.75)

    # ── 3. Extract the inner LTXModel state dict ──────────────────────────────
    # X0Model wraps LTXModel as .velocity_model.  The state dict from the inner
    # model has keys like "transformer_blocks.0.to_q.weight" — exactly what
    # SingleGPUModelBuilder expects when model_sd_ops=None.
    inner_model: torch.nn.Module = getattr(x0_model, "velocity_model", x0_model)
    inner_sd: dict[str, torch.Tensor] = inner_model.state_dict()

    logger.info(
        "Transformer state dict: %d keys, fp8 linears included", len(inner_sd)
    )

    if progress_cb:
        progress_cb(0.82)

    # ── 4. Save pre-quantized file ────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(inner_sd, output_path, metadata={"config": config_json})

    logger.info("Saved pre-quantized fp8 transformer to %s", output_path)

    if progress_cb:
        progress_cb(1.0)

    # ── 5. Cleanup ────────────────────────────────────────────────────────────
    del inner_sd, inner_model, x0_model, ledger
    gc.collect()
