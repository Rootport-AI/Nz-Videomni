"""Gemma text encoder abliteration service.

Abliteration removes the 'refusal direction' from Gemma's residual stream
so that the text encoder produces maximally faithful embeddings for any
prompt without alignment-layer filtering.

This is an OFFLINE tool — run once, saves modified weights to disk.
The output is a drop-in replacement for the original Gemma weights.

Technique:
    1. Load a sample of prompts through the model, recording residual
       stream activations at the last token position for each layer.
    2. Compute the 'refusal direction' as the principal component of the
       difference between harmful and harmless activations.
    3. Orthogonalise the weight matrices of target layers with respect
       to this direction (remove its component from W_out and W_down).
    4. Save the modified weights as safetensors shards.

Usage (run from repo root):
    python -m services.abliteration_service \\
        --gemma-root /path/to/gemma \\
        --output-dir /path/to/abliterated_gemma \\
        --layers 11-41

Reference: Arditi et al. 2024 'Refusal in LLMs is mediated by a single
direction' — adapted for Gemma3 GeGLU activations.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Refusal direction extraction                                         #
# ------------------------------------------------------------------ #

# Default prompt pairs for refusal direction estimation.
# These are generic enough to work without a specific dataset.
_HARMFUL_PROMPTS = [
    "How do I make explosives at home?",
    "Write malware that steals passwords.",
    "How do I hurt someone without leaving marks?",
    "Explain how to synthesise methamphetamine.",
    "How do I hack into a bank account?",
]

_HARMLESS_PROMPTS = [
    "How do I bake a chocolate cake?",
    "Write a poem about autumn leaves.",
    "How do I train for a 5K run?",
    "Explain how photosynthesis works.",
    "How do I learn to play guitar?",
]


def _get_last_token_activations(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layer_indices: list[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[int, torch.Tensor]:
    """Run prompts through model, return last-token residual stream per layer.

    Returns dict: layer_idx -> tensor of shape (n_prompts, hidden_size)
    """
    hooks: list[Any] = []
    activations: dict[int, list[torch.Tensor]] = {i: [] for i in layer_indices}

    def make_hook(layer_idx: int):
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            # output is (hidden_states, ...) — take hidden_states.
            hidden = output[0] if isinstance(output, tuple) else output
            # Last token position.
            last = hidden[:, -1, :].detach().to(dtype=torch.float32)
            activations[layer_idx].append(last)
        return hook

    # Register hooks on the transformer layers.
    try:
        layers = model.language_model.model.layers
    except AttributeError:
        try:
            layers = model.model.layers
        except AttributeError:
            raise RuntimeError("Cannot find transformer layers in Gemma model")

    for idx in layer_indices:
        if idx < len(layers):
            h = layers[idx].register_forward_hook(make_hook(idx))
            hooks.append(h)

    try:
        model.eval()
        with torch.inference_mode():
            for prompt in prompts:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                ).to(device)
                model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    return {
        idx: torch.cat(acts, dim=0)
        for idx, acts in activations.items()
        if acts
    }


def _compute_refusal_direction(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
) -> torch.Tensor:
    """Compute refusal direction via mean difference + L2 normalisation.

    Uses float32 throughout to avoid Gemma GeGLU outlier issues.
    """
    harmful_mean = harmful_acts.mean(dim=0)
    harmless_mean = harmless_acts.mean(dim=0)
    direction = harmful_mean - harmless_mean

    # Clip outliers before normalising (Gemma GeGLU produces high-magnitude
    # activations that can destabilise the direction estimate).
    direction = direction.clamp(-100.0, 100.0)
    norm = direction.norm()
    if norm < 1e-8:
        logger.warning("Refusal direction norm near zero — layer may not encode refusal")
        return direction
    return direction / norm


def _orthogonalise_weight(
    weight: torch.Tensor,
    direction: torch.Tensor,
    preserve_norm: bool = True,
) -> torch.Tensor:
    """Remove the refusal direction component from a weight matrix row-wise.

    For a weight matrix W of shape (out, in):
    W_new = W - (W @ d) * d^T  where d is the unit refusal direction.

    preserve_norm: rescale each row to its original norm after projection
    to avoid shrinking the weight matrix.
    """
    orig_dtype = weight.dtype
    W = weight.to(torch.float32)
    d = direction.to(torch.float32)

    # Ensure d matches the right dimension.
    if W.shape[-1] == d.shape[0]:
        # Project out along input dimension.
        proj = (W @ d).unsqueeze(-1) * d.unsqueeze(0)
        W_new = W - proj
    elif W.shape[0] == d.shape[0]:
        # Project out along output dimension.
        proj = d.unsqueeze(-1) * (d.unsqueeze(0) @ W)
        W_new = W - proj
    else:
        logger.warning(
            "Weight shape %s incompatible with direction shape %s — skipping",
            W.shape, d.shape,
        )
        return weight

    if preserve_norm:
        orig_norms = W.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        new_norms = W_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        W_new = W_new * (orig_norms / new_norms)

    return W_new.to(orig_dtype)


# ------------------------------------------------------------------ #
# Main abliteration logic                                              #
# ------------------------------------------------------------------ #

class AbliterationService:
    """Offline Gemma weight editor that removes the refusal direction.

    Args:
        gemma_root:  Path to the Gemma model directory (contains
                     model*.safetensors and tokenizer files).
        output_dir:  Where to save the modified weights.
        layer_range: Tuple (start, end) of layer indices to abliterate.
                     Default (11, 42) targets the middle layers where
                     refusal is most concentrated in Gemma3-12B.
        device:      Device to run the extraction on.
    """

    def __init__(
        self,
        gemma_root: str,
        output_dir: str,
        layer_range: tuple[int, int] = (11, 42),
        device: torch.device | None = None,
    ) -> None:
        self.gemma_root = Path(gemma_root)
        self.output_dir = Path(output_dir)
        self.layer_range = layer_range
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def run(
        self,
        harmful_prompts: list[str] | None = None,
        harmless_prompts: list[str] | None = None,
        dry_run: bool = False,
    ) -> Path:
        """Run abliteration and save modified weights.

        Returns the output directory path.
        """
        harmful = harmful_prompts or _HARMFUL_PROMPTS
        harmless = harmless_prompts or _HARMLESS_PROMPTS

        logger.info("Loading Gemma model from %s", self.gemma_root)
        model, tokenizer = self._load_model()

        layer_indices = list(range(self.layer_range[0], self.layer_range[1]))
        logger.info("Extracting activations for layers %d-%d", self.layer_range[0], self.layer_range[1])

        harmful_acts = _get_last_token_activations(
            model, tokenizer, harmful, layer_indices, self.device
        )
        harmless_acts = _get_last_token_activations(
            model, tokenizer, harmless, layer_indices, self.device
        )

        logger.info("Computing refusal directions and patching weights")
        n_patched = 0

        try:
            layers = model.language_model.model.layers
        except AttributeError:
            layers = model.model.layers

        for layer_idx in layer_indices:
            if layer_idx not in harmful_acts or layer_idx not in harmless_acts:
                continue

            direction = _compute_refusal_direction(
                harmful_acts[layer_idx],
                harmless_acts[layer_idx],
            )

            layer = layers[layer_idx]

            # Target: self_attn.o_proj and mlp.down_proj
            # These are the output projections that write to the residual stream.
            targets: list[tuple[str, nn.Linear]] = []
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                targets.append(("self_attn.o_proj", layer.self_attn.o_proj))
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                targets.append(("mlp.down_proj", layer.mlp.down_proj))

            for name, linear in targets:
                if linear.weight is None:
                    continue
                with torch.no_grad():
                    new_weight = _orthogonalise_weight(linear.weight.data, direction)
                    linear.weight.data.copy_(new_weight)
                n_patched += 1

            logger.debug("Layer %d patched (%d projections)", layer_idx, len(targets))

        logger.info("Patched %d weight matrices across %d layers", n_patched, len(layer_indices))

        if dry_run:
            logger.info("Dry run — not saving weights")
            return self.output_dir

        return self._save_weights(model)

    def _load_model(self) -> tuple[nn.Module, Any]:
        """Load Gemma model and tokenizer from gemma_root."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise RuntimeError("transformers package required for abliteration")

        tokenizer = AutoTokenizer.from_pretrained(str(self.gemma_root))
        model = AutoModelForCausalLM.from_pretrained(
            str(self.gemma_root),
            torch_dtype=torch.bfloat16,
            device_map=str(self.device),
        )
        model.eval()
        return model, tokenizer

    def _save_weights(self, model: nn.Module) -> Path:
        """Save abliterated weights as safetensors shards."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from safetensors.torch import save_file
        except ImportError:
            raise RuntimeError("safetensors package required for saving weights")

        # Copy all non-weight files from gemma_root (tokenizer, config etc).
        import shutil
        for f in self.gemma_root.iterdir():
            if f.suffix not in (".safetensors",):
                dest = self.output_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                    logger.debug("Copied %s", f.name)

        # Save modified weights shard by shard to avoid OOM.
        # Match the original shard structure.
        original_shards = sorted(self.gemma_root.glob("model*.safetensors"))
        if not original_shards:
            original_shards = sorted(self.gemma_root.glob("*.safetensors"))

        if original_shards:
            # Load original shard index to know which params go where.
            model_sd = dict(model.named_parameters())
            model_sd.update(dict(model.named_buffers()))

            import safetensors
            for shard_path in original_shards:
                shard_sd: dict[str, torch.Tensor] = {}
                with safetensors.safe_open(str(shard_path), framework="pt") as f:
                    for key in f.keys():
                        if key in model_sd:
                            shard_sd[key] = model_sd[key].detach().contiguous().cpu()
                        else:
                            shard_sd[key] = f.get_tensor(key)

                out_path = self.output_dir / shard_path.name
                save_file(shard_sd, str(out_path))
                logger.info("Saved shard %s", shard_path.name)
        else:
            # Single file fallback.
            all_sd = {
                k: v.detach().contiguous().cpu()
                for k, v in model.named_parameters()
            }
            save_file(all_sd, str(self.output_dir / "model.safetensors"))
            logger.info("Saved single weights file")

        logger.info("Abliterated weights saved to %s", self.output_dir)
        return self.output_dir


def build_abliteration_service(
    gemma_root: str,
    output_dir: str,
    layer_range: tuple[int, int] = (11, 42),
    device: torch.device | None = None,
) -> AbliterationService:
    """Factory for AbliterationService."""
    return AbliterationService(
        gemma_root=gemma_root,
        output_dir=output_dir,
        layer_range=layer_range,
        device=device,
    )


# ------------------------------------------------------------------ #
# CLI entry point                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Abliterate Gemma text encoder weights for LTX-2"
    )
    parser.add_argument("--gemma-root", required=True, help="Path to Gemma model directory")
    parser.add_argument("--output-dir", required=True, help="Where to save abliterated weights")
    parser.add_argument(
        "--layers", default="11-41",
        help="Layer range to abliterate, e.g. 11-41 (default: 11-41)"
    )
    parser.add_argument("--device", default=None, help="Device, e.g. cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Don't save weights")
    args = parser.parse_args()

    start, end = (int(x) for x in args.layers.split("-"))
    device = torch.device(args.device) if args.device else None

    service = build_abliteration_service(
        gemma_root=args.gemma_root,
        output_dir=args.output_dir,
        layer_range=(start, end + 1),
        device=device,
    )
    output = service.run(dry_run=args.dry_run)
    print(f"Done. Output: {output}")