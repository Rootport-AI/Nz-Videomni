"""Shared IC-LoRA plumbing for both fuse paths.

Two engine paths consume IC-LoRA adapters:

  * bf16 full-dequant path (``GGUFStateDictLoader._fuse_ic_loras``) — fuses the
    delta in-place into the full BF16 state-dict at load time (Phase A spike).
  * per-layer-quant path (``quant_service.ggml_linear_forward``) — keeps the
    weights compressed in VRAM and adds the delta AT FORWARD TIME onto the
    fresh per-call dequant tensor (Phase B).

Both need the SAME front half: load the LoRA safetensors through the wheel's
``SafetensorsStateDictLoader`` with ``LTXV_LORA_COMFY_RENAMING_MAP`` (which strips
the ``diffusion_model.`` prefix so LoRA keys align with the raw GGUF/model keys),
then pair ``<prefix>.lora_A.weight`` / ``<prefix>.lora_B.weight`` and shape-check.
That reusable front half lives here in :func:`load_ic_lora_pairs`.

The forward-time path additionally needs to attach the A/B factors to the target
``nn.Linear`` instances so they ride ``block.to(device)`` during block-swap — see
:func:`attach_ic_loras` / :func:`detach_ic_loras`.

DELTA FORMULA (kept byte-identical across both paths, per Phase B gate G2):
    delta = torch.matmul(B.float() * strength, A.float())          # fp32 matmul
    weight_bf16 += delta.to(weight_bf16.dtype)                     # single cast, in-place add
The forward-time path recomputes this every call onto the transient dequant
tensor; the load-time path does it once onto the persisted BF16 weight. Same
math, same operation order — do not "optimise" it (a bf16 intermediate or a
different strength fold would break G2).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Attribute holding the per-Linear list of (a_buf_name, b_buf_name, strength)
# tuples. Absent → the forward path is byte-identical to the no-LoRA path.
IC_LORA_SPECS_ATTR = "_ic_lora_specs"

_SUFFIX_A = ".lora_A.weight"
_SUFFIX_B = ".lora_B.weight"

# One configured LoRA: ``audio_strength`` None → the audio side follows
# ``strength`` (historical behaviour, byte-identical delta path).
IcLoraEntry = tuple[str, float, "float | None"]

# Cross-attention blocks are named after their SOURCE stream but write into the
# other one, so their axis is the opposite of what the name suggests; the gates
# and scale-shift tables of the AV cross-attention follow the stream they
# modulate. These exact-name specials are evaluated BEFORE the ``audio_`` prefix
# rule (``audio_to_video_attn`` would otherwise be misread as audio).
_AUDIO_AXIS_EXACT = frozenset({
    "video_to_audio_attn",
    "av_ca_v2a_gate_adaln_single",
    "av_ca_audio_scale_shift_adaln_single",
})
_VIDEO_AXIS_EXACT = frozenset({
    "audio_to_video_attn",
    "av_ca_a2v_gate_adaln_single",
    "av_ca_video_scale_shift_adaln_single",
})

# LTXModel top-level submodules that legitimately fall through to the video
# axis; anything else reaching that fallback is unrecognised (DEBUG trace).
_KNOWN_VIDEO_TOP_LEVEL = frozenset({
    "transformer_blocks",
    "patchify_proj",
    "proj_out",
    "norm_out",
    "scale_shift_table",
    "caption_projection",
    "adaln_single",
    "prompt_adaln_single",
    "video_args_preprocessor",
})


def classify_lora_axis(module_prefix: str) -> str:
    """Return ``"audio"`` or ``"video"`` for a renamed LoRA module prefix.

    The prefix is split on dots and matched component-wise: exact-name specials
    first (see above), then a plain ``audio_`` prefix, else video. Evaluation
    order is part of the contract — do not reorder.
    """
    parts = module_prefix.split(".")
    for part in parts:
        if part in _AUDIO_AXIS_EXACT:
            return "audio"
        if part in _VIDEO_AXIS_EXACT:
            return "video"
    for part in parts:
        if part.startswith("audio_"):
            return "audio"
    if parts[0] not in _KNOWN_VIDEO_TOP_LEVEL:
        logger.debug(
            "IC-LoRA axis: unknown module prefix %s — treating as video", module_prefix
        )
    return "video"


def strength_for_prefix(
    module_prefix: str, strength: float, audio_strength: float | None
) -> float:
    """Pick the per-axis strength for one LoRA key.

    ``audio_strength is None`` returns ``strength`` untouched WITHOUT classifying
    (G-BC: the no-audio_strength path must stay identical to the historical one).
    """
    if audio_strength is None:
        return strength
    return audio_strength if classify_lora_axis(module_prefix) == "audio" else strength


def normalize_ic_loras(
    ic_loras: list[tuple[str, float]] | list[IcLoraEntry],
) -> list[IcLoraEntry]:
    """Accept legacy 2-tuples and 3-tuples, return 3-tuples with float fields."""
    out: list[IcLoraEntry] = []
    for entry in ic_loras:
        if len(entry) == 2:
            path, strength = entry
            audio_strength = None
        elif len(entry) == 3:
            path, strength, audio_strength = entry  # type: ignore[misc]
        else:
            raise ValueError(
                "IC-LoRA entry must be (path, strength[, audio_strength]), "
                f"got {entry!r}"
            )
        out.append(
            (
                str(path),
                float(strength),
                None if audio_strength is None else float(audio_strength),
            )
        )
    return out


def load_ic_lora_pairs(
    ic_loras: list[tuple[str, float]] | list[IcLoraEntry],
) -> list[tuple[str, float, float | None, list[tuple[str, torch.Tensor, torch.Tensor]]]]:
    """Load + rename + pair each configured IC-LoRA safetensors.

    Returns a list (one entry per configured LoRA, in order) of
    ``(path, strength, audio_strength, pairs)`` where ``pairs`` is a list of
    ``(module_prefix, lora_A, lora_B)`` in the LoRA file's own key order.

    ``module_prefix`` is the ``diffusion_model.``-stripped key prefix, which is
    exactly the dotted ``named_modules()`` name of the target Linear in the LTX
    transformer (verified in Phase A: the raw GGUF/model keys and the renamed
    LoRA keys agree exactly). ``lora_A``/``lora_B`` are returned in their native
    dtype (bf16) on CPU; callers cast to fp32 at delta time.

    Raises on a missing file or a lora_A present with no matching lora_B
    (corrupt/unsupported layout). Empty ``ic_loras`` → empty list.
    """
    from ltx_core.loader import (
        LTXV_LORA_COMFY_RENAMING_MAP,
        SafetensorsStateDictLoader,
    )

    loader = SafetensorsStateDictLoader()
    cpu = torch.device("cpu")
    out: list[
        tuple[str, float, float | None, list[tuple[str, torch.Tensor, torch.Tensor]]]
    ] = []

    for path, strength, audio_strength in normalize_ic_loras(ic_loras):
        if not Path(path).exists():
            raise FileNotFoundError(f"IC-LoRA safetensors not found: {path}")
        lora_sd = loader.load(path, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP, device=cpu)
        pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for key_a in lora_sd.sd:
            if not key_a.endswith(_SUFFIX_A):
                continue
            prefix = key_a[: -len(_SUFFIX_A)]
            key_b = prefix + _SUFFIX_B
            lora_a = lora_sd.sd.get(key_a)
            lora_b = lora_sd.sd.get(key_b)
            if lora_b is None:
                raise RuntimeError(
                    f"IC-LoRA {Path(path).name}: {key_a} present but {key_b} "
                    "missing — corrupt/unsupported LoRA layout"
                )
            pairs.append((prefix, lora_a, lora_b))
        out.append((path, strength, audio_strength, pairs))
    return out


def _find_target_model(transformer: nn.Module) -> nn.Module:
    """Return the LTXModel whose Linears carry the per-layer dequant forward.

    ``model_ledger.transformer()`` returns an X0Model that wraps the real
    LTXModel in ``self.velocity_model`` — its ``named_modules()`` would prefix
    every name with ``velocity_model.`` and break the prefix↔module match. The
    module_ops mutator patches ``LTXModel`` instances, so we locate that node and
    resolve names relative to it (matching the renamed LoRA prefixes exactly).
    """
    try:
        from ltx_core.model.transformer.model import LTXModel
    except Exception:  # pragma: no cover - engine venv always has it
        LTXModel = None  # type: ignore[assignment]

    if LTXModel is not None:
        for module in transformer.modules():
            if isinstance(module, LTXModel):
                return module
    # Fallback: X0Model.velocity_model, else the passed module itself.
    inner = getattr(transformer, "velocity_model", None)
    if isinstance(inner, nn.Module):
        return inner
    return transformer


def attach_ic_loras(
    transformer: nn.Module, ic_loras: list[tuple[str, float]] | list[IcLoraEntry]
) -> int:
    """Attach IC-LoRA A/B factors as buffers onto their target Linear modules.

    Resolves each renamed LoRA prefix to the correspondingly-named ``nn.Linear``
    in the LTXModel and registers ``lora_A``/``lora_B`` as **non-persistent
    buffers** (persistent=False → excluded from state_dict, but moved by
    ``module.to(device)`` so they ride block-swap automatically). The forward
    patch reads them via :data:`IC_LORA_SPECS_ATTR`.

    Multiple LoRAs hitting the same Linear accumulate as successive spec entries
    (the forward adds their deltas sequentially — same net result as the wheel's
    summed-deltas path). Returns the total number of (LoRA, Linear) matches.

    A per-entry ``audio_strength`` applies to the audio-axis Linears only (see
    :func:`classify_lora_axis`); a resolved Linear whose effective strength is
    0.0 gets NO buffers and NO spec (mathematically a zero delta, but cheaper in
    VRAM and time) and is reported as muted.

    Fail-loud contract: a shape mismatch raises; 0 RESOLVED prefixes WARN loudly
    (key-format mismatch → the adapter would be a silent no-op). Full mutes are
    resolved, so they do not trip that warning.
    """
    if not ic_loras:
        return 0

    detach_ic_loras(transformer)  # defensive: never stack onto stale specs
    root = _find_target_model(transformer)
    modules_by_name = dict(root.named_modules())

    per_lora = load_ic_lora_pairs(ic_loras)
    total_matches = 0

    for path, strength, audio_strength, pairs in per_lora:
        n_match = 0
        n_resolved = 0
        for prefix, lora_a, lora_b in pairs:
            module = modules_by_name.get(prefix)
            if not isinstance(module, nn.Linear):
                # Prefix has no Linear counterpart (e.g. non-target key). Skip —
                # counted via the 0-resolved WARN below, mirroring the bf16 path.
                continue
            n_resolved += 1
            out_features = module.out_features
            in_features = module.in_features
            # A:(rank,in)  B:(out,rank)  ->  delta = B@A : (out,in) == weight
            if (
                lora_b.dim() != 2
                or lora_a.dim() != 2
                or lora_b.shape[0] != out_features
                or lora_a.shape[1] != in_features
                or lora_b.shape[1] != lora_a.shape[0]
            ):
                raise RuntimeError(
                    f"IC-LoRA {Path(path).name}: {prefix} shape mismatch — "
                    f"A={tuple(lora_a.shape)} B={tuple(lora_b.shape)} vs "
                    f"Linear(out={out_features}, in={in_features})"
                )

            # Shape-checked first (fail-loud stays independent of muting), then
            # muted keys leave the module byte-identical to the no-LoRA state.
            eff = strength_for_prefix(prefix, strength, audio_strength)
            if eff == 0.0:
                continue

            specs = getattr(module, IC_LORA_SPECS_ATTR, None)
            if specs is None:
                specs = []
                setattr(module, IC_LORA_SPECS_ATTR, specs)
            slot = len(specs)
            a_name = f"_ic_lora_A_{slot}"
            b_name = f"_ic_lora_B_{slot}"
            # Keep native dtype (bf16); persistent=False so state_dict is
            # untouched but .to(device) still moves them with the block.
            module.register_buffer(a_name, lora_a.contiguous(), persistent=False)
            module.register_buffer(b_name, lora_b.contiguous(), persistent=False)
            specs.append((a_name, b_name, eff))
            n_match += 1

        total_matches += n_match
        extra = ""
        if audio_strength is not None:
            extra = (
                f", audio_strength={audio_strength:.3f}, "
                f"muted={n_resolved - n_match} linears"
            )
        logger.info(
            "IC-LoRA %s: %d Linear(s) attached for forward-time apply (strength=%.3f%s)",
            Path(path).name, n_match, strength, extra,
        )
        if n_resolved == 0:
            logger.warning(
                "IC-LoRA %s matched 0 Linear modules — LoRA/model KEY-FORMAT "
                "MISMATCH; the adapter would be a silent no-op. Check the "
                "renaming map / module tree.",
                Path(path).name,
            )

    logger.info(
        "IC-LoRA forward-time attach complete: %d LoRA(s), %d total Linear matches",
        len(per_lora), total_matches,
    )
    return total_matches


def detach_ic_loras(transformer: nn.Module) -> int:
    """Remove all attached IC-LoRA buffers + specs from every Linear.

    Restores the module tree to its byte-identical no-LoRA state (used for the
    keep-resident toggle path and defensively before every attach). Returns the
    number of Linear modules cleared.
    """
    root = _find_target_model(transformer)
    cleared = 0
    for module in root.modules():
        specs = getattr(module, IC_LORA_SPECS_ATTR, None)
        if not specs:
            continue
        for a_name, b_name, _strength in specs:
            module._buffers.pop(a_name, None)
            module._buffers.pop(b_name, None)
        try:
            delattr(module, IC_LORA_SPECS_ATTR)
        except AttributeError:
            pass
        cleared += 1
    return cleared
