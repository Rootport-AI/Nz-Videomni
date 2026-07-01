"""Text-only Gemma-3 text-encoder configurator (faithful port of the wheel's
multimodal one, rebuilt around ``Gemma3ForCausalLM`` instead of
``Gemma3ForConditionalGeneration``).

Why this exists
---------------
The wheel's ``GemmaTextEncoderConfigurator.from_config`` builds the FULL multimodal
``Gemma3ForConditionalGeneration`` (vision_tower + multi_modal_projector + the
language model) on the meta device. The LTX text encoder, however, only ever
consumes ``language_model``'s hidden states (``GemmaTextEncoder.precompute`` runs
the model with ``output_hidden_states=True`` and throws the logits away).

Under the QAT ``gemma_root`` reclamation (candidate A) the Gemma weights are no
longer read from ``model*.safetensors`` — they come from the Q4_K_M GGUF, which
carries ONLY the language model. The vision_tower / multi_modal_projector then have
no weights and stay on the meta device. That is fatal for the multimodal build:
``model.device`` is ``next(model.parameters()).device`` (transformers
``get_parameter_device``), and the FIRST parameter of
``Gemma3ForConditionalGeneration`` is ``model.vision_tower...`` (constructed first
inside ``Gemma3Model.__init__``). A meta vision parameter makes ``model.device ==
meta`` -> ``precompute`` builds ``input_ids`` on ``meta`` -> crash.

Building ``Gemma3ForCausalLM`` instead removes vision entirely: its first parameter
is ``model.embed_tokens.weight`` (the language embedding), which always receives a
real weight, so ``model.device`` resolves naturally and the text hidden states are
byte-identical to the multimodal build (same ``Gemma3TextModel`` math, same
weights).

What is a faithful copy vs. what changed
----------------------------------------
Copied verbatim from ``ltx_core.text_encoders.gemma.encoders.encoder_configurator``
(model-class INDEPENDENT — re-imported, not re-implemented):
  * ``_create_feature_extractor`` (V1/V2 selection),
  * the video/audio ``Embeddings1DConnector`` configurators + ``EmbeddingsProcessor``,
  * ``GemmaTextEncoder`` itself (the encode pipeline),
  * the feature_extractor / connector key-op groups.

Changed for the text-only model class (the ONLY deltas):
  1. build ``Gemma3ForCausalLM(text_config)`` instead of
     ``Gemma3ForConditionalGeneration(full_config)``;
  2. the language-model key-op replacement target loses the ``language_model.``
     level: ``language_model.model.`` -> ``model.model.`` (was
     ``model.model.language_model.``), and the embed->lm_head kv-op is keyed on the
     new embed location. The vision_tower / multi_modal_projector key-ops are
     dropped (those modules do not exist here);
  3. ``create_and_populate`` reads ``model.config`` directly (a ``Gemma3TextConfig``;
     ``Gemma3ForCausalLM`` has no ``.text_config``) and drops the vision
     ``position_ids`` registration. The rope ``inv_freq`` + ``embed_scale`` maths are
     kept BYTE-IDENTICAL to the wheel (same head_dim/base/rope_type/hidden_size).

Namespace map (empirically verified, see the session probes)
------------------------------------------------------------
              Gemma3ForConditionalGeneration            Gemma3ForCausalLM (this)
  embed:      model.model.language_model.embed_tokens   model.model.embed_tokens
  layers:     model.model.language_model.layers.N.*     model.model.layers.N.*
  norm:       model.model.language_model.norm.weight    model.model.norm.weight
  lm_head:    model.lm_head.weight                       model.lm_head.weight   (SAME)
The nesting is: GemmaTextEncoder.model (Gemma3ForCausalLM) .model (Gemma3TextModel).
"""

from __future__ import annotations

import torch
from transformers import Gemma3Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from ltx_core.loader import KeyValueOperationResult
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX

# Re-use the wheel's model-class-INDEPENDENT building blocks verbatim (do NOT fork).
from ltx_core.text_encoders.gemma.embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
    _create_feature_extractor,
)


class _ComputeDeviceGemma3ForCausalLM(Gemma3ForCausalLM):
    """``Gemma3ForCausalLM`` whose ``.device`` reports the DECODER (compute) device.

    Why override ``.device``
    ------------------------
    ``GemmaTextEncoder.precompute`` (wheel ``base_encoder.py:57-58``) builds
    ``input_ids`` / ``attention_mask`` on ``self.model.device``, and the prompt-
    enhance path (``base_encoder.py:89,93``) moves inputs / seeds RNG on the same
    device. The stock ``PreTrainedModel.device`` is
    ``next(self.parameters()).device`` (transformers ``get_parameter_device``).

    Under Lever-3 CPU-embed offload the token embedding — the FIRST parameter of a
    text-only ``Gemma3ForCausalLM`` — is held on the CPU, so the stock property
    reports ``cpu``. ``precompute`` then builds ``attention_mask`` on the CPU while
    the decoder runs on the GPU (the CPU-embed forward ships only the hidden states
    to cuda), and the downstream ``feature_extractor`` mixes a cuda ``hidden_states``
    with a cpu ``attention_mask`` -> "Expected all tensors to be on the same device:
    cuda:0 and cpu" at ``feature_extractor.py:78`` (``torch.where``).

    The MULTIMODAL baseline never hit this because its FIRST parameter was
    ``vision_tower.*`` (constructed first in ``Gemma3Model.__init__``), which stayed
    GPU-resident and anchored ``model.device`` to cuda. The CPU-embed offload thus
    implicitly relied on "vision is the device anchor". Removing vision (text-only)
    exposed the CPU embedding.

    Fix (restore the anchor explicitly): report the decoder's final-norm device —
    the SAME device ``_cpu_embed_forward`` uses as its ``compute_dev``
    (``lang.norm.weight.device``, where ``lang`` is this model's ``.model``
    Gemma3TextModel). The final norm is always decoder-resident on the GPU (it stays
    resident even under per-layer offload), so ``model.device`` == compute device ==
    cuda, and ``precompute`` builds input_ids/attention_mask on cuda — exactly the
    device flow the vision-anchored baseline had. The embedding's real device (CPU
    offload) is unchanged; ``_cpu_embed_forward`` still moves input_ids to the CPU
    for the exact-gather lookup (with CPU ``embed_scale``) and returns cuda hidden
    states, so the numerics are byte-identical to the baseline.

    Fallback: if the final norm weight is missing or still on the meta device (only
    possible pre-build; the encode/enhance paths that read ``.device`` always run
    post-build with a GPU-resident norm), defer to the stock parameter-device
    behaviour so nothing regresses off the encode path.
    """

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        norm = getattr(getattr(self, "model", None), "norm", None)
        w = getattr(norm, "weight", None)
        if isinstance(w, torch.Tensor) and w.device.type != "meta":
            return w.device
        # Pre-build / degenerate: fall back to the stock next-parameter device.
        return super().device


class TextOnlyGemmaTextEncoderConfigurator(ModelConfigurator[GemmaTextEncoder]):
    """Faithful text-only twin of the wheel's ``GemmaTextEncoderConfigurator``.

    Identical body to the wheel's ``from_config`` EXCEPT the meta model is a
    ``_ComputeDeviceGemma3ForCausalLM`` (a thin ``Gemma3ForCausalLM`` subclass whose
    ``.device`` reports the decoder/compute device) built from
    ``GEMMA3_CONFIG_FOR_LTX.text_config`` (the same ``Gemma3TextConfig`` the
    multimodal build nests), so no vision modules are constructed. The video/audio
    connectors, embeddings processor and feature extractor are created EXACTLY as the
    wheel does (they are model-class agnostic). The subclass is structurally
    identical to ``Gemma3ForCausalLM`` (same submodules / keys ``model.model.*`` /
    ``model.lm_head.weight``), so the key-ops, loader and module-ops are unaffected —
    only ``.device`` reporting changes (see ``_ComputeDeviceGemma3ForCausalLM``).
    """

    @classmethod
    def from_config(cls, config: dict) -> GemmaTextEncoder:
        transformer_config = config.get("transformer", {})

        # The wheel builds Gemma3Config.from_dict(...).  We take its .text_config —
        # the exact Gemma3TextConfig the multimodal model nests as its language model
        # (verified: hidden_size=3840, num_hidden_layers=48, head_dim=256,
        # vocab_size=262208, rope_scaling={'rope_type':'linear','factor':8.0}).
        gemma_config = Gemma3Config.from_dict(GEMMA3_CONFIG_FOR_LTX.to_dict())
        text_config = gemma_config.text_config
        with torch.device("meta"):
            # Device-anchored subclass: reports the decoder-norm (compute) device from
            # .device so precompute builds input_ids/attention_mask on cuda even when
            # embed_tokens is CPU-offloaded (Lever 3). See the subclass docstring.
            model = _ComputeDeviceGemma3ForCausalLM(text_config)

        # Create video embeddings connector (always needed) — wheel-identical.
        video_connector = Embeddings1DConnectorConfigurator.from_config(config)

        # Create audio embeddings connector — wheel-identical.
        audio_connector = AudioEmbeddings1DConnectorConfigurator.from_config(config)

        # Create embeddings processor with both connectors — wheel-identical.
        embeddings_processor = EmbeddingsProcessor(
            video_connector=video_connector,
            audio_connector=audio_connector,
        )

        feature_extractor = _create_feature_extractor(transformer_config)

        return GemmaTextEncoder(
            feature_extractor=feature_extractor,
            embeddings_processor=embeddings_processor,
            model=model,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Key-ops: faithful port of AV_GEMMA_TEXT_ENCODER_KEY_OPS, text-only namespace.
#
# feature_extractor + connector groups are copied VERBATIM (model-class agnostic).
# The language-model group drops the ``language_model.`` level in its target, and
# the embed->lm_head kv-op is re-keyed to the new embed location. The vision_tower
# and multi_modal_projector groups are removed (no such modules in the text model).
# ──────────────────────────────────────────────────────────────────────────────
TEXT_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS = (
    SDOps("TEXT_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS")
    # 1. Feature extractor (V1: aggregate_embed inside feature_extractor) — VERBATIM.
    .with_matching(prefix="text_embedding_projection.aggregate_embed.")
    .with_replacement("text_embedding_projection.aggregate_embed.", "feature_extractor.aggregate_embed.")
    # V2 dual aggregate embeds — VERBATIM.
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed.")
    .with_matching(prefix="text_embedding_projection.audio_aggregate_embed.")
    .with_replacement("text_embedding_projection.audio_aggregate_embed.", "feature_extractor.audio_aggregate_embed.")
    # 2. Connectors — VERBATIM.
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "embeddings_processor.video_connector.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "embeddings_processor.audio_connector.")
    # 3. Language model layers — TEXT-ONLY target (no ``language_model.`` level).
    #    Wheel:  language_model.model.* -> model.model.language_model.*
    #    Here:   language_model.model.* -> model.model.*
    .with_matching(prefix="language_model.model.")
    .with_replacement("language_model.model.", "model.model.")
    # (vision_tower / multi_modal_projector groups intentionally omitted — the
    #  Gemma3ForCausalLM has no such submodules.)
    # 4. Tie lm_head to embed_tokens — re-keyed to the text-only embed location.
    #    Wheel key_prefix: model.model.language_model.embed_tokens.weight
    #    Here   key_prefix: model.model.embed_tokens.weight ; dup target unchanged
    #    (model.lm_head.weight is identical in both model classes).
    .with_kv_operation(
        operation=lambda key, value: [
            KeyValueOperationResult(key, value),
            KeyValueOperationResult("model.lm_head.weight", value),
        ],
        key_prefix="model.model.embed_tokens.weight",
    )
)


def create_and_populate(module: GemmaTextEncoder) -> GemmaTextEncoder:
    """Text-only twin of the wheel's ``create_and_populate``.

    Registers the rope ``inv_freq`` buffers (global + local) and the embedding
    ``embed_scale`` on the language model. The rope initialisation and embed_scale
    formula are BYTE-IDENTICAL to the wheel; only two things differ:
      * ``config`` is ``model.config`` (a ``Gemma3TextConfig``) directly — the
        text-only model has no ``.text_config`` wrapper the multimodal one has;
      * the vision ``position_ids`` re-registration is dropped (no vision_tower).

    ``l_model`` here is ``Gemma3TextModel`` (``model.model``), the SAME class the
    multimodal path reaches via ``model.model.language_model`` — so the buffers land
    on structurally identical modules (``rotary_emb`` / ``rotary_emb_local`` /
    ``embed_tokens``) with identical values.
    """
    model = module.model  # Gemma3ForCausalLM
    l_model = model.model  # Gemma3TextModel (== the wheel's .model.model.language_model)

    # Gemma3ForCausalLM.config IS a Gemma3TextConfig (no .text_config indirection).
    config = model.config
    dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    base = config.rope_local_base_freq
    local_rope_freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
    inv_freqs, _ = ROPE_INIT_FUNCTIONS[config.rope_scaling["rope_type"]](config)

    embed_scale = torch.tensor(config.hidden_size**0.5, device="cpu")
    l_model.embed_tokens.register_buffer("embed_scale", embed_scale)
    l_model.rotary_emb_local.register_buffer("inv_freq", local_rope_freqs)
    l_model.rotary_emb.register_buffer("inv_freq", inv_freqs)

    return module


# Matcher mirrors the wheel's GEMMA_MODEL_OPS but keys on the text-only class.
# ``isinstance(..., Gemma3ForCausalLM)`` also matches _ComputeDeviceGemma3ForCausalLM
# (a subclass), so keeping the base class here is both correct and robust.
TEXT_ONLY_GEMMA_MODEL_OPS = ModuleOps(
    name="TextOnlyGemmaModel",
    matcher=lambda module: hasattr(module, "model") and isinstance(module.model, Gemma3ForCausalLM),
    mutator=create_and_populate,
)
