"""Per-layer GGUF quantization service for the Gemma-3 text encoder.

Problem
-------
The stock LTX text encoder loads Gemma-3-12b in bf16 directly onto the GPU:

    ModelLedger.text_encoder() ->
        text_encoder_builder.build(device=cuda, dtype=bf16).to(cuda)

The Gemma transformer alone is ~24 GB in bf16, which overflows our 16 GB card
(and forces a ~17 GB WDDM spillover into shared system RAM, making text encoding
crawl). `_install_cpu_text_encoder` sidesteps this by running Gemma on the CPU,
but that is slow.

This service instead keeps the Gemma transformer weights *compressed* in VRAM
(loaded from a single Q4_K_M GGUF, ~7.3 GB) and dequantizes one Linear layer at a
time during the forward pass — the exact mechanism already proven for the LTX
transformer in `services.gguf_quant_service`, applied here to Gemma's nn.Linears.

VRAM comparison for Gemma-3-12b:
  bf16 (2 bytes/param):  ~24 GB VRAM
  Q4_K_M (~0.56 bytes):  ~7.3 GB VRAM (file size)
At inference: +1 layer bf16 (~tens of MB peak overhead, freed after each matmul).

We REPLICATE the proven ComfyUI approach (city96/ComfyUI-GGUF) for the GGUF->HF
gemma3 key map and the RMSNorm "+1" correction; we do NOT invent quantization
math (that is reused verbatim from `services.gguf_quant_service`).

Key differences vs the transformer GGUF service
-----------------------------------------------
1. KEY REMAP IS MANDATORY. The transformer service notes its GGUF raw keys match
   the model keys exactly (gguf_quant_service.py ~L587). That is FALSE for Gemma:
   GGUF uses llama.cpp names (``blk.N.attn_q.weight`` ...) that must be remapped
   to HF names (``model.layers.N.self_attn.q_proj.weight`` ...) and then prefixed
   to the LTX-nested location (``model.model.language_model.layers.N....``).

2. MERGE, NOT REPLACE-ALL. The text encoder's state_dict mixes Gemma transformer
   weights (from the GGUF) with LTX-side weights that are NOT in the GGUF:
     - ``feature_extractor.video_aggregate_embed.{weight,bias}``
     - ``feature_extractor.audio_aggregate_embed.{weight,bias}``
     - ``embeddings_processor.{video,audio}_connector.*``
   These come from the distilled LTX checkpoint (bf16). We load everything from
   safetensors as usual, then OVERLAY the Gemma language-model weights from the
   GGUF, dropping the original bf16 Gemma keys.

3. RMSNorm "+1" CORRECTION. llama.cpp bakes (1 + w) into the stored norm weight,
   but HF's Gemma3RMSNorm computes ``output * (1.0 + self.weight.float())`` itself
   (confirmed in transformers/models/gemma3/modeling_gemma3.py). So every norm
   weight from the GGUF must have 1.0 SUBTRACTED after dequant.

4. EMBEDDINGS / NORMS STAY DEQUANTIZED. The per-layer dequant patch only fires on
   nn.Linear.forward. ``embed_tokens`` is an nn.Embedding and the norms are
   nn.Parameters consumed directly, so they must be plain bf16 tensors (also
   avoids a 262k-vocab quant buffer). Only the Gemma q/k/v/o_proj and
   gate/up/down_proj Linears are wrapped as GGMLQuantizedTensor.

5. bf16 EVERYWHERE. Gemma-3 overflows fp16 (-> NaN / empty output); never fp16.

6. NO llama_permute. gemma3 GGUF Q/K projections are already in HF layout.
"""

from __future__ import annotations

import logging
import types
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import torch

# Reuse the bit-exact-verified quant kernels + per-layer Linear forward machinery
# from the transformer service. Do NOT duplicate the dequant math.
from engine.gguf.quant_service import (
    GGMLQuantizedTensor,
    _GGML_BF16,
    _GGML_F16,
    _GGML_F32,
    dequantize_ggml_tensor,
)

logger = logging.getLogger(__name__)

_FLOAT_GGML_TYPES = {_GGML_F32, _GGML_F16, _GGML_BF16}

# ──────────────────────────────────────────────────────────────────────────────
# GGUF -> HF gemma3 key map (copied verbatim from city96/ComfyUI-GGUF loader.py)
# ──────────────────────────────────────────────────────────────────────────────

LLAMA_SD_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_v_norm.": "self_attn.v_norm.",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}

GEMMA3_SD_MAP = LLAMA_SD_MAP.copy()
GEMMA3_SD_MAP.update(
    {
        "ffn_norm": "pre_feedforward_layernorm",
        "post_ffw_norm": "post_feedforward_layernorm",
        "post_attention_norm": "post_attention_layernorm",
    }
)


def sd_map_replace(raw_sd: dict[str, Any], key_map: dict[str, str]) -> dict[str, Any]:
    """Apply substring replacements to every key (city96/ComfyUI-GGUF verbatim)."""
    sd: dict[str, Any] = {}
    for k, v in raw_sd.items():
        for s, d in key_map.items():
            k = k.replace(s, d)
        sd[k] = v
    return sd


# RMSNorm weights that need the llama.cpp (1 + w) -> w correction (city96 verbatim).
# These are matched as suffixes AFTER the gemma3 remap (HF names).
_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "model.norm.weight",
)


# ──────────────────────────────────────────────────────────────────────────────
# LTX prefix adaptation
#
# After the city96 gemma3 remap, keys are plain HF Gemma3 names, e.g.
#   model.layers.0.self_attn.q_proj.weight
#   model.embed_tokens.weight
#   model.norm.weight
#   lm_head.weight
#
# In the LTX stack the GemmaTextEncoder nests the HF model as
#   GemmaTextEncoder.model            (Gemma3ForConditionalGeneration)
#     .model                          (Gemma3Model)
#       .language_model               (Gemma3TextModel)  -> layers / embed_tokens / norm
#   GemmaTextEncoder.model.lm_head    (nn.Linear, tied to embed_tokens)
#
# This matches ltx_core's AV_GEMMA_TEXT_ENCODER_KEY_OPS, which maps the original
# safetensors keys via:
#   language_model.model.* -> model.model.language_model.*
# and duplicates embed_tokens.weight onto model.lm_head.weight.
# Confirmed in:
#   ltx_core/text_encoders/gemma/encoders/encoder_configurator.py
#   transformers/models/gemma3/modeling_gemma3.py (module attribute names)
# ──────────────────────────────────────────────────────────────────────────────

# Where the Gemma3TextModel (layers/embed_tokens/norm) lives inside GemmaTextEncoder.
_LTX_LM_PREFIX = "model.model.language_model."
# Where lm_head lives inside GemmaTextEncoder (Gemma3ForConditionalGeneration.lm_head).
_LTX_LM_HEAD_KEY = "model.lm_head.weight"

# ── Commit-reduction (Stage 1): skip the bf16 Gemma LM at READ time ───────────
# The base safetensors load used to materialize the FULL ~24 GB bf16 Gemma
# language model into committed CPU RAM (one get_tensor copy per tensor) and then
# DELETE every ``model.model.language_model.*`` key (plus the tied
# ``model.lm_head.weight``) immediately afterward, because the GGUF supplies the
# compressed replacement. Those copies are pure waste (committed, then freed).
#
# Both deleted remapped key sets originate, via AV_GEMMA_TEXT_ENCODER_KEY_OPS,
# EXCLUSIVELY from original safetensors keys under this prefix:
#   AV ops:  "language_model.model." -> "model.model.language_model."   (-> _LTX_LM_PREFIX)
#   AV kv-op: language_model.model.embed_tokens.weight ALSO duplicated to
#             "model.lm_head.weight" (= _LTX_LM_HEAD_KEY)
# No other original prefix maps into _LTX_LM_PREFIX or _LTX_LM_HEAD_KEY, and this
# prefix maps into nothing else. Therefore skipping every ORIGINAL key under this
# prefix at read time yields a base state_dict byte-identical to the prior
# read-then-delete result (the survivors — feature_extractor / connectors /
# vision_tower / multi_modal_projector — come from other prefixes and are kept).
_ORIG_GEMMA_LM_PREFIX = "language_model.model."


class _SkipGemmaLMSDOps:
    """Duck-typed SDOps proxy that drops the bf16 Gemma LM keys at READ time.

    Wraps the real key-ops (e.g. AV_GEMMA_TEXT_ENCODER_KEY_OPS) and delegates
    everything, except it returns ``None`` from ``apply_to_key`` for any ORIGINAL
    safetensors key under ``_ORIG_GEMMA_LM_PREFIX``. ``SafetensorsStateDictLoader``
    does ``if expected_name is None: continue`` (sft_loader.py), so those tensors
    are never ``get_tensor``-copied -> never materialized -> never committed.

    Scoped to the GGUF-Gemma base load only; the shared AV ops object is unchanged.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def apply_to_key(self, key: str) -> str | None:
        if key.startswith(_ORIG_GEMMA_LM_PREFIX):
            # Skip the read entirely (this is exactly the set deleted post-load).
            return None
        if self._inner is None:
            return key
        return self._inner.apply_to_key(key)

    def apply_to_key_value(self, key: str, value: Any) -> Any:
        # Only ever called for kept keys (apply_to_key already returned non-None),
        # so the LM keys never reach here. Delegate verbatim for the survivors.
        if self._inner is None:
            from ltx_core.loader.sd_ops import KeyValueOperationResult

            return [KeyValueOperationResult(key, value)]
        return self._inner.apply_to_key_value(key, value)


def _read_target_vocab_from_header(path: str | list[str]) -> int | None:
    """Read the padded vocab size from the safetensors HEADER (no materialization).

    Previously taken from the materialized base embedding shape; since that tensor
    is now skipped at read, we read just its shape via ``get_slice().get_shape()``
    (header-only, no commit). Returns None if the embedding is not found.
    """
    import safetensors

    orig_embed_key = _ORIG_GEMMA_LM_PREFIX + "embed_tokens.weight"
    paths = path if isinstance(path, list) else [path]
    for shard_path in paths:
        try:
            with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
                if orig_embed_key in f.keys():
                    shape = f.get_slice(orig_embed_key).get_shape()
                    return int(shape[0])
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Gemma GGUF: header vocab probe failed for %s: %s", shard_path, exc)
    return None


def _to_ltx_key(hf_key: str) -> str | None:
    """Map a plain HF gemma3 key to its location inside GemmaTextEncoder.

    Returns None for keys we intentionally drop (the GGUF lm_head; it is tied to
    embed_tokens and supplied from the safetensors checkpoint via the LTX key ops).
    """
    if hf_key == "lm_head.weight":
        # gemma3 GGUFs usually do NOT ship a separate output.weight (weights tied),
        # but guard anyway: prefer the safetensors-provided tied lm_head.
        return None
    if hf_key.startswith("model.layers."):
        return _LTX_LM_PREFIX + hf_key[len("model.") :]
    if hf_key.startswith("model.embed_tokens."):
        return _LTX_LM_PREFIX + hf_key[len("model.") :]
    if hf_key == "model.norm.weight":
        return _LTX_LM_PREFIX + "norm.weight"
    if hf_key.startswith("model.norm."):
        return _LTX_LM_PREFIX + hf_key[len("model.") :]
    # Unknown / unexpected key — skip rather than poison the merged state_dict.
    logger.debug("Gemma GGUF: dropping unmapped remapped key %s", hf_key)
    return None


# Suffixes of the post-LTX-prefix keys that must STAY plain dequantized bf16
# tensors (NOT wrapped as GGMLQuantizedTensor): the token embedding and all norms.
# Everything else under the language model is an nn.Linear weight (q/k/v/o_proj,
# gate/up/down_proj) and is quantized.
def _is_norm_or_embed(ltx_key: str) -> bool:
    if ltx_key.endswith("embed_tokens.weight"):
        return True
    return any(ltx_key.endswith(suf) for suf in _NORM_SUFFIXES)


def _is_quantizable_linear(ltx_key: str) -> bool:
    """True only for the Gemma decoder Linear weights eligible for per-layer quant."""
    if not ltx_key.endswith(".weight"):
        return False
    return any(
        f".{name}.weight" in ltx_key
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    )


# ──────────────────────────────────────────────────────────────────────────────
# State-dict loader: safetensors base + GGUF Gemma overlay (the MERGE)
# ──────────────────────────────────────────────────────────────────────────────

class GemmaGGUFQuantStateDictLoader:
    """A StateDictLoader that produces a MERGED Gemma text-encoder state_dict.

    Steps:
      1. Delegate to the original SafetensorsModelStateDictLoader to load the full
         bf16 state_dict (Gemma transformer + feature_extractor + connectors),
         applying the LTX key remap (AV_GEMMA_TEXT_ENCODER_KEY_OPS) as usual.
      2. Drop every Gemma language-model key (``model.model.language_model.*``)
         from that dict — those are the ~24 GB bf16 weights we are replacing.
      3. Read the Gemma GGUF, remap gemma3 keys -> HF -> LTX-nested prefix, apply
         the RMSNorm +1 correction, dequantize embed/norm to bf16, and wrap the
         decoder Linear weights as GGMLQuantizedTensor.
      4. Overlay the GGUF-derived Gemma weights onto the (Gemma-stripped) base.

    The merged dict is returned so the builder's
    ``meta_model.load_state_dict(sd, strict=False, assign=True)`` populates the
    feature_extractor/connectors from safetensors and the Gemma transformer from
    the GGUF in one pass.
    """

    def __init__(
        self,
        gguf_path: str,
        base_loader: Any,
        embed_cpu_offload: bool = True,
        connector_gguf_path: str | None = None,
        connector_sd_ops: Any = None,
        layer_offload: bool = False,
    ) -> None:
        self.gguf_path = gguf_path
        # The original SafetensorsModelStateDictLoader from the text_encoder_builder.
        self._base_loader = base_loader
        # ── Phase 2 (component files) ────────────────────────────────────────────
        # When set, the 46GB monolith is NO LONGER in the builder's model_path (it is
        # replaced by the standalone text-projection file, which supplies the 4
        # `text_embedding_projection.*aggregate_embed.*` survivors). The monolith's
        # OTHER survivors — the 258 `model.diffusion_model.{video,audio}_embeddings_
        # connector.*` tensors — are then absent from the base load, so we INJECT
        # them here from the LTX transformer GGUF (Option A, VERIFICATION_LOG §9.5).
        # These connector tensors are stored F32/BF16 in the GGUF (NOT K-quantized),
        # so a plain float cast reproduces the monolith's bf16 values exactly. When
        # ``connector_gguf_path`` is None, behavior is the unchanged monolith path.
        self._connector_gguf_path = connector_gguf_path
        # The real AV key-ops (AV_GEMMA_TEXT_ENCODER_KEY_OPS), used to remap the
        # injected connectors' monolith-form keys through the SAME ops the monolith
        # connectors would have passed through (-> embeddings_processor.*_connector.*),
        # yielding byte-identical merged keys/values.
        self._connector_sd_ops = connector_sd_ops
        # VRAM (Lever 3): when True, the 1.9 GB bf16 token embedding (and its tied
        # lm_head) is NOT placed in the returned state_dict; it is stashed on CPU in
        # ``held_embed_cpu`` and assigned to the built model on CPU afterwards (the
        # service does this). Keeping it OUT of the merged sd leaves the meta model's
        # embed_tokens.weight on the meta device, so SingleGPUModelBuilder._return_model
        # early-returns WITHOUT a final ``meta_model.to(device)`` — which would
        # otherwise drag the embedding onto the GPU. See GemmaGGUFQuantLoaderService.
        self.embed_cpu_offload = embed_cpu_offload
        self.held_embed_cpu: torch.Tensor | None = None
        # VRAM (Lever — TE per-layer offload): when True, the per-layer GGUF Gemma
        # decoder weights (and the per-layer norms) under ``...language_model.layers.``
        # are left on CPU rather than moved to ``target_device`` during the GGUF load.
        # They are then streamed to the GPU one window at a time during the encode
        # forward by GemmaLayerOffloadService, capping the ~15 GB encode peak. When
        # False this is a no-op (all layers GPU-resident, today's exact behavior).
        self.layer_offload = layer_offload

    def metadata(self, path: str) -> dict:
        # Config still comes from the distilled checkpoint metadata (it carries the
        # LTX feature-extractor / connector config). GGUF has no LTX config.
        return self._base_loader.metadata(path)

    def load(
        self,
        path: str | list[str],
        sd_ops: Any = None,
        device: torch.device | None = None,
    ) -> Any:
        from ltx_core.loader.primitives import StateDict

        target_device = device or torch.device("cpu")

        # ── 2a. Capture the TARGET vocab size from the HEADER (no materialization) ─
        # The LTX Gemma3 meta model's embed_tokens (and tied lm_head) is sized to the
        # padded vocab (262208 = 262144 + 64 padding-token rows), but the GGUF ships
        # only the real 262144 rows. We must zero-pad the GGUF embedding up to this
        # target, or load_state_dict raises a size mismatch. We used to read this from
        # the materialized base embedding; since that tensor is now SKIPPED at read
        # (commit reduction below), read its shape from the safetensors header instead
        # (get_slice().get_shape() — header-only, no commit). Identical value (262208);
        # falls back to None (no padding) if the base embedding is somehow absent.
        target_vocab: int | None = _read_target_vocab_from_header(path)
        if target_vocab is not None:
            logger.info(
                "Gemma GGUF merge: target (padded) vocab size from base embedding header = %d",
                target_vocab,
            )

        # ── 1. Base bf16 LTX-side weights, LTX-remapped (Gemma LM SKIPPED at read) ─
        # CRITICAL (OOM avoidance): load the BASE safetensors on CPU regardless of
        # the requested target_device, then move only the small kept LTX-side tensors
        # to cuda. The base safetensors contain the FULL ~24 GB bf16 Gemma language
        # model, which we replace with the compressed GGUF below.
        #
        # COMMIT REDUCTION (Stage 1): rather than read those ~24 GB into committed CPU
        # RAM and delete them, we wrap sd_ops in _SkipGemmaLMSDOps so every original
        # ``language_model.model.*`` key returns None from apply_to_key -> the loader
        # skips its get_tensor copy entirely (never materialized, never committed).
        # This is byte-identical to the prior read-then-delete: the skipped set equals
        # exactly the keys formerly deleted at the strip step (see _SkipGemmaLMSDOps /
        # _ORIG_GEMMA_LM_PREFIX). Survivors (feature_extractor / connectors /
        # vision_tower / multi_modal_projector) are kept unchanged.
        cpu_device = torch.device("cpu")
        base = self._base_loader.load(path, sd_ops=_SkipGemmaLMSDOps(sd_ops), device=cpu_device)
        base_sd: dict[str, torch.Tensor] = dict(base.sd)

        n_base = len(base_sd)

        # ── 2. Defensive strip (now a no-op): the bf16 Gemma LM keys were already ──
        #   skipped at read above, so these deletions normally remove nothing. Kept
        #   as belt-and-suspenders in case an upstream key-op ever emits an LM key.
        for k in [k for k in base_sd if k.startswith(_LTX_LM_PREFIX)]:
            del base_sd[k]
        # Also drop the tied lm_head if present — GGUF supplies embed_tokens and
        # lm_head is tied; we re-tie below from the dequantized embedding.
        base_sd.pop(_LTX_LM_HEAD_KEY, None)
        n_stripped = n_base - len(base_sd)

        # ── 2a-inject. (Phase 2) Inject the embeddings connectors from the GGUF ───
        # When the monolith has been dropped from model_path (component-files mode),
        # the base load no longer contains the 258 connector tensors. Re-source them
        # from the LTX transformer GGUF here, with keys + dtype byte-identical to
        # what the monolith path produced, so the downstream merge / module ops see
        # an identical state_dict. Injection happens BEFORE the 2b device move +
        # GGUF-Gemma overlay so the connectors ride the same code path as the kept
        # safetensors survivors.
        if self._connector_gguf_path is not None:
            injected = self._load_gguf_connectors(cpu_device)
            n_inj = len(injected)
            # Guard: the injected connector keys must be DISJOINT from the base (the
            # monolith was dropped, so the base must not already carry them).
            collisions = [k for k in injected if k in base_sd]
            if collisions:
                raise RuntimeError(
                    "Gemma component-files: connector injection collided with "
                    f"{len(collisions)} existing base keys (monolith not dropped?): "
                    f"{collisions[:3]}..."
                )
            base_sd.update(injected)
            logger.info(
                "Gemma component-files: injected %d embeddings_connector tensors from "
                "GGUF %s (monolith dropped; aggregate_embed from projection file)",
                n_inj,
                Path(self._connector_gguf_path).name,
            )

        # ── 2b. Move the KEPT (non-Gemma, LTX-side) tensors to target_device ──────
        # These are the small feature_extractor / connector / vision_tower weights
        # (~3 GB total). They MUST end up on cuda: SingleGPUModelBuilder._return_model
        # only does a final meta_model.to(device) when NO param/buffer is left on the
        # meta device — but our GGMLQuantizedTensor buffers and assign=True loading
        # can leave that final move unreached/partial. So we move kept tensors here
        # explicitly rather than relying on a later builder-side .to(device).
        if target_device.type != "cpu":
            for k, v in base_sd.items():
                if isinstance(v, torch.Tensor) and v.device != target_device:
                    base_sd[k] = v.to(target_device, non_blocking=True)

        logger.info(
            "Gemma GGUF merge: kept %d non-Gemma safetensors tensors (feature_extractor / "
            "connectors) on %s, stripped %d bf16 Gemma tensors (freed in CPU RAM) for GGUF overlay",
            len(base_sd),
            target_device,
            n_stripped,
        )

        # ── 3. Read GGUF, remap, correct, (de)quantize ───────────────────────────
        gguf_sd = self._load_gguf_gemma(target_device, target_vocab)

        embed_key = _LTX_LM_PREFIX + "embed_tokens.weight"

        # ── 3b. (Lever 3) Hold the token embedding back on CPU ────────────────────
        # The 1.9 GB bf16 embedding (and its tied lm_head) is the single largest
        # non-quant GPU resident. The encode path only needs it for the INPUT lookup
        # (embed_tokens), which we can run on CPU and ship only the tiny [B,T,3840]
        # hidden tensor to the GPU; lm_head is unused by encoding. By stashing the
        # embedding here and NOT inserting it (or lm_head) into the merged sd, the
        # meta model's embed_tokens.weight stays on the meta device, so the builder's
        # _return_model early-returns without a final meta_model.to(device) (which
        # would otherwise pull the embedding onto the GPU). The service assigns the
        # CPU embedding + installs the CPU-lookup forward wrapper after build().
        if self.embed_cpu_offload and embed_key in gguf_sd:
            emb = gguf_sd.pop(embed_key)
            if isinstance(emb, torch.Tensor) and emb.device.type != "cpu":
                emb = emb.to("cpu")
            self.held_embed_cpu = emb
            logger.info(
                "Gemma GGUF (Lever 3): holding embed_tokens %s on CPU (out of merged sd) "
                "-> -%.1f MB GPU steady; lm_head left untied/absent on GPU",
                tuple(emb.shape),
                emb.numel() * emb.element_size() / (1024 * 1024),
            )

        # ── 4. Overlay ───────────────────────────────────────────────────────────
        merged = base_sd
        merged.update(gguf_sd)

        # Re-tie lm_head to the (plain bf16, now vocab-padded) embed_tokens so
        # generate()/lm_head paths still work even though we dropped the tied
        # safetensors copy. Both share the [target_vocab, 3840] padded embedding,
        # matching the meta model's tied lm_head shape.
        # When embed is CPU-offloaded (Lever 3) we intentionally do NOT add either
        # embed_tokens or lm_head to the merged sd here — the service re-ties them on
        # CPU after build (see GemmaGGUFQuantLoaderService.patched_text_encoder).
        if embed_key in merged and _LTX_LM_HEAD_KEY not in merged:
            merged[_LTX_LM_HEAD_KEY] = merged[embed_key]

        n_quant = sum(1 for v in merged.values() if isinstance(v, GGMLQuantizedTensor))
        logger.info(
            "Gemma GGUF merge complete: %d total tensors (%d quantized GGMLQuantizedTensor, "
            "rest bf16) on %s",
            len(merged),
            n_quant,
            target_device,
        )

        return StateDict(
            sd=merged,
            device=target_device,
            size=sum(_safe_numel(v) for v in merged.values()),
            dtype={torch.uint8, torch.bfloat16},
        )

    def _load_gguf_gemma(
        self, target_device: torch.device, target_vocab: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Read the GGUF and return Gemma weights keyed for the LTX text encoder.

        ``target_vocab`` is the padded vocab size the LTX meta model expects for
        embed_tokens / lm_head (262208). The GGUF embedding (262144 rows) is
        zero-padded up to it; see the per-key handling below.
        """
        import gguf as gguf_lib
        import numpy as np

        logger.info(
            "Gemma GGUF quant-load from %s -> %s",
            Path(self.gguf_path).name,
            target_device,
        )
        reader = gguf_lib.GGUFReader(self.gguf_path, mode="r")

        # Read raw GGUF tensors keyed by their native llama.cpp names.
        raw_entries: dict[str, tuple[torch.Tensor, int, tuple[int, ...]]] = {}
        for tensor in reader.tensors:
            ggml_type = tensor.tensor_type.value
            # GGUF stores shape reversed (column-major).
            float_shape = tuple(reversed(tensor.shape.tolist()))
            # Explicit owned copy off the GGUF memmap — breaks the np.memmap
            # alias entirely (city96 ComfyUI-GGUF #444 family). copy=True here
            # guarantees raw_np never aliases the mmap, so the torch tensor below
            # owns its bytes outright.
            raw_np = np.array(tensor.data, copy=True)
            raw_flat = torch.from_numpy(raw_np).reshape(-1)  # 1D uint8, owns memory
            raw_entries[tensor.name] = (raw_flat, ggml_type, float_shape)

        # Remap llama.cpp names -> HF gemma3 names (city96 verbatim).
        # sd_map_replace operates on KEYS only; carry the payload tuples as values.
        remapped = sd_map_replace(raw_entries, GEMMA3_SD_MAP)

        # CPU device used to hold the offloaded per-layer tensors back (TE offload).
        cpu_device_off = torch.device("cpu")

        out: dict[str, torch.Tensor] = {}
        n_quant = 0
        n_norm = 0
        n_embed = 0
        n_cpu_layer = 0
        for hf_key, (raw_flat, ggml_type, float_shape) in remapped.items():
            ltx_key = _to_ltx_key(hf_key)
            if ltx_key is None:
                continue

            # ── TE per-layer offload ──────────────────────────────────────────────
            # When layer_offload is on, the per-layer Gemma decoder tensors (the
            # quantized Linears AND the per-layer norms) stay on CPU; only this
            # ``...language_model.layers.*`` keyset is held back. The final
            # ``language_model.norm.weight``, rotary buffers, embeddings (Lever 3),
            # and all LTX-side connector/base weights still go to target_device as
            # today. GemmaLayerOffloadService streams these CPU layers to the GPU per
            # window during encode. ``key_device`` is the device THIS key lands on.
            is_offloaded_layer = (
                self.layer_offload and ".language_model.layers." in ltx_key
            )
            key_device = cpu_device_off if is_offloaded_layer else target_device
            if is_offloaded_layer:
                n_cpu_layer += 1

            if _is_norm_or_embed(ltx_key):
                is_norm = any(ltx_key.endswith(suf) for suf in _NORM_SUFFIXES)
                if is_norm:
                    # Norms are tiny (3840 elems); dequantize directly on the key's
                    # device (CPU when this is an offloaded per-layer norm, else
                    # target_device). Apply the RMSNorm +1 correction AFTER dequant:
                    # llama.cpp baked (1 + w); HF re-adds 1 -> subtract here.
                    deq = _dequant_to_bf16(raw_flat, ggml_type, float_shape, key_device)
                    deq = (deq.float() - 1.0).to(torch.bfloat16)
                    n_norm += 1
                    out[ltx_key] = deq
                    continue

                # ── embed_tokens — VRAM-CRITICAL dequant ──────────────────────────
                # The Q4_K embedding is ~1B params (262144x3840). Dequantizing it on
                # the GPU spikes peak VRAM to ~21 GB transiently (the dequant kernel
                # materializes large fp32 intermediates for the whole tensor at once),
                # which is the dominant cause of the build's ~28 GB peak and the WDDM
                # shared-memory spill. The *result* is only ~1.9 GB bf16.
                #
                # Fix: dequantize (and zero-pad) the embedding ENTIRELY ON CPU, where
                # the transient intermediates land in the machine's ~63 GB system RAM
                # for free, then move the final 1.9 GB bf16 tensor to the GPU once.
                # This removes the ~19 GB GPU transient without changing the numerics
                # (the dequant kernel + the -1/padding logic are byte-identical; only
                # the device of the intermediate work differs). Norms stay on device.
                cpu = torch.device("cpu")
                deq = _dequant_to_bf16(raw_flat, ggml_type, float_shape, cpu)
                # zero-pad the GGUF vocab (262144) up to the LTX meta model's padded
                # vocab (target_vocab, e.g. 262208). The extra rows are reserved
                # padding tokens the tokenizer never emits, so zeros are numerically
                # safe (confirmed: forward cosine vs bf16 = 1.0009 with this padding).
                # Without it, load_state_dict raises a size mismatch on embed_tokens /
                # the tied lm_head. Done on CPU (cheap) before the single GPU move.
                if target_vocab is not None and deq.shape[0] < target_vocab:
                    pad_rows = target_vocab - deq.shape[0]
                    pad = torch.zeros(
                        (pad_rows, deq.shape[1]),
                        dtype=deq.dtype,
                        device=cpu,
                    )
                    deq = torch.cat([deq, pad], dim=0)
                    logger.info(
                        "Gemma GGUF: zero-padded embed_tokens %s -> [%d, %d] "
                        "(+%d padding-token rows)",
                        ltx_key,
                        deq.shape[0],
                        deq.shape[1],
                        pad_rows,
                    )
                # Single move of the finished 1.9 GB bf16 embedding to the GPU —
                # UNLESS it is going to be CPU-offloaded (Lever 3), in which case we
                # leave it on CPU and never touch GPU memory for it at all (step 3b
                # in load() stashes it and keeps it out of the merged sd).
                if target_device.type != "cpu" and not self.embed_cpu_offload:
                    deq = deq.to(target_device, non_blocking=False)
                n_embed += 1
                out[ltx_key] = deq
                continue

            if _is_quantizable_linear(ltx_key):
                if ggml_type in _FLOAT_GGML_TYPES:
                    # A Linear weight stored unquantized in the GGUF — keep bf16.
                    # On key_device (CPU when this is an offloaded per-layer Linear).
                    out[ltx_key] = _dequant_to_bf16(
                        raw_flat, ggml_type, float_shape, key_device
                    )
                else:
                    qt = GGMLQuantizedTensor(raw_flat, ggml_type, float_shape)
                    # Move to GPU only when this key is NOT an offloaded per-layer
                    # weight; offloaded layer weights stay on CPU (key_device==cpu)
                    # and are streamed to the GPU per window during encode.
                    if key_device.type != "cpu":
                        qt = qt.to(key_device, non_blocking=True)
                    out[ltx_key] = qt
                    n_quant += 1
                continue

            # Any other Gemma weight (e.g. biases if present) -> plain bf16 on the
            # key's device (CPU when this is an offloaded per-layer weight).
            out[ltx_key] = _dequant_to_bf16(raw_flat, ggml_type, float_shape, key_device)

        logger.info(
            "Gemma GGUF tensors mapped: %d quantized Linear, %d dequantized norms "
            "(+1 corrected), %d embeddings%s",
            n_quant,
            n_norm,
            n_embed,
            (
                f"; {n_cpu_layer} per-layer tensors held on CPU (TE offload)"
                if self.layer_offload
                else ""
            ),
        )
        return out


    def _load_gguf_connectors(
        self, target_device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Read the 258 embeddings-connector tensors from the LTX transformer GGUF.

        The connectors are stored in the GGUF under BARE keys
        ``{video,audio}_embeddings_connector.*`` as F32/BF16 (NEVER K-quantized).
        To reproduce the monolith path byte-for-byte we:
          1. prepend ``model.diffusion_model.`` to each key (the monolith's original
             key form the AV ops expect),
          2. cast F32->bf16 / pass bf16 through (the monolith stored these bf16),
          3. run the SAME AV key-ops (AV_GEMMA_TEXT_ENCODER_KEY_OPS) over them so the
             final keys become ``embeddings_processor.{video,audio}_connector.*`` —
             identical to what the monolith connectors produced.

        Fails loudly if any connector tensor is a quantized (K-quant) ggml type, since
        the float-only assumption (no dequant kernel) would otherwise corrupt weights.
        """
        import gguf as gguf_lib
        import numpy as np

        path = self._connector_gguf_path
        assert path is not None
        logger.info(
            "Gemma component-files: reading embeddings_connector tensors from GGUF %s",
            Path(path).name,
        )
        reader = gguf_lib.GGUFReader(path, mode="r")

        _ORIG_CONN_PREFIX = "model.diffusion_model."
        orig_form: dict[str, torch.Tensor] = {}
        n_f32 = 0
        n_bf16 = 0
        for tensor in reader.tensors:
            name = tensor.name
            if "embeddings_connector" not in name:
                continue
            # Only the bare {video,audio}_embeddings_connector.* tensors (confirmed
            # to be the only connector-bearing keys in the GGUF).
            if not (
                name.startswith("video_embeddings_connector.")
                or name.startswith("audio_embeddings_connector.")
            ):
                logger.warning(
                    "Gemma component-files: skipping unexpected connector key %s", name
                )
                continue
            ggml_type = tensor.tensor_type.value
            if ggml_type not in _FLOAT_GGML_TYPES:
                raise RuntimeError(
                    "Gemma component-files: connector tensor "
                    f"{name!r} is ggml type {tensor.tensor_type.name} (quantized) — "
                    "expected F32/BF16. Refusing to dequantize connectors (Option A "
                    "assumes float storage)."
                )
            float_shape = tuple(reversed(tensor.shape.tolist()))
            raw_np = np.array(tensor.data, copy=True)
            raw_flat = torch.from_numpy(raw_np).reshape(-1)
            # Cast to bf16 on CPU (these are tiny vs the Gemma LM); device move below.
            deq = _dequant_to_bf16(raw_flat, ggml_type, float_shape, torch.device("cpu"))
            if ggml_type == _GGML_F32:
                n_f32 += 1
            else:
                n_bf16 += 1
            orig_form[_ORIG_CONN_PREFIX + name] = deq

        # Run the SAME AV ops the monolith connectors would have gone through so the
        # final keys/values match exactly (-> embeddings_processor.*_connector.*).
        out: dict[str, torch.Tensor] = {}
        ops = self._connector_sd_ops
        for orig_key, value in orig_form.items():
            if ops is None:
                out[orig_key] = value
                continue
            mapped = ops.apply_to_key(orig_key)
            if mapped is None:
                # The AV ops do not drop connector keys; defensive only.
                continue
            for k, v in ops.apply_to_key_value(mapped, value):
                out[k] = v

        logger.info(
            "Gemma component-files: prepared %d connector tensors (%d F32->bf16, "
            "%d bf16 passthrough) -> embeddings_processor.*_connector.*",
            len(out),
            n_f32,
            n_bf16,
        )
        return out


def _dequant_to_bf16(
    raw_flat: torch.Tensor,
    ggml_type: int,
    float_shape: tuple[int, ...],
    target_device: torch.device,
) -> torch.Tensor:
    """Dequantize (or reinterpret) a raw GGUF tensor to a bf16 tensor on device."""
    if ggml_type in _FLOAT_GGML_TYPES:
        if ggml_type == _GGML_F32:
            t = raw_flat.view(torch.float32).view(float_shape).to(torch.bfloat16)
        elif ggml_type == _GGML_F16:
            t = raw_flat.view(torch.float16).view(float_shape).to(torch.bfloat16)
        else:  # BF16
            t = raw_flat.view(torch.bfloat16).view(float_shape)
    else:
        # Reuse the verified quant kernels; dequant on the target device.
        src = raw_flat.to(target_device) if target_device.type != "cpu" else raw_flat
        return dequantize_ggml_tensor(src, ggml_type, float_shape, torch.bfloat16)
    if target_device.type != "cpu":
        t = t.to(target_device, non_blocking=True)
    return t


def _safe_numel(t: torch.Tensor) -> int:
    try:
        return t.numel() if not isinstance(t, GGMLQuantizedTensor) else t.as_subclass(torch.Tensor).numel()
    except Exception:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# ModuleOps: convert ONLY the Gemma decoder Linears to buffers + patch forward()
# ──────────────────────────────────────────────────────────────────────────────

# Reuse the transformer service's per-Linear patch (buffer swap + dequant forward).
from engine.gguf.quant_service import _patch_linear_for_ggml_dequant  # noqa: E402


def _patch_gemma_skip_full_logits(model: torch.nn.Module) -> None:
    """Make the Gemma encode-path forward compute logits for only ONE token.

    VRAM motivation
    ---------------
    ``GemmaTextEncoder.precompute`` calls ``self.model(input_ids=...,
    output_hidden_states=True)`` and consumes ONLY ``outputs.hidden_states`` — the
    ``logits`` are discarded. But ``Gemma3ForConditionalGeneration.forward``
    unconditionally runs ``self.lm_head(hidden_states[:, slice_indices, :])`` with
    the default ``logits_to_keep=0`` (-> the FULL sequence). At the encoder's fixed
    seq-len of 1024 and the 262208-row vocab that materializes a
    ``[1, 1024, 262208]`` logits tensor (~0.5 GB) plus its matmul intermediates,
    spiking forward-pass VRAM by ~2.4 GB — enough to push the steady ~15 GB build
    over the 16 GB card during text encoding.

    Fix
    ---
    Default ``logits_to_keep=1`` on the inner Gemma model's ``forward`` whenever the
    caller did not request a specific value. This restricts ``lm_head`` to the LAST
    token, shrinking the logits tensor ~1024x. It is numerically safe for BOTH
    consumers:
      * precompute/encode: logits are thrown away, hidden_states are untouched
        (lm_head does not feed back into the decoder), so the encoding is bit-
        identical.
      * generate()/prompt-enhance: autoregressive decoding only ever needs the
        last token's logits — ``logits_to_keep=1`` is exactly transformers' own
        default for generation, so sampling is unchanged.
    A caller that explicitly passes ``logits_to_keep`` (e.g. a future full-logits
    path) still gets its requested value.
    """
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "forward"):
        logger.warning(
            "Gemma GGUF: could not find inner Gemma model to patch logits_to_keep — "
            "full-sequence logits will be computed (higher forward-pass VRAM)."
        )
        return
    orig_forward = inner.forward

    def _forward_min_logits(*args: Any, **kwargs: Any) -> Any:
        if "logits_to_keep" not in kwargs:
            kwargs["logits_to_keep"] = 1
        return orig_forward(*args, **kwargs)

    inner.forward = _forward_min_logits  # type: ignore[method-assign]
    logger.info(
        "Gemma GGUF module_ops: defaulted Gemma forward logits_to_keep=1 "
        "(skip full-sequence lm_head logits in the encode path)"
    )


def _patch_gemma_for_ggml_dequant(model: torch.nn.Module) -> torch.nn.Module:
    """Patch only the Gemma decoder Linears (q/k/v/o_proj, gate/up/down_proj).

    Excludes norms (nn.Module, not Linear) and embeddings (nn.Embedding), which are
    loaded as plain bf16 and consumed directly. lm_head is left untouched (plain
    bf16, tied to embed_tokens; the per-layer dequant patch is unnecessary there
    and would force a 262k-row dequant on every call).

    Also defaults the Gemma forward's ``logits_to_keep=1`` so the discarded encode-
    path logits do not spike forward VRAM (see _patch_gemma_skip_full_logits).
    """
    _TARGET_LEAF_NAMES = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in _TARGET_LEAF_NAMES:
            continue
        # Only patch Linears inside the Gemma language model.
        if "language_model" not in name:
            continue
        _patch_linear_for_ggml_dequant(module)
        count += 1
    logger.info(
        "Gemma GGUF module_ops: patched %d decoder Linear layers (buffer + dequant forward)",
        count,
    )
    _patch_gemma_skip_full_logits(model)
    return model


def _make_gemma_ggml_quant_module_ops() -> Any:
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.text_encoders.gemma import GemmaTextEncoder

    return ModuleOps(
        name="gemma_ggml_per_layer_dequant",
        matcher=lambda model: isinstance(model, GemmaTextEncoder),
        mutator=_patch_gemma_for_ggml_dequant,
    )


def _install_cpu_embed_offload(text_encoder: Any, embed_cpu: torch.Tensor) -> None:
    """(Lever 3) Wire the held-back CPU token embedding into the built encoder.

    Steps:
      1. Assign ``embed_cpu`` to the (currently meta) ``embed_tokens.weight`` and
         re-tie ``lm_head.weight`` to the SAME CPU storage (0 extra bytes).
      2. Replace the outer Gemma3ForConditionalGeneration.forward with a wrapper
         that:
           - runs the token-embedding lookup ON CPU (``embed_tokens`` applies
             Gemma's sqrt(hidden) scaling in its own forward), then moves ONLY the
             small [B,T,3840] bf16 hidden tensor to the GPU,
           - runs the decoder stack on the GPU via the inner Gemma3Model with
             ``inputs_embeds`` (so the GPU never touches the 1.9 GB embedding),
           - computes logits with the tied CPU lm_head on just the last
             ``logits_to_keep`` tokens (tiny; encode discards them, generate only
             needs the last token), returning logits on the compute device.

    Numerically identical to the on-GPU embedding: same weights, same scaling, same
    decoder math — only the embedding lookup's device differs (CPU vs GPU), and the
    lookup is an exact gather. Verified: video/audio encoding std unchanged.
    """
    from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast

    outer = text_encoder.model  # Gemma3ForConditionalGeneration
    inner = outer.model  # Gemma3Model (decoder stack + norm, on the GPU)
    lang = inner.language_model  # Gemma3TextModel (.embed_tokens, .norm)

    with torch.no_grad():
        embed_param = torch.nn.Parameter(embed_cpu, requires_grad=False)
        lang.embed_tokens.weight = embed_param
        # Re-tie lm_head to the identical CPU storage (no duplicate allocation).
        outer.lm_head.weight = embed_param
        # Keep the embedding submodule's non-persistent buffers (Gemma's embed_scale)
        # co-located with the CPU weight. Under a StateDictRegistry the text encoder is
        # built on CPU and moved to the GPU out-of-place; that whole-model move also
        # pulls embed_scale onto the GPU, but the CPU-lookup forward computes
        # embed_tokens(input_ids) * embed_scale on CPU and would otherwise mix devices.
        # No-op on the default GPU-build (DummyRegistry) path where embed_scale is
        # already on CPU.
        embed_dev = embed_param.device
        for _bname, _buf in list(lang.embed_tokens._buffers.items()):
            if _buf is not None and _buf.device != embed_dev:
                lang.embed_tokens._buffers[_bname] = _buf.to(embed_dev)

    orig_forward = outer.forward

    def _cpu_embed_forward(
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        logits_to_keep: int = 1,
        **kwargs: Any,
    ) -> Any:
        # Compute device = where the decoder lives (language-model norm weight).
        compute_dev = lang.norm.weight.device
        embed_dev = lang.embed_tokens.weight.device
        if inputs_embeds is None and input_ids is not None:
            # CPU embedding lookup (+ Gemma scaling), then ship the small hidden to GPU.
            ie = lang.embed_tokens(input_ids.to(embed_dev))
            inputs_embeds = ie.to(compute_dev)
            input_ids = None
        out = inner(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        hidden_states = out[0]
        # lm_head on CPU (tied weight) for only the requested last tokens.
        sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        last = hidden_states[:, sl, :].to(embed_dev)
        logits = torch.nn.functional.linear(last, outer.lm_head.weight).to(compute_dev)
        return Gemma3CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
            image_hidden_states=None,
        )

    outer.forward = _cpu_embed_forward  # type: ignore[method-assign]
    _ = orig_forward  # original retained for clarity; replaced wholesale.
    logger.info(
        "Gemma GGUF (Lever 3): installed CPU-embed forward wrapper "
        "(embed lookup on CPU, decoder on GPU, lm_head on CPU last-token only)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Service — mirrors GGUFQuantLoaderService.install but targets text_encoder_builder
# ──────────────────────────────────────────────────────────────────────────────

class GemmaGGUFQuantLoaderService:
    """Per-layer GGUF dequantization for the Gemma-3 text encoder.

    Keeps the Gemma transformer compressed in VRAM (~7.3 GB Q4_K_M instead of
    ~24 GB bf16). Each decoder Linear dequantizes its weight at forward() time and
    frees the temporary bf16 tensor after the matmul. The LTX-side feature
    extractor / embedding connectors remain bf16 (loaded from the distilled
    checkpoint), merged in by GemmaGGUFQuantStateDictLoader.

    Usage:
        service = GemmaGGUFQuantLoaderService(gguf_path)
        service.install(model_ledger)
    """

    def __init__(
        self,
        gguf_path: str,
        component_text_projection_path: str | None = None,
        connector_gguf_path: str | None = None,
        layer_offload: bool = False,
    ) -> None:
        self.gguf_path = gguf_path
        # ── Phase 2 (component files), both must be set to enable the monolith drop ──
        # component_text_projection_path: standalone bf16 file with the 4 aggregate_
        #   embed survivors; replaces the 46GB monolith as the text-encoder base path.
        # connector_gguf_path: the LTX transformer GGUF that carries the 258 connector
        #   tensors we inject (Option A). When either is None, the loader runs the
        #   unchanged monolith path (monolith still in model_path).
        self.component_text_projection_path = component_text_projection_path
        self.connector_gguf_path = connector_gguf_path
        # When True, keep the 48 GGUF-quantized Gemma decoder layers CPU-resident and
        # stream them to the GPU one window at a time during the encode forward
        # (GemmaLayerOffloadService), capping the ~15 GB encode peak. When False the
        # decoder layers are all GPU-resident (today's exact behavior).
        self.layer_offload = layer_offload

    def install(self, model_ledger: Any) -> None:
        if not Path(self.gguf_path).exists():
            raise FileNotFoundError(f"Gemma GGUF file not found: {self.gguf_path}")

        if not hasattr(model_ledger, "text_encoder_builder"):
            logger.warning(
                "ModelLedger has no text_encoder_builder — Gemma GGUF install skipped "
                "(was a gemma_root provided?)"
            )
            return

        # Warm-import guard: import ltx_core.loader before our tensor-subclass
        # quantization touches the build path (mirrors the transformer service,
        # avoids a circular-import edge during meta-model construction).
        import ltx_core.loader  # noqa: F401

        builder = model_ledger.text_encoder_builder

        # ── Phase 2: component-files mode (drop the 46GB monolith) ────────────────
        # The text_encoder_builder.model_path is (checkpoint_path=MONOLITH, *qat_shards).
        # The monolith contributes EXACTLY two survivor sets to the text encoder:
        #   * 4   text_embedding_projection.*aggregate_embed.*  -> from projection file
        #   * 258 model.diffusion_model.*embeddings_connector.*  -> injected from GGUF
        # (vision_tower / multi_modal_projector / Gemma LM all come from the qat shards,
        # NOT the monolith — confirmed by header inspection). So when both component
        # paths are present we (a) swap the monolith for the standalone projection file
        # in model_path (aggregate_embed source), and (b) hand the connector GGUF +
        # AV ops to the loader for connector injection. With either path absent we keep
        # the monolith in model_path and inject nothing (unchanged behavior).
        new_model_path = builder.model_path
        connector_gguf_path: str | None = None
        connector_sd_ops: Any = None
        component_mode = bool(
            self.component_text_projection_path and self.connector_gguf_path
        )
        if component_mode:
            proj = self.component_text_projection_path
            conn = self.connector_gguf_path
            assert proj is not None and conn is not None
            if not Path(proj).exists():
                raise FileNotFoundError(f"Text projection file not found: {proj}")
            if not Path(conn).exists():
                raise FileNotFoundError(f"Connector GGUF not found: {conn}")
            mp = builder.model_path
            mp_tuple = tuple(mp) if isinstance(mp, tuple) else (mp,)
            mono = str(model_ledger.checkpoint_path)
            # Replace the monolith entry (model_path[0]) with the projection file;
            # keep the qat shards. Guard that the first entry is indeed the monolith.
            if not mp_tuple or str(mp_tuple[0]) != mono:
                raise RuntimeError(
                    "Gemma component-files: text_encoder_builder.model_path[0] "
                    f"({mp_tuple[:1]}) is not the checkpoint monolith ({mono}); "
                    "cannot safely drop the monolith."
                )
            new_model_path = (proj, *mp_tuple[1:])
            connector_gguf_path = conn
            connector_sd_ops = builder.model_sd_ops  # AV_GEMMA_TEXT_ENCODER_KEY_OPS
            logger.info(
                "Gemma component-files: dropping monolith from text encoder model_path; "
                "aggregate_embed <- %s ; connectors <- GGUF %s",
                Path(proj).name,
                Path(conn).name,
            )

        # 1. Wrap the builder's existing safetensors loader so we MERGE rather than
        #    replace: base = original loader output (LTX-remapped bf16), overlay =
        #    GGUF Gemma weights.
        gemma_loader = GemmaGGUFQuantStateDictLoader(
            gguf_path=self.gguf_path,
            base_loader=builder.model_loader,
            embed_cpu_offload=True,
            connector_gguf_path=connector_gguf_path,
            connector_sd_ops=connector_sd_ops,
            layer_offload=self.layer_offload,
        )

        # 2. Add our per-layer dequant module_ops AFTER the existing Gemma module
        #    ops (GEMMA_MODEL_OPS + tokenizer/processor loads). Ordering matters:
        #    GEMMA_MODEL_OPS.create_and_populate registers rope/embed-scale buffers
        #    on the meta model and must run before our Linear-buffer swap — but our
        #    op only swaps Linear.weight, so either order is safe. We append.
        ggml_module_ops = _make_gemma_ggml_quant_module_ops()
        new_builder = dc_replace(
            builder,
            model_path=new_model_path,
            model_loader=gemma_loader,
            module_ops=(*builder.module_ops, ggml_module_ops),
        )
        model_ledger.text_encoder_builder = new_builder

        # 3. Wrap ledger.text_encoder so the model is built WITHOUT a dtype cast.
        #    SingleGPUModelBuilder.build(dtype=bf16) does {k: v.to(bf16)} over the
        #    whole state_dict; calling .to(bf16) on a GGMLQuantizedTensor (uint8
        #    storage) would reinterpret the raw bytes as bf16 -> corruption. Our
        #    loader already returns every float tensor as bf16, so dtype=None is
        #    both correct and necessary. The stock text_encoder() also passes
        #    dtype, so we cannot rely on it.
        original_text_encoder_fn = model_ledger.text_encoder.__func__ if hasattr(
            model_ledger.text_encoder, "__func__"
        ) else None

        ledger_device = model_ledger.device

        gemma_loader_ref = gemma_loader

        def patched_text_encoder() -> Any:
            tb = model_ledger.text_encoder_builder
            # Build on the ledger's TARGET device (CPU when a StateDictRegistry is
            # active, else the GPU). NO dtype cast (see above). With a registry the
            # merged Gemma state_dict is materialized + cached on CPU (commit-cheap);
            # with the default DummyRegistry _target_device() == ledger_device so this
            # is byte-identical to the prior `device=ledger_device` build.
            build_device = model_ledger._target_device()
            model = tb.build(device=build_device)

            # When we built on CPU (registry active), move the model to the GPU for
            # inference OUT-OF-PLACE, skipping any meta tensor (the held-back
            # embed_tokens.weight stays meta until the offload wrapper assigns it on
            # CPU below). _return_model() short-circuits its own .to(device) because of
            # that meta param, so the loader/this move is what places weights on the
            # GPU. The move replaces module params/buffers with fresh GPU tensors and
            # leaves the cached CPU state_dict objects untouched (VRAM-neutral); the
            # GGMLQuantizedTensor.to override is out-of-place and preserves subclass +
            # _ggml_type/_float_shape attrs.
            if build_device != ledger_device:
                # TE per-layer offload caveat: in offload mode the
                # ``language_model.layers.*`` GGUF buffers are deliberately held on
                # CPU (loader left them there) so GemmaLayerOffloadService can stream
                # them per window. This keep-resident GPU-move would drag them all
                # back onto the GPU, defeating the offload. Production runs with
                # keep-resident OFF (build_device == ledger_device), so this branch
                # is skipped there and the offload path is correct. To make the
                # keep-resident + offload combo correct too, we skip the move for
                # tensors already on CPU when layer_offload is on (the CPU-held layer
                # buffers); the GPU-resident kept tensors (connectors / final norm /
                # rotary) still move as before. Non-offload mode is unchanged.
                if gemma_loader_ref.layer_offload:
                    model = model._apply(
                        lambda t: t
                        if t.device.type in ("meta", "cpu")
                        else t.to(ledger_device)
                    )
                else:
                    model = model._apply(
                        lambda t: t if t.device.type == "meta" else t.to(ledger_device)
                    )

            # (Lever 3) Wire the held-back CPU token embedding + CPU-lookup forward.
            # The builder left embed_tokens.weight on the meta device (we kept it out
            # of the merged sd) so the build never pulled the 1.9 GB embedding onto
            # the GPU. Assign it on CPU now and install the offload forward wrapper.
            # Done AFTER the GPU move so the embedding stays resident on CPU.
            if gemma_loader_ref.held_embed_cpu is not None:
                _install_cpu_embed_offload(model, gemma_loader_ref.held_embed_cpu)

            model = model.eval()

            # (TE per-layer offload) Stream the GGUF-quantized Gemma decoder layers
            # CPU->GPU one window at a time during encode, capping the ~15 GB encode
            # peak. The loader already left the ``language_model.layers.*`` buffers on
            # CPU; here we patch each decoder layer's forward with the sliding-window
            # swap. Compute device = the language-model final norm weight device
            # (matches Lever-3's compute_dev). No-op + safe if the layer container
            # can't be located.
            if gemma_loader_ref.layer_offload:
                from engine.gemma.layer_offload_service import (
                    GemmaLayerOffloadService,
                )

                # outer = Gemma3ForConditionalGeneration; outer.model = Gemma3Model;
                # .language_model = Gemma3TextModel (.layers / .norm).
                outer = getattr(model, "model", None)
                inner = getattr(outer, "model", None) if outer is not None else None
                lang = (
                    getattr(inner, "language_model", None)
                    if inner is not None
                    else None
                )
                compute_dev = None
                if lang is not None and getattr(lang, "norm", None) is not None:
                    compute_dev = lang.norm.weight.device
                if compute_dev is None:
                    compute_dev = ledger_device
                offload_service = GemmaLayerOffloadService(
                    layers_on_gpu=2, compute_device=compute_dev
                )
                # Install on the inner Gemma model holding language_model.layers.
                offload_service.install(inner if inner is not None else model)
                logger.info(
                    "Gemma GGUF (TE offload): installed per-layer CPU->GPU streaming "
                    "(layers_on_gpu=2, compute_device=%s)",
                    compute_dev,
                )

            n_quant = sum(
                1
                for buf in model.buffers()
                if isinstance(buf, GGMLQuantizedTensor)
            )
            if n_quant:
                logger.info(
                    "Gemma GGUF per-layer quant active: %d quantized buffers on %s",
                    n_quant,
                    ledger_device,
                )
            else:
                logger.warning(
                    "Gemma GGUF quant: no GGMLQuantizedTensor buffers found after build — "
                    "per-layer dequant inactive. Check the GGUF key mapping / prefix."
                )
            return model

        # Bind as a plain function attribute (ModelLedger.text_encoder is a method;
        # the pipeline reads it as model_ledger.text_encoder()).
        model_ledger.text_encoder = patched_text_encoder  # type: ignore[method-assign]
        _ = original_text_encoder_fn  # retained for clarity; original not chained.

        logger.info(
            "GemmaGGUFQuantLoaderService installed: %s — Gemma stays compressed in VRAM "
            "(~7.3 GB vs ~24 GB bf16)",
            Path(self.gguf_path).name,
        )
