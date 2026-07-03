"""LTX fast video pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Final, cast

import torch

from engine.api_types import ImageConditioningInput
from engine.pipeline.common import default_tiling_config, encode_video_output, video_chunks_number
from engine.pipeline.utils import AudioOrNone, TilingConfigType, device_supports_fp8


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 slice-2 — clip-concatenation / latent-level extend (ENGINE wiring).
#
# GLOBAL monkeypatch installs that are NO-OP unless armed for the CURRENT job.
# The chain endpoint reuses ONE worker process for many jobs, so arming MUST be
# per-job: arm at the start of an extend-enabled generate, DISARM in a finally.
# A normal (non-extend) generate behaves byte-identically to today because both
# wrappers early-return to the original when `_EXTEND.armed` is False.
#
# Mechanism (proven by outputs/phase3_clip_concat_spike/spike.py, result GO):
#   * Wrap ltx_core.tools.VideoLatentTools.create_initial_state: let the original
#     build the patchified all-ones denoise_mask, then override the first K
#     latent-frame token blocks to (1.0 - overlap_strength). Token order is
#     (f h w), temporal patch size = 1, so latent frame f is the contiguous token
#     block f*(H_lat*W_lat):(f+1)*(H_lat*W_lat) in the (b, tokens, 1) mask.
#     overlap_strength=1.0 -> mask 0.0 hard freeze; 0.5 (default) -> soft.
#   * Wrap ltx_pipelines.distilled.denoise_audio_video (COMPOSING with the
#     worker's pre-denoise empty_cache wrapper — we wrap whatever it currently
#     is, not the raw original): on the armed Stage-1 call, inject
#     initial_video_latent=<padded tail> and capture the returned unpatchified
#     (b,c,F,H,W) latent's tail. DISARM after Stage 1 so Stage 2 runs stock.
# ─────────────────────────────────────────────────────────────────────────────
class _ExtendState:
    """Process-global arm state for the extend monkeypatches (one job at a time)."""

    armed: bool = False
    # Stage-1 seed to inject (padded tail, shape == Stage-1 target VideoLatentShape).
    inject_initial_latent: "torch.Tensor | None" = None
    # Number of leading latent frames to soften/freeze (overlap length K).
    overlap_frames: int = 0
    # denoise_mask value written on the overlap frames = 1.0 - overlap_strength.
    overlap_mask_value: float = 0.0
    # Set True the instant the mask override reached create_initial_state (proof
    # the injection reached the load-bearing spot).
    mask_override_applied: bool = False
    # Captured Stage-1 output (unpatchified video latent, (b,c,F,H,W)), cloned.
    captured_stage1_latent: "torch.Tensor | None" = None

    @classmethod
    def reset(cls) -> None:
        cls.armed = False
        cls.inject_initial_latent = None
        cls.overlap_frames = 0
        cls.overlap_mask_value = 0.0
        cls.mask_override_applied = False
        cls.captured_stage1_latent = None


_EXTEND = _ExtendState()
_EXTEND_PATCHES_INSTALLED = False


def _install_extend_patches() -> None:
    """Install the (idempotent) global extend wrappers.

    Wraps the CURRENT ``_distilled.denoise_audio_video`` (which, inside the
    worker, is already the pre-denoise empty_cache wrapper) so both behaviors
    compose. Installed lazily on the first extend-enabled generate; wrappers are
    no-ops until ``_EXTEND.armed`` is set.
    """
    global _EXTEND_PATCHES_INSTALLED
    if _EXTEND_PATCHES_INSTALLED:
        return

    import ltx_core.tools as _tools
    import ltx_pipelines.distilled as _distilled

    _orig_create_initial_state = _tools.VideoLatentTools.create_initial_state
    _prev_denoise_av = _distilled.denoise_audio_video  # may be worker's wrapper

    def _patched_create_initial_state(self, device, dtype, initial_latent=None):
        state = _orig_create_initial_state(self, device, dtype, initial_latent)
        if not _EXTEND.armed or _EXTEND.overlap_frames <= 0:
            return state
        # target_shape gives the latent grid (b, c, F, H, W). Temporal patch
        # size = 1, token order (f h w) -> latent frame f == token block
        # [f*HW : (f+1)*HW] in the patchified (b, tokens, 1) mask.
        tshape = self.target_shape.to_torch_shape()
        F_lat, H_lat, W_lat = int(tshape[2]), int(tshape[3]), int(tshape[4])
        hw = H_lat * W_lat
        k = min(_EXTEND.overlap_frames, F_lat)
        mask = state.denoise_mask.clone()
        assert mask.shape[1] >= F_lat * hw, (
            f"patchified mask tokens {mask.shape[1]} < F*H*W {F_lat * hw}"
        )
        mask[:, : k * hw, ...] = float(_EXTEND.overlap_mask_value)
        from dataclasses import replace as _dc_replace

        new_state = _dc_replace(state, denoise_mask=mask)
        _EXTEND.mask_override_applied = True
        import logging as _lg
        _lg.getLogger(__name__).info(
            "extend: create_initial_state override froze first %d/%d latent frames "
            "(mask=%.3f, %d/%d tokens); grid F=%d H=%d W=%d",
            k, F_lat, _EXTEND.overlap_mask_value, k * hw, mask.shape[1],
            F_lat, H_lat, W_lat,
        )
        return new_state

    def _patched_denoise_av(*args, **kwargs):
        # Compose with the previously-installed wrapper (worker's empty_cache).
        if not _EXTEND.armed:
            return _prev_denoise_av(*args, **kwargs)
        if _EXTEND.inject_initial_latent is not None:
            kwargs["initial_video_latent"] = _EXTEND.inject_initial_latent
            import logging as _lg
            _lg.getLogger(__name__).info(
                "extend: injected initial_video_latent (padded tail) for Stage 1"
            )
        video_state, audio_state = _prev_denoise_av(*args, **kwargs)
        # video_state here is post clear_conditioning + unpatchify -> (b,c,F,H,W).
        _EXTEND.captured_stage1_latent = video_state.latent.detach().clone()
        # Disarm so Stage 2 (and any later call) runs stock.
        _EXTEND.armed = False
        _EXTEND.inject_initial_latent = None
        _EXTEND.overlap_frames = 0
        return video_state, audio_state

    _tools.VideoLatentTools.create_initial_state = _patched_create_initial_state
    _distilled.denoise_audio_video = _patched_denoise_av
    _EXTEND_PATCHES_INSTALLED = True
    import logging as _lg
    _lg.getLogger(__name__).info(
        "installed extend monkeypatches (create_initial_state, denoise_audio_video)"
    )


class LTXFastVideoPipeline:
    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        transformer_device: torch.device | None = None,
        block_swap_blocks_on_gpu: int = 0,
        use_fp8_transformer: bool = False,
        gguf_transformer_path: str = "",
        gguf_per_layer_quant: bool = True,
        vae_spatial_tile_size: int = 0,
        vae_temporal_tile_size: int = 0,
        gguf_gemma_path: str = "",
        keep_resident_weights: bool = False,
        use_component_files: bool = False,
        component_video_vae_path: str = "",
        component_audio_vae_path: str = "",
        component_text_projection_path: str = "",
        te_offload_text_encoder: bool = True,
        dit_cpu_load: bool = True,
    ) -> "LTXFastVideoPipeline":
        return LTXFastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            transformer_device=transformer_device,
            block_swap_blocks_on_gpu=block_swap_blocks_on_gpu,
            use_fp8_transformer=use_fp8_transformer,
            gguf_transformer_path=gguf_transformer_path,
            gguf_per_layer_quant=gguf_per_layer_quant,
            vae_spatial_tile_size=vae_spatial_tile_size,
            vae_temporal_tile_size=vae_temporal_tile_size,
            gguf_gemma_path=gguf_gemma_path,
            keep_resident_weights=keep_resident_weights,
            use_component_files=use_component_files,
            component_video_vae_path=component_video_vae_path,
            component_audio_vae_path=component_audio_vae_path,
            component_text_projection_path=component_text_projection_path,
            te_offload_text_encoder=te_offload_text_encoder,
            dit_cpu_load=dit_cpu_load,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        transformer_device: torch.device | None = None,
        block_swap_blocks_on_gpu: int = 0,
        use_fp8_transformer: bool = False,
        gguf_transformer_path: str = "",
        gguf_per_layer_quant: bool = True,
        vae_spatial_tile_size: int = 0,
        vae_temporal_tile_size: int = 0,
        gguf_gemma_path: str = "",
        keep_resident_weights: bool = False,
        use_component_files: bool = False,
        component_video_vae_path: str = "",
        component_audio_vae_path: str = "",
        component_text_projection_path: str = "",
        te_offload_text_encoder: bool = True,
        dit_cpu_load: bool = True,
    ) -> None:
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        # ── Fail-fast: this GGUF + component-file path must NOT silently fall
        # back to the 43GB monolith / 22.7GB QAT Gemma. Assert the load-bearing
        # standalone sources are all present BEFORE constructing DistilledPipeline
        # (whose lazy builders would otherwise glob the monolith on build()).
        _required = {
            "component_video_vae_path": component_video_vae_path,
            "component_audio_vae_path": component_audio_vae_path,
            "component_text_projection_path": component_text_projection_path,
            "gguf_transformer_path": gguf_transformer_path,
            "gguf_gemma_path": gguf_gemma_path,
        }
        _missing = [name for name, val in _required.items() if not val]
        if _missing:
            raise RuntimeError(
                "LTXFastVideoPipeline requires the GGUF + component-file sources "
                "(the monolith/QAT path is retired); missing/empty: "
                + ", ".join(_missing)
            )

        # Transformer device defaults to primary device if not set.
        self._transformer_device = transformer_device or device
        self._block_swap_blocks_on_gpu = block_swap_blocks_on_gpu
        self._gguf_transformer_path = gguf_transformer_path
        self._gguf_per_layer_quant = gguf_per_layer_quant
        self._vae_spatial_tile_size = vae_spatial_tile_size
        self._vae_temporal_tile_size = vae_temporal_tile_size
        # Phase 1 component-file re-sourcing (text projection path stored but NOT
        # wired here — that is Phase 2).
        self._component_video_vae_path = component_video_vae_path
        self._component_audio_vae_path = component_audio_vae_path
        self._component_text_projection_path = component_text_projection_path
        # TE per-layer offload: stream the GGUF-quantized Gemma decoder layers
        # CPU->GPU per window during encode (caps the ~15 GB encode peak). Default ON;
        # when OFF the Gemma layers are all GPU-resident (today's exact behavior).
        self._te_offload_text_encoder = te_offload_text_encoder
        # DiT CPU-resident build: build the transformer directly on CPU RAM and
        # move only non-block submodules to GPU, eliminating the ~16.9 GB load-time
        # GPU spike. Blocks stay on CPU for the existing block-swap streaming.
        # Default ON; when OFF the transformer is built on GPU then evicted
        # (today's exact behavior).
        self._dit_cpu_load = dit_cpu_load

        # FP8: use setting OR auto-detect CUDA support.
        # The pipeline (transformer/VAE) always runs on device (video GPU, cuda:0).
        use_fp8 = use_fp8_transformer or device_supports_fp8(device)

        # QAT gemma_root reclamation: pass gemma_root=None so the wheel's
        # ModelLedger.build_model_builders() skips its Gemma block entirely
        # (model_ledger.py:158-169) — no `model*.safetensors` glob, no shard paths in
        # model_path, no `text_encoder_builder`. We rebuild that builder ourselves in
        # _install_gemma_gguf without any Gemma shards (weights come from the GGUF).
        # The gemma_root dir now only needs the tokenizer files (~40MB), which we
        # still hand to the Gemma install below so its module_ops can load the
        # tokenizer/processor. Keep the original path (do NOT drop it).
        self._gemma_tokenizer_root = gemma_root
        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=checkpoint_path,
            gemma_root=None,
            spatial_upsampler_path=upsampler_path,
            loras=[],
            device=device,
            quantization=QuantizationPolicy.fp8_cast() if use_fp8 else None,
        )

        # ── Stage 3: load-once / keep-resident weights via StateDictRegistry ──
        # Wire a real StateDictRegistry into the ModelLedger so each submodel's CPU
        # state_dict is loaded ONCE (job1 warmup) and reused on every later job (HIT),
        # eliminating the per-job disk re-materialization that exhausts Windows commit
        # (virtual memory) and crashes the resident worker on job2+ (exit 139).
        # Must run BEFORE the GGUF/Gemma/block-swap installs below so those operate on
        # registry-aware builders; build_model_builders() rebuilds the (lazy) builders
        # carrying the new registry. A non-Dummy registry also flips ModelLedger
        # ._target_device() to CPU, so submodels build on CPU (cached) and move to GPU
        # out-of-place per generate (VRAM-neutral).
        if keep_resident_weights:
            from ltx_core.loader.registry import StateDictRegistry
            self.pipeline.model_ledger.registry = StateDictRegistry()
            self.pipeline.model_ledger.build_model_builders()

        # ── Re-source VIDEO VAE + AUDIO VAE/vocoder from standalone component files ──
        # Phase 1: drop the 46GB monolith for the VAE/audio builders by re-pointing
        # their model_path to small standalone files. Runs AFTER the keep-resident
        # registry wiring (so it operates on the registry-aware builders) and BEFORE
        # the transformer/Gemma GGUF installs. Gated on use_component_files + both
        # VAE paths present.
        if use_component_files and component_video_vae_path and component_audio_vae_path:
            self._install_component_sources(component_video_vae_path, component_audio_vae_path)

        # ── Install GGUF loader (replaces transformer weights source) ──
        if gguf_transformer_path:
            self._install_gguf(gguf_transformer_path, per_layer_quant=gguf_per_layer_quant)

        # ── Install Gemma GGUF text encoder (keep 24GB bf16 Gemma compressed on GPU) ──
        # GGUF keeps Gemma quantized in VRAM (~7.3GB Q4_K_M) with per-layer dequant —
        # fits the 16GB card. This is the only text-encoder path (the CPU text-encode
        # branch was removed as dead: the worker never requested it).
        if gguf_gemma_path:
            # Phase 2: when component files are enabled (and the connector GGUF +
            # projection file are present), re-source the Gemma text encoder's
            # non-Gemma monolith survivors off standalone files so the 46GB monolith
            # is no longer opened by ANY builder: aggregate_embed from the projection
            # file (replaces the monolith in model_path) and the 258 connectors
            # injected from the transformer GGUF. Both must be present to enable the
            # drop; otherwise the monolith-base path is unchanged.
            _gemma_component = (
                use_component_files
                and component_text_projection_path
                and gguf_transformer_path
            )
            self._install_gemma_gguf(
                gguf_gemma_path,
                gemma_tokenizer_root=self._gemma_tokenizer_root,
                component_text_projection_path=(
                    component_text_projection_path if _gemma_component else None
                ),
                connector_gguf_path=(
                    gguf_transformer_path if _gemma_component else None
                ),
                te_offload=self._te_offload_text_encoder,
            )

        # ── Install block swapping ──
        if block_swap_blocks_on_gpu > 0:
            self._install_block_swap(block_swap_blocks_on_gpu)

        # NOTE: attention-tiling and LoRA install branches (guarded by
        # attention_tile_size > 0 / loras) were removed during the engine
        # relocation: their services (AttentionTileService / LoraService) are not
        # part of the first-party engine keep-set, the worker never enables these
        # guards (both default off), and the current T2V/GGUF path never reaches
        # them. The now-dead `attention_tile_size` and `loras` constructor
        # parameters (no caller ever passed them) have also been removed.

    def _install_component_sources(self, video_vae_path: str, audio_vae_path: str) -> None:
        """Re-point the VAE/audio builders at standalone component files.

        Replaces the 46GB monolith as the weight source for the video VAE
        (decoder/encoder) and the audio VAE (decoder/encoder) + vocoder, leaving
        the transformer and text encoder untouched (text projection is Phase 2).

        VIDEO VAE file: keys use BARE prefixes ``decoder.* / encoder.* /
        per_channel_statistics.*`` (no ``vae.`` prefix). The fork's
        VAE_{DECODER,ENCODER}_COMFY_KEYS_FILTER expect ``vae.decoder.* /
        vae.encoder.* / vae.per_channel_statistics.*`` input. We CHAIN a
        prepend-``vae.`` SDOps BEFORE the existing filter so the final
        model-target keys are byte-for-byte identical to the monolith path:
            [match bare {decoder|encoder}./per_channel_statistics.,
             replace {decoder|encoder}. -> vae.{decoder|encoder}.,
             replace per_channel_statistics. -> vae.per_channel_statistics.]
          THEN  [existing VAE_*_COMFY_KEYS_FILTER]
        Chaining merges both ops' mappings into one SDOps; apply_to_key first
        validates against the (now bare-key) matchers, then applies all
        replacements in order (prepend, then strip), reproducing monolith keys.

        AUDIO VAE file: keys are already namespaced exactly like the monolith
        (``audio_vae.{decoder,encoder,per_channel_statistics}.* / vocoder.*``),
        so the existing model_sd_ops are preserved verbatim — only the path moves.

        Builders are frozen dataclasses, so we use dataclasses.replace and
        explicitly preserve the (possibly StateDictRegistry) registry field.
        """
        import dataclasses
        import logging
        from ltx_core.loader.sd_ops import SDOps
        from ltx_core.model.video_vae import (
            VAE_DECODER_COMFY_KEYS_FILTER,
            VAE_ENCODER_COMFY_KEYS_FILTER,
        )
        _log = logging.getLogger(__name__)

        def _prepend_vae(sub: str) -> SDOps:
            # sub: "decoder" or "encoder". Match the BARE file keys and prepend
            # "vae." so the downstream filter (which expects "vae.<sub>." /
            # "vae.per_channel_statistics.") matches and strips correctly.
            return (
                SDOps(name=f"PREPEND_VAE_{sub.upper()}")
                .with_matching(prefix=f"{sub}.")
                .with_matching(prefix="per_channel_statistics.")
                .with_replacement(f"{sub}.", f"vae.{sub}.")
                .with_replacement("per_channel_statistics.", "vae.per_channel_statistics.")
            )

        def _chain(prepend: SDOps, filt: SDOps) -> SDOps:
            return SDOps(name=f"{prepend.name}+{filt.name}", mapping=(*prepend.mapping, *filt.mapping))

        ledger = self.pipeline.model_ledger

        # Video VAE: re-point path + chain prepend BEFORE the existing filter.
        ledger.vae_decoder_builder = dataclasses.replace(
            ledger.vae_decoder_builder,
            model_path=video_vae_path,
            model_sd_ops=_chain(_prepend_vae("decoder"), VAE_DECODER_COMFY_KEYS_FILTER),
            registry=ledger.registry,
        )
        ledger.vae_encoder_builder = dataclasses.replace(
            ledger.vae_encoder_builder,
            model_path=video_vae_path,
            model_sd_ops=_chain(_prepend_vae("encoder"), VAE_ENCODER_COMFY_KEYS_FILTER),
            registry=ledger.registry,
        )

        # Audio VAE + vocoder: keys already match the monolith namespace; keep the
        # existing model_sd_ops, only move the path.
        ledger.audio_decoder_builder = dataclasses.replace(
            ledger.audio_decoder_builder, model_path=audio_vae_path, registry=ledger.registry,
        )
        ledger.audio_encoder_builder = dataclasses.replace(
            ledger.audio_encoder_builder, model_path=audio_vae_path, registry=ledger.registry,
        )
        ledger.vocoder_builder = dataclasses.replace(
            ledger.vocoder_builder, model_path=audio_vae_path, registry=ledger.registry,
        )

        _log.info(
            "Component sources installed: video VAE <- %s ; audio VAE/vocoder <- %s "
            "(text projection NOT wired — Phase 2)",
            video_vae_path, audio_vae_path,
        )

    def _install_gguf(self, gguf_path: str, per_layer_quant: bool = True) -> None:
        try:
            if per_layer_quant:
                from engine.gguf.quant_service import GGUFQuantLoaderService
                service = GGUFQuantLoaderService(gguf_path=gguf_path)
                service.install(self.pipeline.model_ledger)
                self._gguf_service = service
                import logging
                logging.getLogger(__name__).info(
                    "GGUF per-layer quant installed: weights stay compressed in VRAM (%s)", gguf_path
                )
            else:
                from engine.gguf.loader_service import GGUFLoaderService
                service = GGUFLoaderService(gguf_path=gguf_path)
                service.install(self.pipeline.model_ledger)
                self._gguf_service = service
                import logging
                logging.getLogger(__name__).info(
                    "GGUF load-time dequant installed (full BF16 in VRAM): %s", gguf_path
                )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "GGUF install failed (%s) — falling back to safetensors", exc
            )

    def _install_gemma_gguf(
        self,
        gguf_path: str,
        gemma_tokenizer_root: str | None = None,
        component_text_projection_path: str | None = None,
        connector_gguf_path: str | None = None,
        te_offload: bool = True,
    ) -> None:
        """Load the Gemma-3 text encoder from a quantized GGUF, per-layer dequant.

        Replaces the stock ~24GB bf16 full-GPU Gemma load (which overflows the 16GB
        card) with a ~7.3GB Q4_K_M GGUF kept compressed in VRAM. Each Gemma decoder
        Linear dequantizes its weight on-the-fly during the forward pass and frees
        the temporary bf16 tensor after the matmul. The LTX-side feature extractor
        and embedding connectors remain bf16, loaded from the distilled checkpoint
        and merged with the GGUF Gemma weights by GemmaGGUFQuantStateDictLoader.

        Mirrors _install_gguf but targets text_encoder_builder instead of
        transformer_builder. On failure, falls back to the stock GPU text encoder.

        ``gemma_tokenizer_root`` is the (tokenizer-only) gemma_root dir. Because we
        now build DistilledPipeline with gemma_root=None, the wheel does not create
        the text_encoder_builder; the service rebuilds it (shards excluded) and needs
        this dir to load the tokenizer/processor module_ops.
        """
        try:
            from engine.gemma.gguf_quant_service import GemmaGGUFQuantLoaderService
            service = GemmaGGUFQuantLoaderService(
                gguf_path=gguf_path,
                gemma_tokenizer_root=gemma_tokenizer_root,
                component_text_projection_path=component_text_projection_path,
                connector_gguf_path=connector_gguf_path,
                layer_offload=te_offload,
            )
            service.install(self.pipeline.model_ledger)
            if component_text_projection_path and connector_gguf_path:
                import logging
                logging.getLogger(__name__).info(
                    "Gemma component-files active: monolith dropped from text encoder "
                    "(aggregate_embed <- %s ; connectors <- %s)",
                    component_text_projection_path, connector_gguf_path,
                )
            self._gemma_gguf_service = service
            import logging
            logging.getLogger(__name__).info(
                "Gemma GGUF per-layer quant installed: Gemma stays compressed in VRAM (%s)",
                gguf_path,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Gemma GGUF install failed (%s) — falling back to stock GPU text encoder", exc
            )

    def _install_block_swap(self, blocks_on_gpu: int) -> None:
        try:
            from engine.transformer.block_swap_service import BlockSwapService
            from engine.transformer.dit_cpu_load_service import DitCpuLoadService
            service = BlockSwapService(
                blocks_on_gpu=blocks_on_gpu,
                device=self._transformer_device,
            )
            # Wrap model_ledger.transformer() persistently so block swap is
            # re-installed on every build (model_ledger never caches the model).
            original_transformer = self.pipeline.model_ledger.transformer
            ledger = self.pipeline.model_ledger
            # DiT CPU-resident builder (default ON): builds on CPU to avoid the
            # load-time GPU spike, then moves only non-block tensors to GPU.
            dit_service = DitCpuLoadService(self._transformer_device, service)

            def patched_transformer() -> torch.nn.Module:
                if not self._dit_cpu_load:
                    t = original_transformer()
                    service.install(t)
                    return t
                t = dit_service.build_cpu_resident(ledger, original_transformer)
                service.install(t)
                return t

            self.pipeline.model_ledger.transformer = patched_transformer
            self._block_swap_service = service
            import logging
            logging.getLogger(__name__).info(
                "BlockSwap configured: %d blocks on GPU", blocks_on_gpu
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "BlockSwap install failed (%s)", exc
            )

    @staticmethod
    def _make_sigma_subset(num_steps: int) -> list[float]:
        """Return a subset of the distilled sigma schedule for fewer denoising steps.

        The full schedule has 8 steps (9 sigma values).  For fewer steps we pick
        evenly spaced values from the full list, always including the first (1.0)
        and last (0.0) so the denoising range stays correct.
        """
        full = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
        n = len(full) - 1  # 8
        if num_steps >= n:
            return full
        if num_steps <= 1:
            return [full[0], full[-1]]
        return [full[round(i * n / num_steps)] for i in range(num_steps + 1)]

    @staticmethod
    def _make_stg_denoising_func(stg_scale: float, stg_block_index: int):
        """Return a drop-in replacement for simple_denoising_func that applies STG.

        STG (Spatio-Temporal Guidance) improves prompt adherence by running a
        second forward pass with a single transformer block's self-attention
        replaced by identity, then steering away from that degraded prediction:
            output = cond + stg_scale * (cond - perturbed)

        No negative prompt is needed — cost is ~2× per step (one extra pass).
        """
        import logging as _logging
        _stg_log = _logging.getLogger(__name__)

        def _stg_func(video_context, audio_context, transformer):
            _stg_log.info("STG denoising func invoked: scale=%.2f block=%d", stg_scale, stg_block_index)
            from ltx_core.components.guiders import MultiModalGuiderFactory, MultiModalGuiderParams
            from ltx_pipelines.utils.helpers import multi_modal_guider_factory_denoising_func

            video_params = MultiModalGuiderParams(
                cfg_scale=1.0,
                stg_scale=stg_scale,
                stg_blocks=[stg_block_index],
            )
            # No STG on audio — keep it simple / unperturbed.
            audio_params = MultiModalGuiderParams(
                cfg_scale=1.0,
                stg_scale=0.0,
                stg_blocks=[],
            )
            return multi_modal_guider_factory_denoising_func(
                video_guider_factory=MultiModalGuiderFactory.constant(video_params),
                audio_guider_factory=MultiModalGuiderFactory.constant(audio_params),
                v_context=video_context,
                a_context=audio_context,
                transformer=transformer,
            )

        return _stg_func

    @staticmethod
    def _make_linear_sigmas(num_steps: int) -> list[float]:
        """Return a linearly-spaced sigma schedule from 1.0 to 0.0.

        Community-reported to reduce metallic artifacts on the native LTX audio
        track versus the default compressed distilled schedule, by distributing
        denoising steps evenly across the full sigma range instead of clustering
        them near σ=1.0.
        """
        return [1.0 - i / num_steps for i in range(num_steps + 1)]

    @staticmethod
    def _make_linearquadratic_sigmas(num_steps: int) -> list[float]:
        """Return a LinearQuadratic sigma schedule.

        First half of steps is linear (coarse, high-noise region); second half is
        quadratic (fine, low-noise region).  The quadratic tail gives more step
        budget to the low-noise refinement region than the plain linear schedule.
        Params: threshold_noise=0.025, linear_steps=num_steps//2 (library defaults).
        """
        from ltx_core.components.schedulers import LinearQuadraticScheduler
        return LinearQuadraticScheduler().execute(steps=num_steps).tolist()

    @staticmethod
    def _make_beta_sigmas(num_steps: int) -> list[float]:
        """Return a Beta distribution sigma schedule (arXiv 2407.12173).

        Samples timesteps according to a Beta(0.6, 0.6) distribution, producing
        a bell-curve step density concentrated toward midrange noise levels.
        May return fewer than num_steps+1 values due to deduplication of
        identical timesteps — the Euler loop handles variable-length schedules.
        """
        from ltx_core.components.schedulers import BetaScheduler
        return BetaScheduler().execute(steps=num_steps).tolist()

    # ── Phase 3 slice-2 — clip-concatenation carry-tail (de)serialization ──────
    def _stage1_latent_frames(self, num_frames: int) -> int:
        """Stage-1 (lowres) latent-frame count for a pixel `num_frames` (8n+1).

        Temporal VAE compression factor = 8 with a causal +1 keyframe, matching
        the wheel's Patchifier (temporal patch size 1). Latent frames =
        (num_frames - 1) // 8 + 1.
        """
        return (num_frames - 1) // 8 + 1

    def _persist_carry_tail(
        self, stage1_latent: torch.Tensor, overlap_frames: int, out_path: str
    ) -> None:
        """torch.save the last K latent frames of the Stage-1 output as the carry.

        Format: {"tail": <cpu tensor (b,c,K,H,W)>, "overlap_frames": K,
        "shape": tuple, "clip_meta": {...}}. Saved on CPU so the app process
        (torch-less) can pass the path around without materializing CUDA.
        """
        import os as _os

        F_lat = int(stage1_latent.shape[2])
        k = max(1, min(int(overlap_frames), F_lat))
        tail = stage1_latent[:, :, F_lat - k :, :, :].detach().to("cpu").contiguous()
        payload = {
            "tail": tail,
            "overlap_frames": k,
            "shape": tuple(int(x) for x in tail.shape),
            "clip_meta": {
                "stage1_latent_frames": F_lat,
                "dtype": str(tail.dtype),
                "format_version": 1,
            },
        }
        _os.makedirs(_os.path.dirname(_os.path.abspath(out_path)) or ".", exist_ok=True)
        torch.save(payload, out_path)
        import logging as _lg
        _lg.getLogger(__name__).info(
            "extend: persisted carry tail %s (K=%d) -> %s",
            tuple(tail.shape), k, out_path,
        )

    def _load_carry_tail(self, prev_clip_latent_path: str, num_frames: int) -> torch.Tensor:
        """Load a carry tail and PAD it to the Stage-1 target shape for injection.

        The tail occupies latent frames [0..K-1]; frames [K..F-1] are zeros. The
        result MUST equal the Stage-1 target VideoLatentShape exactly (the wheel
        asserts this in create_initial_state). dtype/device are constrained to
        the pipeline (bfloat16 / cuda).
        """
        payload = torch.load(prev_clip_latent_path, map_location=self._transformer_device)
        tail = payload["tail"]
        if not isinstance(tail, torch.Tensor):
            raise RuntimeError(f"carry tail payload missing tensor 'tail': {prev_clip_latent_path}")
        tail = tail.to(device=self._transformer_device, dtype=torch.bfloat16)
        b, c, K = int(tail.shape[0]), int(tail.shape[1]), int(tail.shape[2])
        Hl, Wl = int(tail.shape[3]), int(tail.shape[4])
        F_lat = self._stage1_latent_frames(num_frames)
        if K > F_lat:
            raise RuntimeError(
                f"carry tail K={K} exceeds Stage-1 latent frames F={F_lat} "
                f"(num_frames={num_frames}); reduce overlap or increase clip length"
            )
        padded = torch.zeros((b, c, F_lat, Hl, Wl), dtype=torch.bfloat16, device=self._transformer_device)
        padded[:, :, :K, :, :] = tail
        # Sync the arm's overlap length to the actual carry K (guards against a
        # mismatched overlap_frames arg vs the persisted tail).
        _EXTEND.overlap_frames = min(_EXTEND.overlap_frames, K) if _EXTEND.overlap_frames > 0 else K
        import logging as _lg
        _lg.getLogger(__name__).info(
            "extend: loaded carry tail K=%d padded to Stage-1 shape %s from %s",
            K, tuple(padded.shape), prev_clip_latent_path,
        )
        return padded

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType,
        num_steps: int = 8,
        stg_scale: float = 0.0,
        stg_block_index: int = 19,
        sigma_schedule: str = "distilled",
        denoising_loop: str = "euler",
        ge_gamma: float = 2.0,
        res2s_bongmath: bool = False,
        res2s_bongmath_max_iter: int = 5,
        prev_clip_latent_path: str | None = None,
        overlap_frames: int = 2,
        overlap_strength: float = 0.5,
        carry_latent_out_path: str | None = None,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        import ltx_pipelines.distilled as _distilled_mod

        # Release any fragmented reserved-but-unused CUDA memory before the
        # pipeline runs.  The transformer denoising loop leaves scattered
        # allocations across cuda:0; without this the VAE decoder (which needs
        # a single ~316 MiB contiguous block) can hit OOM even when there is
        # nominally enough free VRAM.
        torch.cuda.empty_cache()

        # Temporarily patch module-level names that DistilledPipeline.__call__
        # reads at call time.  Always restore in the finally block.
        _orig_sigmas = _distilled_mod.DISTILLED_SIGMA_VALUES
        _orig_simple = _distilled_mod.simple_denoising_func
        _orig_euler = _distilled_mod.euler_denoising_loop

        # ── Keyframe conditioning hybrid ──────────────────────────────────────
        # The installed wheel's DistilledPipeline builds image conditionings for
        # BOTH stages via the module-global image_conditionings_by_replacing_latent,
        # which constructs VideoConditionByLatentIndex(latent_idx=img.frame_idx)
        # for EVERY image — treating our PIXEL frame_idx as a LATENT index.  For
        # frame_idx > 0 that overflows the latent token buffer and crashes
        # (RuntimeError: expanded size (0) must match existing size (40) at
        # latent_cond.py:40).  frame_idx == 0 coincides in both index spaces, so
        # single-image I2V worked and masked the bug.
        #
        # This replicates Lightricks' `combined_image_conditionings` (absent in
        # the installed wheel) WITHOUT a wheel upgrade: route per-image by
        # frame_idx — idx == 0 → latent REPLACE (the existing helper, unchanged);
        # idx > 0 → keyframe/guide APPEND via image_conditionings_by_adding_guiding_latent,
        # which builds VideoConditionByKeyframeIndex(frame_idx=img.frame_idx) as a
        # PIXEL RoPE offset (no ÷8).  frame_idx is passed through as-is (already
        # snapped to a multiple of 8 by the API validator).  Both Stage 1 and
        # Stage 2 pick this up because DistilledPipeline reads the module-global
        # name — the same LOAD_GLOBAL mechanism as DISTILLED_SIGMA_VALUES above.
        #
        # DESIGN INVARIANT: images with frame_idx == 0 (and T2V with no images)
        # go through _orig_replace exactly as before → byte-identical to today.
        _orig_replace = _distilled_mod.image_conditionings_by_replacing_latent
        # The guide helper is NOT imported into distilled.py's namespace, so we
        # reference it from the helpers module (its canonical home).  Both helpers
        # share the identical signature (images, height, width, video_encoder,
        # dtype, device) and both return list[ConditioningItem], so delegation +
        # list concatenation is exact.
        from ltx_pipelines.utils.helpers import (
            image_conditionings_by_adding_guiding_latent as _orig_add_guide,
        )

        def _hybrid_image_conditionings(images: list, *args: object, **kwargs: object) -> list:
            replace_imgs = [im for im in images if im.frame_idx == 0]
            guide_imgs = [im for im in images if im.frame_idx > 0]
            conds: list = []
            if replace_imgs:
                conds += _orig_replace(replace_imgs, *args, **kwargs)
            if guide_imgs:
                conds += _orig_add_guide(guide_imgs, *args, **kwargs)
            return conds

        # ── Sigma schedule ───────────────────────────────────────────────────
        if sigma_schedule == "linear":
            _distilled_mod.DISTILLED_SIGMA_VALUES = self._make_linear_sigmas(num_steps)  # type: ignore[attr-defined]
        elif sigma_schedule == "linear_quadratic":
            _distilled_mod.DISTILLED_SIGMA_VALUES = self._make_linearquadratic_sigmas(num_steps)  # type: ignore[attr-defined]
        elif sigma_schedule == "beta":
            _distilled_mod.DISTILLED_SIGMA_VALUES = self._make_beta_sigmas(num_steps)  # type: ignore[attr-defined]
        elif num_steps < 8:
            _distilled_mod.DISTILLED_SIGMA_VALUES = self._make_sigma_subset(num_steps)  # type: ignore[attr-defined]

        # ── STG guidance function ─────────────────────────────────────────────
        if stg_scale > 0.0:
            _distilled_mod.simple_denoising_func = self._make_stg_denoising_func(stg_scale, stg_block_index)  # type: ignore[attr-defined]

        # ── Denoising loop ───────────────────────────────────────────────────
        # "gradient_estimating" applies velocity correction across consecutive
        # steps (paper: openreview.net/pdf?id=o2ND9v0CeK).  We patch the loop
        # name in the distilled module so DistilledPipeline.__call__ picks it
        # up — same LOAD_GLOBAL mechanism used for DISTILLED_SIGMA_VALUES above.
        if denoising_loop == "gradient_estimating":
            from functools import partial
            from ltx_pipelines.utils.samplers import gradient_estimating_euler_denoising_loop
            _distilled_mod.euler_denoising_loop = partial(  # type: ignore[attr-defined]
                gradient_estimating_euler_denoising_loop, ge_gamma=ge_gamma
            )

        elif denoising_loop == "res2s":
            # res2s is a second-order Runge-Kutta sampler with SDE noise injection.
            # distilled.py hardcodes EulerDiffusionStep as its stepper, so we replace
            # the euler_denoising_loop name at module level with a wrapper that
            # substitutes Res2sDiffusionStep and calls res2s instead.
            # Cost: 2× model evaluations per step vs Euler.
            # Benefit: may match Euler quality at half the step count; SDE noise
            # can break up deterministic artifacts.
            from functools import partial
            from ltx_core.components.diffusion_steps import Res2sDiffusionStep
            from ltx_pipelines.utils.samplers import res2s_audio_video_denoising_loop

            _res2s_stepper = Res2sDiffusionStep()

            def _res2s_as_euler(
                sigmas: torch.Tensor,
                video_state: object,
                audio_state: object,
                stepper: object,  # EulerDiffusionStep from distilled.py — ignored
                denoise_fn: object,
                **_kwargs: object,
            ) -> object:
                return res2s_audio_video_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,  # type: ignore[arg-type]
                    audio_state=audio_state,  # type: ignore[arg-type]
                    stepper=_res2s_stepper,
                    denoise_fn=denoise_fn,  # type: ignore[arg-type]
                    noise_seed=seed,
                    bongmath=res2s_bongmath,
                    bongmath_max_iter=res2s_bongmath_max_iter,
                )

            _distilled_mod.euler_denoising_loop = _res2s_as_euler  # type: ignore[attr-defined]

        # ── Keyframe conditioning hybrid rebind ───────────────────────────────
        # Rebind the module-global so both Stage 1 and Stage 2 route non-zero
        # frame_idx images to the keyframe/guide helper (restored in finally).
        _distilled_mod.image_conditionings_by_replacing_latent = _hybrid_image_conditionings  # type: ignore[attr-defined]

        # ── Phase 3 slice-2 — clip-concatenation / latent-level extend arm ─────
        # Active ONLY when extend inputs are present: consuming a prev clip's
        # carry tail (prev_clip_latent_path) OR persisting this clip's Stage-1
        # tail (carry_latent_out_path). When neither is set the wrappers stay
        # disarmed and the run is byte-identical to today.
        extend_active = bool(prev_clip_latent_path) or bool(carry_latent_out_path)
        if extend_active:
            _install_extend_patches()
            _EXTEND.reset()
            _EXTEND.armed = True
            # overlap_strength in [0,1]; overlap-frame denoise_mask value =
            # 1.0 - overlap_strength (strength=1.0 -> mask 0.0 hard freeze;
            # default 0.5 -> soft, per prior-art recommendation).
            _EXTEND.overlap_frames = max(0, int(overlap_frames))
            _EXTEND.overlap_mask_value = 1.0 - max(0.0, min(1.0, float(overlap_strength)))
            if prev_clip_latent_path:
                _EXTEND.inject_initial_latent = self._load_carry_tail(
                    prev_clip_latent_path, num_frames=num_frames,
                )
            else:
                # No prev clip to consume (this is the FIRST clip in a chain):
                # do not soften any frames, just capture the Stage-1 tail.
                _EXTEND.overlap_frames = 0

        try:
            return self.pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
                tiling_config=tiling_config,
            )
        finally:
            # Synchronize before restoring module state so any async CUDA ops
            # queued inside DistilledPipeline.__call__ (e.g. vae_decode_audio)
            # are fully complete before Python GC frees the pipeline's locals.
            # Without this, dual-conditioning (2× CUDA tensors freed at __call__
            # return) can race with in-flight CUDA work → 0xC0000005 on Windows.
            torch.cuda.synchronize()
            _distilled_mod.DISTILLED_SIGMA_VALUES = _orig_sigmas
            _distilled_mod.simple_denoising_func = _orig_simple
            _distilled_mod.euler_denoising_loop = _orig_euler
            _distilled_mod.image_conditionings_by_replacing_latent = _orig_replace  # type: ignore[attr-defined]

            # ── Extend: persist Stage-1 tail, then ALWAYS disarm (no leak) ─────
            # The denoise wrapper disarms `.armed` after Stage 1, but reset here
            # unconditionally so a crash mid-run cannot leave the process armed
            # for the next job (the chain endpoint reuses one worker process).
            if extend_active:
                try:
                    if carry_latent_out_path and _EXTEND.captured_stage1_latent is not None:
                        self._persist_carry_tail(
                            _EXTEND.captured_stage1_latent,
                            overlap_frames=max(0, int(overlap_frames)),
                            out_path=carry_latent_out_path,
                        )
                finally:
                    _EXTEND.reset()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
        num_steps: int = 8,
        stg_scale: float = 0.0,
        stg_block_index: int = 19,
        sigma_schedule: str = "distilled",
        denoising_loop: str = "euler",
        ge_gamma: float = 2.0,
        res2s_bongmath: bool = False,
        res2s_bongmath_max_iter: int = 5,
        prev_clip_latent_path: str | None = None,
        overlap_frames: int = 2,
        overlap_strength: float = 0.5,
        carry_latent_out_path: str | None = None,
    ) -> None:
        tiling_config = default_tiling_config(
            spatial_tile_size=self._vae_spatial_tile_size,
            temporal_tile_size=self._vae_temporal_tile_size,
        )
        video, audio = self._run_inference(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
            num_steps=num_steps,
            stg_scale=stg_scale,
            stg_block_index=stg_block_index,
            sigma_schedule=sigma_schedule,
            denoising_loop=denoising_loop,
            ge_gamma=ge_gamma,
            res2s_bongmath=res2s_bongmath,
            res2s_bongmath_max_iter=res2s_bongmath_max_iter,
            prev_clip_latent_path=prev_clip_latent_path,
            overlap_frames=overlap_frames,
            overlap_strength=overlap_strength,
            carry_latent_out_path=carry_latent_out_path,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)
        # Synchronize BEFORE freeing GPU tensors so any CUDA ops still queued
        # inside encode_video (tiled VAE decode iterator, audio write) are fully
        # complete before Python GC can reclaim the underlying CUDA memory.
        # del-then-sync is wrong: the tensor backing memory is freed immediately
        # on del (refcount → 0) while CUDA is still accessing it → 0xC0000005.
        torch.cuda.synchronize()
        del video, audio
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_chain(
        self,
        clips: "list",
        width: int,
        height: int,
        frame_rate: float,
        num_steps: int,
        seed: int,
        overlap_frames: int,
        overlap_strength: float,
        output_path: str,
        progress=None,
    ) -> dict:
        """Masked AV-latent clip chaining -> ONE continuous mp4 (Phase 3 WP4).

        Delegates to :func:`engine.pipeline.chain_pipeline.run_chain`, which
        reuses THIS pipeline's ledger/components/low-VRAM machinery. ``clips`` is
        a list of ``ChainClipSpec`` (prompt already resolved, images built).
        Returns metadata incl. segment/tile junction pixel-frame indices.
        """
        from engine.pipeline.chain_pipeline import run_chain

        return run_chain(
            self,
            clips=clips,
            width=width,
            height=height,
            frame_rate=frame_rate,
            num_steps=num_steps,
            seed=seed,
            overlap_frames=overlap_frames,
            overlap_strength=overlap_strength,
            output_path=output_path,
            progress=progress,
        )

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9
        tiling_config = default_tiling_config()

        try:
            video, audio = self._run_inference(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                tiling_config=tiling_config,
            )
            chunks = video_chunks_number(warmup_frames, tiling_config)
            encode_video_output(video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        transformer = self.pipeline.model_ledger.transformer()

        compiled = cast(
            torch.nn.Module,
            torch.compile(transformer, mode="reduce-overhead", fullgraph=False),
        )
        setattr(self.pipeline.model_ledger, "transformer", lambda: compiled)