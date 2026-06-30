"""LTX fast video pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Final, cast

import torch

from api_types import ImageConditioningInput
from services.lora_service import LoraEntry
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8


class _CPUTextEncoderWrapper:
    """Thin wrapper that runs a CPU-resident Gemma encoder and returns the
    resulting embeddings on the GPU.

    `encode_text` calls `text_encoder(prompt)` and unpacks the result as
    `v_context, a_context, _ = text_encoder(prompt)`. The wrapped GemmaTextEncoder
    runs entirely on CPU (its forward dispatches to `self.model.device`); we cast
    only the small GemmaEncoderOutput tensors (video/audio encodings + mask) to
    (gpu_device, out_dtype) so the GPU transformer receives them in the expected
    bf16/cuda form. Heavy Gemma weights never touch the GPU.
    """

    def __init__(self, encoder: torch.nn.Module, gpu_device: torch.device, out_dtype: torch.dtype) -> None:
        self._encoder = encoder
        self._gpu_device = gpu_device
        self._out_dtype = out_dtype

    def _to_gpu(self, t: "torch.Tensor | None") -> "torch.Tensor | None":
        if t is None:
            return None
        # Floating-point embeddings → bf16; keep integer/bool masks in their
        # own dtype, only move the device.
        if t.is_floating_point():
            return t.to(device=self._gpu_device, dtype=self._out_dtype)
        return t.to(device=self._gpu_device)

    def __call__(self, prompt: str):
        out = self._encoder(prompt)  # GemmaEncoderOutput(video, audio, mask) on CPU
        video_enc, audio_enc, mask = out
        return (self._to_gpu(video_enc), self._to_gpu(audio_enc), self._to_gpu(mask))

    def __getattr__(self, name: str):
        # Delegate any other attribute access (e.g. tokenizer/processor used by
        # optional prompt-enhancement paths) to the underlying encoder.
        return getattr(self._encoder, name)


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
        attention_tile_size: int = 0,
        use_fp8_transformer: bool = False,
        gguf_transformer_path: str = "",
        gguf_per_layer_quant: bool = True,
        vae_spatial_tile_size: int = 0,
        vae_temporal_tile_size: int = 0,
        pre_quantized_transformer_path: str = "",
        loras: list[LoraEntry] | None = None,
        cpu_text_encode: bool = False,
        gguf_gemma_path: str = "",
        keep_resident_weights: bool = False,
        use_component_files: bool = False,
        component_video_vae_path: str = "",
        component_audio_vae_path: str = "",
        component_text_projection_path: str = "",
        te_offload_text_encoder: bool = True,
    ) -> "LTXFastVideoPipeline":
        return LTXFastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            transformer_device=transformer_device,
            block_swap_blocks_on_gpu=block_swap_blocks_on_gpu,
            attention_tile_size=attention_tile_size,
            use_fp8_transformer=use_fp8_transformer,
            gguf_transformer_path=gguf_transformer_path,
            gguf_per_layer_quant=gguf_per_layer_quant,
            vae_spatial_tile_size=vae_spatial_tile_size,
            vae_temporal_tile_size=vae_temporal_tile_size,
            pre_quantized_transformer_path=pre_quantized_transformer_path,
            loras=loras,
            cpu_text_encode=cpu_text_encode,
            gguf_gemma_path=gguf_gemma_path,
            keep_resident_weights=keep_resident_weights,
            use_component_files=use_component_files,
            component_video_vae_path=component_video_vae_path,
            component_audio_vae_path=component_audio_vae_path,
            component_text_projection_path=component_text_projection_path,
            te_offload_text_encoder=te_offload_text_encoder,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        transformer_device: torch.device | None = None,
        block_swap_blocks_on_gpu: int = 0,
        attention_tile_size: int = 0,
        use_fp8_transformer: bool = False,
        gguf_transformer_path: str = "",
        gguf_per_layer_quant: bool = True,
        vae_spatial_tile_size: int = 0,
        vae_temporal_tile_size: int = 0,
        pre_quantized_transformer_path: str = "",
        loras: list[LoraEntry] | None = None,
        cpu_text_encode: bool = False,
        gguf_gemma_path: str = "",
        keep_resident_weights: bool = False,
        use_component_files: bool = False,
        component_video_vae_path: str = "",
        component_audio_vae_path: str = "",
        component_text_projection_path: str = "",
        te_offload_text_encoder: bool = True,
    ) -> None:
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        # Transformer device defaults to primary device if not set.
        self._transformer_device = transformer_device or device
        self._block_swap_blocks_on_gpu = block_swap_blocks_on_gpu
        self._attention_tile_size = attention_tile_size
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

        # FP8: use setting OR auto-detect CUDA support.
        # The pipeline (transformer/VAE) always runs on device (video GPU, cuda:0).
        use_fp8 = use_fp8_transformer or device_supports_fp8(device)

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=checkpoint_path,
            gemma_root=cast(str, gemma_root),
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

        # ── Swap in pre-quantized FP8 transformer (faster load, no on-the-fly downcast) ──
        # Skip if GGUF is configured — GGUF provides its own transformer weights.
        if use_fp8 and pre_quantized_transformer_path and os.path.exists(pre_quantized_transformer_path) and not gguf_transformer_path:
            self._install_pre_quantized_transformer(pre_quantized_transformer_path)

        # ── Install GGUF loader (replaces transformer weights source) ──
        if gguf_transformer_path:
            self._install_gguf(gguf_transformer_path, per_layer_quant=gguf_per_layer_quant)

        # ── Install Gemma GGUF text encoder (keep 24GB bf16 Gemma compressed on GPU) ──
        # Mutually exclusive with cpu_text_encode: both target model_ledger.text_encoder.
        # GGUF keeps Gemma quantized in VRAM (~7.3GB Q4_K_M) with per-layer dequant —
        # faster than CPU encode while still fitting the 16GB card. Prefer GGUF if both
        # are requested.
        if gguf_gemma_path and cpu_text_encode:
            import logging
            logging.getLogger(__name__).warning(
                "Both gguf_gemma_path and cpu_text_encode set — preferring GGUF "
                "(per-layer quant on GPU); ignoring cpu_text_encode."
            )
            cpu_text_encode = False

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
                component_text_projection_path=(
                    component_text_projection_path if _gemma_component else None
                ),
                connector_gguf_path=(
                    gguf_transformer_path if _gemma_component else None
                ),
                te_offload=self._te_offload_text_encoder,
            )

        # ── Install CPU text encoder (keep 24GB bf16 Gemma off the GPU) ──
        # Wraps model_ledger.text_encoder; independent of the transformer
        # wrappers below (block_swap/LoRA), so it composes without conflict.
        if cpu_text_encode:
            self._install_cpu_text_encoder()

        # ── Install block swapping ──
        if block_swap_blocks_on_gpu > 0:
            self._install_block_swap(block_swap_blocks_on_gpu)

        # ── Install attention tiling ──
        if attention_tile_size > 0:
            self._install_attention_tiling(attention_tile_size)

        # ── Apply LoRAs ──
        if loras:
            self._install_loras(loras)

    def _install_pre_quantized_transformer(self, fp8_path: str) -> None:
        """Replace the transformer builder with a pre-quantized FP8 file loader.

        The pre-quantized file contains LTXModel state dict (velocity_model keys,
        already renamed, already fp8).  No ComfyUI renaming or fp8 downcast needed —
        only UPCAST_DURING_INFERENCE to patch nn.Linear.forward at inference time.
        """
        try:
            from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
            from ltx_core.model.transformer import LTXModelConfigurator
            from ltx_core.quantization import QuantizationPolicy, UPCAST_DURING_INFERENCE
            import logging
            _log = logging.getLogger(__name__)

            ledger = self.pipeline.model_ledger
            ledger.transformer_builder = SingleGPUModelBuilder(
                model_class_configurator=LTXModelConfigurator,
                model_path=fp8_path,
                model_sd_ops=None,  # keys already in LTX format, no renaming needed
                registry=ledger.registry,
            )
            ledger.quantization = QuantizationPolicy(
                sd_ops=None,           # already fp8, no downcast needed
                module_ops=(UPCAST_DURING_INFERENCE,),
            )
            _log.info("Pre-quantized FP8 transformer installed from %s", fp8_path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Pre-quantized FP8 install failed (%s) — falling back to on-the-fly fp8_cast", exc
            )

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
                from services.gguf_quant_service import GGUFQuantLoaderService
                service = GGUFQuantLoaderService(gguf_path=gguf_path)
                service.install(self.pipeline.model_ledger)
                self._gguf_service = service
                import logging
                logging.getLogger(__name__).info(
                    "GGUF per-layer quant installed: weights stay compressed in VRAM (%s)", gguf_path
                )
            else:
                from services.gguf_loader_service import GGUFLoaderService
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
        """
        try:
            from services.gemma_gguf_quant_service import GemmaGGUFQuantLoaderService
            service = GemmaGGUFQuantLoaderService(
                gguf_path=gguf_path,
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

    def _install_cpu_text_encoder(self) -> None:
        """Run the ~24GB bf16 Gemma text encoder on CPU, keeping it off the GPU.

        The stock ModelLedger.text_encoder() does
            text_encoder_builder.build(device=cuda, dtype=bf16).to(cuda)
        which loads the full Gemma3 model onto the 16GB GPU and forces a ~17.7GB
        WDDM spillover into shared system RAM during text encoding.

        Here we wrap model_ledger.text_encoder so the encoder is built on CPU
        (bf16, no quantization → ~24.4GB RAM, well within the 47GB free) and the
        final `.to(cuda)` is skipped. Gemma's forward then runs on CPU, and only
        the small GemmaEncoderOutput embeddings are cast to (cuda, bf16) for the
        GPU transformer. The downstream `del text_encoder; cleanup_memory()` in
        DistilledPipeline.__call__ still frees the CPU weights normally.
        """
        try:
            ledger = self.pipeline.model_ledger
            gpu_device = self._transformer_device
            dtype = ledger.dtype  # bf16

            def cpu_text_encoder():
                # Build Gemma directly on CPU/bf16. Do NOT call the stock
                # ledger.text_encoder(), whose trailing .to(self.device) would
                # move the 24GB model onto the GPU.
                enc = ledger.text_encoder_builder.build(
                    device=torch.device("cpu"), dtype=dtype
                ).eval()
                return _CPUTextEncoderWrapper(enc, gpu_device=gpu_device, out_dtype=dtype)

            ledger.text_encoder = cpu_text_encoder  # type: ignore[method-assign]
            import logging
            logging.getLogger(__name__).info(
                "CPU text encoder installed: Gemma runs on CPU (bf16), embeddings cast to %s/%s",
                gpu_device, dtype,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "CPU text encoder install failed (%s) — falling back to GPU text encode", exc
            )

    def _install_block_swap(self, blocks_on_gpu: int) -> None:
        try:
            from services.block_swap_service import BlockSwapService
            service = BlockSwapService(
                blocks_on_gpu=blocks_on_gpu,
                device=self._transformer_device,
            )
            # Wrap model_ledger.transformer() persistently so block swap is
            # re-installed on every build (model_ledger never caches the model).
            original_transformer = self.pipeline.model_ledger.transformer

            def patched_transformer() -> torch.nn.Module:
                t = original_transformer()
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

    def _install_attention_tiling(self, tile_size: int) -> None:
        try:
            from services.attention_tile_service import AttentionTileService
            service = AttentionTileService(tile_size=tile_size)
            service.install()
            self._attention_tile_service = service
            import logging
            logging.getLogger(__name__).info(
                "AttentionTiling installed: tile_size=%d", tile_size
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "AttentionTiling install failed (%s)", exc
            )

    def _install_loras(self, entries: list[LoraEntry]) -> None:
        import logging
        _log = logging.getLogger(__name__)
        try:
            from services.lora_service import LoraService
            service = LoraService(device=self._transformer_device)
            loaded = service.load_loras(entries)
            if not loaded:
                _log.warning("No LoRAs were successfully loaded")
                return

            # Wrap model_ledger.transformer() persistently so LoRA hooks are
            # re-applied on every build (model_ledger never caches the model).
            # At this point model_ledger.transformer may already be wrapped by
            # _install_block_swap, so we chain on top of that.
            _original_transformer_fn = self.pipeline.model_ledger.transformer

            def _transformer_with_loras() -> torch.nn.Module:
                t = _original_transformer_fn()
                service.apply_hooks_to_transformer(t, loaded)
                return t

            self.pipeline.model_ledger.transformer = _transformer_with_loras
            _log.info("LoRAs applied: %d loaded", len(loaded))
        except Exception as exc:
            _log.warning("LoRA install failed (%s)", exc)

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