"""LTX fast video pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Final, cast

import torch

from engine.api_types import ImageConditioningInput
from engine.pipeline.common import default_tiling_config, encode_video_output, video_chunks_number
from engine.pipeline.utils import AudioOrNone, TilingConfigType, device_supports_fp8


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
        *,
        ic_loras: list[tuple[str, float]] | None = None,
        ic_reference: tuple[str, float] | None = None,
        ic_attention_strength: float = 1.0,
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
            ic_loras=ic_loras,
            ic_reference=ic_reference,
            ic_attention_strength=ic_attention_strength,
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
        *,
        ic_loras: list[tuple[str, float]] | None = None,
        ic_reference: tuple[str, float] | None = None,
        ic_attention_strength: float = 1.0,
    ) -> None:
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        # ── IC-LoRA state (all inert by default) ──────────────────────────────
        # ic_loras: (safetensors_path, strength) LoRAs applied to the GGUF base
        #   transformer. Phase A fused them into the full BF16 state-dict at load
        #   (bf16 path); Phase B adds them at FORWARD time on the per-layer-quant
        #   path (GGUFQuantLoaderService + ggml_linear_forward). Selectable via
        #   gguf_per_layer_quant.
        # ic_reference: (reference_video_path, strength) appended as a
        #   VideoConditionByReferenceLatent on the stage-1 conditioning pass.
        # When both are None/empty every changed path is byte-identical to before.
        #
        # These are CREATE-TIME DEFAULTS. generate(ic_loras=..., ic_reference=...)
        # overrides them per job (keep_resident=0 rebuilds the transformer every
        # job, so the forward-time attach reads the live values). The live values
        # are held in self._ic_loras / self._ic_reference / the resolved factor.
        # ic_attention_strength: IC-LoRA control-adherence knob (0..1, default 1.0).
        #   Forwarded to a ConditioningItemAttentionStrengthWrapper around the
        #   reference conditioning ONLY when < 1.0 (upstream iclora_utils parity);
        #   at 1.0 no wrapper is added → structurally byte-identical to before.
        self._ic_loras_default: list[tuple[str, float]] = list(ic_loras or [])
        self._ic_reference_default: tuple[str, float] | None = ic_reference
        self._ic_attention_strength_default: float = float(ic_attention_strength)
        self._ic_loras: list[tuple[str, float]] = []
        self._ic_reference: tuple[str, float] | None = None
        self._ic_attention_strength: float = 1.0
        self._ic_reference_downscale_factor: int | None = None
        self._set_ic_job(
            self._ic_loras_default,
            self._ic_reference_default,
            self._ic_attention_strength_default,
        )

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
            self._install_gguf(
                gguf_transformer_path,
                per_layer_quant=gguf_per_layer_quant,
                ic_loras=self._ic_loras,
            )

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

    def _set_ic_job(
        self,
        ic_loras: list[tuple[str, float]] | None,
        ic_reference: tuple[str, float] | None,
        ic_attention_strength: float = 1.0,
    ) -> None:
        """Set the live IC-LoRA state for the upcoming build/generate.

        Called from __init__ (create-time defaults) and from generate() (per-job
        override). Recomputes the reference downscale factor from the LoRA
        metadata and re-validates the ``ic_reference requires ic_loras`` contract.
        The forward-time attach reads ``self._ic_loras`` via the provider on the
        next transformer build; the reference conditioning reads
        ``self._ic_reference`` / the resolved factor / the attention strength.
        """
        self._ic_loras = list(ic_loras or [])
        self._ic_reference = ic_reference
        self._ic_attention_strength = float(ic_attention_strength)
        self._ic_reference_downscale_factor = None
        if self._ic_reference is None:
            return
        if not self._ic_loras:
            raise RuntimeError(
                "ic_reference set but no ic_loras — the reference downscale "
                "factor is read from the LoRA metadata; supply the IC-LoRA."
            )
        # Reuse the wheel's own metadata reader (private, but we already
        # monkeypatch this wheel). It returns 1 when the metadata key is
        # absent; for an x2 upscaler LoRA a factor of 1 means the metadata is
        # missing — fail loudly rather than silently running at native res.
        from ltx_pipelines.ic_lora import _read_lora_reference_downscale_factor
        factor = _read_lora_reference_downscale_factor(self._ic_loras[0][0])
        if factor <= 1:
            raise RuntimeError(
                f"IC-LoRA {self._ic_loras[0][0]} reports reference_downscale_factor="
                f"{factor}; expected >1 (metadata missing?). Refusing to run "
                "reference conditioning at factor 1."
            )
        self._ic_reference_downscale_factor = factor

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

    def _install_gguf(
        self,
        gguf_path: str,
        per_layer_quant: bool = True,
        ic_loras: list[tuple[str, float]] | None = None,
    ) -> None:
        ic_loras = list(ic_loras or [])
        try:
            if per_layer_quant:
                # Phase B: the per-layer-quant path applies IC-LoRA at FORWARD
                # time (ggml_linear_forward adds the fp32 delta onto the fresh
                # per-call dequant tensor). We hand the service a provider that
                # returns the CURRENT job's adapters (self._ic_loras), read on
                # every transformer build — so keep_resident=0 can toggle LoRAs
                # per generate() without mutating any compressed/cached bytes.
                from engine.gguf.quant_service import GGUFQuantLoaderService
                service = GGUFQuantLoaderService(
                    gguf_path=gguf_path,
                    ic_loras_provider=lambda: self._ic_loras,
                )
                service.install(self.pipeline.model_ledger)
                self._gguf_service = service
                import logging
                logging.getLogger(__name__).info(
                    "GGUF per-layer quant installed: weights stay compressed in VRAM "
                    "(%s); IC-LoRA applied at forward time (per-job)", gguf_path
                )
            else:
                from engine.gguf.loader_service import GGUFLoaderService
                service = GGUFLoaderService(gguf_path=gguf_path, ic_loras=ic_loras)
                service.install(self.pipeline.model_ledger)
                self._gguf_service = service
                import logging
                logging.getLogger(__name__).info(
                    "GGUF load-time dequant installed (full BF16 in VRAM): %s%s",
                    gguf_path,
                    f" + {len(ic_loras)} IC-LoRA(s)" if ic_loras else "",
                )
        except Exception as exc:
            # Fail-loud when LoRAs were requested: a silent safetensors fallback
            # would produce a plausible-but-wrong (no-LoRA) result and corrupt the
            # spike measurement. With no LoRAs, preserve the historical fallback.
            if ic_loras:
                raise
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

    def _reference_conditioning_for_stage(
        self, full_height: int, num_frames: int, cond_kwargs: dict
    ) -> list:
        """Build the IC-LoRA reference-video conditioning for the current stage.

        Replicates ICLoraPipeline._create_conditionings' reference branch
        (ltx_pipelines/ic_lora.py): load the reference video at
        ``target // downscale_factor`` resolution, VAE-encode it, and wrap it in a
        VideoConditionByReferenceLatent. Returns [] on any pass that is not
        stage 1, so the reference latent is added exactly once.

        Stage discriminator: DistilledPipeline builds stage-1 conditionings at
        HALF resolution (distilled.py stage_1_output_shape uses height//2) and
        stage-2 at full resolution. ``height`` is the least-fragile signal here —
        it is the only per-stage-differing argument passed to this conditioning
        function (num_frames and any stage index are not forwarded to it).
        """
        cond_height = int(cond_kwargs["height"])
        if cond_height != full_height // 2:
            return []  # stage 2 (or unexpected res) — reference added at stage 1 only

        from ltx_core.conditioning import (
            ConditioningItemAttentionStrengthWrapper,
            VideoConditionByReferenceLatent,
        )
        from ltx_pipelines.utils.media_io import load_video_conditioning

        ref_path, ref_strength = self._ic_reference
        scale = self._ic_reference_downscale_factor
        assert scale is not None and scale > 1, "reference downscale factor not initialised"

        cond_width = int(cond_kwargs["width"])
        video_encoder = cond_kwargs["video_encoder"]
        dtype = cond_kwargs["dtype"]
        device = cond_kwargs["device"]

        if cond_height % scale != 0 or cond_width % scale != 0:
            raise ValueError(
                f"Stage-1 dims ({cond_height}x{cond_width}) must be divisible by "
                f"reference_downscale_factor ({scale})"
            )
        ref_height = cond_height // scale
        ref_width = cond_width // scale

        video = load_video_conditioning(
            video_path=ref_path,
            height=ref_height,
            width=ref_width,
            frame_cap=num_frames,
            dtype=dtype,
            device=device,
        )
        encoded_video = video_encoder(video)
        # Control-adherence knob (upstream iclora_utils parity): only when the
        # attention strength is < 1.0 do we wrap the reference conditioning so a
        # scalar additive self-attention mask reaches SDPA. At 1.0 the bare
        # VideoConditionByReferenceLatent is returned (byte-identical to before).
        cond = VideoConditionByReferenceLatent(
            latent=encoded_video,
            downscale_factor=scale,
            strength=ref_strength,
        )
        if self._ic_attention_strength < 1.0:
            cond = ConditioningItemAttentionStrengthWrapper(
                cond, attention_mask=self._ic_attention_strength
            )
        return [cond]

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
            # IC-LoRA reference-video conditioning — appended on the STAGE-1 pass
            # only. Inert (returns []) when no ic_reference is configured, so the
            # keyframe-only path above stays byte-identical.
            if self._ic_reference is not None:
                conds += self._reference_conditioning_for_stage(
                    full_height=height, num_frames=num_frames, cond_kwargs=kwargs,
                )
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
        *,
        ic_loras: list[tuple[str, float]] | None = None,
        ic_reference: tuple[str, float] | None = None,
        ic_attention_strength: float | None = None,
    ) -> None:
        # Per-job IC-LoRA resolution. ``None`` reverts to the create-time default
        # (backward compat — the Phase A harness supplies loras at create()).
        # An explicit list (incl. []) is authoritative for THIS job, so a no-LoRA
        # job after a LoRA job cleanly detaches → byte-identical output (gate G3),
        # with no leak across the resident worker's job loop.
        eff_loras = ic_loras if ic_loras is not None else self._ic_loras_default
        eff_reference = (
            ic_reference if ic_reference is not None else self._ic_reference_default
        )
        eff_attention_strength = (
            ic_attention_strength
            if ic_attention_strength is not None
            else self._ic_attention_strength_default
        )
        self._set_ic_job(eff_loras, eff_reference, eff_attention_strength)

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
        source=None,
        audio_source=None,
        ic_loras: list[tuple[str, float]] | None = None,
    ) -> dict:
        """Masked AV-latent clip chaining -> ONE continuous mp4 (Phase 3 WP4).

        Delegates to :func:`engine.pipeline.chain_pipeline.run_chain`, which
        reuses THIS pipeline's ledger/components/low-VRAM machinery. ``clips`` is
        a list of ``ChainClipSpec`` (prompt already resolved, images built).
        ``source`` (optional ``SourceSpec``) enables video-to-video continuation:
        the source tail is frozen as clip-0's head and trimmed from the output.
        ``audio_source`` (optional ``AudioSourceSpec``) enables audio-to-video:
        the uploaded audio is frozen over the whole timeline and the video is
        driven off it (mutually exclusive with ``source``). Returns metadata
        incl. segment/tile junction pixel-frame indices.

        ``ic_loras`` (style/character IC-LoRA, additive): ``(path, strength)``
        adapters applied via the forward-time weight patch across the WHOLE chain
        (every stage-1 segment + stage-2 tile). run_chain sets them explicitly
        before building the transformer (empty list clears any stale LoRA left by
        a prior single ``generate()`` on the resident pipeline).
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
            source=source,
            audio_source=audio_source,
            ic_loras=ic_loras,
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