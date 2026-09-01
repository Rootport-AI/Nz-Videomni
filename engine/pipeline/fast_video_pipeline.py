"""LTX fast video pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import TYPE_CHECKING, Final, cast

import torch

from engine.api_types import ImageConditioningInput
from engine.gguf import dequant_triton
from engine.pipeline.common import default_tiling_config, encode_video_output, video_chunks_number
from engine.pipeline.utils import AudioOrNone, TilingConfigType, device_supports_fp8
from engine.transformer.nag_service import NagParams, NagService, NagState, encode_negative
from engine.transformer.sage_attention_service import SageAttentionService, SageState
from engine.transformer.vsf_service import VsfParams, VsfService

if TYPE_CHECKING:
    from engine.gguf.ic_lora_common import IcLoraEntry


# keep_resident（ジョブ間のCPU骨格キャッシュ）で registry を差し替える
# ModelLedger のビルダー属性、**全部**。wheel の
# ``ModelLedger.build_model_builders()`` が作る7つ（checkpoint_path 由来6つ＋
# spatial_upsampler_path 由来1つ）に、この engine が自前で組み立てる
# ``text_encoder_builder``（gemma_root=None で wheel が作らないぶんを
# ``engine/gemma/gguf_quant_service.py`` の
# ``_build_shardless_text_encoder_builder`` が作る）を加えた8つ。
#
# ここに載せ忘れた属性は「そのサブモデルだけキャッシュが効かない（かつ
# ledger.registry が非Dummyなので CPU ビルドされたまま）」という**静かな部分
# 故障**になるため、``_swap_registry`` は8つの存在を assert する（欠落＝
# 属性名のタイポか wheel 側の構成変更で、どちらも黙って進めてはいけない）。
_LEDGER_BUILDER_ATTRS: Final[tuple[str, ...]] = (
    "transformer_builder",
    "vae_decoder_builder",
    "vae_encoder_builder",
    "audio_encoder_builder",
    "audio_decoder_builder",
    "vocoder_builder",
    "upsampler_builder",
    "text_encoder_builder",
)


def _set_conv3d_memory_format(module, fmt: torch.memory_format) -> int:
    """Re-lay-out every ``Conv3d`` weight inside ``module`` in place; return how
    many Conv3d modules were touched.

    Conv3d-only instead of a whole-module ``module.to(memory_format=...)``: the
    existing precedent in this repo (``chain_pipeline._chunked_upsample_cpu``,
    the ``for m in upsampler.modules()`` loop) had to be written this way because
    that module mixes in Conv2d (rank-4) weights, on which a whole-module
    ``.to(channels_last_3d)`` raises "required rank 5 tensor". The current
    VideoEncoder happens to hold none, but staying identical to that call site
    keeps this safe if the model's composition ever changes.

    ``Module.to(memory_format=)`` rewrites the storage behind each Parameter /
    buffer without replacing the Python objects, so anything holding references
    to them keeps working; it is also permitted under ``inference_mode``.
    """
    touched = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Conv3d):
            m.to(memory_format=fmt)
            touched += 1
    return touched


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
        component_video_vae_pruned_path: str = "",
        te_offload_text_encoder: bool = True,
        dit_cpu_load: bool = True,
        *,
        ic_loras: list[IcLoraEntry] | None = None,
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
            component_video_vae_pruned_path=component_video_vae_pruned_path,
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
        component_video_vae_pruned_path: str = "",
        te_offload_text_encoder: bool = True,
        dit_cpu_load: bool = True,
        *,
        ic_loras: list[IcLoraEntry] | None = None,
        ic_reference: tuple[str, float] | None = None,
        ic_attention_strength: float = 1.0,
    ) -> None:
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        # ── IC-LoRA state (all inert by default) ──────────────────────────────
        # ic_loras: (safetensors_path, strength, audio_strength) LoRAs applied to
        #   the GGUF base transformer. Phase A fused them into the full BF16
        #   state-dict at load (bf16 path); Phase B adds them at FORWARD time on
        #   the per-layer-quant path (GGUFQuantLoaderService + ggml_linear_forward).
        #   Selectable via gguf_per_layer_quant. audio_strength is None unless the
        #   caller opts in (see engine.gguf.ic_lora_common.IcLoraEntry).
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
        self._ic_loras_default: list[IcLoraEntry] = list(ic_loras or [])
        self._ic_reference_default: tuple[str, float] | None = ic_reference
        self._ic_attention_strength_default: float = float(ic_attention_strength)
        self._ic_loras: list[IcLoraEntry] = []
        self._ic_reference: tuple[str, float] | None = None
        self._ic_attention_strength: float = 1.0
        self._ic_reference_downscale_factor: int | None = None
        self._set_ic_job(
            self._ic_loras_default,
            self._ic_reference_default,
            self._ic_attention_strength_default,
        )

        # ── NAG (Normalized Attention Guidance) state ──────────────────────────
        # One NagState per pipeline instance, scoped to a single generate()/
        # generate_chain() call by _set_nag_job / the try/finally reset in both
        # entry points (see engine/transformer/nag_service.py). Installed
        # unconditionally below (after block-swap), regardless of whether any
        # job ever requests NAG — install() itself is the zero-overhead-when-off
        # gate (D3).
        self._nag = NagState()

        # ── Attention backend (sdpa / SageAttention) state ─────────────────────
        # Same lifetime rules as NagState above: one instance per pipeline,
        # scoped to a single generate()/generate_chain() call by _set_sage_job +
        # the try/finally reset in both entry points. Defaults to "sdpa", so a
        # caller that never mentions the backend gets exactly today's behaviour
        # (see engine/transformer/sage_attention_service.py).
        self._sage = SageState()

        # ── Block-swap prefetch state ──────────────────────────────────────────
        # Same lifetime rules again (set at the entry point, reset in its
        # finally), but the state itself lives on the resident BlockSwapService
        # — that is what install() reads on every rebuilt transformer. These two
        # fields are only the pipeline's copy of the request and the verdict of
        # the job that just finished (read back by the worker).
        self._block_swap_prefetch_requested = False
        self._block_swap_prefetch_used = "off"

        # ── Fused Triton GGUF dequantization state ─────────────────────────────
        # Same lifetime rules as prefetch (armed at the entry point, reset in its
        # finally), but the state itself lives in ``engine/gguf/dequant_triton``
        # as MODULE globals — the dequantization call sites are plain functions
        # deep inside the GGUF loaders with no pipeline handle to reach. There is
        # therefore nothing to mirror on the instance: the pipeline only forwards
        # arm/reset and reads the finished job's verdict back out.

        # ── keep_resident（ジョブ間のCPU骨格キャッシュ）state ──────────────────
        # 上の3つ（NAG/sage/prefetch）と決定的に違うのは**ジョブ終了時に
        # リセットしない**点。残ること自体が機能（次のジョブで再利用されるのが
        # 目的）なので、``generate()``/``generate_chain()`` の finally には
        # 対応するリセットが無い。この非対称は意図的。
        # ``_keep_resident_enabled`` は「いま ledger に非Dummy registry が
        # 刺さっているか」、``_keep_resident_registry`` は保持している
        # StateDictRegistry 実体（OFF で clear() され、再ONで同じ実体を空から
        # 使い直す＝キャッシュの作り直し）。
        self._keep_resident_enabled = False
        self._keep_resident_registry: object | None = None

        # ── PrunaVAED（枝刈り映像VAEデコーダ）state ─────────────────────────────
        # ジョブ単位のトグル。``_set_vae_mode_job`` が毎ジョブ必ず明示設定する
        # ので、prefetch/sage のような finally 側のリセットは**持たない**
        # （持たなくても前ジョブの選択が残らない。§4.3）。この文字列は直前の
        # ジョブが実際に何で復元したか（"off" / "on" / "on->off"）で、
        # ワーカーが done イベントへ載せるために読む。
        self._vae_mode_used = "off"
        # 既定ビルダーのスナップショット。``__init__`` の最後（無条件位置）で
        # 撮る。
        self._default_vae_builder: object | None = None

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
        # PrunaVAED（枝刈り版の映像VAEデコーダ、約690MB）の置き場所。空文字＝
        # 未設定で、その場合 vae_mode="prune_vaed" のジョブは既定デコーダへ降格
        # する。存在確認は**ここでは行わない**（ジョブ単位で行う。
        # ``_set_vae_mode_job`` の docstring 参照）。
        self._component_video_vae_pruned_path = component_video_vae_pruned_path
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

        # ── Re-source VIDEO VAE + AUDIO VAE/vocoder from standalone component files ──
        # Phase 1: drop the 46GB monolith for the VAE/audio builders by re-pointing
        # their model_path to small standalone files. Runs BEFORE the
        # transformer/Gemma GGUF installs. Gated on use_component_files + both
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

        # ── Install NAG (unconditional — D1) ──
        # Wraps ledger.transformer LAST, so NAG's install() runs against the
        # fully-assembled transformer (block-swap/GGUF already applied). Runs
        # every time (not gated on any job ever requesting NAG): the per-job
        # gate lives inside NagService.install() (state.requested check), which
        # is what keeps a NAG-OFF job byte-identical to before this feature
        # existed (D3).
        self._install_nag()

        # ── Install the attention-backend swap (unconditional — same rule) ─────
        # Order relative to _install_nag() does NOT matter: NAG/VSF replace
        # ``Attention.forward`` while this replaces ``Attention.attention_function``,
        # two independent attributes. (The NAG/VSF patched forwards call
        # attention_function themselves, so a NAG+sage job runs its cross-attention
        # through the sage kernel either way.) The only requirement is that both
        # wrap ledger.transformer and therefore run on every build.
        self._install_sage()

        # ── Stage 3: load-once / keep-resident weights via StateDictRegistry ──
        # 各サブモデルのCPU側 state_dict を1回だけ読み、以降のジョブでは再利用
        # する（ジョブ毎のディスク再マテリアライズの除去。前処理 66〜79秒 →
        # 9〜15秒、出力はビット一致）。
        #
        # ここは**create時の初期値**を張るだけの入口で、本番の切り替えは
        # ``generate()``/``generate_chain()`` の ``keep_resident=`` 引数
        # →``_set_keep_resident_job`` が担う。実装を1本にするため、create時も
        # ジョブ時とまったく同じ ``_swap_registry`` を通す（以前はここだけ
        # ``build_model_builders()`` を呼び直す別実装だった。あの方式は
        # install群を**先に**走らせると model_path / model_loader /
        # model_sd_ops / module_ops が作り直しで消えるので、install群より前に
        # 置くしかなかった。``_swap_registry`` は frozen dataclass の
        # ``dataclasses.replace(registry=...)`` なので他フィールドを保存でき、
        # install群の**後**に置ける＝スパイクスクリプトも本番と同じ経路を通る）。
        #
        # 非Dummy registry は ``ModelLedger._target_device()`` を CPU に倒すので、
        # サブモデルはCPUでビルド（＝キャッシュ）され、生成毎にGPUへ
        # out-of-place で移される（VRAM中立）。
        if keep_resident_weights:
            self._swap_registry(True)

        # ── PrunaVAED: 既定の映像VAEデコーダのビルダーを1本だけ控える ──────────
        # 枝刈り側はジョブ毎にここから ``dataclasses.replace`` で作る（保持
        # しない＝同期ずれの余地を作らない。§4.3）。
        #
        # **位置が load-bearing**: ``_install_component_sources()`` の直後では
        # なく、ここ（無条件位置）に置く。前者は ``use_component_files=True``
        # の構成でしか走らないので、直後に置くと単一ファイル構成のサーバーで
        # 枝刈りビルダーが作られない。上の ``_swap_registry(True)`` より**後**
        # なのも意図的で、ここで撮ったスナップショットの ``registry`` は撮った
        # 時点のもので固定される——だからこそ ``_set_vae_mode_job`` は代入の
        # たびに ``registry=ledger.registry`` を注入し直す（§4.4）。
        self._default_vae_builder = self.pipeline.model_ledger.vae_decoder_builder

        # NOTE: attention-tiling and LoRA install branches (guarded by
        # attention_tile_size > 0 / loras) were removed during the engine
        # relocation: their services (AttentionTileService / LoraService) are not
        # part of the first-party engine keep-set, the worker never enables these
        # guards (both default off), and the current T2V/GGUF path never reaches
        # them. The now-dead `attention_tile_size` and `loras` constructor
        # parameters (no caller ever passed them) have also been removed.

    def _set_ic_job(
        self,
        ic_loras: list[IcLoraEntry] | None,
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
        # The wheel's reader returns 1 BOTH for "declared as 1" (a
        # same-resolution reference adapter, e.g. Deblur) and for "key absent"
        # (a style/character LoRA), so key presence must be tested separately:
        # only LoRAs that actually declare the key may vote. A LoRA whose header
        # cannot be read does not vote (the wheel's reader degrades the same way
        # and the registry has already header-validated every config entry).
        from safetensors import safe_open
        from ltx_pipelines.ic_lora import _read_lora_reference_downscale_factor

        declared: dict[str, int] = {}
        for entry in self._ic_loras:
            path = entry[0]
            try:
                with safe_open(path, framework="pt") as fh:
                    metadata = fh.metadata() or {}
            except Exception:
                continue
            if "reference_downscale_factor" in metadata:
                declared[path] = _read_lora_reference_downscale_factor(path)
        if not declared:
            raise RuntimeError(
                "ic_reference set but none of the IC-LoRAs "
                f"{[entry[0] for entry in self._ic_loras]} declares "
                "reference_downscale_factor (metadata missing?). Refusing to "
                "run reference conditioning without a declared factor."
            )
        # One reference video is loaded at ONE resolution, so every declaring
        # LoRA must agree — including a declared 1 against a declared 2, which
        # would silently feed one of the two adapters a reference at a scale it
        # was not trained on.
        factors = sorted(set(declared.values()))
        if len(factors) > 1:
            raise RuntimeError(
                f"Conflicting reference_downscale_factor values in IC-LoRAs: {declared}. "
                "Cannot combine LoRAs that expect different reference resolutions."
            )
        factor = factors[0]
        if factor < 1:
            raise RuntimeError(
                f"IC-LoRA metadata reports reference_downscale_factor={factor} "
                f"({declared}); expected >=1."
            )
        self._ic_reference_downscale_factor = factor

    def _set_nag_job(self, nag: NagParams | VsfParams | None) -> None:
        """Set (or clear, with None) the live non-CFG negative-prompt request
        for the upcoming job — ``NagParams`` for NAG, ``VsfParams`` for VSF.

        The two methods share one NagState slot (and one stale-clear path)
        because at most one of them can be active per job; which one it is is
        carried by the params TYPE, and the only place that reads that type is
        _install_nag's service selection below plus the encode wrapper's slice
        decision. Everything else in this class is method-agnostic.

        Called from generate() directly and from run_chain() (chain_pipeline.py,
        IC-LoRA convention — generate_chain() itself does not call this; run_chain
        is the single place that sets it for the chain path). Always calling this
        — even with None — is what gives stale-clear semantics: a NAG job
        followed by a non-NAG job on the same resident pipeline does not leak the
        prior negative prompt (NagState.set_params clears any encoded contexts
        too, so a caller that forgets the matching set_contexts() call correctly
        hits install()'s "requested but not ready" RuntimeError instead of
        silently reusing a stale encoding).
        """
        self._nag.set_params(nag)

    def _set_sage_job(self, attention_backend: str) -> None:
        """Set the attention backend ("sdpa" / "sage") for the upcoming job.

        NEVER raises — deliberately, and this is load-bearing. Like
        ``_set_nag_job`` this runs OUTSIDE generate()'s try/finally (the state
        has to exist before the transformer is built), so an exception here
        would skip the matching ``self._sage.reset()`` and leak the request into
        the next job on a resident worker. An unrecognised value therefore
        degrades to "sdpa" inside ``SageState.set_backend`` instead; the
        fail-loud gate for unknown values lives at the protocol edge
        (``engine.worker._resolve_attention``), where rejecting the job is still
        possible.

        Chain asymmetry (mirror of the note in ``generate_chain``): this is
        called from ``generate()`` and from ``generate_chain()`` directly, NOT
        from ``run_chain`` — unlike ``_set_nag_job``, which run_chain owns
        because NAG's negative prompt must be encoded between setting the params
        and building the transformer. sage has nothing to encode, so it is set at
        the outermost entry point, next to its own ``finally: reset()``.
        """
        self._sage.set_backend(attention_backend)

    def attention_used(self) -> str:
        """What the last finished job's attention actually ran on: "sdpa",
        "sage", or "sage->sdpa" (sage requested, but the job degraded because a
        kernel call raised).

        Read by the worker AFTER generate()/generate_chain() returns, which is
        why it reports the snapshot taken by ``SageState.reset()`` rather than
        the live state (the pipeline's own finally has already cleared that).
        """
        return self._sage.last_attention_used

    def _set_block_swap_prefetch_job(self, enabled: bool) -> None:
        """Arm block-swap prefetching for the upcoming job.

        NEVER raises, for exactly the reason spelled out in ``_set_sage_job``:
        this runs outside generate()'s try/finally, so an exception here would
        skip the matching reset and leak the request into the next job on a
        resident worker. When block swap is not installed at all the flag is
        simply never read and the job silently runs without prefetching.
        """
        self._block_swap_prefetch_requested = bool(enabled)
        svc = getattr(self, "_block_swap_service", None)
        if svc is not None:
            svc.prefetch_requested = self._block_swap_prefetch_requested
            # Clear the previous job's verdict: a job that dies before install()
            # must not inherit "on" from the job before it.
            svc.last_prefetch_used = None

    def _reset_block_swap_prefetch_job(self) -> None:
        """End-of-job counterpart: snapshot the verdict, then clean up.

        ``last_prefetch_used is None`` means install() never ran (an early
        return or an exception upstream), which for a job that asked for
        prefetching is a degradation — hence "on->off".
        """
        svc = getattr(self, "_block_swap_service", None)
        if svc is not None:
            self._block_swap_prefetch_used = svc.last_prefetch_used or (
                "on->off" if self._block_swap_prefetch_requested else "off"
            )
            # The job owns the prefetch resources; releasing them here (rather
            # than at the next install) is what keeps the finished transformer
            # from being pinned alive between jobs.
            svc.teardown_prefetch()
            svc.prefetch_requested = False
        else:
            self._block_swap_prefetch_used = (
                "on->off" if self._block_swap_prefetch_requested else "off"
            )
        self._block_swap_prefetch_requested = False

    def block_swap_prefetch_used(self) -> str:
        """What the last finished job's block swap actually did: "off", "on", or
        "on->off" (asked for, but degraded to the synchronous path).

        Like ``attention_used`` above, this is the snapshot taken by the reset
        in the entry point's finally, not live state.
        """
        return self._block_swap_prefetch_used

    def _set_fused_dequant_job(self, enabled: bool) -> None:
        """Arm the fused Triton GGUF dequantization kernels for the upcoming job.

        NEVER raises, for exactly the reason spelled out in ``_set_sage_job``:
        this runs outside generate()'s try block, so an exception here would skip
        the matching reset and leak the request into the next job on a resident
        worker. ``set_job`` itself only assigns module globals (no import, no
        CUDA), so there is nothing here that can fail — the discipline is kept
        anyway because the call ORDER is what makes it safe.
        """
        dequant_triton.set_job(bool(enabled))

    def _reset_fused_dequant_job(self) -> None:
        """End-of-job counterpart: freeze this job's verdict and disarm.

        The verdict ("off" / "on" / "on->off") is computed inside
        ``dequant_triton.reset_job()`` from the request, the exception latch and
        the number of tensors actually dequantized on Triton — a job that asked
        for the kernels but dequantized nothing eligible is a degradation, same
        as a latch.
        """
        dequant_triton.reset_job()

    def fused_gguf_dequant_kernel_used(self) -> str:
        """What the last finished job's GGUF dequantization actually did: "off",
        "on", or "on->off" (asked for, but fell back to the eager PyTorch path).

        Like ``block_swap_prefetch_used`` above, this is the snapshot taken by
        the reset in the entry point's finally, not live state.
        """
        return dequant_triton.last_used()

    def _swap_registry(self, enabled: bool) -> None:
        """Arm (``True``) / disarm (``False``) the cross-job CPU-skeleton cache.

        Swaps ``ledger.registry`` AND every builder's own ``registry`` field in
        one step. Both halves are load-bearing:

        * ``ledger.registry`` is what ``ModelLedger._target_device()`` reads
          live — non-Dummy flips submodel builds to CPU (cacheable), Dummy puts
          them back on the GPU.
        * each ``*_builder.registry`` is what ``SingleGPUModelBuilder.load_sd``
          actually consults for the cache HIT/MISS. Builders are FROZEN
          dataclasses holding the registry BY VALUE, so re-pointing the ledger
          alone would leave every builder caching into the old registry.

        ``dataclasses.replace`` is used (not ``build_model_builders()``) because
        the builders here are no longer the wheel's originals: the install group
        above has rewritten ``model_path`` / ``model_loader`` / ``model_sd_ops``
        / ``module_ops`` on them. Rebuilding would silently throw all of that
        away (46GB monolith back in the path, GGUF loaders gone); ``replace``
        preserves every other field by construction.

        No closure captures the registry: the block-swap / NAG / sage wrappers
        and the Gemma service's wrapper all read ``ledger``/``ledger.*_builder``
        live on every build, so a swap takes effect from the very next build.

        All 8 builder attributes must EXIST (assert). ``getattr(..., None)``
        with a silent skip is deliberately NOT used: a typo'd or wheel-renamed
        attribute would then mean "that one submodel is CPU-built with no
        cache", i.e. slower AND memory-heavier with nothing in any log.
        """
        import dataclasses
        import gc
        import logging

        from ltx_core.loader.registry import DummyRegistry, StateDictRegistry

        log = logging.getLogger(__name__)
        ledger = self.pipeline.model_ledger

        missing = [a for a in _LEDGER_BUILDER_ATTRS if not hasattr(ledger, a)]
        assert not missing, (
            "keep_resident: ModelLedger is missing builder attribute(s) "
            f"{missing} — _LEDGER_BUILDER_ATTRS is out of sync with the ledger "
            "(typo, or the wheel/Gemma install changed which builders exist). "
            "Swapping only the ones that happen to exist would leave those "
            "submodels un-cached and CPU-built, which is invisible at runtime."
        )

        old = getattr(ledger, "registry", None)
        if enabled:
            if self._keep_resident_registry is None:
                self._keep_resident_registry = StateDictRegistry()
            new: object = self._keep_resident_registry
        else:
            new = DummyRegistry()

        ledger.registry = new
        for attr in _LEDGER_BUILDER_ATTRS:
            setattr(ledger, attr, dataclasses.replace(getattr(ledger, attr), registry=new))

        if not enabled and old is not None and not isinstance(old, DummyRegistry):
            # 明示的な解放：ここを通らないと約20GBのCPU骨格が居座り続ける。
            # ``clear()`` は state_dict の参照を落とすだけなので、循環参照
            # （block swap の swapped_forward クロージャ等）を確実に回収する
            # ため gc を1回回す。残留はログで追える形にする（実測では
            # prefetch ON時のDiT分が次の install() まで／``_pinned_pool`` ／
            # ``held_embed_cpu`` 約1.9GB が残りうる）。
            old.clear()
            gc.collect()
            log.info(
                "keep_resident OFF: StateDictRegistry cleared + gc.collect() done "
                "(residual CPU memory may remain until the next transformer build: "
                "block-swap CPU masters, pinned pool, held_embed_cpu ~1.9GB)"
            )
        else:
            log.info("keep_resident %s: registry swapped on ledger + %d builders",
                     "ON" if enabled else "OFF", len(_LEDGER_BUILDER_ATTRS))

        self._keep_resident_enabled = bool(enabled)

    def _set_keep_resident_job(self, enabled: bool) -> None:
        """Arm/disarm the cross-job CPU-skeleton cache for the upcoming job.

        NEVER raises, for the same reason as ``_set_block_swap_prefetch_job``:
        this runs at the outermost entry point, OUTSIDE generate()'s
        try/finally. A swap failure must not kill an otherwise runnable job —
        the feature is a preprocessing-time optimization, so "ran, just not
        cached" beats "failed". The state flag is only advanced by a swap that
        actually succeeded, so the next job retries the same transition.

        **Deliberately NOT reset at the end of a job** (the one asymmetry
        against _set_nag_job / _set_sage_job / _set_block_swap_prefetch_job):
        the cache surviving into the next job IS the feature. It is released
        only when a later job explicitly asks for ``keep_resident=False`` —
        which is exactly what an omitted request field resolves to — or when
        the worker dies (model switch: services/pipeline_manager's reload()
        kills the worker process, so a model change can never serve stale
        weights out of this cache; that is the structural guarantee behind
        §1-9's invalidation requirement).
        """
        enabled = bool(enabled)
        if enabled == self._keep_resident_enabled:
            return
        try:
            self._swap_registry(enabled)
        except Exception as exc:  # noqa: BLE001 - arming must never kill a job
            import logging
            logging.getLogger(__name__).error(
                "keep_resident switch to %s FAILED (%r) — the job continues with "
                "keep_resident=%s", enabled, exc, self._keep_resident_enabled,
            )

    def _set_vae_mode_job(self, mode: str) -> None:
        """ジョブ単位の映像VAEデコーダ選択（"default" / "prune_vaed"）。

        sage / prefetch と同じ per-job set 規律に従うが、**reset は持たない**
        ——毎ジョブ必ず明示設定するので、前ジョブの選択が残る余地が無いため
        （§8 E7）。``keep_resident`` の「リセットしない」例外とは理由が違う
        （あちらは残ること自体が機能）ので混同しないこと。

        **重みの存在確認をジョブ単位で行う理由**: パイプライン構築は load オペ
        で1回きりなので、構築時に確認しても利用者が起動後にファイルを消した／
        戻した場合に追随できない。ジョブ単位なら**ワーカーを再起動せずに**
        着脱へ追随できる（§8 E5）。

        **どちらの分岐でも ``registry=ledger.registry`` を注入し直す**のは、
        ``self._default_vae_builder`` が ``__init__`` の1回きりで撮った
        スナップショットで、その中の ``registry`` 参照が撮った時点のもので
        固定されるからである。``keep_resident`` の ON/OFF が後から切り替わると
        ``_swap_registry`` がレジストリを差し替えるので、注入し直さないと
        「キャッシュが効かない」か「解放済みのレジストリを掴む」に化ける
        （§4.4）。注入し直すぶんには呼び出し順序を気にする必要が無い。

        既定デコーダと枝刈りデコーダは**パスが違う**ため
        ``StateDictRegistry`` のキャッシュキー（``sha256(解決済みパス群 +
        sd_ops.name)``）が衝突せず、``keep_resident`` ON なら両方の state_dict
        が同じレジストリに同居する＝ジョブ間で切り替えてもディスクを読み直さ
        ない（オーナー確定事項 0-3 のハードゲート）。
        """
        import dataclasses
        import logging

        from engine.vae.pruned_video_decoder import PrunedVideoDecoderConfigurator

        log = logging.getLogger(__name__)
        ledger = self.pipeline.model_ledger
        want_pruned = mode == "prune_vaed"
        path = self._component_video_vae_pruned_path
        have_pruned = want_pruned and bool(path) and os.path.exists(path)

        if have_pruned:
            ledger.vae_decoder_builder = dataclasses.replace(
                self._default_vae_builder,
                model_path=path,
                model_class_configurator=PrunedVideoDecoderConfigurator,
                # 素通し。変換器がモジュール相対の素キーを直接出力するので
                # SDOps は要らない。**名前だけの SDOps を作ってはならない**
                # ——matcher を1つも持たない SDOps は「何もしない」ではなく
                # 「全キーを捨てる」であり（sd_ops.py:92-97、``any([])`` は
                # False）、``strict=False`` の静かな失敗に直行する（§4.1）。
                model_sd_ops=None,
                registry=ledger.registry,
            )
        else:
            ledger.vae_decoder_builder = dataclasses.replace(
                self._default_vae_builder, registry=ledger.registry
            )
            if want_pruned:
                log.warning(
                    "PrunaVAED decoder not found at %s - falling back to the "
                    "default decoder for this job", path,
                )

        self._vae_mode_used = "on" if have_pruned else ("on->off" if want_pruned else "off")

    def vae_mode_used(self) -> str:
        """What the last job's video VAE decoder actually was: "off" (the stock
        decoder), "on" (PrunaVAED), or "on->off" (PrunaVAED asked for, but the
        weight file was missing so the stock decoder ran).

        Read by the worker exactly like ``block_swap_prefetch_used`` above —
        and for the same reason: the downgrade is decided INSIDE the pipeline
        (the per-job file-existence check), so the worker cannot compute it
        from the request alone the way ``_keep_resident_used`` does.
        """
        return self._vae_mode_used

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
        ic_loras: list[IcLoraEntry] | None = None,
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
            # Always False at create time; the per-job value is written by
            # _set_block_swap_prefetch_job before the transformer is built.
            service.prefetch_requested = self._block_swap_prefetch_requested
            # (No equivalent line is needed for the fused GGUF dequantization
            # kernels: unlike sage/prefetch — whose per-job request has to be
            # re-applied to each freshly built service instance — that state
            # lives in dequant_triton's MODULE globals, which the transformer
            # rebuild does not touch. The GGUF weights are dequantized DURING
            # this build, i.e. after _set_fused_dequant_job armed the flag at
            # the entry point, so the build itself is already covered.)
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

    def _install_nag(self) -> None:
        """Wrap ``ledger.transformer`` so every freshly-built transformer gets
        its cross-attention modules NAG-patched (a no-op patch when the current
        job didn't request NAG — see NagService.install()).

        Deliberately NO try/except, unlike _install_block_swap and _install_gguf
        above: those swallow install failures and fall back to an unpatched/
        unaccelerated path because their features are pure speed/VRAM
        optimizations where "worked, but slower" beats "job failed". NAG affects
        the actual generated output (a negative prompt the user explicitly
        turned on), so a silent fallback here would produce a plausible-looking
        video that quietly ignored the negative prompt — worse than an error.
        Any failure inside NagService.install() (including its own fail-loud
        RuntimeErrors) must abort the job.

        Method selection (NAG vs VSF) is the single isinstance below. Both
        services no-op on a job that requested neither (their install() reads
        the same NagState and returns 0 when nothing was requested), so the
        branch only decides WHICH patch an actively-requesting job gets — it
        never becomes a third code path for an OFF job.
        """
        nag_service = NagService(lambda: self._nag)
        vsf_service = VsfService(lambda: self._nag)
        original_transformer = self.pipeline.model_ledger.transformer

        def patched_transformer() -> torch.nn.Module:
            t = original_transformer()
            if isinstance(self._nag.params, VsfParams):
                vsf_service.install(t)
            else:
                nag_service.install(t)
            return t

        self.pipeline.model_ledger.transformer = patched_transformer

    def _install_sage(self) -> None:
        """Wrap ``ledger.transformer`` so every freshly-built transformer gets
        its ``attention_function``s swapped for SageAttention — a no-op patch
        when the current job didn't request it (SageAttentionService.install()
        returns 0 immediately, having touched nothing).

        No try/except here, unlike _install_block_swap/_install_gguf: this method
        only stores a closure, it cannot fail. The failures that DO matter
        (missing wheel, unhappy kernel) are handled inside the service, which
        degrades to sdpa and records that as ``attention_used="sage->sdpa"`` —
        the discipline is deliberately different from NAG's fail-loud because
        sage changes speed, not output (see sage_attention_service's docstring).
        """
        sage_service = SageAttentionService(lambda: self._sage)
        original_transformer = self.pipeline.model_ledger.transformer

        def patched_transformer() -> torch.nn.Module:
            t = original_transformer()
            sage_service.install(t)
            return t

        self.pipeline.model_ledger.transformer = patched_transformer

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

        from engine.pipeline.common import load_video_conditioning_cpu

        ref_path, ref_strength = self._ic_reference
        scale, ref_height, ref_width = self._reference_pixel_dims(
            cond_height, int(cond_kwargs["width"])
        )

        # CPU-assembled reference pixels: the wheel's load_video_conditioning
        # cats the growing tensor ON THE GPU, so its allocation volume grows
        # with the SQUARE of the frame count (measured 46.8GB reserved at
        # 640x384x257). Numerics are bit-identical — see the docstring.
        #
        # The decode now sits just OUTSIDE the encode's measured/logged interval
        # (§1-15 B5 moved it out of _reference_conditioning_from_pixels, which the
        # chain feeds already-decoded windows). Log-line only: the decode holds one
        # frame at a time on the GPU and produces the same CPU tensor as before, so
        # neither the encode's inputs nor its outputs move a bit.
        video = load_video_conditioning_cpu(
            video_path=ref_path,
            height=ref_height,
            width=ref_width,
            frame_cap=num_frames,
            dtype=cond_kwargs["dtype"],
            device=cond_kwargs["device"],
        )
        conds = self._reference_conditioning_from_pixels(
            video, cond_kwargs=cond_kwargs, scale=scale, strength=ref_strength
        )
        del video  # the callee dropped its own reference; drop this frame's too
        return conds

    def _reference_pixel_dims(self, cond_height: int, cond_width: int) -> tuple[int, int, int]:
        """``(scale, ref_height, ref_width)``: the size the reference video is
        decoded at for a conditioning built at ``cond_height x cond_width``.

        Split out of ``_reference_conditioning_for_stage`` (§1-15 B5) so the chain
        path — which decodes its OWN per-clip pixel windows out of one long
        reference and then calls ``_reference_conditioning_from_pixels`` directly —
        resolves the decode size through the same guards and the same arithmetic.
        Pure; the two guards below are moved verbatim, not re-derived.
        """
        scale = self._ic_reference_downscale_factor
        if scale is None or scale < 1:
            raise RuntimeError(
                f"reference downscale factor not initialised (got {scale}); "
                "_set_ic_job must resolve it before the reference conditioning "
                "is built."
            )
        if cond_height % scale != 0 or cond_width % scale != 0:
            raise ValueError(
                f"Stage-1 dims ({cond_height}x{cond_width}) must be divisible by "
                f"reference_downscale_factor ({scale})"
            )
        return scale, cond_height // scale, cond_width // scale

    def _reference_conditioning_from_pixels(
        self,
        video: torch.Tensor,
        *,
        cond_kwargs: dict,
        scale: int,
        strength: float,
    ) -> list:
        """VAE-encode already-decoded reference PIXELS into ``[conditioning]``.

        The back half of ``_reference_conditioning_for_stage``, split out verbatim
        (§1-15 B5): the channels_last_3d switch for factor-1 adapters, the
        tiled/untiled encode branch, the restore in ``finally`` and the
        attention-strength wrapper. ``video`` is a CPU (1,C,F,H,W) tensor at
        ``_reference_pixel_dims`` resolution.

        NO stage discriminator here on purpose — that check MUST stay in the
        ``_reference_conditioning_for_stage`` wrapper. Moving it inside would make
        the chain path (which builds its own half-res windows and calls this
        directly) either always empty or, worse, attach a reference to stage 2.

        The chain calls this once per stage-1 segment with that segment's window
        of the long reference; the single-generate path calls it once with the
        whole video. The returned latents stay on ``device`` — the conditioning's
        ``apply_to`` indexes against the latent state's device.
        """
        import logging

        from ltx_core.conditioning import (
            ConditioningItemAttentionStrengthWrapper,
            VideoConditionByReferenceLatent,
        )
        from ltx_pipelines.utils.helpers import cleanup_memory

        video_encoder = cond_kwargs["video_encoder"]
        device = cond_kwargs["device"]

        # ONE predicate for both the measurement and the layout switch below:
        # ``device`` may arrive as a plain string (so compare
        # ``torch.device(device).type``, not the object), and the unit tests'
        # fake encoders have no ``.modules()`` to walk.
        _accel = torch.device(device).type == "cuda" and callable(
            getattr(video_encoder, "modules", None)
        )
        _relayout = _accel and scale == 1
        _converted = 0
        if _relayout:
            # torch 2.9's bf16 Conv3d takes an im2col fallback for contiguous
            # weights and materialises an "in_ch*27 x output volume" bf16 matrix
            # per convolution; THAT intermediate — not the activations — is what
            # OOMs the factor-1 (deblur) reference encode, tiled or not.
            # channels_last_3d weights route to the cuDNN direct kernel and the
            # intermediate disappears (measured on the OOM case: peak 6265 ->
            # 1198 MiB, 6.8 -> 5.0 s).
            #
            # This is NOT bit-identical, and this comment is the full extent of
            # the change: only the factor-1 REFERENCE latent moves (rel_rms
            # 1.2e-2 / cos 0.99993). Keyframes, factor-2 references, chain source
            # heads and stage 2 all stay bit-identical (measured), and deblur has
            # no shipped baseline output, so nothing published shifts.
            #
            # The input video is deliberately NOT converted: CausalConv3d's
            # repeat+cat (convolution.py:304-313) re-normalises it to contiguous
            # before the first convolution, so an input-side channels_last is
            # provably inert (measured rel_rms 0.0).
            #
            # Nothing leaks downstream either: tiled_encode returns a contiguous
            # accumulator (video_vae.py:371-404 — ``torch.zeros`` + ``+=``).
            _converted = _set_conv3d_memory_format(video_encoder, torch.channels_last_3d)
        try:
            # cleanup AFTER the switch, so the contiguous weight storages it just
            # dropped (~608MiB) are reclaimed by this same pass. This encode runs
            # OUTSIDE the pre-denoise release window (worker.py's
            # _denoise_with_cache_release), so the reserved pool has to be freed
            # here explicitly — same gc+empty_cache+synchronize as
            # chain_pipeline._encode_source_heads.
            cleanup_memory()
            if _accel:
                _mb = 1024 * 1024
                # The reset below clears BOTH per-job counters worker.py reports
                # (peak_vram_mb from max_memory_allocated, peak_vram_reserved_mb
                # from max_memory_reserved), so carry the job-so-far peaks into
                # the log line rather than losing them. The VRAM gates read the
                # reference-encode numbers from THIS log line, not from the job
                # metadata.
                _prior_peak = torch.cuda.max_memory_allocated() // _mb
                _prior_peak_reserved = torch.cuda.max_memory_reserved() // _mb
                torch.cuda.reset_peak_memory_stats()
                _alloc_before = torch.cuda.memory_allocated() // _mb
                _reserved_before = torch.cuda.memory_reserved() // _mb

            if scale == 1:
                # Factor-1 adapters (deblur) feed the reference at 4x the pixel
                # count of the factor-2 ones, and the untiled encode's
                # intermediates OOM a 16GB card — the same reason
                # chain_pipeline._encode_source_heads is tiled. The tiling config
                # is the DECODE-side default reused here; there is no separate
                # encode config. Factor >= 2 stays untiled (tiling would split it
                # temporally and break byte-identity).
                encoded_video = video_encoder.tiled_encode(
                    video, cond_kwargs.get("tiling_config")
                )
            else:
                # VideoEncoder.forward expects its input already on the compute
                # device (tiled_encode above moves tiles itself; the plain call
                # does not). dtype is already final (normalize_latent).
                video = video.to(device)
                encoded_video = video_encoder(video)
            del video
            if _accel:
                # Logged here — after the encode, BEFORE the restore — so the
                # interval peak is the encode's own and not polluted by the
                # restore's transient (+54MiB, one weight's worth). ``convs`` is
                # the converted Conv3d count (expect 42; a 0 means the layout
                # switch did not run).
                logging.getLogger(__name__).info(
                    "IC-LoRA reference encode (scale=%d tiled=%s channels_last_3d convs=%d): "
                    "allocated %d -> %d MB, reserved %d -> %d MB, "
                    "interval peak allocated %d MB / reserved %d MB "
                    "(job peak before this interval: allocated %d MB, reserved %d MB)",
                    scale, scale == 1, _converted,
                    _alloc_before, torch.cuda.memory_allocated() // _mb,
                    _reserved_before, torch.cuda.memory_reserved() // _mb,
                    torch.cuda.max_memory_allocated() // _mb,
                    torch.cuda.max_memory_reserved() // _mb,
                    _prior_peak, _prior_peak_reserved,
                )
        finally:
            if _relayout:
                try:
                    # Restoring is a CORRECTNESS requirement, not hygiene: stage 2
                    # of the SAME job re-encodes the keyframes through this very
                    # encoder (distilled.py:164-171 / chain_pipeline.py:877) and
                    # those latents do change under channels_last (measured).
                    # Later jobs rebuild the encoder from scratch, so they are
                    # structurally safe regardless — that is the second net, not
                    # the first. The encoder always enters here contiguous, so
                    # "restore" is an unconditional force back to contiguous.
                    _set_conv3d_memory_format(video_encoder, torch.contiguous_format)
                    # ...and release the channels_last storages just discarded:
                    # the chain route has no pre-denoise release wrapper of its
                    # own (chain_pipeline.py:659 — its private denoise never goes
                    # through worker.py's monkeypatch). Costs a few ms.
                    cleanup_memory()
                except Exception:  # must never mask an in-flight encode failure
                    logging.getLogger(__name__).exception(
                        "IC-LoRA reference encode: failed to restore the video "
                        "encoder's contiguous layout"
                    )
        # Control-adherence knob (upstream iclora_utils parity): only when the
        # attention strength is < 1.0 do we wrap the reference conditioning so a
        # scalar additive self-attention mask reaches SDPA. At 1.0 the bare
        # VideoConditionByReferenceLatent is returned (byte-identical to before).
        cond = VideoConditionByReferenceLatent(
            latent=encoded_video,
            downscale_factor=scale,
            strength=strength,
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
        # NAG: encode_text is patched (only when this job requested NAG) so the
        # negative prompt gets encoded into NagState using the same live
        # text_encoder, exactly once, before ledger.transformer() is built
        # (distilled.py:99 -> :108; see _make_nag_encode_text below).
        _orig_encode = _distilled_mod.encode_text

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

        # try/finally window widened (was: only around self.pipeline(...) below)
        # to also cover the patch-application section right below this comment.
        # Latent defect this closes: if any of the patch-building calls between
        # here and the old try (e.g. _make_stg_denoising_func, the sigma-schedule
        # builders) had raised, the _orig_* saves above would never be restored
        # — the NEXT job on this resident pipeline would silently inherit a
        # half-applied previous job's globals. Widening the window one level up
        # is required before adding the encode_text patch below (a 5th global to
        # get this same guarantee).
        try:
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
                    # tiling_config rides along in a COPY of the wheel's kwargs
                    # (the originals above must stay exactly as the wheel passed
                    # them); the factor-1 reference encode needs it.
                    conds += self._reference_conditioning_for_stage(
                        full_height=height, num_frames=num_frames,
                        cond_kwargs={**kwargs, "tiling_config": tiling_config},
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

            # ── NAG negative-prompt encode patch ──────────────────────────────────
            # Only when THIS job requested NAG (NagState.requested) — otherwise
            # encode_text is left untouched, so a NAG-OFF job never even sees this
            # branch (D3). See _make_nag_encode_text: the wrapper encodes the
            # negative prompt into NagState using the SAME live text_encoder,
            # exactly once, before returning the positive result unchanged.
            if self._nag.requested:
                _distilled_mod.encode_text = self._make_nag_encode_text(_orig_encode)  # type: ignore[attr-defined]

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
            _distilled_mod.encode_text = _orig_encode  # type: ignore[attr-defined]

    def _make_nag_encode_text(self, orig):
        """Build the NAG-aware replacement for ``encode_text`` installed (only
        when ``self._nag.requested``) over ``ltx_pipelines.distilled``'s module-
        global during ``_run_inference``.

        Called exactly once per job at distilled.py:99
        (``encode_text(text_encoder, prompts=[prompt])``), before
        ``ledger.transformer()`` is built at distilled.py:108 — this ordering
        (encode negative -> build transformer) is D2's correctness guarantee:
        NagService.install() raises if the transformer is built before the
        negative context is in NagState.

        The positive result is returned to the caller completely unchanged
        (DistilledPipeline never learns NAG exists); the negative prompt is
        encoded as an extra call against the SAME live ``text_encoder``
        instance (avoiding a second Gemma load) and stashed in NagState via
        ``encode_negative``/``set_contexts``.
        """
        def wrapped(text_encoder: object, prompts: list[str]):
            result = orig(text_encoder, prompts=prompts)
            params = self._nag.params
            assert params is not None  # implied by self._nag.requested at the call site
            # VSF concatenates the negative context into ONE shared softmax
            # and negates its values, so it must never see the connector's
            # learned register embeddings — hence the encode-time slice, taken
            # here where the method is known (see encode_negative's docstring).
            # NAG passes False and stays byte-identical.
            video_ctx, audio_ctx = encode_negative(
                text_encoder,
                params.negative_prompt,
                slice_to_real_tokens=isinstance(params, VsfParams),
            )
            self._nag.set_contexts(video_ctx, audio_ctx)
            return result

        return wrapped

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
        ic_loras: list[IcLoraEntry] | None = None,
        ic_reference: tuple[str, float] | None = None,
        ic_attention_strength: float | None = None,
        nag: NagParams | VsfParams | None = None,
        attention_backend: str = "sdpa",
        block_swap_prefetch: bool = False,
        keep_resident: bool | None = None,
        fused_gguf_dequant_kernel: bool = False,
        vae_mode: str = "default",
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
        # NAG has no create-time default (unlike IC-LoRA above) — every job sets
        # it explicitly, ``None`` included, so a NAG job followed by a plain job
        # cleanly detaches instead of leaking the prior negative prompt.
        self._set_nag_job(nag)
        # Attention backend, same "every job sets it explicitly" rule as NAG.
        # Kept LAST of the three _set_*_job calls on purpose: _set_ic_job can
        # raise, and it must not do so with the sage request already armed but
        # the try/finally not yet entered.
        self._set_sage_job(attention_backend)
        # Block-swap prefetch, armed alongside sage for the same reason (no
        # ordering constraint, and its reset lives in the finally below).
        self._set_block_swap_prefetch_job(block_swap_prefetch)
        # Fused Triton GGUF dequantization, armed alongside prefetch for the same
        # reason (no ordering constraint, and its reset lives in the finally
        # below). Must be armed BEFORE the try, because the transformer build
        # that dequantizes the GGUF weights happens inside it.
        self._set_fused_dequant_job(fused_gguf_dequant_kernel)
        # keep_resident: ``None`` = 触らない（現在の状態を維持）。ワーカーは
        # 常に明示的な bool を渡すが、outputs/ 配下のスパイクスクリプトは
        # create時に keep_resident_weights=True を張って直接 generate() を
        # 呼ぶので、既定 False にすると1本目でキャッシュを剥がしてしまう。
        # 上の3つと違い、finally に対応するリセットは**無い**（残ることが機能。
        # ``_set_keep_resident_job`` のdocstring参照）。
        if keep_resident is not None:
            self._set_keep_resident_job(keep_resident)
        # 映像VAEデコーダの選択。keep_resident の**後**に置くのは読みやすさの
        # ためだけで、順序の制約は無い（`_set_vae_mode_job` は代入のたびに
        # registry を注入し直すので、_swap_registry との前後を問わない。§4.4）。
        self._set_vae_mode_job(vae_mode)

        try:
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
        finally:
            # Resident-worker leak guard: NAG state must never survive past the
            # job that requested it (same reasoning as the IC-LoRA per-job reset
            # above, but NAG has no "revert to create-time default" concept, so
            # the only correct end state is fully cleared).
            self._nag.reset()
            # Same guard for the attention backend; reset() also snapshots what
            # this job actually ran on for attention_used() below.
            self._sage.reset()
            # Same again for prefetching, and this one additionally tears the
            # transfer state down (drains the stream, frees the arenas and the
            # CPU masters) so nothing survives into the next job.
            self._reset_block_swap_prefetch_job()
            # Same again for the fused GGUF dequantization kernels; reset also
            # snapshots this job's verdict for
            # fused_gguf_dequant_kernel_used() below.
            self._reset_fused_dequant_job()

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
        retake=None,
        end_source=None,
        ic_loras: list[IcLoraEntry] | None = None,
        ic_reference: tuple[str, float] | None = None,
        ic_attention_strength: float | None = None,
        chunked_upsample: bool = False,
        stage2_v_tile: int | None = None,
        stage2_v_adv: int | None = None,
        nag: NagParams | VsfParams | None = None,
        attention_backend: str = "sdpa",
        block_swap_prefetch: bool = False,
        keep_resident: bool | None = None,
        fused_gguf_dequant_kernel: bool = False,
        vae_mode: str = "default",
    ) -> dict:
        """Masked AV-latent clip chaining -> ONE continuous mp4 (Phase 3 WP4).

        Delegates to :func:`engine.pipeline.chain_pipeline.run_chain`, which
        reuses THIS pipeline's ledger/components/low-VRAM machinery. ``clips`` is
        a list of ``ChainClipSpec`` (prompt already resolved, images built).
        ``source`` (optional ``SourceSpec``) enables video-to-video continuation:
        the source tail is frozen as clip-0's head and trimmed from the output.
        ``audio_source`` (optional ``AudioSourceSpec``) enables audio-to-video:
        the uploaded audio is frozen over the whole timeline and the video is
        driven off it (mutually exclusive with ``source``).
        ``retake`` (optional ``RetakeSpec``) enables temporal inpainting: an
        app-cut window's two ends stay frozen while its middle is regenerated,
        and the WHOLE window is delivered (mutually exclusive with both of the
        above). Like ``source``/``audio_source`` it is a pure pass-through —
        nothing is armed on this pipeline for it. Returns metadata incl.
        segment/tile junction pixel-frame indices.

        ``end_source`` (optional ``EndSourceSpec``, additive) is the mirror of
        ``source`` at the far end: the app-cut material is VAE-encoded and frozen
        as the TAIL of the last stage-1 segment and the last stage-2 tile, so the
        chain ENDS on it. UNLIKE ``source`` nothing is trimmed — the delivered
        length is exactly what it would be without one. Combinable with
        ``source`` (start + end = interpolation), mutually exclusive with
        ``retake`` and ``audio_source``. Another pure pass-through; ``None``
        keeps every other path byte-identical.

        ``ic_loras`` (style/character IC-LoRA, additive): ``(path, strength,
        audio_strength)`` adapters applied via the forward-time weight patch
        across the WHOLE chain
        (every stage-1 segment + stage-2 tile). run_chain sets them explicitly
        before building the transformer (empty list clears any stale LoRA left by
        a prior single ``generate()`` on the resident pipeline).

        ``ic_reference`` / ``ic_attention_strength`` (α, additive): a control-adapter
        reference video ``(path, strength)`` wired to clip-0's STAGE-1 conditioning
        (accepted only for clips=1 chains; the API layer enforces that). ``None``
        reference -> the chain is byte-identical to before; ``ic_attention_strength``
        is normalised to 1.0 when unset so run_chain's ``float`` contract holds.

        ``nag`` (NAG negative-prompt guidance, additive): passed straight through
        to ``run_chain``, which is the single place that calls
        ``pipe._set_nag_job`` for the chain path (IC-LoRA convention — NOT called
        here, mirroring how ``ic_loras`` above is set inside run_chain too). The
        ``finally: self._nag.reset()`` below still runs regardless of which layer
        set it, so NAG state never survives past this call on the resident
        pipeline.

        ``attention_backend`` ("sdpa" / "sage", additive): set HERE rather than
        inside run_chain — the deliberate opposite of ``nag`` above. NAG's
        params have to be set by whoever also encodes the negative prompt,
        because install() rejects "requested but not encoded"; the attention
        backend has no such ordering constraint, so it is armed at the outermost
        entry point where its ``finally: reset()`` also lives. See
        ``_set_sage_job``'s docstring for the mirror of this note.

        ``block_swap_prefetch`` (additive): set HERE too, for the same reason as
        ``attention_backend`` — it has nothing to encode, so it belongs at the
        outermost entry point next to its own ``finally``. The chain builds the
        transformer exactly once (chain_pipeline.py), so one prefetch engine
        serves every segment and every stage-2 tile of the whole chain.

        ``keep_resident`` (additive, ``None`` = leave the current state alone):
        armed here for the same "outermost entry point" reason, but with NO
        counterpart in the finally — the whole point of the CPU-skeleton cache
        is that it survives the job. See ``_set_keep_resident_job``.

        ``fused_gguf_dequant_kernel`` (additive): armed here and reset in the
        finally, same discipline as ``block_swap_prefetch``. One arm covers the
        whole chain because the chain builds (and therefore dequantizes) the
        transformer exactly once.

        ``stage2_v_tile`` / ``stage2_v_adv`` (additive, ``None`` = the frozen
        (22, 18) default): the stage-2 tile geometry, passed straight through to
        ``run_chain`` — nothing is armed on the pipeline for it, unlike the
        acceleration knobs below, because it is pure layout arithmetic that
        ``chain_math.compute_chain_layout`` resolves inside ``run_chain``.

        ``vae_mode`` ("default" / "prune_vaed", additive): armed here too, and
        it is load-bearing that this call is NOT forgotten — the chain's decoder
        is created at ``chain_pipeline.py``'s ``ledger.video_decoder()``, a
        different call site from the single-generate one, and both are reached
        only through the ledger builder this sets (§8 E2). One setting covers
        every clip and every stage of the chain.
        """
        from engine.pipeline.chain_pipeline import run_chain

        self._set_sage_job(attention_backend)
        self._set_block_swap_prefetch_job(block_swap_prefetch)
        self._set_fused_dequant_job(fused_gguf_dequant_kernel)
        if keep_resident is not None:
            self._set_keep_resident_job(keep_resident)
        self._set_vae_mode_job(vae_mode)

        try:
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
                retake=retake,
                end_source=end_source,
                ic_loras=ic_loras,
                ic_reference=ic_reference,
                ic_attention_strength=(
                    1.0 if ic_attention_strength is None else ic_attention_strength
                ),
                chunked_upsample=chunked_upsample,
                stage2_v_tile=stage2_v_tile,
                stage2_v_adv=stage2_v_adv,
                nag=nag,
            )
        finally:
            self._nag.reset()
            self._sage.reset()
            self._reset_block_swap_prefetch_job()
            self._reset_fused_dequant_job()

    @torch.inference_mode()
    def generate_outpaint(
        self,
        *,
        prompt: str,
        canvas_path: str,
        source_path: str | None,
        geometry,
        num_frames: int,
        frame_rate: float,
        num_steps: int,
        seed: int,
        output_path: str,
        ic_loras: list[IcLoraEntry] | None = None,
        ic_reference: tuple[str, float] | None = None,
        ic_attention_strength: float | None = None,
        blend_dilation_stage1: int = 5,
        blend_dilation_stage2: int = 2,
        freeze_source_audio: bool = True,
        progress=None,
        nag: NagParams | VsfParams | None = None,
        attention_backend: str = "sdpa",
        block_swap_prefetch: bool = False,
        keep_resident: bool | None = None,
        fused_gguf_dequant_kernel: bool = False,
        vae_mode: str = "default",
    ) -> dict:
        """Canvas extension (outpainting, Docs/PENDING_TASKS_CLOSED.md §3-70,
        filed as §1-13 at the time) -> ONE mp4.

        Delegates to :func:`engine.pipeline.outpaint_pipeline.run_outpaint`,
        which reuses THIS pipeline's ledger/components/low-VRAM machinery the
        same way ``run_chain`` does. ``canvas_path`` is the green-padded canvas
        the app built; ``ic_reference`` already points at it.

        The acceleration knobs follow ``generate_chain``'s split exactly: ``nag``
        is armed INSIDE run_outpaint (whoever encodes the negative prompt must
        also set it, or NagService.install() rejects the job), everything else is
        armed here at the outermost entry point where its ``finally`` reset also
        lives. ``keep_resident`` deliberately has no reset — surviving the job is
        what the CPU-skeleton cache is for.
        """
        from engine.pipeline.outpaint_pipeline import run_outpaint

        self._set_sage_job(attention_backend)
        self._set_block_swap_prefetch_job(block_swap_prefetch)
        self._set_fused_dequant_job(fused_gguf_dequant_kernel)
        if keep_resident is not None:
            self._set_keep_resident_job(keep_resident)
        self._set_vae_mode_job(vae_mode)

        try:
            return run_outpaint(
                self,
                prompt=prompt,
                canvas_path=canvas_path,
                source_path=source_path,
                geometry=geometry,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_steps=num_steps,
                seed=seed,
                output_path=output_path,
                ic_loras=ic_loras,
                ic_reference=ic_reference,
                ic_attention_strength=(
                    1.0 if ic_attention_strength is None else ic_attention_strength
                ),
                blend_dilation_stage1=blend_dilation_stage1,
                blend_dilation_stage2=blend_dilation_stage2,
                freeze_source_audio=freeze_source_audio,
                nag=nag,
                progress=progress,
            )
        finally:
            self._nag.reset()
            self._sage.reset()
            self._reset_block_swap_prefetch_job()
            self._reset_fused_dequant_job()

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
        # NOT compatible with NAG/VSF: this caches ONE compiled transformer
        # instance and replaces ledger.transformer with a lambda that returns it
        # forever, which defeats D1 (NAG's/VSF's install() must re-run against a
        # FRESH transformer every job, since attn2/audio_attn2 are patched
        # per-job based on that job's NagState). No caller currently uses this
        # method in the production path (see D1's dead-code note), so it is left
        # as-is rather than reworked to cooperate with per-job NAG/VSF install.
        transformer = self.pipeline.model_ledger.transformer()

        compiled = cast(
            torch.nn.Module,
            torch.compile(transformer, mode="reduce-overhead", fullgraph=False),
        )
        setattr(self.pipeline.model_ledger, "transformer", lambda: compiled)