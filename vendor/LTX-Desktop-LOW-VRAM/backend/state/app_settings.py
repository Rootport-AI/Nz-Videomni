"""Canonical app settings schema and patch models."""

from __future__ import annotations

from typing import Any, TypeGuard, TypeVar, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator


def _to_camel_case(field_name: str) -> str:
    special_aliases = {
        "prompt_enhancer_enabled_t2v": "promptEnhancerEnabledT2V",
        "prompt_enhancer_enabled_i2v": "promptEnhancerEnabledI2V",
    }
    if field_name in special_aliases:
        return special_aliases[field_name]

    head, *tail = field_name.split("_")
    return head + "".join(part.title() for part in tail)


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    if value is None:
        return default

    parsed = int(value)
    return max(minimum, min(maximum, parsed))


class SettingsBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel_case,
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore",
    )


class SettingsPatchModel(SettingsBaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel_case,
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )


class FastModelSettings(SettingsBaseModel):
    use_upscaler: bool = True


class ProModelSettings(SettingsBaseModel):
    steps: int = 20
    use_upscaler: bool = True

    @field_validator("steps", mode="before")
    @classmethod
    def _clamp_steps(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=100, default=20)


class AppSettings(SettingsBaseModel):
    use_torch_compile: bool = False
    load_on_startup: bool = False
    ltx_api_key: str = ""
    user_prefers_ltx_api_video_generations: bool = False
    fal_api_key: str = ""
    use_local_text_encoder: bool = False
    fast_model: FastModelSettings = Field(default_factory=FastModelSettings)
    pro_model: ProModelSettings = Field(default_factory=ProModelSettings)
    prompt_cache_size: int = 100
    prompt_enhancer_enabled_t2v: bool = True
    prompt_enhancer_enabled_i2v: bool = False
    gemini_api_key: str = ""
    seed_locked: bool = False
    locked_seed: int = 42
    models_dir: str = ""

    # ── VRAM optimisation settings ──────────────────────────────────────
    block_swap_blocks_on_gpu: int = 0
    use_fp8_transformer: bool = False
    attention_tile_size: int = 0
    text_encoder_device: str = ""
    transformer_device: str = ""
    civitai_loras: str = "[]"
    gguf_transformer_path: str = ""
    gguf_per_layer_quant: bool = True  # True = weights compressed in VRAM (lower VRAM); False = dequant at load (higher VRAM, faster first-load)
    use_abliterated_encoder: bool = False
    use_multi_gpu: bool = False
    # VAE tiling (0 = use library defaults: 512px spatial, 64 frames temporal)
    vae_spatial_tile_size: int = 0
    vae_temporal_tile_size: int = 0
    # Free text encoder from CPU RAM after encoding (single-GPU only).
    # Saves ~9GB system RAM at the cost of ~30s reload on next generation.
    unload_text_encoder_after_encode: bool = False
    # Run Gemma's built-in enhance_t2v() locally before encoding.
    # Uses the already-loaded model — no extra VRAM cost.
    enhance_prompt_locally: bool = False
    # Number of denoising steps for the distilled (fast) pipeline.
    # 8 = full quality, fewer steps = faster but lower quality.
    distilled_num_steps: int = 8
    # Force a full pipeline reload every N generations to defragment VRAM.
    # 0 = disabled. Recommended: 4-8 if you hit OOM after multiple generations.
    reload_pipeline_every_n_gens: int = 0
    # Spatio-Temporal Guidance scale for the distilled pipeline.
    # 0.0 = disabled. Typical range 0.5–2.0. Improves prompt adherence without
    # a negative prompt — runs a second perturbed forward pass per step.
    stg_scale: float = 0.0
    # Transformer block index to perturb for STG (0-47 for LTX-Video 2.3 22B 48-block model).
    # Block 28 is the community-recommended default for 22B. Does nothing when stg_scale=0.
    stg_block_index: int = 28
    # Sigma schedule for the distilled pipeline denoising loop.
    # "distilled"        = LTX-native compressed schedule (default).
    # "linear"           = evenly-spaced sigmas 1.0→0.0; reduces metallic audio artifacts.
    # "linear_quadratic" = linear early steps + quadratic late steps; finer low-noise
    #                      refinement budget (threshold_noise=0.025, linear half).
    # "beta"             = beta-distribution sampling (arXiv 2407.12173); bell-curve
    #                      step density weighted toward midrange noise. alpha=beta=0.6.
    distilled_sigma_schedule: str = "distilled"
    # Denoising loop algorithm for the distilled pipeline.
    # "euler"               = standard first-order Euler sampler (default).
    # "gradient_estimating" = velocity correction across consecutive steps; can improve
    #                         temporal consistency (paper: openreview.net/pdf?id=o2ND9v0CeK).
    # Denoising loop algorithm for the distilled pipeline.
    # "euler"               = standard first-order Euler sampler (default).
    # "gradient_estimating" = velocity correction across consecutive steps; can improve
    #                         temporal consistency (paper: openreview.net/pdf?id=o2ND9v0CeK).
    # "res2s"               = second-order Runge-Kutta + SDE noise injection.  Two model
    #                         evaluations per step (2× slower per step, but may match Euler
    #                         quality at half the step count).  Uses Res2sDiffusionStep;
    #                         stochastic — results vary even at fixed seed.
    denoising_loop: str = "euler"
    # Gradient correction strength. Only used when denoising_loop="gradient_estimating".
    # Default 2.0 (paper default). Higher = stronger correction; may over-smooth motion.
    ge_gamma: float = 2.0
    # res2s anchor-point refinement ("bong iteration").  When enabled, the midpoint
    # estimate is iteratively tightened up to res2s_bongmath_max_iter times for steps
    # where the step size h < 0.5.  With the distilled sigma schedule virtually every
    # step has h < 0.5, so this multiplies compute by res2s_bongmath_max_iter.
    # Disable for a fast/cheap res2s run; enable for higher-quality RK midpoint.
    res2s_bongmath: bool = False
    # Maximum anchor-refinement iterations when res2s_bongmath=True.
    # The distilled schedule has very small step sizes so even 5–10 iter is slow.
    res2s_bongmath_max_iter: int = 5

    @field_validator("block_swap_blocks_on_gpu", mode="before")
    @classmethod
    def _clamp_block_swap(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=48, default=0)

    @field_validator("attention_tile_size", mode="before")
    @classmethod
    def _clamp_attention_tile(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=16384, default=0)

    @field_validator("prompt_cache_size", mode="before")
    @classmethod
    def _clamp_prompt_cache_size(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=1000, default=100)

    @field_validator("locked_seed", mode="before")
    @classmethod
    def _clamp_locked_seed(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=2_147_483_647, default=42)

    @field_validator("distilled_num_steps", mode="before")
    @classmethod
    def _clamp_distilled_num_steps(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=8, default=8)

    @field_validator("reload_pipeline_every_n_gens", mode="before")
    @classmethod
    def _clamp_reload_every_n_gens(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=100, default=0)

    @field_validator("stg_scale", mode="before")
    @classmethod
    def _clamp_stg_scale(cls, value: Any) -> float:
        if value is None:
            return 0.0
        return max(0.0, min(10.0, float(value)))

    @field_validator("stg_block_index", mode="before")
    @classmethod
    def _clamp_stg_block_index(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=47, default=28)

    @field_validator("ge_gamma", mode="before")
    @classmethod
    def _clamp_ge_gamma(cls, value: Any) -> float:
        if value is None:
            return 2.0
        return max(0.0, min(10.0, float(value)))


SettingsModelT = TypeVar("SettingsModelT", bound=SettingsBaseModel)
_PARTIAL_MODEL_CACHE: dict[type[SettingsBaseModel], type[SettingsPatchModel]] = {}


def _wrap_optional(annotation: Any) -> Any:
    if type(None) in get_args(annotation):
        return annotation
    return annotation | None


def _to_partial_annotation(annotation: Any) -> Any:
    if _is_settings_model_annotation(annotation):
        return make_partial_model(annotation)
    return annotation


def make_partial_model(model: type[SettingsModelT]) -> type[SettingsPatchModel]:
    cached = _PARTIAL_MODEL_CACHE.get(model)
    if cached is not None:
        return cached

    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_info in model.model_fields.items():
        partial_annotation = _wrap_optional(_to_partial_annotation(field_info.annotation))
        fields[field_name] = (partial_annotation, Field(default=None))

    partial_model = create_model(
        f"{model.__name__}Patch",
        __base__=SettingsPatchModel,
        **cast(Any, fields),
    )

    _PARTIAL_MODEL_CACHE[model] = partial_model
    return partial_model


def _is_settings_model_annotation(annotation: object) -> TypeGuard[type[SettingsBaseModel]]:
    return isinstance(annotation, type) and issubclass(annotation, SettingsBaseModel)


AppSettingsPatch = make_partial_model(AppSettings)
UpdateSettingsRequest = AppSettingsPatch


class SettingsResponse(SettingsBaseModel):
    use_torch_compile: bool = False
    load_on_startup: bool = False
    has_ltx_api_key: bool = False
    user_prefers_ltx_api_video_generations: bool = False
    has_fal_api_key: bool = False
    use_local_text_encoder: bool = False
    fast_model: FastModelSettings = Field(default_factory=FastModelSettings)
    pro_model: ProModelSettings = Field(default_factory=ProModelSettings)
    prompt_cache_size: int = 100
    prompt_enhancer_enabled_t2v: bool = True
    prompt_enhancer_enabled_i2v: bool = False
    has_gemini_api_key: bool = False
    seed_locked: bool = False
    locked_seed: int = 42
    models_dir: str = ""

    # ── VRAM optimisation settings ──────────────────────────────────────
    block_swap_blocks_on_gpu: int = 0
    use_fp8_transformer: bool = False
    attention_tile_size: int = 0
    text_encoder_device: str = ""
    transformer_device: str = ""
    civitai_loras: str = "[]"
    gguf_transformer_path: str = ""
    gguf_per_layer_quant: bool = True
    use_abliterated_encoder: bool = False
    use_multi_gpu: bool = False
    vae_spatial_tile_size: int = 0
    vae_temporal_tile_size: int = 0
    unload_text_encoder_after_encode: bool = False
    enhance_prompt_locally: bool = False
    distilled_num_steps: int = 8
    reload_pipeline_every_n_gens: int = 0
    stg_scale: float = 0.0
    stg_block_index: int = 28
    distilled_sigma_schedule: str = "distilled"
    denoising_loop: str = "euler"
    ge_gamma: float = 2.0
    res2s_bongmath: bool = False
    res2s_bongmath_max_iter: int = 5


def to_settings_response(settings: AppSettings) -> SettingsResponse:
    data = settings.model_dump(by_alias=False)
    ltx_key = data.pop("ltx_api_key", "")
    fal_key = data.pop("fal_api_key", "")
    gemini_key = data.pop("gemini_api_key", "")
    data["has_ltx_api_key"] = bool(ltx_key)
    data["has_fal_api_key"] = bool(fal_key)
    data["has_gemini_api_key"] = bool(gemini_key)
    # models_dir passes through as-is (not secret)
    return SettingsResponse.model_validate(data)


def should_video_generate_with_ltx_api(*, force_api_generations: bool, settings: AppSettings) -> bool:
    has_ltx_api_key = bool(settings.ltx_api_key.strip())
    return force_api_generations or (
        settings.user_prefers_ltx_api_video_generations and has_ltx_api_key
    )
