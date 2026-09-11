"""Configuration loading for LTX-AviUtl2-Bridge.

Reads ``config.yaml`` into typed Pydantic models. This is the single source of
truth for server, model, VRAM, upload, limits and output settings (spec ch.11).

CLI overrides (``--listen``, ``--port``, ``--api-key``, ``--allow-all-cors``)
are applied in ``main.py`` on top of the loaded ``ServerConfig``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, computed_field

from chain_math import CHAIN_COMFORT_TOKEN_BUDGET

logger = logging.getLogger("ltx.config")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

#: ``model:`` keys that USED to hold a fixed default weight path and were moved
#: into the base-model descriptors (``scripts/manifests/*.json``, see
#: :mod:`services.base_models`) by the multi-engine foundation (§3-97 P3b).
#: Pydantic ignores unknown keys, so an old ``config.yaml`` still loads — but it
#: would load with its values SILENTLY IGNORED, which is exactly the kind of
#: quiet mismatch that made a stale path demote the backend to mock before. So
#: :func:`load_config` names each survivor once, at WARNING.
#:
#: ``scripts/install_ltx.ps1`` carries the same list (``$DeprecatedModelKeys``)
#: and DELETES those lines from an existing config.yaml, so re-running
#: setup.bat is what ends the warning for good. The warning stays here for the
#: environments that never re-run the installer; a contract test
#: (tests/test_base_model_contract.py) keeps the two lists identical.
DEPRECATED_MODEL_KEYS: tuple[str, ...] = (
    "gguf_transformer_path",
    "gguf_gemma_path",
    "component_video_vae_path",
    "component_audio_vae_path",
    "component_text_projection_path",
    "component_video_vae_pruned_path",
    "spatial_upsampler_path",
    "gemma_root",
)


class IcLoraEntry(BaseModel):
    """Dict-value form of an ``ic_loras`` registry entry (Phase C).

    Adds a ``preprocess`` kind alongside the safetensors ``path`` so one
    adapter file (e.g. the Union-Control LoRA) can be exposed under several
    logical names that each imply a different raw-video -> control-signal
    conversion in the engine worker. Plain string registry values (Phase B)
    remain valid and are equivalent to ``preprocess="none"``.
    """

    path: str
    preprocess: Literal["none", "canny", "dwpose", "depth"] = "none"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 18620
    allow_all_cors: bool = False
    api_key: str | None = None
    log_dir: str = "./logs"


class ModelConfig(BaseModel):
    checkpoint_name: str = "ltx-2.3-22b-distilled"
    text_encoder: str = "google/gemma-3-12b-it-qat-q4_0-unquantized"
    pipeline_type: str = "distilled"
    auto_load_on_generate: bool = True
    reload_interval: int = 0

    # Step 7 (real LTX) runtime paths. Populated by scripts/install_ltx.ps1 and
    # consumed only by services/ltx_runner.py. None until the model is installed.
    ltx_repo_dir: str = "./vendor/LTX-2"  # reference only (upstream LTX-2 clone).
    backend: str = "auto"  # "auto" | "mock" | "real"

    # NOTE (2026-07-28, PENDING_TASKS.md 3-26): there used to be a
    # `checkpoint_path: str | None = None` field here, pointing at the 43GB
    # monolith physically deleted in Stage 3. It was removed after confirming it
    # is genuinely inert config: services/ltx_runner.py always sent the worker
    # payload's checkpoint_path as "" whenever this field was unset (its only
    # real-world state, since nothing in config.yaml ever set it to a live
    # path), and DistilledPipeline never opens that string -- it only requires
    # it to be a non-None str so ModelLedger.build_model_builders()
    # (ltx_pipelines/utils/model_ledger.py) populates the lazy builder objects
    # that engine/pipeline/fast_video_pipeline.py's GGUF/component re-sourcing
    # later overwrites via dataclasses.replace(). The engine adapter now
    # hardcodes that same "" directly (see its _build_load_payload), so the
    # worker payload is byte-identical to before with no config knob needed.

    # WHERE THE FIXED DEFAULT WEIGHT PATHS WENT (§3-97 P3b): the eight fields
    # that used to spell out the transformer / GGUF Gemma / VAE / text-projection
    # / spatial-upsampler / tokenizer paths live in the BASE-MODEL DESCRIPTORS
    # (scripts/manifests/*.json -> services/base_models.py) now, as
    # ``categories[].default_file`` and ``assets``. A weight path is a property
    # of a base model, not of this server, so a second base model is a new JSON
    # file rather than a second set of config keys. See DEPRECATED_MODEL_KEYS
    # above for what a leftover key in an old config.yaml does (nothing, loudly).
    # Only the two DIRECTORIES below (manifest_dir / models_dir) stay here.

    # First-party engine package (project root ./engine). ltx_runner launches
    # `python -m engine.worker` with this on PYTHONPATH.
    engine_dir: str = "./engine"
    # Interpreter that runs the first-party engine worker (torch + cu128 + ltx_core
    # / ltx_pipelines + gguf). The dedicated ./.venv-engine, relocated out of the
    # (now-deleted) fork tree in Stage 2b. Separate from the app's torch-free
    # ./.venv. Dependency snapshot: engine/venv-engine.freeze.txt.
    engine_python: str = "./.venv-engine/Scripts/python.exe"
    # Interpreter for the LTX 2.5 worker (§3-98). A SECOND venv, not a second
    # setting for the same one: .venv-engine-ltx25 holds official LTX-2 v1.2.0 +
    # transformers 5.x, which cannot coexist with 2.3's transformers 4.57 in one
    # environment — that incompatibility is the whole reason the 2.5 engine is a
    # separate process tree. ``engine_dir`` has no 2.5 twin because the engine25
    # package location is fixed (it ships in this repository); only the
    # interpreter is an installation detail an operator may have to point
    # elsewhere. Consumed by services/engines/ltx25/adapter.py.
    engine_python_ltx25: str = "./.venv-engine-ltx25/Scripts/python.exe"
    gguf_per_layer_quant: bool = True

    # IC-LoRA adapter registry (Phase B, extended Phase C). Maps a server-side
    # adapter NAME (what the API accepts in GenerateRequest.loras[].name — never
    # a filesystem path) to either a bare safetensors path (string, legacy Phase B
    # form, implies preprocess="none") or an IcLoraEntry (Phase C: path +
    # preprocess kind, for control adapters like Union-Control that need a raw
    # reference video converted to a control signal before use). Absent/empty
    # section -> any loras request is rejected (fail loud, no silent skip).
    ic_loras: dict[str, str | IcLoraEntry] = Field(default_factory=dict)

    # Style / character LoRA directory (S1). A drop-in folder scanned by
    # services.lora_registry.LoraRegistry: every ``*.safetensors`` here is
    # exposed under its filename stem as an additional selectable adapter WITHOUT
    # a config edit (mirrors the model-registry directory scan). config.model.ic_loras
    # stays authoritative — a scanned file whose stem (or on-disk path) collides
    # with a registered adapter yields to the registration. A ``<stem>.png`` next
    # to the weight file is served as its GUI thumbnail. Directory-scanned entries
    # are style adapters unless their safetensors metadata carries
    # ``reference_downscale_factor`` (then control). Absent/empty directory -> no
    # scan entries (fresh checkout tolerated).
    lora_dir: str = "./models/LTX23/StyleLoRA"

    # Model-management registries (additive, Docs/MODEL_MANAGEMENT_DESIGN.md).
    # Category-scoped NAME -> path maps mirroring ic_loras: a server-side model
    # NAME (what GET /models lists and POST /pipeline/load accepts in its
    # optional ``models`` block — never a filesystem path) to a project-relative
    # (or absolute) weight file. Absent/empty sections are the norm:
    # services/model_registry.py always injects a "default" entry per category
    # from the fixed default-path fields above (so the default combination stays
    # byte-identical), and directory scanning discovers additional files in the
    # existing layout without any config edit.
    transformers: dict[str, str] = Field(default_factory=dict)
    text_encoders: dict[str, str] = Field(default_factory=dict)
    video_vaes: dict[str, str] = Field(default_factory=dict)
    audio_models: dict[str, str] = Field(default_factory=dict)

    # Multi-engine foundation (Docs/MULTI_ENGINE_DESIGN.md §4/§5.1). The base
    # models this server knows about are declared by the JSON descriptors in
    # ``manifest_dir`` (services/base_models.py), and every path inside a
    # descriptor is relative to ``models_dir`` — the root of the model store.
    # Named models_dir (not checkpoint_dir): it is the directory the installer
    # populates, not a single checkpoint. Both are directories, not weight
    # files, and both stay in config: the installer and the server must agree
    # on WHERE to look, while WHAT to look for moved into the descriptors.
    manifest_dir: str = "./scripts/manifests"
    models_dir: str = "./models"


class VramConfig(BaseModel):
    low_vram_mode: bool = True
    low_vram_profile: str = "16gb_safe"
    # fp8_transformer / cpu_offload_text_encoder: FROZEN API CONTRACT. Both are
    # emitted by services/low_vram.py (_STATUS_KEYS + metadata_block) into
    # GET /status and metadata.json, so they must NOT be removed. Neither is
    # propagated to the worker payload: fp8 is chosen at runtime by
    # device_supports_fp8 auto-detection, and CPU text-encode offload is driven
    # by the separate te_offload_text_encoder field (LTX_TE_OFFLOAD env) below.
    fp8_transformer: bool = True
    cpu_offload_text_encoder: bool = True
    # Sequential per-layer CPU offload of the GGUF Gemma during text-encode
    # (caps the ~15GB encode peak to ~a few GB; compute stays on GPU). Wired to
    # the real worker via LTX_TE_OFFLOAD. Default ON.
    te_offload_text_encoder: bool = True
    # Build the DiT (transformer) on CPU and move only non-block submodules to
    # GPU, eliminating the ~16.9GB load-time GPU spike. Wired to the real worker
    # via LTX_DIT_CPU_LOAD. Default ON.
    dit_cpu_load: bool = True
    vae_tiling: bool = True
    attention_tiling: bool = False
    attention_tile_size: int | None = None
    block_swap: bool = False
    block_swap_blocks_on_gpu: int | None = None
    # VAE tiling sizes for the real GGUF engine (0 -> engine default, proven to
    # fit 16GB at small resolutions). Consumed by the subprocess worker.
    vae_spatial_tile_size: int = 0
    vae_temporal_tile_size: int = 0
    allow_disable_low_vram: bool = True
    # Phase 1 gate: re-source VIDEO VAE + AUDIO VAE/vocoder from standalone
    # component files (model.component_*_path) instead of the 46GB monolith.
    # Off by default; flip to True to exercise the component-file path.
    use_component_files: bool = False


class CropOutputPreset(BaseModel):
    width: int
    height: int


class GenerationPreset(BaseModel):
    width: int
    height: int
    crop_output: CropOutputPreset | None = None
    num_frames: int


class GenerationDefaults(BaseModel):
    width: int = 512
    height: int = 288
    crop_output: CropOutputPreset | None = None
    num_frames: int = 49
    frame_rate: float = 24.0
    num_inference_steps: int = 8
    guidance_scale: float = 1.0
    seed: int = -1
    pipeline: str = "distilled"
    conditioning_images: list[Any] = Field(default_factory=list)


class UploadConfig(BaseModel):
    dir: str = "./uploads"
    max_image_size_mb: int = 20
    allowed_image_extensions: list[str] = Field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"]
    )
    normalize_to_png: bool = True
    # Reference-video upload (Phase B, POST /upload/video). Stored as-is (no
    # re-encode) under uploads/videos/{video_id}/; the engine's ffmpeg-based
    # video IO reads these containers.
    max_video_size_mb: int = 200
    allowed_video_extensions: list[str] = Field(
        default_factory=lambda: [".mp4", ".mov", ".webm", ".mkv"]
    )
    # Audio-to-video upload (A2V, POST /upload/audio). Stored as-is (no
    # re-encode) under uploads/audios/{audio_id}/; the engine's PyAV-based audio
    # decode (decode_audio_from_file) reads these containers. Codec validity is
    # verified at preflight (ffprobe), not on upload — only extension + size gate.
    max_audio_size_mb: int = 50
    allowed_audio_extensions: list[str] = Field(
        default_factory=lambda: [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"]
    )


class ComfortRow(BaseModel):
    """One comfort-budget row for a single engine family (comfort table,
    2026-08-31 recalibration).

    ``requires`` is written in SERVER vocabulary — the same request field
    names a ``/generate`` acceleration toggle uses (``attention_backend``,
    ``block_swap_prefetch``, ``keep_resident``, ``fused_gguf_dequant_kernel``,
    ``vae_mode``, ...) — so a client builds its own "effective acceleration"
    dict in the same vocabulary and compares it key-for-key. A profile's
    ``rows`` are tried top-down; the FIRST row whose ``requires`` fully
    matches wins. No row matches -> the client falls back to the legacy
    ``spill_free_frames`` table (today's behavior for everyone), so a family
    with no rows at all, or none that match, is never left without a number.
    """

    requires: dict[str, str | bool] = Field(default_factory=dict)
    single_budget: int
    chain_budget: int


class EngineComfortProfile(BaseModel):
    """The comfort-budget rows for one engine family
    (``services.engines.FAMILY_BY_ID`` — "ltx"/"ltx25" — is the id source of
    truth).

    ``spatial_factor``/``temporal_factor`` are the token formula's divisors,
    same formula as ``chain_comfort_token_budget``/``single_comfort_token_budget``
    below: a token is (width // spatial_factor) * (height // spatial_factor)
    per temporal_factor latent frames. Carried per-profile (not a global
    constant) so a future model with a different latent compression ratio can
    override them without touching the client's formula.

    ``outpaint_budget`` is the Outpainting (Edit tab) comfort-token budget
    for this family — a fixed per-family line that deliberately does NOT
    vary with the acceleration configuration, and NOT a general rule:
    keeping the all-on-measured line for every configuration is a
    per-family ruling for the two families below (J1, 2026-09-05 —
    Docs/VERIFICATION_LOG.md §98.11 / Docs/COMFORT_LIMIT_TABLE.md §9.2),
    so when a new family is added, LEAVE THIS ``None`` until that family
    has been calibrated and has received its own ruling ("do not copy the
    precedent" is part of J1). ``None`` means "no line": the client draws
    no outpaint comfort warning at all. Advisory-only like ``rows`` — the
    server never validates a request against it. Single source of truth
    for the numbers: Docs/COMFORT_LIMIT_TABLE.md §9.
    """

    spatial_factor: int = 32
    temporal_factor: int = 8
    rows: list[ComfortRow] = Field(default_factory=list)
    outpaint_budget: int | None = None


def _default_comfort_budgets() -> dict[str, EngineComfortProfile]:
    """Comfort-budget table calibrated 2026-08-31 (real-device run, 37 jobs +
    1 submission failure; primary record
    ``outputs/comfort-calib-2026-08-31/RESULTS.md``; single source of truth
    for every number here: Docs/COMFORT_LIMIT_TABLE.md).

    "ltx" (LTX 2.3) INTENTIONALLY HAS NO EMPTY-``requires`` ROW. With the
    default (non-pruned) VAE decoder, LTX 2.3's comfort boundary is NOT
    monotone in tokens — it coincides with the decode chunk-count increments
    (7->8 / 4->5 / 2->3 at 720p/1080p/1440p) rather than a clean token
    ceiling, so there is no single number that is safe below it and unsafe
    above it except under the one fully-accelerated row below (pruned VAE
    decoder included, where the boundary IS monotone). DO NOT "fix" this by
    adding a default row for LTX 2.3 — a client with no matching row is
    expected to fall back to the legacy ``spill_free_frames`` table, which
    stays calibrated for the default configuration precisely because this
    row does not cover it.

    "ltx25" (LTX 2.5) has no such boundary — VRAM is identical across every
    acceleration combination measured — so its one row has empty
    ``requires`` (always matches) and reuses the same 44,880 ceiling for
    both Single and Chained (Chained's legacy 40,000 was a 2.3 measurement
    carried over; 2.5 gets its own number here).

    ``outpaint_budget`` (§3-135, 2026-09-05): the Outpainting lines from
    the 2026-09-05 all-on recalibration (primary record
    ``outputs/comfort-calib-2026-09-05/``). This field is only the
    delivery path for the two families that already have a ruling; the
    numbers' home is Docs/COMFORT_LIMIT_TABLE.md §9.
    """
    return {
        "ltx": EngineComfortProfile(
            rows=[
                ComfortRow(
                    requires={
                        "attention_backend": "sage",
                        "block_swap_prefetch": True,
                        "keep_resident": True,
                        "fused_gguf_dequant_kernel": True,
                        "vae_mode": "prune_vaed",
                    },
                    single_budget=44880,
                    chain_budget=CHAIN_COMFORT_TOKEN_BUDGET,
                )
            ],
            outpaint_budget=42240,  # Docs/COMFORT_LIMIT_TABLE.md §9 (2026-09-05 all-on run)
        ),
        "ltx25": EngineComfortProfile(
            rows=[ComfortRow(requires={}, single_budget=44880, chain_budget=44880)],
            outpaint_budget=46080,  # Docs/COMFORT_LIMIT_TABLE.md §9 (2026-09-05 all-on run)
        ),
    }


# キーフレーム画像（conditioning_images）の枚数上限。設定項目ではなく契約定数で、
# 公式 LTX Desktop の LOCAL_MULTI_KEYFRAME_MAX_COUNT = 10 に LTX 2.3 / 2.5 共通で揃える。
MAX_CONDITIONING_IMAGES = 10


class LimitsConfig(BaseModel):
    max_width: int = 1920
    max_height: int = 1088
    max_num_frames: int = 481

    @computed_field
    @property
    def max_conditioning_images(self) -> int:
        """入力項目ではない（config.yaml に書かれていても extra='ignore' で無視される）。"""
        return MAX_CONDITIONING_IMAGES

    # frame_idx grid advertised via /config so clients can build the UI grid.
    # Keyframes snap to the latent-frame-START grid: frame_idx 0 is the start-frame
    # (latent-replace path); every OTHER keyframe sits on offset + multiple*n, i.e.
    # the 8n+1 pixels (1, 9, 17, ...). Server snap: (f-1)//8*8+1 clamped in-range.
    conditioning_frame_idx_multiple: int = 8
    conditioning_keyframe_grid_offset: int = 1
    phase1_max_concurrent_jobs: int = 1
    low_vram_disabled_required: bool = False
    # 解像度別 spill-free フレーム数（16GB 実測, §8.4）。API は 481f まで受けるが、
    # これを超えると shared へ溢れ ~2-4x 低速化（OOM せず）→ クライアント UI で警告する。
    # キーは "WxH" 生成サイズ文字列（client が引きやすい形式）。
    spill_free_frames: dict[str, int] = Field(default_factory=dict)
    # V2V continuation (POST /generate/chain source_video.context_frames). Bounds
    # advertised via /config so a UI can build the control. context_frames is 8n+1;
    # the 145 max is a conservative v1 cap (keeps the frozen head inside one
    # stage-2 tile — see api.models.SourceVideoSpec). Defaulted so an old
    # config.yaml (without these keys) still parses (spill_free_frames precedent).
    v2v_context_frames_default: int = 73
    v2v_context_frames_min: int = 25
    v2v_context_frames_max: int = 145
    # End source (POST /generate/chain end_source.context_frames): how many
    # pixel frames at the very END of the chain come from the uploaded video /
    # still. Advertised via /config so a UI can build the control. A MULTIPLE OF
    # 8, not 8n+1 — a tail band is counted back from the end in whole latent
    # groups and never touches the causal VAE's lone keyframe latent (see
    # chain_math.v_tail_latents).
    #
    # THE CLIP COUNT AND THE PRESENCE OF A START SOURCE PICK THE GEOMETRY, AND
    # THE OUTPUT LENGTH DOES NOT CHANGE IN ANY OF THEM. ONE clip -> "in_window":
    # the band is the clip's OWN tail. TWO OR MORE without a source_video ->
    # "reverse": the band is the LAST clip's own tail and the clips are generated
    # last-to-first towards it. TWO OR MORE *with* a source_video -> "bridge":
    # the clips are generated forwards as usual and only the LAST one is
    # conditioned at both ends (のり代 at its head, the band at its tail), so the
    # chain fills the span between the two uploads. (The historical
    # "internal_segment" geometry, which appended the band and grew the output by
    # it, is no longer reachable from the API.)
    #
    # THE DEFAULT 72 IS THE CONTRACT'S DEFAULT, NOT A RECOMMENDED VALUE. The
    # real-run comparison settled on an 8-frame anchor (a longer band spends the
    # window re-rendering the material and costs the generator its invention),
    # and the frontend always sends context_frames=8 explicitly. Sources of
    # truth: Docs/VERIFICATION_LOG.md §61 and api.models.EndSourceSpec.
    #
    # 136 IS THE OPERATIONAL CEILING ON THE AUTOMATIC BAND LENGTH, NOT A
    # GEOMETRIC LIMIT. The band may span as many stage-2 tiles as it needs
    # (ChainLayout.end_tile_bands is the per-tile freeze plan), so no window
    # geometry bounds it any more — the old "8*(v_adv-1), 88 under
    # high_resolution" rule and its per-request 422 are both gone. 136 == 17
    # latent frames ~= 5.67 s at 24 fps is simply the edge of the measured
    # region, kept so nobody ships an unvalidated one. Raising it is a config
    # edit plus a real-run quality gate, and it applies to every stage-2 window
    # preset alike, so a client can now trust end_context_frames_max
    # unconditionally (unlike retake_window_max_frames, which really is
    # preset-dependent). The geometry truth stays in chain_math.
    end_context_frames_default: int = 72
    end_context_frames_min: int = 8
    end_context_frames_max: int = 136
    # Retake window length (POST /generate/chain, clips[0].num_frames when a
    # ``retake`` block is present). Published via /config so a UI can bound its
    # window control. 8n+1 like every other frame count. THESE TWO PUBLISH THE
    # "standard" STAGE-2 WINDOW'S NUMBERS: 169 ==
    # chain_math.retake_max_window_px(STAGE2_V_TILE=22), the largest window that
    # still refines as ONE stage-2 window.
    #
    # The ceiling is NOT a constant any more: retake is allowed with
    # stage2_window="high_resolution" (v_tile=19), where the real ceiling drops
    # to retake_max_window_px(19) = 145. The server enforces the per-preset
    # bound in chain_math.compute_chain_layout (a longer window is a 422 naming
    # the concrete ceiling); a CLIENT that offers the narrower window must
    # mirror the same 8*v_tile-7 formula for its own slider bound rather than
    # trusting retake_window_max_frames unconditionally. The floor (73) is
    # preset-independent — it is a quality bound, not a geometric one. The
    # geometry truth stays in chain_math.
    retake_window_min_frames: int = 73
    retake_window_max_frames: int = 169
    # Comfortable attention-token ceiling for ONE stage-2 window of a chain,
    # published so a client can draw its resolution guides from a served number
    # instead of hard-coding one. PURELY CLIENT ADVICE: the server never
    # consults this — no request is rejected, clamped or altered by it — which
    # is why it lives here rather than in any validation path. The default
    # mirrors chain_math.CHAIN_COMFORT_TOKEN_BUDGET, the single source of truth
    # (a token is (width//32) * (height//32) per window latent frame). Lower it
    # on a smaller GPU / raise it on a larger one to move the client's guides;
    # the geometry itself does not change.
    chain_comfort_token_budget: int = CHAIN_COMFORT_TOKEN_BUDGET
    # Comfortable attention-token ceiling for ONE Create (single-shot
    # `/generate`) request. A single request refines its whole clip in ONE
    # pass (no stage-2 tiling), which is a DIFFERENT workload from
    # chain_comfort_token_budget above (one chain stage-2 window) — the two
    # are separate axes with separate calibrated values, never to be confused.
    # PURELY CLIENT ADVICE, same discipline as chain_comfort_token_budget: the
    # server never consults this — no request is rejected, clamped or altered
    # by it. Published only so the WebUI's Create screen can draw a smart,
    # resolution-exact comfort marker instead of its coarse 5-key
    # spill_free_frames lookup; the client only switches to this derivation
    # while all five acceleration toggles (sage, block_swap_prefetch,
    # keep_resident, fused_gguf_dequant_kernel, vae_mode=prune_vaed) are on —
    # with even one off it falls back to spill_free_frames instead.
    # Literal (no chain_math constant to mirror): calibrated 2026-08-18 from a
    # 4-stage/21-job real-device run across 3 resolutions x both orientations.
    # The token formula is the same as chain_comfort_token_budget's:
    # (width//32) * (height//32) * latent frame count. 44,880 is the largest
    # common comfortable value, anchored at M2 = 1920x1088, 169 frames
    # (exactly 44,880 tokens). Source of truth and the full derivation table:
    # Docs/COMFORT_LIMIT_TABLE.md.
    single_comfort_token_budget: int = 44880
    # Server-side comfort-budget TABLE, keyed by engine family id
    # (services.engines.FAMILY_BY_ID: "ltx"/"ltx25"), superseding the single
    # fixed value above for clients that understand it. Same advisory
    # discipline as chain_comfort_token_budget/single_comfort_token_budget:
    # the server never consults this — no request is rejected, clamped or
    # altered by it. It exists so a client looks up {requires, single_budget,
    # chain_budget} rows per engine family instead of hard-coding "all five
    # acceleration toggles on" the way it does today, and so a future engine
    # or toggle needs only a new row/family here, not a client code change.
    # NOT written to config.yaml (code default only — an operator CAN still
    # override it there like any other field) and carries no validator; a
    # client with no matching row is responsible for falling back to
    # spill_free_frames, exactly as it does today. See
    # _default_comfort_budgets for the default table and its rationale, and
    # Docs/COMFORT_LIMIT_TABLE.md for the calibration.
    # A yaml override REPLACES the whole table (no per-family/per-field
    # merge — pinned by tests/test_comfort_budgets.py), so an override that
    # omits ``outpaint_budget`` silently drops the outpaint line (``None`` =
    # the client draws no outpaint comfort warning).
    comfort_budgets: dict[str, EngineComfortProfile] = Field(
        default_factory=_default_comfort_budgets
    )


class OutputConfig(BaseModel):
    dir: str = "./outputs"
    format: str = "mp4"
    save_metadata_json: bool = True
    keep_raw_frames: bool = False


class TrackingConfig(BaseModel):
    """Object tracking (§3-54) — the utility-AI module, NOT a base model.

    FOUR KEYS, AND THAT IS THE WHOLE SECTION. Everything else about tracking is
    a module constant in ``services/tracking_manager.py`` (the 60s idle expiry,
    the 64 MB frame ceiling) or a plugin-side setting the server never sees (the
    lost threshold, smoothing, keyframe stride). A knob here would be a promise
    to support every value of it.

    This section is deliberately ABSENT from ``config.yaml.example``: tracking is
    an opt-in install (``install-UETrack.bat``) whose defaults are correct for
    every machine that ran it, so an operator has nothing to fill in.

    See Docs/OBJECT_TRACKING_DESIGN.md §6.4.
    """

    # "uetrack" (the real CPU worker) or "mock" (in-process fake, used by the
    # test suite and for wiring checks on a machine with no weights). Any other
    # value takes the real path — same loose typing as ``ModelConfig.backend``,
    # and a typo lands on the worker, which then reports itself as not
    # installed rather than failing the whole server at boot.
    backend: str = "uetrack"
    # Interpreter that runs `python -m tracking.worker`: the dedicated CPU-only
    # ./.venv-utils. A THIRD venv next to the two engine ones, for the same
    # reason they are separate from each other — CPU torch cannot share an
    # environment with the cu128 builds. Same shape as
    # ``ModelConfig.engine_python``; dependency snapshot:
    # tracking/venv-utils.freeze.txt.
    utils_python: str = "./.venv-utils/Scripts/python.exe"
    # Weight location, relative to ``model.models_dir`` — alongside
    # Preprocessors/, i.e. in the part of the model store that belongs to no
    # base model, because the tracker is not one.
    model_dir: str = "UETrack"
    checkpoint: str = "uetrack_base.safetensors"


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    vram: VramConfig = Field(default_factory=VramConfig)
    generation_presets: dict[str, GenerationPreset] = Field(default_factory=dict)
    generation_defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    upload: UploadConfig = Field(default_factory=UploadConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    # Server RUNTIME STATE file (§3-97 P5, services/runtime_state.py): the last
    # active base model + per-base category selection, so a restart resumes the
    # combination the operator had chosen. Top level rather than inside a
    # section because it belongs to no one subsystem — and it is a CACHE, not a
    # setting: deleting it is always safe. Project root by default, gitignored.
    state_file: str = "./state.json"

    # ----- convenience path helpers (always absolute, project-rooted) -----

    def _abs(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    @property
    def output_dir(self) -> Path:
        return self._abs(self.output.dir)

    @property
    def upload_dir(self) -> Path:
        return self._abs(self.upload.dir)

    @property
    def log_dir(self) -> Path:
        return self._abs(self.server.log_dir)

    @property
    def manifest_dir(self) -> Path:
        """Directory holding the base-model descriptors (scripts/manifests)."""
        return self._abs(self.model.manifest_dir)

    @property
    def models_dir(self) -> Path:
        """Root of the model store; descriptor paths are relative to it."""
        return self._abs(self.model.models_dir)

    @property
    def state_path(self) -> Path:
        """Absolute path of the runtime-state file (see :attr:`state_file`)."""
        return self._abs(self.state_file)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load ``config.yaml`` into an :class:`AppConfig`.

    Missing file falls back to model defaults so the server can still boot.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return AppConfig()
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    _warn_deprecated_model_keys(raw)
    return AppConfig.model_validate(raw)


def _warn_deprecated_model_keys(raw: Any) -> None:
    """Name every ``model:`` key that moved into the base-model descriptors.

    Pydantic's default ``extra='ignore'`` means such a key is harmless — the
    file still loads and the server still starts. It is also invisible, which
    is the problem: the operator edits a path, nothing changes, and the reason
    is nowhere on screen. One WARNING per surviving key says where it went.
    """
    model = raw.get("model") if isinstance(raw, dict) else None
    if not isinstance(model, dict):
        return
    for key in DEPRECATED_MODEL_KEYS:
        if key in model:
            logger.warning(
                "config.yaml の model.%s は廃止され、記述子(scripts/manifests)側へ"
                "移りました。値は無視されます。",
                key,
            )
