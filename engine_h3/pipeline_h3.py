"""MiniMax H3 T2V / I2V pipeline: encode -> denoise -> decode, one stage at a time.

Ported in part from DiffSynth-Studio (Apache-2.0), commit 4e34969:
  diffsynth/pipelines/minimax_h3_audio_video.py
    (the noise initializer 245-250, the two-schedule loop 184-198, the per-row
     timesteps and their ``unique`` index 1011-1021)
Copyright (c) the DiffSynth-Studio authors. Licensed under the Apache License,
Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Changes: T2V and
the keyframe I2V only (the condition rows: KeyframeEncoder 440-467, the
packed layout 651-709, the per-row timesteps 1019); no CFG (H3 is guidance-distilled: one forward per step); no padding rows; the
noise is drawn in float32 from one CPU generator (video, then audio); the
model's output is used in its own direction (x0 = x + sigma v, no negation);
every stage builds its model on meta, reads its weights from the file, and
frees it (or, under ``keep_resident``, parks it on the CPU) before the next
stage starts, so the stages' peaks do not stack.

Stages (``progress`` names are the app's shared vocabulary):

1. ``encode``: the Qwen3-VL text tower (:class:`H3TextEncoder`), built on
   meta and read layer by layer straight from the file onto the GPU during its
   one forward (no layer is kept in main memory); released before the DiT is built.
   A job with any key frame adds the key frames' video VAE encode as a 4th
   step of the same progress stage (``image_encode`` in the phases; see I2V
   below) -- also when no key frame goes to the vision tower.
2. ``stage1_denoise``: the DiT. Everything but ``blocks`` goes to the GPU; the
   50 blocks stay on the CPU and ``BlockSwapService`` pages them in (with the
   prefetch engine when asked). That is also what puts each int8 weight AND
   its scale on the GPU before the block's forward -- ``comfy_kitchen``'s CUDA
   backend refuses a CPU weight, and with triton / eager disabled the call
   would raise rather than silently run slowly.
   Each step: video rows and audio rows get their own t = 1 - sigma (the
   job's ``shift_video`` / ``shift_audio``, same step index); text rows share
   the video t and the I2V condition rows carry ``max(t_video, 0.999)``; the
   distinct values (at most 3) are numbered with ``torch.unique`` (fewer when
   values coincide, e.g. M = 1 at step 0 of a T2V job, where both sigmas are 1)
   and each row's AdaLN row is ``number * 3 + modality``.
   VRAM 16GB mode (``vram16_mode``, ``engine_h3.lowvram``; Docs/MINIMAX_H3_DESIGN.md
   §6.5): the activations are cut into head groups and row chunks, the
   allocator is capped for the denoise (the cap is derived at run time from
   ``mem_get_info``), the swap window is at most 4 blocks, and the token
   refiner and ``condition_proj`` leave the GPU after ``refine_text`` (to the
   CPU for a kept DiT, to meta otherwise). Off, this stage is as described above.
3. ``decode``: the video VAE's decoder, then the audio VAE, then the mp4.

I2V (Docs/MINIMAX_H3_DESIGN.md §3): the key frames (any pixel frames 0..F-1,
sorted by frame) are put onto the canvas (cover + centre crop,
``engine_h3.image_cond``) before anything is loaded; the text encoder sees the
first (frame 0) and the last (frame F-1) through its vision tower, as
``<Picture 1>``, ``<Picture 2>`` in that order (a middle key frame only gets
condition rows); after the text encoder is gone the video VAE's encoder
(float32, built for this job only) turns each into one latent frame, i.e. one
block of ``grid`` condition rows per key frame, ``[text | cond | audio |
video]``. Those rows are mixed once with noise drawn AFTER the video and audio
noise (one ``[grid, 96]`` draw shared by every block:
``0.999 * clean + 0.001 * noise``), carry t = max(t_video, 0.999) and the
RoPE time of their pixel frame (``engine_h3.rope.keyframe_t``), and are
handed to the DiT unchanged at every step.

Keeping models between jobs (``keep_resident``, Docs/MINIMAX_H3_DESIGN.md §4): at the end of
a job that asks for it, the DiT (all of it on the CPU, block swap wrappers
stripped), the video decoder and the audio VAE (both moved to the CPU) are
kept; nothing stays on the GPU. A job without the key reuses what is kept and
releases it at its end; a failed job releases it at once.

Inference is wrapped in ``torch.no_grad()`` only -- never the inference-mode context
(the block swap and the prefetch engine rebind ``.data`` on tensors, which an
inference tensor refuses; Docs/MULTI_ENGINE_DESIGN.md §5.8).
"""

from __future__ import annotations

import gc
import logging
import math
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import torch

import h3_format
import sft_quant_format
from engine_h3.constants import (
    ATTENTION_BACKENDS,
    AUDIO_SAMPLE_RATE,
    DEFAULT_SHIFT_AUDIO,
    DEFAULT_SHIFT_VIDEO,
    DEFAULT_STEPS,
    FRAME_RATE,
    KEYFRAME_NOISE_AUG,
    MODALITY_AUDIO,
    MODALITY_VIDEO,
    SPATIAL_FACTOR,
    VAE_SPATIAL,
    VRAM16_CAP_MARGIN_MIB,
    audio_latents,
    latent_frames,
)
from engine_h3.scheduler import euler_step, sigma_schedule

logger = logging.getLogger(__name__)

#: Smallest frame count the video VAE decodes: n = 1 of 17n + 5 (T' = 7, one
#: decode window). F = 5 (n = 0, T' = 2) passes the 17n+5 rule but leaves the
#: decoder no window, so it is refused BEFORE the text encoder and the DiT run.
MIN_NUM_FRAMES = 22


@dataclass
class H3Paths:
    transformer: str
    text_encoder: str
    video_vae: str
    audio_vae: str
    tokenizer_dir: str


@dataclass
class H3LoadOptions:
    """Load-time knobs (one value per worker process).

    * ``blocks_on_gpu``: DiT blocks kept on the GPU by the block swap window.
      ``<= 0`` or ``>= 50`` means NO swapping: the whole DiT (19.5 GiB) goes to
      the GPU and ``block_swap_prefetch_used`` is ``"off"``.
    * ``deterministic``: sets cuDNN's deterministic / no-benchmark flags. Only
      the convolutions use cuDNN: the VAEs (the decode, and the I2V key frames'
      encode) and the vision tower's patch embedding (``Conv3d``), which is read
      only when a key frame at frame 0 or F-1 goes to the text encoder's
      image-reading part (a middle-only job reads no vision tower, as in T2V); the
      DiT and the text tower have none for it to act on.
    """

    blocks_on_gpu: int = 8
    attention_backend: str = "sdpa"
    block_swap_prefetch: bool = True
    deterministic: bool = False


@dataclass(frozen=True)
class H3ImageInput:
    """One conditioning image: a file and the pixel frame it pins (0 .. num_frames-1)."""

    path: str
    frame_idx: int = 0


@dataclass(frozen=True)
class H3LoraInput:
    """One LoRA file and its strength (engine_h3.lora; 台帳 §1-104)."""

    path: str
    strength: float = 1.0


@dataclass
class H3GenerateRequest:
    prompt: str
    seed: int
    width: int
    height: int
    num_frames: int
    output_path: str
    frame_rate: int = FRAME_RATE
    num_inference_steps: int = DEFAULT_STEPS
    shift_video: float = DEFAULT_SHIFT_VIDEO
    shift_audio: float = DEFAULT_SHIFT_AUDIO
    #: 0 images = T2V; 1 or more (distinct frames 0 .. num_frames-1, any order) = I2V.
    images: tuple[H3ImageInput, ...] = ()
    #: Keep the DiT / video decoder / audio VAE (on the CPU) for the next job.
    keep_resident: bool = False
    #: Per job; ``None`` = the load-time value.
    attention_backend: str | None = None
    block_swap_prefetch: bool | None = None
    #: VRAM 16GB mode (engine_h3.lowvram, 台帳 §1-101). The worker's default is off;
    #: the app's default (on for H3) is filled in by its adapter, which always sends it.
    vram16_mode: bool = False
    #: LoRA files applied to the DiT for this job only (engine_h3.lora); () = none.
    loras: tuple[H3LoraInput, ...] = ()


# --------------------------------------------------------------------------- #
# pure helpers (CPU-testable)
# --------------------------------------------------------------------------- #

def validate_request(req: H3GenerateRequest) -> None:
    """The request's own constraints, refused in Japanese before anything is loaded."""
    if not str(req.prompt).strip():
        raise ValueError("プロンプトが空です（MiniMax H3 は文章の条件づけが必須です）")
    w, h = int(req.width), int(req.height)
    if w <= 0 or h <= 0 or w % SPATIAL_FACTOR or h % SPATIAL_FACTOR:
        raise ValueError(f"幅と高さは 32 の倍数にしてください（{w}×{h}）")
    latent_frames(int(req.num_frames))  # raises the 17n+5 message
    if int(req.num_frames) < MIN_NUM_FRAMES:
        raise ValueError(
            f"フレーム数は {MIN_NUM_FRAMES} 以上の 17n+5 にしてください（{req.num_frames}。"
            "5 フレームは映像 VAE が復号できません）"
        )
    if float(req.frame_rate) != FRAME_RATE:
        raise ValueError(f"フレームレートは {FRAME_RATE}fps 固定です（{req.frame_rate}）")
    if int(req.num_inference_steps) < 1:
        raise ValueError(f"ステップ数は 1 以上にしてください（{req.num_inference_steps}）")
    if not float(req.shift_video) > 0 or not float(req.shift_audio) > 0:
        raise ValueError(f"shift は正の値にしてください（映像 {req.shift_video}・音声 {req.shift_audio}）")
    last = int(req.num_frames) - 1
    seen: set[int] = set()
    for img in tuple(req.images or ()):
        n = int(img.frame_idx)
        if not 0 <= n <= last:
            raise ValueError(f"キーフレームの位置は 0〜{last} にしてください（frame_idx={img.frame_idx}）")
        if n in seen:
            raise ValueError(f"キーフレームの位置が重複しています（frame_idx={n}）")
        seen.add(n)
        if not str(img.path or "").strip():
            raise ValueError(f"キーフレームの画像のパスが空です（frame_idx={n}）")
    if req.attention_backend is not None and req.attention_backend not in ATTENTION_BACKENDS:
        raise ValueError(f"attention_backend は {ATTENTION_BACKENDS} のどれかです（{req.attention_backend!r}）")
    loras = tuple(req.loras or ())
    if loras:
        from engine_h3.lora import inspect_lora_file  # noqa: PLC0415

        for lora in loras:
            if not str(lora.path or "").strip():
                raise ValueError("LoRA のファイルのパスが空です")
            try:
                s = float(lora.strength)
            except (TypeError, ValueError):
                raise ValueError(f"LoRA の強度は数値で渡してください（{lora.strength!r}）") from None
            if not (math.isfinite(s) and s > 0):
                raise ValueError(f"LoRA の強度は正の値にしてください（{lora.strength}）")
            # The header only: a missing or broken file is refused here, before
            # the text encoder runs (the keys themselves never refuse a job).
            inspect_lora_file(lora.path)


def require_fp32_matmul() -> None:
    """Refuse a process whose float32 matmuls may run in TF32 (the LoRA path adds in f32).

    On torch 2.13 the default ``torch.backends.cuda.matmul.fp32_precision`` is
    "none" (= inherit from ``torch.backends.fp32_precision``, itself "none"), so
    the legacy flags are the effective ones: ``allow_tf32`` False and
    ``get_float32_matmul_precision()`` "highest". Nothing in this engine sets
    them; this only fails loud if something did.
    """
    matmul = torch.backends.cuda.matmul
    allow_tf32 = matmul.allow_tf32
    precision = torch.get_float32_matmul_precision()
    fp32 = getattr(matmul, "fp32_precision", None)
    if allow_tf32 is not False or precision != "highest" or (fp32 is not None and fp32 not in ("ieee", "none")):
        raise RuntimeError(
            "FP32 の行列積が TF32 に落ちる設定になっています（LoRA の FP32 加算が保証できません）: "
            f"allow_tf32={allow_tf32!r}・float32_matmul_precision={precision!r}・"
            f"cuda.matmul.fp32_precision={fp32!r}"
        )


def time_rows(layout, t_video: float, t_audio: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row t -> ``(t_unique [M] f32, time_index [S] long)``.

    Text and video rows carry the video t, audio rows the audio t, the
    condition rows (I2V) ``max(t_video, 0.999)`` (DiffSynth
    pipelines/minimax_h3_audio_video.py:1006-1021). ``unique`` sorts, so the
    numbering is by value, as in the reference; T2V has no condition rows, so
    only the video and audio values are numbered.
    """
    t = torch.full((layout.total,), float(t_video), dtype=torch.float32, device=device)
    t[layout.audio] = float(t_audio)
    t[layout.cond] = max(float(t_video), KEYFRAME_NOISE_AUG)
    t_unique, time_index = torch.unique(t, sorted=True, return_inverse=True)
    return t_unique, time_index.to(torch.long)


def noisy_condition(cond_clean: torch.Tensor, frame_rows: int, generator: torch.Generator, device=None) -> torch.Tensor:
    """The condition rows handed to the DiT: ``0.999 * clean + 0.001 * noise``.

    ``cond_clean`` is K blocks of ``frame_rows`` rows (one per key frame). The
    noise is ONE ``[frame_rows, 96]`` draw from ``generator`` (the job's third
    draw, after the video and the audio noise), repeated for every block -- all
    key frames share it, as in DiffSynth and ComfyUI; with K = 1 this is the
    same draw and the same values as the first-frame-only engine.
    """
    blocks, rem = divmod(int(cond_clean.shape[0]), int(frame_rows))
    if rem or blocks < 1:
        raise RuntimeError(f"condition rows {list(cond_clean.shape)} are not whole blocks of {frame_rows}")
    noise = torch.randn((int(frame_rows), int(cond_clean.shape[1])), generator=generator, dtype=torch.float32)
    noise = noise.to(device if device is not None else cond_clean.device).repeat(blocks, 1)
    clean = cond_clean.to(noise.device, torch.float32)
    return KEYFRAME_NOISE_AUG * clean + (1.0 - KEYFRAME_NOISE_AUG) * noise


def modality_rows(layout, text_tags: torch.Tensor) -> torch.Tensor:
    """[S] long: the TE's tags on the text rows, 2 on audio, 0 on cond and video."""
    m = torch.full((layout.total,), MODALITY_VIDEO, dtype=torch.long, device=text_tags.device)
    m[layout.text] = text_tags.to(torch.long)
    m[layout.audio] = MODALITY_AUDIO
    return m


def dit_key_map_for(cfg) -> Callable[[str], str | None]:
    """File name -> H3DiT state-dict key: strip the prefix, drop unknown names."""
    prefix = cfg.prefix
    unknown = set(cfg.unknown)

    def key_map(name: str) -> str | None:
        if not name.startswith(prefix):
            return None
        stripped = name[len(prefix):]
        if stripped in unknown:
            return None
        return stripped

    return key_map


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #

def _gib(n) -> float | None:
    return None if n is None else round(n / 2**30, 2)


def _mem_gib() -> dict:
    """Process memory in GiB: ``rss`` / ``rss_peak`` (working set) and ``commit`` /
    ``commit_peak`` (private commit = PagefileUsage). Empty when unavailable.

    The commit is the better answer to "does this process really hold that
    much memory": the working set can look small after the OS trims it, and
    can stay high while the allocator keeps freed regions.
    psutil when present, else the Win32 counters through ctypes.
    """
    try:
        import psutil  # noqa: PLC0415

        info = psutil.Process().memory_info()
        return {
            "rss": _gib(info.rss),
            "rss_peak": _gib(getattr(info, "peak_wset", None)),
            "commit": _gib(getattr(info, "pagefile", None)),
            "commit_peak": _gib(getattr(info, "peak_pagefile", None)),
        }
    except Exception:  # noqa: BLE001
        pass
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        # Declared types are required: without them the pseudo-handle (-1) is
        # passed as a 32-bit int on x64 and the call fails.
        get_proc = ctypes.windll.kernel32.GetCurrentProcess
        get_proc.restype = wintypes.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        if not get_info(get_proc(), ctypes.byref(pmc), pmc.cb):
            return {}
        return {
            "rss": _gib(pmc.WorkingSetSize),
            "rss_peak": _gib(pmc.PeakWorkingSetSize),
            "commit": _gib(pmc.PagefileUsage),
            "commit_peak": _gib(pmc.PeakPagefileUsage),
        }
    except Exception:  # noqa: BLE001 -- not Windows, or no psapi: just omit
        return {}


class _Phases:
    """Per-stage seconds, RSS and VRAM peaks for the ``done`` event's ``h3.phases``."""

    def __init__(self, cuda: bool) -> None:
        self.cuda = cuda
        self.data: dict[str, dict] = {}
        self.peak_alloc_mib = 0
        self.peak_reserved_mib = 0

    def mark(self, phase: str, label: str) -> None:
        mem = _mem_gib()
        entry = self.data.setdefault(phase, {})
        for key in ("rss", "commit"):
            if mem.get(key) is not None:
                entry[f"{key}_{label}_gib"] = mem[key]

    @contextmanager
    def phase(self, name: str):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        entry = self.data.setdefault(name, {})
        try:
            yield entry
        finally:
            entry["seconds"] = round(time.perf_counter() - started, 2)
            mem = _mem_gib()
            for key, name in (
                ("rss", "rss_end_gib"), ("rss_peak", "rss_process_peak_gib"),
                ("commit", "commit_end_gib"), ("commit_peak", "commit_process_peak_gib"),
            ):
                if mem.get(key) is not None:
                    entry[name] = mem[key]
            if self.cuda:
                alloc = int(round(torch.cuda.max_memory_allocated() / 2**20))
                reserved = int(round(torch.cuda.max_memory_reserved() / 2**20))
                entry["peak_vram_mb"] = alloc
                entry["peak_vram_reserved_mb"] = reserved
                self.peak_alloc_mib = max(self.peak_alloc_mib, alloc)
                self.peak_reserved_mib = max(self.peak_reserved_mib, reserved)


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


#: VRAM 16GB mode: the allocator counters recorded as differences over the denoise.
_VRAM16_COUNTERS = ("num_alloc_retries", "num_ooms")

#: VRAM 16GB mode's message for a CUDA OOM (裁定 47). Keeps "CUDA out of memory" so
#: the app classifies it as GPU_OOM; the advice rides in the error's ``detail``,
#: which the app folds into the job's ``error`` (``GPU_OOM: ... (detail)``).
VRAM16_OOM_MESSAGE = (
    "VRAM 16GB mode でも GPU メモリが足りません (CUDA out of memory)。"
    "解像度かフレーム数を下げるか、VRAM 16GB mode を切ってください"
)


def _cuda_index(device) -> int:
    """The device as an integer index: ``set_per_process_memory_fraction`` refuses a
    ``torch.device("cuda")`` without one (the pipeline's device usually has none)."""
    d = torch.device(device)
    return d.index if d.index is not None else torch.cuda.current_device()


def _vram16_counters(device) -> dict:
    stats = torch.cuda.memory_stats(device)
    return {key: int(stats.get(key, 0)) for key in _VRAM16_COUNTERS}


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

class H3Pipeline:
    sampler = "euler"

    def __init__(
        self,
        paths: H3Paths,
        options: H3LoadOptions,
        *,
        device,
        progress: Callable[[str, int, int], None],
        log: Callable[[str], None],
    ) -> None:
        self.paths = paths
        self.options = options
        self.device = torch.device(device)
        self._progress = progress
        self._log = log
        self._dit_header = None
        self._dit_cfg = None
        #: ONE resident BlockSwapService per process, as engine25's stage keeps:
        #: its pinned staging pool (2 slots of the largest block, 371.1 MiB each:
        #: VERIFICATION_LOG §168.6) and transfer stream survive the
        #: per-job teardown. Rebuilding the pool every job is what the pool's
        #: own docstring warns against (block_swap_prefetch.py:234-243).
        self._swap_service = None
        #: What ``keep_resident`` kept from the previous job (Docs/MINIMAX_H3_DESIGN.md §4):
        #: ``{"dit", "cfg", "decoder", "audio_vae"}``, every tensor on the CPU, or None.
        self._resident: dict | None = None
        self.quant_backends: dict = {}
        self.build_report: dict = {}

    # -- load: checks only, no weights ------------------------------------------

    def load(self) -> dict:
        p = self.paths
        h3_format.check_model_paths(
            files={
                "transformer_path": p.transformer, "text_encoder_path": p.text_encoder,
                "video_vae_path": p.video_vae, "audio_vae_path": p.audio_vae,
            },
            dirs={"tokenizer_dir": p.tokenizer_dir},
        )
        if self.options.attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"attention_backend は {ATTENTION_BACKENDS} のどれかです（{self.options.attention_backend!r}）"
            )

        dit_header, dit_cfg = h3_format.inspect_h3_dit(p.transformer)
        if dit_cfg.unknown:
            self._log(
                f"WARNING the DiT file has {len(dit_cfg.unknown)} tensor(s) this engine does not know; "
                f"they will be dropped: {', '.join(dit_cfg.unknown[:10])}"
            )
        _te_header, te_cfg = h3_format.inspect_h3_te(p.text_encoder)
        vv = sft_quant_format.read_header(p.video_vae)
        av = sft_quant_format.read_header(p.audio_vae)

        # Needs CUDA: fail the load here, not on the first int8 call of a job.
        from engine_h3.quant_linear import require_cuda_int8  # noqa: PLC0415

        backends = require_cuda_int8()
        require_fp32_matmul()

        if self.options.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self._dit_header, self._dit_cfg = dit_header, dit_cfg
        self.quant_backends = backends
        self.build_report = {
            "dit": {
                "prefix": dit_cfg.prefix,
                "num_layers": dit_cfg.num_layers,
                "hidden_size": dit_cfg.hidden_size,
                "heads": dit_cfg.num_heads,
                "adaln_grid": dit_cfg.adaln_grid,
                "adaln_rank": dit_cfg.adaln_rank,
                "int8_layers": len(dit_cfg.quant),
                "quant_specs": sorted({f"{q.format}/convrot={q.convrot}/g{q.convrot_groupsize}" for q in dit_cfg.quant.values()}),
                "unknown": list(dit_cfg.unknown),
            },
            "text_encoder": {
                "num_layers": te_cfg.num_layers,
                "int8_layers": len(te_cfg.quant),
                "has_visual": te_cfg.has_visual,
                "metadata": te_cfg.metadata,
            },
            "video_vae_tensors": len(vv.tensors),
            "audio_vae_tensors": len(av.tensors),
            "options": {
                "blocks_on_gpu": self.options.blocks_on_gpu,
                "attention_backend": self.options.attention_backend,
                "block_swap_prefetch": self.options.block_swap_prefetch,
                "deterministic": self.options.deterministic,
            },
            "quant_backends": _backend_summary(backends),
            "device": str(self.device),
        }
        return self.build_report

    # -- generate -----------------------------------------------------------------

    def generate(self, req: H3GenerateRequest) -> dict:
        if self._dit_cfg is None:
            raise RuntimeError("generate before load")
        keep = bool(req.keep_resident)
        started = time.perf_counter()
        phases = _Phases(cuda=self.device.type == "cuda")
        ok = False
        lora_report: list[dict] = []
        try:
            # Inside the try: a job refused here is a failed job and releases
            # what is kept, like any other. Still before the text encoder, so a
            # bad request or image never costs a text-encoder read.
            validate_request(req)
            # Ascending frame order (stable): the condition rows and the vision
            # tower's pictures follow it (first frame then last frame, as in the
            # official FL2VA).
            images = sorted(req.images or (), key=lambda img: int(img.frame_idx))
            cond_frame_idx = [int(img.frame_idx) for img in images]
            canvases: list = []
            source_size: list[list[int]] = []
            if images:
                from engine_h3.image_cond import load_canvas_image  # noqa: PLC0415

                for img in images:
                    canvas, size = load_canvas_image(img.path, int(req.width), int(req.height))
                    canvases.append(canvas)
                    source_size.append([int(size[0]), int(size[1])])
            # Only the first and the last frame go to the vision tower; a middle
            # key frame gets condition rows only (MINIMAX_H3_DESIGN 裁定 59, §3).
            last = int(req.num_frames) - 1
            vision_images = [c for n, c in zip(cond_frame_idx, canvases) if n in (0, last)]
            with torch.no_grad():
                with phases.phase("encode") as entry:
                    text_hidden, text_tags = self._encode(
                        req.prompt, phases, images=vision_images, has_cond=bool(canvases),
                    )
                    entry["text_tokens"] = int(text_hidden.shape[0])
                cond_clean = None
                if canvases:
                    with phases.phase("image_encode") as entry:
                        cond_clean = self._encode_image(canvases, entry)
                        entry["vision_tokens"] = int(phases.data["encode"].get("te_read", {}).get("vision_tokens", 0))
                        entry["canvas"] = [int(req.width), int(req.height)]
                        entry["source_size"] = source_size
                del canvases, vision_images
                with phases.phase("denoise") as entry:
                    video_lat, audio_lat, prefetch_used = self._denoise(
                        req, text_hidden, text_tags, phases, entry,
                        cond_clean=cond_clean, cond_frame_idx=cond_frame_idx, keep=keep,
                        lora_report=lora_report,
                    )
                del text_hidden, text_tags, cond_clean
                _free_cuda()
                with phases.phase("decode") as entry:
                    size_bytes = self._decode(req, video_lat, audio_lat, entry, keep=keep)
                del video_lat, audio_lat
            ok = True
        finally:
            if not (ok and keep):
                # No key (the LTX contract: kept only while the key is true), or a
                # failed job: whatever is kept goes now.
                self.release_resident()
            # Also after a failed job: the VRAM it reserved would otherwise stay
            # cached until the next job's encode, competing with AviUtl2's preview.
            _free_cuda()
        mem = _mem_gib()
        h3 = {"phases": phases.data, "quant_backends": _backend_summary(self.quant_backends)}
        if req.loras:
            h3["loras"] = lora_report   # only for a LoRA job: the LoRA-off payload is unchanged
        return {
            "seed_used": int(req.seed),
            "sampler": self.sampler,
            "num_inference_steps": int(req.num_inference_steps),
            "shift_video": float(req.shift_video),
            "shift_audio": float(req.shift_audio),
            "peak_vram_mb": phases.peak_alloc_mib if phases.cuda else None,
            "peak_vram_reserved_mb": phases.peak_reserved_mib if phases.cuda else None,
            "rss_peak_gib": mem.get("rss_peak"),
            "commit_gib": mem.get("commit"),
            "commit_peak_gib": mem.get("commit_peak"),
            "seconds": round(time.perf_counter() - started, 2),
            "num_frames": int(req.num_frames),
            "frame_rate": int(req.frame_rate),
            "width": int(req.width),
            "height": int(req.height),
            "size_bytes": int(size_bytes),
            "attention_used": self._attention_for(req),
            "block_swap_prefetch_used": prefetch_used,
            "keep_resident_used": "on" if keep else "off",
            "vram16_mode_used": "on" if req.vram16_mode else "off",
            "images_used": len(req.images or ()),
            "h3": h3,
        }

    def _attention_for(self, req: H3GenerateRequest) -> str:
        return req.attention_backend or self.options.attention_backend

    def _prefetch_for(self, req: H3GenerateRequest | None) -> bool:
        if req is None or req.block_swap_prefetch is None:
            return bool(self.options.block_swap_prefetch)
        return bool(req.block_swap_prefetch)

    # -- stage 1: encode ---------------------------------------------------------

    def _encode(
        self, prompt: str, phases: _Phases, images=(), *, has_cond: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The text encoder's one forward. ``images``: the canvases the vision
        tower reads (the first / last key frame); ``has_cond``: whether the job
        has any key frame at all, which decides the stage's step count."""
        from engine_h3.text_encoder import H3TextEncoder  # noqa: PLC0415

        images = list(images)
        te = H3TextEncoder(self.paths.text_encoder, self.paths.tokenizer_dir, device=self.device)
        # A job with any key frame adds a 4th step to the same stage: the key
        # frames' VAE encode (also when none of them goes to the vision tower,
        # so a middle-only job never shows 3/3 then 4/4).
        total = 4 if has_cond else 3
        self._progress("encode", 0, total)
        try:
            info = te.load()
            phases.data["encode"]["te"] = info
            phases.mark("encode", "after_load")
            self._progress("encode", 1, total)
            hidden, tags = te.encode(prompt, images=images)
            phases.data["encode"]["te_read"] = te.read_report
            self._progress("encode", 2, total)
        finally:
            te.release()
            del te
            _free_cuda()
        phases.mark("encode", "after_release")
        self._progress("encode", 3, total)
        self._log(
            f"encode: {hidden.shape[0]} text tokens"
            + (f" (with {len(images)} key frame vision block(s))" if images else "")
        )
        return hidden, tags

    def _encode_image(self, images, entry: dict) -> torch.Tensor:
        """The key frames (in order) -> their clean condition rows ``[K * grid, 96]``
        float32 on the device, one block of ``grid`` rows per key frame.

        The video VAE's encoder is built once for this job only, in float32,
        encodes the frames one at a time, and is dropped before the DiT is
        built -- it is never kept.
        """
        from engine_h3.dit import patchify_video  # noqa: PLC0415
        from engine_h3.image_cond import canvas_to_tensor  # noqa: PLC0415
        from engine_h3.video_vae import H3VideoEncoder  # noqa: PLC0415

        blocks: list[torch.Tensor] = []
        enc = H3VideoEncoder.from_file(self.paths.video_vae, device=self.device, dtype=torch.float32)
        try:
            for image in images:
                z = enc.encode(canvas_to_tensor(image).to(self.device), tiled=True)   # [1, 24, 1, H/16, W/16] f32
                if tuple(z.shape[:3]) != (1, 24, 1):
                    raise RuntimeError(f"image encode: unexpected latent shape {list(z.shape)}")
                blocks.append(patchify_video(z.float()))
                entry["latent_shape"] = [int(v) for v in z.shape]
                del z
        finally:
            del enc
            _free_cuda()
        rows = torch.cat(blocks, dim=0)
        self._progress("encode", 4, 4)
        return rows

    # -- stage 2: denoise (progress ``stage1_denoise``) --------------------------

    def _build_dit(self, entry: dict):
        from engine.sft_quant.sft_reader import TORCH_DTYPES  # noqa: PLC0415
        from engine_h3.dit import H3DiT  # noqa: PLC0415
        from engine_h3.quant_linear import swap_in_int8_linears  # noqa: PLC0415
        from engine_h3.sft_load import load_streaming  # noqa: PLC0415

        # Re-read the header: the file may have been replaced since load, and
        # the offsets must be the ones of the file about to be streamed.
        header, cfg = h3_format.inspect_h3_dit(self.paths.transformer)
        self._dit_header, self._dit_cfg = header, cfg

        def bias_dtype_of(name: str) -> torch.dtype:
            info = header.tensors.get(f"{cfg.prefix}{name}.bias")
            return TORCH_DTYPES.get(info.dtype, torch.bfloat16) if info is not None else torch.bfloat16

        with torch.device("meta"):
            dit = H3DiT(cfg)
            swapped = swap_in_int8_linears(dit, cfg.quant, bias_dtype_of=bias_dtype_of)
        report = load_streaming(dit, self.paths.transformer, header, key_map=dit_key_map_for(cfg))
        # assign=True kept the file's dtypes; bring the non-island tensors to
        # the dtypes the forward expects (a no-op on the official file).
        dit.normalize_dtypes()
        dit.requires_grad_(False)
        dit.eval()
        entry["dit"] = {
            "int8_linears": swapped,
            "loaded_tensors": report.loaded,
            "loaded_gib": round(report.bytes / 2**30, 2),
            "read_seconds": report.seconds,
        }
        return dit, cfg

    @staticmethod
    def _unpatch_block_swap(dit) -> int:
        """Put every block's original ``forward`` back and drop the swap marker.

        ``BlockSwapService.release_installed()`` only lets go of its reference;
        the wrappers stay on the blocks, and installing again on the same DiT
        would wrap a wrapper (the synchronous path saves the CURRENT forward as
        the "original") or drive a torn-down prefetch engine. Same idea as LTX
        2.5's ``_unpatch_block_swap`` (engine25/gguf_transformer.py:1149-1177):
        touches no tensor, only the Python attribute and the shared module's
        own marker (imported, not changed).
        """
        from engine.transformer.block_swap_service import _BLOCK_SWAP_ATTR  # noqa: PLC0415

        cleaned = 0
        for block in dit.blocks:
            original = getattr(block, _BLOCK_SWAP_ATTR, None)
            if original is None:
                continue
            block.forward = original  # type: ignore[method-assign]
            delattr(block, _BLOCK_SWAP_ATTR)
            cleaned += 1
        return cleaned

    def _place_dit(
        self, dit, prefetch: bool | None = None, blocks_on_gpu: int | None = None
    ) -> tuple[object | None, str]:
        """Non-block tensors to the GPU; blocks stay on the CPU behind the swap service.

        Mirrors ``engine25/gguf_transformer.py:_place_transformer``: the blocks
        are skipped by module id, the service gets ``prefetch_requested`` BEFORE
        ``install`` (which reads it), and ``hold_arenas=True``: the ring is
        dropped by ``teardown_prefetch`` in ``_denoise``'s finally, before the
        decode, so it never sits under a peak outside the denoise (off, the
        peak is inside the denoise, as LTX 2.5's is; with VRAM 16GB mode on,
        the decode allocates more than the denoise, VERIFICATION_LOG §169.5).
        Any wrapper a
        kept DiT still wears is stripped first, so a job never wraps twice
        whatever the previous job's prefetch setting was.

        ``blocks_on_gpu``: this job's swap window (None = the load value). It is
        written to the one resident service before every ``install`` (which
        reads it), so a job's window never leaks into the next job.
        """
        from engine.transformer.block_swap_service import BlockSwapService  # noqa: PLC0415
        from engine_h3.sft_load import move_module_tree  # noqa: PLC0415

        self._unpatch_block_swap(dit)
        blocks = list(dit.blocks)
        n = int(self.options.blocks_on_gpu if blocks_on_gpu is None else blocks_on_gpu)
        if n <= 0 or n >= len(blocks):
            moved = move_module_tree(dit, self.device, skip=set())
            self._log(f"denoise: full residency ({moved} tensors to {self.device}; blocks_on_gpu={n})")
            return None, "off"
        skip = {id(m) for block in blocks for m in block.modules()}
        moved = move_module_tree(dit, self.device, skip=skip)
        self._log(f"denoise: {moved} non-block tensors to {self.device}; {len(blocks)} blocks stay on CPU")
        if self._swap_service is None:
            self._swap_service = BlockSwapService(blocks_on_gpu=n, device=self.device, hold_arenas=True)
        service = self._swap_service
        service.blocks_on_gpu = n
        service.prefetch_requested = bool(self.options.block_swap_prefetch if prefetch is None else prefetch)
        service.install(dit)
        return service, str(service.last_prefetch_used or "off")

    @staticmethod
    def _evict_text_modules(dit, keep: bool) -> str:
        """VRAM 16GB mode: the token refiner and ``condition_proj`` leave the GPU after ``refine_text``.

        Neither runs again in this job. A job that keeps the DiT parks them on
        the CPU (``_park_dit`` would anyway); any other job drops them to meta,
        since its DiT is released at the end of the job.
        """
        from engine_h3.sft_load import drop_to_meta, move_module_tree  # noqa: PLC0415

        modules = (dit.token_refiner, dit.condition_proj)
        if keep:
            for module in modules:
                move_module_tree(module, torch.device("cpu"), skip=set())
            return "cpu"
        for module in modules:
            drop_to_meta(module)
        return "meta"

    @staticmethod
    def _vram16_begin(dev, n_load: int, dit):
        """VRAM 16GB mode's set-up at the head of the denoise (before ``_place_dit``).

        Returns ``(plan, record, counters0, cap_set, blocks_on_gpu)``: the split
        plan, the ``h3.phases.denoise.vram16`` record, the allocator counters
        before the denoise (None off CUDA), whether the cap was set, and the
        job's swap window. On CUDA the allocator is capped (it binds this
        process only, 裁定 47); ``_vram16_finish`` lifts it.
        """
        from engine_h3.lowvram import Vram16Plan, blocks_on_gpu_for, derive_memory_cap  # noqa: PLC0415

        plan = Vram16Plan()
        n_blocks = blocks_on_gpu_for(n_load, num_blocks=len(dit.blocks))
        record = {
            "cap_fraction": None, "cap_mib": None, "blocks_on_gpu": n_blocks,
            "head_group": plan.head_group, "row_chunk": plan.row_chunk,
        }
        counters0 = None
        cap_set = False
        if dev.type == "cuda":
            index = _cuda_index(dev)
            free, total = torch.cuda.mem_get_info(index)
            fraction = derive_memory_cap(free, total, torch.cuda.memory_reserved(index), VRAM16_CAP_MARGIN_MIB * 2**20)
            counters0 = _vram16_counters(index)
            torch.cuda.set_per_process_memory_fraction(fraction, index)
            cap_set = True
            record["cap_fraction"] = round(fraction, 4)
            record["cap_mib"] = int(fraction * total / 2**20)
        return plan, record, counters0, cap_set, n_blocks

    def _vram16_finish(self, dev, cap_set: bool, counters0: dict | None, record: dict) -> None:
        """Lift the allocator cap and record the counters. Never raises (it runs in a finally)."""
        try:
            index = _cuda_index(dev)
            if cap_set:
                torch.cuda.set_per_process_memory_fraction(1.0, index)
            if counters0 is not None:
                now = _vram16_counters(index)
                for key in _VRAM16_COUNTERS:
                    record[key] = now[key] - counters0[key]
                peak = torch.cuda.memory_stats(index).get("inactive_split_bytes.all.peak")
                record["inactive_split_peak_mib"] = None if peak is None else int(round(peak / 2**20))
        except Exception as exc:  # noqa: BLE001 -- must not mask the job's own outcome
            self._log(f"VRAM 16GB mode: lifting the cap / reading the counters failed (ignored): {exc!r}")

    def _park_dit(self, dit) -> int:
        """Everything of a kept DiT back on the CPU, wrappers stripped.

        The prefetch teardown has already pointed the blocks at their CPU
        masters; the synchronous path leaves the last window's blocks on the
        GPU, and the non-block tensors are on the GPU either way -- so every
        tensor not on the CPU is moved. Returns how many were moved.
        """
        from engine_h3.sft_load import move_module_tree  # noqa: PLC0415

        moved = move_module_tree(dit, torch.device("cpu"), skip=set())
        self._unpatch_block_swap(dit)
        return moved

    def _denoise(
        self, req, text_hidden, text_tags, phases: _Phases, entry: dict,
        cond_clean=None, cond_frame_idx=(), keep=False, lora_report=None,
    ):
        from engine_h3.dit import (  # noqa: PLC0415
            H3SeqLayout,
            audio_to_rows,
            patchify_video,
            rows_to_audio,
            unpatchify_video,
        )
        from engine_h3.rope import build_coords, rope_tables  # noqa: PLC0415

        dev = self.device
        steps = int(req.num_inference_steps)
        t_lat = latent_frames(int(req.num_frames))
        a_lat = audio_latents(int(req.num_frames))
        gh, gw = int(req.height) // SPATIAL_FACTOR, int(req.width) // SPATIAL_FACTOR
        lh, lw = int(req.height) // VAE_SPATIAL, int(req.width) // VAE_SPATIAL
        cond_frame_idx = [int(n) for n in cond_frame_idx]
        cond_rows_n = len(cond_frame_idx) * gh * gw
        actual = None if cond_clean is None else tuple(cond_clean.shape)
        if actual != ((cond_rows_n, 4 * 24) if cond_rows_n else None):
            raise RuntimeError(
                f"denoise: condition rows {None if actual is None else list(actual)}, "
                f"expected [{cond_rows_n}, 96] for key frames {cond_frame_idx}"
            )

        resident = self._resident
        if resident is not None:
            dit, cfg = resident["dit"], resident["cfg"]
            entry["dit_source"] = "resident"
        else:
            dit, cfg = self._build_dit(entry)
            entry["dit_source"] = "file"
        phases.mark("denoise", "after_load")
        service = None
        prefetch_used = "off"
        ok = False
        n_blocks = int(self.options.blocks_on_gpu)
        vram16 = vram16_entry = counters0 = None
        cap_set = False
        try:
            if req.loras:
                # Attached on the CPU before the placement, so the block swap
                # (its prefetch plan packs every buffer of a block) carries A / B
                # with their block. The file objects go once attached.
                from engine_h3 import lora as _lora  # noqa: PLC0415

                files = [(_lora.load_lora_file(item.path), float(item.strength)) for item in req.loras]
                report = _lora.attach(dit, files)
                del files
                if lora_report is not None:
                    lora_report.extend(report)
                self._log("denoise: LoRA " + ", ".join(
                    f"{r['name']} x{r['strength']} ({r['pairs_applied']} pairs, {len(r['skipped'])} skipped)"
                    for r in report))
            if req.vram16_mode:
                vram16, vram16_entry, counters0, cap_set, n_blocks = self._vram16_begin(dev, n_blocks, dit)
            service, prefetch_used = self._place_dit(dit, prefetch=self._prefetch_for(req), blocks_on_gpu=n_blocks)
            phases.mark("denoise", "after_place")

            text_refined = dit.refine_text(text_hidden.to(dev))
            if vram16 is not None:
                vram16_entry["text_modules"] = self._evict_text_modules(dit, keep)
            L = int(text_refined.shape[0])
            layout = H3SeqLayout.make(
                text_rows=L, cond_rows=cond_rows_n, audio_rows=2 * a_lat, video_rows=t_lat * gh * gw,
            )
            coords = build_coords(
                text_len=L, latent_frames=t_lat, grid_h=gh, grid_w=gw, audio_latents=a_lat,
                cond_frame_idx=cond_frame_idx,
            )
            rope_cos, rope_sin = rope_tables(coords, dit.rope_inv_freq)
            rope_cos, rope_sin = rope_cos.to(dev), rope_sin.to(dev)
            modality = modality_rows(layout, text_tags.to(dev))

            # Noise: ONE CPU generator, float32, video first then audio (the
            # draw order is part of what "same seed" means for this engine).
            # The condition rows' noise comes THIRD (one block's worth, shared by
            # every key frame), so T2V draws only the first two and its noise
            # does not depend on I2V being supported.
            gen = torch.Generator(device="cpu").manual_seed(int(req.seed))
            video = torch.randn((1, cfg.latents_dim, t_lat, lh, lw), generator=gen, dtype=torch.float32).to(dev)
            audio = torch.randn((2, cfg.audio_latents_dim, a_lat), generator=gen, dtype=torch.float32).to(dev)
            cond = None
            if cond_clean is not None:
                cond = noisy_condition(cond_clean, gh * gw, gen, device=dev)

            sig_v = sigma_schedule(steps, float(req.shift_video))
            sig_a = sigma_schedule(steps, float(req.shift_audio))
            entry["sequence"] = {
                "text": L, "cond": cond_rows_n, "audio": 2 * a_lat, "video": t_lat * gh * gw,
                "total": layout.total, "latent_frames": t_lat, "audio_latents": a_lat, "grid": [gh, gw],
            }
            m_per_step: list[int] = []
            attention = self._attention_for(req)
            self._progress("stage1_denoise", 0, steps)
            for i in range(steps):
                t_v = 1.0 - float(sig_v[i])
                t_a = 1.0 - float(sig_a[i])
                t_unique, time_index = time_rows(layout, t_v, t_a, device=dev)
                m_per_step.append(int(t_unique.numel()))
                # The condition rows go in unchanged at every step (they are not
                # part of the denoised state, so nothing ever updates them).
                tokens = dit.embed_rows(text_refined, audio_to_rows(audio), patchify_video(video), cond_rows=cond)
                box = None
                if vram16 is not None:
                    # Handed over in a box: no name here keeps the embedding alive.
                    box, tokens = [tokens], None
                v_video_rows, v_audio_rows = dit(
                    tokens,
                    layout=layout,
                    t_unique=t_unique,
                    time_index=time_index,
                    modality=modality,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    attention_backend=attention,
                    vram16=vram16,
                    tokens_box=box,
                )
                del tokens
                v_video = unpatchify_video(v_video_rows.to(dev), t_lat, gh, gw)
                v_audio = rows_to_audio(v_audio_rows.to(dev), a_lat)
                video = euler_step(video, v_video, float(sig_v[i]), float(sig_v[i + 1]))
                audio = euler_step(audio, v_audio, float(sig_a[i]), float(sig_a[i + 1]))
                del v_video_rows, v_audio_rows, v_video, v_audio
                self._progress("stage1_denoise", i + 1, steps)
            entry["m_per_step"] = m_per_step
            entry["block_swap_prefetch_used"] = prefetch_used
            if not (torch.isfinite(video).all() and torch.isfinite(audio).all()):
                raise RuntimeError("denoise: 潜在に NaN／Inf が出ました（数値が発散しています）")
            ok = True
            return video, audio, prefetch_used
        except torch.OutOfMemoryError as exc:
            if vram16 is None:
                raise
            raise RuntimeError(VRAM16_OOM_MESSAGE) from exc
        finally:
            if vram16_entry is not None:
                self._vram16_finish(dev, cap_set, counters0, vram16_entry)
                entry["vram16"] = vram16_entry
            if service is not None:
                try:
                    service.teardown_prefetch()
                except Exception as exc:  # noqa: BLE001 -- teardown must never fail a job
                    self._log(f"block swap prefetch teardown failed (ignored): {exc!r}")
                service.release_installed()
            # LoRA off the DiT, unconditionally (a no-op without LoRA), so a kept
            # DiT never carries this job's LoRA into the next one. ORDER: after
            # teardown_prefetch, whose ``_release`` writes ``mod._buffers[name]``
            # back BY NAME -- detaching first would put the A / B buffers back
            # (Docs/MINIMAX_H3_DESIGN.md §6.6). Before ``_park_dit``, so they are
            # not moved for nothing.
            from engine_h3 import lora as _lora  # noqa: PLC0415

            _lora.detach(dit)
            if ok and keep:
                entry["dit_parked_tensors"] = self._park_dit(dit)
                self._resident = {**(self._resident or {}), "dit": dit, "cfg": cfg}
            elif self._resident is not None:
                self._resident.pop("dit", None)
                self._resident.pop("cfg", None)
            del dit
            service = None
            _free_cuda()
            phases.mark("denoise", "after_release")

    # -- stage 3: decode ---------------------------------------------------------

    def _decode(self, req, video_lat: torch.Tensor, audio_lat: torch.Tensor, entry: dict, keep=False) -> int:
        from engine_h3.audio_vae import H3AudioVAE  # noqa: PLC0415
        from engine_h3.media_out import audio_keep_samples, write_mp4  # noqa: PLC0415
        from engine_h3.video_vae import H3VideoDecoder  # noqa: PLC0415

        resident = self._resident or {}
        self._progress("decode", 0, 3)
        vae = resident.get("decoder")
        entry["vae_source"] = "resident" if vae is not None else "file"
        if vae is None:
            vae = H3VideoDecoder.from_file(self.paths.video_vae, device=self.device, dtype=torch.bfloat16)
        else:
            vae.to(self.device)
        try:
            # CPU uint8 [F, H, W, 3], converted window by window (video_vae, "Decode output").
            frames = vae.decode(video_lat, tiled=True)
        finally:
            if keep:
                vae.to("cpu")
                self._resident = {**(self._resident or {}), "decoder": vae}
            elif self._resident is not None:
                self._resident.pop("decoder", None)
            del vae
            _free_cuda()
        if int(frames.shape[0]) != int(req.num_frames):
            raise RuntimeError(f"decode: {frames.shape[0]} frames decoded, {req.num_frames} expected")
        self._progress("decode", 1, 3)

        avae = resident.get("audio_vae")
        if avae is None:
            avae = H3AudioVAE.from_file(self.paths.audio_vae, device=self.device)
        else:
            avae.to(self.device)
        try:
            wav = avae.decode(audio_lat).float().cpu()
        finally:
            if keep:
                avae.to("cpu")
                self._resident = {**(self._resident or {}), "audio_vae": avae}
            elif self._resident is not None:
                self._resident.pop("audio_vae", None)
            del avae
            _free_cuda()
        self._progress("decode", 2, 3)

        # write_mp4 creates the output folder (one place for that).
        size = write_mp4(str(req.output_path), frames, fps=int(req.frame_rate), audio=wav, sample_rate=AUDIO_SAMPLE_RATE)
        # What the mp4 holds: write_mp4 cuts the audio to the video's length,
        # round(F * 32000 / 24); the decoder gives up to half a latent more.
        entry["audio_samples"] = int(audio_keep_samples(
            int(wav.shape[-1]), num_frames=int(frames.shape[0]), fps=int(req.frame_rate), sample_rate=AUDIO_SAMPLE_RATE,
        ))
        entry["audio_samples_decoded"] = int(wav.shape[-1])
        entry["frames"] = int(frames.shape[0])
        self._progress("decode", 3, 3)
        return size

    def release_resident(self) -> None:
        """Let go of whatever ``keep_resident`` kept (all on the CPU; nothing to move).

        Public: the worker also calls it when a job fails before ``generate``
        is reached (a malformed message), so every failed job releases.
        """
        if self._resident is None:
            return
        self._resident = None
        gc.collect()

    @property
    def resident_parts(self) -> list[str]:
        """What is kept right now (for the log and the gate)."""
        return sorted(k for k in (self._resident or {}) if k != "cfg")

    def close(self) -> None:
        self.release_resident()
        svc, self._swap_service = self._swap_service, None
        if svc is not None:
            try:
                svc.teardown_prefetch()
            except Exception:  # noqa: BLE001 -- teardown is best effort
                pass
            svc.release_installed()
        _free_cuda()


def _backend_summary(backends: dict) -> dict:
    """``list_backends()`` reduced to what the log and ``done`` need (JSON-safe)."""
    out = {}
    for name, info in (backends or {}).items():
        if isinstance(info, dict):
            out[name] = {
                "available": bool(info.get("available")),
                "disabled": bool(info.get("disabled")),
                "int8_linear": "int8_linear" in (info.get("capabilities") or []),
            }
        else:
            out[name] = str(info)
    return out
