"""MiniMax H3 T2V pipeline: encode -> denoise -> decode, one stage at a time.

Ported in part from DiffSynth-Studio (Apache-2.0), commit 4e34969:
  diffsynth/pipelines/minimax_h3_audio_video.py
    (the noise initializer 245-250, the two-schedule loop 184-198, the per-row
     timesteps and their ``unique`` index 1011-1021)
Copyright (c) the DiffSynth-Studio authors. Licensed under the Apache License,
Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Changes: T2V only;
no CFG (H3 is guidance-distilled: one forward per step); no padding rows; the
noise is drawn in float32 from one CPU generator (video, then audio); the
model's output is used in its own direction (x0 = x + sigma v, no negation);
every stage builds its model on meta, streams it in, and frees it before the
next stage starts, so the peaks do not stack (TE ~27 GB, then DiT ~21 GB).

Stages (``progress`` names are the app's shared vocabulary):

1. ``encode``: the Qwen3-VL text tower (:class:`H3TextEncoder`), its decoder
   layers streamed CPU->GPU; released (GPU and CPU) before the DiT is built.
2. ``stage1_denoise``: the DiT. Everything but ``blocks`` goes to the GPU; the
   50 blocks stay on the CPU and ``BlockSwapService`` pages them in (with the
   prefetch engine when asked). That is also what puts each int8 weight AND
   its scale on the GPU before the block's forward -- ``comfy_kitchen``'s CUDA
   backend refuses a CPU weight, and with triton / eager disabled the call
   would raise rather than silently run slowly.
   Each step: video rows and audio rows get their own t = 1 - sigma (video
   shift 12, audio shift 3, same step index); text rows share the video t; the
   distinct values are numbered with ``torch.unique`` (M = 1 at step 0, where
   both sigmas are 1, and 2 afterwards) and each row's AdaLN row is
   ``number * 3 + modality``.
3. ``decode``: the video VAE, then the audio VAE, then the mp4.

Inference is wrapped in ``torch.no_grad()`` only -- never the inference-mode context
(the block swap and the prefetch engine rebind ``.data`` on tensors, which an
inference tensor refuses; design rule §5.8).
"""

from __future__ import annotations

import gc
import logging
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
    MODALITY_AUDIO,
    MODALITY_VIDEO,
    SPATIAL_FACTOR,
    VAE_SPATIAL,
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
    * ``te_layers_on_gpu``: the text encoder's decoder-layer window.
      ``0`` means ONE layer at a time (window = max(1, n)) -- unlike
      ``blocks_on_gpu``, 0 is the most streamed setting here, not "all on GPU".
    * ``deterministic``: sets cuDNN's deterministic / no-benchmark flags. Only
      the VAEs run cuDNN convolutions, so it affects the VAE decode only; the
      DiT and the text encoder have no convolution for it to act on.
    """

    blocks_on_gpu: int = 8
    te_layers_on_gpu: int = 0
    attention_backend: str = "sdpa"
    block_swap_prefetch: bool = True
    deterministic: bool = False


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


def time_rows(layout, t_video: float, t_audio: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row t -> ``(t_unique [M] f32, time_index [S] long)``.

    Text, cond and video rows carry the video t, audio rows the audio t (the
    T2V subset of DiffSynth pipelines/minimax_h3_audio_video.py:1006-1021;
    condition rows would carry max(t_v, 0.999) from the second stage on).
    ``unique`` sorts, so the numbering is by value, as in the reference.
    """
    t = torch.full((layout.total,), float(t_video), dtype=torch.float32, device=device)
    t[layout.audio] = float(t_audio)
    t_unique, time_index = torch.unique(t, sorted=True, return_inverse=True)
    return t_unique, time_index.to(torch.long)


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


def frames_to_uint8(pixels: torch.Tensor) -> torch.Tensor:
    """[1, 3, F, H, W] float in [0, 1] -> [F, H, W, 3] uint8 on the CPU."""
    x = pixels[0].permute(1, 2, 3, 0).float().clamp(0.0, 1.0)
    return (x * 255.0).round().to(torch.uint8).cpu()


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
    can stay high while the allocator keeps freed regions (REVIEW_code1 M1).
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
        #: its pinned staging pool (2 x 369 MB) and transfer stream survive the
        #: per-job teardown. Rebuilding the pool every job is what the pool's
        #: own docstring warns against (block_swap_prefetch.py:234-243).
        self._swap_service = None
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
                "te_layers_on_gpu": self.options.te_layers_on_gpu,
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
        validate_request(req)
        started = time.perf_counter()
        phases = _Phases(cuda=self.device.type == "cuda")
        try:
            with torch.no_grad():
                with phases.phase("encode") as entry:
                    text_hidden, text_tags = self._encode(req.prompt, phases)
                    entry["text_tokens"] = int(text_hidden.shape[0])
                with phases.phase("denoise") as entry:
                    video_lat, audio_lat, prefetch_used = self._denoise(req, text_hidden, text_tags, phases, entry)
                del text_hidden, text_tags
                _free_cuda()
                with phases.phase("decode") as entry:
                    size_bytes = self._decode(req, video_lat, audio_lat, entry)
                del video_lat, audio_lat
        finally:
            # Also after a failed job: the VRAM it reserved would otherwise stay
            # cached until the next job's encode, competing with AviUtl2's preview.
            _free_cuda()
        mem = _mem_gib()
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
            "attention_used": self.options.attention_backend,
            "block_swap_prefetch_used": prefetch_used,
            "h3": {"phases": phases.data, "quant_backends": _backend_summary(self.quant_backends)},
        }

    # -- stage 1 -----------------------------------------------------------------

    def _encode(self, prompt: str, phases: _Phases) -> tuple[torch.Tensor, torch.Tensor]:
        from engine_h3.text_encoder import H3TextEncoder  # noqa: PLC0415

        te = H3TextEncoder(
            self.paths.text_encoder, self.paths.tokenizer_dir,
            layers_on_gpu=self.options.te_layers_on_gpu, device=self.device,
        )
        self._progress("encode", 0, 3)
        try:
            info = te.load()
            phases.data["encode"]["te"] = info
            phases.mark("encode", "after_load")
            self._progress("encode", 1, 3)
            hidden, tags = te.encode(prompt)
            self._progress("encode", 2, 3)
        finally:
            te.release()
            del te
            _free_cuda()
        phases.mark("encode", "after_release")
        self._progress("encode", 3, 3)
        self._log(f"encode: {hidden.shape[0]} text tokens")
        return hidden, tags

    # -- stage 2 -----------------------------------------------------------------

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

    def _place_dit(self, dit) -> tuple[object | None, str]:
        """Non-block tensors to the GPU; blocks stay on the CPU behind the swap service.

        Mirrors ``engine25/gguf_transformer.py:_place_transformer``: the blocks
        are skipped by module id, the service gets ``prefetch_requested`` BEFORE
        ``install`` (which reads it), and ``hold_arenas=True`` because this
        engine's VRAM peak is inside denoise, as LTX 2.5's is.
        """
        from engine.transformer.block_swap_service import BlockSwapService  # noqa: PLC0415
        from engine_h3.sft_load import move_module_tree  # noqa: PLC0415

        blocks = list(dit.blocks)
        n = int(self.options.blocks_on_gpu)
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
        service.prefetch_requested = bool(self.options.block_swap_prefetch)
        service.install(dit)
        return service, str(service.last_prefetch_used or "off")

    def _denoise(self, req, text_hidden, text_tags, phases: _Phases, entry: dict):
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

        dit, cfg = self._build_dit(entry)
        phases.mark("denoise", "after_load")
        service = None
        prefetch_used = "off"
        try:
            service, prefetch_used = self._place_dit(dit)
            phases.mark("denoise", "after_place")

            text_refined = dit.refine_text(text_hidden.to(dev))
            L = int(text_refined.shape[0])
            layout = H3SeqLayout.make(
                text_rows=L, cond_rows=0, audio_rows=2 * a_lat, video_rows=t_lat * gh * gw,
            )
            coords = build_coords(
                text_len=L, latent_frames=t_lat, grid_h=gh, grid_w=gw, audio_latents=a_lat, cond_frames=0,
            )
            rope_cos, rope_sin = rope_tables(coords, dit.rope_inv_freq)
            rope_cos, rope_sin = rope_cos.to(dev), rope_sin.to(dev)
            modality = modality_rows(layout, text_tags.to(dev))

            # Noise: ONE CPU generator, float32, video first then audio (the
            # draw order is part of what "same seed" means for this engine).
            gen = torch.Generator(device="cpu").manual_seed(int(req.seed))
            video = torch.randn((1, cfg.latents_dim, t_lat, lh, lw), generator=gen, dtype=torch.float32).to(dev)
            audio = torch.randn((2, cfg.audio_latents_dim, a_lat), generator=gen, dtype=torch.float32).to(dev)

            sig_v = sigma_schedule(steps, float(req.shift_video))
            sig_a = sigma_schedule(steps, float(req.shift_audio))
            entry["sequence"] = {
                "text": L, "audio": 2 * a_lat, "video": t_lat * gh * gw, "total": layout.total,
                "latent_frames": t_lat, "audio_latents": a_lat, "grid": [gh, gw],
            }
            m_per_step: list[int] = []
            self._progress("stage1_denoise", 0, steps)
            for i in range(steps):
                t_v = 1.0 - float(sig_v[i])
                t_a = 1.0 - float(sig_a[i])
                t_unique, time_index = time_rows(layout, t_v, t_a, device=dev)
                m_per_step.append(int(t_unique.numel()))
                tokens = dit.embed_rows(text_refined, audio_to_rows(audio), patchify_video(video))
                v_video_rows, v_audio_rows = dit(
                    tokens,
                    layout=layout,
                    t_unique=t_unique,
                    time_index=time_index,
                    modality=modality,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    attention_backend=self.options.attention_backend,
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
            return video, audio, prefetch_used
        finally:
            if service is not None:
                try:
                    service.teardown_prefetch()
                except Exception as exc:  # noqa: BLE001 -- teardown must never fail a job
                    self._log(f"block swap prefetch teardown failed (ignored): {exc!r}")
                service.release_installed()
            del dit
            service = None
            _free_cuda()
            phases.mark("denoise", "after_release")

    # -- stage 3 -----------------------------------------------------------------

    def _decode(self, req, video_lat: torch.Tensor, audio_lat: torch.Tensor, entry: dict) -> int:
        from engine_h3.audio_vae import H3AudioVAE  # noqa: PLC0415
        from engine_h3.media_out import audio_keep_samples, write_mp4  # noqa: PLC0415
        from engine_h3.video_vae import H3VideoVAE  # noqa: PLC0415

        self._progress("decode", 0, 3)
        vae = H3VideoVAE.from_file(self.paths.video_vae, device=self.device, dtype=torch.bfloat16)
        try:
            pixels = vae.decode(video_lat, tiled=True)
        finally:
            del vae
            _free_cuda()
        if int(pixels.shape[2]) != int(req.num_frames):
            raise RuntimeError(f"decode: {pixels.shape[2]} frames decoded, {req.num_frames} expected")
        frames = frames_to_uint8(pixels)
        del pixels
        self._progress("decode", 1, 3)

        avae = H3AudioVAE.from_file(self.paths.audio_vae, device=self.device)
        try:
            wav = avae.decode(audio_lat).float().cpu()
        finally:
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

    def close(self) -> None:
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
