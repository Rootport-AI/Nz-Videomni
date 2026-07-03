"""Persistent LTX-2.3 generation worker (Phase 5, Approach W).

Runs inside the engine venv (.venv-engine), which is the only environment that
has torch + ltx_core/ltx_pipelines@00dc53d + gguf. The
app process (FastAPI, its own torch-less .venv) spawns ONE of these per loaded
model and talks to it over a tiny JSON-lines protocol on stdin/stdout. The mp4 is
written by the engine directly to a shared-disk path; only small control JSON
crosses the pipe.

Bootstrap mirrors outputs/phase4_gguf_gemma/run_t2v_bs8.py VERBATIM (proven to
run on this Windows + 16GB box):
  * TORCH_COMPILE_DISABLE=1 set before importing torch.
  * chdir(ROOT) + sys.path.insert(0, ROOT) so the first-party `engine.*` package
    (and the venv-installed `ltx_core` / `ltx_pipelines`) import.
  * `import ltx_core.loader` BEFORE the pipeline import (rev-00dc53d circular
    import gotcha — required).

Protocol (one JSON object per line; parent -> worker):
  {"op": "load", checkpoint_path, gemma_root, upsampler_path,
   gguf_transformer_path, gguf_gemma_path, gguf_per_layer_quant,
   block_swap_blocks_on_gpu, vae_spatial_tile_size, vae_temporal_tile_size}
  {"op": "generate", prompt, seed, height, width, num_frames, frame_rate,
   num_steps, images:[{path,frame_idx,strength}...], output_path}
  # Phase 3 WP4 — masked AV-latent clip chaining (ONE decode, always-tiled stage2):
  {"op": "generate_chain", width, height, frame_rate, num_steps, seed,
   overlap_frames, overlap_strength, output_path,
   clips:[{prompt, num_frames, images:[{path,frame_idx,strength}...]}...]}
  {"op": "shutdown"}

Replies are framed with a unique prefix so library/tqdm stdout noise can be
ignored by the parent. Every protocol line: @@LTX@@<compact-json>, flushed. All
other logging goes to STDERR.
  @@LTX@@{"event":"ready"}
  @@LTX@@{"event":"done","seed_used":...,"peak_vram_mb":...}
  @@LTX@@{"event":"error","detail":...}
"""

import os
import sys
import json
import gc
import traceback
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# import cwd = project root: make the first-party `engine.*` package (and the
# venv-installed `ltx_core`/`ltx_pipelines`) importable. Launched as
# `python -m engine.worker` with PYTHONPATH=<root>, but we also insert the root
# on sys.path + chdir(ROOT) here so a bare `python engine/worker.py` still works
# (mirrors the pre-relocation chdir/sys.path bootstrap).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Unique frame prefix for every protocol line the parent parses.
PREFIX = "@@LTX@@"


def _log(msg: str) -> None:
    """Human/diagnostic logging -> STDERR only (never the protocol channel)."""
    print(f"[ltx_worker] {msg}", file=sys.stderr, flush=True)


def _emit(event: str, **fields: object) -> None:
    """Write one framed protocol line to STDOUT and flush."""
    payload = {"event": event, **fields}
    print(PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def _detail(exc: BaseException) -> str:
    """repr + last ~2000 chars of traceback for an error reply."""
    tb = "".join(traceback.format_exc())
    return f"{exc!r}\n{tb[-2000:]}"


import torch  # noqa: E402

# Pre-warm to avoid the rev-00dc53d cold-import circular bug (MUST precede any
# quantization / pipeline import). Mirrors run_t2v_bs8.py L52-54.
import ltx_core.loader  # noqa: E402,F401

_log(f"pre-warmed ltx_core.loader; torch {torch.__version__} cuda={torch.cuda.is_available()}")

DEV = torch.device("cuda:0")
# init cuda context + warm + reset stats (mirrors run_t2v_bs8.py L57-61).
torch.cuda.set_device(DEV)
torch.cuda.init()
_ = torch.zeros(1, device=DEV)
torch.cuda.synchronize(DEV)
torch.cuda.reset_peak_memory_stats(DEV)

# Engine imports AFTER ltx_core.loader pre-warm + CUDA init.
from engine.pipeline.fast_video_pipeline import (  # noqa: E402
    LTXFastVideoPipeline,
)
from engine.api_types import ImageConditioningInput  # noqa: E402

_log("imported LTXFastVideoPipeline + ImageConditioningInput")

# Phase 5(B) VRAM fix: release the Windows-stranded caching-allocator reserved
# pool before each denoise. expandable_segments is a no-op on this box, so after
# block-swap loads the full GGUF transformer to GPU then evicts blocks to CPU,
# ~16GB of freed segments stay reserved and spill to WDDM shared during denoise.
# A gc+empty_cache right before denoise drops reserved ~17.5GB -> ~1.5GB (live
# unchanged), keeping denoise inside the 16GB card. Proven in outputs/phase5b_diag
# (tag EC): denoise shared 3969 -> 742MB (ambient), wall 210.9 -> 114.9s, output
# bit-identical.
import ltx_pipelines.distilled as _distilled  # noqa: E402

_orig_denoise_audio_video = _distilled.denoise_audio_video


def _denoise_with_cache_release(*args, **kwargs):
    gc.collect()
    torch.cuda.empty_cache()
    return _orig_denoise_audio_video(*args, **kwargs)


_distilled.denoise_audio_video = _denoise_with_cache_release
_log("installed pre-denoise empty_cache monkeypatch (Phase 5B VRAM fix)")

# Module-global pipeline, built once on the first {"op":"load"}.
_PIPE: LTXFastVideoPipeline | None = None


def _do_load(msg: dict) -> None:
    """Build the pipeline ONCE. Mirrors run_t2v_bs8.py create() arg names/values."""
    global _PIPE
    if _PIPE is not None:
        _emit("ready")
        return

    _log("creating pipeline (GGUF transformer + GGUF Gemma)...")
    _PIPE = LTXFastVideoPipeline.create(
        checkpoint_path=msg["checkpoint_path"],
        gemma_root=msg["gemma_root"],
        upsampler_path=msg["upsampler_path"],
        device=DEV,
        block_swap_blocks_on_gpu=int(msg["block_swap_blocks_on_gpu"]),
        gguf_transformer_path=msg["gguf_transformer_path"],
        gguf_gemma_path=msg["gguf_gemma_path"],
        gguf_per_layer_quant=bool(msg["gguf_per_layer_quant"]),
        vae_spatial_tile_size=int(msg["vae_spatial_tile_size"]),
        vae_temporal_tile_size=int(msg["vae_temporal_tile_size"]),
        # Stage 3: load-once / keep-resident weights (StateDictRegistry). Default ON in
        # the resident worker; the reuse_loop spike toggles LTX_KEEP_RESIDENT=0 for an
        # A/B baseline against the per-job-rebuild known-good.
        keep_resident_weights=(os.environ.get("LTX_KEEP_RESIDENT", "1") == "1"),
        # TE per-layer CPU offload: stream the GGUF-quantized Gemma decoder layers
        # CPU->GPU one window at a time during text-encode (caps the ~15 GB encode
        # peak to a few GB). Default ON when the env var is ABSENT; LTX_TE_OFFLOAD=0
        # reproduces today's all-layers-GPU-resident behavior.
        te_offload_text_encoder=(os.environ.get("LTX_TE_OFFLOAD", "1") == "1"),
        # DiT CPU-resident build: build the transformer on CPU and move only the
        # non-block submodules to GPU, eliminating the ~16.9 GB load-time GPU
        # spike. Default ON when the env var is ABSENT; LTX_DIT_CPU_LOAD=0
        # reproduces today's build-on-GPU-then-evict behavior.
        dit_cpu_load=(os.environ.get("LTX_DIT_CPU_LOAD", "1") == "1"),
        # Phase 1: re-source VIDEO VAE + AUDIO VAE/vocoder from standalone files
        # (gate via LTX_COMPONENT_FILES, default OFF). Text projection path is
        # passed through but NOT wired (Phase 2).
        use_component_files=(os.environ.get("LTX_COMPONENT_FILES", "0") == "1"),
        component_video_vae_path=msg.get("component_video_vae_path", ""),
        component_audio_vae_path=msg.get("component_audio_vae_path", ""),
        component_text_projection_path=msg.get("component_text_projection_path", ""),
        # cpu_text_encode intentionally NOT set -> GGUF Gemma path wins.
    )
    _log("PIPELINE_CREATED_OK")
    _emit("ready")


def _do_generate(msg: dict) -> None:
    """Run one generation; mp4 is written by the engine to msg['output_path']."""
    assert _PIPE is not None, "generate before load"
    output_path = msg["output_path"]
    seed = int(msg["seed"])

    torch.cuda.reset_peak_memory_stats(DEV)

    images = [
        ImageConditioningInput(
            path=i["path"],
            frame_idx=int(i["frame_idx"]),
            strength=float(i["strength"]),
        )
        for i in msg.get("images", [])
    ]

    _log(
        f"generating {msg['width']}x{msg['height']} / {msg['num_frames']} frames "
        f"/ {msg['num_steps']} steps seed={seed} images={len(images)}"
    )
    _PIPE.generate(
        prompt=msg["prompt"],
        seed=seed,
        height=int(msg["height"]),
        width=int(msg["width"]),
        num_frames=int(msg["num_frames"]),
        frame_rate=msg["frame_rate"],
        images=images,
        output_path=output_path,
        num_steps=int(msg["num_steps"]),
    )

    peak = torch.cuda.max_memory_allocated(DEV) // (1024 * 1024)

    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"engine produced no/empty output: {output_path}")

    _log(f"GENERATED_OK peak_vram_mb={peak} -> {output_path}")
    _emit("done", seed_used=seed, peak_vram_mb=int(peak))

    # Resident-reuse: free the just-finished job's transient allocations before
    # the next job. The block-swap transformer carries reference cycles
    # (swapped_forward closures capturing block lists) that plain refcounting
    # can't reclaim, so an explicit gc.collect() is required; empty_cache() then
    # returns the freed CUDA blocks to the driver. Pairs with the keep-latest
    # fix in BlockSwapService.install().
    gc.collect()
    torch.cuda.empty_cache()


def _do_generate_chain(msg: dict) -> None:
    """Run one masked AV-latent clip chain; ONE mp4 written to output_path.

    Emits ``progress`` events (stage1 per segment, tile per stage-2 tile, decode)
    and a terminal ``done`` with the peak VRAM + full junction metadata (segment
    seams AND tile seams) for the review harness.
    """
    assert _PIPE is not None, "generate_chain before load"
    from engine.pipeline.chain_pipeline import ChainClipSpec

    output_path = msg["output_path"]
    seed = int(msg["seed"])
    torch.cuda.reset_peak_memory_stats(DEV)

    clips = [
        ChainClipSpec(
            prompt=c["prompt"],
            num_frames=int(c["num_frames"]),
            images=[
                ImageConditioningInput(
                    path=i["path"], frame_idx=int(i["frame_idx"]), strength=float(i["strength"])
                )
                for i in c.get("images", [])
            ],
        )
        for c in msg["clips"]
    ]

    _log(
        f"generate_chain {msg['width']}x{msg['height']} clips={len(clips)} "
        f"frames={[c.num_frames for c in clips]} seed={seed} "
        f"overlap={msg.get('overlap_frames')}/{msg.get('overlap_strength')}"
    )

    def _progress(stage: str, index: int, total: int) -> None:
        _emit("progress", stage=stage, index=int(index), total=int(total))

    meta = _PIPE.generate_chain(
        clips=clips,
        width=int(msg["width"]),
        height=int(msg["height"]),
        frame_rate=msg["frame_rate"],
        num_steps=int(msg["num_steps"]),
        seed=seed,
        overlap_frames=int(msg["overlap_frames"]),
        overlap_strength=float(msg["overlap_strength"]),
        output_path=output_path,
        progress=_progress,
    )

    peak = torch.cuda.max_memory_allocated(DEV) // (1024 * 1024)
    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"engine produced no/empty chain output: {output_path}")

    _log(f"CHAIN_OK peak_vram_mb={peak} -> {output_path}")
    _emit("done", seed_used=seed, peak_vram_mb=int(peak), chain=meta)

    gc.collect()
    torch.cuda.empty_cache()


def _shutdown() -> None:
    """Best-effort free + sync, then exit 0."""
    global _PIPE
    try:
        _PIPE = None
        if torch.cuda.is_available():
            torch.cuda.synchronize(DEV)
            torch.cuda.empty_cache()
    except Exception:
        pass
    sys.exit(0)


def main() -> None:
    # First, block for the load op (a fatal error here -> exit 1 so the parent
    # sees the worker die before 'ready').
    loaded = False
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            _log(f"ignoring non-JSON stdin line: {line[:120]!r}")
            continue

        op = msg.get("op")

        if not loaded:
            if op != "load":
                # Tolerate shutdown/EOF before load; ignore anything else.
                if op == "shutdown":
                    _shutdown()
                _log(f"ignoring op={op!r} before load")
                continue
            try:
                _do_load(msg)
                loaded = True
            except BaseException as exc:  # noqa: BLE001
                _log("LOAD_FAILED")
                _emit("error", detail=_detail(exc))
                sys.exit(1)
            continue

        # Serving loop (post-load).
        if op == "generate":
            try:
                _do_generate(msg)
            except BaseException as exc:  # noqa: BLE001 - keep serving on error
                _log("GENERATE_FAILED")
                _emit("error", detail=_detail(exc))
            continue
        if op == "generate_chain":
            try:
                _do_generate_chain(msg)
            except BaseException as exc:  # noqa: BLE001 - keep serving on error
                _log("GENERATE_CHAIN_FAILED")
                _emit("error", detail=_detail(exc))
            continue
        if op == "shutdown":
            _shutdown()
        _log(f"ignoring unknown op={op!r}")

    # EOF on stdin -> graceful shutdown.
    _shutdown()


if __name__ == "__main__":
    main()
