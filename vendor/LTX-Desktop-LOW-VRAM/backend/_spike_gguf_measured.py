import os, time, traceback, faulthandler, torch
faulthandler.enable()
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

PROJ = r"S:/OriginalApps/12_Nz-LTX23-backend"
BLOCKS = int(os.environ.get("SPIKE_BLOCKS_ON_GPU", "8"))
CPU_TEXT_ENCODE = os.environ.get("SPIKE_CPU_TEXT_ENCODE", "1") == "1"

def log(m):
    print(f"[spike bs{BLOCKS} {time.strftime('%H:%M:%S')}] {m}", flush=True)

def mb(x):
    return f"{x / (1024*1024):.1f} MB"

try:
    import psutil
    _PROC = psutil.Process()
    def host_ram_mb():
        return _PROC.memory_info().rss / (1024 * 1024)
    def host_ram_avail_mb():
        return psutil.virtual_memory().available / (1024 * 1024)
except Exception:
    def host_ram_mb():
        return 0.0
    def host_ram_avail_mb():
        return 0.0

log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA'}")
log(f"block_swap_blocks_on_gpu = {BLOCKS}  cpu_text_encode = {CPU_TEXT_ENCODE}")
log(f"host RSS={host_ram_mb():.0f}MB  host avail={host_ram_avail_mb():.0f}MB")

# pre-warm: avoid ltx_core.quantization cold-import circular bug (rev 00dc53d)
import ltx_core.loader
log("pre-warmed ltx_core.loader")

# Apply the portable safetensors-loader safe patch (fixes the mmap/non_blocking
# access violation on CPU-Gemma load and intermittent GPU loads). Must run after
# ltx_core.loader is warmed and before any pipeline build.
from services.sft_loader_safe_patch import apply_sft_loader_safe_patch
log(f"sft_loader safe patch applied = {apply_sft_loader_safe_patch()}")

try:
    from services.fast_video_pipeline.ltx_fast_video_pipeline import LTXFastVideoPipeline
    log("imported LTXFastVideoPipeline")
except Exception:
    traceback.print_exc(); log("IMPORT_FAIL"); raise SystemExit(2)

# ---- LOAD PEAK ----
load_peak = None
try:
    torch.cuda.reset_peak_memory_stats()
    log("creating pipeline (loads GGUF transformer + VAE/gemma)...")
    t_load = time.time()
    pipe = LTXFastVideoPipeline.create(
        checkpoint_path=f"{PROJ}/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
        gemma_root=f"{PROJ}/models/gemma-3-12b-it-qat",
        upsampler_path=f"{PROJ}/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        device=torch.device("cuda:0"),
        block_swap_blocks_on_gpu=BLOCKS,
        gguf_transformer_path=f"{PROJ}/models/ltx-2.3-gguf/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
        gguf_per_layer_quant=True,
        vae_spatial_tile_size=0,
        vae_temporal_tile_size=0,
        cpu_text_encode=CPU_TEXT_ENCODE,
    )
    load_peak = torch.cuda.max_memory_allocated()
    log(f"PIPELINE_CREATED_OK  load_elapsed={time.time()-t_load:.1f}s  LOAD_PEAK={mb(load_peak)}")
except Exception:
    load_peak = torch.cuda.max_memory_allocated()
    traceback.print_exc()
    log(f"CREATE_FAIL  LOAD_PEAK_AT_FAIL={mb(load_peak)}")
    raise SystemExit(3)

# ---- host RAM peak sampler (in-process, tracks our own RSS during gen) ----
import threading
_ram_peak = {"rss": 0.0, "min_avail": 1e12}
_ram_stop = threading.Event()
def _ram_watch():
    while not _ram_stop.is_set():
        r = host_ram_mb()
        a = host_ram_avail_mb()
        if r > _ram_peak["rss"]:
            _ram_peak["rss"] = r
        if a < _ram_peak["min_avail"]:
            _ram_peak["min_avail"] = a
        _ram_stop.wait(0.5)
_ram_thread = threading.Thread(target=_ram_watch, daemon=True)
_ram_thread.start()

# ---- GEN PEAK ----
gen_peak = None
try:
    outdir = f"{PROJ}/outputs/forkenv_spike"
    os.makedirs(outdir, exist_ok=True)
    outpath = f"{outdir}/min_bs{BLOCKS}.mp4"
    torch.cuda.reset_peak_memory_stats()
    log("generating 384x256 / 9 frames / 8 steps ...")
    t0 = time.time()
    pipe.generate(
        prompt="a cat walking on grass", seed=10,
        height=256, width=384, num_frames=9, frame_rate=8,
        images=[], output_path=outpath, num_steps=8,
    )
    gen_peak = torch.cuda.max_memory_allocated()
    gen_elapsed = time.time() - t0
    _ram_stop.set()
    log(f"GENERATED_OK elapsed={gen_elapsed:.1f}s  GEN_PEAK={mb(gen_peak)}  out={outpath}")
    log(f"HOST_RAM peak_rss={_ram_peak['rss']:.0f}MB  min_avail={_ram_peak['min_avail']:.0f}MB")
except Exception:
    gen_peak = torch.cuda.max_memory_allocated()
    _ram_stop.set()
    traceback.print_exc()
    log(f"GENERATE_FAIL  GEN_PEAK_AT_FAIL={mb(gen_peak)}  host_peak_rss={_ram_peak['rss']:.0f}MB  min_avail={_ram_peak['min_avail']:.0f}MB")
    raise SystemExit(4)

log(f"RESULT_SUMMARY blocks={BLOCKS} cpu_encode={CPU_TEXT_ENCODE} gen_peak={mb(gen_peak)} host_peak_rss={_ram_peak['rss']:.0f}MB result=GENERATED_OK")
log("SPIKE_DONE")
