import os, time, traceback, torch
os.environ.setdefault("TORCH_COMPILE_DISABLE","1")
PROJ = r"S:/OriginalApps/12_Nz-LTX23-backend"
def log(m): print(f"[spike {time.strftime('%H:%M:%S')}] {m}", flush=True)
log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
import ltx_core.loader  # pre-warm: avoid ltx_core.quantization cold-import circular bug (rev 00dc53d)
log("pre-warmed ltx_core.loader")
try:
    from services.fast_video_pipeline.ltx_fast_video_pipeline import LTXFastVideoPipeline
    log("imported LTXFastVideoPipeline")
except Exception:
    traceback.print_exc(); log("IMPORT_FAIL"); raise SystemExit(2)
try:
    log("creating pipeline (loads GGUF transformer + VAE/gemma)...")
    pipe = LTXFastVideoPipeline.create(
        checkpoint_path=f"{PROJ}/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
        gemma_root=f"{PROJ}/models/gemma-3-12b-it-qat",
        upsampler_path=f"{PROJ}/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        device=torch.device("cuda:0"),
        block_swap_blocks_on_gpu=8,
        gguf_transformer_path=f"{PROJ}/models/ltx-2.3-gguf/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
        gguf_per_layer_quant=True,
        vae_spatial_tile_size=0,
        vae_temporal_tile_size=0,
    )
    log("PIPELINE_CREATED_OK  (<<< load crash avoided)")
except Exception:
    traceback.print_exc(); log("CREATE_FAIL"); raise SystemExit(3)
try:
    os.makedirs(f"{PROJ}/outputs/forkenv_spike", exist_ok=True)
    log("generating 384x256 / 9 frames / 8 steps ...")
    t0=time.time()
    pipe.generate(prompt="a cat walking on grass", seed=10, height=256, width=384,
                  num_frames=9, frame_rate=8, images=[],
                  output_path=f"{PROJ}/outputs/forkenv_spike/min.mp4", num_steps=8)
    log(f"GENERATED_OK elapsed={time.time()-t0:.1f}s")
except Exception:
    traceback.print_exc(); log("GENERATE_FAIL"); raise SystemExit(4)
log("SPIKE_DONE")
