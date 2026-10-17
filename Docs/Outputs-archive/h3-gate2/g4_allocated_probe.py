"""2a-G4 side probe: what the 8.25 MiB still allocated between jobs is (GPU, seconds)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from engine_h3.quant_linear import require_cuda_int8
require_cuda_int8()
r = {}
r["start"] = torch.cuda.memory_allocated()
a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
b = (a @ a).sum().item()
del a
torch.cuda.empty_cache()
r["after_bf16_matmul_and_empty_cache"] = torch.cuda.memory_allocated()
torch._C._cuda_clearCublasWorkspaces()
torch.cuda.empty_cache()
r["after_clearCublasWorkspaces"] = torch.cuda.memory_allocated()
r["g4_between_jobs"] = 8650752
Path(__file__).with_name("g4_allocated_probe.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r))
