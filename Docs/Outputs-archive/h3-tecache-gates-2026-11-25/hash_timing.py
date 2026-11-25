import sys, time, json
sys.dont_write_bytecode = True
sys.path.insert(0, ".")
import torch
from engine_h3.pipeline_h3 import tensor_sha256
L = 6136  # largest text_tokens seen in the gates (G2 clip 2)
x = torch.randn(L, 5120, dtype=torch.float32).to(torch.bfloat16)
secs = []
for _ in range(5):
    t = time.perf_counter(); tensor_sha256(x); secs.append(round(time.perf_counter() - t, 4))
print(json.dumps({"tokens": L, "shape": list(x.shape), "dtype": str(x.dtype), "bytes": x.numel() * 2, "seconds_each": secs}))
