"""Numerical verification: fork _dequant_* kernels vs gguf reference dequantize.
Scoped to the quant types actually present in the Q4_K_M GGUF: Q4_K, Q5_K, Q6_K.
Runs the fork kernels on CUDA (GPU) to mirror the real forward path.
"""
import numpy as np
import torch
import gguf
import gguf.quants as gq
from gguf.constants import GGMLQuantizationType as T

import services.gguf_quant_service as svc

PATH = r"S:/OriginalApps/12_Nz-LTX23-backend/models/ltx-2.3-gguf/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
DEV = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

TYPE_TO_FN = {
    T.Q4_K: (svc._GGML_Q4_K, "Q4_K"),
    T.Q5_K: (svc._GGML_Q5_K, "Q5_K"),
    T.Q6_K: (svc._GGML_Q6_K, "Q6_K"),
}

reader = gguf.GGUFReader(PATH, mode="r")

PER_TYPE = 6
WANT = {T.Q4_K: [], T.Q5_K: [], T.Q6_K: []}
seen_shapes = {T.Q4_K: set(), T.Q5_K: set(), T.Q6_K: set()}
# pass 1: prefer shape diversity (matrices + vectors, different sizes)
for t in reader.tensors:
    tt = t.tensor_type
    if tt in WANT and len(WANT[tt]) < PER_TYPE:
        fs = tuple(reversed(t.shape.tolist()))
        if fs not in seen_shapes[tt]:
            WANT[tt].append(t)
            seen_shapes[tt].add(fs)
# pass 2: top up to PER_TYPE with any remaining tensors
for t in reader.tensors:
    tt = t.tensor_type
    if tt in WANT and len(WANT[tt]) < PER_TYPE and t not in WANT[tt]:
        WANT[tt].append(t)
    if all(len(v) >= PER_TYPE for v in WANT.values()):
        break

print(f"=== dequant numerical verification (fork on {DEV} vs gguf reference) ===")
overall_ok = True
for tt, tensors in WANT.items():
    ggml_const, label = TYPE_TO_FN[tt]
    for t in tensors:
        float_shape = tuple(reversed(t.shape.tolist()))
        raw_np = np.array(t.data, copy=False)
        raw_flat = torch.from_numpy(raw_np.copy()).reshape(-1).to(DEV)  # uint8 on GPU

        fork = svc.dequantize_ggml_tensor(raw_flat, ggml_const, float_shape, torch.float32)
        assert fork.device.type == DEV.type, f"kernel ran on {fork.device}, expected {DEV}"
        fork_np = fork.detach().cpu().numpy().astype(np.float32)

        ref = gq.dequantize(raw_np, tt).astype(np.float32).reshape(float_shape)

        if fork_np.shape != ref.shape:
            print(f"[{label}] {t.name}: SHAPE MISMATCH fork={fork_np.shape} ref={ref.shape}")
            overall_ok = False
            continue

        abs_err = np.abs(fork_np - ref)
        max_abs = float(abs_err.max())
        ref_scale = float(np.abs(ref).max()) + 1e-8
        rel = max_abs / ref_scale
        ok = rel < 0.05
        overall_ok = overall_ok and ok
        print(f"[{label}] {t.name[:46]:46s} shape={str(float_shape):14s} "
              f"max_abs={max_abs:.3e} rel={rel:.3e} {'PASS' if ok else 'FAIL'}")

print("=== RESULT:", "ALL_PASS" if overall_ok else "SOME_FAIL", "===")
