r"""2a-G2 side measurement: where the pipeline's encode phase spends its seconds
(imports / build / direct-read forward / release), in a fresh process, the way
H3Pipeline._encode runs it. GPU; ~20 s.

    .venv-engine-h3\Scripts\python.exe outputs\h3-gate2\g2_encode_phase_timing.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
marks = [("start", time.perf_counter())]
import torch  # noqa: E402

from engine_h3.quant_linear import require_cuda_int8  # noqa: E402

require_cuda_int8()
marks.append(("torch+comfy_kitchen imported", time.perf_counter()))
from engine_h3.text_encoder import H3TextEncoder  # noqa: E402

marks.append(("text_encoder module imported", time.perf_counter()))
import transformers  # noqa: E402,F401
from transformers.models.qwen3_vl import modeling_qwen3_vl  # noqa: E402,F401

marks.append(("transformers qwen3_vl imported", time.perf_counter()))
te = H3TextEncoder(str(ROOT / "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
                   str(ROOT / "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer"), device="cuda")
te.load()
marks.append(("te.load (tokenizer + meta build)", time.perf_counter()))
with torch.no_grad():
    hidden, tags = te.encode("A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in "
                             "puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>")
marks.append(("te.encode (direct read + forward)", time.perf_counter()))
torch.cuda.synchronize()
marks.append(("synchronize", time.perf_counter()))
te.release()
marks.append(("te.release", time.perf_counter()))
out = {"steps": [[name, round(t - marks[i][1], 2)] for i, (name, t) in enumerate(marks[1:])],
       "read_report": te.read_report}
text = json.dumps(out, indent=2)
(Path(__file__).with_name("g2_encode_phase_timing.json")).write_text(text, encoding="utf-8")
print(text)
