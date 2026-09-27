"""Quantized (fp8 / int8) safetensors transformer path (§3-167 B-1, §3-168).

  * ``sft_reader``    — reads safetensors tensors one at a time with
                        seek + readinto (no mmap, no ``safe_open``).
  * ``dequant``       — ``dequantize`` / ``normalize_aux`` / ``hadamard``: the
                        per-scheme maths (fp8, fp8_scaled, int8, int8_convrot).
  * ``quant_service`` — the state-dict loader, the ``sft_quant_linear`` module op and
                        ``SftQuantLoaderService.install`` (the GGUF service's twin).

The acceptance check itself lives in the repository-root ``sft_quant_format``
(torch-free, shared with the app venv).
"""
