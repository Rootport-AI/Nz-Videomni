"""fp8 safetensors transformer path for LTX 2.3 (§3-167 B-1).

  * ``sft_reader``    — reads safetensors tensors one at a time with
                        seek + readinto (no mmap, no ``safe_open``).
  * ``quant_service`` — the state-dict loader, the ``sft_quant_linear`` module op and
                        ``SftQuantLoaderService.install`` (the GGUF service's twin).

The acceptance check itself lives in the repository-root ``sft_quant_format``
(torch-free, shared with the app venv).
"""
