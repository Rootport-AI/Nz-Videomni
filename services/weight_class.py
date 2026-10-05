"""Weight class of a transformer file: ``"4bit"`` / ``"8bit"`` / ``"q6k"`` / ``None``.

The comfort-limit table (the base-model manifests' ``comfort`` block, served
as ``limits.comfort_budgets``) has one row per weight class, keyed by
``requires.weight_class``. The server tells the clients which class the
SELECTED transformer is via ``GET /models``'s
``base_models[].transformer_weight_class``; this module is where that class is
decided, from the file's header alone (the multi-GB body is never read).

The rules' source of truth is Docs/COMFORT_LIMIT_TABLE.md §1. In short:

* GGUF: the majority ggml type among the 2-D ``.weight`` tensors under
  ``transformer_blocks.<n>.`` -- Q4_K (12) -> ``"4bit"``, Q6_K (14) ->
  ``"q6k"``, anything else (Q8_0, F16, BF16, ...) -> ``None``. A Q4_K_M file
  also carries some Q6_K/Q5_K layers; the majority is still Q4_K.
* safetensors: :func:`sft_quant_format.inspect` gives each quantized layer's
  scheme. The 8-bit schemes (fp8, fp8_scaled, int8, int8_convrot) are summed
  and compared with ``w4a8``: the larger side wins, a tie goes to ``"8bit"``.
  No quantized layer at all (bf16) -> ``inspect`` refuses -> ``None``.
* A file that cannot be read or parsed -> ``None``. The operation panel and
  Gradio then leave the class key out of the match, so no row matches: single
  falls back to ``spill_free_frames``, chain to ``chain_comfort_token_budget``.

Pure functions plus one small cache; no heavy dependencies (the app venv stays
torch-free).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import sft_quant_format
from services.gguf_kv import GgufParseError, read_gguf_tensor_types

logger = logging.getLogger("ltx.weight_class")

#: Every value ``requires.weight_class`` may take (anything else in a manifest
#: is a startup error, see api/context.py ``build_comfort_budgets``).
WEIGHT_CLASSES = ("4bit", "8bit", "q6k")

#: ggml type id -> weight class. Types not listed here classify as ``None``.
_GGUF_CLASS_BY_TYPE = {12: "4bit", 14: "q6k"}  # Q4_K, Q6_K

#: safetensors schemes (``sft_quant_format.SCHEMES``) counted as 8-bit.
_SFT_8BIT_SCHEMES = frozenset({"fp8", "fp8_scaled", "int8", "int8_convrot"})
_SFT_4BIT_SCHEME = "w4a8"

_BLOCK_RE = re.compile(r"transformer_blocks\.\d+\.")


def classify_sft_layers(layers: Mapping[str, str]) -> str | None:
    """Layer name -> scheme (``sft_quant_format.inspect(path).layers``) -> class."""
    eight = sum(1 for scheme in layers.values() if scheme in _SFT_8BIT_SCHEMES)
    four = sum(1 for scheme in layers.values() if scheme == _SFT_4BIT_SCHEME)
    if eight == 0 and four == 0:
        return None
    return "4bit" if four > eight else "8bit"


def classify_gguf_tensors(tensors: Mapping[str, tuple[int, int]]) -> str | None:
    """Tensor name -> (n_dims, ggml type) (``read_gguf_tensor_types``) -> class."""
    counts = Counter(
        ggml_type
        for name, (n_dims, ggml_type) in tensors.items()
        if n_dims == 2 and name.endswith(".weight") and _BLOCK_RE.search(name)
    )
    if not counts:
        return None
    majority, _ = counts.most_common(1)[0]
    return _GGUF_CLASS_BY_TYPE.get(majority)


def classify_transformer(path: Path) -> str | None:
    """Weight class of the transformer file at ``path`` (``None`` = unknown).

    Cached on (path, size, mtime) so a repeated ``GET /models`` costs one
    ``stat``; replacing the file under the same name re-classifies it.
    """
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return _classify_cached(str(path), st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=16)
def _classify_cached(path: str, size: int, mtime_ns: int) -> str | None:
    # size / mtime_ns are cache-key parts only.
    p = Path(path)
    try:
        if p.suffix.lower() == ".gguf":
            return classify_gguf_tensors(read_gguf_tensor_types(p))
        if p.suffix.lower() == ".safetensors":
            return classify_sft_layers(sft_quant_format.inspect(p).layers)
    except (sft_quant_format.QuantFormatError, GgufParseError, OSError, ValueError) as exc:
        logger.info("transformer の重みの種別を判定できません（%s）: %s", p.name, exc)
        return None
    return None
