"""Tokenizers for the prompt token counter (``POST /utils/prompt-tokens``).

Shared parts only. WHICH tokenizer a base model uses, where its file is and
how a prompt is counted are facts about the engine, so they live in each
family's adapter as ``count_prompt_tokens`` (dispatched by
``services.engines.count_prompt_tokens``); the adapters build their tokenizer
with the two functions here.

ONE TOKENIZER PER PROCESS. The counter only ever needs the active base
model's tokenizer, so the last one built is kept and a different key (path,
size, mtime, the extra specials / the tensor name) replaces it. Building runs
under a lock; a built tokenizer is never mutated afterwards, so several
threads may encode with it at once.

``tokenizers`` (the Hugging Face Rust tokenizer, no torch) is imported inside
the functions: a venv that has not been re-synced after the update still
starts, and the counter alone fails with a hint to re-run setup.bat.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from services.gguf_kv import read_gguf_tensor_bytes

_lock = threading.Lock()
_cached: tuple[tuple, Any] | None = None


def _tokenizers():
    try:
        import tokenizers
    except ImportError as exc:
        raise RuntimeError(
            "プロンプトのトークン数を数える tokenizers がアプリの仮想環境（.venv）に"
            "入っていません。setup.bat をもう一度実行してください。"
        ) from exc
    return tokenizers


def _file_key(path: Path) -> tuple:
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns)


def _cached_tokenizer(key: tuple, build: Callable[[], Any]) -> Any:
    global _cached
    with _lock:
        if _cached is None or _cached[0] != key:
            _cached = (key, build())
        return _cached[1]


def tokenizer_file(path: Path, extra_special: tuple[str, ...] = ()) -> Any:
    """A ``tokenizers.Tokenizer`` read from a ``tokenizer.json`` file.

    ``extra_special`` are added as special tokens (whole, not normalized) in
    the given order, so their ids follow the vocabulary in that order.
    """
    path = Path(path)

    def build():
        tk = _tokenizers()
        tokenizer = tk.Tokenizer.from_file(str(path))
        if extra_special:
            tokenizer.add_special_tokens(
                [tk.AddedToken(t, special=True, normalized=False) for t in extra_special]
            )
        return tokenizer

    return _cached_tokenizer(("file", *_file_key(path), tuple(extra_special)), build)


def tokenizer_gguf(path: Path, tensor: str) -> Any:
    """A ``tokenizers.Tokenizer`` from the JSON a GGUF carries as tensor ``tensor``."""
    path = Path(path)

    def build():
        tk = _tokenizers()
        return tk.Tokenizer.from_str(read_gguf_tensor_bytes(path, tensor).decode("utf-8"))

    return _cached_tokenizer(("gguf", *_file_key(path), tensor), build)
