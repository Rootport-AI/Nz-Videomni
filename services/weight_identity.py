"""Weight identity — which published weight file a transformer actually is,
told by its whole-file SHA-256 (§1-107: MiniMax H3 FL2VA vs Ref2VA).

A base-model descriptor may list ``known_weights`` (``services/base_models.py``
``KnownWeight``: path, size, sha256, role). This module answers "which of those
is the file at this path?" by HASH ONLY — never by file name, so a renamed
official file is still recognised, and a file that matches none (a merged
"Hybrid", a third-party build) has no role at all rather than a wrong one.

Two entry points, deliberately unequal in cost:

``identify(path, known)``
    Called once per SUCCESSFUL pipeline load (``PipelineManager``). Hashes the
    file when the cache has no current entry for it — 21 GB takes tens of
    seconds on an SSD, minutes on an HDD — and stores the result. A file whose
    size matches no known weight is never hashed (it cannot be one of them).
``cached_role(path, known)``
    Called by ``GET /models`` on every request. A ``stat`` plus a dict lookup,
    nothing else: it never hashes and never re-reads the cache file.

THE CACHE: ``weight_identity.json`` next to ``state.json`` (project root,
gitignored), ``{"schema": 1, "entries": {<absolute path>: {"size",
"mtime_ns", "sha256"}}}``. An entry counts only while the file's size and
mtime_ns still match, so a replaced file is re-hashed. It is read ONCE when
the owning object is built and kept in memory under one lock (the request
thread of ``GET /models`` and the load thread touch it concurrently). Like
``state.json`` (``services/runtime_state.py``) it is a disposable cache:
broken -> renamed to ``.bad`` with a WARNING, unwritable -> WARNING; reading
and saving it never raise.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from api.errors import APIError
from services.base_models import KnownWeight
from services.model_registry import DEFAULT_NAME

if TYPE_CHECKING:
    from config import AppConfig
    from services.model_registry import ModelRegistry

logger = logging.getLogger("ltx.weight_identity")

#: File name of the cache, placed in the directory of ``config.state_path``.
CACHE_FILENAME = "weight_identity.json"
#: A cache whose ``schema`` differs is treated like a corrupt one.
SCHEMA_VERSION = 1
#: Suffix appended to the cache file when it is set aside as unreadable.
BAD_SUFFIX = ".bad"
#: Read size for hashing.
CHUNK_BYTES = 8 * 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path | str) -> str:
    """Lowercase hex SHA-256 of the whole file, read in 8 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(path: Path | str) -> str:
    """The cache key of ``path``: absolute and case-normalised, without
    touching the file system (no symlink resolution)."""
    return os.path.normcase(os.path.abspath(str(path)))


def transformer_path(
    registry: ModelRegistry, base_id: str, selection: Mapping[str, str]
) -> Path | None:
    """Absolute path of the transformer ``selection`` names for ``base_id``.

    The ONE way every caller finds the file: the load path that hashes it,
    ``GET /models`` that reads the result, and ``transformer_weight_class``.
    ``"default"`` resolves to the descriptor's ``default_file`` here — the
    pipeline's selection paths do not carry it. ``None`` when the name does
    not resolve (unknown name, missing file, no transformer category).
    """
    name = selection.get("transformer", DEFAULT_NAME)
    try:
        return registry.resolve("transformer", name, base_model=base_id)
    except APIError:
        return None


def _role_of(sha256: str, known: Iterable[KnownWeight]) -> str | None:
    for weight in known:
        if weight.sha256 == sha256:
            return weight.role
    return None


class WeightIdentity:
    """The in-memory hash cache plus its file. Build it with :meth:`load`."""

    def __init__(self, path: Path | str, entries: dict[str, dict] | None = None) -> None:
        self.path = Path(path)
        self._entries: dict[str, dict] = dict(entries or {})
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ read

    @classmethod
    def for_config(cls, config: AppConfig) -> WeightIdentity:
        """The cache that lives next to ``config.state_path``."""
        return cls.load(config.state_path.with_name(CACHE_FILENAME))

    @classmethod
    def load(cls, path: Path | str) -> WeightIdentity:
        """Read the cache file once. Never raises: absent -> empty, unusable
        -> renamed aside to ``<name>.bad`` (WARNING) and empty."""
        identity = cls(path)
        if not identity.path.exists():
            return identity
        try:
            raw = json.loads(identity.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            identity._discard(f"読み込めませんでした ({exc})")
            return identity
        problem = _validate(raw)
        if problem is not None:
            identity._discard(problem)
            return identity
        identity._entries = {k: dict(v) for k, v in raw["entries"].items()}
        return identity

    def cached_role(self, path: Path | str, known: Iterable[KnownWeight]) -> str | None:
        """Role of ``path`` from the cache alone (stat + memory; never hashes)."""
        try:
            stat = os.stat(path)
        except OSError:
            return None
        with self._lock:
            sha256 = self._fresh_sha256(cache_key(path), stat)
        return None if sha256 is None else _role_of(sha256, known)

    # ----------------------------------------------------------------- write

    def identify(self, path: Path | str, known: Iterable[KnownWeight]) -> str | None:
        """Role of ``path``, hashing it when the cache has no current entry.

        A file whose size matches no known weight is not hashed (``None``).
        The hash runs OUTSIDE the lock, so ``cached_role`` callers are never
        held up by it. A read error while hashing propagates (the caller
        decides that it is only a WARNING).
        """
        known = tuple(known)
        try:
            stat = os.stat(path)
        except OSError:
            return None
        if stat.st_size not in {w.size for w in known}:
            return None
        key = cache_key(path)
        with self._lock:
            sha256 = self._fresh_sha256(key, stat)
        if sha256 is None:
            logger.info(
                "重みの種別を判定するため SHA-256 を計算します（初回のみ・数十秒〜数分）: %s",
                path,
            )
            sha256 = sha256_file(path)
            with self._lock:
                self._entries[key] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256,
                }
                self._save_locked()
        return _role_of(sha256, known)

    # ---------------------------------------------------------------- helper

    def _fresh_sha256(self, key: str, stat: os.stat_result) -> str | None:
        """The cached hash of ``key`` if size and mtime_ns still match.
        CALLED ONLY UNDER ``_lock``."""
        entry = self._entries.get(key)
        if (
            entry is None
            or entry.get("size") != stat.st_size
            or entry.get("mtime_ns") != stat.st_mtime_ns
        ):
            return None
        return entry.get("sha256")

    def _save_locked(self) -> None:
        """Persist the entries atomically. CALLED ONLY UNDER ``_lock``.
        A failed write is a WARNING (the in-memory cache stays valid)."""
        payload = {"schema": SCHEMA_VERSION, "entries": self._entries}
        try:
            self._write_atomic(json.dumps(payload, indent=2, ensure_ascii=False))
        except OSError as exc:
            logger.warning(
                "重みの判定のキャッシュ %s を保存できませんでした (%s)。"
                "次回起動後に同じファイルをもう一度計算します。",
                self.path,
                exc,
            )

    def _write_atomic(self, text: str) -> None:
        """Temp file in the same directory + ``os.replace`` (the same recipe
        as ``RuntimeState._write_atomic``)."""
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(
            dir=str(directory), prefix=self.path.name + ".", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp_path, self.path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise

    def _discard(self, reason: str) -> None:
        """Set the unusable cache file aside and log why (empty cache applies)."""
        bad = self.path.with_name(self.path.name + BAD_SUFFIX)
        try:
            os.replace(self.path, bad)
            moved = f" {bad.name} に退避しました。"
        except OSError as exc:
            moved = f" 退避もできませんでした ({exc})。"
        logger.warning(
            "重みの判定のキャッシュ %s は%s%s空のキャッシュで続けます。",
            self.path,
            reason,
            moved,
        )


def _validate(raw: object) -> str | None:
    """None when ``raw`` is a usable cache document, else the reason it is not."""
    if not isinstance(raw, dict):
        return "JSONオブジェクトではありません。"
    if raw.get("schema") != SCHEMA_VERSION:
        return f"schemaが{SCHEMA_VERSION}ではありません({raw.get('schema')!r})。"
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        return "entriesがオブジェクトではありません。"
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            return f"entries.{key} がオブジェクトではありません。"
        size, mtime_ns, sha256 = entry.get("size"), entry.get("mtime_ns"), entry.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or isinstance(mtime_ns, bool)
            or not isinstance(mtime_ns, int)
            or not isinstance(sha256, str)
            or not _SHA256_RE.fullmatch(sha256)
        ):
            return f"entries.{key} の項目の型が正しくありません。"
    return None
