"""Server runtime state — the tiny file that remembers what was loaded last
(multi-engine foundation, Docs/MULTI_ENGINE_DESIGN.md §5.4).

ONE file, ``state.json`` in the project root, holding exactly two things: which
BASE MODEL was active at the last successful pipeline load, and — per base
model — which category NAME each selectable category was on. Restart the
server and the combination the operator had chosen is still there, instead of
silently reverting to the shipped defaults.

WHAT THIS IS NOT: it is not configuration (the operator never edits it, and
``config.yaml`` never mentions a weight file any more), and it is not user
data. It is a CACHE OF THE LAST CHOICE, so every failure mode here is a
fall-back-to-defaults, never an error that reaches a request:

* file absent (first boot / deleted) -> defaults, silently;
* broken JSON, wrong ``schema``, or a value of the wrong type -> WARNING, the
  file is renamed to ``state.json.bad`` so the next save starts clean and the
  broken one can still be inspected, and defaults apply;
* a write that fails (read-only directory, full disk) -> WARNING only. Losing
  the memory of a selection must never turn a successful load into a 500.

Writes are ATOMIC: a sibling temp file plus :func:`os.replace`, so a crash (or
two saves racing) can leave the file whole-old or whole-new, never truncated —
a half-written state.json is exactly the corruption case this file exists to
avoid having to handle.

The state file is deliberately NOT under ``outputs/`` or ``uploads/`` (see
Docs/STORAGE_POLICY.md: those are the user's material). It sits in the project
root and is gitignored.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger("ltx.state")

#: Bump only for a change no reader can absorb. A file whose ``schema`` is not
#: this value is treated exactly like a corrupt one (renamed aside, defaults
#: applied) — a state file is a cache, so migrating it is never worth code.
SCHEMA_VERSION = 1

#: Suffix appended to the state file when it is set aside as unreadable.
BAD_SUFFIX = ".bad"


class RuntimeState:
    """The last-known active base model + per-base category selections.

    Construct it through :meth:`load` (the only reader) and update it through
    :meth:`save` (the only writer). Both are total functions: neither raises.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        active_base_model: str | None = None,
        selections: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.path = Path(path)
        #: Base-model id of the last successful load, or None when nothing has
        #: ever been loaded (or the file was unusable).
        self.active_base_model: str | None = active_base_model
        #: base model id -> {category: NAME}. A base model absent from this map
        #: simply has no remembered selection.
        self.selections: dict[str, dict[str, str]] = selections or {}

    # ------------------------------------------------------------------ read

    @classmethod
    def load(cls, path: Path | str) -> RuntimeState:
        """Read ``path``. Never raises — an unusable file yields the defaults.

        A file that exists but cannot be understood is RENAMED ASIDE (to
        ``<name>.bad``) rather than left in place: leaving it would make every
        subsequent boot re-log the same warning, and deleting it would destroy
        the only evidence of what went wrong.
        """
        state = cls(path)
        if not state.path.exists():
            # First boot, or the operator deleted it to reset. Not worth a log
            # line: "no state yet" is the normal state of a fresh install.
            return state
        try:
            raw = json.loads(state.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            state._discard(f"読み込めませんでした ({exc})")
            return state

        problem = _validate(raw)
        if problem is not None:
            state._discard(problem)
            return state

        state.active_base_model = raw.get("active_base_model")
        state.selections = {
            base: {c: n for c, n in sel.items()}
            for base, sel in (raw.get("selections") or {}).items()
        }
        return state

    def selection_for(self, base_model: str) -> dict[str, str]:
        """Remembered ``{category: NAME}`` for one base model (empty if none).

        A copy, so a caller cannot mutate the state through the return value.
        """
        return dict(self.selections.get(base_model) or {})

    # ----------------------------------------------------------------- write

    def save(self, active_base_model: str, selection: dict[str, str]) -> None:
        """Record ``selection`` as ``active_base_model``'s and persist.

        Only the named base model's entry is replaced — another base model's
        remembered selection survives a switch, so switching back restores it.
        Never raises: a failed write is a WARNING and the in-memory state stays
        updated (the running server keeps behaving correctly; only the memory
        across a restart is lost).
        """
        self.active_base_model = active_base_model
        self.selections[active_base_model] = dict(selection)
        payload = {
            "schema": SCHEMA_VERSION,
            "active_base_model": self.active_base_model,
            "selections": self.selections,
        }
        try:
            self._write_atomic(json.dumps(payload, indent=2, ensure_ascii=False))
        except OSError as exc:
            logger.warning(
                "サーバー実行時状態 %s を保存できませんでした (%s)。"
                "動作は続きますが、次回起動時は既定の組み合わせに戻ります。",
                self.path,
                exc,
            )

    def _write_atomic(self, text: str) -> None:
        """Write ``text`` to :attr:`path` atomically (temp file + os.replace).

        The temp file is created in the SAME directory on purpose:
        ``os.replace`` is only atomic within one filesystem, and a system temp
        dir may well be on another volume.
        """
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

    # ---------------------------------------------------------------- helper

    def _discard(self, reason: str) -> None:
        """Set the unusable state file aside and log why (defaults then apply)."""
        bad = self.path.with_name(self.path.name + BAD_SUFFIX)
        try:
            os.replace(self.path, bad)
            moved = f" {bad.name} に退避しました。"
        except OSError as exc:
            moved = f" 退避もできませんでした ({exc})。"
        logger.warning(
            "サーバー実行時状態 %s は%s%s既定の組み合わせで起動します。",
            self.path,
            reason,
            moved,
        )


def _validate(raw: object) -> str | None:
    """None when ``raw`` is a usable state document, else the reason it is not.

    Deliberately strict about TYPES and lax about VALUES: an unknown base-model
    id or category name is not a corruption (the model store legitimately
    changes between runs), and the caller's own fallback handles it. A value
    that is not a string, on the other hand, would propagate a wrong type deep
    into the registry, so it is refused here.
    """
    if not isinstance(raw, dict):
        return "JSONオブジェクトではありません。"
    if raw.get("schema") != SCHEMA_VERSION:
        return f"schemaが{SCHEMA_VERSION}ではありません({raw.get('schema')!r})。"
    active = raw.get("active_base_model")
    if active is not None and (not isinstance(active, str) or not active):
        return "active_base_modelが文字列ではありません。"
    selections = raw.get("selections")
    if selections is None:
        return None
    if not isinstance(selections, dict):
        return "selectionsがオブジェクトではありません。"
    for base, sel in selections.items():
        if not isinstance(base, str) or not isinstance(sel, dict):
            return f"selections.{base} がオブジェクトではありません。"
        for category, name in sel.items():
            if not isinstance(category, str) or not isinstance(name, str):
                return f"selections.{base} に文字列でない項目があります。"
    return None
