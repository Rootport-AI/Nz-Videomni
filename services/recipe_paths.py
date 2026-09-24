"""Store-relative file paths in the generation recipe (metadata.json).

Some engine-reported blocks carry absolute local paths (``source_audio_path``,
``source_path``, ``mask_path``, ``canvas_path``). The recipe is written to
``metadata.json`` and embedded into the delivered mp4, so these values are
rewritten relative to the storage area they live in — the same shape as
``output.path`` (``outputs/<job>/output.mp4``) and an upload's ``stored_path``
(``uploads/...``). No absolute path (which would reveal the local user name
and break when the folder moves) is written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# The store folder names a relative recipe path starts with.
_STORE_NAMES = ("uploads", "outputs")


def _deepest_match(path: Path, stores: tuple[tuple[str, Path], ...]) -> str | None:
    """``<store name>/<relative>`` for the deepest store base ``path`` is under.

    Every base of every store is tried, and the one with the most components
    wins, so a store nested inside the other (``uploads`` inside ``outputs``,
    or the reverse) maps to the inner one regardless of the order checked.
    """
    best: tuple[int, str] | None = None
    for name, store_dir in stores:
        for base in _bases(store_dir):
            try:
                rel = path.relative_to(base)
            except ValueError:
                continue
            depth = len(base.parts)
            if best is None or depth > best[0]:
                best = (depth, (Path(name) / rel).as_posix())
    return None if best is None else best[1]


def _bases(base: Path) -> tuple[Path, ...]:
    """``base`` as given and as resolved (8.3 short names, junctions)."""
    try:
        resolved = base.resolve()
    except OSError:
        return (base,)
    return (base, resolved) if resolved != base else (base,)


def _relativize_one(value: str, output_dir: Path, upload_dir: Path) -> str:
    path = Path(value)
    # 1. Relative (``output.path``) — unchanged. A rooted path without a drive
    #    letter (leading backslash) is not "absolute" on Windows but still names
    #    a location on this machine, so it goes on to the rules below.
    if not path.is_absolute() and not path.root:
        return value
    # 2./3. Under the output or the upload store (the deeper base wins when one
    #    store sits inside the other).
    matched = _deepest_match(path, (("outputs", output_dir), ("uploads", upload_dir)))
    if matched is not None:
        return matched
    # 4. Elsewhere (a job from before a rename or a move): keep from the LAST
    #    ``uploads``/``outputs`` component onward, otherwise the file name only.
    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() in _STORE_NAMES:
            return "/".join(parts[index:])
    return path.name


def relativize_recipe_paths(obj: Any, *, output_dir: Path, upload_dir: Path) -> Any:
    """Return a copy of ``obj`` with absolute path values made store-relative.

    Pure and recursive over dicts and lists. Only string values under keys whose
    name ends with ``path`` are touched; ``None`` and every other value pass
    through unchanged. Decision per value:

    1. relative (not absolute and no root) -> unchanged;
    2. under ``output_dir`` -> ``outputs/<relative>``;
    3. under ``upload_dir`` -> ``uploads/<relative>``
       (when one store is inside the other, the deeper one is used);
    4. otherwise -> from the last ``uploads``/``outputs`` component onward,
       else the file name only.

    Separators are ``/`` (``Path.as_posix``), matching ``output.path``.
    """
    output_dir = Path(output_dir)
    upload_dir = Path(upload_dir)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict = {}
            for key, value in node.items():
                if isinstance(key, str) and key.endswith("path") and isinstance(value, str):
                    out[key] = _relativize_one(value, output_dir, upload_dir)
                else:
                    out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(obj)
