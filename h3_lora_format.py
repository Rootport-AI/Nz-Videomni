"""MiniMax H3 LoRA key mapping -- torch-free, shared by the app and the worker.

Lives next to ``h3_format.py`` / ``sft_quant_format.py`` because both
processes import it: the app (its own torch-less ``.venv``) uses it for the
"supported / not supported" verdict of the LoRA list, and the engine
(``engine_h3.lora``) uses it to find the A/B tensors of each target module.
Imports only the standard library; it reads no file (callers pass the tensor
names, and optionally the shapes and dtypes, from a safetensors header).

Accepted naming ("what ComfyUI can read", owner ruling 1 of 2026-10-24):

* dotted: ``[diffusion_model.]<module>.lora_A.weight`` / ``.lora_B.weight``
  or ``.lora_down.weight`` / ``.lora_up.weight``, optional ``<module>.alpha``;
* kohya underscore: ``lora_unet_[token_refiner_]blocks_<i>_<attn_qkv_proj|
  attn_out_proj|mlp_fc1|mlp_fc2>`` with the same suffixes.

``<module>`` is a canonical H3 module name: ``blocks.{i}.<target>`` or
``token_refiner.blocks.{j}.<target>`` with ``<target>`` in
:data:`H3_LORA_TARGETS`. The scale (alpha / rank, 1.0 without alpha) is NOT
computed here because it needs the alpha value; the engine does it.

A file is ``supported`` when it has at least one pair, no DoRA/LoHa/LoKr key,
no key outside the targets (``unmapped`` empty), and -- when shapes / dtypes
are given -- within each pair A and B are 2-D with A.shape[0] == B.shape[1], and
A/B are stored as BF16/F16/F32.
Missing modules (e.g. no token_refiner) do not make a file unsupported.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

#: Module suffixes (relative to a block) a H3 LoRA may target.
H3_LORA_TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")

#: Storage dtypes (safetensors spelling) the engine can lift to f32 losslessly.
H3_LORA_DTYPES = ("BF16", "F16", "F32")

_DOTTED_PREFIX = "diffusion_model."
_TARGET_ALT = "|".join(re.escape(t) for t in H3_LORA_TARGETS)
_DOTTED_MODULE = re.compile(rf"^(?:token_refiner\.)?blocks\.\d+\.(?:{_TARGET_ALT})$")
_UNDERSCORE_MODULE = re.compile(
    r"^lora_unet_(token_refiner_)?blocks_(\d+)_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)$"
)
_UNDERSCORE_TARGET = {
    "attn_qkv_proj": "attn.qkv_proj",
    "attn_out_proj": "attn.out_proj",
    "mlp_fc1": "mlp.fc1",
    "mlp_fc2": "mlp.fc2",
}
#: key suffix -> (role, suffix family). "ab" = lora_A/lora_B, "kohya" = lora_down/lora_up.
_SUFFIXES = (
    (".lora_A.weight", "a", "ab"),
    (".lora_B.weight", "b", "ab"),
    (".lora_down.weight", "a", "kohya"),
    (".lora_up.weight", "b", "kohya"),
    (".alpha", "alpha", None),
)
#: Markers of adapter kinds this engine does not run (DoRA / LoHa / LoKr).
_FOREIGN_ADAPTER = re.compile(r"(?:dora_scale|\.hada_|\.lokr_)")

_NG = "MiniMax H3 の LoRA として使えない可能性があります: "


@dataclass(frozen=True)
class LoraPairKeys:
    """File keys of one target module (``alpha`` is None when the file has none)."""

    a: str
    b: str
    alpha: Optional[str]


@dataclass(frozen=True)
class H3LoraKeyMap:
    layout: str                      # "ab" | "kohya" | "kohya_underscore" | "mixed" | "none"
    pairs: dict[str, LoraPairKeys]   # canonical module name -> file keys
    unmapped: tuple[str, ...]        # keys outside the targets, and keys without their partner
    supported: bool
    reason: Optional[str]            # Japanese, user-facing; None when supported


def _split(name: str) -> tuple[str, str, str] | None:
    """``name`` -> (canonical module, role, naming) or None when it is not a H3 target key.

    naming is "ab" / "kohya" (dotted) or "kohya_underscore" (lora_unet_ names).
    """
    for suffix, role, family in _SUFFIXES:
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        m = _UNDERSCORE_MODULE.match(stem)
        if m:
            refiner, idx, target = m.groups()
            module = f"{'token_refiner.' if refiner else ''}blocks.{idx}.{_UNDERSCORE_TARGET[target]}"
            return module, role, "kohya_underscore"
        if stem.startswith(_DOTTED_PREFIX):
            stem = stem[len(_DOTTED_PREFIX):]
        if _DOTTED_MODULE.match(stem):
            return stem, role, family or "alpha"
        return None
    return None


def map_lora_keys(
    names: Iterable[str],
    shapes: Optional[Mapping[str, Sequence[int]]] = None,
    dtypes: Optional[Mapping[str, str]] = None,
) -> H3LoraKeyMap:
    """Map the tensor names of a LoRA file onto H3 modules (see the module docstring)."""
    names = list(names)
    found: dict[str, dict[str, str]] = {}   # module -> role -> key
    namings: dict[str, set[str]] = {}       # module -> naming of its A/B keys
    unmapped: list[str] = []
    foreign = False
    for name in names:
        if _FOREIGN_ADAPTER.search(name):
            foreign = True
            unmapped.append(name)
            continue
        hit = _split(name)
        if hit is None:
            unmapped.append(name)
            continue
        module, role, naming = hit
        roles = found.setdefault(module, {})
        if role in roles:            # the same module written twice (two namings)
            unmapped.append(name)
            continue
        roles[role] = name
        if role != "alpha":
            namings.setdefault(module, set()).add(naming)

    pairs: dict[str, LoraPairKeys] = {}
    used_namings: set[str] = set()
    for module in sorted(found):
        roles = found[module]
        if "a" in roles and "b" in roles and len(namings.get(module, ())) == 1:
            pairs[module] = LoraPairKeys(a=roles["a"], b=roles["b"], alpha=roles.get("alpha"))
            used_namings |= namings[module]
        else:
            unmapped.extend(roles.values())

    if not used_namings:
        layout = "none"
    elif len(used_namings) == 1:
        layout = next(iter(used_namings))
    else:
        layout = "mixed"

    reason = None
    if not pairs:
        reason = _NG + "H3 の対象モジュール（attn.qkv_proj・attn.out_proj・mlp.fc1・mlp.fc2）の A/B の組がありません"
    elif foreign:
        reason = _NG + "DoRA／LoHa／LoKr のキーがあります（この形式には対応していません）"
    elif unmapped:
        reason = (
            _NG + f"H3 の対象に写らないキー、または相方の無いキーが {len(unmapped)} 個あります"
            f"（例: {sorted(unmapped)[0]}）"
        )
    elif shapes is not None:
        for module, keys in pairs.items():
            a, b = shapes.get(keys.a), shapes.get(keys.b)
            if a is None or b is None or len(a) != 2 or len(b) != 2:
                reason = (
                    _NG + f"{module} の A {list(a) if a is not None else None} と"
                    f" B {list(b) if b is not None else None} が 2 次元ではありません"
                )
                break
            if a[0] != b[1]:
                reason = (
                    _NG + f"{module} の A {list(a) if a is not None else None} と"
                    f" B {list(b) if b is not None else None} の rank が一致しません"
                )
                break
    if reason is None and dtypes is not None:
        for module, keys in pairs.items():
            bad = [dtypes.get(k) for k in (keys.a, keys.b) if dtypes.get(k) not in H3_LORA_DTYPES]
            if bad:
                reason = _NG + f"{module} の dtype {bad[0]} は使えません（BF16／F16／F32 のみ）"
                break

    return H3LoraKeyMap(
        layout=layout,
        pairs=pairs,
        unmapped=tuple(sorted(unmapped)),
        supported=reason is None,
        reason=reason,
    )
