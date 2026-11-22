"""MiniMax H3 pi-H3 file inspection -- torch-free, shared by the app and the worker.

pi-H3 (the low-step distilled difference and DWD, Docs/MINIMAX_H3_RESEARCH_NOTES.md
§18; the experiment and the owner's rulings: Docs/VERIFICATION_LOG.md §223) ships
as one safetensors file per step count. Lives next to ``h3_lora_format.py`` /
``h3_format.py`` / ``sft_quant_format.py`` because both processes import it:
the app (its own torch-less ``.venv``, which never imports ``engine_h3``) for
the acceptance check, and the engine (``engine_h3.pipeline_h3.validate_request``,
``engine_h3.pi_h3``, ``engine_h3.dwd``) before it reads any weight. Imports
only the standard library and the root modules ``sft_quant_format`` /
``h3_format`` (both torch-free), and reads nothing but the headers and the tiny
``comfy_quant`` byte tensors.

What the file holds (keys without prefix; §18.3 of the research notes):

* blocks 0-45 ``attn.out_proj`` / ``mlp.fc1`` / ``mlp.fc2``: LoRA ``.lora_A.weight``
  / ``.lora_B.weight`` (138 pairs, no ``.alpha``);
* blocks 46-49, the same 3 layers: int8 replacements ``weight`` (I8 ``[N, K]``),
  ``weight_scale`` (F32 ``[N, 1]``), ``comfy_quant`` (12 layers);
* every block's ``adaln_proj.linear.weight`` (50) and blocks 46-49's
  ``adaln_proj.linear.bias`` (4);
* ``final_layer``: 7 tensors (:data:`FINAL_KEYS`);
* ``dwd_``: the DWD head (:func:`dwd_tensor_table`; 139 tensors plus 32 markers).

Public names
------------
Constants: :data:`PI_H3_CHOICES` (``"4steps"`` -> 4, ``"2steps"`` -> 2),
:data:`PI_H3_STEPS` (2, 4), :data:`PI_H3_SHIFT_VIDEO` (12.0),
:data:`PI_H3_SHIFT_AUDIO` (3.0), :data:`NUM_BLOCKS`, :data:`LORA_BLOCKS`,
:data:`REPLACED_BLOCKS`, :data:`LEAVES`, :data:`FINAL_KEYS`,
:data:`EXPECTED_COUNTS`, :data:`DWD_HEAD` (the head's widths,
:class:`DWDHeadDims`), :data:`DWD_INT8_LEAVES`.

Functions:

* ``choice_for_steps(steps) -> str`` -- 4 -> ``"4steps"``, 2 -> ``"2steps"``;
  any other value is a ``ValueError``.
* ``classify_keys(names) -> PiH3Keys`` -- the file's tensor names sorted into
  the groups above (``other``: names that fit none).
* ``file_problems(path, *, head=None) -> list[str]`` -- the file alone against
  the expected layout (key sets, dtypes, shapes of the int8 replacements and
  of the DWD part, every marker an int8 marker). ``[]`` when it is as expected;
  else one Japanese sentence per finding, stating only what was found.
* ``model_problems(pi_path, dit_path, *, head=None) -> list[str]`` --
  :func:`file_problems`, then (only when that is empty) every non-``dwd_`` key
  against the DiT file's header by name and shape, and the DWD head's input
  widths against the DiT's. ``[]`` when they fit.
* ``inspect_file(path, *, head=None) -> PiH3File`` -- the header, the key
  groups and the int8 markers' specs, for the engine; a file with problems is
  refused with :class:`PiH3FormatError` (a ``ValueError``) whose message joins
  them.

``head``: the DWD head's widths (None = :data:`DWD_HEAD`, read at call time).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import h3_format
import sft_quant_format

#: The operation panel's choices and their step counts (VERIFICATION_LOG §223.3).
PI_H3_CHOICES = {"4steps": 4, "2steps": 2}
PI_H3_STEPS = tuple(sorted(PI_H3_CHOICES.values()))
#: The shifts pi-H3's sigma lists are made with (research notes §18.4).
PI_H3_SHIFT_VIDEO = 12.0
PI_H3_SHIFT_AUDIO = 3.0

NUM_BLOCKS = 50
LORA_BLOCKS = tuple(range(46))
REPLACED_BLOCKS = (46, 47, 48, 49)
LEAVES = ("attn.out_proj", "mlp.fc1", "mlp.fc2")
FINAL_KEYS = tuple("final_layer." + k for k in (
    "norm.weight", "adaln_proj.linear.weight", "adaln_proj.linear.bias",
    "video_out.weight", "video_out.bias", "audio_out.weight", "audio_out.bias"))
#: The DWD head's int8 layers, relative to ``dwd_pixel_refiner.blocks.<i>.``.
DWD_INT8_LEAVES = ("attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2")
PREFIX_NORM = "dwd_norm_out."
PREFIX_REF = "dwd_pixel_refiner."

#: Float dtypes the engine lifts exactly (safetensors spelling).
_FLOAT = ("F16", "BF16", "F32")

_LORA_RE = re.compile(r"^(blocks\.(\d+)\.(?:attn\.out_proj|mlp\.fc1|mlp\.fc2))\.lora_([AB])\.weight$")
_QUANT_RE = re.compile(r"^(blocks\.(\d+)\.(?:attn\.out_proj|mlp\.fc1|mlp\.fc2))\.(weight|weight_scale|comfy_quant)$")
_ADALN_RE = re.compile(r"^blocks\.(\d+)\.adaln_proj\.linear\.(weight|bias)$")


@dataclass(frozen=True)
class DWDHeadDims:
    """The DWD head's widths (LakonLab's ``SolH3`` head as pi-H3 configures it)."""

    in_channels: int = 5376      # the DiT's hidden size
    t_dim: int = 8               # the pruned AdaLN rank (the time coordinates)
    dim: int = 2048
    heads: int = 32
    head_dim: int = 64
    ffn_mult: int = 4
    num_layers: int = 8
    expansion: tuple[int, int] = (2, 2)
    patch: int = 16
    patch_t: int = 4
    out_ch: int = 3
    num_register: int = 4

    @property
    def first_t(self) -> int:
        """Frames of a block's first latent (1 of ``patch_t``)."""
        return 1

    @property
    def rope_dim(self) -> int:
        return int(self.head_dim * 0.75)


DWD_HEAD = DWDHeadDims()


@dataclass
class PiH3Keys:
    lora: dict = field(default_factory=dict)      # module -> {"A": key, "B": key}
    quant: dict = field(default_factory=dict)     # layer -> {leaf: key}
    adaln_w: dict = field(default_factory=dict)   # block -> key
    adaln_b: dict = field(default_factory=dict)   # block -> key
    final: list = field(default_factory=list)
    dwd: list = field(default_factory=list)
    other: list = field(default_factory=list)

    def counts(self) -> dict:
        return {
            "lora_pairs": sum(1 for v in self.lora.values() if set(v) == {"A", "B"}),
            "quant_layers": sum(1 for v in self.quant.values() if set(v) == {"weight", "weight_scale", "comfy_quant"}),
            "adaln_weights": len(self.adaln_w), "adaln_biases": len(self.adaln_b),
            "final_tensors": len(self.final), "dwd_tensors": len(self.dwd),
        }


#: What :meth:`PiH3Keys.counts` gives on a pi-H3 file.
EXPECTED_COUNTS = {"lora_pairs": 138, "quant_layers": 12, "adaln_weights": 50, "adaln_biases": 4,
                   "final_tensors": 7, "dwd_tensors": 171}


@dataclass(frozen=True)
class PiH3File:
    header: sft_quant_format.Header
    keys: PiH3Keys
    quant_specs: dict          # blocks 46-49 layer -> h3_format.QuantSpec
    dwd_specs: dict            # refiner-relative int8 layer ("blocks.0.attn.to_qkv") -> QuantSpec


class PiH3FormatError(ValueError):
    """A pi-H3 file the engine cannot use; the message is user-facing Japanese."""


def choice_for_steps(steps) -> str:
    """4 -> ``"4steps"``, 2 -> ``"2steps"``."""
    for name, n in PI_H3_CHOICES.items():
        if not isinstance(steps, bool) and isinstance(steps, int) and steps == n:
            return name
    raise ValueError(f"pi-H3 のステップ数は {'・'.join(str(n) for n in PI_H3_STEPS)} のどれかです（{steps!r}）")


def classify_keys(names: Iterable[str]) -> PiH3Keys:
    keys = PiH3Keys()
    for k in sorted(names):
        if k.startswith("dwd_"):
            keys.dwd.append(k)
        elif m := _LORA_RE.match(k):
            keys.lora.setdefault(m.group(1), {})[m.group(3)] = k
        elif m := _QUANT_RE.match(k):
            keys.quant.setdefault(m.group(1), {})[m.group(3)] = k
        elif m := _ADALN_RE.match(k):
            (keys.adaln_w if m.group(2) == "weight" else keys.adaln_b)[int(m.group(1))] = k
        elif k in FINAL_KEYS:
            keys.final.append(k)
        else:
            keys.other.append(k)
    return keys


def dwd_tensor_table(head: DWDHeadDims | None = None) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Every ``dwd_`` tensor but the markers -> (dtype class, shape).

    The dtype class is ``"float"`` (F16 / BF16 / F32), or the exact ``"I8"`` /
    ``"F32"`` of an int8 layer's ``weight`` / ``weight_scale``.
    """
    h = head or DWD_HEAD
    d, (eh, ew), p = h.dim, h.expansion, h.patch
    t: dict[str, tuple[str, tuple[int, ...]]] = {
        PREFIX_NORM + "linear.weight": ("float", (2 * h.in_channels, h.t_dim)),
        PREFIX_NORM + "linear.bias": ("float", (2 * h.in_channels,)),
        PREFIX_REF + "feature_expand.weight": ("float", (d * eh * ew, h.in_channels)),
        PREFIX_REF + "feature_expand.bias": ("float", (d * eh * ew,)),
        PREFIX_REF + "register_tokens": ("float", (1, h.num_register, d)),
        PREFIX_REF + "norm_out.weight": ("float", (d,)),
        PREFIX_REF + "norm_out.bias": ("float", (d,)),
        PREFIX_REF + "proj_out.weight": ("float", (h.out_ch * h.patch_t * p * p, d)),
        PREFIX_REF + "proj_out.bias": ("float", (h.out_ch * h.patch_t * p * p,)),
        PREFIX_REF + "first_frame_proj_out.weight": ("float", (h.out_ch * h.first_t * p * p, d)),
        PREFIX_REF + "first_frame_proj_out.bias": ("float", (h.out_ch * h.first_t * p * p,)),
    }
    lin = {"attn.to_qkv": (3 * d, d), "attn.to_out": (d, d),
           "ff.w1": (2 * h.ffn_mult * d, d), "ff.w2": (d, h.ffn_mult * d)}
    for i in range(h.num_layers):
        b = f"{PREFIX_REF}blocks.{i}."
        for leaf in ("norm1.weight", "norm2.weight", "scale1", "scale2"):
            t[b + leaf] = ("float", (d,))
        for leaf, (o, k) in lin.items():
            t[f"{b}{leaf}.weight"] = ("I8", (o, k))
            t[f"{b}{leaf}.weight_scale"] = ("F32", (o, 1))
            t[f"{b}{leaf}.bias"] = ("float", (o,))
    return t


def dwd_int8_layers(head: DWDHeadDims | None = None) -> list[str]:
    """The head's int8 layers, refiner-relative (``blocks.0.attn.to_qkv`` ...)."""
    h = head or DWD_HEAD
    return [f"blocks.{i}.{leaf}" for i in range(h.num_layers) for leaf in DWD_INT8_LEAVES]


def _name(path) -> str:
    return Path(str(path)).name


def _few(items, n: int = 3) -> str:
    items = sorted(items)
    head = "・".join(str(x) for x in items[:n])
    return head + (f" ほか {len(items) - n} 個" if len(items) > n else "")


def _read(path):
    """``(header, None)`` or ``(None, problem sentence)``."""
    p = Path(str(path or ""))
    if not str(path or "").strip():
        return None, "pi-H3 のファイルのパスが空です"
    if not p.is_file():
        return None, f"pi-H3 のファイルが見つかりません（{path}）"
    try:
        return sft_quant_format.read_header(p), None
    except (sft_quant_format.QuantFormatError, OSError) as exc:
        return None, f"pi-H3 のファイルのヘッダーを読めません（{p.name}: {exc}）"


def _marker(path, header, key: str):
    """The marker's spec, or a problem sentence (int8 is the only accepted format)."""
    try:
        raw = sft_quant_format.read_tensor_bytes(str(path), header, key)
        return h3_format.parse_comfy_quant(raw, accepted=(h3_format.INT8,)), None
    except (h3_format.H3FormatError, sft_quant_format.QuantFormatError, OSError) as exc:
        return None, f"{key} を int8 の量子化の印として読めません（{exc}）"


def _layout_problems(path, header, keys: PiH3Keys, head: DWDHeadDims) -> tuple[list[str], dict, dict]:
    """The key sets, dtypes and own shapes; the markers' specs as a side result."""
    t = header.tensors
    bad: list[str] = []
    if keys.other:
        bad.append(f"pi-H3 のどの部分にも当たらないキーが {len(keys.other)} 個あります（{_few(keys.other)}）")
    want_lora = {f"blocks.{i}.{leaf}" for i in LORA_BLOCKS for leaf in LEAVES}
    pairs = {m for m, ab in keys.lora.items() if set(ab) == {"A", "B"}}
    if pairs != want_lora:
        bad.append(
            f"LoRA の組が {len(pairs)} 組で、ブロック 0〜45 の 3 層（{len(want_lora)} 組）と合いません"
            + (f"（無い組: {_few(want_lora - pairs)}）" if want_lora - pairs else "")
            + (f"（余分な組: {_few(pairs - want_lora)}）" if pairs - want_lora else "")
        )
    half = sorted(m for m, ab in keys.lora.items() if set(ab) != {"A", "B"})
    if half:
        bad.append(f"A と B の片方しか無い LoRA があります（{_few(half)}）")
    for m in sorted(pairs):
        a, b = t[keys.lora[m]["A"]], t[keys.lora[m]["B"]]
        if a.dtype not in _FLOAT or b.dtype not in _FLOAT:
            bad.append(f"LoRA {m} の型が {a.dtype}／{b.dtype} です（F16／BF16／F32 のみ）")
        elif len(a.shape) != 2 or len(b.shape) != 2 or a.shape[0] != b.shape[1]:
            bad.append(f"LoRA {m} の A {list(a.shape)} と B {list(b.shape)} の形が組になっていません")
    want_quant = {f"blocks.{i}.{leaf}" for i in REPLACED_BLOCKS for leaf in LEAVES}
    whole = {m for m, lv in keys.quant.items() if set(lv) == {"weight", "weight_scale", "comfy_quant"}}
    if whole != want_quant or set(keys.quant) != want_quant:
        bad.append(
            f"int8 の置き換えの層（weight・weight_scale・comfy_quant がそろった層）が {len(whole)} 層で、"
            f"ブロック 46〜49 の 3 層（{len(want_quant)} 層）と合いません"
        )
    specs: dict = {}
    for layer in sorted(whole & want_quant):
        lv = keys.quant[layer]
        w, s = t[lv["weight"]], t[lv["weight_scale"]]
        if w.dtype != "I8" or len(w.shape) != 2:
            bad.append(f"{lv['weight']} が {w.dtype} {list(w.shape)} です（I8 の 2 次元のはず）")
        elif s.dtype != "F32" or tuple(s.shape) != (w.shape[0], 1):
            bad.append(f"{lv['weight_scale']} が {s.dtype} {list(s.shape)} です（F32 [{w.shape[0]}, 1] のはず）")
        spec, why = _marker(path, header, lv["comfy_quant"])
        if why:
            bad.append(why)
        else:
            specs[layer] = spec
    if set(keys.adaln_w) != set(range(NUM_BLOCKS)):
        bad.append(f"AdaLN の重みが {len(keys.adaln_w)} ブロック分で、ブロック 0〜49 と合いません")
    if set(keys.adaln_b) != set(REPLACED_BLOCKS):
        bad.append(f"AdaLN の偏りがブロック {sorted(keys.adaln_b)} にあり、ブロック 46〜49 と合いません")
    if set(keys.final) != set(FINAL_KEYS):
        missing = set(FINAL_KEYS) - set(keys.final)
        bad.append(f"出力層のテンソルが {len(keys.final)} 個です（{len(FINAL_KEYS)} 個のはず。無いもの: {_few(missing)}）")
    for k in sorted(list(keys.adaln_w.values()) + list(keys.adaln_b.values()) + keys.final):
        if t[k].dtype not in _FLOAT:
            bad.append(f"{k} の型が {t[k].dtype} です（F16／BF16／F32 のみ）")
    # the DWD part
    table = dwd_tensor_table(head)
    markers_want = {f"{PREFIX_REF}{layer}.comfy_quant" for layer in dwd_int8_layers(head)}
    have = set(keys.dwd)
    missing = (set(table) | markers_want) - have
    extra = have - set(table) - markers_want
    if missing:
        bad.append(f"DWD のテンソルが {len(missing)} 個足りません（{_few(missing)}）")
    if extra:
        bad.append(f"DWD のヘッドに無いテンソルが {len(extra)} 個あります（{_few(extra)}）")
    for k, (kind, shape) in table.items():
        info = t.get(k)
        if info is None:
            continue
        ok_dtype = info.dtype in _FLOAT if kind == "float" else info.dtype == kind
        if not ok_dtype or tuple(info.shape) != shape:
            want = "F16／BF16／F32" if kind == "float" else kind
            bad.append(f"{k} が {info.dtype} {list(info.shape)} です（{want} {list(shape)} のはず）")
    dwd_specs: dict = {}
    for layer in dwd_int8_layers(head):
        key = f"{PREFIX_REF}{layer}.comfy_quant"
        if key not in have:
            continue
        spec, why = _marker(path, header, key)
        if why:
            bad.append(why)
        else:
            dwd_specs[layer] = spec
    return bad, specs, dwd_specs


def file_problems(path, *, head: DWDHeadDims | None = None) -> list[str]:
    """The pi-H3 file alone against the expected layout; ``[]`` when it is as expected."""
    header, why = _read(path)
    if why:
        return [why]
    bad, _specs, _dwd = _layout_problems(path, header, classify_keys(header.tensors), head or DWD_HEAD)
    return bad


def inspect_file(path, *, head: DWDHeadDims | None = None) -> PiH3File:
    """Header, key groups and markers of a pi-H3 file; :class:`PiH3FormatError` when it has problems."""
    header, why = _read(path)
    if why:
        raise PiH3FormatError(why)
    keys = classify_keys(header.tensors)
    bad, specs, dwd_specs = _layout_problems(path, header, keys, head or DWD_HEAD)
    if bad:
        raise PiH3FormatError(f"pi-H3 のファイル（{_name(path)}）の中身が想定と違います: " + "；".join(bad))
    return PiH3File(header=header, keys=keys, quant_specs=specs, dwd_specs=dwd_specs)


def _logical_weight_shape(t: dict, cfg, layer: str):
    """The DiT layer's ``[N, K]`` (a w4a8 weight is stored ``[N, K/2]``), or None when absent."""
    info = t.get(f"{cfg.prefix}{layer}.weight")
    if info is None:
        return None
    shape = tuple(info.shape)
    spec = cfg.quant.get(layer)
    if spec is not None and spec.format == h3_format.W4A8 and len(shape) == 2:
        shape = (shape[0], 2 * shape[1])
    return shape


def model_problems(pi_path, dit_path, *, head: DWDHeadDims | None = None) -> list[str]:
    """:func:`file_problems`, then the pi-H3 keys against the DiT file by name and shape; ``[]`` when they fit."""
    head = head or DWD_HEAD
    bad = file_problems(pi_path, head=head)
    if bad:
        return bad
    try:
        dit_header, cfg = h3_format.inspect_h3_dit(str(dit_path))
    except (h3_format.H3FormatError, OSError) as exc:
        return [f"本体の重み（{_name(dit_path)}）のヘッダーを読めません（{exc}）"]
    pi = sft_quant_format.read_header(Path(str(pi_path)))
    keys = classify_keys(pi.tensors)
    p, t = pi.tensors, dit_header.tensors
    out: list[str] = []
    if cfg.num_layers != NUM_BLOCKS:
        out.append(f"本体のブロックが {cfg.num_layers} 個です（pi-H3 は {NUM_BLOCKS} ブロックの本体に当てるものです）")
        return out
    for m, ab in sorted(keys.lora.items()):
        shape = _logical_weight_shape(t, cfg, m)
        if shape is None:
            out.append(f"LoRA の当て先 {m} が本体にありません")
            continue
        n, k = shape
        a, b = tuple(p[ab["A"]].shape), tuple(p[ab["B"]].shape)
        if a[1] != k or b[0] != n:
            out.append(f"LoRA {m} の A {list(a)}・B {list(b)} が本体の層 [{n}, {k}] と合いません")
    for layer, lv in sorted(keys.quant.items()):
        shape = _logical_weight_shape(t, cfg, layer)
        w = tuple(p[lv["weight"]].shape)
        if shape is None:
            out.append(f"置き換えの層 {layer} が本体にありません")
        elif shape != w:
            out.append(f"置き換えの層 {layer} の重み {list(w)} が本体の層 {list(shape)} と合いません")
    for k in sorted(list(keys.adaln_w.values()) + list(keys.adaln_b.values()) + keys.final):
        info = t.get(cfg.prefix + k)
        if info is None:
            out.append(f"{k} が本体にありません")
        elif tuple(info.shape) != tuple(p[k].shape):
            out.append(f"{k} の形 {list(p[k].shape)} が本体の {list(info.shape)} と合いません")
    if head.in_channels != cfg.hidden_size or head.t_dim != cfg.adaln_rank:
        out.append(
            f"DWD のヘッドの入力（幅 {head.in_channels}・時刻 {head.t_dim}）が本体（幅 {cfg.hidden_size}・"
            f"時刻 {cfg.adaln_rank}）と合いません"
        )
    return out
