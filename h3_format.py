"""MiniMax H3 safetensors header inspection -- torch-free.

Lives next to ``sft_quant_format.py`` / ``chain_math.py`` because the app
process (its own torch-less ``.venv``) will call it for the pre-load check in
the second stage, and the engine (``engine_h3``) calls it before it builds
anything. Built on ``sft_quant_format.read_header`` (the one place that
validates offsets against the file size) and reads nothing but the header and
the tiny ``comfy_quant`` byte tensors.

The detection and shape-derivation rules describe ComfyUI's behaviour as a
SPECIFICATION (R2 report, section H); no ComfyUI code is copied:

* an H3 DiT is a file with BOTH ``{prefix}video_patch_proj.weight`` and
  ``{prefix}audio_patch_proj.weight``; the prefix is one of ``H3_PREFIXES``;
* the configuration comes from tensor shapes (layer counts, widths, heads,
  the pruned AdaLN table's grid size G and rank R);
* a layer's ``comfy_quant`` is read for ``format`` / ``convrot`` /
  ``convrot_groupsize`` only (top level or under ``params``); every other key
  is ignored. Only ``int8_tensorwise`` is accepted in this stage.

What this stage refuses, each with a Japanese reason the UI can show as is:
not H3 / the non-pruned DiT (no ``adaln_t_table``; it needs the time embedder
this engine does not build yet) / a required tensor missing / a shape that
contradicts the others / an int8 layer without its scale or marker.
Names this module does not know are returned in ``unknown`` so the loader can
log them and drop them, instead of refusing a file over an extra tensor.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import sft_quant_format
from sft_quant_format import Header, QuantFormatError, TensorInfo, read_header  # noqa: F401 -- re-exported

#: Accepted ``comfy_quant`` formats. int8 is the only one the engine's
#: ``H3Int8Linear`` (comfy_kitchen.int8_linear) can run.
ACCEPTED_FORMATS = ("int8_tensorwise",)
DEFAULT_CONVROT_GROUPSIZE = 256

H3_PREFIXES = ("", "model.diffusion_model.", "model.model.", "net.")

_NG = "MiniMax H3 のファイルとして使えません: "


class H3FormatError(ValueError):
    """A header that cannot run on this engine; the message is user-facing Japanese."""


@dataclass(frozen=True)
class QuantSpec:
    format: str                 # "int8_tensorwise" only
    convrot: bool
    convrot_groupsize: int      # 256 by default; other values pass through to int8_linear


def check_model_paths(files: dict[str, str], dirs: dict[str, str]) -> list[str]:
    """The one existence check of a load's paths (worker and pipeline both call it).

    ``files`` / ``dirs`` map a payload field name to its path. An empty value
    is a ``ValueError``, a missing file / folder a ``FileNotFoundError``; both
    name the field. Returns one ASCII-safe log line per path.
    """
    lines: list[str] = []
    for field, raw in files.items():
        if not raw:
            raise ValueError(f"load: {field} is required")
        path = Path(str(raw))
        if not path.is_file():
            raise FileNotFoundError(f"{field} does not exist: {path}")
        lines.append(f"load: {field} OK ({path.stat().st_size} bytes) {path}")
    for field, raw in dirs.items():
        if not raw:
            raise ValueError(f"load: {field} is required")
        path = Path(str(raw))
        if not path.is_dir():
            raise FileNotFoundError(f"{field} is not a folder: {path}")
        lines.append(f"load: {field} OK {path}")
    return lines


def parse_comfy_quant(raw: bytes) -> QuantSpec:
    """One layer's ``comfy_quant`` bytes -> :class:`QuantSpec`.

    Reads ``format`` / ``convrot`` / ``convrot_groupsize`` from the top level or
    from a nested ``params`` (the top level wins, as in ComfyUI); everything
    else (``per_row``, ``orig_dtype``, ``full_precision_matrix_mult`` ...) is
    ignored. A group size other than 256 is accepted and handed to
    ``int8_linear`` unchanged (it runs on the non-fused path there).
    """
    try:
        doc = json.loads(bytes(raw).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 -- any decode failure is a format error
        raise H3FormatError(_NG + f"量子化の印（comfy_quant）が JSON として読めません（{exc}）") from exc
    if not isinstance(doc, dict):
        raise H3FormatError(_NG + "量子化の印（comfy_quant）が JSON オブジェクトではありません")
    params = doc.get("params", {})
    if not isinstance(params, dict):
        params = {}
    merged = {**params, **doc}
    fmt = merged.get("format")
    if fmt not in ACCEPTED_FORMATS:
        raise H3FormatError(
            _NG + f"量子化の形式 format={fmt!r} には対応していません（対応: {'・'.join(ACCEPTED_FORMATS)}）"
        )
    convrot = merged.get("convrot", False)
    if not isinstance(convrot, bool):
        raise H3FormatError(_NG + f"量子化の印の convrot={convrot!r} が真偽値ではありません")
    groupsize = merged.get("convrot_groupsize", DEFAULT_CONVROT_GROUPSIZE)
    if isinstance(groupsize, bool) or not isinstance(groupsize, int) or groupsize <= 0:
        raise H3FormatError(_NG + f"量子化の印の convrot_groupsize={groupsize!r} が正の整数ではありません")
    return QuantSpec(format=str(fmt), convrot=convrot, convrot_groupsize=int(groupsize))


# --------------------------------------------------------------------------- #
# DiT
# --------------------------------------------------------------------------- #

def _matching_prefixes(names: set[str]) -> list[str]:
    return [p for p in H3_PREFIXES if f"{p}video_patch_proj.weight" in names]


def detect_h3_prefix(names: Iterable[str]) -> str:
    """The prefix under which ``video_patch_proj.weight`` sits; H3FormatError when none or ambiguous."""
    names = set(names)
    found = _matching_prefixes(names)
    if not found:
        raise H3FormatError(_NG + "video_patch_proj.weight が見つかりません（MiniMax H3 の DiT ではありません）")
    if len(found) > 1:
        raise H3FormatError(_NG + f"接頭辞が 1 つに決まりません（{found}）")
    return found[0]


def is_h3_dit_header(header: Header) -> bool:
    """True when some H3 prefix carries BOTH patch projections (ComfyUI's detection rule)."""
    names = set(header.tensors)
    return any(f"{p}audio_patch_proj.weight" in names for p in _matching_prefixes(names))


@dataclass(frozen=True)
class H3DiTConfig:
    prefix: str
    hidden_size: int            # video_patch_proj.weight.shape[0] = 5376
    num_layers: int             # number of blocks.N = 50
    token_refiner_layers: int   # number of token_refiner.blocks.N = 2
    num_heads: int              # qkv_proj.weight.shape[0] / (3*head_dim) = 56
    head_dim: int               # blocks.0.attn.q_norm.weight.shape[0] = 128
    ffn_hidden: int             # fc1.weight.shape[0] / 2 = 14336
    text_dim: int               # condition_proj.weight.shape[1] = 5120
    latents_dim: int            # final_layer.video_out.weight.shape[0] / 4 = 24
    audio_latents_dim: int      # final_layer.audio_out.weight.shape[0] = 32
    pruned: bool                # adaln_t_table present (always True: the non-pruned file is refused)
    adaln_grid: int | None      # adaln_t_table.shape[0] (1025)
    adaln_rank: int | None      # adaln_t_table.shape[1] (8)
    rope_inv_freq_len: int      # rope.inv_freq.shape[0] = 16
    quant: dict[str, QuantSpec]  # prefix-stripped layer ("blocks.0.attn.qkv_proj") -> spec
    unknown: tuple[str, ...]    # names this module does not know (prefix-stripped; outside-prefix names as is)


_LIN = r"(?:weight|bias|weight_scale|comfy_quant)"
_BLOCK_BODY = (
    rf"attn\.(?:qkv_proj|out_proj)\.{_LIN}|attn\.(?:q_norm|k_norm)\.weight"
    rf"|mlp\.(?:fc1|fc2)\.{_LIN}|norm1\.weight|norm2\.weight"
)
_DIT_KNOWN = re.compile(
    r"^(?:"
    rf"blocks\.\d+\.(?:{_BLOCK_BODY}|adaln_proj\.linear\.(?:weight|bias))"
    rf"|token_refiner\.blocks\.\d+\.(?:{_BLOCK_BODY})"
    r"|token_refiner\.final_norm\.weight"
    r"|condition_proj\.(?:weight|bias)"
    r"|(?:video|audio)_patch_proj\.(?:weight|bias)"
    r"|final_layer\.(?:norm\.weight|adaln_proj\.linear\.(?:weight|bias)|(?:video|audio)_out\.(?:weight|bias))"
    r"|rope\.inv_freq"
    r"|adaln_t_table"
    r")$"
)

#: Tensors every block must carry (relative to ``blocks.N.``); the refiner
#: blocks carry the same minus the AdaLN projection.
_BLOCK_REQUIRED = (
    "attn.qkv_proj.weight", "attn.out_proj.weight", "attn.q_norm.weight", "attn.k_norm.weight",
    "mlp.fc1.weight", "mlp.fc2.weight", "norm1.weight", "norm2.weight", "adaln_proj.linear.weight",
)
_REFINER_REQUIRED = _BLOCK_REQUIRED[:-1]
_DIT_REQUIRED = (
    "video_patch_proj.weight", "audio_patch_proj.weight", "condition_proj.weight",
    "token_refiner.final_norm.weight", "final_layer.norm.weight", "final_layer.adaln_proj.linear.weight",
    "final_layer.video_out.weight", "final_layer.audio_out.weight", "rope.inv_freq",
)


def _read_header(path) -> Header:
    try:
        return read_header(path)
    except QuantFormatError as exc:
        raise H3FormatError(str(exc)) from exc


def _indices(names: Iterable[str], head: str) -> list[int]:
    """Sorted distinct N of ``{head}N.`` names."""
    pat = re.compile(rf"^{re.escape(head)}(\d+)\.")
    found = set()
    for n in names:
        m = pat.match(n)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def _count_contiguous(names: set[str], head: str, required: tuple[str, ...], what: str) -> int:
    idx = _indices(names, head)
    if not idx:
        raise H3FormatError(_NG + f"{what}（{head}N.）が 1 つもありません")
    if idx != list(range(len(idx))):
        raise H3FormatError(_NG + f"{what}の番号が 0 から連続していません（{idx[:3]}…{idx[-3:]}）")
    for i in idx:
        missing = [r for r in required if f"{head}{i}.{r}" not in names]
        if missing:
            raise H3FormatError(_NG + f"{what} {i} に {missing[0]} がありません")
    return len(idx)


def _shape(t: dict[str, TensorInfo], name: str) -> tuple[int, ...]:
    return tuple(t[name].shape)


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise H3FormatError(_NG + "テンソルの形が矛盾しています: " + msg)


def _quant_layers(
    path, header: Header, t: dict[str, TensorInfo], full_name: Callable[[str], str]
) -> dict[str, QuantSpec]:
    """Every layer whose ``.weight`` is I8 -> its spec, with the scale and marker checked.

    ``t`` is keyed by the (prefix-stripped) names; ``full_name`` maps such a
    name back to the key in the file. Reads only the ``comfy_quant`` bytes,
    through ONE file handle (seek + read; never a memory map).
    """
    layers = sorted(
        n[: -len(".weight")] for n, info in t.items() if n.endswith(".weight") and info.dtype == "I8"
    )
    quantized = set(layers)
    stray = sorted(
        n for n in t
        if n.endswith((".weight_scale", ".comfy_quant"))
        and n.rsplit(".", 1)[0] not in quantized
        and f"{n.rsplit('.', 1)[0]}.weight" in t
    )
    if stray:
        raise H3FormatError(
            _NG + f"'{stray[0]}' がありますが、その層の重みは int8 ではありません"
            "（int8_tensorwise 以外の量子化には対応していません）"
        )
    specs: dict[str, QuantSpec] = {}
    if not layers:
        return specs
    with Path(path).open("rb") as fh:
        for layer in layers:
            w = _shape(t, f"{layer}.weight")
            scale, marker = f"{layer}.weight_scale", f"{layer}.comfy_quant"
            if scale not in t:
                raise H3FormatError(_NG + f"int8 の層 '{layer}' に weight_scale がありません")
            if marker not in t:
                old = (
                    "（旧式の __metadata__._quantization_metadata には対応していません）"
                    if "_quantization_metadata" in header.metadata else ""
                )
                raise H3FormatError(_NG + f"int8 の層 '{layer}' に量子化の印（comfy_quant）がありません{old}")
            sinfo = t[scale]
            _expect(len(w) == 2, f"'{layer}.weight' は 2 次元であるべきです（{list(w)}）")
            _expect(
                sinfo.dtype == "F32" and tuple(sinfo.shape) == (w[0], 1),
                f"'{scale}' は F32 [{w[0]}, 1] であるべきです（{sinfo.dtype} {list(sinfo.shape)}）",
            )
            begin, end = header.tensors[full_name(marker)].data_offsets
            fh.seek(header.data_base + begin)
            raw = fh.read(end - begin)
            if len(raw) != end - begin:
                raise H3FormatError(_NG + f"'{marker}' の読み取りが途中で終わりました")
            try:
                specs[layer] = parse_comfy_quant(raw)
            except H3FormatError as exc:
                raise H3FormatError(f"{exc}（層 '{layer}'）") from exc
    return specs


def inspect_h3_dit(path: str) -> tuple[Header, H3DiTConfig]:
    """Header + derived configuration of an H3 DiT file. Refusals are H3FormatError."""
    header = _read_header(path)
    names = set(header.tensors)
    if not is_h3_dit_header(header):
        if any("transformer_blocks.0." in n for n in names):
            raise H3FormatError(_NG + "LTX の DiT です（MiniMax H3 の DiT ではありません）")
        raise H3FormatError(
            _NG + "video_patch_proj.weight と audio_patch_proj.weight がそろっていません"
            "（MiniMax H3 の DiT ではありません）"
        )
    prefix = detect_h3_prefix(names)
    t = {n[len(prefix):]: info for n, info in header.tensors.items() if n.startswith(prefix)}
    outside = sorted(n for n in names if not n.startswith(prefix))

    if "adaln_t_table" not in t:
        if "time_embedder.proj_in.weight" in t:
            raise H3FormatError(
                _NG + "枝刈りしていない版（time_embedder つき）です。この版はまだ使えません。"
                "枝刈り版（adaln_t_table を持つもの。ファイル名に pruned を含むもの）を使ってください"
            )
        raise H3FormatError(_NG + "adaln_t_table がありません（枝刈り版の DiT ではありません）")

    missing = [r for r in _DIT_REQUIRED if r not in t]
    if missing:
        raise H3FormatError(_NG + f"必要なテンソル {missing[0]} がありません")
    stripped = set(t)
    num_layers = _count_contiguous(stripped, "blocks.", _BLOCK_REQUIRED, "本体のブロック")
    refiner_layers = _count_contiguous(stripped, "token_refiner.blocks.", _REFINER_REQUIRED, "token_refiner のブロック")

    vp = _shape(t, "video_patch_proj.weight")
    ap = _shape(t, "audio_patch_proj.weight")
    _expect(len(vp) == 2 and len(ap) == 2, "patch_proj の重みは 2 次元であるべきです")
    hidden = vp[0]
    head_dim = _shape(t, "blocks.0.attn.q_norm.weight")[0]
    qkv = _shape(t, "blocks.0.attn.qkv_proj.weight")
    _expect(head_dim > 0 and qkv[0] % (3 * head_dim) == 0,
            f"qkv_proj の出力 {qkv[0]} が 3×head_dim（{head_dim}）で割り切れません")
    num_heads = qkv[0] // (3 * head_dim)
    fc1 = _shape(t, "blocks.0.mlp.fc1.weight")
    _expect(fc1[0] % 2 == 0, f"fc1 の出力 {fc1[0]} が偶数ではありません")
    ffn = fc1[0] // 2
    cond = _shape(t, "condition_proj.weight")
    _expect(len(cond) == 2, "condition_proj.weight は 2 次元であるべきです")
    text_dim = cond[1]
    vout = _shape(t, "final_layer.video_out.weight")
    aout = _shape(t, "final_layer.audio_out.weight")
    _expect(len(vout) == 2 and len(aout) == 2, "final_layer の出力 2 本の重みは 2 次元であるべきです")
    _expect(vout[0] % 4 == 0, f"video_out の出力 {vout[0]} が 4（patch 1×2×2）で割り切れません")
    latents_dim = vout[0] // 4
    audio_dim = aout[0]
    table = _shape(t, "adaln_t_table")
    _expect(len(table) == 2 and table[0] >= 2 and table[1] >= 1, f"adaln_t_table の形 {list(table)} が不正です")
    grid, rank = table
    inv = _shape(t, "rope.inv_freq")
    _expect(len(inv) == 1 and inv[0] >= 1, f"rope.inv_freq の形 {list(inv)} が不正です")

    inner = num_heads * head_dim
    _expect(vp == (hidden, latents_dim * 4),
            f"video_patch_proj.weight {list(vp)} ≠ [{hidden}, {latents_dim * 4}]"
            "（video_out と食い違っています。蒸留ヘッドの束などには対応していません）")
    _expect(ap == (hidden, audio_dim), f"audio_patch_proj.weight {list(ap)} ≠ [{hidden}, {audio_dim}]")
    _expect(cond[0] == hidden, f"condition_proj.weight {list(cond)} の出力が {hidden} ではありません")
    _expect(vout[1] == hidden and aout[1] == hidden, "final_layer の出力 2 本の入力幅が hidden_size と一致しません")
    fl = _shape(t, "final_layer.adaln_proj.linear.weight")
    _expect(fl == (2 * hidden, rank), f"final_layer.adaln_proj.linear.weight {list(fl)} ≠ [{2 * hidden}, {rank}]")
    expect_block = {
        "attn.qkv_proj.weight": (3 * inner, hidden),
        "attn.out_proj.weight": (hidden, inner),
        "attn.q_norm.weight": (head_dim,),
        "attn.k_norm.weight": (head_dim,),
        "mlp.fc1.weight": (2 * ffn, hidden),
        "mlp.fc2.weight": (hidden, ffn),
        "norm1.weight": (hidden,),
        "norm2.weight": (hidden,),
        # 6 modulation vectors x 3 modalities (video / text / audio), R2 B.
        "adaln_proj.linear.weight": (18 * hidden, rank),
    }
    for i in range(num_layers):
        for leaf, shp in expect_block.items():
            got = _shape(t, f"blocks.{i}.{leaf}")
            _expect(got == shp, f"blocks.{i}.{leaf} {list(got)} ≠ {list(shp)}")
    for i in range(refiner_layers):
        for leaf in _REFINER_REQUIRED:
            got = _shape(t, f"token_refiner.blocks.{i}.{leaf}")
            shp = expect_block[leaf]
            _expect(got == shp, f"token_refiner.blocks.{i}.{leaf} {list(got)} ≠ {list(shp)}")

    quant = _quant_layers(path, header, t, lambda n: prefix + n)
    quant_aux = {f"{layer}.{leaf}" for layer in quant for leaf in ("weight_scale", "comfy_quant")}
    unknown = sorted(n for n in t if not (_DIT_KNOWN.match(n) or n in quant_aux))
    cfg = H3DiTConfig(
        prefix=prefix,
        hidden_size=hidden,
        num_layers=num_layers,
        token_refiner_layers=refiner_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        ffn_hidden=ffn,
        text_dim=text_dim,
        latents_dim=latents_dim,
        audio_latents_dim=audio_dim,
        pruned=True,
        adaln_grid=grid,
        adaln_rank=rank,
        rope_inv_freq_len=inv[0],
        quant=quant,
        unknown=tuple(unknown + outside),
    )
    return header, cfg


# --------------------------------------------------------------------------- #
# Text encoder (Comfy-Org layout: model.layers.N / model.embed_tokens / visual.*)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class H3TEConfig:
    num_layers: int             # number of model.layers.N (50 in the Comfy-Org file)
    quant: dict[str, QuantSpec]  # "model.layers.0.self_attn.q_proj" -> spec
    has_visual: bool            # visual.* present (the I2V vision tower; unused in T2V)
    metadata: dict              # __metadata__, JSON-parsed (minimax_h3_te ...)


#: Qwen3-VL-32B text-tower widths (DiffSynth minimax_h3_text_encoder.py:20-72).
TE_HIDDEN = 5120
TE_Q_OUT = 64 * 128
TE_KV_OUT = 8 * 128
TE_INTERMEDIATE = 25600
#: H3 reads the hidden state after decoder layer 50 (before the final norm).
TE_REQUIRED_LAYERS = 50
_TE_LAYER_REQUIRED = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "input_layernorm.weight", "post_attention_layernorm.weight",
)


def inspect_h3_te(path: str) -> tuple[Header, H3TEConfig]:
    """Header + summary of a Comfy-Org layout Qwen3-VL text encoder for H3."""
    header = _read_header(path)
    t = dict(header.tensors)
    names = set(t)
    if "model.embed_tokens.weight" not in names:
        raise H3FormatError(
            _NG + "model.embed_tokens.weight がありません"
            "（Comfy-Org 形式の Qwen3-VL テキストエンコーダではありません）"
        )
    num_layers = _count_contiguous(names, "model.layers.", _TE_LAYER_REQUIRED, "テキストエンコーダの層")
    if num_layers < TE_REQUIRED_LAYERS:
        raise H3FormatError(
            _NG + f"テキストエンコーダの層が {num_layers} しかありません"
            "（H3 は 50 層目の出力を使うので 50 層以上が要ります）"
        )
    emb = _shape(t, "model.embed_tokens.weight")
    _expect(len(emb) == 2 and emb[1] == TE_HIDDEN,
            f"model.embed_tokens.weight {list(emb)} の幅が {TE_HIDDEN} ではありません")
    expect = {
        "self_attn.q_proj.weight": (TE_Q_OUT, TE_HIDDEN),
        "self_attn.k_proj.weight": (TE_KV_OUT, TE_HIDDEN),
        "self_attn.v_proj.weight": (TE_KV_OUT, TE_HIDDEN),
        "self_attn.o_proj.weight": (TE_HIDDEN, TE_Q_OUT),
        "mlp.gate_proj.weight": (TE_INTERMEDIATE, TE_HIDDEN),
        "mlp.up_proj.weight": (TE_INTERMEDIATE, TE_HIDDEN),
        "mlp.down_proj.weight": (TE_HIDDEN, TE_INTERMEDIATE),
    }
    for i in range(TE_REQUIRED_LAYERS):
        for leaf, shp in expect.items():
            got = _shape(t, f"model.layers.{i}.{leaf}")
            _expect(got == shp, f"model.layers.{i}.{leaf} {list(got)} ≠ {list(shp)}")
    quant = _quant_layers(path, header, t, lambda n: n)
    cfg = H3TEConfig(
        num_layers=num_layers,
        quant=quant,
        has_visual=any(n.startswith("visual.") for n in names),
        metadata=sft_quant_format.parse_metadata(header),
    )
    return header, cfg
