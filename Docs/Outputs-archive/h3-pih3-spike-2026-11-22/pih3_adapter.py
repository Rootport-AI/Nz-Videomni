r"""pi-H3 adapter for the MiniMax H3 spike (2026-11-22): key check against the DiT, and the
non-LoRA part of the adapter applied to a CPU DiT right after ``H3Pipeline._build_dit``.

The owner's ruling on how the difference is applied (not changed here):

* blocks 0-45 LoRA (138 pairs, rank 64, no alpha): the request's ``loras`` at strength 1.0, i.e.
  the product's LoRA path (``engine_h3.lora``) -- nothing in this module;
* blocks 46-49 ``attn.out_proj`` / ``mlp.fc1`` / ``mlp.fc2`` (INT8 ConvRot): the file's ``weight``
  (I8) and ``weight_scale`` (F32 [N, 1]) go into ``H3Int8Linear`` as stored (no re-quantization);
  where the base layer is a BF16 ``nn.Linear`` (Sulphur-3 block 49) the weight is dequantized with
  ``engine.sft_quant.dequant.dequantize`` (float32) and cast to the layer's dtype;
* every block's ``adaln_proj.linear.weight``, blocks 46-49's ``adaln_proj.linear.bias`` and the
  ``final_layer`` tensors: the file's F16 values cast to the engine layer's dtype; an INT8 AdaLN
  (Sulphur-3 blocks 2-48, per-row scale) becomes a float32 ``nn.Linear`` (bias: pi-H3's on 46-48,
  the base's on 2-45).

Every cast is recorded with whether it was exact (round trip back to the file's dtype).
"""

from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn

import h3_format
import sft_quant_format
from engine.sft_quant.dequant import dequantize, normalize_aux
from engine.sft_quant.sft_reader import read_tensors
from engine_h3.quant_linear import H3Int8Linear, H3QuantLinear

PI_DIR = "models/MiniMaxH3/VAE/pi-H3/"
PI_FILES = {
    2: PI_DIR + "pi_h3_2step_dwd_adapter_int8_convrot.safetensors",
    4: PI_DIR + "pi_h3_4step_dwd_adapter_int8_convrot.safetensors",
}
NUM_BLOCKS = 50
LORA_BLOCKS = tuple(range(46))
QUANT_BLOCKS = (46, 47, 48, 49)
LEAVES = ("attn.out_proj", "mlp.fc1", "mlp.fc2")
FINAL_KEYS = tuple("final_layer." + k for k in (
    "norm.weight", "adaln_proj.linear.weight", "adaln_proj.linear.bias",
    "video_out.weight", "video_out.bias", "audio_out.weight", "audio_out.bias"))
EXPECTED = {"lora_pairs": 138, "quant_layers": 12, "adaln_weights": 50, "adaln_biases": 4,
            "final_tensors": 7, "dwd_tensors": 171}
MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}

_LORA_RE = re.compile(r"^(blocks\.(\d+)\.(?:attn\.out_proj|mlp\.fc1|mlp\.fc2))\.lora_([AB])\.weight$")
_QUANT_RE = re.compile(r"^(blocks\.(\d+)\.(?:attn\.out_proj|mlp\.fc1|mlp\.fc2))\.(weight|weight_scale|comfy_quant)$")
_ADALN_RE = re.compile(r"^blocks\.(\d+)\.adaln_proj\.linear\.(weight|bias)$")


@dataclass
class PiKeys:
    lora: dict = field(default_factory=dict)      # module -> {"A": key, "B": key}
    quant: dict = field(default_factory=dict)     # layer -> {leaf: key}
    adaln_w: dict = field(default_factory=dict)   # block -> key
    adaln_b: dict = field(default_factory=dict)
    final: list = field(default_factory=list)
    dwd: list = field(default_factory=list)
    other: list = field(default_factory=list)


def classify(header) -> PiKeys:
    keys = PiKeys()
    for k in sorted(header.tensors):
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


def counts(keys: PiKeys) -> dict:
    return {"lora_pairs": sum(1 for v in keys.lora.values() if set(v) == {"A", "B"}),
            "quant_layers": sum(1 for v in keys.quant.values() if set(v) == {"weight", "weight_scale", "comfy_quant"}),
            "adaln_weights": len(keys.adaln_w), "adaln_biases": len(keys.adaln_b),
            "final_tensors": len(keys.final), "dwd_tensors": len(keys.dwd), "other": len(keys.other)}


def count_problems(keys: PiKeys) -> list[str]:
    c = counts(keys)
    bad = [f"{k}: {c[k]} != {v}" for k, v in EXPECTED.items() if c[k] != v]
    if keys.other:
        bad.append(f"unclassified keys: {keys.other[:5]}")
    want_lora = {f"blocks.{i}.{leaf}" for i in LORA_BLOCKS for leaf in LEAVES}
    want_quant = {f"blocks.{i}.{leaf}" for i in QUANT_BLOCKS for leaf in LEAVES}
    if set(keys.lora) != want_lora:
        bad.append("LoRA modules are not blocks 0-45 x out_proj/fc1/fc2")
    if set(keys.quant) != want_quant:
        bad.append("quantized layers are not blocks 46-49 x out_proj/fc1/fc2")
    if set(keys.adaln_w) != set(range(NUM_BLOCKS)) or set(keys.adaln_b) != set(QUANT_BLOCKS):
        bad.append("AdaLN weights are not blocks 0-49 / biases not 46-49")
    return bad


def read_markers(path, header, keys: PiKeys) -> dict[str, dict]:
    """Quantized layer -> its comfy_quant JSON (only these tiny tensors are read)."""
    return {layer: json.loads(sft_quant_format.read_tensor_bytes(str(path), header, leaves["comfy_quant"]).decode("utf-8"))
            for layer, leaves in sorted(keys.quant.items())}


def marker_problems(markers: dict[str, dict]) -> list[str]:
    bad = []
    for layer, doc in markers.items():
        flat = {**(doc.get("params") or {}), **doc}
        got = {k: flat.get(k) for k in MARKER}
        if got != MARKER:
            bad.append(f"{layer}: {got}")
    return bad


# --------------------------------------------------------------------------- #
# matching against a (meta-built) DiT and the base file
# --------------------------------------------------------------------------- #

def _base_dtype(base_header, prefix: str, key: str) -> str | None:
    info = base_header.tensors.get(prefix + key)
    return None if info is None else info.dtype


def model_match(keys: PiKeys, pi_header, dit: nn.Module, base_header, cfg) -> dict:
    """Every non-dwd pi-H3 key against the DiT's modules / state dict by name and shape."""
    sd = dit.state_dict(keep_vars=True)
    mods = dict(dit.named_modules())
    shape = {k: tuple(v.shape) for k, v in pi_header.tensors.items()}
    cat = collections.Counter()
    bad: list[str] = []
    seen_targets: set[str] = set()

    def target(name: str) -> None:
        if name in seen_targets:
            bad.append(f"two pi-H3 keys aim at {name}")
        seen_targets.add(name)

    for m, ab in sorted(keys.lora.items()):
        mod = mods.get(m)
        if not isinstance(mod, (H3QuantLinear, nn.Linear)):
            bad.append(f"LoRA {m}: no Linear in the model ({type(mod).__name__})")
            continue
        a, b = shape[ab["A"]], shape[ab["B"]]
        if not (len(a) == 2 and len(b) == 2 and a[0] == b[1] and a[1] == mod.in_features and b[0] == mod.out_features):
            bad.append(f"LoRA {m}: A {list(a)} B {list(b)} vs in {mod.in_features} out {mod.out_features}")
        target(m + "@lora")
        cat[f"lora_on_{type(mod).__name__}"] += 1
    for layer, leaves in sorted(keys.quant.items()):
        mod = mods.get(layer)
        w = shape[leaves["weight"]]
        if isinstance(mod, H3Int8Linear):
            for leaf in ("weight", "weight_scale"):
                k = f"{layer}.{leaf}"
                if k not in sd or tuple(sd[k].shape) != shape[leaves[leaf]]:
                    bad.append(f"{k}: model {None if k not in sd else list(sd[k].shape)} file {list(shape[leaves[leaf]])}")
                target(k)
            cat["quant_into_H3Int8Linear"] += 1
        elif isinstance(mod, nn.Linear):
            if (mod.out_features, mod.in_features) != w or shape[leaves["weight_scale"]] != (w[0], 1):
                bad.append(f"{layer}: nn.Linear {mod.out_features}x{mod.in_features} file {list(w)}")
            target(layer + ".weight")
            cat[f"quant_dequantized_into_nn.Linear_{str(mod.weight.dtype).replace('torch.', '')}"] += 1
        else:
            bad.append(f"{layer}: not a Linear in the model ({type(mod).__name__})")
    for i, k in sorted(keys.adaln_w.items()):
        lin = mods.get(f"blocks.{i}.adaln_proj.linear")
        if k not in sd or tuple(sd[k].shape) != shape[k]:
            bad.append(f"{k}: model {None if k not in sd else list(sd[k].shape)} file {list(shape[k])}")
        target(k)
        cat["adaln_w_replace_int8_with_float_linear" if isinstance(lin, H3Int8Linear) else "adaln_w_set_float"] += 1
    for i, k in sorted(keys.adaln_b.items()):
        if k not in sd or tuple(sd[k].shape) != shape[k]:
            bad.append(f"{k}: model {None if k not in sd else list(sd[k].shape)} file {list(shape[k])}")
        target(k)
        cat["adaln_b"] += 1
    for k in keys.final:
        if k not in sd or tuple(sd[k].shape) != shape[k]:
            bad.append(f"{k}: model {None if k not in sd else list(sd[k].shape)} file {list(shape[k])}")
        target(k)
        cat["final"] += 1
    # base dtypes of what is replaced (from the base file's header)
    base = collections.Counter()
    for layer in keys.quant:
        base[f"quant weight {_base_dtype(base_header, cfg.prefix, layer + '.weight')}"] += 1
    for k in list(keys.adaln_w.values()) + list(keys.adaln_b.values()) + keys.final:
        base[f"{k.rsplit('.', 1)[-1]} {'adaln' if k.startswith('blocks.') else k.rsplit('.', 1)[0]} "
             f"{_base_dtype(base_header, cfg.prefix, k)} <- pi {pi_header.tensors[k].dtype}"] += 1
    return {"problems": bad, "categories": dict(cat), "replaced_base_dtypes": dict(base),
            "non_dwd_keys": len(pi_header.tensors) - len(keys.dwd), "passed": not bad}


def dtype_table(base_header, cfg) -> dict:
    """The base file's dtypes per group (block lists), and its INT8 layers' ConvRot flags."""
    p = cfg.prefix
    tab: dict[str, dict[str, list[int]]] = {}
    for leaf in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2", "adaln_proj.linear"):
        groups: dict[str, list[int]] = {}
        for i in range(cfg.num_layers):
            dt = base_header.tensors[f"{p}blocks.{i}.{leaf}.weight"].dtype
            spec = cfg.quant.get(f"blocks.{i}.{leaf}")
            if spec is not None:
                dt += "_convrot" if spec.convrot else "_rowwise"
            groups.setdefault(dt, []).append(i)
        tab[leaf] = groups
    tab["output"] = {k: base_header.tensors[p + k].dtype for k in
                     ("final_layer.video_out.weight", "final_layer.audio_out.weight")}
    return tab


def dtype_expectation_problems(kind: str, tab: dict) -> list[str]:
    allb = list(range(NUM_BLOCKS))
    if kind == "fl2va":
        want = {leaf: {"I8_convrot": allb} for leaf in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")}
        want["adaln_proj.linear"] = {"F16": allb}
        want_out = "F32"
    else:
        keep, mid = [0, 1, 49], list(range(2, 49))
        want = {leaf: {"BF16": keep, "I8_convrot": mid} for leaf in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")}
        want["adaln_proj.linear"] = {"BF16": keep, "I8_rowwise": mid}
        want_out = "BF16"
    bad = [f"{leaf}: {tab[leaf]} != {w}" for leaf, w in want.items()
           if {k: sorted(v) for k, v in tab[leaf].items()} != w]
    bad += [f"{k}: {v} != {want_out}" for k, v in tab["output"].items() if v != want_out]
    return bad


# --------------------------------------------------------------------------- #
# applying to a CPU DiT (after _build_dit)
# --------------------------------------------------------------------------- #

def _cast(value: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, dict]:
    out = value.to(dtype).contiguous()
    exact = bool(torch.equal(out.to(value.dtype), value))
    rec = {"from": str(value.dtype).replace("torch.", ""), "to": str(dtype).replace("torch.", ""), "exact": exact}
    if not exact:
        rec["changed_elements"] = int((out.to(value.dtype) != value).sum())
        rec["max_abs_change"] = float((out.to(torch.float64) - value.to(torch.float64)).abs().max())
    return out, rec


def _set_param(module: nn.Module, leaf: str, value: torch.Tensor) -> None:
    setattr(module, leaf, nn.Parameter(value, requires_grad=False))


def apply_adapter(dit: nn.Module, path, header=None, log=print) -> dict:
    """Apply the non-LoRA, non-DWD part of the pi-H3 file to ``dit`` (all on the CPU)."""
    header = header or sft_quant_format.read_header(str(path))
    keys = classify(header)
    problems = count_problems(keys)
    if problems:
        raise RuntimeError(f"pi-H3: unexpected key set: {problems}")
    names = [k for leaves in keys.quant.values() for k in leaves.values()]
    names += list(keys.adaln_w.values()) + list(keys.adaln_b.values()) + keys.final
    t = dict(read_tensors(str(path), names, header=header))
    conv = collections.Counter()
    rounding: list[dict] = []
    rep: dict = {"quant_layers": {}, "adaln": collections.Counter(), "final": {}}

    def note(where: str, rec: dict) -> None:
        conv[f"{where}: {rec['from']}->{rec['to']}{'' if rec['exact'] else ' (rounded)'}"] += 1
        if not rec["exact"]:
            rounding.append({"tensor": where, **rec})

    for layer, leaves in sorted(keys.quant.items()):
        mod = dit.get_submodule(layer)
        w, s = t[leaves["weight"]], t[leaves["weight_scale"]]
        spec = h3_format.parse_comfy_quant(bytes(t[leaves["comfy_quant"]].numpy()), accepted=h3_format.DIT_FORMATS)
        if isinstance(mod, H3Int8Linear):
            if tuple(mod.weight.shape) != tuple(w.shape) or w.dtype != torch.int8 or s.dtype != torch.float32:
                raise RuntimeError(f"pi-H3 {layer}: {w.dtype}{list(w.shape)} / {s.dtype} vs {list(mod.weight.shape)}")
            spec_changed = mod.spec != spec
            _set_param(mod, "weight", w.contiguous())
            mod.weight_scale = s.reshape(-1, 1).contiguous()
            mod.spec = spec
            rep["quant_layers"][layer] = {"into": "H3Int8Linear", "as_stored": "I8 weight + F32 weight_scale",
                                          "spec": f"{spec.format}/convrot={spec.convrot}/g{spec.convrot_groupsize}",
                                          "spec_changed": spec_changed}
            conv["quant: I8+F32 -> H3Int8Linear (as stored)"] += 1
        elif isinstance(mod, nn.Linear):
            o, i = (int(v) for v in w.shape)
            if (mod.out_features, mod.in_features) != (o, i):
                raise RuntimeError(f"pi-H3 {layer}: {list(w.shape)} vs nn.Linear {mod.out_features}x{mod.in_features}")
            scheme = "int8_convrot" if spec.convrot else "int8"
            aux = {"weight_scale": normalize_aux(scheme, "weight_scale", s, o, i)}
            deq = dequantize(scheme, w, aux, torch.float32)
            value, rec = _cast(deq, mod.weight.dtype)
            _set_param(mod, "weight", value)
            rep["quant_layers"][layer] = {"into": f"nn.Linear {rec['to']}", "dequantized": scheme, **rec}
            note(f"quant dequantized (f32) -> {layer}", rec)
        else:
            raise RuntimeError(f"pi-H3 {layer}: {type(mod).__name__} is not a Linear")
    for i in range(NUM_BLOCKS):
        proj = dit.blocks[i].adaln_proj
        lin = proj.linear
        w = t[keys.adaln_w[i]]
        b_pi = t[keys.adaln_b[i]] if i in keys.adaln_b else None
        if isinstance(lin, H3Int8Linear):
            bias_src = b_pi if b_pi is not None else lin.bias.detach()
            new = nn.Linear(lin.in_features, lin.out_features, bias=True, dtype=torch.float32, device="cpu")
            wv, wr = _cast(w, torch.float32)
            bv, br = _cast(bias_src, torch.float32)
            _set_param(new, "weight", wv)
            _set_param(new, "bias", bv)
            new.requires_grad_(False)
            proj.linear = new
            rep["adaln"]["int8_rowwise -> float32 nn.Linear"] += 1
            note("adaln weight (into a new float32 nn.Linear)", wr)
            note(f"adaln bias ({'pi-H3' if b_pi is not None else 'base'}) (into the new nn.Linear)", br)
        elif isinstance(lin, nn.Linear):
            wv, wr = _cast(w, lin.weight.dtype)
            _set_param(lin, "weight", wv)
            note("adaln weight", wr)
            rep["adaln"][f"set into nn.Linear {wr['to']}"] += 1
            if b_pi is not None:
                bv, br = _cast(b_pi, lin.bias.dtype)
                _set_param(lin, "bias", bv)
                note("adaln bias (pi-H3)", br)
        else:
            raise RuntimeError(f"pi-H3 blocks.{i}.adaln_proj.linear: {type(lin).__name__}")
    for k in keys.final:
        parent, _, leaf = k.rpartition(".")
        mod = dit.get_submodule(parent)
        old = getattr(mod, leaf)
        value, rec = _cast(t[k], old.dtype)
        if tuple(value.shape) != tuple(old.shape):
            raise RuntimeError(f"pi-H3 {k}: {list(value.shape)} vs {list(old.shape)}")
        _set_param(mod, leaf, value)
        rep["final"][k] = rec
        note(k, rec)
    del t
    rep["adaln"] = dict(rep["adaln"])
    rep["counts"] = {"quant_layers": len(rep["quant_layers"]), "adaln_weights": NUM_BLOCKS,
                     "adaln_biases": len(keys.adaln_b), "final_tensors": len(rep["final"])}
    rep["conversions"] = dict(conv)
    rep["rounding"] = rounding
    rep["not_applied_here"] = {"lora_pairs (product LoRA path)": counts(keys)["lora_pairs"],
                               "dwd_tensors (the DWD head)": len(keys.dwd)}
    log(f"pi-H3 adapter: {rep['counts']} rounding={len(rounding)}")
    return rep
