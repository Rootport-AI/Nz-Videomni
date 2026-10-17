r"""Gate G4 (text-encoder part, REVIEW_code1 M3): the H3 text encoder's numbers (GPU).

    .venv-engine-h3\Scripts\python.exe outputs\h3-gate1\g4_te_compare.py [--report outputs\h3-gate1\g4_te_result.json]

(a) layers 0, 1 and 49 of the real Comfy-Org TE, one at a time:
      mine  = the layer as engine_h3 builds it (meta -> swap_in_int8_linears ->
              load_streaming with te_key_map), H3Int8Linear (W8A8, comfy_kitchen CUDA);
      deq   = the same layer with every H3Int8Linear replaced by an nn.Linear
              holding the weight DEQUANTIZED to bf16 (engine/sft_quant/dequant.py, W8A16);
      ref   = an independently built Qwen3VLTextDecoderLayer whose weights are
              assigned straight from the file names (dequantized the same way).
    quant_rel  = |mine - deq| / |deq| on the residual delta (out - x)   (W8A8 vs W8A16; ~1e-2)
    wiring_rel = |deq - ref|  / |ref|                                    (names / mapping; must be 0)
    Input: the G5 prompt's token embeddings (the real first-layer input),
    positions 0..L-1 through the model's own rotary module.
(b) the first N layers (N = 8: what fits on a 16 GB GPU with the embeddings),
    with the layer offload on (window 1, the shipped te_layers_on_gpu=0) and
    off (everything resident): the hidden state after layer N must be
    bit-identical.
(c) the G5 prompt's token ids: AutoTokenizer (the engine's way) vs a
    PreTrainedTokenizerFast built from tokenizer.json directly, with the
    additional_special_tokens of tokenizer_config.json (must match); and the
    same without them (recorded: shows what tokenizer.json alone would do).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import h3_format  # noqa: E402
from engine.gemma.layer_offload_service import GemmaLayerOffloadService  # noqa: E402
from engine.sft_quant.dequant import dequantize  # noqa: E402
from engine.sft_quant.sft_reader import read_tensors  # noqa: E402
from engine_h3.quant_linear import H3Int8Linear, require_cuda_int8, swap_in_int8_linears  # noqa: E402
from engine_h3.sft_load import load_streaming, move_module_tree  # noqa: E402
from engine_h3.text_encoder import (  # noqa: E402
    build_text_config,
    ensure_extra_special_tokens,
    te_key_map,
    te_quant_names,
)

TE = ROOT / "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
TOK = ROOT / "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer"
PROMPT = (
    "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in "
    "puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>"
)
DEV = torch.device("cuda")


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def scheme(spec) -> str:
    return "int8_convrot" if spec.convrot else "int8"


def dequant_linears(module: nn.Module) -> int:
    """Replace every H3Int8Linear under ``module`` with a bf16 nn.Linear (dequantized weight)."""
    n = 0
    for name, child in list(module.named_modules()):
        for leaf, sub in list(child.named_children()):
            if isinstance(sub, H3Int8Linear):
                w = dequantize(scheme(sub.spec), sub.weight.data, {"weight_scale": sub.weight_scale}, torch.bfloat16)
                lin = nn.Linear(sub.in_features, sub.out_features, bias=sub.bias is not None,
                                device=w.device, dtype=torch.bfloat16)
                lin.weight.data.copy_(w)
                if sub.bias is not None:
                    lin.bias.data.copy_(sub.bias.data.to(torch.bfloat16))
                setattr(child, leaf, lin)
                n += 1
    return n


def build_model(num_layers: int, layers_to_read: list[int], header, te_cfg):
    """A Qwen3VLTextModel of ``num_layers`` holding the file's layers ``layers_to_read`` (as 0..)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding

    config = build_text_config()
    config.num_hidden_layers = num_layers
    with torch.device("meta"):
        model = Qwen3VLTextModel(config)
    model.norm = nn.Identity()
    model.to(torch.bfloat16)
    inv, sc = Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters(config, device="cpu")
    model.rotary_emb.inv_freq = inv.float()
    model.rotary_emb.original_inv_freq = inv.float().clone()
    model.rotary_emb.attention_scaling = sc
    remap = {src: dst for dst, src in enumerate(layers_to_read)}

    def key_map(name: str):
        mapped = te_key_map(name)
        if mapped is None or not mapped.startswith("layers."):
            return mapped
        idx = int(mapped.split(".")[1])
        if idx not in remap:
            return None
        return f"layers.{remap[idx]}." + mapped.split(".", 2)[2]

    quant = {}
    for layer, spec in te_quant_names(te_cfg.quant).items():
        idx = int(layer.split(".")[1])
        if idx in remap:
            quant[f"layers.{remap[idx]}." + layer.split(".", 2)[2]] = spec
    with torch.device("meta"):
        swap_in_int8_linears(model, quant, bias_dtype_of=lambda n: torch.bfloat16)
    load_streaming(model, str(TE), header, key_map=key_map)
    model.requires_grad_(False).eval()
    return model, config


def reference_layer(config, file_layer: int, te_cfg):
    """An independent Qwen3VLTextDecoderLayer, weights assigned by FILE name (dequantized)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

    layer = Qwen3VLTextDecoderLayer(config, 0).to(DEV, torch.bfloat16).eval().requires_grad_(False)
    head = f"model.layers.{file_layer}."
    keys = [k for k in h3_format.read_header(str(TE)).tensors if k.startswith(head) and not k.endswith(".comfy_quant")]
    got = dict(read_tensors(str(TE), keys))
    for key in keys:
        if key.endswith(".weight_scale"):
            continue
        sub = key[len(head):]
        target = layer.get_submodule(sub.rsplit(".", 1)[0])
        t = got[key].to(DEV)
        lname = key[: -len(".weight")]
        if t.dtype == torch.int8:
            spec = te_cfg.quant[lname]
            t = dequantize(scheme(spec), t, {"weight_scale": got[lname + ".weight_scale"].to(DEV)}, torch.bfloat16)
        getattr(target, sub.rsplit(".", 1)[1]).data.copy_(t.to(torch.bfloat16))
    return layer


def part_a(header, te_cfg, ids: list[int]) -> dict:
    out = {}
    for file_layer in (0, 1, 49):
        model, config = build_model(1, [file_layer], header, te_cfg)
        move_module_tree(model, DEV, skip=set())
        with torch.no_grad():
            x = model.embed_tokens(torch.tensor([ids], device=DEV))
            pos = torch.arange(len(ids), device=DEV)[None]
            pe = model.rotary_emb(x, pos)
            layer = model.layers[0]
            mine = layer(x, attention_mask=None, position_ids=pos, position_embeddings=pe)
            dequant_linears(layer)
            deq = layer(x, attention_mask=None, position_ids=pos, position_embeddings=pe)
            ref = reference_layer(config, file_layer, te_cfg)
            refo = ref(x, attention_mask=None, position_ids=pos, position_embeddings=pe)
        out[f"layer_{file_layer}"] = {
            "quant_rel_delta": rel(mine - x, deq - x),
            "quant_rel_out": rel(mine, deq),
            "wiring_rel": rel(deq, refo),
            "wiring_bit_identical": bool(torch.equal(deq, refo)),
            "finite": bool(torch.isfinite(mine).all()),
        }
        print(file_layer, out[f"layer_{file_layer}"], flush=True)
        del model, layer, ref, mine, deq, refo, x
        gc.collect()
        torch.cuda.empty_cache()
    return out


def part_b(header, te_cfg, ids: list[int], n: int) -> dict:
    results = {}
    for mode in ("resident", "offload_window1"):
        model, _ = build_model(n, list(range(n)), header, te_cfg)
        if mode == "resident":
            move_module_tree(model, DEV, skip=set())
        else:
            skip = {id(m) for layer in model.layers for m in layer.modules()}
            move_module_tree(model, DEV, skip=skip)
            GemmaLayerOffloadService(layers_on_gpu=1, compute_device=DEV).install(model)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            inp = torch.tensor([ids], device=DEV)
            h = model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False).last_hidden_state[0]
        results[mode] = {
            "hidden": h.cpu(),
            "seconds": round(time.perf_counter() - t0, 2),
            "peak_vram_mb": int(torch.cuda.max_memory_allocated() / 2**20),
        }
        del model, h
        gc.collect()
        torch.cuda.empty_cache()
    a, b = results["resident"]["hidden"], results["offload_window1"]["hidden"]
    return {
        "layers": n,
        "bit_identical": bool(torch.equal(a, b)),
        "max_abs_diff": float((a.float() - b.float()).abs().max()),
        **{m: {k: v for k, v in r.items() if k != "hidden"} for m, r in results.items()},
    }


def part_c() -> dict:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    auto = AutoTokenizer.from_pretrained(str(TOK))
    special = ensure_extra_special_tokens(auto)
    ids_auto = auto(PROMPT, add_special_tokens=False)["input_ids"]
    cfg = json.loads((TOK / "tokenizer_config.json").read_text(encoding="utf-8"))
    extra = cfg["additional_special_tokens"]
    fast = PreTrainedTokenizerFast(tokenizer_file=str(TOK / "tokenizer.json"))
    ids_raw = fast(PROMPT, add_special_tokens=False)["input_ids"]
    fast.add_special_tokens({"additional_special_tokens": extra})
    ids_ref = fast(PROMPT, add_special_tokens=False)["input_ids"]
    return {
        "auto_class": type(auto).__name__,
        "extra_special_tokens": special,
        "ids_engine": ids_auto,
        "ids_reference": ids_ref,
        "match": ids_auto == ids_ref,
        "len": len(ids_auto),
        "ids_tokenizer_json_alone": ids_raw,
        "tokenizer_json_alone_matches": ids_raw == ids_auto,
    }, ids_auto


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(ROOT / "outputs/h3-gate1/g4_te_result.json"))
    ap.add_argument("--first-n", type=int, default=8)
    args = ap.parse_args()
    report: dict = {"te": str(TE), "prompt": PROMPT, "backends": None}
    report["backends"] = {k: v.get("available") for k, v in require_cuda_int8().items() if isinstance(v, dict)}
    report["c_tokens"], ids = part_c()
    print("c", report["c_tokens"]["match"], report["c_tokens"]["tokenizer_json_alone_matches"], flush=True)
    header, te_cfg = h3_format.inspect_h3_te(str(TE))
    report["a_layers"] = part_a(header, te_cfg, ids)
    report["b_offload"] = part_b(header, te_cfg, ids, args.first_n)
    print("b", report["b_offload"], flush=True)
    a = report["a_layers"]
    report["verdict"] = {
        "a_quant_ok": all(v["quant_rel_delta"] < 5e-2 for v in a.values()),
        "a_wiring_zero": all(v["wiring_bit_identical"] for v in a.values()),
        "b_bit_identical": report["b_offload"]["bit_identical"],
        "c_tokens_match": report["c_tokens"]["match"],
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["verdict"]))
    return 0 if all(report["verdict"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
