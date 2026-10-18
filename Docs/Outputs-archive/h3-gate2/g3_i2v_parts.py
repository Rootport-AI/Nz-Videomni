r"""Gate 2a-G3: the I2V parts against DiffSynth-Studio (INTERFACE2 §6). GPU, small.

    set PYTHONPATH=<repo root>
    .venv-engine-h3\Scripts\python.exe outputs\h3-gate2\g3_i2v_parts.py [--image P] [--report P]

The oracle is the verbatim DiffSynth copies in ``outputs/h3-gate2/oracle/diffsynth_h3/``
(commit 4e34969, Apache-2.0): ``models/minimax_h3_text_encoder.py``,
``models/minimax_h3_video_vae.py`` and the coordinate half of the pipeline
(``pipeline_coords.py``, copied from gate 1). Both sides get the SAME canvas
image (cover + centre crop of ``--image`` onto 640 x 384).

A  token ids and per-token tags: ours (H3TextEncoder.encode's ids/tags) vs DiffSynth
   ``image_token_counts`` + ``presentation_fl2va`` -> must be identical.
B  vision inputs: pixel_values / image_grid_thw from both -> identical; the vision
   tower's output (get_image_features, same bf16 weights) -> identical.
C  the text encoder output [L, 5120]:
     ours      = H3TextEncoder (Qwen3VLModel, int8 W8A8 layers read directly)
     ds_wiring = DiffSynth MiniMaxH3TextEncoder.forward with the SAME int8 layers
                 (H3Int8Linear, read the same way)          -> wiring, must be 0
     ds_bf16   = DiffSynth MiniMaxH3TextEncoder.forward with every int8 layer
                 dequantized to bf16 nn.Linear (W8A16)       -> the G4-TE level
D  VAE encode of the key frame: ours bf16 vs DiffSynth bf16 (bit-identical expected;
   if not, max abs diff, bound 1e-2 order); ours f32 vs ours bf16 (recorded).
E  the packed layout: condition-row order, AdaLN tags, timestep numbering for the
   20 steps and the RoPE coordinates vs DiffSynth, float64 exact.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "oracle"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import h3_format  # noqa: E402
import sft_quant_format  # noqa: E402
from engine.sft_quant.dequant import dequantize  # noqa: E402
from engine.sft_quant.sft_reader import read_tensors  # noqa: E402
from engine_h3.constants import KEYFRAME_NOISE_AUG, audio_latents, latent_frames  # noqa: E402
from engine_h3.image_cond import canvas_to_tensor, load_canvas_image  # noqa: E402
from engine_h3.quant_linear import require_cuda_int8, swap_in_int8_linears  # noqa: E402
from engine_h3.scheduler import sigma_schedule  # noqa: E402
from engine_h3.sft_load import drop_to_meta, load_mapped_to, move_module_tree  # noqa: E402
from engine_h3.text_encoder import (  # noqa: E402
    H3TextEncoder,
    install_layer_read_hooks,
    te_quant_names,
    te_read_plan,
    vision_inv_freq,
)

TE = ROOT / "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
TOK = ROOT / "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer"
VAE = ROOT / "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors"
PROMPT = (
    "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in "
    "puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>"
)
W, H, F = 640, 384, 124
DEV = torch.device("cuda")


def rel(a, b) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def free():
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# DiffSynth text encoder, built on meta and read layer by layer
# --------------------------------------------------------------------------- #

def ds_text_encoder():
    from diffsynth_h3.models.minimax_h3_text_encoder import MiniMaxH3TextEncoder
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    with torch.device("meta"):
        ds = MiniMaxH3TextEncoder(50)
    ds.to(torch.bfloat16)
    ds.requires_grad_(False)
    ds.eval()
    for cfg in (ds.config, ds.config.text_config, ds.config.vision_config):
        cfg._attn_implementation = "sdpa"
    m = ds.model
    rot = m.language_model.rotary_emb
    inv, sc = Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters(ds.config.text_config, device="cpu")
    rot.inv_freq, rot.original_inv_freq, rot.attention_scaling = inv.float(), inv.float().clone(), sc
    vrot = m.visual.rotary_pos_emb
    vrot.inv_freq = vision_inv_freq(vrot.dim)
    return ds


def run_ds(ds, header, plans, ids, pixel_values, grid, mode, quant):
    """mode 'int8': our H3Int8Linear in DiffSynth's model; 'bf16': dequantized nn.Linear."""
    embed_plan, layer_plans, visual_plan = plans
    m = ds.model
    lm = m.language_model
    if mode == "int8":
        with torch.device("meta"):
            swap_in_int8_linears(m, te_quant_names(quant), bias_dtype_of=lambda n: torch.bfloat16)
    load_mapped_to(lm.embed_tokens, str(TE), header, embed_plan, DEV)
    load_mapped_to(m.visual, str(TE), header, visual_plan, DEV)
    move_module_tree(m.visual.rotary_pos_emb, DEV, skip=set())
    move_module_tree(lm.rotary_emb, DEV, skip=set())
    stats = {"bytes": 0, "read_seconds": 0.0, "layers": 0}
    if mode == "int8":
        handles = install_layer_read_hooks(lm.layers, layer_plans, str(TE), header, DEV, stats)
    else:
        handles = []
        specs = {k: v for k, v in quant.items()}
        for idx, layer in enumerate(lm.layers):
            plan = layer_plans[idx]

            def pre(module, args, kwargs, _plan=plan, _idx=idx):
                raw = dict((_plan[k], t.to(DEV)) for k, t in read_tensors(str(TE), list(_plan), header))
                sd = {}
                for name, t in raw.items():
                    if name.endswith(".weight_scale"):
                        continue
                    lin = name[: -len(".weight")] if name.endswith(".weight") else None
                    spec = specs.get(f"model.layers.{_idx}.{lin}") if lin else None
                    if spec is not None:
                        scheme = "int8_convrot" if spec.convrot else "int8"
                        if spec.convrot and int(spec.convrot_groupsize) != 256:
                            raise RuntimeError("dequant oracle supports groupsize 256 only")
                        t = dequantize(scheme, t, {"weight_scale": raw[f"{lin}.weight_scale"]}, torch.bfloat16)
                    sd[name] = t
                res = module.load_state_dict(sd, strict=True, assign=True)
                stats["layers"] += 1

            def post(module, args, kwargs, output):
                drop_to_meta(module)

            handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(layer.register_forward_hook(post, with_kwargs=True))
    try:
        input_ids = torch.tensor([ids], dtype=torch.long, device=DEV)
        with torch.no_grad():
            out = ds(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                     pixel_values=pixel_values.to(DEV, torch.bfloat16), image_grid_thw=grid.to(DEV, torch.long))
        out = out.detach().to(torch.bfloat16).cpu()
    finally:
        for h in handles:
            h.remove()
        drop_to_meta(lm.embed_tokens)
        drop_to_meta(m.visual)
        for layer in lm.layers:
            drop_to_meta(layer)
    assert stats["layers"] == 50, stats
    return out


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(HERE / "i2v_input.png"))
    ap.add_argument("--report", default=str(HERE / "g3_i2v_parts.json"))
    args = ap.parse_args()
    t_start = time.perf_counter()
    require_cuda_int8()
    rec: dict = {"image": args.image, "canvas": [W, H], "prompt": PROMPT}
    canvas, src = load_canvas_image(args.image, W, H)
    rec["source_size"] = list(src)

    # ---- A/B/C ours -------------------------------------------------------
    te = H3TextEncoder(str(TE), str(TOK), device=DEV)
    te.load()
    pv_ours, grid_ours, ntok_ours = te.vision_inputs(canvas)
    hidden_ours, tags_ours = te.encode(PROMPT, image=canvas)
    hidden_ours = hidden_ours.cpu()
    tags_ours = tags_ours.cpu()
    ids_ours = list(te.last_ids)
    plans = (te._embed_plan, te._layer_plans, te._visual_plan)
    header = te._header
    rec["ours_read"] = te.read_report
    # the vision tower's own output (for B), read again into our skeleton
    vis = te._model.visual
    load_mapped_to(vis, str(TE), header, te._visual_plan, DEV)
    move_module_tree(vis.rotary_pos_emb, DEV, skip=set())
    with torch.no_grad():
        vo = te._model.get_image_features(pv_ours.to(DEV, torch.bfloat16), grid_ours.to(DEV))
        vis_ours = torch.cat(list(vo.pooler_output), 0).cpu()
        ds_feats_ours = [f.cpu() for f in vo.deepstack_features]
    drop_to_meta(vis)
    te.release()
    del te
    free()

    # ---- A DiffSynth tokens ------------------------------------------------
    from diffsynth_h3.models.minimax_h3_text_encoder import image_token_counts, presentation_fl2va
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(TOK))
    pv_ds, grid_ds, counts = image_token_counts(processor, [canvas])
    ids_ds, tags_ds = presentation_fl2va(processor.tokenizer, PROMPT, counts)
    rec["A_tokens"] = {
        "ids_identical": list(map(int, ids_ds.tolist())) == ids_ours,
        "tags_identical": bool(torch.equal(tags_ds, tags_ours)),
        "length": len(ids_ours), "vision_tokens": ntok_ours, "ds_vision_tokens": counts,
    }
    rec["B_vision_inputs"] = {
        "pixel_values_identical": bool(torch.equal(pv_ds, pv_ours)),
        "grid_identical": bool(torch.equal(grid_ds, grid_ours)),
        "grid": grid_ours.tolist(), "pixel_values_shape": list(pv_ours.shape),
    }

    # ---- B/C DiffSynth text encoder ---------------------------------------
    _, te_cfg = h3_format.inspect_h3_te(str(TE))
    out = {}
    for mode in ("int8", "bf16"):
        ds = ds_text_encoder()
        if mode == "int8":
            # B: the vision tower output in DiffSynth's model (same weights)
            load_mapped_to(ds.model.visual, str(TE), header, plans[2], DEV)
            move_module_tree(ds.model.visual.rotary_pos_emb, DEV, skip=set())
            with torch.no_grad():
                vo = ds.model.get_image_features(pv_ds.to(DEV, torch.bfloat16), grid_ds.to(DEV))
                vis_ds = torch.cat(list(vo.pooler_output), 0).cpu()
                feats_ds = [f.cpu() for f in vo.deepstack_features]
            drop_to_meta(ds.model.visual)
            ds.model.visual.rotary_pos_emb.inv_freq = vision_inv_freq(ds.model.visual.rotary_pos_emb.dim)
            rec["B_vision_inputs"]["vision_output_identical"] = bool(torch.equal(vis_ds, vis_ours))
            rec["B_vision_inputs"]["deepstack_identical"] = all(
                torch.equal(a, b) for a, b in zip(feats_ds, ds_feats_ours))
            rec["B_vision_inputs"]["vision_output_shape"] = list(vis_ours.shape)
        t0 = time.perf_counter()
        out[mode] = run_ds(ds, header, plans, list(map(int, ids_ds.tolist())), pv_ds, grid_ds, mode, te_cfg.quant)
        rec[f"C_seconds_{mode}"] = round(time.perf_counter() - t0, 1)
        del ds
        free()
    vis_rows = (tags_ours == 0)
    rec["C_text_encoder"] = {
        "shape": list(hidden_ours.shape),
        "wiring_bit_identical": bool(torch.equal(out["int8"], hidden_ours)),
        "wiring_max_abs": float((out["int8"].float() - hidden_ours.float()).abs().max()),
        "quant_rel_all_rows": rel(hidden_ours, out["bf16"]),
        "quant_rel_vision_rows": rel(hidden_ours[vis_rows], out["bf16"][vis_rows]),
        "quant_rel_text_rows": rel(hidden_ours[~vis_rows], out["bf16"][~vis_rows]),
        "quant_rel_last_row": rel(hidden_ours[-1:], out["bf16"][-1:]),
        "finite": bool(torch.isfinite(hidden_ours.float()).all()),
    }

    # ---- D VAE encode ---------------------------------------------------------
    from diffsynth_h3.models.minimax_h3_video_vae import MiniMaxH3VideoVAE
    from engine_h3.video_vae import H3VideoEncoder

    x = canvas_to_tensor(canvas).to(DEV)
    zs = {}
    for dtype in (torch.float32, torch.bfloat16):
        enc = H3VideoEncoder.from_file(str(VAE), device=DEV, dtype=dtype)
        with torch.no_grad():
            zs[str(dtype)] = enc.encode(x).float().cpu()
        del enc
        free()
    vh = sft_quant_format.read_header(str(VAE))
    with torch.device("meta"):
        dsv = MiniMaxH3VideoVAE()
    keys = [k for k in vh.tensors if k.startswith(("encoder.", "quant_conv."))]
    sd = dict(read_tensors(str(VAE), keys, vh))
    dsv.load_state_dict(sd, strict=False, assign=True)
    dsv.encoder.to(DEV, torch.bfloat16)
    dsv.quant_conv.to(DEV, torch.bfloat16)
    del sd
    with torch.no_grad():
        z_ds = dsv.encode_video(x, dtype=torch.bfloat16, process_image=True).float().cpu()
    del dsv
    free()
    zb, z32 = zs[str(torch.bfloat16)], zs[str(torch.float32)]
    rec["D_vae_encode"] = {
        "shape": list(zb.shape),
        "bf16_vs_diffsynth_bit_identical": bool(torch.equal(zb, z_ds)),
        "bf16_vs_diffsynth_max_abs": float((zb - z_ds).abs().max()),
        "bf16_vs_diffsynth_rel": rel(zb, z_ds),
        "f32_vs_bf16_max_abs": float((z32 - zb).abs().max()),
        "f32_vs_bf16_rel": rel(zb, z32),
        "f32_vs_diffsynth_rel": rel(z_ds, z32),
        "latent_abs_mean": float(z32.abs().mean()),
    }

    # ---- E layout / tags / time numbering / RoPE -----------------------------
    from diffsynth_h3.pipeline_coords import MiniMaxH3Unit_PackedSequenceBuilder

    from engine_h3.dit import H3SeqLayout
    from engine_h3.pipeline_h3 import modality_rows, time_rows
    from engine_h3.rope import build_coords

    L = len(ids_ours)
    t_lat, a_lat = latent_frames(F), audio_latents(F)
    gh, gw = H // 32, W // 32
    grid_rows = gh * gw
    packed = MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_fl2va(
        SimpleNamespace(dit=SimpleNamespace()), L, t_lat, H // 16, W // 16, a_lat, [0])
    packed["token_tags"][packed["text_pos"]] = tags_ds
    used = L + grid_rows + 2 * a_lat + t_lat * grid_rows
    layout = H3SeqLayout.make(text_rows=L, cond_rows=grid_rows, audio_rows=2 * a_lat, video_rows=t_lat * grid_rows)
    coords = build_coords(text_len=L, latent_frames=t_lat, grid_h=gh, grid_w=gw, audio_latents=a_lat, cond_frames=1)
    img_pos, audio_pos = packed["img_pos"], packed["audio_pos"]
    order_ok = (
        layout.total == used
        and torch.equal(img_pos[:grid_rows], torch.arange(layout.cond.start, layout.cond.stop))
        and torch.equal(img_pos[grid_rows:], torch.arange(layout.video.start, layout.video.stop))
        and torch.equal(audio_pos, torch.arange(layout.audio.start, layout.audio.stop))
    )
    tags_ok = torch.equal(modality_rows(layout, tags_ours), packed["token_tags"][:used])
    rope_ok = torch.equal(coords, packed["img_position_ids"][0, :used])
    sv, sa = sigma_schedule(20, 12.0), sigma_schedule(20, 3.0)
    time_ok, ms = True, []
    for i in range(20):
        t_v, t_a = 1.0 - float(sv[i]), 1.0 - float(sa[i])
        tu, ti = time_rows(layout, t_v, t_a)
        # DiffSynth model_fn_minimax_h3 (pipelines/minimax_h3_audio_video.py:1010-1021), copied:
        timesteps = torch.full((packed["seq_len"],), t_v, dtype=torch.float32)
        timesteps[audio_pos] = t_a
        timesteps[img_pos[:grid_rows]] = max(t_v, KEYFRAME_NOISE_AUG)
        dtu, dti = torch.unique(timesteps, sorted=True, return_inverse=True)
        time_ok &= bool(torch.equal(tu, dtu) and torch.equal(ti, dti[:used].to(torch.long)))
        ms.append(int(tu.numel()))
    rec["E_layout"] = {
        "used_rows": used, "cond_rows": grid_rows, "order_identical": bool(order_ok),
        "adaln_tags_identical": bool(tags_ok), "rope_coords_identical_f64": bool(rope_ok),
        "time_numbering_identical_20_steps": bool(time_ok), "m_per_step": ms,
    }

    rec["verdict"] = {
        "A_tokens": rec["A_tokens"]["ids_identical"] and rec["A_tokens"]["tags_identical"],
        "B_vision": all(rec["B_vision_inputs"][k] for k in (
            "pixel_values_identical", "grid_identical", "vision_output_identical", "deepstack_identical")),
        "C_wiring_zero": rec["C_text_encoder"]["wiring_bit_identical"],
        "C_quant_level": rec["C_text_encoder"]["quant_rel_all_rows"],
        "D_bf16_identical": rec["D_vae_encode"]["bf16_vs_diffsynth_bit_identical"],
        "D_bf16_max_abs": rec["D_vae_encode"]["bf16_vs_diffsynth_max_abs"],
        "E_all": all(rec["E_layout"][k] for k in (
            "order_identical", "adaln_tags_identical", "rope_coords_identical_f64",
            "time_numbering_identical_20_steps")),
    }
    rec["seconds"] = round(time.perf_counter() - t_start, 1)
    rec["peak_vram_reserved_mb"] = int(torch.cuda.max_memory_reserved() / 2**20)
    text = json.dumps(rec, indent=2, ensure_ascii=False, default=str)
    Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    v = rec["verdict"]
    return 0 if (v["A_tokens"] and v["B_vision"] and v["C_wiring_zero"] and v["E_all"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
