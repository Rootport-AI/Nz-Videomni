r"""Gate 1-G2 (ledger §1-97 stage 1, plan §5): the key-frame parts against DiffSynth-Studio. GPU, small.

    rem from the repository root
    set PYTHONPATH=%CD%
    set PYTHONUTF8=1
    .venv-engine-h3\Scripts\python.exe outputs\h3-keyframes-2026-10-19\g2_keyframe_parts.py [--dry-run]

Rewritten from ``Docs/Outputs-archive/h3-gate2/g3_i2v_parts.py`` (2a-G3) for the
multi-image signatures of step 1a: ``H3TextEncoder.vision_inputs(images)``,
``H3TextEncoder.encode(prompt, images=...)``, ``build_coords(cond_frame_idx=...)``
and K blocks of condition rows. The environment, the model paths and the
comparisons are those of g3; only the inputs are now lists.

The oracle is the verbatim DiffSynth copy in ``outputs/h3-gate2/oracle/diffsynth_h3/``
(commit 4e34969, Apache-2.0; the text-encoder and VAE copies exist ONLY there,
``pipeline_coords.py`` is the same file as ``Docs/Outputs-archive/h3-gate1/oracle``).

Two cases, each on the 640 x 384 canvas, 124 frames (T' = 37), the G5 prompt:

  first_last  frame_idx [0, 123]  images [kf_first, kf_last]  DiffSynth keyframe_indices [0, -1]
  last_only   frame_idx [123]     images [kf_last]            DiffSynth keyframe_indices [-1]

The images are the first / last frame of ``outputs/h3-gate1/g5_t2v_after_review.mp4``
(``extract_keyframes.py``; kf_first.png is byte-identical to outputs/h3-gate2/i2v_input.png).
Both are 640 x 384 already, so the canvas is the image itself.

Per case (all exact unless said otherwise):
A  token ids + per-token tags: ours (H3TextEncoder.encode's last_ids / tags) vs DiffSynth
   ``image_token_counts`` + ``presentation_fl2va`` (one ``<Picture i>`` per image, in order).
B  vision inputs: pixel_values / image_grid_thw [K, 3] from both (the processor called once
   with all images); the vision tower's output and DeepStack features (same bf16 weights).
C  the text encoder output [L, 5120]:
     ours      = H3TextEncoder (Qwen3VLModel, int8 W8A8 layers read directly)
     ds_wiring = DiffSynth MiniMaxH3TextEncoder.forward with the SAME int8 layers -> must be 0
     ds_bf16   = DiffSynth forward with every int8 layer dequantized to bf16 (recorded level)
E  the packed layout: condition-row order (K blocks), AdaLN tags, timestep numbering for the
   20 steps (cond rows max(t, 0.999)) and the RoPE coordinates vs DiffSynth
   ``_build_packed_fl2va(keyframe_indices=...)``, float64 ``torch.equal``.
D  (per image, not per case) VAE encode: ours bf16 vs DiffSynth bf16 (difference recorded,
   2a-G3 found max 1.52e-2 from the tile-seam blend dtype), ours f32 vs ours bf16 (recorded).

Exit code 0 = A, B, C-wiring and E true in both cases. D is recorded only.

``--dry-run`` (CPU only, no weights read, no CUDA touched): parses the arguments, checks
the files exist, imports every oracle and engine module the run uses, and computes the
CPU-only parts as a preview -- A with the tokenizer (ours ``presentation`` vs DiffSynth),
B's pixel_values / grid (the image processor), and E (coordinates / layout / tags with the
DiffSynth tags). Written to ``g2_keyframe_parts.dryrun.json``; it is NOT the gate result.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORACLE = ROOT / "outputs" / "h3-gate2" / "oracle"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ORACLE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402,F401

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
    ensure_extra_special_tokens,
    install_layer_read_hooks,
    presentation,
    te_quant_names,
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
LAST = F - 1
DEV = torch.device("cuda")

#: name -> (our frame_idx list, image keys in that order, DiffSynth keyframe_indices)
CASES = {
    "first_last": ([0, LAST], ["first", "last"], [0, -1]),
    "last_only": ([LAST], ["last"], [-1]),
}

LOG_PATH: Path | None = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def rel(a, b) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def free():
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# DiffSynth text encoder, built on meta and read layer by layer (as g3)
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
    """mode 'int8': our H3Int8Linear in DiffSynth's model; 'bf16': dequantized nn.Linear. (g3, unchanged)"""
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
        specs = dict(quant.items())
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
                module.load_state_dict(sd, strict=True, assign=True)
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
# pieces shared by the run and the dry run (CPU)
# --------------------------------------------------------------------------- #

def ds_tokens(processor, canvases):
    """DiffSynth A/B inputs: (pixel_values, grid [K,3], counts, ids [L], tags [L])."""
    from diffsynth_h3.models.minimax_h3_text_encoder import image_token_counts, presentation_fl2va

    pv, grid, counts = image_token_counts(processor, list(canvases))
    ids, tags = presentation_fl2va(processor.tokenizer, PROMPT, counts)
    return pv, grid, counts, ids, tags


def e_layout(L: int, frame_idx: list[int], ds_indices: list[int], tags_ours: torch.Tensor,
             tags_ds: torch.Tensor) -> dict:
    """E: layout / AdaLN tags / 20-step time numbering / RoPE vs DiffSynth, float64 exact."""
    from diffsynth_h3.pipeline_coords import MiniMaxH3Unit_PackedSequenceBuilder

    from engine_h3.dit import H3SeqLayout
    from engine_h3.pipeline_h3 import modality_rows, time_rows
    from engine_h3.rope import build_coords

    K = len(frame_idx)
    t_lat, a_lat = latent_frames(F), audio_latents(F)
    gh, gw = H // 32, W // 32
    grid_rows = gh * gw
    cond_rows = K * grid_rows
    packed = MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_fl2va(
        SimpleNamespace(dit=SimpleNamespace()), L, t_lat, H // 16, W // 16, a_lat, list(ds_indices))
    packed["token_tags"][packed["text_pos"]] = tags_ds
    used = L + cond_rows + 2 * a_lat + t_lat * grid_rows
    layout = H3SeqLayout.make(text_rows=L, cond_rows=cond_rows, audio_rows=2 * a_lat, video_rows=t_lat * grid_rows)
    coords = build_coords(text_len=L, latent_frames=t_lat, grid_h=gh, grid_w=gw, audio_latents=a_lat,
                          cond_frame_idx=list(frame_idx))
    img_pos, audio_pos = packed["img_pos"], packed["audio_pos"]
    order_ok = (
        layout.total == used
        and torch.equal(img_pos[:cond_rows], torch.arange(layout.cond.start, layout.cond.stop))
        and torch.equal(img_pos[cond_rows:], torch.arange(layout.video.start, layout.video.stop))
        and torch.equal(audio_pos, torch.arange(layout.audio.start, layout.audio.stop))
    )
    tags_ok = torch.equal(modality_rows(layout, tags_ours.to(torch.long)), packed["token_tags"][:used])
    ref = packed["img_position_ids"][0, :used]
    rope_ok = torch.equal(coords, ref)
    # the condition blocks' time coordinate, ours vs DiffSynth (one value per block)
    cond_t_ours = [float(coords[L + k * grid_rows, 0]) for k in range(K)]
    cond_t_ds = [float(ref[L + k * grid_rows, 0]) for k in range(K)]
    sv, sa = sigma_schedule(20, 12.0), sigma_schedule(20, 3.0)
    time_ok, ms = True, []
    for i in range(20):
        t_v, t_a = 1.0 - float(sv[i]), 1.0 - float(sa[i])
        tu, ti = time_rows(layout, t_v, t_a)
        # DiffSynth model_fn_minimax_h3 (pipelines/minimax_h3_audio_video.py:1010-1021), copied;
        # the condition rows are img_pos[:num_keyframes * frame_rows] there.
        timesteps = torch.full((packed["seq_len"],), t_v, dtype=torch.float32)
        timesteps[audio_pos] = t_a
        timesteps[img_pos[:cond_rows]] = max(t_v, KEYFRAME_NOISE_AUG)
        dtu, dti = torch.unique(timesteps, sorted=True, return_inverse=True)
        time_ok &= bool(torch.equal(tu, dtu) and torch.equal(ti, dti[:used].to(torch.long)))
        ms.append(int(tu.numel()))
    return {
        "frame_idx": list(frame_idx), "ds_keyframe_indices": list(ds_indices), "text_len": L,
        "used_rows": used, "cond_rows": cond_rows, "cond_blocks": K,
        "cond_t_ours": [repr(v) for v in cond_t_ours], "cond_t_ds": [repr(v) for v in cond_t_ds],
        "order_identical": bool(order_ok), "adaln_tags_identical": bool(tags_ok),
        "rope_coords_identical_f64": bool(rope_ok),
        "rope_max_abs": float((coords - ref).abs().max()) if coords.shape == ref.shape else None,
        "time_numbering_identical_20_steps": bool(time_ok), "m_per_step": ms,
    }


E_KEYS = ("order_identical", "adaln_tags_identical", "rope_coords_identical_f64",
          "time_numbering_identical_20_steps")


def load_images(paths: dict[str, str]) -> tuple[dict, dict]:
    canvases, sizes = {}, {}
    for key, p in paths.items():
        canvas, src = load_canvas_image(p, W, H)
        canvases[key], sizes[key] = canvas, list(src)
    return canvases, sizes


# --------------------------------------------------------------------------- #
# dry run (CPU only)
# --------------------------------------------------------------------------- #

def dry_run(args) -> int:
    rec: dict = {"mode": "dry-run (CPU only; NOT the gate result)", "prompt": PROMPT, "canvas": [W, H],
                 "num_frames": F, "latent_frames": latent_frames(F), "audio_latents": audio_latents(F)}
    files = {"te": TE, "tokenizer": TOK, "vae": VAE, "oracle": ORACLE / "diffsynth_h3",
             "first": Path(args.first), "last": Path(args.last)}
    rec["files_exist"] = {k: p.exists() for k, p in files.items()}
    # every oracle module the run imports
    import diffsynth_h3.models.minimax_h3_text_encoder as ds_te
    import diffsynth_h3.models.minimax_h3_video_vae as ds_vae
    import diffsynth_h3.pipeline_coords as ds_coords
    from engine_h3.video_vae import H3VideoEncoder  # noqa: F401

    rec["oracle_imports"] = {
        "minimax_h3_text_encoder": [n for n in ("MiniMaxH3TextEncoder", "image_token_counts", "presentation_fl2va")
                                    if hasattr(ds_te, n)],
        "minimax_h3_video_vae": [n for n in ("MiniMaxH3VideoVAE",) if hasattr(ds_vae, n)],
        "pipeline_coords": [n for n in ("MiniMaxH3Unit_PackedSequenceBuilder",) if hasattr(ds_coords, n)],
        "files": {"text_encoder": ds_te.__file__, "video_vae": ds_vae.__file__, "coords": ds_coords.__file__},
    }
    rec["cuda_initialized"] = torch.cuda.is_initialized()
    if not all(rec["files_exist"].values()):
        log(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
        return 2

    from transformers import AutoProcessor, AutoTokenizer

    canvases, sizes = load_images({"first": args.first, "last": args.last})
    rec["source_size"] = sizes
    tokenizer = AutoTokenizer.from_pretrained(str(TOK))   # as H3TextEncoder.load does
    ensure_extra_special_tokens(tokenizer)
    processor = AutoProcessor.from_pretrained(str(TOK))
    ours_te = H3TextEncoder(str(TE), str(TOK), device="cpu")   # constructor only: no read
    rec["cases"] = {}
    for name, (frame_idx, keys, ds_idx) in CASES.items():
        imgs = [canvases[k] for k in keys]
        pv_o, grid_o, ntok_o = ours_te.vision_inputs(imgs)
        ids_o, tags_o = presentation(tokenizer, PROMPT, ntok_o)
        pv_d, grid_d, counts, ids_d, tags_d = ds_tokens(processor, imgs)
        tags_o_t = torch.tensor(tags_o, dtype=torch.long)
        case = {
            "A_tokens_preview": {
                "ids_identical": list(map(int, ids_d.tolist())) == list(ids_o),
                "tags_identical": bool(torch.equal(tags_d, tags_o_t)),
                "length": len(ids_o), "vision_tokens": ntok_o, "ds_vision_tokens": counts,
            },
            "B_vision_inputs_preview": {
                "pixel_values_identical": bool(torch.equal(pv_d, pv_o)),
                "grid_identical": bool(torch.equal(grid_d, grid_o)),
                "grid": grid_o.tolist(), "pixel_values_shape": list(pv_o.shape),
            },
            "E_layout_preview": e_layout(len(ids_o), frame_idx, ds_idx, tags_o_t, tags_d),
        }
        rec["cases"][name] = case
    rec["cuda_initialized_at_end"] = torch.cuda.is_initialized()
    text = json.dumps(rec, indent=2, ensure_ascii=False, default=str)
    Path(args.dry_report).write_text(text, encoding="utf-8")
    log(text)
    ok = all(
        c["A_tokens_preview"]["ids_identical"] and c["A_tokens_preview"]["tags_identical"]
        and c["B_vision_inputs_preview"]["pixel_values_identical"] and c["B_vision_inputs_preview"]["grid_identical"]
        and all(c["E_layout_preview"][k] for k in E_KEYS)
        for c in rec["cases"].values()
    )
    log(f"dry-run preview: {'all identical' if ok else 'MISMATCH (see JSON)'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# the gate (GPU)
# --------------------------------------------------------------------------- #

def main() -> int:
    global LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default=str(HERE / "kf_first.png"))
    ap.add_argument("--last", default=str(HERE / "kf_last.png"))
    ap.add_argument("--report", default=str(HERE / "g2_keyframe_parts.json"))
    ap.add_argument("--log", default=None, help="default: <report stem>.log (dry run: .dryrun.log)")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU only: arguments, files, oracle imports and the CPU preview of A/B-inputs/E")
    ap.add_argument("--dry-report", default=str(HERE / "g2_keyframe_parts.dryrun.json"))
    args = ap.parse_args()
    LOG_PATH = Path(args.log) if args.log else Path(args.report).with_suffix(".dryrun.log" if args.dry_run else ".log")
    LOG_PATH.write_text("", encoding="utf-8")
    if args.dry_run:
        return dry_run(args)

    t_start = time.perf_counter()
    rec: dict = {"images": {"first": args.first, "last": args.last}, "canvas": [W, H], "num_frames": F,
                 "prompt": PROMPT, "cases": {}}
    rec["int8_backend"] = require_cuda_int8()
    canvases, sizes = load_images({"first": args.first, "last": args.last})
    rec["source_size"] = sizes

    # ---- A/B/C ours (one skeleton, one encode per case) ---------------------
    te = H3TextEncoder(str(TE), str(TOK), device=DEV)
    te.load()
    plans = (te._embed_plan, te._layer_plans, te._visual_plan)
    header = te._header
    ours: dict = {}
    for name, (frame_idx, keys, _ds_idx) in CASES.items():
        imgs = [canvases[k] for k in keys]
        log(f"{name}: ours encode ({len(imgs)} image(s))")
        pv, grid, ntok = te.vision_inputs(imgs)
        hidden, tags = te.encode(PROMPT, images=imgs)
        o = {"pv": pv, "grid": grid, "ntok": ntok, "hidden": hidden.cpu(), "tags": tags.cpu(),
             "ids": list(te.last_ids), "read": dict(te.read_report)}
        del hidden, tags
        # the vision tower's own output (for B), read again into our skeleton
        vis = te._model.visual
        load_mapped_to(vis, str(TE), header, te._visual_plan, DEV)
        move_module_tree(vis.rotary_pos_emb, DEV, skip=set())
        with torch.no_grad():
            vo = te._model.get_image_features(pv.to(DEV, torch.bfloat16), grid.to(DEV))
            o["vis"] = torch.cat(list(vo.pooler_output), 0).cpu()
            o["deepstack"] = [f.cpu() for f in vo.deepstack_features]
        del vo
        drop_to_meta(vis)
        vis.rotary_pos_emb.inv_freq = vision_inv_freq(vis.rotary_pos_emb.dim)
        ours[name] = o
        free()
    te.release()
    del te
    free()

    # ---- A DiffSynth tokens / B inputs --------------------------------------
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(TOK))
    dsin: dict = {}
    for name, (frame_idx, keys, ds_idx) in CASES.items():
        o = ours[name]
        pv_ds, grid_ds, counts, ids_ds, tags_ds = ds_tokens(processor, [canvases[k] for k in keys])
        dsin[name] = (pv_ds, grid_ds, ids_ds, tags_ds)
        rec["cases"][name] = {
            "frame_idx": frame_idx, "images": keys, "ds_keyframe_indices": ds_idx, "ours_read": o["read"],
            "A_tokens": {
                "ids_identical": list(map(int, ids_ds.tolist())) == o["ids"],
                "tags_identical": bool(torch.equal(tags_ds, o["tags"])),
                "length": len(o["ids"]), "vision_tokens": o["ntok"], "ds_vision_tokens": counts,
            },
            "B_vision_inputs": {
                "pixel_values_identical": bool(torch.equal(pv_ds, o["pv"])),
                "grid_identical": bool(torch.equal(grid_ds, o["grid"])),
                "grid": o["grid"].tolist(), "pixel_values_shape": list(o["pv"].shape),
            },
        }

    # ---- B/C DiffSynth text encoder (fresh model per case and mode, as g3) ---
    _, te_cfg = h3_format.inspect_h3_te(str(TE))
    for name in CASES:
        o, c = ours[name], rec["cases"][name]
        pv_ds, grid_ds, ids_ds, _tags_ds = dsin[name]
        out = {}
        for mode in ("int8", "bf16"):
            log(f"{name}: DiffSynth TE ({mode})")
            ds = ds_text_encoder()
            if mode == "int8":
                load_mapped_to(ds.model.visual, str(TE), header, plans[2], DEV)
                move_module_tree(ds.model.visual.rotary_pos_emb, DEV, skip=set())
                with torch.no_grad():
                    vo = ds.model.get_image_features(pv_ds.to(DEV, torch.bfloat16), grid_ds.to(DEV))
                    vis_ds = torch.cat(list(vo.pooler_output), 0).cpu()
                    feats_ds = [f.cpu() for f in vo.deepstack_features]
                del vo
                drop_to_meta(ds.model.visual)
                ds.model.visual.rotary_pos_emb.inv_freq = vision_inv_freq(ds.model.visual.rotary_pos_emb.dim)
                c["B_vision_inputs"]["vision_output_identical"] = bool(torch.equal(vis_ds, o["vis"]))
                c["B_vision_inputs"]["deepstack_identical"] = (
                    len(feats_ds) == len(o["deepstack"])
                    and all(torch.equal(a, b) for a, b in zip(feats_ds, o["deepstack"])))
                c["B_vision_inputs"]["vision_output_shape"] = list(o["vis"].shape)
            t0 = time.perf_counter()
            out[mode] = run_ds(ds, header, plans, list(map(int, ids_ds.tolist())), pv_ds, grid_ds, mode,
                               te_cfg.quant)
            c[f"C_seconds_{mode}"] = round(time.perf_counter() - t0, 1)
            del ds
            free()
        hid = o["hidden"]
        vis_rows = (o["tags"] == 0)
        c["C_text_encoder"] = {
            "shape": list(hid.shape),
            "wiring_bit_identical": bool(torch.equal(out["int8"], hid)),
            "wiring_max_abs": float((out["int8"].float() - hid.float()).abs().max()),
            "quant_rel_all_rows": rel(hid, out["bf16"]),
            "quant_rel_vision_rows": rel(hid[vis_rows], out["bf16"][vis_rows]),
            "quant_rel_text_rows": rel(hid[~vis_rows], out["bf16"][~vis_rows]),
            "quant_rel_last_row": rel(hid[-1:], out["bf16"][-1:]),
            "finite": bool(torch.isfinite(hid.float()).all()),
        }
        del out
        Path(args.report).write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # ---- D VAE encode, per image ---------------------------------------------
    from diffsynth_h3.models.minimax_h3_video_vae import MiniMaxH3VideoVAE

    from engine_h3.video_vae import H3VideoEncoder

    xs = {k: canvas_to_tensor(canvases[k]).to(DEV) for k in ("first", "last")}
    zs: dict = {k: {} for k in xs}
    for dtype in (torch.float32, torch.bfloat16):
        enc = H3VideoEncoder.from_file(str(VAE), device=DEV, dtype=dtype)
        with torch.no_grad():
            for k, x in xs.items():
                zs[k][str(dtype)] = enc.encode(x).float().cpu()
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
    z_ds = {}
    with torch.no_grad():
        for k, x in xs.items():
            z_ds[k] = dsv.encode_video(x, dtype=torch.bfloat16, process_image=True).float().cpu()
    del dsv
    free()
    rec["D_vae_encode"] = {}
    for k in xs:
        zb, z32, zd = zs[k][str(torch.bfloat16)], zs[k][str(torch.float32)], z_ds[k]
        rec["D_vae_encode"][k] = {
            "shape": list(zb.shape),
            "bf16_vs_diffsynth_bit_identical": bool(torch.equal(zb, zd)),
            "bf16_vs_diffsynth_max_abs": float((zb - zd).abs().max()),
            "bf16_vs_diffsynth_rel": rel(zb, zd),
            "f32_vs_bf16_max_abs": float((z32 - zb).abs().max()),
            "f32_vs_bf16_rel": rel(zb, z32),
            "f32_vs_diffsynth_rel": rel(zd, z32),
            "latent_abs_mean": float(z32.abs().mean()),
        }

    # ---- E layout / tags / time numbering / RoPE ------------------------------
    for name, (frame_idx, _keys, ds_idx) in CASES.items():
        o = ours[name]
        rec["cases"][name]["E_layout"] = e_layout(len(o["ids"]), frame_idx, ds_idx, o["tags"], dsin[name][3])

    # ---- verdict ----------------------------------------------------------------
    for name, c in rec["cases"].items():
        c["verdict"] = {
            "A_tokens": c["A_tokens"]["ids_identical"] and c["A_tokens"]["tags_identical"],
            "B_vision": all(c["B_vision_inputs"].get(k) for k in (
                "pixel_values_identical", "grid_identical", "vision_output_identical", "deepstack_identical")),
            "C_wiring_zero": c["C_text_encoder"]["wiring_bit_identical"],
            "C_quant_level": c["C_text_encoder"]["quant_rel_all_rows"],
            "E_all": all(c["E_layout"][k] for k in E_KEYS),
        }
    rec["verdict"] = {
        "pass_A_B_C_E_both_cases": all(
            v["A_tokens"] and v["B_vision"] and v["C_wiring_zero"] and v["E_all"]
            for v in (c["verdict"] for c in rec["cases"].values())),
        "D_recorded": {k: {"bit_identical": d["bf16_vs_diffsynth_bit_identical"],
                           "max_abs": d["bf16_vs_diffsynth_max_abs"]} for k, d in rec["D_vae_encode"].items()},
    }
    rec["seconds"] = round(time.perf_counter() - t_start, 1)
    rec["peak_vram_reserved_mb"] = int(torch.cuda.max_memory_reserved() / 2**20)
    text = json.dumps(rec, indent=2, ensure_ascii=False, default=str)
    Path(args.report).write_text(text, encoding="utf-8")
    log(text)
    return 0 if rec["verdict"]["pass_A_B_C_E_both_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
