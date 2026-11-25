"""LTX 2.5: is a string's conditioning independent of the bundle it is encoded in?

Run from the repository root with the LTX 2.5 engine interpreter:
    .venv-engine-ltx25/Scripts/python.exe outputs/ltx25-te-bundle-spike-2026-11-25/spike.py

Builds Ltx25Pipeline exactly as engine25/worker.py's load does (same arguments
the app's load payload carries), arms each "job" exactly as the worker does
(set_acceleration_job with the app defaults, set_nag_job(None)), then calls
pipeline.prompt_encoder with the arguments DistilledPipeline.__call__ uses.
No transformer weights are loaded: the pipeline build is lazy and no denoise
runs. Writes results.json beside this file. Paths recorded are repo-relative.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(HERE / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("spike")

import torch  # noqa: E402

from engine25.pipeline25 import (  # noqa: E402
    DEFAULT_BLOCKS_ON_GPU,
    DEFAULT_TE_LAYERS_ON_GPU,
    Ltx25Pipeline,
    ModelFiles,
)

MODELS = "models"
FILES = {
    # state.json selections.LTX25 = all "default" -> manifest default_file
    # (scripts/manifests/20-ltx25.json), under config.yaml model.checkpoint_dir.
    "transformer": f"{MODELS}/LTX25/Weights/LTX-2.5-22B-distilled-transformer.gguf",
    "text_encoder": f"{MODELS}/LTX25/TextEncoder/LTX-2.5-gemma4-12b-text-encoder-Q4_K_M.gguf",
    "video_vae": f"{MODELS}/LTX25/VAE/ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae": f"{MODELS}/LTX25/VAE/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial_upsampler": f"{MODELS}/LTX25/Upscaler/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}

# The app's load payload (services/engines/ltx25/adapter.py _build_load_payload):
# blocks_on_gpu = config vram.block_swap_blocks_on_gpu (8), cache_weights True,
# deterministic True, te_layers_on_gpu NOT sent -> worker default.
LOAD = {
    "blocks_on_gpu": 8,
    "te_layers_on_gpu": DEFAULT_TE_LAYERS_ON_GPU,
    "cache_weights": True,
    "deterministic": True,
}
# Per-job knobs at the app defaults (api/models.py *_DEFAULT): prefetch on,
# fused dequant on, keep_resident off, keep_resident_embeddings off, sdpa.
# Negative prompt not requested -> set_nag_job(None) -> super().__call__ path.
JOB = {
    "block_swap_prefetch": True,
    "fused_gguf_dequant_kernel": True,
    "keep_resident": False,
    "keep_resident_embeddings": False,
    "attention_backend": "sdpa",
}

S = {
    "A": (
        "A young woman in a red raincoat walks slowly along a wet cobblestone street at dusk, "
        "neon signs reflecting in the puddles around her feet. The camera follows her from behind "
        "at shoulder height as light rain falls, and distant traffic hums softly in the background."
    ),
    "B": "A cat sleeps on a windowsill.",
    "C": (
        "Aerial drone shot over a dense pine forest in early morning fog. The camera glides forward "
        "and slowly tilts down toward a winding river, sunlight breaking through the mist in long golden beams."
    ),
    "D": (
        "Inside a busy old-fashioned bakery before dawn, a heavyset baker with flour-dusted forearms "
        "pulls a long wooden peel out of a glowing brick oven, revealing a row of crusty round loaves "
        "that crackle as they meet the cooler air. Steam curls upward under the warm amber lamps hanging "
        "from the low wooden beams. Behind him, two apprentices knead dough at a marble counter, their "
        "hands moving in a practiced rhythm, while a radio on a shelf plays a quiet jazz tune. The camera "
        "starts in a close-up on the bread, then pulls back and drifts to the left in a slow continuous "
        "dolly move, passing baskets of rolls, jars of seeds, and a chalkboard listing the day's specials "
        "in looping handwriting. Through the front window, the street outside is still dark and empty except "
        "for a single delivery bicycle leaning against a lamppost. The baker sets the peel down, wipes his "
        "brow with the back of his wrist, and laughs at something one of the apprentices says, his voice "
        "warm and low. Fine particles of flour float through the shafts of lamplight, and the whole scene "
        "has the soft, grainy texture of film shot on a handheld camera."
    ),
    "E": (
        "Close-up of an old man playing chess in a park, he moves a knight and smiles at his opponent "
        "while pigeons wander across the gravel path behind the table."
    ),
    "S": "Cinematic lighting, smooth camera motion.",
    "N": "blurry, low quality, distorted faces",
    "EMPTY": "",
}

RUNS = [
    ("R1", ["A"]),
    ("R2", ["A"]),
    ("R3", ["A", "N"]),
    ("R4", ["N"]),
    ("R5", ["A", "EMPTY"]),
    ("R6", ["EMPTY"]),
    ("R7", ["A", "B", "C", "D", "E", "S"]),
    ("R8", ["B"]),
    ("R9", ["C"]),
    ("R10", ["D"]),
    ("R11", ["E"]),
    ("R12", ["S"]),
    ("R13", ["B", "A"]),
    ("R14", ["A", "B", "C", "D", "E", "S", "N"]),
]
# reference run (single-string) for each key
REF = {"A": "R1", "N": "R4", "EMPTY": "R6", "B": "R8", "C": "R9", "D": "R10", "E": "R11", "S": "R12"}
FIELDS = ("video_encoding", "audio_encoding", "attention_mask")


def sha(t: torch.Tensor) -> str:
    c = t.detach().to("cpu").contiguous()
    # view as bytes regardless of dtype (bf16 has no numpy dtype)
    b = c.view(torch.uint8) if c.dtype != torch.bool else c.to(torch.uint8)
    return hashlib.sha256(b.numpy().tobytes()).hexdigest()


def gpu_mib() -> dict:
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
    }


def word_count(s: str) -> int:
    return len(s.split())


def run_once(pipe: Ltx25Pipeline, prompts: list[str]):
    """One 'job': worker-style arm, the DistilledPipeline-style call, reset."""
    pipe.set_acceleration_job(**JOB)
    pipe.set_nag_job(None)
    pipe.vram.phases.clear()
    t0 = time.perf_counter()
    try:
        # Production runs the whole job under torch.no_grad()
        # (pipeline25.py Ltx25Pipeline.generate, chain25.py run_chain).
        with torch.no_grad():
            out = pipe.prompt_encoder(
            prompts,
            enhance_first_prompt=False,
            enhance_static_cache=False,
            enhance_prompt_image=None,
        )
        torch.cuda.synchronize()
    finally:
        pipe.reset_acceleration_job()
    wall = time.perf_counter() - t0
    phases = {k: v.get("seconds") for k, v in pipe.vram.phases.items()}
    return list(out), wall, phases


def diff_facts(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None) -> dict:
    a32 = a.detach().float().cpu()
    b32 = b.detach().float().cpu()
    if a32.shape != b32.shape:
        return {"shape_mismatch": [list(a32.shape), list(b32.shape)]}
    ne = a32 != b32
    d = (a32 - b32).abs()
    facts = {"n_diff": int(ne.sum()), "n_total": int(ne.numel()), "max_abs": float(d.max())}
    if a32.dim() >= 2:
        # token axis = dim 1 for [B, L, D] / [B, L]
        tok = ne.reshape(ne.shape[0], ne.shape[1], -1).any(dim=-1).any(dim=0)
        pos = torch.nonzero(tok).flatten().tolist()
        facts["n_token_positions_with_diff"] = len(pos)
        facts["token_positions_first_last"] = [pos[:10], pos[-10:]]
        if mask is not None:
            m = mask.detach().cpu().reshape(-1)
            if m.numel() == tok.numel():
                real = m.bool() if m.dtype == torch.bool else (m > 0.5) if m.is_floating_point() else (m != 0)
                facts["diff_at_real_tokens"] = int((tok & real).sum())
                facts["diff_at_pad_tokens"] = int((tok & ~real).sum())
                facts["n_real_tokens"] = int(real.sum())
    return facts


def main() -> int:
    assert torch.cuda.is_available()
    files = ModelFiles(
        transformer=FILES["transformer"],
        text_encoder=FILES["text_encoder"],
        video_vae=FILES["video_vae"],
        audio_vae=FILES["audio_vae"],
        spatial_upsampler=FILES["spatial_upsampler"],
        text_encoder_assets=None,
    )
    t0 = time.perf_counter()
    pipe = Ltx25Pipeline(
        files,
        blocks_on_gpu=LOAD["blocks_on_gpu"],
        te_layers_on_gpu=LOAD["te_layers_on_gpu"],
        cache_weights=LOAD["cache_weights"],
        deterministic=LOAD["deterministic"],
        progress=None,
    )
    build_s = time.perf_counter() - t0
    log.info("pipeline built in %.2fs; gpu=%s", build_s, gpu_mib())

    results: dict = {
        "conditions": {
            "files": FILES,
            "load": LOAD,
            "load_defaults_note": {
                "DEFAULT_BLOCKS_ON_GPU": DEFAULT_BLOCKS_ON_GPU,
                "DEFAULT_TE_LAYERS_ON_GPU": DEFAULT_TE_LAYERS_ON_GPU,
            },
            "job": JOB,
            "negative_prompt": "not requested (set_nag_job(None)) -> Ltx25PromptEncoder.__call__ takes super().__call__",
            "call_kwargs": {"enhance_first_prompt": False, "enhance_static_cache": False, "enhance_prompt_image": None},
            "env": {k: os.environ.get(k) for k in ("TORCH_COMPILE_DISABLE", "PYTHONUNBUFFERED")},
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "pipeline_build_seconds": round(build_s, 2),
            "build_report": json.loads(json.dumps(pipe.build_report, default=str)),
        },
        "strings": {k: {"text": v, "words": word_count(v)} for k, v in S.items()},
        "runs": [],
    }

    tensors: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for run_id, keys in RUNS:
        prompts = [S[k] for k in keys]
        out, wall, phases = run_once(pipe, prompts)
        assert len(out) == len(keys)
        entry = {"run": run_id, "keys": keys, "n": len(keys), "seconds_total": round(wall, 2), "phases": phases, "items": {}}
        tensors[run_id] = {}
        for k, o in zip(keys, out, strict=True):
            item = {}
            tensors[run_id][k] = {}
            for f in FIELDS:
                t = getattr(o, f)
                item[f] = {"sha256": sha(t), "shape": list(t.shape), "dtype": str(t.dtype)}
                tensors[run_id][k][f] = t.detach().to("cpu").clone()
            entry["items"][k] = item
        del out
        torch.cuda.empty_cache()
        entry["gpu_after"] = gpu_mib()
        results["runs"].append(entry)
        log.info("%s %s %.2fs phases=%s", run_id, keys, wall, phases)
        if run_id == "R2":
            same = all(
                entry["items"]["A"][f]["sha256"] == results["runs"][0]["items"]["A"][f]["sha256"] for f in FIELDS
            )
            if not same:
                log.error("R2 != R1: same-bundle rerun is not reproducible; stopping")
                results["stopped_after_R2"] = True
                break

    # comparisons against each key's single-string run
    comps = []
    by_id = {r["run"]: r for r in results["runs"]}
    for r in results["runs"]:
        for k in r["keys"]:
            ref = "R1" if (r["run"] == "R1") else REF[k]
            if ref == r["run"] or ref not in by_id:
                continue
            for f in FIELDS:
                a = r["items"][k][f]["sha256"]
                b = by_id[ref]["items"][k][f]["sha256"]
                c = {"run": r["run"], "key": k, "position": r["keys"].index(k), "n": r["n"], "ref": ref, "field": f, "match": a == b}
                if a != b:
                    mask = tensors[ref][k]["attention_mask"]
                    c["diff"] = diff_facts(tensors[r["run"]][k][f], tensors[ref][k][f], mask)
                comps.append(c)
    results["comparisons"] = comps
    mismatches = [c for c in comps if not c["match"]]
    results["all_match"] = not mismatches and not results.get("stopped_after_R2")
    log.info("comparisons=%d mismatches=%d", len(comps), len(mismatches))

    # text-encoder stage, once, only if a mismatch exists
    if mismatches:
        first = mismatches[0]
        results["te_stage_check"] = te_stage_check(pipe, first)

    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pipe.close()
    del pipe
    torch.cuda.empty_cache()
    log.info("done; gpu=%s", gpu_mib())
    return 0


def te_stage_check(pipe: Ltx25Pipeline, mismatch: dict) -> dict:
    """Re-run the mismatched bundle and its single-string reference, capturing
    the text encoder's own outputs (the (hidden_states, mask) pairs handed to the
    embeddings processor) by wrapping the built encoder's ``encode`` to record
    its return value. The computation is untouched."""
    run_keys = dict(RUNS)[mismatch["run"]]
    key = mismatch["key"]
    pe = pipe.prompt_encoder
    captured: list = []
    cls_build = type(pe)._build_text_encoder

    def build_and_wrap():
        enc = cls_build(pe)
        inner = enc.encode

        def encode(prompts):
            r = inner(prompts)
            captured.append([(tuple(h.detach().to("cpu").clone() for h in hs), m.detach().to("cpu").clone()) for hs, m in r])
            return r

        enc.encode = encode
        return enc

    pe._build_text_encoder = build_and_wrap
    try:
        out_b, _, _ = run_once(pipe, [S[k] for k in run_keys])
        out_s, _, _ = run_once(pipe, [S[key]])
    finally:
        del pe._build_text_encoder
    hs_b, m_b = captured[0][run_keys.index(key)]
    hs_s, m_s = captured[1][0]
    ob = out_b[run_keys.index(key)]
    os_ = out_s[0]
    rep = {
        "bundle_run": mismatch["run"],
        "bundle": run_keys,
        "key": key,
        "rerun_ep_outputs_match": {f: sha(getattr(ob, f)) == sha(getattr(os_, f)) for f in FIELDS},
        "te_mask_match": sha(m_b) == sha(m_s),
        "te_n_hidden_layers": len(hs_b),
        "te_layers": [],
    }
    first_diff = None
    for i, (a, b) in enumerate(zip(hs_b, hs_s, strict=True)):
        same = sha(a) == sha(b)
        e = {"layer": i, "match": same, "shape": list(a.shape), "dtype": str(a.dtype)}
        if not same:
            e["diff"] = diff_facts(a, b, m_s)
            if first_diff is None:
                first_diff = i
        rep["te_layers"].append(e)
    rep["te_first_layer_with_diff"] = first_diff
    rep["te_all_layers_match"] = first_diff is None
    return rep


if __name__ == "__main__":
    raise SystemExit(main())
