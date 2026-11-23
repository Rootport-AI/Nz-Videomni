r"""Engine-direct rerun of one control item of the pi-H3 spike (I1c / O1c), for the off-path check (item 1).

Runs in the ENGINE venv (.venv-engine-h3; imports torch). It is started by ``h3_pih3_gates.py`` after the
GPU gate, never on its own for a live run. It rebuilds, from the spike's ``runs/<item>/result.json``, the
same ``H3Paths`` / ``H3LoadOptions`` / ``H3GenerateRequest`` the spike passed to ``H3Pipeline.generate``
(``h3_pih3_spike.py`` ``_pipeline`` / ``run_edit``), changes only ``output_path``, and calls the product's
engine with no replacement of any function (the spike's control items also ran the product's sigma
schedule and decoder; its record-only wrap of the paste-back is the one thing left out).

    .venv-engine-h3/Scripts/python.exe -I engine_direct.py --source <spike result.json> --out <dir> --dry-run
    .venv-engine-h3/Scripts/python.exe -I engine_direct.py --source <spike result.json> --out <dir>

``--dry-run``: CUDA hidden, nothing loaded; the request is rebuilt and checked with the product's
``validate_request`` (files present, LoRA header). Writes ``<out>/engine_direct_plan.json``.
Live: writes ``<out>/output.mp4`` and ``<out>/engine_direct.json`` (the build report and the engine's result).
Exit 0 = done, 1 = failed, 2 = bad arguments / refused.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# The spike's fixed paths (h3_pih3_spike.py PATHS); the transformer comes from the item's record.
PATHS = {
    "text_encoder": "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "video_vae": "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
    "tokenizer_dir": "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
}


def log(msg: str) -> None:
    print(f"[engine_direct] {msg}", flush=True)


def rebase(p, old_root: str | None) -> str:
    """A recorded absolute path under the spike's repo root, moved under this repo root."""
    s = str(p)
    if old_root and s.lower().startswith(old_root.lower()):
        return str(REPO / s[len(old_root):].lstrip("\\/"))
    return s


def build(source: Path, out_dir: Path):
    """-> (record, paths, options, request or chain spec). Single requests (the spike's I1c / O1c / I1 / O1),
    chains (C3b: ``request_base`` + the recorded prompts / frames) and retakes (``request.base`` / ``clips`` /
    ``retake``). A pi-H3 item of the spike (``conditions.pi_h3`` set) carried the pi-H3 file as a LoRA of
    strength 1.0 plus its own patches; here that LoRA is taken out and the product's ``pi_h3`` field is set
    (the product's pi-H3 path: its internal LoRA, sigmas, initial state, adapter and DWD decode)."""
    from engine_h3 import pipeline_h3 as P  # noqa: PLC0415

    rec = json.loads(source.read_text(encoding="utf-8"))
    cond = rec["conditions"]
    raw = rec.get("request") or {}
    if "request_base" in rec:
        kind, rq = "chain", dict(rec["request_base"])
    elif "base" in raw:
        kind, rq = "retake", dict(raw["base"])
    else:
        kind, rq = "single", dict(raw)
    old_root = None
    marker = "\\outputs\\h3-pih3-spike-2026-11-22\\"
    if rq.get("output_path") and marker in rq["output_path"]:
        old_root = rq["output_path"].split(marker)[0]
    paths = P.H3Paths(transformer=str(REPO / cond["base_weights"]["path"]),
                      **{k: str(REPO / v) for k, v in PATHS.items()})
    options = P.H3LoadOptions(blocks_on_gpu=int(cond["blocks_on_gpu"]), attention_backend=str(cond["attention"]),
                              block_swap_prefetch=bool(cond["block_swap_prefetch"]),
                              deterministic=bool(cond["deterministic"]))
    pi_cond = cond.get("pi_h3")
    if pi_cond:
        pi_name = Path(pi_cond["path"]).name
        rq["loras"] = [x for x in rq.get("loras") or () if Path(str(x["path"])).name != pi_name]
    rq["output_path"] = str(out_dir / "output.mp4")
    rq["images"] = tuple(P.H3ImageInput(path=rebase(i["path"], old_root), frame_idx=int(i["frame_idx"]))
                         for i in rq.get("images") or ())
    rq["loras"] = tuple(P.H3LoraInput(path=rebase(x["path"], old_root), strength=float(x["strength"]))
                        for x in rq.get("loras") or ())
    if rq.get("references"):
        raise SystemExit("references in the source request: not handled by this helper")
    rq["references"] = ()
    if rq.get("source_audio"):
        raise SystemExit("source_audio in the source request: not handled by this helper")
    for key, cls in (("inpaint", P.H3InpaintInput), ("outpaint", P.H3OutpaintInput)):
        v = rq.get(key)
        if v is not None:
            v = dict(v)
            for pk in ("path", "mask_path"):
                if pk in v:
                    v[pk] = rebase(v[pk], old_root)
            rq[key] = cls(**v)
    rq.pop("pi_h3", None)   # the spike's request had no such field (= None, off)
    if pi_cond:
        rq["pi_h3"] = P.H3PiH3Input(steps=int(cond["steps"]), path=str(REPO / pi_cond["path"]))
    req = P.H3GenerateRequest(**rq)
    if kind == "single":
        return rec, paths, options, req
    from engine_h3.chain_h3 import H3ChainClipSpec, H3ChainRetake, H3ChainSpec  # noqa: PLC0415

    if kind == "chain":   # the spike's chain_spec: clip i = (prompts[i], num_frames[i]), no source / retake
        clips = tuple(H3ChainClipSpec(prompt=p, num_frames=int(f))
                      for p, f in zip(cond["prompts"], cond["num_frames"]))
        return rec, paths, options, H3ChainSpec(base=req, clips=clips, seed=int(cond["seed"]))
    clips = tuple(H3ChainClipSpec(prompt=c["prompt"], num_frames=int(c["num_frames"])) for c in raw["clips"])
    if any(c.get("references") or c.get("images") or c.get("source_audio") for c in raw["clips"]) or raw.get("source"):
        raise SystemExit("clip references / images / source audio or a source in the retake record: not handled")
    rt = raw["retake"]
    return rec, paths, options, H3ChainSpec(base=req, clips=clips, seed=int(raw["seed"]),
                                            retake=H3ChainRetake(path=rebase(rt["path"], old_root),
                                                                 audio_strength=float(rt["audio_strength"])))


def check(P, spec, dit_path: str) -> None:
    """The product's own checks, before anything is loaded (no GPU)."""
    if isinstance(spec, P.H3GenerateRequest):
        P.validate_request(spec, dit_path=dit_path if spec.pi_h3 is not None else None)
        return
    from engine_h3.chain_h3 import _check_spec, clip_requests  # noqa: PLC0415

    _check_spec(spec)
    for r in clip_requests(spec):
        P.validate_request(r, dit_path=dit_path if r.pi_h3 is not None else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the spike's runs/<item>/result.json")
    ap.add_argument("--out", required=True, help="output folder (inside outputs/h3-pih3-gates-2026-11-23/runs/)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if args.dry_run:
        # "-1": no device visible (an empty value is dropped from the environment on Windows).
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    out_dir = Path(args.out).resolve()
    if HERE not in out_dir.parents:
        log(f"--out must be inside {HERE}")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch  # noqa: PLC0415

    from engine_h3 import pipeline_h3 as P  # noqa: PLC0415

    rec, paths, options, req = build(Path(args.source), out_dir)
    is_chain = not isinstance(req, P.H3GenerateRequest)
    plan = {"source_item": rec.get("item"), "source_git": rec.get("git"), "paths": dataclasses.asdict(paths),
            "options": dataclasses.asdict(options), "kind": "chain" if is_chain else "single",
            "request": dataclasses.asdict(req), "torch": torch.__version__}
    if args.dry_run:
        try:
            check(P, req, paths.transformer)
            plan["validate_request"] = "passed"
        except Exception as exc:  # noqa: BLE001
            plan["validate_request"] = f"{type(exc).__name__}: {exc}"
        # torch.cuda is not touched in a dry run (not even is_available, which initialises the driver).
        plan["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        missing = [k for k, v in dataclasses.asdict(paths).items() if not Path(v).exists()]
        plan["missing_paths"] = missing
        (out_dir / "engine_direct_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str),
                                                         encoding="utf-8")
        log(f"plan written; validate_request={plan['validate_request']} missing={missing} "
            f"CUDA_VISIBLE_DEVICES={plan['cuda_visible_devices']}")
        return 0 if plan["validate_request"] == "passed" and not missing else 1
    if not torch.cuda.is_available():
        log("CUDA is not available")
        return 2
    result: dict = {"plan": plan, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    code = 1
    pipe = P.H3Pipeline(paths, options, device=torch.device("cuda"),
                        progress=lambda stage, i, n, **kw: log(f"progress {stage} {i}/{n}"), log=log)
    try:
        t0 = time.perf_counter()
        result["build_report"] = pipe.load()
        result["load_seconds"] = round(time.perf_counter() - t0, 2)
        if is_chain:
            from engine_h3.chain_h3 import run_chain_h3  # noqa: PLC0415

            result["engine"] = run_chain_h3(pipe, req, lambda stage, i, n, **kw: log(f"progress {stage} {i}/{n}"), log)
        else:
            result["engine"] = pipe.generate(req)
        code = 0
    except BaseException as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        log(result["error"])
    finally:
        try:
            pipe.close()
        except Exception as exc:  # noqa: BLE001
            log(f"close failed (ignored): {exc!r}")
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (out_dir / "engine_direct.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                                    encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
