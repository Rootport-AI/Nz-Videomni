r"""§1-116 T2 / T3: the whole H3 text encoder (50 layers, direct read) with a swapped attention function.

Run from the repo root (one condition = one process; run_t2.ps1 / run_t3.ps1 do the lists):

    set PYTHONPATH=%CD%
    set PYTHONUTF8=1
    .venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-te-vram-2026-10-29\t2_te_ab.py --mode today --refs pictures3 --save outputs\h3-te-vram-2026-10-29\today_a.pt --report <json>
    ... --mode f1 --drop-early --refs pictures9 --cap --report <json>
    ... --selfcheck            (CPU only, no weights: refs / token counts / attention-function contract / g-block mask)

Process layout as Docs/Outputs-archive/h3-gate2/g2_te_direct.py::run_encoder: require_cuda_int8(),
then the pipeline's require_fp32_matmul() (pipeline_h3.py:647-659; cuDNN's deterministic flags are set
there only with options.deterministic, False by default, so they are left as they are here), then
H3TextEncoder -> load -> encode(prompt, refs) -> release.

--mode (plan §5): the text side's config._attn_implementation is set to a function registered with
transformers.AttentionInterface after te.load() (the vision tower keeps "sdpa"). Hugging Face builds no
mask for an unregistered mask name, so the function gets attention_mask=None and applies causality itself.
  today  nothing registered (stock "sdpa")
  probe  pass-through to sdpa_attention_forward; the first call's arguments are recorded
  f1     k / v repeat_interleave -> sdpa_kernel([EFFICIENT_ATTENTION]), is_causal=True
  f0     sdpa_kernel([CUDNN_ATTENTION]), is_causal=True, enable_gqa=True
  g256 / g512  math forced, query blocks against the full k / v with bool mask slices (t1's g_blocked_attention)
  sage   sageattention.sageattn(q, k, v, tensor_layout="HND", is_causal=True, sm_scale=module.scaling)
--drop-early (J): forward hooks drop lm.embed_tokens and model.visual to meta right after their forward.
--cap: the allocator cap of VRAM 16GB mode (derive_memory_cap, margin VRAM16_CAP_MARGIN_MIB) during encode.
Always: a pre-hook on layer 0 splits the peak into "before the decoder" and "the decoder"; every layer's
pre / post hook tracks the smallest mem_get_info free.

Exit code 0 = encode finished with a finite hidden state (and, with --baseline, the comparison was
recorded -- equality is NOT a pass criterion); 1 = it raised (OOM included; the JSON still gets written).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from t1_sdpa_kernels import g_blocked_attention, run_kernel  # noqa: E402

TE = ROOT / "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
TOK = ROOT / "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer"
#: The fixed prompt of g2_te_direct.py (G-R2's own prompt is not on record; plan §11 Minor 10).
PROMPT = (
    "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in "
    "puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>"
)
UPLOADS = Path(r"C:\Users\phenotype nezu\.claude\uploads\10f3a842-2a5e-48b4-8a07-419f0020a740")
PICTURES = [UPLOADS / "17fde0c1-image.png", UPLOADS / "d80320dd-image.png", UPLOADS / "31d79f30-image.png"]
VIDEO = ROOT / "outputs/03cffa21-eba7-4950-9229-746601284a28/output.mp4"
OUT_DIR = ROOT / "outputs/h3-te-vram-2026-10-29"
CANVAS_W, CANVAS_H = 1344, 768
VIDEO_FRAMES = 362
MODES = ("today", "probe", "f1", "g256", "g512", "sage", "f0")
REFS = ("pictures3", "pictures9", "video15")
BENCH_PREFIX = "bench_"


# -- references and prompt -----------------------------------------------------------


def build_refs(kind: str) -> tuple[list, str, dict]:
    """``(refs, prompt, info)`` for ``--refs kind`` (CPU only)."""
    from engine_h3.reference import load_reference_image, qwen_pairs, read_reference_video
    from engine_h3.text_encoder import RefMedia

    info: dict = {"refs": kind}
    t0 = time.perf_counter()
    if kind in ("pictures3", "pictures9"):
        count = 3 if kind == "pictures3" else 9
        images, sizes = [], []
        for path in PICTURES:
            tensor, meta = load_reference_image(str(path), CANVAS_W, CANVAS_H)
            images.append(tensor)
            sizes.append({"file": path.name, **meta})
        refs = [RefMedia("picture", slot, image=images[(slot - 1) % 3]) for slot in range(1, count + 1)]
        tags = "".join(f"<Picture {slot}>" for slot in range(1, count + 1))
        if kind == "pictures3":
            refs.append(RefMedia("audio", 1))
            tags += "<Audio 1>"
        info["pictures"] = sizes
        info["picture_order"] = [PICTURES[(slot - 1) % 3].name for slot in range(1, count + 1)]
    elif kind == "video15":
        frames, vinfo = read_reference_video(str(VIDEO), num_frames=VIDEO_FRAMES)
        pairs = qwen_pairs(frames)
        del frames
        if len(pairs) != 16:
            raise AssertionError(f"video15: {len(pairs)} pairs, 16 expected")
        refs = [RefMedia("video", 1, pairs=tuple(pairs))]
        tags = "<Video 1>"
        info["video"] = {"file": str(VIDEO.relative_to(ROOT)), **vinfo, "pairs": len(pairs),
                         "labels": [label for _pair, label in pairs]}
    else:
        raise ValueError(kind)
    info["build_seconds"] = round(time.perf_counter() - t0, 2)
    return refs, f"{tags} {PROMPT}", info


def count_tokens(te, tokenizer, prompt: str, refs) -> dict:
    """The token count ``encode`` will see, from the CPU side only (``ref_vision_inputs`` + ``presentation``)."""
    from engine_h3.text_encoder import presentation

    items, _pv, grid, report = te.ref_vision_inputs(refs)
    ids, tags = presentation(tokenizer, prompt, ref_items=items)
    vision = sum(sum(i.vision_tokens) for i in items)
    per_pair = [list(i.vision_tokens) for i in items if i.kind == "video"]
    return {"tokens": len(ids), "vision_tokens": int(vision), "text_tokens": len(ids) - int(vision),
            "items": len(items), "references": report, "video_tokens_per_pair": per_pair,
            "grid_rows": None if grid is None else int(grid.shape[0])}


# -- attention functions (registered with transformers.AttentionInterface) ----------------

STATE: dict = {"calls": 0, "probe": None, "min_free_attn_mb": None, "track_free": False}


def _note_free() -> None:
    if STATE["track_free"]:
        free_mb = torch.cuda.mem_get_info()[0] / 2**20
        cur = STATE["min_free_attn_mb"]
        STATE["min_free_attn_mb"] = free_mb if cur is None else min(cur, free_mb)


def _require_unmasked(attention_mask, dropout, kwargs) -> None:
    # Causality is applied inside the function, so a mask (or a non-causal call) would be ignored silently.
    if attention_mask is not None:
        raise RuntimeError(f"bench attention: attention_mask is not None ({tuple(attention_mask.shape)})")
    if dropout:
        raise RuntimeError(f"bench attention: dropout {dropout}")
    if kwargs.get("is_causal") is False:
        raise RuntimeError("bench attention: is_causal=False was passed")


def make_attention(mode: str):
    """The function for ``mode``: ``fn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs)``
    -> ``(out.transpose(1, 2).contiguous(), None)``; q (B, 64, N, 128), k / v (B, 8, N, 128)."""
    if mode == "probe":
        from transformers.integrations.sdpa_attention import sdpa_attention_forward, use_gqa_in_sdpa

        def probe_fn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
            if STATE["probe"] is None:
                STATE["probe"] = {
                    "attention_mask_is_none": attention_mask is None,
                    "kwargs_keys": sorted(kwargs),
                    "kwargs_is_causal": kwargs.get("is_causal", "<absent>"),
                    "module_is_causal": getattr(module, "is_causal", "<absent>"),
                    "module_class": type(module).__name__,
                    "num_key_value_groups": getattr(module, "num_key_value_groups", None),
                    "query_shape": list(query.shape), "key_shape": list(key.shape), "value_shape": list(value.shape),
                    "query_dtype": str(query.dtype), "key_dtype": str(key.dtype),
                    "query_contiguous": query.is_contiguous(), "key_contiguous": key.is_contiguous(),
                    "dropout": dropout, "scaling": scaling, "module_scaling": getattr(module, "scaling", None),
                    "use_gqa_in_sdpa": bool(use_gqa_in_sdpa(attention_mask, key, value)),
                }
            STATE["calls"] += 1
            out = sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                         **kwargs)
            _note_free()
            return out

        return probe_fn

    if mode == "sage":
        from sageattention import sageattn

        def kernel(query, key, value, scale):
            return sageattn(query, key, value, tensor_layout="HND", is_causal=True, sm_scale=scale)
    elif mode in ("f0", "f1", "g256", "g512"):
        def kernel(query, key, value, scale):
            return run_kernel(mode, query, key, value, scale=scale)
    else:
        raise ValueError(mode)

    def bench_fn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        _require_unmasked(attention_mask, dropout, kwargs)
        scale = scaling if scaling is not None else module.scaling
        out = kernel(query, key, value, scale)
        STATE["calls"] += 1
        _note_free()
        return out.transpose(1, 2).contiguous(), None

    return bench_fn


def register(mode: str) -> str | None:
    """Register ``mode``'s function; returns its name (None for today = stock sdpa)."""
    if mode == "today":
        return None
    from transformers import AttentionInterface

    name = BENCH_PREFIX + mode
    AttentionInterface.register(name, make_attention(mode))
    return name


# -- hooks --------------------------------------------------------------------------------


class Meters:
    """Layer-0 pre-hook (peak split) + every layer's pre / post hook (smallest free) + optional J hooks."""

    def __init__(self, model, *, drop_early: bool, index: int) -> None:
        from engine_h3.sft_load import drop_to_meta

        self.index = index
        self.rec: dict = {"min_free_pre_mb": None, "min_free_post_mb": None, "dropped": []}
        self.handles: list = []
        lm = model.language_model

        def pre0(module, args, kwargs):
            torch.cuda.synchronize(index)
            self.rec["pre_decoder_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated(index) / 2**20, 1)
            self.rec["pre_decoder_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved(index) / 2**20, 1)
            self.rec["at_decoder_start_alloc_mb"] = round(torch.cuda.memory_allocated(index) / 2**20, 1)
            self.rec["at_decoder_start_reserved_mb"] = round(torch.cuda.memory_reserved(index) / 2**20, 1)
            torch.cuda.reset_peak_memory_stats(index)

        def track(key):
            def hook(*_a, **_k):
                free_mb = round(torch.cuda.mem_get_info(index)[0] / 2**20, 1)
                cur = self.rec[key]
                self.rec[key] = free_mb if cur is None else min(cur, free_mb)
            return hook

        # Registered BEFORE encode installs its read hooks: layer 0's pre-hook runs before layer 0 is read,
        # so "pre_decoder" is the embedding table + vision tower + their activations only.
        self.handles.append(lm.layers[0].register_forward_pre_hook(pre0, with_kwargs=True))
        for layer in lm.layers:
            self.handles.append(layer.register_forward_pre_hook(track("min_free_pre_mb")))
            self.handles.append(layer.register_forward_hook(track("min_free_post_mb")))
        if drop_early:
            for label, module in (("embed_tokens", lm.embed_tokens), ("visual", model.visual)):
                def drop(mod, _args, _out, _label=label):
                    drop_to_meta(mod)
                    self.rec["dropped"].append(_label)
                self.handles.append(module.register_forward_hook(drop))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


# -- comparison ---------------------------------------------------------------------------


def compare(hidden: torch.Tensor, tags: torch.Tensor, ids: list, base: dict) -> dict:
    bh = base["hidden"]
    rec = {"ids_identical": list(base["ids"]) == list(ids),
           "tags_identical": bool(torch.equal(base["tags"], tags)) if base["tags"].shape == tags.shape else False,
           "shape_match": list(bh.shape) == list(hidden.shape), "baseline_mode": base.get("mode")}
    if not rec["shape_match"]:
        return rec
    a, b = hidden.float(), bh.float()
    d = (a - b).abs()
    cos = F.cosine_similarity(a, b, dim=-1)
    rec.update({
        "bit_identical": bool(torch.equal(hidden, bh)),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "min_token_cosine": float(cos.min()),
        "min_token_cosine_index": int(cos.argmin()),
        "mean_token_cosine": float(cos.mean()),
        "tokens_not_identical": int((hidden != bh).any(dim=-1).sum()),
        "baseline_abs_mean": float(b.abs().mean()),
    })
    return rec


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return OUT_DIR / p if p.parent == Path(".") else p


def _mb(x) -> float:
    return round(x / 2**20, 1)


# -- the run ---------------------------------------------------------------------------------


def run(args) -> int:
    from engine_h3.constants import VRAM16_CAP_MARGIN_MIB
    from engine_h3.lowvram import derive_memory_cap
    from engine_h3.pipeline_h3 import _cuda_index, _mem_gib, require_fp32_matmul
    from engine_h3.quant_linear import require_cuda_int8
    from engine_h3.text_encoder import H3TextEncoder

    rec: dict = {"mode": args.mode, "drop_early": args.drop_early, "refs": args.refs, "cap": args.cap,
                 "torch": torch.__version__, "mem": {}, "seconds": {}}
    report = Path(args.report or HERE / f"t2_{args.refs}_{args.mode}{'_dropearly' if args.drop_early else ''}"
                                        f"{'_cap' if args.cap else ''}.json")
    ok = False
    exit_code = 1
    te = None
    meters = None
    cap_set = False
    dev = torch.device("cuda")
    index = None
    try:
        backends = require_cuda_int8()
        rec["cuda_backend"] = {k: bool(v.get("available")) and not bool(v.get("disabled"))
                               for k, v in backends.items() if isinstance(v, dict)}
        require_fp32_matmul()   # pipeline_h3.py:655 (deterministic flags: options.deterministic=False -> untouched)
        rec["numerics"] = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                           "float32_matmul_precision": torch.get_float32_matmul_precision(),
                           "cudnn_deterministic": torch.backends.cudnn.deterministic,
                           "cudnn_benchmark": torch.backends.cudnn.benchmark}
        rec["device_name"] = torch.cuda.get_device_name(0)
        if args.mode == "sage":
            import importlib.metadata as md
            rec["sageattention"] = md.version("sageattention")
        name = register(args.mode)
        rec["attn_implementation"] = name or "sdpa"

        te = H3TextEncoder(str(TE), str(TOK), device=dev)
        index = _cuda_index(dev)
        rec["mem"]["start"] = _mem_gib()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        rec["load_info"] = te.load()
        t1 = time.perf_counter()
        rec["seconds"]["load"] = round(t1 - t0, 2)
        rec["mem"]["after_load"] = _mem_gib()
        model = te._model
        if name:
            model.language_model.config._attn_implementation = name
        rec["attn_check"] = {
            "text": model.language_model.layers[0].self_attn.config._attn_implementation,
            "vision": model.visual.config._attn_implementation,
        }
        if rec["attn_check"]["text"] != (name or "sdpa") or rec["attn_check"]["vision"] != "sdpa":
            raise RuntimeError(f"attention implementation not as intended: {rec['attn_check']}")

        refs, prompt, rinfo = build_refs(args.refs)
        rec["refs_info"] = rinfo
        rec["prompt"] = prompt
        t2 = time.perf_counter()
        rec["expected"] = count_tokens(te, te._tokenizer, prompt, refs)
        rec["seconds"]["precount"] = round(time.perf_counter() - t2, 2)
        if args.refs == "video15" and rec["expected"]["tokens"] < 16000:
            raise AssertionError(f"video15: N={rec['expected']['tokens']} < 16,000 (the plan's condition)")

        meters = Meters(model, drop_early=args.drop_early, index=index)
        STATE["track_free"] = True
        stats0 = torch.cuda.memory_stats(index)
        if args.cap:
            free, total = torch.cuda.mem_get_info(index)
            reserved = torch.cuda.memory_reserved(index)
            fraction = derive_memory_cap(free, total, reserved, VRAM16_CAP_MARGIN_MIB * 2**20)
            torch.cuda.set_per_process_memory_fraction(fraction, index)
            cap_set = True
            rec["cap_info"] = {"fraction": fraction, "cap_mib": round(fraction * total / 2**20, 1),
                               "free_mib": _mb(free), "total_mib": _mb(total), "reserved_mib": _mb(reserved),
                               "margin_mib": VRAM16_CAP_MARGIN_MIB}
        rec["free_before_encode_mb"] = _mb(torch.cuda.mem_get_info(index)[0])
        t3 = time.perf_counter()
        try:
            with torch.no_grad():
                hidden, tags = te.encode(prompt, refs=refs)
            torch.cuda.synchronize()
        finally:
            t4 = time.perf_counter()
            rec["seconds"]["encode"] = round(t4 - t3, 2)
            STATE["track_free"] = False
            meters.remove()
            if cap_set:
                torch.cuda.set_per_process_memory_fraction(1.0, index)
                cap_set = False
            stats1 = torch.cuda.memory_stats(index)
            rec["allocator"] = {k: int(stats1.get(k, 0)) - int(stats0.get(k, 0)) for k in ("num_ooms", "num_alloc_retries")}
            rec["decoder_peak_alloc_mb"] = _mb(torch.cuda.max_memory_allocated(index))
            rec["decoder_peak_reserved_mb"] = _mb(torch.cuda.max_memory_reserved(index))
            rec["hooks"] = meters.rec
            rec["min_free_attn_mb"] = None if STATE["min_free_attn_mb"] is None else round(STATE["min_free_attn_mb"], 1)
            rec["attention_calls"] = STATE["calls"]
            rec["probe"] = STATE["probe"]
            rec["mem"]["after_encode"] = _mem_gib()
        ids = list(te.last_ids)
        rec["read_report"] = te.read_report
        rec["tokens"] = len(ids)
        hidden_cpu, tags_cpu = hidden.detach().cpu(), tags.detach().cpu()
        del hidden, tags
        t5 = time.perf_counter()
        te.release()
        te = None
        rec["seconds"]["release"] = round(time.perf_counter() - t5, 2)
        rec["seconds"]["total"] = round(time.perf_counter() - t0, 2)
        rec["mem"]["after_release"] = _mem_gib()
        rec["hidden"] = {"shape": list(hidden_cpu.shape), "dtype": str(hidden_cpu.dtype),
                         "finite": bool(torch.isfinite(hidden_cpu.float()).all()),
                         "abs_mean": float(hidden_cpu.float().abs().mean())}
        rec["tokens_match_expected"] = rec["tokens"] == rec["expected"]["tokens"]
        save = _resolve(args.save)
        if save:
            save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"hidden": hidden_cpu, "tags": tags_cpu, "ids": ids, "prompt": prompt, "mode": args.mode,
                        "drop_early": args.drop_early, "refs": args.refs}, save)
            rec["saved"] = str(save)
        base_path = _resolve(args.baseline)
        if base_path:
            base = torch.load(base_path)
            rec["baseline"] = str(base_path)
            rec["vs_baseline"] = compare(hidden_cpu, tags_cpu, ids, base)
            if base.get("prompt") != prompt:
                rec["vs_baseline"]["prompt_differs"] = True
        ok = (rec["hidden"]["finite"] and rec["tokens_match_expected"]
              and rec["attention_calls"] == (50 if name else 0))
        exit_code = 0 if ok else 1
    except torch.cuda.OutOfMemoryError as exc:
        rec["oom"] = True
        rec["error"] = f"OutOfMemoryError: {exc}"
        rec["traceback"] = traceback.format_exc()
    except BaseException as exc:  # noqa: BLE001 -- the reason goes into the JSON
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
    finally:
        if cap_set and index is not None:
            torch.cuda.set_per_process_memory_fraction(1.0, index)
        if te is not None:
            try:
                te.release()
            except Exception:  # noqa: BLE001
                pass
        if torch.cuda.is_available():
            rec["peak_alloc_mb_at_exit"] = _mb(torch.cuda.max_memory_allocated())
        rec.setdefault("oom", False)
        rec["ok"] = ok
        text = json.dumps(rec, indent=2, ensure_ascii=False, default=str)
        report.write_text(text, encoding="utf-8")
        print(text)
    return exit_code


# -- self-check (CPU, no weights) ---------------------------------------------------------


def _line(results: list, ok: bool, what: str) -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {what}")


def selfcheck() -> int:
    import inspect

    from transformers import AttentionInterface, AutoTokenizer, Qwen3VLModel, Qwen3VLTextConfig, Qwen3VLTextModel
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    from engine_h3.text_encoder import TEXT_CONFIG, H3TextEncoder, build_config, ensure_extra_special_tokens

    results: list[bool] = []
    tokenizer = AutoTokenizer.from_pretrained(str(TOK))
    ensure_extra_special_tokens(tokenizer)
    te = H3TextEncoder("unused.safetensors", str(TOK), device="cpu")
    for kind in REFS:
        t0 = time.perf_counter()
        refs, prompt, info = build_refs(kind)
        c = count_tokens(te, tokenizer, prompt, refs)
        print(f"      {kind}: refs={len(refs)} tokens={c['tokens']} vision={c['vision_tokens']} text={c['text_tokens']} "
              f"per_pair={c['video_tokens_per_pair']} ({time.perf_counter() - t0:.1f}s) prompt head={prompt[:60]!r}")
        if kind == "pictures3":
            _line(results, len(refs) == 4 and [r.kind for r in refs] == ["picture"] * 3 + ["audio"]
                  and prompt.startswith("<Picture 1><Picture 2><Picture 3><Audio 1> "),
                  f"pictures3: 3 pictures + audio, N={c['tokens']}")
        elif kind == "pictures9":
            _line(results, c["items"] == 9 and [r.slot for r in refs] == list(range(1, 10))
                  and info["picture_order"][3] == PICTURES[0].name,
                  f"pictures9: 9 items (slots 1..9, cycled), N={c['tokens']}")
        else:
            pairs = c["video_tokens_per_pair"][0]
            _line(results, len(pairs) == 16 and c["vision_tokens"] >= 16000 and c["tokens"] >= 16000,
                  f"video15: {len(pairs)} pairs x {sorted(set(pairs))} tokens = {c['vision_tokens']} vision, "
                  f"N={c['tokens']} (>= 16,000)")

    # The registered name reaches the text layers only (meta model, no weights).
    config = build_config()
    with torch.device("meta"):
        model = Qwen3VLModel(config)
    AttentionInterface.register("bench_selfcheck", make_attention("g512"))
    model.language_model.config._attn_implementation = "bench_selfcheck"
    _line(results, model.visual.config._attn_implementation == "sdpa"
          and model.language_model.layers[0].self_attn.config._attn_implementation == "bench_selfcheck"
          and model.language_model.layers[-1].self_attn.config._attn_implementation == "bench_selfcheck"
          and all(b.attn.config._attn_implementation == "sdpa" for b in model.visual.blocks),
          "meta model: text layers -> bench_selfcheck, vision tower stays sdpa")
    del model

    # The call site and the function contract (transformers sources).
    sig = inspect.signature(sdpa_attention_forward)
    _line(results, list(sig.parameters)[:7] == ["module", "query", "key", "value", "attention_mask", "dropout", "scaling"],
          f"sdpa_attention_forward{sig}")

    # End to end on a tiny random Qwen3-VL text model (CPU, float32): stock sdpa vs the registered functions.
    small = dict(TEXT_CONFIG, hidden_size=512, intermediate_size=256, num_attention_heads=8, num_key_value_heads=1,
                 num_hidden_layers=2, vocab_size=1000, dtype="float32")
    cfg = Qwen3VLTextConfig(**small)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    tiny = Qwen3VLTextModel(cfg).eval()
    for n in (300, 1100):
        ids = torch.randint(0, 1000, (1, n))
        with torch.no_grad():
            tiny.config._attn_implementation = "sdpa"
            ref = tiny(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
            for mode in ("probe", "g256", "g512"):
                name = f"bench_sc_{mode}"
                AttentionInterface.register(name, make_attention(mode))
                tiny.config._attn_implementation = name
                STATE["calls"], STATE["probe"] = 0, None
                out = tiny(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
                d = float((out - ref).abs().max())
                extra = ""
                if mode == "probe":
                    p = STATE["probe"]
                    extra = (f" mask_none={p['attention_mask_is_none']} kwargs_is_causal={p['kwargs_is_causal']} "
                             f"module_is_causal={p['module_is_causal']} use_gqa_in_sdpa={p['use_gqa_in_sdpa']} "
                             f"q={p['query_shape']} k={p['key_shape']} scaling={p['scaling']}")
                    good = p["attention_mask_is_none"] and p["use_gqa_in_sdpa"] and p["scaling"] == p["module_scaling"]
                else:
                    good = True
                _line(results, good and d <= 1e-4 and STATE["calls"] == 2,
                      f"tiny text model N={n} {mode} vs stock sdpa: max|d|={d:.3g} calls={STATE['calls']}{extra}")
        tiny.config._attn_implementation = "sdpa"

    # A mask reaching a bench function must stop it (causality would be applied twice / ignored).
    try:
        make_attention("g512")(None, torch.zeros(1, 8, 4, 128), torch.zeros(1, 1, 4, 128), torch.zeros(1, 1, 4, 128),
                               torch.ones(1, 1, 4, 4, dtype=torch.bool))
        _line(results, False, "bench function refuses a non-None attention_mask")
    except RuntimeError:
        _line(results, True, "bench function refuses a non-None attention_mask")

    # g-block mask on the plain call (as t1 --selfcheck), the shape the model hands over.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, 64, 700, 128)
    k = torch.randn(1, 8, 700, 128)
    v = torch.randn(1, 8, 700, 128)
    with sdpa_kernel([SDPBackend.MATH]):
        full = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        blk = g_blocked_attention(q, k, v, 256)
    d = float((full - blk).abs().max())
    _line(results, d <= 2e-6, f"g-block (Bq=256, N=700 ragged) vs is_causal math: max|d|={d:.3g} "
                              f"torch.equal={bool(torch.equal(full, blk))}")

    try:
        import importlib.metadata as md

        from sageattention import sageattn
        ssig = inspect.signature(sageattn)
        _line(results, all(p in ssig.parameters for p in ("tensor_layout", "is_causal", "sm_scale")),
              f"sageattention {md.version('sageattention')}: sageattn{ssig}")
    except Exception as exc:  # noqa: BLE001
        _line(results, False, f"sageattention import: {exc}")

    ok = all(results)
    print(f"SELFCHECK {'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--refs", choices=REFS, default="pictures3")
    ap.add_argument("--drop-early", action="store_true", help="J: drop embed_tokens / visual right after their forward")
    ap.add_argument("--cap", action="store_true", help="the VRAM 16GB mode allocator cap during encode")
    ap.add_argument("--save", help=f"write hidden/tags/ids .pt (a bare file name goes under {OUT_DIR})")
    ap.add_argument("--baseline", help="a --save .pt to compare with")
    ap.add_argument("--report")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    if args.mode is None:
        ap.error("--mode is required (or --selfcheck)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
