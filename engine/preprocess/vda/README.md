# NOTICE — vendored third-party code

This directory contains the **inference core of Video-Depth-Anything (vits /
"Small")**, copied from

- Upstream: <https://github.com/DepthAnything/Video-Depth-Anything>
- Copyright (2025) Bytedance Ltd. and/or its affiliates
- Licensed under the **Apache License, Version 2.0** — full text in `LICENSE`

Parts of the copied tree carry their own upstream attributions, which are kept
verbatim in the file headers:

- `video_depth_anything/dinov2.py`, `video_depth_anything/dinov2_layers/*` —
  DINOv2, Copyright (c) Meta Platforms, Inc. and affiliates (Apache-2.0).
- `video_depth_anything/motion_module/motion_module.py` — derived from
  AnimateDiff (`guoyww/AnimateDiff`, Apache-2.0).
- `video_depth_anything/motion_module/attention.py` — derived from HuggingFace
  Diffusers, Copyright 2022 The HuggingFace Team (Apache-2.0).

It is used by `engine/preprocess/depth.py` to turn an uploaded reference video
into the grayscale depth control signal the Union-Control IC-LoRA consumes.

## What was copied

Only what `VideoDepthAnything.infer_video_depth` needs. Everything else from
upstream — `run.py`, `run_streaming.py`, `app.py`, `benchmark/`, `loss/`,
`utils/dc_utils.py`, `video_depth_anything/video_depth_stream.py` — is **not**
here. `dc_utils.py` in particular was excluded because it pulls in `decord`,
`matplotlib` and `imageio`, none of which are installed in the engine venv;
video decode/encode is done with cv2 by `engine/preprocess/driver.py` instead.

The upstream directory layout (`video_depth_anything/` + `utils/`) is preserved
so a future upstream diff stays readable.

The `.pth` checkpoint is **not** in git. It is installed to
`models/preprocessors-vda/video_depth_anything_vits.pth` (see
`scripts/install_ltx.ps1`).

## Modifications (this is the complete list)

Class names, module layout, attribute names and constructor parameters are
UNCHANGED — the checkpoint is loaded with `strict=True`, so any rename would
break it.

1. **Relative imports.** `video_depth_anything/video_depth.py` imported
   `from utils.util import ...` — a TOP-LEVEL import that would silently bind to
   any other `utils` package that happens to be on `sys.path`. It is now
   `from ..utils.util import ...`.

2. **`easydict` removed.** `video_depth_anything/dpt_temporal.py` used
   `EasyDict(...)` as a keyword bag that is only ever splatted as `**kwargs`;
   it is now a plain `dict(...)`. `easydict` is not installed in the engine venv
   and this was its only use in the copied tree.

3. **SDPA instead of materialised attention (two places).** xFormers is not
   installed, and upstream's no-xFormers fallback builds the full NxN score
   matrix. Measured in the G0 smoke at 1920x1088: 14.7 GB reserved with the
   upstream fallback vs ~4.4 GB with SDPA, and SDPA was also faster; the
   numerical difference was well inside fp16 tolerance.
   - `video_depth_anything/dinov2_layers/attention.py` — `Attention.forward`
     now calls `F.scaled_dot_product_attention`. The explicit `q * self.scale`
     is dropped because SDPA applies the same `1/sqrt(head_dim)` internally.
     `MemEffAttention` keeps its name (referenced by `dinov2_layers/__init__.py`
     and `block.py`) but its xFormers body is gone; it inherits the SDPA
     forward. `attn_drop` (p=0.0, inference only) is not wired in.
   - `video_depth_anything/motion_module/attention.py` —
     `CrossAttention._attention` now calls `F.scaled_dot_product_attention`
     with `scale=self.scale` passed EXPLICITLY (upstream applied it as
     `baddbmm`'s `alpha`, and it is not necessarily `1/sqrt(head_dim)` here).

4. **xFormers import guards removed in three files.** In
   `motion_module/attention.py` and `motion_module/motion_module.py` the
   `except ImportError` branch did a bare `print(...)`, and the worker's STDOUT
   is its framed protocol channel. Removed along with the branches they gated:
   `CrossAttention._memory_efficient_attention_xformers`,
   `CrossAttention._memory_efficient_attention_split`, the xFormers arm of
   `CrossAttention.forward` and of `TemporalAttention.forward`, and the two dead
   `self._use_memory_efficient_attention_xformers` assignments. The remaining
   path is exactly the one upstream already took whenever xFormers was absent —
   i.e. the configuration the G0 smoke measured. The guard in
   `dinov2_layers/attention.py` went with the SDPA rewrite.

   The guards in `dinov2_layers/block.py` and `dinov2_layers/swiglu_ffn.py` are
   left UNTOUCHED: they log through `logging` (not STDOUT) and their fallbacks
   are pure-torch, so they behave correctly as-is.
