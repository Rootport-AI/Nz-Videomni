# Provenance notice — tracking/vendor/uetrack (single-object tracker)

`tracking/vendor/uetrack/` is **third-party source copied into this repository**,
unlike `engine/`, which is first-party code with a third-party ancestry. Nothing
here was written by this project; the modifications listed below are the only
changes, and each one is marked with an `NZ:` comment at the site.

- **Origin**: **UETrack** — <https://github.com/kangben258/UETrack>, commit
  **`fd13b0eaf16d51536008295f3b27807c69eaad50`** (2026-03-20, "Update README.md";
  the tree at the time of vendoring, 2026-09-11). UETrack is itself built on
  SUTrack and on **Fast-iTPN** (<https://github.com/sunsmarterjie/iTPN>, MIT, ©
  2023 University of Chinese Academy of Sciences) — `fastitpn.py` carries the
  Fast-iTPN header, which in turn credits EVA-02, timm and DeiT.

- **Weights**: the official checkpoint is
  `uetrack/checkpoints/uetrack_base.tar` (1,267,204,495 B) from
  <https://huggingface.co/kangben258/UETrack>. This project does **not**
  redistribute that file. `scripts/trim_uetrack_checkpoint.py` reduces it to the
  257 tensors a forward pass actually reads (107,164,976 B, safetensors), and
  that is what `install-UETrack.bat` fetches — from
  <https://huggingface.co/Rootport/Nz-UETrack>, a repository of its own that
  holds nothing else, so it can be withdrawn without touching any other
  distribution (see the licence note below). See
  `scripts/manifests/30-uetrack.json`.

- **License: NOT YET SETTLED — the owner is asking the authors.** The upstream
  repository carries no LICENSE file, and its README states no terms. The
  vendored `fastitpn.py` is the one file with an explicit license (MIT, from
  Fast-iTPN). Until the authors answer, treat this directory and the
  redistributed weights as **permission-pending**: the containment plan is that
  the weights live in their own HuggingFace repository, which can be deleted or
  made private on its own, and this directory can be removed without touching
  anything else in the tree (nothing outside `tracking/` imports it).

## What was taken

| Vendored file | Upstream source |
|---|---|
| `fastitpn.py` | `lib/models/uetrack/fastitpn.py` (verbatim but for the timm import) |
| `decoder.py` | `lib/models/uetrack/decoder.py` |
| `task_decoder.py` | `lib/models/uetrack/task_decoder.py` |
| `encoder.py` | `lib/models/uetrack/encoder.py` (student half only) |
| `uetrack.py` | `lib/models/uetrack/uetrack.py` (the `UETrack` module + `build_uetrack_inference`) |
| `preprocess.py` | `lib/test/tracker/utils.py` (`Preprocessor`, `sample_target`, `transform_image_to_crop`) |
| `hann.py` | `lib/test/utils/hann.py` (`hann1d`, `hann2d`) |
| `box_ops.py` | `lib/utils/box_ops.py` (`clip_box`, `box_xyxy_to_cxcywh`) |
| `timm_compat.py` | **written here**, replacing four `timm` symbols (see its header) |

## What was deliberately NOT taken

- **`lib/train/**`, `lib/test/evaluation/**`, `lib/test/analysis/**`** — training
  and benchmark harnesses.
- **`lib/models/uetrack/fastitpn_teacher.py` and `Encoder_Teacher`** — the
  distillation teacher. It only exists while training the student, and it is
  most of the official checkpoint's bulk.
- **`lib/models/uetrack/clip.py` and the CLIP text path** — tracking by text
  description. `MULTI_MODAL_LANGUAGE` is pinned `False`
  (`tracking/uetrack_runtime.py`), so the text encoder is never built. Measured
  against the official checkpoint (2026-09-11): with the text branch off,
  `load_state_dict(strict=False)` reports **0 missing keys** and 455 unexpected
  ones — 452 `text_encoder.*`, 2 `interface_text_proj.*` (the CLIP seam) and 1
  `adjust_layers.0.weight` (the distillation adapter). Nothing the tracker needs
  is absent. (The canonical record of these numbers is
  `Docs/VERIFICATION_LOG.md` §104.3.)
- **`lib/config/uetrack/config.py` + `experiments/uetrack/uetrack_base.yaml`**
  (and with them yacs / easydict / PyYAML) — the resolved Base values are folded
  into `_BASE_CFG` in `tracking/uetrack_runtime.py`, one commented line each,
  naming which upstream file the value came from.
- **`TEST.UPDATE_INTERVALS` / `TEST.UPDATE_THRESHOLD`** — upstream's template
  auto-update. Its defaults (interval 999999, threshold 1.0) disable it, and this
  project's first release does not offer it, so it is not ported.
- **`lib/utils/misc.py`** — reached only for `is_main_process()` in the encoder
  factory; see below.

## Modifications

1. **CUDA pins removed (14 call sites).** Upstream hardcodes `.cuda()`; this port
   is CPU-only by design (the tracker must run while the GPU is generating). The
   14 sites and where they went:

   | Upstream site | Count | Here |
   |---|---:|---|
   | `lib/test/tracker/uetrack.py:20` `network.cuda()` | 1 | `load_tracker()` → `network.to(device)` |
   | `lib/test/tracker/uetrack.py:27` Hanning window | 1 | `Tracker.__init__` → `.to(device)` |
   | `lib/test/tracker/uetrack.py:58,60,62,64,66` task-index tensors | 5 | dropped — there is no benchmark to name, so `task_index_batch` is `None` |
   | `lib/test/tracker/utils.py:11-14` normalisation constants | 4 | `preprocess.Preprocessor(device=...)` |
   | `lib/test/tracker/utils.py:24` input tensor | 1 | `preprocess.Preprocessor.process` → `.to(self.device)` |
   | `lib/models/uetrack/decoder.py:85,87` corner-head grids | 2 | `Corner_Predictor(..., device="cpu")` (class is never built here — `DECODER.TYPE` is `CENTER`) |

2. **`timm` replaced by `timm_compat.py`** — `register_model` (a registry this
   port never queries), `to_2tuple`, `drop_path`, `trunc_normal_`. Three of the
   four never touch a trained weight; see that file's header.

3. **`torchvision` dropped** — upstream's `box_ops.py` imports `box_area` for the
   IoU/GIoU training losses only.

4. **`interface_text_proj` made conditional** (`uetrack.py`). Upstream reads
   `self.text_encoder.textencoder_dim` unconditionally, so upstream's own
   `MULTI_MODAL_LANGUAGE=False` path raises `AttributeError` as written.

5. **`pretrained=is_main_process()` → `pretrained=False`** (`encoder.py`). That
   flag loads `ENCODER.PRETRAIN_TYPE`, an ImageNet backbone used as a *training*
   starting point (and `''` in the Base config). Dropping it also drops
   `lib.utils.misc`, and with it `torch.distributed`.

6. **Training-side classes not carried** — `SUTRACK`, `ADAPTIVE_NET`,
   `build_uetrack`, `forward_textencoder_inference`, `map_box_back_batch`,
   `extract_token_from_nlp_clip`, the `debug`/`cv2.imshow` visualisation, and
   `resize_sample_target`.

Everything else — the network geometry, the normalisation constants, the crop
maths, the Hanning penalty, the box decoding — is upstream's, unchanged. That the
port is faithful is checked by construction: the trimmed safetensors and the
official `.tar` were driven over the same 176-frame clip and returned **bit-identical
boxes and scores** (recorded in `Docs/VERIFICATION_LOG.md` §104.3; the raw
spike output lives outside the repository, under the gitignored
`outputs/uetrack-spike-2026-09-11/`).
