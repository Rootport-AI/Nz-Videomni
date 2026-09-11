"""Turn the official UETrack checkpoint into the weight file this project ships.

    .venv-utils\\Scripts\\python.exe scripts\\trim_uetrack_checkpoint.py <in.tar> <out.safetensors>

WHY THIS EXISTS
---------------
huggingface.co/kangben258/UETrack distributes `uetrack/checkpoints/uetrack_base.tar`,
about 1.18 GB. Almost none of that is the tracker: the archive is a training
snapshot, so alongside the ~27M-parameter student it carries the distillation
teacher, the CLIP text encoder, and (in some checkpoints) optimiser moments --
none of which a forward pass touches. Selecting only the names the inference
network actually declares cuts it to roughly a tenth, and the result is written
as safetensors so that installing the weights never unpickles anything.

The selection rule is deliberately mechanical: take
`build_uetrack_inference(...).state_dict()` as the authority and keep exactly the
keys it declares. Nothing is named by hand here, so this script cannot drift away
from the vendored model -- if tracking/vendor/uetrack ever grows or loses a
tensor, the output follows on the next run. Keys the network wants but the
checkpoint does not have are reported, not invented (see tracking/VENDOR_NOTICE.md
for why `interface_text_proj.*` is expected to be among them).

Run this on the `.venv-utils` interpreter: it imports the vendored tracker, which
needs torch. The output file is NOT committed -- it is uploaded to the project's
HuggingFace repo and fetched by install-UETrack.bat (scripts/manifests/30-uetrack.json).
"""

from __future__ import annotations

import argparse
import os
import sys

# Import `tracking` as a package no matter where this is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import save_file  # noqa: E402

from tracking.uetrack_runtime import _BASE_CFG, _load_state_dict, _namespace  # noqa: E402
from tracking.vendor.uetrack import build_uetrack_inference  # noqa: E402


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n / 1.0:,.1f} {unit}"
        n /= 1024.0
    return f"{n} B"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="official uetrack_base.tar (a torch pickle)")
    ap.add_argument("output", help="where to write the trimmed .safetensors")
    args = ap.parse_args()

    in_size = os.path.getsize(args.input)
    print(f"input : {args.input}  ({_human(in_size)})")

    # Read through tracking/uetrack_runtime.py's `_load_state_dict`, not a second
    # torch.load here: that function is the ONE place that knows how either
    # weight format is opened (and, for the .tar, the module stubs the official
    # pickle needs). A private copy would be the first thing to rot.
    src = _load_state_dict(args.input)
    print(f"        tensors in archive: {len(src):,}")

    wanted = build_uetrack_inference(_namespace(_BASE_CFG)).state_dict()
    print(f"        tensors the inference network declares: {len(wanted):,}")

    kept, missing, shape_mismatch = {}, [], []
    for name, ref in wanted.items():
        if name not in src:
            missing.append(name)
            continue
        t = src[name]
        if tuple(t.shape) != tuple(ref.shape):
            shape_mismatch.append((name, tuple(t.shape), tuple(ref.shape)))
            continue
        # contiguous().clone() because safetensors refuses views that share
        # storage with another tensor in the same file.
        kept[name] = t.detach().cpu().contiguous().clone()

    dropped = [k for k in src.keys() if k not in kept]

    print("")
    print(f"kept   : {len(kept):,} tensors, "
          f"{sum(v.numel() for v in kept.values()):,} parameters")
    for name in sorted(kept):
        v = kept[name]
        print(f"         + {name}  {tuple(v.shape)}  {v.dtype}")

    if missing:
        print("")
        print(f"missing: {len(missing)} name(s) the network declares but the archive "
              f"does not carry (expected: the CLIP text seam)")
        for name in missing:
            print(f"         ? {name}")

    if shape_mismatch:
        print("")
        print(f"MISMATCH: {len(shape_mismatch)} name(s) present but the wrong shape. "
              f"The vendored geometry does not match this checkpoint -- refusing to write.")
        for name, got, want in shape_mismatch:
            print(f"         ! {name}  archive {got}  network {want}")
        return 2

    print("")
    print(f"dropped: {len(dropped):,} tensor(s) from the archive "
          f"(teacher / CLIP / training state)")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_file(kept, args.output, metadata={"format": "pt"})
    out_size = os.path.getsize(args.output)
    print("")
    print(f"output: {args.output}  ({_human(out_size)})")
    print(f"        {out_size / in_size:.1%} of the official archive")
    print(f"        manifest 'min' suggestion (8% under): {int(out_size * 0.92):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
