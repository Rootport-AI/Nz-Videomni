"""Standalone self-check for loading the REAL PrunaVAED decoder file (gate G2).

Run with the ENGINE venv (needs torch + ltx_core; the app venv has neither, and
.venv-engine has no pytest, which is why this is a script and not a test module):

    .venv-engine\\Scripts\\python.exe -m engine.vae.prunavaed_g2_selfcheck [PATH]

``PATH`` defaults to config.yaml's ``model.component_video_vae_pruned_path``
(itself defaulting to
``models/LTX23/VAE/prunavaed/PrunaVAED-decoder-bf16.safetensors``).

Same conventions as the other ``*_selfcheck`` modules: every check either PASSes
or FAILs loudly, and the exit code is 0 only when all of them pass. ONE extra
state exists here — the converted file may not have been produced yet, in which
case the script exits **2** and says so. Two is deliberately not zero: "the file
was not there" must never be readable as "the check passed".

The structural claims (parameter count, key set, block kinds, norm3 type) are
already proven weight-free by ``tests/test_pruned_video_decoder.py``. What
ONLY this script can prove is the half that needs the real bytes:

  G2-1  the builder path used in production builds the decoder with ZERO
        "Uninitialized parameters or buffers" warnings. That warning is the only
        signal ``load_state_dict(strict=False, assign=True)`` gives when the key
        names do not line up (single_gpu_model_builder.py:77-84) — a file whose
        keys are wrong otherwise produces a perfectly healthy-looking model full
        of meta tensors.
  G2-2  ``load_state_dict``'s RETURN VALUE has empty ``missing_keys`` and
        ``unexpected_keys`` (strict=False hides them from raising, but they are
        still reported), and the file carries exactly 102 tensors.
  G2-3  the built module's parameter count is exactly 345,006,256.
  G2-4  the measured tensor shapes match §4.1.
  G2-5  both projection resnets carry ChannelLayerNorm3d, NOT nn.GroupNorm.
  G2-6  the same build succeeds on the GPU as well as on the CPU (skipped, and
        reported as such, when no CUDA device is present).

Authority: ``Docs/PRUNAVAED_WORKORDER.md`` §9 G2.
"""

from __future__ import annotations

import logging
import os
import sys

import torch
from torch import nn

from ltx_core.loader.registry import DummyRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.video_vae.resnet import ResnetBlock3D, UNetMidBlock3D
from ltx_core.model.video_vae.sampling import DepthToSpaceUpsample

from engine.vae.pruned_video_decoder import (
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_STATE_DICT_KEYS,
    ChannelLayerNorm3d,
    PrunedVideoDecoderConfigurator,
)

DEFAULT_PATH = os.path.join(
    "models", "LTX23", "VAE", "prunavaed",
    "PrunaVAED-decoder-bf16.safetensors",
)

_EXPECTED_KINDS = [
    UNetMidBlock3D, DepthToSpaceUpsample, UNetMidBlock3D, ResnetBlock3D,
    DepthToSpaceUpsample, UNetMidBlock3D, ResnetBlock3D, DepthToSpaceUpsample,
    UNetMidBlock3D, DepthToSpaceUpsample, UNetMidBlock3D,
]

_EXPECTED_SHAPES = {
    "conv_in.conv.weight": (1024, 128, 3, 3, 3),
    "up_blocks.1.conv.conv.weight": (4096, 1024, 3, 3, 3),
    "up_blocks.3.conv_shortcut.weight": (384, 512, 1, 1, 1),
    "up_blocks.3.norm3.weight": (512,),
    "up_blocks.4.conv.conv.weight": (3072, 384, 3, 3, 3),
    "up_blocks.6.conv_shortcut.weight": (256, 384, 1, 1, 1),
    "up_blocks.6.norm3.weight": (384,),
    "up_blocks.7.conv.conv.weight": (256, 256, 3, 3, 3),
    "up_blocks.9.conv.conv.weight": (256, 128, 3, 3, 3),
    "conv_out.conv.weight": (48, 64, 3, 3, 3),
}

_RESULTS: list[tuple[str, bool, str]] = []


class _WarningTrap(logging.Handler):
    """Catches the ONE log line that betrays a silent key mismatch."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _builder(path: str) -> SingleGPUModelBuilder:
    # EXACTLY what _set_vae_mode_job installs on the ledger: the fixed-spec
    # configurator, the pruned path, no sd_ops (bare module-relative keys).
    return SingleGPUModelBuilder(
        model_class_configurator=PrunedVideoDecoderConfigurator,
        model_path=path,
        model_sd_ops=None,
        registry=DummyRegistry(),
    )


def _run(name: str, fn) -> None:
    try:
        detail = fn()
        _RESULTS.append((name, True, detail or ""))
        print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
    except Exception as exc:  # noqa: BLE001 - a failed check is a reported check
        _RESULTS.append((name, False, repr(exc)))
        print(f"FAIL  {name}\n      {exc!r}")


def _build(path: str, device: torch.device) -> tuple[object, list[str]]:
    trap = _WarningTrap()
    log = logging.getLogger("ltx_core.loader.single_gpu_model_builder")
    log.addHandler(trap)
    try:
        model = _builder(path).build(device=device)
    finally:
        log.removeHandler(trap)
    return model, trap.messages


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else DEFAULT_PATH
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print(
            f"CANNOT RUN: the converted PrunaVAED decoder is not at {path}.\n"
            "Produce it with the converter (Nz-GGUF-Converter-LTX23's "
            "`convert-vae`) or point this script at the file, then re-run. "
            "Exit code 2 = not run; it is NOT a pass."
        )
        return 2

    print(f"PrunaVAED G2 self-check on {path}")
    print(f"file size: {os.path.getsize(path):,} bytes\n")

    cpu = torch.device("cpu")
    model, warnings_seen = _build(path, cpu)

    def check_no_uninitialized() -> str:
        offenders = [m for m in warnings_seen if "Uninitialized" in m]
        if offenders:
            raise ValueError(
                "the builder reported uninitialized parameters/buffers — the "
                f"file's keys do not match the module: {offenders}"
            )
        # Belt and braces: the builder RETURNS the meta model instead of moving
        # it when that happens, so verify no tensor stayed on meta either.
        stranded = [
            n for n, t in
            (*model.named_parameters(), *model.named_buffers())
            if str(t.device) == "meta"
        ]
        if stranded:
            raise ValueError(f"tensors left on the meta device: {stranded[:8]}")
        return f"{len(warnings_seen)} warning(s) total, 0 about uninitialized"

    def check_keys_and_count() -> str:
        from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader

        sd = SafetensorsModelStateDictLoader().load(path, None, cpu).sd
        if len(sd) != EXPECTED_STATE_DICT_KEYS:
            raise ValueError(
                f"the file holds {len(sd)} tensors, expected "
                f"{EXPECTED_STATE_DICT_KEYS}"
            )
        with torch.device("meta"):
            fresh = PrunedVideoDecoderConfigurator.from_config(
                _builder(path).model_config()
            )
        result = fresh.load_state_dict(sd, strict=False, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
            )
        return f"{len(sd)} tensors, no missing/unexpected keys"

    def check_parameter_count() -> str:
        total = sum(p.numel() for p in model.parameters())
        if total != EXPECTED_PARAMETER_COUNT:
            raise ValueError(
                f"parameter count {total:,} != {EXPECTED_PARAMETER_COUNT:,}"
            )
        return f"{total:,} parameters"

    def check_shapes() -> str:
        sd = model.state_dict()
        for key, want in _EXPECTED_SHAPES.items():
            got = tuple(sd[key].shape)
            if got != want:
                raise ValueError(f"{key}: {got} != {want}")
        kinds = [type(b) for b in model.up_blocks]
        if kinds != _EXPECTED_KINDS:
            raise ValueError(f"up_blocks kinds mismatch: {kinds}")
        return f"{len(_EXPECTED_SHAPES)} shapes + 11 block kinds"

    def check_norm3() -> str:
        for idx in (3, 6):
            norm3 = model.up_blocks[idx].norm3
            if not isinstance(norm3, ChannelLayerNorm3d):
                raise ValueError(
                    f"up_blocks.{idx}.norm3 is {type(norm3).__name__}, expected "
                    "ChannelLayerNorm3d"
                )
            if isinstance(norm3, nn.GroupNorm):
                raise ValueError(
                    f"up_blocks.{idx}.norm3 is a GroupNorm — the weights were "
                    "trained with a channel-only LayerNorm (§2.5)"
                )
        return "up_blocks.3 / up_blocks.6 both ChannelLayerNorm3d"

    def check_gpu_build() -> str:
        if not torch.cuda.is_available():
            raise ValueError("no CUDA device present — RE-RUN THIS ON THE GPU BOX")
        gpu_model, gpu_warnings = _build(path, torch.device("cuda:0"))
        offenders = [m for m in gpu_warnings if "Uninitialized" in m]
        if offenders:
            raise ValueError(f"GPU build reported {offenders}")
        total = sum(p.numel() for p in gpu_model.parameters())
        if total != EXPECTED_PARAMETER_COUNT:
            raise ValueError(f"GPU parameter count {total:,}")
        dev = {str(p.device) for p in gpu_model.parameters()}
        del gpu_model
        torch.cuda.empty_cache()
        return f"built on {sorted(dev)}"

    _run("G2-1 CPU build produces no 'Uninitialized parameters' warning",
         check_no_uninitialized)
    _run("G2-2 102 tensors, no missing/unexpected keys", check_keys_and_count)
    _run("G2-3 parameter count is exactly 345,006,256", check_parameter_count)
    # ASCII only in these printed names, deliberately: STDOUT on a Japanese
    # Windows console is cp932, and a section sign would come out as mojibake in
    # the log this gate gets pasted into.
    _run("G2-4 measured shapes and block kinds match the workorder 4.1 table",
         check_shapes)
    _run("G2-5 both projection norm3 are ChannelLayerNorm3d", check_norm3)
    _run("G2-6 GPU build succeeds too", check_gpu_build)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    print(f"\n{n_pass}/{len(_RESULTS)} checks passed")
    return 0 if n_pass == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
