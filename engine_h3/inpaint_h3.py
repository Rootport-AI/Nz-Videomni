"""MiniMax H3 Inpainting: the mask read, the token values and the record (台帳 §1-124).

計画 ``enumerated-questing-cupcake.md`` §4.6。The geometry, the token mask and
the paste-back are the torch-free ``engine_h3.inpaint_geometry``; the source
read, the fixed tokens and the self-check are Outpainting's
(``engine_h3.outpaint_h3``). This module holds the torch side the single
``H3Pipeline.generate`` adds for an Inpainting job:

* :func:`read_inpaint_mask`: the app's cut of the mask video (exactly F
  frames), each frame binarised (red >= 128 = white) and shrunk to the 32 x 32
  blocks on the spot (the full pixels are never stacked).
* :func:`apply_inpaint_tokens`: the geometry's token values (the right /
  bottom gaps' tokens generated) maxed with the mask's.
* :func:`inpaint_record`: ``done.h3.inpaint``.
"""

from __future__ import annotations

import numpy as np
import torch

from engine_h3.inpaint_geometry import CHUNK_FRAMES

#: The mask's rule as recorded in ``h3.inpaint.mask_rule``.
MASK_RULE = (
    "赤 128 以上を白・32 画素のブロックに 1 画素でも白なら生成・17 枚の塊の和集合・"
    "潜在は塊ごとに 5 個で末尾 3 個を捨てる"
)
#: The red threshold of the binarisation (the LTX mask contract's).
MASK_RED_MIN = 128


def read_inpaint_mask(path: str, num_frames: int, source_size: tuple[int, int], geom) -> tuple[np.ndarray, dict]:
    """The cut mask video -> ``(blocks bool [F, gh, gw], proof)``.

    Decoded with ``source_tail._decode_frames`` (PyAV, rgb24); each frame is
    binarised (``red >= 128``) and shrunk to the 32 x 32 blocks
    (``inpaint_geometry.block_mask_from_pixels``) as it is decoded. Raises
    ``RuntimeError`` (Japanese) when the file does not hold exactly
    ``num_frames`` frames or a frame is not ``source_size`` (``[W_s, H_s]``);
    a missing / unreadable file is ``source_tail``'s ``ValueError``.
    ``proof``: ``decoded_frames``, ``white_ratio`` (white pixels /
    ``F * H_s * W_s``) and ``token_ratio`` (the mean of
    ``inpaint_geometry.token_values_from_chunks``).
    マスク由来のトークンの面積率。右・下の隙間は含まない。
    """
    from engine_h3.constants import latent_frames  # noqa: PLC0415
    from engine_h3.inpaint_geometry import (  # noqa: PLC0415
        block_mask_from_pixels,
        chunk_block_masks,
        token_values_from_chunks,
    )
    from engine_h3.source_tail import _check_count, _decode_frames  # noqa: PLC0415

    frames = int(num_frames)
    sw, sh = int(source_size[0]), int(source_size[1])
    label = "Inpainting のマスク動画の切り出し"
    white = 0

    def convert(frame) -> np.ndarray:
        nonlocal white
        if (int(frame.width), int(frame.height)) != (sw, sh):
            raise RuntimeError(
                f"{label}の寸法が元動画と合いません（{frame.width}×{frame.height}。"
                f"元動画は {sw}×{sh}）: {path}"
            )
        pix = frame.to_ndarray(format="rgb24")[..., 0] >= MASK_RED_MIN
        white += int(np.count_nonzero(pix))
        return block_mask_from_pixels(pix, geom)

    out, _size, read = _decode_frames(path, frames, label, convert)
    try:
        _check_count(read, frames, label, path)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from None
    blocks = np.stack(out).astype(bool, copy=False)
    proof = {
        "decoded_frames": int(read),
        "white_ratio": float(white) / float(frames * sh * sw),
        # The mask's own tokens only (the right / bottom gaps' tokens are not counted).
        "token_ratio": float(token_values_from_chunks(
            chunk_block_masks(blocks, frames), latent_frames(frames)).mean()),
    }
    return blocks, proof


def apply_inpaint_tokens(video_fix, values_np: np.ndarray):
    """``(clean_v, values)`` -> ``(clean_v, max(values, mask))``.

    ``values``: ``outpaint_fix``'s geometry values ``[T', gh, gw]`` (0 fixed, 1
    generated; the tokens over the right / bottom gaps are 1). ``values_np``:
    ``inpaint_geometry.token_values_from_chunks``, the same shape. A shape
    mismatch is a ``RuntimeError``. The audio side is not touched.
    """
    clean_v, values = video_fix
    mask = torch.from_numpy(np.ascontiguousarray(values_np))
    if tuple(mask.shape) != tuple(values.shape):
        raise RuntimeError(
            f"inpaint: mask token values {list(mask.shape)}, expected {list(values.shape)}"
        )
    return clean_v, torch.maximum(values, mask.to(device=values.device, dtype=values.dtype)).contiguous()


def inpaint_record(
    geom, values: torch.Tensor, *, chunks, audio_strength: float, source_had_audio: bool,
    audio_latents_encoded: int, audio_output, feather_px: int, inject_clean: float, mask_proof: dict,
    cropped_to,
) -> dict:
    """``done.h3.inpaint``. ``values``: the token values after :func:`apply_inpaint_tokens`
    (``[T', gh, gw]``, 0 fixed / 1 generated); ``mask_proof``: :func:`read_inpaint_mask`'s
    proof, copied as is (its ``token_ratio`` is the mask's own: マスク由来のトークンの
    面積率。右・下の隙間は含まない). Called once the self-check has passed, so
    ``self_check`` is always true."""
    v = values.detach().to("cpu", torch.float32)
    t_lat = int(v.shape[0])
    generated = v == 1
    return {
        "canvas": [geom.canvas_w, geom.canvas_h],
        "pads": {"left": geom.pad_left, "right": geom.pad_right, "top": geom.pad_top, "bottom": geom.pad_bottom},
        "source": [geom.source_w, geom.source_h],
        "mask_rule": MASK_RULE,
        "chunk_frames": CHUNK_FRAMES,
        "chunks": [[int(a), int(b)] for a, b in chunks],
        "latent_frames": t_lat,
        "fixed_tokens_total": int((v == 0).sum()),
        "generated_tokens_total": int(generated.sum()),
        "generated_per_latent_frame": [int(x) for x in generated.reshape(t_lat, -1).sum(1).tolist()],
        "audio_strength": float(audio_strength),
        "source_had_audio": bool(source_had_audio),
        "audio_latents_encoded": int(audio_latents_encoded),
        "audio_output": audio_output,
        "feather_px": int(feather_px),
        "inject_clean": float(inject_clean),
        "self_check": {"fixed_latents_equal": True},
        "mask_proof": {
            "decoded_frames": int(mask_proof["decoded_frames"]),
            "white_ratio": float(mask_proof["white_ratio"]),
            "token_ratio": float(mask_proof["token_ratio"]),
        },
        "cropped_to": None if cropped_to is None else [int(cropped_to[0]), int(cropped_to[1])],
    }
