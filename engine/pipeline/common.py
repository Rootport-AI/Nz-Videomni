"""Shared helpers and primitives for LTX video pipeline wrappers."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from engine.pipeline.utils import AudioOrNone, TilingConfigType


def default_tiling_config(
    spatial_tile_size: int = 0,
    temporal_tile_size: int = 0,
) -> TilingConfigType:
    """Return a TilingConfig, optionally overriding the spatial/temporal tile sizes.

    Passing 0 (the default) for an axis uses that axis of the library's
    ``TilingConfig.default()``.
    """
    from ltx_core.model.video_vae import TilingConfig, SpatialTilingConfig, TemporalTilingConfig

    if spatial_tile_size <= 0 and temporal_tile_size <= 0:
        return TilingConfig.default()

    default = TilingConfig.default()
    spatial = (
        SpatialTilingConfig(
            tile_size_in_pixels=max(64, (spatial_tile_size // 32) * 32),
            tile_overlap_in_pixels=64,
        )
        if spatial_tile_size > 0
        else default.spatial_config
    )
    temporal = (
        TemporalTilingConfig(
            tile_size_in_frames=max(16, (temporal_tile_size // 8) * 8),
            tile_overlap_in_frames=24,
        )
        if temporal_tile_size > 0
        else default.temporal_config
    )
    return TilingConfig(spatial_config=spatial, temporal_config=temporal)


def video_chunks_number(num_frames: int, tiling_config: TilingConfigType | None) -> int:
    from ltx_core.model.video_vae import get_video_chunks_number

    return int(get_video_chunks_number(num_frames, tiling_config))


def encode_video_output(
    video: torch.Tensor | Iterator[torch.Tensor],
    audio: AudioOrNone,
    fps: int,
    output_path: str,
    video_chunks_number_value: int,
) -> None:
    from ltx_pipelines.utils.media_io import encode_video

    encode_video(
        video=video,
        fps=fps,
        audio=audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number_value,
    )


def iter_video_conditioning_cpu(
    *,
    video_path: str,
    height: int,
    width: int,
    frame_cap: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Iterator[torch.Tensor]:
    """Frame-by-frame generator behind :func:`load_video_conditioning_cpu`.

    Yields ONE (1,C,1,H,W) CPU tensor per decoded frame, in decode order, having
    run the exact same per-frame ops on the exact same device as the whole-tensor
    loader below — ``decode_video_from_file`` -> ``resize_and_center_crop`` on
    float32 -> ``normalize_latent`` to ``dtype`` -> ``.to("cpu")``. The loader is
    literally ``torch.cat(list(this), dim=2)``, so the two are byte-identical by
    construction.

    A separate generator because of the clip-wise chain reference: a chain needs
    one DIFFERENT overlapping pixel window per stage-1 segment from one long
    reference video, and materialising the whole video just to slice it would
    cost the full ``total_px`` (capped by ``api/models.py``'s
    ``MAX_CHAIN_TOTAL_PIXEL_FRAMES``) on CPU at once. ``chain_pipeline``'s window
    generator consumes this stream lazily instead, holding at most
    ``max(clip_frames)`` frames.

    The GPU-side per-frame temporaries are released BEFORE the yield, so a slow
    consumer never pins a decoded frame on the device.
    """
    from ltx_pipelines.utils.media_io import (
        decode_video_from_file,
        normalize_latent,
        resize_and_center_crop,
    )

    for f in decode_video_from_file(path=video_path, frame_cap=frame_cap, device=device):
        # Same ops, same device (GPU), same dtype flow as load_video_conditioning.
        frame = resize_and_center_crop(f.to(torch.float32), height, width)
        frame = normalize_latent(frame, device, dtype)
        frame_cpu = frame.to("cpu")
        del f, frame
        yield frame_cpu


def load_video_conditioning_cpu(
    *,
    video_path: str,
    height: int,
    width: int,
    frame_cap: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Numerics-preserving, low-VRAM twin of ``media_io.load_video_conditioning``.

    The installed ``load_video_conditioning`` decodes every frame onto ``device``
    (GPU), preprocesses it there, and ``torch.cat``s the growing (1,C,F,H,W)
    tensor **on the GPU** — allocating a fresh full-size buffer per frame while
    the previous one is still live, so the total allocated volume grows with the
    SQUARE of the frame count (F(F+1)/2 frame-sized buffers). At 640x384x257
    that is a measured 46.8GB of reserved VRAM for a tensor whose final size is
    ~380MB; on Windows it does not even fail, it silently spills into shared
    memory and drags the whole job down.

    This twin runs the EXACT same per-frame ops on the EXACT same device (GPU) in
    the same dtype flow — ``decode_video_from_file`` -> ``resize_and_center_crop``
    on float32 -> ``normalize_latent`` to ``dtype`` — but moves each processed
    frame to CPU immediately and assembles the (1,C,F,H,W) tensor on CPU with a
    single ``torch.cat``. The GPU holds at most one frame at a time. Because the
    resize/normalize math still runs on the GPU, the per-frame values are
    bit-identical to the installed loader; the CPU copy is a pure device transfer
    (no arithmetic), and ``torch.cat`` is a deterministic copy, so the assembled
    tensor is byte-identical to the installed loader's output apart from residing
    on CPU. Measured on the same 640x384x257 case: reserved 46,794MB -> 20MB,
    9.45s -> 0.90s, output ``torch.equal`` with the installed loader.

    Callers that hand the result to ``VideoEncoder.tiled_encode`` can pass the
    CPU tensor straight through (tiled_encode streams tiles back to the GPU one
    at a time); callers that call the encoder directly must ``.to(device)`` first.

    Behaviour difference on a 0-frame video: the installed loader returns ``None``
    (its accumulator never gets a first frame), whereas this twin raises from
    ``torch.cat`` on an empty list. Both are failure paths for callers that
    require pixels; no production caller feeds a 0-frame source.

    The per-frame work lives in :func:`iter_video_conditioning_cpu`; this is the
    single ``torch.cat`` over it, so the two cannot drift apart.
    """
    return torch.cat(
        list(
            iter_video_conditioning_cpu(
                video_path=video_path,
                height=height,
                width=width,
                frame_cap=frame_cap,
                dtype=dtype,
                device=device,
            )
        ),
        dim=2,
    )


def decode_mask_video(
    path: str,
    *,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
    threshold: int = 128,
) -> torch.Tensor:
    """Decode an inpainting mask video to ``(F, 1, height, width)`` uint8 (0/255).

    The mask-only sibling of :func:`load_video_conditioning_cpu`, and the list of
    things it deliberately does NOT do is the whole specification:

    * **it does not resize.** A mask is a statement about which pixels of a
      SPECIFIC picture get repainted; rescaling it would move the boundary by a
      sub-pixel amount that the caller cannot see and cannot correct. The
      contract (``Docs/INPAINTING_DESIGN.md`` §6.2) is that the mask arrives at
      the source's own resolution or the request is refused with a 422, so a
      size mismatch here means the API guard was bypassed and this raises;
    * **it does not normalise.** ``load_video_conditioning_cpu`` maps pixels to
      [-1, 1] for the VAE; there is no VAE in this path. The values that come
      out are 0 and 255 and nothing else;
    * **it does not interpolate between the two levels.** The plugin writes the
      mask through Media Foundation and the app re-encodes nothing, but H.264's
      4:2:0 chroma and its deblocking filter still leave a soft grey ring around
      every edge. ``threshold`` (128, the same value the green-fill filtergraph's
      ``lut`` uses — one binarisation rule, spelt twice because ffmpeg and torch
      cannot share a constant) is what turns that ring back into a decision.

    ``decode_video_from_file`` hands back one ``(1, H, W, C)`` uint8 tensor per
    frame, with C == 3 even for a grey source (it goes through
    ``frame.to_rgb()``), so the RED channel is taken and the other two are
    dropped: for a genuinely grey mask all three are equal. (The ``lut`` in
    ``video_io.fill_mask_green_mp4`` thresholds the same red channel, taken
    with ``extractplanes=r``, so a mask that is not grey binarises the same
    way on both sides.) The frames are NOT put through
    ``resize_and_center_crop`` the way ``_load_canvas_pixels_u8`` puts the canvas
    — that helper's whole job is to make a video fit a target size, which is
    exactly what must not happen here.

    ``num_frames`` is both the decode cap and an assertion: a mask that runs
    short would silently leave the tail of the clip unrepainted, so a shortfall
    raises rather than being padded.
    """
    from ltx_pipelines.utils.media_io import decode_video_from_file

    if num_frames <= 0:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")

    planes: list[torch.Tensor] = []
    for f in decode_video_from_file(path=path, frame_cap=num_frames, device=device):
        if f.ndim != 4 or f.shape[0] != 1:
            raise ValueError(
                f"mask video {path}: expected one (1, H, W, C) frame per step, "
                f"got {tuple(f.shape)}"
            )
        got_h, got_w = int(f.shape[1]), int(f.shape[2])
        if (got_h, got_w) != (height, width):
            raise ValueError(
                f"mask video {path} is {got_w}x{got_h} but the source is "
                f"{width}x{height}; the mask is never resized "
                "(Docs/INPAINTING_DESIGN.md §6.2)"
            )
        # (1, H, W, C) -> (H, W), red channel only, thresholded to 0/255.
        red = f[0, :, :, 0]
        planes.append((red >= threshold).to(torch.uint8).mul_(255).cpu())
        del f, red

    if len(planes) != num_frames:
        raise ValueError(
            f"mask video {path} decoded {len(planes)} frames but the job needs "
            f"{num_frames}"
        )
    return torch.stack(planes, dim=0).unsqueeze(1)
