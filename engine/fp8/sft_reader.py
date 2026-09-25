"""Read safetensors tensors one by one with seek + readinto (§3-167 B-1).

Why not the usual readers: ``safetensors.safe_open`` and ``torch.frombuffer``
over a memory map would map the whole ~29 GB transformer file, and on Windows
that is charged against the commit limit — exactly the failure the plan's
§2 ruling forbids. Here every tensor gets its own freshly allocated uint8
buffer, filled straight from the file, and reinterpreted in place; nothing
larger than the one tensor being read is ever held on the reader's behalf.

The header is parsed by the torch-free ``sft_fp8_format.read_header`` (the one
place that validates offsets against the file size and element counts).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch

#: safetensors dtype spelling -> torch dtype. Anything else is refused.
TORCH_DTYPES: dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F16": torch.float16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "U8": torch.uint8,
}


def read_tensors(
    path: str, keys: Iterable[str], header=None
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, tensor)`` for each of ``keys``, in file (data_offsets) order.

    Each tensor is a fresh CPU tensor that owns its bytes. ``header`` is an
    ``sft_fp8_format.Header``; when omitted it is read from ``path``. A key
    missing from the header, an unsupported dtype, or a short read raises.
    """
    if header is None:
        import sft_fp8_format

        header = sft_fp8_format.read_header(path)
    infos = [(key, header.tensors[key]) for key in keys]
    infos.sort(key=lambda item: item[1].data_offsets[0])
    with open(path, "rb") as f:
        for key, info in infos:
            dtype = TORCH_DTYPES.get(info.dtype)
            if dtype is None:
                raise ValueError(
                    f"sft_reader: '{key}' has dtype {info.dtype}, which this reader "
                    f"does not handle ({', '.join(TORCH_DTYPES)})"
                )
            start, end = info.data_offsets
            nbytes = end - start
            buf = torch.empty(nbytes, dtype=torch.uint8)
            if nbytes:
                f.seek(header.data_base + start)
                view = memoryview(buf.numpy())
                got = 0
                while got < nbytes:
                    n = f.readinto(view[got:])
                    if not n:
                        break
                    got += n
                if got != nbytes:
                    raise OSError(
                        f"sft_reader: short read for '{key}' in {path}: "
                        f"{got} of {nbytes} bytes"
                    )
            # buf is 1-D, so a 0-dim tensor goes 1-D (1 element) -> () here.
            yield key, buf.view(dtype).view(tuple(info.shape))
