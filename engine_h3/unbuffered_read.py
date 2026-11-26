"""Read the text encoder's tensors past the OS file cache, through one reused staging buffer (台帳 1-303, 方向 A).

The text encoder reads about 24 GiB from its file once per encode stage (the
direct read of ``engine_h3.text_encoder``). Read through the OS cache, a machine
whose main memory cannot keep the file re-reads it from the disk every clip and
pays the cache's copy on top; ``FILE_FLAG_NO_BUFFERING`` reads go from the disk
straight into this module's staging buffer. The buffer is allocated once per
stage (``2**27`` bytes, pinned when the device is CUDA: a power of two, so the
pinned allocation is not rounded up) and reused for every tensor; the stage's
owner (``H3TextEncoder``) releases it. The caching host allocator keeps a
pinned block once it was taken, so the 128 MiB stay with the process and the
next stage reuses them.

One path to the device for every tensor: an empty uint8 tensor on the device,
the wanted bytes copied into it from the staging buffer one window at a time
(``copy_`` without ``non_blocking``: the copy is done before it returns, so the
buffer can be refilled), then ``.view(dtype).view(shape)``. A tensor larger than
the buffer is read in several windows; the bytes it ends up with are the same.
Nothing that points into the staging buffer is handed out.

``FILE_FLAG_NO_BUFFERING`` needs file offsets, lengths and the buffer address
to be multiples of the volume's sector size; 4,096 is a multiple of every
sector size in use, so each window is widened to 4,096-byte boundaries and the
buffer is used from its first 4,096-aligned address.

Whether the read works here is decided once, when the reader is made: the
staging buffer is allocated, the file is opened with ``FILE_FLAG_NO_BUFFERING``
and its first 4,096 bytes are read. Any failure there raises
:class:`UnbufferedReadUnavailable` with what failed (the operation and the
Windows error number and text) and the caller reads the buffered way instead. A
read that fails later, in the middle of a stage, is an ordinary ``OSError``:
the buffered read would fail on it too.

The file is opened and closed once per :meth:`UnbufferedReader.read_tensors`
call (one decoder layer, the embedding table, or the vision tower), as the
buffered read does. Messages name the file by its name only, never its folder.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from collections.abc import Iterable, Iterator

import torch

from engine.sft_quant.sft_reader import TORCH_DTYPES

logger = logging.getLogger(__name__)

#: File offsets, lengths and the buffer address are multiples of this (any sector size divides it).
ALIGN = 4096
#: The staging buffer: 128 MiB. The largest tensor of a decoder layer is 125 MiB.
STAGING_BYTES = 2**27
#: One ReadFile reads at most this much.
MAX_READ = 64 * 2**20

_IS_WINDOWS = sys.platform == "win32"

_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 0x00000001
_OPEN_EXISTING = 3
_FILE_FLAG_NO_BUFFERING = 0x20000000
_FILE_BEGIN = 0
_FORMAT_MESSAGE_FROM_SYSTEM = 0x00001000
_FORMAT_MESSAGE_IGNORE_INSERTS = 0x00000200
_LANG_EN_US = 0x0409
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class UnbufferedReadUnavailable(RuntimeError):
    """The unbuffered read cannot be used here; ``str()`` says what failed (ASCII, no folder names)."""


def aligned_window(offset: int, nbytes: int, align: int = ALIGN) -> tuple[int, int, int]:
    """``[offset, offset + nbytes)`` -> ``(start, end, shift)``: ``start`` rounded down and
    ``end`` rounded up to multiples of ``align``, ``shift = offset - start`` (where the
    wanted bytes begin in the window)."""
    if offset < 0 or nbytes < 0:
        raise ValueError(f"aligned_window: offset {offset} / nbytes {nbytes} must not be negative")
    start = offset // align * align
    end = -(-(offset + nbytes) // align) * align
    return start, end, offset - start


def _ascii(text: str) -> str:
    """Keep a message ASCII on one line (non-ASCII characters escaped)."""
    return " ".join(str(text).split()).encode("ascii", "backslashreplace").decode("ascii")


_K32 = None


def _kernel32():
    """A private ``kernel32`` with every signature this module calls declared (the
    process-wide ``ctypes.windll.kernel32`` is left as it is)."""
    global _K32
    if _K32 is None:
        from ctypes import wintypes

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        k.CreateFileW.restype = wintypes.HANDLE
        k.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                               ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        k.ReadFile.restype = wintypes.BOOL
        k.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong),
                                       wintypes.DWORD]
        k.SetFilePointerEx.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        k.FormatMessageW.argtypes = [wintypes.DWORD, wintypes.LPCVOID, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.LPWSTR, wintypes.DWORD, wintypes.LPVOID]
        k.FormatMessageW.restype = wintypes.DWORD
        _K32 = k
    return _K32


def _win_error_text(code: int) -> str:
    """``error N (text)``; the text is the system's US English one, left out when the
    system has none in English (a localized text would not be ASCII)."""
    try:
        buf = ctypes.create_unicode_buffer(512)
        n = _kernel32().FormatMessageW(
            _FORMAT_MESSAGE_FROM_SYSTEM | _FORMAT_MESSAGE_IGNORE_INSERTS, None, code, _LANG_EN_US, buf, len(buf), None,
        )
        text = " ".join(buf.value[:n].split()) if n else ""
    except Exception:  # noqa: BLE001 -- the number alone is still the fact
        text = ""
    if text and text.isascii():
        return f"error {code} ({text})"
    return f"error {code}"


def _allocate_staging(nbytes: int, pinned: bool) -> torch.Tensor:
    """The staging buffer, exactly ``nbytes`` (not padded: a padded 2**27 would be
    rounded up to 2**28 by the pinned allocator); used from its first aligned byte."""
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned)


def _win_failure(operation: str) -> OSError:
    code = ctypes.get_last_error()
    return OSError(f"unbuffered read: {operation} failed: {_win_error_text(code)}")


class UnbufferedReader:
    """``FILE_FLAG_NO_BUFFERING`` reads of one safetensors file through one reused staging buffer.

    Made once per text-encoder stage; :meth:`close` lets the buffer go (safe to
    call twice, never raises). Construction raises
    :class:`UnbufferedReadUnavailable` when the read cannot be used here.
    """

    def __init__(self, path: str, device, *, staging_bytes: int = STAGING_BYTES) -> None:
        if staging_bytes < 3 * ALIGN:
            raise ValueError(f"unbuffered read: staging_bytes {staging_bytes} is below {3 * ALIGN}")
        self.path = str(path)
        self.name = os.path.basename(self.path)
        self.device = torch.device(device)
        self._staging: torch.Tensor | None = None
        self._window: torch.Tensor | None = None
        self._usable = 0
        if not _IS_WINDOWS:
            raise UnbufferedReadUnavailable(f"unbuffered read needs Windows (sys.platform is {sys.platform!r})")
        pinned = self.device.type == "cuda"
        try:
            staging = _allocate_staging(int(staging_bytes), pinned)
        except Exception as exc:  # noqa: BLE001 -- any failure to get the buffer means the buffered read
            first = str(exc).splitlines()[0] if str(exc) else ""
            raise UnbufferedReadUnavailable(_ascii(
                f"unbuffered read: allocating the {int(staging_bytes)}-byte staging buffer ({'pinned' if pinned else 'not pinned'}) "
                f"failed: {type(exc).__name__}: {first}"
            )) from None
        shift = (-int(staging.data_ptr())) % ALIGN
        usable = (int(staging_bytes) - shift) // ALIGN * ALIGN
        self._staging = staging
        self._window = staging[shift:shift + usable]
        self._usable = usable
        step = "getting the size"
        try:
            try:
                self._file_size = os.path.getsize(self.path)
            except OSError as exc:   # str(exc) would carry the full path
                raise OSError(
                    f"unbuffered read: getting the size of {self.name} failed: {type(exc).__name__} "
                    f"errno {exc.errno} winerror {getattr(exc, 'winerror', None)}"
                ) from None
            step = "opening"
            handle = self._open_handle()
            try:
                step = "the trial read of the first 4096 bytes"
                self._fill(handle, 0, ALIGN)
                step = "closing"   # only after the trial read succeeded
            finally:
                self._close_handle(handle)
        except Exception as exc:  # noqa: BLE001 -- any failure of the trial means the buffered read
            self.close()
            if isinstance(exc, OSError) and str(exc).startswith("unbuffered read: "):
                reason = str(exc)   # this module's own message: operation, error number and text, file name
            else:
                first = str(exc).splitlines()[0] if str(exc) else ""
                reason = (f"unbuffered read: {step} of {self.name} failed: {type(exc).__name__}: "
                          f"{first.replace(self.path, self.name)}")
            raise UnbufferedReadUnavailable(_ascii(reason)) from None

    # -- Win32 ------------------------------------------------------------------

    def _open_handle(self):
        handle = _kernel32().CreateFileW(
            self.path, _GENERIC_READ, _FILE_SHARE_READ, None, _OPEN_EXISTING, _FILE_FLAG_NO_BUFFERING, None,
        )
        if handle is None or handle == 0 or handle == _INVALID_HANDLE_VALUE:
            raise _win_failure(f"CreateFileW (FILE_FLAG_NO_BUFFERING) of {self.name}")
        return handle

    def _close_handle(self, handle) -> None:
        if not _kernel32().CloseHandle(handle):
            logger.warning("%s", _ascii(str(_win_failure(f"CloseHandle of {self.name}"))))

    def _read_file(self, handle, address: int, nbytes: int) -> int:
        """One ``ReadFile`` of ``nbytes`` into ``address``; returns the bytes read."""
        from ctypes import wintypes

        got = wintypes.DWORD(0)
        if not _kernel32().ReadFile(handle, ctypes.c_void_p(address), nbytes, ctypes.byref(got), None):
            raise _win_failure(f"ReadFile of {nbytes} bytes of {self.name}")
        return int(got.value)

    def _fill(self, handle, start: int, length: int) -> int:
        """Read the aligned ``[start, start + length)`` of the file into the window's
        front; returns the bytes read (fewer only where the file ends)."""
        if not _kernel32().SetFilePointerEx(handle, start, None, _FILE_BEGIN):
            raise _win_failure(f"SetFilePointerEx to {start} in {self.name}")
        address = int(self._window.data_ptr())
        got = 0
        while got < length:
            want = min(MAX_READ, length - got)
            n = self._read_file(handle, address + got, want)
            if n != want:
                # Short (or not a multiple of 4,096) only where the file ends.
                if start + got + n != self._file_size:
                    raise OSError(
                        f"unbuffered read: ReadFile returned {n} of {want} bytes at offset {start + got} "
                        f"of {self.name}, before the end of the file ({self._file_size} bytes)"
                    )
                got += n
                break
            got += n
        return got

    # -- read -------------------------------------------------------------------

    def read_tensors(self, path: str, keys: Iterable[str], header, device) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``(key, tensor on device)`` for each of ``keys``, in file (data_offsets) order.

        The arguments are those of ``engine.sft_quant.sft_reader.read_tensors``
        plus ``device``; ``path`` must be the file this reader was made for.
        Each tensor owns its own device memory and keeps the file's dtype and
        shape. The file is opened once for the call and closed when it ends.
        """
        if self._window is None:
            raise RuntimeError("unbuffered read: the reader was closed")
        if os.path.normcase(os.path.abspath(str(path))) != os.path.normcase(os.path.abspath(self.path)):
            raise ValueError(f"unbuffered read: this reader reads {self.name}, not {os.path.basename(str(path))}")
        device = torch.device(device)
        infos = [(key, header.tensors[key]) for key in keys]
        infos.sort(key=lambda item: item[1].data_offsets[0])
        window = self._window
        handle = self._open_handle()
        try:
            for key, info in infos:
                dtype = TORCH_DTYPES.get(info.dtype)
                if dtype is None:
                    raise ValueError(
                        f"unbuffered read: '{key}' has dtype {info.dtype}, which this reader "
                        f"does not handle ({', '.join(TORCH_DTYPES)})"
                    )
                begin, end = info.data_offsets
                nbytes = end - begin
                out = torch.empty(nbytes, dtype=torch.uint8, device=device)
                pos = header.data_base + begin
                stop = pos + nbytes
                done = 0
                while pos < stop:
                    take = min(stop - pos, self._usable - pos % ALIGN)
                    start, finish, shift = aligned_window(pos, take)
                    got = self._fill(handle, start, finish - start)
                    if got < shift + take:
                        raise OSError(
                            f"unbuffered read: short read for '{key}' in {self.name}: "
                            f"{got} of {shift + take} bytes at offset {start}"
                        )
                    out[done:done + take].copy_(window[shift:shift + take])
                    pos += take
                    done += take
                # out is 1-D, so a 0-dim tensor goes 1-D (1 element) -> () here.
                yield key, out.view(dtype).view(tuple(info.shape))
                del out
        finally:
            self._close_handle(handle)

    # -- close ------------------------------------------------------------------

    def close(self) -> None:
        """Let the staging buffer go (the caching host allocator keeps a pinned block for
        the next reader). Safe to call twice."""
        self._window = None
        self._staging = None
        self._usable = 0

    @property
    def closed(self) -> bool:
        return self._window is None
