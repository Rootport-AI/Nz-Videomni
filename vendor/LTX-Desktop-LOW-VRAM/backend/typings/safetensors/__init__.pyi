from __future__ import annotations

from types import TracebackType


class _SafeOpenReader:
    def __enter__(self) -> _SafeOpenReader: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...
    def metadata(self) -> dict[str, str] | None: ...


def safe_open(filename: str, framework: str, device: str = ...) -> _SafeOpenReader: ...
