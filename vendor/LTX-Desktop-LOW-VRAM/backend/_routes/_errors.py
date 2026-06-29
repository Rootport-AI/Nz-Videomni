"""Typed HTTP error for route functions."""


class HTTPError(Exception):
    """Raised by route functions to signal an HTTP error response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)
