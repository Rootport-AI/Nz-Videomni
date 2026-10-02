"""main._require_port_free: the double-launch guard run.ps1 relies on (exit code 3).

Binds real loopback sockets only; uvicorn is never started.
"""

from __future__ import annotations

import socket

import pytest

import main


def test_require_port_free_exits_when_port_is_listening_on_another_address():
    # The guard binds 0.0.0.0, so a server on 127.0.0.1 must still be detected.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        with pytest.raises(SystemExit) as excinfo:
            main._require_port_free(port)
        assert excinfo.value.code == main.EXIT_PORT_IN_USE


def test_require_port_free_passes_after_port_is_released():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
    main._require_port_free(port)
