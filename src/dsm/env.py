"""Offline / no-telemetry environment for every DSM entry point.

DSM is meant to run on air-gapped machines. ``enforce_offline()`` sets the
environment flags understood by the Hugging Face libraries, vLLM, Ollama and
common experiment trackers so that nothing tries to reach the network.
``block_network()`` goes further and makes any non-loopback socket connection
raise, which is how ``dsm audit offline`` proves that inference is local.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import socket
from collections.abc import Iterator

OFFLINE_ENV: dict[str, str] = {
    # Hugging Face hub / transformers / datasets
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    # generic opt-out honoured by many tools
    "DO_NOT_TRACK": "1",
    # serving engines
    "VLLM_NO_USAGE_STATS": "1",
    "OLLAMA_NO_CLOUD": "1",
    # experiment trackers
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
}


def enforce_offline(override: bool = False) -> dict[str, str]:
    """Set the offline flags (call before importing huggingface libraries).

    Existing values are kept unless ``override`` is true, so a user can still
    opt in to downloads explicitly (e.g. ``HF_HUB_OFFLINE=0 dsm ...``) for the
    one-time fetch scripts. Returns the flags that were applied.
    """
    applied: dict[str, str] = {}
    for key, value in OFFLINE_ENV.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def offline_status() -> dict[str, str | None]:
    """Current value of every offline flag (``None`` when unset)."""
    return {key: os.environ.get(key) for key in OFFLINE_ENV}


def _is_local(address: object) -> bool:
    if isinstance(address, (str, bytes)):  # AF_UNIX socket path
        return True
    if not (isinstance(address, tuple) and address):
        return False
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode()
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


class NetworkBlockedError(OSError):
    """Raised when code tries to open a non-loopback connection while blocked."""


@contextlib.contextmanager
def block_network() -> Iterator[None]:
    """Forbid every non-loopback socket connection inside the ``with`` block."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo

    def guarded_connect(sock: socket.socket, address: object) -> None:
        if not _is_local(address):
            raise NetworkBlockedError(f"network access blocked by dsm: connect({address!r})")
        return original_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: object) -> int:
        if not _is_local(address):
            raise NetworkBlockedError(f"network access blocked by dsm: connect_ex({address!r})")
        return original_connect_ex(sock, address)

    def guarded_create_connection(address: object, *args: object, **kwargs: object) -> socket.socket:
        if not _is_local(address):
            raise NetworkBlockedError(f"network access blocked by dsm: create_connection({address!r})")
        return original_create_connection(address, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_getaddrinfo(host: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if host not in (None, "localhost") and not _is_local((host,)):
            raise NetworkBlockedError(f"network access blocked by dsm: getaddrinfo({host!r})")
        return original_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = original_getaddrinfo  # type: ignore[assignment]
