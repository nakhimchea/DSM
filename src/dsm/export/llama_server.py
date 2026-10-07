"""Launch ``llama-server`` for a DSM GGUF, bound to localhost and fully offline.

    dsm export serve --gguf exports/DSM1.0-0.8B-Q8_0.gguf --port 8080

The server exposes an OpenAI-compatible API (``/v1/chat/completions`` with
tools) and uses the chat template stored in the GGUF (``--jinja``). DSM is
fast-only, so reasoning extraction is disabled. Build llama.cpp without network
code (``-DLLAMA_CURL=OFF``, see scripts/build_llama_cpp.sh).
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

LOOPBACK = ("127.0.0.1", "localhost", "::1")


@dataclass
class ServerConfig:
    gguf: str
    host: str = "127.0.0.1"
    port: int = 8080
    ctx: int = 8192
    gpu_layers: int = 99
    parallel: int = 2
    threads: int | None = None
    api_key: str | None = None
    webui: bool = False
    extra: list[str] = field(default_factory=list)
    allow_remote_bind: bool = False
    # CPU weight repacking. At the pinned llama.cpp commit it corrupts this architecture's
    # outputs when two sequences are decoded in one batch (docs/11), so by default it is used
    # only when a single request runs at a time (it speeds up CPU prompt processing).
    repack: bool | None = None


def server_command(cfg: ServerConfig, binary: str | Path) -> list[str]:
    if cfg.host not in LOOPBACK and not cfg.allow_remote_bind:
        raise ValueError(
            f"refusing to bind {cfg.host!r}: DSM serves on localhost only (use a reverse proxy with auth to share)"
        )
    cmd = [
        str(binary),
        "-m", str(cfg.gguf),
        "--host", cfg.host,
        "--port", str(cfg.port),
        "-c", str(cfg.ctx),
        "-ngl", str(cfg.gpu_layers),
        "-np", str(cfg.parallel),
        "--jinja",
        "--reasoning-format", "none",
        "--offline",
    ]  # fmt: skip
    if not cfg.webui:
        cmd.append("--no-webui")
    if not (cfg.repack if cfg.repack is not None else cfg.parallel == 1):
        cmd.append("--no-repack")
    if cfg.threads:
        cmd += ["-t", str(cfg.threads)]
    if cfg.api_key:
        cmd += ["--api-key", cfg.api_key]
    return cmd + list(cfg.extra)


def wait_healthy(host: str, port: int, timeout_s: float = 120.0) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with opener.open(f"http://{host}:{port}/health", timeout=2) as response:
                if json.loads(response.read().decode()).get("status") == "ok":
                    return True
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    return False


def start(cfg: ServerConfig, binary: str | Path, log_file: str | Path | None = None) -> subprocess.Popen:
    """Start the server in the background (offline environment) and wait until it is healthy."""
    from dsm.export.gguf import offline_env

    log = open(log_file, "ab") if log_file else subprocess.DEVNULL  # noqa: SIM115 - handed to the child process
    process = subprocess.Popen(
        server_command(cfg, binary), stdout=log, stderr=subprocess.STDOUT, env=offline_env()
    )
    if not wait_healthy(cfg.host, cfg.port):
        process.terminate()
        raise RuntimeError(f"llama-server did not become healthy (see {log_file or 'its output'})")
    return process
