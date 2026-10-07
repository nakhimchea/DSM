"""Test-suite wide setup: offline environment, network guard, kernel sanity."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsm.env import OFFLINE_ENV, block_network  # noqa: E402

# Must happen before anything imports huggingface libraries.
for _key, _value in OFFLINE_ENV.items():
    os.environ.setdefault(_key, _value)


def pytest_configure(config: pytest.Config) -> None:
    import torch

    if not torch.cuda.is_available() and importlib.util.find_spec("fla") is not None:
        raise pytest.UsageError(
            "flash-linear-attention is installed on a CPU-only machine. The transformers reference "
            "implementation would try to use its CUDA kernels; uninstall it for the CPU test-suite."
        )


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest):
    """Every test runs with non-loopback networking disabled unless marked ``network``."""
    if request.node.get_closest_marker("network") is not None:
        yield
        return
    with block_network():
        yield


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT
