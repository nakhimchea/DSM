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


@pytest.fixture(scope="session")
def tiny_tokenizer_json(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny byte-level BPE (km/en/zh) with the real pre-tokeniser and control tokens."""
    from tokenizer_fixtures import build_tiny_tokenizer

    return build_tiny_tokenizer(tmp_path_factory.mktemp("tiny_tokenizer"))


@pytest.fixture(scope="session")
def real_tokenizer_dir() -> Path:
    """The real 248K-token upstream tokenizer (pinned download, cached); skips when unavailable."""
    from tokenizer_fixtures import real_tokenizer_dir as fetch

    path = fetch()
    if path is None:
        pytest.skip("real upstream tokenizer files unavailable (offline?)")
    return path
