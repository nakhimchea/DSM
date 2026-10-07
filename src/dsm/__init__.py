"""DSM - a compact, offline, trilingual (Khmer / English / Chinese) language model.

The top-level package is intentionally light: it does not import torch or
transformers, so data-processing workers stay small. Importing
``dsm.modeling`` registers the DSM architecture with the transformers Auto
classes (``AutoConfig`` / ``AutoModelForCausalLM``).
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent


def repo_root() -> Path:
    """Best-effort path of the repository checkout (for default config paths)."""
    return PACKAGE_ROOT.parent.parent


__all__ = ["__version__", "PACKAGE_ROOT", "repo_root"]
