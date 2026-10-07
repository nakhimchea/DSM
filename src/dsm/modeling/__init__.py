"""DSM architecture (importing this package registers it with transformers).

After ``import dsm.modeling`` the standard transformers entry points work::

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("path/to/DSM1.0-0.8B")
"""

from __future__ import annotations

import contextlib

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_dsm import DSMConfig
from .modeling_dsm import (
    DSMAttention,
    DSMDecoderLayer,
    DSMForCausalLM,
    DSMGatedDeltaNet,
    DSMModel,
    DSMPreTrainedModel,
)

# Files that make up a self-contained ``trust_remote_code`` copy of the model code.
REMOTE_CODE_FILES = ("configuration_dsm.py", "modeling_dsm.py", "ops_dsm.py", "cache_dsm.py")
AUTO_MAP = {
    "AutoConfig": "configuration_dsm.DSMConfig",
    "AutoModel": "modeling_dsm.DSMModel",
    "AutoModelForCausalLM": "modeling_dsm.DSMForCausalLM",
}


def register() -> None:
    """Register DSM with the transformers Auto classes (idempotent)."""
    with contextlib.suppress(ValueError):  # already registered
        AutoConfig.register(DSMConfig.model_type, DSMConfig)
    for auto_cls, model_cls in ((AutoModel, DSMModel), (AutoModelForCausalLM, DSMForCausalLM)):
        with contextlib.suppress(ValueError):
            auto_cls.register(DSMConfig, model_cls)


register()

__all__ = [
    "AUTO_MAP",
    "REMOTE_CODE_FILES",
    "DSMAttention",
    "DSMConfig",
    "DSMDecoderLayer",
    "DSMForCausalLM",
    "DSMGatedDeltaNet",
    "DSMModel",
    "DSMPreTrainedModel",
    "register",
]
