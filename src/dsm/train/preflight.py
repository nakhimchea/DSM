"""``dsm preflight`` - check the training machine before a long run.

Checks (FAIL stops a run, WARN deserves a look):

* Python / torch / transformers versions (transformers pinned to 5.19.*);
* a CUDA GPU with compute capability >= 8.0 (bf16) and enough memory;
* flash-linear-attention importable AND numerically equal to the exact torch
  path on a small Gated DeltaNet problem (the torch path is ~10x slower);
* Linux or WSL2 (Triton kernels);
* free disk space for checkpoints;
* offline environment variables;
* an estimate of peak GPU memory for a training config.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

GIB = 2**30


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


def estimate_memory_gb(
    n_params: int,
    n_trainable: int,
    *,
    seq_len: int,
    micro_batch: int,
    hidden: int,
    layers: int,
    vocab: int,
    loss_chunk: int = 1024,
    checkpointing: bool = True,
) -> float:
    """Rough peak GPU memory of a training step (fp32 master weights + AdamW, bf16 activations)."""
    weights = 4 * n_params
    grads_and_adam = 12 * n_trainable
    tokens = seq_len * micro_batch
    if checkpointing:
        # one layer input kept per layer + one layer recomputed in full (~34 hidden-sized tensors)
        activations = 2 * tokens * hidden * (layers + 34)
    else:
        activations = 2 * tokens * hidden * 34 * layers
    logits = 4 * loss_chunk * vocab * 3  # fp32 logits chunk, softmax and gradient
    overhead = 1.0 * GIB  # CUDA context, allocator fragmentation
    return (weights + grads_and_adam + activations + logits + overhead) / GIB


def _gdn_kernel_check() -> Check:
    import torch

    from dsm.modeling.ops_dsm import gated_delta_rule

    torch.manual_seed(0)
    b, t, h, d = 1, 128, 2, 64
    device = "cuda"
    q, k = (torch.randn(b, t, h, d, device=device) for _ in range(2))
    v = torch.randn(b, t, h, d, device=device)
    g = -torch.rand(b, t, h, device=device)
    beta = torch.rand(b, t, h, device=device)
    common = {"initial_state": None, "output_final_state": False, "single_step": False}
    ref, _ = gated_delta_rule(q, k, v, g, beta, backend="torch", **common)
    out, _ = gated_delta_rule(q, k, v, g, beta, backend="fla", **common)
    err = float((out.float() - ref.float()).abs().max())
    if err < 5e-2:
        return Check(
            "fla kernels", "PASS", f"chunked Gated DeltaNet matches the torch path (max err {err:.1e})"
        )
    return Check("fla kernels", "FAIL", f"fla output differs from the torch path (max err {err:.2e})")


def run_checks(config_path: str | None = None, output_dir: str = ".") -> list[Check]:
    checks: list[Check] = []
    py = sys.version_info
    checks.append(Check("python", "PASS" if py >= (3, 10) else "FAIL", platform.python_version()))
    system = platform.system()
    wsl = "microsoft" in platform.release().lower()
    checks.append(
        Check(
            "os",
            "PASS" if system == "Linux" else "WARN",
            f"{system}{' (WSL2)' if wsl else ''}: Triton kernels need Linux/WSL2",
        )
    )
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover
        return [*checks, Check("packages", "FAIL", str(exc))]
    checks.append(Check("torch", "PASS", torch.__version__))
    checks.append(
        Check(
            "transformers",
            "PASS" if transformers.__version__.startswith("5.19") else "WARN",
            f"{transformers.__version__} (pinned 5.19.*: the cache API changes between releases)",
        )
    )
    gpu_mem = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_mem = props.total_memory / GIB
        major, minor = torch.cuda.get_device_capability(0)
        checks.append(
            Check(
                "gpu",
                "PASS" if major >= 8 else "FAIL",
                f"{props.name}, compute {major}.{minor}, {gpu_mem:.1f} GiB",
            )
        )
        if importlib.util.find_spec("fla") is None:
            checks.append(
                Check(
                    "fla kernels",
                    "FAIL",
                    "pip install flash-linear-attention (dsm-llm[kernels]); torch path is ~10x slower",
                )
            )
        else:
            try:
                checks.append(_gdn_kernel_check())
            except Exception as exc:  # noqa: BLE001 - report any kernel failure
                checks.append(Check("fla kernels", "FAIL", f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(Check("gpu", "FAIL", "no CUDA GPU visible (CPU is fine for tests and data work only)"))
    free = (
        shutil.disk_usage(Path(output_dir).resolve() if Path(output_dir).exists() else Path.cwd()).free / GIB
    )
    checks.append(
        Check(
            "disk",
            "PASS" if free >= 100 else "WARN",
            f"{free:.0f} GiB free (checkpoints of a 0.8B model are ~9 GiB each with optimizer state)",
        )
    )
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    checks.append(
        Check(
            "offline env",
            "PASS" if offline else "WARN",
            "HF_HUB_OFFLINE=1" if offline else "HF_HUB_OFFLINE is not 1 (dsm commands set it themselves)",
        )
    )
    if config_path:
        checks.append(_config_check(config_path, gpu_mem))
    return checks


def _config_check(config_path: str, gpu_mem: float) -> Check:
    import json

    from dsm.train.config import load_config

    cfg = load_config(config_path)
    model_config = Path(cfg.model) / "config.json" if cfg.model else None
    if model_config and model_config.is_file():
        mc = json.loads(model_config.read_text())
    elif cfg.model_config:
        from dsm.utils import load_yaml

        mc = load_yaml(cfg.model_config)
    else:
        return Check("memory estimate", "WARN", "model config not found")
    n_params = _count_params(mc)
    embed = mc.get("vocab_size", 248320) * mc.get("hidden_size", 1024)
    trainable = {"all": n_params, "no_embeddings": n_params - embed}.get(cfg.trainable, embed)
    need = estimate_memory_gb(
        n_params,
        trainable,
        seq_len=cfg.seq_len,
        micro_batch=cfg.micro_batch_size,
        hidden=mc.get("hidden_size", 1024),
        layers=mc.get("num_hidden_layers", 24),
        vocab=mc.get("vocab_size", 248320),
        loss_chunk=cfg.loss_chunk_size,
        checkpointing=cfg.gradient_checkpointing,
    )
    status = "PASS" if gpu_mem and need < 0.9 * gpu_mem else ("WARN" if not gpu_mem else "FAIL")
    return Check(
        "memory estimate",
        status,
        f"~{need:.1f} GiB peak for {config_path} ({n_params / 1e6:.0f}M params, {cfg.micro_batch_size}x{cfg.seq_len} tokens per micro-batch)",
    )


def _count_params(mc: dict) -> int:
    import torch

    from dsm.modeling import DSMConfig, DSMForCausalLM

    generic = {
        "architectures",
        "auto_map",
        "model_type",
        "transformers_version",
        "dtype",
        "torch_dtype",
        "_name_or_path",
    }
    keys = {
        k: v
        for k, v in mc.items()
        if (k in DSMConfig.__dataclass_fields__ and k not in generic) or k == "full_attention_interval"
    }
    with torch.device("meta"):
        model = DSMForCausalLM(DSMConfig(**keys))
    return sum(p.numel() for p in model.parameters())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dsm preflight", description="Check the GPU machine before training."
    )
    parser.add_argument("--config", help="training config to estimate memory for")
    parser.add_argument("--output-dir", default=".", help="where checkpoints will be written")
    args = parser.parse_args(argv)
    checks = run_checks(args.config, args.output_dir)
    width = max(len(c.name) for c in checks)
    for check in checks:
        print(f"[{check.status:4}] {check.name.ljust(width)}  {check.detail}")
    failed = [c for c in checks if c.status == "FAIL"]
    print(f"\n{len(failed)} failed, {sum(c.status == 'WARN' for c in checks)} warnings")
    return 1 if failed else 0
