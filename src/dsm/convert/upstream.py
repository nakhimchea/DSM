"""Convert an upstream hybrid Gated-DeltaNet checkpoint into a DSM checkpoint.

    dsm convert --src models/upstream-base --out models/DSM1.0-0.8B-base

What happens:
1. The text decoder config is read (from ``text_config`` for multimodal
   checkpoints). The output-embedding tie flag is taken from the TOP-LEVEL
   config, because the text sub-config defaults to untied.
2. Text weights are renamed ``model.language_model.* -> model.*`` and
   validated against a DSM model (no missing / unexpected keys, identical
   shapes). Vision-encoder and multi-token-prediction tensors are moved to a
   sibling ``<out>-extras/`` directory. Source dtypes are kept by default
   (bf16, with the decay parameters and GDN norm in fp32).
3. The tokenizer vocabulary is copied verbatim; the tokenizer config, chat
   template, generation config, model card, LICENSE (upstream, verbatim) and
   NOTICE are written fresh. The DSM modelling files are copied in so the
   directory also loads with ``trust_remote_code=True``.
4. The result is audited: no upstream branding outside LICENSE / NOTICE.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

FP32_KEEP = ("A_log", "linear_attn.norm.weight")
CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
    "rope_parameters",
    "attention_bias",
    "attention_dropout",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "layer_types",
)
MODIFICATIONS = [
    "vision encoder and multi-token-prediction head removed (text-only model)",
    "decoder re-implemented in the DSM code base; configuration and tokenizer metadata rewritten",
    "new chat template (fast responses, DSM identity as default system prompt)",
]


@dataclass
class ConversionReport:
    out_dir: Path
    num_parameters: int = 0
    tied_embeddings: bool = True
    moved_to_extras: dict[str, int] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    audit_findings: list = field(default_factory=list)

    def summary(self) -> str:
        extras = ", ".join(f"{k}: {v} tensors" for k, v in self.moved_to_extras.items()) or "none"
        return (
            f"DSM checkpoint written to {self.out_dir}\n"
            f"  parameters: {self.num_parameters:,} (tied embeddings: {self.tied_embeddings})\n"
            f"  moved to extras: {extras}\n"
            f"  audit: {'clean' if not self.audit_findings else f'{len(self.audit_findings)} finding(s)'}"
        )


def _read_config(src: Path) -> tuple[dict, dict, bool]:
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    composite = "text_config" in config
    return config, text, composite


def build_dsm_config(src: Path, upstream: dict):
    """A :class:`DSMConfig` equivalent to the source text decoder."""
    from dsm.modeling import DSMConfig
    from dsm.tokenizer import special_token_ids

    config, text, _ = _read_config(src)
    known = set(upstream["upstream"]["model_types"].values())
    if config.get("model_type") not in known and text.get("model_type") not in known:
        raise ValueError(
            f"unsupported source model_type {config.get('model_type')!r}; expected one of {sorted(known)}"
        )
    if not text.get("attn_output_gate", True):
        raise ValueError("DSM requires gated attention (attn_output_gate=true)")
    if text.get("mlp_only_layers") or text.get("num_experts"):
        raise ValueError("mixture-of-experts / mlp-only layers are not supported by DSM")

    kwargs = {k: text[k] for k in CONFIG_FIELDS if k in text}
    if "rope_parameters" not in kwargs and "rope_theta" in text:  # older config layout
        kwargs["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": text["rope_theta"],
            "partial_rotary_factor": text.get("partial_rotary_factor", 0.25),
        }
    if "layer_types" not in kwargs and "full_attention_interval" in text:
        kwargs["full_attention_interval"] = text["full_attention_interval"]
    specials = special_token_ids(src / "tokenizer.json")
    tie = config.get("tie_word_embeddings", text.get("tie_word_embeddings", False))
    return DSMConfig(
        **kwargs,
        tie_word_embeddings=bool(tie),
        eos_token_id=specials["<|im_end|>"],
        pad_token_id=specials["<|endoftext|>"],
        bos_token_id=None,
    )


def _source_files(src: Path) -> list[Path]:
    index = src / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return sorted({src / name for name in weight_map.values()})
    files = sorted(src.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {src}")
    return files


def map_tensors(
    src: Path, tie: bool
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]], list[str]]:
    """Rename source tensors to the DSM layout; split off vision / MTP tensors."""
    from safetensors import safe_open

    text: dict[str, torch.Tensor] = {}
    extras: dict[str, dict[str, torch.Tensor]] = {"vision": {}, "mtp": {}}
    dropped: list[str] = []
    for file in _source_files(src):
        with safe_open(str(file), framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not a mapping
                if key.startswith("model.language_model."):
                    text["model." + key[len("model.language_model.") :]] = handle.get_tensor(key)
                elif key.startswith(("model.visual.", "visual.")):
                    extras["vision"][key] = handle.get_tensor(key)
                elif key.startswith("mtp."):
                    extras["mtp"][key] = handle.get_tensor(key)
                elif key == "lm_head.weight":
                    if tie:
                        dropped.append(key)
                    else:
                        text[key] = handle.get_tensor(key)
                elif key.startswith("model."):
                    text[key] = handle.get_tensor(key)
                else:
                    raise ValueError(f"unexpected tensor in source checkpoint: {key}")
    return text, extras, dropped


def validate_against_model(tensors: dict[str, torch.Tensor], config) -> int:
    """Check names and shapes against a DSM model built on the meta device; returns #parameters."""
    from dsm.modeling import DSMForCausalLM

    with torch.device("meta"):
        model = DSMForCausalLM(config)
    expected = {k: v.shape for k, v in model.state_dict().items()}
    if config.tie_word_embeddings:
        expected.pop("lm_head.weight", None)
    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(set(tensors) - set(expected))
    if missing or unexpected:
        raise ValueError(f"tensor mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    for key, shape in expected.items():
        if tuple(tensors[key].shape) != tuple(shape):
            raise ValueError(f"shape mismatch for {key}: {tuple(tensors[key].shape)} vs {tuple(shape)}")
    return sum(p.numel() for p in model.parameters())


def cast_tensors(tensors: dict[str, torch.Tensor], dtype: str) -> dict[str, torch.Tensor]:
    if dtype == "keep":
        return tensors
    target = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    return {
        k: (v.to(torch.float32) if k.endswith(FP32_KEEP) else v.to(target)) if v.is_floating_point() else v
        for k, v in tensors.items()
    }


def copy_remote_code(out_dir: Path) -> None:
    from dsm import modeling

    package = Path(modeling.__file__).parent
    for name in modeling.REMOTE_CODE_FILES:
        shutil.copyfile(package / name, out_dir / name)


def write_lineage(out: Path, src: Path, num_parameters: int) -> None:
    """``lineage.json`` starts here: the source revision (``dsm fetch`` writes FETCH.json next to
    the checkpoints) and the hash of every source weight file. No upstream names: see NOTICE."""
    import datetime as dt

    from dsm.utils import sha256_file

    fetch = src.parent / "FETCH.json"
    manifest = json.loads(fetch.read_text(encoding="utf-8")) if fetch.is_file() else {}
    entry = {
        "stage": "convert",
        "source": "upstream checkpoint (see NOTICE)",
        "source_revision": manifest.get(src.name, {}).get("revision"),
        "source_weights_sha256": {p.name: sha256_file(p) for p in sorted(src.glob("*.safetensors"))},
        "num_parameters": num_parameters,
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "lineage.json").write_text(json.dumps([entry], indent=2) + "\n", encoding="utf-8")


def convert(
    src: str | Path,
    out: str | Path,
    identity_path: str | Path = "configs/identity.yaml",
    upstream_path: str | Path = "configs/upstream.yaml",
    dtype: str = "keep",
    extras_dir: str | Path | None = None,
    overwrite: bool = False,
    audit: bool = True,
) -> ConversionReport:
    from safetensors.torch import save_file

    from dsm.audit.scrub import scrub_model_dir
    from dsm.convert.artifacts import write_generation_config, write_license, write_model_card, write_notice
    from dsm.modeling import AUTO_MAP
    from dsm.tokenizer import write_tokenizer_files
    from dsm.utils import load_identity, load_upstream

    src, out = Path(src), Path(out)
    identity, upstream = load_identity(identity_path), load_upstream(upstream_path)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty (use --overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    config = build_dsm_config(src, upstream)
    tensors, extras, dropped = map_tensors(src, config.tie_word_embeddings)
    report = ConversionReport(out_dir=out, tied_embeddings=config.tie_word_embeddings, dropped=dropped)
    report.num_parameters = validate_against_model(tensors, config)
    tensors = cast_tensors(tensors, dtype)

    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        str(out / "model.safetensors"),
        metadata={"format": "pt"},
    )
    extras_path = Path(extras_dir) if extras_dir else out.parent / f"{out.name}-extras"
    for name, group in extras.items():
        if group:
            extras_path.mkdir(parents=True, exist_ok=True)
            save_file({k: v.contiguous() for k, v in group.items()}, str(extras_path / f"{name}.safetensors"))
            report.moved_to_extras[name] = len(group)

    config.architectures = ["DSMForCausalLM"]
    config.auto_map = dict(AUTO_MAP)
    floating = [v.dtype for v in tensors.values() if v.is_floating_point()]
    config.dtype = max(set(floating), key=floating.count) if floating else torch.bfloat16
    config.save_pretrained(out)
    write_generation_config(
        out, eos_ids=[config.eos_token_id, config.pad_token_id], pad_id=config.pad_token_id
    )
    write_tokenizer_files(
        out,
        src / "tokenizer.json",
        identity["default_system_prompt"],
        model_max_length=config.max_position_embeddings,
    )
    copy_remote_code(out)
    write_license(out, src / "LICENSE")
    write_notice(out, identity, upstream, MODIFICATIONS)
    write_model_card(out, identity, report.num_parameters)
    write_lineage(out, src, report.num_parameters)

    if audit:
        report.audit_findings = scrub_model_dir(out, upstream)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dsm convert", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src",
        required=True,
        help="upstream checkpoint directory (config.json, *.safetensors, tokenizer.json)",
    )
    parser.add_argument("--out", required=True, help="output DSM model directory")
    parser.add_argument("--identity", default="configs/identity.yaml")
    parser.add_argument("--upstream", default="configs/upstream.yaml")
    parser.add_argument("--dtype", choices=["keep", "bf16", "fp16", "fp32"], default="keep")
    parser.add_argument(
        "--extras-dir", default=None, help="where vision / MTP tensors go (default: <out>-extras)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = convert(
        args.src,
        args.out,
        identity_path=args.identity,
        upstream_path=args.upstream,
        dtype=args.dtype,
        extras_dir=args.extras_dir,
        overwrite=args.overwrite,
    )
    print(report.summary())
    for finding in report.audit_findings:
        print(f"  ! {finding}", file=sys.stderr)
    return 1 if report.audit_findings else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
