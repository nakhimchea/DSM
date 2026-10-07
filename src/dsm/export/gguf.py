"""GGUF export through llama.cpp's own converter, with a small DSM shim.

llama.cpp chooses its converter class by ``architectures[0]`` in ``config.json``.
DSM ships ``DSMForCausalLM`` (no upstream branding), so this module registers
a ``DSMForCausalLM`` converter that subclasses llama.cpp's class for the
upstream text model (named in ``configs/upstream.yaml``; DSM keeps the exact
tensor layout, so every weight transform - zero-centred norm +1, ``A_log`` ->
``-exp``, V-head reordering, M-RoPE sections - is reused unchanged) and adds:

* ``no_mtp`` - DSM has no multi-token-prediction head;
* the full-attention interval derived from ``layer_types`` (the converter
  otherwise assumes 4);
* a pre-tokeniser fallback: a Khmer-extended vocabulary changes llama.cpp's
  tokenizer fingerprint, but keeps DSM's pre-tokenisation regex, so it maps to
  the same pre-tokeniser id;
* DSM metadata (name, organisation, licence) via a metadata override.

Two GGUF values necessarily keep the upstream engine id (``general.architecture``
and ``tokenizer.ggml.pre``): they select llama.cpp's compute graph and
pre-tokeniser. ``dsm audit gguf`` treats exactly those as expected.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("dsm.export")

OUTTYPES = ("f32", "f16", "bf16", "q8_0")
_CONVERTER_CLASS = None


def find_llama_cpp(path: str | Path | None = None) -> Path:
    """The llama.cpp checkout (``--llama-cpp``, ``$DSM_LLAMA_CPP`` or ``./third_party/llama.cpp``)."""
    candidates = [path, os.environ.get("DSM_LLAMA_CPP"), "third_party/llama.cpp"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "conversion" / "__init__.py").is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError(
        "llama.cpp checkout not found: clone the pinned commit (scripts/70_export.sh) and pass --llama-cpp "
        "or set DSM_LLAMA_CPP"
    )


def attention_interval(layer_types: list[str]) -> int:
    """Period of the full-attention layers (DSM: one every N layers, starting at layer N-1)."""
    positions = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
    if not positions:
        raise ValueError("model has no full-attention layer")
    interval = positions[0] + 1
    expected = [i for i in range(len(layer_types)) if (i + 1) % interval == 0]
    if positions != expected:
        raise ValueError(
            f"full-attention layers {positions} are not periodic; llama.cpp needs a fixed interval"
        )
    return interval


def uses_dsm_pretokenizer(model_dir: str | Path) -> bool:
    from dsm.tokenizer import PRETOKENIZE_REGEX

    data = json.loads((Path(model_dir) / "tokenizer.json").read_text(encoding="utf-8"))
    pre = data.get("pre_tokenizer") or {}
    steps = pre.get("pretokenizers", [pre])
    patterns = [s.get("pattern", {}).get("Regex") for s in steps if s.get("type") == "Split"]
    return patterns == [PRETOKENIZE_REGEX]


def converter_class(llama_cpp: str | Path | None = None):
    """Import llama.cpp's converter and return the registered ``DSMForCausalLM`` class."""
    global _CONVERTER_CLASS
    if _CONVERTER_CLASS is not None:
        return _CONVERTER_CLASS
    from dsm.utils import load_upstream

    root = find_llama_cpp(llama_cpp)
    for entry in (str(root / "gguf-py"), str(root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    # The converter needs the gguf-py that ships with it (new metadata keys), not an older pip copy.
    loaded = sys.modules.get("gguf")
    if loaded is not None and not str(getattr(loaded, "__file__", "")).startswith(str(root)):
        for name in [m for m in sys.modules if m == "gguf" or m.startswith("gguf.")]:
            del sys.modules[name]
    import conversion  # llama.cpp's converter package

    upstream = load_upstream()["upstream"]
    ids = upstream["engine_ids"]
    base = conversion.get_model_class(upstream["architectures"]["causal_lm"])
    if base.__name__ != ids["llama_cpp_text_model_class"]:
        raise RuntimeError(
            f"llama.cpp maps the upstream model to {base.__name__}, expected {ids['llama_cpp_text_model_class']}"
        )

    class DSMForCausalLMConverter(base):  # type: ignore[misc, valid-type]
        model_arch = base.model_arch  # the upstream compute graph (gguf arch id)
        no_mtp = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hparams["full_attention_interval"] = attention_interval(self.hparams["layer_types"])

        def get_vocab_base_pre(self, tokenizer) -> str:
            # A Khmer-extended vocabulary (dsm tokenizer extend) encodes llama.cpp's fingerprint
            # text differently, so its hash is unknown; the pre-tokeniser itself is unchanged.
            extended = (Path(self.dir_model) / "extension.json").is_file()
            if extended and uses_dsm_pretokenizer(self.dir_model):
                log.info("Khmer-extended vocabulary, same pre-tokeniser -> %s", ids["gguf_tokenizer_pre"])
                return ids["gguf_tokenizer_pre"]
            try:
                return super().get_vocab_base_pre(tokenizer)
            except NotImplementedError:
                if not uses_dsm_pretokenizer(self.dir_model):
                    raise
                log.info(
                    "tokenizer fingerprint unknown to llama.cpp; same pre-tokeniser -> %s",
                    ids["gguf_tokenizer_pre"],
                )
                return ids["gguf_tokenizer_pre"]

    conversion.ModelBase.register("DSMForCausalLM")(DSMForCausalLMConverter)
    conversion.TEXT_MODEL_MAP["DSMForCausalLM"] = conversion.TEXT_MODEL_MAP[
        upstream["architectures"]["causal_lm"]
    ]
    _CONVERTER_CLASS = DSMForCausalLMConverter
    return DSMForCausalLMConverter


def metadata_override(identity: dict, version: str | None = None) -> dict:
    """GGUF ``general.*`` metadata for DSM (the licence comes from the model card's front matter)."""
    return {
        "general.name": identity["full_name"],
        "general.basename": identity["name"],
        "general.organization": identity["developer"],
        "general.version": version or identity["version"],
        "general.description": f"{identity['display_name']} by {identity['developer']}: a compact Khmer / English / Chinese assistant for local, offline use.",
        "general.languages": list(identity.get("languages", ["km", "en", "zh"])),
    }


def convert(
    model_dir: str | Path, out_file: str | Path, *, outtype: str = "bf16", llama_cpp: str | Path | None = None
) -> Path:
    """Convert a DSM model directory to one GGUF file."""
    import torch

    import dsm.modeling  # noqa: F401 - registers the "dsm" model type for the converter's config loading
    from dsm.utils import load_identity

    if outtype not in OUTTYPES:
        raise ValueError(f"outtype must be one of {OUTTYPES} (use quantize() for k-quants)")
    cls = converter_class(llama_cpp)
    import gguf

    ftype = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    }[outtype]
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    identity = load_identity()
    with tempfile.TemporaryDirectory() as tmp:
        override = Path(tmp) / "metadata.json"
        override.write_text(json.dumps(metadata_override(identity)), encoding="utf-8")
        with torch.inference_mode():
            model = cls(
                Path(model_dir), ftype, out_file, metadata_override=override, model_name=identity["full_name"]
            )
            model.write()
    return out_file


# -- llama.cpp binaries -----------------------------------------------------------------------


def find_binary(name: str, bin_dir: str | Path | None = None, llama_cpp: str | Path | None = None) -> Path:
    import shutil

    candidates = []
    if bin_dir:
        candidates.append(Path(bin_dir) / name)
    if os.environ.get("DSM_LLAMA_BIN"):
        candidates.append(Path(os.environ["DSM_LLAMA_BIN"]) / name)
    with contextlib.suppress(FileNotFoundError):
        candidates.append(find_llama_cpp(llama_cpp) / "build" / "bin" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError(f"{name} not found: build llama.cpp (scripts/70_export.sh) or pass --llama-bin")


def quantize(
    src: str | Path,
    dst: str | Path,
    qtype: str,
    *,
    imatrix: str | Path | None = None,
    bin_dir=None,
    llama_cpp=None,
) -> Path:
    """``llama-quantize [--imatrix FILE] SRC DST TYPE`` (e.g. Q8_0, Q6_K, Q4_K_M)."""
    cmd = [str(find_binary("llama-quantize", bin_dir, llama_cpp))]
    if imatrix:
        cmd += ["--imatrix", str(imatrix)]
    cmd += [str(src), str(dst), qtype]
    subprocess.run(cmd, check=True, env=offline_env())
    return Path(dst)


def compute_imatrix(
    model_gguf: str | Path,
    calibration_text: str | Path,
    out: str | Path,
    *,
    bin_dir=None,
    llama_cpp=None,
    chunks: int = 200,
) -> Path:
    """Importance matrix from mixed km / en / zh text (keeps low-bit quants good on Khmer)."""
    cmd = [
        str(find_binary("llama-imatrix", bin_dir, llama_cpp)),
        "-m",
        str(model_gguf),
        "-f",
        str(calibration_text),
        "-o",
        str(out),
        "--chunks",
        str(chunks),
    ]
    subprocess.run(cmd, check=True, env=offline_env())
    return Path(out)


def kl_divergence(
    reference_gguf: str | Path,
    candidate_gguf: str | Path,
    text: str | Path,
    *,
    bin_dir=None,
    llama_cpp=None,
    ctx: int = 512,
) -> dict[str, float]:
    """Mean KL and top-1 agreement of a quantised GGUF against the reference, via llama-perplexity."""
    import re

    perplexity = str(find_binary("llama-perplexity", bin_dir, llama_cpp))
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base.kld"
        subprocess.run(
            [
                perplexity,
                "-m",
                str(reference_gguf),
                "-f",
                str(text),
                "-c",
                str(ctx),
                "--kl-divergence-base",
                str(base),
                "--no-repack",  # repacked CPU weights + several sequences per batch -> NaN (docs/11)
            ],
            check=True,
            capture_output=True,
            env=offline_env(),
        )
        run = subprocess.run(
            [
                perplexity,
                "-m",
                str(candidate_gguf),
                "-f",
                str(text),
                "-c",
                str(ctx),
                "--kl-divergence-base",
                str(base),
                "--kl-divergence",
                "--no-repack",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=offline_env(),
        )
    output = run.stdout + run.stderr
    out = {}
    if match := re.search(r"Mean\s+KLD:\s*([0-9.eE+-]+)", output):
        out["kl"] = float(match.group(1))
    if match := re.search(r"Same top p:\s*([0-9.]+)", output):
        out["top1_agreement"] = float(match.group(1)) / 100
    return out


def offline_env() -> dict[str, str]:
    from dsm.env import OFFLINE_ENV

    env = dict(os.environ)
    env.update(OFFLINE_ENV)
    env["LLAMA_ARG_OFFLINE"] = "1"
    return env


def write_release_files(model_dir: str | Path, out_dir: str | Path) -> None:
    """LICENSE / NOTICE / model card next to the GGUF files (Apache-2.0 section 4)."""
    import shutil

    for name in ("LICENSE", "NOTICE", "README.md"):
        source = Path(model_dir) / name
        if source.is_file():
            shutil.copyfile(source, Path(out_dir) / name)
