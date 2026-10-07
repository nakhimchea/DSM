"""Offline-readiness audit of a model directory.

Checks that a model directory is safe and self-contained for an air-gapped
deployment:

* only data formats that cannot execute code (safetensors, JSON, Jinja,
  Markdown, text, GGUF) - no pickle files (``.bin``/``.pt``/``.pkl``);
* the only Python files are byte-identical copies of the DSM modelling code;
* a ``SHA256SUMS`` manifest can be written and later verified;
* the model loads and generates with every network socket blocked.
"""

from __future__ import annotations

from pathlib import Path

from dsm.utils import sha256_file

DATA_SUFFIXES = {".json", ".safetensors", ".jinja", ".md", ".txt", ".gguf"}
PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt"}
ALLOWED_NAMES = {"LICENSE", "NOTICE", "SHA256SUMS"}
MANIFEST = "SHA256SUMS"


def check_files(model_dir: str | Path) -> list[str]:
    from dsm import modeling

    root = Path(model_dir)
    package = Path(modeling.__file__).parent
    problems = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if path.name in ALLOWED_NAMES or path.suffix in DATA_SUFFIXES:
            continue
        if path.suffix in PICKLE_SUFFIXES:
            problems.append(
                f"{rel}: pickle-based file (can execute code when loaded) - convert to safetensors"
            )
        elif path.suffix == ".py":
            reference = package / path.name
            if path.name not in modeling.REMOTE_CODE_FILES or not reference.is_file():
                problems.append(f"{rel}: unknown Python file")
            elif sha256_file(path) != sha256_file(reference):
                problems.append(f"{rel}: differs from the DSM modelling code shipped with this package")
        else:
            problems.append(f"{rel}: unexpected file type")
    return problems


def write_manifest(model_dir: str | Path) -> Path:
    root = Path(model_dir)
    lines = [
        f"{sha256_file(p)}  {p.relative_to(root).as_posix()}"
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != MANIFEST
    ]
    target = root / MANIFEST
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def verify_manifest(model_dir: str | Path) -> list[str]:
    root = Path(model_dir)
    manifest = root / MANIFEST
    if not manifest.is_file():
        return [f"{MANIFEST} missing (run `dsm audit offline --write-manifest`)"]
    problems = []
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        listed.add(rel)
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: listed in manifest but missing")
        elif sha256_file(path) != digest:
            problems.append(f"{rel}: checksum mismatch")
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_file() and path.name != MANIFEST and rel not in listed:
            problems.append(f"{rel}: not listed in manifest")
    return problems


def offline_generation_smoke(model_dir: str | Path, prompt: str = "Hello", max_new_tokens: int = 4) -> str:
    """Load tokenizer + model and generate with all non-loopback networking blocked."""
    import torch

    from dsm.env import block_network, enforce_offline

    enforce_offline(override=True)
    with block_network():
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import dsm.modeling  # noqa: F401  (registers the architecture)
        from dsm.serve.generate import SamplingConfig, generate

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"]
        result = generate(model, ids, sampling=SamplingConfig(max_new_tokens=max_new_tokens))
        return tokenizer.decode(result.new_tokens[0])
