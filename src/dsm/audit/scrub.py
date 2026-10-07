"""Upstream-branding audit.

* ``scrub_model_dir`` - a shipped DSM model directory must not mention the
  upstream model or vendor anywhere except in ``LICENSE`` / ``NOTICE`` (legally
  required attribution). Safetensors headers and tokenizer metadata are
  scanned; the BPE vocabulary itself is data and is exempt (it legitimately
  contains tokens such as company names).
* ``scrub_gguf`` - the same for a GGUF file. The engine architecture id is
  allowed in ``general.architecture``, ``tokenizer.ggml.pre`` and as the
  namespace prefix of architecture keys (``<arch>.block_count``): llama.cpp
  selects its compute graph with it.
* ``scrub_repo`` - repository files outside ``.scrub-allowlist``.
"""

from __future__ import annotations

import json
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {".json", ".jinja", ".md", ".txt", ".py", ".yaml", ".yml", ".toml", ".cfg", ".sh", ".csv", ""}


@dataclass(frozen=True)
class Finding:
    path: str
    pattern: str
    context: str

    def __str__(self) -> str:
        return f"{self.path}: {self.pattern!r} in ...{self.context}..."


def _denylist(upstream: dict) -> list[str]:
    return [p.lower() for p in upstream["scrub"]["denylist"]]


def scan_text(text: str, patterns: list[str], path: str) -> list[Finding]:
    lowered = text.lower()
    findings = []
    for pattern in patterns:
        start = 0
        while (index := lowered.find(pattern, start)) >= 0:
            context = text[max(0, index - 30) : index + len(pattern) + 30].replace("\n", " ")
            findings.append(Finding(path, pattern, context))
            start = index + len(pattern)
    return findings


def safetensors_header(path: Path) -> str:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return handle.read(length).decode("utf-8")


def tokenizer_metadata(path: Path) -> str:
    """tokenizer.json without the vocabulary / merges (which are data)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    model = dict(data.get("model") or {})
    model.pop("vocab", None)
    model.pop("merges", None)
    data["model"] = model
    return json.dumps(data, ensure_ascii=False)


def scrub_model_dir(model_dir: str | Path, upstream: dict | None = None) -> list[Finding]:
    from dsm.utils import load_upstream

    upstream = upstream or load_upstream()
    patterns = _denylist(upstream)
    allow = set(upstream["scrub"]["artifact_allow"])
    root = Path(model_dir)
    findings: list[Finding] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root))
        if path.name in allow:
            continue
        findings += scan_text(rel, patterns, rel)  # file names count too
        if path.suffix == ".safetensors":
            text = safetensors_header(path)
        elif path.name == "tokenizer.json":
            text = tokenizer_metadata(path)
        elif path.suffix == ".gguf":
            findings += scrub_gguf(path, upstream)
            continue
        elif path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            findings.append(Finding(rel, "unexpected file type", path.suffix))
            continue
        findings += scan_text(text, patterns, rel)
    return findings


def scrub_gguf(path: str | Path, upstream: dict | None = None) -> list[Finding]:
    from gguf import GGUFReader, GGUFValueType

    from dsm.utils import load_upstream

    upstream = upstream or load_upstream()
    patterns = _denylist(upstream)
    engine = upstream["upstream"]["engine_ids"]
    arch, pre = engine["gguf_arch"], engine["gguf_tokenizer_pre"]
    exempt_values = {"general.architecture": arch, "tokenizer.ggml.pre": pre}
    data_keys = {
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.scores",
    }
    reader = GGUFReader(str(path))
    name = Path(path).name
    findings: list[Finding] = []
    for key, field in reader.fields.items():
        if key in data_keys:
            continue
        bare_key = key[len(arch) + 1 :] if key.startswith(arch + ".") else key
        findings += scan_text(bare_key, patterns, f"{name}:{key}")
        if field.types and field.types[0] == GGUFValueType.STRING:
            value = bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
            if exempt_values.get(key) == value:
                continue
            findings += scan_text(value, patterns, f"{name}:{key}")
    return findings


def _repo_files(root: Path) -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [root / line for line in out.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]


def scrub_repo(root: str | Path = ".", upstream: dict | None = None) -> list[Finding]:
    from dsm.utils import load_upstream

    root = Path(root).resolve()
    upstream = upstream or load_upstream(root / "configs" / "upstream.yaml")
    patterns = _denylist(upstream)
    allow_file = root / ".scrub-allowlist"
    allowed = set()
    if allow_file.is_file():
        allowed = {
            line.strip()
            for line in allow_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
    findings: list[Finding] = []
    for path in _repo_files(root):
        rel = path.relative_to(root).as_posix()
        if rel in allowed or not path.is_file() or path.suffix not in TEXT_SUFFIXES | {".jsonl", ".ini"}:
            continue
        findings += scan_text(path.read_text(encoding="utf-8", errors="replace"), patterns, rel)
    return findings
