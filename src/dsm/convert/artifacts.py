"""Files written next to DSM weights: model card, NOTICE, LICENSE, generation config."""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path

from dsm import repo_root

DEFAULT_GENERATION = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.05,
    "max_new_tokens": 1024,
}


def write_generation_config(out_dir: Path, eos_ids: list[int], pad_id: int) -> None:
    from transformers import GenerationConfig

    config = GenerationConfig(
        bos_token_id=None, eos_token_id=eos_ids, pad_token_id=pad_id, **DEFAULT_GENERATION
    )
    config.save_pretrained(out_dir)


def write_license(out_dir: Path, source_license: Path | None) -> None:
    """Keep the upstream LICENSE verbatim (Apache-2.0 §4 requires it); fall back to the repo copy."""
    target = out_dir / "LICENSE"
    if source_license is not None and source_license.is_file():
        shutil.copyfile(source_license, target)
    else:
        shutil.copyfile(repo_root() / "LICENSE", target)


def write_notice(out_dir: Path, identity: dict, upstream: dict, modifications: list[str]) -> None:
    up = upstream["upstream"]
    year = _dt.date.today().year
    lines = [
        f"{identity['full_name']}",
        f"Copyright {year} {identity['developer']}",
        "",
        f"This model is a modified version of {up['base_repo_id']} / {up['instruct_repo_id']},",
        f"{up['copyright']}, licensed under the Apache License, Version 2.0 (see LICENSE).",
        "",
        f"Modifications by {identity['developer']}:",
        *[f"  * {item}" for item in modifications],
        "",
        "Redistributions must include LICENSE and this NOTICE.",
    ]
    (out_dir / "NOTICE").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_card(out_dir: Path, identity: dict, num_parameters: int, extra_sections: str = "") -> None:
    name = identity["full_name"]
    langs = identity.get("languages", ["km", "en", "zh"])
    caps = "\n".join(f"- {c}" for c in identity.get("capabilities", {}).get("en", []))
    lims = "\n".join(f"- {c}" for c in identity.get("limitations", {}).get("en", []))
    card = f"""---
license: apache-2.0
language: {json.dumps(langs)}
pipeline_tag: text-generation
library_name: transformers
tags: ["dsm", "offline", "khmer", "chinese", "english"]
---

# {name}

{name} is a compact trilingual (Khmer · English · Chinese) language model developed by
**{identity["developer"]}** for fast, fully offline use. It is built on an open-source
foundation model (see `NOTICE`) and adapted by {identity["developer"]}.

- Parameters: {num_parameters / 1e6:.0f}M (text decoder, tied embeddings)
- Architecture: hybrid Gated DeltaNet + gated attention (3:1), SwiGLU MLPs
- Responses: direct ("fast") answers; tool calling; Markdown and LaTeX formatting

## Capabilities
{caps}

## Limitations
{lims}

## Usage

```python
import dsm.modeling  # registers the DSM architecture (or pass trust_remote_code=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{name}")
model = AutoModelForCausalLM.from_pretrained("{name}", dtype="bfloat16")
messages = [{{"role": "user", "content": "Hello!"}}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
out = model.generate(**inputs, max_new_tokens=128)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

For CPU / edge inference export to GGUF with `dsm export gguf` and serve with `llama-server`.
Normalise user text with `dsm.text.normalize_text` before tokenisation (Khmer cluster order).
{extra_sections}
## License

Apache-2.0. Distribute together with `LICENSE` and `NOTICE`.
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")
