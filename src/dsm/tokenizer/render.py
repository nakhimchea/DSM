"""Render conversations with the DSM chat template and build training labels.

The template (``chat_template.jinja``) is rendered with the same Jinja
environment transformers uses for ``apply_chat_template`` (sandboxed,
``trim_blocks``/``lstrip_blocks``, ``tojson`` with ``ensure_ascii=False``), so
training and serving see byte-identical prompts.

Training labels are derived from character offsets: the rendered text is
tokenised once, and a token is trained when its start offset lies inside an
assistant completion. A completion starts after ``<|im_start|>assistant\\n``
(and after the pre-filled empty think block, which is never trained) and ends
with ``<|im_end|>`` inclusive.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2
import jinja2.ext
import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE_PATH = Path(__file__).with_name("chat_template.jinja")
SYSTEM_PROMPT_MARKER = "__DSM_DEFAULT_SYSTEM_PROMPT__"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
END_OF_TEXT = "<|endoftext|>"
ASSISTANT_HEADER = IM_START + "assistant\n"
THINK_PREFILL = "<think>\n\n</think>\n\n"
# Control strings that must never occur inside message content.
RESERVED_STRINGS = (IM_START, IM_END, END_OF_TEXT, "<think>", "</think>")


def build_template(default_system_prompt: str) -> str:
    """The DSM chat template with the default system prompt filled in."""
    raw = TEMPLATE_PATH.read_text(encoding="utf-8")
    escaped = json.dumps(default_system_prompt, ensure_ascii=False)[1:-1]
    if SYSTEM_PROMPT_MARKER not in raw:  # pragma: no cover - guards template edits
        raise RuntimeError("chat template lost its default system prompt marker")
    return raw.replace(SYSTEM_PROMPT_MARKER, escaped)


def _tojson(
    value: Any, ensure_ascii: bool = False, indent=None, separators=None, sort_keys: bool = False
) -> str:
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def _raise_exception(message: str):
    raise jinja2.exceptions.TemplateError(message)


@functools.lru_cache(maxsize=8)
def _compile(template: str) -> jinja2.Template:
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    return env.from_string(template)


def render(
    messages: Sequence[dict],
    template: str,
    tools: Sequence[dict] | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Render a conversation to text (same output as ``apply_chat_template(tokenize=False)``)."""
    return _compile(template).render(
        messages=list(messages),
        tools=list(tools) if tools else None,
        add_generation_prompt=add_generation_prompt,
    )


def completion_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of every assistant completion in a rendered conversation."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while (header := text.find(ASSISTANT_HEADER, pos)) >= 0:
        start = header + len(ASSISTANT_HEADER)
        if text.startswith(THINK_PREFILL, start):
            start += len(THINK_PREFILL)
        end = text.find(IM_END, start)
        if end < 0:  # generation prompt / unfinished turn: nothing to train
            break
        end += len(IM_END)
        spans.append((start, end))
        pos = end
    return spans


@dataclass
class TrainingExample:
    input_ids: list[int]
    labels: list[int]  # -100 where the token is not trained
    text: str

    @property
    def num_trained(self) -> int:
        return sum(1 for label in self.labels if label != -100)


def _backend(tokenizer):
    """Accept a ``tokenizers.Tokenizer`` or a transformers tokenizer."""
    return getattr(tokenizer, "backend_tokenizer", tokenizer)


def encode_for_training(
    tokenizer,
    messages: Sequence[dict],
    template: str,
    tools: Sequence[dict] | None = None,
    train_on: str = "all",
    max_length: int | None = None,
) -> TrainingExample:
    """Render + tokenise a conversation with assistant-only labels.

    ``train_on``: ``"all"`` assistant turns or only the ``"last"`` one.
    Raises ``ValueError`` if message content contains reserved control strings.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and any(r in content for r in RESERVED_STRINGS):
            raise ValueError(f"reserved control string inside {message.get('role')} content")
    text = render(messages, template, tools=tools)
    spans = completion_spans(text)
    n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
    if len(spans) != n_assistant:  # pragma: no cover - template/renderer drift
        raise RuntimeError(f"found {len(spans)} completion spans for {n_assistant} assistant messages")
    if train_on == "last":
        spans = spans[-1:]
    elif train_on != "all":
        raise ValueError("train_on must be 'all' or 'last'")

    encoding = _backend(tokenizer).encode(text, add_special_tokens=False)
    ids = np.asarray(encoding.ids, dtype=np.int64)
    starts = np.asarray([offset[0] for offset in encoding.offsets], dtype=np.int64)
    trained = np.zeros(len(ids), dtype=bool)
    for span_start, span_end in spans:
        trained |= (starts >= span_start) & (starts < span_end)
    labels = np.where(trained, ids, -100)
    if max_length is not None and len(ids) > max_length:
        ids, labels = ids[:max_length], labels[:max_length]
    return TrainingExample(input_ids=ids.tolist(), labels=labels.tolist(), text=text)


def encode_prompt(
    tokenizer, messages: Sequence[dict], template: str, tools: Sequence[dict] | None = None
) -> list[int]:
    """Token ids of a prompt ending with the assistant generation header."""
    text = render(messages, template, tools=tools, add_generation_prompt=True)
    return _backend(tokenizer).encode(text, add_special_tokens=False).ids
