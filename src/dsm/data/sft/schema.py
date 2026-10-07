"""Unified record formats for supervised fine-tuning and preference data.

SFT record (one JSON object per line)::

    {
      "id": "identity-km-000123",
      "source": "dsm-identity",          # builder or dataset name
      "license": "Apache-2.0",           # licence of this sample's content
      "generated_by": "template",        # template | human | dataset:<name> | model:<teacher>
      "lang": "km",                       # km | en | zh | mixed
      "category": "identity",            # see CATEGORIES
      "system": null,                     # null -> the chat template's default system prompt
      "tools": [ {OpenAI function schema}, ... ],      # optional
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "tool_calls": [{"name": "...", "arguments": {...}}]},
        {"role": "tool", "name": "...", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "meta": {}
    }

Preference record: same header fields plus ``prompt`` (messages ending with a
user turn), ``chosen`` and ``rejected`` (single assistant messages).
"""

from __future__ import annotations

from typing import Any

CATEGORIES = (
    "chat",
    "identity",
    "safety",
    "tools",
    "tables",
    "math",
    "code",
    "format",
    "rag",
    "translation",
    "khmer",
)
LANGS = ("km", "en", "zh", "mixed")
ROLES = ("user", "assistant", "tool")


def sft_record(
    *,
    id: str,
    source: str,
    messages: list[dict],
    lang: str,
    category: str,
    license: str = "Apache-2.0",
    generated_by: str = "template",
    system: str | None = None,
    tools: list[dict] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict:
    record = {
        "id": id,
        "source": source,
        "license": license,
        "generated_by": generated_by,
        "lang": lang,
        "category": category,
        "system": system,
        "messages": messages,
        "meta": meta or {},
    }
    if tools:
        record["tools"] = tools
    return record


def preference_record(
    *,
    id: str,
    source: str,
    prompt: list[dict],
    chosen: dict,
    rejected: dict,
    lang: str,
    category: str,
    license: str = "Apache-2.0",
    generated_by: str = "template",
    system: str | None = None,
    tools: list[dict] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict:
    record = {
        "id": id,
        "source": source,
        "license": license,
        "generated_by": generated_by,
        "lang": lang,
        "category": category,
        "system": system,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "meta": meta or {},
    }
    if tools:
        record["tools"] = tools
    return record


def template_messages(record: dict, messages: list[dict] | None = None) -> list[dict]:
    """Messages as passed to the chat template (system prompt prepended when set)."""
    body = list(record["messages"] if messages is None else messages)
    if record.get("system") is not None:
        return [{"role": "system", "content": record["system"]}, *body]
    return body


def function_schema(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    """An OpenAI-style function tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required or [])},
        },
    }
