"""DSM tool-call text format (shared by data builders, evaluation and serving).

The model writes calls as XML blocks (identical to what the chat template renders
for training data)::

    <tool_call>
    <function=get_exchange_rate>
    <parameter=base>
    USD
    </parameter>
    </function>
    </tool_call>

Parameter values are text; objects / arrays are JSON. ``parse_tool_calls``
turns model output back into structured calls, coercing values with the tool's
JSON schema when it is known (like the parsers in llama-server / vLLM).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\n?(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def as_message_call(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}


def format_value(value: Any) -> str:
    """How the chat template prints a parameter value (JSON for containers, ``str`` otherwise)."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_tool_calls(calls: list[ToolCall | dict], content: str = "") -> str:
    """Text the template produces after the assistant ``content`` (without ``<|im_end|>``)."""
    out = []
    for index, call in enumerate(calls):
        call = (
            call if isinstance(call, ToolCall) else ToolCall(call["name"], dict(call.get("arguments") or {}))
        )
        if index == 0:
            out.append("\n\n<tool_call>\n" if content.strip() else "<tool_call>\n")
        else:
            out.append("\n<tool_call>\n")
        out.append(f"<function={call.name}>\n")
        for key, value in call.arguments.items():
            out.append(f"<parameter={key}>\n{format_value(value)}\n</parameter>\n")
        out.append("</function>\n</tool_call>")
    return "".join(out)


def _schema_types(tools: list[dict] | None, name: str) -> dict[str, str]:
    for tool in tools or []:
        fn = tool.get("function", tool)
        if fn.get("name") == name:
            props = (fn.get("parameters") or {}).get("properties") or {}
            return {k: (v.get("type") if isinstance(v, dict) else None) for k, v in props.items()}
    return {}


def coerce(raw: str, expected: str | list | None) -> Any:
    """Convert a parameter string to the JSON-schema type (best effort)."""
    text = raw.strip()
    kinds = expected if isinstance(expected, list) else [expected]
    for kind in kinds:
        try:
            if kind == "integer":
                return int(text)
            if kind == "number":
                number = float(text)
                return (
                    int(number)
                    if number.is_integer() and "." not in text and "e" not in text.lower()
                    else number
                )
            if kind == "boolean" and text.lower() in ("true", "false", "1", "0"):
                return text.lower() in ("true", "1")
            if kind in ("object", "array"):
                return json.loads(text)
            if kind == "string":
                return raw
            if kind == "null" and text.lower() in ("null", "none"):
                return None
        except (ValueError, json.JSONDecodeError):
            continue
    if expected is None:  # unknown schema: JSON literal if it parses, else text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return raw
    return raw


def parse_tool_calls(text: str, tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
    """Split model output into (plain content, tool calls)."""
    calls = []
    for match in _CALL.finditer(text):
        name = match.group(1).strip()
        types = _schema_types(tools, name)
        arguments = {
            key.strip(): coerce(value, types.get(key.strip()))
            for key, value in _PARAM.findall(match.group(2))
        }
        calls.append(ToolCall(name, arguments))
    first = text.find("<tool_call>")
    content = (text if first < 0 else text[:first]).strip()
    return content, calls
