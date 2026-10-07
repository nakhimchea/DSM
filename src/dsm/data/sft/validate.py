"""Validation of SFT and preference records (run by every builder and before training)."""

from __future__ import annotations

from dsm.data.sft.schema import CATEGORIES, LANGS, ROLES
from dsm.data.synth.filters import identity_leak, is_normalized

# Control strings that must never appear in message text (tool calls are structured).
FORBIDDEN_IN_CONTENT = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)


def _tool_index(tools: list[dict] | None) -> dict[str, dict]:
    index = {}
    for tool in tools or []:
        fn = tool.get("function", tool)
        index[fn["name"]] = fn.get("parameters") or {"type": "object", "properties": {}}
    return index


def _validate_arguments(name: str, arguments: dict, schema: dict) -> list[str]:
    problems = []
    for key in schema.get("required", []):
        if key not in arguments:
            problems.append(f"tool call {name}: missing required argument {key!r}")
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema is in the [data] extra
        return problems
    try:
        jsonschema.validate(arguments, schema)
    except jsonschema.ValidationError as exc:
        problems.append(f"tool call {name}: {exc.message}")
    return problems


def _check_text(role: str, text: object, problems: list[str], check_normalized: bool) -> None:
    if not isinstance(text, str):
        problems.append(f"{role}: content must be a string")
        return
    for control in FORBIDDEN_IN_CONTENT:
        if control in text:
            problems.append(f"{role}: content contains control string {control!r}")
    if check_normalized and not is_normalized(text):
        problems.append(f"{role}: text is not normalised (run dsm.text.normalize_text)")


def validate_messages(
    messages: list[dict],
    tools: list[dict] | None,
    *,
    must_end_with: str | None = "assistant",
    allow_other_identity: bool = False,
    check_normalized: bool = True,
) -> list[str]:
    problems: list[str] = []
    if not messages:
        return ["no messages"]
    if messages[0].get("role") != "user":
        problems.append("first message must come from the user")
    if must_end_with and messages[-1].get("role") != must_end_with:
        problems.append(f"last message must come from the {must_end_with}")
    index = _tool_index(tools)
    previous = None
    for i, message in enumerate(messages):
        role = message.get("role")
        if role not in ROLES:
            problems.append(f"message {i}: unknown role {role!r}")
            continue
        _check_text(role, message.get("content", ""), problems, check_normalized)
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if not (message.get("content") or "").strip() and not calls:
                problems.append(f"message {i}: empty assistant turn")
            for call in calls:
                name, arguments = call.get("name"), call.get("arguments")
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    problems.append(f"message {i}: tool calls need a name and an arguments object")
                    continue
                if index and name not in index:
                    problems.append(f"message {i}: call to undeclared tool {name!r}")
                elif index:
                    problems += _validate_arguments(name, arguments, index[name])
            if not allow_other_identity and (leak := identity_leak(message.get("content") or "")):
                problems.append(f"message {i}: identity leak {leak!r}")
        elif role == "tool":
            prev_calls = previous is not None and (
                previous.get("role") == "tool" or bool(previous.get("tool_calls"))
            )
            if not prev_calls:
                problems.append(f"message {i}: tool result without a preceding tool call")
        elif role == "user" and previous is not None and previous.get("role") == "user":
            problems.append(f"message {i}: two consecutive user turns")
        previous = message
    return problems


def validate_sft(record: dict, check_normalized: bool = True) -> list[str]:
    problems = []
    for key in ("id", "source", "license", "generated_by", "lang", "category", "messages"):
        if key not in record:
            problems.append(f"missing field {key!r}")
    if problems:
        return problems
    if record["category"] not in CATEGORIES:
        problems.append(f"unknown category {record['category']!r}")
    if record["lang"] not in LANGS:
        problems.append(f"unknown lang {record['lang']!r}")
    if record.get("system") is not None:
        _check_text("system", record["system"], problems, check_normalized)
    meta = record.get("meta") or {}
    problems += validate_messages(
        record["messages"],
        record.get("tools"),
        allow_other_identity=bool(meta.get("allow_other_identity")),
        check_normalized=check_normalized,
    )
    return problems


def validate_preference(record: dict, check_normalized: bool = True) -> list[str]:
    problems = []
    for key in ("id", "prompt", "chosen", "rejected", "lang", "category"):
        if key not in record:
            problems.append(f"missing field {key!r}")
    if problems:
        return problems
    problems += validate_messages(
        record["prompt"], record.get("tools"), must_end_with="user", check_normalized=check_normalized
    )
    for side in ("chosen", "rejected"):
        message = record[side]
        if message.get("role") != "assistant":
            problems.append(f"{side} must be an assistant message")
        allow = side == "rejected"  # rejected answers may contain the undesired identity on purpose
        problems += [
            f"{side}: {p}"
            for p in validate_messages(
                [{"role": "user", "content": "x"}, message],
                record.get("tools"),
                allow_other_identity=allow,
                check_normalized=check_normalized,
            )
        ]
    if record["chosen"] == record["rejected"]:
        problems.append("chosen and rejected are identical")
    return problems


def validate_document(record: dict, check_normalized: bool = True) -> list[str]:
    """Document records for (continued) pre-training: ``{"id", "text", "lang", "source", "license", ...}``."""
    problems = [
        f"missing field {key!r}" for key in ("id", "text", "lang", "source", "license") if key not in record
    ]
    if problems:
        return problems
    text = record["text"]
    if not isinstance(text, str) or not text.strip():
        return ["empty text"]
    if check_normalized and not is_normalized(text):
        problems.append("text is not normalised (run dsm.text.normalize_text)")
    if leak := identity_leak(text):
        problems.append(f"identity leak {leak!r}")
    return problems


def record_kind(record: dict) -> str:
    """``sft`` | ``preference`` | ``document``."""
    if "chosen" in record:
        return "preference"
    if "messages" in record:
        return "sft"
    return "document"


def validate_record(record: dict, check_normalized: bool = True) -> list[str]:
    check = {"sft": validate_sft, "preference": validate_preference, "document": validate_document}[
        record_kind(record)
    ]
    return check(record, check_normalized=check_normalized)
