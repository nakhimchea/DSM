"""Converters from public datasets to DSM SFT / preference records.

Every converter takes ONE raw row - as loaded with ``datasets.load_dataset`` or
from the original JSON / JSONL release - and returns a :class:`Converted`
conversation, a list of them, or ``None`` when the row cannot be converted
faithfully (missing fields, unparsable tool calls, reasoning traces, foreign
identities ...). :func:`convert_dataset` wraps the result into records with
licence / provenance fields and runs the record validator, so a converter never
has to be trusted blindly.

Formats handled (field names of the public releases):

=================  ==========================================================
chat / messages    ``messages`` | ``conversations`` with role/content or
                   from/value (SmolTalk2, Nemotron post-training, When2Call
                   and other OpenAI-style data incl. ``tool_calls``)
gsm8k              ``question``, ``answer`` (``<<a=b>>`` calculator notes,
                   ``#### 42`` final answer)
numinamath / math  ``problem``, ``solution`` (+ validity flags in 1.5)
xlam               ``query``, ``tools`` (JSON string), ``answers`` (JSON string)
hermes_fc          ``conversations`` with ``<tools>`` in the system prompt,
                   ``<tool_call>`` / ``<tool_response>`` blocks
glaive_fc_v2       ``system`` ("... -\\n{json}{json}"), ``chat`` with
                   ``USER:`` / ``ASSISTANT: <functioncall>`` / ``FUNCTION RESPONSE:``
toolace            ``system`` with a JSON tool list, ``conversations`` with
                   Python-style calls ``[f(a=1), g(b="x")]``
tabfact            ``table_text`` (``#``-separated), ``statement``, ``label``
wtq                ``table`` {header, rows}, ``question``, ``answers``
tatqa              ``table`` {table}, ``paragraphs``, ``questions`` (one row ->
                   several records)
finqa              ``pre_text``, ``table``, ``post_text``, ``qa`` {question,
                   program, exe_ans}; the program is re-executed and rendered
                   as visible steps
massive            ``utt``, ``annot_utt`` ("[time : five am]"), ``intent``,
                   ``locale``; intents become functions, slots arguments
helpsteer3         ``context``, ``response1``, ``response2``,
                   ``overall_preference`` -> preference pairs
=================  ==========================================================

Licences in :data:`DATASETS` are the defaults of the public releases; the
licence gate in ``configs/data/sources.yaml`` is authoritative and is passed in
by the CLI (``dsm build convert``).
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from dsm.data.sft.schema import preference_record, sft_record
from dsm.data.sft.validate import validate_preference, validate_sft
from dsm.data.synth.filters import detect_lang, foreign_identity, identity_leak, is_identity_question
from dsm.format.validate import check_format, dollars_to_usd
from dsm.tables.serialize import grid_to_text
from dsm.text import normalize_text
from dsm.tools.calculator import evaluate as calculator_evaluate

MAX_TABLE_ROWS = 60


@dataclass
class Converted:
    """One conversation produced by a converter (wrapped into a record by :func:`convert_dataset`)."""

    messages: list[dict]
    tools: list[dict] | None = None
    system: str | None = None
    lang: str | None = None  # None -> detected from the text
    category: str | None = None  # None -> the dataset's default category
    meta: dict = field(default_factory=dict)


@dataclass
class ConvertedPair:
    prompt: list[dict]
    chosen: str
    rejected: str
    lang: str | None = None
    category: str | None = None
    meta: dict = field(default_factory=dict)


# -- small parsing helpers -----------------------------------------------------------------


def _load(value: Any) -> Any:
    """Parse JSON, falling back to Python literals (several datasets store ``repr`` output)."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return None


def _arguments(value: Any) -> dict | None:
    if value is None or value == "":
        return {}
    parsed = _load(value)
    return parsed if isinstance(parsed, dict) else None


def _json_objects(text: str) -> list | None:
    """All top-level JSON values in ``text`` (``None`` if one of them is malformed)."""
    decoder = json.JSONDecoder()
    values, i = [], 0
    while True:
        starts = [j for j in (text.find("{", i), text.find("[", i)) if j >= 0]
        if not starts:
            return values
        i = min(starts)
        try:
            value, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            return None
        values.append(value)


def _text(content: Any) -> str | None:
    """Message content as text (``None`` for non-text parts such as images)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            else:
                return None
        return "".join(parts)
    return None


def _literal(node: ast.AST) -> Any:
    """``ast.literal_eval`` that also accepts JSON's true / false / null."""
    names = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values, strict=True)}
    return ast.literal_eval(node)


def pythonic_calls(text: str) -> list[dict] | None:
    """Parse ``[f(a=1, b="x"), g()]`` (ToolACE / BFCL style) into structured calls."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return None
    nodes = tree.body.elts if isinstance(tree.body, ast.List) else [tree.body]
    calls = []
    for node in nodes:
        if not isinstance(node, ast.Call) or node.args:
            return None
        try:
            arguments = {kw.arg: _literal(kw.value) for kw in node.keywords}
        except (ValueError, TypeError, SyntaxError):
            return None
        if None in arguments:  # **kwargs
            return None
        calls.append({"name": ast.unparse(node.func), "arguments": arguments})
    return calls or None


# -- tool schemas ----------------------------------------------------------------------------

_TYPES = {
    "str": "string",
    "string": "string",
    "text": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "dict": "object",
    "object": "object",
    "list": "array",
    "array": "array",
    "tuple": "array",
    "set": "array",
    "null": "null",
}


def _json_type(raw: Any) -> str | None:
    text = str(raw).strip().lower().replace(", optional", "").replace(",optional", "").strip()
    if text.startswith("optional[") and text.endswith("]"):
        text = text[len("optional[") : -1]
    for prefix, kind in (("list", "array"), ("tuple", "array"), ("set", "array"), ("dict", "object")):
        if text.startswith(prefix):
            return kind
    return _TYPES.get(text)


def _fix_schema(node: Any) -> Any:
    """Map dataset-specific schema dialects ("dict", "float", "str, optional" ...) to JSON Schema."""
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "type":
            kinds = [k for k in (_json_type(v) for v in (value if isinstance(value, list) else [value])) if k]
            if kinds:
                out["type"] = kinds[0] if len(kinds) == 1 else kinds
        elif key == "properties" and isinstance(value, dict):
            props = {name: _fix_schema(prop) for name, prop in value.items()}
            flagged = [n for n, p in value.items() if isinstance(p, dict) and p.get("required") is True]
            for name in props:
                if isinstance(props[name], dict):
                    props[name].pop("required", None)
            out["properties"] = props
            if flagged:
                out["required"] = sorted(set(out.get("required", [])) | set(flagged))
        elif key == "items":
            out["items"] = _fix_schema(value)
        elif key == "required":
            if isinstance(value, list):
                out["required"] = sorted(set(out.get("required", [])) | {str(v) for v in value})
        else:
            out[key] = value
    if "properties" in out:
        out["type"] = "object"
        out["required"] = [r for r in out.get("required", []) if r in out["properties"]]
    return out


def tool_from_spec(spec: Any) -> dict | None:
    """An OpenAI-style function tool from an OpenAI, flat, or xLAM-style specification."""
    if not isinstance(spec, dict):
        return None
    fn = spec.get("function", spec)
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    params = fn.get("parameters") or {}
    if not isinstance(params, dict):
        return None
    if "properties" in params or params.get("type") in ("object", "dict"):
        schema = _fix_schema({**params, "properties": params.get("properties") or {}})
    else:  # xLAM: {"param": {"type": "str, optional", "description": ..., "default": ...}}
        properties, required = {}, []
        for pname, pspec in params.items():
            if not isinstance(pspec, dict):
                return None
            prop = {k: v for k, v in pspec.items() if k not in ("type", "default", "required")}
            if (kind := _json_type(pspec.get("type", ""))) is not None:
                prop["type"] = kind
            properties[pname] = prop
            if "optional" not in str(pspec.get("type", "")).lower() and "default" not in pspec:
                required.append(pname)
        schema = {"type": "object", "properties": properties, "required": required}
    return {
        "type": "function",
        "function": {"name": name, "description": str(fn.get("description") or ""), "parameters": schema},
    }


def _tools(specs: Any) -> list[dict] | None:
    specs = _load(specs)
    if not isinstance(specs, list) or not specs:
        return None
    tools = [tool_from_spec(s) for s in specs]
    return None if any(t is None for t in tools) else tools


def _call(obj: Any) -> dict | None:
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function", obj)
    name, arguments = fn.get("name"), _arguments(fn.get("arguments", fn.get("parameters")))
    if not isinstance(name, str) or not name or arguments is None:
        return None
    return {"name": name, "arguments": arguments}


# -- chat / OpenAI-style messages ----------------------------------------------------------------

_ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "chatgpt": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "ipython": "tool",
    "observation": "tool",
}
_THINK = re.compile(r"<think>(.*?)</think>\s*", re.S)
_MODE_FLAG = re.compile(r"[ \t]*/(?:no_)?think\b[ \t]*")
_HERMES_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_HERMES_RESPONSE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)
_NV_CALL = re.compile(r"<TOOLCALL>\s*(.*?)\s*</TOOLCALL>", re.S)


def _text_calls(text: str) -> tuple[str, list[dict]] | None:
    """Tool calls written into assistant text (Hermes ``<tool_call>``, ``<TOOLCALL>[...]``)."""
    calls: list[dict] = []
    for pattern, many in ((_HERMES_CALL, False), (_NV_CALL, True)):
        for payload in pattern.findall(text):
            parsed = _load(payload)
            for obj in parsed if many and isinstance(parsed, list) else [parsed]:
                call = _call(obj)
                if call is None:
                    return None
                calls.append(call)
        text = pattern.sub("", text)
    return text.strip(), calls


def _messages_field(row: dict) -> list | None:
    for key in ("messages", "conversations", "conversation"):
        if isinstance(row.get(key), list):
            return row[key]
    return None


def _upstream_model_names() -> tuple[str, ...]:
    from dsm.utils import load_upstream

    try:
        return tuple(n.lower() for n in load_upstream()["scrub"]["probe_names"])
    except FileNotFoundError:  # pragma: no cover
        return ()


def _mentions_upstream_model(text: str) -> bool:
    lowered = text.lower()
    return any(name in lowered for name in _upstream_model_names())


def convert_chat(row: dict, *, keep_identity_questions: bool = False) -> Converted | None:
    """Generic conversation converter (role/content or from/value; OpenAI tool calls supported).

    Drops: reasoning traces (DSM answers directly), non-text content, foreign
    identities in assistant text, identity questions (DSM's own identity data
    answers those), and rows mentioning the upstream model by name.
    """
    raw = _messages_field(row)
    if not raw or str(row.get("reasoning", "off")).lower() == "on":
        return None
    tools = _tools(row["tools"]) if row.get("tools") else None
    if row.get("tools") and tools is None:
        return None
    system, messages, pending_calls = None, [], []
    for i, message in enumerate(raw):
        role = _ROLES.get(str(message.get("role", message.get("from", ""))).lower())
        content = _text(message["content"] if "content" in message else message.get("value"))
        if role is None or content is None:
            return None
        if role == "system":
            if i != 0:
                return None
            system = _MODE_FLAG.sub(" ", content).strip() or None
            continue
        if role == "user":
            content = _MODE_FLAG.sub(" ", content).strip()
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            if message.get("reasoning_content"):
                return None
            match = _THINK.search(content)
            if match and match.group(1).strip():
                return None
            content = _THINK.sub("", content)
            parsed = _text_calls(content)
            if parsed is None:
                return None
            content, calls = parsed
            structured = [_call(c) for c in message.get("tool_calls") or []]
            if any(c is None for c in structured):
                return None
            calls = structured + calls
            out = {"role": "assistant", "content": content.strip()}
            if calls:
                out["tool_calls"] = calls
                pending_calls = [c["name"] for c in calls]
            messages.append(out)
        else:  # tool result(s)
            responses = _HERMES_RESPONSE.findall(content) or [content]
            for response in responses:
                name = message.get("name") or (pending_calls.pop(0) if pending_calls else None)
                parsed = _load(response)
                if isinstance(parsed, dict) and set(parsed) == {"name", "content"}:  # Hermes envelope
                    content = parsed["content"]
                    name = parsed["name"]
                    response = (
                        content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                    )
                if name is None:
                    return None
                messages.append({"role": "tool", "name": name, "content": response.strip()})
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if not messages or messages[0]["role"] != "user":
        return None
    if any(m.get("tool_calls") for m in messages) and not tools:
        return None
    assistant_text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    if identity_leak(assistant_text) or _mentions_upstream_model(assistant_text):
        return None
    if not keep_identity_questions and is_identity_question(messages[0]["content"]):
        return None
    if system and (foreign_identity(system) or _mentions_upstream_model(system)):
        system = None  # e.g. "You are <other assistant>, trained by ..." -> DSM default prompt
    category = "tools" if tools else None
    meta = {k: row[k] for k in ("source", "subset", "category") if isinstance(row.get(k), str)}
    return Converted(messages, tools=tools, system=system, category=category, meta=meta)


# -- math ---------------------------------------------------------------------------------------

_CALC_NOTE = re.compile(r"<<[^>]*>>")


def convert_gsm8k(row: dict) -> Converted | None:
    question, answer = row.get("question"), row.get("answer")
    if not question or not answer or "####" not in answer:
        return None
    body, _, final = answer.rpartition("####")
    steps = dollars_to_usd(_CALC_NOTE.sub("", body).strip())
    final = final.strip().replace(",", "")
    text = f"{steps}\n\n**Answer: {final}**"
    if check_format(text):
        return None
    return Converted(
        [{"role": "user", "content": question.strip()}, {"role": "assistant", "content": text}], lang="en"
    )


def convert_numinamath(row: dict, *, skip_proofs: bool = True) -> Converted | None:
    """NuminaMath (1.5 / CoT) and MATH-style ``problem`` / ``solution`` rows."""
    problem, solution = row.get("problem"), row.get("solution")
    if not problem or not solution:
        return None
    for key in ("problem_is_valid", "solution_is_valid"):
        value = row.get(key)
        if value is not None and str(value).strip().lower() not in ("yes", "true", "1", "valid"):
            return None
    answer = str(row.get("answer", "")).strip().lower()
    if skip_proofs and (
        str(row.get("question_type", "")).lower() == "proof" or answer in ("proof", "notfound")
    ):
        return None
    if check_format(solution):
        return None
    meta = {
        k: row[k] for k in ("source", "problem_type", "type", "level") if isinstance(row.get(k), (str, int))
    }
    return Converted(
        [{"role": "user", "content": problem.strip()}, {"role": "assistant", "content": solution.strip()}],
        lang=detect_lang(problem),
        meta=meta,
    )


# -- tool calling ----------------------------------------------------------------------------------


def convert_xlam(row: dict) -> Converted | None:
    query, tools, answers = row.get("query"), _tools(row.get("tools")), _load(row.get("answers"))
    if not query or not tools or not isinstance(answers, list) or not answers:
        return None
    calls = [_call(a) for a in answers]
    if any(c is None for c in calls):
        return None
    return Converted(
        [
            {"role": "user", "content": query.strip()},
            {"role": "assistant", "content": "", "tool_calls": calls},
        ],
        tools=tools,
    )


_TOOLS_BLOCK = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.S)


def convert_hermes_fc(row: dict) -> Converted | None:
    """Hermes function calling: tools in the system prompt, XML-wrapped JSON calls / responses."""
    raw = _messages_field(row)
    if not raw or _ROLES.get(str(raw[0].get("from", raw[0].get("role", ""))).lower()) != "system":
        return None
    tools = None
    # The preamble itself mentions "<tools> </tools>", so take the first block that parses.
    for block in _TOOLS_BLOCK.findall(_text(raw[0].get("value", raw[0].get("content"))) or ""):
        specs = _load(block)
        if specs is None:
            specs = _json_objects(block)
        if specs and (tools := _tools(specs if isinstance(specs, list) else [specs])):
            break
    if not tools:
        return None  # json-mode / no tools
    return convert_chat({"conversations": raw[1:], "tools": tools}, keep_identity_questions=False)


_GLAIVE_TURN = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):\s*")
_GLAIVE_CALL = re.compile(r"\{\s*\"name\"\s*:\s*\"([^\"]+)\"\s*,\s*\"arguments\"\s*:\s*'(.*)'\s*\}", re.S)


def _glaive_call(text: str) -> dict | None:
    if match := _GLAIVE_CALL.search(text):
        arguments = _arguments(match.group(2))
        return {"name": match.group(1), "arguments": arguments} if arguments is not None else None
    objects = _json_objects(text)
    return _call(objects[0]) if objects else None


def convert_glaive_fc(row: dict, *, require_tools: bool = True) -> Converted | None:
    system, chat = row.get("system") or "", row.get("chat") or ""
    _, _, spec_text = system.partition("-")
    specs = _json_objects(spec_text) if "{" in spec_text else []
    if specs is None:
        return None
    tools = _tools(specs) if specs else None
    if require_tools and not tools:
        return None
    parts = _GLAIVE_TURN.split(chat)
    messages, last_call = [], None
    for label, text in zip(parts[1::2], parts[2::2], strict=True):
        text = text.replace("<|endoftext|>", "").strip()
        if label == "USER":
            messages.append({"role": "user", "content": text})
        elif label == "ASSISTANT":
            before, marker, call_text = text.partition("<functioncall>")
            if marker:
                call = _glaive_call(call_text)
                if call is None:
                    return None
                last_call = call["name"]
                messages.append({"role": "assistant", "content": before.strip(), "tool_calls": [call]})
            else:
                messages.append({"role": "assistant", "content": text})
        else:
            if last_call is None:
                return None
            messages.append({"role": "tool", "name": last_call, "content": text})
    return convert_chat({"messages": messages, "tools": tools} if tools else {"messages": messages})


def convert_toolace(row: dict) -> Converted | None:
    system = row.get("system") or ""
    start = system.find("[")
    if start < 0:
        return None
    try:
        specs, _ = json.JSONDecoder().raw_decode(system, start)
    except json.JSONDecodeError:
        return None
    tools = _tools(specs)
    if not tools:
        return None
    messages = []
    for message in row.get("conversations") or []:
        role = _ROLES.get(str(message.get("from", message.get("role", ""))).lower())
        text = (_text(message.get("value", message.get("content"))) or "").strip()
        if role == "assistant" and text.startswith("[") and text.endswith("]") and "(" in text:
            calls = pythonic_calls(text)
            if calls is None:
                return None
            messages.append({"role": "assistant", "content": "", "tool_calls": calls})
        elif role == "tool":
            results = _load(text)
            for result in results if isinstance(results, list) else [results]:
                if not isinstance(result, dict) or "name" not in result:
                    return None
                payload = result.get("results", result.get("content", {}))
                messages.append(
                    {
                        "role": "tool",
                        "name": result["name"],
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                )
        elif role in ("user", "assistant"):
            messages.append({"role": role, "content": text})
        else:
            return None
    return convert_chat({"messages": messages, "tools": tools})


# -- tables -------------------------------------------------------------------------------------


def _table_too_long(rows: list) -> bool:
    return len(rows) - 1 > MAX_TABLE_ROWS


def convert_tabfact(row: dict) -> Converted | None:
    table_text, statement, label = row.get("table_text"), row.get("statement"), row.get("label")
    if not table_text or not statement or label not in (0, 1):
        return None
    rows = [line.split("#") for line in table_text.strip("\n").split("\n")]
    if _table_too_long(rows):
        return None
    table = grid_to_text(rows, row.get("table_caption") or "Table", style="grid")
    user = f"{table}\n\nAccording to the table, is this statement true or false?\n\n{statement.strip()}"
    answer = (
        "**True.** The table supports the statement."
        if label == 1
        else "**False.** The table does not support the statement."
    )
    return Converted([{"role": "user", "content": user}, {"role": "assistant", "content": answer}], lang="en")


def convert_wtq(row: dict) -> Converted | None:
    table, question, answers = row.get("table") or {}, row.get("question"), row.get("answers")
    header, body = table.get("header"), table.get("rows")
    if not question or not answers or not header or body is None:
        return None
    rows = [list(header), *[list(r) for r in body]]
    if _table_too_long(rows):
        return None
    name = str(table.get("name") or "Table").rsplit("/", 1)[-1].removesuffix(".csv")
    text = grid_to_text(rows, name, style="grid")
    answer = ", ".join(str(a) for a in answers)
    return Converted(
        [
            {"role": "user", "content": f"{text}\n\n{question.strip()}"},
            {"role": "assistant", "content": answer},
        ],
        lang="en",
    )


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).replace(",", "").replace("$", "").strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()% ")
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def _fmt_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return f"{round(value):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _tatqa_answer(question: dict) -> str | None:
    answer, scale = question.get("answer"), (question.get("scale") or "").strip()
    if isinstance(answer, list):
        if not answer:
            return None
        text = ", ".join(str(a).strip() for a in answer)
    elif isinstance(answer, (int, float)):
        text = _fmt_number(float(answer))
    elif isinstance(answer, str) and answer.strip():
        text = answer.strip()
    else:
        return None
    if scale == "percent":
        return f"{text}%"
    return f"{text} {scale}" if scale else text


def _tatqa_derivation(question: dict) -> str | None:
    """The arithmetic derivation, if it re-computes to the gold answer."""
    derivation, answer = question.get("derivation") or "", question.get("answer")
    if question.get("answer_type") != "arithmetic" or not derivation or not isinstance(answer, (int, float)):
        return None
    try:
        value = float(calculator_evaluate(derivation.replace(",", "").replace("$", "")))
    except (ValueError, ZeroDivisionError, TypeError, SyntaxError, OverflowError):
        return None
    candidates = [value, value * 100] if question.get("scale") == "percent" else [value]
    if not any(math.isclose(c, float(answer), rel_tol=5e-3, abs_tol=0.006) for c in candidates):
        return None
    suffix = "%" if question.get("scale") == "percent" else ""
    return f"{derivation.strip()} = {_fmt_number(float(answer))}{suffix}"


def convert_tatqa(row: dict) -> list[Converted] | None:
    """TAT-QA (financial report table + paragraphs): one conversation per question."""
    table = (row.get("table") or {}).get("table")
    if not table or _table_too_long(table):
        return None
    paragraphs = sorted(row.get("paragraphs") or [], key=lambda p: p.get("order", 0))
    context = grid_to_text(table, "Report", header_rows=1, style="markdown")
    if paragraphs:
        context += "\n\n" + "\n\n".join(p["text"].strip() for p in paragraphs if p.get("text"))
    out = []
    for question in row.get("questions") or []:
        answer = _tatqa_answer(question)
        if not answer or not question.get("question"):
            continue
        derivation = _tatqa_derivation(question)
        reply = f"{derivation}\n\n**Answer:** {answer}" if derivation else f"**Answer:** {answer}"
        out.append(
            Converted(
                [
                    {"role": "user", "content": f"{context}\n\nQuestion: {question['question'].strip()}"},
                    {"role": "assistant", "content": reply},
                ],
                lang="en",
                meta={"answer_type": question.get("answer_type"), "answer_from": question.get("answer_from")},
            )
        )
    return out or None


_FINQA_STEP = re.compile(r"(\w+)\(([^()]*)\)")
_FINQA_OPS: dict[str, tuple[str, Callable[[float, float], Any]]] = {
    "add": ("+", lambda a, b: a + b),
    "subtract": ("−", lambda a, b: a - b),
    "multiply": ("×", lambda a, b: a * b),
    "divide": ("÷", lambda a, b: a / b),
    "exp": ("^", lambda a, b: a**b),
    "greater": (">", lambda a, b: a > b),
}
_FINQA_TABLE_OPS = {
    "table_sum": ("sum", sum),
    "table_average": ("average", lambda xs: sum(xs) / len(xs)),
    "table_max": ("maximum", max),
    "table_min": ("minimum", min),
}


def run_finqa_program(program: str, table: list[list[str]]) -> tuple[Any, list[str]] | None:
    """Execute a FinQA program (``subtract(5829, 5735), divide(#0, 5735)``) -> (result, readable steps)."""
    results: list[Any] = []
    steps: list[str] = []

    def arg(token: str) -> float | None:
        token = token.strip()
        if token.startswith("#"):
            index = int(token[1:])
            return results[index] if index < len(results) else None
        if token.startswith("const_"):
            constant = token[len("const_") :]
            return -1.0 if constant == "m1" else _number(constant)
        number = _number(token)
        if number is not None and token.endswith("%"):
            number /= 100
        return number

    for op, raw_args in _FINQA_STEP.findall(program):
        args = [a.strip() for a in raw_args.split(",")]
        if op in _FINQA_OPS and len(args) == 2:
            a, b = arg(args[0]), arg(args[1])
            if a is None or b is None:
                return None
            symbol, fn = _FINQA_OPS[op]
            try:
                value = fn(a, b)
            except (ZeroDivisionError, OverflowError):
                return None
            shown = "yes" if value is True else "no" if value is False else _fmt_number(value)
            steps.append(f"{_fmt_number(a)} {symbol} {_fmt_number(b)} = {shown}")
        elif op in _FINQA_TABLE_OPS:
            label = args[0].lower()
            match = next((r for r in table if r and str(r[0]).strip().lower() == label), None)
            values = [v for v in (_number(c) for c in (match or [])[1:]) if v is not None]
            if not values:
                return None
            word, fn = _FINQA_TABLE_OPS[op]
            value = fn(values)
            steps.append(
                f"{word} of “{match[0]}” ({', '.join(_fmt_number(v) for v in values)}) = {_fmt_number(value)}"
            )
        else:
            return None
        results.append(value)
    return (results[-1], steps) if results else None


def convert_finqa(row: dict) -> Converted | None:
    qa, table = row.get("qa") or {}, row.get("table") or []
    question, program, gold = qa.get("question"), qa.get("program"), qa.get("exe_ans")
    if not question or not program or gold is None or not table or _table_too_long(table):
        return None
    executed = run_finqa_program(program, table)
    if executed is None:
        return None
    value, steps = executed
    if isinstance(gold, str):  # yes / no questions
        if gold.lower() not in ("yes", "no") or (value is True) != (gold.lower() == "yes"):
            return None
        answer = gold.lower()
    else:
        if isinstance(value, bool) or not math.isclose(float(value), float(gold), rel_tol=1e-3, abs_tol=1e-4):
            return None
        answer = str(qa.get("answer") or "").strip() or _fmt_number(float(gold))
    context = " ".join(row.get("pre_text") or []).strip()
    context += "\n\n" + grid_to_text(table, "Table", style="markdown")
    if post := " ".join(row.get("post_text") or []).strip():
        context += "\n\n" + post
    reply = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps)) + f"\n\n**Answer:** {answer}"
    return Converted(
        [
            {"role": "user", "content": f"{context}\n\nQuestion: {question.strip()}"},
            {"role": "assistant", "content": reply},
        ],
        lang="en",
        meta={"program": program},
    )


# -- MASSIVE (multilingual intents / slots -> function calls) ------------------------------------------

_SLOT = re.compile(r"\[\s*([\w-]+)\s*:\s*([^\]]+?)\s*\]")
_LOCALES = {"km": "km", "zh-cn": "zh", "en": "en"}


def _massive_lang(locale: str) -> str | None:
    locale = locale.lower().replace("_", "-")
    if locale in _LOCALES:
        return _LOCALES[locale]
    return _LOCALES.get(locale.split("-")[0]) if locale.split("-")[0] != "zh" else None


class MassiveConverter:
    """Intents become functions, slot types become string parameters.

    Each sample offers every function of the utterance's scenario (e.g.
    ``alarm_set``, ``alarm_query``, ``alarm_remove``), so the model has to pick
    the right one. Built from the whole split first, because the function
    schemas are derived from all slot types seen per intent.
    """

    def __init__(self, rows: Iterable[dict], intent_names: list[str] | None = None) -> None:
        self.intent_names = intent_names
        slots: dict[str, dict[str, set[str]]] = {}
        for row in rows:
            intent = self._intent(row)
            if intent is None:
                continue
            scenario = intent.split("_")[0]
            types = {m.group(1) for m in _SLOT.finditer(row.get("annot_utt", ""))}
            slots.setdefault(scenario, {}).setdefault(intent, set()).update(types)
        self.tools = {
            scenario: [self._schema(intent, sorted(types)) for intent, types in sorted(intents.items())]
            for scenario, intents in slots.items()
        }

    def _intent(self, row: dict) -> str | None:
        intent = row.get("intent")
        if isinstance(intent, int):
            return (
                self.intent_names[intent]
                if self.intent_names and 0 <= intent < len(self.intent_names)
                else None
            )
        return intent if isinstance(intent, str) and intent else None

    @staticmethod
    def _schema(intent: str, slot_types: list[str]) -> dict:
        scenario, _, action = intent.partition("_")
        properties = {s: {"type": "string", "description": s.replace("_", " ")} for s in slot_types}
        return {
            "type": "function",
            "function": {
                "name": intent,
                "description": f"{scenario} skill: {action.replace('_', ' ') or scenario} request.",
                "parameters": {"type": "object", "properties": properties, "required": []},
            },
        }

    def __call__(self, row: dict) -> Converted | None:
        intent = self._intent(row)
        lang = _massive_lang(row.get("locale", "en-US"))
        utterance = normalize_text(row.get("utt") or "").strip()
        if intent is None or lang is None or not utterance or intent == "general_quirky":
            return None
        arguments: dict[str, str] = {}
        for match in _SLOT.finditer(row.get("annot_utt", "")):
            if match.group(1) in arguments:
                return None  # repeated slot type: not representable as one string argument
            arguments[match.group(1)] = normalize_text(match.group(2)).strip()
        call = {"name": intent, "arguments": arguments}
        return Converted(
            [
                {"role": "user", "content": utterance},
                {"role": "assistant", "content": "", "tool_calls": [call]},
            ],
            tools=self.tools[intent.split("_")[0]],
            lang=lang,
            meta={"intent": intent},
        )


# -- preference data ----------------------------------------------------------------------------------


def convert_helpsteer3(row: dict, *, min_margin: int = 2) -> ConvertedPair | None:
    """HelpSteer3 preference: ``overall_preference`` < 0 means response1 is better."""
    context, preference = row.get("context"), row.get("overall_preference")
    first, second = row.get("response1"), row.get("response2")
    if not isinstance(context, list) or preference is None or not first or not second:
        return None
    if abs(int(preference)) < min_margin:
        return None
    chosen, rejected = (first, second) if preference < 0 else (second, first)
    prompt = []
    for message in context:
        role = _ROLES.get(str(message.get("role", "")).lower())
        content = _text(message.get("content"))
        if role not in ("user", "assistant") or content is None:
            return None
        prompt.append({"role": role, "content": content.strip()})
    if not prompt or prompt[-1]["role"] != "user" or is_identity_question(prompt[0]["content"]):
        return None
    lang = detect_lang(prompt[-1]["content"])
    if str(row.get("language", "")).lower() not in ("", "english", "chinese", "en", "zh"):
        return None
    return ConvertedPair(
        prompt, chosen.strip(), rejected.strip(), lang=lang, meta={"domain": row.get("domain")}
    )


# -- registry and driver ------------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    convert: Callable[[dict], Any] | None
    category: str
    license: str | None
    kind: str = "sft"  # sft | preference
    prepare: Callable[[list[dict]], Callable[[dict], Any]] | None = None


DATASETS: dict[str, DatasetSpec] = {
    "chat": DatasetSpec(convert_chat, "chat", None),  # SmolTalk2, Nemotron, When2Call ... licence per source
    "gsm8k": DatasetSpec(convert_gsm8k, "math", "MIT"),
    "numinamath": DatasetSpec(convert_numinamath, "math", "Apache-2.0"),
    "competition_math": DatasetSpec(convert_numinamath, "math", "MIT"),
    "xlam": DatasetSpec(convert_xlam, "tools", "CC-BY-4.0"),
    "hermes_fc": DatasetSpec(convert_hermes_fc, "tools", "Apache-2.0"),
    "glaive_fc_v2": DatasetSpec(convert_glaive_fc, "tools", "Apache-2.0"),
    "toolace": DatasetSpec(convert_toolace, "tools", "Apache-2.0"),
    "tabfact": DatasetSpec(convert_tabfact, "tables", "CC-BY-4.0"),
    "wtq": DatasetSpec(convert_wtq, "tables", "CC-BY-SA-4.0"),
    "tatqa": DatasetSpec(convert_tatqa, "tables", "CC-BY-4.0"),
    "finqa": DatasetSpec(convert_finqa, "tables", "MIT"),
    "massive": DatasetSpec(None, "tools", "CC-BY-4.0", prepare=MassiveConverter),
    "helpsteer3": DatasetSpec(convert_helpsteer3, "chat", "CC-BY-4.0", kind="preference"),
}


def _finalize_messages(messages: list[dict]) -> list[dict]:
    out = []
    for message in messages:
        message = dict(message)
        message["content"] = normalize_text(message.get("content") or "")
        out.append(message)
    return out


def convert_dataset(
    name: str,
    rows: Iterable[dict],
    *,
    license: str | None = None,
    generated_by: str | None = None,
    source: str | None = None,
) -> tuple[list[dict], Counter]:
    """Convert and validate a whole dataset -> (records, statistics of kept / dropped rows)."""
    spec = DATASETS[name]
    license = license or spec.license
    if not license:
        raise ValueError(
            f"dataset {name!r} needs an explicit licence (take it from configs/data/sources.yaml)"
        )
    source = source or name
    generated_by = generated_by or f"dataset:{source}"
    if spec.prepare is not None:
        rows = list(rows)
        convert = spec.prepare(rows)
    else:
        convert = spec.convert
    records: list[dict] = []
    stats: Counter = Counter()
    for index, row in enumerate(rows):
        try:
            result = convert(row)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError):
            stats["malformed"] += 1
            continue
        if not result:
            stats["dropped"] += 1
            continue
        items = result if isinstance(result, list) else [result]
        for k, item in enumerate(items):
            rid = f"{source}-{index:07d}" + (f"-{k}" if len(items) > 1 else "")
            if spec.kind == "preference":
                record = preference_record(
                    id=rid,
                    source=source,
                    license=license,
                    generated_by=generated_by,
                    prompt=_finalize_messages(item.prompt),
                    chosen={"role": "assistant", "content": normalize_text(item.chosen)},
                    rejected={"role": "assistant", "content": normalize_text(item.rejected)},
                    lang=item.lang or "en",
                    category=item.category or spec.category,
                    meta=item.meta,
                )
                problems = validate_preference(record)
            else:
                messages = _finalize_messages(item.messages)
                lang = item.lang or detect_lang(" ".join(m["content"] for m in messages[:2]))
                record = sft_record(
                    id=rid,
                    source=source,
                    license=license,
                    generated_by=generated_by,
                    messages=messages,
                    lang=lang,
                    category=item.category or spec.category,
                    system=normalize_text(item.system) if item.system else None,
                    tools=item.tools,
                    meta=item.meta,
                )
                problems = validate_sft(record)
            if problems:
                stats["invalid"] += 1
                stats[f"invalid: {problems[0].split(':')[0]}"] += 1
                continue
            records.append(record)
            stats["kept"] += 1
    return records, stats
