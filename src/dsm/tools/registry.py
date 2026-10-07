"""Tool registry: JSON-schema tools that DSM can call, with argument validation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dsm.data.sft.schema import function_schema


@dataclass
class Tool:
    name: str
    description: str
    properties: dict[str, dict]
    required: list[str]
    fn: Callable[..., Any]
    # state-changing / money-moving tools must be confirmed by the user first
    side_effect: bool = False
    tags: list[str] = field(default_factory=list)

    def schema(self) -> dict:
        return function_schema(self.name, self.description, self.properties, self.required)

    def validate(self, arguments: dict) -> list[str]:
        problems = [f"missing required argument {key!r}" for key in self.required if key not in arguments]
        problems += [f"unknown argument {key!r}" for key in arguments if key not in self.properties]
        try:
            import jsonschema

            jsonschema.validate(arguments, self.schema()["function"]["parameters"])
        except ImportError:  # pragma: no cover
            pass
        except Exception as exc:  # jsonschema.ValidationError
            problems.append(getattr(exc, "message", str(exc)))
        return problems

    def __call__(self, **arguments: Any) -> Any:
        problems = self.validate(arguments)
        if problems:
            return {"error": "; ".join(problems)}
        try:
            return self.fn(**arguments)
        except (KeyError, ValueError, TypeError) as exc:
            return {"error": str(exc)}


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __getitem__(self, name: str) -> Tool:
        return self.tools[name]

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        selected = names if names is not None else list(self.tools)
        return [self.tools[name].schema() for name in selected]

    def call(self, name: str, arguments: dict) -> str:
        """Execute a call and return the JSON text placed in the ``tool`` message."""
        if name not in self.tools:
            return json.dumps({"error": f"unknown tool {name!r}"})
        return json.dumps(self.tools[name](**arguments), ensure_ascii=False)
