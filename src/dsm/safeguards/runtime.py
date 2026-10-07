"""Runtime safeguards for applications that serve DSM.

The model is one layer of defence; the application is the other. These helpers
implement the rules of ``configs/safety_policy.yaml`` that must NOT depend on
the model behaving:

* :func:`mask_pii` - card / account / phone numbers, e-mail addresses and
  one-time codes are masked before anything is written to a log;
* :class:`ToolGate` - only allow-listed tools run, arguments are validated, and
  tools with side effects (money transfers, card blocking) run only after the
  USER confirmed the exact action in the application. A model that claims
  ``confirmed=true`` on its own gets a confirmation request back instead;
* :func:`flag_injection` - marks tool results / documents that contain
  instructions aimed at the model ("ignore previous instructions ..."), so the
  application can log or strip them.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

_CARD = re.compile(r"\b(?:\d[ -]?){12,18}(\d{4})\b")
_ACCOUNT = re.compile(r"\b(ACC)-(\d{2,})\b")
_PHONE = re.compile(r"(?<![\w+])(\+?855|0)[ -]?(\d{2})[ -]?(\d{3})[ -]?(\d{3,4})\b")
_EMAIL = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_CODE = re.compile(
    r"(?i)\b(OTP|PIN|code|passcode|verification code|验证码|密码|លេខកូដ)(\s*(?:is|:|：|=)?\s*)(\d{4,8})\b"
)


def mask_pii(text: str) -> str:
    """Mask personal / secret data for logs (keeps enough to debug: last 4 digits, domain)."""
    text = _CODE.sub(lambda m: f"{m.group(1)}{m.group(2)}{'*' * len(m.group(3))}", text)
    text = _CARD.sub(lambda m: "**** **** **** " + m.group(1), text)
    text = _ACCOUNT.sub(lambda m: f"{m.group(1)}-{'*' * (len(m.group(2)) - 2)}{m.group(2)[-2:]}", text)
    text = _PHONE.sub(lambda m: f"{m.group(1)} ** *** {m.group(4)[-2:].rjust(len(m.group(4)), '*')}", text)
    return _EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)


class PIIMaskingFilter(logging.Filter):
    """Attach to any logger / handler: every record is masked before it is written."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask_pii(record.getMessage())
        record.args = ()
        return True


_INJECTION = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|messages|rules)"
    r"|disregard (?:all |the )?(?:previous|above|system)|system (?:notice|override|prompt)\s*:"
    r"|you are now (?:a|an|in)\b|new instructions\s*:|reveal (?:your|the) system prompt"
    r"|忽略(?:之前|以上|先前)的(?:指令|指示|说明)|系统通知[:：]|你现在是"
    r"|មិនអើពើនឹងការណែនាំ",
    re.I,
)


def flag_injection(text: str) -> bool:
    """True if text from a tool / document tries to instruct the model."""
    return _INJECTION.search(text) is not None


@dataclass
class GateDecision:
    action: str  # execute | confirm | reject
    result: str  # JSON text for the tool message
    summary: dict = field(default_factory=dict)


class ToolGate:
    """Allow-list, validation and user confirmation for tool calls.

    ``decide`` never executes a side-effect tool unless ``user_confirmed`` is
    True - and only the application (a button, a typed "yes" handled by the
    app, a PIN prompt) may set it, never the model.
    """

    def __init__(self, registry, allowed: set[str] | None = None) -> None:
        self.registry = registry
        self.allowed = set(allowed) if allowed is not None else set(registry.tools)

    def decide(self, call: dict, user_confirmed: bool = False) -> GateDecision:
        name, arguments = call.get("name"), dict(call.get("arguments") or {})
        if name not in self.allowed or name not in self.registry:
            return GateDecision("reject", json.dumps({"error": f"tool {name!r} is not available"}))
        tool = self.registry[name]
        if problems := tool.validate(
            {k: v for k, v in arguments.items() if k != "confirmed" or "confirmed" in tool.properties}
        ):
            return GateDecision("reject", json.dumps({"error": "; ".join(problems)}))
        if tool.side_effect and not user_confirmed:
            summary = {k: v for k, v in arguments.items() if k != "confirmed"}
            payload = {
                "status": "confirmation_required",
                "action": name,
                **summary,
                "note": "waiting for the customer's confirmation in the app",
            }
            return GateDecision("confirm", json.dumps(payload, ensure_ascii=False), summary)
        if tool.side_effect and "confirmed" in tool.properties:
            arguments["confirmed"] = True
        return GateDecision("execute", self.registry.call(name, arguments))
