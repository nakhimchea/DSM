"""``dsm chat`` - an offline terminal chat with DSM (local checkpoint or llama-server).

    dsm chat --model work/runs/dpo/final           # PyTorch, streaming
    dsm chat --endpoint http://127.0.0.1:8080/v1   # a served GGUF

Commands: ``/load FILE`` (xlsx / csv, serialised with the table contract and
attached to your next message), ``/system TEXT``, ``/tools on|off`` (bank
simulator demo with the confirmation gate), ``/reset``, ``/save FILE``,
``/help``, ``/quit``.

User text is normalised (NFC, Khmer cluster order) before it reaches the model,
exactly as during training; logs are PII-masked.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dsm.text import normalize_text

HELP = """commands:
  /load FILE     attach an .xlsx / .csv file (serialised as text) to your next message
  /system TEXT   set a system prompt (empty = DSM default)
  /tools on|off  enable the bank-simulator tools (side-effect tools ask you to confirm)
  /reset         clear the conversation
  /save FILE     write the conversation as JSON
  /quit          exit"""

MAX_TOOL_ROUNDS = 4


@dataclass
class ChatSession:
    """Conversation state and the tool loop, independent of how text is generated.

    ``complete(messages, tools) -> str`` returns the raw assistant text
    (tool calls as XML), ``confirm(summary) -> bool`` asks the user.
    """

    complete: Callable[[list[dict], list[dict] | None], str]
    confirm: Callable[[dict], bool] = lambda summary: False
    system: str | None = None
    messages: list[dict] = field(default_factory=list)
    attachment: str | None = None
    gate: object | None = None
    logger: logging.Logger | None = None

    def tools(self) -> list[dict] | None:
        return self.gate.registry.schemas(sorted(self.gate.allowed)) if self.gate else None

    def full(self) -> list[dict]:
        return ([{"role": "system", "content": self.system}] if self.system else []) + self.messages

    def send(self, text: str) -> str:
        from dsm.tools.toolcall import parse_tool_calls

        content = normalize_text(text)
        if self.attachment:
            content = f"{self.attachment}\n\n{content}"
            self.attachment = None
        self.messages.append({"role": "user", "content": content})
        self._log("user", content)
        for _ in range(MAX_TOOL_ROUNDS + 1):
            raw = self.complete(self.full(), self.tools())
            answer, calls = parse_tool_calls(raw, self.tools())
            message = {"role": "assistant", "content": answer}
            if calls and self.gate:
                message["tool_calls"] = [c.as_message_call() for c in calls]
            self.messages.append(message)
            self._log("assistant", raw)
            if not (calls and self.gate):
                return answer
            for call in message["tool_calls"]:
                decision = self.gate.decide(call)
                if decision.action == "confirm" and self.confirm(decision.summary | {"action": call["name"]}):
                    decision = self.gate.decide(call, user_confirmed=True)
                self.messages.append({"role": "tool", "name": call["name"], "content": decision.result})
                self._log("tool", decision.result)
        return answer

    def command(self, line: str) -> str | None:
        """Handle a ``/command``; returns text to show, or None to quit."""
        name, _, arg = line.strip().partition(" ")
        arg = arg.strip()
        if name in ("/quit", "/exit"):
            return None
        if name == "/help":
            return HELP
        if name == "/reset":
            self.messages.clear()
            self.attachment = None
            return "conversation cleared"
        if name == "/system":
            self.system = normalize_text(arg) or None
            return "system prompt set" if self.system else "default system prompt"
        if name == "/load":
            from dsm.tables.serialize import workbook_to_text

            path = Path(arg).expanduser()
            if not path.is_file():
                return f"no such file: {path}"
            self.attachment = workbook_to_text(path)
            return f"attached {path.name} ({len(self.attachment.splitlines())} lines) to your next message"
        if name == "/tools":
            if arg == "on":
                from dsm.safeguards.runtime import ToolGate
                from dsm.tools.bank_sim import BankSim

                self.gate = ToolGate(BankSim(seed=0).registry())
                return "bank simulator tools enabled (demo data)"
            self.gate = None
            return "tools disabled"
        if name == "/save":
            Path(arg).write_text(json.dumps(self.full(), ensure_ascii=False, indent=2), encoding="utf-8")
            return f"saved {arg}"
        return f"unknown command {name}; /help lists the commands"

    def _log(self, role: str, text: str) -> None:
        if self.logger:
            self.logger.info("%s: %s", role, text)


# -- backends -----------------------------------------------------------------------------------------


class LocalBackend:
    """Streaming generation with a local DSM checkpoint."""

    def __init__(
        self, model_dir: str, device: str = "auto", max_new_tokens: int = 1024, temperature: float = 0.7
    ) -> None:
        from dsm.eval.generator import HFGenerator

        self.gen = HFGenerator(model_dir, device=device, batch_size=1)
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        cfg = (
            json.loads((Path(model_dir) / "generation_config.json").read_text())
            if (Path(model_dir) / "generation_config.json").is_file()
            else {}
        )
        self.top_p, self.top_k = cfg.get("top_p", 0.8), cfg.get("top_k", 20)
        self.repetition_penalty = cfg.get("repetition_penalty", 1.05)

    def __call__(
        self, messages: list[dict], tools: list[dict] | None, stream: Callable[[str], None] | None = None
    ) -> str:
        import torch

        from dsm.serve.generate import SamplingConfig, generate
        from dsm.tokenizer.render import encode_prompt

        gen = self.gen
        ids = torch.tensor(
            [encode_prompt(gen.tokenizer, messages, gen.template, tools=tools)], device=gen.device
        )
        sampling = SamplingConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            eos_token_ids=gen.eos_ids,
        )
        produced: list[int] = []
        shown = [0]

        def on_token(token) -> None:
            value = int(token[0])
            if value in gen.eos_ids:
                return
            produced.append(value)
            text = gen.tokenizer.decode(produced, skip_special_tokens=True)
            if stream and not text.endswith("�"):  # wait for complete UTF-8 characters
                stream(text[shown[0] :])
                shown[0] = len(text)

        generate(gen.model, ids, sampling=sampling, on_token=on_token, pad_token_id=gen.pad_id)
        return gen.tokenizer.decode(produced, skip_special_tokens=True)


class EndpointBackend:
    def __init__(self, base_url: str) -> None:
        from dsm.eval.generator import EndpointGenerator

        self.gen = EndpointGenerator(base_url)

    def __call__(self, messages: list[dict], tools: list[dict] | None, stream=None) -> str:
        from dsm.eval.generator import Request
        from dsm.tools.toolcall import render_tool_calls

        reply = self.gen.generate([Request(messages, tools=tools, max_new_tokens=1024)])[0]
        text = reply.content + (
            render_tool_calls(reply.tool_calls, reply.content) if reply.tool_calls else ""
        )
        if stream:
            stream(reply.content)
        return text


def _ask(summary: dict) -> bool:
    shown = ", ".join(f"{k}={v}" for k, v in summary.items())
    return input(f"\n[confirm] {shown}\nType 'yes' to confirm: ").strip().lower() in (
        "yes",
        "y",
        "បាទ",
        "ចាស",
        "是",
        "确认",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsm chat", description="Chat with DSM in the terminal (offline).")
    parser.add_argument("--model", help="DSM model directory")
    parser.add_argument("--endpoint", help="OpenAI-compatible base URL (llama-server)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--log", help="append a PII-masked transcript to this file")
    args = parser.parse_args(argv)
    if not (args.model or args.endpoint):
        parser.error("pass --model DIR or --endpoint URL")
    backend = (
        EndpointBackend(args.endpoint)
        if args.endpoint
        else LocalBackend(args.model, args.device, temperature=args.temperature)
    )
    logger = None
    if args.log:
        from dsm.safeguards.runtime import PIIMaskingFilter

        logger = logging.getLogger("dsm.chat")
        handler = logging.FileHandler(args.log, encoding="utf-8")
        handler.addFilter(PIIMaskingFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def complete(messages, tools):
        return backend(messages, tools, stream=lambda piece: print(piece, end="", flush=True))

    session = ChatSession(complete=complete, confirm=_ask, logger=logger)
    print("DSM chat - /help for commands, /quit to exit")
    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line.strip():
            continue
        if line.startswith("/"):
            reply = session.command(line)
            if reply is None:
                return 0
            print(reply)
            continue
        session.send(line)
        print()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
