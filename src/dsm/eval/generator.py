"""Answer generators for evaluation: a local DSM checkpoint or an OpenAI-compatible endpoint.

The endpoint generator evaluates exactly what is deployed (e.g. a quantised GGUF
behind ``llama-server --jinja``), so the same suites gate both the trained
checkpoint and the exported model. Decoding is greedy in both cases.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dsm.tools.toolcall import parse_tool_calls


@dataclass
class Request:
    messages: list[dict]
    tools: list[dict] | None = None
    system: str | None = None
    max_new_tokens: int = 512

    def full_messages(self) -> list[dict]:
        return ([{"role": "system", "content": self.system}] if self.system is not None else []) + list(
            self.messages
        )


@dataclass
class Reply:
    text: str  # raw assistant text (tool-call XML included when generated locally)
    content: str  # text without tool calls
    tool_calls: list[dict] = field(default_factory=list)  # [{"name", "arguments"}]

    @property
    def has_think(self) -> bool:
        return "<think>" in self.text or "</think>" in self.text


def reply_from_text(text: str, tools: list[dict] | None) -> Reply:
    content, calls = parse_tool_calls(text, tools)
    return Reply(text=text, content=content, tool_calls=[c.as_message_call() for c in calls])


class HFGenerator:
    """Greedy decoding with a local DSM checkpoint (batched, left-padded)."""

    def __init__(
        self, model_dir: str | Path, device: str = "auto", dtype: str = "auto", batch_size: int = 8
    ) -> None:
        import torch
        from tokenizers import Tokenizer

        import dsm.modeling  # noqa: F401 - registers DSM
        from dsm.modeling import DSMForCausalLM

        self.model_dir = Path(model_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype == "auto":
            dtype = "bfloat16" if self.device.type == "cuda" else "float32"
        self.model = (
            DSMForCausalLM.from_pretrained(self.model_dir, dtype=getattr(torch, dtype)).to(self.device).eval()
        )
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.template = (self.model_dir / "chat_template.jinja").read_text(encoding="utf-8")
        generation = self.model_dir / "generation_config.json"
        eos = json.loads(generation.read_text()).get("eos_token_id") if generation.is_file() else None
        eos = eos if isinstance(eos, list) else [eos if eos is not None else self.model.config.eos_token_id]
        self.eos_ids = [int(e) for e in eos if e is not None]
        self.pad_id = int(
            self.model.config.pad_token_id if self.model.config.pad_token_id is not None else self.eos_ids[0]
        )
        self.batch_size = batch_size
        self.name = f"local:{self.model_dir.name}"

    def _decode(self, ids: list[int]) -> str:
        for position, token in enumerate(ids):
            if token in self.eos_ids:
                ids = ids[:position]
                break
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def generate(self, requests: list[Request]) -> list[Reply]:
        import torch

        from dsm.serve.generate import SamplingConfig, generate
        from dsm.tokenizer.render import encode_prompt

        replies: list[Reply] = []
        for start in range(0, len(requests), self.batch_size):
            chunk = requests[start : start + self.batch_size]
            prompts = [
                encode_prompt(self.tokenizer, r.full_messages(), self.template, tools=r.tools) for r in chunk
            ]
            width = max(len(p) for p in prompts)
            ids = torch.full((len(prompts), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(prompts), width), dtype=torch.long)
            for row, prompt in enumerate(prompts):
                ids[row, width - len(prompt) :] = torch.tensor(prompt)
                mask[row, width - len(prompt) :] = 1
            sampling = SamplingConfig(
                max_new_tokens=max(r.max_new_tokens for r in chunk),
                temperature=0.0,
                eos_token_ids=self.eos_ids,
            )
            result = generate(
                self.model, ids.to(self.device), mask.to(self.device), sampling, pad_token_id=self.pad_id
            )
            for row, request in enumerate(chunk):
                text = self._decode(result.new_tokens[row].tolist()[: request.max_new_tokens])
                replies.append(reply_from_text(text, request.tools))
        return replies


class EndpointGenerator:
    """``POST {base_url}/chat/completions`` on a local server (llama-server, vLLM)."""

    def __init__(
        self, base_url: str, model: str = "dsm", timeout_s: float = 300, allow_remote: bool = False
    ) -> None:
        from dsm.data.synth.client import is_private_host

        host = urlparse(base_url).hostname or ""
        if not allow_remote and not is_private_host(host):
            raise ValueError(f"refusing non-local endpoint {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.name = f"endpoint:{self.base_url}"

    def _one(self, request: Request) -> Reply:
        payload = {
            "model": self.model,
            "messages": request.full_messages(),
            "temperature": 0.0,
            "max_tokens": request.max_new_tokens,
        }
        if request.tools:
            payload["tools"] = request.tools
        http = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener.open(http, timeout=self.timeout_s) as response:
            message = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]
        content = message.get("content") or ""
        calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function", call)
            arguments = fn.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            calls.append({"name": fn.get("name"), "arguments": arguments})
        if calls:
            return Reply(text=content, content=content.strip(), tool_calls=calls)
        return reply_from_text(content, request.tools)  # server without a tool-call parser

    def generate(self, requests: list[Request]) -> list[Reply]:
        return [self._one(r) for r in requests]
