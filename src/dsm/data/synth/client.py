"""Client for a LOCAL teacher model behind an OpenAI-compatible API (llama-server, vLLM).

Used for synthetic responses, translation and judging. Guarantees:
* only private / loopback hosts unless ``allow_remote: true`` (keeps the
  pipeline cloud-free and avoids sending data out of the building);
* the teacher's licence must be in the configured allowlist (Apache-2.0 / MIT
  by default) because model licences can propagate to trained models;
* every response is cached on disk (JSONL) keyed by the exact request, so
  re-running a builder is free and reproducible.

Standard library only (``urllib``), so it runs inside air-gapped environments.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from dsm.utils import load_yaml

_PRIVATE_SUFFIXES = (".local", ".internal", ".lan", ".home", ".corp", ".intranet")


def is_private_host(host: str) -> bool:
    if host in ("localhost", "") or host.endswith(_PRIVATE_SUFFIXES) or "." not in host:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


class TeacherError(RuntimeError):
    pass


class LocalTeacher:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        name: str | None = None,
        license: str | None = None,
        allowlist_licenses: list[str] | None = None,
        api_key: str | None = None,
        timeout_s: float = 180,
        max_retries: int = 3,
        cache: str | Path | None = None,
        allow_remote: bool = False,
        generation: dict | None = None,
    ) -> None:
        host = urlparse(base_url).hostname or ""
        if not allow_remote and not is_private_host(host):
            raise TeacherError(
                f"refusing non-local teacher endpoint {base_url!r} (set allow_remote: true to override)"
            )
        allow = [a.lower() for a in (allowlist_licenses or ["apache-2.0", "mit"])]
        if license is None or license.lower() not in allow:
            raise TeacherError(f"teacher licence {license!r} is not in the allowlist {allow}")
        self.base_url = base_url.rstrip("/")
        # Never route requests for a local teacher through an HTTP(S) proxy from the environment.
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({} if is_private_host(host) else None)
        )
        self.model = model
        self.name = name or model
        self.license = license
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.defaults = dict(generation or {})
        self.cache_path = Path(cache) if cache else None
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        if self.cache_path and self.cache_path.is_file():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._cache[entry["key"]] = entry["response"]

    @classmethod
    def from_config(cls, path: str | Path = "configs/teacher.yaml", role: str = "teacher") -> LocalTeacher:
        cfg = load_yaml(path)
        section = dict(cfg[role])
        section.setdefault("allowlist_licenses", cfg.get("allowlist_licenses"))
        section.setdefault("cache", cfg.get("cache"))
        section.setdefault("allow_remote", cfg.get("allow_remote", False))
        return cls(**section)

    @property
    def generated_by(self) -> str:
        return f"model:{self.name}"

    def _key(self, messages: list[dict], params: dict) -> str:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "params": params}, sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        with self._opener.open(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def chat(self, messages: list[dict], **params) -> str | None:
        """One completion (``None`` after ``max_retries`` failures)."""
        merged = {**self.defaults, **params}
        key = self._key(messages, merged)
        if key in self._cache:
            return self._cache[key]
        payload = {"model": self.model, "messages": messages, **merged}
        for attempt in range(self.max_retries):
            try:
                text = self._post(payload)["choices"][0]["message"]["content"]
                break
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(2**attempt)
        with self._lock:
            self._cache[key] = text
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "response": text}, ensure_ascii=False) + "\n")
        return text

    def map(self, conversations: list[list[dict]], workers: int = 8, **params) -> list[str | None]:
        """Run many requests concurrently (order preserved)."""
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda messages: self.chat(messages, **params), conversations))
