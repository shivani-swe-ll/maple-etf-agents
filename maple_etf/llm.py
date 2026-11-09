"""Tiny LLM client with no SDK dependency (uses only the Python standard library).

Supported:
* LLM_PROVIDER=openai     -> any OpenAI-compatible API (OpenAI, Groq, Together, Ollama, LM Studio...)
                             set OPENAI_BASE_URL to switch, e.g. http://localhost:11434/v1 for Ollama
* LLM_PROVIDER=anthropic  -> Anthropic Messages API

If LLM_PROVIDER or LLM_MODEL is not set, the app runs in rules-only mode.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


@dataclass
class LLMClient:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    timeout: int = 60
    usage: dict[str, int] = field(default_factory=lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})

    @classmethod
    def from_env(cls) -> "LLMClient | None":
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        model = os.getenv("LLM_MODEL", "").strip()
        if not provider or not model:
            return None
        if provider == "anthropic":
            return cls(provider, model, os.getenv("ANTHROPIC_API_KEY", ""), "https://api.anthropic.com/v1")
        if provider == "openai":
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            return cls(provider, model, os.getenv("OPENAI_API_KEY", ""), base)
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'openai' or 'anthropic'.")

    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMResponse:
        start = time.perf_counter()
        if self.provider == "anthropic":
            url = f"{self.base_url}/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "temperature": 0,
            }
        else:
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            body = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }

        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"LLM request failed ({exc.code}): {detail}") from exc

        if self.provider == "anthropic":
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            u = data.get("usage", {})
            tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
        else:
            text = data["choices"][0]["message"]["content"] or ""
            u = data.get("usage", {}) or {}
            tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

        self.usage["calls"] += 1
        self.usage["input_tokens"] += tin
        self.usage["output_tokens"] += tout
        return LLMResponse(text.strip(), tin, tout, int((time.perf_counter() - start) * 1000))


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model reply")
    return json.loads(match.group(0))
