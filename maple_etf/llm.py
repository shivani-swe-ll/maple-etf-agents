"""Tiny LLM client with no SDK dependency (uses only the Python standard library).

Supported:
* LLM_PROVIDER=openai     -> any OpenAI-compatible API (OpenAI, Groq, Together, Ollama, LM Studio...)
                             set OPENAI_BASE_URL to switch, e.g. http://localhost:11434/v1 for Ollama
* LLM_PROVIDER=anthropic  -> Anthropic Messages API

Two ways to call it:
* complete()   -> one prompt in, text out
* chat_step()  -> one step of a TOOL-CALLING conversation (the model may ask to run tools)

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
from typing import Any


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolStep:
    """One model turn. If tool_calls is not empty, run them and send the results back."""

    text: str
    tool_calls: list[ToolCall]
    assistant_message: dict          # provider-native message to append to the history
    input_tokens: int = 0
    output_tokens: int = 0


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

    # ---------------- low level ----------------
    def _post(self, body: dict) -> dict:
        if self.provider == "anthropic":
            url = f"{self.base_url}/messages"
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
        else:
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"LLM request failed ({exc.code}): {detail}") from exc

    def _count(self, data: dict) -> tuple[int, int]:
        u = data.get("usage", {}) or {}
        if self.provider == "anthropic":
            tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
        else:
            tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        self.usage["calls"] += 1
        self.usage["input_tokens"] += tin
        self.usage["output_tokens"] += tout
        return tin, tout

    # ---------------- simple completion ----------------
    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMResponse:
        start = time.perf_counter()
        if self.provider == "anthropic":
            body = {"model": self.model, "max_tokens": max_tokens, "system": system, "temperature": 0,
                    "messages": [{"role": "user", "content": user}]}
        else:
            body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        data = self._post(body)
        if self.provider == "anthropic":
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        else:
            text = data["choices"][0]["message"]["content"] or ""
        tin, tout = self._count(data)
        return LLMResponse(text.strip(), tin, tout, int((time.perf_counter() - start) * 1000))

    # ---------------- tool calling ----------------
    @staticmethod
    def user_message(text: str) -> dict:
        return {"role": "user", "content": text}

    def chat_step(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 800) -> ToolStep:
        """tools: [{"name", "description", "parameters": <JSON schema>}] (same format for both providers)."""
        if self.provider == "anthropic":
            body = {"model": self.model, "max_tokens": max_tokens, "system": system, "temperature": 0,
                    "messages": messages,
                    "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                              for t in tools]}
            data = self._post(body)
            tin, tout = self._count(data)
            blocks = data.get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            calls = [ToolCall(b["id"], b["name"], b.get("input") or {}) for b in blocks if b.get("type") == "tool_use"]
            return ToolStep(text, calls, {"role": "assistant", "content": blocks}, tin, tout)

        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, *messages],
                "tools": [{"type": "function", "function": t} for t in tools]}
        data = self._post(body)
        tin, tout = self._count(data)
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = {"_invalid_json": raw}
            calls.append(ToolCall(tc["id"], tc["function"]["name"], args))
        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            assistant["tool_calls"] = msg["tool_calls"]
        return ToolStep((msg.get("content") or "").strip(), calls, assistant, tin, tout)

    def tool_result_messages(self, results: list[tuple[ToolCall, str]]) -> list[dict]:
        """Messages that send tool outputs back to the model, in the provider's own format."""
        if self.provider == "anthropic":
            return [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call.id, "content": output} for call, output in results
            ]}]
        return [{"role": "tool", "tool_call_id": call.id, "content": output} for call, output in results]


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model reply")
    return json.loads(match.group(0))
