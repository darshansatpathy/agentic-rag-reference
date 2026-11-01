"""Chat model providers behind one interface.

- `ScriptedProvider`: deterministic, offline; for tests and demos.
- `OpenAICompatibleProvider`: any Chat Completions endpoint: OpenAI, Azure OpenAI, vLLM,
  Ollama, LiteLLM. Bedrock and Vertex models are commonly fronted the same way.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


@dataclass(frozen=True)
class ModelReply:
    content: str | None


class ModelProvider(Protocol):
    def complete(self, messages: Sequence[Message], tools: Sequence[dict[str, Any]]) -> ModelReply: ...


class ProviderError(Exception):
    pass


@dataclass
class ScriptedProvider:
    replies: list[str]
    seen: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: Sequence[Message], tools: Sequence[dict[str, Any]]) -> ModelReply:
        self.seen.append([dict(m) for m in messages])
        return ModelReply(self.replies.pop(0) if self.replies else "")


@dataclass
class OpenAICompatibleProvider:
    base_url: str
    model: str
    api_key: str = ""
    azure_api_version: str | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    transport: Any = None

    @classmethod
    def from_env(cls) -> OpenAICompatibleProvider:
        return cls(
            base_url=os.environ.get("ARAG_LLM_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("ARAG_LLM_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("ARAG_LLM_API_KEY", ""),
            azure_api_version=os.environ.get("ARAG_AZURE_API_VERSION") or None,
        )

    def complete(self, messages: Sequence[Message], tools: Sequence[dict[str, Any]]) -> ModelReply:
        import httpx

        headers = {"Content-Type": "application/json"}
        params: dict[str, str] = {}
        if self.azure_api_version:
            headers["api-key"] = self.api_key
            params["api-version"] = self.azure_api_version
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            resp = client.post(
                self.base_url.rstrip("/") + "/chat/completions", headers=headers, params=params, json=body
            )
        if resp.status_code >= 400:
            raise ProviderError(f"model endpoint returned {resp.status_code}: {resp.text[:300]}")
        try:
            return ModelReply(resp.json()["choices"][0]["message"].get("content"))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("response has no choices[0].message") from exc
