"""Backend interface and shared chat-template fallback."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


def fold_system_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deterministically prepend system text to the first user turn.

    Used only if a tokenizer rejects the native system role.
    """
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    normal = [dict(m) for m in messages if m["role"] != "system"]
    if system:
        if normal and normal[0]["role"] == "user":
            normal[0]["content"] = f"{system}\n\n{normal[0]['content']}"
        else:
            normal.insert(0, {"role": "user", "content": system})
    return normal


class Backend(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, top_p: float) -> GenerationResult: ...

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    def metrics(self) -> dict[str, Any]:
        return {}

