"""
The SWAPPABLE LLM layer.

Every backend (Groq, Claude, Ollama) implements the same tiny interface:
a `generate(system, user)` method that returns text. The rest of the app only
talks to this interface -- so switching backends is a one-line config change,
with zero changes anywhere else. THIS is what makes the project vendor-neutral.

Pick the backend in config (LLM_BACKEND) or the .env file.
"""
from typing import Protocol

from src.config import settings


class LLMProvider(Protocol):
    """The contract every backend must satisfy."""
    def generate(self, system: str, user: str) -> str: ...


class GroqProvider:
    """Free, fast, hosted open-source Llama via Groq."""

    def __init__(self) -> None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and put it in your .env file (see .env.example)."
            )
        import groq
        self._groq = groq
        self.client = groq.Groq(api_key=settings.GROQ_API_KEY)
        # Try the strong model first, then fall back if it's rate-limited.
        self.models = [settings.GROQ_MODEL, settings.GROQ_FALLBACK_MODEL]

    def generate(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err = None
        for model in self.models:
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS,        # cap length (no runaway loops)
                    frequency_penalty=settings.LLM_FREQUENCY_PENALTY,  # discourage repetition
                )
                return resp.choices[0].message.content
            except self._groq.RateLimitError as e:
                last_err = e          # this model is throttled -> try the next one
                continue
        raise last_err               # all models throttled


class ClaudeProvider:
    """Optional paid backend: Anthropic Claude. Used only if LLM_BACKEND='claude'."""

    def __init__(self) -> None:
        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (needed for LLM_BACKEND='claude').")
        import anthropic
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.CLAUDE_MODEL

    def generate(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class OllamaProvider:
    """Optional fully-local backend via Ollama. Used only if LLM_BACKEND='ollama'."""

    def __init__(self) -> None:
        import ollama
        self.client = ollama
        self.model = settings.OLLAMA_MODEL

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": settings.LLM_TEMPERATURE},
        )
        return resp["message"]["content"]


def get_provider() -> LLMProvider:
    """Factory: return the provider chosen in config. This is the ONLY swap point."""
    backend = settings.LLM_BACKEND.lower()
    if backend == "groq":
        return GroqProvider()
    if backend == "claude":
        return ClaudeProvider()
    if backend == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown LLM_BACKEND: {backend!r} (use 'groq', 'claude', or 'ollama').")
