"""
Multi-turn conversation logic with an agentic router.

For every message the assistant first DECIDES what kind of message it is:
  - "chat"   -> greetings, thanks, "who are you", small talk
               => reply naturally, no retrieval, no citations
  - "search" -> a real research question about the ML papers
               => condense the question, retrieve, and answer with citations

This makes it feel like a normal conversation while keeping answers grounded
whenever the user actually asks about the papers.
"""
from typing import Callable

from src.config import settings
from src.generation.provider import get_provider, LLMProvider
from src.generation.answer import generate_answer, generate_web_answer, generate_code
from src.web.search import web_search

# A code/implementation request -> generate code grounded in the papers.
CODE_HINTS = ("implement", "in pytorch", "pytorch code", "in python", "to python",
              "python code", "write code", "code for", "as code", "code this",
              "in numpy", "in tensorflow")


def is_code_request(message: str) -> bool:
    m = message.lower()
    return any(h in m for h in CODE_HINTS)

# --- prompts -----------------------------------------------------------------
ROUTER_SYSTEM = """You are a router. Read the user's latest message (with brief \
context) and classify it. Reply with EXACTLY one word:
- SEARCH : it's a research/technical question that needs looking things up in ML
  research papers (methods, results, concepts, comparisons, definitions, "tell me
  more about that", etc.)
- CHAT : it's a greeting, thanks, small talk, or a question about you/the assistant
  itself, or anything that does NOT need the papers.
Reply with only SEARCH or CHAT."""

CHAT_SYSTEM = """You are PaperSage, a friendly assistant that answers questions \
grounded in a corpus of 200 machine-learning research papers (LLMs, RAG, agents).
Respond naturally, warmly, and briefly — like a normal conversation.
- For greetings or small talk, reply in a sentence or two.
- If asked what you can do or who you are, explain that you answer questions about
  ML research papers with citations from real papers, and invite them to ask one.
Keep it concise. Do not make up research facts here."""

CONDENSE_SYSTEM = """You rewrite a user's follow-up question into a standalone \
question that makes sense on its own, using the conversation for context. Resolve \
pronouns (it, they, this) into what they refer to. Output ONLY the rewritten \
question, nothing else. If it's already standalone, return it unchanged."""


def _recent(history: list[dict], n: int = 4, cap: int = 300) -> str:
    """Format the last n turns as a compact transcript."""
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:cap]}"
        for m in history[-n:]
    )


# --- pieces ------------------------------------------------------------------
def route(history: list[dict], message: str, provider: LLMProvider) -> str:
    """Decide whether this message needs the papers ('search') or not ('chat')."""
    context = _recent(history, n=2)
    user = (f"Conversation:\n{context}\n\n" if context else "") + \
           f"Latest message: {message}\n\nClassification:"
    out = provider.generate(ROUTER_SYSTEM, user).strip().upper()
    return "search" if out.startswith("SEARCH") else "chat"


def chat_reply(history: list[dict], message: str, provider: LLMProvider) -> str:
    """A natural, non-RAG conversational reply (greetings, meta questions, etc.)."""
    context = _recent(history, n=4)
    user = (f"{context}\n" if context else "") + f"User: {message}\nPaperSage:"
    return provider.generate(CHAT_SYSTEM, user)


def condense_question(history: list[dict], message: str, provider: LLMProvider) -> str:
    """Turn a context-dependent follow-up into a self-contained search query."""
    if not history:
        return message
    user = f"Conversation:\n{_recent(history)}\n\nFollow-up question: {message}\n\nStandalone question:"
    rewritten = provider.generate(CONDENSE_SYSTEM, user).strip()
    return rewritten or message


# --- the full step -----------------------------------------------------------
def answer_in_conversation(history: list[dict], message: str,
                           search_fn: Callable[[str], list[dict]],
                           provider: LLMProvider | None = None) -> dict:
    """
    One conversational turn:
      1. route: chat vs search
      2a. chat  -> friendly reply, no sources
      2b. search-> condense -> retrieve -> grounded, cited answer
    """
    provider = provider or get_provider()

    if route(history, message, provider) == "chat":
        reply = chat_reply(history, message, provider)
        return {"question": message, "answer": reply, "sources": [], "mode": "chat"}

    # research question -> retrieve relevant passages
    standalone = condense_question(history, message, provider)
    passages = search_fn(standalone)

    # Code/implementation request -> generate code grounded in those passages.
    if is_code_request(message):
        result = generate_code(message, passages, provider=provider, history=history)
        result["standalone_query"] = standalone
        return result

    # Otherwise: let generate_answer decide grounded vs general.
    result = generate_answer(message, passages, provider=provider, history=history)
    result["standalone_query"] = standalone   # mode is set by generate_answer

    # If the papers didn't cover it, try the WEB before settling for general knowledge.
    if result["mode"] == "general" and settings.WEB_SEARCH_ENABLED:
        web_results = web_search(standalone)
        if web_results:
            web = generate_web_answer(message, web_results, provider=provider, history=history)
            web["standalone_query"] = standalone
            return web

    return result
