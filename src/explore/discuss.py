"""
A GENERAL follow-up discussion helper, reusable by any feature.

Given some CONTEXT (a comparison table, a quiz, an analysis, ...) and the SOURCE papers
it came from, answer the user's follow-up question grounded in that context. One function
serves Compare, Quiz, and any future feature -- reuse over repetition.
"""
from src.generation.provider import get_provider

DISCUSS_SYSTEM = """You are a helpful research assistant continuing a conversation about some
material you produced. You are given the CONTEXT (e.g. a comparison table, quiz, or analysis) and
the SOURCE papers it is based on. Answer the user's follow-up clearly and specifically, grounded in
the sources (cite [n] when you use them). If something genuinely isn't in the sources, use general
knowledge but never fabricate citations. Be concise and helpful."""


def discuss(context: str, sources: list[dict], history: list[dict],
            question: str, provider=None) -> str:
    """Answer a follow-up grounded in `context` + `sources`, using recent `history`."""
    provider = provider or get_provider()
    # A source may carry its text under "context" (papers we compared) or "text" (chunks).
    src_block = "\n\n".join(
        f'[{i}] (from "{s.get("title", "")}")\n{(s.get("context") or s.get("text") or "")[:600]}'
        for i, s in enumerate(sources, 1)
    )
    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
        for m in history[-4:]
    )
    user = (f"CONTEXT:\n{context[:2500]}\n\nSOURCES:\n{src_block}\n\n"
            + (f"CONVERSATION:\n{convo}\n\n" if convo else "")
            + f"USER FOLLOW-UP: {question}\n\nAnswer:")
    return provider.generate(DISCUSS_SYSTEM, user, max_tokens=800)
