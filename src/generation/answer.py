"""
Turn (question + retrieved passages) into a grounded, CITED answer.

The whole trustworthiness of the system lives in the prompt here:
  - answer ONLY from the provided passages (no outside knowledge)
  - cite every claim with [1], [2], ... matching the numbered passages
  - if the passages don't contain the answer, say so instead of bluffing

Quick test:  python -m src.generation.answer "How can multi-agent systems improve RAG privacy?"
"""
import sys

from src.generation.provider import get_provider, LLMProvider
from src.citations.validator import validate_citations

SYSTEM_PROMPT = """You are PaperSage, a precise research assistant for machine-learning papers.
You are given numbered SOURCES (passages from real papers) and a QUESTION. Decide how to answer:

A) GROUNDED — if the SOURCES genuinely answer the question, answer using ONLY them. Start with a
   direct answer, then supporting detail, and cite every claim with [n] (e.g. [1], [2][3]). Do
   not use outside knowledge in this case.

B) GENERAL KNOWLEDGE — if the SOURCES are only loosely related and do NOT actually answer the
   question (for example a basic or general question the papers assume but never explain), then
   IGNORE the sources and answer correctly from your own general knowledge. In this case you MUST
   begin your answer with exactly "ℹ️ General knowledge:" and use NO [n] citations.

Be accurate and concise. Decide A or B by whether the sources truly answer THIS question. Never
invent citations, and never cite a source that doesn't actually support the claim."""


def build_user_prompt(question: str, passages: list[dict]) -> str:
    """Format the numbered sources + the question into the user message."""
    blocks = []
    for i, p in enumerate(passages, 1):
        # Each source shows its number, paper title, arXiv id, and the text snippet.
        blocks.append(
            f"[{i}] (from \"{p['title']}\", {p['arxiv_id']})\n{p['text']}"
        )
    sources = "\n\n".join(blocks)
    return f"SOURCES:\n{sources}\n\nQUESTION: {question}\n\nAnswer:"


def format_history(history: list[dict], max_turns: int = 4, max_chars: int = 400) -> str:
    """Format the last few conversation turns as context for a follow-up answer."""
    recent = history[-max_turns:]
    lines = []
    for m in recent:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content'][:max_chars]}")
    return "CONVERSATION SO FAR:\n" + "\n".join(lines)


def generate_answer(question: str, passages: list[dict],
                    provider: LLMProvider | None = None,
                    history: list[dict] | None = None) -> dict:
    """Produce an answer + the list of sources it was allowed to use.

    If `history` (previous turns) is given, it's added as context so follow-up
    questions are answered coherently within the conversation.
    """
    provider = provider or get_provider()
    user_prompt = build_user_prompt(question, passages)
    if history:
        user_prompt = format_history(history) + "\n\n" + user_prompt
    answer_text = provider.generate(SYSTEM_PROMPT, user_prompt)

    # Pair each passage with its citation number so the UI can render sources.
    sources = [
        {
            "n": i,
            "title": p["title"],
            "arxiv_id": p["arxiv_id"],
            "url": p["url"],
            "text": p["text"],
        }
        for i, p in enumerate(passages, 1)
    ]

    # Verify the citations are real ("show your receipts").
    validation = validate_citations(answer_text, num_sources=len(sources))

    # Decide whether this came out GROUNDED (cited from sources) or GENERAL knowledge.
    head = answer_text.strip()[:40].lower()
    if "general knowledge" in head:
        mode = "general"
    elif validation["has_citations"] or validation["is_refusal"]:
        mode = "search"
    else:
        mode = "general"   # answered with no citations and no general-knowledge marker

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources if mode == "search" else [],
        "validation": validation,
        "mode": mode,
    }


WEB_SYSTEM = """You answer a question using WEB RESULTS (the ML paper corpus didn't cover it).
Answer accurately and concisely, and cite every claim with [n] referring to the numbered web
results. Begin your answer with exactly "🌐 From the web:". If the results don't actually answer
the question, say so briefly. Never invent citations."""


def generate_web_answer(question: str, results: list[dict],
                        provider: LLMProvider | None = None,
                        history: list[dict] | None = None) -> dict:
    """Answer a question from web search results, citing each by [n] -> URL."""
    provider = provider or get_provider()
    blocks = [f"[{i}] ({r['title']} — {r['domain']})\n{r['content'][:500]}"
              for i, r in enumerate(results, 1)]
    user_prompt = ("WEB RESULTS:\n" + "\n\n".join(blocks)
                   + f"\n\nQUESTION: {question}\n\nAnswer:")
    if history:
        user_prompt = format_history(history) + "\n\n" + user_prompt
    answer_text = provider.generate(WEB_SYSTEM, user_prompt)

    sources = [
        {"n": i, "title": r["title"], "arxiv_id": r["domain"],
         "url": r["url"], "text": r["content"][:320], "kind": "web"}
        for i, r in enumerate(results, 1)
    ]
    validation = validate_citations(answer_text, num_sources=len(sources))
    return {"question": question, "answer": answer_text, "sources": sources,
            "validation": validation, "mode": "web"}


def _demo(question: str) -> None:
    # Lazy import so this file doesn't load the heavy retriever unless run directly.
    from src.retrieval.search import Retriever
    retriever = Retriever()
    passages = retriever.search(question)
    result = generate_answer(question, passages)

    print(f"\nQUESTION: {result['question']}\n")
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['title'][:65]}  (arXiv:{s['arxiv_id']})")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "How can multi-agent systems improve privacy in RAG?"
    _demo(q)
