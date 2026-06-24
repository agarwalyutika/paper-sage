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

SYSTEM_PROMPT = """You are a precise research assistant that answers questions about \
machine-learning papers, grounded strictly in the provided sources.

Follow these rules:
1. Answer ONLY using the numbered SOURCES. Do not use outside knowledge.
2. START with a direct, clear answer to the question in one or two sentences, then add
   supporting detail. Write a coherent, well-organized answer — synthesize across the
   sources, do not just list disconnected facts.
3. Cite every factual claim with the source number(s) in square brackets, like [1] or [2][3].
4. If the SOURCES only mention the topic in passing and don't really answer the question,
   say so briefly, then summarize what they do say. If they contain nothing relevant at all,
   reply exactly: "I don't have enough information in the provided papers to answer that."
5. Be technical and accurate. Never invent citations or facts."""


def build_user_prompt(question: str, passages: list[dict]) -> str:
    """Format the numbered sources + the question into the user message."""
    blocks = []
    for i, p in enumerate(passages, 1):
        # Each source shows its number, paper title, arXiv id, and the text snippet.
        blocks.append(
            f"[{i}] (from \"{p['title']}\", arXiv:{p['arxiv_id']})\n{p['text']}"
        )
    sources = "\n\n".join(blocks)
    return f"SOURCES:\n{sources}\n\nQUESTION: {question}\n\nGrounded, cited answer:"


def generate_answer(question: str, passages: list[dict],
                    provider: LLMProvider | None = None) -> dict:
    """Produce an answer + the list of sources it was allowed to use."""
    provider = provider or get_provider()
    user_prompt = build_user_prompt(question, passages)
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

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "validation": validation,
    }


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
