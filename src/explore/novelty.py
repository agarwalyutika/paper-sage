"""
"Find Novelty" -- position a research idea against the existing literature.

Given a user's idea, we retrieve the most related papers and ask the LLM to analyze:
similar work, what's already done, research gaps, and novel contribution ideas. It's
RAG applied to literature gap-analysis (a different prompt on the same retrieval engine).
"""
from src.generation.provider import get_provider

NOVELTY_SYSTEM = """You are a research advisor. The user describes a RESEARCH IDEA, and you are
given the most RELATED PAPERS (passages) from a research corpus. Analyze the idea against them.

Respond in EXACTLY these Markdown sections:

### 🔍 Similar / related work
Which of the related papers are closest to the idea, and how. Cite [n].

### ✅ What's already been done
Approaches, methods, or results these papers already cover that overlap the idea. Cite [n].

### 🕳️ Research gaps
What seems missing, under-explored, or not addressed by the related papers.

### 💡 Novel contribution ideas
3-5 concrete, specific ways the user's idea could be novel (new method, setting, data,
combination, evaluation, etc.).

Be specific and grounded in the papers; cite [n] where relevant. If there is little related work
in the corpus, say so plainly -- that itself can indicate novelty. Do not invent papers."""


def find_novelty(idea: str, passages: list[dict], provider=None) -> dict:
    """Analyze a research idea against retrieved related papers."""
    provider = provider or get_provider()

    # Dedupe to DISTINCT papers (breadth) -- one passage per paper, up to 6.
    seen, distinct = set(), []
    for p in passages:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            distinct.append(p)
        if len(distinct) >= 6:
            break

    blocks = [f'[{i}] (from "{p["title"]}", {p["arxiv_id"]})\n{p["text"]}'
              for i, p in enumerate(distinct, 1)]
    user = (f"RESEARCH IDEA: {idea}\n\nRELATED PAPERS:\n" + "\n\n".join(blocks)
            + "\n\nAnalysis:")
    analysis = provider.generate(NOVELTY_SYSTEM, user, max_tokens=1500)

    sources = [{"n": i, "title": p["title"], "arxiv_id": p["arxiv_id"],
                "url": p["url"], "text": p["text"]}
               for i, p in enumerate(distinct, 1)]
    return {"analysis": analysis, "sources": sources}
