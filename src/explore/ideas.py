"""
Research Idea Generator -- propose graduate-project ideas grounded in recent papers.

Given a TOPIC, we retrieve the most related papers and ask the LLM to invent several
concrete project ideas. Each idea is scored for novelty + difficulty and comes with
suggested datasets, an expected contribution, and a step-by-step implementation
roadmap. It's RAG applied to research ideation (the same retrieval engine, a new prompt
+ a structured, parseable output).
"""
import json

from src.generation.provider import get_provider

IDEAS_SYSTEM = """You are a research advisor helping a master's/PhD student pick a project.
The user gives a TOPIC, and you are given RELATED PAPERS (passages) from a research corpus.
Propose __N__ DISTINCT, concrete project ideas, inspired by and grounded in these papers.

Return ONLY a JSON array (no prose, no markdown code fences). Each element must be:
{
  "title": "short, specific project title",
  "summary": "1-2 sentence description of the idea",
  "novelty": 4,
  "difficulty": "Easy",
  "datasets": ["a concrete dataset or data source", "..."],
  "contributions": "the new knowledge or artifact this project would contribute",
  "roadmap": ["step 1", "step 2", "step 3", "step 4"]
}
Rules:
- "novelty" is an integer 1-5 (5 = highly novel).
- "difficulty" is exactly one of "Easy", "Medium", "Hard".
- Make ideas specific: method + setting + evaluation. Avoid generic ideas.
- Vary novelty and difficulty across the set. Ground ideas in the papers where possible.
- Do not invent fake papers or datasets that don't exist."""


def _parse(raw: str) -> list[dict]:
    """Best-effort JSON parse -- models sometimes wrap or lightly malform the array."""
    candidates = [raw]
    if "[" in raw and "]" in raw:
        candidates.append(raw[raw.find("["): raw.rfind("]") + 1])
    for c in candidates:
        try:
            data = json.loads(c)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except Exception:
            continue
    return []


def generate_ideas(topic: str, passages: list[dict], n: int = 6, provider=None) -> dict:
    """Generate `n` project ideas for `topic`, grounded in retrieved `passages`."""
    provider = provider or get_provider()

    # Dedupe to DISTINCT papers (breadth) -- one passage per paper, up to 8.
    seen, distinct = set(), []
    for p in passages:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            distinct.append(p)
        if len(distinct) >= 8:
            break

    blocks = [f'[{i}] (from "{p["title"]}", {p["arxiv_id"]})\n{p["text"]}'
              for i, p in enumerate(distinct, 1)]
    user = (f"TOPIC: {topic}\n\nRELATED PAPERS:\n" + "\n\n".join(blocks)
            + f"\n\nReturn a JSON array of exactly {n} project ideas.")
    raw = provider.generate(IDEAS_SYSTEM.replace("__N__", str(n)), user, max_tokens=3000)

    sources = [{"n": i, "title": p["title"], "arxiv_id": p["arxiv_id"],
                "url": p["url"], "text": p["text"]}
               for i, p in enumerate(distinct, 1)]
    return {"ideas": _parse(raw), "raw": raw, "sources": sources}
