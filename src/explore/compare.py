"""
Compare multiple papers side-by-side.

We gather each paper's abstract + a few key chunks, then ask the LLM to synthesize a
Markdown comparison TABLE across aspects like problem, method, dataset, results,
strengths, and limitations. This is a multi-document synthesis task -- noticeably more
advanced than single-paper Q&A.
"""
import json
import re

from src.config import DATA_DIR, CHUNKS_PATH
from src.generation.provider import get_provider

_META = {p["arxiv_id"]: p
         for p in json.loads((DATA_DIR / "papers_meta.json").read_text(encoding="utf-8"))}

COMPARE_SYSTEM = """You compare machine-learning papers. You are given 2-4 papers (title + key
content). Produce ONE concise COMPARISON TABLE in GitHub-flavored Markdown.

- First column is "Aspect"; one column per paper, headed by a SHORT version of its title.
- Rows (include those the content supports): Problem / Goal, Method / Architecture,
  Dataset / Setup, Key Results, Strengths, Limitations.
- Keep every cell short (a few words to one line). If a paper doesn't mention something,
  put "—". Base every cell ONLY on the provided content; do not invent details.

Output ONLY the Markdown table -- no preamble, no notes after it."""


def _paper_context(arxiv_id: str, by_paper: dict, max_chunks: int = 3) -> str:
    meta = _META.get(arxiv_id, {})
    parts = []
    if meta.get("abstract"):
        parts.append("Abstract: " + meta["abstract"])
    for c in by_paper.get(arxiv_id, [])[:max_chunks]:
        parts.append(c["text"][:900])
    return "\n".join(parts)[:3200]


def _clean_table(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def compare_papers(arxiv_ids: list[str], provider=None) -> dict:
    """Return {'table': markdown, 'papers': [...]} comparing the given papers."""
    provider = provider or get_provider()

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    by_paper: dict[str, list] = {}
    for c in chunks:
        by_paper.setdefault(c["arxiv_id"], []).append(c)

    blocks, papers = [], []
    for i, aid in enumerate(arxiv_ids, 1):
        meta = _META.get(aid, {})
        title = meta.get("title", aid)
        blocks.append(f'PAPER {i}: "{title}"\n{_paper_context(aid, by_paper)}')
        papers.append({"n": i, "arxiv_id": aid, "title": title, "url": meta.get("url", "")})

    user = "\n\n".join(blocks) + "\n\nComparison table (Markdown):"
    table = _clean_table(provider.generate(COMPARE_SYSTEM, user))
    return {"table": table, "papers": papers}
