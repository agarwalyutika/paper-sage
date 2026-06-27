"""
Compare multiple papers side-by-side -- DETAILED, and works for corpus papers AND
your own uploaded PDFs.

Design idea: the core `compare(papers)` doesn't care WHERE a paper came from. It takes a
generic list of {title, context, url}. Helpers build that context either from the corpus
(`build_corpus_papers`) or from an uploaded PDF (`upload_context`). Same synthesis logic
for both -- this is "separate what to compare from where it came from".

It's a multi-document synthesis task: gather each paper's text, then ask the LLM for one
rich Markdown comparison table.
"""
import json
import re

from src.config import DATA_DIR, CHUNKS_PATH
from src.generation.provider import get_provider

_META = {p["arxiv_id"]: p
         for p in json.loads((DATA_DIR / "papers_meta.json").read_text(encoding="utf-8"))}

# How much output room the table gets (detailed tables need far more than the 700 default).
COMPARE_MAX_TOKENS = 2000

COMPARE_SYSTEM = """You compare machine-learning papers IN DETAIL. You are given 2-4 papers
(title + content). Produce ONE rich COMPARISON TABLE in GitHub-flavored Markdown.

- First column is "Aspect"; one column per paper, headed by a SHORT version of its title.
- Include these rows: Problem / Goal, Method / Architecture, Key Idea / Novelty,
  Dataset / Setup, Key Results, Strengths, Limitations.
- Each cell must be 1-3 FULL sentences with SPECIFICS pulled from the content -- name the
  actual methods, datasets, metrics, and numbers. Do NOT write one-word cells.
- Use "—" only if the content genuinely doesn't cover that aspect.
- Base every cell ONLY on the provided content; never invent details.

Output ONLY the Markdown table -- no preamble and nothing after it."""


def _clean_table(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _load_by_paper() -> dict[str, list]:
    """Group all chunks by their paper id (so we can pull a paper's text quickly)."""
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    by: dict[str, list] = {}
    for c in chunks:
        by.setdefault(c["arxiv_id"], []).append(c)
    return by


def corpus_context(arxiv_id: str, by_paper: dict, max_chunks: int = 6) -> str:
    """Build a rich context for a corpus paper: its abstract + several leading chunks."""
    meta = _META.get(arxiv_id, {})
    parts = []
    if meta.get("abstract"):
        parts.append("Abstract: " + meta["abstract"])
    for c in by_paper.get(arxiv_id, [])[:max_chunks]:
        parts.append(c["text"][:1000])
    return "\n".join(parts)[:5000]


def build_corpus_papers(arxiv_ids: list[str]) -> list[dict]:
    """Turn corpus paper ids into {title, context, url} entries for comparison."""
    by_paper = _load_by_paper()
    out = []
    for aid in arxiv_ids:
        meta = _META.get(aid, {})
        out.append({"title": meta.get("title", aid),
                    "context": corpus_context(aid, by_paper),
                    "url": meta.get("url", "")})
    return out


def upload_context(filename: str, data: bytes, max_chars: int = 5000) -> str:
    """Extract text from an uploaded PDF (reuses the corpus PDF reader)."""
    from src.retrieval.uploaded_docs import extract_chunks
    chunks = extract_chunks(filename, data)
    return "\n".join(c["text"] for c in chunks[:6])[:max_chars]


def compare(papers: list[dict], provider=None) -> dict:
    """Compare a generic list of {title, context, url}. Returns {'table', 'papers'}."""
    provider = provider or get_provider()
    blocks = [f'PAPER {i}: "{p["title"]}"\n{p["context"]}'
              for i, p in enumerate(papers, 1)]
    user = "\n\n".join(blocks) + "\n\nDetailed comparison table (Markdown):"
    table = _clean_table(provider.generate(COMPARE_SYSTEM, user, max_tokens=COMPARE_MAX_TOKENS))
    return {"table": table, "papers": papers}
