"""
Step 1 of the pipeline: BUILD A TIGHTLY-FOCUSED PAPER CORPUS.

What this does, in order:
  1. Looks at every PDF already on disk and fetches its metadata (incl. ABSTRACT)
     from arXiv in fast batched calls.
  2. Discovers NEW focused papers via a narrowed arXiv search (LLM / RAG / agents).
  3. Judges every candidate by a relevance filter on TITLE + ABSTRACT.
  4. Keeps the relevant ones (downloading any missing PDFs) and DELETES the
     off-topic "noise" PDFs (music, MRI, 3D, etc.).
  5. Saves a clean catalog to data/papers_meta.json.

Run it with:   python -m src.ingestion.fetch
"""
import json
import os
import arxiv
from tqdm import tqdm

from src.config import settings, PAPERS_DIR, DATA_DIR

META_PATH = DATA_DIR / "papers_meta.json"

# One shared client; polite delay so arXiv doesn't rate-limit us.
client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)


def is_relevant(title: str, abstract: str) -> bool:
    """True if the paper is about LLMs / RAG / agents, judged on title + abstract."""
    text = f"{title} {abstract}".lower().replace("-", " ")
    return any(kw in text for kw in settings.RELEVANCE_KEYWORDS)


def result_to_meta(result) -> dict:
    """Turn an arXiv API result object into our small metadata dict."""
    arxiv_id = result.get_short_id().split("v")[0]
    return {
        "arxiv_id": arxiv_id,
        "title": result.title.strip().replace("\n", " "),
        "abstract": result.summary.strip().replace("\n", " "),
        "authors": [a.name for a in result.authors],
        "published": result.published.strftime("%Y-%m-%d"),
        "url": result.entry_id,
        "pdf_path": str(PAPERS_DIR / f"{arxiv_id}.pdf"),
        "categories": result.categories,
        "_result": result,   # kept temporarily so we can download the PDF later
    }


def fetch_metadata_for_ids(ids: list[str]) -> dict[str, dict]:
    """Fetch metadata (incl. abstract) for specific arXiv ids, in batches of 100."""
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        search = arxiv.Search(id_list=batch, max_results=len(batch))
        for result in client.results(search):
            meta = result_to_meta(result)
            out[meta["arxiv_id"]] = meta
    return out


def discover_focused() -> dict[str, dict]:
    """Run the narrowed search to find recent focused papers (with abstracts)."""
    category_filter = " OR ".join(f"cat:{c}" for c in settings.ARXIV_CATEGORIES)
    full_query = f"({settings.ARXIV_QUERY}) AND ({category_filter})"
    search = arxiv.Search(
        query=full_query,
        max_results=settings.MAX_PAPERS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    out: dict[str, dict] = {}
    for result in client.results(search):
        meta = result_to_meta(result)
        out[meta["arxiv_id"]] = meta
    return out


def build_corpus() -> list[dict]:
    # --- 1. Gather candidates from disk (existing PDFs) + new discovery ---
    disk_ids = [f[:-4] for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]
    print(f"Found {len(disk_ids)} PDFs already on disk. Fetching their abstracts...")
    candidates = fetch_metadata_for_ids(disk_ids)

    print("Discovering new focused papers from arXiv...")
    discovered = discover_focused()
    for arxiv_id, meta in discovered.items():
        candidates.setdefault(arxiv_id, meta)   # don't overwrite disk ones
    print(f"Total unique candidates: {len(candidates)}")

    # --- 2. Split into relevant vs noise ---
    relevant, noise = [], []
    for meta in candidates.values():
        (relevant if is_relevant(meta["title"], meta["abstract"]) else noise).append(meta)
    print(f"Relevant: {len(relevant)}   |   Off-topic noise: {len(noise)}")

    # --- 3. Delete noise PDFs that exist on disk ---
    deleted = 0
    for meta in noise:
        pdf = PAPERS_DIR / f"{meta['arxiv_id']}.pdf"
        if pdf.exists():
            pdf.unlink()
            deleted += 1
    print(f"Deleted {deleted} off-topic PDFs from disk.")

    # --- 4. Keep up to TARGET_PAPERS relevant ones; download any missing PDFs ---
    relevant = relevant[:settings.TARGET_PAPERS]
    final: list[dict] = []
    for meta in tqdm(relevant, desc="Ensuring PDFs"):
        pdf = PAPERS_DIR / f"{meta['arxiv_id']}.pdf"
        if not pdf.exists():
            try:
                meta["_result"].download_pdf(dirpath=str(PAPERS_DIR),
                                             filename=f"{meta['arxiv_id']}.pdf")
            except Exception as e:
                print(f"  ! Skipping {meta['arxiv_id']} (download failed: {e})")
                continue
        meta.pop("_result", None)   # drop the un-serializable arXiv object
        final.append(meta)

    # --- 5. Save clean catalog ---
    META_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\nClean focused corpus: {len(final)} papers -> {META_PATH}")
    return final


if __name__ == "__main__":
    build_corpus()
