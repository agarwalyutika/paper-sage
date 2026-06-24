"""
Step 2 of the pipeline: READ PDFs and CUT THEM INTO CHUNKS.

For every paper in the catalog we:
  1. extract the raw text (using PyMuPDF, imported as "fitz"),
  2. clean it up a little,
  3. slice it into overlapping ~1200-character chunks,
  4. attach metadata (which paper / arXiv id / title / url) to each chunk.

The result is saved to data/chunks.json -- a flat list of chunks that the
search engine will index in the next step.

Run it with:   python -m src.ingestion.chunk
"""
import json
import re
import fitz  # this is PyMuPDF
from tqdm import tqdm

from src.config import settings, CHUNKS_PATH, DATA_DIR

META_PATH = DATA_DIR / "papers_meta.json"


def extract_text(pdf_path: str) -> str:
    """Open a PDF and return all its text as one big string."""
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def clean_text(text: str) -> str:
    """Light cleanup: collapse weird whitespace, drop empty lines."""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)        # squash runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)     # squash big gaps of blank lines
    return text.strip()


def split_into_chunks(text: str, size: int, overlap: int) -> list[str]:
    """
    Slide a window of `size` characters across the text, moving forward by
    (size - overlap) each step so consecutive chunks share `overlap` characters.
    """
    chunks = []
    start = 0
    step = size - overlap
    while start < len(text):
        chunk = text[start:start + size].strip()
        if len(chunk) > 100:        # ignore tiny leftover fragments
            chunks.append(chunk)
        start += step
    return chunks


def chunk_all_papers() -> list[dict]:
    """Turn every paper in the catalog into a list of chunk dicts."""
    papers = json.loads(META_PATH.read_text(encoding="utf-8"))
    all_chunks: list[dict] = []

    for paper in tqdm(papers, desc="Chunking papers"):
        try:
            raw = extract_text(paper["pdf_path"])
        except Exception as e:
            print(f"  ! Could not read {paper['arxiv_id']}: {e}")
            continue

        cleaned = clean_text(raw)
        pieces = split_into_chunks(cleaned, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

        for i, piece in enumerate(pieces):
            all_chunks.append({
                # a unique id like "2401.01234_3" (paper + chunk number)
                "chunk_id": f"{paper['arxiv_id']}_{i}",
                "text": piece,
                # --- source metadata: this is what powers citations later ---
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "url": paper["url"],
            })

    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"\nCreated {len(all_chunks)} chunks from {len(papers)} papers "
          f"-> {CHUNKS_PATH}")
    return all_chunks


if __name__ == "__main__":
    chunk_all_papers()
