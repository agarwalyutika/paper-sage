"""
In-memory search over user-UPLOADED files (PDF / txt / md).

Unlike the 200-paper corpus (which lives in Qdrant on disk), uploaded files are
temporary and small, so we keep their vectors in RAM and search with plain numpy.
This reuses the SAME embedder + reranker as the corpus, so uploaded docs get the
same quality treatment -- just without a database (no disk, no lock).

  uploaded file -> extract text -> chunk -> embed (in memory)
                -> cosine search -> cross-encoder rerank -> top passages
"""
import fitz  # PyMuPDF
import numpy as np

from src.config import settings
from src.ingestion.chunk import clean_text, split_into_chunks


def _make_chunk(title: str, locator: str, text: str, i: int) -> dict:
    # Same shape as corpus passages so the rest of the pipeline just works.
    # For uploads there's no arXiv id/url: we put the page (e.g. "p.3") in arxiv_id.
    return {"chunk_id": f"{title}_{i}", "text": text,
            "title": title, "arxiv_id": locator, "url": ""}


def extract_chunks(filename: str, data: bytes) -> list[dict]:
    """Turn one uploaded file's bytes into chunk dicts (with page numbers for PDFs)."""
    chunks: list[dict] = []
    if filename.lower().endswith((".txt", ".md")):
        text = clean_text(data.decode("utf-8", errors="ignore"))
        for piece in split_into_chunks(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP):
            chunks.append(_make_chunk(filename, "text", piece, len(chunks)))
    else:  # treat as PDF
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, 1):
                text = clean_text(page.get_text())
                for piece in split_into_chunks(text, settings.CHUNK_SIZE,
                                               settings.CHUNK_OVERLAP):
                    chunks.append(_make_chunk(filename, f"p.{page_num}", piece, len(chunks)))
    return chunks


class UploadedIndex:
    """Builds an in-memory index from uploaded files and searches it."""

    def __init__(self, embedder, reranker) -> None:
        # Reuse the corpus's already-loaded models (no extra memory/setup).
        self.embedder = embedder
        self.reranker = reranker
        self.chunks: list[dict] = []
        self.vectors: np.ndarray | None = None

    def build(self, files: list[tuple[str, bytes]]) -> int:
        """Embed all uploaded files. `files` = list of (filename, bytes). Returns #chunks."""
        chunks: list[dict] = []
        for filename, data in files:
            try:
                chunks.extend(extract_chunks(filename, data))
            except Exception as e:
                print(f"  ! Could not read {filename}: {e}")
        self.chunks = chunks
        if chunks:
            vecs = self.embedder.encode(
                [c["text"] for c in chunks],
                batch_size=32, normalize_embeddings=True, show_progress_bar=False,
            )
            self.vectors = np.asarray(vecs, dtype="float32")
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Cosine search + cross-encoder rerank over the uploaded chunks."""
        top_k = top_k or settings.TOP_K_RERANK
        if not self.chunks:
            return []
        qvec = self.embedder.encode([query], normalize_embeddings=True)[0]
        # vectors are normalized, so dot product == cosine similarity.
        sims = self.vectors @ qvec
        n_cand = min(30, len(self.chunks))
        top_idx = np.argsort(sims)[::-1][:n_cand]
        candidates = [dict(self.chunks[i]) for i in top_idx]

        scores = self.reranker.predict([(query, c["text"]) for c in candidates])
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]
