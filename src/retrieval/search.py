"""
Step 4 of the pipeline: THE SEARCH ENGINE (hybrid retrieval + reranking).

Given a question, we:
  1. VECTOR search  -- find chunks with similar MEANING (Qdrant + BGE embeddings)
  2. BM25 search    -- find chunks with matching KEYWORDS
  3. FUSE both lists with Reciprocal Rank Fusion (RRF) into one ranking
  4. RERANK the top candidates with a cross-encoder for precision
  5. return the best `top_k` passages (each with its source info for citations)

Quick test:   python -m src.retrieval.search "How does LoRA reduce memory?"
"""
import pickle
import sys

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.config import settings, QDRANT_DIR, BM25_PATH


class Retriever:
    """Loads the models + indexes once, then answers many search queries."""

    def __init__(self) -> None:
        print("Loading retriever (models + indexes)...")
        # Embedding model: turns the QUERY into a vector (same model used for chunks).
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        # Cross-encoder reranker: scores (query, passage) pairs for precision.
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)
        # Local Qdrant vector DB.
        self.client = QdrantClient(path=str(QDRANT_DIR))
        # BM25 keyword index + the chunk list it was built from.
        data = pickle.loads(BM25_PATH.read_bytes())
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        print("Retriever ready.\n")

    # ----- 1. semantic / meaning-based search -----
    def vector_search(self, query: str, k: int) -> list[dict]:
        qvec = self.embedder.encode([query], normalize_embeddings=True)[0]
        hits = self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=qvec.tolist(),
            limit=k,
        )
        return [h.payload for h in hits]   # payload = chunk_id, text, arxiv_id, title, url

    # ----- 2. keyword search -----
    def bm25_search(self, query: str, k: int) -> list[dict]:
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_idx]

    # ----- 3. fuse the two ranked lists -----
    @staticmethod
    def reciprocal_rank_fusion(lists: list[list[dict]], k: int = 60) -> list[dict]:
        """
        RRF: each chunk earns 1/(k + rank) from every list it appears in, then we
        sort by total score. Chunks both methods rank highly bubble to the top.
        """
        scores: dict[str, float] = {}
        store: dict[str, dict] = {}
        for ranked in lists:
            for rank, item in enumerate(ranked):
                cid = item["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                store[cid] = item
        ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
        return [store[c] for c in ordered]

    # ----- 4. rerank with the cross-encoder -----
    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]

    # ----- the full pipeline -----
    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or settings.TOP_K_RERANK
        vec = self.vector_search(query, settings.TOP_K_VECTOR)
        kw = self.bm25_search(query, settings.TOP_K_BM25)
        fused = self.reciprocal_rank_fusion([vec, kw])
        # Only rerank a manageable shortlist (the fused top ~30).
        shortlist = fused[:30]
        return self.rerank(query, shortlist, top_k)


def _demo(query: str) -> None:
    retriever = Retriever()
    results = retriever.search(query)
    print(f'QUESTION: "{query}"\n')
    print(f"Top {len(results)} passages after hybrid search + reranking:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['title'][:70]}  (arXiv:{r['arxiv_id']})")
        print(f"    score={r['rerank_score']:.3f}")
        print(f"    {r['text'][:180].strip()}...\n")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "How does retrieval-augmented generation reduce hallucination?"
    _demo(q)
