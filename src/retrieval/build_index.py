"""
Step 3 of the pipeline: BUILD THE SEARCHABLE INDEX (local, fast, no GPU).

The heavy part -- turning 14k chunks into vectors -- is done on Colab GPU and
saved to data/embeddings.npy.  THIS script just assembles those pre-computed
vectors into two indexes:

  1. A VECTOR index (Qdrant) for meaning-based / semantic search.
  2. A BM25 index (rank_bm25) for keyword search.

Because we only write files here (no AI model runs), it's fast and light --
it will not hang your laptop.

Workflow:
  1. Run notebooks/embed_on_colab.ipynb on Colab -> download embeddings.npy
  2. Put embeddings.npy into the data/ folder
  3. Run:  python -m src.retrieval.build_index
"""
import json
import pickle

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.config import settings, CHUNKS_PATH, BM25_PATH, QDRANT_DIR, DATA_DIR

EMB_PATH = DATA_DIR / "embeddings.npy"


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def build_vector_index(chunks: list[dict], vectors: np.ndarray) -> None:
    """Store pre-computed vectors + metadata in a local Qdrant database."""
    client = QdrantClient(path=str(QDRANT_DIR))

    # (Re)create the collection -- a "table" that holds our vectors.
    client.recreate_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE),
    )

    # Each "point" = one chunk's vector + its metadata (payload) for citations.
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "chunk_id": chunks[i]["chunk_id"],
                "text": chunks[i]["text"],
                "arxiv_id": chunks[i]["arxiv_id"],
                "title": chunks[i]["title"],
                "url": chunks[i]["url"],
            },
        )
        for i in range(len(chunks))
    ]

    # Upload in batches so we don't hold everything in memory at once.
    BATCH = 256
    for start in tqdm(range(0, len(points), BATCH), desc="Writing to Qdrant"):
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points[start:start + BATCH],
        )
    print(f"Vector index ready: {len(points):,} vectors in Qdrant.")


def build_bm25_index(chunks: list[dict]) -> None:
    """Build a classic keyword index and save it to disk."""
    print("Building BM25 keyword index...")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"BM25 index ready -> {BM25_PATH}")


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks):,} chunks.")

    if not EMB_PATH.exists():
        print("\n  embeddings.npy NOT FOUND in data/.")
        print("  -> Run notebooks/embed_on_colab.ipynb on Colab (GPU), download")
        print("     embeddings.npy, and place it in the data/ folder. Then re-run.")
        return

    vectors = np.load(EMB_PATH)
    print(f"Loaded embeddings: shape {vectors.shape}")
    if len(vectors) != len(chunks):
        raise ValueError(
            f"Mismatch: {len(vectors)} vectors but {len(chunks)} chunks. "
            "Make sure embeddings.npy was built from THIS chunks.json."
        )

    build_vector_index(chunks, vectors)
    print()
    build_bm25_index(chunks)
    print("\nAll indexes built. Step 3 complete.")


if __name__ == "__main__":
    main()
