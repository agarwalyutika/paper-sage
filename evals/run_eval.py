"""
Part A, step 2: EVALUATE RETRIEVAL QUALITY (the ablation table).

For every eval question we know the correct source chunk. We run retrieval in three
modes and check WHERE the correct chunk lands in the ranking:

  1. vector-only      (semantic search alone)
  2. hybrid           (vector + BM25 fused with RRF)
  3. hybrid + rerank  (then a cross-encoder reorders the top candidates)

Metrics (higher = better):
  Hit@1/3/10 : fraction of questions where the gold chunk is in the top 1/3/10
  MRR        : mean reciprocal rank (1/rank of the gold chunk)
  nDCG@10    : rank-quality score (rewards putting the gold chunk higher)

This uses NO LLM tokens -- it only exercises retrieval.

Run it with:   python -m evals.run_eval
"""
import json
import math

from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.retrieval.search import Retriever

DATA = PROJECT_ROOT / "evals" / "dataset.jsonl"
RESULTS = PROJECT_ROOT / "evals" / "results.json"


def rank_of(gold: str, ranked_ids: list[str]) -> int | None:
    for i, cid in enumerate(ranked_ids, 1):
        if cid == gold:
            return i
    return None


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    found = [r for r in ranks if r is not None]
    return {
        "Hit@1": sum(r <= 1 for r in found) / n,
        "Hit@3": sum(r <= 3 for r in found) / n,
        "Hit@10": sum(r <= 10 for r in found) / n,
        "MRR": sum(1 / r for r in found) / n,
        "nDCG@10": sum(1 / math.log2(r + 1) for r in found) / n,
    }


def main() -> None:
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    print(f"Evaluating on {len(rows)} questions...\n")
    R = Retriever()

    modes: dict[str, list] = {"vector-only": [], "hybrid": [], "hybrid+rerank": []}
    for row in tqdm(rows, desc="Evaluating"):
        q, gold = row["query"], row["gold_chunk_id"]

        vec = R.vector_search(q, 20)
        modes["vector-only"].append(rank_of(gold, [c["chunk_id"] for c in vec[:10]]))

        bm = R.bm25_search(q, 20)
        fused = R.reciprocal_rank_fusion([vec, bm])
        modes["hybrid"].append(rank_of(gold, [c["chunk_id"] for c in fused[:10]]))

        reranked = R.rerank(q, fused[:30], top_k=10)
        modes["hybrid+rerank"].append(rank_of(gold, [c["chunk_id"] for c in reranked]))

    results = {mode: summarize(ranks) for mode, ranks in modes.items()}

    # Pretty table.
    metrics = ["Hit@1", "Hit@3", "Hit@10", "MRR", "nDCG@10"]
    print("\n" + "=" * 68)
    print(f"{'mode':<18}" + "".join(f"{m:>10}" for m in metrics))
    print("-" * 68)
    for mode in modes:
        print(f"{mode:<18}" + "".join(f"{results[mode][m]:>10.3f}" for m in metrics))
    print("=" * 68)

    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {RESULTS}")


if __name__ == "__main__":
    main()
