"""
Phase 2, step 3: COMPARE the base reranker vs the fine-tuned reranker.

Runs the same eval questions through the full hybrid pipeline, reranking the SAME
candidates with (a) the original BGE reranker and (b) your fine-tuned one, then
prints the before/after metrics. This is the "did fine-tuning help?" headline.

Prereq: unzip the Colab output into  models/bge-reranker-base-ft/

Run it with:   python -m evals.compare_reranker
"""
import json
import math

import numpy as np
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.retrieval.search import Retriever

DATA = PROJECT_ROOT / "evals" / "dataset.jsonl"
FT_PATH = PROJECT_ROOT / "models" / "bge-reranker-base-ft"


def rank_of(gold, ids):
    for i, cid in enumerate(ids, 1):
        if cid == gold:
            return i
    return None


def summarize(ranks):
    n = len(ranks)
    found = [r for r in ranks if r is not None]
    return {
        "Hit@1": sum(r <= 1 for r in found) / n,
        "Hit@3": sum(r <= 3 for r in found) / n,
        "Hit@10": sum(r <= 10 for r in found) / n,
        "MRR": sum(1 / r for r in found) / n,
        "nDCG@10": sum(1 / math.log2(r + 1) for r in found) / n,
    }


def rerank_ids(reranker, query, candidates, top_k=10):
    scores = reranker.predict([(query, c["text"]) for c in candidates])
    order = np.argsort(scores)[::-1][:top_k]
    return [candidates[i]["chunk_id"] for i in order]


def main() -> None:
    if not FT_PATH.exists():
        print(f"Fine-tuned model not found at {FT_PATH}.\n"
              "Run notebooks/finetune_reranker_colab.ipynb on Colab, then unzip the "
              "downloaded model into models/bge-reranker-base-ft/ and re-run this.")
        return

    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    R = Retriever()                       # R.reranker = the original BGE reranker
    ft = CrossEncoder(str(FT_PATH))       # the fine-tuned reranker

    base_ranks, ft_ranks = [], []
    for row in tqdm(rows, desc="Comparing"):
        q, gold = row["query"], row["gold_chunk_id"]
        vec = R.vector_search(q, 20)
        bm = R.bm25_search(q, 20)
        fused = R.reciprocal_rank_fusion([vec, bm])[:30]
        base_ranks.append(rank_of(gold, rerank_ids(R.reranker, q, fused)))
        ft_ranks.append(rank_of(gold, rerank_ids(ft, q, fused)))

    base, fine = summarize(base_ranks), summarize(ft_ranks)
    metrics = ["Hit@1", "Hit@3", "Hit@10", "MRR", "nDCG@10"]
    print("\n" + "=" * 76)
    print(f"{'reranker':<22}" + "".join(f"{m:>10}" for m in metrics))
    print("-" * 76)
    print(f"{'base (BGE)':<22}" + "".join(f"{base[m]:>10.3f}" for m in metrics))
    print(f"{'fine-tuned':<22}" + "".join(f"{fine[m]:>10.3f}" for m in metrics))
    print(f"{'delta':<22}" + "".join(f"{fine[m]-base[m]:>+10.3f}" for m in metrics))
    print("=" * 76)


if __name__ == "__main__":
    main()
