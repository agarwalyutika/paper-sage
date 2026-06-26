"""
Phase 2, step 1: BUILD TRAINING DATA TO FINE-TUNE THE RERANKER.

For each sampled chunk we:
  - generate a question it answers (the POSITIVE: query <-> this chunk), via the 8B model
  - mine "hard negatives" with BM25: passages that look similar but are NOT the answer
    (these teach the reranker to tell near-misses apart) -- no LLM tokens needed

We EXCLUDE the chunks used in the eval set, so fine-tuning never sees the test data
(an honest before/after comparison). Output: evals/train_pairs.jsonl  (resumable).

Run it with:   python -m evals.generate_train_pairs
"""
import json
import pickle
import random
import re

import numpy as np
import groq
from tqdm import tqdm

from src.config import settings, CHUNKS_PATH, BM25_PATH, PROJECT_ROOT

OUT_PATH = PROJECT_ROOT / "evals" / "train_pairs.jsonl"
EVAL_PATH = PROJECT_ROOT / "evals" / "dataset.jsonl"
N_TRAIN = 250            # training queries (more = better fine-tune, but more 8B tokens)
N_NEGATIVES = 4          # hard negatives per query
RANDOM_SEED = 7

GEN_SYSTEM = """You are given a passage from a machine-learning paper. Write ONE \
specific, self-contained question that this passage directly answers. Mention the \
specific method/concept. Output ONLY the question."""


def looks_usable(text: str) -> bool:
    if len(text) < 700:
        return False
    return sum(c.isalpha() for c in text) / len(text) > 0.6


def main() -> None:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    by_id = {c["chunk_id"]: c for c in chunks}

    # Chunks used by the eval set -> must NOT be used for training (no leakage).
    eval_ids = {json.loads(l)["gold_chunk_id"] for l in open(EVAL_PATH, encoding="utf-8")}

    usable = [c for c in chunks if looks_usable(c["text"]) and c["chunk_id"] not in eval_ids]
    random.seed(RANDOM_SEED)
    sample = random.sample(usable, min(N_TRAIN, len(usable)))

    # BM25 index for mining hard negatives (no LLM tokens).
    bm = pickle.loads(BM25_PATH.read_bytes())
    bm25, bm_chunks = bm["bm25"], bm["chunks"]

    # Resume: skip queries already written.
    done_positives = set()
    if OUT_PATH.exists():
        for l in open(OUT_PATH, encoding="utf-8"):
            done_positives.add(json.loads(l)["positive_id"])
    sample = [c for c in sample if c["chunk_id"] not in done_positives]
    print(f"Generating {len(sample)} training pairs ({len(done_positives)} already done)...")

    client = groq.Groq(api_key=settings.GROQ_API_KEY)
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for c in tqdm(sample, desc="Train pairs"):
            try:
                resp = client.chat.completions.create(
                    model=settings.GROQ_FALLBACK_MODEL,           # 8B
                    messages=[{"role": "system", "content": GEN_SYSTEM},
                              {"role": "user", "content": c["text"][:1500] + "\n\nQuestion:"}],
                    temperature=0.3, max_tokens=80,
                )
                q = resp.choices[0].message.content.strip().strip('"')
                q = re.sub(r"^(question:?\s*)", "", q, flags=re.IGNORECASE).strip()
            except Exception as e:
                print(f"  ! skip {c['chunk_id']}: {e}")
                continue
            if len(q) < 10:
                continue

            # Hard negatives: top BM25 hits for the query that aren't the positive.
            scores = bm25.get_scores(q.lower().split())
            neg_texts = []
            for idx in np.argsort(scores)[::-1]:
                cand = bm_chunks[idx]
                if cand["chunk_id"] != c["chunk_id"]:
                    neg_texts.append(cand["text"])
                if len(neg_texts) >= N_NEGATIVES:
                    break

            out.write(json.dumps({
                "query": q,
                "positive_id": c["chunk_id"],
                "positive": c["text"],
                "negatives": neg_texts,
            }) + "\n")
            out.flush()

    total = sum(1 for _ in open(OUT_PATH, encoding="utf-8"))
    print(f"\nTotal training pairs: {total} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
