"""
Part A, step 1: BUILD A "GOLD STANDARD" EVALUATION SET.

For retrieval evaluation we need questions whose correct source passage is KNOWN.
Trick: sample real chunks, and for each, ask the LLM to write a question that
*that specific chunk* answers. Then "did retrieval return that chunk?" is a clean,
automatic correctness check.

We use the small 8B model (separate, higher free quota) so this doesn't burn the
70B budget. Output: evals/dataset.jsonl  (one {query, gold_chunk_id, ...} per line).

Run it with:   python -m evals.generate_dataset
"""
import json
import random
import re

import groq
from tqdm import tqdm

from src.config import settings, CHUNKS_PATH, PROJECT_ROOT

OUT_PATH = PROJECT_ROOT / "evals" / "dataset.jsonl"
N_QUERIES = 80          # size of the eval set
RANDOM_SEED = 42        # reproducible sample

GEN_SYSTEM = """You are given a passage from a machine-learning paper. Write ONE \
specific, self-contained question that this passage directly answers. The question \
must be answerable from THIS passage alone and mention the specific method/concept \
(not "this paper"). Output ONLY the question, nothing else."""


def looks_usable(text: str) -> bool:
    """Skip tiny chunks and reference-list / equation-heavy junk."""
    if len(text) < 700:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / len(text) > 0.6        # mostly prose, not symbols/refs


def main() -> None:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    usable = [c for c in chunks if looks_usable(c["text"])]
    random.seed(RANDOM_SEED)
    sample = random.sample(usable, min(N_QUERIES, len(usable)))
    print(f"{len(usable):,} usable chunks; sampling {len(sample)} for the eval set.")

    client = groq.Groq(api_key=settings.GROQ_API_KEY)
    rows = []
    for c in tqdm(sample, desc="Generating questions"):
        try:
            resp = client.chat.completions.create(
                model=settings.GROQ_FALLBACK_MODEL,   # 8B: cheap + separate quota
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
        rows.append({
            "query": q,
            "gold_chunk_id": c["chunk_id"],
            "gold_arxiv_id": c["arxiv_id"],
            "gold_title": c["title"],
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(rows)} eval examples -> {OUT_PATH}")


if __name__ == "__main__":
    main()
