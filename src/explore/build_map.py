"""
Build the data for the Research Map (a 2D map of the whole paper corpus).

Pipeline (uses the EXISTING embeddings -- NO LLM tokens):
  1. average the chunk embeddings per paper  -> one vector per paper
  2. project those vectors to 2D with t-SNE  -> (x, y) per paper
  3. cluster the papers into topics with KMeans
  4. name each topic from its papers' title keywords (token-free)
  5. save everything to data/research_map.json for the UI to plot

Run it with:   python -m src.explore.build_map
"""
import json
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from src.config import DATA_DIR, CHUNKS_PATH

EMB_PATH = DATA_DIR / "embeddings.npy"
META_PATH = DATA_DIR / "papers_meta.json"
OUT_PATH = DATA_DIR / "research_map.json"

N_CLUSTERS = 8
STOPWORDS = set(
    "a an the of for and to in on with using via based from into over via are is "
    "we our this that these those it its as at by be can model models method methods "
    "approach learning towards toward new large language".split()
)


def build_map() -> list[dict]:
    emb = np.load(EMB_PATH)                                   # (n_chunks, 384)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    meta = {p["arxiv_id"]: p
            for p in json.loads(META_PATH.read_text(encoding="utf-8"))}

    # 1) average chunk vectors -> one vector per paper
    sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(emb.shape[1], dtype="float64"))
    counts: dict[str, int] = defaultdict(int)
    for i, c in enumerate(chunks):
        sums[c["arxiv_id"]] += emb[i]
        counts[c["arxiv_id"]] += 1
    paper_ids = [a for a in sums if a in meta]
    vecs = np.array([sums[a] / counts[a] for a in paper_ids])
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    print(f"{len(paper_ids)} papers -> projecting to 2D...")

    # 2) project to 2D
    perplexity = min(30, max(5, len(paper_ids) // 4))
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  random_state=42).fit_transform(vecs)

    # 3) cluster into topics
    k = min(N_CLUSTERS, len(paper_ids))
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(vecs)

    # 4) name each cluster from its papers' title keywords (no LLM)
    topic_name: dict[int, str] = {}
    for cl in range(k):
        words = []
        for a, lab in zip(paper_ids, labels):
            if lab == cl:
                words += [w for w in re.findall(r"[a-z]+", meta[a]["title"].lower())
                          if len(w) > 3 and w not in STOPWORDS]
        top = [w for w, _ in Counter(words).most_common(3)]
        topic_name[cl] = ", ".join(top) if top else f"cluster {cl}"

    # 5) assemble + save
    out = []
    for a, (x, y), lab in zip(paper_ids, coords, labels):
        out.append({
            "arxiv_id": a,
            "title": meta[a]["title"],
            "url": meta[a]["url"],
            "x": float(x), "y": float(y),
            "cluster": int(lab),
            "topic": topic_name[int(lab)],
        })
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {len(out)} papers in {k} topics -> {OUT_PATH}")
    for cl in range(k):
        print(f"  topic {cl}: {topic_name[cl]}")
    return out


if __name__ == "__main__":
    build_map()
