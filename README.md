# 📚 PaperSage

### Agentic RAG over ML research papers — grounded, cited answers

A full-stack, **fully open-source** Retrieval-Augmented Generation (RAG) system that answers
natural-language questions about machine-learning research papers — with **grounded, cited
answers** and an honest "I don't know" when the evidence isn't there.

> **Ask a question → get an accurate answer drawn only from real ML papers, with clickable citations.**
> Think "ChatGPT, but it can only answer from real papers, and it always shows its receipts."

---

## Why this exists

ML researchers, engineers, and students face hundreds of new arXiv papers daily. Keyword search
misses conceptually-related work that uses different terminology; reading full papers to extract a
single methodological detail is slow. This system lets you ask in plain English and get a precise,
**source-grounded** answer — fast.

It is built to be **trustworthy**: every claim is cited to a specific paper, citations are
validated, and the system **refuses to answer** when the papers don't support one (no hallucinated
facts).

---

## ✨ Features

- **Hybrid retrieval** — combines BM25 keyword search + dense vector search (Reciprocal Rank Fusion)
- **Cross-encoder reranking** — re-scores candidates for precision (BGE reranker)
- **Grounded generation** — answers strictly from retrieved passages, never outside knowledge
- **Citation enforcement + validation** — every claim cites `[n]`; invalid citations are flagged
- **Grounded-vs-general routing** — the generator decides whether the retrieved passages truly answer the question: grounds + cites when they do, or falls back when they don't
- **Web-search fallback** — when the papers don't cover a question, it searches the web (DuckDuckGo; Tavily optional) and answers with **clickable web citations**, before settling for general knowledge. Full agentic source chain: *papers → web → general knowledge*
- **Honest refusal** — says "not enough information" instead of bluffing
- **Agentic router** — an LLM decides per message: just chat (greetings, "who are you") vs. retrieve-and-cite
- **Conversational chat with memory** — multi-turn follow-ups ("what about its limitations?") via question condensing
- **Persistent sessions** — ChatGPT-style sidebar of saved chats (SQLite); your history survives restarts
- **Bring your own documents** — attach a PDF/txt in the chat box and ask questions about *your* file (in-memory index, page-level citations)
- **Paper comparison** — pick 2–4 corpus papers **and/or upload your own PDFs**, and get a detailed LLM-synthesized side-by-side comparison table (problem, method, novelty, dataset, results, strengths, limitations) — multi-document synthesis
- **Research Map** — an interactive 2D map of all 200 papers (t-SNE over the embeddings + KMeans topics); click a dot to open the paper. Zero extra LLM calls
- **Auto-diagrams** — turn any answer into a Mermaid concept flowchart on demand (one click, rendered inline)
- **Swappable LLM backend** — free hosted Llama (Groq) by default; Claude or local Ollama via one config flag
- **100% open-source & free** — open models for embeddings/reranking, free Groq tier for generation
- **Streamlit UI** — chat interface with a 📎 attach button and expandable source cards

---

## 🏗️ Architecture

```
arXiv papers ──► chunk ──► embed (BGE, on GPU) ──► [Qdrant vectors + BM25 index]
                                                            │
 user question ─► Streamlit ─► hybrid retrieval (BM25 + vector, fused via RRF)
                                   └─► cross-encoder rerank → top 6 passages
                                          └─► LLM (Llama via Groq) writes a CITED answer
                                                 └─► citation validator → grounded answer + sources
```

---

## 🧰 Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.12 |
| Ingestion | `arxiv`, `PyMuPDF` |
| Embeddings | `BAAI/bge-small-en-v1.5` (open-source) |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder) |
| Vector DB | Qdrant (local, embedded) |
| Keyword search | `rank_bm25` |
| Generation | Llama 3.3 70B via **Groq** (free) — swappable for Claude / Ollama |
| UI | Streamlit |
| GPU (one-time indexing) | Google Colab |

---

## 📊 Retrieval evaluation

An automatic benchmark of **80 questions** (each generated from a known source passage,
so the correct answer is known). Measured across three retrieval configurations — uses
**no LLM tokens** (pure retrieval). Each component earns its place:

| Mode | Hit@1 | Hit@3 | Hit@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| Vector-only | 0.45 | 0.63 | 0.76 | 0.55 | 0.60 |
| + BM25 (hybrid, RRF) | 0.56 | 0.85 | 0.93 | 0.70 | 0.75 |
| **+ Cross-encoder rerank** | **0.78** | **0.95** | **0.96** | **0.86** | **0.89** |

Hybrid retrieval + reranking lifts **nDCG@10 from 0.60 → 0.89 (+48%)** and **Hit@1 from
45% → 78%** over vector-only. Reproduce with `python -m evals.run_eval`.

> *Note: this is a **synthetic** benchmark (questions auto-generated from passages), so the
> absolute scores are likely a bit optimistic vs. human-written questions. The synthetic
> bias applies equally to all three modes, so the **relative comparison is fair** — the
> reranking gain is real.*

### Fine-tuned reranker (Phase 2)

I fine-tuned the BGE cross-encoder on **249 in-domain (question → passage) pairs** with
BM25-mined **hard negatives** (trained on Colab GPU; eval chunks excluded — no leakage).
Re-running the same held-out benchmark:

| Reranker | Hit@1 | Hit@3 | MRR | nDCG@10 |
|---|---|---|---|---|
| Base BGE | 0.76 | 0.95 | 0.85 | 0.88 |
| **Fine-tuned** | **0.83** | **0.96** | **0.89** | **0.91** |

Fine-tuning lifted **Hit@1 from 76% → 83% (+8%)** and nDCG@10 to **0.91** — the correct
passage lands at rank #1 more often, with no regression on any metric. Reproduce with
`python -m evals.compare_reranker`.

## 🚀 Setup

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd agentic-rag-ml-papers

# 2. Create an isolated environment
python -m venv .venv
# Windows:  .\.venv\Scripts\activate     |  macOS/Linux:  source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your free Groq key (https://console.groq.com — no card needed)
cp .env.example .env       # then paste your gsk_... key into .env
```

### Build the corpus and indexes

```bash
# Download + curate ML papers from arXiv
python -m src.ingestion.fetch

# Slice papers into chunks
python -m src.ingestion.chunk

# Create embeddings:
#   - Fast path: run notebooks/embed_on_colab.ipynb on a free Colab GPU,
#     download embeddings.npy into data/
#   (the chunks are CPU-light at query time; only bulk embedding wants a GPU)

# Build the vector + keyword indexes from the embeddings
python -m src.retrieval.build_index
```

### Run it

```bash
# Command line
python -m src.generation.answer "How can multi-agent systems improve privacy in RAG?"

# Web UI
streamlit run app/app.py
```

---

## 📁 Project structure

```
agentic-rag-ml-papers/
├─ src/
│  ├─ ingestion/   fetch.py, chunk.py             # download + curate + chunk papers
│  ├─ retrieval/   build_index.py, search.py,     # hybrid search + reranking
│  │               uploaded_docs.py               # in-memory search over uploaded files
│  ├─ generation/  provider.py, answer.py         # swappable LLM + cited answers
│  ├─ citations/   validator.py                   # citation validation
│  ├─ chat/        store.py, conversation.py       # persistent sessions + multi-turn router
│  └─ config.py                                   # all settings in one place
├─ app/            app.py                     # Streamlit UI
├─ notebooks/      embed_on_colab.ipynb       # GPU embedding
├─ evals/                                     # evaluation (planned)
└─ data/                                      # papers, indexes (git-ignored)
```

---

## 🗺️ Roadmap

- [x] **Conversational chat + persistent sessions** — multi-turn follow-ups, saved chat history
- [x] **Upload-your-own-documents** — ask questions about your own PDFs/reports
- [x] **Evaluation pipeline** — Hit@k / MRR / nDCG@10 across vector-only vs hybrid vs +rerank (ablation table above)
- [x] **Fine-tuned reranker** — domain-tuned the cross-encoder; Hit@1 76% → 83%, nDCG@10 → 0.91 (see above)
- [ ] **Web search** — optionally cite live web pages alongside papers
- [ ] **Next.js front end** — production full-stack UI

---

## 📝 Notes

- Embeddings and indexes are **rebuildable** and git-ignored to keep the repo small.
- Generation runs on a free hosted endpoint by default; set `LLM_BACKEND=ollama` for fully-offline local inference.
- Built as a portfolio project demonstrating production RAG patterns: hybrid retrieval, reranking,
  grounded generation, citation validation, and vendor-neutral LLM abstraction.
