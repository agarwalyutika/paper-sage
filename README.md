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
- **Honest refusal** — says "not enough information" instead of bluffing
- **Agentic router** — an LLM decides per message: just chat (greetings, "who are you") vs. retrieve-and-cite
- **Conversational chat with memory** — multi-turn follow-ups ("what about its limitations?") via question condensing
- **Persistent sessions** — ChatGPT-style sidebar of saved chats (SQLite); your history survives restarts
- **Bring your own documents** — attach a PDF/txt in the chat box and ask questions about *your* file (in-memory index, page-level citations)
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
- [ ] **Evaluation pipeline** — Ragas/DeepEval metrics + a vector-only vs hybrid vs +rerank ablation table
- [ ] **Fine-tuned reranker** — domain-tune the reranker on the corpus; report nDCG@10 before/after
- [ ] **Web search** — optionally cite live web pages alongside papers
- [ ] **Next.js front end** — production full-stack UI

---

## 📝 Notes

- Embeddings and indexes are **rebuildable** and git-ignored to keep the repo small.
- Generation runs on a free hosted endpoint by default; set `LLM_BACKEND=ollama` for fully-offline local inference.
- Built as a portfolio project demonstrating production RAG patterns: hybrid retrieval, reranking,
  grounded generation, citation validation, and vendor-neutral LLM abstraction.
