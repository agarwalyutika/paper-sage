# Agentic RAG over ML Research Papers — Design & Roadmap

**Resume line:** *Agentic RAG System over ML Research Papers — full-stack retrieval-augmented QA with hybrid retrieval (BM25 + vector), cross-encoder reranking, citation enforcement, and a CI-gated evaluation pipeline.*

---

## 1. Problem (the "why" recruiters read)
ML researchers/engineers/students face hundreds of new arXiv papers daily. Keyword search misses
conceptually-related work using different terminology; reading full papers to extract one detail is slow;
many real questions are **multi-hop** (synthesize across 2+ papers). This system answers natural-language
questions with **accurate, source-grounded, citable** answers over a corpus of real ML papers, handling
both direct and multi-hop questions, fully open-source-friendly and runnable on free-tier GPU infra.

---

## 2. Decisions locked
- **LLM stack: Hybrid.** OSS embeddings + OSS cross-encoder reranker (local/free); generation via an
  API (Claude) behind a `LLMProvider` interface so it can be swapped for a local model later.
- **Frontend: Streamlit now**, migrate to Next.js + FastAPI later (interface already split so the swap is clean).
- **Mode: design-first** — this doc is the contract; we scaffold code after review.

---

## 3. Architecture (high level)

```
                                ┌─────────────────────────────┐
   arXiv API ──► Ingestion ──►  │  Chunk + Embed + Index       │
   (PDF/LaTeX)   pipeline       │  - BM25 index (rank_bm25)    │
                                │  - Vector index (Qdrant)     │
                                │  - chunk metadata (Postgres) │
                                └──────────────┬──────────────┘
                                               │
 User ─► Streamlit UI ─► FastAPI ─► Agent (LangGraph) ──► Retrieval Layer
                                        │                    ├─ BM25 search
                                        │                    ├─ Vector search
                                        │                    ├─ Reciprocal Rank Fusion
                                        │                    └─ Cross-encoder rerank
                                        │
                                        ├─ Multi-hop planner (decompose → sub-queries → gather)
                                        ├─ Generation (LLMProvider → Claude)  ◄─ citation enforcement
                                        └─ Answer + inline [^n] citations + source passages
                                               │
                          Langfuse tracing  ◄──┘     Eval pipeline (Ragas/DeepEval) ──► CI gate
```

---

## 4. Tech stack (concrete)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | standard for AI eng |
| API | FastAPI + Uvicorn | async, Swagger docs, the "backend" signal |
| Agent/orchestration | LangGraph | explicit graph = controllable multi-hop, easy to show in README |
| Vector DB | Qdrant (Docker, local) | free, fast, hybrid-friendly; Cloud free tier exists |
| Keyword search | `rank_bm25` (or Qdrant sparse vectors) | the BM25 half of hybrid |
| Embeddings (OSS) | `BAAI/bge-small-en-v1.5` (or `bge-base`) | strong, runs on CPU/free GPU |
| Reranker (OSS) | `BAAI/bge-reranker-base` cross-encoder | the precision step; the "fine-tuned reranking" line |
| Generation | Claude via `anthropic` SDK, behind `LLMProvider` | reliable demos; swappable for local Llama/Qwen later |
| Metadata store | Postgres (or SQLite to start) | chunk ↔ paper ↔ section mapping for citations |
| Tracing | Langfuse (self-host or free cloud) | production-maturity signal |
| Eval | Ragas + DeepEval | faithfulness, context recall/precision, answer relevancy |
| Frontend | Streamlit → (later) Next.js | ship fast now |
| CI | GitHub Actions | runs eval suite, **gates merge on score thresholds** |
| Deploy | HF Spaces / Streamlit Cloud + Qdrant Cloud | free, clickable live link |
| Packaging | Docker + docker-compose | one-command run |

**Swappable interface (the key to "hybrid + future migration"):**
```python
class LLMProvider(Protocol):
    def generate(self, prompt: str, **kw) -> str: ...
# AnthropicProvider(...) now;  VLLMProvider(...) / OllamaProvider(...) later — no other code changes.
```

---

## 5. Data pipeline
1. **Fetch:** arXiv API by category (`cs.CL`, `cs.LG`, `cs.AI`) — start with ~200–500 papers (enough to be real, small enough to be free).
2. **Parse:** PDFs via `PyMuPDF`/`GROBID` → keep section structure (Abstract, Method, Results). Section-aware = better citations.
3. **Chunk:** ~512-token chunks with overlap, **store metadata** {paper_id, title, authors, arxiv_url, section, chunk_id}.
4. **Embed + index:** write vectors to Qdrant, build BM25 index, store metadata in Postgres.
5. **Idempotent + incremental:** re-running ingestion adds only new papers (shows engineering, not a one-off script).

---

## 6. Retrieval layer (the core)
- **Hybrid:** run BM25 and vector search in parallel → combine with **Reciprocal Rank Fusion (RRF)**.
- **Rerank:** cross-encoder (`bge-reranker-base`) scores top-K (e.g. 30 → 6). This is the precision jump you'll graph in the README.
- **Stretch (the "fine-tuned reranking" headline):** generate a small synthetic (query, relevant-passage) set from your corpus and fine-tune the reranker; report nDCG@10 before/after. *This is the single most senior-signaling piece — keep it as a Phase 2 stretch.*

---

## 7. Agentic layer (multi-hop)
LangGraph state machine:
1. **Router/planner** — classify question as *direct* vs *multi-hop*; if multi-hop, decompose into sub-questions.
2. **Retrieve** per sub-question (hybrid + rerank).
3. **Gather/synthesize** — combine evidence, dedupe sources.
4. **Generate** with citation enforcement.
5. **Self-check** — verify each claim is supported by a retrieved passage; if not, retry or say "not enough evidence."

Keep it minimal and reliable (hiring teams reward restraint over a sprawling agent zoo).

---

## 8. Citation enforcement (trust layer)
- Prompt + output schema forces inline markers `[^1]`, `[^2]` mapped to real chunk IDs.
- **Post-generation validator:** every citation must resolve to a retrieved passage; strip/flag unsupported claims.
- UI shows the answer with clickable sources (title, section, arXiv link, highlighted passage).
- Explicit **"I don't have enough evidence"** path — never bluff.

---

## 9. Evaluation pipeline (the crown jewel — what gets you hired)
- **Eval set:** 40–60 curated Q→A pairs (direct + multi-hop) with gold source paper IDs. Store as `evals/dataset.jsonl`.
- **Metrics (Ragas/DeepEval):** faithfulness, context precision, context recall, answer relevancy; plus **retrieval** metrics (hit-rate, MRR, nDCG@10).
- **Ablations to report in README:** vector-only vs hybrid vs hybrid+rerank → a table/graph showing the lift. This *is* your portfolio story.
- **CI gate:** GitHub Actions runs the eval suite on PRs; **merge blocked if faithfulness/recall drop below thresholds.** This one feature signals more production maturity than the whole rest of the project.

---

## 10. Repo structure (when we scaffold)
```
agentic-rag-ml-papers/
├─ README.md            # problem, architecture diagram, eval table, live demo link
├─ docker-compose.yml   # app + qdrant + postgres + langfuse
├─ pyproject.toml
├─ .github/workflows/eval.yml      # CI-gated eval
├─ src/
│  ├─ ingestion/        # fetch, parse, chunk, index
│  ├─ retrieval/        # bm25, vector, rrf, rerank
│  ├─ agent/            # langgraph nodes, planner, self-check
│  ├─ generation/       # LLMProvider interface + AnthropicProvider
│  ├─ citations/        # validator
│  └─ api/              # FastAPI routes
├─ app/                 # Streamlit UI
├─ evals/               # dataset.jsonl + runner + thresholds
└─ tests/
```

---

## 11. Roadmap (week by week, ~part-time)
- **Week 1 — Foundations.** Repo + docker-compose (Qdrant, Postgres). Ingest 200 papers. BM25 + vector search returning raw chunks. *Milestone: ask a question via script, get relevant chunks.*
- **Week 2 — Hybrid + rerank + API.** RRF fusion, cross-encoder rerank, FastAPI `/ask`, `LLMProvider`→Claude, basic citations. *Milestone: grounded answer with sources over the API.*
- **Week 3 — Agentic multi-hop + citation enforcement.** LangGraph planner, self-check node, citation validator, Streamlit UI, Langfuse tracing. *Milestone: a multi-hop question answered correctly with valid citations in the UI.*
- **Week 4 — Evals + CI + deploy + README.** Build eval set, Ragas/DeepEval, ablation table, GitHub Actions CI gate, deploy to HF Spaces, write the README + record a 2-min demo. *Milestone: live link + green CI + eval graph.*
- **Phase 2 stretches.** Fine-tune the reranker (nDCG before/after); migrate UI to Next.js + FastAPI; add a local OSS generation provider; conversation memory.

---

## 12. Resume bullets (use after building)
- Built a full-stack **agentic RAG** system over ~500 ML papers with **hybrid retrieval (BM25 + dense) + cross-encoder reranking**, improving retrieval nDCG@10 by **X%** over vector-only.
- Engineered **citation enforcement + self-verification**, eliminating unsupported claims and adding an explicit no-answer path.
- Shipped a **CI-gated evaluation pipeline** (Ragas/DeepEval) blocking merges on faithfulness/recall regressions; full Langfuse tracing.
- Deployed end-to-end (Docker, Qdrant, FastAPI, Streamlit) on free-tier infra with a live public demo.

---

## 13. Open questions to resolve before scaffolding
1. Corpus size for v1 — 200 (fast) vs 500 (more impressive)?
2. Topic focus — broad ML, or niche down (e.g. only LLM/RAG/agents papers) for sharper multi-hop demos?
3. Langfuse now or add in Week 3?
4. Postgres from day 1, or SQLite first then migrate?
```
