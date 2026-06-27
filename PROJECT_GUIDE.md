# 📘 PaperSage — Complete Project Guide & Study Notes

> A plain-English, end-to-end explanation of everything we built: the concepts, the tech,
> the process, the challenges, a glossary, and interview Q&A. Read top to bottom and you'll
> understand *and be able to defend* the whole project.

---

## Table of Contents
1. [The 30-second pitch](#1-the-30-second-pitch)
2. [The big idea: what is RAG, really?](#2-the-big-idea-what-is-rag-really)
3. [The full architecture (the map)](#3-the-full-architecture-the-map)
4. [Every component, explained](#4-every-component-explained)
5. [The tech stack (what & why)](#5-the-tech-stack-what--why)
6. [The power features](#6-the-power-features)
7. [Challenges we hit & how we solved them](#7-challenges-we-hit--how-we-solved-them)
8. [Results & metrics](#8-results--metrics)
9. [Glossary of every term](#9-glossary-of-every-term)
10. [Interview questions & answers](#10-interview-questions--answers)
11. [Resume bullets & how to talk about it](#11-resume-bullets--how-to-talk-about-it)

---

## 1. The 30-second pitch

**PaperSage** is an *agentic RAG* system that answers questions about machine-learning research
papers with **grounded, cited answers**. You ask in plain English; it finds the right passages from
200 real papers, writes an answer, and **shows its sources**. If the papers don't cover your
question, it falls back to **web search** (cited), then to **general knowledge** (clearly labeled).
On top of chat, it can **compare papers**, **generate code from a paper**, **make quizzes**, draw a
**research map**, and **analyze the novelty of a research idea**.

**Why it matters:** a normal chatbot can confidently make things up. PaperSage is built to be
**trustworthy** — every research answer is traceable to a source, and it never passes off a guess as
grounded.

---

## 2. The big idea: what is RAG, really?

**RAG = Retrieval-Augmented Generation.**

A large language model (LLM) like Llama or GPT knows a lot, but:
- its knowledge is **frozen** at training time (no new papers),
- it can **hallucinate** (make up confident nonsense),
- it can't cite **specific sources**.

**RAG fixes this by adding a "retrieve" step before "generate":**

```
Normal LLM:   question ─────────────────► answer   (from memory, may hallucinate)

RAG:          question ─► SEARCH your documents ─► feed top passages ─► answer (grounded + cited)
```

So RAG = **"open-book exam"** for an LLM. Instead of answering from memory, it first looks up the
relevant pages, then answers using them. That's the whole concept. Everything else is making each
step (search, rank, generate) better and more trustworthy.

---

## 3. The full architecture (the map)

```
                         ┌──────────────── ONE-TIME SETUP (the "index") ───────────────┐
  arXiv papers ─► curate ─► chunk ─► embed (Colab GPU) ─► [ Qdrant vectors + BM25 index ]
                                                                       │
  ┌──────────────────────────── AT QUERY TIME ──────────────────────── │ ───────────────┐
  │                                                                     │                │
  │  your question ─► router: chit-chat? code? research?                │                │
  │                       │ research                                    │                │
  │                       ▼                                             ▼                │
  │              hybrid retrieval  =  BM25 (keywords) + vector (meaning), fused with RRF  │
  │                       ▼                                                              │
  │              cross-encoder rerank (FINE-TUNED)  → top 6 passages                     │
  │                       ▼                                                              │
  │        "do these passages actually answer it?"                                      │
  │           ├─ yes ─► grounded, cited answer from papers                               │
  │           └─ no  ─► 🌐 web search (cited) ─► ℹ️ general knowledge                     │
  │                       ▼                                                              │
  │        citation validation · concept diagram · saved to SQLite                       │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

There are **two phases**:
- **Indexing (done once):** turn papers into a searchable index. Slow, GPU-heavy.
- **Querying (every question):** search the index + write an answer. Fast, runs on CPU.

This split is *the* key design decision in any RAG system.

---

## 4. Every component, explained

### 4.1 Ingestion (`src/ingestion/fetch.py`)
**What:** download ~200 ML papers from arXiv and keep only the relevant ones.
**How:** the `arxiv` library queries arXiv's API by topic + category, downloads PDFs, and saves a
"catalog" (`papers_meta.json`) with each paper's title, abstract, authors, and link.
**The clever bit — relevance filtering:** even a focused query leaks off-topic papers. So we read
each paper's **abstract** and keep it only if it contains LLM/RAG/agent keywords. This is **corpus
curation** — keeping the knowledge base on-domain so retrieval stays precise.
**Engineering habits:** the download is **idempotent** (re-running skips files we already have) and
**fault-tolerant** (one bad download doesn't crash the batch).

### 4.2 Chunking (`src/ingestion/chunk.py`)
**What:** cut each paper into small overlapping pieces ("chunks") — 14,348 of them.
**Why:** a paper is ~10 pages. Feeding a whole paper to the LLM is slow, expensive, and vague.
Instead we find the *exact* small pieces that answer a question. We chunk to ~1200 characters with
~200 overlap.
**Why overlap?** If we cut blindly, a key sentence could be split across two chunks. Overlap means
the boundary sentence appears whole in at least one chunk.
**Each chunk keeps its source label** (which paper) — that's what powers **citations** later.

### 4.3 Embeddings (`embeddings.npy`, Colab notebook)
**What:** turn each chunk's text into a **vector** (a list of 384 numbers) using the **BGE** model.
**The core idea:** an embedding model maps text into a "meaning space" where **similar meanings get
similar numbers** — even if they use different words. So "reduces memory" and "lowers VRAM" land
close together. This is how we search by *meaning*, not keywords.
**Why Colab:** embedding 14,348 chunks is GPU-heavy and the laptop hangs on CPU, so we did this
one-time job on a free Colab GPU, then downloaded the `embeddings.npy` file.

### 4.4 The indexes (`src/retrieval/build_index.py`)
We build **two** indexes from the chunks:
- **Vector index (Qdrant):** stores the 384-number vectors → enables **semantic / meaning** search.
- **BM25 index (`rank_bm25`):** a classic **keyword** index → enables exact-term search.
Having both is what "hybrid" means.

### 4.5 The search engine (`src/retrieval/search.py`) — the heart
For each question:
1. **Vector search** → finds chunks with similar *meaning*.
2. **BM25 search** → finds chunks with matching *keywords* (great for exact terms like "LoRA", model
   names, acronyms that vectors blur).
3. **Reciprocal Rank Fusion (RRF)** → merges the two ranked lists fairly. Each chunk earns
   `1/(k+rank)` from every list it appears in; chunks both methods like rise to the top.
4. **Cross-encoder reranking** → a second, slower-but-precise model reads `(question, passage)`
   *together* and scores how well each passage answers the question. We rerank the top ~30 and keep
   the best 6.

**Bi-encoder vs cross-encoder (key distinction):** the embedding model is a *bi-encoder* — it
encodes the query and passage *separately* (fast, approximate). The reranker is a *cross-encoder* —
it reads query+passage *together* (slow, precise). The pattern is **retrieve wide (fast), then
rerank narrow (precise).**

### 4.6 Generation (`src/generation/`)
- **`provider.py`** — the **swappable LLM layer**. Every backend (Groq/Llama, Gemini, Claude,
  Ollama) implements the same `generate(system, user)` method. Switching is a one-line config
  change. This is **vendor-neutrality** / the *adapter pattern*.
- **`answer.py`** — builds a strict prompt: *"answer ONLY from these numbered sources, cite every
  claim with [n], and if they don't answer it, say so."* Then it decides **grounded vs general**:
  if the passages truly answer the question → cited answer; if not → labeled general knowledge.
- **`citations/validator.py`** — checks that every `[n]` in the answer points to a *real* source
  (catches invented citations). The "show your receipts" guarantee.

### 4.7 The agentic layer (`src/chat/conversation.py`)
This makes it feel like a real assistant:
- **Router:** classifies each message — *chit-chat* (say hi back), *code request* (generate code),
  or *research* (retrieve + answer). "Agentic" = the system *decides* what to do.
- **Question condensing:** for follow-ups like *"what about its limitations?"*, it rewrites the
  question into a standalone one ("limitations of multi-agent RAG privacy?") so search works.
- **Source chain:** papers → web → general knowledge, each clearly labeled.

### 4.8 Persistence (`src/chat/store.py`)
A small **SQLite** database stores chat sessions, messages, and saved novelty analyses — so your
work survives closing the app. Pattern: *a table + save/list/get/delete functions.*

---

## 5. The tech stack (what & why)

| Tool | What it is | Why we chose it |
|---|---|---|
| **Python 3.12** | Language | The standard for AI/ML |
| **arxiv, PyMuPDF** | Fetch + read PDFs | Free, simple |
| **BGE (`bge-small-en-v1.5`)** | Embedding model | Top-ranked, small (runs on CPU at query time), open-source |
| **BGE reranker (`bge-reranker-base`)** | Cross-encoder | Precision step; we **fine-tuned** it |
| **Qdrant** | Vector database | Free, runs locally (no server) |
| **rank_bm25** | Keyword search | The BM25 half of hybrid |
| **Groq / Gemini / Claude / Ollama** | LLM backends | Swappable; Groq+Gemini are free |
| **DuckDuckGo (`ddgs`)** | Web search | Free, no API key |
| **scikit-learn** | t-SNE + KMeans | The Research Map |
| **Plotly** | Interactive charts | The clickable map |
| **Mermaid.js** | Diagrams | The auto-diagrams |
| **Streamlit** | Web UI | Fast to build, deploys free |
| **SQLite** | Storage | Built into Python, zero setup |
| **Google Colab** | GPU | Free GPU for embedding + fine-tuning |
| **Git/GitHub** | Version control | Portfolio link |

---

## 6. The power features

| Feature | How it works (one line) |
|---|---|
| **Cited chat** | retrieve → rerank → grounded answer with `[n]` citations |
| **Agentic source chain** | papers → web search (cited) → general knowledge, all labeled |
| **Doc upload** | your PDF → extract → in-memory index → ask about *your* file |
| **Paper comparison** | gather N papers' text → LLM makes a detailed side-by-side table |
| **Code generation** | retrieve a method's chunks → LLM writes runnable PyTorch, cites `# [n]` |
| **Quiz generator** | paper context → LLM returns JSON → MCQs / flashcards / interview Qs |
| **Research Map** | average embeddings per paper → t-SNE to 2D → KMeans topics → clickable Plotly map |
| **Auto-diagrams** | answer → LLM writes Mermaid flowchart → rendered inline |
| **Find Novelty** | idea → retrieve related work → analyze similar/done/gaps/novel + follow-up chat |

**The pattern to notice:** once the retrieval layer is solid, **each new feature is mostly a new
prompt** on the same engine. That's the payoff of clean foundations.

---

## 7. Challenges we hit & how we solved them

These are the real engineering stories — *gold* for interviews.

1. **Off-topic papers polluted the corpus.**
   *Fix:* abstract-level relevance filtering during ingestion. Lesson: garbage in → garbage out;
   curate the knowledge base.

2. **Laptop hangs on CPU for heavy compute.**
   *Fix:* do the GPU-heavy one-time jobs (embedding, fine-tuning) on **Colab**, keep query-time
   light on CPU. Lesson: separate heavy one-time work from light per-query work.

3. **Long batch jobs got interrupted.**
   *Fix:* made them **resumable** (save progress in shards/incrementally). Lesson: idempotent,
   resumable pipelines.

4. **Qdrant local mode is single-process** (the app and a script can't both open it).
   *Fix:* understood the file-lock; for concurrency you'd run Qdrant as a server. Lesson: know your
   storage engine's concurrency model.

5. **Free LLM daily token limit hit** (Groq 70B exhausted → weak 8B fallback).
   *Fix:* added **automatic model fallback** (70B → 8B), then a **4th backend (Gemini)** for a
   fresh strong free model. Lesson: design for rate limits; vendor-neutral abstraction makes
   swapping trivial.

6. **"What is an LLM?" gave a weak grounded answer.** The papers *use* LLMs but don't *define* them.
   *Fix:* **grounded-vs-general routing** — the LLM decides whether the passages actually answer the
   question; if not, it answers from general knowledge (labeled) or the web. Lesson: a RAG answer is
   only as good as the corpus fit.

7. **Reranker score didn't separate "answerable" from "on-topic."** We measured it: "what is an LLM"
   scored 0.88 (high — papers *are* about LLMs) yet couldn't be answered. *Insight:* the reranker
   measures **topical relevance, not answerability.** *Fix:* let the *generator* judge answerability
   (it can read the passages). Lesson: pick the right signal for the job; measure before assuming.

8. **The model leaked its reasoning** ("[1] is not relevant…") and prompt labels ("A) GROUNDED")
   into answers. *Fix:* tightened prompts to forbid meta-commentary and label output. Lesson:
   prompt engineering is real engineering; weak models copy whatever's in the prompt.

9. **Runaway repetition loops** on a weak model. *Fix:* a `max_tokens` cap + frequency penalty.

10. **Weak models emit broken JSON** (quiz feature). *Fix:* a **defensive parser** (strip fences,
    extract the array) with a **raw-text fallback** — never crash on model output.

11. **Comparison was too brief.** *Fix:* three levers — more context per paper, a bigger
    `max_tokens` budget, and a prompt demanding specifics. Lesson: answer quality = context +
    token budget + prompt, not just the model.

---

## 8. Results & metrics

**Retrieval ablation** (80 auto-generated "gold" questions; each has a known correct passage):

| Mode | Hit@1 | Hit@3 | Hit@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| Vector-only | 0.45 | 0.63 | 0.76 | 0.55 | 0.60 |
| + BM25 (hybrid, RRF) | 0.56 | 0.85 | 0.93 | 0.70 | 0.75 |
| + Cross-encoder rerank | **0.78** | **0.95** | **0.96** | **0.86** | **0.89** |

→ Hybrid + reranking lifts **nDCG@10 from 0.60 → 0.89 (+48%)**. Each component earns its place.

**Fine-tuning the reranker** (trained on 249 query→passage pairs + hard negatives, on Colab GPU,
eval chunks excluded so there's no leakage):

| Reranker | Hit@1 | nDCG@10 |
|---|---|---|
| Base BGE | 0.76 | 0.88 |
| **Fine-tuned** | **0.83** | **0.91** |

→ **Hit@1 76% → 83% (+8%)**, no regressions. *This is the "I trained a model and measured it" headline.*

**Honest caveat (say this in interviews):** the eval set is *synthetic* (questions auto-generated
from passages), so absolute scores are a bit optimistic — but the bias is equal across all modes, so
the **relative** comparison (reranking helps) is fair.

---

## 9. Glossary of every term

- **LLM (Large Language Model):** an AI trained on huge text that predicts the next token; powers
  ChatGPT, Llama, Gemini.
- **Token:** a chunk of text (~¾ of a word) the model reads/writes in. Models have token limits.
- **RAG (Retrieval-Augmented Generation):** retrieve relevant documents, then generate an answer
  from them. "Open-book LLM."
- **Embedding / vector:** a list of numbers representing text's *meaning*. Similar meaning → similar
  vectors.
- **Bi-encoder:** encodes query and passage *separately* (fast). Used for first-stage retrieval.
- **Cross-encoder:** reads query + passage *together* (slow, precise). Used for reranking.
- **Vector database (Qdrant):** stores vectors and finds the nearest ones fast.
- **Cosine similarity:** measures the angle between two vectors; how "close in meaning" they are.
- **BM25:** a classic keyword-ranking algorithm (term frequency + rarity). Catches exact words.
- **Hybrid retrieval:** combining semantic (vector) + keyword (BM25) search.
- **RRF (Reciprocal Rank Fusion):** a simple way to merge multiple ranked lists by rank position.
- **Reranking:** reordering retrieved candidates with a more precise model for better top results.
- **Chunking:** splitting documents into small pieces for retrieval.
- **Fine-tuning:** continuing to train a pretrained model on your own data to specialize it.
- **Hard negatives:** wrong-but-similar examples used in training so the model learns fine
  distinctions.
- **Hit@k:** fraction of questions where the correct item is in the top *k* results.
- **MRR (Mean Reciprocal Rank):** average of `1/(rank of the correct item)`. Higher = ranked higher.
- **nDCG@10:** a ranking-quality score (0–1) that rewards putting the right item higher.
- **Grounding:** answering strictly from provided sources (not the model's memory).
- **Hallucination:** the model confidently inventing false information.
- **Agentic:** the system *decides* what action to take (chat vs. search vs. code vs. web).
- **Inference:** *using* a trained model (vs. *training* it).
- **t-SNE:** an algorithm that projects high-dimensional vectors to 2D for visualization.
- **KMeans:** a clustering algorithm that groups similar points (used for topic clusters).
- **Adapter / swappable provider:** an interface that lets you swap implementations (LLM backends)
  without changing the rest of the code.
- **Idempotent:** re-running produces the same result without duplicating work.

---

## 10. Interview questions & answers

**Q: What is RAG and why use it?**
A: Retrieval-Augmented Generation — retrieve relevant documents, then generate from them. It fixes
the LLM's frozen knowledge, hallucinations, and lack of citations by grounding answers in real
sources. It gives domain expertise *without* fine-tuning the LLM.

**Q: Why hybrid retrieval instead of just vector search?**
A: Vector search captures meaning but blurs exact terms (model names, acronyms). BM25 catches exact
keywords but misses synonyms. Fusing both (via RRF) covers each other's weaknesses — I measured
nDCG@10 going from 0.60 (vector-only) to 0.75 (hybrid).

**Q: What's the difference between the embedding model and the reranker?**
A: The embedder is a bi-encoder — it encodes query and passage separately, so it's fast and used to
retrieve a wide candidate set. The reranker is a cross-encoder — it reads query+passage together, so
it's precise but slow, and only runs on the top ~30 candidates. Retrieve wide, rerank narrow.

**Q: How did you evaluate retrieval?**
A: I built an automatic benchmark of 80 questions where the correct source passage is known
(generated from passages), then measured Hit@k, MRR, and nDCG@10 across vector-only, hybrid, and
hybrid+rerank. It's a synthetic set, so I treat the *relative* gains as the trustworthy signal.

**Q: You fine-tuned a model — walk me through it.**
A: I fine-tuned the BGE cross-encoder reranker. I auto-generated 249 (question → correct passage)
pairs from the corpus, mined BM25 "hard negatives" (similar-but-wrong passages), excluded the eval
chunks to avoid leakage, and trained on Colab GPU. Re-running the benchmark, Hit@1 went 76% → 83%
with no regressions.

**Q: How do you prevent hallucinations / ensure trust?**
A: Three layers — (1) the prompt forces answers to come only from retrieved sources with `[n]`
citations; (2) a validator checks every citation maps to a real source; (3) a grounded-vs-general
decision: if the passages don't actually answer the question, it doesn't fake a grounded answer — it
labels it as general knowledge or searches the web. So it never passes off a guess as grounded.

**Q: What was the hardest problem?**
A: Realizing the reranker score measures *topical relevance*, not *answerability*. "What is an LLM"
scored high (the papers are about LLMs) but couldn't be answered (they don't define it). A
confidence threshold couldn't fix it. So I moved the judgment to the generator, which can read the
passages and decide. The lesson: measure before assuming, and pick the right signal for the job.

**Q: Why didn't you use LangChain?**
A: I built the retrieval and orchestration from scratch to understand the fundamentals and keep the
code transparent and debuggable. The agentic source chain is the kind of flow LangGraph orchestrates
— I could refactor to it, but building it myself demonstrates I understand what's happening under the
hood.

**Q: How would you deploy / scale it?**
A: Streamlit on a free host (HF Spaces / Streamlit Cloud), Qdrant as a server (not local file) for
concurrency, embeddings/indexes rebuilt on the host or fetched as artifacts, secrets as host env
vars. For scale: batch embedding, caching, a managed vector DB, and a hosted LLM endpoint.

**Q: Is it open-source or does it need paid APIs?**
A: Fully free/open-source capable — open models for embeddings/reranking, free Groq/Gemini for
generation, free DuckDuckGo for web, swappable to local Ollama for 100% offline. No paid dependency.

---

## 11. Resume bullets & how to talk about it

**Resume:**
> Built **PaperSage**, an agentic RAG system over 200 ML papers: hybrid retrieval (BM25 + dense,
> RRF) + a **fine-tuned cross-encoder reranker** (improved Hit@1 76%→83%, nDCG@10 0.89), an
> evaluation pipeline, and an agentic source chain (corpus → web → general, all cited). Added paper
> comparison, code-from-paper generation, quiz generation, an interactive research map, and a
> novelty-analysis assistant. Vendor-neutral LLM layer (Groq/Gemini/Claude/Ollama). Python,
> Streamlit, Qdrant, SQLite.

**How to tell the story (the arc that impresses):**
1. Problem → built a RAG system → **measured** it (eval table).
2. Hit a subtle bug → **measured** the cause (reranker score vs. answerability) → redesigned.
3. **Trained** a model (reranker) and **proved** it helped (+8% Hit@1).
4. Made it **trustworthy** (grounding, citations, honest fallback chain).
5. Layered on power features by reusing the clean retrieval core.

That arc — *build → measure → diagnose → improve → prove* — is exactly how real ML engineers work.

---

*You built all of this. You understand all of this. Go get the job. 🚀*
