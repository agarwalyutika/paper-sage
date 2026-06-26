"""
Central configuration for the whole project.

Everything tunable lives here so we never hard-code values deep inside the code.
Import it anywhere with:  from src.config import settings
"""
from pathlib import Path
from dotenv import load_dotenv
import os

# Load secrets from the .env file (e.g. your ANTHROPIC_API_KEY) into the environment.
load_dotenv()

# --- Where things live on disk ---
# PROJECT_ROOT = the agentic-rag-ml-papers folder (two levels up from this file).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"        # downloaded PDFs go here
QDRANT_DIR = DATA_DIR / "qdrant"        # local vector DB files go here
CHUNKS_PATH = DATA_DIR / "chunks.json"  # all chunked text + metadata
BM25_PATH = DATA_DIR / "bm25.pkl"       # saved keyword-search index

# Make sure the folders exist (created automatically on first run).
for _d in (DATA_DIR, PAPERS_DIR, QDRANT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings:
    # --- arXiv ingestion ---
    # Tightly focused on LLM / RAG / agents.  cs.CL = NLP/language models, cs.AI = AI.
    # (We dropped cs.LG on purpose -- it's broad ML and pulled in off-topic papers.)
    ARXIV_CATEGORIES = ["cs.CL", "cs.AI"]
    ARXIV_QUERY = (
        "large language models OR retrieval augmented generation OR RAG "
        "OR LLM agents OR in-context learning OR instruction tuning "
        "OR prompt engineering OR chain of thought OR LLM reasoning"
    )
    MAX_PAPERS = 350         # candidates to request; relevance filter trims this down
    TARGET_PAPERS = 200      # stop once we have this many focused papers

    # Relevance filter: a paper is kept only if its title or abstract contains one of
    # these keywords (text is lowercased and hyphens are turned into spaces first,
    # so "Large-Language-Model" still matches "large language model").
    RELEVANCE_KEYWORDS = [
        "llm", "large language model", "language model", "rag",
        "retrieval augmented", "agent", "in context", "instruction tun",
        "prompt", "chain of thought", "fine tun", "transformer", "gpt",
        "reasoning", "question answering", "dialogue", "text generation",
        "few shot", "zero shot", "tokeniz",
    ]

    # --- Chunking (how we slice each paper) ---
    CHUNK_SIZE = 1200        # characters per chunk (~250-300 words)
    CHUNK_OVERLAP = 200      # overlap so a sentence isn't cut awkwardly between chunks

    # --- Models (small on purpose, so CPU query-time stays fast) ---
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"      # text -> vector
    RERANKER_MODEL = "BAAI/bge-reranker-base"       # re-scores candidates for precision
    # Automatically use the fine-tuned reranker if it has been trained + placed here.
    if (PROJECT_ROOT / "models" / "bge-reranker-base-ft").exists():
        RERANKER_MODEL = str(PROJECT_ROOT / "models" / "bge-reranker-base-ft")

    # --- Vector DB ---
    QDRANT_COLLECTION = "ml_papers"
    EMBEDDING_DIM = 384      # bge-small outputs 384-number vectors

    # --- Retrieval knobs ---
    TOP_K_VECTOR = 20        # how many candidates vector search returns
    TOP_K_BM25 = 20          # how many candidates keyword search returns
    TOP_K_RERANK = 6         # final passages we keep after reranking -> sent to the LLM

    # --- Generation (the answer-writer) ---
    # Which backend writes the answer. "groq" = free hosted Llama (default).
    # Swappable: set to "claude" or "ollama" later with no other code changes.
    LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")

    # Groq (free, hosted open-source Llama)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = "llama-3.3-70b-versatile"        # strong primary model
    # If the primary model hits its daily rate limit, fall back to this faster,
    # higher-quota model so the app keeps working.
    GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"

    # Claude (optional, paid) -- used only if LLM_BACKEND == "claude"
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL = "claude-opus-4-8"

    # Ollama (optional, fully local) -- used only if LLM_BACKEND == "ollama"
    OLLAMA_MODEL = "llama3.2:3b"

    # Gemini (free tier, strong, generous limits) -- used if LLM_BACKEND == "gemini"
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = "gemini-2.0-flash"

    # How creative the answer-writer is. Low = factual & grounded (what we want).
    LLM_TEMPERATURE = 0.2
    # Hard cap on answer length -> prevents runaway repetition loops.
    LLM_MAX_TOKENS = 700
    # Penalize repeated tokens a bit, so weaker models don't get stuck looping.
    LLM_FREQUENCY_PENALTY = 0.3

    # --- Web search fallback (used when the papers don't cover the question) ---
    WEB_SEARCH_ENABLED = True
    # "duckduckgo" = free, no key (default). "tavily" = free key, more reliable content.
    WEB_SEARCH_BACKEND = os.getenv("WEB_SEARCH_BACKEND", "duckduckgo")
    WEB_SEARCH_K = 5
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


settings = Settings()
