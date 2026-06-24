"""
The Streamlit UI -- the clickable front end for your RAG system.

Run it (from the project root) with:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

# Make the project's `src` package importable when Streamlit runs this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.retrieval.search import Retriever
from src.generation.answer import generate_answer

# ----------------------------------------------------------------------------
st.set_page_config(page_title="Ask My ML Papers", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading models + indexes (first time only)...")
def load_retriever() -> Retriever:
    """Load the retriever ONCE and reuse it across questions (cached by Streamlit)."""
    return Retriever()


EXAMPLES = [
    "How can multi-agent systems improve privacy in RAG?",
    "What methods improve reasoning in large language models?",
    "How does instruction tuning affect LLM performance?",
    "What are common ways to evaluate retrieval quality?",
]

# ---------------------------------- Header ----------------------------------
st.title("📚 Ask My ML Papers")
st.caption(
    "Agentic RAG over 200 real ML research papers — hybrid retrieval (BM25 + vectors) "
    "+ cross-encoder reranking + grounded, cited answers. Fully open-source (Llama via Groq)."
)

# Keep the current question in session state so example buttons can set it.
if "question" not in st.session_state:
    st.session_state.question = ""

# ------------------------------ Example chips -------------------------------
st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state.question = ex

# ------------------------------ Question input ------------------------------
question = st.text_input(
    "Your question about ML papers:",
    value=st.session_state.question,
    placeholder="e.g. How does retrieval-augmented generation reduce hallucination?",
)

ask = st.button("🔍 Ask", type="primary")

# -------------------------------- Run + render ------------------------------
if ask and question.strip():
    retriever = load_retriever()

    with st.spinner("Searching papers, reranking, and writing a cited answer..."):
        passages = retriever.search(question)
        result = generate_answer(question, passages)

    v = result["validation"]

    # --- Answer ---
    st.subheader("Answer")
    st.markdown(result["answer"])

    # --- Trust badges ---
    if v["is_refusal"]:
        st.info("🛡️ The system found no supporting evidence in the papers and declined "
                "to answer (rather than making something up).")
    elif v["is_grounded"]:
        cited = ", ".join(f"[{n}]" for n in v["cited_sources"])
        st.success(f"✅ Grounded answer — every claim is backed by sources {cited}.")
    if v["invalid_citations"]:
        st.warning(f"⚠️ The model cited sources that don't exist: {v['invalid_citations']}")

    # --- Sources ---
    st.subheader("Sources")
    cited_set = set(v["cited_sources"])
    for s in result["sources"]:
        used = s["n"] in cited_set
        label = f"{'✅ ' if used else ''}[{s['n']}] {s['title'][:80]}  ·  arXiv:{s['arxiv_id']}"
        with st.expander(label, expanded=used):
            st.markdown(f"**Paper:** [{s['title']}]({s['url']})")
            st.markdown("**Passage the model could use:**")
            st.write(s["text"])

elif ask:
    st.warning("Please type a question first.")
