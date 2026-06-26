"""
PaperSage -- a ChatGPT-style UI for asking questions about ML research papers.

Features:
  - multiple chat sessions (sidebar) with a "New chat" button
  - persistent history (your chats are saved and reload when you return)
  - multi-turn conversation (follow-up questions work)
  - grounded, cited answers with a trust badge + expandable sources

Run it (from the project root) with:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.retrieval.search import Retriever
from src.retrieval.uploaded_docs import UploadedIndex
from src.chat import store
from src.chat.conversation import answer_in_conversation
from src.citations.validator import validate_citations

st.set_page_config(page_title="PaperSage", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading models + indexes (first time only)...")
def load_retriever() -> Retriever:
    """Load the retriever ONCE and reuse it across questions (cached by Streamlit)."""
    return Retriever()


def render_sources(answer_text: str, sources: list[dict]) -> None:
    """Show the trust badge + expandable source cards under an assistant message."""
    v = validate_citations(answer_text, len(sources))

    if v["is_refusal"]:
        st.info("🛡️ No supporting evidence found — declined to answer rather than guessing.")
    elif v["is_grounded"]:
        cited = ", ".join(f"[{n}]" for n in v["cited_sources"])
        st.success(f"✅ Grounded — every claim is backed by sources {cited}.")
    if v["invalid_citations"]:
        st.warning(f"⚠️ Model cited sources that don't exist: {v['invalid_citations']}")

    if not sources:
        return
    cited_set = set(v["cited_sources"])
    with st.expander(f"📄 Sources ({len(sources)})"):
        for s in sources:
            used = s["n"] in cited_set
            head = f"{'✅ ' if used else ''}**[{s['n']}] {s['title'][:75]}**"
            if s.get("url"):
                st.markdown(f"{head} · [arXiv:{s['arxiv_id']}]({s['url']})")
            else:
                st.markdown(f"{head} · {s['arxiv_id']}")
            st.caption(s["text"][:320].strip() + "…")


# ------------------------------------------------------------------ session
# Make sure there's an active chat selected.
if "session_id" not in st.session_state:
    existing = store.list_sessions()
    st.session_state.session_id = existing[0]["id"] if existing else store.create_session()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📚 PaperSage")
    if st.button("➕  New chat", use_container_width=True):
        st.session_state.session_id = store.create_session()
        st.rerun()

    st.divider()
    st.caption("YOUR CHATS")
    for s in store.list_sessions():
        c1, c2 = st.columns([0.82, 0.18])
        is_active = s["id"] == st.session_state.session_id
        label = s["title"][:26] + ("…" if len(s["title"]) > 26 else "")
        if c1.button(label, key=f"sel_{s['id']}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.session_id = s["id"]
            st.rerun()
        if c2.button("🗑", key=f"del_{s['id']}", help="Delete this chat"):
            store.delete_session(s["id"])
            if is_active:
                remaining = store.list_sessions()
                st.session_state.session_id = (
                    remaining[0]["id"] if remaining else store.create_session()
                )
            st.rerun()

sid = st.session_state.session_id

# ------------------------------------------------------------------ main pane
st.caption(
    "Agentic RAG over 200 ML research papers — hybrid retrieval + reranking + "
    "grounded, cited answers. Fully open-source (Llama via Groq)."
)

# Active knowledge source: uploaded files (if you attached any) or the corpus.
_uidx = st.session_state.get("uploaded_index")
if _uidx is not None and _uidx.chunks:
    files_label = ", ".join(sorted({c["title"] for c in _uidx.chunks}))[:60]
    USE_UPLOADS = st.toggle(
        f"📎 Answer from my uploaded file(s): {files_label}",
        value=st.session_state.get("use_uploads", True),
        key="use_uploads",
    )
else:
    USE_UPLOADS = False
st.caption(f"🔎 Answering from: **{'📎 your uploaded files' if USE_UPLOADS else '📚 the ML paper corpus'}**")

# Replay the saved conversation.
history = store.get_messages(sid)
if not history:
    st.info("👋 Ask me anything about the ML paper corpus — e.g. "
            "*\"How can multi-agent systems improve RAG privacy?\"*")
for m in history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        # Only research answers carry sources; chat replies (greetings) don't.
        if m["role"] == "assistant" and m["sources"]:
            render_sources(m["content"], m["sources"])

# Chat input with a 📎 attach button built right into the box (ChatGPT-style).
user_input = st.chat_input(
    "Ask about the papers…  (📎 attach a PDF to ask about your own file)",
    accept_file="multiple",
    file_type=["pdf", "txt", "md"],
)

if user_input:
    prompt = user_input.text
    attached = user_input.files

    # If file(s) were attached, build an in-memory index for this conversation.
    if attached:
        sig = tuple((f.name, f.size) for f in attached)
        if st.session_state.get("upload_sig") != sig:
            with st.spinner("Reading and indexing your file(s)…"):
                r = load_retriever()
                idx = UploadedIndex(r.embedder, r.reranker)
                idx.build([(f.name, f.getvalue()) for f in attached])
            st.session_state.uploaded_index = idx
            st.session_state.upload_sig = sig
        st.session_state.use_uploads = True   # default to searching the new file

    # Attached a file but didn't type a question yet -> just index and wait.
    if not prompt:
        st.rerun()

    # Decide the source for THIS question.
    uidx = st.session_state.get("uploaded_index")
    use_uploads = bool(st.session_state.get("use_uploads") and uidx and uidx.chunks)

    store.add_message(sid, "user", prompt)
    current = next((x for x in store.list_sessions() if x["id"] == sid), None)
    if current and current["title"] == "New chat":
        store.rename_session(sid, prompt[:40])

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            retriever = load_retriever()
            search_fn = uidx.search if use_uploads else retriever.search
            result = answer_in_conversation(history, prompt, search_fn)
        st.markdown(result["answer"])
        if result.get("mode") == "search":
            render_sources(result["answer"], result["sources"])

    store.add_message(sid, "assistant", result["answer"], sources=result["sources"])
    st.rerun()
