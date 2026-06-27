"""
PaperSage -- a ChatGPT-style UI for ML papers, with a Research Map view.

Two views (sidebar selector):
  💬 Chat          -- conversational, multi-turn, cited answers; attach your own PDFs
  📍 Research Map  -- an interactive 2D map of all 200 papers, clustered by topic

Run it (from the project root) with:
    streamlit run app/app.py
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import streamlit.components.v1 as components

from src.config import DATA_DIR
from src.retrieval.search import Retriever
from src.retrieval.uploaded_docs import UploadedIndex
from src.chat import store
from src.chat.conversation import answer_in_conversation
from src.citations.validator import validate_citations
from src.explore.diagram import generate_mermaid

st.set_page_config(page_title="PaperSage", page_icon="📚", layout="wide")

MAP_PATH = DATA_DIR / "research_map.json"

# ----------------------------------------------------------------- VIBRANT THEME
_THEME_CSS = """
<style>
/* hide default Streamlit chrome for a cleaner, app-like feel */
#MainMenu, footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent;}

/* gradient brand header */
.ps-header {
  background: linear-gradient(100deg, #a855f7 0%, #6366f1 48%, #22d3ee 100%);
  padding: 20px 26px; border-radius: 16px; margin: 2px 0 16px 0;
  box-shadow: 0 10px 30px rgba(124, 108, 255, 0.28);
}
.ps-title { font-size: 30px; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
.ps-tag   { font-size: 14px; color: rgba(255,255,255,0.92); margin-top: 3px; }

/* sidebar nav: turn the radio into pill buttons, gradient the active one */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  background: #16182e; border: 1px solid #2a2d4a; border-radius: 10px;
  padding: 9px 12px; margin-bottom: 7px; transition: all 0.15s ease; width: 100%;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { border-color: #a855f7; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(100deg, #a855f7, #6366f1);
  border-color: transparent; box-shadow: 0 4px 14px rgba(124,108,255,0.35);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color: #fff; font-weight: 600; }
section[data-testid="stSidebar"] div[role="radiogroup"] input { display: none; }  /* hide the dot */

/* primary buttons: gradient fill */
.stButton > button[kind="primary"] {
  background: linear-gradient(100deg, #a855f7, #6366f1);
  border: none; font-weight: 600; transition: filter 0.15s ease;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.1); }

/* example-question chips */
.stButton > button[kind="secondary"] { border-radius: 10px; border-color: #2a2d4a; }

/* chat bubbles + cards: softer, rounded */
[data-testid="stChatMessage"] { border-radius: 14px; }
[data-testid="stExpander"] { border-radius: 12px; border-color: #2a2d4a; }
.stChatInput textarea { border-radius: 12px; }
</style>
"""


def inject_theme() -> None:
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        '<div class="ps-header">'
        '<div class="ps-title">📚 PaperSage</div>'
        '<div class="ps-tag">Cited answers from real ML research papers · '
        'hybrid retrieval + reranking + grounded generation</div>'
        '</div>',
        unsafe_allow_html=True,
    )


inject_theme()


@st.cache_resource(show_spinner="Loading models + indexes (first time only)...")
def load_retriever() -> Retriever:
    return Retriever()


@st.cache_data
def load_map() -> list[dict]:
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_meta() -> dict:
    return {p["arxiv_id"]: p
            for p in json.loads((DATA_DIR / "papers_meta.json").read_text(encoding="utf-8"))}


def render_mermaid(code: str) -> None:
    """Render a Mermaid diagram inline using mermaid.js (from a CDN, client-side)."""
    html = f"""
    <div class="mermaid" style="background:#0e1117;color:#fafafa">{code}</div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true, theme:"dark", securityLevel:"loose"}});</script>
    """
    components.html(html, height=480, scrolling=True)


def diagram_widget(answer: str, key: str) -> None:
    """A 'Show concept diagram' button that generates + renders a Mermaid diagram on demand."""
    h = hashlib.md5(answer.encode("utf-8")).hexdigest()
    cache = st.session_state.setdefault("diagrams", {})
    if st.button("🗺️ Show concept diagram", key=key):
        with st.spinner("Drawing diagram…"):
            try:
                cache[h] = generate_mermaid(answer)
            except Exception as e:
                cache[h] = None
                st.warning(f"Couldn't generate a diagram right now: {e}")
    if cache.get(h):
        render_mermaid(cache[h])


def render_sources(answer_text: str, sources: list[dict]) -> None:
    is_web = bool(sources) and sources[0].get("kind") == "web"
    v = validate_citations(answer_text, len(sources))
    if is_web:
        st.info("🌐 Answered from the **web** — your paper corpus didn't cover this. "
                "Sources are linked below.")
    elif v["is_refusal"]:
        st.info("🛡️ No supporting evidence found — declined to answer rather than guessing.")
    elif v["is_grounded"]:
        cited = ", ".join(f"[{n}]" for n in v["cited_sources"])
        st.success(f"✅ Grounded — every claim is backed by sources {cited}.")
    if v["invalid_citations"]:
        st.warning(f"⚠️ Model cited sources that don't exist: {v['invalid_citations']}")
    if not sources:
        return
    cited_set = set(v["cited_sources"])
    label = "🌐 Web sources" if is_web else "📄 Sources"
    with st.expander(f"{label} ({len(sources)})"):
        for s in sources:
            used = s["n"] in cited_set
            head = f"{'✅ ' if used else ''}**[{s['n']}] {s['title'][:75]}**"
            if s.get("kind") == "web":
                st.markdown(f"{head} · 🌐 [{s['arxiv_id']}]({s['url']})")
            elif s.get("url"):
                st.markdown(f"{head} · [arXiv:{s['arxiv_id']}]({s['url']})")
            else:
                st.markdown(f"{head} · {s['arxiv_id']}")
            if s.get("text"):
                st.caption(s["text"][:320].strip() + "…")


def followup_chat(key: str, context: str, sources: list[dict]) -> None:
    """Reusable follow-up chat for ANY feature, grounded in `context` + `sources`.
    `key` namespaces this chat in session state (so Compare and Quiz don't collide)."""
    msgs = st.session_state.setdefault(f"{key}_chat", [])
    st.markdown("#### 💬 Ask a follow-up")
    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Ask about this…", key=f"{key}_input")
    if q:
        msgs.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                from src.explore.discuss import discuss
                ans = discuss(context, sources, msgs[:-1], q)
            st.markdown(ans)
        msgs.append({"role": "assistant", "content": ans})
        st.rerun()


# =================================================================== CHAT VIEW
def render_chat_view() -> None:
    if "session_id" not in st.session_state:
        existing = store.list_sessions()
        st.session_state.session_id = (existing[0]["id"] if existing
                                       else store.create_session())

    with st.sidebar:
        if st.button("➕  New chat", use_container_width=True):
            st.session_state.session_id = store.create_session()
            st.rerun()
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
                    rest = store.list_sessions()
                    st.session_state.session_id = (rest[0]["id"] if rest
                                                   else store.create_session())
                st.rerun()

    sid = st.session_state.session_id
    st.caption("Agentic RAG over 200 ML papers — hybrid retrieval + reranking + "
               "grounded, cited answers. Fully open-source (Llama via Groq).")

    _uidx = st.session_state.get("uploaded_index")
    if _uidx is not None and _uidx.chunks:
        files_label = ", ".join(sorted({c["title"] for c in _uidx.chunks}))[:60]
        use_uploads = st.toggle(f"📎 Answer from my uploaded file(s): {files_label}",
                                value=st.session_state.get("use_uploads", True),
                                key="use_uploads")
    else:
        use_uploads = False
    st.caption(f"🔎 Answering from: "
               f"**{'📎 your uploaded files' if use_uploads else '📚 the ML paper corpus'}**")

    history = store.get_messages(sid)
    pending = None                       # set if the user clicks an example chip
    if not history:
        st.info("👋 Ask me anything about the ML papers — or try an example:")
        examples = [
            "How can multi-agent systems improve RAG privacy?",
            "Bi-encoder vs cross-encoder — what's the difference?",
            "How does LoRA fine-tuning work?",
        ]
        for c, ex in zip(st.columns(len(examples)), examples):
            if c.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
                pending = ex
    for i, m in enumerate(history):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m["sources"]:
                render_sources(m["content"], m["sources"])
                diagram_widget(m["content"], key=f"dia_{sid}_{i}")

    user_input = st.chat_input(
        "Ask about the papers…  (📎 attach a PDF to ask about your own file)",
        accept_file="multiple", file_type=["pdf", "txt", "md"],
    )
    # A question can come from the chat box OR from an example chip.
    prompt = user_input.text if user_input else (pending or "")
    attached = user_input.files if user_input else None
    if not (prompt or attached):
        return

    if attached:
        sig = tuple((f.name, f.size) for f in attached)
        if st.session_state.get("upload_sig") != sig:
            with st.spinner("Reading and indexing your file(s)…"):
                r = load_retriever()
                idx = UploadedIndex(r.embedder, r.reranker)
                idx.build([(f.name, f.getvalue()) for f in attached])
            st.session_state.uploaded_index = idx
            st.session_state.upload_sig = sig
        st.session_state.use_uploads = True
    if not prompt:
        st.rerun()

    uidx = st.session_state.get("uploaded_index")
    use_up = bool(st.session_state.get("use_uploads") and uidx and uidx.chunks)
    store.add_message(sid, "user", prompt)
    current = next((x for x in store.list_sessions() if x["id"] == sid), None)
    if current and current["title"] == "New chat":
        store.rename_session(sid, prompt[:40])

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            retriever = load_retriever()
            search_fn = uidx.search if use_up else retriever.search
            result = answer_in_conversation(history, prompt, search_fn)
        # Make sure general-knowledge answers are always clearly labeled.
        answer_text = result["answer"]
        if (result.get("mode") == "general"
                and "general knowledge" not in answer_text[:40].lower()):
            answer_text = ("ℹ️ *General knowledge — not grounded in your paper "
                           "corpus.*\n\n" + answer_text)
        st.markdown(answer_text)
        if result["sources"]:        # grounded (papers) OR web answer
            render_sources(result["answer"], result["sources"])
    store.add_message(sid, "assistant", answer_text, sources=result["sources"])
    st.rerun()


# =================================================================== MAP VIEW
def render_map_view() -> None:
    st.subheader("📍 Research Map")
    if not MAP_PATH.exists():
        st.warning("The research map hasn't been built yet. Run:\n\n"
                   "```\npython -m src.explore.build_map\n```")
        return

    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(load_map())
    st.caption(f"All **{len(df)} papers** placed by topic similarity — papers near each "
               "other are about similar things. **Click any dot** to open its paper. "
               "**Hover** for the title; **click a topic in the legend** to toggle it.")

    fig = px.scatter(
        df, x="x", y="y", color="topic", hover_name="title",
        custom_data=["arxiv_id", "url", "topic"],   # carried into click events
        hover_data={"x": False, "y": False, "topic": False},
        template="plotly_dark", height=620,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="#111")))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(legend_title_text="Topic", margin=dict(l=0, r=0, t=10, b=0))

    # on_select="rerun" makes clicks come back to us as a selection event.
    event = st.plotly_chart(fig, use_container_width=True, key="map",
                            on_select="rerun", selection_mode="points")

    # Figure out which paper was clicked (if any).
    clicked_id = None
    try:
        pts = event["selection"]["points"]
        if pts:
            clicked_id = pts[0]["customdata"][0]   # arxiv_id
    except (KeyError, TypeError, IndexError):
        clicked_id = None

    # Fall back to a dropdown if nothing is clicked yet.
    st.markdown("---")
    if clicked_id:
        target_id = clicked_id
    else:
        pick = st.selectbox("🔍 …or find a paper by name:",
                            ["—"] + sorted(df["title"].tolist()))
        target_id = (df[df["title"] == pick].iloc[0]["arxiv_id"]
                     if pick != "—" else None)

    if target_id:
        row = df[df["arxiv_id"] == target_id].iloc[0]
        meta = load_meta().get(target_id, {})
        st.markdown(f"### {row['title']}")
        st.caption(f"topic: *{row['topic']}*  ·  arXiv:{target_id}")
        st.link_button("📄  Open paper on arXiv  ↗", row["url"])
        if meta.get("abstract"):
            st.caption(meta["abstract"][:700] + "…")
    else:
        st.caption("👆 Click any dot on the map to open its paper.")


# ================================================================== COMPARE VIEW
def render_compare_view() -> None:
    st.subheader("⚖️ Compare Papers")
    st.caption("Pick papers from the corpus **and/or upload your own PDFs**, then compare. "
               "PaperSage reads each one and synthesizes a detailed side-by-side table "
               "(problem, method, novelty, dataset, results, strengths, limitations).")

    meta = load_meta()                              # {arxiv_id: paper}
    title_to_id = {p["title"]: aid for aid, p in meta.items()}
    picked = st.multiselect("Corpus papers:", sorted(title_to_id), max_selections=4)
    uploads = st.file_uploader("…or upload your own PDFs to include:",
                               type=["pdf"], accept_multiple_files=True)

    total = len(picked) + (len(uploads) if uploads else 0)
    if total > 4:
        st.warning("Comparing up to 4 papers works best — using the first 4.")

    if total >= 2 and st.button("⚖️  Compare", type="primary"):
        from src.explore.compare import build_corpus_papers, upload_context, compare
        with st.spinner("Reading the papers and building a detailed comparison…"):
            papers = build_corpus_papers([title_to_id[t] for t in picked])
            for f in (uploads or []):
                papers.append({"title": f.name,
                               "context": upload_context(f.name, f.getvalue()),
                               "url": ""})
            res = compare(papers[:4])
        st.session_state.compare_result = res
        st.session_state.compare_chat = []          # fresh chat for a new comparison
    elif total < 2:
        st.info("Select and/or upload at least 2 papers total to compare.")

    # Render the (stored) comparison + a follow-up chat about it.
    res = st.session_state.get("compare_result")
    if res:
        st.markdown(res["table"])
        st.markdown("**Papers compared:**")
        for p in res["papers"]:
            if p.get("url"):
                st.markdown(f"- [{p['title']}]({p['url']})")
            else:
                st.markdown(f"- {p['title']}  *(your upload)*")
        st.markdown("---")
        followup_chat("compare", res["table"], res["papers"])


# ===================================================================== QUIZ VIEW
def _render_quiz_items(qtype: str, items: list[dict]) -> None:
    if qtype == "MCQs":
        for i, q in enumerate(items, 1):
            st.markdown(f"**Q{i}. {q.get('question', '')}**")
            opts = q.get("options", [])
            for j, o in enumerate(opts):
                st.markdown(f"&nbsp;&nbsp;{'ABCD'[j] if j < 4 else j}. {o}")
            with st.expander("Show answer"):
                ans = q.get("answer", 0)
                if isinstance(ans, int) and 0 <= ans < len(opts):
                    st.success(f"Correct: {'ABCD'[ans]}. {opts[ans]}")
                if q.get("explanation"):
                    st.caption(q["explanation"])
    elif qtype == "Flashcards":
        for c in items:
            with st.expander(f"🃏  {c.get('front', '')}"):
                st.write(c.get("back", ""))
    elif qtype == "Coding questions":
        for i, q in enumerate(items, 1):
            st.markdown(f"**{i}. {q.get('question', '')}**")
            if q.get("hint"):
                with st.expander("Hint"):
                    st.caption(q["hint"])
    else:  # Interview questions
        for i, q in enumerate(items, 1):
            st.markdown(f"**Q{i}. {q.get('question', '')}**")
            with st.expander("Model answer"):
                st.write(q.get("answer", ""))


def render_quiz_view() -> None:
    st.subheader("🎓 Quiz & Study")
    st.caption("Turn any paper into study material. Pick a corpus paper or upload your own, "
               "choose a format, and generate.")

    meta = load_meta()
    title_to_id = {p["title"]: aid for aid, p in meta.items()}

    src = st.radio("Paper source:", ["Corpus paper", "Upload a PDF"], horizontal=True)
    context, paper_name = None, None
    if src == "Corpus paper":
        pick = st.selectbox("Pick a paper:", ["—"] + sorted(title_to_id))
        if pick != "—":
            from src.explore.compare import build_corpus_papers
            context = build_corpus_papers([title_to_id[pick]])[0]["context"]
            paper_name = pick
    else:
        up = st.file_uploader("Upload a PDF:", type=["pdf"])
        if up:
            from src.explore.compare import upload_context
            context = upload_context(up.name, up.getvalue())
            paper_name = up.name

    from src.explore.quiz import QUIZ_TYPES, generate_quiz
    c1, c2 = st.columns([0.6, 0.4])
    qtype = c1.radio("Generate:", QUIZ_TYPES, horizontal=True)
    n = c2.slider("How many:", 3, 8, 5)

    if not context:
        st.info("Pick or upload a paper first.")
        return
    if st.button("🎓  Generate", type="primary"):
        with st.spinner(f"Generating {qtype} from “{paper_name[:50]}”…"):
            res = generate_quiz(context, qtype, n)
        st.session_state.quiz_result = {"res": res, "qtype": qtype,
                                        "paper": paper_name, "context": context}
        st.session_state.quiz_chat = []

    qr = st.session_state.get("quiz_result")
    if qr:
        res = qr["res"]
        if res["items"]:
            _render_quiz_items(qr["qtype"], res["items"])
        else:
            st.warning("Couldn't parse the output cleanly — showing it as text:")
            st.markdown(res["raw"])
        st.markdown("---")
        followup_chat("quiz", res["raw"], [{"title": qr["paper"], "context": qr["context"]}])


# ================================================================== NOVELTY VIEW
def render_novelty_view() -> None:
    st.subheader("💡 Find Novelty")
    st.caption("Describe a research idea — PaperSage finds related work, gaps, and novel directions, "
               "then lets you **discuss it** in a chat. Analyses and their chats are saved.")

    # Dropdown of saved analyses; the active one is tracked in session state so that
    # creating, reloading, and chatting all stay in sync.
    saved = store.list_novelty()
    options = {"➕ New analysis": None}
    for s in saved:
        options[f"{s['idea'][:55]}  ·  {s['created_at'][:10]}"] = s["id"]
    keys = list(options.keys())
    active = st.session_state.get("active_novelty")
    default_idx = next((i for i, k in enumerate(keys) if options[k] == active), 0)
    choice = st.selectbox("📚 Your analyses:", keys, index=default_idx)
    nid = options[choice]
    st.session_state.active_novelty = nid

    # ---- New analysis ----
    if nid is None:
        idea = st.text_area("Your research idea:", height=100,
                            placeholder="e.g. detect diabetic retinopathy using Vision Transformers")
        if st.button("💡  Analyze novelty", type="primary"):
            if not idea.strip():
                st.warning("Describe your idea first.")
                return
            retriever = load_retriever()
            with st.spinner("Searching related work and analyzing novelty…"):
                passages = retriever.search(idea, top_k=10)
                from src.explore.novelty import find_novelty
                res = find_novelty(idea, passages)
            new_nid = store.save_novelty(idea, res["analysis"], res["sources"])
            st.session_state.active_novelty = new_nid          # jump to it (with chat)
            st.rerun()
        return

    # ---- Show a saved analysis + a follow-up chat about it ----
    data = store.get_novelty(nid)
    if data is None:                       # stale selection -> reset
        st.session_state.active_novelty = None
        st.rerun()
    st.markdown(f"**💡 Idea:** {data['idea']}")
    st.markdown(data["analysis"])
    if data["sources"]:
        render_sources(data["analysis"], data["sources"])
    if st.button("🗑  Delete this analysis"):
        store.delete_novelty(nid)
        st.session_state.active_novelty = None
        st.rerun()

    st.markdown("#### 💬 Ask about this analysis")
    msgs = store.get_novelty_messages(nid)
    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask a follow-up… e.g. \"explain gap 2\" or \"how would I do novel idea 3?\"")
    if q:
        store.add_novelty_message(nid, "user", q)
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                from src.explore.novelty import discuss_novelty
                ans = discuss_novelty(data["idea"], data["sources"], data["analysis"], msgs, q)
            st.markdown(ans)
        store.add_novelty_message(nid, "assistant", ans)
        st.rerun()


# ===================================================================== DISPATCH
with st.sidebar:
    st.title("📚 PaperSage")
    st.caption("Agentic RAG over ML papers")
    VIEW = st.radio("View", ["💬 Chat", "📍 Research Map", "⚖️ Compare", "🎓 Quiz", "💡 Find Novelty"],
                    label_visibility="collapsed")
    st.divider()

render_header()

if VIEW == "💬 Chat":
    render_chat_view()
elif VIEW == "📍 Research Map":
    render_map_view()
elif VIEW == "⚖️ Compare":
    render_compare_view()
elif VIEW == "🎓 Quiz":
    render_quiz_view()
else:
    render_novelty_view()
