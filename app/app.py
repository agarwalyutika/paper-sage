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
from src.chat import auth
from src.chat.conversation import answer_in_conversation
from src.citations.validator import validate_citations
from src.explore.diagram import generate_mermaid

st.set_page_config(page_title="PaperSage", page_icon="📚", layout="wide")

MAP_PATH = DATA_DIR / "research_map.json"

# --------------------------------------------------------------- EDITORIAL THEME
_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Syne:wght@600;700;800&display=swap');

/* hide default Streamlit chrome */
#MainMenu, footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent;}

/* EDITORIAL typography — Inter for body, Syne for big display headlines */
html, body, [class*="css"], .stMarkdown, p, span, div, button, input, textarea, label {
  font-family: 'Inter', -apple-system, sans-serif;
}
.ps-title, .ed-title, h1, h2, h3 { font-family: 'Syne', 'Inter', sans-serif; }
a, a:visited { color: #ff5436; }

/* flat near-black 'gallery' canvas — editorial is flat and type-led, not glassy */
.stApp { background: #0c0c0d; }
section[data-testid="stSidebar"] > div {
  background: #0e0e0f; border-right: 1px solid rgba(242,239,233,0.10);
}

/* lift + cap the page width like a magazine column */
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 1.6rem; max-width: 1180px; }

/* ---------- magazine masthead (the per-view section hero) ---------- */
.ed-masthead { margin: 2px 0 24px; padding-bottom: 16px; border-bottom: 2px solid #ff5436; }
.ed-kicker { font-size: 12px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase;
  color: #ff5436; margin-bottom: 10px; }
.ed-title { font-size: 54px; font-weight: 800; line-height: 0.96; letter-spacing: -1.6px; color: #f2efe9; }
.ed-sub { font-size: 15px; color: #908d85; margin-top: 12px; max-width: 620px; line-height: 1.5; }

/* buttons — sharp, flat, editorial */
.stButton > button { border-radius: 4px; font-weight: 600;
  transition: transform .1s ease, background .12s ease, color .12s ease, border-color .12s ease; }
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button[kind="primary"], .stFormSubmitButton > button {
  background: #ff5436; border: none; color: #0c0c0d; font-weight: 700; letter-spacing: 0.2px;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover { background: #ff6a50; }
.stButton > button[kind="secondary"] {
  background: transparent; border: 1px solid rgba(242,239,233,0.18); color: #cfccc4; }
.stButton > button[kind="secondary"]:hover { border-color: #ff5436; color: #f2efe9; }

/* brand (sidebar) */
.ps-header { padding: 0; margin: 0 0 8px 0; }
.ps-title { font-size: 26px; font-weight: 800; color: #f2efe9; letter-spacing: -0.8px; }
.ps-tag   { font-size: 13px; color: #908d85; margin-top: 2px; }

/* inputs + cards — sharp corners, flat surfaces */
.stChatInput textarea, input, textarea { border-radius: 4px !important; }
[data-baseweb="select"] > div { border-radius: 4px !important; }
[data-testid="stChatMessage"] { border-radius: 6px;
  background: #131312; border: 1px solid rgba(242,239,233,0.07); }
[data-testid="stExpander"] { border-radius: 6px;
  border: 1px solid rgba(242,239,233,0.12); background: #111110; }

/* login/register card */
.ps-authcard { text-align: center; margin: 8px 0 4px; }
.ps-authcard h3 { margin-bottom: 2px; }

/* sidebar profile card — flat, editorial (square accent avatar, outline badge) */
.ps-profile { display: flex; align-items: center; gap: 11px; background: #141413;
  border: 1px solid rgba(242,239,233,0.12); border-radius: 6px; padding: 11px 12px; margin: 4px 0 6px; }
.ps-avatar { width: 38px; height: 38px; border-radius: 4px; background: #ff5436;
  color: #0c0c0d; display: flex; align-items: center; justify-content: center;
  font-family: 'Syne', sans-serif; font-weight: 800; font-size: 17px; flex: 0 0 auto; }
.ps-pname { font-weight: 700; color: #f2efe9; font-size: 14px; line-height: 1.15; }
.ps-prole { color: #908d85; font-size: 12px; }
.ps-badge { display: inline-block; background: transparent; color: #ff5436; border: 1px solid #ff5436;
  font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
  padding: 2px 9px; border-radius: 3px; margin: 0 0 6px; }

/* stepper — editorial: square tiles, accent fill when lit */
.ps-stepper { display: flex; align-items: center; gap: 12px; margin: 2px 0 20px; }
.ps-step { display: flex; align-items: center; gap: 10px; }
.ps-stepnum { width: 30px; height: 30px; border-radius: 4px; flex: 0 0 auto;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Syne', sans-serif; font-weight: 800;
  background: transparent; color: #6f6c65; border: 1px solid rgba(242,239,233,0.18); }
.ps-step.on .ps-stepnum { background: #ff5436; color: #0c0c0d; border-color: #ff5436; }
.ps-steptitle { font-weight: 700; color: #f2efe9; font-size: 13px; line-height: 1.1; }
.ps-stepsub { color: #908d85; font-size: 11px; }
.ps-stepline { flex: 1 1 auto; height: 1px; background: rgba(242,239,233,0.14); }

/* tables (Compare) — editorial: accent top rule, clean hairlines, single ink */
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%;
  border-top: 2px solid #ff5436; }
[data-testid="stMarkdownContainer"] th { color: #f2efe9; text-align: left; padding: 11px 12px;
  font-family: 'Syne', sans-serif; font-weight: 700; border-bottom: 1px solid rgba(242,239,233,0.20); }
[data-testid="stMarkdownContainer"] td { padding: 11px 12px; border-bottom: 1px solid rgba(242,239,233,0.08); }
[data-testid="stMarkdownContainer"] td:first-child { color: #908d85; font-weight: 600;
  text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px; }
[data-testid="stMarkdownContainer"] td:nth-child(n+2) { color: #ddd9d0; }

/* ---------- landing / intro page (brand-first, minimal) ---------- */
.lp-eyebrow { font-size: 12px; font-weight: 700; letter-spacing: 4px; text-transform: uppercase;
  color: #ff5436; text-align: center; margin: 64px 0 18px; }
.lp-title { font-family: 'Syne', sans-serif; font-size: 74px; font-weight: 800; line-height: 0.95;
  letter-spacing: -2.5px; color: #f2efe9; text-align: center; margin: 0; }
.lp-divider { width: 60px; height: 3px; background: #ff5436; margin: 28px auto; border: none; }
.lp-tagline { font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 600; color: #d8d4cb;
  text-align: center; margin: 0 auto 16px; }
.lp-sub { font-size: 17px; color: #908d85; text-align: center; max-width: 560px;
  margin: 0 auto 12px; line-height: 1.6; }
.lp-foot { font-size: 13px; color: #6f6c65; text-align: center; margin-top: 12px; }
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


def sidebar_profile() -> None:
    """A profile card (avatar + name + role + plan badge) at the top of the sidebar."""
    u = st.session_state.user
    initial = (u["username"][:1] or "?").upper()
    st.markdown(
        f'<div class="ps-profile"><div class="ps-avatar">{initial}</div>'
        f'<div><div class="ps-pname">{u["username"]}</div>'
        f'<div class="ps-prole">Researcher</div></div></div>'
        f'<span class="ps-badge">✦ Free Plan</span>',
        unsafe_allow_html=True,
    )


def section_hero(kicker: str, title: str, subtitle: str) -> None:
    """A magazine masthead at the top of each view: eyebrow kicker + huge title + subtitle."""
    st.markdown(
        f'<div class="ed-masthead"><div class="ed-kicker">{kicker}</div>'
        f'<div class="ed-title">{title}</div>'
        f'<div class="ed-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def stepper(steps: list[tuple[str, str]], active: int = 0) -> None:
    """A numbered stepper: steps = [(title, subtitle), ...]; everything up to `active` is lit."""
    parts = []
    for i, (title, sub) in enumerate(steps):
        on = "on" if i <= active else ""
        parts.append(
            f'<div class="ps-step {on}"><div class="ps-stepnum">{i+1}</div>'
            f'<div><div class="ps-steptitle">{title}</div>'
            f'<div class="ps-stepsub">{sub}</div></div></div>'
        )
        if i < len(steps) - 1:
            parts.append('<div class="ps-stepline"></div>')
    st.markdown('<div class="ps-stepper">' + "".join(parts) + "</div>",
                unsafe_allow_html=True)


inject_theme()


def render_landing_page() -> None:
    """A clean, brand-first intro page; the CTA leads into sign-up/login."""
    st.markdown('<div class="lp-eyebrow">Your AI research companion</div>', unsafe_allow_html=True)
    st.markdown('<div class="lp-title">📚 PaperSage</div>', unsafe_allow_html=True)
    st.markdown('<hr class="lp-divider">', unsafe_allow_html=True)
    st.markdown('<div class="lp-tagline">Wisdom, drawn from the papers.</div>', unsafe_allow_html=True)
    st.markdown('<div class="lp-sub">Ask machine-learning research papers anything, in plain English '
                '— and get clear, trustworthy answers you can actually cite.</div>',
                unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Enter  →", type="primary", use_container_width=True):
            st.session_state.show_auth = True
            st.rerun()
    st.markdown('<div class="lp-foot">Free · sign in to begin</div>', unsafe_allow_html=True)


def render_auth_page() -> None:
    """The login / register screen shown before anyone is signed in."""
    if st.button("← Back", key="auth_back"):
        st.session_state.show_auth = False
        st.rerun()
    render_header()
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown('<div class="ps-authcard"><h3>Welcome 👋</h3>'
                    '<p>Log in or create an account to start exploring ML papers.</p></div>',
                    unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["🔑  Log in", "✨  Create account"])

        with tab_login:
            with st.form("login_form"):
                login = st.text_input("Username or email")
                pwd = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in", type="primary",
                                                  use_container_width=True)
            if submitted:
                user = auth.verify_user(login, pwd)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Wrong username/email or password.")

        with tab_register:
            with st.form("register_form"):
                u = st.text_input("Username", help="At least 3 characters")
                e = st.text_input("Email")
                p = st.text_input("Password", type="password", help="At least 6 characters")
                p2 = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account", type="primary",
                                                  use_container_width=True)
            if submitted:
                if p != p2:
                    st.error("Passwords don't match.")
                else:
                    ok, msg = auth.register_user(u, e, p)
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.caption("👆 Now switch to the **Log in** tab.")

        st.caption("🔒 Your password is salted + hashed (PBKDF2-SHA256) — it's never "
                   "stored in plain text.")


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
                # Collapse newlines and strip leading markdown markers (e.g. a leading
                # "#" from a web snippet would otherwise render as a giant heading).
                snippet = " ".join(s["text"].split()).lstrip("#*->=|`~ ").strip()
                st.caption(snippet[:320] + "…")


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
def _is_image(f) -> bool:
    """True if an uploaded file is an image (→ vision path) vs a document (→ RAG)."""
    if (getattr(f, "type", "") or "").startswith("image"):
        return True
    return f.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))


def render_chat_view() -> None:
    uid = st.session_state.user["id"]
    if "session_id" not in st.session_state:
        existing = store.list_sessions(uid)
        st.session_state.session_id = (existing[0]["id"] if existing
                                       else store.create_session(uid))

    with st.sidebar:
        if st.button("➕  New chat", use_container_width=True):
            st.session_state.session_id = store.create_session(uid)
            st.rerun()
        st.caption("YOUR CHATS")
        for s in store.list_sessions(uid):
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
                    rest = store.list_sessions(uid)
                    st.session_state.session_id = (rest[0]["id"] if rest
                                                   else store.create_session(uid))
                st.rerun()

    sid = st.session_state.session_id
    section_hero("01 — Workspace", "Ask the corpus.",
                 "Grounded, cited answers drawn only from real ML research papers.")

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
        "Ask about the papers…  (📎 attach a PDF, or an image to explain)",
        accept_file="multiple", file_type=["pdf", "txt", "md", "png", "jpg", "jpeg", "webp"],
    )
    # A question can come from the chat box OR from an example chip.
    prompt = user_input.text if user_input else (pending or "")
    attached = user_input.files if user_input else None
    if not (prompt or attached):
        return

    images = [f for f in (attached or []) if _is_image(f)]
    docs = [f for f in (attached or []) if not _is_image(f)]

    # ---- Vision path: an image was attached -> explain it (multimodal) ----
    if images:
        img = images[0]
        store.add_message(sid, "user", prompt or f"[image: {img.name}]")
        current = next((x for x in store.list_sessions(uid) if x["id"] == sid), None)
        if current and current["title"] == "New chat":
            store.rename_session(sid, (prompt or f"Image: {img.name}")[:40])
        with st.chat_message("user"):
            st.image(img.getvalue(), width=300)
            if prompt:
                st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Looking at your image…"):
                from src.vision.explain import describe_image
                ans = describe_image(img.getvalue(),
                                     getattr(img, "type", "") or "image/png", prompt)
            body = "🖼️ *Answered from your uploaded image.*\n\n" + ans
            st.markdown(body)
        store.add_message(sid, "assistant", body)
        st.rerun()

    # ---- Document path: index uploaded PDFs/text for RAG ----
    if docs:
        sig = tuple((f.name, f.size) for f in docs)
        if st.session_state.get("upload_sig") != sig:
            with st.spinner("Reading and indexing your file(s)…"):
                r = load_retriever()
                idx = UploadedIndex(r.embedder, r.reranker)
                idx.build([(f.name, f.getvalue()) for f in docs])
            st.session_state.uploaded_index = idx
            st.session_state.upload_sig = sig
        st.session_state.use_uploads = True
    if not prompt:
        st.rerun()

    uidx = st.session_state.get("uploaded_index")
    use_up = bool(st.session_state.get("use_uploads") and uidx and uidx.chunks)
    store.add_message(sid, "user", prompt)
    current = next((x for x in store.list_sessions(uid) if x["id"] == sid), None)
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
    section_hero("02 — Atlas", "Research Map.",
                 "Every paper, placed by topic similarity. Click any node to open it.")
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
    section_hero("03 — Analysis", "Compare Papers.", "Side-by-side insights. Smarter decisions.")
    has_result = bool(st.session_state.get("compare_result"))
    stepper([("Add Papers", "Upload or select"),
             ("Compare", "AI analyzes & aligns"),
             ("Insights", "Explore differences")],
            active=2 if has_result else 0)

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
    section_hero("04 — Study", "Quiz & Study.",
                 "Turn any paper into MCQs, flashcards, coding or interview questions.")

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
    section_hero("05 — Frontier", "Find Novelty.",
                 "Position your idea against the literature — gaps and novel directions.")

    # Dropdown of saved analyses; the active one is tracked in session state so that
    # creating, reloading, and chatting all stay in sync.
    uid = st.session_state.user["id"]
    saved = store.list_novelty(uid)
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
            new_nid = store.save_novelty(uid, idea, res["analysis"], res["sources"])
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
# ===================================================================== IDEAS VIEW
def _render_ideas(ideas: list[dict]) -> None:
    for i, idea in enumerate(ideas, 1):
        nov = max(0, min(5, int(idea.get("novelty", 0) or 0)))
        stars = "★" * nov + "☆" * (5 - nov)
        diff = idea.get("difficulty", "—")
        st.markdown(f"### {i}. {idea.get('title', '(untitled)')}")
        st.markdown(f"**Novelty** {stars} ({nov}/5)  ·  **Difficulty:** {diff}")
        if idea.get("summary"):
            st.markdown(idea["summary"])
        if idea.get("datasets"):
            st.markdown("**Datasets:** " + ", ".join(str(d) for d in idea["datasets"]))
        if idea.get("contributions"):
            st.markdown(f"**Expected contribution:** {idea['contributions']}")
        if idea.get("roadmap"):
            with st.expander("🗺️ Implementation roadmap"):
                for j, step in enumerate(idea["roadmap"], 1):
                    st.markdown(f"{j}. {step}")
        st.markdown("---")


def render_ideas_view() -> None:
    section_hero("06 — Ideation", "Idea Lab.",
                 "Generate graduate-project ideas grounded in recent papers — each scored "
                 "for novelty and difficulty, with datasets and an implementation roadmap.")

    topic = st.text_input("Topic / area:",
                          placeholder="e.g. retrieval-augmented generation (RAG)")
    n = st.slider("How many ideas:", 3, 10, 6)

    if topic and st.button("🧪  Generate ideas", type="primary"):
        retriever = load_retriever()
        with st.spinner("Reading related papers and inventing project ideas…"):
            passages = retriever.search(topic, top_k=12)
            from src.explore.ideas import generate_ideas
            res = generate_ideas(topic, passages, n)
        st.session_state.ideas_result = res
        st.session_state.ideas_chat = []
    elif not topic:
        st.info("Enter a topic to generate project ideas.")

    res = st.session_state.get("ideas_result")
    if res:
        if res["ideas"]:
            _render_ideas(res["ideas"])
        else:
            st.warning("Couldn't parse the ideas cleanly — showing the raw output:")
            st.markdown(res["raw"])
        if res.get("sources"):
            with st.expander(f"📄 Papers these ideas drew on ({len(res['sources'])})"):
                for s in res["sources"]:
                    st.markdown(f"**[{s['n']}] {s['title'][:75]}** · "
                                f"[arXiv:{s['arxiv_id']}]({s['url']})")
        st.markdown("---")
        followup_chat("ideas", res["raw"], res.get("sources", []))


# ---- Gate: landing/about page → sign-up/login → app ----
if "user" not in st.session_state:
    if st.session_state.get("show_auth"):
        render_auth_page()
    else:
        render_landing_page()
    st.stop()

# Left panel: brand + profile card (Chat view adds New chat + history below;
# Log out is pinned at the very bottom further down).
with st.sidebar:
    st.title("📚 PaperSage")
    sidebar_profile()

# ---- Top navigation: section tabs across the top ----
NAV = ["💬 Chat", "📍 Research Map", "⚖️ Compare", "🎓 Quiz", "💡 Find Novelty", "🧪 Idea Lab"]
st.session_state.setdefault("view", NAV[0])
for col, name in zip(st.columns(len(NAV)), NAV):
    if col.button(name, key=f"nav_{name}", use_container_width=True,
                  type="primary" if st.session_state.view == name else "secondary"):
        st.session_state.view = name
        st.rerun()
st.divider()
VIEW = st.session_state.view

if VIEW == "💬 Chat":
    render_chat_view()
elif VIEW == "📍 Research Map":
    render_map_view()
elif VIEW == "⚖️ Compare":
    render_compare_view()
elif VIEW == "🎓 Quiz":
    render_quiz_view()
elif VIEW == "💡 Find Novelty":
    render_novelty_view()
else:
    render_ideas_view()

# ---- Log out: pinned to the bottom of the left panel ----
with st.sidebar:
    st.divider()
    if st.button("🚪  Log out", use_container_width=True):
        for k in ("user", "session_id", "active_novelty",
                  "compare_result", "quiz_result", "ideas_result", "view"):
            st.session_state.pop(k, None)
        st.rerun()
