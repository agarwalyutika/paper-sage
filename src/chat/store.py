"""
Persistent chat storage -- so conversations survive closing the app.

We use SQLite (a small file-based database at data/chats.db). Two tables:
  - sessions: one row per conversation (id, title, timestamps)
  - messages: every message in every conversation (role, content, sources)

This is what makes "come back later and your chats are still here" work.
Everything here is plain, synchronous SQLite -- no server needed.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone

from src.config import DATA_DIR

DB_PATH = DATA_DIR / "chats.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    return conn


def init_db() -> None:
    """Create the tables if they don't exist yet (safe to call every startup)."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                source_mode TEXT NOT NULL DEFAULT 'corpus',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,          -- 'user' or 'assistant'
                content     TEXT NOT NULL,
                sources     TEXT,                   -- JSON list of source dicts (or NULL)
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS novelty (
                id          TEXT PRIMARY KEY,
                idea        TEXT NOT NULL,
                analysis    TEXT NOT NULL,
                sources     TEXT,                   -- JSON list of related-paper dicts
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS novelty_messages (
                id          TEXT PRIMARY KEY,
                novelty_id  TEXT NOT NULL,          -- which analysis this follow-up belongs to
                role        TEXT NOT NULL,          -- 'user' or 'assistant'
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            """
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------- Sessions ----------------------------
def create_session(title: str = "New chat", source_mode: str = "corpus") -> str:
    """Start a new conversation; returns its id."""
    sid = str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, source_mode, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, title, source_mode, now, now),
        )
    return sid


def list_sessions() -> list[dict]:
    """All conversations, most recently used first (for the sidebar)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, source_mode, updated_at FROM sessions "
            "ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def rename_session(session_id: str, title: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))


def delete_session(session_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def _touch(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))


# ---------------------------- Messages ----------------------------
def add_message(session_id: str, role: str, content: str,
                sources: list[dict] | None = None) -> None:
    """Append one message to a conversation and bump its 'last used' time."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, sources, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), session_id, role, content,
             json.dumps(sources) if sources is not None else None, _now()),
        )
        _touch(conn, session_id)


def get_messages(session_id: str) -> list[dict]:
    """All messages in a conversation, oldest first (to replay the thread)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, sources FROM messages "
            "WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "role": r["role"],
            "content": r["content"],
            "sources": json.loads(r["sources"]) if r["sources"] else [],
        })
    return out


# ---------------------------- Novelty analyses ----------------------------
def save_novelty(idea: str, analysis: str, sources: list[dict] | None = None) -> str:
    """Persist a Find-Novelty analysis so it can be reloaded later."""
    nid = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO novelty (id, idea, analysis, sources, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (nid, idea, analysis,
             json.dumps(sources) if sources is not None else None, _now()),
        )
    return nid


def list_novelty() -> list[dict]:
    """Saved analyses, newest first (for the dropdown)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, idea, created_at FROM novelty ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_novelty(nid: str) -> dict | None:
    with _connect() as conn:
        r = conn.execute(
            "SELECT idea, analysis, sources FROM novelty WHERE id = ?", (nid,)
        ).fetchone()
    if not r:
        return None
    return {"idea": r["idea"], "analysis": r["analysis"],
            "sources": json.loads(r["sources"]) if r["sources"] else []}


def delete_novelty(nid: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM novelty_messages WHERE novelty_id = ?", (nid,))
        conn.execute("DELETE FROM novelty WHERE id = ?", (nid,))


def add_novelty_message(novelty_id: str, role: str, content: str) -> None:
    """Append a follow-up message to a novelty analysis's conversation."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO novelty_messages (id, novelty_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), novelty_id, role, content, _now()),
        )


def get_novelty_messages(novelty_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM novelty_messages "
            "WHERE novelty_id = ? ORDER BY created_at ASC",
            (novelty_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# Make sure the database + tables exist as soon as this module is imported.
init_db()
