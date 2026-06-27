"""
User accounts + secure password storage for PaperSage.

We never store a password. We store a per-user random *salt* and the PBKDF2-SHA256
*hash* of (password + salt). At login we recompute the hash and compare in constant
time. This is the standard, safe way to store passwords -- and it uses only Python's
standard library (hashlib/hmac/os), so there's no extra dependency.

The users live in the same SQLite file as the chats (data/chats.db).
"""
import hashlib
import hmac
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from src.config import DATA_DIR

DB_PATH = DATA_DIR / "chats.db"
_ITERATIONS = 200_000          # PBKDF2 work factor (higher = slower to brute-force)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_auth_db() -> None:
    """Create the users table if it doesn't exist (safe to call on every startup)."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id         TEXT PRIMARY KEY,
                username   TEXT NOT NULL UNIQUE,
                email      TEXT NOT NULL UNIQUE,
                pwd_hash   TEXT NOT NULL,        -- 'salt_hex$hash_hex'
                created_at TEXT NOT NULL
            )
            """
        )


def _pbkdf2(password: str, salt: bytes) -> str:
    """Derive the hash of a password with a given salt (hex string)."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return dk.hex()


def _make_hash(password: str) -> str:
    """Make a fresh salt and return the storable 'salt$hash' string."""
    salt = os.urandom(16)
    return f"{salt.hex()}${_pbkdf2(password, salt)}"


def _check(password: str, stored: str) -> bool:
    """Verify a password against a stored 'salt$hash' (constant-time compare)."""
    try:
        salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    calc = _pbkdf2(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(calc, hash_hex)


def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """Create a new account. Returns (ok, message). Validates input + uniqueness."""
    username, email = username.strip(), email.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if "@" not in email or "." not in email:
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (id, username, email, pwd_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), username, email, _make_hash(password), _now()),
            )
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."
    return True, "Account created! You can log in now."


def verify_user(login: str, password: str) -> dict | None:
    """Check a login (username OR email) + password. Returns the user dict or None."""
    login = login.strip().lower()
    with _connect() as conn:
        r = conn.execute(
            "SELECT id, username, email, pwd_hash FROM users "
            "WHERE lower(username) = ? OR email = ?",
            (login, login),
        ).fetchone()
    if r and _check(password, r["pwd_hash"]):
        return {"id": r["id"], "username": r["username"], "email": r["email"]}
    return None


# Make sure the users table exists as soon as this module is imported.
init_auth_db()
