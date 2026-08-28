"""
Accounts, sessions, and the per-user library.

Voting requires an account. That is not a product decision, it is a measurement
one: the previous localStorage voter id meant one person could vote unlimited
times by clearing their browser storage, which makes every tally meaningless.
An account is the cheapest thing that makes a vote count stand for a person.

What an account also unlocks:

  watchlist / watched   - a per-user library, which is ordinary and expected
  view history          - what the user has looked at
  reviews               - and this is the important one

Reviews close the loop the project could not close otherwise. Report S4 needed
review text to score experiential axes, and TMDB has 499 reviews across 500
shows with a median of zero per show. A review written here is text about the
viewing experience, attached to a known show id, written by a known user, with
no licensing problem. `review_text_for()` feeds it straight back into the
labelling pipeline.

Password storage
----------------
scrypt (RFC 7914) from hashlib - memory-hard, in the standard library, no
dependency. Per-user random salt, parameters stored alongside the hash so they
can be raised later without invalidating existing passwords. Plaintext
passwords are never stored, logged, or returned.
"""

import hashlib
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "accounts.db"

# scrypt cost parameters. n is the work factor; 2**14 keeps a login around a
# few tens of milliseconds on this machine while making bulk cracking painful.
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1

SESSION_DAYS = 30
MIN_PASSWORD = 8
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")

STATUSES = ("watchlist", "watching", "watched")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT    NOT NULL,
    salt          TEXT    NOT NULL,
    -- Stored so the cost parameters can be raised later and old hashes still
    -- verify against the parameters they were created with.
    kdf           TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT    PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT    NOT NULL,
    expires_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id);

CREATE TABLE IF NOT EXISTS library (
    user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    show_id  INTEGER NOT NULL,
    status   TEXT    NOT NULL CHECK (status IN ('watchlist', 'watching', 'watched')),
    added_at TEXT    NOT NULL,
    PRIMARY KEY (user_id, show_id)
);

CREATE TABLE IF NOT EXISTS reviews (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    show_id    INTEGER NOT NULL,
    body       TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    -- One review per user per show; posting again edits it.
    UNIQUE (user_id, show_id)
);
CREATE INDEX IF NOT EXISTS idx_reviews_show ON reviews (show_id);

CREATE TABLE IF NOT EXISTS history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    show_id   INTEGER NOT NULL,
    viewed_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_user ON history (user_id, viewed_at DESC);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def hash_password(password, salt=None):
    """Derive a password hash. Returns (hex_hash, hex_salt, kdf_description)."""
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=64,
    )
    return derived.hex(), salt.hex(), f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}"


def verify_password(password, stored_hash, salt_hex, kdf):
    """
    Check a password against a stored hash.

    Uses compare_digest rather than == so the comparison takes the same time
    whether the hash matches on the first byte or the last; a plain == leaks
    how much of the hash was correct through timing.
    """
    try:
        _, n, r, p = kdf.split("$")
        derived = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=64,
        )
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(derived.hex(), stored_hash)


class AccountStore:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------- accounts

    @staticmethod
    def validate(username, password):
        """Return an error string, or None if the credentials are acceptable."""
        if not USERNAME_RE.match(username or ""):
            return "Username must be 3-32 characters: letters, numbers, . _ - only."
        if len(password or "") < MIN_PASSWORD:
            return f"Password must be at least {MIN_PASSWORD} characters."
        return None

    def create_user(self, username, password):
        error = self.validate(username, password)
        if error:
            return None, error

        password_hash, salt, kdf = hash_password(password)
        try:
            with self.conn:
                cur = self.conn.execute(
                    "INSERT INTO users (username, password_hash, salt, kdf, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (username, password_hash, salt, kdf, now_iso()),
                )
            return cur.lastrowid, None
        except sqlite3.IntegrityError:
            return None, "That username is already taken."

    def authenticate(self, username, password):
        row = self.conn.execute(
            "SELECT * FROM users WHERE username = ?", (username or "",)
        ).fetchone()

        if row is None:
            # Hash anyway so a missing username and a wrong password take
            # roughly the same time; otherwise the response time tells an
            # attacker which usernames exist.
            hash_password(password or "x")
            return None
        if not verify_password(password or "", row["password_hash"], row["salt"], row["kdf"]):
            return None
        return row["id"]

    # ------------------------------------------------------------- sessions

    def open_session(self, user_id):
        token = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
        with self.conn:
            self.conn.execute(
                "INSERT INTO sessions (token, user_id, created_at, expires_at)"
                " VALUES (?, ?, ?, ?)",
                (token, user_id, now_iso(), expires.isoformat()),
            )
        return token

    def user_for_token(self, token):
        """The user behind a session token, or None if absent or expired."""
        if not token:
            return None
        row = self.conn.execute(
            """
            SELECT u.id, u.username, s.expires_at
            FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
        if row is None:
            return None
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            self.close_session(token)
            return None
        return {"id": row["id"], "username": row["username"]}

    def close_session(self, token):
        with self.conn:
            self.conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

    # -------------------------------------------------------------- library

    def set_status(self, user_id, show_id, status):
        """Add or move a show in the user's library. status=None removes it."""
        with self.conn:
            if status is None:
                self.conn.execute(
                    "DELETE FROM library WHERE user_id = ? AND show_id = ?",
                    (user_id, show_id),
                )
                return
            if status not in STATUSES:
                raise ValueError(f"status must be one of {STATUSES}")
            self.conn.execute(
                """
                INSERT INTO library (user_id, show_id, status, added_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (user_id, show_id) DO UPDATE SET
                    status = excluded.status, added_at = excluded.added_at
                """,
                (user_id, show_id, status, now_iso()),
            )

    def library(self, user_id, status=None):
        if status:
            rows = self.conn.execute(
                "SELECT * FROM library WHERE user_id = ? AND status = ? ORDER BY added_at DESC",
                (user_id, status),
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM library WHERE user_id = ? ORDER BY added_at DESC",
                (user_id,),
            )
        return [dict(r) for r in rows]

    def status_for(self, user_id, show_id):
        row = self.conn.execute(
            "SELECT status FROM library WHERE user_id = ? AND show_id = ?",
            (user_id, show_id),
        ).fetchone()
        return row["status"] if row else None

    # -------------------------------------------------------------- reviews

    def put_review(self, user_id, show_id, body):
        body = (body or "").strip()
        if len(body) < 20:
            return "A review needs at least 20 characters."
        if len(body) > 8000:
            return "A review cannot be longer than 8000 characters."
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO reviews (user_id, show_id, body, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (user_id, show_id) DO UPDATE SET
                    body = excluded.body, created_at = excluded.created_at
                """,
                (user_id, show_id, body, now_iso()),
            )
        return None

    def reviews_for(self, show_id, limit=50):
        return [
            dict(r)
            for r in self.conn.execute(
                """
                SELECT r.id, r.body, r.created_at, u.username
                FROM reviews r JOIN users u ON u.id = r.user_id
                WHERE r.show_id = ? ORDER BY r.created_at DESC LIMIT ?
                """,
                (show_id, limit),
            )
        ]

    def my_review(self, user_id, show_id):
        row = self.conn.execute(
            "SELECT body FROM reviews WHERE user_id = ? AND show_id = ?",
            (user_id, show_id),
        ).fetchone()
        return row["body"] if row else None

    def review_text_for(self, show_id):
        """
        All review text for one show, for the labelling pipeline.

        This is the point of collecting reviews at all. TMDB gave 499 reviews
        across 500 shows, median zero, which could not support the experiential
        axes. Text written here is about the viewing experience, tied to a
        known show, and carries no licensing problem.
        """
        return [
            r["body"]
            for r in self.conn.execute(
                "SELECT body FROM reviews WHERE show_id = ? ORDER BY created_at",
                (show_id,),
            )
        ]

    def review_counts(self):
        """Per-show review counts, so progress toward a usable corpus is visible."""
        return {
            r["show_id"]: r["n"]
            for r in self.conn.execute(
                "SELECT show_id, COUNT(*) AS n FROM reviews GROUP BY show_id"
            )
        }

    # -------------------------------------------------------------- history

    def record_view(self, user_id, show_id):
        with self.conn:
            self.conn.execute(
                "INSERT INTO history (user_id, show_id, viewed_at) VALUES (?, ?, ?)",
                (user_id, show_id, now_iso()),
            )

    def history(self, user_id, limit=40):
        """Most recent distinct shows, newest first."""
        return [
            dict(r)
            for r in self.conn.execute(
                """
                SELECT show_id, MAX(viewed_at) AS viewed_at
                FROM history WHERE user_id = ?
                GROUP BY show_id ORDER BY viewed_at DESC LIMIT ?
                """,
                (user_id, limit),
            )
        ]

    def totals(self):
        return dict(
            self.conn.execute(
                """
                SELECT (SELECT COUNT(*) FROM users)   AS users,
                       (SELECT COUNT(*) FROM reviews) AS reviews,
                       (SELECT COUNT(*) FROM library) AS library_entries
                """
            ).fetchone()
        )
