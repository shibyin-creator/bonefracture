"""SQLite persistence for clinicians, audit trails, and inference history."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from werkzeug.security import generate_password_hash

from config import Config


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    Config.ensure_directories()
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'clinician',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_login TEXT
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    username TEXT,
    action TEXT NOT NULL,
    details TEXT,
    ip_address TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS inferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    detector_model TEXT,
    classifier_model TEXT,
    predicted_class TEXT,
    anatomical_site TEXT,
    confidence REAL,
    detections_json TEXT,
    needs_refixation INTEGER NOT NULL DEFAULT 0,
    gradcam_path TEXT,
    yolo_path TEXT,
    refix_path TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (Config.DEFAULT_ADMIN_USER,),
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, full_name, role, is_active, created_at)
                VALUES (?, ?, ?, 'admin', 1, ?)
                """,
                (
                    Config.DEFAULT_ADMIN_USER,
                    generate_password_hash(Config.DEFAULT_ADMIN_PASSWORD),
                    "System Administrator",
                    utcnow(),
                ),
            )


def fetchone(query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(query, params).fetchone()


def fetchall(query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(query, params).fetchall()


def execute(query: str, params: tuple[Any, ...] = ()) -> int:
    with get_connection() as conn:
        cur = conn.execute(query, params)
        return int(cur.lastrowid)


def log_audit(user: dict[str, Any] | None, action: str, details: str = "", ip: str = "") -> None:
    execute(
        """
        INSERT INTO audit_logs (user_id, username, action, details, ip_address, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user.get("id") if user else None,
            user.get("username") if user else "anonymous",
            action,
            details,
            ip,
            utcnow(),
        ),
    )
