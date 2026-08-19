"""SQLite-backed admin authentication with PBKDF2-HMAC password hashing."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

import config

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        config.PBKDF2_ITERATIONS,
    )
    return salt.hex(), digest.hex()


def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    _, candidate = _hash_password(password, salt=salt)
    return hmac.compare_digest(candidate, hash_hex)


@dataclass
class AdminUser:
    username: str
    created_at: str
    last_login: Optional[str]


class AdminAuthManager:
    """Secure administrator authentication persisted in SQLite."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = str(db_path or config.AUTH_DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS admins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    event TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
        if not self.user_exists(config.DEFAULT_ADMIN_USERNAME):
            self.create_admin(
                config.DEFAULT_ADMIN_USERNAME,
                config.DEFAULT_ADMIN_PASSWORD,
            )
            logger.info("Provisioned default administrator account.")

    def _log_event(self, username: Optional[str], event: str, detail: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO auth_events (username, event, detail, created_at) VALUES (?, ?, ?, ?)",
                (username, event, detail, _utc_now()),
            )

    def user_exists(self, username: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM admins WHERE username = ?",
                (username,),
            ).fetchone()
        return row is not None

    def create_admin(self, username: str, password: str) -> None:
        if len(password) < 10:
            raise ValueError("Administrator password must be at least 10 characters.")
        salt_hex, hash_hex = _hash_password(password)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO admins (username, salt, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, salt_hex, hash_hex, _utc_now()),
            )
        self._log_event(username, "admin_created")

    def authenticate(self, username: str, password: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT salt, password_hash FROM admins WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            self._log_event(username, "login_failed", "unknown_user")
            return False
        ok = _verify_password(password, row["salt"], row["password_hash"])
        if ok:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE admins SET last_login = ? WHERE username = ?",
                    (_utc_now(), username),
                )
            self._log_event(username, "login_success")
            return True
        self._log_event(username, "login_failed", "invalid_password")
        return False

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        if not self.authenticate(username, old_password):
            return False
        salt_hex, hash_hex = _hash_password(new_password)
        with self._connect() as conn:
            conn.execute(
                "UPDATE admins SET salt = ?, password_hash = ? WHERE username = ?",
                (salt_hex, hash_hex, username),
            )
        self._log_event(username, "password_changed")
        return True

    def list_auth_events(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT username, event, detail, created_at FROM auth_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
