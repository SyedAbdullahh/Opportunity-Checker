import json
import os
import sqlite3
from typing import Any

DB_PATH = os.getenv("DATABASE_PATH", "opportunity_check.db")


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_email TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def get_user_profile(user_email: str) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT profile_json FROM user_profiles WHERE user_email = ?",
            (user_email,),
        ).fetchone()

    if not row:
        return None

    try:
        return json.loads(row["profile_json"])
    except json.JSONDecodeError:
        return None


def save_user_profile(user_email: str, profile_data: dict[str, Any]) -> None:
    profile_json = json.dumps(profile_data, ensure_ascii=False)
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO user_profiles (user_email, profile_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_email)
            DO UPDATE SET profile_json = excluded.profile_json, updated_at = CURRENT_TIMESTAMP
            """,
            (user_email, profile_json),
        )
        connection.commit()
