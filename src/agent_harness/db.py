"""Postgres: chat sessions and messages for the UI.

Agent memory (the full LangGraph state, including tool calls and thinking blocks) lives
separately in the LangGraph Postgres checkpointer tables, keyed by the same session id.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

SCHEMA = """
CREATE TABLE IF NOT EXISTS harness_sessions (
    id          UUID PRIMARY KEY,
    title       TEXT NOT NULL,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS harness_messages (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID NOT NULL REFERENCES harness_sessions(id) ON DELETE CASCADE,
    role          TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool', 'error')),
    content       TEXT NOT NULL DEFAULT '',
    tool_name     TEXT,
    tool_call_id  TEXT,
    tool_input    JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS harness_messages_session ON harness_messages (session_id, id);
"""

MAX_TOOL_OUTPUT_STORED = 20_000


def make_pool(url: str) -> AsyncConnectionPool:
    # autocommit + dict_row are what the LangGraph checkpointer requires; we share the pool.
    return AsyncConnectionPool(
        url, open=False, min_size=1, max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )


class Database:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def setup(self) -> None:
        # One statement at a time: the pool prepares every statement (the checkpointer
        # needs prepare_threshold=0), and Postgres can't prepare several at once.
        async with self.pool.connection() as conn:
            for statement in filter(str.strip, SCHEMA.split(";")):
                await conn.execute(statement)

    # ---- sessions -------------------------------------------------------------

    async def create_session(self, provider: str, model: str, title: str = "New chat") -> dict:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO harness_sessions (id, title, provider, model) VALUES (%s, %s, %s, %s) "
                "RETURNING *", (uuid.uuid4(), title, provider, model))
            return _session(await cur.fetchone())

    async def list_sessions(self) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT s.*, (SELECT count(*) FROM harness_messages m WHERE m.session_id = s.id "
                "AND m.role IN ('user', 'assistant')) AS message_count "
                "FROM harness_sessions s ORDER BY updated_at DESC")
            return [_session(r) for r in await cur.fetchall()]

    async def get_session(self, session_id: str) -> dict | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute("SELECT * FROM harness_sessions WHERE id = %s",
                                     (_uuid(session_id),))
            row = await cur.fetchone()
        return _session(row) if row else None

    async def rename_session(self, session_id: str, title: str) -> dict | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE harness_sessions SET title = %s WHERE id = %s RETURNING *",
                (title, _uuid(session_id)))
            row = await cur.fetchone()
        return _session(row) if row else None

    async def touch_session(self, session_id: str) -> None:
        async with self.pool.connection() as conn:
            await conn.execute("UPDATE harness_sessions SET updated_at = now() WHERE id = %s",
                               (_uuid(session_id),))

    async def delete_session(self, session_id: str) -> bool:
        async with self.pool.connection() as conn:
            cur = await conn.execute("DELETE FROM harness_sessions WHERE id = %s",
                                     (_uuid(session_id),))
            return cur.rowcount > 0

    # ---- messages -------------------------------------------------------------

    async def add_message(self, session_id: str, role: str, content: str = "", *,
                          tool_name: str | None = None, tool_call_id: str | None = None,
                          tool_input: Any = None) -> dict:
        if role == "tool" and len(content) > MAX_TOOL_OUTPUT_STORED:
            content = content[:MAX_TOOL_OUTPUT_STORED] + "\n[truncated]"
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO harness_messages (session_id, role, content, tool_name, tool_call_id, "
                "tool_input) VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
                (_uuid(session_id), role, content, tool_name, tool_call_id,
                 Jsonb(tool_input) if tool_input is not None else None))
            return _message(await cur.fetchone())

    async def set_tool_result(self, session_id: str, tool_call_id: str, content: str) -> None:
        if len(content) > MAX_TOOL_OUTPUT_STORED:
            content = content[:MAX_TOOL_OUTPUT_STORED] + "\n[truncated]"
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE harness_messages SET content = %s WHERE session_id = %s "
                "AND tool_call_id = %s AND role = 'tool'",
                (content, _uuid(session_id), tool_call_id))

    async def list_messages(self, session_id: str) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM harness_messages WHERE session_id = %s ORDER BY id",
                (_uuid(session_id),))
            return [_message(r) for r in await cur.fetchall()]


def _uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except ValueError:
        raise KeyError(value) from None


def _session(row: dict) -> dict:
    return {
        "id": str(row["id"]), "title": row["title"], "provider": row["provider"],
        "model": row["model"], "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
        **({"message_count": row["message_count"]} if "message_count" in row else {}),
    }


def _message(row: dict) -> dict:
    return {
        "id": row["id"], "role": row["role"], "content": row["content"],
        "tool_name": row["tool_name"], "tool_call_id": row["tool_call_id"],
        "tool_input": row["tool_input"], "created_at": row["created_at"].isoformat(),
    }
