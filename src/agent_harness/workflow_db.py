"""Postgres: JIRA-ticket-to-pull-request workflows (platform->repo mapping, workflow state,
per-stage PR tracking, and the event log the UI streams).

Separate from `db.py` (which is scoped to chat sessions/messages) because this is a different
lifecycle and query surface, but it shares the same connection pool.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from .db import MAX_TOOL_OUTPUT_STORED, _uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_repos (
    id          BIGSERIAL PRIMARY KEY,
    platform    TEXT NOT NULL UNIQUE,
    repo_owner  TEXT NOT NULL,
    repo_name   TEXT NOT NULL,
    base_branch TEXT NOT NULL DEFAULT 'main',
    clone_url   TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS workflows (
    id             UUID PRIMARY KEY,
    ticket_key     TEXT NOT NULL,
    ticket_summary TEXT NOT NULL DEFAULT '',
    platform       TEXT NOT NULL,
    repo_owner     TEXT NOT NULL,
    repo_name      TEXT NOT NULL,
    base_branch    TEXT NOT NULL,
    clone_url      TEXT NOT NULL,
    slug           TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending','running','awaiting_approval','merging',
                                      'advancing','completed','failed')),
    current_stage  TEXT NOT NULL DEFAULT 'spec'
                   CHECK (current_stage IN ('spec','plan','tasks','implement','done')),
    error          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS workflows_ticket ON workflows (ticket_key);
CREATE TABLE IF NOT EXISTS workflow_stages (
    id            BIGSERIAL PRIMARY KEY,
    workflow_id   UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    stage         TEXT NOT NULL CHECK (stage IN ('spec','plan','tasks','implement')),
    branch        TEXT NOT NULL,
    pr_number     INTEGER,
    pr_url        TEXT,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','running','pr_open','approved','merged','failed')),
    artifact_path TEXT,
    error         TEXT,
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workflow_id, stage)
);
CREATE TABLE IF NOT EXISTS workflow_events (
    id          BIGSERIAL PRIMARY KEY,
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    stage       TEXT,
    type        TEXT NOT NULL,
    payload     JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS workflow_events_workflow ON workflow_events (workflow_id, id);
"""


class WorkflowDatabase:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def setup(self) -> None:
        async with self.pool.connection() as conn:
            for statement in filter(str.strip, SCHEMA.split(";")):
                await conn.execute(statement)

    # ---- platform_repos --------------------------------------------------------

    async def upsert_platform_repo(self, platform: str, repo_owner: str, repo_name: str,
                                   base_branch: str, clone_url: str) -> dict:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO platform_repos (platform, repo_owner, repo_name, base_branch, clone_url) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (platform) DO UPDATE SET "
                "repo_owner = EXCLUDED.repo_owner, repo_name = EXCLUDED.repo_name, "
                "base_branch = EXCLUDED.base_branch, clone_url = EXCLUDED.clone_url, "
                "updated_at = now() RETURNING *",
                (platform, repo_owner, repo_name, base_branch, clone_url))
            return _platform_repo(await cur.fetchone())

    async def get_platform_repo(self, platform: str) -> dict | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute("SELECT * FROM platform_repos WHERE platform = %s", (platform,))
            row = await cur.fetchone()
        return _platform_repo(row) if row else None

    async def list_platform_repos(self) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute("SELECT * FROM platform_repos ORDER BY platform")
            return [_platform_repo(r) for r in await cur.fetchall()]

    async def delete_platform_repo(self, platform: str) -> bool:
        async with self.pool.connection() as conn:
            cur = await conn.execute("DELETE FROM platform_repos WHERE platform = %s", (platform,))
            return cur.rowcount > 0

    # ---- workflows --------------------------------------------------------------

    async def create_workflow(self, ticket_key: str, ticket_summary: str, platform: str,
                              repo_owner: str, repo_name: str, base_branch: str, clone_url: str,
                              slug: str) -> dict:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO workflows (id, ticket_key, ticket_summary, platform, repo_owner, "
                "repo_name, base_branch, clone_url, slug) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "RETURNING *",
                (uuid.uuid4(), ticket_key, ticket_summary, platform, repo_owner, repo_name,
                 base_branch, clone_url, slug))
            return _workflow(await cur.fetchone())

    async def get_workflow(self, workflow_id: str) -> dict | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute("SELECT * FROM workflows WHERE id = %s", (_uuid(workflow_id),))
            row = await cur.fetchone()
        return _workflow(row) if row else None

    async def list_workflows(self) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute("SELECT * FROM workflows ORDER BY updated_at DESC")
            return [_workflow(r) for r in await cur.fetchall()]

    async def list_active_workflows(self) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM workflows WHERE status NOT IN ('completed', 'failed')")
            return [_workflow(r) for r in await cur.fetchall()]

    async def find_active_workflow_for_ticket(self, ticket_key: str) -> dict | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM workflows WHERE ticket_key = %s AND status NOT IN ('completed', 'failed') "
                "ORDER BY created_at DESC LIMIT 1", (ticket_key,))
            row = await cur.fetchone()
        return _workflow(row) if row else None

    async def update_workflow_status(self, workflow_id: str, *, status: str | None = None,
                                     current_stage: str | None = None,
                                     error: str | None = "__unset__") -> dict | None:
        sets, params = ["updated_at = now()"], []
        if status is not None:
            sets.append("status = %s"); params.append(status)
        if current_stage is not None:
            sets.append("current_stage = %s"); params.append(current_stage)
        if error != "__unset__":
            sets.append("error = %s"); params.append(error)
        params.append(_uuid(workflow_id))
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"UPDATE workflows SET {', '.join(sets)} WHERE id = %s RETURNING *", params)
            row = await cur.fetchone()
        return _workflow(row) if row else None

    # ---- workflow_stages ----------------------------------------------------------

    async def create_stage(self, workflow_id: str, stage: str, branch: str) -> dict:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO workflow_stages (workflow_id, stage, branch, status, started_at) "
                "VALUES (%s, %s, %s, 'running', now()) "
                "ON CONFLICT (workflow_id, stage) DO UPDATE SET branch = EXCLUDED.branch "
                "RETURNING *", (_uuid(workflow_id), stage, branch))
            return _stage(await cur.fetchone())

    async def get_stage(self, workflow_id: str, stage: str) -> dict | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM workflow_stages WHERE workflow_id = %s AND stage = %s",
                (_uuid(workflow_id), stage))
            row = await cur.fetchone()
        return _stage(row) if row else None

    async def list_stages(self, workflow_id: str) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM workflow_stages WHERE workflow_id = %s ORDER BY id",
                (_uuid(workflow_id),))
            return [_stage(r) for r in await cur.fetchall()]

    async def update_stage(self, workflow_id: str, stage: str, **fields: Any) -> dict | None:
        if not fields:
            return await self.get_stage(workflow_id, stage)
        allowed = {"branch", "pr_number", "pr_url", "status", "artifact_path", "error", "finished_at"}
        if unknown := set(fields) - allowed:
            raise ValueError(f"unknown stage field(s): {unknown}")
        sets = [f"{k} = %s" for k in fields]
        params = [*fields.values(), _uuid(workflow_id), stage]
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"UPDATE workflow_stages SET {', '.join(sets)} WHERE workflow_id = %s AND stage = %s "
                "RETURNING *", params)
            row = await cur.fetchone()
        return _stage(row) if row else None

    # ---- workflow_events ----------------------------------------------------------

    async def add_event(self, workflow_id: str, type: str, *, stage: str | None = None,
                        payload: dict | None = None) -> dict:
        payload = dict(payload or {})
        if isinstance(payload.get("text"), str) and len(payload["text"]) > MAX_TOOL_OUTPUT_STORED:
            payload["text"] = payload["text"][:MAX_TOOL_OUTPUT_STORED] + "\n[truncated]"
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO workflow_events (workflow_id, stage, type, payload) VALUES (%s, %s, %s, %s) "
                "RETURNING *", (_uuid(workflow_id), stage, type, Jsonb(payload)))
            return _event(await cur.fetchone())

    async def list_events(self, workflow_id: str, after_id: int = 0) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM workflow_events WHERE workflow_id = %s AND id > %s ORDER BY id",
                (_uuid(workflow_id), after_id))
            return [_event(r) for r in await cur.fetchall()]


def _platform_repo(row: dict) -> dict:
    return {
        "platform": row["platform"], "repo_owner": row["repo_owner"], "repo_name": row["repo_name"],
        "base_branch": row["base_branch"], "clone_url": row["clone_url"],
        "created_at": row["created_at"].isoformat(), "updated_at": row["updated_at"].isoformat(),
    }


def _workflow(row: dict) -> dict:
    return {
        "id": str(row["id"]), "ticket_key": row["ticket_key"], "ticket_summary": row["ticket_summary"],
        "platform": row["platform"], "repo_owner": row["repo_owner"], "repo_name": row["repo_name"],
        "base_branch": row["base_branch"], "clone_url": row["clone_url"], "slug": row["slug"],
        "status": row["status"], "current_stage": row["current_stage"], "error": row["error"],
        "created_at": row["created_at"].isoformat(), "updated_at": row["updated_at"].isoformat(),
    }


def _stage(row: dict) -> dict:
    return {
        "id": row["id"], "workflow_id": str(row["workflow_id"]), "stage": row["stage"],
        "branch": row["branch"], "pr_number": row["pr_number"], "pr_url": row["pr_url"],
        "status": row["status"], "artifact_path": row["artifact_path"], "error": row["error"],
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
        "created_at": row["created_at"].isoformat(),
    }


def _event(row: dict) -> dict:
    return {
        "id": row["id"], "workflow_id": str(row["workflow_id"]), "stage": row["stage"],
        "type": row["type"], "payload": row["payload"], "created_at": row["created_at"].isoformat(),
    }
