"""WorkflowDatabase: platform_repos, workflows, workflow_stages, workflow_events."""

import asyncio

from agent_harness.db import make_pool
from agent_harness.workflow_db import WorkflowDatabase


async def _db(database_url) -> WorkflowDatabase:
    pool = make_pool(database_url)
    await pool.open(wait=True, timeout=15)
    db = WorkflowDatabase(pool)
    await db.setup()
    return db


def test_platform_repo_crud(database_url):
    async def run():
        db = await _db(database_url)
        assert await db.get_platform_repo("web") is None
        repo = await db.upsert_platform_repo("web", "acme", "webapp", "main", "https://x/acme/webapp.git")
        assert repo["platform"] == "web" and repo["repo_owner"] == "acme"
        updated = await db.upsert_platform_repo("web", "acme", "webapp", "develop", "https://x/acme/webapp.git")
        assert updated["base_branch"] == "develop"
        assert [r["platform"] for r in await db.list_platform_repos()] == ["web"]
        assert await db.delete_platform_repo("web") is True
        assert await db.delete_platform_repo("web") is False
    asyncio.run(run())


def test_workflow_lifecycle(database_url):
    async def run():
        db = await _db(database_url)
        wf = await db.create_workflow("PROJ-1", "Add widget", "web", "acme", "webapp", "main",
                                      "https://x/acme/webapp.git", "add-widget")
        assert wf["status"] == "pending" and wf["current_stage"] == "spec"
        assert await db.find_active_workflow_for_ticket("PROJ-1") is not None

        stage = await db.create_stage(wf["id"], "spec", "proj-1-add-widget-spec")
        assert stage["status"] == "running"
        stage = await db.update_stage(wf["id"], "spec", status="pr_open", pr_number=7,
                                      pr_url="https://github.com/acme/webapp/pull/7")
        assert stage["pr_number"] == 7 and stage["status"] == "pr_open"
        assert [s["stage"] for s in await db.list_stages(wf["id"])] == ["spec"]

        updated = await db.update_workflow_status(wf["id"], status="awaiting_approval")
        assert updated["status"] == "awaiting_approval"

        e1 = await db.add_event(wf["id"], "stage_start", stage="spec", payload={"text": "starting"})
        e2 = await db.add_event(wf["id"], "stage_pr_opened", stage="spec", payload={"pr_number": 7})
        events = await db.list_events(wf["id"])
        assert [e["id"] for e in events] == [e1["id"], e2["id"]]
        assert await db.list_events(wf["id"], after_id=e1["id"]) == [e2]

        await db.update_workflow_status(wf["id"], status="completed")
        assert await db.find_active_workflow_for_ticket("PROJ-1") is None
        assert wf["id"] not in [w["id"] for w in await db.list_active_workflows()]
    asyncio.run(run())


def test_event_payload_text_is_truncated(database_url):
    async def run():
        db = await _db(database_url)
        wf = await db.create_workflow("PROJ-2", "x", "web", "acme", "webapp", "main",
                                      "https://x/acme/webapp.git", "x")
        event = await db.add_event(wf["id"], "agent_text", payload={"text": "a" * 30_000})
        assert len(event["payload"]["text"]) < 30_000
        assert event["payload"]["text"].endswith("[truncated]")
    asyncio.run(run())
