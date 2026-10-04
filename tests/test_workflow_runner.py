"""WorkflowRunner: full spec -> plan -> tasks -> implement pipeline against fakes and a real
local git remote, plus the resume-after-restart path."""

import asyncio
import subprocess

import pytest
from conftest import FakeGithubClient, FakeJiraClient, FakeToolModel, tool_call
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_harness.config import GithubConfig, WorkflowConfig
from agent_harness.db import make_pool
from agent_harness.jira_client import JiraTicket
from agent_harness.workflow_db import WorkflowDatabase
from agent_harness.workflow_runner import STAGES, UnknownPlatform, WorkflowBusy, WorkflowRunner


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def bare_remote(tmp_path):
    bare = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    _git(["init", "--bare", "-b", "main", str(bare)], cwd=tmp_path)
    _git(["init", "-b", "main", str(seed)], cwd=tmp_path)
    (seed / "README.md").write_text("hello\n")
    _git(["-c", "user.name=seed", "-c", "user.email=seed@example.com", "add", "-A"], cwd=seed)
    _git(["-c", "user.name=seed", "-c", "user.email=seed@example.com", "commit", "-m", "init"], cwd=seed)
    _git(["remote", "add", "origin", str(bare)], cwd=seed)
    _git(["push", "origin", "main"], cwd=seed)
    return str(bare)


async def _workflow_db(database_url) -> WorkflowDatabase:
    pool = make_pool(database_url)
    await pool.open(wait=True, timeout=15)
    db = WorkflowDatabase(pool)
    await db.setup()
    return db


def _artifacts_for(slug_dir: str) -> dict:
    return {
        "spec": (f"specs/{slug_dir}/spec.md", "# Spec\n"),
        "plan": (f"specs/{slug_dir}/plan.md", "# Plan\n"),
        "tasks": (f"specs/{slug_dir}/tasks.md", "- [ ] T1: do it\n"),
        "implement": (f"specs/{slug_dir}/tasks.md", "- [x] T1: do it\n"),
    }


def _stage_model_factory(artifacts: dict):
    """Each call advances to the next stage in STAGES order, scripting a model that writes
    that stage's artifact then finishes."""
    order = iter(STAGES)

    def factory():
        stage = next(order)
        path, content = artifacts[stage]
        return FakeToolModel(responses=[
            tool_call("write_file", {"path": path, "content": content}),
            AIMessage(f"{stage} done"),
        ], seen=[])

    return factory


def _git_merge_hook(bare_remote: str, tmp_path):
    """on_merge callback for FakeGithubClient: actually fast-forward-merges the PR's head
    branch into its base on the real bare remote, the way GitHub's real merge API would --
    so later stages (and final verification) see earlier stages' committed artifacts."""
    counter = {"n": 0}

    def on_merge(owner, repo, number, head, base):
        counter["n"] += 1
        merge_dir = tmp_path / f"merge-{counter['n']}"
        _git(["clone", bare_remote, str(merge_dir)], cwd=tmp_path)
        _git(["fetch", "origin", head], cwd=merge_dir)
        _git(["merge", "--ff-only", f"origin/{head}"], cwd=merge_dir)
        _git(["push", "origin", base], cwd=merge_dir)

    return on_merge


async def _wait_for(check, timeout=10):
    """Poll an async `check()` until it returns a truthy value; return that value."""
    async def poll():
        while True:
            value = await check()
            if value:
                return value
            await asyncio.sleep(0.02)
    return await asyncio.wait_for(poll(), timeout)


def test_happy_path_runs_all_stages_and_merges_each_pr(tmp_path, database_url, bare_remote):
    async def run():
        db = await _workflow_db(database_url)
        await db.upsert_platform_repo("web", "acme", "webapp", "main", bare_remote)

        jira = FakeJiraClient({"PROJ-1": JiraTicket(key="PROJ-1", summary="Add a widget",
                                                     description_html="<p>desc</p>", platform="web")})
        github = FakeGithubClient(on_merge=_git_merge_hook(bare_remote, tmp_path))
        workflow_cfg = WorkflowConfig(enabled=True, workdir=str(tmp_path / "work"),
                                      shell_timeout_seconds=30)
        github_cfg = GithubConfig(enabled=True, poll_interval_seconds=0.05)
        model_factory = _stage_model_factory(_artifacts_for("proj-1-add-a-widget"))
        runner = WorkflowRunner(db, jira, github, model_factory, InMemorySaver(),
                                workflow_cfg, github_cfg)

        workflow = await runner.start_new("PROJ-1")
        assert workflow["platform"] == "web" and workflow["slug"] == "add-a-widget"
        assert runner.is_running(workflow["id"])

        for stage in STAGES:
            async def pr_opened(s=stage):
                row = await db.get_stage(workflow["id"], s)
                return row["pr_number"] if row and row["pr_number"] else None
            pr_number = await _wait_for(pr_opened)
            github.approve(pr_number)

        async def finished():
            return not runner.is_running(workflow["id"])
        await _wait_for(finished)

        final = await db.get_workflow(workflow["id"])
        assert final["status"] == "completed", final
        stages = await db.list_stages(workflow["id"])
        assert [s["status"] for s in stages] == ["merged", "merged", "merged", "merged"]
        assert github.merged == [1, 2, 3, 4]

        # Verify the artifacts and the implement-stage code change actually landed on main.
        verify_dir = tmp_path / "verify"
        _git(["clone", bare_remote, str(verify_dir)], cwd=tmp_path)
        tasks_file = verify_dir / "specs" / "proj-1-add-a-widget" / "tasks.md"
        assert tasks_file.read_text() == "- [x] T1: do it\n"
        assert (verify_dir / "specs" / "proj-1-add-a-widget" / "spec.md").is_file()
        assert (verify_dir / "specs" / "proj-1-add-a-widget" / "plan.md").is_file()

    asyncio.run(run())


def test_subscribe_delivers_live_events_ending_in_sentinel(tmp_path, database_url, bare_remote):
    """This is the exact mechanism the API's SSE endpoint depends on (subscribe/unsubscribe,
    every broadcast event carrying a durable id, a None sentinel on completion) -- tested here
    directly rather than over real HTTP, since httpx's ASGITransport fully buffers responses
    and can't exercise live/incremental delivery (confirmed separately)."""
    async def run():
        db = await _workflow_db(database_url)
        await db.upsert_platform_repo("web", "acme", "webapp", "main", bare_remote)
        jira = FakeJiraClient({"PROJ-1": JiraTicket(key="PROJ-1", summary="Add a widget",
                                                     description_html="", platform="web")})
        github = FakeGithubClient(on_merge=_git_merge_hook(bare_remote, tmp_path))
        workflow_cfg = WorkflowConfig(enabled=True, workdir=str(tmp_path / "work"),
                                      shell_timeout_seconds=30)
        github_cfg = GithubConfig(enabled=True, poll_interval_seconds=0.05)
        model_factory = _stage_model_factory(_artifacts_for("proj-1-add-a-widget"))
        runner = WorkflowRunner(db, jira, github, model_factory, InMemorySaver(),
                                workflow_cfg, github_cfg)

        workflow = await runner.start_new("PROJ-1")
        queue = runner.subscribe(workflow["id"])
        received = []

        async def drain():
            while (event := await queue.get()) is not None:
                received.append(event)
                if event["type"] == "stage_pr_opened":
                    github.approve(event["payload"]["pr_number"])

        await asyncio.wait_for(drain(), timeout=20)

        assert all(isinstance(e["id"], int) for e in received)
        assert [e["id"] for e in received] == sorted(e["id"] for e in received)
        assert received[-1]["type"] == "workflow_completed"
        assert "stage_pr_opened" in {e["type"] for e in received}

        final = await db.get_workflow(workflow["id"])
        assert final["status"] == "completed"
        runner.unsubscribe(workflow["id"], queue)   # harmless no-op once already finished

    asyncio.run(run())


def test_busy_rejects_concurrent_workflow_for_same_ticket(tmp_path, database_url, bare_remote):
    """start_new()'s busy-check only queries workflow_db for a non-terminal row -- it doesn't
    need a real spawned/running background task, so this inserts the "already active"
    workflow directly rather than going through start_new() + a live WorkflowRunner task.
    (A live task was used here previously; letting asyncio.run() implicitly cancel it on exit
    hangs on Windows -- cancelling a task parked inside asyncio.to_thread(subprocess.run(...))
    blocks until that thread actually finishes. Avoiding a live task sidesteps that entirely.)"""
    async def run():
        db = await _workflow_db(database_url)
        await db.upsert_platform_repo("web", "acme", "webapp", "main", bare_remote)
        jira = FakeJiraClient({"PROJ-1": JiraTicket(key="PROJ-1", summary="Add a widget",
                                                     description_html="", platform="web")})
        runner = WorkflowRunner(db, jira, FakeGithubClient(), lambda: None, InMemorySaver(),
                                WorkflowConfig(enabled=True, workdir=str(tmp_path / "work")),
                                GithubConfig(enabled=True))

        await db.create_workflow("PROJ-1", "Add a widget", "web", "acme", "webapp", "main",
                                 bare_remote, "add-a-widget")

        with pytest.raises(WorkflowBusy):
            await runner.start_new("PROJ-1")

    asyncio.run(run())


def test_unknown_platform_is_reported(tmp_path, database_url, bare_remote):
    async def run():
        db = await _workflow_db(database_url)
        jira = FakeJiraClient({"PROJ-9": JiraTicket(key="PROJ-9", summary="x", description_html="",
                                                     platform="no-such-platform")})
        runner = WorkflowRunner(db, jira, FakeGithubClient(), lambda: None, InMemorySaver(),
                                WorkflowConfig(enabled=True, workdir=str(tmp_path / "work")),
                                GithubConfig(enabled=True))
        with pytest.raises(UnknownPlatform):
            await runner.start_new("PROJ-9")

    asyncio.run(run())


def test_resume_in_flight_continues_from_current_stage(tmp_path, database_url, bare_remote):
    """Simulates an app restart that happened right after the spec stage merged: the DB/git
    state is built directly (no live runner involved for 'spec'), then a fresh WorkflowRunner's
    resume_in_flight() must pick up at 'plan', not redo 'spec'."""
    async def run():
        db = await _workflow_db(database_url)
        await db.upsert_platform_repo("web", "acme", "webapp", "main", bare_remote)

        # Simulate the spec stage having already merged: push spec.md to main directly.
        merge_dir = tmp_path / "pre-merge"
        _git(["clone", bare_remote, str(merge_dir)], cwd=tmp_path)
        (merge_dir / "specs" / "proj-1-add-a-widget").mkdir(parents=True)
        (merge_dir / "specs" / "proj-1-add-a-widget" / "spec.md").write_text("# Spec\n")
        _git(["add", "-A"], cwd=merge_dir)
        _git(["-c", "user.name=bot", "-c", "user.email=bot@example.com", "commit", "-m", "spec"],
             cwd=merge_dir)
        _git(["push", "origin", "main"], cwd=merge_dir)

        workflow = await db.create_workflow("PROJ-1", "Add a widget", "web", "acme", "webapp",
                                            "main", bare_remote, "add-a-widget")
        await db.create_stage(workflow["id"], "spec", "proj-1-add-a-widget-spec")
        await db.update_stage(workflow["id"], "spec", status="merged", pr_number=1,
                              pr_url="https://github.com/acme/webapp/pull/1")
        await db.update_workflow_status(workflow["id"], status="advancing", current_stage="plan")

        jira = FakeJiraClient({"PROJ-1": JiraTicket(key="PROJ-1", summary="Add a widget",
                                                     description_html="", platform="web")})
        github = FakeGithubClient(on_merge=_git_merge_hook(bare_remote, tmp_path))
        workflow_cfg = WorkflowConfig(enabled=True, workdir=str(tmp_path / "work"))
        github_cfg = GithubConfig(enabled=True, poll_interval_seconds=0.05)
        # Only plan/tasks/implement will run, so the factory only needs those three stages.
        artifacts = _artifacts_for("proj-1-add-a-widget")
        remaining = iter(("plan", "tasks", "implement"))

        def model_factory():
            stage = next(remaining)
            path, content = artifacts[stage]
            return FakeToolModel(responses=[
                tool_call("write_file", {"path": path, "content": content}),
                AIMessage(f"{stage} done"),
            ], seen=[])

        runner = WorkflowRunner(db, jira, github, model_factory, InMemorySaver(),
                                workflow_cfg, github_cfg)
        await runner.resume_in_flight()
        assert runner.is_running(workflow["id"])

        for stage in ("plan", "tasks", "implement"):
            async def pr_opened(s=stage):
                row = await db.get_stage(workflow["id"], s)
                return row["pr_number"] if row and row["pr_number"] else None
            number = await _wait_for(pr_opened)
            github.approve(number)

        async def finished():
            return not runner.is_running(workflow["id"])
        await _wait_for(finished)

        final = await db.get_workflow(workflow["id"])
        assert final["status"] == "completed", final
        stages = await db.list_stages(workflow["id"])
        by_stage = {s["stage"]: s["status"] for s in stages}
        assert by_stage == {"spec": "merged", "plan": "merged", "tasks": "merged",
                            "implement": "merged"}
        # Only plan/tasks/implement were (re)opened and merged on this FakeGithubClient --
        # the spec stage's pre-seeded "merged" row was never redone. (Numbering restarts at 1
        # on this fresh fake client, so plan's real PR legitimately becomes #1.)
        assert github.merged == [1, 2, 3]

    asyncio.run(run())

