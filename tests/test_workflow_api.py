"""End-to-end through the real FastAPI app for the JIRA-ticket-to-PR workflow pipeline:
platform-repos CRUD, starting a workflow, driving it through all 4 stages via the fakes,
and reading back its event log once finished.

Live/incremental SSE delivery is exercised directly against WorkflowRunner.subscribe() in
test_workflow_runner.py, not here: httpx's ASGITransport fully buffers responses (confirmed
separately), so a real client.stream() read over it can't see events until the whole request
finishes -- which never happens for a still-running workflow. This file drives the workflow by
polling the REST endpoints (the same thing a dumb client/curl could do), then reads the
*finished* workflow's event log in one shot, which exercises the SSE endpoint's catch-up/replay
path (the `id:` line format, event ordering) without needing live streaming."""

import asyncio
import json
import subprocess

import httpx
import pytest
from conftest import FakeGithubClient, FakeJiraClient, FakeToolModel, tool_call
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_harness.api import create_app
from agent_harness.config import (
    AgentConfig,
    ClaudeConfig,
    Config,
    CopilotConfig,
    DatabaseConfig,
    GithubConfig,
    JiraConfig,
    ServerConfig,
    WorkflowConfig,
)
from agent_harness.jira_client import JiraTicket
from agent_harness.workflow_runner import STAGES


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


def _git_merge_hook(bare_remote: str, tmp_path):
    counter = {"n": 0}

    def on_merge(owner, repo, number, head, base):
        counter["n"] += 1
        merge_dir = tmp_path / f"merge-{counter['n']}"
        _git(["clone", bare_remote, str(merge_dir)], cwd=tmp_path)
        _git(["fetch", "origin", head], cwd=merge_dir)
        _git(["merge", "--ff-only", f"origin/{head}"], cwd=merge_dir)
        _git(["push", "origin", base], cwd=merge_dir)

    return on_merge


def _artifacts_for(slug_dir: str) -> dict:
    return {
        "spec": (f"specs/{slug_dir}/spec.md", "# Spec\n"),
        "plan": (f"specs/{slug_dir}/plan.md", "# Plan\n"),
        "tasks": (f"specs/{slug_dir}/tasks.md", "- [ ] T1: do it\n"),
        "implement": (f"specs/{slug_dir}/tasks.md", "- [x] T1: do it\n"),
    }


def _stage_model_factory(artifacts: dict):
    order = iter(STAGES)

    def factory():
        stage = next(order)
        path, content = artifacts[stage]
        return FakeToolModel(responses=[
            tool_call("write_file", {"path": path, "content": content}),
            AIMessage(f"{stage} done"),
        ], seen=[])

    return factory


def _workflow_app(database_url, tmp_path, jira, github):
    """Builds Config directly (bypassing TOML) and wires the fakes in via create_app's DI
    factories -- mirrors the `harness` fixture in conftest.py but for the workflow feature."""
    cfg = Config(
        path=tmp_path / "config.toml",
        server=ServerConfig(),
        database=DatabaseConfig(url=database_url),
        claude=ClaudeConfig(models=("claude-test",), default_model="claude-test"),
        copilot=CopilotConfig(enabled=False),
        agent=AgentConfig(),
        jira=JiraConfig(enabled=True),
        github=GithubConfig(enabled=True, poll_interval_seconds=0.05),
        workflow=WorkflowConfig(enabled=True, workdir=str(tmp_path / "work"), shell_timeout_seconds=30),
        mcp_servers=(),
    )
    return create_app(cfg, providers_factory=lambda cfg, mcp, checkpointer: {},
                      jira_factory=lambda cfg: jira, github_factory=lambda cfg: github,
                      workflow_model_factory=lambda cfg: _stage_model_factory(
                          _artifacts_for("proj-1-add-a-widget")))


class Client:
    def __init__(self, app):
        self.app = app
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


@pytest.fixture
def run_app():
    def run(app, body):
        async def main():
            async with app.router.lifespan_context(app):
                client = Client(app)
                try:
                    await body(client)
                finally:
                    await client.http.aclose()
        asyncio.run(main())
    return run


async def _wait_for(check, timeout=10):
    async def poll():
        while True:
            value = await check()
            if value:
                return value
            await asyncio.sleep(0.02)
    return await asyncio.wait_for(poll(), timeout)


def test_platform_repo_crud_over_http(database_url, tmp_path, run_app):
    github = FakeGithubClient()
    jira = FakeJiraClient({})
    app = _workflow_app(database_url, tmp_path, jira, github)

    async def body(c):
        assert (await c.http.get("/api/platform-repos")).json() == []

        r = await c.http.post("/api/platform-repos", json={
            "platform": "web", "repo_owner": "acme", "repo_name": "webapp",
            "clone_url": "https://x/acme/webapp.git"})
        assert r.status_code == 201 and r.json()["base_branch"] == "main"

        assert len((await c.http.get("/api/platform-repos")).json()) == 1

        assert (await c.http.delete("/api/platform-repos/web")).status_code == 204
        assert (await c.http.delete("/api/platform-repos/web")).status_code == 404

    run_app(app, body)


def test_workflow_rejects_unknown_platform_and_busy_ticket(database_url, tmp_path, bare_remote, run_app):
    github = FakeGithubClient(on_merge=_git_merge_hook(bare_remote, tmp_path))
    jira = FakeJiraClient({
        "PROJ-1": JiraTicket(key="PROJ-1", summary="Add a widget", description_html="", platform="web"),
        "PROJ-9": JiraTicket(key="PROJ-9", summary="x", description_html="", platform="no-such-platform"),
    })
    app = _workflow_app(database_url, tmp_path, jira, github)

    async def body(c):
        bad = await c.http.post("/api/workflows", json={"ticket_key": "PROJ-9"})
        assert bad.status_code == 400

        await c.http.post("/api/platform-repos", json={
            "platform": "web", "repo_owner": "acme", "repo_name": "webapp", "clone_url": bare_remote})

        r = await c.http.post("/api/workflows", json={"ticket_key": "PROJ-1"})
        assert r.status_code == 201
        workflow_id = r.json()["id"]

        busy = await c.http.post("/api/workflows", json={"ticket_key": "PROJ-1"})
        assert busy.status_code == 409

        # Drain the workflow to completion so the app's lifespan shutdown (wait_idle) is clean.
        # Iterate by explicit stage name (not "any pr_open stage") -- otherwise a check that
        # lands before a just-approved stage has transitioned to merged can re-find the same
        # stage twice, wasting a loop iteration and never reaching the last one.
        for stage_name in STAGES:
            async def pr_number(s=stage_name):
                detail = (await c.http.get(f"/api/workflows/{workflow_id}")).json()
                row = next((x for x in detail["stages"] if x["stage"] == s), None)
                return row["pr_number"] if row and row["status"] == "pr_open" else None
            number = await _wait_for(pr_number)
            github.approve(number)

        async def completed():
            detail = (await c.http.get(f"/api/workflows/{workflow_id}")).json()
            return detail["status"] == "completed"
        await _wait_for(completed)

    run_app(app, body)


def test_workflow_happy_path_over_http_and_event_replay(database_url, tmp_path, bare_remote, run_app):
    github = FakeGithubClient(on_merge=_git_merge_hook(bare_remote, tmp_path))
    jira = FakeJiraClient({"PROJ-1": JiraTicket(key="PROJ-1", summary="Add a widget",
                                                description_html="<p>desc</p>", platform="web")})
    app = _workflow_app(database_url, tmp_path, jira, github)

    async def body(c):
        await c.http.post("/api/platform-repos", json={
            "platform": "web", "repo_owner": "acme", "repo_name": "webapp", "clone_url": bare_remote})

        created = await c.http.post("/api/workflows", json={"ticket_key": "PROJ-1"})
        assert created.status_code == 201
        workflow = created.json()
        workflow_id = workflow["id"]
        assert workflow["platform"] == "web" and workflow["slug"] == "add-a-widget"

        seen_pr_numbers = []
        for stage in STAGES:
            async def pr_for_stage(s=stage):
                detail = (await c.http.get(f"/api/workflows/{workflow_id}")).json()
                row = next((x for x in detail["stages"] if x["stage"] == s), None)
                return row["pr_number"] if row and row["status"] == "pr_open" else None
            number = await _wait_for(pr_for_stage)
            seen_pr_numbers.append(number)
            github.approve(number)

        async def completed():
            detail = (await c.http.get(f"/api/workflows/{workflow_id}")).json()
            return detail if detail["status"] == "completed" else None
        final = await _wait_for(completed)

        assert [s["status"] for s in final["stages"]] == ["merged", "merged", "merged", "merged"]
        assert seen_pr_numbers == [1, 2, 3, 4]

        listed = await c.http.get("/api/workflows")
        assert [w["id"] for w in listed.json()] == [workflow_id]
        assert len(listed.json()[0]["stages"]) == 4

        # Workflow already finished: the events endpoint serves pure catch-up/replay, no live
        # subscribe -- a real GET that returns promptly, exercising the `id:`/`data:` SSE framing.
        events_resp = await c.http.get(f"/api/workflows/{workflow_id}/events")
        assert events_resp.status_code == 200
        frames = [f for f in events_resp.text.split("\n\n") if f.strip()]
        parsed = []
        for frame in frames:
            lines = frame.split("\n")
            id_line = next(l for l in lines if l.startswith("id: "))
            data_line = next(l for l in lines if l.startswith("data: "))
            parsed.append((int(id_line[len("id: "):]), json.loads(data_line[len("data: "):])))

        ids = [i for i, _ in parsed]
        assert ids == sorted(ids) and len(ids) == len(set(ids))
        types = [e["type"] for _, e in parsed]
        assert types[-1] == "workflow_completed"
        assert types.count("stage_pr_opened") == 4
        assert types.count("stage_merged") == 4

        # Last-Event-ID replay: asking for events after the second-to-last id returns only the tail.
        tail = await c.http.get(f"/api/workflows/{workflow_id}/events",
                                headers={"Last-Event-ID": str(ids[-2])})
        tail_frames = [f for f in tail.text.split("\n\n") if f.strip()]
        assert len(tail_frames) == 1

    run_app(app, body)
