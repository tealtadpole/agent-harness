"""Drive a JIRA ticket through spec -> plan -> tasks -> implement, each stage its own
branch + GitHub PR, auto-merged once approved, before the next stage starts.

Modeled on `runner.TurnRunner`: one long-running `asyncio.Task` per workflow id. "Wait for
approval" is just one more `await` inside that task -- there is no separate poller/scheduler.
`resume_in_flight()` re-spawns a task for every non-terminal `workflows` row after an app
restart; every stage step is idempotent on resume via `workflow_stages.status`.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.types import Checkpointer

from . import git_ops
from .config import GithubConfig, WorkflowConfig
from .github_client import GithubClient
from .jira_client import JiraClient, JiraError, JiraTicket
from .workflow_agents import build_stage_agent, build_stage_prompt
from .workflow_db import WorkflowDatabase

log = logging.getLogger(__name__)

STAGES = ("spec", "plan", "tasks", "implement")
OPENING_STATUSES = ("pending", "running")   # stage states that mean "(re)do the open-PR sequence"


class WorkflowBusy(Exception):
    pass


class UnknownPlatform(Exception):
    pass


class WorkflowRunner:
    def __init__(self, db: WorkflowDatabase, jira: JiraClient, github: GithubClient,
                 model_factory: Callable[[], BaseChatModel], checkpointer: Checkpointer,
                 workflow_cfg: WorkflowConfig, github_cfg: GithubConfig):
        self.db = db
        self.jira = jira
        self.github = github
        self.model_factory = model_factory
        self.checkpointer = checkpointer
        self.workflow_cfg = workflow_cfg
        self.github_cfg = github_cfg
        self._workdir_root = Path(workflow_cfg.workdir) if workflow_cfg.workdir else (
            Path.home() / ".cache" / "agent-harness" / "workflows")
        self._active: dict[str, asyncio.Task] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def is_running(self, workflow_id: str) -> bool:
        task = self._active.get(workflow_id)
        return task is not None and not task.done()

    async def wait_idle(self) -> None:
        tasks = [t for t in self._active.values() if not t.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=30)

    async def resume_in_flight(self) -> None:
        for workflow in await self.db.list_active_workflows():
            ticket: JiraTicket | None = None
            try:
                ticket = await self.jira.get_ticket(workflow["ticket_key"])
            except JiraError as e:
                log.warning("Could not refetch ticket %s on resume: %s", workflow["ticket_key"], e)
            self._spawn(workflow, ticket)

    async def start_new(self, ticket_key: str) -> dict:
        if await self.db.find_active_workflow_for_ticket(ticket_key):
            raise WorkflowBusy(f"Ticket {ticket_key} already has an active workflow.")
        ticket = await self.jira.get_ticket(ticket_key)
        repo = await self.db.get_platform_repo(ticket.platform)
        if repo is None:
            raise UnknownPlatform(f"No repo mapped for platform {ticket.platform!r}. "
                                  "Add one under Platform Repos first.")
        slug = _slugify(ticket.summary)
        workflow = await self.db.create_workflow(
            ticket_key, ticket.summary, ticket.platform, repo["repo_owner"], repo["repo_name"],
            repo["base_branch"], repo["clone_url"], slug)
        self._spawn(workflow, ticket)
        return workflow

    def subscribe(self, workflow_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(workflow_id, []).append(queue)
        return queue

    def unsubscribe(self, workflow_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(workflow_id)
        if subs and queue in subs:
            subs.remove(queue)
            if not subs:
                self._subscribers.pop(workflow_id, None)

    async def events_since(self, workflow_id: str, after_id: int) -> list[dict]:
        return await self.db.list_events(workflow_id, after_id)

    def _spawn(self, workflow: dict, ticket: JiraTicket | None) -> None:
        self._active[workflow["id"]] = asyncio.create_task(self._run(workflow, ticket))

    async def _emit(self, workflow_id: str, type: str, *, stage: str | None = None,
                    payload: dict | None = None) -> None:
        event = await self.db.add_event(workflow_id, type, stage=stage, payload=payload)
        for queue in self._subscribers.get(workflow_id, []):
            queue.put_nowait(event)

    def _close_subscribers(self, workflow_id: str) -> None:
        for queue in self._subscribers.pop(workflow_id, []):
            queue.put_nowait(None)

    async def _run(self, workflow: dict, ticket: JiraTicket | None) -> None:
        workflow_id = workflow["id"]
        try:
            await self.db.update_workflow_status(workflow_id, status="running")
            repo_dir = self._workdir_root / workflow_id / "repo"
            token = self.github_cfg.token or None
            await git_ops.ensure_repo(repo_dir, workflow["clone_url"], token)

            start_index = STAGES.index(workflow["current_stage"]) \
                if workflow["current_stage"] in STAGES else 0
            for stage in STAGES[start_index:]:
                await self.db.update_workflow_status(workflow_id, current_stage=stage)
                await self._run_stage(workflow, ticket, repo_dir, stage)

            await self.db.update_workflow_status(workflow_id, status="completed", current_stage="done")
            await self._emit(workflow_id, "workflow_completed")
        except Exception as e:
            log.exception("Workflow %s failed", workflow_id)
            await self.db.update_workflow_status(workflow_id, status="failed", error=str(e))
            await self._emit(workflow_id, "workflow_failed", payload={"text": str(e)})
        finally:
            self._close_subscribers(workflow_id)

    async def _run_stage(self, workflow: dict, ticket: JiraTicket | None, repo_dir: Path,
                         stage: str) -> None:
        workflow_id = workflow["id"]
        token = self.github_cfg.token or None
        branch = f"{workflow['ticket_key'].lower()}-{workflow['slug']}-{stage}"
        existing = await self.db.get_stage(workflow_id, stage)

        if existing is None or existing["status"] in OPENING_STATUSES:
            if existing is None:
                existing = await self.db.create_stage(workflow_id, stage, branch)
            await self._emit(workflow_id, "stage_start", stage=stage)

            await git_ops.checkout_new_branch(repo_dir, workflow["base_branch"], branch, token)
            agent = build_stage_agent(stage, self.model_factory(), repo_dir, self.checkpointer,
                                      self.workflow_cfg.shell_timeout_seconds)
            prompt = build_stage_prompt(stage, workflow, ticket)
            config = {"configurable": {"thread_id": f"{workflow_id}:{stage}"},
                     "recursion_limit": self.workflow_cfg.recursion_limit}
            result = await agent.ainvoke({"messages": [HumanMessage(prompt)]}, config)
            final_text = result["messages"][-1].text if result.get("messages") else ""
            await self._emit(workflow_id, "agent_text", stage=stage, payload={"text": final_text})

            changed = await git_ops.commit_all(
                repo_dir, f"{stage}: {workflow['ticket_summary']}",
                self.workflow_cfg.git_user_name, self.workflow_cfg.git_user_email)
            if not changed:
                raise RuntimeError(f"The {stage} stage made no file changes to commit.")
            await git_ops.push_branch(repo_dir, branch, token)

            pr = await self.github.create_pull_request(
                workflow["repo_owner"], workflow["repo_name"], branch, workflow["base_branch"],
                title=f"{stage}: {workflow['ticket_key']} {workflow['ticket_summary']}",
                body=f"Automated {stage} for {workflow['ticket_key']}, opened by agent-harness.")
            await self.db.update_stage(workflow_id, stage, status="pr_open", pr_number=pr.number,
                                       pr_url=pr.html_url)
            await self.db.update_workflow_status(workflow_id, status="awaiting_approval")
            await self._emit(workflow_id, "stage_pr_opened", stage=stage,
                            payload={"pr_number": pr.number, "pr_url": pr.html_url})
            existing = await self.db.get_stage(workflow_id, stage)

        if existing["status"] == "pr_open":
            await self._await_approval(workflow, stage, existing["pr_number"])
            existing = await self.db.update_stage(workflow_id, stage, status="approved")

        if existing["status"] == "approved":
            await self.db.update_workflow_status(workflow_id, status="merging")
            await self.github.merge_pull_request(
                workflow["repo_owner"], workflow["repo_name"], existing["pr_number"],
                self.github_cfg.merge_method)
            await self.db.update_stage(workflow_id, stage, status="merged",
                                       finished_at=datetime.now(timezone.utc))
            await self._emit(workflow_id, "stage_merged", stage=stage,
                            payload={"pr_number": existing["pr_number"]})
            await self.db.update_workflow_status(workflow_id, status="advancing")

    async def _await_approval(self, workflow: dict, stage: str, pr_number: int) -> None:
        workflow_id = workflow["id"]
        owner, repo = workflow["repo_owner"], workflow["repo_name"]
        while True:
            await asyncio.sleep(self.github_cfg.poll_interval_seconds)
            pr = await self.github.get_pull_request(owner, repo, pr_number)
            if pr.state == "closed" and not pr.merged:
                raise RuntimeError(f"PR #{pr_number} for the {stage} stage was closed without merging.")
            reviews = await self.github.list_review_states(owner, repo, pr_number)
            if self.github.is_approved(reviews):
                await self._emit(workflow_id, "stage_approved", stage=stage,
                                payload={"pr_number": pr_number})
                return


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = slug[:max_len].rstrip("-")
    return slug or "ticket"
