"""FastAPI app: JSON API, streaming chat (Server-Sent Events), and the built React UI."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field

from .config import Config
from .db import Database, make_pool
from .github_client import GithubClient
from .jira_client import JiraClient
from .mcp_tools import McpTools
from .providers.claude import ClaudeProvider
from .providers.copilot import CopilotProvider, CopilotRuntime
from .providers.llama import LlamaProvider
from .runner import TurnBusy, TurnRunner
from .workflow_agents import default_workflow_model_factory
from .workflow_db import WorkflowDatabase
from .workflow_runner import UnknownPlatform, WorkflowBusy, WorkflowRunner

log = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@dataclass
class AppState:
    db: Database
    providers: dict
    mcp: McpTools
    runner: TurnRunner
    workflow_db: WorkflowDatabase
    jira: JiraClient
    github: GithubClient
    workflow_runner: WorkflowRunner | None


class NewSession(BaseModel):
    provider: str
    model: str = ""
    title: str = Field(default="New chat", max_length=200)


class RenameSession(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class NewMessage(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)


class NewPlatformRepo(BaseModel):
    platform: str = Field(min_length=1, max_length=100)
    repo_owner: str = Field(min_length=1, max_length=200)
    repo_name: str = Field(min_length=1, max_length=200)
    base_branch: str = Field(default="main", min_length=1, max_length=200)
    clone_url: str = Field(min_length=1, max_length=500)


class NewWorkflow(BaseModel):
    ticket_key: str = Field(min_length=1, max_length=50)


ProvidersFactory = Callable[[Config, McpTools, AsyncPostgresSaver], dict]
JiraFactory = Callable[[Config], JiraClient]
GithubFactory = Callable[[Config], GithubClient]
WorkflowModelFactory = Callable[[Config], Callable]


def default_providers(cfg: Config, mcp: McpTools, checkpointer: AsyncPostgresSaver) -> dict:
    providers: dict = {}
    if cfg.claude.enabled:
        providers["claude"] = ClaudeProvider(cfg.claude, cfg.agent, mcp.tools, checkpointer)
    if cfg.copilot.enabled:
        runtime = CopilotRuntime(cfg.copilot, cfg.agent, cfg.mcp_servers, mcp.tool_names,
                                 workdir=Path.home() / ".cache" / "agent-harness" / "copilot-workdir")
        providers["copilot"] = CopilotProvider(cfg.copilot, runtime, checkpointer)
    if cfg.llama.enabled:
        providers["llama"] = LlamaProvider(cfg.llama, cfg.agent, mcp.tools, checkpointer)
    return providers


def default_jira_client(cfg: Config) -> JiraClient:
    return JiraClient(cfg.jira)


def default_github_client(cfg: Config) -> GithubClient:
    return GithubClient(cfg.github)


def create_app(cfg: Config, providers_factory: ProvidersFactory = default_providers,
              jira_factory: JiraFactory = default_jira_client,
              github_factory: GithubFactory = default_github_client,
              workflow_model_factory: WorkflowModelFactory = default_workflow_model_factory) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = make_pool(cfg.database.url)
        await pool.open(wait=True, timeout=15)
        db = Database(pool)
        await db.setup()
        workflow_db = WorkflowDatabase(pool)
        await workflow_db.setup()
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        mcp = McpTools(cfg.mcp_servers)
        await mcp.start()
        providers = providers_factory(cfg, mcp, checkpointer)
        runner = TurnRunner(db, providers, cfg.agent.recursion_limit)
        jira = jira_factory(cfg)
        github = github_factory(cfg)
        workflow_runner = None
        if cfg.workflow.enabled:
            workflow_runner = WorkflowRunner(workflow_db, jira, github,
                                             workflow_model_factory(cfg), checkpointer,
                                             cfg.workflow, cfg.github)
            await workflow_runner.resume_in_flight()
        app.state.harness = AppState(db=db, providers=providers, mcp=mcp, runner=runner,
                                     workflow_db=workflow_db, jira=jira, github=github,
                                     workflow_runner=workflow_runner)
        app.state.checkpointer = checkpointer
        try:
            yield
        finally:
            await runner.wait_idle()
            if workflow_runner:
                await workflow_runner.wait_idle()
            for p in providers.values():
                if runtime := getattr(p, "runtime", None):
                    await runtime.close()
            await mcp.close()
            await pool.close()

    app = FastAPI(title="Agent Harness", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(cfg.server.allowed_hosts))

    @app.middleware("http")
    async def same_origin_only(request: Request, call_next):
        # Browsers send Origin on cross-site requests; refuse any that come from another site,
        # so a web page you visit can't use this local API (and your keys) on your behalf.
        origin = request.headers.get("origin")
        if origin and request.url.path.startswith("/api/") \
                and urlsplit(origin).hostname not in cfg.server.allowed_hosts:
            return JSONResponse({"detail": "Cross-origin request refused"}, status_code=403)
        return await call_next(request)

    def state() -> AppState:
        return app.state.harness

    async def session_or_404(session_id: str) -> dict:
        try:
            session = await state().db.get_session(session_id)
        except KeyError:
            session = None
        if session is None:
            raise HTTPException(404, "Chat not found")
        return session

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.get("/api/providers")
    async def providers():
        statuses = [(await p.status()).to_dict() for p in state().providers.values()]
        return {"providers": statuses, "mcp_servers": state().mcp.status()}

    @app.get("/api/sessions")
    async def list_sessions():
        return await state().db.list_sessions()

    @app.post("/api/sessions", status_code=201)
    async def create_session(body: NewSession):
        provider = state().providers.get(body.provider)
        if provider is None:
            raise HTTPException(400, f"Unknown or disabled provider {body.provider!r}")
        status = await provider.status()
        if not status.available:
            raise HTTPException(400, status.detail or f"{status.label} is not available")
        model = body.model or status.default_model
        if status.models and model not in status.models:
            raise HTTPException(400, f"Unknown model {model!r} for {body.provider}")
        return await state().db.create_session(body.provider, model, body.title)

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        session = await session_or_404(session_id)
        return {**session, "messages": await state().db.list_messages(session_id),
                "running": state().runner.is_running(session_id)}

    @app.patch("/api/sessions/{session_id}")
    async def rename_session(session_id: str, body: RenameSession):
        await session_or_404(session_id)
        return await state().db.rename_session(session_id, body.title.strip())

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        session = await session_or_404(session_id)
        if state().runner.is_running(session_id):
            raise HTTPException(409, "Wait for the current reply to finish before deleting.")
        await state().db.delete_session(session_id)
        await app.state.checkpointer.adelete_thread(session_id)
        provider = state().providers.get(session["provider"])
        if runtime := getattr(provider, "runtime", None):
            await runtime.delete(session_id)

    @app.post("/api/sessions/{session_id}/messages")
    async def send_message(session_id: str, body: NewMessage):
        session = await session_or_404(session_id)
        if session["provider"] not in state().providers:
            raise HTTPException(400, f"Provider {session['provider']!r} is disabled in config")
        if state().runner.is_running(session_id):
            raise HTTPException(409, "A reply is still being generated for this chat.")

        async def sse():
            try:
                async for event in state().runner.start(session, body.content):
                    yield f"data: {json.dumps(event, default=str)}\n\n"
            except TurnBusy as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/platform-repos")
    async def list_platform_repos():
        return await state().workflow_db.list_platform_repos()

    @app.post("/api/platform-repos", status_code=201)
    async def upsert_platform_repo(body: NewPlatformRepo):
        return await state().workflow_db.upsert_platform_repo(
            body.platform, body.repo_owner, body.repo_name, body.base_branch, body.clone_url)

    @app.delete("/api/platform-repos/{platform}", status_code=204)
    async def delete_platform_repo(platform: str):
        if not await state().workflow_db.delete_platform_repo(platform):
            raise HTTPException(404, "No repo mapped for that platform")

    @app.get("/api/workflows")
    async def list_workflows():
        workflows = await state().workflow_db.list_workflows()
        for w in workflows:
            w["stages"] = await state().workflow_db.list_stages(w["id"])
        return workflows

    @app.post("/api/workflows", status_code=201)
    async def create_workflow(body: NewWorkflow):
        runner = state().workflow_runner
        if runner is None:
            raise HTTPException(400, "[workflow] is disabled in config")
        try:
            return await runner.start_new(body.ticket_key)
        except WorkflowBusy as e:
            raise HTTPException(409, str(e)) from e
        except UnknownPlatform as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/workflows/{workflow_id}")
    async def get_workflow(workflow_id: str):
        workflow = await state().workflow_db.get_workflow(workflow_id)
        if workflow is None:
            raise HTTPException(404, "Workflow not found")
        return {**workflow, "stages": await state().workflow_db.list_stages(workflow_id)}

    @app.get("/api/workflows/{workflow_id}/events")
    async def workflow_events(workflow_id: str, request: Request, since: int = 0):
        if await state().workflow_db.get_workflow(workflow_id) is None:
            raise HTTPException(404, "Workflow not found")
        runner = state().workflow_runner
        last_event_id = request.headers.get("last-event-id")
        after = int(last_event_id) if last_event_id else since

        async def sse():
            for row in await state().workflow_db.list_events(workflow_id, after):
                yield f"id: {row['id']}\ndata: {json.dumps(row, default=str)}\n\n"
            if runner is None or not runner.is_running(workflow_id):
                return
            queue = runner.subscribe(workflow_id)
            try:
                while (event := await queue.get()) is not None:
                    yield f"id: {event['id']}\ndata: {json.dumps(event, default=str)}\n\n"
            finally:
                runner.unsubscribe(workflow_id, queue)

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if FRONTEND_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            file = (FRONTEND_DIST / path).resolve()
            if path and file.is_file() and file.is_relative_to(FRONTEND_DIST):
                return FileResponse(file)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app
