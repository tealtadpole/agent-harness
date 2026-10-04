"""FastAPI app: JSON API, streaming chat (Server-Sent Events), and the built React UI."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field

from .config import Config
from .db import Database, make_pool
from .mcp_tools import McpTools
from .providers.claude import ClaudeProvider
from .providers.copilot import CopilotProvider, CopilotRuntime
from .runner import TurnBusy, TurnRunner

log = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@dataclass
class AppState:
    db: Database
    providers: dict
    mcp: McpTools
    runner: TurnRunner


class NewSession(BaseModel):
    provider: str
    model: str = ""
    title: str = Field(default="New chat", max_length=200)


class RenameSession(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class NewMessage(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)


ProvidersFactory = Callable[[Config, McpTools, AsyncPostgresSaver], dict]


def default_providers(cfg: Config, mcp: McpTools, checkpointer: AsyncPostgresSaver) -> dict:
    providers: dict = {}
    if cfg.claude.enabled:
        providers["claude"] = ClaudeProvider(cfg.claude, cfg.agent, mcp.tools, checkpointer)
    if cfg.copilot.enabled:
        runtime = CopilotRuntime(cfg.copilot, cfg.agent, cfg.mcp_servers, mcp.tool_names,
                                 workdir=Path.home() / ".cache" / "agent-harness" / "copilot-workdir")
        providers["copilot"] = CopilotProvider(cfg.copilot, runtime, checkpointer)
    return providers


def create_app(cfg: Config, providers_factory: ProvidersFactory = default_providers) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = make_pool(cfg.database.url)
        await pool.open(wait=True, timeout=15)
        db = Database(pool)
        await db.setup()
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        mcp = McpTools(cfg.mcp_servers)
        await mcp.start()
        providers = providers_factory(cfg, mcp, checkpointer)
        runner = TurnRunner(db, providers, cfg.agent.recursion_limit)
        app.state.harness = AppState(db=db, providers=providers, mcp=mcp, runner=runner)
        app.state.checkpointer = checkpointer
        try:
            yield
        finally:
            await runner.wait_idle()
            for p in providers.values():
                if runtime := getattr(p, "runtime", None):
                    await runtime.close()
            await mcp.close()
            await pool.close()

    app = FastAPI(title="Agent Harness", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(cfg.server.allowed_hosts))

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
