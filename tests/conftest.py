from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import pgserver
import psycopg
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from agent_harness.api import create_app
from agent_harness.config import load_config
from agent_harness.providers.claude import ClaudeProvider
from agent_harness.providers.copilot import CopilotProvider

HERE = Path(__file__).parent


# ---- Postgres ---------------------------------------------------------------

@pytest.fixture(scope="session")
def pg_server(tmp_path_factory):
    srv = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="stop")
    yield srv
    srv.cleanup()


@pytest.fixture
def database_url(pg_server):
    name = f"t_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(pg_server.get_uri(), autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {name}")
    return pg_server.get_uri(name)


# ---- Fake models --------------------------------------------------------------

class FakeToolModel(BaseChatModel):
    """Replays scripted AIMessages, streaming text word by word; records what it was sent."""

    responses: list[Any]   # AIMessage, or an Exception to raise
    seen: list[list] = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _next(self, messages) -> AIMessage:
        self.seen.append(list(messages))
        msg = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        if isinstance(msg, Exception):
            raise msg
        return msg

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._next(messages)
        words = msg.content.split(" ") if msg.content else []
        for i, word in enumerate(words):
            text = word if i == len(words) - 1 else word + " "
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk
        if msg.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(msg.tool_calls)]))


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="Let me check the wiki.",
                     tool_calls=[{"name": name, "args": args, "id": call_id}])


class FakeCopilotRuntime:
    """Mimics CopilotRuntime.run_turn: emits tokens and one tool call through `emit`."""

    def __init__(self):
        self.prompts: list[tuple[str, str, str]] = []
        self.deleted: list[str] = []
        self.hold: asyncio.Event | None = None

    async def status(self):
        return True, "Signed in as tester", ["gpt-test", "claude-test"]

    async def run_turn(self, session_id, model, prompt, emit):
        self.prompts.append((session_id, model, prompt))
        if self.hold:
            await self.hold.wait()
        emit({"type": "tool_start", "id": "cp1", "name": "search_confluence", "input": {"query": prompt}})
        emit({"type": "tool_end", "id": "cp1", "output": "25 days"})
        for word in ["Copilot", " says", " 25", " days."]:
            emit({"type": "token", "text": word})
        return "Copilot says 25 days."

    async def delete(self, session_id):
        self.deleted.append(session_id)

    async def close(self):
        pass


# ---- App -----------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path, database_url, monkeypatch):
    """Factory: (claude responses) -> (app, fake model, fake copilot runtime)."""
    monkeypatch.setenv("FAKE_RAG_SECRET", "s3cret")
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(f"""
[database]
url = "{database_url}"

[claude]
models = ["claude-test"]
default_model = "claude-test"

[mcp.servers.confluence]
command = "{sys.executable}"
args = ["{HERE / 'fake_mcp_server.py'}"]
env = {{ FAKE_RAG_SECRET = "${{FAKE_RAG_SECRET}}" }}
""")
    cfg = load_config(cfg_file)

    def make(responses: list[Any]):
        model = FakeToolModel(responses=responses, seen=[])
        copilot = FakeCopilotRuntime()

        def providers(cfg, mcp, checkpointer):
            return {
                "claude": ClaudeProvider(cfg.claude, cfg.agent, mcp.tools, checkpointer,
                                         model_factory=lambda name: model),
                "copilot": CopilotProvider(cfg.copilot, copilot, checkpointer),
            }

        return create_app(cfg, providers_factory=providers), model, copilot

    return make


class Client:
    def __init__(self, app):
        self.app = app
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                      base_url="http://127.0.0.1")

    async def chat(self, session_id: str, text: str) -> list[dict]:
        r = await self.http.post(f"/api/sessions/{session_id}/messages", json={"content": text})
        assert r.status_code == 200, r.text
        return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def run_app():
    """Run an async test body with the app's lifespan started."""
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
