"""GitHub Copilot through the official Copilot SDK.

Copilot runs its own agent loop, so this provider is a one-node LangGraph graph: the
node forwards the user's message to a Copilot session and streams Copilot's events
(tokens, tool calls) out through LangGraph's custom stream. LangGraph still
checkpoints the conversation, so history works the same as for Claude.

Safety: Copilot's built-in tools (shell, file edits, web) are switched off. Only the
configured MCP servers' tools are allowed, enforced twice: an `available_tools`
allow-list and a permission handler that rejects anything else.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path

from copilot import CopilotClient, MCPStdioServerConfig, ToolSet
from copilot.generated.session_events import PermissionRequestMcp
from copilot.rpc import PermissionDecisionApproveOnce, PermissionDecisionReject
from copilot.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    SessionErrorData,
    SessionIdleData,
    ToolExecutionCompleteData,
    ToolExecutionStartData,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Checkpointer

from ..config import AgentConfig, CopilotConfig, McpServerConfig
from ..mcp_tools import expand_env
from .base import ProviderStatus

log = logging.getLogger(__name__)

TURN_TIMEOUT_SECONDS = 600
INHERITED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "XDG_CACHE_HOME")

Emit = Callable[[dict | None], None]


class CopilotRuntime:
    """Owns the Copilot SDK client (which runs the Copilot runtime process) and live sessions."""

    def __init__(self, cfg: CopilotConfig, agent: AgentConfig,
                 mcp_servers: tuple[McpServerConfig, ...], tool_names: dict[str, list[str]],
                 workdir: Path):
        self.cfg = cfg
        self.agent_cfg = agent
        self.mcp_servers = [s for s in mcp_servers if tool_names.get(s.name)]
        self.tool_names = tool_names
        self.workdir = workdir
        self._client: CopilotClient | None = None
        self._client_lock = asyncio.Lock()
        self._sessions: dict[str, object] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

    async def client(self) -> CopilotClient:
        async with self._client_lock:
            if self._client is None:
                kwargs: dict = {"working_directory": str(self.workdir)}
                if token := self.cfg.github_token:
                    kwargs["github_token"] = token
                self.workdir.mkdir(parents=True, exist_ok=True)
                client = CopilotClient(**kwargs)
                await client.start()   # downloads the Copilot runtime on first use
                self._client = client
            return self._client

    async def close(self) -> None:
        for session in self._sessions.values():
            try:
                await session.disconnect()
            except Exception:
                pass
        if self._client is not None:
            await self._client.stop()

    async def status(self) -> tuple[bool, str, list[str]]:
        try:
            client = await self.client()
            auth = await client.get_auth_status()
        except Exception as e:
            return False, f"Copilot runtime unavailable: {e}", []
        if not auth.isAuthenticated:
            return False, ("Not signed in to GitHub. Run `gh auth login` (GitHub CLI) "
                           f"or set ${self.cfg.github_token_env}."), []
        who = f"Signed in as {auth.login}" if auth.login else "Signed in"
        # Listing models is also the cheapest check that this account may use Copilot here:
        # a signed-in account without the right Copilot plan gets 403 on every request.
        try:
            available = [m.id for m in await client.list_models()]
        except Exception as e:
            reason = "403: this GitHub account is not entitled to use Copilot through the SDK" \
                if "403" in str(e) else str(e)[:300]
            return False, f"{who}, but Copilot refused the request ({reason}).", []
        return True, who, list(self.cfg.models) or available

    async def run_turn(self, session_id: str, model: str, prompt: str, emit: Emit) -> str:
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            session = await self._session(session_id, model)
            done = asyncio.Event()
            final: list[str] = []
            error: list[str] = []
            loop = asyncio.get_running_loop()

            def on_event(event) -> None:
                data = event.data
                match data:
                    case AssistantMessageDeltaData():
                        if data.delta_content:
                            emit({"type": "token", "text": data.delta_content})
                    case AssistantMessageData():
                        if data.content:
                            final.append(data.content)
                    case ToolExecutionStartData():
                        emit({"type": "tool_start", "id": data.tool_call_id,
                              "name": data.mcp_tool_name or data.tool_name,
                              "input": data.arguments})
                    case ToolExecutionCompleteData():
                        emit({"type": "tool_end", "id": data.tool_call_id,
                              "output": _tool_output(data)})
                    case SessionErrorData():
                        error.append(data.message or "Copilot session error")
                        loop.call_soon_threadsafe(done.set)
                    case SessionIdleData():
                        loop.call_soon_threadsafe(done.set)

            unsubscribe = session.on(on_event)
            try:
                await session.send(prompt)
                await asyncio.wait_for(done.wait(), TURN_TIMEOUT_SECONDS)
            except TimeoutError:
                await session.abort()
                raise RuntimeError(f"Copilot did not finish within {TURN_TIMEOUT_SECONDS}s")
            finally:
                unsubscribe()
            if error:
                raise RuntimeError(error[0])
            return "\n\n".join(final)

    async def delete(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        self._session_locks.pop(session_id, None)
        if self._client is None:
            return
        try:
            if session is not None:
                await session.disconnect()
            await self._client.delete_session(session_id)
        except Exception as e:
            log.info("Copilot session %s not deleted (%s)", session_id, e)

    async def _session(self, session_id: str, model: str):
        if session_id in self._sessions:
            return self._sessions[session_id]
        client = await self.client()
        options = self._session_options(model)
        try:
            session = await client.resume_session(session_id, **options)
        except Exception:
            session = await client.create_session(session_id=session_id, **options)
        self._sessions[session_id] = session
        return session

    def _session_options(self, model: str) -> dict:
        allowed = ToolSet()
        for server, names in self.tool_names.items():
            for name in names:
                allowed.add_mcp(f"{server}-{name}")
        options = {
            "on_permission_request": self._permission,
            "streaming": True,
            "system_message": {"mode": "append", "content": self.agent_cfg.system_prompt},
            "mcp_servers": {s.name: self._mcp_config(s) for s in self.mcp_servers},
            "available_tools": allowed.to_list(),
            "working_directory": str(self.workdir),
            "enable_config_discovery": False,   # don't pick up the user's own Copilot MCP config
            "skip_custom_instructions": True,
            "enable_skills": False,
            "enable_session_store": False,
        }
        if model:
            options["model"] = model
        if self.cfg.reasoning_effort:
            options["reasoning_effort"] = self.cfg.reasoning_effort
        return options

    def _mcp_config(self, server: McpServerConfig) -> MCPStdioServerConfig:
        env = {k: os.environ[k] for k in INHERITED_ENV if k in os.environ}
        env.update(expand_env(server.env))
        config: MCPStdioServerConfig = {"type": "stdio", "command": server.command,
                                        "args": list(server.args), "env": env, "tools": ["*"]}
        if server.cwd:
            config["working_directory"] = server.cwd
        return config

    def _permission(self, request, invocation):
        if (isinstance(request, PermissionRequestMcp)
                and request.server_name in self.tool_names
                and request.tool_name in self.tool_names[request.server_name]
                and not getattr(request, "managed_approval_required", False)):
            return PermissionDecisionApproveOnce()
        log.warning("Rejected Copilot tool request: %s", type(request).__name__)
        return PermissionDecisionReject(feedback="Only the configured MCP tools are allowed here.")


class CopilotProvider:
    name = "copilot"
    label = "GitHub Copilot"

    def __init__(self, cfg: CopilotConfig, runtime: CopilotRuntime, checkpointer: Checkpointer):
        self.cfg = cfg
        self.runtime = runtime
        self._graph = build_copilot_graph(runtime, checkpointer)
        self._status_cache: tuple[float, ProviderStatus] | None = None

    async def status(self) -> ProviderStatus:
        loop = asyncio.get_running_loop()
        if self._status_cache and loop.time() - self._status_cache[0] < 60:
            return self._status_cache[1]
        ok, detail, models = await self.runtime.status()
        status = ProviderStatus(name=self.name, label=self.label, available=ok, detail=detail,
                                models=models, default_model=self.cfg.default_model
                                or (models[0] if models else ""))
        self._status_cache = (loop.time(), status)
        return status

    def graph(self, model: str):
        return self._graph


def build_copilot_graph(runtime: CopilotRuntime, checkpointer: Checkpointer):
    async def copilot_turn(state: MessagesState, config: RunnableConfig) -> dict:
        writer = get_stream_writer()
        configurable = config["configurable"]
        prompt = state["messages"][-1].text
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def emit(event: dict | None) -> None:   # may be called from SDK callbacks
            loop.call_soon_threadsafe(queue.put_nowait, event)

        async def run() -> str:
            try:
                return await runtime.run_turn(configurable["thread_id"],
                                              configurable.get("model", ""), prompt, emit)
            finally:
                emit(None)

        task = asyncio.create_task(run())
        while (event := await queue.get()) is not None:
            writer(event)
        return {"messages": [AIMessage(content=await task)]}

    graph = StateGraph(MessagesState)
    graph.add_node("copilot", copilot_turn)
    graph.add_edge(START, "copilot")
    return graph.compile(checkpointer=checkpointer, name="copilot")


def _tool_output(data: ToolExecutionCompleteData) -> str:
    if not data.success:
        return f"Error: {data.error}"
    result = data.result
    if result is None:
        return ""
    return result.content or result.detailed_content or ""
