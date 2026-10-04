"""Connect to the configured MCP servers (e.g. confluence-rag) and expose their tools.

Each server is started once and kept running for the life of the app. The adapter's
default would start a fresh server process for every tool call, which for a RAG server
means reloading the embedding model each time.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import AsyncExitStack

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .config import McpServerConfig

log = logging.getLogger(__name__)


class McpTools:
    def __init__(self, servers: tuple[McpServerConfig, ...]):
        self.servers = servers
        self.tools: list[BaseTool] = []
        self.tool_names: dict[str, list[str]] = {}   # server name -> its tool names
        self.errors: dict[str, str] = {}
        self._stack: AsyncExitStack | None = None

    async def start(self) -> None:
        self._stack = AsyncExitStack()
        client = MultiServerMCPClient({s.name: _connection(s) for s in self.servers})
        for server in self.servers:
            try:
                session = await self._stack.enter_async_context(client.session(server.name))
                tools = await load_mcp_tools(session, server_name=server.name)
            except Exception as e:   # one broken server shouldn't stop the harness
                self.errors[server.name] = f"{type(e).__name__}: {e}"
                log.error("MCP server %r failed to start: %s", server.name, e)
                continue
            if server.tools:
                if missing := set(server.tools) - {t.name for t in tools}:
                    log.warning("MCP server %r has no tool(s) %s", server.name, sorted(missing))
                tools = [t for t in tools if t.name in server.tools]
            self.tools.extend(tools)
            self.tool_names[server.name] = [t.name for t in tools]
            log.info("MCP server %r: %d tools (%s)", server.name, len(tools),
                     ", ".join(t.name for t in tools))

    async def close(self) -> None:
        if self._stack:
            await self._stack.aclose()

    def status(self) -> list[dict]:
        return [{"name": s.name, "ok": s.name in self.tool_names,
                 "tools": self.tool_names.get(s.name, []), "error": self.errors.get(s.name)}
                for s in self.servers]


def _connection(server: McpServerConfig) -> dict:
    conn = {"transport": "stdio", "command": server.command, "args": list(server.args)}
    if server.env:
        conn["env"] = dict(server.env)   # the adapter expands ${VAR} from our environment
    if server.cwd:
        conn["cwd"] = server.cwd
    return conn


def expand_env(env: dict[str, str]) -> dict[str, str]:
    """Expand ${VAR} references the same way the LangChain adapter does."""
    return {k: re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), v)
            for k, v in env.items()}
