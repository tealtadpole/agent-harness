import asyncio
import shutil
import sys
from pathlib import Path

import pytest
from conftest import HERE

from agent_harness.config import AgentConfig, ConfigError, CopilotConfig, McpServerConfig, load_config
from agent_harness.mcp_tools import McpTools
from agent_harness.providers.copilot import CopilotRuntime

ROOT = Path(__file__).parents[1]


def test_example_config_loads(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    shutil.copy(ROOT / "config.example.toml", tmp_path / "config.toml")
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.claude.default_model == "claude-opus-5-5"
    assert cfg.mcp_servers[0].tools == ("search_confluence", "get_confluence_page")
    assert "change-me" not in repr(cfg.database)


def test_database_url_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x@/y")
    (tmp_path / "c.toml").write_text("")
    assert load_config(tmp_path / "c.toml").database.url == "postgresql://x@/y"


def test_typos_are_reported(tmp_path):
    (tmp_path / "c.toml").write_text("[claude]\nmodel = 'x'\n")
    with pytest.raises(ConfigError, match="model"):
        load_config(tmp_path / "c.toml")


def test_tool_allow_list_filters_mcp_tools():
    server = McpServerConfig(name="rag", command=sys.executable,
                             args=(str(HERE / "fake_mcp_server.py"),), tools=("nothing_else",))

    async def run():
        mcp = McpTools((server,))
        await mcp.start()
        try:
            return mcp.tool_names
        finally:
            await mcp.close()

    assert asyncio.run(run()) == {"rag": []}


def test_copilot_403_is_reported_as_unavailable(tmp_path):
    class Auth:
        isAuthenticated, login = True, "someone"

    class Client:
        async def get_auth_status(self):
            return Auth()

        async def list_models(self):
            raise RuntimeError('Request models.list failed: {"status":403}')

    rt = CopilotRuntime(CopilotConfig(), AgentConfig(), (), {}, tmp_path)
    rt._client = Client()
    ok, detail, models = asyncio.run(rt.status())
    assert not ok and "403" in detail and "someone" in detail and models == []
