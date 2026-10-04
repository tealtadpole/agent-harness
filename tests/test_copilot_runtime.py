"""Copilot lockdown: only the configured MCP tools may run."""

import dataclasses

from pathlib import Path

from copilot.generated.session_events import PermissionRequestMcp, PermissionRequestShell
from copilot.rpc import PermissionDecisionApproveOnce, PermissionDecisionReject

from agent_harness.config import AgentConfig, CopilotConfig, McpServerConfig
from agent_harness.providers.copilot import CopilotRuntime


def runtime(tmp_path: Path) -> CopilotRuntime:
    servers = (McpServerConfig(name="confluence", command="/bin/rag", args=("serve",),
                               env={"CONFLUENCE_PAT": "${TEST_PAT}"}),
               McpServerConfig(name="broken", command="/bin/nope"))
    return CopilotRuntime(CopilotConfig(), AgentConfig(), servers,
                          {"confluence": ["search_confluence", "get_confluence_page"]}, tmp_path)


def blank(cls, **values):
    """An SDK request object with every settable field None except `values`."""
    fields = {f.name: None for f in dataclasses.fields(cls) if f.init}
    return cls(**{**fields, **values})


def mcp_request(server, tool):
    return blank(PermissionRequestMcp, server_name=server, tool_name=tool, tool_title=tool,
                 read_only=True, managed_approval_required=False)


def test_session_options_allow_only_configured_mcp_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PAT", "pat-123")
    opts = runtime(tmp_path)._session_options("gpt-test")
    assert opts["model"] == "gpt-test"
    assert sorted(opts["available_tools"]) == ["mcp:confluence-get_confluence_page",
                                              "mcp:confluence-search_confluence"]
    assert list(opts["mcp_servers"]) == ["confluence"]          # failed servers are left out
    server = opts["mcp_servers"]["confluence"]
    assert server["command"] == "/bin/rag" and server["env"]["CONFLUENCE_PAT"] == "pat-123"
    assert opts["enable_config_discovery"] is False and opts["enable_skills"] is False
    assert opts["system_message"]["mode"] == "append"           # keeps Copilot's own guardrails


def test_permission_handler(tmp_path):
    rt = runtime(tmp_path)
    allow = rt._permission(mcp_request("confluence", "search_confluence"), {})
    assert isinstance(allow, PermissionDecisionApproveOnce)
    assert isinstance(rt._permission(mcp_request("confluence", "delete_everything"), {}),
                      PermissionDecisionReject)
    assert isinstance(rt._permission(mcp_request("other", "search_confluence"), {}),
                      PermissionDecisionReject)
    assert isinstance(rt._permission(blank(PermissionRequestShell), {}), PermissionDecisionReject)
