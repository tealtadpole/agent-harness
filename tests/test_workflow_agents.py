"""Stage-agent construction and per-stage prompts, driven by the existing FakeToolModel."""

import asyncio

from conftest import FakeToolModel, tool_call
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_harness.workflow_agents import artifact_path, build_stage_agent, build_stage_prompt, stage_tools
from agent_harness.jira_client import JiraTicket

WORKFLOW = {"ticket_key": "PROJ-1", "ticket_summary": "Add a widget", "slug": "add-widget"}


def test_artifact_path_uses_ticket_key_and_slug():
    assert artifact_path(WORKFLOW, "spec") == "specs/proj-1-add-widget/spec.md"
    assert artifact_path(WORKFLOW, "plan") == "specs/proj-1-add-widget/plan.md"
    assert artifact_path(WORKFLOW, "tasks") == "specs/proj-1-add-widget/tasks.md"


def test_spec_prompt_includes_ticket_description():
    ticket = JiraTicket(key="PROJ-1", summary="Add a widget", description_html="<p>Needs a widget.</p>",
                        platform="web")
    prompt = build_stage_prompt("spec", WORKFLOW, ticket)
    assert "PROJ-1" in prompt and "Needs a widget" in prompt
    assert "specs/proj-1-add-widget/spec.md" in prompt


def test_plan_prompt_points_at_spec_file():
    prompt = build_stage_prompt("plan", WORKFLOW, ticket=None)
    assert "specs/proj-1-add-widget/spec.md" in prompt
    assert "specs/proj-1-add-widget/plan.md" in prompt


def test_spec_agent_writes_artifact_via_tool_call(tmp_path):
    model = FakeToolModel(responses=[
        tool_call("write_file", {"path": artifact_path(WORKFLOW, "spec"), "content": "# Spec\n"}),
        AIMessage("Done."),
    ], seen=[])
    agent = build_stage_agent("spec", model, tmp_path, InMemorySaver(), shell_timeout_seconds=30)

    async def run():
        config = {"configurable": {"thread_id": "wf:spec"}}
        prompt = build_stage_prompt("spec", WORKFLOW, ticket=None)
        return await agent.ainvoke({"messages": [HumanMessage(prompt)]}, config)

    result = asyncio.run(run())
    assert result["messages"][-1].content == "Done."
    written = tmp_path / "specs" / "proj-1-add-widget" / "spec.md"
    assert written.read_text() == "# Spec\n"


def test_only_implement_stage_gets_the_shell_tool(tmp_path):
    for stage in ("spec", "plan", "tasks"):
        names = {t.name for t in stage_tools(stage, tmp_path, shell_timeout_seconds=30)}
        assert names == {"read_file", "write_file", "list_dir", "search_files"}
    implement_names = {t.name for t in stage_tools("implement", tmp_path, shell_timeout_seconds=30)}
    assert implement_names == {"read_file", "write_file", "list_dir", "search_files", "run_shell"}
