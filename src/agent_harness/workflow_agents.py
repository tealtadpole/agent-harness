"""Per-stage system prompts and agent construction for the JIRA-to-PR workflow pipeline.

Each stage is a `langchain.agents.create_agent` tool-calling agent, same as `ClaudeProvider`
(see `providers/claude.py`), just with a different system prompt and tool set, and a
`thread_id` of `f"{workflow_id}:{stage}"` on the shared checkpointer.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langgraph.types import Checkpointer

from .config import Config
from .jira_client import JiraTicket
from .workflow_tools import make_fs_tools, make_shell_tool

SPEC_SYSTEM_PROMPT = (
    "You are a senior engineer writing a feature spec from a JIRA ticket, following GitHub "
    "spec-kit conventions. Write a single markdown file at the path you are given, with these "
    "sections: ## Overview, ## User Scenarios, ## Functional Requirements, "
    "## Non-functional Requirements, ## Out of Scope, ## Open Questions. Read the repository "
    "first (list_dir, read_file, search_files) so the spec reflects how the codebase actually "
    "works. Be concrete and specific; do not pad with filler. When the file is written, stop."
)

PLAN_SYSTEM_PROMPT = (
    "You are a senior engineer turning an approved spec into a technical plan, following "
    "GitHub spec-kit conventions. Read the spec file you are given, and the relevant parts of "
    "the repository, then write a single markdown file at the given plan path with sections: "
    "## Architecture, ## File-Level Changes (a concrete list of files to add/modify and what "
    "changes in each), ## Risks. Be concrete: name real files and functions you found in the "
    "repository, not hypothetical ones. When the file is written, stop."
)

TASKS_SYSTEM_PROMPT = (
    "You are a senior engineer turning an approved technical plan into an ordered task list, "
    "following GitHub spec-kit conventions. Read the plan file you are given, then write a "
    "single markdown file at the given tasks path: a numbered, dependency-ordered checklist, "
    "one markdown checkbox per task (`- [ ] T1: ...`), each task small enough to implement and "
    "verify independently. When the file is written, stop."
)

IMPLEMENT_SYSTEM_PROMPT = (
    "You are a senior engineer implementing an approved task list in a real repository "
    "checkout. Read the tasks file you are given, then work through the unchecked tasks in "
    "order: make the code change for a task, run relevant build/test commands with run_shell "
    "and fix failures, then edit the tasks file to check that task off (`- [x]`) before moving "
    "to the next. Only use the tools you have been given; do not ask the user anything, since "
    "no one is watching interactively. When every task is checked off, stop."
)

STAGE_PROMPTS = {"spec": SPEC_SYSTEM_PROMPT, "plan": PLAN_SYSTEM_PROMPT,
                 "tasks": TASKS_SYSTEM_PROMPT, "implement": IMPLEMENT_SYSTEM_PROMPT}


def spec_dir_name(workflow: dict) -> str:
    return f"{workflow['ticket_key'].lower()}-{workflow['slug']}"


def artifact_path(workflow: dict, stage: str) -> str:
    name = {"spec": "spec.md", "plan": "plan.md", "tasks": "tasks.md"}[stage]
    return f"specs/{spec_dir_name(workflow)}/{name}"


def stage_tools(stage: str, repo_dir: Path, shell_timeout_seconds: int) -> list:
    tools = make_fs_tools(repo_dir)
    if stage == "implement":
        tools = [*tools, make_shell_tool(repo_dir, shell_timeout_seconds)]
    return tools


def build_stage_agent(stage: str, model: BaseChatModel, repo_dir: Path,
                      checkpointer: Checkpointer, shell_timeout_seconds: int):
    tools = stage_tools(stage, repo_dir, shell_timeout_seconds)
    return create_agent(model=model, tools=tools, system_prompt=STAGE_PROMPTS[stage],
                        checkpointer=checkpointer, name=f"workflow:{stage}")


def build_stage_prompt(stage: str, workflow: dict, ticket: JiraTicket | None) -> str:
    if stage == "spec":
        description = ticket.description_html if ticket else ""
        return (f"Ticket {workflow['ticket_key']}: {workflow['ticket_summary']}\n\n"
               f"Description:\n{description}\n\n"
               f"Write the spec to `{artifact_path(workflow, 'spec')}`.")
    if stage == "plan":
        return (f"Read `{artifact_path(workflow, 'spec')}` and the repository structure, then "
               f"write the technical plan to `{artifact_path(workflow, 'plan')}`.")
    if stage == "tasks":
        return (f"Read `{artifact_path(workflow, 'plan')}` and write the task checklist to "
               f"`{artifact_path(workflow, 'tasks')}`.")
    return (f"Read `{artifact_path(workflow, 'tasks')}` top to bottom and implement every "
           f"unchecked task, checking each one off as you finish it.")


def default_workflow_model_factory(cfg: Config) -> Callable[[], BaseChatModel]:
    def factory() -> BaseChatModel:
        return ChatAnthropic(model=cfg.workflow.model or cfg.claude.default_model,
                             api_key=cfg.claude.api_key, max_tokens=cfg.claude.max_tokens,
                             streaming=True)
    return factory
