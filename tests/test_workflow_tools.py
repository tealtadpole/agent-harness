"""Sandbox-scoped filesystem and shell tools for workflow stage agents."""

import asyncio

from agent_harness.workflow_tools import make_fs_tools, make_shell_tool


def _tools(repo_dir):
    return {t.name: t for t in make_fs_tools(repo_dir)}


def test_write_then_read_round_trips(tmp_path):
    tools = _tools(tmp_path)

    async def run():
        out = await tools["write_file"].ainvoke({"path": "specs/x/spec.md", "content": "# Spec\n"})
        assert "Wrote" in out
        return await tools["read_file"].ainvoke({"path": "specs/x/spec.md"})

    assert asyncio.run(run()) == "# Spec\n"


def test_list_dir_is_relative_to_repo_root(tmp_path):
    tools = _tools(tmp_path)

    async def run():
        await tools["write_file"].ainvoke({"path": "specs/x/spec.md", "content": "hi"})
        return await tools["list_dir"].ainvoke({"path": "specs/x"})

    assert asyncio.run(run()) == "specs/x/spec.md"


def test_search_files_finds_matching_lines(tmp_path):
    tools = _tools(tmp_path)

    async def run():
        await tools["write_file"].ainvoke({"path": "a.md", "content": "needle here\nhay\n"})
        await tools["write_file"].ainvoke({"path": "b.md", "content": "hay only\n"})
        return await tools["search_files"].ainvoke({"query": "needle", "glob": "**/*.md"})

    out = asyncio.run(run())
    assert "a.md:1: needle here" in out
    assert "b.md" not in out


def test_read_file_missing_reports_error_not_exception(tmp_path):
    tools = _tools(tmp_path)
    out = asyncio.run(tools["read_file"].ainvoke({"path": "nope.md"}))
    assert out.startswith("Error:")


def test_path_traversal_is_rejected(tmp_path):
    tools = _tools(tmp_path)

    async def run():
        return await tools["read_file"].ainvoke({"path": "../outside.txt"})

    out = asyncio.run(run())
    assert "escapes the sandbox" in out


def test_run_shell_captures_output_and_exit_code(tmp_path):
    shell = make_shell_tool(tmp_path, timeout_seconds=10)
    out = asyncio.run(shell.ainvoke({"command": "echo hello"}))
    assert "exit code 0" in out and "hello" in out


def test_run_shell_times_out(tmp_path):
    shell = make_shell_tool(tmp_path, timeout_seconds=1)

    async def run():
        return await shell.ainvoke({"command": "python -c \"import time; time.sleep(5)\""})

    out = asyncio.run(run())
    assert "timed out" in out
