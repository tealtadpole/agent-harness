"""Filesystem and shell tools for workflow stage agents, scoped to one sandbox checkout.

App-level guardrail only, not OS-level isolation: `_resolve` rejects any path that escapes
`repo_dir`, but the process itself runs with the harness's own privileges (same limitation
already accepted for MCP-server subprocesses -- see `mcp_tools.py`). `run_shell` in particular
is arbitrary code execution with no container/seccomp/network policy around it.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

MAX_OUTPUT = 20_000


def _resolve(repo_dir: Path, path: str) -> Path:
    repo_dir = repo_dir.resolve()
    candidate = (repo_dir / path).resolve()
    if not candidate.is_relative_to(repo_dir):
        raise ValueError(f"path escapes the sandbox: {path!r}")
    return candidate


def _truncate(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + "\n[truncated]"


class ReadFileArgs(BaseModel):
    path: str = Field(description="File path relative to the repository root")


class WriteFileArgs(BaseModel):
    path: str = Field(description="File path relative to the repository root")
    content: str = Field(description="Full file content to write")


class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="Directory path relative to the repository root")


class SearchFilesArgs(BaseModel):
    query: str = Field(description="Text to search for")
    glob: str = Field(default="**/*", description="Glob pattern limiting which files are searched")


def make_fs_tools(repo_dir: Path) -> list[StructuredTool]:
    repo_dir = repo_dir.resolve()   # resolve once so every comparison below uses the same form

    async def read_file(path: str) -> str:
        try:
            file = _resolve(repo_dir, path)
        except ValueError as e:
            return f"Error: {e}"
        if not file.is_file():
            return f"Error: no such file: {path}"
        return _truncate(file.read_text(errors="replace"))

    async def write_file(path: str, content: str) -> str:
        try:
            file = _resolve(repo_dir, path)
        except ValueError as e:
            return f"Error: {e}"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"

    async def list_dir(path: str = ".") -> str:
        try:
            directory = _resolve(repo_dir, path)
        except ValueError as e:
            return f"Error: {e}"
        if not directory.is_dir():
            return f"Error: no such directory: {path}"
        entries = sorted(
            p.relative_to(repo_dir).as_posix() + ("/" if p.is_dir() else "")
            for p in directory.iterdir() if p.name != ".git")
        return "\n".join(entries) or "(empty)"

    async def search_files(query: str, glob: str = "**/*") -> str:
        hits: list[str] = []
        for file in sorted(repo_dir.glob(glob)):
            if len(hits) >= 200:
                break
            if not file.is_file() or ".git" in file.parts:
                continue
            try:
                text = file.read_text(errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    hits.append(f"{file.relative_to(repo_dir).as_posix()}:{i}: {line.strip()}")
                    if len(hits) >= 200:
                        break
        return "\n".join(hits) or "(no matches)"

    return [
        StructuredTool.from_function(
            coroutine=read_file, name="read_file",
            description="Read a text file from the repository checkout.", args_schema=ReadFileArgs),
        StructuredTool.from_function(
            coroutine=write_file, name="write_file",
            description="Write (create or overwrite) a text file in the repository checkout.",
            args_schema=WriteFileArgs),
        StructuredTool.from_function(
            coroutine=list_dir, name="list_dir",
            description="List files and directories at a path in the repository checkout.",
            args_schema=ListDirArgs),
        StructuredTool.from_function(
            coroutine=search_files, name="search_files",
            description="Search file contents for a substring, optionally limited by a glob pattern.",
            args_schema=SearchFilesArgs),
    ]


class RunShellArgs(BaseModel):
    command: str = Field(description="Shell command to run in the repository checkout")


def make_shell_tool(repo_dir: Path, timeout_seconds: int) -> StructuredTool:
    async def run_shell(command: str) -> str:
        # A worker thread + sync subprocess.run, not asyncio.create_subprocess_shell: the
        # latter needs a Proactor event loop on Windows, which conflicts with psycopg's async
        # driver (needs Selector) running in the same process. to_thread works under any loop.
        try:
            result = await asyncio.to_thread(
                subprocess.run, command, shell=True, cwd=str(repo_dir),
                capture_output=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout_seconds}s"
        except OSError as e:
            return f"Error: {e}"
        output = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
        return f"(exit code {result.returncode})\n{_truncate(output)}"

    return StructuredTool.from_function(
        coroutine=run_shell, name="run_shell",
        description=("Run a shell command (e.g. a build or test command) in the repository "
                     "checkout and return its combined output. No OS-level sandboxing: this "
                     "runs with the harness process's own privileges."),
        args_schema=RunShellArgs)
