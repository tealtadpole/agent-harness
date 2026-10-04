"""Local git operations for the workflow pipeline: clone/fetch, branch-from-fresh-base,
commit, push -- all via plain `git` subprocesses (no shell=True, explicit arg lists).

Push/clone without SSH keys, token never in argv/URL/on disk: a per-invocation
`credential.helper` script reads the token from that subprocess's own environment, so it
never appears in a command line or gets written into `.git/config`. `token` is optional
throughout, so the exact same functions run unmodified against a tokenless local/`file://`
remote in tests.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

# Reads the token from $GITHUB_TOKEN in the subprocess's own env, set only for this one call.
CREDENTIAL_HELPER = "!f() { echo username=x-access-token; echo password=$GITHUB_TOKEN; }; f"

# Never let a git call hang the workflow forever (a stuck credential prompt, an unreachable
# remote, antivirus scanning the .git dir, ...). Generous, since a slow clone is legitimate.
GIT_TIMEOUT_SECONDS = 120


class GitOpsError(Exception):
    pass


async def _run_git(args: list[str], cwd: Path, token: str | None = None) -> str:
    full_args = [*args]
    env = None
    if token:
        full_args = ["-c", f"credential.helper={CREDENTIAL_HELPER}", *args]
        env = {**os.environ, "GITHUB_TOKEN": token}
    # A worker thread + sync subprocess.run, not asyncio.create_subprocess_exec: the latter
    # needs a Proactor event loop on Windows, which conflicts with psycopg's async driver
    # (needs Selector) running in the same process. to_thread works under any loop policy.
    try:
        result = await asyncio.to_thread(
            subprocess.run, ["git", *full_args], cwd=str(cwd), env=env, capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as e:
        raise GitOpsError(f"git {' '.join(args)} timed out after {GIT_TIMEOUT_SECONDS}s") from e
    if result.returncode != 0:
        raise GitOpsError(f"git {' '.join(args)} failed: "
                          f"{result.stderr.decode(errors='replace').strip()}")
    return result.stdout.decode(errors="replace")


async def ensure_repo(workdir: Path, clone_url: str, token: str | None) -> Path:
    """Clone `clone_url` into `workdir` if it isn't already a checkout there; otherwise fetch.
    Returns `workdir`."""
    if (workdir / ".git").is_dir():
        await _run_git(["fetch", "origin"], workdir, token)
        return workdir
    workdir.parent.mkdir(parents=True, exist_ok=True)
    await _run_git(["clone", clone_url, str(workdir)], workdir.parent, token)
    return workdir


async def checkout_new_branch(repo_dir: Path, base_branch: str, new_branch: str,
                              token: str | None = None) -> None:
    """Always branches from the freshly-fetched remote tip, never a possibly-stale local
    branch -- this is what lets a later stage see the previous stage's just-merged artifact."""
    await _run_git(["fetch", "origin", base_branch], repo_dir, token)
    await _run_git(["checkout", "-B", new_branch, f"origin/{base_branch}"], repo_dir, token)


async def commit_all(repo_dir: Path, message: str, user_name: str, user_email: str) -> bool:
    """Stage everything and commit. Returns False (no-op) if there was nothing to commit."""
    await _run_git(["add", "-A"], repo_dir)
    status = await _run_git(["status", "--porcelain"], repo_dir)
    if not status.strip():
        return False
    await _run_git(["-c", f"user.name={user_name}", "-c", f"user.email={user_email}",
                   "commit", "-m", message], repo_dir)
    return True


async def push_branch(repo_dir: Path, branch: str, token: str | None = None) -> None:
    """Safe to re-push on resume: --force-with-lease won't clobber unexpected concurrent
    changes but allows re-pushing our own prior push of this same branch."""
    await _run_git(["push", "-u", "origin", branch, "--force-with-lease"], repo_dir, token)
