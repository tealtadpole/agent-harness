"""git_ops: clone/fetch, branch-from-fresh-base, commit, push -- against a real local git
remote (a bare repo in tmp_path), no mocking."""

import asyncio
import subprocess

import pytest

from agent_harness.git_ops import GitOpsError, checkout_new_branch, commit_all, ensure_repo, push_branch


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def bare_remote(tmp_path):
    """A bare repo seeded with one commit on main -- stands in for a real GitHub remote."""
    bare = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    _git(["init", "--bare", "-b", "main", str(bare)], cwd=tmp_path)
    _git(["init", "-b", "main", str(seed)], cwd=tmp_path)
    (seed / "README.md").write_text("hello\n")
    _git(["-c", "user.name=seed", "-c", "user.email=seed@example.com", "add", "-A"], cwd=seed)
    _git(["-c", "user.name=seed", "-c", "user.email=seed@example.com", "commit", "-m", "init"], cwd=seed)
    _git(["remote", "add", "origin", str(bare)], cwd=seed)
    _git(["push", "origin", "main"], cwd=seed)
    return str(bare)


def test_ensure_repo_clones_then_fetches(tmp_path, bare_remote):
    repo_dir = tmp_path / "work" / "repo"

    async def run():
        await ensure_repo(repo_dir, bare_remote, token=None)
        assert (repo_dir / "README.md").is_file()
        # Second call on an existing checkout should fetch, not re-clone, and not raise.
        await ensure_repo(repo_dir, bare_remote, token=None)

    asyncio.run(run())


def test_checkout_new_branch_tracks_fresh_remote_tip(tmp_path, bare_remote):
    repo_dir = tmp_path / "work" / "repo"

    async def run():
        await ensure_repo(repo_dir, bare_remote, token=None)
        await checkout_new_branch(repo_dir, "main", "spec-branch")
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=repo_dir,
                                capture_output=True, text=True, check=True).stdout.strip()
        assert branch == "spec-branch"

    asyncio.run(run())


def test_commit_all_noop_when_nothing_changed(tmp_path, bare_remote):
    repo_dir = tmp_path / "work" / "repo"

    async def run():
        await ensure_repo(repo_dir, bare_remote, token=None)
        changed = await commit_all(repo_dir, "nothing to see", "bot", "bot@example.com")
        assert changed is False

    asyncio.run(run())


def test_commit_all_and_push_land_on_remote(tmp_path, bare_remote):
    repo_dir = tmp_path / "work" / "repo"
    clone_dir = tmp_path / "verify"

    async def run():
        await ensure_repo(repo_dir, bare_remote, token=None)
        await checkout_new_branch(repo_dir, "main", "spec-branch")
        (repo_dir / "spec.md").write_text("# Spec\n")
        changed = await commit_all(repo_dir, "add spec.md", "bot", "bot@example.com")
        assert changed is True
        await push_branch(repo_dir, "spec-branch", token=None)

    asyncio.run(run())

    _git(["clone", "--branch", "spec-branch", bare_remote, str(clone_dir)], cwd=tmp_path)
    assert (clone_dir / "spec.md").is_file()


def test_plan_stage_sees_spec_merged_into_main(tmp_path, bare_remote):
    """Simulates the real sequencing: spec branch merges into main, then a 'plan' checkout
    cut fresh from main must see spec.md -- this is the auto-merge-on-approval guarantee."""
    repo_dir = tmp_path / "work" / "repo"

    async def run():
        await ensure_repo(repo_dir, bare_remote, token=None)
        await checkout_new_branch(repo_dir, "main", "spec-branch")
        (repo_dir / "spec.md").write_text("# Spec\n")
        await commit_all(repo_dir, "add spec.md", "bot", "bot@example.com")
        await push_branch(repo_dir, "spec-branch", token=None)

    asyncio.run(run())

    # Simulate the harness merging the approved PR itself (what GithubClient.merge_pull_request
    # does on GitHub's side) by fast-forward-merging on the bare remote's main.
    merge_dir = tmp_path / "merge"
    _git(["clone", bare_remote, str(merge_dir)], cwd=tmp_path)
    _git(["fetch", "origin", "spec-branch"], cwd=merge_dir)
    _git(["merge", "--ff-only", "origin/spec-branch"], cwd=merge_dir)
    _git(["push", "origin", "main"], cwd=merge_dir)

    async def run_plan():
        await checkout_new_branch(repo_dir, "main", "plan-branch")
        assert (repo_dir / "spec.md").is_file()

    asyncio.run(run_plan())


def test_git_ops_error_on_failed_command(tmp_path):
    missing_source = tmp_path / "does-not-exist.git"   # local path: fails fast, no network needed

    async def run():
        with pytest.raises(GitOpsError):
            await ensure_repo(tmp_path / "nope", str(missing_source), None)

    asyncio.run(run())
