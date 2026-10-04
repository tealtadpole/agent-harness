"""GitHub REST API client: create/inspect/merge pull requests.

Raw REST via httpx rather than PyGithub: PyGithub is synchronous and would need
executor-wrapping to avoid blocking the event loop, unlike every other I/O path in this
codebase; this keeps the same explicit, async-native style as `jira_client.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import GithubConfig

API_BASE = "https://api.github.com"
API_VERSION = "2022-11-28"


@dataclass(frozen=True)
class PullRequest:
    number: int
    html_url: str
    state: str    # "open" | "closed"
    merged: bool


class GithubError(Exception):
    pass


class GithubClient:
    def __init__(self, cfg: GithubConfig):
        self.cfg = cfg

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=API_BASE, timeout=30, headers={
            "Authorization": f"Bearer {self.cfg.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
        })

    async def create_pull_request(self, owner: str, repo: str, head: str, base: str,
                                   title: str, body: str) -> PullRequest:
        async with self._client() as client:
            resp = await _request(client, "POST", f"/repos/{owner}/{repo}/pulls",
                                  json={"head": head, "base": base, "title": title, "body": body})
        return _pull_request(resp.json())

    async def get_pull_request(self, owner: str, repo: str, number: int) -> PullRequest:
        async with self._client() as client:
            resp = await _request(client, "GET", f"/repos/{owner}/{repo}/pulls/{number}")
        return _pull_request(resp.json())

    async def list_review_states(self, owner: str, repo: str, number: int) -> dict[str, str]:
        """{login: latest_state}, keeping only each reviewer's most recent review -- matches
        GitHub's own 'latest review per person counts' approval semantics."""
        async with self._client() as client:
            resp = await _request(client, "GET", f"/repos/{owner}/{repo}/pulls/{number}/reviews")
        latest: dict[str, tuple[str, str]] = {}
        for review in resp.json():
            login = (review.get("user") or {}).get("login") or ""
            submitted_at = review.get("submitted_at") or ""
            state = review.get("state") or ""
            if not login or state == "COMMENTED":
                continue
            if login not in latest or submitted_at >= latest[login][0]:
                latest[login] = (submitted_at, state)
        return {login: state for login, (_, state) in latest.items()}

    def is_approved(self, review_states: dict[str, str]) -> bool:
        states = review_states.values()
        return "APPROVED" in states and "CHANGES_REQUESTED" not in states

    async def merge_pull_request(self, owner: str, repo: str, number: int, merge_method: str) -> None:
        async with self._client() as client:
            await _request(client, "PUT", f"/repos/{owner}/{repo}/pulls/{number}/merge",
                           json={"merge_method": merge_method})


async def _request(client: httpx.AsyncClient, method: str, path: str, **kwargs) -> httpx.Response:
    try:
        resp = await client.request(method, path, **kwargs)
    except httpx.HTTPError as e:
        raise GithubError(f"Could not reach GitHub: {e}") from e
    if resp.status_code >= 400:
        raise GithubError(f"GitHub {method} {path} returned {resp.status_code}: {resp.text[:300]}")
    return resp


def _pull_request(body: dict) -> PullRequest:
    return PullRequest(number=body["number"], html_url=body["html_url"],
                       state=body["state"], merged=bool(body.get("merged", False)))


async def github_status(cfg: GithubConfig) -> tuple[bool, str]:
    """GET /user with the token -- the cheapest reachability+auth check, for `cli.py check`."""
    if not cfg.token:
        return False, f"Set ${cfg.token_env} to enable GitHub."
    try:
        async with httpx.AsyncClient(base_url=API_BASE, timeout=15, headers={
                "Authorization": f"Bearer {cfg.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": API_VERSION}) as client:
            resp = await client.get("/user")
    except httpx.HTTPError as e:
        return False, f"Could not reach GitHub: {e}"
    if resp.status_code == 401:
        return False, "GitHub rejected the token (401)."
    if resp.status_code != 200:
        return False, f"GitHub returned {resp.status_code}."
    who = resp.json()
    return True, f"Signed in as {who.get('login') or 'unknown user'}"
