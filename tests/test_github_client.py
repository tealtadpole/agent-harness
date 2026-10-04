"""GithubClient: request shapes, review-state aggregation, approval semantics."""

import asyncio

import httpx
import pytest

from agent_harness.config import GithubConfig
from agent_harness.github_client import GithubClient, GithubError, github_status

CFG = GithubConfig(enabled=True, token_env="TEST_GITHUB_TOKEN", merge_method="squash")


@pytest.fixture(autouse=True)
def github_token(monkeypatch):
    monkeypatch.setenv("TEST_GITHUB_TOKEN", "ghp_sekret")


def _client_with(handler) -> GithubClient:
    client = GithubClient(CFG)
    client._client = lambda: httpx.AsyncClient(
        base_url="https://api.github.com",
        headers={"Authorization": f"Bearer {CFG.token}", "Accept": "application/vnd.github+json"},
        transport=httpx.MockTransport(handler))
    return client


def test_create_pull_request_sends_expected_body_and_parses_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = request.content
        assert request.headers["authorization"] == "Bearer ghp_sekret"
        return httpx.Response(201, json={"number": 7, "html_url": "https://github.com/a/b/pull/7",
                                         "state": "open", "merged": False})

    pr = asyncio.run(_client_with(handler).create_pull_request(
        "acme", "webapp", "spec-branch", "main", "spec: add widget", "see spec.md"))
    assert seen["method"] == "POST" and seen["path"] == "/repos/acme/webapp/pulls"
    assert b"spec-branch" in seen["body"] and b"main" in seen["body"]
    assert pr.number == 7 and pr.state == "open" and not pr.merged


def test_get_pull_request_merged_state():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/acme/webapp/pulls/7"
        return httpx.Response(200, json={"number": 7, "html_url": "https://x/pull/7",
                                         "state": "closed", "merged": True})

    pr = asyncio.run(_client_with(handler).get_pull_request("acme", "webapp", 7))
    assert pr.merged and pr.state == "closed"


def test_list_review_states_keeps_latest_per_reviewer_and_ignores_comments():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[
            {"user": {"login": "alice"}, "state": "CHANGES_REQUESTED", "submitted_at": "2026-01-01T00:00:00Z"},
            {"user": {"login": "alice"}, "state": "APPROVED", "submitted_at": "2026-01-02T00:00:00Z"},
            {"user": {"login": "bob"}, "state": "COMMENTED", "submitted_at": "2026-01-01T00:00:00Z"},
        ])

    client = _client_with(handler)
    states = asyncio.run(client.list_review_states("acme", "webapp", 7))
    assert states == {"alice": "APPROVED"}
    assert client.is_approved(states)


def test_is_approved_false_if_anyone_requested_changes():
    client = GithubClient(CFG)
    assert not client.is_approved({"alice": "APPROVED", "bob": "CHANGES_REQUESTED"})
    assert not client.is_approved({})


def test_merge_pull_request_uses_configured_method():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.content
        return httpx.Response(200, json={"merged": True})

    asyncio.run(_client_with(handler).merge_pull_request("acme", "webapp", 7, "squash"))
    assert seen["path"] == "/repos/acme/webapp/pulls/7/merge"
    assert b"squash" in seen["body"]


def test_error_response_raises_github_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"message": "Validation failed"})

    with pytest.raises(GithubError, match="422"):
        asyncio.run(_client_with(handler).create_pull_request("a", "b", "h", "base", "t", ""))


def test_github_status_missing_token():
    ok, detail = asyncio.run(github_status(GithubConfig(enabled=True, token_env="NOPE_TOKEN_VAR")))
    assert not ok and "NOPE_TOKEN_VAR" in detail
