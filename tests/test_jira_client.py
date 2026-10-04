"""JiraClient: request shape, renderedFields description, Platform field (string or object)."""

import asyncio

import httpx
import pytest

from agent_harness.config import JiraConfig
from agent_harness.jira_client import JiraClient, JiraError, jira_status

CFG = JiraConfig(enabled=True, base_url="https://example.atlassian.net",
                 email_env="TEST_JIRA_EMAIL", api_token_env="TEST_JIRA_TOKEN",
                 platform_field_id="customfield_999")


@pytest.fixture(autouse=True)
def jira_creds(monkeypatch):
    monkeypatch.setenv("TEST_JIRA_EMAIL", "bot@example.com")
    monkeypatch.setenv("TEST_JIRA_TOKEN", "sekret")


def _client_with(handler) -> JiraClient:
    client = JiraClient(CFG)
    client._client = lambda: httpx.AsyncClient(
        base_url=CFG.base_url, auth=(CFG.email, CFG.api_token),
        transport=httpx.MockTransport(handler))
    return client


def test_get_ticket_parses_rendered_description_and_object_platform_field():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rest/api/3/issue/PROJ-1"
        assert request.headers["authorization"].startswith("Basic ")
        assert "customfield_999" in request.url.params["fields"]
        assert request.url.params["expand"] == "renderedFields"
        return httpx.Response(200, json={
            "key": "PROJ-1",
            "fields": {"summary": "Add widget", "customfield_999": {"value": "web"}},
            "renderedFields": {"description": "<p>Please add a widget.</p>"},
        })

    ticket = asyncio.run(_client_with(handler).get_ticket("PROJ-1"))
    assert ticket.key == "PROJ-1"
    assert ticket.summary == "Add widget"
    assert ticket.description_html == "<p>Please add a widget.</p>"
    assert ticket.platform == "web"


def test_get_ticket_handles_bare_string_platform_field():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "key": "PROJ-2", "fields": {"summary": "x", "customfield_999": "mobile"},
            "renderedFields": {"description": ""},
        })

    ticket = asyncio.run(_client_with(handler).get_ticket("PROJ-2"))
    assert ticket.platform == "mobile"


def test_get_ticket_404_raises_jira_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"errorMessages": ["Issue not found"]})

    with pytest.raises(JiraError, match="PROJ-404"):
        asyncio.run(_client_with(handler).get_ticket("PROJ-404"))


def test_jira_status_reports_auth_failure():
    cfg = JiraConfig(enabled=True, base_url="https://example.atlassian.net",
                     email_env="TEST_JIRA_EMAIL", api_token_env="TEST_JIRA_TOKEN")

    async def run():
        import agent_harness.jira_client as mod
        orig = httpx.AsyncClient

        class Patched(httpx.AsyncClient):
            def __init__(self, *a, **kw):
                kw["transport"] = httpx.MockTransport(lambda r: httpx.Response(401))
                super().__init__(*a, **kw)

        mod.httpx.AsyncClient = Patched
        try:
            return await jira_status(cfg)
        finally:
            mod.httpx.AsyncClient = orig

    ok, detail = asyncio.run(run())
    assert not ok and "401" in detail


def test_jira_status_missing_config_reported():
    ok, detail = asyncio.run(jira_status(JiraConfig(enabled=True, base_url="")))
    assert not ok and "base_url" in detail
