"""JIRA Cloud REST API v3 client: fetch a ticket's summary, description and "Platform" field.

Auth is HTTP Basic with an Atlassian Cloud API token (the documented auth for JIRA Cloud).
`expand=renderedFields` asks JIRA to return `description` as plain HTML instead of raw ADF
(Atlassian Document Format) JSON, so we don't need an ADF parser.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import JiraConfig


@dataclass(frozen=True)
class JiraTicket:
    key: str
    summary: str
    description_html: str
    platform: str


class JiraError(Exception):
    pass


class JiraClient:
    def __init__(self, cfg: JiraConfig):
        self.cfg = cfg

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.cfg.base_url.rstrip("/"),
                                 auth=(self.cfg.email, self.cfg.api_token), timeout=30)

    async def get_ticket(self, ticket_key: str) -> JiraTicket:
        field_id = self.cfg.platform_field_id
        async with self._client() as client:
            try:
                resp = await client.get(f"/rest/api/3/issue/{ticket_key}", params={
                    "fields": f"summary,description,{field_id}", "expand": "renderedFields"})
            except httpx.HTTPError as e:
                raise JiraError(f"Could not reach JIRA: {e}") from e
        if resp.status_code == 404:
            raise JiraError(f"JIRA ticket {ticket_key!r} not found")
        if resp.status_code != 200:
            raise JiraError(f"JIRA returned {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        fields = body.get("fields", {})
        rendered = body.get("renderedFields", {})
        return JiraTicket(
            key=body.get("key", ticket_key),
            summary=fields.get("summary") or "",
            description_html=rendered.get("description") or "",
            platform=_platform_value(fields.get(field_id)),
        )


def _platform_value(raw) -> str:
    """The Platform field may be a bare string or a select-field object {"id", "value", ...}."""
    if isinstance(raw, dict):
        return str(raw.get("value") or raw.get("name") or "")
    return str(raw) if raw else ""


async def jira_status(cfg: JiraConfig) -> tuple[bool, str]:
    """GET /rest/api/3/myself -- the cheapest reachability+auth check, for `cli.py check`."""
    if not cfg.base_url:
        return False, "Set [jira] base_url to enable JIRA."
    if not (cfg.email and cfg.api_token):
        return False, f"Set ${cfg.email_env} and ${cfg.api_token_env} to enable JIRA."
    try:
        async with httpx.AsyncClient(base_url=cfg.base_url.rstrip("/"),
                                     auth=(cfg.email, cfg.api_token), timeout=15) as client:
            resp = await client.get("/rest/api/3/myself")
    except httpx.HTTPError as e:
        return False, f"Could not reach JIRA: {e}"
    if resp.status_code == 401:
        return False, "JIRA rejected the credentials (401)."
    if resp.status_code != 200:
        return False, f"JIRA returned {resp.status_code}."
    who = resp.json()
    return True, f"Signed in as {who.get('displayName') or who.get('emailAddress') or 'unknown user'}"
