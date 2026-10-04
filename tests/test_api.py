"""End-to-end through the real FastAPI app, Postgres, LangGraph and a real stdio MCP server."""

import asyncio

from conftest import tool_call
from langchain_core.messages import AIMessage, HumanMessage


def kinds(events):
    return [e["type"] for e in events]


def test_claude_turn_uses_mcp_tool_and_keeps_history(harness, run_app):
    app, model, _ = harness([
        tool_call("search_confluence", {"query": "vacation days"}),
        AIMessage("Employees get 25 days of paid vacation."),
        AIMessage("You asked about vacation."),
    ])

    async def body(c):
        providers = (await c.http.get("/api/providers")).json()
        assert {p["name"] for p in providers["providers"]} == {"claude", "copilot"}
        assert providers["mcp_servers"][0] == {"name": "confluence", "ok": True,
                                               "tools": ["search_confluence"], "error": None}

        s = (await c.http.post("/api/sessions", json={"provider": "claude"})).json()
        assert s["model"] == "claude-test" and s["title"] == "New chat"

        events = await c.chat(s["id"], "How many vacation days do I get?")
        assert kinds(events)[0] == "user" and kinds(events)[-1] == "done"
        assert {"title", "tool_start", "tool_end", "token", "assistant"} <= set(kinds(events))
        tool_end = next(e for e in events if e["type"] == "tool_end")
        assert "25 days" in tool_end["output"]
        assert "secret=s3cret" in tool_end["output"]      # ${VAR} env reached the MCP server
        tokens = "".join(e["text"] for e in events if e["type"] == "token")
        assert tokens == "Let me check the wiki.Employees get 25 days of paid vacation."

        detail = (await c.http.get(f"/api/sessions/{s['id']}")).json()
        assert detail["title"] == "How many vacation days do I get?"
        assert [m["role"] for m in detail["messages"]] == ["user", "tool", "assistant"]
        tool = detail["messages"][1]
        assert tool["tool_name"] == "search_confluence"
        assert tool["tool_input"] == {"query": "vacation days"} and "25 days" in tool["content"]
        assert detail["messages"][2]["content"] == (
            "Let me check the wiki.\n\nEmployees get 25 days of paid vacation.")

        # Second turn: the checkpointer replays the earlier conversation to the model.
        await c.chat(s["id"], "What did I ask?")
        history = model.seen[-1]
        assert any(isinstance(m, HumanMessage) and "vacation days" in m.text for m in history)
        assert len([m for m in history if isinstance(m, HumanMessage)]) == 2

    run_app(app, body)


def test_copilot_turn_streams_custom_events(harness, run_app):
    app, _, copilot = harness([AIMessage("unused")])

    async def body(c):
        s = (await c.http.post("/api/sessions",
                               json={"provider": "copilot", "model": "gpt-test"})).json()
        events = await c.chat(s["id"], "vacation?")
        assert copilot.prompts == [(s["id"], "gpt-test", "vacation?")]
        assert "".join(e["text"] for e in events if e["type"] == "token") == "Copilot says 25 days."
        detail = (await c.http.get(f"/api/sessions/{s['id']}")).json()
        assert [(m["role"], m["content"]) for m in detail["messages"]] == [
            ("user", "vacation?"), ("tool", "25 days"), ("assistant", "Copilot says 25 days.")]

        bad = await c.http.post("/api/sessions", json={"provider": "copilot", "model": "nope"})
        assert bad.status_code == 400

        assert (await c.http.delete(f"/api/sessions/{s['id']}")).status_code == 204
        assert copilot.deleted == [s["id"]]

    run_app(app, body)


def test_session_crud_and_checkpoint_cleanup(harness, run_app):
    app, _, _ = harness([AIMessage("Hi there.")])

    async def body(c):
        a = (await c.http.post("/api/sessions", json={"provider": "claude"})).json()
        b = (await c.http.post("/api/sessions", json={"provider": "claude"})).json()
        await c.chat(a["id"], "hello")
        listed = (await c.http.get("/api/sessions")).json()
        assert [x["id"] for x in listed] == [a["id"], b["id"]]   # most recently used first
        assert listed[0]["message_count"] == 2

        r = await c.http.patch(f"/api/sessions/{b['id']}", json={"title": "Renamed"})
        assert r.json()["title"] == "Renamed"

        saver = app.state.checkpointer
        config = {"configurable": {"thread_id": a["id"]}}
        assert await saver.aget_tuple(config) is not None
        assert (await c.http.delete(f"/api/sessions/{a['id']}")).status_code == 204
        assert await saver.aget_tuple(config) is None
        assert (await c.http.get(f"/api/sessions/{a['id']}")).status_code == 404
        assert (await c.http.get("/api/sessions/not-a-uuid")).status_code == 404
        assert (await c.http.post("/api/sessions", json={"provider": "gpt"})).status_code == 400

    run_app(app, body)


def test_model_error_is_reported_and_saved(harness, run_app):
    app, _, _ = harness([RuntimeError("upstream exploded")])

    async def body(c):
        s = (await c.http.post("/api/sessions", json={"provider": "claude"})).json()
        events = await c.chat(s["id"], "hi")
        err = next(e for e in events if e["type"] == "error")
        assert "upstream exploded" in err["message"]
        roles = [m["role"] for m in (await c.http.get(f"/api/sessions/{s['id']}")).json()["messages"]]
        assert roles == ["user", "error"]

    run_app(app, body)


def test_second_message_while_busy_is_rejected(harness, run_app):
    app, _, copilot = harness([AIMessage("unused")])
    copilot.hold = asyncio.Event()

    async def body(c):
        s = (await c.http.post("/api/sessions", json={"provider": "copilot"})).json()
        first = asyncio.create_task(c.chat(s["id"], "one"))
        for _ in range(100):
            await asyncio.sleep(0.02)
            if app.state.harness.runner.is_running(s["id"]):
                break
        r = await c.http.post(f"/api/sessions/{s['id']}/messages", json={"content": "two"})
        assert r.status_code == 409
        copilot.hold.set()
        assert "done" in kinds(await first)

    run_app(app, body)


def test_foreign_host_header_is_rejected(harness, run_app):
    app, _, _ = harness([AIMessage("x")])

    async def body(c):
        r = await c.http.get("/api/sessions", headers={"Host": "evil.example"})
        assert r.status_code == 400

    run_app(app, body)
