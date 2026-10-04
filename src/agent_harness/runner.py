"""Run one chat turn through a provider's LangGraph graph, stream events, save history.

A turn runs as a background task that is independent of the HTTP request: if the
browser disconnects mid-answer, the turn still finishes and is saved.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from .db import Database

log = logging.getLogger(__name__)

TITLE_LENGTH = 60


class TurnBusy(Exception):
    pass


class TurnRunner:
    def __init__(self, db: Database, providers: dict, recursion_limit: int):
        self.db = db
        self.providers = providers
        self.recursion_limit = recursion_limit
        self._active: dict[str, asyncio.Task] = {}

    def is_running(self, session_id: str) -> bool:
        task = self._active.get(session_id)
        return task is not None and not task.done()

    async def start(self, session: dict, text: str) -> AsyncIterator[dict]:
        """Start a turn and yield its events. The turn keeps running if iteration stops early."""
        sid = session["id"]
        if self.is_running(sid):
            raise TurnBusy("A reply is still being generated for this chat.")
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._active[sid] = asyncio.create_task(self._run(session, text, queue.put_nowait))
        while (event := await queue.get()) is not None:
            yield event

    async def wait_idle(self) -> None:
        tasks = [t for t in self._active.values() if not t.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=30)

    async def _run(self, session: dict, text: str, emit) -> None:
        sid = session["id"]
        answer: list[str] = []
        tools_seen: set[str] = set()
        try:
            user_msg = await self.db.add_message(sid, "user", text)
            emit({"type": "user", "message": user_msg})
            if session["title"] == "New chat":
                title = " ".join(text.split())[:TITLE_LENGTH] or "New chat"
                await self.db.rename_session(sid, title)
                emit({"type": "title", "title": title})

            provider = self.providers[session["provider"]]
            graph = provider.graph(session["model"])
            config = {"configurable": {"thread_id": sid, "model": session["model"]},
                      "recursion_limit": self.recursion_limit}

            async for mode, chunk in graph.astream({"messages": [HumanMessage(text)]}, config,
                                                   stream_mode=["messages", "updates", "custom"]):
                if mode == "messages":
                    msg, _meta = chunk
                    # Only streamed model tokens; whole messages arrive again via "updates".
                    if isinstance(msg, AIMessageChunk) and (t := msg.text):
                        answer.append(t)
                        emit({"type": "token", "text": t})
                elif mode == "custom":   # Copilot streams through the custom channel
                    await self._custom_event(sid, chunk, answer, tools_seen, emit)
                elif mode == "updates":
                    for update in chunk.values():
                        for msg in (update or {}).get("messages", []):
                            await self._state_message(sid, msg, answer, tools_seen, emit)
        except Exception as e:
            log.exception("Turn failed for session %s", sid)
            message = _error_text(e)
            await self._safe(self.db.add_message(sid, "error", message))
            emit({"type": "error", "message": message})
        finally:
            content = "".join(answer).strip()
            if content:
                saved = await self._safe(self.db.add_message(sid, "assistant", content))
                if saved:
                    emit({"type": "assistant", "message": saved})
            await self._safe(self.db.touch_session(sid))
            emit({"type": "done"})
            emit(None)

    async def _state_message(self, sid, msg, answer, tools_seen, emit) -> None:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                if call["id"] not in tools_seen:
                    tools_seen.add(call["id"])
                    _paragraph_break(answer)
                    row = await self.db.add_message(sid, "tool", "", tool_name=call["name"],
                                                    tool_call_id=call["id"], tool_input=call["args"])
                    emit({"type": "tool_start", "message": row})
            if msg.response_metadata.get("stop_reason") == "refusal":
                emit({"type": "notice", "text": "The model declined to answer this request."})
        elif isinstance(msg, ToolMessage):
            output = msg.text if isinstance(msg.content, list) else str(msg.content)
            await self.db.set_tool_result(sid, msg.tool_call_id, output)
            emit({"type": "tool_end", "tool_call_id": msg.tool_call_id, "output": output})

    async def _custom_event(self, sid, event, answer, tools_seen, emit) -> None:
        kind = event.get("type")
        if kind == "token":
            answer.append(event["text"])
            emit(event)
        elif kind == "tool_start" and event["id"] not in tools_seen:
            tools_seen.add(event["id"])
            _paragraph_break(answer)
            row = await self.db.add_message(sid, "tool", "", tool_name=event["name"],
                                            tool_call_id=event["id"],
                                            tool_input=_jsonable(event.get("input")))
            emit({"type": "tool_start", "message": row})
        elif kind == "tool_end":
            await self.db.set_tool_result(sid, event["id"], event.get("output", ""))
            emit({"type": "tool_end", "tool_call_id": event["id"], "output": event.get("output", "")})

    @staticmethod
    async def _safe(coro):
        try:
            return await coro
        except Exception:
            log.exception("Failed to save turn data")
            return None


def _paragraph_break(answer: list[str]) -> None:
    """Keep text written before and after a tool call from running together."""
    if answer and not answer[-1].endswith("\n"):
        answer.append("\n\n")


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _error_text(e: Exception) -> str:
    name = type(e).__name__
    if name == "AuthenticationError":
        return "Authentication failed: check your API key."
    if name == "GraphRecursionError":
        return "The agent hit its step limit before finishing (agent.recursion_limit)."
    return f"{name}: {e}"
