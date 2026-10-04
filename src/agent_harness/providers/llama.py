"""A local LLM through Ollama: a LangChain tool-calling agent compiled as a LangGraph graph.

Same shape as the Claude provider, swapping `ChatAnthropic` for `ChatOllama`. No API key:
availability is instead "can we reach the Ollama server and does it have the model pulled"
(`ollama pull <model>`), checked via Ollama's own `/api/tags` endpoint.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.types import Checkpointer

from ..config import AgentConfig, LlamaConfig
from .base import ProviderStatus

STATUS_TIMEOUT_SECONDS = 5


class LlamaProvider:
    name = "llama"
    label = "Local LLM (Ollama)"

    def __init__(self, cfg: LlamaConfig, agent: AgentConfig, tools: list[BaseTool],
                 checkpointer: Checkpointer,
                 model_factory: Callable[[str], BaseChatModel] | None = None):
        self.cfg = cfg
        self.agent_cfg = agent
        self.tools = tools
        self.checkpointer = checkpointer
        self.model_factory = model_factory or self._chat_model
        self._graphs: dict[str, object] = {}

    async def status(self) -> ProviderStatus:
        try:
            async with httpx.AsyncClient(timeout=STATUS_TIMEOUT_SECONDS) as client:
                resp = await client.get(f"{self.cfg.base_url}/api/tags")
                resp.raise_for_status()
                pulled = {m["name"] for m in resp.json().get("models", [])}
        except Exception as e:
            return ProviderStatus(
                name=self.name, label=self.label, available=False,
                detail=f"Can't reach Ollama at {self.cfg.base_url} ({e}). "
                       "Run `ollama serve`, or `curl -fsSL https://ollama.com/install.sh | sh`.")
        missing = [m for m in self.cfg.models if m not in pulled]
        if missing:
            return ProviderStatus(
                name=self.name, label=self.label, available=False,
                detail=f"Pull missing model(s) first: {', '.join(f'ollama pull {m}' for m in missing)}")
        return ProviderStatus(name=self.name, label=self.label, available=True,
                              models=list(self.cfg.models), default_model=self.cfg.default_model)

    def graph(self, model: str):
        if model not in self.cfg.models:
            raise ValueError(f"Unknown local model {model!r}")
        if model not in self._graphs:
            self._graphs[model] = create_agent(
                model=self.model_factory(model), tools=self.tools,
                system_prompt=self.agent_cfg.system_prompt, checkpointer=self.checkpointer,
                name=f"llama:{model}")
        return self._graphs[model]

    def _chat_model(self, model: str) -> BaseChatModel:
        kwargs: dict = {"model": model, "base_url": self.cfg.base_url}
        if self.cfg.num_ctx:
            kwargs["num_ctx"] = self.cfg.num_ctx
        return ChatOllama(**kwargs)
