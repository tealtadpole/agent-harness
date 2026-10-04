"""Claude through the Anthropic API: a LangChain tool-calling agent compiled as a LangGraph graph."""

from __future__ import annotations

from collections.abc import Callable

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.types import Checkpointer

from ..config import AgentConfig, ClaudeConfig
from .base import ProviderStatus

# Models whose API accepts `output_config.effort`.
NO_EFFORT_PREFIXES = ("claude-haiku-4-5",)
# Models that support the server-side refusal fallback (`fallbacks: "default"`).
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeProvider:
    name = "claude"
    label = "Claude (Anthropic API)"

    def __init__(self, cfg: ClaudeConfig, agent: AgentConfig, tools: list[BaseTool],
                 checkpointer: Checkpointer,
                 model_factory: Callable[[str], BaseChatModel] | None = None):
        self.cfg = cfg
        self.agent_cfg = agent
        self.tools = tools
        self.checkpointer = checkpointer
        self.model_factory = model_factory or self._chat_model
        self._graphs: dict[str, object] = {}

    async def status(self) -> ProviderStatus:
        ok = bool(self.cfg.api_key) or self.model_factory != self._chat_model
        return ProviderStatus(
            name=self.name, label=self.label, available=ok,
            detail="" if ok else f"Set ${self.cfg.api_key_env} to enable Claude.",
            models=list(self.cfg.models), default_model=self.cfg.default_model)

    def graph(self, model: str):
        if model not in self.cfg.models:
            raise ValueError(f"Unknown Claude model {model!r}")
        if model not in self._graphs:
            self._graphs[model] = create_agent(
                model=self.model_factory(model), tools=self.tools,
                system_prompt=self.agent_cfg.system_prompt, checkpointer=self.checkpointer,
                name=f"claude:{model}")
        return self._graphs[model]

    def _chat_model(self, model: str) -> BaseChatModel:
        kwargs: dict = {
            "model": model,
            "api_key": self.cfg.api_key,
            "max_tokens": self.cfg.max_tokens,
            "streaming": True,
        }
        if self.cfg.base_url:
            kwargs["anthropic_api_url"] = self.cfg.base_url
        if self.cfg.effort and not model.startswith(NO_EFFORT_PREFIXES):
            kwargs["output_config"] = {"effort": self.cfg.effort}
        if self.cfg.refusal_fallback and model in FALLBACK_MODELS and not self.cfg.base_url:
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["model_kwargs"] = {"fallbacks": "default"}
        return ChatAnthropic(**kwargs)
