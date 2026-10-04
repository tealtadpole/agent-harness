from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass
class ProviderStatus:
    name: str
    label: str
    available: bool
    detail: str = ""
    models: list[str] = field(default_factory=list)
    default_model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Provider(Protocol):
    name: str
    label: str

    async def status(self) -> ProviderStatus: ...

    def graph(self, model: str):
        """A compiled LangGraph graph (with checkpointer) that answers for `model`."""
        ...
