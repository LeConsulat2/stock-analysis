"""Small shared contracts. No SDK, network, or investment policy here."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Instrument:
    key: str
    name: str
    ticker: str | None
    currency: str | None
    timezone: str
    close_hour: int
    close_minute: int = 0
    identity_terms: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    note: str = ""


@dataclass
class Evidence:
    id: str
    kind: str
    subject: str
    source: str
    retrieved_at: str
    as_of: str | None
    data: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    properties: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": self.properties,
                "required": list(self.properties), "additionalProperties": False}


@dataclass
class Call:
    id: str
    name: str
    arguments: Any


@dataclass
class Turn:
    text: str = ""
    calls: list[Call] = field(default_factory=list)
    terminal_error: str | None = None


class Session(Protocol):
    def send(self, results: list[tuple[Call, dict[str, Any]]]) -> Turn: ...


@dataclass
class AgentResult:
    role: str
    provider: str
    model: str
    status: str
    content: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class CompanyReport:
    instrument: Instrument
    snapshot: Evidence
    analysts: list[AgentResult]
    skeptic: AgentResult
    decision: AgentResult
    published_action: str
    blocks: list[str]
    evidence: list[Evidence]
