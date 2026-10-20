"""Data types shared across the router."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class OutputSchema:
    """Expected structured output.

    `model` validates the parsed JSON. `check` adds cross-field rules and returns an
    error string or None. `json_schema` is sent to providers that take a schema (the
    Bedrock tool definition); it is not used for validation. `normalize_tool_input` is
    applied to a tool call's arguments before they are serialized to text.
    """

    name: str
    description: str
    model: type[BaseModel]
    json_schema: dict[str, Any]
    check: Callable[[BaseModel], str | None] | None = None
    normalize_tool_input: Callable[[dict[str, Any]], dict[str, Any]] | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None

    @property
    def billed_output_tokens(self) -> int | None:
        if self.output_tokens is None and self.thinking_tokens is None:
            return None
        return (self.output_tokens or 0) + (self.thinking_tokens or 0)


@dataclass
class Attempt:
    """One provider request (the initial call or the repair call)."""

    kind: Literal["initial", "repair"]
    cache_key: str
    cache_hit: bool
    text: str | None = None
    parse_error: str | None = None
    error: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: float | None = None
    provider_attempts: int = 0
    cost_usd: float | None = None


@dataclass
class Response:
    provider: str
    model_id: str
    model_version: str | None
    text: str | None
    parsed: dict[str, Any] | None
    parse_error: str | None
    error: str | None
    repaired: bool
    attempts: list[Attempt]

    @property
    def ok(self) -> bool:
        return self.parsed is not None and self.error is None

    @property
    def cache_hit(self) -> bool:
        return all(a.cache_hit for a in self.attempts)

    @property
    def usage(self) -> Usage:
        def total(attr: str) -> int | None:
            values = [getattr(a.usage, attr) for a in self.attempts]
            known = [v for v in values if v is not None]
            return sum(known) if known else None

        return Usage(total("input_tokens"), total("output_tokens"), total("thinking_tokens"))

    @property
    def latency_ms(self) -> float | None:
        known = [a.latency_ms for a in self.attempts if a.latency_ms is not None]
        return sum(known) if known else None

    @property
    def cost_usd(self) -> float | None:
        known = [a.cost_usd for a in self.attempts]
        return None if any(c is None for c in known) else sum(known)  # type: ignore[arg-type]
