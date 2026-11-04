"""Optional narration providers (Tier 3). Disabled by default.

A provider may only rewrite the *narrative prose* of an already-built template scenario. Its output
is rejected (and the template kept) if it contains any digit or forbidden language, so an external
model can never introduce numbers, probabilities, or certainty claims. Calls are rate limited and
counted; every external narration is labelled ``narrative_source="llm_external"``.

The ``anthropic`` provider is a disabled stub in v0.1: no network calls are made by this codebase.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from hpe.contracts.schemas import DecisionRequest, Scenario
from hpe.futures.validator import check_external_narration


class NarrationProvider(Protocol):
    name: str
    external: bool

    def narrate(self, scenario: Scenario, request: DecisionRequest) -> str | None: ...


class NoNarration:
    name = "none"
    external = False

    def narrate(self, scenario: Scenario, request: DecisionRequest) -> str | None:
        return None


class MockNarration:
    """Deterministic stand-in for an external model, used to test the external path offline."""

    name = "mock"
    external = True

    def narrate(self, scenario: Scenario, request: DecisionRequest) -> str | None:
        return (
            f"(Mock external narration) Picture a version of you exploring the path '{scenario.title}'. "
            "This is an imaginative sketch built on the assumptions listed, not a forecast."
        )


class AnthropicNarrationStub:
    """Placeholder for an optional Claude-based narrator. Intentionally not implemented in v0.1."""

    name = "anthropic"
    external = True

    def narrate(self, scenario: Scenario, request: DecisionRequest) -> str | None:
        return None  # no network calls in v0.1; template narrative is used


@dataclass
class NarrationGate:
    provider: NarrationProvider
    max_calls_per_hour: int = 0
    calls: int = 0
    rejected: int = 0
    _window: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def apply(self, scenario: Scenario, request: DecisionRequest) -> Scenario:
        if not self.provider.external or self.max_calls_per_hour <= 0:
            return scenario
        with self._lock:
            now = time.monotonic()
            self._window = [t for t in self._window if now - t < 3600.0]
            if len(self._window) >= self.max_calls_per_hour:
                return scenario
            self._window.append(now)
            self.calls += 1
        text = self.provider.narrate(scenario, request)
        if not text:
            return scenario
        if check_external_narration(text):
            self.rejected += 1
            return scenario
        return scenario.model_copy(update={"narrative": text, "narrative_source": "llm_external"})


def build_provider(name: str) -> NarrationProvider:
    name = (name or "none").lower()
    if name == "mock":
        return MockNarration()
    if name == "anthropic":
        return AnthropicNarrationStub()
    return NoNarration()
