"""Browser bridge for the Future Self engine (runs under Pyodide; also plain CPython).

The static deployment has no API server. Instead of porting the engine to TypeScript, the web
app installs this package into Pyodide and calls the two functions below. They build the engine
exactly as ``hpe_api.context`` does (default evidence registry, narration provider ``none``) so
the output is the same JSON the API returns for ``POST /scenarios``. Pure Python: no FastAPI,
no storage, no network. User-private persistence (consent, decisions, reflections) is handled by
the browser in ``apps/web/src/api/static/storage.ts``.
"""

from __future__ import annotations

import json

from hpe.contracts.schemas import DecisionRequest
from hpe.futures.evidence import DEFAULT_REGISTRY
from hpe.futures.narration import NarrationGate, build_provider
from hpe.futures.scenarios import ScenarioEngine

CONSENT_MESSAGE = (
    "consent_given must be true: the Future Self demo needs explicit consent before any processing"
)


class ConsentRequired(ValueError):
    """Raised when ``consent_given`` is false (the API answers 400 with the same message)."""


_ENGINE = ScenarioEngine(
    DEFAULT_REGISTRY, narration=NarrationGate(build_provider("none"), max_calls_per_hour=0)
)


def generate(request_json: str, decision_id: str | None = None) -> str:
    """DecisionRequest JSON → ScenarioResponse JSON, with the API's semantics.

    * ``consent_given`` false → ``ConsentRequired`` (API: 400).
    * invalid request → ``pydantic.ValidationError`` (API: 422).
    * crisis language → a response with no scenarios and a crisis notice (API: 200).
    * template text that violates the no-probability validator → ``ForbiddenLanguageError`` (API: 500).
    ``decision_id`` lets callers (and tests) fix the otherwise random id.
    """
    request = DecisionRequest.model_validate_json(request_json)
    if not request.consent_given:
        raise ConsentRequired(CONSENT_MESSAGE)
    response = _ENGINE.generate(request, decision_id=decision_id)
    return response.model_dump_json()


def evidence_catalog() -> str:
    """The static evidence/assumption library (what ``GET /evidence`` lists)."""
    return json.dumps([r.model_dump(mode="json") for r in DEFAULT_REGISTRY.all()])
