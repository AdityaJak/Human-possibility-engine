"""Future Self scenario engine, evidence registry, safety routing (Tier 5).

Owned by the platform/backend lane. Must never import ``hpe.world``, ``hpe.development`` or
``hpe.evaluation``: simulated-society results are not evidence about a real person (ADR-0005).
"""

from hpe.futures.evidence import DEFAULT_REGISTRY, EvidenceRegistry
from hpe.futures.narration import NarrationGate, build_provider
from hpe.futures.safety import classify
from hpe.futures.scenarios import ConsentRequiredError, ScenarioEngine
from hpe.futures.validator import ForbiddenLanguageError, check_scenario, check_text

__all__ = [
    "DEFAULT_REGISTRY",
    "ConsentRequiredError",
    "EvidenceRegistry",
    "ForbiddenLanguageError",
    "NarrationGate",
    "ScenarioEngine",
    "build_provider",
    "check_scenario",
    "check_text",
    "classify",
]
