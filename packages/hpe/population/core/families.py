"""Family registry: which studies form a family, how they are parsed, and which partition file governs them.

Roles are fixed by docs/phase3/PHASE3_RESEARCH_THESIS.md §3 and must not be edited after a freeze.
Adapters for new studies live in ``hpe.population.families`` and register themselves on import.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hpe.evaluation.human import psych101_v2
from hpe.evaluation.human.psych101 import Participant

REPO_ROOT = Path(__file__).resolve().parents[4]
P2_PARTITIONS = REPO_ROOT / "reports" / "phase2" / "partitions" / "psych101.json"
P3_PARTITIONS = REPO_ROOT / "reports" / "phase3" / "partitions" / "psych101_p3.json"

Parser = Callable[[str, str, str], Participant]

ROLES = ("counted_primary", "counted", "counted_if_validated", "secondary", "external", "contaminated")


@dataclass(frozen=True, slots=True)
class FamilySpec:
    name: str
    prefixes: tuple[str, ...]
    n_actions: int
    partition_file: Path
    parser: Parser
    role: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}")


FAMILIES: dict[str, FamilySpec] = {}


def register_family(spec: FamilySpec) -> None:
    if spec.name in FAMILIES and FAMILIES[spec.name] != spec:
        raise ValueError(f"family {spec.name!r} already registered differently")
    FAMILIES[spec.name] = spec


def get_family(name: str) -> FamilySpec:
    if name not in FAMILIES:
        import hpe.population.families  # noqa: F401  (adapters register on import)
    if name not in FAMILIES:
        raise KeyError(f"unknown family {name!r}; known: {sorted(FAMILIES)}")
    return FAMILIES[name]


def counted_families() -> list[str]:
    import hpe.population.families  # noqa: F401

    return sorted(n for n, s in FAMILIES.items() if s.role.startswith("counted"))


# Phase II generic families: parsers from hpe.evaluation.human.psych101_v2, partitions from Phase II.
_P2_ROLES = {
    "garcia": "counted",
    "described": "counted",
    "bandit2": "secondary",
    "srl3": "secondary",
    "horizon": "secondary",
    "casino": "secondary",
    "wpt": "secondary",
}
for _name, _role in _P2_ROLES.items():
    register_family(
        FamilySpec(
            name=_name,
            prefixes=psych101_v2.PARADIGM_PREFIXES[_name],
            n_actions=psych101_v2.PARADIGM_ACTIONS[_name],
            partition_file=P2_PARTITIONS,
            parser=psych101_v2.PARSERS[_name],
            role=_role,
        )
    )
