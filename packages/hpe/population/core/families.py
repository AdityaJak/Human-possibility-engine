"""Family registry: which studies form a family, how they are parsed, and which partition file governs them.

Roles are fixed by docs/phase3/PHASE3_RESEARCH_THESIS.md §3 and must not be edited after a freeze.
Adapters for new studies live in ``hpe.population.families`` and register themselves on import.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    # Non-Psych-101 datasets supply their own locked loader: (partition, experiment_id, k_max) -> (seqs, notes).
    # It must apply the same Phase III test lock (check_partition_access with the Phase III freeze dir).
    loader: Callable[[str, str, int], tuple[list[Any], list[str]]] | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}")


FAMILIES: dict[str, FamilySpec] = {}

# Canonical owner of every study prefix covered by the pinned partition files (critic finding T-2): a family
# registered under another name over any of these studies is refused, whether or not the canonical family
# has been imported yet.
CANONICAL_PREFIXES: dict[str, str] = {
    **{p: n for n, ps in psych101_v2.PARADIGM_PREFIXES.items() for p in ps},
    "bahrami2020four/": "restless4",
    "kool2016when/": "two_step",
    "kool2017cost/": "two_step",
    "zorowitz2023data/": "two_step",
    "schulz2020finding/": "spatial_bandit",
    "sadeghiyeh2020temporal/": "horizon_external",
    "somerville2017charting/": "horizon_external",
    "feng2021dynamics/": "horizon_external",
    "collsiöö2023MCPL/": "mcpl",
    "steingroever2015data/": "igt",
}


def register_family(spec: FamilySpec) -> None:
    if spec.name in FAMILIES and FAMILIES[spec.name] != spec:
        raise ValueError(f"family {spec.name!r} already registered differently")
    for prefix in spec.prefixes:
        owner = next(
            (n for p, n in CANONICAL_PREFIXES.items() if p.startswith(prefix) or prefix.startswith(p)), None
        )
        if owner is not None and owner != spec.name:
            raise ValueError(f"family {spec.name!r} overlaps the studies of {owner!r}")
    for other in FAMILIES.values():  # critic finding T-2: no second name over the same studies
        if other.name != spec.name and any(
            a.startswith(b) or b.startswith(a) for a in spec.prefixes for b in other.prefixes
        ):
            raise ValueError(f"family {spec.name!r} overlaps the studies of {other.name!r}")
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
