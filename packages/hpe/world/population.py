"""Initial population, trait inheritance and descriptive persona text.

Traits, base parameters and value weights are abstract model parameters in [0, 1]. They are
*not* personality measurements, and no demographic attribute (sex, ethnicity, etc.) is
modelled at all, so none can be hard-coded as destiny. Persona text is a generated, purely
descriptive biography: it is what the static-persona baseline "knows" about itself and is
never used by the world to decide outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRAIT_KEYS: tuple[str, ...] = ("curiosity", "diligence", "sociability", "risk_tolerance", "patience")
VALUE_KEYS: tuple[str, ...] = (
    "autonomy",
    "income",
    "relationships",
    "stability",
    "learning",
    "health",
    "time",
)
VALUE_LABELS: dict[str, str] = {
    "autonomy": "autonomy",
    "income": "income",
    "relationships": "relationships",
    "stability": "stability",
    "learning": "learning",
    "health": "health",
    "time": "free time",
}

_NAME_SYLLABLES = (
    "an",
    "be",
    "ci",
    "do",
    "el",
    "fa",
    "go",
    "hi",
    "is",
    "jo",
    "ka",
    "lu",
    "ma",
    "ne",
    "or",
    "pa",
    "qui",
    "ra",
    "so",
    "ta",
    "ul",
    "vi",
    "wo",
    "xa",
    "ye",
    "zu",
)


@dataclass(slots=True)
class PersonSpec:
    """Initial-condition draft for one agent (before ids/policies are assigned)."""

    age: float
    traits: dict[str, float]
    base_params: dict[str, float]
    values: dict[str, float]
    name: str
    parents_idx: list[int] = field(default_factory=list)
    partner_idx: int | None = None


@dataclass(slots=True)
class HouseholdSpec:
    members: list[PersonSpec]
    resources: float


def draw_traits(rng: np.random.Generator) -> dict[str, float]:
    return {k: float(np.clip(rng.beta(4, 4), 0.0, 1.0)) for k in TRAIT_KEYS}


def draw_base_params(rng: np.random.Generator) -> dict[str, float]:
    return {
        "learning_rate_scale": float(np.clip(rng.lognormal(0.0, 0.25), 0.4, 2.5)),
        "memory_capacity": float(int(np.clip(rng.normal(256, 48), 96, 512))),
        "exploration": float(np.clip(rng.beta(2, 12), 0.01, 0.5)),
    }


def normalise_values(v: dict[str, float]) -> dict[str, float]:
    s = sum(max(0.0, x) for x in v.values()) or 1.0
    return {k: max(0.0, float(v[k])) / s for k in VALUE_KEYS}


def draw_values(rng: np.random.Generator) -> dict[str, float]:
    raw = rng.dirichlet(np.full(len(VALUE_KEYS), 2.0))
    return normalise_values({k: float(x) for k, x in zip(VALUE_KEYS, raw, strict=True)})


def draw_name(rng: np.random.Generator) -> str:
    n = int(rng.integers(2, 4))
    return "".join(str(s) for s in rng.choice(_NAME_SYLLABLES, size=n)).capitalize()


def inherit_traits(
    p_traits: list[dict[str, float]], rng: np.random.Generator, noise: float = 0.12
) -> dict[str, float]:
    """Child traits = parents' mean + noise (an inherited *initial condition*, not acquired skill)."""
    out = {}
    for k in TRAIT_KEYS:
        mean = float(np.mean([t[k] for t in p_traits])) if p_traits else 0.5
        out[k] = float(np.clip(mean + rng.normal(0, noise), 0.0, 1.0))
    return out


def inherit_base_params(p_params: list[dict[str, float]], rng: np.random.Generator) -> dict[str, float]:
    fresh = draw_base_params(rng)
    if not p_params:
        return fresh
    out = {}
    for k, v in fresh.items():
        mean = float(np.mean([p[k] for p in p_params]))
        out[k] = 0.5 * mean + 0.5 * v
    out["memory_capacity"] = float(round(out["memory_capacity"]))
    return out


def inherit_values(p_values: list[dict[str, float]], rng: np.random.Generator) -> dict[str, float]:
    fresh = draw_values(rng)
    if not p_values:
        return fresh
    out = {k: 0.6 * float(np.mean([p[k] for p in p_values])) + 0.4 * fresh[k] for k in VALUE_KEYS}
    return normalise_values(out)


def _adjective(trait: str, v: float) -> str | None:
    hi = {
        "curiosity": "curious",
        "diligence": "diligent",
        "sociability": "sociable",
        "risk_tolerance": "adventurous",
        "patience": "patient",
    }
    lo = {
        "curiosity": "incurious",
        "diligence": "easygoing",
        "sociability": "reserved",
        "risk_tolerance": "cautious",
        "patience": "impatient",
    }
    if v >= 0.62:
        return hi[trait]
    if v <= 0.38:
        return lo[trait]
    return None


def persona_text(
    name: str, household_id: str, traits: dict[str, float], values: dict[str, float], age: float
) -> str:
    adjs = [a for k in TRAIT_KEYS if (a := _adjective(k, traits[k]))]
    top = sorted(VALUE_KEYS, key=lambda k: (-values[k], k))[:2]
    desc = ", ".join(adjs) if adjs else "even-tempered"
    return (
        f"{name} of household {household_id}, {age:.0f} years old at entry. Described as {desc}. "
        f"Cares most about {VALUE_LABELS[top[0]]} and {VALUE_LABELS[top[1]]}. "
        "(Generated description of model parameters; not a real person.)"
    )


def _person(rng: np.random.Generator, age: float) -> PersonSpec:
    return PersonSpec(
        age=float(age),
        traits=draw_traits(rng),
        base_params=draw_base_params(rng),
        values=draw_values(rng),
        name=draw_name(rng),
    )


def initial_households(
    n_agents: int, n_households: int | None, distribution: str, rng: np.random.Generator
) -> list[HouseholdSpec]:
    """Return household drafts containing exactly ``n_agents`` people."""
    n_hh = max(1, min(n_agents, n_households or max(1, n_agents // 4)))
    hhs: list[HouseholdSpec] = []

    def resources(n_members: int) -> float:
        return float(rng.lognormal(np.log(4.0), 0.6) * max(1, n_members))

    if distribution == "uniform_0_60":
        people = [_person(rng, rng.uniform(0, 60)) for _ in range(n_agents)]
        buckets: list[list[PersonSpec]] = [[] for _ in range(n_hh)]
        for i, p in enumerate(people):
            buckets[i % n_hh].append(p)
        for b in buckets:
            if not b:
                continue
            _link_family(b)
            hhs.append(HouseholdSpec(b, resources(len(b))))
        return hhs

    if distribution == "newborn_cohort":
        # one third caregiving adults, two thirds newborns (all counted in n_agents)
        n_adults = max(1, n_agents // 3)
        n_hh = max(1, min(n_hh, max(1, n_adults // 2)))
        adults = [_person(rng, rng.uniform(24, 40)) for _ in range(n_adults)]
        babies = [_person(rng, rng.uniform(0, 0.25)) for _ in range(n_agents - n_adults)]
        buckets = [[] for _ in range(n_hh)]
        for i, p in enumerate(adults):
            buckets[i % n_hh].append(p)
        for i, p in enumerate(babies):
            buckets[i % n_hh].append(p)
        for b in buckets:
            if b:
                _link_family(b)
                hhs.append(HouseholdSpec(b, resources(len(b))))
        return hhs

    # young_families (default): ~15% elder households, the rest couples with children.
    n_elder_hh = int(round(0.15 * n_hh)) if n_hh >= 4 else 0
    n_family_hh = n_hh - n_elder_hh
    remaining = n_agents
    elder_buckets: list[list[PersonSpec]] = []
    for _ in range(n_elder_hh):
        k = min(remaining - 2 * n_family_hh, int(rng.integers(1, 3)))
        if k <= 0:
            break
        elder_buckets.append([_person(rng, rng.uniform(62, 85)) for _ in range(k)])
        remaining -= k
    family_buckets: list[list[PersonSpec]] = []
    for _ in range(n_family_hh):
        if remaining <= 0:
            break
        k = min(2, remaining)
        a1 = rng.uniform(24, 45)
        family_buckets.append([_person(rng, float(np.clip(a1 + rng.normal(0, 3), 22, 47))) for _ in range(k)])
        remaining -= k
    i = 0
    while remaining > 0 and family_buckets:
        fam = family_buckets[i % len(family_buckets)]
        youngest_parent = min(p.age for p in fam[:2])
        max_child_age = max(0.0, min(15.0, youngest_parent - 20.0))
        fam.append(_person(rng, rng.uniform(0, max_child_age) if max_child_age > 0 else 0.0))
        remaining -= 1
        i += 1
    while remaining > 0:  # degenerate tiny populations
        elder_buckets.append([_person(rng, rng.uniform(25, 60))])
        remaining -= 1
    for b in family_buckets + elder_buckets:
        _link_family(b)
        hhs.append(HouseholdSpec(b, resources(len(b))))
    return hhs


def _link_family(members: list[PersonSpec]) -> None:
    """Pair the two oldest adults as partners; adults >=18y older than a member become parents."""
    adults = sorted((i for i, p in enumerate(members) if p.age >= 20), key=lambda i: (-members[i].age, i))
    if len(adults) >= 2 and abs(members[adults[0]].age - members[adults[1]].age) < 15:
        a, b = adults[0], adults[1]
        members[a].partner_idx = b
        members[b].partner_idx = a
    for _i, p in enumerate(members):
        if p.age >= 20:
            continue
        parents = [j for j in adults[:2] if members[j].age - p.age >= 18]
        p.parents_idx = parents
