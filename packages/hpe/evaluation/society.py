"""Society-experiment protocol: twin/multi-arm societies built from matched seeds.

Each (arm, seed) builds ``hpe.world.World`` (imported lazily — owned by the sim/world lane) with:

* ``seed``, ``population_seed`` and ``task_seed`` equal to the experiment seed in EVERY arm, so arms
  are clones of the same initial population and task world (common random numbers, ADR-0002);
* the arm's policy class/params, interventions and world overrides — nothing else differs.

Metrics are computed only from contract-level ``AgentState`` fields (``capabilities[...].measured_*``
from held-out assessment, ``age_years``, ``tutored``, ``job_id``, ``household_id``, ``parents`` ...),
so this module does not depend on world internals. Snapshots can be taken at intermediate years.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from hpe.contracts.protocols import WorldRunner
from hpe.contracts.runtime import PolicyFactory
from hpe.contracts.schemas import AgentState, Intervention, OutcomeObservation, RunManifest, SimulationConfig
from hpe.tasks.symbols import TRAINING_FAMILIES

WorldBuilder = Callable[[SimulationConfig, PolicyFactory, str], WorldRunner]


def default_world_builder(config: SimulationConfig, factory: PolicyFactory, run_id: str) -> WorldRunner:
    mod = importlib.import_module("hpe.world")
    world_cls = mod.World
    return world_cls(config, factory, sink=None, run_id=run_id)  # type: ignore[no-any-return]


def world_available() -> bool:
    try:
        mod = importlib.import_module("hpe.world")
    except ImportError:
        return False
    return hasattr(mod, "World")


# ---------------------------------------------------------------------------------------------
# Metric extractors: AgentState list -> float
# ---------------------------------------------------------------------------------------------


def _acc(a: AgentState, families: Sequence[str]) -> float | None:
    vals = [
        a.capabilities[f].measured_accuracy
        for f in families
        if f in a.capabilities and a.capabilities[f].measured_accuracy is not None
    ]
    return float(np.mean(vals)) if vals else None  # type: ignore[arg-type]


def _select(
    agents: Sequence[AgentState],
    *,
    age_min: float = 0.0,
    age_max: float = 200.0,
    alive: bool | None = True,
    tutored: bool | None = None,
    enrolled: bool | None = None,
) -> list[AgentState]:
    out = []
    for a in agents:
        if alive is not None and a.alive != alive:
            continue
        if not (age_min <= a.age_years <= age_max):
            continue
        if tutored is not None and a.tutored != tutored:
            continue
        if enrolled is not None and a.enrolled_school != enrolled:
            continue
        out.append(a)
    return out


def cohort_mean_accuracy(
    agents: Sequence[AgentState], *, families: Sequence[str] | str = "training", **sel: Any
) -> float:
    fams = (
        list(TRAINING_FAMILIES)
        if families == "training"
        else ([families] if isinstance(families, str) else list(families))
    )
    vals = [v for a in _select(agents, **sel) if (v := _acc(a, fams)) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def cohort_size(agents: Sequence[AgentState], **sel: Any) -> float:
    return float(len(_select(agents, **sel)))


def employment_rate(agents: Sequence[AgentState], **sel: Any) -> float:
    pool = _select(agents, **sel)
    return float(np.mean([a.job_id is not None for a in pool])) if pool else float("nan")


def mean_resources(agents: Sequence[AgentState], **sel: Any) -> float:
    pool = _select(agents, **sel)
    return float(np.mean([a.resources for a in pool])) if pool else float("nan")


def mean_education_years(agents: Sequence[AgentState], **sel: Any) -> float:
    pool = _select(agents, **sel)
    return float(np.mean([a.education_years for a in pool])) if pool else float("nan")


def mean_friends(agents: Sequence[AgentState], **sel: Any) -> float:
    pool = _select(agents, **sel)
    return float(np.mean([a.n_friends for a in pool])) if pool else float("nan")


def tutored_share(agents: Sequence[AgentState], **sel: Any) -> float:
    pool = _select(agents, **sel)
    return float(np.mean([a.tutored for a in pool])) if pool else float("nan")


def population(agents: Sequence[AgentState], **sel: Any) -> float:
    return float(sum(a.alive for a in agents))


def resource_gini(agents: Sequence[AgentState], **sel: Any) -> float:
    pool = _select(agents, **sel)
    x = np.sort(np.clip([a.resources for a in pool], 0, None))
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def _parent_child_corr(agents: Sequence[AgentState], use: str, families: Sequence[str], **sel: Any) -> float:
    """Correlation of child accuracy with the mean accuracy of (a) biological ``parents`` or
    (b) the other adults in the child's rearing ``household``. Assumes the world keeps biological
    parents in ``AgentState.parents`` and the rearing home in ``household_id`` (flag if not)."""
    by_id = {a.agent_id: a for a in agents}
    xs, ys = [], []
    for c in _select(agents, **sel):
        cv = _acc(c, families)
        if cv is None:
            continue
        if use == "biological":
            adults = [by_id[p] for p in c.parents if p in by_id]
        else:
            adults = [
                a
                for a in agents
                if a.household_id == c.household_id and a.agent_id != c.agent_id and a.age_years >= 25
            ]
        vals = [v for a in adults if (v := _acc(a, families)) is not None]
        if vals:
            xs.append(float(np.mean(vals)))
            ys.append(cv)
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def biological_parent_corr(agents: Sequence[AgentState], **kw: Any) -> float:
    fams = kw.pop("families", list(TRAINING_FAMILIES))
    return _parent_child_corr(agents, "biological", fams, **kw)


def rearing_household_corr(agents: Sequence[AgentState], **kw: Any) -> float:
    fams = kw.pop("families", list(TRAINING_FAMILIES))
    return _parent_child_corr(agents, "household", fams, **kw)


def rearing_minus_biological_corr(agents: Sequence[AgentState], **kw: Any) -> float:
    return rearing_household_corr(agents, **dict(kw)) - biological_parent_corr(agents, **dict(kw))


METRIC_FUNCTIONS: dict[str, Callable[..., float]] = {
    "cohort_mean_accuracy": cohort_mean_accuracy,
    "cohort_size": cohort_size,
    "employment_rate": employment_rate,
    "mean_resources": mean_resources,
    "mean_education_years": mean_education_years,
    "mean_friends": mean_friends,
    "tutored_share": tutored_share,
    "population": population,
    "resource_gini": resource_gini,
    "biological_parent_corr": biological_parent_corr,
    "rearing_household_corr": rearing_household_corr,
    "rearing_minus_biological_corr": rearing_minus_biological_corr,
}


# ---------------------------------------------------------------------------------------------
# Running one (arm, seed)
# ---------------------------------------------------------------------------------------------


def build_config(
    base: SimulationConfig,
    *,
    seed: int,
    policy_class: str,
    policy_params: dict[str, Any],
    interventions: Sequence[Intervention],
    world_overrides: dict[str, Any],
    params: dict[str, Any],
    label: str,
) -> SimulationConfig:
    """Arm config = base config + arm settings. Seeds are identical across arms (matched clones)."""
    data = base.model_dump(mode="json")
    data.update(
        seed=seed,
        population_seed=seed,
        task_seed=seed,
        policy_class=policy_class,
        policy_params=dict(policy_params),
        label=label,
    )
    data["world"] = {**data["world"], **world_overrides, **params.get("world_overrides", {})}
    data["interventions"] = list(data["interventions"]) + [i.model_dump(mode="json") for i in interventions]
    for k in ("years", "n_agents", "dt_days"):
        if k in params:
            data[k] = params[k]
    return SimulationConfig.model_validate(data)


def run_society(
    config: SimulationConfig,
    factory: PolicyFactory,
    *,
    run_id: str,
    experiment_id: str,
    arm: str,
    params: dict[str, Any],
    builder: WorldBuilder | None = None,
) -> tuple[list[OutcomeObservation], RunManifest | None, dict[str, Any]]:
    """Run one society and extract preregistered metrics. Returns (outcomes, manifest, extra)."""
    world = (builder or default_world_builder)(config, factory, run_id)
    metrics: list[dict[str, Any]] = list(params.get("metrics", []))
    snap_years = sorted({float(m.get("at_year", config.years)) for m in metrics} | {float(config.years)})
    out: list[OutcomeObservation] = []
    extra: dict[str, Any] = {}

    def obs(
        metric: str,
        value: float,
        *,
        agent_id: str | None = None,
        age: float | None = None,
        tick: int | None = None,
    ) -> OutcomeObservation:
        return OutcomeObservation(
            run_id=run_id,
            experiment_id=experiment_id,
            arm=arm,
            seed=config.seed,
            agent_id=agent_id,
            metric=metric,
            value=float(value),
            tick=tick,
            age_years=age,
            split="holdout",
            source="simulation",
        )

    for y in snap_years:
        if y > world.year + 1e-9:
            world.run(until_year=y)
        agents = world.agents(include_dead=True)
        for m in metrics:
            if abs(float(m.get("at_year", config.years)) - y) > 1e-9:
                continue
            fn = METRIC_FUNCTIONS[m["fn"]]
            out.append(obs(m["name"], fn(agents, **dict(m.get("args", {}))), tick=world.tick))
        if params.get("record_agent_accuracy", True):
            for a in agents:
                if a.alive and (v := _acc(a, list(TRAINING_FAMILIES))) is not None:
                    out.append(
                        obs(f"agent_accuracy@{y:g}", v, agent_id=a.agent_id, age=a.age_years, tick=world.tick)
                    )
    manifest: RunManifest | None = None
    try:
        manifest = world.manifest()
        extra["event_stream_hash"] = world.event_stream_hash()
    except Exception as exc:
        extra["manifest_error"] = repr(exc)
    return out, manifest, extra
