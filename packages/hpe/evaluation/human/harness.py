"""Teacher-forced held-out likelihood evaluation of policy classes on human choice data.

Protocol (preregistered in experiments/exp07_human_heldout.yaml):

1. Participant-level split (stratified by experiment, seeded) into TRAIN and TEST.
2. For every model and every hyperparameter setting, run a *fresh* policy through each TRAIN
   participant's trials: before each trial record p(human's actual choice) from
   ``action_probabilities``; then call ``learn`` with the human's actual action and the outcome
   the human saw (teacher forcing). Policies never choose actions here.
3. A lapse rate eps mixes predictions with uniform: p' = (1-eps) p + eps/K (fit like any other
   hyperparameter; applied identically to every model).
4. Select hyperparameters minimising mean per-participant NLL on TRAIN only; evaluate once on
   TEST. The "strongest baseline" for each contrast is also chosen on TRAIN.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hpe.contracts import registry
from hpe.contracts.runtime import ACTIVITY_ACTIONS, AgentInit, Policy
from hpe.core.rng import derive_seed, stream
from hpe.evaluation.human.definition import HumanHeldoutDefinition
from hpe.evaluation.human.models import (
    FAMILIES_META,
    IGT_ACTIONS,
    MCPL_ACTIONS,
    REFERENCE_MODELS,
    Step,
    encode,
    reference_factory,
)
from hpe.evaluation.human.psych101 import Participant
from hpe.tasks.symbols import SymbolWorld

Factory = Callable[[AgentInit], Policy]


class ModelUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class ModelResult:
    model: str
    status: str  # "ok" | "unavailable" | "error"
    note: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    lapse: float | None = None
    train_objective: float | None = None
    n_grid: int = 0
    test_nll: dict[str, float] = field(default_factory=dict)  # participant key -> mean NLL/choice
    test_top1: dict[str, float] = field(default_factory=dict)
    test_n_choices: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def split_participants(
    parts: list[Participant], train_fraction: float, seed: int
) -> tuple[list[Participant], list[Participant]]:
    train: list[Participant] = []
    test: list[Participant] = []
    for exp in sorted({p.experiment for p in parts}):
        group = [p for p in parts if p.experiment == exp]
        order = stream(seed, "human_split", exp).permutation(len(group))
        n_train = int(round(train_fraction * len(group)))
        for rank, idx in enumerate(order):
            (train if rank < n_train else test).append(group[int(idx)])
    key = lambda p: p.key  # noqa: E731
    return sorted(train, key=key), sorted(test, key=key)


def make_init(participant_key: str, seed: int) -> AgentInit:
    families = {**SymbolWorld(seed=0).families_meta(), **FAMILIES_META}
    return AgentInit(
        agent_id=participant_key,
        birth_tick=0,
        seed=derive_seed(seed, "human", participant_key),
        traits={
            "curiosity": 0.5,
            "diligence": 0.5,
            "sociability": 0.5,
            "risk_tolerance": 0.5,
            "patience": 0.5,
        },
        base_params={"learning_rate_scale": 1.0, "memory_capacity": 256.0, "exploration": 0.1},
        families=families,
        activity_actions=list(ACTIVITY_ACTIONS),
    )


def run_participant(factory: Factory, steps: list[Step], init: AgentInit) -> tuple[np.ndarray, np.ndarray]:
    """Return (p(human choice), top-1 credit) per trial under teacher forcing."""
    pol = factory(init)
    n = len(steps)
    pc = np.empty(n)
    top = np.empty(n)
    for i, (obs, action, fb) in enumerate(steps):
        p = np.asarray(pol.action_probabilities(obs), dtype=np.float64)
        pc[i] = p[action]
        is_max = np.isclose(p, p.max(), rtol=0, atol=1e-12)
        top[i] = is_max[action] / is_max.sum()
        pol.learn(obs, action, fb)
    return pc, top


def lapse_nll(pc: np.ndarray, lapse: float, k: int) -> float:
    p = (1.0 - lapse) * pc + lapse / k
    return float(-np.log(np.clip(p, 1e-12, 1.0)).mean())


def _grid(spec: dict[str, list[float]] | None) -> list[dict[str, Any]]:
    if not spec:
        return [{}]
    keys = sorted(spec)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(spec[k] for k in keys))]


def _factory_for(model: str, params: dict[str, Any], train: list[Participant], paradigm: str) -> Factory:
    if model in REFERENCE_MODELS:
        return reference_factory(model, params, train, paradigm)
    try:
        return registry.get_factory(model, params)
    except (KeyError, ImportError) as e:  # lane not integrated in this build
        raise ModelUnavailable(
            f"policy class {model!r} not registered in this build ({type(e).__name__}: {e})"
        ) from e


def evaluate_model(
    model: str,
    paradigm: str,
    train: list[Participant],
    test: list[Participant],
    defn: HumanHeldoutDefinition,
    *,
    seed: int,
) -> ModelResult:
    t0 = time.perf_counter()
    k = MCPL_ACTIONS if paradigm == "mcpl" else IGT_ACTIONS
    if model == "ref_population_static" and paradigm != "mcpl":
        return ModelResult(model, "unavailable", note="defined for MCPL only")
    base_grid = _grid(defn.grids.get(model, {}).get(paradigm))
    scales = defn.igt_reward_scale_grid if paradigm == "igt" else [0.0]
    enc_cache: dict[tuple[str, float], list[Step]] = {}

    def steps_for(p: Participant, scale: float) -> list[Step]:
        key = (p.key, scale)
        if key not in enc_cache:
            enc_cache[key] = encode(p, scale=scale or 500.0, age_years=defn.age_years)
        return enc_cache[key]

    best: tuple[float, dict[str, Any], float] | None = None
    n_grid = 0
    try:
        for scale in scales:
            for params in base_grid:
                factory = _factory_for(model, params, train, paradigm)
                pcs = [
                    run_participant(factory, steps_for(p, scale), make_init(p.key, seed))[0] for p in train
                ]
                for lapse in defn.lapse_grid:
                    n_grid += 1
                    obj = float(np.mean([lapse_nll(pc, lapse, k) for pc in pcs]))
                    full = {**params, **({"igt_reward_scale": scale} if paradigm == "igt" else {})}
                    if best is None or obj < best[0] - 1e-12:
                        best = (obj, full, lapse)
    except ModelUnavailable as e:
        return ModelResult(model, "unavailable", note=str(e), seconds=time.perf_counter() - t0)
    except (
        Exception
    ) as e:  # incompatibility of a policy class with the task encoding is a result, not a crash
        return ModelResult(model, "error", note=f"{type(e).__name__}: {e}", seconds=time.perf_counter() - t0)

    assert best is not None
    obj, full, lapse = best
    scale = float(full.get("igt_reward_scale", 0.0))
    params = {kk: v for kk, v in full.items() if kk != "igt_reward_scale"}
    factory = _factory_for(model, params, train, paradigm)
    res = ModelResult(model, "ok", params=full, lapse=lapse, train_objective=obj, n_grid=n_grid)
    for p in test:
        pc, top = run_participant(factory, steps_for(p, scale), make_init(p.key, seed))
        res.test_nll[p.key] = lapse_nll(pc, lapse, k)
        res.test_top1[p.key] = float(top.mean())
        res.test_n_choices[p.key] = len(pc)
    res.seconds = time.perf_counter() - t0
    return res


def evaluate_paradigm(
    paradigm: str,
    parts: list[Participant],
    defn: HumanHeldoutDefinition,
    *,
    split_seed: int,
    log: Callable[[str], None] = print,
) -> tuple[list[ModelResult], list[Participant], list[Participant]]:
    train, test = split_participants(parts, defn.split.train_fraction, split_seed)
    log(f"[{paradigm}] split seed {split_seed}: {len(train)} train / {len(test)} test participants")
    results = []
    for model in defn.models:
        r = evaluate_model(model, paradigm, train, test, defn, seed=split_seed)
        msg = f"  {model:22s} {r.status:11s}"
        if r.status == "ok":
            msg += f" train={r.train_objective:.4f} test={np.mean(list(r.test_nll.values())):.4f} params={r.params} lapse={r.lapse} ({r.seconds:.1f}s)"
        else:
            msg += f" {r.note}"
        log(msg)
        results.append(r)
    return results, train, test
