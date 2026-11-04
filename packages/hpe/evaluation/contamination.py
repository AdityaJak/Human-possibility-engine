"""Contamination and leakage checks (spec §15.1, §38).

* ``InstrumentedSymbolWorld`` records every symbol id drawn for practice, so a run can prove that no
  held-out symbol was ever shown during training.
* ``check_assessment_no_mutation`` proves measurement does not train the policy.
* ``check_eval_sets_fixed`` proves held-out sets are a pure function of the task seed.
* ``run_contamination_suite`` runs all checks for every registered policy class and returns a report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hpe.contracts import registry
from hpe.contracts.conformance import make_init
from hpe.contracts.runtime import Policy
from hpe.core.rng import stream
from hpe.tasks.assessment import assess, policy_state_hash, practice
from hpe.tasks.symbols import ALL_FAMILIES, TRAINING_FAMILIES, Split, SymbolWorld, TaskItem, TaskWorldConfig


@dataclass(slots=True)
class _Ledger:
    practice_symbols: set[int] = field(default_factory=set)
    practice_items: int = 0


class InstrumentedSymbolWorld(SymbolWorld):
    """SymbolWorld that records the symbols of every TRAIN-split item it samples (practice, demonstrations,
    and the train-split generalisation-gap set). Any holdout symbol in this ledger is leakage."""

    def __init__(self, seed: int, config: TaskWorldConfig | None = None) -> None:
        super().__init__(seed=seed, config=config or TaskWorldConfig())
        self.ledger = _Ledger()

    def sample_item(
        self, family: str, rng: np.random.Generator, split: Split = "train", difficulty: float | None = None
    ) -> TaskItem:
        item = super().sample_item(family, rng, split=split, difficulty=difficulty)
        if split == "train":
            self.ledger.practice_symbols.update(int(i) for i in item.symbol_ids)
            self.ledger.practice_items += 1
        return item

    def leakage_counts(self) -> dict[str, int]:
        holdout = {int(i) for i in self.holdout_ids}
        return {
            "practice_items": self.ledger.practice_items,
            "distinct_practice_symbols": len(self.ledger.practice_symbols),
            "holdout_symbols_in_practice": len(self.ledger.practice_symbols & holdout),
        }


def check_assessment_no_mutation(policy: Policy, world: SymbolWorld, age_years: float = 20.0) -> bool:
    before = policy_state_hash(policy)
    for fam in ALL_FAMILIES:
        assess(policy, world, fam, age_years)
    return policy_state_hash(policy) == before


def check_eval_sets_fixed(seed: int) -> bool:
    a, b = SymbolWorld(seed=seed), SymbolWorld(seed=seed)
    return all(
        [it.symbol_ids for it in a.eval_set(f)] == [it.symbol_ids for it in b.eval_set(f)]
        for f in ALL_FAMILIES
    )


def run_contamination_suite(seed: int = 0, episodes: int = 120) -> dict[str, Any]:
    """Train every registered policy class in an instrumented world and audit for leakage."""
    report: dict[str, Any] = {"seed": seed, "episodes_per_family": episodes, "policies": {}}
    report["task_world_leakage"] = SymbolWorld(seed=seed).leakage_report()
    report["eval_sets_fixed"] = check_eval_sets_fixed(seed)
    for name in registry.registered():
        world = InstrumentedSymbolWorld(seed)
        pol = registry.get_factory(name)(make_init(world))
        for fam in TRAINING_FAMILIES:
            practice(
                pol,
                world,
                fam,
                episodes,
                stream(seed, "contam", name, fam),
                agent_id="a0000",
                tick=0,
                age_years=20.0,
                mode="bandit",
            )
        report["policies"][name] = {
            **world.leakage_counts(),
            "assessment_mutates_policy": not check_assessment_no_mutation(pol, world),
        }
    report["clean"] = bool(
        report["task_world_leakage"]["clean"]
        and report["eval_sets_fixed"]
        and all(
            p["holdout_symbols_in_practice"] == 0 and not p["assessment_mutates_policy"]
            for p in report["policies"].values()
        )
    )
    return report
