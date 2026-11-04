"""Gate B family 3 — ``composition`` ("rule-of-rules"), authored by a candidate-free lane.

Spec: ``docs/phase2/specs/GATE_B_FAMILY_3_AND_GATE_A_CONTROLS.md`` §1. Same contracts and held-out
discipline as ``hpe.tasks.SymbolWorld``: ±1 feature symbols with generated names, disjoint
train/holdout symbol pools, fixed holdout-only assessment sets, label noise on feedback only.
``CompositionWorld`` is duck-typed to ``SymbolWorld`` so ``hpe.tasks.practice`` / ``assess`` /
``transfer_probe`` and ``hpe.evaluation.learning_lab`` (``task_world="composition"``) work unchanged.

Primitives (drawn from the task seed):

* ``r1(x)`` — a *conjunction* over feature group A (features 0–5): ``n_literals`` seeded literals
  ``x[i] == s_i`` must all hold.
* ``r2(x)`` — a *parity* over feature group B (features 6–11): the product of a seeded subset of
  group-B features, with a seeded sign.
* ``op`` — one composition operator per task seed from {AND, OR, XOR, IMPLIES}.

Families (``n_actions = 2`` everywhere):

* ``primitive_a``       correct = r1(x)
* ``primitive_b``       correct = r2(x)
* ``composition``       correct = op(r1(x), r2(x))                       (single slot)
* ``composition_pair``  two symbols, exactly one satisfies the composition; correct = its slot

Shared vs specialist structure. With ``shared_structure=True`` the composition uses the *same*
primitives the primitive families teach, so a learner that has abstracted r1 and r2 can compose
them. With ``shared_structure=False`` (the specialist-should-win control) the composition is built
from independently drawn primitives, so nothing learned on ``primitive_a``/``primitive_b`` transfers.

Class balance. The composed label is unbalanced for AND/IMPLIES (P ≈ 1/8 with two literals), so
items are sampled **label-stratified** from the pools (half 0, half 1, seeded) for practice and for
the fixed evaluation sets; the symbol pools themselves stay fixed and disjoint, and the pool drawing
is rejected until every pool holds at least ``min_per_label`` symbols of each label for every
family. Consequently a majority-class policy scores 0.5 on every family.

Difficulty proxy. ``TaskItem.margin = n_features - n_flip`` where ``n_flip`` is the number of single
feature flips that change the composed label (fewer flips = farther from the boundary = easier).
Pre-registered expectation (not a claim): a linear raw-feature learner cannot exceed ≈ 0.75 on
``composition`` because the parity component is non-linear; the measured ceiling is recorded by
``reports/phase2/evaluation_v2/scripts/composition_ceiling.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from hpe.contracts.runtime import Feedback, FeedbackMode, FeedbackSource, Observation
from hpe.core.hashing import stable_hash
from hpe.core.rng import stream
from hpe.tasks.symbols import _SYLLABLES, TaskItem

Split = Literal["train", "holdout"]

COMPOSITION_FAMILIES: tuple[str, ...] = ("primitive_a", "primitive_b", "composition", "composition_pair")
COMPOSITION_SPECS: dict[str, dict[str, int]] = {
    "primitive_a": {"n_slots": 1, "n_actions": 2},
    "primitive_b": {"n_slots": 1, "n_actions": 2},
    "composition": {"n_slots": 1, "n_actions": 2},
    "composition_pair": {"n_slots": 2, "n_actions": 2},
}
OPS: tuple[str, ...] = ("AND", "OR", "XOR", "IMPLIES")
GROUP_A = tuple(range(0, 6))
GROUP_B = tuple(range(6, 12))


def compose(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_, b_ = a.astype(bool), b.astype(bool)
    if op == "AND":
        out = a_ & b_
    elif op == "OR":
        out = a_ | b_
    elif op == "XOR":
        out = a_ ^ b_
    elif op == "IMPLIES":
        out = (~a_) | b_
    else:  # pragma: no cover - guarded by OPS
        raise KeyError(op)
    return out.astype(np.int64)


@dataclass(frozen=True, slots=True)
class Primitives:
    """One pair of primitive rules (r1 conjunction literals/signs, r2 parity subset/sign)."""

    lit_a: tuple[int, ...]
    sign_a: tuple[int, ...]
    sub_b: tuple[int, ...]
    sign_b: int

    def r1(self, x: np.ndarray) -> np.ndarray:
        cols = x[..., list(self.lit_a)]
        return np.all(cols == np.asarray(self.sign_a, dtype=x.dtype), axis=-1).astype(np.int64)

    def r2(self, x: np.ndarray) -> np.ndarray:
        prod = np.prod(x[..., list(self.sub_b)], axis=-1) * self.sign_b
        return (prod > 0).astype(np.int64)

    def describe(self) -> str:
        lits = " & ".join(
            f"x{i}={'+' if s > 0 else '-'}" for i, s in zip(self.lit_a, self.sign_a, strict=True)
        )
        return f"r1 = [{lits}]; r2 = parity({','.join(f'x{i}' for i in self.sub_b)}){'' if self.sign_b > 0 else ' negated'}"


@dataclass(frozen=True, slots=True)
class CompositionConfig:
    n_features: int = 12
    n_train_symbols: int = 64
    n_holdout_symbols: int = 32
    shared_structure: bool = True
    n_literals: int = 2
    parity_size: int = 3
    eval_items_per_family: int = 60
    min_per_label: int = 8
    families: tuple[str, ...] = COMPOSITION_FAMILIES


def _draw_primitives(rng: np.random.Generator, cfg: CompositionConfig) -> Primitives:
    lit = sorted(int(i) for i in rng.choice(GROUP_A, size=cfg.n_literals, replace=False))
    sign = [int(s) for s in rng.choice([-1, 1], size=cfg.n_literals)]
    sub = sorted(int(i) for i in rng.choice(GROUP_B, size=cfg.parity_size, replace=False))
    sign_b = int(rng.choice([-1, 1]))
    return Primitives(tuple(lit), tuple(sign), tuple(sub), sign_b)


@dataclass(slots=True)
class CompositionWorld:
    seed: int
    config: CompositionConfig = field(default_factory=CompositionConfig)

    symbols: np.ndarray = field(init=False)
    names: list[str] = field(init=False)
    train_ids: np.ndarray = field(init=False)
    holdout_ids: np.ndarray = field(init=False)
    op: str = field(init=False)
    taught: Primitives = field(init=False)  # what primitive_a / primitive_b teach
    composed: Primitives = field(init=False)  # what composition uses (== taught iff shared_structure)
    labels: dict[str, np.ndarray] = field(init=False)
    margins: np.ndarray = field(init=False)
    _by_label: dict[tuple[str, str, int], np.ndarray] = field(init=False)
    _eval_cache: dict = field(init=False)
    practice_items: int = field(init=False, default=0)
    practice_symbols: set[int] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        cfg = self.config
        d = cfg.n_features
        rng = stream(self.seed, "tasks_v2", "composition", "rules")
        self.op = OPS[int(rng.integers(len(OPS)))]
        self.taught = _draw_primitives(rng, cfg)
        ctrl = _draw_primitives(stream(self.seed, "tasks_v2", "composition", "specialist"), cfg)
        self.composed = self.taught if cfg.shared_structure else ctrl
        n_total = cfg.n_train_symbols + cfg.n_holdout_symbols
        pool_rng = stream(self.seed, "tasks_v2", "composition", "symbols")
        for _attempt in range(500):
            codes = pool_rng.choice(2**d, size=n_total, replace=False)
            bits = ((codes[:, None] >> np.arange(d)[None, :]) & 1).astype(np.float32)
            symbols = bits * 2.0 - 1.0
            perm = pool_rng.permutation(n_total)
            train = np.sort(perm[: cfg.n_train_symbols])
            hold = np.sort(perm[cfg.n_train_symbols :])
            labels = self._label_tables(symbols)
            ok = all(
                min(int((labels[f][ids] == y).sum()) for y in (0, 1)) >= cfg.min_per_label
                for f in ("primitive_a", "primitive_b", "composition")
                for ids in (train, hold)
            )
            if ok:
                break
        else:  # pragma: no cover
            raise RuntimeError("could not draw balanced symbol pools")
        self.symbols = symbols
        self.train_ids, self.holdout_ids = train, hold
        self.labels = labels
        self.names = self._make_names(rng, n_total)
        self.margins = self._margins(symbols)
        self._by_label = {}
        for f in ("primitive_a", "primitive_b", "composition"):
            for split, ids in (("train", train), ("holdout", hold)):
                for y in (0, 1):
                    self._by_label[(f, split, y)] = ids[labels[f][ids] == y]
        self._eval_cache = {}
        self.practice_items = 0
        self.practice_symbols = set()

    # ------------------------------------------------------------------ rules
    def _label_tables(self, symbols: np.ndarray) -> dict[str, np.ndarray]:
        x = symbols.astype(np.float64)
        la, lb = self.taught.r1(x), self.taught.r2(x)
        comp = compose(self.op, self.composed.r1(x), self.composed.r2(x))
        return {"primitive_a": la, "primitive_b": lb, "composition": comp, "composition_pair": comp}

    def _margins(self, symbols: np.ndarray) -> np.ndarray:
        """n_features - number of single-feature flips that change the composed label."""
        x = symbols.astype(np.float64)
        base = compose(self.op, self.composed.r1(x), self.composed.r2(x))
        n_flip = np.zeros(len(x), dtype=np.int64)
        for j in range(self.config.n_features):
            xf = x.copy()
            xf[:, j] *= -1.0
            n_flip += (compose(self.op, self.composed.r1(xf), self.composed.r2(xf)) != base).astype(np.int64)
        return (self.config.n_features - n_flip).astype(np.float64)

    @staticmethod
    def _make_names(rng: np.random.Generator, n: int) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        while len(out) < n:
            name = "".join(rng.choice(_SYLLABLES, size=int(rng.integers(2, 4))))
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def score(self, family: str, symbol_ids: tuple[int, ...]) -> tuple[int, float]:
        """(correct_action, margin) under the hidden rule. Never exposed to policies."""
        if family == "composition_pair":
            y = self.labels["composition"][list(symbol_ids)]
            if int(y.sum()) != 1:
                raise ValueError("composition_pair items must hold exactly one positive symbol")
            return int(np.argmax(y)), float(min(self.margins[list(symbol_ids)]))
        if family not in COMPOSITION_SPECS:
            raise KeyError(family)
        i = symbol_ids[0]
        m = float(self.margins[i]) if family == "composition" else float(self.config.n_features)
        return int(self.labels[family][i]), m

    # ------------------------------------------------------------------ sampling (label-stratified)
    def pool(self, split: Split) -> np.ndarray:
        return self.train_ids if split == "train" else self.holdout_ids

    def _draw_ids(
        self, family: str, rng: np.random.Generator, split: Split, forced: int | None = None
    ) -> tuple[int, ...]:
        """Label-stratified draw. ``forced`` fixes the label (or, for pairs, the positive slot) so the
        fixed evaluation sets are exactly balanced; practice draws a seeded coin."""
        if family == "composition_pair":
            pos = self._by_label[("composition", split, 1)]
            neg = self._by_label[("composition", split, 0)]
            a, b = int(rng.choice(pos)), int(rng.choice(neg))
            first_pos = (rng.random() < 0.5) if forced is None else forced == 0
            return (a, b) if first_pos else (b, a)
        y = int(rng.integers(2)) if forced is None else int(forced)
        return (int(rng.choice(self._by_label[(family, split, y)])),)

    def _item(self, family: str, ids: tuple[int, ...]) -> TaskItem:
        correct, margin = self.score(family, ids)
        return TaskItem(family, ids, self.symbols[list(ids)].astype(np.float32), correct, 2, margin)

    def sample_item(
        self, family: str, rng: np.random.Generator, split: Split = "train", difficulty: float | None = None
    ) -> TaskItem:
        n_cand = 1 if difficulty is None else 8
        cands = [self._draw_ids(family, rng, split) for _ in range(n_cand)]
        if difficulty is not None:
            cands.sort(key=lambda ids: -self.score(family, ids)[1])
            idx = min(len(cands) - 1, int(round(min(max(float(difficulty), 0.0), 1.0) * (len(cands) - 1))))
            ids = cands[idx]
        else:
            ids = cands[0]
        if split == "train":
            self.practice_items += 1
            self.practice_symbols.update(int(i) for i in ids)
        return self._item(family, ids)

    def eval_set(self, family: str, split: Split = "holdout", n: int | None = None) -> list[TaskItem]:
        n = n or self.config.eval_items_per_family
        key = (family, split, n)
        if key not in self._eval_cache:
            rng = stream(self.seed, "tasks_v2", "composition", "eval", family, split)
            self._eval_cache[key] = [
                self._item(family, self._draw_ids(family, rng, split, forced=k % 2)) for k in range(n)
            ]
        return self._eval_cache[key]

    # ------------------------------------------------------------------ perception & feedback
    def visible_features(self, age_years: float, enable_gate: bool = True) -> int:
        return self.config.n_features

    def perceive(self, context: np.ndarray, age_years: float, enable_gate: bool = True) -> np.ndarray:
        return context.copy()

    def observe(
        self, item: TaskItem, agent_id: str, tick: int, age_years: float, enable_gate: bool = True
    ) -> Observation:
        return Observation(
            agent_id,
            tick,
            age_years,
            "task",
            item.family,
            item.context.copy(),
            item.n_actions,
            meta={"symbols": [self.names[i] for i in item.symbol_ids]},
        )

    def feedback(
        self,
        item: TaskItem,
        action: int,
        mode: FeedbackMode,
        rng: np.random.Generator,
        *,
        noise: float = 0.0,
        source: FeedbackSource = "environment",
    ) -> Feedback:
        if mode == "none":
            return Feedback(mode="none", source=source)
        told = item.correct_action
        if rng.random() < noise:
            told = 1 - told
        if mode == "bandit":
            return Feedback(mode="bandit", reward=1.0 if action == told else 0.0, source=source)
        rewards = np.zeros(item.n_actions)
        rewards[told] = 1.0
        return Feedback(
            mode="full",
            reward=float(rewards[action]),
            correct_action=told,
            rewards_all=rewards,
            source=source,
        )

    # ------------------------------------------------------------------ metadata & audits
    def families_meta(self) -> dict[str, dict[str, int]]:
        return {
            f: {**COMPOSITION_SPECS[f], "n_features": self.config.n_features} for f in self.config.families
        }

    def describe(self) -> dict[str, str | bool]:
        return {
            "op": self.op,
            "taught": self.taught.describe(),
            "composed": self.composed.describe(),
            "shared_structure": self.config.shared_structure,
        }

    def fingerprint(self) -> str:
        return stable_hash(
            {
                "seed": self.seed,
                "config": asdict(self.config),
                "symbols": self.symbols,
                "op": self.op,
                "taught": asdict(self.taught),
                "composed": asdict(self.composed),
            }
        )

    def class_shares(self, split: Split = "holdout") -> dict[str, float]:
        ids = self.pool(split)
        return {f: float(self.labels[f][ids].mean()) for f in ("primitive_a", "primitive_b", "composition")}

    def leakage_report(self) -> dict[str, int | bool]:
        train = {tuple(v) for v in self.symbols[self.train_ids].tolist()}
        hold = {tuple(v) for v in self.symbols[self.holdout_ids].tolist()}
        overlap = len(train & hold)
        return {
            "train_symbols": len(train),
            "holdout_symbols": len(hold),
            "overlap": overlap,
            "clean": overlap == 0,
        }

    def leakage_counts(self) -> dict[str, int]:
        holdout = {int(i) for i in self.holdout_ids}
        return {
            "practice_items": self.practice_items,
            "distinct_practice_symbols": len(self.practice_symbols),
            "holdout_symbols_in_practice": len(self.practice_symbols & holdout),
        }
