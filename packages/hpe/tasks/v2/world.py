"""Phase II task families (v2): non-linear, compositional, non-stationary, delayed, ordinal.

Same contracts and held-out discipline as ``hpe.tasks.SymbolWorld``: symbols are random ±1 feature
vectors with generated names, split into disjoint train/holdout pools; assessment uses fixed
holdout-only item sets; label noise corrupts feedback only. ``V2World`` is duck-typed to
``SymbolWorld`` so ``hpe.tasks.practice`` / ``assess`` / ``transfer_probe`` work unchanged.

Families (``n_slots = 1`` everywhere):

* ``conjunction``  2 actions — correct = 1 iff exactly one of two feature groups (features 0–4 and
                   5–9) has a positive majority: an XOR-type conjunction of group majorities. Exactly
                   balanced, NOT linearly separable in raw features, and independent of the latent
                   projection (so the shared/specialist control does not apply to it).
* ``context_switch`` 3 actions — features 0–1 are a context code (4 contexts); the class readout
                   depends on the context: the same content symbol maps to different classes in
                   different contexts (compositional rule).
* ``nonstationary`` 3 actions — a linear readout that is replaced every ``regime_length`` training
                   samples of this family; assessment uses the current regime's rule, so recovery
                   after a switch is measurable.
* ``delayed``      2 actions — a linear sign rule whose feedback the practice loop delivers ``k``
                   episodes late (``hpe.tasks.v2.delayed.practice_delayed``); the world itself is
                   stationary.
* ``ordinal_cue``  9 actions — two 5-level cues (thermometer-coded) map to a 9-level criterion through
                   a non-linear interaction; feedback jitters to a neighbouring level with probability
                   ``noise``. Five of the 25 cue configurations are held out for assessment.

Shared vs specialist structure: with ``shared_structure=True`` every x-based family reads out the same
latent projection ``A`` (k = 3), so a shared encoder can help; with ``False`` each family has its own
independent projection — the control in which specialists should do at least as well.
Scripted assumptions: there is no perceptual age gate in v2 (``perceive`` returns the context).
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

V2_FAMILIES: tuple[str, ...] = ("conjunction", "context_switch", "nonstationary", "delayed", "ordinal_cue")
V2_SPECS: dict[str, dict[str, int]] = {
    "conjunction": {"n_slots": 1, "n_actions": 2},
    "context_switch": {"n_slots": 1, "n_actions": 3},
    "nonstationary": {"n_slots": 1, "n_actions": 3},
    "delayed": {"n_slots": 1, "n_actions": 2},
    "ordinal_cue": {"n_slots": 1, "n_actions": 9},
}
N_CONTEXTS = 4
GROUP = 5  # feature-group size for the conjunction family
ORDINAL_LEVELS = 5
ORDINAL_HOLDOUT_CONFIGS = 5


@dataclass(frozen=True, slots=True)
class V2Config:
    n_features: int = 12
    latent_dim: int = 3
    n_train_symbols: int = 128
    n_holdout_symbols: int = 64
    shared_structure: bool = True
    regime_length: int = 150
    eval_items_per_family: int = 60
    families: tuple[str, ...] = V2_FAMILIES


@dataclass(slots=True)
class V2World:
    seed: int
    config: V2Config = field(default_factory=V2Config)
    symbols: np.ndarray = field(init=False)
    names: list[str] = field(init=False)
    train_ids: np.ndarray = field(init=False)
    holdout_ids: np.ndarray = field(init=False)
    A: dict[str, np.ndarray] = field(init=False)
    readouts: dict[str, np.ndarray] = field(init=False)
    ctx_readouts: np.ndarray = field(init=False)  # (N_CONTEXTS, 3, k)
    regimes: list[np.ndarray] = field(init=False)  # nonstationary readouts
    regime_bias: list[np.ndarray] = field(init=False)
    ctx_bias: np.ndarray = field(init=False)
    ordinal_params: np.ndarray = field(init=False)
    ordinal_train: list[tuple[int, int]] = field(init=False)
    ordinal_holdout: list[tuple[int, int]] = field(init=False)
    train_counter: dict[str, int] = field(init=False)
    _eval_cache: dict = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.config
        d, k = cfg.n_features, cfg.latent_dim
        rng = stream(self.seed, "tasks_v2", "rules")
        n_total = cfg.n_train_symbols + cfg.n_holdout_symbols
        codes = rng.choice(2**d, size=n_total, replace=False)
        bits = ((codes[:, None] >> np.arange(d)[None, :]) & 1).astype(np.float32)
        self.symbols = bits * 2.0 - 1.0
        self.names = self._make_names(rng, n_total)
        perm = rng.permutation(n_total)
        self.train_ids = np.sort(perm[: cfg.n_train_symbols])
        self.holdout_ids = np.sort(perm[cfg.n_train_symbols :])
        # content features for context_switch are dims 2.. (dims 0-1 carry the context code)
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        shared = (q[:, :k].T * np.sqrt(d / k)).astype(np.float64)
        self.A = {}
        for fam in cfg.families:
            if cfg.shared_structure:
                self.A[fam] = shared
            else:
                qf, _ = np.linalg.qr(stream(self.seed, "tasks_v2", "specialist", fam).normal(size=(d, d)))
                self.A[fam] = (qf[:, :k].T * np.sqrt(d / k)).astype(np.float64)
        self.readouts = {
            "conjunction": rng.normal(size=(2, k)),
            "delayed": rng.normal(size=(1, k)),
        }
        self.ctx_readouts = rng.normal(size=(N_CONTEXTS, 3, k))
        self.regimes = [rng.normal(size=(3, k)) for _ in range(6)]
        self.ordinal_params = rng.normal(size=4)
        configs = [(a, b) for a in range(ORDINAL_LEVELS) for b in range(ORDINAL_LEVELS)]
        order = rng.permutation(len(configs))
        self.ordinal_holdout = sorted(configs[i] for i in order[:ORDINAL_HOLDOUT_CONFIGS])
        self.ordinal_train = sorted(configs[i] for i in order[ORDINAL_HOLDOUT_CONFIGS:])
        self._balance(rng)
        self.train_counter = {f: 0 for f in cfg.families}
        self._eval_cache = {}

    def _balance_bias(self, fam: str, W: np.ndarray) -> np.ndarray:
        """Bias making argmax classes roughly uniform over all symbols (as Phase I's _balance_rules)."""
        z = self._z(fam, self.symbols)
        s = z @ W.T
        b = np.zeros(W.shape[0])
        for _ in range(300):
            counts = np.bincount(np.argmax(s + b, axis=1), minlength=W.shape[0]) / len(s)
            b -= 0.5 * (counts - 1.0 / W.shape[0])
        return b

    def _balance(self, rng: np.random.Generator) -> None:
        self.ctx_bias = np.stack(
            [self._balance_bias("context_switch", self.ctx_readouts[c]) for c in range(N_CONTEXTS)]
        )
        self.regime_bias = [self._balance_bias("nonstationary", W) for W in self.regimes]

    @staticmethod
    def _make_names(rng: np.random.Generator, n: int) -> list[str]:
        names: set[str] = set()
        out: list[str] = []
        while len(out) < n:
            name = "".join(rng.choice(_SYLLABLES, size=int(rng.integers(2, 4))))
            if name not in names:
                names.add(name)
                out.append(name)
        return out

    # ------------------------------------------------------------------ rules (hidden)
    def _z(self, fam: str, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float64) @ self.A[fam].T

    def regime(self, fam: str = "nonstationary") -> int:
        return (self.train_counter[fam] // self.config.regime_length) % len(self.regimes)

    def score(
        self, family: str, symbol_ids: tuple[int, ...], context: int = 0, regime: int | None = None
    ) -> tuple[int, float]:
        if family == "ordinal_cue":
            c1, c2 = symbol_ids
            return self._ordinal_label(c1, c2), 1.0
        x = self.symbols[list(symbol_ids)][0]
        z = self._z(family, x)
        if family == "conjunction":
            g1, g2 = float(x[:GROUP].sum()), float(x[GROUP : 2 * GROUP].sum())
            return int((g1 > 0) != (g2 > 0)), float(min(abs(g1), abs(g2)))
        if family == "delayed":
            v = float((self.readouts["delayed"] @ z)[0])
            return int(v > 0), abs(v)
        if family == "context_switch":
            s = self.ctx_readouts[context] @ z + self.ctx_bias[context]
        elif family == "nonstationary":
            r = self.regime() if regime is None else regime
            s = self.regimes[r] @ z + self.regime_bias[r]
        else:
            raise KeyError(family)
        top = np.sort(s)[::-1]
        return int(np.argmax(s)), float(top[0] - top[1])

    def _ordinal_label(self, c1: int, c2: int) -> int:
        a, b, c, e = self.ordinal_params
        u, v = (c1 - 2) / 2.0, (c2 - 2) / 2.0
        s = a * u + b * v + 1.5 * c * u * v + 0.5 * e * (u * u - v * v)
        return int(np.clip(round(4 + 4 * np.tanh(s)), 0, 8))

    # ------------------------------------------------------------------ items
    def _context_code(self, ctx: int) -> np.ndarray:
        return np.array([1.0 if ctx & 1 else -1.0, 1.0 if ctx & 2 else -1.0], dtype=np.float32)

    def _ordinal_context(self, c1: int, c2: int) -> np.ndarray:
        x = -np.ones(self.config.n_features, dtype=np.float32)
        x[:ORDINAL_LEVELS] = np.where(np.arange(ORDINAL_LEVELS) <= c1, 1.0, -1.0)
        x[ORDINAL_LEVELS : 2 * ORDINAL_LEVELS] = np.where(np.arange(ORDINAL_LEVELS) <= c2, 1.0, -1.0)
        x[10] = (c1 - 2) / 2.0
        x[11] = (c2 - 2) / 2.0
        return x[None, :]

    def _item(self, family: str, ids: tuple[int, ...], rng: np.random.Generator, split: Split) -> TaskItem:
        if family == "ordinal_cue":
            pool = self.ordinal_train if split == "train" else self.ordinal_holdout
            c1, c2 = pool[int(rng.integers(len(pool)))]
            y, m = self.score(family, (c1, c2))
            return TaskItem(family, (c1, c2), self._ordinal_context(c1, c2), y, 9, m)
        x = self.symbols[list(ids)].astype(np.float32).copy()
        if family == "context_switch":
            ctx = int(rng.integers(N_CONTEXTS))
            x[0, :2] = self._context_code(ctx)
            y, m = self.score(family, ids, context=ctx)
        else:
            y, m = self.score(family, ids)
        return TaskItem(family, ids, x, y, V2_SPECS[family]["n_actions"], m)

    def pool(self, split: Split) -> np.ndarray:
        return self.train_ids if split == "train" else self.holdout_ids

    def sample_item(
        self, family: str, rng: np.random.Generator, split: Split = "train", difficulty: float | None = None
    ) -> TaskItem:
        ids = (
            tuple(rng.choice(self.pool(split), size=1, replace=False).tolist())
            if family != "ordinal_cue"
            else ()
        )
        item = self._item(family, ids, rng, split)
        if split == "train":
            self.train_counter[family] += 1
        return item

    def eval_set(self, family: str, split: Split = "holdout", n: int | None = None) -> list[TaskItem]:
        n = n or self.config.eval_items_per_family
        key = (family, split, n, self.regime() if family == "nonstationary" else 0)
        if key not in self._eval_cache:
            rng = stream(self.seed, "tasks_v2", "eval", family, split, key[3])
            items = []
            for _ in range(n):
                ids = (
                    tuple(rng.choice(self.pool(split), size=1, replace=False).tolist())
                    if family != "ordinal_cue"
                    else ()
                )
                items.append(self._item(family, ids, rng, split))
            self._eval_cache[key] = items
        return self._eval_cache[key]

    # ------------------------------------------------------------------ perception & feedback
    def visible_features(self, age_years: float, enable_gate: bool = True) -> int:
        return self.config.n_features

    def perceive(self, context: np.ndarray, age_years: float, enable_gate: bool = True) -> np.ndarray:
        return context.copy()

    def observe(
        self, item: TaskItem, agent_id: str, tick: int, age_years: float, enable_gate: bool = True
    ) -> Observation:
        meta: dict = {
            "symbols": [self.names[i] for i in item.symbol_ids]
            if item.family != "ordinal_cue"
            else list(item.symbol_ids)
        }
        if item.family == "ordinal_cue":
            meta["ordinal"] = True
        return Observation(
            agent_id, tick, age_years, "task", item.family, item.context.copy(), item.n_actions, meta=meta
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
        flip = bool(rng.random() < noise)
        told = item.correct_action
        if flip:
            if item.family == "ordinal_cue":
                step = 1 if rng.random() < 0.5 else -1
                told = told + step if 0 <= told + step <= 8 else told - step
            else:
                told = int(rng.choice([a for a in range(item.n_actions) if a != told]))
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

    # ------------------------------------------------------------------ metadata
    def families_meta(self) -> dict[str, dict[str, int]]:
        out = {f: {**V2_SPECS[f], "n_features": self.config.n_features} for f in self.config.families}
        if "ordinal_cue" in out:
            out["ordinal_cue"]["ordinal"] = 1
        return out

    def fingerprint(self) -> str:
        return stable_hash(
            {
                "seed": self.seed,
                "config": asdict(self.config),
                "symbols": self.symbols,
                "A": {k: v for k, v in self.A.items()},
                "readouts": self.readouts,
                "ctx": self.ctx_readouts,
                "regimes": self.regimes,
                "ordinal": self.ordinal_params,
            }
        )

    def leakage_report(self) -> dict[str, int | bool]:
        train = {tuple(v) for v in self.symbols[self.train_ids].tolist()}
        hold = {tuple(v) for v in self.symbols[self.holdout_ids].tolist()}
        o_overlap = len(set(self.ordinal_train) & set(self.ordinal_holdout))
        overlap = len(train & hold)
        return {
            "train_symbols": len(train),
            "holdout_symbols": len(hold),
            "overlap": overlap,
            "ordinal_overlap": o_overlap,
            "clean": overlap == 0 and o_overlap == 0,
        }
