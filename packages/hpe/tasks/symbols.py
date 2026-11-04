"""Novel-symbol task world: the learnable environment shared by every policy class.

Design goals (docs/ADRS/0003-task-world.md):

* **No pretrained-knowledge leakage.** Symbols are random binary feature vectors with
  generated nonsense names; the rules mapping symbols to correct actions are drawn from the
  task seed. No language model, and no human, knows them in advance. Skill must be acquired.
* **Held-out generalisation.** Symbols are split into a *train pool* (the only symbols that
  ever appear in practice) and a *holdout pool* (only ever used for assessment). Assessment
  therefore measures generalisation to unseen symbols, not recall.
* **Shared latent structure → transfer is possible but not guaranteed.** Every family's
  rule is a readout of the same latent projection ``z = A x``. A learner that discovers the
  shared representation can learn a never-practised family (``barter``) from fewer examples.
  In ``transfer_mode="orthogonal"`` the transfer family uses a latent space orthogonal to
  ``A``; there a shared representation should give no benefit (negative-transfer control).
* **Objectively scored.** Correct answers are computed from the hidden rules; feedback can
  be corrupted by label noise, but assessment always scores against the true rule.

Perception: symbols are presented as ±1 feature vectors. The (optional) age gate zeroes
features an agent cannot yet perceive, so 0 means "not perceived". The gate is a scripted
modelling assumption applied identically to every policy class; ablate it with
``enable_age_gate=False``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from hpe.contracts.runtime import Feedback, FeedbackMode, FeedbackSource, Observation
from hpe.core.hashing import stable_hash
from hpe.core.rng import stream

Split = Literal["train", "holdout"]
TransferMode = Literal["shared", "orthogonal"]

TRAINING_FAMILIES: tuple[str, ...] = ("sort", "compare", "signal")
TRANSFER_FAMILIES: tuple[str, ...] = ("barter",)
ALL_FAMILIES: tuple[str, ...] = TRAINING_FAMILIES + TRANSFER_FAMILIES

FAMILY_SPECS: dict[str, dict[str, int]] = {
    # n_slots = symbols shown at once; n_actions = choices
    "sort": {"n_slots": 1, "n_actions": 3},  # which of 3 categories does this symbol belong to?
    "compare": {"n_slots": 2, "n_actions": 2},  # which of two symbols is more valuable?
    "signal": {"n_slots": 1, "n_actions": 2},  # approach (1) or avoid (0) this symbol?
    "barter": {"n_slots": 2, "n_actions": 2},  # transfer family: which symbol trades better?
}

FAMILY_DESCRIPTIONS: dict[str, str] = {
    "sort": "Categorise a novel symbol into one of three hidden classes.",
    "compare": "Choose the more valuable of two novel symbols under a hidden value rule.",
    "signal": "Decide whether to approach or avoid a novel symbol under a hidden safety rule.",
    "barter": "Held-out transfer family (never practised during development): choose the better trade.",
}

_SYLLABLES = (
    "ka",
    "vo",
    "ri",
    "tem",
    "ul",
    "zan",
    "osh",
    "py",
    "qe",
    "lor",
    "mi",
    "dax",
    "fen",
    "iru",
    "gal",
    "sho",
    "wex",
    "bel",
    "ny",
    "tor",
    "ak",
    "ves",
    "ju",
    "pim",
)


@dataclass(frozen=True, slots=True)
class TaskWorldConfig:
    n_features: int = 12
    latent_dim: int = 3
    n_train_symbols: int = 64
    n_holdout_symbols: int = 32
    transfer_mode: TransferMode = "shared"
    eval_items_per_family: int = 60
    gate_min_features: int = 4
    gate_full_age: float = 10.0


@dataclass(slots=True)
class TaskItem:
    family: str
    symbol_ids: tuple[int, ...]
    context: np.ndarray  # (n_slots, n_features) ±1 float32, ungated
    correct_action: int
    n_actions: int
    margin: float  # distance to the decision boundary in latent units (difficulty proxy)


@dataclass(slots=True)
class SymbolWorld:
    seed: int
    config: TaskWorldConfig = field(default_factory=TaskWorldConfig)

    # populated in __post_init__
    symbols: np.ndarray = field(init=False)  # (N, d) in {-1, +1}
    names: list[str] = field(init=False)
    train_ids: np.ndarray = field(init=False)
    holdout_ids: np.ndarray = field(init=False)
    A: np.ndarray = field(init=False)  # (k, d) shared latent projection
    B: np.ndarray = field(init=False)  # (k, d) orthogonal projection for negative-transfer control
    readouts: dict[str, np.ndarray] = field(init=False)
    biases: dict[str, np.ndarray] = field(init=False)
    _eval_cache: dict[tuple[str, Split, int], list[TaskItem]] = field(init=False)
    _ctx32: np.ndarray = field(init=False)
    _sort_scores: np.ndarray = field(init=False)
    _values: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.config
        d, k = cfg.n_features, cfg.latent_dim
        rng = stream(self.seed, "tasks", "rules")
        n_total = cfg.n_train_symbols + cfg.n_holdout_symbols
        if n_total > 2**d:
            raise ValueError("more symbols requested than distinct feature vectors")
        codes = rng.choice(2**d, size=n_total, replace=False)
        bits = ((codes[:, None] >> np.arange(d)[None, :]) & 1).astype(np.float32)
        self.symbols = bits * 2.0 - 1.0
        self.names = self._make_names(rng, n_total)
        perm = rng.permutation(n_total)
        self.train_ids = np.sort(perm[: cfg.n_train_symbols])
        self.holdout_ids = np.sort(perm[cfg.n_train_symbols :])

        # Shared latent space A, and an orthogonal complement B for the control world.
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        self.A = (q[:, :k].T * np.sqrt(d / k)).astype(np.float64)
        self.B = (q[:, k : 2 * k].T * np.sqrt(d / k)).astype(np.float64)

        self.readouts = {
            "sort": rng.normal(size=(3, k)),
            "compare": rng.normal(size=(1, k)),
            "signal": rng.normal(size=(1, k)),
            "barter": rng.normal(size=(1, k)),
        }
        self.biases = {f: np.zeros(FAMILY_SPECS[f]["n_actions"] if f == "sort" else 1) for f in ALL_FAMILIES}
        self._balance_rules()
        self._eval_cache = {}
        # Hot-path caches (pure functions of the rules above; identical results, ~3x faster sampling).
        self._ctx32 = self.symbols.astype(np.float32)
        self._sort_scores = np.stack([self._sort_score_row(i) for i in range(n_total)])
        self._values = {
            f: np.array([float(self._value(f, self.symbols[[i]])[0]) for i in range(n_total)])
            for f in ("signal", "compare", "barter")
        }

    # ------------------------------------------------------------------ construction
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

    def _latent(self, family: str, x: np.ndarray) -> np.ndarray:
        proj = (
            self.B if (family in TRANSFER_FAMILIES and self.config.transfer_mode == "orthogonal") else self.A
        )
        return x @ proj.T

    def _balance_rules(self) -> None:
        """Choose biases so classes are roughly balanced over all symbols (no trivial majority answer)."""
        allx = self.symbols.astype(np.float64)
        z = self._latent("sort", allx)
        s = z @ self.readouts["sort"].T
        b = np.zeros(3)
        for _ in range(200):
            counts = np.bincount(np.argmax(s + b, axis=1), minlength=3) / len(s)
            b -= 0.5 * (counts - 1 / 3)
        self.biases["sort"] = b
        zs = self._latent("signal", allx) @ self.readouts["signal"].T
        self.biases["signal"] = np.array([-float(np.median(zs))])

    # ------------------------------------------------------------------ scoring rules
    def _value(self, family: str, x: np.ndarray) -> np.ndarray:
        z = self._latent(family, x.astype(np.float64))
        out: np.ndarray = (z @ self.readouts[family].T)[..., 0] + self.biases[family][0]
        return out

    def _sort_score_row(self, i: int) -> np.ndarray:
        z = self._latent("sort", self.symbols[[i]].astype(np.float64))
        row: np.ndarray = (z @ self.readouts["sort"].T + self.biases["sort"])[0]
        return row

    def score(self, family: str, symbol_ids: tuple[int, ...]) -> tuple[int, float]:
        """Return (correct_action, margin) under the hidden rule. Never exposed to policies."""
        if family == "sort":
            s = self._sort_scores[symbol_ids[0]]
            top = np.sort(s)[::-1]
            return int(np.argmax(s)), float(top[0] - top[1])
        if family == "signal":
            sv = float(self._values["signal"][symbol_ids[0]])
            return int(sv > 0), abs(sv)
        if family in ("compare", "barter"):
            vals = self._values[family]
            a, b = float(vals[symbol_ids[0]]), float(vals[symbol_ids[1]])
            return int(b > a), abs(b - a)
        raise KeyError(family)

    # ------------------------------------------------------------------ sampling
    def pool(self, split: Split) -> np.ndarray:
        return self.train_ids if split == "train" else self.holdout_ids

    def sample_item(
        self, family: str, rng: np.random.Generator, split: Split = "train", difficulty: float | None = None
    ) -> TaskItem:
        """Sample one item. ``difficulty`` in [0,1] selects items nearer the boundary (curriculum)."""
        n_slots = FAMILY_SPECS[family]["n_slots"]
        pool = self.pool(split)
        n_cand = 1 if difficulty is None else 12
        cands = []
        for _ in range(n_cand):
            ids = tuple(rng.choice(pool, size=n_slots, replace=False).tolist())
            correct, margin = self.score(family, ids)
            cands.append((ids, correct, margin))
        if difficulty is not None:
            cands.sort(key=lambda c: -c[2])  # easiest (largest margin) first
            idx = min(len(cands) - 1, int(round(min(max(float(difficulty), 0.0), 1.0) * (len(cands) - 1))))
            ids, correct, margin = cands[idx]
        else:
            ids, correct, margin = cands[0]
        return TaskItem(
            family=family,
            symbol_ids=ids,
            context=self._ctx32[list(ids)],
            correct_action=correct,
            n_actions=FAMILY_SPECS[family]["n_actions"],
            margin=margin,
        )

    def eval_set(self, family: str, split: Split = "holdout", n: int | None = None) -> list[TaskItem]:
        """Fixed, seed-determined assessment items. Holdout items use only holdout symbols."""
        n = n or self.config.eval_items_per_family
        key = (family, split, n)
        if key not in self._eval_cache:
            rng = stream(self.seed, "tasks", "eval", family, split)
            self._eval_cache[key] = [self.sample_item(family, rng, split=split) for _ in range(n)]
        return self._eval_cache[key]

    # ------------------------------------------------------------------ perception & feedback
    def visible_features(self, age_years: float, enable_gate: bool = True) -> int:
        d = self.config.n_features
        if not enable_gate:
            return d
        frac = min(max(float(age_years) / self.config.gate_full_age, 0.0), 1.0)  # == np.clip, ~10x cheaper
        return int(round(self.config.gate_min_features + (d - self.config.gate_min_features) * frac))

    def perceive(self, context: np.ndarray, age_years: float, enable_gate: bool = True) -> np.ndarray:
        n_vis = self.visible_features(age_years, enable_gate)
        if n_vis >= context.shape[-1]:
            return context.copy()
        out = context.copy()
        out[..., n_vis:] = 0.0
        return out

    def observe(
        self, item: TaskItem, agent_id: str, tick: int, age_years: float, enable_gate: bool = True
    ) -> Observation:
        return Observation(
            agent_id=agent_id,
            tick=tick,
            age_years=age_years,
            kind="task",
            family=item.family,
            context=self.perceive(item.context, age_years, enable_gate),
            n_actions=item.n_actions,
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
        """Reveal outcome information. Label noise corrupts what the learner is told, never the truth."""
        if mode == "none":
            return Feedback(mode="none", source=source)
        flip = bool(rng.random() < noise)
        correct = item.correct_action
        if mode == "bandit":
            r = 1.0 if action == correct else 0.0
            if flip:
                r = 1.0 - r
            return Feedback(mode="bandit", reward=r, source=source)
        told = correct
        if flip:
            others = [a for a in range(item.n_actions) if a != correct]
            told = int(rng.choice(others))
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
        return {f: {**FAMILY_SPECS[f], "n_features": self.config.n_features} for f in ALL_FAMILIES}

    def fingerprint(self) -> str:
        return stable_hash(
            {
                "seed": self.seed,
                "config": asdict(self.config),
                "symbols": self.symbols,
                "A": self.A,
                "readouts": self.readouts,
            }
        )

    def leakage_report(self) -> dict[str, int | bool]:
        """Check that no holdout symbol is identical to a train symbol."""
        train = {tuple(v) for v in self.symbols[self.train_ids].tolist()}
        hold = {tuple(v) for v in self.symbols[self.holdout_ids].tolist()}
        overlap = len(train & hold)
        return {
            "train_symbols": len(train),
            "holdout_symbols": len(hold),
            "overlap": overlap,
            "clean": overlap == 0,
        }
