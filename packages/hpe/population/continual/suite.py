"""Gate E synthetic switching suite (docs/phase3/PHASE3_RESEARCH_THESIS.md §12, amendment A3-2).

Generative model (fixed by A3-2; the choices A3-2 leaves open are marked *impl*):

* 300 trials, K = 4 options, C = 3 latent contexts per sequence. Each context is a Bernoulli reward
  vector: one option at 0.8, the others drawn uniformly from [0.15, 0.45]. *impl*: the three contexts have
  **distinct** best options (a random 3-of-4 draw), so every switch changes the best option.
* Blocks of 20–40 trials. *impl*: lengths are drawn so that they tile the 300 trials exactly (each draw is
  uniform on [20, min(40, remaining − 20)]; the last block takes the remainder, which is in [20, 40]).
* Context order: *impl*: the first context is uniform; each later block switches to a context drawn
  uniformly from the other C − 1. An order is redrawn (deterministically) until every context appears and
  there are ≥ 2 returns (blocks whose context occurred in an earlier block).
* Context identity is never observed. Learners **act**: at every trial they choose an option and receive
  that option's Bernoulli outcome (bandit feedback). Outcomes are pre-drawn per (trial, option) from the
  seed, so every learner faces the same outcome table (common random numbers; paired comparisons).

Seeds: development 0–199. Confirmatory seeds 3000–3049 are reserved: ``generate`` refuses every seed outside
0–199 unless a confirmatory unlock names an experiment whose freeze record validates
(``hpe.population.core.freeze.validate_freeze``). The refusal happens before any random draw.

Metrics (A3-2) use the **expected** reward of the chosen option, ``p[c_t, a_t]`` (same expectation as the
realised reward, lower variance); realised rewards are recorded too. The oracle's expected reward is 0.8 on
every trial.

* post-switch adaptation: mean expected reward in the 10 trials after each switch divided by the oracle's
  (0.8), averaged over the switches of a sequence;
* savings on return: (first 10 trials of a return block) − (first 10 trials of that context's first block);
* forgetting: (first 10 trials of a return block) − (last 10 trials of that context's previous block);
  negative values mean forgetting.

Reference solutions: ``OracleAgent`` (knows the context and its reward vector) and ``BayesFilterAgent``
(Rao-Blackwellised particle filter over the latent context sequence with Beta posteriors per context and
option, semi-Markov switch hazard from the block-length distribution; Thompson or greedy action choice).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from hpe.core.rng import stream

T_TRIALS = 300
K_OPTIONS = 4
N_CONTEXTS = 3
BEST_P = 0.8
LOW_RANGE = (0.15, 0.45)
BLOCK_RANGE = (20, 40)
MIN_RETURNS = 2
WINDOW = 10
DEV_SEEDS = range(0, 200)
CONFIRMATORY_SEEDS = range(3000, 3050)
GENERATOR_VERSION = "gate_e_suite_v1"


class ReservedSeedError(RuntimeError):
    """Raised before any draw when a seed outside the development range is requested without a freeze."""


@dataclass(frozen=True)
class SwitchingSequence:
    seed: int
    probs: np.ndarray  # [C, K] reward probability of each option in each context
    contexts: np.ndarray  # [T] latent context of every trial
    blocks: tuple[tuple[int, int, int], ...]  # (start, end_exclusive, context)
    rewards: np.ndarray  # [T, K] pre-drawn Bernoulli outcomes

    @property
    def T(self) -> int:
        return int(self.contexts.shape[0])

    @property
    def K(self) -> int:
        return int(self.probs.shape[1])

    @property
    def best(self) -> np.ndarray:
        return self.probs.argmax(axis=1)

    def returns(self) -> list[int]:
        """Indices of blocks whose context occurred in an earlier block."""
        seen: set[int] = set()
        out = []
        for b, (_s, _e, c) in enumerate(self.blocks):
            if c in seen:
                out.append(b)
            seen.add(c)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "probs": np.round(self.probs, 4).tolist(),
            "blocks": [list(b) for b in self.blocks],
            "n_returns": len(self.returns()),
        }


def _check_seed(seed: int, unlock: str | None) -> None:
    if int(seed) in DEV_SEEDS:
        return
    if unlock is None:
        raise ReservedSeedError(
            f"seed {seed} is outside the development range 0-199; confirmatory seeds 3000-3049 are reserved "
            "until a freeze record validates (thesis §12)"
        )
    from hpe.population.core.freeze import validate_freeze

    validate_freeze(unlock)  # raises TestPartitionLocked unless the freeze is committed and approved
    if int(seed) not in CONFIRMATORY_SEEDS:
        raise ReservedSeedError(f"seed {seed} is neither a development nor a confirmatory seed")


def _block_lengths(rng: np.random.Generator, T: int, lo: int, hi: int) -> list[int]:
    out: list[int] = []
    remaining = T
    while remaining > hi:
        top = min(hi, remaining - lo)
        n = int(rng.integers(lo, top + 1))
        out.append(n)
        remaining -= n
    out.append(remaining)
    return out


def _context_order(rng: np.random.Generator, n_blocks: int, C: int) -> list[int]:
    for _ in range(1000):
        order = [int(rng.integers(C))]
        for _b in range(1, n_blocks):
            others = [c for c in range(C) if c != order[-1]]
            order.append(others[int(rng.integers(len(others)))])
        seen: set[int] = set()
        n_ret = 0
        for c in order:
            n_ret += c in seen
            seen.add(c)
        if len(seen) == C and n_ret >= MIN_RETURNS:
            return order
    raise RuntimeError("could not draw a valid context order")  # pragma: no cover


def generate(seed: int, *, confirmatory_unlock: str | None = None) -> SwitchingSequence:
    """One sequence of the Gate E suite. Seeds outside 0-199 are refused unless a freeze validates."""
    _check_seed(seed, confirmatory_unlock)
    rng = stream(int(seed), GENERATOR_VERSION)
    K, C = K_OPTIONS, N_CONTEXTS
    bests = rng.permutation(K)[:C]
    probs = rng.uniform(LOW_RANGE[0], LOW_RANGE[1], size=(C, K))
    probs[np.arange(C), bests] = BEST_P
    lengths = _block_lengths(rng, T_TRIALS, *BLOCK_RANGE)
    order = _context_order(rng, len(lengths), C)
    contexts = np.repeat(np.asarray(order, dtype=np.int64), lengths)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    blocks = tuple((int(s), int(s + n), int(c)) for s, n, c in zip(starts, lengths, order, strict=True))
    rewards = (rng.random((T_TRIALS, K)) < probs[contexts]).astype(np.float64)
    return SwitchingSequence(int(seed), probs, contexts, blocks, rewards)


def generate_many(seeds: Iterable[int]) -> list[SwitchingSequence]:
    return [generate(s) for s in seeds]


def permute_options(seq: SwitchingSequence, perm: np.ndarray) -> SwitchingSequence:
    """Relabel options (a symmetry of the task): new option j is old option ``perm[j]``.

    Used only for meta-training augmentation of development seeds; never creates new seeds."""
    perm = np.asarray(perm, dtype=np.int64)
    return SwitchingSequence(seq.seed, seq.probs[:, perm], seq.contexts, seq.blocks, seq.rewards[:, perm])


# ----------------------------------------------------------------------------------------- agents
class Agent(Protocol):
    """A learner that acts. ``reset`` starts a new sequence; ``act`` may use only past observations."""

    name: str

    def reset(self, rng: np.random.Generator) -> None: ...

    def act(self) -> int: ...

    def observe(self, action: int, reward: float) -> None: ...


@dataclass(slots=True)
class Trajectory:
    seed: int
    actions: np.ndarray  # [T]
    rewards: np.ndarray  # [T] realised
    expected: np.ndarray  # [T] p[c_t, a_t]
    info: dict[str, Any] = field(default_factory=dict)


def rollout(agent: Agent, seq: SwitchingSequence, rng: np.random.Generator) -> Trajectory:
    """Run ``agent`` through ``seq``. The agent sees only its own actions and their realised outcomes."""
    agent.reset(rng)
    T = seq.T
    acts = np.zeros(T, dtype=np.int64)
    rew = np.zeros(T)
    for t in range(T):
        a = int(agent.act())
        if not 0 <= a < seq.K:
            raise ValueError(f"{agent.name}: invalid action {a}")
        r = float(seq.rewards[t, a])
        agent.observe(a, r)
        acts[t], rew[t] = a, r
    return Trajectory(seq.seed, acts, rew, seq.probs[seq.contexts, acts])


class OracleAgent:
    """Upper bound: knows the latent context and its reward vector (privileged; not a learner)."""

    name = "oracle"

    def __init__(self, seq: SwitchingSequence) -> None:
        self.seq = seq
        self.t = 0

    def reset(self, rng: np.random.Generator) -> None:
        self.t = 0

    def act(self) -> int:
        return int(self.seq.best[self.seq.contexts[self.t]])

    def observe(self, action: int, reward: float) -> None:
        self.t += 1


class RandomAgent:
    name = "random"

    def __init__(self, K: int = K_OPTIONS) -> None:
        self.K = K
        self.rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self) -> int:
        assert self.rng is not None
        return int(self.rng.integers(self.K))

    def observe(self, action: int, reward: float) -> None:
        return None


class StationaryThompsonAgent:
    """Naive Bayesian learner that assumes one fixed context (Beta(1,1) per option, no switches)."""

    name = "stationary_ts"

    def __init__(self, K: int = K_OPTIONS) -> None:
        self.K = K
        self.a = np.ones(K)
        self.b = np.ones(K)
        self.rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.a = np.ones(self.K)
        self.b = np.ones(self.K)

    def act(self) -> int:
        assert self.rng is not None
        return int(np.argmax(self.rng.beta(self.a, self.b)))

    def observe(self, action: int, reward: float) -> None:
        self.a[action] += reward
        self.b[action] += 1.0 - reward


class DeltaRuleAgent:
    """Naive model-free learner: Rescorla-Wagner values with a softmax (no memory of contexts)."""

    name = "delta_rule"

    def __init__(self, alpha: float = 0.3, beta: float = 10.0, K: int = K_OPTIONS) -> None:
        self.alpha, self.beta, self.K = float(alpha), float(beta), K
        self.q = np.full(K, 0.5)
        self.rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.q = np.full(self.K, 0.5)

    def act(self) -> int:
        assert self.rng is not None
        z = self.beta * (self.q - self.q.max())
        p = np.exp(z) / np.exp(z).sum()
        return int(self.rng.choice(self.K, p=p))

    def observe(self, action: int, reward: float) -> None:
        self.q[action] += self.alpha * (reward - self.q[action])


def block_hazard(r: np.ndarray, lo: int = BLOCK_RANGE[0], hi: int = BLOCK_RANGE[1]) -> np.ndarray:
    """P(block ends after ``r`` trials | it lasted ``r`` trials) for lengths uniform on {lo..hi}."""
    r = np.asarray(r)
    h = np.where(r < lo, 0.0, 1.0 / np.maximum(hi + 1 - r, 1))
    return np.clip(h, 0.0, 1.0)


class BayesFilterAgent:
    """Reference learner: Bayes filter over latent contexts with Beta posteriors (Rao-Blackwellised).

    Each particle holds one hypothesis about the context sequence so far — labels in order of first
    appearance, the current label, its run length and the number of contexts seen — together with the exact
    Beta posterior counts of every (label, option) under that hypothesis (prior ``Beta(1, 1)``). The switch
    prior is the generator's: semi-Markov hazard of block lengths uniform on 20–40 and, on a switch, a
    uniform draw among the other C − 1 contexts (so a new label has probability (C − n_seen)/(C − 1)).
    The filter is *fully adapted*: for every particle all C transitions are enumerated, weighted by their
    prior probability times the Beta predictive probability of the observed outcome; the particle weight is
    multiplied by the marginal and the transition is sampled from its posterior; systematic resampling when
    ESS < N/2. ``floor`` (default 0) adds a small switch probability where the generator forbids one.

    ``policy="greedy"``: argmax of the posterior predictive reward probability (ties broken at random).
    ``policy="thompson"``: sample a particle, its transition and reward probabilities, take the argmax.
    Action choice is myopic, so this is the Bayes-optimal *inference* with a heuristic policy, a strong
    reference rather than a proven optimum of the reward-maximisation problem.
    """

    def __init__(
        self,
        n_particles: int = 256,
        policy: str = "greedy",
        floor: float = 0.0,
        prior: tuple[float, float] = (1.0, 1.0),
        K: int = K_OPTIONS,
        C: int = N_CONTEXTS,
    ) -> None:
        if policy not in ("thompson", "greedy"):
            raise ValueError(policy)
        self.N, self.policy, self.floor, self.prior = int(n_particles), policy, float(floor), prior
        self.K, self.C = K, C
        self.name = f"bayes_filter_{policy}"
        self.rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        N, C, K = self.N, self.C, self.K
        self.rng = rng
        self.label = np.zeros(N, dtype=np.int64)
        self.run = np.zeros(N, dtype=np.int64)  # trials already spent in the current block
        self.n_seen = np.ones(N, dtype=np.int64)
        self.alpha = np.full((N, C, K), self.prior[0])
        self.beta = np.full((N, C, K), self.prior[1])
        self.logw = np.zeros(N)
        self.t = 0

    def transition_prior(self) -> np.ndarray:
        """[N, C] prior probability that the coming trial belongs to each label, per particle."""
        N, C = self.N, self.C
        q = np.zeros((N, C))
        if self.t == 0:
            q[:, 0] = 1.0
            return q
        h = np.maximum(block_hazard(self.run), self.floor)
        lab = np.arange(C)[None, :]
        seen_other = (lab < self.n_seen[:, None]) & (lab != self.label[:, None])
        new = lab == self.n_seen[:, None]  # empty when every context has been seen
        q = np.where(seen_other, h[:, None] / (C - 1), 0.0)
        q = q + np.where(new, (h * (C - self.n_seen) / (C - 1))[:, None], 0.0)
        q[np.arange(N), self.label] = 1.0 - h
        return q

    def _weights(self) -> np.ndarray:
        w = np.exp(self.logw - self.logw.max())
        return np.asarray(w / w.sum())

    def predictive(self) -> np.ndarray:
        """Posterior predictive reward probability of each option for the coming trial."""
        q = self.transition_prior()
        mean = self.alpha / (self.alpha + self.beta)  # [N, C, K]
        per_particle = np.einsum("nc,nck->nk", q, mean)
        return np.asarray(self._weights() @ per_particle)

    def act(self) -> int:
        assert self.rng is not None
        if self.policy == "greedy":
            m = self.predictive()
            best = np.flatnonzero(m >= m.max() - 1e-12)
            return int(best[int(self.rng.integers(best.size))])
        i = int(self.rng.choice(self.N, p=self._weights()))
        q = self.transition_prior()[i]
        j = int(self.rng.choice(self.C, p=q / q.sum()))
        return int(np.argmax(self.rng.beta(self.alpha[i, j], self.beta[i, j])))

    def observe(self, action: int, reward: float) -> None:
        assert self.rng is not None
        N = self.N
        idx = np.arange(N)
        q = self.transition_prior()
        p1 = self.alpha[:, :, action] / (self.alpha[:, :, action] + self.beta[:, :, action])  # [N, C]
        lik = p1 if reward > 0.5 else 1.0 - p1
        joint = q * lik
        marg = joint.sum(axis=1)
        self.logw = self.logw + np.log(np.maximum(marg, 1e-300))
        post = joint / np.maximum(marg, 1e-300)[:, None]
        u = self.rng.random(N)
        j = np.minimum((np.cumsum(post, axis=1) < u[:, None]).sum(axis=1), self.C - 1)
        switched = j != self.label
        self.n_seen = np.where(j == self.n_seen, self.n_seen + 1, self.n_seen)
        self.run = np.where(switched, 1, self.run + 1)
        self.label = j
        if reward > 0.5:
            self.alpha[idx, j, action] += 1.0
        else:
            self.beta[idx, j, action] += 1.0
        self.t += 1
        w = self._weights()
        if 1.0 / np.sum(w**2) < N / 2:
            pos = (self.rng.random() + np.arange(N)) / N
            keep = np.minimum(np.searchsorted(np.cumsum(w), pos), N - 1)
            self.label, self.run, self.n_seen = self.label[keep], self.run[keep], self.n_seen[keep]
            self.alpha, self.beta = self.alpha[keep].copy(), self.beta[keep].copy()
            self.logw = np.zeros(N)

    def context_posterior(self) -> np.ndarray:
        """Posterior over the label of the most recent trial."""
        return np.bincount(self.label, weights=self._weights(), minlength=self.C)


# ----------------------------------------------------------------------------------------- metrics
def _mean(x: Sequence[float]) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def sequence_metrics(seq: SwitchingSequence, traj: Trajectory, window: int = WINDOW) -> dict[str, float]:
    """Gate E metrics of one trajectory (A3-2). Expected-reward based; ``*_realised`` use outcomes."""
    out: dict[str, float] = {}
    for tag, r in (("", traj.expected), ("_realised", traj.rewards)):
        oracle = (
            np.full(seq.T, BEST_P) if tag == "" else seq.rewards[np.arange(seq.T), seq.best[seq.contexts]]
        )
        adapt, sav, forg = [], [], []
        first_block: dict[int, int] = {}
        last_block: dict[int, int] = {}
        for b, (s, _e, c) in enumerate(seq.blocks):
            if b > 0:
                o = float(oracle[s : s + window].mean())
                adapt.append(float(r[s : s + window].mean()) / o if o > 0 else float("nan"))
            if c in first_block:
                fs = seq.blocks[first_block[c]][0]
                ps, pe = seq.blocks[last_block[c]][:2]
                sav.append(float(r[s : s + window].mean() - r[fs : fs + window].mean()))
                forg.append(float(r[s : s + window].mean() - r[max(ps, pe - window) : pe].mean()))
            else:
                first_block[c] = b
            last_block[c] = b
        out[f"adaptation{tag}"] = _mean([a for a in adapt if np.isfinite(a)])
        out[f"savings{tag}"] = _mean(sav)
        out[f"forgetting{tag}"] = _mean(forg)
        out[f"reward{tag}"] = float(r.mean())
    out["reward_fraction"] = out["reward"] / BEST_P
    return out


METRICS = ("adaptation", "savings", "forgetting", "reward_fraction")


def average_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = sorted(rows[0])
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def evaluate_agent(
    make_agent: Any, seqs: Sequence[SwitchingSequence], *, seed: int, name: str, repeats: int = 1
) -> dict[int, dict[str, float]]:
    """Per-seed metrics averaged over ``repeats`` rollouts (policy randomness from named streams).

    ``make_agent(seq)`` returns a fresh agent; it receives ``seq`` only so that the oracle can be built."""
    out: dict[int, dict[str, float]] = {}
    for seq in seqs:
        rows = []
        for rep in range(repeats):
            agent = make_agent(seq)
            traj = rollout(agent, seq, stream(seed, "gate_e_policy", name, seq.seed, rep))
            rows.append(sequence_metrics(seq, traj))
        out[seq.seed] = average_metrics(rows)
    return out
