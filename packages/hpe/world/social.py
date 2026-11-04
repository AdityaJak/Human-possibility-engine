"""Social mechanism v2 (Phase II, Candidate E) — flag-guarded, instrumented.

Enabled only when ``SimulationConfig.policy_params["social_v2"]`` is present (a dict of
:class:`SocialV2Config` fields or ``True``). With the flag absent, ``hpe.world.engine`` never calls
anything in this module and Phase I behaviour is bit-identical (tested).

Why v2 exists (Phase I, Exp 3): developmental agents formed 0.018 friends in 15 years and peer
demonstrations contributed nothing. Two faults: plaza partners were drawn afresh every tick, so ties
never accumulated the repeated contact that friendship requires; and demonstrations were random items
with outcome-only feedback, so they carried almost no information a learner could not get alone.

What v2 changes — mechanisms that make social information *useful*, never a reward for the desired
experimental outcome:

1. **Persistent relationships.** Agents at the plaza prefer partners they already know (probability
   ``p_known``), weighted by tie strength (interaction count). Ties decay without contact; friendships
   revert to acquaintance when strength falls below ``friend_min_strength``.
2. **Reliability-tracked trust.** After every demonstration whose correctness is revealed, the
   learner's trust in that demonstrator moves toward the observed reliability. Misinformers therefore
   lose influence through evidence, not by rule.
3. **Targeted teaching.** A demonstrator teaches the family in which the *learner's* own self-estimate
   is weakest. If the teacher's last measured accuracy in that family is at least
   ``competence_threshold`` the demonstration carries instructional feedback (the teacher's answer as
   a label — right only if the teacher is right, plus label noise); otherwise outcome-only feedback.
4. **Cooperation.** One shared item per meeting: both act, the environment reveals the correct answer
   to both (information), and the joint payoff feeds the partners' mutual trust.
5. **Control** ``information=False`` ("social reward only"): all social dynamics above run, but no
   demonstration or cooperation item is delivered — the learner receives no information.
6. **Learning-progress signal.** A socialising agent's ``learn`` signal (the same term self-study
   reports to the activity bandit) is the mean information gain of the demonstrations it received
   this tick. ``learn_signal=False`` switches this off (dev variant).

Instrumentation (counters emitted yearly as a ``SOCIAL_INTERACTION`` event with mechanism
``world.social.v2.stats``): meetings, repeat meetings, ties formed/decayed, friendships, demonstrations
by demonstrator type and correctness, information gain, cooperation items, mean trust toward honest
vs misinforming demonstrators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from hpe.contracts.runtime import Demonstration, Feedback, Observation
from hpe.contracts.schemas import EventType
from hpe.tasks import TRAINING_FAMILIES

if TYPE_CHECKING:  # pragma: no cover - typing only; the world package never imports learners
    from hpe.world.engine import World
    from hpe.world.records import AgentRec

STATS_KEYS: tuple[str, ...] = (
    "meetings",
    "repeat_meetings",
    "ties_formed",
    "ties_decayed",
    "friendships",
    "friendships_lost",
    "demos",
    "demos_full",
    "demos_correct",
    "demos_from_misinformers",
    "demos_from_misinformers_correct",
    "info_gain_sum",
    "info_gain_n",
    "cooperation_items",
    "cooperation_joint_correct",
)


@dataclass(frozen=True, slots=True)
class SocialV2Config:
    p_known: float = 0.7  # probability of preferring a known partner who is present
    tie_decay_per_year: float = 0.7  # strength multiplier per year without contact
    friend_min_strength: int = 2  # friendship reverts below this strength
    trust_reliability_rate: float = 0.2  # trust -> observed demonstration reliability
    competence_threshold: float = 0.7  # teacher's measured accuracy needed for instructional feedback
    information: bool = True  # False = social-reward-only control (no demonstrations, no cooperation)
    learn_signal: bool = True  # socialising reports mean information gain as its learning-progress term
    cooperation: bool = True
    demos_per_meeting: int | None = None  # None -> WorldParams.peer_demonstrations_per_social_tick

    @classmethod
    def from_params(cls, policy_params: dict[str, Any] | None) -> SocialV2Config | None:
        if not policy_params:
            return None
        raw = policy_params.get("social_v2")
        if raw is None or raw is False:
            return None
        if raw is True:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("social_v2 must be a dict of SocialV2Config fields or true")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown social_v2 params: {unknown}; known: {sorted(known)}")
        return cls(**raw)

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(slots=True)
class SocialV2State:
    """Mutable per-run state; JSON-serialisable for snapshots."""

    last_contact: dict[str, int] = field(default_factory=dict)  # "a|b" -> tick of last meeting
    stats: dict[str, float] = field(default_factory=lambda: dict.fromkeys(STATS_KEYS, 0.0))
    year_stats: dict[str, float] = field(default_factory=lambda: dict.fromkeys(STATS_KEYS, 0.0))
    trust_honest: list[float] = field(default_factory=list)  # yearly samples
    trust_misinformer: list[float] = field(default_factory=list)
    last_decay_year: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_contact": dict(self.last_contact),
            "stats": dict(self.stats),
            "year_stats": dict(self.year_stats),
            "trust_honest": list(self.trust_honest),
            "trust_misinformer": list(self.trust_misinformer),
            "last_decay_year": self.last_decay_year,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SocialV2State:
        s = cls()
        s.last_contact = {str(k): int(v) for k, v in d.get("last_contact", {}).items()}
        s.stats = dict.fromkeys(STATS_KEYS, 0.0) | {k: float(v) for k, v in d.get("stats", {}).items()}
        s.year_stats = dict.fromkeys(STATS_KEYS, 0.0) | {
            k: float(v) for k, v in d.get("year_stats", {}).items()
        }
        s.trust_honest = [float(x) for x in d.get("trust_honest", [])]
        s.trust_misinformer = [float(x) for x in d.get("trust_misinformer", [])]
        s.last_decay_year = int(d.get("last_decay_year", 0))
        return s


def _key(a: str, b: str) -> str:
    return f"{a}|{b}" if a < b else f"{b}|{a}"


def _bump(st: SocialV2State, key: str, v: float = 1.0) -> None:
    st.stats[key] += v
    st.year_stats[key] += v


# ------------------------------------------------------------------------------------- matching


def plaza_pairs(w: World, pool: list[str], srng: np.random.Generator) -> list[tuple[str, str]]:
    """Pair socialising agents, preferring known partners weighted by tie strength."""
    cfg = w._sv2
    assert cfg is not None
    order = [pool[i] for i in srng.permutation(len(pool))] if pool else []
    present = set(pool)
    matched: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for aid in order:
        if aid in matched:
            continue
        partner: str | None = None
        known = [
            (b, float(e.interactions + 1))
            for b, e in sorted(w._edges.get(aid, {}).items())
            if b in present
            and b not in matched
            and b != aid
            and e.kind in ("friend", "acquaintance", "classmate", "coworker")
        ]
        if known and srng.random() < cfg.p_known:
            weights = np.array([k[1] for k in known], dtype=np.float64)
            partner = known[int(srng.choice(len(known), p=weights / weights.sum()))][0]
        else:
            for b in order:
                if b != aid and b not in matched:
                    partner = b
                    break
        if partner is None:
            continue
        matched.update((aid, partner))
        pairs.append((aid, partner) if aid < partner else (partner, aid))
    return pairs


# ------------------------------------------------------------------------------------- teaching


def _weakest_family(w: World, learner_id: str) -> str:
    """The training family where the learner's own self-estimate is lowest (uncertainty-targeted)."""
    est = w._policies[learner_id].self_estimates()
    best, best_v = TRAINING_FAMILIES[0], math.inf
    for fam in TRAINING_FAMILIES:
        v = est.get(fam, {})
        mean = float(v.get("mean", 0.5)) if isinstance(v, dict) else 0.5
        if mean < best_v:
            best, best_v = fam, mean
    return best


def _teacher_competent(w: World, teacher: AgentRec, family: str) -> bool:
    cfg = w._sv2
    assert cfg is not None
    m = teacher.measured.get(family)
    if not m:
        return False
    return float(m["accuracy"]) >= cfg.competence_threshold


def demonstrate_v2(
    w: World,
    teacher_id: str,
    learner: AgentRec,
    family: str | None,
    rng: np.random.Generator,
    acc: dict[str, Any] | None,
    source_label: str,
    scratch_entry: dict[str, Any] | None = None,
) -> None:
    """Targeted, reliability-tracked demonstration (replaces ``World._demonstrate`` when v2 is on).

    ``family=None`` selects the learner's weakest family. The learner's information gain
    (log p(correct) after minus before) is measured on the learner's own policy, so the counter
    reports what the demonstration actually changed, not what it was meant to teach.
    """
    cfg, st = w._sv2, w._sv2_state
    assert cfg is not None
    if not cfg.information:
        return
    teacher = w._agents[teacher_id]
    sw = w.symbol_world
    fam = family or _weakest_family(w, learner.id)
    item = sw.sample_item(fam, rng, split="train")
    t_obs = sw.observe(item, teacher_id, w.tick, w._age(teacher), w.p.enable_age_gate)
    t_dec = w._policies[teacher_id].decide(t_obs, rng, explore=False)
    misinformer = bool(teacher.flags.get("misinformer", False))
    if misinformer:
        wrong = [x for x in range(item.n_actions) if x != item.correct_action]
        action = int(rng.choice(wrong))
        rewards = np.zeros(item.n_actions)
        rewards[action] = 1.0
        fb = Feedback(mode="full", reward=1.0, correct_action=action, rewards_all=rewards, source="peer")
        full = True
    else:
        action = int(t_dec.action)
        full = _teacher_competent(w, teacher, fam)
        if full:
            # instructional feedback: the teacher asserts its answer as the label (right only if the
            # teacher is right); label noise still applies to what is told
            flip = bool(rng.random() < w.p.label_noise)
            told = action
            if flip:
                others = [x for x in range(item.n_actions) if x != action]
                told = int(rng.choice(others))
            rewards = np.zeros(item.n_actions)
            rewards[told] = 1.0
            fb = Feedback(
                mode="full",
                reward=float(rewards[action]),
                correct_action=told,
                rewards_all=rewards,
                source="peer",
            )
        else:
            fb = sw.feedback(item, action, "bandit", rng, noise=w.p.label_noise, source="peer")
    l_obs = sw.observe(item, learner.id, w.tick, w._age(learner), w.p.enable_age_gate)
    pol = w._policies[learner.id]
    p_before = float(pol.action_probabilities(l_obs)[item.correct_action])
    demo = Demonstration(l_obs, action, fb, teacher_id, w._trust(learner.id, teacher_id))
    upd = pol.learn_from_demonstration(demo)
    p_after = float(pol.action_probabilities(l_obs)[item.correct_action])
    gain = math.log(max(p_after, 1e-6)) - math.log(max(p_before, 1e-6))
    correct = action == item.correct_action
    # reliability-tracked trust: the learner observes whether the demonstrated action paid off
    e = w._edges.get(learner.id, {}).get(teacher_id)
    if e is not None:
        e.trust = float(
            np.clip(e.trust + cfg.trust_reliability_rate * ((1.0 if correct else 0.0) - e.trust), 0.0, 1.0)
        )
    _bump(st, "demos")
    _bump(st, "demos_full", 1.0 if full else 0.0)
    _bump(st, "demos_correct", 1.0 if correct else 0.0)
    if misinformer:
        _bump(st, "demos_from_misinformers")
        _bump(st, "demos_from_misinformers_correct", 1.0 if correct else 0.0)
    _bump(st, "info_gain_sum", gain)
    _bump(st, "info_gain_n")
    if scratch_entry is not None:
        scratch_entry["sv2_gain"] = scratch_entry.get("sv2_gain", 0.0) + gain
        scratch_entry["sv2_n"] = scratch_entry.get("sv2_n", 0) + 1
    if acc is not None:
        src = acc.setdefault(source_label, {"n": 0, "correct": 0, "reward": 0.0})
        src["n"] += 1
        src["correct"] += int(correct)
        src["reward"] += 0.0 if math.isnan(fb.reward) else fb.reward
        if upd is not None:
            acc["_delta"] = acc.get("_delta", 0.0) + upd.param_delta_norm
            acc["_updates"] = acc.get("_updates", 0) + 1


def cooperate(
    w: World, a_id: str, b_id: str, rng: np.random.Generator, scratch: dict[str, dict[str, Any]]
) -> None:
    """One shared item: both act, both learn the environment's answer; joint payoff updates mutual trust."""
    cfg, st = w._sv2, w._sv2_state
    assert cfg is not None
    if not (cfg.information and cfg.cooperation):
        return
    sw = w.symbol_world
    fam = TRAINING_FAMILIES[(w.tick + int(a_id[1:]) + int(b_id[1:])) % len(TRAINING_FAMILIES)]
    item = sw.sample_item(fam, rng, split="train")
    results: dict[str, bool] = {}
    for me in (a_id, b_id):
        ag = w._agents[me]
        obs = sw.observe(item, me, w.tick, w._age(ag), w.p.enable_age_gate)
        pol = w._policies[me]
        dec = pol.decide(obs, w.streams.get("agent", me, "learn"), explore=True)
        results[me] = int(dec.action) == item.correct_action
        fb = sw.feedback(item, int(dec.action), "full", rng, noise=w.p.label_noise, source="peer")
        pol.learn(obs, int(dec.action), fb)
        if me in scratch:
            acc = scratch[me]["acc"]
            src = acc.setdefault("cooperation", {"n": 0, "correct": 0, "reward": 0.0})
            src["n"] += 1
            src["correct"] += int(results[me])
            src["reward"] += fb.reward if math.isfinite(fb.reward) else 0.0
    joint = results[a_id] and results[b_id]
    for me, other in ((a_id, b_id), (b_id, a_id)):
        e = w._edges.get(me, {}).get(other)
        if e is not None:
            e.trust = float(np.clip(e.trust + 0.1 * ((1.0 if results[other] else 0.0) - e.trust), 0.0, 1.0))
    _bump(st, "cooperation_items")
    _bump(st, "cooperation_joint_correct", 1.0 if joint else 0.0)


# ------------------------------------------------------------------------------------- meetings


def meeting_v2(w: World, a_id: str, b_id: str, scratch: dict[str, dict[str, Any]]) -> None:
    """Plaza/family meeting under v2: trust game, targeted demonstrations, cooperation, ties."""
    cfg, st = w._sv2, w._sv2_state
    assert cfg is not None
    a, b = w._agents[a_id], w._agents[b_id]
    for x in (a_id, b_id):
        if x in scratch:
            scratch[x]["met"] = True
    family = w._is_family(a_id, b_id)
    key = _key(a_id, b_id)
    repeat = key in st.last_contact
    new_tie = b_id not in w._edges.get(a_id, {})
    e_ab = w._edges[a_id].get(b_id) or w._edge(a_id, b_id, "acquaintance")
    e_ba = w._edges[b_id].get(a_id) or w._edge(b_id, a_id, "acquaintance")
    st.last_contact[key] = w.tick
    _bump(st, "meetings")
    if repeat:
        _bump(st, "repeat_meetings")
    if new_tie:
        _bump(st, "ties_formed")
    # trust game (stag hunt) — unchanged from Phase I
    acts: dict[str, int] = {}
    obss: dict[str, Observation] = {}
    for me, _other, e in ((a_id, b_id, e_ab), (b_id, a_id, e_ba)):
        ctx = np.array(
            [[e.trust, 1.0 if family else 0.0, 1.0 if e.kind == "friend" else 0.0, 1.0]], dtype=np.float32
        )
        obs = Observation(me, w.tick, w._age(w._agents[me]), "social", "trust_game", ctx, 2)
        obss[me] = obs
        acts[me] = int(w._policies[me].decide(obs, w.streams.get("agent", me, "act"), explore=True).action)
    from hpe.world.engine import TRUST_PAYOFF, _clip01  # local import: no cycle at module load

    for me, other, e in ((a_id, b_id, e_ab), (b_id, a_id, e_ba)):
        pay = TRUST_PAYOFF[(acts[me], acts[other])]
        w._policies[me].learn(obss[me], acts[me], Feedback(mode="bandit", reward=pay, source="peer"))
        e.trust = _clip01(e.trust + 0.2 * (acts[other] - e.trust))
        e.interactions += 1
    # targeted demonstrations each way (isolated agents only learn from family)
    n_demo = (
        cfg.demos_per_meeting
        if cfg.demos_per_meeting is not None
        else w.p.peer_demonstrations_per_social_tick
    )
    n_demo = round(n_demo * w.k)
    for teacher, learner in ((a_id, b_id), (b_id, a_id)):
        L = w._agents[learner]
        if L.flags.get("isolated", False) and not family:
            continue
        rng = w.streams.get("agent", learner, "learn")
        entry = scratch.get(learner)
        acc = entry["acc"] if entry is not None else None
        for _ in range(n_demo):
            demonstrate_v2(w, teacher, L, None, rng, acc, "peer", entry)
    if not (a.flags.get("isolated", False) or b.flags.get("isolated", False)):
        cooperate(w, a_id, b_id, w.streams.get("social", "v2", "coop"), scratch)
    # learning-progress signal for the activity bandit (same term self-study uses)
    if cfg.learn_signal:
        for x in (a_id, b_id):
            entry = scratch.get(x)
            if entry is not None and entry.get("sv2_n", 0) > 0 and w._agents[x].activity == "socialize":
                entry["learn"] = float(np.clip(entry["sv2_gain"] / entry["sv2_n"], 0.0, 1.0))
    # friendship (same thresholds as Phase I; v2 makes repeated contact possible)
    if (
        not family
        and e_ab.kind != "friend"
        and min(e_ab.interactions, e_ba.interactions) >= 3
        and min(e_ab.trust, e_ba.trust) >= 0.6
    ):
        from hpe.world import population as popmod

        e_ab.kind = e_ba.kind = "friend"
        for rec in (a, b):
            rec.values["relationships"] = rec.values.get("relationships", 0.0) + 0.01
            rec.values = popmod.normalise_values(rec.values)
        _bump(st, "friendships")
        w._emit(
            EventType.FRIENDSHIP_FORMED,
            a_id,
            {"with": b_id, "trust": [round(e_ab.trust, 4), round(e_ba.trust, 4)], "v2": True},
            mechanism="world.social.v2.friendship",
            other_ids=[b_id],
            milestone=True,
        )
    # partnership — unchanged from Phase I
    ages = (w._age(a), w._age(b))
    if (
        not family
        and a.partner_id is None
        and b.partner_id is None
        and all(20 <= x <= 45 for x in ages)
        and a.household_id != b.household_id
        and min(e_ab.trust, e_ba.trust) >= 0.6
        and min(e_ab.interactions, e_ba.interactions) >= 4
        and w.streams.get("agent", min(a_id, b_id), "family").random() < 0.25 * w.k
    ):
        w._form_partnership(a, b)


# ------------------------------------------------------------------------------------- yearly


def yearly(w: World) -> None:
    """Tie decay, trust sampling by demonstrator type, and the yearly instrumentation event."""
    cfg, st = w._sv2, w._sv2_state
    assert cfg is not None
    year = int(math.floor(w.year))
    if year <= st.last_decay_year:
        return
    st.last_decay_year = year
    ticks_per_year = max(1, round(365.25 / w.dt_days))
    for aid in sorted(w._edges):
        for b, e in sorted(w._edges[aid].items()):
            if e.kind not in ("friend", "acquaintance"):
                continue
            last = st.last_contact.get(_key(aid, b), -1)
            if last >= 0 and w.tick - last <= ticks_per_year:
                continue
            before = e.interactions
            e.interactions = int(math.floor(e.interactions * cfg.tie_decay_per_year))
            if e.interactions < before:
                _bump(st, "ties_decayed")
            if e.kind == "friend" and e.interactions < cfg.friend_min_strength:
                e.kind = "acquaintance"
                _bump(st, "friendships_lost")
    honest: list[float] = []
    mis: list[float] = []
    alive = [aid for aid in sorted(w._agents) if w._agents[aid].alive]
    for aid in alive:
        for b, e in w._edges.get(aid, {}).items():
            tb = w._agents.get(b)
            if tb is None or e.kind not in ("friend", "acquaintance"):
                continue
            (mis if tb.flags.get("misinformer", False) else honest).append(e.trust)
    st.trust_honest.append(float(np.mean(honest)) if honest else float("nan"))
    st.trust_misinformer.append(float(np.mean(mis)) if mis else float("nan"))
    n_alive = max(1, len(alive))
    payload = {
        "year": year,
        "config": cfg.as_dict(),
        "year_stats": {k: round(v, 6) for k, v in st.year_stats.items()},
        "cumulative": {k: round(v, 6) for k, v in st.stats.items()},
        "per_alive_agent": {
            k: round(v / n_alive, 6) for k, v in st.year_stats.items() if k != "info_gain_sum"
        },
        "mean_info_gain": round(st.year_stats["info_gain_sum"] / st.year_stats["info_gain_n"], 6)
        if st.year_stats["info_gain_n"]
        else None,
        "mean_trust_honest": None if math.isnan(st.trust_honest[-1]) else round(st.trust_honest[-1], 4),
        "mean_trust_misinformer": None
        if math.isnan(st.trust_misinformer[-1])
        else round(st.trust_misinformer[-1], 4),
        "friend_ties": sum(1 for aid in alive for e in w._edges.get(aid, {}).values() if e.kind == "friend")
        / 2,
        "acquaintance_ties": sum(
            1 for aid in alive for e in w._edges.get(aid, {}).values() if e.kind == "acquaintance"
        )
        / 2,
    }
    w._emit(EventType.SOCIAL_INTERACTION, None, payload, mechanism="world.social.v2.stats")
    st.year_stats = dict.fromkeys(STATS_KEYS, 0.0)
