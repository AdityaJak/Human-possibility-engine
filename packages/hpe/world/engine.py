"""Deterministic, checkpointable society kernel (``hpe.world.World``).

The world is an inspectable state machine, not a narrative generator. Every mechanism below
is an explicit, abstract modelling assumption (see ``ASSUMPTIONS.md``); none is calibrated to
a real population, and simulated outcomes say nothing about real human lives.

One tick (default 30 simulated days) runs these phases in fixed order:

1. interventions & institutions (assignment flags, school enrolment/graduation, stages)
2. activity decisions (``kind="activity"``) for every alive agent aged >= 3, sorted by id
3. individual activities: caregiving demonstrations (ages 1-5), school, tutoring, self-study
   (with a ``kind="practice"`` family choice), work (job-family episodes -> wage), rest
4. social phase: classmate / coworker contact, mentoring, plaza meetings with a stag-hunt
   trust game (``kind="social"``), trust-weighted peer demonstrations, friendship and
   partnership formation
5. economy (consumption, benefits, pensions), affect-proxy updates, values-weighted activity
   reward -> ``policy.learn``; ``policy.tick_lifecycle``
6. exogenous shocks, mortality, births (inherited initial conditions, fresh policy)
7. on year boundaries: held-out assessment (``hpe.tasks.assess``, learning disabled), job
   market, wage records, parameter-summary snapshots

Skill only ever changes through ``hpe.tasks.practice`` / demonstrations handed to the policy,
and is only ever *measured* by ``hpe.tasks.assess``. Interventions change inputs (episode
counts, feedback quality, access, contacts, resources), never measured outcomes.

Randomness: every stochastic purpose draws from its own ``RngStreams`` key — per agent
``act`` (decisions), ``learn`` (task items, feedback noise), ``life`` (mortality), ``family``
(partnership/birth acceptance) — plus global ``social``/``shocks``/``market``/``map`` streams
and stateless keyed draws for intervention assignment. Twin arms with equal seeds therefore
share common random numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from hpe.contracts.protocols import EventSink
from hpe.contracts.runtime import (
    ACTIVITY_ACTIONS,
    AgentInit,
    Demonstration,
    Feedback,
    FeedbackMode,
    FeedbackSource,
    Observation,
    Policy,
    PolicyFactory,
)
from hpe.contracts.schemas import (
    SCHEMA_VERSION,
    AffectProxies,
    AgentFrame,
    AgentState,
    CapabilityState,
    ComputeReport,
    EventType,
    ExperienceEvent,
    Intervention,
    InterventionKind,
    JobStatus,
    LifeStage,
    OutcomeObservation,
    Place,
    PolicyDescriptor,
    RunManifest,
    SimulationConfig,
    SocialEdge,
    SocialEdgeKind,
    WorldMap,
    WorldTick,
)
from hpe.core.hashing import stable_hash, stable_json
from hpe.core.rng import RngStreams, derive_seed, stream
from hpe.tasks import ALL_FAMILIES, TRAINING_FAMILIES, SymbolWorld, assess, practice
from hpe.tasks.assessment import PracticeStats
from hpe.world import geography as geo
from hpe.world import population as popmod
from hpe.world.records import EDGE_PRIORITY, FAMILY_EDGE_KINDS, AgentRec, EdgeRec, Household, Job

SNAPSHOT_FORMAT = "hpe.world.snapshot/1"
WORLD_VERSION = "0.1.0"

# Initial agents receive a condensed, seeded practice history so they are not tabula rasa at
# t=0 (see ASSUMPTIONS.md §2). Applied identically to every policy class.
PREHISTORY_MAX_EPISODES = 400
ACT_INDEX = {a: i for i, a in enumerate(ACTIVITY_ACTIONS)}
STAGE_ORDER = [s.value for s in LifeStage]
_PUBLIC_EDGE_KINDS = frozenset(k.value for k in SocialEdgeKind)
TRUST_INIT = {"parent": 0.9, "child": 0.9, "partner": 0.9, "sibling": 0.8, "mentor": 0.7, "tutor": 0.8}

# Stag-hunt payoffs for the trust game: (my_action, partner_action) -> my payoff.
TRUST_PAYOFF = {(1, 1): 1.0, (1, 0): 0.0, (0, 1): 0.4, (0, 0): 0.4}


def stage_for_age(age: float) -> str:
    if age < 3:
        return LifeStage.INFANT.value
    if age < 12:
        return LifeStage.CHILD.value
    if age < 18:
        return LifeStage.ADOLESCENT.value
    if age < 65:
        return LifeStage.ADULT.value
    return LifeStage.ELDER.value


def _gini(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    x = np.sort(np.clip(x, 0.0, None))
    total = x.sum()
    if total <= 0:
        return 0.0
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * total))


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _r(x: float, nd: int = 6) -> float:
    return round(float(x), nd)


def _jsonable(x: Any) -> Any:
    return json.loads(stable_json(x))


def _hash_unit(*parts: int) -> float:
    """Stateless uniform in [0,1) from integers (used for log sampling only)."""
    return geo._mix(parts[0], parts[1] if len(parts) > 1 else 0) / 4294967296.0


def _code_version() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=os.path.dirname(__file__),
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - environment dependent
        return None


def _hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:  # pragma: no cover
        pass
    return f"{cpu}; {os.cpu_count()} logical CPUs; {platform.system()} {platform.machine()}; Python {platform.python_version()}"


class World:
    """Society simulation implementing ``hpe.contracts.protocols.WorldRunner``."""

    def __init__(
        self,
        config: SimulationConfig,
        policy_factory: PolicyFactory,
        sink: EventSink | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.factory = policy_factory
        self.sink = sink
        self.config_hash = stable_hash(config.model_dump(mode="json"))
        self.run_id = run_id or f"run_{self.config_hash[:12]}"
        self.p = config.world
        self.dt_days = int(config.dt_days)
        self.dt_years = self.dt_days / 365.25
        self.scale = self.dt_days / 30.0  # per-tick rates were specified for 30-day ticks
        self.k = min(1.0, self.scale)
        self.ticks_per_year = max(1, round(365.25 / self.dt_days))
        self.max_population = int(self.p.max_population or math.ceil(1.5 * config.n_agents))
        self.symbol_world = SymbolWorld(seed=config.resolved_task_seed())
        self.task_world_hash = self.symbol_world.fingerprint()
        self.streams = RngStreams(config.seed)
        self.on_checkpoint: Callable[[int, dict[str, Any]], None] | None = None
        self._reset_state()

    # ================================================================== state containers
    def _reset_state(self) -> None:
        self.tick = 0
        self._initialized = False
        self._truncated = False
        self._agents: dict[str, AgentRec] = {}
        self._init_specs: dict[str, dict[str, Any]] = {}
        self._policies: dict[str, Policy] = {}
        self._households: dict[str, Household] = {}
        self._places: dict[str, Place] = {}
        self._jobs: dict[str, Job] = {}
        self._edges: dict[str, dict[str, EdgeRec]] = {}
        self._counters = {"agent": 0, "household": 0, "job": 0, "seq": 0, "births": 0, "deaths": 0}
        self._hash = hashlib.sha256(b"hpe.world.events/1").hexdigest()
        self._downturn_until = -1
        self._commons = 0.0
        self._ticks: list[WorldTick] = []
        self._outcomes: list[list[Any]] = []
        self._timelines: dict[str, list[dict[str, Any]]] = {}
        self._skill_curves: dict[str, list[dict[str, Any]]] = {}
        self._param_history: dict[str, list[dict[str, Any]]] = {}
        self._last_explanation: dict[str, Any] = {}
        self._compute = {"wall_time_s": 0.0, "cpu_time_s": 0.0, "agent_years": 0.0}
        self._tick_events = 0
        self._births_tick = 0
        self._deaths_tick = 0

    # ================================================================== properties
    @property
    def year(self) -> float:
        return self.tick * self.dt_days / 365.25

    @property
    def day(self) -> int:
        return self.tick * self.dt_days

    @property
    def done(self) -> bool:
        return self._truncated or self.year >= self.config.years - 1e-9

    def _age(self, a: AgentRec, tick: int | None = None) -> float:
        t = self.tick if tick is None else tick
        return (t - a.birth_tick) * self.dt_years

    def _alive_ids(self) -> list[str]:
        return [i for i in sorted(self._agents) if self._agents[i].alive]

    # ================================================================== events
    def _emit(
        self,
        etype: EventType,
        agent_id: str | None = None,
        payload: dict[str, Any] | None = None,
        mechanism: str = "world",
        other_ids: list[str] | None = None,
        source: str = "world",
        milestone: bool = False,
    ) -> None:
        if self.config.max_events is not None and self._counters["seq"] >= self.config.max_events:
            self._truncated = True
            return
        d: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "seq": self._counters["seq"],
            "tick": self.tick,
            "day": self.day,
            "type": etype.value,
            "agent_id": agent_id,
            "other_ids": list(other_ids or []),
            "payload": payload or {},
            "provenance": {
                "source": source,
                "mechanism": mechanism,
                "rng_stream": None,
                "code_version": None,
            },
            "namespace": "synthetic_simulation",
        }
        self._counters["seq"] += 1
        self._tick_events += 1
        try:  # fast path: payloads are built from JSON-native types (rounded floats)
            canon = json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            canon = stable_json(d)
        self._hash = hashlib.sha256((self._hash + canon).encode("utf-8")).hexdigest()
        if self.sink is not None or milestone:
            full = json.loads(canon)
            full["run_id"] = self.run_id
            if self.sink is not None:
                self.sink.emit(ExperienceEvent.model_validate(full))
            if milestone:
                for aid in [agent_id, *(other_ids or [])]:
                    if aid is not None and aid in self._timelines:
                        self._timelines[aid].append(full)

    def event_stream_hash(self) -> str:
        return self._hash

    # ================================================================== initialisation
    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self._initialize()

    def _new_agent_id(self) -> str:
        i = self._counters["agent"]
        self._counters["agent"] += 1
        return f"a{i:05d}"

    def _new_household(self, members: list[str], resources: float) -> Household:
        i = self._counters["household"]
        self._counters["household"] += 1
        hid = f"h{i:04d}"
        home = geo.make_home(hid, self.streams.get("map", "homes"))
        self._places[home.place_id] = home
        hh = Household(
            id=hid,
            home_place_id=home.place_id,
            members=list(members),
            resources=float(resources),
            formed_tick=self.tick,
        )
        self._households[hid] = hh
        return hh

    def _make_policy(self, aid: str) -> Policy:
        spec = self._init_specs[aid]
        init = AgentInit(
            agent_id=aid,
            birth_tick=int(spec["birth_tick"]),
            seed=int(spec["seed"]),
            traits=dict(spec["traits"]),
            base_params=dict(spec["base_params"]),
            families={k: dict(v) for k, v in spec["families"].items()},
            activity_actions=list(spec["activity_actions"]),
            inherited=dict(spec["inherited"]),
        )
        return self.factory(init)

    def _register_agent(self, rec: AgentRec) -> None:
        self._agents[rec.id] = rec
        self._edges.setdefault(rec.id, {})
        self._timelines[rec.id] = []
        self._skill_curves[rec.id] = []
        self._param_history[rec.id] = []
        self._init_specs[rec.id] = {
            "birth_tick": rec.birth_tick,
            "seed": rec.seed,
            "traits": dict(rec.traits),
            "base_params": dict(rec.base_params),
            "families": self.symbol_world.families_meta(),
            "activity_actions": list(ACTIVITY_ACTIONS),
            "inherited": dict(rec.inherited),
        }
        self._policies[rec.id] = self._make_policy(rec.id)

    def _initialize(self) -> None:
        self._initialized = True
        cfg = self.config
        for pl in geo.civic_places(cfg.n_agents):
            self._places[pl.place_id] = pl
        prng = stream(cfg.resolved_population_seed(), "population")
        drafts = popmod.initial_households(
            cfg.n_agents, self.p.n_households, self.p.initial_age_distribution, prng
        )
        self._emit(
            EventType.RUN_START,
            payload={
                "config_hash": self.config_hash,
                "task_world_hash": self.task_world_hash,
                "n_agents": cfg.n_agents,
                "policy_class": str(cfg.policy_class.value),
                "world_version": WORLD_VERSION,
            },
            mechanism="world.init",
        )
        pop_seed = cfg.resolved_population_seed()
        for hd in drafts:
            hh = self._new_household([], hd.resources)
            ids: list[str] = []
            for ps in hd.members:
                aid = self._new_agent_id()
                ids.append(aid)
                birth_tick = -round(ps.age / self.dt_years)
                rec = AgentRec(
                    id=aid,
                    name=ps.name,
                    household_id=hh.id,
                    birth_tick=birth_tick,
                    birth_cohort_year=math.floor(birth_tick * self.dt_years),
                    traits=ps.traits,
                    base_params=ps.base_params,
                    values=ps.values,
                    persona=popmod.persona_text(ps.name, hh.id, ps.traits, ps.values, ps.age),
                    seed=derive_seed(pop_seed, "policy", aid),
                )
                rec.stage = stage_for_age(ps.age)
                rec.energy = float(np.clip(prng.normal(0.75, 0.1), 0.3, 1.0))
                rec.social = float(np.clip(prng.normal(0.5, 0.1), 0.1, 0.9))
                self._register_agent(rec)
                hh.members.append(aid)
            for i, ps in enumerate(hd.members):
                rec = self._agents[ids[i]]
                if ps.partner_idx is not None:
                    rec.partner_id = ids[ps.partner_idx]
                    self._link(rec.id, rec.partner_id, "partner")
                for pj in ps.parents_idx:
                    rec.parents.append(ids[pj])
                    self._agents[ids[pj]].children.append(rec.id)
                    self._link(rec.id, ids[pj], "child")  # rec is child of pj
            kids = [i for i in ids if self._agents[i].parents]
            for x in kids:
                for y in kids:
                    if x < y:
                        self._link(x, y, "sibling")
        for aid in sorted(self._agents):
            a = self._agents[aid]
            self._emit(
                EventType.BIRTH,
                aid,
                {
                    "initial": True,
                    "age_at_start": _r(self._age(a), 3),
                    "household_id": a.household_id,
                    "persona": a.persona,
                },
                mechanism="world.init.population",
                other_ids=list(a.parents),
                milestone=True,
            )
        for aid in sorted(self._agents):
            self._prehistory(self._agents[aid])
        self._apply_interventions()
        self._update_institutions()
        self._assess_all()
        self._job_market(initial=True)
        self._param_snapshot()
        tick = self._make_tick(0.0, include_frames=False)
        self._ticks.append(tick)
        if self.sink is not None:
            self.sink.emit_tick(tick)

    def _prehistory(self, a: AgentRec) -> None:
        """Condensed, seeded developmental history for agents who exist at t=0."""
        age = self._age(a)
        if age < 3:
            return
        pol = self._policies[a.id]
        rng = self.streams.get("agent", a.id, "prehistory")
        total = int(PREHISTORY_MAX_EPISODES * min(age, 25.0) / 25.0)
        years = np.arange(3.0, age, 1.0)
        if years.size == 0 or total <= 0:
            return
        per_year = max(1, total // years.size)
        job_family = TRAINING_FAMILIES[int(rng.integers(0, len(TRAINING_FAMILIES)))]
        n_done = 0
        for y in years:
            if n_done >= total:
                break
            n = min(per_year, total - n_done)
            in_school = self.p.school_start_age <= y < self.p.school_end_age
            if in_school:
                a.education_years += 1.0
                n_full = round(n * self.p.school_full_feedback_fraction)
                for i, fam in enumerate(TRAINING_FAMILIES):
                    nf = n_full // 3 + (1 if i < n_full % 3 else 0)
                    nb = (n - n_full) // 3 + (1 if i < (n - n_full) % 3 else 0)
                    self._practice(pol, a, fam, nf, rng, "full", "school", float(y), tick=-1)
                    self._practice(pol, a, fam, nb, rng, "bandit", "school", float(y), tick=-1)
            elif y >= self.p.school_end_age:
                self._practice(pol, a, job_family, n, rng, "bandit", "job", float(y), tick=-1)
            else:
                fam = TRAINING_FAMILIES[int(y) % 3]
                self._practice(pol, a, fam, n, rng, "bandit", "self", float(y), tick=-1)
            n_done += n
            pol.tick_lifecycle(float(y), 1.0)
        pol.tick_lifecycle(age, self.dt_years)
        if self.config.log.log_learning_episodes:
            self._emit(
                EventType.LEARNING_EPISODE,
                a.id,
                {"prehistory": True, "episodes": n_done, "school_years": a.education_years},
                mechanism="world.init.prehistory",
            )

    # ================================================================== social graph
    def _link(self, a: str, b: str, kind: str, trust: float | None = None) -> EdgeRec:
        """Create/upgrade directed edges a->b and b->a. ``kind`` is a's role w.r.t. b for family."""
        inverse = {"child": "parent", "parent": "child"}
        e_ab = self._edge(a, b, kind, trust)
        self._edge(b, a, inverse.get(kind, kind), trust)
        return e_ab

    def _edge(self, a: str, b: str, kind: str, trust: float | None = None) -> EdgeRec:
        adj = self._edges.setdefault(a, {})
        e = adj.get(b)
        if e is None:
            t = TRUST_INIT.get(kind, 0.5) if trust is None else trust
            e = EdgeRec(kind=kind, trust=float(t), interactions=0, formed_tick=self.tick)
            adj[b] = e
        elif EDGE_PRIORITY.get(kind, 0) > EDGE_PRIORITY.get(e.kind, 0):
            e.kind = kind
            if trust is not None:
                e.trust = max(e.trust, trust)
        return e

    def _trust(self, a: str, b: str) -> float:
        e = self._edges.get(a, {}).get(b)
        return e.trust if e is not None else 0.5

    def _is_family(self, a: str, b: str) -> bool:
        e = self._edges.get(a, {}).get(b)
        return e is not None and e.kind in FAMILY_EDGE_KINDS

    # ================================================================== learning helpers
    def _practice(
        self,
        pol: Policy,
        a: AgentRec,
        family: str,
        n: int,
        rng: np.random.Generator,
        mode: FeedbackMode,
        source: FeedbackSource,
        age: float,
        *,
        tick: int | None = None,
        difficulty: float | None = None,
        acc: dict[str, Any] | None = None,
    ) -> PracticeStats | None:
        if n <= 0:
            return None
        st = practice(
            pol,
            self.symbol_world,
            family,
            n,
            rng,
            agent_id=a.id,
            tick=self.tick if tick is None else tick,
            age_years=age,
            mode=mode,
            source=source,
            noise=self.p.label_noise,
            enable_gate=self.p.enable_age_gate,
            difficulty=difficulty,
        )
        a.practice_counts[family] = a.practice_counts.get(family, 0) + st.n
        if st.n:
            prev = a.recent_reward.get(family, st.mean_reward)
            a.recent_reward[family] = 0.7 * prev + 0.3 * (st.rewards / st.n if mode != "none" else prev)
            a.last_practice_tick[family] = self.tick if tick is None else tick
        if acc is not None:
            src = acc.setdefault(str(source), {"n": 0, "correct": 0, "reward": 0.0})
            src["n"] += st.n
            src["correct"] += st.correct
            src["reward"] += st.rewards
            acc["_delta"] = acc.get("_delta", 0.0) + st.delta_norm_sum
            acc["_updates"] = acc.get("_updates", 0) + st.n_updates
            fams = acc.setdefault("_families", {})
            fams[family] = fams.get(family, 0) + st.n
        return st

    def _demonstrate(
        self,
        teacher_id: str,
        learner: AgentRec,
        family: str,
        rng: np.random.Generator,
        acc: dict[str, Any] | None,
        source_label: str,
    ) -> None:
        """Teacher acts greedily on a train item; learner observes action + outcome (trust-weighted)."""
        teacher = self._agents[teacher_id]
        sw = self.symbol_world
        item = sw.sample_item(family, rng, split="train")
        t_obs = sw.observe(item, teacher_id, self.tick, self._age(teacher), self.p.enable_age_gate)
        t_dec = self._policies[teacher_id].decide(t_obs, rng, explore=False)
        if teacher.flags.get("misinformer", False):
            wrong = [x for x in range(item.n_actions) if x != item.correct_action]
            action = int(rng.choice(wrong))
            rewards = np.zeros(item.n_actions)
            rewards[action] = 1.0
            fb = Feedback(mode="full", reward=1.0, correct_action=action, rewards_all=rewards, source="peer")
        else:
            action = int(t_dec.action)
            fb = sw.feedback(item, action, "bandit", rng, noise=self.p.label_noise, source="peer")
        l_obs = sw.observe(item, learner.id, self.tick, self._age(learner), self.p.enable_age_gate)
        demo = Demonstration(l_obs, action, fb, teacher_id, self._trust(learner.id, teacher_id))
        upd = self._policies[learner.id].learn_from_demonstration(demo)
        if acc is not None:
            src = acc.setdefault(source_label, {"n": 0, "correct": 0, "reward": 0.0})
            src["n"] += 1
            src["correct"] += int(action == item.correct_action)
            src["reward"] += 0.0 if math.isnan(fb.reward) else fb.reward
            if upd is not None:
                acc["_delta"] = acc.get("_delta", 0.0) + upd.param_delta_norm
                acc["_updates"] = acc.get("_updates", 0) + 1

    def _difficulty_for(self, a: AgentRec, family: str) -> float:
        m = a.measured.get(family)
        if not m:
            return 0.0
        chance = 1.0 / self.symbol_world.families_meta()[family]["n_actions"]
        return float(np.clip((m["accuracy"] - chance) / (1 - chance), 0.0, 1.0))

    # ================================================================== interventions
    def _iv_active(self, iv: Intervention) -> bool:
        end = math.inf if iv.end_year is None else iv.end_year
        return iv.start_year <= self.year + 1e-9 < end

    def _iv_targeted(self, iv: Intervention, a: AgentRec) -> bool:
        key = f"iv:{iv.name}:assigned"
        if key not in a.flags:
            t = iv.target
            if t.agent_ids is not None:
                hit = a.id in t.agent_ids
            elif t.household_ids is not None:
                hit = a.household_id in t.household_ids
            else:
                hit = bool(stream(self.config.seed, "intervention", iv.name, a.id).random() < t.fraction)
            a.flags[key] = hit
        return a.flags[key]

    def _iv_eligible(self, iv: Intervention, a: AgentRec, age: float) -> bool:
        return (
            self._iv_active(iv) and iv.target.age_min <= age <= iv.target.age_max and self._iv_targeted(iv, a)
        )

    def _apply_interventions(self) -> None:
        """Set per-tick mechanism flags. Interventions alter inputs/access only."""
        cash_households: dict[str, float] = {}
        new_year = self._year_boundary_crossed()
        for aid in self._alive_ids():
            a = self._agents[aid]
            age = self._age(a)
            base_tutor = (
                self.p.baseline_tutoring_fraction > 0
                and 5 <= age < self.p.school_end_age
                and stream(self.config.seed, "baseline_tutoring", aid).random()
                < self.p.baseline_tutoring_fraction
            )
            flags = {
                "tutored": base_tutor,
                "school_blocked": False,
                "adult_ed": False,
                "isolated": False,
                "misinformer": False,
                "care_extra": False,
            }
            for iv in self.config.interventions:
                if iv.kind == InterventionKind.CROSS_FOSTER or not self._iv_eligible(iv, a, age):
                    continue
                kind = iv.kind
                if kind == InterventionKind.TUTORING_ACCESS:
                    flags["tutored"] = True
                elif kind == InterventionKind.SCHOOL_ACCESS:
                    flags["school_blocked"] = not bool(iv.params.get("access", False))
                elif kind == InterventionKind.MIDLIFE_EDUCATION:
                    flags["adult_ed"] = True
                elif kind == InterventionKind.SOCIAL_ISOLATION:
                    flags["isolated"] = True
                elif kind == InterventionKind.MISINFORMATION:
                    flags["misinformer"] = True
                elif kind == InterventionKind.CAREGIVING_SUPPORT:
                    flags["care_extra"] = True
                elif kind == InterventionKind.CASH_TRANSFER:
                    period = str(iv.params.get("period", "once"))
                    amount = float(iv.params.get("amount", 5.0))
                    once_key = f"iv:{iv.name}:paid"
                    pay = (
                        (period == "monthly")
                        or (period == "yearly" and (new_year or not a.flags.get(once_key, False)))
                        or (period == "once" and not a.flags.get(once_key, False))
                    )
                    if pay:
                        a.flags[once_key] = True
                        if a.household_id not in cash_households:
                            cash_households[a.household_id] = amount * (
                                self.scale if period == "monthly" else 1.0
                            )
                announce = f"iv:{iv.name}:announced"
                if not a.flags.get(announce, False):
                    a.flags[announce] = True
                    self._emit(
                        EventType.INTERVENTION,
                        aid,
                        {
                            "name": iv.name,
                            "kind": kind.value,
                            "params": _jsonable(iv.params),
                            "age": _r(age, 3),
                        },
                        mechanism=f"world.intervention.{kind.value}",
                        source="intervention",
                        milestone=True,
                    )
            if flags["tutored"] and not a.tutored:
                self._emit(
                    EventType.TUTORING_SESSION,
                    aid,
                    {"started": True, "age": _r(age, 3)},
                    mechanism="world.tutoring.start",
                    milestone=True,
                )
            a.tutored = bool(flags["tutored"])
            for k in ("school_blocked", "adult_ed", "isolated", "misinformer", "care_extra"):
                a.flags[k] = bool(flags[k])
        for hid in sorted(cash_households):
            if hid in self._households:
                self._households[hid].resources += cash_households[hid]

    # ================================================================== institutions
    def _school_for(self, a: AgentRec) -> str:
        if "school_south" in self._places and int(a.household_id[1:]) % 2 == 1:
            return "school_south"
        return "school_north"

    def _update_institutions(self) -> None:
        for aid in self._alive_ids():
            a = self._agents[aid]
            age = self._age(a)
            st = stage_for_age(age)
            if st != a.stage:
                old = a.stage
                a.stage = st
                self._emit(
                    EventType.STAGE_CHANGE,
                    aid,
                    {"from": old, "to": st, "age": _r(age, 3)},
                    mechanism="world.lifecycle.stage",
                    milestone=True,
                )
            school_age = self.p.school_start_age <= age < self.p.school_end_age
            can = school_age and not a.flags.get("school_blocked", False)
            if can and not a.enrolled:
                a.enrolled = True
                self._emit(
                    EventType.SCHOOL_ENROLL,
                    aid,
                    {"school": self._school_for(a), "age": _r(age, 3)},
                    mechanism="world.school.enrol",
                    milestone=True,
                )
            elif a.enrolled and not can:
                a.enrolled = False
                graduated = age >= self.p.school_end_age
                a.graduated = a.graduated or graduated
                self._emit(
                    EventType.SCHOOL_GRADUATE,
                    aid,
                    {
                        "graduated": graduated,
                        "reason": "completed" if graduated else "access_removed",
                        "education_years": _r(a.education_years, 3),
                    },
                    mechanism="world.school.exit",
                    milestone=True,
                )
            if a.job_id is not None and age >= self.p.retirement_age:
                self._end_job(a, "retirement")

    # ================================================================== tick
    def _year_boundary_crossed(self) -> bool:
        if self.tick == 0:
            return False
        return math.floor(self.year + 1e-9) > math.floor((self.tick - 1) * self.dt_days / 365.25 + 1e-9)

    def _interval_crossed(self, interval: float | None) -> bool:
        if not interval or interval <= 0 or self.tick == 0:
            return False
        prev = (self.tick - 1) * self.dt_days / 365.25
        return math.floor(self.year / interval + 1e-9) > math.floor(prev / interval + 1e-9)

    def step(self, n_ticks: int = 1, *, include_frames: bool = False) -> WorldTick:
        self._ensure_initialized()
        last: WorldTick | None = None
        for i in range(max(1, n_ticks)):
            if self.done:
                break
            last = self._step_one(include_frames=include_frames and i == n_ticks - 1)
        if last is None:
            last = self._ticks[-1]
            if include_frames:
                last = last.model_copy(update={"frames": self.frames()})
        return last

    def run(
        self, *, until_year: float | None = None, progress: Callable[[float], None] | None = None
    ) -> WorldTick:
        self._ensure_initialized()
        target = self.config.years if until_year is None else min(until_year, self.config.years)
        last = self._ticks[-1]
        report_every = max(1, self.ticks_per_year)
        while not self.done and self.year < target - 1e-9:
            last = self._step_one(include_frames=False)
            if progress is not None and self.tick % report_every == 0:
                progress(min(1.0, self.year / target if target > 0 else 1.0))
        if progress is not None:
            progress(1.0)
        return last

    def _maybe_end(self) -> None:
        if self.done and not getattr(self, "_ended", False):
            self._ended = True
            self._emit(
                EventType.RUN_END,
                payload={
                    "reason": "max_events" if self._truncated else "completed",
                    "population": len(self._alive_ids()),
                    "births": self._counters["births"],
                    "deaths": self._counters["deaths"],
                },
                mechanism="world.run.end",
            )
            if self.sink is not None:
                self.sink.flush()

    def _step_one(self, include_frames: bool) -> WorldTick:
        t0, c0 = time.perf_counter(), time.process_time()
        self.tick += 1
        self._tick_events = 0
        self._births_tick = 0
        self._deaths_tick = 0
        alive_start = self._alive_ids()
        self._compute["agent_years"] += len(alive_start) * self.dt_years

        self._apply_interventions()
        self._update_institutions()
        scratch = self._decide_activities()
        self._run_activities(scratch)
        self._social_phase(scratch)
        self._economy_and_affect(scratch)
        self._shocks()
        self._mortality()
        self._partnering()
        self._births()

        if self._interval_crossed(self.config.log.assessment_interval_years):
            self._assess_all()
        if self._year_boundary_crossed():
            self._job_market(initial=False)
            self._yearly_records()
            self._param_snapshot()
        if self.on_checkpoint is not None and self._interval_crossed(self.config.log.checkpoint_every_years):
            self.on_checkpoint(self.tick, self.snapshot())

        self._maybe_end()
        wall = time.perf_counter() - t0
        self._compute["wall_time_s"] += wall
        self._compute["cpu_time_s"] += time.process_time() - c0
        fe = self.config.log.frames_every_ticks
        sink_frames = self.sink is not None and fe > 0 and self.tick % fe == 0
        tick = self._make_tick(wall * 1000.0, include_frames=include_frames or sink_frames)
        if self.sink is not None:
            self.sink.emit_tick(tick if sink_frames else tick.model_copy(update={"frames": None}))
        stored = tick.model_copy(update={"frames": None}) if tick.frames is not None else tick
        self._ticks.append(stored)
        if not include_frames and tick.frames is not None:
            tick = stored
        return tick

    # ------------------------------------------------------------------ phase 2: decisions
    def _activity_context(self, a: AgentRec, age: float) -> np.ndarray:
        hh = self._households.get(a.household_id)
        per = (hh.resources / max(1, len(hh.members))) if hh else 0.0
        return np.array(
            [
                [
                    a.energy,
                    a.stress,
                    a.social,
                    math.tanh(max(per, 0.0) / 5.0),
                    min(age / 100.0, 1.2),
                    1.0 if a.job_id else 0.0,
                    1.0 if a.enrolled else 0.0,
                    1.0,
                ]
            ],
            dtype=np.float32,
        )

    def _activity_mask(self, a: AgentRec, age: float) -> np.ndarray:
        m = np.ones(len(ACTIVITY_ACTIONS), dtype=bool)
        if age < 18 or age >= self.p.retirement_age or a.enrolled:
            m[ACT_INDEX["work"]] = False
        return m

    def _decide_activities(self) -> dict[str, dict[str, Any]]:
        scratch: dict[str, dict[str, Any]] = {}
        rate = self.config.log.decision_sample_rate
        for aid in self._alive_ids():
            a = self._agents[aid]
            age = self._age(a)
            s: dict[str, Any] = {
                "age": age,
                "e0": a.energy,
                "s0": a.social,
                "st0": a.stress,
                "income": 0.0,
                "learn": None,
                "met": False,
                "acc": {},
            }
            if age < 3:
                a.activity = "rest"
                s["action"] = None
            else:
                obs = Observation(
                    aid,
                    self.tick,
                    age,
                    "activity",
                    "activity",
                    self._activity_context(a, age),
                    len(ACTIVITY_ACTIONS),
                    self._activity_mask(a, age),
                )
                dec = self._policies[aid].decide(obs, self.streams.get("agent", aid, "act"), explore=True)
                a.activity = ACTIVITY_ACTIONS[dec.action]
                s["obs"], s["action"] = obs, dec.action
                self._last_explanation[aid] = {
                    "tick": self.tick,
                    "kind": "activity",
                    "chosen": a.activity,
                    "probs": {ACTIVITY_ACTIONS[i]: _r(p, 4) for i, p in enumerate(dec.probs)},
                    "context": {
                        "energy": _r(a.energy, 3),
                        "stress": _r(a.stress, 3),
                        "social": _r(a.social, 3),
                        "employed": a.job_id is not None,
                        "enrolled": a.enrolled,
                    },
                    "policy_class": dec.policy_class,
                    "explanation": dec.explanation,
                }
                if rate > 0 and _hash_unit(int(aid[1:]), self.tick) < rate:
                    self._emit(
                        EventType.DECISION,
                        aid,
                        _jsonable(self._last_explanation[aid]),
                        mechanism="world.activity.decide",
                        source="policy",
                    )
            scratch[aid] = s
        return scratch

    # ------------------------------------------------------------------ phase 3: activities
    def _episodes(self, base: int, a: AgentRec) -> int:
        n = round(base * self.scale)
        if a.energy < 0.25:
            n //= 2
        return n

    def _caregiver_for(self, a: AgentRec) -> str | None:
        hh = self._households.get(a.household_id)
        if hh is None:
            return None
        adults = [
            m
            for m in sorted(hh.members)
            if m != a.id and self._agents[m].alive and self._age(self._agents[m]) >= 18
        ]
        if not adults:
            return None
        parents = [m for m in adults if m in a.parents]
        return parents[0] if parents else max(adults, key=lambda m: (self._age(self._agents[m]), m))

    def _run_activities(self, scratch: dict[str, dict[str, Any]]) -> None:
        p = self.p
        fams = TRAINING_FAMILIES
        for aid in sorted(scratch):
            a = self._agents[aid]
            s = scratch[aid]
            age = s["age"]
            pol = self._policies[aid]
            rng = self.streams.get("agent", aid, "learn")
            acc = s["acc"]
            idx = int(aid[1:])
            # caregiving demonstrations (ages 1 .. school start)
            if 1.0 <= age < p.school_start_age:
                cg = self._caregiver_for(a)
                if cg is not None:
                    n = self._episodes(p.caregiver_episodes_per_tick, a)
                    if a.flags.get("care_extra", False):
                        n += round(p.caregiver_episodes_per_tick * self.scale)
                    for i in range(n):
                        self._demonstrate(cg, a, fams[(self.tick + idx + i) % 3], rng, acc, "caregiver")
            # school
            if a.enrolled:
                a.education_years += self.dt_years
                n = self._episodes(p.school_episodes_per_tick, a)
                n_full = round(n * p.school_full_feedback_fraction)
                for j in range(3):
                    fam = fams[(self.tick + idx + j) % 3]
                    nf = n_full // 3 + (1 if j < n_full % 3 else 0)
                    nb = (n - n_full) // 3 + (1 if j < (n - n_full) % 3 else 0)
                    self._practice(pol, a, fam, nf, rng, "full", "school", age, acc=acc)
                    self._practice(pol, a, fam, nb, rng, "bandit", "school", age, acc=acc)
            # tutoring (adaptive: weakest measured training family, matched difficulty)
            if a.tutored and age >= 3:
                n = self._episodes(p.tutoring_episodes_per_tick, a)
                for iv in self.config.interventions:
                    if (
                        iv.kind == InterventionKind.TUTORING_ACCESS
                        and "episodes" in iv.params
                        and self._iv_eligible(iv, a, age)
                    ):
                        n = self._episodes(int(iv.params["episodes"]), a)
                fam = min(
                    fams,
                    key=lambda f: (
                        a.measured.get(f, {}).get("accuracy", 0.0),
                        (fams.index(f) + self.tick) % 3,
                    ),
                )
                self._practice(
                    pol,
                    a,
                    fam,
                    n,
                    rng,
                    "full",
                    "tutor",
                    age,
                    difficulty=self._difficulty_for(a, fam),
                    acc=acc,
                )
            act = a.activity
            if act == "study" and s["action"] is not None:
                self._study(a, pol, s, rng, age)
            elif act == "work":
                self._work(a, pol, s, rng, age)
            self._set_place(a, age)

    def _study(
        self, a: AgentRec, pol: Policy, s: dict[str, Any], rng: np.random.Generator, age: float
    ) -> None:
        fams = TRAINING_FAMILIES
        total = sum(a.practice_counts.get(f, 0) for f in fams) or 1
        ctx = np.array(
            [
                [
                    *(a.recent_reward.get(f, 0.0) for f in fams),
                    *(a.practice_counts.get(f, 0) / total for f in fams),
                    1.0,
                ]
            ],
            dtype=np.float32,
        )
        obs = Observation(a.id, self.tick, age, "practice", "practice", ctx, len(fams))
        dec = pol.decide(obs, self.streams.get("agent", a.id, "act"), explore=True)
        fam = fams[dec.action]
        if a.flags.get("adult_ed", False) and age >= 18:
            n = self._episodes(self.p.school_episodes_per_tick, a)
            st = self._practice(
                pol,
                a,
                fam,
                n,
                rng,
                "full",
                "school",
                age,
                difficulty=self._difficulty_for(a, fam),
                acc=s["acc"],
            )
            a.education_years += self.dt_years
        else:
            n = self._episodes(self.p.self_study_episodes_per_tick, a)
            if age < self.p.school_start_age:
                n = max(1, n // 2)
            st = self._practice(pol, a, fam, n, rng, "bandit", "self", age, acc=s["acc"])
        r = st.mean_reward if st is not None and st.n else 0.0
        s["learn"] = r
        pol.learn(obs, dec.action, Feedback(mode="bandit", reward=float(r), source="self"))

    def _work(
        self, a: AgentRec, pol: Policy, s: dict[str, Any], rng: np.random.Generator, age: float
    ) -> None:
        down = 0.7 if self.tick <= self._downturn_until else 1.0
        if a.job_id is not None:
            job = self._jobs[a.job_id]
            n = self._episodes(self.p.work_episodes_per_tick, a)
            st = self._practice(pol, a, job.family, n, rng, "bandit", "job", age, acc=s["acc"])
            accy = (st.correct / st.n) if st is not None and st.n else 0.0
            wage = self.p.wage_base * (0.5 + accy) * self.scale * down
            a.work_ticks_year += 1
        else:  # gig work while unemployed
            fam = TRAINING_FAMILIES[(self.tick + int(a.id[1:])) % 3]
            n = max(1, self._episodes(self.p.work_episodes_per_tick, a) // 2)
            st = self._practice(pol, a, fam, n, rng, "bandit", "job", age, acc=s["acc"])
            accy = (st.correct / st.n) if st is not None and st.n else 0.0
            wage = 0.3 * self.p.wage_base * (0.5 + accy) * self.scale * down
        s["income"] = wage
        a.income_year += wage
        hh = self._households.get(a.household_id)
        if hh is not None:
            hh.resources += wage

    def _set_place(self, a: AgentRec, age: float) -> None:
        home = self._households[a.household_id].home_place_id if a.household_id in self._households else None
        act = a.activity
        if act == "work":
            a.place_id = self._jobs[a.job_id].workplace_id if a.job_id else "work_market"
        elif act == "socialize":
            if a.flags.get("isolated", False):
                a.place_id = home
            else:
                a.place_id = "elder_centre" if age >= 65 else "plaza"
        elif a.enrolled:
            a.place_id = "tutoring_centre" if (a.tutored and self.tick % 2 == 0) else self._school_for(a)
        elif act == "study" and a.flags.get("adult_ed", False):
            a.place_id = self._school_for(a)
        else:
            a.place_id = home

    # ------------------------------------------------------------------ phase 4: social
    def _social_phase(self, scratch: dict[str, dict[str, Any]]) -> None:
        srng = self.streams.get("social", "matching")
        # (a) classmates and (b) coworkers: one contact per tick
        groups: dict[str, list[str]] = {}
        for aid in sorted(scratch):
            a = self._agents[aid]
            if a.flags.get("isolated", False):
                continue
            if a.enrolled:
                groups.setdefault("school:" + self._school_for(a), []).append(aid)
            elif a.activity == "work" and a.job_id is not None:
                groups.setdefault("work:" + self._jobs[a.job_id].workplace_id, []).append(aid)
        for gkey in sorted(groups):
            members = groups[gkey]
            if len(members) < 2:
                continue
            kind = "classmate" if gkey.startswith("school:") else "coworker"
            picks = srng.integers(0, len(members) - 1, size=len(members))
            for i, aid in enumerate(members):
                j = int(picks[i])
                other = members[j if j < i else j + 1]
                e = self._edge(aid, other, kind)
                e.interactions += 1
                e.trust += 0.05 * self.k * (0.6 - e.trust)
                self._edge(other, aid, kind)
        # (c) mentoring demonstrations for new hires
        for aid in sorted(scratch):
            a = self._agents[aid]
            if a.mentor_id and self.tick <= a.mentor_until_tick and a.activity == "work" and a.job_id:
                m = self._agents.get(a.mentor_id)
                if m is not None and m.alive and not a.flags.get("isolated", False):
                    self._demonstrate(
                        m.id,
                        a,
                        self._jobs[a.job_id].family,
                        self.streams.get("agent", aid, "learn"),
                        scratch[aid]["acc"],
                        "mentor",
                    )
        # (d) plaza / family meetings
        social = [aid for aid in sorted(scratch) if self._agents[aid].activity == "socialize"]
        pool = [aid for aid in social if not self._agents[aid].flags.get("isolated", False)]
        order = [pool[i] for i in srng.permutation(len(pool))] if pool else []
        matched: set[str] = set()
        pairs: list[tuple[str, str]] = []
        pool_set = set(pool)
        for aid in order:
            if aid in matched:
                continue
            friends = [
                b
                for b, e in sorted(self._edges.get(aid, {}).items())
                if e.kind == "friend" and b in pool_set and b not in matched and b != aid
            ]
            partner = None
            if friends and srng.random() < 0.5:
                partner = friends[int(srng.integers(0, len(friends)))]
            else:
                for b in order:
                    if b != aid and b not in matched:
                        partner = b
                        break
            if partner is None:
                continue
            matched.update((aid, partner))
            pairs.append((aid, partner) if aid < partner else (partner, aid))
        for aid in social:
            if self._agents[aid].flags.get("isolated", False):
                hh = self._households.get(self._agents[aid].household_id)
                fam = [m for m in (hh.members if hh else []) if m != aid and self._agents[m].alive]
                if fam:
                    pairs.append((aid, fam[int(srng.integers(0, len(fam)))]))
        for a_id, b_id in pairs:
            self._meeting(a_id, b_id, scratch)

    def _meeting(self, a_id: str, b_id: str, scratch: dict[str, dict[str, Any]]) -> None:
        a, b = self._agents[a_id], self._agents[b_id]
        for x in (a_id, b_id):
            if x in scratch:
                scratch[x]["met"] = True
        family = self._is_family(a_id, b_id)
        e_ab = self._edges[a_id].get(b_id) or self._edge(a_id, b_id, "acquaintance")
        e_ba = self._edges[b_id].get(a_id) or self._edge(b_id, a_id, "acquaintance")
        # trust game (stag hunt)
        acts: dict[str, int] = {}
        obss: dict[str, Observation] = {}
        for me, _other, e in ((a_id, b_id, e_ab), (b_id, a_id, e_ba)):
            ctx = np.array(
                [[e.trust, 1.0 if family else 0.0, 1.0 if e.kind == "friend" else 0.0, 1.0]], dtype=np.float32
            )
            obs = Observation(me, self.tick, self._age(self._agents[me]), "social", "trust_game", ctx, 2)
            obss[me] = obs
            acts[me] = int(
                self._policies[me].decide(obs, self.streams.get("agent", me, "act"), explore=True).action
            )
        for me, other, e in ((a_id, b_id, e_ab), (b_id, a_id, e_ba)):
            pay = TRUST_PAYOFF[(acts[me], acts[other])]
            self._policies[me].learn(obss[me], acts[me], Feedback(mode="bandit", reward=pay, source="peer"))
            e.trust = _clip01(e.trust + 0.2 * (acts[other] - e.trust))
            e.interactions += 1
        # trust-weighted peer demonstrations (isolated agents only learn from family)
        n_demo = round(self.p.peer_demonstrations_per_social_tick * self.k)
        for teacher, learner in ((a_id, b_id), (b_id, a_id)):
            L = self._agents[learner]
            if L.flags.get("isolated", False) and not family:
                continue
            rng = self.streams.get("agent", learner, "learn")
            acc = scratch[learner]["acc"] if learner in scratch else None
            for i in range(n_demo):
                fam = TRAINING_FAMILIES[(self.tick + int(learner[1:]) + i) % 3]
                self._demonstrate(teacher, L, fam, rng, acc, "peer")
        if (
            self.config.log.decision_sample_rate > 0
            and _hash_unit(int(a_id[1:]) * 7 + int(b_id[1:]), self.tick)
            < self.config.log.decision_sample_rate
        ):
            self._emit(
                EventType.SOCIAL_INTERACTION,
                a_id,
                {
                    "with": b_id,
                    "cooperate": [acts[a_id], acts[b_id]],
                    "trust": [_r(e_ab.trust, 4), _r(e_ba.trust, 4)],
                    "family": family,
                },
                mechanism="world.social.meeting",
                other_ids=[b_id],
            )
        # friendship
        if (
            not family
            and e_ab.kind != "friend"
            and min(e_ab.interactions, e_ba.interactions) >= 3
            and min(e_ab.trust, e_ba.trust) >= 0.6
        ):
            e_ab.kind = e_ba.kind = "friend"
            for rec in (a, b):
                rec.values["relationships"] = rec.values.get("relationships", 0.0) + 0.01
                rec.values = popmod.normalise_values(rec.values)
            self._emit(
                EventType.FRIENDSHIP_FORMED,
                a_id,
                {"with": b_id, "trust": [_r(e_ab.trust, 4), _r(e_ba.trust, 4)]},
                mechanism="world.social.friendship",
                other_ids=[b_id],
                milestone=True,
            )
        # partnership
        ages = (self._age(a), self._age(b))
        if (
            not family
            and a.partner_id is None
            and b.partner_id is None
            and all(20 <= x <= 45 for x in ages)
            and a.household_id != b.household_id
            and min(e_ab.trust, e_ba.trust) >= 0.6
            and min(e_ab.interactions, e_ba.interactions) >= 4
            and self.streams.get("agent", min(a_id, b_id), "family").random() < 0.25 * self.k
        ):
            self._form_partnership(a, b)

    def _partnering(self) -> None:
        """Background partner search: unpartnered adults 20-40 search with a small hazard and
        choose among unrelated, unpartnered adults within 8 years of age, weighting existing ties
        by trust. Social ties shape *who* partners, the hazard keeps demography policy-neutral."""
        hazard = 0.2 * self.dt_years
        taken: set[str] = set()
        alive = self._alive_ids()
        singles = [
            aid
            for aid in alive
            if self._agents[aid].partner_id is None and 20 <= self._age(self._agents[aid]) <= 45
        ]
        for aid in singles:
            a = self._agents[aid]
            if aid in taken or self._age(a) > 40:
                continue
            rng = self.streams.get("agent", aid, "family")
            if rng.random() >= hazard:
                continue
            age = self._age(a)
            cands = [
                b
                for b in singles
                if b != aid
                and b not in taken
                and abs(self._age(self._agents[b]) - age) <= 8
                and self._agents[b].household_id != a.household_id
                and not self._is_family(aid, b)
            ]
            if not cands:
                continue
            w = np.array(
                [1.0 + 3.0 * (self._edges[aid][b].trust if b in self._edges[aid] else 0.0) for b in cands]
            )
            b_id = cands[
                int(
                    np.searchsorted(np.cumsum(w), rng.random() * w.sum(), side="right").clip(
                        0, len(cands) - 1
                    )
                )
            ]
            taken.update((aid, b_id))
            self._form_partnership(a, self._agents[b_id])

    def _form_partnership(self, a: AgentRec, b: AgentRec) -> None:
        """A new couple founds a household; each partner's dependent children (< 18) move with them."""
        shares = 0.0
        movers: list[str] = [a.id, b.id]
        for x in (a, b):
            old = self._households.get(x.household_id)
            if old is None:
                continue
            deps = [c for c in sorted(old.members) if c in x.children and self._age(self._agents[c]) < 18]
            leaving = [x.id, *deps]
            share = max(0.0, old.resources) * len(leaving) / max(1, len(old.members))
            old.resources -= share
            shares += share
            for m in leaving:
                old.members.remove(m)
            if not old.members:
                shares += old.resources
                old.resources = 0.0
            movers.extend(deps)
        hh = self._new_household(movers, shares)
        for m in movers:
            self._agents[m].household_id = hh.id
        a.partner_id, b.partner_id = b.id, a.id
        self._link(a.id, b.id, "partner", trust=0.9)
        self._emit(
            EventType.PARTNERSHIP_FORMED,
            a.id,
            {"with": b.id, "household_id": hh.id, "dependents_moved": movers[2:]},
            mechanism="world.lifecycle.partnership",
            other_ids=[b.id],
            milestone=True,
        )

    # ------------------------------------------------------------------ phase 5: economy & affect
    def _economy_and_affect(self, scratch: dict[str, dict[str, Any]]) -> None:
        p = self.p
        k = self.k
        winter = ((self.day % 365) < 59) or ((self.day % 365) >= 334)
        shortfall: dict[str, float] = {}
        for hid in sorted(self._households):
            hh = self._households[hid]
            alive = [m for m in hh.members if self._agents[m].alive]
            need = 0.0
            for m in alive:
                ag = self._agents[m]
                age = self._age(ag)
                # children consume 60% of an adult share (assumption, ASSUMPTIONS.md §5)
                need += p.consumption_per_member * self.scale * (0.6 if age < 18 else 1.0)
                if age >= p.retirement_age:
                    hh.resources += 0.45 * p.wage_base * self.scale
                elif age >= 18 and ag.job_id is None and ag.activity != "work" and not ag.enrolled:
                    hh.resources += 0.25 * p.wage_base * self.scale
            hh.resources -= need
            if hh.resources < 0.0:  # no debt: unmet need becomes material stress
                shortfall[hid] = min(1.0, -hh.resources / need) if need > 0 else 0.0
                hh.resources = 0.0
        for aid in sorted(scratch):
            a = self._agents[aid]
            s = scratch[aid]
            act = a.activity
            if act == "rest":
                a.energy += 0.35 * k * (1 - a.energy) * (0.9 if winter else 1.0)
                a.stress -= 0.06 * k
            elif act == "study":
                a.energy -= 0.06 * k
                a.stress += 0.01 * k
            elif act == "work":
                a.energy -= 0.10 * k
                a.stress += 0.03 * k
            elif act == "socialize":
                a.energy -= 0.03 * k
                a.stress -= 0.05 * k
                a.social += (0.25 if s["met"] else 0.05) * k * (1 - a.social)
            if a.enrolled:
                a.energy -= 0.04 * k
            if a.job_id is not None and act != "work":
                a.absent_ticks_year += 1
            a.energy += 0.05 * k * (1 - a.energy)
            a.social -= 0.04 * k * a.social
            home = self._households.get(a.household_id)
            if home is not None and len(home.members) > 1:
                a.social += 0.02 * k
            a.stress += 0.05 * k * (0.2 - a.stress)
            per = (home.resources / max(1, len(home.members))) if home else 0.0
            if per < 1.0:
                a.stress += 0.05 * k
            a.stress += 0.1 * k * shortfall.get(a.household_id, 0.0)
            if self.tick <= a.illness_until_tick:
                a.energy -= 0.05 * k
            a.energy = _clip01(a.energy)
            a.stress = _clip01(a.stress)
            a.social = _clip01(a.social)
            if s["action"] is not None:
                r = self._activity_reward(a, s)
                a.motivation = 0.9 * a.motivation + 0.1 * _clip01((r + 1.0) / 3.0)
                if s["learn"] is not None and s["learn"] > 0.7:
                    a.values["learning"] = a.values.get("learning", 0.0) + 0.005
                    a.values = popmod.normalise_values(a.values)
                self._policies[aid].learn(
                    s["obs"], s["action"], Feedback(mode="bandit", reward=r, source="environment")
                )
            self._policies[aid].tick_lifecycle(s["age"], self.dt_years)
            acc = s["acc"]
            if self.config.log.log_learning_episodes and any(not k_.startswith("_") for k_ in acc):
                payload = {
                    "by_source": {
                        k_: {"n": v["n"], "correct": v["correct"], "reward": _r(v["reward"], 4)}
                        for k_, v in sorted(acc.items())
                        if not k_.startswith("_")
                    },
                    "families": dict(sorted(acc.get("_families", {}).items())),
                    "delta_norm": _r(acc.get("_delta", 0.0), 6),
                    "n_updates": int(acc.get("_updates", 0)),
                    "activity": a.activity,
                }
                self._emit(
                    EventType.LEARNING_EPISODE,
                    aid,
                    payload,
                    mechanism="world.learning.episodes",
                    source="policy",
                )

    def _activity_reward(self, a: AgentRec, s: dict[str, Any]) -> float:
        v = a.values
        act = a.activity
        income = s["income"] / max(1e-9, self.p.wage_base * self.scale)
        rel = (a.social - s["s0"]) * 4.0
        health = a.energy - 0.5
        stab = -a.stress
        learn = s["learn"] if s["learn"] is not None else 0.0
        leisure = 0.5 if act in ("rest", "socialize") else 0.0
        autonomy = 0.4 if act in ("study", "socialize") else (0.2 if act == "rest" else 0.0)
        r = (
            v.get("income", 0) * income
            + v.get("relationships", 0) * rel
            + v.get("health", 0) * health
            + v.get("stability", 0) * stab
            + v.get("learning", 0) * learn
            + v.get("time", 0) * leisure
            + v.get("autonomy", 0) * autonomy
        )
        return max(-2.0, min(3.0, 3.0 * float(r)))

    # ------------------------------------------------------------------ phase 6: shocks, death, birth
    def _shocks(self) -> None:
        rng = self.streams.get("shocks")
        n = int(rng.poisson(self.p.shock_rate_per_year * self.dt_years))
        for _ in range(n):
            u = float(rng.random())
            alive = self._alive_ids()
            if not alive:
                return
            if u < 0.5:
                v = self._agents[alive[int(rng.integers(0, len(alive)))]]
                v.energy = max(0.0, v.energy - 0.4)
                v.stress = min(1.0, v.stress + 0.3)
                v.illness_until_tick = self.tick + max(1, round(90 / self.dt_days))
                v.values["health"] = v.values.get("health", 0.0) + 0.03
                v.values = popmod.normalise_values(v.values)
                self._emit(
                    EventType.SHOCK,
                    v.id,
                    {"kind": "illness"},
                    mechanism="world.shock.illness",
                    milestone=True,
                )
            elif u < 0.75:
                hids = sorted(h for h, hh in self._households.items() if hh.members)
                if not hids:
                    continue
                hh = self._households[hids[int(rng.integers(0, len(hids)))]]
                loss = 0.5 * hh.resources if hh.resources > 0 else 2.0
                hh.resources -= loss
                for m in hh.members:
                    ag = self._agents[m]
                    ag.stress = min(1.0, ag.stress + 0.2)
                    ag.values["stability"] = ag.values.get("stability", 0.0) + 0.02
                    ag.values = popmod.normalise_values(ag.values)
                self._emit(
                    EventType.SHOCK,
                    None,
                    {"kind": "flood", "household_id": hh.id, "loss": _r(loss, 4)},
                    mechanism="world.shock.flood",
                    other_ids=sorted(hh.members),
                    milestone=True,
                )
            else:
                self._downturn_until = self.tick + self.ticks_per_year
                for aid in alive:
                    ag = self._agents[aid]
                    if ag.job_id is not None:
                        ag.stress = min(1.0, ag.stress + 0.1)
                        ag.values["stability"] = ag.values.get("stability", 0.0) + 0.01
                        ag.values = popmod.normalise_values(ag.values)
                self._emit(
                    EventType.SHOCK,
                    None,
                    {"kind": "downturn", "until_tick": self._downturn_until},
                    mechanism="world.shock.downturn",
                )

    def _mortality(self) -> None:
        p = self.p
        for aid in self._alive_ids():
            a = self._agents[aid]
            age = self._age(a)
            u = float(self.streams.get("agent", aid, "life").random())  # always drawn: common random numbers
            hazard = p.mortality_gompertz_a * math.exp(p.mortality_gompertz_b * age)
            if self.tick <= a.illness_until_tick:
                hazard *= 3.0
            prob = 1.0 - math.exp(-hazard * self.dt_years)
            if u < prob or age >= p.max_age_years:
                self._die(a, age)

    def _die(self, a: AgentRec, age: float) -> None:
        a.alive = False
        a.death_tick = self.tick
        self._counters["deaths"] += 1
        self._deaths_tick += 1
        if a.job_id is not None:
            self._end_job(a, "death")
        if a.partner_id and a.partner_id in self._agents:
            self._agents[a.partner_id].partner_id = None
        self._emit(
            EventType.DEATH,
            a.id,
            {"age": _r(age, 3), "cause": "modelled_hazard"},
            mechanism="world.lifecycle.mortality",
            milestone=True,
        )
        hh = self._households.get(a.household_id)
        if hh is not None and a.id in hh.members:
            hh.members.remove(a.id)
            if not hh.members:
                heirs = sorted(
                    {
                        self._agents[c].household_id
                        for c in a.children
                        if self._agents[c].alive and self._agents[c].household_id in self._households
                    }
                )
                amount = max(0.0, hh.resources)
                if heirs:
                    for h in heirs:
                        self._households[h].resources += amount / len(heirs)
                else:
                    self._commons += amount
                hh.resources = 0.0
                self._emit(
                    EventType.INHERITANCE,
                    a.id,
                    {"from_household": hh.id, "amount": _r(amount, 4), "to_households": heirs},
                    mechanism="world.lifecycle.inheritance",
                )

    def _births(self) -> None:
        p = self.p
        pop = len(self._alive_ids())
        crowd = float(np.clip(2.0 * (1.0 - pop / max(1, self.max_population)), 0.0, 1.5))
        if crowd <= 0:
            return
        rate = p.annual_birth_rate_per_couple * self.dt_years * crowd
        cross = [
            iv
            for iv in self.config.interventions
            if iv.kind == InterventionKind.CROSS_FOSTER and self._iv_active(iv)
        ]
        for hid in sorted(self._households):
            hh = self._households[hid]
            couple = None
            for m in sorted(hh.members):
                ag = self._agents[m]
                if ag.alive and ag.partner_id and ag.partner_id in hh.members and m < ag.partner_id:
                    couple = (m, ag.partner_id)
                    break
            if couple is None:
                continue
            pa, pb = self._agents[couple[0]], self._agents[couple[1]]
            if not all(20 <= self._age(x) <= 45 for x in (pa, pb)) or pa.n_children_born >= 4:
                continue
            if self.streams.get("agent", couple[0], "family").random() >= rate:
                continue
            self._birth(pa, pb, cross)

    def _birth(self, pa: AgentRec, pb: AgentRec, cross: list[Intervention]) -> None:
        aid = self._new_agent_id()
        rng = stream(self.config.seed, "birth", aid)
        traits = popmod.inherit_traits([pa.traits, pb.traits], rng)
        base = popmod.inherit_base_params([pa.base_params, pb.base_params], rng)
        values = popmod.inherit_values([pa.values, pb.values], rng)
        name = popmod.draw_name(rng)
        home_hh = pa.household_id
        inherited: dict[str, Any] = {
            "traits_from": [pa.id, pb.id],
            "base_params_from": [pa.id, pb.id],
            "values_from": [pa.id, pb.id],
            "biological_household": home_hh,
            "knowledge_inherited": False,
        }
        for iv in cross:
            if (
                iv.target.age_min <= 0.0
                and stream(self.config.seed, "intervention", iv.name, aid).random() < iv.target.fraction
            ):
                options = sorted(
                    h
                    for h, hh in self._households.items()
                    if h != home_hh
                    and any(
                        self._agents[m].alive and 20 <= self._age(self._agents[m]) <= 55 for m in hh.members
                    )
                )
                if options:
                    home_hh = options[
                        int(
                            stream(self.config.seed, "intervention", iv.name, aid, "dest").integers(
                                0, len(options)
                            )
                        )
                    ]
                    inherited["cross_fostered"] = True
                    inherited["intervention"] = iv.name
                break
        inherited["rearing_household"] = home_hh
        rec = AgentRec(
            id=aid,
            name=name,
            household_id=home_hh,
            birth_tick=self.tick,
            birth_cohort_year=math.floor(self.year),
            traits=traits,
            base_params=base,
            values=values,
            persona=popmod.persona_text(name, home_hh, traits, values, 0.0),
            seed=derive_seed(self.config.resolved_population_seed(), "policy", aid),
            inherited=inherited,
            parents=[pa.id, pb.id],
            energy=0.8,
            social=0.5,
        )
        self._register_agent(rec)
        self._households[home_hh].members.append(aid)
        pa.children.append(aid)
        pb.children.append(aid)
        pa.n_children_born += 1
        pb.n_children_born += 1
        if home_hh == pa.household_id:
            self._link(aid, pa.id, "child")
            self._link(aid, pb.id, "child")
            for sib in sorted(set(pa.children) | set(pb.children)):
                if sib != aid and self._agents[sib].alive:
                    self._link(aid, sib, "sibling")
        else:
            for m in sorted(self._households[home_hh].members):
                if m != aid and self._age(self._agents[m]) >= 18:
                    self._link(aid, m, "child")
        self._counters["births"] += 1
        self._births_tick += 1
        self._emit(
            EventType.BIRTH,
            aid,
            {"initial": False, "household_id": home_hh, "inherited": inherited, "persona": rec.persona},
            mechanism="world.lifecycle.birth",
            other_ids=[pa.id, pb.id],
            milestone=True,
        )

    # ------------------------------------------------------------------ phase 7: yearly
    def _assess_all(self) -> None:
        sw = self.symbol_world
        for aid in self._alive_ids():
            a = self._agents[aid]
            age = self._age(a)
            if age < 2:
                continue
            pol = self._policies[aid]
            fam_out: dict[str, Any] = {}
            for fam in ALL_FAMILIES:
                res = assess(pol, sw, fam, age, split="holdout", enable_gate=self.p.enable_age_gate)
                split = "transfer" if fam not in TRAINING_FAMILIES else "holdout"
                a.measured[fam] = {
                    "accuracy": res.accuracy,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "log_loss": res.log_loss,
                    "n_items": res.n_items,
                    "tick": self.tick,
                }
                self._skill_curves[aid].append(
                    {
                        "tick": self.tick,
                        "age_years": _r(age, 4),
                        "family": fam,
                        "accuracy": _r(res.accuracy, 6),
                        "ci_low": _r(res.ci_low, 6),
                        "ci_high": _r(res.ci_high, 6),
                        "log_loss": _r(res.log_loss, 6),
                        "n_items": res.n_items,
                        "split": split,
                    }
                )
                self._outcomes.append(
                    [aid, f"holdout_accuracy/{fam}", _r(res.accuracy, 6), self.tick, _r(age, 4), split]
                )
                self._outcomes.append(
                    [aid, f"holdout_log_loss/{fam}", _r(res.log_loss, 6), self.tick, _r(age, 4), split]
                )
                fam_out[fam] = {
                    "accuracy": _r(res.accuracy, 6),
                    "ci_low": _r(res.ci_low, 6),
                    "ci_high": _r(res.ci_high, 6),
                    "log_loss": _r(res.log_loss, 6),
                    "expected_accuracy": _r(res.expected_accuracy, 6),
                    "chance": _r(res.chance, 6),
                    "split": split,
                }
            self._emit(
                EventType.ASSESSMENT,
                aid,
                {
                    "age": _r(age, 4),
                    "families": fam_out,
                    "self_estimates": _jsonable(pol.self_estimates()),
                    "n_items": sw.config.eval_items_per_family,
                    "learning_disabled": True,
                },
                mechanism="tasks.assess.holdout",
                source="evaluation",
            )

    def _end_job(self, a: AgentRec, reason: str) -> None:
        job = self._jobs.get(a.job_id or "")
        if job is not None:
            job.holder = None
            job.start_tick = None
        self._emit(
            EventType.JOB_END,
            a.id,
            {"job_id": a.job_id, "reason": reason},
            mechanism=f"world.economy.job_end.{reason}",
            milestone=True,
        )
        a.job_id = None
        a.mentor_id = None

    def _job_market(self, initial: bool) -> None:
        p = self.p
        rng = self.streams.get("market")
        alive = self._alive_ids()
        down = self.tick <= self._downturn_until
        if not initial:
            for aid in alive:
                a = self._agents[aid]
                if a.job_id is None:
                    continue
                u = float(rng.random())
                if a.absent_ticks_year > 0.5 * self.ticks_per_year:
                    self._end_job(a, "absence")
                elif u < (0.12 if down else 0.04):
                    self._end_job(a, "layoff")
        target = max(1, round(p.n_jobs_per_100_agents * len(alive) / 100.0))
        jobs = sorted(self._jobs)
        if len(jobs) < target:
            for _ in range(target - len(jobs)):
                i = self._counters["job"]
                self._counters["job"] += 1
                fam = TRAINING_FAMILIES[i % 3]
                wp = next(w for w, f in geo.WORKPLACE_FAMILIES.items() if f == fam)
                self._jobs[f"j{i:04d}"] = Job(id=f"j{i:04d}", workplace_id=wp, family=fam)
        elif len(jobs) > target:
            vacant = [j for j in reversed(jobs) if self._jobs[j].holder is None]
            for j in vacant[: len(jobs) - target]:
                del self._jobs[j]
        vacancies = sorted(j for j, job in self._jobs.items() if job.holder is None)
        if not vacancies:
            return
        cands = [
            aid
            for aid in alive
            if self._agents[aid].job_id is None
            and 18 <= self._age(self._agents[aid]) < p.retirement_age
            and not self._agents[aid].enrolled
        ]
        noise = {aid: float(rng.random()) * 1e-3 for aid in cands}
        hired: set[str] = set()
        for jid in vacancies:
            job = self._jobs[jid]
            best = None
            best_score = -1.0
            for aid in cands:
                if aid in hired:
                    continue
                sc = self._agents[aid].measured.get(job.family, {}).get("accuracy", 0.0) + noise[aid]
                if sc > best_score:
                    best, best_score = aid, sc
            if best is None:
                break
            hired.add(best)
            a = self._agents[best]
            job.holder, job.start_tick = best, self.tick
            a.job_id = jid
            self._emit(
                EventType.JOB_START,
                best,
                {
                    "job_id": jid,
                    "family": job.family,
                    "workplace": job.workplace_id,
                    "interview_accuracy": _r(best_score, 4),
                },
                mechanism="world.economy.hire",
                milestone=True,
            )
            mentors = [
                j.holder
                for j in self._jobs.values()
                if j.family == job.family and j.holder and j.holder != best and self._agents[j.holder].alive
            ]
            if mentors:
                m = max(
                    sorted(mentors),
                    key=lambda x: self._agents[x].measured.get(job.family, {}).get("accuracy", 0.0),
                )
                a.mentor_id = m
                a.mentor_until_tick = self.tick + max(1, round(182 / self.dt_days))
                self._link(best, m, "mentor")
                self._emit(
                    EventType.MENTORSHIP,
                    best,
                    {"mentor": m, "family": job.family},
                    mechanism="world.economy.mentor",
                    other_ids=[m],
                    milestone=True,
                )

    def _yearly_records(self) -> None:
        for aid in self._alive_ids():
            a = self._agents[aid]
            age = self._age(a)
            if a.income_year > 0:
                self._emit(
                    EventType.WAGE,
                    aid,
                    {"income_year": _r(a.income_year, 4), "job_id": a.job_id},
                    mechanism="world.economy.wage",
                )
            hh = self._households.get(a.household_id)
            per = (hh.resources / max(1, len(hh.members))) if hh else 0.0
            if age >= 18:
                self._outcomes.append(
                    [aid, "employed", 1.0 if a.job_id else 0.0, self.tick, _r(age, 4), "n/a"]
                )
                self._outcomes.append(
                    [aid, "income_year", _r(a.income_year, 4), self.tick, _r(age, 4), "n/a"]
                )
            self._outcomes.append(
                [aid, "education_years", _r(a.education_years, 4), self.tick, _r(age, 4), "n/a"]
            )
            self._outcomes.append(
                [aid, "household_resources_per_member", _r(per, 4), self.tick, _r(age, 4), "n/a"]
            )
            a.income_last_year = a.income_year
            a.income_year = 0.0
            a.work_ticks_year = 0
            a.absent_ticks_year = 0

    def _param_snapshot(self) -> None:
        for aid in self._alive_ids():
            pol = self._policies[aid]
            summ = _jsonable(pol.parameter_summary())
            self._param_history[aid].append(
                {"tick": self.tick, "age_years": _r(self._age(self._agents[aid]), 4), "summary": summ}
            )
            self._emit(
                EventType.POLICY_UPDATE,
                aid,
                {"parameter_summary": summ},
                mechanism="world.policy.summary",
                source="policy",
            )

    # ================================================================== observability
    def _make_tick(self, wall_ms: float, include_frames: bool) -> WorldTick:
        alive = self._alive_ids()
        accs: dict[str, list[float]] = {f: [] for f in ALL_FAMILIES}
        hist: dict[str, int] = {}
        working = employed = school_age = enrolled = 0
        for aid in alive:
            a = self._agents[aid]
            age = self._age(a)
            for f, m in a.measured.items():
                accs[f].append(m["accuracy"])
            dec = min(9, int(age // 10))
            key = f"{dec * 10}-{dec * 10 + 9}" if dec < 9 else "90+"
            hist[key] = hist.get(key, 0) + 1
            if 18 <= age < self.p.retirement_age and not a.enrolled:
                working += 1
                employed += int(a.job_id is not None)
            if self.p.school_start_age <= age < self.p.school_end_age:
                school_age += 1
                enrolled += int(a.enrolled)
        res = np.array([hh.resources for hh in self._households.values() if hh.members], dtype=float)
        return WorldTick(
            run_id=self.run_id,
            tick=self.tick,
            day=self.day,
            year=_r(self.year, 6),
            population=len(alive),
            births=self._births_tick,
            deaths=self._deaths_tick,
            mean_measured_accuracy={f: _r(float(np.mean(v)), 6) for f, v in accs.items() if v},
            employment_rate=_r(employed / working, 6) if working else 0.0,
            school_enrolment_rate=_r(enrolled / school_age, 6) if school_age else 0.0,
            resource_gini=_r(_gini(res), 6),
            mean_resources=_r(float(res.mean()), 6) if res.size else 0.0,
            events_emitted=self._tick_events,
            age_histogram=dict(sorted(hist.items())),
            wall_time_ms=_r(wall_ms, 3),
            frames=self.frames() if include_frames else None,
        )

    def frames(self) -> list[AgentFrame]:
        self._ensure_initialized()
        out = []
        for aid in self._alive_ids():
            a = self._agents[aid]
            pl = self._places.get(a.place_id or "") or self._places.get(
                self._households[a.household_id].home_place_id
            )
            x, y = (pl.x, pl.y) if pl else (0.0, 0.0)
            radius = 1.2 if (pl is not None and pl.kind == "home") else 3.5
            jx, jy = geo.jitter(int(aid[1:]), self.tick, radius)
            out.append(
                AgentFrame(
                    id=aid,
                    x=_r(float(np.clip(x + jx, 0, geo.WIDTH)), 3),
                    y=_r(float(np.clip(y + jy, 0, geo.HEIGHT)), 3),
                    place_id=a.place_id,
                    activity=a.activity,
                    age=_r(self._age(a), 3),
                    alive=True,
                    household_id=a.household_id,
                )
            )
        return out

    def world_map(self) -> WorldMap:
        self._ensure_initialized()
        return geo.build_map(self._places)

    @property
    def ticks_history(self) -> list[WorldTick]:
        self._ensure_initialized()
        return list(self._ticks)

    # ------------------------------------------------------------------ agent views
    def agent(self, agent_id: str) -> AgentState:
        self._ensure_initialized()
        a = self._agents[agent_id]
        pol = self._policies[agent_id]
        age = self._age(a, a.death_tick if not a.alive and a.death_tick is not None else None)
        hh = self._households.get(a.household_id)
        selfest = pol.self_estimates()
        caps: dict[str, CapabilityState] = {}
        for fam in ALL_FAMILIES:
            m = a.measured.get(fam, {})
            se = selfest.get(fam, {}) if isinstance(selfest, dict) else {}
            caps[fam] = CapabilityState(
                family=fam,
                measured_accuracy=m.get("accuracy"),
                measured_ci_low=m.get("ci_low"),
                measured_ci_high=m.get("ci_high"),
                measured_log_loss=m.get("log_loss"),
                n_assessment_items=int(m.get("n_items", 0)),
                last_assessed_tick=m.get("tick"),
                self_estimate_mean=se.get("mean") if isinstance(se, dict) else None,
                self_estimate_sd=se.get("sd") if isinstance(se, dict) else None,
                practice_episodes=a.practice_counts.get(fam, 0),
                last_practice_tick=a.last_practice_tick.get(fam),
            )
        frame_x = frame_y = 0.0
        pl = self._places.get(a.place_id or "")
        if pl is not None:
            frame_x, frame_y = pl.x, pl.y
        summ = _jsonable(pol.parameter_summary())
        summ.setdefault("implementation_class", getattr(pol, "policy_class", "unknown"))
        return AgentState(
            agent_id=a.id,
            run_id=self.run_id,
            household_id=a.household_id,
            birth_tick=a.birth_tick,
            birth_cohort_year=a.birth_cohort_year,
            age_years=_r(age, 4),
            stage=LifeStage(a.stage),
            alive=a.alive,
            death_tick=a.death_tick,
            parents=list(a.parents),
            children=list(a.children),
            partner_id=a.partner_id,
            traits={k: _r(v, 4) for k, v in a.traits.items()},
            persona=a.persona,
            resources=_r(a.income_last_year, 4),
            household_resources=_r(hh.resources, 4) if hh else 0.0,
            place_id=a.place_id,
            x=frame_x,
            y=frame_y,
            activity=a.activity,
            job_id=a.job_id,
            education_years=_r(a.education_years, 4),
            enrolled_school=a.enrolled,
            tutored=a.tutored,
            affect=AffectProxies(
                stress=_r(a.stress, 4),
                energy=_r(a.energy, 4),
                social_connection=_r(a.social, 4),
                motivation=_r(a.motivation, 4),
            ),
            values={k: _r(v, 4) for k, v in a.values.items()},
            capabilities=caps,
            policy=PolicyDescriptor(
                policy_class=self.config.policy_class,
                version=str(getattr(pol, "version", "")),
                parameter_summary=summ,
            ),
            memory_items=int(pol.compute.memory_items) if hasattr(pol, "compute") else 0,
            n_friends=sum(1 for e in self._edges.get(a.id, {}).values() if e.kind == "friend"),
        )

    def agents(self, *, include_dead: bool = True) -> list[AgentState]:
        self._ensure_initialized()
        return [self.agent(aid) for aid in sorted(self._agents) if include_dead or self._agents[aid].alive]

    def agent_ids(self, *, include_dead: bool = True) -> list[str]:
        self._ensure_initialized()
        return [aid for aid in sorted(self._agents) if include_dead or self._agents[aid].alive]

    def policy_of(self, agent_id: str) -> Policy:
        self._ensure_initialized()
        return self._policies[agent_id]

    def edges(self, agent_id: str | None = None) -> list[SocialEdge]:
        self._ensure_initialized()
        srcs = [agent_id] if agent_id is not None else sorted(self._edges)
        out = []
        for s in srcs:
            for d, e in sorted(self._edges.get(s, {}).items()):
                if e.kind not in _PUBLIC_EDGE_KINDS:
                    continue  # internal "acquaintance" ties have no contract enum value yet
                out.append(
                    SocialEdge(
                        src=s,
                        dst=d,
                        kind=SocialEdgeKind(e.kind),
                        trust=_r(float(np.clip(e.trust, 0, 1)), 4),
                        interactions=e.interactions,
                        formed_tick=max(0, e.formed_tick),
                    )
                )
        return out

    def timeline(self, agent_id: str) -> list[ExperienceEvent]:
        self._ensure_initialized()
        return [ExperienceEvent.model_validate(d) for d in self._timelines.get(agent_id, [])]

    def skill_curve(self, agent_id: str) -> list[dict[str, Any]]:
        self._ensure_initialized()
        return list(self._skill_curves.get(agent_id, []))

    def parameter_history(self, agent_id: str) -> list[dict[str, Any]]:
        self._ensure_initialized()
        return list(self._param_history.get(agent_id, []))

    def latest_explanation(self, agent_id: str) -> dict[str, Any] | None:
        e = self._last_explanation.get(agent_id)
        return _jsonable(e) if e is not None else None

    def outcomes(self) -> list[OutcomeObservation]:
        self._ensure_initialized()
        return [
            OutcomeObservation(
                run_id=self.run_id,
                seed=self.config.seed,
                agent_id=o[0],
                metric=o[1],
                value=o[2],
                tick=o[3],
                age_years=o[4],
                split=o[5],
                unit="accuracy" if o[1].startswith("holdout_accuracy") else "",
            )
            for o in self._outcomes
        ]

    def outcome_rows(self) -> list[list[Any]]:
        """Compact outcome rows ``[agent_id, metric, value, tick, age_years, split]`` (fast path)."""
        self._ensure_initialized()
        return [list(o) for o in self._outcomes]

    def manifest(self) -> RunManifest:
        status = (
            JobStatus.QUEUED
            if not self._initialized
            else (JobStatus.COMPLETED if self.done else JobStatus.RUNNING)
        )
        pol_version = None
        if self._policies:
            first = self._policies[sorted(self._policies)[0]]
            pol_version = str(getattr(first, "version", None))
        return RunManifest(
            run_id=self.run_id,
            status=status,
            code_version=_code_version(),
            config=self.config,
            config_hash=self.config_hash,
            policy_class=self.config.policy_class,
            policy_version=pol_version,
            task_world_hash=self.task_world_hash,
            dataset_hashes={"event_stream_sha256_chain": self._hash},
            n_events=self._counters["seq"],
            final_state_hash=self.state_hash() if self._initialized else None,
            compute=ComputeReport(
                wall_time_s=_r(self._compute["wall_time_s"], 4),
                cpu_time_s=_r(self._compute["cpu_time_s"], 4),
                peak_rss_mb=_r(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2),
                ticks=self.tick,
                agent_years=_r(self._compute["agent_years"], 4),
                events=self._counters["seq"],
                llm_calls=0,
                llm_tokens=0,
                estimated_cost_usd=0.0,
                hardware=_hardware(),
            ),
            nondeterminism=[] if not self._truncated else ["run truncated at max_events"],
        )

    def state_hash(self) -> str:
        """Hash of the simulation state (agents, households, jobs, edges, policies)."""
        return stable_hash(
            {
                "tick": self.tick,
                "agents": [self._agents[a].to_dict() for a in sorted(self._agents)],
                "households": [self._households[h].to_dict() for h in sorted(self._households)],
                "jobs": [self._jobs[j].to_dict() for j in sorted(self._jobs)],
                "edges": {
                    s: {d: e.to_list() for d, e in sorted(v.items())} for s, v in sorted(self._edges.items())
                },
                "policies": {
                    a: {k: v for k, v in self._policies[a].state_dict().items() if k != "compute"}
                    for a in sorted(self._policies)
                },
            },
            length=32,
        )

    # ================================================================== snapshots
    def snapshot(self) -> dict[str, Any]:
        self._ensure_initialized()
        return {
            "format": SNAPSHOT_FORMAT,
            "world_version": WORLD_VERSION,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "config": self.config.model_dump(mode="json"),
            "tick": self.tick,
            "truncated": self._truncated,
            "ended": bool(getattr(self, "_ended", False)),
            "agents": [self._agents[a].to_dict() for a in sorted(self._agents)],
            "init_specs": {a: self._init_specs[a] for a in sorted(self._init_specs)},
            "policies": {a: self._policies[a].state_dict() for a in sorted(self._policies)},
            "households": [self._households[h].to_dict() for h in sorted(self._households)],
            "places": [self._places[p].model_dump(mode="json") for p in sorted(self._places)],
            "jobs": [self._jobs[j].to_dict() for j in sorted(self._jobs)],
            "edges": {
                s: {d: e.to_list() for d, e in sorted(v.items())} for s, v in sorted(self._edges.items())
            },
            "counters": dict(self._counters),
            "hash": self._hash,
            "streams": self.streams.state_dict(),
            "downturn_until": self._downturn_until,
            "commons": self._commons,
            "ticks": [t.model_dump(mode="json") for t in self._ticks],
            "outcomes": [list(o) for o in self._outcomes],
            "timelines": self._timelines,
            "skill_curves": self._skill_curves,
            "param_history": self._param_history,
            "last_explanation": _jsonable(self._last_explanation),
            "compute": dict(self._compute),
        }

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("format") != SNAPSHOT_FORMAT:
            raise ValueError(f"unsupported snapshot format {snapshot.get('format')!r}")
        if snapshot["config_hash"] != self.config_hash:
            raise ValueError("snapshot was produced with a different SimulationConfig")
        self._reset_state()
        self._initialized = True
        self.run_id = snapshot["run_id"]
        self.tick = int(snapshot["tick"])
        self._truncated = bool(snapshot["truncated"])
        if snapshot.get("ended"):
            self._ended = True
        for d in snapshot["agents"]:
            rec = AgentRec.from_dict(d)
            self._agents[rec.id] = rec
        self._init_specs = {a: dict(v) for a, v in snapshot["init_specs"].items()}
        for aid in sorted(self._agents):
            pol = self._make_policy(aid)
            pol.load_state_dict(snapshot["policies"][aid])
            self._policies[aid] = pol
        self._households = {d["id"]: Household.from_dict(d) for d in snapshot["households"]}
        self._places = {d["place_id"]: Place.model_validate(d) for d in snapshot["places"]}
        self._jobs = {d["id"]: Job.from_dict(d) for d in snapshot["jobs"]}
        self._edges = {
            s: {d: EdgeRec.from_list(v) for d, v in m.items()} for s, m in snapshot["edges"].items()
        }
        for aid in self._agents:
            self._edges.setdefault(aid, {})
        self._counters = {k: int(v) for k, v in snapshot["counters"].items()}
        self._hash = snapshot["hash"]
        self.streams.load_state_dict(snapshot["streams"])
        self._downturn_until = int(snapshot["downturn_until"])
        self._commons = float(snapshot["commons"])
        self._ticks = [WorldTick.model_validate(t) for t in snapshot["ticks"]]
        self._outcomes = [list(o) for o in snapshot["outcomes"]]
        self._timelines = {k: list(v) for k, v in snapshot["timelines"].items()}
        self._skill_curves = {k: list(v) for k, v in snapshot["skill_curves"].items()}
        self._param_history = {k: list(v) for k, v in snapshot["param_history"].items()}
        self._last_explanation = dict(snapshot["last_explanation"])
        self._compute = {k: float(v) for k, v in snapshot["compute"].items()}
