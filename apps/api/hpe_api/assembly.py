"""Assemble API read models (AgentSummary, AgentDetail) from run views and the event store.

Skill curves come ONLY from ASSESSMENT events (held-out measurement with learning disabled).
Parameter history comes from POLICY_UPDATE events. Nothing here invents values: if an event
type is absent, the corresponding list is empty.
"""

from __future__ import annotations

from typing import Any

from hpe.contracts.api import AgentDetail, AgentSummary, ParameterPoint, SkillPoint
from hpe.contracts.schemas import AgentState, EventType, ExperienceEvent
from hpe.storage import EventRepository
from hpe.tasks import TRAINING_FAMILIES, TRANSFER_FAMILIES, TaskWorldConfig
from hpe_api.run_service import RunView

MILESTONE_TYPES: tuple[str, ...] = tuple(
    t.value
    for t in (
        EventType.BIRTH,
        EventType.DEATH,
        EventType.STAGE_CHANGE,
        EventType.SCHOOL_ENROLL,
        EventType.SCHOOL_GRADUATE,
        EventType.JOB_START,
        EventType.JOB_END,
        EventType.FRIENDSHIP_FORMED,
        EventType.PARTNERSHIP_FORMED,
        EventType.MENTORSHIP,
        EventType.SHOCK,
        EventType.INTERVENTION,
        EventType.INHERITANCE,
    )
)
MAX_TIMELINE = 400
MAX_PARAMETER_POINTS = 200


def mean_measured_accuracy(state: AgentState) -> float | None:
    vals = [
        c.measured_accuracy
        for fam, c in state.capabilities.items()
        if fam in TRAINING_FAMILIES and c.measured_accuracy is not None
    ]
    return float(sum(vals) / len(vals)) if vals else None


def agent_summary(state: AgentState) -> AgentSummary:
    return AgentSummary(
        agent_id=state.agent_id,
        household_id=state.household_id,
        age_years=state.age_years,
        stage=str(getattr(state.stage, "value", state.stage)),
        alive=state.alive,
        activity=state.activity,
        job_id=state.job_id,
        education_years=state.education_years,
        resources=state.resources,
        mean_measured_accuracy=mean_measured_accuracy(state),
        n_friends=state.n_friends,
    )


def _age_at(tick: int, state: AgentState, dt_days: int) -> float:
    return max(0.0, (tick - state.birth_tick) * dt_days / 365.0)


# World assessments use the fixed held-out set of TaskWorldConfig.eval_items_per_family items.
DEFAULT_EVAL_ITEMS = TaskWorldConfig().eval_items_per_family


def _assessment_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept the shapes a world may emit: one result, a list of results, or {family: result}."""
    for key in ("results", "assessments", "families"):
        val = payload.get(key)
        if isinstance(val, list):
            return [v for v in val if isinstance(v, dict)]
        if isinstance(val, dict):
            return [{"family": k, **v} for k, v in val.items() if isinstance(v, dict)]
    return [payload] if "accuracy" in payload else []


def skill_points(events: list[ExperienceEvent], state: AgentState, dt_days: int) -> list[SkillPoint]:
    out: list[SkillPoint] = []
    for ev in events:
        payload = ev.payload or {}
        for item in _assessment_items(payload):
            fam = item.get("family")
            if not fam or item.get("accuracy") is None:
                continue
            acc = float(item["accuracy"])
            age = item.get("age_years", payload.get("age_years", payload.get("age")))
            split = "transfer" if (fam in TRANSFER_FAMILIES or item.get("split") == "transfer") else "holdout"
            ll = item.get("log_loss")
            out.append(
                SkillPoint(
                    tick=ev.tick,
                    age_years=float(age) if age is not None else _age_at(ev.tick, state, dt_days),
                    family=str(fam),
                    accuracy=acc,
                    ci_low=float(item.get("ci_low", acc)),
                    ci_high=float(item.get("ci_high", acc)),
                    log_loss=float(ll) if ll is not None else None,
                    n_items=int(item.get("n_items", payload.get("n_items", DEFAULT_EVAL_ITEMS))),
                    split=split,  # type: ignore[arg-type]
                )
            )
    return out


def _summary_of(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("parameter_summary", "summary"):
        val = payload.get(key)
        if isinstance(val, dict):
            return val
    return payload


def parameter_points(events: list[ExperienceEvent], state: AgentState, dt_days: int) -> list[ParameterPoint]:
    if len(events) > MAX_PARAMETER_POINTS:
        step = len(events) / MAX_PARAMETER_POINTS
        events = [events[int(i * step)] for i in range(MAX_PARAMETER_POINTS - 1)] + [events[-1]]
    out = []
    for ev in events:
        payload = ev.payload or {}
        age = payload.get("age_years")
        out.append(
            ParameterPoint(
                tick=ev.tick,
                age_years=float(age) if age is not None else _age_at(ev.tick, state, dt_days),
                summary=_summary_of(payload),
            )
        )
    return out


def timeline(events_repo: EventRepository, run_id: str, agent_id: str) -> list[ExperienceEvent]:
    milestones, _ = events_repo.query(
        run_id, agent_id=agent_id, types=list(MILESTONE_TYPES), limit=MAX_TIMELINE
    )
    # formative learning: the first recorded learning episode per task family, and first tutoring session
    learning, _ = events_repo.query(
        run_id, agent_id=agent_id, types=[EventType.LEARNING_EPISODE.value], limit=400
    )
    first_by_family: dict[str, ExperienceEvent] = {}
    for ev in learning:
        fam = str((ev.payload or {}).get("family", "?"))
        first_by_family.setdefault(fam, ev)
    tutoring, _ = events_repo.query(
        run_id, agent_id=agent_id, types=[EventType.TUTORING_SESSION.value], limit=1
    )
    wages, _ = events_repo.query(run_id, agent_id=agent_id, types=[EventType.WAGE.value], limit=1)
    decisions, _ = events_repo.query(
        run_id, agent_id=agent_id, types=[EventType.DECISION.value], limit=20, descending=True
    )
    merged = {ev.seq: ev for ev in [*milestones, *first_by_family.values(), *tutoring, *wages, *decisions]}
    return [merged[k] for k in sorted(merged)][:MAX_TIMELINE]


def agent_detail(view: RunView, events_repo: EventRepository, agent_id: str) -> AgentDetail:
    state = view.agent(agent_id)
    run_id = view.run_id
    dt = view.dt_days()
    assessments, _ = events_repo.query(
        run_id, agent_id=agent_id, types=[EventType.ASSESSMENT.value], limit=5000
    )
    updates, _ = events_repo.query(
        run_id, agent_id=agent_id, types=[EventType.POLICY_UPDATE.value], limit=5000
    )
    last_decision, _ = events_repo.query(
        run_id, agent_id=agent_id, types=[EventType.DECISION.value], limit=1, descending=True
    )
    explanation: dict[str, Any] | None = None
    if last_decision:
        p = last_decision[0].payload or {}
        expl = p.get("explanation")
        explanation = {**({"tick": last_decision[0].tick}), **(expl if isinstance(expl, dict) else p)}
    memories = view.recent_memories(agent_id)
    if memories is None:
        memories = []
        for ev in reversed(updates):
            mem = _summary_of(ev.payload or {}).get("recent_memories")
            if isinstance(mem, list):
                memories = mem
                break
    return AgentDetail(
        state=state,
        timeline=timeline(events_repo, run_id, agent_id),
        skill_curve=skill_points(assessments, state, dt),
        parameter_history=parameter_points(updates, state, dt),
        relationships=view.edges(agent_id),
        recent_memories=[m for m in memories if isinstance(m, dict)][:50],
        latest_decision_explanation=explanation,
    )
