"""Template scenario engine for the Future Self experience (Tier 5).

Produces 2–4 *illustrative scenarios* for a user's decision, each conditioned on explicit
assumptions, with qualitative trade-offs against the user's own value weights, deterministic
calculations from the user's own numbers, explicit "insufficient evidence to predict" entries,
and small reversible experiments. It never outputs probabilities of success or happiness, never
recommends an option, and never consumes synthetic-society results (ADR-0005; this module must
not import ``hpe.world``, ``hpe.development`` or ``hpe.evaluation``).
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from hpe.contracts.schemas import (
    Calculation,
    DecisionOption,
    DecisionRequest,
    EvidenceRecord,
    EvidenceSourceType,
    ReversibleExperiment,
    Scenario,
    ScenarioResponse,
    Tradeoff,
    ValidationStatus,
    ValueWeight,
)
from hpe.core.hashing import stable_hash
from hpe.futures import calculators as calc
from hpe.futures.evidence import DEFAULT_REGISTRY, EvidenceRegistry
from hpe.futures.narration import NarrationGate
from hpe.futures.safety import classify
from hpe.futures.validator import ForbiddenLanguageError, check_scenario


class ConsentRequiredError(ValueError):
    pass


Direction = str  # "likely_better" | "likely_worse" | "mixed" | "unclear"

_PRIORS: dict[str, dict[str, tuple[Direction, str, tuple[str, ...]]]] = {
    "current_path": {
        "stability": (
            "likely_better",
            "Keeps your known role, team and income stream, assuming conditions stay similar.",
            ("assume-status-quo-known",),
        ),
        "income": (
            "unclear",
            "Depends on raises and conditions on your current path; no change is assumed here.",
            ("assume-status-quo-known",),
        ),
        "learning": (
            "mixed",
            "Learning continues in a familiar context, with fewer new challenges than a change would bring.",
            ("assume-status-quo-known",),
        ),
        "relationships": (
            "likely_better",
            "Existing work relationships are kept.",
            ("assume-status-quo-known",),
        ),
    },
    "new_job": {
        "stability": (
            "mixed",
            "A new employer resets known quantities such as team and track record, though it may offer its own stability.",
            ("assume-new-role-stability",),
        ),
        "learning": (
            "likely_better",
            "A new role usually involves learning new tools, people and expectations.",
            ("assume-new-role-learning-curve",),
        ),
        "relationships": (
            "mixed",
            "You would meet new colleagues, while some current work relationships may fade.",
            ("assume-new-role-stability",),
        ),
    },
    "start_project": {
        "autonomy": (
            "likely_better",
            "Running your own project usually increases control over what you work on, along with responsibility.",
            ("assume-project-autonomy",),
        ),
        "income": (
            "mixed",
            "Near-term income is typically irregular and may be zero for a while; longer-term income cannot be predicted.",
            ("assume-project-income-variable",),
        ),
        "stability": (
            "likely_worse",
            "Irregular income and unknown demand make life less predictable.",
            ("assume-project-income-variable",),
        ),
        "learning": (
            "likely_better",
            "A new venture exposes you to many unfamiliar tasks.",
            ("assume-project-autonomy",),
        ),
        "time": (
            "mixed",
            "Often demands long or irregular hours, although you set the schedule.",
            ("assume-project-autonomy",),
        ),
        "relationships": (
            "mixed",
            "Can create new professional ties while competing for time with existing ones.",
            ("assume-skill-time-cost",),
        ),
    },
    "learn_skill": {
        "learning": (
            "likely_better",
            "A direct, deliberate investment in a new skill.",
            ("assume-skill-time-cost",),
        ),
        "time": (
            "likely_worse",
            "Requires sustained time each week alongside your other commitments.",
            ("assume-skill-time-cost",),
        ),
        "income": (
            "unclear",
            "Any income payoff depends on factors this tool cannot observe.",
            ("assume-skill-payoff-unknown",),
        ),
        "stability": (
            "likely_better",
            "Can usually be pursued while keeping your current income.",
            ("assume-status-quo-known",),
        ),
        "autonomy": (
            "mixed",
            "Broadens future options but does not change your current role by itself.",
            ("assume-skill-payoff-unknown",),
        ),
    },
    "education": {
        "learning": (
            "likely_better",
            "Structured study with feedback on a new subject.",
            ("assume-skill-time-cost",),
        ),
        "time": ("likely_worse", "Formal study takes substantial weekly time.", ("assume-skill-time-cost",)),
        "income": (
            "unclear",
            "Any income payoff depends on factors this tool cannot observe.",
            ("assume-skill-payoff-unknown",),
        ),
        "stability": (
            "mixed",
            "May require reduced working hours or a period with less income.",
            ("assume-skill-time-cost",),
        ),
    },
    "other": {},
}

_KIND_SENTENCE = {
    "current_path": "You stayed on your current path and kept building on what you already know.",
    "new_job": "You moved to the new role and spent the first months getting to know new people and expectations.",
    "start_project": "You started the project and spent much of your time testing ideas with real people, with income that was irregular at first.",
    "learn_skill": "You set aside regular time each week to learn the new skill alongside your other commitments.",
    "education": "You enrolled in the programme and reorganised your weeks around study.",
    "other": "You took this path; this tool has few specifics about what it involves, so most of the picture is open.",
}

_KIND_UNCERTAINTY = {
    "current_path": "Whether conditions on your current path stay similar.",
    "new_job": "What the new team, manager and day-to-day work are actually like.",
    "start_project": "Whether people would want or pay for what you build.",
    "learn_skill": "Whether the skill is valued where you want to use it.",
    "education": "Whether the qualification is valued where you want to use it.",
    "other": "Most consequences of this path, because the tool knows little about it.",
}

_KIND_CANNOT_PREDICT = {
    "new_job": "Insufficient evidence to predict long-term career outcomes of this specific role.",
    "start_project": "Insufficient evidence to predict whether the venture would succeed.",
    "learn_skill": "Insufficient evidence to predict the income payoff of this skill for you.",
    "education": "Insufficient evidence to predict the income payoff of this programme for you.",
}

_EXPERIMENTS: dict[str, tuple[tuple[str, str, int, str, tuple[str, ...]], ...]] = {
    "current_path": (
        (
            "Ask about growth on your current path",
            "Have a conversation with your manager or a mentor about the next step available where you are.",
            14,
            "free",
            ("learning", "income"),
        ),
        (
            "Track your week",
            "Note your energy and satisfaction once a week for a month to see what your current path feels like in practice.",
            30,
            "free",
            ("health", "time"),
        ),
    ),
    "new_job": (
        (
            "Talk to two people in the role",
            "Ask two people who do this job (or worked at this employer) what a normal week is like.",
            7,
            "free",
            ("learning", "relationships"),
        ),
        (
            "Ask for a realistic preview",
            "Request to shadow the team, or ask for a written description of hours, expectations and growth.",
            14,
            "free",
            ("time", "stability"),
        ),
        (
            "Compare the offers in writing",
            "Write down total pay, hours, commute and flexibility side by side.",
            3,
            "free",
            ("income", "time"),
        ),
    ),
    "start_project": (
        (
            "Run five customer conversations",
            "Talk to five people who might use what you would build, before building much.",
            14,
            "free",
            ("income", "autonomy"),
        ),
        (
            "Budget stress test",
            "Live for one month on the income you would expect from the project, keeping your current job.",
            30,
            "free",
            ("income", "stability"),
        ),
        (
            "Weekend prototype",
            "Build the smallest version of the idea in two weekends and show it to someone.",
            14,
            "low",
            ("learning", "autonomy"),
        ),
    ),
    "learn_skill": (
        (
            "Two-week trial course",
            "Take a free or low-cost introductory course for two weeks before committing to more.",
            14,
            "free or low",
            ("learning", "time"),
        ),
        (
            "Talk to someone who made the switch",
            "Ask a person who learned this skill how long it took and how it changed their work.",
            7,
            "free",
            ("learning", "income"),
        ),
    ),
    "education": (
        (
            "Sit in on a class",
            "Ask to attend one session or watch a sample lecture of the programme.",
            7,
            "free",
            ("learning", "time"),
        ),
        (
            "Map the weekly schedule",
            "Draft a realistic week including study, work, rest and people you care about.",
            3,
            "free",
            ("time", "relationships"),
        ),
    ),
    "other": (
        (
            "Write down what would have to be true",
            "List the assumptions this path depends on, then check the easiest one first.",
            7,
            "free",
            ("stability",),
        ),
    ),
}

_COMMON_EXPERIMENT = (
    "Talk it through with people who know you",
    "Describe the options to a family member or friend and ask what they notice about how you talk about each.",
    7,
    "free",
    ("relationships",),
)

_DIR_PHRASE = {
    "likely_better": "probably more room for {v}",
    "likely_worse": "probably less room for {v}",
    "mixed": "a mix of gains and costs for {v}",
    "unclear": "no clear direction for {v}",
}


@dataclass(slots=True)
class _OptionContext:
    option: DecisionOption
    merged: dict[str, float | str]


_PRONOUNS = {
    "i": "you",
    "i'm": "you're",
    "i've": "you've",
    "my": "your",
    "me": "you",
    "myself": "yourself",
    "mine": "yours",
}


def _second_person(text: str) -> str:
    """Turn a first-person option label ('Continue in my current role') into second person."""
    words = text.split(" ")
    return " ".join(_PRONOUNS.get(w.lower(), w) for w in words)


def _fmt(v: float | str) -> str:
    if isinstance(v, float) and v.is_integer():
        return f"{int(v):,}"
    if isinstance(v, int | float):
        return f"{v:,}"
    return str(v)


class ScenarioEngine:
    def __init__(
        self, registry: EvidenceRegistry = DEFAULT_REGISTRY, narration: NarrationGate | None = None
    ) -> None:
        self.registry = registry
        self.narration = narration

    # ------------------------------------------------------------------ public
    def generate(self, request: DecisionRequest, decision_id: str | None = None) -> ScenarioResponse:
        if not request.consent_given:
            raise ConsentRequiredError("explicit consent is required before exploring a decision")
        decision_id = decision_id or f"dec-{uuid.uuid4().hex[:12]}"
        safety = classify(request.question, *(f"{o.label} {o.description}" for o in request.options))
        if safety.category == "crisis":
            return ScenarioResponse(
                decision_id=decision_id,
                scenarios=[],
                evidence=[],
                safety=safety,
                value_alignment_note="Scenario exploration is paused. Please reach out to someone who can support you now.",
            )

        evidence: dict[str, EvidenceRecord] = {}
        fact_ids = self._user_facts(decision_id, request.constraints, None, evidence)
        current_income = self._current_income(request)
        available_hours = calc._num(request.constraints, "available_hours_per_week")

        scenarios: list[Scenario] = []
        for opt in request.options:
            merged: dict[str, float | str] = {**request.constraints, **opt.inputs}
            opt_fact_ids = fact_ids + self._user_facts(decision_id, opt.inputs, opt.option_id, evidence)
            scenarios.append(
                self._scenario(
                    request,
                    decision_id,
                    _OptionContext(opt, merged),
                    opt_fact_ids,
                    current_income,
                    available_hours,
                    evidence,
                )
            )

        for s in scenarios:
            violations = check_scenario(s)
            if violations:  # a template bug, never user-visible
                raise ForbiddenLanguageError(violations)

        if self.narration is not None:
            scenarios = [self.narration.apply(s, request) for s in scenarios]

        return ScenarioResponse(
            decision_id=decision_id,
            scenarios=scenarios,
            evidence=[evidence[k] for k in sorted(evidence)],
            safety=None if safety.category == "none" else safety,
            value_alignment_note=self._alignment_note(request.values),
        )

    # ------------------------------------------------------------------ evidence helpers
    def _user_facts(
        self,
        decision_id: str,
        values: Mapping[str, float | str],
        option_id: str | None,
        evidence: dict[str, EvidenceRecord],
    ) -> list[str]:
        ids = []
        for key in sorted(values):
            eid = f"{decision_id}-{option_id + '-' if option_id else ''}fact-{key}"
            is_assumption = key == "expected_monthly_income_increase"
            evidence[eid] = EvidenceRecord(
                evidence_id=eid,
                domain="user_input",
                source_type=EvidenceSourceType.ASSUMPTION if is_assumption else EvidenceSourceType.USER_FACT,
                claim=(
                    f"Your assumption: {key} = {_fmt(values[key])}."
                    if is_assumption
                    else f"You reported {key} = {_fmt(values[key])}."
                ),
                data_ref=f"request.{'options[' + option_id + '].inputs' if option_id else 'constraints'}.{key}",
                applicability="Provided by you; not checked by this tool.",
                uncertainty="As accurate as the value you entered.",
                validation_status=ValidationStatus.UNVALIDATED,
            )
            ids.append(eid)
        return ids

    def _registry_ids(self, ids: tuple[str, ...], evidence: dict[str, EvidenceRecord]) -> list[str]:
        out = []
        for eid in ids:
            rec = self.registry.get(eid)
            if rec is not None:
                evidence[eid] = rec
                out.append(eid)
        return out

    @staticmethod
    def _current_income(request: DecisionRequest) -> float | None:
        v = calc._num(request.constraints, "current_monthly_income")
        if v is not None:
            return v
        for o in request.options:
            if o.kind == "current_path":
                return calc._num(o.inputs, "monthly_income")
        return None

    # ------------------------------------------------------------------ scenario construction
    def _scenario(
        self,
        request: DecisionRequest,
        decision_id: str,
        ctx: _OptionContext,
        fact_ids: list[str],
        current_income: float | None,
        available_hours: float | None,
        evidence: dict[str, EvidenceRecord],
    ) -> Scenario:
        opt = ctx.option
        kind = opt.kind
        horizon = request.horizon_years
        calcs: list[Calculation] = []
        missing: list[str] = []

        def add(result: calc.CalcResult | None, hint: str | None = None) -> calc.CalcResult | None:
            if result is None:
                if hint:
                    missing.append(hint)
                return None
            eid = f"{decision_id}-{opt.option_id}-calc-{result.name}"
            evidence[eid] = EvidenceRecord(
                evidence_id=eid,
                domain="arithmetic",
                source_type=EvidenceSourceType.DETERMINISTIC_CALCULATION,
                claim=result.claim,
                data_ref=result.formula,
                applicability="Arithmetic on the inputs you provided.",
                uncertainty=result.note,
                validation_status=ValidationStatus.UNVALIDATED,
            )
            calcs.append(
                Calculation(
                    name=result.name,
                    formula=result.formula,
                    inputs=result.inputs,
                    result=result.result,
                    unit=result.unit,
                    evidence_id=eid,
                    note=result.note,
                )
            )
            return result

        runway = add(
            calc.cash_runway(ctx.merged),
            "Add savings, monthly_expenses and monthly_income to see a cash-runway calculation for this path."
            if kind in ("start_project", "education")
            else None,
        )
        income = None
        if kind != "current_path":
            income = add(
                calc.income_difference(opt.inputs, current_income, horizon),
                "Add this path's monthly_income and your current_monthly_income to compare income arithmetically."
                if kind in ("new_job", "start_project")
                else None,
            )
        hours = add(calc.time_budget(opt.inputs, available_hours, horizon))
        add(calc.training_break_even(opt.inputs))

        # ---- trade-offs per user value
        priors = _PRIORS.get(kind, {})
        tradeoffs: list[Tradeoff] = []
        assumption_ids: set[str] = set()
        for vw in sorted(request.values, key=lambda v: (-v.weight, v.key)):
            direction, rationale, ids = priors.get(
                vw.key,
                (
                    "unclear",
                    f"This tool has no basis for saying how this path affects {vw.label.lower()}.",
                    (),
                ),
            )
            ev_ids = self._registry_ids(ids, evidence)
            if vw.key == "income" and income is not None and income.result is not None:
                direction = (
                    "likely_better" if income.result > 0 else "likely_worse" if income.result < 0 else "mixed"
                )
                rationale = "Based only on the monthly incomes you entered; " + (
                    "this path pays more per month."
                    if income.result > 0
                    else "this path pays less per month."
                    if income.result < 0
                    else "both pay the same per month."
                )
                ev_ids = [f"{decision_id}-{opt.option_id}-calc-{income.name}"]
            if (
                vw.key == "time"
                and hours is not None
                and hours.name == "weekly_hours_remaining"
                and hours.result is not None
            ):
                direction = "likely_worse" if hours.result < 0 else direction
                if hours.result < 0:
                    rationale = "Under your inputs this path needs more weekly hours than you said you have."
                    ev_ids = [f"{decision_id}-{opt.option_id}-calc-{hours.name}"]
            if (
                vw.key == "stability"
                and runway is not None
                and runway.result is not None
                and runway.result < 6
            ):
                direction = "likely_worse"
                rationale = (
                    "Under your inputs, savings would cover the monthly shortfall for less than half a year."
                )
                ev_ids = [f"{decision_id}-{opt.option_id}-calc-{runway.name}"]
            assumption_ids.update(i for i in ev_ids if i in self.registry)
            tradeoffs.append(
                Tradeoff(value_key=vw.key, direction=direction, rationale=rationale, evidence_ids=ev_ids)
            )  # type: ignore[arg-type]

        # ---- uncertainty & limits
        always = self._registry_ids(
            (
                "ext-affective-forecasting-2005",
                "assume-values-change",
                "method-no-individual-forecast",
                "method-simulation-separation",
            ),
            evidence,
        )
        uncertainties = [
            "How you would feel about this path later: research on affective forecasting suggests people often mispredict this.",
            f"Your priorities may change over {horizon:g} years; the weights you entered describe today.",
            _KIND_UNCERTAINTY.get(kind, _KIND_UNCERTAINTY["other"]),
            *missing,
        ]
        cannot_predict = ["Insufficient evidence to predict your future happiness on this path."]
        if kind in _KIND_CANNOT_PREDICT:
            cannot_predict.append(_KIND_CANNOT_PREDICT[kind])

        experiments = [
            ReversibleExperiment(title=t, description=d, duration_days=n, cost_note=c, learns_about=list(la))
            for (t, d, n, c, la) in (*_EXPERIMENTS.get(kind, _EXPERIMENTS["other"]), _COMMON_EXPERIMENT)
        ]

        # ---- narrative (template, illustrative)
        narrative = self._narrative(opt, horizon, tradeoffs, request.values)
        narr_id = f"{decision_id}-{opt.option_id}-narrative"
        evidence[narr_id] = EvidenceRecord(
            evidence_id=narr_id,
            domain="narrative",
            source_type=EvidenceSourceType.NARRATIVE,
            claim="Template-generated illustrative narrative for this path.",
            applicability="Imaginative sketch conditioned on the listed assumptions; not a prediction.",
            uncertainty="Entirely illustrative.",
            validation_status=ValidationStatus.ILLUSTRATIVE,
        )

        assumption_records = [self.registry.require(i) for i in sorted(assumption_ids)]
        assumption_records += [
            evidence[i] for i in fact_ids if evidence[i].source_type == EvidenceSourceType.ASSUMPTION
        ]
        evidence_ids = sorted(
            {
                *fact_ids,
                *always,
                narr_id,
                *(c.evidence_id for c in calcs),
                *(i for t in tradeoffs for i in t.evidence_ids),
            }
        )
        return Scenario(
            scenario_id=f"sc-{stable_hash([decision_id, opt.option_id], 12)}",
            decision_id=decision_id,
            option_id=opt.option_id,
            title=f"Illustrative path: {opt.label}",
            narrative=narrative,
            assumptions=assumption_records,
            tradeoffs=tradeoffs,
            calculations=calcs,
            uncertainties=uncertainties,
            cannot_predict=cannot_predict,
            reversible_experiments=experiments,
            evidence_ids=evidence_ids,
        )

    @staticmethod
    def _narrative(
        opt: DecisionOption, horizon: float, tradeoffs: list[Tradeoff], values: list[ValueWeight]
    ) -> str:
        labels = {v.key: v.label.lower() for v in values}
        top = [t for t in tradeoffs if t.value_key in labels][:3]
        if top:
            parts = [_DIR_PHRASE[t.direction].format(v=labels[t.value_key]) for t in top]
            joined = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
            values_sentence = f"If the assumptions listed below hold, the things you said matter most might look like this: {joined}."
        else:
            values_sentence = "You have not weighted any values yet, so this sketch does not compare against your priorities."
        horizon_txt = "a year" if horizon == 1 else f"{horizon:g} years"
        label = opt.label.strip().rstrip(".")
        label = _second_person(label[0].lower() + label[1:]) if label else "take this path"
        return (
            f"Imagine it is {horizon_txt} from now, in one version of events where you chose to {label}. "
            f"{_KIND_SENTENCE.get(opt.kind, _KIND_SENTENCE['other'])} "
            f"{values_sentence} "
            "Much of this cannot be known in advance, including how you would feel about it. "
            "This is one possibility among many, not a forecast."
        )

    @staticmethod
    def _alignment_note(values: list[ValueWeight]) -> str:
        ranked = [v for v in sorted(values, key=lambda v: (-v.weight, v.key)) if v.weight > 0]
        if not ranked:
            return "You have not weighted any values yet. Add priorities in Your North Star to compare paths against them."
        top = ", ".join(f"{v.label} ({v.weight:.1f})" for v in ranked[:3])
        return (
            f"Your highest-weighted priorities: {top}. Each path's trade-offs are shown against your weights "
            "qualitatively. No path is recommended, and you can change your weights at any time."
        )
