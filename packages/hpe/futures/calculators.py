"""Deterministic calculators (spec §19): arithmetic on user-provided inputs only.

Each calculator returns ``None`` when a required input is missing, so the engine can say which
input would enable it instead of inventing a value. Results are exact consequences of the inputs;
they are not forecasts.

Recognised input keys (per option ``inputs`` or request ``constraints``):
``monthly_income``, ``monthly_expenses``, ``savings``, ``hours_per_week``,
``available_hours_per_week``, ``current_monthly_income``, ``training_cost``,
``expected_monthly_income_increase`` (a user *assumption*, labelled as such).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class CalcResult:
    name: str
    formula: str
    inputs: dict[str, float]
    result: float | None
    unit: str
    note: str
    claim: str


def _num(src: Mapping[str, object], key: str) -> float | None:
    v = src.get(key)
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def cash_runway(inputs: Mapping[str, object]) -> CalcResult | None:
    savings = _num(inputs, "savings")
    expenses = _num(inputs, "monthly_expenses")
    income = _num(inputs, "monthly_income")
    if savings is None or expenses is None or income is None:
        return None
    burn = expenses - income
    used = {"savings": savings, "monthly_expenses": expenses, "monthly_income": income}
    formula = "savings / (monthly_expenses - monthly_income)"
    if burn <= EPSILON:
        return CalcResult(
            "cash_runway_months",
            formula,
            used,
            None,
            "months",
            "Under these inputs income covers expenses, so savings are not being drawn down; runway is not limited by savings.",
            "Your stated income covers your stated expenses, so savings would not be drawn down under these inputs.",
        )
    if savings <= 0:
        return CalcResult(
            "cash_runway_months",
            formula,
            used,
            0.0,
            "months",
            "No savings buffer under these inputs; expenses exceed income from the first month.",
            "With no savings and expenses above income, there is no runway under these inputs.",
        )
    months = savings / max(burn, EPSILON)
    return CalcResult(
        "cash_runway_months",
        formula,
        used,
        round(months, 2),
        "months",
        "Exact arithmetic on your inputs; real runway changes if income, expenses or savings change.",
        f"Savings of {savings:,.0f} cover a monthly shortfall of {burn:,.0f} for about {months:.1f} months.",
    )


def income_difference(
    option_inputs: Mapping[str, object], current_income: float | None, horizon_years: float
) -> CalcResult | None:
    new = _num(option_inputs, "monthly_income")
    if new is None or current_income is None:
        return None
    diff = (new - current_income) * 12.0 * horizon_years
    return CalcResult(
        "income_difference_over_horizon",
        "(option monthly_income - current monthly_income) * 12 * horizon_years",
        {"monthly_income": new, "current_monthly_income": current_income, "horizon_years": horizon_years},
        round(diff, 2),
        "currency units",
        "Gross difference before taxes, raises, bonuses or job changes; assumes both incomes stay constant.",
        f"If both incomes stayed constant, the difference over {horizon_years:g} years would be {diff:,.0f}.",
    )


def time_budget(
    option_inputs: Mapping[str, object], available: float | None, horizon_years: float
) -> CalcResult | None:
    hours = _num(option_inputs, "hours_per_week")
    if hours is None:
        return None
    if available is not None:
        remaining = available - hours
        return CalcResult(
            "weekly_hours_remaining",
            "available_hours_per_week - hours_per_week",
            {"available_hours_per_week": available, "hours_per_week": hours},
            round(remaining, 2),
            "hours/week",
            "A negative value means the path needs more hours than you said you have.",
            f"This path uses {hours:g} of your {available:g} available hours per week, leaving {remaining:g}.",
        )
    total = hours * 52.0 * horizon_years
    return CalcResult(
        "total_hours_over_horizon",
        "hours_per_week * 52 * horizon_years",
        {"hours_per_week": hours, "horizon_years": horizon_years},
        round(total, 1),
        "hours",
        "Assumes the weekly commitment stays constant every week.",
        f"At {hours:g} hours per week, this path takes about {total:,.0f} hours over {horizon_years:g} years.",
    )


def training_break_even(option_inputs: Mapping[str, object]) -> CalcResult | None:
    cost = _num(option_inputs, "training_cost")
    gain = _num(option_inputs, "expected_monthly_income_increase")
    if cost is None or gain is None or cost <= 0:
        return None
    used = {"training_cost": cost, "expected_monthly_income_increase": gain}
    formula = "training_cost / expected_monthly_income_increase"
    if gain <= EPSILON:
        return CalcResult(
            "training_break_even_months",
            formula,
            used,
            None,
            "months",
            "No break-even under these inputs because the assumed income increase is zero or negative.",
            "With no assumed income increase, the training cost is not recovered through income under these inputs.",
        )
    months = cost / gain
    return CalcResult(
        "training_break_even_months",
        formula,
        used,
        round(months, 2),
        "months",
        "The income increase is YOUR assumption, not a forecast; the result is only as good as that assumption.",
        f"If your assumed increase of {gain:,.0f} per month held, the cost of {cost:,.0f} would be recovered in about {months:.1f} months.",
    )
