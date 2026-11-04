"""Age-dependent cognitive schedules.

THESE ARE SCRIPTED MODELLING ASSUMPTIONS, not empirical estimates of human development.
They are deliberately simple, monotone, inspectable functions of simulated age so that their
contribution can be separated from learning by ablation (``use_plasticity_schedule=False``,
``use_forgetting=False``). Nothing here encodes *what* an agent knows; they only scale how
fast parameters move (plasticity), how many episodes can be held for replay (working memory),
and how quickly unpractised skills decay (forgetting).

* plasticity(age): 1.5 in childhood (<12), linear to 1.0 at 25, flat to 50, then linear
  decline to a floor of 0.4 at 90.
* working_memory(age): 25% of base replay capacity at birth, full by age 12, declining after
  70 to a floor of 50%.
* forgetting_multiplier(age): 1 until 60, then (1 + (age-60)/10)^2 (x9 at 80).
"""

from __future__ import annotations


def plasticity(age_years: float, enabled: bool = True) -> float:
    if not enabled:
        return 1.0
    a = float(age_years)
    if a < 12.0:
        return 1.5
    if a < 25.0:
        return 1.5 - 0.5 * (a - 12.0) / 13.0
    if a < 50.0:
        return 1.0
    return max(0.4, 1.0 - 0.6 * (a - 50.0) / 40.0)


def working_memory(age_years: float, enabled: bool = True) -> float:
    if not enabled:
        return 1.0
    a = float(age_years)
    frac = min(1.0, 0.25 + 0.75 * max(0.0, a) / 12.0)
    if a > 70.0:
        frac *= max(0.5, 1.0 - 0.01 * (a - 70.0))
    return frac


def forgetting_multiplier(age_years: float, enabled: bool = True) -> float:
    if not enabled:
        return 1.0
    a = float(age_years)
    if a <= 60.0:
        return 1.0
    return (1.0 + (a - 60.0) / 10.0) ** 2
