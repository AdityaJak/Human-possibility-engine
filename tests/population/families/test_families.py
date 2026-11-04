"""Family adapters (evaluation team): two_step and horizon_external parsers on synthetic transcripts."""

from __future__ import annotations

import pytest

from hpe.evaluation.human.psych101 import ParseError
from hpe.population.core.encoding import Layout, encode
from hpe.population.core.families import get_family
from hpe.population.families.horizon_external import parse_horizon_external
from hpe.population.families.two_step import parse_two_step

ONE_STAGE = """Each day you will either be presented with spaceships N and T or with spaceships K and W.
These spaceships will take you to two different planets R and B.
Your goal is to get as much treasure and as little antimatter as possible over the next 125 days.

You are presented with spaceships T and N. You press <<T>>. You end up on planet R. You find 2 pieces of antimatter.
You are presented with spaceships K and W. You do not respond in time on this day. You do not go to any planet. You find nothing.
You are presented with spaceships K and W. You press <<W>>. You end up on planet B. You find 5 pieces of space treasure.
You are presented with spaceships N and T. You press <<N>>. You end up on planet B. You find nothing.
"""

TWO_STAGE = """You will be taking one of the spaceships F or Q to one of the planets M or Z.
Planet M has aliens D and E, and planet Z has aliens H and J on it.
If there is a treasure multiplier, you will receive 5 times the amount of treasure you will find.
Your goal is to get as much treasure as possible over the course of the next 200 days.

There is a treasure multiplier. You are presented with spaceships Q and F. You press <<Q>>. You end up on planet M. You see alien D and alien E. You press <<E>>. You find 1 pieces of space treasure.
There is no treasure multiplier. You are presented with spaceships F and Q. You press <<F>>. You end up on planet Z. You see alien H and alien J. You do not respond in time for this day. You do not ask any alien for treasure. You do not get any treasure.
There is no treasure multiplier. You are presented with spaceships F and Q. You press <<F>>. You end up on planet Z. You see alien H and alien J. You press <<H>>. You find 0 pieces of space treasure.
"""

ZOROWITZ = """To visit a planet, you will choose one rocket ship from two by pressing the corresponding button.

You are presented with two spaceships called A and B. You press <<B>>. You end up on the blue planet. You see a blue alien named X and a blue alien named Y. You press <<Y>>. You find treasure.
You are presented with two spaceships called A and B. You press <<A>>. You end up on the red planet. You see a red alien named P and a red alien named Q. You press <<P>>. You find junk.
"""


def test_two_step_one_stage_design():
    p = parse_two_step(ONE_STAGE, "kool2016when/exp1.csv", "3")
    assert p.meta["design"] == "one_stage" and p.meta["missed_stage1"] == 1
    assert [t.choice for t in p.trials] == [1, 1, 0]  # canonical: instruction order N,T / K,W
    assert [t.mode for t in p.trials] == ["bandit"] * 3
    assert [t.reward for t in p.trials] == pytest.approx([2 / 9, 1.0, 4 / 9])  # (r + 4) / 9
    f0, f1, f2 = (t.features for t in p.trials)
    assert f0[2] == -1.0 and f1[2] == 1.0  # start pair
    assert f0[6] == 1.0 and f1[6] == -1.0 and f2[6] == -1.0  # presented order swapped / not
    assert f0[4] == 0.0 and f1[4] == 0.0 and f2[4] == 1.0  # previous planet (0 after a missed day)
    assert all(-1.0 <= x <= 1.0 for t in p.trials for x in t.features)


def test_two_step_two_stage_with_multiplier_and_missed_alien():
    p = parse_two_step(TWO_STAGE, "kool2017cost/exp2.csv", "1")
    assert p.meta["design"] == "two_stage" and p.meta["missed_stage2"] == 1
    phases = [t.phase for t in p.trials]
    assert phases == ["stage1", "stage2", "stage1", "stage1", "stage2"]
    assert [t.mode for t in p.trials] == ["none", "bandit", "none", "none", "bandit"]
    assert [t.choice for t in p.trials] == [1, 1, 0, 0, 0]
    assert p.trials[1].reward == 1.0 and p.trials[4].reward == 0.0
    assert p.trials[0].features[5] == 1.0 and p.trials[2].features[5] == -1.0  # multiplier flag
    assert p.trials[1].features[3] == -1.0 and p.trials[4].features[3] == 1.0  # current planet M / Z
    assert p.trials[3].features[4] == 1.0  # previous day reached Z although the alien choice was missed
    s = encode(p, "two_step", 4)
    L = Layout(4)
    # the stage-1 step has no feedback, so the next step's previous-feedback flag is 0
    assert s.x[2, L.prev_feedback] == 1.0 and s.x[1, L.prev_feedback] == 0.0


def test_two_step_zorowitz_first_presentation_order():
    p = parse_two_step(ZOROWITZ, "zorowitz2023data/exp1.csv", "0")
    assert [t.choice for t in p.trials] == [1, 1, 0, 0]
    assert p.trials[1].features[3] == 1.0 and p.trials[3].features[3] == -1.0  # blue = +1, red = -1
    assert p.trials[1].reward == 1.0 and p.trials[3].reward == 0.0


def test_two_step_rejects_out_of_range_and_wrong_alien():
    bad = TWO_STAGE.replace("You find 0 pieces", "You find -165 pieces")
    with pytest.raises(ParseError, match="outside the fixed range"):
        parse_two_step(bad, "kool2017cost/exp2.csv", "1")
    bad2 = TWO_STAGE.replace(
        "You see alien D and alien E. You press <<E>>", "You see alien H and alien J. You press <<J>>"
    )
    with pytest.raises(ParseError):
        parse_two_step(bad2, "kool2017cost/exp2.csv", "1")
    bad3 = ONE_STAGE.replace("You press <<W>>", "You press <<N>>")
    with pytest.raises(ParseError, match="not among presented"):
        parse_two_step(bad3, "kool2016when/exp1.csv", "3")


def test_codes_are_salted_per_participant():
    a = parse_two_step(ONE_STAGE, "kool2016when/exp1.csv", "3")
    b = parse_two_step(ONE_STAGE, "kool2016when/exp1.csv", "4")
    assert a.trials[0].features[:8] == b.trials[0].features[:8]
    assert any(a.trials[i].features[8:] != b.trials[i].features[8:] for i in range(3))


def test_horizon_external_reuses_the_horizon_parser():
    text = (
        "You are participating in multiple games involving two slot machines, labeled J and F.\n"
        "Game 1. There are 5 trials in this game.\n"
        "You are instructed to press J and get 54 points.\n"
        "You press <<F>> and get 61 points.\n"
    )
    p = parse_horizon_external(text, "sadeghiyeh2020temporal/exp1.csv", "2")
    assert p.paradigm == "horizon_external" and [t.forced for t in p.trials] == [True, False]
    assert get_family("horizon_external").role == "external"
    assert get_family("two_step").role == "counted_if_validated"
