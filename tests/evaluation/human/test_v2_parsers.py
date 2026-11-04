"""Synthetic transcripts in the exact Psych-101 line formats of the five new families (no real data)."""

from __future__ import annotations

import numpy as np
import pytest

from hpe.evaluation.human import psych101_v2 as v2
from hpe.evaluation.human.models2 import encode_generic
from hpe.evaluation.human.psych101 import ParseError


def bandit2_text(rng, games=3, trials=4):
    L = [
        "In this task, you have to repeatedly choose between two slot machines labeled U and P.",
        "Machine U will not always give you the same points when you select it again, but machine P will always give 0 points when you select it.",
        "",
    ]
    for g in range(1, games + 1):
        L.append(f"Game {g}:")
        for _ in range(trials):
            m = rng.choice(["U", "P"])
            L.append(f"You press <<{m}>> and get {int(rng.integers(-3, 4)) if m == 'U' else 0} points.")
        L.append("")
    return "\n".join(L)


def srl3_text(rng):
    L = [
        "The three responses available are S, F, and A.",
        "You will play 2 games, each with a different mapping from stimuli to responses.",
        "",
        "Game 1:",
        "There are 2 different stimuli.",
    ]
    for _ in range(6):
        L.append(
            f"You see stimulus {int(rng.integers(0, 2))}. You press <<{rng.choice(list('SFA'))}>> and get {int(rng.integers(-1, 2))} points."
        )
    L += ["", "Game 2:", "There are 6 different stimuli."]
    for _ in range(6):
        L.append(
            f"You see stimulus {int(rng.integers(0, 6))}. You press <<{rng.choice(list('SFA'))}>> and get {int(rng.integers(0, 2))} points."
        )
    return "\n".join(L)


def horizon_text(rng):
    L = ["You are participating in multiple games involving two slot machines, labeled C and A.", ""]
    for g, n in ((1, 5), (2, 10)):
        L.append(f"Game {g}. There are {n} trials in this game.")
        for _ in range(4):
            L.append(
                f"You are instructed to press {rng.choice(['C', 'A'])} and get {int(rng.integers(0, 101))} points."
            )
        for _ in range(n - 4):
            L.append(f"You press <<{rng.choice(['C', 'A'])}>> and get {int(rng.integers(0, 101))} points.")
        L.append("")
    return "\n".join(L)


def casino_text(rng):
    L = [
        "Each casino owns two slot machines that return either 0 or 0.5 points stochastically with different probabilities.",
        "",
    ]
    pairs = {1: ("F", "I"), 2: ("L", "J"), 3: ("B", "C"), 4: ("P", "T")}
    for _ in range(16):
        c = int(rng.integers(1, 5))
        a, b = pairs[c]
        L.append(
            f"You go to casino {c}. You can choose between machines {a} and {b}. You press <<{rng.choice([a, b])}>> and receive {rng.choice([0.0, 0.5])} points."
        )
    return "\n".join(L)


def wpt_text(rng):
    L = [
        "Your task is to decide if the combination of cards presented predicts rainy weather (by pressing E) or fine weather (by pressing J).",
        "",
    ]
    for _ in range(12):
        cards = sorted(rng.choice([1, 2, 3, 4], size=int(rng.integers(1, 4)), replace=False))
        key = rng.choice(["E", "J"])
        weather = rng.choice(["rainy", "fine"])
        correct = (key == "E") == (weather == "rainy")
        L.append(
            f"You are seeing the following: {', '.join(f'card {c}' for c in cards)}. You press <<{key}>>. You are {'correct' if correct else 'wrong'}, the weather is {'indeed ' if correct else ''}{weather}."
        )
    return "\n".join(L)


def garcia_text(rng):
    L = ["This experiment is composed of three parts.", ""]
    for _ in range(6):
        a, b = rng.choice(list("QLSYVH"), size=2, replace=False)
        pressed = rng.choice([a, b])
        other = b if pressed == a else a
        L.append(
            f"You can choose between option {a} and option {b}. You press <<{pressed}>> and get {rng.choice([-1.0, 1.0])} points. You would have gotten {rng.choice([-1.0, 1.0])} points had you chosen option {other} instead. "
        )
    for _ in range(4):
        a = rng.choice(list("QLS"))
        L.append(
            f"You can choose between option {a} and the following option by pressing button I: -1.0 point with 30.0% chance and 1.0 point with 70.0% chance. You press <<{rng.choice([a, 'I'])}>>."
        )
    L.append("Q: What are the odds that option Q gives 1.0 points? A: <<50.0>>%.")
    return "\n".join(L)


def described_text(rng, same_letter=False):
    L = ["You will encounter each problem 25 times.", ""]
    for _prob in range(2):
        a, b = ("T", "T") if same_letter else ("F", "X")
        L.append(f"Option {a} delivers 3 points with 80.0% chance, 94 points with 20.0% chance.")
        L.append(f"Option {b} delivers -19 points with unknown chance, 59 points with unknown chance.")
        for _ in range(2):
            L.append(f"You press <<{rng.choice([a, b])}>>.")
        for _ in range(3):
            pressed = rng.choice([a, b])
            other = b if pressed == a else a
            L.append(
                f"You press <<{pressed}>> and gain 59 points. You would have lost 19 points had you chosen option {other}."
            )
        L.append("")
    return "\n".join(L)


@pytest.mark.parametrize(
    "paradigm,maker,n_trials,n_actions",
    [
        ("bandit2", bandit2_text, 12, 2),
        ("srl3", srl3_text, 12, 3),
        ("horizon", horizon_text, 15, 2),
        ("casino", casino_text, 16, 2),
        ("wpt", wpt_text, 12, 2),
        ("garcia", garcia_text, 10, 2),
        ("described", described_text, 10, 2),
    ],
)
def test_parsers_produce_valid_generic_trials(paradigm, maker, n_trials, n_actions):
    rng = np.random.default_rng(0)
    p = v2.PARSERS[paradigm](maker(rng), f"{paradigm}/exp1.csv", "0")
    assert p.paradigm == paradigm and len(p.trials) == n_trials
    rep = v2.validate_generic([p])
    assert rep[f"{paradigm}/exp1.csv"]["participants"] == 1
    steps = encode_generic(p)
    assert len(steps) == n_trials
    for obs, a, fb, _forced in steps:
        assert obs.context.shape == (1, 12) and obs.n_actions == n_actions and 0 <= a < n_actions
        assert np.all(np.abs(obs.context) <= 1.0)
        if fb.mode == "full":
            assert (
                fb.correct_action is not None
                and fb.rewards_all is not None
                and len(fb.rewards_all) == n_actions
            )


def test_horizon_forced_trials_are_marked_and_not_scored():
    p = v2.parse_horizon(horizon_text(np.random.default_rng(1)), "wilson2014humans/exp1.csv", "0")
    forced = [t.forced for t in p.trials]
    assert forced[:4] == [True] * 4 and forced[4] is False and sum(forced) == 8
    assert p.trials[0].meta["horizon"] == 5 and p.trials[5].meta["horizon"] == 10


def test_garcia_excludes_judgements_and_marks_described_option():
    p = v2.parse_garcia(garcia_text(np.random.default_rng(2)), "garcia2023experiential/exp1.csv", "0")
    assert p.meta["excluded_judgements"] == 1
    part2 = [t for t in p.trials if t.phase == "part2"]
    assert len(part2) == 4 and all(t.mode == "none" for t in part2)
    assert all(abs(t.meta["p_win_described"] - 0.7) < 1e-9 for t in part2)


def test_described_unknown_probabilities_flagged_and_ambiguous_problems_skipped():
    rng = np.random.default_rng(3)
    p = v2.parse_described(described_text(rng), "plonsky2018when/exp1.csv", "0")
    assert len(p.trials) == 10
    # option 2 (index 6..11 of the 12 features) has the unknown flag set (feature 11 == 1)
    assert p.trials[0].features[11] == 1.0 and p.trials[0].features[5] == -1.0
    q = v2.parse_described(described_text(rng, same_letter=True), "plonsky2018when/exp1.csv", "1")
    assert len(q.trials) == 0 and q.meta["ambiguous_problem_choices"] == 10


def test_wpt_verdict_mismatch_raises():
    bad = "predicts rainy weather (by pressing E) or fine weather (by pressing J).\n\nYou are seeing the following: card 1. You press <<E>>. You are correct, the weather is fine."
    with pytest.raises(ParseError):
        v2.parse_wpt(bad, "speekenbrink2008learning/exp1.csv", "0")


def test_paradigm_lookup():
    assert v2.paradigm_of("wilson2014humans/exp3.csv") == "horizon"
    assert v2.paradigm_of("collsiöö2023MCPL/exp1.csv") is None
