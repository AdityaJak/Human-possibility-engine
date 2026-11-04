"""spatial_bandit — Schulz et al. (2020) "Finding structure in multi-armed bandits" (8 options, rounds of 10 trials),
via Psych-101 (Apache-2.0). Role ``secondary``; needs ``k_max: 8`` in an experiment definition.

Transcript: "You will be playing a game for R rounds." / "Each round contains N trials." / "... a reward between L
and H points." / "You can choose between options 1, 2, …, 7 and 8 ...", then per round "You are playing round r:"
and one line per trial "You press <<k>> and get x points." Action index = position in the instruction's option
list (the options are ordered keys; their order is the spatial order). Reward → (x − L) / (H − L) with the stated
range (a value outside it is a parse error). Features: [1, trial-in-round progress, first-trial-of-round flag,
round progress] in [-1, 1]; block = round (options reset every round).
"""

from __future__ import annotations

import re

from hpe.evaluation.human.psych101 import ParseError, Participant
from hpe.evaluation.human.psych101_v2 import GenericTrial, _pad
from hpe.population.core.families import P3_PARTITIONS, FamilySpec, register_family

_ROUNDS = re.compile(r"playing a game for (\d+) rounds\.")
_TRIALS = re.compile(r"Each round contains (\d+) trials\.")
_RANGE = re.compile(r"reward between (-?\d+(?:\.\d+)?) and (-?\d+(?:\.\d+)?) points\.")
_OPTIONS = re.compile(r"You can choose between options ([^.]+?) by pressing")
_ROUND = re.compile(r"^You are playing round (\d+):\s*$")
_TRIAL = re.compile(r"^You press <<([^>]+)>> and get (-?\d+(?:\.\d+)?) points\.\s*$")


def parse_spatial_bandit(text: str, experiment: str, participant: str) -> Participant:
    where = f"{experiment}#{participant}"
    mr, mt, mg, mo = _ROUNDS.search(text), _TRIALS.search(text), _RANGE.search(text), _OPTIONS.search(text)
    if not (mr and mt and mg and mo):
        raise ParseError(f"{where}: spatial_bandit instructions not found")
    n_rounds, n_trials = int(mr.group(1)), int(mt.group(1))
    lo, hi = float(mg.group(1)), float(mg.group(2))
    labels = [s.strip() for s in re.split(r",| and ", mo.group(1)) if s.strip()]
    if len(labels) != 8 or len(set(labels)) != 8:
        raise ParseError(f"{where}: expected 8 distinct options, got {labels}")
    p = Participant(experiment, participant, "spatial_bandit", 8, meta={"labels": labels, "range": [lo, hi]})
    rnd, t_in = -1, 0
    for ln in text.split("\n"):
        s = ln.strip()
        g = _ROUND.match(s)
        if g:
            rnd, t_in = int(g.group(1)) - 1, 0
            continue
        if "<<" not in s:
            continue
        m = _TRIAL.match(s)
        if not m or m.group(1) not in labels or rnd < 0:
            raise ParseError(f"{where}: unparsable spatial_bandit line {s[:100]!r}")
        x = float(m.group(2))
        if not (lo - 1e-9 <= x <= hi + 1e-9):
            raise ParseError(f"{where}: reward {x} outside the stated range [{lo}, {hi}]")
        if t_in >= n_trials or rnd >= n_rounds:
            raise ParseError(f"{where}: more trials or rounds than stated")
        feats = _pad(
            [
                1.0,
                (t_in / max(n_trials - 1, 1)) * 2 - 1,
                1.0 if t_in == 0 else -1.0,
                (rnd / max(n_rounds - 1, 1)) * 2 - 1,
            ]
        )
        p.trials.append(
            GenericTrial(
                labels.index(m.group(1)), 8, feats, "bandit", (x - lo) / (hi - lo), block=rnd, phase="main"
            )
        )
        t_in += 1
    return p


register_family(
    FamilySpec(
        name="spatial_bandit",
        prefixes=("schulz2020finding/",),
        n_actions=8,
        partition_file=P3_PARTITIONS,
        parser=parse_spatial_bandit,
        role="secondary",
        note="Schulz et al. (2020), 8-armed structured bandit; needs k_max 8",
    )
)
