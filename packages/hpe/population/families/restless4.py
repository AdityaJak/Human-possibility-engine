"""restless4 — Bahrami & Navajas (2020) four-armed restless bandit, via Psych-101 (Apache-2.0).

Transcript: "You will be asked to repeatedly choose between four different options labeled L, G, O, and U."
then one line per trial "You press <<X>> and get N points." Labels are per participant; the option order
in the instruction line defines action indices 0..3. Points (0-100) are mapped to [0, 1] by /100.
No context features (all zeros): the task has one fixed set of four options throughout.
"""

from __future__ import annotations

import re

from hpe.evaluation.human.psych101 import ParseError, Participant
from hpe.evaluation.human.psych101_v2 import N_FEATURES, GenericTrial
from hpe.population.core.families import P3_PARTITIONS, FamilySpec, register_family

_LABELS = re.compile(r"four different options labeled ([A-Z]), ([A-Z]), ([A-Z]), and ([A-Z])\.")
_TRIAL = re.compile(r"^You press <<([A-Z])>> and get (-?\d+(?:\.\d+)?) points\.\s*$")


def parse_restless4(text: str, experiment: str, participant: str) -> Participant:
    m = _LABELS.search(text)
    if not m:
        raise ParseError(f"{experiment}#{participant}: restless4 option labels not found")
    labels = list(m.groups())
    p = Participant(experiment, participant, "restless4", 4)
    zeros = (0.0,) * N_FEATURES
    for ln in text.split("\n"):
        if "<<" not in ln:
            continue
        mt = _TRIAL.match(ln.strip())
        if not mt:
            raise ParseError(f"{experiment}#{participant}: unparsable restless4 line {ln[:100]!r}")
        key, pts = mt.groups()
        if key not in labels:
            raise ParseError(f"{experiment}#{participant}: unknown key {key!r}")
        r = min(1.0, max(0.0, float(pts) / 100.0))
        p.trials.append(GenericTrial(labels.index(key), 4, zeros, "bandit", r, block=0, phase="main"))
    return p


register_family(
    FamilySpec(
        name="restless4",
        prefixes=("bahrami2020four/",),
        n_actions=4,
        partition_file=P3_PARTITIONS,
        parser=parse_restless4,
        role="counted_primary",
        note="Bahrami & Navajas (2020); 4-armed restless bandit",
    )
)
