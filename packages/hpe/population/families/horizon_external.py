"""horizon_external — the horizon task (Wilson et al., 2014 design) in three further studies, via Psych-101:
Sadeghiyeh et al. (2020), Somerville et al. (2017), Feng et al. (2021).

The transcripts use the same vocabulary as ``wilson2014humans`` ("Game N. There are N trials in this game.",
"You are instructed to press X and get N points.", "You press <<X>> and get N points."), so the Phase II
parser ``psych101_v2.parse_horizon`` is reused unchanged (validated on dev only:
``docs/phase3/HUMAN_DATA_VALIDATION.md``). Role ``external``: models are fitted on ``horizon`` (wilson2014)
only; dev/tune of these studies serve parser validation (thesis §3).
"""

from __future__ import annotations

from hpe.evaluation.human.psych101 import Participant
from hpe.evaluation.human.psych101_v2 import parse_horizon
from hpe.population.core.families import P3_PARTITIONS, FamilySpec, register_family

PREFIXES = ("sadeghiyeh2020temporal/", "somerville2017charting/", "feng2021dynamics/")


def parse_horizon_external(text: str, experiment: str, participant: str) -> Participant:
    p = parse_horizon(text, experiment, participant)
    p.paradigm = "horizon_external"
    return p


register_family(
    FamilySpec(
        name="horizon_external",
        prefixes=PREFIXES,
        n_actions=2,
        partition_file=P3_PARTITIONS,
        parser=parse_horizon_external,
        role="external",
        note="horizon task replications (sadeghiyeh2020, somerville2017, feng2021); parser psych101_v2.parse_horizon",
    )
)
