"""gbe_risk — external (non-Psych-101) family for the population-vs-adaptation question (exploratory, X3-007).

Rutledge's Great Brain Experiment risky-decision task (Dryad doi:10.5061/dryad.prr4xgxkk, CC0-1.0), adapted by
the affective-social lane (``hpe.population.affect.gbe``) with its participant partition
(``reports/phase3/affect/partitions_gbe.json``, seed 20261108, written before any fit). Here the affect columns
are stripped (encoding v3 only) because the question is individual adaptation of choice, not affect. To fit
the CPU budget, fixed seeded subsamples are used: 2,000 dev and 1,000 tune participants, first 4 plays (≤ 120
choices) each. Registered by the director while the evaluation lane is blocked. Role ``external``: not a
counted Gate C family (the counted families were fixed before fitting).
"""

from __future__ import annotations

from functools import lru_cache

from hpe.population.affect import gbe
from hpe.population.core.encoding import EncodedSeq, input_dim
from hpe.population.core.families import P3_PARTITIONS, FamilySpec, register_family

N_SUBSAMPLE = {"dev": 2000, "tune": 1000, "test": 0}  # 0 = all (test is only ever read after a freeze)
MAX_PLAYS = 4
SUBSAMPLE_SEED = 20261109


def _strip(seq: EncodedSeq, k_max: int) -> EncodedSeq:
    d = input_dim(k_max)
    return EncodedSeq(
        seq.key, "gbe_risk", k_max, seq.n_actions, seq.x[:, :d].copy(), seq.actions, seq.avail, seq.scored
    )


@lru_cache(maxsize=4)
def _load(partition: str, experiment_id: str) -> tuple[tuple[EncodedSeq, ...], tuple[str, ...]]:
    data, notes = gbe.load_gbe([partition], experiment_id=experiment_id, max_plays=MAX_PLAYS)
    seqs = gbe.subsample(data[partition], N_SUBSAMPLE[partition], SUBSAMPLE_SEED, f"gbe_risk_{partition}")
    return tuple(seqs), tuple(notes[:50])


def load(partition: str, experiment_id: str, k_max: int) -> tuple[list[EncodedSeq], list[str]]:
    seqs, notes = _load(partition, experiment_id)
    if seqs and seqs[0].k_max != k_max:
        raise ValueError(f"gbe_risk is encoded with k_max {seqs[0].k_max}, run asked for {k_max}")
    return [_strip(s, k_max) for s in seqs], list(notes)


def _no_parser(*_a: object) -> None:
    raise RuntimeError("gbe_risk is loaded through its own loader")


register_family(
    FamilySpec(
        name="gbe_risk",
        prefixes=("gbe_risk/",),
        n_actions=2,
        partition_file=P3_PARTITIONS,
        parser=_no_parser,  # type: ignore[arg-type]
        role="external",
        note="Great Brain Experiment risky choice (CC0); affect columns stripped; subsampled",
        loader=load,
    )
)
