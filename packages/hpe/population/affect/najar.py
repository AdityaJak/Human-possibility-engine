"""Adapter for Najar, Bonnet, Bahrami & Palminteri (2020) social reinforcement learning data (Gate F, social).

Source (opened 2026-11-04): https://github.com/hrl-team/mfree_imitation (data/task1.csv = experiment 1,
24 participants; data/task2.csv = experiment 2, 44 participants), repository licence "MIT License,
Copyright (c) 2019 hrl-team". Paper: PLoS Biology 18(12): e3001028. Set ``HPE_NAJAR_DIR`` to a folder with
the two CSVs (never committed).

Task: two-armed probabilistic learning on 3 symbol pairs ("states") per session; in observational
states, the participant's own choices are interleaved with choices of a demonstrator (``id == "dem"``),
whose outcomes are never shown. Each own choice is one step (``K = 2``); demonstrations are *not* steps.

Labels. The CSVs code a choice as good/bad symbol (``cgb``). A population model trained on that coding
would learn "pick action 1" without learning anything. Each (participant, state) therefore gets a seeded
random label flip: action = ``cgb XOR flip``. The flip is fixed per participant and state, so learning
within a person is unaffected.

Encoding v3 base (first ``input_dim(2)`` columns): features = state-within-session one-hot (3), session
one-hot (4), own choices so far on this state / 20; previous own action/outcome (outcome +1/-1 mapped to
1/0, chosen option only); block start at each session. Two blocks of :data:`N_EXTRA` social columns are
appended — the observed one, then a causal within-participant resampling (the matched-complexity control):

    [0] any demonstration on the current state so far
    [1] last demonstration on this state chose action 0   [2] ... chose action 1
    [3] share of this state's demonstrations that chose action 1 (0.5 if none)
    [4] demonstrations on this state since the participant's previous choice on it, / 3

In the control block every demonstration's action is replaced by the action of a uniformly drawn
*earlier* demonstration of the same participant (any state; the first one is kept). Because labels are
flipped independently per state, a borrowed action says nothing about the current pair.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hpe.core.rng import stream
from hpe.evaluation.human.partitions import PARTITIONS, check_partition_access
from hpe.population.core.data import P3_FREEZE_DIR
from hpe.population.core.encoding import EncodedSeq, Layout, input_dim
from hpe.population.core.families import REPO_ROOT

DATASET = "najar2020_imitation"
DATA_ENV = "HPE_NAJAR_DIR"
FILES = {
    "exp1": ("task1.csv", "b481100301b1443d42049bf86d148c2883a51f6ec9cde599983672c6cbca7453"),
    "exp2": ("task2.csv", "927a2eb40709c46668779956e16a42f86c991a4c9815cdddba159bcef899f3eb"),
}
SOURCE = {
    "title": "Data for Najar, Bonnet, Bahrami & Palminteri (2020) PLoS Biology 18(12): e3001028",
    "repository": "https://github.com/hrl-team/mfree_imitation",
    "licence": "MIT License, Copyright (c) 2019 hrl-team (repository LICENSE; covers data/)",
    "licence_sha256": "169e0cad079e46cedeaae3f4f31973a310259ce75e7b681b5f69beff5c579530",
    "files": {k: {"path": f"data/{v[0]}", "sha256": v[1]} for k, v in FILES.items()},
    "accessed": "2026-11-04",
}
PARTITION_FILE = REPO_ROOT / "reports" / "phase3" / "affect" / "partitions_najar2020.json"
SPLIT_SEED = 20261108
FRACTIONS = {"dev": 0.4, "tune": 0.2, "test": 0.4}
K = 2
D_BASE = input_dim(K)
N_EXTRA = 5
LABEL_SEED = 20261108
SHUFFLE_SEED = 20261108
# Deviation DA-1 (docs/phase3/AFFECTIVE_SOCIAL_VALIDATION.md): while inspecting the CSV layout, before the
# partition existed, the first 12 rows (6 own choices with outcomes, 6 demonstrations) of these participants
# were printed. One of them fell into the test partition; it is never scored in a confirmatory run.
EXPOSED = frozenset({"najar2020_imitation#exp1_3"})


class NajarDataUnavailable(RuntimeError):
    pass


def key_of(exp: str, expid: str) -> str:
    return f"{DATASET}#{exp}_{expid}"


def data_dir() -> Path:
    p = os.environ.get(DATA_ENV)
    if not p or not Path(p).is_dir():
        raise NajarDataUnavailable(
            f"set {DATA_ENV} to a folder with task1.csv and task2.csv ({SOURCE['repository']})"
        )
    return Path(p)


def _read(folder: Path) -> dict[str, list[dict[str, str]]]:
    """Rows per participant key, in file order (sorted by session, trial in the source)."""
    out: dict[str, list[dict[str, str]]] = {}
    for exp, (fname, sha) in sorted(FILES.items()):
        raw = (folder / fname).read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        if h != sha:
            raise NajarDataUnavailable(f"{fname}: sha256 {h} != expected {sha}")
        for r in csv.DictReader(raw.decode("utf-8").splitlines()):
            out.setdefault(key_of(exp, r["expid"]), []).append(r)
    return out


# --------------------------------------------------------------------------- partition


def test_digest(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def make_partition(keys: Sequence[str], seed: int = SPLIT_SEED) -> dict[str, Any]:
    ks = sorted(set(keys))
    order = stream(seed, "phase3_affect_partition", DATASET).permutation(len(ks))
    n_dev = int(round(FRACTIONS["dev"] * len(ks)))
    n_tune = int(round(FRACTIONS["tune"] * len(ks)))
    parts = {
        "dev": sorted(ks[i] for i in order[:n_dev]),
        "tune": sorted(ks[i] for i in order[n_dev : n_dev + n_tune]),
        "test": sorted(ks[i] for i in order[n_dev + n_tune :]),
    }
    return {
        "dataset": DATASET,
        "unit": "participant (experiment, expid)",
        "split_seed": seed,
        "fractions": FRACTIONS,
        "source": SOURCE,
        "counts": {p: len(v) for p, v in parts.items()},
        "test_partition_sha256": test_digest(parts["test"]),
        **parts,
    }


def load_partition(path: Path = PARTITION_FILE) -> dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if test_digest(raw["test"]) != raw["test_partition_sha256"]:
        raise ValueError(f"{path}: test digest mismatch — the frozen partition was modified")
    out: dict[str, str] = {}
    for part in PARTITIONS:
        for k in raw[part]:
            if k in out:
                raise ValueError(f"{path}: {k} in two partitions")
            out[k] = part
    return out


def write_partition(folder: Path, out: Path = PARTITION_FILE) -> dict[str, Any]:
    rec = make_partition(sorted(_read(folder)))
    body = ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in rec.items())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("{\n" + body + "\n}\n", encoding="utf-8")
    return rec


# --------------------------------------------------------------------------- encoding


def _social(events: list[tuple[int, int, int]], own: list[tuple[int, int]], n_steps: int) -> np.ndarray:
    """Social columns per own step.

    ``events``: demonstrations as (position, state, action) in row order; ``own``: own steps as
    (position, state). A demonstration counts for an own step iff its position is smaller.
    """
    out = np.zeros((n_steps, N_EXTRA), dtype=np.float32)
    last: dict[int, int] = {}
    n1: dict[int, int] = {}
    n: dict[int, int] = {}
    since: dict[int, int] = {}
    ei = 0
    for t, (pos, s) in enumerate(own):
        while ei < len(events) and events[ei][0] < pos:
            _, es, ea = events[ei]
            last[es] = ea
            n[es] = n.get(es, 0) + 1
            n1[es] = n1.get(es, 0) + ea
            since[es] = since.get(es, 0) + 1
            ei += 1
        if n.get(s, 0):
            out[t, 0] = 1.0
            out[t, 1 + last[s]] = 1.0
            out[t, 3] = n1[s] / n[s]
        else:
            out[t, 3] = 0.5
        out[t, 4] = since.get(s, 0) / 3.0
        since[s] = 0
    return out


def shuffle_events(
    events: list[tuple[int, int, int]], key: str, seed: int = SHUFFLE_SEED
) -> list[tuple[int, int, int]]:
    """Causal within-participant resampling: event ``k >= 1`` takes the action of a random earlier event."""
    rng = stream(seed, "social_past_shuffle")  # index-only schedule: key-independent (strict check)
    out: list[tuple[int, int, int]] = []
    for k, (pos, s, a) in enumerate(events):
        a2 = a if k == 0 else events[int(rng.integers(0, k))][2]
        out.append((pos, s, a2))
    return out


def encode_participant(key: str, rows: list[dict[str, str]]) -> EncodedSeq:
    L = Layout(K)
    own_rows = [(i, r) for i, r in enumerate(rows) if r["id"] != "dem"]
    expid = rows[0]["expid"]
    if any(r["expid"] != expid for r in rows):
        raise ValueError(f"{key}: mixed participants")
    flip: dict[int, int] = {}

    def label(state: int, cgb: str) -> int:
        if state not in flip:
            flip[state] = int(stream(LABEL_SEED, "najar_label_flip", key, state).integers(0, 2))
        return int(cgb) ^ flip[state]

    sessions = sorted({int(r["session"]) for r in rows})
    states_by_session = {
        s: sorted({int(r["state"]) for r in rows if int(r["session"]) == s}) for s in sessions
    }
    T = len(own_rows)
    x = np.zeros((T, D_BASE + 2 * N_EXTRA), dtype=np.float32)
    actions = np.zeros(T, dtype=np.int64)
    own_count: dict[int, int] = {}
    prev: tuple[int, float] | None = None
    prev_feat: np.ndarray | None = None
    prev_session: int | None = None
    events: list[tuple[int, int, int]] = []
    for i, r in enumerate(rows):
        if r["id"] == "dem":
            st = int(r["state"])
            events.append((i, st, label(st, r["cgb"])))
    own_pos: list[tuple[int, int]] = []
    for t, (i, r) in enumerate(own_rows):
        st, sess = int(r["state"]), int(r["session"])
        feat = np.zeros(12, dtype=np.float32)
        feat[states_by_session[sess].index(st)] = 1.0
        feat[3 + min(sessions.index(sess), 3)] = 1.0
        feat[7] = own_count.get(st, 0) / 20.0
        own_count[st] = own_count.get(st, 0) + 1
        x[t, L.feat] = feat
        x[t, L.avail] = 1.0
        if prev is not None and prev_feat is not None:
            a_prev, out_prev = prev
            x[t, L.prev_action.start + a_prev] = 1.0
            x[t, L.prev_outcomes.start + a_prev] = out_prev
            x[t, L.prev_observed.start + a_prev] = 1.0
            x[t, L.prev_reward] = out_prev
            x[t, L.prev_feedback] = 1.0
            x[t, L.prev_feat] = prev_feat
        x[t, L.block_start] = 1.0 if sess != prev_session else 0.0
        a = label(st, r["cgb"])
        actions[t] = a
        if r["out"] not in ("1", "-1"):
            raise ValueError(f"{key}: own choice without outcome")
        prev, prev_feat, prev_session = (a, 1.0 if r["out"] == "1" else 0.0), feat, sess
        own_pos.append((i, st))
    x[:, D_BASE : D_BASE + N_EXTRA] = _social(events, own_pos, T)
    x[:, D_BASE + N_EXTRA :] = _social(shuffle_events(events, key), own_pos, T)
    return EncodedSeq(
        key,
        DATASET,
        K,
        K,
        x,
        actions,
        np.ones((T, K), dtype=bool),
        np.ones(T, dtype=bool),
        meta={"n_demos": len(events), "n_extra": N_EXTRA},
    )


def load_najar(
    partitions: Sequence[str],
    *,
    experiment_id: str,
    folder: Path | None = None,
    partition_file: Path = PARTITION_FILE,
    freeze_dir: Path = P3_FREEZE_DIR,
) -> tuple[dict[str, list[EncodedSeq]], list[str]]:
    for part in partitions:
        check_partition_access(part, experiment_id, freeze_dir)
    assign = load_partition(partition_file)
    rows = _read(folder or data_dir())
    out: dict[str, list[EncodedSeq]] = {p: [] for p in partitions}
    notes: list[str] = []
    for key in sorted(rows):
        part = assign.get(key, "")
        if part not in out:
            continue  # other partitions (incl. test) are dropped before encoding
        if part == "test" and key in EXPOSED:
            notes.append(
                f"{key}: rows displayed before the partition was written (deviation DA-1); not scored"
            )
            continue
        out[part].append(encode_participant(key, rows[key]))
    return out, notes


if __name__ == "__main__":  # pragma: no cover
    import sys

    if sys.argv[1:2] == ["partition"]:
        r = write_partition(data_dir())
        print(json.dumps({k: r[k] for k in ("dataset", "split_seed", "counts", "test_partition_sha256")}))
