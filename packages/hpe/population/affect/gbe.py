"""Adapter for the Great Brain Experiment risky-decision-and-happiness data (Gate F, exploratory).

Source (opened 2026-11-04): Rutledge, R. B. "Risky decision and happiness task: The Great Brain Experiment
smartphone app", Dryad doi:10.5061/dryad.prr4xgxkk (licence CC0-1.0), downloaded from its Zenodo mirror
record 4421847 (licence ``cc-zero``), file ``Rutledge_risk_and_happiness_task_2023.zip``. 47,067
participants, 91,058 plays of 30 trials; each play has a happiness rating ("How happy are you right
now?", 0-100) before trial 1 and after every 2-3 trials. Data files are never committed; set
``HPE_GBE_MAT`` to the extracted ``Rutledge_GBE_risk_data_TOD.mat``.

Encoding (``K = 2``: action 0 = safe option, 1 = gamble): the first ``input_dim(2)`` columns of ``x``
follow encoding v3 exactly; :data:`N_EXTRA` affect columns are appended after them and are used only by
models that ask for them:

    extra[0]  most recent happiness rating available *before* this choice, / 100
    extra[1]  1 if that rating was given since the previous choice (the pre-play rating counts for trial 1)
    extra[2]  trials since that rating, / 3

The rating stored on row ``t`` of a play (made after trial ``t``'s outcome) first appears on row ``t + 1``;
the current trial's own rating is never an input. Test-partition participants are dropped right after
the ``.mat`` is read, before any record is encoded, unless a freeze record exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hpe.core.rng import stream
from hpe.evaluation.human.partitions import PARTITIONS, check_partition_access
from hpe.population.core.data import P3_FREEZE_DIR
from hpe.population.core.encoding import F, EncodedSeq, Layout, input_dim
from hpe.population.core.families import REPO_ROOT

DATASET = "gbe_risk_happiness"
DATA_ENV = "HPE_GBE_MAT"
SOURCE = {
    "title": "Risky decision and happiness task: The Great Brain Experiment smartphone app",
    "author": "Robb B. Rutledge",
    "dryad_doi": "10.5061/dryad.prr4xgxkk",
    "zenodo_record": "https://zenodo.org/records/4421847",
    "licence": "CC0-1.0 (Dryad: https://spdx.org/licenses/CC0-1.0.html; Zenodo metadata: cc-zero)",
    "zip": "Rutledge_risk_and_happiness_task_2023.zip",
    "zip_md5": "899fb2380a25b208bf9a445e0c06b9d8",
    "zip_sha256": "456859176aece7ac40dc2303aec210490a90bb5d395f713ddf75ffb91fb82168",
    "mat": "Rutledge_GBE_risk_data_TOD.mat",
    "mat_sha256": "b653513277ea27e89daee94806cae012ea7cfcb34de2383b1b80c7b01476ac4e",
    "accessed": "2026-11-04",
}
PARTITION_FILE = REPO_ROOT / "reports" / "phase3" / "affect" / "partitions_gbe.json"
SPLIT_SEED = 20261108
FRACTIONS = {"dev": 0.4, "tune": 0.2, "test": 0.4}
K = 2
D_BASE = input_dim(K)
N_EXTRA = 3
MAX_PLAYS = 10  # plays per participant kept (in order); 99th percentile of plays is 12
TRIALS_PER_PLAY = 30
VALUE_SCALE = 150.0  # largest absolute amount in any design

# raw data columns (README table, 1-based in MATLAB -> 0-based here)
C_SAFE, C_WIN, C_LOSS, C_CHOICE, C_OUTCOME, C_HAPPY = 2, 3, 4, 6, 7, 9


class GBEDataUnavailable(RuntimeError):
    pass


def key_of(pid: int) -> str:
    return f"{DATASET}#{int(pid)}"


# --------------------------------------------------------------------------- partition


def make_partition(ids: Iterable[int], seed: int = SPLIT_SEED) -> dict[str, Any]:
    """Seeded participant-level split into dev / tune / test (fractions :data:`FRACTIONS`)."""
    ids_sorted = sorted({int(i) for i in ids})
    n = len(ids_sorted)
    order = stream(seed, "phase3_affect_partition", DATASET).permutation(n)
    n_dev = int(round(FRACTIONS["dev"] * n))
    n_tune = int(round(FRACTIONS["tune"] * n))
    parts = {
        "dev": sorted(ids_sorted[i] for i in order[:n_dev]),
        "tune": sorted(ids_sorted[i] for i in order[n_dev : n_dev + n_tune]),
        "test": sorted(ids_sorted[i] for i in order[n_dev + n_tune :]),
    }
    return {
        "dataset": DATASET,
        "unit": "participant (subjData.id)",
        "split_seed": seed,
        "fractions": FRACTIONS,
        "source": SOURCE,
        "counts": {p: len(v) for p, v in parts.items()},
        "test_partition_sha256": test_digest(parts["test"]),
        **parts,
    }


def test_digest(test_ids: Iterable[int]) -> str:
    keys = sorted(key_of(i) for i in test_ids)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def load_partition(path: Path = PARTITION_FILE) -> dict[int, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if test_digest(raw["test"]) != raw["test_partition_sha256"]:
        raise ValueError(f"{path}: test digest mismatch — the frozen partition was modified")
    out: dict[int, str] = {}
    for part in PARTITIONS:
        for pid in raw[part]:
            if int(pid) in out:
                raise ValueError(f"{path}: participant {pid} in two partitions")
            out[int(pid)] = part
    return out


# --------------------------------------------------------------------------- raw data


@dataclass(slots=True)
class GBEParticipant:
    pid: int
    plays: list[np.ndarray]  # each [30, 15] float64, raw README columns
    design: list[int]  # designVersion per play


def mat_path() -> Path:
    p = os.environ.get(DATA_ENV)
    if not p or not Path(p).exists():
        raise GBEDataUnavailable(
            f"set {DATA_ENV} to {SOURCE['mat']} extracted from {SOURCE['zip']} ({SOURCE['zenodo_record']})"
        )
    return Path(p)


def verify_mat(path: Path) -> None:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if h != SOURCE["mat_sha256"]:
        raise GBEDataUnavailable(f"{path}: sha256 {h} != expected {SOURCE['mat_sha256']}")


def _as_list(v: Any) -> list[Any]:
    if isinstance(v, np.ndarray) and v.dtype == object:
        return list(v.ravel())
    return [v]


def read_ids(path: Path) -> list[int]:
    import scipy.io as sio

    verify_mat(path)
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False, variable_names=["subjData"])
    return [int(s.id) for s in m["subjData"]]


def read_participants(path: Path, keep: set[int]) -> list[GBEParticipant]:
    """Read only participants whose id is in ``keep``; every other record is discarded unread."""
    import scipy.io as sio

    verify_mat(path)
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False, variable_names=["subjData"])
    out: list[GBEParticipant] = []
    for s in m["subjData"]:
        pid = int(s.id)
        if pid not in keep:
            continue
        plays = [np.asarray(d, dtype=np.float64) for d in _as_list(s.data)]
        design = [int(v) for v in np.atleast_1d(s.designVersion)]
        out.append(GBEParticipant(pid, plays, design))
    del m
    out.sort(key=lambda p: p.pid)
    return out


def check_play(play: np.ndarray) -> str | None:
    """Return a reason if a play violates the README's invariants, else None."""
    if play.shape != (TRIALS_PER_PLAY, 15):
        return f"shape {play.shape}"
    ch = play[:, C_CHOICE]
    if not np.all(np.isin(ch, (0.0, 1.0))):
        return "choice not in {0,1}"
    safe, win, loss, out = play[:, C_SAFE], play[:, C_WIN], play[:, C_LOSS], play[:, C_OUTCOME]
    if np.any((ch == 0) & (out != safe)):
        return "safe choice with outcome != safe value"
    if np.any((ch == 1) & (out != win) & (out != loss)):
        return "gamble outcome not one of its two values"
    vals = np.concatenate([safe, win, loss])
    if not np.all(np.isfinite(vals)) or np.abs(vals).max() > VALUE_SCALE:
        return "option values missing or out of range"
    h = play[:, C_HAPPY]
    rated = np.isfinite(h)
    if not rated[0]:
        return "no pre-play rating"
    if np.any((h[rated] < 0) | (h[rated] > 100)):
        return "rating outside 0-100"
    return None


# --------------------------------------------------------------------------- encoding


def _features(row: np.ndarray, t_in_play: int, play_idx: int, design: int) -> np.ndarray:
    safe, win, loss = row[C_SAFE], row[C_WIN], row[C_LOSS]
    ev = 0.5 * (win + loss)
    f = np.zeros(F, dtype=np.float32)
    f[0] = safe / VALUE_SCALE
    f[1] = win / VALUE_SCALE
    f[2] = loss / VALUE_SCALE
    f[3] = ev / VALUE_SCALE
    f[4] = (ev - safe) / VALUE_SCALE
    f[5] = float(safe > 0)
    f[6] = float(safe == 0)
    f[7] = float(safe < 0)
    f[8] = t_in_play / (TRIALS_PER_PLAY - 1)
    f[9] = min(play_idx, MAX_PLAYS - 1) / (MAX_PLAYS - 1)
    f[10] = float(design >= 3)  # "uncorrelated" design (versions 3-5) vs "ratio" design (1-2)
    f[11] = (win - loss) / (2 * VALUE_SCALE)
    return f


def _norm_outcome(v: float) -> float:
    return float((v + VALUE_SCALE) / (2 * VALUE_SCALE))


def encode_participant(p: GBEParticipant, max_plays: int = MAX_PLAYS) -> tuple[EncodedSeq | None, list[str]]:
    """Encode one participant's valid plays (in order, at most ``max_plays``) with affect columns."""
    notes: list[str] = []
    plays: list[tuple[np.ndarray, int]] = []
    for i, (play, des) in enumerate(zip(p.plays, p.design, strict=True)):
        why = check_play(play)
        if why is not None:
            notes.append(f"{key_of(p.pid)} play {i}: excluded ({why})")
            continue
        plays.append((play, des))
        if len(plays) == max_plays:
            break
    if not plays:
        return None, notes
    L = Layout(K)
    T = TRIALS_PER_PLAY * len(plays)
    x = np.zeros((T, D_BASE + N_EXTRA), dtype=np.float32)
    actions = np.zeros(T, dtype=np.int64)
    avail = np.ones((T, K), dtype=bool)
    scored = np.ones(T, dtype=bool)
    ratings = np.full(T, np.nan)  # rating made after the outcome of row t (row 0 of a play: before trial 1)
    t = 0
    prev_feat: np.ndarray | None = None
    prev_row: np.ndarray | None = None
    for pi, (play, des) in enumerate(plays):
        last_rating, since = float("nan"), 0
        for tp in range(TRIALS_PER_PLAY):
            row = play[tp]
            feat = _features(row, tp, pi, des)
            x[t, L.feat] = feat
            x[t, L.avail] = 1.0
            if prev_row is not None and prev_feat is not None:
                a_prev = int(prev_row[C_CHOICE])
                x[t, F + K + a_prev] = 1.0
                po = L.prev_outcomes.start
                pm = L.prev_observed.start
                x[t, po + 0] = _norm_outcome(prev_row[C_SAFE])  # the safe amount is always shown
                x[t, pm + 0] = 1.0
                if a_prev == 1:  # the spinner is spun only when the gamble is chosen
                    x[t, po + 1] = _norm_outcome(prev_row[C_OUTCOME])
                    x[t, pm + 1] = 1.0
                x[t, L.prev_reward] = _norm_outcome(prev_row[C_OUTCOME])
                x[t, L.prev_feedback] = 1.0
                x[t, L.prev_feat] = prev_feat
            x[t, L.block_start] = 1.0 if tp == 0 else 0.0
            # affect: the latest rating made before this choice
            if tp == 0:
                last_rating, since, fresh = float(row[C_HAPPY]), 0, 1.0  # pre-play rating (README)
            elif np.isfinite(play[tp - 1, C_HAPPY]) and tp - 1 > 0:
                last_rating, since, fresh = float(play[tp - 1, C_HAPPY]), 0, 1.0
            else:
                since += 1
                fresh = 0.0
            x[t, D_BASE + 0] = last_rating / 100.0
            x[t, D_BASE + 1] = fresh
            x[t, D_BASE + 2] = since / 3.0
            actions[t] = int(row[C_CHOICE])
            ratings[t] = row[C_HAPPY]
            prev_feat, prev_row = feat, row
            t += 1
    seq = EncodedSeq(
        key_of(p.pid),
        DATASET,
        K,
        K,
        x,
        actions,
        avail,
        scored,
        meta={"n_plays": len(plays), "ratings": ratings, "n_extra": N_EXTRA},
    )
    return seq, notes


# --------------------------------------------------------------------------- locked loading


def load_gbe(
    partitions: Sequence[str],
    *,
    experiment_id: str,
    path: Path | None = None,
    partition_file: Path = PARTITION_FILE,
    freeze_dir: Path = P3_FREEZE_DIR,
    max_plays: int = MAX_PLAYS,
) -> tuple[dict[str, list[EncodedSeq]], list[str]]:
    """Encoded sequences per requested partition. ``test`` needs a freeze record for ``experiment_id``."""
    for part in partitions:
        check_partition_access(part, experiment_id, freeze_dir)
    assign = load_partition(partition_file)
    keep = {pid for pid, part in assign.items() if part in set(partitions)}
    parts = read_participants(path or mat_path(), keep)
    out: dict[str, list[EncodedSeq]] = {p: [] for p in partitions}
    notes: list[str] = []
    for p in parts:
        seq, n = encode_participant(p, max_plays)
        notes.extend(n)
        if seq is None:
            notes.append(f"{key_of(p.pid)}: no valid play")
            continue
        out[assign[p.pid]].append(seq)
    for v in out.values():
        v.sort(key=lambda s: s.key)
    return out, notes


def subsample(seqs: Sequence[EncodedSeq], n: int, seed: int, tag: str) -> list[EncodedSeq]:
    if n <= 0 or n >= len(seqs):
        return list(seqs)
    idx = stream(seed, "phase3_affect_subsample", tag).choice(len(seqs), size=n, replace=False)
    return [seqs[i] for i in sorted(idx)]


def write_partition(path: Path, out: Path = PARTITION_FILE) -> dict[str, Any]:
    rec = make_partition(read_ids(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    body = ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in rec.items())
    out.write_text("{\n" + body + "\n}\n", encoding="utf-8")
    return rec


if __name__ == "__main__":  # pragma: no cover
    import sys

    if sys.argv[1:2] == ["partition"]:
        r = write_partition(mat_path())
        print(json.dumps({k: r[k] for k in ("dataset", "split_seed", "counts", "test_partition_sha256")}))
