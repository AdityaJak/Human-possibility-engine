"""HeartSteps V1 decision-point features and the pre-listed effect moderators.

Moderators were listed in the directive before any analysis: time of day, weekday/weekend, prior
30-minute steps, location (home), and day-in-study (decay). Each is encoded as one centred or binary
column so that the moderated WCLS coefficient has a plain reading.

Local time = ``sugg.decision.utime`` (UTC) + ``sugg.gmtoff`` seconds, per the dataset's time columns.

Analysis sample (same as Phase I): available decision points (``avail``) with a recorded ``send`` and
30-minute outcome/pre-outcome. Note on ``is.randomized``: the dataset documentation defines ``send``
as "the time point must be randomly selected (...) and the person must be available"; in the data,
among available points ``is.randomized == True`` coincides with ``send == True`` (3,908 rows) and
``is.randomized == False`` with ``send == False`` (2,682 rows). The column therefore records the
realised selection to send, not whether a point was randomised; every available point was a
randomised decision point with P(send) ≈ 0.6 (empirical 0.594). It must not be used as a filter.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

MODERATOR_NAMES = ("evening", "weekend", "log_pre_c", "home", "day_c")
MODERATOR_DESCRIPTIONS = {
    "evening": "decision point at local hour >= 17 (time of day)",
    "weekend": "local weekday is Saturday or Sunday",
    "log_pre_c": "log(steps in the 30 min before + 0.5), centred at the sample mean (prior activity)",
    "home": "dec.location.category == 'home'",
    "day_c": "study day / 41 - 0.5 (day-in-study decay; 6-week trial)",
}


@dataclass(slots=True)
class DecisionRow:
    user: int
    decision_index: int
    study_day: int
    available: bool
    is_randomized: bool
    send: bool | None
    send_walking: bool | None
    send_sedentary: bool | None
    steps30: float | None
    steps30_pre: float | None
    steps30_complete: float | None
    local_hour: float | None
    weekday: int | None
    location: str


def _b(v: str) -> bool | None:
    return {"True": True, "False": False}.get(v)


def _f(v: str) -> float | None:
    return float(v) if v not in ("", "NA") else None


def _local(utime: str, gmtoff: str) -> tuple[float | None, int | None]:
    if utime in ("", "NA") or gmtoff in ("", "NA"):
        return None, None
    t = datetime.strptime(utime[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC) + timedelta(
        seconds=float(gmtoff)
    )
    return t.hour + t.minute / 60.0, t.weekday()


def load_rows(path: Path) -> list[DecisionRow]:
    out: list[DecisionRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            di = int(r["decision.index"])
            hour, wd = _local(r["sugg.decision.utime"], r["sugg.gmtoff"])
            out.append(
                DecisionRow(
                    user=int(r["user.index"]),
                    decision_index=di,
                    study_day=di // 5,
                    available=r["avail"] == "True",
                    is_randomized=r["is.randomized"] == "True",
                    send=_b(r["send"]),
                    send_walking=_b(r["send.active"]),
                    send_sedentary=_b(r["send.sedentary"]),
                    steps30=_f(r["jbsteps30.zero"]),
                    steps30_pre=_f(r["jbsteps30pre.zero"]),
                    steps30_complete=_f(r["jbsteps30"]),
                    local_hour=hour,
                    weekday=wd,
                    location=r["dec.location.category"],
                )
            )
    out.sort(key=lambda d: (d.user, d.decision_index))
    return out


@dataclass(slots=True)
class Frame:
    """Analysis sample as aligned arrays (one row per available decision point)."""

    users: np.ndarray
    y: np.ndarray  # log(steps30 + 0.5)
    steps: np.ndarray
    a: np.ndarray  # any suggestion delivered (0/1)
    a_walk: np.ndarray
    a_sed: np.ndarray
    log_pre: np.ndarray
    study_day: np.ndarray
    moderators: dict[str, np.ndarray]
    is_randomized: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def controls(self) -> np.ndarray:
        return np.column_stack([np.ones(len(self)), self.log_pre, self.study_day])

    def subset(self, mask: np.ndarray) -> Frame:
        return Frame(
            self.users[mask],
            self.y[mask],
            self.steps[mask],
            self.a[mask],
            self.a_walk[mask],
            self.a_sed[mask],
            self.log_pre[mask],
            self.study_day[mask],
            {k: v[mask] for k, v in self.moderators.items()},
            self.is_randomized[mask],
        )


def build_frame(rows: list[DecisionRow], *, users: list[int] | None = None) -> Frame:
    """Available points with recorded send and outcome data, optionally restricted to a user set."""
    allowed = None if users is None else set(users)
    keep = [
        d
        for d in rows
        if d.available
        and d.send is not None
        and d.steps30 is not None
        and d.steps30_pre is not None
        and d.local_hour is not None
        and (allowed is None or d.user in allowed)
    ]
    steps = np.array([d.steps30 for d in keep], dtype=float)
    log_pre = np.log(np.array([d.steps30_pre for d in keep], dtype=float) + 0.5)
    day = np.array([d.study_day for d in keep], dtype=float)
    hour = np.array([d.local_hour for d in keep], dtype=float)
    wd = np.array([d.weekday for d in keep], dtype=int)
    mods = {
        "evening": (hour >= 17).astype(float),
        "weekend": (wd >= 5).astype(float),
        "log_pre_c": log_pre - log_pre.mean(),
        "home": np.array([d.location == "home" for d in keep], dtype=float),
        "day_c": day / 41.0 - 0.5,
    }
    return Frame(
        users=np.array([d.user for d in keep]),
        y=np.log(steps + 0.5),
        steps=steps,
        a=np.array([float(bool(d.send)) for d in keep]),
        a_walk=np.array([float(bool(d.send and d.send_walking)) for d in keep]),
        a_sed=np.array([float(bool(d.send and d.send_sedentary)) for d in keep]),
        log_pre=log_pre,
        study_day=day,
        moderators=mods,
        is_randomized=np.array([d.is_randomized for d in keep]),
    )


def describe(frame: Frame) -> dict:
    return {
        "n_obs": int(len(frame)),
        "n_users": int(len(np.unique(frame.users))),
        "p_send": float(frame.a.mean()),
        "p_walk": float(frame.a_walk.mean()),
        "p_sed": float(frame.a_sed.mean()),
        "mean_steps30": float(frame.steps.mean()),
        "is_randomized_equals_send": bool(np.array_equal(frame.is_randomized.astype(float), frame.a)),
        "moderator_means": {k: float(v.mean()) for k, v in frame.moderators.items()},
    }
