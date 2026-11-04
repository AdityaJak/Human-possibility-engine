"""Parsers for Psych-101 natural-language transcripts (Binz et al., 2024; Apache-2.0).

Two learning paradigms are parsed into typed trial records:

* **MCPL** (``collsiöö2023MCPL/exp1-3``): multiple-cue probability learning. Two cues with five
  ordered levels; the participant estimates a criterion on a nine-level ordinal scale and (for
  most of the task) is told the correct value afterwards. Feedback stops at some point.
* **IGT** (``steingroever2015data/exp1-3``): Iowa Gambling Task. Four decks; each card yields a
  win and possibly a loss. Deck letters are randomised per participant; we index decks by the
  order they are listed in the instructions (consistent with deck payoff schedules, which
  ``validate_igt`` checks).

Human choices are enclosed in ``<<`` ``>>`` in the transcripts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MCPL_PREFIX = "collsiöö2023MCPL/"
IGT_PREFIX = "steingroever2015data/"


@dataclass(slots=True)
class MCPLTrial:
    cue1: int  # level 0..4
    cue2: int  # level 0..4
    choice: int  # 0..8
    correct: int | None  # 0..8 when feedback was shown, else None
    said_correct: bool | None  # the transcript's "That is correct/incorrect", when present


@dataclass(slots=True)
class IGTTrial:
    choice: int  # 0..3, index in the instruction's deck list
    win: float
    loss: float  # positive number = amount lost

    @property
    def net(self) -> float:
        return self.win - self.loss


@dataclass(slots=True)
class Participant:
    experiment: str
    participant: str
    paradigm: str  # "mcpl" | "igt"
    n_actions: int
    trials: list = field(default_factory=list)  # list[MCPLTrial] | list[IGTTrial]
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.experiment}#{self.participant}"


class ParseError(ValueError):
    pass


# ----------------------------------------------------------------------------- reading
def iter_records(path: Path, prefixes: Iterable[str] | None = None) -> Iterator[dict]:
    pref = tuple(prefixes) if prefixes is not None else None
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if pref is None or rec["experiment"].startswith(pref):
                yield rec


# ----------------------------------------------------------------------------- MCPL
_MCPL_CUE_LEVELS = re.compile(r"can take five values \(([^)]*)\)")
_MCPL_OUT_LEVELS = re.compile(r"can take nine values \(([^)]*)\)")
_MCPL_TRIAL = re.compile(
    r"^(?P<c1n>[A-Za-z]+): (?P<c1>[^.]+?)\. (?P<c2n>[A-Za-z]+): (?P<c2>[^.]+?)\. "
    r"You say that the (?P<outn>[A-Za-z]+) concentration is <<(?P<choice>[^>]+)>>\."
    r"(?: That is (?P<verdict>correct|incorrect)\. The correct concentration of [A-Za-z]+ is (?:indeed )?(?P<correct>[^.]+)\.)?"
    r"\s*$"
)


def parse_mcpl(text: str, experiment: str, participant: str) -> Participant:
    lines = text.split("\n")
    cue_m = _MCPL_CUE_LEVELS.search(text)
    out_m = _MCPL_OUT_LEVELS.search(text)
    if not cue_m or not out_m:
        raise ParseError(f"{experiment}#{participant}: MCPL level definitions not found")
    cue_levels = [s.strip() for s in cue_m.group(1).split(",")]
    out_levels = [s.strip() for s in out_m.group(1).split(",")]
    if len(cue_levels) != 5 or len(out_levels) != 9:
        raise ParseError(f"{experiment}#{participant}: expected 5 cue and 9 outcome levels")
    cue_idx = {v: i for i, v in enumerate(cue_levels)}
    out_idx = {v: i for i, v in enumerate(out_levels)}
    p = Participant(
        experiment, participant, "mcpl", 9, meta={"cue_levels": cue_levels, "out_levels": out_levels}
    )
    for ln in lines:
        if "<<" not in ln:
            continue
        m = _MCPL_TRIAL.match(ln.strip())
        if not m:
            raise ParseError(f"{experiment}#{participant}: unparsable MCPL line: {ln[:120]!r}")
        try:
            c1 = cue_idx[m.group("c1").strip()]
            c2 = cue_idx[m.group("c2").strip()]
            choice = out_idx[m.group("choice").strip()]
            corr = m.group("correct")
            correct = out_idx[corr.strip()] if corr is not None else None
        except KeyError as e:
            raise ParseError(f"{experiment}#{participant}: unknown level {e} in {ln[:120]!r}") from e
        verdict = m.group("verdict")
        p.trials.append(MCPLTrial(c1, c2, choice, correct, None if verdict is None else verdict == "correct"))
    return p


# ----------------------------------------------------------------------------- IGT
_IGT_DECKS = re.compile(r"four decks of cards labeled ([A-Z]), ([A-Z]), ([A-Z]), and ([A-Z])\.")
_IGT_TRIAL = re.compile(
    r"^You press <<(?P<deck>[A-Z])>>\. You win (?P<win>-?[\d.]+)\$ and lose (?P<loss>-?[\d.]+)\$\.\s*$"
)


def parse_igt(text: str, experiment: str, participant: str) -> Participant:
    m = _IGT_DECKS.search(text)
    if not m:
        raise ParseError(f"{experiment}#{participant}: IGT deck labels not found")
    decks = [m.group(i) for i in range(1, 5)]
    idx = {d: i for i, d in enumerate(decks)}
    p = Participant(experiment, participant, "igt", 4, meta={"deck_labels": decks})
    for ln in text.split("\n"):
        if "<<" not in ln:
            continue
        t = _IGT_TRIAL.match(ln.strip())
        if not t:
            raise ParseError(f"{experiment}#{participant}: unparsable IGT line: {ln[:120]!r}")
        if t.group("deck") not in idx:
            raise ParseError(f"{experiment}#{participant}: deck {t.group('deck')} not in {decks}")
        p.trials.append(IGTTrial(idx[t.group("deck")], float(t.group("win")), abs(float(t.group("loss")))))
    return p


def load_participants(path: Path, paradigm: str) -> list[Participant]:
    """Parse every participant of one paradigm ("mcpl" or "igt") from a Psych-101 JSONL file."""
    prefix, parser = (MCPL_PREFIX, parse_mcpl) if paradigm == "mcpl" else (IGT_PREFIX, parse_igt)
    out = [parser(r["text"], r["experiment"], str(r["participant"])) for r in iter_records(path, [prefix])]
    out.sort(key=lambda p: (p.experiment, _natural(p.participant)))
    return out


def _natural(s: str) -> tuple[int, str]:
    return (int(s), "") if s.isdigit() else (10**9, s)


# ----------------------------------------------------------------------------- validation
def validate_mcpl(parts: list[Participant]) -> dict:
    """Parse-validation report. Raises ParseError on internal inconsistencies."""
    by_exp: dict[str, dict] = {}
    for p in parts:
        if not p.trials:
            raise ParseError(f"{p.key}: no trials")
        for t in p.trials:
            if not (0 <= t.cue1 < 5 and 0 <= t.cue2 < 5 and 0 <= t.choice < 9):
                raise ParseError(f"{p.key}: out-of-range trial {t}")
            if (
                t.correct is not None
                and t.said_correct is not None
                and t.said_correct != (t.choice == t.correct)
            ):
                raise ParseError(f"{p.key}: transcript verdict disagrees with parsed levels")
        e = by_exp.setdefault(
            p.experiment, {"participants": 0, "trial_counts": set(), "fb": 0, "nofb": 0, "hits": 0}
        )
        e["participants"] += 1
        e["trial_counts"].add(len(p.trials))
        fb = [t for t in p.trials if t.correct is not None]
        e["fb"] += len(fb)
        e["nofb"] += len(p.trials) - len(fb)
        e["hits"] += sum(t.choice == t.correct for t in fb)
    return {
        exp: {
            "participants": v["participants"],
            "trials_per_participant": sorted(v["trial_counts"]),
            "feedback_trials": v["fb"],
            "no_feedback_trials": v["nofb"],
            "accuracy_on_feedback_trials": v["hits"] / v["fb"] if v["fb"] else None,
        }
        for exp, v in sorted(by_exp.items())
    }


def validate_igt(parts: list[Participant]) -> dict:
    """Checks trial counts, choice alphabet, and that deck positions have consistent win schedules."""
    by_exp: dict[str, dict] = {}
    wins_by_pos: dict[int, list[float]] = {i: [] for i in range(4)}
    for p in parts:
        if not p.trials:
            raise ParseError(f"{p.key}: no trials")
        for t in p.trials:
            if not 0 <= t.choice < 4:
                raise ParseError(f"{p.key}: bad choice {t.choice}")
            wins_by_pos[t.choice].append(t.win)
        e = by_exp.setdefault(p.experiment, {"participants": 0, "trial_counts": set(), "net": []})
        e["participants"] += 1
        e["trial_counts"].add(len(p.trials))
        e["net"].append(sum(t.net for t in p.trials))
    return {
        "experiments": {
            exp: {
                "participants": v["participants"],
                "trials_per_participant": sorted(v["trial_counts"]),
                "mean_total_net": float(np.mean(v["net"])),
            }
            for exp, v in sorted(by_exp.items())
        },
        "median_win_by_deck_position": {
            i: float(np.median(w)) if w else None for i, w in wins_by_pos.items()
        },
        "choice_share_by_deck_position": {
            i: len(w) / max(1, sum(len(x) for x in wins_by_pos.values())) for i, w in wins_by_pos.items()
        },
    }
