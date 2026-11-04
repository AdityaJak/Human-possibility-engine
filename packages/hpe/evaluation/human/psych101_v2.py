"""Phase II parsers for five further Psych-101 learning families (Binz et al., 2024; Apache-2.0).

Every family is parsed into ``GenericTrial`` records with a 12-feature context already scaled to
[-1, 1], a discrete choice, and a feedback specification, so one encoder (``models2.encode_generic``)
drives every model through the same ``Policy`` protocol. Instructed trials (horizon task) are kept
as ``forced=True``: models learn from them but no likelihood is counted.

Family (paradigm key) -> experiment prefixes -> what is parsed
    bandit2   gershman2018deconstructing/  two slot machines (one risky, one safe), 20 games x 10 trials
    srl3      gershman2020reward/          stimulus-response learning: 6 stimuli -> 3 responses, 13 games
    horizon   wilson2014humans/, waltz2020differential/   4 instructed + 1 or 6 free trials per game
    casino    lefebvre2017behavioural/     4 contexts ("casinos") x 2 machines, probabilistic 0/0.5 or ±0.5
    wpt       speekenbrink2008learning/    weather prediction: 1-3 of 4 cards -> rainy/fine, corrective feedback
    garcia    garcia2023experiential/      part 1: letter pairs with counterfactual feedback; part 2:
                                           letter vs described option, no feedback; part 3 (judgements) excluded
    described plonsky2018when/             described gambles, 5 no-feedback + 20 full-feedback repetitions

Reward normalisation (documented per family, identical for every model): rewards are mapped to
[0, 1] by a fixed affine map of the family's stated outcome range.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from hpe.evaluation.human.psych101 import ParseError, Participant

N_FEATURES = 12

PARADIGM_PREFIXES: dict[str, tuple[str, ...]] = {
    "bandit2": ("gershman2018deconstructing/",),
    "srl3": ("gershman2020reward/",),
    "horizon": ("wilson2014humans/", "waltz2020differential/"),
    "casino": ("lefebvre2017behavioural/",),
    "wpt": ("speekenbrink2008learning/",),
    "garcia": ("garcia2023experiential/",),
    "described": ("plonsky2018when/",),
}
PARADIGM_ACTIONS: dict[str, int] = {
    "bandit2": 2,
    "srl3": 3,
    "horizon": 2,
    "casino": 2,
    "wpt": 2,
    "garcia": 2,
    "described": 2,
}


@dataclass(slots=True)
class GenericTrial:
    choice: int
    n_actions: int
    features: tuple[float, ...]  # length N_FEATURES, in [-1, 1]
    mode: str  # "none" | "bandit" | "full"
    reward: float | None = None  # normalised to [0, 1]
    correct: int | None = None  # full feedback: correct / best-in-hindsight action
    rewards_all: tuple[float, ...] | None = None
    forced: bool = False
    block: int = 0
    phase: str = ""
    meta: dict = field(default_factory=dict)


def _pad(xs: Iterable[float]) -> tuple[float, ...]:
    v = [float(np.clip(x, -1.0, 1.0)) for x in xs]
    if len(v) > N_FEATURES:
        raise ValueError(f"too many features: {len(v)}")
    return tuple(v + [0.0] * (N_FEATURES - len(v)))


def _unit(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0)) if hi > lo else 0.5


def _code(symbol: str, dims: int, salt: str = "") -> list[float]:
    """Deterministic ±1 code for an arbitrary symbol (letters differ per participant)."""
    h = hashlib.sha256(f"{salt}:{symbol}".encode()).digest()
    return [1.0 if (h[i // 8] >> (i % 8)) & 1 else -1.0 for i in range(dims)]


# ----------------------------------------------------------------------------- bandit2 (gershman2018)
_B2_MACHINES = re.compile(r"two slot machines labeled ([A-Z]) and ([A-Z])\.")
_B2_SAFE = re.compile(r"machine ([A-Z]) will always give 0 points")
_B2_TRIAL = re.compile(r"^You press <<([A-Z])>> and get (-?\d+) points\.\s*$")
_GAME = re.compile(r"^Game (\d+)[:.]")


def parse_bandit2(text: str, experiment: str, participant: str) -> Participant:
    m = _B2_MACHINES.search(text)
    if not m:
        raise ParseError(f"{experiment}#{participant}: bandit2 machine labels not found")
    labels = [m.group(1), m.group(2)]
    safe = _B2_SAFE.search(text)
    safe_label = safe.group(1) if safe else None
    idx = {lab: i for i, lab in enumerate(labels)}
    p = Participant(experiment, participant, "bandit2", 2, meta={"labels": labels, "safe": safe_label})
    block, t_in = -1, 0
    for ln in text.split("\n"):
        g = _GAME.match(ln)
        if g:
            block, t_in = int(g.group(1)) - 1, 0
            continue
        if "<<" not in ln:
            continue
        t = _B2_TRIAL.match(ln.strip())
        if not t or t.group(1) not in idx:
            raise ParseError(f"{experiment}#{participant}: unparsable bandit2 line {ln[:100]!r}")
        pts = int(t.group(2))
        feats = _pad([1.0, _unit(t_in, 0, 9) * 2 - 1, 1.0 if t_in == 0 else -1.0])
        p.trials.append(
            GenericTrial(
                idx[t.group(1)],
                2,
                feats,
                "bandit",
                _unit(pts, -3, 3),
                block=max(block, 0),
                meta={"points": pts},
            )
        )
        t_in += 1
    return p


# ----------------------------------------------------------------------------- srl3 (gershman2020)
_S3_RESP = re.compile(r"The three responses available are ([A-Z]), ([A-Z]), and ([A-Z])\.")
_S3_TRIAL = re.compile(r"^You see stimulus (\d+)\. You press <<([A-Z])>> and get (-?\d+) points\.\s*$")


def parse_srl3(text: str, experiment: str, participant: str) -> Participant:
    m = _S3_RESP.search(text)
    if not m:
        raise ParseError(f"{experiment}#{participant}: srl3 responses not found")
    labels = [m.group(i) for i in (1, 2, 3)]
    idx = {lab: i for i, lab in enumerate(labels)}
    p = Participant(experiment, participant, "srl3", 3, meta={"labels": labels})
    block = 0
    for ln in text.split("\n"):
        g = _GAME.match(ln)
        if g:
            block = int(g.group(1)) - 1
            continue
        if "<<" not in ln:
            continue
        t = _S3_TRIAL.match(ln.strip())
        if not t or t.group(2) not in idx:
            raise ParseError(f"{experiment}#{participant}: unparsable srl3 line {ln[:100]!r}")
        stim = int(t.group(1))
        onehot = [-1.0] * 8
        if 0 <= stim < 8:
            onehot[stim] = 1.0
        pts = int(t.group(3))  # 1 correct, 0 incorrect; some variants penalise with -1
        p.trials.append(
            GenericTrial(
                idx[t.group(2)],
                3,
                _pad([*onehot, 1.0]),
                "bandit",
                _unit(pts, -1, 1),
                block=block,
                meta={"stimulus": stim, "points": pts},
            )
        )
    return p


# ----------------------------------------------------------------------------- horizon (wilson2014, waltz2020)
_HZ_MACHINES = re.compile(r"two slot machines, labeled ([A-Z]) and ([A-Z])\.")
_HZ_GAME = re.compile(r"^Game (\d+)\. There are (\d+) trials in this game\.")
_HZ_FORCED = re.compile(r"^You are instructed to press ([A-Z]) and get (-?\d+) points\.\s*$")
_HZ_FREE = re.compile(r"^You press <<([A-Z])>> and get (-?\d+) points\.\s*$")


def parse_horizon(text: str, experiment: str, participant: str) -> Participant:
    m = _HZ_MACHINES.search(text)
    if not m:
        raise ParseError(f"{experiment}#{participant}: horizon machine labels not found")
    labels = [m.group(1), m.group(2)]
    idx = {lab: i for i, lab in enumerate(labels)}
    p = Participant(experiment, participant, "horizon", 2, meta={"labels": labels})
    block, horizon, t_in = -1, 5, 0
    for ln in text.split("\n"):
        g = _HZ_GAME.match(ln)
        if g:
            block, horizon, t_in = int(g.group(1)) - 1, int(g.group(2)), 0
            continue
        f = _HZ_FORCED.match(ln.strip())
        c = _HZ_FREE.match(ln.strip()) if not f else None
        if not f and not c:
            if "<<" in ln:
                raise ParseError(f"{experiment}#{participant}: unparsable horizon line {ln[:100]!r}")
            continue
        mm = f or c
        assert mm is not None
        lab, pts = mm.group(1), int(mm.group(2))
        if lab not in idx:
            raise ParseError(f"{experiment}#{participant}: machine {lab} not in {labels}")
        feats = _pad(
            [1.0, 1.0 if horizon >= 10 else -1.0, _unit(t_in, 0, 9) * 2 - 1, 1.0 if t_in == 0 else -1.0]
        )
        p.trials.append(
            GenericTrial(
                idx[lab],
                2,
                feats,
                "bandit",
                _unit(pts, 0, 100),
                forced=f is not None,
                block=max(block, 0),
                meta={"points": pts, "horizon": horizon},
            )
        )
        t_in += 1
    return p


# ----------------------------------------------------------------------------- casino (lefebvre2017)
_CS_RANGE = re.compile(r"return either (-?[\d.]+) or (-?[\d.]+) points")
_CS_TRIAL = re.compile(
    r"^You go to casino (\d)\. You can choose between machines ([A-Z]) and ([A-Z])\. You press <<([A-Z])>> and receive (-?[\d.]+) points\.\s*$"
)


def parse_casino(text: str, experiment: str, participant: str) -> Participant:
    rng_m = _CS_RANGE.search(text)
    if not rng_m:
        raise ParseError(f"{experiment}#{participant}: casino outcome range not found")
    lo, hi = sorted((float(rng_m.group(1)), float(rng_m.group(2))))
    p = Participant(experiment, participant, "casino", 2, meta={"outcomes": [lo, hi]})
    pairs: dict[int, tuple[str, str]] = {}
    for ln in text.split("\n"):
        if "<<" not in ln:
            continue
        t = _CS_TRIAL.match(ln.strip())
        if not t:
            raise ParseError(f"{experiment}#{participant}: unparsable casino line {ln[:100]!r}")
        casino = int(t.group(1))
        pair = pairs.setdefault(casino, (t.group(2), t.group(3)))
        if {t.group(2), t.group(3)} != set(pair):
            raise ParseError(f"{experiment}#{participant}: casino {casino} machines changed")
        if t.group(4) not in pair:
            raise ParseError(f"{experiment}#{participant}: machine {t.group(4)} not in casino {casino}")
        onehot = [-1.0] * 4
        onehot[casino - 1] = 1.0
        x = float(t.group(5))
        p.trials.append(
            GenericTrial(
                pair.index(t.group(4)),
                2,
                _pad([*onehot, 1.0]),
                "bandit",
                _unit(x, lo, hi),
                block=0,
                meta={"casino": casino, "points": x},
            )
        )
    p.meta["pairs"] = {str(k): list(v) for k, v in pairs.items()}
    return p


# ----------------------------------------------------------------------------- wpt (speekenbrink2008)
_WPT_KEYS = re.compile(
    r"predicts rainy weather \(by pressing ([A-Z])\) or fine weather \(by pressing ([A-Z])\)"
)
_WPT_TRIAL = re.compile(
    r"^You are seeing the following: (?P<cards>[^.]+)\. You press <<(?P<key>[A-Z])>>\. You are (?P<verdict>correct|wrong), the weather is (?:indeed )?(?P<weather>rainy|fine)\.\s*$"
)


def parse_wpt(text: str, experiment: str, participant: str) -> Participant:
    m = _WPT_KEYS.search(text)
    if not m:
        raise ParseError(f"{experiment}#{participant}: WPT keys not found")
    keys = [m.group(1), m.group(2)]  # index 0 = rainy, 1 = fine
    idx = {k: i for i, k in enumerate(keys)}
    p = Participant(experiment, participant, "wpt", 2, meta={"keys": keys})
    for ln in text.split("\n"):
        if "<<" not in ln:
            continue
        t = _WPT_TRIAL.match(ln.strip())
        if not t or t.group("key") not in idx:
            raise ParseError(f"{experiment}#{participant}: unparsable WPT line {ln[:100]!r}")
        cards = [-1.0] * 4
        for c in re.findall(r"card (\d)", t.group("cards")):
            cards[int(c) - 1] = 1.0
        choice = idx[t.group("key")]
        correct = 0 if t.group("weather") == "rainy" else 1
        if (t.group("verdict") == "correct") != (choice == correct):
            raise ParseError(f"{experiment}#{participant}: WPT verdict disagrees with parsed outcome")
        ra = (1.0, 0.0) if correct == 0 else (0.0, 1.0)
        p.trials.append(
            GenericTrial(
                choice,
                2,
                _pad([*cards, 1.0]),
                "full",
                float(choice == correct),
                correct,
                ra,
                meta={"cards": cards},
            )
        )
    return p


# ----------------------------------------------------------------------------- garcia (garcia2023)
_GA_P1 = re.compile(
    r"^You can choose between option ([A-Z]) and option ([A-Z])\. You press <<([A-Z])>> and get (-?[\d.]+) points\. You would have gotten (-?[\d.]+) points had you chosen option ([A-Z])(?: instead)?\.\s*$"
)
_GA_P2 = re.compile(
    r"^You can choose between option ([A-Z]) and the following option by pressing button ([A-Z]): (-?[\d.]+) point with ([\d.]+)% chance and (-?[\d.]+) point with ([\d.]+)% chance\. You press <<([A-Z])>>\.\s*$"
)
_GA_P3 = re.compile(r"^Q: What are the odds")


def parse_garcia(text: str, experiment: str, participant: str) -> Participant:
    p = Participant(experiment, participant, "garcia", 2, meta={"excluded_judgements": 0})
    salt = f"{experiment}#{participant}"
    for ln in text.split("\n"):
        if "<<" not in ln:
            continue
        s = ln.strip()
        if _GA_P3.match(s):
            p.meta["excluded_judgements"] += 1
            continue
        m1 = _GA_P1.match(s)
        if m1:
            a, b, pressed, got, other, other_lab = m1.groups()
            if pressed not in (a, b) or other_lab not in (a, b):
                raise ParseError(f"{experiment}#{participant}: garcia part-1 letters mismatch {s[:100]!r}")
            choice = 0 if pressed == a else 1
            own, alt = float(got), float(other)
            rewards = [own, alt] if choice == 0 else [alt, own]
            ra = tuple(_unit(r, -1, 1) for r in rewards)
            feats = _pad(_code(a, 4, salt) + _code(b, 4, salt) + [-1.0, 0.0, -1.0, 1.0])
            p.trials.append(
                GenericTrial(
                    choice,
                    2,
                    feats,
                    "full",
                    _unit(own, -1, 1),
                    int(np.argmax(ra)),
                    ra,
                    block=0,
                    phase="part1",
                )
            )
            continue
        m2 = _GA_P2.match(s)
        if m2:
            a, b, o1, p1, _o2, p2, pressed = m2.groups()
            if pressed not in (a, b):
                raise ParseError(f"{experiment}#{participant}: garcia part-2 letters mismatch {s[:100]!r}")
            p_win = (float(p1) if float(o1) > 0 else float(p2)) / 100.0
            feats = _pad(_code(a, 4, salt) + _code(b, 4, salt) + [1.0, p_win * 2 - 1, 1.0, 1.0])
            p.trials.append(
                GenericTrial(
                    0 if pressed == a else 1,
                    2,
                    feats,
                    "none",
                    block=1,
                    phase="part2",
                    meta={"p_win_described": p_win},
                )
            )
            continue
        raise ParseError(f"{experiment}#{participant}: unparsable garcia line {s[:100]!r}")
    return p


# ----------------------------------------------------------------------------- described (plonsky2018)
_PL_OPTION = re.compile(r"^Option ([A-Z]) delivers (.+)\.\s*$")
_PL_OUTCOME = re.compile(r"(-?\d+(?:\.\d+)?) points? with ([\d.]+)% chance")
_PL_NOFB = re.compile(r"^You press <<([A-Z])>>\.\s*$")
_PL_FB = re.compile(
    r"^You press <<([A-Z])>> and (gain|lose) (\d+(?:\.\d+)?) points\. You would have (gained|lost) (\d+(?:\.\d+)?) points had you chosen option ([A-Z])\.\s*$"
)


def _option_features(desc: str) -> tuple[list[float], dict]:
    unknown = "unknown" in desc
    pairs = [(float(v), float(pr) / 100.0) for v, pr in _PL_OUTCOME.findall(desc)]
    vals = (
        [float(v) for v in re.findall(r"(-?\d+(?:\.\d+)?) points?", desc)]
        if not pairs
        else [v for v, _ in pairs]
    )
    if not vals:
        raise ParseError(f"unparsable option description {desc[:80]!r}")
    if pairs and abs(sum(pr for _, pr in pairs) - 1.0) < 0.02:
        probs = [pr for _, pr in pairs]
    else:
        probs = [1.0 / len(vals)] * len(vals)  # unknown probabilities: uniform placeholder
    ev = float(sum(v * q for v, q in zip(vals, probs, strict=True)))
    sd = float(np.sqrt(sum(q * (v - ev) ** 2 for v, q in zip(vals, probs, strict=True))))
    p_gain = float(sum(q for v, q in zip(vals, probs, strict=True) if v > 0))
    feats = [
        ev / 100.0,
        sd / 100.0,
        min(vals) / 100.0,
        max(vals) / 100.0,
        p_gain * 2 - 1,
        1.0 if unknown else -1.0,
    ]
    return feats, {"ev": ev, "sd": sd, "unknown": unknown, "n_outcomes": len(vals)}


def parse_described(text: str, experiment: str, participant: str) -> Participant:
    p = Participant(experiment, participant, "described", 2, meta={})
    options: list[tuple[str, list[float], dict]] = []
    block = -1
    t_in = 0
    for ln in text.split("\n"):
        s = ln.strip()
        om = _PL_OPTION.match(s)
        if om:
            if len(options) >= 2:
                options = []
            if not options:
                block += 1
                t_in = 0
            ofeats, info = _option_features(om.group(2))
            options.append((om.group(1), ofeats, info))
            continue
        if "<<" not in s:
            continue
        if len(options) != 2:
            raise ParseError(f"{experiment}#{participant}: choice before two option descriptions")
        labels = [options[0][0], options[1][0]]
        if labels[0] == labels[1]:
            # Transcript defect: both options carry the same letter, so option identity is undecidable.
            # The whole problem is excluded and counted; never guessed.
            p.meta["ambiguous_problem_choices"] = p.meta.get("ambiguous_problem_choices", 0) + 1
            continue
        fvec = _pad(options[0][1] + options[1][1])
        nf = _PL_NOFB.match(s)
        if nf:
            if nf.group(1) not in labels:
                raise ParseError(f"{experiment}#{participant}: option {nf.group(1)} not in {labels}")
            p.trials.append(
                GenericTrial(labels.index(nf.group(1)), 2, fvec, "none", block=block, phase="nofb")
            )
            t_in += 1
            continue
        fb = _PL_FB.match(s)
        if not fb:
            raise ParseError(f"{experiment}#{participant}: unparsable described line {s[:100]!r}")
        pressed, sign1, v1, sign2, v2, other = fb.groups()
        if pressed not in labels or other not in labels or pressed == other:
            raise ParseError(f"{experiment}#{participant}: described letters mismatch {s[:100]!r}")
        own = float(v1) * (1 if sign1 == "gain" else -1)
        alt = float(v2) * (1 if sign2 == "gained" else -1)
        choice = labels.index(pressed)
        rewards = [own, alt] if choice == 0 else [alt, own]
        ra = tuple(_unit(r, -100, 100) for r in rewards)
        p.trials.append(
            GenericTrial(
                choice,
                2,
                fvec,
                "full",
                _unit(own, -100, 100),
                int(np.argmax(ra)),
                ra,
                block=block,
                phase="fb",
            )
        )
        t_in += 1
    return p


# ----------------------------------------------------------------------------- dispatch + validation
PARSERS = {
    "bandit2": parse_bandit2,
    "srl3": parse_srl3,
    "horizon": parse_horizon,
    "casino": parse_casino,
    "wpt": parse_wpt,
    "garcia": parse_garcia,
    "described": parse_described,
}


def parse_record(rec: dict, paradigm: str) -> Participant:
    return PARSERS[paradigm](rec["text"], rec["experiment"], str(rec["participant"]))


def paradigm_of(experiment: str) -> str | None:
    for par, prefixes in PARADIGM_PREFIXES.items():
        if experiment.startswith(prefixes):
            return par
    return None


def validate_generic(parts: list[Participant]) -> dict:
    """Parse-validation descriptives per experiment file; raises on internal inconsistencies."""
    out: dict[str, dict] = {}
    for p in parts:
        if not p.trials:
            raise ParseError(f"{p.key}: no trials")
        e = out.setdefault(
            p.experiment,
            {
                "participants": 0,
                "trials": [],
                "free_trials": [],
                "blocks": set(),
                "rewards": [],
                "feedback_modes": {},
            },
        )
        e["participants"] += 1
        e["trials"].append(len(p.trials))
        free = [t for t in p.trials if not t.forced]
        e["free_trials"].append(len(free))
        for t in p.trials:
            if not (0 <= t.choice < t.n_actions):
                raise ParseError(f"{p.key}: choice {t.choice} outside {t.n_actions} actions")
            if len(t.features) != N_FEATURES or max(abs(x) for x in t.features) > 1.0 + 1e-9:
                raise ParseError(f"{p.key}: bad feature vector")
            if t.mode != "none" and t.reward is not None:
                if not (0.0 <= t.reward <= 1.0):
                    raise ParseError(f"{p.key}: reward {t.reward} outside [0, 1]")
                e["rewards"].append(t.reward)
            e["blocks"].add(t.block)
            e["feedback_modes"][t.mode] = e["feedback_modes"].get(t.mode, 0) + 1
    return {
        exp: {
            "participants": v["participants"],
            "trials_per_participant": [min(v["trials"]), max(v["trials"])],
            "free_trials_per_participant": [min(v["free_trials"]), max(v["free_trials"])],
            "n_blocks": len(v["blocks"]),
            "mean_normalised_reward": float(np.mean(v["rewards"])) if v["rewards"] else None,
            "feedback_modes": v["feedback_modes"],
        }
        for exp, v in sorted(out.items())
    }
