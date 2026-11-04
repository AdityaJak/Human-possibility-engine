"""two_step — the "spaceship" two-step tasks of Kool, Cushman & Gershman (2016), Kool, Gershman & Cushman
(2017) and Zorowitz et al. (2023), via Psych-101 (Binz et al., 2024; Apache-2.0).

Five experiment files share one transcript vocabulary but two designs:

* **one-stage** (``kool2016when/exp1``, ``kool2017cost/exp1``): each day one of two spaceship *pairs* is
  presented; each spaceship flies deterministically to one of two planets whose single alien yields an
  outcome. ``kool2016when/exp1``: "You find N pieces of space treasure" (+N), "... of antimatter" (−N) or
  "You find nothing" (0); observed range on dev −4..+5. ``kool2017cost/exp1``: a treasure multiplier
  (×5, announced before the choice) scales what is *received*; the alien's outcome is what is *found*
  (0..9) and is what we encode (the multiplier is a context feature; ``receive = find × 5`` is checked).
* **two-stage** (``kool2016when/exp2``, ``kool2017cost/exp2``, ``zorowitz2023data/exp1``): one spaceship
  pair; each spaceship flies to a planet with common/rare transitions, where the participant chooses
  between two aliens; binary outcome (0/1 pieces of treasure, or treasure/junk).

Encoding (one ``GenericTrial`` per choice, K = 2):

* one-stage day → one step, ``mode="bandit"``, reward on the chosen spaceship;
* two-stage day → a stage-1 step (spaceship, ``mode="none"``: the planet reached is shown in the next
  step's features) and a stage-2 step (alien, ``mode="bandit"``).

Action indices are canonical per participant, fixed by information the participant had before the first
choice: the instruction text order of spaceships/planets/aliens (Kool studies) or the first presentation
order (Zorowitz, whose instructions name no labels). The transcript's presentation order varies from day to
day in some files (screen side), so it is a feature, not the index.

Reward normalisation (fixed affine maps; identical for every model): kool2016when/exp1 r → (r + 4) / 9;
kool2017cost/exp1 found r → r / 9; binary outcomes unchanged. A value outside the map's range is a parse
error (never clipped silently).

Context features (12, in [-1, 1]; all known before the choice of the step):

    0  stage                 −1 spaceship choice, +1 alien choice
    1  design                −1 one-stage, +1 two-stage
    2  start pair            −1 / +1 which spaceship pair is shown (one-stage designs); 0 otherwise
    3  current planet        −1 / +1 on alien steps; 0 on spaceship steps
    4  previous planet       planet reached on the previous day (−1 / +1); 0 on the first day or after a
                             day without a spaceship choice
    5  multiplier            +1 multiplier announced, −1 no multiplier, 0 design without multipliers
    6  presentation swapped  +1 if the transcript lists the options in reverse canonical order, else −1
    7  progress              day index / (days − 1) mapped to [-1, 1] (days stated in the instructions;
                             Zorowitz: 201, the design length seen on dev)
    8–9   code of canonical option 0 (``psych101_v2._code``, 2 dims, salted per participant)
    10–11 code of canonical option 1

Missed responses ("You do not respond in time") produce no step; they are counted in ``meta``. A day with
a spaceship choice but a missed alien choice keeps its stage-1 step.
"""

from __future__ import annotations

import re

from hpe.evaluation.human.psych101 import ParseError, Participant
from hpe.evaluation.human.psych101_v2 import GenericTrial, _code, _pad
from hpe.population.core.families import P3_PARTITIONS, FamilySpec, register_family

PREFIXES = ("kool2016when/", "kool2017cost/", "zorowitz2023data/")

# instruction lines
_ONE_PAIRS = re.compile(
    r"either be presented with spaceships ([A-Z]) and ([A-Z]) or with spaceships ([A-Z]) and ([A-Z])\."
)
_ONE_PLANETS = re.compile(r"two different planets ([A-Z]) and ([A-Z])\.")
_TWO_SHIPS = re.compile(
    r"one of the spaceships ([A-Z]) or ([A-Z]) to one of the planets ([A-Z]) or ([A-Z])\."
)
_TWO_ALIENS = re.compile(
    r"Planet ([A-Z]) has aliens ([A-Z]) and ([A-Z]), and planet ([A-Z]) has aliens ([A-Z]) and ([A-Z]) on it\."
)
_DAYS = re.compile(r"next (\d+) days")
_MULT_FACTOR = re.compile(r"you will receive (\d+) times the amount")

# day lines
_DAY = re.compile(
    r"^(?:There is (?P<mult>no|a) treasure multiplier\. )?"
    r"You are presented with (?:two )?spaceships (?:called )?(?P<s1>[A-Z]) and (?P<s2>[A-Z])\. (?P<rest>.*)$"
)
_MISS1 = re.compile(
    r"^You do not respond in time (?:on|for) this day\. You do not go to any planet\. "
    r"You (?:find nothing|do not get any treasure)\.$"
)
_STAGE1 = re.compile(
    r"^You press <<(?P<c>[A-Z])>>\. You end up on (?:planet (?P<pl>[A-Z])|the (?P<col>red|blue) planet)\. "
    r"(?P<rest>.*)$"
)
_ONE_OUT = re.compile(
    r"^You find (?:(?P<n>-?\d+) pieces of (?P<what>space treasure|antimatter)|(?P<nothing>nothing))\."
    r"(?: You receive (?P<recv>\d+) pieces of space treasure\.)?$"
)
_ALIENS = re.compile(
    r"^You see (?:a (?P<col1>red|blue) )?alien (?:named )?(?P<a1>[A-Z]) and "
    r"(?:a (?P<col2>red|blue) )?alien (?:named )?(?P<a2>[A-Z])\. (?P<rest>.*)$"
)
_STAGE2 = re.compile(
    r"^You press <<(?P<c>[A-Z])>>\. You find (?:(?P<n>-?\d+) pieces of space treasure|(?P<w>junk|treasure))\.$"
)
_MISS2 = re.compile(
    r"^You do not respond in time for this day\. You do not ask any alien for treasure\. "
    r"You do not get any treasure\.$"
)

# fixed affine reward maps, by experiment prefix: (lo, hi)
REWARD_RANGE: dict[str, tuple[float, float]] = {
    "kool2016when/exp1": (-4.0, 5.0),
    "kool2017cost/exp1": (0.0, 9.0),
    "kool2016when/exp2": (0.0, 1.0),
    "kool2017cost/exp2": (0.0, 1.0),
    "zorowitz2023data/exp1": (0.0, 1.0),
}
ZOROWITZ_DAYS = 201  # every dev transcript has 201 day lines; the instructions state no length
COLOURS = ("red", "blue")


def _range_of(experiment: str) -> tuple[float, float]:
    for k, v in REWARD_RANGE.items():
        if experiment.startswith(k):
            return v
    raise ParseError(f"{experiment}: no reward range registered for this two_step file")


def _norm(r: float, rng: tuple[float, float], where: str) -> float:
    lo, hi = rng
    if not (lo - 1e-9 <= r <= hi + 1e-9):
        raise ParseError(f"{where}: outcome {r} outside the fixed range [{lo}, {hi}]")
    return (r - lo) / (hi - lo)


def _sign(i: int) -> float:
    return -1.0 if i == 0 else 1.0


def parse_two_step(text: str, experiment: str, participant: str) -> Participant:
    where = f"{experiment}#{participant}"
    salt = where
    rrange = _range_of(experiment)
    p = Participant(experiment, participant, "two_step", 2)
    m_pairs = _ONE_PAIRS.search(text)
    m_ships = _TWO_SHIPS.search(text)
    design: str
    pairs: list[tuple[str, str]] = []
    planets: list[str] = []
    aliens: dict[str, tuple[str, str]] = {}
    if m_pairs:
        design = "one_stage"
        g = m_pairs.groups()
        pairs = [(g[0], g[1]), (g[2], g[3])]
        mp = _ONE_PLANETS.search(text)
        if not mp:
            raise ParseError(f"{where}: one-stage planets not found")
        planets = [mp.group(1), mp.group(2)]
    elif m_ships:
        design = "two_stage"
        g = m_ships.groups()
        pairs = [(g[0], g[1])]
        planets = [g[2], g[3]]
        ma = _TWO_ALIENS.search(text)
        if not ma:
            raise ParseError(f"{where}: two-stage alien assignment not found")
        a = ma.groups()
        aliens = {a[0]: (a[1], a[2]), a[3]: (a[4], a[5])}
        if set(aliens) != set(planets):
            raise ParseError(f"{where}: alien planets {sorted(aliens)} != planets {planets}")
    elif experiment.startswith("zorowitz2023data/"):
        design = "two_stage"
        planets = list(COLOURS)  # canonical: red = 0, blue = 1 (colour words, identical for everyone)
    else:
        raise ParseError(f"{where}: two_step design not recognised")
    md = _DAYS.search(text)
    n_days = int(md.group(1)) if md else ZOROWITZ_DAYS
    mf = _MULT_FACTOR.search(text)
    mult_factor = int(mf.group(1)) if mf else None
    has_mult = "treasure multiplier" in text

    p.meta.update(
        design=design,
        n_days=n_days,
        missed_stage1=0,
        missed_stage2=0,
        common_planet={},
        reward_range=list(rrange),
    )
    prev_planet = 0.0
    day = -1
    for ln in text.split("\n"):
        s = ln.strip()
        md2 = _DAY.match(s)
        if not md2:
            if "<<" in s or "You are presented" in s:
                raise ParseError(f"{where}: unparsable two_step line {s[:120]!r}")
            continue
        day += 1
        progress = (day / max(n_days - 1, 1)) * 2.0 - 1.0
        if not (-1.0 <= progress <= 1.0):
            raise ParseError(f"{where}: more days than the stated {n_days}")
        mult_flag = 0.0
        if has_mult:
            if md2.group("mult") is None:
                raise ParseError(f"{where}: multiplier statement missing on a day line")
            mult_flag = 1.0 if md2.group("mult") == "a" else -1.0
        shown = (md2.group("s1"), md2.group("s2"))
        if design == "two_stage" and not pairs:  # Zorowitz: canonical order = first presentation
            pairs = [shown]
        pair_idx = next((i for i, pr in enumerate(pairs) if set(pr) == set(shown)), None)
        if pair_idx is None:
            raise ParseError(f"{where}: spaceships {shown} not in the instructed pairs {pairs}")
        canon = pairs[pair_idx]
        rest = md2.group("rest")
        if _MISS1.match(rest):
            p.meta["missed_stage1"] += 1
            prev_planet = 0.0
            continue
        m1 = _STAGE1.match(rest)
        if not m1:
            raise ParseError(f"{where}: unparsable spaceship choice {s[:120]!r}")
        c1 = m1.group("c")
        if c1 not in canon:
            raise ParseError(f"{where}: pressed {c1} not among presented {shown}")
        planet = m1.group("pl") or m1.group("col")
        if planet not in planets:
            raise ParseError(f"{where}: planet {planet} not in {planets}")
        pl_idx = planets.index(planet)
        p.meta["common_planet"].setdefault(c1, {}).setdefault(planet, 0)
        p.meta["common_planet"][c1][planet] += 1
        f1 = _pad(
            [
                -1.0,
                -1.0 if design == "one_stage" else 1.0,
                _sign(pair_idx) if len(pairs) > 1 else 0.0,
                0.0,
                prev_planet,
                mult_flag,
                1.0 if shown != canon else -1.0,
                progress,
                *_code(canon[0], 2, salt),
                *_code(canon[1], 2, salt),
            ]
        )
        rest2 = m1.group("rest")
        if design == "one_stage":
            mo = _ONE_OUT.match(rest2)
            if not mo:
                raise ParseError(f"{where}: unparsable one-stage outcome {s[:120]!r}")
            r = 0.0 if mo.group("nothing") else float(mo.group("n"))
            if mo.group("what") == "antimatter":
                r = -r
            if mo.group("recv") is not None:
                expect = r * (mult_factor if (mult_flag > 0 and mult_factor) else 1)
                if float(mo.group("recv")) != expect:
                    raise ParseError(f"{where}: received {mo.group('recv')} != found {r} × multiplier")
            p.trials.append(
                GenericTrial(
                    canon.index(c1),
                    2,
                    f1,
                    "bandit",
                    _norm(r, rrange, where),
                    block=0,
                    phase="stage1",
                    meta={"day": day, "planet": pl_idx, "points": r},
                )
            )
            prev_planet = _sign(pl_idx)
            continue
        # two-stage: spaceship step without feedback, then the alien step
        p.trials.append(
            GenericTrial(
                canon.index(c1), 2, f1, "none", block=0, phase="stage1", meta={"day": day, "planet": pl_idx}
            )
        )
        ma2 = _ALIENS.match(rest2)
        if not ma2:
            raise ParseError(f"{where}: unparsable alien presentation {s[:120]!r}")
        shown_a = (ma2.group("a1"), ma2.group("a2"))
        if ma2.group("col1") is not None and {ma2.group("col1"), ma2.group("col2")} != {planet}:
            raise ParseError(f"{where}: alien colours do not match the {planet} planet")
        if planet not in aliens:
            if aliens and set(shown_a) & {x for pr in aliens.values() for x in pr}:
                raise ParseError(f"{where}: aliens {shown_a} already seen on another planet")
            aliens[planet] = shown_a  # Zorowitz: canonical order = first presentation on that planet
        canon_a = aliens[planet]
        if set(shown_a) != set(canon_a):
            raise ParseError(f"{where}: aliens {shown_a} on planet {planet} != instructed {canon_a}")
        rest3 = ma2.group("rest")
        prev_planet_next = _sign(pl_idx)
        if _MISS2.match(rest3):
            p.meta["missed_stage2"] += 1
            prev_planet = prev_planet_next
            continue
        m2 = _STAGE2.match(rest3)
        if not m2:
            raise ParseError(f"{where}: unparsable alien choice {s[:120]!r}")
        c2 = m2.group("c")
        if c2 not in canon_a:
            raise ParseError(f"{where}: pressed alien {c2} not among {shown_a}")
        r2 = (
            float(m2.group("n"))
            if m2.group("n") is not None
            else (1.0 if m2.group("w") == "treasure" else 0.0)
        )
        f2 = _pad(
            [
                1.0,
                1.0,
                0.0,
                _sign(pl_idx),
                prev_planet,
                mult_flag,
                1.0 if shown_a != canon_a else -1.0,
                progress,
                *_code(canon_a[0], 2, salt),
                *_code(canon_a[1], 2, salt),
            ]
        )
        p.trials.append(
            GenericTrial(
                canon_a.index(c2),
                2,
                f2,
                "bandit",
                _norm(r2, rrange, where),
                block=0,
                phase="stage2",
                meta={"day": day, "planet": pl_idx, "points": r2},
            )
        )
        prev_planet = prev_planet_next
    if day < 0:
        raise ParseError(f"{where}: no two_step day lines")
    p.meta["days_seen"] = day + 1
    p.meta["labels"] = {"pairs": [list(x) for x in pairs], "planets": planets, "aliens": aliens}
    return p


register_family(
    FamilySpec(
        name="two_step",
        prefixes=PREFIXES,
        n_actions=2,
        partition_file=P3_PARTITIONS,
        parser=parse_two_step,
        role="counted_if_validated",
        note="Kool et al. 2016 (exp1 one-stage, exp2 two-stage), Kool et al. 2017 (multiplier), Zorowitz et al. 2023",
    )
)
