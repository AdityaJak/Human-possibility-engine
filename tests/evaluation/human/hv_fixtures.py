"""Synthetic fixtures that mimic the Psych-101 transcript and HeartSteps CSV formats.

No real external data is used or committed. Generators are seeded and tiny.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

CUE_WORDS = ["very little", "a little", "average", "a lot", "very much"]
OUT_WORDS = [
    "extremely low",
    "very low",
    "low",
    "somewhat low",
    "normal",
    "somewhat high",
    "high",
    "very high",
    "extremely high",
]


def mcpl_text(
    rng: np.random.Generator, n_trials: int = 40, n_feedback: int = 30, numeric: bool = False
) -> str:
    levels = [str(i) for i in range(1, 6)] if numeric else CUE_WORDS
    head = [
        "Your task is to estimate the blood concentration of the hormone Caldionine based on information about the amount of two other hormones, Progladine and Amalydine, in multiple individuals' urine.",
        f"Both Progladine and Amalydine can take five values ({', '.join(levels)}).",
        f"Caldionine can take nine values ({', '.join(OUT_WORDS)}).",
        "Your goal is to estimate the concentration of Caldionine correctly.",
        "You will receive feedback about the actual concentration after making your estimate.",
        "This feedback will stop at some point.",
        "",
    ]
    lines = []
    w = 0.0  # simulated learner: delta rule toward truth = c1 + c2
    for t in range(n_trials):
        c1, c2 = int(rng.integers(0, 5)), int(rng.integers(0, 5))
        truth = int(np.clip(c1 + c2 + rng.integers(-1, 2), 0, 8))
        guess = int(np.clip(round(4 + w * (c1 + c2 - 4) + rng.normal(0, 1)), 0, 8))
        s = f"Progladine: {levels[c1]}. Amalydine: {levels[c2]}. You say that the Caldionine concentration is <<{OUT_WORDS[guess]}>>. "
        if t < n_feedback:
            if guess == truth:
                s += f"That is correct. The correct concentration of Caldionine is indeed {OUT_WORDS[truth]}."
            else:
                s += f"That is incorrect. The correct concentration of Caldionine is {OUT_WORDS[truth]}."
            w += 0.1 * (1.0 - w)
        lines.append(s)
    return "\n".join(head + lines)


def igt_text(rng: np.random.Generator, n_trials: int = 30) -> str:
    labels = list(rng.choice(list("ABCDEFGHJK"), size=4, replace=False))
    head = [
        f"You see in front of you four decks of cards labeled {labels[0]}, {labels[1]}, {labels[2]}, and {labels[3]}.",
        "You get a loan of 2000$ of play money.",
        "You have to select one card at a time, from any of the four decks, for 100 trials.",
        "",
    ]
    wins = [100.0, 100.0, 50.0, 50.0]
    lines = []
    q = np.zeros(4)
    for _ in range(n_trials):
        p = np.exp(q - q.max())
        p /= p.sum()
        a = int(rng.choice(4, p=p))
        loss = float(rng.choice([0.0, 0.0, 250.0])) if a < 2 else float(rng.choice([0.0, 50.0]))
        lines.append(f"You press <<{labels[a]}>>. You win {wins[a]}$ and lose {loss}$.")
        q[a] += 0.2 * ((wins[a] - loss) / 100 - q[a])
    return "\n".join(head + lines)


@pytest.fixture
def psych_jsonl(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    path = tmp_path / "psych.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for exp, n in (("collsiöö2023MCPL/exp1.csv", 10), ("collsiöö2023MCPL/exp2.csv", 8)):
            for i in range(n):
                txt = mcpl_text(rng, numeric=exp.endswith("exp2.csv"))
                f.write(
                    json.dumps({"text": txt, "experiment": exp, "participant": i}, ensure_ascii=False) + "\n"
                )
        for i in range(14):
            f.write(
                json.dumps(
                    {"text": igt_text(rng), "experiment": "steingroever2015data/exp1.csv", "participant": i}
                )
                + "\n"
            )
        f.write(
            json.dumps({"text": "irrelevant", "experiment": "other2020/exp1.csv", "participant": 0}) + "\n"
        )
    return path


HS_COLUMNS = [
    "user.index",
    "decision.index",
    "avail",
    "send",
    "send.active",
    "send.sedentary",
    "jbsteps30.zero",
    "jbsteps30pre.zero",
    "jbsteps30",
]


def write_heartsteps_csv(
    path: Path, *, beta_walk: float, beta_sed: float, n_users: int = 30, n_dec: int = 150, seed: int = 0
) -> None:
    rng = np.random.default_rng(seed)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HS_COLUMNS)
        for u in range(1, n_users + 1):
            ui = rng.normal(0, 0.5)
            for d in range(n_dec):
                avail = rng.random() < 0.8
                pre = float(rng.poisson(150))
                if avail:
                    r = rng.random()
                    walk, sed = r < 0.3, 0.3 <= r < 0.6
                    send = walk or sed
                else:
                    walk = sed = send = False
                mu = 4.0 + ui + 0.3 * np.log(pre + 0.5) - 0.01 * (d // 5) + beta_walk * walk + beta_sed * sed
                steps = float(max(0.0, np.round(np.exp(mu + rng.normal(0, 0.8)) - 0.5)))
                complete = "" if rng.random() < 0.1 else steps
                w.writerow([u, d, avail, send if avail else "", walk, sed, steps, pre, complete])
