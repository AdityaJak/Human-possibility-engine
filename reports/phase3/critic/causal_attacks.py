#!/usr/bin/env python3
"""Critic (audit/critic-v3): can the Phase III causality check and runner be fooled? Toy data only.

    PYTHONPATH=packages python reports/phase3/critic/causal_attacks.py \
        --out reports/phase3/critic/causal_attacks.json

No human data is read. Every "participant" below is synthetic (``hpe.core.rng`` streams). Each attack is
a ``SequencePredictor`` whose predictions use information that is not available when the choice is made
(the action being predicted, the sequence length, the participant key, other held-out participants) or
whose Gate B/A ablation modes are under its own control. For each attack we record

* ``production_check``: violations returned by ``hpe.population.core.conformance.check_causal`` called
  exactly as ``runner.run_model_family`` calls it (selected model, first select-partition sequence,
  after the select-partition predictions have been made);
* ``strict_check``: violations returned by ``strict_check`` below, the stronger check proposed in
  ``docs/phase3/INDEPENDENT_SCIENTIFIC_AUDIT.md`` (fresh deep copy per call, randomised key,
  prefix-truncation with a guaranteed flip of the predicted action, donor path, state-mutation test).

Part B runs the production ``runner.run_model_family`` + ``runner.analyze`` on two toy families
registered in-process as counted, to show what the summary reports for a model the check flags and for
one it does not. Nothing is written outside a temporary directory except ``--out``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from hpe.core.rng import stream
from hpe.evaluation.human.psych101 import Participant
from hpe.evaluation.human.psych101_v2 import N_FEATURES, GenericTrial
from hpe.population.core import runner
from hpe.population.core.conformance import check_causal
from hpe.population.core.encoding import EncodedSeq, encode, input_dim
from hpe.population.core.families import P3_PARTITIONS, FamilySpec, register_family
from hpe.population.core.interface import ModelSpec, register_model

SEED = 20261104
K_MAX = 4


# ----------------------------------------------------------------------------- synthetic participants
def toy_participant(i: int, *, k: int, T: int, shift: float = 0.0, tag: str = "a") -> Participant:
    """A sticky chooser with a person-specific preference; ``shift`` moves the population mean."""
    rng = stream(SEED, "critic_toy", tag, i)
    pref = rng.normal(0.0, 1.5, size=k)
    pref[1 % k] += shift
    p = Participant(f"critictoy_{tag}/exp1.csv", str(i), f"critictoy_{tag}", k)
    prev = None
    for _t in range(T):
        logits = pref.copy()
        if prev is not None:
            logits[prev] += 1.0
        pr = np.exp(logits - logits.max())
        pr /= pr.sum()
        c = int(rng.choice(k, p=pr))
        p.trials.append(GenericTrial(c, k, (0.0,) * N_FEATURES, "bandit", float(rng.uniform() < 0.5)))
        prev = c
    return p


def toy_seqs(
    n: int, *, k: int, T: int, shift: float = 0.0, tag: str = "a", start: int = 0
) -> list[EncodedSeq]:
    return [
        encode(toy_participant(start + i, k=k, T=T, shift=shift, tag=tag), f"critictoy_{tag}", K_MAX)
        for i in range(n)
    ]


# ----------------------------------------------------------------------------- toy predictors
def _log_norm(w: np.ndarray, avail: np.ndarray) -> np.ndarray:
    out = np.full(w.shape, -np.inf)
    ww = np.where(avail, w, 0.0)
    out[avail] = np.log(ww[avail] / ww.sum())
    return out


class Honest:
    """Population choice frequencies (fitted on dev) as a prior + the person's running counts. Causal."""

    adaptive = False

    def __init__(self, name: str = "critic_honest") -> None:
        self.name = name
        self.prior = np.ones(K_MAX) / K_MAX

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        c = np.ones(K_MAX)
        for s in train:
            np.add.at(c, s.actions[s.scored], 1.0)
        self.prior = c / c.sum()
        return {"n_train": len(train)}

    def _honest(self, seq: EncodedSeq, prior: np.ndarray, weight: float = 5.0) -> np.ndarray:
        counts = prior * weight
        out = np.full((seq.T, seq.k_max), -np.inf)
        for t in range(seq.T):
            out[t] = _log_norm(counts.copy(), seq.avail[t])
            counts[seq.actions[t]] += 1.0  # only after row t has been produced
        return out

    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        return self._honest(seq, self.prior)

    def param_count(self) -> int:
        return K_MAX

    def describe(self) -> dict[str, Any]:
        return {"population_params": K_MAX, "per_person_params": 0}


def _peek_row(seq: EncodedSeq, t: int) -> np.ndarray:
    w = np.where(seq.avail[t], 0.1, 0.0)
    w[seq.actions[t]] = 0.9
    return _log_norm(w, seq.avail[t])


class PeekAll(Honest):
    """Reads the action it predicts at every row (the repo's own `_Cheat`). Control: must be caught."""

    def predict_logprobs(self, seq, *, mode="adapt", donor=None):  # type: ignore[no-untyped-def]
        out = self._honest(seq, self.prior)
        for t in range(seq.T):
            out[t] = _peek_row(seq, t)
        return out


class ProbeDodger(Honest):
    """Peeks everywhere except at the three deterministic probe rows of check_causal."""

    def predict_logprobs(self, seq, *, mode="adapt", donor=None):  # type: ignore[no-untyped-def]
        out = self._honest(seq, self.prior)
        probes = {max(1, seq.T // 4), seq.T // 2, max(1, (3 * seq.T) // 4)}
        for t in range(seq.T):
            if t not in probes:
                out[t] = _peek_row(seq, t)
        return out


class KeyCache(PeekAll):
    """Peeks on the first call for a key and memoises by ``seq.key`` (a common 'speed-up')."""

    def __init__(self, name: str = "critic_keycache") -> None:
        super().__init__(name)
        self.cache: dict[str, np.ndarray] = {}

    def predict_logprobs(self, seq, *, mode="adapt", donor=None):  # type: ignore[no-untyped-def]
        if seq.key not in self.cache:
            self.cache[seq.key] = super().predict_logprobs(seq, mode=mode)
        return self.cache[seq.key].copy()


class Transductive(Honest):
    """Causal within a person, but pools every *new* participant it predicts into its population prior:
    later held-out participants are predicted with earlier held-out participants' whole sequences."""

    def __init__(self, name: str = "critic_transductive") -> None:
        super().__init__(name)
        self.pool = np.zeros(K_MAX)
        self.seen: set[str] = set()

    def predict_logprobs(self, seq, *, mode="adapt", donor=None):  # type: ignore[no-untyped-def]
        prior = self.prior * 50.0 + self.pool
        out = self._honest(seq, prior / prior.sum())
        if seq.key not in self.seen:
            self.seen.add(seq.key)
            np.add.at(self.pool, seq.actions[seq.scored], 1.0)
        return out


class InvariantFields(Honest):
    """Uses fields check_causal never perturbs: sequence length, the scored mask beyond t, the key."""

    def predict_logprobs(self, seq, *, mode="adapt", donor=None):  # type: ignore[no-untyped-def]
        out = self._honest(seq, self.prior)
        h = int(hashlib.sha256(seq.key.encode()).hexdigest(), 16)
        for t in range(seq.T):
            j = (seq.T + int(seq.scored[t + 1 :].sum()) + h) % int(seq.avail[t].sum())
            w = np.exp(np.where(seq.avail[t], out[t], -np.inf))
            w[j] *= 1.5
            out[t] = _log_norm(w, seq.avail[t])
        return out


class DonorPathPeek(Honest):
    """Causal in 'adapt' and 'frozen'; the donor (Gate A2) path reads the target's own actions."""

    adaptive = True

    def predict_logprobs(self, seq, *, mode="adapt", donor=None):  # type: ignore[no-untyped-def]
        if donor is None:
            return self._honest(seq, self.prior)
        out = self._honest(seq, self.prior)
        for t in range(seq.T):
            out[t] = _peek_row(seq, t)
        return out


class SabotagedAblations(Honest):
    """'adapt' is identical to the Honest comparator (zero added value); 'frozen' and the donor path are
    deliberately degraded. Causal everywhere. Shows that Gate B1/A2 contrasts are under the candidate's
    own control as long as the candidate implements its ablation modes."""

    adaptive = True

    def predict_logprobs(self, seq, *, mode="adapt", donor=None):  # type: ignore[no-untyped-def]
        if mode == "adapt" and donor is None:
            return self._honest(seq, self.prior)
        bad = np.full(K_MAX, 0.02)
        bad[K_MAX - 1] = 1.0  # a confidently wrong 'population state' for the ablations
        return self._honest(seq, bad / bad.sum(), weight=40.0)


# ----------------------------------------------------------------------------- the proposed strict check
def _state_digest(model: Any) -> str:
    try:
        return hashlib.sha256(pickle.dumps(model.__dict__, protocol=4)).hexdigest()
    except Exception as e:  # pragma: no cover - unpicklable state is itself a finding
        return f"unpicklable:{type(e).__name__}"


def _rekey(seq: EncodedSeq, key: str) -> EncodedSeq:
    return EncodedSeq(
        key,
        seq.family,
        seq.k_max,
        seq.n_actions,
        seq.x.copy(),
        seq.actions.copy(),
        seq.avail.copy(),
        seq.scored.copy(),
    )


def _prefix_flip(seq: EncodedSeq, t: int, key: str) -> EncodedSeq:
    """seq truncated to rows <= t, with the action at t replaced by a *different* available action."""
    a = seq.actions[: t + 1].copy()
    av = np.flatnonzero(seq.avail[t])
    if av.size > 1:
        a[t] = av[(int(np.searchsorted(av, a[t])) + 1) % av.size]
    return EncodedSeq(
        key,
        seq.family,
        seq.k_max,
        seq.n_actions,
        seq.x[: t + 1].copy(),
        a,
        seq.avail[: t + 1].copy(),
        seq.scored[: t + 1].copy(),
    )


def strict_check(
    model: Any,
    seqs: list[EncodedSeq],
    *,
    seed: int,
    n_probe: int = 24,
    atol: float = 0.0,
    donors: bool = True,
) -> list[str]:
    """Stronger causality/state check (proposal). Empty list = no violation found."""
    problems: list[str] = []
    rng = stream(seed, "critic_strict")

    def fresh(s: EncodedSeq, **kw: Any) -> np.ndarray:
        return copy.deepcopy(model).predict_logprobs(s, **kw)

    def differ(a: np.ndarray, b: np.ndarray) -> bool:
        if atol == 0.0:
            return not np.array_equal(a, b)
        fa, fb = np.isfinite(a), np.isfinite(b)
        return bool((fa != fb).any() or np.abs(a[fa] - b[fa]).max(initial=0.0) > atol)

    for si, seq in enumerate(seqs):
        rkey = f"critic-random-{int(rng.integers(1 << 62))}"
        # 1. predictions must not depend on the participant key (identity / experiment id / caches)
        base = {m: fresh(_rekey(seq, rkey), mode=m) for m in ("adapt", "frozen")}
        if differ(base["adapt"], fresh(seq, mode="adapt")):
            problems.append(f"seq{si}: predictions depend on seq.key")
        # 2. predicting a participant it has never seen must not change the model (no per-key caches, no
        #    transductive pooling of held-out participants); a fresh random key defeats 'seen-before' guards
        probe_model = copy.deepcopy(model)
        d0 = _state_digest(probe_model)
        probe_model.predict_logprobs(_rekey(seq, f"critic-unseen-{int(rng.integers(1 << 62))}"))
        if _state_digest(probe_model) != d0:
            problems.append(f"seq{si}: predict_logprobs mutated model state")
        # 3. prefix truncation + guaranteed flip of the predicted action, every mode and the donor path
        T = seq.T
        dense = (
            list(range(T))
            if T <= 64
            else sorted({0, 1, 2, T - 2, T - 1, *(int(v) for v in rng.integers(0, T, n_probe))})
        )
        donor = seqs[(si + 1) % len(seqs)] if donors and len(seqs) > 1 else None
        base_d = fresh(_rekey(seq, rkey), donor=donor) if donor is not None else None
        for t in dense:
            pre = _prefix_flip(seq, t, rkey)
            for m in ("adapt", "frozen"):
                if differ(base[m][: t + 1], fresh(pre, mode=m)[: t + 1]):
                    problems.append(f"seq{si} mode={m}: rows <= {t} depend on data at or after t")
                    break
            if base_d is not None and differ(base_d[: t + 1], fresh(pre, donor=donor)[: t + 1]):
                problems.append(f"seq{si} donor: rows <= {t} depend on the target's data at or after t")
        if len(problems) > 20:
            break
    # de-duplicate per kind for readability
    seen: dict[str, str] = {}
    for p in problems:
        kind = p.split(":", 1)[1].split("<=")[0].strip() if ":" in p else p
        seen.setdefault(kind, p)
    return sorted(seen.values())


# ----------------------------------------------------------------------------- part A: check vs attacks
def production_check_like_runner(model: Any, select: list[EncodedSeq]) -> list[str]:
    _ = [model.predict_logprobs(s) for s in select]  # runner.run_model_family:136
    return check_causal(model, select[0], seed=SEED)  # runner.run_model_family:160


def part_a() -> dict[str, Any]:
    dev = toy_seqs(40, k=4, T=60, tag="a")
    tune = toy_seqs(12, k=4, T=60, tag="a", start=1000)
    res: dict[str, Any] = {}
    for cls in (
        Honest,
        PeekAll,
        ProbeDodger,
        KeyCache,
        Transductive,
        InvariantFields,
        DonorPathPeek,
        SabotagedAblations,
    ):
        m = cls()  # type: ignore[abstract]
        m.fit(dev, seed=SEED, threads=1, budget_s=60)
        prod = production_check_like_runner(m, tune)
        m2 = cls()  # type: ignore[abstract]
        m2.fit(dev, seed=SEED, threads=1, budget_s=60)
        _ = [m2.predict_logprobs(s) for s in tune]  # same history as in the runner
        strict = strict_check(m2, tune[:6], seed=SEED)
        res[cls.__name__] = {
            "production_check": prod,
            "production_flags": bool(prod),
            "strict_check": strict,
            "strict_flags": bool(strict),
        }
    # self-peek on 2-action sequences: how often does the production check miss it?
    seqs2 = toy_seqs(400, k=2, T=150, tag="k2")
    missed = sum(1 for s in seqs2 if not check_causal(PeekAll(), s, seed=SEED))
    res["PeekAll_k2_miss_rate"] = {
        "sequences": len(seqs2),
        "missed_by_production_check": missed,
        "rate": missed / len(seqs2),
        "note": "random replacement equals the true action with p=1/2 at each of 3 probe rows; expected miss rate 1/8",
    }
    return res


# ----------------------------------------------------------------------------- part B: runner + analyze
def part_b() -> dict[str, Any]:
    named = {
        "critic_honest": Honest,
        "critic_peek_all": type(
            "CriticPeekAll", (PeekAll,), {"__init__": lambda self: Honest.__init__(self, "critic_peek_all")}
        ),
        "critic_keycache": KeyCache,
        "critic_transductive": Transductive,
        "critic_sabotaged": type(
            "CriticSabotaged",
            (SabotagedAblations,),
            {"__init__": lambda self: Honest.__init__(self, "critic_sabotaged")},
        ),
    }
    for name, cls in named.items():

        def factory(config: dict[str, Any], k: int, d: int, _cls: type = cls) -> Any:
            return _cls()

        register_model(ModelSpec(name, factory, "critic", "toy", {}, adaptive=name == "critic_sabotaged"))
    fams = {"critictoy_a": "counted_primary", "critictoy_b": "counted"}
    for fam, role in fams.items():
        register_family(FamilySpec(fam, (f"{fam}/",), 4, P3_PARTITIONS, lambda *_: None, role))  # type: ignore[arg-type, return-value]
    defn = runner.P3Definition(
        experiment_id="critic_toy",
        title="critic toy",
        lane="audit/critic-v3",
        hypothesis="toy",
        families=list(fams),
        seed=SEED,
        threads=1,
        models={
            "critic_honest": runner.ModelRun(role="comparator"),
            "critic_peek_all": runner.ModelRun(role="candidate"),
            "critic_keycache": runner.ModelRun(role="candidate"),
            "critic_transductive": runner.ModelRun(role="candidate"),
            "critic_sabotaged": runner.ModelRun(role="candidate", variants=["frozen", "mismatched"]),
        },
    )
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td) / defn.experiment_id
        (outdir / "parts").mkdir(parents=True)
        (outdir / "definition.json").write_text(
            json.dumps({**defn.model_dump(), "preregistration_hash": defn.preregistration_hash()})
        )
        for fam in fams:
            tag = fam.rsplit("_", 1)[1]
            # dev population prefers action 0; the reported (held-out) population is shifted towards action 1
            data = {
                "dev": [_with_family(s, fam) for s in toy_seqs(60, k=4, T=80, tag=tag)],
                "tune": [
                    _with_family(s, fam) for s in toy_seqs(40, k=4, T=80, shift=2.0, tag=tag, start=5000)
                ],
            }
            for name, mrun in defn.models.items():
                r = runner.run_model_family(defn, fam, name, mrun, data)
                (outdir / "parts" / f"{fam}__{name}.json").write_text(json.dumps(r))
                out.setdefault("causality_violations", {}).setdefault(fam, {})[name] = r[
                    "causality_violations"
                ]
        summary = runner.analyze(outdir)
    out["best_comparator"] = {f: summary["families"][f]["best_comparator"] for f in fams}
    out["candidate_vs_best_comparator"] = {
        c: {f: {k: v[k] for k in ("estimate", "p_holm", "meets_0.02")} for f, v in by.items()}
        for c, by in summary["candidate_vs_best_comparator"].items()
    }
    out["sabotaged_variant_contrasts"] = {
        f: [
            {k: c[k] for k in ("first", "second", "kind", "estimate", "p_one_sided")}
            for c in summary["families"][f]["contrasts"]
            if c["first"] == "critic_sabotaged"
        ]
        for f in fams
    }
    return out


def _with_family(s: EncodedSeq, fam: str) -> EncodedSeq:
    s.family = fam
    s.key = f"{fam}/exp1.csv#{s.key.split('#', 1)[1]}"
    return s


# ----------------------------------------------------------------------------- part C: pop_gru under the strict check
def part_c() -> dict[str, Any]:
    try:
        import torch

        from hpe.population.core.interface import get_model
    except ModuleNotFoundError:
        return {"skipped": "torch not installed"}
    torch.set_num_threads(1)
    seqs = toy_seqs(16, k=4, T=60, tag="gru")
    m = get_model("pop_gru").factory(
        {"hidden": 8, "lr": 0.01, "max_epochs": 2, "patience": 2}, K_MAX, input_dim(K_MAX)
    )
    m.fit(seqs, seed=SEED, threads=1, budget_s=60)
    exact = strict_check(m, seqs[:3], seed=SEED, n_probe=8, donors=False)
    tol = strict_check(m, seqs[:3], seed=SEED, n_probe=8, atol=1e-6, donors=False)
    worst = 0.0
    full = m.predict_logprobs(seqs[0])
    for t in (5, 20, 40, 59):
        pre = m.predict_logprobs(_prefix_flip(seqs[0], t, "k"))
        fin = np.isfinite(full[: t + 1])
        worst = max(worst, float(np.abs(full[: t + 1][fin] - pre[: t + 1][fin]).max()))
    return {
        "strict_exact": exact,
        "strict_atol_1e-6": tol,
        "max_abs_prefix_vs_full": worst,
        "production_check": check_causal(m, seqs[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/phase3/critic/causal_attacks.json"))
    args = ap.parse_args()
    res = {
        "what": "toy attacks on hpe.population.core.conformance.check_causal and the runner (no human data)",
        "part_a_check_vs_attacks": part_a(),
        "part_b_runner_and_analyze": part_b(),
        "part_c_pop_gru_strict": part_c(),
    }
    args.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    a = res["part_a_check_vs_attacks"]
    print(f"{'attack':22s} production-check-flags  strict-check-flags")
    for k, v in a.items():
        if "production_flags" in v:
            print(f"{k:22s} {v['production_flags']!s:23s} {v['strict_flags']!s}")
    print("PeekAll on 2-action sequences missed by production check:", a["PeekAll_k2_miss_rate"])
    b = res["part_b_runner_and_analyze"]
    print("runner causality_violations:", json.dumps(b["causality_violations"]))
    print("analyze candidate_vs_best_comparator:", json.dumps(b["candidate_vs_best_comparator"]))
    print("sabotaged variants:", json.dumps(b["sabotaged_variant_contrasts"]))
    print("pop_gru strict:", json.dumps(res["part_c_pop_gru_strict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
