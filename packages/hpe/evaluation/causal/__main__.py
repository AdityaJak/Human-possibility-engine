"""Phase II causal-validity pipeline.

    python -m hpe.evaluation.causal partition            # frozen dev/test split (refuses to change it)
    python -m hpe.evaluation.causal simulate [--n 500]   # estimator validation by simulation
    python -m hpe.evaluation.causal hte --partition dev  # moderator discovery (exploratory)
    python -m hpe.evaluation.causal hte --partition test --spec experiments/phase2/causal/hte_confirm.yaml
    python -m hpe.evaluation.causal ope --partition dev|test [--spec ...]

Every command writes JSON under reports/phase2/causal/ with code version, CPU seconds and the data
provenance. ``--partition test`` requires a pre-specification YAML and refuses to run twice unless
``--force`` is given (the second run is then recorded as such).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hpe.evaluation.causal import features, hte, mrt_sim, ope, partition
from hpe.evaluation.human import data as hdata

REPORTS = Path("reports/phase2/causal")
PARTITION = REPORTS / "partition.json"


def _code_version() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
        ).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # pragma: no cover
        return "unknown"


def _write(name: str, payload: dict[str, Any], t0: float) -> Path:
    payload = {
        "code_version": _code_version(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_seconds": round(time.process_time(), 2),
        "wall_seconds": round(time.time() - t0, 2),
        **payload,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / name
    out.write_text(json.dumps(payload, indent=1, default=float) + "\n")
    return out


def _load_rows() -> tuple[list[features.DecisionRow], dict]:
    src = hdata.HEARTSTEPS_SUGGESTIONS
    path = hdata.locate(src)
    return features.load_rows(path), hdata.provenance(src, path)


def _frame(rows: list[features.DecisionRow], part: str) -> features.Frame:
    rec = partition.load_partition(PARTITION)
    return features.build_frame(rows, users=rec[part])


def _guard_test(part: str, spec: str | None, marker: str, force: bool) -> dict | None:
    if part != "test":
        return None
    if not spec:
        raise SystemExit("--partition test requires --spec <pre-specification yaml> written before the run")
    pre = yaml.safe_load(Path(spec).read_text())
    m = REPORTS / marker
    if m.exists() and not force:
        raise SystemExit(
            f"{m} exists: the test partition was already evaluated for this spec; pass --force to record a re-run"
        )
    REPORTS.mkdir(parents=True, exist_ok=True)
    m.write_text(
        json.dumps(
            {"spec": spec, "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "forced": force}
        )
        + "\n"
    )
    return pre


def cmd_partition(args: argparse.Namespace) -> int:
    rows, prov = _load_rows()
    users = sorted({r.user for r in rows})
    rec = partition.write_partition(users, PARTITION)
    print(
        json.dumps(
            {
                "dev": rec["dev"],
                "test": rec["test"],
                "test_sha256": rec["test_sha256"],
                "source": prov.get("sha256"),
            },
            indent=1,
        )
    )
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    t0 = time.time()
    res = mrt_sim.validate_estimator(n_replicates=args.n, design=mrt_sim.MRTDesign(), seed=args.seed)
    power: dict[str, Any] = {}
    for label, b in (("phase1_point_estimate_0.115", 0.115), ("published_0.131", 0.131), ("half_0.06", 0.06)):
        d = mrt_sim.MRTDesign(beta0=b)
        rej = [
            mrt_sim.fit_marginal(mrt_sim.simulate("constant", d, args.seed * 7919 + r))["p"] < 0.05
            for r in range(args.n)
        ]
        power[label] = float(sum(rej) / len(rej))
    res["power_constant_effect_heartsteps_size"] = power
    out = _write("simulation_validation.json", res, t0)
    for k, v in res["scenarios"].items():
        extra = (
            f" | beta1 bias={v['moderator']['bias']:+.4f} cov={v['moderator']['coverage_95']:.3f}"
            if "moderator" in v
            else ""
        )
        print(
            f"{k:10s} bias={v['bias']:+.4f} coverage={v['coverage_95']:.3f} "
            f"reject={v['rejection_rate_alpha_05']:.3f}{extra}"
        )
    print("power:", power, "->", out)
    return 0


def cmd_hte(args: argparse.Namespace) -> int:
    t0 = time.time()
    pre = _guard_test(args.partition, args.spec, "hte_test_run.marker", args.force)
    rows, prov = _load_rows()
    fr = _frame(rows, args.partition)
    p = float(fr.a.mean()) if args.centring == "empirical" else 0.6
    mods = list(pre["moderators"]) if pre else list(features.MODERATOR_NAMES)
    marg = hte.marginal_effect(fr, p=p, n_bootstrap=args.bootstrap, seed=args.seed)
    res = hte.moderated_effects(fr, mods, p=p, n_bootstrap=args.bootstrap, seed=args.seed)
    payload: dict[str, Any] = {
        "partition": args.partition,
        "label": "CONFIRMATORY (pre-specified, single run)" if pre else "EXPLORATORY (dev partition)",
        "spec": pre,
        "provenance": prov,
        "sample": features.describe(fr),
        "centring_p": p,
        "marginal_any_suggestion": marg,
        "moderators": [r.as_dict() for r in res],
        "moderator_descriptions": {m: features.MODERATOR_DESCRIPTIONS[m] for m in mods},
        "selected_for_confirmation": hte.select_moderators(res) if not pre else None,
    }
    if pre:
        crit = pre["decision_rule"]
        payload["verdicts"] = [
            {
                "moderator": r.moderator,
                "beta1": r.beta1,
                "p_holm": r.p_holm,
                "expected_sign": crit["expected_sign"][r.moderator],
                "passed": bool(
                    r.p_holm < crit["alpha_holm"] and np.sign(r.beta1) == crit["expected_sign"][r.moderator]
                ),
            }
            for r in res
        ]
    out = _write(f"hte_{args.partition}.json", payload, t0)
    print(
        f"[{payload['label']}] n={len(fr)} users={payload['sample']['n_users']} "
        f"marginal={marg['estimate']:+.4f} ({marg['percent_effect']:+.1f}%)"
    )
    for r in res:
        print(
            f"  {r.moderator:10s} beta1={r.beta1:+.4f} se={r.se1:.4f} p={r.p_raw:.3f} holm={r.p_holm:.3f} "
            f"boot CI=[{r.ci_low_boot:+.3f},{r.ci_high_boot:+.3f}]"
        )
    print(
        "  verdicts:" if pre else "  selected for confirmation:",
        payload["verdicts"] if pre else payload["selected_for_confirmation"],
    )
    print("->", out)
    return 0


def cmd_ope(args: argparse.Namespace) -> int:
    t0 = time.time()
    pre = _guard_test(args.partition, args.spec, "ope_test_run.marker", args.force)
    rows, prov = _load_rows()
    dev = _frame(rows, "dev")
    fr = _frame(rows, args.partition)
    p = float(dev.a.mean()) if args.centring == "empirical" else 0.6
    if pre:
        mods = list(pre["policy_moderators"])
    else:
        mods = args.moderators.split(",") if args.moderators else list(features.MODERATOR_NAMES)
    q = ope.OutcomeModel.fit(dev, mods)  # outcome model and contextual policy are learned on dev only
    policies = {
        "never": ope.never,
        "always": ope.always,
        "behaviour": ope.behaviour(p),
        "contextual_greedy": ope.contextual_greedy(q),
    }
    res = ope.evaluate_policies(fr, policies, p=p, q=q, n_bootstrap=args.bootstrap, seed=args.seed)
    payload: dict[str, Any] = {
        "partition": args.partition,
        "label": "CONFIRMATORY (pre-specified, single run)"
        if pre
        else "EXPLORATORY (dev partition; policy learned on the same rows)",
        "spec": pre,
        "provenance": prov,
        "sample": features.describe(fr),
        "logging_p": p,
        "outcome_model": {"moderators": mods, "coef": q.coef.tolist(), "fitted_on": "dev"},
        "policies": res,
        "outcome_scale": "mean log(steps in 30 min after decision + 0.5) over available decision points",
        "caveat": "Retrospective off-policy estimate under randomisation-based identification; not a prospective result.",
    }
    if pre:
        c = res["contextual_greedy"]
        lo, hi = c["dr_minus_never_ci"]
        payload["verdict"] = {
            "contextual_minus_never_dr": c["dr_minus_never"],
            "ci": [lo, hi],
            "passed": bool(lo > pre["decision_rule"]["min_improvement_over_never"]),
            "contextual_minus_always_dr": c["dr"] - res["always"]["dr"],
        }
    out = _write(f"ope_{args.partition}.json", payload, t0)
    print(f"[{payload['label']}] n={len(fr)} p={p:.3f}")
    for name, r in res.items():
        vs = (
            f"  vs never DR={r['dr_minus_never']:+.4f} [{r['dr_minus_never_ci'][0]:+.3f},{r['dr_minus_never_ci'][1]:+.3f}]"
            if "dr_minus_never" in r
            else ""
        )
        print(
            f"  {name:18s} IPW={r['ipw']:.4f} [{r['ipw_ci'][0]:.3f},{r['ipw_ci'][1]:.3f}]  "
            f"DR={r['dr']:.4f} [{r['dr_ci'][0]:.3f},{r['dr_ci'][1]:.3f}]  send={r['share_send']:.2f}{vs}"
        )
    if pre:
        print("  verdict:", payload["verdict"])
    print("->", out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hpe.evaluation.causal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("partition")
    s = sub.add_parser("simulate")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--seed", type=int, default=20261105)
    for name in ("hte", "ope"):
        q = sub.add_parser(name)
        q.add_argument("--partition", choices=["dev", "test"], required=True)
        q.add_argument("--spec", default=None)
        q.add_argument("--force", action="store_true")
        q.add_argument("--bootstrap", type=int, default=2000)
        q.add_argument("--seed", type=int, default=20261105)
        q.add_argument("--centring", choices=["empirical", "design"], default="empirical")
        if name == "ope":
            q.add_argument(
                "--moderators", default=None, help="comma-separated outcome-model moderators (dev)"
            )
    args = ap.parse_args(argv)
    return {"partition": cmd_partition, "simulate": cmd_simulate, "hte": cmd_hte, "ope": cmd_ope}[args.cmd](
        args
    )


if __name__ == "__main__":
    sys.exit(main())
