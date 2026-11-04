"""``python -m hpe.evaluation`` — run preregistered experiments and regenerate reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hpe.evaluation.spec import (
    is_simulation_spec_file,
    load_spec,
    scaled_for_pipeline_validation,
    simulation_spec_files,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hpe.evaluation", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser(
        "run", help="run an experiment YAML (preregisters, runs arms x seeds, analyses raw logs)"
    )
    r.add_argument("spec", nargs="+", help="experiment YAML file(s)")
    r.add_argument("--out", default="reports/experiments")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--seeds", type=int, default=None, help="pipeline validation only: number of seeds")
    r.add_argument(
        "--pipeline-validation", action="store_true", help="scaled-down run, labelled as pipeline validation"
    )
    r.add_argument("--scale", type=float, default=0.25, help="budget scale for --pipeline-validation")
    r.add_argument(
        "--force", action="store_true", help="overwrite a directory with a different preregistration hash"
    )
    r.add_argument(
        "--no-skip-missing", action="store_true", help="fail instead of skipping unregistered policy arms"
    )
    a = sub.add_parser("analyze", help="regenerate result.json/report.md/figures from raw logs")
    a.add_argument("exp_dir", nargs="+")
    a.add_argument("--no-figures", action="store_true")
    c = sub.add_parser(
        "contamination", help="leakage / measurement-integrity audit for all registered policies"
    )
    c.add_argument("--out", default=None)
    c.add_argument("--seed", type=int, default=0)
    sub.add_parser("list", help="list experiment YAMLs")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        from hpe.evaluation.runner import run_experiment

        if args.seeds is not None and not args.pipeline_validation:
            ap.error("--seeds changes the preregistered design; only allowed with --pipeline-validation")
        out = Path(args.out) / ("pipeline_validation" if args.pipeline_validation else "")
        rc = 0
        for path in args.spec:
            spec = load_spec(path)
            if args.pipeline_validation:
                spec = scaled_for_pipeline_validation(spec, n_seeds=args.seeds or 3, scale=args.scale)
            res = run_experiment(
                spec, out, workers=args.workers, skip_missing_arms=not args.no_skip_missing, force=args.force
            )
            if res is not None:
                print(f"[{res.status.upper()}] {res.experiment_id}: {res.status_reason}")
                print(f"  -> {out / res.experiment_id / 'report.md'}")
        return rc
    if args.cmd == "analyze":
        from hpe.evaluation.analysis import analyze

        for d in args.exp_dir:
            res = analyze(d, make_figures=not args.no_figures)
            print(f"[{res.status.upper()}] {res.experiment_id}: {res.status_reason}")
        return 0
    if args.cmd == "contamination":
        from hpe.evaluation.contamination import run_contamination_suite

        rep = run_contamination_suite(seed=args.seed)
        txt = json.dumps(rep, indent=2, sort_keys=True)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(txt + "\n")
        print(txt)
        return 0 if rep["clean"] else 1
    if args.cmd == "list":
        for p in simulation_spec_files("experiments"):
            s = load_spec(p)
            print(f"{p}  [{s.protocol.kind}]  {s.definition.title}")
        for p in sorted(Path("experiments").glob("*.yaml")):
            if not is_simulation_spec_file(p):
                print(f"{p}  [human dataset — run with python -m hpe.evaluation.human]")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
