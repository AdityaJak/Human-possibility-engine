"""CLI: ``python -m hpe.evaluation.human <command>``.

run experiments/exp07_human_heldout.yaml   [--data PATH] [--out DIR] [--no-robustness]
run experiments/exp07b_causal_grounding.yaml [--data PATH] [--out DIR]
analyze reports/human_validation/<experiment_id>   # regenerate result/figures/report from logs
fetch psych101|heartsteps                        # download + verify into datasets/external/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hpe.evaluation.human import data
from hpe.evaluation.human.definition import CausalGroundingDefinition, HumanHeldoutDefinition, load_definition

DEFAULT_OUT = data.REPO_ROOT / "reports" / "human_validation"


def _resolve(source: data.DatasetSource, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return data.locate(source)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hpe.evaluation.human")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("definition")
    r.add_argument("--data")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--no-robustness", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("outdir")
    r2 = sub.add_parser("run2", help="Phase II partition-aware evaluation (kind: human_v2)")
    r2.add_argument("definition")
    r2.add_argument("--data")
    r2.add_argument("--out", default=str(data.REPO_ROOT / "reports" / "phase2" / "human"))
    r2.add_argument(
        "--report-partition",
        choices=["dev", "tune", "test"],
        help="override the definition (test needs a freeze record)",
    )
    a2 = sub.add_parser("analyze2", help="regenerate summary/report of a Phase II run from its logs")
    a2.add_argument("outdir")
    f = sub.add_parser("fetch")
    f.add_argument("dataset", choices=["psych101", "heartsteps"])
    args = ap.parse_args(argv)

    from hpe.evaluation.human import runner

    if args.cmd == "run":
        defn = load_definition(Path(args.definition))
        outdir = Path(args.out) / defn.experiment_id
        if isinstance(defn, HumanHeldoutDefinition):
            try:
                path = _resolve(data.PSYCH101_LEARNING_SUBSET, args.data)
            except data.DatasetUnavailable:
                path = _resolve(data.PSYCH101_FULL, args.data)
            out = runner.run_exp07(defn, path, outdir, robustness=not args.no_robustness)
        else:
            assert isinstance(defn, CausalGroundingDefinition)
            out = runner.run_exp07b(defn, _resolve(data.HEARTSTEPS_SUGGESTIONS, args.data), outdir)
        print(f"status: {out['result']['status']} — {out['result']['status_reason']}")
        print(f"wrote {outdir}")
        return 0
    if args.cmd == "run2":
        from hpe.evaluation.human import runner2
        from hpe.evaluation.human.definition import HumanV2Definition

        defn = load_definition(Path(args.definition))
        if not isinstance(defn, HumanV2Definition):
            print("run2 needs a definition with kind: human_v2", file=sys.stderr)
            return 2
        if args.report_partition:
            defn.report_partition = args.report_partition
        try:
            path = _resolve(data.PSYCH101_LEARNING_SUBSET, args.data)
        except data.DatasetUnavailable:
            path = _resolve(data.PSYCH101_FULL, args.data)
        out = runner2.run_v2(defn, path, Path(args.out) / defn.experiment_id)
        for par, s in out["summary"]["paradigms"].items():
            best = min(
                (m for m, v in s["models"].items() if v.get("nll_mean") is not None),
                key=lambda m: s["models"][m]["nll_mean"],
                default=None,
            )
            print(f"{par}: best model on {defn.report_partition} = {best}")
        print(f"wrote {Path(args.out) / defn.experiment_id}")
        return 0
    if args.cmd == "analyze2":
        from hpe.evaluation.human import runner2

        runner2.finalize_v2(Path(args.outdir))
        print(f"regenerated {args.outdir}")
        return 0
    if args.cmd == "analyze":
        outdir = Path(args.outdir)
        fn = runner.finalize_exp07 if (outdir / "fits.json").exists() else runner.finalize_exp07b
        out = fn(outdir)
        print(f"regenerated {outdir}: status {out['result']['status']}")
        return 0
    if args.cmd == "fetch":
        if args.dataset == "heartsteps":
            dest = data.default_path(data.HEARTSTEPS_SUGGESTIONS)
            data.download(data.HEARTSTEPS_SUGGESTIONS.url, dest, data.HEARTSTEPS_SUGGESTIONS.sha256)
        else:
            full = data.default_path(data.PSYCH101_FULL)
            if not full.exists():
                data.download(data.PSYCH101_FULL.url, full, data.PSYCH101_FULL.sha256)
            dest = data.default_path(data.PSYCH101_LEARNING_SUBSET)
            data.filter_jsonl(full, dest, data.LEARNING_EXPERIMENT_PREFIXES)
            data.verify(dest, data.PSYCH101_LEARNING_SUBSET.sha256)
        print(f"verified {dest}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
