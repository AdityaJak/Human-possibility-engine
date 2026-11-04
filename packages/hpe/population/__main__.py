"""python -m hpe.population run <definition.yaml> --out <dir> [--resume] [--families a,b] | analyze <outdir>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hpe.population")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("definition", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--families", default=None)
    a = sub.add_parser("analyze")
    a.add_argument("outdir", type=Path)
    args = ap.parse_args(argv)
    from hpe.population.core import runner

    if args.cmd == "run":
        fams = args.families.split(",") if args.families else None
        out = runner.run(
            runner.load_definition(args.definition), args.out, resume=args.resume, only_families=fams
        )
        print(f"wrote {out}")
    else:
        runner.analyze(args.outdir)
        print(f"re-analysed {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
