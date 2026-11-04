"""python -m hpe.population.continual gate-e <definition.yaml> --out <dir> [--resume]
python -m hpe.population.continual gate-d <definition.yaml> --out <dir> [--resume] [--targets a,b]
python -m hpe.population.continual gate-d-analyze <outdir>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hpe.population.continual")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("gate-e")
    e.add_argument("definition", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--resume", action="store_true")
    d = sub.add_parser("gate-d")
    d.add_argument("definition", type=Path)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--resume", action="store_true")
    d.add_argument("--targets", default=None)
    da = sub.add_parser("gate-d-analyze")
    da.add_argument("outdir", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "gate-e":
        from hpe.population.continual import gate_e

        out = gate_e.run(gate_e.load_definition(args.definition), args.out, resume=args.resume)
        print(f"wrote {out}")
    elif args.cmd == "gate-d":
        from hpe.population.continual import transfer

        targets = args.targets.split(",") if args.targets else None
        out = transfer.run(
            transfer.load_definition(args.definition), args.out, resume=args.resume, targets=targets
        )
        print(f"wrote {out}")
    else:
        from hpe.population.continual import transfer

        transfer.analyze(args.outdir)
        print(f"re-analysed {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
