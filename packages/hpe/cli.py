"""``hpe`` command-line interface. Subcommands are wired up as lanes integrate."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hpe", description="Human Possibility Engine CLI (no API keys required)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="print versions and registered policy classes")
    args = ap.parse_args(argv)
    if args.cmd == "info":
        from hpe import __version__
        from hpe.contracts import CONTRACT_VERSION
        from hpe.contracts.registry import registered

        print(f"hpe {__version__} contract {CONTRACT_VERSION} policies={registered()}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
