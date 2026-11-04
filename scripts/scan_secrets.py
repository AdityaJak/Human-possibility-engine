#!/usr/bin/env python3
"""Dependency-free secrets scanner used by CI (``python scripts/scan_secrets.py``).

Scans files that git tracks or would track (``git ls-files -co --exclude-standard``) for
high-signal credential patterns and committed ``.env`` files. Exit code 1 if anything is found.
A line can be allow-listed with the marker ``secret-scan: ignore`` (use sparingly; reviewed).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"),
    "aws_access_key_id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b"),
    "anthropic_api_key": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),
    "openai_style_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}\b"),
    "slack_token": re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "stripe_live_key": re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{20,}\b"),
    "generic_assignment": re.compile(
        r"""(?i)\b(?:api[_-]?key|secret[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|password)\b"""
        r"""\s*[:=]\s*['"]([A-Za-z0-9+/_\-]{20,})['"]"""
    ),
}

SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".gz",
    ".zip",
    ".woff",
    ".woff2",
    ".ttf",
    ".parquet",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".pyc",
}
SKIP_NAMES = {"package-lock.json", "uv.lock"}
ALLOW_MARKER = "secret-scan: ignore"
MAX_BYTES = 2_000_000


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    kind: str
    excerpt: str


def candidate_files(root: Path) -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return [root / p for p in out.splitlines() if p]
    except (OSError, subprocess.CalledProcessError):
        skip_dirs = {".git", ".venv", "node_modules", "data", "dist", "__pycache__", ".claude"}
        return [
            p for p in root.rglob("*") if p.is_file() and not (set(p.relative_to(root).parts) & skip_dirs)
        ]


def _redact(s: str) -> str:
    s = s.strip()
    return (s[:12] + "…[redacted]") if len(s) > 12 else "[redacted]"


def scan_text(text: str, path: str) -> list[Finding]:
    findings: list[Finding] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        for kind, pat in PATTERNS.items():
            m = pat.search(line)
            if m:
                findings.append(Finding(path, i, kind, _redact(m.group(0))))
    return findings


def scan_paths(paths: Iterable[Path], root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for p in paths:
        rel = str(p.relative_to(root)) if p.is_absolute() and p.is_relative_to(root) else str(p)
        name = p.name
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            findings.append(Finding(rel, 0, "committed_env_file", name))
            continue
        if p.suffix.lower() in SKIP_SUFFIXES or name in SKIP_NAMES or not p.is_file():
            continue
        try:
            if p.stat().st_size > MAX_BYTES:
                continue
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        findings.extend(scan_text(text, rel))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="files to scan (default: all git-visible files)")
    args = ap.parse_args(argv)
    paths = [Path(p).resolve() for p in args.paths] if args.paths else candidate_files(ROOT)
    findings = scan_paths(paths, ROOT)
    for f in findings:
        print(f"{f.path}:{f.line}: {f.kind}: {f.excerpt}")
    print(f"secret scan: {len(paths)} files scanned, {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
