#!/usr/bin/env python3
"""Phase II integrity guards (Independent Critic lane). Makes the registry rules checkable by machine.

    python scripts/phase2_guard.py check-test-access [PATH ...] [--partitions FILE] [--out REPORT.json]
        Scan files for Phase II test-partition participant keys (``<experiment>#<participant>``) and for
        confirmatory synthetic seeds (500-509) in run manifests / experiment definitions outside
        ``reports/phase2/confirmatory/``. Exit 1 on any violation. Paths listed in ALLOWLIST are reported
        (counts) but do not fail: they are disclosed Phase I artifacts that predate the freeze.

    python scripts/phase2_guard.py check-freeze ID [--freeze-dir reports/phase2/freeze] [--require-clean]
        Verify a freeze record: the commit exists and is an ancestor of HEAD, every listed file hashes to
        the recorded SHA-256 *at that commit*, the test-partition SHA-256 matches the committed partitions,
        and (with --require-clean) the working tree is clean. Exit 1 on any mismatch.

    python scripts/phase2_guard.py budget [--registry docs/phase2/EXPERIMENT_REGISTRY.md] [--reports reports/phase2]
        Sum CPU seconds recorded under reports/phase2/<lane>/ (run manifests ``compute.cpu_time_s``,
        or ``budget.json`` files with ``cpu_seconds``) and compare with the registry budgets. Exit 1 if
        any lane is over budget.

No model is fitted here; the guard only reads files and git metadata.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARTITIONS = Path("reports/phase2/partitions/psych101.json")
CONFIRMATORY_SEEDS = frozenset(range(500, 510))
CONFIRMATORY_DIR = "reports/phase2/confirmatory"
FREEZE_DIR = "reports/phase2/freeze"
# Disclosed Phase I artifacts that contain test-participant scores from before the freeze (registry §1,
# "contact disclosure"). Reported, never silently ignored, never a pass for Phase II code or reports.
ALLOWLIST: tuple[str, ...] = ("reports/human_validation/", "reports/phase2/partitions/")
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "data",
}
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".yaml",
    ".yml",
    ".py",
    ".ts",
    ".tsx",
    ".txt",
    ".csv",
    ".log",
    ".toml",
}
KEY_RE = re.compile(r"([A-Za-z0-9_.\-À-ɏ]+/exp\d+\.csv)#([A-Za-z0-9_\-]+)")
LANES = {
    "cognitive": "cognitive-architecture",
    "cognitive-architecture": "cognitive-architecture",
    "human": "human-validation",
    "human-validation": "human-validation",
    "causal": "causal",
    "affective": "affective",
    "social": "social-developmental",
    "social-developmental": "social-developmental",
    "critic": "critic",
}


# ----------------------------------------------------------------------------- helpers
def load_test_keys(partitions: Path) -> set[str]:
    data = json.loads(partitions.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for fam in data["families"].values():
        for exp, v in fam.items():
            keys.update(f"{exp}#{p}" for p in v["test"])
    return keys


def iter_files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_file():
            yield p
            continue
        for f in sorted(p.rglob("*")):
            if f.name.startswith("guard_run") and f.parent.name == "critic":
                continue  # the guard's own reports quote sample keys of disclosed Phase I contact (orchestrator, 2026-11-04)
            if any(part in SKIP_DIRS for part in f.relative_to(p).parts):
                continue
            if f.is_file() and (
                f.suffix in TEXT_SUFFIXES or f.name.endswith(".jsonl.gz") or f.name.endswith(".json.gz")
            ):
                yield f


def read_lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[operator]
            yield from fh
    except (OSError, EOFError):
        return


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _is_allowlisted(rel: str) -> bool:
    return any(rel.startswith(a) for a in ALLOWLIST)


def _seeds_in(obj: Any) -> set[int]:
    """Collect seed-like integers from a manifest/definition: keys named seed(s)/population_seed/task_seed."""
    found: set[int] = set()

    def walk(o: Any, key: str | None) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)
        elif isinstance(o, int) and not isinstance(o, bool) and key and ("seed" in key.lower()):
            found.add(o)

    walk(obj, None)
    return found


# ----------------------------------------------------------------------------- check-test-access
def check_test_access(paths: list[Path], partitions: Path, root: Path = ROOT) -> dict[str, Any]:
    test_keys = load_test_keys(partitions)
    partitions_rel = _rel(partitions, root)
    violations: list[dict[str, Any]] = []
    disclosed: list[dict[str, Any]] = []
    scanned = 0
    for f in iter_files(paths):
        rel = _rel(f, root)
        if rel == partitions_rel:
            continue
        scanned += 1
        hits: dict[str, int] = {}
        in_confirmatory = rel.startswith(CONFIRMATORY_DIR)
        is_manifest = f.name in {"manifest.json", "definition.json", "run_info.json"} or f.suffix in {
            ".yaml",
            ".yml",
        }
        seed_hits: set[int] = set()
        if is_manifest and not in_confirmatory and f.suffix == ".json":
            try:
                seed_hits = _seeds_in(json.loads(f.read_text(encoding="utf-8"))) & CONFIRMATORY_SEEDS
            except (ValueError, OSError):
                seed_hits = set()
        for line in read_lines(f):
            for m in KEY_RE.finditer(line):
                key = f"{m.group(1)}#{m.group(2)}"
                if key in test_keys:
                    hits[key] = hits.get(key, 0) + 1
            if (
                is_manifest
                and not in_confirmatory
                and f.suffix in {".yaml", ".yml"}
                and "seed" in line.lower()
            ):
                for n in re.findall(r"\b(\d{3})\b", line):
                    if int(n) in CONFIRMATORY_SEEDS:
                        seed_hits.add(int(n))
        if hits:
            rec = {
                "file": rel,
                "distinct_test_keys": len(hits),
                "occurrences": sum(hits.values()),
                "sample": sorted(hits)[:3],
            }
            (disclosed if _is_allowlisted(rel) else violations).append(
                {"kind": "test_participant_key", **rec}
            )
        if seed_hits:
            violations.append(
                {
                    "kind": "confirmatory_seed_outside_confirmatory_dir",
                    "file": rel,
                    "seeds": sorted(seed_hits),
                }
            )
    return {
        "check": "check-test-access",
        "partitions": partitions_rel,
        "n_test_keys": len(test_keys),
        "files_scanned": scanned,
        "violations": violations,
        "disclosed_phase1_contact": disclosed,
        "clean": not violations,
    }


# ----------------------------------------------------------------------------- check-freeze
def _git(args: list[str], cwd: Path, *, raw: bool = False) -> tuple[int, str]:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if raw:  # exact file content (keep trailing newline) for hashing
        return r.returncode, r.stdout if r.returncode == 0 else r.stderr.strip()
    return r.returncode, (r.stdout or r.stderr).strip()


def check_freeze(
    freeze_id: str, freeze_dir: Path, repo: Path = ROOT, *, require_clean: bool = False
) -> dict[str, Any]:
    rec_path = freeze_dir / f"{freeze_id}.json"
    problems: list[str] = []
    if not rec_path.exists():
        return {
            "check": "check-freeze",
            "id": freeze_id,
            "record": str(rec_path),
            "problems": [f"no freeze record at {rec_path}"],
            "ok": False,
        }
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    for field in ("id", "commit", "created_at", "preregistration_hash", "files", "test_partition_sha256"):
        if field not in rec:
            problems.append(f"missing field {field!r}")
    commit = str(rec.get("commit", ""))
    if commit:
        code, _ = _git(["cat-file", "-e", f"{commit}^{{commit}}"], repo)
        if code != 0:
            problems.append(f"commit {commit} does not exist")
        else:
            code, _ = _git(["merge-base", "--is-ancestor", commit, "HEAD"], repo)
            if code != 0:
                problems.append(f"commit {commit} is not an ancestor of HEAD")
    for path, want in (rec.get("files") or {}).items():
        code, content = _git(["show", f"{commit}:{path}"], repo, raw=True)
        if code != 0:
            problems.append(f"{path}: not present at {commit}")
            continue
        got = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if got != want:
            problems.append(f"{path}: sha256 {got[:12]} != recorded {str(want)[:12]}")
    parts_path = repo / DEFAULT_PARTITIONS
    if parts_path.exists():
        committed = json.loads(parts_path.read_text(encoding="utf-8")).get("test_partition_sha256")
        if rec.get("test_partition_sha256") != committed:
            problems.append("test_partition_sha256 differs from the committed partitions file")
    if require_clean:
        code, status = _git(["status", "--porcelain"], repo)
        if status:
            problems.append("working tree is not clean")
    return {
        "check": "check-freeze",
        "id": freeze_id,
        "record": _rel(rec_path, repo),
        "commit": commit,
        "problems": problems,
        "ok": not problems,
    }


# ----------------------------------------------------------------------------- budget
def parse_budgets(registry_md: Path) -> dict[str, float]:
    budgets: dict[str, float] = {}
    for line in registry_md.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*([a-z][a-z\-]+)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|", line)
        if m and m.group(1) in LANES.values():
            budgets[m.group(1)] = float(m.group(2))
    return budgets


def _lane_of(rel: str) -> str | None:
    parts = rel.split("/")
    if len(parts) >= 3 and parts[0] == "reports" and parts[1] == "phase2":
        return LANES.get(parts[2])
    return None


def budget(registry_md: Path, reports: Path, root: Path = ROOT) -> dict[str, Any]:
    budgets = parse_budgets(registry_md)
    spent: dict[str, float] = {k: 0.0 for k in budgets}
    sources: dict[str, int] = {k: 0 for k in budgets}
    for f in iter_files([reports]):
        if f.suffix != ".json" or f.name not in {"manifest.json", "budget.json", "run_info.json"}:
            continue
        lane = _lane_of(_rel(f, root))
        if lane is None:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        cpu = None
        if isinstance(d.get("compute"), dict):
            cpu = d["compute"].get("cpu_time_s", d["compute"].get("wall_time_s"))
        if cpu is None:
            cpu = d.get("cpu_seconds", d.get("cpu_time_s"))
        if isinstance(cpu, int | float):
            spent[lane] = spent.get(lane, 0.0) + float(cpu)
            sources[lane] = sources.get(lane, 0) + 1
    rows = [
        {
            "lane": lane,
            "budget_cpu_h": budgets.get(lane),
            "spent_cpu_h": round(spent.get(lane, 0.0) / 3600.0, 3),
            "sources": sources.get(lane, 0),
            "over": budgets.get(lane) is not None and spent.get(lane, 0.0) / 3600.0 > budgets[lane],
        }
        for lane in sorted(set(budgets) | set(spent))
    ]
    return {
        "check": "budget",
        "registry": _rel(registry_md, root),
        "lanes": rows,
        "ok": not any(r["over"] for r in rows),
    }


# ----------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="phase2_guard", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("check-test-access")
    a.add_argument("paths", nargs="*", default=["."])
    a.add_argument("--partitions", default=str(DEFAULT_PARTITIONS))
    a.add_argument("--out")
    b = sub.add_parser("check-freeze")
    b.add_argument("id")
    b.add_argument("--freeze-dir", default=FREEZE_DIR)
    b.add_argument("--require-clean", action="store_true")
    b.add_argument("--out")
    c = sub.add_parser("budget")
    c.add_argument("--registry", default="docs/phase2/EXPERIMENT_REGISTRY.md")
    c.add_argument("--reports", default="reports/phase2")
    c.add_argument("--out")
    args = ap.parse_args(argv)
    if args.cmd == "check-test-access":
        report = check_test_access([Path(p) for p in args.paths], Path(args.partitions))
        ok = report["clean"]
    elif args.cmd == "check-freeze":
        report = check_freeze(args.id, Path(args.freeze_dir), require_clean=args.require_clean)
        ok = report["ok"]
    else:
        report = budget(Path(args.registry), Path(args.reports))
        ok = report["ok"]
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
