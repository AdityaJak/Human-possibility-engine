"""Guard tooling for the Phase II registry rules (critic lane). Fixtures are tiny and synthetic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "phase2_guard.py"
spec = importlib.util.spec_from_file_location("phase2_guard", SCRIPT)
assert spec and spec.loader
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

PARTS = {
    "split_seed": 1,
    "families": {
        "fam": {
            "a2020x/exp1.csv": {
                "n": 4,
                "dev": ["1"],
                "tune": ["2"],
                "test": ["3", "4"],
                "counts": {"dev": 1, "tune": 1, "test": 2},
            }
        }
    },
    "test_partition_sha256": hashlib.sha256(b"a2020x/exp1.csv#3\na2020x/exp1.csv#4").hexdigest(),
}


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    parts = tmp_path / "reports/phase2/partitions/psych101.json"
    parts.parent.mkdir(parents=True)
    parts.write_text(json.dumps(PARTS))
    return tmp_path


def test_test_key_in_phase2_report_is_a_violation(repo: Path) -> None:
    bad = repo / "reports/phase2/cognitive/dev/outcomes.jsonl"
    bad.parent.mkdir(parents=True)
    bad.write_text('{"agent_id": "a2020x/exp1.csv#3", "v": 1}\n{"agent_id": "a2020x/exp1.csv#1"}\n')
    rep = guard.check_test_access([repo], repo / "reports/phase2/partitions/psych101.json", root=repo)
    assert not rep["clean"]
    assert rep["violations"][0]["kind"] == "test_participant_key"
    assert rep["violations"][0]["distinct_test_keys"] == 1 and rep["violations"][0]["sample"] == [
        "a2020x/exp1.csv#3"
    ]


def test_allowlisted_phase1_artifact_is_disclosed_not_failed(repo: Path) -> None:
    p1 = repo / "reports/human_validation/old/outcomes.jsonl"
    p1.parent.mkdir(parents=True)
    p1.write_text('{"agent_id": "a2020x/exp1.csv#4"}\n')
    rep = guard.check_test_access([repo], repo / "reports/phase2/partitions/psych101.json", root=repo)
    assert rep["clean"] and rep["violations"] == []
    assert rep["disclosed_phase1_contact"][0]["file"].startswith("reports/human_validation/")


def test_dev_only_keys_are_clean_and_partitions_file_skipped(repo: Path) -> None:
    ok = repo / "reports/phase2/human/dev.json"
    ok.parent.mkdir(parents=True)
    ok.write_text(json.dumps({"rows": ["a2020x/exp1.csv#1", "a2020x/exp1.csv#2"]}))
    rep = guard.check_test_access([repo], repo / "reports/phase2/partitions/psych101.json", root=repo)
    assert rep["clean"] and rep["disclosed_phase1_contact"] == []


def test_confirmatory_seed_outside_confirmatory_dir(repo: Path) -> None:
    m = repo / "reports/phase2/cognitive/runs/r1/manifest.json"
    m.parent.mkdir(parents=True)
    m.write_text(json.dumps({"config": {"seed": 505, "n_agents": 10}}))
    rep = guard.check_test_access([repo], repo / "reports/phase2/partitions/psych101.json", root=repo)
    assert [v["kind"] for v in rep["violations"]] == ["confirmatory_seed_outside_confirmatory_dir"]
    assert rep["violations"][0]["seeds"] == [505]
    y = repo / "experiments/phase2/x.yaml"
    y.parent.mkdir(parents=True)
    y.write_text("seeds: [1000, 1001, 507]\n")
    rep = guard.check_test_access([repo], repo / "reports/phase2/partitions/psych101.json", root=repo)
    assert sum(v["kind"].startswith("confirmatory_seed") for v in rep["violations"]) == 2


def test_confirmatory_seed_inside_confirmatory_dir_is_fine(repo: Path) -> None:
    m = repo / "reports/phase2/confirmatory/c1/runs/r/manifest.json"
    m.parent.mkdir(parents=True)
    m.write_text(json.dumps({"config": {"seed": 500}}))
    rep = guard.check_test_access([repo], repo / "reports/phase2/partitions/psych101.json", root=repo)
    assert rep["clean"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture()
def git_repo(repo: Path) -> Path:
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "root")
    cfg = repo / "experiments/phase2/c.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("hypothesis: x\nseeds: [500]\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "cfg")
    return repo


def _freeze(repo: Path, commit: str, sha: str, test_sha: str) -> Path:
    rec = repo / "reports/phase2/freeze/C-1.json"
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text(
        json.dumps(
            {
                "id": "C-1",
                "commit": commit,
                "created_at": "2026-11-04",
                "preregistration_hash": "abc",
                "files": {"experiments/phase2/c.yaml": sha},
                "test_partition_sha256": test_sha,
            }
        )
    )
    return rec


def test_check_freeze_passes_on_matching_record(git_repo: Path) -> None:
    commit = _git(git_repo, "rev-parse", "HEAD")
    sha = hashlib.sha256((git_repo / "experiments/phase2/c.yaml").read_bytes()).hexdigest()
    _freeze(git_repo, commit, sha, PARTS["test_partition_sha256"])
    rep = guard.check_freeze("C-1", git_repo / "reports/phase2/freeze", repo=git_repo)
    assert rep["ok"], rep


def test_check_freeze_detects_tampering_and_dirty_tree(git_repo: Path) -> None:
    commit = _git(git_repo, "rev-parse", "HEAD")
    _freeze(git_repo, commit, "0" * 64, "deadbeef")
    rep = guard.check_freeze("C-1", git_repo / "reports/phase2/freeze", repo=git_repo, require_clean=True)
    assert not rep["ok"]
    joined = " ".join(rep["problems"])
    assert "sha256" in joined and "test_partition_sha256" in joined and "not clean" in joined
    assert not guard.check_freeze("nope", git_repo / "reports/phase2/freeze", repo=git_repo)["ok"]


def test_budget_sums_manifests_and_flags_overrun(repo: Path) -> None:
    reg = repo / "docs/phase2/EXPERIMENT_REGISTRY.md"
    reg.parent.mkdir(parents=True)
    reg.write_text("| Lane | Budget | Spent |\n|---|---|---|\n| causal | 1 | 0 |\n| critic | 6 | 0 |\n")
    m = repo / "reports/phase2/causal/runs/a/manifest.json"
    m.parent.mkdir(parents=True)
    m.write_text(json.dumps({"compute": {"cpu_time_s": 5400.0}}))
    (repo / "reports/phase2/critic").mkdir(parents=True)
    (repo / "reports/phase2/critic/budget.json").write_text(json.dumps({"cpu_seconds": 60}))
    rep = guard.budget(reg, repo / "reports/phase2", root=repo)
    by = {r["lane"]: r for r in rep["lanes"]}
    assert by["causal"]["spent_cpu_h"] == 1.5 and by["causal"]["over"]
    assert by["critic"]["spent_cpu_h"] == pytest.approx(60 / 3600, abs=1e-3) and not by["critic"]["over"]
    assert not rep["ok"]
