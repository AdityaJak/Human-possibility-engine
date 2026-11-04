"""Tests for scripts/scan_secrets.py. Fake credentials are assembled at runtime so this file
itself never contains a literal secret-shaped string."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("scan_secrets", ROOT / "scripts" / "scan_secrets.py")
assert _spec and _spec.loader
scan_secrets = importlib.util.module_from_spec(_spec)
sys.modules["scan_secrets"] = scan_secrets
_spec.loader.exec_module(scan_secrets)


def _kinds(text: str) -> set[str]:
    return {f.kind for f in scan_secrets.scan_text(text, "x")}


def test_detects_high_signal_patterns() -> None:
    assert "aws_access_key_id" in _kinds("key = " + "AKIA" + "Q" * 16)
    assert "github_token" in _kinds("token: " + "ghp_" + "a1" * 20)
    assert "anthropic_api_key" in _kinds("ANTHROPIC_API_KEY=" + "sk-" + "ant-" + "b" * 30)
    assert "private_key" in _kinds("-----BEGIN " + "RSA PRIVATE KEY-----")
    assert "generic_assignment" in _kinds('api_key = "' + "Z9" * 12 + '"')


def test_ignores_benign_text_and_allow_marker() -> None:
    assert _kinds("api_key = os.environ['ANTHROPIC_API_KEY']") == set()
    assert _kinds("# ANTHROPIC_API_KEY=             # set only in your local .env") == set()
    assert _kinds("token = " + "ghp_" + "a1" * 20 + "  # secret-scan: ignore") == set()


def test_redacts_excerpts() -> None:
    f = scan_secrets.scan_text("x=" + "AKIA" + "Q" * 16, "f.py")[0]
    assert "redacted" in f.excerpt and ("Q" * 16) not in f.excerpt


def test_flags_committed_env_files_but_not_example(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("HPE_DATA_DIR=./data\n")
    example = tmp_path / ".env.example"
    example.write_text("HPE_DATA_DIR=./data\n")
    kinds = {f.kind for f in scan_secrets.scan_paths([env, example], tmp_path)}
    assert kinds == {"committed_env_file"}


def test_repository_is_clean() -> None:
    findings = scan_secrets.scan_paths(scan_secrets.candidate_files(ROOT), ROOT)
    assert findings == [], findings
