"""Runtime settings from environment variables (see .env.example). No secrets are required."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("HPE_DATA_DIR", "./data")))
    reports_dir: Path = field(default_factory=lambda: Path(os.environ.get("HPE_REPORTS_DIR", "./reports")))
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.environ.get("HPE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(
                ","
            )
            if o.strip()
        ]
    )
    max_agents_per_run: int = field(default_factory=lambda: _int("HPE_MAX_AGENTS_PER_RUN", 2000))
    max_years_per_run: float = field(default_factory=lambda: _float("HPE_MAX_YEARS_PER_RUN", 100.0))
    max_concurrent_jobs: int = field(default_factory=lambda: max(1, _int("HPE_MAX_CONCURRENT_JOBS", 2)))
    max_interactive_runs: int = field(default_factory=lambda: max(1, _int("HPE_MAX_INTERACTIVE_RUNS", 4)))
    llm_provider: str = field(default_factory=lambda: os.environ.get("HPE_LLM_PROVIDER", "none"))
    llm_max_calls_per_hour: int = field(default_factory=lambda: _int("HPE_LLM_MAX_CALLS_PER_HOUR", 0))
    rate_limit_per_minute: float = field(default_factory=lambda: _float("HPE_RATE_LIMIT_PER_MINUTE", 30.0))
    rate_limit_burst: int = field(default_factory=lambda: _int("HPE_RATE_LIMIT_BURST", 10))
    interactive_tick_delay_s: float = field(
        default_factory=lambda: _float("HPE_INTERACTIVE_TICK_DELAY_S", 0.05)
    )
    background_chunk_ticks: int = field(
        default_factory=lambda: max(1, _int("HPE_BACKGROUND_CHUNK_TICKS", 12))
    )
    save_checkpoints: bool = field(
        default_factory=lambda: os.environ.get("HPE_SAVE_CHECKPOINTS", "1") not in ("0", "false")
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "hpe.sqlite3"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"
