from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))

from _fake_world import fake_world_factory

from hpe_api.main import create_app
from hpe_api.settings import Settings


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "reports_dir": tmp_path / "reports",
        "cors_origins": ["http://localhost:5173"],
        "max_agents_per_run": 50,
        "max_years_per_run": 100.0,
        "max_concurrent_jobs": 2,
        "rate_limit_per_minute": 600.0,
        "rate_limit_burst": 100,
        "interactive_tick_delay_s": 0.0,
        "background_chunk_ticks": 6,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def client_factory(tmp_path: Path) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(
        world_factory: Any = fake_world_factory, app_kwargs: dict[str, Any] | None = None, **overrides: Any
    ) -> TestClient:
        app = create_app(
            make_settings(tmp_path, **overrides), world_factory=world_factory, **(app_kwargs or {})
        )
        c = TestClient(app)
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(client_factory: Callable[..., TestClient]) -> TestClient:
    return client_factory()


def small_config(**over: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {"seed": 7, "n_agents": 4, "years": 2.0, "dt_days": 30, "policy_class": "random"}
    cfg.update(over)
    return cfg


def wait_for(
    client: TestClient, run_id: str, statuses: tuple[str, ...] = ("completed",), timeout: float = 20.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = client.get(f"/runs/{run_id}").json()
        if info["status"] in statuses:
            return info
        time.sleep(0.02)
    raise AssertionError(
        f"run {run_id} did not reach {statuses}: {client.get(f'/runs/{run_id}').json()['status']}"
    )
