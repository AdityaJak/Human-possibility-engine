"""Application context: one instance per FastAPI app, holding services and repositories."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from hpe.futures import DEFAULT_REGISTRY, EvidenceRegistry, NarrationGate, ScenarioEngine, build_provider
from hpe.storage import (
    AuditLog,
    Database,
    EventRepository,
    EvidenceRepository,
    StoredRunLoader,
    UserRepository,
)
from hpe_api.experiment_service import ExperimentRunner, ExperimentService
from hpe_api.ratelimit import TokenBucketLimiter
from hpe_api.run_service import PolicyResolver, RunService, WorldFactory
from hpe_api.settings import Settings


@dataclass
class AppContext:
    settings: Settings
    db: Database
    loader: StoredRunLoader
    runs: RunService
    experiments: ExperimentService
    events: EventRepository
    users: UserRepository
    audit: AuditLog
    evidence_repo: EvidenceRepository
    registry: EvidenceRegistry
    engine: ScenarioEngine
    limiter: TokenBucketLimiter

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        world_factory: WorldFactory | None = None,
        policy_resolver: PolicyResolver | None = None,
        experiment_runner_resolver: object | None = None,
    ) -> AppContext:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        db = Database(settings.db_path)
        loader = StoredRunLoader(settings.reports_dir)
        runs = RunService(settings, db, loader, world_factory=world_factory, policy_resolver=policy_resolver)
        experiments = ExperimentService(
            settings,
            db,
            runner_resolver=experiment_runner_resolver,  # type: ignore[arg-type]
        )
        evidence_repo = EvidenceRepository(db)
        for rec in DEFAULT_REGISTRY.all():  # the static library is persisted in the evidence namespace
            evidence_repo.upsert(rec)
        gate = NarrationGate(
            build_provider(settings.llm_provider), max_calls_per_hour=settings.llm_max_calls_per_hour
        )
        return cls(
            settings=settings,
            db=db,
            loader=loader,
            runs=runs,
            experiments=experiments,
            events=EventRepository(db),
            users=UserRepository(db),
            audit=AuditLog(db),
            evidence_repo=evidence_repo,
            registry=DEFAULT_REGISTRY,
            engine=ScenarioEngine(DEFAULT_REGISTRY, narration=gate),
            limiter=TokenBucketLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst),
        )

    def close(self) -> None:
        self.runs.shutdown()
        self.experiments.shutdown()
        self.db.close()


def get_ctx(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.ctx
    return ctx


__all__ = ["AppContext", "ExperimentRunner", "get_ctx"]
