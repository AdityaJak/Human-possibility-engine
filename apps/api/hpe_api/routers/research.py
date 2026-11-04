"""Experiments, research metrics, claims (Evidence page), benchmarks, health."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends

from hpe import __version__
from hpe.contracts.api import BenchmarkRecord, ClaimRecord, ExperimentCreate, ExperimentInfo, Health
from hpe.contracts.runtime import CONTRACT_VERSION
from hpe_api.context import AppContext, get_ctx
from hpe_api.deps import rate_limited
from hpe_api.experiment_service import read_json_records
from hpe_api.run_service import world_engine_available

log = logging.getLogger("hpe_api.research")
router = APIRouter()


@router.get("/health", response_model=Health, tags=["health"])
def health(ctx: AppContext = Depends(get_ctx)) -> Health:
    return Health(
        status="ok" if world_engine_available() else "degraded",
        version=__version__,
        contract_version=CONTRACT_VERSION,
        llm_provider=ctx.settings.llm_provider,
        offline_demo_runs=ctx.loader.count_demo_runs(),
    )


# ------------------------------------------------------------------ experiments
@router.post("/experiments", response_model=ExperimentInfo, status_code=201, tags=["experiments"])
def create_experiment(body: ExperimentCreate, ctx: AppContext = Depends(get_ctx)) -> ExperimentInfo:
    """Register (preregister) an experiment. The definition hash is frozen before any run."""
    return ctx.experiments.create(body.definition)


@router.get("/experiments", response_model=list[ExperimentInfo], tags=["experiments"])
def list_experiments(ctx: AppContext = Depends(get_ctx)) -> list[ExperimentInfo]:
    return ctx.experiments.list_all()


@router.post(
    "/experiments/{experiment_id}/run",
    response_model=ExperimentInfo,
    tags=["experiments"],
    dependencies=[Depends(rate_limited("experiments"))],
)
def run_experiment(experiment_id: str, ctx: AppContext = Depends(get_ctx)) -> ExperimentInfo:
    """Execute a registered experiment as a job (501 if the evaluation runner is not part of this build)."""
    return ctx.experiments.run(experiment_id)


@router.get("/experiments/{experiment_id}", response_model=ExperimentInfo, tags=["experiments"])
def get_experiment(experiment_id: str, ctx: AppContext = Depends(get_ctx)) -> ExperimentInfo:
    return ctx.experiments.get(experiment_id)


@router.get("/experiments/{experiment_id}/results", response_model=ExperimentInfo, tags=["experiments"])
def experiment_results(experiment_id: str, ctx: AppContext = Depends(get_ctx)) -> ExperimentInfo:
    return ctx.experiments.get(experiment_id)


# ------------------------------------------------------------------ research summaries
def _claims(ctx: AppContext) -> list[ClaimRecord]:
    p = ctx.settings.reports_dir / "evidence" / "claims.json"
    if not p.exists():
        return []
    out = []
    for rec in read_json_records([p]):
        try:
            out.append(ClaimRecord.model_validate(rec))
        except ValueError as e:
            log.warning("skipping invalid claim record: %s", e)
    return out


def _benchmarks(ctx: AppContext) -> list[BenchmarkRecord]:
    root = ctx.settings.reports_dir / "scale"
    if not root.exists():
        return []
    out = []
    for rec in read_json_records(sorted(root.glob("*.json"))):
        try:
            out.append(BenchmarkRecord.model_validate(rec))
        except ValueError as e:
            log.warning("skipping invalid benchmark record: %s", e)
    return out


@router.get("/research/claims", response_model=list[ClaimRecord], tags=["research"])
def research_claims(ctx: AppContext = Depends(get_ctx)) -> list[ClaimRecord]:
    """Claims ledger for the investor Evidence page. Read verbatim from reports/evidence/claims.json."""
    return _claims(ctx)


@router.get("/benchmarks", response_model=list[BenchmarkRecord], tags=["research"])
def benchmarks(ctx: AppContext = Depends(get_ctx)) -> list[BenchmarkRecord]:
    """Measured scale benchmarks only (reports/scale/*.json)."""
    return _benchmarks(ctx)


@router.get("/research/metrics", tags=["research"])
def research_metrics(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    runs = ctx.runs.list_runs(include_experiment_runs=True)
    return {
        "contract_version": CONTRACT_VERSION,
        "experiments": ctx.experiments.summary(),
        "runs": {
            "live": sum(1 for r in runs if r.source == "live"),
            "stored_real_output": sum(1 for r in runs if r.source == "stored_real_output"),
            "total": len(runs),
        },
        "benchmarks": len(_benchmarks(ctx)),
        "claims": len(_claims(ctx)),
        "note": "Synthetic research prototype; not validated for predicting human futures.",
    }
