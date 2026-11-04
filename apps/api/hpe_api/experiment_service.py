"""Experiment registration, execution (via the evaluation lane's runner) and stored results.

Preregistration: on creation the definition is hashed (``stable_hash`` of the definition without
``preregistration_hash``/``created_at``) and frozen to ``<data>/experiments/<id>/definition.json``
before any run. Results are only ever read from the runner's output or from committed
``reports/experiments/<id>/result.json``; this service never computes or edits results itself.

Runner contract expected from ``hpe.evaluation.runner`` (science/evaluation lane)::

    run_experiment(definition: ExperimentDefinition, out_dir: pathlib.Path) -> ExperimentResult | dict | None
"""

from __future__ import annotations

import gzip
import importlib
import json
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from hpe.contracts.api import ExperimentInfo
from hpe.contracts.schemas import ExperimentDefinition, ExperimentResult, JobStatus, OutcomeObservation
from hpe.core.hashing import stable_hash
from hpe.storage import Database, ExperimentRepository
from hpe_api.errors import Conflict, NotFound, NotImplementedYet
from hpe_api.settings import Settings

log = logging.getLogger("hpe_api.experiments")

ExperimentRunner = Callable[[ExperimentDefinition, Path], Any]


def preregistration_hash(defn: ExperimentDefinition) -> str:
    return stable_hash(defn.model_dump(mode="json", exclude={"preregistration_hash", "created_at"}))


def default_runner_resolver() -> ExperimentRunner:
    try:
        mod = importlib.import_module("hpe.evaluation.runner")
    except ImportError as e:
        raise NotImplementedYet(
            "experiment execution is not available in this build: hpe.evaluation.runner is not installed "
            f"({e}). Run experiments from the CLI and serve their stored results instead."
        ) from e
    fn = getattr(mod, "run_experiment", None)
    if not callable(fn):
        raise NotImplementedYet(
            "experiment execution is not available in this build: hpe.evaluation.runner.run_experiment is missing"
        )

    def _run(defn: ExperimentDefinition, out_dir: Path) -> Any:
        try:
            return fn(defn, out_dir=out_dir)
        except TypeError:
            return fn(defn)

    return _run


def _outcome_files(exp_dir: Path) -> list[Path]:
    """Outcome logs of an experiment: a legacy top-level file, or one per run under ``runs/``.

    Sensitivity variants reuse arm names and seeds, so when ``__main__`` runs exist only those are
    served; the page would otherwise see duplicate (arm, seed) pairs.
    """
    top = [exp_dir / n for n in ("outcomes.jsonl.gz", "outcomes.jsonl") if (exp_dir / n).exists()]
    if top:
        return top[:1]
    runs = sorted(p for p in exp_dir.glob("runs/*/outcomes.jsonl*") if p.is_file())
    main = [p for p in runs if "__main__" in p.parent.name]
    return main or runs


def _read_rows(p: Path) -> list[OutcomeObservation]:
    opener = gzip.open if p.name.endswith(".gz") else open
    rows: list[OutcomeObservation] = []
    with opener(p, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(OutcomeObservation.model_validate_json(line))
            except ValueError:
                continue
    return rows


def _is_curve(o: OutcomeObservation) -> bool:
    return o.metric.startswith(("curve/", "transfer_curve/"))


def _read_outcomes(exp_dir: Path, limit: int = 4000) -> list[OutcomeObservation]:
    """A bounded, representative sample of an experiment's outcome rows for the web pages.

    Run-level rows only (per-agent rows are regenerable from the raw logs and too many to ship),
    leakage counters dropped; scalar metrics are kept in full and curve rows are thinned by a fixed
    stride so every run fits the same per-run budget. Nothing is aggregated or altered.
    """
    files = _outcome_files(exp_dir)
    if not files:
        return []
    per_run = max(60, limit // len(files))
    out: list[OutcomeObservation] = []
    for p in files:
        rows = [o for o in _read_rows(p) if not o.metric.startswith("leakage/")]
        run_level = [o for o in rows if o.agent_id is None]
        rows = run_level or rows
        scalars = [o for o in rows if not _is_curve(o)]
        curves = [o for o in rows if _is_curve(o)]
        room = max(0, per_run - len(scalars))
        if curves and room < len(curves):
            stride = -(-len(curves) // room) if room else 0
            curves = curves[::stride] if stride else []
        out.extend(scalars[:per_run])
        out.extend(curves)
        if len(out) >= limit * 2:
            break
    return out


class ExperimentService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        *,
        runner_resolver: Callable[[], ExperimentRunner] | None = None,
    ) -> None:
        self.settings = settings
        self.repo = ExperimentRepository(db)
        self.runner_resolver = runner_resolver or default_runner_resolver
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hpe-exp")
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ stored (committed) experiments
    def _stored_dir(self, experiment_id: str) -> Path:
        return self.settings.reports_dir / "experiments" / experiment_id

    def _stored_ids(self) -> list[str]:
        root = self.settings.reports_dir / "experiments"
        if not root.exists():
            return []
        return sorted(p.parent.name for p in root.glob("*/definition.json"))

    def _stored_info(self, experiment_id: str) -> ExperimentInfo | None:
        d = self._stored_dir(experiment_id)
        dpath = d / "definition.json"
        if not dpath.exists():
            return None
        try:
            defn = ExperimentDefinition.model_validate_json(dpath.read_text())
        except ValueError as e:
            log.warning("invalid stored definition %s: %s", dpath, e)
            return None
        result = self._read_result(d)
        return ExperimentInfo(
            experiment_id=experiment_id,
            status=JobStatus.COMPLETED if result else JobStatus.QUEUED,
            source="stored_real_output",
            definition=defn,
            result=result,
            outcomes_sample=_read_outcomes(d),
        )

    @staticmethod
    def _read_result(d: Path) -> ExperimentResult | None:
        p = d / "result.json"
        if not p.exists():
            return None
        try:
            return ExperimentResult.model_validate_json(p.read_text())
        except ValueError as e:
            log.warning("invalid stored result %s: %s", p, e)
            return None

    # ------------------------------------------------------------------ API operations
    def create(self, defn: ExperimentDefinition) -> ExperimentInfo:
        with self._lock:
            if (
                self.repo.get(defn.experiment_id) is not None
                or self._stored_info(defn.experiment_id) is not None
            ):
                raise Conflict(
                    f"experiment {defn.experiment_id!r} already exists; preregistered definitions are immutable"
                )
            computed = preregistration_hash(defn)
            if defn.preregistration_hash is None:
                defn = defn.model_copy(update={"preregistration_hash": computed})
            out_dir = self.settings.experiments_dir / defn.experiment_id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "definition.json").write_text(defn.model_dump_json(indent=2) + "\n")
            self.repo.create(defn)
            self.repo.set_status(defn.experiment_id, JobStatus.QUEUED.value, output_dir=str(out_dir))
        return self.get(defn.experiment_id)

    def get(self, experiment_id: str) -> ExperimentInfo:
        row = self.repo.get(experiment_id)
        if row is not None:
            defn = ExperimentDefinition.model_validate_json(row["definition_json"])
            out_dir = (
                Path(row["output_dir"])
                if row.get("output_dir")
                else self.settings.experiments_dir / experiment_id
            )
            result = self.repo.get_result(experiment_id) or self._read_result(out_dir)
            return ExperimentInfo(
                experiment_id=experiment_id,
                status=JobStatus(row["status"]),
                source="live",
                definition=defn,
                result=result,
                outcomes_sample=_read_outcomes(out_dir),
            )
        stored = self._stored_info(experiment_id)
        if stored is None:
            raise NotFound(f"experiment {experiment_id!r} not found")
        return stored

    def list_all(self) -> list[ExperimentInfo]:
        out = [self.get(r["experiment_id"]) for r in self.repo.list()]
        seen = {e.experiment_id for e in out}
        for eid in self._stored_ids():
            if eid not in seen:
                info = self._stored_info(eid)
                if info is not None:
                    out.append(info)
        return out

    def run(self, experiment_id: str) -> ExperimentInfo:
        row = self.repo.get(experiment_id)
        if row is None:
            if self._stored_info(experiment_id) is not None:
                raise Conflict(
                    "this is a committed stored experiment; re-run it from the CLI to regenerate its report, "
                    "or POST a new experiment definition"
                )
            raise NotFound(f"experiment {experiment_id!r} not found")
        if row["status"] in (JobStatus.RUNNING.value,):
            raise Conflict("experiment is already running")
        runner = self.runner_resolver()  # raises NotImplementedYet (501) when unavailable
        defn = ExperimentDefinition.model_validate_json(row["definition_json"])
        out_dir = (
            Path(row["output_dir"])
            if row.get("output_dir")
            else self.settings.experiments_dir / experiment_id
        )
        self.repo.set_status(experiment_id, JobStatus.RUNNING.value)
        self.executor.submit(self._job, runner, defn, out_dir)
        return self.get(experiment_id)

    def _job(self, runner: ExperimentRunner, defn: ExperimentDefinition, out_dir: Path) -> None:
        try:
            raw = runner(defn, out_dir)
            result: ExperimentResult | None
            if isinstance(raw, ExperimentResult):
                result = raw
            elif isinstance(raw, dict):
                result = ExperimentResult.model_validate(raw)
            else:
                result = self._read_result(out_dir)
            if result is None:
                raise RuntimeError("runner finished without producing an ExperimentResult")
            if result.preregistration_hash != defn.preregistration_hash:
                log.warning(
                    "result hash %s differs from preregistered %s",
                    result.preregistration_hash,
                    defn.preregistration_hash,
                )
            self.repo.save_result(result)
            (out_dir / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
            self.repo.set_status(defn.experiment_id, JobStatus.COMPLETED.value)
        except Exception as e:
            log.exception("experiment %s failed", defn.experiment_id)
            self.repo.set_status(defn.experiment_id, JobStatus.FAILED.value, error=f"{type(e).__name__}: {e}")

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for info in self.list_all():
            primary = info.definition.analysis.primary_metric
            effects = [
                e.model_dump(mode="json")
                for e in (info.result.effects if info.result else [])
                if e.metric == primary
            ]
            rows.append(
                {
                    "experiment_id": info.experiment_id,
                    "title": info.definition.title,
                    "status": info.status.value,
                    "source": info.source,
                    "result_status": info.result.status if info.result else None,
                    "primary_metric": primary,
                    "primary_effects": effects,
                }
            )
        return rows

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def read_json_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Read JSON files that hold a record, a list of records, or {"records"|"claims": [...]}."""
    out: list[dict[str, Any]] = []
    for p in paths:
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            log.warning("cannot read %s: %s", p, e)
            continue
        if isinstance(data, dict):
            for key in ("records", "claims", "benchmarks"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if isinstance(data, dict):
            out.append(data)
        elif isinstance(data, list):
            out.extend(d for d in data if isinstance(d, dict))
    return out
