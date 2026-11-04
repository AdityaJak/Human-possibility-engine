"""Preregistered experiment runner.

Order of operations (each step leaves files on disk, see docs/INTERFACES.md §6):

1. Load the YAML spec, compute the preregistration hash, write ``definition.json`` (definition with
   hash) and ``protocol.json`` — BEFORE any run. Refuse to reuse a directory whose stored hash differs.
2. Expand jobs = variants x arms x seeds (deterministic run ids), write ``runs/index.json``.
3. Execute jobs (optionally in a process pool). Each job writes ``runs/<run_id>/outcomes.jsonl.gz``
   (raw OutcomeObservation records, deterministic gzip) and ``runs/<run_id>/manifest.json``.
4. ``hpe.evaluation.analysis.analyze`` regenerates ``result.json``, figures and ``report.md`` from
   those raw files only.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import os
import platform
import resource
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpe.contracts import registry
from hpe.contracts.schemas import (
    ComputeReport,
    ExperimentResult,
    JobStatus,
    OutcomeObservation,
    PolicyClass,
    RunManifest,
    SimulationConfig,
)
from hpe.core.hashing import stable_hash
from hpe.evaluation.spec import ExperimentSpec, load_spec
from hpe.tasks.symbols import SymbolWorld, TaskWorldConfig

MAIN_VARIANT = "main"


class PreregistrationMismatch(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Job:
    experiment_id: str
    variant: str
    arm: str
    seed: int
    run_id: str
    protocol_kind: str
    params: dict[str, Any]
    arm_def: dict[str, Any]
    base_config: dict[str, Any]
    exp_dir: str


def code_version() -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def hardware() -> str:
    return f"{platform.machine()} {os.cpu_count()} cpu, {platform.system()} {platform.release()}, python {platform.python_version()}"


def write_jsonl_gz(path: Path, records: list[OutcomeObservation]) -> tuple[str, int]:
    """Deterministic gzip (mtime=0) so identical outcomes give identical bytes and sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(r.model_dump_json() + "\n" for r in records).encode("utf-8")
    with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, filename="") as gz:
        gz.write(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest(), len(records)


def read_jsonl_gz(path: Path) -> list[OutcomeObservation]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [OutcomeObservation.model_validate_json(line) for line in fh if line.strip()]


def _ensure_providers() -> None:
    importlib.import_module("hpe.evaluation.baselines")


def policy_available(policy_class: str, params: dict[str, Any] | None = None) -> tuple[bool, str]:
    _ensure_providers()
    try:
        registry.get_factory(policy_class, params or {})
    except (KeyError, ImportError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def expand_jobs(spec: ExperimentSpec, exp_dir: Path) -> list[Job]:
    d = spec.definition
    variants: list[tuple[str, dict[str, Any], list[str] | None]] = [(MAIN_VARIANT, {}, None)]
    variants += [(v.name, v.params, v.arms) for v in spec.protocol.sensitivity]
    jobs: list[Job] = []
    for vname, vparams, varms in variants:
        params = {**spec.protocol.params, **(vparams if spec.protocol.kind == "learning_lab" else {})}
        if spec.protocol.kind == "society" and vparams:
            params = {**params, "world_overrides": {**params.get("world_overrides", {}), **vparams}}
        for arm in d.arms:
            if varms is not None and arm.name not in varms:
                continue
            for seed in d.seeds:
                run_id = f"{d.experiment_id}__{vname}__{arm.name}__s{seed}"
                jobs.append(
                    Job(
                        experiment_id=d.experiment_id,
                        variant=vname,
                        arm=arm.name,
                        seed=int(seed),
                        run_id=run_id,
                        protocol_kind=spec.protocol.kind,
                        params=params,
                        arm_def=arm.model_dump(mode="json"),
                        base_config=d.base_config.model_dump(mode="json"),
                        exp_dir=str(exp_dir),
                    )
                )
    return jobs


def execute_job(job: Job) -> dict[str, Any]:
    """Run one (variant, arm, seed). Top-level function so it can run in a worker process."""
    _ensure_providers()
    t0, c0 = time.perf_counter(), time.process_time()
    arm = job.arm_def
    factory = registry.get_factory(arm["policy_class"], arm.get("policy_params", {}))
    run_dir = Path(job.exp_dir) / "runs" / job.run_id
    extra: dict[str, Any] = {}
    world_manifest: RunManifest | None = None
    if job.protocol_kind == "learning_lab":
        from hpe.evaluation.learning_lab import LabContext, resolve_params, run_learning_lab

        # In the learning lab the "world" is the practice environment: arm.world_overrides set per-arm
        # environment conditions such as feedback_mode (Exp 1) or demonstrations (social ablation).
        p = resolve_params({**job.params, **arm.get("world_overrides", {})})
        ctx = LabContext(job.experiment_id, job.arm, job.seed, job.run_id, p)
        outcomes = run_learning_lab(factory, ctx)
        config = SimulationConfig(
            seed=job.seed,
            n_agents=int(p["n_agents"]),
            years=max(1e-3, float(p["age_end"] - p["age_start"])),
            policy_class=PolicyClass(arm["policy_class"]),
            policy_params=dict(arm.get("policy_params", {})),
            label=f"learning_lab:{job.variant}",
        )
        task_hash = SymbolWorld(seed=job.seed, config=TaskWorldConfig(**p["task_config"])).fingerprint()
        config_hash = stable_hash({"params": p, "arm": arm, "seed": job.seed})
        agent_years = float(p["n_agents"]) * float(p["age_end"] - p["age_start"])
        ticks = int(p["episodes_per_family"]) * len(p["families"])
    else:
        from hpe.contracts.schemas import ArmDefinition
        from hpe.evaluation.society import build_config, run_society

        ad = ArmDefinition.model_validate(arm)
        config = build_config(
            SimulationConfig.model_validate(job.base_config),
            seed=job.seed,
            policy_class=ad.policy_class.value,
            policy_params=ad.policy_params,
            interventions=ad.interventions,
            world_overrides=ad.world_overrides,
            params=job.params,
            label=f"society:{job.variant}:{job.arm}",
        )
        outcomes, world_manifest, extra = run_society(
            config,
            factory,
            run_id=job.run_id,
            experiment_id=job.experiment_id,
            arm=job.arm,
            params=job.params,
        )
        task_hash = SymbolWorld(seed=config.resolved_task_seed()).fingerprint()
        config_hash = stable_hash(config.model_dump(mode="json"))
        agent_years = float(config.n_agents) * float(config.years)
        ticks = int(round(config.years * 365 / config.dt_days))
    finite = [o for o in outcomes if o.value == o.value and abs(o.value) != float("inf")]
    sha, n = write_jsonl_gz(run_dir / "outcomes.jsonl.gz", finite)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    manifest = RunManifest(
        run_id=job.run_id,
        status=JobStatus.COMPLETED,
        code_version=code_version(),
        config=config,
        config_hash=config_hash,
        experiment_id=job.experiment_id,
        arm=job.arm,
        policy_class=PolicyClass(arm["policy_class"]),
        policy_version=world_manifest.policy_version if world_manifest else None,
        task_world_hash=task_hash,
        event_log_path="outcomes.jsonl.gz",
        event_log_sha256=sha,
        n_events=n,
        final_state_hash=extra.get("event_stream_hash"),
        compute=ComputeReport(
            wall_time_s=round(wall, 3),
            cpu_time_s=round(cpu, 3),
            peak_rss_mb=round(peak, 1),
            ticks=ticks,
            agent_years=agent_years,
            events=n,
            hardware=hardware(),
        ),
        nondeterminism=[]
        if not extra.get("manifest_error")
        else [f"world manifest unavailable: {extra['manifest_error']}"],
    )
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    if world_manifest is not None:
        (run_dir / "world_manifest.json").write_text(world_manifest.model_dump_json(indent=2) + "\n")
    return {
        "run_id": job.run_id,
        "n_outcomes": n,
        "dropped_nonfinite": len(outcomes) - n,
        "wall_time_s": wall,
    }


def prepare(spec: ExperimentSpec, out_root: Path, *, force: bool = False) -> tuple[ExperimentSpec, Path]:
    frozen = spec.frozen()
    exp_dir = out_root / frozen.definition.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    def_path = exp_dir / "definition.json"
    if def_path.exists() and not force:
        stored = json.loads(def_path.read_text()).get("preregistration_hash")
        if stored != frozen.definition.preregistration_hash:
            raise PreregistrationMismatch(
                f"{def_path} holds preregistration {stored}, but the spec now hashes to "
                f"{frozen.definition.preregistration_hash}. Refusing to mix results; use a new experiment_id."
            )
    def_path.write_text(frozen.definition.model_dump_json(indent=2) + "\n")
    (exp_dir / "protocol.json").write_text(frozen.protocol.model_dump_json(indent=2) + "\n")
    return frozen, exp_dir


def run_experiment(
    spec_or_path: ExperimentSpec | str | Path,
    out_root: str | Path = "reports/experiments",
    *,
    workers: int = 1,
    skip_missing_arms: bool = True,
    force: bool = False,
    analyze_after: bool = True,
    make_figures: bool = True,
) -> ExperimentResult | None:
    spec = spec_or_path if isinstance(spec_or_path, ExperimentSpec) else load_spec(spec_or_path)
    frozen, exp_dir = prepare(spec, Path(out_root), force=force)
    jobs = expand_jobs(frozen, exp_dir)
    skipped: dict[str, str] = {}
    if frozen.protocol.kind == "society":
        from hpe.evaluation.society import world_available

        if not world_available():
            skipped = {a.name: "hpe.world.World not available in this build" for a in frozen.definition.arms}
    for arm in frozen.definition.arms:
        if arm.name in skipped:
            continue
        ok, why = policy_available(arm.policy_class.value, arm.policy_params)
        if not ok:
            if not skip_missing_arms:
                raise RuntimeError(f"arm {arm.name}: policy unavailable ({why})")
            skipped[arm.name] = (
                f"policy class {arm.policy_class.value!r} not registered in this build ({why})"
            )
    runnable = [j for j in jobs if j.arm not in skipped]
    (exp_dir / "runs").mkdir(exist_ok=True)
    (exp_dir / "runs" / "index.json").write_text(
        json.dumps(
            {
                "preregistration_hash": frozen.definition.preregistration_hash,
                "jobs": [
                    {"run_id": j.run_id, "variant": j.variant, "arm": j.arm, "seed": j.seed} for j in runnable
                ],
                "skipped_arms": skipped,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if runnable:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(execute_job, runnable))
        else:
            for j in runnable:
                execute_job(j)
    if not analyze_after:
        return None
    from hpe.evaluation.analysis import analyze

    return analyze(exp_dir, make_figures=make_figures)
