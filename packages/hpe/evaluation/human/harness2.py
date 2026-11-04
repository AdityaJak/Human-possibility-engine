"""Phase II partition-aware evaluation harness (dev / tune / test with the test lock).

Protocol (docs/phase2/EXPERIMENT_REGISTRY.md):

1. Models that need fitting (static references, GRU) are fitted on the **fit** partition (``dev``).
2. Every model class gets a committed grid of <= 24 hyper-parameter configurations; the shared lapse
   grid is applied identically. Selection minimises mean per-participant NLL on the **select**
   partition (``tune``).
3. The selected configuration is scored on the **report** partition. In the dev pilot the report
   partition is ``tune`` (selection-optimistic; exploratory). A confirmatory run reports on ``test``,
   which ``partitions.check_partition_access`` refuses without a freeze record.

Per participant we record: mean NLL per free choice (lapse-mixed), top-1 accuracy, calibration
inputs (confidence = max predicted probability, hit = argmax matches the human), NLL by trial
quartile (sample efficiency), and the number of free choices. Instructed trials are learned from
but never scored.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hpe.contracts import registry
from hpe.contracts.runtime import ACTIVITY_ACTIONS, AgentInit, Policy
from hpe.core.rng import derive_seed
from hpe.evaluation.human.harness import ModelUnavailable, lapse_nll
from hpe.evaluation.human.models2 import (
    FAMILIES_META2,
    PARADIGM_N_ACTIONS,
    REFERENCE_MODELS2,
    Step2,
    encode_any,
    parameter_count,
    reference_factory2,
)
from hpe.evaluation.human.psych101 import Participant
from hpe.tasks.symbols import SymbolWorld

Factory = Callable[[AgentInit], Policy]
GridSpec = dict[str, list[float]]


@dataclass(slots=True)
class ParticipantScore:
    nll: float
    top1: float
    n: int
    conf: np.ndarray  # per free trial: max predicted probability (after lapse mixing)
    hit: np.ndarray  # per free trial: top-1 credit
    nll_by_quarter: list[float]


@dataclass(slots=True)
class ModelResult2:
    model: str
    status: str
    note: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    lapse: float | None = None
    n_grid: int = 0
    grid_edges: list[str] = field(default_factory=list)
    select_objective: float | None = None
    cpu_seconds: float = 0.0
    param_count: int | None = None
    fit_log: dict[str, Any] | None = None
    scores: dict[str, ParticipantScore] = field(default_factory=dict)  # report-partition participant -> score

    def mean_nll(self) -> float:
        return float(np.mean([s.nll for s in self.scores.values()])) if self.scores else float("nan")


def make_init(participant_key: str, seed: int) -> AgentInit:
    families = {**SymbolWorld(seed=0).families_meta(), **FAMILIES_META2}
    return AgentInit(
        agent_id=participant_key,
        birth_tick=0,
        seed=derive_seed(seed, "human", participant_key),
        traits={
            "curiosity": 0.5,
            "diligence": 0.5,
            "sociability": 0.5,
            "risk_tolerance": 0.5,
            "patience": 0.5,
        },
        base_params={"learning_rate_scale": 1.0, "memory_capacity": 256.0, "exploration": 0.1},
        families=families,
        activity_actions=list(ACTIVITY_ACTIONS),
    )


def run_participant2(
    factory: Factory, steps: list[Step2], init: AgentInit
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Teacher forcing. Returns per-step (p(human choice), top-1 credit, full prob vectors max, free mask)."""
    pol = factory(init)
    n = len(steps)
    pc, top, pmax = np.empty(n), np.empty(n), np.empty(n)
    free = np.empty(n, dtype=bool)
    for i, (obs, action, fb, forced) in enumerate(steps):
        p = np.asarray(pol.action_probabilities(obs), dtype=np.float64)
        pc[i] = p[action]
        is_max = np.isclose(p, p.max(), rtol=0, atol=1e-12)
        top[i] = is_max[action] / is_max.sum()
        pmax[i] = p.max()
        free[i] = not forced
        pol.learn(obs, action, fb)
    return pc, top, pmax, free


def score_participant(
    pc: np.ndarray, top: np.ndarray, pmax: np.ndarray, free: np.ndarray, lapse: float, k: int
) -> ParticipantScore:
    pcf, topf, pmf = pc[free], top[free], pmax[free]
    nll = lapse_nll(pcf, lapse, k)
    conf = (1.0 - lapse) * pmf + lapse / k
    q = np.array_split(np.arange(len(pcf)), 4)
    byq = [lapse_nll(pcf[idx], lapse, k) if len(idx) else float("nan") for idx in q]
    return ParticipantScore(
        nll, float(topf.mean()) if len(topf) else float("nan"), int(len(pcf)), conf, topf, byq
    )


def expected_calibration_error(
    conf: np.ndarray, hit: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """Top-label ECE: |mean confidence - accuracy| weighted by bin mass (Guo et al., 2017)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    n = max(1, len(conf))
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            bins.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "count": 0,
                    "confidence": float("nan"),
                    "accuracy": float("nan"),
                }
            )
            continue
        c, a = float(conf[m].mean()), float(hit[m].mean())
        ece += (m.sum() / n) * abs(c - a)
        bins.append({"lo": float(lo), "hi": float(hi), "count": int(m.sum()), "confidence": c, "accuracy": a})
    return float(ece), bins


def _grid(spec: GridSpec | None) -> list[dict[str, Any]]:
    if not spec:
        return [{}]
    keys = sorted(spec)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(spec[k] for k in keys))]


def _edges(params: dict[str, Any], spec: GridSpec | None) -> list[str]:
    if not spec:
        return []
    out = []
    for k, v in params.items():
        vals = spec.get(k)
        if vals and len(vals) > 1 and v in (min(vals), max(vals)):
            out.append(k)
    return out


def _factory(
    model: str,
    params: dict[str, Any],
    fit_parts: list[Participant],
    paradigm: str,
    seed: int,
    cache: dict[str, Any],
) -> Factory:
    if model in REFERENCE_MODELS2:
        return reference_factory2(model, params, fit_parts, paradigm, seed=seed, cache=cache)
    try:
        return registry.get_factory(model, params)
    except (KeyError, ImportError) as e:
        raise ModelUnavailable(
            f"policy class {model!r} not registered in this build ({type(e).__name__}: {e})"
        ) from e


def evaluate_model2(
    model: str,
    paradigm: str,
    fit_parts: list[Participant],
    select_parts: list[Participant],
    report_parts: list[Participant],
    *,
    grid: GridSpec | None,
    lapse_grid: list[float],
    igt_scales: list[float],
    seed: int,
    max_grid: int = 24,
) -> ModelResult2:
    t_cpu = time.process_time()
    k = PARADIGM_N_ACTIONS[paradigm]
    configs = _grid(grid)
    scales = igt_scales if paradigm == "igt" else [0.0]
    n_cfg = len(configs) * len(scales)
    if n_cfg > max_grid:
        return ModelResult2(
            model, "error", note=f"grid has {n_cfg} configurations > {max_grid} (equal-tuning rule)"
        )
    enc: dict[tuple[str, float], list[Step2]] = {}

    def steps_for(p: Participant, scale: float) -> list[Step2]:
        key = (p.key, scale)
        if key not in enc:
            enc[key] = encode_any(p, igt_scale=scale or 500.0)
        return enc[key]

    cache: dict[str, Any] = {}
    best: tuple[float, dict[str, Any], float] | None = None
    n_grid = 0
    try:
        for scale in scales:
            for params in configs:
                full = {**params, **({"igt_reward_scale": scale} if paradigm == "igt" else {})}
                factory = _factory(model, params, fit_parts, paradigm, seed, cache)
                runs = [
                    run_participant2(factory, steps_for(p, scale), make_init(p.key, seed))
                    for p in select_parts
                ]
                for lapse in lapse_grid:
                    n_grid += 1
                    obj = float(np.mean([lapse_nll(pc[free], lapse, k) for pc, _, _, free in runs]))
                    if best is None or obj < best[0] - 1e-12:
                        best = (obj, full, lapse)
    except ModelUnavailable as e:
        return ModelResult2(model, "unavailable", note=str(e), cpu_seconds=time.process_time() - t_cpu)
    except (ValueError, KeyError) as e:  # model not defined for this paradigm, or incompatible encoding
        return ModelResult2(
            model, "unavailable", note=f"{type(e).__name__}: {e}", cpu_seconds=time.process_time() - t_cpu
        )
    except Exception as e:
        return ModelResult2(
            model, "error", note=f"{type(e).__name__}: {e}", cpu_seconds=time.process_time() - t_cpu
        )
    assert best is not None
    obj, full, lapse = best
    scale = float(full.get("igt_reward_scale", 0.0))
    params = {kk: v for kk, v in full.items() if kk != "igt_reward_scale"}
    factory = _factory(model, params, fit_parts, paradigm, seed, cache)
    res = ModelResult2(
        model,
        "ok",
        params=full,
        lapse=lapse,
        n_grid=n_grid,
        grid_edges=_edges(params, grid),
        select_objective=obj,
        param_count=parameter_count(model, params, paradigm) if model in REFERENCE_MODELS2 else None,
    )
    gkey = next((kk for kk in cache if kk.startswith("gru:")), None)
    if gkey is not None:
        res.fit_log = cache[gkey].get("fit_log")
        res.param_count = cache[gkey].get("param_count")
    for p in report_parts:
        pc, top, pmax, free = run_participant2(factory, steps_for(p, scale), make_init(p.key, seed))
        res.scores[p.key] = score_participant(pc, top, pmax, free, lapse, k)
    res.cpu_seconds = time.process_time() - t_cpu
    return res


def evaluate_paradigm2(
    paradigm: str,
    parts: dict[str, list[Participant]],
    *,
    models: list[str],
    grids: dict[str, dict[str, GridSpec]],
    lapse_grid: list[float],
    igt_scales: list[float],
    seed: int,
    fit: str = "dev",
    select: str = "tune",
    report: str = "tune",
    log: Callable[[str], None] = print,
) -> list[ModelResult2]:
    out = []
    log(
        f"[{paradigm}] fit on {fit} ({len(parts[fit])}), select on {select} ({len(parts[select])}), report on {report} ({len(parts[report])})"
    )
    for model in models:
        r = evaluate_model2(
            model,
            paradigm,
            parts[fit],
            parts[select],
            parts[report],
            grid=grids.get(model, {}).get(paradigm),
            lapse_grid=lapse_grid,
            igt_scales=igt_scales,
            seed=seed,
        )
        if r.status == "ok":
            log(
                f"  {model:22s} ok   select={r.select_objective:.4f} report={r.mean_nll():.4f} params={r.params} lapse={r.lapse} edges={r.grid_edges} cpu={r.cpu_seconds:.1f}s"
            )
        else:
            log(f"  {model:22s} {r.status:11s} {r.note[:140]}")
        out.append(r)
    return out
