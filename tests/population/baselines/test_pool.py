"""Pool aggregation across result directories: best comparator, adaptation contrasts, seed consistency."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from hpe.population.baselines.pool import aggregate, render


def _write(d: Path, exp: str, models: dict[str, dict], seed: int = 1) -> None:
    (d / "parts").mkdir(parents=True)
    (d / "definition.json").write_text(
        json.dumps({"experiment_id": exp, "seed": seed, "report_partition": "tune"})
    )
    summ: dict = {"families": {"restless4": {"models": {}}}}
    rows = []
    for name, spec in models.items():
        variants = {}
        for var, nlls in spec["variants"].items():
            label = name if var == "main" else f"{name}[{var}]"
            variants[var] = [{"key": f"p{i}", "nll": v} for i, v in enumerate(nlls)]
            summ["families"]["restless4"]["models"][label] = {
                "report_nll": sum(nlls) / len(nlls),
                "ci": [0, 1],
                "n": len(nlls),
            }
            rows += [
                {"family": "restless4", "model": label, "key": f"p{i}", "nll": v} for i, v in enumerate(nlls)
            ]
        part = {
            "family": "restless4",
            "model": name,
            "role": "comparator",
            "adaptive": spec.get("adaptive", False),
            "selected": {"select_nll": spec["select"], "config": {}, "lapse": 0.0, "grid_edges": []},
            "params": 10,
            "describe": {"per_person_params": 0},
            "fits": [{"fit_cpu_s": 1.0}],
            "causality_violations": spec.get("viol", []),
            "variants": variants,
        }
        (d / "parts" / f"restless4__{name}.json").write_text(json.dumps(part))
    (d / "summary.json").write_text(json.dumps(summ))
    with gzip.open(d / "outcomes.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_aggregate_picks_best_causal_comparator_and_adaptation_contrasts(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "x_a", {"pop_gru": {"select": 0.70, "variants": {"main": [0.7, 0.71, 0.69]}}})
    _write(
        b,
        "x_b",
        {
            "ft_head": {
                "select": 0.68,
                "adaptive": True,
                "variants": {
                    "main": [0.68, 0.69, 0.67],
                    "frozen": [0.7, 0.71, 0.69],
                    "mismatched": [0.72, 0.7, 0.7],
                },
            },
            "cheat": {"select": 0.10, "viol": ["lookahead"], "variants": {"main": [0.1, 0.1, 0.1]}},
        },
    )
    res = aggregate([a, b])
    fam = res["families"]["restless4"]
    assert fam["best_comparator"] == "ft_head"  # the causality violator is excluded
    kinds = {(c["first"], c["second"], c["kind"]) for c in fam["contrasts"]}
    assert ("ft_head", "ft_head[frozen]", "adapt_vs_frozen") in kinds
    assert ("ft_head", "pop_gru", "best_vs_other") in kinds
    assert "ft_head" in render(res)


def test_aggregate_refuses_mixed_seeds(tmp_path: Path) -> None:
    _write(tmp_path / "a", "x_a", {"pop_gru": {"select": 0.7, "variants": {"main": [0.7]}}}, seed=1)
    _write(tmp_path / "b", "x_b", {"pop_lstm": {"select": 0.7, "variants": {"main": [0.7]}}}, seed=2)
    with pytest.raises(ValueError, match="disagree"):
        aggregate([tmp_path / "a", tmp_path / "b"])
