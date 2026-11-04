"""End-to-end pipelines on synthetic fixtures (no real data, no network)."""

import json
from pathlib import Path

import pytest
from hv_fixtures import write_heartsteps_csv

from hpe.evaluation.human import runner
from hpe.evaluation.human.definition import load_definition, preregistration_hash

ROOT = Path(__file__).resolve().parents[3]


def _small_exp07():
    d = load_definition(ROOT / "experiments/exp07_human_heldout.yaml")
    d.grids = {
        "ref_bias": {"mcpl": {"smoothing": [1.0]}, "igt": {"smoothing": [1.0]}},
        "ref_population_static": {"mcpl": {"smoothing": [1.0]}},
        "ref_delta_rule": {"mcpl": {"alpha": [0.1], "sigma": [1.0]}, "igt": {"alpha": [0.2], "beta": [3.0]}},
        "ref_instance_based": {
            "mcpl": {"h": [1.0], "s": [1.0], "kappa": [0.05]},
            "igt": {"decay": [0.8], "beta": [3.0]},
        },
    }
    d.lapse_grid = [0.0, 0.1]
    d.igt_reward_scale_grid = [500.0]
    d.analysis.n_bootstrap = 200
    d.split.robustness_seeds = [3]
    d.preregistration_hash = preregistration_hash(d)
    return d


def test_preregistration_hash_is_stable_and_content_sensitive():
    a = load_definition(ROOT / "experiments/exp07_human_heldout.yaml")
    b = load_definition(ROOT / "experiments/exp07_human_heldout.yaml")
    assert a.preregistration_hash == b.preregistration_hash
    b.lapse_grid = [0.0]
    assert preregistration_hash(b) != a.preregistration_hash
    c = load_definition(ROOT / "experiments/exp07b_causal_grounding.yaml")
    assert c.preregistration_hash and len(c.published) == 2


def test_exp07_end_to_end_and_regeneration_from_disk(psych_jsonl: Path, tmp_path: Path):
    defn = _small_exp07()
    out = tmp_path / "exp07"
    res = runner.run_exp07(defn, psych_jsonl, out, robustness=True, verify_hash=False, log=lambda s: None)
    for f in (
        "definition.json",
        "fits.json",
        "outcomes.jsonl.gz",
        "result.json",
        "summary.json",
        "report.md",
        "figures/heldout_nll_mcpl.png",
        "figures/heldout_nll_igt.png",
        "validation.json",
        "run_info.json",
    ):
        assert (out / f).exists(), f
    r = res["result"]
    # the developmental class is not registered in this lane's worktree -> primary contrast not evaluated
    if "developmental" not in json.loads((out / "run_info.json").read_text())["registered_policy_classes"]:
        assert r["status"] == "inconclusive" and "not available" in r["status_reason"]
    assert (
        json.loads((out / "definition.json").read_text())["preregistration_hash"] == defn.preregistration_hash
    )
    first = (out / "result.json").read_text()
    runner.finalize_exp07(out)  # regenerate purely from files on disk
    assert (out / "result.json").read_text() == first
    summ = json.loads((out / "summary.json").read_text())
    assert set(summ["splits"]) == {str(defn.split.seed), "3"}
    mcpl = summ["splits"][str(defn.split.seed)]["mcpl"]
    assert any(c.get("label") == "best_adaptive - best_static" for c in mcpl["contrasts"])
    assert "Real" in r["synthetic_disclaimer"] or "real" in r["synthetic_disclaimer"]


def test_exp07b_end_to_end_recovers_planted_effect(tmp_path: Path):
    csv_path = tmp_path / "hs.csv"
    write_heartsteps_csv(csv_path, beta_walk=0.25, beta_sed=0.25, n_users=35, n_dec=150, seed=4)
    defn = load_definition(ROOT / "experiments/exp07b_causal_grounding.yaml")
    defn.n_bootstrap = 150
    out = tmp_path / "exp07b"
    res = runner.run_exp07b(defn, csv_path, out, verify_hash=False, log=lambda s: None)
    est = json.loads((out / "estimates.json").read_text())
    any_term = est["primary"]["terms"][0]
    assert any_term["term"] == "any_suggestion" and abs(any_term["estimate"] - 0.25) < 0.12
    assert res["result"]["status"] in ("passed", "failed")
    first = (out / "result.json").read_text()
    runner.finalize_exp07b(out)
    assert (out / "result.json").read_text() == first
    assert (out / "figures/effects_vs_published.png").exists()


def test_import_boundary_no_simulation_dependencies():
    src = "\n".join(p.read_text() for p in (ROOT / "packages/hpe/evaluation/human").glob("*.py"))
    for forbidden in ("hpe.world", "hpe.development", "import hpe.futures"):
        assert forbidden not in src, forbidden


@pytest.mark.skipif(
    not (ROOT / "reports/human_validation/exp07b_causal_grounding/estimates.json").exists(),
    reason="no committed report",
)
def test_committed_reports_regenerate_identically(tmp_path: Path):
    import shutil

    for name, fn in (
        ("exp07b_causal_grounding", runner.finalize_exp07b),
        ("exp07_human_heldout", runner.finalize_exp07),
    ):
        src = ROOT / "reports/human_validation" / name
        if not (src / "result.json").exists():
            continue
        dst = tmp_path / name
        shutil.copytree(src, dst)
        fn(dst)
        assert json.loads((dst / "result.json").read_text()) == json.loads((src / "result.json").read_text())


def test_plot_tolerates_float_noise_in_constant_nll_ci(tmp_path: Path):
    from hpe.evaluation.human.report import plot_exp07

    m = 2.1972245773362196
    row = {"model": "random", "test_nll_mean": m, "test_nll_ci_low": m + 1e-16, "test_nll_ci_high": m - 1e-16}
    summary = {"splits": {"1": {"mcpl": {"models": [row], "chance_nll": m, "n_test": 3}}}}
    assert plot_exp07(summary, 1, tmp_path)[0].exists()
