import hashlib
import json

import pytest
from eval_helpers import lab_spec

from hpe.evaluation.analysis import analyze
from hpe.evaluation.runner import PreregistrationMismatch, read_jsonl_gz, run_experiment
from hpe.evaluation.spec import ExperimentSpec, load_spec, scaled_for_pipeline_validation


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_preregistration_written_before_runs_and_hash_stable(tmp_path, tiny_spec):
    h = tiny_spec.preregistration_hash()
    assert h == lab_spec().preregistration_hash()
    res = run_experiment(tiny_spec, tmp_path, make_figures=False)
    d = json.loads((tmp_path / "tiny_lab" / "definition.json").read_text())
    assert d["preregistration_hash"] == h == res.preregistration_hash


def test_changed_definition_is_refused(tmp_path, tiny_spec):
    run_experiment(tiny_spec, tmp_path, analyze_after=False)
    changed = lab_spec(seeds=[0, 1, 2, 3])
    with pytest.raises(PreregistrationMismatch):
        run_experiment(changed, tmp_path, analyze_after=False)


def test_analysis_regenerates_identical_result_from_raw_logs(tmp_path, tiny_spec):
    run_experiment(tiny_spec, tmp_path, make_figures=False)
    exp = tmp_path / "tiny_lab"
    first = (exp / "result.json").read_text()
    (exp / "result.json").unlink()
    (exp / "report.md").unlink()
    analyze(exp, make_figures=False)
    assert (exp / "result.json").read_text() == first
    assert (exp / "report.md").exists()


def test_rerun_from_seeds_reproduces_raw_outcomes(tmp_path, tiny_spec):
    run_experiment(tiny_spec, tmp_path / "a", analyze_after=False)
    run_experiment(tiny_spec, tmp_path / "b", analyze_after=False)
    for run in sorted((tmp_path / "a" / "tiny_lab" / "runs").glob("*__s*")):
        other = tmp_path / "b" / "tiny_lab" / "runs" / run.name / "outcomes.jsonl.gz"
        assert _sha(run / "outcomes.jsonl.gz") == _sha(other), run.name


def test_parallel_equals_serial(tmp_path, tiny_spec):
    run_experiment(tiny_spec, tmp_path / "s", analyze_after=False, workers=1)
    run_experiment(tiny_spec, tmp_path / "p", analyze_after=False, workers=2)
    for run in sorted((tmp_path / "s" / "tiny_lab" / "runs").glob("*__s*")):
        assert _sha(run / "outcomes.jsonl.gz") == _sha(
            tmp_path / "p" / "tiny_lab" / "runs" / run.name / "outcomes.jsonl.gz"
        )


def test_result_structure_and_verdict(tmp_path, tiny_spec):
    res = run_experiment(tiny_spec, tmp_path)
    assert res.status in {"passed", "failed", "inconclusive"}
    assert {a.arm for a in res.arms} == {"static_persona", "plain_rl"}
    assert res.effects and res.effects[0].n_units == 3
    assert "final_heldout_accuracy" in res.arms[0].metrics
    assert len(res.run_ids) == 6
    manifest = json.loads((tmp_path / "tiny_lab" / "runs" / res.run_ids[0] / "manifest.json").read_text())
    assert manifest["event_log_sha256"] == _sha(
        tmp_path / "tiny_lab" / "runs" / res.run_ids[0] / "outcomes.jsonl.gz"
    )
    assert manifest["compute"]["llm_calls"] == 0


def test_missing_policy_arm_is_skipped_and_reported(tmp_path):
    spec = lab_spec(
        exp_id="with_missing",
        arms=[{"name": "plain_rl", "policy_class": "plain_rl"}, {"name": "ghost", "policy_class": "random"}],
        contrasts=[
            {
                "name": "c",
                "primary": True,
                "treatment": "ghost",
                "control": "plain_rl",
                "metric": "final_heldout_accuracy",
            }
        ],
    )
    # 'random' is registered, so make it unavailable by monkeypatching the registry lookup
    import hpe.evaluation.runner as runner

    orig = runner.policy_available
    runner.policy_available = lambda pc, params=None: (pc != "random", "simulated missing")
    try:
        res = run_experiment(spec, tmp_path, make_figures=False)
    finally:
        runner.policy_available = orig
    assert res.status == "inconclusive" and "not computed" in res.status_reason
    assert any("ghost" in lim for lim in res.limitations)


def test_strongest_of_control_and_sensitivity_variant(tmp_path):
    spec = lab_spec(
        exp_id="strongest",
        arms=[
            {"name": "static_persona", "policy_class": "static_persona"},
            {"name": "memory_only", "policy_class": "memory_only"},
            {"name": "plain_rl", "policy_class": "plain_rl"},
        ],
        contrasts=[
            {
                "name": "rl_vs_best",
                "primary": True,
                "treatment": "plain_rl",
                "control": {"strongest_of": ["static_persona", "memory_only"]},
                "metric": "final_heldout_accuracy",
            },
        ],
        sensitivity=[{"name": "noisy", "params": {"label_noise": 0.2}}],
        seeds=[0, 1],
    )
    res = run_experiment(spec, tmp_path)
    assert "strongest of" in res.status_reason
    assert res.sensitivity and res.sensitivity[0]["variant"] == "noisy"


def test_raw_logs_contain_zero_leakage(tmp_path, tiny_spec):
    res = run_experiment(tiny_spec, tmp_path, make_figures=False)
    for rid in res.run_ids:
        rows = read_jsonl_gz(tmp_path / "tiny_lab" / "runs" / rid / "outcomes.jsonl.gz")
        leak = [r.value for r in rows if r.metric == "leakage/holdout_symbols_in_practice"]
        mut = [r.value for r in rows if r.metric == "leakage/assessment_mutated_policy"]
        assert leak == [0.0] and set(mut) == {0.0}


def test_all_experiment_yamls_load_and_hash():
    from pathlib import Path

    from hpe.evaluation.spec import simulation_spec_files

    files = simulation_spec_files(Path("experiments"))
    assert len(files) >= 8
    hashes = {load_spec(f).preregistration_hash() for f in files}
    assert len(hashes) == len(files)
    for f in files:
        s = load_spec(f)
        assert len(s.definition.seeds) >= 5
        pv = scaled_for_pipeline_validation(s)
        assert pv.protocol.pipeline_validation and pv.preregistration_hash() != s.preregistration_hash()


def test_spec_requires_single_primary():
    with pytest.raises(ValueError):
        lab_spec(
            contrasts=[
                {
                    "name": "a",
                    "primary": True,
                    "treatment": "plain_rl",
                    "control": "static_persona",
                    "metric": "m",
                },
                {
                    "name": "b",
                    "primary": True,
                    "treatment": "plain_rl",
                    "control": "static_persona",
                    "metric": "m",
                },
            ]
        )


def test_spec_rejects_unknown_arm():
    with pytest.raises(ValueError):
        lab_spec(
            contrasts=[
                {
                    "name": "a",
                    "primary": True,
                    "treatment": "nobody",
                    "control": "static_persona",
                    "metric": "m",
                }
            ]
        )


def test_spec_roundtrip(tiny_spec):
    again = ExperimentSpec.model_validate_json(tiny_spec.model_dump_json())
    assert again.preregistration_hash() == tiny_spec.preregistration_hash()
