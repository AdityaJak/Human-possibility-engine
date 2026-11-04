"""Gate verdict rules: primary requirement, ≥ 2 families, not-worse, Holm over the frozen counted list."""

from __future__ import annotations

from hpe.population.core.gates import conservative_p, gate_a2, gate_b, gate_c, tune_stop_rule_c

COUNTED = ["restless4", "garcia", "described", "two_step"]


def c(est: float, p: float = 0.0005, n: int = 200, sd: float = 0.05) -> dict:
    return {"estimate": est, "ci_low": est - 0.01, "ci_high": est + 0.01, "p_one_sided": p, "n": n, "sd": sd}


def test_gate_c_passes_with_primary_and_two_families():
    v = gate_c(
        {"restless4": c(-0.03), "garcia": c(-0.025), "described": c(0.0), "two_step": c(0.01)},
        COUNTED,
        "restless4",
    )
    assert v.passed, v.reasons


def test_gate_c_requires_primary_two_wins_and_no_large_loss():
    assert not gate_c(
        {"garcia": c(-0.03), "described": c(-0.03), "restless4": c(0.0), "two_step": c(0.0)},
        COUNTED,
        "restless4",
    ).passed
    assert not gate_c(
        {"restless4": c(-0.03), "garcia": c(-0.01), "described": c(0.0), "two_step": c(0.0)},
        COUNTED,
        "restless4",
    ).passed
    v = gate_c(
        {"restless4": c(-0.03), "garcia": c(-0.03), "described": c(0.03), "two_step": c(0.0)},
        COUNTED,
        "restless4",
    )
    assert not v.passed and any("worse" in r for r in v.reasons)


def test_missing_counted_family_counts_against_and_keeps_holm_size():
    v = gate_c({"restless4": c(-0.03), "garcia": c(-0.03)}, COUNTED, "restless4")
    assert not v.passed and any("without results" in r for r in v.reasons)
    # Holm over 4 counted families even though only 2 are present
    assert v.families["restless4"]["p_holm"] >= 4 * 0.0005 - 1e-12


def test_conservative_p_uses_the_larger_of_bootstrap_and_t():
    weak = c(-0.005, p=0.0005, n=20, sd=0.05)  # bootstrap claims p = 0.0005; t-test disagrees
    assert conservative_p(weak) > 0.2


def test_gate_b_needs_both_b1_and_b2_with_primary():
    twin = {"restless4": c(-0.02), "garcia": c(-0.02), "described": c(0.0), "two_step": c(0.0)}
    adapter_good = {"restless4": c(-0.01), "garcia": c(-0.01), "described": c(0.0), "two_step": c(0.0)}
    adapter_bad = {"restless4": c(0.0), "garcia": c(-0.01), "described": c(-0.01), "two_step": c(0.0)}
    assert gate_b(twin, adapter_good, COUNTED, "restless4").passed
    assert not gate_b(twin, adapter_bad, COUNTED, "restless4").passed


def test_gate_a2_and_tune_stop_rule():
    assert gate_a2({"restless4": c(-0.05), "garcia": c(-0.02)}, COUNTED).passed
    assert not gate_a2({"restless4": c(-0.05)}, COUNTED).passed
    assert tune_stop_rule_c({"restless4": c(-0.021), "described": c(-0.03)}, COUNTED, "restless4")
    assert not tune_stop_rule_c({"garcia": c(-0.05), "described": c(-0.03)}, COUNTED, "restless4")
