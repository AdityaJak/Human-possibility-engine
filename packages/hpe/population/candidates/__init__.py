"""Design-team candidates (Phase III, lane ``research/adaptation``). Importing registers them.

Candidates (docs/phase3/POPULATION_ADAPTATION_ARCHITECTURES.md):

* ``f1_latent``  — population GRU + individual latent [z_trait, z_ctx] inferred online (CAVIA-style),
  surprise-gated context step size. Ablations: ``f1_trait`` (trait only, same total latent size),
  ``f1_context`` (context only, same size), ``f1_fixedstep`` (no surprise gating), ``f1_cavia`` (the
  standard counterpart: one latent, fixed-step gradient inner loop, meta-trained), ``f1_bolt`` (latent
  bolted onto the plain population backbone, no meta-training).
* ``f3_arbiter`` — population GRU + personal episodic memory with a learned reliability gate.
  Ablations / challengers: ``f3_fixedmix`` (constant mixture weight), ``f3_fixedshare`` (fixed-share
  Bayesian mixture), ``f3_gruepi`` (second GRU fed the episodic prediction).
* ``f2_world_policy`` — population outcome world model + individual policy with an online Laplace
  posterior under an empirical-Bayes population prior. Ablation: ``f2_delta`` (delta-rule world model);
  the population-only policy is the ``frozen`` variant.

F1/F3 share one cached population backbone per (training data, backbone config) inside a process; it is
trained with the core trainer and the same defaults as ``pop_gru``. Every candidate subclasses
``core.adaptation.AdaptiveBase``: the harness owns the adapt / frozen / donor schedules (critic G-1); the
candidates only define ``init_state`` / ``update_state`` / ``predict_with_states`` and ``twin_config``.
"""

from __future__ import annotations

from typing import Any

from hpe.population.candidates.f1_latent import F1Latent
from hpe.population.candidates.f2_world_policy import F2WorldPolicy
from hpe.population.candidates.f3_arbiter import F3Arbiter
from hpe.population.core.interface import ModelSpec, register_model


def _latent_total(cfg: dict[str, Any]) -> int:
    return int(cfg.get("d_trait", 2)) + int(cfg.get("d_ctx", 2))


def _f1(cfg: dict[str, Any], k: int, d: int) -> F1Latent:
    return F1Latent(cfg, k, d, name="f1_latent")


def _f1_trait(cfg: dict[str, Any], k: int, d: int) -> F1Latent:
    return F1Latent({**cfg, "d_trait": _latent_total(cfg), "d_ctx": 0}, k, d, name="f1_trait")


def _f1_context(cfg: dict[str, Any], k: int, d: int) -> F1Latent:
    return F1Latent({**cfg, "d_trait": 0, "d_ctx": _latent_total(cfg)}, k, d, name="f1_context")


def _f1_fixedstep(cfg: dict[str, Any], k: int, d: int) -> F1Latent:
    return F1Latent({**cfg, "beta": 0.0}, k, d, name="f1_fixedstep")


def _f1_cavia(cfg: dict[str, Any], k: int, d: int) -> F1Latent:
    return F1Latent(
        {**cfg, "d_trait": _latent_total(cfg), "d_ctx": 0, "inference": "gd", "beta": 0.0, "train_mode": "meta"},
        k,
        d,
        name="f1_cavia",
    )


def _f1_bolt(cfg: dict[str, Any], k: int, d: int) -> F1Latent:
    return F1Latent({**cfg, "meta": False}, k, d, name="f1_bolt")


def _f3(cfg: dict[str, Any], k: int, d: int) -> F3Arbiter:
    return F3Arbiter(cfg, k, d, arbiter="gate", name="f3_arbiter")


def _f3_fixedmix(cfg: dict[str, Any], k: int, d: int) -> F3Arbiter:
    return F3Arbiter(cfg, k, d, arbiter="fixed", name="f3_fixedmix")


def _f3_fixedshare(cfg: dict[str, Any], k: int, d: int) -> F3Arbiter:
    return F3Arbiter(cfg, k, d, arbiter="fixedshare", name="f3_fixedshare")


def _f3_gruepi(cfg: dict[str, Any], k: int, d: int) -> F3Arbiter:
    return F3Arbiter(cfg, k, d, arbiter="gru", name="f3_gruepi")


def _f2(cfg: dict[str, Any], k: int, d: int) -> F2WorldPolicy:
    return F2WorldPolicy(cfg, k, d, world="gru", name="f2_world_policy")


def _f2_delta(cfg: dict[str, Any], k: int, d: int) -> F2WorldPolicy:
    return F2WorldPolicy(cfg, k, d, world="delta", name="f2_delta")


_SPECS: list[tuple[str, Any, dict[str, list[Any]], str]] = [
    (
        "f1_latent",
        _f1,
        {"beta": [1.0, 3.0]},
        "F1: online individual latent, split trait/context, surprise-gated",
    ),
    ("f1_trait", _f1_trait, {}, "F1 ablation: trait latent only"),
    ("f1_context", _f1_context, {"beta": [1.0, 3.0]}, "F1 ablation: context latent only"),
    ("f1_fixedstep", _f1_fixedstep, {}, "F1 ablation: no surprise gating"),
    ("f1_cavia", _f1_cavia, {"inner_lr": [0.3, 1.0, 3.0]}, "F1 ablation: single latent, fixed-step CAVIA"),
    ("f1_bolt", _f1_bolt, {}, "F1 ablation: latent on the plain backbone, no meta-training"),
    ("f3_arbiter", _f3, {"bandwidth": [0.25, 0.5, 1.0]}, "F3: episodic memory + learned reliability gate"),
    ("f3_fixedmix", _f3_fixedmix, {"bandwidth": [0.25, 0.5, 1.0]}, "F3 challenger: constant mixture"),
    ("f3_fixedshare", _f3_fixedshare, {"share": [0.01, 0.05, 0.2]}, "F3 challenger: fixed-share mixture"),
    ("f3_gruepi", _f3_gruepi, {}, "F3 challenger: GRU fed the episodic prediction"),
    (
        "f2_world_policy",
        _f2,
        {"prior_scale": [0.5, 1.0, 2.0]},
        "F2: outcome world model + online individual policy",
    ),
    ("f2_delta", _f2_delta, {"delta_alpha": [0.1, 0.3, 0.6]}, "F2 ablation: delta-rule world model"),
]

for _name, _factory, _grid, _note in _SPECS:
    register_model(
        ModelSpec(
            _name,
            _factory,
            owner="candidates",
            kind="candidate",
            default_grid=_grid,
            adaptive=True,
            note=_note,
        )
    )

__all__ = ["F1Latent", "F2WorldPolicy", "F3Arbiter"]
