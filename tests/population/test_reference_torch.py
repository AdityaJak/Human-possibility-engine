"""The reference population RNN trains, is causal and deterministic (skipped without the research extra)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hpe.population.core.conformance import check_causal  # noqa: E402
from hpe.population.core.encoding import encode, input_dim  # noqa: E402
from hpe.population.core.interface import get_model  # noqa: E402

from .test_core import _participant  # noqa: E402


def test_pop_gru_fits_is_causal_and_deterministic():
    seqs = [encode(_participant(30, i), "toy", 4) for i in range(10)]
    for i, s in enumerate(seqs):
        s.key = f"toy#{i}"
    spec = get_model("pop_gru")
    m1 = spec.factory({"hidden": 8, "lr": 0.01, "max_epochs": 3}, 4, input_dim(4))
    m2 = spec.factory({"hidden": 8, "lr": 0.01, "max_epochs": 3}, 4, input_dim(4))
    log = m1.fit(seqs, seed=5, threads=1, budget_s=60)
    m2.fit(seqs, seed=5, threads=1, budget_s=60)
    assert log["epochs"] >= 1 and np.isfinite(log["best_val_nll"])
    a, b = m1.predict_logprobs(seqs[0]), m2.predict_logprobs(seqs[0])
    assert np.array_equal(a, b)
    assert np.allclose(np.exp(a[:, :3]).sum(axis=1), 1.0, atol=1e-5) and np.isneginf(a[:, 3]).all()
    assert check_causal(m1, seqs[0]) == []
