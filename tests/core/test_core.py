import numpy as np

from hpe.core import RngStreams, derive_seed, stable_hash, stream


def test_streams_are_reproducible_and_independent():
    a = stream(1, "agent", "a0001", "mortality").random(5)
    b = stream(1, "agent", "a0001", "mortality").random(5)
    c = stream(1, "agent", "a0002", "mortality").random(5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_common_random_numbers_unaffected_by_other_streams():
    s1, s2 = RngStreams(9), RngStreams(9)
    s1.get("agent", "x", "school").random(100)  # extra draws on an unrelated stream
    assert np.array_equal(
        s1.get("agent", "y", "mortality").random(3), s2.get("agent", "y", "mortality").random(3)
    )


def test_rng_state_roundtrip():
    s = RngStreams(3)
    s.get("a").random(7)
    snap = s.state_dict()
    expected = s.get("a").random(4)
    t = RngStreams(0)
    t.load_state_dict(snap)
    assert np.array_equal(t.get("a").random(4), expected)


def test_derive_seed_stable():
    assert derive_seed(5, "x") == derive_seed(5, "x") != derive_seed(5, "y")


def test_stable_hash_insensitive_to_key_order_and_float_noise():
    assert stable_hash({"a": 1, "b": 0.1 + 0.2}) == stable_hash({"b": 0.3, "a": 1})
    assert stable_hash({"a": np.float64(1.5)}) == stable_hash({"a": 1.5})
