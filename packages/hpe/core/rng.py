"""Named, hierarchical random-number streams.

Why: twin-society experiments need *common random numbers*. If agent 17 receives tutoring
in the treatment arm, that must not shift the random draws used for agent 42's mortality or
for the weather. Every stochastic purpose therefore draws from its own stream, keyed by a
stable path such as ``("agent", "a0042", "mortality")``. Two runs that share a seed get
identical streams for identical keys, regardless of what else happened in the run.

Keys are hashed with SHA-256 (not Python's salted ``hash``), so streams are stable across
processes, platforms and Python versions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Hashable

import numpy as np


def _key_to_ints(keys: tuple[Hashable, ...]) -> list[int]:
    out: list[int] = []
    for k in keys:
        digest = hashlib.sha256(repr(k).encode("utf-8")).digest()
        # four 32-bit words per key component keeps collisions astronomically unlikely
        out.extend(int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4))
    return out


def derive_seed(seed: int, *keys: Hashable) -> int:
    """Derive a 63-bit integer seed from a root seed and a key path."""
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=tuple(_key_to_ints(keys)))
    return int(ss.generate_state(2, dtype=np.uint32).view(np.uint64)[0] >> np.uint64(1))


def stream(seed: int, *keys: Hashable) -> np.random.Generator:
    """Return an independent ``numpy.random.Generator`` for ``(seed, *keys)``."""
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=tuple(_key_to_ints(keys)))
    return np.random.Generator(np.random.PCG64(ss))


class RngStreams:
    """Lazily-created cache of named streams under one root seed.

    The cache is part of simulation state: ``state_dict`` captures every generator's
    bit-generator state so a checkpoint resumes with identical randomness.
    """

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._cache: dict[tuple[Hashable, ...], np.random.Generator] = {}

    def get(self, *keys: Hashable) -> np.random.Generator:
        g = self._cache.get(keys)
        if g is None:
            g = stream(self.seed, *keys)
            self._cache[keys] = g
        return g

    def __len__(self) -> int:
        return len(self._cache)

    def state_dict(self) -> dict:
        return {
            "seed": self.seed,
            "streams": [
                {"key": [repr(k) for k in key], "raw_key": list(key), "state": g.bit_generator.state}
                for key, g in self._cache.items()
            ],
        }

    def load_state_dict(self, d: dict) -> None:
        self.seed = int(d["seed"])
        self._cache = {}
        for entry in d["streams"]:
            key = tuple(entry["raw_key"])
            g = stream(self.seed, *key)
            g.bit_generator.state = entry["state"]
            self._cache[key] = g
