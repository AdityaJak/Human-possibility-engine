"""Bounded episodic replay memory.

Stores labelled task episodes the agent itself experienced (never held-out items: practice
only ever samples train-pool symbols). Capacity is ``base capacity x working_memory(age)``;
when capacity shrinks in late life, the excess items are dropped. Replay selection is a
deterministic strided cursor, so the policy needs no hidden randomness and state round-trips
exactly through ``state_dict``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MODE_CODES = {"bandit": 1, "full": 2}
CODE_MODES = {v: k for k, v in MODE_CODES.items()}
REPLAY_STRIDE = 7


class ReplayMemory:
    __slots__ = (
        "action",
        "capacity",
        "ctx",
        "fam",
        "label",
        "max_slots",
        "mode",
        "n_features",
        "ptr",
        "replay_ptr",
        "reward",
        "size",
    )

    def __init__(self, capacity: int, max_slots: int, n_features: int) -> None:
        self.capacity = max(1, int(capacity))
        self.max_slots = max_slots
        self.n_features = n_features
        self.ctx = np.zeros((self.capacity, max_slots, n_features), dtype=np.float32)
        self.fam = np.zeros(self.capacity, dtype=np.int16)
        self.action = np.zeros(self.capacity, dtype=np.int16)
        self.label = np.full(self.capacity, -1, dtype=np.int16)  # correct action for full feedback
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.mode = np.zeros(self.capacity, dtype=np.int8)
        self.size = 0
        self.ptr = 0
        self.replay_ptr = 0

    def add(
        self,
        effective_capacity: int,
        fam_idx: int,
        context: np.ndarray,
        action: int,
        mode: str,
        reward: float,
        label: int | None,
    ) -> None:
        eff = max(1, min(self.capacity, int(effective_capacity)))
        if self.size > eff:
            self.size = eff
        if self.size < eff:
            idx = self.size
            self.size += 1
        else:
            idx = self.ptr % eff
            self.ptr += 1
        n_slots = context.shape[0]
        self.ctx[idx, :n_slots] = context
        if n_slots < self.max_slots:
            self.ctx[idx, n_slots:] = 0.0
        self.fam[idx] = fam_idx
        self.action[idx] = action
        self.label[idx] = -1 if label is None else label
        self.reward[idx] = reward
        self.mode[idx] = MODE_CODES[mode]

    def shrink(self, effective_capacity: int) -> None:
        eff = max(1, min(self.capacity, int(effective_capacity)))
        if self.size > eff:
            self.size = eff

    def next_replay_indices(self, n: int) -> list[int]:
        if self.size == 0 or n <= 0:
            return []
        out = []
        for _ in range(n):
            out.append(self.replay_ptr % self.size)
            self.replay_ptr += REPLAY_STRIDE
        return out

    def state_dict(self) -> dict[str, Any]:
        s = self.size
        return {
            "capacity": self.capacity,
            "max_slots": self.max_slots,
            "n_features": self.n_features,
            "size": s,
            "ptr": self.ptr,
            "replay_ptr": self.replay_ptr,
            "ctx": self.ctx[:s].astype(np.int8).tolist()
            if _integral(self.ctx[:s])
            else self.ctx[:s].tolist(),
            "fam": self.fam[:s].tolist(),
            "action": self.action[:s].tolist(),
            "label": self.label[:s].tolist(),
            "reward": self.reward[:s].tolist(),
            "mode": self.mode[:s].tolist(),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.__init__(int(d["capacity"]), int(d["max_slots"]), int(d["n_features"]))  # type: ignore[misc]
        s = int(d["size"])
        self.size = s
        self.ptr = int(d["ptr"])
        self.replay_ptr = int(d["replay_ptr"])
        if s:
            self.ctx[:s] = np.asarray(d["ctx"], dtype=np.float32)
            self.fam[:s] = np.asarray(d["fam"], dtype=np.int16)
            self.action[:s] = np.asarray(d["action"], dtype=np.int16)
            self.label[:s] = np.asarray(d["label"], dtype=np.int16)
            self.reward[:s] = np.asarray(d["reward"], dtype=np.float32)
            self.mode[:s] = np.asarray(d["mode"], dtype=np.int8)


def _integral(a: np.ndarray) -> bool:
    return bool(a.size == 0 or np.all(a == np.round(a)))
