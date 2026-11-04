"""``developmental_social``: the Phase I developmental learner plus Phase II social options.

Registered on import of this module (``import hpe.development.social``). The policy is the unchanged
:class:`~hpe.development.learner.DevelopmentalLearner` with two additions, both off by default so the
Phase I defaults are preserved exactly:

* ``social_v2`` — a dict consumed by the **world** (``hpe.world.social``); it is stripped here so the
  same ``policy_params`` can carry the world flag without tripping ``DevConfig``'s unknown-key check.
* ``teaching_weight`` (float, default 1.0) — multiplies the learning-rate weight of demonstrations that
  carry instructional feedback (``Feedback.mode == "full"``), i.e. targeted teaching from a competent
  demonstrator. ``1.0`` is identical to the Phase I learner.

It also records, per policy, how many demonstrations arrived with instructional vs outcome-only
feedback (``demo_counts``) for instrumentation.
"""

from __future__ import annotations

from typing import Any, ClassVar

from hpe.contracts import registry
from hpe.contracts.runtime import AgentInit, Demonstration, LearningUpdateRecord, Policy, PolicyFactory
from hpe.development.config import DevConfig
from hpe.development.learner import DevelopmentalLearner

SOCIAL_PARAM_KEYS: tuple[str, ...] = ("social_v2", "teaching_weight")


class DevelopmentalSocialLearner(DevelopmentalLearner):
    policy_class: ClassVar[str] = "developmental_social"

    def __init__(
        self, init: AgentInit, config: DevConfig | None = None, *, teaching_weight: float = 1.0
    ) -> None:
        super().__init__(init, config)
        self.teaching_weight = float(teaching_weight)
        self.demo_counts: dict[str, int] = {"full": 0, "bandit": 0, "none": 0}

    def learn_from_demonstration(self, demo: Demonstration) -> LearningUpdateRecord | None:
        self.demo_counts[demo.feedback.mode] = self.demo_counts.get(demo.feedback.mode, 0) + 1
        if self.teaching_weight == 1.0 or demo.feedback.mode != "full":
            return super().learn_from_demonstration(demo)
        # scale the trust-derived weight of instructional demonstrations without touching the base class
        scaled = Demonstration(
            demo.observation,
            demo.action,
            demo.feedback,
            demo.demonstrator_id,
            min(1.0, max(0.0, demo.trust * self.teaching_weight)),
        )
        return super().learn_from_demonstration(scaled)

    def parameter_summary(self) -> dict[str, Any]:
        out = super().parameter_summary()
        out["teaching_weight"] = self.teaching_weight
        out["demo_counts"] = dict(self.demo_counts)
        return out

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        sd["teaching_weight"] = self.teaching_weight
        sd["demo_counts"] = dict(self.demo_counts)
        return sd

    def load_state_dict(self, state: dict[str, Any]) -> None:
        state = dict(state)
        self.teaching_weight = float(state.pop("teaching_weight", 1.0))
        self.demo_counts = {k: int(v) for k, v in dict(state.pop("demo_counts", {})).items()} or {
            "full": 0,
            "bandit": 0,
            "none": 0,
        }
        super().load_state_dict(state)


def split_params(params: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(developmental params, social params): the world flag and teaching options are removed."""
    params = dict(params or {})
    social = {k: params.pop(k) for k in SOCIAL_PARAM_KEYS if k in params}
    return params, social


def build_social_factory(params: dict[str, Any] | None = None) -> PolicyFactory:
    dev_params, social = split_params(params)
    cfg = DevConfig.from_params(dev_params)
    teaching_weight = float(social.get("teaching_weight", 1.0))

    def factory(init: AgentInit) -> Policy:
        return DevelopmentalSocialLearner(init, cfg, teaching_weight=teaching_weight)  # type: ignore[return-value]

    return factory


registry.register("developmental_social", build_social_factory)

__all__ = ["DevelopmentalSocialLearner", "build_social_factory", "split_params"]
