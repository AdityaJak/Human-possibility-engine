"""Configuration (ablation flags + hyperparameters) for the developmental learner.

Every flag defaults to the *full* proposed model. Experiments switch individual flags off to
produce ablations (spec §38: remove memory, environmental feedback, and policy learning
individually). Unknown keys raise, so a typo in an experiment YAML cannot silently run the
default model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

ABLATION_FLAGS: tuple[str, ...] = (
    "use_learning",
    "use_feedback",
    "use_memory",
    "use_shared_representation",
    "use_plasticity_schedule",
    "use_forgetting",
    "use_curriculum",
    "use_social_learning",
    "use_skepticism",
)


@dataclass(frozen=True, slots=True)
class DevConfig:
    # ---- ablation flags (True = component enabled) ---------------------------------------
    use_learning: bool = True  # False freezes every learnable parameter
    use_feedback: bool = True  # False: rewards/labels are ignored (treated as feedback mode "none")
    use_memory: bool = True  # episodic replay buffer for consolidation
    use_shared_representation: bool = True  # False: independent specialist encoder per family
    use_plasticity_schedule: bool = True  # age-dependent learning rate + working-memory capacity
    use_forgetting: bool = True  # decay toward initial parameters when a family is not practised
    use_curriculum: bool = True  # learning-progress-driven choice of what to practise
    use_social_learning: bool = True  # trust-weighted learning from demonstrations
    use_skepticism: bool = True  # down-weight demonstrations that contradict confident beliefs

    # ---- task model ----------------------------------------------------------------------
    latent_dim: int = 4  # k_hat; the task world's true latent dim is 3 (unknown to the agent)
    encoder_init_scale: float = 0.29  # ~1/sqrt(12): random projection with O(1) latent magnitude
    lr_head: float = 0.35
    lr_encoder: float = 0.06
    # consolidation: per-family step size shrinks as lr * (1 + n_f / tau) ** -power with practice
    # count n_f (lower-variance estimates once a skill is well practised); tau <= 0 disables
    lr_decay_tau: float = 100.0
    lr_decay_power: float = 0.5
    n_replay: int = 1  # memories replayed per replay step
    replay_every: int = 2  # replay step every N task updates (compute budget control)
    replay_weight: float = 0.5
    bandit_negative_floor: float = 0.05  # numerical floor for 1 - p in the negative-label update

    # ---- self-model (Bayesian mastery) ---------------------------------------------------
    mastery_decay: float = 0.98  # per feedback event; effective window ~50 episodes
    mastery_decay_per_year: float = 0.5  # toward the prior while not practising
    exploration_base: float = 0.3  # temperature = 1 + kappa * sd/sd_max; kappa = base + 2*exploration

    # ---- curriculum ----------------------------------------------------------------------
    curriculum_temperature: float = 0.1
    curriculum_weakness_bonus: float = 0.3

    # ---- social learning -----------------------------------------------------------------
    social_weight: float = 0.5
    unlabeled_imitation_weight: float = 0.5
    skepticism_floor: float = 0.05
    credibility_rate: float = 0.25  # how fast per-source credibility tracks consistency (x confidence)
    max_tracked_sources: int = 64

    # ---- lifecycle -----------------------------------------------------------------------
    forgetting_rate_per_year: float = 0.05
    forgetting_noise: float = 0.5
    encoder_forgetting_fraction: float = 0.25

    # ---- activity / social bandits ---------------------------------------------------------
    bandit_discount: float = 0.99
    bandit_prior_precision: float = 1.0
    bandit_ucb: float = 0.5
    bandit_temperature: float = 0.5

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> DevConfig:
        params = dict(params or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(params) - known)
        if unknown:
            raise ValueError(f"unknown developmental policy params: {unknown}; known: {sorted(known)}")
        return cls(**params)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flags(self) -> dict[str, bool]:
        return {k: bool(getattr(self, k)) for k in ABLATION_FLAGS}
