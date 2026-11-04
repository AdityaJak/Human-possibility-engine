"""Configuration for Candidate A — the hybrid parametric + exemplar learner (Phase II).

Flags default to the full proposed model; experiments switch them off for ablations and matched
controls. Unknown keys raise so a typo in a YAML cannot silently run the default model.

Matched controls (docs/phase2/CANDIDATE_ARCHITECTURES.md, Candidate A):
    exemplar-only       use_parametric=False
    parametric-only     use_exemplar=False   (≈ the Phase I developmental learner)
    fixed 50/50 mixture use_learned_gate=False, fixed_gate=0.5
    no ordinal structure use_ordinal=False
    no consolidation    use_consolidation=False
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

HYBRID_FLAGS: tuple[str, ...] = (
    "use_learning",
    "use_exemplar",
    "use_parametric",
    "use_learned_gate",
    "use_learned_kernel",
    "use_ordinal",
    "use_consolidation",
    "use_forgetting",
)


@dataclass(frozen=True, slots=True)
class HybridConfig:
    # ---- ablation flags -------------------------------------------------------------------
    use_learning: bool = True  # False freezes gate, kernel and exemplar store (and the inner learner)
    use_exemplar: bool = True  # episodic exemplar pathway
    use_parametric: bool = True  # parametric pathway (the Phase I developmental learner)
    use_learned_gate: bool = True  # learned arbitration; False = fixed mixture ``fixed_gate``
    use_learned_kernel: bool = True  # learn per-feature similarity weights; False = isotropic kernel
    use_ordinal: bool = True  # graded similarity between neighbouring actions on ordinal families
    use_consolidation: bool = True  # periodic distillation of exemplars into the parametric pathway
    use_forgetting: bool = True  # prune exemplars the parametric pathway already captures

    # ---- exemplar pathway -----------------------------------------------------------------
    exemplar_space: str = "raw"  # "raw" features or "latent" (the inner shared encoder's codes)
    exemplar_capacity: int = 512
    kernel_width: float = 1.0  # h: Gaussian kernel width in the (weighted) feature space
    kernel_lr: float = 0.05  # step size for the log feature weights
    recency_decay: float = 0.995  # per task update; 1.0 = no recency weighting
    kappa: float = 0.05  # pseudo-count added to exemplar evidence (uniform prior mass)
    exemplar_beta: float = 6.0  # inverse temperature for value-based (bandit) exemplar evidence
    ordinal_spread: float = 1.0  # s: spread of the action-distance kernel on ordinal families
    ordinal_families: tuple[str, ...] = ("mcpl", "ordinal_cue")

    # ---- arbitration gate -----------------------------------------------------------------
    fixed_gate: float = 0.5  # exemplar share when the gate is not learned
    gate_lr: float = 0.1
    gate_init_bias: float = 0.0  # logit of the initial exemplar share (0 -> 0.5)
    advantage_ema: float = 0.1  # EMA rate of the exemplar-minus-parametric log-likelihood advantage

    # ---- consolidation / forgetting -------------------------------------------------------
    consolidate_every: int = 25  # task updates between consolidation steps (per family)
    n_consolidate: int = 4  # exemplars distilled per consolidation step
    prune_every: int = 100  # task updates between pruning passes (per family)
    prune_frac: float = 0.1  # share of exemplars pruned per pass (best-captured by parameters first)

    # ---- inner parametric learner (hpe.development.DevConfig overrides) -------------------
    inner: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> HybridConfig:
        params = dict(params or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(params) - known)
        if unknown:
            raise ValueError(f"unknown hybrid_exemplar params: {unknown}; known: {sorted(known)}")
        if "ordinal_families" in params:
            params["ordinal_families"] = tuple(params["ordinal_families"])
        return cls(**params)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ordinal_families"] = list(self.ordinal_families)
        return d
