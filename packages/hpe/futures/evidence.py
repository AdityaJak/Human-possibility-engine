"""Evidence registry for the Future Self experience (spec §20).

Static library entries are general assumptions, the product's own method statements, and a very
small number of external studies whose citations we are highly confident exist. External studies
are ``validation_status=unvalidated`` (they are not validated for *this* use), state their limited
applicability, and are flagged ``citation pending independent verification`` until the audit lane
checks them against the primary source. Simulated outcomes are never registered here (ADR-0005).
"""

from __future__ import annotations

from hpe.contracts.schemas import EvidenceRecord, EvidenceSourceType, ValidationStatus

PENDING = "citation pending independent verification"

_LIBRARY: tuple[EvidenceRecord, ...] = (
    EvidenceRecord(
        evidence_id="ext-affective-forecasting-2005",
        domain="decision_psychology",
        source_type=EvidenceSourceType.EXTERNAL_STUDY,
        claim=(
            "People often mispredict how strongly and for how long future events will affect their feelings "
            "(affective forecasting)."
        ),
        citation=(
            "Wilson, T. D., & Gilbert, D. T. (2005). Affective forecasting: Knowing what to want. "
            "Current Directions in Psychological Science, 14(3), 131-134."
        ),
        applicability=(
            "Review of group-level research findings. Used here only to explain why imagined future feelings in "
            f"any scenario are uncertain; it says nothing about you specifically. Status: {PENDING}."
        ),
        date="2005",
        uncertainty="Group-level tendency; individual forecasting accuracy varies.",
        validation_status=ValidationStatus.UNVALIDATED,
    ),
    EvidenceRecord(
        evidence_id="ext-future-self-renderings-2011",
        domain="decision_psychology",
        source_type=EvidenceSourceType.EXTERNAL_STUDY,
        claim=(
            "In experiments, people who interacted with age-progressed renderings of themselves allocated more money "
            "to hypothetical retirement savings."
        ),
        citation=(
            "Hershfield, H. E., Goldstein, D. G., Sharpe, W. F., Fox, J., Yeykelis, L., Carstensen, L. L., & "
            "Bailenson, J. N. (2011). Increasing saving behavior through age-progressed renderings of the future self. "
            "Journal of Marketing Research, 48(SPL), S23-S37."
        ),
        applicability=(
            "Laboratory/online studies of saving allocations. Motivates the 'future self' framing of this product; "
            f"it is NOT evidence that this app improves anyone's decisions. Status: {PENDING}."
        ),
        date="2011",
        uncertainty="Effect sizes and durability outside the study settings are unknown to us.",
        validation_status=ValidationStatus.UNVALIDATED,
    ),
    EvidenceRecord(
        evidence_id="method-no-individual-forecast",
        domain="product_method",
        source_type=EvidenceSourceType.ASSUMPTION,
        claim=(
            "This prototype has no validated model of individual life outcomes, so it does not estimate how likely any "
            "path is to succeed or make you happy."
        ),
        applicability="All Future Self scenarios.",
        uncertainty="Insufficient evidence to predict individual outcomes.",
        validation_status=ValidationStatus.INSUFFICIENT_EVIDENCE,
    ),
    EvidenceRecord(
        evidence_id="method-simulation-separation",
        domain="product_method",
        source_type=EvidenceSourceType.ASSUMPTION,
        claim="Results from the synthetic agent society are research observations and are never used as evidence about you.",
        applicability="All Future Self scenarios (ADR-0005); enforced in code by an import-boundary test.",
        uncertainty="Not applicable.",
        validation_status=ValidationStatus.UNVALIDATED,
    ),
    EvidenceRecord(
        evidence_id="assume-new-role-learning-curve",
        domain="career",
        source_type=EvidenceSourceType.ASSUMPTION,
        claim="A new role usually involves a period of learning new tools, people and expectations.",
        applicability="General assumption; may not hold for roles very similar to your current one.",
        uncertainty="Length and intensity of the learning period vary widely.",
        validation_status=ValidationStatus.UNVALIDATED,
    ),
    EvidenceRecord(
        evidence_id="assume-new-role-stability",
        domain="career",
        source_type=EvidenceSourceType.ASSUMPTION,
        claim="Changing employer resets some known quantities (team, manager, track record) that currently give stability.",
        applicability="General assumption.",
        uncertainty="Depends on the employer, contract terms and labour market.",
        validation_status=ValidationStatus.UNVALIDATED,
    ),
    EvidenceRecord(
        evidence_id="assume-project-income-variable",
        domain="career",
        source_type=EvidenceSourceType.ASSUMPTION,
        claim="Income from a new venture or independent project is typically irregular at first and may be zero for a while.",
        applicability="General assumption for new ventures; does not apply to funded roles with a salary.",
        uncertainty="Timing and amount of any income are unknown.",
        validation_status=ValidationStatus.UNVALIDATED,
    ),
    EvidenceRecord(
        evidence_id="assume-project-autonomy",
        domain="career",
        source_type=EvidenceSourceType.ASSUMPTION,
        claim="Running your own project usually increases control over what you work on, alongside responsibility for outcomes.",
        applicability="General assumption.",
        uncertainty="Investors, customers or partners can reduce autonomy in practice.",
        validation_status=ValidationStatus.UNVALIDATED,
    ),
    EvidenceRecord(
        evidence_id="assume-skill-time-cost",
        domain="learning",
        source_type=EvidenceSourceType.ASSUMPTION,
        claim="Learning a substantial new skill takes sustained time each week, competing with rest, relationships and work.",
        applicability="General assumption.",
        uncertainty="Time needed depends on the skill, prior knowledge and the course format.",
        validation_status=ValidationStatus.UNVALIDATED,
    ),
    EvidenceRecord(
        evidence_id="assume-skill-payoff-unknown",
        domain="learning",
        source_type=EvidenceSourceType.ASSUMPTION,
        claim="Whether a new skill changes your income depends on factors this tool cannot observe (demand, credentials, timing).",
        applicability="General assumption.",
        uncertainty="Insufficient evidence to predict the payoff for you.",
        validation_status=ValidationStatus.INSUFFICIENT_EVIDENCE,
    ),
    EvidenceRecord(
        evidence_id="assume-status-quo-known",
        domain="career",
        source_type=EvidenceSourceType.ASSUMPTION,
        claim="Staying on your current path keeps more things known, but conditions on that path can still change.",
        applicability="General assumption.",
        uncertainty="Employers, roles and personal priorities change over time.",
        validation_status=ValidationStatus.UNVALIDATED,
    ),
    EvidenceRecord(
        evidence_id="assume-values-change",
        domain="decision_psychology",
        source_type=EvidenceSourceType.ASSUMPTION,
        claim="Your priorities can shift over the horizon you chose; the value weights you entered describe today.",
        applicability="All scenarios; edit your North Star at any time.",
        uncertainty="Unknown direction and size of change.",
        validation_status=ValidationStatus.UNVALIDATED,
    ),
)


class EvidenceRegistry:
    def __init__(self, records: tuple[EvidenceRecord, ...] = _LIBRARY) -> None:
        self._records = {r.evidence_id: r for r in records}

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return self._records.get(evidence_id)

    def require(self, evidence_id: str) -> EvidenceRecord:
        r = self._records.get(evidence_id)
        if r is None:
            raise KeyError(evidence_id)
        return r

    def all(self) -> list[EvidenceRecord]:
        return [self._records[k] for k in sorted(self._records)]

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._records


DEFAULT_REGISTRY = EvidenceRegistry()
