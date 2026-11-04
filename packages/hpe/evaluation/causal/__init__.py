"""Causal-validity workstream (Phase II, lane research/causal).

Contents:
- ``mrt_sim``      synthetic micro-randomised trials with known effects → estimator validation
- ``partition``    participant-level dev/test split of HeartSteps V1 (frozen before analysis)
- ``features``     HeartSteps decision-point features and pre-listed moderators
- ``hte``          heterogeneous treatment effects with WCLS + moderators, Holm, cluster bootstrap
- ``ope``          off-policy evaluation (IPW, doubly robust) of suggestion policies
- ``__main__``     CLI writing every artifact under reports/phase2/causal/

Nothing here estimates a prospective benefit of any policy: all estimates are retrospective, under
the stated identification assumptions, from 37 participants. Gate E needs a prospective study.
"""
