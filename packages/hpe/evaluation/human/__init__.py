"""Human-data validation (Gate 4) and causal grounding (Experiment 7).

Namespace: ``external_dataset``. Nothing here imports or consumes simulation outputs, and no
raw external data is ever written into the repository (only derived metrics).

Modules:
    data       dataset sources, SHA-256 verification, download/filter
    psych101   Psych-101 MCPL and IGT transcript parsers + validation
    heartsteps HeartSteps V1 micro-randomised trial loader + validation
    models     reference cognitive models implementing the Policy protocol
    harness    teacher-forced held-out likelihood evaluation + hyperparameter fitting
    causal     centred & weighted least squares (WCLS) causal excursion effects
    definition preregistration schema for human-data experiments
    report     result/figure/report generation from derived outcome logs
"""
