"""Phase III: population-trained models with individual adaptation (docs/phase3/PHASE3_RESEARCH_THESIS.md).

Sub-packages: ``core`` (orchestrator: encoding, locked data access, interface, evaluation, causality
check, reference population RNN), ``families`` (dataset adapters), ``baselines`` (evaluation team),
``candidates`` (design team), ``continual``, ``affect``. Torch is an optional dependency (ADR-0009):
``core.encoding``, ``core.data``, ``core.evaluate`` and ``core.interface`` import without it.
"""
