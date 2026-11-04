"""Evaluation lane (science/evaluation): baselines, statistics, preregistered experiment runner.

Modules:

- ``baselines``     static_persona, memory_only, plain_rl policies (registered on import)
- ``stats``         bootstrap / permutation / effect sizes / Holm / calibration
- ``spec``          experiment YAML format + preregistration hashing
- ``learning_lab``  controlled, matched-input learning protocol (no world needed)
- ``society``       twin/multi-arm society protocol on ``hpe.world.World`` (lazy import)
- ``runner``        arms x seeds execution, raw outcome logs, manifests
- ``analysis``      result.json / figures / report.md regenerated from raw logs only
- ``contamination`` leakage and measurement-integrity checks

CLI: ``python -m hpe.evaluation {run,analyze,contamination,list}``.
"""
