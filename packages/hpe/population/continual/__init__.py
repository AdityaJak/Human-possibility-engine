"""Continual learning and transfer (lane ``research/continual-transfer``; thesis §12, amendment A3-2).

* ``suite`` — Gate E synthetic switching suite: generator with the reserved-seed guard, oracle and Bayes
  filter references, metrics (numpy only).
* ``learners`` — Gate E comparators: online fine-tuning, EWC and experience replay of a small value network.
* ``f4`` — F4, the meta-learned recurrent (RL²-style) learner (torch).
* ``gate_e`` — Gate E development experiment driver.
* ``transfer`` — Gate D leave-one-family-out transfer on human data (tune partition; torch).

Importing this package imports nothing heavy and registers no Phase III sequence model.
"""
