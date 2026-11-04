"""R1 development selection on universe 0 (environment v3), condition `iid`, seeds 0-199 only.

python dev_select.py ira      # needs results_r1/u0/ira_library_beta{0.5,1.0,2.0}.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from u1.env import make_universe
from u1.ira import IRA, IRAConfig, Library
from u1.play import play

OUT = Path(os.environ.get("U1_OUT", Path(__file__).resolve().parents[1] / "results_r1"))
U = make_universe(0)


def ira() -> dict:
    res = {}
    for beta in (0.5, 1.0, 2.0):
        lib = Library.from_json(json.loads((OUT / "u0" / f"ira_library_beta{beta}.json").read_text()))
        t0 = time.process_time()
        ms = [
            play(IRA(lib, IRAConfig(beta=beta), np.random.default_rng([0, s, 606])), U, "iid", s)
            for s in range(200)
        ]
        res[str(beta)] = {
            "iid_dev_norm": round(float(np.mean([m["norm"] for m in ms])), 4),
            "probes_per_world": round(float(np.mean([m["probes"] for m in ms])), 2),
            "cpu_s_per_world": round((time.process_time() - t0) / len(ms), 3),
        }
        print(beta, res[str(beta)], flush=True)
    return res


if __name__ == "__main__":
    out = {"ira": ira}[sys.argv[1]]()
    (OUT / "dev" / f"select_{sys.argv[1]}.json").write_text(json.dumps(out, indent=1))
