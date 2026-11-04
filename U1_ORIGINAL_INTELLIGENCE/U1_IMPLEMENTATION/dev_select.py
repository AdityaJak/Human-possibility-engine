"""Development selection on universe 0, condition `iid`, seeds 0-199 only (EXPERIMENTS_AND_BASELINES.md §3).

python dev_select.py ira      # needs results/u0/ira_library_beta{0.5,1.0,2.0}.json
python dev_select.py metarl   # needs results/u0/metarl_{h128,h256}.{pt,json}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from u1.env import make_universe
from u1.ira import IRA, IRAConfig, Library
from u1.play import play

OUT = Path(__file__).resolve().parents[1] / "results"
U = make_universe(0)
SEEDS = range(200)


def ira() -> dict:
    res = {}
    for beta in (0.5, 1.0, 2.0):
        lib = Library.from_json(json.loads((OUT / "u0" / f"ira_library_beta{beta}.json").read_text()))
        t0 = time.process_time()
        r = [
            play(IRA(lib, IRAConfig(beta=beta), np.random.default_rng([0, s, 606])), U, "iid", s)["norm"]
            for s in SEEDS
        ]
        res[str(beta)] = {
            "iid_dev_norm": round(float(np.mean(r)), 4),
            "cpu_s_per_world": (time.process_time() - t0) / len(r),
        }
        print(beta, res[str(beta)], flush=True)
    return res


def metarl() -> dict:
    import torch
    from u1.metarl import MetaRLAgent, Net

    torch.set_num_threads(1)
    res = {}
    for tag in ("h128", "h256"):
        info = json.loads((OUT / "u0" / f"metarl_{tag}.json").read_text())
        net = Net(info["config"]["hidden"])
        net.load_state_dict(torch.load(OUT / "u0" / f"metarl_{tag}.pt"))
        net.eval()
        for greedy in (True, False):
            r = [
                play(
                    MetaRLAgent(net, info["config"]["hidden"], greedy, np.random.default_rng([0, s, 609])),
                    U,
                    "iid",
                    s,
                )["norm"]
                for s in SEEDS
            ]
            res[f"{tag}_{'greedy' if greedy else 'sample'}"] = round(float(np.mean(r)), 4)
            print(tag, greedy, res[f"{tag}_{'greedy' if greedy else 'sample'}"], flush=True)
    return res


if __name__ == "__main__":
    which = sys.argv[1]
    out = {"ira": ira, "metarl": metarl}[which]()
    (OUT / "dev" / f"select_{which}.json").write_text(json.dumps(out, indent=1))
