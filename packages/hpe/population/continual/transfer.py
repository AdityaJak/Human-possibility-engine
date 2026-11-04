"""Gate D exploratory run (tune partition): leave-one-family-out transfer (thesis §12, amendment A3-2).

    python -m hpe.population.continual gate-d experiments/phase3/continual/<file>.yaml \
        --out reports/phase3/continual [--resume] [--targets restless4,garcia]

Design as fixed by A3-2 (choices A3-2 leaves open are marked *impl*):

* Targets: the counted families (restless4, garcia, described; two_step is not registered in core, so it is
  neither a target nor a source). Sources for target T: the **dev** partitions of every other generic family
  (bandit2, srl3, horizon, casino, wpt, garcia, described, restless4) except T; MCPL/IGT excluded.
* **History-only inputs**: the current- and previous-context feature slots (``Layout.feat``,
  ``Layout.prev_feat``) are zeroed for every model, in pretraining and on the target.
* k ∈ {0, 16, 64} target participants drawn from the target's dev partition by a seeded sampler, 5 draws per
  k > 0. Fine-tuning epochs and learning rate come from one fixed grid of ≤ 6 settings shared by all models
  and are selected (with the lapse λ) on the target's **tune** partition by mean NLL averaged over draws;
  each tune participant's score is the mean over the 5 draws at the selected setting.
* Models (all the same ``PopRNN`` GRU class and size): ``pretrained_pop_gru`` (pooled pretraining on the
  sources, then fine-tuned on k); ``specialist`` (trained only on the k, from a seeded fresh init);
  ``meta_reptile`` — the F4-style candidate: the pooled-pretrained network refined by first-order
  meta-learning across source *families* (Reptile; Nichol et al., 2018; each outer step adapts to one
  family, families sampled uniformly), then fine-tuned on k. ``hb_cog`` is pending until the evaluation
  lane's implementation is merged.
* *impl*: pretraining splits source sequences into chunks of ≤ ``chunk`` trials (each starts from a zero
  hidden state) so that very long sources (horizon, up to 2,400 trials) neither dominate padding nor CPU;
  early stopping uses ``nets.train``'s seeded 20 % split of those chunks.

The tune partition is both the selection and the report partition here: every number is exploratory and
selection-optimistic; no gate is evaluated.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field
from torch import nn

from hpe.core.rng import derive_seed, stream
from hpe.population.core import evaluate, nets
from hpe.population.core.conformance import check_causal
from hpe.population.core.data import load_family
from hpe.population.core.encoding import EncodedSeq, Layout, input_dim
from hpe.population.core.gates import conservative_p

MODELS = ("pretrained_pop_gru", "specialist", "meta_reptile")
CANDIDATE = "meta_reptile"


class GateDDefinition(BaseModel):
    experiment_id: str
    title: str
    lane: str = "research/continual-transfer"
    hypothesis: str
    seed: int
    threads: int = 1
    k_max: int = 4
    targets: list[str]
    sources: list[str]
    ks: list[int] = Field(default_factory=lambda: [0, 16, 64])
    draws: int = 5
    finetune_lr: list[float] = Field(default_factory=lambda: [0.003, 0.01])
    finetune_epochs: list[int] = Field(default_factory=lambda: [5, 20, 60])
    batch_size: int = 8
    pretrain: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
    models: list[str] = Field(default_factory=lambda: list(MODELS))
    lapse_grid: list[float] = Field(default_factory=lambda: list(evaluate.LAPSE_GRID))
    budget_s_per_model_target: float = 2700.0
    n_boot: int = 2000
    notes: list[str] = Field(default_factory=list)

    def preregistration_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:16]

    def settings(self) -> list[tuple[float, int]]:
        s = list(itertools.product(self.finetune_lr, self.finetune_epochs))
        if len(s) > 6:
            raise ValueError("A3-2 allows at most 6 shared fine-tuning settings")
        return s


def load_definition(path: Path) -> GateDDefinition:
    return GateDDefinition.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# ----------------------------------------------------------------------------------------- data
def history_only(seq: EncodedSeq) -> EncodedSeq:
    """Copy with the current- and previous-context feature slots zeroed (A3-2, critic G-4)."""
    L = Layout(seq.k_max)
    x = seq.x.copy()
    x[:, L.feat] = 0.0
    x[:, L.prev_feat] = 0.0
    return EncodedSeq(
        seq.key,
        seq.family,
        seq.k_max,
        seq.n_actions,
        x,
        seq.actions.copy(),
        seq.avail.copy(),
        seq.scored.copy(),
    )


def chunk(seqs: list[EncodedSeq], length: int) -> list[EncodedSeq]:
    out = []
    for s in seqs:
        for i, st in enumerate(range(0, s.T, length)):
            sl = slice(st, min(s.T, st + length))
            if s.scored[sl].any():
                out.append(
                    EncodedSeq(
                        f"{s.key}@{i}",
                        s.family,
                        s.k_max,
                        s.n_actions,
                        s.x[sl].copy(),
                        s.actions[sl].copy(),
                        s.avail[sl].copy(),
                        s.scored[sl].copy(),
                    )
                )
    return out


def draw_indices(n: int, k: int, *, seed: int, target: str, draw: int) -> list[int]:
    if k > n:
        raise ValueError(f"{target}: k={k} > {n} dev participants")
    rng = stream(seed, "gate_d_draw", target, k, draw)
    return sorted(int(i) for i in rng.choice(n, size=k, replace=False))


# ----------------------------------------------------------------------------------------- models
class HistoryOnlyPredictor:
    """``SequencePredictor`` wrapper (non-adaptive): history-only masking, then the GRU (for check_causal)."""

    adaptive = False

    def __init__(self, net: nets.PopRNN, name: str) -> None:
        self.net, self.name = net, name

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        raise NotImplementedError("fitted by the Gate D driver")

    @torch.no_grad()
    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        s = history_only(seq)
        b = nets.Batch([s])
        logp, _ = self.net(b.x, b.avail)
        return nets.logp_numpy(logp[0], s)

    def param_count(self) -> int:
        return nets.n_params(self.net)

    def describe(self) -> dict[str, Any]:
        return {"population_params": self.param_count(), "per_person_params": 0}


def new_net(defn: GateDDefinition, seed: int, *key: Any) -> nets.PopRNN:
    torch.manual_seed(derive_seed(seed, "gate_d_init", *key) % (2**31 - 1))
    return nets.PopRNN(input_dim(defn.k_max), int(defn.pretrain.get("hidden", 32)), defn.k_max)


def _loss(m: nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
    logp, _ = m(b.x, b.avail)
    return nets.nll_sum(logp, b)


@torch.no_grad()
def predict_many(net: nn.Module, seqs: list[EncodedSeq], batch: int = 16) -> list[np.ndarray]:
    net.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(seqs), batch):
        part = seqs[i : i + batch]
        b = nets.Batch(part)
        logp, _ = net(b.x, b.avail)
        out += [nets.logp_numpy(logp[j], s) for j, s in enumerate(part)]
    return out


def finetune(
    net: nn.Module,
    seqs: list[EncodedSeq],
    *,
    lr: float,
    checkpoints: list[int],
    batch_size: int,
    seed: int,
    key: tuple[Any, ...],
) -> dict[int, dict[str, torch.Tensor]]:
    """Adam for max(checkpoints) epochs over ``seqs`` (no early stopping: k is too small to split);
    returns the weights at each checkpoint epoch. One run serves every epoch setting of a learning rate."""
    net = copy.deepcopy(net)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = stream(seed, "gate_d_finetune_batches", *key)
    out: dict[int, dict[str, torch.Tensor]] = {}
    for ep in range(1, max(checkpoints) + 1):
        order = rng.permutation(len(seqs))
        for i in range(0, len(seqs), batch_size):
            b = nets.Batch([seqs[j] for j in order[i : i + batch_size]])
            s, k = _loss(net, b)
            opt.zero_grad()
            (s / max(k, 1)).backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if ep in checkpoints:
            out[ep] = {n: t.detach().clone() for n, t in net.state_dict().items()}
    return out


def reptile(
    init: nn.Module,
    by_family: dict[str, list[EncodedSeq]],
    *,
    seed: int,
    outer_steps: int,
    inner_steps: int,
    inner_lr: float,
    outer_lr: float,
    batch_size: int,
    budget_s: float,
) -> tuple[nn.Module, dict[str, Any]]:
    """First-order meta-learning across source families (Reptile): θ ← θ + ε_t (θ'_f − θ), where θ'_f is θ after
    ``inner_steps`` Adam steps on minibatches of one family f (families uniform); ε_t decays linearly to 0."""
    net = copy.deepcopy(init)
    fams = sorted(by_family)
    rng = stream(seed, "gate_d_reptile")
    t0 = time.process_time()
    stopped = "outer_steps"
    done = 0
    for step in range(outer_steps):
        f = fams[int(rng.integers(len(fams)))]
        seqs = by_family[f]
        fast = copy.deepcopy(net)
        fast.train()
        opt = torch.optim.Adam(fast.parameters(), lr=inner_lr)
        for _ in range(inner_steps):
            idx = rng.choice(len(seqs), size=min(batch_size, len(seqs)), replace=False)
            b = nets.Batch([seqs[int(j)] for j in sorted(idx)])
            s, k = _loss(fast, b)
            opt.zero_grad()
            (s / max(k, 1)).backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(fast.parameters(), 1.0)
            opt.step()
        eps = outer_lr * (1.0 - step / outer_steps)
        with torch.no_grad():
            for p, q in zip(net.parameters(), fast.parameters(), strict=True):
                p.add_(eps * (q - p))
        done = step + 1
        if time.process_time() - t0 > budget_s:
            stopped = "budget"
            break
    net.eval()
    return net, {"outer_steps_done": done, "stopped": stopped, "cpu_s": round(time.process_time() - t0, 1)}


# ----------------------------------------------------------------------------------------- scoring
def nll_by_lapse(
    seqs: list[EncodedSeq], lps: list[np.ndarray], grid: list[float]
) -> dict[str, dict[str, float]]:
    """{lapse: {participant key: NLL}} on the scored trials."""
    return {
        str(lam): {s.key: evaluate.score(s, lp, lam).nll for s, lp in zip(seqs, lps, strict=True)}
        for lam in grid
    }


def _mean(d: dict[str, float]) -> float:
    return float(np.mean(list(d.values())))


def run_target(defn: GateDDefinition, target: str, outdir: Path, log: Any, resume: bool) -> dict[str, Any]:
    """All models and k for one held-out target family; returns the part record (also written to disk)."""
    part_path = outdir / "parts" / f"{target}.json"
    if resume and part_path.exists():
        log(f"[{target}] checkpointed; skipping")
        return dict(json.loads(part_path.read_text()))
    eid = defn.experiment_id
    t_target = time.process_time()
    nets.setup_torch(defn.seed, defn.threads, "gate_d", target)
    sources = [f for f in defn.sources if f != target]
    src: dict[str, list[EncodedSeq]] = {}
    for f in sources:
        seqs, _ex = load_family(f, "dev", experiment_id=eid, k_max=defn.k_max)
        src[f] = [history_only(s) for s in seqs]
    dev, ex_dev = load_family(target, "dev", experiment_id=eid, k_max=defn.k_max)
    tune, ex_tune = load_family(target, "tune", experiment_id=eid, k_max=defn.k_max)
    dev = [history_only(s) for s in dev]
    tune = [history_only(s) for s in tune]
    log(
        f"[{target}] sources {{{', '.join(f'{f}: {len(v)}' for f, v in src.items())}}}; target dev {len(dev)}, tune {len(tune)}"
    )
    pc = dict(defn.pretrain)
    clen = int(pc.get("chunk", 300))
    # ---- pooled pretraining on the sources (shared by pretrained_pop_gru and meta_reptile)
    pre_path = outdir / "models" / f"{target}__pretrained.pt"
    pre_log_path = outdir / "models" / f"{target}__pretrained.json"
    net0 = new_net(defn, defn.seed, target, "pretrained")
    if resume and pre_path.exists() and pre_log_path.exists():
        net0.load_state_dict(torch.load(pre_path, weights_only=True))
        pre_log = json.loads(pre_log_path.read_text())
    else:
        pool = chunk([s for f in sorted(src) for s in src[f]], clen)
        c0 = time.process_time()
        pre_log = nets.train(
            net0,
            _loss,
            pool,
            seed=derive_seed(defn.seed, "gate_d_pretrain", target) % (2**31 - 1),
            key=f"gate_d_pretrain_{target}",
            lr=float(pc.get("lr", 0.003)),
            max_epochs=int(pc.get("max_epochs", 60)),
            patience=int(pc.get("patience", 8)),
            batch_size=int(pc.get("batch_size", 8)),
            budget_s=float(pc.get("budget_s", 1500.0)),
        )
        pre_log["cpu_s_total"] = round(time.process_time() - c0, 1)
        pre_log["n_chunks"] = len(pool)
        pre_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net0.state_dict(), pre_path)
        pre_log_path.write_text(json.dumps(pre_log, indent=1) + "\n")
    net0.eval()
    log(
        f"[{target}] pretrained: val NLL {pre_log['best_val_nll']:.4f}, epochs {pre_log['epochs']}, {pre_log['stopped']}, CPU {pre_log.get('cpu_s_total')}"
    )
    inits: dict[str, nn.Module | None] = {"pretrained_pop_gru": net0, "specialist": None}
    meta_log: dict[str, Any] = {}
    if "meta_reptile" in defn.models:
        mc = dict(defn.meta)
        by_family = {f: chunk(v, clen) for f, v in src.items()}
        net_m, meta_log = reptile(
            net0,
            by_family,
            seed=derive_seed(defn.seed, "gate_d_meta", target) % (2**31 - 1),
            outer_steps=int(mc.get("outer_steps", 400)),
            inner_steps=int(mc.get("inner_steps", 5)),
            inner_lr=float(mc.get("inner_lr", 0.003)),
            outer_lr=float(mc.get("outer_lr", 0.5)),
            batch_size=int(mc.get("batch_size", 8)),
            budget_s=float(mc.get("budget_s", 600.0)),
        )
        inits["meta_reptile"] = net_m
        log(f"[{target}] meta_reptile: {meta_log}")
    settings = defn.settings()
    lrs = sorted({lr for lr, _ in settings})
    eps = sorted({e for _, e in settings})
    record: dict[str, Any] = {
        "target": target,
        "sources": sources,
        "n_dev": len(dev),
        "n_tune": len(tune),
        "exclusions": {"dev": ex_dev, "tune": ex_tune},
        "pretrain": pre_log,
        "meta": meta_log,
        "models": {},
    }
    uniform = {s.key: float(np.mean(np.log(s.avail[s.scored].sum(axis=1)))) for s in tune}
    record["uniform_tune_nll"] = _mean(uniform)
    for model in defn.models:
        c_model = time.process_time()
        mrec: dict[str, Any] = {"by_k": {}}
        init = inits.get(model)
        for k in defn.ks:
            if k == 0:
                if init is None:
                    continue  # a specialist trained on 0 participants does not exist
                lps = predict_many(init, tune)
                by_l = nll_by_lapse(tune, lps, defn.lapse_grid)
                lam = min(defn.lapse_grid, key=lambda v: (_mean(by_l[str(v)]), v))
                mrec["by_k"]["0"] = {
                    "selected": {"setting": None, "lapse": lam},
                    "select_nll": _mean(by_l[str(lam)]),
                    "scores": by_l[str(lam)],
                    "settings_table": {},
                }
                log(f"[{target}] {model} k=0: tune NLL {_mean(by_l[str(lam)]):.4f} (lapse {lam})")
                continue
            # table[(lr, ep)][lapse][draw] = {key: nll}
            table: dict[str, dict[str, list[dict[str, float]]]] = {}
            states_d0: dict[str, dict[str, torch.Tensor]] = {}
            for d in range(defn.draws):
                idx = draw_indices(len(dev), k, seed=defn.seed, target=target, draw=d)
                kseqs = [dev[i] for i in idx]
                base = init if init is not None else new_net(defn, defn.seed, target, "specialist", k, d)
                for lr in lrs:
                    ckpts = finetune(
                        base,
                        kseqs,
                        lr=lr,
                        checkpoints=eps,
                        batch_size=defn.batch_size,
                        seed=defn.seed,
                        key=(target, model, k, d, lr),
                    )
                    for ep, st in ckpts.items():
                        if (lr, ep) not in settings:
                            continue
                        m = copy.deepcopy(base)
                        m.load_state_dict(st)
                        by_l = nll_by_lapse(tune, predict_many(m, tune), defn.lapse_grid)
                        tag = f"lr={lr},epochs={ep}"
                        for lam_s, sc in by_l.items():
                            table.setdefault(tag, {}).setdefault(lam_s, []).append(sc)
                        if d == 0:
                            states_d0[tag] = st
            best = min(
                ((tag, lam_s) for tag in table for lam_s in table[tag]),
                key=lambda tl: (float(np.mean([_mean(x) for x in table[tl[0]][tl[1]]])), tl[1], tl[0]),
            )
            tag, lam_s = best
            draws = table[tag][lam_s]
            keys = sorted(draws[0])
            scores = {key: float(np.mean([dd[key] for dd in draws])) for key in keys}
            # strict causality check of one selected fine-tuned model (draw 0)
            m = cast(nets.PopRNN, copy.deepcopy(base))
            m.load_state_dict(states_d0[tag])
            viol = check_causal(HistoryOnlyPredictor(m, model), tune[:5], seed=defn.seed, donors=False)
            mrec["by_k"][str(k)] = {
                "selected": {"setting": tag, "lapse": float(lam_s)},
                "select_nll": _mean(scores),
                "scores": scores,
                "per_draw_mean_nll": [_mean(x) for x in draws],
                "settings_table": {
                    t: {ls: float(np.mean([_mean(x) for x in v])) for ls, v in by.items()}
                    for t, by in table.items()
                },
                "causality_violations": viol,
            }
            log(
                f"[{target}] {model} k={k}: tune NLL {_mean(scores):.4f} ({tag}, lapse {lam_s}); "
                f"draws {[round(_mean(x), 4) for x in draws]}; causal={'ok' if not viol else viol}"
            )
        mrec["cpu_s"] = round(time.process_time() - c_model, 1)
        mrec["params"] = nets.n_params(net0)
        mrec["n_configs"] = len(settings) if model == "specialist" else 1 + len(settings)
        record["models"][model] = mrec
    record["cpu_s"] = round(time.process_time() - t_target, 1)
    part_path.parent.mkdir(parents=True, exist_ok=True)
    part_path.write_text(json.dumps(record) + "\n")
    return record


def run(defn: GateDDefinition, out: Path, *, resume: bool = False, targets: list[str] | None = None) -> Path:
    outdir = Path(out) / defn.experiment_id
    (outdir / "parts").mkdir(parents=True, exist_ok=True)
    (outdir / "definition.json").write_text(
        json.dumps({**defn.model_dump(), "preregistration_hash": defn.preregistration_hash()}, indent=1)
        + "\n"
    )
    logf = (outdir / "run.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t_wall, t_cpu = time.time(), time.process_time()
    for target in defn.targets:
        if targets and target not in targets:
            continue
        run_target(defn, target, outdir, log, resume)
    from hpe.population.core.runner import code_version

    info = outdir / "run_info.json"
    hist = json.loads(info.read_text())["invocations"] if info.exists() else []
    hist.append(
        {
            "code_version": code_version(),
            "hardware": f"{platform.machine()} python {platform.python_version()}",
            "threads": defn.threads,
            "targets": targets or defn.targets,
            "wall_s": round(time.time() - t_wall, 1),
            "cpu_s": round(time.process_time() - t_cpu, 1),
            "llm_calls": 0,
            "partitions_read": ["dev", "tune"],
        }
    )
    info.write_text(json.dumps({"invocations": hist}, indent=1) + "\n")
    logf.close()
    analyze(outdir)
    return outdir


# ----------------------------------------------------------------------------------------- analysis
def contrast(
    a: dict[str, float], b: dict[str, float], *, seed: int, key: tuple[str, ...], n_boot: int
) -> dict[str, Any]:
    """Paired a − b over tune participants (negative favours a), with the A3-1 conservative p."""
    sa = [evaluate.ParticipantScore(k, v, 1, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)) for k, v in a.items()]
    sb = [evaluate.ParticipantScore(k, v, 1, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)) for k, v in b.items()]
    c = evaluate.paired(sa, sb, seed=seed, key=key, n_boot=n_boot)
    return {**c, "p": conservative_p(c)}


def analyze(outdir: Path) -> dict[str, Any]:
    outdir = Path(outdir)
    raw = json.loads((outdir / "definition.json").read_text())
    defn = GateDDefinition.model_validate({k: v for k, v in raw.items() if k != "preregistration_hash"})
    parts = {p.stem: json.loads(p.read_text()) for p in sorted((outdir / "parts").glob("*.json"))}
    summary: dict[str, Any] = {
        "experiment_id": defn.experiment_id,
        "report_partition": "tune (also the selection partition; exploratory)",
        "targets": {},
        "hb_cog": "pending: the evaluation lane's hb_cog is not merged into the director branch",
    }
    rows = []
    for target, rec in parts.items():
        t: dict[str, Any] = {
            "n_tune": rec["n_tune"],
            "n_dev": rec["n_dev"],
            "uniform_tune_nll": rec["uniform_tune_nll"],
            "by_k": {},
        }
        for k in defn.ks:
            ks = str(k)
            models = {m: r["by_k"][ks] for m, r in rec["models"].items() if ks in r["by_k"]}
            kk: dict[str, Any] = {"models": {}, "contrasts": []}
            for m, r in models.items():
                arr = np.array(list(r["scores"].values()))
                est, lo, hi, _ = evaluate.bootstrap(
                    arr, seed=defn.seed, key=(target, m, ks, "mean"), n_boot=defn.n_boot
                )
                kk["models"][m] = {
                    "tune_nll": est,
                    "ci": [lo, hi],
                    "n": len(arr),
                    "selected": r["selected"],
                    "per_draw_mean_nll": r.get("per_draw_mean_nll"),
                    "causality_violations": r.get("causality_violations", []),
                    "params": rec["models"][m]["params"],
                    "cpu_s_model_target": rec["models"][m]["cpu_s"],
                }
                for key, v in r["scores"].items():
                    rows.append({"target": target, "model": m, "k": k, "key": key, "nll": v})
            pairs = [
                (CANDIDATE, "specialist"),
                (CANDIDATE, "pretrained_pop_gru"),
                ("pretrained_pop_gru", "specialist"),
            ]
            if k == 0:
                pairs = [(CANDIDATE, "pretrained_pop_gru")]
            for a, b in pairs:
                if a in models and b in models:
                    c = contrast(
                        models[a]["scores"],
                        models[b]["scores"],
                        seed=defn.seed,
                        key=(target, ks, a, b),
                        n_boot=defn.n_boot,
                    )
                    kk["contrasts"].append({"first": a, "second": b, **c})
            t["by_k"][ks] = kk
        summary["targets"][target] = t
    summary["stop_rule_tune"] = stop_rule(summary, defn)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    with gzip.open(outdir / "outcomes.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (outdir / "report.md").write_text(render(defn, summary))
    return summary


def stop_rule(
    summary: dict[str, Any], defn: GateDDefinition, k: str = "16", margin: float = 0.02
) -> dict[str, Any]:
    """A3-2 tune stop rule for the candidate at k = 16: beat the specialist, the pretrained pop_gru and hb_cog
    by ≥ margin each (Holm over the counted targets on the conservative p) in ≥ 2 counted targets."""
    out: dict[str, Any] = {"candidate": CANDIDATE, "k": int(k), "margin": margin, "comparators": {}}
    counted = list(defn.targets)
    passes_by_target = {t: True for t in counted}
    for comp in ("specialist", "pretrained_pop_gru"):
        cons = {}
        for t in counted:
            cs = summary["targets"].get(t, {}).get("by_k", {}).get(k, {}).get("contrasts", [])
            c = next((x for x in cs if x["first"] == CANDIDATE and x["second"] == comp), None)
            cons[t] = c
        adj = evaluate.holm({t: (float(c["p"]) if c else 1.0) for t, c in cons.items()})
        per = {}
        for t, c in cons.items():
            ok = bool(c is not None and c["estimate"] <= -margin and adj[t] < 0.05)
            per[t] = {"estimate": c["estimate"] if c else None, "p_holm": adj[t], "meets": ok}
            passes_by_target[t] = passes_by_target[t] and ok
        out["comparators"][comp] = per
    out["comparators"]["hb_cog"] = {"status": "pending (not merged); the rule cannot be met without it"}
    n_ok = sum(passes_by_target.values())
    out["targets_meeting_specialist_and_pop_gru"] = n_ok
    out["meets_tune_stop_rule"] = False  # hb_cog comparison missing: cannot be met
    out["would_meet_without_hb_cog"] = n_ok >= 2
    return out


def _f(x: Any, nd: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(defn: GateDDefinition, s: dict[str, Any]) -> str:
    out = [
        f"# {defn.experiment_id} — {defn.title}",
        "",
        "> **EXPLORATORY (tune partition; selection-optimistic).** Real human data (Psych-101, Apache-2.0) via "
        "`load_family` (dev/tune only). History-only inputs. No gate is evaluated here.",
        "",
        f"* Lane {defn.lane}; definition hash `{defn.preregistration_hash()}`; seed {defn.seed}; threads {defn.threads}",
        f"* Hypothesis: {defn.hypothesis}",
        f"* Sources: {', '.join(defn.sources)} (dev; the target excluded). Fine-tuning settings (shared): "
        f"lr {defn.finetune_lr} × epochs {defn.finetune_epochs}; {defn.draws} draws per k",
        f"* hb_cog: {s['hb_cog']}",
        "",
    ]
    for target, t in s["targets"].items():
        out += [
            f"## {target} (tune n = {t['n_tune']}, dev n = {t['n_dev']}; uniform-choice NLL {_f(t['uniform_tune_nll'])})",
            "",
            "| k | model | tune NLL [95% CI] | selected setting | lapse | per-draw mean NLL | params | CPU s (model, target) | causal |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for k, kk in t["by_k"].items():
            for m, r in sorted(kk["models"].items(), key=lambda kv: kv[1]["tune_nll"]):
                pd = ", ".join(_f(v, 3) for v in (r["per_draw_mean_nll"] or [])) or "—"
                out.append(
                    f"| {k} | {m} | {_f(r['tune_nll'])} [{_f(r['ci'][0])}, {_f(r['ci'][1])}] | "
                    f"{r['selected']['setting'] or 'zero-shot'} | {r['selected']['lapse']} | {pd} | {r['params']} | "
                    f"{r['cpu_s_model_target']} | {'ok' if not r['causality_violations'] else 'VIOLATION'} |"
                )
        out += [
            "",
            "| k | contrast (negative favours first) | estimate | 95% CI | p (max boot, t) | sd | n |",
            "|---|---|---|---|---|---|---|",
        ]
        for k, kk in t["by_k"].items():
            for c in kk["contrasts"]:
                out.append(
                    f"| {k} | {c['first']} − {c['second']} | {c['estimate']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                    f"{c['p']:.4f} | {_f(c['sd'])} | {c['n']} |"
                )
        out.append("")
    st = s["stop_rule_tune"]
    out += [
        f"## Tune stop rule (A3-2): candidate `{st['candidate']}` at k = {st['k']}, margin {st['margin']}",
        "",
        "| comparator | target | estimate | Holm p | meets |",
        "|---|---|---|---|---|",
    ]
    for comp, per in st["comparators"].items():
        if "status" in per:
            out.append(f"| {comp} | — | — | — | {per['status']} |")
            continue
        for t, v in per.items():
            out.append(f"| {comp} | {t} | {_f(v['estimate'])} | {_f(v['p_holm'])} | {v['meets']} |")
    out += [
        "",
        f"Targets meeting both available comparisons: {st['targets_meeting_specialist_and_pop_gru']}; "
        f"**tune stop rule met: {st['meets_tune_stop_rule']}** (hb_cog pending). Exploratory; no gate is evaluated.",
        "",
    ]
    return "\n".join(out) + "\n"
