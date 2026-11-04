"""Evidence ledger: every public claim, generated from committed artifacts (never hand-typed).

    python -m hpe.ledger            # writes reports/evidence/claims.json and docs/EVIDENCE.md

Sources (all produced by code in this repository):
    reports/experiments/<id>/result.json            preregistered synthetic experiments
    reports/human_validation/<id>/result.json       external human-data analyses
    reports/runs/<id>/{manifest,replay_verification}.json   stored society runs + replay checks
    reports/experiments/contamination_audit.json    leakage audit
    reports/scale/*.json                             measured scale benchmarks
Claims for which no artifact exists are emitted as ``not_tested`` so the gap is visible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hpe.contracts.api import BenchmarkRecord, ClaimRecord

ROOT = Path(__file__).resolve().parents[2]

_STATUS = {
    "passed": "supported",
    "failed": "not_supported",
    "inconclusive": "inconclusive",
    "error": "blocked",
}


def _fmt_effect(e: dict[str, Any]) -> str:
    return (
        f"{e['estimate']:+.4f} (95% CI [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}], "
        f"n={e['n_units']} {str(e.get('unit_of_analysis', 'units')).split(' ')[0]}s)"
    )


def _experiment_claims(base: Path, kind: str) -> list[ClaimRecord]:
    out: list[ClaimRecord] = []
    for res_path in sorted(base.glob("*/result.json")):
        if "pipeline_validation" in res_path.parts:
            continue
        r = json.loads(res_path.read_text())
        definition_path = res_path.parent / "definition.json"
        d = json.loads(definition_path.read_text()) if definition_path.exists() else {}
        primary = r["effects"][0] if r.get("effects") else None
        run_ids = list(r.get("run_ids", []))
        limitations = list(r.get("limitations", []))
        if kind == "synthetic_simulation":
            limitations.append(r.get("synthetic_disclaimer", ""))
        out.append(
            ClaimRecord(
                claim_id=r["experiment_id"],
                claim=str(d.get("hypothesis") or d.get("title") or r["experiment_id"]).strip(),
                status=_STATUS.get(r["status"], "inconclusive"),  # type: ignore[arg-type]
                evidence_kind=kind,  # type: ignore[arg-type]
                run_ids=run_ids[:12] + ([f"... +{len(run_ids) - 12} more"] if len(run_ids) > 12 else []),
                metric=primary["metric"] if primary else None,
                value=(_fmt_effect(primary) + " — " + r["status_reason"]) if primary else r["status_reason"],
                config_ref=f"{res_path.parent.relative_to(ROOT)}/definition.json "
                f"(preregistration {r.get('preregistration_hash', '?')})",
                limitations=[x for x in limitations if x],
            )
        )
    return out


def _run_claims() -> list[ClaimRecord]:
    out: list[ClaimRecord] = []
    for man_path in sorted((ROOT / "reports/runs").glob("*/manifest.json")):
        m = json.loads(man_path.read_text())
        cfg = m["config"]
        comp = m.get("compute") or {}
        rv_path = man_path.parent / "replay_verification.json"
        rv = json.loads(rv_path.read_text()) if rv_path.exists() else None
        status = "not_tested" if rv is None else ("supported" if rv.get("passed") else "not_supported")
        detail = ""
        if rv:
            ck = rv.get("from_checkpoint") or {}
            detail = (
                f"replay from seed identical={rv['from_seed']['chain_identical'] and rv['from_seed']['state_identical']}; "
                f"resume from checkpoint tick {ck.get('checkpoint_tick')} identical="
                f"{bool(ck) and ck.get('chain_identical') and ck.get('state_identical')}"
            )
        out.append(
            ClaimRecord(
                claim_id=f"run:{m['run_id']}",
                claim=(
                    f"A {cfg['n_agents']}-agent society ({cfg['policy_class']} policy) runs {cfg['years']:g} simulated "
                    "years with births, schooling, work, social ties and deaths, and replays bit-identically from its "
                    "seed and from a mid-run checkpoint."
                ),
                status=status,  # type: ignore[arg-type]
                evidence_kind="engineering",
                run_ids=[m["run_id"]],
                metric="event_stream_sha256_chain / final_state_hash",
                value=(
                    f"{m.get('n_events', 0):,} events; wall {comp.get('wall_time_s', 0):.1f} s; "
                    f"peak RSS {comp.get('peak_rss_mb') or 0:.0f} MB; LLM calls {comp.get('llm_calls', 0)}; {detail}"
                ),
                config_ref=f"{man_path.relative_to(ROOT)} (config {m.get('config_hash')}, code {m.get('code_version')})",
                limitations=[
                    "Engineering property of the simulator; says nothing about human realism.",
                    f"Hardware: {comp.get('hardware', 'unrecorded')}",
                ],
            )
        )
    return out


def _contamination_claim() -> list[ClaimRecord]:
    p = ROOT / "reports/experiments/contamination_audit.json"
    if not p.exists():
        return []
    a = json.loads(p.read_text())
    return [
        ClaimRecord(
            claim_id="contamination_audit",
            claim="No held-out symbol is ever used in practice, assessment never changes a policy, and eval sets are fixed.",
            status="supported" if a.get("clean") else "not_supported",
            evidence_kind="engineering",
            metric="leakage checks",
            value=json.dumps({k: v for k, v in a.items() if k != "details"})[:400],
            config_ref=str(p.relative_to(ROOT)),
            limitations=["Checks the synthetic task world only."],
        )
    ]


def _scale_claims() -> list[ClaimRecord]:
    out: list[ClaimRecord] = []
    records = []
    for p in sorted((ROOT / "reports/scale").glob("*.json")):
        try:
            records.append((p, BenchmarkRecord.model_validate_json(p.read_text())))
        except ValueError:
            continue
    for p, b in records:
        out.append(
            ClaimRecord(
                claim_id=f"bench:{b.benchmark_id}",
                claim=f"{b.n_agents:,} agents ({b.policy_class}) simulate {b.years:g} years at dt={b.dt_days} d.",
                status="supported" if b.status == "completed" else "not_supported",
                evidence_kind="engineering",
                metric="wall time / throughput",
                value=f"{b.wall_time_s:.1f} s; {b.agent_years_per_s:.1f} agent-years/s; peak RSS {b.peak_rss_mb or 0:.0f} MB; {b.status}",
                config_ref=str(p.relative_to(ROOT)),
                limitations=[f"Hardware: {b.hardware}", b.notes] if b.notes else [f"Hardware: {b.hardware}"],
            )
        )
    measured = {b.n_agents for _, b in records if b.status == "completed"}
    for n in (1_000, 10_000):
        if not any(m >= n for m in measured):
            out.append(
                ClaimRecord(
                    claim_id=f"bench:{n}_agents",
                    claim=f"The simulator runs {n:,} agents.",
                    status="not_tested",
                    evidence_kind="none",
                    limitations=["No completed benchmark record exists yet; do not claim this."],
                )
            )
    return out


def _phase2_claims() -> list[ClaimRecord]:
    """Phase II ledger rows from docs/phase2/EXPERIMENT_REGISTRY.md (exploratory X- and confirmatory C- tables).

    The registry is the versioned record of Phase II; each row cites the artifact directory it was computed
    from, which is carried into ``config_ref``. Exploratory rows are never 'supported' (dev/tune only).
    """
    reg = ROOT / "docs/phase2/EXPERIMENT_REGISTRY.md"
    if not reg.exists():
        return []
    out: list[ClaimRecord] = []
    for line in reg.read_text().splitlines():
        if not line.startswith("| X-") and not line.startswith("| C-"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        rid = cells[0]
        confirmatory = rid.startswith("C-")
        if confirmatory:
            question, verdict, artifact = (
                cells[2] + " — " + cells[4],
                cells[5],
                cells[6] if len(cells) > 6 else "",
            )
            low = verdict.lower()
            status = (
                "supported"
                if "passed" in low
                else "not_supported"
                if "not supported" in low or "failed" in low
                else "inconclusive"
            )
            kind = "human_dataset"
        else:
            _lane, question, partition, verdict, artifact = (
                cells[2],
                cells[3],
                cells[4],
                cells[5],
                cells[6] if len(cells) > 6 else "",
            )
            status = "inconclusive"  # exploratory, dev/tune only: never a supported claim
            kind = (
                "human_dataset"
                if ("Psych-101" in partition or "HeartSteps" in partition or "participants" in partition)
                else "synthetic_simulation"
            )
            question = f"[EXPLORATORY, {partition}] {question}"
        out.append(
            ClaimRecord(
                claim_id=f"phase2:{rid}",
                claim=question.replace("**", ""),
                status=status,  # type: ignore[arg-type]
                evidence_kind=kind,  # type: ignore[arg-type]
                metric="see registry row",
                value=verdict.replace("**", "")[:600],
                config_ref=artifact.replace("`", "")[:300],
                limitations=[
                    "Phase II exploratory result on dev/tune partitions; selection-optimistic; not a confirmatory finding"
                    if not confirmatory
                    else "Confirmatory on a frozen test partition; see registry for recorded protocol deviations",
                    "Synthetic rows: simulated world, not evidence about people"
                    if kind == "synthetic_simulation"
                    else "Real human data (licensed); not a validation of the simulator",
                ],
            )
        )
    return out


def _phase3_claims() -> list[ClaimRecord]:
    """Phase III rows from docs/phase3/EXPERIMENT_REGISTRY_V3.md.

    Exploratory ``X3-`` rows (| ID | Date | Lane | Question | Partition | Result | Artifacts |) are never
    'supported'. Confirmatory ``C3-`` rows (| ID | Date | Gate | Preregistration | Freeze record | Result |
    Verdict | Artifacts |) take their status from the Verdict cell.
    """
    reg = ROOT / "docs/phase3/EXPERIMENT_REGISTRY_V3.md"
    if not reg.exists():
        return []
    out: list[ClaimRecord] = []
    for line in reg.read_text().splitlines():
        if not (line.startswith("| X3-") or line.startswith("| C3-")):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rid = cells[0]
        if rid.startswith("C3-"):
            if len(cells) < 8:
                continue
            gate, prereg, result, verdict, artifact = cells[2], cells[3], cells[5], cells[6], cells[7]
            low = verdict.lower()
            status = (
                "supported"
                if low.startswith("pass")
                else "not_supported"
                if ("not supported" in low or low.startswith("fail"))
                else "inconclusive"
            )
            claim = f"[CONFIRMATORY, Gate {gate}] {prereg}"
            value = f"{result} — {verdict}"
            limitation = "Confirmatory on a frozen held-out partition; see the registry for deviations"
        else:
            if len(cells) < 7:
                continue
            _lane, question, partition, result, artifact = cells[2], cells[3], cells[4], cells[5], cells[6]
            status = "inconclusive"
            claim = f"[EXPLORATORY, {partition}] {question}"
            value = result
            limitation = "Phase III exploratory result (dev/tune or development seeds); selection-optimistic"
        synthetic = "synthetic" in claim.lower() or "seed" in claim.lower()
        out.append(
            ClaimRecord(
                claim_id=f"phase3:{rid}",
                claim=claim.replace("**", "")[:600],
                status=status,  # type: ignore[arg-type]
                evidence_kind="synthetic_simulation" if synthetic else "human_dataset",  # type: ignore[arg-type]
                metric="see registry row",
                value=value.replace("**", "")[:600],
                config_ref=artifact.replace("`", "")[:300],
                limitations=[
                    limitation,
                    "Synthetic rows: simulated world, not evidence about people"
                    if synthetic
                    else "Real human data (Psych-101, Apache-2.0); not evidence about any individual's future",
                ],
            )
        )
    return out


def _prohibited_claims() -> list[ClaimRecord]:
    rows = [
        ("validated_human_prediction", "The system is validated for predicting individual human futures."),
        ("human_like_intelligence", "Agents have human-like intelligence or experience emotions."),
        (
            "real_world_causal_effects_from_simulation",
            "Simulated intervention effects estimate real-world causal effects.",
        ),
        (
            "future_self_outcome_probabilities",
            "Future Self provides calibrated individual outcome probabilities.",
        ),
        ("deployed", "The product is deployed for public use."),
    ]
    return [
        ClaimRecord(
            claim_id=cid,
            claim=text,
            status="not_supported",
            evidence_kind="none",
            limitations=["No evidence exists for this claim; it is prohibited by docs/CLAIMS_POLICY.md."],
        )
        for cid, text in rows
    ]


def build_claims() -> list[ClaimRecord]:
    return (
        _experiment_claims(ROOT / "reports/experiments", "synthetic_simulation")
        + _experiment_claims(ROOT / "reports/human_validation", "human_dataset")
        + _run_claims()
        + _contamination_claim()
        + _scale_claims()
        + _phase2_claims()
        + _phase3_claims()
        + _prohibited_claims()
    )


def render_markdown(claims: list[ClaimRecord]) -> str:
    icon = {
        "supported": "SUPPORTED",
        "not_supported": "NOT SUPPORTED",
        "inconclusive": "INCONCLUSIVE",
        "not_tested": "NOT TESTED",
        "blocked": "BLOCKED",
    }
    lines = [
        "# Evidence ledger",
        "",
        "_Generated by `python -m hpe.ledger` from committed artifacts. Do not edit by hand._",
        "",
        "Every row links to the artifact it was computed from. Synthetic results come from a simulated world and "
        "are not evidence about people. Human-dataset rows use external data with verified licences (docs/DATA_SOURCES.md).",
        "",
        "| Claim | Status | Evidence | Metric / value | Source |",
        "|---|---|---|---|---|",
    ]
    for c in claims:
        val = (c.value or "").replace("|", "\\|")
        lines.append(
            f"| **{c.claim_id}** — {c.claim.replace('|', '/')} | {icon[c.status]} | {c.evidence_kind} | "
            f"{(c.metric or '—')}: {val} | {c.config_ref or '—'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    claims = build_claims()
    out = ROOT / "reports/evidence/claims.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([c.model_dump(mode="json") for c in claims], indent=2) + "\n")
    (ROOT / "docs/EVIDENCE.md").write_text(render_markdown(claims))
    counts: dict[str, int] = {}
    for c in claims:
        counts[c.status] = counts.get(c.status, 0) + 1
    print(f"wrote {out.relative_to(ROOT)} and docs/EVIDENCE.md: {len(claims)} claims {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
