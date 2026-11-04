"""Future Self (Tier 5): scenarios, reflections, evidence, journey, privacy controls.

Only pseudonymous demo users are supported (ADR-0005). Nothing is stored without consent;
every write and every export/delete is recorded in the audit log.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends

from hpe.contracts.api import PrivacyDeleteResult, PrivacyExport, PrivacyRequest
from hpe.contracts.schemas import (
    ConsentRecord,
    DataNamespace,
    DecisionRequest,
    EvidenceRecord,
    Reflection,
    ScenarioResponse,
)
from hpe.futures import ConsentRequiredError
from hpe_api.context import AppContext, get_ctx
from hpe_api.deps import rate_limited
from hpe_api.errors import BadRequest, Forbidden, NotFound

router = APIRouter(tags=["future-self"])

STORAGE_PURPOSES = ("future_self_demo", "reflection_storage")


def _generate(body: DecisionRequest, ctx: AppContext) -> ScenarioResponse:
    if not body.consent_given:
        raise BadRequest(
            "consent_given must be true: the Future Self demo needs explicit consent before any processing"
        )
    try:
        resp = ctx.engine.generate(body)
    except ConsentRequiredError as e:  # pragma: no cover - guarded above
        raise BadRequest(str(e)) from e
    if not ctx.users.has_consent(body.user_id, ["future_self_demo"]):
        ctx.users.add_consent(ConsentRecord(user_id=body.user_id, purpose="future_self_demo", granted=True))
        ctx.audit.record("consent.implicit_from_request", DataNamespace.USER_PRIVATE, subject=body.user_id)
    ctx.users.save_decision(resp.decision_id, body)
    ctx.users.save_scenarios(body.user_id, resp.scenarios)
    user_records = [e for e in resp.evidence if e.evidence_id.startswith(resp.decision_id)]
    ctx.users.save_evidence(body.user_id, resp.decision_id, user_records)
    ctx.audit.record(
        "scenarios.generate",
        DataNamespace.USER_PRIVATE,
        subject=body.user_id,
        detail={
            "decision_id": resp.decision_id,
            "n_scenarios": len(resp.scenarios),
            "safety": resp.safety.category if resp.safety else "none",
        },
    )
    return resp


@router.post("/scenarios", response_model=ScenarioResponse, dependencies=[Depends(rate_limited("scenarios"))])
def create_scenarios(body: DecisionRequest, ctx: AppContext = Depends(get_ctx)) -> ScenarioResponse:
    """Illustrative scenarios for 2-4 options. No individualized probabilities; never uses simulation results."""
    return _generate(body, ctx)


@router.post(
    "/decisions/scenarios", response_model=ScenarioResponse, dependencies=[Depends(rate_limited("scenarios"))]
)
def create_decision_scenarios(body: DecisionRequest, ctx: AppContext = Depends(get_ctx)) -> ScenarioResponse:
    """Alias of POST /scenarios (spec §21)."""
    return _generate(body, ctx)


@router.post("/decisions/reflect", response_model=Reflection)
def reflect(body: Reflection, ctx: AppContext = Depends(get_ctx)) -> Reflection:
    """Record what happened after a decision. A single reflection is never treated as causal evidence."""
    if not ctx.users.has_consent(body.user_id, list(STORAGE_PURPOSES)):
        raise Forbidden("no active consent for this user; grant consent before storing reflections")
    known = {d["decision_id"] for d in ctx.users.decisions(body.user_id)}
    if body.decision_id not in known:
        raise NotFound(f"decision {body.decision_id!r} not found for this user")
    ctx.users.add_reflection(body)
    ctx.audit.record(
        "reflection.add",
        DataNamespace.USER_PRIVATE,
        subject=body.user_id,
        detail={"decision_id": body.decision_id},
    )
    return body


@router.get("/journey/{user_id}")
def journey(user_id: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "decisions": ctx.users.decisions(user_id),
        "reflections": [r.model_dump(mode="json") for r in ctx.users.reflections(user_id)],
    }


@router.get("/evidence", response_model=list[EvidenceRecord])
def list_evidence(ctx: AppContext = Depends(get_ctx)) -> list[EvidenceRecord]:
    """The static evidence/assumption library (namespace: evidence). User-specific records are not listed."""
    return ctx.evidence_repo.list()


@router.get("/evidence/{evidence_id}", response_model=EvidenceRecord)
def get_evidence(
    evidence_id: str, user_id: str | None = None, ctx: AppContext = Depends(get_ctx)
) -> EvidenceRecord:
    rec = ctx.evidence_repo.get(evidence_id)
    if rec is not None:
        return rec
    if user_id:
        user_rec = ctx.users.get_evidence(evidence_id)
        owned = {r["evidence_id"] for r in ctx.users.export(user_id).get("user_evidence", [])}
        if user_rec is not None and evidence_id in owned:
            return user_rec
    raise NotFound(f"evidence {evidence_id!r} not found")


# ------------------------------------------------------------------ privacy (spec §17 step 10)
@router.post("/privacy/consent", response_model=ConsentRecord, tags=["privacy"])
def record_consent(body: ConsentRecord, ctx: AppContext = Depends(get_ctx)) -> ConsentRecord:
    ctx.users.add_consent(body)
    ctx.audit.record(
        "consent.grant" if body.granted else "consent.revoke",
        DataNamespace.USER_PRIVATE,
        subject=body.user_id,
        detail={"purpose": body.purpose},
    )
    return body


@router.post("/privacy/export", response_model=PrivacyExport, tags=["privacy"])
def privacy_export(body: PrivacyRequest, ctx: AppContext = Depends(get_ctx)) -> PrivacyExport:
    records = ctx.users.export(body.user_id)
    records["audit_log"] = ctx.audit.for_subject(body.user_id)
    ctx.audit.record("privacy.export", DataNamespace.USER_PRIVATE, subject=body.user_id)
    return PrivacyExport(user_id=body.user_id, exported_at=datetime.now(UTC).isoformat(), records=records)


@router.post("/privacy/delete", response_model=PrivacyDeleteResult, tags=["privacy"])
def privacy_delete(body: PrivacyRequest, ctx: AppContext = Depends(get_ctx)) -> PrivacyDeleteResult:
    """Delete every user_private row for this user. The audit entry keeps only the pseudonymous id and counts."""
    counts = ctx.users.delete(body.user_id)
    ctx.audit.record("privacy.delete", DataNamespace.USER_PRIVATE, subject=body.user_id, detail=counts)
    return PrivacyDeleteResult(user_id=body.user_id, deleted=counts)
