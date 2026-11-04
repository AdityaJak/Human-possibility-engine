"""Simulation runs (spec §21/§49) and the spec §21 world/agent aliases."""

from __future__ import annotations

import zlib
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from hpe.contracts.api import (
    AgentDetail,
    AgentPage,
    EventPage,
    FramesResponse,
    RunControl,
    RunCreate,
    RunInfo,
    StepRequest,
    TimeSeriesResponse,
)
from hpe.contracts.schemas import WorldTick
from hpe_api.assembly import agent_detail, agent_summary, timeline
from hpe_api.context import AppContext, get_ctx
from hpe_api.deps import rate_limited, writes_allowed
from hpe_api.errors import BadRequest, NotFound

router = APIRouter(tags=["runs"])


def _parse_types(types: list[str] | None) -> list[str] | None:
    if not types:
        return None
    out: list[str] = []
    for t in types:
        out.extend(x.strip() for x in t.split(",") if x.strip())
    return out or None


# ---------------------------------------------------------------------------------------- runs
@router.post(
    "/runs",
    response_model=RunInfo,
    status_code=201,
    dependencies=[Depends(writes_allowed), Depends(rate_limited("runs"))],
)
def create_run(body: RunCreate, ctx: AppContext = Depends(get_ctx)) -> RunInfo:
    """Start a background job or an interactive in-memory world. No API keys involved."""
    return ctx.runs.create(body)


@router.get("/runs", response_model=list[RunInfo])
def list_runs(include_experiment_runs: bool = False, ctx: AppContext = Depends(get_ctx)) -> list[RunInfo]:
    return ctx.runs.list_runs(include_experiment_runs=include_experiment_runs)


@router.get("/runs/{run_id}", response_model=RunInfo)
def get_run(run_id: str, ctx: AppContext = Depends(get_ctx)) -> RunInfo:
    return ctx.runs.info(run_id)


@router.post("/runs/{run_id}/step", response_model=WorldTick, dependencies=[Depends(writes_allowed)])
def step_run(run_id: str, body: StepRequest | None = None, ctx: AppContext = Depends(get_ctx)) -> WorldTick:
    return ctx.runs.step(run_id, body or StepRequest())


@router.post("/runs/{run_id}/control", response_model=RunInfo, dependencies=[Depends(writes_allowed)])
def control_run(run_id: str, body: RunControl, ctx: AppContext = Depends(get_ctx)) -> RunInfo:
    return ctx.runs.control(run_id, body)


@router.get("/runs/{run_id}/agents", response_model=AgentPage)
def list_agents(
    run_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=20000),
    q: str | None = None,
    alive: bool | None = None,
    stage: str | None = None,
    ctx: AppContext = Depends(get_ctx),
) -> AgentPage:
    agents = ctx.runs.view(run_id).agents()
    if q:
        ql = q.lower()
        agents = [a for a in agents if ql in a.agent_id.lower() or ql in a.household_id.lower()]
    if alive is not None:
        agents = [a for a in agents if a.alive == alive]
    if stage:
        agents = [a for a in agents if str(getattr(a.stage, "value", a.stage)) == stage]
    agents = sorted(agents, key=lambda a: a.agent_id)
    page = agents[offset : offset + limit]
    return AgentPage(items=[agent_summary(a) for a in page], total=len(agents), offset=offset, limit=limit)


@router.get("/runs/{run_id}/agents/{agent_id}", response_model=AgentDetail)
def get_agent(run_id: str, agent_id: str, ctx: AppContext = Depends(get_ctx)) -> AgentDetail:
    return agent_detail(ctx.runs.view(run_id), ctx.events, agent_id)


@router.get("/runs/{run_id}/events", response_model=EventPage)
def list_events(
    run_id: str,
    agent_id: str | None = None,
    types: list[str] | None = Query(None, description="event types; repeat or comma-separate"),
    tick_from: int | None = None,
    tick_to: int | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=5000),
    ctx: AppContext = Depends(get_ctx),
) -> EventPage:
    ctx.runs.view(run_id)  # 404 for unknown runs; imports stored event logs on first access
    items, total = ctx.events.query(
        run_id,
        agent_id=agent_id,
        types=_parse_types(types),
        tick_from=tick_from,
        tick_to=tick_to,
        offset=offset,
        limit=limit,
    )
    return EventPage(items=items, total=total, offset=offset, limit=limit)


@router.get("/runs/{run_id}/frames", response_model=FramesResponse)
def get_frames(
    run_id: str,
    tick_from: int | None = None,
    tick_to: int | None = None,
    every: int = Query(1, ge=1),
    limit: int = Query(500, ge=1, le=5000),
    ctx: AppContext = Depends(get_ctx),
) -> FramesResponse:
    view = ctx.runs.view(run_id)
    ticks = [
        t
        for t in view.frame_ticks()
        if (tick_from is None or t >= tick_from) and (tick_to is None or t <= tick_to)
    ][::every][:limit]
    frames = [view.frames_at(t) for t in ticks]
    if not ticks and tick_from is None and tick_to is None:
        cur = view.current_frames()
        if cur is not None and cur[1]:
            ticks, frames = [cur[0]], [cur[1]]
    return FramesResponse(run_id=run_id, map=view.world_map(), ticks=ticks, frames=frames)


@router.get("/runs/{run_id}/timeseries", response_model=TimeSeriesResponse)
def get_timeseries(
    run_id: str, every: int = Query(1, ge=1), ctx: AppContext = Depends(get_ctx)
) -> TimeSeriesResponse:
    ticks = ctx.runs.view(run_id).ticks()
    if every > 1 and ticks:
        last = ticks[-1]
        ticks = ticks[::every]
        if ticks[-1].tick != last.tick:
            ticks.append(last)
    return TimeSeriesResponse(run_id=run_id, ticks=ticks)


def _gzip_stream(lines: Iterator[str]) -> Iterator[bytes]:
    comp = zlib.compressobj(6, zlib.DEFLATED, 31)  # wbits=31 -> gzip container
    for line in lines:
        chunk = comp.compress(line.encode("utf-8") + b"\n")
        if chunk:
            yield chunk
    yield comp.flush()


@router.get("/runs/{run_id}/download", response_class=StreamingResponse)
def download_run(run_id: str, ctx: AppContext = Depends(get_ctx)) -> StreamingResponse:
    """Raw replay log: first line is the RunManifest, then every ExperienceEvent in seq order (gzip JSONL)."""
    ctx.runs.info(run_id)  # 404 early
    lines = ctx.runs.download_lines(run_id)
    return StreamingResponse(
        _gzip_stream(lines),
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.jsonl.gz"'},
    )


# ---------------------------------------------------------------------------------------- aliases (spec §21)
@router.post(
    "/worlds",
    response_model=RunInfo,
    status_code=201,
    dependencies=[Depends(writes_allowed), Depends(rate_limited("runs"))],
)
def create_world(body: RunCreate, ctx: AppContext = Depends(get_ctx)) -> RunInfo:
    """Alias of POST /runs."""
    return ctx.runs.create(body)


@router.get("/worlds/{run_id}", response_model=RunInfo)
def get_world(run_id: str, ctx: AppContext = Depends(get_ctx)) -> RunInfo:
    """Alias of GET /runs/{id}."""
    return ctx.runs.info(run_id)


@router.post("/worlds/{run_id}/run", response_model=RunInfo, dependencies=[Depends(writes_allowed)])
def run_world(run_id: str, ctx: AppContext = Depends(get_ctx)) -> RunInfo:
    """Alias: resume an interactive world (background runs already advance on their own)."""
    info = ctx.runs.info(run_id)
    if info.mode == "interactive" and info.status.value == "paused":
        return ctx.runs.control(run_id, RunControl(action="resume"))
    return info


@router.get("/worlds/{run_id}/agents", response_model=AgentPage)
def list_world_agents(
    run_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=20000),
    ctx: AppContext = Depends(get_ctx),
) -> AgentPage:
    return list_agents(run_id, offset=offset, limit=limit, q=None, alive=None, stage=None, ctx=ctx)


def _resolve_agent_run(ctx: AppContext, agent_id: str, run_id: str | None) -> str:
    if run_id:
        return run_id
    candidates = ctx.runs.live_run_ids() + ctx.loader.run_ids()
    for rid in candidates:
        try:
            ctx.runs.view(rid).agent(agent_id)
            return rid
        except NotFound:
            continue
    raise BadRequest(f"agent {agent_id!r} not found in any live or stored run; pass ?run_id=")


@router.get("/agents/{agent_id}", response_model=AgentDetail)
def get_agent_alias(
    agent_id: str, run_id: str | None = None, ctx: AppContext = Depends(get_ctx)
) -> AgentDetail:
    """Alias of GET /runs/{run_id}/agents/{agent_id}; searches live then stored runs if run_id is omitted."""
    rid = _resolve_agent_run(ctx, agent_id, run_id)
    return agent_detail(ctx.runs.view(rid), ctx.events, agent_id)


@router.get("/agents/{agent_id}/timeline", response_model=EventPage)
def get_agent_timeline(
    agent_id: str, run_id: str | None = None, ctx: AppContext = Depends(get_ctx)
) -> EventPage:
    rid = _resolve_agent_run(ctx, agent_id, run_id)
    ctx.runs.view(rid).agent(agent_id)
    items = timeline(ctx.events, rid, agent_id)
    return EventPage(items=items, total=len(items), offset=0, limit=len(items))
