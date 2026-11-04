"""Shared FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import HTTPException, Request

READ_ONLY_DETAIL = (
    "This deployment is read-only (HPE_READ_ONLY=1): it serves stored real outputs and does not start "
    "simulations or experiments. Run the stack locally with `make demo` for live runs."
)


def rate_limited(bucket: str) -> Callable[[Request], None]:
    """Token-bucket limit for an expensive endpoint group; 429 with Retry-After when exhausted."""

    def _dep(request: Request) -> None:
        request.app.state.ctx.limiter.dependency(bucket)(request)

    return _dep


def writes_allowed(request: Request) -> None:
    """405 on endpoints that would start work or change research state when the deployment is read-only."""
    if request.app.state.ctx.settings.read_only:
        raise HTTPException(status_code=405, detail=READ_ONLY_DETAIL, headers={"Allow": "GET"})
