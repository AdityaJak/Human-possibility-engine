"""Shared FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Request


def rate_limited(bucket: str) -> Callable[[Request], None]:
    """Token-bucket limit for an expensive endpoint group; 429 with Retry-After when exhausted."""

    def _dep(request: Request) -> None:
        request.app.state.ctx.limiter.dependency(bucket)(request)

    return _dep
