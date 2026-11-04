"""FastAPI entrypoint: ``uvicorn hpe_api.main:app``.

Synthetic research prototype; not validated for predicting human futures. Every endpoint works
without API keys. See docs/INTERFACES.md §4 for the contract and docs/openapi.json for the schema.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from hpe import __version__
from hpe_api.context import AppContext
from hpe_api.errors import ServiceError
from hpe_api.routers import futures, research, runs
from hpe_api.run_service import PolicyResolver, WorldFactory
from hpe_api.settings import Settings

DESCRIPTION = """
Human Possibility Engine API.

* **Research lab** (`/runs`, `/experiments`, `/research/*`, `/benchmarks`): synthetic developmental
  society simulations and preregistered experiments. Results are simulator outputs, not estimates of
  real-world causal effects.
* **Future Self** (`/scenarios`, `/decisions/*`, `/evidence`, `/journey`, `/privacy/*`): illustrative,
  consent-based decision exploration. It never outputs individualized probabilities and never uses
  simulated-society results as evidence about a person.
"""


def create_app(
    settings: Settings | None = None,
    *,
    world_factory: WorldFactory | None = None,
    policy_resolver: PolicyResolver | None = None,
    experiment_runner_resolver: object | None = None,
) -> FastAPI:
    settings = settings or Settings()
    ctx = AppContext.build(
        settings,
        world_factory=world_factory,
        policy_resolver=policy_resolver,
        experiment_runner_resolver=experiment_runner_resolver,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        ctx.close()

    app = FastAPI(
        title="Human Possibility Engine API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.ctx = ctx
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    app.include_router(research.router)
    app.include_router(runs.router)
    app.include_router(futures.router)
    return app


_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    """Lazily build the default app so importing this module has no side effects (uvicorn reads ``app``)."""
    global _app
    if name == "app":
        if _app is None:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
            _app = create_app()
        return _app
    raise AttributeError(name)
