"""
MASCOT — Main FastAPI application.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import sentry_sdk
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routers import documents_router, eval_router, health_router, retrieval_router
from app.core.config import settings
from app.core.exceptions import MascotError, ValidationError
from app.core.logging import configure_logging, get_logger
from app.db.models import Base, engine
from app.monitoring.metrics import setup_sentry

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    setup_sentry()
    logger.info("mascot_starting", env=settings.ENV)

    # Run migrations / create tables in dev
    if settings.ENV == "development":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_tables_ensured")

    yield

    logger.info("mascot_shutting_down")
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MASCOT",
        description="Multi-stage Adaptive Structured Content Operations Toolkit — "
                    "RAG pipeline with ingestion, processing, retrieval, and evaluation.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(MascotError)
    async def mascot_error_handler(request: Request, exc: MascotError) -> JSONResponse:
        logger.warning("handled_error", code=exc.code, message=exc.message, path=request.url.path)
        status_code = 422 if isinstance(exc, ValidationError) else 500
        return JSONResponse(
            status_code=status_code,
            content={"error": exc.message, "code": exc.code},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_error", error=str(exc), path=request.url.path, exc_info=True)
        if settings.SENTRY_DSN:
            sentry_sdk.capture_exception(exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "code": "INTERNAL_ERROR"},
        )

    # Routers
    prefix = settings.API_PREFIX
    app.include_router(health_router)
    app.include_router(documents_router, prefix=prefix)
    app.include_router(retrieval_router, prefix=prefix)
    app.include_router(eval_router, prefix=prefix)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )