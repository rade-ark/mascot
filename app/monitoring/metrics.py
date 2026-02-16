"""
Monitoring layer — Prometheus metrics, health checks, Sentry integration.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import sentry_sdk
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Prometheus Metrics ─────────────────────────────────────────────────────────

documents_ingested = Counter(
    "mascot_documents_ingested_total",
    "Total documents ingested",
    ["status", "mime_type"],
)

chunks_created = Counter(
    "mascot_chunks_created_total",
    "Total chunks created",
)

embeddings_generated = Counter(
    "mascot_embeddings_generated_total",
    "Total embeddings generated",
)

queries_total = Counter(
    "mascot_queries_total",
    "Total queries processed",
    ["status"],
)

query_latency = Histogram(
    "mascot_query_latency_seconds",
    "Query latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

retrieval_score = Histogram(
    "mascot_retrieval_score",
    "Distribution of retrieval scores",
    buckets=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
)

eval_faithfulness = Gauge("mascot_eval_faithfulness", "Latest eval faithfulness score")
eval_relevance = Gauge("mascot_eval_relevance", "Latest eval relevance score")
eval_overall = Gauge("mascot_eval_overall", "Latest eval overall score")

active_workers = Gauge("mascot_active_workers", "Number of active Celery workers")


def setup_sentry() -> None:
    if not settings.SENTRY_DSN:
        return
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENV,
        traces_sample_rate=0.1,
        profiles_sample_rate=0.1,
    )
    logger.info("sentry_initialized", dsn_prefix=settings.SENTRY_DSN[:20])


def get_metrics_response() -> tuple[bytes, str]:
    """Return Prometheus metrics as bytes + content type."""
    return generate_latest(), CONTENT_TYPE_LATEST


@asynccontextmanager
async def track_query(status_label: str = "success") -> AsyncIterator[None]:
    """Context manager to track query metrics."""
    t0 = time.monotonic()
    try:
        yield
        queries_total.labels(status="success").inc()
    except Exception:
        queries_total.labels(status="error").inc()
        raise
    finally:
        query_latency.observe(time.monotonic() - t0)


# ── Health Checks ──────────────────────────────────────────────────────────────

class HealthStatus:
    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}

    @property
    def healthy(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "checks": self.checks,
        }


async def run_health_checks(db: AsyncSession) -> HealthStatus:
    status = HealthStatus()

    # DB check
    try:
        await db.execute(text("SELECT 1"))
        status.checks["database"] = True
    except Exception as e:
        logger.error("health_check_db_failed", error=str(e))
        status.checks["database"] = False

    # pgvector extension check
    try:
        await db.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
        status.checks["pgvector"] = True
    except Exception as e:
        logger.error("health_check_pgvector_failed", error=str(e))
        status.checks["pgvector"] = False

    return status