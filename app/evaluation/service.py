"""
Evaluation layer — measures RAG quality using faithfulness, relevance, precision, recall.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

import openai
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import EvaluationError
from app.core.logging import get_logger
from app.db.models import EvalResult, EvalRun
from app.retrieval.service import RetrievalService, SearchResult

logger = get_logger(__name__)


@dataclass
class EvalSample:
    question: str
    expected_answer: str | None = None
    document_ids: list[UUID] | None = None


@dataclass
class EvalMetrics:
    faithfulness: float      # Does the answer stick to the context?
    relevance: float         # Is the answer relevant to the question?
    context_precision: float # Are retrieved chunks actually useful?
    context_recall: float    # Did we retrieve all necessary context?

    @property
    def overall(self) -> float:
        return (self.faithfulness + self.relevance + self.context_precision + self.context_recall) / 4


class EvaluationService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self._retrieval = RetrievalService(db)
        self._llm = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def run_evaluation(
        self,
        name: str,
        samples: list[EvalSample],
        config: dict | None = None,
    ) -> EvalRun:
        """Run a full evaluation suite and persist results."""
        logger.info("eval_run_started", name=name, sample_count=len(samples))

        run = EvalRun(
            name=name,
            status="running",
            config=config or {},
        )
        self.db.add(run)
        await self.db.flush()

        results: list[EvalResult] = []
        all_metrics: list[EvalMetrics] = []

        # Process in batches
        for i in range(0, len(samples), settings.EVAL_BATCH_SIZE):
            batch = samples[i : i + settings.EVAL_BATCH_SIZE]
            batch_results = await asyncio.gather(
                *[self._eval_sample(run.id, sample) for sample in batch],
                return_exceptions=True,
            )
            for item in batch_results:
                if isinstance(item, Exception):
                    logger.error("eval_sample_failed", error=str(item))
                else:
                    result, metrics = item
                    results.append(result)
                    all_metrics.append(metrics)
                    self.db.add(result)

            await self.db.flush()

        # Compute summary
        if all_metrics:
            summary = {
                "faithfulness": _mean([m.faithfulness for m in all_metrics]),
                "relevance": _mean([m.relevance for m in all_metrics]),
                "context_precision": _mean([m.context_precision for m in all_metrics]),
                "context_recall": _mean([m.context_recall for m in all_metrics]),
                "overall": _mean([m.overall for m in all_metrics]),
                "sample_count": len(all_metrics),
            }
        else:
            summary = {"error": "No samples evaluated successfully"}

        run.status = "completed"
        run.summary_metrics = summary
        run.completed_at = datetime.now(timezone.utc)
        await self.db.flush()

        logger.info(
            "eval_run_completed",
            run_id=str(run.id),
            name=name,
            overall=summary.get("overall"),
        )
        return run

    async def _eval_sample(
        self, run_id: UUID, sample: EvalSample
    ) -> tuple[EvalResult, EvalMetrics]:
        t0 = time.monotonic()

        response = await self._retrieval.query(
            question=sample.question,
            document_ids=sample.document_ids,
        )

        metrics = await self._score(
            question=sample.question,
            answer=response.answer,
            expected=sample.expected_answer,
            chunks=response.sources,
        )

        latency_ms = int((time.monotonic() - t0) * 1000)

        result = EvalResult(
            run_id=run_id,
            question=sample.question,
            expected_answer=sample.expected_answer,
            generated_answer=response.answer,
            retrieved_chunks=[
                {"chunk_id": s.chunk_id, "score": s.score, "content_preview": s.content[:200]}
                for s in response.sources
            ],
            faithfulness_score=metrics.faithfulness,
            relevance_score=metrics.relevance,
            context_precision=metrics.context_precision,
            context_recall=metrics.context_recall,
            latency_ms=latency_ms,
        )

        return result, metrics

    async def _score(
        self,
        question: str,
        answer: str,
        expected: str | None,
        chunks: list[SearchResult],
    ) -> EvalMetrics:
        """LLM-as-judge scoring."""
        context = "\n---\n".join(c.content for c in chunks[:5])

        prompt = f"""Evaluate this RAG response. Return ONLY a JSON object with these float scores (0.0-1.0):
- faithfulness: Does the answer only use information from the context? (1.0 = fully grounded)
- relevance: Does the answer address the question? (1.0 = perfectly relevant)
- context_precision: Are the retrieved chunks useful for answering? (1.0 = all chunks relevant)
- context_recall: Does the context contain enough to answer? (1.0 = complete coverage)

Question: {question}
Context: {context[:2000]}
Generated Answer: {answer[:1000]}
Expected Answer: {expected or "N/A"}

Return only: {{"faithfulness": X, "relevance": X, "context_precision": X, "context_recall": X}}"""

        try:
            response = await self._llm.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            scores = json.loads(raw)
            return EvalMetrics(
                faithfulness=_clamp(scores.get("faithfulness", 0.5)),
                relevance=_clamp(scores.get("relevance", 0.5)),
                context_precision=_clamp(scores.get("context_precision", 0.5)),
                context_recall=_clamp(scores.get("context_recall", 0.5)),
            )
        except Exception as e:
            logger.warning("scoring_failed", error=str(e))
            return EvalMetrics(
                faithfulness=0.0, relevance=0.0, context_precision=0.0, context_recall=0.0
            )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))