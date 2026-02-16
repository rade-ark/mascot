"""
Retrieval layer — vector search, optional reranking, LLM answer synthesis.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

import openai
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.db.models import Chunk
from app.structuring.service import StructuringService

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a precise and factual assistant. Answer the user's question using ONLY
the provided context. If the context does not contain enough information to answer,
say so clearly. Do not hallucinate or invent information.

Context:
{context}
"""


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    score: float
    metadata: dict


@dataclass
class QueryResponse:
    answer: str
    sources: list[SearchResult]
    query: str
    latency_ms: int
    model: str


class RetrievalService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self._structuring = StructuringService(db)
        self._llm = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_ids: list[UUID] | None = None,
    ) -> list[SearchResult]:
        """Semantic vector search over embedded chunks."""
        top_k = top_k or settings.RETRIEVAL_TOP_K
        score_threshold = score_threshold or settings.RETRIEVAL_SCORE_THRESHOLD

        query_embedding = await self._structuring.embed_query(query)

        # pgvector cosine similarity search
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        sql = text("""
            SELECT
                c.id,
                c.document_id,
                c.content,
                c.metadata,
                d.name as document_name,
                1 - (c.embedding <=> :embedding ::vector) AS score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
              AND 1 - (c.embedding <=> :embedding ::vector) >= :threshold
              {doc_filter}
            ORDER BY score DESC
            LIMIT :top_k
        """.format(
            doc_filter="AND c.document_id = ANY(:doc_ids)" if document_ids else ""
        ))

        params: dict = {
            "embedding": embedding_str,
            "threshold": score_threshold,
            "top_k": top_k,
        }
        if document_ids:
            params["doc_ids"] = [str(d) for d in document_ids]

        result = await self.db.execute(sql, params)
        rows = result.fetchall()

        results = [
            SearchResult(
                chunk_id=str(row.id),
                document_id=str(row.document_id),
                document_name=row.document_name,
                content=row.content,
                score=float(row.score),
                metadata=row.metadata or {},
            )
            for row in rows
        ]

        logger.info(
            "search_completed",
            query=query[:80],
            result_count=len(results),
            top_score=results[0].score if results else 0,
        )
        return results

    async def query(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: list[UUID] | None = None,
        stream: bool = False,
    ) -> QueryResponse:
        """Full RAG: retrieve context + generate answer."""
        t0 = time.monotonic()

        # 1. Retrieve relevant chunks
        chunks = await self.search(
            query=question,
            top_k=top_k or settings.RETRIEVAL_TOP_K,
            document_ids=document_ids,
        )

        # 2. Rerank (MMR-style deduplication)
        reranked = _mmr_rerank(chunks, top_n=settings.RERANK_TOP_N)

        if not reranked:
            return QueryResponse(
                answer="I could not find relevant information to answer your question.",
                sources=[],
                query=question,
                latency_ms=int((time.monotonic() - t0) * 1000),
                model=settings.LLM_MODEL,
            )

        # 3. Build context
        context = "\n\n---\n\n".join(
            f"[{i + 1}] (from: {r.document_name})\n{r.content}"
            for i, r in enumerate(reranked)
        )

        # 4. Generate answer
        try:
            response = await self._llm.chat.completions.create(
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                    {"role": "user", "content": question},
                ],
            )
            answer = response.choices[0].message.content or ""
        except openai.OpenAIError as e:
            raise RetrievalError(f"LLM generation failed: {e}") from e

        latency_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "query_completed",
            question=question[:80],
            source_count=len(reranked),
            latency_ms=latency_ms,
        )

        return QueryResponse(
            answer=answer,
            sources=reranked,
            query=question,
            latency_ms=latency_ms,
            model=settings.LLM_MODEL,
        )


def _mmr_rerank(
    results: list[SearchResult],
    top_n: int,
    diversity_weight: float = 0.3,
) -> list[SearchResult]:
    """
    Maximal Marginal Relevance reranking to reduce redundancy.
    Balances relevance vs. diversity.
    """
    if not results:
        return []
    if len(results) <= top_n:
        return results

    selected: list[SearchResult] = [results[0]]
    candidates = results[1:]

    while len(selected) < top_n and candidates:
        best_idx = 0
        best_score = float("-inf")

        for i, candidate in enumerate(candidates):
            # Relevance component
            relevance = candidate.score

            # Diversity: penalise if too similar to already selected (by content overlap)
            max_sim = max(
                _jaccard_similarity(candidate.content, s.content) for s in selected
            )
            mmr_score = (1 - diversity_weight) * relevance - diversity_weight * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(candidates.pop(best_idx))

    return selected


def _jaccard_similarity(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)