"""
Structuring layer — generates embeddings for chunks, stores them in pgvector.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator
from uuid import UUID

import openai
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import EmbeddingError, StructuringError
from app.core.logging import get_logger
from app.db.models import Chunk, Document, DocumentStatus

logger = get_logger(__name__)


class StructuringService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self._client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def embed_document(self, document: Document) -> int:
        """Generate and store embeddings for all chunks of a document."""
        logger.info("embedding_document", document_id=str(document.id))

        try:
            document.status = DocumentStatus.EMBEDDING
            await self.db.flush()

            # Fetch unembedded chunks
            result = await self.db.execute(
                select(Chunk)
                .where(Chunk.document_id == document.id)
                .where(Chunk.embedding.is_(None))
                .order_by(Chunk.chunk_index)
            )
            chunks = list(result.scalars().all())

            if not chunks:
                logger.warning("no_chunks_to_embed", document_id=str(document.id))
                document.status = DocumentStatus.COMPLETED
                await self.db.flush()
                return 0

            # Embed in batches
            total_embedded = 0
            async for batch in _batch_iter(chunks, settings.EMBEDDING_BATCH_SIZE):
                texts = [c.content for c in batch]
                embeddings = await self._embed_texts(texts)
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding
                total_embedded += len(batch)
                await self.db.flush()
                logger.debug("batch_embedded", count=len(batch), total=total_embedded)

            document.status = DocumentStatus.COMPLETED
            await self.db.flush()

            logger.info(
                "document_embedded",
                document_id=str(document.id),
                chunk_count=total_embedded,
            )
            return total_embedded

        except EmbeddingError:
            raise
        except Exception as e:
            raise StructuringError(f"Embedding failed for document {document.id}: {e}") from e

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embeddings API with retry."""
        for attempt in range(3):
            try:
                response = await self._client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=texts,
                    encoding_format="float",
                )
                return [item.embedding for item in response.data]
            except openai.RateLimitError:
                wait = 2 ** attempt
                logger.warning("rate_limit_hit", attempt=attempt, wait_seconds=wait)
                await asyncio.sleep(wait)
            except openai.OpenAIError as e:
                raise EmbeddingError(f"OpenAI embedding error: {e}") from e

        raise EmbeddingError("Embedding failed after 3 retries (rate limited)")

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = await self._embed_texts([query])
        return results[0]


async def _batch_iter(items: list, batch_size: int) -> AsyncIterator[list]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
        await asyncio.sleep(0)  # yield control