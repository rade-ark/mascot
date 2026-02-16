"""
Processing layer — extracts clean text from raw documents, chunks it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import ProcessingError
from app.core.logging import get_logger
from app.db.models import Chunk, Document, DocumentStatus
from app.ingestion.parsers import parse_document
from app.ingestion.storage import StorageBackend

logger = get_logger(__name__)


@dataclass
class TextChunk:
    content: str
    chunk_index: int
    token_count: int
    metadata: dict


class ProcessingService:
    def __init__(self, db: AsyncSession, storage: StorageBackend) -> None:
        self.db = db
        self.storage = storage

    async def process_document(self, document: Document) -> list[Chunk]:(
    ):
        """Extract text, chunk, and persist chunks for a document."""
        logger.info("processing_document", document_id=str(document.id))

        try:
            document.status = DocumentStatus.PROCESSING
            await self.db.flush()

            # Fetch raw content from storage
            storage_key = document.metadata_.get("storage_key")
            if not storage_key:
                raise ProcessingError(f"No storage key for document {document.id}")

            raw_bytes = await self.storage.get(storage_key)

            # Parse to clean text
            text = await parse_document(
                content=raw_bytes,
                mime_type=document.mime_type,
                filename=document.name,
            )

            if not text or not text.strip():
                raise ProcessingError(f"No extractable text in document {document.id}")

            document.raw_content = text

            # Chunk the text
            text_chunks = chunk_text(
                text=text,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )

            # Persist chunks (without embeddings yet)
            chunks = []
            for tc in text_chunks:
                chunk = Chunk(
                    document_id=document.id,
                    content=tc.content,
                    chunk_index=tc.chunk_index,
                    token_count=tc.token_count,
                    metadata_={
                        "document_name": document.name,
                        **tc.metadata,
                        **document.metadata_,
                    },
                )
                self.db.add(chunk)
                chunks.append(chunk)

            document.token_count = sum(tc.token_count for tc in text_chunks)
            document.status = DocumentStatus.STRUCTURING
            await self.db.flush()

            logger.info(
                "document_processed",
                document_id=str(document.id),
                chunk_count=len(chunks),
                total_tokens=document.token_count,
            )
            return chunks

        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Unexpected processing error: {e}") from e


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[TextChunk]:
    """
    Sentence-aware recursive chunking.
    Splits on paragraphs → sentences → words as a fallback.
    """
    text = _normalize_whitespace(text)
    if not text:
        return []

    chunks: list[TextChunk] = []
    splits = _split_sentences(text)
    current_tokens: list[str] = []
    current_size = 0

    for split in splits:
        split_tokens = _rough_token_count(split)

        if current_size + split_tokens > chunk_size and current_tokens:
            content = " ".join(current_tokens).strip()
            if content:
                chunks.append(
                    TextChunk(
                        content=content,
                        chunk_index=len(chunks),
                        token_count=current_size,
                        metadata={},
                    )
                )
            # Keep overlap
            overlap_tokens = current_tokens[-chunk_overlap:] if chunk_overlap else []
            current_tokens = overlap_tokens + [split]
            current_size = sum(_rough_token_count(t) for t in current_tokens)
        else:
            current_tokens.append(split)
            current_size += split_tokens

    # Final chunk
    if current_tokens:
        content = " ".join(current_tokens).strip()
        if content:
            chunks.append(
                TextChunk(
                    content=content,
                    chunk_index=len(chunks),
                    token_count=current_size,
                    metadata={},
                )
            )

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _rough_token_count(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars)."""
    return max(1, len(text) // 4)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()