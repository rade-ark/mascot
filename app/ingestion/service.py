"""
Ingestion layer — accepts raw documents, stores them, kicks off pipeline.
"""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import IngestionError, ValidationError
from app.core.logging import get_logger
from app.db.models import Document, DocumentStatus
from app.ingestion.storage import StorageBackend

logger = get_logger(__name__)

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/html",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/json",
    "text/csv",
}

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


class IngestionService:
    def __init__(self, db: AsyncSession, storage: StorageBackend) -> None:
        self.db = db
        self.storage = storage

    async def ingest_file(
        self,
        file_obj: BinaryIO,
        filename: str,
        metadata: dict | None = None,
    ) -> Document:
        """Ingest a file upload into the system."""
        content = file_obj.read()
        size = len(content)

        if size > MAX_FILE_SIZE_BYTES:
            raise ValidationError(
                f"File too large: {size} bytes (max {MAX_FILE_SIZE_BYTES})"
            )

        mime_type = _detect_mime(filename, content)
        if mime_type not in ALLOWED_MIME_TYPES:
            raise ValidationError(f"Unsupported file type: {mime_type}")

        content_hash = hashlib.sha256(content).hexdigest()

        # Deduplicate by content hash
        existing = await self._find_by_hash(content_hash)
        if existing:
            logger.info("duplicate_document_skipped", hash=content_hash, id=str(existing.id))
            return existing

        # Store raw file
        storage_key = await self.storage.put(
            key=f"raw/{content_hash}/{filename}",
            data=content,
            content_type=mime_type,
        )

        doc = Document(
            name=filename,
            mime_type=mime_type,
            status=DocumentStatus.PENDING,
            metadata_={
                "content_hash": content_hash,
                "storage_key": storage_key,
                "file_size_bytes": size,
                **(metadata or {}),
            },
        )
        self.db.add(doc)
        await self.db.flush()

        logger.info(
            "document_ingested",
            document_id=str(doc.id),
            filename=filename,
            mime_type=mime_type,
            size_bytes=size,
        )
        return doc

    async def ingest_url(self, url: str, metadata: dict | None = None) -> Document:
        """Ingest a document from a URL."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise IngestionError(f"Failed to fetch URL {url}: {e}") from e

        content = response.content
        content_type = response.headers.get("content-type", "text/html").split(";")[0].strip()
        filename = Path(url.split("?")[0]).name or "document"

        import io
        return await self.ingest_file(
            file_obj=io.BytesIO(content),
            filename=filename,
            metadata={"source_url": url, "content_type": content_type, **(metadata or {})},
        )

    async def get_document(self, document_id: UUID) -> Document | None:
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        status: DocumentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Document]:
        query = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        if status:
            query = query.where(Document.status == status)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def _find_by_hash(self, content_hash: str) -> Document | None:
        result = await self.db.execute(
            select(Document).where(
                Document.metadata_["content_hash"].astext == content_hash
            )
        )
        return result.scalar_one_or_none()

    async def update_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> None:
        doc = await self.get_document(document_id)
        if doc:
            doc.status = status
            if error_message:
                doc.error_message = error_message
            await self.db.flush()


def _detect_mime(filename: str, content: bytes) -> str:
    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        return guessed
    # Sniff first bytes
    if content[:4] == b"%PDF":
        return "application/pdf"
    if content[:2] in (b"PK",):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "text/plain"