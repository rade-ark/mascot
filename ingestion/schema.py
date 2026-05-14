#pydantic models

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid

class RawDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_path: str
    content_type: str           # "pdf", "docx", "html", "txt"
    raw_text: str
    metadata: dict = {}
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    text: str
    chunk_index: int
    token_count: int
    metadata: dict = {}         # inherits + chunk-level keys