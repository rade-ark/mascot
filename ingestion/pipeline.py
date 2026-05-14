#orchestrate parse->clean->chunk

from pathlib import Path
from ingestion.parsers.pdf_parser import PDFParser
from ingestion.parsers.docx_parser import DOCXParser
from ingestion.chunkers.semantic import SemanticChunker
from ingestion.schema import Chunk

PARSERS = [PDFParser(), DOCXParser()]  # add more as you build them

def ingest_file(file_path: str) -> list[Chunk]:
    path = file_path.lower()
    parser = next((p for p in PARSERS if p.can_parse(path)), None)
    if not parser:
        raise ValueError(f"No parser for {path}")

    doc = parser.parse(file_path)
    chunker = SemanticChunker()
    return chunker.chunk(doc)