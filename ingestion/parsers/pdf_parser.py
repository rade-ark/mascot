import fitz  # pymupdf
from pathlib import Path
from .base import BaseParser
from ingestion.schema import RawDocument

class PDFParser(BaseParser):
    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: str) -> RawDocument:
        with fitz.open(file_path) as doc:
            pages = [page.get_text() for page in doc]
            text = "\n\n".join(pages)

        # If text is sparse, the PDF is likely scanned and needs OCR
        if len(text.strip()) < 100:
            from .ocr_parser import OCRParser
            return OCRParser().parse(file_path)

        return RawDocument(
            source_path=file_path,
            content_type="pdf",
            raw_text=text,
            metadata={"page_count": len(pages)}
        )