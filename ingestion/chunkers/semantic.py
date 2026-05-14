#primary chunker

import tiktoken
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from ingestion.schema import Chunk, RawDocument


class SemanticChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=chunk_size,
        )

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        texts = self.splitter.split_text(doc.raw_text)
        enc = tiktoken.get_encoding("cl100k_base")

        return [
            Chunk(
                document_id=doc.id,
                text=t,
                chunk_index=i,
                token_count=len(enc.encode(t)),
                metadata={**doc.metadata, "source_path": doc.source_path}
            )
            for i, t in enumerate(texts)
        ]