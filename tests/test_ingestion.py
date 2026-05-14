import os
import sys
from pathlib import Path

from ingestion.pipeline import ingest_file


def main() -> int:
    file_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("INGESTION_SAMPLE_FILE", "ainesh_resume.pdf")
    path = Path(file_path)

    if not path.exists():
        print(f"Skipping ingestion smoke test: sample file not found at {path}")
        return 0

    chunks = ingest_file(str(path))

    for chunk in chunks[:3]:
        print(f"--- Chunk {chunk.chunk_index} ---")
        print(f"Tokens : {chunk.token_count}")
        print(f"Text   : {chunk.text[:200]}")
        print(f"Meta   : {chunk.metadata}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())