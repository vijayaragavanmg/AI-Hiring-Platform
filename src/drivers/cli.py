"""
CLI driver — batch resume processing.

Wires all adapter implementations and delegates to BatchProcessor.
Mirrors the same adapter stack used by the FastAPI server so CLI and
API behaviour are identical.

Requires a running ChromaDB server:
    docker compose up chromadb
    # or standalone: chroma run --host localhost --port 8001

Connection is configured via .env:
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, CHROMA_SERVER_TOKEN

Usage:
    # Process a whole folder
    python -m src.drivers.cli ./resumes/

    # Process specific files
    python -m src.drivers.cli cv1.pdf cv2.docx cv3.txt

    # Custom report output path
    python -m src.drivers.cli ./resumes/ --report my_report.json
"""

from __future__ import annotations

import argparse
import logging
import sys

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.adapters import (
    BatchProcessor,
    ChromaHttpVectorStore,
    LangChainDocumentLoader,
    LangGraphPipeline,
    GeminiLLMExtractor,
    SectionChunker,
)
from src.domain.config import (
    CHROMA_COLLECTION,
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_SERVER_TOKEN,
    GEMINI_EMBEDDING_MODEL,
    LOG_LEVEL,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _build_pipeline() -> LangGraphPipeline:
    """Construct the same adapter stack used by the API server."""
    vector_store = ChromaHttpVectorStore(
        embedder=        GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL),
        host=            CHROMA_HOST,
        port=            CHROMA_PORT,
        collection_name= CHROMA_COLLECTION,
        server_token=    CHROMA_SERVER_TOKEN,
    )

    return LangGraphPipeline(
        loader=       LangChainDocumentLoader(),
        extractor=    GeminiLLMExtractor(),
        chunker=      SectionChunker(),
        vector_store= vector_store,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="resume-cli",
        description="Batch-ingest resumes into ChromaDB via the LangGraph pipeline.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[],
        metavar="PATH",
        help="Files or directories to process",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        dest="extra_paths",
        metavar="PATH",
        help="Alternative to positional paths (e.g. --paths /dir1 /dir2)",
        default=[],
    )
    parser.add_argument(
        "--report",
        default="batch_report.json",
        metavar="FILE",
        help="Output path for the JSON report (default: batch_report.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel workers (default: BATCH_MAX_WORKERS from .env)",
    )
    args = parser.parse_args()

    # Merge positional paths and --paths flag; fall back to ./resumes
    all_paths = args.paths + args.extra_paths
    if not all_paths:
        all_paths = ["./resumes"]

    log.info(
        "ChromaDB → http://%s:%s  collection=%s",
        CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION,
    )

    pipeline  = _build_pipeline()
    processor = BatchProcessor(
        pipeline=pipeline,
        **({"max_workers": args.workers} if args.workers else {}),
    )

    summary = processor.run(all_paths)
    processor.export_report(summary, output_path=args.report)

    # Exit with non-zero code if any file failed — useful for CI pipelines
    if summary.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
