"""
Ports — abstract interfaces the domain depends on.

Data types (RawDocument, Chunk, SearchResult, RankedResult, Job, JobStatus)
live in src.domain.entities — ports import them from there, not the other way.
Adapters in src/adapters/ provide the concrete implementations.
"""

from src.ports.chuncker import Chunker
from src.ports.document_loader import DocumentLoader
from src.ports.job_repository import JobRepository
from src.ports.llm_extractor import ExtractionError, LLMExtractor
from src.ports.reranker import Reranker
from src.ports.vector_store import VectorStore

__all__ = [
    "Chunker",
    "DocumentLoader",
    "ExtractionError",
    "JobRepository",
    "LLMExtractor",
    "Reranker",
    "VectorStore",
]