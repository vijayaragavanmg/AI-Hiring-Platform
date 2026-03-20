"""
FastAPI dependency providers.

All injectable objects are constructed once here via @lru_cache and
reused across requests via Depends().  Swapping any adapter or reranker
is a one-line change in this file only.

Retrieval flow:
  query → VectorStore.search(fetch_k=20) → Reranker.rerank(top_n=5)

Reranker options:
  - CrossEncoderReranker  → local, free, no API key  (default)
  - CohereReranker        → cloud API, best quality  (requires COHERE_API_KEY)

Vector store options:
  - ChromaHttpVectorStore → docker-compose / production (default)
  - ChromaVectorStore     → local dev without Docker
  - HuggingFaceVectorStore→ local dev with free embeddings

Embedder options (passed into ChromaHttpVectorStore):
  - OpenAIEmbeddings
  - GoogleGenerativeAIEmbeddings
  - HuggingFaceEmbeddings
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import aiofiles
from fastapi import HTTPException, UploadFile
from langchain_openai import OpenAIEmbeddings

from src.adapters import (
    BatchProcessor,
    ChromaHttpVectorStore,
    CrossEncoderReranker,
    InMemoryJobRepository,
    LangChainDocumentLoader,
    LangGraphPipeline,
    OpenAILLMExtractor,
    RetrievalPipeline,
    SectionChunker,
)
from src.domain.config import (
    CHROMA_COLLECTION,
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_SERVER_TOKEN,
    EMBEDDING_MODEL,
    RERANK_FETCH_K,
    SUPPORTED_EXTENSIONS,
)


# ── Adapter singletons (built once per process) ────────────────────────────

@lru_cache(maxsize=1)
def get_vector_store() -> ChromaHttpVectorStore:
    return ChromaHttpVectorStore(
        embedder=        OpenAIEmbeddings(model=EMBEDDING_MODEL),
        host=            CHROMA_HOST,
        port=            CHROMA_PORT,
        collection_name= CHROMA_COLLECTION,
        server_token=    CHROMA_SERVER_TOKEN,
    )


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    """
    Default: local cross-encoder (free, no API key needed).
    Swap to CohereReranker for higher quality at the cost of an API call:

        from src.adapters import CohereReranker
        return CohereReranker()
    """
    return CrossEncoderReranker()


@lru_cache(maxsize=1)
def get_retrieval_pipeline() -> RetrievalPipeline:
    """Two-stage retrieval: vector search → rerank."""
    return RetrievalPipeline(
        store=    get_vector_store(),
        reranker= get_reranker(),
        fetch_k=  RERANK_FETCH_K,
    )


@lru_cache(maxsize=1)
def get_pipeline() -> LangGraphPipeline:
    return LangGraphPipeline(
        loader=       LangChainDocumentLoader(),
        extractor=    OpenAILLMExtractor(),
        chunker=      SectionChunker(),
        vector_store= get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_job_repo() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@lru_cache(maxsize=1)
def get_batch_processor() -> BatchProcessor:
    return BatchProcessor(pipeline=get_pipeline())


# ── Request helpers ────────────────────────────────────────────────────────

def validate_extension(filename: str) -> str:
    """Raise HTTP 415 if the file extension is not supported."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )
    return ext


async def save_upload(upload: UploadFile, dest: Path) -> None:
    """Stream an uploaded file to disk in 64 KB chunks."""
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(65_536):
            await f.write(chunk)

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import aiofiles
from fastapi import HTTPException, UploadFile
from langchain_openai import OpenAIEmbeddings

from src.adapters import (
    BatchProcessor,
    ChromaHttpVectorStore,
    InMemoryJobRepository,
    LangChainDocumentLoader,
    LangGraphPipeline,
    OpenAILLMExtractor,
    SectionChunker,
)
from src.domain.config import (
    CHROMA_COLLECTION,
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_SERVER_TOKEN,
    EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS,
)


# ── Adapter singletons (built once per process) ────────────────────────────

@lru_cache(maxsize=1)
def get_vector_store() -> ChromaHttpVectorStore:
    return ChromaHttpVectorStore(
        embedder=        OpenAIEmbeddings(model=EMBEDDING_MODEL),
        host=            CHROMA_HOST,
        port=            CHROMA_PORT,
        collection_name= CHROMA_COLLECTION,
        server_token=    CHROMA_SERVER_TOKEN,
    )


@lru_cache(maxsize=1)
def get_pipeline() -> LangGraphPipeline:
    return LangGraphPipeline(
        loader=       LangChainDocumentLoader(),
        extractor=    OpenAILLMExtractor(),
        chunker=      SectionChunker(),
        vector_store= get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_job_repo() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@lru_cache(maxsize=1)
def get_batch_processor() -> BatchProcessor:
    return BatchProcessor(pipeline=get_pipeline())


# ── Request helpers ────────────────────────────────────────────────────────

def validate_extension(filename: str) -> str:
    """Raise HTTP 415 if the file extension is not supported."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )
    return ext


async def save_upload(upload: UploadFile, dest: Path) -> None:
    """Stream an uploaded file to disk in 64 KB chunks."""
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(65_536):
            await f.write(chunk)