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
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.adapters import (
    BatchProcessor,
    ChromaHttpVectorStore,
    CohereReranker,
    InMemoryJobRepository,
    LangChainDocumentLoader,
    LangGraphPipeline,
    GeminiLLMExtractor,
    RetrievalPipeline,
    SectionChunker,
)
from src.drivers.config import (
    BATCH_MAX_WORKERS,
    CHROMA_COLLECTION,
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_SERVER_TOKEN,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_LLM_MODEL,
    GEMINI_LLM_TEMPERATURE,
    GEMINI_API_KEY,
    RERANK_FETCH_K,
    SUPPORTED_EXTENSIONS,
)


# ── Adapter singletons (built once per process) ────────────────────────────

@lru_cache(maxsize=1)
def get_vector_store() -> ChromaHttpVectorStore:
    return ChromaHttpVectorStore(
        embedder=        GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL),
        host=            CHROMA_HOST,
        port=            CHROMA_PORT,
        collection_name= CHROMA_COLLECTION,
        server_token=    CHROMA_SERVER_TOKEN,
    )


@lru_cache(maxsize=1)
def get_reranker() -> CohereReranker:
    """Reranker: Cohere API (requires COHERE_API_KEY).
    
    Swap to CrossEncoderReranker for local, free alternative:
        from src.adapters import CrossEncoderReranker
        return CrossEncoderReranker()
    """
    return CohereReranker(
        api_key=COHERE_API_KEY,
        model=COHERE_RERANK_MODEL,
    )


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
    """Resume processing pipeline with all ports injected."""
    return LangGraphPipeline(
        loader=LangChainDocumentLoader(
            supported_extensions=set(SUPPORTED_EXTENSIONS),
        ),
        extractor=GeminiLLMExtractor(
            api_key=GEMINI_API_KEY,
            model=GEMINI_LLM_MODEL,
            temperature=GEMINI_LLM_TEMPERATURE,
        ),
        chunker=SectionChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        ),
        vector_store=get_vector_store(),
        supported_extensions=set(SUPPORTED_EXTENSIONS),
    )


@lru_cache(maxsize=1)
def get_job_repo() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@lru_cache(maxsize=1)
def get_batch_processor() -> BatchProcessor:
    """Batch processor with thread pool configuration."""
    return BatchProcessor(
        pipeline=get_pipeline(),
        max_workers=BATCH_MAX_WORKERS,
        supported_extensions=set(SUPPORTED_EXTENSIONS),
    )


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
