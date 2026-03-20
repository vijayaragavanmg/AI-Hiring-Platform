"""
Adapter: GeminiVectorStore

Implements VectorStore using Chroma for persistence and
Google Generative AI Embeddings for vectorisation.
Drop-in replacement for ChromaVectorStore.

Install:
    pip install langchain-google-genai

.env:
    GEMINI_API_KEY=your-key
    GEMINI_EMBEDDING_MODEL=models/text-embedding-004
    VECTORSTORE_PATH=./resume_chroma_db
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.ports.vector_store import Chunk, SearchResult, VectorStore


class GeminiVectorStore(VectorStore):

    def __init__(
        self,
        api_key: str,
        embedding_model: str = "gemini-embedding-2-preview",
        persist_dir: str = "./resume_chroma_db",
    ) -> None:
        """Initialize Gemini vector store.
        
        Args:
            api_key: Google Gemini API key (required)
            embedding_model: Gemini embedding model (default: gemini-embedding-2-preview)
            persist_dir: Persistency directory for Chroma (default: ./resume_chroma_db)
        """
        self._persist_dir = persist_dir
        self._embedder = GoogleGenerativeAIEmbeddings(
            api_key=api_key,
            model=embedding_model,
            task_type="retrieval_document",  # optimised for semantic search
        )

    # ── VectorStore interface ──────────────────────────────────────────────

    def store(self, chunks: List[Chunk]) -> None:
        lc_docs = [
            LCDocument(page_content=c.content, metadata=c.metadata)
            for c in chunks
        ]
        vs = Chroma.from_documents(
            documents=lc_docs,
            embedding=self._embedder,
            persist_directory=self._persist_dir,
        )
        vs.persist()

    def search(
        self,
        query:  str,
        k:      int                      = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        # Use query-optimised task type for retrieval
        query_embedder = GoogleGenerativeAIEmbeddings(
            model=self._embedder.model,
            google_api_key=GEMINI_API_KEY,
            task_type="retrieval_query",
        )
        vs      = self._open(query_embedder)
        results = vs.similarity_search_with_relevance_scores(query, k=k, filter=filter)
        return [
            SearchResult(
                content=  doc.page_content,
                metadata= dict(doc.metadata),
                score=    score,
            )
            for doc, score in results
        ]

    def list_all_metadata(self) -> List[Dict[str, Any]]:
        collection = self._open(self._embedder).get()
        return [m for m in collection.get("metadatas", []) if m]

    def list_candidates(self) -> List[str]:
        """Return sorted list of all unique candidate names in the store."""
        metadatas = self.list_all_metadata()
        return sorted({m.get("candidate", "Unknown") for m in metadatas if m})

    # ── Internal ───────────────────────────────────────────────────────────

    def _open(self, embedder: GoogleGenerativeAIEmbeddings) -> Chroma:
        return Chroma(
            persist_directory=self._persist_dir,
            embedding_function=embedder,
        )
