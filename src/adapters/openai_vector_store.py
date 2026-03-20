"""
Adapter: ChromaVectorStore

Implements VectorStore using Chroma (via LangChain).
Translates between the port's Chunk / SearchResult types
and LangChain's Document type so no LangChain types
leak into the domain or other adapters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from src.ports.vector_store import Chunk, SearchResult, VectorStore


class OpenAIVectorStore(VectorStore):

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        persist_dir: str = "./resume_chroma_db",
    ) -> None:
        """Initialize OpenAI vector store.
        
        Args:
            embedding_model: OpenAI embedding model (default: text-embedding-3-small)
            persist_dir: Persistency directory for Chroma (default: ./resume_chroma_db)
        """
        self._persist_dir = persist_dir
        self._embedder = OpenAIEmbeddings(model=embedding_model)

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
        vs      = self._open()
        results = vs.similarity_search_with_relevance_scores(query, k=k, filter=filter)
        return [
            SearchResult(
                content=doc.page_content,
                metadata=dict(doc.metadata),
                score=score,
            )
            for doc, score in results
        ]

    def list_all_metadata(self) -> List[Dict[str, Any]]:
        collection = self._open().get()
        return [m for m in collection.get("metadatas", []) if m]

    def list_candidates(self) -> List[str]:
        """Return a sorted list of all unique candidate names in the store."""
        return sorted({
            m.get("candidate", "Unknown")
            for m in self.list_all_metadata()
        })

    # ── Internal ───────────────────────────────────────────────────────────

    def _open(self) -> Chroma:
        """Open the persisted store (read-only handle)."""
        return Chroma(
            persist_directory=self._persist_dir,
            embedding_function=self._embedder,
        )
