"""
Adapter: ChromaHttpVectorStore

Implements VectorStore using the ChromaDB HTTP client.
Connects to a remote (or Docker-hosted) ChromaDB server instead of
embedding a local Chroma instance — the correct approach for production
and containerised deployments.

The embedder is injected at construction time as a LangChain Embeddings
instance — this store has no opinion about which embedding model is used.
Pass any compatible embedder from dependencies.py:

    ChromaHttpVectorStore(
        embedder=OpenAIEmbeddings(model="text-embedding-3-small"),
        ...
    )

    ChromaHttpVectorStore(
        embedder=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        ...
    )

    ChromaHttpVectorStore(
        embedder=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5"),
        ...
    )

Use this adapter in docker-compose / Kubernetes.
Use ChromaVectorStore (local) for local dev without Docker.

Install:
    pip install chromadb

.env:
    CHROMA_HOST=localhost
    CHROMA_PORT=8001
    CHROMA_COLLECTION=resumes
    CHROMA_SERVER_TOKEN=         # leave blank to disable auth
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import Chroma

from src.domain.config import (
    CHROMA_COLLECTION,
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_SERVER_TOKEN,
)
from src.ports.vector_store import Chunk, SearchResult, VectorStore

log = logging.getLogger(__name__)


class ChromaHttpVectorStore(VectorStore):
    """
    VectorStore backed by a ChromaDB server reached over HTTP.

    Embeddings are computed locally by the injected *embedder* and
    then sent to the ChromaDB server for ANN indexing and persistence.
    The server owns all storage — no local chroma_db directory is needed.
    """

    def __init__(
        self,
        embedder:        Embeddings,
        host:            str = CHROMA_HOST,
        port:            int = CHROMA_PORT,
        collection_name: str = CHROMA_COLLECTION,
        server_token:    str = CHROMA_SERVER_TOKEN,
    ) -> None:
        self._embedder        = embedder
        self._collection_name = collection_name
        self._client          = self._build_client(host, port, server_token)

        log.info(
            "ChromaHttpVectorStore → http://%s:%s  collection=%s  embedder=%s",
            host, port, collection_name, type(embedder).__name__,
        )

    # ── VectorStore interface ──────────────────────────────────────────────

    def store(self, chunks: List[Chunk]) -> None:
        lc_docs = [
            LCDocument(page_content=c.content, metadata=c.metadata)
            for c in chunks
        ]
        Chroma.from_documents(
            documents=lc_docs,
            embedding=self._embedder,
            client=self._client,
            collection_name=self._collection_name,
        )
        log.debug("Stored %d chunks → collection '%s'", len(chunks), self._collection_name)

    def search(
        self,
        query:  str,
        k:      int                      = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        results = self._open().similarity_search_with_relevance_scores(
            query, k=k, filter=filter
        )
        return [
            SearchResult(
                content=  doc.page_content,
                metadata= dict(doc.metadata),
                score=    score,
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

    @staticmethod
    def _build_client(host: str, port: int, token: str) -> chromadb.HttpClient:
        """Build a ChromaDB HTTP client, with optional token auth."""
        settings = Settings(anonymized_telemetry=False)

        if token:
            settings = Settings(
                anonymized_telemetry=False,
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=token,
            )

        return chromadb.HttpClient(host=host, port=port, settings=settings)

    def _open(self) -> Chroma:
        """Return a LangChain Chroma handle bound to the HTTP client."""
        return Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self._embedder,
        )