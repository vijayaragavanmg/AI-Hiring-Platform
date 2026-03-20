"""
Adapter: RetrievalPipeline

Composes a VectorStore and a Reranker into a single two-stage retrieval
pipeline using the overfetch pattern:

  Stage 1 — Vector search
    Fetch fetch_k candidates (e.g. 20) from the vector store.
    High recall, moderate precision.

  Stage 2 — Rerank
    Score all fetch_k (query, passage) pairs with a cross-encoder or
    API reranker. Return the top_n most relevant results.
    High precision.

This is the correct place to compose these two concerns — neither the
VectorStore nor the Reranker knows about each other.

Usage:

    pipeline = RetrievalPipeline(
        store=    ChromaHttpVectorStore(...),
        reranker= CrossEncoderReranker(),
        fetch_k=  20,    # candidates fetched from vector store
    )

    results = pipeline.search(query="Python ML engineer", top_n=5)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.ports.reranker import RankedResult, Reranker
from src.ports.vector_store import SearchResult, VectorStore

log = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    Two-stage retrieval: vector search → rerank.

    Not a VectorStore subclass — it is a higher-level orchestrator
    that consumes both a VectorStore and a Reranker.
    """

    def __init__(
        self,
        store:    VectorStore,
        reranker: Reranker,
        fetch_k:  int = 20,
    ) -> None:
        self._store    = store
        self._reranker = reranker
        self._fetch_k  = fetch_k

    def search(
        self,
        query:   str,
        top_n:   int                      = 5,
        filter:  Optional[Dict[str, Any]] = None,
    ) -> List[RankedResult]:
        """
        Fetch *fetch_k* candidates from the vector store, rerank them,
        and return the *top_n* most relevant results.
        """
        # Stage 1: broad vector recall
        fetch_k = max(self._fetch_k, top_n)
        candidates: List[SearchResult] = self._store.search(
            query, k=fetch_k, filter=filter
        )

        log.debug(
            "RetrievalPipeline: fetched %d candidates for query='%s'",
            len(candidates), query[:60],
        )

        if not candidates:
            return []

        # Stage 2: cross-encoder rerank → precision
        return self._reranker.rerank(query, candidates, top_n=top_n)
