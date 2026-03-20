"""
Adapter: CohereReranker

Implements Reranker using the Cohere Rerank API.
Best quality — Cohere's rerank models are purpose-built for
retrieval and consistently top leaderboards.

Install:
    pip install cohere

.env:
    COHERE_API_KEY=your-cohere-api-key
    COHERE_RERANK_MODEL=rerank-english-v3.0
"""

from __future__ import annotations

import logging
from typing import List

import cohere

from src.ports.reranker import RankedResult, Reranker
from src.ports.vector_store import SearchResult

log = logging.getLogger(__name__)


class CohereReranker(Reranker):
    """
    Reranks candidates using the Cohere Rerank API.

    Sends all (query, passage) pairs to Cohere in a single batch call
    and returns results sorted by Cohere's relevance score.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
    ) -> None:
        """Initialize with API key and model.
        
        Args:
            api_key: Cohere API key (required)
            model: Cohere rerank model (default: rerank-english-v3.0)
        """
        self._client = cohere.Client(api_key=api_key)
        self._model = model
        log.info("CohereReranker initialised  model=%s", model)

    def rerank(
        self,
        query:   str,
        results: List[SearchResult],
        top_n:   int,
    ) -> List[RankedResult]:
        if not results:
            return []

        documents = [r.content for r in results]

        response = self._client.rerank(
            model=     self._model,
            query=     query,
            documents= documents,
            top_n=     min(top_n, len(results)),
        )

        ranked: List[RankedResult] = []
        for hit in response.results:
            original = results[hit.index]
            ranked.append(RankedResult(
                content=         original.content,
                metadata=        original.metadata,
                relevance_score= hit.relevance_score,
                vector_score=    original.score,
            ))

        log.debug(
            "Cohere reranked %d → %d  query='%s'",
            len(results), len(ranked), query[:60],
        )
        return ranked
