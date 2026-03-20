"""
Port: Reranker

Defines how the domain requests relevance reranking of a candidate list.
Adapters (Cohere, cross-encoder, etc.) implement this.

Reranking is a two-stage retrieval pattern:
  1. Vector search fetches a broad candidate set (high recall, lower precision).
  2. Reranker scores every (query, passage) pair and re-orders by true relevance.

The reranker sees the full text of both the query and each passage,
giving it far more context than a vector similarity score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.domain.entities import RankedResult, SearchResult


class Reranker(ABC):
    """Rerank a list of candidate results against a query."""

    @abstractmethod
    def rerank(
        self,
        query:   str,
        results: List[SearchResult],
        top_n:   int,
    ) -> List[RankedResult]:
        """
        Score every (query, result.content) pair and return the *top_n*
        most relevant results ordered by descending relevance score.

        *results* is the raw output of VectorStore.search() — typically
        a larger candidate set (fetch_k) than the final *top_n* requested.
        """
