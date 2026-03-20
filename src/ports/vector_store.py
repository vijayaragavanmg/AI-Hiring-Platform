"""
Port: VectorStore

Defines how the domain stores and retrieves embedded document chunks.
Adapters (Chroma, Pinecone, FAISS, etc.) implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.domain.entities import Chunk, SearchResult


class VectorStore(ABC):
    """Persist chunks and answer similarity queries against them."""

    @abstractmethod
    def store(self, chunks: List[Chunk]) -> None:
        """Embed and persist *chunks*. Idempotent on duplicate content."""

    @abstractmethod
    def search(
        self,
        query:  str,
        k:      int                    = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Return the *k* most similar chunks to *query*.
        Optionally restrict results to chunks whose metadata matches *filter*.
        """

    @abstractmethod
    def list_all_metadata(self) -> List[Dict[str, Any]]:
        """Return the metadata dict for every stored chunk (no embeddings)."""

    @abstractmethod
    def list_candidates(self) -> List[str]:
        """Return a sorted list of all unique candidate names in the store."""
