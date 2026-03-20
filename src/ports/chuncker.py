"""
Port: Chunker

Defines how the domain converts extracted ResumeData (or raw text)
into a flat list of Chunks ready for embedding.
Adapters (section-aware chunker, sliding-window, etc.) implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.entities import Chunk, RawDocument, ResumeData


class Chunker(ABC):
    """Convert resume content into embeddable Chunks."""

    @abstractmethod
    def chunk(
        self,
        resume_data:   Optional[ResumeData],
        raw_documents: List[RawDocument],
    ) -> List[Chunk]:
        """
        Produce Chunks from *resume_data* when available.
        Fall back to splitting *raw_documents* when extraction failed
        and *resume_data* is None.
        """