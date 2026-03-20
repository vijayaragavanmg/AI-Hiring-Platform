"""
Port: DocumentLoader

Defines how the domain requests raw text from a file.
Adapters (LangChain loaders, plain open(), etc.) implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.domain.entities import RawDocument


class DocumentLoader(ABC):
    """Load a file from disk and return its pages as RawDocuments."""

    @abstractmethod
    def load(self, file_path: str) -> List[RawDocument]:
        """
        Read the file at *file_path* and return one RawDocument per page.
        Raise ValueError for unsupported extensions.
        Raise FileNotFoundError if the path does not exist.
        """