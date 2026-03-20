"""
Port: LLMExtractor

Defines how the domain requests structured extraction from raw text.
Adapters (OpenAI, Anthropic, local models, etc.) implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities import ResumeData


class LLMExtractor(ABC):
    """Extract structured ResumeData from raw resume text."""

    @abstractmethod
    def extract(self, raw_text: str) -> ResumeData:
        """
        Parse *raw_text* and return a populated ResumeData instance.
        Raise ExtractionError on unrecoverable parse failures.
        """


class ExtractionError(Exception):
    """Raised when the LLM response cannot be parsed into ResumeData."""