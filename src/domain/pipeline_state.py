"""
LangGraph pipeline state.

Lives in domain because it describes *what* flows through the pipeline,
not *how* it is processed.  The only import from outside stdlib is the
domain entity ResumeData.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from src.domain.entities import ResumeData


class ResumeState(TypedDict):
    """Mutable state threaded through every LangGraph node."""

    # ── Input ──────────────────────────────────────────────
    file_path:        str

    # ── Populated by the loader adapter ───────────────────
    raw_text:         str
    raw_documents:    List[Any]          # List[langchain.schema.Document]

    # ── Populated by the extractor adapter ────────────────
    resume_data:      Optional[ResumeData]
    extraction_error: Optional[str]

    # ── Populated by the chunker adapter ──────────────────
    chunks:           List[Any]          # List[langchain.schema.Document]

    # ── Append-only log reduced by LangGraph ──────────────
    status_log:       Annotated[List[str], operator.add]

    # ── Threaded config ───────────────────────────────────
    vectorstore_path: str