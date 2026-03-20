"""
Domain entities.

The heart of the application.  No framework, no library, no I/O.
Every other layer imports from here — nothing here imports from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Value objects ──────────────────────────────────────────────────────────

class JobEntry(BaseModel):
    """A single role in the candidate's work history."""
    company: str = Field(description="Company or organisation name")
    title: str = Field(description="Job title or role")
    start_date: str = Field(description="Start date, e.g. 'Jan 2020'")
    end_date: str = Field(description="End date or 'Present'")
    responsibilities: List[str] = Field(description="Key responsibilities / achievements")

class ResumeData(BaseModel):
    """Structured fields extracted from a resume."""
    name: str = Field(description="Full name of the candidate")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    summary: str = Field(description="Professional summary or objective")
    skills: List[str] = Field(description="Technical and soft skills")
    job_history: List[JobEntry] = Field(description="Work experience in reverse chronological order")
    education: List[str] = Field(default_factory=list, description="Degrees, institutions, years")

# ── Result objects ─────────────────────────────────────────────────────────

@dataclass
class ProcessingResult:
    """
    Outcome of processing a single resume file.
    Always returned — never raises.  Callers check .success.
    """
    file_path:        str
    success:          bool
    candidate_name:   Optional[str]        = None
    skills_count:     int                  = 0
    jobs_count:       int                  = 0
    chunks_count:     int                  = 0
    duration_seconds: float                = 0.0
    error:            Optional[str]        = None
    status_log:       List[str]            = field(default_factory=list)
    resume_data:      Optional[ResumeData] = None


@dataclass
class BatchSummary:
    """Aggregated outcome of a batch processing run."""
    total:          int                    = 0
    succeeded:      int                    = 0
    failed:         int                    = 0
    total_duration: float                  = 0.0
    results:        List[ProcessingResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return (self.succeeded / self.total * 100) if self.total else 0.0


# ── Document loading value objects ─────────────────────────────────────────────

@dataclass(frozen=True)
class RawDocument:
    """A single page of text read from a source file."""
    content: str
    page: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Vector store value objects ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Chunk:
    """A piece of text with metadata, ready to be embedded and stored."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk with its vector similarity score."""
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


# ── Reranking value objects ────────────────────────────────────────────────────

@dataclass(frozen=True)
class RankedResult:
    """A reranked search result with an explicit cross-encoder relevance score."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    vector_score: Optional[float] = None


# ── Job tracking entities ──────────────────────────────────────────────────────

JobStatus = Literal["queued", "processing", "done", "failed"]


@dataclass
class Job:
    """A single resume processing job tracked through its lifecycle."""
    job_id: str
    status: JobStatus
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    meta: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None