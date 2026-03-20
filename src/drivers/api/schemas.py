"""
API schemas — FastAPI request / response models.

These are the HTTP contract. They are intentionally separate from domain
entities so the API surface can evolve independently.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.domain.entities import ResumeData


# ── Resume upload ──────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id:    str
    file_name: str
    message:   str


class ResumeResponse(BaseModel):
    job_id:           str
    candidate:        ResumeData
    chunks_stored:    int
    duration_seconds: float


# ── Batch upload ───────────────────────────────────────────────────────────

class BatchUploadResponse(BaseModel):
    batch_id:       str
    files_accepted: int
    files_skipped:  List[str]
    message:        str


class BatchStatusResponse(BaseModel):
    batch_id:         str
    status:           str
    total:            int
    succeeded:        int
    failed:           int
    duration_seconds: Optional[float]      = None
    results:          Optional[List[Dict]] = None


# ── Job polling ────────────────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    job_id:     str
    status:     str                    # queued | processing | done | failed
    created_at: str
    updated_at: str
    file_name:  Optional[str]  = None
    result:     Optional[Dict] = None
    error:      Optional[str]  = None


# ── Search ─────────────────────────────────────────────────────────────────

class SearchResultSchema(BaseModel):
    content:   str
    section:   str
    candidate: str
    metadata:  Dict[str, Any]
    score:     Optional[float] = None


class SearchResponse(BaseModel):
    query:   str
    results: List[SearchResultSchema]
    total:   int
