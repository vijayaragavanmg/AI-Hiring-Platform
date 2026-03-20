"""
Port: JobRepository

Defines how the application persists and queries processing job state.
Adapters (in-memory dict, Redis, SQL, etc.) implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from src.domain.entities import Job, JobStatus


class JobRepository(ABC):
    """Persist and retrieve job records."""

    @abstractmethod
    def create(self, job: Job) -> None:
        """Persist a new job. Raises ValueError if job_id already exists."""

    @abstractmethod
    def update(self, job_id: str, **fields: Any) -> None:
        """
        Patch the job with *job_id*.
        Always updates updated_at to utcnow().
        Raises KeyError if job_id is not found.
        """

    @abstractmethod
    def get(self, job_id: str) -> Optional[Job]:
        """Return the Job or None if not found."""

    @abstractmethod
    def list(self, status: Optional[JobStatus] = None) -> List[Job]:
        """Return all jobs, optionally filtered by status."""