"""
Adapter: InMemoryJobRepository

Implements JobRepository using a plain dict.
Suitable for single-process deployments and tests.
Replace with RedisJobRepository or SQLJobRepository for production
without touching any other layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.ports.job_repository import Job, JobRepository, JobStatus


class InMemoryJobRepository(JobRepository):

    def __init__(self) -> None:
        self._store: Dict[str, Job] = {}

    def create(self, job: Job) -> None:
        if job.job_id in self._store:
            raise ValueError(f"Job '{job.job_id}' already exists.")
        self._store[job.job_id] = job

    def update(self, job_id: str, **fields: Any) -> None:
        job = self._store.get(job_id)
        if job is None:
            raise KeyError(f"Job '{job_id}' not found.")

        for key, value in fields.items():
            if key in ("job_id", "created_at"):
                continue                         # these are immutable
            if hasattr(job, key):
                setattr(job, key, value)         # direct Job field (status, result, error)
            else:
                job.meta[key] = value            # unknown fields go into meta

        job.updated_at = datetime.utcnow()

    def get(self, job_id: str) -> Optional[Job]:
        return self._store.get(job_id)

    def list(self, status: Optional[JobStatus] = None) -> List[Job]:
        jobs = list(self._store.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs