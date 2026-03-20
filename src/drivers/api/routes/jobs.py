"""Routes: job status polling."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.drivers.api.dependencies import get_job_repo
from src.drivers.api.schemas import JobStatusResponse
from src.ports.job_repository import JobRepository

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get the status and result of a single processing job",
)
async def get_job(
    job_id:   str,
    job_repo: JobRepository = Depends(get_job_repo),
):
    job = job_repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JobStatusResponse(
        job_id=    job.job_id,
        status=    job.status,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        file_name= job.meta.get("file_name"),
        result=    job.result,
        error=     job.error,
    )


@router.get(
    "",
    summary="List all jobs, optionally filtered by status",
)
async def list_jobs(
    status:   Optional[str] = Query(None, description="queued | processing | done | failed"),
    job_repo: JobRepository  = Depends(get_job_repo),
):
    jobs = job_repo.list(status=status)
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id":     j.job_id,
                "status":     j.status,
                "created_at": j.created_at.isoformat(),
                "updated_at": j.updated_at.isoformat(),
                "file_name":  j.meta.get("file_name"),
                "error":      j.error,
            }
            for j in jobs
        ],
    }
