"""
Background job runners.

These are the only functions that bridge FastAPI's BackgroundTasks
with the adapter layer.  They update the JobRepository directly,
keeping all side-effect logic out of route handlers.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

from src.adapters.batch_processor import BatchProcessor
from src.adapters.langgraph_pipeline import LangGraphPipeline
from src.domain.config import VECTORSTORE_PATH
from src.domain.entities import BatchSummary, ProcessingResult
from src.ports.job_repository import Job, JobRepository

log = logging.getLogger(__name__)


def run_single_job(
    job_id:   str,
    file_path: str,
    pipeline: LangGraphPipeline,
    job_repo: JobRepository,
) -> None:
    """Process one resume file and write the outcome to the job repository."""
    job_repo.update(job_id, status="processing")
    try:
        result: ProcessingResult = pipeline.process(file_path, str(VECTORSTORE_PATH))
        if result.success:
            rd = result.resume_data
            job_repo.update(
                job_id,
                status="done",
                result={
                    "candidate_name":   result.candidate_name,
                    "skills_count":     result.skills_count,
                    "jobs_count":       result.jobs_count,
                    "chunks_stored":    result.chunks_count,
                    "duration_seconds": result.duration_seconds,
                    "resume_data":      rd.model_dump() if rd else None,
                    "status_log":       result.status_log,
                },
            )
        else:
            job_repo.update(job_id, status="failed", error=result.error)

    except Exception as exc:
        log.exception("Single job %s crashed", job_id)
        job_repo.update(job_id, status="failed", error=str(exc))


def run_batch_job(
    batch_id:   str,
    file_paths: List[str],
    processor:  BatchProcessor,
    job_repo:   JobRepository,
) -> None:
    """Process a list of files and write the aggregate outcome to the job repository."""
    job_repo.update(batch_id, status="processing")
    try:
        summary: BatchSummary = processor.run(file_paths, show_progress=False)

        job_repo.update(
            batch_id,
            status="done",
            total=summary.total,
            succeeded=summary.succeeded,
            failed=summary.failed,
            duration_seconds=summary.total_duration,
            results=[
                {
                    "file":             Path(r.file_path).name,
                    "success":          r.success,
                    "candidate":        r.candidate_name,
                    "skills_count":     r.skills_count,
                    "jobs_count":       r.jobs_count,
                    "chunks_stored":    r.chunks_count,
                    "duration_seconds": r.duration_seconds,
                    "error":            r.error,
                }
                for r in summary.results
            ],
        )

    except Exception as exc:
        log.exception("Batch job %s crashed", batch_id)
        job_repo.update(batch_id, status="failed", error=str(exc))
