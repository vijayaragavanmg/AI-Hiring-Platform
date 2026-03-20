"""Routes: single resume upload (async and sync)."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from src.adapters import LangGraphPipeline
from src.domain.config import UPLOAD_DIR, VECTORSTORE_PATH
from src.drivers.api.dependencies import (
    get_job_repo,
    get_pipeline,
    save_upload,
    validate_extension,
)
from src.drivers.api.job_runner import run_single_job
from src.drivers.api.schemas import ResumeResponse, UploadResponse
from src.ports.job_repository import Job, JobRepository

router = APIRouter(prefix="/resumes", tags=["Resumes"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=202,
    summary="Upload a single resume for async processing",
)
async def upload_resume(
    background_tasks: BackgroundTasks,
    file:     Annotated[UploadFile, File(description="PDF, DOCX, or TXT resume file")],
    pipeline: LangGraphPipeline = Depends(get_pipeline),
    job_repo: JobRepository     = Depends(get_job_repo),
):
    """
    Saves the file and starts a background LangGraph pipeline.
    Poll `GET /jobs/{job_id}` for the result.
    """
    validate_extension(file.filename)
    job_id = str(uuid.uuid4())
    dest   = UPLOAD_DIR / f"{job_id}_{file.filename}"

    await save_upload(file, dest)

    job_repo.create(Job(
        job_id=job_id,
        status="queued",
        meta={"file_name": file.filename, "file_path": str(dest)},
    ))

    background_tasks.add_task(
        run_single_job, job_id, str(dest), pipeline, job_repo
    )

    return UploadResponse(
        job_id=job_id,
        file_name=file.filename,
        message=f"Resume accepted. Poll GET /jobs/{job_id} for status.",
    )


@router.post(
    "/upload/sync",
    response_model=ResumeResponse,
    summary="Upload and process a resume synchronously",
)
async def upload_resume_sync(
    file:     Annotated[UploadFile, File(description="PDF, DOCX, or TXT resume file")],
    pipeline: LangGraphPipeline = Depends(get_pipeline),
):
    """
    Blocks until the pipeline finishes and returns the result immediately.
    Use the async endpoint for bulk ingestion.
    """
    validate_extension(file.filename)
    job_id = str(uuid.uuid4())
    dest   = UPLOAD_DIR / f"{job_id}_{file.filename}"

    await save_upload(file, dest)

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, pipeline.process, str(dest), str(VECTORSTORE_PATH)
    )

    if not result.success:
        raise HTTPException(status_code=422, detail=result.error)

    return ResumeResponse(
        job_id=job_id,
        candidate=result.resume_data,
        chunks_stored=result.chunks_count,
        duration_seconds=result.duration_seconds,
    )