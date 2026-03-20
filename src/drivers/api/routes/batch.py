"""Routes: batch upload and batch status polling."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from src.adapters import BatchProcessor
from src.drivers.config import BATCH_MAX_FILES, SUPPORTED_EXTENSIONS, UPLOAD_DIR
from src.drivers.api.dependencies import get_batch_processor, get_job_repo
from src.drivers.api.job_runner import run_batch_job
from src.drivers.api.schemas import BatchStatusResponse, BatchUploadResponse
from src.ports.job_repository import Job, JobRepository

router = APIRouter(tags=["Batch"])
log   = logging.getLogger(__name__)


@router.post(
    "/resumes/batch",
    response_model=BatchUploadResponse,
    status_code=202,
    summary="Upload multiple resumes for parallel async processing",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "files": {
                                "type":  "array",
                                "items": {"type": "string", "format": "binary"},
                            }
                        },
                        "required": ["files"],
                    }
                }
            }
        }
    },
)
async def upload_batch(
    background_tasks: BackgroundTasks,
    request:   Request,
    processor: BatchProcessor = Depends(get_batch_processor),
    job_repo:  JobRepository  = Depends(get_job_repo),
):
    """
    Accepts up to 50 resume files processed in parallel.

    In Swagger UI (/docs):
      1. Click **Try it out**
      2. Click **Add item** — each item shows **Choose File**
      3. Select one PDF, DOCX, or TXT file per item
      4. Click **Execute**

    Poll GET /batch/{batch_id} until status is "done".
    """
    form  = await request.form()
    items = form.getlist("files")

    # Temporary debug — prints directly to stdout regardless of LOG_LEVEL
    print(f"[BATCH DEBUG] content-type: {request.headers.get('content-type')}")
    print(f"[BATCH DEBUG] form keys: {list(form.keys())}")
    print(f"[BATCH DEBUG] files count: {len(items)}")
    for i, item in enumerate(items):
        print(f"[BATCH DEBUG] item[{i}] type={type(item).__name__!r} value={item!r}")

    if not items:
        raise HTTPException(
            status_code=422,
            detail="No files received. Field name must be 'files' in multipart/form-data.",
        )

    if len(items) > BATCH_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {BATCH_MAX_FILES} files per request.",
        )

    batch_id:    str       = str(uuid.uuid4())
    saved_paths: List[str] = []
    skipped:     List[str] = []

    for item in items:
        # Duck-type check — avoid isinstance which fails across import paths
        filename_raw = getattr(item, "filename", None)
        if not filename_raw:
            print(f"[LOOP] skipping — no filename attribute on {type(item)}")
            continue

        filename = Path(filename_raw).name  # strip any OS path prefix
        ext      = Path(filename).suffix.lower()

        print(f"[LOOP] filename={filename!r}  ext={ext!r}")

        if ext not in SUPPORTED_EXTENSIONS:
            print(f"[LOOP] skipping unsupported ext {ext!r}")
            skipped.append(filename)
            continue

        dest = UPLOAD_DIR / f"{batch_id}_{filename}"
        async with aiofiles.open(dest, "wb") as f:
            while chunk := await item.read(65_536):
                await f.write(chunk)

        saved_paths.append(str(dest))
        print(f"[LOOP] saved → {dest}")

    if not saved_paths:
        detail = (
            f"No supported files found. "
            f"Accepted extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
        if skipped:
            detail += f"  Skipped (wrong extension): {', '.join(skipped)}"
        raise HTTPException(status_code=400, detail=detail)

    job_repo.create(Job(
        job_id=batch_id,
        status="queued",
        meta={
            "type":        "batch",
            "files_count": len(saved_paths),
            "file_paths":  saved_paths,
            "total":       len(saved_paths),
            "succeeded":   0,
            "failed":      0,
        },
    ))

    background_tasks.add_task(
        run_batch_job, batch_id, saved_paths, processor, job_repo
    )

    return BatchUploadResponse(
        batch_id=batch_id,
        files_accepted=len(saved_paths),
        files_skipped=skipped,
        message=f"Batch queued ({len(saved_paths)} files). Poll GET /batch/{batch_id}.",
    )


@router.get(
    "/batch/{batch_id}",
    response_model=BatchStatusResponse,
    summary="Get the status and results of a batch job",
)
async def get_batch(
    batch_id: str,
    job_repo: JobRepository = Depends(get_job_repo),
):
    job = job_repo.get(batch_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")

    meta = job.meta or {}
    return BatchStatusResponse(
        batch_id=batch_id,
        status=job.status,
        total=meta.get("total", 0),
        succeeded=meta.get("succeeded", 0),
        failed=meta.get("failed", 0),
        duration_seconds=meta.get("duration_seconds"),
        results=meta.get("results"),
    )