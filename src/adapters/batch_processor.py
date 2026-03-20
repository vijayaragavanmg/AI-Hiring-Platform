"""
Adapter: BatchProcessor

Fans out LangGraphPipeline.process() across a thread pool,
collects results into a BatchSummary, and optionally writes
a JSON report to disk.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List

from src.domain.entities import BatchSummary, ProcessingResult
from src.adapters.langgraph_pipeline import LangGraphPipeline

log = logging.getLogger(__name__)


class BatchProcessor:
    """
    Processes a list of resume files with configurable parallelism.
    Thread-safe — each worker uses the same pipeline instance because
    LangGraphPipeline.process() builds a fresh graph invocation per call.
    """

    def __init__(
        self,
        pipeline: LangGraphPipeline,
        max_workers: int = 4,
        supported_extensions: set = None,
    ) -> None:
        """Initialize batch processor with pipeline and thread pool size.
        
        Args:
            pipeline: LangGraphPipeline instance
            max_workers: Thread pool size (default: 4)
            supported_extensions: Set of allowed file extensions (default: {.pdf, .docx, .txt})
        """
        self._pipeline = pipeline
        self._max_workers = max_workers
        self._supported_extensions = supported_extensions or {".pdf", ".docx", ".txt"}

    # ── File discovery ─────────────────────────────────────────────────────

    def collect_files(self, paths: List[str]) -> List[str]:
        """
        Accepts a mix of file paths and directory paths.
        Recursively discovers supported files inside directories.
        """
        collected: List[str] = []
        for p in paths:
            path = Path(p)
            if path.is_dir():
                for ext in self._supported_extensions:
                    collected.extend(str(f) for f in path.rglob(f"*{ext}"))
            elif path.is_file():
                if path.suffix.lower() in self._supported_extensions:
                    collected.append(str(path))
                else:
                    log.warning("Skipping unsupported file: %s", path)
            else:
                log.warning("Path not found, skipping: %s", path)
        return sorted(set(collected))

    # ── Batch run ──────────────────────────────────────────────────────────

    def run(self, paths: List[str], show_progress: bool = True) -> BatchSummary:
        """Process all discovered files in parallel and return a BatchSummary."""
        files   = self.collect_files(paths)
        summary = BatchSummary(total=len(files))

        if not files:
            log.warning("No supported resume files found.")
            return summary

        log.info("Starting batch: %d file(s) | workers=%d", len(files), self._max_workers)
        start = time.time()

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._pipeline.process, f): f
                for f in files
            }
            for i, future in enumerate(as_completed(futures), start=1):
                result: ProcessingResult = future.result()
                summary.results.append(result)

                if result.success:
                    summary.succeeded += 1
                    if show_progress:
                        log.info(
                            "[%d/%d] ✓  %s → %s  (%d skills, %d roles, %d chunks, %ss)",
                            i, len(files),
                            Path(result.file_path).name,
                            result.candidate_name,
                            result.skills_count,
                            result.jobs_count,
                            result.chunks_count,
                            result.duration_seconds,
                        )
                else:
                    summary.failed += 1
                    if show_progress:
                        log.error(
                            "[%d/%d] ✗  %s → %s",
                            i, len(files),
                            Path(result.file_path).name,
                            result.error,
                        )

        summary.total_duration = round(time.time() - start, 2)
        self._log_summary(summary)
        return summary

    # ── Reporting ──────────────────────────────────────────────────────────

    def export_report(
        self,
        summary:     BatchSummary,
        output_path: str = "batch_report.json",
    ) -> str:
        """Write a JSON report of the batch run to disk."""
        report = {
            "generated_at":           datetime.utcnow().isoformat(),
            "total":                  summary.total,
            "succeeded":              summary.succeeded,
            "failed":                 summary.failed,
            "success_rate_pct":       round(summary.success_rate, 1),
            "total_duration_seconds": summary.total_duration,
            "results": [
                {
                    "file":             Path(r.file_path).name,
                    "success":          r.success,
                    "candidate":        r.candidate_name,
                    "skills_count":     r.skills_count,
                    "jobs_count":       r.jobs_count,
                    "chunks_count":     r.chunks_count,
                    "duration_seconds": r.duration_seconds,
                    "error":            r.error,
                }
                for r in summary.results
            ],
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Report saved → %s", output_path)
        return output_path

    def _log_summary(self, summary: BatchSummary) -> None:
        sep = "─" * 54
        log.info(sep)
        log.info("  Batch complete in %ss", summary.total_duration)
        log.info("  Total    : %d", summary.total)
        log.info("  Succeeded: %d  (%.0f%%)", summary.succeeded, summary.success_rate)
        log.info("  Failed   : %d", summary.failed)
        for r in summary.results:
            if not r.success:
                log.error("    ✗ %s: %s", Path(r.file_path).name, r.error)
        log.info(sep)
