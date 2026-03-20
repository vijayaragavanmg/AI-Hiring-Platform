"""
Adapter: SectionChunker

Implements Chunker.
When ResumeData is available, produces one Chunk per logical section
(summary, skills, one per job role, education) with rich metadata.
Falls back to a sliding-window character splitter on raw text when
extraction failed.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.domain.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.domain.entities import ResumeData
from src.ports.chuncker import Chunker
from src.ports.document_loader import RawDocument
from src.ports.vector_store import Chunk


class SectionChunker(Chunker):

    def __init__(
        self,
        chunk_size:    int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(
        self,
        resume_data:   Optional[ResumeData],
        raw_documents: List[RawDocument],
    ) -> List[Chunk]:
        if resume_data is not None:
            return self._chunk_from_structured(resume_data)
        return self._chunk_from_raw(raw_documents)

    # ── Structured path ────────────────────────────────────────────────────

    def _chunk_from_structured(self, rd: ResumeData) -> List[Chunk]:
        chunks: List[Chunk] = []

        chunks.append(Chunk(
            content=f"Professional Summary: {rd.summary}",
            metadata={"candidate": rd.name, "section": "summary", "email": rd.email or ""},
        ))

        chunks.append(Chunk(
            content="Skills: " + ", ".join(rd.skills),
            metadata={"candidate": rd.name, "section": "skills"},
        ))

        for job in rd.job_history:
            chunks.append(Chunk(
                content=(
                    f"Role: {job.title} at {job.company} "
                    f"({job.start_date} – {job.end_date})\n"
                    + "\n".join(f"- {r}" for r in job.responsibilities)
                ),
                metadata={
                    "candidate":  rd.name,
                    "section":    "job_history",
                    "company":    job.company,
                    "title":      job.title,
                    "start_date": job.start_date,
                    "end_date":   job.end_date,
                },
            ))

        if rd.education:
            chunks.append(Chunk(
                content="Education: " + " | ".join(rd.education),
                metadata={"candidate": rd.name, "section": "education"},
            ))

        return chunks

    # ── Raw fallback path ──────────────────────────────────────────────────

    def _chunk_from_raw(self, raw_documents: List[RawDocument]) -> List[Chunk]:
        combined = "\n".join(doc.content for doc in raw_documents)
        texts    = self._splitter.split_text(combined)
        return [Chunk(content=t, metadata={"section": "raw"}) for t in texts]
